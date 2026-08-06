//! Session-owned restore projection helpers shared by TUI and line CLI.
//!
//! Phase L extracts the common "project restore snapshot including child
//! sessions" path. Agent restore and frontend timeline mapping remain outside.

use std::path::Path;
use std::sync::{Arc, Mutex};

use anyhow::Result;
use async_openai::config::Config;

use crate::agent::Agent;
use crate::config::ModelRoute;
use crate::permission::PermissionMode;
use crate::session::context_scope::{apply_prepared_context_scope, prepare_context_scope};
use crate::session::lifecycle::{cleanup_replaced_empty_session, replace_live_transcript};
use crate::transcript::transcript_projection::{
    RuntimeRestoreSnapshot, SessionContextCursor, project_runtime_restore_snapshot,
};
use crate::transcript::{TranscriptRecord, TranscriptRecorder, list_child_sessions_for_parent};

/// Default cursor for resume: active branch tip (no explicit leaf cut).
pub fn default_resume_cursor() -> SessionContextCursor {
    SessionContextCursor {
        branch_id: None,
        leaf_sequence: None,
    }
}

/// Project a runtime restore snapshot, resolving child sessions under
/// `sessions_dir` from the first-pass projection records.
pub fn project_runtime_restore_snapshot_with_children(
    session_id: impl Into<String>,
    records: Vec<TranscriptRecord>,
    cursor: SessionContextCursor,
    sessions_dir: impl AsRef<Path>,
) -> Result<RuntimeRestoreSnapshot> {
    let session_id = session_id.into();
    let resolved =
        project_runtime_restore_snapshot(session_id.clone(), records.clone(), cursor.clone(), &[])?;
    let children = list_child_sessions_for_parent(sessions_dir.as_ref(), &resolved.records);
    if children.is_empty() {
        // Common case: skip a second full projection when there is nothing to attach.
        return Ok(resolved);
    }
    project_runtime_restore_snapshot(session_id, records, cursor, &children)
}

/// Session-owned resume package: records + restore snapshot + open recorder
/// with legacy branch adopted. Agent restore remains the caller's job.
pub struct PreparedResume {
    pub session_id: String,
    #[allow(dead_code)]
    pub records: Vec<TranscriptRecord>,
    pub snapshot: RuntimeRestoreSnapshot,
    pub recorder: crate::transcript::TranscriptRecorder,
}

#[derive(Debug)]
pub struct ResumeInstallError {
    error: anyhow::Error,
    pub fast_mode_auto_disabled: bool,
}

impl ResumeInstallError {
    fn new(error: anyhow::Error, fast_mode_auto_disabled: bool) -> Self {
        Self {
            error,
            fast_mode_auto_disabled,
        }
    }
}

impl std::fmt::Display for ResumeInstallError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Display::fmt(&self.error, formatter)
    }
}

impl std::error::Error for ResumeInstallError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(self.error.as_ref())
    }
}

/// Load records, project restore snapshot (with children), open the transcript,
/// and adopt the restored branch on the recorder.
pub fn prepare_resume_package(
    sessions_dir: impl AsRef<Path>,
    session_id: impl Into<String>,
) -> Result<PreparedResume> {
    use crate::session::lifecycle::{
        load_session_records_with_fingerprint, open_resume_transcript_with_records_at_fingerprint,
    };

    let sessions_dir = sessions_dir.as_ref();
    let session_id = session_id.into();
    let (records, fingerprint) = load_session_records_with_fingerprint(sessions_dir, &session_id)?;
    let snapshot = project_runtime_restore_snapshot_with_children(
        session_id.clone(),
        records.clone(),
        default_resume_cursor(),
        sessions_dir,
    )?;
    let mut recorder = open_resume_transcript_with_records_at_fingerprint(
        sessions_dir,
        &session_id,
        &records,
        &fingerprint,
    )?;
    recorder.adopt_legacy_linear_branch(&snapshot.branch_id)?;
    Ok(PreparedResume {
        session_id,
        records,
        snapshot,
        recorder,
    })
}

/// Apply a prepared resume package onto the agent: restore runtime snapshot,
/// adopt the latest model when present, and sync context scope from the recorder.
///
/// Does **not** swap the live transcript recorder — callers still own that
/// under their locking / cleanup rules.
#[cfg(test)]
pub(crate) fn apply_prepared_resume_to_agent<C: Config>(
    agent: &mut Agent<C>,
    prepared: &PreparedResume,
) -> Result<()> {
    agent.restore_new_session_runtime_snapshot(
        prepared.snapshot.protocol_frames.clone(),
        prepared.snapshot.snapshot.clone(),
        prepared.snapshot.max_turn_id,
    )?;
    if let Some(model) = prepared.snapshot.latest_model.as_deref() {
        agent.set_model(model);
    }
    apply_restored_permission_mode(agent, prepared.snapshot.latest_permission_mode.as_deref());
    let prepared_scope = prepare_context_scope(&prepared.recorder)?;
    apply_prepared_context_scope(agent, prepared_scope);
    Ok(())
}

pub(crate) enum PreparedRestoredRoute<C: Config> {
    Prepared {
        target_model: String,
        route: crate::agent::PreparedPrimaryRoute<C>,
    },
    ModelOnly(String),
}

impl<C: Config> PreparedRestoredRoute<C> {
    pub(crate) fn target_model(&self) -> &str {
        match self {
            Self::Prepared { target_model, .. } => target_model,
            Self::ModelOnly(model) => model,
        }
    }

    pub(crate) fn apply(self, agent: &mut Agent<C>) {
        match self {
            Self::Prepared { route, .. } => agent.apply_prepared_route(route),
            Self::ModelOnly(model) => agent.set_model(model),
        }
    }
}

pub(crate) fn prepare_restored_model_route<C: Config>(
    agent: &Agent<C>,
    latest_model: Option<&str>,
) -> Result<Option<PreparedRestoredRoute<C>>>
where
    C: Clone,
{
    let Some(model) = latest_model else {
        return Ok(None);
    };
    let Some(active_route) = agent.primary_route().cloned() else {
        return Ok(Some(PreparedRestoredRoute::ModelOnly(model.to_string())));
    };

    if let Some((provider, model_id)) = model.split_once('/')
        && provider == active_route.provider
    {
        let candidate = ModelRoute::new(provider, model_id);
        if let Ok(route) = agent.prepare_primary_route(candidate) {
            return Ok(Some(PreparedRestoredRoute::Prepared {
                target_model: model_id.to_string(),
                route,
            }));
        }
    }

    let legacy_candidate = ModelRoute::new(active_route.provider.clone(), model);
    if let Ok(route) = agent.prepare_primary_route(legacy_candidate) {
        return Ok(Some(PreparedRestoredRoute::Prepared {
            target_model: model.to_string(),
            route,
        }));
    }

    if let Some((provider, model_id)) = model.split_once('/')
        && provider != active_route.provider
    {
        let candidate = ModelRoute::new(provider, model_id);
        if let Ok(route) = agent.prepare_primary_route(candidate) {
            return Ok(Some(PreparedRestoredRoute::Prepared {
                target_model: model_id.to_string(),
                route,
            }));
        }
    }

    Err(anyhow::anyhow!(
        "recorded model '{model}' is not configured for provider '{}' or as a provider-qualified route",
        active_route.provider
    ))
}

pub(crate) fn apply_prepared_restored_route<C: Config>(
    agent: &mut Agent<C>,
    route: Option<PreparedRestoredRoute<C>>,
) {
    if let Some(route) = route {
        route.apply(agent);
    }
}

pub(crate) fn apply_restored_permission_mode<C: Config>(
    agent: &mut Agent<C>,
    mode: Option<&str>,
) {
    if let Some(mode) = mode.and_then(PermissionMode::parse) {
        agent.set_permission_mode(mode);
    }
}

#[cfg(test)]
pub(crate) fn apply_restored_model_route(
    agent: &mut Agent<async_openai::config::OpenAIConfig>,
    latest_model: Option<&str>,
) -> Result<()> {
    let route = prepare_restored_model_route(agent, latest_model)?;
    apply_prepared_restored_route(agent, route);
    Ok(())
}

/// Apply prepared resume state, swap the live recorder, then clean a prior empty file.
///
/// Build resume event payloads from `prepared` before this call (recorder is moved).
#[cfg(test)]
pub fn install_prepared_resume_for_agent<C: Config>(
    agent: &mut Agent<C>,
    live: &Arc<Mutex<TranscriptRecorder>>,
    prepared: PreparedResume,
) -> std::result::Result<bool, ResumeInstallError> {
    let model = prepared
        .snapshot
        .latest_model
        .as_deref()
        .unwrap_or(agent.model())
        .to_string();
    let fast_mode_auto_disabled = agent
        .auto_disable_fast_mode_for_model(&model)
        .map_err(|error| ResumeInstallError::new(error, false))?;
    apply_prepared_resume_to_agent(agent, &prepared)
        .map_err(|error| ResumeInstallError::new(error, fast_mode_auto_disabled))?;
    let new_path = prepared.recorder.path().to_path_buf();
    let old_path = replace_live_transcript(live, prepared.recorder)
        .map_err(|error| ResumeInstallError::new(error, fast_mode_auto_disabled))?;
    let _ = cleanup_replaced_empty_session(old_path, &new_path);
    Ok(fast_mode_auto_disabled)
}

/// Apply prepared resume state to a provider-routed primary agent, swap the
/// live recorder, then clean a prior empty file.
pub fn install_prepared_routed_resume_for_agent(
    agent: &mut Agent<async_openai::config::OpenAIConfig>,
    live: &Arc<Mutex<TranscriptRecorder>>,
    prepared: PreparedResume,
) -> std::result::Result<(bool, crate::session::event::TokenUsageEvent), ResumeInstallError> {
    let route = prepare_restored_model_route(agent, prepared.snapshot.latest_model.as_deref())
        .map_err(|error| ResumeInstallError::new(error, false))?;
    let target_model = route
        .as_ref()
        .map_or_else(|| agent.model(), PreparedRestoredRoute::target_model);
    let token_usage =
        restored_session_token_usage(agent, target_model, &prepared.snapshot.snapshot)
            .map_err(|error| ResumeInstallError::new(error, false))?;
    let prepared_scope = prepare_context_scope(&prepared.recorder)
        .map_err(|error| ResumeInstallError::new(error, false))?;
    let (protocol_frames, runtime_snapshot) = agent
        .validate_runtime_snapshot_restore(
            prepared.snapshot.protocol_frames.clone(),
            prepared.snapshot.snapshot.clone(),
        )
        .map_err(|error| ResumeInstallError::new(error, false))?;
    let fast_mode_auto_disabled = agent
        .auto_disable_fast_mode_for_model(target_model)
        .map_err(|error| ResumeInstallError::new(error, false))?;
    let new_path = prepared.recorder.path().to_path_buf();
    let old_path = replace_live_transcript(live, prepared.recorder)
        .map_err(|error| ResumeInstallError::new(error, fast_mode_auto_disabled))?;
    apply_prepared_restored_route(agent, route);
    apply_restored_permission_mode(agent, prepared.snapshot.latest_permission_mode.as_deref());
    agent.install_validated_runtime_snapshot(protocol_frames, runtime_snapshot);
    agent.restore_turn_sequence(prepared.snapshot.max_turn_id);
    apply_prepared_context_scope(agent, prepared_scope);
    let _ = cleanup_replaced_empty_session(old_path, &new_path);
    Ok((fast_mode_auto_disabled, token_usage))
}

/// Timeline-facing conversation messages restored from protocol frames.
pub fn restored_messages_from_protocol_frames(
    protocol_frames: &[crate::protocol_frames::ProtocolFrame],
) -> Vec<crate::agent::ConversationMessage> {
    crate::protocol_frames::history_items_from_frames(protocol_frames)
        .into_iter()
        .filter_map(|item| match item {
            crate::request_builder::HistoryItem::ContextSummary { text } => {
                Some(crate::agent::ConversationMessage {
                    role: crate::agent::ConversationRole::Summary,
                    content: text,
                })
            }
            crate::request_builder::HistoryItem::UserMessage { content } => {
                Some(crate::agent::ConversationMessage {
                    role: crate::agent::ConversationRole::User,
                    content: content.display_text(),
                })
            }
            crate::request_builder::HistoryItem::InternalContinuation { text } => {
                Some(crate::agent::ConversationMessage {
                    role: crate::agent::ConversationRole::User,
                    content: text,
                })
            }
            crate::request_builder::HistoryItem::AssistantText { text } => {
                Some(crate::agent::ConversationMessage {
                    role: crate::agent::ConversationRole::Assistant,
                    content: text,
                })
            }
            crate::request_builder::HistoryItem::AssistantToolCalls { text, .. } => {
                text.map(|content| crate::agent::ConversationMessage {
                    role: crate::agent::ConversationRole::Assistant,
                    content,
                })
            }
            _ => None,
        })
        .collect()
}

/// Fresh token estimate for a restored session request.
///
/// Response and cache accounting are not persisted in transcripts, so they must
/// not cross a session boundary (always zeroed here).
pub fn restored_session_token_usage<C: Config>(
    agent: &Agent<C>,
    model_id: &str,
    runtime_snapshot: &crate::runtime_context::RuntimeSnapshot,
) -> Result<crate::session::event::TokenUsageEvent> {
    let usage = agent.candidate_session_token_usage(model_id, runtime_snapshot)?;
    Ok(crate::session::event::TokenUsageEvent::with_breakdown(
        usage.used_tokens,
        usage.context_window_tokens,
        usage.input_tokens,
        0,
        0,
    ))
}

#[cfg(test)]
fn session_resumed_event(
    prepared: &PreparedResume,
    runtime_context: crate::runtime_context::RuntimeActiveContext,
    token_usage: Option<crate::session::event::TokenUsageEvent>,
) -> crate::session::runner::SessionTransportEvent {
    let snapshot = &prepared.snapshot;
    crate::session::runner::SessionTransportEvent::SessionResumed {
        session_id: prepared.session_id.clone(),
        branch_id: snapshot.branch_id.clone(),
        messages: restored_messages_from_protocol_frames(&snapshot.protocol_frames),
        records: snapshot.records.clone(),
        evidence_count: snapshot.snapshot.evidence.len(),
        model_id: snapshot.latest_model.clone(),
        token_usage,
        runtime_context,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use async_openai::{Client, config::OpenAIConfig};
    use std::time::{SystemTime, UNIX_EPOCH};

    struct RecordingRouteFactory {
        accepted_routes: Vec<ModelRoute>,
        applied_route: Arc<Mutex<Option<ModelRoute>>>,
    }

    impl crate::agent::PrimaryRouteFactory<OpenAIConfig> for RecordingRouteFactory {
        fn prepare_route(
            &self,
            route: ModelRoute,
        ) -> Result<crate::agent::PreparedPrimaryRoute<OpenAIConfig>> {
            if !self.accepted_routes.contains(&route) {
                anyhow::bail!("route is not configured: {}", route.display_name());
            }
            *self.applied_route.lock().expect("capture route") = Some(route.clone());
            let client = Client::with_config(
                OpenAIConfig::new()
                    .with_api_base("http://127.0.0.1:9/v1")
                    .with_api_key("test-key"),
            );
            Ok(crate::agent::PreparedPrimaryRoute::new(
                client,
                route,
                crate::config::ApiProtocol::Responses,
                std::collections::HashMap::new(),
                std::collections::HashMap::new(),
                crate::config::RetryConfig::default(),
            ))
        }
    }

    fn temp_dir() -> std::path::PathBuf {
        std::env::temp_dir().join(format!(
            "letcode-session-restore-fast-mode-{}",
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .expect("time ok")
                .as_nanos()
        ))
    }

    fn test_agent() -> Agent<OpenAIConfig> {
        let config = OpenAIConfig::new()
            .with_api_base("http://127.0.0.1:9/v1")
            .with_api_key("test-key");
        Agent::new(Client::with_config(config), "gpt-5.5", 1, 1)
    }

    struct SelectiveRouteFactory {
        accepted_routes: Vec<ModelRoute>,
        attempted_routes: Arc<Mutex<Vec<ModelRoute>>>,
    }

    impl crate::agent::PrimaryRouteFactory<OpenAIConfig> for SelectiveRouteFactory {
        fn prepare_route(
            &self,
            route: ModelRoute,
        ) -> Result<crate::agent::PreparedPrimaryRoute<OpenAIConfig>> {
            self.attempted_routes
                .lock()
                .expect("capture attempted route")
                .push(route.clone());
            if !self.accepted_routes.contains(&route) {
                anyhow::bail!("route is not configured: {}", route.display_name());
            }
            Ok(crate::agent::PreparedPrimaryRoute::new(
                Client::with_config(
                    OpenAIConfig::new()
                        .with_api_base("http://127.0.0.1:9/v1")
                        .with_api_key("test-key"),
                ),
                route,
                crate::config::ApiProtocol::Responses,
                std::collections::HashMap::new(),
                std::collections::HashMap::new(),
                crate::config::RetryConfig::default(),
            ))
        }
    }

    #[test]
    fn restored_model_route_resolves_active_provider_qualified_and_legacy_slash_models() {
        let mut agent = test_agent();
        agent.set_primary_route(ModelRoute::new("primary", "old-model"));
        let attempted_routes = Arc::new(Mutex::new(Vec::new()));
        agent.set_primary_route_factory(Arc::new(SelectiveRouteFactory {
            accepted_routes: vec![
                ModelRoute::new("primary", "vendor/model/with/slash"),
                ModelRoute::new("primary", "shared"),
                ModelRoute::new("expert", "shared"),
            ],
            attempted_routes: Arc::clone(&attempted_routes),
        }));

        apply_restored_model_route(&mut agent, Some("vendor/model/with/slash"))
            .expect("legacy slash-containing model should restore as a full model id");
        assert_eq!(
            agent.primary_route(),
            Some(&ModelRoute::new("primary", "vendor/model/with/slash"))
        );
        assert_eq!(agent.model(), "vendor/model/with/slash");
        assert_eq!(
            *attempted_routes.lock().expect("attempts"),
            vec![ModelRoute::new("primary", "vendor/model/with/slash")]
        );

        attempted_routes.lock().expect("clear attempts").clear();
        apply_restored_model_route(&mut agent, Some("primary/shared"))
            .expect("active-provider qualified model should restore as a route");
        assert_eq!(
            agent.primary_route(),
            Some(&ModelRoute::new("primary", "shared"))
        );
        assert_eq!(agent.model(), "shared");
        assert_eq!(
            *attempted_routes.lock().expect("attempts"),
            vec![ModelRoute::new("primary", "shared")]
        );

        attempted_routes.lock().expect("clear attempts").clear();
        apply_restored_model_route(&mut agent, Some("expert/shared"))
            .expect("qualified model route should restore after the legacy candidate fails");
        assert_eq!(
            agent.primary_route(),
            Some(&ModelRoute::new("expert", "shared"))
        );
        assert_eq!(agent.model(), "shared");
        assert_eq!(
            *attempted_routes.lock().expect("attempts"),
            vec![
                ModelRoute::new("primary", "expert/shared"),
                ModelRoute::new("expert", "shared"),
            ]
        );
    }

    #[test]
    fn legacy_resume_reapplies_the_active_provider_route() {
        let mut agent = test_agent();
        agent.set_primary_route(ModelRoute::new("primary", "old-model"));
        let applied_route = Arc::new(Mutex::new(None));
        agent.set_primary_route_factory(Arc::new(RecordingRouteFactory {
            accepted_routes: vec![ModelRoute::new("primary", "legacy-model")],
            applied_route: Arc::clone(&applied_route),
        }));

        apply_restored_model_route(&mut agent, Some("legacy-model"))
            .expect("legacy restore should reapply the active provider route");

        let expected_route = ModelRoute::new("primary", "legacy-model");
        assert_eq!(agent.primary_route(), Some(&expected_route));
        assert_eq!(
            *applied_route.lock().expect("captured route"),
            Some(expected_route)
        );
        assert_eq!(agent.model(), "legacy-model");
    }

    #[test]
    fn routed_resume_switches_the_provider_route() {
        let sessions_dir = temp_dir();
        let mut recorder = TranscriptRecorder::create(&sessions_dir).expect("create transcript");
        recorder
            .record_session_started("primary/shared")
            .expect("record session start");
        recorder
            .record_model_changed("primary/shared", "expert/shared")
            .expect("record model change");
        let session_id = recorder.session_id().to_string();
        drop(recorder);

        let mut agent = test_agent();
        agent.set_primary_route(ModelRoute::new("primary", "shared"));
        let expected_route = ModelRoute::new("expert", "shared");
        let applied_route = Arc::new(Mutex::new(None));
        agent.set_primary_route_factory(Arc::new(RecordingRouteFactory {
            accepted_routes: vec![ModelRoute::new("expert", "shared")],
            applied_route: Arc::clone(&applied_route),
        }));
        let live = Arc::new(Mutex::new(
            TranscriptRecorder::create(&sessions_dir).expect("create live transcript"),
        ));
        let prepared = prepare_resume_package(&sessions_dir, session_id).expect("prepare resume");
        let (fast_mode_auto_disabled, token_usage) =
            install_prepared_routed_resume_for_agent(&mut agent, &live, prepared)
                .expect("install routed resume");
        assert_eq!(agent.primary_route(), Some(&expected_route));
        assert_eq!(
            *applied_route.lock().expect("captured route"),
            Some(expected_route)
        );
        assert_eq!(agent.model(), "shared");
        assert!(!fast_mode_auto_disabled);
        assert_eq!(token_usage.output_tokens, 0);
    }

    #[test]
    fn restored_models_reconcile_persisted_fast_mode() {
        for (restored_model, expected_enabled) in [("claude-4", false), ("gpt-5.5-mini", true)] {
            let sessions_dir = temp_dir();
            let mut recorder =
                TranscriptRecorder::create(&sessions_dir).expect("create transcript");
            recorder
                .record_session_started("gpt-5.5")
                .expect("record session start");
            recorder
                .record_model_changed("gpt-5.5", restored_model)
                .expect("record model change");
            let session_id = recorder.session_id().to_string();
            drop(recorder);

            let mut agent = test_agent();
            let fast_mode_path = sessions_dir.join("letcode.toml");
            std::fs::write(
                &fast_mode_path,
                r#"active_provider = "primary"

[providers.primary]
base_url = "https://primary.invalid/v1"
api_key = "primary-key"
protocol = "responses"
[providers.primary.models."gpt-5.5"]
"#,
            )
            .expect("write Fast Mode config");
            let fast_mode = crate::fast_mode::FastMode::load(fast_mode_path, true);
            agent.set_fast_mode(fast_mode);
            let live = Arc::new(Mutex::new(
                TranscriptRecorder::create(&sessions_dir).expect("create live transcript"),
            ));
            let prepared =
                prepare_resume_package(&sessions_dir, session_id).expect("prepare resume");

            let auto_disabled = install_prepared_resume_for_agent(&mut agent, &live, prepared)
                .expect("install resume");
            assert_eq!(auto_disabled, !expected_enabled);
            assert_eq!(agent.model(), restored_model);
            assert_eq!(agent.fast_mode_enabled(), expected_enabled);
        }
    }

    #[test]
    fn token_usage_preparation_failure_leaves_live_session_and_agent_unchanged() {
        let sessions_dir = temp_dir();
        let mut target =
            TranscriptRecorder::create(&sessions_dir).expect("create target transcript");
        target
            .record_session_started("primary/shared")
            .expect("record target session start");
        let target_session_id = target.session_id().to_string();
        drop(target);

        let mut agent = test_agent();
        agent.set_primary_route(ModelRoute::new("primary", "gpt-5.5"));
        agent.set_primary_route_factory(Arc::new(SelectiveRouteFactory {
            accepted_routes: vec![ModelRoute::new("primary", "shared")],
            attempted_routes: Arc::new(Mutex::new(Vec::new())),
        }));
        let mut prepared =
            prepare_resume_package(&sessions_dir, target_session_id).expect("prepare resume");
        prepared.snapshot.snapshot.set_evidence(vec![
            crate::evidence::EvidenceRecord {
                id: "duplicate".into(),
                sequence: 1,
                timestamp_ms: 0,
                evidence_kind: crate::evidence::EvidenceKind::Decision,
                title: "duplicate".into(),
                summary: "duplicate".into(),
                detail: None,
                source: crate::evidence::EvidenceSource::Transcript { sequence: 1 },
                tags: Vec::new(),
            },
            crate::evidence::EvidenceRecord {
                id: "duplicate".into(),
                sequence: 2,
                timestamp_ms: 0,
                evidence_kind: crate::evidence::EvidenceKind::Decision,
                title: "duplicate".into(),
                summary: "duplicate".into(),
                detail: None,
                source: crate::evidence::EvidenceSource::Transcript { sequence: 2 },
                tags: Vec::new(),
            },
        ]);
        let live = Arc::new(Mutex::new(
            TranscriptRecorder::create(&sessions_dir).expect("create live transcript"),
        ));
        let live_session_id = live
            .lock()
            .expect("live transcript")
            .session_id()
            .to_string();
        let error = install_prepared_routed_resume_for_agent(&mut agent, &live, prepared)
            .expect_err("token usage preparation must fail before commit");
        assert!(error.to_string().contains("duplicate evidence id"));
        assert_eq!(agent.model(), "gpt-5.5");
        assert_eq!(
            agent.primary_route(),
            Some(&ModelRoute::new("primary", "gpt-5.5"))
        );
        assert_eq!(
            live.lock().expect("live transcript").session_id(),
            live_session_id,
            "the live transcript must remain unchanged when token usage preparation fails"
        );
    }

    #[test]
    fn failed_resume_reports_persisted_fast_mode_auto_disable() {
        let sessions_dir = temp_dir();
        let mut recorder = TranscriptRecorder::create(&sessions_dir).expect("create transcript");
        recorder
            .record_session_started("claude-4")
            .expect("record session start");
        let session_id = recorder.session_id().to_string();
        drop(recorder);

        let mut agent = test_agent();
        let fast_mode_path = sessions_dir.join("letcode.toml");
        std::fs::write(
            &fast_mode_path,
            r#"active_provider = "primary"

[providers.primary]
base_url = "https://primary.invalid/v1"
api_key = "primary-key"
protocol = "responses"
[providers.primary.models."gpt-5.5"]
"#,
        )
        .expect("write Fast Mode config");
        let fast_mode = crate::fast_mode::FastMode::load(fast_mode_path, true);
        agent.set_fast_mode(fast_mode);
        let live = Arc::new(Mutex::new(
            TranscriptRecorder::create(&sessions_dir).expect("create live transcript"),
        ));
        let mut prepared =
            prepare_resume_package(&sessions_dir, session_id).expect("prepare resume");
        let evidence = crate::evidence::EvidenceRecord {
            id: "duplicate".into(),
            sequence: 1,
            timestamp_ms: 0,
            evidence_kind: crate::evidence::EvidenceKind::Decision,
            title: "duplicate".into(),
            summary: "duplicate".into(),
            detail: None,
            source: crate::evidence::EvidenceSource::Transcript { sequence: 1 },
            tags: Vec::new(),
        };
        prepared
            .snapshot
            .snapshot
            .set_evidence(vec![evidence.clone(), evidence]);

        let error = install_prepared_resume_for_agent(&mut agent, &live, prepared)
            .expect_err("invalid restore should fail");
        assert!(error.fast_mode_auto_disabled);
        assert!(
            std::error::Error::source(&error).is_some(),
            "the wrapped anyhow error must remain in the source chain"
        );
        assert!(!agent.fast_mode_enabled());
    }
}
