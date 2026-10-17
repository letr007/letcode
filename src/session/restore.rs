//! Session-owned restore projection helpers shared by TUI and line CLI.
//!
//! Phase L extracts the common "project restore snapshot including child
//! sessions" path. Agent restore and frontend timeline mapping remain outside.

use std::path::Path;
use std::sync::{Arc, Mutex};

use anyhow::{Result, bail};

use crate::agent::Agent;
use crate::config::ModelRoute;
use crate::permission::PermissionMode;
use crate::session::context_scope::{apply_prepared_context_scope, prepare_context_scope};
use crate::session::lifecycle::{cleanup_replaced_empty_session, replace_live_transcript};
use crate::transcript::transcript_projection::{
    RuntimeRestoreSnapshot, SessionContextCursor, project_runtime_restore_snapshot,
};
use crate::transcript::{TranscriptRecord, TranscriptRecorder, list_child_sessions_for_parent};

/// Attempts to load and open a transcript whose archive state may change while
/// a background archive pass is running.
const RESUME_ARCHIVE_RACE_ATTEMPTS: usize = 2;

/// Load the transcript and open it for appending, resolving the archive state
/// and retrying once when a background archive pass is in the way.
fn load_and_open_for_resume(
    sessions_dir: &Path,
    session_id: &str,
) -> Result<(Vec<TranscriptRecord>, TranscriptRecorder)> {
    load_and_open_for_resume_with(sessions_dir, session_id, || {})
}

fn load_and_open_for_resume_with(
    sessions_dir: &Path,
    session_id: &str,
    after_load: impl Fn(),
) -> Result<(Vec<TranscriptRecord>, TranscriptRecorder)> {
    use crate::session::lifecycle::{
        load_session_records_with_fingerprint, open_resume_transcript_with_records_at_fingerprint,
    };

    let last_attempt = RESUME_ARCHIVE_RACE_ATTEMPTS - 1;
    for attempt in 0..RESUME_ARCHIVE_RACE_ATTEMPTS {
        let displaced = match load_session_records_with_fingerprint(sessions_dir, session_id) {
            Ok((records, fingerprint)) => {
                after_load();
                match open_resume_transcript_with_records_at_fingerprint(
                    sessions_dir,
                    session_id,
                    &records,
                    &fingerprint,
                ) {
                    Ok(recorder) => return Ok((records, recorder)),
                    Err(error) => error,
                }
            }
            Err(error) => error,
        };

        if attempt == last_attempt {
            return Err(displaced);
        }
        tracing::debug!(
            session_id = %session_id,
            error = %displaced,
            "session transcript was not openable while resuming; resolving its archive state and retrying"
        );
        // The family may have moved into the archive since the load, or an
        // archive pass may still hold its lock before the archive files are
        // visible; resolving here waits that pass out and restores the family.
        super::archive::restore_session_family_for_resume(sessions_dir, session_id)?;
    }

    bail!("failed to open session {session_id} for resume")
}

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
    pub subagents_policy: crate::config::SubagentsConfig,
    #[allow(dead_code)]
    pub records: Vec<TranscriptRecord>,
    pub snapshot: RuntimeRestoreSnapshot,
    pub recorder: crate::transcript::TranscriptRecorder,
}

#[derive(Debug)]
pub struct ResumeInstallError {
    error: anyhow::Error,
    #[allow(dead_code)]
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
    let sessions_dir = sessions_dir.as_ref();
    let session_id = session_id.into();
    // A resume of an archived session restores its whole family first, so the
    // rest of this path only ever sees live transcripts.
    if let Some(stats) =
        super::archive::restore_session_family_for_resume(sessions_dir, &session_id)?
    {
        tracing::debug!(
            session_id = %session_id,
            transcripts = stats.transcripts,
            original_bytes = stats.original_bytes,
            "restored an archived session family for resume"
        );
    }
    let (mut records, mut recorder) = load_and_open_for_resume(sessions_dir, &session_id)?;
    let subagents_policy = crate::transcript::restore_subagents_policy(&records)?;
    if subagents_policy.mode == crate::config::SubagentMode::Swarm {
        crate::subagent::swarm::project(sessions_dir, &records)?;
    }
    let mut snapshot = project_runtime_restore_snapshot_with_children(
        session_id.clone(),
        records.clone(),
        default_resume_cursor(),
        sessions_dir,
    )?;
    recorder.adopt_legacy_linear_branch(&snapshot.branch_id)?;

    if let Some(turn_id) = snapshot.snapshot.current_turn_id {
        for (call_id, name) in
            super::interrupt::unfinished_tool_calls_in_active_turn(&snapshot.records)
        {
            recorder.record_tool_call_cancelled(call_id, name)?;
        }
        for run in super::interrupt::unfinished_subagent_runs_in_active_turn(&snapshot.records) {
            recorder.record_subagent_result_structured(
                run.run_id,
                run.parent_session_id,
                run.parent_run_id,
                run.child_session_id,
                run.agent_name,
                "cancelled",
                "Interrupted while resuming the parent session",
                None,
            )?;
        }
        recorder.record_turn_interrupted(Some(turn_id))?;
        records = crate::transcript::read_resumable_records_with_fingerprint(recorder.path())?.0;
        snapshot = project_runtime_restore_snapshot_with_children(
            session_id.clone(),
            records.clone(),
            default_resume_cursor(),
            sessions_dir,
        )?;
    }

    Ok(PreparedResume {
        session_id,
        subagents_policy,
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
pub(crate) fn apply_prepared_resume_to_agent(
    agent: &mut Agent,
    prepared: &PreparedResume,
) -> Result<()> {
    agent.restore_new_session_runtime_snapshot(
        prepared.snapshot.snapshot.clone(),
        prepared.snapshot.max_turn_id,
    )?;
    if let Some(model) = prepared.snapshot.latest_model.as_deref() {
        agent.set_model(model);
    }
    apply_restored_permission_mode(agent, prepared.snapshot.latest_permission_mode.as_deref());
    agent.clear_session_reasoning_efforts();
    let _ = apply_restored_reasoning_effort(agent, &prepared.snapshot.records);
    let prepared_scope = prepare_context_scope(&prepared.recorder)?;
    apply_prepared_context_scope(agent, prepared_scope);
    agent.set_subagents_policy(prepared.subagents_policy.clone());
    Ok(())
}

pub(crate) enum PreparedRestoredRoute {
    Prepared {
        target_model: String,
        route: crate::agent::PreparedPrimaryRoute,
    },
    ModelOnly(String),
}

impl PreparedRestoredRoute {
    pub(crate) fn target_model(&self) -> &str {
        match self {
            Self::Prepared { target_model, .. } => target_model,
            Self::ModelOnly(model) => model,
        }
    }

    pub(crate) fn candidate_session_usage_with_composition(
        &self,
        agent: &Agent,
        runtime_snapshot: &crate::runtime_context::RuntimeSnapshot,
    ) -> Result<(
        crate::agent::TokenUsageEstimate,
        Vec<crate::agent::PromptCompositionEntry>,
    )> {
        match self {
            Self::Prepared { route, .. } => {
                route.candidate_session_usage_with_composition(agent, runtime_snapshot)
            }
            Self::ModelOnly(model) => {
                agent.candidate_session_usage_with_composition(model, runtime_snapshot)
            }
        }
    }

    pub(crate) fn apply(self, agent: &mut Agent) {
        match self {
            Self::Prepared { route, .. } => agent.apply_prepared_route(route),
            Self::ModelOnly(model) => agent.set_model(model),
        }
    }
}

pub(crate) fn prepare_restored_model_route(
    agent: &Agent,
    latest_model: Option<&str>,
    fallback_route: Option<&ModelRoute>,
) -> Option<PreparedRestoredRoute> {
    let model = latest_model?;
    let Some(active_route) = agent.primary_route().cloned() else {
        return Some(PreparedRestoredRoute::ModelOnly(model.to_string()));
    };

    if let Some((provider, model_id)) = model.split_once('/')
        && provider == active_route.provider
    {
        let candidate = ModelRoute::new(provider, model_id);
        if let Ok(route) = agent.prepare_primary_route(candidate) {
            return Some(PreparedRestoredRoute::Prepared {
                target_model: model_id.to_string(),
                route,
            });
        }
    }

    let legacy_candidate = ModelRoute::new(active_route.provider.clone(), model);
    if let Ok(route) = agent.prepare_primary_route(legacy_candidate) {
        return Some(PreparedRestoredRoute::Prepared {
            target_model: model.to_string(),
            route,
        });
    }

    if let Some((provider, model_id)) = model.split_once('/')
        && provider != active_route.provider
    {
        let candidate = ModelRoute::new(provider, model_id);
        if let Ok(route) = agent.prepare_primary_route(candidate) {
            return Some(PreparedRestoredRoute::Prepared {
                target_model: model_id.to_string(),
                route,
            });
        }
    }

    let fallback = fallback_route.unwrap_or(&active_route).clone();
    if let Ok(route) = agent.prepare_primary_route(fallback.clone()) {
        return Some(PreparedRestoredRoute::Prepared {
            target_model: fallback.model.clone(),
            route,
        });
    }

    None
}

pub(crate) fn apply_prepared_restored_route(
    agent: &mut Agent,
    route: Option<PreparedRestoredRoute>,
) {
    if let Some(route) = route {
        route.apply(agent);
    }
}

pub(crate) fn apply_restored_permission_mode(agent: &mut Agent, mode: Option<&str>) {
    if let Some(mode) = mode.and_then(PermissionMode::parse) {
        agent.set_permission_mode(mode);
    }
}

pub(crate) fn apply_restored_reasoning_effort(
    agent: &mut Agent,
    records: &[TranscriptRecord],
) -> Option<String> {
    let model_id = agent.route_display_name();
    let effort = crate::transcript::restore_latest_reasoning_effort(records, &model_id)?;
    match agent.set_reasoning_effort(effort.clone()) {
        Ok(()) => None,
        Err(error) => Some(format!(
            "saved reasoning effort '{}' is unavailable for model '{}': {error}; using the model default",
            effort.as_str(),
            model_id
        )),
    }
}

#[cfg(test)]
pub(crate) fn apply_restored_model_route(
    agent: &mut Agent,
    latest_model: Option<&str>,
) -> Result<()> {
    let route = prepare_restored_model_route(agent, latest_model, None);
    apply_prepared_restored_route(agent, route);
    Ok(())
}

/// Apply prepared resume state, swap the live recorder, then clean a prior empty file.
///
/// Build resume event payloads from `prepared` before this call (recorder is moved).
#[cfg(test)]
pub fn install_prepared_resume_for_agent(
    agent: &mut Agent,
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
pub struct PreparedRoutedResumeInstall {
    prepared: PreparedResume,
    route: Option<PreparedRestoredRoute>,
    runtime_snapshot: crate::runtime_context::RuntimeSnapshot,
    token_usage: crate::session::event::TokenUsageEvent,
    prepared_scope: crate::session::context_scope::PreparedContextScope,
    prepared_fast_mode_disable: Option<crate::fast_mode::PreparedFastModeDisable>,
    old_path: std::path::PathBuf,
    new_path: std::path::PathBuf,
}

pub fn prepare_routed_resume_install(
    agent: &Agent,
    live: &Arc<Mutex<TranscriptRecorder>>,
    prepared: PreparedResume,
    fallback_route: Option<&ModelRoute>,
) -> std::result::Result<PreparedRoutedResumeInstall, ResumeInstallError> {
    let route = prepare_restored_model_route(
        agent,
        prepared.snapshot.latest_model.as_deref(),
        fallback_route,
    );
    let target_model = route
        .as_ref()
        .map_or_else(|| agent.model(), PreparedRestoredRoute::target_model);
    let runtime_snapshot = agent
        .validate_runtime_snapshot_restore(prepared.snapshot.snapshot.clone())
        .map_err(|error| ResumeInstallError::new(error, false))?;
    let token_usage = match route.as_ref() {
        Some(route) => {
            let (usage, prompt_composition) = route
                .candidate_session_usage_with_composition(agent, &runtime_snapshot)
                .map_err(|error| ResumeInstallError::new(error, false))?;
            crate::session::event::TokenUsageEvent::with_breakdown(
                usage.used_tokens,
                usage.context_window_tokens,
                usage.input_tokens,
                0,
                0,
            )
            .with_prompt_composition(prompt_composition)
        }
        None => restored_session_token_usage(agent, target_model, &runtime_snapshot)
            .map_err(|error| ResumeInstallError::new(error, false))?,
    };
    let prepared_scope = prepare_context_scope(&prepared.recorder)
        .map_err(|error| ResumeInstallError::new(error, false))?;
    let prepared_fast_mode_disable = agent
        .prepare_fast_mode_auto_disable(target_model)
        .map_err(|error| ResumeInstallError::new(error, false))?;
    let old_path = live
        .lock()
        .map_err(|_| {
            ResumeInstallError::new(anyhow::anyhow!("transcript recorder poisoned"), false)
        })?
        .path()
        .to_path_buf();
    let new_path = prepared.recorder.path().to_path_buf();
    Ok(PreparedRoutedResumeInstall {
        prepared,
        route,
        runtime_snapshot,
        token_usage,
        prepared_scope,
        prepared_fast_mode_disable,
        old_path,
        new_path,
    })
}

impl PreparedRoutedResumeInstall {
    pub(crate) fn session(&self) -> &PreparedResume {
        &self.prepared
    }

    pub(crate) fn fast_mode_auto_disabled(&self) -> bool {
        self.prepared_fast_mode_disable.is_some()
    }

    pub(crate) fn commit(
        self,
        agent: &mut Agent,
        live: &Arc<Mutex<TranscriptRecorder>>,
    ) -> (bool, crate::session::event::TokenUsageEvent) {
        let Self {
            prepared,
            route,
            runtime_snapshot,
            token_usage,
            prepared_scope,
            prepared_fast_mode_disable,
            old_path,
            new_path,
        } = self;
        let fast_mode_auto_disabled = prepared_fast_mode_disable.is_some();
        if let Some(prepared_fast_mode_disable) = prepared_fast_mode_disable {
            prepared_fast_mode_disable.commit();
        }
        let old_path_at_commit = replace_live_transcript(live, prepared.recorder)
            .expect("live transcript lock must remain healthy after preparation");
        debug_assert_eq!(old_path_at_commit, old_path);
        agent.clear_session_reasoning_efforts();
        apply_prepared_restored_route(agent, route);
        apply_restored_permission_mode(agent, prepared.snapshot.latest_permission_mode.as_deref());
        let _ = apply_restored_reasoning_effort(agent, &prepared.snapshot.records);
        agent.set_subagents_policy(prepared.subagents_policy);
        agent.install_validated_runtime_snapshot(runtime_snapshot);
        agent.restore_turn_sequence(prepared.snapshot.max_turn_id);
        apply_prepared_context_scope(agent, prepared_scope);
        let _ = cleanup_replaced_empty_session(old_path, &new_path);
        (fast_mode_auto_disabled, token_usage)
    }
}

#[cfg(test)]
pub fn install_prepared_routed_resume_for_agent(
    agent: &mut Agent,
    live: &Arc<Mutex<TranscriptRecorder>>,
    prepared: PreparedResume,
) -> std::result::Result<(bool, crate::session::event::TokenUsageEvent), ResumeInstallError> {
    let prepared = prepare_routed_resume_install(agent, live, prepared, None)?;
    Ok(prepared.commit(agent, live))
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
            crate::request_builder::HistoryItem::AssistantTurn {
                text: Some(text), ..
            } => Some(crate::agent::ConversationMessage {
                role: crate::agent::ConversationRole::Assistant,
                content: text,
            }),
            crate::request_builder::HistoryItem::AssistantTurn { text, .. } => {
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
pub fn restored_session_token_usage(
    agent: &Agent,
    model_id: &str,
    runtime_snapshot: &crate::runtime_context::RuntimeSnapshot,
) -> Result<crate::session::event::TokenUsageEvent> {
    let (usage, prompt_composition) =
        agent.candidate_session_usage_with_composition(model_id, runtime_snapshot)?;
    Ok(crate::session::event::TokenUsageEvent::with_breakdown(
        usage.used_tokens,
        usage.context_window_tokens,
        usage.input_tokens,
        0,
        0,
    )
    .with_prompt_composition(prompt_composition))
}

#[cfg(test)]
#[path = "../../tests/internal/session/restore/integration_tests.rs"]
mod integration_tests;
