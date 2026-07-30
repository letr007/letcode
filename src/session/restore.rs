//! Session-owned restore projection helpers shared by TUI and line CLI.
//!
//! Phase L extracts the common "project restore snapshot including child
//! sessions" path. Agent restore and frontend timeline mapping remain outside.

use std::path::Path;
use std::sync::{Arc, Mutex};

use anyhow::Result;
use async_openai::config::Config;

use crate::agent::Agent;
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
    project_runtime_restore_snapshot(session_id, records, cursor, &children)
}

/// Session-owned resume package: records + restore snapshot + open recorder
/// with legacy branch adopted. Agent restore remains the caller's job.
pub struct PreparedResume {
    pub session_id: String,
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
    use crate::session::lifecycle::{load_session_records, open_resume_transcript};

    let sessions_dir = sessions_dir.as_ref();
    let session_id = session_id.into();
    let records = load_session_records(sessions_dir, &session_id)?;
    let snapshot = project_runtime_restore_snapshot_with_children(
        session_id.clone(),
        records.clone(),
        default_resume_cursor(),
        sessions_dir,
    )?;
    let mut recorder = open_resume_transcript(sessions_dir, &session_id)?;
    recorder.adopt_legacy_linear_branch(&snapshot.branch_id)?;
    Ok(PreparedResume {
        session_id,
        records,
        snapshot,
        recorder,
    })
}

/// Apply a prepared resume package onto the agent: restore runtime snapshot,
/// adopt latest model when present, and sync context-scope from the recorder.
///
/// Does **not** swap the live transcript recorder — callers still own that
/// under their locking / cleanup rules.
pub(crate) fn apply_prepared_resume_to_agent<C: Config>(
    agent: &mut Agent<C>,
    prepared: &PreparedResume,
) -> Result<()> {
    let model = prepared
        .snapshot
        .latest_model
        .as_deref()
        .unwrap_or(agent.model())
        .to_string();
    agent.restore_new_session_runtime_snapshot(
        prepared.snapshot.protocol_frames.clone(),
        prepared.snapshot.snapshot.clone(),
        prepared.snapshot.max_turn_id,
    )?;
    agent.set_model(model);
    let prepared_scope = prepare_context_scope(&prepared.recorder)?;
    apply_prepared_context_scope(agent, prepared_scope);
    Ok(())
}

/// Apply prepared resume state, swap the live recorder, then clean a prior empty file.
///
/// Build resume event payloads from `prepared` before this call (recorder is moved).
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

/// Build the session transport event emitted after a successful resume install.
///
/// Call this **before** moving `prepared.recorder` into the live transcript slot.
#[cfg(test)]
mod tests {
    use super::*;
    use async_openai::{Client, config::OpenAIConfig};
    use std::time::{SystemTime, UNIX_EPOCH};

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
            let fast_mode_dir = sessions_dir.join("fast-mode");
            let fast_mode =
                crate::fast_mode::FastMode::load(&fast_mode_dir).expect("load fast mode");
            fast_mode.toggle(agent.model()).expect("enable fast mode");
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
            assert_eq!(
                crate::fast_mode::FastMode::load(&fast_mode_dir)
                    .expect("reload fast mode")
                    .enabled(),
                expected_enabled
            );
        }
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
        let fast_mode_dir = sessions_dir.join("fast-mode");
        let fast_mode = crate::fast_mode::FastMode::load(&fast_mode_dir).expect("load fast mode");
        fast_mode.toggle(agent.model()).expect("enable fast mode");
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
        assert!(
            !crate::fast_mode::FastMode::load(&fast_mode_dir)
                .expect("reload fast mode")
                .enabled()
        );
    }
}

pub(crate) fn session_resumed_event(
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
