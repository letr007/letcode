//! Session-owned idle settings mutations shared by TUI and line CLI.
//!
//! These helpers apply agent configuration changes and record transcript
//! provenance when the value actually changes. Frontends retain presentation
//! concerns (confirm dialogs, model catalog validation, status printing).

use std::sync::{Arc, Mutex};

use anyhow::{Result, anyhow};
use async_openai::config::Config;

use crate::agent::Agent;
use crate::permission::PermissionMode;
use crate::request_builder::ModelReasoningEffort;
use crate::transcript::TranscriptRecorder;

/// The result of a model apply that could persistently disable Fast Mode before
/// transcript provenance is recorded.
#[derive(Debug)]
pub struct ModelApplyError {
    error: anyhow::Error,
    fast_mode_auto_disabled: bool,
}

impl ModelApplyError {
    pub fn fast_mode_auto_disabled(&self) -> bool {
        self.fast_mode_auto_disabled
    }
}

impl std::fmt::Display for ModelApplyError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.error.fmt(formatter)
    }
}

impl std::error::Error for ModelApplyError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(self.error.as_ref())
    }
}

/// Apply a permission mode and record provenance when it changes.
pub fn apply_permission_mode<C: Config>(
    agent: &mut Agent<C>,
    transcript: &Arc<Mutex<TranscriptRecorder>>,
    mode: PermissionMode,
) -> Result<()> {
    let previous = agent.permission_mode();
    agent.set_permission_mode(mode);
    if previous != mode {
        transcript
            .lock()
            .map_err(|_| anyhow!("transcript recorder poisoned"))?
            .record_permission_mode_changed(previous.to_string(), mode.to_string())?;
    }
    Ok(())
}

/// Apply a model id and record provenance when it changes.
///
/// Callers that gate against a provider catalog should validate before this.
pub fn apply_model<C: Config>(
    agent: &mut Agent<C>,
    transcript: &Arc<Mutex<TranscriptRecorder>>,
    model: impl Into<String>,
) -> std::result::Result<bool, ModelApplyError> {
    let model = model.into();
    let previous = agent.model().to_string();
    let fast_mode_auto_disabled =
        agent
            .auto_disable_fast_mode_for_model(&model)
            .map_err(|error| ModelApplyError {
                error,
                fast_mode_auto_disabled: false,
            })?;
    if previous != model {
        transcript
            .lock()
            .map_err(|_| ModelApplyError {
                error: anyhow!("transcript recorder poisoned"),
                fast_mode_auto_disabled,
            })?
            .record_model_changed(previous, model.clone())
            .map_err(|error| ModelApplyError {
                error,
                fast_mode_auto_disabled,
            })?;
    }
    agent.set_model(model);
    Ok(fast_mode_auto_disabled)
}

/// Apply reasoning effort. No transcript provenance event today.
pub fn apply_reasoning_effort<C: Config>(
    agent: &mut Agent<C>,
    effort: ModelReasoningEffort,
) -> Result<()> {
    agent.set_reasoning_effort(effort)
}

#[cfg(test)]
mod tests {
    use super::*;
    use async_openai::{Client, config::OpenAIConfig};
    use serde_json::json;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn temp_transcript() -> (std::path::PathBuf, Arc<Mutex<TranscriptRecorder>>, String) {
        let base_dir = std::env::temp_dir().join(format!(
            "letcode-session-settings-test-{}",
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .expect("time ok")
                .as_nanos()
        ));
        let recorder = TranscriptRecorder::create(&base_dir).expect("create transcript");
        let session_id = recorder.session_id().to_string();
        (base_dir, Arc::new(Mutex::new(recorder)), session_id)
    }

    fn test_agent() -> Agent<OpenAIConfig> {
        let config = OpenAIConfig::new()
            .with_api_base("http://127.0.0.1:9/v1")
            .with_api_key("test-key");
        let client = Client::with_config(config);
        Agent::new(client, "gpt-5.5", 1, 1)
    }

    #[test]
    fn model_recording_failure_leaves_model_and_fast_mode_unchanged() {
        let (_base_dir, transcript, _session_id) = temp_transcript();
        let cloned = Arc::clone(&transcript);
        let join = std::thread::spawn(move || {
            let _guard = cloned.lock().expect("lock transcript");
            panic!("poison transcript mutex for test");
        });
        let _ = join.join();
        let mut agent = test_agent();
        let fast_mode_dir = std::env::temp_dir().join(format!(
            "letcode-session-settings-fast-mode-{}",
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .expect("time ok")
                .as_nanos()
        ));
        let fast_mode = crate::fast_mode::FastMode::load(&fast_mode_dir).expect("load fast mode");
        fast_mode.toggle(agent.model()).expect("enable fast mode");
        agent.set_fast_mode(fast_mode);

        let error = apply_model(&mut agent, &transcript, "claude-4")
            .expect_err("poisoned transcript must fail");
        assert!(error.fast_mode_auto_disabled());
        assert_eq!(agent.model(), "gpt-5.5");
        assert!(!agent.fast_mode_enabled());
        assert!(
            !crate::fast_mode::FastMode::load(&fast_mode_dir)
                .expect("reload fast mode")
                .enabled()
        );
    }

    #[test]
    fn fast_mode_persistence_failure_prevents_model_provenance() {
        let (base_dir, transcript, session_id) = temp_transcript();
        let mut agent = test_agent();
        let fast_mode_dir = std::env::temp_dir().join(format!(
            "letcode-session-settings-fast-mode-write-failure-{}",
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .expect("time ok")
                .as_nanos()
        ));
        let fast_mode = crate::fast_mode::FastMode::load(&fast_mode_dir).expect("load fast mode");
        fast_mode.toggle(agent.model()).expect("enable fast mode");
        std::fs::remove_dir_all(&fast_mode_dir).expect("remove fast mode directory");
        std::fs::write(&fast_mode_dir, "blocking file").expect("block persistence");
        agent.set_fast_mode(fast_mode);

        let error = apply_model(&mut agent, &transcript, "claude-4")
            .expect_err("fast mode persistence must fail");
        assert!(!error.fast_mode_auto_disabled());
        assert_eq!(agent.model(), "gpt-5.5");
        assert!(agent.fast_mode_enabled());
        assert!(
            crate::transcript::read_records(base_dir.join(format!("{session_id}.jsonl")))
                .expect("read records")
                .is_empty()
        );
    }

    #[test]
    fn apply_settings_records_provenance_only_on_change() {
        let (base_dir, transcript, session_id) = temp_transcript();
        let mut agent = test_agent();

        apply_permission_mode(&mut agent, &transcript, PermissionMode::Safe).expect("permission");
        apply_permission_mode(&mut agent, &transcript, PermissionMode::Safe).expect("idempotent");
        apply_model(&mut agent, &transcript, "gpt-5.5-mini").expect("model");
        apply_model(&mut agent, &transcript, "gpt-5.5-mini").expect("idempotent model");

        assert_eq!(agent.permission_mode(), PermissionMode::Safe);
        assert_eq!(agent.model(), "gpt-5.5-mini");

        let records = crate::transcript::read_records(base_dir.join(format!("{session_id}.jsonl")))
            .expect("read records");
        assert_eq!(records.len(), 2);
        let first = serde_json::to_value(&records[0]).expect("serialize");
        assert_eq!(first.get("kind"), Some(&json!("permission_mode_changed")));
        let second = serde_json::to_value(&records[1]).expect("serialize");
        assert_eq!(second.get("kind"), Some(&json!("model_changed")));
    }
}
