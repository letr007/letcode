use super::*;
use crate::permission::PermissionMode;
use crate::request_builder::ModelReasoningEffort;
use std::time::{SystemTime, UNIX_EPOCH};

fn temp_transcript() -> Arc<Mutex<TranscriptRecorder>> {
    let base_dir = std::env::temp_dir().join(format!(
        "letcode-session-coordinator-test-{}",
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("time ok")
            .as_nanos()
    ));
    Arc::new(Mutex::new(
        TranscriptRecorder::create(&base_dir).expect("create transcript"),
    ))
}

fn test_agent() -> Agent {
    Agent::new("gpt-5.5", 1, 1)
}

#[test]
fn fast_mode_toggle_changes_memory_without_config_write() {
    let transcript = temp_transcript();
    let mut agent = test_agent();
    let fast_mode_path = std::env::temp_dir().join(format!(
        "letcode-session-coordinator-fast-mode-file-{}.toml",
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("time ok")
            .as_nanos()
    ));
    std::fs::write(&fast_mode_path, "not a config file").expect("create config file");
    let before = std::fs::read_to_string(&fast_mode_path).expect("read config before toggle");
    agent.set_fast_mode(crate::fast_mode::FastMode::load(&fast_mode_path, false));
    let (tx, mut rx) = mpsc::unbounded_channel();

    assert_eq!(
        SessionCoordinator::dispatch_idle_command(
            SessionCommand::ToggleFastMode,
            &mut agent,
            &transcript,
            &tx,
            None,
            &ChildLiveText::default(),
        )
        .expect("toggle dispatch"),
        IdleDispatch::Handled
    );
    assert!(agent.fast_mode_enabled());
    assert_eq!(
        std::fs::read_to_string(&fast_mode_path).expect("read config after toggle"),
        before
    );
    assert!(matches!(
        rx.try_recv().expect("toggle event"),
        SessionTransportEvent::FastModeChanged { enabled: true }
    ));
    assert!(rx.try_recv().is_err());
}

fn history_navigation_with_unsupported_model()
-> (Arc<Mutex<TranscriptRecorder>>, Agent, std::path::PathBuf) {
    let transcript = temp_transcript();
    {
        let mut recorder = transcript.lock().expect("recorder");
        recorder.record_user_message("first").expect("first user");
        recorder
            .record_model_changed("gpt-5.5", "claude-4")
            .expect("model changed");
        recorder.record_user_message("second").expect("second user");
    }
    let mut agent = test_agent();
    let fast_mode_dir = std::env::temp_dir().join(format!(
        "letcode-session-coordinator-navigation-fast-mode-{}",
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("time ok")
            .as_nanos()
    ));
    let fast_mode_config_path = fast_mode_dir.join("letcode.toml");
    std::fs::create_dir_all(&fast_mode_dir).expect("create Fast Mode config directory");
    std::fs::write(
        &fast_mode_config_path,
        r#"active_provider = "primary"

[providers.primary]
base_url = "https://primary.invalid/v1"
api_key = "primary-key"
protocol = "responses"
[providers.primary.models."gpt-5.5"]
"#,
    )
    .expect("write Fast Mode config");
    let fast_mode = crate::fast_mode::FastMode::load(&fast_mode_config_path, true);
    agent.set_fast_mode(fast_mode);
    (transcript, agent, fast_mode_config_path)
}

#[test]
fn navigation_fast_mode_auto_disable_is_memory_only() {
    let (transcript, mut agent, fast_mode_path) = history_navigation_with_unsupported_model();
    let config_before =
        std::fs::read_to_string(&fast_mode_path).expect("read config before navigation");
    let records_before = {
        let recorder = transcript.lock().expect("recorder");
        crate::transcript::read_records(recorder.path()).expect("read initial records")
    };
    let (tx, mut rx) = mpsc::unbounded_channel();

    SessionCoordinator::dispatch_idle_command(
        SessionCommand::NavigateHistory {
            target_entry_id: "entry-3".into(),
        },
        &mut agent,
        &transcript,
        &tx,
        None,
        &ChildLiveText::default(),
    )
    .expect("dispatch");

    assert_eq!(agent.model(), "claude-4");
    assert!(!agent.fast_mode_enabled());
    let after = {
        let recorder = transcript.lock().expect("recorder");
        crate::transcript::read_records(recorder.path()).expect("read final records")
    };
    assert!(after.len() > records_before.len());
    assert_eq!(
        std::fs::read_to_string(&fast_mode_path).expect("read config after navigation"),
        config_before
    );
    assert!(matches!(
        rx.try_recv().expect("navigation result"),
        SessionTransportEvent::FastModeChanged { enabled: false }
    ));
    assert!(matches!(
        rx.try_recv().expect("navigation notice"),
        SessionTransportEvent::Notice(_)
    ));
    assert!(matches!(
        rx.try_recv().expect("navigation result event"),
        SessionTransportEvent::SessionResumed { .. }
    ));
    assert!(rx.try_recv().is_err());
}

#[test]
fn navigation_commit_failure_keeps_fast_mode_unchanged() {
    let (transcript, mut agent, _fast_mode_path) = history_navigation_with_unsupported_model();
    let before = {
        let recorder = transcript.lock().expect("recorder");
        crate::transcript::read_records(recorder.path()).expect("read initial records")
    };
    FAIL_HISTORY_NAVIGATION_COMMIT.with(|fail| fail.set(true));
    let (tx, mut rx) = mpsc::unbounded_channel();

    SessionCoordinator::dispatch_idle_command(
        SessionCommand::NavigateHistory {
            target_entry_id: "entry-3".into(),
        },
        &mut agent,
        &transcript,
        &tx,
        None,
        &ChildLiveText::default(),
    )
    .expect("dispatch");

    assert_eq!(agent.model(), "gpt-5.5");
    assert!(agent.fast_mode_enabled());
    let after = {
        let recorder = transcript.lock().expect("recorder");
        crate::transcript::read_records(recorder.path()).expect("read final records")
    };
    assert_eq!(after.len(), before.len());
    assert!(matches!(
        rx.try_recv().expect("navigation error"),
        SessionTransportEvent::Error(error)
            if error.message.contains("injected history navigation commit failure")
    ));
    assert!(rx.try_recv().is_err());
}

#[test]
fn history_prepare_failure_prevents_navigation_commit() {
    let transcript = temp_transcript();
    {
        let mut recorder = transcript.lock().expect("recorder");
        recorder.record_user_message("first").expect("first user");
    }
    let before = {
        let recorder = transcript.lock().expect("recorder");
        crate::transcript::read_records(recorder.path()).expect("records before navigation")
    };
    let mut agent = test_agent();
    let (tx, mut rx) = mpsc::unbounded_channel();

    SessionCoordinator::dispatch_idle_command_with_history_prepare(
        SessionCommand::NavigateHistory {
            target_entry_id: "entry-0".into(),
        },
        &mut agent,
        &transcript,
        &tx,
        None,
        &ChildLiveText::default(),
        |_| anyhow::bail!("expert factory unavailable"),
    )
    .expect("dispatch");

    let after = {
        let recorder = transcript.lock().expect("recorder");
        crate::transcript::read_records(recorder.path()).expect("records after navigation")
    };
    assert_eq!(after.len(), before.len());
    assert_eq!(
        after
            .iter()
            .map(|record| record.sequence)
            .collect::<Vec<_>>(),
        before
            .iter()
            .map(|record| record.sequence)
            .collect::<Vec<_>>()
    );
    assert!(matches!(
        rx.try_recv().expect("navigation error"),
        SessionTransportEvent::Error(error)
            if error.message.contains("expert factory unavailable")
    ));
}

#[test]
fn navigation_install_is_infallible_after_precommit_validation() {
    let transcript = temp_transcript();
    {
        let mut recorder = transcript.lock().expect("recorder");
        recorder.record_user_message("first").expect("first user");
    }
    let mut agent = test_agent();
    let (tx, mut rx) = mpsc::unbounded_channel();
    assert_eq!(
        SessionCoordinator::dispatch_idle_command(
            SessionCommand::NavigateHistory {
                target_entry_id: "entry-0".into(),
            },
            &mut agent,
            &transcript,
            &tx,
            None,
            &ChildLiveText::default(),
        )
        .expect("dispatch"),
        IdleDispatch::HistoryNavigated
    );
    assert!(matches!(
        rx.try_recv().expect("navigation result"),
        SessionTransportEvent::SessionResumed { .. }
    ));
    assert_eq!(
        agent.runtime_snapshot_for_test().active_context.branch_id,
        "history-2"
    );
}

#[test]
fn history_navigation_announces_the_permission_mode_it_restored() {
    let transcript = temp_transcript();
    {
        let mut recorder = transcript.lock().expect("recorder");
        recorder
            .record_session_started("gpt-test")
            .expect("started");
        recorder
            .record_permission_mode_changed("default", "yolo")
            .expect("mode change");
        recorder.record_user_message("first").expect("first user");
        recorder
            .record_assistant_message("first answer")
            .expect("first answer");
        recorder.record_user_message("second").expect("second user");
        recorder
            .record_assistant_message("second answer")
            .expect("second answer");
    }
    let mut agent = test_agent();
    let (tx, mut rx) = mpsc::unbounded_channel();

    assert_eq!(
        SessionCoordinator::dispatch_idle_command(
            SessionCommand::Undo,
            &mut agent,
            &transcript,
            &tx,
            None,
            &ChildLiveText::default(),
        )
        .expect("dispatch"),
        IdleDispatch::HistoryNavigated
    );

    assert_eq!(agent.permission_mode().to_string(), "yolo");
    let mut events = Vec::new();
    while let Ok(event) = rx.try_recv() {
        events.push(event);
    }
    let mode_index = events
        .iter()
        .position(|event| {
            matches!(
                event,
                SessionTransportEvent::PermissionModeChanged { mode } if mode == "yolo"
            )
        })
        .expect("the restored mode is announced");
    let resumed_index = events
        .iter()
        .position(|event| matches!(event, SessionTransportEvent::SessionResumed { .. }))
        .expect("navigation reports the session it installed");
    assert!(
        mode_index < resumed_index,
        "the restored mode is announced before the session report: mode at {mode_index}, session at {resumed_index}"
    );
}

fn dispatch_navigation(
    command: SessionCommand,
    agent: &mut Agent,
    transcript: &Arc<Mutex<TranscriptRecorder>>,
    tx: &mpsc::UnboundedSender<SessionTransportEvent>,
    rx: &mut mpsc::UnboundedReceiver<SessionTransportEvent>,
) {
    assert_eq!(
        SessionCoordinator::dispatch_idle_command(
            command,
            agent,
            transcript,
            tx,
            None,
            &ChildLiveText::default(),
        )
        .expect("dispatch navigation"),
        IdleDispatch::HistoryNavigated
    );
    match rx.try_recv().expect("navigation result") {
        SessionTransportEvent::SessionResumed { .. } => {}
        other => panic!("expected history navigation to resume session, got {other:?}"),
    }
}

#[test]
fn durable_navigation_reopens_with_multilevel_redo_and_preserves_siblings_on_new_prompt() {
    let transcript = temp_transcript();
    let path = transcript.lock().expect("recorder").path().to_path_buf();
    {
        let mut recorder = transcript.lock().expect("recorder");
        recorder
            .record_session_started("gpt-test")
            .expect("started");
        recorder.record_user_message("first").expect("first user");
        recorder
            .record_assistant_message("first answer")
            .expect("first answer");
        recorder.record_user_message("second").expect("second user");
        recorder
            .record_assistant_message("second answer")
            .expect("second answer");
    }
    let mut agent = test_agent();
    let (tx, mut rx) = mpsc::unbounded_channel();

    dispatch_navigation(SessionCommand::Undo, &mut agent, &transcript, &tx, &mut rx);
    dispatch_navigation(SessionCommand::Undo, &mut agent, &transcript, &tx, &mut rx);
    let records = crate::transcript::read_records(&path).expect("records after undo");
    assert_eq!(
        crate::transcript::transcript_projection::history_navigation_state(&records),
        Some(
            crate::transcript::transcript_projection::HistoryNavigationState {
                target_sequence: 0,
                redo_stack: vec![5, 3],
            }
        )
    );
    assert!(matches!(
        records[8].event,
        crate::transcript::TranscriptEvent::ContextBranchCreated { .. }
    ));
    assert!(matches!(
        records[9].event,
        crate::transcript::TranscriptEvent::ContextCheckout { .. }
    ));
    assert!(matches!(
        records[10].event,
        crate::transcript::TranscriptEvent::HistoryNavigation { .. }
    ));

    let first_navigation = &records[5..8];
    let transaction_id = match &first_navigation[0].event {
        crate::transcript::TranscriptEvent::ContextBranchCreated { .. } => {
            first_navigation[0].sequence
        }
        other => panic!("expected branch creation, got {other:?}"),
    };
    assert_eq!(transaction_id, 6);

    let (sessions_dir, session_id) = {
        let recorder = transcript.lock().expect("recorder");
        (
            recorder
                .path()
                .parent()
                .expect("sessions directory")
                .to_path_buf(),
            recorder.session_id().to_string(),
        )
    };
    drop(transcript);
    let reopened = Arc::new(Mutex::new(
        TranscriptRecorder::open_existing(&sessions_dir, &session_id).expect("reopen"),
    ));
    dispatch_navigation(SessionCommand::Redo, &mut agent, &reopened, &tx, &mut rx);
    dispatch_navigation(SessionCommand::Redo, &mut agent, &reopened, &tx, &mut rx);

    {
        let mut recorder = reopened.lock().expect("reopened recorder");
        recorder
            .record_user_message("new branch")
            .expect("new prompt");
    }
    let records = crate::transcript::read_records(&path).expect("final records");
    assert_eq!(
        crate::transcript::transcript_projection::history_navigation_state(&records),
        None,
        "a new prompt invalidates persisted redo"
    );
    let entries = crate::transcript::transcript_projection::project_session_history_tree(&records);
    assert!(entries.iter().any(|entry| entry.label == "second"));
    assert!(entries.iter().any(|entry| entry.label == "new branch"));
    assert!(entries.iter().any(|entry| {
        entry.label == "new branch" && entry.parent_id.as_deref() == Some("entry-5")
    }));
}
