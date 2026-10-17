use super::*;

fn runtime(dir: &std::path::Path) -> HistorianRuntime {
    let recorder = TranscriptRecorder::create(dir).expect("recorder");
    let (event_tx, _event_rx) = tokio::sync::mpsc::unbounded_channel();
    HistorianRuntime::new(
        SubagentPool::new(),
        dir.to_path_buf(),
        Arc::new(Mutex::new(recorder)),
        event_tx,
    )
}

#[test]
fn pass_limit_only_shrinks_as_oversized_rejections_repeat() {
    let dir = tempfile::tempdir().expect("temp dir");
    let runtime = runtime(dir.path());
    assert_eq!(runtime.pass_limit(), None);
    runtime.reduce_pass_limit(600);
    assert_eq!(runtime.pass_limit(), Some(300));
    runtime.reduce_pass_limit(600);
    assert_eq!(runtime.pass_limit(), Some(300));
    runtime.reduce_pass_limit(80);
    assert_eq!(runtime.pass_limit(), Some(40));
    runtime.reduce_pass_limit(1);
    assert_eq!(runtime.pass_limit(), Some(1));
}

#[test]
fn failure_tracking_is_scoped_to_the_failing_region() {
    let dir = tempfile::tempdir().expect("temp dir");
    let runtime = runtime(dir.path());
    assert!(!runtime.exhausted(Some("raw:1")));
    for _ in 0..MAX_HISTORIAN_ATTEMPTS {
        runtime.record_failure(HistorianFailure {
            first_source_id: Some("raw:1".into()),
            source_len: 600,
            class: FailureClass::Contract,
            attempts: 0,
            failed_at: Instant::now(),
            message: "cancelled".into(),
        });
    }
    assert_eq!(
        runtime.failure().map(|failure| failure.attempts),
        Some(MAX_HISTORIAN_ATTEMPTS)
    );
    assert_eq!(
        runtime.failure().map(|failure| failure.source_len),
        Some(600)
    );
    // The region that keeps failing stops being re-issued; a later region is
    // still worth an attempt.
    assert!(runtime.exhausted(Some("raw:1")));
    assert!(!runtime.exhausted(Some("raw:9")));
    assert!(!runtime.exhausted(None));
    runtime.clear_failure();
    assert!(runtime.failure().is_none());
    assert!(!runtime.exhausted(Some("raw:1")));
}

#[test]
fn a_parked_region_is_offered_again_after_the_cooldown() {
    let dir = tempfile::tempdir().expect("temp dir");
    let runtime = runtime(dir.path());
    let failure = |at: Instant| HistorianFailure {
        first_source_id: Some("raw:1".into()),
        source_len: 600,
        class: FailureClass::Transport,
        attempts: 0,
        failed_at: at,
        message: "cancelled".into(),
    };
    // A stale series does not carry over: this attempt starts a new one.
    runtime.record_failure(failure(Instant::now() - HISTORIAN_RETRY_COOLDOWN));
    assert_eq!(runtime.failure().map(|failure| failure.attempts), Some(1));
    assert!(!runtime.exhausted(Some("raw:1")));
    runtime.record_failure(failure(Instant::now()));
    runtime.record_failure(failure(Instant::now()));
    assert_eq!(
        runtime.failure().map(|failure| failure.attempts),
        Some(2),
        "the region is offered again instead of staying parked for the session"
    );
    runtime.record_failure(failure(Instant::now()));
    assert_eq!(
        runtime.failure().map(|failure| failure.attempts),
        Some(MAX_HISTORIAN_ATTEMPTS)
    );
    assert!(runtime.exhausted(Some("raw:1")));
}

fn historian_summary(status: SubagentStatus) -> crate::subagent::SubagentRunSummary {
    crate::subagent::SubagentRunSummary {
        run_id: "run-historian".into(),
        child_session_id: "child-historian".into(),
        agent_name: "historian".into(),
        status,
        failure_kind: None,
        summary: String::new(),
        structured_result: crate::subagent::StructuredSubagentResult {
            status: status.as_str().into(),
            summary: String::new(),
            malformed: false,
            findings: Vec::new(),
            files_read: Vec::new(),
            files_changed: Vec::new(),
            commands_run: Vec::new(),
            validation: Vec::new(),
            blockers: Vec::new(),
            next_steps: Vec::new(),
            run_id: "run-historian".into(),
            child_session_id: "child-historian".into(),
            raw_excerpt: None,
        },
    }
}

#[test]
fn dropped_historian_run_emits_a_child_terminal_only_when_dropped() {
    let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();
    emit_historian_terminal_on_drop(&tx, &Ok(historian_summary(SubagentStatus::Completed)));
    assert!(
        rx.try_recv().is_err(),
        "a completed historian published its own terminal"
    );
    for status in [SubagentStatus::Cancelled, SubagentStatus::TimedOut] {
        emit_historian_terminal_on_drop(&tx, &Ok(historian_summary(status)));
        assert!(matches!(
            rx.try_recv(),
            Ok(SessionTransportEvent::ChildSessionEvent {
                event: SessionEvent::Interrupted,
                ..
            })
        ));
    }
}
