use super::*;
use crate::permission::PermissionApproval;

fn cli_view() -> CliView {
    CliView {
        session_id: "session".into(),
        model_id: "provider/model".into(),
        model_label: "Model".into(),
        permission_mode_label: "default".into(),
        reasoning_effort: None,
    }
}

fn project_cli_event(view: &mut CliView, event: SessionTransportEvent) -> EventMarker {
    present_transport_event(
        event,
        OutputMode::FinalOnly,
        false,
        &mut HashMap::new(),
        &mut false,
        view,
        &mut String::new(),
        &mut None,
        &mut Vec::new(),
        None,
    )
    .expect("event projection")
}

#[test]
fn permission_mode_changes_only_on_authoritative_event() {
    let mut view = cli_view();
    let marker = project_cli_event(
        &mut view,
        SessionTransportEvent::PermissionModeChanged {
            mode: "safe".into(),
        },
    );

    assert!(marker.permission_mode_changed);
    assert_eq!(view.permission_mode_label, "safe");
}

#[test]
fn permission_error_does_not_change_cli_state() {
    let mut view = cli_view();
    let marker = project_cli_event(
        &mut view,
        SessionTransportEvent::Error(crate::session::ErrorEvent::new("failed")),
    );

    assert!(marker.error);
    assert_eq!(view.permission_mode_label, "default");
}

#[test]
fn reasoning_effort_changes_only_on_authoritative_event() {
    let mut view = cli_view();
    let marker = project_cli_event(
        &mut view,
        SessionTransportEvent::ReasoningEffortChanged {
            effort: ModelReasoningEffort::High,
        },
    );

    assert!(marker.reasoning_effort_changed);
    assert_eq!(view.reasoning_effort, Some(ModelReasoningEffort::High));
}

#[test]
fn terminal_permission_input_parses_once_always_and_denial() {
    assert_eq!(
        permission_approval_from_input("o", true),
        PermissionApproval::AllowOnce
    );
    assert_eq!(
        permission_approval_from_input("a", true),
        PermissionApproval::AllowAlways
    );
    assert_eq!(
        permission_approval_from_input("always", false),
        PermissionApproval::Deny
    );
    assert_eq!(
        permission_approval_from_input("n", true),
        PermissionApproval::Deny
    );
}

#[tokio::test]
async fn prompt_wait_collects_only_the_accepted_submission() {
    let (sender, mut events) = mpsc::unbounded_channel();
    let peer = UserMessageSubmission::new("peer", UserMessageContent::new("peer", Vec::new()));
    let user = UserMessageSubmission::new("user", UserMessageContent::new("user", Vec::new()));
    for event in [
        SessionTransportEvent::QueuedPromptAccepted { prompt: peer },
        SessionTransportEvent::AssistantDelta(crate::session::AssistantDeltaEvent::new(
            "peer output",
        )),
        SessionTransportEvent::Error(crate::session::ErrorEvent::new("peer error")),
        SessionTransportEvent::Done,
        SessionTransportEvent::QueuedPromptAccepted { prompt: user },
        SessionTransportEvent::AssistantDelta(crate::session::AssistantDeltaEvent::new(
            "user output",
        )),
        SessionTransportEvent::Done,
    ] {
        sender.send(event).expect("event receiver");
    }
    let outcome = wait_for_command(
        &mut events,
        &mut cli_view(),
        CommandWait::UntilDone,
        OutputMode::FinalOnly,
        Some("user"),
        None,
    )
    .await
    .expect("user turn completes");
    assert_eq!(outcome.assistant, "user output");
    assert!(outcome.error.is_none());
}

#[tokio::test]
async fn terminal_lines_share_one_input_source() {
    let (sender, lines) = std::sync::mpsc::channel();
    let ready = Arc::new(tokio::sync::Notify::new());
    let input = TerminalInput {
        lines,
        ready: ready.clone(),
    };
    sender.send(Ok("prompt\n".into())).expect("line receiver");
    ready.notify_one();
    assert_eq!(input.next_line().await.expect("prompt line"), "prompt\n");
    sender.send(Ok("y\n".into())).expect("line receiver");
    assert_eq!(input.read_line().expect("permission line"), "y\n");
}

#[tokio::test]
async fn spontaneous_permission_request_uses_terminal_input() {
    let (sender, lines) = std::sync::mpsc::channel();
    sender.send(Ok("y\n".into())).expect("line receiver");
    let input = TerminalInput {
        lines,
        ready: Arc::new(tokio::sync::Notify::new()),
    };
    let (reply, answer) = tokio::sync::oneshot::channel();
    present_transport_event(
        SessionTransportEvent::PermissionRequested {
            event: PermissionRequestEvent::new("peer-tool", "shell__exec", "Run command"),
            handle: crate::session::RunnerPermissionRequest::new(reply),
        },
        OutputMode::Streaming,
        true,
        &mut HashMap::new(),
        &mut false,
        &mut cli_view(),
        &mut String::new(),
        &mut None,
        &mut Vec::new(),
        Some(&input),
    )
    .expect("permission event");
    assert_eq!(
        answer.await.expect("permission answer"),
        crate::session::PermissionResponse::AllowOnce
    );
}

#[tokio::test]
async fn manual_compaction_waits_for_its_accepted_result() {
    for failure in [false, true] {
        let (sender, mut events) = mpsc::unbounded_channel();
        for event in [
            SessionTransportEvent::CompactionCommitted { summary: None },
            SessionTransportEvent::CompactionNoProgress {
                blockers: vec!["automatic".into()],
            },
            SessionTransportEvent::CompactionFailed,
            SessionTransportEvent::Notice(crate::session::NoticeEvent::info("peer notice")),
            SessionTransportEvent::Error(crate::session::ErrorEvent::new("peer error")),
            SessionTransportEvent::Done,
        ] {
            sender.send(event).unwrap();
        }
        let mut view = cli_view();
        let waiting = wait_for_command(
            &mut events,
            &mut view,
            CommandWait::UntilManualCompaction,
            OutputMode::FinalOnly,
            None,
            None,
        );
        tokio::pin!(waiting);
        assert!(
            tokio::time::timeout(Duration::from_millis(10), &mut waiting)
                .await
                .is_err()
        );
        sender
            .send(SessionTransportEvent::ManualCompactionStarted)
            .unwrap();
        if failure {
            sender
                .send(SessionTransportEvent::Error(
                    crate::session::ErrorEvent::new("missing credentials"),
                ))
                .unwrap();
        } else {
            sender
                .send(SessionTransportEvent::CompactionCommitted { summary: None })
                .unwrap();
            sender.send(SessionTransportEvent::Done).unwrap();
        }
        let outcome = waiting.await.unwrap();
        assert_eq!(
            outcome.error.as_deref(),
            failure.then_some("missing credentials")
        );
        assert!(outcome.notices.is_empty());
    }
}
