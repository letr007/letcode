use super::*;
use crate::session::{SessionEngine, TurnOutcome, engine::SessionEngineControl};
use crate::tui::{AppPhase, TuiState, map_key_event};
use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};

fn runtime() -> TuiRuntime {
    let (_tx, rx) = tokio::sync::mpsc::unbounded_channel();
    TuiRuntime::new(
        TuiState::default(),
        rx,
        vec![],
        Vec::new(),
        std::env::temp_dir(),
        std::env::temp_dir(),
    )
}

#[test]
fn idle_compact_reaches_engine_and_returns_to_completed() {
    for phase in [AppPhase::Idle, AppPhase::Completed] {
        let mut runtime = runtime();
        runtime.state.phase = phase;
        runtime.state_mut().set_input("/compact");
        let command = runtime
            .handle_input_action(crate::tui::InputAction::Submit)
            .unwrap()
            .expect("idle compact produces a command");
        let (mut engine, ingress, _egress) = SessionEngine::new();

        dispatch_command(&mut runtime, command, &ingress, true);

        assert!(matches!(
            engine.try_recv_control(),
            Ok(SessionEngineControl::Command(
                crate::session::engine::SessionEngineCommand::Compact
            ))
        ));
        assert!(runtime.session_turn_active);
        assert_eq!(runtime.state.phase, AppPhase::Running);
        assert_eq!(
            runtime.state.toast().map(|toast| toast.message.clone()),
            Some(runtime.state.t("runtime.compacting_context")),
        );

        runtime.apply_session_transport_event(SessionTransportEvent::CompactionStarted);
        runtime.apply_session_transport_event(SessionTransportEvent::CompactionNoProgress {
            blockers: Vec::new(),
        });
        runtime.apply_session_transport_event(SessionTransportEvent::TurnFinished {
            outcome: TurnOutcome::Completed,
        });
        assert!(!runtime.engine_turn_is_active());
        assert_eq!(runtime.state.phase, AppPhase::Completed);
    }
}

#[test]
fn idle_at_prefixed_prompt_reaches_engine_as_text() {
    let mut runtime = runtime();
    runtime
        .state_mut()
        .set_input("@explorer inspect the workspace");
    let command = runtime
        .handle_input_action(crate::tui::InputAction::Submit)
        .unwrap()
        .expect("idle prompt produces a command");
    let (mut engine, ingress, _egress) = SessionEngine::new();

    dispatch_command(&mut runtime, command, &ingress, true);

    assert!(matches!(
        engine.try_recv_control(),
        Ok(SessionEngineControl::Command(
            crate::session::engine::SessionEngineCommand::Prompt(prompt)
        )) if prompt.content.text == "@explorer inspect the workspace"
    ));
    assert!(runtime.session_turn_active);
    assert_eq!(runtime.state.phase, AppPhase::Running);
    assert_eq!(
        runtime
            .state
            .timeline
            .items()
            .iter()
            .filter(|item| matches!(item, crate::tui::TimelineItem::User(_)))
            .count(),
        1
    );
}

#[test]
fn compact_rejects_active_turn_without_ending_it() {
    let mut runtime = runtime();
    runtime.session_turn_active = true;
    runtime.state.phase = AppPhase::Running;
    runtime.state_mut().set_input("/compact");
    assert!(
        runtime
            .handle_input_action(crate::tui::InputAction::Submit)
            .unwrap()
            .is_none()
    );
    assert_eq!(runtime.state.input_buffer, "/compact");
    let (mut engine, ingress, _egress) = SessionEngine::new();
    let command = crate::session::SessionCommand::Compact;
    dispatch_command(&mut runtime, command, &ingress, true);
    assert!(matches!(
        engine.try_recv_control(),
        Err(tokio::sync::mpsc::error::TryRecvError::Empty)
    ));
    assert!(runtime.session_turn_active);
    assert_eq!(runtime.state.phase, AppPhase::Running);
    assert_eq!(
        runtime.state.toast().map(|toast| toast.message.clone()),
        Some(runtime.state.t("runtime.turn_running"))
    );
}

#[test]
fn failed_compact_dispatch_does_not_leave_running_state() {
    let mut runtime = runtime();
    runtime.state_mut().set_input("/compact");
    let command = runtime
        .handle_input_action(crate::tui::InputAction::Submit)
        .unwrap()
        .unwrap();
    let (engine, ingress, _egress) = SessionEngine::new();
    drop(engine);

    dispatch_command(&mut runtime, command, &ingress, true);

    assert!(!runtime.engine_turn_is_active());
    assert_eq!(runtime.state.phase, AppPhase::Error);
}

#[test]
fn failed_resume_dispatch_clears_pending_state() {
    let mut runtime = runtime();
    runtime.state_mut().set_input("/resume session-1");
    let command = runtime
        .handle_input_action(crate::tui::InputAction::Submit)
        .expect("resume command succeeds")
        .expect("resume command is dispatched");
    assert!(runtime.session_resume_pending);

    let (engine, ingress, _egress) = SessionEngine::new();
    drop(engine);
    dispatch_command(&mut runtime, command, &ingress, true);

    assert!(!runtime.session_resume_pending);
    assert_eq!(runtime.state().phase, AppPhase::Error);
}

#[tokio::test]
async fn running_destructive_command_is_rejected_before_engine_ingress() {
    let mut runtime = runtime();
    runtime.session_turn_active = true;
    runtime.state.phase = AppPhase::Running;
    let (mut engine, ingress, _egress) = SessionEngine::new();

    dispatch_command(
        &mut runtime,
        crate::session::SessionCommand::NewSession,
        &ingress,
        true,
    );

    assert!(matches!(
        engine.try_recv_control(),
        Err(tokio::sync::mpsc::error::TryRecvError::Empty)
    ));
    let expected = runtime.state().t("runtime.turn_running");
    assert_eq!(
        runtime.state().toast().map(|toast| toast.message.as_str()),
        Some(expected.as_str())
    );
}

#[tokio::test]
async fn running_setting_dispatch_projects_pending_value_without_local_notice() {
    let mut runtime = runtime();
    runtime.session_turn_active = true;
    runtime.state.phase = AppPhase::Running;
    let (mut engine, ingress, _egress) = SessionEngine::new();

    dispatch_command(
        &mut runtime,
        crate::session::SessionCommand::SetModel("provider/model".into()),
        &ingress,
        true,
    );

    assert!(matches!(
        engine.recv_control().await,
        Some(SessionEngineControl::Command(
            crate::session::engine::SessionEngineCommand::SetModel(model)
        )) if model == "provider/model"
    ));
    assert!(runtime.state().toast().is_none());
    assert_eq!(
        runtime.state().pending_composer_settings.model,
        Some(("provider/model".into(), "provider/model".into()))
    );
    assert_eq!(runtime.state().model_id, "pending-runtime-model");
}

#[test]
fn failed_mcp_dispatch_clears_updating_state() {
    let mut runtime = runtime();
    runtime
        .state_mut()
        .set_mcp_server_updating("docs".into(), true);
    let (engine, ingress, _egress) = SessionEngine::new();
    drop(engine);

    dispatch_command(
        &mut runtime,
        crate::session::SessionCommand::ToggleMcpServer("docs".into()),
        &ingress,
        true,
    );

    assert!(!runtime.state().mcp_updating.contains("docs"));
    assert_eq!(runtime.state().phase, AppPhase::Error);
}

#[test]
fn failed_setting_dispatch_does_not_project_pending_value() {
    let mut runtime = runtime();
    runtime.session_turn_active = true;
    runtime.state.phase = AppPhase::Running;
    let (engine, ingress, _egress) = SessionEngine::new();
    drop(engine);

    dispatch_command(
        &mut runtime,
        crate::session::SessionCommand::SetPermissionMode(crate::permission::PermissionMode::Safe),
        &ingress,
        true,
    );

    assert_eq!(
        runtime.state().pending_composer_settings.permission_mode,
        None
    );
    assert_eq!(runtime.state().phase, AppPhase::Error);
}

#[test]
fn idle_prompt_dispatch_shows_message_without_queue_marker() {
    let mut runtime = runtime();
    runtime.state_mut().set_input("same");
    let command = runtime
        .handle_input_action(crate::tui::InputAction::Submit)
        .expect("submit succeeds")
        .expect("prompt command");
    let RuntimeCommand::SubmitPrompt(prompt) = &command else {
        panic!("expected prompt command");
    };
    let submission_id = prompt.id.clone();
    let (mut engine, ingress, _egress) = SessionEngine::new();

    dispatch_command(&mut runtime, command, &ingress, true);

    assert!(matches!(
        engine.try_recv_control(),
        Ok(SessionEngineControl::Command(
            crate::session::engine::SessionEngineCommand::Prompt(submitted)
        )) if submitted.id == submission_id
    ));
    assert!(runtime.queued_prompts.is_empty());
    assert!(matches!(
        runtime.state().timeline.items(),
        [crate::tui::TimelineItem::User(message)] if !message.queued
    ));
    runtime.apply_session_transport_event(SessionTransportEvent::QueuedPromptAccepted {
        prompt: crate::user_content::UserMessageSubmission::new(
            submission_id.clone(),
            "same".into(),
        ),
    });
    assert!(matches!(
        runtime.state().timeline.items(),
        [crate::tui::TimelineItem::User(message)] if !message.queued
    ));
    assert_eq!(
        runtime
            .state()
            .timeline
            .items()
            .iter()
            .filter(|item| matches!(
                item,
                crate::tui::TimelineItem::User(message)
                    if message.submission_id.as_deref() == Some(submission_id.as_str())
            ))
            .count(),
        1
    );
}

#[test]
fn matching_user_message_event_does_not_duplicate_local_echo() {
    let mut runtime = runtime();
    runtime.apply_session_transport_event(SessionTransportEvent::TurnFinished {
        outcome: TurnOutcome::Interrupted,
    });
    runtime.state_mut().set_input("same");
    let command = runtime
        .handle_input_action(crate::tui::InputAction::Submit)
        .expect("submit succeeds")
        .expect("prompt command");
    let RuntimeCommand::SubmitPrompt(prompt) = &command else {
        panic!("expected prompt command");
    };
    let submitted_prompt = prompt.clone();
    let (mut engine, ingress, _egress) = SessionEngine::new();
    dispatch_command(&mut runtime, command, &ingress, true);
    let _ = engine.try_recv_control();

    runtime.apply_session_transport_event(SessionTransportEvent::UserMessage(
        crate::tui::UserMessageEvent::from_submission(submitted_prompt.clone()),
    ));

    assert_eq!(
        runtime
            .state()
            .timeline
            .items()
            .iter()
            .filter(|item| matches!(
                item,
                crate::tui::TimelineItem::User(message)
                    if message.submission_id.as_deref() == Some(submitted_prompt.id.as_str())
            ))
            .count(),
        1
    );
    runtime.apply_session_transport_event(SessionTransportEvent::ToolPending(
        crate::session::ToolPendingEvent::new("read-1", "fs__read"),
    ));
    assert!(
        runtime
            .state()
            .timeline
            .items()
            .iter()
            .any(|item| { matches!(item, crate::tui::TimelineItem::Tool(_)) })
    );
}

#[test]
fn different_submission_id_with_same_text_remains_visible() {
    let mut runtime = runtime();
    runtime.state_mut().set_input("same");
    let command = runtime
        .handle_input_action(crate::tui::InputAction::Submit)
        .expect("submit succeeds")
        .expect("prompt command");
    let (mut engine, ingress, _egress) = SessionEngine::new();
    dispatch_command(&mut runtime, command, &ingress, true);
    let _ = engine.try_recv_control();

    runtime.apply_session_transport_event(SessionTransportEvent::UserMessage(
        crate::tui::UserMessageEvent::new("same"),
    ));

    assert_eq!(
        runtime
            .state()
            .timeline
            .items()
            .iter()
            .filter(|item| matches!(
                item,
                crate::tui::TimelineItem::User(message) if message.text == "same"
            ))
            .count(),
        2
    );
}

#[test]
fn failed_queued_prompt_dispatch_clears_preview() {
    let mut runtime = runtime();
    runtime.session_turn_active = true;
    runtime.state.phase = AppPhase::Running;
    runtime.state_mut().set_input("queued prompt");
    let command = runtime
        .handle_input_action(crate::tui::InputAction::Submit)
        .expect("queue prompt succeeds")
        .expect("queued command");
    let (engine, ingress, _egress) = SessionEngine::new();
    drop(engine);

    dispatch_command(&mut runtime, command, &ingress, true);

    assert!(runtime.queued_prompts.is_empty());
    assert!(!runtime.engine_turn_is_active());
}

#[test]
fn failed_prompt_dispatch_does_not_create_local_echo() {
    let mut runtime = runtime();
    runtime.state_mut().set_input("same");
    let command = runtime
        .handle_input_action(crate::tui::InputAction::Submit)
        .expect("submit succeeds")
        .expect("prompt command");
    let (engine, ingress, _egress) = SessionEngine::new();
    drop(engine);

    dispatch_command(&mut runtime, command, &ingress, true);

    assert!(
        !runtime
            .state()
            .timeline
            .items()
            .iter()
            .any(|item| matches!(item, crate::tui::TimelineItem::User(_)))
    );
}

#[tokio::test]
async fn double_escape_dispatches_frontend_neutral_interrupt() {
    let mut runtime = runtime();
    runtime.session_turn_active = true;
    runtime.state.phase = AppPhase::Error;
    let (mut engine, ingress, _egress) = SessionEngine::new();
    let escape = KeyEvent::new(KeyCode::Esc, KeyModifiers::NONE);

    let first = runtime
        .handle_input_action(map_key_event(runtime.state(), escape))
        .expect("first Esc is accepted");
    assert_eq!(first, None);
    let second = runtime
        .handle_input_action(map_key_event(runtime.state(), escape))
        .expect("second Esc is accepted")
        .expect("second Esc requests interruption");

    dispatch_command(&mut runtime, second, &ingress, true);
    assert!(matches!(
        engine.recv_control().await,
        Some(SessionEngineControl::Interrupt)
    ));
}
