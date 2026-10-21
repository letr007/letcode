use super::*;

/// Compaction holds the agent, so every command that would touch it waits
/// without a notice; only transcript navigation is serviced meanwhile.
#[test]
fn compaction_services_navigation_and_queues_the_rest() {
    let (control_tx, mut control_rx) = mpsc::unbounded_channel();
    control_tx
        .send(SessionEngineControl::Command(
            SessionEngineCommand::SetModel("next".into()),
        ))
        .expect("queue a setting change");
    control_tx
        .send(SessionEngineControl::Command(
            SessionEngineCommand::ViewChild {
                navigation: crate::command::ChildNavigation::Next,
                anchor_child_session_id: None,
            },
        ))
        .expect("queue a child view request");
    let mut deferred_commands = VecDeque::new();

    let drained = drain_manual_compaction_controls(&mut control_rx, &mut deferred_commands);

    assert!(matches!(
        drained.navigation,
        Some(ManualCompactionNavigation::ViewChild { .. })
    ));
    assert!(matches!(
        deferred_commands.pop_front(),
        Some(SessionEngineCommand::SetModel(model)) if model == "next"
    ));
    assert!(deferred_commands.is_empty());
}

fn peer_message(text: &str) -> SessionEngineCommand {
    let address = crate::peer::PeerAddress {
        instance_id: "instance".into(),
        session_id: "session".into(),
        generation: 0,
    };
    SessionEngineCommand::PeerMessage(crate::peer::PeerMessage {
        from: address.clone(),
        to: address,
        message: text.into(),
    })
}

#[tokio::test]
async fn interrupt_discards_only_peer_messages_before_its_fifo_boundary() {
    let (tx, mut rx) = mpsc::unbounded_channel();
    let mut deferred = VecDeque::from([peer_message("deferred")]);
    for control in [
        SessionEngineControl::Command(peer_message("before")),
        SessionEngineControl::Interrupt,
        SessionEngineControl::Command(peer_message("after")),
    ] {
        tx.send(control).unwrap();
    }
    let operation = std::future::pending::<()>();
    tokio::pin!(operation);
    assert!(matches!(
        select_active_session_operation(&mut rx, &mut deferred, operation.as_mut()).await,
        ActiveSessionOperation::Interrupted
    ));
    assert_eq!(deferred.len(), 1);
    assert!(matches!(
        next_idle_session_operation(&mut rx, &mut deferred).await,
        Some(IdleSessionOperation::Command(SessionEngineCommand::PeerMessage(message)))
            if message.message == "after"
    ));
    for control in [
        SessionEngineControl::Command(peer_message("before compact stop")),
        SessionEngineControl::Interrupt,
        SessionEngineControl::Command(peer_message("after compact stop")),
    ] {
        tx.send(control).unwrap();
    }
    let drained = drain_manual_compaction_controls(&mut rx, &mut deferred);
    assert!(drained.interrupted);
    assert_eq!(deferred.len(), 1);
    assert!(matches!(
        next_idle_session_operation(&mut rx, &mut deferred).await,
        Some(IdleSessionOperation::Command(SessionEngineCommand::PeerMessage(message)))
            if message.message == "after compact stop"
    ));
    for control in [
        SessionEngineControl::Command(peer_message("before first stop")),
        SessionEngineControl::Interrupt,
        SessionEngineControl::Command(peer_message("between stops")),
        SessionEngineControl::Interrupt,
        SessionEngineControl::Command(peer_message("after last stop")),
    ] {
        tx.send(control).unwrap();
    }
    assert!(matches!(
        select_active_session_operation(&mut rx, &mut deferred, operation.as_mut()).await,
        ActiveSessionOperation::Interrupted
    ));
    assert_eq!(deferred.len(), 1);
    assert!(matches!(
        next_idle_session_operation(&mut rx, &mut deferred).await,
        Some(IdleSessionOperation::Command(SessionEngineCommand::PeerMessage(message)))
            if message.message == "after last stop"
    ));
    deferred.push_back(peer_message("idle queued"));
    tx.send(SessionEngineControl::Interrupt).unwrap();
    tx.send(SessionEngineControl::Command(peer_message(
        "after idle stop",
    )))
    .unwrap();
    assert!(matches!(
        next_idle_session_operation(&mut rx, &mut deferred).await,
        Some(IdleSessionOperation::Interrupt)
    ));
    assert_eq!(deferred.len(), 1);
    assert!(matches!(
        next_idle_session_operation(&mut rx, &mut deferred).await,
        Some(IdleSessionOperation::Command(SessionEngineCommand::PeerMessage(message)))
            if message.message == "after idle stop"
    ));
}

#[test]
fn discarding_peer_messages_keeps_local_prompts() {
    let mut commands = VecDeque::from([
        peer_message("before"),
        SessionEngineCommand::Prompt(crate::user_content::UserMessageSubmission::new(
            "local",
            crate::user_content::UserMessageContent::from("local"),
        )),
        peer_message("after"),
    ]);

    discard_queued_peer_messages(&mut commands);

    assert_eq!(commands.len(), 1);
    assert!(matches!(
        &commands[0],
        SessionEngineCommand::Prompt(prompt) if prompt.id == "local"
    ));
}

#[test]
fn queued_inputs_keep_fifo_order_across_flush_and_local_cancellation() {
    let prompt = |id: &str| {
        SessionEngineCommand::Prompt(crate::user_content::UserMessageSubmission::new(
            id,
            crate::user_content::UserMessageContent::from(id),
        ))
    };
    let mut parked = VecDeque::from([prompt("first"), peer_message("second"), prompt("cancel")]);
    let mut deferred = VecDeque::from([peer_message("fourth"), prompt("fifth")]);
    enqueue_deferred_command(
        &mut deferred,
        SessionEngineCommand::CancelQueuedPrompt("cancel".into()),
    );
    cancel_queued_prompt(&mut parked, "cancel");
    flush_parked_commands(&mut deferred, &mut parked);
    let labels: Vec<_> = deferred
        .into_iter()
        .filter_map(|command| match command {
            SessionEngineCommand::Prompt(prompt) => Some(prompt.id),
            SessionEngineCommand::PeerMessage(message) => Some(message.message),
            _ => None,
        })
        .collect();
    assert_eq!(labels, ["first", "second", "fourth", "fifth"]);
}
