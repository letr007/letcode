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
