use super::*;

fn record(sequence: u64, event: TranscriptEvent) -> TranscriptRecord {
    TranscriptRecord {
        session_id: "session".into(),
        sequence,
        timestamp_ms: 0,
        context_branch_id: None,
        event,
    }
}

fn branch_record(sequence: u64, branch_id: &str, event: TranscriptEvent) -> TranscriptRecord {
    TranscriptRecord {
        context_branch_id: Some(branch_id.into()),
        ..record(sequence, event)
    }
}

#[test]
fn navigation_state_preserves_the_complete_redo_stack() {
    let records = [record(
        1,
        TranscriptEvent::HistoryNavigation {
            operation: crate::transcript::HistoryNavigationOperation::Undo,
            target_sequence: 0,
            redo_stack: vec![8, 4],
            redo_target_sequence: None,
        },
    )];

    assert_eq!(
        history_navigation_state(&records),
        Some(HistoryNavigationState {
            target_sequence: 0,
            redo_stack: vec![8, 4],
        })
    );
}

#[test]
fn active_branch_append_invalidates_redo_but_global_metadata_and_session_titles_do_not() {
    let navigation = TranscriptEvent::HistoryNavigation {
        operation: crate::transcript::HistoryNavigationOperation::Undo,
        target_sequence: 4,
        redo_stack: vec![8],
        redo_target_sequence: None,
    };
    let checkout = TranscriptEvent::ContextCheckout {
        branch_id: "history-1".into(),
        leaf_sequence: 4,
    };
    let metadata = TranscriptEvent::ContextBranchSummary {
        branch_id: "history-1".into(),
        leaf_sequence: 4,
        summary: "metadata only".into(),
    };
    let expected = Some(HistoryNavigationState {
        target_sequence: 4,
        redo_stack: vec![8],
    });

    assert_eq!(
        history_navigation_state(&[
            record(1, navigation.clone()),
            record(2, checkout.clone()),
            record(3, metadata),
        ]),
        expected
    );
    assert_eq!(
        history_navigation_state(&[
            record(1, navigation.clone()),
            record(2, checkout.clone()),
            record(
                3,
                TranscriptEvent::SessionTitle {
                    title: "delayed title".into(),
                },
            ),
        ]),
        expected
    );
    assert_eq!(
        history_navigation_state(&[
            record(1, navigation),
            record(2, checkout),
            branch_record(
                3,
                "history-1",
                TranscriptEvent::ModelChanged {
                    previous_model: "old".into(),
                    new_model: "new".into(),
                },
            ),
        ]),
        None
    );
}
