//! History validation replay equivalence and source-scope tests.

use super::*;
use crate::context_history::{
    HistoryApplication, HistoryCompartment, HistoryFact, HistoryFactCategory, HistoryPublication,
    HistorySelection, HistoryTier,
};
use crate::request_builder::HistoryItem;
use crate::tool::ToolResult;
use serde_json::json;

fn entry(sequence: u64, event: TranscriptEvent) -> TranscriptRecord {
    TranscriptRecord {
        session_id: "s".into(),
        sequence,
        timestamp_ms: 0,
        context_branch_id: None,
        event,
    }
}

fn publication(id: &str, source: u64) -> HistoryPublication {
    let sources = vec![format!("raw:{source}")];
    HistoryPublication {
        id: id.into(),
        project_path: None,
        external_fact_ids: vec![],
        source_ids: sources.clone(),
        compartments: vec![HistoryCompartment {
            id: format!("{id}-c"),
            title: id.into(),
            source_ids: sources.clone(),
            importance: 50,
            detailed: format!("{id} detail"),
            compact: id.into(),
            anchor: id.into(),
        }],
        facts: vec![HistoryFact {
            id: format!("{id}-f"),
            category: HistoryFactCategory::Constraints,
            text: format!("{id} constraint"),
            source_ids: sources,
            supersedes: vec![],
        }],
        withdrawn_fact_ids: vec![],
    }
}

fn application(id: &str) -> HistoryApplication {
    HistoryApplication {
        publication_ids: vec![id.into()],
        baseline: vec![HistorySelection {
            compartment_id: format!("{id}-c"),
            tier: HistoryTier::Compact,
        }],
        delta: vec![],
        baseline_fact_ids: vec![format!("{id}-f")],
        delta_fact_ids: vec![],
        withdrawn_fact_ids: vec![],
        first_kept_entry_id: None,
        legacy_summary: None,
    }
}

fn append_history(records: &mut Vec<TranscriptRecord>, id: &str, branch: Option<&str>) {
    let start = records.len() as u64 + 1;
    for (offset, event) in [
        TranscriptEvent::UserMessage { content: id.into() },
        TranscriptEvent::HistoryPublished(publication(id, start)),
        TranscriptEvent::HistoryApplied(application(id)),
    ]
    .into_iter()
    .enumerate()
    {
        let mut record = entry(start + offset as u64, event);
        record.context_branch_id = branch.map(str::to_owned);
        records.push(record);
    }
}

#[test]
fn history_validation_reuses_published_prefixes() {
    let mut records = vec![];
    for index in 0..50 {
        append_history(&mut records, &format!("p{index}"), None);
    }
    let mut cache = HistoryProjectionValidationCache::default();
    for (index, record) in records.iter().enumerate().filter(|(_, record)| {
        matches!(
            record.event,
            TranscriptEvent::HistoryPublished(_) | TranscriptEvent::HistoryApplied(_)
        )
    }) {
        let scope = context_compaction_validation_scope(
            &records,
            record.sequence - 1,
            SessionContextCursor {
                branch_id: Some(ROOT_CONTEXT_BRANCH_ID.into()),
                leaf_sequence: None,
            },
        )
        .unwrap();
        let (mut archive, history) = cache.snapshot(scope.selected_history_records());
        validate_history_event_in_scope(&scope, &record.event).unwrap();
        validate_history_event_projection(&mut archive, &history, &record.event).unwrap();
        assert_eq!(
            archive,
            crate::context_history::HistoryArchive::from_records(&records[..=index]).unwrap()
        );
    }
    assert_eq!(cache.total_applied_records, records.len() - 1);
    validate_context_projection_events(&records).unwrap();
}

#[test]
fn invalid_history_events_are_rejected_before_later_updates() {
    let mut bad_withdrawal = publication("invalid-withdrawal", 1);
    bad_withdrawal
        .withdrawn_fact_ids
        .push("unknown-fact".into());
    for bad in [
        TranscriptEvent::HistoryPublished(publication("invalid-source", 99)),
        TranscriptEvent::HistoryApplied(application("unpublished")),
        TranscriptEvent::HistoryPublished(bad_withdrawal),
    ] {
        let records = vec![
            entry(
                1,
                TranscriptEvent::UserMessage {
                    content: "source".into(),
                },
            ),
            entry(2, bad),
            entry(
                3,
                TranscriptEvent::HistoryPublished(publication("later", 1)),
            ),
        ];
        let scope = context_compaction_validation_scope(
            &records,
            1,
            SessionContextCursor {
                branch_id: Some(ROOT_CONTEXT_BRANCH_ID.into()),
                leaf_sequence: None,
            },
        )
        .unwrap();
        let mut cache = HistoryProjectionValidationCache::default();
        let (mut archive, history) = cache.snapshot(scope.selected_history_records());
        let before = archive.clone();
        assert!(validate_history_event_in_scope(&scope, &records[1].event).is_err());
        assert!(
            validate_history_event_projection(&mut archive, &history, &records[1].event).is_err()
        );
        assert_eq!(archive, before);
        assert!(validate_context_projection_events(&records).is_err());
    }
}

#[test]
fn history_validation_rebuilds_for_sibling_branch_sources() {
    let mut records = vec![];
    append_history(&mut records, "root", None);
    for branch in ["child", "sibling"] {
        records.push(entry(
            records.len() as u64 + 1,
            TranscriptEvent::ContextBranchCreated {
                branch_id: branch.into(),
                parent_branch_id: ROOT_CONTEXT_BRANCH_ID.into(),
                base_sequence: 3,
                label: None,
            },
        ));
        records.push(entry(
            records.len() as u64 + 1,
            TranscriptEvent::ContextCheckout {
                branch_id: branch.into(),
                leaf_sequence: 3,
            },
        ));
        append_history(&mut records, branch, Some(branch));
    }
    validate_context_projection_events(&records).unwrap();
    let mut cache = HistoryProjectionValidationCache::default();
    for (branch, excluded) in [
        ("child", "sibling"),
        ("sibling", "child"),
        ("child", "sibling"),
    ] {
        let scope = context_compaction_validation_scope(
            &records,
            records.last().unwrap().sequence,
            SessionContextCursor {
                branch_id: Some(branch.into()),
                leaf_sequence: None,
            },
        )
        .unwrap();
        let selected = scope.selected_history_records();
        let (archive, history) = cache.snapshot(selected);
        assert_eq!(
            archive,
            crate::context_history::HistoryArchive::from_records(selected).unwrap()
        );
        assert!(archive.publications.contains_key(branch));
        assert!(!archive.publications.contains_key(excluded));
        assert_eq!(
            format!("{history:?}"),
            format!("{:?}", restore_history_projection(selected))
        );
        assert_eq!(cache.total_applied_records, selected.len());
    }
}

#[test]
fn history_snapshot_normalization_preserves_pending_tool_state() {
    let prefix = vec![
        entry(
            1,
            TranscriptEvent::UserMessage {
                content: "read file".into(),
            },
        ),
        entry(
            2,
            TranscriptEvent::ToolCallStarted {
                call_id: "read".into(),
                name: "fs__read".into(),
                args: json!({"path":"file"}),
            },
        ),
    ];
    let completed = entry(
        3,
        TranscriptEvent::ToolCallFinished {
            call_id: "read".into(),
            name: "fs__read".into(),
            ok: true,
            output: ToolResult::ok("fs__read", json!({"content":"file content"})),
        },
    );
    let mut state = HistoryProjectionState::default();
    for record in &prefix {
        state.apply_record(record);
    }
    let (_, first) = state.snapshot();
    assert_eq!(
        first.len(),
        1,
        "unfinished call is absent from the normalized view"
    );
    state.apply_record(&completed);
    let (_, complete) = state.snapshot();
    assert!(complete.iter().any(|entry| matches!(&entry.item, HistoryItem::AssistantTurn { calls, .. } if calls.iter().any(|call| call.call_id == "read"))));
    assert!(complete.iter().any(
        |entry| matches!(&entry.item, HistoryItem::ToolOutput { call_id, .. } if call_id == "read")
    ));
    let mut records = prefix.clone();
    records.push(completed);
    assert_eq!(
        format!("{complete:?}"),
        format!("{:?}", restore_history_projection(&records))
    );

    let mut cancelled = HistoryProjectionState::default();
    for record in &prefix {
        cancelled.apply_record(record);
    }
    let cancellation = entry(
        3,
        TranscriptEvent::ToolCallCancelled {
            call_id: "read".into(),
            name: "fs__read".into(),
        },
    );
    cancelled.apply_record(&cancellation);
    let (_, view) = cancelled.snapshot();
    assert!(view.iter().any(
        |entry| matches!(&entry.item, HistoryItem::ToolOutput { call_id, .. } if call_id == "read")
    ));
    let next = entry(
        4,
        TranscriptEvent::UserMessage {
            content: "next".into(),
        },
    );
    cancelled.apply_record(&next);
    let (_, view) = cancelled.snapshot();
    let mut records = prefix;
    records.extend([cancellation, next]);
    assert_eq!(
        format!("{view:?}"),
        format!("{:?}", restore_history_projection(&records))
    );
    crate::protocol_frames::validate_history_items_complete(
        &view.into_iter().map(|entry| entry.item).collect::<Vec<_>>(),
        None,
    )
    .unwrap();
}
