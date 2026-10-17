use super::*;
use crate::request_builder::HistoryItem;
use crate::transcript::{TranscriptEvent as E, TranscriptRecord, restore_session_history};
fn record(sequence: u64, event: E) -> TranscriptRecord {
    TranscriptRecord {
        session_id: "history-test".into(),
        sequence,
        timestamp_ms: 0,
        context_branch_id: None,
        event,
    }
}
fn fixture() -> (
    Vec<TranscriptRecord>,
    HistoryPublication,
    HistoryApplication,
) {
    let records = vec![
        record(
            1,
            E::UserMessage {
                content: "Find the parser failure".into(),
            },
        ),
        record(
            2,
            E::AssistantMessage {
                content: "UTF-8 boundary caused the failure".into(),
            },
        ),
        record(
            3,
            E::UserMessage {
                content: "Continue with the fix".into(),
            },
        ),
    ];
    let source_ids = vec!["raw:1".into(), "raw:2".into()];
    let p = HistoryPublication {
        id: "p1".into(),
        project_path: None,
        external_fact_ids: vec![],
        withdrawn_fact_ids: vec![],
        source_ids: source_ids.clone(),
        compartments: vec![HistoryCompartment {
            id: "c1".into(),
            title: "Parser failure".into(),
            source_ids: source_ids.clone(),
            importance: 80,
            detailed: "UTF-8 boundary caused the failure".into(),
            compact: "Parser boundary error".into(),
            anchor: "UTF-8".into(),
        }],
        facts: vec![HistoryFact {
            id: "f1".into(),
            category: HistoryFactCategory::Constraints,
            text: "Parser offsets are byte offsets".into(),
            source_ids,
            supersedes: vec![],
        }],
    };
    let a = HistoryApplication {
        publication_ids: vec!["p1".into()],
        baseline: vec![HistorySelection {
            compartment_id: "c1".into(),
            tier: HistoryTier::Compact,
        }],
        delta: vec![],
        baseline_fact_ids: vec![],
        delta_fact_ids: vec![],
        withdrawn_fact_ids: vec![],
        first_kept_entry_id: Some("raw:3".into()),
        legacy_summary: None,
    };
    (records, p, a)
}
#[test]
fn history_publication_and_application_replay_separately() {
    let (mut records, p, a) = fixture();
    records.push(record(4, E::HistoryPublished(p)));
    assert_eq!(restore_session_history(&records).unwrap().len(), 3);
    records.push(record(5, E::HistoryApplied(a)));
    let active = restore_session_history(&records).unwrap();
    assert!(
        matches!(&active[0],HistoryItem::ContextSummary {text} if text.contains("Parser boundary error"))
    );
    assert_eq!(active.len(), 2);
    let entries = source_entries(&records).unwrap();
    assert!(
        entries
            .iter()
            .any(|e| e.id == "raw:2" && e.text.contains("UTF-8 boundary"))
    );
    assert!(entries.iter().any(|e| e.id == "evidence:fact:f1"));
}
#[test]
fn history_cannot_retire_uncovered_tail_or_publish_unknown_source() {
    let (mut records, p, mut a) = fixture();
    records.push(record(4, E::HistoryPublished(p)));
    a.first_kept_entry_id = None;
    records.push(record(5, E::HistoryApplied(a)));
    assert!(restore_session_history(&records).is_err());
    let (mut records, mut p, _) = fixture();
    p.source_ids = vec!["raw:99".into()];
    p.compartments[0].source_ids = p.source_ids.clone();
    p.facts.clear();
    records.push(record(4, E::HistoryPublished(p)));
    assert!(restore_session_history(&records).is_err());
}
#[test]
fn history_branch_scope_cannot_read_sibling_sources() {
    let (mut records, p, _) = fixture();
    records.push(record(
        4,
        E::ContextBranchCreated {
            branch_id: "other".into(),
            parent_branch_id: crate::transcript::ROOT_CONTEXT_BRANCH_ID.into(),
            base_sequence: 2,
            label: None,
        },
    ));
    let mut sibling = record(
        5,
        E::UserMessage {
            content: "sibling only".into(),
        },
    );
    sibling.context_branch_id = Some("other".into());
    records.push(sibling);
    let selected = crate::transcript::transcript_projection::selected_source_records(
        &records,
        crate::transcript::transcript_projection::SessionContextCursor {
            branch_id: Some(crate::transcript::ROOT_CONTEXT_BRANCH_ID.into()),
            leaf_sequence: None,
        },
    )
    .unwrap();
    assert!(
        !source_entries(&selected)
            .unwrap()
            .iter()
            .any(|e| e.text.contains("sibling only"))
    );
    let mut invalid = p;
    invalid.source_ids = vec!["raw:5".into()];
    invalid.compartments[0].source_ids = invalid.source_ids.clone();
    invalid.facts.clear();
    records.push(record(6, E::HistoryPublished(invalid)));
    assert!(restore_session_history(&records).is_err());
}
#[test]
fn canonical_legacy_tool_source_includes_every_call_in_its_group() {
    let records = vec![
        record(
            1,
            E::ToolCallStarted {
                call_id: "a".into(),
                name: "fs__read".into(),
                args: serde_json::json!({"path":"first.rs"}),
            },
        ),
        record(
            2,
            E::ToolCallStarted {
                call_id: "b".into(),
                name: "fs__read".into(),
                args: serde_json::json!({"path":"second.rs"}),
            },
        ),
        record(
            3,
            E::ToolCallFinished {
                call_id: "a".into(),
                name: "fs__read".into(),
                ok: true,
                output: crate::tool::ToolResult::ok(
                    "fs__read",
                    serde_json::json!({"content":"first"}),
                ),
            },
        ),
        record(
            4,
            E::ToolCallFinished {
                call_id: "b".into(),
                name: "fs__read".into(),
                ok: true,
                output: crate::tool::ToolResult::ok(
                    "fs__read",
                    serde_json::json!({"content":"second"}),
                ),
            },
        ),
    ];
    let entries = source_entries(&records).unwrap();
    let group = entries.iter().find(|e| e.id == "raw:1").unwrap();
    assert!(group.text.contains("first.rs"));
    assert!(group.text.contains("second.rs"));
}

#[test]
fn independent_facts_from_one_publication_are_not_deduplicated() {
    let (mut records, mut p, mut a) = fixture();
    a.baseline_fact_ids = vec!["f1".into(), "f2".into()];
    let mut second = p.facts[0].clone();
    second.id = "f2".into();
    second.text = "Parser input is UTF-8".into();
    p.facts.push(second);
    records.push(record(4, E::HistoryPublished(p)));
    records.push(record(5, E::HistoryApplied(a)));
    let facts = crate::evidence::restore_evidence_records(&records).unwrap();
    let (message, ids, _) = crate::evidence::evidence_context_message(&facts, "Parser", 3000);
    assert!(
        message.is_none() && ids.is_empty(),
        "facts are rendered by the frozen history slots, not injected twice"
    );
    let rendered = restore_session_history(&records).unwrap();
    assert!(
        matches!(&rendered[0],HistoryItem::ContextSummary {text} if text.contains("Parser offsets are byte offsets") && text.contains("Parser input is UTF-8"))
    );
}

#[test]
fn legacy_bootstrap_cannot_be_dropped_by_an_application() {
    let (mut records, mut p, mut a) = fixture();
    records.push(record(
        4,
        E::ContextCompaction(crate::agent::ContextCompactionEvent::succeeded_at(
            "legacy decision",
            Some("raw:3".into()),
        )),
    ));
    p.source_ids = vec!["raw:3".into()];
    p.compartments[0].source_ids = p.source_ids.clone();
    p.facts.clear();
    records.push(record(5, E::HistoryPublished(p)));
    a.first_kept_entry_id = None;
    records.push(record(6, E::HistoryApplied(a.clone())));
    assert!(restore_session_history(&records).is_err());
    a.legacy_summary = Some("legacy decision".into());
    records.last_mut().unwrap().event = E::HistoryApplied(a);
    let history = restore_session_history(&records).unwrap();
    assert!(
        matches!(&history[0], HistoryItem::ContextSummary { text } if text.contains("legacy decision"))
    );
}

#[test]
fn facts_revise_projection_without_deleting_original_observation() {
    let (mut records, p, a) = fixture();
    records.push(record(4, E::HistoryPublished(p.clone())));
    assert!(
        crate::evidence::restore_evidence_records(&records)
            .unwrap()
            .is_empty()
    );
    records.push(record(5, E::HistoryApplied(a)));
    let mut next = p;
    next.id = "p2".into();
    next.source_ids = vec!["raw:3".into()];
    next.compartments[0].id = "c2".into();
    next.compartments[0].source_ids = next.source_ids.clone();
    next.facts[0].id = "f2".into();
    next.facts[0].text = "Corrected parser constraint".into();
    next.facts[0].source_ids = next.source_ids.clone();
    next.facts[0].supersedes = vec!["f1".into()];
    records.push(record(6, E::HistoryPublished(next)));
    assert_eq!(
        crate::evidence::restore_evidence_records(&records).unwrap()[0].id,
        "fact:f1"
    );
    records.push(record(
        7,
        E::HistoryApplied(HistoryApplication {
            publication_ids: vec!["p2".into()],
            baseline: vec![],
            delta: vec![],
            baseline_fact_ids: vec![],
            delta_fact_ids: vec![],
            withdrawn_fact_ids: vec![],
            first_kept_entry_id: None,
            legacy_summary: None,
        }),
    ));
    restore_session_history(&records).unwrap();
    let archived = source_entries(&records).unwrap();
    let previous = archived
        .iter()
        .find(|e| e.id == "evidence:fact:f1")
        .unwrap();
    assert_eq!(previous.source_ids, vec!["raw:1", "raw:2"]);
    assert!(previous.text.contains("false"));
    let facts = crate::evidence::restore_evidence_records(&records).unwrap();
    assert_eq!(facts.len(), 1);
    assert_eq!(facts[0].id, "fact:f2");
    assert!(matches!(&records[3].event,E::HistoryPublished(p) if p.facts[0].id=="f1"));
}
