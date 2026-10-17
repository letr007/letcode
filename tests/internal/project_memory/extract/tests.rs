use super::*;
use crate::agent::TurnFinalizedEvent;

fn record(sequence: u64, event: TranscriptEvent) -> TranscriptRecord {
    TranscriptRecord {
        session_id: "s".into(),
        sequence,
        timestamp_ms: sequence as u128,
        context_branch_id: None,
        event,
    }
}

fn finalized() -> TranscriptEvent {
    TranscriptEvent::TurnFinalized(TurnFinalizedEvent {
        turn_id: 1,
        outcome: "completed".into(),
        tool_call_count: 0,
        continuation_count: 0,
        write_effects: 0,
        validation_effects: 0,
        failed_validation_effects: 0,
        validation_advisory_emitted: false,
    })
}

fn known_memory(id: &str) -> MemoryRecord {
    MemoryRecord {
        id: id.into(),
        kind: "decision".into(),
        title: "old decision".into(),
        summary: "old summary".into(),
        status: "useful".into(),
        session_id: "old-session".into(),
        branch_id: "main".into(),
        source_ids: vec!["raw:1".into()],
        paths: vec!["src/example.rs".into()],
        created_at_ms: 0,
        state: "active".into(),
        recall_count: 0,
        last_recalled_at: None,
        curated_at_ms: None,
    }
}

#[test]
fn processes_only_new_completed_work() {
    let records = vec![
        record(
            1,
            TranscriptEvent::UserMessage {
                content: "old".into(),
            },
        ),
        record(2, finalized()),
        record(
            3,
            TranscriptEvent::UserMessage {
                content: "新的设计".into(),
            },
        ),
    ];
    assert!(prepare_batch(&records, 2, &[]).unwrap().is_none());
    let mut records = records;
    records.push(record(4, finalized()));
    let batch = prepare_batch(&records, 2, &[]).unwrap().unwrap();
    assert_eq!(batch.end_sequence, 4);
    assert_eq!(batch.source_ids, vec!["raw:3", "raw:4"]);
    assert!(!batch.input.text.contains("old"));
}

#[test]
fn declared_structured_output_uses_the_memory_contract() {
    use crate::model_runtime::{StructuredOutput, StructuredOutputSupport};

    assert!(structured_output(None).is_none());
    assert_eq!(
        structured_output(Some(StructuredOutputSupport::JsonObject)),
        Some(StructuredOutput::JsonObject)
    );
    let Some(StructuredOutput::JsonSchema(schema)) =
        structured_output(Some(StructuredOutputSupport::JsonSchema))
    else {
        panic!("json_schema support must request an enforced schema");
    };
    assert!(schema.strict);
    assert_eq!(schema.schema["additionalProperties"], false);
    assert_eq!(
        schema.schema["required"],
        json!(["memories", "withdrawn_ids"])
    );
    assert!(
        schema.schema["properties"].get("compartments").is_none(),
        "the memory contract must not reuse the historian publication schema"
    );
}

#[test]
fn batches_multiple_observed_turns_in_one_extraction() {
    let records = vec![
        record(
            1,
            TranscriptEvent::UserMessage {
                content: "first".into(),
            },
        ),
        record(2, finalized()),
        record(
            3,
            TranscriptEvent::UserMessage {
                content: "second".into(),
            },
        ),
        record(4, finalized()),
    ];

    let batch = prepare_batch(&records, 0, &[]).unwrap().unwrap();
    assert_eq!(batch.end_sequence, 4);
    assert_eq!(batch.source_ids, ["raw:1", "raw:2", "raw:3", "raw:4"]);
}

#[test]
fn batching_stops_before_a_different_branch() {
    let mut first_terminal = record(2, finalized());
    first_terminal.context_branch_id = Some("branch-a".into());
    let mut second_user = record(
        3,
        TranscriptEvent::UserMessage {
            content: "other branch".into(),
        },
    );
    second_user.context_branch_id = Some("branch-b".into());
    let mut second_terminal = record(4, finalized());
    second_terminal.context_branch_id = Some("branch-b".into());
    let records = vec![
        {
            let mut first_user = record(
                1,
                TranscriptEvent::UserMessage {
                    content: "first branch".into(),
                },
            );
            first_user.context_branch_id = Some("branch-a".into());
            first_user
        },
        first_terminal,
        second_user,
        second_terminal,
    ];

    let batch = prepare_batch(&records, 0, &[]).unwrap().unwrap();
    assert_eq!(batch.end_sequence, 2);
    assert_eq!(batch.branch_id, "branch-a");
    assert_eq!(batch.source_ids, ["raw:1", "raw:2"]);
}

#[test]
fn interrupted_and_failed_turns_are_consumable_boundaries() {
    let interrupted = vec![
        record(
            1,
            TranscriptEvent::UserMessage {
                content: "unfinished".into(),
            },
        ),
        record(2, TranscriptEvent::TurnInterrupted { turn_id: Some(1) }),
    ];
    let batch = prepare_batch(&interrupted, 0, &[]).unwrap().unwrap();
    assert_eq!(batch.end_sequence, 2);
    assert_eq!(batch.source_ids, vec!["raw:1", "raw:2"]);

    let errored = vec![
        record(
            1,
            TranscriptEvent::UserMessage {
                content: "unfinished".into(),
            },
        ),
        record(
            2,
            TranscriptEvent::Error {
                message: "failed".into(),
            },
        ),
    ];
    let batch = prepare_batch(&errored, 0, &[]).unwrap().unwrap();
    assert_eq!(batch.end_sequence, 2);
    assert_eq!(batch.source_ids, vec!["raw:1", "raw:2"]);
}

#[test]
fn parser_accepts_empty_unknown_fields_and_deduplicates_sources() {
    let records = vec![
        record(
            1,
            TranscriptEvent::UserMessage {
                content: "decision".into(),
            },
        ),
        record(2, finalized()),
    ];
    let batch = prepare_batch(&records, 0, &[]).unwrap().unwrap();
    let update = parse_update(r#"{"memories":[],"withdrawn_ids":[]}"#, &batch, &[]).unwrap();
    assert!(update.memories.is_empty());
    assert!(update.withdrawn_ids.is_empty());
    let bad = r#"{"memories":[{"kind":"decision","title":"x","summary":"y","status":"useful","source_ids":["raw:99"],"paths":[],"supersedes":[]}],"withdrawn_ids":[]}"#;
    assert!(parse_update(bad, &batch, &[]).is_err());
}

#[test]
fn parser_ignores_unknown_fields_and_deduplicates_sources() {
    let records = vec![record(1, finalized())];
    let batch = prepare_batch(&records, 0, &[]).unwrap().unwrap();
    let known = [known_memory("memory-1")];
    let update = parse_update(
            r#"{"memories":[{"kind":"decision","title":"x","summary":"y","status":"useful","source_ids":["raw:1","raw:1"],"paths":[],"supersedes":["memory-1","memory-1"],"confidence":0.9}],"withdrawn_ids":[],"trace_id":"ignored"}"#,
            &batch,
            &known,
        )
        .unwrap();
    assert_eq!(update.memories[0].source_ids, ["raw:1"]);
    assert_eq!(update.memories[0].supersedes, ["memory-1"]);
}

#[test]
fn revisions_must_target_known_ids_once() {
    let records = vec![record(1, finalized())];
    let batch = prepare_batch(&records, 0, &[]).unwrap().unwrap();
    let known = [known_memory("memory-1"), known_memory("memory-2")];
    let supersedes_unknown = r#"{"memories":[{"kind":"decision","title":"x","summary":"y","status":"useful","source_ids":["raw:1"],"paths":[],"supersedes":["missing"]}],"withdrawn_ids":[]}"#;
    assert!(parse_update(supersedes_unknown, &batch, &known).is_err());
    let withdraws_unknown = r#"{"memories":[],"withdrawn_ids":["missing"]}"#;
    assert!(parse_update(withdraws_unknown, &batch, &known).is_err());
    let revises_twice = r#"{"memories":[{"kind":"decision","title":"x","summary":"y","status":"useful","source_ids":["raw:1"],"paths":[],"supersedes":["memory-1"]}],"withdrawn_ids":["memory-1"]}"#;
    assert!(parse_update(revises_twice, &batch, &known).is_err());
}

#[test]
fn long_unicode_input_is_bounded_without_splitting_characters() {
    let records = vec![
        record(
            1,
            TranscriptEvent::UserMessage {
                content: "记忆".repeat(100_000).into(),
            },
        ),
        record(2, finalized()),
    ];
    let batch = prepare_batch(&records, 0, &[]).unwrap().unwrap();
    assert!(batch.input.text.chars().count() <= MAX_BATCH_CHARS);
    assert!(batch.input.text.contains("truncated"));
}

#[test]
fn complete_turn_keeps_all_source_ids_when_fields_are_compressed() {
    let mut records = (1..=30)
        .map(|sequence| {
            record(
                sequence,
                TranscriptEvent::UserMessage {
                    content: "x".repeat(8_000).into(),
                },
            )
        })
        .collect::<Vec<_>>();
    records.push(record(31, finalized()));
    let batch = prepare_batch(&records, 0, &[]).unwrap().unwrap();
    assert_eq!(batch.source_ids.len(), 31);
    assert!(batch.input.text.chars().count() <= MAX_BATCH_CHARS);
}

#[test]
fn impossible_complete_coverage_fails_instead_of_dropping_records() {
    let mut records = (1..=2_000)
        .map(|sequence| {
            record(
                sequence,
                TranscriptEvent::UserMessage {
                    content: "x".into(),
                },
            )
        })
        .collect::<Vec<_>>();
    records.push(record(2_001, finalized()));
    let error = prepare_batch(&records, 0, &[]).unwrap_err().to_string();
    assert!(error.contains("complete source coverage"), "{error}");
}

#[test]
fn chinese_rewording_keeps_related_old_memories_as_candidates() {
    let mut related = known_memory("old-related");
    related.title = "缓存设计".into();
    related.summary = "缓存键必须稳定".into();
    let entries = [json!({"text": "新的缓存策略需要验证"})];
    assert!(memory_relevance(&related, &entries[0].to_string()) > 0);
}

#[test]
fn old_batch_related_memories_are_candidates_alongside_recent_memories() {
    let mut related = known_memory("old-related");
    related.title = "缓存设计".into();
    related.summary = "缓存键必须稳定".into();
    related.paths = vec!["src/cache.rs".into()];
    let mut known = (0..40)
        .map(|index| {
            let mut memory = known_memory(&format!("recent-{index}"));
            memory.created_at_ms = 100 + index;
            memory.title = format!("unrelated {index}");
            memory
        })
        .collect::<Vec<_>>();
    known.push(related);
    let records = vec![
        record(
            1,
            TranscriptEvent::UserMessage {
                content: "缓存策略见 src/cache.rs".into(),
            },
        ),
        record(2, finalized()),
    ];
    let batch = prepare_batch(&records, 0, &known).unwrap().unwrap();
    assert!(batch.input.text.contains("old-related"));
}
