use super::*;
use crate::config::ApiProtocol;
use crate::protocol_frames::{analyze_history_items, history_items_from_frames};
use crate::request_builder::{ModelRequestMetadata, RequestBuilderInput, build_request};
use crate::subagent::StructuredSubagentResult;
use serde_json::json;
use std::sync::{Arc, Mutex};

#[derive(Clone, Copy)]
enum FailPoint {
    Write,
    Flush,
    Sync,
}

struct FailingSink {
    fail: FailPoint,
    calls: Arc<Mutex<Vec<&'static str>>>,
}

impl JournalSink for FailingSink {
    fn write_all(&mut self, _: &[u8]) -> io::Result<()> {
        self.calls.lock().unwrap().push("write");
        if matches!(self.fail, FailPoint::Write) {
            Err(io::Error::other("injected write failure"))
        } else {
            Ok(())
        }
    }

    fn flush(&mut self) -> io::Result<()> {
        self.calls.lock().unwrap().push("flush");
        if matches!(self.fail, FailPoint::Flush) {
            Err(io::Error::other("injected flush failure"))
        } else {
            Ok(())
        }
    }

    fn sync_data(&mut self) -> io::Result<()> {
        self.calls.lock().unwrap().push("sync");
        if matches!(self.fail, FailPoint::Sync) {
            Err(io::Error::other("injected sync failure"))
        } else {
            Ok(())
        }
    }
}

fn recorder_with_sink(sink: impl JournalSink + 'static) -> TranscriptRecorder {
    TranscriptRecorder {
        session_id: "test-session".into(),
        path: PathBuf::from("unused.jsonl"),
        sink: Box::new(sink),
        sequence: 0,
        health: RecorderHealth::Healthy,
        current_context_branch_id: None,
        context_scope_state: Arc::new(Mutex::new(ContextScopeState::default())),
    }
}

fn journal_test_dir(name: &str) -> PathBuf {
    std::env::temp_dir().join(format!("letcode-journal-{name}-{}", unix_timestamp_ms()))
}

fn legacy_record(sequence: u64) -> TranscriptRecord {
    TranscriptRecord {
        session_id: "session".into(),
        sequence,
        timestamp_ms: sequence as u128,
        context_branch_id: None,
        event: TranscriptEvent::SessionTitle {
            title: format!("title-{sequence}"),
        },
    }
}

fn v1_record(sequence: u64) -> JournalRecordV1 {
    let record = legacy_record(sequence);
    JournalRecordV1 {
        schema_version: JOURNAL_SCHEMA_VERSION,
        event_id: format!("{}:{sequence}", record.session_id),
        scope: journal_scope_for(&record),
        base_revision: sequence - 1,
        resulting_revision: sequence,
        transaction_id: None,
        transaction_index: None,
        transaction_count: None,
        record,
    }
}

#[test]
fn journal_v1_round_trips_and_writes_envelope() {
    let base_dir = journal_test_dir("v1-roundtrip");
    let mut recorder = TranscriptRecorder::create(&base_dir).unwrap();
    recorder.record_user_message("hello").unwrap();
    let path = base_dir.join(format!("{}.jsonl", recorder.session_id()));
    let raw = fs::read_to_string(&path).unwrap();
    assert!(raw.contains("\"schema_version\":1"));
    assert!(raw.contains("\"scope\":\"global\""));
    assert!(raw.contains("\"base_revision\":0"));
    assert!(raw.contains("\"resulting_revision\":1"));
    let records = read_records(path).unwrap();
    assert_eq!(records.len(), 1);
    assert!(matches!(
        records[0].event,
        TranscriptEvent::UserMessage { .. }
    ));
}

#[test]
fn serialize_journal_record_matches_frozen_envelope_bytes() {
    struct Case {
        name: &'static str,
        record: JournalRecordV1,
        expected: &'static [u8],
    }

    let cases = [
        Case {
            name: "session title",
            record: JournalRecordV1 {
                schema_version: JOURNAL_SCHEMA_VERSION,
                event_id: "session:1".into(),
                scope: JournalScope::Global,
                base_revision: 0,
                resulting_revision: 1,
                transaction_id: None,
                transaction_index: None,
                transaction_count: None,
                record: TranscriptRecord {
                    session_id: "session".into(),
                    sequence: 1,
                    timestamp_ms: 0,
                    context_branch_id: None,
                    event: TranscriptEvent::SessionTitle {
                        title: "title".into(),
                    },
                },
            },
            expected: b"{\"schema_version\":1,\"event_id\":\"session:1\",\"scope\":\"global\",\"base_revision\":0,\"resulting_revision\":1,\"session_id\":\"session\",\"sequence\":1,\"timestamp_ms\":0,\"kind\":\"session_title\",\"title\":\"title\"}",
        },
        Case {
            name: "logical checkpoint transaction",
            record: JournalRecordV1 {
                schema_version: JOURNAL_SCHEMA_VERSION,
                event_id: "session:2".into(),
                scope: JournalScope::Branch,
                base_revision: 1,
                resulting_revision: 2,
                transaction_id: Some("transaction".into()),
                transaction_index: Some(0),
                transaction_count: Some(1),
                record: TranscriptRecord {
                    session_id: "session".into(),
                    sequence: 2,
                    timestamp_ms: 1,
                    context_branch_id: Some("branch".into()),
                    event: TranscriptEvent::LogicalCheckpoint(LogicalCheckpointEventV1 {
                        schema_version: 1,
                        checkpoint_id: "checkpoint".into(),
                        turn_id: 1,
                        previous_segment_id: 0,
                        segment_id: 1,
                        previous_checkpoint_id: None,
                        boundary_sequence: 1,
                        context_scope_revision: 1,
                        covered_source_spans: Vec::new(),
                        retained_items: Vec::new(),
                    }),
                },
            },
            expected: b"{\"base_revision\":1,\"boundary_sequence\":1,\"checkpoint_id\":\"checkpoint\",\"context_branch_id\":\"branch\",\"context_scope_revision\":1,\"covered_source_spans\":[],\"event_id\":\"session:2\",\"journal_schema_version\":1,\"kind\":\"logical_checkpoint\",\"previous_segment_id\":0,\"resulting_revision\":2,\"retained_items\":[],\"schema_version\":1,\"scope\":\"branch\",\"segment_id\":1,\"sequence\":2,\"session_id\":\"session\",\"timestamp_ms\":1,\"transaction_count\":1,\"transaction_id\":\"transaction\",\"transaction_index\":0,\"turn_id\":1}",
        },
    ];

    for case in cases {
        assert_eq!(
            serialize_journal_record(&case.record).expect("serialize journal record"),
            case.expected,
            "{}",
            case.name
        );
    }
}

#[test]
fn journal_reader_accepts_legacy_and_legacy_to_v1_records() {
    let base_dir = journal_test_dir("legacy");
    fs::create_dir_all(&base_dir).unwrap();
    let legacy_path = base_dir.join("legacy.jsonl");
    fs::write(
        &legacy_path,
        format!("{}\n", serde_json::to_string(&legacy_record(1)).unwrap()),
    )
    .unwrap();
    assert_eq!(read_records(&legacy_path).unwrap()[0].sequence, 1);

    let mixed_path = base_dir.join("mixed.jsonl");
    fs::write(
        &mixed_path,
        format!(
            "{}\n{}\n",
            serde_json::to_string(&legacy_record(1)).unwrap(),
            serde_json::to_string(&v1_record(2)).unwrap()
        ),
    )
    .unwrap();
    let records = read_records(&mixed_path).unwrap();
    assert_eq!(
        records
            .iter()
            .map(|record| record.sequence)
            .collect::<Vec<_>>(),
        [1, 2]
    );
}

#[test]
fn journal_reader_rejects_invalid_contracts() {
    let base_dir = journal_test_dir("invalid");
    fs::create_dir_all(&base_dir).unwrap();
    let cases = [
        ("duplicate-event", {
            let first = v1_record(1);
            let mut second = v1_record(2);
            second.event_id = first.event_id.clone();
            vec![first, second]
        }),
        ("revision", {
            let first = v1_record(1);
            let mut second = v1_record(2);
            second.base_revision = 0;
            vec![first, second]
        }),
        ("sequence", {
            let first = v1_record(1);
            let second = v1_record(1);
            vec![first, second]
        }),
        ("session", {
            let first = v1_record(1);
            let mut second = v1_record(2);
            second.record.session_id = "other".into();
            vec![first, second]
        }),
        ("scope", {
            let first = v1_record(1);
            let mut second = v1_record(2);
            second.scope = JournalScope::Branch;
            vec![first, second]
        }),
    ];
    for (name, records) in cases {
        let path = base_dir.join(format!("{name}.jsonl"));
        fs::write(
            &path,
            records
                .iter()
                .map(|record| serde_json::to_string(record).unwrap())
                .collect::<Vec<_>>()
                .join("\n")
                + "\n",
        )
        .unwrap();
        assert!(read_records(path).is_err(), "{name} must be rejected");
    }
}

#[test]
fn journal_reader_rejects_v1_with_nonzero_initial_base_revision() {
    let base_dir = journal_test_dir("v1-initial-base");
    fs::create_dir_all(&base_dir).unwrap();
    let path = base_dir.join("invalid.jsonl");
    let record = v1_record(2);
    fs::write(
        &path,
        format!("{}\n", serde_json::to_string(&record).unwrap()),
    )
    .unwrap();

    assert!(read_records(path).is_err());
}

#[test]
fn journal_reader_rejects_v1_with_forged_event_id() {
    let base_dir = journal_test_dir("v1-event-id");
    fs::create_dir_all(&base_dir).unwrap();
    let path = base_dir.join("invalid.jsonl");
    let mut record = v1_record(1);
    record.event_id = "forged:1".into();
    fs::write(
        &path,
        format!("{}\n", serde_json::to_string(&record).unwrap()),
    )
    .unwrap();

    assert!(read_records(path).is_err());
}

#[test]
fn journal_io_failures_poison_recorder_without_advancing_sequence() {
    for fail in [FailPoint::Write, FailPoint::Flush, FailPoint::Sync] {
        let calls = Arc::new(Mutex::new(Vec::new()));
        let mut recorder = recorder_with_sink(FailingSink {
            fail,
            calls: Arc::clone(&calls),
        });
        assert!(recorder.record_user_message("first").is_err());
        assert_eq!(recorder.health, RecorderHealth::Poisoned);
        assert_eq!(recorder.sequence, 0);
        assert!(recorder.record_user_message("second").is_err());
        assert_eq!(recorder.sequence, 0);
        let call_count = calls.lock().unwrap().len();
        assert_eq!(
            call_count,
            match fail {
                FailPoint::Write => 1,
                FailPoint::Flush => 2,
                FailPoint::Sync => 3,
            }
        );
    }
}

#[test]
fn transaction_round_trip_commits_all_records_and_uncommitted_tail_is_ignored() {
    let base_dir = journal_test_dir("transaction-tail");
    let mut recorder = TranscriptRecorder::create(&base_dir).unwrap();
    recorder
        .append_transaction(vec![
            (
                TranscriptEvent::SessionTitle {
                    title: "first".into(),
                },
                None,
            ),
            (
                TranscriptEvent::AssistantMessage {
                    content: "second".into(),
                },
                Some("branch-a".into()),
            ),
        ])
        .unwrap();
    let path = base_dir.join(format!("{}.jsonl", recorder.session_id()));
    let records = read_records(&path).unwrap();
    assert_eq!(records.len(), 2);
    assert_eq!(records[0].sequence, 1);
    assert_eq!(records[1].sequence, 2);

    let lines = fs::read_to_string(&path)
        .unwrap()
        .lines()
        .map(str::to_owned)
        .collect::<Vec<_>>();
    let mut corrupt_commit: Value = serde_json::from_str(lines.last().unwrap()).unwrap();
    corrupt_commit["payload_digest"] = Value::String("wrong".into());
    let mut corrupt_lines = lines.clone();
    *corrupt_lines.last_mut().unwrap() = serde_json::to_string(&corrupt_commit).unwrap();
    fs::write(&path, corrupt_lines.join("\n") + "\n").unwrap();
    assert!(read_records(&path).is_err());

    let mut lines = lines;
    lines.pop(); // Remove only the private transaction commit marker.
    fs::write(&path, lines.join("\n") + "\n").unwrap();
    assert!(read_records(&path).unwrap().is_empty());
    fs::OpenOptions::new()
        .append(true)
        .open(&path)
        .unwrap()
        .write_all(format!("{}\n", serde_json::to_string(&legacy_record(3)).unwrap()).as_bytes())
        .unwrap();
    assert!(read_records(&path).is_err());
    fs::write(&path, lines.join("\n") + "\n").unwrap();
    assert!(TranscriptRecorder::open_existing(&base_dir, recorder.session_id()).is_err());
}

#[test]
fn journal_reader_rejects_transaction_commit_with_mismatched_resulting_revision() {
    let base_dir = journal_test_dir("transaction-resulting-revision");
    let mut recorder = TranscriptRecorder::create(&base_dir).unwrap();
    recorder
        .append_transaction(vec![
            (
                TranscriptEvent::SessionTitle {
                    title: "first".into(),
                },
                None,
            ),
            (
                TranscriptEvent::AssistantMessage {
                    content: "second".into(),
                },
                None,
            ),
        ])
        .unwrap();
    let path = base_dir.join(format!("{}.jsonl", recorder.session_id()));
    let mut lines = fs::read_to_string(&path)
        .unwrap()
        .lines()
        .map(str::to_owned)
        .collect::<Vec<_>>();
    let mut commit: Value = serde_json::from_str(lines.last().unwrap()).unwrap();
    commit["resulting_revision"] = Value::from(1);
    *lines.last_mut().unwrap() = serde_json::to_string(&commit).unwrap();
    fs::write(&path, lines.join("\n") + "\n").unwrap();

    assert!(read_records(path).is_err());
}

#[test]
fn transaction_io_failure_poison_does_not_advance_or_switch_scope() {
    for fail in [FailPoint::Write, FailPoint::Flush, FailPoint::Sync] {
        let calls = Arc::new(Mutex::new(Vec::new()));
        let mut recorder = recorder_with_sink(FailingSink {
            fail,
            calls: Arc::clone(&calls),
        });
        recorder.set_current_context_branch_id(Some("parent".into()));
        assert!(
            recorder
                .append_transaction(vec![(
                    TranscriptEvent::AssistantMessage {
                        content: "atomic".into(),
                    },
                    Some("child".into()),
                )])
                .is_err()
        );
        assert_eq!(recorder.sequence, 0);
        assert_eq!(recorder.current_context_branch_id(), Some("parent"));
        assert_eq!(recorder.health, RecorderHealth::Poisoned);
        assert_eq!(
            *calls.lock().unwrap(),
            match fail {
                FailPoint::Write => vec!["write"],
                FailPoint::Flush => vec!["write", "flush"],
                FailPoint::Sync => vec!["write", "flush", "sync"],
            }
        );
    }
}

#[test]
fn records_model_and_permission_mode_changes_with_expected_shape() {
    let base_dir = std::env::temp_dir().join(format!(
        "letcode-transcript-provenance-test-{}",
        unix_timestamp_ms()
    ));
    let mut recorder = TranscriptRecorder::create(&base_dir).expect("create recorder");

    recorder
        .record_model_changed("gpt-5.5", "gpt-5.5-mini")
        .expect("record model change");
    recorder
        .record_permission_mode_changed("default", "safe")
        .expect("record permission change");

    let records = read_records(base_dir.join(format!("{}.jsonl", recorder.session_id())))
        .expect("read records");

    assert_eq!(records.len(), 2);

    let first = serde_json::to_value(&records[0]).expect("serialize");
    assert_eq!(first.get("kind"), Some(&json!("model_changed")));
    assert_eq!(first.get("previous_model"), Some(&json!("gpt-5.5")));
    assert_eq!(first.get("new_model"), Some(&json!("gpt-5.5-mini")));

    let second = serde_json::to_value(&records[1]).expect("serialize");
    assert_eq!(second.get("kind"), Some(&json!("permission_mode_changed")));
    assert_eq!(second.get("previous_mode"), Some(&json!("default")));
    assert_eq!(second.get("new_mode"), Some(&json!("safe")));
}

#[test]
fn restore_latest_model_replays_session_start_and_model_changes() {
    let records = vec![
        TranscriptRecord {
            session_id: "s".into(),
            sequence: 1,
            timestamp_ms: 0,
            context_branch_id: None,
            event: TranscriptEvent::SessionStarted { model: "m1".into() },
        },
        TranscriptRecord {
            session_id: "s".into(),
            sequence: 2,
            timestamp_ms: 1,
            context_branch_id: None,
            event: TranscriptEvent::ModelChanged {
                previous_model: "m1".into(),
                new_model: "m2".into(),
            },
        },
        TranscriptRecord {
            session_id: "s".into(),
            sequence: 3,
            timestamp_ms: 2,
            context_branch_id: None,
            event: TranscriptEvent::ModelChanged {
                previous_model: "m2".into(),
                new_model: "m3".into(),
            },
        },
    ];

    assert_eq!(restore_latest_model(&records).as_deref(), Some("m3"));
}

#[test]
fn restore_conversation_messages_ignores_provenance_events() {
    let records = vec![
        TranscriptRecord {
            session_id: "s".into(),
            sequence: 1,
            timestamp_ms: 0,
            context_branch_id: None,
            event: TranscriptEvent::UserMessage {
                content: "hi".into(),
            },
        },
        TranscriptRecord {
            session_id: "s".into(),
            sequence: 2,
            timestamp_ms: 1,
            context_branch_id: None,
            event: TranscriptEvent::TurnStarted(TurnStartedEvent {
                turn_id: 1,
                intent: "engineering".into(),
                directive: "none".into(),
                validation_reminder: "focused".into(),
            }),
        },
        TranscriptRecord {
            session_id: "s".into(),
            sequence: 3,
            timestamp_ms: 2,
            context_branch_id: None,
            event: TranscriptEvent::SubagentLifecycle {
                run_id: "sub-1".into(),
                parent_session_id: "s".into(),
                parent_run_id: "turn-1".into(),
                agent_name: "explorer".into(),
                status: "running".into(),
                detail: None,
            },
        },
        TranscriptRecord {
            session_id: "s".into(),
            sequence: 4,
            timestamp_ms: 3,
            context_branch_id: None,
            event: TranscriptEvent::ModelChanged {
                previous_model: "a".into(),
                new_model: "b".into(),
            },
        },
        TranscriptRecord {
            session_id: "s".into(),
            sequence: 5,
            timestamp_ms: 4,
            context_branch_id: None,
            event: TranscriptEvent::AssistantMessage {
                content: "hello".into(),
            },
        },
        TranscriptRecord {
            session_id: "s".into(),
            sequence: 6,
            timestamp_ms: 5,
            context_branch_id: None,
            event: TranscriptEvent::PermissionModeChanged {
                previous_mode: "default".into(),
                new_mode: "safe".into(),
            },
        },
        TranscriptRecord {
            session_id: "s".into(),
            sequence: 7,
            timestamp_ms: 6,
            context_branch_id: None,
            event: TranscriptEvent::AutoContinuationScheduled {
                continuation_count: 1,
                remaining_unfinished: 2,
            },
        },
        TranscriptRecord {
            session_id: "s".into(),
            sequence: 8,
            timestamp_ms: 7,
            context_branch_id: None,
            event: TranscriptEvent::ValidationAdvisory(ValidationAdvisory {
                write_effects: 1,
                validation_effects: 0,
                failed_validation_effects: 0,
                message: "validation reminder".into(),
            }),
        },
        TranscriptRecord {
            session_id: "s".into(),
            sequence: 9,
            timestamp_ms: 8,
            context_branch_id: None,
            event: TranscriptEvent::ToolExecutionSummary(ToolExecutionSummaryEvent {
                turn_id: 1,
                call_id: "call-1".into(),
                name: "fs__write".into(),
                status: "executed".into(),
                rejection: None,
                effect_kind: "write".into(),
                primary_path: Some("src/main.rs".into()),
                command: None,
            }),
        },
        TranscriptRecord {
            session_id: "s".into(),
            sequence: 10,
            timestamp_ms: 9,
            context_branch_id: None,
            event: TranscriptEvent::TurnFinalized(TurnFinalizedEvent {
                turn_id: 1,
                outcome: "completed".into(),
                tool_call_count: 1,
                continuation_count: 0,
                write_effects: 1,
                validation_effects: 0,
                failed_validation_effects: 0,
                validation_advisory_emitted: true,
            }),
        },
    ];

    let restored = restore_conversation_messages(&records).expect("restore messages");
    assert_eq!(restored.len(), 2);
    assert!(matches!(restored[0].role, ConversationRole::User));
    assert_eq!(restored[0].content, "hi");
    assert!(matches!(restored[1].role, ConversationRole::Assistant));
    assert_eq!(restored[1].content, "hello");
}

#[test]
fn restore_session_history_uses_latest_compaction_view() {
    let records = vec![
        TranscriptRecord {
            session_id: "s".into(),
            sequence: 1,
            timestamp_ms: 0,
            context_branch_id: None,
            event: TranscriptEvent::UserMessage {
                content: "old user".into(),
            },
        },
        TranscriptRecord {
            session_id: "s".into(),
            sequence: 2,
            timestamp_ms: 1,
            context_branch_id: None,
            event: TranscriptEvent::UserMessage {
                content: "tail user".into(),
            },
        },
        TranscriptRecord {
            session_id: "s".into(),
            sequence: 3,
            timestamp_ms: 1,
            context_branch_id: None,
            event: TranscriptEvent::AssistantMessage {
                content: "tail assistant".into(),
            },
        },
        TranscriptRecord {
            session_id: "s".into(),
            sequence: 4,
            timestamp_ms: 2,
            context_branch_id: None,
            event: TranscriptEvent::ContextCompaction(ContextCompactionEvent {
                outcome: "succeeded".into(),
                summary: "目标\n- 保留摘要".into(),
                tail_start_index: 1,
                original_history_items: 3,
                retained_history_items: 3,
                retired_source_spans: Vec::new(),
                frame_identity_bindings: Vec::new(),
                detail: None,
            }),
        },
        TranscriptRecord {
            session_id: "s".into(),
            sequence: 5,
            timestamp_ms: 3,
            context_branch_id: None,
            event: TranscriptEvent::AssistantMessage {
                content: "new assistant".into(),
            },
        },
    ];

    let history = restore_session_history(&records).expect("restore history");
    assert!(matches!(history[0], HistoryItem::ContextSummary { .. }));
    assert!(matches!(history[1], HistoryItem::UserMessage { .. }));
    assert!(matches!(history[2], HistoryItem::AssistantText { .. }));
    assert!(matches!(history[3], HistoryItem::AssistantText { .. }));

    let messages = restore_compacted_conversation_messages(&records).expect("restore messages");
    assert!(matches!(messages[0].role, ConversationRole::Summary));
    assert_eq!(messages[1].content, "tail user");
    assert_eq!(messages[2].content, "tail assistant");
    assert_eq!(messages[3].content, "new assistant");

    let compaction = serde_json::to_value(&records[3]).expect("serialize compaction");
    assert_eq!(compaction["original_history_items"], json!(3));
    assert_eq!(compaction["retained_history_items"], json!(3));
    assert!(
        compaction
            .get("original_history_items")
            .unwrap()
            .is_number()
    );
    assert!(
        compaction
            .get("retained_history_items")
            .unwrap()
            .is_number()
    );
    let compaction_text = serde_json::to_string(&records[3]).expect("serialize compaction text");
    assert!(!compaction_text.contains("old user"));
    assert!(!compaction_text.contains("tail assistant"));
}

#[test]
fn restore_session_history_preserves_tool_calls_permission_decisions_and_cancelled_tools() {
    let records = vec![
        TranscriptRecord {
            session_id: "s".into(),
            sequence: 1,
            timestamp_ms: 0,
            context_branch_id: None,
            event: TranscriptEvent::ToolCallStarted {
                call_id: "call-1".into(),
                name: "shell__exec".into(),
                args: json!({"command": "cargo test"}),
            },
        },
        TranscriptRecord {
            session_id: "s".into(),
            sequence: 2,
            timestamp_ms: 1,
            context_branch_id: None,
            event: TranscriptEvent::PermissionDecision {
                call_id: Some("call-1".into()),
                tool: "shell__exec".into(),
                args: json!({"command": "cargo test"}),
                allowed: false,
                reason: Some("Denied by user from TUI permission prompt".into()),
            },
        },
        TranscriptRecord {
            session_id: "s".into(),
            sequence: 3,
            timestamp_ms: 2,
            context_branch_id: None,
            event: TranscriptEvent::ToolCallCancelled {
                call_id: "call-1".into(),
                name: "shell__exec".into(),
            },
        },
    ];

    let history = restore_session_history(&records).expect("restore history");
    assert!(matches!(
        history.first(),
        Some(HistoryItem::AssistantToolCalls { calls, .. })
            if calls.len() == 1 && calls[0].call_id == "call-1"
    ));
    assert!(matches!(
        history.get(1),
        Some(HistoryItem::ToolOutput {
            call_id,
            output_json,
        }) if call_id == "call-1"
            && output_json == r#"{"status":"cancelled","summary":"user cancelled"}"#
    ));
}

#[test]
fn records_validation_advisory_with_expected_shape() {
    let base_dir = std::env::temp_dir().join(format!(
        "letcode-transcript-validation-advisory-test-{}",
        unix_timestamp_ms()
    ));
    let mut recorder = TranscriptRecorder::create(&base_dir).expect("create recorder");

    recorder
        .record_validation_advisory(ValidationAdvisory {
            write_effects: 2,
            validation_effects: 0,
            failed_validation_effects: 1,
            message: "validation reminder".into(),
        })
        .expect("record validation advisory");

    let records = read_records(base_dir.join(format!("{}.jsonl", recorder.session_id())))
        .expect("read records");

    assert_eq!(records.len(), 1);
    let record = serde_json::to_value(&records[0]).expect("serialize");
    assert_eq!(record.get("kind"), Some(&json!("validation_advisory")));
    assert_eq!(record.get("write_effects"), Some(&json!(2)));
    assert_eq!(record.get("validation_effects"), Some(&json!(0)));
    assert_eq!(record.get("failed_validation_effects"), Some(&json!(1)));
    assert_eq!(record.get("message"), Some(&json!("validation reminder")));
}

#[test]
fn records_turn_lifecycle_and_tool_summary_with_expected_shape() {
    let base_dir = std::env::temp_dir().join(format!(
        "letcode-transcript-turn-audit-test-{}",
        unix_timestamp_ms()
    ));
    let mut recorder = TranscriptRecorder::create(&base_dir).expect("create recorder");

    recorder
        .record_turn_started(TurnStartedEvent {
            turn_id: 7,
            intent: "engineering".into(),
            directive: "none".into(),
            validation_reminder: "targeted".into(),
        })
        .expect("record turn started");
    recorder
        .record_tool_execution_summary(ToolExecutionSummaryEvent {
            turn_id: 7,
            call_id: "call-1".into(),
            name: "shell__exec".into(),
            status: "executed".into(),
            rejection: None,
            effect_kind: "validation".into(),
            primary_path: Some("src/agent.rs".into()),
            command: Some("cargo test transcript".into()),
        })
        .expect("record tool summary");
    recorder
        .record_turn_finalized(TurnFinalizedEvent {
            turn_id: 7,
            outcome: "completed".into(),
            tool_call_count: 3,
            continuation_count: 1,
            write_effects: 1,
            validation_effects: 1,
            failed_validation_effects: 0,
            validation_advisory_emitted: false,
        })
        .expect("record turn finalized");

    let records = read_records(base_dir.join(format!("{}.jsonl", recorder.session_id())))
        .expect("read records");
    assert_eq!(records.len(), 3);

    let started = serde_json::to_value(&records[0]).expect("serialize");
    assert_eq!(started.get("kind"), Some(&json!("turn_started")));
    assert_eq!(started.get("turn_id"), Some(&json!(7)));
    assert_eq!(started.get("intent"), Some(&json!("engineering")));

    let summary = serde_json::to_value(&records[1]).expect("serialize");
    assert_eq!(summary.get("kind"), Some(&json!("tool_execution_summary")));
    assert_eq!(summary.get("call_id"), Some(&json!("call-1")));
    assert!(summary.get("output").is_none());

    let finalized = serde_json::to_value(&records[2]).expect("serialize");
    assert_eq!(finalized.get("kind"), Some(&json!("turn_finalized")));
    assert_eq!(finalized.get("turn_id"), Some(&json!(7)));
    assert_eq!(finalized.get("outcome"), Some(&json!("completed")));
    assert_eq!(
        finalized.get("validation_advisory_emitted"),
        Some(&json!(false))
    );
}

#[test]
fn failed_compaction_is_recorded_as_error_without_rewriting_history() {
    let base_dir = std::env::temp_dir().join(format!(
        "letcode-transcript-compaction-failure-test-{}",
        unix_timestamp_ms()
    ));
    let mut recorder = TranscriptRecorder::create(&base_dir).expect("create recorder");

    recorder
        .record_context_compaction(ContextCompactionEvent {
            outcome: "failed".into(),
            summary: String::new(),
            tail_start_index: 0,
            original_history_items: 3,
            retained_history_items: 3,
            retired_source_spans: Vec::new(),
            frame_identity_bindings: Vec::new(),
            detail: Some("summary model returned empty output".into()),
        })
        .expect("record failed compaction");

    let records = read_records(base_dir.join(format!("{}.jsonl", recorder.session_id())))
        .expect("read records");
    assert_eq!(records.len(), 1);
    let value = serde_json::to_value(&records[0]).expect("serialize");
    assert_eq!(value.get("kind"), Some(&json!("error")));
    assert_eq!(
        value.get("message"),
        Some(&json!(
            "context compaction failed: summary model returned empty output"
        ))
    );
}

#[test]
fn compaction_event_deserializes_without_retired_source_spans() {
    let event: ContextCompactionEvent = serde_json::from_value(json!({
        "outcome": "succeeded",
        "summary": "summary",
        "tail_start_index": 1,
        "original_history_items": 3,
        "retained_history_items": 2
    }))
    .expect("legacy compaction event deserializes");

    assert!(event.retired_source_spans.is_empty());
}

#[test]
fn prepared_telemetry_v1_deserializes_without_evidence_selection_fields() {
    let event: TranscriptEvent = serde_json::from_value(json!({
        "kind": "llm_request_telemetry",
        "version": 1,
        "logical_request_id": "turn-1-iteration-0",
        "turn_id": 1,
        "iteration": 0,
        "attempt": 1,
        "phase": "prepared",
        "model": "test-model",
        "protocol": "responses",
        "context_window_tokens": 8192,
        "input_budget_tokens": 7000,
        "estimated_request_tokens": 100,
        "estimated_prelude_tokens": 10,
        "estimated_protected_tokens": 20,
        "estimated_retained_history_tokens": 30,
        "estimated_tools_tokens": 0,
        "estimated_evidence_tokens": 0,
        "estimated_required_fallback_tokens": 0,
        "original_history_items": 1,
        "retained_history_items": 1,
        "dropped_history_items": 0,
        "selected_evidence_items": 0,
        "dropped_evidence_items": 0,
        "truncated": false,
        "prompt_segment_count": 1,
        "prompt_contributor_count": 1,
        "plan_total_prompt_tokens": 100,
        "plan_stable_prompt_tokens": 100,
        "plan_volatile_prompt_tokens": 0,
        "plan_cacheable_prefix_tokens": 100,
        "plan_stable_after_boundary_tokens": 0,
        "cache_configured": false,
        "cache_hint_serialized": false,
        "cache_stable_prefix_segments": 0,
        "cache_stable_prompt_tokens": 0,
        "cache_volatile_prompt_tokens": 0,
        "cacheable_prefix_tokens": 0,
        "cache_stable_after_boundary_tokens": 0,
        "tool_call_count_before": 0,
        "tool_definitions_count": 0
    }))
    .expect("v1 prepared telemetry deserializes");

    let TranscriptEvent::LlmRequestTelemetry {
        version,
        selected_evidence_ids,
        evidence_fingerprint,
        protected_safe_ceiling_tokens,
        protected_reserve_tokens,
        estimated_foldable_protected_tokens,
        estimated_provider_folded_protected_tokens,
        estimated_unaddressable_protected_tokens,
        provider_folded_output_count,
        usage_completeness,
        ..
    } = event
    else {
        panic!("prepared telemetry event")
    };
    assert_eq!(version, 1);
    assert!(selected_evidence_ids.is_empty());
    assert!(evidence_fingerprint.is_empty());
    assert_eq!(protected_safe_ceiling_tokens, 0);
    assert_eq!(protected_reserve_tokens, 0);
    assert_eq!(estimated_foldable_protected_tokens, 0);
    assert_eq!(estimated_provider_folded_protected_tokens, 0);
    assert_eq!(estimated_unaddressable_protected_tokens, 0);
    assert_eq!(provider_folded_output_count, 0);
    assert_eq!(usage_completeness, "legacy_unknown");
}

#[test]
fn old_layout_telemetry_fields_are_ignored_without_affecting_restore() {
    let base_dir = journal_test_dir("legacy-layout-telemetry");
    fs::create_dir_all(&base_dir).expect("create temp dir");
    let path = base_dir.join("legacy-layout-telemetry.jsonl");
    fs::write(
        &path,
        r#"{"session_id":"s","sequence":1,"timestamp_ms":0,"kind":"user_message","content":"keep this request"}
{"session_id":"s","sequence":2,"timestamp_ms":1,"kind":"llm_request_telemetry","version":5,"logical_request_id":"turn-1-iteration-0","turn_id":1,"iteration":0,"attempt":1,"phase":"prepared","model":"test-model","protocol":"responses","context_window_tokens":8192,"input_budget_tokens":7000,"estimated_request_tokens":100,"selected_layout":"v1","alternate_layout":"v2","alternate_estimated_request_tokens":101,"estimated_prelude_tokens":10,"estimated_protected_tokens":20,"estimated_retained_history_tokens":30,"estimated_tools_tokens":0,"estimated_evidence_tokens":0,"estimated_required_fallback_tokens":0,"original_history_items":1,"retained_history_items":1,"dropped_history_items":0,"selected_evidence_items":0,"dropped_evidence_items":0,"truncated":false,"prompt_segment_count":1,"prompt_contributor_count":1,"plan_total_prompt_tokens":100,"plan_stable_prompt_tokens":100,"plan_volatile_prompt_tokens":0,"plan_cacheable_prefix_tokens":100,"plan_stable_after_boundary_tokens":0,"cache_configured":false,"cache_hint_serialized":false,"cache_stable_prefix_segments":0,"cache_stable_prompt_tokens":0,"cache_volatile_prompt_tokens":0,"cacheable_prefix_tokens":0,"cache_stable_after_boundary_tokens":0,"tool_call_count_before":0,"tool_definitions_count":0}
{"session_id":"s","sequence":3,"timestamp_ms":2,"kind":"assistant_message","content":"keep this response"}
"#,
    )
    .expect("write legacy transcript");

    let records = read_records(&path).expect("read legacy layout telemetry");
    assert_eq!(records.len(), 3);
    assert!(matches!(
        records[1].event,
        TranscriptEvent::LlmRequestTelemetry { version: 5, .. }
    ));

    let history = restore_session_history(&records).expect("restore semantic history");
    assert_eq!(history.len(), 2);
    let messages = restore_conversation_messages(&records).expect("restore semantic messages");
    assert_eq!(
        messages
            .iter()
            .map(|message| message.content.as_str())
            .collect::<Vec<_>>(),
        ["keep this request", "keep this response"]
    );
    let snapshot = restore_runtime_snapshot(&records).expect("restore runtime snapshot");
    assert!(!snapshot.frames.is_empty());
}

#[test]
fn record_context_compaction_populates_retired_source_spans_when_missing() {
    let base_dir = std::env::temp_dir().join(format!(
        "letcode-transcript-compaction-span-test-{}",
        unix_timestamp_ms()
    ));
    let mut recorder = TranscriptRecorder::create(&base_dir).expect("create recorder");
    recorder
        .append(TranscriptEvent::UserMessage {
            content: UserMessageContent::from("old user"),
        })
        .expect("record user");
    recorder
        .append(TranscriptEvent::AssistantMessage {
            content: "tail note".into(),
        })
        .expect("record assistant");

    recorder
        .record_context_compaction(ContextCompactionEvent {
            outcome: "succeeded".into(),
            summary: "summary".into(),
            tail_start_index: 1,
            original_history_items: 2,
            retained_history_items: 2,
            retired_source_spans: Vec::new(),
            frame_identity_bindings: Vec::new(),
            detail: None,
        })
        .expect("record compaction");

    let records = read_records(base_dir.join(format!("{}.jsonl", recorder.session_id())))
        .expect("read records");
    let event = records
        .iter()
        .find_map(|record| match &record.event {
            TranscriptEvent::ContextCompaction(event) => Some(event),
            _ => None,
        })
        .expect("compaction event present");
    assert_eq!(event.retired_source_spans.len(), 1);
    assert_eq!(event.retired_source_spans[0].start_sequence, 1);
    assert_eq!(event.retired_source_spans[0].end_sequence, 1);
}

#[test]
fn write_summary_still_restores_legacy_write_observed_state() {
    let records = vec![
        TranscriptRecord {
            session_id: "s".into(),
            sequence: 1,
            timestamp_ms: 0,
            context_branch_id: None,
            event: TranscriptEvent::ContextExperimentStarted {
                branch_id: "branch-1".into(),
                parent_branch_id: ROOT_CONTEXT_BRANCH_ID.into(),
                base_sequence: 4,
            },
        },
        TranscriptRecord {
            session_id: "s".into(),
            sequence: 2,
            timestamp_ms: 1,
            context_branch_id: Some("branch-1".into()),
            event: TranscriptEvent::ToolExecutionSummary(ToolExecutionSummaryEvent {
                turn_id: 1,
                call_id: "call-write".into(),
                name: "fs__write".into(),
                status: "executed".into(),
                rejection: None,
                effect_kind: "write".into(),
                primary_path: Some("src/lib.rs".into()),
                command: None,
            }),
        },
    ];

    let state = reconstruct_context_scope_state(&records).expect("reconstruct state");
    assert!(
        state
            .active_experiment
            .as_ref()
            .is_some_and(|experiment| experiment.writes_observed)
    );
}

#[test]
fn records_tool_cancellation_and_turn_interruption() {
    let base_dir = std::env::temp_dir().join(format!(
        "letcode-transcript-interrupt-test-{}",
        unix_timestamp_ms()
    ));
    let mut recorder = TranscriptRecorder::create(&base_dir).expect("create recorder");

    recorder
        .record_tool_call_cancelled("call-1", "shell__exec")
        .expect("record tool cancellation");
    recorder
        .record_turn_interrupted(Some(7))
        .expect("record turn interruption");

    let records = read_records(base_dir.join(format!("{}.jsonl", recorder.session_id())))
        .expect("read records");
    assert_eq!(records.len(), 2);

    let cancelled = serde_json::to_value(&records[0]).expect("serialize cancelled");
    assert_eq!(cancelled.get("kind"), Some(&json!("tool_call_cancelled")));
    assert_eq!(cancelled.get("call_id"), Some(&json!("call-1")));

    let interrupted = serde_json::to_value(&records[1]).expect("serialize interrupted");
    assert_eq!(interrupted.get("kind"), Some(&json!("turn_interrupted")));
    assert_eq!(interrupted.get("turn_id"), Some(&json!(7)));
}

#[test]
fn restore_max_turn_id_includes_turn_interrupted_events() {
    let records = vec![TranscriptRecord {
        session_id: "s".into(),
        sequence: 1,
        timestamp_ms: 0,
        context_branch_id: None,
        event: TranscriptEvent::TurnInterrupted { turn_id: Some(9) },
    }];

    assert_eq!(restore_max_turn_id(&records), 9);
}

#[test]
fn restore_session_history_closes_dangling_user_turn_on_interrupt() {
    let records = vec![
        TranscriptRecord {
            session_id: "s".into(),
            sequence: 1,
            timestamp_ms: 0,
            context_branch_id: None,
            event: TranscriptEvent::UserMessage {
                content: "unfinished".into(),
            },
        },
        TranscriptRecord {
            session_id: "s".into(),
            sequence: 2,
            timestamp_ms: 1,
            context_branch_id: None,
            event: TranscriptEvent::TurnInterrupted { turn_id: Some(1) },
        },
    ];

    let history = restore_session_history(&records).expect("restore history");
    assert!(matches!(
        history.as_slice(),
        [HistoryItem::UserMessage { content }, HistoryItem::AssistantText { text: assistant_text }]
            if content.text == "unfinished" && assistant_text.is_empty()
    ));

    let messages = restore_conversation_messages(&records).expect("restore messages");
    assert_eq!(messages.len(), 2);
    assert!(matches!(messages[0].role, ConversationRole::User));
    assert!(matches!(messages[1].role, ConversationRole::Assistant));
    assert!(messages[1].content.is_empty());
}

#[test]
fn restore_session_history_closes_interrupted_turn_after_tool_output() {
    let records = vec![
        TranscriptRecord {
            session_id: "s".into(),
            sequence: 1,
            timestamp_ms: 0,
            context_branch_id: None,
            event: TranscriptEvent::UserMessage {
                content: "run it".into(),
            },
        },
        TranscriptRecord {
            session_id: "s".into(),
            sequence: 2,
            timestamp_ms: 1,
            context_branch_id: None,
            event: TranscriptEvent::ToolCallStarted {
                call_id: "call-1".into(),
                name: "shell__exec".into(),
                args: json!({"command": "sleep 10"}),
            },
        },
        TranscriptRecord {
            session_id: "s".into(),
            sequence: 3,
            timestamp_ms: 2,
            context_branch_id: None,
            event: TranscriptEvent::ToolCallFinished {
                call_id: "call-1".into(),
                name: "shell__exec".into(),
                ok: true,
                output: ToolResult::ok("shell__exec", json!({"stdout": "started"})),
            },
        },
        TranscriptRecord {
            session_id: "s".into(),
            sequence: 4,
            timestamp_ms: 3,
            context_branch_id: None,
            event: TranscriptEvent::TurnInterrupted { turn_id: Some(1) },
        },
    ];

    let messages = restore_conversation_messages(&records).expect("restore messages");
    assert_eq!(messages.len(), 2);
    assert!(matches!(messages[0].role, ConversationRole::User));
    assert!(matches!(messages[1].role, ConversationRole::Assistant));
    assert_eq!(messages[0].content, "run it");
    assert!(messages[1].content.is_empty());
}

#[test]
fn evidence_records_round_trip_and_restore_from_transcript() {
    let base_dir = std::env::temp_dir().join(format!(
        "letcode-transcript-evidence-test-{}",
        unix_timestamp_ms()
    ));
    let mut recorder = TranscriptRecorder::create(&base_dir).expect("create recorder");
    let draft = EvidenceDraft {
        id: Some("ev-test".into()),
        evidence_kind: EvidenceKind::FileExcerpt,
        title: "read config".into(),
        summary: "config has active provider".into(),
        detail: Some("active_provider = openai".into()),
        source: EvidenceSource::File {
            path: "letcode.toml".into(),
            start_line: Some(1),
            end_line: Some(1),
        },
        tags: vec!["letcode.toml".into()],
    };

    let record = recorder.record_evidence(draft).expect("record evidence");
    assert_eq!(record.id, "ev-test");

    let records = read_records(base_dir.join(format!("{}.jsonl", recorder.session_id())))
        .expect("read records");
    let evidence = restore_session_evidence(&records).expect("restore evidence");
    assert_eq!(evidence.len(), 1);
    assert_eq!(evidence[0].id, "ev-test");
    assert_eq!(evidence[0].summary, "config has active provider");
    assert!(
        restore_conversation_messages(&records)
            .expect("restore messages")
            .is_empty()
    );
}

#[test]
fn child_transcript_records_parent_attribution_without_affecting_parent_restore() {
    let base_dir = std::env::temp_dir().join(format!(
        "letcode-transcript-child-test-{}",
        unix_timestamp_ms()
    ));

    let mut parent = TranscriptRecorder::create(&base_dir).expect("create parent recorder");
    parent
        .record_user_message("parent question")
        .expect("record parent user");
    parent
        .record_assistant_message("parent answer")
        .expect("record parent assistant");

    let child_dir = child_sessions_dir(&base_dir);
    let mut child = TranscriptRecorder::create(&child_dir).expect("create child recorder");
    child
        .record_session_started("gpt-test")
        .expect("record child start");
    child
        .record_subagent_lifecycle(
            "sub-1",
            parent.session_id(),
            "turn-1",
            "explorer",
            "running",
            Some("inspect src".into()),
        )
        .expect("record lifecycle");
    child
        .record_assistant_message("child summary")
        .expect("record child message");

    let parent_records = read_records(base_dir.join(format!("{}.jsonl", parent.session_id())))
        .expect("read parent records");
    let child_records = read_records(child_dir.join(format!("{}.jsonl", child.session_id())))
        .expect("read child records");

    let restored = restore_conversation_messages(&parent_records).expect("restore messages");
    assert_eq!(restored.len(), 2);
    assert_eq!(restored[0].content, "parent question");
    assert_eq!(restored[1].content, "parent answer");
    assert!(matches!(
        child_records[1].event,
        TranscriptEvent::SubagentLifecycle { .. }
    ));

    match &child_records[1].event {
        TranscriptEvent::SubagentLifecycle {
            parent_session_id,
            parent_run_id,
            agent_name,
            status,
            ..
        } => {
            assert_eq!(parent_session_id, parent.session_id());
            assert_eq!(parent_run_id, "turn-1");
            assert_eq!(agent_name, "explorer");
            assert_eq!(status, "running");
        }
        other => panic!("unexpected event: {other:?}"),
    }
}

#[test]
fn child_session_helpers_only_list_existing_children_and_restore_child_records() {
    let base_dir = std::env::temp_dir().join(format!(
        "letcode-transcript-child-helper-test-{}",
        unix_timestamp_ms()
    ));

    let mut parent = TranscriptRecorder::create(&base_dir).expect("create parent recorder");
    let child_dir = child_sessions_dir(&base_dir);
    fs::create_dir_all(&child_dir).expect("create child dir");
    let parent_session_id = parent.session_id().to_string();

    parent
        .record_subagent_result(
            "run-1",
            &parent_session_id,
            "turn-1",
            "placeholder-existing",
            "explorer",
            "completed",
            "inspected src/tool.rs",
        )
        .expect("record first child result");
    parent
        .record_subagent_result(
            "run-2",
            &parent_session_id,
            "turn-2",
            "missing-child",
            "explorer",
            "completed",
            "should be ignored",
        )
        .expect("record second child result");

    let mut child = TranscriptRecorder::create(&child_dir).expect("create child recorder");
    let child_session_id = child.session_id().to_string();
    child
        .record_user_message("inspect state")
        .expect("record child user message");
    child
        .record_assistant_message("done")
        .expect("record child assistant message");

    let mut parent_records = read_records(base_dir.join(format!("{}.jsonl", parent.session_id())))
        .expect("read parent records");
    match &mut parent_records[0].event {
        TranscriptEvent::SubagentResult {
            child_session_id: recorded_id,
            ..
        } => *recorded_id = child_session_id.clone(),
        other => panic!("unexpected event: {other:?}"),
    }

    let children = list_child_sessions_for_parent(&base_dir, &parent_records);
    assert_eq!(children.len(), 1);
    assert_eq!(children[0].child_session_id, child_session_id);
    assert_eq!(children[0].agent_name, "explorer");
    assert_eq!(children[0].status, "completed");
    assert_eq!(children[0].summary, "inspected src/tool.rs");

    let child_records = read_child_session_records(&base_dir, &children[0].child_session_id)
        .expect("read child session records");
    assert_eq!(child_records.len(), 2);
    assert!(matches!(
        child_records[0].event,
        TranscriptEvent::UserMessage { .. }
    ));
    assert!(matches!(
        child_records[1].event,
        TranscriptEvent::AssistantMessage { .. }
    ));
}

#[test]
fn child_session_listing_uses_parent_results_not_lifecycle_records() {
    let base_dir = std::env::temp_dir().join(format!(
        "letcode-transcript-child-listing-test-{}",
        unix_timestamp_ms()
    ));

    let mut parent = TranscriptRecorder::create(&base_dir).expect("create parent recorder");
    let child_dir = child_sessions_dir(&base_dir);
    let child = TranscriptRecorder::create(&child_dir).expect("create child recorder");
    let parent_session_id = parent.session_id().to_string();
    let child_session_id = child.session_id().to_string();

    parent
        .record_subagent_lifecycle(
            "run-1",
            &parent_session_id,
            "turn-1",
            "explorer",
            "running",
            Some("inspect src".into()),
        )
        .expect("record lifecycle");

    let records = read_records(base_dir.join(format!("{}.jsonl", parent.session_id())))
        .expect("read parent records");
    assert!(list_child_sessions_for_parent(&base_dir, &records).is_empty());

    parent
        .record_subagent_result(
            "run-1",
            &parent_session_id,
            "turn-1",
            &child_session_id,
            "explorer",
            "completed",
            "inspection done",
        )
        .expect("record result");

    let records = read_records(base_dir.join(format!("{}.jsonl", parent.session_id())))
        .expect("read updated parent records");
    let children = list_child_sessions_for_parent(&base_dir, &records);
    assert_eq!(children.len(), 1);
    assert_eq!(children[0].child_session_id, child_session_id);
    assert_eq!(children[0].status, "completed");
}

#[test]
fn duplicate_child_results_are_listed_once_with_latest_summary() {
    let base_dir = std::env::temp_dir().join(format!(
        "letcode-transcript-child-dedupe-test-{}",
        unix_timestamp_ms()
    ));

    let mut parent = TranscriptRecorder::create(&base_dir).expect("create parent recorder");
    let child_dir = child_sessions_dir(&base_dir);
    let child = TranscriptRecorder::create(&child_dir).expect("create child recorder");
    let parent_session_id = parent.session_id().to_string();
    let child_session_id = child.session_id().to_string();

    parent
        .record_subagent_result(
            "run-1",
            &parent_session_id,
            "turn-1",
            &child_session_id,
            "explorer",
            "running",
            "first summary",
        )
        .expect("record first result");
    parent
        .record_subagent_result(
            "run-1",
            &parent_session_id,
            "turn-1",
            &child_session_id,
            "explorer",
            "completed",
            "latest summary",
        )
        .expect("record second result");

    let records = read_records(base_dir.join(format!("{}.jsonl", parent.session_id())))
        .expect("read parent records");
    let children = list_child_sessions_for_parent(&base_dir, &records);

    assert_eq!(children.len(), 1);
    assert_eq!(children[0].child_session_id, child_session_id);
    assert_eq!(children[0].status, "completed");
    assert_eq!(children[0].summary, "latest summary");
}

#[test]
fn subagent_result_round_trips_structured_payload() {
    let base_dir = std::env::temp_dir().join(format!(
        "letcode-transcript-structured-subagent-test-{}",
        unix_timestamp_ms()
    ));
    let mut recorder = TranscriptRecorder::create(&base_dir).expect("create recorder");

    recorder
        .record_subagent_result_structured(
            "run-1",
            "parent-session",
            "turn-1",
            "child-session",
            "explorer",
            "completed",
            "inspection done",
            Some(StructuredSubagentResult {
                status: "completed".into(),
                summary: "inspection done".into(),
                malformed: false,
                findings: vec!["found contract".into()],
                files_read: vec!["src/subagent.rs".into()],
                files_changed: vec![],
                commands_run: vec!["cargo test subagent::tests".into()],
                validation: vec!["passed".into()],
                blockers: vec![],
                next_steps: vec!["report".into()],
                run_id: "run-1".into(),
                child_session_id: "child-session".into(),
                raw_excerpt: None,
            }),
        )
        .expect("record structured result");

    let records = read_records(base_dir.join(format!("{}.jsonl", recorder.session_id())))
        .expect("read records");
    match &records[1].event {
        TranscriptEvent::Evidence {
            source:
                EvidenceSource::Subagent {
                    run_id,
                    child_session_id,
                    parent_tool,
                    parent_turn_id,
                    parent_session_id,
                    ..
                },
            summary,
            detail,
            ..
        } => {
            assert_eq!(run_id, "run-1");
            assert_eq!(child_session_id, "child-session");
            assert_eq!(parent_tool, tool_names::TOOL_AGENT_EXPLORE);
            assert_eq!(parent_turn_id.as_deref(), Some("turn-1"));
            assert_eq!(parent_session_id.as_deref(), Some("parent-session"));
            assert_eq!(summary, "inspection done");
            let detail = detail.as_deref().expect("structured detail");
            assert!(detail.contains("found contract"));
            assert!(detail.contains("src/subagent.rs"));
        }
        other => panic!("unexpected event: {other:?}"),
    }
}

#[test]
fn restore_job_board_derives_unreconciled_reconciled_and_reusable_states() {
    let base_dir = std::env::temp_dir().join(format!(
        "letcode-transcript-job-board-test-{}",
        unix_timestamp_ms()
    ));
    let mut recorder = TranscriptRecorder::create(&base_dir).expect("create recorder");

    recorder
        .record_subagent_result_structured(
            "run-1",
            "parent-session",
            "turn-1",
            "child-1",
            "explorer",
            "completed",
            "done",
            Some(StructuredSubagentResult {
                status: "completed".into(),
                summary: "done".into(),
                malformed: false,
                findings: vec![],
                files_read: vec!["src/lib.rs".into()],
                files_changed: vec![],
                commands_run: vec![],
                validation: vec![],
                blockers: vec![],
                next_steps: vec![],
                run_id: "run-1".into(),
                child_session_id: "child-1".into(),
                raw_excerpt: None,
            }),
        )
        .expect("record result");
    let records = read_records(base_dir.join(format!("{}.jsonl", recorder.session_id())))
        .expect("read records");
    let job_board = restore_job_board(&base_dir, &records).expect("derive unreconciled board");
    assert_eq!(job_board.len(), 1);
    assert!(job_board[0].unreconciled);
    assert!(!job_board[0].reconciled);
    assert!(!job_board[0].reusable_eligible);

    let mut recorder = TranscriptRecorder::open_existing(&base_dir, recorder.session_id())
        .expect("reopen recorder");
    recorder
        .record_subagent_reconciliation(
            "run-1",
            "child-1",
            "explorer",
            "turn-2",
            "reconciled child run run-1",
        )
        .expect("record reconciliation");
    let records = read_records(base_dir.join(format!("{}.jsonl", recorder.session_id())))
        .expect("read updated records");
    let job_board = restore_job_board(&base_dir, &records).expect("derive reconciled board");
    assert_eq!(job_board.len(), 1);
    assert!(!job_board[0].unreconciled);
    assert!(job_board[0].reconciled);
    assert!(job_board[0].reusable_eligible);
}

#[test]
fn context_view_remove_is_append_only_metadata_not_raw_purge() {
    let base_dir = std::env::temp_dir().join(format!(
        "letcode-transcript-context-view-append-only-test-{}",
        unix_timestamp_ms()
    ));
    let mut recorder = TranscriptRecorder::create(&base_dir).expect("create recorder");

    recorder
        .record_assistant_message("soft note that may be hidden from derived view")
        .expect("record assistant note");
    recorder
        .record_context_view_operation_metadata(
            "remove_from_view",
            Some("block-seq-1-note".into()),
            None,
            Some("hide from prompt-derived context view only".into()),
        )
        .expect("record remove-from-view metadata");

    let transcript_path = base_dir.join(format!("{}.jsonl", recorder.session_id()));
    let records = read_records(&transcript_path).expect("read records");

    assert_eq!(records.len(), 2);
    assert!(matches!(
        &records[0].event,
        TranscriptEvent::AssistantMessage { content }
            if content == "soft note that may be hidden from derived view"
    ));
    assert!(matches!(
        &records[1].event,
        TranscriptEvent::ContextViewOperationMetadata {
            operation,
            block_id,
            node_id,
            ..
        } if operation == "remove_from_view"
            && block_id.as_deref() == Some("block-seq-1-note")
            && node_id.is_none()
    ));
    assert_eq!(
        records
            .iter()
            .map(|record| record.sequence)
            .collect::<Vec<_>>(),
        vec![1, 2]
    );

    let mut reopened = TranscriptRecorder::open_existing(&base_dir, recorder.session_id())
        .expect("reopen recorder");
    reopened
        .record_user_message("new message after reopen")
        .expect("append after reopen");

    let reopened_records = read_records(&transcript_path).expect("read reopened records");
    assert_eq!(reopened_records.len(), 3);
    assert!(matches!(
        &reopened_records[0].event,
        TranscriptEvent::AssistantMessage { content }
            if content == "soft note that may be hidden from derived view"
    ));
    assert!(matches!(
        &reopened_records[1].event,
        TranscriptEvent::ContextViewOperationMetadata { operation, .. }
            if operation == "remove_from_view"
    ));
    assert!(matches!(
        &reopened_records[2].event,
        TranscriptEvent::UserMessage { content }
            if content.display_text() == "new message after reopen"
    ));
    let sequences = reopened_records
        .iter()
        .map(|record| record.sequence)
        .collect::<Vec<_>>();
    assert_eq!(sequences, vec![1, 2, 3]);
    assert_eq!(
        reopened_records.last().map(|record| record.sequence),
        Some(
            records
                .iter()
                .map(|record| record.sequence)
                .max()
                .unwrap_or(0)
                + 1
        )
    );
}

#[test]
fn context_resolve_pending_metadata_is_recorded() {
    let base_dir = std::env::temp_dir().join(format!(
        "letcode-transcript-context-resolve-metadata-test-{}",
        unix_timestamp_ms()
    ));
    let mut recorder = TranscriptRecorder::create(&base_dir).expect("create recorder");
    recorder
        .record_error("context view projection unavailable")
        .expect("record error");

    let output = ToolResult::ok(
        tool_names::TOOL_CONTEXT_RESOLVE,
        json!({
            "ok": true,
            "operation_metadata": {"operation": "resolve", "block_id": "block-seq-1-error"},
            "pending_recording": true
        }),
    );
    recorder
        .record_context_tool_pending_metadata(tool_names::TOOL_CONTEXT_RESOLVE, true, &output)
        .expect("record resolve metadata");

    let transcript_path = base_dir.join(format!("{}.jsonl", recorder.session_id()));
    let records = read_records(&transcript_path).expect("read records");
    assert_eq!(records.len(), 2);
    assert!(matches!(
        &records[1].event,
        TranscriptEvent::ContextViewOperationMetadata { operation, block_id, .. }
            if operation == "resolve" && block_id.as_deref() == Some("block-seq-1-error")
    ));

    let projection = transcript_projection::project_context_view(&records)
        .expect("project context view with resolve metadata");
    assert_eq!(
        projection
            .view_state
            .status(&crate::context_view::ContextBlockId::new("block-seq-1-error").expect("id")),
        Some(crate::context_view::ContextViewStatus::Resolved)
    );
}

#[test]
fn context_tool_pending_metadata_is_gated_by_tool_name_and_success() {
    let base_dir = std::env::temp_dir().join(format!(
        "letcode-transcript-context-tool-metadata-gating-test-{}",
        unix_timestamp_ms()
    ));
    let mut recorder = TranscriptRecorder::create(&base_dir).expect("create recorder");
    let pending_output = ToolResult::ok(
        tool_names::TOOL_CONTEXT_PIN,
        json!({
            "pending_recording": true,
            "operation_metadata": {"operation": "pin", "block_id": "block-seq-1-note"}
        }),
    );

    recorder
        .record_context_tool_pending_metadata(tool_names::TOOL_FS_READ, true, &pending_output)
        .expect("ignore non-context metadata");
    recorder
        .record_context_tool_pending_metadata(tool_names::TOOL_CONTEXT_PIN, false, &pending_output)
        .expect("ignore failed context metadata");

    let transcript_path = base_dir.join(format!("{}.jsonl", recorder.session_id()));
    let records = read_records(&transcript_path).expect("read records");
    assert!(records.is_empty());
}

#[test]
fn successful_context_open_block_records_open_detail_metadata() {
    let base_dir = std::env::temp_dir().join(format!(
        "letcode-transcript-context-open-metadata-test-{}",
        unix_timestamp_ms()
    ));
    let mut recorder = TranscriptRecorder::create(&base_dir).expect("create recorder");
    recorder
        .record_assistant_message("visible note")
        .expect("record visible note");

    let output = ToolResult::ok(
        tool_names::TOOL_CONTEXT_OPEN,
        json!({
            "ok": true,
            "ref_type": "block",
            "ref_id": "block-seq-1-note",
            "operation_metadata": {"operation": "open_detail", "block_id": "block-seq-1-note"},
            "pending_recording": true
        }),
    );
    recorder
        .record_context_tool_pending_metadata(tool_names::TOOL_CONTEXT_OPEN, true, &output)
        .expect("record open detail metadata");

    let transcript_path = base_dir.join(format!("{}.jsonl", recorder.session_id()));
    let records = read_records(&transcript_path).expect("read records");
    assert_eq!(records.len(), 2);
    assert!(matches!(
        &records[1].event,
        TranscriptEvent::ContextViewOperationMetadata { operation, block_id, .. }
            if operation == "open_detail" && block_id.as_deref() == Some("block-seq-1-note")
    ));

    let projection = transcript_projection::project_context_view(&records)
        .expect("project context view with open detail metadata");
    assert_eq!(
        projection
            .view_state
            .open_detail_block_id()
            .map(|block_id| block_id.as_str()),
        Some("block-seq-1-note")
    );
}

#[test]
fn restore_job_board_derives_active_state_from_child_transcript() {
    let base_dir = std::env::temp_dir().join(format!(
        "letcode-transcript-active-job-board-test-{}",
        unix_timestamp_ms()
    ));
    let child_dir = child_sessions_dir(&base_dir);
    let mut child = TranscriptRecorder::create(&child_dir).expect("create child recorder");
    let child_session_id = child.session_id().to_string();
    child
        .record_subagent_lifecycle(
            "run-active",
            "parent-session",
            "turn-1",
            "fixer",
            "running",
            Some("apply patch".into()),
        )
        .expect("record running lifecycle");

    let mut child_file = OpenOptions::new()
        .append(true)
        .open(child.path())
        .expect("open child transcript for partial append");
    child_file
        .write_all(
            br#"{"session_id":"child","sequence":2,"timestamp_ms":1,"kind":"tool_call_finished""#,
        )
        .expect("append partial live record");

    let job_board = restore_job_board(&base_dir, &[]).expect("derive active board");
    assert_eq!(job_board.len(), 1);
    assert!(job_board[0].active);
    assert_eq!(job_board[0].child_session_id, child_session_id);
    assert_eq!(job_board[0].status, "running");
}

#[test]
fn read_records_accepts_legacy_subagent_result_without_structured_payload() {
    let base_dir = std::env::temp_dir().join(format!(
        "letcode-transcript-legacy-subagent-test-{}",
        unix_timestamp_ms()
    ));
    fs::create_dir_all(&base_dir).expect("create temp dir");
    let path = base_dir.join("legacy.jsonl");
    fs::write(
        &path,
        r#"{"session_id":"s","sequence":1,"timestamp_ms":0,"kind":"subagent_result","run_id":"run-1","parent_session_id":"parent","parent_run_id":"turn-1","child_session_id":"child","agent_name":"explorer","status":"completed","summary":"done"}
"#,
    )
    .expect("write transcript");

    let records = read_records(&path).expect("read legacy transcript");
    match &records[0].event {
        TranscriptEvent::SubagentResult { .. } => {}
        other => panic!("unexpected event: {other:?}"),
    }
}

#[test]
fn child_session_summaries_sort_by_timestamp_then_session_id() {
    let mut children = vec![
        ChildSessionSummary {
            parent_session_id: "parent".into(),
            parent_run_id: "turn".into(),
            child_session_id: "child-c".into(),
            agent_name: "explorer".into(),
            status: "completed".into(),
            summary: "third".into(),
            timestamp_ms: 2,
        },
        ChildSessionSummary {
            parent_session_id: "parent".into(),
            parent_run_id: "turn".into(),
            child_session_id: "child-b".into(),
            agent_name: "explorer".into(),
            status: "completed".into(),
            summary: "second".into(),
            timestamp_ms: 1,
        },
        ChildSessionSummary {
            parent_session_id: "parent".into(),
            parent_run_id: "turn".into(),
            child_session_id: "child-a".into(),
            agent_name: "explorer".into(),
            status: "completed".into(),
            summary: "first".into(),
            timestamp_ms: 1,
        },
    ];

    sort_child_session_summaries(&mut children);

    let ordered_ids = children
        .iter()
        .map(|child| child.child_session_id.as_str())
        .collect::<Vec<_>>();
    assert_eq!(ordered_ids, ["child-a", "child-b", "child-c"]);
}

#[test]
fn rapidly_created_recorders_get_unique_session_ids_and_paths() {
    let base_dir = std::env::temp_dir().join(format!(
        "letcode-transcript-unique-id-test-{}",
        unix_timestamp_ms()
    ));

    let first = TranscriptRecorder::create(&base_dir).expect("create first recorder");
    let second = TranscriptRecorder::create(&base_dir).expect("create second recorder");

    assert_ne!(first.session_id(), second.session_id());
    assert_ne!(first.path(), second.path());
    assert!(first.path().exists());
    assert!(second.path().exists());
}

#[test]
fn duplicate_evidence_ids_fail_restore() {
    let records = vec![
        TranscriptRecord {
            session_id: "s".into(),
            sequence: 1,
            timestamp_ms: 0,
            context_branch_id: None,
            event: TranscriptEvent::Evidence {
                id: "ev-1".into(),
                evidence_kind: EvidenceKind::Decision,
                title: "one".into(),
                summary: "one".into(),
                detail: None,
                source: EvidenceSource::Transcript { sequence: 1 },
                tags: vec![],
            },
        },
        TranscriptRecord {
            session_id: "s".into(),
            sequence: 2,
            timestamp_ms: 1,
            context_branch_id: None,
            event: TranscriptEvent::Evidence {
                id: "ev-1".into(),
                evidence_kind: EvidenceKind::Decision,
                title: "two".into(),
                summary: "two".into(),
                detail: None,
                source: EvidenceSource::Transcript { sequence: 2 },
                tags: vec![],
            },
        },
    ];

    assert!(restore_session_evidence(&records).is_err());
}

#[test]
fn todo_and_auto_continue_events_round_trip_and_restore_latest_state() {
    let base_dir = std::env::temp_dir().join(format!(
        "letcode-transcript-todo-test-{}",
        unix_timestamp_ms()
    ));
    let mut recorder = TranscriptRecorder::create(&base_dir).expect("create recorder");

    recorder
        .record_todo_snapshot(vec![TodoItem {
            id: "t1".into(),
            content: "inspect".into(),
            status: crate::agent::TodoStatus::Pending,
        }])
        .expect("record first todo snapshot");
    recorder
        .record_auto_continue_changed(AutoContinueState {
            enabled: true,
            max_continuations: 2,
        })
        .expect("record auto-continue");
    recorder
        .record_auto_continuation_scheduled(1, 1)
        .expect("record auto-continuation scheduled");
    recorder
        .record_todo_snapshot(vec![
            TodoItem {
                id: "t1".into(),
                content: "inspect".into(),
                status: crate::agent::TodoStatus::Completed,
            },
            TodoItem {
                id: "t2".into(),
                content: "validate".into(),
                status: crate::agent::TodoStatus::InProgress,
            },
        ])
        .expect("record second todo snapshot");

    let records = read_records(base_dir.join(format!("{}.jsonl", recorder.session_id())))
        .expect("read records");

    let latest_todos = restore_latest_todo_snapshot(&records).expect("latest todos");
    assert_eq!(latest_todos.len(), 2);
    assert_eq!(latest_todos[0].status, crate::agent::TodoStatus::Completed);
    assert_eq!(latest_todos[1].status, crate::agent::TodoStatus::InProgress);

    let auto_continue = restore_latest_auto_continue_state(&records).expect("latest auto-continue");
    assert!(auto_continue.enabled);
    assert_eq!(auto_continue.max_continuations, 2);
    assert!(
        restore_conversation_messages(&records)
            .expect("restore messages")
            .is_empty()
    );
    assert!(matches!(
        records[2].event,
        TranscriptEvent::AutoContinuationScheduled {
            continuation_count: 1,
            remaining_unfinished: 1,
        }
    ));
}

#[test]
fn restore_latest_workflow_state_resets_on_new_turn_and_error() {
    let stale_todo = TodoItem {
        id: "stale".into(),
        content: "stale task".into(),
        status: crate::agent::TodoStatus::InProgress,
    };
    let records = vec![
        TranscriptRecord {
            session_id: "s".into(),
            sequence: 1,
            timestamp_ms: 0,
            context_branch_id: None,
            event: TranscriptEvent::TodoSnapshot {
                items: vec![stale_todo.clone()],
            },
        },
        TranscriptRecord {
            session_id: "s".into(),
            sequence: 2,
            timestamp_ms: 1,
            context_branch_id: None,
            event: TranscriptEvent::AutoContinueChanged {
                state: AutoContinueState {
                    enabled: true,
                    max_continuations: 2,
                },
            },
        },
        TranscriptRecord {
            session_id: "s".into(),
            sequence: 3,
            timestamp_ms: 2,
            context_branch_id: None,
            event: TranscriptEvent::Error {
                message: "tool event failed".into(),
            },
        },
    ];

    assert!(restore_latest_todo_snapshot(&records).is_none());
    assert!(restore_latest_auto_continue_state(&records).is_none());

    let mut records = records;
    records.push(TranscriptRecord {
        session_id: "s".into(),
        sequence: 4,
        timestamp_ms: 3,
        context_branch_id: None,
        event: TranscriptEvent::TodoSnapshot {
            items: vec![stale_todo],
        },
    });
    records.push(TranscriptRecord {
        session_id: "s".into(),
        sequence: 5,
        timestamp_ms: 4,
        context_branch_id: None,
        event: TranscriptEvent::UserMessage {
            content: crate::user_content::UserMessageContent::new("next", Vec::new()),
        },
    });

    assert!(restore_latest_todo_snapshot(&records).is_none());
    assert!(restore_latest_auto_continue_state(&records).is_none());
}

#[test]
fn session_started_only_is_not_session_content() {
    let mut records = vec![TranscriptRecord {
        session_id: "s".into(),
        sequence: 1,
        timestamp_ms: 0,
        context_branch_id: None,
        event: TranscriptEvent::SessionStarted {
            model: "gpt-test".into(),
        },
    }];
    assert!(!has_session_content(&records));

    records.push(TranscriptRecord {
        session_id: "s".into(),
        sequence: 2,
        timestamp_ms: 1,
        context_branch_id: None,
        event: TranscriptEvent::UserMessage {
            content: "hello".into(),
        },
    });
    assert!(has_session_content(&records));
}

#[test]
fn session_title_does_not_make_session_non_empty() {
    let records = vec![TranscriptRecord {
        session_id: "s".into(),
        sequence: 1,
        timestamp_ms: 0,
        context_branch_id: None,
        event: TranscriptEvent::SessionTitle {
            title: "hello".into(),
        },
    }];

    assert!(!has_session_content(&records));
}

#[test]
fn unknown_transcript_events_are_read_and_ignored_for_restore() {
    let base_dir = std::env::temp_dir().join(format!(
        "letcode-transcript-unknown-event-test-{}",
        unix_timestamp_ms()
    ));
    fs::create_dir_all(&base_dir).expect("create temp dir");
    let path = base_dir.join("unknown.jsonl");
    fs::write(
        &path,
        r#"{"session_id":"s","sequence":1,"timestamp_ms":0,"kind":"future_audit_event","extra":"ignored"}
{"session_id":"s","sequence":2,"timestamp_ms":1,"kind":"user_message","content":"hi"}
"#,
    )
    .expect("write transcript");

    let records = read_records(&path).expect("read unknown transcript event");
    assert_eq!(records.len(), 2);
    assert!(matches!(records[0].event, TranscriptEvent::Unknown));

    let restored = restore_conversation_messages(&records).expect("restore messages");
    assert_eq!(restored.len(), 1);
    assert_eq!(restored[0].content, "hi");
}

#[test]
fn known_transcript_events_with_missing_required_fields_still_fail() {
    let base_dir = std::env::temp_dir().join(format!(
        "letcode-transcript-known-event-fail-test-{}",
        unix_timestamp_ms()
    ));
    fs::create_dir_all(&base_dir).expect("create temp dir");
    let path = base_dir.join("malformed-known.jsonl");
    fs::write(
        &path,
        r#"{"session_id":"s","sequence":1,"timestamp_ms":0,"kind":"user_message"}
"#,
    )
    .expect("write transcript");

    let error = read_records(&path).expect_err("known malformed event should fail");
    assert!(error.to_string().contains("failed to parse line 1"));
}

#[test]
fn strict_read_records_fails_on_partial_tail_but_live_read_ignores_it() {
    let base_dir = std::env::temp_dir().join(format!(
        "letcode-transcript-partial-tail-test-{}",
        unix_timestamp_ms()
    ));
    fs::create_dir_all(&base_dir).expect("create temp dir");
    let path = base_dir.join("partial.jsonl");
    fs::write(
        &path,
        concat!(
            r#"{"session_id":"s","sequence":1,"timestamp_ms":0,"kind":"user_message","content":"hi"}"#,
            "\n",
            r#"{"session_id":"s","sequence":2,"timestamp_ms":1,"kind":"tool_call_finished""#
        ),
    )
    .expect("write partial transcript");

    let strict_error = read_records(&path).expect_err("strict read should reject partial tail");
    assert!(strict_error.to_string().contains("failed to parse line 2"));

    let records = read_records_allow_partial_tail(&path).expect("live read ignores partial tail");
    assert_eq!(records.len(), 1);
    assert!(matches!(
        records[0].event,
        TranscriptEvent::UserMessage { .. }
    ));
}

#[test]
fn recovery_ignores_an_uncommitted_logical_checkpoint_tail_and_keeps_legacy_projection() {
    let base_dir = journal_test_dir("checkpoint-uncommitted-tail");
    fs::create_dir_all(&base_dir).expect("create temp dir");
    let path = base_dir.join("checkpoint-tail.jsonl");
    let legacy = TranscriptRecord {
        session_id: "s".into(),
        sequence: 1,
        timestamp_ms: 0,
        context_branch_id: None,
        event: TranscriptEvent::UserMessage {
            content: UserMessageContent::from("legacy request"),
        },
    };
    let checkpoint = JournalRecordV1 {
        schema_version: JOURNAL_SCHEMA_VERSION,
        event_id: "s:2".into(),
        scope: JournalScope::Branch,
        base_revision: 1,
        resulting_revision: 2,
        transaction_id: Some("checkpoint-tail".into()),
        transaction_index: Some(0),
        transaction_count: Some(1),
        record: TranscriptRecord {
            session_id: "s".into(),
            sequence: 2,
            timestamp_ms: 0,
            context_branch_id: Some(ROOT_CONTEXT_BRANCH_ID.into()),
            event: TranscriptEvent::LogicalCheckpoint(LogicalCheckpointEventV1 {
                schema_version: 1,
                checkpoint_id: "checkpoint-tail".into(),
                turn_id: 1,
                previous_segment_id: 0,
                segment_id: 1,
                previous_checkpoint_id: None,
                boundary_sequence: 1,
                context_scope_revision: 0,
                covered_source_spans: vec![LogicalCheckpointSourceSpanV1 {
                    start_sequence: 1,
                    end_sequence: 1,
                }],
                retained_items: Vec::new(),
            }),
        },
    };
    let legacy_line = serde_json::to_string(&legacy).expect("serialize legacy record");
    let checkpoint_line = String::from_utf8(
        serialize_journal_record(&checkpoint).expect("serialize checkpoint record"),
    )
    .expect("checkpoint JSON is UTF-8");
    // The final commit marker is both malformed and physically incomplete,
    // matching a crash while acknowledging a checkpoint transaction.
    fs::write(
        &path,
        format!("{legacy_line}\n{checkpoint_line}\n{{\"journal_entry\":\"transaction_commit\""),
    )
    .expect("write interrupted journal");

    let recovered = read_records_allow_partial_tail(&path).expect("recover live prefix");
    assert_eq!(recovered.len(), 1);
    assert!(matches!(
        recovered[0].event,
        TranscriptEvent::UserMessage { .. }
    ));
    assert!(
        restore_runtime_snapshot(&recovered)
            .expect("restore legacy prefix")
            .current_segment_id
            .is_none()
    );
    assert_eq!(
        restore_conversation_messages(&recovered)
            .expect("restore legacy messages")
            .into_iter()
            .map(|message| message.content)
            .collect::<Vec<_>>(),
        ["legacy request"]
    );
}

#[test]
fn phase1b_committed_tool_result_restores_aggregate_but_crash_prefix_cannot() {
    let base_dir = journal_test_dir("phase1b-tool-result-crash-window");
    let mut recorder = TranscriptRecorder::create(&base_dir).expect("create journal");
    recorder
        .record_tool_call_started("call-1", "shell__exec", json!({"command": "true"}))
        .expect("commit start");
    recorder
        .record_tool_call_finished(
            "call-1",
            "shell__exec",
            true,
            ToolResult::ok(
                "shell__exec",
                json!({"stdout": "x".repeat(5000), "status": 0}),
            ),
        )
        .expect("commit finished result");
    let path = base_dir.join(format!("{}.jsonl", recorder.session_id()));
    let committed = read_records(&path).expect("read committed journal");
    assert!(
        crate::context_view::restore_folded_outputs(
            &committed,
            crate::context_view::DEFAULT_FOLDED_OUTPUT_THRESHOLD_BYTES,
        )
        .expect("restore committed aggregate")
        .contains_key("folded-output-seq-2-tool-result")
    );

    let torn_path = base_dir.join("torn-tool-result.jsonl");
    let raw = fs::read_to_string(&path).expect("read journal bytes");
    let start_line = raw.lines().next().expect("committed start record");
    fs::write(
        &torn_path,
        format!("{start_line}\n{{\"schema_version\":1,\"event_id\":\"torn"),
    )
    .expect("write crash prefix");
    let recovered = read_records_allow_partial_tail(&torn_path).expect("recover complete prefix");
    assert!(
        crate::context_view::restore_folded_outputs(
            &recovered,
            crate::context_view::DEFAULT_FOLDED_OUTPUT_THRESHOLD_BYTES,
        )
        .expect("start-only prefix remains restorable")
        .is_empty()
    );
}

#[test]
fn live_read_records_keeps_complete_tail_strict() {
    let base_dir = std::env::temp_dir().join(format!(
        "letcode-transcript-complete-malformed-tail-test-{}",
        unix_timestamp_ms()
    ));
    fs::create_dir_all(&base_dir).expect("create temp dir");
    let path = base_dir.join("malformed-tail.jsonl");
    fs::write(
        &path,
        concat!(
            r#"{"session_id":"s","sequence":1,"timestamp_ms":0,"kind":"user_message","content":"hi"}"#,
            "\n",
            r#"{"session_id":"s","sequence":2,"timestamp_ms":1,"kind":"tool_call_finished""#,
            "\n"
        ),
    )
    .expect("write malformed complete transcript");

    let error = read_records_allow_partial_tail(&path)
        .expect_err("complete malformed tail should still fail");
    assert!(error.to_string().contains("failed to parse line 2"));
}

#[test]
fn live_partial_tail_keeps_incomplete_batch_protected_until_final_output_arrives() {
    let base_dir = std::env::temp_dir().join(format!(
        "letcode-transcript-live-batch-tail-test-{}",
        unix_timestamp_ms()
    ));
    fs::create_dir_all(&base_dir).expect("create temp dir");
    let path = base_dir.join("live.jsonl");
    let calls = vec![
        HistoryToolCall {
            call_id: "call-1".into(),
            name: "fs__read".into(),
            arguments_json: r#"{"path":"one"}"#.into(),
        },
        HistoryToolCall {
            call_id: "call-2".into(),
            name: "fs__read".into(),
            arguments_json: r#"{"path":"two"}"#.into(),
        },
    ];
    let prefix = vec![
        TranscriptRecord {
            session_id: "live".into(),
            sequence: 1,
            timestamp_ms: 0,
            context_branch_id: None,
            event: TranscriptEvent::TurnStarted(TurnStartedEvent {
                turn_id: 1,
                intent: "inspect".into(),
                directive: "read both files".into(),
                validation_reminder: String::new(),
            }),
        },
        TranscriptRecord {
            session_id: "live".into(),
            sequence: 2,
            timestamp_ms: 1,
            context_branch_id: None,
            event: TranscriptEvent::UserMessage {
                content: UserMessageContent::from("inspect both"),
            },
        },
        TranscriptRecord {
            session_id: "live".into(),
            sequence: 3,
            timestamp_ms: 2,
            context_branch_id: None,
            event: TranscriptEvent::AssistantToolCallBatch { text: None, calls },
        },
        TranscriptRecord {
            session_id: "live".into(),
            sequence: 4,
            timestamp_ms: 3,
            context_branch_id: None,
            event: TranscriptEvent::ToolCallFinished {
                call_id: "call-1".into(),
                name: "fs__read".into(),
                ok: true,
                output: ToolResult::ok("fs__read", json!({"contents":"one"})),
            },
        },
    ];
    let final_record = TranscriptRecord {
        session_id: "live".into(),
        sequence: 5,
        timestamp_ms: 4,
        context_branch_id: None,
        event: TranscriptEvent::ToolCallFinished {
            call_id: "call-2".into(),
            name: "fs__read".into(),
            ok: true,
            output: ToolResult::ok("fs__read", json!({"contents":"two"})),
        },
    };
    let final_line = serde_json::to_string(&final_record).expect("serialize final output");
    let partial_len = final_line.len() - 1;
    let mut content = prefix
        .iter()
        .map(|record| serde_json::to_string(record).expect("serialize prefix"))
        .collect::<Vec<_>>()
        .join("\n");
    content.push('\n');
    content.push_str(&final_line[..partial_len]);
    fs::write(&path, content).expect("write live partial transcript");

    let live_records = read_records_allow_partial_tail(&path).expect("read complete live prefix");
    assert_eq!(live_records.len(), 4);
    let live = transcript_projection::project_runtime_restore_snapshot(
        "live".into(),
        live_records.clone(),
        transcript_projection::SessionContextCursor {
            branch_id: None,
            leaf_sequence: None,
        },
        &[],
    )
    .expect("project incomplete live runtime");
    let live_history = history_items_from_frames(&live.protocol_frames);
    assert!(
        analyze_history_items(&live_history, None)
            .expect("analyze live group")
            .has_incomplete_tool_call_groups()
    );
    assert!(live.snapshot.compaction.protected_frame_ids.len() >= 3);
    let model = ModelRequestMetadata {
        supports_tools: true,
        ..Default::default()
    };
    for protocol in [ApiProtocol::Responses, ApiProtocol::Completions] {
        assert!(
            build_request(RequestBuilderInput {
                protocol,
                model_id: "gpt-test",
                model: model.clone(),
                prelude: &[],
                snapshot: &live.snapshot,
                tools: &[]
            })
            .is_err(),
            "{protocol:?} must reject the incomplete batch"
        );
    }
    assert_eq!(
        serde_json::to_value(&live_records).expect("serialize live records"),
        serde_json::to_value(&prefix).expect("serialize source prefix")
    );

    let mut file = OpenOptions::new()
        .append(true)
        .open(&path)
        .expect("open live transcript");
    file.write_all(final_line[partial_len..].as_bytes())
        .expect("complete final record");
    file.write_all(b"\n").expect("terminate final record");
    file.flush().expect("flush final record");
    let complete_records =
        read_records_allow_partial_tail(&path).expect("read completed live transcript");
    assert_eq!(complete_records.len(), 5);
    let complete = transcript_projection::project_runtime_restore_snapshot(
        "live".into(),
        complete_records,
        transcript_projection::SessionContextCursor {
            branch_id: None,
            leaf_sequence: None,
        },
        &[],
    )
    .expect("project complete live runtime");
    let complete_history = history_items_from_frames(&complete.protocol_frames);
    assert!(
        !analyze_history_items(&complete_history, None)
            .expect("analyze completed group")
            .has_incomplete_tool_call_groups()
    );
    for protocol in [ApiProtocol::Responses, ApiProtocol::Completions] {
        build_request(RequestBuilderInput {
            protocol,
            model_id: "gpt-test",
            model: model.clone(),
            prelude: &[],
            snapshot: &complete.snapshot,
            tools: &[],
        })
        .expect("complete batch builds for both protocols");
    }
}

#[test]
fn audit_and_unknown_events_are_not_session_content() {
    let records = vec![
        TranscriptRecord {
            session_id: "s".into(),
            sequence: 1,
            timestamp_ms: 0,
            context_branch_id: None,
            event: TranscriptEvent::SessionStarted {
                model: "gpt-test".into(),
            },
        },
        TranscriptRecord {
            session_id: "s".into(),
            sequence: 2,
            timestamp_ms: 1,
            context_branch_id: None,
            event: TranscriptEvent::TurnStarted(TurnStartedEvent {
                turn_id: 1,
                intent: "engineering".into(),
                directive: "none".into(),
                validation_reminder: "targeted".into(),
            }),
        },
        TranscriptRecord {
            session_id: "s".into(),
            sequence: 3,
            timestamp_ms: 2,
            context_branch_id: None,
            event: TranscriptEvent::ToolExecutionSummary(ToolExecutionSummaryEvent {
                turn_id: 1,
                call_id: "call-1".into(),
                name: "fs__read".into(),
                status: "executed".into(),
                rejection: None,
                effect_kind: "read".into(),
                primary_path: Some("src/main.rs".into()),
                command: None,
            }),
        },
        TranscriptRecord {
            session_id: "s".into(),
            sequence: 4,
            timestamp_ms: 3,
            context_branch_id: None,
            event: TranscriptEvent::TurnFinalized(TurnFinalizedEvent {
                turn_id: 1,
                outcome: "completed".into(),
                tool_call_count: 1,
                continuation_count: 0,
                write_effects: 0,
                validation_effects: 0,
                failed_validation_effects: 0,
                validation_advisory_emitted: false,
            }),
        },
        TranscriptRecord {
            session_id: "s".into(),
            sequence: 5,
            timestamp_ms: 4,
            context_branch_id: None,
            event: TranscriptEvent::Unknown,
        },
    ];

    assert!(!has_session_content(&records));
}

#[test]
fn restore_max_turn_id_uses_all_turn_audit_events() {
    let records = vec![
        TranscriptRecord {
            session_id: "s".into(),
            sequence: 1,
            timestamp_ms: 0,
            context_branch_id: None,
            event: TranscriptEvent::TurnStarted(TurnStartedEvent {
                turn_id: 3,
                intent: "engineering".into(),
                directive: "none".into(),
                validation_reminder: "targeted".into(),
            }),
        },
        TranscriptRecord {
            session_id: "s".into(),
            sequence: 2,
            timestamp_ms: 1,
            context_branch_id: None,
            event: TranscriptEvent::ToolExecutionSummary(ToolExecutionSummaryEvent {
                turn_id: 5,
                call_id: "call-1".into(),
                name: "shell__exec".into(),
                status: "executed".into(),
                rejection: None,
                effect_kind: "validation".into(),
                primary_path: None,
                command: Some("cargo test".into()),
            }),
        },
        TranscriptRecord {
            session_id: "s".into(),
            sequence: 3,
            timestamp_ms: 2,
            context_branch_id: None,
            event: TranscriptEvent::TurnFinalized(TurnFinalizedEvent {
                turn_id: 4,
                outcome: "completed".into(),
                tool_call_count: 1,
                continuation_count: 0,
                write_effects: 0,
                validation_effects: 1,
                failed_validation_effects: 0,
                validation_advisory_emitted: false,
            }),
        },
    ];

    assert_eq!(restore_max_turn_id(&records), 5);
}

#[test]
fn context_checkpoint_finishes_on_old_branch_then_switches_subsequent_records() {
    let base_dir = std::env::temp_dir().join(format!(
        "letcode-transcript-context-checkpoint-test-{}",
        unix_timestamp_ms()
    ));
    let mut recorder = TranscriptRecorder::create(&base_dir).expect("create recorder");
    recorder
        .record_session_started("gpt-test")
        .expect("session started");
    recorder
        .record_user_message("root prompt")
        .expect("root prompt");
    recorder
        .record_tool_call_started(
            "call-1",
            tool_names::TOOL_CONTEXT_CHECKPOINT,
            json!({"label": "Try parser fix", "reason": "Need risky exploration"}),
        )
        .expect("tool started");

    recorder
        .record_tool_call_finished_and_apply_context_control(
            "call-1",
            tool_names::TOOL_CONTEXT_CHECKPOINT,
            true,
            ToolResult::ok(
                tool_names::TOOL_CONTEXT_CHECKPOINT,
                json!({
                    "label": "Try parser fix",
                    "reason": "Need risky exploration",
                    "context_only": true,
                    "filesystem_rolled_back": false,
                    "message": "Created a context checkpoint request."
                }),
            ),
        )
        .expect("tool finished with checkpoint");
    recorder
        .record_assistant_message("branch-only response")
        .expect("assistant on new branch");

    let records = read_records(base_dir.join(format!("{}.jsonl", recorder.session_id())))
        .expect("read records");
    assert!(matches!(
        &records[2],
        TranscriptRecord {
            context_branch_id: None,
            event: TranscriptEvent::ToolCallStarted { .. },
            ..
        }
    ));
    assert!(matches!(
        &records[3],
        TranscriptRecord {
            context_branch_id: None,
            event: TranscriptEvent::ToolCallFinished { .. },
            ..
        }
    ));
    assert!(matches!(
        &records[4].event,
        TranscriptEvent::ContextBranchCreated {
            branch_id,
            parent_branch_id,
            base_sequence,
            label,
        } if branch_id == "try-parser-fix"
            && parent_branch_id == ROOT_CONTEXT_BRANCH_ID
            && *base_sequence == 4
            && label.as_deref() == Some("Try parser fix")
    ));
    assert!(matches!(
        &records[5].event,
        TranscriptEvent::ContextCheckout { branch_id, leaf_sequence }
            if branch_id == "try-parser-fix" && *leaf_sequence == 4
    ));
    assert!(matches!(
        &records[6].event,
        TranscriptEvent::ContextExperimentStarted { branch_id, parent_branch_id, base_sequence }
            if branch_id == "try-parser-fix"
                && parent_branch_id == ROOT_CONTEXT_BRANCH_ID
                && *base_sequence == 4
    ));
    assert!(matches!(
        &records[7].event,
        TranscriptEvent::ContextNodeCreated {
            node_id,
            parent_node_id,
            label,
            purpose,
            source_ref,
            ..
        } if node_id == "branch/try-parser-fix"
            && parent_node_id.as_deref() == Some("root")
            && label.as_deref() == Some("Try parser fix")
            && purpose.as_deref() == Some("Need risky exploration")
            && source_ref.as_ref().is_some_and(|source| source.source_kind == "context_branch"
                && source.source_id.as_deref() == Some("try-parser-fix"))
    ));
    assert!(matches!(
        &records[8].event,
        TranscriptEvent::ContextNodeLifecycle { node_id, status }
            if node_id == "root" && *status == ContextNodeStatus::Inactive
    ));
    assert!(matches!(
        &records[9].event,
        TranscriptEvent::ContextNodeLifecycle { node_id, status }
            if node_id == "branch/try-parser-fix" && *status == ContextNodeStatus::Active
    ));
    assert_eq!(
        records[10].context_branch_id.as_deref(),
        Some("try-parser-fix")
    );
    assert!(matches!(
        &records[10].event,
        TranscriptEvent::AssistantMessage { content } if content == "branch-only response"
    ));
    assert_eq!(recorder.current_context_branch_id(), Some("try-parser-fix"));
    assert!(matches!(
        recorder.active_context_experiment(),
        Some(ActiveContextExperiment { branch_id, parent_branch_id, base_sequence, writes_observed })
            if branch_id == "try-parser-fix"
                && parent_branch_id == ROOT_CONTEXT_BRANCH_ID
                && base_sequence == 4
                && !writes_observed
    ));

    let tree = transcript_projection::project_context_tree(&records).expect("project context tree");
    assert_eq!(
        tree.active_node_id().map(|id| id.as_str()),
        Some("branch/try-parser-fix")
    );
}

#[test]
fn non_checkpoint_tool_finished_does_not_switch_branch() {
    let base_dir = std::env::temp_dir().join(format!(
        "letcode-transcript-non-checkpoint-tool-test-{}",
        unix_timestamp_ms()
    ));
    let mut recorder = TranscriptRecorder::create(&base_dir).expect("create recorder");
    recorder
        .record_session_started("gpt-test")
        .expect("session started");
    recorder
        .record_tool_call_started("call-1", "fs__read", json!({"path": "src/main.rs"}))
        .expect("tool started");

    recorder
        .record_tool_call_finished_and_apply_context_control(
            "call-1",
            "fs__read",
            true,
            ToolResult::ok("fs__read", json!({"content": "ok"})),
        )
        .expect("tool finished");
    recorder
        .record_assistant_message("still on main")
        .expect("assistant message");

    let records = read_records(base_dir.join(format!("{}.jsonl", recorder.session_id())))
        .expect("read records");
    assert_eq!(records.len(), 4);
    assert!(matches!(
        records[2].event,
        TranscriptEvent::ToolCallFinished { .. }
    ));
    assert!(matches!(
        records[3].event,
        TranscriptEvent::AssistantMessage { .. }
    ));
    assert_eq!(records[3].context_branch_id, None);
    assert_eq!(recorder.current_context_branch_id(), None);
}

#[test]
fn context_return_switches_back_to_parent_and_carries_summary_forward() {
    let base_dir = std::env::temp_dir().join(format!(
        "letcode-transcript-context-return-test-{}",
        unix_timestamp_ms()
    ));
    let mut recorder = TranscriptRecorder::create(&base_dir).expect("create recorder");
    recorder
        .record_session_started("gpt-test")
        .expect("session started");
    recorder
        .record_user_message("root prompt")
        .expect("root prompt");
    recorder
        .record_tool_call_started(
            "call-1",
            tool_names::TOOL_CONTEXT_CHECKPOINT,
            json!({"label": "Try parser fix", "reason": "Need risky exploration"}),
        )
        .expect("checkpoint started");
    recorder
        .record_tool_call_finished_and_apply_context_control(
            "call-1",
            tool_names::TOOL_CONTEXT_CHECKPOINT,
            true,
            ToolResult::ok(
                tool_names::TOOL_CONTEXT_CHECKPOINT,
                json!({
                    "label": "Try parser fix",
                    "reason": "Need risky exploration",
                    "context_only": true,
                    "filesystem_rolled_back": false,
                    "message": "Created a context checkpoint request."
                }),
            ),
        )
        .expect("checkpoint finished");
    recorder
        .record_tool_execution_summary(ToolExecutionSummaryEvent {
            turn_id: 1,
            call_id: "call-write".into(),
            name: "fs__write".into(),
            status: "completed".into(),
            rejection: None,
            effect_kind: "write".into(),
            primary_path: Some("src/lib.rs".into()),
            command: None,
        })
        .expect("write summary");
    {
        let scope_state = recorder.context_scope_state();
        let mut state = scope_state.lock().expect("scope state lock");
        state
            .active_experiment
            .as_mut()
            .expect("active experiment")
            .writes_observed = true;
    }
    recorder
        .record_tool_call_started(
            "call-2",
            tool_names::TOOL_CONTEXT_RETURN,
            json!({"outcome": "useful", "summary": "Parser path found the root cause", "next_action": "apply the fix on main"}),
        )
        .expect("return started");
    recorder
        .record_tool_call_finished_and_apply_context_control(
            "call-2",
            tool_names::TOOL_CONTEXT_RETURN,
            true,
            ToolResult::ok(
                tool_names::TOOL_CONTEXT_RETURN,
                json!({
                    "outcome": "useful",
                    "summary": "Parser path found the root cause",
                    "next_action": "apply the fix on main",
                    "context_restored": true,
                    "filesystem_rolled_back": false,
                    "message": "Returned from the current context experiment to the parent context. Files were not reverted."
                }),
            ),
        )
        .expect("return finished");

    let records = read_records(base_dir.join(format!("{}.jsonl", recorder.session_id())))
        .expect("read records");
    assert!(matches!(
        &records[11],
        TranscriptRecord {
            context_branch_id: Some(branch_id),
            event: TranscriptEvent::ToolCallStarted { name, .. },
            ..
        } if branch_id == "try-parser-fix" && name == tool_names::TOOL_CONTEXT_RETURN
    ));
    assert!(matches!(
        &records[12],
        TranscriptRecord {
            context_branch_id: Some(branch_id),
            event: TranscriptEvent::ToolCallFinished { output, .. },
            ..
        } if branch_id == "try-parser-fix"
            && output.data.as_ref().and_then(|data| data.get("warning")).and_then(serde_json::Value::as_str)
                == Some("Context restored, files were NOT reverted")
    ));
    assert!(matches!(
        &records[13].event,
        TranscriptEvent::ContextCheckout { branch_id, leaf_sequence }
            if branch_id == ROOT_CONTEXT_BRANCH_ID && *leaf_sequence == 4
    ));
    assert!(matches!(
        &records[14],
        TranscriptRecord {
            context_branch_id: None,
            event: TranscriptEvent::ContextExperimentReturned {
                branch_id,
                parent_branch_id,
                base_sequence,
                outcome,
                summary,
                next_action,
                had_writes,
            },
            ..
        } if branch_id == "try-parser-fix"
            && parent_branch_id == ROOT_CONTEXT_BRANCH_ID
            && *base_sequence == 4
            && outcome == "useful"
            && summary == "Parser path found the root cause"
            && next_action.as_deref() == Some("apply the fix on main")
            && *had_writes
    ));
    assert!(matches!(
        &records[15].event,
        TranscriptEvent::ContextNodeLifecycle { node_id, status }
            if node_id == "branch/try-parser-fix" && *status == ContextNodeStatus::Archived
    ));
    assert!(matches!(
        &records[16].event,
        TranscriptEvent::ContextNodeLifecycle { node_id, status }
            if node_id == "root" && *status == ContextNodeStatus::Active
    ));
    assert_eq!(recorder.current_context_branch_id(), None);
    assert!(recorder.active_context_experiment().is_none());

    let tree = transcript_projection::project_context_tree(&records).expect("project context tree");
    assert_eq!(tree.active_node_id().map(|id| id.as_str()), Some("root"));
    assert_eq!(
        tree.node(&ContextNodeId::new("branch/try-parser-fix").expect("node id"))
            .map(|node| &node.status),
        Some(&ContextNodeStatus::Archived)
    );

    let history = restore_session_history(&records).expect("restore history");
    assert!(matches!(
        history.last(),
        Some(HistoryItem::ContextSummary { text })
            if text.contains("Parser path found the root cause")
                && text.contains("files were NOT reverted")
    ));
}

#[test]
fn list_sessions_skips_session_started_only_transcripts() {
    let base_dir = std::env::temp_dir().join(format!(
        "letcode-transcript-list-empty-test-{}",
        unix_timestamp_ms()
    ));

    let mut empty = TranscriptRecorder::create(&base_dir).expect("create empty recorder");
    empty
        .record_session_started("gpt-test")
        .expect("record empty session start");

    let mut content = TranscriptRecorder::create(&base_dir).expect("create content recorder");
    content
        .record_session_started("gpt-test")
        .expect("record content session start");
    content
        .record_user_message("keep me")
        .expect("record user message");

    let sessions = list_sessions(&base_dir).expect("list sessions");

    assert_eq!(sessions.len(), 1);
    assert_eq!(sessions[0].session_id, content.session_id());
}

#[test]
fn list_sessions_prefers_latest_recorded_title() {
    let base_dir = std::env::temp_dir().join(format!(
        "letcode-transcript-list-title-test-{}",
        unix_timestamp_ms()
    ));

    let mut recorder = TranscriptRecorder::create(&base_dir).expect("create recorder");
    recorder
        .record_session_started("gpt-test")
        .expect("record session start");
    recorder
        .record_user_message("please help debug startup")
        .expect("record user message");
    recorder
        .record_session_title("Debug startup")
        .expect("record first title");
    recorder
        .record_session_title("Debug startup failure")
        .expect("record latest title");

    let sessions = list_sessions(&base_dir).expect("list sessions");

    assert_eq!(sessions.len(), 1);
    assert_eq!(sessions[0].title.as_deref(), Some("Debug startup failure"));
    assert_eq!(
        sessions[0].last_user_summary.as_deref(),
        Some("please help debug startup")
    );
}

#[test]
fn list_sessions_reports_latest_model_after_model_changes() {
    let base_dir = std::env::temp_dir().join(format!(
        "letcode-transcript-list-model-test-{}",
        unix_timestamp_ms()
    ));

    let mut recorder = TranscriptRecorder::create(&base_dir).expect("create recorder");
    recorder
        .record_session_started("gpt-test")
        .expect("record session start");
    recorder
        .record_model_changed("gpt-test", "gpt-test-mini")
        .expect("record model change");
    recorder
        .record_user_message("keep me")
        .expect("record user message");

    let sessions = list_sessions(&base_dir).expect("list sessions");

    assert_eq!(sessions.len(), 1);
    assert_eq!(sessions[0].model.as_deref(), Some("gpt-test-mini"));
}

#[test]
fn remove_empty_session_file_only_deletes_empty_transcripts() {
    let base_dir = std::env::temp_dir().join(format!(
        "letcode-transcript-remove-empty-test-{}",
        unix_timestamp_ms()
    ));

    let mut empty = TranscriptRecorder::create(&base_dir).expect("create empty recorder");
    empty
        .record_session_started("gpt-test")
        .expect("record empty session start");
    let empty_path = empty.path().to_path_buf();

    assert!(remove_empty_session_file(&empty_path).expect("remove empty session"));
    assert!(!empty_path.exists());

    let mut content = TranscriptRecorder::create(&base_dir).expect("create content recorder");
    content
        .record_session_started("gpt-test")
        .expect("record content session start");
    content
        .record_user_message("keep me")
        .expect("record user message");
    let content_path = content.path().to_path_buf();

    assert!(!remove_empty_session_file(&content_path).expect("keep content session"));
    assert!(content_path.exists());
}

#[test]
fn logical_checkpoint_branch_resolution_prefers_cursor_then_checkout_then_root() {
    assert_eq!(
        logical_checkpoint_branch_id(&[], None).expect("fresh root branch"),
        ROOT_CONTEXT_BRANCH_ID
    );

    let records = vec![
        TranscriptRecord {
            session_id: "s".into(),
            sequence: 1,
            timestamp_ms: 0,
            context_branch_id: None,
            event: TranscriptEvent::UserMessage {
                content: "root content".into(),
            },
        },
        TranscriptRecord {
            session_id: "s".into(),
            sequence: 2,
            timestamp_ms: 1,
            context_branch_id: None,
            event: TranscriptEvent::ContextBranchCreated {
                branch_id: "checked-out".into(),
                parent_branch_id: ROOT_CONTEXT_BRANCH_ID.into(),
                base_sequence: 1,
                label: None,
            },
        },
        TranscriptRecord {
            session_id: "s".into(),
            sequence: 3,
            timestamp_ms: 2,
            context_branch_id: None,
            event: TranscriptEvent::ContextCheckout {
                branch_id: "checked-out".into(),
                leaf_sequence: 1,
            },
        },
    ];

    assert_eq!(
        logical_checkpoint_branch_id(&records, None).expect("checkout fallback"),
        "checked-out"
    );
    assert_eq!(
        logical_checkpoint_branch_id(&records, Some(ROOT_CONTEXT_BRANCH_ID))
            .expect("explicit cursor"),
        ROOT_CONTEXT_BRANCH_ID
    );
}

#[test]
fn logical_checkpoint_uses_root_for_a_fresh_recorder() {
    let event = LogicalCheckpointEventV1 {
        schema_version: 1,
        checkpoint_id: "cp-1".into(),
        turn_id: 7,
        previous_segment_id: 0,
        segment_id: 1,
        previous_checkpoint_id: None,
        boundary_sequence: 4,
        context_scope_revision: 0,
        covered_source_spans: vec![
            LogicalCheckpointSourceSpanV1 {
                start_sequence: 2,
                end_sequence: 2,
            },
            LogicalCheckpointSourceSpanV1 {
                start_sequence: 4,
                end_sequence: 4,
            },
        ],
        retained_items: vec![LogicalCheckpointRetainedItemV1 {
            kind: LogicalCheckpointRetainedKindV1::UserRequirement,
            title: "Goal".into(),
            detail: "Keep protocol pairs".into(),
            audit_source: LogicalCheckpointAuditSourceV1::TranscriptSpan {
                start_sequence: 2,
                end_sequence: 2,
            },
        }],
    };
    assert_eq!(
        render_checkpoint_v1(&event).expect("render"),
        "[logical-checkpoint-v1]\n{\"schema_version\":1,\"checkpoint_id\":\"cp-1\",\"turn_id\":7,\"previous_segment_id\":0,\"segment_id\":1}\n[retained-items]\n{\"kind\":\"user_requirement\",\"title\":\"Goal\",\"detail\":\"Keep protocol pairs\",\"audit_source\":{\"type\":\"transcript_span\",\"start_sequence\":2,\"end_sequence\":2}}"
    );
    assert_eq!(
        render_checkpoint_continuation_v1(&event),
        "Resume the same user turn from logical checkpoint cp-1. Treat the retained checkpoint context above as authoritative; retired sources are audit-only and are not directly openable."
    );

    let base_dir = std::env::temp_dir().join(format!(
        "letcode-logical-checkpoint-{}",
        unix_timestamp_ms()
    ));
    let mut recorder = TranscriptRecorder::create(&base_dir).expect("recorder");
    recorder.record_session_started("test").expect("session");
    recorder.record_user_message("goal").expect("user");
    recorder
        .record_turn_started(TurnStartedEvent {
            turn_id: 7,
            intent: "i".into(),
            directive: "d".into(),
            validation_reminder: "v".into(),
        })
        .expect("turn");
    recorder
        .record_assistant_message("done")
        .expect("assistant");
    recorder
        .record_logical_checkpoint(event)
        .expect("checkpoint transaction");
    let records = read_records(recorder.path()).expect("committed records");
    assert!(
        matches!(records.last().map(|record| &record.event), Some(TranscriptEvent::LogicalCheckpoint(event)) if event.checkpoint_id == "cp-1")
    );
    assert_eq!(
        records
            .last()
            .and_then(|record| record.context_branch_id.as_deref()),
        Some(ROOT_CONTEXT_BRANCH_ID)
    );
    let snapshot = restore_runtime_snapshot(&records).expect("checkpoint restore");
    assert_eq!(snapshot.current_turn_id, Some(7));
    assert_eq!(snapshot.current_segment_id, Some(1));
    assert!(snapshot.active_history_items().iter().any(|item| matches!(item, HistoryItem::ContextSummary { text } if text.starts_with("[logical-checkpoint-v1]"))));
}

#[test]
fn public_compatibility_restores_reject_malformed_logical_checkpoints() {
    let records = vec![TranscriptRecord {
        session_id: "s".into(),
        sequence: 1,
        timestamp_ms: 0,
        context_branch_id: None,
        event: TranscriptEvent::LogicalCheckpoint(LogicalCheckpointEventV1 {
            schema_version: 0,
            checkpoint_id: "invalid".into(),
            turn_id: 1,
            previous_segment_id: 0,
            segment_id: 1,
            previous_checkpoint_id: None,
            boundary_sequence: 0,
            context_scope_revision: 0,
            covered_source_spans: Vec::new(),
            retained_items: Vec::new(),
        }),
    }];

    assert!(restore_session_history(&records).is_err());
    assert!(restore_compacted_conversation_messages(&records).is_err());
    assert!(restore_conversation_messages(&records).is_err());
    assert!(restore_session_protocol_frames(&records).is_err());
    assert!(restore_runtime_snapshot(&records).is_err());
}

#[test]
fn prepare_logical_checkpoint_is_deterministic_valid_and_non_persistent() {
    let base_dir = std::env::temp_dir().join(format!(
        "letcode-prepare-logical-checkpoint-{}",
        unix_timestamp_ms()
    ));
    let mut recorder = TranscriptRecorder::create(&base_dir).expect("recorder");
    recorder.record_session_started("test").expect("session");
    recorder
        .record_user_message("keep this requirement")
        .expect("user");
    recorder
        .record_turn_started(TurnStartedEvent {
            turn_id: 9,
            intent: "test".into(),
            directive: "verify preparation".into(),
            validation_reminder: String::new(),
        })
        .expect("turn");
    recorder
        .record_assistant_message("working")
        .expect("assistant");
    let before = read_records(recorder.path()).expect("records before prepare");

    let first = recorder
        .prepare_logical_checkpoint()
        .expect("first candidate");
    let second = recorder
        .prepare_logical_checkpoint()
        .expect("second candidate");

    assert_eq!(first.expected_journal_frontier, recorder.sequence);
    assert_eq!(first.event, second.event);
    assert!(first.event.checkpoint_id.starts_with("lcp-v1-"));
    assert_eq!(
        serde_json::to_value(read_records(recorder.path()).expect("records after prepare"))
            .expect("serialize records after prepare"),
        serde_json::to_value(&before).expect("serialize records before prepare")
    );
    transcript_projection::validate_logical_checkpoint_candidate(
        recorder.session_id(),
        &before,
        Some(ROOT_CONTEXT_BRANCH_ID.into()),
        recorder.sequence,
        recorder.sequence + 1,
        &first.event,
    )
    .expect("prepared event satisfies the Phase3a contract");
}

#[test]
fn prepare_logical_checkpoint_rejects_incomplete_or_inactive_input() {
    let base_dir = std::env::temp_dir().join(format!(
        "letcode-prepare-logical-checkpoint-invalid-{}",
        unix_timestamp_ms()
    ));
    let mut recorder = TranscriptRecorder::create(&base_dir).expect("recorder");
    recorder.record_session_started("test").expect("session");
    assert!(recorder.prepare_logical_checkpoint().is_err());
    recorder.record_user_message("goal").expect("user");
    recorder
        .record_turn_started(TurnStartedEvent {
            turn_id: 1,
            intent: "test".into(),
            directive: "reject incomplete tools".into(),
            validation_reminder: String::new(),
        })
        .expect("turn");
    recorder
        .record_assistant_tool_call_batch(
            None,
            vec![HistoryToolCall {
                call_id: "unfinished".into(),
                name: "read".into(),
                arguments_json: "{}".into(),
            }],
        )
        .expect("tool call");
    assert!(recorder.prepare_logical_checkpoint().is_err());
}
