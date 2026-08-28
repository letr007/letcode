use super::*;
use crate::config::ApiProtocol;
use crate::protocol_frames::{analyze_history_items, history_items_from_frames};
use crate::request_builder::{ModelRequestMetadata, RequestBuilderInput, build_request};
use crate::subagent::{StructuredSubagentResult, SubagentPool};
use crate::tool_names;
use crate::transcript::transcript_projection::{
    SessionContextCursor, project_runtime_restore_snapshot,
};
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
        reasoning_started_at: std::collections::HashMap::new(),
        active_turn_trackers: std::collections::HashMap::from([(
            ROOT_CONTEXT_BRANCH_ID.to_string(),
            super::recorder::ActiveTurnTracker::default(),
        )]),
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
fn expert_model_changes_require_durable_commit() {
    let calls = Arc::new(Mutex::new(Vec::new()));
    let mut recorder = recorder_with_sink(FailingSink {
        fail: FailPoint::Sync,
        calls: Arc::clone(&calls),
    });

    assert!(
        recorder
            .record_expert_model_changed("explorer", "provider/model")
            .is_err()
    );
    assert_eq!(*calls.lock().unwrap(), vec!["write", "flush", "sync"]);
    assert_eq!(recorder.health, RecorderHealth::Poisoned);
    assert_eq!(recorder.sequence, 0);
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
    let records = read_records(&path).unwrap();
    assert!(
        TranscriptRecorder::open_existing_with_records(&base_dir, recorder.session_id(), &records,)
            .is_err()
    );
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
        assert_eq!(recorder.active_turn_trackers.get("parent"), None);
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
fn interrupt_transaction_failure_leaves_sequence_events_and_tracker_unchanged() {
    let base_dir = journal_test_dir("interrupt-transaction-atomicity");
    let mut recorder = TranscriptRecorder::create(&base_dir).expect("create recorder");
    recorder
        .record_turn_started(TurnStartedEvent {
            turn_id: 1,
            intent: "interrupt".into(),
            directive: "cancel active work".into(),
            validation_reminder: String::new(),
        })
        .expect("start turn");
    recorder
        .record_tool_call_started("call-1", "shell__exec", json!({"command": "sleep 1"}))
        .expect("start tool call");
    let before_records = read_records(recorder.path()).expect("read records before failure");
    let before_tracker = recorder
        .unfinished_tool_calls_in_active_turn()
        .expect("tracker should be available");
    let before_sequence = recorder.sequence;
    let calls = Arc::new(Mutex::new(Vec::new()));
    recorder.replace_sink_for_test(Box::new(FailingSink {
        fail: FailPoint::Write,
        calls: Arc::clone(&calls),
    }));

    assert!(
        recorder
            .append_transaction(vec![
                (
                    TranscriptEvent::ToolCallCancelled {
                        call_id: "call-1".into(),
                        name: "shell__exec".into(),
                    },
                    None,
                ),
                (TranscriptEvent::TurnInterrupted { turn_id: Some(1) }, None,),
            ])
            .is_err()
    );

    assert_eq!(recorder.sequence, before_sequence);
    assert_eq!(
        recorder
            .unfinished_tool_calls_in_active_turn()
            .expect("tracker should remain available"),
        before_tracker
    );
    let after_records = read_records(recorder.path()).expect("read records after failure");
    assert_eq!(
        serde_json::to_value(after_records).expect("serialize records after failure"),
        serde_json::to_value(before_records).expect("serialize records before failure")
    );
    assert_eq!(*calls.lock().unwrap(), vec!["write"]);
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
                reviewer: None,
                approval: None,
                risk: None,
                reviewer_child_session_id: None,
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
            ..
        }) if call_id == "call-1"
            && output_json == r#"{"status":"cancelled","summary":"user cancelled"}"#
    ));
}

#[test]
fn open_existing_with_records_preserves_sequence_and_context_scope() {
    let base_dir = journal_test_dir("open-with-records");
    let mut recorder = TranscriptRecorder::create(&base_dir).expect("create recorder");
    recorder
        .record_context_branch_created("branch-1", ROOT_CONTEXT_BRANCH_ID, 0, None)
        .expect("create branch definition");
    recorder
        .append_metadata(TranscriptEvent::ContextExperimentStarted {
            branch_id: "branch-1".into(),
            parent_branch_id: ROOT_CONTEXT_BRANCH_ID.into(),
            base_sequence: 0,
        })
        .expect("start experiment");
    recorder.set_current_context_branch_id(Some("branch-1".into()));
    recorder
        .record_tool_execution_summary(ToolExecutionSummaryEvent {
            turn_id: 1,
            call_id: "call-write".into(),
            name: "fs__write".into(),
            status: "executed".into(),
            rejection: None,
            effect_kind: "write".into(),
            primary_path: Some("src/lib.rs".into()),
            command: None,
        })
        .expect("record write");
    let session_id = recorder.session_id().to_string();
    let path = recorder.path().to_path_buf();
    let records = read_records(&path).expect("load records");
    drop(recorder);

    let mut reopened =
        TranscriptRecorder::open_existing_with_records(&base_dir, &session_id, &records)
            .expect("open using records");
    assert_eq!(
        reopened.active_context_experiment(),
        Some(ActiveContextExperiment {
            branch_id: "branch-1".into(),
            parent_branch_id: ROOT_CONTEXT_BRANCH_ID.into(),
            base_sequence: 0,
            writes_observed: true,
        })
    );
    reopened
        .record_user_message("continued")
        .expect("append after records-backed open");

    assert_eq!(
        read_records(path)
            .expect("read appended records")
            .iter()
            .map(|record| record.sequence)
            .collect::<Vec<_>>(),
        vec![1, 2, 3, 4]
    );
}

#[test]
fn checkout_to_root_reconstructs_root_tracker_after_reopen() {
    let base_dir = journal_test_dir("checkout-root-tracker");
    let mut recorder = TranscriptRecorder::create(&base_dir).expect("create recorder");
    recorder
        .record_turn_started(TurnStartedEvent {
            turn_id: 1,
            intent: "root".into(),
            directive: "root turn".into(),
            validation_reminder: String::new(),
        })
        .expect("start root turn");
    recorder
        .record_turn_finalized(TurnFinalizedEvent {
            turn_id: 1,
            outcome: "completed".into(),
            tool_call_count: 0,
            continuation_count: 0,
            write_effects: 0,
            validation_effects: 0,
            failed_validation_effects: 0,
            validation_advisory_emitted: false,
        })
        .expect("finish root turn");
    let root_base = recorder.sequence;
    recorder
        .record_context_branch_created("branch-a", ROOT_CONTEXT_BRANCH_ID, root_base, None)
        .expect("create branch");
    recorder
        .record_context_checkout("branch-a", root_base)
        .expect("checkout branch");
    recorder.set_current_context_branch_id(Some("branch-a".into()));
    recorder
        .record_turn_started(TurnStartedEvent {
            turn_id: 2,
            intent: "branch".into(),
            directive: "branch turn".into(),
            validation_reminder: String::new(),
        })
        .expect("start branch turn");
    recorder
        .record_context_checkout(ROOT_CONTEXT_BRANCH_ID, root_base)
        .expect("checkout root");
    recorder.set_current_context_branch_id(None);
    recorder
        .record_turn_started(TurnStartedEvent {
            turn_id: 3,
            intent: "root again".into(),
            directive: "root turn after checkout".into(),
            validation_reminder: String::new(),
        })
        .expect("start root turn after checkout");
    recorder
        .record_tool_call_started("root-call", "fs__read", json!({}))
        .expect("start root tool");
    let session_id = recorder.session_id().to_string();
    drop(recorder);

    let mut reopened = TranscriptRecorder::open_existing(&base_dir, &session_id)
        .expect("reopen root checkout transcript");
    assert_eq!(reopened.current_context_branch_id(), None);
    assert_eq!(
        reopened
            .unfinished_tool_calls_in_active_turn()
            .expect("root tracker should be available"),
        vec![("root-call".into(), "fs__read".into())]
    );
}

#[test]
fn open_reconstructs_metadata_only_branch_tracker_from_parent_projection() {
    let base_dir = journal_test_dir("metadata-only-branch-tracker");
    let mut recorder = TranscriptRecorder::create(&base_dir).expect("create recorder");
    recorder
        .record_turn_started(TurnStartedEvent {
            turn_id: 1,
            intent: "root".into(),
            directive: "parent active turn".into(),
            validation_reminder: String::new(),
        })
        .expect("start parent turn");
    recorder
        .record_tool_call_started("parent-call", "fs__read", json!({}))
        .expect("start parent tool");
    let base_sequence = recorder.sequence;
    recorder
        .record_context_branch_created("metadata-only", ROOT_CONTEXT_BRANCH_ID, base_sequence, None)
        .expect("create metadata-only branch");
    let session_id = recorder.session_id().to_string();
    drop(recorder);

    let mut reopened = TranscriptRecorder::open_existing(&base_dir, &session_id)
        .expect("reopen metadata-only branch transcript");
    reopened.set_current_context_branch_id(Some("metadata-only".into()));
    assert_eq!(
        reopened
            .unfinished_tool_calls_in_active_turn()
            .expect("metadata-only tracker should be available"),
        vec![("parent-call".into(), "fs__read".into())]
    );
}

#[test]
fn open_rejects_malformed_branch_projection_without_tracker_fallback() {
    let base_dir = journal_test_dir("malformed-branch-projection");
    let mut recorder = TranscriptRecorder::create(&base_dir).expect("create recorder");
    recorder
        .append_transaction(vec![(
            TranscriptEvent::AssistantMessage {
                content: "scoped without a branch definition".into(),
            },
            Some("missing-branch".into()),
        )])
        .expect("write malformed scoped record");
    let session_id = recorder.session_id().to_string();
    drop(recorder);

    let error = match TranscriptRecorder::open_existing(&base_dir, &session_id) {
        Ok(_) => panic!("malformed branch projection must fail open"),
        Err(error) => error,
    };
    assert!(
        error.to_string().contains("branch")
            || error.to_string().contains("projection")
            || error.to_string().contains("scope")
    );
}

#[test]
fn open_existing_with_records_rejects_mismatched_session_records() {
    let base_dir = journal_test_dir("open-with-records-session-mismatch");
    let mut recorder = TranscriptRecorder::create(&base_dir).expect("create recorder");
    recorder
        .record_user_message("message")
        .expect("record message");
    let session_id = recorder.session_id().to_string();
    let mut records = read_records(recorder.path()).expect("load records");
    records[0].session_id = "other-session".into();
    drop(recorder);

    let result = TranscriptRecorder::open_existing_with_records(&base_dir, &session_id, &records);
    assert!(result.is_err());
    let error = result.err().expect("mismatched records must be rejected");
    assert!(error.to_string().contains("different session"));
}

#[test]
fn open_existing_with_records_rejects_stale_committed_frontier() {
    let base_dir = journal_test_dir("open-with-records-stale-frontier");
    let mut recorder = TranscriptRecorder::create(&base_dir).expect("create recorder");
    recorder
        .record_user_message("loaded")
        .expect("record loaded message");
    let session_id = recorder.session_id().to_string();
    let (records, fingerprint) =
        read_records_with_fingerprint(recorder.path()).expect("load records");
    recorder
        .record_assistant_message("appended later")
        .expect("append after load");
    drop(recorder);

    let result = TranscriptRecorder::open_existing_with_records_at_fingerprint(
        &base_dir,
        &session_id,
        &records,
        &fingerprint,
    );
    assert!(result.is_err());
    let error = result.err().expect("stale records must be rejected");
    assert!(
        error
            .to_string()
            .contains("changed after records were loaded")
    );
}

#[test]
fn legacy_linear_branch_adoption_scopes_future_records_without_topology_mutation() {
    let base_dir = journal_test_dir("legacy-linear-adoption");
    let mut recorder = TranscriptRecorder::create(&base_dir).expect("create recorder");
    recorder.record_user_message("root").expect("root message");
    recorder
        .record_context_branch_created("legacy-child", ROOT_CONTEXT_BRANCH_ID, 1, None)
        .expect("create legacy branch");
    recorder
        .append_metadata(TranscriptEvent::ContextExperimentStarted {
            branch_id: "legacy-child".into(),
            parent_branch_id: ROOT_CONTEXT_BRANCH_ID.into(),
            base_sequence: 1,
        })
        .expect("legacy experiment");
    let path = recorder.path().to_path_buf();
    let session_id = recorder.session_id().to_string();
    drop(recorder);

    let mut reopened = TranscriptRecorder::open_existing(&base_dir, &session_id).expect("reopen");
    let frontier = read_records(&path).expect("read frontier").len();
    reopened
        .adopt_legacy_linear_branch("legacy-child")
        .expect("adopt matching branch");
    assert_eq!(reopened.current_context_branch_id(), Some("legacy-child"));
    assert!(reopened.active_context_experiment().is_none());
    let reconstructed = TranscriptRecorder::open_existing(&base_dir, &session_id)
        .expect("reconstruct legacy state");
    assert_eq!(reconstructed.current_context_branch_id(), None);
    assert_eq!(
        reconstructed.active_context_experiment(),
        Some(ActiveContextExperiment {
            branch_id: "legacy-child".into(),
            parent_branch_id: ROOT_CONTEXT_BRANCH_ID.into(),
            base_sequence: 1,
            writes_observed: false,
        })
    );
    reopened
        .record_user_message("continued")
        .expect("user append");
    reopened
        .record_assistant_message("continued response")
        .expect("assistant append");
    reopened
        .record_turn_finalized(TurnFinalizedEvent {
            turn_id: 1,
            outcome: "completed".into(),
            tool_call_count: 0,
            continuation_count: 0,
            write_effects: 0,
            validation_effects: 0,
            failed_validation_effects: 0,
            validation_advisory_emitted: false,
        })
        .expect("finalization append");

    let records = read_records(&path).expect("read adopted journal");
    assert_eq!(records.len(), frontier + 3, "adoption must not append");
    assert!(
        records[frontier..]
            .iter()
            .all(|record| record.context_branch_id.as_deref() == Some("legacy-child"))
    );
    assert!(records[frontier..].iter().all(|record| !matches!(
        record.event,
        TranscriptEvent::ContextBranchCreated { .. }
            | TranscriptEvent::ContextCheckout { .. }
            | TranscriptEvent::ContextExperimentReturned { .. }
    )));

    let reopened_again =
        TranscriptRecorder::open_existing(&base_dir, &session_id).expect("reopen again");
    assert_eq!(
        reopened_again.current_context_branch_id(),
        None,
        "open remains branch-neutral"
    );
}

#[test]
fn legacy_linear_branch_adoption_rejects_unreturned_experiment_on_another_branch_without_mutation()
{
    let base_dir = journal_test_dir("legacy-linear-adoption-mismatch");
    let mut recorder = TranscriptRecorder::create(&base_dir).expect("create recorder");
    recorder
        .append_metadata(TranscriptEvent::ContextExperimentStarted {
            branch_id: "legacy-child".into(),
            parent_branch_id: ROOT_CONTEXT_BRANCH_ID.into(),
            base_sequence: 0,
        })
        .expect("legacy experiment");
    let path = recorder.path().to_path_buf();
    let session_id = recorder.session_id().to_string();
    drop(recorder);

    let mut reopened = TranscriptRecorder::open_existing(&base_dir, &session_id).expect("reopen");
    let before = read_records(&path).expect("read before");
    let error = reopened
        .adopt_legacy_linear_branch(ROOT_CONTEXT_BRANCH_ID)
        .expect_err("root must reject child experiment");
    assert!(error.to_string().contains("main"));
    assert!(error.to_string().contains("legacy-child"));
    assert_eq!(reopened.current_context_branch_id(), None);
    assert!(reopened.active_context_experiment().is_some());
    let after = read_records(&path).expect("read after");
    assert_eq!(after.len(), before.len());
    assert_eq!(
        after
            .iter()
            .map(|record| record.sequence)
            .collect::<Vec<_>>(),
        before
            .iter()
            .map(|record| record.sequence)
            .collect::<Vec<_>>()
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
fn restore_session_history_preserves_multimodal_tool_output_images() {
    let image = crate::user_content::UserImageAttachment::from_bytes(
        "pixel.png",
        "image/png",
        b"image-bytes",
    );
    let output = ToolResult::ok(
        "fs__read",
        json!({"path": "pixel.png", "kind": "image", "mime": "image/png"}),
    )
    .with_images(vec![image.clone()]);
    let records = vec![
        TranscriptRecord {
            session_id: "s".into(),
            sequence: 1,
            timestamp_ms: 0,
            context_branch_id: None,
            event: TranscriptEvent::ToolCallStarted {
                call_id: "call-image".into(),
                name: "fs__read".into(),
                args: json!({"path": "pixel.png", "offset": 1, "limit": 10}),
            },
        },
        TranscriptRecord {
            session_id: "s".into(),
            sequence: 2,
            timestamp_ms: 1,
            context_branch_id: None,
            event: TranscriptEvent::ToolCallFinished {
                call_id: "call-image".into(),
                name: "fs__read".into(),
                ok: true,
                output,
            },
        },
    ];

    let history = restore_session_history(&records).expect("restore image tool output");
    let HistoryItem::ToolOutput {
        output_json,
        images,
        ..
    } = &history[1]
    else {
        panic!("expected restored tool output");
    };
    assert_eq!(images, &[image]);
    assert!(!output_json.contains("data:image/png;base64,"));
    assert!(!output_json.contains("\"images\""));
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
    let child_session_id = child.session_id().to_string();
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
fn restore_job_board_derives_active_state_from_child_transcript() {
    let base_dir = std::env::temp_dir().join(format!(
        "letcode-transcript-active-job-board-test-{}",
        unix_timestamp_ms()
    ));
    let mut parent = TranscriptRecorder::create(&base_dir).expect("create parent recorder");
    let parent_session_id = parent.session_id().to_string();
    let child_dir = child_sessions_dir(&base_dir);
    let mut child = TranscriptRecorder::create(&child_dir).expect("create child recorder");
    let child_session_id = child.session_id().to_string();
    parent
        .record_subagent_started(
            "run-active",
            &parent_session_id,
            "turn-1",
            &child_session_id,
            "fixer",
            "apply patch",
            7,
        )
        .expect("register child");
    child
        .record_subagent_lifecycle(
            "run-active",
            &parent_session_id,
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

    let parent_records = read_records(base_dir.join(format!("{parent_session_id}.jsonl")))
        .expect("read parent records");
    let job_board = restore_job_board(&base_dir, &parent_records).expect("derive active board");
    assert_eq!(job_board.len(), 1);
    assert!(job_board[0].active);
    assert_eq!(job_board[0].child_session_id, child_session_id);
    assert_eq!(job_board[0].status, "running");

    let projected = project_subagent_jobs(&base_dir, &parent_records).expect("project jobs");
    assert_eq!(projected.len(), 1);
    assert_eq!(projected[0].run_id, "run-active");
    assert_eq!(projected[0].pool_ordinal, 7);
}

#[test]
fn project_subagent_jobs_ignores_results_from_unowned_child_sessions() {
    let base_dir = std::env::temp_dir().join(format!(
        "letcode-transcript-unowned-job-test-{}",
        unix_timestamp_ms()
    ));
    let mut parent = TranscriptRecorder::create(&base_dir).expect("create parent recorder");
    let parent_session_id = parent.session_id().to_string();
    parent
        .record_subagent_result(
            "run-unowned",
            &parent_session_id,
            "turn-1",
            "missing-child",
            "explorer",
            "completed",
            "foreign result",
        )
        .expect("record result");
    let parent_records = read_records(parent.path()).expect("read parent records");

    let projected = project_subagent_jobs(&base_dir, &parent_records).expect("project jobs");
    assert!(projected.is_empty());
}

#[test]
fn project_subagent_jobs_requires_started_ownership_even_when_child_file_exists() {
    let base_dir = std::env::temp_dir().join(format!(
        "letcode-transcript-unowned-existing-child-test-{}",
        unix_timestamp_ms()
    ));
    let mut parent = TranscriptRecorder::create(&base_dir).expect("create parent recorder");
    let parent_session_id = parent.session_id().to_string();
    let child_dir = child_sessions_dir(&base_dir);
    let child = TranscriptRecorder::create(&child_dir).expect("create child recorder");
    let child_session_id = child.session_id().to_string();
    parent
        .record_subagent_result(
            "run-unowned-existing",
            &parent_session_id,
            "turn-1",
            &child_session_id,
            "explorer",
            "completed",
            "foreign result",
        )
        .expect("record result");
    let parent_records = read_records(parent.path()).expect("read parent records");

    let projected = project_subagent_jobs(&base_dir, &parent_records).expect("project jobs");
    assert!(projected.is_empty());
    assert!(SubagentPool::child_sessions(&base_dir, &parent_records).is_empty());
}

#[test]
fn read_records_accepts_legacy_prompt_composition_shapes() {
    let base_dir = std::env::temp_dir().join(format!(
        "letcode-transcript-legacy-prompt-composition-test-{}",
        unix_timestamp_ms()
    ));
    fs::create_dir_all(&base_dir).expect("create temp dir");
    let path = base_dir.join("legacy.jsonl");
    let telemetry = |sequence: u64, prompt_composition: serde_json::Value| {
        let mut value = serde_json::to_value(TranscriptRecord {
            session_id: "s".into(),
            sequence,
            timestamp_ms: sequence as u128,
            context_branch_id: None,
            event: TranscriptEvent::LlmRequestTelemetry {
                version: 6,
                logical_request_id: format!("request-{sequence}"),
                turn_id: 1,
                iteration: 0,
                attempt: 1,
                phase: "prepared".into(),
                error_class: None,
                model: "test".into(),
                protocol: "responses".into(),
                context_window_tokens: 1_000_000,
                input_budget_tokens: 700_000,
                estimated_request_tokens: 100,
                estimated_prelude_tokens: 10,
                estimated_protected_tokens: 10,
                protected_safe_ceiling_tokens: 0,
                protected_reserve_tokens: 0,
                estimated_unaddressable_protected_tokens: 0,
                estimated_retained_history_tokens: 10,
                estimated_tools_tokens: 10,
                estimated_evidence_tokens: 0,
                estimated_required_fallback_tokens: 0,
                original_history_items: 1,
                retained_history_items: 1,
                dropped_history_items: 0,
                selected_evidence_items: 0,
                dropped_evidence_items: 0,
                selected_evidence_ids: Vec::new(),
                evidence_fingerprint: String::new(),
                truncated: false,
                prompt_segment_count: 1,
                prompt_contributor_count: 1,
                prompt_composition: Vec::new(),
                prompt_stable_prefix_hash: None,
                cache_first_volatile_index: None,
                plan_total_prompt_tokens: 10,
                plan_stable_prompt_tokens: 10,
                plan_volatile_prompt_tokens: 0,
                plan_cacheable_prefix_tokens: 10,
                plan_stable_after_boundary_tokens: 0,
                cache_configured: false,
                cache_hint_serialized: false,
                cache_retention_sent: None,
                cache_stable_prefix_segments: 0,
                cache_stable_prompt_tokens: 10,
                cache_volatile_prompt_tokens: 0,
                cacheable_prefix_tokens: 0,
                cache_stable_after_boundary_tokens: 0,
                tool_call_count_before: 0,
                tool_definitions_count: 0,
                local_prefix_fingerprint: None,
                routing_key: None,
                provider_cached_tokens: None,
                provider_input_tokens: None,
                provider_output_tokens: None,
                provider_total_tokens: None,
                provider_response_id: None,
                adjacent_lcp_units: None,
                adjacent_lcp_bytes: None,
                adjacent_lcp_estimated_tokens: None,
                current_unit_count: 0,
                first_breaker: None,
                cohort_comparable: false,
                cohort_changed: false,
                usage_completeness: "usage_missing".into(),
                cache_write_tokens: None,
            },
        })
        .expect("serialize telemetry");
        value
            .as_object_mut()
            .expect("telemetry object")
            .insert("prompt_composition".into(), prompt_composition);
        value
    };
    fs::write(
        &path,
        format!(
            "{}\n{}\n",
            telemetry(1, json!([])),
            telemetry(
                2,
                json!([{
                    "key": "SkillMaterial:Developer:skill_catalog",
                    "role": "developer",
                    "source_label": "skill_catalog",
                    "tokens": 42,
                    "segments": 1
                }])
            )
        ),
    )
    .expect("write transcript");

    let records = read_records(&path).expect("read legacy telemetry");
    match &records[1].event {
        TranscriptEvent::LlmRequestTelemetry {
            prompt_composition, ..
        } => {
            assert_eq!(
                prompt_composition[0].category,
                "SkillMaterial:Developer:skill_catalog"
            );
            assert_eq!(prompt_composition[0].estimated_tokens, 42);
            assert_eq!(prompt_composition[0].segments, 1);
        }
        other => panic!("unexpected event: {other:?}"),
    }
}

#[test]
fn structured_subagent_result_and_evidence_commit_atomically() {
    let base_dir = journal_test_dir("subagent-result-transaction");
    let mut recorder = TranscriptRecorder::create(&base_dir).expect("create recorder");
    let session_id = recorder.session_id().to_string();
    let structured = StructuredSubagentResult {
        status: "completed".into(),
        summary: "done".into(),
        malformed: false,
        findings: Vec::new(),
        files_read: Vec::new(),
        files_changed: Vec::new(),
        commands_run: Vec::new(),
        validation: Vec::new(),
        blockers: Vec::new(),
        next_steps: Vec::new(),
        run_id: "run-transaction".into(),
        child_session_id: "child-transaction".into(),
        raw_excerpt: None,
    };
    recorder
        .record_subagent_result_structured(
            "run-transaction",
            &session_id,
            "turn-1",
            "child-transaction",
            "explorer",
            "completed",
            "done",
            Some(structured),
        )
        .expect("record structured result");

    let records = read_records(recorder.path()).expect("read records");
    assert_eq!(records.len(), 2);
    assert!(matches!(
        records[0].event,
        TranscriptEvent::SubagentResult { .. }
    ));
    assert!(matches!(
        &records[1].event,
        TranscriptEvent::Evidence {
            source: EvidenceSource::Subagent { run_id, .. },
            ..
        } if run_id == "run-transaction"
    ));
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
                state: AutoContinueState { enabled: true },
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
            event: TranscriptEvent::AssistantToolCallBatch {
                text: None,
                reasoning_content: None,
                reasoning_wire: None,
                calls,
            },
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
    let protected = &live.snapshot.compaction.protected_frame_ids;
    assert_eq!(protected.len(), 3);
    assert!(live.snapshot.frames.iter().any(|frame| {
        protected.contains(&frame.id)
            && frame.kind == crate::runtime_context::RuntimeFrameKind::ToolCall
    }));
    let model = ModelRequestMetadata {
        supports_tools: true,
        ..Default::default()
    };
    for protocol in [ApiProtocol::Responses, ApiProtocol::Completions] {
        assert!(
            build_request(RequestBuilderInput {
                protocol,
                provider: None,
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
    file.write_all(&final_line.as_bytes()[partial_len..])
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
            provider: None,
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
fn list_sessions_persists_and_reuses_sidecar_index() {
    let base_dir = std::env::temp_dir().join(format!(
        "letcode-transcript-list-index-test-{}",
        unix_timestamp_ms()
    ));

    let mut recorder = TranscriptRecorder::create(&base_dir).expect("create recorder");
    recorder
        .record_session_started("gpt-test")
        .expect("record session start");
    recorder
        .record_user_message("indexed session")
        .expect("record user message");
    recorder
        .record_session_title("Indexed")
        .expect("record title");

    assert!(
        !base_dir.join("sessions-index.json").exists(),
        "append must not create the session index sidecar"
    );

    let first = list_sessions(&base_dir).expect("first list");
    assert_eq!(first.len(), 1);
    assert_eq!(first[0].title.as_deref(), Some("Indexed"));
    assert!(
        base_dir.join("sessions-index.json").is_file(),
        "sidecar index should be written"
    );

    let second = list_sessions(&base_dir).expect("second list hits index");
    assert_eq!(first, second);

    std::fs::write(base_dir.join("sessions-index.json"), b"{}").expect("invalidate sidecar");
    let appended_during_scan = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let append_once = Arc::clone(&appended_during_scan);
    let transcript_path = recorder.path().to_path_buf();
    let during_scan =
        super::session_index::list_sessions_with_index(&base_dir, |path, session_id| {
            let summary = summarize_session_file(path, session_id)?;
            if !append_once.swap(true, std::sync::atomic::Ordering::SeqCst) {
                recorder
                    .record_session_title("Concurrent")
                    .expect("append during sidecar rebuild");
            }
            Ok(summary)
        })
        .expect("list while transcript advances");
    assert_eq!(during_scan[0].title.as_deref(), Some("Indexed"));
    let after_concurrent_append =
        list_sessions(&base_dir).expect("stale scan stamp forces rebuild");
    assert_eq!(
        after_concurrent_append[0].title.as_deref(),
        Some("Concurrent")
    );

    let path = transcript_path;
    let stale_index =
        std::fs::read(base_dir.join("sessions-index.json")).expect("read existing sidecar");
    recorder
        .record_session_title("Updated")
        .expect("append updated title");
    assert_eq!(
        std::fs::read(base_dir.join("sessions-index.json")).expect("read sidecar after append"),
        stale_index,
        "append must not update the session index sidecar"
    );
    let after_append = list_sessions(&base_dir).expect("list after append");
    assert_eq!(after_append[0].title.as_deref(), Some("Updated"));

    // Stale stamp forces a rescan when the transcript changes outside the recorder.

    let mut file = std::fs::OpenOptions::new()
        .append(true)
        .open(&path)
        .expect("open transcript");
    use std::io::Write;
    file.write_all(b"\n").expect("touch file");
    drop(file);

    let third = list_sessions(&base_dir).expect("list after stamp change");
    assert_eq!(third.len(), 1);
    assert_eq!(third[0].session_id, recorder.session_id());
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

#[cfg(any())]
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

#[cfg(any())]
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
            None,
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

#[cfg(test)]
mod compaction_legacy_schema_tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn recorder_rejects_legacy_compaction_shape_while_legacy_jsonl_replays() {
        let base_dir = std::env::temp_dir().join(format!(
            "letcode-compaction-boundary-{}",
            unix_timestamp_ms()
        ));
        let mut recorder = TranscriptRecorder::create(&base_dir).expect("recorder");
        recorder.record_session_started("test").expect("session");
        recorder.record_user_message("request").expect("user");
        recorder
            .record_assistant_tool_call_batch(
                None,
                None,
                None,
                vec![HistoryToolCall {
                    call_id: "call-1".into(),
                    name: "read".into(),
                    arguments_json: "{}".into(),
                }],
            )
            .expect("tool call");
        recorder
            .append(TranscriptEvent::ToolCallFinished {
                call_id: "call-1".into(),
                name: "read".into(),
                ok: true,
                output: ToolResult {
                    ok: true,
                    tool: "read".into(),
                    data: Some(serde_json::json!("ok")),
                    images: Vec::new(),
                    error: None,
                },
            })
            .expect("tool result");
        let error = recorder
            .record_context_compaction(ContextCompactionEvent::succeeded("summary", 2))
            .expect_err("new recorder appends reject legacy compaction fields");
        assert!(error.to_string().contains("tail_start_index"));
    }

    #[test]
    fn modern_compaction_replays_from_a_stable_first_kept_entry_id() {
        let event = ContextCompactionEvent::succeeded_at("summary", Some("raw:2".into()));
        let serialized = serde_json::to_value(&event).expect("serialize modern event");
        assert_eq!(serialized["first_kept_entry_id"], "raw:2");
        assert!(serialized.get("tail_start_index").is_none());
        assert!(serialized.get("checkpoint").is_none());

        let records = vec![
            TranscriptRecord {
                session_id: "modern".into(),
                sequence: 1,
                timestamp_ms: 0,
                context_branch_id: None,
                event: TranscriptEvent::UserMessage {
                    content: UserMessageContent::from("retired"),
                },
            },
            TranscriptRecord {
                session_id: "modern".into(),
                sequence: 2,
                timestamp_ms: 0,
                context_branch_id: None,
                event: TranscriptEvent::AssistantMessage {
                    content: "kept".into(),
                },
            },
            TranscriptRecord {
                session_id: "modern".into(),
                sequence: 3,
                timestamp_ms: 0,
                context_branch_id: None,
                event: TranscriptEvent::ContextCompaction(event),
            },
        ];

        let restored = restore_session_history(&records).expect("modern compaction replays");
        assert!(matches!(
            restored.as_slice(),
            [HistoryItem::ContextSummary { text }, HistoryItem::AssistantText { text: kept }]
                if text == "summary" && kept == "kept"
        ));
    }

    #[test]
    fn legacy_compaction_jsonl_replays_tolerantly() {
        let raw = json!({
            "session_id": "legacy",
            "sequence": 3,
            "timestamp_ms": 0,
            "kind": "context_compaction",
            "summary": "legacy summary",
            "tail_start_index": 1
        });
        let compacted: TranscriptRecord =
            serde_json::from_value(raw).expect("legacy record deserializes");
        let records = vec![
            TranscriptRecord {
                session_id: "legacy".into(),
                sequence: 1,
                timestamp_ms: 0,
                context_branch_id: None,
                event: TranscriptEvent::UserMessage {
                    content: UserMessageContent::from("old"),
                },
            },
            TranscriptRecord {
                session_id: "legacy".into(),
                sequence: 2,
                timestamp_ms: 0,
                context_branch_id: None,
                event: TranscriptEvent::AssistantMessage {
                    content: "reply".into(),
                },
            },
            compacted,
        ];

        let restored = restore_session_history(&records).expect("legacy compaction replays");
        assert!(matches!(
            restored.as_slice(),
            [HistoryItem::ContextSummary { text }, HistoryItem::AssistantText { text: reply }]
                if text == "legacy summary" && reply == "reply"
        ));
    }
}

#[test]
fn expert_model_changes_restore_latest_route_per_agent() {
    let records = vec![
        TranscriptRecord {
            session_id: "s".into(),
            sequence: 1,
            timestamp_ms: 0,
            context_branch_id: None,
            event: TranscriptEvent::ExpertModelChanged {
                agent_name: "explorer".into(),
                model: "p/first".into(),
            },
        },
        TranscriptRecord {
            session_id: "s".into(),
            sequence: 2,
            timestamp_ms: 1,
            context_branch_id: None,
            event: TranscriptEvent::ExpertModelChanged {
                agent_name: "reviewer".into(),
                model: "p/reviewer".into(),
            },
        },
        TranscriptRecord {
            session_id: "s".into(),
            sequence: 3,
            timestamp_ms: 2,
            context_branch_id: None,
            event: TranscriptEvent::ExpertModelChanged {
                agent_name: "explorer".into(),
                model: "p/latest".into(),
            },
        },
    ];

    let restored = restore_latest_expert_models(&records);
    assert_eq!(
        restored.get("explorer").map(String::as_str),
        Some("p/latest")
    );
    assert_eq!(
        restored.get("reviewer").map(String::as_str),
        Some("p/reviewer")
    );
}

#[test]
fn expert_model_restore_for_cursor_ignores_sibling_branch_updates() {
    let records = vec![
        TranscriptRecord {
            session_id: "s".into(),
            sequence: 1,
            timestamp_ms: 0,
            context_branch_id: None,
            event: TranscriptEvent::ExpertModelChanged {
                agent_name: "explorer".into(),
                model: "p/root".into(),
            },
        },
        TranscriptRecord {
            session_id: "s".into(),
            sequence: 2,
            timestamp_ms: 1,
            context_branch_id: None,
            event: TranscriptEvent::ContextBranchCreated {
                branch_id: "left".into(),
                parent_branch_id: ROOT_CONTEXT_BRANCH_ID.into(),
                base_sequence: 1,
                label: None,
            },
        },
        TranscriptRecord {
            session_id: "s".into(),
            sequence: 3,
            timestamp_ms: 2,
            context_branch_id: Some("left".into()),
            event: TranscriptEvent::ExpertModelChanged {
                agent_name: "explorer".into(),
                model: "p/left".into(),
            },
        },
        TranscriptRecord {
            session_id: "s".into(),
            sequence: 4,
            timestamp_ms: 3,
            context_branch_id: None,
            event: TranscriptEvent::ContextBranchCreated {
                branch_id: "right".into(),
                parent_branch_id: ROOT_CONTEXT_BRANCH_ID.into(),
                base_sequence: 1,
                label: None,
            },
        },
        TranscriptRecord {
            session_id: "s".into(),
            sequence: 5,
            timestamp_ms: 4,
            context_branch_id: Some("right".into()),
            event: TranscriptEvent::ExpertModelChanged {
                agent_name: "explorer".into(),
                model: "p/right".into(),
            },
        },
    ];

    let restored = restore_latest_expert_models_for_cursor(
        "s",
        &records,
        SessionContextCursor {
            branch_id: Some("left".into()),
            leaf_sequence: None,
        },
    )
    .expect("left branch projection");

    assert_eq!(restored.get("explorer").map(String::as_str), Some("p/left"));
}
