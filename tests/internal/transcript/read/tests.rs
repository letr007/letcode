use super::*;
use std::io::Write;
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

use crate::transcript::{
    JOURNAL_SCHEMA_VERSION, JournalRecordEnvelope, TranscriptAssistantTurn, TranscriptEvent,
    TranscriptRecord, TranscriptRecorder, journal_scope_for, read_records_allow_partial_tail,
    read_resumable_records_with_fingerprint, serialize_journal_record,
};

fn journal_line(session_id: &str, sequence: u64, event: TranscriptEvent) -> String {
    let record = TranscriptRecord {
        session_id: session_id.into(),
        sequence,
        timestamp_ms: sequence as u128,
        context_branch_id: None,
        event,
    };
    let envelope = JournalRecordEnvelope {
        schema_version: JOURNAL_SCHEMA_VERSION,
        event_id: format!("{session_id}:{sequence}"),
        scope: journal_scope_for(&record),
        base_revision: sequence - 1,
        resulting_revision: sequence,
        transaction_id: None,
        transaction_index: None,
        transaction_count: None,
        record,
    };
    String::from_utf8(serialize_journal_record(&envelope).expect("serialize journal record"))
        .expect("journal record is utf8")
}

fn user_line(session_id: &str, sequence: u64) -> String {
    journal_line(
        session_id,
        sequence,
        TranscriptEvent::UserMessage {
            content: crate::user_content::UserMessageContent::new(
                format!("message-{sequence}"),
                Vec::new(),
            ),
        },
    )
}

fn assistant_line(session_id: &str, sequence: u64, text: &str) -> String {
    journal_line(
        session_id,
        sequence,
        TranscriptEvent::AssistantTurn(TranscriptAssistantTurn {
            text: Some(text.into()),
            reasoning_content: None,
            replay: None,
            calls: Vec::new(),
        }),
    )
}

fn temp_dir() -> PathBuf {
    std::env::temp_dir().join(format!(
        "letcode-transcript-read-test-{}",
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system time is valid")
            .as_nanos()
    ))
}

/// A journal whose match is not its last record.
fn journal_with_two_prompts() -> (PathBuf, PathBuf) {
    let dir = temp_dir();
    let mut recorder = TranscriptRecorder::create(&dir).expect("create transcript");
    recorder
        .record_session_started("test/model")
        .expect("record session start");
    recorder
        .record_user_message("first")
        .expect("record prompt");
    recorder
        .record_user_message("second")
        .expect("record prompt");
    let path = recorder.path().to_path_buf();
    drop(recorder);
    (dir, path)
}

fn is_user_message(record: &TranscriptRecord) -> bool {
    matches!(&record.event, TranscriptEvent::UserMessage { .. })
}

#[test]
fn answers_from_a_record_before_the_end_of_the_journal() {
    let (dir, path) = journal_with_two_prompts();
    let total = read_records(&path).expect("read the journal").len();
    let mut seen = 0usize;

    let matched = any_record_where(&path, |record| {
        seen += 1;
        is_user_message(record)
    })
    .expect("scan the journal");

    assert!(matched);
    assert!(
        seen < total,
        "stopped early: read {seen} of {total} records"
    );
    let _ = fs::remove_dir_all(dir);
}

#[test]
fn falls_back_to_the_full_read_when_nothing_matches() {
    let (dir, path) = journal_with_two_prompts();
    assert!(!any_record_where(&path, |_| false).expect("scan the journal"));
    let _ = fs::remove_dir_all(dir);
}

#[test]
fn reports_a_missing_journal_like_the_full_read() {
    let dir = temp_dir();
    let path = dir.join("absent.jsonl");
    assert!(any_record_where(&path, |_| true).is_err());
    let _ = fs::remove_dir_all(dir);
}

#[test]
fn a_match_before_a_torn_tail_answers_and_a_missing_match_still_reports_it() {
    let (dir, path) = journal_with_two_prompts();
    let mut file = fs::OpenOptions::new()
        .append(true)
        .open(&path)
        .expect("open the journal");
    file.write_all(br#"{"schema_version":2,"event_i"#)
        .expect("write a torn record");
    drop(file);

    assert!(
        any_record_where(&path, is_user_message).expect("scan the journal"),
        "the answer is reached before the torn tail"
    );
    assert!(
        any_record_where(&path, |_| false).is_err(),
        "a full read reports the torn tail"
    );
    let _ = fs::remove_dir_all(dir);
}

mod integration_tests {
    use super::*;

    /// A torn assistant tail is tolerated only by dropping the record entirely.
    #[test]
    fn incomplete_assistant_tail_is_rejected_strictly_and_dropped_by_the_live_reader() {
        let dir = temp_dir();
        fs::create_dir_all(&dir).expect("create temp dir");
        let path = dir.join("s.jsonl");
        let torn = assistant_line("s", 3, "unfinished");
        let content = format!(
            "{}\n{}\n{}",
            user_line("s", 1),
            user_line("s", 2),
            &torn[..torn.len() - 2]
        );
        fs::write(&path, content).expect("write the journal");

        assert!(
            read_records(&path).is_err(),
            "a strict read reports the torn assistant tail"
        );
        let live = read_records_allow_partial_tail(&path).expect("live read drops the torn tail");
        assert_eq!(
            live.len(),
            2,
            "the incomplete message contributes no record"
        );
        assert!(
            !live
                .iter()
                .any(|record| matches!(record.event, TranscriptEvent::AssistantTurn(_))),
            "the dropped tail is not projected as an assistant turn"
        );
        let _ = fs::remove_dir_all(dir);
    }

    /// A fully written record without its delimiter reads as finished; only append-open refuses it.
    #[test]
    fn undelimited_assistant_tail_is_read_but_blocks_resume_append() {
        let dir = temp_dir();
        fs::create_dir_all(&dir).expect("create temp dir");
        let path = dir.join("s.jsonl");
        let content = format!(
            "{}\n{}",
            user_line("s", 1),
            assistant_line("s", 2, "answer")
        );
        fs::write(&path, content).expect("write the journal");

        assert_eq!(
            read_records(&path).expect("strict read").len(),
            2,
            "a complete record without its delimiter is still a record"
        );
        assert_eq!(
            read_records_allow_partial_tail(&path)
                .expect("live read")
                .len(),
            2
        );
        assert!(
            read_resumable_records_with_fingerprint(&path).is_ok(),
            "the resume load sees the complete record"
        );
        let error = match TranscriptRecorder::open_existing(&dir, "s") {
            Ok(_) => panic!("append-open requires the delimiter"),
            Err(error) => error,
        };
        assert!(
            error.to_string().contains("append delimiter"),
            "unexpected error: {error}"
        );
        let _ = fs::remove_dir_all(dir);
    }
}
