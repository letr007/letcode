use super::*;

fn child(id: &str) -> ChildSessionSummary {
    ChildSessionSummary {
        parent_session_id: "parent".into(),
        parent_run_id: "run".into(),
        child_session_id: id.into(),
        agent_name: "explorer".into(),
        status: "done".into(),
        summary: String::new(),
        timestamp_ms: 0,
        pool_ordinal: 1,
    }
}

use crate::transcript::{
    JOURNAL_SCHEMA_VERSION, JournalRecordEnvelope, TranscriptAssistantTurn, TranscriptEvent,
    TranscriptRecord, journal_scope_for, serialize_journal_record,
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

fn test_dir(name: &str) -> std::path::PathBuf {
    std::env::temp_dir().join(format!(
        "letcode-child-view-{name}-{}",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("system time is valid")
            .as_nanos()
    ))
}

/// Write a child journal whose final line is `tail` without a trailing newline.
fn write_child_journal(dir: &Path, child_id: &str, tail: &str) {
    let child_dir = child_sessions_dir(dir);
    std::fs::create_dir_all(&child_dir).expect("create child dir");
    let content = format!(
        "{}\n{}\n{}",
        journal_line(
            child_id,
            1,
            TranscriptEvent::SessionStarted {
                model: "test".into()
            }
        ),
        journal_line(
            child_id,
            2,
            TranscriptEvent::UserMessage {
                content: crate::user_content::UserMessageContent::new("question", Vec::new()),
            }
        ),
        tail
    );
    std::fs::write(child_dir.join(format!("{child_id}.jsonl")), content)
        .expect("write child journal");
}

fn project_child(dir: &Path, child_id: &str) -> ChildViewProjection {
    let mut summary = child(child_id);
    summary.status = "running".into();
    project_child_session_view_with_children(
        dir,
        "parent".into(),
        vec![summary],
        crate::command::ChildNavigation::First,
        None,
        &ChildLiveText::default(),
    )
    .expect("project the child view")
    .expect("a child to view")
}

/// The from-file child view drops a torn assistant tail instead of showing it as in-progress.
#[test]
fn an_incomplete_assistant_tail_contributes_no_in_progress_message() {
    let dir = test_dir("incomplete-tail");
    let child_id = "child-a";
    let torn = assistant_line(child_id, 3, "unfinished");
    write_child_journal(&dir, child_id, &torn[..torn.len() - 2]);

    let view = project_child(&dir, child_id);
    assert_eq!(view.records.len(), 2, "the torn tail is not a record");
    assert!(
        !view
            .records
            .iter()
            .any(|record| matches!(record.event, TranscriptEvent::AssistantTurn(_))),
        "the torn tail is not projected as an assistant turn"
    );
    assert!(
        view.in_progress_assistant_text.is_none(),
        "a from-file projection carries no in-progress text"
    );
    let _ = std::fs::remove_dir_all(dir);
}

/// A fully written assistant record without its delimiter reads as finished.
#[test]
fn an_undelimited_complete_assistant_tail_reads_as_finished() {
    let dir = test_dir("undelimited-tail");
    let child_id = "child-a";
    write_child_journal(&dir, child_id, &assistant_line(child_id, 3, "answer"));

    let view = project_child(&dir, child_id);
    assert_eq!(view.records.len(), 3, "the complete record is projected");
    assert!(
        view.records.iter().any(|record| matches!(
            &record.event,
            TranscriptEvent::AssistantTurn(turn) if turn.text.as_deref() == Some("answer")
        )),
        "the complete assistant record is present"
    );
    assert!(
        view.in_progress_assistant_text.is_none(),
        "a complete record is not treated as in-progress"
    );
    let _ = std::fs::remove_dir_all(dir);
}
