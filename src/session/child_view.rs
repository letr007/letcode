//! Child/parent session view projection shared by frontends.
//!
//! Phase R extracts navigation selection and restore projection for child and
//! parent transcript viewing. Event emission remains frontend-owned.

use std::collections::HashMap;
use std::path::Path;
use std::sync::{Arc, Mutex};

use anyhow::{Result, anyhow};

use crate::command::ChildNavigation;
use crate::runtime_context::RuntimeActiveContext;
use crate::session::restore::{
    default_resume_cursor, project_runtime_restore_snapshot_with_children,
};
use crate::subagent::SubagentPool;
use crate::transcript::transcript_projection::project_runtime_restore_snapshot;
use crate::transcript::transcript_projection::{
    RuntimeRestoreSnapshot, SessionContextCursor, project_child_session_summaries_from_file,
};
use crate::transcript::{
    ChildSessionSummary, TranscriptRecord, TranscriptRecorder, child_sessions_dir,
    read_child_session_records_allow_partial_tail, read_records,
};

/// Streaming assistant text of a child session that is still in flight.
///
/// The journal only holds durable messages, so a child view projection reads
/// this buffer to include the child's current, unpersisted answer.
#[derive(Clone, Default)]
pub(crate) struct ChildLiveText {
    inner: Arc<Mutex<HashMap<String, String>>>,
}

impl ChildLiveText {
    pub(crate) fn append(&self, child_session_id: &str, delta: &str) {
        let Ok(mut inner) = self.inner.lock() else {
            return;
        };
        inner
            .entry(child_session_id.to_string())
            .or_default()
            .push_str(delta);
    }

    pub(crate) fn get(&self, child_session_id: &str) -> Option<String> {
        self.inner.lock().ok()?.get(child_session_id).cloned()
    }

    pub(crate) fn clear(&self, child_session_id: &str) {
        if let Ok(mut inner) = self.inner.lock() {
            inner.remove(child_session_id);
        }
    }
}

/// Parent-session view projection (frontend maps this to SessionResumed-like UI).
pub struct ParentViewProjection {
    pub snapshot: RuntimeRestoreSnapshot,
    pub runtime_context: RuntimeActiveContext,
}

/// Child-session view projection (frontend maps this to ChildSessionViewed).
pub struct ChildViewProjection {
    pub parent_session_id: String,
    pub child_session_id: String,
    pub agent_name: String,
    pub index: usize,
    pub total: usize,
    pub pool_ordinal: u32,
    pub records: Vec<TranscriptRecord>,
    pub runtime_context: RuntimeActiveContext,
    pub in_progress_assistant_text: Option<String>,
}

/// Resolve which child index to open for a navigation command.
pub fn select_child_navigation_index(
    children: &[ChildSessionSummary],
    navigation: ChildNavigation,
    anchor_child_session_id: Option<&str>,
) -> Option<usize> {
    if children.is_empty() {
        return None;
    }
    let current_index = anchor_child_session_id.and_then(|child_session_id| {
        children
            .iter()
            .position(|child| child.child_session_id == child_session_id)
    });
    Some(match navigation {
        ChildNavigation::Toggle | ChildNavigation::First => 0,
        ChildNavigation::Next => current_index
            .map(|index| (index + 1) % children.len())
            .unwrap_or(0),
        ChildNavigation::Prev => current_index
            .map(|index| {
                if index == 0 {
                    children.len() - 1
                } else {
                    index - 1
                }
            })
            .unwrap_or(children.len() - 1),
    })
}

/// Sessions directory for a live transcript recorder (parent of the jsonl path).
pub fn sessions_dir_from_transcript(
    transcript: &Arc<Mutex<TranscriptRecorder>>,
) -> Result<std::path::PathBuf> {
    let recorder = transcript
        .lock()
        .map_err(|_| anyhow!("transcript recorder poisoned"))?;
    recorder
        .path()
        .parent()
        .map(Path::to_path_buf)
        .ok_or_else(|| anyhow!("transcript path has no parent directory"))
}

/// Project the current live transcript as a parent/root view restore package.
pub fn project_parent_session_view(
    transcript: &Arc<Mutex<TranscriptRecorder>>,
    sessions_dir: impl AsRef<Path>,
) -> Result<ParentViewProjection> {
    let (session_id, records, branch_id) = {
        let recorder = transcript
            .lock()
            .map_err(|_| anyhow!("transcript recorder poisoned"))?;
        (
            recorder.session_id().to_string(),
            read_records(recorder.path())?,
            recorder.current_context_branch_id().map(str::to_string),
        )
    };
    let snapshot = project_runtime_restore_snapshot_with_children(
        session_id,
        records,
        SessionContextCursor {
            branch_id,
            leaf_sequence: None,
        },
        sessions_dir,
    )?;
    let runtime_context = RuntimeActiveContext::try_from(&snapshot.snapshot)?;
    Ok(ParentViewProjection {
        snapshot,
        runtime_context,
    })
}

/// Project a child view while discovering parent children from a streaming scan.
pub fn project_child_session_view_from_file(
    sessions_dir: impl AsRef<Path>,
    parent_session_id: impl Into<String>,
    navigation: ChildNavigation,
    anchor_child_session_id: Option<&str>,
    live_text: &ChildLiveText,
) -> Result<Option<ChildViewProjection>> {
    let sessions_dir = sessions_dir.as_ref();
    let parent_session_id = parent_session_id.into();
    let parent_path = sessions_dir.join(format!("{parent_session_id}.jsonl"));
    let children =
        project_child_session_summaries_from_file(&child_sessions_dir(sessions_dir), &parent_path)?;
    let children = SubagentPool::child_sessions_from_summaries(children);
    project_child_session_view_with_children(
        sessions_dir,
        parent_session_id,
        children,
        navigation,
        anchor_child_session_id,
        live_text,
    )
}

fn project_child_session_view_with_children(
    sessions_dir: &Path,
    parent_session_id: String,
    children: Vec<ChildSessionSummary>,
    navigation: ChildNavigation,
    anchor_child_session_id: Option<&str>,
    live_text: &ChildLiveText,
) -> Result<Option<ChildViewProjection>> {
    let Some(index) = select_child_navigation_index(&children, navigation, anchor_child_session_id)
    else {
        return Ok(None);
    };
    let child = &children[index];
    let records =
        read_child_session_records_allow_partial_tail(sessions_dir, &child.child_session_id)?;
    // Child sessions cannot own nested subagents today. Project their transcript
    // once instead of performing the parent restore path's two-pass child lookup.
    let snapshot = project_runtime_restore_snapshot(
        child.child_session_id.clone(),
        records,
        default_resume_cursor(),
        &[],
    )?;
    let runtime_context = RuntimeActiveContext::try_from(&snapshot.snapshot)?;
    let records = snapshot.records;
    Ok(Some(ChildViewProjection {
        parent_session_id,
        child_session_id: child.child_session_id.clone(),
        agent_name: child.agent_name.clone(),
        index,
        total: children.len(),
        pool_ordinal: child.pool_ordinal,
        records,
        runtime_context,
        in_progress_assistant_text: live_text.get(&child.child_session_id),
    }))
}

#[cfg(test)]
mod tests {
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

    /// TEMPORARY measurement: times the child-view switch projection on real journals.
    #[test]
    #[ignore = "measurement harness: set LETCODE_BENCH_SESSIONS and LETCODE_BENCH_PARENT_SESSION"]
    fn switch_measure() {
        use std::time::Instant;

        let dir = std::path::PathBuf::from(
            std::env::var("LETCODE_BENCH_SESSIONS").expect("LETCODE_BENCH_SESSIONS"),
        );
        let parent_session_id =
            std::env::var("LETCODE_BENCH_PARENT_SESSION").expect("LETCODE_BENCH_PARENT_SESSION");
        let parent_path = dir.join(format!("{parent_session_id}.jsonl"));
        let parent_size = std::fs::metadata(&parent_path).expect("stat parent").len();
        println!(
            "\n### parent {parent_session_id} — {:.1} MB",
            parent_size as f64 / 1048576.0
        );

        let mut navigation = crate::command::ChildNavigation::First;
        let mut anchor: Option<String> = None;
        for round in 0..4 {
            let start = Instant::now();
            let view = project_child_session_view_from_file(
                &dir,
                parent_session_id.clone(),
                navigation,
                anchor.as_deref(),
                &ChildLiveText::default(),
            )
            .expect("project the child view")
            .expect("a child to view");
            let elapsed = start.elapsed();
            let child_size = std::fs::metadata(
                crate::transcript::child_sessions_dir(&dir)
                    .join(format!("{}.jsonl", view.child_session_id)),
            )
            .map(|metadata| metadata.len())
            .unwrap_or_default();
            let _ = &view.runtime_context;
            println!(
                "  round {round}: {:>8.1?}  child {} {:.1} MB  {} records",
                elapsed,
                view.child_session_id,
                child_size as f64 / 1048576.0,
                view.records.len(),
            );
            anchor = Some(view.child_session_id.clone());
            navigation = crate::command::ChildNavigation::Next;
        }

        let target = anchor.expect("a child was viewed");
        for round in 0..3 {
            let start = Instant::now();
            let records = crate::transcript::read_records(&parent_path).expect("read the parent");
            let children = SubagentPool::child_sessions(&dir, &records);
            let parent_ms = start.elapsed();
            let start = Instant::now();
            let child_records =
                crate::transcript::read_child_session_records_allow_partial_tail(&dir, &target)
                    .expect("read the child");
            let child_ms = start.elapsed();
            println!(
                "  poll pass {round}: parent full read + children {:>8.1?} ({} records, {} children) · child read {:>8.1?} ({} records)",
                parent_ms,
                records.len(),
                children.len(),
                child_ms,
                child_records.len()
            );
        }
    }
}
