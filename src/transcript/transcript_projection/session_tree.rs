use crate::transcript::{TranscriptEvent, TranscriptRecord};
use crate::user_content::UserMessageContent;

/// A stable, append-only transcript entry suitable for session-history UIs.
///
/// IDs deliberately derive from the journal sequence rather than mutable tree
/// position. Parent links follow the persisted branch path so legacy transcripts
/// remain readable without being rewritten.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct SessionHistoryEntry {
    pub id: String,
    pub parent_id: Option<String>,
    pub sequence: u64,
    pub branch_id: String,
    pub kind: SessionHistoryEntryKind,
    pub label: String,
    pub user_content: Option<UserMessageContent>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum SessionHistoryEntryKind {
    User,
    Assistant,
    Other,
}

#[derive(Debug)]
struct BranchDefinition {
    parent_branch_id: String,
    base_sequence: u64,
}

pub(crate) fn project_session_history_tree(
    records: &[TranscriptRecord],
) -> Vec<SessionHistoryEntry> {
    use std::collections::BTreeMap;

    let mut entries = Vec::new();
    let mut branches = BTreeMap::<String, BranchDefinition>::new();
    let mut branch_leaves = BTreeMap::<String, String>::new();

    for record in records {
        if let TranscriptEvent::ContextBranchCreated {
            branch_id,
            parent_branch_id,
            base_sequence,
            ..
        } = &record.event
        {
            branches.insert(
                branch_id.clone(),
                BranchDefinition {
                    parent_branch_id: parent_branch_id.clone(),
                    base_sequence: *base_sequence,
                },
            );
            continue;
        }

        if !record.event.is_session_history_entry() {
            continue;
        }

        let branch_id = record
            .context_branch_id
            .clone()
            .unwrap_or_else(|| crate::transcript::ROOT_CONTEXT_BRANCH_ID.into());
        let parent_id = branch_leaves
            .get(&branch_id)
            .cloned()
            .or_else(|| branch_anchor_entry_id(&branch_id, &branches, &entries));
        let id = format!("entry-{}", record.sequence);
        let (kind, label, user_content) = entry_details(&record.event);
        entries.push(SessionHistoryEntry {
            id: id.clone(),
            parent_id,
            sequence: record.sequence,
            branch_id: branch_id.clone(),
            kind,
            label,
            user_content,
        });
        branch_leaves.insert(branch_id, id.clone());
    }

    entries
}

fn branch_anchor_entry_id(
    branch_id: &str,
    branches: &std::collections::BTreeMap<String, BranchDefinition>,
    entries: &[SessionHistoryEntry],
) -> Option<String> {
    let definition = branches.get(branch_id)?;
    entry_at_or_before(
        &definition.parent_branch_id,
        definition.base_sequence,
        branches,
        entries,
    )
}

fn entry_at_or_before(
    branch_id: &str,
    sequence: u64,
    branches: &std::collections::BTreeMap<String, BranchDefinition>,
    entries: &[SessionHistoryEntry],
) -> Option<String> {
    if let Some(entry) = entries
        .iter()
        .rev()
        .find(|entry| entry.branch_id == branch_id && entry.sequence <= sequence)
    {
        return Some(entry.id.clone());
    }

    let definition = branches.get(branch_id)?;
    entry_at_or_before(
        &definition.parent_branch_id,
        sequence.min(definition.base_sequence),
        branches,
        entries,
    )
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct HistoryNavigationState {
    pub target_sequence: u64,
    pub redo_stack: Vec<u64>,
}

pub(crate) fn history_navigation_state(
    records: &[TranscriptRecord],
) -> Option<HistoryNavigationState> {
    let active_branch_id = active_branch_id(records);
    for record in records.iter().rev() {
        match &record.event {
            TranscriptEvent::HistoryNavigation {
                target_sequence,
                redo_stack,
                redo_target_sequence,
                ..
            } => {
                let mut redo_stack = redo_stack.clone();
                // Journals written before redo became a stack remain readable.
                if redo_stack.is_empty() {
                    redo_stack.extend(redo_target_sequence);
                }
                return Some(HistoryNavigationState {
                    target_sequence: *target_sequence,
                    redo_stack,
                });
            }
            // Any branch-owned append on the active branch begins a new path.
            // Global topology and projection metadata remain redo-neutral.
            _ if record.context_branch_id.as_deref() == active_branch_id.as_deref() => {
                return None;
            }
            _ => {}
        }
    }
    None
}

fn active_branch_id(records: &[TranscriptRecord]) -> Option<String> {
    for record in records.iter().rev() {
        if let TranscriptEvent::ContextCheckout { branch_id, .. } = &record.event {
            return Some(branch_id.clone());
        }
    }
    Some(crate::transcript::ROOT_CONTEXT_BRANCH_ID.into())
}

fn entry_details(
    event: &TranscriptEvent,
) -> (SessionHistoryEntryKind, String, Option<UserMessageContent>) {
    match event {
        TranscriptEvent::UserMessage { content } => (
            SessionHistoryEntryKind::User,
            content.display_text(),
            Some(content.clone()),
        ),
        TranscriptEvent::AssistantMessage { content } => {
            (SessionHistoryEntryKind::Assistant, content.clone(), None)
        }
        TranscriptEvent::ReasoningMessage { content, .. } => {
            (SessionHistoryEntryKind::Assistant, content.clone(), None)
        }
        TranscriptEvent::AssistantToolCallBatch { text, .. } => (
            SessionHistoryEntryKind::Assistant,
            text.clone().unwrap_or_else(|| "Tool calls".into()),
            None,
        ),
        TranscriptEvent::ToolCallStarted { name, .. } => {
            (SessionHistoryEntryKind::Assistant, name.to_string(), None)
        }
        TranscriptEvent::ToolCallFinished { name, ok, .. } => (
            SessionHistoryEntryKind::Other,
            format!("{name} {}", if *ok { "completed" } else { "failed" }),
            None,
        ),
        TranscriptEvent::ToolCallCancelled { name, .. } => (
            SessionHistoryEntryKind::Other,
            format!("{name} cancelled"),
            None,
        ),
        TranscriptEvent::InternalContinuation { text, .. } => {
            (SessionHistoryEntryKind::Other, text.clone(), None)
        }
        TranscriptEvent::ContextCompaction(_) => (
            SessionHistoryEntryKind::Other,
            "Context compacted".into(),
            None,
        ),
        TranscriptEvent::LogicalCheckpoint(_) => (
            SessionHistoryEntryKind::Other,
            "Logical checkpoint".into(),
            None,
        ),
        TranscriptEvent::ContextExperimentReturned { summary, .. } => {
            (SessionHistoryEntryKind::Other, summary.clone(), None)
        }
        TranscriptEvent::Error { message } => {
            (SessionHistoryEntryKind::Other, message.clone(), None)
        }
        _ => unreachable!("only session history events reach the tree projection"),
    }
}

#[cfg(test)]
mod tests {
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
}
