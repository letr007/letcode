use super::DialogItem;
use crate::transcript::transcript_projection::{SessionHistoryEntry, SessionHistoryEntryKind};

pub(super) fn history_tree_dialog_items(entries: &[SessionHistoryEntry]) -> Vec<DialogItem> {
    entries
        .iter()
        .map(|entry| {
            let label = match entry.kind {
                SessionHistoryEntryKind::User => format!("You: {}", entry.label),
                SessionHistoryEntryKind::Assistant => format!("Assistant: {}", entry.label),
                SessionHistoryEntryKind::Other => entry.label.clone(),
            };
            DialogItem::new(entry.id.clone(), label, None)
                .with_section(format!("@{}", entry.sequence))
        })
        .collect()
}
