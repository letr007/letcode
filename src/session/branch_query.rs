//! Session-owned context-branch queries shared by TUI and line CLI.
//!
//! These helpers keep transcript branch listing policy on the session boundary
//! so frontends do not re-implement the same load + format logic.

use std::sync::{Arc, Mutex};

use anyhow::{Result, anyhow};

use crate::transcript::transcript_projection::{self, ContextBranchInfo};
use crate::transcript::{TranscriptRecorder, read_records};

/// Load context branches for the active transcript recorder.
pub fn load_context_branches(
    transcript: &Arc<Mutex<TranscriptRecorder>>,
) -> Result<Vec<ContextBranchInfo>> {
    let recorder = transcript
        .lock()
        .map_err(|_| anyhow!("transcript recorder poisoned"))?;
    let records = read_records(recorder.path())?;
    transcript_projection::list_context_branches(&records, recorder.current_context_branch_id())
        .map_err(Into::into)
}

/// Compact single-line branch listing (TUI notices, dense CLI output).
pub fn format_branch_listing(branches: &[ContextBranchInfo]) -> String {
    if branches.is_empty() {
        return "no context branches".into();
    }
    format_branch_entries(branches).join(" · ")
}

/// Multi-line branch listing for line CLI `/tree` and `/branches`.
pub fn format_branch_listing_multiline(branches: &[ContextBranchInfo]) -> String {
    if branches.is_empty() {
        return "no context branches".into();
    }
    format_branch_entries(branches).join("\n")
}

fn format_branch_entries(branches: &[ContextBranchInfo]) -> Vec<String> {
    branches
        .iter()
        .map(|branch| {
            let marker = if branch.is_current { '*' } else { '-' };
            let mut text = format!("{marker} {}@{}", branch.branch_id, branch.tip_sequence);
            if let Some(label) = &branch.label {
                text.push_str(&format!(" ({label})"));
            }
            text
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn format_branch_listing_marks_current_and_labels() {
        let branches = vec![
            ContextBranchInfo {
                branch_id: "main".into(),
                parent_branch_id: None,
                label: None,
                tip_sequence: 3,
                is_current: true,
            },
            ContextBranchInfo {
                branch_id: "feature".into(),
                parent_branch_id: Some("main".into()),
                label: Some("wip".into()),
                tip_sequence: 9,
                is_current: false,
            },
        ];
        assert_eq!(
            format_branch_listing(&branches),
            "* main@3 · - feature@9 (wip)"
        );
        assert_eq!(
            format_branch_listing_multiline(&branches),
            "* main@3\n- feature@9 (wip)"
        );
    }

    #[test]
    fn format_branch_listing_empty() {
        assert_eq!(format_branch_listing(&[]), "no context branches");
        assert_eq!(format_branch_listing_multiline(&[]), "no context branches");
    }
}
