//! Git-branch background polling, isolated from the runtime orchestrator.
//!
//! The branch refresh runs off the TUI event loop: a short-lived thread queries
//! git and pushes the result over a channel, and `BranchPoller::poll` drains
//! that channel once per frame. Owning the channel, throttle timestamp, and
//! workspace path here keeps `TuiRuntime` focused on orchestration and makes
//! this unit independently testable.

use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{Duration, Instant};

use tokio::sync::mpsc;

use crate::tui::state::TuiState;

const GIT_BRANCH_REFRESH_INTERVAL: Duration = Duration::from_secs(2);

/// Polls the current git branch without blocking the TUI for frames.
pub(crate) struct BranchPoller {
    workspace_dir: Option<PathBuf>,
    branch_rx: Option<mpsc::UnboundedReceiver<Option<String>>>,
    next_refresh: Instant,
}

impl BranchPoller {
    pub(crate) fn new() -> Self {
        Self {
            workspace_dir: None,
            branch_rx: None,
            next_refresh: Instant::now(),
        }
    }

    /// (Re)point the poller at a workspace and schedule an immediate refresh.
    pub(crate) fn set_workspace_dir(&mut self, workspace_dir: PathBuf) {
        self.workspace_dir = Some(workspace_dir);
        self.next_refresh = Instant::now();
    }

    /// Advance one frame's worth of git-branch polling.
    pub(crate) fn poll(&mut self, state: &mut TuiState) {
        if let Some(rx) = self.branch_rx.as_mut() {
            match rx.try_recv() {
                Ok(branch) => {
                    self.branch_rx = None;
                    state.set_git_branch(branch);
                    self.next_refresh = Instant::now() + GIT_BRANCH_REFRESH_INTERVAL;
                }
                Err(mpsc::error::TryRecvError::Disconnected) => {
                    self.branch_rx = None;
                    state.set_git_branch(None);
                    self.next_refresh = Instant::now() + GIT_BRANCH_REFRESH_INTERVAL;
                }
                Err(mpsc::error::TryRecvError::Empty) => return,
            }
        }
        if self.branch_rx.is_some() || Instant::now() < self.next_refresh {
            return;
        }
        let Some(workspace_dir) = self.workspace_dir.clone() else {
            return;
        };
        let (tx, rx) = mpsc::unbounded_channel();
        self.branch_rx = Some(rx);
        std::thread::spawn(move || {
            let _ = tx.send(read_git_branch(&workspace_dir));
        });
    }

    /// Test hook: enqueue a branch refresh for the next `poll` to consume.
    #[cfg(test)]
    pub(crate) fn enqueue_for_test(&mut self, branch: Option<String>) {
        let (tx, rx) = mpsc::unbounded_channel();
        tx.send(branch).expect("queue branch refresh");
        self.branch_rx = Some(rx);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tui::state::TuiState;

    fn poller() -> (BranchPoller, TuiState) {
        (BranchPoller::new(), TuiState::default())
    }

    #[test]
    fn poll_consumes_enqueued_branch_into_state_and_clears_channel() {
        let (mut poller, mut state) = poller();
        poller.enqueue_for_test(Some("main".into()));
        poller.poll(&mut state);
        assert_eq!(state.git_branch.as_deref(), Some("main"));
        assert!(poller.branch_rx.is_none());
    }

    #[test]
    fn poll_treats_disconnected_channel_as_no_branch() {
        let (mut poller, mut state) = poller();
        let (tx, rx) = mpsc::unbounded_channel();
        drop(tx);
        poller.branch_rx = Some(rx);
        state.set_git_branch(Some("stale".into()));
        poller.poll(&mut state);
        assert_eq!(state.git_branch, None);
        assert!(poller.branch_rx.is_none());
    }

    #[test]
    fn poll_leaves_pending_channel_when_empty() {
        let (mut poller, mut state) = poller();
        let (tx, rx) = mpsc::unbounded_channel();
        poller.branch_rx = Some(rx);
        // Sender still alive, nothing queued yet: try_recv is Empty, channel kept.
        poller.poll(&mut state);
        assert!(poller.branch_rx.is_some());
        assert_eq!(state.git_branch, None);
        drop(tx);
        // Now disconnected: next poll clears it and reports no branch.
        poller.poll(&mut state);
        assert!(poller.branch_rx.is_none());
    }

    #[test]
    fn re_enqueued_branch_replaces_previous_value() {
        let (mut poller, mut state) = poller();
        poller.enqueue_for_test(Some("main".into()));
        poller.poll(&mut state);
        assert_eq!(state.git_branch.as_deref(), Some("main"));
        poller.enqueue_for_test(None);
        poller.poll(&mut state);
        assert_eq!(state.git_branch, None);
    }
}

/// Resolve the current git branch, falling back to a short commit hash.
pub(crate) fn read_git_branch(workspace_dir: &Path) -> Option<String> {
    let branch = Command::new("git")
        .args(["symbolic-ref", "--quiet", "--short", "HEAD"])
        .current_dir(workspace_dir)
        .output()
        .ok()
        .filter(|output| output.status.success())
        .and_then(|output| String::from_utf8(output.stdout).ok())
        .map(|branch| branch.trim().to_string())
        .filter(|branch| !branch.is_empty());
    if branch.is_some() {
        return branch;
    }

    Command::new("git")
        .args(["rev-parse", "--short", "HEAD"])
        .current_dir(workspace_dir)
        .output()
        .ok()
        .filter(|output| output.status.success())
        .and_then(|output| String::from_utf8(output.stdout).ok())
        .map(|commit| format!("detached@{}", commit.trim()))
        .filter(|commit| !commit.ends_with('@'))
}
