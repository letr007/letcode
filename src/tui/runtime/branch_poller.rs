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
