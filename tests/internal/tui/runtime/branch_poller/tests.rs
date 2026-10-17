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

#[test]
fn poll_abandons_a_stalled_query_and_slows_the_next_refresh() {
    let (mut poller, mut state) = poller();
    let (tx, rx) = mpsc::unbounded_channel();
    poller.branch_rx = Some(rx);
    poller.deadline = Some(Instant::now() - Duration::from_secs(1));
    state.set_git_branch(Some("stale".into()));
    poller.poll(&mut state);
    assert!(poller.branch_rx.is_none());
    assert_eq!(state.git_branch.as_deref(), Some("stale"));
    assert!(poller.next_refresh > Instant::now() + GIT_BRANCH_REFRESH_INTERVAL);
    drop(tx);
}
