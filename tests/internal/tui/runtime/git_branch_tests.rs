use super::TuiRuntime;
use crate::tui::TuiState;
use crate::tui::state::ToastKind;
use std::time::{SystemTime, UNIX_EPOCH};
use tokio::sync::mpsc;

fn temp_dir(label: &str) -> std::path::PathBuf {
    std::env::temp_dir().join(format!(
        "letcode-git-branch-{label}-{}",
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("time")
            .as_nanos()
    ))
}

fn runtime() -> TuiRuntime {
    let (_tx, rx) = mpsc::unbounded_channel();
    TuiRuntime::new(
        TuiState::default(),
        rx,
        Vec::new(),
        Vec::new(),
        temp_dir("sessions"),
        temp_dir("preferences"),
    )
}

fn deliver_branch_refresh(runtime: &mut TuiRuntime, branch: Option<String>) {
    runtime.branch_poller.enqueue_for_test(branch);
    runtime.poll_git_branch();
}

#[test]
fn branch_refresh_replaces_and_clears_cached_state() {
    let mut runtime = runtime();

    deliver_branch_refresh(&mut runtime, Some("main".into()));
    assert_eq!(runtime.state().git_branch.as_deref(), Some("main"));

    deliver_branch_refresh(&mut runtime, Some("detached@abc1234".into()));
    assert_eq!(
        runtime.state().git_branch.as_deref(),
        Some("detached@abc1234")
    );

    deliver_branch_refresh(&mut runtime, None);
    assert_eq!(runtime.state().git_branch, None);
}

#[test]
fn update_check_result_shows_localized_info_toast_once() {
    let mut runtime = runtime();
    let (tx, rx) = mpsc::unbounded_channel();
    runtime.update_check_rx = Some(rx);
    tx.send(Ok(Some("0.5.3".into())))
        .expect("update check result should send");

    runtime.poll_update_check();

    let toast = runtime
        .state()
        .toast()
        .expect("update toast should be shown");
    assert_eq!(toast.kind, ToastKind::Info);
    assert!(toast.message.contains("0.5.3"));
    assert!(toast.message.contains("letcode update"));
    assert!(runtime.update_check_rx.is_none());
}

#[test]
fn failed_update_check_is_silent() {
    let mut runtime = runtime();
    let (tx, rx) = mpsc::unbounded_channel();
    runtime.update_check_rx = Some(rx);
    tx.send(Err(anyhow::anyhow!("offline")))
        .expect("update check error should send");

    runtime.poll_update_check();

    assert!(runtime.state().toast().is_none());
    assert!(runtime.update_check_rx.is_none());
}

mod integration_tests {
    use super::super::branch_poller::read_git_branch;
    use super::*;
    use std::path::Path;
    use std::process::Command;

    fn git(path: &Path, args: &[&str]) {
        let output = Command::new("git")
            .args(args)
            .current_dir(path)
            .output()
            .expect("run git");
        assert!(
            output.status.success(),
            "git {:?} failed: {}",
            args,
            String::from_utf8_lossy(&output.stderr)
        );
    }

    #[test]
    fn reads_named_and_detached_git_branches() {
        let path = temp_dir("repository");
        std::fs::create_dir_all(&path).expect("create temp repository");
        git(&path, &["init", "-q", "-b", "main"]);
        git(&path, &["config", "user.email", "test@example.com"]);
        git(&path, &["config", "user.name", "LetCode Test"]);
        std::fs::write(path.join("file.txt"), "content").expect("write file");
        git(&path, &["add", "file.txt"]);
        git(&path, &["commit", "-q", "-m", "initial"]);

        assert_eq!(read_git_branch(&path).as_deref(), Some("main"));
        git(&path, &["checkout", "-q", "--detach", "HEAD"]);
        let branch = read_git_branch(&path).expect("detached branch label");
        assert!(branch.starts_with("detached@"), "{branch}");

        let _ = std::fs::remove_dir_all(path);
    }

    #[test]
    fn non_git_directory_has_no_branch() {
        let path = temp_dir("non-repository");
        std::fs::create_dir_all(&path).expect("create temp directory");

        assert_eq!(read_git_branch(&path), None);

        let _ = std::fs::remove_dir_all(path);
    }
}
