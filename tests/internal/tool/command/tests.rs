use super::*;
use anyhow::Context;
use serde_json::json;
use std::path::PathBuf;
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use tokio::time::{sleep, timeout};

#[cfg(unix)]
fn process_fixture_dir(name: &str) -> PathBuf {
    let unique = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system time")
        .as_nanos();
    let path = std::env::temp_dir().join(format!("letcode-{name}-{unique}"));
    std::fs::create_dir_all(&path).expect("create process fixture directory");
    path
}
#[cfg(unix)]
async fn wait_for_file(path: &std::path::Path) {
    timeout(Duration::from_secs(2), async {
        while !path.exists() {
            sleep(Duration::from_millis(10)).await;
        }
    })
    .await
    .unwrap_or_else(|_| panic!("timed out waiting for {}", path.display()));
}
#[cfg(unix)]
async fn assert_file_does_not_appear(path: &std::path::Path) {
    let appeared = timeout(Duration::from_millis(300), async {
        loop {
            if path.exists() {
                return;
            }
            sleep(Duration::from_millis(10)).await;
        }
    })
    .await;
    assert!(
        appeared.is_err(),
        "unexpected side effect: {}",
        path.display()
    );
}
#[cfg(unix)]
fn delayed_side_effect_command(
    ready: &std::path::Path,
    release: &std::path::Path,
    marker: &std::path::Path,
) -> String {
    format!(
        "touch {}; (while [ ! -e {} ]; do :; done; touch {}) & wait",
        ready.display(),
        release.display(),
        marker.display()
    )
}
#[cfg(windows)]
#[test]
fn windows_shells_prefer_powershell_7_then_fall_back() {
    let shells = super::SHELL_INVOCATIONS
        .iter()
        .map(|invocation| invocation.executable)
        .collect::<Vec<_>>();
    assert_eq!(shells, ["pwsh.exe", "powershell.exe", "cmd.exe"]);
}

#[test]
fn executable_not_found_only_accepts_direct_spawn_error() {
    let missing = anyhow::Error::new(std::io::Error::new(
        std::io::ErrorKind::NotFound,
        "missing executable",
    ));
    assert!(super::executable_not_found(&missing));

    let contextual = anyhow::Error::new(std::io::Error::new(
        std::io::ErrorKind::NotFound,
        "other missing resource",
    ))
    .context("shell setup failed");
    assert!(!super::executable_not_found(&contextual));
    assert!(!super::executable_not_found(&anyhow::anyhow!(
        "other error"
    )));
}

#[test]
fn command_timeout_uses_default_and_accepts_explicit_value() {
    assert_eq!(super::command_timeout_secs(&json!({})).unwrap(), 300);
    assert_eq!(
        super::command_timeout_secs(&json!({"timeout_secs": null})).unwrap(),
        300
    );
    assert_eq!(
        super::command_timeout_secs(&json!({"timeout_secs": 900})).unwrap(),
        900
    );
}

#[test]
fn command_timeout_rejects_invalid_boundaries() {
    assert!(
        super::command_timeout_secs(&json!({"timeout_secs": 0}))
            .unwrap_err()
            .to_string()
            .contains("greater than 0")
    );
    assert_eq!(
        super::command_timeout_secs(&json!({"timeout_secs": 3601}))
            .unwrap_err()
            .to_string(),
        "field 'timeout_secs' must be at most 3600"
    );
}

mod integration_tests {
    use super::*;

    #[cfg(unix)]
    #[tokio::test]
    async fn streaming_shell_command_honors_explicit_timeout() {
        use crate::tool::ToolHandler;

        let mut emit = |_stream, _chunk| Ok(());
        let result = RunCommandTool
            .execute_streaming(
                json!({"command": "sleep 2", "timeout_secs": 1}),
                ToolExecutionContext::default(),
                &mut emit,
            )
            .await
            .expect("shell tool result");

        assert!(result.ok);
        assert_eq!(
            result
                .data
                .as_ref()
                .and_then(|data| data.get("error"))
                .and_then(serde_json::Value::as_str),
            Some("command timed out after 1s")
        );
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn dropping_streaming_shell_command_kills_process_group() {
        let dir = process_fixture_dir("streaming-drop");
        let ready = dir.join("ready");
        let release = dir.join("release");
        let marker = dir.join("marker");
        let script = delayed_side_effect_command(&ready, &release, &marker);
        let task = tokio::spawn(async move {
            let mut emit = |_stream, _chunk| Ok(());
            run_workspace_shell_command_streaming(&script, 30, &mut emit).await
        });
        wait_for_file(&ready).await;
        task.abort();
        assert!(
            task.await
                .expect_err("task should be cancelled")
                .is_cancelled()
        );
        std::fs::write(&release, "go").expect("release side effect");
        assert_file_does_not_appear(&marker).await;
        let _ = std::fs::remove_dir_all(dir);
    }
}

#[tokio::test]
async fn fold_bytes_output_persists_large_bodies_and_keeps_preview() {
    let big = vec![b'x'; 70 * 1024];
    let output = super::fold_bytes_output(&big, "out").await;
    assert!(output.folded);
    assert!(output.truncated);
    let path = output.local_path.expect("local path");
    assert!(path.ends_with(".out"));
    let on_disk = tokio::fs::read(&path).await.expect("read artifact");
    assert_eq!(on_disk, big);
    assert_eq!(output.text.len(), 8 * 1024);
    let _ = tokio::fs::remove_file(path).await;
}

#[tokio::test]
async fn fold_bytes_output_leaves_small_bodies_inline() {
    let output = super::fold_bytes_output(b"hello world", "out").await;
    assert!(!output.folded);
    assert!(!output.truncated);
    assert!(output.local_path.is_none());
    assert_eq!(output.text, "hello world");
}

#[test]
fn stream_accumulator_folds_after_threshold() {
    let mut acc = super::StreamAccumulator::new();
    let chunk = "x".repeat(16 * 1024);
    for _ in 0..5 {
        acc.push(&chunk);
    }
    assert!(acc.output.folded);
    assert!(acc.output.truncated);
    assert!(acc.output.local_path.is_some());
    assert_eq!(acc.output.text.len(), 8 * 1024);
    acc.push(&"y".repeat(1000));
    assert_eq!(acc.output.text.len(), 8 * 1024, "preview stays capped");
    let path = acc.output.local_path.clone().unwrap();
    let on_disk = std::fs::read(&path).expect("read artifact");
    assert_eq!(on_disk.len(), 5 * 16 * 1024 + 1000);
    assert!(on_disk[..5 * 16 * 1024].iter().all(|&b| b == b'x'));
    assert!(on_disk[5 * 16 * 1024..].iter().all(|&b| b == b'y'));
    let _ = std::fs::remove_file(path);
}

mod output {
    use super::*;

    mod integration_tests {
        use super::*;

        #[cfg(unix)]
        #[tokio::test]
        async fn streaming_shell_command_folds_large_output() {
            use crate::tool::ToolHandler;

            let mut emit = |_stream, _chunk| Ok(());
            let result = RunCommandTool
                .execute_streaming(
                    json!({"command": "yes x | head -c 70000", "timeout_secs": 30}),
                    ToolExecutionContext::default(),
                    &mut emit,
                )
                .await
                .expect("shell tool result");

            assert!(result.ok);
            let data = result.data.as_ref().expect("data");
            assert_eq!(
                data.get("status").and_then(serde_json::Value::as_i64),
                Some(0)
            );
            assert_eq!(
                data.get("stdout_folded")
                    .and_then(serde_json::Value::as_bool),
                Some(true)
            );
            assert_eq!(
                data.get("stdout_truncated")
                    .and_then(serde_json::Value::as_bool),
                Some(true)
            );
            let path = data
                .get("stdout_local_path")
                .and_then(serde_json::Value::as_str)
                .expect("local path");
            let on_disk = tokio::fs::read(path).await.expect("read artifact");
            assert_eq!(on_disk.len(), 70000);
            assert!(
                on_disk.iter().all(|&b| b == b'x' || b == b'\n'),
                "yes emits x per line"
            );
            let _ = tokio::fs::remove_file(path).await;
        }
    }
}
