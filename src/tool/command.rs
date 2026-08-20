use anyhow::{Context, Result, anyhow};
use async_trait::async_trait;
use serde_json::{Value, json};
use std::io::Write;
use std::process::Stdio;
use std::time::{SystemTime, UNIX_EPOCH};
use tokio::io::{AsyncRead, AsyncReadExt};
use tokio::process::Command;
use tokio::sync::mpsc;
use tokio::time::{Duration, sleep, timeout};
use tracing::{debug, warn};

use super::fold_artifact::{
    COMMAND_ARTIFACT_DIR, FOLD_PREVIEW_CHARS, FOLD_THRESHOLD_BYTES, fold_preview, write_artifact,
};
use super::args::required_string;
use super::delegation::optional_u64;
use super::{
    COMMAND_TIMEOUT_SECS, ToolExecutionContext, ToolHandler, ToolOutputEmitter, ToolOutputStream,
    ToolRegistry, workspace_root,
};

const MAX_COMMAND_OUTPUT_BYTES: usize = 256 * 1024;
const MAX_COMMAND_TIMEOUT_SECS: u64 = 3_600;

pub(super) fn register(registry: &mut ToolRegistry) {
    registry.register(RunCommandTool);
}

struct RunCommandTool;

#[async_trait]
impl ToolHandler for RunCommandTool {
    fn name(&self) -> &'static str {
        "shell__exec"
    }

    fn description(&self) -> &'static str {
        "Run a shell command in the current workspace when specialized tools are not a better fit (prefer fs__/search__/git__/edit tools for file and repo work). Avoid high-impact irreversible commands without clear scope; keep compound commands (&&, ||, ;, pipes) from chaining steps that need separate confirmation; ensure loops/listeners have exit/timeout limits. Authorization is handled by the tool-level permission policy."
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "Shell command to run, e.g. cargo check or ls -la"
                },
                "timeout_secs": {
                    "type": ["integer", "null"],
                    "minimum": 1,
                    "maximum": MAX_COMMAND_TIMEOUT_SECS,
                    "description": "Command timeout in seconds. Defaults to 300, maximum 3600"
                }
            },
            "required": ["command", "timeout_secs"],
            "additionalProperties": false
        })
    }

    async fn execute(&self, args: Value) -> Result<Value> {
        run_command(args).await
    }

    async fn execute_streaming(
        &self,
        args: Value,
        _context: ToolExecutionContext,
        emit: ToolOutputEmitter<'_>,
    ) -> Result<super::ToolResult> {
        let command = required_string(&args, "command")?;
        let timeout_secs = command_timeout_secs(&args)?;
        run_workspace_shell_command_streaming(command, timeout_secs, emit)
            .await
            .map(|data| super::ToolResult::ok(self.name(), data))
    }
}

fn command_timeout_secs(args: &Value) -> Result<u64> {
    let timeout_secs = optional_u64(args, "timeout_secs")?.unwrap_or(COMMAND_TIMEOUT_SECS);
    if timeout_secs > MAX_COMMAND_TIMEOUT_SECS {
        return Err(anyhow!(
            "field 'timeout_secs' must be at most {MAX_COMMAND_TIMEOUT_SECS}"
        ));
    }
    Ok(timeout_secs)
}

async fn run_command(args: Value) -> Result<Value> {
    let command = required_string(&args, "command")?;
    let timeout_secs = command_timeout_secs(&args)?;
    run_workspace_shell_command(command, timeout_secs).await
}

struct CommandOutput {
    status: std::process::ExitStatus,
    stdout: Vec<u8>,
    stderr: Vec<u8>,
}

/// Owns a child until it has been waited for.  Dropping it kills the child and,
/// on Unix, its process group so shell grandchildren cannot outlive a cancelled tool call.
struct ManagedChild {
    child: Option<tokio::process::Child>,
    #[cfg(unix)]
    process_group: Option<i32>,
}

impl ManagedChild {
    fn spawn(mut command: Command) -> Result<Self> {
        #[cfg(unix)]
        {
            // Put the command in a new group before it can execute user code.
            // This lets cancellation kill shell descendants as well as the shell itself.
            unsafe {
                command.pre_exec(|| {
                    if libc::setpgid(0, 0) == -1 {
                        return Err(std::io::Error::last_os_error());
                    }
                    Ok(())
                });
            }
        }

        let child = command.spawn()?;
        #[cfg(unix)]
        let process_group = child.id().map(|pid| pid as i32);
        Ok(Self {
            child: Some(child),
            #[cfg(unix)]
            process_group,
        })
    }

    fn take_stdout(&mut self) -> Option<tokio::process::ChildStdout> {
        self.child.as_mut()?.stdout.take()
    }

    fn take_stderr(&mut self) -> Option<tokio::process::ChildStderr> {
        self.child.as_mut()?.stderr.take()
    }

    async fn wait(&mut self) -> Result<std::process::ExitStatus> {
        self.child
            .as_mut()
            .ok_or_else(|| anyhow!("child is no longer managed"))?
            .wait()
            .await
            .map_err(Into::into)
    }

    fn terminate(&mut self) {
        #[cfg(unix)]
        if let Some(process_group) = self.process_group {
            // A negative PID addresses the process group. Ignore ESRCH because the
            // process may have exited between timeout/cancellation and this signal.
            unsafe {
                libc::kill(-process_group, libc::SIGKILL);
            }
        }

        if let Some(child) = self.child.as_mut() {
            let _ = child.start_kill();
        }
    }

    async fn terminate_and_wait(&mut self) -> Result<std::process::ExitStatus> {
        self.terminate();
        self.wait().await
    }

    fn disarm(&mut self) {
        self.child.take();
    }
}

impl Drop for ManagedChild {
    fn drop(&mut self) {
        self.terminate();
        let Some(mut child) = self.child.take() else {
            return;
        };

        // Tool futures are polled inside Tokio. Reap asynchronously so cancellation
        // does not leave a zombie while keeping Drop non-blocking.
        if let Ok(runtime) = tokio::runtime::Handle::try_current() {
            runtime.spawn(async move {
                let _ = child.wait().await;
            });
        }
    }
}

async fn read_all_command_stream<R>(mut reader: R) -> Result<Vec<u8>>
where
    R: AsyncRead + Unpin,
{
    let mut bytes = Vec::new();
    reader.read_to_end(&mut bytes).await?;
    Ok(bytes)
}

pub(super) async fn run_workspace_command(
    command: &str,
    args: &[String],
    timeout_secs: u64,
) -> Result<Value> {
    let root = workspace_root()?;
    debug!(command = %command, args = ?args, "running workspace command");

    let mut command = Command::new(command);
    command
        .args(args)
        .current_dir(root)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    let mut child = ManagedChild::spawn(command)?;
    let stdout = child
        .take_stdout()
        .ok_or_else(|| anyhow!("failed to capture command stdout"))?;
    let stderr = child
        .take_stderr()
        .ok_or_else(|| anyhow!("failed to capture command stderr"))?;
    let stdout_reader = tokio::spawn(async move { read_all_command_stream(stdout).await });
    let stderr_reader = tokio::spawn(async move { read_all_command_stream(stderr).await });

    let status = match timeout(Duration::from_secs(timeout_secs), child.wait()).await {
        Ok(status) => status?,
        Err(_) => {
            let _ = child.terminate_and_wait().await;
            child.disarm();
            return Ok(json!({
                "error": format!("command timed out after {timeout_secs}s")
            }));
        }
    };
    child.disarm();
    let output = CommandOutput {
        status,
        stdout: stdout_reader
            .await
            .context("command stdout reader failed")??,
        stderr: stderr_reader
            .await
            .context("command stderr reader failed")??,
    };

    let stdout = truncate_utf8(
        &String::from_utf8_lossy(&output.stdout),
        MAX_COMMAND_OUTPUT_BYTES,
    );
    let stderr = truncate_utf8(
        &String::from_utf8_lossy(&output.stderr),
        MAX_COMMAND_OUTPUT_BYTES,
    );

    Ok(json!({
        "status": output.status.code(),
        "success": output.status.success(),
        "stdout": stdout.text,
        "stdout_truncated": stdout.truncated,
        "stderr": stderr.text,
        "stderr_truncated": stderr.truncated,
    }))
}

async fn run_workspace_shell_command(command: &str, timeout_secs: u64) -> Result<Value> {
    let root = workspace_root()?;
    let (shell, shell_flag) = shell_invocation();
    debug!(command = %command, shell = %shell, "running workspace shell command");

    let mut shell_command = Command::new(shell);
    shell_command
        .arg(shell_flag)
        .arg(command)
        .current_dir(root)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    let mut child = ManagedChild::spawn(shell_command)?;
    let stdout = child
        .take_stdout()
        .ok_or_else(|| anyhow!("failed to capture command stdout"))?;
    let stderr = child
        .take_stderr()
        .ok_or_else(|| anyhow!("failed to capture command stderr"))?;
    let stdout_reader = tokio::spawn(async move { read_all_command_stream(stdout).await });
    let stderr_reader = tokio::spawn(async move { read_all_command_stream(stderr).await });

    let status = match timeout(Duration::from_secs(timeout_secs), child.wait()).await {
        Ok(status) => status?,
        Err(_) => {
            let _ = child.terminate_and_wait().await;
            child.disarm();
            return Ok(json!({
                "command": command,
                "error": format!("command timed out after {timeout_secs}s")
            }));
        }
    };
    child.disarm();
    let output = CommandOutput {
        status,
        stdout: stdout_reader
            .await
            .context("command stdout reader failed")??,
        stderr: stderr_reader
            .await
            .context("command stderr reader failed")??,
    };

    let stdout = fold_bytes_output(&output.stdout, "out").await;
    let stderr = fold_bytes_output(&output.stderr, "err").await;
    let mut data = json!({
        "command": command,
        "shell": shell,
        "status": output.status.code(),
        "success": output.status.success(),
    });
    let fields = data.as_object_mut().expect("command output object");
    add_stream_fields(fields, "stdout", &stdout);
    add_stream_fields(fields, "stderr", &stderr);
    Ok(data)
}

async fn run_workspace_shell_command_streaming(
    command: &str,
    timeout_secs: u64,
    emit: ToolOutputEmitter<'_>,
) -> Result<Value> {
    let root = workspace_root()?;
    let (shell, shell_flag) = shell_invocation();
    debug!(command = %command, shell = %shell, "running streaming workspace shell command");

    let mut shell_command = Command::new(shell);
    shell_command
        .arg(shell_flag)
        .arg(command)
        .current_dir(root)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    let mut child = ManagedChild::spawn(shell_command)
        .with_context(|| format!("failed to spawn shell command: {command}"))?;
    let stdout = child
        .take_stdout()
        .ok_or_else(|| anyhow!("failed to capture command stdout"))?;
    let stderr = child
        .take_stderr()
        .ok_or_else(|| anyhow!("failed to capture command stderr"))?;

    let (tx, mut rx) = mpsc::unbounded_channel();
    tokio::spawn(read_command_stream(
        ToolOutputStream::Stdout,
        stdout,
        tx.clone(),
    ));
    tokio::spawn(read_command_stream(ToolOutputStream::Stderr, stderr, tx));

    let mut stdout = StreamAccumulator::new();
    let mut stderr = StreamAccumulator::new();
    let timeout_sleep = sleep(Duration::from_secs(timeout_secs));
    tokio::pin!(timeout_sleep);
    let mut timed_out = false;
    let mut status = None;

    loop {
        tokio::select! {
            Some((stream, chunk)) = rx.recv() => {
                match stream { ToolOutputStream::Stdout => stdout.push(&chunk), ToolOutputStream::Stderr => stderr.push(&chunk) }
                emit(stream, chunk)?;
            }
            result = child.wait() => { status = Some(result?); break; }
            _ = &mut timeout_sleep => { timed_out = true; break; }
        }
    }
    let status = if timed_out {
        child.terminate_and_wait().await?
    } else {
        status.ok_or_else(|| anyhow!("command exited without status"))?
    };
    child.disarm();

    while let Some((stream, chunk)) = rx.recv().await {
        match stream {
            ToolOutputStream::Stdout => stdout.push(&chunk),
            ToolOutputStream::Stderr => stderr.push(&chunk),
        }
        emit(stream, chunk)?;
    }

    let mut data = json!({ "command": command, "shell": shell, "status": status.code(), "success": status.success() && !timed_out });
    let fields = data.as_object_mut().expect("command output object");
    add_stream_fields(fields, "stdout", &stdout.output);
    add_stream_fields(fields, "stderr", &stderr.output);
    if timed_out {
        data["error"] = Value::String(format!("command timed out after {timeout_secs}s"));
    }
    Ok(data)
}

async fn read_command_stream<R>(
    stream: ToolOutputStream,
    mut reader: R,
    tx: mpsc::UnboundedSender<(ToolOutputStream, String)>,
) where
    R: AsyncRead + Unpin + Send + 'static,
{
    let mut buffer = [0u8; 8192];
    loop {
        match reader.read(&mut buffer).await {
            Ok(0) => break,
            Ok(read) => {
                let chunk = String::from_utf8_lossy(&buffer[..read]).to_string();
                if tx.send((stream, chunk)).is_err() {
                    break;
                }
            }
            Err(error) => {
                warn!(stream = stream.as_str(), error = %error, "failed to read command output stream");
                break;
            }
        }
    }
}

/// Inline presentation of one command output stream (stdout or stderr).
struct StreamOutput {
    /// Full text while small enough; a short preview once folded.
    text: String,
    /// Inline text is incomplete (folded or truncation fallback).
    truncated: bool,
    /// Full output was persisted to a local artifact.
    folded: bool,
    local_path: Option<String>,
}

/// Where streaming output is currently being accumulated.
enum StreamSink {
    /// Buffering inline until the fold threshold is crossed.
    Buffering,
    /// Folded: full output is streamed to a local artifact, inline keeps a preview.
    Writing(std::fs::File),
    /// Artifact writing failed; fall back to the legacy inline truncation.
    Truncating,
}

struct StreamAccumulator {
    output: StreamOutput,
    sink: StreamSink,
}

impl StreamAccumulator {
    fn new() -> Self {
        Self {
            output: StreamOutput {
                text: String::new(),
                truncated: false,
                folded: false,
                local_path: None,
            },
            sink: StreamSink::Buffering,
        }
    }

    fn push(&mut self, chunk: &str) {
        match &mut self.sink {
            StreamSink::Writing(file) => {
                if let Err(error) = file.write_all(chunk.as_bytes()) {
                    warn!(
                        error = %error,
                        "failed to write folded command artifact; truncating inline instead"
                    );
                    self.sink = StreamSink::Truncating;
                }
            }
            StreamSink::Truncating => self.push_truncating(chunk),
            StreamSink::Buffering => {
                self.output.text.push_str(chunk);
                if self.output.text.len() > FOLD_THRESHOLD_BYTES {
                    self.start_fold();
                }
            }
        }
    }

    fn push_truncating(&mut self, chunk: &str) {
        if self.output.text.len() >= MAX_COMMAND_OUTPUT_BYTES {
            self.output.truncated = true;
            return;
        }
        self.output.text.push_str(chunk);
        if self.output.text.len() > MAX_COMMAND_OUTPUT_BYTES {
            self.output.truncated = true;
            self.output.text.truncate(MAX_COMMAND_OUTPUT_BYTES);
            while !self.output.text.is_char_boundary(self.output.text.len()) {
                self.output.text.pop();
            }
        }
    }

    /// Fold the buffered text to a local artifact, keeping only a preview inline.
    fn start_fold(&mut self) {
        let (file, path) = match open_stream_artifact("out") {
            Ok(ok) => ok,
            Err(error) => {
                warn!(
                    error = %error,
                    "failed to open command artifact; truncating inline instead"
                );
                self.sink = StreamSink::Truncating;
                return;
            }
        };
        let mut file = file;
        if let Err(error) = file.write_all(self.output.text.as_bytes()) {
            warn!(
                error = %error,
                "failed to seed command artifact; truncating inline instead"
            );
            self.sink = StreamSink::Truncating;
            return;
        }
        self.output.text = fold_preview(&self.output.text, FOLD_PREVIEW_CHARS);
        self.output.truncated = true;
        self.output.folded = true;
        self.output.local_path = Some(path);
        self.sink = StreamSink::Writing(file);
    }
}

/// Fold output that was fully captured in memory (non-streaming path). Persists
/// the whole body to a content-addressed artifact when it crosses the threshold.
async fn fold_bytes_output(bytes: &[u8], ext: &str) -> StreamOutput {
    let lost = String::from_utf8_lossy(bytes);
    if bytes.len() <= FOLD_THRESHOLD_BYTES {
        return StreamOutput {
            text: lost.into_owned(),
            truncated: false,
            folded: false,
            local_path: None,
        };
    }
    match write_artifact(COMMAND_ARTIFACT_DIR, bytes, ext).await {
        Ok(path) => StreamOutput {
            text: fold_preview(&lost, FOLD_PREVIEW_CHARS),
            truncated: true,
            folded: true,
            local_path: Some(path),
        },
        Err(error) => {
            warn!(
                error = %error,
                "failed to fold large command output; truncating inline instead"
            );
            let truncated = truncate_utf8(&lost, MAX_COMMAND_OUTPUT_BYTES);
            StreamOutput {
                text: truncated.text,
                truncated: true,
                folded: false,
                local_path: None,
            }
        }
    }
}

fn add_stream_fields(map: &mut serde_json::Map<String, Value>, label: &str, stream: &StreamOutput) {
    map.insert(label.to_string(), json!(stream.text));
    map.insert(format!("{label}_truncated"), json!(stream.truncated));
    if stream.folded {
        map.insert(format!("{label}_folded"), json!(true));
        map.insert(format!("{label}_local_path"), json!(stream.local_path));
    }
}

fn open_stream_artifact(ext: &str) -> Result<(std::fs::File, String)> {
    let dir = std::env::temp_dir().join(COMMAND_ARTIFACT_DIR);
    std::fs::create_dir_all(&dir)
        .with_context(|| format!("failed to create command artifact dir {}", dir.display()))?;
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .context("system time")?
        .as_nanos();
    let name = format!("stream-{}-{}.{}", std::process::id(), nanos, ext);
    let path = dir.join(name);
    let file = std::fs::File::create(&path)
        .with_context(|| format!("failed to create command artifact {}", path.display()))?;
    Ok((file, path.to_string_lossy().to_string()))
}

fn shell_invocation() -> (&'static str, &'static str) {
    if cfg!(windows) {
        ("cmd", "/C")
    } else {
        ("/bin/sh", "-c")
    }
}

struct TruncatedText {
    text: String,
    truncated: bool,
}
fn truncate_utf8(text: &str, max_bytes: usize) -> TruncatedText {
    if text.len() <= max_bytes {
        return TruncatedText {
            text: text.to_string(),
            truncated: false,
        };
    }
    let mut end = max_bytes;
    while !text.is_char_boundary(end) {
        end -= 1;
    }
    TruncatedText {
        text: text[..end].to_string(),
        truncated: true,
    }
}

#[cfg(test)]
mod tests {
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

    #[cfg(unix)]
    #[tokio::test]
    async fn streaming_shell_command_honors_explicit_timeout() {
        use super::ToolHandler;

        let mut emit = |_stream, _chunk| Ok(());
        let result = super::RunCommandTool
            .execute_streaming(
                json!({"command": "sleep 2", "timeout_secs": 1}),
                super::ToolExecutionContext::default(),
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
            super::run_workspace_shell_command_streaming(&script, 30, &mut emit).await
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

    #[cfg(unix)]
    #[tokio::test]
    async fn streaming_shell_command_folds_large_output() {
        use super::ToolHandler;

        let mut emit = |_stream, _chunk| Ok(());
        let result = super::RunCommandTool
            .execute_streaming(
                json!({"command": "yes x | head -c 70000", "timeout_secs": 30}),
                super::ToolExecutionContext::default(),
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
