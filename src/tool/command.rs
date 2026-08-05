use anyhow::{Context, Result, anyhow};
use async_trait::async_trait;
use serde_json::{Value, json};
use std::process::Stdio;
use tokio::io::{AsyncRead, AsyncReadExt};
use tokio::process::Command;
use tokio::sync::mpsc;
use tokio::time::{Duration, sleep, timeout};
use tracing::{debug, warn};

use super::{
    COMMAND_TIMEOUT_SECS, ToolExecutionContext, ToolHandler, ToolOutputEmitter, ToolOutputStream,
    ToolRegistry, required_string, workspace_root,
};

const MAX_COMMAND_OUTPUT_BYTES: usize = 256 * 1024;

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
                }
            },
            "required": ["command"],
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
    ) -> Result<Value> {
        let command = required_string(&args, "command")?;
        run_workspace_shell_command_streaming(command, COMMAND_TIMEOUT_SECS, emit).await
    }
}

async fn run_command(args: Value) -> Result<Value> {
    let command = required_string(&args, "command")?;
    run_workspace_shell_command(command, COMMAND_TIMEOUT_SECS).await
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

    let stdout = truncate_utf8(
        &String::from_utf8_lossy(&output.stdout),
        MAX_COMMAND_OUTPUT_BYTES,
    );
    let stderr = truncate_utf8(
        &String::from_utf8_lossy(&output.stderr),
        MAX_COMMAND_OUTPUT_BYTES,
    );

    Ok(json!({
        "command": command,
        "shell": shell,
        "status": output.status.code(),
        "success": output.status.success(),
        "stdout": stdout.text,
        "stdout_truncated": stdout.truncated,
        "stderr": stderr.text,
        "stderr_truncated": stderr.truncated,
    }))
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

    let mut data = json!({ "command": command, "shell": shell, "status": status.code(), "success": status.success() && !timed_out, "stdout": stdout.text, "stdout_truncated": stdout.truncated, "stderr": stderr.text, "stderr_truncated": stderr.truncated });
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

struct StreamAccumulator {
    text: String,
    truncated: bool,
}
impl StreamAccumulator {
    fn new() -> Self {
        Self {
            text: String::new(),
            truncated: false,
        }
    }
    fn push(&mut self, chunk: &str) {
        if self.text.len() >= MAX_COMMAND_OUTPUT_BYTES {
            self.truncated = true;
            return;
        }
        self.text.push_str(chunk);
        if self.text.len() > MAX_COMMAND_OUTPUT_BYTES {
            self.truncated = true;
            self.text.truncate(MAX_COMMAND_OUTPUT_BYTES);
            while !self.text.is_char_boundary(self.text.len()) {
                self.text.pop();
            }
        }
    }
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
    #[cfg(unix)]
    #[tokio::test]
    async fn managed_commands_complete_normally() {
        let args = vec!["-c".to_string(), "printf command-ok".to_string()];
        let command = super::run_workspace_command("sh", &args, 2)
            .await
            .expect("command output");
        assert_eq!(command["stdout"], json!("command-ok"));
        assert_eq!(command["success"], json!(true));
        let shell = super::run_workspace_shell_command("printf shell-ok", 2)
            .await
            .expect("shell output");
        assert_eq!(shell["stdout"], json!("shell-ok"));
        assert_eq!(shell["success"], json!(true));
    }
}
