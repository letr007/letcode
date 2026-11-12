use anyhow::{Context, Result, anyhow};
use async_trait::async_trait;
use serde_json::{Value, json};
use std::io::Write;
use std::path::Path;
use std::process::Stdio;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::{Instant, SystemTime, UNIX_EPOCH};
use tokio::io::{AsyncRead, AsyncReadExt};
use tokio::process::Command;
use tokio::sync::{mpsc, oneshot};
use tokio::time::{Duration, sleep, timeout};
use tracing::{debug, warn};

use crate::process_tree::ManagedChild;

use super::args::{optional_bool, optional_string, optional_usize, required_string};
use super::background::{self, OutputSink, TERMINAL_PREVIEW_BYTES};
use super::delegation::optional_u64;
use super::fold_artifact::{
    COMMAND_ARTIFACT_DIR, FOLD_PREVIEW_CHARS, FOLD_THRESHOLD_BYTES, fold_preview, write_artifact,
};
use super::{
    ToolExecutionContext, ToolHandler, ToolOutputEmitter, ToolOutputStream, ToolRegistry,
    ToolResult, workspace_root,
};

const MAX_COMMAND_OUTPUT_BYTES: usize = 256 * 1024;
const MAX_COMMAND_TIMEOUT_SECS: u64 = 3_600;
const SHELL_TIMEOUT_SECS: u64 = 120;
const MAX_OUTPUT_CHUNK_BYTES: usize = 64 * 1024;

pub(super) fn register(registry: &mut ToolRegistry) {
    registry.register(RunCommandTool);
    registry.register(ShellOutputTool);
    registry.register(ShellCancelTool);
}

struct RunCommandTool;

#[async_trait]
impl ToolHandler for RunCommandTool {
    fn name(&self) -> &'static str {
        "shell__exec"
    }

    fn description(&self) -> &'static str {
        "Run a shell command in the current workspace when specialized tools are not a better fit (prefer fs__/search__/git__/edit tools for file and repo work). On Windows, commands prefer PowerShell 7 (pwsh.exe), then Windows PowerShell (powershell.exe), and fall back to cmd.exe; other platforms use /bin/sh. Avoid high-impact irreversible commands without clear scope; keep compound commands from chaining steps that need separate confirmation; ensure loops/listeners have exit/timeout limits. Authorization is handled by the tool-level permission policy."
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "Shell command to run. On Windows prefers pwsh.exe, then powershell.exe, then cmd.exe; uses /bin/sh elsewhere, e.g. cargo check"
                },
                "title": {
                    "type": ["string", "null"],
                    "maxLength": 80,
                    "description": "Short user-facing title for this command, in the user's language. Describe the purpose, not the command line; omit paths and secrets. Used by task panels and tool cards."
                },
                "timeout_secs": {
                    "type": ["integer", "null"],
                    "minimum": 1,
                    "maximum": MAX_COMMAND_TIMEOUT_SECS,
                    "description": "How long to wait in the foreground, in seconds. Defaults to 120, maximum 3600. At the deadline the main session moves the command to the background and returns a task_id; without background delivery the command is terminated."
                },
                "background": {
                    "type": ["boolean", "null"],
                    "description": "Skip foreground waiting and move the command to the background immediately. Use for builds, installs, and other long jobs you do not need to block on. Available in the main session; subagent runs cannot receive background completions."
                }
            },
            // OpenAI strict tool schemas require every properties key to appear in
            // required (optional fields use type unions that include null).
            "required": ["command", "title", "timeout_secs", "background"],
            "additionalProperties": false
        })
    }

    async fn execute(&self, args: Value) -> Result<Value> {
        run_command(args).await
    }

    async fn execute_streaming(
        &self,
        args: Value,
        context: ToolExecutionContext,
        emit: ToolOutputEmitter<'_>,
    ) -> Result<super::ToolResult> {
        let command = required_string(&args, "command")?;
        let timeout_secs = command_timeout_secs(&args)?;
        let background = optional_bool(&args, "background").unwrap_or(false);
        let title = command_title(&args);
        let can_background = context.can_background_shell && context.session_id.is_some();
        if background && !can_background {
            return Err(anyhow!(
                "background shell requires a session that receives completions"
            ));
        }
        run_shell(
            command,
            timeout_secs,
            background,
            can_background,
            title.as_deref(),
            context.session_id.as_deref(),
            emit,
        )
        .await
    }
}

fn command_title(args: &Value) -> Option<String> {
    optional_string(args, "title")
        .map(str::trim)
        .filter(|title| !title.is_empty())
        .map(str::to_string)
}

fn command_timeout_secs(args: &Value) -> Result<u64> {
    let timeout_secs = optional_u64(args, "timeout_secs")?.unwrap_or(SHELL_TIMEOUT_SECS);
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
    let (shell, mut child) = spawn_workspace_shell(command, &root)?;
    debug!(command = %command, shell = %shell, "running workspace shell command");
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

struct ProcessExit {
    status: Option<std::process::ExitStatus>,
    killed: Option<String>,
}

// Dropping the guard aborts the driver, which terminates the process group.
struct ForegroundGuard {
    handle: Option<tokio::task::JoinHandle<()>>,
    task_id: String,
    promotable: Arc<AtomicBool>,
}

impl ForegroundGuard {
    fn disarm(&mut self) {
        self.handle = None;
    }
}

impl Drop for ForegroundGuard {
    fn drop(&mut self) {
        self.promotable.store(false, Ordering::Relaxed);
        background::discard_pending(&self.task_id);
        if let Some(handle) = self.handle.take() {
            handle.abort();
        }
    }
}

fn lock_sink(sink: &Arc<Mutex<OutputSink>>) -> MutexGuard<'_, OutputSink> {
    sink.lock().unwrap_or_else(|error| error.into_inner())
}

async fn run_shell(
    command: &str,
    timeout_secs: u64,
    background: bool,
    can_background: bool,
    title: Option<&str>,
    session_id: Option<&str>,
    emit: ToolOutputEmitter<'_>,
) -> Result<ToolResult> {
    let root = workspace_root()?;
    let (shell, mut child) = spawn_workspace_shell(command, &root)
        .with_context(|| format!("failed to spawn shell command: {command}"))?;
    debug!(command = %command, shell = %shell, "running streaming workspace shell command");
    let stdout = child
        .take_stdout()
        .ok_or_else(|| anyhow!("failed to capture command stdout"))?;
    let stderr = child
        .take_stderr()
        .ok_or_else(|| anyhow!("failed to capture command stderr"))?;

    let task_id = background::allocate_id();
    let sink = Arc::new(Mutex::new(OutputSink::new(task_id.clone())));
    let (delta_tx, mut delta_rx) = mpsc::unbounded_channel();
    let (done_tx, mut done_rx) = oneshot::channel();
    let (cancel_tx, cancel_rx) = oneshot::channel();
    let started = Instant::now();

    let mut cancel_tx = Some(cancel_tx);
    let promoted = background
        && can_background
        && promote(&task_id, session_id, command, title, &sink, &mut cancel_tx)
            == background::RegisterOutcome::Registered;
    let promotable = Arc::new(AtomicBool::new(can_background && !promoted));

    let mut run = ForegroundGuard {
        handle: Some(tokio::spawn(drive_process(
            task_id.clone(),
            child,
            stdout,
            stderr,
            Arc::clone(&sink),
            delta_tx,
            done_tx,
            cancel_rx,
            started,
            Arc::clone(&promotable),
        ))),
        task_id: task_id.clone(),
        promotable: Arc::clone(&promotable),
    };

    let mut stdout_acc = StreamAccumulator::new();
    let mut stderr_acc = StreamAccumulator::new();

    if promoted {
        run.disarm();
        return Ok(background_result(&task_id, command, shell, &sink));
    }

    let deadline = sleep(Duration::from_secs(timeout_secs));
    tokio::pin!(deadline);

    loop {
        tokio::select! {
            Some((stream, chunk)) = delta_rx.recv() => {
                match stream {
                    ToolOutputStream::Stdout => stdout_acc.push(&chunk),
                    ToolOutputStream::Stderr => stderr_acc.push(&chunk),
                }
                emit(stream, chunk)?;
            }
            result = &mut done_rx => {
                let exit = result.ok();
                drain_delta(&mut delta_rx, &mut stdout_acc, &mut stderr_acc);
                return Ok(exited_result(command, shell, exit, &stdout_acc, &stderr_acc));
            }
            _ = &mut deadline => {
                if can_background {
                    match promote(&task_id, session_id, command, title, &sink, &mut cancel_tx) {
                        background::RegisterOutcome::Registered => {
                            run.disarm();
                            return Ok(background_result(&task_id, command, shell, &sink));
                        }
                        background::RegisterOutcome::AlreadyCompleted => {
                            if let Some(path) = lock_sink(&sink).output_path() {
                                let _ = std::fs::remove_file(path);
                            }
                            let exit = done_rx.await.ok();
                            drain_delta(&mut delta_rx, &mut stdout_acc, &mut stderr_acc);
                            return Ok(exited_result(
                                command,
                                shell,
                                exit,
                                &stdout_acc,
                                &stderr_acc,
                            ));
                        }
                        background::RegisterOutcome::Rejected => {}
                    }
                }
                if let Some(tx) = cancel_tx.take() {
                    let _ = tx.send(());
                }
                let reason = if can_background {
                    "background_capacity_exceeded"
                } else {
                    "background_unavailable"
                };
                return Ok(killed_result(
                    command,
                    shell,
                    reason,
                    &stdout_acc,
                    &stderr_acc,
                ));
            }
        }
    }
}

async fn drive_process(
    task_id: String,
    mut child: ManagedChild,
    stdout: tokio::process::ChildStdout,
    stderr: tokio::process::ChildStderr,
    sink: Arc<Mutex<OutputSink>>,
    delta_tx: mpsc::UnboundedSender<(ToolOutputStream, String)>,
    done_tx: oneshot::Sender<ProcessExit>,
    cancel_rx: oneshot::Receiver<()>,
    started: Instant,
    promotable: Arc<AtomicBool>,
) {
    let stdout_task = tokio::spawn(read_command_stream(
        ToolOutputStream::Stdout,
        stdout,
        Arc::clone(&sink),
        delta_tx.clone(),
    ));
    let stderr_task = tokio::spawn(read_command_stream(
        ToolOutputStream::Stderr,
        stderr,
        Arc::clone(&sink),
        delta_tx.clone(),
    ));
    drop(delta_tx);

    let mut cancelled = false;
    let waited = tokio::select! {
        status = child.wait() => Some(status),
        _ = cancel_rx => {
            cancelled = true;
            None
        }
    };
    let exit = if cancelled {
        ProcessExit {
            status: child.terminate_and_wait().await.ok(),
            killed: Some("cancelled".to_string()),
        }
    } else {
        match waited {
            Some(Ok(status)) => ProcessExit {
                status: Some(status),
                killed: None,
            },
            _ => ProcessExit {
                status: None,
                killed: Some("wait_failed".to_string()),
            },
        }
    };
    child.disarm();
    let _ = stdout_task.await;
    let _ = stderr_task.await;

    let (preview, total, path) = {
        let mut guard = lock_sink(&sink);
        (
            guard.tail(TERMINAL_PREVIEW_BYTES),
            guard.total_bytes(),
            guard.output_path(),
        )
    };
    let state = match (exit.killed.as_deref(), exit.status) {
        (Some(reason), _) => background::TaskState::Killed {
            reason: reason.to_string(),
        },
        (None, Some(status)) => background::TaskState::Exited {
            code: status.code(),
        },
        (None, None) => background::TaskState::Killed {
            reason: "unknown".to_string(),
        },
    };
    let duration_ms = started.elapsed().as_millis() as u64;
    if background::complete(
        &task_id,
        state,
        duration_ms,
        total,
        path.clone(),
        preview,
        &promotable,
    )
    .is_none()
        && let Some(path) = path
    {
        let _ = std::fs::remove_file(path);
    }
    let _ = done_tx.send(exit);
}

fn drain_delta(
    rx: &mut mpsc::UnboundedReceiver<(ToolOutputStream, String)>,
    stdout: &mut StreamAccumulator,
    stderr: &mut StreamAccumulator,
) {
    while let Ok((stream, chunk)) = rx.try_recv() {
        match stream {
            ToolOutputStream::Stdout => stdout.push(&chunk),
            ToolOutputStream::Stderr => stderr.push(&chunk),
        }
    }
}

fn promote(
    task_id: &str,
    session_id: Option<&str>,
    command: &str,
    title: Option<&str>,
    sink: &Arc<Mutex<OutputSink>>,
    cancel_tx: &mut Option<oneshot::Sender<()>>,
) -> background::RegisterOutcome {
    if cancel_tx.is_none() || !background::has_capacity(session_id) {
        return background::RegisterOutcome::Rejected;
    }
    let log = {
        let mut guard = lock_sink(sink);
        if guard.persist().is_err() {
            return background::RegisterOutcome::Rejected;
        }
        guard.state()
    };
    background::register(task_id, session_id, command, title, log, cancel_tx)
}

fn background_result(
    task_id: &str,
    command: &str,
    shell: &str,
    sink: &Arc<Mutex<OutputSink>>,
) -> ToolResult {
    let (path, preview, total) = {
        let mut guard = lock_sink(sink);
        let _ = guard.persist();
        (
            guard.output_path(),
            guard.tail(FOLD_PREVIEW_CHARS),
            guard.total_bytes(),
        )
    };
    ToolResult::ok(
        "shell__exec",
        json!({
            "command": command,
            "shell": shell,
            "state": "background",
            "task_id": task_id,
            "output_path": path.map(|path| path.to_string_lossy().to_string()),
            "output_bytes": total,
            "preview": preview,
        }),
    )
}

fn exited_result(
    command: &str,
    shell: &str,
    exit: Option<ProcessExit>,
    stdout: &StreamAccumulator,
    stderr: &StreamAccumulator,
) -> ToolResult {
    let status = exit.as_ref().and_then(|exit| exit.status);
    let mut data = json!({
        "command": command,
        "shell": shell,
        "state": "exited",
        "status": status.and_then(|status| status.code()),
        "success": status.is_some_and(|status| status.success()),
    });
    let fields = data.as_object_mut().expect("command output object");
    add_stream_fields(fields, "stdout", &stdout.output);
    add_stream_fields(fields, "stderr", &stderr.output);
    ToolResult::ok("shell__exec", data)
}

fn killed_result(
    command: &str,
    shell: &str,
    reason: &str,
    stdout: &StreamAccumulator,
    stderr: &StreamAccumulator,
) -> ToolResult {
    let mut data = json!({
        "command": command,
        "shell": shell,
        "state": "killed",
        "reason": reason,
    });
    let fields = data.as_object_mut().expect("command output object");
    add_stream_fields(fields, "stdout", &stdout.output);
    add_stream_fields(fields, "stderr", &stderr.output);
    ToolResult::ok("shell__exec", data)
}

async fn read_command_stream<R>(
    stream: ToolOutputStream,
    mut reader: R,
    sink: Arc<Mutex<OutputSink>>,
    tx: mpsc::UnboundedSender<(ToolOutputStream, String)>,
) where
    R: AsyncRead + Unpin + Send + 'static,
{
    let mut buffer = [0u8; 8192];
    loop {
        match reader.read(&mut buffer).await {
            Ok(0) => break,
            Ok(read) => {
                lock_sink(&sink).push(&buffer[..read]);
                let chunk = String::from_utf8_lossy(&buffer[..read]).to_string();
                let _ = tx.send((stream, chunk));
            }
            Err(error) => {
                warn!(stream = stream.as_str(), error = %error, "failed to read command output stream");
                break;
            }
        }
    }
}

struct ShellOutputTool;

#[async_trait]
impl ToolHandler for ShellOutputTool {
    fn name(&self) -> &'static str {
        "shell__output"
    }

    fn description(&self) -> &'static str {
        "Read output from a background shell task starting at a byte offset. Returns the task state, the retained window start, the next offset, and the output text."
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "task_id": {
                    "type": "string",
                    "description": "Background task id returned by shell__exec."
                },
                "offset": {
                    "type": ["integer", "null"],
                    "minimum": 0,
                    "description": "Byte offset to resume from. Omit to continue from the start of the retained window."
                }
            },
            "required": ["task_id", "offset"],
            "additionalProperties": false
        })
    }

    async fn execute(&self, args: Value) -> Result<Value> {
        let task_id = required_string(&args, "task_id")?;
        let offset = optional_usize(&args, "offset").unwrap_or(0) as u64;
        let chunk = background::read_output(task_id, offset, MAX_OUTPUT_CHUNK_BYTES)?;
        let (exit_code, reason) = match &chunk.state {
            background::TaskState::Exited { code } => (*code, None),
            background::TaskState::Killed { reason } => (None, Some(reason.clone())),
            background::TaskState::Running => (None, None),
        };
        Ok(json!({
            "task_id": task_id,
            "state": chunk.state.as_str(),
            "exit_code": exit_code,
            "reason": reason,
            "window_start": chunk.window_start,
            "next_offset": chunk.next_offset,
            "total_bytes": chunk.total_bytes,
            "output_path": chunk.output_path,
            "text": chunk.text,
        }))
    }
}

struct ShellCancelTool;

#[async_trait]
impl ToolHandler for ShellCancelTool {
    fn name(&self) -> &'static str {
        "shell__cancel"
    }

    fn description(&self) -> &'static str {
        "Terminate a running background shell task."
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "task_id": {
                    "type": "string",
                    "description": "Background task id returned by shell__exec."
                }
            },
            "required": ["task_id"],
            "additionalProperties": false
        })
    }

    async fn execute(&self, args: Value) -> Result<Value> {
        let task_id = required_string(&args, "task_id")?;
        let cancelled = background::cancel(task_id);
        Ok(json!({ "task_id": task_id, "cancelled": cancelled }))
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

#[derive(Debug, Clone, Copy)]
struct ShellInvocation {
    executable: &'static str,
    args: &'static [&'static str],
}

#[cfg(windows)]
const SHELL_INVOCATIONS: &[ShellInvocation] = &[
    ShellInvocation {
        executable: "pwsh.exe",
        args: &["-NoLogo", "-NoProfile", "-NonInteractive", "-Command"],
    },
    ShellInvocation {
        executable: "powershell.exe",
        args: &["-NoLogo", "-NoProfile", "-NonInteractive", "-Command"],
    },
    ShellInvocation {
        executable: "cmd.exe",
        args: &["/D", "/S", "/C"],
    },
];

#[cfg(not(windows))]
const SHELL_INVOCATIONS: &[ShellInvocation] = &[ShellInvocation {
    executable: "/bin/sh",
    args: &["-c"],
}];

const UNCACHED_SHELL_INDEX: usize = usize::MAX;
static CACHED_SHELL_INDEX: AtomicUsize = AtomicUsize::new(UNCACHED_SHELL_INDEX);

fn spawn_workspace_shell(script: &str, root: &Path) -> Result<(&'static str, ManagedChild)> {
    let mut missing = Vec::new();
    let cached_index = cached_shell_index();

    if let Some(index) = cached_index {
        let invocation = &SHELL_INVOCATIONS[index];
        match spawn_shell_candidate(invocation, script, root) {
            Ok(child) => return Ok((invocation.executable, child)),
            Err(error) if executable_not_found(&error) => {
                invalidate_cached_shell(index);
                debug!(
                    shell = invocation.executable,
                    %error,
                    "cached shell executable unavailable; probing fallbacks"
                );
                missing.push(invocation.executable);
            }
            Err(error) => {
                return Err(error)
                    .with_context(|| format!("failed to spawn shell '{}'", invocation.executable));
            }
        }
    }

    for (index, invocation) in SHELL_INVOCATIONS.iter().enumerate() {
        if cached_index == Some(index) {
            continue;
        }
        match spawn_shell_candidate(invocation, script, root) {
            Ok(child) => {
                CACHED_SHELL_INDEX.store(index, Ordering::Release);
                return Ok((invocation.executable, child));
            }
            Err(error) if executable_not_found(&error) => {
                debug!(
                    shell = invocation.executable,
                    %error,
                    "shell executable unavailable; trying fallback"
                );
                missing.push(invocation.executable);
            }
            Err(error) => {
                return Err(error)
                    .with_context(|| format!("failed to spawn shell '{}'", invocation.executable));
            }
        }
    }
    Err(anyhow!(
        "no supported shell executable found (tried: {})",
        missing.join(", ")
    ))
}

fn cached_shell_index() -> Option<usize> {
    let index = CACHED_SHELL_INDEX.load(Ordering::Acquire);
    (index < SHELL_INVOCATIONS.len()).then_some(index)
}

fn invalidate_cached_shell(index: usize) {
    let _ = CACHED_SHELL_INDEX.compare_exchange(
        index,
        UNCACHED_SHELL_INDEX,
        Ordering::AcqRel,
        Ordering::Acquire,
    );
}

fn spawn_shell_candidate(
    invocation: &ShellInvocation,
    script: &str,
    root: &Path,
) -> Result<ManagedChild> {
    let mut command = Command::new(invocation.executable);
    configure_shell_command(&mut command, invocation, script);
    command
        .current_dir(root)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    ManagedChild::spawn(command)
}

fn executable_not_found(error: &anyhow::Error) -> bool {
    let cause: &(dyn std::error::Error + Send + Sync + 'static) = error.as_ref();
    cause
        .downcast_ref::<std::io::Error>()
        .is_some_and(|error| error.kind() == std::io::ErrorKind::NotFound)
}

fn configure_shell_command(command: &mut Command, invocation: &ShellInvocation, script: &str) {
    #[cfg(windows)]
    {
        let mut raw = invocation.args.join(" ");
        raw.push(' ');
        raw.push_str(&quote_windows_arg(script));
        command.raw_arg(raw);
    }
    #[cfg(not(windows))]
    {
        command.args(invocation.args.iter().copied()).arg(script);
    }
}

#[cfg(windows)]
fn quote_windows_arg(value: &str) -> String {
    let mut quoted = String::from("\"");
    let mut backslashes = 0;
    for ch in value.chars() {
        match ch {
            '\\' => backslashes += 1,
            '"' => {
                quoted.push_str(&"\\".repeat(backslashes * 2 + 1));
                quoted.push('"');
                backslashes = 0;
            }
            _ => {
                quoted.push_str(&"\\".repeat(backslashes));
                quoted.push(ch);
                backslashes = 0;
            }
        }
    }
    quoted.push_str(&"\\".repeat(backslashes * 2));
    quoted.push('"');
    quoted
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
#[path = "../../tests/internal/tool/command/tests.rs"]
mod tests;
