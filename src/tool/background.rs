//! Background shell tasks: registry, rolling log window, terminal delivery.

use std::collections::HashMap;
use std::fs::{File, OpenOptions};
use std::io::{Seek, SeekFrom, Write};
use std::path::PathBuf;
use std::sync::{Arc, Mutex, MutexGuard, OnceLock};

use anyhow::{Context, Result};
use tokio::sync::oneshot;

use super::fold_artifact::COMMAND_ARTIFACT_DIR;

pub(super) const MAX_BACKGROUND_TASKS: usize = 8;
const LOG_WINDOW_BYTES: u64 = 32 * 1024 * 1024;
const INLINE_BUFFER_BYTES: usize = 64 * 1024;
pub(super) const TERMINAL_PREVIEW_BYTES: usize = 8 * 1024;

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum TaskState {
    Running,
    Exited { code: Option<i32> },
    Killed { reason: String },
}

impl TaskState {
    pub(crate) fn as_str(&self) -> &'static str {
        match self {
            Self::Running => "running",
            Self::Exited { .. } => "exited",
            Self::Killed { .. } => "killed",
        }
    }

    fn is_terminal(&self) -> bool {
        !matches!(self, Self::Running)
    }
}

#[derive(Debug, Clone)]
pub(crate) struct TaskOutcome {
    pub task_id: String,
    pub session_id: Option<String>,
    pub command: String,
    pub state: TaskState,
    pub duration_ms: u64,
    pub output_bytes: u64,
    pub output_path: Option<String>,
    pub preview: String,
}

pub(crate) enum BackgroundEvent {
    Started {
        task_id: String,
        session_id: Option<String>,
        command: String,
        title: Option<String>,
    },
    Completed(TaskOutcome),
}

pub(crate) type Notifier = Arc<dyn Fn(BackgroundEvent) + Send + Sync>;

#[derive(Debug, Default)]
pub(super) struct LogState {
    path: Option<PathBuf>,
    dropped: u64,
    written: u64,
}

pub(super) struct OutputSink {
    task_id: String,
    buffer: Vec<u8>,
    file: Option<File>,
    state: Arc<Mutex<LogState>>,
}

impl OutputSink {
    pub(super) fn new(task_id: String) -> Self {
        Self {
            task_id,
            buffer: Vec::new(),
            file: None,
            state: Arc::new(Mutex::new(LogState::default())),
        }
    }

    pub(super) fn state(&self) -> Arc<Mutex<LogState>> {
        Arc::clone(&self.state)
    }

    pub(super) fn push(&mut self, bytes: &[u8]) {
        if bytes.is_empty() {
            return;
        }
        if self.file.is_some() {
            self.write_to_file(bytes);
            return;
        }
        self.buffer.extend_from_slice(bytes);
        if self.buffer.len() > INLINE_BUFFER_BYTES {
            let _ = self.persist();
        }
    }

    pub(super) fn persist(&mut self) -> Result<()> {
        if self.file.is_some() {
            return Ok(());
        }
        let dir = std::env::temp_dir().join(COMMAND_ARTIFACT_DIR);
        std::fs::create_dir_all(&dir)
            .with_context(|| format!("failed to create background log dir {}", dir.display()))?;
        let path = dir.join(format!("{}.log", self.task_id));
        let mut file = OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(true)
            .open(&path)
            .with_context(|| format!("failed to open background log {}", path.display()))?;
        file.write_all(&self.buffer)
            .with_context(|| format!("failed to seed background log {}", path.display()))?;
        let written = self.buffer.len() as u64;
        self.buffer.clear();
        {
            let mut state = lock(&self.state);
            state.path = Some(path);
            state.written = written;
            state.dropped = 0;
        }
        self.file = Some(file);
        Ok(())
    }

    pub(super) fn tail(&mut self, max_bytes: usize) -> String {
        if self.file.is_some() {
            let path = lock(&self.state).path.clone();
            if let Some(path) = path {
                return read_tail(&path, max_bytes).unwrap_or_default();
            }
            return String::new();
        }
        let len = self.buffer.len();
        let mut start = len.saturating_sub(max_bytes);
        while start < len && !is_char_boundary(&self.buffer, start) {
            start += 1;
        }
        String::from_utf8_lossy(&self.buffer[start..]).into_owned()
    }

    pub(super) fn total_bytes(&self) -> u64 {
        let state = lock(&self.state);
        state.dropped + state.written
    }

    pub(super) fn output_path(&self) -> Option<PathBuf> {
        lock(&self.state).path.clone()
    }

    fn write_to_file(&mut self, bytes: &[u8]) {
        let Some(file) = self.file.as_mut() else {
            return;
        };
        let mut state = lock(&self.state);
        let len = bytes.len() as u64;
        if state.written > 0
            && state.written.saturating_add(len) > LOG_WINDOW_BYTES
            && file.set_len(0).is_ok()
            && file.seek(SeekFrom::Start(0)).is_ok()
        {
            state.dropped = state.dropped.saturating_add(state.written);
            state.written = 0;
        }
        if file.write_all(bytes).is_ok() {
            state.written = state.written.saturating_add(len);
        }
    }
}

fn read_tail(path: &PathBuf, max_bytes: usize) -> Result<String> {
    let bytes = std::fs::read(path)?;
    let start = bytes.len().saturating_sub(max_bytes);
    let mut start = start;
    while start < bytes.len() && !is_char_boundary(&bytes, start) {
        start += 1;
    }
    Ok(String::from_utf8_lossy(&bytes[start..]).into_owned())
}

fn is_char_boundary(bytes: &[u8], index: usize) -> bool {
    if index == 0 || index >= bytes.len() {
        return true;
    }
    bytes[index] & 0b1100_0000 != 0b1000_0000
}

struct TaskRecord {
    session_id: Option<String>,
    command: String,
    state: TaskState,
    log: Option<Arc<Mutex<LogState>>>,
    cancel: Option<oneshot::Sender<()>>,
    observed: bool,
}

#[derive(Default)]
struct Registry {
    next_id: u64,
    notifier: Option<Notifier>,
    tasks: HashMap<String, TaskRecord>,
}

fn registry() -> &'static Mutex<Registry> {
    static REGISTRY: OnceLock<Mutex<Registry>> = OnceLock::new();
    REGISTRY.get_or_init(|| Mutex::new(Registry::default()))
}

fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(|error| error.into_inner())
}

fn live_count(registry: &Registry, session_id: Option<&str>) -> usize {
    registry
        .tasks
        .values()
        .filter(|task| !task.state.is_terminal() && task.session_id.as_deref() == session_id)
        .count()
}

pub(super) fn allocate_id() -> String {
    let mut registry = lock(registry());
    registry.next_id += 1;
    format!("bg-{}", registry.next_id)
}

pub(crate) fn set_notifier(notifier: Notifier) {
    lock(registry()).notifier = Some(notifier);
}

pub(super) fn register(
    task_id: &str,
    session_id: Option<&str>,
    command: &str,
    title: Option<&str>,
    log: Arc<Mutex<LogState>>,
    cancel: &mut Option<oneshot::Sender<()>>,
) -> bool {
    let notifier = {
        let mut registry = lock(registry());
        if live_count(&registry, session_id) >= MAX_BACKGROUND_TASKS {
            return false;
        }
        let Some(cancel) = cancel.take() else {
            return false;
        };
        registry.tasks.insert(
            task_id.to_string(),
            TaskRecord {
                session_id: session_id.map(str::to_string),
                command: command.to_string(),
                state: TaskState::Running,
                log: Some(log),
                cancel: Some(cancel),
                observed: false,
            },
        );
        registry.notifier.clone()
    };
    if let Some(notifier) = notifier {
        notifier(BackgroundEvent::Started {
            task_id: task_id.to_string(),
            session_id: session_id.map(str::to_string),
            command: command.to_string(),
            title: title.map(str::to_string),
        });
    }
    true
}

pub(crate) fn complete(
    task_id: &str,
    state: TaskState,
    duration_ms: u64,
    output_bytes: u64,
    output_path: Option<PathBuf>,
    preview: String,
) -> Option<TaskOutcome> {
    let (notifier, outcome) = {
        let mut registry = lock(registry());
        let task = registry.tasks.get_mut(task_id)?;
        if task.state.is_terminal() {
            return None;
        }
        task.state = state.clone();
        task.cancel = None;
        let outcome = TaskOutcome {
            task_id: task_id.to_string(),
            session_id: task.session_id.clone(),
            command: task.command.clone(),
            state,
            duration_ms,
            output_bytes,
            output_path: output_path.map(|path| path.to_string_lossy().to_string()),
            preview,
        };
        (registry.notifier.clone(), outcome)
    };
    if let Some(notifier) = notifier {
        notifier(BackgroundEvent::Completed(outcome.clone()));
    }
    Some(outcome)
}

pub(super) fn has_capacity(session_id: Option<&str>) -> bool {
    live_count(&lock(registry()), session_id) < MAX_BACKGROUND_TASKS
}

fn mark_observed(task_id: &str) {
    if let Some(task) = lock(registry()).tasks.get_mut(task_id) {
        task.observed = true;
    }
}

pub(crate) fn take_observed(task_id: &str) -> bool {
    let mut registry = lock(registry());
    let Some(task) = registry.tasks.get_mut(task_id) else {
        return false;
    };
    std::mem::replace(&mut task.observed, false)
}

pub(super) fn cancel(task_id: &str) -> bool {
    let mut registry = lock(registry());
    let Some(task) = registry.tasks.get_mut(task_id) else {
        return false;
    };
    if task.state.is_terminal() {
        return false;
    }
    if let Some(cancel) = task.cancel.take() {
        task.observed = true;
        let _ = cancel.send(());
        return true;
    }
    false
}

pub(super) struct OutputChunk {
    pub state: TaskState,
    pub window_start: u64,
    pub next_offset: u64,
    pub total_bytes: u64,
    pub output_path: Option<String>,
    pub text: String,
}

pub(super) fn read_output(task_id: &str, offset: u64, max_bytes: usize) -> Result<OutputChunk> {
    let (state, log) = {
        let registry = lock(registry());
        let Some(task) = registry.tasks.get(task_id) else {
            return Err(anyhow::anyhow!("unknown background task: {task_id}"));
        };
        (task.state.clone(), task.log.clone())
    };
    if state.is_terminal() {
        mark_observed(task_id);
    }
    let Some(log) = log else {
        return Ok(OutputChunk {
            state,
            window_start: 0,
            next_offset: 0,
            total_bytes: 0,
            output_path: None,
            text: String::new(),
        });
    };
    let (path, dropped, written) = {
        let log = lock(&log);
        (log.path.clone(), log.dropped, log.written)
    };
    let window_start = dropped;
    let end = dropped.saturating_add(written);
    let Some(path) = path else {
        return Ok(OutputChunk {
            state,
            window_start,
            next_offset: window_start,
            total_bytes: window_start,
            output_path: None,
            text: String::new(),
        });
    };
    let bytes = std::fs::read(&path)
        .with_context(|| format!("failed to read background log {}", path.display()))?;
    let begin = offset.max(window_start).saturating_sub(window_start) as usize;
    let begin = begin.min(bytes.len());
    let limit = bytes.len().min(begin.saturating_add(max_bytes));
    let mut begin = begin;
    while begin < limit && !is_char_boundary(&bytes, begin) {
        begin += 1;
    }
    let mut limit = limit;
    while limit > begin && !is_char_boundary(&bytes, limit) {
        limit -= 1;
    }
    let text = String::from_utf8_lossy(&bytes[begin..limit]).into_owned();
    Ok(OutputChunk {
        state,
        window_start,
        next_offset: window_start.saturating_add(limit as u64),
        total_bytes: end,
        output_path: Some(path.to_string_lossy().to_string()),
        text,
    })
}

pub(crate) fn end_session(session_id: &str) {
    let mut removed = Vec::new();
    {
        let mut registry = lock(registry());
        let ids = registry
            .tasks
            .iter()
            .filter(|(_, task)| task.session_id.as_deref() == Some(session_id))
            .map(|(id, _)| id.clone())
            .collect::<Vec<_>>();
        for id in ids {
            if let Some(task) = registry.tasks.remove(&id) {
                removed.push(task);
            }
        }
    }
    for mut task in removed {
        if let Some(cancel) = task.cancel.take() {
            let _ = cancel.send(());
        }
        if let Some(log) = task.log.as_ref() {
            let path = lock(log).path.clone();
            if let Some(path) = path {
                let _ = std::fs::remove_file(path);
            }
        }
    }
}
