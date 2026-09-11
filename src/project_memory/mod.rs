//! Persistent, workspace-scoped knowledge, independent of session compaction.
pub(crate) mod extract;
pub(crate) mod store;

use crate::agent::{Agent, AgentFactory, AgentTemplate};
use anyhow::{Context, Result, anyhow};
use std::path::Path;
use std::sync::{LazyLock, Mutex};
use std::time::Duration;
use store::MemoryStore;

static STORE: LazyLock<Mutex<Option<MemoryStore>>> = LazyLock::new(|| Mutex::new(None));

pub(crate) fn configure(root: &Path, workspace: &Path) -> Result<()> {
    let store = MemoryStore::open(root, workspace)?;
    *STORE
        .lock()
        .map_err(|_| anyhow!("project memory configuration poisoned"))? = Some(store);
    Ok(())
}

pub(crate) fn configured_store() -> Result<Option<MemoryStore>> {
    Ok(STORE
        .lock()
        .map_err(|_| anyhow!("project memory configuration poisoned"))?
        .clone())
}

/// Register only the journal frontier observed on first use of this system.
/// Resuming an older session does not implicitly import its earlier messages.
pub(crate) fn enroll(recorder: &crate::transcript::TranscriptRecorder) -> Result<()> {
    if let Some(store) = configured_store()? {
        store.register_source(recorder.session_id(), recorder.path(), recorder.sequence)?;
    }
    Ok(())
}

#[derive(Default)]
pub(crate) struct MemoryWorker {
    task: Option<tokio::task::JoinHandle<Result<()>>>,
}

impl MemoryWorker {
    pub(crate) async fn tick(&mut self, parent: &Agent) -> Result<()> {
        if self.task.as_ref().is_some_and(|task| !task.is_finished()) {
            return Ok(());
        }
        if let Some(task) = self.task.take() {
            task.await.context("project memory worker failed")??;
        }
        let Some(store) = configured_store()? else {
            return Ok(());
        };
        let mut template = AgentTemplate::historian();
        template.system_prompt = extract::MEMORY_PROMPT.into();
        template.purpose = "后台项目记忆整理".into();
        let helper = AgentFactory::create_child_with_route_and_max_tool_calls(
            parent,
            &template,
            None,
            false,
            Some(0),
        )?;
        self.task = Some(tokio::spawn(
            async move { process_pending(store, helper).await },
        ));
        Ok(())
    }

    pub(crate) async fn shutdown(&mut self) -> Result<()> {
        let Some(mut task) = self.task.take() else {
            return Ok(());
        };
        match tokio::time::timeout(Duration::from_secs(2), &mut task).await {
            Ok(result) => {
                result.context("project memory worker failed during shutdown")??;
            }
            Err(_) => {
                task.abort();
                let _ = task.await;
            }
        }
        Ok(())
    }
}

impl Drop for MemoryWorker {
    fn drop(&mut self) {
        if let Some(task) = self.task.take() {
            task.abort();
        }
    }
}

async fn blocking<T: Send + 'static>(f: impl FnOnce() -> Result<T> + Send + 'static) -> Result<T> {
    tokio::task::spawn_blocking(f)
        .await
        .context("project memory worker join failed")?
}

async fn process_pending(store: MemoryStore, helper: Agent) -> Result<()> {
    use fs4::fs_std::FileExt;
    let lock_path = store.worker_lock_path();
    let lock = blocking(move || {
        let file = std::fs::OpenOptions::new()
            .create(true)
            .truncate(false)
            .read(true)
            .write(true)
            .open(lock_path)?;
        match file.try_lock_exclusive() {
            Ok(true) => Ok(Some(file)),
            Ok(false) => Ok(None),
            Err(error) => Err(error.into()),
        }
    })
    .await?;
    let Some(_lock) = lock else {
        return Ok(());
    };
    let source_store = store.clone();
    let sources = blocking(move || source_store.sources()).await?;
    // One pass is bounded; unfinished sources are resumed on subsequent idle ticks.
    let mut batches = 0usize;
    for source in sources {
        if batches >= 8 {
            break;
        }
        let input_store = store.clone();
        let input_source = source.clone();
        let prepared = blocking(move || {
            let known = input_store.known_memories(10_000)?;
            let mut records =
                crate::transcript::read_records_allow_partial_tail(&input_source.path)?;
            records.retain(|record| record.sequence <= input_source.observed_sequence);
            let batch = extract::prepare_batch(&records, input_source.cursor, &known)?;
            Ok((known, batch))
        })
        .await;
        let result = async {
            let (known, batch) = prepared?;
            let Some(batch) = batch else { return Ok(()); };
            batches += 1;
            let (text, usage) = tokio::time::timeout(
                Duration::from_secs(180),
                helper.run_structured_oneshot(
                    &batch.input,
                    extract::structured_output,
                    |_: &str| std::future::ready(Ok::<(), crate::model_runtime::ModelFailure>(())),
                ),
            )
            .await
            .context("project memory extraction timed out")??;
            let update = extract::parse_update(&text, &batch, &known)?;
            let ids = known.iter().map(|record| record.id.clone()).collect::<Vec<_>>();
            let commit_store = store.clone();
            let commit_source = source.clone();
            blocking(move || commit_store.complete(&commit_source, batch.end_sequence, &batch.branch_id, &update, &ids)).await?;
            tracing::info!(session_id = %source.session_id, usage = ?usage, "project memory extraction completed");
            Ok::<_, anyhow::Error>(())
        }.await;
        if let Err(error) = result {
            tracing::warn!(session_id = %source.session_id, error = %error, "project memory extraction failed");
            let failure_store = store.clone();
            let failure_source = source.clone();
            blocking(move || {
                failure_store.fail_source(
                    &failure_source,
                    "Memory extraction failed; see application log for details",
                )
            })
            .await?;
        }
    }
    Ok(())
}
