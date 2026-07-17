use std::sync::{Arc, Mutex as StdMutex};

use anyhow::{Result, anyhow};
use async_openai::config::Config;

use crate::agent::Agent;
use crate::runtime_context::RuntimeActiveContext;
use crate::transcript::{TranscriptRecorder, read_records, transcript_projection};

use super::restore_projection::project_runtime_restore_snapshot_with_children;

pub(super) struct PreparedContextScope {
    state: Arc<StdMutex<crate::transcript::ContextScopeState>>,
    restore_point: Option<(
        crate::transcript::ActiveContextExperiment,
        Vec<crate::protocol_frames::ProtocolFrame>,
        crate::runtime_context::RuntimeSnapshot,
    )>,
}

pub(super) fn prepare_context_scope(recorder: &TranscriptRecorder) -> Result<PreparedContextScope> {
    let state = recorder.context_scope_state();
    let restore_point = if let Some(scope) = recorder.active_context_experiment() {
        let snapshot = project_runtime_restore_snapshot_with_children(
            recorder.session_id(),
            read_records(recorder.path())?,
            transcript_projection::SessionContextCursor {
                branch_id: Some(scope.parent_branch_id.clone()),
                leaf_sequence: Some(scope.base_sequence),
            },
            recorder
                .path()
                .parent()
                .ok_or_else(|| anyhow!("transcript path has no parent directory"))?,
        )?;
        RuntimeActiveContext::try_from(&snapshot.snapshot)?;
        Some((scope, snapshot.protocol_frames, snapshot.snapshot))
    } else {
        None
    };
    Ok(PreparedContextScope {
        state,
        restore_point,
    })
}

pub(super) fn apply_prepared_context_scope<C>(agent: &mut Agent<C>, prepared: PreparedContextScope)
where
    C: Config,
{
    agent.set_context_scope_state(prepared.state);
    if let Some((scope, frames, snapshot)) = prepared.restore_point {
        agent.set_context_experiment_restore_point(scope, frames, snapshot);
    } else {
        agent.clear_context_experiment_restore_point();
    }
}
