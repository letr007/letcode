//! Context-scope prepare/apply shared by TUI and line CLI.
//!
//! Phase N extracts the recorder → agent context-scope handoff. Live agents no
//! longer adopt experiment restore points, but the prepare path still projects
//! a restore package for compatibility and validation.

use std::sync::{Arc, Mutex};

use anyhow::{Result, anyhow};
use async_openai::config::Config;

use crate::agent::Agent;
use crate::protocol_frames::ProtocolFrame;
use crate::runtime_context::{RuntimeActiveContext, RuntimeSnapshot};
use crate::session::restore::project_runtime_restore_snapshot_with_children;
use crate::transcript::transcript_projection::SessionContextCursor;
use crate::transcript::{
    ActiveContextExperiment, ContextScopeState, TranscriptRecorder, read_records,
};

/// Prepared context-scope state ready to apply onto an agent.
pub struct PreparedContextScope {
    state: Arc<Mutex<ContextScopeState>>,
    restore_point: Option<(ActiveContextExperiment, Vec<ProtocolFrame>, RuntimeSnapshot)>,
}

/// Capture the recorder's context-scope state and optional experiment restore
/// projection for the parent branch/base sequence.
pub fn prepare_context_scope(recorder: &TranscriptRecorder) -> Result<PreparedContextScope> {
    let state = recorder.context_scope_state();
    let restore_point = if let Some(scope) = recorder.active_context_experiment() {
        let sessions_dir = recorder
            .path()
            .parent()
            .ok_or_else(|| anyhow!("transcript path has no parent directory"))?;
        let snapshot = project_runtime_restore_snapshot_with_children(
            recorder.session_id(),
            read_records(recorder.path())?,
            SessionContextCursor {
                branch_id: Some(scope.parent_branch_id.clone()),
                leaf_sequence: Some(scope.base_sequence),
            },
            sessions_dir,
        )?;
        // Validate the projected runtime snapshot can become an active context.
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

/// Apply a prepared context-scope package onto the agent.
pub fn apply_prepared_context_scope<C: Config>(
    agent: &mut Agent<C>,
    prepared: PreparedContextScope,
) {
    agent.set_context_scope_state(prepared.state);
    if let Some((scope, frames, snapshot)) = prepared.restore_point {
        agent.set_context_experiment_restore_point(scope, frames, snapshot);
    } else {
        agent.clear_context_experiment_restore_point();
    }
}

/// Prepare from the recorder and apply in one step.
pub fn sync_agent_context_scope_from_recorder<C: Config>(
    agent: &mut Agent<C>,
    recorder: &TranscriptRecorder,
) -> Result<()> {
    let prepared = prepare_context_scope(recorder)?;
    apply_prepared_context_scope(agent, prepared);
    Ok(())
}
