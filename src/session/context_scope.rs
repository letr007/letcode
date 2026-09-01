//! Context-scope prepare/apply shared by TUI and line CLI.
//!
//! Phase N extracts the recorder → agent context-scope handoff. Live agents no
//! longer adopt experiment restore points, but the prepare path still projects
//! a restore package for compatibility and validation.

use std::sync::{Arc, Mutex};

use crate::agent::Agent;
use crate::transcript::{ContextScopeState, TranscriptRecorder};
use anyhow::Result;

/// Prepared context-scope state ready to apply onto an agent.
pub struct PreparedContextScope {
    state: Arc<Mutex<ContextScopeState>>,
}

/// Capture recorder-owned decode compatibility state. Context-experiment events
/// are retained for journal decoding only and never restore a live agent state.
pub fn prepare_context_scope(recorder: &TranscriptRecorder) -> Result<PreparedContextScope> {
    Ok(PreparedContextScope {
        state: recorder.context_scope_state(),
    })
}

/// Apply a prepared context-scope package onto the agent.
pub fn apply_prepared_context_scope(agent: &mut Agent, prepared: PreparedContextScope) {
    agent.set_context_scope_state(prepared.state);
}

/// Prepare from the recorder and apply in one step.
pub fn sync_agent_context_scope_from_recorder(
    agent: &mut Agent,
    recorder: &TranscriptRecorder,
) -> Result<()> {
    let prepared = prepare_context_scope(recorder)?;
    apply_prepared_context_scope(agent, prepared);
    Ok(())
}
