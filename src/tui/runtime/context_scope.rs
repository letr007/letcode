//! Thin re-export of session-owned context-scope prepare/apply helpers.

pub(super) use crate::session::{
    PreparedContextScope, apply_prepared_context_scope, prepare_context_scope,
    sync_agent_context_scope_from_recorder,
};
