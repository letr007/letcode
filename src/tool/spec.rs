//! Single source of truth for each tool's static classification.
//!
//! Permission, explorer visibility, and effect kind are separate dimensions:
//! a read-only tool is not automatically visible to the read-only explorer, and
//! an effect kind is not a permission class. They are declared together here so
//! consumers do not maintain parallel name lists.

use crate::permission::ToolPermissionClass;
use crate::tool_names;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ToolEffectKind {
    Read,
    Write,
    Command,
    Validation,
    WorkflowControl,
    Diagnostic,
    Unknown,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct ToolClass {
    pub permission: ToolPermissionClass,
    pub explorer_allowed: bool,
    pub effect: ToolEffectKind,
}

pub(crate) fn tool_class(name: &str) -> Option<ToolClass> {
    use ToolEffectKind as Effect;
    use ToolPermissionClass as Permission;

    let spec = match name {
        // Read-only results that the read-only explorer may run.
        tool_names::TOOL_SKILL
        | tool_names::TOOL_FS_LIST
        | tool_names::TOOL_FS_READ
        | tool_names::TOOL_SEARCH_RG
        | tool_names::TOOL_WEB_FETCH
        | tool_names::TOOL_GIT
        | tool_names::TOOL_CODE_AST_SEARCH => ToolClass {
            permission: Permission::Read,
            explorer_allowed: true,
            effect: Effect::Read,
        },
        // Read-only, explorer-allowed, but with no file or command effect.
        tool_names::TOOL_UTIL_ECHO
        | tool_names::TOOL_QUESTION
        | tool_names::TOOL_SKILL_RESOURCE_LIST
        | tool_names::TOOL_SKILL_RESOURCE_READ => ToolClass {
            permission: Permission::Read,
            explorer_allowed: true,
            effect: Effect::Unknown,
        },
        // Read-only but not exposed to the read-only explorer.
        tool_names::TOOL_MEMORY_RECALL
        | tool_names::TOOL_CONFIG_VALIDATE
        | tool_names::TOOL_PEER_DISCOVER => ToolClass {
            permission: Permission::Read,
            explorer_allowed: false,
            effect: Effect::Unknown,
        },
        // Session-history retrieval is agent-facing and needs no approval.
        tool_names::TOOL_CONTEXT_SEARCH | tool_names::TOOL_CONTEXT_EXPAND => ToolClass {
            permission: Permission::Read,
            explorer_allowed: false,
            effect: Effect::Read,
        },
        // Preview commands that only read: AST diffs, read-only delegates, and
        // subagent control.
        tool_names::TOOL_CODE_AST_REPLACE_PREVIEW
        | tool_names::TOOL_AGENT_EXPLORE
        | tool_names::TOOL_AGENT_ORACLE
        | tool_names::TOOL_AGENT_DESIGNER
        | tool_names::TOOL_AGENT_LIBRARIAN
        | tool_names::TOOL_AGENT_JOBS
        | tool_names::TOOL_AGENT_STATUS
        | tool_names::TOOL_AGENT_WAIT
        | tool_names::TOOL_AGENT_CANCEL
        | tool_names::TOOL_AGENT_SEND => ToolClass {
            permission: Permission::Preview,
            explorer_allowed: false,
            effect: Effect::Read,
        },
        tool_names::TOOL_WORKFLOW_TODOS | tool_names::TOOL_WORKFLOW_AUTO_CONTINUE => ToolClass {
            permission: Permission::Preview,
            explorer_allowed: false,
            effect: Effect::WorkflowControl,
        },
        tool_names::TOOL_PEER_SEND => ToolClass {
            permission: Permission::Preview,
            explorer_allowed: false,
            effect: Effect::Unknown,
        },
        tool_names::TOOL_AGENT_FIXER
        | tool_names::TOOL_AGENT_GENERAL
        | tool_names::TOOL_FS_WRITE
        | tool_names::TOOL_FS_APPEND
        | tool_names::TOOL_FS_MKDIR
        | tool_names::TOOL_EDIT_APPLY_PATCH => ToolClass {
            permission: Permission::Write,
            explorer_allowed: false,
            effect: Effect::Write,
        },
        tool_names::TOOL_SHELL_EXEC => ToolClass {
            permission: Permission::Command,
            explorer_allowed: false,
            effect: Effect::Command,
        },
        _ => return None,
    };
    Some(spec)
}

#[cfg(test)]
#[path = "../../tests/internal/tool/spec/tests.rs"]
mod tests;
