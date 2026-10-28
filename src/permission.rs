use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

use crate::tool_names;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum PermissionMode {
    Safe,
    #[default]
    Default,
    Auto,
    #[serde(alias = "solo")]
    Yolo,
}

impl PermissionMode {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Safe => "safe",
            Self::Default => "default",
            Self::Auto => "auto",
            Self::Yolo => "yolo",
        }
    }

    pub fn parse(value: &str) -> Option<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "safe" => Some(Self::Safe),
            "default" => Some(Self::Default),
            "auto" => Some(Self::Auto),
            "yolo" | "solo" => Some(Self::Yolo),
            _ => None,
        }
    }

    pub fn supports_session_grants(self) -> bool {
        matches!(self, Self::Default)
    }
}

impl std::fmt::Display for PermissionMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(dead_code)]
pub enum PermissionDecision {
    Allow,
    Ask,
    Deny,
}

/// The explicit result of an interactive permission request.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PermissionApproval {
    Deny,
    AllowOnce,
    AllowAlways,
}

impl PermissionApproval {
    pub fn allowed(self) -> bool {
        !matches!(self, Self::Deny)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ToolPermissionClass {
    Read,
    Preview,
    Write,
    Command,
    Unknown,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ToolScope {
    #[default]
    FullAccess,
    ReadOnlyExplorer,
}

impl ToolScope {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::FullAccess => "full_access",
            Self::ReadOnlyExplorer => "read_only_explorer",
        }
    }

    pub fn allows_tool(self, tool: &str) -> bool {
        match self {
            Self::FullAccess => true,
            Self::ReadOnlyExplorer => is_read_only_explorer_tool(tool),
        }
    }

    pub fn rejection_message(self, tool: &str) -> String {
        format!("tool '{tool}' is not allowed in {} scope", self.as_str())
    }
}

fn is_read_only_explorer_tool(tool: &str) -> bool {
    crate::tool::tool_class(tool).is_some_and(|class| class.explorer_allowed)
}

impl std::fmt::Display for ToolScope {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

impl ToolPermissionClass {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Read => "read",
            Self::Preview => "preview",
            Self::Write => "write",
            Self::Command => "command",
            Self::Unknown => "unknown",
        }
    }
}

impl std::fmt::Display for ToolPermissionClass {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

#[derive(Debug, Clone)]
pub struct PermissionRequest {
    pub call_id: Option<String>,
    pub tool: String,
    pub args: Value,
    pub class: ToolPermissionClass,
    pub summary: String,
    pub preview: Option<String>,
    pub can_allow_always: bool,
    pub grant_summary: Option<String>,
}

/// A canonical, session-local description of what an Allow Always approval covers.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum PermissionResource {
    Exact {
        tool: String,
        value: String,
    },
    ExactPath {
        tool: String,
        path: PathBuf,
    },
    Directory {
        tool: String,
        path: PathBuf,
    },
    PatchTargets {
        tool: String,
        paths: BTreeSet<PathBuf>,
    },
}

impl PermissionResource {
    pub fn matches(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Exact { tool: a, value: av }, Self::Exact { tool: b, value: bv }) => {
                a == b && av == bv
            }
            (Self::ExactPath { tool: a, path: ap }, Self::ExactPath { tool: b, path: bp }) => {
                a == b && ap == bp
            }
            (Self::Directory { tool: a, path: ap }, Self::Directory { tool: b, path: bp }) => {
                a == b && bp.starts_with(ap)
            }
            (Self::Directory { tool: a, path: ap }, Self::ExactPath { tool: b, path: bp }) => {
                a == b && bp.starts_with(ap)
            }
            (
                Self::PatchTargets { tool: a, paths: ap },
                Self::PatchTargets { tool: b, paths: bp },
            ) => a == b && ap == bp,
            _ => false,
        }
    }

    pub fn summary(&self) -> String {
        match self {
            Self::Exact { tool, value } => format!("{tool}: {value}"),
            Self::ExactPath { tool, path } => format!("{tool}: {}", path_preview(path)),
            Self::Directory { tool, path } => format!("{tool}: {} (subtree)", path_preview(path)),
            Self::PatchTargets { tool, paths } => format!("{tool}: {} target path(s)", paths.len()),
        }
    }
}

/// Produces a readable path label without using a lossy representation as identity.
pub(crate) fn path_preview(path: &Path) -> String {
    if let Some(path) = path.to_str() {
        return path.to_string();
    }

    #[cfg(unix)]
    {
        use std::os::unix::ffi::OsStrExt;
        let bytes = path.as_os_str().as_bytes();
        format!(
            "{} [raw-bytes:{}]",
            path.display(),
            bytes
                .iter()
                .map(|byte| format!("{byte:02x}"))
                .collect::<String>()
        )
    }

    #[cfg(not(unix))]
    path.display().to_string()
}

#[derive(Debug, Clone, Default)]
pub struct PermissionGrantSet {
    grants: Vec<PermissionResource>,
}

impl PermissionGrantSet {
    pub fn allows(&self, resource: &PermissionResource) -> bool {
        self.grants.iter().any(|grant| grant.matches(resource))
    }
    pub fn insert(&mut self, resource: PermissionResource) {
        if !self.grants.contains(&resource) {
            self.grants.push(resource);
        }
    }
    pub fn clear(&mut self) {
        self.grants.clear();
    }
}

/// Which round of the auto-review ladder a call lands on.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AutoReviewRound {
    /// The requester's round: the call goes back to it for an explanation.
    Requester,
    /// The user's round: the requester has explained the call and it is still
    /// unsettled, so only the user can settle it.
    User,
}

#[derive(Debug, Clone, Default)]
pub struct PermissionSessionState {
    policy: PermissionPolicy,
    grants: PermissionGrantSet,
    /// Call signatures an auto-reviewer has already handed back to the requester.
    auto_review_returns: BTreeSet<(String, String)>,
    generation: u64,
}

impl PermissionSessionState {
    pub fn mode(&self) -> PermissionMode {
        self.policy.mode()
    }
    pub fn set_mode(&mut self, mode: PermissionMode) {
        if self.policy.mode() != mode {
            self.policy.set_mode(mode);
            self.grants.clear();
            self.auto_review_returns.clear();
            self.generation = self.generation.wrapping_add(1);
        }
    }
    pub fn decision(
        &self,
        tool: &str,
        args: &Value,
        class: ToolPermissionClass,
    ) -> PermissionDecision {
        self.policy.check_class(tool, args, class)
    }
    pub fn allows_grant(&self, resource: &PermissionResource) -> bool {
        self.grants.allows(resource)
    }
    pub fn grant(&mut self, resource: PermissionResource) {
        self.grants.insert(resource);
    }
    pub fn approval_snapshot(
        &self,
        resource: Option<&PermissionResource>,
        tool: &str,
        args: &Value,
        class: ToolPermissionClass,
        external_workspace_access: bool,
        internal_tool: bool,
    ) -> (PermissionMode, u64, PermissionDecision, bool) {
        let base_decision = self.decision(tool, args, class);
        let decision = if base_decision == PermissionDecision::Deny {
            PermissionDecision::Deny
        } else if internal_tool {
            PermissionDecision::Allow
        } else if matches!(self.mode(), PermissionMode::Default | PermissionMode::Auto)
            && base_decision == PermissionDecision::Allow
            && external_workspace_access
        {
            PermissionDecision::Ask
        } else {
            base_decision
        };
        let grant_allowed = self.mode().supports_session_grants()
            && decision == PermissionDecision::Ask
            && resource.is_some_and(|resource| self.allows_grant(resource));
        (self.mode(), self.generation, decision, grant_allowed)
    }
    /// Inserts an approval grant only when it still belongs to the permission
    /// generation and mode that produced the request.
    pub fn grant_if_current_session(
        &mut self,
        generation: u64,
        resource: PermissionResource,
    ) -> bool {
        if self.mode().supports_session_grants() && self.generation == generation {
            self.grant(resource);
            true
        } else {
            false
        }
    }

    #[cfg(test)]
    pub fn grant_if_current_default(
        &mut self,
        generation: u64,
        resource: PermissionResource,
    ) -> bool {
        self.grant_if_current_session(generation, resource)
    }
    pub fn clear_grants(&mut self) {
        self.grants.clear();
        self.auto_review_returns.clear();
        self.generation = self.generation.wrapping_add(1);
    }

    /// Accounts for one auto-reviewer `ask_user` on this call and reports the
    /// ladder round it lands on.
    ///
    /// A call signature gets one round of explanation per session: the first ask
    /// returns the call to the requester, whose narration the next review reads;
    /// an ask on the second round goes to the user. Different arguments are a
    /// different signature, and the counters reset with the mode.
    pub fn auto_review_round(&mut self, tool: &str, args: &Value) -> AutoReviewRound {
        let signature = (tool.to_string(), args.to_string());
        if self.auto_review_returns.insert(signature) {
            AutoReviewRound::Requester
        } else {
            AutoReviewRound::User
        }
    }

    pub fn fork_without_grants(&self) -> Self {
        Self {
            policy: self.policy.clone(),
            grants: PermissionGrantSet::default(),
            auto_review_returns: BTreeSet::default(),
            generation: 0,
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct PermissionPolicy {
    mode: PermissionMode,
}

impl PermissionPolicy {
    pub fn mode(&self) -> PermissionMode {
        self.mode
    }

    pub fn set_mode(&mut self, mode: PermissionMode) {
        self.mode = mode;
    }

    #[cfg(test)]
    pub fn check(&self, tool: &str, args: &Value) -> PermissionDecision {
        let class = classify_tool(tool);
        self.check_class(tool, args, class)
    }

    pub fn check_class(
        &self,
        tool: &str,
        _args: &Value,
        class: ToolPermissionClass,
    ) -> PermissionDecision {
        match self.mode {
            PermissionMode::Safe => PermissionDecision::Ask,
            PermissionMode::Default | PermissionMode::Auto
                if tool == tool_names::TOOL_WEB_FETCH =>
            {
                PermissionDecision::Ask
            }
            PermissionMode::Default | PermissionMode::Auto => match class {
                ToolPermissionClass::Read | ToolPermissionClass::Preview => {
                    PermissionDecision::Allow
                }
                ToolPermissionClass::Write
                | ToolPermissionClass::Command
                | ToolPermissionClass::Unknown => PermissionDecision::Ask,
            },
            PermissionMode::Yolo => PermissionDecision::Allow,
        }
    }
}

pub fn is_internal_tool(tool: &str) -> bool {
    matches!(
        tool,
        tool_names::TOOL_UTIL_ECHO
            | tool_names::TOOL_QUESTION
            | tool_names::TOOL_SKILL
            | tool_names::TOOL_SKILL_RESOURCE_LIST
            | tool_names::TOOL_SKILL_RESOURCE_READ
            | tool_names::TOOL_MEMORY_RECALL
            | tool_names::TOOL_WORKFLOW_TODOS
            | tool_names::TOOL_WORKFLOW_AUTO_CONTINUE
            | tool_names::TOOL_AGENT_JOBS
            | tool_names::TOOL_AGENT_STATUS
            | tool_names::TOOL_AGENT_WAIT
            | tool_names::TOOL_AGENT_CANCEL
            | tool_names::TOOL_AGENT_SEND
            | tool_names::TOOL_PEER_DISCOVER
            | tool_names::TOOL_PEER_SEND
    )
}

pub fn classify_tool(tool: &str) -> ToolPermissionClass {
    crate::tool::tool_class(tool)
        .map(|class| class.permission)
        .unwrap_or(ToolPermissionClass::Unknown)
}

#[cfg(test)]
#[path = "../tests/internal/permission/tests.rs"]
mod tests;
