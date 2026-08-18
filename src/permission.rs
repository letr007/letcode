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

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ExecutionDirective {
    #[default]
    None,
    ReadOnly,
    PlanOnly,
    AnalyzeOnly,
    DoNotEdit,
}

impl ExecutionDirective {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::None => "none",
            Self::ReadOnly => "read_only",
            Self::PlanOnly => "plan_only",
            Self::AnalyzeOnly => "analyze_only",
            Self::DoNotEdit => "do_not_edit",
        }
    }

    pub fn restricts_writes(self) -> bool {
        !matches!(self, Self::None)
    }

    pub fn restricts_commands_to_read_only(self) -> bool {
        !matches!(self, Self::None)
    }
}

impl std::fmt::Display for ExecutionDirective {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
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
    matches!(
        tool,
        tool_names::TOOL_UTIL_ECHO
            | tool_names::TOOL_QUESTION
            | tool_names::TOOL_SKILL
            | tool_names::TOOL_SKILL_RESOURCE_LIST
            | tool_names::TOOL_SKILL_RESOURCE_READ
            | tool_names::TOOL_FS_LIST
            | tool_names::TOOL_FS_READ
            | tool_names::TOOL_SEARCH_RG
            | tool_names::TOOL_WEB_FETCH
            | tool_names::TOOL_GIT_STATUS
            | tool_names::TOOL_GIT_DIFF
            | tool_names::TOOL_GIT_LOG
            | tool_names::TOOL_CODE_AST_SEARCH
    )
}

impl std::fmt::Display for ToolScope {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Shell command allowlist buckets for Default. Everything else Asks.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CommandRisk {
    ReadOnly,
    LowRisk,
    Ask,
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
    pub directive: ExecutionDirective,
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

#[derive(Debug, Clone, Default)]
pub struct PermissionSessionState {
    policy: PermissionPolicy,
    grants: PermissionGrantSet,
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
            self.generation = self.generation.wrapping_add(1);
        }
    }
    pub fn decision(
        &self,
        tool: &str,
        args: &Value,
        class: ToolPermissionClass,
        directive: ExecutionDirective,
    ) -> PermissionDecision {
        self.policy
            .check_class_with_directive(tool, args, class, directive)
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
        directive: ExecutionDirective,
        external_workspace_access: bool,
        internal_tool: bool,
    ) -> (PermissionMode, u64, PermissionDecision, bool) {
        let base_decision = self.decision(tool, args, class, directive);
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
        self.generation = self.generation.wrapping_add(1);
    }

    pub fn fork_without_grants(&self) -> Self {
        Self {
            policy: self.policy.clone(),
            grants: PermissionGrantSet::default(),
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
        self.check_with_directive(tool, args, ExecutionDirective::None)
    }

    #[cfg(test)]
    pub fn check_with_directive(
        &self,
        tool: &str,
        args: &Value,
        directive: ExecutionDirective,
    ) -> PermissionDecision {
        let class = classify_tool(tool);

        self.check_class_with_directive(tool, args, class, directive)
    }

    pub fn check_class_with_directive(
        &self,
        tool: &str,
        args: &Value,
        class: ToolPermissionClass,
        directive: ExecutionDirective,
    ) -> PermissionDecision {
        if restricted_by_directive_with_class(tool, args, class, directive).is_some() {
            return if self.mode == PermissionMode::Auto {
                PermissionDecision::Ask
            } else {
                PermissionDecision::Deny
            };
        }

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
                ToolPermissionClass::Write | ToolPermissionClass::Unknown => {
                    PermissionDecision::Ask
                }
                ToolPermissionClass::Command => match classify_command_risk(args) {
                    CommandRisk::ReadOnly | CommandRisk::LowRisk => PermissionDecision::Allow,
                    CommandRisk::Ask => PermissionDecision::Ask,
                },
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
            | tool_names::TOOL_AGENT_RECONCILE
    )
}

#[cfg(test)]
pub fn restricted_by_directive(
    tool: &str,
    args: &Value,
    directive: ExecutionDirective,
) -> Option<String> {
    restricted_by_directive_with_class(tool, args, classify_tool(tool), directive)
}

pub fn restricted_by_directive_with_class(
    tool: &str,
    args: &Value,
    class: ToolPermissionClass,
    directive: ExecutionDirective,
) -> Option<String> {
    if matches!(directive, ExecutionDirective::None) {
        return None;
    }

    match class {
        ToolPermissionClass::Write if directive.restricts_writes() => Some(format!(
            "blocked by {directive} directive: tool '{tool}' modifies the workspace"
        )),
        ToolPermissionClass::Command if directive.restricts_commands_to_read_only() => {
            match classify_command_risk(args) {
                CommandRisk::ReadOnly => None,
                _ => Some(format!(
                    "blocked by {directive} directive: command tool '{tool}' is not read-only compatible"
                )),
            }
        }
        ToolPermissionClass::Unknown => Some(format!(
            "blocked by {directive} directive: tool '{tool}' is not classified as read-only"
        )),
        _ => None,
    }
}

pub fn classify_tool(tool: &str) -> ToolPermissionClass {
    match tool {
        tool_names::TOOL_UTIL_ECHO
        | tool_names::TOOL_QUESTION
        | tool_names::TOOL_SKILL
        | tool_names::TOOL_SKILL_RESOURCE_LIST
        | tool_names::TOOL_SKILL_RESOURCE_READ
        | tool_names::TOOL_FS_LIST
        | tool_names::TOOL_FS_READ
        | tool_names::TOOL_MEMORY_RECALL
        | tool_names::TOOL_CONFIG_VALIDATE
        | tool_names::TOOL_SEARCH_RG
        | tool_names::TOOL_WEB_FETCH
        | tool_names::TOOL_GIT_STATUS
        | tool_names::TOOL_GIT_DIFF
        | tool_names::TOOL_GIT_LOG
        | tool_names::TOOL_CODE_AST_SEARCH => ToolPermissionClass::Read,
        tool_names::TOOL_CODE_AST_REPLACE_PREVIEW
        | tool_names::TOOL_WORKFLOW_TODOS
        | tool_names::TOOL_WORKFLOW_AUTO_CONTINUE
        | tool_names::TOOL_AGENT_EXPLORE
        | tool_names::TOOL_AGENT_ORACLE
        | tool_names::TOOL_AGENT_DESIGNER
        | tool_names::TOOL_AGENT_LIBRARIAN
        | tool_names::TOOL_AGENT_GENERAL
        | tool_names::TOOL_AGENT_RECONCILE => ToolPermissionClass::Preview,
        tool_names::TOOL_AGENT_FIXER
        | tool_names::TOOL_FS_WRITE
        | tool_names::TOOL_FS_APPEND
        | tool_names::TOOL_FS_MKDIR
        | tool_names::TOOL_EDIT_APPLY_PATCH => ToolPermissionClass::Write,
        tool_names::TOOL_SHELL_EXEC => ToolPermissionClass::Command,
        _ => ToolPermissionClass::Unknown,
    }
}

pub fn classify_command_risk(args: &Value) -> CommandRisk {
    let command = args
        .get("command")
        .and_then(Value::as_str)
        .map(str::trim)
        .unwrap_or_default();
    classify_command_text(command)
}

fn classify_command_text(command: &str) -> CommandRisk {
    if command.is_empty() {
        return CommandRisk::Ask;
    }

    let normalized = command.to_ascii_lowercase();
    let trimmed = normalized.trim();

    // Compound / redirect / write-flag forms must not inherit a ReadOnly prefix match
    // (e.g. `git status && …`, `git diff --output=`).
    if contains_shell_control_syntax(trimmed) || contains_write_capable_read_option(trimmed) {
        return CommandRisk::Ask;
    }
    if is_read_only_command(trimmed) {
        return CommandRisk::ReadOnly;
    }
    if is_low_risk_validation_command(trimmed) {
        return CommandRisk::LowRisk;
    }
    CommandRisk::Ask
}

fn is_read_only_command(command: &str) -> bool {
    command_has_prefix(command, "git status")
        || command_has_prefix(command, "git diff")
        || command_has_prefix(command, "git log")
        || command_has_prefix(command, "rg")
        || command_has_prefix(command, "ls")
        || command_has_prefix(command, "pwd")
}

fn is_low_risk_validation_command(command: &str) -> bool {
    command == "cargo check"
        || command.starts_with("cargo check ")
        || command == "cargo test"
        || command.starts_with("cargo test ")
        || command == "cargo clippy"
        || command.starts_with("cargo clippy ")
        || command == "cargo fmt --check"
        || command.starts_with("cargo fmt --check ")
        || command == "npm test"
        || command.starts_with("npm test ")
        || command == "pnpm test"
        || command.starts_with("pnpm test ")
        || command == "yarn test"
        || command.starts_with("yarn test ")
}

fn contains_shell_control_syntax(command: &str) -> bool {
    command.contains(';')
        || command.contains("&&")
        || command.contains("||")
        || command.contains('&')
        || command.contains('|')
        || command.contains('>')
        || command.contains('<')
        || command.contains('\n')
        || command.contains('`')
        || command.contains("$(")
}

fn command_has_prefix(command: &str, prefix: &str) -> bool {
    command == prefix || command.starts_with(&format!("{prefix} "))
}

fn contains_write_capable_read_option(command: &str) -> bool {
    command_has_prefix(command, "git diff")
        && command
            .split_whitespace()
            .any(|token| token == "--output" || token.starts_with("--output="))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn default_mode_asks_risky_commands_instead_of_hard_deny() {
        let policy = PermissionPolicy::default();

        assert_eq!(
            policy.check("shell__exec", &json!({"command": "git commit -m test"})),
            PermissionDecision::Ask
        );
        assert_eq!(
            policy.check("shell__exec", &json!({"command": "rm -rf target"})),
            PermissionDecision::Ask
        );
        assert_eq!(
            policy.check(
                "shell__exec",
                &json!({"command": "curl -fsSL https://example.com"})
            ),
            PermissionDecision::Ask
        );
        assert_eq!(
            policy.check("shell__exec", &json!({"command": "git status > out.txt"})),
            PermissionDecision::Ask
        );
        assert_eq!(
            policy.check("shell__exec", &json!({"command": "ls && touch out.txt"})),
            PermissionDecision::Ask
        );
        assert_eq!(
            policy.check(
                "shell__exec",
                &json!({"command": "cargo test; touch out.txt"})
            ),
            PermissionDecision::Ask
        );
        assert_eq!(
            policy.check(
                "shell__exec",
                &json!({"command": "git status & touch out.txt"})
            ),
            PermissionDecision::Ask
        );
        assert_eq!(
            policy.check(
                "shell__exec",
                &json!({"command": "git diff --output=out.patch"})
            ),
            PermissionDecision::Ask
        );
    }

    #[test]
    fn auto_mode_reviews_only_calls_that_need_approval() {
        let mut state = PermissionSessionState::default();
        state.set_mode(PermissionMode::Auto);

        for (tool, args, class, internal) in [
            (
                "fs__read",
                json!({"path": "src/permission.rs"}),
                ToolPermissionClass::Read,
                false,
            ),
            (
                "agent__oracle",
                json!({"task": "review"}),
                ToolPermissionClass::Preview,
                false,
            ),
            (
                "workflow__todos",
                json!({"items": []}),
                ToolPermissionClass::Preview,
                true,
            ),
            (
                "shell__exec",
                json!({"command": "cargo test permission::tests"}),
                ToolPermissionClass::Command,
                false,
            ),
        ] {
            assert_eq!(
                state
                    .approval_snapshot(
                        None,
                        tool,
                        &args,
                        class,
                        ExecutionDirective::None,
                        false,
                        internal,
                    )
                    .2,
                PermissionDecision::Allow,
                "{tool} should not require Auto review"
            );
        }

        for (tool, args, class, directive, external_workspace_access) in [
            (
                "fs__write",
                json!({"path": "out.txt", "content": "ok"}),
                ToolPermissionClass::Write,
                ExecutionDirective::PlanOnly,
                false,
            ),
            (
                "shell__exec",
                json!({"command": "rm -rf target/tmp"}),
                ToolPermissionClass::Command,
                ExecutionDirective::None,
                false,
            ),
            (
                "fs__read",
                json!({"path": "/tmp/outside.txt"}),
                ToolPermissionClass::Read,
                ExecutionDirective::None,
                true,
            ),
        ] {
            assert_eq!(
                state
                    .approval_snapshot(
                        None,
                        tool,
                        &args,
                        class,
                        directive,
                        external_workspace_access,
                        false,
                    )
                    .2,
                PermissionDecision::Ask,
                "{tool} should be reviewed in Auto mode"
            );
        }
    }

    #[test]
    fn trusted_reads_do_not_escalate_to_ask_but_other_external_reads_do() {
        // Default mode: reads inside the workspace, or at trusted fold-artifact
        // paths (which the agent folds into external=false before this call),
        // stay on the read-only default Allow and do not prompt.
        let state = PermissionSessionState::default();
        assert_eq!(
            state
                .approval_snapshot(
                    None,
                    "fs__read",
                    &json!({"path": "/tmp/letcode-command/x.out"}),
                    ToolPermissionClass::Read,
                    ExecutionDirective::None,
                    false,
                    false,
                )
                .2,
            PermissionDecision::Allow,
        );
        // Any genuinely external (untrusted) read still escalates to Ask.
        assert_eq!(
            state
                .approval_snapshot(
                    None,
                    "fs__read",
                    &json!({"path": "/tmp/other.txt"}),
                    ToolPermissionClass::Read,
                    ExecutionDirective::None,
                    true,
                    false,
                )
                .2,
            PermissionDecision::Ask,
        );
    }

    #[test]
    fn auto_mode_ignores_session_grants() {
        let resource = PermissionResource::Exact {
            tool: "shell__exec".into(),
            value: "rm -rf /".into(),
        };
        let args = json!({"command": "rm -rf /"});
        let mut state = PermissionSessionState::default();
        state.set_mode(PermissionMode::Auto);
        state.grant(resource.clone());

        let (_, generation, decision, grant_allowed) = state.approval_snapshot(
            Some(&resource),
            "shell__exec",
            &args,
            ToolPermissionClass::Command,
            ExecutionDirective::None,
            false,
            false,
        );
        assert_eq!(decision, PermissionDecision::Ask);
        assert!(
            !grant_allowed,
            "Auto mode must not reuse session grants for reviewed calls"
        );
        assert!(
            !state.grant_if_current_session(generation, resource),
            "Auto mode must not create reusable session grants"
        );
    }

    #[test]
    fn restricted_directive_overrides_otherwise_allowed_command() {
        let policy = PermissionPolicy::default();

        assert_eq!(
            policy.check_with_directive(
                "shell__exec",
                &json!({"command": "cargo test permission::tests"}),
                ExecutionDirective::ReadOnly,
            ),
            PermissionDecision::Deny
        );
        assert_eq!(
            policy.check_with_directive(
                "shell__exec",
                &json!({"command": "git diff -- src/permission.rs"}),
                ExecutionDirective::ReadOnly,
            ),
            PermissionDecision::Allow
        );
        assert_eq!(
            policy.check_with_directive(
                "shell__exec",
                &json!({"command": "git diff -- src/permission.rs > out.txt"}),
                ExecutionDirective::ReadOnly,
            ),
            PermissionDecision::Deny
        );
        assert_eq!(
            policy.check_with_directive(
                "shell__exec",
                &json!({"command": "git status & touch out.txt"}),
                ExecutionDirective::ReadOnly,
            ),
            PermissionDecision::Deny
        );
        assert_eq!(
            policy.check_with_directive(
                "shell__exec",
                &json!({"command": "git diff --output=out.patch"}),
                ExecutionDirective::ReadOnly,
            ),
            PermissionDecision::Deny
        );
        assert_eq!(
            policy.check_with_directive(
                "workflow__todos",
                &json!({"items": []}),
                ExecutionDirective::ReadOnly,
            ),
            PermissionDecision::Allow
        );
    }

    #[test]
    fn restricted_directive_blocks_edit_tools() {
        let message = restricted_by_directive(
            "fs__write",
            &json!({"path": "a.txt", "content": "x"}),
            ExecutionDirective::PlanOnly,
        )
        .expect("write tool should be blocked");

        assert!(message.contains("plan_only"));
        assert!(message.contains("fs__write"));

        let unknown = restricted_by_directive(
            "future__maybe_write",
            &json!({"path": "a.txt"}),
            ExecutionDirective::ReadOnly,
        )
        .expect("unknown tools should fail closed under restrictive directives");

        assert!(unknown.contains("not classified as read-only"));
    }

    #[test]
    fn yolo_mode_allows_commands_that_other_modes_ask() {
        let mut policy = PermissionPolicy::default();
        policy.set_mode(PermissionMode::Yolo);

        assert_eq!(
            policy.check("shell__exec", &json!({"command": "rm -rf ."})),
            PermissionDecision::Allow
        );
        assert_eq!(
            policy.check("shell__exec", &json!({"command": "curl --version"})),
            PermissionDecision::Allow
        );
    }

    #[test]
    fn fork_without_grants_keeps_mode_and_drops_grants() {
        let resource = PermissionResource::Exact {
            tool: "shell__exec".into(),
            value: "curl --version".into(),
        };
        let mut parent = PermissionSessionState::default();
        parent.set_mode(PermissionMode::Auto);
        parent.grant(resource.clone());

        let child = parent.fork_without_grants();
        assert_eq!(child.mode(), PermissionMode::Auto);
        assert!(!child.allows_grant(&resource));
        assert!(parent.allows_grant(&resource));
    }

    #[test]
    fn grants_bypass_ask_for_high_risk_commands_and_generation_guards_allow_always() {
        let resource = PermissionResource::Exact {
            tool: "shell__exec".into(),
            value: "rm -rf /".into(),
        };
        let args = json!({"command": "rm -rf /"});
        let mut state = PermissionSessionState::default();
        state.grant(resource.clone());

        let (_, generation, decision, grant_allowed) = state.approval_snapshot(
            Some(&resource),
            "shell__exec",
            &args,
            ToolPermissionClass::Command,
            ExecutionDirective::None,
            false,
            false,
        );
        assert_eq!(decision, PermissionDecision::Ask);
        assert!(
            grant_allowed,
            "session grant should satisfy Ask for formerly blacklisted commands"
        );

        // Directive hard-deny still cannot be overridden by a grant.
        let (_, _, directed, directed_grant) = state.approval_snapshot(
            Some(&resource),
            "shell__exec",
            &args,
            ToolPermissionClass::Command,
            ExecutionDirective::ReadOnly,
            false,
            false,
        );
        assert_eq!(directed, PermissionDecision::Deny);
        assert!(!directed_grant, "a grant must never override policy denial");

        state.clear_grants();
        assert!(
            !state.grant_if_current_default(generation, resource),
            "an approval from a previous generation must not create a grant"
        );
    }

    #[test]
    fn mode_changes_and_clear_api_advance_permission_generation() {
        let mut state = PermissionSessionState::default();
        let (_, initial, _, _) = state.approval_snapshot(
            None,
            "fs__write",
            &json!({}),
            ToolPermissionClass::Write,
            ExecutionDirective::None,
            false,
            false,
        );
        state.set_mode(PermissionMode::Safe);
        let (_, after_mode, _, _) = state.approval_snapshot(
            None,
            "fs__write",
            &json!({}),
            ToolPermissionClass::Write,
            ExecutionDirective::None,
            false,
            false,
        );
        assert_ne!(initial, after_mode);
        state.clear_grants();
        let (_, after_clear, _, _) = state.approval_snapshot(
            None,
            "fs__write",
            &json!({}),
            ToolPermissionClass::Write,
            ExecutionDirective::None,
            false,
            false,
        );
        assert_ne!(after_mode, after_clear);
    }

    #[test]
    fn resource_matching_keeps_string_and_path_identities_separate() {
        let directory = PermissionResource::Directory {
            tool: "fs__read".into(),
            path: PathBuf::from("/workspace/src"),
        };
        let exact_path = PermissionResource::ExactPath {
            tool: "fs__read".into(),
            path: PathBuf::from("/workspace/src/file.rs"),
        };
        let exact = PermissionResource::Exact {
            tool: "fs__read".into(),
            value: "/workspace/src/file.rs".into(),
        };

        assert!(directory.matches(&exact_path));
        assert!(!directory.matches(&exact));
        assert!(!exact.matches(&exact_path));
        assert!(!exact_path.matches(&exact));
    }

    #[cfg(unix)]
    #[test]
    fn raw_byte_paths_are_not_lossily_collapsed_in_grants_or_previews() {
        use std::ffi::OsString;
        use std::os::unix::ffi::OsStringExt;

        let first = PathBuf::from(OsString::from_vec(b"/tmp/letcode-raw-\x80".to_vec()));
        let second = PathBuf::from(OsString::from_vec(b"/tmp/letcode-raw-\x81".to_vec()));
        assert_eq!(first.to_string_lossy(), second.to_string_lossy());

        let first_resource = PermissionResource::ExactPath {
            tool: "fs__write".into(),
            path: first,
        };
        let second_resource = PermissionResource::ExactPath {
            tool: "fs__write".into(),
            path: second,
        };
        let mut grants = PermissionGrantSet::default();
        grants.insert(first_resource.clone());

        assert!(grants.allows(&first_resource));
        assert!(!grants.allows(&second_resource));
        assert!(first_resource.summary().contains("raw-bytes:"));
        assert_ne!(first_resource.summary(), second_resource.summary());
    }
}
