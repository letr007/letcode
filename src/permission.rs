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
    #[serde(alias = "solo")]
    Yolo,
}

impl PermissionMode {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Safe => "safe",
            Self::Default => "default",
            Self::Yolo => "yolo",
        }
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CommandRisk {
    ReadOnly,
    LowRisk,
    Ask,
    Deny,
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
        return format!(
            "{} [raw-bytes:{}]",
            path.display(),
            bytes
                .iter()
                .map(|byte| format!("{byte:02x}"))
                .collect::<String>()
        );
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
        } else if self.mode() == PermissionMode::Default
            && base_decision == PermissionDecision::Allow
            && external_workspace_access
        {
            PermissionDecision::Ask
        } else {
            base_decision
        };
        let grant_allowed = decision == PermissionDecision::Ask
            && resource.is_some_and(|resource| self.allows_grant(resource));
        (self.mode(), self.generation, decision, grant_allowed)
    }
    /// Inserts an approval grant only when it still belongs to the permission
    /// generation and mode that produced the request.
    pub fn grant_if_current_default(
        &mut self,
        generation: u64,
        resource: PermissionResource,
    ) -> bool {
        if self.mode() == PermissionMode::Default && self.generation == generation {
            self.grant(resource);
            true
        } else {
            false
        }
    }
    pub fn clear_grants(&mut self) {
        self.grants.clear();
        self.generation = self.generation.wrapping_add(1);
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
            return PermissionDecision::Deny;
        }

        match self.mode {
            PermissionMode::Safe => PermissionDecision::Ask,
            PermissionMode::Default if tool == tool_names::TOOL_WEB_FETCH => {
                PermissionDecision::Ask
            }
            PermissionMode::Default => match class {
                ToolPermissionClass::Read | ToolPermissionClass::Preview => {
                    PermissionDecision::Allow
                }
                ToolPermissionClass::Write | ToolPermissionClass::Unknown => {
                    PermissionDecision::Ask
                }
                ToolPermissionClass::Command => match classify_command_risk(args) {
                    CommandRisk::ReadOnly | CommandRisk::LowRisk => PermissionDecision::Allow,
                    CommandRisk::Ask => PermissionDecision::Ask,
                    CommandRisk::Deny => PermissionDecision::Deny,
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

    if contains_any_substring(
        trimmed,
        &[
            "rm -rf",
            "rm -fr",
            "mkfs",
            "dd if=",
            "sudo ",
            "shutdown",
            "reboot",
            "halt",
            "poweroff",
            "diskutil erase",
            "git reset --hard",
            "git clean -fd",
            "git clean -xdf",
            "curl ",
            "wget ",
        ],
    ) || contains_pipe_to_shell(trimmed)
    {
        return CommandRisk::Deny;
    }

    if contains_shell_control_syntax(trimmed) {
        return CommandRisk::Ask;
    }

    if contains_write_capable_read_option(trimmed) {
        return CommandRisk::Ask;
    }

    if is_read_only_command(trimmed) {
        return CommandRisk::ReadOnly;
    }

    if is_low_risk_validation_command(trimmed) {
        return CommandRisk::LowRisk;
    }

    if starts_with_any(
        trimmed,
        &[
            "git add",
            "git commit",
            "git push",
            "git pull",
            "git fetch",
            "git merge",
            "git rebase",
            "git checkout",
            "git switch",
            "git restore",
            "cargo fmt",
            "npm ",
            "pnpm ",
            "yarn ",
            "mkdir ",
            "touch ",
            "cp ",
            "mv ",
            "python ",
            "node ",
            "ssh ",
            "scp ",
            "rsync ",
            "gh ",
        ],
    ) {
        return CommandRisk::Ask;
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

fn starts_with_any(text: &str, prefixes: &[&str]) -> bool {
    prefixes.iter().any(|prefix| text.starts_with(prefix))
}

fn contains_any_substring(text: &str, needles: &[&str]) -> bool {
    needles.iter().any(|needle| text.contains(needle))
}

fn contains_pipe_to_shell(command: &str) -> bool {
    command.contains("| sh") || command.contains("| bash") || command.contains("| zsh")
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
    fn default_mode_allows_read_only_and_low_risk_commands() {
        let policy = PermissionPolicy::default();

        assert_eq!(
            policy.check("shell__exec", &json!({"command": "git status --short"})),
            PermissionDecision::Allow
        );
        assert_eq!(
            policy.check(
                "shell__exec",
                &json!({"command": "cargo test permission::tests"})
            ),
            PermissionDecision::Allow
        );
    }

    #[test]
    fn default_mode_asks_before_network_fetches() {
        let policy = PermissionPolicy::default();

        assert_eq!(
            classify_tool(tool_names::TOOL_WEB_FETCH),
            ToolPermissionClass::Read
        );
        assert_eq!(
            policy.check(
                tool_names::TOOL_WEB_FETCH,
                &json!({"url": "https://example.com/"})
            ),
            PermissionDecision::Ask
        );
    }

    #[test]
    fn default_mode_allows_registered_mcp_read_tools() {
        let policy = PermissionPolicy::default();

        assert_eq!(
            policy.check_class_with_directive(
                "websearch__web_search_exa",
                &json!({"query": "rust async"}),
                ToolPermissionClass::Read,
                ExecutionDirective::None,
            ),
            PermissionDecision::Allow
        );
    }

    #[test]
    fn skill_tool_is_classified_as_read_and_allowed_for_explorer_scope() {
        assert_eq!(classify_tool("skill"), ToolPermissionClass::Read);
        assert!(ToolScope::ReadOnlyExplorer.allows_tool("skill"));
    }

    #[test]
    fn read_tools_are_classified_and_allowed_for_explorer_scope() {
        for tool in ["skill__resource_list", "skill__resource_read", "web__fetch"] {
            assert_eq!(classify_tool(tool), ToolPermissionClass::Read, "{tool}");
            assert!(ToolScope::ReadOnlyExplorer.allows_tool(tool), "{tool}");
        }
    }

    #[test]
    fn subagent_tools_keep_expected_permission_classes() {
        for tool in [
            "agent__explore",
            "agent__oracle",
            "agent__designer",
            "agent__librarian",
            "agent__general",
            "agent__reconcile",
        ] {
            assert_eq!(classify_tool(tool), ToolPermissionClass::Preview, "{tool}");
            assert!(!ToolScope::ReadOnlyExplorer.allows_tool(tool), "{tool}");
        }
        assert_eq!(classify_tool("agent__fixer"), ToolPermissionClass::Write);
        assert!(!ToolScope::ReadOnlyExplorer.allows_tool("agent__fixer"));

        let policy = PermissionPolicy::default();
        assert_eq!(
            policy.check("agent__explore", &json!({"task": "inspect"})),
            PermissionDecision::Allow
        );
        assert_eq!(
            policy.check("agent__oracle", &json!({"task": "review"})),
            PermissionDecision::Allow
        );
        assert_eq!(
            policy.check(
                "agent__reconcile",
                &json!({
                    "run_id": "run-1",
                    "child_session_id": "child-1",
                    "agent_name": "explorer",
                    "decision": "accepted",
                    "summary": "accepted child result"
                })
            ),
            PermissionDecision::Allow
        );
        assert_eq!(
            policy.check("agent__fixer", &json!({"task": "implement"})),
            PermissionDecision::Ask
        );
    }

    #[test]
    fn default_mode_asks_or_denies_risky_commands() {
        let policy = PermissionPolicy::default();

        assert_eq!(
            policy.check("shell__exec", &json!({"command": "git commit -m test"})),
            PermissionDecision::Ask
        );
        assert_eq!(
            policy.check("shell__exec", &json!({"command": "rm -rf target"})),
            PermissionDecision::Deny
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
    fn permission_mode_serializes_yolo_and_deserializes_legacy_solo() {
        assert_eq!(
            serde_json::to_string(&PermissionMode::Yolo).unwrap(),
            "\"yolo\""
        );
        assert_eq!(
            serde_json::from_str::<PermissionMode>("\"solo\"").unwrap(),
            PermissionMode::Yolo
        );
    }

    #[test]
    fn yolo_mode_allows_commands_that_other_modes_deny() {
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
    fn grants_only_bypass_ask_and_generation_guards_allow_always() {
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
        assert_eq!(decision, PermissionDecision::Deny);
        assert!(!grant_allowed, "a grant must never override policy denial");

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
