use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum PermissionMode {
    Safe,
    #[default]
    Default,
    Solo,
}

impl PermissionMode {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Safe => "safe",
            Self::Default => "default",
            Self::Solo => "solo",
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ToolPermissionClass {
    Read,
    Preview,
    Write,
    Command,
    Unknown,
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

    pub fn check(&self, tool: &str, _args: &Value) -> PermissionDecision {
        let class = classify_tool(tool);

        match self.mode {
            PermissionMode::Safe => PermissionDecision::Ask,
            PermissionMode::Default => match class {
                ToolPermissionClass::Read | ToolPermissionClass::Preview => {
                    PermissionDecision::Allow
                }
                ToolPermissionClass::Write
                | ToolPermissionClass::Command
                | ToolPermissionClass::Unknown => PermissionDecision::Ask,
            },
            PermissionMode::Solo => match class {
                ToolPermissionClass::Unknown => PermissionDecision::Ask,
                ToolPermissionClass::Read
                | ToolPermissionClass::Preview
                | ToolPermissionClass::Write
                | ToolPermissionClass::Command => PermissionDecision::Allow,
            },
        }
    }
}

pub fn classify_tool(tool: &str) -> ToolPermissionClass {
    match tool {
        "util__echo" | "fs__list" | "fs__read" | "search__rg" | "git__status" | "git__diff"
        | "git__log" | "code__ast_search" => ToolPermissionClass::Read,
        "code__ast_replace_preview" => ToolPermissionClass::Preview,
        "fs__write" | "fs__append" | "fs__mkdir" | "edit__apply_patch" => {
            ToolPermissionClass::Write
        }
        "shell__exec" => ToolPermissionClass::Command,
        _ => ToolPermissionClass::Unknown,
    }
}
