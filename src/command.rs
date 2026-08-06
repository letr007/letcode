use crate::delegation::{
    delegation_help_summary, delegation_usage_list, find_expert, unknown_expert_error,
};
use crate::permission::PermissionMode;
use crate::request_builder::ModelReasoningEffort;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CommandMetadata {
    pub name: &'static str,
    pub insert_text: &'static str,
    pub description: &'static str,
    pub usage: &'static str,
    pub visible_in_slash: bool,
    pub visible_in_help: bool,
    pub visible_in_summary: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChildNavigation {
    First,
    Next,
    Prev,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ToolOutputMode {
    Toggle,
    Expanded,
    Truncated,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TranscriptScrollbarMode {
    Toggle,
    Visible,
    Hidden,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ThemeName {
    Dark,
    Rainbow,
}

impl ThemeName {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Dark => "dark",
            Self::Rainbow => "rainbow",
        }
    }

    pub fn parse(value: &str) -> Option<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "dark" | "default" => Some(Self::Dark),
            "rainbow" => Some(Self::Rainbow),
            _ => None,
        }
    }
}

impl Default for ThemeName {
    fn default() -> Self {
        Self::Dark
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ThemeCommand {
    Show,
    Set(String),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CommandIntent {
    Prompt(String),
    Delegate { agent_name: String, task: String },
    Help,
    Exit,
    PermissionShow,
    PermissionSet(PermissionMode),
    ModelShow,
    ModelSet(String),
    AgentsShow,
    FastToggle,
    ReasoningShow,
    ReasoningSet(ModelReasoningEffort),
    ToolOutputSet(ToolOutputMode),
    TranscriptScrollbarSet(TranscriptScrollbarMode),
    Theme(ThemeCommand),
    Compact,
    Tree,
    Undo,
    Redo,
    ResumeShow,
    Resume(String),
    NewSession,
    ContextBrowse,
    McpBrowse,
    SkillBrowse,
    Child(ChildNavigation),
    Parent,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CommandParseError {
    message: String,
}

impl CommandParseError {
    fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }

    pub fn message(&self) -> &str {
        &self.message
    }
}

const COMMANDS: &[CommandMetadata] = &[
    CommandMetadata {
        name: "/help",
        insert_text: "/help",
        description: "Show available local commands",
        usage: "/help",
        visible_in_slash: true,
        visible_in_help: true,
        visible_in_summary: true,
    },
    CommandMetadata {
        name: "/?",
        insert_text: "/?",
        description: "Show available local commands",
        usage: "/?",
        visible_in_slash: false,
        visible_in_help: true,
        visible_in_summary: false,
    },
    CommandMetadata {
        name: "/exit",
        insert_text: "/exit",
        description: "Exit the current session",
        usage: "/exit",
        visible_in_slash: true,
        visible_in_help: true,
        visible_in_summary: true,
    },
    CommandMetadata {
        name: "/quit",
        insert_text: "/quit",
        description: "Exit the current session",
        usage: "/quit",
        visible_in_slash: true,
        visible_in_help: true,
        visible_in_summary: true,
    },
    CommandMetadata {
        name: "/permission",
        insert_text: "/permission ",
        description: "Show or switch permission mode",
        usage: "/permission <safe|default|auto|yolo>",
        visible_in_slash: true,
        visible_in_help: true,
        visible_in_summary: true,
    },
    CommandMetadata {
        name: "/perm",
        insert_text: "/perm ",
        description: "Alias for /permission",
        usage: "/perm <safe|default|auto|yolo>",
        visible_in_slash: false,
        visible_in_help: true,
        visible_in_summary: false,
    },
    CommandMetadata {
        name: "/model",
        insert_text: "/model ",
        description: "Show or switch the active model",
        usage: "/model <id>",
        visible_in_slash: true,
        visible_in_help: true,
        visible_in_summary: true,
    },
    CommandMetadata {
        name: "/agents",
        insert_text: "/agents",
        description: "Configure expert models",
        usage: "/agents",
        visible_in_slash: true,
        visible_in_help: true,
        visible_in_summary: true,
    },
    CommandMetadata {
        name: "/fast",
        insert_text: "/fast",
        description: "Toggle Fast Mode",
        usage: "/fast",
        visible_in_slash: true,
        visible_in_help: true,
        visible_in_summary: true,
    },
    CommandMetadata {
        name: "/reasoning",
        insert_text: "/reasoning ",
        description: "Show or switch reasoning effort",
        usage: "/reasoning <off|none|minimal|low|medium|high|xhigh>",
        visible_in_slash: true,
        visible_in_help: true,
        visible_in_summary: true,
    },
    CommandMetadata {
        name: "/think",
        insert_text: "/think ",
        description: "Alias for /reasoning",
        usage: "/think <off|none|minimal|low|medium|high|xhigh>",
        visible_in_slash: false,
        visible_in_help: true,
        visible_in_summary: false,
    },
    CommandMetadata {
        name: "/tool-output",
        insert_text: "/tool-output ",
        description: "Toggle tool output display mode",
        usage: "/tool-output <on|off|expanded|truncated|full|compact>",
        visible_in_slash: true,
        visible_in_help: true,
        visible_in_summary: true,
    },
    CommandMetadata {
        name: "/scrollbar",
        insert_text: "/scrollbar ",
        description: "Show or hide transcript scrollbar",
        usage: "/scrollbar [on|off]",
        visible_in_slash: true,
        visible_in_help: true,
        visible_in_summary: true,
    },
    CommandMetadata {
        name: "/theme",
        insert_text: "/theme ",
        description: "Show or switch the TUI theme",
        usage: "/theme [dark|rainbow|<themes/*.toml>]",
        visible_in_slash: true,
        visible_in_help: true,
        visible_in_summary: true,
    },
    CommandMetadata {
        name: "/compact",
        insert_text: "/compact",
        description: "Compact current session context",
        usage: "/compact",
        visible_in_slash: true,
        visible_in_help: true,
        visible_in_summary: true,
    },
    CommandMetadata {
        name: "/tree",
        insert_text: "/tree",
        description: "Browse session history",
        usage: "/tree",
        visible_in_slash: true,
        visible_in_help: true,
        visible_in_summary: true,
    },
    CommandMetadata {
        name: "/undo",
        insert_text: "/undo",
        description: "Move to the previous completed turn",
        usage: "/undo",
        visible_in_slash: true,
        visible_in_help: true,
        visible_in_summary: true,
    },
    CommandMetadata {
        name: "/redo",
        insert_text: "/redo",
        description: "Move to the next undone turn",
        usage: "/redo",
        visible_in_slash: true,
        visible_in_help: true,
        visible_in_summary: true,
    },
    CommandMetadata {
        name: "/resume",
        insert_text: "/resume ",
        description: "Resume a previous session",
        usage: "/resume <session_id>",
        visible_in_slash: true,
        visible_in_help: true,
        visible_in_summary: true,
    },
    CommandMetadata {
        name: "/new",
        insert_text: "/new",
        description: "Start a new session",
        usage: "/new",
        visible_in_slash: true,
        visible_in_help: true,
        visible_in_summary: true,
    },
    CommandMetadata {
        name: "/context",
        insert_text: "/context",
        description: "Browse context details",
        usage: "/context",
        visible_in_slash: true,
        visible_in_help: true,
        visible_in_summary: true,
    },
    CommandMetadata {
        name: "/mcp",
        insert_text: "/mcp",
        description: "Browse MCP tools",
        usage: "/mcp",
        visible_in_slash: true,
        visible_in_help: true,
        visible_in_summary: true,
    },
    CommandMetadata {
        name: "/skill",
        insert_text: "/skill",
        description: "Browse local skills",
        usage: "/skill",
        visible_in_slash: true,
        visible_in_help: true,
        visible_in_summary: true,
    },
    CommandMetadata {
        name: "/child",
        insert_text: "/child",
        description: "View child subagent transcript",
        usage: "/child <first|next|prev>",
        visible_in_slash: true,
        visible_in_help: true,
        visible_in_summary: true,
    },
    CommandMetadata {
        name: "/children",
        insert_text: "/children",
        description: "View child subagent transcripts",
        usage: "/children <first|next|prev>",
        visible_in_slash: true,
        visible_in_help: true,
        visible_in_summary: false,
    },
    CommandMetadata {
        name: "/parent",
        insert_text: "/parent",
        description: "Return to parent transcript",
        usage: "/parent",
        visible_in_slash: true,
        visible_in_help: true,
        visible_in_summary: true,
    },
];

pub fn command_metadata() -> &'static [CommandMetadata] {
    COMMANDS
}

pub fn help_summary() -> String {
    let commands = [
        "/help",
        "/exit",
        "/quit",
        "/model",
        "/agents",
        "/fast",
        "/reasoning",
        "/permission",
        "/tool-output",
        "/scrollbar",
        "/theme",
        "/compact",
        "/tree",
        "/undo",
        "/redo",
        "/resume",
        "/new",
        "/context",
        "/mcp",
        "/skill",
        "/child",
        "/parent",
    ]
    .join(", ");
    format!(
        "Commands: {commands} · Delegation: {}",
        delegation_help_summary()
    )
}

pub fn parse_command(input: &str) -> Result<CommandIntent, CommandParseError> {
    let trimmed = input.trim();

    if trimmed.is_empty() {
        return Ok(CommandIntent::Prompt(String::new()));
    }

    if trimmed.eq_ignore_ascii_case("exit") || trimmed.eq_ignore_ascii_case("quit") {
        return Ok(CommandIntent::Exit);
    }

    if trimmed.starts_with("@skill(") {
        return Ok(CommandIntent::Prompt(trimmed.to_string()));
    }

    if trimmed.starts_with('@') {
        return parse_delegate_command(trimmed);
    }

    if !trimmed.starts_with('/') {
        return Ok(CommandIntent::Prompt(trimmed.to_string()));
    }

    let parts = trimmed.split_whitespace().collect::<Vec<_>>();
    let Some(name) = parts.first().copied() else {
        return Ok(CommandIntent::Prompt(String::new()));
    };
    let name = name.to_ascii_lowercase();

    if parts[0] != name {
        return Err(CommandParseError::new(format!(
            "Unknown command: {}. Type /help for available local commands.",
            parts[0]
        )));
    }

    match name.as_str() {
        "/help" | "/?" => expect_no_extra_args(&parts, name.as_str(), CommandIntent::Help),
        "/exit" | "/quit" => expect_no_extra_args(&parts, name.as_str(), CommandIntent::Exit),
        "/permission" | "/perm" => parse_permission(&parts),
        "/model" => parse_model(&parts),
        "/agents" => expect_no_extra_args(&parts, "/agents", CommandIntent::AgentsShow),
        "/fast" => expect_no_extra_args(&parts, "/fast", CommandIntent::FastToggle),
        "/reasoning" | "/think" => parse_reasoning(&parts),
        "/tool-output" => parse_tool_output(&parts),
        "/scrollbar" => parse_transcript_scrollbar(&parts),
        "/theme" => parse_theme(&parts),
        "/compact" => expect_no_extra_args(&parts, "/compact", CommandIntent::Compact),
        "/tree" => expect_no_extra_args(&parts, "/tree", CommandIntent::Tree),
        "/undo" => expect_no_extra_args(&parts, "/undo", CommandIntent::Undo),
        "/redo" => expect_no_extra_args(&parts, "/redo", CommandIntent::Redo),
        "/resume" => parse_resume(&parts),
        "/new" => expect_no_extra_args(&parts, "/new", CommandIntent::NewSession),
        "/context" => expect_no_extra_args(&parts, "/context", CommandIntent::ContextBrowse),
        "/mcp" => expect_no_extra_args(&parts, "/mcp", CommandIntent::McpBrowse),
        "/skill" => expect_no_extra_args(&parts, "/skill", CommandIntent::SkillBrowse),
        "/child" | "/children" => parse_child_navigation(&parts),
        "/parent" => expect_no_extra_args(&parts, "/parent", CommandIntent::Parent),
        _ => Err(CommandParseError::new(format!(
            "Unknown command: {}. Type /help for available local commands.",
            parts[0]
        ))),
    }
}

fn expect_no_extra_args(
    parts: &[&str],
    usage: &str,
    intent: CommandIntent,
) -> Result<CommandIntent, CommandParseError> {
    if parts.len() == 1 {
        Ok(intent)
    } else {
        Err(CommandParseError::new(format!("Usage: {usage}")))
    }
}

fn parse_permission(parts: &[&str]) -> Result<CommandIntent, CommandParseError> {
    match parts {
        ["/permission"] | ["/perm"] => Ok(CommandIntent::PermissionShow),
        ["/permission", mode] | ["/perm", mode] => match parse_permission_mode(mode) {
            Some(mode) => Ok(CommandIntent::PermissionSet(mode)),
            None => Err(CommandParseError::new(format!(
                "Unknown permission mode: {}. Use safe, default, auto, or yolo.",
                mode
            ))),
        },
        ["/permission", ..] => Err(CommandParseError::new(
            "Usage: /permission <safe|default|auto|yolo>",
        )),
        ["/perm", ..] => Err(CommandParseError::new(
            "Usage: /perm <safe|default|auto|yolo>",
        )),
        _ => unreachable!(),
    }
}

fn parse_model(parts: &[&str]) -> Result<CommandIntent, CommandParseError> {
    match parts {
        ["/model"] => Ok(CommandIntent::ModelShow),
        ["/model", model_id] => Ok(CommandIntent::ModelSet((*model_id).to_string())),
        ["/model", ..] => Err(CommandParseError::new("Usage: /model <id>")),
        _ => unreachable!(),
    }
}

fn parse_reasoning(parts: &[&str]) -> Result<CommandIntent, CommandParseError> {
    match parts {
        ["/reasoning"] | ["/think"] => Ok(CommandIntent::ReasoningShow),
        ["/reasoning", value] | ["/think", value] => match parse_reasoning_effort(value) {
            Some(effort) => Ok(CommandIntent::ReasoningSet(effort)),
            None => Err(CommandParseError::new(format!(
                "Unknown reasoning effort: {}. Use off, none, minimal, low, medium, high, xhigh, or max.",
                value.trim()
            ))),
        },
        ["/reasoning", ..] => Err(CommandParseError::new(
            "Usage: /reasoning <off|none|minimal|low|medium|high|xhigh|max>",
        )),
        ["/think", ..] => Err(CommandParseError::new(
            "Usage: /think <off|none|minimal|low|medium|high|xhigh|max>",
        )),
        _ => unreachable!(),
    }
}

fn parse_tool_output(parts: &[&str]) -> Result<CommandIntent, CommandParseError> {
    match parts {
        ["/tool-output"] => Ok(CommandIntent::ToolOutputSet(ToolOutputMode::Toggle)),
        ["/tool-output", value] => match parse_tool_output_mode(value) {
            Some(mode) => Ok(CommandIntent::ToolOutputSet(mode)),
            None => Err(CommandParseError::new(
                "Unknown tool output mode. Use on, off, expanded, truncated, full, or compact.",
            )),
        },
        ["/tool-output", ..] => Err(CommandParseError::new(
            "Usage: /tool-output <on|off|expanded|truncated|full|compact>",
        )),
        _ => unreachable!(),
    }
}

fn parse_transcript_scrollbar(parts: &[&str]) -> Result<CommandIntent, CommandParseError> {
    match parts {
        ["/scrollbar"] => Ok(CommandIntent::TranscriptScrollbarSet(
            TranscriptScrollbarMode::Toggle,
        )),
        ["/scrollbar", value] => match parse_transcript_scrollbar_mode(value) {
            Some(mode) => Ok(CommandIntent::TranscriptScrollbarSet(mode)),
            None => Err(CommandParseError::new(
                "Unknown scrollbar mode. Use on, off, show, hide, visible, or hidden.",
            )),
        },
        ["/scrollbar", ..] => Err(CommandParseError::new("Usage: /scrollbar [on|off]")),
        _ => unreachable!(),
    }
}

fn parse_theme(parts: &[&str]) -> Result<CommandIntent, CommandParseError> {
    match parts {
        ["/theme"] => Ok(CommandIntent::Theme(ThemeCommand::Show)),
        ["/theme", value] => match normalize_theme_command_id(value) {
            Some(theme) => Ok(CommandIntent::Theme(ThemeCommand::Set(theme))),
            None => Err(CommandParseError::new(
                "Unknown theme. Use dark, rainbow, or a themes/*.toml id (ocean, forest, rose, tokyonight, …).",
            )),
        },
        ["/theme", ..] => Err(CommandParseError::new(
            "Usage: /theme <dark|rainbow|<themes/*.toml>>",
        )),
        _ => unreachable!(),
    }
}

fn normalize_theme_command_id(value: &str) -> Option<String> {
    let trimmed = value.trim();
    if trimmed.is_empty() {
        return None;
    }
    if let Some(builtin) = ThemeName::parse(trimmed) {
        return Some(builtin.as_str().to_string());
    }
    let id = trimmed.to_ascii_lowercase();
    let id = match id.as_str() {
        "tokyo-night" | "tokyo_night" => "tokyonight".to_string(),
        _ => id,
    };
    if id
        .chars()
        .all(|ch| ch.is_ascii_alphanumeric() || ch == '-' || ch == '_')
    {
        Some(id)
    } else {
        None
    }
}

fn parse_resume(parts: &[&str]) -> Result<CommandIntent, CommandParseError> {
    match parts {
        ["/resume"] => Ok(CommandIntent::ResumeShow),
        ["/resume", session_id] => Ok(CommandIntent::Resume((*session_id).to_string())),
        ["/resume", ..] => Err(CommandParseError::new("Usage: /resume <session_id>")),
        _ => unreachable!(),
    }
}

fn parse_delegate_command(input: &str) -> Result<CommandIntent, CommandParseError> {
    let trimmed = input.trim();
    let Some(rest) = trimmed.strip_prefix('@') else {
        unreachable!();
    };
    let mut parts = rest.splitn(2, char::is_whitespace);
    let agent_name = parts.next().unwrap_or_default().trim();
    let task = parts.next().map(str::trim).unwrap_or_default();

    if agent_name.is_empty() {
        return Err(CommandParseError::new(format!(
            "Usage: {}",
            delegation_usage_list()
        )));
    }

    let Some(expert) = find_expert(agent_name) else {
        return Err(CommandParseError::new(unknown_expert_error(agent_name)));
    };

    if task.is_empty() {
        return Err(CommandParseError::new(format!("Usage: {}", expert.usage)));
    }

    Ok(CommandIntent::Delegate {
        agent_name: expert.agent_name.to_string(),
        task: task.to_string(),
    })
}

fn parse_child_navigation(parts: &[&str]) -> Result<CommandIntent, CommandParseError> {
    match parts {
        ["/child"] | ["/children"] => Ok(CommandIntent::Child(ChildNavigation::First)),
        ["/child", value] | ["/children", value] => match value.to_ascii_lowercase().as_str() {
            "first" => Ok(CommandIntent::Child(ChildNavigation::First)),
            "next" => Ok(CommandIntent::Child(ChildNavigation::Next)),
            "prev" | "previous" => Ok(CommandIntent::Child(ChildNavigation::Prev)),
            other => Err(CommandParseError::new(format!(
                "Unknown child navigation: {other}. Use first, next, or prev."
            ))),
        },
        ["/child", ..] => Err(CommandParseError::new("Usage: /child <first|next|prev>")),
        ["/children", ..] => Err(CommandParseError::new("Usage: /children <first|next|prev>")),
        _ => unreachable!(),
    }
}

fn parse_permission_mode(value: &str) -> Option<PermissionMode> {
    PermissionMode::parse(value)
}

pub fn parse_reasoning_effort(value: &str) -> Option<ModelReasoningEffort> {
    match value.trim().to_ascii_lowercase().as_str() {
        "off" | "none" => Some(ModelReasoningEffort::None),
        "minimal" => Some(ModelReasoningEffort::Minimal),
        "low" => Some(ModelReasoningEffort::Low),
        "medium" => Some(ModelReasoningEffort::Medium),
        "high" => Some(ModelReasoningEffort::High),
        "xhigh" | "x-high" | "extra-high" => Some(ModelReasoningEffort::Xhigh),
        "max" => Some(ModelReasoningEffort::Max),
        _ => None,
    }
}

fn parse_tool_output_mode(value: &str) -> Option<ToolOutputMode> {
    if value.eq_ignore_ascii_case("on")
        || value.eq_ignore_ascii_case("expanded")
        || value.eq_ignore_ascii_case("full")
    {
        Some(ToolOutputMode::Expanded)
    } else if value.eq_ignore_ascii_case("off")
        || value.eq_ignore_ascii_case("truncated")
        || value.eq_ignore_ascii_case("compact")
    {
        Some(ToolOutputMode::Truncated)
    } else {
        None
    }
}

fn parse_transcript_scrollbar_mode(value: &str) -> Option<TranscriptScrollbarMode> {
    if value.eq_ignore_ascii_case("on")
        || value.eq_ignore_ascii_case("show")
        || value.eq_ignore_ascii_case("visible")
    {
        Some(TranscriptScrollbarMode::Visible)
    } else if value.eq_ignore_ascii_case("off")
        || value.eq_ignore_ascii_case("hide")
        || value.eq_ignore_ascii_case("hidden")
    {
        Some(TranscriptScrollbarMode::Hidden)
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_unknown_and_invalid_commands_locally() {
        assert_eq!(
            parse_command("/bogus"),
            Err(CommandParseError::new(
                "Unknown command: /bogus. Type /help for available local commands."
            ))
        );
        assert_eq!(
            parse_command("/MODEL gpt-5.5"),
            Err(CommandParseError::new(
                "Unknown command: /MODEL. Type /help for available local commands."
            ))
        );
        assert_eq!(
            parse_command("/model a b"),
            Err(CommandParseError::new("Usage: /model <id>"))
        );
        assert_eq!(
            parse_command("/fast now"),
            Err(CommandParseError::new("Usage: /fast"))
        );
        assert_eq!(
            parse_command("/compact now"),
            Err(CommandParseError::new("Usage: /compact"))
        );
        for retired in ["/branch", "/checkout"] {
            assert_eq!(
                parse_command(retired),
                Err(CommandParseError::new(format!(
                    "Unknown command: {retired}. Type /help for available local commands."
                )))
            );
        }
        assert_eq!(
            parse_command("/scrollbar maybe"),
            Err(CommandParseError::new(
                "Unknown scrollbar mode. Use on, off, show, hide, visible, or hidden."
            ))
        );
        assert_eq!(
            parse_command("/theme bad!id"),
            Err(CommandParseError::new(
                "Unknown theme. Use dark, rainbow, or a themes/*.toml id (ocean, forest, rose, tokyonight, …)."
            ))
        );
        assert_eq!(
            parse_command("/theme tokyonight"),
            Ok(CommandIntent::Theme(ThemeCommand::Set("tokyonight".into())))
        );
        assert_eq!(
            parse_command("/theme sunset"),
            Ok(CommandIntent::Theme(ThemeCommand::Set("sunset".into())))
        );
        assert_eq!(
            parse_command("@fixer"),
            Err(CommandParseError::new("Usage: @fixer <task>"))
        );
        assert_eq!(
            parse_command("@unknown foo"),
            Err(CommandParseError::new(
                "Unknown expert: @unknown. Use @explorer, @fixer, @oracle, @designer, @librarian, or @general."
            ))
        );
        assert_eq!(
            parse_command("/explore inspect src/main.rs"),
            Err(CommandParseError::new(
                "Unknown command: /explore. Type /help for available local commands."
            ))
        );
        assert_eq!(
            parse_command("/fixer wire child view"),
            Err(CommandParseError::new(
                "Unknown command: /fixer. Type /help for available local commands."
            ))
        );
        assert_eq!(
            parse_command("/child sideways"),
            Err(CommandParseError::new(
                "Unknown child navigation: sideways. Use first, next, or prev."
            ))
        );
        assert_eq!(
            parse_command("/reasoning absurd"),
            Err(CommandParseError::new(
                "Unknown reasoning effort: absurd. Use off, none, minimal, low, medium, high, xhigh, or max."
            ))
        );
    }
}
