use crate::delegation::{delegation_help_summary, delegation_usage_list, find_expert};
use crate::permission::PermissionMode;
use crate::request_builder::ModelReasoningEffort;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CommandMetadata {
    pub name: &'static str,
    pub insert_text: &'static str,
    pub description_key: &'static str,
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

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum ThoughtsDisplayMode {
    Compact,
    Titles,
    #[default]
    Full,
}

impl ThoughtsDisplayMode {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Compact => "compact",
            Self::Titles => "titles",
            Self::Full => "full",
        }
    }

    pub fn parse(value: &str) -> Option<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "1" | "compact" => Some(Self::Compact),
            "2" | "titles" => Some(Self::Titles),
            "3" | "full" => Some(Self::Full),
            _ => None,
        }
    }

    pub const fn label(self) -> &'static str {
        match self {
            Self::Compact => "Compact",
            Self::Titles => "Titles",
            Self::Full => "Full",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
#[derive(Default)]
pub enum ThemeName {
    #[default]
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

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ThemeCommand {
    Show,
    Set(String),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CommandIntent {
    Prompt(String),
    Language(Option<String>),
    Delegate { agent_name: String, task: String },
    Help,
    Exit,
    PermissionShow,
    PermissionSet(PermissionMode),
    ModelShow,
    ModelSet(String),
    AgentsShow,
    AnchoredToggle,
    FastToggle,
    ReasoningShow,
    ReasoningSet(ModelReasoningEffort),
    ThoughtsShow,
    ThoughtsSet(ThoughtsDisplayMode),
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
    key: String,
    args: Vec<(String, String)>,
}

impl CommandParseError {
    fn new(message: impl Into<String>) -> Self {
        let message = message.into();
        if let Some(command) = message
            .strip_prefix("Unknown command: ")
            .and_then(|value| value.strip_suffix(". Type /help for available local commands."))
        {
            return Self::unknown_command(command);
        }
        if let Some(usage) = message.strip_prefix("Usage: ") {
            return Self::usage(usage);
        }
        Self::with_args("parse.literal", [("message", message.as_str())])
    }

    fn unknown_command(command: &str) -> Self {
        Self::with_args("parse.unknown_command", [("command", command)])
    }

    fn usage(usage: &str) -> Self {
        Self::with_args("parse.usage", [("usage", usage)])
    }

    fn with_args<const N: usize>(key: &str, args: [(&str, &str); N]) -> Self {
        Self {
            key: key.to_string(),
            args: args
                .into_iter()
                .map(|(name, value)| (name.to_string(), value.to_string()))
                .collect(),
        }
    }

    #[cfg(test)]
    pub fn key(&self) -> &str {
        &self.key
    }

    #[cfg(test)]
    pub fn args(&self) -> &[(String, String)] {
        &self.args
    }

    pub fn render(&self, translator: &crate::tui::i18n::Translator) -> String {
        let args = self
            .args
            .iter()
            .map(|(name, value)| (name.as_str(), value.as_str()))
            .collect::<Vec<_>>();
        translator.t_fmt(&self.key, &args)
    }
}

const COMMANDS: &[CommandMetadata] = &[
    CommandMetadata {
        name: "/help",
        insert_text: "/help",
        description_key: "command.help",
        usage: "/help",
        visible_in_slash: true,
        visible_in_help: true,
        visible_in_summary: true,
    },
    CommandMetadata {
        name: "/?",
        insert_text: "/?",
        description_key: "command.help",
        usage: "/?",
        visible_in_slash: false,
        visible_in_help: true,
        visible_in_summary: false,
    },
    CommandMetadata {
        name: "/exit",
        insert_text: "/exit",
        description_key: "command.exit",
        usage: "/exit",
        visible_in_slash: true,
        visible_in_help: true,
        visible_in_summary: true,
    },
    CommandMetadata {
        name: "/quit",
        insert_text: "/quit",
        description_key: "command.exit",
        usage: "/quit",
        visible_in_slash: true,
        visible_in_help: true,
        visible_in_summary: true,
    },
    CommandMetadata {
        name: "/permission",
        insert_text: "/permission ",
        description_key: "command.permission",
        usage: "/permission <safe|default|auto|yolo>",
        visible_in_slash: true,
        visible_in_help: true,
        visible_in_summary: true,
    },
    CommandMetadata {
        name: "/perm",
        insert_text: "/perm ",
        description_key: "command.permission",
        usage: "/perm <safe|default|auto|yolo>",
        visible_in_slash: false,
        visible_in_help: true,
        visible_in_summary: false,
    },
    CommandMetadata {
        name: "/language",
        insert_text: "/language ",
        description_key: "command.language",
        usage: "/language [en|zh-CN]",
        visible_in_slash: true,
        visible_in_help: true,
        visible_in_summary: true,
    },
    CommandMetadata {
        name: "/lang",
        insert_text: "/lang ",
        description_key: "command.language",
        usage: "/lang [en|zh-CN]",
        visible_in_slash: false,
        visible_in_help: true,
        visible_in_summary: false,
    },
    CommandMetadata {
        name: "/model",
        insert_text: "/model ",
        description_key: "command.model",
        usage: "/model <id>",
        visible_in_slash: true,
        visible_in_help: true,
        visible_in_summary: true,
    },
    CommandMetadata {
        name: "/anchored",
        insert_text: "/anchored",
        description_key: "command.anchored",
        usage: "/anchored",
        visible_in_slash: true,
        visible_in_help: true,
        visible_in_summary: true,
    },
    CommandMetadata {
        name: "/agents",
        insert_text: "/agents",
        description_key: "command.agents",
        usage: "/agents",
        visible_in_slash: true,
        visible_in_help: true,
        visible_in_summary: true,
    },
    CommandMetadata {
        name: "/fast",
        insert_text: "/fast",
        description_key: "command.fast",
        usage: "/fast",
        visible_in_slash: true,
        visible_in_help: true,
        visible_in_summary: true,
    },
    CommandMetadata {
        name: "/reasoning",
        insert_text: "/reasoning ",
        description_key: "command.reasoning",
        usage: "/reasoning <off|none|minimal|low|medium|high|xhigh>",
        visible_in_slash: true,
        visible_in_help: true,
        visible_in_summary: true,
    },
    CommandMetadata {
        name: "/think",
        insert_text: "/think ",
        description_key: "command.reasoning",
        usage: "/think <off|none|minimal|low|medium|high|xhigh>",
        visible_in_slash: false,
        visible_in_help: true,
        visible_in_summary: false,
    },
    CommandMetadata {
        name: "/thoughts",
        insert_text: "/thoughts",
        description_key: "command.thoughts",
        usage: "/thoughts <compact|titles|full>",
        visible_in_slash: true,
        visible_in_help: true,
        visible_in_summary: true,
    },
    CommandMetadata {
        name: "/tool-output",
        insert_text: "/tool-output ",
        description_key: "command.tool_output",
        usage: "/tool-output <on|off|expanded|truncated|full|compact>",
        visible_in_slash: true,
        visible_in_help: true,
        visible_in_summary: true,
    },
    CommandMetadata {
        name: "/scrollbar",
        insert_text: "/scrollbar ",
        description_key: "command.scrollbar",
        usage: "/scrollbar [on|off]",
        visible_in_slash: true,
        visible_in_help: true,
        visible_in_summary: true,
    },
    CommandMetadata {
        name: "/theme",
        insert_text: "/theme ",
        description_key: "command.theme",
        usage: "/theme [dark|rainbow|<themes/*.toml>]",
        visible_in_slash: true,
        visible_in_help: true,
        visible_in_summary: true,
    },
    CommandMetadata {
        name: "/compact",
        insert_text: "/compact",
        description_key: "command.compact",
        usage: "/compact",
        visible_in_slash: true,
        visible_in_help: true,
        visible_in_summary: true,
    },
    CommandMetadata {
        name: "/tree",
        insert_text: "/tree",
        description_key: "command.tree",
        usage: "/tree",
        visible_in_slash: true,
        visible_in_help: true,
        visible_in_summary: true,
    },
    CommandMetadata {
        name: "/undo",
        insert_text: "/undo",
        description_key: "command.undo",
        usage: "/undo",
        visible_in_slash: true,
        visible_in_help: true,
        visible_in_summary: true,
    },
    CommandMetadata {
        name: "/redo",
        insert_text: "/redo",
        description_key: "command.redo",
        usage: "/redo",
        visible_in_slash: true,
        visible_in_help: true,
        visible_in_summary: true,
    },
    CommandMetadata {
        name: "/resume",
        insert_text: "/resume ",
        description_key: "command.resume",
        usage: "/resume <session_id>",
        visible_in_slash: true,
        visible_in_help: true,
        visible_in_summary: true,
    },
    CommandMetadata {
        name: "/new",
        insert_text: "/new",
        description_key: "command.new",
        usage: "/new",
        visible_in_slash: true,
        visible_in_help: true,
        visible_in_summary: true,
    },
    CommandMetadata {
        name: "/context",
        insert_text: "/context",
        description_key: "command.context",
        usage: "/context",
        visible_in_slash: true,
        visible_in_help: true,
        visible_in_summary: true,
    },
    CommandMetadata {
        name: "/mcp",
        insert_text: "/mcp",
        description_key: "command.mcp",
        usage: "/mcp",
        visible_in_slash: true,
        visible_in_help: true,
        visible_in_summary: true,
    },
    CommandMetadata {
        name: "/skill",
        insert_text: "/skill",
        description_key: "command.skill",
        usage: "/skill",
        visible_in_slash: true,
        visible_in_help: true,
        visible_in_summary: true,
    },
    CommandMetadata {
        name: "/child",
        insert_text: "/child",
        description_key: "command.child",
        usage: "/child <first|next|prev>",
        visible_in_slash: true,
        visible_in_help: true,
        visible_in_summary: true,
    },
    CommandMetadata {
        name: "/children",
        insert_text: "/children",
        description_key: "command.children",
        usage: "/children <first|next|prev>",
        visible_in_slash: true,
        visible_in_help: true,
        visible_in_summary: false,
    },
    CommandMetadata {
        name: "/parent",
        insert_text: "/parent",
        description_key: "command.parent",
        usage: "/parent",
        visible_in_slash: true,
        visible_in_help: true,
        visible_in_summary: true,
    },
];

pub fn command_metadata() -> &'static [CommandMetadata] {
    COMMANDS
}

pub fn help_summary(translator: &crate::tui::i18n::Translator) -> String {
    let commands = [
        "/help",
        "/exit",
        "/quit",
        "/model",
        "/agents",
        "/anchored",
        "/fast",
        "/reasoning",
        "/thoughts",
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
    translator.t_fmt(
        "help.summary",
        &[
            ("commands", &commands),
            ("delegation", &delegation_help_summary()),
        ],
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
        return Err(CommandParseError::unknown_command(parts[0]));
    }

    match name.as_str() {
        "/help" | "/?" => expect_no_extra_args(&parts, name.as_str(), CommandIntent::Help),
        "/language" | "/lang" => parse_language(&parts),
        "/exit" | "/quit" => expect_no_extra_args(&parts, name.as_str(), CommandIntent::Exit),
        "/permission" | "/perm" => parse_permission(&parts),
        "/model" => parse_model(&parts),
        "/agents" => expect_no_extra_args(&parts, "/agents", CommandIntent::AgentsShow),
        "/anchored" => expect_no_extra_args(&parts, "/anchored", CommandIntent::AnchoredToggle),
        "/fast" => expect_no_extra_args(&parts, "/fast", CommandIntent::FastToggle),
        "/reasoning" | "/think" => parse_reasoning(&parts),
        "/thoughts" => parse_thoughts(&parts),
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
        _ => Err(CommandParseError::unknown_command(parts[0])),
    }
}

fn parse_language(parts: &[&str]) -> Result<CommandIntent, CommandParseError> {
    match parts {
        ["/language"] | ["/lang"] => Ok(CommandIntent::Language(None)),
        ["/language", value] | ["/lang", value] => {
            Ok(CommandIntent::Language(Some((*value).to_string())))
        }
        [name, ..] => Err(CommandParseError::usage(&format!("{name} [en|zh-CN]"))),
        _ => unreachable!(),
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
        Err(CommandParseError::usage(usage))
    }
}

fn parse_permission(parts: &[&str]) -> Result<CommandIntent, CommandParseError> {
    match parts {
        ["/permission"] | ["/perm"] => Ok(CommandIntent::PermissionShow),
        ["/permission", mode] | ["/perm", mode] => match parse_permission_mode(mode) {
            Some(mode) => Ok(CommandIntent::PermissionSet(mode)),
            None => Err(CommandParseError::with_args(
                "parse.unknown_permission_mode",
                [("mode", mode)],
            )),
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
            None => Err(CommandParseError::with_args(
                "parse.unknown_reasoning_effort",
                [("value", value.trim())],
            )),
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

fn parse_thoughts(parts: &[&str]) -> Result<CommandIntent, CommandParseError> {
    match parts {
        ["/thoughts"] => Ok(CommandIntent::ThoughtsShow),
        ["/thoughts", value] => match ThoughtsDisplayMode::parse(value) {
            Some(mode) => Ok(CommandIntent::ThoughtsSet(mode)),
            None => Err(CommandParseError::with_args("parse.unknown_thoughts", [])),
        },
        ["/thoughts", ..] => Err(CommandParseError::new(
            "Usage: /thoughts <compact|titles|full>",
        )),
        _ => unreachable!(),
    }
}

fn parse_tool_output(parts: &[&str]) -> Result<CommandIntent, CommandParseError> {
    match parts {
        ["/tool-output"] => Ok(CommandIntent::ToolOutputSet(ToolOutputMode::Toggle)),
        ["/tool-output", value] => match parse_tool_output_mode(value) {
            Some(mode) => Ok(CommandIntent::ToolOutputSet(mode)),
            None => Err(CommandParseError::with_args(
                "parse.unknown_tool_output",
                [],
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
            None => Err(CommandParseError::with_args("parse.unknown_scrollbar", [])),
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
            None => Err(CommandParseError::with_args("parse.unknown_theme", [])),
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
        return Err(CommandParseError::with_args(
            "parse.unknown_expert",
            [("value", agent_name)],
        ));
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
            other => Err(CommandParseError::with_args(
                "parse.unknown_child_navigation",
                [("value", other)],
            )),
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
    fn renders_structured_parse_errors_for_each_language() {
        let error = CommandParseError::unknown_command("/bogus");
        assert_eq!(error.key(), "parse.unknown_command");
        assert_eq!(error.args(), [("command".into(), "/bogus".into())]);
        assert_eq!(
            error.render(&crate::tui::i18n::Translator::new(
                crate::tui::i18n::Language::En,
            )),
            "Unknown command: /bogus. Type /help for available local commands."
        );
        assert_eq!(
            error.render(&crate::tui::i18n::Translator::new(
                crate::tui::i18n::Language::ZhCn,
            )),
            "未知命令：/bogus。输入 /help 查看可用的本地命令。"
        );
    }

    #[test]
    fn parses_language_aliases_and_values() {
        assert_eq!(
            parse_command("/language"),
            Ok(CommandIntent::Language(None))
        );
        assert_eq!(parse_command("/lang"), Ok(CommandIntent::Language(None)));
        assert_eq!(
            parse_command("/language zh-CN"),
            Ok(CommandIntent::Language(Some("zh-CN".into())))
        );
        assert_eq!(
            parse_command("/lang en"),
            Ok(CommandIntent::Language(Some("en".into())))
        );
        assert_eq!(
            parse_command("/lang en zh-CN"),
            Err(CommandParseError::usage("/lang [en|zh-CN]"))
        );
    }

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
            Err(CommandParseError::with_args("parse.unknown_scrollbar", []))
        );
        assert_eq!(
            parse_command("/theme bad!id"),
            Err(CommandParseError::with_args("parse.unknown_theme", []))
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
            Err(CommandParseError::with_args(
                "parse.unknown_expert",
                [("value", "unknown")]
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
            Err(CommandParseError::with_args(
                "parse.unknown_child_navigation",
                [("value", "sideways")]
            ))
        );
        assert_eq!(
            parse_command("/reasoning absurd"),
            Err(CommandParseError::with_args(
                "parse.unknown_reasoning_effort",
                [("value", "absurd")]
            ))
        );
        assert_eq!(parse_command("/thoughts"), Ok(CommandIntent::ThoughtsShow));
        assert_eq!(
            parse_command("/thoughts 2"),
            Ok(CommandIntent::ThoughtsSet(ThoughtsDisplayMode::Titles))
        );
        assert_eq!(
            parse_command("/thoughts full"),
            Ok(CommandIntent::ThoughtsSet(ThoughtsDisplayMode::Full))
        );
        assert_eq!(
            parse_command("/thoughts verbose"),
            Err(CommandParseError::with_args("parse.unknown_thoughts", []))
        );
    }
}
