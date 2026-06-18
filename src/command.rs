use crate::permission::PermissionMode;
use crate::request_builder::ModelReasoningEffort;

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

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CommandIntent {
    Prompt(String),
    Help,
    Exit,
    PermissionShow,
    PermissionSet(PermissionMode),
    ModelShow,
    ModelSet(String),
    ReasoningShow,
    ReasoningSet(ModelReasoningEffort),
    ToolOutputSet(ToolOutputMode),
    ResumeShow,
    Resume(String),
    NewSession,
    Explore(String),
    Fixer(String),
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
        usage: "/permission <safe|default|solo>",
        visible_in_slash: true,
        visible_in_help: true,
        visible_in_summary: true,
    },
    CommandMetadata {
        name: "/perm",
        insert_text: "/perm ",
        description: "Alias for /permission",
        usage: "/perm <safe|default|solo>",
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
        name: "/explore",
        insert_text: "/explore ",
        description: "Run read-only explorer subagent",
        usage: "/explore <task>",
        visible_in_slash: true,
        visible_in_help: true,
        visible_in_summary: true,
    },
    CommandMetadata {
        name: "/fixer",
        insert_text: "/fixer ",
        description: "Run writable fixer subagent",
        usage: "/fixer <task>",
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
        "/reasoning",
        "/permission",
        "/tool-output",
        "/resume",
        "/new",
        "/explore",
        "/fixer",
        "/child",
        "/parent",
    ]
    .join(", ");
    format!("Commands: {commands}")
}

pub fn parse_command(input: &str) -> Result<CommandIntent, CommandParseError> {
    let trimmed = input.trim();

    if trimmed.is_empty() {
        return Ok(CommandIntent::Prompt(String::new()));
    }

    if trimmed.eq_ignore_ascii_case("exit") || trimmed.eq_ignore_ascii_case("quit") {
        return Ok(CommandIntent::Exit);
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
        "/reasoning" | "/think" => parse_reasoning(&parts),
        "/tool-output" => parse_tool_output(&parts),
        "/resume" => parse_resume(&parts),
        "/new" => expect_no_extra_args(&parts, "/new", CommandIntent::NewSession),
        "/explore" => parse_task_command(trimmed, "/explore", CommandIntent::Explore),
        "/fixer" => parse_task_command(trimmed, "/fixer", CommandIntent::Fixer),
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
                "Unknown permission mode: {}. Use safe, default, or solo.",
                mode
            ))),
        },
        ["/permission", ..] => Err(CommandParseError::new(
            "Usage: /permission <safe|default|solo>",
        )),
        ["/perm", ..] => Err(CommandParseError::new("Usage: /perm <safe|default|solo>")),
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
                "Unknown reasoning effort: {}. Use off, none, minimal, low, medium, high, or xhigh.",
                value.trim()
            ))),
        },
        ["/reasoning", ..] => Err(CommandParseError::new(
            "Usage: /reasoning <off|none|minimal|low|medium|high|xhigh>",
        )),
        ["/think", ..] => Err(CommandParseError::new(
            "Usage: /think <off|none|minimal|low|medium|high|xhigh>",
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

fn parse_resume(parts: &[&str]) -> Result<CommandIntent, CommandParseError> {
    match parts {
        ["/resume"] => Ok(CommandIntent::ResumeShow),
        ["/resume", session_id] => Ok(CommandIntent::Resume((*session_id).to_string())),
        ["/resume", ..] => Err(CommandParseError::new("Usage: /resume <session_id>")),
        _ => unreachable!(),
    }
}

fn parse_task_command(
    input: &str,
    command_name: &str,
    build: fn(String) -> CommandIntent,
) -> Result<CommandIntent, CommandParseError> {
    let task = input
        .strip_prefix(command_name)
        .map(str::trim)
        .unwrap_or_default();
    if task.is_empty() {
        return Err(CommandParseError::new(format!(
            "Usage: {command_name} <task>"
        )));
    }
    Ok(build(task.to_string()))
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
    match value.trim().to_ascii_lowercase().as_str() {
        "safe" => Some(PermissionMode::Safe),
        "default" => Some(PermissionMode::Default),
        "solo" => Some(PermissionMode::Solo),
        _ => None,
    }
}

pub fn parse_reasoning_effort(value: &str) -> Option<ModelReasoningEffort> {
    match value.trim().to_ascii_lowercase().as_str() {
        "off" | "none" => Some(ModelReasoningEffort::None),
        "minimal" => Some(ModelReasoningEffort::Minimal),
        "low" => Some(ModelReasoningEffort::Low),
        "medium" => Some(ModelReasoningEffort::Medium),
        "high" => Some(ModelReasoningEffort::High),
        "xhigh" | "x-high" | "extra-high" => Some(ModelReasoningEffort::Xhigh),
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn metadata_covers_shared_commands_and_aliases() {
        let names = command_metadata()
            .iter()
            .map(|command| command.name)
            .collect::<Vec<_>>();
        for name in [
            "/help",
            "/exit",
            "/quit",
            "/permission",
            "/perm",
            "/model",
            "/reasoning",
            "/think",
            "/tool-output",
            "/resume",
            "/new",
            "/explore",
            "/fixer",
            "/child",
            "/children",
            "/parent",
        ] {
            assert!(names.contains(&name), "missing {name}");
        }
    }

    #[test]
    fn parses_plain_prompts_and_exit_aliases() {
        assert_eq!(
            parse_command("hello there"),
            Ok(CommandIntent::Prompt("hello there".into()))
        );
        assert_eq!(parse_command(" exit "), Ok(CommandIntent::Exit));
        assert_eq!(parse_command("/quit"), Ok(CommandIntent::Exit));
    }

    #[test]
    fn parses_shared_commands_and_aliases() {
        assert_eq!(parse_command("/?"), Ok(CommandIntent::Help));
        assert_eq!(
            parse_command("/perm safe"),
            Ok(CommandIntent::PermissionSet(PermissionMode::Safe))
        );
        assert_eq!(
            parse_command("/model gpt-5.5"),
            Ok(CommandIntent::ModelSet("gpt-5.5".into()))
        );
        assert_eq!(
            parse_command("/think x-high"),
            Ok(CommandIntent::ReasoningSet(ModelReasoningEffort::Xhigh))
        );
        assert_eq!(
            parse_command("/tool-output compact"),
            Ok(CommandIntent::ToolOutputSet(ToolOutputMode::Truncated))
        );
        assert_eq!(parse_command("/resume"), Ok(CommandIntent::ResumeShow));
        assert_eq!(
            parse_command("/explore inspect src/main.rs"),
            Ok(CommandIntent::Explore("inspect src/main.rs".into()))
        );
        assert_eq!(
            parse_command("/fixer wire child view"),
            Ok(CommandIntent::Fixer("wire child view".into()))
        );
        assert_eq!(
            parse_command("/children previous"),
            Ok(CommandIntent::Child(ChildNavigation::Prev))
        );
        assert_eq!(parse_command("/parent"), Ok(CommandIntent::Parent));
    }

    #[test]
    fn parses_whitespace_consistently() {
        assert_eq!(
            parse_command("  /permission   default  "),
            Ok(CommandIntent::PermissionSet(PermissionMode::Default))
        );
        assert_eq!(
            parse_command("  /explore   inspect src/lib.rs  "),
            Ok(CommandIntent::Explore("inspect src/lib.rs".into()))
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
            parse_command("/explore"),
            Err(CommandParseError::new("Usage: /explore <task>"))
        );
        assert_eq!(
            parse_command("/fixer"),
            Err(CommandParseError::new("Usage: /fixer <task>"))
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
                "Unknown reasoning effort: absurd. Use off, none, minimal, low, medium, high, or xhigh."
            ))
        );
    }
}
