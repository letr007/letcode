//! Slash commands ACP clients send as prompt content.
//!
//! ACP carries a command as prompt text (`/model test/model`), so dispatch
//! reads the same vocabulary the terminal frontend parses. An advertised
//! command reaches the engine as a session command and is answered from the
//! engine report carrying its outcome, so it is either applied or refused with
//! the engine's reason.
//!
//! Text that names a letcode command this frontend does not dispatch is refused
//! as well: the client was told which commands this session accepts, and a
//! command ACP cannot apply has no engine turn to start.

use agent_client_protocol::schema::v1::{
    AvailableCommand, AvailableCommandInput, AvailableCommandsUpdate, SessionUpdate,
    UnstructuredCommandInput,
};

use crate::command::{command_metadata, parse_command};
use crate::session::SessionCommand;

/// A command this frontend dispatches to the engine.
struct SlashCommand {
    /// The name clients advertise. The slash they send it with is implied.
    name: &'static str,
    description: &'static str,
    /// The values the command's argument selects, shown until the client sends
    /// one. A command taking no argument advertises none.
    hint: Option<&'static str>,
}

/// The commands whose outcome the engine reports.
const COMMANDS: [SlashCommand; 9] = [
    SlashCommand {
        name: "permission",
        description: "Set the session permission mode",
        hint: Some("safe|default|auto|yolo"),
    },
    SlashCommand {
        name: "model",
        description: "Switch the model the session runs with",
        hint: Some("model id"),
    },
    SlashCommand {
        name: "reasoning",
        description: "Set the reasoning effort of the model the session runs with",
        hint: Some("off|none|minimal|low|medium|high|xhigh"),
    },
    SlashCommand {
        name: "compact",
        description: "Compact the session context",
        hint: None,
    },
    SlashCommand {
        name: "fast",
        description: "Toggle fast mode",
        hint: None,
    },
    SlashCommand {
        name: "new",
        description: "Start a new session",
        hint: None,
    },
    SlashCommand {
        name: "resume",
        description: "Resume a session by id",
        hint: Some("session id"),
    },
    SlashCommand {
        name: "undo",
        description: "Undo the last turn",
        hint: None,
    },
    SlashCommand {
        name: "redo",
        description: "Redo the last undone turn",
        hint: None,
    },
];

/// The update that tells a client which commands this session accepts.
pub(super) fn update() -> SessionUpdate {
    SessionUpdate::AvailableCommandsUpdate(AvailableCommandsUpdate::new(
        COMMANDS
            .iter()
            .map(|command| {
                let advertised = AvailableCommand::new(command.name, command.description);
                match command.hint {
                    Some(hint) => advertised.input(AvailableCommandInput::Unstructured(
                        UnstructuredCommandInput::new(hint),
                    )),
                    None => advertised,
                }
            })
            .collect(),
    ))
}

/// What the text of a prompt asks this frontend to do.
#[derive(Debug, PartialEq, Eq)]
pub(super) enum SlashRequest {
    /// Ordinary prompt content: the engine runs it as a turn.
    Prompt,
    /// A command this frontend dispatches to the engine.
    Command(SessionCommand),
    /// Command text this frontend answers with an error instead of dispatching.
    Rejected(String),
}

/// Reads prompt text as a command.
pub(super) fn request(text: &str) -> SlashRequest {
    let Some(command) = advertised_command(text) else {
        return refused_command(text);
    };
    match parse_command(text) {
        // The command has no engine form when its argument is missing or names
        // no value: there is nothing for the engine to apply.
        Ok(intent) => match SessionCommand::from_command_intent(intent) {
            Some(dispatched) => SlashRequest::Command(dispatched),
            None => SlashRequest::Rejected(usage(command)),
        },
        Err(_) => SlashRequest::Rejected(usage(command)),
    }
}

/// Slash text that names a letcode command this frontend does not dispatch.
///
/// A command this session does not accept is answered as such instead of
/// reaching the model, because a client sends commands it expects letcode to
/// run. Text naming no command at all stays prompt content: a client that
/// writes a path, a sentence, or a word letcode does not know keeps reaching
/// the engine the way it did before commands existed.
fn refused_command(text: &str) -> SlashRequest {
    let Some(name) = named_command(text) else {
        return SlashRequest::Prompt;
    };
    SlashRequest::Rejected(format!("{name} is not available over ACP"))
}

/// The letcode command `text` opens with, if it names one.
fn named_command(text: &str) -> Option<&'static str> {
    let token = text.split_whitespace().next()?;
    command_metadata()
        .iter()
        .find(|entry| entry.name == token)
        .map(|entry| entry.name)
}

/// The advertised command `text` opens with.
fn advertised_command(text: &str) -> Option<&'static SlashCommand> {
    let rest = text.trim_start().strip_prefix('/')?;
    COMMANDS.iter().find(|command| {
        rest.strip_prefix(command.name)
            .is_some_and(|rest| rest.is_empty() || rest.starts_with(char::is_whitespace))
    })
}

/// The usage line the command vocabulary documents `command` with.
fn usage(command: &SlashCommand) -> String {
    let usage = command_metadata()
        .iter()
        .find(|entry| entry.name.strip_prefix('/') == Some(command.name))
        .map_or(command.name, |entry| entry.usage);
    format!("Usage: {usage}")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::permission::PermissionMode;
    use crate::request_builder::ModelReasoningEffort;

    #[test]
    fn advertised_commands_dispatch_the_commands_the_engine_confirms() {
        assert_eq!(
            request("/permission yolo"),
            SlashRequest::Command(SessionCommand::SetPermissionMode(PermissionMode::Yolo))
        );
        assert_eq!(
            request("/model test/model"),
            SlashRequest::Command(SessionCommand::SetModel("test/model".into()))
        );
        assert_eq!(
            request("/reasoning high"),
            SlashRequest::Command(SessionCommand::SetReasoningEffort(
                ModelReasoningEffort::High
            ))
        );
        assert_eq!(
            request("/compact"),
            SlashRequest::Command(SessionCommand::Compact)
        );
        assert_eq!(
            request("/fast"),
            SlashRequest::Command(SessionCommand::ToggleFastMode)
        );
        assert_eq!(
            request("/new"),
            SlashRequest::Command(SessionCommand::NewSession)
        );
        assert_eq!(
            request("/resume session-1"),
            SlashRequest::Command(SessionCommand::ResumeSession("session-1".into()))
        );
        assert_eq!(
            request("/undo"),
            SlashRequest::Command(SessionCommand::Undo)
        );
        assert_eq!(
            request("/redo"),
            SlashRequest::Command(SessionCommand::Redo)
        );
    }

    #[test]
    fn advertised_commands_name_commands_the_project_serves() {
        for command in &COMMANDS {
            let entry = command_metadata()
                .iter()
                .find(|entry| entry.name.strip_prefix('/') == Some(command.name))
                .unwrap_or_else(|| panic!("the project serves no /{}", command.name));
            assert!(
                entry.visible_in_slash,
                "/{} is advertised to clients but hidden from the local command list",
                command.name
            );
        }
    }

    #[test]
    fn a_command_written_without_its_value_reports_how_to_write_it() {
        for command in COMMANDS.iter().filter(|command| command.hint.is_some()) {
            let request = request(&format!("/{}", command.name));
            assert!(
                matches!(request, SlashRequest::Rejected(ref message) if message.starts_with("Usage: /")),
                "/{} without a value answers {request:?}",
                command.name
            );
        }
        assert_eq!(
            request("/model"),
            SlashRequest::Rejected("Usage: /model <id>".to_string())
        );
        assert_eq!(
            request("/permission plan"),
            SlashRequest::Rejected("Usage: /permission <safe|default|auto|yolo>".to_string())
        );
        assert_eq!(
            request("/resume"),
            SlashRequest::Rejected("Usage: /resume <session_id>".to_string())
        );
    }

    #[test]
    fn commands_this_frontend_does_not_dispatch_are_refused() {
        for text in [
            "/help",
            "/?",
            "/exit",
            "/quit",
            "/language en",
            "/lang",
            "/perm",
            "/think high",
            "/thoughts",
            "/tools detailed",
            "/tool-output",
            "/scrollbar on",
            "/panel off",
            "/theme rainbow",
            "/theme nonsense",
            "/fake off",
            "/tree",
            "/context",
            "/mcp",
            "/skill",
            "/child next",
            "/children first",
            "/parent",
        ] {
            assert!(
                matches!(request(text), SlashRequest::Rejected(ref message) if message.ends_with("is not available over ACP")),
                "{text:?} answers {:?}",
                request(text)
            );
        }
        assert_eq!(
            request("/theme dark"),
            SlashRequest::Rejected("/theme is not available over ACP".to_string())
        );
    }

    #[test]
    fn text_that_names_no_command_stays_prompt_content() {
        for text in [
            "explain this file",
            "exit",
            "quit",
            "/notacommand",
            "/models",
            "/modelx test",
            "/HELP",
            "/",
            "@explorer investigate",
            "",
        ] {
            assert_eq!(request(text), SlashRequest::Prompt, "{text:?}");
        }
    }
}
