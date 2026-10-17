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
#[path = "../../tests/internal/acp/slash/tests.rs"]
mod tests;
