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
