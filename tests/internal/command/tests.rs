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
fn parses_config_command() {
    assert_eq!(parse_command("/config"), Ok(CommandIntent::ConfigShow));
    assert_eq!(
        parse_command("/config extra"),
        Err(CommandParseError::new("Usage: /config"))
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
        parse_command("/panel"),
        Ok(CommandIntent::PanelSet(PanelMode::Toggle))
    );
    assert_eq!(
        parse_command("/panel on"),
        Ok(CommandIntent::PanelSet(PanelMode::Visible))
    );
    assert_eq!(
        parse_command("/panel off"),
        Ok(CommandIntent::PanelSet(PanelMode::Hidden))
    );
    assert_eq!(
        parse_command("/panel maybe"),
        Err(CommandParseError::with_args("parse.unknown_panel", []))
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
        parse_command("/theme wireframe"),
        Ok(CommandIntent::Theme(ThemeCommand::Set("wireframe".into())))
    );
    assert_eq!(
        parse_command("/theme sunset"),
        Ok(CommandIntent::Theme(ThemeCommand::Set("sunset".into())))
    );
    assert_eq!(
        parse_command("/fake"),
        Ok(CommandIntent::Fake(FakeCommand::Show))
    );
    assert_eq!(
        parse_command("/fake auto"),
        Ok(CommandIntent::Fake(FakeCommand::Set(Some(
            crate::fake::FakeClient::Auto
        ))))
    );
    assert_eq!(
        parse_command("/fake codex"),
        Ok(CommandIntent::Fake(FakeCommand::Set(Some(
            crate::fake::FakeClient::Codex
        ))))
    );
    assert_eq!(
        parse_command("/fake anthropic"),
        Ok(CommandIntent::Fake(FakeCommand::Set(Some(
            crate::fake::FakeClient::Anthropic
        ))))
    );
    assert_eq!(
        parse_command("/fake off"),
        Ok(CommandIntent::Fake(FakeCommand::Set(None)))
    );
    assert_eq!(
        parse_command("/fake maybe"),
        Err(CommandParseError::with_args(
            "parse.unknown_fake",
            [("value", "maybe")]
        ))
    );
    for input in ["@", "@fixer", "@explorer inspect src/", "@unknown foo"] {
        assert_eq!(
            parse_command(input),
            Ok(CommandIntent::Prompt(input.into()))
        );
    }
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
        parse_command("/child"),
        Ok(CommandIntent::Child(ChildNavigation::Toggle))
    );
    assert_eq!(
        parse_command("/child first"),
        Ok(CommandIntent::Child(ChildNavigation::First))
    );
    assert_eq!(
        parse_command("/child next"),
        Ok(CommandIntent::Child(ChildNavigation::Next))
    );
    assert_eq!(
        parse_command("/child prev"),
        Ok(CommandIntent::Child(ChildNavigation::Prev))
    );
    assert_eq!(
        parse_command("/child previous"),
        Ok(CommandIntent::Child(ChildNavigation::Prev))
    );
    assert_eq!(
        parse_command("/children"),
        Err(CommandParseError::unknown_command("/children"))
    );
    assert_eq!(
        parse_command("/parent"),
        Err(CommandParseError::unknown_command("/parent"))
    );
    assert_eq!(
        parse_command("/child Session-Id"),
        Ok(CommandIntent::Child(ChildNavigation::Session(
            "Session-Id".into()
        )))
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
        parse_command("/thoughts 3"),
        Ok(CommandIntent::ThoughtsSet(ThoughtsDisplayMode::Scroll))
    );
    assert_eq!(
        parse_command("/thoughts full"),
        Ok(CommandIntent::ThoughtsSet(ThoughtsDisplayMode::Full))
    );
    assert_eq!(
        parse_command("/thoughts verbose"),
        Err(CommandParseError::with_args("parse.unknown_thoughts", []))
    );
    assert_eq!(parse_command("/tools"), Ok(CommandIntent::ToolsShow));
    assert_eq!(
        parse_command("/tools 1"),
        Ok(CommandIntent::ToolsSet(ToolsDisplayMode::Compact))
    );
    assert_eq!(
        parse_command("/tools detailed"),
        Ok(CommandIntent::ToolsSet(ToolsDisplayMode::Detailed))
    );
    assert_eq!(
        parse_command("/tool-output off"),
        Ok(CommandIntent::ToolsSet(ToolsDisplayMode::Compact))
    );
    assert_eq!(
        parse_command("/tools verbose"),
        Err(CommandParseError::with_args("parse.unknown_tools", []))
    );
}
