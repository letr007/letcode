use super::*;
use crate::tui::{
    help::{HelpAction, HelpState},
    i18n::Language,
    state::{DialogKind, DialogState},
};
use ratatui::{Terminal, backend::TestBackend};

fn state(language: Language) -> TuiState {
    let mut state = TuiState::default();
    state.set_language(Some(language));
    let mut dialog = DialogState::new(DialogKind::Help, "", None, Vec::new());
    dialog.help = Some(HelpState::new(language));
    state.open_dialog(dialog);
    state
}

fn draw_buffer(
    state: &mut TuiState,
    width: u16,
    height: u16,
    theme: Theme,
) -> ratatui::buffer::Buffer {
    let mut terminal = Terminal::new(TestBackend::new(width, height)).expect("terminal");
    terminal
        .draw(|frame| render_help_reader(frame, state, frame.area(), theme))
        .expect("draw");
    terminal.backend().buffer().clone()
}

fn draw(state: &mut TuiState, width: u16, height: u16) -> String {
    let buffer = draw_buffer(state, width, height, Theme::dark());
    let mut text = String::new();
    for row in buffer.content().chunks(usize::from(width)) {
        let mut skip = 0;
        for cell in row {
            if skip > 0 {
                skip -= 1;
                continue;
            }
            text.push_str(cell.symbol());
            skip = display_width(cell.symbol()).saturating_sub(1);
        }
        text.push('\n');
    }
    text
}

#[test]
fn reader_shell_follows_picker_theme_and_header_styles() {
    let area = Rect::new(0, 0, 120, 30);
    for theme in [
        Theme::dark(),
        Theme::plain_for(None),
        Theme::glass(),
        Theme::wireframe(),
    ] {
        let mut state = state(Language::En);
        let buffer = draw_buffer(&mut state, area.width, area.height, theme);
        let panel = reader_area(area);
        let inner = panel.inner(if theme.card_frame {
            Margin::new(2, 1)
        } else {
            Margin::new(3, 2)
        });
        let corner = &buffer[(panel.x, panel.y)];
        assert_eq!(corner.symbol(), if theme.card_frame { "┌" } else { " " });
        assert_eq!(
            corner.bg,
            if theme.card_frame {
                theme.root_bg
            } else {
                theme.elevated_bg
            }
        );
        let title = &buffer[(inner.x, inner.y)];
        assert_eq!(title.fg, theme.text);
        assert_eq!(title.bg, theme.elevated_bg);
        assert!(title.modifier.contains(Modifier::BOLD));
        let escape = &buffer[(inner.right() - 3, inner.y)];
        assert_eq!(escape.fg, theme.muted_text);
        assert_eq!(escape.bg, theme.elevated_bg);
        let footer_y = inner.bottom() - 1;
        let key = &buffer[(inner.x, footer_y)];
        assert_eq!(key.fg, theme.accent);
        assert!(key.modifier.contains(Modifier::BOLD));
        let label = &buffer[(inner.x + 4, footer_y)];
        assert_eq!(label.fg, theme.muted_text);
        assert_eq!(label.bg, theme.elevated_bg);
    }
}

#[test]
fn contents_highlight_fills_the_focused_row_like_picker_items() {
    let theme = Theme::dark();
    let area = Rect::new(0, 0, 40, 10);
    let mut help = HelpState::new(Language::En);
    help.apply(HelpAction::NextChapter);
    for focused in [true, false] {
        help.contents_focused = focused;
        let mut terminal =
            Terminal::new(TestBackend::new(area.width, area.height)).expect("terminal");
        terminal
            .draw(|frame| render_contents(frame, &help, area, theme))
            .expect("draw");
        let buffer = terminal.backend().buffer();
        for column in area.x..area.right() {
            let selected = &buffer[(column, help.selected as u16)];
            assert_eq!(
                selected.bg,
                if focused {
                    theme.element_bg
                } else {
                    theme.elevated_bg
                }
            );
            assert_eq!(selected.fg, theme.text);
            assert!(selected.modifier.contains(Modifier::BOLD));
            assert_eq!(buffer[(column, area.y)].bg, theme.elevated_bg);
        }
    }
}

#[test]
fn wide_reader_shows_contents_and_rendered_markdown() {
    let mut state = state(Language::En);
    let rendered = draw(&mut state, 120, 30);
    assert!(rendered.contains("Quick start"));
    assert!(rendered.contains("Command reference"));
    assert!(rendered.contains("Experts and child sessions"));
    assert!(rendered.contains("Esc"));
    assert!(rendered.contains("esc"));
    assert!(!rendered.contains("## Quick start"));
}

#[test]
fn narrow_reader_switches_between_contents_and_document() {
    let mut state = state(Language::En);
    let document = draw(&mut state, 70, 24);
    assert!(!document.contains("Command reference"));
    state
        .dialog_mut()
        .unwrap()
        .help
        .as_mut()
        .unwrap()
        .apply(HelpAction::FocusContents);
    let contents = draw(&mut state, 70, 24);
    assert!(contents.contains("Command reference"));
    state
        .dialog_mut()
        .unwrap()
        .help
        .as_mut()
        .unwrap()
        .apply(HelpAction::NextChapter);
    state
        .dialog_mut()
        .unwrap()
        .help
        .as_mut()
        .unwrap()
        .apply(HelpAction::FocusDocument);
    let commands = draw(&mut state, 70, 24);
    assert!(commands.contains("/help"));
    assert!(!commands.contains("{{commands}}"));
}

#[test]
fn reader_handles_small_terminals_and_chinese_content() {
    let mut state = state(Language::ZhCn);
    for (width, height) in [(1, 1), (8, 4), (30, 12), (80, 24), (120, 40)] {
        draw(&mut state, width, height);
        let help = state.dialog().unwrap().help.as_ref().unwrap();
        assert!(help.scroll <= help.scroll_max);
    }
    let rendered = draw(&mut state, 80, 24);
    assert!(rendered.contains("快速开始"));
    assert!(rendered.contains("关闭"));
}
