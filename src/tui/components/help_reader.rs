use ratatui::{
    Frame,
    layout::{Alignment, Constraint, Direction, Layout, Margin, Rect},
    style::Modifier,
    text::{Line, Span},
    widgets::{Block, Clear, Paragraph},
};

use super::picker;

use crate::tui::{
    help::HelpState, measure::display_width, scrollbar::ScrollbarGeometry, state::TuiState,
    theme::Theme, transcript_ratatui::line_to_ratatui,
};

pub fn render_help_reader(frame: &mut Frame<'_>, state: &mut TuiState, area: Rect, theme: Theme) {
    if area.is_empty() {
        return;
    }
    let language = state.language();
    let translator = state.translator();
    let Some(help) = state.dialog_mut().and_then(|dialog| dialog.help.as_mut()) else {
        return;
    };

    let panel = reader_area(area);
    frame.render_widget(Clear, panel);
    frame.render_widget(Block::default().style(theme.elevated_style()), panel);
    picker::render_three_sided_frame(frame, panel, theme);
    let inner = panel.inner(if theme.card_frame {
        Margin::new(2, 1)
    } else {
        Margin::new(3, 2)
    });
    if inner.is_empty() {
        return;
    }
    if inner.height < 6 {
        picker::render_header(
            frame,
            Rect::new(inner.x, inner.y, inner.width, 1),
            theme,
            &translator.t("help.title"),
        );
        return;
    }

    let areas = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(1),
            Constraint::Length(2),
            Constraint::Min(0),
            Constraint::Length(1),
            Constraint::Length(1),
        ])
        .split(inner);
    let header = areas[0];
    let body = areas[2];
    let footer = areas[4];
    let wide = body.width >= 88;
    let (contents, document) = if wide {
        let areas = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([
                Constraint::Length(32),
                Constraint::Length(3),
                Constraint::Min(0),
            ])
            .split(body);
        let separator = Rect::new(areas[1].x + 1, areas[1].y, 1, areas[1].height);
        frame.render_widget(
            Paragraph::new(vec![Line::from("│"); separator.height as usize])
                .style(picker::muted_style(theme)),
            separator,
        );
        (areas[0], areas[2])
    } else {
        (body, body)
    };
    let text_area = Rect::new(
        document.x,
        document.y,
        document.width.saturating_sub(1),
        document.height,
    );
    let document_theme = Theme {
        root_bg: theme.elevated_bg,
        ..theme
    };
    help.prepare(
        usize::from(text_area.width),
        usize::from(text_area.height),
        document_theme,
        language,
    );

    let title = translator.t("help.title");
    picker::render_header(frame, header, theme, &title);
    let title_width = display_width(&title) as u16;
    let chapter_width = header.width.saturating_sub(title_width.saturating_add(6));
    if chapter_width > 0 {
        frame.render_widget(
            Paragraph::new(format!("·  {}", help.chapters[help.selected].title))
                .style(picker::muted_style(theme)),
            Rect::new(header.x + title_width + 2, header.y, chapter_width, 1),
        );
    }
    if wide || help.contents_focused {
        render_contents(frame, help, contents, theme);
    }
    if wide || !help.contents_focused {
        render_document(frame, help, text_area, document, document_theme);
    }

    let progress = format!(
        "{}/{} · {}%",
        help.selected + 1,
        help.chapters.len(),
        help.scroll
            .saturating_add(help.viewport_rows)
            .min(help.line_count())
            * 100
            / help.line_count().max(1),
    );
    let progress_width = display_width(&progress) as u16 + 1;
    let hint_key = if footer.width < 80 {
        "help.hint_compact"
    } else if help.contents_focused {
        "help.hint_contents"
    } else {
        "help.hint_document"
    };
    let hint_width = footer.width.saturating_sub(progress_width);
    let hint = translator.t(hint_key);
    let mut spans = Vec::new();
    for (index, shortcut) in hint.split(" · ").enumerate() {
        if index > 0 {
            spans.push(Span::styled("  ·  ", picker::muted_style(theme)));
        }
        let (key, label) = shortcut.split_once(' ').unwrap_or((shortcut, ""));
        spans.push(Span::styled(key.to_string(), picker::accent_style(theme)));
        spans.push(Span::styled(
            format!(" {label}"),
            picker::muted_style(theme),
        ));
    }
    frame.render_widget(
        Paragraph::new(Line::from(spans)).style(theme.elevated_style()),
        Rect::new(footer.x, footer.y, hint_width, footer.height),
    );
    if footer.width >= progress_width {
        frame.render_widget(
            Paragraph::new(progress)
                .alignment(Alignment::Right)
                .style(theme.elevated_style().fg(theme.dim_text)),
            Rect::new(
                footer.x + hint_width,
                footer.y,
                progress_width,
                footer.height,
            ),
        );
    }
}

fn reader_area(area: Rect) -> Rect {
    let width = area.width.saturating_sub(2).clamp(1, 120);
    let height = area.height.saturating_sub(2).max(1);
    Rect::new(
        area.x + area.width.saturating_sub(width) / 2,
        area.y + area.height.saturating_sub(height) / 2,
        width,
        height,
    )
}

fn render_contents(frame: &mut Frame<'_>, help: &HelpState, area: Rect, theme: Theme) {
    let start = help
        .selected
        .saturating_sub(usize::from(area.height).saturating_sub(1));
    for (row, (index, chapter)) in help
        .chapters
        .iter()
        .enumerate()
        .skip(start)
        .take(usize::from(area.height))
        .enumerate()
    {
        let selected = index == help.selected;
        let style = if selected && help.contents_focused {
            picker::selected_item_style(theme)
        } else if selected {
            picker::item_style(theme).add_modifier(Modifier::BOLD)
        } else {
            picker::item_style(theme)
        };
        let row_area = Rect::new(area.x, area.y + row as u16, area.width, 1);
        frame.render_widget(Block::default().style(style), row_area);
        frame.render_widget(
            Paragraph::new(format!(
                "{}{}",
                if selected { "● " } else { "  " },
                chapter.title
            ))
            .style(style),
            row_area.inner(Margin::new(1, 0)),
        );
    }
}

fn render_document(
    frame: &mut Frame<'_>,
    help: &HelpState,
    text_area: Rect,
    area: Rect,
    theme: Theme,
) {
    let lines = help
        .visible_lines()
        .iter()
        .map(line_to_ratatui)
        .collect::<Vec<_>>();
    frame.render_widget(
        Paragraph::new(lines).style(theme.elevated_style()),
        text_area,
    );
    if let Some(geometry) = ScrollbarGeometry::new(
        usize::from(area.height),
        help.line_count(),
        help.viewport_rows,
        help.scroll,
    ) {
        let column = area.right().saturating_sub(1);
        for row in 0..area.height {
            let (symbol, style) = geometry.cell(usize::from(row), theme, theme.elevated_bg);
            if let Some(cell) = frame.buffer_mut().cell_mut((column, area.y + row)) {
                cell.set_char(symbol).set_style(style);
            }
        }
    }
}

#[cfg(test)]
mod tests {
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
}
