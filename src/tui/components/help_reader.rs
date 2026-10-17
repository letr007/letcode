use ratatui::{
    Frame,
    layout::{Alignment, Constraint, Direction, Layout, Margin, Rect},
    style::{Color, Modifier},
    text::{Line, Span},
    widgets::{Block, Clear, Paragraph},
};

use super::picker;

use crate::tui::{
    assistant_fade, help::HelpState, measure::display_width, scrollbar::ScrollbarGeometry,
    state::TuiState, theme::Theme, transcript_ratatui::line_to_ratatui,
};

pub fn render_help_reader(frame: &mut Frame<'_>, state: &mut TuiState, area: Rect, theme: Theme) {
    if area.is_empty() {
        return;
    }
    let language = state.language();
    let translator = state.translator();
    let terminal_bg = state.terminal_background();
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
        let backdrop = assistant_fade::fade_backdrop(document_theme, terminal_bg);
        render_document(frame, help, text_area, document, document_theme, backdrop);
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
    backdrop: Color,
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
            let (symbol, style) = geometry.cell(usize::from(row), theme, backdrop);
            if let Some(cell) = frame.buffer_mut().cell_mut((column, area.y + row)) {
                cell.set_char(symbol).set_style(style);
            }
        }
    }
}

#[cfg(test)]
#[path = "../../../tests/internal/tui/components/help_reader/tests.rs"]
mod tests;
