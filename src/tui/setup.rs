use anyhow::Result;
use crossterm::event::{Event, KeyCode, KeyEvent, KeyModifiers};
use ratatui::{
    Frame,
    layout::{Constraint, Direction, Layout, Rect},
    style::{Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Clear, Paragraph, Wrap},
};
use std::path::Path;
use std::time::Duration;

use super::{
    OwnedTerminal, components::composer, state::TuiState, surface, terminal_input::TerminalInput,
    theme::Theme,
};

const SETUP_PANEL_WIDTH: u16 = 72;
const SETUP_PANEL_HEIGHT: u16 = 22;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum SetupField {
    BaseUrl,
    ApiKey,
    Model,
}

impl SetupField {
    fn next(self) -> Self {
        match self {
            Self::BaseUrl => Self::ApiKey,
            Self::ApiKey => Self::Model,
            Self::Model => Self::BaseUrl,
        }
    }

    fn previous(self) -> Self {
        match self {
            Self::BaseUrl => Self::Model,
            Self::ApiKey => Self::BaseUrl,
            Self::Model => Self::ApiKey,
        }
    }

    fn label(self, state: &TuiState) -> String {
        state.t(match self {
            Self::BaseUrl => "setup.base_url",
            Self::ApiKey => "setup.api_key",
            Self::Model => "setup.model",
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct SetupState {
    pub base_url: String,
    pub api_key: String,
    pub model: String,
    pub active_field: SetupField,
    pub cursor: usize,
    pub error: Option<String>,
}

impl Default for SetupState {
    fn default() -> Self {
        Self {
            base_url: "https://api.openai.com/v1".into(),
            api_key: String::new(),
            model: String::new(),
            active_field: SetupField::BaseUrl,
            cursor: "https://api.openai.com/v1".len(),
            error: None,
        }
    }
}

impl SetupState {
    pub(crate) fn active_value(&self) -> &str {
        match self.active_field {
            SetupField::BaseUrl => &self.base_url,
            SetupField::ApiKey => &self.api_key,
            SetupField::Model => &self.model,
        }
    }

    fn active_value_mut(&mut self) -> &mut String {
        match self.active_field {
            SetupField::BaseUrl => &mut self.base_url,
            SetupField::ApiKey => &mut self.api_key,
            SetupField::Model => &mut self.model,
        }
    }

    fn select_field(&mut self, field: SetupField) {
        self.active_field = field;
        self.cursor = self.active_value().len();
    }

    pub(crate) fn next_field(&mut self) {
        self.select_field(self.active_field.next());
    }

    pub(crate) fn previous_field(&mut self) {
        self.select_field(self.active_field.previous());
    }

    fn insert(&mut self, text: &str) {
        let cursor = self.cursor.min(self.active_value().len());
        let mut inserted = String::new();
        for ch in text.chars() {
            if !ch.is_control() && ch != '\n' && ch != '\r' {
                inserted.push(ch);
            }
        }
        self.active_value_mut().insert_str(cursor, &inserted);
        self.cursor = cursor + inserted.len();
        self.error = None;
    }

    fn backspace(&mut self) {
        let value = self.active_value().to_string();
        let cursor = self.cursor.min(value.len());
        let Some(previous) = value[..cursor]
            .char_indices()
            .next_back()
            .map(|(index, _)| index)
        else {
            return;
        };
        self.active_value_mut().drain(previous..cursor);
        self.cursor = previous;
        self.error = None;
    }

    fn delete(&mut self) {
        let value = self.active_value().to_string();
        let cursor = self.cursor.min(value.len());
        let Some((_, ch)) = value[cursor..].char_indices().next() else {
            return;
        };
        self.active_value_mut()
            .drain(cursor..cursor + ch.len_utf8());
        self.error = None;
    }

    fn move_cursor_left(&mut self) {
        let value = self.active_value().to_string();
        self.cursor = value[..self.cursor.min(value.len())]
            .char_indices()
            .next_back()
            .map_or(0, |(index, _)| index);
    }

    fn move_cursor_right(&mut self) {
        let value = self.active_value().to_string();
        let cursor = self.cursor.min(value.len());
        self.cursor = value[cursor..]
            .char_indices()
            .nth(1)
            .map_or(value.len(), |(index, _)| cursor + index);
    }

    fn move_cursor_home(&mut self) {
        self.cursor = 0;
    }

    fn move_cursor_end(&mut self) {
        self.cursor = self.active_value().len();
    }
}

pub(crate) fn run(config_path: &Path) -> Result<bool> {
    let mut terminal = OwnedTerminal::new()?;
    let mut input = TerminalInput::new()?;
    let state = TuiState::new("pending-runtime-model", "pending runtime model", "default");
    let mut setup = SetupState::default();

    loop {
        terminal
            .terminal_mut()
            .draw(|frame| render_setup(frame, &state, &setup))?;

        let Some(event) = input.read(Duration::from_millis(100))? else {
            continue;
        };
        match event {
            Event::Key(key) if is_cancel_key(key) => return Ok(false),
            Event::Key(key) => {
                if handle_key(&mut setup, key) {
                    if setup.active_field == SetupField::Model {
                        match crate::config::initialize_config(
                            config_path,
                            &setup.base_url,
                            &setup.api_key,
                            &setup.model,
                        ) {
                            Ok(()) => return Ok(true),
                            Err(error) => setup.error = Some(format!("{error:#}")),
                        }
                    } else {
                        setup.next_field();
                    }
                }
            }
            Event::Paste(text) => setup.insert(&text),
            _ => {}
        }
    }
}

fn is_cancel_key(key: KeyEvent) -> bool {
    matches!(key.code, KeyCode::Esc)
        || (key.modifiers.contains(KeyModifiers::CONTROL) && matches!(key.code, KeyCode::Char('c')))
}

fn handle_key(setup: &mut SetupState, key: KeyEvent) -> bool {
    match key.code {
        KeyCode::Tab | KeyCode::Down => setup.next_field(),
        KeyCode::BackTab | KeyCode::Up => setup.previous_field(),
        KeyCode::Enter => return true,
        KeyCode::Backspace => setup.backspace(),
        KeyCode::Delete => setup.delete(),
        KeyCode::Left => setup.move_cursor_left(),
        KeyCode::Right => setup.move_cursor_right(),
        KeyCode::Home => setup.move_cursor_home(),
        KeyCode::End => setup.move_cursor_end(),
        KeyCode::Char('a') if key.modifiers.contains(KeyModifiers::CONTROL) => {
            setup.move_cursor_home()
        }
        KeyCode::Char('e') if key.modifiers.contains(KeyModifiers::CONTROL) => {
            setup.move_cursor_end()
        }
        KeyCode::Char(ch) if key.modifiers.is_empty() || key.modifiers == KeyModifiers::SHIFT => {
            setup.insert(&ch.to_string())
        }
        _ => {}
    }
    false
}

pub(crate) fn render_setup(frame: &mut Frame<'_>, state: &TuiState, setup: &SetupState) {
    let area = frame.area();
    if area.is_empty() {
        return;
    }

    let theme = state.theme();
    frame.render_widget(Block::new().style(theme.app_style()), area);

    let panel = centered_panel(area);
    if panel.width < 40 || panel.height < 12 {
        render_compact(frame, panel, state, setup, theme);
        return;
    }

    frame.render_widget(Clear, panel);
    let Some(shell) = composer::render_connected_prompt_shell_without_bar(
        frame,
        panel,
        theme,
        surface::SurfaceEmphasis::Notice,
        1,
    ) else {
        return;
    };

    let content = shell.content_area;
    let sections = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3),
            Constraint::Length(12),
            Constraint::Length(if setup.error.is_some() { 2 } else { 0 }),
            Constraint::Min(1),
        ])
        .split(content);
    let heading_area = sections[0];
    let fields_area = sections[1];
    let error_area = sections[2];
    let footer_area = sections[3];

    let heading = vec![
        Line::from(Span::styled(
            state.t("setup.title"),
            Style::default().fg(theme.text).add_modifier(Modifier::BOLD),
        )),
        Line::from(Span::styled(
            state.t("setup.description"),
            Style::default().fg(theme.muted_text),
        )),
    ];
    frame.render_widget(
        Paragraph::new(heading)
            .style(Style::default().bg(theme.element_bg).fg(theme.text))
            .wrap(Wrap { trim: true }),
        heading_area,
    );

    render_fields(frame, fields_area, state, setup, theme);

    if let Some(error) = &setup.error {
        frame.render_widget(
            Paragraph::new(Line::from(Span::styled(
                error.clone(),
                Style::default().fg(theme.error),
            )))
            .style(Style::default().bg(theme.element_bg).fg(theme.error))
            .wrap(Wrap { trim: true }),
            error_area,
        );
    }

    let footer = Line::from(vec![
        Span::styled("↑/↓", key_style(theme)),
        Span::styled(
            format!(" {}", state.t("setup.navigate")),
            muted_style(theme),
        ),
        Span::styled("  •  ", muted_style(theme)),
        Span::styled("Enter", key_style(theme)),
        Span::styled(
            format!(" {}", state.t("setup.continue")),
            muted_style(theme),
        ),
        Span::styled("  •  ", muted_style(theme)),
        Span::styled("Esc", key_style(theme)),
        Span::styled(format!(" {}", state.t("setup.quit")), muted_style(theme)),
    ]);
    frame.render_widget(
        Paragraph::new(footer)
            .style(Style::default().bg(theme.element_bg).fg(theme.muted_text))
            .wrap(Wrap { trim: true }),
        footer_area,
    );
}

fn render_fields(
    frame: &mut Frame<'_>,
    area: Rect,
    state: &TuiState,
    setup: &SetupState,
    theme: Theme,
) {
    let fields = [
        (SetupField::BaseUrl, setup.base_url.as_str()),
        (SetupField::ApiKey, setup.api_key.as_str()),
        (SetupField::Model, setup.model.as_str()),
    ];
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(4),
            Constraint::Length(4),
            Constraint::Length(4),
        ])
        .split(area);

    for ((field, value), field_area) in fields.into_iter().zip(chunks.iter().copied()) {
        let active = field == setup.active_field;
        let label = field.label(state);
        let value_line = render_value_line(field, value, active, setup.cursor, state, theme);
        let border_style = if active {
            Style::default()
                .fg(theme.accent)
                .add_modifier(Modifier::BOLD)
        } else {
            Style::default().fg(theme.border)
        };
        let block = Block::default()
            .borders(Borders::ALL)
            .border_style(border_style)
            .style(Style::default().bg(theme.element_bg));
        let inner = block.inner(field_area);
        frame.render_widget(block, field_area);
        if inner.is_empty() {
            continue;
        }

        let label_area = Rect::new(inner.x, inner.y, inner.width, 1);
        let value_area = Rect::new(
            inner.x,
            inner.y.saturating_add(1),
            inner.width,
            inner.height.saturating_sub(1),
        );
        frame.render_widget(
            Paragraph::new(Span::styled(label, Style::default().fg(theme.muted_text))),
            label_area,
        );
        frame.render_widget(
            Paragraph::new(value_line).wrap(Wrap { trim: false }),
            value_area,
        );
    }
}

fn render_value_line(
    field: SetupField,
    value: &str,
    active: bool,
    cursor: usize,
    state: &TuiState,
    theme: Theme,
) -> Line<'static> {
    let text = if field == SetupField::ApiKey {
        mask_secret(value)
    } else {
        value.to_string()
    };
    if !active {
        return Line::from(Span::styled(
            if text.is_empty() {
                state.t("setup.empty")
            } else {
                text
            },
            Style::default().fg(theme.muted_text),
        ));
    }

    let cursor_chars = value[..cursor.min(value.len())].chars().count();
    let split_at = text
        .char_indices()
        .nth(cursor_chars)
        .map_or(text.len(), |(index, _)| index);
    let (left, right) = text.split_at(split_at);
    Line::from(vec![
        Span::styled(left.to_string(), Style::default().fg(theme.text)),
        Span::styled("▏", key_style(theme)),
        Span::styled(right.to_string(), Style::default().fg(theme.text)),
    ])
}

fn render_compact(
    frame: &mut Frame<'_>,
    area: Rect,
    state: &TuiState,
    setup: &SetupState,
    theme: Theme,
) {
    if area.is_empty() {
        return;
    }
    let active_value = if setup.active_field == SetupField::ApiKey {
        mask_secret(setup.active_value())
    } else {
        setup.active_value().to_string()
    };
    let active_value = if active_value.is_empty() {
        state.t("setup.empty")
    } else {
        active_value
    };
    let mut lines = vec![
        Line::from(Span::styled(
            state.t("setup.title"),
            Style::default().fg(theme.text).add_modifier(Modifier::BOLD),
        )),
        Line::from(Span::styled(
            state.t("setup.resize"),
            Style::default().fg(theme.muted_text),
        )),
        Line::default(),
        Line::from(Span::styled(
            format!("{}: {active_value}", setup.active_field.label(state)),
            Style::default().fg(theme.text),
        )),
    ];
    if let Some(error) = &setup.error {
        lines.push(Line::from(Span::styled(
            error.clone(),
            Style::default().fg(theme.error),
        )));
    }
    frame.render_widget(
        Paragraph::new(lines)
            .style(theme.elevated_style())
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .border_style(theme.border),
            )
            .wrap(Wrap { trim: true }),
        area,
    );
}

fn centered_panel(area: Rect) -> Rect {
    let width = SETUP_PANEL_WIDTH.min(area.width.saturating_sub(2)).max(1);
    let height = SETUP_PANEL_HEIGHT.min(area.height.saturating_sub(2)).max(1);
    Rect::new(
        area.x + area.width.saturating_sub(width) / 2,
        area.y + area.height.saturating_sub(height) / 2,
        width,
        height,
    )
}

fn mask_secret(value: &str) -> String {
    "•".repeat(value.chars().count())
}

fn key_style(theme: Theme) -> Style {
    Style::default()
        .fg(theme.accent)
        .add_modifier(Modifier::BOLD)
}

fn muted_style(theme: Theme) -> Style {
    Style::default().fg(theme.muted_text)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ratatui::{Terminal, backend::TestBackend};

    fn draw(setup: &SetupState, width: u16, height: u16) -> String {
        let mut state = TuiState::new("pending", "pending", "default");
        state.set_language(Some(crate::tui::i18n::Language::En));
        let backend = TestBackend::new(width, height);
        let mut terminal = Terminal::new(backend).expect("terminal");
        terminal
            .draw(|frame| render_setup(frame, &state, setup))
            .expect("render");
        terminal
            .backend()
            .buffer()
            .content()
            .iter()
            .map(|cell| cell.symbol())
            .collect()
    }

    #[test]
    fn setup_screen_renders_fields_and_masks_api_key() {
        let setup = SetupState {
            api_key: "secret-key".into(),
            model: "gpt-5.5".into(),
            ..SetupState::default()
        };
        let rendered = draw(&setup, 100, 30);

        assert!(rendered.contains("Set up letcode"), "{rendered}");
        assert!(rendered.contains("Base URL"), "{rendered}");
        assert!(rendered.contains("API key"), "{rendered}");
        assert!(rendered.contains("gpt-5.5"), "{rendered}");
        assert!(rendered.contains("••••••••••"), "{rendered}");
        assert!(!rendered.contains(surface::ACCENT_BAR_GLYPH), "{rendered}");
        assert!(!rendered.contains("secret-key"), "{rendered}");
    }

    #[test]
    fn setup_screen_has_a_compact_fallback() {
        let setup = SetupState {
            active_field: SetupField::ApiKey,
            api_key: "secret-key".into(),
            ..SetupState::default()
        };
        let rendered = draw(&setup, 32, 10);
        assert!(rendered.contains("Set up letcode"), "{rendered}");
        assert!(rendered.contains("Resize"), "{rendered}");
        assert!(rendered.contains("••••••••••"), "{rendered}");
        assert!(!rendered.contains(surface::ACCENT_BAR_GLYPH), "{rendered}");
        assert!(!rendered.contains("secret-key"), "{rendered}");
    }
}
