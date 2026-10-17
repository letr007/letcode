use ratatui::{
    Frame,
    layout::{Constraint, Direction, Layout, Margin, Rect},
    style::{Modifier, Style},
    text::{Line, Span, Text},
    widgets::{Block, Clear, Paragraph, Wrap},
};
use unicode_segmentation::UnicodeSegmentation;

use crate::tui::{
    components::tool_card::truncate_display_width,
    measure::{display_width, wrap_text_to_width},
    state::{
        ConfigEditState, ConfigFieldRef, DialogItem, DialogKind, DialogState, SessionPickerScope,
        TuiState,
    },
    theme::Theme,
};

const PICKER_MIN_WIDTH: u16 = 64;
const PICKER_MAX_WIDTH: u16 = 96;
const MAX_DESCRIPTION_ROWS: u16 = 4;
const PICKER_MIN_HEIGHT: u16 = 18;
const PICKER_MAX_HEIGHT: u16 = 28;
// Right-aligned details (session timestamps, status labels) never squeeze the row's
// label below this width; longer details are truncated instead.
const MIN_LEFT_LABEL_WIDTH: u16 = 18;
// Expert rows wrap their model list onto continuation rows instead of truncating it.
const AGENT_MODEL_INDENT: u16 = 4;
const MODEL_SEPARATOR: &str = " · ";

pub fn render_picker(
    frame: &mut Frame<'_>,
    state: &mut TuiState,
    area: Rect,
    theme: Theme,
    dialog: &DialogState,
) {
    let picker_area = centered_picker_area(area);
    frame.render_widget(Clear, picker_area);
    frame.render_widget(Block::default().style(theme.elevated_style()), picker_area);

    let inner = if theme.card_frame {
        picker_area.inner(Margin::new(2, 1))
    } else {
        let vertical_margin = if dialog.kind == DialogKind::ConfigEditor && picker_area.height < 9 {
            1
        } else {
            2
        };
        picker_area.inner(Margin::new(3, vertical_margin))
    };
    if inner.is_empty() {
        return;
    }

    if dialog.kind == DialogKind::ConfigEditor {
        render_config_editor(frame, state, inner, theme, dialog);
        render_three_sided_frame(frame, picker_area, theme);
        return;
    }

    render_header(
        frame,
        Rect::new(inner.x, inner.y, inner.width, 1),
        theme,
        &dialog.title,
    );

    let footer_y = inner.bottom().saturating_sub(1);
    let body_y = if dialog.kind == DialogKind::McpToolsPicker {
        let description_y = inner.y.saturating_add(2);
        if let Some(description) = mcp_tools_description(dialog)
            && description_y < footer_y
        {
            render_description(
                frame,
                Rect::new(inner.x, description_y, inner.width, 1),
                theme,
                description,
            );
        }

        let search_y = description_y.saturating_add(2);
        if search_y < footer_y {
            render_search(
                frame,
                Rect::new(inner.x, search_y, inner.width, 1),
                theme,
                state,
                dialog,
            );
        }

        search_y.saturating_add(2)
    } else if picker_has_search(dialog) {
        let search_y = inner.y.saturating_add(3);
        if search_y < inner.bottom() {
            render_search(
                frame,
                Rect::new(inner.x, search_y, inner.width, 1),
                theme,
                state,
                dialog,
            );
        }

        search_y.saturating_add(2)
    } else if let Some(description) = dialog.description.as_deref() {
        let description_y = inner.y.saturating_add(2);
        if description_y < footer_y {
            render_description(
                frame,
                Rect::new(inner.x, description_y, inner.width, 1),
                theme,
                description,
            );
        }

        description_y.saturating_add(2)
    } else {
        inner.y.saturating_add(3)
    };

    let body_height = footer_y.saturating_sub(body_y).saturating_sub(1);
    let body_area = Rect::new(inner.x, body_y, inner.width, body_height);
    if body_height > 0 {
        if dialog.kind == DialogKind::ContextPicker {
            render_context_picker_body(frame, body_area, theme, state, dialog);
        } else {
            render_picker_body(frame, body_area, theme, state, dialog);
        }
    }

    if footer_y > inner.y {
        let footer_area = Rect::new(inner.x, footer_y, inner.width, 1);
        if dialog.kind == DialogKind::ContextPicker {
            render_context_picker_footer(frame, footer_area, theme, dialog);
        } else if matches!(
            dialog.kind,
            DialogKind::McpPicker | DialogKind::McpToolsPicker
        ) {
            render_mcp_picker_footer(frame, footer_area, theme, state, dialog.kind.clone());
        } else if matches!(dialog.kind, DialogKind::ExpertModelPicker(_)) {
            render_expert_model_picker_footer(frame, footer_area, theme, state);
        } else if dialog.kind == DialogKind::SessionPicker {
            render_session_picker_footer(frame, footer_area, theme, state, dialog);
        } else {
            frame.render_widget(Block::default().style(theme.elevated_style()), footer_area);
        }
    }

    render_three_sided_frame(frame, picker_area, theme);
}

fn render_config_editor(
    frame: &mut Frame<'_>,
    state: &mut TuiState,
    inner: Rect,
    theme: Theme,
    dialog: &DialogState,
) {
    let confirming = dialog.config_close_selected.is_some();
    let title = if confirming {
        state.t("config.unsaved")
    } else {
        dialog.title.clone()
    };
    render_header(
        frame,
        Rect::new(inner.x, inner.y, inner.width, 1),
        theme,
        &title,
    );
    let footer_y = inner.bottom().saturating_sub(1);
    let gap = u16::from(inner.height >= 12);
    let mut y = inner.y.saturating_add(1).saturating_add(gap);
    let expanded = dialog.config_expanded.is_some();
    let search = !confirming && !expanded && state.config_edit.is_none();
    let body_rows = if confirming {
        3
    } else if expanded {
        1
    } else {
        2
    };
    let reserved = body_rows + u16::from(search) + gap * 2;
    let description = dialog.config_error.as_deref().or(if confirming {
        None
    } else {
        dialog.description.as_deref()
    });
    if let Some(description) = description {
        let lines = wrap_text_to_width(description, inner.width as usize);
        let height = (lines.len().min(MAX_DESCRIPTION_ROWS as usize) as u16)
            .min(footer_y.saturating_sub(y).saturating_sub(reserved));
        if height > 0 {
            let style = if dialog.config_error.is_some() {
                theme.error_style()
            } else {
                theme.elevated_style()
            };
            frame.render_widget(
                Paragraph::new(lines.into_iter().map(Line::from).collect::<Vec<_>>()).style(style),
                Rect::new(inner.x, y, inner.width, height),
            );
            y = y.saturating_add(height).saturating_add(gap);
        }
    }
    if search && footer_y.saturating_sub(y) > body_rows {
        render_search(
            frame,
            Rect::new(inner.x, y, inner.width, 1),
            theme,
            state,
            dialog,
        );
        y = y.saturating_add(1).saturating_add(gap);
    }
    let body = Rect::new(inner.x, y, inner.width, footer_y.saturating_sub(y));
    if let Some(selected) = dialog.config_close_selected {
        let rows = body.height as usize;
        let start = selected
            .saturating_add(1)
            .saturating_sub(rows)
            .min(3usize.saturating_sub(rows));
        for (index, key) in [
            "config.save_and_close",
            "config.discard_and_close",
            "config.continue_editing",
        ]
        .into_iter()
        .enumerate()
        .skip(start)
        .take(rows)
        {
            let style = if index == selected {
                selected_item_style(theme)
            } else {
                item_style(theme)
            };
            let marker = if index == selected { "● " } else { "  " };
            let row = Rect::new(body.x, body.y + (index - start) as u16, body.width, 1);
            frame.render_widget(Block::default().style(style), row);
            frame.render_widget(
                Paragraph::new(format!("{marker}{}", state.t(key))).style(style),
                row,
            );
        }
    } else if expanded {
        render_config_split_body(frame, body, theme, state, dialog);
    } else {
        render_picker_body(frame, body, theme, state, dialog);
    }
    if footer_y > inner.y {
        render_config_footer(
            frame,
            Rect::new(inner.x, footer_y, inner.width, 1),
            theme,
            state,
        );
    }
}

pub(super) fn render_three_sided_frame(frame: &mut Frame<'_>, area: Rect, theme: Theme) {
    if !theme.card_frame || area.width < 3 || area.height < 3 {
        return;
    }

    let style = Style::default().fg(theme.border).bg(theme.root_bg);
    let horizontal = "─".repeat(area.width.saturating_sub(2) as usize);
    frame.render_widget(
        Paragraph::new(Line::from(Span::styled(format!("┌{horizontal}┐"), style))),
        Rect::new(area.x, area.y, area.width, 1),
    );

    let side_lines = (0..area.height.saturating_sub(2))
        .map(|_| Line::from(Span::styled("│", style)))
        .collect::<Vec<_>>();
    frame.render_widget(
        Paragraph::new(Text::from(side_lines.clone())),
        Rect::new(area.x, area.y + 1, 1, area.height.saturating_sub(2)),
    );
    frame.render_widget(
        Paragraph::new(Text::from(side_lines)),
        Rect::new(
            area.right().saturating_sub(1),
            area.y + 1,
            1,
            area.height.saturating_sub(2),
        ),
    );

    frame.render_widget(
        Paragraph::new(Line::from(Span::styled(format!("└{horizontal}┘"), style))),
        Rect::new(area.x, area.bottom().saturating_sub(1), area.width, 1),
    );
}

fn mcp_tools_description(dialog: &DialogState) -> Option<&str> {
    let description = dialog.description.as_deref()?;
    if description.starts_with("Offline") {
        Some("Offline")
    } else if description.starts_with("Online") || description.starts_with("Disabled") {
        Some(description)
    } else {
        None
    }
}

fn picker_has_search(dialog: &DialogState) -> bool {
    dialog.kind.is_searchable() || dialog.kind == DialogKind::AgentPicker
}

fn render_description(frame: &mut Frame<'_>, area: Rect, theme: Theme, description: &str) {
    frame.render_widget(
        Paragraph::new(description.to_string())
            .style(theme.elevated_style())
            .wrap(Wrap { trim: true }),
        area,
    );
}

pub(super) fn render_header(frame: &mut Frame<'_>, area: Rect, theme: Theme, title: &str) {
    frame.render_widget(
        Paragraph::new(Line::from(Span::styled(
            title.to_string(),
            Style::default()
                .fg(theme.text)
                .bg(theme.elevated_bg)
                .add_modifier(Modifier::BOLD),
        )))
        .style(theme.elevated_style()),
        area,
    );

    let esc_width = 3.min(area.width);
    let esc_area = Rect::new(
        area.right().saturating_sub(esc_width),
        area.y,
        esc_width,
        area.height,
    );
    frame.render_widget(
        Paragraph::new(Line::from(Span::styled("esc", muted_style(theme))))
            .style(theme.elevated_style()),
        esc_area,
    );
}

fn render_search(
    frame: &mut Frame<'_>,
    area: Rect,
    theme: Theme,
    state: &TuiState,
    dialog: &DialogState,
) {
    let text = if dialog.query.is_empty() {
        Span::styled(state.t("ui.search"), muted_style(theme))
    } else {
        Span::styled(dialog.query.clone(), item_style(theme))
    };

    frame.render_widget(
        Paragraph::new(Line::from(text)).style(theme.elevated_style()),
        area,
    );
}

fn render_picker_body(
    frame: &mut Frame<'_>,
    area: Rect,
    theme: Theme,
    state: &TuiState,
    dialog: &DialogState,
) {
    let mut y = area.y;
    let rows = area.bottom().saturating_sub(y) as usize;
    if rows == 0 {
        return;
    }

    let mut rendered_any = false;
    for entry in visible_picker_entries(dialog, rows, area.width) {
        if y >= area.bottom() {
            break;
        }
        let entry_rows = entry.rows();
        match entry {
            PickerEntry::Heading(section) => {
                render_section_heading(
                    frame,
                    Rect::new(area.x, y, area.width, 1),
                    theme,
                    section,
                    theme.accent,
                );
            }
            PickerEntry::Item { index, item, .. } => {
                rendered_any = true;
                let row = Rect::new(
                    area.x,
                    y,
                    area.width,
                    entry_rows.min(area.bottom().saturating_sub(y)),
                );
                let selected = index == dialog.selected;
                match dialog.kind {
                    DialogKind::ModelPicker => render_model_row(
                        frame,
                        row,
                        theme,
                        item,
                        selected,
                        item.id == state.model_id,
                    ),
                    DialogKind::AgentPicker => render_agent_row(frame, row, theme, item, selected),
                    DialogKind::ExpertModelPicker(_) => {
                        render_model_row(frame, row, theme, item, selected, item.checked)
                    }
                    DialogKind::SessionPicker
                    | DialogKind::ChildPicker
                    | DialogKind::HistoryTree
                    | DialogKind::ContextPicker
                    | DialogKind::SkillPicker
                    | DialogKind::McpToolsPicker => {
                        render_session_row(frame, row, theme, item, selected, None)
                    }
                    DialogKind::McpPicker => render_session_row(
                        frame,
                        row,
                        theme,
                        item,
                        selected,
                        Some(mcp_status_color(item, theme, state.language())),
                    ),
                    DialogKind::PermissionPicker => render_permission_row(
                        frame,
                        row,
                        theme,
                        item,
                        selected,
                        item.id == state.permission_mode_label,
                    ),
                    DialogKind::ThemePicker => render_permission_row(
                        frame,
                        row,
                        theme,
                        item,
                        selected,
                        item.id == state.theme_id,
                    ),
                    DialogKind::FakePicker => render_permission_row(
                        frame,
                        row,
                        theme,
                        item,
                        selected,
                        item.id
                            == state
                                .fake_client
                                .map(|client| client.as_str())
                                .unwrap_or("off"),
                    ),
                    DialogKind::LanguagePicker => render_permission_row(
                        frame,
                        row,
                        theme,
                        item,
                        selected,
                        item.id == state.language().id(),
                    ),
                    DialogKind::ReasoningPicker => render_reasoning_row(
                        frame,
                        row,
                        theme,
                        item,
                        selected,
                        reasoning_item_is_current(
                            state.reasoning_effort_label.as_deref(),
                            &item.id,
                        ),
                    ),
                    DialogKind::ThoughtsPicker => render_reasoning_row(
                        frame,
                        row,
                        theme,
                        item,
                        selected,
                        item.id == state.thoughts_display.as_str(),
                    ),
                    DialogKind::ToolsPicker => render_reasoning_row(
                        frame,
                        row,
                        theme,
                        item,
                        selected,
                        item.id == state.tools_display.as_str(),
                    ),
                    DialogKind::ContextDetail => {
                        render_session_row(frame, row, theme, item, selected, None)
                    }
                    DialogKind::Help => unreachable!("help uses the document reader"),
                    DialogKind::SessionDeleteConfirm => {
                        unreachable!("session delete confirmation uses the plain dialog")
                    }
                    DialogKind::ConfigEditor => render_config_row(
                        frame,
                        row,
                        theme,
                        item,
                        selected,
                        dialog.config_fields.get(index),
                        state.config_edit.as_ref(),
                    ),
                }
            }
        }
        y = y.saturating_add(entry_rows);
    }

    if !rendered_any && y < area.bottom() {
        let empty_label = match dialog.kind {
            DialogKind::LanguagePicker => state.t("language.no_match"),
            DialogKind::SessionPicker => match dialog.session_scope {
                SessionPickerScope::Workspace => state.t("dialog.no_workspace_sessions"),
                SessionPickerScope::All => state.t("dialog.no_sessions"),
            },
            DialogKind::HistoryTree => state.t("dialog.no_history"),
            DialogKind::ContextPicker => state.t("dialog.no_context"),
            DialogKind::McpPicker => dialog
                .description
                .clone()
                .unwrap_or_else(|| state.t("dialog.no_mcp_tools")),
            DialogKind::McpToolsPicker => state.t("dialog.no_server_tools"),
            DialogKind::SkillPicker => state.t("dialog.no_skills"),
            DialogKind::PermissionPicker => state.t("dialog.no_permission_modes"),
            DialogKind::ThemePicker => state.t("dialog.no_themes"),
            DialogKind::FakePicker => state.t("dialog.no_fake_clients"),
            DialogKind::ReasoningPicker => state.t("dialog.no_reasoning"),
            DialogKind::ThoughtsPicker => state.t("dialog.no_thoughts"),
            DialogKind::ToolsPicker => state.t("dialog.no_items"),
            DialogKind::AgentPicker => state.t("dialog.no_experts"),
            DialogKind::ExpertModelPicker(_) | DialogKind::ModelPicker => {
                state.t("dialog.no_models")
            }
            _ => state.t("dialog.no_items"),
        };
        frame.render_widget(
            Paragraph::new(Line::from(Span::styled(empty_label, muted_style(theme))))
                .style(theme.elevated_style()),
            Rect::new(area.x, y, area.width, 1),
        );
    }
}

fn render_session_picker_footer(
    frame: &mut Frame<'_>,
    area: Rect,
    theme: Theme,
    state: &TuiState,
    dialog: &DialogState,
) {
    let target = match dialog.session_scope {
        SessionPickerScope::Workspace => state.t("ui.session_scope_all"),
        SessionPickerScope::All => state.t("ui.session_scope_this"),
    };
    let spans = vec![
        Span::styled("←/→", accent_style(theme)),
        Span::styled(format!(" {target}"), muted_style(theme)),
        Span::styled("  ·  ", muted_style(theme)),
        Span::styled("Enter", accent_style(theme)),
        Span::styled(format!(" {}", state.t("ui.select")), muted_style(theme)),
        Span::styled("  ·  ", muted_style(theme)),
        Span::styled("Ctrl-D", accent_style(theme)),
        Span::styled(format!(" {}", state.t("ui.delete")), muted_style(theme)),
        Span::styled("  ·  ", muted_style(theme)),
        Span::styled("Esc", accent_style(theme)),
        Span::styled(format!(" {}", state.t("ui.close")), muted_style(theme)),
    ];
    frame.render_widget(
        Paragraph::new(Line::from(spans)).style(theme.elevated_style()),
        area,
    );
}

fn render_expert_model_picker_footer(
    frame: &mut Frame<'_>,
    area: Rect,
    theme: Theme,
    state: &TuiState,
) {
    let spans = vec![
        Span::styled("Space", accent_style(theme)),
        Span::styled(format!(" {}", state.t("ui.toggle")), muted_style(theme)),
        Span::styled("  ·  ", muted_style(theme)),
        Span::styled("Enter", accent_style(theme)),
        Span::styled(format!(" {}", state.t("ui.confirm")), muted_style(theme)),
        Span::styled("  ·  ", muted_style(theme)),
        Span::styled("Esc", accent_style(theme)),
        Span::styled(format!(" {}", state.t("ui.back")), muted_style(theme)),
    ];
    frame.render_widget(
        Paragraph::new(Line::from(spans)).style(theme.elevated_style()),
        area,
    );
}

fn render_config_footer(frame: &mut Frame<'_>, area: Rect, theme: Theme, state: &TuiState) {
    let Some(dialog) = state.dialog() else {
        return;
    };
    let confirming = dialog.config_close_selected.is_some();
    let editing = state.config_edit.is_some();
    let expanded = dialog.config_expanded.is_some();
    let list_content = dialog.config_detail_target.is_none()
        && dialog.config_expanded.is_some_and(|index| {
            matches!(
                dialog.config_fields.get(index),
                Some(ConfigFieldRef::List(_))
            )
        });
    let mut hints = if confirming || editing {
        vec![
            (
                "Enter",
                state.t(if confirming {
                    "ui.select"
                } else {
                    "ui.confirm"
                }),
            ),
            ("Esc", state.t("ui.cancel")),
        ]
    } else {
        vec![
            ("Ctrl-S", state.t("ui.save")),
            (
                "Esc",
                state.t(if expanded {
                    "ui.collapse"
                } else if dialog.config_path.is_empty() {
                    "ui.close"
                } else {
                    "ui.back"
                }),
            ),
            (
                "Enter",
                state.t(if expanded && !list_content {
                    "ui.select"
                } else {
                    "ui.edit"
                }),
            ),
        ]
    };
    if !confirming && !editing && list_content {
        hints.push(("a", state.t("ui.add")));
        hints.push(("d", state.t("ui.remove")));
    }
    if !editing {
        hints.push(("↑/↓", state.t("ui.navigate")));
        if expanded && !confirming {
            hints.push(("←", state.t("ui.collapse")));
        }
    }
    let separator = "  ·  ";
    let core_width = hints
        .iter()
        .take(2)
        .map(|(key, label)| display_width(key) + 1 + display_width(label))
        .sum::<usize>()
        + display_width(separator);
    let mut spans = Vec::new();
    let mut width = 0;
    let unsaved = state.t("config.unsaved");
    let unsaved_width = display_width("● ") + display_width(&unsaved);
    if dialog.config_dirty
        && !editing
        && !confirming
        && unsaved_width + display_width(separator) + core_width <= area.width as usize
    {
        spans.push(Span::styled("● ", accent_style(theme)));
        spans.push(Span::styled(unsaved, muted_style(theme)));
        width = unsaved_width;
    }
    for (index, (key, label)) in hints.into_iter().enumerate() {
        let hint_width = display_width(key) + 1 + display_width(&label);
        let separator_width = if spans.is_empty() {
            0
        } else {
            display_width(separator)
        };
        if index >= 2 && width + separator_width + hint_width > area.width as usize {
            break;
        }
        if !spans.is_empty() {
            spans.push(Span::styled(separator, muted_style(theme)));
        }
        spans.push(Span::styled(key, accent_style(theme)));
        spans.push(Span::styled(format!(" {label}"), muted_style(theme)));
        width += separator_width + hint_width;
    }
    frame.render_widget(
        Paragraph::new(Line::from(spans)).style(theme.elevated_style()),
        area,
    );
}

fn render_mcp_picker_footer(
    frame: &mut Frame<'_>,
    area: Rect,
    theme: Theme,
    state: &TuiState,
    kind: DialogKind,
) {
    let spans = match kind {
        DialogKind::McpPicker => vec![
            Span::styled("Space", accent_style(theme)),
            Span::styled(format!(" {}", state.t("ui.toggle")), muted_style(theme)),
            Span::styled("  ·  ", muted_style(theme)),
            Span::styled("Enter", accent_style(theme)),
            Span::styled(format!(" {}", state.t("ui.tools")), muted_style(theme)),
            Span::styled("  ·  ", muted_style(theme)),
            Span::styled("Esc", accent_style(theme)),
            Span::styled(format!(" {}", state.t("ui.close")), muted_style(theme)),
        ],
        DialogKind::McpToolsPicker => vec![
            Span::styled("Esc", accent_style(theme)),
            Span::styled(format!(" {}", state.t("ui.back")), muted_style(theme)),
        ],
        _ => unreachable!("MCP picker footer only renders MCP picker kinds"),
    };
    frame.render_widget(
        Paragraph::new(Line::from(spans)).style(theme.elevated_style()),
        area,
    );
}

fn render_context_picker_body(
    frame: &mut Frame<'_>,
    area: Rect,
    theme: Theme,
    state: &mut TuiState,
    dialog: &DialogState,
) {
    if area.is_empty() {
        return;
    }

    let [list_area, gap_area, preview_area] = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Percentage(46),
            Constraint::Length(2),
            Constraint::Percentage(54),
        ])
        .split(area)
        .as_ref()
        .try_into()
        .unwrap_or([
            area,
            Rect::new(area.x, area.y, 0, 0),
            Rect::new(area.x, area.y, 0, 0),
        ]);

    render_picker_body(frame, list_area, theme, state, dialog);

    if gap_area.width > 0 {
        frame.render_widget(
            Paragraph::new(vec![
                Line::from(Span::styled("│", muted_style(theme)));
                gap_area.height as usize
            ])
            .style(theme.elevated_style()),
            Rect::new(
                gap_area
                    .x
                    .saturating_add(gap_area.width.saturating_sub(1) / 2),
                gap_area.y,
                1.min(gap_area.width),
                gap_area.height,
            ),
        );
    }

    render_context_preview(frame, preview_area, theme, state, dialog);
}

fn render_context_preview(
    frame: &mut Frame<'_>,
    area: Rect,
    theme: Theme,
    state: &mut TuiState,
    dialog: &DialogState,
) {
    if area.is_empty() {
        return;
    }

    state.update_context_picker_detail_viewport(area.width, area.height);
    let detail_scroll = state
        .dialog()
        .filter(|dialog| dialog.kind == DialogKind::ContextPicker)
        .map(|dialog| dialog.detail_scroll.min(dialog.detail_scroll_max))
        .unwrap_or_else(|| dialog.detail_scroll.min(dialog.detail_scroll_max));

    let mut lines = Vec::new();
    if let Some(detail) = state.active_context_open_detail() {
        lines.push(Line::from(Span::styled(
            detail.title,
            Style::default()
                .fg(theme.text)
                .bg(theme.elevated_bg)
                .add_modifier(Modifier::BOLD),
        )));
        if !detail.badges.is_empty() {
            lines.push(Line::from(Span::styled(
                detail.badges.join(" · "),
                muted_style(theme),
            )));
        }
        if !detail.lines.is_empty() {
            lines.push(Line::default());
            lines.extend(
                detail
                    .lines
                    .into_iter()
                    .map(|line| Line::from(Span::styled(line, item_style(theme)))),
            );
        }
    } else {
        lines.push(Line::from(Span::styled(
            state.t("ui.no_detail"),
            muted_style(theme),
        )));
    }

    frame.render_widget(
        Paragraph::new(lines)
            .style(theme.elevated_style())
            .wrap(Wrap { trim: false })
            .scroll((detail_scroll, 0)),
        area,
    );
}

fn render_context_picker_footer(
    frame: &mut Frame<'_>,
    area: Rect,
    theme: Theme,
    dialog: &DialogState,
) {
    frame.render_widget(
        Paragraph::new(Line::from(vec![
            Span::styled("↑/↓", accent_style(theme)),
            Span::styled(
                if dialog.detail_focused {
                    " scroll"
                } else {
                    " browse"
                },
                muted_style(theme),
            ),
            Span::styled("  •  ", muted_style(theme)),
            if dialog.detail_focused {
                Span::styled("Esc", accent_style(theme))
            } else {
                Span::styled("Enter", accent_style(theme))
            },
            Span::styled(
                if dialog.detail_focused {
                    " back"
                } else {
                    " detail"
                },
                muted_style(theme),
            ),
            if dialog.detail_focused {
                Span::styled("", muted_style(theme))
            } else {
                Span::styled("  •  ", muted_style(theme))
            },
            if dialog.detail_focused {
                Span::styled("", muted_style(theme))
            } else {
                Span::styled("Esc", accent_style(theme))
            },
            Span::styled(
                if dialog.detail_focused { "" } else { " close" },
                muted_style(theme),
            ),
        ]))
        .style(theme.elevated_style()),
        area,
    );
}

enum PickerEntry<'a> {
    Heading(&'a str),
    Item {
        index: usize,
        item: &'a DialogItem,
        rows: u16,
    },
}

impl PickerEntry<'_> {
    fn rows(&self) -> u16 {
        match self {
            Self::Heading(_) => 1,
            Self::Item { rows, .. } => *rows,
        }
    }
}

fn visible_picker_entries<'a>(
    dialog: &'a DialogState,
    rows: usize,
    body_width: u16,
) -> Vec<PickerEntry<'a>> {
    if rows == 0 {
        return Vec::new();
    }

    let entries = picker_entries(dialog, body_width);
    let total_rows: usize = entries.iter().map(|entry| entry.rows() as usize).sum();
    if total_rows <= rows {
        return entries;
    }

    let selected_position = entries
        .iter()
        .position(
            |entry| matches!(entry, PickerEntry::Item { index, .. } if *index == dialog.selected),
        )
        .unwrap_or(0);
    let mut start = selected_position;
    let mut used_rows = 0usize;
    for position in (0..=selected_position).rev() {
        let height = entries[position].rows() as usize;
        if used_rows + height > rows {
            break;
        }
        used_rows += height;
        start = position;
    }

    entries.into_iter().skip(start).collect()
}

fn picker_entries<'a>(dialog: &'a DialogState, body_width: u16) -> Vec<PickerEntry<'a>> {
    let mut entries = Vec::new();
    let mut previous_section: Option<&str> = None;

    for (index, item) in dialog.visible_items() {
        if matches!(
            dialog.kind,
            DialogKind::ModelPicker
                | DialogKind::AgentPicker
                | DialogKind::ExpertModelPicker(_)
                | DialogKind::SessionPicker
                | DialogKind::HistoryTree
                | DialogKind::ContextPicker
                | DialogKind::ConfigEditor
        ) {
            let section = item.section.as_deref().unwrap_or_else(|| {
                if dialog.kind == DialogKind::HistoryTree {
                    "Branches"
                } else if dialog.kind == DialogKind::ContextPicker {
                    "Context"
                } else if dialog.kind == DialogKind::AgentPicker {
                    "Experts"
                } else if matches!(
                    dialog.kind,
                    DialogKind::ModelPicker | DialogKind::ExpertModelPicker(_)
                ) {
                    "Models"
                } else {
                    "Sessions"
                }
            });
            if previous_section != Some(section) {
                entries.push(PickerEntry::Heading(section));
                previous_section = Some(section);
            }
        }

        entries.push(PickerEntry::Item {
            index,
            item,
            rows: picker_item_rows(dialog, item, body_width),
        });
    }

    entries
}

fn picker_item_rows(dialog: &DialogState, item: &DialogItem, body_width: u16) -> u16 {
    if dialog.kind != DialogKind::AgentPicker {
        return 1;
    }
    let Some(models) = item.right_detail.as_deref() else {
        return 1;
    };
    let text_width = agent_model_text_width(body_width) as usize;
    if text_width == 0 {
        return 1;
    }
    1 + wrap_model_list(models, text_width).len() as u16
}

fn agent_model_text_width(body_width: u16) -> u16 {
    body_width
        .saturating_sub(2)
        .saturating_sub(AGENT_MODEL_INDENT)
}

fn render_section_heading(
    frame: &mut Frame<'_>,
    area: Rect,
    theme: Theme,
    heading: &str,
    color: ratatui::style::Color,
) {
    frame.render_widget(
        Paragraph::new(Line::from(Span::styled(
            heading.to_string(),
            Style::default()
                .fg(color)
                .bg(theme.elevated_bg)
                .add_modifier(Modifier::BOLD),
        )))
        .style(theme.elevated_style()),
        area,
    );
}

fn render_model_row(
    frame: &mut Frame<'_>,
    area: Rect,
    theme: Theme,
    item: &DialogItem,
    selected: bool,
    current: bool,
) {
    let row_style = if selected {
        selected_item_style(theme)
    } else {
        item_style(theme)
    };
    frame.render_widget(Block::default().style(row_style), area);

    let content = area.inner(Margin::new(1, 0));
    if content.is_empty() {
        return;
    }

    let marker = if current {
        "● "
    } else if item.checked {
        "✓ "
    } else {
        "  "
    };
    let mut spans = vec![Span::styled(marker, row_style)];
    spans.push(Span::styled(item.label.clone(), row_style));

    if let Some(detail) = &item.detail {
        spans.push(Span::styled(" ", row_style));
        spans.push(Span::styled(
            detail.clone(),
            if selected {
                selected_muted_style(theme)
            } else {
                muted_style(theme)
            },
        ));
    }

    frame.render_widget(Paragraph::new(Line::from(spans)).style(row_style), content);
}

fn render_config_split_body(
    frame: &mut Frame<'_>,
    area: Rect,
    theme: Theme,
    state: &mut TuiState,
    dialog: &DialogState,
) {
    if area.width < 50 {
        render_config_detail(frame, area, theme, dialog, state.config_edit.as_ref());
        return;
    }
    let [list_area, gap_area, detail_area] = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Percentage(52),
            Constraint::Length(2),
            Constraint::Percentage(46),
        ])
        .split(area)
        .as_ref()
        .try_into()
        .unwrap_or([
            area,
            Rect::new(area.x, area.y, 0, 0),
            Rect::new(area.x, area.y, 0, 0),
        ]);

    render_picker_body(frame, list_area, theme, state, dialog);
    frame.render_widget(Block::default().style(theme.elevated_style()), gap_area);
    render_config_detail(
        frame,
        detail_area,
        theme,
        dialog,
        state.config_edit.as_ref(),
    );
}

fn render_config_detail(
    frame: &mut Frame<'_>,
    area: Rect,
    theme: Theme,
    dialog: &DialogState,
    edit: Option<&ConfigEditState>,
) {
    if area.is_empty() {
        return;
    }
    let rows = area.height as usize;
    let start = dialog
        .config_detail_selected
        .saturating_add(1)
        .saturating_sub(rows)
        .min(dialog.config_detail_items.len().saturating_sub(rows));
    for (index, item) in dialog
        .config_detail_items
        .iter()
        .enumerate()
        .skip(start)
        .take(rows)
    {
        let row = area.y.saturating_add((index - start) as u16);
        let selected = index == dialog.config_detail_selected;
        let row_style = if selected {
            selected_item_style(theme)
        } else {
            item_style(theme)
        };
        let row_area = Rect::new(area.x, row, area.width, 1);
        frame.render_widget(Block::default().style(row_style), row_area);
        let marker = if selected { "● " } else { "  " };
        let editing = edit.filter(|edit| {
            matches!(&edit.field, ConfigFieldRef::ListItem(_, selected) if *selected == index)
        });
        let spans = if let Some(edit) = editing {
            let marker = if area.width < 4 { "" } else { marker };
            let width = area.width.saturating_sub(1) as usize - display_width(marker);
            vec![
                Span::styled(marker, row_style),
                Span::styled(config_edit_text(edit, width), row_style),
            ]
        } else {
            vec![
                Span::styled(marker, row_style),
                Span::styled(item.label.clone(), row_style),
            ]
        };
        frame.render_widget(
            Paragraph::new(Line::from(spans)).style(row_style),
            Rect::new(
                row_area.x.saturating_add(1),
                row,
                row_area.width.saturating_sub(1),
                1,
            ),
        );
    }
}

fn config_edit_text(edit: &ConfigEditState, width: usize) -> String {
    if width == 0 {
        return String::new();
    }
    let (before, after) = edit.buffer.split_at(edit.cursor.min(edit.buffer.len()));
    let before = before
        .replace('\n', "\\n")
        .replace('\r', "\\r")
        .replace('\t', "\\t");
    let after = after
        .replace('\n', "\\n")
        .replace('\r', "\\r")
        .replace('\t', "\\t");
    let after_width = display_width(&after).min(width / 3);
    let before_width = width.saturating_sub(1 + after_width);
    let mut used = 0;
    let mut start = before.len();
    for (index, grapheme) in before.grapheme_indices(true).rev() {
        let next = used + display_width(grapheme);
        if next > before_width {
            break;
        }
        used = next;
        start = index;
    }
    let mut text = format!("{}▏", &before[start..]);
    used += 1;
    for grapheme in after.graphemes(true) {
        let next = used + display_width(grapheme);
        if next > width {
            break;
        }
        used = next;
        text.push_str(grapheme);
    }
    text
}

fn render_config_row(
    frame: &mut Frame<'_>,
    area: Rect,
    theme: Theme,
    item: &DialogItem,
    selected: bool,
    field: Option<&ConfigFieldRef>,
    edit: Option<&ConfigEditState>,
) {
    let row_style = if selected {
        selected_item_style(theme)
    } else {
        item_style(theme)
    };
    frame.render_widget(Block::default().style(row_style), area);

    let content = area.inner(Margin::new(1, 0));
    if content.is_empty() {
        return;
    }

    let editing = edit.filter(|edit| {
        field == Some(&edit.field)
            || matches!((field, &edit.field), (Some(ConfigFieldRef::Choice(path)), ConfigFieldRef::Field(edit_path)) if path == edit_path)
    });
    if let Some(edit) = editing {
        let marker = if content.width < 4 {
            ""
        } else if selected {
            "● "
        } else {
            "  "
        };
        let available = content.width as usize - display_width(marker);
        let label_width = display_width(&item.label).min(available.saturating_sub(3) / 3);
        let label = truncate_display_width(&item.label, label_width);
        let prefix = if label.is_empty() {
            String::new()
        } else {
            format!("{label}  ")
        };
        let text = config_edit_text(edit, available.saturating_sub(display_width(&prefix)));
        frame.render_widget(
            Paragraph::new(format!("{marker}{prefix}{text}")).style(row_style),
            content,
        );
        return;
    }
    let affordance = item.right_detail.as_deref().unwrap_or("");
    let right_width =
        (display_width(affordance) as u16).min(content.width.saturating_sub(MIN_LEFT_LABEL_WIDTH));
    let left_width = content.width.saturating_sub(right_width.saturating_add(2));
    let left_area = Rect::new(content.x, content.y, left_width, content.height);
    let right_area = Rect::new(
        content.right().saturating_sub(right_width),
        content.y,
        right_width,
        content.height,
    );

    let marker = if selected { "● " } else { "  " };
    let mut spans = vec![
        Span::styled(marker, row_style),
        Span::styled(item.label.clone(), row_style),
    ];
    if let Some(detail) = &item.detail {
        spans.push(Span::styled("  ", row_style));
        spans.push(Span::styled(
            detail.clone(),
            if selected {
                selected_muted_style(theme)
            } else {
                muted_style(theme)
            },
        ));
    }
    frame.render_widget(
        Paragraph::new(Line::from(spans)).style(row_style),
        left_area,
    );

    if right_width > 0 {
        frame.render_widget(
            Paragraph::new(Line::from(Span::styled(
                affordance.to_string(),
                if selected {
                    selected_item_style(theme)
                } else {
                    muted_style(theme)
                },
            )))
            .style(row_style),
            right_area,
        );
    }
}

fn render_permission_row(
    frame: &mut Frame<'_>,
    area: Rect,
    theme: Theme,
    item: &DialogItem,
    selected: bool,
    current: bool,
) {
    render_model_row(frame, area, theme, item, selected, current);
}

fn render_reasoning_row(
    frame: &mut Frame<'_>,
    area: Rect,
    theme: Theme,
    item: &DialogItem,
    selected: bool,
    current: bool,
) {
    render_model_row(frame, area, theme, item, selected, current);
}

fn reasoning_item_is_current(current: Option<&str>, item_id: &str) -> bool {
    match current {
        Some("off") => item_id == "none",
        Some(value) => value == item_id,
        None => item_id == "none",
    }
}

fn render_agent_row(
    frame: &mut Frame<'_>,
    area: Rect,
    theme: Theme,
    item: &DialogItem,
    selected: bool,
) {
    let row_style = if selected {
        selected_item_style(theme)
    } else {
        item_style(theme)
    };
    frame.render_widget(Block::default().style(row_style), area);

    let content = area.inner(Margin::new(1, 0));
    if content.is_empty() {
        return;
    }

    let marker = if selected { "● " } else { "  " };
    frame.render_widget(
        Paragraph::new(Line::from(vec![
            Span::styled(marker, row_style),
            Span::styled(item.label.clone(), row_style),
        ]))
        .style(row_style),
        Rect::new(content.x, content.y, content.width, 1),
    );

    let Some(models) = item.right_detail.as_deref() else {
        return;
    };
    let indent = AGENT_MODEL_INDENT.min(content.width);
    let text_width = content.width.saturating_sub(indent);
    if text_width == 0 {
        return;
    }

    let detail_style = Style::default().fg(theme.muted_text).bg(if selected {
        theme.element_bg
    } else {
        theme.elevated_bg
    });
    for (offset, line) in wrap_model_list(models, text_width as usize)
        .into_iter()
        .enumerate()
    {
        let y = content.y.saturating_add(1).saturating_add(offset as u16);
        if y >= area.bottom() {
            break;
        }
        frame.render_widget(
            Paragraph::new(Line::from(Span::styled(line, detail_style))).style(row_style),
            Rect::new(content.x + indent, y, text_width, 1),
        );
    }
}

/// Packs an expert's model list into rows without splitting a route name; only a single
/// route wider than the row falls back to a hard wrap.
fn wrap_model_list(models: &str, width: usize) -> Vec<String> {
    if width == 0 {
        return Vec::new();
    }

    let mut rows: Vec<String> = Vec::new();
    let mut current = String::new();
    for model in models.split(MODEL_SEPARATOR) {
        let candidate_width = if current.is_empty() {
            display_width(model)
        } else {
            display_width(&current) + display_width(MODEL_SEPARATOR) + display_width(model)
        };
        if !current.is_empty() && candidate_width > width {
            rows.push(std::mem::take(&mut current));
        }
        if !current.is_empty() {
            current.push_str(MODEL_SEPARATOR);
        }
        current.push_str(model);
    }
    if !current.is_empty() {
        rows.push(current);
    }
    if rows.is_empty() {
        rows.push(String::new());
    }

    rows.into_iter()
        .flat_map(|row| wrap_text_to_width(&row, width))
        .collect()
}

fn render_session_row(
    frame: &mut Frame<'_>,
    area: Rect,
    theme: Theme,
    item: &DialogItem,
    selected: bool,
    status_color: Option<ratatui::style::Color>,
) {
    let row_style = if selected {
        selected_item_style(theme)
    } else {
        item_style(theme)
    };
    frame.render_widget(Block::default().style(row_style), area);

    let content = area.inner(Margin::new(1, 0));
    if content.is_empty() {
        return;
    }

    let right_width = item
        .right_detail
        .as_ref()
        .map(|detail| display_width(detail) as u16)
        .unwrap_or(0)
        .min(content.width.saturating_sub(MIN_LEFT_LABEL_WIDTH));
    let left_width = content.width.saturating_sub(right_width.saturating_add(2));
    let left_area = Rect::new(content.x, content.y, left_width, content.height);
    let right_area = Rect::new(
        content.right().saturating_sub(right_width),
        content.y,
        right_width,
        content.height,
    );

    let marker = if selected { "● " } else { "  " };
    frame.render_widget(
        Paragraph::new(Line::from(vec![
            Span::styled(marker, row_style),
            Span::styled(item.label.clone(), row_style),
        ]))
        .style(row_style),
        left_area,
    );

    if let Some(right_detail) = &item.right_detail
        && right_width > 0
    {
        frame.render_widget(
            Paragraph::new(Line::from(Span::styled(
                truncate_display_width(right_detail, right_width as usize),
                if let Some(color) = status_color {
                    Style::default().fg(color).bg(if selected {
                        theme.element_bg
                    } else {
                        theme.elevated_bg
                    })
                } else if selected {
                    selected_item_style(theme)
                } else {
                    muted_style(theme)
                },
            )))
            .style(row_style),
            right_area,
        );
    }
}

fn mcp_status_color(
    item: &DialogItem,
    theme: Theme,
    language: crate::tui::i18n::Language,
) -> ratatui::style::Color {
    let translator = crate::tui::i18n::Translator::new(language);
    match item.right_detail.as_deref() {
        Some(status) if status.contains(&translator.t("status.online")) => theme.success,
        Some(status) if status.contains(&translator.t("status.offline")) => theme.error,
        _ => theme.muted_text,
    }
}

fn centered_picker_area(area: Rect) -> Rect {
    let target_width = area.width.saturating_mul(3) / 4;
    let width = target_width
        .clamp(PICKER_MIN_WIDTH, PICKER_MAX_WIDTH)
        .min(area.width.saturating_sub(2))
        .max(1);
    let target_height = area.height.saturating_mul(4) / 5;
    let height = target_height
        .clamp(PICKER_MIN_HEIGHT, PICKER_MAX_HEIGHT)
        .min(area.height.saturating_sub(2))
        .max(1);

    Rect::new(
        area.x + area.width.saturating_sub(width) / 2,
        area.y + area.height.saturating_sub(height) / 2,
        width,
        height,
    )
}

pub(super) fn item_style(theme: Theme) -> Style {
    Style::default().fg(theme.text).bg(theme.elevated_bg)
}

pub(super) fn selected_item_style(theme: Theme) -> Style {
    Style::default()
        .fg(theme.text)
        .bg(theme.element_bg)
        .add_modifier(Modifier::BOLD)
}

pub(super) fn muted_style(theme: Theme) -> Style {
    Style::default().fg(theme.muted_text).bg(theme.elevated_bg)
}

pub(super) fn accent_style(theme: Theme) -> Style {
    Style::default()
        .fg(theme.accent)
        .bg(theme.elevated_bg)
        .add_modifier(Modifier::BOLD)
}

fn selected_muted_style(theme: Theme) -> Style {
    Style::default().fg(theme.muted_text).bg(theme.element_bg)
}

#[cfg(test)]
#[path = "../../../tests/internal/tui/components/picker/tests.rs"]
mod tests;
