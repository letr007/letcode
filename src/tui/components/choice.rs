//! Shared rendering for interactive choice rows.
//!
//! The question prompt and the permission approval prompt both expose a list of
//! single-select rows, so they share the same highlighted-row marker and the same
//! inverted chip style for the active row. Each prompt supplies its own accent.

use ratatui::style::{Color, Modifier, Style};

use crate::tui::theme::Theme;

/// 高亮行的行首标记。
pub(crate) fn choice_marker(active: bool) -> &'static str {
    if active { "›" } else { " " }
}

/// 高亮项的反白样式；`accent` 为调用方 prompt 的强调色。
pub(crate) fn highlighted_choice_style(theme: Theme, accent: Color) -> Style {
    Style::default()
        .fg(theme.root_bg)
        .bg(accent)
        .add_modifier(Modifier::BOLD)
}
