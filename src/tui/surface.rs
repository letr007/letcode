use ratatui::style::{Color, Style};

use super::theme::Theme;

pub const OUTER_PAD_X: u16 = 2;
pub const SESSION_PAD_BOTTOM: u16 = 1;
pub const CONTENT_GAP: u16 = 1;
pub const TRANSCRIPT_TOP_SPACER: usize = 1;

pub const ACCENT_BAR_WIDTH: u16 = 1;
pub const ACCENT_BAR_GLYPH: &str = "┃";
pub const PROMPT_BOTTOM_LEFT_GLYPH: &str = "╹";
pub const PROMPT_BOTTOM_CAP_GLYPH: &str = "▀";

pub const CARD_PAD_RIGHT: u16 = 1;

pub const PROMPT_INNER_PAD_X: u16 = 2;
pub const PROMPT_INNER_PAD_TOP: u16 = 1;
pub const PROMPT_METADATA_PAD_TOP: u16 = 1;
pub const PROMPT_INNER_PAD_BOTTOM: u16 = 1;
pub const TEXTAREA_MIN_ROWS: u16 = 1;
pub const TEXTAREA_MAX_ROWS: u16 = 6;
pub const WELCOME_PROMPT_MAX_WIDTH: u16 = 75;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SurfaceKind {
    Root,
    Element,
    Elevated,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SurfaceEmphasis {
    User,
    Approval,
    Notice,
}

pub fn surface_bg(theme: Theme, kind: SurfaceKind) -> Color {
    match kind {
        SurfaceKind::Root => theme.root_bg,
        SurfaceKind::Element => theme.element_bg,
        SurfaceKind::Elevated => theme.elevated_bg,
    }
}

pub fn accent_color(theme: Theme, emphasis: SurfaceEmphasis) -> Color {
    match emphasis {
        SurfaceEmphasis::User => theme.user,
        SurfaceEmphasis::Approval => theme.approval,
        SurfaceEmphasis::Notice => theme.notice,
    }
}

pub fn surface_style(theme: Theme, kind: SurfaceKind) -> Style {
    Style::default().bg(surface_bg(theme, kind)).fg(theme.text)
}

pub fn accent_style(theme: Theme, emphasis: SurfaceEmphasis, kind: SurfaceKind) -> Style {
    Style::default()
        .fg(accent_color(theme, emphasis))
        .bg(surface_bg(theme, kind))
}

pub fn muted_style(theme: Theme, kind: SurfaceKind) -> Style {
    Style::default()
        .fg(theme.muted_text)
        .bg(surface_bg(theme, kind))
}
