use ratatui::style::{Color, Modifier, Style};

/// Centralized dark theme tokens inspired by the old letcode-old TUI's visual style.
///
/// This module is intentionally limited to shared color/style tokens used by the
/// Ratatui view layer. It documents the visual reference to `letcode-old` without
/// pulling presentation policy, runtime behavior, or state architecture back into
/// a monolithic design.

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Theme {
    pub root_bg: Color,
    pub surface_bg: Color,
    pub element_bg: Color,
    pub elevated_bg: Color,
    pub border: Color,
    pub text: Color,
    pub muted_text: Color,
    pub dim_text: Color,
    pub accent: Color,
    pub assistant: Color,
    pub user: Color,
    pub success: Color,
    pub warning: Color,
    pub error: Color,
    pub approval: Color,
    pub notice: Color,
}

impl Theme {
    pub const fn dark() -> Self {
        Self {
            root_bg: Color::Rgb(18, 18, 18),
            surface_bg: Color::Rgb(24, 24, 26),
            element_bg: Color::Rgb(30, 30, 32),
            elevated_bg: Color::Rgb(38, 38, 40),
            border: Color::Rgb(50, 50, 54),
            text: Color::Rgb(220, 220, 220),
            muted_text: Color::Rgb(130, 130, 130),
            dim_text: Color::Rgb(80, 80, 80),
            accent: Color::Rgb(80, 180, 220),
            assistant: Color::Rgb(100, 210, 130),
            user: Color::Rgb(80, 180, 220),
            success: Color::Rgb(100, 200, 100),
            warning: Color::Rgb(180, 180, 100),
            error: Color::Rgb(220, 80, 80),
            approval: Color::Rgb(220, 180, 60),
            notice: Color::Rgb(100, 100, 100),
        }
    }

    pub fn app_style(self) -> Style {
        Style::default().bg(self.root_bg).fg(self.text)
    }

    pub fn surface_style(self) -> Style {
        Style::default().bg(self.surface_bg).fg(self.text)
    }

    pub fn elevated_style(self) -> Style {
        Style::default().bg(self.elevated_bg).fg(self.text)
    }

    pub fn element_style(self) -> Style {
        Style::default().bg(self.element_bg).fg(self.text)
    }

    pub fn border_style(self) -> Style {
        Style::default().fg(self.border).bg(self.surface_bg)
    }

    pub fn title_style(self) -> Style {
        Style::default()
            .fg(self.accent)
            .bg(self.surface_bg)
            .add_modifier(Modifier::BOLD)
    }

    pub fn muted_style(self) -> Style {
        Style::default().fg(self.muted_text).bg(self.surface_bg)
    }

    pub fn dim_style(self) -> Style {
        Style::default().fg(self.dim_text).bg(self.root_bg)
    }

    pub fn user_style(self) -> Style {
        Style::default().fg(self.user).bg(self.surface_bg)
    }

    pub fn assistant_style(self) -> Style {
        Style::default().fg(self.assistant).bg(self.surface_bg)
    }

    pub fn success_style(self) -> Style {
        Style::default().fg(self.success).bg(self.surface_bg)
    }

    pub fn warning_style(self) -> Style {
        Style::default().fg(self.warning).bg(self.surface_bg)
    }

    pub fn error_style(self) -> Style {
        Style::default().fg(self.error).bg(self.surface_bg)
    }

    pub fn approval_style(self) -> Style {
        Style::default()
            .fg(self.approval)
            .bg(self.elevated_bg)
            .add_modifier(Modifier::BOLD)
    }
}

impl Default for Theme {
    fn default() -> Self {
        Self::dark()
    }
}
