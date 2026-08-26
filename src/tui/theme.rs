use ratatui::style::{Color, Modifier, Style};

pub use crate::command::ThemeName;

/// Shared TUI color tokens.
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
    pub fake: Color,
    pub diff_add_bg: Color,
    pub diff_delete_bg: Color,
    pub diff_hunk_bg: Color,
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
            fake: Color::Rgb(232, 121, 249),
            diff_add_bg: Color::Rgb(22, 45, 32),
            diff_delete_bg: Color::Rgb(54, 32, 42),
            diff_hunk_bg: Color::Rgb(31, 40, 60),
        }
    }

    pub const fn card_guide(self) -> Color {
        self.border
    }

    pub const fn card_bg(self) -> Color {
        self.element_bg
    }

    pub fn for_name(name: ThemeName, frame: usize) -> Self {
        let theme = Self::dark();
        if name == ThemeName::Rainbow {
            theme.with_rainbow_accent(frame)
        } else {
            theme
        }
    }

    fn with_rainbow_accent(mut self, frame: usize) -> Self {
        const COLORS: [(u8, u8, u8); 6] = [
            (232, 105, 105),
            (232, 167, 80),
            (213, 205, 83),
            (91, 201, 125),
            (79, 178, 224),
            (185, 123, 222),
        ];
        let color = |offset: usize| {
            let (red, green, blue) = COLORS[((frame / 3) + offset) % COLORS.len()];
            Color::Rgb(red, green, blue)
        };
        self.accent = color(0);
        self.user = color(1);
        self.assistant = color(2);
        self.approval = color(3);
        self.notice = color(4);
        self.border = color(5);
        self
    }

    pub fn app_style(self) -> Style {
        Style::default().bg(self.root_bg).fg(self.text)
    }

    pub fn elevated_style(self) -> Style {
        Style::default().bg(self.elevated_bg).fg(self.text)
    }

    pub fn user_style(self) -> Style {
        Style::default().fg(self.user).bg(self.surface_bg)
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rainbow_cycles_accent_without_changing_error_semantics() {
        let first = Theme::for_name(ThemeName::Rainbow, 0);
        let next = Theme::for_name(ThemeName::Rainbow, 3);

        assert_ne!(first.accent, next.accent);
        assert_ne!(first.user, next.user);
        assert_ne!(first.assistant, next.assistant);
        assert_ne!(first.border, next.border);
        assert_eq!(first.error, next.error);
        assert_eq!(first.warning, next.warning);
        assert_eq!(first.success, next.success);
        assert_eq!(Theme::for_name(ThemeName::Dark, 0), Theme::dark());
    }
}
