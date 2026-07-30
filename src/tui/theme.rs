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

    pub const fn ocean() -> Self {
        Self {
            root_bg: Color::Rgb(10, 22, 32),
            surface_bg: Color::Rgb(14, 31, 44),
            element_bg: Color::Rgb(19, 41, 57),
            elevated_bg: Color::Rgb(26, 52, 70),
            border: Color::Rgb(48, 83, 105),
            text: Color::Rgb(218, 234, 240),
            muted_text: Color::Rgb(137, 169, 181),
            dim_text: Color::Rgb(82, 114, 128),
            accent: Color::Rgb(62, 190, 211),
            assistant: Color::Rgb(101, 207, 171),
            user: Color::Rgb(80, 175, 224),
            success: Color::Rgb(105, 202, 132),
            warning: Color::Rgb(218, 183, 96),
            error: Color::Rgb(228, 101, 105),
            approval: Color::Rgb(231, 187, 83),
            notice: Color::Rgb(113, 163, 183),
        }
    }

    pub const fn forest() -> Self {
        Self {
            root_bg: Color::Rgb(17, 27, 20),
            surface_bg: Color::Rgb(23, 36, 27),
            element_bg: Color::Rgb(31, 47, 35),
            elevated_bg: Color::Rgb(42, 61, 45),
            border: Color::Rgb(70, 92, 73),
            text: Color::Rgb(225, 232, 218),
            muted_text: Color::Rgb(153, 171, 146),
            dim_text: Color::Rgb(95, 115, 91),
            accent: Color::Rgb(150, 199, 92),
            assistant: Color::Rgb(107, 205, 139),
            user: Color::Rgb(126, 189, 103),
            success: Color::Rgb(111, 203, 126),
            warning: Color::Rgb(214, 184, 91),
            error: Color::Rgb(227, 99, 94),
            approval: Color::Rgb(222, 184, 81),
            notice: Color::Rgb(138, 166, 112),
        }
    }

    pub const fn rose() -> Self {
        Self {
            root_bg: Color::Rgb(31, 19, 29),
            surface_bg: Color::Rgb(43, 25, 40),
            element_bg: Color::Rgb(57, 32, 53),
            elevated_bg: Color::Rgb(73, 42, 67),
            border: Color::Rgb(101, 65, 95),
            text: Color::Rgb(239, 224, 234),
            muted_text: Color::Rgb(186, 151, 174),
            dim_text: Color::Rgb(122, 89, 113),
            accent: Color::Rgb(228, 130, 181),
            assistant: Color::Rgb(129, 203, 169),
            user: Color::Rgb(205, 132, 219),
            success: Color::Rgb(112, 204, 143),
            warning: Color::Rgb(226, 180, 96),
            error: Color::Rgb(233, 102, 112),
            approval: Color::Rgb(232, 177, 84),
            notice: Color::Rgb(186, 130, 168),
        }
    }

    pub fn for_name(name: ThemeName, frame: usize) -> Self {
        let theme = match name {
            ThemeName::Dark | ThemeName::Rainbow => Self::dark(),
            ThemeName::Ocean => Self::ocean(),
            ThemeName::Forest => Self::forest(),
            ThemeName::Rose => Self::rose(),
        };
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn regular_presets_are_distinct() {
        assert_ne!(
            Theme::for_name(ThemeName::Dark, 0),
            Theme::for_name(ThemeName::Ocean, 0)
        );
        assert_ne!(
            Theme::for_name(ThemeName::Ocean, 0),
            Theme::for_name(ThemeName::Forest, 0)
        );
        assert_ne!(
            Theme::for_name(ThemeName::Forest, 0),
            Theme::for_name(ThemeName::Rose, 0)
        );
    }

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
    }
}
