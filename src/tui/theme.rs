use ratatui::style::{Color, Modifier, Style};

use super::terminal_bg::Rgb;

pub use crate::command::ThemeName;

const SURFACE_DEPTH: f32 = 0.45;
const ELEMENT_DEPTH: f32 = 0.30;
const ELEVATED_DEPTH: f32 = 0.10;

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
    /// Ink for text placed on an accent-filled block such as a chip or a selection.
    pub on_accent: Color,
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
            on_accent: Color::Rgb(18, 18, 18),
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

    /// Dark palette with the screen left to the terminal: `root_bg` becomes `Color::Reset`, so
    /// gaps and the empty transcript keep terminal transparency. Panels are shaded relative to a
    /// detected dark background, or fall back to the fixed ladder when there is none.
    pub fn plain_for(terminal_bg: Option<Rgb>) -> Self {
        let mut theme = Self::dark();
        theme.root_bg = Color::Reset;
        let (surface_bg, element_bg, elevated_bg) = match terminal_bg.filter(|bg| is_dark(*bg)) {
            Some(bg) => (
                darkened(bg, SURFACE_DEPTH),
                darkened(bg, ELEMENT_DEPTH),
                darkened(bg, ELEVATED_DEPTH),
            ),
            None => (
                Color::Rgb(12, 12, 13),
                Color::Rgb(18, 18, 20),
                Color::Rgb(32, 32, 35),
            ),
        };
        theme.surface_bg = surface_bg;
        theme.element_bg = element_bg;
        theme.elevated_bg = elevated_bg;
        theme
    }

    pub fn for_name(name: ThemeName, frame: usize, terminal_bg: Option<Rgb>) -> Self {
        match name {
            ThemeName::Dark => Self::dark(),
            ThemeName::Plain => Self::plain_for(terminal_bg),
            ThemeName::Rainbow => Self::dark().with_rainbow_accent(frame),
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

/// Shading a light background yields mid-gray panels that the light panel text cannot sit on.
fn is_dark((red, green, blue): Rgb) -> bool {
    // Rec. 601 luma weights.
    let luma = 299 * u32::from(red) + 587 * u32::from(green) + 114 * u32::from(blue);
    luma < 128 * 1000
}

fn darkened((red, green, blue): Rgb, depth: f32) -> Color {
    let channel = |value: u8| (f32::from(value) * (1.0 - depth)).round() as u8;
    Color::Rgb(channel(red), channel(green), channel(blue))
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
        let first = Theme::for_name(ThemeName::Rainbow, 0, None);
        let next = Theme::for_name(ThemeName::Rainbow, 3, None);

        assert_ne!(first.accent, next.accent);
        assert_ne!(first.user, next.user);
        assert_ne!(first.assistant, next.assistant);
        assert_ne!(first.border, next.border);
        assert_eq!(first.error, next.error);
        assert_eq!(first.warning, next.warning);
        assert_eq!(first.success, next.success);
        assert_eq!(Theme::for_name(ThemeName::Dark, 0, None), Theme::dark());
    }

    fn channel_sum(color: Color) -> u32 {
        let Color::Rgb(red, green, blue) = color else {
            panic!("expected an rgb surface, got {color:?}");
        };
        u32::from(red) + u32::from(green) + u32::from(blue)
    }

    #[test]
    fn plain_theme_leaves_the_screen_to_the_terminal() {
        let plain = Theme::for_name(ThemeName::Plain, 0, None);
        let dark = Theme::dark();

        assert_eq!(plain.root_bg, Color::Reset);
        assert_eq!(plain.text, dark.text);
        assert_eq!(plain.accent, dark.accent);
        assert_eq!(plain.on_accent, dark.on_accent);
        assert_eq!(plain.diff_add_bg, dark.diff_add_bg);
        assert_eq!(plain.diff_delete_bg, dark.diff_delete_bg);
    }

    #[test]
    fn plain_theme_darkens_panels_into_a_monotonic_ladder() {
        let plain = Theme::plain_for(None);
        let dark = Theme::dark();

        for (plain_surface, dark_surface) in [
            (plain.surface_bg, dark.surface_bg),
            (plain.element_bg, dark.element_bg),
            (plain.elevated_bg, dark.elevated_bg),
        ] {
            assert!(
                channel_sum(plain_surface) < channel_sum(dark_surface),
                "{plain_surface:?} should sit below {dark_surface:?}"
            );
        }

        assert!(channel_sum(plain.surface_bg) < channel_sum(plain.element_bg));
        assert!(channel_sum(plain.element_bg) < channel_sum(plain.elevated_bg));
    }

    #[test]
    fn plain_shades_the_ladder_below_a_detected_background() {
        let background: Rgb = (26, 27, 38);
        let plain = Theme::plain_for(Some(background));
        let background = channel_sum(Color::Rgb(background.0, background.1, background.2));

        for surface in [plain.surface_bg, plain.element_bg, plain.elevated_bg] {
            assert!(
                channel_sum(surface) < background,
                "{surface:?} should sit below the terminal background"
            );
        }
        assert!(channel_sum(plain.surface_bg) < channel_sum(plain.element_bg));
        assert!(channel_sum(plain.element_bg) < channel_sum(plain.elevated_bg));
    }

    #[test]
    fn plain_keeps_the_fixed_ladder_for_light_backgrounds() {
        let fixed = Theme::plain_for(None);

        assert_eq!(Theme::plain_for(Some((255, 255, 255))), fixed);
        assert_eq!(Theme::plain_for(Some((250, 250, 250))), fixed);
    }
}
