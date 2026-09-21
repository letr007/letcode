use ratatui::style::{Color, Modifier, Style};

use super::terminal_bg::Rgb;

pub use crate::command::ThemeName;

const SURFACE_LIFT: f32 = 0.05;
const ELEMENT_LIFT: f32 = 0.12;
const ELEVATED_LIFT: f32 = 0.20;

const ASSUMED_BACKGROUND: Rgb = (24, 25, 34);

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

    /// The background the theme draws on.
    pub const fn canvas(self) -> Color {
        match self.root_bg {
            Color::Reset => Color::Rgb(0, 0, 0),
            color => color,
        }
    }

    /// Dark palette with the screen left to the terminal: `root_bg` becomes `Color::Reset`, so gaps
    /// and the empty transcript stay transparent, and panels lift above a detected background.
    pub fn plain_for(terminal_bg: Option<Rgb>) -> Self {
        let background = usable_background(terminal_bg);
        let mut theme = Self::dark();
        theme.root_bg = Color::Reset;
        theme.surface_bg = lifted(background, SURFACE_LIFT);
        theme.element_bg = lifted(background, ELEMENT_LIFT);
        theme.elevated_bg = lifted(background, ELEVATED_LIFT);
        theme
    }

    /// Every surface, panels included, left to the terminal: the app paints ink only.
    pub const fn glass() -> Self {
        let mut theme = Self::dark();
        theme.root_bg = Color::Reset;
        theme.surface_bg = Color::Reset;
        theme.element_bg = Color::Reset;
        theme.elevated_bg = Color::Reset;
        theme
    }

    /// Whether panels are filled; marks drawn in the surface tone need one.
    pub const fn paints_panels(self) -> bool {
        !matches!(self.element_bg, Color::Reset)
    }

    pub fn for_name(name: ThemeName, frame: usize, terminal_bg: Option<Rgb>) -> Self {
        match name {
            ThemeName::Dark => Self::dark(),
            ThemeName::Plain => Self::plain_for(terminal_bg),
            ThemeName::Glass => Self::glass(),
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

fn usable_background(terminal_bg: Option<Rgb>) -> Rgb {
    terminal_bg
        .filter(|bg| is_dark(*bg))
        .unwrap_or(ASSUMED_BACKGROUND)
}

/// Lifting needs a dark terminal: a lighter background cannot host a panel under light text.
fn is_dark((red, green, blue): Rgb) -> bool {
    // Rec. 601 luma weights.
    let luma = 299 * u32::from(red) + 587 * u32::from(green) + 114 * u32::from(blue);
    luma < 64 * 1000
}

/// Move each channel toward the background's own bright end, so panels keep the background's hue.
fn lifted((red, green, blue): Rgb, lift: f32) -> Color {
    let peak = red.max(green).max(blue);
    let anchor = if peak == 0 {
        // Black has no hue to keep.
        [u8::MAX; 3]
    } else {
        let scale = 255.0 / f32::from(peak);
        [red, green, blue].map(|channel| (f32::from(channel) * scale).round() as u8)
    };
    let channel = |value: u8, anchor: u8| {
        let value = f32::from(value);
        (value + (f32::from(anchor) - value) * lift).round() as u8
    };
    Color::Rgb(
        channel(red, anchor[0]),
        channel(green, anchor[1]),
        channel(blue, anchor[2]),
    )
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

    fn channels(color: Color) -> Rgb {
        let Color::Rgb(red, green, blue) = color else {
            panic!("expected an rgb surface, got {color:?}");
        };
        (red, green, blue)
    }

    fn channel_sum(color: Color) -> u32 {
        let (red, green, blue) = channels(color);
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
    fn plain_theme_lifts_panels_above_a_detected_background() {
        let background: Rgb = (26, 27, 38);
        let plain = Theme::plain_for(Some(background));
        let background = channel_sum(Color::Rgb(background.0, background.1, background.2));

        for surface in [plain.surface_bg, plain.element_bg, plain.elevated_bg] {
            assert!(
                channel_sum(surface) > background,
                "{surface:?} should sit above the terminal background"
            );
        }
        assert!(channel_sum(plain.surface_bg) < channel_sum(plain.element_bg));
        assert!(channel_sum(plain.element_bg) < channel_sum(plain.elevated_bg));
    }

    #[test]
    fn plain_assumes_a_dark_background_when_there_is_none_to_use() {
        let assumed = Theme::plain_for(None);

        assert!(channel_sum(assumed.element_bg) > channel_sum(Color::Rgb(24, 25, 34)));
        assert_eq!(Theme::plain_for(Some((255, 255, 255))), assumed);
        assert_eq!(Theme::plain_for(Some((250, 250, 250))), assumed);
        assert_eq!(Theme::plain_for(Some((120, 120, 120))), assumed);
    }

    #[test]
    fn plain_keeps_the_background_hue_in_its_panels() {
        let background: Rgb = (26, 27, 38);
        let plain = Theme::plain_for(Some(background));

        let cast = background.2 - background.0;

        for surface in [plain.surface_bg, plain.element_bg, plain.elevated_bg] {
            let channels = channels(surface);
            assert!(
                channels.2 - channels.0 >= cast,
                "{channels:?} should keep the background's blue cast"
            );
        }
    }

    #[test]
    fn glass_theme_leaves_every_surface_to_the_terminal() {
        let glass = Theme::for_name(ThemeName::Glass, 0, Some((26, 27, 38)));

        for surface in [
            glass.root_bg,
            glass.surface_bg,
            glass.element_bg,
            glass.elevated_bg,
        ] {
            assert_eq!(surface, Color::Reset);
        }
        assert!(!glass.paints_panels());
        assert!(Theme::plain_for(None).paints_panels());
    }

    #[test]
    fn canvas_stays_concrete_without_a_painted_root() {
        assert_eq!(Theme::dark().canvas(), Theme::dark().root_bg);
        assert_eq!(Theme::plain_for(None).canvas(), Color::Rgb(0, 0, 0));
        assert_eq!(Theme::glass().canvas(), Color::Rgb(0, 0, 0));
    }

    #[test]
    fn plain_lifts_panels_off_a_black_background() {
        let plain = Theme::plain_for(Some((0, 0, 0)));

        for surface in [plain.surface_bg, plain.element_bg, plain.elevated_bg] {
            assert!(
                channel_sum(surface) > 0,
                "{surface:?} should not stay black"
            );
        }
    }
}
