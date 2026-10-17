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
    pub card_frame: bool,
    pub light: bool,
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
            card_frame: false,
            light: false,
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

    pub const fn canvas(self) -> Color {
        match self.root_bg {
            Color::Reset => Color::Rgb(0, 0, 0),
            color => color,
        }
    }

    pub fn plain_for(terminal_bg: Option<Rgb>) -> Self {
        let background = usable_background(terminal_bg);
        let mut theme = Self::dark();
        theme.root_bg = Color::Reset;
        theme.surface_bg = lifted(background, SURFACE_LIFT);
        theme.element_bg = lifted(background, ELEMENT_LIFT);
        theme.elevated_bg = lifted(background, ELEVATED_LIFT);
        theme
    }

    pub const fn glass() -> Self {
        let mut theme = Self::dark();
        theme.root_bg = Color::Reset;
        theme.surface_bg = Color::Reset;
        theme.element_bg = Color::Reset;
        theme.elevated_bg = Color::Reset;
        theme
    }

    pub const fn wireframe() -> Self {
        let mut theme = Self::glass();
        theme.card_frame = true;
        theme.border = Color::Rgb(0x70, 0x80, 0x9a);
        theme.text = Color::Rgb(0xde, 0xe7, 0xf2);
        theme.muted_text = Color::Rgb(0x9b, 0xa8, 0xbb);
        theme.dim_text = Color::Rgb(0x66, 0x73, 0x88);
        theme.on_accent = Color::Rgb(0x11, 0x18, 0x21);
        theme.accent = Color::Rgb(0x84, 0xc8, 0xff);
        theme.assistant = Color::Rgb(0x8b, 0xd7, 0xaa);
        theme.user = Color::Rgb(0x8c, 0xca, 0xff);
        theme.success = Color::Rgb(0x8e, 0xd9, 0xa6);
        theme.warning = Color::Rgb(0xe2, 0xc1, 0x77);
        theme.error = Color::Rgb(0xf0, 0x8b, 0x92);
        theme.approval = Color::Rgb(0xe9, 0xc2, 0x71);
        theme.notice = Color::Rgb(0x9a, 0xc9, 0xda);
        theme.fake = Color::Rgb(0xd8, 0xa1, 0xed);
        theme
    }

    pub const fn light() -> Self {
        let mut theme = Self::dark();
        theme.light = true;
        theme.root_bg = Color::Rgb(0xf7, 0xf7, 0xf7);
        theme.surface_bg = Color::Rgb(0xf0, 0xf0, 0xf2);
        theme.element_bg = Color::Rgb(0xe9, 0xe9, 0xeb);
        theme.elevated_bg = Color::Rgb(0xe0, 0xe0, 0xe2);
        theme.border = Color::Rgb(0xc9, 0xc9, 0xce);
        theme.text = Color::Rgb(0x28, 0x28, 0x28);
        theme.muted_text = Color::Rgb(0x7d, 0x7d, 0x7d);
        theme.dim_text = Color::Rgb(0xaf, 0xaf, 0xaf);
        theme.on_accent = Color::Rgb(0xf7, 0xf7, 0xf7);
        theme.accent = Color::Rgb(0x14, 0x73, 0x9b);
        theme.assistant = Color::Rgb(0x23, 0x87, 0x46);
        theme.user = Color::Rgb(0x14, 0x73, 0x9b);
        theme.success = Color::Rgb(0x23, 0x82, 0x32);
        theme.warning = Color::Rgb(0x8c, 0x78, 0x14);
        theme.error = Color::Rgb(0xb9, 0x28, 0x28);
        theme.approval = Color::Rgb(0xa0, 0x6e, 0x0a);
        theme.notice = Color::Rgb(0x6e, 0x6e, 0x6e);
        theme.fake = Color::Rgb(0xa0, 0x32, 0xbe);
        theme.diff_add_bg = Color::Rgb(0xde, 0xf0, 0xe4);
        theme.diff_delete_bg = Color::Rgb(0xfa, 0xe2, 0xe6);
        theme.diff_hunk_bg = Color::Rgb(0xde, 0xec, 0xfa);
        theme
    }

    pub const fn paints_panels(self) -> bool {
        !matches!(self.element_bg, Color::Reset)
    }

    pub fn for_name(name: ThemeName, frame: usize, terminal_bg: Option<Rgb>) -> Self {
        match name {
            ThemeName::Dark => Self::dark(),
            ThemeName::Plain => Self::plain_for(terminal_bg),
            ThemeName::Glass => Self::glass(),
            ThemeName::Wireframe => Self::wireframe(),
            ThemeName::Rainbow => Self::dark().with_rainbow_accent(frame),
            ThemeName::Light => Self::light(),
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

/// Kept low: a brighter background cannot host lifted panels under the palette's light text.
fn is_dark((red, green, blue): Rgb) -> bool {
    // Rec. 601 luma weights.
    let luma = 299 * u32::from(red) + 587 * u32::from(green) + 114 * u32::from(blue);
    luma < 64 * 1000
}

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
#[path = "../../tests/internal/tui/theme/tests.rs"]
mod tests;
