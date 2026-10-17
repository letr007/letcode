use std::collections::VecDeque;
use std::time::{Duration, Instant};

use ratatui::style::{Color, Style};
use unicode_segmentation::UnicodeSegmentation;

use crate::tui::theme::Theme;
use crate::tui::transcript_render::{CopyMode, Document, SourceRange, Span};

pub(crate) const ASSISTANT_FADE_DURATION: Duration = Duration::from_millis(200);
const ASSISTANT_FADE_MIN_BRIGHTNESS: f32 = 0.3;

pub(crate) struct AssistantFade<'a> {
    pub(crate) reveals: &'a VecDeque<(Instant, u32)>,
    pub(crate) now: Instant,
    pub(crate) backdrop: Color,
}

pub(crate) fn fade_backdrop(theme: Theme, terminal_bg: Option<(u8, u8, u8)>) -> Color {
    match theme.root_bg {
        Color::Rgb(..) => theme.root_bg,
        _ => terminal_bg
            .map(|(red, green, blue)| Color::Rgb(red, green, blue))
            .unwrap_or_else(|| theme.canvas()),
    }
}

pub(crate) fn apply_fade(document: &mut Document<Style>, fade: AssistantFade<'_>) {
    let Some(backdrop) = rgb(fade.backdrop) else {
        return;
    };
    let mut index = 0usize;
    let mut done = false;
    for line in document.lines.iter_mut().rev() {
        if done {
            break;
        }
        let spans = std::mem::take(&mut line.spans);
        let mut rebuilt = Vec::with_capacity(spans.len());
        for span in spans.into_iter().rev() {
            if done {
                rebuilt.push(span);
                continue;
            }
            let pieces = fade_span(span, &mut index, &fade, backdrop, &mut done);
            rebuilt.extend(pieces.into_iter().rev());
        }
        rebuilt.reverse();
        line.spans = rebuilt;
    }
}

fn fade_span(
    span: Span<Style>,
    index: &mut usize,
    fade: &AssistantFade<'_>,
    backdrop: (u8, u8, u8),
    done: &mut bool,
) -> Vec<Span<Style>> {
    if span.source.is_none() || span.text.is_empty() {
        return vec![span];
    }
    let graphemes: Vec<&str> = span.text.graphemes(true).collect();
    let length = graphemes.len();
    let base = *index;
    *index += length;

    if span.copy_mode == CopyMode::Atomic {
        return match weight(fade, base) {
            Some(weight) if weight < 1.0 => {
                let mut span = span;
                span.style = faded_style(span.style, backdrop, weight);
                vec![span]
            }
            _ => {
                *done = true;
                vec![span]
            }
        };
    }

    let mut faded = 0usize;
    while faded < length {
        match weight(fade, base + faded) {
            Some(weight) if weight < 1.0 => faded += 1,
            _ => break,
        }
    }
    if faded == 0 {
        *done = true;
        return vec![span];
    }

    let split = length - faded;
    let range = span.source.expect("source span checked above");
    let mut spans = Vec::with_capacity(faded + 1);
    let mut start = range.start;
    if split > 0 {
        let text: String = graphemes[..split].concat();
        let end = start + text.chars().count();
        spans.push(Span::source_with_mode(
            text,
            span.style,
            SourceRange::new(range.block_index, start, end),
            span.copy_mode,
            span.copy_join,
            span.interaction.clone(),
        ));
        start = end;
    }
    for (offset, grapheme) in graphemes[split..].iter().enumerate() {
        let end = start + grapheme.chars().count();
        let weight = weight(fade, base + length - 1 - (split + offset)).unwrap_or(1.0);
        spans.push(Span::source_with_mode(
            (*grapheme).to_string(),
            faded_style(span.style, backdrop, weight),
            SourceRange::new(range.block_index, start, end),
            span.copy_mode,
            span.copy_join,
            span.interaction.clone(),
        ));
        start = end;
    }
    spans
}

fn weight(fade: &AssistantFade<'_>, index: usize) -> Option<f32> {
    let mut remaining = index;
    for (revealed_at, count) in fade.reveals.iter().rev() {
        let count = *count as usize;
        if remaining < count {
            let age = fade.now.saturating_duration_since(*revealed_at);
            if age >= ASSISTANT_FADE_DURATION {
                return Some(1.0);
            }
            let progress = age.as_secs_f32() / ASSISTANT_FADE_DURATION.as_secs_f32();
            return Some(
                ASSISTANT_FADE_MIN_BRIGHTNESS + (1.0 - ASSISTANT_FADE_MIN_BRIGHTNESS) * progress,
            );
        }
        remaining -= count;
    }
    None
}

fn faded_style(style: Style, backdrop: (u8, u8, u8), weight: f32) -> Style {
    let Some(Color::Rgb(red, green, blue)) = style.fg else {
        return style;
    };
    Style {
        fg: Some(Color::Rgb(
            mix(backdrop.0, red, weight),
            mix(backdrop.1, green, weight),
            mix(backdrop.2, blue, weight),
        )),
        ..style
    }
}

fn mix(backdrop: u8, foreground: u8, weight: f32) -> u8 {
    (f32::from(backdrop) + (f32::from(foreground) - f32::from(backdrop)) * weight)
        .round()
        .clamp(0.0, 255.0) as u8
}

fn rgb(color: Color) -> Option<(u8, u8, u8)> {
    match color {
        Color::Rgb(red, green, blue) => Some((red, green, blue)),
        _ => None,
    }
}

#[cfg(test)]
#[path = "../../tests/internal/tui/assistant_fade/tests.rs"]
mod tests;
