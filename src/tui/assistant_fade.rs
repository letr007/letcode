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
mod tests {
    use super::*;
    use crate::tui::transcript_render::{Break, Line as RenderLine};

    fn full_style() -> Style {
        Style::default().fg(Color::Rgb(220, 220, 220))
    }

    fn document(text: &str) -> Document<Style> {
        let mut document = Document::default();
        let block = document.add_source(text.to_string());
        let range = SourceRange::new(block, 0, text.chars().count());
        document.push_line(
            RenderLine {
                spans: vec![Span::source(text.to_string(), full_style(), range)],
            },
            Break::End,
        );
        document
    }

    fn red(span: &Span<Style>) -> u8 {
        match span.style.fg {
            Some(Color::Rgb(red, _, _)) => red,
            other => panic!("expected an rgb foreground, got {other:?}"),
        }
    }

    #[test]
    fn fresh_batch_dims_the_tail_most() {
        let now = Instant::now();
        let mut document = document("abcd");
        let reveals = VecDeque::from([(now - Duration::from_millis(150), 2u32), (now, 2u32)]);
        apply_fade(
            &mut document,
            AssistantFade {
                reveals: &reveals,
                now,
                backdrop: Color::Rgb(0, 0, 0),
            },
        );

        assert!(document.validate());
        let spans = &document.lines[0].spans;
        assert_eq!(
            spans
                .iter()
                .map(|span| span.text.as_str())
                .collect::<Vec<_>>(),
            ["a", "b", "c", "d"]
        );
        assert!(red(&spans[0]) > red(&spans[2]));
        assert_eq!(red(&spans[2]), red(&spans[3]));
    }

    #[test]
    fn expired_batch_leaves_text_untouched() {
        let now = Instant::now();
        let mut document = document("abcd");
        let reveals = VecDeque::from([(now - Duration::from_millis(250), 4u32)]);
        apply_fade(
            &mut document,
            AssistantFade {
                reveals: &reveals,
                now,
                backdrop: Color::Rgb(0, 0, 0),
            },
        );

        assert_eq!(document.lines[0].spans.len(), 1);
        assert_eq!(document.lines[0].spans[0].style, full_style());
    }

    #[test]
    fn decoration_spans_are_neither_faded_nor_counted() {
        let now = Instant::now();
        let mut document = document("ab");
        document.lines[0].spans.push(Span::decoration(
            "  ",
            Style::default().fg(Color::Rgb(80, 80, 80)),
        ));
        let reveals = VecDeque::from([(now, 2u32)]);
        apply_fade(
            &mut document,
            AssistantFade {
                reveals: &reveals,
                now,
                backdrop: Color::Rgb(0, 0, 0),
            },
        );

        let spans = &document.lines[0].spans;
        assert_eq!(spans.len(), 3);
        assert_eq!(spans[0].text, "a");
        assert_eq!(spans[1].text, "b");
        assert_eq!(spans[2].text, "  ");
        assert_eq!(spans[2].style.fg, Some(Color::Rgb(80, 80, 80)));
    }

    #[test]
    fn atomic_spans_fade_without_splitting() {
        let now = Instant::now();
        let mut document = Document::default();
        let block = document.add_source("row".to_string());
        document.push_line(
            RenderLine {
                spans: vec![Span::source_atomic(
                    "row",
                    full_style(),
                    SourceRange::new(block, 0, 3),
                )],
            },
            Break::End,
        );
        let reveals = VecDeque::from([(now, 3u32)]);
        apply_fade(
            &mut document,
            AssistantFade {
                reveals: &reveals,
                now,
                backdrop: Color::Rgb(0, 0, 0),
            },
        );

        assert!(document.validate());
        assert_eq!(document.lines[0].spans.len(), 1);
        assert_ne!(document.lines[0].spans[0].style, full_style());
    }

    #[test]
    fn backdrop_prefers_the_theme_root_then_the_detected_terminal() {
        let dark = Theme::dark();
        assert_eq!(fade_backdrop(dark, Some((1, 2, 3))), dark.root_bg);

        let glass = Theme::glass();
        assert_eq!(fade_backdrop(glass, Some((1, 2, 3))), Color::Rgb(1, 2, 3));
        assert_eq!(fade_backdrop(glass, None), glass.canvas());
    }
}
