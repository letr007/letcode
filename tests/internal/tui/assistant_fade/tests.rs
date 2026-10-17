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
