//! Ratatui bridge for renderer-neutral transcript layout.

use std::{
    collections::HashSet,
    io::{self, Write},
};

use ratatui::{
    backend::{Backend, CrosstermBackend},
    buffer::{Buffer, Cell},
    layout::Rect,
    text::{Line, Span},
};
use unicode_segmentation::UnicodeSegmentation;

use crate::tui::transcript_render;

/// Preserve the exact Ratatui styles stored in the neutral layout document.
pub fn line_to_ratatui(line: &transcript_render::Line<ratatui::style::Style>) -> Line<'static> {
    Line::from(
        line.spans
            .iter()
            .map(|span| Span::styled(span.text.clone(), span.style))
            .collect::<Vec<_>>(),
    )
}

pub fn document_to_ratatui(
    document: &transcript_render::Document<ratatui::style::Style>,
) -> Vec<Line<'static>> {
    document.lines.iter().map(line_to_ratatui).collect()
}

/// A visible plain-text cell that should receive a terminal-native hyperlink after Ratatui flushes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HyperlinkCell {
    position: (u16, u16),
    url: String,
    expected: Vec<Cell>,
}

/// Collect hyperlink cells without changing the Ratatui buffer or its display-width accounting.
pub fn collect_hyperlink_cells(
    buffer: &Buffer,
    area: Rect,
    lines: &[Option<&transcript_render::Line<ratatui::style::Style>>],
) -> Vec<HyperlinkCell> {
    let mut cells = Vec::new();
    let mut seen = HashSet::new();
    for (row, line) in lines.iter().enumerate() {
        if row >= area.height as usize {
            break;
        }
        let Some(line) = line else {
            continue;
        };
        let mut column = 0u16;
        for span in &line.spans {
            let width = Line::from(span.text.as_str()).width() as u16;
            if let Some(transcript_render::Interaction::OpenUrl(url)) = &span.interaction
                && safe_hyperlink_url(url)
                && width > 0
                && column < area.width
            {
                let mut grapheme_column = column;
                for grapheme in span.text.graphemes(true) {
                    let grapheme_width = Line::from(grapheme).width() as u16;
                    if grapheme_width == 0 {
                        continue;
                    }
                    if grapheme_column >= area.width {
                        break;
                    }
                    let position = (area.x + grapheme_column, area.y + row as u16);
                    if seen.insert(position) {
                        let expected = cell_footprint(
                            buffer,
                            position,
                            grapheme_width.min(area.width - grapheme_column),
                        );
                        if !expected.is_empty() {
                            cells.push(HyperlinkCell {
                                position,
                                url: url.clone(),
                                expected,
                            });
                        }
                    }
                    grapheme_column = grapheme_column.saturating_add(grapheme_width);
                }
            }
            column = column.saturating_add(width);
        }
    }
    cells
}

#[derive(Debug, Default)]
pub struct HyperlinkOverlayPlan {
    plain: Vec<((u16, u16), Cell)>,
    linked: Vec<HyperlinkCell>,
    pub applied: Vec<HyperlinkCell>,
}

/// Repaint previous hyperlink cells as plain text, then apply links that still match the completed
/// frame. Matching the captured cell prevents later dialogs and toasts from becoming hyperlinks.
pub fn plan_hyperlink_overlay(
    buffer: &Buffer,
    previous: &[HyperlinkCell],
    desired: &[HyperlinkCell],
) -> HyperlinkOverlayPlan {
    let mut plan = HyperlinkOverlayPlan::default();
    let mut plain_seen = HashSet::new();
    for hyperlink in previous {
        let width = hyperlink.expected.len() as u16;
        for (position, cell) in leading_cells_intersecting(
            buffer,
            hyperlink.position.0,
            hyperlink.position.0.saturating_add(width),
            hyperlink.position.1,
        ) {
            if plain_seen.insert(position) {
                plan.plain.push((position, cell));
            }
        }
    }

    let mut linked_seen = HashSet::new();
    for hyperlink in desired {
        if !linked_seen.insert(hyperlink.position) || !safe_hyperlink_url(&hyperlink.url) {
            continue;
        }
        let current = cell_footprint(buffer, hyperlink.position, hyperlink.expected.len() as u16);
        if current == hyperlink.expected {
            let applied = HyperlinkCell {
                position: hyperlink.position,
                url: hyperlink.url.clone(),
                expected: current,
            };
            plan.linked.push(applied.clone());
            plan.applied.push(applied);
        }
    }
    plan
}

/// Write the overlay only after Ratatui has flushed its plain buffer. Each cell closes its OSC 8
/// sequence immediately, so unrelated terminal output can never inherit the link.
pub fn write_hyperlink_overlay<W: Write>(
    backend: &mut CrosstermBackend<W>,
    plan: &HyperlinkOverlayPlan,
) -> io::Result<()> {
    for (position, cell) in &plan.plain {
        backend.draw(std::iter::once((position.0, position.1, cell)))?;
    }
    for hyperlink in &plan.linked {
        if let Err(error) = backend.write_all(format!("\x1b]8;;{}\x07", hyperlink.url).as_bytes()) {
            let _ = backend.write_all(b"\x1b]8;;\x07");
            return Err(error);
        }
        let draw_result = backend.draw(std::iter::once((
            hyperlink.position.0,
            hyperlink.position.1,
            &hyperlink.expected[0],
        )));
        let close_result = backend.write_all(b"\x1b]8;;\x07");
        draw_result?;
        close_result?;
    }
    Write::flush(backend)
}

fn cell_footprint(buffer: &Buffer, position: (u16, u16), width: u16) -> Vec<Cell> {
    (0..width)
        .filter_map(|offset| buffer.cell((position.0.saturating_add(offset), position.1)))
        .cloned()
        .collect()
}

fn leading_cells_intersecting(
    buffer: &Buffer,
    start: u16,
    end: u16,
    row: u16,
) -> Vec<((u16, u16), Cell)> {
    let mut cells = Vec::new();
    let mut column = buffer.area.x;
    while column < buffer.area.right() && column < end {
        let Some(cell) = buffer.cell((column, row)) else {
            break;
        };
        let width = (Line::from(cell.symbol()).width() as u16).max(1);
        if column.saturating_add(width) > start {
            cells.push(((column, row), cell.clone()));
        }
        column = column.saturating_add(width);
    }
    cells
}

pub(crate) fn safe_hyperlink_url(url: &str) -> bool {
    !url.chars().any(char::is_control)
        && matches!(
            url.split_once(':').map(|(scheme, _)| scheme),
            Some("http" | "https")
        )
}

/// Open a previously-validated http(s) URL in the platform browser.
///
/// Mouse capture owns clicks in this TUI, so OSC 8 alone cannot open links; the
/// click handler must call this. Spawns and returns — never blocks the UI thread.
pub(crate) fn open_hyperlink_url(url: &str) -> io::Result<()> {
    if !safe_hyperlink_url(url) {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "refusing to open non-http(s) hyperlink",
        ));
    }

    // ponytail: std process spawn per platform; swap for `open` crate if we need
    // WSL/flatpak edge cases later.
    let mut command = {
        #[cfg(target_os = "macos")]
        {
            let mut command = std::process::Command::new("open");
            command.arg(url);
            command
        }
        #[cfg(target_os = "windows")]
        {
            let mut command = std::process::Command::new("cmd");
            command.args(["/c", "start", "", url]);
            command
        }
        #[cfg(not(any(target_os = "macos", target_os = "windows")))]
        {
            let mut command = std::process::Command::new("xdg-open");
            command.arg(url);
            command
        }
    };
    command.stdin(std::process::Stdio::null());
    command.stdout(std::process::Stdio::null());
    command.stderr(std::process::Stdio::null());
    command.spawn().map(|_| ())
}

#[cfg(test)]
mod tests {
    use super::*;
    use ratatui::style::Style;

    #[derive(Default)]
    struct FailOnceAfterOpeningLink {
        output: Vec<u8>,
        fail_next_write: bool,
    }

    impl Write for FailOnceAfterOpeningLink {
        fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
            if self.fail_next_write {
                self.fail_next_write = false;
                return Err(io::Error::other("injected overlay write failure"));
            }
            self.output.extend_from_slice(bytes);
            if bytes.starts_with(b"\x1b]8;;http") {
                self.fail_next_write = true;
            }
            Ok(bytes.len())
        }

        fn flush(&mut self) -> io::Result<()> {
            Ok(())
        }
    }

    fn linked_document(url: &str) -> transcript_render::Document<Style> {
        let mut document = transcript_render::Document::default();
        let block = document.add_source("docs");
        document.push_line(
            transcript_render::Line {
                spans: vec![transcript_render::Span::source_with_interaction(
                    "docs",
                    Style::default(),
                    transcript_render::SourceRange::new(block, 0, 4),
                    transcript_render::CopyJoin::Concat,
                    Some(transcript_render::Interaction::OpenUrl(url.into())),
                )],
            },
            transcript_render::Break::End,
        );
        document
    }

    #[test]
    fn osc8_is_written_after_layout_without_changing_buffer_symbols() {
        let document = linked_document("https://example.test");
        let area = Rect::new(0, 0, 8, 1);
        let mut buffer = Buffer::empty(area);
        buffer.set_line(0, 0, &line_to_ratatui(&document.lines[0]), area.width);

        let desired = collect_hyperlink_cells(&buffer, area, &[Some(&document.lines[0])]);
        let plan = plan_hyperlink_overlay(&buffer, &[], &desired);
        let mut output = Vec::new();
        {
            let mut backend = CrosstermBackend::new(&mut output);
            write_hyperlink_overlay(&mut backend, &plan).expect("write overlay");
        }
        let output = String::from_utf8(output).expect("utf-8 terminal output");

        assert_eq!(buffer[(0, 0)].symbol(), "d");
        assert_eq!(buffer[(3, 0)].symbol(), "s");
        assert_eq!(plan.applied.len(), 4);
        assert!(output.contains("\x1b]8;;https://example.test\x07"));
        assert!(output.contains("\x1b]8;;\x07"));
    }

    #[test]
    fn hyperlink_is_closed_when_drawing_the_linked_cell_fails() {
        let document = linked_document("https://example.test");
        let area = Rect::new(0, 0, 8, 1);
        let mut buffer = Buffer::empty(area);
        buffer.set_line(0, 0, &line_to_ratatui(&document.lines[0]), area.width);
        let desired = collect_hyperlink_cells(&buffer, area, &[Some(&document.lines[0])]);
        let plan = plan_hyperlink_overlay(&buffer, &[], &desired);
        let mut writer = FailOnceAfterOpeningLink::default();

        let error = {
            let mut backend = CrosstermBackend::new(&mut writer);
            write_hyperlink_overlay(&mut backend, &plan).expect_err("draw must fail")
        };

        assert_eq!(error.kind(), io::ErrorKind::Other);
        assert!(writer.output.ends_with(b"\x1b]8;;\x07"));
    }

    #[test]
    fn refuses_unsafe_hyperlink_targets() {
        let area = Rect::new(0, 0, 8, 1);
        for url in ["https://bad\x1b.test", "file:///tmp/unsafe"] {
            let document = linked_document(url);
            let mut buffer = Buffer::empty(area);
            buffer.set_line(0, 0, &line_to_ratatui(&document.lines[0]), area.width);

            assert!(collect_hyperlink_cells(&buffer, area, &[Some(&document.lines[0])]).is_empty());
            assert_eq!(buffer[(0, 0)].symbol(), "d");
            assert!(open_hyperlink_url(url).is_err());
        }
    }

    #[test]
    fn stale_links_are_repainted_plain_and_covered_links_are_not_applied() {
        let document = linked_document("https://example.test");
        let area = Rect::new(0, 0, 8, 1);
        let mut buffer = Buffer::empty(area);
        buffer.set_line(0, 0, &line_to_ratatui(&document.lines[0]), area.width);
        let previous = collect_hyperlink_cells(&buffer, area, &[Some(&document.lines[0])]);
        buffer[(0, 0)].set_symbol("X");

        let plan = plan_hyperlink_overlay(&buffer, &previous, &previous);

        assert_eq!(plan.plain.len(), 4);
        assert_eq!(plan.linked.len(), 3);
        assert_eq!(plan.applied.len(), 3);
    }

    #[test]
    fn hyperlink_collection_skips_wide_character_continuation_cells() {
        let mut document = transcript_render::Document::default();
        let block = document.add_source("界a");
        document.push_line(
            transcript_render::Line {
                spans: vec![transcript_render::Span::source_with_interaction(
                    "界a",
                    Style::default(),
                    transcript_render::SourceRange::new(block, 0, 2),
                    transcript_render::CopyJoin::Concat,
                    Some(transcript_render::Interaction::OpenUrl(
                        "https://example.test".into(),
                    )),
                )],
            },
            transcript_render::Break::End,
        );
        let area = Rect::new(0, 0, 4, 1);
        let mut buffer = Buffer::empty(area);
        buffer.set_line(0, 0, &line_to_ratatui(&document.lines[0]), area.width);

        let cells = collect_hyperlink_cells(&buffer, area, &[Some(&document.lines[0])]);

        assert_eq!(buffer[(0, 0)].symbol(), "界");
        assert_eq!(buffer[(1, 0)].symbol(), " ");
        assert_eq!(buffer[(2, 0)].symbol(), "a");
        assert_eq!(cells.len(), 2);
    }
}
