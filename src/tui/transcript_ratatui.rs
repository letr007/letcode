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
