//! Ratatui bridge for renderer-neutral transcript layout.

use ratatui::text::{Line, Span};

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
