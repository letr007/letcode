//! Mermaid facade: classify diagrams and expose neutral rendered spans.

mod canvas;
mod class;
mod class_ir;
mod er;
mod er_ir;
mod flowchart;
mod flowchart_ir;
mod flowchart_parser;
mod gantt;
mod gantt_ir;
mod mindmap;
mod routing;
mod sequence;
mod sequence_ir;
mod state;
mod state_ir;
mod timeline;

const MAX_SOURCE_CHARS: usize = 16_384;
const MAX_SOURCE_LINES: usize = 512;
const MAX_RENDER_LINES: usize = 1_024;

fn source_within_limits(source: &str) -> bool {
    source.chars().count() <= MAX_SOURCE_CHARS && source.lines().count() <= MAX_SOURCE_LINES
}

fn render_line_count_within_limits(line_count: usize) -> bool {
    line_count <= MAX_RENDER_LINES
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct MermaidSourceSpan {
    pub(crate) start: usize,
    pub(crate) end: usize,
}

impl MermaidSourceSpan {
    pub(crate) const fn new(start: usize, end: usize) -> Self {
        Self { start, end }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct MermaidRender {
    pub(crate) lines: Vec<Vec<MermaidRenderSpan>>,
}
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct MermaidRenderSpan {
    pub(crate) text: String,
    pub(crate) source: Option<MermaidSourceSpan>,
    pub(crate) atomic: bool,
}
impl MermaidRenderSpan {
    fn decoration(text: impl Into<String>) -> Self {
        Self {
            text: text.into(),
            source: None,
            atomic: false,
        }
    }
    fn source(text: impl Into<String>, source: MermaidSourceSpan, atomic: bool) -> Self {
        Self {
            text: text.into(),
            source: Some(source),
            atomic,
        }
    }
}

pub(crate) fn render(source: &str, width: usize) -> Option<MermaidRender> {
    if !source_within_limits(source) {
        return None;
    }
    let first = source.lines().next()?;
    let lines = match first {
        "sequenceDiagram" => sequence::render(source, width)?,
        "classDiagram" => class::render(source, width)?,
        "erDiagram" => er::render(source, width)?,
        "gantt" => gantt::render(source, width)?,
        "mindmap" => mindmap::render(source, width)?,
        "timeline" | "timeline LR" => timeline::render(source, width)?,
        "stateDiagram" | "stateDiagram-v2" => state::render(source, width)?,
        header
            if header == "graph TD"
                || header.starts_with("graph ")
                || header == "flowchart TD"
                || header.starts_with("flowchart ") =>
        {
            flowchart::render(source, width)?
        }
        _ => return None,
    };
    Some(MermaidRender { lines })
}

#[cfg(test)]
mod tests;
