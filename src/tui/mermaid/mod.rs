//! Mermaid facade: classify diagrams and expose neutral rendered spans.

mod canvas;
mod flowchart;
mod flowchart_ir;
mod flowchart_parser;
mod sequence;
mod sequence_ir;

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
    let first = source.lines().next()?.trim();
    let lines = if first == "sequenceDiagram" {
        sequence::render(source, width)?
    } else {
        flowchart::render(source, width)?
    };
    Some(MermaidRender { lines })
}

#[cfg(test)]
mod tests;
