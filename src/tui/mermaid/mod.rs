//! Mermaid facade: classify diagrams and expose neutral rendered spans.

use std::{
    collections::VecDeque,
    sync::{Mutex, OnceLock},
};

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
mod gitgraph;
mod gitgraph_ir;
mod journey;
mod journey_ir;
mod mindmap;
mod pie;
mod pie_ir;
mod routing;
mod sequence;
mod sequence_ir;
mod state;
mod state_ir;
mod timeline;

const MAX_SOURCE_CHARS: usize = 16_384;
const MAX_SOURCE_LINES: usize = 512;
const MAX_RENDER_LINES: usize = 1_024;
const MAX_RENDER_CACHE_ENTRIES: usize = 32;

fn source_within_limits(source: &str) -> bool {
    source.chars().count() <= MAX_SOURCE_CHARS && source.lines().count() <= MAX_SOURCE_LINES
}

fn render_line_count_within_limits(line_count: usize) -> bool {
    line_count <= MAX_RENDER_LINES
}

fn render_math_label(label: &str) -> Option<(String, bool)> {
    let mut rendered = String::new();
    let mut cursor = 0;
    let mut had_math = false;
    while let Some(relative_start) = label[cursor..].find("$$") {
        let start = cursor + relative_start;
        rendered.push_str(&label[cursor..start]);
        let content_start = start + 2;
        let relative_end = label[content_start..].find("$$")?;
        let end = content_start + relative_end;
        let source = &label[content_start..end];
        if source.is_empty() {
            return None;
        }
        let math = crate::tui::math::render_text(source, false)?;
        if math.trim().is_empty() || math.contains('\n') {
            return None;
        }
        rendered.push_str(&math);
        had_math = true;
        cursor = end + 2;
    }
    if !had_math {
        return Some((label.to_string(), false));
    }
    rendered.push_str(&label[cursor..]);
    Some((rendered, true))
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
    if let Some(rendered) = render_cache()
        .lock()
        .unwrap_or_else(|error| error.into_inner())
        .get(source, width)
    {
        return Some(rendered);
    }

    let rendered = render_uncached(source, width)?;
    render_cache()
        .lock()
        .unwrap_or_else(|error| error.into_inner())
        .insert(source, width, rendered.clone());
    Some(rendered)
}

fn render_uncached(source: &str, width: usize) -> Option<MermaidRender> {
    let first = source.lines().next()?;
    let lines = match first {
        "sequenceDiagram" => sequence::render(source, width)?,
        "classDiagram" => class::render(source, width)?,
        "erDiagram" => er::render(source, width)?,
        "gantt" => gantt::render(source, width)?,
        "gitGraph" => gitgraph::render(source, width)?,
        "journey" => journey::render(source, width)?,
        "mindmap" => mindmap::render(source, width)?,
        header if header == "pie" || header.starts_with("pie ") => pie::render(source, width)?,
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

#[derive(Default)]
struct MermaidRenderCache {
    entries: VecDeque<(String, usize, MermaidRender)>,
}

impl MermaidRenderCache {
    fn get(&mut self, source: &str, width: usize) -> Option<MermaidRender> {
        let index = self
            .entries
            .iter()
            .position(|(cached_source, cached_width, _)| {
                cached_source == source && *cached_width == width
            })?;
        let entry = self.entries.remove(index)?;
        let rendered = entry.2.clone();
        self.entries.push_back(entry);
        Some(rendered)
    }

    fn insert(&mut self, source: &str, width: usize, rendered: MermaidRender) {
        if self.entries.len() == MAX_RENDER_CACHE_ENTRIES {
            self.entries.pop_front();
        }
        self.entries
            .push_back((source.to_string(), width, rendered));
    }
}

fn render_cache() -> &'static Mutex<MermaidRenderCache> {
    static CACHE: OnceLock<Mutex<MermaidRenderCache>> = OnceLock::new();
    CACHE.get_or_init(|| Mutex::new(MermaidRenderCache::default()))
}

#[cfg(test)]
mod tests;
