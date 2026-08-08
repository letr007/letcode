//! Renderer-neutral transcript layout primitives.
//!
//! The core deliberately contains no terminal-backend types. `S` is supplied by a
//! bridge, so callers can retain a backend's exact style value without coupling the
//! document model to that backend.

use unicode_segmentation::UnicodeSegmentation;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SourceRange {
    pub block_index: usize,
    /// Unicode scalar offsets, never byte offsets.
    pub start: usize,
    pub end: usize,
}

impl SourceRange {
    pub const fn new(block_index: usize, start: usize, end: usize) -> Self {
        Self {
            block_index,
            start,
            end,
        }
    }

    pub fn validate(self, source: &str) -> bool {
        self.start < self.end
            && self.end <= source.chars().count()
            && is_grapheme_boundary(source, self.start)
            && is_grapheme_boundary(source, self.end)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum CopyMode {
    #[default]
    Exact,
    /// The source range is copied in full when any visual character is selected.
    Atomic,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum CopyJoin {
    #[default]
    Concat,
    Space,
}

/// A local action associated with a rendered semantic span.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Interaction {
    OpenUrl(String),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Span<S> {
    pub text: String,
    pub style: S,
    /// `None` marks chrome: labels, borders, padding, badges, and animations.
    pub source: Option<SourceRange>,
    /// Controls whether selection copies the selected source slice or its full range.
    pub copy_mode: CopyMode,
    /// Separator inserted before this leaf when copying across distinct sources.
    pub copy_join: CopyJoin,
    /// Semantic local action retained through layout transformations.
    pub interaction: Option<Interaction>,
}

impl<S> Span<S> {
    pub fn decoration(text: impl Into<String>, style: S) -> Self {
        Self {
            text: text.into(),
            style,
            source: None,
            copy_mode: CopyMode::Exact,
            copy_join: CopyJoin::Concat,
            interaction: None,
        }
    }

    /// A copyable leaf must carry the exact range that produced its visible text.
    pub fn source(text: impl Into<String>, style: S, source: SourceRange) -> Self {
        Self::source_with_join(text, style, source, CopyJoin::Concat)
    }

    pub fn source_with_join(
        text: impl Into<String>,
        style: S,
        source: SourceRange,
        copy_join: CopyJoin,
    ) -> Self {
        Self::source_with_interaction(text, style, source, copy_join, None)
    }

    pub fn source_with_interaction(
        text: impl Into<String>,
        style: S,
        source: SourceRange,
        copy_join: CopyJoin,
        interaction: Option<Interaction>,
    ) -> Self {
        Self::source_with_mode(text, style, source, CopyMode::Exact, copy_join, interaction)
    }

    pub fn source_atomic(text: impl Into<String>, style: S, source: SourceRange) -> Self {
        Self::source_with_mode(
            text,
            style,
            source,
            CopyMode::Atomic,
            CopyJoin::Concat,
            None,
        )
    }

    pub fn source_with_mode(
        text: impl Into<String>,
        style: S,
        source: SourceRange,
        copy_mode: CopyMode,
        copy_join: CopyJoin,
        interaction: Option<Interaction>,
    ) -> Self {
        Self {
            text: text.into(),
            style,
            source: Some(source),
            copy_mode,
            copy_join,
            interaction,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Line<S> {
    pub spans: Vec<Span<S>>,
}

/// A semantic leaf assembled by a component before it enters a `Document`.
/// Copyability is declared here; the document creates the matching source block.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SemanticSpan<S> {
    pub text: String,
    pub style: S,
    pub copy: bool,
    pub copy_mode: CopyMode,
    pub copy_join: CopyJoin,
    pub interaction: Option<Interaction>,
}

impl<S> SemanticSpan<S> {
    pub fn decoration(text: impl Into<String>, style: S) -> Self {
        Self {
            text: text.into(),
            style,
            copy: false,
            copy_mode: CopyMode::Exact,
            copy_join: CopyJoin::Concat,
            interaction: None,
        }
    }

    pub fn source(text: impl Into<String>, style: S) -> Self {
        Self::source_with_join(text, style, CopyJoin::Concat)
    }

    pub fn source_with_join(text: impl Into<String>, style: S, copy_join: CopyJoin) -> Self {
        Self::source_with_interaction(text, style, copy_join, None)
    }

    pub fn source_with_interaction(
        text: impl Into<String>,
        style: S,
        copy_join: CopyJoin,
        interaction: Option<Interaction>,
    ) -> Self {
        Self::source_with_mode(text, style, CopyMode::Exact, copy_join, interaction)
    }

    pub fn source_with_mode(
        text: impl Into<String>,
        style: S,
        copy_mode: CopyMode,
        copy_join: CopyJoin,
        interaction: Option<Interaction>,
    ) -> Self {
        Self {
            text: text.into(),
            style,
            copy: true,
            copy_mode,
            copy_join,
            interaction,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SemanticLine<S> {
    pub spans: Vec<SemanticSpan<S>>,
    pub boundary: Break,
}

impl<S> Default for SemanticLine<S> {
    fn default() -> Self {
        Self {
            spans: Vec::new(),
            boundary: Break::End,
        }
    }
}

/// The semantic boundary after a visual line.
///
/// `SoftWrap` is a layout-only boundary and is omitted from copied text. `HardBreak`
/// and `BlockBreak` are author-declared boundaries and become one newline while
/// copying. `End` is valid only for a document's final line.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Break {
    SoftWrap,
    HardBreak,
    BlockBreak,
    #[default]
    End,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SourceBlock {
    pub source: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Document<S> {
    pub source_blocks: Vec<SourceBlock>,
    pub lines: Vec<Line<S>>,
    /// One explicit boundary per line. Ratatui ignores this field; copy and
    /// selection consume it directly.
    pub breaks: Vec<Break>,
}

impl<S> Document<S> {
    pub fn add_source(&mut self, source: impl Into<String>) -> usize {
        self.source_blocks.push(SourceBlock {
            source: source.into(),
        });
        self.source_blocks.len() - 1
    }

    pub fn push_line(&mut self, line: Line<S>, boundary: Break) {
        // A following visual line proves the former terminal boundary was only
        // provisional. Keep the document valid while components are assembled.
        if let Some(previous) = self.breaks.last_mut()
            && *previous == Break::End
        {
            *previous = Break::BlockBreak;
        }
        self.lines.push(line);
        self.breaks.push(boundary);
    }

    /// Insert semantic leaves into the document. Components provide the full
    /// source and exact range; this method only assigns document-local block ids.
    pub fn push_semantic_line(&mut self, line: SemanticLine<S>) {
        let boundary = line.boundary;
        let spans = line
            .spans
            .into_iter()
            .map(|span| {
                if span.copy && !span.text.is_empty() {
                    let end = span.text.chars().count();
                    let block = self.add_source(span.text.clone());
                    Span::source_with_mode(
                        span.text,
                        span.style,
                        SourceRange::new(block, 0, end),
                        span.copy_mode,
                        span.copy_join,
                        span.interaction,
                    )
                } else {
                    Span::decoration(span.text, span.style)
                }
            })
            .collect();
        self.push_line(Line { spans }, boundary);
    }

    /// Append a component document, remapping source blocks without examining
    /// visual text or layout spans.
    pub fn append(&mut self, mut other: Document<S>) {
        if !self.lines.is_empty() && !other.lines.is_empty() {
            self.finish();
            if let Some(boundary) = self.breaks.last_mut() {
                *boundary = Break::BlockBreak;
            }
        }
        let source_base = self.source_blocks.len();
        self.source_blocks.append(&mut other.source_blocks);
        for mut line in other.lines {
            for span in &mut line.spans {
                if let Some(range) = &mut span.source {
                    range.block_index += source_base;
                }
            }
            self.lines.push(line);
        }
        self.breaks.append(&mut other.breaks);
    }

    pub fn break_after(&self, line_index: usize) -> Option<Break> {
        self.breaks.get(line_index).copied()
    }

    pub fn finish(&mut self) {
        if let Some(boundary) = self.breaks.last_mut() {
            *boundary = Break::End;
        }
    }

    /// Renderer-independent integrity check for layout and copy contracts.
    pub fn validate(&self) -> bool {
        if self.lines.len() != self.breaks.len()
            || self
                .breaks
                .last()
                .is_some_and(|boundary| *boundary != Break::End)
            || self.breaks[..self.breaks.len().saturating_sub(1)]
                .iter()
                .any(|boundary| *boundary == Break::End)
        {
            return false;
        }

        self.lines
            .iter()
            .flat_map(|line| &line.spans)
            .all(|span| match span.source {
                None => true,
                Some(range) => self
                    .source_blocks
                    .get(range.block_index)
                    .is_some_and(|block| {
                        range.validate(&block.source)
                            && !span.text.contains(['\n', '\r'])
                            && (span.copy_mode == CopyMode::Atomic
                                || slice_chars(&block.source, range.start, range.end) == span.text)
                    }),
            })
    }
}

fn slice_chars(text: &str, start: usize, end: usize) -> String {
    text.chars()
        .skip(start)
        .take(end.saturating_sub(start))
        .collect()
}

fn is_grapheme_boundary(text: &str, offset: usize) -> bool {
    offset == 0
        || text
            .grapheme_indices(true)
            .any(|(byte_offset, _)| text[..byte_offset].chars().count() == offset)
        || offset == text.chars().count()
}

/// Expand two inclusive grapheme-start endpoints to an exclusive scalar range.
pub fn inclusive_grapheme_bounds(text: &str, start: usize, end: usize) -> (usize, usize) {
    let char_len = text.chars().count();
    let start = start.min(char_len);
    let end = end.min(char_len);
    let mut lower = char_len;
    let mut upper = char_len;
    let mut offset = 0usize;

    for grapheme in text.graphemes(true) {
        let next = offset + grapheme.chars().count();
        if lower == char_len && start < next {
            lower = offset;
        }
        if end < next {
            upper = next;
            break;
        }
        offset = next;
    }

    if start == char_len {
        lower = char_len;
    }
    (lower, upper)
}

/// Transcript elements produce backend-neutral layout into a document.
pub trait Component<S> {
    fn render(&self, document: &mut Document<S>);
}
