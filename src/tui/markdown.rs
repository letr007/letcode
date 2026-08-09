use pulldown_cmark::{CodeBlockKind, Event, HeadingLevel, Options, Parser, Tag, TagEnd};
#[cfg(test)]
use ratatui::text::Line;
use ratatui::{
    style::{Color, Modifier, Style},
    text::Span,
};
use std::sync::OnceLock;
use syntect::{
    easy::HighlightLines,
    highlighting::{FontStyle, Style as SyntectStyle, Theme as SyntectTheme, ThemeSet},
    parsing::{SyntaxReference, SyntaxSet},
};
use unicode_segmentation::UnicodeSegmentation;

use crate::tui::{
    measure::display_width,
    theme::Theme,
    transcript_render::{
        Break, CopyJoin, Document, Line as RenderLine, SourceRange, Span as RenderSpan,
    },
};

#[derive(Debug, Clone, Copy)]
pub struct MarkdownRenderOptions {
    pub width: usize,
}

impl MarkdownRenderOptions {
    pub fn new(width: usize) -> Self {
        Self {
            width: width.max(1),
        }
    }
}

/// Legacy Ratatui bridge. The semantic document is the sole Markdown renderer.
#[cfg(test)]
pub(crate) fn render_markdown(
    markdown: &str,
    theme: Theme,
    options: MarkdownRenderOptions,
) -> Vec<Line<'static>> {
    crate::tui::transcript_ratatui::document_to_ratatui(&render_markdown_document(
        markdown, theme, options,
    ))
}

/// Render Markdown directly into renderer-neutral semantic lines. Every visible
/// parser leaf receives a source range before styling or wrapping.
pub fn render_markdown_document(
    markdown: &str,
    theme: Theme,
    options: MarkdownRenderOptions,
) -> Document<Style> {
    MarkdownRenderer::new(theme, options).render_document(markdown)
}

#[derive(Debug, Default)]
struct NativeTableState {
    rows: Vec<Vec<Vec<RenderSpan<Style>>>>,
    current_row: Vec<Vec<RenderSpan<Style>>>,
    header_rows: usize,
}

#[derive(Debug)]
struct MarkdownRenderer {
    theme: Theme,
    options: MarkdownRenderOptions,
    document: Document<Style>,
    spans: Vec<RenderSpan<Style>>,
    inline: InlineState,
    block: BlockState,
    lists: Vec<ListState>,
    quote_depth: usize,
    item_prefix: Option<String>,
    in_code_block: Option<CodeBlockState>,
    table: Option<NativeTableState>,
}

impl MarkdownRenderer {
    fn new(theme: Theme, options: MarkdownRenderOptions) -> Self {
        Self {
            theme,
            options,
            document: Document::default(),
            spans: Vec::new(),
            inline: InlineState::default(),
            block: BlockState::Document,
            lists: Vec::new(),
            quote_depth: 0,
            item_prefix: None,
            in_code_block: None,
            table: None,
        }
    }

    fn render_document(mut self, markdown: &str) -> Document<Style> {
        let markdown = normalize_markdown_math_delimiters(markdown);
        for event in Parser::new_ext(&markdown, markdown_options()) {
            self.handle_event(event);
        }
        self.flush_spans(Break::End);
        if self.document.lines.is_empty() {
            self.document.push_line(
                RenderLine {
                    spans: vec![RenderSpan::decoration("…", muted_style(self.theme))],
                },
                Break::End,
            );
        } else {
            self.document.finish();
        }
        debug_assert!(self.document.validate());
        self.document
    }

    fn handle_event(&mut self, event: Event<'_>) {
        match event {
            Event::Start(tag) => self.start_tag(tag),
            Event::End(tag) => self.end_tag(tag),
            Event::Text(text) => {
                if let Some(code) = self.in_code_block.as_mut() {
                    code.content.push_str(&text);
                } else {
                    self.push_text(&text);
                }
            }
            Event::Code(code) => self.push_inline_code(&code),
            Event::InlineMath(code) => self.push_math(&code, false),
            Event::DisplayMath(code) => {
                self.flush_spans(Break::BlockBreak);
                self.push_math(&code, true);
                self.flush_spans(Break::BlockBreak);
            }
            Event::Html(html) | Event::InlineHtml(html) => self.push_text(&html),
            Event::SoftBreak => self.push_text(" "),
            Event::HardBreak => self.flush_spans(Break::HardBreak),
            Event::Rule => self.push_rule(),
            Event::TaskListMarker(done) => self.push_text(if done { "☑ " } else { "☐ " }),
            Event::FootnoteReference(reference) => self.push_text(&format!("[{reference}]")),
        }
    }

    fn start_tag(&mut self, tag: Tag<'_>) {
        match tag {
            Tag::Paragraph => {
                self.flush_spans(Break::BlockBreak);
                self.block = BlockState::Paragraph;
            }
            Tag::Heading { level, .. } => {
                self.flush_spans(Break::BlockBreak);
                self.block = BlockState::Heading(level);
            }
            Tag::BlockQuote(_) => {
                self.flush_spans(Break::BlockBreak);
                self.quote_depth = self.quote_depth.saturating_add(1);
            }
            Tag::CodeBlock(kind) => {
                self.flush_spans(Break::BlockBreak);
                self.in_code_block = Some(CodeBlockState {
                    language: code_block_language(kind),
                    content: String::new(),
                });
            }
            Tag::List(start) => {
                self.flush_spans(Break::BlockBreak);
                self.lists.push(ListState::new(start));
            }
            Tag::Item => {
                self.flush_spans(Break::BlockBreak);
                self.item_prefix = Some(self.next_item_prefix());
            }
            Tag::Emphasis => self.inline.emphasis = self.inline.emphasis.saturating_add(1),
            Tag::Strong => self.inline.strong = self.inline.strong.saturating_add(1),
            Tag::Strikethrough => {
                self.inline.strikethrough = self.inline.strikethrough.saturating_add(1)
            }
            Tag::Link { dest_url, .. } => self.inline.links.push(dest_url.to_string()),
            Tag::Table(_) => {
                self.flush_spans(Break::BlockBreak);
                self.block = BlockState::Table;
                self.table = Some(NativeTableState::default());
            }
            Tag::TableHead => {}
            Tag::TableRow => {
                if let Some(table) = self.table.as_mut() {
                    table.current_row.clear();
                }
            }
            Tag::TableCell => self.spans.clear(),
            Tag::FootnoteDefinition(name) => {
                self.flush_spans(Break::BlockBreak);
                self.push_text(&format!("[{name}] "));
            }
            Tag::DefinitionListDefinition => {
                self.flush_spans(Break::BlockBreak);
                self.item_prefix = Some("  ".to_string());
            }
            Tag::Image {
                title, dest_url, ..
            } => {
                self.push_text(if title.is_empty() { "image" } else { &title });
                if !dest_url.is_empty() {
                    self.push_text(&format!(" <{dest_url}>"));
                }
            }
            Tag::HtmlBlock
            | Tag::MetadataBlock(_)
            | Tag::DefinitionList
            | Tag::DefinitionListTitle
            | Tag::Superscript
            | Tag::Subscript => {}
        }
    }

    fn end_tag(&mut self, tag: TagEnd) {
        match tag {
            TagEnd::Paragraph => {
                self.flush_spans(Break::BlockBreak);
                self.block = BlockState::Document;
            }
            TagEnd::Heading(_) => {
                self.flush_spans(Break::BlockBreak);
                self.block = BlockState::Document;
            }
            TagEnd::BlockQuote(_) => {
                self.flush_spans(Break::BlockBreak);
                self.quote_depth = self.quote_depth.saturating_sub(1);
            }
            TagEnd::CodeBlock => {
                if let Some(code) = self.in_code_block.take() {
                    self.push_code_block(code);
                }
            }
            TagEnd::List(_) => {
                self.flush_spans(Break::BlockBreak);
                self.lists.pop();
            }
            TagEnd::Item => {
                self.flush_spans(Break::BlockBreak);
                self.item_prefix = None;
            }
            TagEnd::Emphasis => self.inline.emphasis = self.inline.emphasis.saturating_sub(1),
            TagEnd::Strong => self.inline.strong = self.inline.strong.saturating_sub(1),
            TagEnd::Strikethrough => {
                self.inline.strikethrough = self.inline.strikethrough.saturating_sub(1)
            }
            TagEnd::Link => {
                if let Some(dest_url) = self.inline.links.pop()
                    && !dest_url.is_empty()
                {
                    // Destination is display/copy only. Jump stays on the underlined label.
                    self.push_styled_with_interaction(
                        &format!(" <{dest_url}>"),
                        link_dest_style(self.theme),
                        None,
                    );
                }
            }
            TagEnd::Table => {
                if let Some(table) = self.table.take() {
                    self.push_table(table);
                }
                self.block = BlockState::Document;
            }
            TagEnd::TableHead => {
                if let Some(table) = self.table.as_mut()
                    && !table.current_row.is_empty()
                {
                    table.rows.push(std::mem::take(&mut table.current_row));
                    table.header_rows = table.rows.len();
                }
            }
            TagEnd::TableRow => {
                if let Some(table) = self.table.as_mut()
                    && !table.current_row.is_empty()
                {
                    table.rows.push(std::mem::take(&mut table.current_row));
                }
            }
            TagEnd::TableCell => {
                if let Some(table) = self.table.as_mut() {
                    table.current_row.push(std::mem::take(&mut self.spans));
                }
            }
            TagEnd::FootnoteDefinition
            | TagEnd::HtmlBlock
            | TagEnd::DefinitionList
            | TagEnd::DefinitionListTitle
            | TagEnd::DefinitionListDefinition
            | TagEnd::MetadataBlock(_)
            | TagEnd::Image => self.flush_spans(Break::BlockBreak),
            TagEnd::Superscript | TagEnd::Subscript => {}
        }
    }

    /// Parser leaf construction point: never derive provenance from rendered output.
    fn push_styled(&mut self, text: &str, style: Style) {
        let interaction = self.inline.links.last().cloned().and_then(|url| {
            (!url.is_empty()).then_some(crate::tui::transcript_render::Interaction::OpenUrl(url))
        });
        self.push_styled_with_interaction(text, style, interaction);
    }

    fn push_styled_with_interaction(
        &mut self,
        text: &str,
        style: Style,
        interaction: Option<crate::tui::transcript_render::Interaction>,
    ) {
        if text.is_empty() {
            return;
        }
        let block = self.document.add_source(text);
        self.spans.push(RenderSpan::source_with_interaction(
            text,
            style,
            SourceRange::new(block, 0, text.chars().count()),
            CopyJoin::Concat,
            interaction,
        ));
    }

    fn push_text(&mut self, text: &str) {
        self.push_styled(text, self.inline_style());
    }
    fn push_inline_code(&mut self, text: &str) {
        self.push_styled(text, inline_code_style(self.theme));
    }

    fn push_math(&mut self, text: &str, display: bool) {
        let Some(layout) = crate::tui::math::render(text, display) else {
            self.push_math_fallback(text, display);
            return;
        };
        let (first_prefix, next_prefix) = self.line_prefixes();
        let prefix_width = if display {
            display_width(&first_prefix).max(display_width(&next_prefix))
        } else {
            display_width(&first_prefix)
        };
        let available_width = self.options.width.saturating_sub(prefix_width);
        if layout.width == 0
            || layout.width > available_width
            || (!display && layout.rows.len() != 1)
        {
            self.push_math_fallback(text, display);
            return;
        }

        let style = if display {
            self.theme.app_style()
        } else {
            inline_code_style(self.theme)
        };
        let block = self.document.add_source(text);
        let source = SourceRange::new(block, 0, text.chars().count());
        if display {
            let row_count = layout.rows.len();
            for (index, row) in layout.rows.into_iter().enumerate() {
                let prefix = if index == 0 {
                    &first_prefix
                } else {
                    &next_prefix
                };
                let mut spans = Vec::new();
                if !prefix.is_empty() {
                    spans.push(RenderSpan::decoration(prefix.clone(), self.prefix_style()));
                }
                spans.push(RenderSpan::source_atomic(row, style, source));
                self.document.push_line(
                    RenderLine { spans },
                    if index + 1 == row_count {
                        Break::BlockBreak
                    } else {
                        Break::HardBreak
                    },
                );
            }
        } else if let Some(row) = layout.rows.into_iter().next() {
            self.spans
                .push(RenderSpan::source_atomic(row, style, source));
        }
    }

    fn push_math_fallback(&mut self, text: &str, display: bool) {
        let style = if display {
            self.theme.app_style()
        } else {
            inline_code_style(self.theme)
        };
        let block = self.document.add_source(text);
        let source = SourceRange::new(block, 0, text.chars().count());
        self.spans
            .push(RenderSpan::source_atomic(text, style, source));
    }

    fn flush_spans(&mut self, final_break: Break) {
        if self.spans.is_empty() {
            return;
        }
        let spans = std::mem::take(&mut self.spans);
        let (first_prefix, next_prefix) = self.line_prefixes();
        let first_width = self
            .options
            .width
            .saturating_sub(display_width(&first_prefix))
            .max(1);
        let next_width = self
            .options
            .width
            .saturating_sub(display_width(&next_prefix))
            .max(1);
        let mut wrapped = wrap_render_spans_with_prefixes(
            spans,
            first_width,
            next_width,
            RenderSpan::decoration(first_prefix, self.prefix_style()),
            RenderSpan::decoration(next_prefix, self.prefix_style()),
            self.block_style(),
        );
        if let Some((_, boundary)) = wrapped.last_mut() {
            *boundary = final_break;
        }
        for (line, boundary) in wrapped {
            self.document.push_line(line, boundary);
        }
    }

    fn push_rule(&mut self) {
        self.flush_spans(Break::BlockBreak);
        let (prefix, _) = self.line_prefixes();
        let width = self
            .options
            .width
            .saturating_sub(display_width(&prefix))
            .max(1);
        let mut spans = Vec::new();
        if !prefix.is_empty() {
            spans.push(RenderSpan::decoration(prefix, self.prefix_style()));
        }
        spans.push(RenderSpan::decoration(
            "─".repeat(width),
            muted_style(self.theme),
        ));
        self.document
            .push_line(RenderLine { spans }, Break::BlockBreak);
    }

    fn push_code_block(&mut self, code: CodeBlockState) {
        self.flush_spans(Break::BlockBreak);
        if self
            .document
            .lines
            .last()
            .is_some_and(|line| !line.spans.is_empty())
        {
            self.document
                .push_line(RenderLine::default(), Break::BlockBreak);
        }
        if code
            .language
            .as_deref()
            .is_some_and(|language| language.eq_ignore_ascii_case("mermaid"))
            && self.push_mermaid_block(&code)
        {
            return;
        }
        let (first_prefix, next_prefix) = self.line_prefixes();
        let prefix_style = self.prefix_style();
        let prefix_width = display_width(&first_prefix).max(display_width(&next_prefix));
        let width = self.options.width.saturating_sub(prefix_width).max(1);
        let border = code_block_border_style(self.theme);
        let label_style = code_block_label_style(self.theme);
        let label = code
            .language
            .as_deref()
            .filter(|value| !value.is_empty())
            .unwrap_or("code");
        self.push_code_line(
            &first_prefix,
            prefix_style,
            vec![
                RenderSpan::decoration("╭─ ", border),
                RenderSpan::decoration(label, label_style),
            ],
            width,
            Break::BlockBreak,
        );

        let mut highlighter = CodeHighlighter::new(code.language.as_deref(), self.theme);
        let body_width = width.saturating_sub(3).max(1);
        let mut body = false;
        for raw in code.content.lines() {
            body = true;
            let block = self.document.add_source(raw);
            let chunks = wrap_source_text(raw, block, body_width, code_block_style(self.theme));
            let chunk_count = chunks.len();
            for (index, chunk) in chunks.into_iter().enumerate() {
                let mut content = vec![RenderSpan::decoration("│ ", border)];
                content.extend(highlight_render_spans(
                    &mut highlighter,
                    chunk,
                    code_block_style(self.theme),
                ));
                self.push_code_line(
                    &next_prefix,
                    prefix_style,
                    content,
                    width,
                    if index + 1 == chunk_count {
                        Break::HardBreak
                    } else {
                        Break::SoftWrap
                    },
                );
            }
        }
        if !body {
            self.push_code_line(
                &next_prefix,
                prefix_style,
                vec![RenderSpan::decoration("│ ", border)],
                width,
                Break::HardBreak,
            );
        }
        self.push_code_line(
            &next_prefix,
            prefix_style,
            vec![RenderSpan::decoration("╰", border)],
            width,
            Break::BlockBreak,
        );
        self.document
            .push_line(RenderLine::default(), Break::BlockBreak);
    }

    fn push_mermaid_block(&mut self, code: &CodeBlockState) -> bool {
        let available_width = self.options.width.saturating_sub(
            display_width(&self.line_prefixes().0).max(display_width(&self.line_prefixes().1)),
        );
        let Some(rendered) = crate::tui::mermaid::render(&code.content, available_width) else {
            return false;
        };
        let block = self.document.add_source(code.content.clone());
        let (first_prefix, next_prefix) = self.line_prefixes();
        let output_len = rendered.lines.len();
        for (index, line_spans) in rendered.lines.into_iter().enumerate() {
            let mut spans = Vec::new();
            let prefix = if index == 0 {
                &first_prefix
            } else {
                &next_prefix
            };
            if !prefix.is_empty() {
                spans.push(RenderSpan::decoration(prefix.clone(), self.prefix_style()));
            }
            for span in line_spans {
                if let Some(source) = span.source {
                    let source = SourceRange::new(block, source.start, source.end);
                    spans.push(if span.atomic {
                        RenderSpan::source_atomic(span.text, self.theme.app_style(), source)
                    } else {
                        RenderSpan::source(span.text, self.theme.app_style(), source)
                    });
                } else {
                    spans.push(RenderSpan::decoration(
                        span.text,
                        self.theme.app_style().fg(self.theme.accent),
                    ));
                }
            }
            self.document.push_line(
                RenderLine { spans },
                if index + 1 == output_len {
                    Break::BlockBreak
                } else {
                    Break::HardBreak
                },
            );
        }
        self.document
            .push_line(RenderLine::default(), Break::BlockBreak);
        true
    }

    fn push_code_line(
        &mut self,
        prefix: &str,
        prefix_style: Style,
        content: Vec<RenderSpan<Style>>,
        width: usize,
        boundary: Break,
    ) {
        let mut spans = Vec::new();
        if !prefix.is_empty() {
            spans.push(RenderSpan::decoration(prefix, prefix_style));
        }
        spans.extend(content);
        let mut spans = truncate_render_spans(&spans, self.options.width);
        pad_render_line(
            &mut spans,
            width.saturating_add(display_width(prefix)),
            code_block_style(self.theme),
        );
        self.document.push_line(RenderLine { spans }, boundary);
    }

    fn push_table(&mut self, table: NativeTableState) {
        if table.rows.is_empty() {
            return;
        }
        let (first_prefix, next_prefix) = self.line_prefixes();
        let pane_width = self
            .options
            .width
            .saturating_sub(display_width(&first_prefix).max(display_width(&next_prefix)))
            .max(1);
        let columns = table.rows.iter().map(Vec::len).max().unwrap_or(0);
        let mut widths = vec![1; columns];
        for row in &table.rows {
            for (index, cell) in row.iter().enumerate() {
                widths[index] = widths[index].max(display_width(&render_span_text(cell)));
            }
        }
        fit_column_widths(&mut widths, pane_width);
        for (row_index, row) in table.rows.into_iter().enumerate() {
            let prefix = if row_index == 0 {
                &first_prefix
            } else {
                &next_prefix
            };
            let style = if row_index < table.header_rows {
                table_header_style(self.theme)
            } else {
                table_style(self.theme)
            };
            let mut spans = Vec::new();
            if !prefix.is_empty() {
                spans.push(RenderSpan::decoration(prefix.clone(), self.prefix_style()));
            }
            for (index, width) in widths.iter().copied().enumerate() {
                if index > 0 {
                    spans.push(RenderSpan::decoration(
                        " │ ",
                        table_border_style(self.theme),
                    ));
                }
                let mut cell =
                    truncate_render_spans(row.get(index).map(Vec::as_slice).unwrap_or(&[]), width);
                if index > 0
                    && let Some(span) = cell.iter_mut().find(|span| span.source.is_some())
                {
                    span.copy_join = CopyJoin::Space;
                }
                for span in &mut cell {
                    span.style = style;
                }
                let used = display_width(&render_span_text(&cell));
                spans.extend(cell);
                if width > used {
                    spans.push(RenderSpan::decoration(" ".repeat(width - used), style));
                }
            }
            pad_render_line(&mut spans, self.options.width, self.theme.app_style());
            self.document
                .push_line(RenderLine { spans }, Break::HardBreak);
            if table.header_rows > 0 && row_index + 1 == table.header_rows {
                let mut separator = Vec::new();
                if !next_prefix.is_empty() {
                    separator.push(RenderSpan::decoration(
                        next_prefix.clone(),
                        self.prefix_style(),
                    ));
                }
                separator.push(RenderSpan::decoration(
                    table_separator(&widths),
                    table_border_style(self.theme),
                ));
                pad_render_line(&mut separator, self.options.width, self.theme.app_style());
                self.document
                    .push_line(RenderLine { spans: separator }, Break::HardBreak);
            }
        }
    }

    fn next_item_prefix(&mut self) -> String {
        let depth = self.lists.len().saturating_sub(1);
        let indent = "  ".repeat(depth);
        let Some(list) = self.lists.last_mut() else {
            return format!("{indent}• ");
        };
        match &mut list.kind {
            ListKind::Unordered => format!("{indent}• "),
            ListKind::Ordered(next) => {
                let marker = format!("{next}. ");
                *next = next.saturating_add(1);
                format!("{indent}{marker}")
            }
        }
    }

    fn line_prefixes(&self) -> (String, String) {
        let quote = "│ ".repeat(self.quote_depth);
        let marker = match self.block {
            BlockState::Heading(level) if heading_number(level) <= 2 => "▌ ".to_string(),
            BlockState::Heading(_) => "• ".to_string(),
            _ => self.item_prefix.clone().unwrap_or_default(),
        };
        let first = format!("{quote}{marker}");
        (
            first,
            format!("{}{}", quote, " ".repeat(display_width(&marker))),
        )
    }

    fn prefix_style(&self) -> Style {
        match self.block {
            BlockState::Heading(_) => heading_marker_style(self.theme),
            _ if self.quote_depth > 0 => quote_style(self.theme),
            _ if self.item_prefix.is_some() => list_marker_style(self.theme),
            _ => self.theme.app_style(),
        }
    }
    fn block_style(&self) -> Style {
        match self.block {
            BlockState::Heading(level) => heading_style(self.theme, level),
            BlockState::Table => table_style(self.theme),
            _ => self.inline_style(),
        }
    }
    fn inline_style(&self) -> Style {
        let mut style = self.theme.app_style();
        if self.inline.emphasis > 0 {
            style = style.add_modifier(Modifier::ITALIC);
        }
        if self.inline.strong > 0 {
            style = style.add_modifier(Modifier::BOLD);
        }
        if self.inline.strikethrough > 0 {
            style = style.add_modifier(Modifier::CROSSED_OUT);
        }
        if !self.inline.links.is_empty() {
            style = style
                .fg(self.theme.accent)
                .add_modifier(Modifier::UNDERLINED);
        }
        style
    }
}

fn append_render_grapheme(
    target: &mut Vec<RenderSpan<Style>>,
    text: &str,
    style: Style,
    source: Option<SourceRange>,
    interaction: Option<crate::tui::transcript_render::Interaction>,
    copy_mode: crate::tui::transcript_render::CopyMode,
) {
    if let Some(last) = target.last_mut()
        && last.style == style
        && last.interaction == interaction
        && last.copy_mode == copy_mode
        && copy_mode == crate::tui::transcript_render::CopyMode::Exact
        && match (last.source, source) {
            (None, None) => true,
            (Some(previous), Some(next)) => {
                previous.block_index == next.block_index && previous.end == next.start
            }
            _ => false,
        }
    {
        last.text.push_str(text);
        if let (Some(last), Some(source)) = (&mut last.source, source) {
            last.end = source.end;
        }
    } else if let Some(source) = source {
        target.push(RenderSpan::source_with_mode(
            text,
            style,
            source,
            copy_mode,
            CopyJoin::Concat,
            interaction,
        ));
    } else {
        target.push(RenderSpan::decoration(text, style));
    }
}

fn wrap_render_spans_with_prefixes(
    spans: Vec<RenderSpan<Style>>,
    first_width: usize,
    next_width: usize,
    first_prefix: RenderSpan<Style>,
    next_prefix: RenderSpan<Style>,
    fallback: Style,
) -> Vec<(RenderLine<Style>, Break)> {
    let mut lines = Vec::new();
    let mut current = Vec::new();
    let mut used = 0usize;
    let mut limit = first_width.max(1);
    let mut continuation = false;
    let mut at_start = true;
    for span in spans {
        let mut offset = span.source.map(|range| range.start).unwrap_or(0);
        for grapheme in span.text.graphemes(true) {
            let count = grapheme.chars().count();
            let source = if span.copy_mode == crate::tui::transcript_render::CopyMode::Atomic {
                span.source
            } else {
                span.source
                    .map(|range| SourceRange::new(range.block_index, offset, offset + count))
            };
            offset += count;
            if grapheme == "\n" {
                let mut output = Vec::new();
                let prefix = if continuation {
                    &next_prefix
                } else {
                    &first_prefix
                };
                if !prefix.text.is_empty() {
                    output.push(prefix.clone());
                }
                output.append(&mut current);
                lines.push((RenderLine { spans: output }, Break::HardBreak));
                used = 0;
                limit = next_width.max(1);
                continuation = true;
                at_start = true;
                continue;
            }
            let width = display_width(grapheme);
            if used > 0 && used.saturating_add(width) > limit {
                let mut output = Vec::new();
                let prefix = if continuation {
                    &next_prefix
                } else {
                    &first_prefix
                };
                if !prefix.text.is_empty() {
                    output.push(prefix.clone());
                }
                output.append(&mut current);
                lines.push((RenderLine { spans: output }, Break::SoftWrap));
                used = 0;
                limit = next_width.max(1);
                continuation = true;
                at_start = true;
            }
            if at_start && grapheme == " " {
                continue;
            }
            if width > limit && current.is_empty() {
                continue;
            }
            append_render_grapheme(
                &mut current,
                grapheme,
                span.style,
                source,
                span.interaction.clone(),
                span.copy_mode,
            );
            used = used.saturating_add(width);
            at_start = false;
        }
    }
    if current.is_empty() && lines.is_empty() {
        current.push(RenderSpan::decoration("", fallback));
    }
    if !current.is_empty() {
        let mut output = Vec::new();
        let prefix = if continuation {
            next_prefix
        } else {
            first_prefix
        };
        if !prefix.text.is_empty() {
            output.push(prefix);
        }
        output.append(&mut current);
        lines.push((RenderLine { spans: output }, Break::End));
    }
    lines
}

fn wrap_source_text(
    text: &str,
    block: usize,
    width: usize,
    style: Style,
) -> Vec<RenderSpan<Style>> {
    let mut rows = Vec::new();
    let mut start = 0;
    let mut end = 0;
    let mut used = 0usize;
    for grapheme in text.graphemes(true) {
        let count = grapheme.chars().count();
        let cell_width = display_width(grapheme);
        if used > 0 && used.saturating_add(cell_width) > width {
            rows.push(RenderSpan::source(
                text.chars()
                    .skip(start)
                    .take(end - start)
                    .collect::<String>(),
                style,
                SourceRange::new(block, start, end),
            ));
            start = end;
            used = 0;
        }
        if cell_width > width && start == end {
            continue;
        }
        end += count;
        used += cell_width;
    }
    if start < end {
        rows.push(RenderSpan::source(
            text.chars()
                .skip(start)
                .take(end - start)
                .collect::<String>(),
            style,
            SourceRange::new(block, start, end),
        ));
    }
    rows
}

fn highlight_render_spans(
    highlighter: &mut CodeHighlighter<'_>,
    source: RenderSpan<Style>,
    fallback: Style,
) -> Vec<RenderSpan<Style>> {
    let mut offset = source.source.expect("code source").start;
    let range = source.source.expect("code source");
    highlighter
        .highlight_line(&source.text, fallback)
        .into_iter()
        .map(|span| {
            let count = span.content.chars().count();
            let result = RenderSpan::source(
                span.content.into_owned(),
                span.style,
                SourceRange::new(range.block_index, offset, offset + count),
            );
            offset += count;
            result
        })
        .collect()
}

fn render_span_text(spans: &[RenderSpan<Style>]) -> String {
    spans.iter().map(|span| span.text.as_str()).collect()
}

fn truncate_render_spans(spans: &[RenderSpan<Style>], width: usize) -> Vec<RenderSpan<Style>> {
    let mut out = Vec::new();
    let mut used = 0usize;
    for span in spans {
        let mut offset = span.source.map(|range| range.start).unwrap_or(0);
        for grapheme in span.text.graphemes(true) {
            let count = grapheme.chars().count();
            let grapheme_width = display_width(grapheme);
            if used.saturating_add(grapheme_width) > width {
                return out;
            }
            append_render_grapheme(
                &mut out,
                grapheme,
                span.style,
                if span.copy_mode == crate::tui::transcript_render::CopyMode::Atomic {
                    span.source
                } else {
                    span.source
                        .map(|range| SourceRange::new(range.block_index, offset, offset + count))
                },
                span.interaction.clone(),
                span.copy_mode,
            );
            offset += count;
            used += grapheme_width;
        }
    }
    out
}

fn pad_render_line(spans: &mut Vec<RenderSpan<Style>>, width: usize, style: Style) {
    let used = display_width(&render_span_text(spans));
    if width > used {
        spans.push(RenderSpan::decoration(" ".repeat(width - used), style));
    }
}

#[derive(Debug, Default)]
struct InlineState {
    emphasis: usize,
    strong: usize,
    strikethrough: usize,
    links: Vec<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BlockState {
    Document,
    Paragraph,
    Heading(HeadingLevel),
    Table,
}

#[derive(Debug)]
struct ListState {
    kind: ListKind,
}

impl ListState {
    fn new(start: Option<u64>) -> Self {
        Self {
            kind: start.map_or(ListKind::Unordered, ListKind::Ordered),
        }
    }
}

#[derive(Debug)]
enum ListKind {
    Unordered,
    Ordered(u64),
}

#[derive(Debug)]
struct CodeBlockState {
    language: Option<String>,
    content: String,
}

struct CodeHighlighter<'a> {
    syntax_set: &'static SyntaxSet,
    highlighter: Option<HighlightLines<'a>>,
    theme: Theme,
}

impl<'a> CodeHighlighter<'a> {
    fn new(language: Option<&str>, theme: Theme) -> Self {
        let syntax_set = syntax_set();
        let syntax = language.and_then(|language| find_syntax(syntax_set, language));
        let highlighter = syntax.map(|syntax| HighlightLines::new(syntax, syntect_theme()));

        Self {
            syntax_set,
            highlighter,
            theme,
        }
    }

    fn highlight_line(&mut self, line: &str, fallback_style: Style) -> Vec<Span<'static>> {
        let Some(highlighter) = self.highlighter.as_mut() else {
            return vec![Span::styled(line.to_string(), fallback_style)];
        };

        match highlighter.highlight_line(line, self.syntax_set) {
            Ok(ranges) => ranges
                .into_iter()
                .filter(|(_, text)| !text.is_empty())
                .map(|(style, text)| {
                    Span::styled(
                        text.to_string(),
                        syntect_to_ratatui_style(style, self.theme),
                    )
                })
                .collect(),
            Err(_) => vec![Span::styled(line.to_string(), fallback_style)],
        }
    }
}

fn syntax_set() -> &'static SyntaxSet {
    static SYNTAX_SET: OnceLock<SyntaxSet> = OnceLock::new();
    SYNTAX_SET.get_or_init(SyntaxSet::load_defaults_newlines)
}

fn syntect_theme() -> &'static SyntectTheme {
    static THEME: OnceLock<SyntectTheme> = OnceLock::new();
    THEME.get_or_init(|| {
        let theme_set = ThemeSet::load_defaults();
        theme_set
            .themes
            .get("base16-ocean.dark")
            .or_else(|| theme_set.themes.values().next())
            .cloned()
            .unwrap_or_default()
    })
}

fn find_syntax<'a>(syntax_set: &'a SyntaxSet, language: &str) -> Option<&'a SyntaxReference> {
    let normalized = normalize_language(language);
    syntax_set
        .find_syntax_by_token(&normalized)
        .or_else(|| syntax_set.find_syntax_by_extension(&normalized))
        .or_else(|| syntax_set.find_syntax_by_name(&normalized))
}

fn normalize_language(language: &str) -> String {
    match language.trim().to_ascii_lowercase().as_str() {
        "rs" => "rust".to_string(),
        "js" | "jsx" => "javascript".to_string(),
        "ts" | "tsx" => "typescript".to_string(),
        "sh" | "zsh" | "shell" => "bash".to_string(),
        "yml" => "yaml".to_string(),
        "md" => "markdown".to_string(),
        other => other.to_string(),
    }
}

fn syntect_to_ratatui_style(style: SyntectStyle, theme: Theme) -> Style {
    let mut tui_style = Style::default()
        .fg(Color::Rgb(
            style.foreground.r,
            style.foreground.g,
            style.foreground.b,
        ))
        .bg(theme.element_bg);

    if style.font_style.contains(FontStyle::BOLD) {
        tui_style = tui_style.add_modifier(Modifier::BOLD);
    }
    if style.font_style.contains(FontStyle::ITALIC) {
        tui_style = tui_style.add_modifier(Modifier::ITALIC);
    }
    if style.font_style.contains(FontStyle::UNDERLINE) {
        tui_style = tui_style.add_modifier(Modifier::UNDERLINED);
    }

    tui_style
}

fn normalize_markdown_math_delimiters(markdown: &str) -> String {
    fn escaped_at(source: &str, index: usize) -> bool {
        source.as_bytes()[..index]
            .iter()
            .rev()
            .take_while(|byte| **byte == b'\\')
            .count()
            % 2
            == 1
    }

    fn fence_marker(line: &str) -> Option<(u8, usize, bool)> {
        let content = line.trim_end_matches(&['\n', '\r'][..]);
        let indent = content.bytes().take_while(|byte| *byte == b' ').count();
        if indent > 3 {
            return None;
        }
        let marker = *content.as_bytes().get(indent)?;
        if !matches!(marker, b'`' | b'~') {
            return None;
        }
        let count = content.as_bytes()[indent..]
            .iter()
            .take_while(|byte| **byte == marker)
            .count();
        if count < 3 {
            return None;
        }
        let closing = content[indent + count..].trim().is_empty();
        Some((marker, count, closing))
    }

    let mut protected = vec![false; markdown.len()];
    let mut fence: Option<(u8, usize)> = None;
    let mut line_offset = 0usize;
    for line in markdown.split_inclusive('\n') {
        let line_end = line_offset + line.len();
        if let Some((marker, count, closing)) = fence_marker(line) {
            match fence {
                Some((open_marker, open_count))
                    if marker == open_marker && count >= open_count && closing =>
                {
                    protected[line_offset..line_end].fill(true);
                    fence = None;
                    line_offset = line_end;
                    continue;
                }
                None => {
                    protected[line_offset..line_end].fill(true);
                    fence = Some((marker, count));
                    line_offset = line_end;
                    continue;
                }
                _ => {}
            }
        }
        if fence.is_some() {
            protected[line_offset..line_end].fill(true);
        }
        line_offset = line_end;
    }

    let mut code_span = None;
    let mut index = 0usize;
    while index < markdown.len() {
        if protected[index] {
            code_span = None;
            index += 1;
            continue;
        }
        if markdown.as_bytes()[index] != b'`' || escaped_at(markdown, index) {
            index += 1;
            continue;
        }
        let ticks = markdown.as_bytes()[index..]
            .iter()
            .take_while(|byte| **byte == b'`')
            .count();
        if let Some((open_index, open_ticks)) = code_span {
            if ticks == open_ticks {
                protected[open_index..index + ticks].fill(true);
                code_span = None;
            }
        } else {
            code_span = Some((index, ticks));
        }
        index += ticks;
    }

    #[derive(Clone, Copy, PartialEq, Eq)]
    enum Delimiter {
        Parenthesis,
        Bracket,
    }

    let mut replacements = vec![0u8; markdown.len()];
    let mut pending = None;
    let mut index = 0usize;
    while index + 1 < markdown.len() {
        if protected[index] {
            pending = None;
            index += 1;
            continue;
        }
        if markdown.as_bytes()[index] == b'\n' {
            if pending.is_some_and(|(delimiter, _)| delimiter == Delimiter::Parenthesis) {
                pending = None;
            } else if pending.is_some_and(|(delimiter, _)| delimiter == Delimiter::Bracket) {
                let rest = &markdown.as_bytes()[index + 1..];
                if rest
                    .iter()
                    .take_while(|byte| **byte != b'\n')
                    .all(|byte| matches!(*byte, b' ' | b'\t' | b'\r'))
                {
                    pending = None;
                }
            }
            index += 1;
            continue;
        }
        if markdown.as_bytes()[index] != b'\\' || escaped_at(markdown, index) {
            index += 1;
            continue;
        }
        let token = match markdown.as_bytes()[index + 1] {
            b'(' => Some((Delimiter::Parenthesis, true)),
            b')' => Some((Delimiter::Parenthesis, false)),
            b'[' => Some((Delimiter::Bracket, true)),
            b']' => Some((Delimiter::Bracket, false)),
            _ => None,
        };
        if let Some((delimiter, opening)) = token {
            if opening {
                pending = Some((delimiter, index));
            } else if let Some((open_delimiter, open_index)) = pending {
                if open_delimiter == delimiter {
                    let width = if delimiter == Delimiter::Bracket {
                        2
                    } else {
                        1
                    };
                    replacements[open_index] = width;
                    replacements[index] = width;
                }
                pending = None;
            }
            index += 2;
        } else {
            index += 1;
        }
    }

    let mut output = String::with_capacity(markdown.len());
    let mut index = 0usize;
    while index < markdown.len() {
        let replacement = replacements[index];
        if replacement > 0 {
            output.push_str(if replacement == 1 { "$" } else { "$$" });
            index += 2;
        } else {
            let ch = markdown[index..].chars().next().unwrap();
            output.push(ch);
            index += ch.len_utf8();
        }
    }
    output
}

fn markdown_options() -> Options {
    Options::ENABLE_TABLES
        | Options::ENABLE_FOOTNOTES
        | Options::ENABLE_STRIKETHROUGH
        | Options::ENABLE_TASKLISTS
        | Options::ENABLE_HEADING_ATTRIBUTES
        | Options::ENABLE_MATH
        | Options::ENABLE_GFM
}

fn code_block_language(kind: CodeBlockKind<'_>) -> Option<String> {
    match kind {
        CodeBlockKind::Fenced(info) => info
            .split_whitespace()
            .next()
            .filter(|language| !language.is_empty())
            .map(ToString::to_string),
        CodeBlockKind::Indented => None,
    }
}

fn heading_number(level: HeadingLevel) -> u8 {
    match level {
        HeadingLevel::H1 => 1,
        HeadingLevel::H2 => 2,
        HeadingLevel::H3 => 3,
        HeadingLevel::H4 => 4,
        HeadingLevel::H5 => 5,
        HeadingLevel::H6 => 6,
    }
}

fn fit_column_widths(widths: &mut [usize], total_width: usize) {
    if widths.is_empty() {
        return;
    }

    let separator_width = widths.len().saturating_sub(1).saturating_mul(3);
    let available = total_width
        .saturating_sub(separator_width)
        .max(widths.len());

    // Natural content widths are preferred. Only shrink (widest first) when the
    // table would exceed the available markdown pane width.
    while widths.iter().sum::<usize>() > available {
        let Some((index, _)) = widths.iter().enumerate().max_by_key(|(_, width)| *width) else {
            return;
        };
        if widths[index] <= 1 {
            break;
        }
        widths[index] = widths[index].saturating_sub(1);
    }
}

fn table_separator(widths: &[usize]) -> String {
    widths
        .iter()
        .map(|width| "─".repeat((*width).max(1)))
        .collect::<Vec<_>>()
        .join("─┼─")
}

fn heading_style(theme: Theme, level: HeadingLevel) -> Style {
    let mut style = theme
        .app_style()
        .fg(theme.text)
        .add_modifier(Modifier::BOLD);
    if heading_number(level) <= 2 {
        style = style.fg(theme.text);
    } else {
        style = style.fg(theme.muted_text);
    }
    style
}

fn heading_marker_style(theme: Theme) -> Style {
    theme
        .app_style()
        .fg(theme.accent)
        .add_modifier(Modifier::BOLD)
}

fn list_marker_style(theme: Theme) -> Style {
    theme
        .app_style()
        .fg(theme.accent)
        .add_modifier(Modifier::BOLD)
}

fn quote_style(theme: Theme) -> Style {
    theme
        .app_style()
        .fg(theme.notice)
        .add_modifier(Modifier::BOLD)
}

fn inline_code_style(theme: Theme) -> Style {
    Style::default().fg(theme.accent).bg(theme.element_bg)
}

fn code_block_style(theme: Theme) -> Style {
    Style::default().fg(theme.text).bg(theme.element_bg)
}

fn code_block_border_style(theme: Theme) -> Style {
    Style::default().fg(theme.accent).bg(theme.element_bg)
}

fn code_block_label_style(theme: Theme) -> Style {
    Style::default()
        .fg(theme.approval)
        .bg(theme.element_bg)
        .add_modifier(Modifier::BOLD)
}

fn table_style(theme: Theme) -> Style {
    theme.app_style().fg(theme.text)
}

fn table_header_style(theme: Theme) -> Style {
    theme
        .app_style()
        .fg(theme.text)
        .add_modifier(Modifier::BOLD)
}

fn table_border_style(theme: Theme) -> Style {
    theme.app_style().fg(theme.dim_text)
}

fn link_dest_style(theme: Theme) -> Style {
    theme.app_style().fg(theme.muted_text)
}

fn muted_style(theme: Theme) -> Style {
    Style::default().fg(theme.muted_text).bg(theme.root_bg)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rendered(markdown: &str, width: usize) -> Vec<Line<'static>> {
        render_markdown(markdown, Theme::dark(), MarkdownRenderOptions::new(width))
    }

    #[test]
    fn markdown_links_retain_open_url_interaction_through_wrapping() {
        let document = render_markdown_document(
            "[a long link label](https://example.test/path)",
            Theme::dark(),
            MarkdownRenderOptions::new(4),
        );
        assert!(document.validate());
        let links = document
            .lines
            .iter()
            .flat_map(|line| &line.spans)
            .filter_map(|span| span.interaction.as_ref())
            .collect::<Vec<_>>();
        assert!(links.iter().all(|interaction| matches!(
            interaction,
            crate::tui::transcript_render::Interaction::OpenUrl(url)
                if url == "https://example.test/path"
        )));
        assert!(links.len() > 1, "{document:?}");
        assert!(
            document
                .lines
                .iter()
                .flat_map(|line| &line.spans)
                .filter(|span| span.text.contains("link") || span.text.contains("label"))
                .all(|span| span.interaction.is_some())
        );
        assert!(
            document
                .lines
                .iter()
                .flat_map(|line| &line.spans)
                .filter(|span| span.text.contains('<') || span.text.contains("example"))
                .all(|span| span.interaction.is_none()),
            "destination suffix must not be jumpable: {document:?}"
        );
    }

    #[test]
    fn legacy_bridge_preserves_markdown_visuals() {
        let lines = rendered("# Title\n- **item** with `code`", 80);
        let text = lines
            .iter()
            .map(Line::to_string)
            .collect::<Vec<_>>()
            .join("\n");
        assert!(text.contains("▌ Title"), "{text}");
        assert!(text.contains("• item with code"), "{text}");
        assert!(!text.contains("**item**"), "{text}");
        assert!(!text.contains('`'), "{text}");
    }

    #[test]
    fn code_blocks_preserve_card_layout_and_spacing() {
        let lines = rendered("before\n\n```rust\nlet x = 1;\n```\n\nafter", 24);
        let text = lines
            .iter()
            .map(Line::to_string)
            .collect::<Vec<_>>()
            .join("\n");

        assert_eq!(lines[0].to_string(), "before");
        assert_eq!(lines[1].to_string(), "");
        assert!(lines[2].to_string().contains("╭─ rust"), "{text}");
        assert!(
            lines
                .iter()
                .any(|line| line.to_string().contains("│ let x = 1;")),
            "{text}"
        );
        assert!(
            lines.iter().any(|line| line.to_string().contains('╰')),
            "{text}"
        );
        assert_eq!(lines[lines.len() - 2].to_string(), "");
        assert_eq!(lines.last().expect("after line").to_string(), "after");
        for line in &lines[2..=4] {
            assert_eq!(display_width(&line.to_string()), 24, "{line:?}");
        }
    }

    #[test]
    fn code_blocks_apply_syntax_highlighting_and_plain_fallback() {
        let theme = Theme::dark();
        let highlighted = rendered("```rust\nlet x = 1;\n```", 32);
        let body = highlighted
            .iter()
            .find(|line| line.to_string().contains("let x"))
            .expect("highlighted code body line");
        assert!(
            body.spans.iter().any(|span| {
                !span.content.trim().is_empty()
                    && span.content.as_ref() != "│ "
                    && span.style.bg == Some(theme.element_bg)
                    && span.style.fg != Some(theme.text)
            }),
            "{body:?}"
        );

        let plain = rendered("```definitely-not-a-language\nlet x = 1;\n```", 32);
        let body = plain
            .iter()
            .find(|line| line.to_string().contains("let x"))
            .expect("plain code body line");
        assert!(
            body.spans.iter().any(|span| {
                span.content.as_ref() == "let x = 1;"
                    && span.style.fg == Some(theme.text)
                    && span.style.bg == Some(theme.element_bg)
            }),
            "{body:?}"
        );
    }

    #[test]
    fn latex_parenthesis_and_bracket_delimiters_render_outside_code() {
        let markdown = r"inline \(e^{i\pi}+1=0\).

\[\int_{-\infty}^{+\infty} e^{-x^2}\,dx = \sqrt{\pi}\]

`\(code\)`

```text
\[code block\]
```";
        let document =
            render_markdown_document(markdown, Theme::dark(), MarkdownRenderOptions::new(80));
        let text = document
            .lines
            .iter()
            .map(|line| render_span_text(&line.spans))
            .collect::<Vec<_>>()
            .join("\n");
        assert!(text.contains("e^(iπ)+1 = 0"), "{text}");
        assert!(text.contains('∫'), "{text}");
        assert!(text.contains("√π"), "{text}");
        assert!(text.contains(r"\(code\)"), "{text}");
        assert!(text.contains(r"\[code block\]"), "{text}");
        assert!(document.validate(), "{document:?}");
    }

    #[test]
    fn math_delimiter_normalization_requires_pairs_and_respects_code_boundaries() {
        let markdown = "paired \\(x\\)\nunclosed \\(x\nescaped \\\\(x\\\\)\n`\\(inline code\\)`\n````text\n\\[fenced code\\]\n```\n\\(still fenced\\)\n````\nmath \\(y\\)";
        assert_eq!(
            normalize_markdown_math_delimiters(markdown),
            "paired $x$\nunclosed \\(x\nescaped \\\\(x\\\\)\n`\\(inline code\\)`\n````text\n\\[fenced code\\]\n```\n\\(still fenced\\)\n````\nmath $y$"
        );
        assert_eq!(
            normalize_markdown_math_delimiters("`unclosed \\(x\\)"),
            "`unclosed $x$"
        );
        assert_eq!(
            normalize_markdown_math_delimiters("mismatched \\(x\\]"),
            "mismatched \\(x\\]"
        );
        assert_eq!(
            normalize_markdown_math_delimiters("unclosed \\(x\nlater \\)"),
            "unclosed \\(x\nlater \\)"
        );
        assert_eq!(
            normalize_markdown_math_delimiters("unclosed \\[x\n\nlater \\]"),
            "unclosed \\[x\n\nlater \\]"
        );
        assert_eq!(
            normalize_markdown_math_delimiters("display \\[x +\n y\\]"),
            "display $$x +\n y$$"
        );
    }

    #[test]
    fn mermaid_dag_renders_and_invalid_graph_falls_back_to_code_card() {
        let supported = render_markdown_document(
            "```mermaid\ngraph TD\nA[Start]\nB[Finish]\nA --> B\n```",
            Theme::dark(),
            MarkdownRenderOptions::new(32),
        );
        let supported_text = supported
            .lines
            .iter()
            .map(|line| render_span_text(&line.spans))
            .collect::<Vec<_>>()
            .join("\n");
        assert!(supported_text.contains("Start"), "{supported_text}");
        assert!(supported_text.contains("Finish"), "{supported_text}");
        // 二维画布：节点框化，垂直箭头连接。
        assert!(supported_text.contains('╭'), "{supported_text}");
        assert!(supported_text.contains('│'), "{supported_text}");
        assert!(supported_text.contains('v'), "{supported_text}");
        assert!(!supported_text.contains("╭─ mermaid"), "{supported_text}");
        assert!(supported.validate(), "{supported:?}");

        let invalid = rendered("```mermaid\ngraph TD\nA --> B\nB --> A\n```", 32);
        let invalid_text = invalid
            .iter()
            .map(Line::to_string)
            .collect::<Vec<_>>()
            .join("\n");
        assert!(invalid_text.contains("╭─ mermaid"), "{invalid_text}");
    }

    #[test]
    fn mermaid_flowchart_labels_and_sequence_diagram_render() {
        let flowchart = render_markdown_document(
            "```mermaid\nflowchart TD\nA[开始] --> B{渲染类型}\nB -->|LaTeX| C[解析数学公式]\nB -->|Mermaid| D[解析流程图]\n```",
            Theme::dark(),
            MarkdownRenderOptions::new(80),
        );
        let flowchart_text = flowchart
            .lines
            .iter()
            .map(|line| render_span_text(&line.spans))
            .collect::<Vec<_>>()
            .join("\n");
        assert!(flowchart_text.contains("渲染类型"), "{flowchart_text}");
        assert!(flowchart_text.contains("LaTeX"), "{flowchart_text}");
        assert!(flowchart_text.contains("Mermaid"), "{flowchart_text}");
        assert!(!flowchart_text.contains("╭─ mermaid"), "{flowchart_text}");
        assert!(flowchart.validate(), "{flowchart:?}");

        let sequence = render_markdown_document(
            "```mermaid\nsequenceDiagram\nparticipant U as 用户\nparticipant C as 客户端\nparticipant R as 渲染器\nU->>C: 提交 Markdown\nC->>R: 发送 LaTeX / Mermaid 内容\nR-->>C: 返回渲染结果\n```",
            Theme::dark(),
            MarkdownRenderOptions::new(80),
        );
        let sequence_text = sequence
            .lines
            .iter()
            .map(|line| render_span_text(&line.spans))
            .collect::<Vec<_>>()
            .join("\n");
        assert!(sequence_text.contains("用户"), "{sequence_text}");
        assert!(sequence_text.contains("客户端"), "{sequence_text}");
        assert!(sequence_text.contains("渲染器"), "{sequence_text}");
        assert!(
            sequence_text
                .lines()
                .any(|line| line.contains("提交 Markdown") && line.contains('▶')),
            "{sequence_text}"
        );
        assert!(
            sequence_text
                .lines()
                .any(|line| line.contains("返回渲染结果") && line.contains('◀')),
            "{sequence_text}"
        );
        assert!(!sequence_text.contains("╭─ mermaid"), "{sequence_text}");
        assert!(sequence.validate(), "{sequence:?}");

        let structured = rendered(
            "```mermaid\nsequenceDiagram\nparticipant A as A\nloop retry\nA->>A: again\nend\n```",
            80,
        );
        let structured_text = structured
            .iter()
            .map(Line::to_string)
            .collect::<Vec<_>>()
            .join("\n");
        assert!(!structured_text.contains("╭─ mermaid"), "{structured_text}");
        assert!(structured_text.contains("loop retry"), "{structured_text}");
        assert!(
            structured_text.contains("A ──▶ A  again"),
            "{structured_text}"
        );
        assert!(structured_text.contains("end"), "{structured_text}");

        let too_wide = rendered(
            "```mermaid\nsequenceDiagram\nparticipant A as A\nparticipant B as B\nA->>B: a message that cannot fit\n```",
            12,
        );
        let too_wide_text = too_wide
            .iter()
            .map(Line::to_string)
            .collect::<Vec<_>>()
            .join("\n");
        assert!(too_wide_text.contains("╭─ mermaid"), "{too_wide_text}");
    }

    #[test]
    fn common_mermaid_families_render_and_fallback_cleanly() {
        let cases = [
            (
                "```mermaid\nclassDiagram\nclass User {\n  +name String\n}\nclass Service\nUser --> Service : calls\n```",
                "calls",
            ),
            (
                "```mermaid\nerDiagram\nUSER {\n  string name\n}\nORDER {\n  int id\n}\nUSER ||--o{ ORDER : creates\n```",
                "creates",
            ),
            (
                "```mermaid\ngantt\ntitle Release plan\ndateFormat YYYY-MM-DD\nsection Client\nImplement renderer : active, render, 2026-08-09, 3d\n```",
                "Implement renderer",
            ),
            (
                "```mermaid\nstateDiagram-v2\nstate \"Waiting\" as Waiting\nstate Done\n[*] --> Waiting : start\nWaiting --> Done : finish\nDone --> [*]\n```",
                "finish",
            ),
        ];
        for (markdown, expected) in cases {
            let document =
                render_markdown_document(markdown, Theme::dark(), MarkdownRenderOptions::new(80));
            let text = document
                .lines
                .iter()
                .map(|line| render_span_text(&line.spans))
                .collect::<Vec<_>>()
                .join("\n");
            assert!(text.contains(expected), "{text}");
            assert!(!text.contains("╭─ mermaid"), "{text}");
            assert!(document.validate(), "{document:?}");
        }

        for markdown in [
            "```mermaid\npie\ntitle unsupported\n```",
            "```mermaid\nclassDiagram\nclass A {\n  +x\n```",
            "```mermaid\nstateDiagram\ndirection LR\nA --> B\n```",
        ] {
            let text = rendered(markdown, 80)
                .iter()
                .map(Line::to_string)
                .collect::<Vec<_>>()
                .join("\n");
            assert!(text.contains("╭─ mermaid"), "{text}");
        }

        let narrow = rendered(
            "```mermaid\ngantt\ntitle A release plan that cannot fit\n```",
            12,
        )
        .iter()
        .map(Line::to_string)
        .collect::<Vec<_>>()
        .join("\n");
        assert!(narrow.contains("╭─ mermaid"), "{narrow}");
    }

    #[test]
    fn mermaid_supports_directions_shapes_edge_styles_comments_and_subgraphs() {
        let source = concat!(
            "```mermaid\n",
            // 方向别名 + 注释 + 分号多语句
            "flowchart BT\n",
            "%% a comment line\n",
            "subgraph SG[服务层]\n",
            "A[开始] --> B[(数据库)]\n",
            "B --> C((缓存)); C ==> D{{结果}}\n",
            "end\n",
            "D --- E[/中间/]\n",
            "E -.-> F[结束]\n",
            "```",
        );
        let document =
            render_markdown_document(source, Theme::dark(), MarkdownRenderOptions::new(80));
        assert!(document.validate(), "{document:?}");
        let text = document
            .lines
            .iter()
            .map(|line| render_span_text(&line.spans))
            .collect::<Vec<_>>()
            .join("\n");
        assert!(!text.contains("╭─ mermaid"), "{text}");
        assert!(text.contains("开始"), "{text}");
        assert!(text.contains("数据库"), "{text}");
        assert!(text.contains("缓存"), "{text}");
        assert!(text.contains("结果"), "{text}");
        assert!(text.contains("中间"), "{text}");
        assert!(text.contains("结束"), "{text}");
        // 二维画布：节点框化 + 垂直连线。
        assert!(text.contains('╭'), "{text}");
        assert!(text.contains('│'), "{text}");
        assert!(text.contains('v'), "{text}");
    }

    #[test]
    fn mermaid_edge_text_labels_and_more_directions_render() {
        let source = concat!(
            "```mermaid\n",
            "graph RL\n",
            "A -- text --> B\n",
            "C ==>|thick| D\n",
            "E -. dashed .-> F\n",
            "```",
        );
        let document =
            render_markdown_document(source, Theme::dark(), MarkdownRenderOptions::new(80));
        assert!(document.validate(), "{document:?}");
        let text = document
            .lines
            .iter()
            .map(|line| render_span_text(&line.spans))
            .collect::<Vec<_>>()
            .join("\n");
        assert!(!text.contains("╭─ mermaid"), "{text}");
        assert!(text.contains("text"), "{text}");
        assert!(text.contains("thick"), "{text}");
        assert!(text.contains("dashed"), "{text}");
    }

    #[test]
    fn mermaid_supports_tb_bu_and_lr_aliases() {
        for dir in ["TB", "TD", "BT", "LR", "RL"] {
            let document = render_markdown_document(
                &format!("```mermaid\ngraph {dir}\nA[Start] --> B[End]\n```"),
                Theme::dark(),
                MarkdownRenderOptions::new(40),
            );
            assert!(document.validate(), "{document:?}");
            let text = document
                .lines
                .iter()
                .map(|line| render_span_text(&line.spans))
                .collect::<Vec<_>>()
                .join("\n");
            assert!(text.contains("Start"), "{dir}: {text}");
            assert!(text.contains("End"), "{dir}: {text}");
            assert!(!text.contains("╭─ mermaid"), "{dir}: {text}");
            // 垂直与横向布局都保留方向对应的连线。
            if matches!(dir, "TB" | "TD" | "BT") {
                assert!(text.contains('╭'), "{dir}: {text}");
                assert!(text.contains('│'), "{dir}: {text}");
            } else if dir == "LR" {
                assert!(text.contains('▶'), "{dir}: {text}");
            } else {
                assert!(text.contains('◀'), "{dir}: {text}");
            }
        }
    }

    #[test]
    fn math_renders_unicode_ast_and_falls_back_atomically_when_too_wide() {
        let supported = rendered("inline $\\alpha^2$ and $$x_1$$", 32);
        let supported_text = supported
            .iter()
            .map(Line::to_string)
            .collect::<Vec<_>>()
            .join("\n");
        assert!(supported_text.contains("α²"), "{supported_text}");
        assert!(supported_text.contains("x₁"), "{supported_text}");

        let exact_width = render_markdown_document(
            "$\\alpha^2$",
            Theme::dark(),
            MarkdownRenderOptions::new(display_width("α²")),
        );
        assert_eq!(
            exact_width
                .lines
                .iter()
                .map(|line| render_span_text(&line.spans))
                .collect::<String>(),
            "α²"
        );
        assert!(
            exact_width
                .lines
                .iter()
                .flat_map(|line| &line.spans)
                .all(|span| span.copy_mode == crate::tui::transcript_render::CopyMode::Atomic),
            "{exact_width:?}"
        );

        let too_wide = rendered("$\\frac{1}{x}$", 2);
        let too_wide_text = too_wide
            .iter()
            .map(Line::to_string)
            .collect::<Vec<_>>()
            .join("");
        assert!(too_wide_text.contains("\\frac{1}{x}"), "{too_wide_text}");
    }

    #[test]
    fn math_box_layout_supports_frac_sqrt_integral_matrices_and_scripts() {
        let document = render_markdown_document(
            "$$\\frac{\\sqrt{x}}{2} + \\int_0^1 f(x) dx$$\n\n$$\\begin{pmatrix}a&b\\\\c&d\\end{pmatrix}$$",
            Theme::dark(),
            MarkdownRenderOptions::new(64),
        );
        let text = document
            .lines
            .iter()
            .map(|line| render_span_text(&line.spans))
            .collect::<Vec<_>>()
            .join("\n");
        assert!(text.contains('─'), "{text}");
        assert!(text.contains('√'), "{text}");
        assert!(text.contains('∫'), "{text}");
        assert!(text.contains("⎛ a │ b ⎞"), "{text}");
        assert!(text.contains("⎝ c │ d ⎠"), "{text}");
        assert!(document.validate(), "{document:?}");
    }

    #[test]
    fn transformed_mermaid_output_has_exact_label_ranges_and_isolated_nodes() {
        let prefixed = render_markdown_document(
            "> ```mermaid\n> graph LR\n> A[One]\n> B[Two]\n> C[Alone]\n> A --> B\n> ```",
            Theme::dark(),
            MarkdownRenderOptions::new(40),
        );
        let prefixed_text = prefixed
            .lines
            .iter()
            .map(|line| render_span_text(&line.spans))
            .collect::<Vec<_>>();
        let rendered_prefixed_lines = prefixed_text
            .iter()
            .filter(|text| text.contains("One") || text.contains("Two") || text.contains("Alone"))
            .collect::<Vec<_>>();
        assert!(rendered_prefixed_lines.len() >= 2);
        for label in ["One", "Two", "Alone"] {
            assert!(
                rendered_prefixed_lines
                    .iter()
                    .any(|line| line.contains(label)),
                "missing {label}: {prefixed_text:?}"
            );
        }
        assert!(
            rendered_prefixed_lines
                .iter()
                .all(|line| line.starts_with("│ "))
        );

        let markdown = "```mermaid\ngraph LR\nA[One]\nB[Two]\nC[Alone]\nA --> B\n```";
        let document =
            render_markdown_document(markdown, Theme::dark(), MarkdownRenderOptions::new(40));
        assert!(document.validate(), "{document:?}");
        assert!(
            document
                .lines
                .iter()
                .flat_map(|line| &line.spans)
                .any(|span| span.text == "Alone")
        );
        let source = &document.source_blocks[0].source;
        for span in document.lines.iter().flat_map(|line| &line.spans) {
            if let Some(range) = span.source {
                assert_eq!(
                    source
                        .chars()
                        .skip(range.start)
                        .take(range.end - range.start)
                        .collect::<String>(),
                    span.text
                );
            }
        }

        let sequence_source =
            "sequenceDiagram\nparticipant A as A\nparticipant B as 接收方\nA->>B: A sends\n";
        let sequence = render_markdown_document(
            &format!("```mermaid\n{sequence_source}```"),
            Theme::dark(),
            MarkdownRenderOptions::new(40),
        );
        assert!(sequence.validate(), "{sequence:?}");
        let sequence_block = sequence
            .source_blocks
            .iter()
            .position(|block| block.source == sequence_source)
            .expect("sequence source block");
        for span in sequence.lines.iter().flat_map(|line| &line.spans) {
            let Some(range) = span.source else {
                continue;
            };
            if range.block_index != sequence_block {
                continue;
            }
            assert_eq!(
                sequence.source_blocks[range.block_index]
                    .source
                    .chars()
                    .skip(range.start)
                    .take(range.end - range.start)
                    .collect::<String>(),
                span.text
            );
        }
    }

    #[test]
    fn transformed_math_has_atomic_full_source_mapping_and_width_fallback() {
        let transformed = render_markdown_document(
            "The value is $$\\frac{1}{x}$$.",
            Theme::dark(),
            MarkdownRenderOptions::new(40),
        );
        assert!(transformed.validate(), "{transformed:?}");
        let math_span = transformed
            .lines
            .iter()
            .flat_map(|line| &line.spans)
            .find(|span| span.text.contains('─'))
            .expect("transformed fraction span");
        assert_eq!(
            math_span.copy_mode,
            crate::tui::transcript_render::CopyMode::Atomic
        );
        let range = math_span.source.expect("atomic source range");
        assert_eq!(
            transformed.source_blocks[range.block_index].source,
            "\\frac{1}{x}"
        );
        assert_eq!(range.start, 0);
        assert_eq!(range.end, "\\frac{1}{x}".chars().count());

        let fallback = render_markdown_document(
            "$\\frac{1}{x}$",
            Theme::dark(),
            MarkdownRenderOptions::new(2),
        );
        assert!(fallback.validate(), "{fallback:?}");
        let fallback_text = fallback
            .lines
            .iter()
            .map(|line| render_span_text(&line.spans))
            .collect::<String>();
        assert!(fallback_text.contains("\\frac{1}{x}"), "{fallback_text}");
        assert!(
            fallback
                .lines
                .iter()
                .flat_map(|line| &line.spans)
                .any(|span| span.copy_mode == crate::tui::transcript_render::CopyMode::Atomic)
        );
    }

    #[test]
    fn display_math_accepts_normalized_multiline_input_and_rejects_malformed_groups() {
        assert!(crate::tui::math::render("\\int _{0}\n ^{1} f(x) + \\frac{1}{x}", true).is_some());
        let document = render_markdown_document(
            "$$\\int_{0}^{1} f(x)\\frac{1}{x}$$",
            Theme::dark(),
            MarkdownRenderOptions::new(64),
        );
        let text = document
            .lines
            .iter()
            .map(|line| render_span_text(&line.spans))
            .collect::<Vec<_>>()
            .join("\n");
        assert!(text.contains('∫'), "{text}");
        assert!(text.contains('─'), "{text}");
        assert!(document.validate(), "{document:?}");

        let malformed = render_markdown_document(
            "$\\frac{1}{x$",
            Theme::dark(),
            MarkdownRenderOptions::new(64),
        );
        let fallback = malformed
            .lines
            .iter()
            .map(|line| render_span_text(&line.spans))
            .collect::<String>();
        assert!(fallback.contains("\\frac{1}{x"), "{fallback}");
        assert!(malformed.validate(), "{malformed:?}");
    }

    #[test]
    fn latex_official_corpus_inline_and_malformed_cases() {
        let cases = [
            (r"\mathbb{C}^3 \to \mathbb{C}^3", "ℂ³ → ℂ³"),
            (
                r"\{3x+2y,\; 27x^2-4z-1,\; x(x-1)(x+1)\} \quad\Rightarrow\quad x \in \{0, \pm 1\},",
                "{3x+2y, 27x²-4z-1, x(x-1)(x+1)} ⇒ x ∈ {0, ± 1},",
            ),
            (r"F_1 = -\frac{1}{4x^2}.", "F₁ = -1/(4x²)."),
            (r"\mathbb{C}^*", "ℂ^*"),
            (
                r"s \mapsto (s,\, -\tfrac{3}{2s},\, \tfrac{13}{2s^2})",
                "s ↦ (s, -3/(2s), 13/(2s²))",
            ),
            (
                r"\boxed{1\ \text{milliwatt per square metre}}",
                "[1 milliwatt per square metre]",
            ),
            (
                r"\pi(2.5\ \text{km})^2 = 19.6\ \text{km}^2",
                "π(2.5 km)² = 19.6 km²",
            ),
            (
                r"\det\!\left(\frac{\partial(F_1,F_2,F_3)}{\partial(x,y,z)}\right)=-2.",
                "det((∂(F₁,F₂,F₃))/(∂(x,y,z))) = -2.",
            ),
            (r"e^{i\pi}+1=0", "e^(iπ)+1 = 0"),
            (
                r"x=\frac{-b\pm\sqrt{b^2-4ac}}{2a}",
                "x = (-b±√(b²-4ac))/(2a)",
            ),
            (
                r"\int_0^\infty e^{-x^2}\,dx=\frac{\sqrt{\pi}}{2}",
                "∫₀^∞ e^(-x²) dx = (√π)/2",
            ),
            (
                r"\sum_{n=1}^{\infty}\frac{1}{n^2}=\frac{\pi^2}{6}",
                "∑ₙ₌₁^∞1/(n²) = π²/6",
            ),
            (r"\lim_{x\to 0}\frac{\sin x}{x}=1", "lim[x→0] (sin x)/x = 1"),
            (
                r"\sqrt[2]{x}+\sqrt[3]{x}+\sqrt[4]{x}+\sqrt[n]{x}+\sqrt[k]{x+1}",
                "√x+∛x+∜x+ⁿ√x+ᵏ√(x+1)",
            ),
            (
                r"\acute{x}+\grave{y}+\widehat{xyz}+\overrightarrow{AB}",
                "x́+ỳ+widehat(xyz)+overrightarrow(AB)",
            ),
            (
                r"\textnormal{hello}+\mbox{world}+\boldsymbol{x}",
                "hello+world+x",
            ),
            (r"A\not\subseteq B,\quad x\not\in X", "A ⊈ B, x ∉ X"),
            (
                r"\lvert{x}\rvert+\lVert{v}\rVert+\left.\frac{dy}{dx}\right|_{x=0}",
                "|x|+‖v‖+dy/(dx)|ₓ₌₀",
            ),
            (
                r"\operatorname*{arg\,max}_{x\in X} f(x)",
                "arg max[x∈X] f(x)",
            ),
            (r"a\bmod n,\quad a\equiv b\pmod n", "a mod n, a ≡ b (mod n)"),
            (
                r"\overset{!}{=}+\underset{n}{x}+\stackrel{def}{=}",
                "=^!+xₙ+=ᵈᵉᶠ",
            ),
        ];
        for (source, expected) in cases {
            assert_eq!(
                crate::tui::math::render_text(source, false).as_deref(),
                Some(expected),
                "{source}"
            );
        }
        for source in [
            r"x + \unknown{y}",
            r"\frac{1}{x",
            "x}",
            r"\begin{matrix}1 & 2",
            r"x\\",
        ] {
            assert!(
                crate::tui::math::render_text(source, false).is_none(),
                "{source}"
            );
        }
    }

    #[test]
    fn latex_official_corpus_display_golden_layouts() {
        let cases = [
            (r"\sum_{i=0}^n x_i", " n\n ∑  xᵢ\ni=0"),
            (r"\min_{x\in X} f(x)", "min f(x)\nx∈X"),
            (
                r"\operatorname*{arg\,max}_{x\in X} f(x)",
                "arg max f(x)\n  x∈X",
            ),
            (
                r"x=\frac{-b\pm\sqrt{b^2-4ac}}{2a}",
                "    -b±√(b²-4ac)\nx = ────────────\n         2a",
            ),
            (r"\frac{x^2+1}{x-1}", "x²+1\n────\nx-1"),
            (
                r"\begin{cases}a & x<0 \\ b & x=0 \\ c & x>0\end{cases}",
                "⎧ a if x < 0\n⎨ b if x = 0\n⎩ c if x > 0",
            ),
            (
                r"\begin{pmatrix}1&200\\3000&4\end{pmatrix}",
                "⎛ 1    │ 200 ⎞\n⎝ 3000 │ 4   ⎠",
            ),
        ];
        for (source, expected) in cases {
            assert_eq!(
                crate::tui::math::render_text(source, true).as_deref(),
                Some(expected),
                "{source}"
            );
        }
    }

    #[test]
    fn ordered_lists_blockquotes_tasks_and_links_keep_visual_semantics() {
        let lines = rendered("> 1. [x] done\n> 2. [ ] [docs](https://example.com)", 80);
        let text = lines
            .iter()
            .map(Line::to_string)
            .collect::<Vec<_>>()
            .join("\n");
        assert!(text.contains("│ 1. ☑ done"), "{text}");
        assert!(text.contains("│ 2. ☐ docs <https://example.com>"), "{text}");
    }

    #[test]
    fn tables_keep_headers_values_separator_and_wide_cells() {
        let md = "| 套件 | 结果 |\n| --- | --- |\n| runtime_context::tests | 12 passed (含 heal / replace_frames) |\n| runtime::tests | 184 passed |";
        let lines = rendered(md, 80);
        let text = lines
            .iter()
            .map(Line::to_string)
            .collect::<Vec<_>>()
            .join("\n");
        assert!(text.contains("套件"), "{text}");
        assert!(text.contains('┼'), "{text}");
        assert!(
            text.contains("12 passed (含 heal / replace_frames)"),
            "{text}"
        );
    }
}
