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
        for event in Parser::new_ext(markdown, markdown_options()) {
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
            Event::Code(code) | Event::InlineMath(code) | Event::DisplayMath(code) => {
                self.push_inline_code(&code)
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
) {
    if let Some(last) = target.last_mut()
        && last.style == style
        && last.interaction == interaction
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
        target.push(RenderSpan::source_with_interaction(
            text,
            style,
            source,
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
            let source = span
                .source
                .map(|range| SourceRange::new(range.block_index, offset, offset + count));
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
                span.source
                    .map(|range| SourceRange::new(range.block_index, offset, offset + count)),
                span.interaction.clone(),
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

fn markdown_options() -> Options {
    Options::ENABLE_TABLES
        | Options::ENABLE_FOOTNOTES
        | Options::ENABLE_STRIKETHROUGH
        | Options::ENABLE_TASKLISTS
        | Options::ENABLE_HEADING_ATTRIBUTES
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
    fn native_document_metadata_survives_inline_styles_and_wraps() {
        let document = render_markdown_document(
            "same *same* **same** `same` [same](https://example.test)",
            Theme::dark(),
            MarkdownRenderOptions::new(8),
        );
        assert!(document.validate());
        let copied = document
            .lines
            .iter()
            .flat_map(|line| &line.spans)
            .filter(|span| span.source.is_some())
            .map(|span| span.text.as_str())
            .collect::<String>();
        assert_eq!(copied, "same same same same same<https://example.test>");
        assert!(
            document
                .breaks
                .iter()
                .any(|boundary| *boundary == Break::SoftWrap)
        );
    }

    #[test]
    fn markdown_chrome_is_decoration_while_code_and_table_values_are_copyable() {
        let document = render_markdown_document(
            "```rust\ne\u{301}👩‍💻\n```\n\n| left | right |\n| --- | --- |\n| one | two |",
            Theme::dark(),
            MarkdownRenderOptions::new(40),
        );
        assert!(document.validate());
        let copied = document
            .lines
            .iter()
            .flat_map(|line| &line.spans)
            .filter(|span| span.source.is_some())
            .map(|span| span.text.as_str())
            .collect::<String>();
        assert!(copied.contains("e\u{301}👩‍💻"));
        assert!(copied.contains("left"));
        assert!(copied.contains("right"));
        assert!(copied.contains("one"));
        assert!(copied.contains("two"));
        assert!(!copied.contains("│"));
        assert!(!copied.contains("┼"));
    }

    #[test]
    fn author_breaks_are_hard_and_layout_breaks_are_soft() {
        let document = render_markdown_document(
            "first  \nsecond\n\nthird fourth fifth",
            Theme::dark(),
            MarkdownRenderOptions::new(8),
        );
        assert!(document.validate());
        assert!(document.breaks.contains(&Break::HardBreak));
        assert!(document.breaks.contains(&Break::BlockBreak));
        assert!(document.breaks.contains(&Break::SoftWrap));
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
    fn inline_styles_survive_wrapping() {
        let lines = rendered("**abcdef** `ghij`", 8);
        assert!(lines.len() >= 2, "{lines:?}");
        assert!(
            lines[0]
                .spans
                .iter()
                .any(|span| span.style.add_modifier.contains(Modifier::BOLD)),
            "{:?}",
            lines[0]
        );
        assert!(
            lines
                .iter()
                .flat_map(|line| line.spans.iter())
                .any(|span| span.style.bg == Some(Theme::dark().element_bg)),
            "{lines:?}"
        );
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
    fn code_blocks_are_width_safe_and_keep_quote_context() {
        for width in 1..=10 {
            let lines = rendered("```very-long-language-label\nabcdef\n```", width);
            for line in lines {
                let measured = display_width(&line.to_string());
                assert!(
                    measured <= width,
                    "width={width} measured={measured}: {line:?}"
                );
            }
        }

        let lines = rendered("> ```rust\n> \n> x\n> ```", 24);
        let text = lines
            .iter()
            .map(Line::to_string)
            .collect::<Vec<_>>()
            .join("\n");
        assert!(text.contains("│ ╭─ rust"), "{text}");
        assert!(text.contains("│ │ x"), "{text}");
        assert_eq!(lines.len(), 4, "{text}");
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

    #[test]
    fn rules_fill_available_markdown_width() {
        let lines = rendered("before\n\n---\n\nafter", 120);
        let rule = lines
            .iter()
            .find(|line| line.to_string().chars().all(|ch| ch == '─'))
            .expect("rule line present");
        assert_eq!(display_width(&rule.to_string()), 120, "{rule:?}");
    }
}
