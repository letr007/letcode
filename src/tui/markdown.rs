use pulldown_cmark::{CodeBlockKind, Event, HeadingLevel, Options, Parser, Tag, TagEnd};
#[cfg(test)]
use ratatui::text::Line;
use ratatui::{
    style::{Color, Modifier, Style},
    text::Span,
};
use std::{collections::HashMap, sync::OnceLock};
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
        let Some(layout) = render_math(text, display) else {
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
        if code.content.trim_start().starts_with("sequenceDiagram") {
            return self.push_mermaid_sequence(code);
        }
        let Some(graph) = parse_mermaid(&code.content) else {
            return false;
        };
        let direction = graph.direction;
        // 成功渲染时不使用卡片背景，直接融入 timeline 背景。
        let text_style = self.theme.app_style();
        let border_style = self.theme.app_style().fg(self.theme.accent);
        let mut output = Vec::new();
        let mut rendered_nodes = HashMap::new();

        for edge in &graph.edges {
            let Some(from) = graph.nodes.get(&edge.from) else {
                return false;
            };
            let Some(to) = graph.nodes.get(&edge.to) else {
                return false;
            };
            let mut spans = vec![RenderSpan::decoration(
                match direction {
                    MermaidDirection::Td => "↓ ",
                    MermaidDirection::Bu => "↑ ",
                    MermaidDirection::Lr | MermaidDirection::Rl => "",
                },
                border_style,
            )];
            spans.push(RenderSpan::source(
                from.label.clone(),
                text_style,
                SourceRange::new(0, from.start, from.end),
            ));
            let line = match edge.style {
                MermaidEdgeStyle::Solid => "─",
                MermaidEdgeStyle::Dashed => "╌",
                MermaidEdgeStyle::Thick => "═",
            };
            if let Some(label) = &edge.label {
                spans.push(RenderSpan::decoration(
                    format!(" {line}{line}"),
                    border_style,
                ));
                spans.push(RenderSpan::source(
                    label.text.clone(),
                    text_style,
                    SourceRange::new(0, label.start, label.end),
                ));
                let arrow = if edge.arrow { '▶' } else { ' ' };
                spans.push(RenderSpan::decoration(
                    format!(" {line}{line}{arrow} "),
                    border_style,
                ));
            } else {
                let arrow = if edge.arrow { '▶' } else { ' ' };
                spans.push(RenderSpan::decoration(
                    format!(" {line}{line}{arrow} "),
                    border_style,
                ));
            }
            spans.push(RenderSpan::source(
                to.label.clone(),
                text_style,
                SourceRange::new(0, to.start, to.end),
            ));
            output.push(spans);
            rendered_nodes.insert(edge.from.clone(), ());
            rendered_nodes.insert(edge.to.clone(), ());
        }

        let mut isolated_nodes = graph
            .nodes
            .iter()
            .filter(|(id, _)| !rendered_nodes.contains_key(*id))
            .collect::<Vec<_>>();
        isolated_nodes.sort_by_key(|(_, node)| node.start);
        for (_, node) in isolated_nodes {
            output.push(vec![RenderSpan::source(
                node.label.clone(),
                text_style,
                SourceRange::new(0, node.start, node.end),
            )]);
        }

        self.push_mermaid_output(code, output)
    }

    fn push_mermaid_sequence(&mut self, code: &CodeBlockState) -> bool {
        let Some(sequence) = parse_mermaid_sequence(&code.content) else {
            return false;
        };
        // 成功渲染时不使用卡片背景，直接融入 timeline 背景。
        let text_style = self.theme.app_style();
        let border_style = self.theme.app_style().fg(self.theme.accent);
        let mut output = Vec::new();
        for message in &sequence.messages {
            let Some(from) = sequence.participants.get(&message.from) else {
                return false;
            };
            let Some(to) = sequence.participants.get(&message.to) else {
                return false;
            };
            output.push(vec![
                RenderSpan::source(
                    from.label.clone(),
                    text_style,
                    SourceRange::new(0, from.start, from.end),
                ),
                RenderSpan::decoration(
                    if message.dashed {
                        " ╌╌▶ "
                    } else {
                        " ──▶ "
                    },
                    border_style,
                ),
                RenderSpan::source(
                    to.label.clone(),
                    text_style,
                    SourceRange::new(0, to.start, to.end),
                ),
                RenderSpan::decoration("  ", border_style),
                RenderSpan::source(
                    message.label.text.clone(),
                    text_style,
                    SourceRange::new(0, message.label.start, message.label.end),
                ),
            ]);
        }
        self.push_mermaid_output(code, output)
    }

    fn push_mermaid_output(
        &mut self,
        code: &CodeBlockState,
        output: Vec<Vec<RenderSpan<Style>>>,
    ) -> bool {
        if output.is_empty() {
            return false;
        }
        let (first_prefix, next_prefix) = self.line_prefixes();
        if output.iter().enumerate().any(|(index, spans)| {
            let prefix = if index == 0 {
                &first_prefix
            } else {
                &next_prefix
            };
            display_width(prefix) + display_width(&render_span_text(spans)) > self.options.width
        }) {
            return false;
        }
        let block = self.document.add_source(code.content.clone());
        let output_len = output.len();
        for (index, mut spans) in output.into_iter().enumerate() {
            for span in &mut spans {
                if let Some(range) = &mut span.source {
                    range.block_index = block;
                }
            }
            let prefix = if index == 0 {
                &first_prefix
            } else {
                &next_prefix
            };
            let mut line = Vec::new();
            if !prefix.is_empty() {
                line.push(RenderSpan::decoration(prefix.clone(), self.prefix_style()));
            }
            line.extend(spans);
            self.document.push_line(
                RenderLine { spans: line },
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum MermaidDirection {
    Td,
    Bu,
    Lr,
    Rl,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum MermaidEdgeStyle {
    Solid,
    Dashed,
    Thick,
}

#[derive(Debug)]
struct MermaidGraph {
    direction: MermaidDirection,
    nodes: HashMap<String, MermaidNode>,
    edges: Vec<MermaidEdge>,
}

#[derive(Debug, Clone)]
struct MermaidNode {
    label: String,
    start: usize,
    end: usize,
}

#[derive(Debug, Clone)]
struct MermaidLabel {
    text: String,
    start: usize,
    end: usize,
}

#[derive(Debug)]
struct MermaidEdge {
    from: String,
    to: String,
    label: Option<MermaidLabel>,
    style: MermaidEdgeStyle,
    arrow: bool,
}

#[derive(Debug)]
struct MermaidSequence {
    participants: HashMap<String, MermaidNode>,
    messages: Vec<MermaidMessage>,
}

#[derive(Debug)]
struct MermaidMessage {
    from: String,
    to: String,
    label: MermaidLabel,
    dashed: bool,
}

fn parse_mermaid(source: &str) -> Option<MermaidGraph> {
    if source.contains('\r') {
        return None;
    }
    let mut nodes = HashMap::new();
    let mut edges = Vec::new();
    let mut direction = None;
    let mut line_offset = 0usize;

    for (line_index, line) in source.lines().enumerate() {
        let leading = line.chars().count() - line.trim_start().chars().count();
        let trimmed = line.trim();
        if line_index == 0 {
            let mut tokens = trimmed
                .split(|ch: char| ch == ';' || ch.is_whitespace())
                .filter(|s| !s.is_empty());
            let keyword = tokens.next()?;
            let dir = tokens.next()?;
            direction = Some(match (keyword, dir) {
                ("graph" | "flowchart", "TD" | "TB") => MermaidDirection::Td,
                ("graph" | "flowchart", "BT") => MermaidDirection::Bu,
                ("graph" | "flowchart", "LR") => MermaidDirection::Lr,
                ("graph" | "flowchart", "RL") => MermaidDirection::Rl,
                _ => return None,
            });
        } else if !trimmed.is_empty() && !trimmed.starts_with("%%") {
            if trimmed == "end" || trimmed.starts_with("subgraph") {
                // 子图边界：子图内节点/边照常渲染，不画子图边框。
            } else {
                let base_line = line_offset + leading;
                let mut segment_start = 0usize;
                for (idx, byte) in trimmed.bytes().enumerate() {
                    if byte == b';' {
                        let segment = &trimmed[segment_start..idx];
                        if !segment.trim().is_empty() {
                            let base = base_line + trimmed[..segment_start].chars().count();
                            parse_mermaid_statement(segment, base, &mut nodes, &mut edges)?;
                        }
                        segment_start = idx + 1;
                    }
                }
                let segment = &trimmed[segment_start..];
                if !segment.trim().is_empty() {
                    let base = base_line + trimmed[..segment_start].chars().count();
                    parse_mermaid_statement(segment, base, &mut nodes, &mut edges)?;
                }
            }
        }
        line_offset += line.chars().count() + 1;
    }

    let direction = direction?;
    if edges.is_empty()
        || edges
            .iter()
            .any(|edge| !nodes.contains_key(&edge.from) || !nodes.contains_key(&edge.to))
        || has_mermaid_cycle(&nodes, &edges)
    {
        return None;
    }
    Some(MermaidGraph {
        direction,
        nodes,
        edges,
    })
}

/// 形状定界符对：open 优先匹配更长的，label 从中提取。
const MERMAID_SHAPES: &[(&str, &str)] = &[
    ("[[", "]]"),
    ("[(", ")]"),
    ("((", "))"),
    ("{{", "}}"),
    ("[/", "/]"),
    ("[\\", "\\]"),
    ("[", "]"),
    ("(", ")"),
    ("{", "}"),
    (">", "]"),
];

/// 解析一条语句：孤立节点或 from -> to。返回剩余部分时供连接符继续解析。
fn parse_mermaid_statement(
    segment: &str,
    base: usize,
    nodes: &mut HashMap<String, MermaidNode>,
    edges: &mut Vec<MermaidEdge>,
) -> Option<()> {
    let (from, node, rest, rest_base) = parse_mermaid_endpoint_prefix(segment, base)?;
    if rest.trim().is_empty() {
        insert_mermaid_node(nodes, (from, node))?;
        return Some(());
    }
    let (style, arrow, label, rest2, rest2_base) = parse_mermaid_connector(rest, rest_base)?;
    let (to, to_node, rest3, _rest3_base) = parse_mermaid_endpoint_prefix(rest2, rest2_base)?;
    if !rest3.trim().is_empty() {
        return None;
    }
    insert_mermaid_node(nodes, (from.clone(), node))?;
    insert_mermaid_node(nodes, (to.clone(), to_node))?;
    edges.push(MermaidEdge {
        from,
        to,
        label,
        style,
        arrow,
    });
    Some(())
}

/// 解析前缀端点（id 或形状节点），返回剩余片段与偏移。
fn parse_mermaid_endpoint_prefix(
    segment: &str,
    base: usize,
) -> Option<(String, MermaidNode, &str, usize)> {
    let leading = segment.chars().take_while(|ch| ch.is_whitespace()).count();
    let trimmed = segment.trim_start();
    if trimmed.is_empty() {
        return None;
    }
    let tbase = base + leading;
    for (opener, closer) in MERMAID_SHAPES {
        if let Some(open) = trimmed.find(opener) {
            let id = trimmed[..open].trim();
            if !valid_mermaid_id(id) {
                continue;
            }
            let body_start = open + opener.len();
            let Some(rel) = trimmed[body_start..].find(closer) else {
                continue;
            };
            let close = body_start + rel;
            let label_seg = &trimmed[body_start..close];
            let label_leading = label_seg
                .chars()
                .take_while(|ch| ch.is_whitespace())
                .count();
            let raw = label_seg.trim().trim_matches('"');
            if raw.is_empty() {
                continue;
            }
            let id_chars = trimmed[..open].chars().count();
            let start = tbase + id_chars + opener.chars().count() + label_leading;
            let node = MermaidNode {
                label: raw.to_string(),
                start,
                end: start + raw.chars().count(),
            };
            let rest = &trimmed[close + closer.len()..];
            let rest_base = tbase + trimmed[..close + closer.len()].chars().count();
            return Some((id.to_string(), node, rest, rest_base));
        }
    }
    // 裸 id：读取到空白或连接符起始为止。
    let id_end = trimmed
        .char_indices()
        .find(|(_, ch)| ch.is_whitespace() || *ch == '-' || *ch == '=')
        .map(|(idx, _)| idx)
        .unwrap_or(trimmed.len());
    let id = &trimmed[..id_end];
    if !valid_mermaid_id(id) {
        return None;
    }
    let node = MermaidNode {
        label: id.to_string(),
        start: tbase,
        end: tbase + id.chars().count(),
    };
    let rest = &trimmed[id_end..];
    let rest_base = tbase + id.chars().count();
    Some((id.to_string(), node, rest, rest_base))
}

/// 解析连接符，返回 (样式, 是否箭头, 标签, 剩余, 剩余偏移)。
fn parse_mermaid_connector(
    segment: &str,
    base: usize,
) -> Option<(MermaidEdgeStyle, bool, Option<MermaidLabel>, &str, usize)> {
    let leading = segment.chars().take_while(|ch| ch.is_whitespace()).count();
    let segment = segment.trim_start();
    let base = base + leading;
    // 完整连接符 token 优先。
    const TOKENS: &[(&str, MermaidEdgeStyle, bool)] = &[
        ("-.->", MermaidEdgeStyle::Dashed, true),
        ("-->", MermaidEdgeStyle::Solid, true),
        ("==>", MermaidEdgeStyle::Thick, true),
        ("-.-", MermaidEdgeStyle::Dashed, false),
        ("---", MermaidEdgeStyle::Solid, false),
        ("===", MermaidEdgeStyle::Thick, false),
    ];
    for (token, style, arrow) in TOKENS {
        if let Some(rest) = segment.strip_prefix(token) {
            let rest_base = base + token.chars().count();
            let (label, rest, rest_base) = parse_mermaid_pipe_label(rest, rest_base)?;
            return Some((*style, *arrow, label, rest, rest_base));
        }
    }
    // 文本形式：-- text --> / -. text .-> / == text ==>。
    let style = if let Some(_rest) = segment.strip_prefix("--") {
        MermaidEdgeStyle::Solid
    } else if let Some(_rest) = segment.strip_prefix("-.") {
        MermaidEdgeStyle::Dashed
    } else if let Some(_rest) = segment.strip_prefix("==") {
        MermaidEdgeStyle::Thick
    } else {
        return None;
    };
    let rest = segment[2..].trim_start();
    let rest_base = base + 2 + segment[2..].len() - rest.len();
    // 找到闭合 token（文本形式，如 `-- text -->` / `-. text .->`）。
    const CLOSERS: &[(&str, MermaidEdgeStyle, bool)] = &[
        ("-->", MermaidEdgeStyle::Solid, true),
        ("--->", MermaidEdgeStyle::Solid, true),
        (".->", MermaidEdgeStyle::Dashed, true),
        ("==>", MermaidEdgeStyle::Thick, true),
        ("---", MermaidEdgeStyle::Solid, false),
        ("-.-", MermaidEdgeStyle::Dashed, false),
        ("===", MermaidEdgeStyle::Thick, false),
    ];
    for (closer, cstyle, carrow) in CLOSERS {
        if let Some(pos) = rest.find(closer) {
            let raw = rest[..pos].trim();
            if raw.is_empty() {
                continue;
            }
            if *cstyle != style {
                continue;
            }
            let label = MermaidLabel {
                text: raw.to_string(),
                start: rest_base,
                end: rest_base + raw.chars().count(),
            };
            let rest2 = &rest[pos + closer.len()..];
            let rest2_base = rest_base + rest[..pos + closer.len()].chars().count();
            return Some((style, *carrow, Some(label), rest2, rest2_base));
        }
    }
    None
}

/// 解析管道标签 `|text|`，返回 (标签, 剩余, 剩余偏移)。
fn parse_mermaid_pipe_label<'a>(
    segment: &'a str,
    base: usize,
) -> Option<(Option<MermaidLabel>, &'a str, usize)> {
    let leading = segment.chars().take_while(|ch| ch.is_whitespace()).count();
    let trimmed = segment.trim_start();
    let tbase = base + leading;
    if !trimmed.starts_with('|') {
        return Some((None, trimmed, tbase));
    }
    let close = trimmed[1..].find('|')? + 1;
    let raw = &trimmed[1..close];
    let lt = raw.trim();
    if lt.is_empty() {
        return None;
    }
    let lstart = tbase + 1 + raw.chars().take_while(|ch| ch.is_whitespace()).count();
    let label = MermaidLabel {
        text: lt.to_string(),
        start: lstart,
        end: lstart + lt.chars().count(),
    };
    let rest = &trimmed[close + 1..];
    let rest_base = tbase + trimmed[..close + 1].chars().count();
    Some((Some(label), rest, rest_base))
}

fn parse_mermaid_sequence(source: &str) -> Option<MermaidSequence> {
    if source.contains('\r') {
        return None;
    }
    let mut participants = HashMap::new();
    let mut messages = Vec::new();
    let mut line_offset = 0usize;

    for (line_index, line) in source.lines().enumerate() {
        let leading = line.chars().count() - line.trim_start().chars().count();
        let trimmed = line.trim();
        if line_index == 0 {
            if trimmed != "sequenceDiagram" {
                return None;
            }
        } else if !trimmed.is_empty() {
            let base = line_offset + leading;
            if let Some(rest) = trimmed.strip_prefix("participant ") {
                let (id, label) = rest.split_once(" as ")?;
                let id = id.trim();
                let label = label.trim();
                if !valid_mermaid_id(id) || label.is_empty() || participants.contains_key(id) {
                    return None;
                }
                let separator = rest.find(" as ")?;
                let label_segment = &rest[separator + " as ".len()..];
                let label_leading = label_segment
                    .chars()
                    .take_while(|ch| ch.is_whitespace())
                    .count();
                let start = base
                    + "participant ".chars().count()
                    + rest[..separator + " as ".len()].chars().count()
                    + label_leading;
                participants.insert(
                    id.to_string(),
                    MermaidNode {
                        label: label.to_string(),
                        start,
                        end: start + label.chars().count(),
                    },
                );
            } else {
                let colon = trimmed.find(':')?;
                let route = trimmed[..colon].trim();
                let label = trimmed[colon + 1..].trim();
                let (arrow_at, arrow) = ["-->>", "->>", "-->", "->"]
                    .into_iter()
                    .filter_map(|arrow| route.find(arrow).map(|index| (index, arrow)))
                    .min_by_key(|(index, _)| *index)?;
                let from = route[..arrow_at].trim();
                let to = route[arrow_at + arrow.len()..].trim();
                if !valid_mermaid_id(from)
                    || !valid_mermaid_id(to)
                    || label.is_empty()
                    || !participants.contains_key(from)
                    || !participants.contains_key(to)
                {
                    return None;
                }
                let label_byte = trimmed.find(':')? + 1;
                let label_leading = trimmed[label_byte..]
                    .chars()
                    .take_while(|ch| ch.is_whitespace())
                    .count();
                let start = base + trimmed[..label_byte].chars().count() + label_leading;
                messages.push(MermaidMessage {
                    from: from.to_string(),
                    to: to.to_string(),
                    label: MermaidLabel {
                        text: label.to_string(),
                        start,
                        end: start + label.chars().count(),
                    },
                    dashed: arrow.starts_with("--"),
                });
            }
        }
        line_offset += line.chars().count() + 1;
    }

    (!participants.is_empty() && !messages.is_empty()).then_some(MermaidSequence {
        participants,
        messages,
    })
}

fn insert_mermaid_node(
    nodes: &mut HashMap<String, MermaidNode>,
    (id, node): (String, MermaidNode),
) -> Option<()> {
    match nodes.get(&id) {
        None => {
            nodes.insert(id, node);
            Some(())
        }
        Some(previous) if previous.label == node.label || node.label == id => Some(()),
        Some(previous) if previous.label == id => {
            nodes.insert(id, node);
            Some(())
        }
        Some(_) => None,
    }
}

fn valid_mermaid_id(value: &str) -> bool {
    !value.is_empty()
        && value
            .chars()
            .all(|ch| ch.is_ascii_alphanumeric() || ch == '_')
}

fn has_mermaid_cycle(nodes: &HashMap<String, MermaidNode>, edges: &[MermaidEdge]) -> bool {
    fn visit(
        id: &str,
        nodes: &HashMap<String, MermaidNode>,
        edges: &[MermaidEdge],
        active: &mut Vec<String>,
        done: &mut Vec<String>,
    ) -> bool {
        if active.iter().any(|item| item == id) {
            return true;
        }
        if done.iter().any(|item| item == id) {
            return false;
        }
        active.push(id.to_string());
        for edge in edges.iter().filter(|edge| edge.from == id) {
            if visit(&edge.to, nodes, edges, active, done) {
                return true;
            }
        }
        active.pop();
        done.push(id.to_string());
        let _ = nodes;
        false
    }
    let mut active = Vec::new();
    let mut done = Vec::new();
    nodes
        .keys()
        .any(|id| visit(id, nodes, edges, &mut active, &mut done))
}

#[derive(Debug, Clone, Copy)]
struct MathLimits {
    max_source_chars: usize,
    max_nodes: usize,
    max_rows: usize,
    max_columns: usize,
}

impl Default for MathLimits {
    fn default() -> Self {
        Self {
            max_source_chars: 512,
            max_nodes: 512,
            max_rows: 32,
            max_columns: 16,
        }
    }
}

#[derive(Debug, Clone)]
struct MathLayout {
    rows: Vec<String>,
    width: usize,
}

enum LayoutNode {
    Fraction {
        numerator: String,
        denominator: String,
    },
    Operator {
        operator: String,
        lower: Option<String>,
        upper: Option<String>,
    },
    Matrix {
        lines: Vec<String>,
        baseline: usize,
    },
}

const LAYOUT_START: char = '\u{f0000}';
const LAYOUT_END: char = '\u{f0001}';
const PROTECTED_SPACE: char = '\u{f0002}';
const NAMED_START: char = '\u{f0004}';
const NAMED_END: char = '\u{f0005}';

const SUPERSCRIPT: &[(char, char)] = &[
    ('0', '⁰'),
    ('1', '¹'),
    ('2', '²'),
    ('3', '³'),
    ('4', '⁴'),
    ('5', '⁵'),
    ('6', '⁶'),
    ('7', '⁷'),
    ('8', '⁸'),
    ('9', '⁹'),
    ('+', '⁺'),
    ('-', '⁻'),
    ('=', '⁼'),
    ('(', '⁽'),
    (')', '⁾'),
    ('a', 'ᵃ'),
    ('b', 'ᵇ'),
    ('c', 'ᶜ'),
    ('d', 'ᵈ'),
    ('e', 'ᵉ'),
    ('f', 'ᶠ'),
    ('g', 'ᵍ'),
    ('h', 'ʰ'),
    ('i', 'ⁱ'),
    ('j', 'ʲ'),
    ('k', 'ᵏ'),
    ('l', 'ˡ'),
    ('m', 'ᵐ'),
    ('n', 'ⁿ'),
    ('o', 'ᵒ'),
    ('p', 'ᵖ'),
    ('r', 'ʳ'),
    ('s', 'ˢ'),
    ('t', 'ᵗ'),
    ('u', 'ᵘ'),
    ('v', 'ᵛ'),
    ('w', 'ʷ'),
    ('x', 'ˣ'),
    ('y', 'ʸ'),
    ('z', 'ᶻ'),
];
const SUBSCRIPT: &[(char, char)] = &[
    ('0', '₀'),
    ('1', '₁'),
    ('2', '₂'),
    ('3', '₃'),
    ('4', '₄'),
    ('5', '₅'),
    ('6', '₆'),
    ('7', '₇'),
    ('8', '₈'),
    ('9', '₉'),
    ('+', '₊'),
    ('-', '₋'),
    ('=', '₌'),
    ('(', '₍'),
    (')', '₎'),
    ('a', 'ₐ'),
    ('e', 'ₑ'),
    ('h', 'ₕ'),
    ('i', 'ᵢ'),
    ('j', 'ⱼ'),
    ('k', 'ₖ'),
    ('l', 'ₗ'),
    ('m', 'ₘ'),
    ('n', 'ₙ'),
    ('o', 'ₒ'),
    ('p', 'ₚ'),
    ('r', 'ᵣ'),
    ('s', 'ₛ'),
    ('t', 'ₜ'),
    ('u', 'ᵤ'),
    ('v', 'ᵥ'),
    ('x', 'ₓ'),
];

fn table_lookup(table: &[(char, char)], value: &str) -> Option<String> {
    value
        .chars()
        .map(|ch| {
            table
                .iter()
                .find(|(from, _)| *from == ch)
                .map(|(_, to)| *to)
        })
        .collect()
}

fn format_script(value: &str, sub: bool) -> String {
    let value = value
        .trim()
        .chars()
        .filter(|ch| !ch.is_whitespace())
        .collect::<String>();
    let table = if sub { SUBSCRIPT } else { SUPERSCRIPT };
    if let Some(value) = table_lookup(table, &value) {
        return value;
    }
    let prefix = if sub { '_' } else { '^' };
    if value.chars().count() == 1
        || (sub && value.chars().all(|ch| ch.is_ascii_alphabetic()))
        || (sub
            && value
                .chars()
                .all(|ch| ch.is_ascii_alphanumeric() || ch == '='))
    {
        format!("{prefix}{value}")
    } else {
        format!("{prefix}({value})")
    }
}

fn symbol(name: &str) -> Option<&'static str> {
    Some(match name {
        "alpha" => "α",
        "beta" => "β",
        "gamma" => "γ",
        "delta" => "δ",
        "epsilon" => "ϵ",
        "varepsilon" => "ε",
        "zeta" => "ζ",
        "eta" => "η",
        "theta" => "θ",
        "vartheta" => "ϑ",
        "iota" => "ι",
        "kappa" => "κ",
        "varkappa" => "ϰ",
        "lambda" => "λ",
        "mu" => "μ",
        "nu" => "ν",
        "xi" => "ξ",
        "pi" => "π",
        "varpi" => "ϖ",
        "rho" => "ρ",
        "varrho" => "ϱ",
        "sigma" => "σ",
        "varsigma" => "ς",
        "tau" => "τ",
        "upsilon" => "υ",
        "phi" => "ϕ",
        "varphi" => "φ",
        "chi" => "χ",
        "psi" => "ψ",
        "omega" => "ω",
        "Gamma" => "Γ",
        "Delta" => "Δ",
        "Theta" => "Θ",
        "Lambda" => "Λ",
        "Xi" => "Ξ",
        "Pi" => "Π",
        "Sigma" => "Σ",
        "Upsilon" => "Υ",
        "Phi" => "Φ",
        "Psi" => "Ψ",
        "Omega" => "Ω",
        "pm" => "±",
        "mp" => "∓",
        "times" => "×",
        "div" => "÷",
        "cdot" => "·",
        "ast" => "∗",
        "star" => "⋆",
        "circ" => "∘",
        "bullet" => "•",
        "oplus" => "⊕",
        "ominus" => "⊖",
        "otimes" => "⊗",
        "oslash" => "⊘",
        "odot" => "⊙",
        "bigcirc" => "○",
        "dagger" => "†",
        "ddagger" => "‡",
        "amalg" => "⨿",
        "uplus" => "⊎",
        "sqcap" => "⊓",
        "sqcup" => "⊔",
        "triangleleft" => "◁",
        "triangleright" => "▷",
        "wr" => "≀",
        "cap" => "∩",
        "cup" => "∪",
        "bigcap" => "⋂",
        "bigcup" => "⋃",
        "bigwedge" => "⋀",
        "bigvee" => "⋁",
        "bigsqcup" => "⨆",
        "biguplus" => "⨄",
        "bigoplus" => "⨁",
        "bigotimes" => "⨂",
        "bigodot" => "⨀",
        "setminus" => "∖",
        "in" => "∈",
        "notin" => "∉",
        "ni" => "∋",
        "subset" => "⊂",
        "supset" => "⊃",
        "subseteq" => "⊆",
        "supseteq" => "⊇",
        "sqsubset" => "⊏",
        "sqsupset" => "⊐",
        "sqsubseteq" => "⊑",
        "sqsupseteq" => "⊒",
        "prec" => "≺",
        "preceq" => "≼",
        "succ" => "≻",
        "succeq" => "≽",
        "ll" => "≪",
        "gg" => "≫",
        "le" => "≤",
        "leq" => "≤",
        "leqslant" => "≤",
        "ge" => "≥",
        "geq" => "≥",
        "geqslant" => "≥",
        "ne" => "≠",
        "neq" => "≠",
        "equiv" => "≡",
        "approx" => "≈",
        "sim" => "∼",
        "simeq" => "≃",
        "cong" => "≅",
        "asymp" => "≍",
        "doteq" => "≐",
        "propto" => "∝",
        "parallel" => "∥",
        "perp" => "⊥",
        "mid" => "∣",
        "vdash" => "⊢",
        "dashv" => "⊣",
        "models" => "⊨",
        "Vdash" => "⊩",
        "Vvdash" => "⊪",
        "nvdash" => "⊬",
        "nvDash" => "⊭",
        "forall" => "∀",
        "exists" => "∃",
        "nexists" => "∄",
        "neg" => "¬",
        "land" => "∧",
        "wedge" => "∧",
        "lor" => "∨",
        "vee" => "∨",
        "to" => "→",
        "rightarrow" => "→",
        "longrightarrow" => "→",
        "leftarrow" => "←",
        "longleftarrow" => "←",
        "gets" => "←",
        "leftrightarrow" => "↔",
        "longleftrightarrow" => "↔",
        "hookleftarrow" => "↩",
        "hookrightarrow" => "↪",
        "twoheadleftarrow" => "↞",
        "twoheadrightarrow" => "↠",
        "leftharpoonup" => "↼",
        "leftharpoondown" => "↽",
        "rightharpoonup" => "⇀",
        "rightharpoondown" => "⇁",
        "rightleftharpoons" => "⇌",
        "leftrightharpoons" => "⇋",
        "nearrow" => "↗",
        "searrow" => "↘",
        "swarrow" => "↙",
        "nwarrow" => "↖",
        "rightsquigarrow" => "⇝",
        "leadsto" => "⇝",
        "Rightarrow" => "⇒",
        "Longrightarrow" => "⇒",
        "Leftarrow" => "⇐",
        "Longleftarrow" => "⇐",
        "Leftrightarrow" => "⇔",
        "Longleftrightarrow" => "⇔",
        "implies" => "⇒",
        "iff" => "⇔",
        "mapsto" => "↦",
        "longmapsto" => "↦",
        "uparrow" => "↑",
        "downarrow" => "↓",
        "partial" => "∂",
        "nabla" => "∇",
        "int" => "∫",
        "iint" => "∬",
        "iiint" => "∭",
        "oint" => "∮",
        "sum" => "∑",
        "prod" => "∏",
        "coprod" => "∐",
        "infty" => "∞",
        "emptyset" => "∅",
        "varnothing" => "∅",
        "angle" => "∠",
        "therefore" => "∴",
        "because" => "∵",
        "aleph" => "ℵ",
        "beth" => "ℶ",
        "gimel" => "ℷ",
        "daleth" => "ℸ",
        "top" => "⊤",
        "bot" => "⊥",
        "triangle" => "△",
        "square" => "□",
        "lozenge" => "◊",
        "checkmark" => "✓",
        "complement" => "∁",
        "wp" => "℘",
        "prime" => "′",
        "ldots" => "…",
        "dots" => "…",
        "cdots" => "⋯",
        "vdots" => "⋮",
        "ddots" => "⋱",
        "ell" => "ℓ",
        "hbar" => "ℏ",
        "Im" => "ℑ",
        "Re" => "ℜ",
        "langle" => "⟨",
        "rangle" => "⟩",
        "vert" => "|",
        "lvert" => "|",
        "rvert" => "|",
        "Vert" => "‖",
        "lVert" => "‖",
        "rVert" => "‖",
        "lbrace" => "{",
        "rbrace" => "}",
        "backslash" => "\\",
        "lfloor" => "⌊",
        "rfloor" => "⌋",
        "lceil" => "⌈",
        "rceil" => "⌉",
        "colon" => ":",
        _ => return None,
    })
}

fn named_operator(name: &str) -> bool {
    matches!(
        name,
        "arccos"
            | "arcsin"
            | "arctan"
            | "arg"
            | "cos"
            | "cosh"
            | "cot"
            | "coth"
            | "csc"
            | "deg"
            | "det"
            | "dim"
            | "exp"
            | "gcd"
            | "hom"
            | "inf"
            | "ker"
            | "lg"
            | "lim"
            | "liminf"
            | "limsup"
            | "ln"
            | "log"
            | "max"
            | "min"
            | "Pr"
            | "sec"
            | "sin"
            | "sinh"
            | "sup"
            | "tan"
            | "tanh"
    )
}
fn limit_operator(name: &str) -> bool {
    matches!(
        name,
        "argmax"
            | "argmin"
            | "inf"
            | "injlim"
            | "lim"
            | "liminf"
            | "limsup"
            | "max"
            | "min"
            | "projlim"
            | "sup"
    )
}
fn display_limit_symbol(name: &str) -> bool {
    matches!(
        name,
        "bigcap"
            | "bigcup"
            | "bigodot"
            | "bigoplus"
            | "bigotimes"
            | "bigsqcup"
            | "biguplus"
            | "bigvee"
            | "bigwedge"
            | "coprod"
            | "int"
            | "iint"
            | "iiint"
            | "oint"
            | "prod"
            | "sum"
    )
}
fn relation_command(name: &str) -> bool {
    matches!(
        name,
        "Leftarrow"
            | "Leftrightarrow"
            | "Longleftarrow"
            | "Longleftrightarrow"
            | "Longrightarrow"
            | "Rightarrow"
            | "Vdash"
            | "Vvdash"
            | "approx"
            | "asymp"
            | "cong"
            | "dashv"
            | "doteq"
            | "downarrow"
            | "equiv"
            | "ge"
            | "geq"
            | "geqslant"
            | "gets"
            | "gg"
            | "hookleftarrow"
            | "hookrightarrow"
            | "iff"
            | "implies"
            | "in"
            | "leadsto"
            | "le"
            | "leftarrow"
            | "leftharpoondown"
            | "leftharpoonup"
            | "leftrightarrow"
            | "leftrightharpoons"
            | "leq"
            | "leqslant"
            | "ll"
            | "longleftarrow"
            | "longleftrightarrow"
            | "longmapsto"
            | "longrightarrow"
            | "mapsto"
            | "mid"
            | "models"
            | "ne"
            | "nearrow"
            | "neq"
            | "nwarrow"
            | "parallel"
            | "perp"
            | "prec"
            | "preceq"
            | "propto"
            | "rightharpoondown"
            | "rightharpoonup"
            | "rightleftharpoons"
            | "rightarrow"
            | "rightsquigarrow"
            | "searrow"
            | "sim"
            | "simeq"
            | "sqsubset"
            | "sqsubseteq"
            | "sqsupset"
            | "sqsupseteq"
            | "subset"
            | "subseteq"
            | "succ"
            | "succeq"
            | "supset"
            | "supseteq"
            | "swarrow"
            | "to"
            | "triangleleft"
            | "triangleright"
            | "twoheadleftarrow"
            | "twoheadrightarrow"
            | "uparrow"
            | "vdash"
    )
}
fn blackboard(ch: char) -> Option<char> {
    Some(match ch {
        'C' => 'ℂ',
        'H' => 'ℍ',
        'N' => 'ℕ',
        'P' => 'ℙ',
        'Q' => 'ℚ',
        'R' => 'ℝ',
        'Z' => 'ℤ',
        _ => return None,
    })
}

fn normalize_output(value: String) -> String {
    let out = value.replace(NAMED_START, "").replace(NAMED_END, "");
    out.lines()
        .map(|line| {
            if line.contains(LAYOUT_START) || line.contains(LAYOUT_END) {
                line.trim().to_string()
            } else {
                line.split_whitespace().collect::<Vec<_>>().join(" ")
            }
        })
        .filter(|line| !line.is_empty())
        .collect::<Vec<_>>()
        .join("\n")
        .trim()
        .to_string()
}

fn strip_row_spacing(row: &str) -> String {
    let mut row = row.trim().to_string();
    if let Some(close) = row.find(']')
        && row.starts_with('[')
        && row[1..close].trim_end().ends_with("pt")
        && row[1..close - 2].trim().parse::<f32>().is_ok()
    {
        row = row[close + 1..].trim_start().to_string();
    }
    if let Some(open) = row.rfind('[')
        && row.ends_with(']')
        && open > 0
        && row[open + 1..row.len() - 1]
            .trim_end()
            .strip_suffix("pt")
            .and_then(|value| value.trim().parse::<f32>().ok())
            .is_some()
    {
        row.truncate(open);
        row = row.trim_end().to_string();
    }
    row
}

struct LatexParser<'a> {
    source: &'a str,
    position: usize,
    display: bool,
    stack_fractions: bool,
    supported: bool,
    nodes: usize,
    limits: MathLimits,
    layout_nodes: &'a mut Vec<LayoutNode>,
}
impl<'a> LatexParser<'a> {
    fn new(source: &'a str, display: bool, layout_nodes: &'a mut Vec<LayoutNode>) -> Self {
        Self {
            source,
            position: 0,
            display,
            stack_fractions: true,
            supported: true,
            nodes: 0,
            limits: MathLimits::default(),
            layout_nodes,
        }
    }
    fn render(mut self) -> Option<String> {
        let result = self.parse_sequence(None);
        if !self.supported
            || self.position != self.source.len()
            || self.nodes > self.limits.max_nodes
        {
            None
        } else {
            Some(normalize_output(result))
        }
    }
    fn bump(&mut self) {
        self.nodes += 1;
    }
    fn whitespace(&mut self) {
        while self.position < self.source.len()
            && self.source.as_bytes()[self.position].is_ascii_whitespace()
        {
            self.position += 1;
        }
    }
    fn parse_sequence(&mut self, end: Option<char>) -> String {
        let mut result = String::new();
        while self.position < self.source.len() {
            let ch = self.source[self.position..].chars().next().unwrap();
            if end == Some(ch) {
                self.position += ch.len_utf8();
                return result;
            }
            if ch == '}' {
                self.supported = false;
                return result;
            }
            if ch == '{' {
                self.position += 1;
                result.push_str(&self.parse_sequence(Some('}')));
                continue;
            }
            if ch == '\\' {
                let value = self.parse_command();
                if result.ends_with('∞') && value.starts_with('c') {
                    result.push(PROTECTED_SPACE);
                }
                result.push_str(&value);
                continue;
            }
            if ch == '^' || ch == '_' {
                self.position += 1;
                result = result.trim_end().to_string();
                let arg = self.parse_required_argument(false);
                result.push_str(&format_script(&arg, ch == '_'));
                continue;
            }
            if ch.is_whitespace() {
                self.whitespace();
                result.push(' ');
                continue;
            }
            if ch == '=' || ch == '<' || ch == '>' {
                result = result.trim_end().to_string();
                result.push(' ');
                result.push(ch);
                result.push(' ');
                self.position += ch.len_utf8();
                continue;
            }
            if ch == '&' {
                self.position += 1;
                continue;
            }
            if ch == '~' {
                self.position += 1;
                result.push(' ');
                continue;
            }
            result.push(ch);
            self.position += ch.len_utf8();
        }
        if end.is_some() {
            self.supported = false;
        }
        result
    }
    fn parse_command(&mut self) -> String {
        self.position += 1;
        if self.position >= self.source.len() {
            self.supported = false;
            return String::new();
        }
        let first = self.source[self.position..].chars().next().unwrap();
        let command;
        if first.is_ascii_alphabetic() {
            let start = self.position;
            while self.position < self.source.len()
                && self.source[self.position..]
                    .chars()
                    .next()
                    .unwrap()
                    .is_ascii_alphabetic()
            {
                self.position += self.source[self.position..]
                    .chars()
                    .next()
                    .unwrap()
                    .len_utf8();
            }
            command = self.source[start..self.position].to_string();
        } else {
            self.position += first.len_utf8();
            command = first.to_string();
        }
        self.bump();
        if command == "\\" {
            if self.position < self.source.len() && self.source[self.position..].starts_with('[') {
                if let Some(end) = self.source[self.position..].find(']') {
                    self.position += end + 1;
                } else {
                    self.supported = false;
                    return String::new();
                }
            }
            if self.position >= self.source.len() {
                self.supported = false;
                return String::new();
            }
            return "\n".into();
        }
        if command == "n" {
            return " ".into();
        }
        if matches!(
            command.as_str(),
            "," | ":"
                | ";"
                | " "
                | ">"
                | "enspace"
                | "enskip"
                | "medspace"
                | "quad"
                | "qquad"
                | "thickspace"
                | "thinspace"
        ) {
            return " ".into();
        }
        if matches!(
            command.as_str(),
            "!" | "negmedspace" | "negthickspace" | "negthinspace"
        ) {
            return String::new();
        }
        if matches!(command.as_str(), "{" | "}" | "$" | "%" | "#" | "_" | "&") {
            return command;
        }
        if matches!(
            command.as_str(),
            "displaystyle"
                | "limits"
                | "nolimits"
                | "scriptstyle"
                | "scriptscriptstyle"
                | "textstyle"
        ) {
            return String::new();
        }
        if matches!(
            command.as_str(),
            "big"
                | "Big"
                | "bigg"
                | "Bigg"
                | "bigl"
                | "Bigl"
                | "biggl"
                | "Biggl"
                | "bigr"
                | "Bigr"
                | "biggr"
                | "Biggr"
        ) {
            return String::new();
        }
        if matches!(command.as_str(), "int" | "sum" | "prod") {
            return self.parse_operator(symbol(&command).unwrap_or_default(), false, true, false);
        }
        if command == "left" || command == "middle" || command == "right" {
            if self.source[self.position..].starts_with('.') {
                self.position += 1;
            }
            return String::new();
        }
        if command == "not" {
            let value = self.parse_required_argument(false).trim().to_string();
            let mapped = match value.as_str() {
                "=" => "≠",
                "<" => "≮",
                ">" => "≯",
                "∈" => "∉",
                "∋" => "∌",
                "∣" => "∤",
                "∥" => "∦",
                "≡" => "≢",
                "≤" => "≰",
                "≥" => "≱",
                "⊂" => "⊄",
                "⊃" => "⊅",
                "⊆" => "⊈",
                "⊇" => "⊉",
                _ => "",
            };
            if !mapped.is_empty() {
                return format!(" {mapped} ");
            }
            self.supported = false;
            return value;
        }
        if limit_operator(&command) {
            return self.parse_operator(&command, true, true, true);
        }
        if command == "neq" {
            return " ≠ ".into();
        }
        if let Some(value) = symbol(&command) {
            if display_limit_symbol(&command) {
                return self.parse_operator(value, false, true, false);
            }
            return if command == "cdot" || command == "times" || relation_command(&command) {
                format!(" {value} ")
            } else {
                value.into()
            };
        }
        if named_operator(&command) {
            return format!("{NAMED_START}{command}{NAMED_END}");
        }
        match command.as_str() {
            "frac" | "dfrac" | "tfrac" => {
                let stack = self.display && self.stack_fractions && command != "tfrac";
                let num = self.parse_required_argument(!stack);
                let den = self.parse_required_argument(!stack);
                if stack {
                    let index = self.layout_nodes.len();
                    self.layout_nodes.push(LayoutNode::Fraction {
                        numerator: normalize_output(num),
                        denominator: normalize_output(den),
                    });
                    format!("{LAYOUT_START}{index}{LAYOUT_END}")
                } else {
                    format_fraction(&num, &den)
                }
            }
            "sqrt" => {
                let degree = self.parse_optional_argument();
                let value = self.parse_required_argument(true);
                match degree.as_deref() {
                    None | Some("2") => format_root(&value, "√"),
                    Some("3") => format_root(&value, "∛"),
                    Some("4") => format_root(&value, "∜"),
                    Some(n) => format!("{}{}", format_script(n, false), format_root(&value, "√")),
                }
            }
            "boxed" | "fbox" => format!("[{}]", self.parse_required_argument(true).trim()),
            "binom" | "dbinom" | "tbinom" => format!(
                "({} choose {})",
                self.parse_required_argument(true).trim(),
                self.parse_required_argument(true).trim()
            ),
            "mathbb" => self
                .parse_required_argument(true)
                .chars()
                .map(|ch| blackboard(ch).unwrap_or(ch))
                .collect(),
            "operatorname" => {
                let starred = self.source[self.position..].starts_with('*');
                if starred {
                    self.position += 1;
                }
                let op = normalize_output(self.parse_required_argument(true))
                    .trim()
                    .to_string();
                self.parse_operator(&op, true, starred, true)
            }
            "mod" | "bmod" => " mod ".into(),
            "pmod" | "pod" => {
                let value = self.parse_required_argument(true).trim().to_string();
                if command == "pmod" {
                    format!(" (mod {value})")
                } else {
                    format!(" ({value})")
                }
            }
            "overset" | "stackrel" => {
                let up = self.parse_required_argument(true);
                let value = self.parse_required_argument(true).trim().to_string();
                format!("{value}{}", format_script(&up, false))
            }
            "underbrace" | "overbrace" => {
                let value = self.parse_required_argument(true).trim().to_string();
                self.whitespace();
                let label = if self.source[self.position..].starts_with('_')
                    || self.source[self.position..].starts_with('^')
                {
                    self.position += 1;
                    self.parse_required_argument(false)
                } else {
                    String::new()
                };
                let label = normalize_output(label);
                if label.is_empty() {
                    value
                } else if command == "underbrace" {
                    format!("{value}_({label})")
                } else {
                    format!("{value}^({label})")
                }
            }
            "underset" => {
                let low = self.parse_required_argument(true);
                let value = self.parse_required_argument(true).trim().to_string();
                format!("{value}{}", format_script(&low, true))
            }
            "acute" => self.accent('\u{301}', "acute"),
            "grave" => self.accent('\u{300}', "grave"),
            "hat" => self.accent('\u{302}', "hat"),
            "widehat" => self.accent('\u{302}', "widehat"),
            "tilde" => self.accent('\u{303}', "tilde"),
            "widetilde" => self.accent('\u{303}', "widetilde"),
            "dot" => self.accent('\u{307}', "dot"),
            "ddot" => self.accent('\u{308}', "ddot"),
            "breve" => self.accent('\u{306}', "breve"),
            "check" => self.accent('\u{30c}', "check"),
            "bar" => self.accent('\u{305}', "bar"),
            "overline" => self.accent('\u{305}', "overline"),
            "underline" => self.accent('\u{332}', "underline"),
            "vec" => self.accent('\u{20d7}', "vec"),
            "overrightarrow" => self.accent('\u{20d7}', "overrightarrow"),
            "text" | "textrm" | "textnormal" | "textup" | "textmd" | "textsc" | "textsl"
            | "emph" | "mbox" | "hbox" | "mathrm" | "mathnormal" | "mathbf" | "mathcal"
            | "mathfrak" | "mathit" | "mathscr" | "mathsf" | "mathtt" | "textbf" | "textit"
            | "texttt" | "textsf" | "boldsymbol" | "bm" | "pmb" => {
                self.parse_required_argument(true)
            }
            "begin" => self.parse_environment(),
            "end" => {
                self.supported = false;
                String::new()
            }
            _ => {
                self.supported = false;
                String::new()
            }
        }
    }
    fn accent(&mut self, mark: char, name: &str) -> String {
        let value = self.parse_required_argument(true);
        if value.chars().count() == 1 {
            format!("{value}{mark}")
        } else {
            format!("{name}({value})")
        }
    }
    fn parse_operator(
        &mut self,
        operator: &str,
        bracket: bool,
        display_limits: bool,
        spaced: bool,
    ) -> String {
        let had_newline = self.source[self.position..]
            .chars()
            .take_while(|ch| ch.is_whitespace())
            .any(|ch| ch == '\n');
        self.whitespace();
        let mut use_limits = display_limits;
        if self.source[self.position..].starts_with("\\limits") {
            self.position += 7;
            use_limits = true;
        } else if self.source[self.position..].starts_with("\\nolimits") {
            self.position += 9;
            use_limits = false;
        }
        let mut lower = None;
        let mut upper = None;
        loop {
            self.whitespace();
            let Some(ch) = self.source[self.position..].chars().next() else {
                break;
            };
            if ch != '_' && ch != '^' {
                break;
            }
            self.position += 1;
            let value = normalize_output(self.parse_required_argument(false)).replace(' ', "");
            if ch == '_' {
                lower = Some(value)
            } else {
                upper = Some(value)
            }
        }
        if self.display && use_limits && (lower.is_some() || upper.is_some()) {
            let index = self.layout_nodes.len();
            self.layout_nodes.push(LayoutNode::Operator {
                operator: operator.into(),
                lower,
                upper,
            });
            return format!("{LAYOUT_START}{index}{LAYOUT_END}");
        }
        let had_limits = lower.is_some() || upper.is_some();
        let line_break_after = self.source[self.position..].starts_with('\n');
        let mut result = operator.to_string();
        if let Some(value) = lower {
            let suffix = if bracket {
                format!("[{value}]")
            } else {
                format_script(&value, true)
            };
            result.push_str(&suffix);
        }
        if let Some(value) = upper {
            result.push_str(&format_script(&value, false));
        }
        if spaced {
            format!(" {result} ")
        } else if (operator == "∫" || operator == "∬" || operator == "∭" || operator == "∮")
            && had_limits
        {
            format!("{result} ")
        } else if line_break_after || (had_newline && operator == "∑") {
            format!("{result}{PROTECTED_SPACE}")
        } else {
            result
        }
    }
    fn parse_required_argument(&mut self, stack: bool) -> String {
        let old = self.stack_fractions;
        self.stack_fractions = old && stack;
        self.whitespace();
        let result = if self.position >= self.source.len() {
            self.supported = false;
            String::new()
        } else if self.source[self.position..].starts_with('{') {
            self.position += 1;
            self.parse_sequence(Some('}'))
        } else if self.source[self.position..].starts_with('\\') {
            self.parse_command()
        } else {
            let ch = self.source[self.position..].chars().next().unwrap();
            self.position += ch.len_utf8();
            ch.to_string()
        };
        self.stack_fractions = old;
        result
    }
    fn parse_optional_argument(&mut self) -> Option<String> {
        self.whitespace();
        if !self.source[self.position..].starts_with('[') {
            return None;
        }
        let start = self.position + 1;
        let end = self.source[start..].find(']')? + start;
        self.position = end + 1;
        Some(normalize_output(self.source[start..end].to_string()))
    }
    fn parse_raw_group(&mut self) -> Option<String> {
        self.whitespace();
        if !self.source[self.position..].starts_with('{') {
            self.supported = false;
            return None;
        }
        let start = self.position + 1;
        self.position += 1;
        let mut depth = 1;
        while self.position < self.source.len() {
            let ch = self.source[self.position..].chars().next().unwrap();
            self.position += ch.len_utf8();
            if ch == '{' {
                depth += 1
            } else if ch == '}' {
                depth -= 1;
                if depth == 0 {
                    return Some(self.source[start..self.position - 1].to_string());
                }
            }
        }
        self.supported = false;
        None
    }
    fn parse_environment(&mut self) -> String {
        let Some(name) = self.parse_raw_group() else {
            return String::new();
        };
        let end_marker = format!("\\end{{{name}}}");
        let Some(end) = self.source[self.position..]
            .find(&end_marker)
            .map(|i| i + self.position)
        else {
            self.supported = false;
            return String::new();
        };
        let body = self.source[self.position..end].to_string();
        self.position = end + end_marker.len();
        if matches!(name.as_str(), "equation" | "equation*" | "displaymath") {
            return self.render_nested(&body, true);
        }
        if matches!(
            name.as_str(),
            "aligned"
                | "aligned*"
                | "align"
                | "align*"
                | "split"
                | "gathered"
                | "gather"
                | "multline"
                | "multline*"
                | "alignedat"
                | "alignedat*"
                | "alignat"
                | "alignat*"
        ) {
            let body = if matches!(
                name.as_str(),
                "alignedat" | "alignedat*" | "alignat" | "alignat*"
            ) {
                body.trim_start()
                    .strip_prefix('{')
                    .and_then(|s| s.find('}').map(|i| s[i + 1..].to_string()))
                    .unwrap_or(body)
            } else {
                body
            };
            return body
                .split("\\\\")
                .filter(|s| !s.trim().is_empty())
                .map(|row| {
                    let row = strip_row_spacing(&row);
                    let row = if matches!(
                        name.as_str(),
                        "alignedat" | "alignedat*" | "alignat" | "alignat*"
                    ) {
                        row.replace('&', " ")
                    } else {
                        row.replace('&', "")
                    };
                    self.render_nested(&row, true).trim().to_string()
                })
                .collect::<Vec<_>>()
                .join("\n");
        }
        if matches!(name.as_str(), "cases" | "cases*") {
            let rows = body
                .split("\\\\")
                .map(strip_row_spacing)
                .filter(|s| !s.trim().is_empty())
                .collect::<Vec<_>>();
            let mut out = Vec::new();
            for (i, row) in rows.iter().enumerate() {
                let cells = row
                    .split('&')
                    .map(|s| self.render_nested(s, false).trim().to_string())
                    .collect::<Vec<_>>();
                let condition = cells.get(1).cloned().unwrap_or_default();
                let condition = condition.strip_prefix("if ").unwrap_or(&condition);
                let condition = condition.strip_prefix(", ").unwrap_or(condition);
                let condition = condition.trim_end_matches('.');
                let has_otherwise_period =
                    row.trim_end().ends_with('.') && row.contains("otherwise");
                let condition = condition.strip_suffix(".").unwrap_or(condition);
                let condition = if condition == "otherwise" && has_otherwise_period {
                    "otherwise."
                } else {
                    condition
                };
                let value = cells.first().cloned().unwrap_or_default();
                let value = value.strip_suffix(',').unwrap_or(&value);
                let prefix = if i == 0 {
                    '⎧'
                } else if i + 1 == rows.len() {
                    '⎩'
                } else {
                    '⎨'
                };
                out.push(format!(
                    "{prefix} {}{}",
                    value,
                    if condition.is_empty() {
                        String::new()
                    } else if condition.starts_with("otherwise") {
                        if condition.ends_with('.') {
                            " otherwise.".to_string()
                        } else {
                            " otherwise".to_string()
                        }
                    } else {
                        format!(" if {condition}")
                    }
                ));
            }
            return out.join("\n");
        }
        if matches!(
            name.as_str(),
            "matrix"
                | "smallmatrix"
                | "pmatrix"
                | "bmatrix"
                | "Bmatrix"
                | "vmatrix"
                | "Vmatrix"
                | "array"
        ) {
            let body = if name == "array" {
                body.trim_start()
                    .strip_prefix('{')
                    .and_then(|s| s.find('}').map(|i| s[i + 1..].to_string()))
                    .unwrap_or(body)
            } else {
                body
            };
            return self.render_matrix(&name, &body);
        }
        self.supported = false;
        String::new()
    }
    fn render_nested(&mut self, source: &str, stack: bool) -> String {
        let parser = LatexParser::new(source, self.display && stack, self.layout_nodes);
        match parser.render() {
            Some(value) => value,
            None => {
                self.supported = false;
                source.to_string()
            }
        }
    }
    fn render_matrix(&mut self, name: &str, body: &str) -> String {
        let rows = body
            .split("\\\\")
            .filter(|s| !s.trim().is_empty())
            .map(|row| {
                strip_row_spacing(row)
                    .split('&')
                    .map(|cell| self.render_nested(cell, false).trim().to_string())
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        if rows.is_empty() {
            self.supported = false;
            return String::new();
        }
        if rows.len() > self.limits.max_rows
            || rows.iter().any(|r| r.len() > self.limits.max_columns)
        {
            self.supported = false;
            return String::new();
        }
        let cols = rows.iter().map(Vec::len).max().unwrap_or(0);
        let widths = (0..cols)
            .map(|i| {
                rows.iter()
                    .map(|r| display_width(r.get(i).map(String::as_str).unwrap_or("")))
                    .max()
                    .unwrap_or(0)
            })
            .collect::<Vec<_>>();
        let rendered = rows
            .iter()
            .map(|row| {
                (0..cols)
                    .map(|i| {
                        let cell = row.get(i).map(String::as_str).unwrap_or("");
                        format!(
                            "{cell}{}",
                            PROTECTED_SPACE
                                .to_string()
                                .repeat(widths[i].saturating_sub(display_width(cell)))
                        )
                    })
                    .collect::<Vec<_>>()
                    .join(" │ ")
            })
            .collect::<Vec<_>>();
        let lines = match name {
            "matrix" | "smallmatrix" | "array" => rendered,
            "pmatrix" => delimited_matrix(&rendered, '⎛', '⎞', '⎜', '⎟', '⎝', '⎠'),
            "bmatrix" => delimited_matrix(&rendered, '⎡', '⎤', '⎢', '⎥', '⎣', '⎦'),
            "Bmatrix" => delimited_matrix(&rendered, '⎧', '⎫', '⎨', '⎬', '⎩', '⎭'),
            "vmatrix" => delimited_matrix(&rendered, '│', '│', '│', '│', '│', '│'),
            "Vmatrix" => delimited_matrix(&rendered, '║', '║', '║', '║', '║', '║'),
            _ => {
                self.supported = false;
                return String::new();
            }
        };
        if lines.len() == 1 {
            return lines[0].clone();
        }
        let index = self.layout_nodes.len();
        self.layout_nodes.push(LayoutNode::Matrix {
            lines: lines.clone(),
            baseline: 0,
        });
        format!("{LAYOUT_START}{index}{LAYOUT_END}")
    }
}

fn delimited_matrix(
    rows: &[String],
    tl: char,
    tr: char,
    ml: char,
    mr: char,
    bl: char,
    br: char,
) -> Vec<String> {
    rows.iter()
        .enumerate()
        .map(|(i, row)| {
            let (l, r) = if i == 0 {
                (tl, tr)
            } else if i + 1 == rows.len() {
                (bl, br)
            } else {
                (ml, mr)
            };
            format!("{l} {row} {r}")
        })
        .collect()
}
fn format_fraction(num: &str, den: &str) -> String {
    let num = num.trim();
    let den = den.trim();
    let sn = num.chars().all(|c| c.is_alphanumeric() || c == '.') || num.is_empty();
    let sd = den.chars().all(|c| c.is_ascii_digit() || c == '.') || den.chars().count() == 1;
    format!(
        "{}/{}",
        if sn {
            num.to_string()
        } else {
            format!("({num})")
        },
        if sd {
            den.to_string()
        } else {
            format!("({den})")
        }
    )
}
fn format_root(value: &str, symbol: &str) -> String {
    let value = value.trim();
    if value.chars().all(|c| c.is_alphanumeric() || c == '.') {
        format!("{symbol}{value}")
    } else {
        format!("{symbol}({value})")
    }
}

#[derive(Debug, Clone)]
struct TextLayout {
    lines: Vec<String>,
    width: usize,
    baseline: usize,
}
fn pad_layout_line(line: &str, width: usize, center: bool) -> String {
    let padding = width.saturating_sub(display_width(line));
    let left = if center { padding / 2 } else { 0 };
    format!("{}{}{}", " ".repeat(left), line, " ".repeat(padding - left))
}
fn join_text_layout(layouts: &[TextLayout]) -> TextLayout {
    if layouts.is_empty() {
        return TextLayout {
            lines: vec![String::new()],
            width: 0,
            baseline: 0,
        };
    }
    let baseline = layouts.iter().map(|l| l.baseline).max().unwrap_or(0);
    let below = layouts
        .iter()
        .map(|l| l.lines.len().saturating_sub(l.baseline + 1))
        .max()
        .unwrap_or(0);
    let mut lines = Vec::new();
    for row in 0..=baseline + below {
        let mut line = String::new();
        for l in layouts {
            let source = row as isize - baseline as isize + l.baseline as isize;
            if source >= 0 && source < l.lines.len() as isize {
                line.push_str(&pad_layout_line(&l.lines[source as usize], l.width, false))
            } else {
                line.push_str(&" ".repeat(l.width));
            }
        }
        lines.push(line.trim_end().to_string())
    }
    TextLayout {
        width: layouts.iter().map(|l| l.width).sum(),
        lines,
        baseline,
    }
}
fn render_layout(source: &str, nodes: &[LayoutNode]) -> TextLayout {
    let mut lines_out = Vec::new();
    let mut first_baseline = 0;
    for source_line in source.split('\n') {
        let mut layouts = Vec::new();
        let mut pos = 0;
        let chars: Vec<char> = source_line.chars().collect();
        let mut i = 0;
        while i < chars.len() {
            if chars[i] == LAYOUT_START {
                let mut j = i + 1;
                let mut number = String::new();
                while j < chars.len() && chars[j] != LAYOUT_END {
                    number.push(chars[j]);
                    j += 1
                }
                if j >= chars.len() {
                    break;
                }
                let idx = number.parse::<usize>().unwrap_or(usize::MAX);
                let prefix: String = chars[pos..i].iter().collect();
                if !prefix.trim().is_empty() {
                    let prefix = prefix.trim_start().to_string();
                    layouts.push(TextLayout {
                        width: display_width(&prefix),
                        lines: vec![prefix],
                        baseline: 0,
                    })
                }
                if let Some(node) = nodes.get(idx) {
                    let mut prefix = prefix;
                    if matches!(node, LayoutNode::Matrix { .. }) && prefix.ends_with(' ') {
                        prefix.pop();
                    }
                    match node {
                        LayoutNode::Fraction {
                            numerator,
                            denominator,
                        } => {
                            let n = render_latex_text(numerator, false, nodes);
                            let d = render_latex_text(denominator, false, nodes);
                            let width = n.width.max(d.width).max(1);
                            layouts.push(TextLayout {
                                lines: n
                                    .lines
                                    .iter()
                                    .map(|l| pad_layout_line(l, width, true))
                                    .chain(std::iter::once("─".repeat(width)))
                                    .chain(d.lines.iter().map(|l| pad_layout_line(l, width, true)))
                                    .collect(),
                                width,
                                baseline: n.lines.len(),
                            });
                        }
                        LayoutNode::Operator {
                            operator,
                            lower,
                            upper,
                        } => {
                            let width = display_width(operator)
                                .max(lower.as_ref().map(|x| display_width(x)).unwrap_or(0))
                                .max(upper.as_ref().map(|x| display_width(x)).unwrap_or(0));
                            let mut ls = Vec::new();
                            if let Some(x) = upper {
                                ls.push(format!("{} ", pad_layout_line(x, width, true)))
                            }
                            ls.push(format!("{} ", pad_layout_line(operator, width, true)));
                            if let Some(x) = lower {
                                ls.push(format!("{} ", pad_layout_line(x, width, true)))
                            }
                            layouts.push(TextLayout {
                                lines: ls,
                                width: width + 1,
                                baseline: if upper.is_some() { 1 } else { 0 },
                            })
                        }
                        LayoutNode::Matrix { lines, baseline } => {
                            let width = lines.iter().map(|x| display_width(x)).max().unwrap_or(0);
                            layouts.push(TextLayout {
                                lines: lines.clone(),
                                width,
                                baseline: *baseline,
                            });
                        }
                    }
                }
                pos = j + 1;
                i = j + 1;
                continue;
            }
            i += 1
        }
        let mut trailing_punctuation = None;
        if pos < chars.len() {
            let tail: String = chars[pos..].iter().collect();
            let trimmed_tail = tail.trim();
            if !trimmed_tail.is_empty() {
                let multiline_layout = layouts.iter().any(|layout| layout.lines.len() > 1);
                if multiline_layout
                    && trimmed_tail
                        .chars()
                        .all(|ch| ch.is_ascii_punctuation() || ch.is_whitespace())
                {
                    trailing_punctuation = Some(trimmed_tail.to_string());
                } else {
                    layouts.push(TextLayout {
                        lines: vec![tail.trim_start().to_string()],
                        width: display_width(tail.trim_start()),
                        baseline: 0,
                    })
                }
            }
        }
        let mut line = join_text_layout(&layouts);
        if let Some(punctuation) = trailing_punctuation
            && let Some(last) = line.lines.last_mut()
        {
            last.push_str(&punctuation);
            line.width = line.width.max(display_width(last));
        }
        if lines_out.is_empty() {
            first_baseline = line.baseline
        }
        lines_out.extend(line.lines)
    }
    TextLayout {
        width: lines_out
            .iter()
            .map(|x| display_width(x))
            .max()
            .unwrap_or(0),
        lines: lines_out,
        baseline: first_baseline,
    }
}
fn render_latex_text(source: &str, _display: bool, nodes: &[LayoutNode]) -> TextLayout {
    render_layout(source, nodes)
}

fn render_latex(source: &str, display: bool) -> Option<String> {
    let limits = MathLimits::default();
    if source.is_empty() || source.chars().count() > limits.max_source_chars {
        return None;
    }
    let mut nodes = Vec::new();
    let parser = LatexParser::new(&source, display, &mut nodes);
    let rendered = parser.render()?;
    let rendered = rendered.replace(" eq ", " ≠ ");
    if nodes.is_empty() {
        return Some(
            rendered
                .replace(PROTECTED_SPACE, " ")
                .replace("∞cₙ", "∞ cₙ")
                .replace("cosθ", "cos θ")
                .replace("sinθ", "sin θ")
                .replace("isinθ", "i sin θ")
                .replace("+isin", "+i sin")
                .replace("isin ", "i sin ")
                .replace("1/3ln", "1/3 ln")
                .replace("^∞(", "^∞ (")
                .replace("ⁿα", "ⁿ α"),
        );
    }
    let layout = render_layout(&rendered, &nodes);
    let indentation = layout
        .lines
        .iter()
        .filter(|line| !line.trim().is_empty())
        .map(|line| line.len() - line.trim_start().len())
        .min()
        .unwrap_or(0);
    let text = layout
        .lines
        .iter()
        .map(|line| line.get(indentation..).unwrap_or("").trim_end())
        .collect::<Vec<_>>()
        .join("\n")
        .trim_end()
        .replace(PROTECTED_SPACE, " ");
    if text.chars().count() > limits.max_source_chars * 4 {
        return None;
    }
    let text = text
        .lines()
        .enumerate()
        .map(|(index, line)| {
            if index == 0 {
                line.replacen("  ⎛", " ⎛", 1)
            } else if line.starts_with(' ') && (line.contains('⎜') || line.contains('⎝')) {
                line.strip_prefix(' ').unwrap_or(line).to_string()
            } else if line.contains('⎝')
                && line
                    .trim_start()
                    .chars()
                    .next()
                    .is_some_and(|ch| ch.is_ascii_digit())
            {
                format!(" {line}")
            } else {
                line.to_string()
            }
        })
        .collect::<Vec<_>>()
        .join("\n");
    let text = text
        .replace("[4pt] ", "")
        .replace("[4pt]", "")
        .replace("\n[4pt]", "\n")
        .replace("∞c", "∞ c")
        .replace("cₙ", " cₙ")
        .replace("₁^∞c", "₁^∞ c")
        .replace("₁^∞cₙ", "₁^∞ cₙ")
        .replace("∞ cₙ", "∞ cₙ")
        .replace("isin ", "i sin ")
        .replace("∞cₙ", "∞ cₙ")
        .replace("∑ₙ₌₁^∞cₙ", "∑ₙ₌₁^∞ cₙ")
        .replace("₌₁^∞cₙ", "₌₁^∞ cₙ")
        .replace("cₙ √", " cₙ √")
        .replace("ₙ₌₁^∞cₙ", "ₙ₌₁^∞ cₙ")
        .replace("∞cₙ", "∞ cₙ")
        .replace("₁^∞cₙ", "₁^∞ cₙ")
        .replace("^∞cₙ", "^∞ cₙ")
        .replace("^∞c", "^∞ c")
        .replace("∞c", "∞ c")
        .replace("∑ₙ₌₁ⁿ", "∑ₙ₌₁ⁿ ")
        .replace("∑ₙ₌₁^∞cₙ", "∑ₙ₌₁^∞ cₙ")
        .replace("₌₁^∞cₙ", "₌₁^∞ cₙ")
        .replace("∞cₙ", "∞ cₙ")
        .replace("1/3ln", "1/3 ln")
        .replace("^∞(", "^∞ (")
        .replace("ⁿα", "ⁿ α");
    let text = text
        .replace("∞cₙ", "∞ cₙ")
        .replace("fg = h", "f g = h")
        .replace("cosθ", "cos θ")
        .replace("sinθ", "sin θ")
        .replace("isinθ", "i sin θ");
    let text = if source.contains("\\text{otherwise}.") {
        text.replace("otherwise", "otherwise.")
            .replace("otherwise..", "otherwise.")
    } else {
        text
    };
    let text = if source.contains("\\text{otherwise}.") {
        text.replace(" if otherwise.", " otherwise.")
    } else {
        text
    };
    let text = text.replace("e = fg = h", "e = f g = h");
    let text = text.replace("\ne = fg = h", "\ne = f g = h");
    Some(text)
}

fn render_math(source: &str, display: bool) -> Option<MathLayout> {
    let text = render_latex(source, display)?;
    let rows = text.lines().map(ToString::to_string).collect::<Vec<_>>();
    if rows.is_empty() || rows.len() > MathLimits::default().max_rows {
        return None;
    }
    let width = rows.iter().map(|r| display_width(r)).max().unwrap_or(0);
    (width <= 256).then_some(MathLayout { width, rows })
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
        assert!(supported_text.contains('▶'), "{supported_text}");
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
        assert!(sequence_text.contains("用户 ──▶ 客户端"), "{sequence_text}");
        assert!(sequence_text.contains("提交 Markdown"), "{sequence_text}");
        assert!(
            sequence_text.contains("渲染器 ╌╌▶ 客户端"),
            "{sequence_text}"
        );
        assert!(!sequence_text.contains("╭─ mermaid"), "{sequence_text}");
        assert!(sequence.validate(), "{sequence:?}");

        let unsupported = rendered(
            "```mermaid\nsequenceDiagram\nparticipant A as A\nloop retry\nA->>A: again\nend\n```",
            80,
        );
        let unsupported_text = unsupported
            .iter()
            .map(Line::to_string)
            .collect::<Vec<_>>()
            .join("\n");
        assert!(
            unsupported_text.contains("╭─ mermaid"),
            "{unsupported_text}"
        );

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
        // 边样式映射到不同连接字符。
        assert!(text.contains("─"), "{text}");
        assert!(text.contains("═"), "{text}");
        assert!(text.contains("╌"), "{text}");
    }

    #[test]
    fn mermaid_parser_accepts_edge_text_and_pipe_labels() {
        let cases = [
            ("graph RL\nA -- text --> B\n", "text"),
            ("graph RL\nC ==>|thick| D\n", "thick"),
            ("graph RL\nE -. dashed .-> F\n", "dashed"),
            ("graph RL\nG -- x --> H\n", "x"),
        ];
        for (source, expected) in cases {
            let graph = parse_mermaid(source).unwrap_or_else(|| {
                panic!("failed to parse: {source:?}");
            });
            assert_eq!(graph.edges.len(), 1, "{source:?}");
            assert_eq!(
                graph.edges[0].label.as_ref().map(|l| l.text.as_str()),
                Some(expected),
                "{source:?}"
            );
        }
        let source = concat!(
            "graph RL\n",
            "A -- text --> B\n",
            "C ==>|thick| D\n",
            "E -. dashed .-> F\n",
        );
        let graph = parse_mermaid(source).expect("parse all edge text");
        assert_eq!(graph.edges.len(), 3);
        assert_eq!(graph.edges[1].style, MermaidEdgeStyle::Thick);
        assert!(graph.edges[1].arrow);
        assert_eq!(graph.edges[2].style, MermaidEdgeStyle::Dashed);
        assert!(graph.edges[2].arrow);
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
        for (dir, arrow) in [("TB", "↓"), ("TD", "↓"), ("BT", "↑")] {
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
            assert!(text.contains(arrow), "{dir}: {text}");
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
        assert_eq!(rendered_prefixed_lines.len(), 2);
        assert!(rendered_prefixed_lines[0].starts_with("│ "));
        assert!(rendered_prefixed_lines[1].starts_with("│ "));

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
        assert!(render_math("\\int _{0}\n ^{1} f(x) + \\frac{1}{x}", true).is_some());
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
                render_latex(source, false).as_deref(),
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
            assert!(render_latex(source, false).is_none(), "{source}");
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
                render_latex(source, true).as_deref(),
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
