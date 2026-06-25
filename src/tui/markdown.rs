use pulldown_cmark::{CodeBlockKind, Event, HeadingLevel, Options, Parser, Tag, TagEnd};
use ratatui::{
    style::{Color, Modifier, Style},
    text::{Line, Span},
};
use std::sync::OnceLock;
use syntect::{
    easy::HighlightLines,
    highlighting::{FontStyle, Style as SyntectStyle, Theme as SyntectTheme, ThemeSet},
    parsing::{SyntaxReference, SyntaxSet},
};
use unicode_width::UnicodeWidthChar;

use crate::tui::{measure::display_width, theme::Theme};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SemanticMarkdownBlock {
    pub source: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct SemanticMarkdownLineOrigin {
    pub block_index: Option<usize>,
    pub content_prefix_chars: usize,
    pub content_char_offset: usize,
    pub content_char_len: usize,
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct SemanticMarkdownRender {
    pub source_blocks: Vec<SemanticMarkdownBlock>,
    pub line_origins: Vec<SemanticMarkdownLineOrigin>,
}

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

pub fn render_markdown(
    markdown: &str,
    theme: Theme,
    options: MarkdownRenderOptions,
) -> Vec<Line<'static>> {
    MarkdownRenderer::new(theme, options).render(markdown)
}

pub fn render_markdown_semantic_blocks(markdown: &str, width: usize) -> SemanticMarkdownRender {
    SemanticMarkdownRenderer::new(width).render(markdown)
}

#[derive(Debug)]
struct MarkdownRenderer {
    theme: Theme,
    options: MarkdownRenderOptions,
    lines: Vec<Line<'static>>,
    spans: Vec<Span<'static>>,
    inline: InlineState,
    block: BlockState,
    lists: Vec<ListState>,
    quote_depth: usize,
    item_prefix: Option<String>,
    in_code_block: Option<CodeBlockState>,
    table: Option<TableState>,
}

#[derive(Debug)]
struct SemanticMarkdownRenderer {
    width: usize,
    lines: Vec<SemanticLineEntry>,
    source_blocks: Vec<SemanticMarkdownBlock>,
    text: String,
    block: BlockState,
    lists: Vec<ListState>,
    quote_depth: usize,
    item_prefix: Option<String>,
    in_code_block: Option<CodeBlockState>,
    table: Option<TableState>,
    links: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct SemanticLineEntry {
    origin: SemanticMarkdownLineOrigin,
    is_blank: bool,
}

impl SemanticLineEntry {
    fn decoration(is_blank: bool) -> Self {
        Self {
            origin: SemanticMarkdownLineOrigin::default(),
            is_blank,
        }
    }
}

impl SemanticMarkdownRenderer {
    fn new(width: usize) -> Self {
        Self {
            width: width.max(1),
            lines: Vec::new(),
            source_blocks: Vec::new(),
            text: String::new(),
            block: BlockState::Document,
            lists: Vec::new(),
            quote_depth: 0,
            item_prefix: None,
            in_code_block: None,
            table: None,
            links: Vec::new(),
        }
    }

    fn render(mut self, markdown: &str) -> SemanticMarkdownRender {
        let parser = Parser::new_ext(markdown, markdown_options());
        for event in parser {
            self.handle_event(event);
        }
        self.flush_text();
        while self.lines.last().is_some_and(|line| line.is_blank) {
            self.lines.pop();
        }

        SemanticMarkdownRender {
            source_blocks: self.source_blocks,
            line_origins: self.lines.into_iter().map(|line| line.origin).collect(),
        }
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
            Event::Code(code)
            | Event::InlineMath(code)
            | Event::DisplayMath(code)
            | Event::Html(code)
            | Event::InlineHtml(code) => self.push_text(&code),
            Event::SoftBreak => self.push_text(" "),
            Event::HardBreak => self.push_text("\n"),
            Event::Rule => self.push_rule(),
            Event::TaskListMarker(done) => self.push_text(if done { "☑ " } else { "☐ " }),
            Event::FootnoteReference(reference) => self.push_text(&format!("[{reference}]")),
        }
    }

    fn start_tag(&mut self, tag: Tag<'_>) {
        match tag {
            Tag::Paragraph => {
                self.flush_text();
                self.block = BlockState::Paragraph;
            }
            Tag::Heading { level, .. } => {
                self.flush_text();
                self.block = BlockState::Heading(level);
            }
            Tag::BlockQuote(_) => {
                self.flush_text();
                self.quote_depth = self.quote_depth.saturating_add(1);
            }
            Tag::CodeBlock(kind) => {
                self.flush_text();
                self.in_code_block = Some(CodeBlockState {
                    language: code_block_language(kind),
                    content: String::new(),
                });
            }
            Tag::List(start) => {
                self.flush_text();
                self.lists.push(ListState::new(start));
            }
            Tag::Item => {
                self.flush_text();
                self.item_prefix = Some(self.next_item_prefix());
            }
            Tag::Link { dest_url, .. } => self.links.push(dest_url.to_string()),
            Tag::Table(_) => {
                self.flush_text();
                self.block = BlockState::Table;
                self.table = Some(TableState::default());
            }
            Tag::TableHead => {}
            Tag::TableRow => {
                if let Some(table) = self.table.as_mut() {
                    table.current_row.clear();
                }
            }
            Tag::TableCell => self.text.clear(),
            Tag::FootnoteDefinition(name) => {
                self.flush_text();
                self.push_text(&format!("[{name}] "));
            }
            Tag::DefinitionListDefinition => {
                self.flush_text();
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
            | Tag::Emphasis
            | Tag::Strong
            | Tag::Strikethrough
            | Tag::Superscript
            | Tag::Subscript => {}
        }
    }

    fn end_tag(&mut self, tag: TagEnd) {
        match tag {
            TagEnd::Paragraph => {
                self.flush_text();
                self.block = BlockState::Document;
            }
            TagEnd::Heading(_) => {
                self.flush_text();
                self.block = BlockState::Document;
            }
            TagEnd::BlockQuote(_) => {
                self.flush_text();
                self.quote_depth = self.quote_depth.saturating_sub(1);
            }
            TagEnd::CodeBlock => {
                if let Some(code) = self.in_code_block.take() {
                    self.push_code_block(code);
                }
            }
            TagEnd::List(_) => {
                self.flush_text();
                self.lists.pop();
            }
            TagEnd::Item => {
                self.flush_text();
                self.item_prefix = None;
            }
            TagEnd::Link => {
                if let Some(dest_url) = self.links.pop()
                    && !dest_url.is_empty()
                {
                    self.push_text(&format!(" <{dest_url}>"));
                }
            }
            TagEnd::Table => {
                if let Some(table) = self.table.take() {
                    self.push_table(table);
                }
                self.block = BlockState::Document;
            }
            TagEnd::TableHead => {
                if let Some(table) = self.table.as_mut() {
                    if !self.text.is_empty() {
                        table.current_row.push(std::mem::take(&mut self.text));
                    }
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
                    table.current_row.push(std::mem::take(&mut self.text));
                }
            }
            TagEnd::FootnoteDefinition
            | TagEnd::HtmlBlock
            | TagEnd::DefinitionList
            | TagEnd::DefinitionListTitle
            | TagEnd::DefinitionListDefinition
            | TagEnd::MetadataBlock(_)
            | TagEnd::Image => self.flush_text(),
            TagEnd::Emphasis
            | TagEnd::Strong
            | TagEnd::Strikethrough
            | TagEnd::Superscript
            | TagEnd::Subscript => {}
        }
    }

    fn push_text(&mut self, text: &str) {
        self.text.push_str(text);
    }

    fn flush_text(&mut self) {
        if self.text.is_empty() {
            return;
        }
        let source = std::mem::take(&mut self.text);
        let quote_prefix = if self.quote_depth == 0 {
            String::new()
        } else {
            "│ ".repeat(self.quote_depth)
        };

        let (source, first_prefix_chars, next_prefix_chars) = match self.block {
            BlockState::Heading(_) => {
                let (first_prefix, next_prefix) = self.line_prefixes();
                (
                    source,
                    display_width(&first_prefix),
                    display_width(&next_prefix),
                )
            }
            _ => {
                let marker = self.item_prefix.clone().unwrap_or_default();
                let mut semantic_source = marker.clone();
                semantic_source.push_str(&source);
                (
                    semantic_source,
                    display_width(&quote_prefix),
                    display_width(&quote_prefix).saturating_add(display_width(&marker)),
                )
            }
        };

        self.push_wrapped_block(source, first_prefix_chars, next_prefix_chars);
    }

    fn push_wrapped_block(&mut self, source: String, first_prefix_chars: usize, next_prefix_chars: usize) {
        let block_index = self.source_blocks.len();
        self.source_blocks.push(SemanticMarkdownBlock {
            source: source.clone(),
        });
        let first_width = self.width.saturating_sub(first_prefix_chars).max(1);
        let next_width = self.width.saturating_sub(next_prefix_chars).max(1);
        let chunks = wrap_plain_text_with_prefix_offsets(&source, first_width, next_width);
        for (index, chunk) in chunks.into_iter().enumerate() {
            self.lines.push(SemanticLineEntry {
                origin: SemanticMarkdownLineOrigin {
                    block_index: Some(block_index),
                    content_prefix_chars: if index == 0 {
                        first_prefix_chars
                    } else {
                        next_prefix_chars
                    },
                    content_char_offset: chunk.source_start_char,
                    content_char_len: chunk.source_end_char.saturating_sub(chunk.source_start_char),
                },
                is_blank: chunk.text.is_empty(),
            });
        }
    }

    fn push_rule(&mut self) {
        self.flush_text();
        self.lines.push(SemanticLineEntry::decoration(false));
    }

    fn push_code_block(&mut self, code: CodeBlockState) {
        self.flush_text();
        self.push_blank_line_before_block_card();
        self.lines.push(SemanticLineEntry::decoration(false));

        let (first_prefix, next_prefix) = self.line_prefixes();
        let prefix_width = display_width(&first_prefix).max(display_width(&next_prefix));
        let width = self.width.saturating_sub(prefix_width).max(1);
        let prefix_chars = display_width(&next_prefix).saturating_add(2);
        // 与真实 renderer 保持一致：MarkdownRenderer::push_code_block 中正文可用宽度为
        // `width.saturating_sub(3)`，其中 2 来自 `│ ` 边框，另 1 留给与 padded_line 相同
        // 的保守换行边界。若这里宽 1 格，会让 semantic wrap 比真实显示少换 1 行，
        // 从而把后续 origin 全部带歪。
        let body_width = width.saturating_sub(3).max(1);
        let mut pushed_body = false;

        for raw in code.content.lines() {
            pushed_body = true;
            let block_index = self.source_blocks.len();
            self.source_blocks.push(SemanticMarkdownBlock {
                source: raw.to_string(),
            });
            let chunks = wrap_plain_text_with_prefix_offsets(raw, body_width, body_width);
            for chunk in chunks {
                self.lines.push(SemanticLineEntry {
                    origin: SemanticMarkdownLineOrigin {
                        block_index: Some(block_index),
                        content_prefix_chars: prefix_chars,
                        content_char_offset: chunk.source_start_char,
                        content_char_len: chunk.source_end_char.saturating_sub(chunk.source_start_char),
                    },
                    is_blank: chunk.text.is_empty(),
                });
            }
        }

        if !pushed_body {
            let block_index = self.source_blocks.len();
            self.source_blocks.push(SemanticMarkdownBlock {
                source: String::new(),
            });
            self.lines.push(SemanticLineEntry {
                origin: SemanticMarkdownLineOrigin {
                    block_index: Some(block_index),
                    content_prefix_chars: prefix_chars,
                    content_char_offset: 0,
                    content_char_len: 0,
                },
                is_blank: true,
            });
        }

        self.lines.push(SemanticLineEntry::decoration(false));
        self.lines.push(SemanticLineEntry::decoration(true));
        let _ = body_width;
    }

    fn push_blank_line_before_block_card(&mut self) {
        if self.lines.last().is_some_and(|line| !line.is_blank) {
            self.lines.push(SemanticLineEntry::decoration(true));
        }
    }

    fn push_table(&mut self, table: TableState) {
        self.flush_text();
        if table.rows.is_empty() {
            return;
        }
        for (row_index, row) in table.rows.into_iter().enumerate() {
            self.push_wrapped_block(row.join(" │ "), 0, 0);
            if table.header_rows > 0 && row_index + 1 == table.header_rows {
                self.lines.push(SemanticLineEntry::decoration(false));
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
        let quote = if self.quote_depth == 0 {
            String::new()
        } else {
            "│ ".repeat(self.quote_depth)
        };

        let block_prefix = match self.block {
            BlockState::Heading(level) => {
                if heading_number(level) <= 2 {
                    "▌ ".to_string()
                } else {
                    "• ".to_string()
                }
            }
            _ => self.item_prefix.clone().unwrap_or_default(),
        };

        let first = format!("{quote}{block_prefix}");
        let continuation = format!("{}{}", quote, " ".repeat(display_width(&block_prefix)));
        (first, continuation)
    }
}

impl MarkdownRenderer {
    fn new(theme: Theme, options: MarkdownRenderOptions) -> Self {
        Self {
            theme,
            options,
            lines: Vec::new(),
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

    fn render(mut self, markdown: &str) -> Vec<Line<'static>> {
        let parser = Parser::new_ext(markdown, markdown_options());

        for event in parser {
            self.handle_event(event);
        }

        self.flush_spans();
        trim_trailing_blank_lines(&mut self.lines);

        if self.lines.is_empty() {
            self.lines
                .push(Line::from(Span::styled("…", muted_style(self.theme))));
        }

        self.lines
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
            Event::Html(html) | Event::InlineHtml(html) => self.push_text(&html),
            Event::SoftBreak => self.push_text(" "),
            Event::HardBreak => self.push_text("\n"),
            Event::Rule => self.push_rule(),
            Event::TaskListMarker(done) => self.push_text(if done { "☑ " } else { "☐ " }),
            Event::FootnoteReference(reference) => self.push_text(&format!("[{reference}]")),
            Event::InlineMath(math) | Event::DisplayMath(math) => self.push_inline_code(&math),
        }
    }

    fn start_tag(&mut self, tag: Tag<'_>) {
        match tag {
            Tag::Paragraph => {
                self.flush_spans();
                self.block = BlockState::Paragraph;
            }
            Tag::Heading { level, .. } => {
                self.flush_spans();
                self.block = BlockState::Heading(level);
            }
            Tag::BlockQuote(_) => {
                self.flush_spans();
                self.quote_depth = self.quote_depth.saturating_add(1);
            }
            Tag::CodeBlock(kind) => {
                self.flush_spans();
                self.in_code_block = Some(CodeBlockState {
                    language: code_block_language(kind),
                    content: String::new(),
                });
            }
            Tag::List(start) => {
                self.flush_spans();
                self.lists.push(ListState::new(start));
            }
            Tag::Item => {
                self.flush_spans();
                self.item_prefix = Some(self.next_item_prefix());
            }
            Tag::Emphasis => self.inline.emphasis = self.inline.emphasis.saturating_add(1),
            Tag::Strong => self.inline.strong = self.inline.strong.saturating_add(1),
            Tag::Strikethrough => {
                self.inline.strikethrough = self.inline.strikethrough.saturating_add(1)
            }
            Tag::Link { dest_url, .. } => {
                self.inline.links.push(dest_url.to_string());
            }
            Tag::Table(_) => {
                self.flush_spans();
                self.block = BlockState::Table;
                self.table = Some(TableState::default());
            }
            Tag::TableHead => {}
            Tag::TableRow => {
                if let Some(table) = self.table.as_mut() {
                    table.current_row.clear();
                }
            }
            Tag::TableCell => self.spans.clear(),
            Tag::FootnoteDefinition(name) => {
                self.flush_spans();
                self.push_text(&format!("[{name}] "));
            }
            Tag::HtmlBlock
            | Tag::MetadataBlock(_)
            | Tag::DefinitionList
            | Tag::DefinitionListTitle => {}
            Tag::DefinitionListDefinition => {
                self.flush_spans();
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
            Tag::Superscript | Tag::Subscript => {}
        }
    }

    fn end_tag(&mut self, tag: TagEnd) {
        match tag {
            TagEnd::Paragraph => {
                self.flush_spans();
                self.block = BlockState::Document;
            }
            TagEnd::Heading(_) => {
                self.flush_spans();
                self.block = BlockState::Document;
            }
            TagEnd::BlockQuote(_) => {
                self.flush_spans();
                self.quote_depth = self.quote_depth.saturating_sub(1);
            }
            TagEnd::CodeBlock => {
                if let Some(code) = self.in_code_block.take() {
                    self.push_code_block(code);
                }
            }
            TagEnd::List(_) => {
                self.flush_spans();
                self.lists.pop();
            }
            TagEnd::Item => {
                self.flush_spans();
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
                    self.push_styled(&format!(" <{dest_url}>"), link_dest_style(self.theme));
                }
            }
            TagEnd::Table => {
                if let Some(table) = self.table.take() {
                    self.push_table(table);
                }
                self.block = BlockState::Document;
            }
            TagEnd::TableHead => {
                if let Some(table) = self.table.as_mut() {
                    if !table.current_row.is_empty() {
                        table.rows.push(std::mem::take(&mut table.current_row));
                    }
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
                let cell = self.spans_to_plain_text();
                if let Some(table) = self.table.as_mut() {
                    table.current_row.push(cell);
                }
            }
            TagEnd::FootnoteDefinition
            | TagEnd::HtmlBlock
            | TagEnd::DefinitionList
            | TagEnd::DefinitionListTitle
            | TagEnd::DefinitionListDefinition
            | TagEnd::MetadataBlock(_)
            | TagEnd::Image => {
                self.flush_spans();
            }
            TagEnd::Superscript | TagEnd::Subscript => {}
        }
    }

    fn push_text(&mut self, text: &str) {
        self.push_styled(text, self.inline_style());
    }

    fn push_inline_code(&mut self, text: &str) {
        self.push_styled(text, inline_code_style(self.theme));
    }

    fn push_styled(&mut self, text: &str, style: Style) {
        if text.is_empty() {
            return;
        }
        self.spans.push(Span::styled(text.to_string(), style));
    }

    fn flush_spans(&mut self) {
        if self.spans.is_empty() {
            return;
        }

        let spans = std::mem::take(&mut self.spans);
        let (first_prefix, next_prefix) = self.line_prefixes();
        let content_width = self
            .options
            .width
            .saturating_sub(display_width(&first_prefix))
            .max(1);
        let next_width = self
            .options
            .width
            .saturating_sub(display_width(&next_prefix))
            .max(1);
        let style = self.block_style();
        let wrapped = wrap_spans_with_prefixes(
            spans,
            content_width,
            next_width,
            Prefix::new(first_prefix, self.prefix_style()),
            Prefix::new(next_prefix, self.prefix_style()),
            style,
        );
        self.lines.extend(wrapped);
    }

    fn push_rule(&mut self) {
        self.flush_spans();
        let (prefix, _) = self.line_prefixes();
        let prefix_style = self.prefix_style();
        let rule_width = self
            .options
            .width
            .saturating_sub(display_width(&prefix))
            .max(1);

        let mut spans = Vec::new();
        if !prefix.is_empty() {
            spans.push(Span::styled(prefix, prefix_style));
        }
        spans.push(Span::styled(
            "─".repeat(rule_width),
            muted_style(self.theme),
        ));
        self.lines.push(Line::from(spans));
    }

    fn push_code_block(&mut self, code: CodeBlockState) {
        self.flush_spans();
        self.push_blank_line_before_block_card();

        let (first_prefix, next_prefix) = self.line_prefixes();
        let prefix_style = self.prefix_style();
        let prefix_width = display_width(&first_prefix).max(display_width(&next_prefix));
        let width = self.options.width.saturating_sub(prefix_width).max(1);
        let code_style = code_block_style(self.theme);
        let border_style = code_block_border_style(self.theme);
        let label_style = code_block_label_style(self.theme);
        let mut highlighter = CodeHighlighter::new(code.language.as_deref(), self.theme);

        let label = code
            .language
            .as_deref()
            .filter(|language| !language.is_empty())
            .unwrap_or("code");
        self.push_prefixed_code_line(
            &first_prefix,
            prefix_style,
            padded_line(
                vec![
                    Span::styled("╭─ ", border_style),
                    Span::styled(label.to_string(), label_style),
                ],
                width,
                border_style,
            ),
        );

        let body_width = width.saturating_sub(3).max(1);
        let mut pushed_body = false;
        for raw in code.content.lines() {
            if raw.is_empty() {
                pushed_body = true;
                self.push_code_body_line(
                    &next_prefix,
                    prefix_style,
                    Vec::new(),
                    width,
                    code_style,
                    border_style,
                );
                continue;
            }

            for wrapped in wrap_plain_to_width(raw, body_width) {
                pushed_body = true;
                let highlighted = highlighter.highlight_line(&wrapped, code_style);
                self.push_code_body_line(
                    &next_prefix,
                    prefix_style,
                    highlighted,
                    width,
                    code_style,
                    border_style,
                );
            }
        }

        if !pushed_body {
            self.push_code_body_line(
                &next_prefix,
                prefix_style,
                Vec::new(),
                width,
                code_style,
                border_style,
            );
        }

        self.push_prefixed_code_line(
            &next_prefix,
            prefix_style,
            padded_line(vec![Span::styled("╰", border_style)], width, border_style),
        );
        self.lines.push(Line::default());
    }

    fn push_blank_line_before_block_card(&mut self) {
        if self
            .lines
            .last()
            .is_some_and(|line| !line.to_string().is_empty())
        {
            self.lines.push(Line::default());
        }
    }

    fn push_code_body_line(
        &mut self,
        prefix: &str,
        prefix_style: Style,
        content: Vec<Span<'static>>,
        width: usize,
        code_style: Style,
        border: Style,
    ) {
        self.push_prefixed_code_line(
            prefix,
            prefix_style,
            padded_line(code_body_spans(content, border), width, code_style),
        );
    }

    fn push_prefixed_code_line(&mut self, prefix: &str, prefix_style: Style, line: Line<'static>) {
        let mut spans = Vec::new();
        if !prefix.is_empty() {
            spans.push(Span::styled(prefix.to_string(), prefix_style));
        }
        spans.extend(line.spans);
        self.lines.push(Line::from(spans));
    }

    fn push_table(&mut self, table: TableState) {
        self.flush_spans();
        if table.rows.is_empty() {
            return;
        }

        let (first_prefix, next_prefix) = self.line_prefixes();
        let prefix_style = self.prefix_style();
        let width = self
            .options
            .width
            .saturating_sub(display_width(&first_prefix).max(display_width(&next_prefix)))
            .max(1);
        let col_count = table.rows.iter().map(Vec::len).max().unwrap_or(0);
        if col_count == 0 {
            return;
        }

        let mut widths = vec![1usize; col_count];
        for row in &table.rows {
            for (index, cell) in row.iter().enumerate() {
                widths[index] = widths[index].max(display_width(cell).min(24));
            }
        }
        fit_column_widths(&mut widths, width);

        for (row_index, row) in table.rows.iter().enumerate() {
            let prefix = if row_index == 0 {
                &first_prefix
            } else {
                &next_prefix
            };
            let row_style = if row_index < table.header_rows {
                table_header_style(self.theme)
            } else {
                table_style(self.theme)
            };
            let row_spans =
                table_row_spans(row, &widths, row_style, table_border_style(self.theme));
            self.push_prefixed_code_line(
                prefix,
                prefix_style,
                padded_line(row_spans, width, self.theme.app_style()),
            );

            if table.header_rows > 0 && row_index + 1 == table.header_rows {
                self.push_prefixed_code_line(
                    &next_prefix,
                    prefix_style,
                    padded_line(
                        vec![Span::styled(
                            table_separator(&widths),
                            table_border_style(self.theme),
                        )],
                        width,
                        self.theme.app_style(),
                    ),
                );
            }
        }
    }

    fn spans_to_plain_text(&mut self) -> String {
        std::mem::take(&mut self.spans)
            .into_iter()
            .map(|span| span.content.into_owned())
            .collect::<String>()
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
        let quote = if self.quote_depth == 0 {
            String::new()
        } else {
            "│ ".repeat(self.quote_depth)
        };

        let block_prefix = match self.block {
            BlockState::Heading(level) => {
                if heading_number(level) <= 2 {
                    "▌ ".to_string()
                } else {
                    "• ".to_string()
                }
            }
            _ => self.item_prefix.clone().unwrap_or_default(),
        };

        let first = format!("{quote}{block_prefix}");
        let continuation = format!("{}{}", quote, " ".repeat(display_width(&block_prefix)));
        (first, continuation)
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
        if self.inline.links.last().is_some() {
            style = style
                .fg(self.theme.accent)
                .add_modifier(Modifier::UNDERLINED);
        }
        style
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

fn code_body_spans(mut content: Vec<Span<'static>>, border: Style) -> Vec<Span<'static>> {
    let mut spans = vec![Span::styled("│ ", border)];
    spans.append(&mut content);
    spans
}

#[derive(Debug, Default)]
struct TableState {
    rows: Vec<Vec<String>>,
    current_row: Vec<String>,
    header_rows: usize,
}

#[derive(Debug, Clone)]
struct Prefix {
    content: String,
    style: Style,
}

impl Prefix {
    fn new(content: String, style: Style) -> Self {
        Self { content, style }
    }
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

fn wrap_spans_with_prefixes(
    spans: Vec<Span<'static>>,
    first_width: usize,
    next_width: usize,
    first_prefix: Prefix,
    next_prefix: Prefix,
    fallback_style: Style,
) -> Vec<Line<'static>> {
    let mut lines = Vec::new();
    let mut current = Vec::new();
    let mut current_width = 0usize;
    let mut current_limit = first_width.max(1);
    let mut at_line_start = true;
    let mut emitted_any = false;

    for span in spans {
        let style = span.style;
        for ch in span.content.chars() {
            if ch == '\n' {
                push_wrapped_line(
                    &mut lines,
                    &mut current,
                    &first_prefix,
                    &next_prefix,
                    emitted_any,
                );
                current_width = 0;
                current_limit = next_width.max(1);
                at_line_start = true;
                emitted_any = true;
                continue;
            }

            let ch_width = UnicodeWidthChar::width(ch).unwrap_or(0);
            if ch_width == 0 {
                push_char_span(&mut current, ch, style);
                continue;
            }

            if ch_width > current_limit && current.is_empty() {
                continue;
            }

            if current_width > 0 && current_width.saturating_add(ch_width) > current_limit {
                push_wrapped_line(
                    &mut lines,
                    &mut current,
                    &first_prefix,
                    &next_prefix,
                    emitted_any,
                );
                current_width = 0;
                current_limit = next_width.max(1);
                at_line_start = true;
                emitted_any = true;
            }

            if at_line_start && ch == ' ' {
                continue;
            }

            push_char_span(&mut current, ch, style);
            current_width = current_width.saturating_add(ch_width);
            at_line_start = false;
        }
    }

    if current.is_empty() && lines.is_empty() {
        current.push(Span::styled(String::new(), fallback_style));
    }
    if !current.is_empty() {
        push_wrapped_line(
            &mut lines,
            &mut current,
            &first_prefix,
            &next_prefix,
            emitted_any,
        );
    }

    lines
}

fn wrap_plain_text_with_prefix_offsets(
    text: &str,
    first_width: usize,
    next_width: usize,
) -> Vec<crate::tui::measure::WrappedChunk> {
    let mut lines = Vec::new();
    let mut current = String::new();
    let mut current_width = 0usize;
    let mut current_start = 0usize;
    let mut current_limit = first_width.max(1);
    let mut next_char_offset = 0usize;
    let mut at_line_start = true;
    let mut emitted_any = false;

    for (char_offset, ch) in text.chars().enumerate() {
        next_char_offset = char_offset + 1;
        if ch == '\n' {
            lines.push(crate::tui::measure::WrappedChunk {
                text: std::mem::take(&mut current),
                source_start_char: current_start,
                source_end_char: char_offset,
            });
            current_width = 0;
            current_start = next_char_offset;
            current_limit = next_width.max(1);
            at_line_start = true;
            emitted_any = true;
            continue;
        }

        let ch_width = UnicodeWidthChar::width(ch).unwrap_or(0);
        if ch_width == 0 {
            current.push(ch);
            continue;
        }

        if ch_width > current_limit && current.is_empty() {
            continue;
        }

        if current_width > 0 && current_width.saturating_add(ch_width) > current_limit {
            lines.push(crate::tui::measure::WrappedChunk {
                text: std::mem::take(&mut current),
                source_start_char: current_start,
                source_end_char: char_offset,
            });
            current_width = 0;
            current_start = char_offset;
            current_limit = next_width.max(1);
            at_line_start = true;
            emitted_any = true;
        }

        if at_line_start && ch == ' ' {
            current_start = next_char_offset;
            continue;
        }

        current.push(ch);
        current_width = current_width.saturating_add(ch_width);
        at_line_start = false;
    }

    if current.is_empty() && lines.is_empty() && !emitted_any {
        lines.push(crate::tui::measure::WrappedChunk {
            text: String::new(),
            source_start_char: 0,
            source_end_char: 0,
        });
    } else if !current.is_empty() || text.ends_with('\n') {
        lines.push(crate::tui::measure::WrappedChunk {
            text: current,
            source_start_char: current_start,
            source_end_char: next_char_offset,
        });
    }

    lines
}

fn push_wrapped_line(
    lines: &mut Vec<Line<'static>>,
    current: &mut Vec<Span<'static>>,
    first_prefix: &Prefix,
    next_prefix: &Prefix,
    continuation: bool,
) {
    let prefix = if continuation {
        next_prefix
    } else {
        first_prefix
    };
    let mut spans = Vec::new();
    if !prefix.content.is_empty() {
        spans.push(Span::styled(prefix.content.clone(), prefix.style));
    }
    spans.append(current);
    lines.push(Line::from(spans));
}

fn push_char_span(spans: &mut Vec<Span<'static>>, ch: char, style: Style) {
    if let Some(last) = spans.last_mut()
        && last.style == style
    {
        last.content.to_mut().push(ch);
        return;
    }

    spans.push(Span::styled(ch.to_string(), style));
}

fn wrap_plain_to_width(text: &str, width: usize) -> Vec<String> {
    let mut rows = Vec::new();
    let mut current = String::new();
    let mut current_width = 0usize;

    for ch in text.chars() {
        let ch_width = UnicodeWidthChar::width(ch).unwrap_or(0);
        if ch_width == 0 {
            current.push(ch);
            continue;
        }
        if current_width > 0 && current_width.saturating_add(ch_width) > width {
            rows.push(current);
            current = String::new();
            current_width = 0;
        }
        if ch_width <= width {
            current.push(ch);
            current_width = current_width.saturating_add(ch_width);
        }
    }

    if !current.is_empty() || rows.is_empty() {
        rows.push(current);
    }
    rows
}

fn padded_line(mut spans: Vec<Span<'static>>, width: usize, fill_style: Style) -> Line<'static> {
    spans = truncate_spans(spans, width);
    let used = display_width(
        &spans
            .iter()
            .map(|span| span.content.as_ref())
            .collect::<String>(),
    );
    if width > used {
        spans.push(Span::styled(" ".repeat(width - used), fill_style));
    }
    Line::from(spans)
}

fn truncate_spans(spans: Vec<Span<'static>>, width: usize) -> Vec<Span<'static>> {
    let mut result = Vec::new();
    let mut used = 0usize;

    for span in spans {
        if used >= width {
            break;
        }
        let mut content = String::new();
        for ch in span.content.chars() {
            let ch_width = UnicodeWidthChar::width(ch).unwrap_or(0);
            if ch_width == 0 {
                content.push(ch);
                continue;
            }
            if used.saturating_add(ch_width) > width {
                break;
            }
            content.push(ch);
            used = used.saturating_add(ch_width);
        }
        if !content.is_empty() {
            result.push(Span::styled(content, span.style));
        }
    }

    result
}

fn fit_column_widths(widths: &mut [usize], total_width: usize) {
    if widths.is_empty() {
        return;
    }

    let separator_width = widths.len().saturating_sub(1).saturating_mul(3);
    let available = total_width
        .saturating_sub(separator_width)
        .max(widths.len());
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

fn table_row_spans(
    row: &[String],
    widths: &[usize],
    cell_style: Style,
    separator_style: Style,
) -> Vec<Span<'static>> {
    let mut spans = Vec::new();
    for (index, width) in widths.iter().copied().enumerate() {
        if index > 0 {
            spans.push(Span::styled(" │ ", separator_style));
        }
        let cell = row.get(index).map(String::as_str).unwrap_or_default();
        let rendered = pad_cell(cell, width);
        spans.push(Span::styled(rendered, cell_style));
    }
    spans
}

fn pad_cell(cell: &str, width: usize) -> String {
    let truncated = truncate_string(cell.trim(), width);
    let used = display_width(&truncated);
    if used < width {
        format!("{truncated}{}", " ".repeat(width - used))
    } else {
        truncated
    }
}

fn truncate_string(text: &str, width: usize) -> String {
    let mut result = String::new();
    let mut used = 0usize;
    for ch in text.chars() {
        let ch_width = UnicodeWidthChar::width(ch).unwrap_or(0);
        if ch_width == 0 {
            result.push(ch);
            continue;
        }
        if used.saturating_add(ch_width) > width {
            break;
        }
        result.push(ch);
        used = used.saturating_add(ch_width);
    }
    result
}

fn table_separator(widths: &[usize]) -> String {
    widths
        .iter()
        .map(|width| "─".repeat((*width).max(1)))
        .collect::<Vec<_>>()
        .join("─┼─")
}

fn trim_trailing_blank_lines(lines: &mut Vec<Line<'static>>) {
    while lines.last().is_some_and(|line| line.to_string().is_empty()) {
        lines.pop();
    }
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

    fn semantic(markdown: &str, width: usize) -> SemanticMarkdownRender {
        render_markdown_semantic_blocks(markdown, width)
    }

    #[test]
    fn renders_headings_lists_and_inline_markdown() {
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
    fn renders_code_blocks_as_padded_cards_with_language_label() {
        let lines = rendered("```rust\nlet x = 1;\n```", 24);
        let text = lines
            .iter()
            .map(Line::to_string)
            .collect::<Vec<_>>()
            .join("\n");

        assert!(text.contains("╭─ rust"), "{text}");
        assert!(text.contains("│ let x = 1;"), "{text}");
        assert!(text.contains('╰'), "{text}");
        assert_eq!(lines.len(), 3, "{text}");
        for line in lines.iter().filter(|line| !line.to_string().is_empty()) {
            assert_eq!(display_width(&line.to_string()), 24, "{line:?}");
        }
    }

    #[test]
    fn code_blocks_have_vertical_spacing_from_surrounding_text() {
        let lines = rendered("before\n\n```sh\npwd\n```\n\nafter", 24);
        let text = lines
            .iter()
            .map(Line::to_string)
            .collect::<Vec<_>>()
            .join("\n");

        assert_eq!(lines[0].to_string(), "before");
        assert_eq!(lines[1].to_string(), "");
        assert!(lines[2].to_string().contains("╭─ sh"), "{text}");
        assert!(lines[4].to_string().contains('╰'), "{text}");
        assert_eq!(lines[5].to_string(), "");
        assert_eq!(lines[6].to_string(), "after");
    }

    #[test]
    fn code_blocks_apply_syntax_highlighting_for_known_languages() {
        let theme = Theme::dark();
        let lines = rendered("```rust\nlet x = 1;\n```", 32);
        let body = lines
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
    }

    #[test]
    fn unknown_code_languages_fall_back_to_plain_code_style() {
        let theme = Theme::dark();
        let lines = rendered("```definitely-not-a-language\nlet x = 1;\n```", 32);
        let body = lines
            .iter()
            .find(|line| line.to_string().contains("let x"))
            .expect("plain code body line");

        assert!(
            body.spans
                .iter()
                .any(|span| span.content.as_ref() == "let x = 1;"
                    && span.style.fg == Some(theme.text)
                    && span.style.bg == Some(theme.element_bg)),
            "{body:?}"
        );
    }

    #[test]
    fn keeps_selected_inline_styles_after_wrapping() {
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
    fn renders_ordered_lists_blockquotes_and_task_markers() {
        let lines = rendered("> 1. [x] done\n> 2. [ ] todo", 80);
        let text = lines
            .iter()
            .map(Line::to_string)
            .collect::<Vec<_>>()
            .join("\n");

        assert!(text.contains("│ 1. ☑ done"), "{text}");
        assert!(text.contains("│ 2. ☐ todo"), "{text}");
    }

    #[test]
    fn code_blocks_are_width_safe_at_tiny_widths_and_long_labels() {
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
    }

    #[test]
    fn code_blocks_keep_quote_context_and_preserve_leading_empty_rows() {
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
    fn links_preserve_destinations() {
        let text = rendered("[docs](https://example.com)", 80)
            .iter()
            .map(Line::to_string)
            .collect::<Vec<_>>()
            .join("\n");

        assert!(text.contains("docs <https://example.com>"), "{text}");
    }

    #[test]
    fn renders_tables_with_header_separator() {
        let text = rendered("| A | B |\n|---|---|\n| one | two |", 40)
            .iter()
            .map(Line::to_string)
            .collect::<Vec<_>>()
            .join("\n");

        assert!(text.contains("A"), "{text}");
        assert!(text.contains("one"), "{text}");
        assert!(text.contains('┼'), "{text}");
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

    #[test]
    fn semantic_render_keeps_paragraphs_unbroken_across_visual_wrap() {
        let semantic = semantic("alpha beta gamma delta", 8);

        assert_eq!(semantic.source_blocks.len(), 1);
        assert_eq!(semantic.source_blocks[0].source, "alpha beta gamma delta");
        assert!(semantic.line_origins.len() > 1, "{semantic:?}");
        assert!(semantic
            .line_origins
            .iter()
            .all(|origin| origin.block_index == Some(0)));
    }

    #[test]
    fn semantic_render_strips_code_block_chrome_but_preserves_code_lines() {
        let rendered = rendered("```rust\nlet x = 1;\nlet y = 2;\n```", 24);
        let semantic = semantic("```rust\nlet x = 1;\nlet y = 2;\n```", 24);

        assert_eq!(rendered.len(), semantic.line_origins.len());
        assert_eq!(
            semantic
                .source_blocks
                .iter()
                .map(|block| block.source.as_str())
                .collect::<Vec<_>>(),
            vec!["let x = 1;", "let y = 2;"]
        );
        assert_eq!(semantic.line_origins[0].block_index, None);
        assert_eq!(semantic.line_origins[1].block_index, Some(0));
        assert_eq!(semantic.line_origins[2].block_index, Some(1));
        assert_eq!(semantic.line_origins[3].block_index, None);
    }

    #[test]
    fn semantic_render_keeps_list_markers_but_not_quote_guides() {
        let semantic = semantic("> 1. item\n> - next", 24);

        assert_eq!(
            semantic
                .source_blocks
                .iter()
                .map(|block| block.source.as_str())
                .collect::<Vec<_>>(),
            vec!["1. item", "• next"]
        );
        assert!(semantic
            .line_origins
            .iter()
            .all(|origin| origin.content_prefix_chars == 2));
    }

    #[test]
    fn semantic_render_stays_line_aligned_with_rendered_for_mixed_paragraph_and_code_block() {
        let markdown = "这里的实现思路是：\n\n```text\n模型不能凭空发明工具\n↓\n只能调用 ToolRegistry 中注册过的工具\n↓\nToolRegistry 再按 scope 过滤模型能看到什么\n```\n\n下一段说明";
        let rendered = rendered(markdown, 44);
        let semantic = semantic(markdown, 44);

        assert_eq!(
            rendered.len(),
            semantic.line_origins.len(),
            "rendered/semantic line count mismatch: rendered={:?} semantic={:?}",
            rendered.iter().map(|line| line.to_string()).collect::<Vec<_>>(),
            semantic.line_origins,
        );
    }
}
