use ratatui::{
    Frame,
    layout::{Constraint, Layout, Rect},
    style::{Modifier, Style},
    text::{Line, Span, Text},
    widgets::{Block, Paragraph, Scrollbar, ScrollbarOrientation, ScrollbarState},
};

use crate::subagent::{
    looks_like_structured_subagent_output, try_parse_structured_subagent_result,
};
use crate::tui::{
    markdown::{MarkdownRenderOptions, render_markdown_document},
    measure::{display_width, wrap_text_to_width, wrap_text_to_width_with_offsets},
    surface,
    theme::Theme,
    timeline::{
        DelegationView, ErrorView, MessageView, PermissionPromptStatus, PermissionView,
        ReasoningView, TimelineItem, ToolView,
    },
    transcript_ratatui,
    transcript_render::{
        Break, Component, CopyJoin, Document, Line as RenderLine, SourceRange, Span as RenderSpan,
        inclusive_grapheme_bounds,
    },
};
use crate::user_content::UserImageAttachment;

use super::super::state::TuiState;
use super::{
    composer::one_line_snippet, reviewer_cards, structured_subagent, todo_card, tool_card,
};

#[derive(Debug, Clone, Default)]
pub struct TranscriptRenderCache {
    width: Option<usize>,
    theme: Option<Theme>,
    timeline_cache_id: Option<u64>,
    row_metadata_revision: Option<u64>,
    total_rows: Option<usize>,
    entries: Vec<TranscriptRenderCacheEntry>,
    row_starts: Vec<usize>,
    row_counts: Vec<usize>,
    #[cfg(test)]
    row_count_rebuilds: usize,
}

impl TranscriptRenderCache {
    pub fn clear(&mut self) {
        self.width = None;
        self.theme = None;
        self.timeline_cache_id = None;
        self.row_metadata_revision = None;
        self.total_rows = None;
        self.entries.clear();
        self.row_starts.clear();
        self.row_counts.clear();
    }

    #[cfg(test)]
    pub(crate) fn is_empty(&self) -> bool {
        self.width.is_none()
            && self.theme.is_none()
            && self.timeline_cache_id.is_none()
            && self.row_metadata_revision.is_none()
            && self.total_rows.is_none()
            && self.entries.is_empty()
            && self.row_starts.is_empty()
            && self.row_counts.is_empty()
    }

    pub(crate) fn prepare(&mut self, width: usize, theme: Theme, timeline_cache_id: u64) {
        if self.width != Some(width)
            || self.theme != Some(theme)
            || self.timeline_cache_id != Some(timeline_cache_id)
        {
            self.width = Some(width);
            self.theme = Some(theme);
            self.timeline_cache_id = Some(timeline_cache_id);
            self.row_metadata_revision = None;
            self.total_rows = None;
            self.entries.clear();
            self.row_starts.clear();
            self.row_counts.clear();
        }
    }

    /// 获取缓存条目的引用（用于文本选择）
    pub fn entries(&self) -> &[TranscriptRenderCacheEntry] {
        &self.entries
    }

    #[cfg(test)]
    pub(crate) fn set_entries_for_test(&mut self, entries: Vec<TranscriptRenderCacheEntry>) {
        self.entries = entries;
    }

    #[cfg(test)]
    pub(crate) fn set_row_metadata_for_test(
        &mut self,
        row_starts: Vec<usize>,
        row_counts: Vec<usize>,
    ) {
        self.row_starts = row_starts;
        self.row_counts = row_counts;
    }

    /// 获取行起始位置的引用（用于坐标映射）
    pub fn row_starts(&self) -> &[usize] {
        &self.row_starts
    }
}

impl PartialEq for TranscriptRenderCache {
    fn eq(&self, _other: &Self) -> bool {
        true
    }
}

impl Eq for TranscriptRenderCache {}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TranscriptRenderCacheEntry {
    pub(crate) revision: Option<u64>,
    /// The engine document is the sole display and copy-mapping artifact.
    pub document: Document<Style>,
}

pub fn render_transcript(frame: &mut Frame<'_>, state: &mut TuiState, area: Rect, theme: Theme) {
    if area.is_empty() {
        return;
    }

    if state.active_timeline().items().is_empty() {
        // Welcome rendering is handled at a higher level.
        frame.render_widget(Block::new().style(theme.app_style()), area);
        return;
    }

    let has_scrollbar = state.transcript_scrollbar_visible && area.width >= 24;
    let (content_area, scrollbar_area) = if has_scrollbar {
        let chunks = Layout::horizontal([Constraint::Min(1), Constraint::Length(1)]).split(area);
        (chunks[0], Some(chunks[1]))
    } else {
        (area, None)
    };

    // 存储实际文本渲染区域（不含 scrollbar 列）与解析后的 top-relative 滚动偏移，
    // 供鼠标坐标映射与选择高亮使用——三者必须共用同一坐标系。
    state.last_transcript_area = content_area;

    let width = content_area.width.max(1) as usize;
    let total_rows = cached_transcript_row_count(state, theme, width);
    state.sync_transcript_viewport_rows(total_rows);
    let visible_rows = content_area.height;
    let scroll = crate::tui::measure::resolved_scroll_offset(
        total_rows,
        visible_rows,
        state.transcript_scroll,
        state.auto_scroll,
    );
    state.last_transcript_scroll_top = scroll;

    let visible_lines = visible_cached_transcript_lines(state, theme, width, visible_rows, scroll);
    let paragraph = Paragraph::new(Text::from(visible_lines)).style(theme.app_style());

    frame.render_widget(paragraph, content_area);
    let visible_lines = visible_document_lines(state, visible_rows, scroll);
    state.frame_hyperlink_cells = transcript_ratatui::collect_hyperlink_cells(
        frame.buffer_mut(),
        content_area,
        &visible_lines,
    );

    if let Some(scrollbar_area) = scrollbar_area
        && total_rows > visible_rows as usize
        && visible_rows > 0
    {
        let mut scrollbar_state = ScrollbarState::new(total_rows)
            .position(scroll as usize)
            .viewport_content_length(visible_rows as usize);
        let scrollbar = Scrollbar::new(ScrollbarOrientation::VerticalRight)
            .begin_symbol(None)
            .end_symbol(None)
            .thumb_style(Style::default().fg(theme.dim_text).bg(theme.root_bg))
            .track_style(Style::default().fg(theme.element_bg).bg(theme.root_bg));
        frame.render_stateful_widget(scrollbar, scrollbar_area, &mut scrollbar_state);
    }
}

#[cfg(test)]
fn visible_transcript_lines(
    lines: &[Line<'static>],
    visible_rows: u16,
    top_scroll: u16,
) -> Vec<Line<'static>> {
    let visible_rows = visible_rows as usize;
    if visible_rows == 0 {
        return Vec::new();
    }

    let start = (top_scroll as usize).min(lines.len());
    let end = start.saturating_add(visible_rows).min(lines.len());
    lines[start..end].to_vec()
}

#[cfg(test)]
pub fn transcript_row_count(state: &TuiState, theme: Theme, width: usize) -> usize {
    transcript_lines(state, theme, width).len()
}

#[cfg(test)]
pub(crate) fn transcript_lines(state: &TuiState, theme: Theme, width: usize) -> Vec<Line<'static>> {
    let mut lines = Vec::new();
    let items = state.active_timeline().items();

    if !items.is_empty() {
        lines.extend((0..surface::TRANSCRIPT_TOP_SPACER).map(|_| Line::from("")));
    }

    for (index, item) in items.iter().enumerate() {
        if timeline_item_needs_separator_before(index, items, state.thoughts_display) {
            lines.push(Line::from(""));
        }

        lines.extend(transcript_ratatui::document_to_ratatui(
            &render_timeline_item_document(
                item,
                theme,
                width,
                state.status_spinner_frame,
                tool_output_expanded_for_item(state, item),
                is_reviewer_child_view(state),
                state.thoughts_display,
                index + 1 < items.len() && matches!(items[index + 1], TimelineItem::Reasoning(_)),
            ),
        ));
    }

    lines
}

fn cached_transcript_row_count(state: &mut TuiState, theme: Theme, width: usize) -> usize {
    let item_count = state.active_timeline().items().len();
    if item_count == 0 {
        return 0;
    }

    let timeline = state.active_timeline();
    let timeline_cache_id = timeline.cache_id();
    let mutation_revision = timeline.mutation_revision();
    state
        .transcript_render_cache
        .prepare(width, theme, timeline_cache_id);

    if state.transcript_render_cache.row_metadata_revision == Some(mutation_revision) {
        return state
            .transcript_render_cache
            .total_rows
            .expect("current transcript row metadata has a total row count");
    }

    #[cfg(test)]
    {
        state.transcript_render_cache.row_count_rebuilds += 1;
    }

    state
        .transcript_render_cache
        .entries
        .resize_with(item_count, || TranscriptRenderCacheEntry {
            revision: None,
            document: Document::default(),
        });

    let mut rows = surface::TRANSCRIPT_TOP_SPACER;
    state.transcript_render_cache.row_starts.clear();
    state.transcript_render_cache.row_counts.clear();

    for index in 0..item_count {
        let separator_rows = if timeline_item_needs_separator_before(
            index,
            &state.active_timeline().items(),
            state.thoughts_display,
        ) {
            1
        } else {
            0
        };
        rows = rows.saturating_add(separator_rows);
        state.transcript_render_cache.row_starts.push(rows);
        let line_count = cached_item_line_count(state, index, theme, width);
        state.transcript_render_cache.row_counts.push(line_count);
        rows = rows.saturating_add(line_count);
    }

    state.transcript_render_cache.entries.truncate(item_count);
    state
        .transcript_render_cache
        .row_starts
        .truncate(item_count);
    state
        .transcript_render_cache
        .row_counts
        .truncate(item_count);
    state.transcript_render_cache.row_metadata_revision = Some(mutation_revision);
    state.transcript_render_cache.total_rows = Some(rows);
    rows
}

fn visible_cached_transcript_lines(
    state: &mut TuiState,
    theme: Theme,
    width: usize,
    visible_rows: u16,
    top_scroll: u16,
) -> Vec<Line<'static>> {
    let visible_rows = visible_rows as usize;
    if visible_rows == 0 || state.active_timeline().items().is_empty() {
        return Vec::new();
    }

    state
        .transcript_render_cache
        .prepare(width, theme, state.active_timeline().cache_id());
    if !transcript_row_metadata_is_current(state) {
        cached_transcript_row_count(state, theme, width);
    }

    let start = top_scroll as usize;
    let end = start.saturating_add(visible_rows);
    let mut visible = Vec::with_capacity(visible_rows);

    let top_spacer_end = surface::TRANSCRIPT_TOP_SPACER.min(end);
    for row in start..top_spacer_end {
        if row < surface::TRANSCRIPT_TOP_SPACER {
            visible.push(Line::from(""));
        }
    }

    let item_count = state.active_timeline().items().len();
    let first_item = state
        .transcript_render_cache
        .row_starts
        .partition_point(|row_start| *row_start < start)
        .saturating_sub(1);

    for index in first_item..item_count {
        let item_start = state.transcript_render_cache.row_starts[index];
        let item_count = state.transcript_render_cache.row_counts[index];
        let separator_rows = if timeline_item_needs_separator_before(
            index,
            &state.active_timeline().items(),
            state.thoughts_display,
        ) {
            1
        } else {
            0
        };
        let separator_start = item_start.saturating_sub(separator_rows);
        let item_end = item_start.saturating_add(item_count);

        if separator_start >= end || visible.len() >= visible_rows {
            break;
        }

        if separator_rows > 0 && separator_start >= start && separator_start < end {
            visible.push(Line::from(""));
        }

        if item_end <= start {
            continue;
        }

        let line_start = start.saturating_sub(item_start).min(item_count);
        let line_end = end.saturating_sub(item_start).min(item_count);
        let lines = cached_item_lines(state, index, theme, width);
        for line in &lines[line_start..line_end] {
            visible.push(line.clone());
            if visible.len() >= visible_rows {
                break;
            }
        }
    }

    // 应用选择高亮
    if let Some(selection) = &state.text_selection {
        apply_selection_highlight(&mut visible, selection, state, theme, top_scroll);
    }

    visible
}

fn visible_document_lines(
    state: &TuiState,
    visible_rows: u16,
    top_scroll: u16,
) -> Vec<Option<&RenderLine<Style>>> {
    let mut rows = vec![None; surface::TRANSCRIPT_TOP_SPACER];
    for (index, entry) in state.transcript_render_cache.entries.iter().enumerate() {
        if timeline_item_needs_separator_before(
            index,
            &state.active_timeline().items(),
            state.thoughts_display,
        ) {
            rows.push(None);
        }
        rows.extend(entry.document.lines.iter().map(Some));
    }
    let start = (top_scroll as usize).min(rows.len());
    let end = start.saturating_add(visible_rows as usize).min(rows.len());
    rows.drain(start..end).collect()
}

fn transcript_row_metadata_is_current(state: &TuiState) -> bool {
    let timeline = state.active_timeline();
    state.transcript_render_cache.row_metadata_revision == Some(timeline.mutation_revision())
        && state.transcript_render_cache.total_rows.is_some()
}

fn timeline_item_needs_separator_before(
    index: usize,
    items: &[TimelineItem],
    thoughts_display: crate::command::ThoughtsDisplayMode,
) -> bool {
    if index == 0 {
        return false;
    }
    match (&items[index - 1], &items[index]) {
        (TimelineItem::Todo(_), TimelineItem::Todo(_)) => false,
        (TimelineItem::Reasoning(_), TimelineItem::Reasoning(_)) => {
            thoughts_display != crate::command::ThoughtsDisplayMode::Compact
        }
        _ => true,
    }
}

fn tool_output_expanded_for_item(state: &TuiState, item: &TimelineItem) -> bool {
    let key = match item {
        TimelineItem::Tool(tool) => Some(tool.call_id.as_str()),
        TimelineItem::AutoReview(_) => None,
        _ => return state.tool_output_expanded,
    };
    let key = key.map(str::to_string).or_else(|| match item {
        TimelineItem::AutoReview(decision) => Some(format!("auto-review:{}", decision.call_id)),
        _ => None,
    });
    key.and_then(|key| state.tool_output_overrides.get(&key).copied())
        .unwrap_or(state.tool_output_expanded)
}

fn cached_item_line_count(state: &mut TuiState, index: usize, theme: Theme, width: usize) -> usize {
    refresh_cached_item_document(state, index, theme, width);
    state.transcript_render_cache.entries[index]
        .document
        .lines
        .len()
}

fn cached_item_lines(
    state: &mut TuiState,
    index: usize,
    theme: Theme,
    width: usize,
) -> Vec<Line<'static>> {
    refresh_cached_item_document(state, index, theme, width);
    transcript_ratatui::document_to_ratatui(&state.transcript_render_cache.entries[index].document)
}

fn refresh_cached_item_document(state: &mut TuiState, index: usize, theme: Theme, width: usize) {
    if state.transcript_render_cache.entries.len() <= index {
        state
            .transcript_render_cache
            .entries
            .resize_with(index + 1, || TranscriptRenderCacheEntry {
                revision: None,
                document: Document::default(),
            });
    }

    let revision = state.active_timeline().item_revisions().get(index).copied();
    let live = matches!(
        &state.active_timeline().items()[index],
        TimelineItem::Tool(tool)
            if matches!(
                tool.status,
                crate::tui::timeline::ToolExecutionStatus::Pending
                    | crate::tui::timeline::ToolExecutionStatus::Running
            )
    ) || matches!(
        &state.active_timeline().items()[index],
        TimelineItem::Reasoning(reasoning) if reasoning.streaming
    );
    if state.transcript_render_cache.entries[index].revision == revision && !live {
        return;
    }

    let item = state.active_timeline().items()[index].clone();
    let items = state.active_timeline().items();
    let next_reasoning =
        index + 1 < items.len() && matches!(items[index + 1], TimelineItem::Reasoning(_));
    let document = render_timeline_item_document(
        &item,
        theme,
        width,
        state.status_spinner_frame,
        tool_output_expanded_for_item(state, &item),
        is_reviewer_child_view(state),
        state.thoughts_display,
        next_reasoning,
    );
    let entry = &mut state.transcript_render_cache.entries[index];
    entry.revision = revision;
    entry.document = document;
}

struct TimelineDocument {
    document: Document<Style>,
}

impl TimelineDocument {
    fn add_source(&mut self, source: impl Into<String>) -> usize {
        self.document.add_source(source)
    }

    fn push_content(
        &mut self,
        prefix: impl Into<String>,
        prefix_style: Style,
        text: impl Into<String>,
        text_style: Style,
        source: SourceRange,
        suffix: impl Into<String>,
        suffix_style: Style,
        boundary: Break,
    ) {
        self.document.push_line(
            RenderLine {
                spans: vec![
                    RenderSpan::decoration(prefix, prefix_style),
                    RenderSpan::source(text, text_style, source),
                    RenderSpan::decoration(suffix, suffix_style),
                ],
            },
            normalize_boundary(boundary),
        );
    }

    fn push_decoration(&mut self, line: Line<'static>, boundary: Break) {
        self.document.push_line(
            RenderLine {
                spans: line
                    .spans
                    .into_iter()
                    .map(|span| RenderSpan::decoration(span.content.into_owned(), span.style))
                    .collect(),
            },
            normalize_boundary(boundary),
        );
    }

    fn push_line(&mut self, line: RenderLine<Style>, boundary: Break) {
        self.document.push_line(line, normalize_boundary(boundary));
    }
}

fn normalize_boundary(boundary: Break) -> Break {
    match boundary {
        Break::End => Break::SoftWrap,
        boundary => boundary,
    }
}

struct TimelineItemComponent<'a> {
    item: &'a TimelineItem,
    theme: Theme,
    width: usize,
    frame: usize,
    expanded_output: bool,
    reviewer_view: bool,
    thoughts_display: crate::command::ThoughtsDisplayMode,
    next_reasoning: bool,
}

impl Component<Style> for TimelineItemComponent<'_> {
    fn render(&self, document: &mut Document<Style>) {
        if self.reviewer_view
            && let Some(specialized) =
                try_render_reviewer_view_item(self.item, self.theme, self.width)
        {
            *document = specialized;
            return;
        }

        let mut out = TimelineDocument {
            document: std::mem::take(document),
        };
        match self.item {
            TimelineItem::User(message) => {
                build_user_message(&mut out, message, self.theme, self.width)
            }
            TimelineItem::Reasoning(reasoning) => build_reasoning_lines(
                &mut out,
                reasoning,
                self.theme,
                self.width,
                self.thoughts_display,
                self.next_reasoning,
            ),
            TimelineItem::Delegation(delegation) => {
                build_delegation_lines(&mut out, delegation, self.theme, self.width)
            }
            TimelineItem::Assistant(message) => build_assistant_message_lines(
                &mut out,
                message_text(message),
                message.streaming,
                self.theme,
                self.width,
            ),
            TimelineItem::Tool(tool) => build_tool_lines(
                &mut out,
                tool,
                self.theme,
                self.width,
                self.frame,
                self.expanded_output,
            ),
            TimelineItem::Todo(todo) => build_todo_card(&mut out, todo, self.theme, self.width),
            TimelineItem::Permission(permission) => {
                build_permission_lines(&mut out, permission, self.theme, self.width)
            }
            TimelineItem::AutoReview(decision) => build_auto_review_lines(
                &mut out,
                decision,
                self.theme,
                self.width,
                self.expanded_output,
            ),
            TimelineItem::Error(error) => {
                build_error_lines(&mut out, error, self.theme, self.width)
            }
            TimelineItem::Compaction(view) => build_compaction_block_lines(
                &mut out,
                &view.summary,
                view.streaming,
                self.theme,
                self.width,
            ),
        }
        *document = out.document;
    }
}

fn is_reviewer_child_view(state: &TuiState) -> bool {
    state
        .child_view_metadata()
        .is_some_and(|meta| meta.agent_name == "reviewer")
}

fn try_render_reviewer_view_item(
    item: &TimelineItem,
    theme: Theme,
    width: usize,
) -> Option<Document<Style>> {
    match item {
        TimelineItem::User(message) => {
            let card = reviewer_cards::parse_review_request(message_text(message))?;
            Some(reviewer_cards::render_review_request_card_document(
                &card,
                theme,
                card_content_width(width),
            ))
        }
        TimelineItem::Assistant(message) => {
            let card = reviewer_cards::parse_review_decision(message_text(message))?;
            Some(reviewer_cards::render_review_decision_card_document(
                &card,
                theme,
                card_content_width(width),
            ))
        }
        _ => None,
    }
}

fn render_timeline_item_document(
    item: &TimelineItem,
    theme: Theme,
    width: usize,
    frame: usize,
    expanded_output: bool,
    reviewer_view: bool,
    thoughts_display: crate::command::ThoughtsDisplayMode,
    next_reasoning: bool,
) -> Document<Style> {
    let mut document = Document::default();
    TimelineItemComponent {
        item,
        theme,
        width,
        frame,
        expanded_output,
        reviewer_view,
        thoughts_display,
        next_reasoning,
    }
    .render(&mut document);
    document.finish();
    debug_assert!(document.validate());
    document
}

/// Compaction lifecycle: opening rule, streamed markdown body, then a closing
/// rule only after the compaction has committed.
fn build_compaction_block_lines(
    out: &mut TimelineDocument,
    summary: &str,
    streaming: bool,
    theme: Theme,
    width: usize,
) {
    let width = card_content_width(width);
    if width == 0 {
        return;
    }
    push_drawn_horizontal_rule(out, theme, width);

    if !summary.is_empty() {
        build_assistant_message_lines(out, summary, streaming, theme, width);
    }

    if !streaming {
        push_drawn_horizontal_rule(out, theme, width);
    }
}

fn card_content_width(width: usize) -> usize {
    width.saturating_sub(surface::CARD_PAD_RIGHT as usize)
}

/// Full-width drawn divider (box-drawing line), not a character label string.
fn push_drawn_horizontal_rule(out: &mut TimelineDocument, theme: Theme, width: usize) {
    if width == 0 {
        return;
    }
    out.push_decoration(
        Line::from(Span::styled("─".repeat(width), root_muted_style(theme))),
        Break::SoftWrap,
    );
}

fn build_reasoning_lines(
    out: &mut TimelineDocument,
    reasoning: &ReasoningView,
    theme: Theme,
    width: usize,
    display: crate::command::ThoughtsDisplayMode,
    next_reasoning: bool,
) {
    let content_width = width.saturating_sub(2).max(1);
    let (title, body) = reasoning_title_and_body(&reasoning.text);
    let title = title.unwrap_or_else(|| {
        if reasoning.streaming {
            "Thinking".to_string()
        } else {
            "Thought".to_string()
        }
    });
    let elapsed = format_reasoning_elapsed(reasoning);

    if display == crate::command::ThoughtsDisplayMode::Compact {
        if next_reasoning {
            return;
        }
        push_reasoning_status_line(out, reasoning.streaming, &elapsed, theme, width);
        push_wrapped_reasoning_title(out, &title, theme, width, Break::End);
        return;
    }

    let suffix = format!(" · {elapsed}");
    if 2 + "Thought: ".chars().count() + title.chars().count() + suffix.chars().count() <= width {
        push_reasoning_title_line(out, "Thought: ", &title, &suffix, theme, Break::HardBreak);
    } else {
        push_reasoning_status_line(out, reasoning.streaming, &elapsed, theme, width);
        push_wrapped_reasoning_title(out, &title, theme, width, Break::HardBreak);
    }
    if display == crate::command::ThoughtsDisplayMode::Titles {
        return;
    }

    let cleaned_body: String = {
        let mut s = String::new();
        let mut first = true;
        for raw in body.lines() {
            if !first {
                s.push('\n');
            }
            first = false;
            s.push_str(&clean_reasoning_line(raw));
        }
        s
    };
    let body_block = out.add_source(cleaned_body.clone());
    let chunks = wrap_text_to_width_with_offsets(&cleaned_body, content_width);

    let mut pushed = false;
    for (index, chunk) in chunks.iter().enumerate() {
        if chunk.source_start_char == chunk.source_end_char {
            continue;
        }
        pushed = true;
        out.push_content(
            "  ",
            theme.app_style(),
            chunk.text.clone(),
            reasoning_text_style(theme),
            SourceRange::new(body_block, chunk.source_start_char, chunk.source_end_char),
            "",
            reasoning_text_style(theme),
            chunk_boundary(&chunks, index),
        );
    }

    if !pushed && reasoning.streaming {
        out.push_decoration(
            Line::from(Span::styled("  …", root_dim_style(theme))),
            Break::End,
        );
    }
}

fn push_reasoning_status_line(
    out: &mut TimelineDocument,
    streaming: bool,
    elapsed: &str,
    theme: Theme,
    width: usize,
) {
    let label = if streaming { "Thinking…" } else { "Thought" };
    let status = format!("  {label} · {elapsed}");
    for line in wrap_text_to_width(&status, width) {
        out.push_decoration(
            Line::from(Span::styled(line, reasoning_label_style(theme))),
            Break::HardBreak,
        );
    }
}

fn push_reasoning_title_line(
    out: &mut TimelineDocument,
    label: &str,
    title: &str,
    suffix: &str,
    theme: Theme,
    boundary: Break,
) {
    let title_block = out.add_source(title.to_string());
    let title_len = title.chars().count();
    out.push_content(
        format!("  {label}"),
        reasoning_label_style(theme),
        title.to_string(),
        reasoning_label_style(theme),
        SourceRange::new(title_block, 0, title_len),
        suffix.to_string(),
        reasoning_label_style(theme),
        boundary,
    );
}

fn push_wrapped_reasoning_title(
    out: &mut TimelineDocument,
    title: &str,
    theme: Theme,
    width: usize,
    boundary: Break,
) {
    let title_block = out.add_source(title.to_string());
    let chunks = wrap_text_to_width_with_offsets(title, width.saturating_sub(2).max(1));
    let last_chunk = chunks.len().saturating_sub(1);

    for (index, chunk) in chunks.into_iter().enumerate() {
        if chunk.source_start_char == chunk.source_end_char {
            continue;
        }
        out.push_content(
            "  ",
            reasoning_label_style(theme),
            chunk.text,
            reasoning_label_style(theme),
            SourceRange::new(title_block, chunk.source_start_char, chunk.source_end_char),
            "",
            reasoning_label_style(theme),
            if index == last_chunk {
                boundary
            } else {
                Break::SoftWrap
            },
        );
    }
}

fn format_reasoning_elapsed(reasoning: &ReasoningView) -> String {
    let elapsed_ms = reasoning.duration_ms.or_else(|| {
        reasoning.started_at.map(|started_at| {
            u64::try_from(
                std::time::Instant::now()
                    .saturating_duration_since(started_at)
                    .as_millis(),
            )
            .unwrap_or(u64::MAX)
        })
    });
    elapsed_ms
        .map(format_elapsed_duration)
        .unwrap_or_else(|| "—".into())
}

fn format_elapsed_duration(elapsed_ms: u64) -> String {
    if elapsed_ms < 1_000 {
        format!("{elapsed_ms}ms")
    } else if elapsed_ms < 60_000 {
        format!("{}s", elapsed_ms / 1_000)
    } else {
        let total_seconds = elapsed_ms / 1_000;
        let minutes = total_seconds / 60;
        let seconds = total_seconds % 60;
        if seconds == 0 {
            format!("{minutes}m")
        } else {
            format!("{minutes}m {seconds}s")
        }
    }
}

fn build_delegation_lines(
    out: &mut TimelineDocument,
    delegation: &DelegationView,
    theme: Theme,
    width: usize,
) {
    let status = format!("delegate @{}", delegation.agent_name);
    let summary = one_line_snippet(&delegation.task, width.saturating_sub(20).max(1));
    let summary_len = summary.chars().count();
    let block = out.add_source(summary.clone());
    out.push_content(
        format!("  {status} · "),
        theme.user_style().add_modifier(Modifier::BOLD),
        summary,
        theme.app_style(),
        SourceRange::new(block, 0, summary_len),
        "",
        theme.app_style(),
        Break::End,
    );
}

fn reasoning_title_and_body(text: &str) -> (Option<String>, String) {
    let mut title = None;
    let mut body_lines = Vec::new();
    let mut consumed_title = false;

    for raw in text.lines() {
        if !consumed_title {
            let cleaned = clean_reasoning_line(raw);
            if cleaned.is_empty() {
                continue;
            }
            title = Some(
                wrap_text_to_width(&cleaned, 80)
                    .into_iter()
                    .next()
                    .unwrap_or(cleaned),
            );
            consumed_title = true;
            continue;
        }
        body_lines.push(raw);
    }

    while body_lines
        .first()
        .is_some_and(|line| line.trim().is_empty())
    {
        body_lines.remove(0);
    }
    (title, body_lines.join("\n"))
}

fn clean_reasoning_line(raw: &str) -> String {
    let mut text = raw.trim().to_string();
    while let Some(rest) = text.strip_prefix('#') {
        text = rest.trim_start().to_string();
    }
    for marker in ["**", "__", "`"] {
        text = text.replace(marker, "");
    }
    text
}

fn message_text(message: &MessageView) -> &str {
    if message.text.is_empty() && message.streaming {
        "…"
    } else {
        &message.text
    }
}

fn build_user_message(
    out: &mut TimelineDocument,
    message: &MessageView,
    theme: Theme,
    width: usize,
) {
    let text = message_text(message);
    let content_width = width.saturating_sub(5).max(1);

    // 整段 text 一次 wrap，得到每个视觉行对应原文的字符区间。
    let chunks = wrap_text_to_width_with_offsets(text, content_width);
    let block_index = out.add_source(text.to_string());

    // 顶部空 card 行（decoration）
    push_user_card_line_into(out, "", None, width, theme, None);

    // 内容行：每个 chunk 一行
    let mut pushed = false;
    for (index, chunk) in chunks.iter().enumerate() {
        pushed = true;
        let boundary = chunk_boundary(&chunks, index);
        push_user_card_content_line_into(
            out,
            vec![if chunk.source_start_char < chunk.source_end_char {
                (
                    RenderSpan::source(
                        chunk.text.clone(),
                        user_prompt_panel_style(theme),
                        SourceRange::new(
                            block_index,
                            chunk.source_start_char,
                            chunk.source_end_char,
                        ),
                    ),
                    true,
                )
            } else {
                (
                    RenderSpan::decoration("", user_prompt_panel_style(theme)),
                    false,
                )
            }],
            None,
            width,
            theme,
            boundary,
        );
    }
    if !pushed && message.attachments.is_empty() && message.selected_skills.is_empty() {
        push_user_card_content_line_into(
            out,
            vec![(
                RenderSpan::decoration("", user_prompt_panel_style(theme)),
                false,
            )],
            None,
            width,
            theme,
            Break::SoftWrap,
        );
    }

    if !message.selected_skills.is_empty() {
        push_user_card_line_into(out, "", Some("SKILLS"), width, theme, None);
    }

    for (index, name) in message.selected_skills.iter().enumerate() {
        let skill_text = transcript_skill_text(index, name, content_width);
        let skill_block = out.add_source(skill_text.clone());
        push_user_card_content_line_into(
            out,
            transcript_skill_item_render_spans(index, &skill_text, skill_block, theme),
            None,
            width,
            theme,
            Break::SoftWrap,
        );
    }

    if !message.attachments.is_empty() {
        push_user_card_line_into(out, "", Some("ATTACHMENTS"), width, theme, None);
    }

    for (index, attachment) in message.attachments.iter().enumerate() {
        let attachment_text = transcript_attachment_text(index, attachment, content_width);
        let attachment_block = out.add_source(attachment_text.clone());
        push_user_card_content_line_into(
            out,
            transcript_attachment_item_render_spans(
                index,
                &attachment_text,
                attachment_block,
                theme,
            ),
            None,
            width,
            theme,
            Break::SoftWrap,
        );
    }

    if message.queued {
        push_user_card_line_into(out, "", Some("QUEUED"), width, theme, None);
    }

    // 底部空 card 行（decoration）
    push_user_card_line_into(out, "", None, width, theme, None);
}

/// 与 `push_user_card_line` 等价的构造，并同时记录 Span 级来源。
/// `origin` 为 `None` 表示装饰行（顶部 spacer / QUEUED badge / 底部 spacer），
/// 否则为 `(block_index, content_prefix_chars, source_start, source_end)`。
fn push_user_card_line_into(
    out: &mut TimelineDocument,
    content: &str,
    badge: Option<&str>,
    width: usize,
    theme: Theme,
    origin: Option<(usize, usize, usize, usize)>,
) {
    let content = match origin {
        Some((block_index, _, start, end)) if start < end => (
            RenderSpan::source(
                content,
                user_prompt_panel_style(theme),
                SourceRange::new(block_index, start, end),
            ),
            true,
        ),
        _ => (
            RenderSpan::decoration(content, user_prompt_panel_style(theme)),
            false,
        ),
    };
    push_user_card_content_line_into(out, vec![content], badge, width, theme, Break::SoftWrap);
}

fn chunk_boundary(chunks: &[crate::tui::measure::WrappedChunk], index: usize) -> Break {
    let Some(current) = chunks.get(index) else {
        return Break::SoftWrap;
    };
    let Some(next) = chunks.get(index + 1) else {
        return Break::SoftWrap;
    };
    if next.source_start_char > current.source_end_char {
        Break::HardBreak
    } else {
        Break::SoftWrap
    }
}

fn push_user_card_content_line_into(
    out: &mut TimelineDocument,
    content_spans: Vec<(RenderSpan<ratatui::style::Style>, bool)>,
    badge: Option<&str>,
    width: usize,
    theme: Theme,
    boundary: Break,
) {
    let bar_style = surface::accent_style(
        theme,
        surface::SurfaceEmphasis::User,
        surface::SurfaceKind::Root,
    );
    let pad_style = user_prompt_padding_style(theme);
    let badge_style = queued_badge_style(theme);

    let mut spans = vec![
        RenderSpan::decoration(surface::ACCENT_BAR_GLYPH, bar_style),
        RenderSpan::decoration("  ", pad_style),
    ];

    if let Some(badge) = badge {
        spans.push(RenderSpan::decoration(" ", badge_style));
        spans.push(RenderSpan::decoration(badge, badge_style));
        spans.push(RenderSpan::decoration(" ", badge_style));
        spans.push(RenderSpan::decoration(" ", pad_style));
    }

    let has_source = content_spans.iter().any(|(_, source)| *source);
    spans.extend(content_spans.into_iter().map(|(span, _)| span));

    let used = spans.iter().map(|span| display_width(&span.text)).sum();
    if width > used {
        spans.push(RenderSpan::decoration(" ".repeat(width - used), pad_style));
    } else {
        spans.push(RenderSpan::decoration("  ", pad_style));
    }

    out.push_line(
        RenderLine { spans },
        if has_source {
            boundary
        } else {
            Break::SoftWrap
        },
    );
}

fn build_assistant_message_lines(
    out: &mut TimelineDocument,
    text: &str,
    streaming: bool,
    theme: Theme,
    width: usize,
) {
    if text.is_empty() {
        out.push_decoration(
            Line::from(Span::styled("  …", root_muted_style(theme))),
            Break::End,
        );
        return;
    }

    if let Some(result) = try_parse_structured_subagent_result(text) {
        out.document.append(
            structured_subagent::render_structured_subagent_result_document(
                &result,
                theme,
                card_content_width(width),
            ),
        );
        return;
    }

    if streaming && looks_like_structured_subagent_output(text) {
        out.push_decoration(
            Line::from(Span::styled("  …", root_muted_style(theme))),
            Break::End,
        );
        return;
    }

    let content_width = width.saturating_sub(2).max(1);
    let mut document =
        render_markdown_document(text, theme, MarkdownRenderOptions::new(content_width));
    for line in &mut document.lines {
        line.spans
            .insert(0, RenderSpan::decoration("  ", theme.app_style()));
    }
    out.document.append(document);
}

fn append_component_document(out: &mut TimelineDocument, document: Document<Style>) {
    out.document.append(document);
}

fn build_tool_lines(
    out: &mut TimelineDocument,
    tool: &ToolView,
    theme: Theme,
    width: usize,
    frame: usize,
    expanded_output: bool,
) {
    append_component_document(
        out,
        tool_card::render_tool_card_document(
            tool,
            theme,
            card_content_width(width),
            frame,
            expanded_output,
        ),
    );
}

fn build_todo_card(
    out: &mut TimelineDocument,
    todo: &crate::tui::timeline::TodoView,
    theme: Theme,
    width: usize,
) {
    append_component_document(
        out,
        todo_card::render_todo_card_document(todo, theme, card_content_width(width)),
    );
}

fn build_permission_lines(
    out: &mut TimelineDocument,
    permission: &PermissionView,
    theme: Theme,
    width: usize,
) {
    if permission.status != PermissionPromptStatus::Pending {
        append_component_document(
            out,
            tool_card::render_permission_card_document(
                permission,
                theme,
                card_content_width(width),
            ),
        );
    }
}

fn build_auto_review_lines(
    out: &mut TimelineDocument,
    decision: &crate::tui::timeline::AutoReviewDecisionView,
    theme: Theme,
    width: usize,
    expanded: bool,
) {
    let width = card_content_width(width);
    if width == 0 {
        return;
    }
    let status = if decision.allowed {
        "approved"
    } else {
        "denied"
    };
    let title = format!(
        "⛨ Auto-review {status} {} · {}{}",
        decision.tool_name,
        decision.approval,
        if expanded {
            " · collapse"
        } else {
            " · expand"
        },
    );
    let status_style = auto_review_status_style(decision.allowed, theme);
    for line in wrap_text_to_width(&title, width) {
        out.push_decoration(
            Line::from(Span::styled(line, status_style)),
            Break::HardBreak,
        );
    }
    if expanded {
        let risk = decision.risk.as_deref().unwrap_or("unknown");
        let detail = format!(
            "risk {risk} · {} · call {}",
            decision.rationale, decision.call_id,
        );
        push_auto_review_text(out, &detail, root_dim_style(theme), width, Break::HardBreak);
    }
}

fn auto_review_status_style(allowed: bool, theme: Theme) -> Style {
    Style::default()
        .fg(if allowed { theme.success } else { theme.error })
        .bg(theme.root_bg)
}

fn push_auto_review_text(
    out: &mut TimelineDocument,
    text: &str,
    style: Style,
    width: usize,
    final_boundary: Break,
) {
    let block = out.add_source(text);
    let chunks = wrap_text_to_width_with_offsets(text, width.max(1));
    for (index, chunk) in chunks.iter().enumerate() {
        if chunk.source_start_char < chunk.source_end_char {
            out.push_line(
                RenderLine {
                    spans: vec![RenderSpan::source(
                        chunk.text.clone(),
                        style,
                        SourceRange::new(block, chunk.source_start_char, chunk.source_end_char),
                    )],
                },
                if index + 1 < chunks.len() {
                    chunk_boundary(&chunks, index)
                } else {
                    final_boundary
                },
            );
        }
    }
}

fn build_error_lines(out: &mut TimelineDocument, error: &ErrorView, theme: Theme, width: usize) {
    let width = card_content_width(width);
    if width == 0 {
        return;
    }
    let bar = card_bar_style(theme.error, theme.root_bg);
    let fill = ratatui::style::Style::default().bg(theme.elevated_bg);
    push_error_decoration(out, width, bar, fill);
    push_error_content(
        out,
        &error.message,
        elevated_error_message_style(theme),
        bar,
        width,
    );
    if let Some(details) = error
        .details
        .as_deref()
        .filter(|details| !details.is_empty())
    {
        push_error_decoration(out, width, bar, fill);
        push_error_content(out, details, elevated_error_detail_style(theme), bar, width);
    }
    push_error_decoration(out, width, bar, fill);
}

fn push_error_decoration(
    out: &mut TimelineDocument,
    width: usize,
    bar: ratatui::style::Style,
    fill: ratatui::style::Style,
) {
    let mut spans = vec![RenderSpan::decoration(surface::ACCENT_BAR_GLYPH, bar)];
    if width > 1 {
        spans.push(RenderSpan::decoration(" ".repeat(width - 1), fill));
    }
    out.push_line(RenderLine { spans }, Break::SoftWrap);
}

fn push_error_content(
    out: &mut TimelineDocument,
    content: &str,
    value_style: ratatui::style::Style,
    bar: ratatui::style::Style,
    width: usize,
) {
    if width == 1 {
        out.push_line(
            RenderLine {
                spans: vec![RenderSpan::decoration(surface::ACCENT_BAR_GLYPH, bar)],
            },
            Break::SoftWrap,
        );
        return;
    }
    let has_padding = width > 2;
    let content_width = width.saturating_sub(1 + usize::from(has_padding)).max(1);
    let block = out.add_source(content);
    let chunks = wrap_text_to_width_with_offsets(content, content_width);
    for (index, chunk) in chunks.iter().enumerate() {
        let mut spans = vec![RenderSpan::decoration(surface::ACCENT_BAR_GLYPH, bar)];
        if has_padding {
            spans.push(RenderSpan::decoration(" ", value_style));
        }
        if chunk.source_start_char < chunk.source_end_char {
            spans.push(RenderSpan::source(
                chunk.text.clone(),
                value_style,
                SourceRange::new(block, chunk.source_start_char, chunk.source_end_char),
            ));
        }
        let used = spans.iter().map(|span| display_width(&span.text)).sum();
        if width > used {
            spans.push(RenderSpan::decoration(
                " ".repeat(width - used),
                value_style,
            ));
        }
        out.push_line(RenderLine { spans }, chunk_boundary(&chunks, index));
    }
}

fn card_bar_style(
    accent: ratatui::style::Color,
    bg: ratatui::style::Color,
) -> ratatui::style::Style {
    ratatui::style::Style::default().fg(accent).bg(bg)
}

fn transcript_skill_text(index: usize, name: &str, content_width: usize) -> String {
    let title = format!("skill {}", index + 1);
    let prefix_width = display_width(&format!("{title} · "));
    let detail = one_line_snippet(
        name.trim(),
        content_width.saturating_sub(prefix_width + 2).max(1),
    );

    if detail.is_empty() {
        title
    } else {
        format!("{title} · {detail}")
    }
}

fn transcript_skill_item_render_spans(
    index: usize,
    skill_text: &str,
    block_index: usize,
    theme: Theme,
) -> Vec<(RenderSpan<ratatui::style::Style>, bool)> {
    let title = format!("skill {}", index + 1);
    let detail = skill_text
        .strip_prefix(&title)
        .unwrap_or(skill_text)
        .trim_start()
        .trim_start_matches('·')
        .trim_start()
        .to_string();

    let title_len = title.chars().count();
    let mut spans = vec![
        (
            RenderSpan::decoration("↳ ", user_prompt_padding_style(theme)),
            false,
        ),
        (
            RenderSpan::source(
                title,
                attachment_block_title_style(theme),
                SourceRange::new(block_index, 0, title_len),
            ),
            true,
        ),
    ];
    if !detail.is_empty() {
        let detail_start = skill_text
            .chars()
            .count()
            .saturating_sub(detail.chars().count());
        spans.push((
            RenderSpan::decoration(" · ", user_prompt_padding_style(theme)),
            false,
        ));
        spans.push((
            RenderSpan::source_with_join(
                detail.clone(),
                attachment_block_item_style(theme),
                SourceRange::new(
                    block_index,
                    detail_start,
                    detail_start + detail.chars().count(),
                ),
                CopyJoin::Space,
            ),
            true,
        ));
    }
    spans
}

fn transcript_attachment_text(
    index: usize,
    attachment: &UserImageAttachment,
    content_width: usize,
) -> String {
    let title = format!("image {}", index + 1);
    let prefix_width = display_width(&format!("{title} · "));
    let detail = one_line_snippet(
        &attachment_source_label(attachment),
        content_width.saturating_sub(prefix_width + 2).max(1),
    );

    if detail.is_empty() {
        title
    } else {
        format!("{title} · {detail}")
    }
}

fn transcript_attachment_item_render_spans(
    index: usize,
    attachment_text: &str,
    block_index: usize,
    theme: Theme,
) -> Vec<(RenderSpan<ratatui::style::Style>, bool)> {
    let title = format!("image {}", index + 1);
    let detail = attachment_text
        .strip_prefix(&title)
        .unwrap_or(attachment_text)
        .trim_start()
        .trim_start_matches('·')
        .trim_start()
        .to_string();

    let title_len = title.chars().count();
    let mut spans = vec![
        (
            RenderSpan::decoration("↳ ", user_prompt_padding_style(theme)),
            false,
        ),
        (
            RenderSpan::source(
                title,
                attachment_block_title_style(theme),
                SourceRange::new(block_index, 0, title_len),
            ),
            true,
        ),
    ];
    if !detail.is_empty() {
        let detail_start = attachment_text
            .chars()
            .count()
            .saturating_sub(detail.chars().count());
        spans.push((
            RenderSpan::decoration(" · ", user_prompt_padding_style(theme)),
            false,
        ));
        spans.push((
            RenderSpan::source_with_join(
                detail.clone(),
                attachment_block_item_style(theme),
                SourceRange::new(
                    block_index,
                    detail_start,
                    detail_start + detail.chars().count(),
                ),
                CopyJoin::Space,
            ),
            true,
        ));
    }
    spans
}

fn attachment_source_label(attachment: &UserImageAttachment) -> String {
    let label = attachment.label.trim();
    if !label.is_empty() {
        return label.to_string();
    }

    let mime = attachment.mime.trim();
    if !mime.is_empty() {
        return mime.to_string();
    }

    "image".into()
}

fn user_prompt_panel_style(theme: Theme) -> ratatui::style::Style {
    ratatui::style::Style::default()
        .fg(theme.text)
        .bg(theme.element_bg)
}

fn attachment_block_title_style(theme: Theme) -> ratatui::style::Style {
    ratatui::style::Style::default()
        .fg(theme.user)
        .bg(theme.element_bg)
        .add_modifier(Modifier::BOLD)
}

fn attachment_block_item_style(theme: Theme) -> ratatui::style::Style {
    ratatui::style::Style::default()
        .fg(theme.text)
        .bg(theme.element_bg)
}

fn user_prompt_padding_style(theme: Theme) -> ratatui::style::Style {
    ratatui::style::Style::default()
        .fg(theme.muted_text)
        .bg(theme.element_bg)
}

fn queued_badge_style(theme: Theme) -> ratatui::style::Style {
    ratatui::style::Style::default()
        .fg(theme.root_bg)
        .bg(theme.user)
        .add_modifier(Modifier::BOLD)
}

fn elevated_error_message_style(theme: Theme) -> ratatui::style::Style {
    ratatui::style::Style::default()
        .fg(theme.error)
        .bg(theme.elevated_bg)
        .add_modifier(Modifier::BOLD)
}

fn elevated_error_detail_style(theme: Theme) -> ratatui::style::Style {
    ratatui::style::Style::default()
        .fg(theme.muted_text)
        .bg(theme.elevated_bg)
}

fn root_muted_style(theme: Theme) -> ratatui::style::Style {
    ratatui::style::Style::default()
        .fg(theme.muted_text)
        .bg(theme.root_bg)
}

fn root_dim_style(theme: Theme) -> ratatui::style::Style {
    ratatui::style::Style::default()
        .fg(theme.dim_text)
        .bg(theme.root_bg)
}

fn reasoning_label_style(theme: Theme) -> ratatui::style::Style {
    ratatui::style::Style::default()
        .fg(theme.user)
        .bg(theme.root_bg)
}

fn reasoning_text_style(theme: Theme) -> ratatui::style::Style {
    ratatui::style::Style::default()
        .fg(theme.muted_text)
        .bg(theme.root_bg)
}

/// 应用选择高亮到可见行
fn apply_selection_highlight(
    lines: &mut [Line<'static>],
    selection: &crate::tui::state::TextSelection,
    state: &crate::tui::state::TuiState,
    theme: Theme,
    scroll_offset: u16,
) {
    use ratatui::style::Style;

    let (start, end) = selection.normalize();
    let cache = &state.transcript_render_cache;

    // 计算选择范围的绝对行号
    if start.item_index >= cache.row_starts().len() || end.item_index >= cache.row_starts().len() {
        return;
    }

    let sel_start_row = cache.row_starts()[start.item_index] + start.rendered_line_offset;
    let sel_end_row = cache.row_starts()[end.item_index] + end.rendered_line_offset;

    // 选择高亮样式：亮色背景 + 深色文字。
    // 注意不要叠加 `Modifier::REVERSED`——它会在终端层面互换 fg/bg 显示，
    // 使本应做背景的 accent 反相到文字上、背景反而变成 root_bg（与正常背景同色），
    // 视觉上表现为"背景没变、文字变蓝"。
    let selection_style = Style::default().bg(theme.accent).fg(theme.root_bg);

    // 遍历可见行，应用高亮
    for (idx, line) in lines.iter_mut().enumerate() {
        let absolute_row = scroll_offset as usize + idx;

        if absolute_row < sel_start_row || absolute_row > sel_end_row {
            continue;
        }

        let item_index = match cache.row_starts().binary_search(&absolute_row) {
            Ok(idx) => idx,
            Err(idx) => idx.saturating_sub(1),
        };
        if item_index >= cache.entries().len() {
            continue;
        }
        let item_start_row = cache.row_starts()[item_index];
        let local_line_index = absolute_row.saturating_sub(item_start_row);
        let Some(document_line) = cache.entries()[item_index]
            .document
            .lines
            .get(local_line_index)
        else {
            continue;
        };
        if !document_line.spans.iter().any(|span| span.source.is_some()) {
            continue;
        }

        let visual_len = document_line
            .spans
            .iter()
            .map(|span| span.text.chars().count())
            .sum::<usize>();
        let (char_start, char_end) = if absolute_row == sel_start_row && absolute_row == sel_end_row
        {
            (start.char_offset, end.char_offset)
        } else if absolute_row == sel_start_row {
            (start.char_offset, visual_len)
        } else if absolute_row == sel_end_row {
            (0, end.char_offset)
        } else {
            (0, visual_len)
        };
        if char_start > char_end {
            continue;
        }

        *line = highlight_line_spans(
            line.clone(),
            document_line,
            char_start,
            char_end,
            selection_style,
        );
    }
}

/// 高亮 Line 中指定字符范围的 Spans
fn highlight_line_spans(
    line: Line<'static>,
    document_line: &RenderLine<Style>,
    char_start: usize,
    char_end: usize,
    selection_style: ratatui::style::Style,
) -> Line<'static> {
    use ratatui::text::Span;

    let mut new_spans = Vec::new();
    let mut current_offset = 0;

    for (span, document_span) in line.spans.into_iter().zip(&document_line.spans) {
        let span_len = span.content.chars().count();
        let span_end = current_offset + span_len;

        if document_span.source.is_none() || span_end <= char_start || current_offset > char_end {
            // Chrome is never selected or highlighted.
            new_spans.push(span);
        } else {
            let requested_start = char_start.saturating_sub(current_offset).min(span_len);
            let requested_end = char_end.saturating_sub(current_offset).min(span_len);
            let (hl_start, hl_end) =
                inclusive_grapheme_bounds(&span.content, requested_start, requested_end);
            if hl_start >= hl_end {
                new_spans.push(span);
            } else {
                let (prefix, highlighted, suffix) =
                    split_grapheme_span(&span.content, hl_start, hl_end);
                if !prefix.is_empty() {
                    new_spans.push(Span::styled(prefix, span.style));
                }
                new_spans.push(Span::styled(highlighted, span.style.patch(selection_style)));
                if !suffix.is_empty() {
                    new_spans.push(Span::styled(suffix, span.style));
                }
            }
        }

        current_offset = span_end;
    }

    Line::from(new_spans)
}

fn split_grapheme_span(text: &str, start: usize, end: usize) -> (String, String, String) {
    let mut char_offset = 0;
    let mut start_byte = 0;
    let mut end_byte = text.len();
    for (byte, grapheme) in unicode_segmentation::UnicodeSegmentation::grapheme_indices(text, true)
    {
        if char_offset == start {
            start_byte = byte;
        }
        char_offset += grapheme.chars().count();
        if char_offset == end {
            end_byte = byte + grapheme.len();
            break;
        }
    }
    (
        text[..start_byte].to_string(),
        text[start_byte..end_byte].to_string(),
        text[end_byte..].to_string(),
    )
}

#[cfg(test)]
mod tests {
    use super::{
        cached_transcript_row_count, render_timeline_item_document, render_transcript,
        transcript_lines, transcript_row_count, try_render_reviewer_view_item,
        visible_cached_transcript_lines, visible_transcript_lines,
    };
    use crate::{
        agent::{AutoContinueState, TodoItem, TodoStatus},
        tool::ToolResult,
        transcript::{TranscriptEvent, TranscriptRecord},
        tui::{
            AssistantDeltaEvent, ErrorEvent, PermissionRequestEvent, ReasoningDeltaEvent,
            ReasoningDoneEvent, SessionEvent, ToolFinishedEvent, ToolOutcome, ToolStartedEvent,
            UserMessageEvent,
            events::{AutoContinueChangedEvent, TodoSnapshotEvent},
            measure::display_width,
            state::{ContextDetailTarget, TuiState},
            theme::{Theme, ThemeName},
            timeline::{
                CompactionView, ErrorView, MessageRole, MessageView, PermissionPromptStatus,
                PermissionView, Timeline, TimelineItem, TodoView, ToolExecutionStatus, ToolView,
            },
        },
        user_content::{UserImageAttachment, UserMessageContent, UserMessageSubmission},
    };
    use ratatui::{Terminal, backend::TestBackend, layout::Rect};
    use serde_json::json;

    fn test_attachment(id: &str, label: &str) -> UserImageAttachment {
        UserImageAttachment {
            id: id.into(),
            label: label.into(),
            mime: "image/png".into(),
            data_url: "data:image/png;base64,AAAA".into(),
        }
    }

    #[allow(dead_code)]
    fn record(sequence: u64, event: TranscriptEvent) -> TranscriptRecord {
        TranscriptRecord {
            session_id: "s".into(),
            sequence,
            timestamp_ms: sequence as u128,
            context_branch_id: None,
            event,
        }
    }

    #[allow(dead_code)]
    fn context_records() -> Vec<TranscriptRecord> {
        vec![
            record(
                1,
                TranscriptEvent::ContextNodeLifecycle {
                    node_id: "root".into(),
                    status: crate::context_tree::ContextNodeStatus::Inactive,
                },
            ),
            record(
                2,
                TranscriptEvent::ContextNodeCreated {
                    node_id: "node-a".into(),
                    parent_node_id: Some("root".into()),
                    label: Some("Investigate".into()),
                    purpose: Some("Trace context flow".into()),
                    block_ref: None,
                    source_ref: Some(crate::context_tree::ContextSourceRef {
                        source_kind: "summary".into(),
                        source_id: Some("sum-1".into()),
                    }),
                },
            ),
            record(
                3,
                TranscriptEvent::ContextNodeLifecycle {
                    node_id: "node-a".into(),
                    status: crate::context_tree::ContextNodeStatus::Active,
                },
            ),
            record(
                4,
                TranscriptEvent::AssistantMessage {
                    content: "Captured note for summary source".into(),
                },
            ),
            record(
                5,
                TranscriptEvent::ContextViewOperationMetadata {
                    operation: "pin".into(),
                    node_id: None,
                    block_id: Some("block-seq-4-note".into()),
                    detail: None,
                },
            ),
            record(
                6,
                TranscriptEvent::AssistantMessage {
                    content: "Archived block note".into(),
                },
            ),
            record(
                7,
                TranscriptEvent::ContextViewOperationMetadata {
                    operation: "archive".into(),
                    node_id: None,
                    block_id: Some("block-seq-6-note".into()),
                    detail: None,
                },
            ),
            record(
                8,
                TranscriptEvent::ToolCallStarted {
                    call_id: "call-1".into(),
                    name: "shell__exec".into(),
                    args: json!({"command": "cargo test --bin letcode"}),
                },
            ),
            record(
                9,
                TranscriptEvent::ToolCallFinished {
                    call_id: "call-1".into(),
                    name: "shell__exec".into(),
                    ok: true,
                    output: ToolResult::ok(
                        "shell__exec",
                        json!({
                            "stdout": "x".repeat(5_000),
                            "stdout_truncated": false,
                            "stderr": "",
                            "stderr_truncated": false
                        }),
                    ),
                },
            ),
            record(
                10,
                TranscriptEvent::ContextSummaryArtifactMetadata {
                    node_id: "node-a".into(),
                    artifact_id: "sum-1".into(),
                    artifact_kind: "session_summary".into(),
                    version: Some(1),
                    summary: Some("Short retained summary".into()),
                    source_node_id: Some("node-a".into()),
                    source_block_id: Some("block-seq-4-note".into()),
                    source_start_sequence: Some(4),
                    source_end_sequence: Some(4),
                },
            ),
        ]
    }

    #[test]
    fn tool_permission_and_error_cards_wrap_to_target_width() {
        let mut state = TuiState::default();
        state.apply_event(SessionEvent::UserMessage(UserMessageEvent::new("seed")));

        let mut tool_started = ToolStartedEvent::new("call-tool", "shell__exec", "run");
        tool_started.arguments = Some("--really-long-arg ".repeat(20));
        state.apply_event(SessionEvent::ToolStarted(tool_started));
        let mut tool_finished =
            ToolFinishedEvent::new("call-tool", "shell__exec", "run", ToolOutcome::Failure);
        tool_finished.output = Some("output=".to_string() + &"x".repeat(200));
        state.apply_event(SessionEvent::ToolFinished(tool_finished));

        let mut request = PermissionRequestEvent::new("call-perm", "shell__exec", "needs approval");
        request.arguments = Some("arg ".repeat(60));
        request.rationale = Some("because ".repeat(80));
        state.apply_event(SessionEvent::PermissionRequested(request));

        let mut err = ErrorEvent::new("boom");
        err.details = Some("detail ".repeat(90));
        state.apply_event(SessionEvent::Error(err));

        let theme = Theme::dark();
        let width = 44usize;
        let lines = transcript_lines(&state, theme, width);

        // Ensure no generated line exceeds target width in display cells.
        for line in &lines {
            let w = crate::tui::measure::display_width(&line.to_string());
            assert!(
                w <= width,
                "line width {w} > {width}: {:?}",
                line.to_string()
            );
        }
    }

    #[test]
    fn auto_review_renders_lock_status_and_expanded_rationale() {
        let item = TimelineItem::AutoReview(crate::tui::timeline::AutoReviewDecisionView {
            call_id: "call-deny".into(),
            tool_name: "shell__exec".into(),
            approval: "deny".into(),
            risk: Some("high".into()),
            rationale: "unsafe command".into(),
            allowed: false,
        });
        let theme = Theme::dark();

        let collapsed = render_timeline_item_document(
            &item,
            theme,
            80,
            0,
            false,
            false,
            crate::command::ThoughtsDisplayMode::Full,
            false,
        );
        let expanded = render_timeline_item_document(
            &item,
            theme,
            80,
            0,
            true,
            false,
            crate::command::ThoughtsDisplayMode::Full,
            false,
        );
        let collapsed_text = collapsed
            .lines
            .iter()
            .flat_map(|line| line.spans.iter())
            .map(|span| span.text.as_str())
            .collect::<String>();
        let expanded_text = expanded
            .lines
            .iter()
            .flat_map(|line| line.spans.iter())
            .map(|span| span.text.as_str())
            .collect::<String>();

        assert!(collapsed_text.contains("⛨ Auto-review denied shell__exec · deny · expand"));
        assert!(!collapsed_text.contains("unsafe command"));
        assert!(expanded_text.contains("⛨ Auto-review denied shell__exec · deny · collapse"));
        assert!(expanded_text.contains("risk high · unsafe command · call call-deny"));
        assert!(expanded.validate());

        let narrow = render_timeline_item_document(
            &item,
            theme,
            24,
            0,
            true,
            false,
            crate::command::ThoughtsDisplayMode::Full,
            false,
        );
        assert!(narrow.validate());
        assert!(
            narrow.lines.iter().all(|line| {
                crate::tui::measure::display_width(
                    &line
                        .spans
                        .iter()
                        .map(|span| span.text.as_str())
                        .collect::<String>(),
                ) <= 24usize.saturating_sub(crate::tui::surface::CARD_PAD_RIGHT as usize)
            }),
            "{narrow:?}"
        );
    }

    #[test]
    fn transcript_cards_use_right_pad_without_narrowing_ordinary_assistant() {
        let theme = Theme::dark();
        let width = 32usize;
        let long_text = "x".repeat(width);
        let cards = [
            TimelineItem::Tool(ToolView {
                call_id: "call-tool".into(),
                name: "shell__exec".into(),
                summary: "run command".into(),
                arguments: Some("arg".into()),
                output: Some("output".into()),
                status: ToolExecutionStatus::Failed,
            }),
            TimelineItem::Todo(TodoView {
                items: vec![TodoItem {
                    id: "todo-1".into(),
                    content: long_text.clone(),
                    status: TodoStatus::Pending,
                }],
                auto_continue: AutoContinueState::default(),
            }),
            TimelineItem::Permission(PermissionView {
                call_id: "call-perm".into(),
                tool_name: "shell__exec".into(),
                summary: long_text.clone(),
                arguments: None,
                rationale: Some("because".into()),
                origin_label: None,
                can_allow_always: false,
                grant_summary: None,
                status: PermissionPromptStatus::Approved,
                resolution_reason: Some("approved".into()),
            }),
            TimelineItem::Error(ErrorView {
                message: long_text.clone(),
                details: Some("details".into()),
            }),
            TimelineItem::Compaction(CompactionView {
                summary: long_text.clone(),
                streaming: false,
            }),
            TimelineItem::Assistant(MessageView {
                id: None,
                submission_id: None,
                role: MessageRole::Assistant,
                text: json!({
                    "status": "completed",
                    "summary": long_text.clone(),
                    "findings": ["finding"],
                })
                .to_string(),
                attachments: Vec::new(),
                selected_skills: Vec::new(),
                streaming: false,
                queued: false,
            }),
        ];

        let max_card_width = width.saturating_sub(crate::tui::surface::CARD_PAD_RIGHT as usize);
        for item in cards {
            let lines =
                ratatui::text::Text::from(crate::tui::transcript_ratatui::document_to_ratatui(
                    &render_timeline_item_document(
                        &item,
                        theme,
                        width,
                        0,
                        false,
                        false,
                        crate::command::ThoughtsDisplayMode::Full,
                        false,
                    ),
                ));
            assert!(
                lines.lines.iter().all(|line| {
                    crate::tui::measure::display_width(&line.to_string()) <= max_card_width
                }),
                "card exceeded {max_card_width}: {lines:?}"
            );
        }

        let ordinary = TimelineItem::Assistant(MessageView {
            id: None,
            submission_id: None,
            role: MessageRole::Assistant,
            text: long_text,
            attachments: Vec::new(),
            selected_skills: Vec::new(),
            streaming: false,
            queued: false,
        });
        let ordinary_lines =
            crate::tui::transcript_ratatui::document_to_ratatui(&render_timeline_item_document(
                &ordinary,
                theme,
                width,
                0,
                false,
                false,
                crate::command::ThoughtsDisplayMode::Full,
                false,
            ));
        assert!(
            ordinary_lines
                .iter()
                .any(|line| { crate::tui::measure::display_width(&line.to_string()) == width })
        );

        let review_request = TimelineItem::User(MessageView {
            id: None,
            submission_id: None,
            role: MessageRole::User,
            text: "Approve or deny this tool permission request.\nTool: shell__exec\nClass: tool\nSummary: run command\ncan_allow_always: false".into(),
            attachments: Vec::new(),
            selected_skills: Vec::new(),
            streaming: false,
            queued: false,
        });
        let review_decision = TimelineItem::Assistant(MessageView {
            id: None,
            submission_id: None,
            role: MessageRole::Assistant,
            text: json!({"decision": "deny", "risk": "high", "rationale": "unsafe"}).to_string(),
            attachments: Vec::new(),
            selected_skills: Vec::new(),
            streaming: false,
            queued: false,
        });
        for item in [&review_request, &review_decision] {
            let document = try_render_reviewer_view_item(item, theme, width)
                .expect("reviewer card should parse");
            assert!(
                crate::tui::transcript_ratatui::document_to_ratatui(&document)
                    .iter()
                    .all(|line| crate::tui::measure::display_width(&line.to_string())
                        <= max_card_width)
            );
        }
    }

    #[test]
    fn error_card_stays_within_narrow_widths() {
        let mut state = TuiState::default();
        let mut error = ErrorEvent::new("stream stopped");
        error.details = Some("retry after backoff".into());
        state.apply_event(SessionEvent::Error(error));

        for width in 1..=4 {
            for line in transcript_lines(&state, Theme::dark(), width) {
                assert!(
                    crate::tui::measure::display_width(&line.to_string()) <= width,
                    "error card overflowed width {width}: {:?}",
                    line.to_string()
                );
            }
        }
    }

    #[test]
    fn committed_compaction_renders_durable_block_with_summary() {
        let mut state = TuiState::default();
        state.timeline.push_restored_compaction(
            "## Goal\n\nEarlier context was summarized here.\n\n- keep path `src/agent.rs`",
        );

        let lines = transcript_lines(&state, Theme::dark(), 80)
            .into_iter()
            .map(|line| line.to_string())
            .collect::<Vec<_>>();
        let nonempty_lines = lines
            .iter()
            .filter(|line| !line.is_empty())
            .map(String::as_str)
            .collect::<Vec<_>>();

        let rule = "─".repeat(80 - crate::tui::surface::CARD_PAD_RIGHT as usize);
        assert_eq!(
            nonempty_lines.iter().filter(|line| **line == rule).count(),
            2,
            "expected top and bottom drawn rules: {lines:?}"
        );
        assert!(
            nonempty_lines
                .iter()
                .any(|line| line.contains("Earlier context was summarized")),
            "{lines:?}"
        );
        // Markdown body should render (not raw ## markers as the only content).
        assert!(
            nonempty_lines.iter().any(|line| line.contains("Goal")),
            "expected markdown heading content: {lines:?}"
        );
    }

    #[test]
    fn visible_window_clips_transcript_rows_before_rendering() {
        let lines = (0..20)
            .map(|index| ratatui::text::Line::from(format!("row-{index}")))
            .collect::<Vec<_>>();

        let visible = visible_transcript_lines(&lines, 5, 12)
            .into_iter()
            .map(|line| line.to_string())
            .collect::<Vec<_>>();

        assert_eq!(
            visible,
            vec!["row-12", "row-13", "row-14", "row-15", "row-16"]
        );

        let bottom = visible_transcript_lines(&lines, 5, 18)
            .into_iter()
            .map(|line| line.to_string())
            .collect::<Vec<_>>();
        assert_eq!(bottom, vec!["row-18", "row-19"]);
    }

    #[test]
    fn compact_reasoning_cache_refreshes_when_adjacent_item_is_added() {
        let start = std::time::Instant::now();
        let mut state = TuiState::default();
        state.set_thoughts_display(crate::command::ThoughtsDisplayMode::Compact);
        state.apply_event(SessionEvent::ReasoningDelta(ReasoningDeltaEvent::at(
            "reasoning-1",
            "First title\nFirst body",
            start,
        )));
        state.apply_event(SessionEvent::ReasoningDone(ReasoningDoneEvent::at(
            "reasoning-1",
            "First title\nFirst body",
            start + std::time::Duration::from_millis(400),
        )));

        let theme = Theme::dark();
        let width = 80;
        cached_transcript_row_count(&mut state, theme, width);
        let before = visible_cached_transcript_lines(&mut state, theme, width, 8, 0)
            .into_iter()
            .map(|line| line.to_string())
            .collect::<Vec<_>>()
            .join("\n");
        assert!(before.contains("First title"), "{before}");

        state.apply_event(SessionEvent::ReasoningDelta(ReasoningDeltaEvent::at(
            "reasoning-2",
            "Second title\nSecond body",
            start + std::time::Duration::from_millis(500),
        )));
        cached_transcript_row_count(&mut state, theme, width);
        let after = visible_cached_transcript_lines(&mut state, theme, width, 8, 0)
            .into_iter()
            .map(|line| line.to_string())
            .collect::<Vec<_>>()
            .join("\n");

        assert!(!after.contains("First title"), "{after}");
        assert!(after.contains("Second title"), "{after}");
    }

    #[test]
    fn streaming_reasoning_status_wraps_at_narrow_width() {
        let mut state = TuiState::default();
        state.set_thoughts_display(crate::command::ThoughtsDisplayMode::Compact);
        state.apply_event(SessionEvent::ReasoningDelta(ReasoningDeltaEvent::new(
            "reasoning-1",
            "检查缓存状态",
        )));

        for width in [8, 12, 18] {
            let lines = transcript_lines(&state, Theme::dark(), width);
            assert!(
                lines
                    .iter()
                    .all(|line| display_width(&line.to_string()) <= width),
                "width {width}: {lines:?}"
            );
        }
    }

    #[test]
    fn reasoning_display_modes_and_narrow_width_render_cleanly() {
        let start = std::time::Instant::now();
        let mut state = TuiState::default();
        state.apply_event(SessionEvent::ReasoningDelta(ReasoningDeltaEvent::at(
            "reasoning-1",
            "Inspecting cache\nRead the cached document and compare adjacent items.",
            start,
        )));
        state.apply_event(SessionEvent::ReasoningDone(ReasoningDoneEvent::at(
            "reasoning-1",
            "Inspecting cache\nRead the cached document and compare adjacent items.",
            start + std::time::Duration::from_millis(1_250),
        )));

        for mode in [
            crate::command::ThoughtsDisplayMode::Compact,
            crate::command::ThoughtsDisplayMode::Titles,
            crate::command::ThoughtsDisplayMode::Full,
        ] {
            state.set_thoughts_display(mode);
            let lines = transcript_lines(&state, Theme::dark(), 18);
            assert!(
                lines
                    .iter()
                    .all(|line| display_width(&line.to_string()) <= 18)
            );
            let text = lines
                .iter()
                .map(ToString::to_string)
                .collect::<Vec<_>>()
                .join("\n");
            assert!(text.contains("Inspecting cache"), "{mode:?}: {text}");
            assert_eq!(
                text.contains("Read the cached"),
                mode == crate::command::ThoughtsDisplayMode::Full,
                "{mode:?}: {text}"
            );
        }
    }

    #[test]
    fn consecutive_reasoning_titles_are_separated_outside_compact_mode() {
        let start = std::time::Instant::now();
        let mut state = TuiState::default();
        for (index, title) in ["First thought", "Second thought", "Third thought"]
            .into_iter()
            .enumerate()
        {
            let observed_at = start + std::time::Duration::from_millis(index as u64 * 100);
            let item_id = format!("reasoning-{index}");
            state.apply_event(SessionEvent::ReasoningDelta(ReasoningDeltaEvent::at(
                &item_id,
                title,
                observed_at,
            )));
            state.apply_event(SessionEvent::ReasoningDone(ReasoningDoneEvent::at(
                item_id,
                title,
                observed_at + std::time::Duration::from_millis(50),
            )));
        }

        for mode in [
            crate::command::ThoughtsDisplayMode::Titles,
            crate::command::ThoughtsDisplayMode::Full,
        ] {
            state.set_thoughts_display(mode);
            let lines = transcript_lines(&state, Theme::dark(), 80)
                .into_iter()
                .map(|line| line.to_string())
                .collect::<Vec<_>>();
            let title_rows = lines
                .iter()
                .enumerate()
                .filter_map(|(index, line)| line.contains("Thought:").then_some(index))
                .collect::<Vec<_>>();

            assert_eq!(title_rows.len(), 3, "{mode:?}: {lines:?}");
            assert!(
                title_rows
                    .windows(2)
                    .all(|rows| lines[rows[0] + 1..rows[1]].iter().any(String::is_empty)),
                "{mode:?}: {lines:?}"
            );

            let total_rows = cached_transcript_row_count(&mut state, Theme::dark(), 80);
            let cached = visible_cached_transcript_lines(
                &mut state,
                Theme::dark(),
                80,
                u16::try_from(total_rows).expect("test transcript fits in u16 rows"),
                0,
            )
            .into_iter()
            .map(|line| line.to_string())
            .collect::<Vec<_>>();
            assert_eq!(cached, lines, "{mode:?}");
        }

        state.set_thoughts_display(crate::command::ThoughtsDisplayMode::Compact);
        let compact = transcript_lines(&state, Theme::dark(), 80)
            .into_iter()
            .map(|line| line.to_string())
            .collect::<Vec<_>>()
            .join("\n");
        assert!(!compact.contains("First thought"), "{compact}");
        assert!(!compact.contains("Second thought"), "{compact}");
        assert!(compact.contains("Third thought"), "{compact}");
    }

    #[test]
    fn cached_visible_transcript_matches_full_transcript_window() {
        let mut state = TuiState::default();
        state.apply_event(SessionEvent::UserMessage(UserMessageEvent::new("seed")));
        for index in 0..30 {
            state
                .timeline
                .push_assistant_delta(AssistantDeltaEvent::new(format!("history line {index}")));
        }
        state.apply_event(SessionEvent::AssistantDelta(AssistantDeltaEvent::new(
            "# Heading\n```rust\nlet value = 42;\n```\n- done",
        )));
        state.apply_event(SessionEvent::AssistantDone { message_id: None });

        let theme = Theme::dark();
        let width = 72;
        let full = transcript_lines(&state, theme, width);
        let total_rows = cached_transcript_row_count(&mut state, theme, width);
        let visible = visible_cached_transcript_lines(&mut state, theme, width, 9, 12)
            .into_iter()
            .map(|line| line.to_string())
            .collect::<Vec<_>>();
        let expected = visible_transcript_lines(&full, 9, 12)
            .into_iter()
            .map(|line| line.to_string())
            .collect::<Vec<_>>();

        assert_eq!(total_rows, full.len());
        assert_eq!(visible, expected);
    }

    #[test]
    fn cached_row_count_uses_stable_timeline_fast_path() {
        let mut state = TuiState::default();
        state.apply_event(SessionEvent::UserMessage(UserMessageEvent::new("first")));
        state.apply_event(SessionEvent::UserMessage(UserMessageEvent::new("second")));

        let theme = Theme::dark();
        let width = 80;
        let expected_rows = cached_transcript_row_count(&mut state, theme, width);
        let rebuilds = state.transcript_render_cache.row_count_rebuilds;

        assert_eq!(
            cached_transcript_row_count(&mut state, theme, width),
            expected_rows
        );
        assert_eq!(state.transcript_render_cache.row_count_rebuilds, rebuilds);
    }

    #[test]
    fn cached_row_count_refreshes_after_timeline_mutation() {
        let mut state = TuiState::default();
        state.apply_event(SessionEvent::AssistantDelta(AssistantDeltaEvent::new(
            "first",
        )));

        let theme = Theme::dark();
        let width = 80;
        let before_rows = cached_transcript_row_count(&mut state, theme, width);
        let before_rebuilds = state.transcript_render_cache.row_count_rebuilds;

        state.apply_event(SessionEvent::AssistantDelta(AssistantDeltaEvent::new(
            " second",
        )));

        let after_rows = cached_transcript_row_count(&mut state, theme, width);
        assert_eq!(
            state.transcript_render_cache.row_count_rebuilds,
            before_rebuilds + 1
        );
        assert!(after_rows >= before_rows);
        assert_eq!(after_rows, transcript_lines(&state, theme, width).len());
    }

    #[test]
    fn transcript_cache_invalidates_streaming_assistant_item() {
        let mut state = TuiState::default();
        state.apply_event(SessionEvent::AssistantDelta(AssistantDeltaEvent::new(
            "first",
        )));

        let theme = Theme::dark();
        let width = 80;
        let before_rows = cached_transcript_row_count(&mut state, theme, width);
        let before_revision = state.transcript_render_cache.entries[0].revision;

        state.apply_event(SessionEvent::AssistantDelta(AssistantDeltaEvent::new(
            " second",
        )));

        let after_rows = cached_transcript_row_count(&mut state, theme, width);
        let after_revision = state.transcript_render_cache.entries[0].revision;
        let visible = visible_cached_transcript_lines(&mut state, theme, width, 8, 0)
            .into_iter()
            .map(|line| line.to_string())
            .collect::<Vec<_>>()
            .join("\n");

        assert_eq!(before_rows, after_rows);
        assert_ne!(before_revision, after_revision);
        assert!(visible.contains("first second"), "{visible}");
    }

    #[test]
    fn transcript_cache_is_namespaced_by_timeline_replacement() {
        let mut state = TuiState::default();
        state
            .timeline
            .push_assistant_delta(AssistantDeltaEvent::new("old timeline"));
        let theme = Theme::dark();
        let width = 80;

        cached_transcript_row_count(&mut state, theme, width);
        assert!(
            visible_cached_transcript_lines(&mut state, theme, width, 8, 0)
                .into_iter()
                .any(|line| line.to_string().contains("old timeline"))
        );

        state.timeline = Timeline::new();
        state
            .timeline
            .push_assistant_delta(AssistantDeltaEvent::new("new timeline"));

        let visible = visible_cached_transcript_lines(&mut state, theme, width, 8, 0)
            .into_iter()
            .map(|line| line.to_string())
            .collect::<Vec<_>>()
            .join("\n");

        assert!(visible.contains("new timeline"), "{visible}");
        assert!(!visible.contains("old timeline"), "{visible}");
    }

    #[test]
    fn manual_history_view_stays_anchored_when_streaming_rows_append() {
        let theme = Theme::dark();
        let width = 80;
        let viewport_rows = 5;
        let mut state = TuiState::default();

        for index in 0..24 {
            state
                .timeline
                .push_assistant_delta(AssistantDeltaEvent::new(format!("history line {index}")));
        }

        let before_lines = transcript_lines(&state, theme, width);
        state.sync_transcript_viewport_rows(before_lines.len());
        let target_top = 6usize;
        let before_max_scroll = crate::tui::measure::max_scroll(before_lines.len(), viewport_rows);
        state.transcript_scroll = before_max_scroll.saturating_sub(target_top as u16);
        state.auto_scroll = false;

        let before_top = crate::tui::measure::resolved_scroll_offset(
            before_lines.len(),
            viewport_rows,
            state.transcript_scroll,
            state.auto_scroll,
        );
        let before_visible = visible_transcript_lines(&before_lines, viewport_rows, before_top);
        let before_first = before_visible
            .first()
            .map(|line| line.to_string())
            .expect("visible row before append");

        state.apply_event(SessionEvent::AssistantDelta(AssistantDeltaEvent::new(
            "streaming row one\nstreaming row two\nstreaming row three",
        )));

        let after_lines = transcript_lines(&state, theme, width);
        state.sync_transcript_viewport_rows(after_lines.len());
        let after_top = crate::tui::measure::resolved_scroll_offset(
            after_lines.len(),
            viewport_rows,
            state.transcript_scroll,
            state.auto_scroll,
        );
        let after_visible = visible_transcript_lines(&after_lines, viewport_rows, after_top);
        let after_first = after_visible
            .first()
            .map(|line| line.to_string())
            .expect("visible row after append");

        assert_eq!(after_top, before_top);
        assert_eq!(after_first, before_first);
        assert!(!state.auto_scroll);
    }

    #[test]
    fn pending_permission_is_hidden_from_transcript_while_panel_is_active() {
        let mut state = TuiState::default();
        let mut request = PermissionRequestEvent::new("call-perm", "shell__exec", "cargo test all");
        request.arguments = Some("cargo test all".into());
        request.rationale = Some("tests need confirmation".into());
        state.apply_event(SessionEvent::PermissionRequested(request));

        let lines = transcript_lines(&state, Theme::dark(), 60)
            .into_iter()
            .map(|line| line.to_string())
            .collect::<Vec<_>>();

        assert!(
            !lines
                .iter()
                .any(|line| line.contains("Permission required")),
            "{lines:?}"
        );
        assert!(
            !lines.iter().any(|line| line.contains("cargo test all")),
            "{lines:?}"
        );
    }

    #[test]
    fn subagent_parent_transcript_stays_compact_and_keeps_child_details_out() {
        let mut state = TuiState::default();
        state.apply_event(SessionEvent::ToolStarted(ToolStartedEvent {
            call_id: "run-1".into(),
            name: "agent__explore".into(),
            summary: "explorer running · child-sessio".into(),
            arguments: Some(serde_json::json!({"task":"inspect runner flow"}).to_string()),
        }));
        state.apply_event(SessionEvent::ToolFinished(ToolFinishedEvent {
            call_id: "run-1".into(),
            name: "agent__explore".into(),
            summary: "explorer completed · child-sessio".into(),
            outcome: ToolOutcome::Success,
            output: Some(
                serde_json::json!({
                    "ok": true,
                    "tool": "agent__explore",
                    "data": {
                        "agent_name": "Explorer",
                        "status": "completed",
                        "summary": "inspected runner flow and isolated parent timeline noise",
                        "full_summary": "inspected runner flow and isolated parent timeline noise\nlong child body line should stay in child view",
                        "child_session_id": "child-session-1234567890"
                    }
                })
                .to_string(),
            ),
        }));

        let lines = transcript_lines(&state, Theme::dark(), 120)
            .into_iter()
            .map(|line| line.to_string())
            .collect::<Vec<_>>();

        let non_empty = lines
            .into_iter()
            .filter(|line| !line.trim().is_empty())
            .collect::<Vec<_>>();
        let agent_lines = non_empty
            .iter()
            .filter(|line| line.contains("Explorer") || line.contains("/child"))
            .collect::<Vec<_>>();

        assert_eq!(agent_lines.len(), 1, "{non_empty:?}");
        assert!(
            agent_lines[0].contains("completed Explorer inspected runner flow and isolated parent timeline noise · /child"),
            "{}",
            agent_lines[0]
        );
        assert!(
            !non_empty
                .iter()
                .any(|line| line.contains("long child body line should stay in child view")),
            "{non_empty:?}"
        );
    }

    #[test]
    fn structured_subagent_json_renders_as_compact_card_at_narrow_width() {
        let mut state = TuiState::default();
        state.apply_event(SessionEvent::AssistantDelta(AssistantDeltaEvent::new(
            json!({
                "status": "completed",
                "summary": "Updated transcript rendering",
                "findings": ["Structured results no longer expose raw JSON"],
                "files_read": ["src/tui/components/transcript.rs"],
                "files_changed": ["src/tui/components/transcript.rs"],
                "commands_run": ["cargo test transcript"],
                "validation": ["focused tests passed"],
                "blockers": [],
                "next_steps": ["review the card"],
            })
            .to_string(),
        )));
        state.apply_event(SessionEvent::AssistantDone { message_id: None });

        let lines = transcript_lines(&state, Theme::dark(), 24)
            .into_iter()
            .map(|line| line.to_string())
            .collect::<Vec<_>>();
        let rendered = lines.join("\n");

        assert!(rendered.contains("Subagent"), "{rendered}");
        assert!(rendered.contains("completed"), "{rendered}");
        assert!(rendered.contains("Updated transcript"), "{rendered}");
        assert!(rendered.contains("read"), "{rendered}");
        assert!(rendered.contains("Findings"), "{rendered}");
        assert!(!rendered.contains("\"status\""), "{rendered}");
        assert!(!rendered.contains('{'), "{rendered}");
        assert!(
            lines.iter().all(|line| line.chars().count() <= 24),
            "{lines:?}"
        );
    }

    #[test]
    fn structured_subagent_json_accepts_object_summary_and_findings() {
        let mut state = TuiState::default();
        state.apply_event(SessionEvent::AssistantDelta(AssistantDeltaEvent::new(
            json!({
                "status": "completed",
                "summary": {"conclusion": "Reviewed the child result"},
                "findings": [{"evidence": "The child returned structured evidence"}],
                "files_read": [],
            })
            .to_string(),
        )));
        state.apply_event(SessionEvent::AssistantDone { message_id: None });

        let lines = transcript_lines(&state, Theme::dark(), 32)
            .into_iter()
            .map(|line| line.to_string())
            .collect::<Vec<_>>();
        let rendered = lines.join("\n");

        assert!(rendered.contains("Subagent"), "{rendered}");
        assert!(rendered.contains("completed"), "{rendered}");
        assert!(rendered.contains("Reviewed the child"), "{rendered}");
        assert!(rendered.contains("Findings"), "{rendered}");
        assert!(!rendered.contains("\"conclusion\""), "{rendered}");
        assert!(!rendered.contains('{'), "{rendered}");
        assert!(
            lines.iter().all(|line| line.chars().count() <= 32),
            "{lines:?}"
        );
    }

    #[test]
    fn non_structured_assistant_json_uses_markdown_fallback() {
        let mut state = TuiState::default();
        state.apply_event(SessionEvent::AssistantDelta(AssistantDeltaEvent::new(
            r#"{"message":"ordinary JSON"}"#,
        )));
        state.apply_event(SessionEvent::AssistantDone { message_id: None });

        let rendered = transcript_lines(&state, Theme::dark(), 80)
            .into_iter()
            .map(|line| line.to_string())
            .collect::<Vec<_>>()
            .join("\n");

        assert!(rendered.contains("message"), "{rendered}");
        assert!(rendered.contains("ordinary JSON"), "{rendered}");
        assert!(!rendered.contains("Subagent"), "{rendered}");
    }

    #[test]
    fn streaming_compaction_renders_opening_rule_then_preview_body() {
        let mut state = TuiState::default();
        state.apply_event(SessionEvent::CompactionStarted);

        let rule = "─".repeat(80 - crate::tui::surface::CARD_PAD_RIGHT as usize);
        let started = transcript_lines(&state, Theme::dark(), 80)
            .into_iter()
            .map(|line| line.to_string())
            .collect::<Vec<_>>();
        assert_eq!(started.iter().filter(|line| *line == &rule).count(), 1);
        assert!(!started.iter().any(|line| line.contains('…')));

        state.apply_event(SessionEvent::CompactionPreviewDelta {
            delta: "A transient summary preview".into(),
        });
        let streaming = transcript_lines(&state, Theme::dark(), 80)
            .into_iter()
            .map(|line| line.to_string())
            .collect::<Vec<_>>();
        let rule_index = streaming
            .iter()
            .position(|line| line == &rule)
            .expect("opening rule");
        let preview_index = streaming
            .iter()
            .position(|line| line.contains("A transient summary preview"))
            .expect("preview body");
        assert_eq!(streaming.iter().filter(|line| *line == &rule).count(), 1);
        assert!(rule_index < preview_index);
        assert!(!streaming.iter().any(|line| line.contains('…')));

        state.apply_event(SessionEvent::CompactionCommitted {
            summary: Some("A transient summary preview".into()),
        });
        let committed = transcript_lines(&state, Theme::dark(), 80)
            .into_iter()
            .map(|line| line.to_string())
            .collect::<Vec<_>>();
        assert_eq!(committed.iter().filter(|line| *line == &rule).count(), 2);
        let last_rule = committed
            .iter()
            .rposition(|line| line == &rule)
            .expect("closing rule");
        let body = committed
            .iter()
            .position(|line| line.contains("A transient summary preview"))
            .expect("committed body");
        assert!(body < last_rule);

        let narrow = transcript_lines(&state, Theme::dark(), 10)
            .into_iter()
            .map(|line| line.to_string())
            .collect::<Vec<_>>();
        assert!(narrow.iter().all(|line| line.chars().count() <= 10));
    }

    #[test]
    fn assistant_reasoning_and_tool_trace_share_left_indent() {
        let mut state = TuiState::default();
        state.apply_event(SessionEvent::ReasoningDone(ReasoningDoneEvent::new(
            "r-1",
            "Answering\nThinking body",
        )));
        state.apply_event(SessionEvent::AssistantDelta(AssistantDeltaEvent::new(
            "# Title",
        )));
        state.apply_event(SessionEvent::AssistantDone { message_id: None });
        state.apply_event(SessionEvent::ToolStarted(ToolStartedEvent::new(
            "call-list",
            "fs__list",
            "List src",
        )));

        let lines = transcript_lines(&state, Theme::dark(), 80)
            .into_iter()
            .map(|line| line.to_string())
            .collect::<Vec<_>>();
        let reasoning = lines
            .iter()
            .find(|line| line.contains("Thought: Answering"))
            .expect("reasoning line renders");
        let markdown = lines
            .iter()
            .find(|line| line.contains("▌ Title"))
            .expect("assistant markdown line renders");
        let tool = lines
            .iter()
            .find(|line| line.contains("List src"))
            .expect("tool trace line renders");

        for line in [reasoning, markdown, tool] {
            assert!(line.starts_with("  "), "{line:?}");
            assert!(!line.starts_with("   "), "{line:?}");
        }
    }

    #[test]
    fn user_card_bar_is_separated_from_card_background() {
        let mut state = TuiState::default();
        state.apply_event(SessionEvent::UserMessage(UserMessageEvent::new("hello")));

        let backend = TestBackend::new(40, 8);
        let mut terminal = Terminal::new(backend).expect("terminal");
        terminal
            .draw(|frame| {
                let area = Rect::new(0, 0, 40, 8);
                render_transcript(frame, &mut state, area, Theme::dark());
            })
            .expect("draw");

        let buffer = terminal.backend().buffer();
        let bar_cell = &buffer[(0, 2)];
        let card_cell = &buffer[(1, 2)];

        assert_eq!(bar_cell.bg, Theme::dark().root_bg);
        assert_eq!(card_cell.bg, Theme::dark().element_bg);
    }

    #[test]
    fn queued_user_message_renders_badge() {
        let mut state = TuiState::default();
        state.apply_event(SessionEvent::UserMessage(UserMessageEvent::queued(
            "follow up",
        )));

        let lines = transcript_lines(&state, Theme::dark(), 40)
            .into_iter()
            .map(|line| line.to_string())
            .collect::<Vec<_>>();

        assert!(
            lines.iter().any(|line| line.contains("QUEUED")),
            "{lines:?}"
        );
    }

    #[test]
    fn user_message_renders_inline_image_placeholder_and_original_attachment_row() {
        let mut state = TuiState::default();
        state.apply_event(SessionEvent::UserMessage(
            UserMessageEvent::from_submission(UserMessageSubmission::new(
                "user-with-inline-image",
                UserMessageContent::from_parts(vec![
                    crate::user_content::UserMessagePart::Text {
                        text: "[Image 1]".into(),
                    },
                    crate::user_content::UserMessagePart::Image {
                        attachment: test_attachment("img-1", "clipboard"),
                    },
                    crate::user_content::UserMessagePart::Text {
                        text: " 测试消息".into(),
                    },
                ]),
            )),
        ));

        let lines = transcript_lines(&state, Theme::dark(), 60)
            .into_iter()
            .map(|line| line.to_string())
            .collect::<Vec<_>>();

        let body_index = lines
            .iter()
            .position(|line| line.contains("[Image 1] 测试消息"))
            .expect("inline image placeholder in message body");
        let attachment_index = lines
            .iter()
            .position(|line| line.contains("ATTACHMENTS"))
            .expect("attachment badge line");

        assert!(body_index < attachment_index, "{lines:?}");
        assert!(
            lines
                .iter()
                .any(|line| line.contains("image 1") && line.contains("clipboard")),
            "{lines:?}"
        );
    }

    #[test]
    fn user_message_renders_selected_skills_beneath_body() {
        let mut state = TuiState::default();
        state.apply_event(SessionEvent::UserMessage(
            UserMessageEvent::from_submission(UserMessageSubmission::new(
                "user-with-skill",
                UserMessageContent::from("use this skill")
                    .with_selected_skills(vec!["rust-audit".into()]),
            )),
        ));

        let lines = transcript_lines(&state, Theme::dark(), 60)
            .into_iter()
            .map(|line| line.to_string())
            .collect::<Vec<_>>();

        let body_index = lines
            .iter()
            .position(|line| line.contains("use this skill"))
            .expect("message body line");
        let skills_index = lines
            .iter()
            .position(|line| line.contains("SKILLS"))
            .expect("skills badge line");

        assert!(skills_index > body_index, "{lines:?}");
        assert!(
            lines
                .iter()
                .any(|line| line.contains("skill 1") && line.contains("rust-audit")),
            "{lines:?}"
        );
    }

    #[test]
    fn user_message_renders_attachment_placeholders_beneath_body() {
        let mut state = TuiState::default();
        state.apply_event(SessionEvent::UserMessage(
            UserMessageEvent::from_submission(UserMessageSubmission::new(
                "user-with-images",
                UserMessageContent::new(
                    "describe this",
                    vec![
                        test_attachment("img-1", "clipboard"),
                        test_attachment("img-2", "diagram.png"),
                    ],
                ),
            )),
        ));

        let lines = transcript_lines(&state, Theme::dark(), 60)
            .into_iter()
            .map(|line| line.to_string())
            .collect::<Vec<_>>();

        let body_index = lines
            .iter()
            .position(|line| line.contains("describe this"))
            .expect("message body line");
        let attachment_index = lines
            .iter()
            .position(|line| line.contains("ATTACHMENTS"))
            .expect("attachment badge line");
        let queued_index = lines.iter().position(|line| line.contains("QUEUED"));

        assert!(attachment_index > body_index, "{lines:?}");
        assert!(queued_index.is_none(), "{lines:?}");
        assert!(
            lines
                .iter()
                .any(|line| line.contains("image 1") && line.contains("clipboard")),
            "{lines:?}"
        );
        assert!(
            lines
                .iter()
                .any(|line| line.contains("image 2") && line.contains("diagram.png")),
            "{lines:?}"
        );
    }

    #[test]
    fn tool_card_then_todo_card_keeps_timeline_separator() {
        let mut state = TuiState::default();
        state.apply_event(SessionEvent::ToolStarted(ToolStartedEvent {
            call_id: "call-shell".into(),
            name: "shell__exec".into(),
            summary: "run".into(),
            arguments: Some(json!({"command": "cargo test"}).to_string()),
        }));
        state.apply_event(SessionEvent::ToolFinished(ToolFinishedEvent {
            call_id: "call-shell".into(),
            name: "shell__exec".into(),
            summary: "exit 0 · stdout 3 lines".into(),
            outcome: ToolOutcome::Success,
            output: Some(
                json!({
                    "ok": true,
                    "tool": "shell__exec",
                    "data": {
                        "status": 0,
                        "success": true,
                        "stdout": "line1\nline2\nline3\n",
                        "stdout_truncated": false,
                        "stderr": "",
                        "stderr_truncated": false
                    }
                })
                .to_string(),
            ),
        }));
        state.apply_event(SessionEvent::TodoSnapshot(TodoSnapshotEvent::new(vec![
            TodoItem {
                id: "t1".into(),
                content: "Do something".into(),
                status: TodoStatus::InProgress,
            },
        ])));

        let lines = transcript_lines(&state, Theme::dark(), 80)
            .into_iter()
            .map(|line| line.to_string())
            .collect::<Vec<_>>();

        let shell_end = lines
            .iter()
            .position(|line| line.contains("line3"))
            .expect("shell output line");
        let todo_title = lines
            .iter()
            .position(|line| line.contains("# Todos"))
            .expect("todo card title");
        let between = &lines[shell_end + 1..todo_title];
        assert!(
            between.iter().any(|line| line.is_empty()),
            "expected a blank timeline separator between tool output and todo card: {lines:?}"
        );
    }
}
