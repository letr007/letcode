//! Render the active `Timeline` into the terminal frame.
//!
//! The terminal-facing transcript component: takes the already model-ized
//! `Timeline` plus markdown/theme, and draws the transcript viewport to the
//! `Frame`. Layout primitives come from `transcript_render` via the
//! `transcript_ratatui` bridge; the model itself comes from
//! `transcript_read_model`.

use unicode_segmentation::UnicodeSegmentation;

use ratatui::{
    Frame,
    layout::{Constraint, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span, Text},
    widgets::{Block, Paragraph},
};

use crate::subagent::try_parse_structured_subagent_result;
use crate::tui::{
    assistant_fade::{self, AssistantFade},
    i18n::Translator,
    markdown::{MarkdownRenderOptions, StreamingMarkdownRenderer, render_markdown_document},
    measure::{display_width, wrap_text_to_width, wrap_text_to_width_with_offsets},
    scrollbar::ScrollbarGeometry,
    surface,
    theme::Theme,
    timeline::{
        ErrorView, MessageView, PermissionPromptStatus, PermissionView, ReasoningView,
        TimelineItem, ToolExecutionStatus, ToolView,
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
    composer::one_line_snippet, delegation_card, historian_report, reviewer_cards,
    structured_subagent, todo_card, tool_card,
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
    live_render_frames: Vec<Option<usize>>,
    #[cfg(test)]
    row_count_rebuilds: usize,
    /// Characters of assistant text handed to the markdown renderer.
    #[cfg(test)]
    assistant_markdown_chars: usize,
    streaming_markdown: Option<StreamingMarkdownSlot>,
}

#[derive(Debug, Clone)]
struct StreamingMarkdownSlot {
    item_index: usize,
    width: usize,
    theme: Theme,
    renderer: StreamingMarkdownRenderer,
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
        self.live_render_frames.clear();
        self.streaming_markdown = None;
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
            && self.live_render_frames.is_empty()
            && self.streaming_markdown.is_none()
    }

    pub(crate) fn prepare(&mut self, width: usize, theme: Theme, timeline_cache_id: u64) -> bool {
        let width_changed = self.width.is_some() && self.width != Some(width);
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
            self.live_render_frames.clear();
            self.streaming_markdown = None;
        }
        width_changed
    }

    /// 获取缓存条目的引用（用于文本选择）
    pub fn entries(&self) -> &[TranscriptRenderCacheEntry] {
        &self.entries
    }

    #[cfg(test)]
    pub(crate) fn set_entries_for_test(&mut self, entries: Vec<TranscriptRenderCacheEntry>) {
        self.live_render_frames.resize(entries.len(), None);
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

    /// Invalidate row geometry while retaining item documents for incremental updates.
    pub(crate) fn invalidate_row_metadata(&mut self) {
        self.row_metadata_revision = None;
        self.total_rows = None;
        self.row_starts.clear();
        self.row_counts.clear();
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
    state.last_scrollbar_area = scrollbar_area.unwrap_or_default();

    let width = content_area.width.max(1) as usize;
    let (total_rows, width_reflowed) = cached_transcript_row_count_with_reflow(state, theme, width);
    state.sync_transcript_viewport_rows_with_reflow(total_rows, width_reflowed);
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
        && visible_rows > 0
        && let Some(geometry) = ScrollbarGeometry::new(
            usize::from(scrollbar_area.height),
            total_rows,
            usize::from(visible_rows),
            scroll,
        )
    {
        let backdrop = assistant_fade::fade_backdrop(theme, state.terminal_background());
        let buffer = frame.buffer_mut();
        for row in 0..scrollbar_area.height {
            let (symbol, style) = geometry.cell(usize::from(row), theme, backdrop);
            if let Some(cell) = buffer.cell_mut((scrollbar_area.x, scrollbar_area.y + row)) {
                cell.set_char(symbol).set_style(style);
            }
        }
    }
}

#[cfg(test)]
fn visible_transcript_lines(
    lines: &[Line<'static>],
    visible_rows: u16,
    top_scroll: usize,
) -> Vec<Line<'static>> {
    let visible_rows = visible_rows as usize;
    if visible_rows == 0 {
        return Vec::new();
    }

    let start = top_scroll.min(lines.len());
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
        if timeline_item_needs_separator_before(
            index,
            items,
            state.thoughts_display,
            state.tools_display,
        ) {
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
                state
                    .is_historian_child_view()
                    .then_some(state.historian_report_options),
                state.thoughts_display,
                index + 1 < items.len() && matches!(items[index + 1], TimelineItem::Reasoning(_)),
                state.tools_display,
                index + 1 < items.len() && is_compact_tool_group_item(&items[index + 1]),
                compact_tool_group_stats(items, index),
                compact_reasoning_group_elapsed_ms(items, index),
                std::time::Instant::now(),
                theme.canvas(),
                &state.translator(),
                None,
            ),
        ));
    }

    lines
}

fn cached_transcript_row_count(state: &mut TuiState, theme: Theme, width: usize) -> usize {
    cached_transcript_row_count_with_reflow(state, theme, width).0
}

fn cached_transcript_row_count_with_reflow(
    state: &mut TuiState,
    theme: Theme,
    width: usize,
) -> (usize, bool) {
    let item_count = state.active_timeline().items().len();
    if item_count == 0 {
        return (0, false);
    }

    let timeline = state.active_timeline();
    let timeline_cache_id = timeline.cache_id();
    let mutation_revision = timeline.mutation_revision();
    let width_changed = state
        .transcript_render_cache
        .prepare(width, theme, timeline_cache_id);
    if width_changed {
        state.on_timeline_changed();
    }

    if transcript_row_metadata_is_current(state) {
        return (
            state
                .transcript_render_cache
                .total_rows
                .expect("current transcript row metadata has a total row count"),
            width_changed,
        );
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
    state
        .transcript_render_cache
        .live_render_frames
        .resize(item_count, None);

    let mut rows = surface::TRANSCRIPT_TOP_SPACER;
    state.transcript_render_cache.row_starts.clear();
    state.transcript_render_cache.row_counts.clear();

    for index in 0..item_count {
        let separator_rows = if timeline_item_needs_separator_before(
            index,
            state.active_timeline().items(),
            state.thoughts_display,
            state.tools_display,
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
    (rows, width_changed)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TranscriptViewportRange {
    Spacer {
        start: usize,
        end: usize,
    },
    Separator,
    Item {
        index: usize,
        line_start: usize,
        line_end: usize,
    },
}

fn transcript_viewport_ranges(
    state: &TuiState,
    visible_rows: u16,
    top_scroll: usize,
) -> Vec<TranscriptViewportRange> {
    let visible_rows = visible_rows as usize;
    if visible_rows == 0 {
        return Vec::new();
    }

    let total_rows = state.transcript_render_cache.total_rows.unwrap_or(0);
    let start = top_scroll.min(total_rows);
    let end = start.saturating_add(visible_rows).min(total_rows);
    if start >= end {
        return Vec::new();
    }

    let mut ranges = Vec::new();
    let spacer_end = surface::TRANSCRIPT_TOP_SPACER.min(end);
    if start < spacer_end {
        ranges.push(TranscriptViewportRange::Spacer {
            start,
            end: spacer_end,
        });
    }

    let items = state.active_timeline().items();
    let first_item = state
        .transcript_render_cache
        .row_starts
        .partition_point(|row_start| *row_start < start)
        .saturating_sub(1);
    for index in first_item..items.len() {
        let item_start = state.transcript_render_cache.row_starts[index];
        let item_count = state.transcript_render_cache.row_counts[index];
        let separator_rows = usize::from(timeline_item_needs_separator_before(
            index,
            items,
            state.thoughts_display,
            state.tools_display,
        ));
        let separator_start = item_start.saturating_sub(separator_rows);
        let item_end = item_start.saturating_add(item_count);

        if separator_start >= end {
            break;
        }
        if separator_rows > 0 && separator_start >= start {
            ranges.push(TranscriptViewportRange::Separator);
        }
        if item_end <= start {
            continue;
        }

        let line_start = start.saturating_sub(item_start).min(item_count);
        let line_end = end.saturating_sub(item_start).min(item_count);
        if line_start < line_end {
            ranges.push(TranscriptViewportRange::Item {
                index,
                line_start,
                line_end,
            });
        }
    }

    ranges
}

fn visible_cached_transcript_lines(
    state: &mut TuiState,
    theme: Theme,
    width: usize,
    visible_rows: u16,
    top_scroll: usize,
) -> Vec<Line<'static>> {
    if visible_rows == 0 || state.active_timeline().items().is_empty() {
        return Vec::new();
    }

    let width_changed =
        state
            .transcript_render_cache
            .prepare(width, theme, state.active_timeline().cache_id());
    if width_changed {
        state.on_timeline_changed();
    }
    if !transcript_row_metadata_is_current(state) {
        cached_transcript_row_count(state, theme, width);
    }

    let ranges = transcript_viewport_ranges(state, visible_rows, top_scroll);
    let mut visible = Vec::with_capacity(visible_rows as usize);
    for range in ranges {
        match range {
            TranscriptViewportRange::Spacer { start, end } => {
                visible.extend((start..end).map(|_| Line::from("")));
            }
            TranscriptViewportRange::Separator => visible.push(Line::from("")),
            TranscriptViewportRange::Item {
                index,
                line_start,
                line_end,
            } => {
                // Slice the same document used to calculate the viewport geometry.
                let document = &state.transcript_render_cache.entries[index].document;
                if line_start == 0 && line_end == document.lines.len() {
                    visible.extend(transcript_ratatui::document_to_ratatui(document));
                } else {
                    visible.extend(
                        document.lines[line_start..line_end]
                            .iter()
                            .map(transcript_ratatui::line_to_ratatui),
                    );
                }
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
    top_scroll: usize,
) -> Vec<Option<&RenderLine<Style>>> {
    let ranges = transcript_viewport_ranges(state, visible_rows, top_scroll);
    let mut rows = Vec::with_capacity(visible_rows as usize);
    for range in ranges {
        match range {
            TranscriptViewportRange::Spacer { start, end } => {
                rows.extend((start..end).map(|_| None));
            }
            TranscriptViewportRange::Separator => rows.push(None),
            TranscriptViewportRange::Item {
                index,
                line_start,
                line_end,
            } => rows.extend(
                state.transcript_render_cache.entries[index].document.lines[line_start..line_end]
                    .iter()
                    .map(Some),
            ),
        }
    }
    rows.truncate(visible_rows as usize);
    rows
}

fn transcript_row_metadata_is_current(state: &TuiState) -> bool {
    let timeline = state.active_timeline();
    state.transcript_render_cache.row_metadata_revision == Some(timeline.mutation_revision())
        && state.transcript_render_cache.total_rows.is_some()
        && state
            .transcript_render_cache
            .live_render_frames
            .iter()
            .flatten()
            .all(|frame| *frame == state.status_spinner_frame)
}

fn tool_keeps_card_in_compact_mode(tool: &ToolView) -> bool {
    crate::agent::is_subagent_tool_name(&tool.name)
        || crate::agent::is_subagent_control_tool_name(&tool.name)
}

fn is_compact_tool_group_item(item: &TimelineItem) -> bool {
    matches!(item, TimelineItem::Tool(tool) if !tool_keeps_card_in_compact_mode(tool))
}

fn timeline_item_needs_separator_before(
    index: usize,
    items: &[TimelineItem],
    thoughts_display: crate::command::ThoughtsDisplayMode,
    tools_display: crate::command::ToolsDisplayMode,
) -> bool {
    if index == 0 {
        return false;
    }
    if tools_display == crate::command::ToolsDisplayMode::Compact
        && let TimelineItem::Tool(tool) = &items[index]
    {
        if tool_keeps_card_in_compact_mode(tool) {
            return true;
        }
        if index + 1 < items.len() && is_compact_tool_group_item(&items[index + 1]) {
            return false;
        }
        let group_start = (0..index)
            .rev()
            .take_while(|previous| is_compact_tool_group_item(&items[*previous]))
            .last()
            .unwrap_or(index);
        return group_start > 0;
    }
    match (&items[index - 1], &items[index]) {
        (TimelineItem::Todo(_), TimelineItem::Todo(_)) => false,
        (TimelineItem::Reasoning(_), TimelineItem::Reasoning(_)) => {
            thoughts_display != crate::command::ThoughtsDisplayMode::Compact
        }
        _ => true,
    }
}

fn compact_tool_group_stats(items: &[TimelineItem], index: usize) -> tool_card::ToolGroupStats {
    if !items.get(index).is_some_and(is_compact_tool_group_item) {
        return tool_card::ToolGroupStats::default();
    }
    let start = (0..index)
        .rev()
        .take_while(|previous| is_compact_tool_group_item(&items[*previous]))
        .last()
        .unwrap_or(index);
    let mut stats = tool_card::ToolGroupStats::default();
    for tool in items[start..=index].iter().filter_map(|item| match item {
        TimelineItem::Tool(tool) => Some(tool),
        _ => None,
    }) {
        stats.call_count = stats.call_count.saturating_add(1);
        stats.failed_count = stats
            .failed_count
            .saturating_add(usize::from(tool.status == ToolExecutionStatus::Failed));
        stats.cancelled_count = stats
            .cancelled_count
            .saturating_add(usize::from(tool.status == ToolExecutionStatus::Cancelled));
        stats.active |= tool_card::tool_is_active(tool);
    }
    stats
}

fn compact_reasoning_group_elapsed_ms(items: &[TimelineItem], index: usize) -> Option<u64> {
    if !matches!(items.get(index), Some(TimelineItem::Reasoning(_))) {
        return None;
    }
    let now = std::time::Instant::now();
    let mut total = 0u64;
    let mut found = false;
    for item in items[..=index].iter().rev() {
        let TimelineItem::Reasoning(reasoning) = item else {
            break;
        };
        let elapsed = reasoning.duration_ms.or_else(|| {
            reasoning.started_at.map(|started_at| {
                u64::try_from(now.saturating_duration_since(started_at).as_millis())
                    .unwrap_or(u64::MAX)
            })
        });
        if let Some(elapsed) = elapsed {
            total = total.saturating_add(elapsed);
            found = true;
        }
    }
    found.then_some(total)
}

fn compact_tool_group_revision(
    items: &[TimelineItem],
    revisions: &[u64],
    index: usize,
) -> Option<u64> {
    if !items.get(index).is_some_and(is_compact_tool_group_item) {
        return revisions.get(index).copied();
    }
    let start = (0..index)
        .rev()
        .take_while(|previous| is_compact_tool_group_item(&items[*previous]))
        .last()
        .unwrap_or(index);
    revisions.get(start..=index)?.iter().copied().max()
}

fn compact_reasoning_group_revision(
    items: &[TimelineItem],
    revisions: &[u64],
    index: usize,
) -> Option<u64> {
    if !matches!(items.get(index), Some(TimelineItem::Reasoning(_))) {
        return revisions.get(index).copied();
    }
    let start = (0..index)
        .rev()
        .take_while(|previous| matches!(items[*previous], TimelineItem::Reasoning(_)))
        .last()
        .unwrap_or(index);
    revisions.get(start..=index)?.iter().copied().max()
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

#[cfg(test)]
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

    let streaming = matches!(
        state.active_timeline().items().get(index),
        Some(TimelineItem::Assistant(message)) if message.streaming
    );
    let mut streaming_slot = state.transcript_render_cache.streaming_markdown.take();
    if streaming {
        let slot = streaming_slot.get_or_insert_with(|| StreamingMarkdownSlot {
            item_index: index,
            width,
            theme,
            renderer: StreamingMarkdownRenderer::new(
                theme,
                MarkdownRenderOptions::new(width.saturating_sub(2).max(1)),
            ),
        });
        if slot.item_index != index || slot.width != width || slot.theme != theme {
            slot.item_index = index;
            slot.width = width;
            slot.theme = theme;
            slot.renderer = StreamingMarkdownRenderer::new(
                theme,
                MarkdownRenderOptions::new(width.saturating_sub(2).max(1)),
            );
        }
    }

    let now = std::time::Instant::now();
    let fade_backdrop = assistant_fade::fade_backdrop(theme, state.terminal_background());
    let (revision, live, item, next_reasoning, next_tool, tool_group_stats) = {
        let timeline = state.active_timeline();
        let items = timeline.items();
        let revisions = timeline.item_revisions();
        let item = &items[index];
        let next_reasoning =
            index + 1 < items.len() && matches!(items[index + 1], TimelineItem::Reasoning(_));
        let next_tool = index + 1 < items.len() && is_compact_tool_group_item(&items[index + 1]);
        let tool_group_stats =
            if state.tools_display == crate::command::ToolsDisplayMode::Compact && !next_tool {
                compact_tool_group_stats(items, index)
            } else {
                tool_card::ToolGroupStats::default()
            };
        let revision = if state.tools_display == crate::command::ToolsDisplayMode::Compact
            && is_compact_tool_group_item(item)
            && !next_tool
        {
            compact_tool_group_revision(items, revisions, index)
        } else if state.thoughts_display == crate::command::ThoughtsDisplayMode::Compact
            && matches!(item, TimelineItem::Reasoning(_))
            && !next_reasoning
        {
            compact_reasoning_group_revision(items, revisions, index)
        } else {
            revisions.get(index).copied()
        };
        let live = if state.tools_display == crate::command::ToolsDisplayMode::Compact
            && is_compact_tool_group_item(item)
            && !next_tool
        {
            tool_group_stats.active
        } else {
            matches!(
                item,
                TimelineItem::Tool(tool)
                    if matches!(
                        tool.status,
                        crate::tui::timeline::ToolExecutionStatus::Pending
                            | crate::tui::timeline::ToolExecutionStatus::Running
                    )
            ) || matches!(
                item,
                TimelineItem::Reasoning(reasoning)
                    if reasoning.streaming || reasoning.transition_started_at.is_some()
            ) || matches!(item, TimelineItem::Assistant(message) if message.is_fading(now))
        };
        (
            revision,
            live,
            item,
            next_reasoning,
            next_tool,
            tool_group_stats,
        )
    };
    let frame = state.status_spinner_frame;
    if state.transcript_render_cache.entries[index].revision == revision
        && (!live || state.transcript_render_cache.live_render_frames[index] == Some(frame))
    {
        state.transcript_render_cache.streaming_markdown = streaming_slot;
        return;
    }
    let document = render_timeline_item_document(
        item,
        theme,
        width,
        state.status_spinner_frame,
        tool_output_expanded_for_item(state, item),
        is_reviewer_child_view(state),
        state
            .is_historian_child_view()
            .then_some(state.historian_report_options),
        state.thoughts_display,
        next_reasoning,
        state.tools_display,
        next_tool,
        tool_group_stats,
        if state.thoughts_display == crate::command::ThoughtsDisplayMode::Compact && next_reasoning
        {
            None
        } else {
            compact_reasoning_group_elapsed_ms(state.active_timeline().items(), index)
        },
        now,
        fade_backdrop,
        &state.translator(),
        streaming_slot.as_mut().map(|slot| &mut slot.renderer),
    );
    #[cfg(test)]
    let assistant_markdown_chars = match item {
        TimelineItem::Assistant(message) if message.streaming => streaming_slot
            .as_ref()
            .map(|slot| slot.renderer.last_parse_chars())
            .unwrap_or_else(|| message_text(message).len()),
        TimelineItem::Assistant(message) => message_text(message).len(),
        _ => 0,
    };
    state.transcript_render_cache.streaming_markdown = streaming_slot;
    #[cfg(test)]
    {
        state.transcript_render_cache.assistant_markdown_chars += assistant_markdown_chars;
    }
    let entry = &mut state.transcript_render_cache.entries[index];
    entry.revision = revision;
    entry.document = document;
    state.transcript_render_cache.live_render_frames[index] = live.then_some(frame);
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
    historian_view: Option<historian_report::ReportOptions>,
    thoughts_display: crate::command::ThoughtsDisplayMode,
    next_reasoning: bool,
    tools_display: crate::command::ToolsDisplayMode,
    next_tool: bool,
    tool_group_stats: tool_card::ToolGroupStats,
    compact_reasoning_elapsed_ms: Option<u64>,
    now: std::time::Instant,
    fade_backdrop: Color,
    translator: &'a Translator,
    streaming_markdown: Option<&'a mut StreamingMarkdownRenderer>,
}

impl Component<Style> for TimelineItemComponent<'_> {
    fn render(&mut self, document: &mut Document<Style>) {
        if self.reviewer_view
            && let Some(specialized) =
                try_render_reviewer_view_item(self.item, self.theme, self.width)
        {
            *document = specialized;
            return;
        }

        if let Some(options) = self.historian_view
            && let TimelineItem::Assistant(message) = self.item
            && let Some(report) = historian_report::parse_report(message_text(message))
        {
            *document = pad_card_document(
                historian_report::render_report(
                    &report,
                    options,
                    self.theme,
                    card_content_width(self.width, self.theme),
                    self.translator,
                ),
                self.width,
                self.theme,
            );
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
                self.compact_reasoning_elapsed_ms,
            ),
            TimelineItem::Assistant(message) => {
                let fade = message.is_fading(self.now).then_some(AssistantFade {
                    reveals: &message.reveal_tail,
                    now: self.now,
                    backdrop: self.fade_backdrop,
                });
                build_assistant_message_lines(
                    &mut out,
                    message_text(message),
                    message.streaming,
                    self.theme,
                    self.width,
                    self.streaming_markdown.as_deref_mut(),
                    fade,
                )
            }
            TimelineItem::Tool(tool) => {
                if self.tools_display == crate::command::ToolsDisplayMode::Compact {
                    if tool_keeps_card_in_compact_mode(tool) {
                        build_tool_lines(
                            &mut out,
                            tool,
                            self.theme,
                            self.width,
                            self.frame,
                            self.expanded_output,
                            self.translator,
                        );
                    } else if !self.next_tool {
                        out.document = tool_card::render_tool_group_document(
                            tool,
                            self.tool_group_stats,
                            self.theme,
                            self.width,
                            self.frame,
                            self.translator,
                        );
                    }
                } else {
                    build_tool_lines(
                        &mut out,
                        tool,
                        self.theme,
                        self.width,
                        self.frame,
                        self.expanded_output,
                        self.translator,
                    );
                }
            }
            TimelineItem::Delegation(delegation) => {
                append_card_document(
                    &mut out,
                    delegation_card::render_delegation_card_document(
                        delegation,
                        self.theme,
                        card_content_width(self.width, self.theme),
                        self.translator,
                    ),
                    self.width,
                    self.theme,
                );
            }
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
    let document = match item {
        TimelineItem::User(message) => {
            let card = reviewer_cards::parse_review_request(message_text(message))?;
            reviewer_cards::render_review_request_card_document(
                &card,
                theme,
                card_content_width(width, theme),
            )
        }
        TimelineItem::Assistant(message) => {
            let card = reviewer_cards::parse_review_decision(message_text(message))?;
            reviewer_cards::render_review_decision_card_document(
                &card,
                theme,
                card_content_width(width, theme),
            )
        }
        _ => return None,
    };
    Some(pad_card_document(document, width, theme))
}

fn render_timeline_item_document(
    item: &TimelineItem,
    theme: Theme,
    width: usize,
    frame: usize,
    expanded_output: bool,
    reviewer_view: bool,
    historian_view: Option<historian_report::ReportOptions>,
    thoughts_display: crate::command::ThoughtsDisplayMode,
    next_reasoning: bool,
    tools_display: crate::command::ToolsDisplayMode,
    next_tool: bool,
    tool_group_stats: tool_card::ToolGroupStats,
    compact_reasoning_elapsed_ms: Option<u64>,
    now: std::time::Instant,
    fade_backdrop: Color,
    translator: &Translator,
    streaming_markdown: Option<&mut StreamingMarkdownRenderer>,
) -> Document<Style> {
    let mut document = Document::default();
    TimelineItemComponent {
        item,
        theme,
        width,
        frame,
        expanded_output,
        reviewer_view,
        historian_view,
        thoughts_display,
        next_reasoning,
        tools_display,
        next_tool,
        tool_group_stats,
        compact_reasoning_elapsed_ms,
        now,
        fade_backdrop,
        translator,
        streaming_markdown,
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
    if width == 0 {
        return;
    }
    push_drawn_horizontal_rule(out, theme, width);

    if !summary.is_empty() {
        let start = out.document.lines.len();
        build_assistant_message_lines(
            out,
            summary,
            streaming,
            theme,
            card_content_width(width, theme),
            None,
            None,
        );
        pad_card_lines(&mut out.document, start, width);
    }

    if !streaming {
        push_drawn_horizontal_rule(out, theme, width);
    }
}

fn card_content_width(width: usize, theme: Theme) -> usize {
    width
        .saturating_sub(surface::CARD_PAD_RIGHT as usize)
        .saturating_sub(usize::from(theme.card_frame))
}

fn pad_card_document(mut document: Document<Style>, width: usize, theme: Theme) -> Document<Style> {
    if theme.card_frame {
        frame_card_document(&mut document, width, theme);
    } else {
        pad_card_lines(&mut document, 0, width);
    }
    document
}

fn pad_card_lines(document: &mut Document<Style>, start: usize, width: usize) {
    for line in document.lines.iter_mut().skip(start) {
        let used = line
            .spans
            .iter()
            .map(|span| display_width(&span.text))
            .sum::<usize>();
        if width > used {
            let style = line.spans.last().map(|span| span.style).unwrap_or_default();
            line.spans
                .push(RenderSpan::decoration(" ".repeat(width - used), style));
        }
    }
}

fn frame_card_document(document: &mut Document<Style>, width: usize, theme: Theme) {
    if document.lines.is_empty() || width < 2 {
        pad_card_lines(document, 0, width);
        return;
    }

    let border_style = Style::default().fg(theme.border).bg(theme.root_bg);
    let content_width = width.saturating_sub(1);
    let line_count = document.lines.len();
    for line in document.lines.iter_mut().take(line_count) {
        if let Some(first) = line.spans.first_mut()
            && first.text == surface::ACCENT_BAR_GLYPH
        {
            first.text = "│".into();
            first.style = border_style;
        } else {
            line.spans
                .insert(0, RenderSpan::decoration("│", border_style));
        }

        let used = line
            .spans
            .iter()
            .map(|span| display_width(&span.text))
            .sum::<usize>();
        if content_width > used {
            let style = line.spans.last().map(|span| span.style).unwrap_or_default();
            line.spans.push(RenderSpan::decoration(
                " ".repeat(content_width - used),
                style,
            ));
        }
        line.spans.push(RenderSpan::decoration("│", border_style));
    }

    let previous_final_break = document.breaks.last().copied().unwrap_or(Break::End);
    if let Some(boundary) = document.breaks.last_mut() {
        *boundary = Break::BlockBreak;
    }
    document.lines.insert(
        0,
        RenderLine {
            spans: vec![RenderSpan::decoration(
                format!("┌{}┐", "─".repeat(width.saturating_sub(2))),
                border_style,
            )],
        },
    );
    document.breaks.insert(0, Break::HardBreak);
    document.lines.push(RenderLine {
        spans: vec![RenderSpan::decoration(
            format!("└{}┘", "─".repeat(width.saturating_sub(2))),
            border_style,
        )],
    });
    document.breaks.push(previous_final_break);
}

fn frame_user_card_lines(document: &mut Document<Style>, start: usize, width: usize, theme: Theme) {
    if !theme.card_frame || width < 3 {
        return;
    }

    let Some(end) = document.lines.len().checked_sub(1) else {
        return;
    };
    if start > end {
        return;
    }

    let border_style = Style::default().fg(theme.border).bg(theme.root_bg);
    let horizontal = "─".repeat(width.saturating_sub(2));
    let top = RenderLine {
        spans: vec![RenderSpan::decoration(
            format!("┌{horizontal}┐"),
            border_style,
        )],
    };
    let bottom = RenderLine {
        spans: vec![RenderSpan::decoration(
            format!("└{horizontal}┘"),
            border_style,
        )],
    };
    document.lines[start] = top;
    document.lines[end] = bottom;
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

const REASONING_SCROLL_LINES: usize = 3;

fn build_reasoning_lines(
    out: &mut TimelineDocument,
    reasoning: &ReasoningView,
    theme: Theme,
    width: usize,
    display: crate::command::ThoughtsDisplayMode,
    next_reasoning: bool,
    compact_group_elapsed_ms: Option<u64>,
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
    let elapsed = compact_group_elapsed_ms
        .map(format_elapsed_duration)
        .unwrap_or_else(|| format_reasoning_elapsed(reasoning));
    let title = if display == crate::command::ThoughtsDisplayMode::Compact {
        animated_compact_reasoning_title(reasoning, &title, content_width)
    } else {
        title
    };

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

    let visible = chunks
        .iter()
        .enumerate()
        .filter(|(_, chunk)| chunk.source_start_char != chunk.source_end_char)
        .map(|(index, _)| index)
        .collect::<Vec<_>>();
    let first = if display == crate::command::ThoughtsDisplayMode::Scroll {
        visible.len().saturating_sub(REASONING_SCROLL_LINES)
    } else {
        0
    };
    let mut pushed = false;
    for &index in &visible[first..] {
        let chunk = &chunks[index];
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

fn animated_compact_reasoning_title(
    reasoning: &ReasoningView,
    title: &str,
    width: usize,
) -> String {
    animated_compact_reasoning_title_at(reasoning, title, width, std::time::Instant::now())
}

fn animated_compact_reasoning_title_at(
    reasoning: &ReasoningView,
    title: &str,
    width: usize,
    now: std::time::Instant,
) -> String {
    let target = tool_card::truncate_display_width(title, width);
    let (Some(previous), Some(started_at)) = (
        reasoning.transition_from.as_deref(),
        reasoning.transition_started_at,
    ) else {
        return target;
    };
    let previous = reasoning_title_and_body(previous)
        .0
        .unwrap_or_else(|| previous.to_string());
    let previous = tool_card::truncate_display_width(&previous, width);
    let elapsed_ms = now.saturating_duration_since(started_at).as_millis();
    let frame = usize::try_from(
        elapsed_ms / u128::from(crate::tui::timeline::REASONING_TRANSITION_FRAME_MS),
    )
    .unwrap_or(usize::MAX);

    let previous = previous.graphemes(true).collect::<Vec<_>>();
    let target = target.graphemes(true).collect::<Vec<_>>();
    let positions = previous.len().max(target.len());
    let replaced = frame.min(positions);
    if replaced >= positions {
        return target.concat();
    }
    let mut rendered = String::new();
    for index in 0..positions {
        if index < replaced {
            if let Some(grapheme) = target.get(index) {
                rendered.push_str(grapheme);
            }
        } else if let Some(grapheme) = previous.get(index) {
            rendered.push_str(grapheme);
        }
    }
    tool_card::truncate_display_width(&rendered, width)
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
    let card_start = out.document.lines.len();
    let content_width = width.saturating_sub(5).max(1);

    // 整段 text 一次 wrap，得到每个视觉行对应原文的字符区间。
    let chunks = wrap_text_to_width_with_offsets(text, content_width);
    let block_index = out.add_source(text.to_string());

    // 顶部空 card 行（decoration）
    push_user_card_line_into(out, "", None, width, theme, None);
    if theme.card_frame {
        push_user_card_line_into(out, "", None, width, theme, None);
    }
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
    if theme.card_frame {
        push_user_card_line_into(out, "", None, width, theme, None);
    }
    frame_user_card_lines(&mut out.document, card_start, width, theme);
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
    let framed = theme.card_frame && width >= 3;
    let bar_style = if framed {
        Style::default().fg(theme.border).bg(theme.root_bg)
    } else {
        surface::accent_style(
            theme,
            surface::SurfaceEmphasis::User,
            surface::SurfaceKind::Root,
        )
    };
    let pad_style = user_prompt_padding_style(theme);
    let badge_style = queued_badge_style(theme);
    let left_edge = if framed {
        "│"
    } else {
        surface::ACCENT_BAR_GLYPH
    };

    let mut spans = vec![
        RenderSpan::decoration(left_edge, bar_style),
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
    let target_width = width.saturating_sub(usize::from(framed));
    if target_width > used {
        spans.push(RenderSpan::decoration(
            " ".repeat(target_width - used),
            pad_style,
        ));
    } else if !framed {
        spans.push(RenderSpan::decoration("  ", pad_style));
    }
    if framed {
        spans.push(RenderSpan::decoration(
            "│",
            Style::default().fg(theme.border).bg(theme.root_bg),
        ));
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
    streaming_markdown: Option<&mut StreamingMarkdownRenderer>,
    fade: Option<AssistantFade<'_>>,
) {
    if text.is_empty() {
        out.push_decoration(
            Line::from(Span::styled("  …", root_muted_style(theme))),
            Break::End,
        );
        return;
    }

    if let Some(result) = try_parse_structured_subagent_result(text) {
        out.document.append(pad_card_document(
            structured_subagent::render_structured_subagent_result_document(
                &result,
                theme,
                card_content_width(width, theme),
            ),
            width,
            theme,
        ));
        return;
    }

    let content_width = width.saturating_sub(2).max(1);
    let mut document = match streaming_markdown {
        Some(renderer) if streaming => renderer.render(text),
        _ => render_markdown_document(text, theme, MarkdownRenderOptions::new(content_width)),
    };
    if let Some(fade) = fade {
        assistant_fade::apply_fade(&mut document, fade);
    }
    for line in &mut document.lines {
        line.spans
            .insert(0, RenderSpan::decoration("  ", theme.app_style()));
    }
    out.document.append(document);
}

fn append_card_document(
    out: &mut TimelineDocument,
    document: Document<Style>,
    width: usize,
    theme: Theme,
) {
    out.document
        .append(pad_card_document(document, width, theme));
}

fn append_tool_document(
    out: &mut TimelineDocument,
    mut document: Document<Style>,
    width: usize,
    theme: Theme,
) {
    if theme.card_frame && document.lines.len() <= 1 {
        pad_card_lines(&mut document, 0, width);
        out.document.append(document);
    } else {
        append_card_document(out, document, width, theme);
    }
}

fn build_tool_lines(
    out: &mut TimelineDocument,
    tool: &ToolView,
    theme: Theme,
    width: usize,
    frame: usize,
    expanded_output: bool,
    translator: &Translator,
) {
    append_tool_document(
        out,
        tool_card::render_tool_card_document(
            tool,
            theme,
            card_content_width(width, theme),
            frame,
            expanded_output,
            translator,
        ),
        width,
        theme,
    );
}

fn build_todo_card(
    out: &mut TimelineDocument,
    todo: &crate::tui::timeline::TodoView,
    theme: Theme,
    width: usize,
) {
    append_card_document(
        out,
        todo_card::render_todo_card_document(todo, theme, card_content_width(width, theme)),
        width,
        theme,
    );
}

fn build_permission_lines(
    out: &mut TimelineDocument,
    permission: &PermissionView,
    theme: Theme,
    width: usize,
) {
    if permission.status != PermissionPromptStatus::Pending {
        append_card_document(
            out,
            tool_card::render_permission_card_document(
                permission,
                theme,
                card_content_width(width, theme),
            ),
            width,
            theme,
        );
    }
}

const AUTO_REVIEW_INDENT: &str = "  ";

fn build_auto_review_lines(
    out: &mut TimelineDocument,
    decision: &crate::tui::timeline::AutoReviewDecisionView,
    theme: Theme,
    width: usize,
    expanded: bool,
) {
    let card_width = width;
    let width = card_content_width(width, theme);
    let content_width = width.saturating_sub(display_width(AUTO_REVIEW_INDENT));
    if content_width == 0 {
        return;
    }
    let start = out.document.lines.len();
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
    for line in wrap_text_to_width(&title, content_width) {
        out.push_decoration(
            Line::from(vec![
                Span::styled(AUTO_REVIEW_INDENT, status_style),
                Span::styled(line, status_style),
            ]),
            Break::HardBreak,
        );
    }
    if expanded {
        let risk = decision.risk.as_deref().unwrap_or("unknown");
        let detail = format!(
            "risk {risk} · {} · call {}",
            decision.rationale, decision.call_id,
        );
        push_auto_review_text(
            out,
            &detail,
            root_dim_style(theme),
            content_width,
            Break::HardBreak,
        );
    }
    pad_card_lines(&mut out.document, start, card_width);
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
                    spans: vec![
                        RenderSpan::decoration(AUTO_REVIEW_INDENT, style),
                        RenderSpan::source(
                            chunk.text.clone(),
                            style,
                            SourceRange::new(block, chunk.source_start_char, chunk.source_end_char),
                        ),
                    ],
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
    let card_width = width;
    let width = card_content_width(width, theme);
    if width == 0 {
        return;
    }
    let start = out.document.lines.len();
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
    pad_card_lines(&mut out.document, start, card_width);
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
        .fg(theme.on_accent)
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
    scroll_offset: usize,
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
    let selection_style = Style::default().bg(theme.accent).fg(theme.on_accent);

    // 遍历可见行，应用高亮
    for (idx, line) in lines.iter_mut().enumerate() {
        let absolute_row = scroll_offset + idx;

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
#[path = "../../../tests/internal/tui/components/transcript/tests.rs"]
mod tests;
