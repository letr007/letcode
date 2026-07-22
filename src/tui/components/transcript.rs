use ratatui::{
    Frame,
    layout::{Constraint, Layout, Rect},
    style::{Modifier, Style},
    text::{Line, Span, Text},
    widgets::{Block, Paragraph, Scrollbar, ScrollbarOrientation, ScrollbarState},
};

use crate::tui::{
    markdown::{MarkdownRenderOptions, render_markdown, render_markdown_semantic_blocks},
    measure::{display_width, wrap_text_to_width, wrap_text_to_width_with_offsets},
    surface,
    theme::Theme,
    timeline::{
        COMPACTION_SEPARATOR_LABEL, DelegationView, ErrorView, MessageView, PermissionPromptStatus,
        PermissionView, ReasoningView, TimelineItem, ToolView,
    },
};
use crate::user_content::UserImageAttachment;

use super::super::state::TuiState;
use super::{composer::one_line_snippet, todo_card, tool_card};

#[derive(Debug, Clone, Default)]
pub struct TranscriptRenderCache {
    width: Option<usize>,
    theme: Option<Theme>,
    timeline_cache_id: Option<u64>,
    entries: Vec<TranscriptRenderCacheEntry>,
    row_starts: Vec<usize>,
    row_counts: Vec<usize>,
}

impl TranscriptRenderCache {
    pub fn clear(&mut self) {
        self.width = None;
        self.theme = None;
        self.timeline_cache_id = None;
        self.entries.clear();
        self.row_starts.clear();
        self.row_counts.clear();
    }

    #[cfg(test)]
    pub(crate) fn is_empty(&self) -> bool {
        self.width.is_none()
            && self.theme.is_none()
            && self.timeline_cache_id.is_none()
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
            self.entries.clear();
            self.row_starts.clear();
            self.row_counts.clear();
        }
    }

    /// 获取缓存条目的引用（用于文本选择）
    pub fn entries(&self) -> &[TranscriptRenderCacheEntry] {
        &self.entries
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
    revision: Option<u64>,
    pub lines: Vec<Line<'static>>,
    /// 每条渲染行对应的源文本区间映射；长度必须等于 `lines` 长度。
    /// 用于把视觉选择/复制坐标还原到 transcript 原始文本，避免在剪贴板里混入
    /// card 边框、padding、wrap 折行等渲染装饰。
    pub line_origins: Vec<RenderedLineOrigin>,
    /// 该 item 涉及的源文本块；`line_origins[i].block_index` 索引本数组。
    pub source_blocks: Vec<RenderedSourceBlock>,
}

/// 单条 TimelineItem 渲染时使用的一系列源文本块。
///
/// "源文本"指 TimelineItem 里该 block 对应的原始字段（如 message.text、
/// reasoning.text、notice.message、error.message/error.details），不含任何渲染装饰。
/// 对复杂 item（工具卡/todo/permission/markdown），先用占位 block 表示，
/// 后续按 P1 逐步补全 source 区间。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RenderedSourceBlock {
    pub source: String,
}

/// 一条渲染行到源文本的映射。
///
/// - `block_index`：所在 `Vec<RenderedSourceBlock>` 索引；`None` 表示纯装饰行
///   （card 顶部/底部空行、separator、compaction rule、badge 行等），
///   命中映射与复制时都跳过。
/// - `content_prefix_chars`：渲染行左侧装饰字符宽度（cell 数），
///   例如 user card 行 = `┃` + 两格 padding + 可选 badge 段 = 3 或更多。
/// - `content_char_offset`：渲染行在 `block.source` 中起始字符位置；
///   即此前所有 chunk 的字符数之和。空行 / 装饰行使用 0。
/// - `content_char_len`：本渲染行对应的纯内容字符长度；装饰行 / spacer = 0。
///
/// 复制时只用 `block_index` + `content_char_offset..content_char_offset+content_char_len`
/// 从 `source` 切片；同 block 的连续行合并为一次切片以保留原文中的换行。
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct RenderedLineOrigin {
    pub block_index: Option<usize>,
    pub content_prefix_chars: usize,
    pub content_char_offset: usize,
    pub content_char_len: usize,
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

pub fn transcript_lines(state: &TuiState, theme: Theme, width: usize) -> Vec<Line<'static>> {
    let mut lines = Vec::new();

    if !state.active_timeline().items().is_empty() {
        lines.extend((0..surface::TRANSCRIPT_TOP_SPACER).map(|_| Line::from("")));
    }

    for (index, item) in state.active_timeline().items().iter().enumerate() {
        if index > 0 && timeline_item_needs_separator_before(item) {
            lines.push(Line::from(""));
        }

        lines.extend(
            render_timeline_item_lines(
                item,
                theme,
                width,
                state.status_spinner_frame,
                state.tool_output_expanded,
            )
            .lines,
        );
    }

    lines
}

fn cached_transcript_row_count(state: &mut TuiState, theme: Theme, width: usize) -> usize {
    let item_count = state.active_timeline().items().len();
    if item_count == 0 {
        return 0;
    }

    state
        .transcript_render_cache
        .prepare(width, theme, state.active_timeline().cache_id());
    state
        .transcript_render_cache
        .entries
        .resize_with(item_count, || TranscriptRenderCacheEntry {
            revision: None,
            lines: Vec::new(),
            line_origins: Vec::new(),
            source_blocks: Vec::new(),
        });

    let mut rows = surface::TRANSCRIPT_TOP_SPACER;
    state.transcript_render_cache.row_starts.clear();
    state.transcript_render_cache.row_counts.clear();

    for index in 0..item_count {
        let separator_rows = if index > 0
            && timeline_item_needs_separator_before(&state.active_timeline().items()[index])
        {
            1
        } else {
            0
        };
        rows = rows.saturating_add(separator_rows);
        state.transcript_render_cache.row_starts.push(rows);
        let line_count = cached_item_lines(state, index, theme, width).len();
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
        let separator_rows = if index > 0
            && timeline_item_needs_separator_before(&state.active_timeline().items()[index])
        {
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

fn transcript_row_metadata_is_current(state: &TuiState) -> bool {
    let item_count = state.active_timeline().items().len();
    let cache = &state.transcript_render_cache;
    cache.row_starts.len() == item_count
        && cache.row_counts.len() == item_count
        && cache.entries.len() >= item_count
        && state
            .active_timeline()
            .item_revisions()
            .iter()
            .enumerate()
            .all(|(index, revision)| cache.entries[index].revision == Some(*revision))
}

fn timeline_item_needs_separator_before(item: &TimelineItem) -> bool {
    !matches!(item, TimelineItem::Todo(_))
}

fn cached_item_lines(
    state: &mut TuiState,
    index: usize,
    theme: Theme,
    width: usize,
) -> &[Line<'static>] {
    let item = state.active_timeline().items()[index].clone();
    let revision = state.active_timeline().item_revisions().get(index).copied();
    let cache = &mut state.transcript_render_cache;

    if cache.entries.len() <= index {
        cache
            .entries
            .resize_with(index + 1, || TranscriptRenderCacheEntry {
                revision: None,
                lines: Vec::new(),
                line_origins: Vec::new(),
                source_blocks: Vec::new(),
            });
    }

    let entry = &mut cache.entries[index];
    let live = matches!(
        &item,
        TimelineItem::Tool(tool)
            if matches!(
                tool.status,
                crate::tui::timeline::ToolExecutionStatus::Pending
                    | crate::tui::timeline::ToolExecutionStatus::Running
            )
    );
    if entry.revision != revision || live {
        entry.revision = revision;
        let rendered = render_timeline_item_lines(
            &item,
            theme,
            width,
            state.status_spinner_frame,
            state.tool_output_expanded,
        );
        entry.lines = rendered.lines;
        entry.line_origins = rendered.line_origins;
        entry.source_blocks = rendered.source_blocks;
    }

    &entry.lines
}

/// `render_timeline_item_lines` 的产物：渲染行 + 源映射 + 源文本块。
///
/// 不变式：`lines.len() == line_origins.len()`。
struct RenderedTimelineItem {
    lines: Vec<Line<'static>>,
    line_origins: Vec<RenderedLineOrigin>,
    source_blocks: Vec<RenderedSourceBlock>,
}

impl RenderedTimelineItem {
    fn new() -> Self {
        Self {
            lines: Vec::new(),
            line_origins: Vec::new(),
            source_blocks: Vec::new(),
        }
    }

    /// 注册源文本块，返回其索引。
    fn add_source(&mut self, source: impl Into<String>) -> usize {
        self.source_blocks.push(RenderedSourceBlock {
            source: source.into(),
        });
        self.source_blocks.len() - 1
    }

    /// 推一条对应某 block 的渲染行。
    /// `content_prefix_chars` = 行内左侧装饰宽度；`content_char_offset/len` 在
    /// `source_blocks[block_index].source` 中的字符区间。
    fn push_content(
        &mut self,
        line: Line<'static>,
        block_index: usize,
        content_prefix_chars: usize,
        content_char_offset: usize,
        content_char_len: usize,
    ) {
        self.lines.push(line);
        self.line_origins.push(RenderedLineOrigin {
            block_index: Some(block_index),
            content_prefix_chars,
            content_char_offset,
            content_char_len,
        });
    }

    /// 推一条装饰行（card 边框 / spacer / badge / separator / compaction rule 等），
    /// 命中或复制时一律跳过。
    fn push_decoration(&mut self, line: Line<'static>) {
        self.lines.push(line);
        self.line_origins.push(RenderedLineOrigin::default());
    }

    /// 兼容旧风格 push_*：把 lines 整体作为装饰写入（用于尚未做 origin 的 item 类型）。
    /// P0 阶段保证安全（不会被复制）；后续 P1 逐个替换。
    fn extend_decoration_from(&mut self, lines: impl IntoIterator<Item = Line<'static>>) {
        for line in lines {
            self.push_decoration(line);
        }
    }

    /// 兼容旧行为：把渲染后的每一行当作一个独立 source block，整行都可选择/复制。
    /// 这会保留旧问题（soft-wrap 泄漏换行、装饰字符混入复制），但对尚未完成 source
    /// 映射的 item 类型可避免回归成“完全选不中”。
    fn extend_legacy_rendered_from(&mut self, lines: impl IntoIterator<Item = Line<'static>>) {
        for line in lines {
            let text = line.to_string();
            let len = text.chars().count();
            let block_index = self.add_source(text);
            self.push_content(line, block_index, 0, 0, len);
        }
    }
}

fn render_timeline_item_lines(
    item: &TimelineItem,
    theme: Theme,
    width: usize,
    frame: usize,
    expanded_output: bool,
) -> RenderedTimelineItem {
    let mut out = RenderedTimelineItem::new();
    match item {
        TimelineItem::User(message) => build_user_message(&mut out, message, theme, width),
        TimelineItem::Reasoning(reasoning) => {
            build_reasoning_lines(&mut out, reasoning, theme, width)
        }
        TimelineItem::Delegation(delegation) => {
            build_delegation_lines(&mut out, delegation, theme, width)
        }
        TimelineItem::Assistant(message) => build_assistant_message_lines(
            &mut out,
            message_text(message),
            message.streaming,
            theme,
            width,
        ),
        TimelineItem::Tool(tool) => {
            build_tool_lines(&mut out, tool, theme, width, frame, expanded_output)
        }
        TimelineItem::Todo(todo) => build_todo_card(&mut out, todo, theme, width),
        TimelineItem::Permission(permission) => {
            build_permission_lines(&mut out, permission, theme, width)
        }
        TimelineItem::Error(error) => build_error_lines(&mut out, error, theme, width),
        TimelineItem::CompactionPending(pending) => {
            // Legacy pending path: same visual as streaming Compaction block.
            build_compaction_block_lines(
                &mut out,
                &pending.preview,
                true,
                theme,
                width,
                frame,
            )
        }
        TimelineItem::Compaction(view) => {
            build_compaction_block_lines(
                &mut out,
                &view.summary,
                view.streaming,
                theme,
                width,
                frame,
            )
        }
        TimelineItem::CompactionSeparator => {
            build_compaction_separator_line(&mut out, COMPACTION_SEPARATOR_LABEL, theme, width)
        }
    }
    out
}

/// Render compaction as a durable transcript block:
///   [Context full — compacting…]
///   ──── rule ────
///   summary body (streaming or final)
///   ──── rule ────
///   [Compaction complete]   (only when finalized)
fn build_compaction_block_lines(
    out: &mut RenderedTimelineItem,
    summary: &str,
    streaming: bool,
    theme: Theme,
    width: usize,
    frame: usize,
) {
    if width == 0 {
        return;
    }
    let _ = frame;
    let muted = root_muted_style(theme);
    let trigger = if streaming {
        crate::tui::timeline::COMPACTION_TRIGGER_LABEL
    } else {
        crate::tui::timeline::COMPACTION_TRIGGER_LABEL
    };
    out.push_decoration(Line::from(Span::styled(
        tool_card::truncate_display_width(trigger, width),
        muted,
    )));
    out.push_decoration(Line::from(Span::styled(
        tool_card::truncate_display_width(COMPACTION_SEPARATOR_LABEL, width),
        muted,
    )));
    if summary.is_empty() {
        if streaming {
            out.push_decoration(Line::from(Span::styled(
                tool_card::truncate_display_width("…", width),
                muted,
            )));
        }
    } else {
        for line in wrap_text_to_width(summary, width.max(1)) {
            out.push_decoration(Line::from(Span::styled(line, muted)));
        }
        if streaming {
            // trailing ellipsis hint while still streaming
            out.push_decoration(Line::from(Span::styled(
                tool_card::truncate_display_width("…", width),
                muted,
            )));
        }
    }
    out.push_decoration(Line::from(Span::styled(
        tool_card::truncate_display_width(COMPACTION_SEPARATOR_LABEL, width),
        muted,
    )));
    if !streaming {
        out.push_decoration(Line::from(Span::styled(
            tool_card::truncate_display_width(
                crate::tui::timeline::COMPACTION_COMPLETE_LABEL,
                width,
            ),
            muted,
        )));
    }
}

fn build_reasoning_lines(
    out: &mut RenderedTimelineItem,
    reasoning: &ReasoningView,
    theme: Theme,
    width: usize,
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
    let title_suffix = if reasoning.streaming { " …" } else { "" };

    // 标题行：作为单独 source block（已清洗，prefix "  Thought: " 长度 11，
    // 但只用原文本部分，suffix " …" 不计入）。
    let title_block = out.add_source(title.clone());
    let title_prefix_chars = "  Thought: ".chars().count();
    let title_len = title.chars().count();
    out.push_content(
        Line::from(vec![
            Span::styled("  Thought: ", reasoning_label_style(theme)),
            Span::styled(title, reasoning_label_style(theme)),
            Span::styled(title_suffix, reasoning_label_style(theme)),
        ]),
        title_block,
        title_prefix_chars,
        0,
        title_len,
    );

    // body 行：先清洗每行拼成 cleaned_body，再 wrap 一次得到 chunk + 字符区间。
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
    for chunk in &chunks {
        // wrap_text_with_offsets 对空段产出空 chunk，filter 掉全是空的情况会改变行数，
        // 但渲染原逻辑是非空 raw 才产生行——此处近似处理：保留所有 chunk 对应一行。
        pushed = true;
        let line = Line::from(vec![
            Span::styled("  ", theme.app_style()),
            Span::styled(chunk.text.clone(), reasoning_text_style(theme)),
        ]);
        let len = chunk
            .source_end_char
            .saturating_sub(chunk.source_start_char);
        out.push_content(line, body_block, 2, chunk.source_start_char, len);
    }

    if !pushed && reasoning.streaming {
        out.push_decoration(Line::from(Span::styled("  …", root_dim_style(theme))));
    }
}

fn build_delegation_lines(
    out: &mut RenderedTimelineItem,
    delegation: &DelegationView,
    theme: Theme,
    width: usize,
) {
    // delegation 是合成单行（status + " · " + 截断 task），不是纯 task 原文。
    // P0 暂不映射，作为装饰行；后续可把 task 抽成 source block。
    let status = format!("delegate @{}", delegation.agent_name);
    let summary = one_line_snippet(&delegation.task, width.saturating_sub(20).max(1));
    let line = Line::from(vec![
        Span::styled("  ", theme.app_style()),
        Span::styled(status, theme.user_style().add_modifier(Modifier::BOLD)),
        Span::styled(" · ", theme.muted_style()),
        Span::styled(summary, theme.app_style()),
    ]);
    out.push_decoration(line);
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
    out: &mut RenderedTimelineItem,
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
    for chunk in &chunks {
        pushed = true;
        push_user_card_content_line_into(
            out,
            vec![Span::styled(
                chunk.text.clone(),
                user_prompt_panel_style(theme),
            )],
            None,
            width,
            theme,
            Some((
                block_index,
                3,
                chunk.source_start_char,
                chunk.source_end_char,
            )),
        );
    }
    if !pushed && message.attachments.is_empty() {
        push_user_card_content_line_into(
            out,
            vec![Span::styled(String::new(), user_prompt_panel_style(theme))],
            None,
            width,
            theme,
            Some((block_index, 3, 0, 0)),
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
            transcript_attachment_item_spans(index, &attachment_text, theme),
            None,
            width,
            theme,
            Some((attachment_block, 5, 0, attachment_text.chars().count())),
        );
    }

    if message.queued {
        push_user_card_line_into(out, "", Some("QUEUED"), width, theme, None);
    }

    // 底部空 card 行（decoration）
    push_user_card_line_into(out, "", None, width, theme, None);
}

/// 与 `push_user_card_line` 等价的构造，但写入 `RenderedTimelineItem`，
/// 同时把 origin 元数据与 line 配对。
/// `origin` 为 `None` 表示装饰行（顶部 spacer / QUEUED badge / 底部 spacer），
/// 否则为 `(block_index, content_prefix_chars, source_start, source_end)`。
fn push_user_card_line_into(
    out: &mut RenderedTimelineItem,
    content: &str,
    badge: Option<&str>,
    width: usize,
    theme: Theme,
    origin: Option<(usize, usize, usize, usize)>,
) {
    push_user_card_content_line_into(
        out,
        vec![Span::styled(
            content.to_string(),
            user_prompt_panel_style(theme),
        )],
        badge,
        width,
        theme,
        origin,
    );
}

fn push_user_card_content_line_into(
    out: &mut RenderedTimelineItem,
    content_spans: Vec<Span<'static>>,
    badge: Option<&str>,
    width: usize,
    theme: Theme,
    origin: Option<(usize, usize, usize, usize)>,
) {
    let bar_style = surface::accent_style(
        theme,
        surface::SurfaceEmphasis::User,
        surface::SurfaceKind::Root,
    );
    let pad_style = user_prompt_padding_style(theme);
    let badge_style = queued_badge_style(theme);

    let mut spans = vec![
        Span::styled(surface::ACCENT_BAR_GLYPH, bar_style),
        Span::styled("  ", pad_style),
    ];

    if let Some(badge) = badge {
        spans.push(Span::styled(" ", badge_style));
        spans.push(Span::styled(badge.to_string(), badge_style));
        spans.push(Span::styled(" ", badge_style));
        spans.push(Span::styled(" ", pad_style));
    }

    spans.extend(content_spans);

    let mut line = Line::from(spans);

    let used = display_width(&line.to_string());
    if width > used {
        line.spans
            .push(Span::styled(" ".repeat(width - used), pad_style));
    } else {
        line.spans.push(Span::styled("  ", pad_style));
    }

    match origin {
        Some((block_index, prefix_chars, src_start, src_end)) => {
            let content_char_len = src_end.saturating_sub(src_start);
            out.push_content(line, block_index, prefix_chars, src_start, content_char_len);
        }
        None => out.push_decoration(line),
    }
}

fn build_assistant_message_lines(
    out: &mut RenderedTimelineItem,
    text: &str,
    _streaming: bool,
    theme: Theme,
    width: usize,
) {
    let content_width = width.saturating_sub(2).max(1);
    let lines: Vec<Line<'static>> = if text.is_empty() {
        vec![Line::from(Span::styled("  …", root_muted_style(theme)))]
    } else {
        render_markdown(text, theme, MarkdownRenderOptions::new(content_width))
            .into_iter()
            .map(|rendered| {
                let mut spans = vec![Span::styled("  ", theme.app_style())];
                spans.extend(rendered.spans);
                Line::from(spans)
            })
            .collect()
    };

    if text.is_empty() {
        out.extend_decoration_from(lines);
        return;
    }

    let semantic = render_markdown_semantic_blocks(text, content_width);
    let line_count = lines.len();
    if semantic.line_origins.len() != line_count {
        // assistant semantic-copy 的前提是：semantic line origins 与真实 rendered lines
        // 一一对齐。若某些 markdown 结构（尤其 code block / table 等）在 wrap 宽度
        // 上仍有 1 行级偏差，强行套 origin 会把整段选择带歪。这里先安全回退到
        // legacy 行级复制，避免“各种错位”。后续再逐类补齐精确映射。
        out.extend_legacy_rendered_from(lines);
        return;
    }
    let block_base = out.source_blocks.len();

    out.source_blocks.extend(
        semantic
            .source_blocks
            .into_iter()
            .map(|block| RenderedSourceBlock {
                source: block.source,
            }),
    );

    out.lines.extend(lines);
    for origin in semantic.line_origins.into_iter().take(line_count) {
        out.line_origins.push(RenderedLineOrigin {
            block_index: origin.block_index.map(|idx| block_base + idx),
            content_prefix_chars: origin
                .block_index
                .map(|_| origin.content_prefix_chars.saturating_add(2))
                .unwrap_or_default(),
            content_char_offset: origin.content_char_offset,
            content_char_len: origin.content_char_len,
        });
    }
    while out.line_origins.len() < out.lines.len() {
        out.line_origins.push(RenderedLineOrigin::default());
    }
}

fn build_tool_lines(
    out: &mut RenderedTimelineItem,
    tool: &ToolView,
    theme: Theme,
    width: usize,
    frame: usize,
    expanded_output: bool,
) {
    // TODO(P1): tool card 输出来源映射。
    let lines =
        tool_card::render_tool_card_lines_with_frame(tool, theme, width, frame, expanded_output);
    out.extend_legacy_rendered_from(lines);
}

fn build_todo_card(
    out: &mut RenderedTimelineItem,
    todo: &crate::tui::timeline::TodoView,
    theme: Theme,
    width: usize,
) {
    // TODO(P1): todo item 内容映射。
    let lines = todo_card::render_todo_card_lines(todo, theme, width);
    out.extend_legacy_rendered_from(lines);
}

fn build_permission_lines(
    out: &mut RenderedTimelineItem,
    permission: &PermissionView,
    theme: Theme,
    width: usize,
) {
    // TODO(P1): permission 卡输出映射。
    if permission.status == PermissionPromptStatus::Pending {
        return;
    }
    let lines = tool_card::render_permission_card_lines(permission, theme, width);
    out.extend_legacy_rendered_from(lines);
}

fn build_error_lines(
    out: &mut RenderedTimelineItem,
    error: &ErrorView,
    theme: Theme,
    width: usize,
) {
    // TODO(P1): error 卡输出映射；message / details 应可作为 source block。
    let accent = theme.error;
    let bg = theme.elevated_bg;
    let mut lines: Vec<Line<'static>> = Vec::new();
    push_card_blank_line(&mut lines, accent, bg, theme, width);
    push_wrapped_error_card_line(
        &mut lines,
        &error.message,
        accent,
        elevated_error_message_style(theme),
        theme,
        width,
    );
    if let Some(details) = error
        .details
        .as_deref()
        .filter(|details| !details.is_empty())
    {
        push_card_blank_line(&mut lines, accent, bg, theme, width);
        push_wrapped_error_card_line(
            &mut lines,
            details,
            accent,
            elevated_error_detail_style(theme),
            theme,
            width,
        );
    }
    push_card_blank_line(&mut lines, accent, bg, theme, width);
    out.extend_legacy_rendered_from(lines);
}

fn build_compaction_separator_line(
    out: &mut RenderedTimelineItem,
    label: &str,
    theme: Theme,
    width: usize,
) {
    if width == 0 {
        return;
    }
    out.push_decoration(Line::from(Span::styled(
        tool_card::truncate_display_width(label, width),
        root_muted_style(theme),
    )));
}

fn push_wrapped_error_card_line(
    lines: &mut Vec<Line<'static>>,
    content: &str,
    accent: ratatui::style::Color,
    value_style: ratatui::style::Style,
    theme: Theme,
    width: usize,
) {
    if width == 0 {
        return;
    }
    if width == 1 {
        lines.push(Line::from(Span::styled(
            surface::ACCENT_BAR_GLYPH,
            card_bar_style(accent, theme.root_bg),
        )));
        return;
    }

    let has_padding = width > 2;
    let content_width = width.saturating_sub(1 + usize::from(has_padding)).max(1);
    for wrapped in wrap_text_to_width(content, content_width) {
        let mut spans = vec![Span::styled(
            surface::ACCENT_BAR_GLYPH,
            card_bar_style(accent, theme.root_bg),
        )];
        if has_padding {
            spans.push(Span::styled(" ", value_style));
        }
        spans.push(Span::styled(wrapped, value_style));
        let mut line = Line::from(spans);
        pad_card_line_to_width(&mut line, width, value_style);
        lines.push(line);
    }
}

fn push_card_blank_line(
    lines: &mut Vec<Line<'static>>,
    accent: ratatui::style::Color,
    bg: ratatui::style::Color,
    theme: Theme,
    width: usize,
) {
    if width == 0 {
        return;
    }

    let fill_style = ratatui::style::Style::default().bg(bg);
    let mut line = Line::from(vec![Span::styled(
        surface::ACCENT_BAR_GLYPH,
        card_bar_style(accent, theme.root_bg),
    )]);
    pad_card_line_to_width(&mut line, width, fill_style);
    lines.push(line);
}

fn pad_card_line_to_width(
    line: &mut Line<'static>,
    width: usize,
    fill_style: ratatui::style::Style,
) {
    let used = display_width(&line.to_string());
    if width > used {
        line.spans
            .push(Span::styled(" ".repeat(width - used), fill_style));
    }
}

fn card_bar_style(
    accent: ratatui::style::Color,
    bg: ratatui::style::Color,
) -> ratatui::style::Style {
    ratatui::style::Style::default().fg(accent).bg(bg)
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

fn transcript_attachment_item_spans(
    index: usize,
    attachment_text: &str,
    theme: Theme,
) -> Vec<Span<'static>> {
    let title = format!("image {}", index + 1);
    let detail = attachment_text
        .strip_prefix(&title)
        .unwrap_or(attachment_text)
        .trim_start()
        .trim_start_matches('·')
        .trim_start()
        .to_string();

    let mut spans = vec![
        Span::styled("↳ ", user_prompt_padding_style(theme)),
        Span::styled(title, attachment_block_title_style(theme)),
    ];
    if !detail.is_empty() {
        spans.push(Span::styled(" · ", user_prompt_padding_style(theme)));
        spans.push(Span::styled(detail, attachment_block_item_style(theme)));
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

fn element_muted_style(theme: Theme) -> ratatui::style::Style {
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
        .fg(theme.accent)
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
        if local_line_index >= cache.entries()[item_index].line_origins.len() {
            continue;
        }
        let origin = &cache.entries()[item_index].line_origins[local_line_index];
        if origin.block_index.is_none() {
            continue;
        }

        // 计算该行在“渲染行字符坐标系”中的选择范围：prefix 之后才是可选内容。
        let (content_start, content_end) =
            if absolute_row == sel_start_row && absolute_row == sel_end_row {
                (start.char_offset, end.char_offset)
            } else if absolute_row == sel_start_row {
                (start.char_offset, origin.content_char_len)
            } else if absolute_row == sel_end_row {
                (0, end.char_offset)
            } else {
                (0, origin.content_char_len)
            };
        let char_start = origin
            .content_prefix_chars
            .saturating_add(content_start.min(origin.content_char_len));
        let char_end = origin
            .content_prefix_chars
            .saturating_add(content_end.min(origin.content_char_len));
        if char_start >= char_end {
            continue;
        }

        // 重新构建带高亮的 Line
        *line = highlight_line_spans(line.clone(), char_start, char_end, selection_style);
    }
}

/// 高亮 Line 中指定字符范围的 Spans
fn highlight_line_spans(
    line: Line<'static>,
    char_start: usize,
    char_end: usize,
    selection_style: ratatui::style::Style,
) -> Line<'static> {
    use ratatui::text::Span;

    let mut new_spans = Vec::new();
    let mut current_offset = 0;

    for span in line.spans {
        let span_chars: Vec<char> = span.content.chars().collect();
        let span_len = span_chars.len();
        let span_end = current_offset + span_len;

        if span_end <= char_start || current_offset >= char_end {
            // 完全不在选择范围内
            new_spans.push(span);
        } else {
            // 需要拆分 Span：前缀 + 高亮部分 + 后缀
            let hl_start = char_start.saturating_sub(current_offset);
            let hl_end = char_end.saturating_sub(current_offset).min(span_len);

            // 前缀（未选中部分）
            if hl_start > 0 {
                let prefix: String = span_chars[..hl_start].iter().collect();
                new_spans.push(Span::styled(prefix, span.style));
            }

            // 高亮部分
            let highlighted: String = span_chars[hl_start..hl_end].iter().collect();
            // 合并原有样式和选择样式
            let combined_style = span.style.patch(selection_style);
            new_spans.push(Span::styled(highlighted, combined_style));

            // 后缀（未选中部分）
            if hl_end < span_len {
                let suffix: String = span_chars[hl_end..].iter().collect();
                new_spans.push(Span::styled(suffix, span.style));
            }
        }

        current_offset = span_end;
    }

    Line::from(new_spans)
}

#[cfg(test)]
mod tests {
    use super::{
        cached_transcript_row_count, render_transcript, transcript_lines, transcript_row_count,
        visible_cached_transcript_lines, visible_transcript_lines,
    };
    use crate::{
        agent::{AutoContinueState, TodoItem, TodoStatus},
        context_view::DEFAULT_FOLDED_OUTPUT_THRESHOLD_BYTES,
        tool::ToolResult,
        transcript::{TranscriptEvent, TranscriptRecord},
        tui::{
            AppEvent, AssistantDeltaEvent, ErrorEvent, PermissionRequestEvent, ReasoningDeltaEvent,
            ReasoningDoneEvent, ToolFinishedEvent, ToolOutcome, ToolStartedEvent, UserMessageEvent,
            events::{AutoContinueChangedEvent, TodoSnapshotEvent},
            state::{ContextDetailTarget, TuiState},
            theme::Theme,
            timeline::Timeline,
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

    fn record(sequence: u64, event: TranscriptEvent) -> TranscriptRecord {
        TranscriptRecord {
            session_id: "s".into(),
            sequence,
            timestamp_ms: sequence as u128,
            context_branch_id: None,
            event,
        }
    }

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
                            "stdout": "x".repeat(DEFAULT_FOLDED_OUTPUT_THRESHOLD_BYTES + 32),
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
    fn transcript_rows_wrap_using_display_width() {
        let mut state = TuiState::default();
        state.apply_event(crate::tui::events::AppEvent::UserMessage(
            UserMessageEvent::new("a你b"),
        ));

        let theme = Theme::dark();
        let lines = transcript_lines(&state, theme, 6);

        assert_eq!(transcript_row_count(&state, theme, 6), lines.len());
        assert_eq!(lines.len(), 6);
        assert!(lines.iter().any(|line| line.to_string().contains('你')));
    }

    #[test]
    fn tool_permission_and_error_cards_wrap_to_target_width() {
        let mut state = TuiState::default();
        state.apply_event(AppEvent::UserMessage(UserMessageEvent::new("seed")));

        let mut tool_started = ToolStartedEvent::new("call-tool", "shell__exec", "run");
        tool_started.arguments = Some("--really-long-arg ".repeat(20));
        state.apply_event(AppEvent::ToolStarted(tool_started));
        let mut tool_finished =
            ToolFinishedEvent::new("call-tool", "shell__exec", "run", ToolOutcome::Failure);
        tool_finished.output = Some("output=".to_string() + &"x".repeat(200));
        state.apply_event(AppEvent::ToolFinished(tool_finished));

        let mut request = PermissionRequestEvent::new("call-perm", "shell__exec", "needs approval");
        request.arguments = Some("arg ".repeat(60));
        request.rationale = Some("because ".repeat(80));
        state.apply_event(AppEvent::PermissionRequested(request));

        let mut err = ErrorEvent::new("boom");
        err.details = Some("detail ".repeat(90));
        state.apply_event(AppEvent::Error(err));

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
    fn error_card_is_message_first_and_uses_semantic_error_styling() {
        let mut state = TuiState::default();
        let mut error = ErrorEvent::new("stream stopped");
        error.details = Some("retry after backoff".into());
        state.apply_event(AppEvent::Error(error));

        let theme = Theme::dark();
        let lines = transcript_lines(&state, theme, 64);
        let error_line = lines
            .iter()
            .find(|line| line.to_string().contains("stream stopped"))
            .expect("error line renders");
        let guide = error_line.spans.first().expect("error line has guide");

        assert_eq!(
            guide.content.as_ref(),
            crate::tui::surface::ACCENT_BAR_GLYPH
        );
        assert_eq!(guide.style.fg, Some(theme.error));
        assert_eq!(guide.style.bg, Some(theme.root_bg));

        let card_pad = error_line.spans.get(1).expect("error line has card pad");
        assert_eq!(card_pad.content.as_ref(), " ");
        assert_eq!(card_pad.style.bg, Some(theme.elevated_bg));
        let message = error_line.spans.get(2).expect("error message renders");
        assert_eq!(message.style.fg, Some(theme.error));
        assert_eq!(message.style.bg, Some(theme.elevated_bg));
        assert!(
            message
                .style
                .add_modifier
                .contains(ratatui::style::Modifier::BOLD)
        );

        let error_index = lines
            .iter()
            .position(|line| line.to_string().contains("stream stopped"))
            .expect("error line index");
        assert!(error_index > 0, "error card has top padding row");
        let top_pad = &lines[error_index - 1];
        let bottom_pad = lines.last().expect("error card has bottom padding row");
        for pad in [top_pad, bottom_pad] {
            assert_eq!(pad.spans[0].style.fg, Some(theme.error));
            assert_eq!(pad.spans[0].style.bg, Some(theme.root_bg));
            assert!(pad.spans[1].content.as_ref().starts_with(' '));
            assert_eq!(pad.spans[1].style.bg, Some(theme.elevated_bg));
        }

        let rendered = lines
            .iter()
            .map(|line| line.to_string())
            .collect::<String>();
        assert!(rendered.contains("stream stopped"));
        assert!(rendered.contains("retry after backoff"));
        assert!(!rendered.contains("error stream stopped"));
        assert!(!rendered.contains("details"));
    }

    #[test]
    fn error_card_stays_within_narrow_widths() {
        let mut state = TuiState::default();
        let mut error = ErrorEvent::new("stream stopped");
        error.details = Some("retry after backoff".into());
        state.apply_event(AppEvent::Error(error));

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
    fn todo_timeline_items_render_full_card_sections() {
        let mut state = TuiState::default();
        state.apply_event(AppEvent::AutoContinueChanged(
            AutoContinueChangedEvent::new(AutoContinueState {
                enabled: true,
                max_continuations: 2,
            }),
        ));
        state.apply_event(AppEvent::TodoSnapshot(TodoSnapshotEvent::new(vec![
            TodoItem {
                id: "t1".into(),
                content: "Inspect timeline integration".into(),
                status: TodoStatus::InProgress,
            },
            TodoItem {
                id: "t2".into(),
                content: "Keep wrapping stable at narrow widths".into(),
                status: TodoStatus::Pending,
            },
            TodoItem {
                id: "t3".into(),
                content: "Snapshot final layout".into(),
                status: TodoStatus::Completed,
            },
        ])));

        let lines = transcript_lines(&state, Theme::dark(), 56)
            .into_iter()
            .map(|line| line.to_string())
            .collect::<Vec<_>>();
        let joined = lines.join("\n");

        assert!(joined.contains("# Todos"));
        assert!(joined.contains("[•] Inspect timeline integration"));
        assert!(joined.contains("[ ] Keep wrapping stable at narrow widths"));
        assert!(joined.contains("[✓] Snapshot final layout"));
        assert!(!joined.contains("auto on"));
        assert!(!joined.contains("current"));
        assert!(!joined.contains("items · auto-continue"));

        for rendered in lines {
            let measured = crate::tui::measure::display_width(&rendered);
            assert!(measured <= 56, "line width {measured} > 56: {rendered:?}");
        }
    }

    #[test]
    fn todo_cards_do_not_get_extra_timeline_separator() {
        let mut state = TuiState::default();
        state.apply_event(AppEvent::TodoSnapshot(TodoSnapshotEvent::new(vec![
            TodoItem {
                id: "t1".into(),
                content: "First snapshot".into(),
                status: TodoStatus::Completed,
            },
        ])));
        state.apply_event(AppEvent::TodoSnapshot(TodoSnapshotEvent::new(vec![
            TodoItem {
                id: "t1".into(),
                content: "Second snapshot".into(),
                status: TodoStatus::InProgress,
            },
        ])));

        let lines = transcript_lines(&state, Theme::dark(), 56)
            .into_iter()
            .map(|line| line.to_string())
            .collect::<Vec<_>>();
        let title_indices = lines
            .iter()
            .enumerate()
            .filter_map(|(index, line)| line.contains("# Todos").then_some(index))
            .collect::<Vec<_>>();
        assert_eq!(title_indices.len(), 2, "{lines:?}");

        let between_cards = &lines[title_indices[0] + 1..title_indices[1]];
        assert!(
            !between_cards.iter().any(|line| line.is_empty()),
            "unexpected blank timeline separator between todo cards: {lines:?}"
        );
    }

    #[test]
    fn committed_compaction_renders_durable_block_with_summary() {
        let mut state = TuiState::default();
        state
            .timeline
            .push_restored_compaction("Earlier context was summarized here.");

        let lines = transcript_lines(&state, Theme::dark(), 80)
            .into_iter()
            .map(|line| line.to_string())
            .collect::<Vec<_>>();
        let nonempty_lines = lines
            .iter()
            .filter(|line| !line.is_empty())
            .map(String::as_str)
            .collect::<Vec<_>>();

        assert!(
            nonempty_lines
                .iter()
                .any(|line| *line == crate::tui::timeline::COMPACTION_TRIGGER_LABEL),
            "{lines:?}"
        );
        assert_eq!(
            nonempty_lines
                .iter()
                .filter(|line| **line == crate::tui::timeline::COMPACTION_SEPARATOR_LABEL)
                .count(),
            2,
            "expected top and bottom rules: {lines:?}"
        );
        assert!(
            nonempty_lines
                .iter()
                .any(|line| line.contains("Earlier context was summarized")),
            "{lines:?}"
        );
        assert!(
            nonempty_lines
                .iter()
                .any(|line| *line == crate::tui::timeline::COMPACTION_COMPLETE_LABEL),
            "{lines:?}"
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
    fn cached_visible_transcript_matches_full_transcript_window() {
        let mut state = TuiState::default();
        state.apply_event(AppEvent::UserMessage(UserMessageEvent::new("seed")));
        for index in 0..30 {
            state
                .timeline
                .push_assistant_delta(AssistantDeltaEvent::new(format!("history line {index}")));
        }
        state.apply_event(AppEvent::AssistantDelta(AssistantDeltaEvent::new(
            "# Heading\n```rust\nlet value = 42;\n```\n- done",
        )));
        state.apply_event(AppEvent::AssistantDone { message_id: None });

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
    fn transcript_cache_invalidates_streaming_assistant_item() {
        let mut state = TuiState::default();
        state.apply_event(AppEvent::AssistantDelta(AssistantDeltaEvent::new("first")));

        let theme = Theme::dark();
        let width = 80;
        let before_rows = cached_transcript_row_count(&mut state, theme, width);
        let before_revision = state.transcript_render_cache.entries[0].revision;

        state.apply_event(AppEvent::AssistantDelta(AssistantDeltaEvent::new(
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

        state.apply_event(AppEvent::AssistantDelta(AssistantDeltaEvent::new(
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
        state.apply_event(AppEvent::PermissionRequested(request));

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
    fn delegation_item_renders_as_dedicated_transcript_line() {
        let mut state = TuiState::default();
        state.timeline.push_delegation("fixer", "fix failing test");

        let rendered = transcript_lines(&state, Theme::dark(), 80)
            .into_iter()
            .map(|line| line.to_string())
            .collect::<Vec<_>>()
            .join("\n");

        assert!(rendered.contains("delegate @fixer"), "{rendered}");
        assert!(rendered.contains("fix failing test"), "{rendered}");
    }

    #[test]
    fn subagent_parent_transcript_stays_compact_and_keeps_child_details_out() {
        let mut state = TuiState::default();
        state.apply_event(AppEvent::ToolStarted(ToolStartedEvent {
            call_id: "run-1".into(),
            name: "agent__explore".into(),
            summary: "explorer running · child-sessio".into(),
            arguments: Some(serde_json::json!({"task":"inspect runner flow"}).to_string()),
        }));
        state.apply_event(AppEvent::ToolFinished(ToolFinishedEvent {
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
    fn fixer_parent_transcript_stays_compact_and_keeps_child_details_out() {
        let mut state = TuiState::default();
        state.apply_event(AppEvent::ToolStarted(ToolStartedEvent {
            call_id: "run-2".into(),
            name: "agent__fixer".into(),
            summary: "fixer running · child-sessio".into(),
            arguments: Some(serde_json::json!({"task":"wire fixer tool"}).to_string()),
        }));
        state.apply_event(AppEvent::ToolFinished(ToolFinishedEvent {
            call_id: "run-2".into(),
            name: "agent__fixer".into(),
            summary: "fixer completed · child-sessio".into(),
            outcome: ToolOutcome::Success,
            output: Some(
                serde_json::json!({
                    "ok": true,
                    "tool": "agent__fixer",
                    "data": {
                        "agent_name": "fixer",
                        "status": "completed",
                        "summary": "wired fixer tool end to end",
                        "full_summary": "wired fixer tool end to end\nchild details stay hidden",
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
            .filter(|line| line.contains("fixer") || line.contains("/child"))
            .collect::<Vec<_>>();

        assert_eq!(agent_lines.len(), 1, "{non_empty:?}");
        assert!(
            agent_lines[0].contains("completed fixer wired fixer tool end to end · /child"),
            "{}",
            agent_lines[0]
        );
        assert!(
            !non_empty
                .iter()
                .any(|line| line.contains("child details stay hidden")),
            "{non_empty:?}"
        );
    }

    #[test]
    fn reasoning_content_renders_inline_in_transcript() {
        let mut state = TuiState::default();
        state.apply_event(AppEvent::ReasoningDelta(ReasoningDeltaEvent::new(
            "r-1",
            "Inspecting workflow",
        )));
        state.apply_event(AppEvent::ReasoningDone(ReasoningDoneEvent::new(
            "r-1",
            "Inspecting workflow",
        )));

        let lines = transcript_lines(&state, Theme::dark(), 60)
            .into_iter()
            .map(|line| line.to_string())
            .collect::<Vec<_>>();

        assert!(
            lines.iter().any(|line| line.contains("Thought")),
            "{lines:?}"
        );
        assert!(
            lines
                .iter()
                .any(|line| line.contains("Inspecting workflow")),
            "{lines:?}"
        );
    }

    #[test]
    fn reasoning_title_strips_markdown_and_body_is_indented() {
        let mut state = TuiState::default();
        state.apply_event(AppEvent::ReasoningDone(ReasoningDoneEvent::new(
            "r-1",
            "**Evaluating code status**\n\nI need to check `git diff` output.",
        )));

        let lines = transcript_lines(&state, Theme::dark(), 80)
            .into_iter()
            .map(|line| line.to_string())
            .collect::<Vec<_>>();

        assert!(
            lines
                .iter()
                .any(|line| line.contains("Thought: Evaluating code status")),
            "{lines:?}"
        );
        assert!(!lines.iter().any(|line| line.contains("**")), "{lines:?}");
        assert!(
            lines
                .iter()
                .any(|line| line.contains("I need to check git diff output.")),
            "{lines:?}"
        );
    }

    #[test]
    fn assistant_markdown_is_rendered_as_formatted_plain_tui_lines() {
        let mut state = TuiState::default();
        state.apply_event(AppEvent::AssistantDelta(
            crate::tui::events::AssistantDeltaEvent::new(
                "# Title\n- **item** with `code`\n```\nlet x = 1;\n```",
            ),
        ));
        state.apply_event(AppEvent::AssistantDone { message_id: None });

        let lines = transcript_lines(&state, Theme::dark(), 80)
            .into_iter()
            .map(|line| line.to_string())
            .collect::<Vec<_>>()
            .join("\n");

        assert!(lines.contains("▌ Title"), "{lines}");
        assert!(lines.contains("• item with code"), "{lines}");
        assert!(lines.contains("let x = 1;"), "{lines}");
        assert!(!lines.contains("**item**"), "{lines}");
        assert!(!lines.contains("```"), "{lines}");
    }

    #[test]
    fn streaming_compaction_renders_trigger_rules_and_preview_body() {
        let mut state = TuiState::default();
        state.apply_event(AppEvent::CompactionStarted);
        state.apply_event(AppEvent::CompactionPreviewDelta {
            delta: "A transient summary preview".into(),
        });

        let lines = transcript_lines(&state, Theme::dark(), 80)
            .into_iter()
            .map(|line| line.to_string())
            .collect::<Vec<_>>();
        assert!(
            lines
                .iter()
                .any(|line| line == crate::tui::timeline::COMPACTION_TRIGGER_LABEL)
        );
        assert!(
            lines
                .iter()
                .any(|line| line.contains("A transient summary preview"))
        );
        // Complete label only after commit.
        assert!(
            lines
                .iter()
                .all(|line| line != crate::tui::timeline::COMPACTION_COMPLETE_LABEL)
        );

        let narrow = transcript_lines(&state, Theme::dark(), 10)
            .into_iter()
            .map(|line| line.to_string())
            .collect::<Vec<_>>();
        assert!(narrow.iter().all(|line| line.chars().count() <= 10));
    }

    #[test]
    fn assistant_reasoning_and_tool_trace_share_left_indent() {
        let mut state = TuiState::default();
        state.apply_event(AppEvent::ReasoningDone(ReasoningDoneEvent::new(
            "r-1",
            "Answering\nThinking body",
        )));
        state.apply_event(AppEvent::AssistantDelta(AssistantDeltaEvent::new(
            "# Title",
        )));
        state.apply_event(AppEvent::AssistantDone { message_id: None });
        state.apply_event(AppEvent::ToolStarted(ToolStartedEvent::new(
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
        state.apply_event(AppEvent::UserMessage(UserMessageEvent::new("hello")));

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
        state.apply_event(AppEvent::UserMessage(UserMessageEvent::queued("follow up")));

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
    fn user_message_renders_attachment_placeholders_beneath_body() {
        let mut state = TuiState::default();
        state.apply_event(AppEvent::UserMessage(UserMessageEvent::from_submission(
            UserMessageSubmission::new(
                "user-with-images",
                UserMessageContent::new(
                    "describe this",
                    vec![
                        test_attachment("img-1", "clipboard"),
                        test_attachment("img-2", "diagram.png"),
                    ],
                ),
            ),
        )));

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
    fn context_is_not_rendered_but_remains_available_to_the_inspector() {
        let mut state = TuiState::default();
        state.replace_session_timeline_from_records(&context_records());

        let rendered = transcript_lines(&state, Theme::dark(), 80)
            .into_iter()
            .map(|line| line.to_string())
            .collect::<Vec<_>>()
            .join("\n");
        assert!(!rendered.contains("Active · Investigate"), "{rendered}");
        assert!(!rendered.contains("Pinned"), "{rendered}");

        let items = crate::tui::state::context_dialog_items(state.active_context());
        assert!(items.iter().any(|item| item.id == "node:node-a"));
        assert!(
            items
                .iter()
                .any(|item| item.id == "folded:folded-output-seq-9-stdout")
        );

        state.open_context_detail(Some(ContextDetailTarget::Summary("sum-1".into())));
        let detail = state.active_context_open_detail().expect("summary detail");
        assert_eq!(detail.title, "Summary sum-1");
        assert!(detail.lines.iter().any(|line| line == "Source · node-a"));
        assert!(
            detail
                .lines
                .iter()
                .any(|line| line == "Block · block-seq-4-note")
        );
    }

    #[test]
    fn context_folded_output_remains_available_to_the_inspector() {
        let mut state = TuiState::default();
        state.replace_session_timeline_from_records(&context_records());
        state.open_context_detail(Some(ContextDetailTarget::FoldedOutput(
            "folded-output-seq-9-stdout".into(),
        )));

        let detail = state
            .active_context_open_detail()
            .expect("folded output detail");
        assert_eq!(detail.title, "Folded output folded-output-seq-9-stdout");
        assert!(detail.lines.iter().any(|line| line == "Tool · shell__exec"));
        assert!(
            detail
                .lines
                .iter()
                .any(|line| line.starts_with("Open detail ·"))
        );
    }

    #[test]
    fn removed_folded_output_is_hidden_from_context_inspector() {
        let mut records = context_records();
        records.push(record(
            11,
            TranscriptEvent::ContextViewOperationMetadata {
                operation: "remove_from_view".into(),
                node_id: None,
                block_id: Some("block-seq-9-folded-output-folded-output-seq-9-stdout".into()),
                detail: None,
            },
        ));

        let mut state = TuiState::default();
        state.replace_session_timeline_from_records(&records);
        state.open_context_detail(Some(ContextDetailTarget::FoldedOutput(
            "folded-output-seq-9-stdout".into(),
        )));

        assert!(state.active_context_open_detail().is_none());
    }
}
