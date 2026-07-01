use super::components::transcript::TranscriptRenderCache;
use super::events::{
    AppEvent, AutoContinueChangedEvent, PermissionDecision, PermissionRequestEvent,
    PermissionResolutionEvent, TokenUsageEvent, ToolOutcome, UserMessageEvent,
};
use super::measure;
use super::slash;
use super::timeline::{
    ContextBlockLineView, ContextNodeLineView, ContextOpenDetailView, ContextTimelineView,
    PermissionView, Timeline, TodoView,
};
use crate::agent::{AutoContinueState, ConversationMessage};
use crate::context_tree::{ContextNodeStatus, ContextTreeState};
use crate::context_view::{
    self, ContextBlock, ContextBlockSource, ContextViewProjection, ContextViewStatus,
    FoldedOutputMetadata,
};
use crate::transcript::transcript_projection;
use crate::transcript::{
    TranscriptEvent, TranscriptRecord, restore_latest_auto_continue_state,
    restore_latest_todo_snapshot,
};
use crate::user_content::{UserImageAttachment, UserMessageContent, UserMessageSubmission};
use anyhow::Result;

/// 文本选择范围
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TextSelection {
    pub start: SelectionAnchor,
    pub end: SelectionAnchor,
}

impl TextSelection {
    /// 规范化选择方向（确保 start 在 end 之前）
    pub fn normalize(&self) -> (SelectionAnchor, SelectionAnchor) {
        if self.start <= self.end {
            (self.start.clone(), self.end.clone())
        } else {
            (self.end.clone(), self.start.clone())
        }
    }
}

/// 选择锚点：定位到 transcript 中的具体字符位置
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SelectionAnchor {
    /// Timeline 中的 item 索引
    pub item_index: usize,
    /// Item 内的渲染行偏移
    pub rendered_line_offset: usize,
    /// 行内字符偏移（Unicode 字符计数）
    pub char_offset: usize,
}

impl PartialOrd for SelectionAnchor {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for SelectionAnchor {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.item_index
            .cmp(&other.item_index)
            .then(self.rendered_line_offset.cmp(&other.rendered_line_offset))
            .then(self.char_offset.cmp(&other.char_offset))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum AppPhase {
    #[default]
    Idle,
    Editing,
    Running,
    WaitingForPermission,
    Completed,
    Error,
    Quitting,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FooterStatus {
    pub summary: String,
    pub detail: Option<String>,
    pub is_error: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ToastKind {
    Info,
    Success,
    Error,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ToastState {
    pub message: String,
    pub kind: ToastKind,
    ticks_remaining: u8,
}

impl ToastState {
    pub const DEFAULT_TICKS: u8 = 54;

    pub fn new(message: impl Into<String>, kind: ToastKind, ticks_remaining: u8) -> Self {
        Self {
            message: message.into(),
            kind,
            ticks_remaining,
        }
    }

    pub fn ticks_remaining(&self) -> u8 {
        self.ticks_remaining
    }

    fn tick(&mut self) -> bool {
        if self.ticks_remaining > 0 {
            self.ticks_remaining -= 1;
        }

        self.ticks_remaining == 0
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ModelTokenUsage {
    pub used_tokens: u64,
    pub context_window_tokens: u64,
    pub input_tokens: u64,
    pub output_tokens: u64,
    pub cached_tokens: u64,
}

impl Default for FooterStatus {
    fn default() -> Self {
        Self {
            summary: "Ready".into(),
            detail: Some("Ctrl-C or q to quit once keybindings are wired".into()),
            is_error: false,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DialogItem {
    pub id: String,
    pub label: String,
    pub detail: Option<String>,
    pub section: Option<String>,
    pub right_detail: Option<String>,
}

impl DialogItem {
    pub fn new(id: impl Into<String>, label: impl Into<String>, detail: Option<String>) -> Self {
        Self {
            id: id.into(),
            label: label.into(),
            detail,
            section: None,
            right_detail: None,
        }
    }

    pub fn with_section(mut self, section: impl Into<String>) -> Self {
        self.section = Some(section.into());
        self
    }

    pub fn with_right_detail(mut self, right_detail: impl Into<String>) -> Self {
        self.right_detail = Some(right_detail.into());
        self
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DialogKind {
    ModelPicker,
    PermissionPicker,
    ReasoningPicker,
    SessionPicker,
    BranchPicker,
    ContextPicker,
    ContextDetail,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ContextDetailTarget {
    Node(String),
    Block(String),
    Summary(String),
    FoldedOutput(String),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ContextPaneState {
    pub tree: ContextTreeState,
    pub view: ContextViewProjection,
    pub open_detail: Option<ContextDetailTarget>,
}

impl Default for ContextPaneState {
    fn default() -> Self {
        Self {
            tree: ContextTreeState::with_default_root(),
            view: ContextViewProjection::default(),
            open_detail: None,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TranscriptViewState {
    Parent,
    Child {
        parent_session_id: String,
        child_session_id: String,
        agent_name: String,
        index: usize,
        total: usize,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChildViewMetadata {
    pub parent_session_id: String,
    pub child_session_id: String,
    pub agent_name: String,
    pub index: usize,
    pub total: usize,
    pub model: Option<String>,
    pub record_count: usize,
}

impl TranscriptViewState {
    pub fn is_child(&self) -> bool {
        matches!(self, Self::Child { .. })
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct ChildTranscriptState {
    timeline: Timeline,
    model: Option<String>,
    record_count: usize,
    live_streaming: bool,
    context: ContextPaneState,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DialogState {
    pub kind: DialogKind,
    pub title: String,
    pub description: Option<String>,
    pub items: Vec<DialogItem>,
    pub selected: usize,
    pub query: String,
}

impl DialogState {
    pub fn new(
        kind: DialogKind,
        title: impl Into<String>,
        description: Option<String>,
        items: Vec<DialogItem>,
    ) -> Self {
        Self {
            kind,
            title: title.into(),
            description,
            items,
            selected: 0,
            query: String::new(),
        }
    }

    pub fn select_next(&mut self) {
        let visible = self.visible_indices();
        if visible.is_empty() {
            self.selected = 0;
            return;
        }

        let current = visible
            .iter()
            .position(|index| *index == self.selected)
            .unwrap_or(0);
        self.selected = visible[(current + 1) % visible.len()];
    }

    pub fn select_previous(&mut self) {
        let visible = self.visible_indices();
        if visible.is_empty() {
            self.selected = 0;
            return;
        }

        let current = visible
            .iter()
            .position(|index| *index == self.selected)
            .unwrap_or(0);
        self.selected = if current == 0 {
            *visible.last().expect("visible indices should not be empty")
        } else {
            visible[current - 1]
        };
    }

    pub fn insert_query_char(&mut self, ch: char) {
        self.query.push(ch);
        self.clamp_selection_to_visible();
    }

    pub fn pop_query_char(&mut self) -> bool {
        let changed = self.query.pop().is_some();
        if changed {
            self.clamp_selection_to_visible();
        }
        changed
    }

    pub fn visible_items(&self) -> impl Iterator<Item = (usize, &DialogItem)> {
        self.items
            .iter()
            .enumerate()
            .filter(|(_, item)| self.item_matches_query(item))
    }

    fn clamp_selection_to_visible(&mut self) {
        if self.item_matches_query_at(self.selected) {
            return;
        }

        if let Some(index) = self.visible_indices().first().copied() {
            self.selected = index;
        } else {
            self.selected = 0;
        }
    }

    fn visible_indices(&self) -> Vec<usize> {
        self.items
            .iter()
            .enumerate()
            .filter_map(|(index, item)| self.item_matches_query(item).then_some(index))
            .collect()
    }

    fn item_matches_query_at(&self, index: usize) -> bool {
        self.items
            .get(index)
            .map(|item| self.item_matches_query(item))
            .unwrap_or(false)
    }

    fn item_matches_query(&self, item: &DialogItem) -> bool {
        let query = self.query.trim();
        if query.is_empty() {
            return true;
        }

        let query = query.to_lowercase();
        item.id.to_lowercase().contains(&query)
            || item.label.to_lowercase().contains(&query)
            || item
                .detail
                .as_deref()
                .map(|detail| detail.to_lowercase().contains(&query))
                .unwrap_or(false)
    }

    pub fn selected_item(&self) -> Option<&DialogItem> {
        if self.item_matches_query_at(self.selected) {
            self.items.get(self.selected)
        } else {
            self.visible_items().next().map(|(_, item)| item)
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TuiState {
    pub input_buffer: String,
    pub input_cursor: usize,
    pub composer_attachments: Vec<UserImageAttachment>,
    pub composer_attachment_cursor: Option<usize>,
    pub timeline: Timeline,
    context: ContextPaneState,
    child_timeline: Option<ChildTranscriptState>,
    pub active_session: bool,
    pub pending_permission: Option<PermissionView>,
    pub slash_panel_selected: usize,
    pub slash_panel_dismissed: bool,
    pub slash_panel_query: String,
    pub phase: AppPhase,
    pub dialog: Option<DialogState>,
    pub provider_label: String,
    pub model_id: String,
    pub model_label: String,
    pub model_token_usage: Option<ModelTokenUsage>,
    pub reasoning_effort_label: Option<String>,
    pub permission_mode_label: String,
    pub current_context_branch: String,
    pub active_tool_call_id: Option<String>,
    pub latest_auto_continue: AutoContinueState,
    pub latest_todo: Option<TodoView>,
    pub transcript_view: TranscriptViewState,
    pub footer_status: FooterStatus,
    pub transcript_scroll: u16,
    pub auto_scroll: bool,
    pub transcript_scrollbar_visible: bool,
    pub child_navigation_prefix: bool,
    pub child_navigation_prefix_ticks_remaining: u8,
    pub tool_output_expanded: bool,
    pub transcript_render_cache: TranscriptRenderCache,
    last_transcript_total_rows: Option<usize>,
    pub status_spinner_frame: usize,
    pub toast: Option<ToastState>,
    pub quit_requested: bool,
    ignore_late_tool_events: bool,
    /// 当前文本选择范围（如果有）
    pub text_selection: Option<TextSelection>,
    /// 是否正在进行鼠标拖拽选择
    pub selection_in_progress: bool,
    /// 最后渲染的 transcript 文本区域（content_area，不含 scrollbar 列，用于鼠标坐标映射）
    pub last_transcript_area: ratatui::layout::Rect,
    /// 最后渲染时已解析为 top-relative 的滚动顶部偏移（0 = 全文第一行可见）
    /// `transcript_scroll` 是 bottom-relative，选择锚点/高亮必须用 top-relative，否则
    /// 底部 auto-scroll 时会把点击映射到全文顶部不可见区域。
    pub last_transcript_scroll_top: u16,
    /// 拖拽选择期间最后一次鼠标位置，用于边缘自动滚动
    pub selection_last_mouse: Option<(u16, u16)>,
}

impl Default for TuiState {
    fn default() -> Self {
        Self {
            input_buffer: String::new(),
            input_cursor: 0,
            composer_attachments: Vec::new(),
            composer_attachment_cursor: None,
            timeline: Timeline::default(),
            context: ContextPaneState::default(),
            child_timeline: None,
            active_session: false,
            pending_permission: None,
            slash_panel_selected: 0,
            slash_panel_dismissed: false,
            slash_panel_query: String::new(),
            phase: AppPhase::Idle,
            dialog: None,
            provider_label: "provider".into(),
            model_id: "pending-runtime-model".into(),
            model_label: "pending runtime model".into(),
            model_token_usage: None,
            reasoning_effort_label: None,
            permission_mode_label: "default".into(),
            current_context_branch: crate::transcript::ROOT_CONTEXT_BRANCH_ID.into(),
            active_tool_call_id: None,
            latest_auto_continue: AutoContinueState::default(),
            latest_todo: None,
            transcript_view: TranscriptViewState::Parent,
            footer_status: FooterStatus::default(),
            transcript_scroll: 0,
            auto_scroll: true,
            transcript_scrollbar_visible: true,
            child_navigation_prefix: false,
            child_navigation_prefix_ticks_remaining: 0,
            tool_output_expanded: false,
            transcript_render_cache: TranscriptRenderCache::default(),
            last_transcript_total_rows: None,
            status_spinner_frame: 0,
            toast: None,
            quit_requested: false,
            ignore_late_tool_events: false,
            text_selection: None,
            selection_in_progress: false,
            last_transcript_area: ratatui::layout::Rect::default(),
            last_transcript_scroll_top: 0,
            selection_last_mouse: None,
        }
    }
}

impl TuiState {
    pub fn is_read_only_child_view(&self) -> bool {
        self.transcript_view.is_child()
    }

    pub fn new(
        model_id: impl Into<String>,
        model_label: impl Into<String>,
        permission_mode_label: impl Into<String>,
    ) -> Self {
        Self {
            model_id: model_id.into(),
            model_label: model_label.into(),
            permission_mode_label: permission_mode_label.into(),
            ..Self::default()
        }
    }

    pub fn set_model(&mut self, model_id: impl Into<String>, model_label: impl Into<String>) {
        self.model_id = model_id.into();
        self.model_label = model_label.into();
    }

    pub fn set_model_context_window(&mut self, context_window_tokens: Option<u64>) {
        self.model_token_usage =
            context_window_tokens.map(|context_window_tokens| ModelTokenUsage {
                used_tokens: 0,
                context_window_tokens,
                input_tokens: 0,
                output_tokens: 0,
                cached_tokens: 0,
            });
    }

    pub fn set_reasoning_effort_label(&mut self, label: Option<String>) {
        self.reasoning_effort_label = label;
    }

    pub fn set_tool_output_expanded(&mut self, expanded: bool) {
        if self.tool_output_expanded != expanded {
            self.tool_output_expanded = expanded;
            self.invalidate_transcript_cache();
            self.last_transcript_total_rows = None;
        }
    }

    pub fn set_transcript_scrollbar_visible(&mut self, visible: bool) {
        if self.transcript_scrollbar_visible != visible {
            self.transcript_scrollbar_visible = visible;
            self.invalidate_transcript_cache();
            self.last_transcript_total_rows = None;
        }
    }

    pub fn set_token_usage(&mut self, usage: ModelTokenUsage) {
        self.model_token_usage = Some(usage);
    }

    pub fn dialog(&self) -> Option<&DialogState> {
        self.dialog.as_ref()
    }

    pub fn show_dashboard(&self) -> bool {
        !self.active_session
            && self.active_timeline().items().is_empty()
            && self.pending_permission.is_none()
    }

    pub fn active_timeline(&self) -> &Timeline {
        if self.is_read_only_child_view() {
            self.child_timeline
                .as_ref()
                .map(|state| &state.timeline)
                .unwrap_or(&self.timeline)
        } else {
            &self.timeline
        }
    }

    pub fn active_context(&self) -> &ContextPaneState {
        if self.is_read_only_child_view() {
            self.child_timeline
                .as_ref()
                .map(|state| &state.context)
                .unwrap_or(&self.context)
        } else {
            &self.context
        }
    }

    pub fn open_context_detail(&mut self, target: Option<ContextDetailTarget>) {
        if self.is_read_only_child_view() {
            if let Some(child) = self.child_timeline.as_mut() {
                child.context.open_detail = target;
                self.sync_child_context_timeline_view();
                return;
            }
        }

        self.context.open_detail = target;
        self.sync_parent_context_timeline_view();
    }

    pub fn mark_session_active(&mut self) {
        self.active_session = true;
    }

    pub fn push_queued_user_message_preview(&mut self, submission: UserMessageSubmission) {
        self.active_session = true;
        self.timeline
            .push_user_message(UserMessageEvent::queued_submission(submission));
        self.reset_slash_panel();
    }

    pub fn activate_queued_user_message(&mut self, submission_id: &str) -> bool {
        if !self.timeline.activate_queued_user_message(submission_id) {
            return false;
        }

        self.active_session = true;
        self.begin_user_turn_state();
        true
    }

    pub fn activate_all_queued_user_message_previews(&mut self) -> usize {
        let activated = self.timeline.activate_queued_user_message_previews();
        if activated > 0 {
            self.active_session = true;
            self.begin_user_turn_state();
        }
        activated
    }

    fn begin_user_turn_state(&mut self) {
        self.latest_auto_continue = AutoContinueState::default();
        self.latest_todo = None;
        self.phase = AppPhase::Running;
        self.active_tool_call_id = None;
        self.pending_permission = None;
        self.ignore_late_tool_events = false;
        self.reset_slash_panel();
        self.footer_status = FooterStatus {
            summary: "Waiting for assistant".into(),
            detail: Some("Streaming output will appear in the timeline".into()),
            is_error: false,
        };
    }

    pub fn dialog_mut(&mut self) -> Option<&mut DialogState> {
        self.dialog.as_mut()
    }

    pub fn dialog_is_open(&self) -> bool {
        self.dialog.is_some()
    }

    pub fn open_dialog(&mut self, dialog: DialogState) {
        self.dialog = Some(dialog);
    }

    pub fn close_dialog(&mut self) {
        self.dialog = None;
    }

    pub fn set_input(&mut self, input: impl Into<String>) {
        self.input_buffer = input.into();
        self.input_cursor = self.input_buffer.len();
        self.composer_attachment_cursor = None;
        self.sync_input_phase();
        self.sync_slash_panel();
    }

    pub fn clear_input(&mut self) {
        self.input_buffer.clear();
        self.input_cursor = 0;
        self.composer_attachment_cursor = None;
        self.sync_input_phase();
        self.sync_slash_panel();
    }

    pub fn composer_content(&self) -> UserMessageContent {
        UserMessageContent::new(self.input_buffer.clone(), self.composer_attachments.clone())
    }

    pub fn clear_composer_attachments(&mut self) {
        self.composer_attachments.clear();
        self.composer_attachment_cursor = None;
    }

    pub fn add_composer_attachment(&mut self, attachment: UserImageAttachment) {
        self.composer_attachments.push(attachment);
        self.composer_attachment_cursor = self.composer_attachments.len().checked_sub(1);
        self.sync_input_phase();
    }

    pub fn remove_composer_attachment(&mut self, attachment_id: &str) -> bool {
        let original_len = self.composer_attachments.len();
        self.composer_attachments
            .retain(|attachment| attachment.id != attachment_id);
        let changed = self.composer_attachments.len() != original_len;
        if changed {
            self.normalize_composer_attachment_cursor();
            self.sync_input_phase();
        }
        changed
    }

    pub fn remove_composer_attachment_at(&mut self, index: usize) -> bool {
        if index >= self.composer_attachments.len() {
            return false;
        }

        self.composer_attachments.remove(index);
        self.normalize_composer_attachment_cursor();
        self.sync_input_phase();
        true
    }

    pub fn normalize_composer_attachment_cursor(&mut self) {
        self.composer_attachment_cursor = match self.composer_attachment_cursor {
            Some(_index) if self.composer_attachments.is_empty() => None,
            Some(index) => Some(index.min(self.composer_attachments.len().saturating_sub(1))),
            None => None,
        };
    }

    pub fn sync_input_phase(&mut self) {
        if self.pending_permission.is_some()
            || matches!(
                self.phase,
                AppPhase::Running | AppPhase::WaitingForPermission | AppPhase::Quitting
            )
        {
            return;
        }

        self.phase = if self.input_buffer.is_empty() && self.composer_attachments.is_empty() {
            AppPhase::Idle
        } else {
            AppPhase::Editing
        };
    }

    pub fn transcript_scroll_offset(&self) -> u16 {
        self.transcript_scroll
    }

    pub fn slash_panel_is_open(&self) -> bool {
        !self.is_read_only_child_view()
            && self.dialog.is_none()
            && self.pending_permission.is_none()
            && !self.slash_panel_dismissed
            && slash::completion_query(&self.input_buffer).is_some()
    }

    pub fn dismiss_slash_panel(&mut self) {
        self.slash_panel_dismissed = true;
    }

    pub fn reset_slash_panel(&mut self) {
        self.slash_panel_selected = 0;
        self.slash_panel_dismissed = false;
        self.slash_panel_query.clear();
    }

    pub fn sync_slash_panel(&mut self) {
        if self.pending_permission.is_some() {
            return;
        }

        let Some(query) = slash::completion_query(&self.input_buffer) else {
            self.reset_slash_panel();
            return;
        };

        if self.slash_panel_query != query {
            self.slash_panel_query = query;
            self.slash_panel_selected = 0;
            self.slash_panel_dismissed = false;
        }
    }

    pub fn transcript_is_at_bottom(&self, total_rows: usize, viewport_rows: u16) -> bool {
        measure::is_at_bottom(total_rows, viewport_rows, self.transcript_scroll)
    }

    pub fn scroll_transcript_up(&mut self, rows: u16) {
        self.transcript_scroll = self.transcript_scroll.saturating_add(rows);
        self.auto_scroll = self.transcript_scroll == 0;
    }

    pub fn scroll_transcript_down(&mut self, rows: u16) {
        self.transcript_scroll = self.transcript_scroll.saturating_sub(rows);
        self.auto_scroll = self.transcript_scroll == 0;
    }

    pub fn scroll_transcript_page_up(&mut self) {
        self.scroll_transcript_up(10);
    }

    pub fn scroll_transcript_page_down(&mut self) {
        self.scroll_transcript_down(10);
    }

    pub fn scroll_transcript_to_bottom(&mut self) {
        self.transcript_scroll = 0;
        self.auto_scroll = true;
    }

    pub fn sync_transcript_viewport_rows(&mut self, total_rows: usize) {
        if !self.auto_scroll
            && let Some(previous_total_rows) = self.last_transcript_total_rows
            && total_rows > previous_total_rows
        {
            let delta = total_rows.saturating_sub(previous_total_rows);
            let delta = u16::try_from(delta).unwrap_or(u16::MAX);
            self.transcript_scroll = self.transcript_scroll.saturating_add(delta);
        }

        self.last_transcript_total_rows = Some(total_rows);
    }

    pub fn set_permission_mode_label(&mut self, label: impl Into<String>) {
        self.permission_mode_label = label.into();
    }

    pub fn set_current_context_branch(&mut self, branch_id: impl Into<String>) {
        self.current_context_branch = branch_id.into();
    }

    pub fn set_provider_label(&mut self, label: impl Into<String>) {
        self.provider_label = label.into();
    }

    pub fn set_footer(&mut self, summary: impl Into<String>, detail: Option<String>) {
        self.footer_status = FooterStatus {
            summary: summary.into(),
            detail,
            is_error: false,
        };
    }

    pub fn show_toast(&mut self, message: impl Into<String>, kind: ToastKind) {
        self.toast = Some(ToastState::new(message, kind, ToastState::DEFAULT_TICKS));
    }

    pub fn toast(&self) -> Option<&ToastState> {
        self.toast.as_ref()
    }

    pub fn replace_session_timeline(&mut self, messages: Vec<ConversationMessage>) {
        self.timeline = Timeline::from_conversation(messages);
        self.context = ContextPaneState::default();
        self.sync_parent_context_timeline_view();
        self.child_timeline = None;
        self.latest_auto_continue = AutoContinueState::default();
        self.latest_todo = None;
        self.transcript_view = TranscriptViewState::Parent;
        self.reset_after_session_timeline_replace();
    }

    pub fn replace_session_timeline_from_records(&mut self, records: &[TranscriptRecord]) {
        self.try_replace_session_timeline_from_records(records)
            .expect("context projection should be valid when replacing session timeline");
    }

    pub fn try_replace_session_timeline_from_records(
        &mut self,
        records: &[TranscriptRecord],
    ) -> Result<()> {
        let context = project_context_pane(records)?;
        self.active_session = true;
        self.timeline = Timeline::from_transcript_records(records);
        self.context = context;
        self.sync_parent_context_timeline_view();
        self.child_timeline = None;
        self.latest_auto_continue = restore_latest_auto_continue_state(records).unwrap_or_default();
        self.latest_todo = restore_latest_todo_snapshot(records).map(|items| TodoView {
            items,
            auto_continue: self.latest_auto_continue.clone(),
        });
        self.transcript_view = TranscriptViewState::Parent;
        self.reset_after_session_timeline_replace();
        Ok(())
    }

    pub fn replace_child_timeline_from_records(
        &mut self,
        records: &[TranscriptRecord],
        parent_session_id: impl Into<String>,
        child_session_id: impl Into<String>,
        agent_name: impl Into<String>,
        index: usize,
        total: usize,
    ) {
        self.try_replace_child_timeline_from_records(
            records,
            parent_session_id,
            child_session_id,
            agent_name,
            index,
            total,
        )
        .expect("context projection should be valid when replacing child timeline");
    }

    pub fn try_replace_child_timeline_from_records(
        &mut self,
        records: &[TranscriptRecord],
        parent_session_id: impl Into<String>,
        child_session_id: impl Into<String>,
        agent_name: impl Into<String>,
        index: usize,
        total: usize,
    ) -> Result<()> {
        let child_state = project_child_timeline_state(records)?;
        self.active_session = true;
        self.input_buffer.clear();
        self.input_cursor = 0;
        self.sync_input_phase();
        self.close_dialog();
        self.reset_slash_panel();
        self.child_timeline = Some(child_state);
        self.transcript_view = TranscriptViewState::Child {
            parent_session_id: parent_session_id.into(),
            child_session_id: child_session_id.into(),
            agent_name: agent_name.into(),
            index,
            total,
        };
        self.sync_child_context_timeline_view();
        self.scroll_transcript_to_bottom();
        self.invalidate_transcript_cache();
        self.last_transcript_total_rows = None;
        Ok(())
    }

    pub fn refresh_child_timeline_from_records(&mut self, records: &[TranscriptRecord]) {
        self.try_refresh_child_timeline_from_records(records)
            .expect("context projection should be valid when refreshing child timeline");
    }

    pub fn try_refresh_child_timeline_from_records(
        &mut self,
        records: &[TranscriptRecord],
    ) -> Result<()> {
        if !self.transcript_view.is_child() {
            return Ok(());
        }

        self.child_timeline = Some(project_child_timeline_state(records)?);
        self.sync_child_context_timeline_view();
        self.ignore_late_tool_events = false;
        self.invalidate_transcript_cache();
        self.last_transcript_total_rows = None;
        self.reproject_pending_permission();
        Ok(())
    }

    pub fn child_view_has_live_stream(&self) -> bool {
        self.child_timeline
            .as_ref()
            .map(|child| child.live_streaming)
            .unwrap_or(false)
    }

    pub fn child_view_metadata(&self) -> Option<ChildViewMetadata> {
        let TranscriptViewState::Child {
            parent_session_id,
            child_session_id,
            agent_name,
            index,
            total,
        } = &self.transcript_view
        else {
            return None;
        };

        let child = self.child_timeline.as_ref()?;
        Some(ChildViewMetadata {
            parent_session_id: parent_session_id.clone(),
            child_session_id: child_session_id.clone(),
            agent_name: agent_name.clone(),
            index: *index,
            total: *total,
            model: child.model.clone(),
            record_count: child.record_count,
        })
    }

    pub fn restore_parent_timeline_view(&mut self) {
        self.transcript_view = TranscriptViewState::Parent;
        self.close_dialog();
        self.reset_slash_panel();
        self.scroll_transcript_to_bottom();
        self.invalidate_transcript_cache();
        self.last_transcript_total_rows = None;
        self.reproject_pending_permission();
    }

    fn reset_after_session_timeline_replace(&mut self) {
        self.pending_permission = None;
        self.active_tool_call_id = None;
        self.ignore_late_tool_events = false;
        self.phase = AppPhase::Completed;
        self.model_token_usage = None;
        self.close_dialog();
        self.reset_slash_panel();
        self.scroll_transcript_to_bottom();
        self.invalidate_transcript_cache();
        self.last_transcript_total_rows = None;
    }

    fn accepts_tool_events(&self) -> bool {
        !self.ignore_late_tool_events
    }

    pub fn set_pending_permission_projection(&mut self, permission: Option<PermissionView>) {
        self.pending_permission = permission;
        self.reproject_pending_permission();
    }

    fn reproject_pending_permission(&mut self) {
        let Some(permission) = self.pending_permission.as_ref() else {
            return;
        };
        self.phase = AppPhase::WaitingForPermission;
        self.active_tool_call_id = Some(permission.call_id.clone());
        let subject = permission
            .origin_label
            .as_deref()
            .map(|origin| format!("{origin} · {}", permission.tool_name))
            .unwrap_or_else(|| permission.tool_name.clone());
        self.footer_status = FooterStatus {
            summary: format!("Permission required for {subject}"),
            detail: Some(permission.summary.clone()),
            is_error: false,
        };
    }

    fn sync_parent_context_timeline_view(&mut self) {
        let view = project_context_timeline_view(&self.context);
        self.timeline.set_context_view(view);
        self.invalidate_transcript_cache();
        self.last_transcript_total_rows = None;
    }

    fn sync_child_context_timeline_view(&mut self) {
        if let Some(child) = self.child_timeline.as_mut() {
            let view = project_context_timeline_view(&child.context);
            child.timeline.set_context_view(view);
        }
        self.invalidate_transcript_cache();
        self.last_transcript_total_rows = None;
    }

    pub fn apply_child_app_event(&mut self, child_session_id: &str, event: AppEvent) {
        let viewing_child = matches!(
            &self.transcript_view,
            TranscriptViewState::Child {
                child_session_id: active_child_session_id,
                ..
            } if active_child_session_id == child_session_id
        );

        self.project_child_event_to_parent_subagent_tool(child_session_id, &event);

        if self.apply_child_context_event(child_session_id, &event, viewing_child) {
            return;
        }

        match event {
            AppEvent::PermissionRequested(request) => {
                self.apply_permission_requested_projection(&request);
                if viewing_child && let Some(child_timeline) = self.child_timeline.as_mut() {
                    child_timeline.timeline.push_permission_request(request);
                    child_timeline.live_streaming = true;
                    self.invalidate_transcript_cache();
                    self.last_transcript_total_rows = None;
                }
            }
            AppEvent::PermissionResolved(resolution) => {
                self.apply_permission_resolved_projection(&resolution);
                if viewing_child && let Some(child_timeline) = self.child_timeline.as_mut() {
                    child_timeline.timeline.resolve_permission(resolution);
                    child_timeline.live_streaming = true;
                    self.invalidate_transcript_cache();
                    self.last_transcript_total_rows = None;
                }
            }
            event if viewing_child => {
                let accepts_tool_events = self.accepts_tool_events();
                let Some(child_timeline) = self.child_timeline.as_mut() else {
                    return;
                };
                apply_projected_app_event(
                    EventProjection {
                        active_session: &mut self.active_session,
                        latest_auto_continue: &mut self.latest_auto_continue,
                        latest_todo: &mut self.latest_todo,
                        phase: &mut self.phase,
                        active_tool_call_id: &mut self.active_tool_call_id,
                        pending_permission: &mut self.pending_permission,
                        footer_status: &mut self.footer_status,
                        model_token_usage: &mut self.model_token_usage,
                        ignore_late_tool_events: &mut self.ignore_late_tool_events,
                        quit_requested: &mut self.quit_requested,
                        status_spinner_frame: &mut self.status_spinner_frame,
                        toast: &mut self.toast,
                        timeline: &mut child_timeline.timeline,
                        live_streaming: None,
                        accepts_tool_events: true,
                    }
                    .with_live_streaming(&mut child_timeline.live_streaming)
                    .with_tool_event_acceptance(accepts_tool_events),
                    event,
                );

                self.invalidate_transcript_cache();
                self.last_transcript_total_rows = None;
            }
            _ => {}
        }
    }

    pub fn apply_event(&mut self, event: AppEvent) {
        if self.apply_context_event(&event) {
            return;
        }

        if let AppEvent::PermissionRequested(request) = event.clone() {
            self.on_permission_requested(request);
            return;
        }

        if let AppEvent::PermissionResolved(resolution) = event.clone() {
            self.apply_permission_resolved_projection(&resolution);
        }

        let accepts_tool_events = self.accepts_tool_events();
        apply_projected_app_event(
            EventProjection {
                active_session: &mut self.active_session,
                latest_auto_continue: &mut self.latest_auto_continue,
                latest_todo: &mut self.latest_todo,
                phase: &mut self.phase,
                active_tool_call_id: &mut self.active_tool_call_id,
                pending_permission: &mut self.pending_permission,
                footer_status: &mut self.footer_status,
                model_token_usage: &mut self.model_token_usage,
                ignore_late_tool_events: &mut self.ignore_late_tool_events,
                quit_requested: &mut self.quit_requested,
                status_spinner_frame: &mut self.status_spinner_frame,
                toast: &mut self.toast,
                timeline: &mut self.timeline,
                live_streaming: None,
                accepts_tool_events: true,
            }
            .with_tool_event_acceptance(accepts_tool_events),
            event,
        );
    }

    fn on_user_message(&mut self, message: UserMessageEvent) {
        self.active_session = true;
        self.timeline.push_user_message(message);
        self.begin_user_turn_state();
    }

    fn on_permission_requested(&mut self, request: PermissionRequestEvent) {
        self.apply_permission_requested_projection(&request);
        self.timeline.push_permission_request(request);
    }

    fn apply_permission_requested_projection(&mut self, request: &PermissionRequestEvent) {
        self.phase = AppPhase::WaitingForPermission;
        self.active_tool_call_id = Some(request.call_id.clone());
        self.pending_permission = Some(PermissionView::from_request(request.clone()));
        self.slash_panel_dismissed = false;
        let subject = request
            .origin_label
            .as_deref()
            .map(|origin| format!("{origin} · {}", request.tool_name))
            .unwrap_or_else(|| request.tool_name.clone());
        self.footer_status = FooterStatus {
            summary: format!("Permission required for {subject}"),
            detail: Some(request.summary.clone()),
            is_error: false,
        };
    }

    fn apply_context_event(&mut self, event: &AppEvent) -> bool {
        match event {
            AppEvent::ContextTreeUpdated(update) => {
                self.context.tree = update.tree.clone();
                self.sync_parent_context_timeline_view();
                true
            }
            AppEvent::ContextViewUpdated(update) => {
                self.context.view = update.projection.clone();
                if let Some(target) = self.context.open_detail.clone()
                    && !context_detail_target_exists(&self.context, &target)
                {
                    self.context.open_detail = None;
                }
                self.sync_parent_context_timeline_view();
                true
            }
            AppEvent::ContextDetailOpened(update) => {
                self.context.open_detail = update
                    .open_detail_block_id
                    .clone()
                    .map(ContextDetailTarget::Block);
                self.sync_parent_context_timeline_view();
                true
            }
            AppEvent::FoldedOutputsUpdated(update) => {
                self.context.view.folded_outputs = update
                    .folded_outputs
                    .iter()
                    .cloned()
                    .map(|metadata| (metadata.output_id.clone(), metadata))
                    .collect();
                self.sync_parent_context_timeline_view();
                true
            }
            AppEvent::ContextSummaryUpdated(update) => {
                self.context.view.summary_artifacts = update.summaries.clone();
                self.sync_parent_context_timeline_view();
                true
            }
            _ => false,
        }
    }

    fn apply_child_context_event(
        &mut self,
        child_session_id: &str,
        event: &AppEvent,
        viewing_child: bool,
    ) -> bool {
        let Some(child) = self.child_timeline.as_mut() else {
            return false;
        };
        if !matches!(
            &self.transcript_view,
            TranscriptViewState::Child {
                child_session_id: active_child_session_id,
                ..
            } if active_child_session_id == child_session_id
        ) {
            return false;
        }

        let handled = match event {
            AppEvent::ContextTreeUpdated(update) => {
                child.context.tree = update.tree.clone();
                true
            }
            AppEvent::ContextViewUpdated(update) => {
                child.context.view = update.projection.clone();
                if let Some(target) = child.context.open_detail.clone()
                    && !context_detail_target_exists(&child.context, &target)
                {
                    child.context.open_detail = None;
                }
                true
            }
            AppEvent::ContextDetailOpened(update) => {
                child.context.open_detail = update
                    .open_detail_block_id
                    .clone()
                    .map(ContextDetailTarget::Block);
                true
            }
            AppEvent::FoldedOutputsUpdated(update) => {
                child.context.view.folded_outputs = update
                    .folded_outputs
                    .iter()
                    .cloned()
                    .map(|metadata| (metadata.output_id.clone(), metadata))
                    .collect();
                true
            }
            AppEvent::ContextSummaryUpdated(update) => {
                child.context.view.summary_artifacts = update.summaries.clone();
                true
            }
            _ => false,
        };

        if handled && viewing_child {
            self.sync_child_context_timeline_view();
        }
        handled
    }

    fn apply_permission_resolved_projection(&mut self, resolution: &PermissionResolutionEvent) {
        self.active_tool_call_id = None;
        if self.pending_permission.as_ref().map(|p| p.call_id.as_str())
            == Some(resolution.call_id.as_str())
        {
            self.pending_permission = None;
        }
        self.footer_status = match resolution.decision {
            PermissionDecision::Approved => FooterStatus::permission_resolved(true),
            PermissionDecision::Denied => FooterStatus::permission_resolved(false),
        };
        self.phase = AppPhase::Running;
    }

    fn project_child_event_to_parent_subagent_tool(
        &mut self,
        child_session_id: &str,
        event: &AppEvent,
    ) {
        if matches!(
            &self.transcript_view,
            TranscriptViewState::Child {
                child_session_id: active_child_session_id,
                ..
            } if active_child_session_id != child_session_id
        ) {
            return;
        }

        let Some((status, summary)) =
            child_event_projection_payload(self.pending_permission.as_ref(), event)
        else {
            return;
        };

        if self.timeline.update_active_subagent_tool_live_summary(
            child_session_id,
            &status,
            &summary,
        ) {
            self.invalidate_transcript_cache();
            self.last_transcript_total_rows = None;
        }
    }

    /// 将终端坐标映射到选择锚点
    ///
    /// 使用渲染时存的 `last_transcript_area`（content_area，不含 scrollbar 列）
    /// 与 `last_transcript_scroll_top`（已解析为 top-relative 偏移）。这两者都来自
    /// 渲染阶段，保证点击坐标和高亮坐标系完全一致。
    pub fn map_mouse_to_anchor(
        &self,
        terminal_col: u16,
        terminal_row: u16,
    ) -> Option<SelectionAnchor> {
        let area = self.last_transcript_area;

        // 1. 命中检测：必须在 content_area 内，否则不映射
        if terminal_col < area.left()
            || terminal_col >= area.right()
            || terminal_row < area.top()
            || terminal_row >= area.bottom()
            || area.width == 0
            || area.height == 0
        {
            return None;
        }

        // 2. Terminal row → Viewport row → Absolute row（top-relative）
        let viewport_row = terminal_row - area.y;
        let absolute_row = viewport_row as usize + self.last_transcript_scroll_top as usize;

        // 3. 找到对应的 TimelineItem（二分查找）
        let cache = &self.transcript_render_cache;
        if cache.row_starts().is_empty() {
            return None;
        }

        // 顶部 spacer / separator 不可映射：absolute_row 必须落在某个 item 内
        let item_index = match cache.row_starts().binary_search(&absolute_row) {
            Ok(idx) => idx,
            Err(idx) => idx.saturating_sub(1),
        };

        if item_index >= cache.entries().len() {
            return None;
        }

        // 4. 计算 Item 内的行偏移；超出该 item 范围（落在 separator/spacer）则放弃
        let item_start_row = cache.row_starts()[item_index];
        let item_line_count = cache.entries()[item_index].lines.len();
        let rendered_line_offset = absolute_row.saturating_sub(item_start_row);
        if rendered_line_offset >= item_line_count {
            return None;
        }

        let origin = &cache.entries()[item_index].line_origins[rendered_line_offset];
        let Some(block_index) = origin.block_index else {
            return None;
        };

        // 5. 获取该行对应的纯内容文本，按 content_area 本地列计算“内容内偏移”。
        // SelectionAnchor::char_offset 对外保持“该渲染行对应内容片段内的字符偏移”，
        // 不再包含左侧 card border / padding / badge 等装饰字符。
        let local_col = terminal_col - area.x;
        let content_col = local_col.saturating_sub(origin.content_prefix_chars as u16);
        let source = &cache.entries()[item_index].source_blocks[block_index].source;
        let chunk_text = slice_chars(
            source,
            origin.content_char_offset,
            origin
                .content_char_offset
                .saturating_add(origin.content_char_len),
        );
        let char_offset =
            column_to_char_offset(&chunk_text, content_col).min(origin.content_char_len);

        Some(SelectionAnchor {
            item_index,
            rendered_line_offset,
            char_offset,
        })
    }

    /// Timeline 更新时调用，清除选择状态
    pub fn on_timeline_changed(&mut self) {
        self.text_selection = None;
        self.selection_in_progress = false;
    }

    /// 使 transcript 渲染缓存失效，并同步清除基于该缓存的选择锚点
    ///
    /// 缓存的 `row_starts` / `entries` 一旦被清空，`TextSelection` 中的
    /// `item_index` / `rendered_line_offset` 即指向不存在的位置，必须一并清除，
    /// 否则会高亮或复制到错位的内容。
    pub fn invalidate_transcript_cache(&mut self) {
        self.transcript_render_cache.clear();
        self.on_timeline_changed();
    }
}

/// 将列坐标转换为字符偏移（考虑 Unicode 宽度）
fn column_to_char_offset(text: &str, target_col: u16) -> usize {
    use unicode_width::UnicodeWidthChar;

    let mut current_width = 0;
    let mut char_count = 0;

    for ch in text.chars() {
        let ch_width = ch.width().unwrap_or(1);
        if current_width >= target_col as usize {
            break;
        }
        current_width += ch_width;
        char_count += 1;
    }

    char_count
}

fn slice_chars(text: &str, start: usize, end: usize) -> String {
    text.chars()
        .skip(start)
        .take(end.saturating_sub(start))
        .collect()
}

struct EventProjection<'a> {
    active_session: &'a mut bool,
    latest_auto_continue: &'a mut AutoContinueState,
    latest_todo: &'a mut Option<TodoView>,
    phase: &'a mut AppPhase,
    active_tool_call_id: &'a mut Option<String>,
    pending_permission: &'a mut Option<PermissionView>,
    footer_status: &'a mut FooterStatus,
    model_token_usage: &'a mut Option<ModelTokenUsage>,
    ignore_late_tool_events: &'a mut bool,
    quit_requested: &'a mut bool,
    status_spinner_frame: &'a mut usize,
    toast: &'a mut Option<ToastState>,
    timeline: &'a mut Timeline,
    live_streaming: Option<&'a mut bool>,
    accepts_tool_events: bool,
}

impl<'a> EventProjection<'a> {
    fn with_live_streaming(mut self, live_streaming: &'a mut bool) -> Self {
        self.live_streaming = Some(live_streaming);
        self
    }

    fn with_tool_event_acceptance(mut self, accepts_tool_events: bool) -> Self {
        self.accepts_tool_events = accepts_tool_events;
        self
    }
}

fn apply_projected_app_event(mut projection: EventProjection<'_>, event: AppEvent) {
    let terminal_event = matches!(
        event,
        AppEvent::Interrupted | AppEvent::Error(_) | AppEvent::Done
    );

    match event {
        AppEvent::Tick => {
            *projection.status_spinner_frame = projection.status_spinner_frame.wrapping_add(1);
            if projection.toast.as_mut().is_some_and(ToastState::tick) {
                *projection.toast = None;
            }
            return;
        }
        AppEvent::UserMessage(message) => {
            *projection.active_session = true;
            projection.timeline.push_user_message(message);
            *projection.latest_auto_continue = AutoContinueState::default();
            *projection.latest_todo = None;
            *projection.phase = AppPhase::Running;
            *projection.active_tool_call_id = None;
            *projection.pending_permission = None;
            *projection.ignore_late_tool_events = false;
            *projection.footer_status = FooterStatus {
                summary: "Waiting for assistant".into(),
                detail: Some("Streaming output will appear in the timeline".into()),
                is_error: false,
            };
        }
        AppEvent::ReasoningDelta(reasoning) => {
            *projection.phase = AppPhase::Running;
            projection.timeline.push_reasoning_delta(reasoning);
            *projection.footer_status = FooterStatus::streaming();
        }
        AppEvent::ReasoningDone(reasoning) => {
            projection
                .timeline
                .finalize_reasoning(&reasoning.item_id, &reasoning.text);
        }
        AppEvent::AssistantDelta(delta) => {
            *projection.phase = AppPhase::Running;
            projection.timeline.push_assistant_delta(delta);
            *projection.footer_status = FooterStatus::streaming();
        }
        AppEvent::AssistantDone { message_id } => {
            projection
                .timeline
                .finalize_assistant_message(message_id.as_deref());
            *projection.footer_status = FooterStatus::ready_for_next_prompt();
        }
        AppEvent::TokenUsage(usage) => {
            *projection.model_token_usage = Some(ModelTokenUsage::from(usage));
        }
        AppEvent::ToolPending(tool) => {
            if projection.accepts_tool_events && projection.timeline.push_tool_pending(tool.clone())
            {
                *projection.active_tool_call_id = Some(tool.call_id.clone());
                *projection.phase = AppPhase::Running;
                *projection.footer_status = FooterStatus::preparing_tool(&tool.name);
            }
        }
        AppEvent::ToolCancelled(tool) => {
            if projection.accepts_tool_events {
                projection.timeline.cancel_tool(&tool.call_id, &tool.name);
                if projection.active_tool_call_id.as_deref() == Some(tool.call_id.as_str()) {
                    *projection.active_tool_call_id = None;
                }
                *projection.footer_status = FooterStatus::tool_cancelled(&tool.name);
            }
        }
        AppEvent::ToolStarted(tool) => {
            if projection.accepts_tool_events && projection.timeline.push_tool_started(tool.clone())
            {
                *projection.active_tool_call_id = Some(tool.call_id.clone());
                *projection.phase = AppPhase::Running;
                *projection.footer_status = FooterStatus::running_tool(&tool.name, &tool.summary);
            }
        }
        AppEvent::ToolFinished(tool) => {
            if projection.accepts_tool_events
                && projection.timeline.push_tool_finished(tool.clone())
            {
                if projection.active_tool_call_id.as_deref() == Some(tool.call_id.as_str()) {
                    *projection.active_tool_call_id = None;
                }
                *projection.footer_status = match tool.outcome {
                    ToolOutcome::Success => FooterStatus::tool_finished(&tool.name, true),
                    ToolOutcome::Failure => FooterStatus::tool_finished(&tool.name, false),
                };
            }
        }
        AppEvent::ToolOutputDelta(delta) => {
            if projection.accepts_tool_events {
                projection.timeline.push_tool_output_delta(delta);
            }
        }
        AppEvent::TodoSnapshot(todo) => {
            let auto_continue = projection.latest_auto_continue.clone();
            *projection.latest_todo = Some(TodoView {
                items: todo.items.clone(),
                auto_continue: auto_continue.clone(),
            });
            projection.timeline.push_todo_snapshot(todo);
            projection
                .timeline
                .apply_auto_continue_changed(AutoContinueChangedEvent::new(auto_continue));
        }
        AppEvent::AutoContinueChanged(event) => {
            *projection.latest_auto_continue = event.state.clone();
            if let Some(todo) = projection.latest_todo.as_mut() {
                todo.auto_continue = event.state.clone();
                projection.timeline.apply_auto_continue_changed(event);
            }
        }
        AppEvent::Notice(notice) => {
            projection.timeline.push_notice(notice.message);
        }
        AppEvent::ProcessIssue(issue) => {
            *projection.phase = AppPhase::Running;
            *projection.footer_status = FooterStatus::process_issue(
                &issue.message,
                issue.detail.as_deref(),
                issue.action.as_deref(),
            );
            projection.timeline.push_notice(issue.message);
        }
        AppEvent::Interrupted => {
            *projection.phase = AppPhase::Completed;
            *projection.active_tool_call_id = None;
            *projection.pending_permission = None;
            *projection.latest_auto_continue = AutoContinueState::default();
            *projection.latest_todo = None;
            *projection.ignore_late_tool_events = true;
            projection.timeline.cancel_active_tools();
            *projection.footer_status = FooterStatus {
                summary: "Interrupted".into(),
                detail: Some("Current assistant turn stopped".into()),
                is_error: false,
            };
            projection.timeline.push_notice("Interrupted by user");
        }
        AppEvent::Error(error) => {
            *projection.phase = AppPhase::Error;
            *projection.active_tool_call_id = None;
            *projection.pending_permission = None;
            *projection.latest_auto_continue = AutoContinueState::default();
            *projection.latest_todo = None;
            *projection.footer_status = FooterStatus::error(&error.message);
            projection.timeline.push_error(error);
        }
        AppEvent::Done => {
            *projection.phase = AppPhase::Completed;
            *projection.active_tool_call_id = None;
            *projection.pending_permission = None;
            *projection.footer_status = FooterStatus::ready_for_next_prompt();
        }
        AppEvent::PermissionResolved(resolution) => {
            projection.timeline.resolve_permission(resolution);
        }
        AppEvent::ContextTreeUpdated(_)
        | AppEvent::ContextViewUpdated(_)
        | AppEvent::ContextDetailOpened(_)
        | AppEvent::FoldedOutputsUpdated(_)
        | AppEvent::ContextSummaryUpdated(_) => {}
        AppEvent::Quit => {
            *projection.phase = AppPhase::Quitting;
            *projection.quit_requested = true;
            *projection.footer_status = FooterStatus {
                summary: "Exiting".into(),
                detail: None,
                is_error: false,
            };
        }
        AppEvent::PermissionRequested(_) => {}
    }

    if !terminal_event && let Some(live_streaming) = projection.live_streaming.as_deref_mut() {
        *live_streaming = true;
    }
}

fn child_event_projection_payload(
    pending_permission: Option<&PermissionView>,
    event: &AppEvent,
) -> Option<(String, String)> {
    match event {
        AppEvent::ToolPending(tool) => Some((
            "preparing".into(),
            compact_child_projection_text(&format!("{} preparing input", tool.name)),
        )),
        AppEvent::ToolCancelled(tool) => Some((
            "cancelled".into(),
            compact_child_projection_text(&format!("{} cancelled", tool.name)),
        )),
        AppEvent::ToolStarted(tool) => Some((
            "running".into(),
            compact_child_projection_text(&child_tool_projection_summary(
                &tool.name,
                &tool.summary,
            )),
        )),
        AppEvent::ToolFinished(tool) => Some((
            match tool.outcome {
                ToolOutcome::Success => "completed",
                ToolOutcome::Failure => "failed",
            }
            .into(),
            compact_child_projection_text(&child_tool_projection_summary(
                &tool.name,
                &tool.summary,
            )),
        )),
        AppEvent::PermissionRequested(request) => Some((
            "approval".into(),
            compact_child_projection_text(&format!(
                "approval needed · {}",
                child_tool_projection_summary(&request.tool_name, &request.summary)
            )),
        )),
        AppEvent::PermissionResolved(resolution) => {
            let subject = pending_permission
                .filter(|permission| permission.call_id == resolution.call_id)
                .map(|permission| {
                    child_tool_projection_summary(&permission.tool_name, &permission.summary)
                })
                .unwrap_or_else(|| "permission request".into());
            let status = match resolution.decision {
                PermissionDecision::Approved => "approved",
                PermissionDecision::Denied => "denied",
            };
            let summary = resolution
                .reason
                .as_deref()
                .map(str::trim)
                .filter(|reason| !reason.is_empty())
                .map(|reason| format!("approval {status} · {subject} · {reason}"))
                .unwrap_or_else(|| format!("approval {status} · {subject}"));
            Some((status.into(), compact_child_projection_text(&summary)))
        }
        AppEvent::ProcessIssue(issue) => Some((
            "issue".into(),
            compact_child_projection_text(&issue.message),
        )),
        AppEvent::Error(error) => Some((
            "error".into(),
            compact_child_projection_text(&error.message),
        )),
        AppEvent::Interrupted => Some(("interrupted".into(), "child session interrupted".into())),
        _ => None,
    }
}

fn child_tool_projection_summary(name: &str, summary: &str) -> String {
    let summary = summary.trim();
    if summary.is_empty() {
        return name.to_string();
    }
    if summary.starts_with(name) {
        return summary.to_string();
    }
    format!("{name} — {summary}")
}

fn compact_child_projection_text(text: &str) -> String {
    let single_line = text.split_whitespace().collect::<Vec<_>>().join(" ");
    let limit = 160;
    if single_line.chars().count() <= limit {
        return single_line;
    }

    let mut truncated = single_line.chars().take(limit).collect::<String>();
    truncated.push('…');
    truncated
}

fn project_child_timeline_state(records: &[TranscriptRecord]) -> Result<ChildTranscriptState> {
    Ok(ChildTranscriptState {
        timeline: Timeline::from_transcript_records(records),
        model: child_transcript_model(records),
        record_count: records.len(),
        live_streaming: false,
        context: project_context_pane(records)?,
    })
}

fn project_context_pane(records: &[TranscriptRecord]) -> Result<ContextPaneState> {
    let tree = transcript_projection::project_context_tree(records)?;
    let view = transcript_projection::project_context_view(records)?;
    let open_detail = view
        .view_state
        .open_detail_block_id()
        .map(|block_id| ContextDetailTarget::Block(block_id.as_str().to_string()));
    Ok(ContextPaneState {
        tree,
        view,
        open_detail,
    })
}

fn context_detail_target_exists(context: &ContextPaneState, target: &ContextDetailTarget) -> bool {
    match target {
        ContextDetailTarget::Node(node_id) => context
            .tree
            .nodes()
            .any(|node| node.node_id.as_str() == node_id),
        ContextDetailTarget::Block(block_id) => context
            .view
            .blocks
            .keys()
            .any(|candidate| candidate.as_str() == block_id),
        ContextDetailTarget::Summary(artifact_id) => context
            .view
            .summary_artifacts
            .iter()
            .any(|artifact| artifact.artifact_id == *artifact_id),
        ContextDetailTarget::FoldedOutput(output_id) => folded_output_visible(context, output_id),
    }
}

fn folded_output_visible(context: &ContextPaneState, output_id: &str) -> bool {
    context.view.folded_outputs.contains_key(output_id)
        && !context.view.blocks.values().any(|block| {
            block.folded_output_id.as_deref() == Some(output_id)
                && context.view.view_state.status(&block.block_id)
                    == Some(ContextViewStatus::RemovedFromView)
        })
}

fn project_context_timeline_view(context: &ContextPaneState) -> Option<ContextTimelineView> {
    let has_display_blocks = context.view.blocks.values().any(|block| {
        if context.view.view_state.status(&block.block_id)
            == Some(ContextViewStatus::RemovedFromView)
        {
            return false;
        }
        matches!(
            context.view.view_state.status(&block.block_id),
            Some(ContextViewStatus::Pinned | ContextViewStatus::Archived)
        ) || block
            .folded_output_id
            .as_deref()
            .is_some_and(|output_id| folded_output_visible(context, output_id))
            || matches!(block.source, ContextBlockSource::SummaryArtifact { .. })
    });
    let has_visible_folded_outputs = context
        .view
        .folded_outputs
        .keys()
        .any(|output_id| folded_output_visible(context, output_id));
    let has_context = context.tree.node_count() > 1
        || has_display_blocks
        || !context.view.summary_artifacts.is_empty()
        || has_visible_folded_outputs;
    if !has_context {
        return None;
    }

    let active_node = context.tree.active_node_id().and_then(|node_id| {
        context.tree.node(node_id).map(|node| {
            let label = node
                .label
                .clone()
                .unwrap_or_else(|| node.node_id.as_str().to_string());
            (label, node.status == ContextNodeStatus::Archived)
        })
    });

    let node_lines = context
        .tree
        .nodes()
        .filter(|node| node.node_id != *context.tree.root_node_id())
        .map(|node| {
            let depth = context_node_depth(&context.tree, node.node_id.as_str());
            let mut badges = Vec::new();
            if context.tree.active_node_id() == Some(&node.node_id) {
                badges.push("Active".into());
            }
            if node.status == ContextNodeStatus::Archived {
                badges.push("Archived".into());
            }
            if context
                .view
                .summary_artifacts
                .iter()
                .any(|artifact| artifact.node_id == node.node_id.as_str())
            {
                badges.push("Summary".into());
            }
            if context.view.folded_outputs.values().any(|output| {
                output.node_id.as_deref() == Some(node.node_id.as_str())
                    && folded_output_visible(context, &output.output_id)
            }) {
                badges.push("Folded output".into());
            }
            if node.source_ref.is_some() {
                badges.push("Source".into());
            }
            ContextNodeLineView {
                depth,
                label: node
                    .label
                    .clone()
                    .unwrap_or_else(|| node.node_id.as_str().to_string()),
                badges,
            }
        })
        .collect::<Vec<_>>();

    let block_lines = context
        .view
        .blocks
        .values()
        .filter_map(|block| {
            let mut badges = Vec::new();
            let status = context.view.view_state.status(&block.block_id);
            match status {
                Some(ContextViewStatus::Pinned) => badges.push("Pinned".into()),
                Some(ContextViewStatus::Archived) => badges.push("Archived".into()),
                Some(ContextViewStatus::Resolved) => badges.push("Resolved".into()),
                Some(ContextViewStatus::RemovedFromView) => return None,
                _ => {}
            }
            if block.folded_output_id.is_some() {
                badges.push("Folded output".into());
            }
            if matches!(block.source, ContextBlockSource::SummaryArtifact { .. }) {
                badges.push("Summary".into());
            }
            if block.is_protected() {
                badges.push("Protected".into());
            }
            if badges.is_empty() {
                return None;
            }
            Some(ContextBlockLineView {
                label: block.title.clone(),
                badges,
            })
        })
        .collect::<Vec<_>>();

    let open_detail = context
        .open_detail
        .as_ref()
        .and_then(|target| project_context_open_detail(context, target));

    Some(ContextTimelineView {
        active_label: active_node.as_ref().map(|(label, _)| label.clone()),
        active_archived: active_node.map(|(_, archived)| archived).unwrap_or(false),
        node_lines,
        block_lines,
        open_detail,
    })
}

fn project_context_open_detail(
    context: &ContextPaneState,
    target: &ContextDetailTarget,
) -> Option<ContextOpenDetailView> {
    match target {
        ContextDetailTarget::Block(block_id) => {
            let block = context
                .view
                .blocks
                .iter()
                .find(|(candidate, _)| candidate.as_str() == block_id)
                .map(|(_, block)| block)?;
            let mut badges = Vec::new();
            match context.view.view_state.status(&block.block_id) {
                Some(ContextViewStatus::Pinned) => badges.push("Pinned".into()),
                Some(ContextViewStatus::Archived) => badges.push("Archived".into()),
                Some(ContextViewStatus::Resolved) => badges.push("Resolved".into()),
                Some(ContextViewStatus::RemovedFromView) => return None,
                _ => {}
            }
            if block.folded_output_id.is_some() {
                badges.push("Folded output".into());
            }
            if block.is_protected() {
                badges.push("Protected".into());
            }
            let mut lines = vec![truncate_context_line(&block.detail, 120)];
            lines.extend(context_block_source_lines(block, &context.view));
            if let Some(output_id) = block.folded_output_id.as_deref()
                && let Some(opened) = context
                    .view
                    .open_folded_output(output_id, context_view::DEFAULT_OPEN_CONTENT_MAX_BYTES)
            {
                lines.push(format!("Open detail · {} bytes", opened.returned_bytes));
                lines.extend(
                    opened
                        .content
                        .lines()
                        .take(3)
                        .map(|line| truncate_context_line(line, 120)),
                );
            }
            Some(ContextOpenDetailView {
                title: block.title.clone(),
                badges,
                lines,
            })
        }
        ContextDetailTarget::Summary(artifact_id) => {
            let artifact = context.view.open_summary_artifact(artifact_id)?;
            let mut lines = vec![truncate_context_line(&artifact.summary, 120)];
            if let Some(node_id) = artifact.source_node_id.as_deref() {
                lines.push(format!("Source · {node_id}"));
            }
            if let Some(block_id) = artifact.source_block_id.as_deref() {
                lines.push(format!("Block · {block_id}"));
            }
            Some(ContextOpenDetailView {
                title: format!("Summary {}", artifact.artifact_id),
                badges: vec!["Summary".into()],
                lines,
            })
        }
        ContextDetailTarget::FoldedOutput(output_id) => {
            if !folded_output_visible(context, output_id) {
                return None;
            }
            let metadata = context.view.folded_outputs.get(output_id)?;
            let opened = context
                .view
                .open_folded_output(output_id, context_view::DEFAULT_OPEN_CONTENT_MAX_BYTES)?;
            let mut lines = folded_output_source_lines(metadata);
            lines.push(format!("Open detail · {} bytes", opened.returned_bytes));
            lines.extend(
                opened
                    .content
                    .lines()
                    .take(3)
                    .map(|line| truncate_context_line(line, 120)),
            );
            Some(ContextOpenDetailView {
                title: format!("Folded output {}", metadata.output_id),
                badges: vec!["Folded output".into()],
                lines,
            })
        }
        ContextDetailTarget::Node(node_id) => {
            let node = context
                .tree
                .nodes()
                .find(|node| node.node_id.as_str() == node_id)?;
            let mut badges = Vec::new();
            if context.tree.active_node_id() == Some(&node.node_id) {
                badges.push("Active".into());
            }
            if node.status == ContextNodeStatus::Archived {
                badges.push("Archived".into());
            }
            let mut lines = Vec::new();
            if let Some(purpose) = node.purpose.as_deref() {
                lines.push(truncate_context_line(purpose, 120));
            }
            if let Some(source_ref) = node.source_ref.as_ref() {
                lines.push(match source_ref.source_id.as_deref() {
                    Some(source_id) => format!("Source · {}:{}", source_ref.source_kind, source_id),
                    None => format!("Source · {}", source_ref.source_kind),
                });
            }
            Some(ContextOpenDetailView {
                title: node
                    .label
                    .clone()
                    .unwrap_or_else(|| node.node_id.as_str().to_string()),
                badges,
                lines,
            })
        }
    }
}

fn context_node_depth(tree: &ContextTreeState, node_id: &str) -> usize {
    let mut depth = 0usize;
    let mut current = tree
        .nodes()
        .find(|node| node.node_id.as_str() == node_id)
        .and_then(|node| node.parent_node_id.clone());
    while let Some(parent_id) = current {
        if parent_id == *tree.root_node_id() {
            break;
        }
        depth = depth.saturating_add(1);
        current = tree
            .node(&parent_id)
            .and_then(|node| node.parent_node_id.clone());
    }
    depth
}

fn context_block_source_lines(block: &ContextBlock, view: &ContextViewProjection) -> Vec<String> {
    let mut lines = Vec::new();
    match &block.source {
        ContextBlockSource::TranscriptSpan {
            start_sequence,
            end_sequence,
        } => lines.push(format!(
            "Source · transcript @{}–@{}",
            start_sequence, end_sequence
        )),
        ContextBlockSource::SummaryArtifact { artifact_id } => {
            lines.push(format!("Source · summary {artifact_id}"));
            if let Some(artifact) = view.open_summary_artifact(artifact_id) {
                if let Some(node_id) = artifact.source_node_id.as_deref() {
                    lines.push(format!("Node · {node_id}"));
                }
                if let Some(source_block_id) = artifact.source_block_id.as_deref() {
                    lines.push(format!("Block · {source_block_id}"));
                }
            }
        }
        ContextBlockSource::FoldedOutput { output_id } => {
            lines.push(format!("Source · folded output {output_id}"));
            if let Some(metadata) = view.folded_outputs.get(output_id) {
                lines.extend(folded_output_source_lines(metadata));
            }
        }
    }
    lines
}

fn folded_output_source_lines(metadata: &FoldedOutputMetadata) -> Vec<String> {
    let mut lines = Vec::new();
    if let Some(tool_name) = metadata.tool_name.as_deref() {
        lines.push(format!("Tool · {tool_name}"));
    }
    if let Some(stream) = metadata.stream.as_deref() {
        lines.push(format!("Stream · {stream}"));
    }
    if let Some(command) = metadata.shell_command.as_deref() {
        lines.push(truncate_context_line(&format!("Command · {command}"), 120));
    }
    lines
}

fn truncate_context_line(text: &str, max_chars: usize) -> String {
    let trimmed = text.split_whitespace().collect::<Vec<_>>().join(" ");
    if trimmed.chars().count() <= max_chars {
        return trimmed;
    }
    let mut out = trimmed.chars().take(max_chars).collect::<String>();
    out.push('…');
    out
}

fn child_transcript_model(records: &[TranscriptRecord]) -> Option<String> {
    let mut model = None;
    for record in records {
        match &record.event {
            TranscriptEvent::SessionStarted {
                model: session_model,
            } => model = Some(session_model.clone()),
            TranscriptEvent::ModelChanged { new_model, .. } => model = Some(new_model.clone()),
            _ => {}
        }
    }
    model
}

impl From<TokenUsageEvent> for ModelTokenUsage {
    fn from(event: TokenUsageEvent) -> Self {
        Self {
            used_tokens: event.used_tokens,
            context_window_tokens: event.context_window_tokens,
            input_tokens: event.input_tokens,
            output_tokens: event.output_tokens,
            cached_tokens: event.cached_tokens,
        }
    }
}

trait FooterStatusExt {
    fn streaming() -> Self;
    fn ready_for_next_prompt() -> Self;
    fn preparing_tool(tool_name: &str) -> Self;
    fn running_tool(tool_name: &str, summary: &str) -> Self;
    fn tool_finished(tool_name: &str, success: bool) -> Self;
    fn tool_cancelled(tool_name: &str) -> Self;
    fn permission_resolved(approved: bool) -> Self;
    fn error(message: &str) -> Self;
    fn process_issue(message: &str, detail: Option<&str>, action: Option<&str>) -> Self;
}

impl FooterStatusExt for FooterStatus {
    fn streaming() -> Self {
        Self {
            summary: "Streaming response".into(),
            detail: Some("Assistant output is still arriving".into()),
            is_error: false,
        }
    }

    fn ready_for_next_prompt() -> Self {
        Self {
            summary: "Ready".into(),
            detail: Some("Enter a prompt when the runtime loop is wired".into()),
            is_error: false,
        }
    }

    fn preparing_tool(tool_name: &str) -> Self {
        Self {
            summary: format!("Preparing tool: {tool_name}"),
            detail: Some("Tool input is still arriving".into()),
            is_error: false,
        }
    }

    fn running_tool(tool_name: &str, summary: &str) -> Self {
        Self {
            summary: format!("Running tool: {tool_name}"),
            detail: Some(summary.to_string()),
            is_error: false,
        }
    }

    fn tool_finished(tool_name: &str, success: bool) -> Self {
        Self {
            summary: if success {
                format!("Tool finished: {tool_name}")
            } else {
                format!("Tool failed: {tool_name}")
            },
            detail: None,
            is_error: !success,
        }
    }

    fn tool_cancelled(tool_name: &str) -> Self {
        Self {
            summary: format!("Tool cancelled: {tool_name}"),
            detail: Some("The model stream ended before a complete tool call was received".into()),
            is_error: false,
        }
    }

    fn permission_resolved(approved: bool) -> Self {
        Self {
            summary: if approved {
                "Permission approved".into()
            } else {
                "Permission denied".into()
            },
            detail: None,
            is_error: !approved,
        }
    }

    fn error(message: &str) -> Self {
        Self {
            summary: "Error".into(),
            detail: Some(message.to_string()),
            is_error: true,
        }
    }

    fn process_issue(message: &str, detail: Option<&str>, action: Option<&str>) -> Self {
        let detail = match (detail, action) {
            (Some(detail), Some(action)) => Some(format!("{detail} · {action}")),
            (Some(detail), None) => Some(detail.to_string()),
            (None, Some(action)) => Some(action.to_string()),
            (None, None) => None,
        };
        Self {
            summary: message.to_string(),
            detail,
            is_error: true,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::{AutoContinueState, TodoItem, TodoStatus};
    use crate::transcript::{TranscriptEvent, TranscriptRecord};
    use crate::tui::events::{
        AppEvent, AutoContinueChangedEvent, ContextTreeUpdatedEvent, PermissionResolutionEvent,
        ProcessIssueEvent, TodoSnapshotEvent, ToolCancelledEvent, ToolPendingEvent,
    };

    #[test]
    fn tool_pending_updates_state_and_footer() {
        let mut state = TuiState::default();

        state.apply_event(AppEvent::ToolPending(ToolPendingEvent::new(
            "call-pending",
            "edit__apply_patch",
        )));

        assert_eq!(state.phase, AppPhase::Running);
        assert_eq!(state.active_tool_call_id.as_deref(), Some("call-pending"));
        assert_eq!(
            state.footer_status.summary,
            "Preparing tool: edit__apply_patch"
        );
        assert_eq!(
            state.footer_status.detail.as_deref(),
            Some("Tool input is still arriving")
        );
        assert!(matches!(
            state.timeline.items().last(),
            Some(crate::tui::timeline::TimelineItem::Tool(tool))
                if tool.call_id == "call-pending"
                    && tool.status == crate::tui::timeline::ToolExecutionStatus::Pending
        ));
    }

    #[test]
    fn process_issue_keeps_running_phase_and_marks_footer_error() {
        let mut state = TuiState::default();
        state.phase = AppPhase::Running;

        state.apply_event(AppEvent::ProcessIssue(ProcessIssueEvent {
            message: "Model stream interrupted".into(),
            detail: Some("Partial assistant output was preserved".into()),
            action: Some("Continuing with a fresh model iteration".into()),
        }));

        assert_eq!(state.phase, AppPhase::Running);
        assert_eq!(state.footer_status.summary, "Model stream interrupted");
        assert!(state.footer_status.is_error);
    }

    #[test]
    fn tool_cancelled_updates_pending_tool_and_footer() {
        let mut state = TuiState::default();
        state.apply_event(AppEvent::ToolPending(ToolPendingEvent::new(
            "call-pending",
            "edit__apply_patch",
        )));

        state.apply_event(AppEvent::ToolCancelled(ToolCancelledEvent::new(
            "call-pending",
            "edit__apply_patch",
        )));

        assert_eq!(state.active_tool_call_id, None);
        assert_eq!(
            state.footer_status.summary,
            "Tool cancelled: edit__apply_patch"
        );
        assert_eq!(
            state.footer_status.detail.as_deref(),
            Some("The model stream ended before a complete tool call was received")
        );
        assert!(matches!(
            state.timeline.items().last(),
            Some(crate::tui::timeline::TimelineItem::Tool(tool))
                if tool.call_id == "call-pending"
                    && tool.status == crate::tui::timeline::ToolExecutionStatus::Cancelled
        ));
    }

    #[test]
    fn tool_output_preference_clears_transcript_render_cache() {
        let mut state = TuiState::default();
        state
            .transcript_render_cache
            .prepare(80, crate::tui::Theme::dark(), 1);
        assert!(!state.transcript_render_cache.is_empty());

        state.set_tool_output_expanded(true);

        assert!(state.tool_output_expanded);
        assert!(state.transcript_render_cache.is_empty());
    }

    #[test]
    fn permission_resolved_clears_active_tool_and_pending_permission() {
        let mut state = TuiState::default();
        let request = PermissionRequestEvent::new("call-1", "shell__exec", "run ls");

        state
            .set_pending_permission_projection(Some(PermissionView::from_request(request.clone())));
        state.apply_event(AppEvent::PermissionRequested(request));
        assert_eq!(state.phase, AppPhase::WaitingForPermission);
        assert_eq!(state.active_tool_call_id.as_deref(), Some("call-1"));
        assert!(state.pending_permission.is_some());

        state.apply_event(AppEvent::PermissionResolved(
            PermissionResolutionEvent::approved("call-1"),
        ));

        assert_eq!(state.phase, AppPhase::Running);
        assert_eq!(state.active_tool_call_id, None);
        assert!(state.pending_permission.is_none());
        assert_eq!(state.footer_status.summary, "Permission approved");
        let permission = state
            .timeline
            .items()
            .iter()
            .find_map(|item| match item {
                crate::tui::timeline::TimelineItem::Permission(permission) => Some(permission),
                _ => None,
            })
            .expect("permission item exists");
        assert_eq!(
            permission.status,
            crate::tui::timeline::PermissionPromptStatus::Approved
        );
    }

    #[test]
    fn interrupted_event_cancels_active_tool_and_clears_permission() {
        let mut state = TuiState::default();
        state.apply_event(AppEvent::ToolPending(ToolPendingEvent::new(
            "call-1",
            "shell__exec",
        )));
        state.set_pending_permission_projection(Some(PermissionView::from_request(
            PermissionRequestEvent::new("call-1", "shell__exec", "run ls"),
        )));
        state.apply_event(AppEvent::PermissionRequested(PermissionRequestEvent::new(
            "call-1",
            "shell__exec",
            "run ls",
        )));

        state.apply_event(AppEvent::Interrupted);

        assert_eq!(state.phase, AppPhase::Completed);
        assert_eq!(state.active_tool_call_id, None);
        assert!(state.pending_permission.is_none());
        assert!(matches!(
            state.timeline.items().iter().find_map(|item| match item {
                crate::tui::timeline::TimelineItem::Tool(tool) => Some(tool),
                _ => None,
            }),
            Some(tool) if tool.status == crate::tui::timeline::ToolExecutionStatus::Cancelled
        ));
    }

    #[test]
    fn unseen_late_tool_events_do_not_revive_parent_state_after_interrupt() {
        let mut state = TuiState::default();
        state.apply_event(AppEvent::ToolPending(ToolPendingEvent::new(
            "call-1",
            "shell__exec",
        )));
        state.apply_event(AppEvent::Interrupted);

        state.apply_event(AppEvent::ToolFinished(
            crate::tui::events::ToolFinishedEvent::new(
                "late-call",
                "fs__write",
                "fs__write completed",
                ToolOutcome::Success,
            ),
        ));

        assert_eq!(state.phase, AppPhase::Completed);
        assert_eq!(state.active_tool_call_id, None);
        assert_eq!(state.footer_status.summary, "Interrupted");
        assert_eq!(
            state
                .timeline
                .items()
                .iter()
                .filter(|item| matches!(item, crate::tui::timeline::TimelineItem::Tool(_)))
                .count(),
            1
        );
    }

    #[test]
    fn unseen_late_tool_cancelled_does_not_override_interrupt_state() {
        let mut state = TuiState::default();
        state.apply_event(AppEvent::ToolPending(ToolPendingEvent::new(
            "call-1",
            "shell__exec",
        )));
        state.apply_event(AppEvent::Interrupted);

        state.apply_event(AppEvent::ToolCancelled(ToolCancelledEvent::new(
            "late-call",
            "fs__write",
        )));

        assert_eq!(state.phase, AppPhase::Completed);
        assert_eq!(state.active_tool_call_id, None);
        assert_eq!(state.footer_status.summary, "Interrupted");
        assert_eq!(
            state
                .timeline
                .items()
                .iter()
                .filter(|item| matches!(item, crate::tui::timeline::TimelineItem::Tool(_)))
                .count(),
            1
        );
    }

    #[test]
    fn unseen_late_tool_events_do_not_revive_child_state_after_interrupt() {
        let mut state = TuiState::default();
        state.replace_child_timeline_from_records(
            &[],
            "parent-session",
            "child-session",
            "explorer",
            0,
            1,
        );
        state.apply_child_app_event(
            "child-session",
            AppEvent::ToolPending(ToolPendingEvent::new("call-1", "shell__exec")),
        );
        state.apply_child_app_event("child-session", AppEvent::Interrupted);

        state.apply_child_app_event(
            "child-session",
            AppEvent::ToolFinished(crate::tui::events::ToolFinishedEvent::new(
                "late-call",
                "fs__write",
                "fs__write completed",
                ToolOutcome::Success,
            )),
        );

        assert_eq!(state.phase, AppPhase::Completed);
        assert_eq!(state.active_tool_call_id, None);
        assert_eq!(state.footer_status.summary, "Interrupted");
        assert!(matches!(
            state.active_timeline().items().iter().find_map(|item| match item {
                crate::tui::timeline::TimelineItem::Tool(tool) => Some(tool),
                _ => None,
            }),
            Some(tool) if tool.status == crate::tui::timeline::ToolExecutionStatus::Cancelled
        ));
        assert_eq!(
            state
                .active_timeline()
                .items()
                .iter()
                .filter(|item| matches!(item, crate::tui::timeline::TimelineItem::Tool(_)))
                .count(),
            1
        );
    }

    #[test]
    fn transcript_scroll_uses_bottom_relative_offset() {
        let mut state = TuiState::default();

        assert_eq!(state.transcript_scroll_offset(), 0);
        assert!(state.auto_scroll);

        state.scroll_transcript_up(3);
        assert_eq!(state.transcript_scroll_offset(), 3);
        assert!(!state.auto_scroll);

        state.scroll_transcript_down(2);
        assert_eq!(state.transcript_scroll_offset(), 1);
        assert!(!state.auto_scroll);

        state.scroll_transcript_down(10);
        assert_eq!(state.transcript_scroll_offset(), 0);
        assert!(state.auto_scroll);
    }

    #[test]
    fn transcript_append_preserves_manual_scroll_offset() {
        let mut state = TuiState::default();
        state.scroll_transcript_up(4);

        state.apply_event(AppEvent::UserMessage(UserMessageEvent::new("hello")));

        assert_eq!(state.transcript_scroll_offset(), 4);
        assert!(!state.auto_scroll);
        assert_eq!(state.timeline.items().len(), 1);
    }

    #[test]
    fn toast_replaces_previous_message_and_resets_lifetime() {
        let mut state = TuiState::default();
        state.show_toast("Copied to clipboard", ToastKind::Success);
        state.apply_event(AppEvent::Tick);
        let remaining_after_one_tick = state
            .toast()
            .expect("toast remains visible after one tick")
            .ticks_remaining();

        state.show_toast("Copy failed", ToastKind::Error);

        let toast = state.toast().expect("replacement toast exists");
        assert_eq!(toast.message, "Copy failed");
        assert_eq!(toast.kind, ToastKind::Error);
        assert_eq!(toast.ticks_remaining(), ToastState::DEFAULT_TICKS);
        assert!(remaining_after_one_tick < ToastState::DEFAULT_TICKS);
    }

    #[test]
    fn toast_auto_dismisses_after_ticks() {
        let mut state = TuiState::default();
        state.toast = Some(ToastState::new(
            "Copied to clipboard",
            ToastKind::Success,
            2,
        ));

        state.apply_event(AppEvent::Tick);
        assert!(state.toast().is_some());

        state.apply_event(AppEvent::Tick);
        assert!(state.toast().is_none());
    }

    #[test]
    fn manual_transcript_viewport_tracks_new_rows_without_top_shift() {
        let mut state = TuiState::default();
        state.sync_transcript_viewport_rows(100);
        state.scroll_transcript_up(4);

        state.sync_transcript_viewport_rows(103);

        assert_eq!(state.transcript_scroll_offset(), 7);
        assert!(!state.auto_scroll);
    }

    #[test]
    fn transcript_bottom_detection_handles_fitting_and_scrolled_content() {
        let state = TuiState::default();

        assert!(state.transcript_is_at_bottom(3, 10));
        assert!(state.transcript_is_at_bottom(20, 5));

        let mut scrolled = TuiState::default();
        scrolled.scroll_transcript_up(5);

        assert!(!scrolled.transcript_is_at_bottom(20, 5));
        assert!(scrolled.transcript_is_at_bottom(3, 10));
    }

    #[test]
    fn slash_panel_opens_dismisses_and_reopens_when_query_changes() {
        let mut state = TuiState::default();

        state.set_input("/");
        assert!(state.slash_panel_is_open());

        state.dismiss_slash_panel();
        assert!(!state.slash_panel_is_open());

        state.set_input("/p");
        assert!(state.slash_panel_is_open());
        assert_eq!(state.slash_panel_selected, 0);
    }

    #[test]
    fn expert_panel_opens_in_parent_view_and_filters_query() {
        let mut state = TuiState::default();

        state.set_input("@fi");

        assert!(state.slash_panel_is_open());
        assert_eq!(state.slash_panel_query, "@fi");
    }

    #[test]
    fn slash_panel_is_hidden_in_child_view() {
        let mut state = TuiState::default();

        state.set_input("/p");
        assert!(state.slash_panel_is_open());

        state.replace_child_timeline_from_records(
            &[],
            "parent-session",
            "child-session",
            "explorer",
            0,
            1,
        );

        assert!(!state.slash_panel_is_open());
        assert!(state.input_buffer.is_empty());
    }

    #[test]
    fn expert_panel_is_hidden_in_child_view() {
        let mut state = TuiState::default();
        state.set_input("@or");
        assert!(state.slash_panel_is_open());

        state.replace_child_timeline_from_records(
            &[],
            "parent-session",
            "child-session",
            "explorer",
            0,
            1,
        );

        assert!(!state.slash_panel_is_open());
    }

    #[test]
    fn todo_events_update_latest_state_and_timeline() {
        let mut state = TuiState::default();

        state.apply_event(AppEvent::AutoContinueChanged(
            AutoContinueChangedEvent::new(AutoContinueState {
                enabled: true,
                max_continuations: 2,
            }),
        ));
        assert!(state.latest_todo.is_none());
        assert_eq!(state.timeline.items().len(), 0);

        state.apply_event(AppEvent::TodoSnapshot(TodoSnapshotEvent::new(vec![
            TodoItem {
                id: "t1".into(),
                content: "inspect".into(),
                status: TodoStatus::Pending,
            },
        ])));

        let todo = state.latest_todo.as_ref().expect("todo state exists");
        assert!(todo.auto_continue.enabled);
        assert_eq!(todo.auto_continue.max_continuations, 2);
        assert_eq!(todo.items.len(), 1);
        assert!(matches!(
            state.timeline.items().last(),
            Some(crate::tui::timeline::TimelineItem::Todo(todo)) if todo.items.len() == 1
        ));

        state.apply_event(AppEvent::UserMessage(UserMessageEvent::new("next turn")));
        assert!(state.latest_todo.is_none());
        assert!(!state.latest_auto_continue.enabled);
    }

    #[test]
    fn replacing_child_and_parent_timelines_updates_transcript_view_state() {
        let parent_records = vec![TranscriptRecord {
            session_id: "parent-session".into(),
            sequence: 1,
            timestamp_ms: 0,
            context_branch_id: None,
            event: TranscriptEvent::UserMessage {
                content: "parent prompt".into(),
            },
        }];
        let child_records = vec![TranscriptRecord {
            session_id: "child-session".into(),
            sequence: 1,
            timestamp_ms: 1,
            context_branch_id: None,
            event: TranscriptEvent::AssistantMessage {
                content: "child response".into(),
            },
        }];

        let mut state = TuiState::default();
        state.replace_session_timeline_from_records(&parent_records);
        state.replace_child_timeline_from_records(
            &child_records,
            "parent-session",
            "child-session",
            "explorer",
            2,
            3,
        );

        assert!(matches!(
            state.transcript_view,
            TranscriptViewState::Child {
                ref parent_session_id,
                ref child_session_id,
                ref agent_name,
                index: 2,
                total: 3,
            } if parent_session_id == "parent-session"
                && child_session_id == "child-session"
                && agent_name == "explorer"
        ));
        let metadata = state.child_view_metadata().expect("child metadata");
        assert_eq!(metadata.model, None);
        assert_eq!(metadata.record_count, 1);
        assert!(matches!(
            state.active_timeline().items().first(),
            Some(crate::tui::timeline::TimelineItem::Assistant(message))
                if message.text == "child response"
        ));

        state.apply_event(AppEvent::Error(crate::tui::events::ErrorEvent::new(
            "parent failure",
        )));
        assert!(matches!(
            state.active_timeline().items().first(),
            Some(crate::tui::timeline::TimelineItem::Assistant(message))
                if message.text == "child response"
        ));
        assert!(matches!(
            state.timeline.items().last(),
            Some(crate::tui::timeline::TimelineItem::Error(error))
                if error.message == "parent failure"
        ));

        state.restore_parent_timeline_view();
        assert_eq!(state.transcript_view, TranscriptViewState::Parent);
        assert!(matches!(
            state.active_timeline().items().first(),
            Some(crate::tui::timeline::TimelineItem::User(message))
                if message.text == "parent prompt"
        ));

        state.replace_session_timeline_from_records(&parent_records);
        assert_eq!(state.transcript_view, TranscriptViewState::Parent);
    }

    #[test]
    fn replacing_child_timeline_projects_child_context_immediately() {
        let child_records = vec![
            TranscriptRecord {
                session_id: "child-session".into(),
                sequence: 1,
                timestamp_ms: 0,
                context_branch_id: None,
                event: TranscriptEvent::ContextNodeCreated {
                    node_id: "child-node".into(),
                    parent_node_id: Some("root".into()),
                    label: Some("Child lane".into()),
                    purpose: Some("Review child context".into()),
                    block_ref: None,
                    source_ref: None,
                },
            },
            TranscriptRecord {
                session_id: "child-session".into(),
                sequence: 2,
                timestamp_ms: 1,
                context_branch_id: None,
                event: TranscriptEvent::ContextNodeLifecycle {
                    node_id: "root".into(),
                    status: ContextNodeStatus::Inactive,
                },
            },
            TranscriptRecord {
                session_id: "child-session".into(),
                sequence: 3,
                timestamp_ms: 2,
                context_branch_id: None,
                event: TranscriptEvent::ContextNodeLifecycle {
                    node_id: "child-node".into(),
                    status: ContextNodeStatus::Active,
                },
            },
        ];

        let mut state = TuiState::default();
        state.replace_child_timeline_from_records(
            &child_records,
            "parent-session",
            "child-session",
            "explorer",
            0,
            1,
        );

        assert!(matches!(
            state.active_timeline().items().first(),
            Some(crate::tui::timeline::TimelineItem::Context(context))
                if context.active_label.as_deref() == Some("Child lane")
        ));
    }

    #[test]
    fn parent_context_update_while_viewing_child_updates_parent_timeline() {
        let parent_records = vec![TranscriptRecord {
            session_id: "parent-session".into(),
            sequence: 1,
            timestamp_ms: 0,
            context_branch_id: None,
            event: TranscriptEvent::UserMessage {
                content: "parent prompt".into(),
            },
        }];
        let child_records = vec![
            TranscriptRecord {
                session_id: "child-session".into(),
                sequence: 1,
                timestamp_ms: 0,
                context_branch_id: None,
                event: TranscriptEvent::ContextNodeCreated {
                    node_id: "child-node".into(),
                    parent_node_id: Some("root".into()),
                    label: Some("Child lane".into()),
                    purpose: None,
                    block_ref: None,
                    source_ref: None,
                },
            },
            TranscriptRecord {
                session_id: "child-session".into(),
                sequence: 2,
                timestamp_ms: 1,
                context_branch_id: None,
                event: TranscriptEvent::ContextNodeLifecycle {
                    node_id: "root".into(),
                    status: ContextNodeStatus::Inactive,
                },
            },
            TranscriptRecord {
                session_id: "child-session".into(),
                sequence: 3,
                timestamp_ms: 2,
                context_branch_id: None,
                event: TranscriptEvent::ContextNodeLifecycle {
                    node_id: "child-node".into(),
                    status: ContextNodeStatus::Active,
                },
            },
        ];
        let parent_context_records = vec![
            TranscriptRecord {
                session_id: "parent-session".into(),
                sequence: 2,
                timestamp_ms: 1,
                context_branch_id: None,
                event: TranscriptEvent::ContextNodeCreated {
                    node_id: "parent-node".into(),
                    parent_node_id: Some("root".into()),
                    label: Some("Parent lane".into()),
                    purpose: None,
                    block_ref: None,
                    source_ref: None,
                },
            },
            TranscriptRecord {
                session_id: "parent-session".into(),
                sequence: 3,
                timestamp_ms: 2,
                context_branch_id: None,
                event: TranscriptEvent::ContextNodeLifecycle {
                    node_id: "root".into(),
                    status: ContextNodeStatus::Inactive,
                },
            },
            TranscriptRecord {
                session_id: "parent-session".into(),
                sequence: 4,
                timestamp_ms: 3,
                context_branch_id: None,
                event: TranscriptEvent::ContextNodeLifecycle {
                    node_id: "parent-node".into(),
                    status: ContextNodeStatus::Active,
                },
            },
        ];

        let mut state = TuiState::default();
        state.replace_session_timeline_from_records(&parent_records);
        state.replace_child_timeline_from_records(
            &child_records,
            "parent-session",
            "child-session",
            "explorer",
            0,
            1,
        );
        assert!(matches!(
            state.active_timeline().items().first(),
            Some(crate::tui::timeline::TimelineItem::Context(context))
                if context.active_label.as_deref() == Some("Child lane")
        ));

        let parent_context = project_context_pane(&parent_context_records).unwrap();
        state.apply_event(AppEvent::ContextTreeUpdated(ContextTreeUpdatedEvent {
            tree: parent_context.tree,
        }));

        assert!(matches!(
            state.active_timeline().items().first(),
            Some(crate::tui::timeline::TimelineItem::Context(context))
                if context.active_label.as_deref() == Some("Child lane")
        ));
        assert!(matches!(
            state.timeline.items().first(),
            Some(crate::tui::timeline::TimelineItem::Context(context))
                if context.active_label.as_deref() == Some("Parent lane")
        ));

        state.restore_parent_timeline_view();
        assert!(matches!(
            state.active_timeline().items().first(),
            Some(crate::tui::timeline::TimelineItem::Context(context))
                if context.active_label.as_deref() == Some("Parent lane")
        ));
    }

    #[test]
    fn child_view_metadata_prefers_latest_model_change() {
        let records = vec![
            TranscriptRecord {
                session_id: "child-session".into(),
                sequence: 1,
                timestamp_ms: 0,
                context_branch_id: None,
                event: TranscriptEvent::SessionStarted {
                    model: "gpt-5.5".into(),
                },
            },
            TranscriptRecord {
                session_id: "child-session".into(),
                sequence: 2,
                timestamp_ms: 1,
                context_branch_id: None,
                event: TranscriptEvent::ModelChanged {
                    previous_model: "gpt-5.5".into(),
                    new_model: "gpt-5.5-mini".into(),
                },
            },
        ];
        let mut state = TuiState::default();
        state.replace_child_timeline_from_records(
            &records,
            "parent-session",
            "child-session",
            "fixer",
            1,
            2,
        );

        let metadata = state.child_view_metadata().expect("child metadata");
        assert_eq!(metadata.model.as_deref(), Some("gpt-5.5-mini"));
        assert_eq!(metadata.record_count, 2);
    }

    #[test]
    fn child_view_replacement_preserves_running_phase_and_pending_permission() {
        let mut state = TuiState::default();
        let request = PermissionRequestEvent::new("call-1", "shell__exec", "run ls");
        state
            .set_pending_permission_projection(Some(PermissionView::from_request(request.clone())));
        state.apply_event(AppEvent::PermissionRequested(request));
        state.open_dialog(DialogState::new(
            DialogKind::ModelPicker,
            "Model",
            None,
            vec![DialogItem::new("m1", "Model 1", None)],
        ));
        state.set_input("/p");
        state.scroll_transcript_up(3);

        state.replace_child_timeline_from_records(
            &[],
            "parent-session",
            "child-session",
            "explorer",
            0,
            1,
        );

        assert_eq!(state.phase, AppPhase::WaitingForPermission);
        assert_eq!(state.active_tool_call_id.as_deref(), Some("call-1"));
        assert!(state.pending_permission.is_some());
        assert!(state.dialog().is_none());
        assert!(state.input_buffer.is_empty());
        assert!(!state.slash_panel_is_open());
        assert_eq!(state.transcript_scroll_offset(), 0);
        assert!(state.auto_scroll);
    }

    #[test]
    fn child_view_refresh_preserves_runtime_state_and_updates_record_count() {
        let mut state = TuiState::default();
        state.phase = AppPhase::Running;
        state.active_tool_call_id = Some("call-2".into());
        state.pending_permission = Some(PermissionView::from_request(PermissionRequestEvent::new(
            "call-2",
            "shell__exec",
            "run cargo test",
        )));
        state.replace_child_timeline_from_records(
            &[],
            "parent-session",
            "child-session",
            "explorer",
            0,
            1,
        );
        state.open_dialog(DialogState::new(
            DialogKind::ModelPicker,
            "Model",
            None,
            vec![DialogItem::new("m1", "Model 1", None)],
        ));

        state.refresh_child_timeline_from_records(&[
            TranscriptRecord {
                session_id: "child-session".into(),
                sequence: 1,
                timestamp_ms: 0,
                context_branch_id: None,
                event: TranscriptEvent::SessionStarted {
                    model: "gpt-test".into(),
                },
            },
            TranscriptRecord {
                session_id: "child-session".into(),
                sequence: 2,
                timestamp_ms: 1,
                context_branch_id: None,
                event: TranscriptEvent::AssistantMessage {
                    content: "updated child output".into(),
                },
            },
        ]);

        let metadata = state.child_view_metadata().expect("child metadata");
        assert_eq!(metadata.record_count, 2);
        assert_eq!(metadata.model.as_deref(), Some("gpt-test"));
        assert_eq!(state.phase, AppPhase::WaitingForPermission);
        assert_eq!(state.active_tool_call_id.as_deref(), Some("call-2"));
        assert!(state.pending_permission.is_some());
        assert!(state.dialog().is_some());
    }

    #[test]
    fn restore_parent_view_preserves_pending_permission() {
        let mut state = TuiState::default();
        state.set_pending_permission_projection(Some(PermissionView::from_request(
            PermissionRequestEvent::new("call-1", "shell__exec", "run ls"),
        )));
        state.apply_event(AppEvent::PermissionRequested(PermissionRequestEvent::new(
            "call-1",
            "shell__exec",
            "run ls",
        )));
        state.replace_child_timeline_from_records(
            &[],
            "parent-session",
            "child-session",
            "explorer",
            0,
            1,
        );

        state.restore_parent_timeline_view();

        assert_eq!(state.phase, AppPhase::WaitingForPermission);
        assert!(state.pending_permission.is_some());
        assert_eq!(state.active_tool_call_id.as_deref(), Some("call-1"));
    }

    #[test]
    fn matching_child_app_event_updates_child_timeline_only() {
        let mut state = TuiState::default();
        state.replace_child_timeline_from_records(
            &[],
            "parent-session",
            "child-session",
            "explorer",
            0,
            1,
        );

        state.apply_child_app_event(
            "child-session",
            AppEvent::AssistantDelta(crate::tui::events::AssistantDeltaEvent::new("hello")),
        );

        assert!(state.timeline.items().is_empty());
        assert!(matches!(
            state.active_timeline().items().last(),
            Some(crate::tui::timeline::TimelineItem::Assistant(message)) if message.text == "hello"
        ));
    }

    #[test]
    fn child_tool_events_project_into_active_parent_subagent_card() {
        let mut state = TuiState::default();
        state.apply_event(AppEvent::ToolStarted(
            crate::tui::events::ToolStartedEvent::new(
                "parent-call",
                "agent__explore",
                "inspect src/tui",
            ),
        ));

        state.apply_child_app_event(
            "child-session",
            AppEvent::ToolStarted(crate::tui::events::ToolStartedEvent::new(
                "child-call",
                "shell__exec",
                "cargo build --bin letcode",
            )),
        );

        let tool = state
            .timeline
            .items()
            .iter()
            .find_map(|item| match item {
                crate::tui::timeline::TimelineItem::Tool(tool) => Some(tool),
                _ => None,
            })
            .expect("parent subagent tool exists");
        assert_eq!(tool.name, "agent__explore");
        assert_eq!(
            tool.status,
            crate::tui::timeline::ToolExecutionStatus::Running
        );
        assert_eq!(tool.summary, "shell__exec — cargo build --bin letcode");
        let output = tool.output.as_deref().expect("live summary payload exists");
        assert!(output.contains("child-session"), "{output}");
        assert!(
            output.contains("shell__exec — cargo build --bin letcode"),
            "{output}"
        );
    }

    #[test]
    fn child_tool_cancelled_projects_into_active_parent_subagent_card() {
        let mut state = TuiState::default();
        state.apply_event(AppEvent::ToolStarted(
            crate::tui::events::ToolStartedEvent::new(
                "parent-call",
                "agent__explore",
                "inspect src/tui",
            ),
        ));

        state.apply_child_app_event(
            "child-session",
            AppEvent::ToolCancelled(ToolCancelledEvent::new("child-call", "shell__exec")),
        );

        let tool = state
            .timeline
            .items()
            .iter()
            .find_map(|item| match item {
                crate::tui::timeline::TimelineItem::Tool(tool) => Some(tool),
                _ => None,
            })
            .expect("parent subagent tool exists");
        assert_eq!(tool.summary, "shell__exec cancelled");
        let output = tool.output.as_deref().expect("live summary payload exists");
        assert!(output.contains("cancelled"), "{output}");
        assert!(output.contains("child-session"), "{output}");
    }

    #[test]
    fn child_permission_events_project_into_active_parent_subagent_card() {
        let mut state = TuiState::default();
        state.apply_event(AppEvent::ToolStarted(
            crate::tui::events::ToolStartedEvent::new(
                "parent-call",
                "agent__fixer",
                "apply requested fix",
            ),
        ));

        let mut request =
            PermissionRequestEvent::new("perm-1", "shell__exec", "run cargo test --bin letcode");
        request.rationale = Some("validation".into());

        state.apply_child_app_event("child-session", AppEvent::PermissionRequested(request));

        let tool = state
            .timeline
            .items()
            .iter()
            .find_map(|item| match item {
                crate::tui::timeline::TimelineItem::Tool(tool) => Some(tool),
                _ => None,
            })
            .expect("parent subagent tool exists");
        let output = tool.output.as_deref().expect("live summary payload exists");
        assert!(output.contains("approval needed"), "{output}");
        assert!(output.contains("shell__exec"), "{output}");
        assert!(output.contains("run cargo test --bin letcode"), "{output}");
    }

    #[test]
    fn non_matching_child_app_event_is_ignored() {
        let mut state = TuiState::default();
        state.replace_child_timeline_from_records(
            &[],
            "parent-session",
            "child-session",
            "explorer",
            0,
            1,
        );

        state.apply_child_app_event(
            "other-child",
            AppEvent::AssistantDelta(crate::tui::events::AssistantDeltaEvent::new("hello")),
        );

        assert!(state.timeline.items().is_empty());
        assert!(state.active_timeline().items().is_empty());
    }
}
