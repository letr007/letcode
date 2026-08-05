use super::components::transcript::TranscriptRenderCache;
use super::events::{
    AutoContinueChangedEvent, PermissionDecision, PermissionRequestEvent,
    PermissionResolutionEvent, SessionEvent, TokenUsageEvent, ToolOutcome, UserMessageEvent,
};
use super::measure;
use super::slash;
use super::theme::{Theme, ThemeName};
use super::timeline::{ContextOpenDetailView, PermissionView, Timeline, TimelineItem, TodoView};
use super::transcript_render::Interaction;
#[cfg(test)]
use crate::agent::ConversationMessage;
use crate::agent::{AutoContinueState, CacheUsageReport};
use crate::context_tree::{ContextNodeStatus, ContextTreeState};
use crate::context_view::{
    ContextBlock, ContextBlockSource, ContextViewProjection, ContextViewStatus,
};
use crate::runtime_context::RuntimeActiveContext;
use crate::skills::SkillCard;
use crate::tool::{QuestionOption, QuestionRequest, QuestionResponse, QuestionSpec};

#[cfg(test)]
use crate::transcript::transcript_projection;
use crate::transcript::{
    TranscriptEvent, TranscriptRecord, restore_latest_auto_continue_state,
    restore_latest_todo_snapshot,
};
use crate::user_content::{
    UserImageAttachment, UserMessageContent, UserMessagePart, UserMessageSubmission,
};

pub const COMPOSER_ATTACHMENT_MARKER: char = '\u{fffc}';
pub const COMPOSER_ATTACHMENT_MARKER_STR: &str = "\u{fffc}";

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ComposerToken {
    Image(UserImageAttachment),
    Skill(String),
}

impl ComposerToken {
    pub fn display_text(&self, image_index: usize) -> String {
        match self {
            Self::Image(_) => format!("[Image {}]", image_index + 1),
            Self::Skill(name) => format!("[Skill: {name}]"),
        }
    }

    #[cfg(test)]
    pub fn image(&self) -> Option<&UserImageAttachment> {
        match self {
            Self::Image(attachment) => Some(attachment),
            Self::Skill(_) => None,
        }
    }

    #[cfg(test)]
    pub fn skill_name(&self) -> Option<&str> {
        match self {
            Self::Image(_) => None,
            Self::Skill(name) => Some(name),
        }
    }
}

use anyhow::Result;
use std::collections::{HashMap, HashSet};

/// 文本选择范围
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TextSelection {
    pub start: SelectionAnchor,
    pub end: SelectionAnchor,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TranscriptClickTarget {
    ToolCard(String),
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

    #[cfg(test)]
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

/// Sticky retry notice shown in the top-right toast while waiting to reissue.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RetryNoticeState {
    pub attempt: usize,
    pub max_attempts: usize,
    pub delay_secs: u64,
    pub error: String,
    pub secs_remaining: u64,
    ticks_in_current_second: u8,
}

impl RetryNoticeState {
    /// Matches `TUI_FRAME_POLL_INTERVAL` (~33ms): about one displayed second.
    pub const TICKS_PER_SECOND: u8 = 30;

    pub fn from_lifecycle(event: crate::session::RetryLifecycleEvent) -> Self {
        Self {
            attempt: event.attempt,
            max_attempts: event.max_attempts,
            delay_secs: event.delay_secs,
            error: event.error,
            secs_remaining: event.delay_secs,
            ticks_in_current_second: 0,
        }
    }

    pub fn toast_message(&self) -> String {
        format!(
            "Retrying in {}s · attempt {} of {}",
            self.secs_remaining, self.attempt, self.max_attempts
        )
    }

    fn tick_frame(&mut self) {
        self.ticks_in_current_second = self.ticks_in_current_second.saturating_add(1);
        if self.ticks_in_current_second < Self::TICKS_PER_SECOND {
            return;
        }
        self.ticks_in_current_second = 0;
        self.secs_remaining = self.secs_remaining.saturating_sub(1);
    }

    fn sticky_toast(&self) -> ToastState {
        // ticks are ignored while retry is active; Tick refreshes this toast.
        ToastState::new(
            self.toast_message(),
            ToastKind::Error,
            ToastState::DEFAULT_TICKS,
        )
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelTokenUsage {
    pub used_tokens: u64,
    pub context_window_tokens: u64,
    pub input_tokens: u64,
    pub output_tokens: u64,
    pub cached_tokens: u64,
    pub cache_report: Option<CacheUsageReport>,
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
    AgentPicker,
    ExpertModelPicker(String),
    PermissionPicker,
    ReasoningPicker,
    ThemePicker,
    SessionPicker,
    HistoryTree,
    ContextPicker,
    #[allow(dead_code)] // Context detail is constructed by runtime dialog routing.
    ContextDetail,
    McpPicker,
    McpToolsPicker,
    SkillPicker,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum McpDiscoveryState {
    #[default]
    Loading,
    Ready,
    Unavailable,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ContextDetailTarget {
    Node(String),
    Block(String),
    Summary(String),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ContextPaneState {
    pub tree: ContextTreeState,
    pub view: ContextViewProjection,
    pub runtime_context: Option<RuntimeActiveContext>,
    pub open_detail: Option<ContextDetailTarget>,
}

impl Default for ContextPaneState {
    fn default() -> Self {
        Self {
            tree: ContextTreeState::with_default_root(),
            view: ContextViewProjection::default(),
            runtime_context: None,
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
        pool_ordinal: u32,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChildViewMetadata {
    pub parent_session_id: String,
    pub child_session_id: String,
    pub agent_name: String,
    pub index: usize,
    pub total: usize,
    pub pool_ordinal: u32,
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
    session_id: String,
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
    pub mcp_server_name: Option<String>,
    pub mcp_primary_query: Option<String>,
    pub mcp_primary_selected_server: Option<String>,
    pub expert_primary_query: Option<String>,
    pub expert_primary_selected_agent: Option<String>,
    pub detail_focused: bool,
    pub detail_scroll: u16,
    pub detail_scroll_max: u16,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PendingQuestionItem {
    pub question: String,
    pub header: String,
    pub options: Vec<QuestionOption>,
    pub multiple: bool,
    pub selected_labels: Vec<String>,
    pub custom_text: String,
    pub custom_cursor: usize,
    pub custom_edit_text: String,
    pub custom_edit_cursor: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PendingQuestionState {
    pub questions: Vec<PendingQuestionItem>,
    pub active_tab: usize,
    pub active_row: usize,
    pub editing_custom: bool,
    pub origin_label: Option<String>,
}

impl PendingQuestionState {
    pub fn new(request: QuestionRequest, origin_label: Option<String>) -> Self {
        Self {
            questions: request
                .questions
                .into_iter()
                .map(|question| PendingQuestionItem::from_spec(question))
                .collect(),
            active_tab: 0,
            active_row: 0,
            editing_custom: false,
            origin_label,
        }
    }

    pub fn total_tabs(&self) -> usize {
        self.questions.len() + usize::from(self.show_confirm_tab())
    }

    pub fn show_confirm_tab(&self) -> bool {
        !self.single_select_fast_path()
    }

    pub fn single_select_fast_path(&self) -> bool {
        self.questions.len() == 1
            && self
                .questions
                .first()
                .is_some_and(|question| !question.multiple)
    }

    pub fn is_confirm_tab(&self) -> bool {
        self.show_confirm_tab() && self.active_tab == self.questions.len()
    }

    pub fn active_tab_label(&self, index: usize) -> Option<&str> {
        if self.show_confirm_tab() && index == self.questions.len() {
            Some("Confirm")
        } else {
            self.questions
                .get(index)
                .map(|question| question.header.as_str())
        }
    }

    pub fn current_question(&self) -> Option<&PendingQuestionItem> {
        self.questions.get(self.active_tab)
    }

    pub fn current_question_mut(&mut self) -> Option<&mut PendingQuestionItem> {
        self.questions.get_mut(self.active_tab)
    }

    pub fn current_row_count(&self) -> usize {
        self.current_question()
            .map(|question| question.options.len() + 1)
            .unwrap_or(0)
    }

    pub fn custom_row_index(&self) -> Option<usize> {
        self.current_question()
            .map(|question| question.options.len())
    }

    pub fn active_custom_row(&self) -> bool {
        self.custom_row_index()
            .is_some_and(|custom_row| custom_row == self.active_row)
    }

    pub fn move_next_row(&mut self) {
        if self.is_confirm_tab() {
            return;
        }
        let row_count = self.current_row_count();
        if row_count > 0 {
            self.active_row = (self.active_row + 1) % row_count;
        }
    }

    pub fn move_prev_row(&mut self) {
        if self.is_confirm_tab() {
            return;
        }
        let row_count = self.current_row_count();
        if row_count > 0 {
            self.active_row = if self.active_row == 0 {
                row_count - 1
            } else {
                self.active_row - 1
            };
        }
    }

    pub fn move_next_tab(&mut self) {
        let total_tabs = self.total_tabs();
        if total_tabs == 0 {
            return;
        }
        self.editing_custom = false;
        self.active_tab = (self.active_tab + 1) % total_tabs;
        self.clamp_active_row();
    }

    pub fn move_prev_tab(&mut self) {
        let total_tabs = self.total_tabs();
        if total_tabs == 0 {
            return;
        }
        self.editing_custom = false;
        self.active_tab = if self.active_tab == 0 {
            total_tabs - 1
        } else {
            self.active_tab - 1
        };
        self.clamp_active_row();
    }

    pub fn pick_option(&mut self, option_index: usize) -> QuestionAdvance {
        let active_tab = self.active_tab;
        let questions_len = self.questions.len();
        let show_confirm = self.show_confirm_tab();
        let Some(question) = self.questions.get_mut(active_tab) else {
            return QuestionAdvance::None;
        };
        if option_index >= question.options.len() {
            return QuestionAdvance::None;
        }

        let label = question.options[option_index].label.clone();
        if question.multiple {
            if let Some(existing) = question
                .selected_labels
                .iter()
                .position(|item| item == &label)
            {
                question.selected_labels.remove(existing);
            } else {
                question.selected_labels.push(label);
            }
            return QuestionAdvance::None;
        }

        question.selected_labels.clear();
        question.selected_labels.push(label);
        question.custom_text.clear();
        question.custom_cursor = 0;
        self.editing_custom = false;

        self.advance_after_answer(active_tab, questions_len, show_confirm)
    }

    pub fn pick_row(&mut self, row_index: usize) -> QuestionAdvance {
        if self
            .current_question()
            .is_some_and(|question| row_index == question.options.len())
        {
            return self.activate_custom_row();
        }

        self.pick_option(row_index)
    }

    pub fn activate_custom_row(&mut self) -> QuestionAdvance {
        if !self.active_custom_row() {
            return QuestionAdvance::None;
        }
        self.begin_custom_edit();
        QuestionAdvance::Editing
    }

    pub fn begin_custom_edit(&mut self) {
        if self.active_custom_row() {
            self.editing_custom = true;
            if let Some(question) = self.current_question_mut() {
                question.custom_edit_text = question.custom_text.clone();
                question.custom_edit_cursor = question.custom_edit_text.len();
            }
        }
    }

    pub fn stop_custom_edit(&mut self) {
        self.editing_custom = false;
    }

    pub fn commit_custom_answer(&mut self) -> QuestionAdvance {
        let active_tab = self.active_tab;
        let questions_len = self.questions.len();
        let show_confirm = self.show_confirm_tab();
        let Some(question) = self.questions.get_mut(active_tab) else {
            return QuestionAdvance::None;
        };
        let custom = question.custom_edit_text.trim().to_string();
        if !question.multiple && !custom.is_empty() {
            question.selected_labels.clear();
        }
        if custom.is_empty() {
            question.custom_text.clear();
            question.custom_cursor = 0;
        } else {
            question.custom_text = custom.clone();
            question.custom_cursor = custom.len();
        }
        question.custom_edit_text = question.custom_text.clone();
        question.custom_edit_cursor = question.custom_cursor;
        self.editing_custom = false;

        if custom.is_empty() {
            return QuestionAdvance::None;
        }

        if question.multiple && questions_len == 1 {
            return QuestionAdvance::None;
        }

        self.advance_after_answer(active_tab, questions_len, show_confirm)
    }

    pub fn build_response(&self) -> QuestionResponse {
        QuestionResponse {
            answers: self
                .questions
                .iter()
                .map(PendingQuestionItem::answers)
                .collect(),
        }
    }

    pub fn all_answered(&self) -> bool {
        self.questions.iter().all(PendingQuestionItem::is_answered)
    }

    pub fn first_unanswered_tab(&self) -> Option<usize> {
        self.questions
            .iter()
            .position(|question| !question.is_answered())
    }

    pub fn has_invalid_single_response(&self) -> bool {
        self.questions
            .iter()
            .any(|question| !question.multiple && question.answers().len() > 1)
    }

    pub fn focus_tab(&mut self, tab_index: usize) {
        if tab_index >= self.total_tabs() {
            return;
        }

        self.editing_custom = false;
        self.active_tab = tab_index;
        self.clamp_active_row();
    }

    fn advance_after_answer(
        &mut self,
        active_tab: usize,
        questions_len: usize,
        show_confirm: bool,
    ) -> QuestionAdvance {
        if questions_len == 1 && !show_confirm {
            QuestionAdvance::Submit
        } else if active_tab + 1 < questions_len {
            self.active_tab += 1;
            self.active_row = 0;
            QuestionAdvance::Advanced
        } else if show_confirm {
            self.active_tab = questions_len;
            self.active_row = 0;
            QuestionAdvance::Advanced
        } else {
            QuestionAdvance::None
        }
    }

    fn clamp_active_row(&mut self) {
        if self.is_confirm_tab() {
            self.active_row = 0;
            return;
        }
        let row_count = self.current_row_count();
        if row_count == 0 {
            self.active_row = 0;
        } else {
            self.active_row = self.active_row.min(row_count - 1);
        }
    }
}

impl PendingQuestionItem {
    fn from_spec(question: QuestionSpec) -> Self {
        Self {
            question: question.question,
            header: question.header,
            options: question.options,
            multiple: question.multiple,
            selected_labels: Vec::new(),
            custom_text: String::new(),
            custom_cursor: 0,
            custom_edit_text: String::new(),
            custom_edit_cursor: 0,
        }
    }

    pub fn answers(&self) -> Vec<String> {
        let custom = self.custom_text.trim();
        if self.multiple {
            let mut answers = self.selected_labels.clone();
            if !custom.is_empty() {
                answers.push(custom.to_string());
            }
            return answers;
        }

        if !custom.is_empty() {
            vec![custom.to_string()]
        } else {
            self.selected_labels.iter().take(1).cloned().collect()
        }
    }

    pub fn option_selected(&self, label: &str) -> bool {
        self.selected_labels.iter().any(|item| item == label)
    }

    pub fn custom_selected(&self) -> bool {
        !self.custom_text.trim().is_empty()
    }

    pub fn is_answered(&self) -> bool {
        !self.answers().is_empty()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuestionAdvance {
    None,
    Editing,
    Advanced,
    Submit,
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
            mcp_server_name: None,
            mcp_primary_query: None,
            mcp_primary_selected_server: None,
            expert_primary_query: None,
            expert_primary_selected_agent: None,
            detail_focused: false,
            detail_scroll: 0,
            detail_scroll_max: u16::MAX,
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
        self.reset_detail_focus();
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
        self.reset_detail_focus();
    }

    pub fn insert_query_char(&mut self, ch: char) {
        self.query.push(ch);
        self.clamp_selection_to_visible();
        self.reset_detail_focus();
    }

    pub fn pop_query_char(&mut self) -> bool {
        let changed = self.query.pop().is_some();
        if changed {
            self.clamp_selection_to_visible();
            self.reset_detail_focus();
        }
        changed
    }

    pub fn reset_detail_focus(&mut self) {
        self.detail_focused = false;
        self.detail_scroll = 0;
    }

    pub fn scroll_detail_next(&mut self) {
        self.detail_scroll = self
            .detail_scroll
            .saturating_add(1)
            .min(self.detail_scroll_max);
    }

    pub fn scroll_detail_previous(&mut self) {
        self.detail_scroll = self.detail_scroll.saturating_sub(1);
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
    pub composer_tokens: Vec<ComposerToken>,
    pub timeline: Timeline,
    context: ContextPaneState,
    child_timeline: Option<ChildTranscriptState>,
    child_timeline_cache: HashMap<String, ChildTranscriptState>,
    pub active_session: bool,
    pub pending_permission: Option<PermissionView>,
    pub pending_question: Option<PendingQuestionState>,
    pub slash_panel_selected: usize,
    pub slash_panel_dismissed: bool,
    pub slash_panel_query: String,
    pub phase: AppPhase,
    pub dialog: Option<DialogState>,
    pub mcp_servers: Vec<crate::mcp::McpServerCatalogEntry>,
    pub mcp_server_tools: HashMap<String, Vec<crate::mcp::McpToolCatalogEntry>>,
    pub mcp_updating: HashSet<String>,
    pub mcp_discovery: McpDiscoveryState,
    pub mcp_discovery_error: Option<String>,
    pub skill_cards: Vec<SkillCard>,
    pub provider_label: String,
    pub model_id: String,
    pub model_label: String,
    pub fast_mode_enabled: bool,
    pub model_token_usage: Option<ModelTokenUsage>,
    /// 上下文压缩进行中：footer 指示条改用开火车式往返扫描，隐藏过期的 token 数字。
    pub compaction_active: bool,
    /// 压缩动画相对于全局 spinner 的起始帧，确保每次都从左→右填充开始。
    pub compaction_animation_start_frame: usize,
    pub reasoning_effort_label: Option<String>,
    pub permission_mode_label: String,
    pub session_id: Option<String>,
    pub current_context_branch: String,
    pub active_tool_call_id: Option<String>,
    pub latest_auto_continue: AutoContinueState,
    pub latest_todo: Option<TodoView>,
    pub retry: Option<RetryNoticeState>,
    pub transcript_view: TranscriptViewState,
    pub transcript_scroll: u16,
    pub auto_scroll: bool,
    pub transcript_scrollbar_visible: bool,
    pub child_navigation_prefix: bool,
    pub child_navigation_prefix_ticks_remaining: u8,
    pub tool_output_expanded: bool,
    pub tool_output_overrides: HashMap<String, bool>,
    pub theme_name: ThemeName,
    pub transcript_render_cache: TranscriptRenderCache,
    pub frame_hyperlink_cells: Vec<super::transcript_ratatui::HyperlinkCell>,
    last_transcript_total_rows: Option<usize>,
    pub status_spinner_frame: usize,
    pub toast: Option<ToastState>,
    pub quit_requested: bool,
    ignore_late_tool_events: bool,
    /// 当前文本选择范围（如果有）
    pub text_selection: Option<TextSelection>,
    /// 是否正在进行鼠标拖拽选择
    pub selection_in_progress: bool,
    /// The current press has emitted a drag event and must not trigger a click action.
    pub selection_dragged: bool,
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
            composer_tokens: Vec::new(),
            timeline: Timeline::default(),
            context: ContextPaneState::default(),
            child_timeline: None,
            child_timeline_cache: HashMap::new(),
            active_session: false,
            pending_permission: None,
            pending_question: None,
            slash_panel_selected: 0,
            slash_panel_dismissed: false,
            slash_panel_query: String::new(),
            phase: AppPhase::Idle,
            dialog: None,
            mcp_servers: Vec::new(),
            mcp_server_tools: HashMap::new(),
            mcp_updating: HashSet::new(),
            mcp_discovery: McpDiscoveryState::Loading,
            mcp_discovery_error: None,
            skill_cards: Vec::new(),
            provider_label: "provider".into(),
            model_id: "pending-runtime-model".into(),
            model_label: "pending runtime model".into(),
            fast_mode_enabled: false,
            model_token_usage: None,
            compaction_active: false,
            compaction_animation_start_frame: 0,
            reasoning_effort_label: None,
            permission_mode_label: "default".into(),
            session_id: None,
            current_context_branch: crate::transcript::ROOT_CONTEXT_BRANCH_ID.into(),
            active_tool_call_id: None,
            latest_auto_continue: AutoContinueState::default(),
            latest_todo: None,
            retry: None,
            transcript_view: TranscriptViewState::Parent,
            transcript_scroll: 0,
            auto_scroll: true,
            transcript_scrollbar_visible: true,
            child_navigation_prefix: false,
            child_navigation_prefix_ticks_remaining: 0,
            tool_output_expanded: false,
            tool_output_overrides: HashMap::new(),
            theme_name: ThemeName::default(),
            transcript_render_cache: TranscriptRenderCache::default(),
            frame_hyperlink_cells: Vec::new(),
            last_transcript_total_rows: None,
            status_spinner_frame: 0,
            toast: None,
            quit_requested: false,
            ignore_late_tool_events: false,
            text_selection: None,
            selection_in_progress: false,
            selection_dragged: false,
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

    pub fn set_fast_mode_enabled(&mut self, enabled: bool) {
        self.fast_mode_enabled = enabled;
    }

    pub fn set_model_context_window(&mut self, context_window_tokens: Option<u64>) {
        self.model_token_usage =
            context_window_tokens.map(|context_window_tokens| ModelTokenUsage {
                used_tokens: 0,
                context_window_tokens,
                input_tokens: 0,
                output_tokens: 0,
                cached_tokens: 0,
                cache_report: None,
            });
    }

    pub fn set_reasoning_effort_label(&mut self, label: Option<String>) {
        self.reasoning_effort_label = label;
    }

    pub fn theme(&self) -> Theme {
        Theme::for_name(self.theme_name, self.status_spinner_frame)
    }

    pub fn transcript_theme(&self) -> Theme {
        Theme::for_name(self.theme_name, 0)
    }

    pub fn set_theme_name(&mut self, theme_name: ThemeName) {
        if self.theme_name != theme_name {
            self.theme_name = theme_name;
            self.invalidate_transcript_cache();
            self.last_transcript_total_rows = None;
        }
    }

    pub fn set_tool_output_expanded(&mut self, expanded: bool) {
        if self.tool_output_expanded != expanded {
            self.tool_output_expanded = expanded;
            self.invalidate_transcript_cache();
            self.last_transcript_total_rows = None;
        }
    }

    pub fn toggle_tool_output(&mut self, call_id: &str) {
        let expanded = !self
            .tool_output_overrides
            .get(call_id)
            .copied()
            .unwrap_or(self.tool_output_expanded);
        self.tool_output_overrides
            .insert(call_id.to_string(), expanded);
        self.invalidate_transcript_cache();
        self.last_transcript_total_rows = None;
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
            && self.pending_question.is_none()
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

    pub fn active_context_open_detail(&self) -> Option<ContextOpenDetailView> {
        let context = self.active_context();
        context
            .open_detail
            .as_ref()
            .and_then(|target| project_context_open_detail(context, target))
    }

    #[cfg(test)]
    pub fn set_parent_context_for_test(&mut self, context: ContextPaneState) {
        self.context = context;
    }

    #[cfg(test)]
    pub fn set_child_context_for_test(&mut self, context: ContextPaneState) {
        if let Some(child) = self.child_timeline.as_mut() {
            child.context = context;
        }
    }

    pub fn sync_context_picker_preview(&mut self) {
        let read_only_child_view = self.is_read_only_child_view();
        let Some(dialog) = self
            .dialog
            .as_mut()
            .filter(|dialog| dialog.kind == DialogKind::ContextPicker)
        else {
            return;
        };

        if read_only_child_view {
            if let Some(child) = self.child_timeline.as_mut() {
                sync_context_picker_preview_for(dialog, &mut child.context);
            }
            return;
        }

        sync_context_picker_preview_for(dialog, &mut self.context);
    }

    pub fn update_context_picker_detail_viewport(&mut self, width: u16, height: u16) {
        let max_scroll = self.context_picker_detail_max_scroll(width, height);
        let Some(dialog) = self
            .dialog
            .as_mut()
            .filter(|dialog| dialog.kind == DialogKind::ContextPicker)
        else {
            return;
        };
        dialog.detail_scroll_max = max_scroll;
        dialog.detail_scroll = dialog.detail_scroll.min(max_scroll);
    }

    fn context_picker_detail_max_scroll(&self, width: u16, height: u16) -> u16 {
        let Some(detail) = self.active_context_open_detail() else {
            return 0;
        };
        let width = width as usize;
        if width == 0 || height == 0 {
            return 0;
        }

        let mut rows = measure::wrapped_row_count(&detail.title, width);
        if !detail.badges.is_empty() {
            rows = rows.saturating_add(measure::wrapped_row_count(
                &detail.badges.join(" · "),
                width,
            ));
        }
        if !detail.lines.is_empty() {
            rows = rows.saturating_add(1);
            rows = rows.saturating_add(
                detail
                    .lines
                    .iter()
                    .map(|line| measure::wrapped_row_count(line, width))
                    .sum::<usize>(),
            );
        }

        measure::max_scroll(rows, height)
    }

    pub fn open_context_detail(&mut self, target: Option<ContextDetailTarget>) {
        if self.is_read_only_child_view() {
            if let Some(child) = self.child_timeline.as_mut() {
                child.context.open_detail = target;
                return;
            }
        }

        self.context.open_detail = target;
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
        self.pending_question = None;
        self.ignore_late_tool_events = false;
        self.reset_slash_panel();
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

    pub fn set_skill_cards(&mut self, cards: Vec<SkillCard>) {
        self.skill_cards = cards;
    }

    pub fn set_mcp_servers(&mut self, servers: Vec<crate::mcp::McpServerCatalogEntry>) {
        self.mcp_servers = servers;
        self.mcp_discovery = McpDiscoveryState::Ready;
        self.mcp_discovery_error = None;
    }

    pub fn update_mcp_server(&mut self, server: crate::mcp::McpServerCatalogEntry) {
        if let Some(existing) = self
            .mcp_servers
            .iter_mut()
            .find(|item| item.name == server.name)
        {
            *existing = server;
        } else {
            self.mcp_servers.push(server);
        }
    }

    pub fn set_mcp_server_updating(&mut self, name: String, updating: bool) {
        if updating {
            self.mcp_updating.insert(name);
        } else {
            self.mcp_updating.remove(&name);
        }
    }

    pub fn set_mcp_server_tools(
        &mut self,
        name: String,
        tools: Vec<crate::mcp::McpToolCatalogEntry>,
    ) {
        self.mcp_server_tools.insert(name, tools);
    }

    pub fn mark_mcp_discovery_unavailable(&mut self, error: String) {
        self.mcp_discovery = McpDiscoveryState::Unavailable;
        self.mcp_discovery_error = Some(error);
    }

    pub fn set_input(&mut self, input: impl Into<String>) {
        self.input_buffer = input.into().replace(COMPOSER_ATTACHMENT_MARKER, "");
        self.input_cursor = self.input_buffer.len();
        self.composer_tokens.clear();
        self.sync_input_phase();
        self.sync_slash_panel();
    }

    pub fn set_composer_content(&mut self, content: UserMessageContent) {
        let mut parts = content.parts().into_iter().peekable();
        self.input_buffer.clear();
        self.composer_tokens.clear();
        for name in content.selected_skills {
            self.input_buffer.push(COMPOSER_ATTACHMENT_MARKER);
            self.composer_tokens.push(ComposerToken::Skill(name));
        }

        let mut image_index = 0usize;
        while let Some(part) = parts.next() {
            match part {
                UserMessagePart::Text { text }
                    if matches!(parts.peek(), Some(UserMessagePart::Image { .. }))
                        && text == format!("[Image {}]", image_index + 1) => {}
                UserMessagePart::Text { text } => self.input_buffer.push_str(&text),
                UserMessagePart::Image { attachment } => {
                    image_index += 1;
                    self.input_buffer.push(COMPOSER_ATTACHMENT_MARKER);
                    self.composer_tokens.push(ComposerToken::Image(attachment));
                }
            }
        }
        self.input_cursor = self.input_buffer.len();
        self.assert_composer_token_invariant();
        self.sync_input_phase();
        self.sync_slash_panel();
    }

    pub fn clear_input(&mut self) {
        self.input_buffer.clear();
        self.input_cursor = 0;
        self.composer_tokens.clear();
        self.sync_input_phase();
        self.sync_slash_panel();
    }

    pub fn composer_content(&self) -> UserMessageContent {
        self.assert_composer_token_invariant();
        let mut tokens = self.composer_tokens.iter();
        let mut parts = Vec::new();
        let mut selected_skills = Vec::new();
        let mut text = String::new();
        let mut image_index = 0usize;

        let flush_text = |parts: &mut Vec<UserMessagePart>, text: &mut String| {
            if !text.is_empty() {
                parts.push(UserMessagePart::Text {
                    text: std::mem::take(text),
                });
            }
        };

        for ch in self.input_buffer.chars() {
            if ch == COMPOSER_ATTACHMENT_MARKER {
                flush_text(&mut parts, &mut text);
                match tokens.next() {
                    Some(ComposerToken::Image(attachment)) => {
                        image_index += 1;
                        parts.push(UserMessagePart::Text {
                            text: format!("[Image {image_index}]"),
                        });
                        parts.push(UserMessagePart::Image {
                            attachment: attachment.clone(),
                        });
                    }
                    Some(ComposerToken::Skill(name)) => selected_skills.push(name.clone()),
                    None => unreachable!("composer token invariant violated"),
                }
            } else {
                text.push(ch);
            }
        }
        flush_text(&mut parts, &mut text);
        UserMessageContent::from_parts(parts).with_selected_skills(selected_skills)
    }

    pub fn clear_composer_tokens(&mut self) {
        self.composer_tokens.clear();
        self.input_buffer
            .retain(|ch| ch != COMPOSER_ATTACHMENT_MARKER);
        self.input_cursor = self.input_cursor.min(self.input_buffer.len());
        self.sync_input_phase();
        self.sync_slash_panel();
    }

    pub fn add_composer_attachment(&mut self, attachment: UserImageAttachment) {
        self.insert_composer_token(ComposerToken::Image(attachment));
    }

    pub fn add_composer_skill(&mut self, name: String) -> bool {
        if self
            .composer_tokens
            .iter()
            .any(|token| matches!(token, ComposerToken::Skill(existing) if existing == &name))
        {
            return false;
        }
        self.insert_composer_token(ComposerToken::Skill(name));
        true
    }

    fn insert_composer_token(&mut self, token: ComposerToken) {
        self.assert_composer_token_invariant();
        self.input_cursor = self.input_cursor.min(self.input_buffer.len());
        while self.input_cursor > 0 && !self.input_buffer.is_char_boundary(self.input_cursor) {
            self.input_cursor -= 1;
        }
        let token_index = self
            .input_buffer
            .get(..self.input_cursor)
            .map(|prefix| {
                prefix
                    .chars()
                    .filter(|ch| *ch == COMPOSER_ATTACHMENT_MARKER)
                    .count()
            })
            .unwrap_or(0);
        self.composer_tokens.insert(token_index, token);
        self.input_buffer
            .insert(self.input_cursor, COMPOSER_ATTACHMENT_MARKER);
        self.input_cursor += COMPOSER_ATTACHMENT_MARKER.len_utf8();
        self.sync_input_phase();
        self.sync_slash_panel();
    }

    pub fn remove_composer_token_at_marker(&mut self, marker_start: usize) -> bool {
        self.assert_composer_token_invariant();
        let marker_start = marker_start.min(self.input_buffer.len());
        let marker_index = self
            .input_buffer
            .get(..marker_start)
            .map(|prefix| {
                prefix
                    .chars()
                    .filter(|ch| *ch == COMPOSER_ATTACHMENT_MARKER)
                    .count()
            })
            .unwrap_or(0);
        let marker_end = marker_start.saturating_add(COMPOSER_ATTACHMENT_MARKER.len_utf8());
        if self.input_buffer.get(marker_start..marker_end) != Some(COMPOSER_ATTACHMENT_MARKER_STR) {
            return false;
        }

        self.input_buffer.drain(marker_start..marker_end);
        self.composer_tokens.get(marker_index).unwrap_or_else(|| {
            panic!("composer token marker {marker_index} has no matching token")
        });
        self.composer_tokens.remove(marker_index);
        self.input_cursor = marker_start.min(self.input_buffer.len());
        self.sync_input_phase();
        self.sync_slash_panel();
        true
    }

    pub(crate) fn assert_composer_token_invariant(&self) {
        let markers = self
            .input_buffer
            .chars()
            .filter(|ch| *ch == COMPOSER_ATTACHMENT_MARKER)
            .count();
        assert_eq!(
            markers,
            self.composer_tokens.len(),
            "composer token markers must match tokens"
        );
    }

    pub fn sync_input_phase(&mut self) {
        if self.pending_permission.is_some()
            || self.pending_question.is_some()
            || matches!(
                self.phase,
                AppPhase::Running | AppPhase::WaitingForPermission | AppPhase::Quitting
            )
        {
            return;
        }

        self.phase = if self.input_buffer.is_empty() && self.composer_tokens.is_empty() {
            AppPhase::Idle
        } else {
            AppPhase::Editing
        };
    }

    #[cfg(test)]
    pub fn transcript_scroll_offset(&self) -> u16 {
        self.transcript_scroll
    }

    pub fn slash_panel_is_open(&self) -> bool {
        !self.is_read_only_child_view()
            && self.dialog.is_none()
            && self.pending_permission.is_none()
            && self.pending_question.is_none()
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
        if self.pending_permission.is_some() || self.pending_question.is_some() {
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

    #[cfg(test)]
    pub fn transcript_is_at_bottom(&self, total_rows: usize, viewport_rows: u16) -> bool {
        measure::is_at_bottom(total_rows, viewport_rows, self.transcript_scroll)
    }

    pub fn scroll_transcript_up(&mut self, rows: u16) {
        self.transcript_scroll = self.transcript_scroll.saturating_add(rows);
        if let Some(total_rows) = self.last_transcript_total_rows {
            self.transcript_scroll = self.transcript_scroll.min(measure::max_scroll(
                total_rows,
                self.last_transcript_area.height,
            ));
        }
        self.auto_scroll = self.transcript_scroll == 0;
    }

    pub fn scroll_transcript_down(&mut self, rows: u16) {
        self.transcript_scroll = self.transcript_scroll.saturating_sub(rows);
        self.auto_scroll = self.transcript_scroll == 0;
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

        let max_scroll = measure::max_scroll(total_rows, self.last_transcript_area.height);
        self.transcript_scroll = self.transcript_scroll.min(max_scroll);
        self.auto_scroll = self.transcript_scroll == 0;
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

    pub fn set_provider_label_from_model_route(&mut self, model_id: &str) {
        if let Some((provider, _)) = model_id.split_once('/') {
            self.set_provider_label(provider);
        }
    }

    pub fn show_toast(&mut self, message: impl Into<String>, kind: ToastKind) {
        self.toast = Some(ToastState::new(message, kind, ToastState::DEFAULT_TICKS));
    }

    pub fn toast(&self) -> Option<&ToastState> {
        self.toast.as_ref()
    }

    #[cfg(test)]
    pub fn replace_session_timeline(&mut self, messages: Vec<ConversationMessage>) {
        self.timeline = Timeline::from_conversation(messages);
        self.context = ContextPaneState::default();
        self.child_timeline = None;
        self.latest_auto_continue = AutoContinueState::default();
        self.latest_todo = None;
        self.retry = None;
        self.transcript_view = TranscriptViewState::Parent;
        self.reset_after_session_timeline_replace();
    }

    #[cfg(test)]
    pub fn replace_session_timeline_from_records(&mut self, records: &[TranscriptRecord]) {
        self.try_replace_session_timeline_from_records(records)
            .expect("context projection should be valid when replacing session timeline");
    }

    #[cfg(test)]
    pub fn try_replace_session_timeline_from_records(
        &mut self,
        records: &[TranscriptRecord],
    ) -> Result<()> {
        let context = project_context_pane(records)?;
        self.active_session = true;
        self.timeline = Timeline::from_transcript_records(records);
        self.context = context;
        self.child_timeline = None;
        self.latest_auto_continue = restore_latest_auto_continue_state(records).unwrap_or_default();
        self.latest_todo = restore_latest_todo_snapshot(records).map(|items| TodoView {
            items,
            auto_continue: self.latest_auto_continue.clone(),
        });
        self.retry = None;
        self.transcript_view = TranscriptViewState::Parent;
        self.reset_after_session_timeline_replace();
        Ok(())
    }

    /// Installs a restored session only after its canonical runtime context has
    /// been projected and validated by the caller. Raw transcript projections
    /// remain available for legacy compatibility paths, but are not context
    /// authority for lifecycle transitions.
    pub fn try_replace_session_timeline_from_records_with_runtime_context(
        &mut self,
        records: &[TranscriptRecord],
        runtime_context: RuntimeActiveContext,
    ) -> Result<()> {
        validate_lifecycle_records(records, &runtime_context)?;
        let timeline = Timeline::from_transcript_records(records);
        let mut context = ContextPaneState::default();
        apply_runtime_context(
            &mut context,
            runtime_context,
            crate::tui::events::RuntimeContextDisposition::ReplaceScope,
        );
        let latest_auto_continue = restore_latest_auto_continue_state(records).unwrap_or_default();
        let latest_todo = restore_latest_todo_snapshot(records).map(|items| TodoView {
            items,
            auto_continue: latest_auto_continue.clone(),
        });

        self.active_session = true;
        self.timeline = timeline;
        self.context = context;
        self.child_timeline = None;
        self.latest_auto_continue = latest_auto_continue;
        self.latest_todo = latest_todo;
        self.retry = None;
        self.transcript_view = TranscriptViewState::Parent;
        self.reset_after_session_timeline_replace();
        Ok(())
    }

    #[cfg(test)]
    pub fn replace_child_timeline_from_records(
        &mut self,
        records: &[TranscriptRecord],
        parent_session_id: impl Into<String>,
        child_session_id: impl Into<String>,
        agent_name: impl Into<String>,
        index: usize,
        total: usize,
        pool_ordinal: u32,
    ) {
        self.try_replace_child_timeline_from_records(
            records,
            parent_session_id,
            child_session_id,
            agent_name,
            index,
            total,
            pool_ordinal,
        )
        .expect("context projection should be valid when replacing child timeline");
    }

    #[cfg(test)]
    pub fn try_replace_child_timeline_from_records(
        &mut self,
        records: &[TranscriptRecord],
        parent_session_id: impl Into<String>,
        child_session_id: impl Into<String>,
        agent_name: impl Into<String>,
        index: usize,
        total: usize,
        pool_ordinal: u32,
    ) -> Result<()> {
        let child_state = project_child_timeline_state(records)?;
        self.active_session = true;
        self.clear_input();
        self.close_dialog();
        self.reset_slash_panel();
        self.child_timeline = Some(child_state);
        self.retry = None;
        self.transcript_view = TranscriptViewState::Child {
            parent_session_id: parent_session_id.into(),
            child_session_id: child_session_id.into(),
            agent_name: agent_name.into(),
            index,
            total,
            pool_ordinal,
        };
        self.scroll_transcript_to_bottom();
        self.invalidate_transcript_cache();
        self.last_transcript_total_rows = None;
        Ok(())
    }

    pub fn try_replace_child_timeline_from_records_with_runtime_context(
        &mut self,
        records: &[TranscriptRecord],
        parent_session_id: impl Into<String>,
        child_session_id: impl Into<String>,
        agent_name: impl Into<String>,
        index: usize,
        total: usize,
        pool_ordinal: u32,
        runtime_context: RuntimeActiveContext,
    ) -> Result<()> {
        validate_lifecycle_records(records, &runtime_context)?;
        let child_session_id = child_session_id.into();
        if self.child_timeline.as_ref().is_some_and(|active| {
            active.session_id == child_session_id && active.record_count == records.len()
        }) {
            self.active_session = true;
            self.clear_input();
            self.close_dialog();
            self.reset_slash_panel();
            self.retry = None;
            self.transcript_view = TranscriptViewState::Child {
                parent_session_id: parent_session_id.into(),
                child_session_id,
                agent_name: agent_name.into(),
                index,
                total,
                pool_ordinal,
            };
            self.scroll_transcript_to_bottom();
            self.invalidate_transcript_cache();
            self.last_transcript_total_rows = None;
            return Ok(());
        }

        let child_state = match self.child_timeline_cache.remove(&child_session_id) {
            Some(cached) if cached.record_count == records.len() => cached,
            _ => {
                let mut context = ContextPaneState::default();
                apply_runtime_context(
                    &mut context,
                    runtime_context,
                    crate::tui::events::RuntimeContextDisposition::ReplaceScope,
                );
                ChildTranscriptState {
                    session_id: child_session_id.clone(),
                    timeline: Timeline::from_transcript_records(records),
                    model: child_transcript_model(records),
                    record_count: records.len(),
                    live_streaming: false,
                    context,
                }
            }
        };

        if let Some(active_child) = self.child_timeline.take() {
            self.child_timeline_cache
                .insert(active_child.session_id.clone(), active_child);
        }
        self.child_timeline = Some(child_state);
        self.active_session = true;
        self.clear_input();
        self.close_dialog();
        self.reset_slash_panel();
        self.retry = None;
        self.transcript_view = TranscriptViewState::Child {
            parent_session_id: parent_session_id.into(),
            child_session_id,
            agent_name: agent_name.into(),
            index,
            total,
            pool_ordinal,
        };
        self.scroll_transcript_to_bottom();
        self.invalidate_transcript_cache();
        self.last_transcript_total_rows = None;
        Ok(())
    }

    #[cfg(test)]
    pub fn refresh_child_timeline_from_records(&mut self, records: &[TranscriptRecord]) {
        self.try_refresh_child_timeline_from_records(records)
            .expect("context projection should be valid when refreshing child timeline");
    }

    #[cfg(test)]
    pub fn try_refresh_child_timeline_from_records(
        &mut self,
        records: &[TranscriptRecord],
    ) -> Result<()> {
        if !self.transcript_view.is_child() {
            return Ok(());
        }

        self.child_timeline = Some(project_child_timeline_state(records)?);
        self.ignore_late_tool_events = false;
        self.invalidate_transcript_cache();
        self.last_transcript_total_rows = None;
        self.reproject_pending_permission();
        Ok(())
    }

    #[cfg(test)]
    pub fn try_refresh_child_timeline_from_records_with_runtime_context(
        &mut self,
        records: &[TranscriptRecord],
        runtime_context: RuntimeActiveContext,
    ) -> Result<()> {
        if !self.transcript_view.is_child() {
            return Ok(());
        }

        let mut context = ContextPaneState::default();
        apply_runtime_context(
            &mut context,
            runtime_context,
            crate::tui::events::RuntimeContextDisposition::ReplaceScope,
        );
        let child_state = ChildTranscriptState {
            session_id: records
                .first()
                .map(|record| record.session_id.clone())
                .unwrap_or_default(),
            timeline: Timeline::from_transcript_records(records),
            model: child_transcript_model(records),
            record_count: records.len(),
            live_streaming: false,
            context,
        };
        self.child_timeline = Some(child_state);
        self.ignore_late_tool_events = false;
        self.invalidate_transcript_cache();
        self.last_transcript_total_rows = None;
        self.reproject_pending_permission();
        rebuild_active_context_picker(
            self,
            crate::tui::events::RuntimeContextDisposition::ReplaceScope,
        );
        Ok(())
    }

    #[cfg(test)]
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
            pool_ordinal,
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
            pool_ordinal: *pool_ordinal,
            model: child.model.clone(),
            record_count: child.record_count,
        })
    }

    pub fn restore_parent_timeline_view(&mut self) {
        self.retry = None;
        self.transcript_view = TranscriptViewState::Parent;
        self.close_dialog();
        self.reset_slash_panel();
        self.scroll_transcript_to_bottom();
        self.invalidate_transcript_cache();
        self.last_transcript_total_rows = None;
        self.reproject_pending_permission();
    }

    fn reset_after_session_timeline_replace(&mut self) {
        self.retry = None;
        self.pending_permission = None;
        self.pending_question = None;
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
    }

    #[cfg(test)]
    pub fn apply_child_session_event(&mut self, child_session_id: &str, event: SessionEvent) {
        self.apply_child_session_event_with_agent(child_session_id, None, None, event);
    }

    pub fn apply_child_session_event_with_agent(
        &mut self,
        child_session_id: &str,
        agent_name: Option<&str>,
        parent_tool_call_id: Option<&str>,
        event: SessionEvent,
    ) {
        match event {
            SessionEvent::Notice(notice) => {
                let kind = match notice.kind {
                    crate::tui::events::NoticeKind::Info => ToastKind::Info,
                    crate::tui::events::NoticeKind::Success => ToastKind::Success,
                    crate::tui::events::NoticeKind::RecoverableError => ToastKind::Error,
                };
                self.show_toast(notice.message, kind);
                return;
            }
            SessionEvent::ProcessIssue(issue) => {
                self.show_toast(issue.message, ToastKind::Error);
                return;
            }
            _ => {}
        }

        let viewing_child = matches!(
            &self.transcript_view,
            TranscriptViewState::Child {
                child_session_id: active_child_session_id,
                ..
            } if active_child_session_id == child_session_id
        );

        self.project_child_event_to_parent_subagent_tool(
            child_session_id,
            agent_name,
            parent_tool_call_id,
            &event,
        );

        if self.apply_child_context_event(child_session_id, &event, viewing_child) {
            return;
        }

        match event {
            SessionEvent::PermissionRequested(request) => {
                self.apply_permission_requested_projection(&request);
                if viewing_child && let Some(child_timeline) = self.child_timeline.as_mut() {
                    child_timeline.timeline.push_permission_request(request);
                    child_timeline.live_streaming = true;
                    self.invalidate_transcript_cache();
                    self.last_transcript_total_rows = None;
                }
            }
            SessionEvent::PermissionResolved(resolution) => {
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
                apply_projected_session_event(
                    EventProjection {
                        active_session: &mut self.active_session,
                        latest_auto_continue: &mut self.latest_auto_continue,
                        latest_todo: &mut self.latest_todo,
                        retry: &mut self.retry,
                        phase: &mut self.phase,
                        active_tool_call_id: &mut self.active_tool_call_id,
                        pending_permission: &mut self.pending_permission,
                        model_token_usage: &mut self.model_token_usage,
                        compaction_active: &mut self.compaction_active,
                        compaction_animation_start_frame: &mut self
                            .compaction_animation_start_frame,
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

    pub fn apply_event(&mut self, event: SessionEvent) {
        if self.apply_context_event(&event) {
            return;
        }

        if let SessionEvent::PermissionRequested(request) = event.clone() {
            self.on_permission_requested(request);
            return;
        }

        if let SessionEvent::PermissionResolved(resolution) = event.clone() {
            self.apply_permission_resolved_projection(&resolution);
        }

        let accepts_tool_events = self.accepts_tool_events();
        apply_projected_session_event(
            EventProjection {
                active_session: &mut self.active_session,
                latest_auto_continue: &mut self.latest_auto_continue,
                latest_todo: &mut self.latest_todo,
                retry: &mut self.retry,
                phase: &mut self.phase,
                active_tool_call_id: &mut self.active_tool_call_id,
                pending_permission: &mut self.pending_permission,
                model_token_usage: &mut self.model_token_usage,
                compaction_active: &mut self.compaction_active,
                compaction_animation_start_frame: &mut self.compaction_animation_start_frame,
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

    fn on_permission_requested(&mut self, request: PermissionRequestEvent) {
        self.apply_permission_requested_projection(&request);
        self.timeline.push_permission_request(request);
    }

    fn apply_permission_requested_projection(&mut self, request: &PermissionRequestEvent) {
        self.phase = AppPhase::WaitingForPermission;
        self.active_tool_call_id = Some(request.call_id.clone());
        self.pending_permission = Some(PermissionView::from_request(request.clone()));
        self.slash_panel_dismissed = false;
    }

    fn apply_context_event(&mut self, event: &SessionEvent) -> bool {
        match event {
            SessionEvent::RuntimeContextUpdated(update) => {
                if !context_update_is_accepted(&self.context, update) {
                    return true;
                }
                apply_runtime_context(
                    &mut self.context,
                    update.context.clone(),
                    update.disposition,
                );
                if !self.is_read_only_child_view() {
                    rebuild_active_context_picker(self, update.disposition);
                }
                true
            }
            SessionEvent::ContextTreeUpdated(update) => {
                self.context.tree = update.tree.clone();
                true
            }
            SessionEvent::ContextViewUpdated(update) => {
                self.context.view = update.projection.clone();
                let mut inspected_target_disappeared = false;
                if let Some(target) = self.context.open_detail.clone()
                    && !context_detail_target_exists(&self.context, &target)
                {
                    self.context.open_detail = None;
                    inspected_target_disappeared = true;
                    if matches!(
                        self.dialog.as_ref().map(|dialog| &dialog.kind),
                        Some(DialogKind::ContextDetail)
                    ) {
                        self.close_dialog();
                    }
                }
                self.sync_context_picker_preview();
                if inspected_target_disappeared {
                    self.show_toast(
                        "Context detail closed · Item no longer available",
                        ToastKind::Info,
                    );
                }
                true
            }
            SessionEvent::ContextDetailOpened(update) => {
                self.context.open_detail = update
                    .open_detail_block_id
                    .clone()
                    .map(ContextDetailTarget::Block);
                true
            }
            SessionEvent::ContextSummaryUpdated(update) => {
                self.context.view.summary_artifacts = update.summaries.clone();
                true
            }
            _ => false,
        }
    }

    fn apply_child_context_event(
        &mut self,
        child_session_id: &str,
        event: &SessionEvent,
        viewing_child: bool,
    ) -> bool {
        if !self.child_event_targets_loaded_child(child_session_id) {
            return false;
        }
        let Some(child) = self.child_timeline.as_mut() else {
            return false;
        };

        let mut detail_closed = false;

        let handled = match event {
            SessionEvent::RuntimeContextUpdated(update)
                if update.context.session_id == child_session_id =>
            {
                if context_update_is_accepted(&child.context, update) {
                    apply_runtime_context(
                        &mut child.context,
                        update.context.clone(),
                        update.disposition,
                    );
                }
                true
            }
            SessionEvent::RuntimeContextUpdated(_) => false,
            SessionEvent::ContextTreeUpdated(update) => {
                child.context.tree = update.tree.clone();
                true
            }
            SessionEvent::ContextViewUpdated(update) => {
                child.context.view = update.projection.clone();
                if let Some(target) = child.context.open_detail.clone()
                    && !context_detail_target_exists(&child.context, &target)
                {
                    child.context.open_detail = None;
                    detail_closed = true;
                }
                true
            }
            SessionEvent::ContextDetailOpened(update) => {
                child.context.open_detail = update
                    .open_detail_block_id
                    .clone()
                    .map(ContextDetailTarget::Block);
                true
            }
            SessionEvent::ContextSummaryUpdated(update) => {
                child.context.view.summary_artifacts = update.summaries.clone();
                true
            }
            _ => false,
        };

        if handled {
            if viewing_child {
                rebuild_active_context_picker(
                    self,
                    match event {
                        SessionEvent::RuntimeContextUpdated(update) => update.disposition,
                        _ => crate::tui::events::RuntimeContextDisposition::Advance,
                    },
                );
            }
            if detail_closed {
                if matches!(
                    self.dialog.as_ref().map(|dialog| &dialog.kind),
                    Some(DialogKind::ContextDetail)
                ) {
                    self.close_dialog();
                }
                if viewing_child {
                    self.show_toast(
                        "Context detail closed · Item no longer available",
                        ToastKind::Info,
                    );
                }
            }
        }
        handled && viewing_child
    }

    fn child_event_targets_loaded_child(&self, child_session_id: &str) -> bool {
        matches!(
            &self.transcript_view,
            TranscriptViewState::Child {
                child_session_id: active_child_session_id,
                ..
            } if active_child_session_id == child_session_id
        ) || self
            .child_timeline
            .as_ref()
            .is_some_and(|child| child.session_id == child_session_id)
    }

    fn apply_permission_resolved_projection(&mut self, resolution: &PermissionResolutionEvent) {
        self.active_tool_call_id = None;
        if self.pending_permission.as_ref().map(|p| p.call_id.as_str())
            == Some(resolution.call_id.as_str())
        {
            self.pending_permission = None;
        }
        self.phase = AppPhase::Running;
    }

    fn project_child_event_to_parent_subagent_tool(
        &mut self,
        child_session_id: &str,
        agent_name: Option<&str>,
        parent_tool_call_id: Option<&str>,
        event: &SessionEvent,
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
            agent_name,
            parent_tool_call_id,
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
    pub fn transcript_click_target(
        &self,
        terminal_col: u16,
        terminal_row: u16,
    ) -> Option<TranscriptClickTarget> {
        let area = self.last_transcript_area;
        if terminal_col < area.left()
            || terminal_col >= area.right()
            || terminal_row < area.top()
            || terminal_row >= area.bottom()
            || area.width == 0
            || area.height == 0
        {
            return None;
        }

        let absolute_row =
            (terminal_row - area.y) as usize + self.last_transcript_scroll_top as usize;
        let cache = &self.transcript_render_cache;
        let item_index = match cache.row_starts().binary_search(&absolute_row) {
            Ok(index) => index,
            Err(index) => index.saturating_sub(1),
        };
        let entry = cache.entries().get(item_index)?;
        let rendered_line_offset =
            absolute_row.saturating_sub(*cache.row_starts().get(item_index)?);
        let line = entry.document.lines.get(rendered_line_offset)?;
        let local_col = terminal_col - area.x;
        let mut visual_col = 0u16;
        for span in &line.spans {
            let span_width = crate::tui::measure::display_width(&span.text) as u16;
            if local_col >= visual_col && local_col < visual_col.saturating_add(span_width) {
                if matches!(span.interaction, Some(Interaction::OpenUrl(_))) {
                    return None;
                }
                break;
            }
            visual_col = visual_col.saturating_add(span_width);
        }

        match self.active_timeline().items().get(item_index) {
            Some(TimelineItem::Tool(tool)) => {
                Some(TranscriptClickTarget::ToolCard(tool.call_id.clone()))
            }
            _ => None,
        }
    }

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
        let entry = &cache.entries()[item_index];
        let rendered_line_offset = absolute_row.saturating_sub(item_start_row);
        let Some(line) = entry.document.lines.get(rendered_line_offset) else {
            return None;
        };

        // Anchors retain visual line/character semantics. Hit testing walks the
        // document's display-cell spans, ignoring chrome until a source-backed span
        // is hit; this handles CJK/wide characters without treating border/padding
        // as selectable text. Anchors are the leading boundary of a grapheme so a
        // forward or reverse release can expand to include both endpoint graphemes.
        let local_col = terminal_col - area.x;
        let mut visual_col = 0u16;
        let mut visual_char = 0usize;
        for span in &line.spans {
            let span_width = crate::tui::measure::display_width(&span.text) as u16;
            let span_chars = span.text.chars().count();
            if local_col >= visual_col && local_col < visual_col.saturating_add(span_width) {
                let range = span.source?;
                let within =
                    column_to_char_offset(&span.text, local_col - visual_col).min(span_chars);
                let _source = entry.document.source_blocks.get(range.block_index)?;
                return Some(SelectionAnchor {
                    item_index,
                    rendered_line_offset,
                    char_offset: visual_char + within,
                });
            }
            visual_col = visual_col.saturating_add(span_width);
            visual_char += span_chars;
        }
        None
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

/// Lifecycle events carry canonical context separately from presentation records.
/// Validate their shared session identity before replacing visible state so malformed
/// timeline input cannot partially install a new lifecycle scope.
fn validate_lifecycle_records(
    records: &[TranscriptRecord],
    runtime_context: &RuntimeActiveContext,
) -> Result<()> {
    if let Some(record) = records
        .iter()
        .find(|record| record.session_id != runtime_context.session_id)
    {
        anyhow::bail!(
            "lifecycle record session '{}' does not match runtime context session '{}'",
            record.session_id,
            runtime_context.session_id
        );
    }
    Ok(())
}

/// Convert a display-cell coordinate into the leading Unicode-scalar offset of
/// the hit extended grapheme cluster. Selection extraction expands endpoint
/// boundaries, so both forward and reverse gestures include the hit graphemes.
fn column_to_char_offset(text: &str, target_col: u16) -> usize {
    use unicode_segmentation::UnicodeSegmentation;

    let mut width = 0usize;
    let mut offset = 0usize;
    for grapheme in text.graphemes(true) {
        let grapheme_width = crate::tui::measure::display_width(grapheme);
        if width + grapheme_width > target_col as usize {
            return offset;
        }
        width += grapheme_width;
        offset += grapheme.chars().count();
    }
    offset
}

mod event_projection;
use event_projection::*;
pub(crate) use event_projection::{context_detail_target_exists, context_dialog_items};

impl From<TokenUsageEvent> for ModelTokenUsage {
    fn from(event: TokenUsageEvent) -> Self {
        Self {
            used_tokens: event.used_tokens,
            context_window_tokens: event.context_window_tokens,
            input_tokens: event.input_tokens,
            output_tokens: event.output_tokens,
            cached_tokens: event.cached_tokens,
            cache_report: event.cache_report,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::{AutoContinueState, TodoItem, TodoStatus};
    use crate::tool::{QuestionOption, QuestionRequest, QuestionSpec};
    use crate::transcript::{TranscriptEvent, TranscriptRecord};
    use crate::tui::events::{
        AutoContinueChangedEvent, ContextTreeUpdatedEvent, ContextViewUpdatedEvent, ErrorEvent,
        NoticeEvent, NoticeKind, PermissionResolutionEvent, ProcessIssueEvent, RetryLifecycleEvent,
        SessionEvent, TodoSnapshotEvent, ToolCancelledEvent, ToolPendingEvent,
    };

    fn question_state(questions: Vec<QuestionSpec>) -> PendingQuestionState {
        PendingQuestionState::new(QuestionRequest { questions }, None)
    }

    fn runtime_context(
        session_id: &str,
        branch_id: &str,
        revision: u64,
        leaf: u64,
    ) -> RuntimeActiveContext {
        let snapshot = crate::transcript::transcript_projection::project_runtime_restore_snapshot(
            session_id.into(),
            Vec::new(),
            crate::transcript::transcript_projection::SessionContextCursor {
                branch_id: Some(crate::transcript::ROOT_CONTEXT_BRANCH_ID.into()),
                leaf_sequence: None,
            },
            &[],
        )
        .unwrap()
        .snapshot;
        let mut context = RuntimeActiveContext::try_from(&snapshot).unwrap();
        context.active_context.branch_id = branch_id.into();
        context.context_scope_revision = revision;
        context.leaf_sequence = leaf;
        context
    }

    #[test]
    fn retry_projection_clears_when_the_attempt_starts_or_turn_ends() {
        let mut state = TuiState::default();
        let retry = RetryLifecycleEvent {
            attempt: 2,
            max_attempts: 3,
            delay_secs: 1,
            error: "temporary upstream failure".into(),
        };

        state.apply_event(SessionEvent::RetryScheduled(retry.clone()));
        let notice = state.retry.as_ref().expect("retry notice");
        assert_eq!(notice.attempt, 2);
        assert_eq!(notice.secs_remaining, 1);
        assert_eq!(state.phase, AppPhase::Running);
        let toast = state.toast().expect("sticky retry toast");
        assert_eq!(toast.message, "Retrying in 1s · attempt 2 of 3");
        assert_eq!(toast.kind, ToastKind::Error);

        state.apply_event(SessionEvent::RetryStarted(retry));
        assert_eq!(state.retry, None);
        assert!(state.toast().is_none());

        state.apply_event(SessionEvent::RetryScheduled(RetryLifecycleEvent {
            attempt: 3,
            max_attempts: 3,
            delay_secs: 2,
            error: "temporary upstream failure".into(),
        }));
        state.apply_event(SessionEvent::Error(ErrorEvent::new("request failed")));
        assert_eq!(state.retry, None);
    }

    #[test]
    fn retry_toast_counts_down_and_stays_visible_across_ticks() {
        let mut state = TuiState::default();
        state.apply_event(SessionEvent::RetryScheduled(RetryLifecycleEvent {
            attempt: 1,
            max_attempts: 3,
            delay_secs: 2,
            error: "temporary upstream failure".into(),
        }));

        for _ in 0..RetryNoticeState::TICKS_PER_SECOND {
            state.apply_event(SessionEvent::Tick);
        }
        let toast = state.toast().expect("countdown toast remains");
        assert_eq!(toast.kind, ToastKind::Error);
        assert_eq!(toast.message, "Retrying in 1s · attempt 1 of 3");
        assert_eq!(
            state.retry.as_ref().map(|retry| retry.secs_remaining),
            Some(1)
        );

        for _ in 0..RetryNoticeState::TICKS_PER_SECOND {
            state.apply_event(SessionEvent::Tick);
        }
        assert_eq!(
            state.toast().map(|toast| toast.message.as_str()),
            Some("Retrying in 0s · attempt 1 of 3")
        );
        assert!(state.retry.is_some());
    }

    #[test]
    fn retry_projection_clears_when_replacing_session_or_transcript_view() {
        let retry = RetryNoticeState::from_lifecycle(RetryLifecycleEvent {
            attempt: 2,
            max_attempts: 3,
            delay_secs: 1,
            error: "temporary upstream failure".into(),
        });
        let mut state = TuiState::default();

        state.retry = Some(retry.clone());
        state.replace_session_timeline(Vec::new());
        assert_eq!(state.retry, None);

        state.retry = Some(retry.clone());
        state
            .try_replace_child_timeline_from_records(&[], "parent", "child", "explorer", 0, 1, 1)
            .expect("child view installs");
        assert_eq!(state.retry, None);

        state.retry = Some(retry);
        state.restore_parent_timeline_view();
        assert_eq!(state.retry, None);
    }

    #[test]
    fn group_16_runtime_context_update_keeps_canonical_picker_ids_and_scope_rules() {
        let snapshot = crate::runtime_context::group_16_runtime_snapshot();
        let canonical = RuntimeActiveContext::try_from(&snapshot).expect("canonical context");
        let mut state = TuiState::default();
        state.open_dialog(DialogState::new(
            DialogKind::ContextPicker,
            "Context",
            None,
            Vec::new(),
        ));
        state.apply_event(SessionEvent::RuntimeContextUpdated(
            crate::tui::events::RuntimeContextUpdatedEvent {
                context: canonical.clone(),
                disposition: crate::tui::events::RuntimeContextDisposition::Advance,
            },
        ));

        let ids = state
            .dialog()
            .expect("context picker")
            .items
            .iter()
            .map(|item| item.id.as_str())
            .collect::<Vec<_>>();
        assert!(ids.contains(&"block:active-block"));
        assert!(ids.contains(&"block:pinned-block"));
        assert!(!ids.contains(&"block:removed-block"));
        assert!(!ids.contains(&"block:retired-raw-block"));
        let selected = state
            .dialog_mut()
            .expect("context picker")
            .items
            .iter()
            .position(|item| item.id == "block:pinned-block")
            .expect("pinned item");
        state.dialog_mut().expect("context picker").selected = selected;
        state.dialog_mut().expect("context picker").query = "pinned".into();
        state.dialog_mut().expect("context picker").detail_focused = true;
        state.sync_context_picker_preview();
        assert_eq!(
            state.active_context().open_detail,
            Some(ContextDetailTarget::Block("pinned-block".into()))
        );

        let mut advanced = canonical.clone();
        advanced.leaf_sequence += 1;
        state.apply_event(SessionEvent::RuntimeContextUpdated(
            crate::tui::events::RuntimeContextUpdatedEvent {
                context: advanced,
                disposition: crate::tui::events::RuntimeContextDisposition::Advance,
            },
        ));
        assert_eq!(
            state
                .dialog()
                .and_then(|dialog| dialog.selected_item())
                .map(|item| item.id.as_str()),
            Some("block:pinned-block")
        );
        assert_eq!(
            state.active_context().open_detail,
            Some(ContextDetailTarget::Block("pinned-block".into()))
        );

        let mut replacement = canonical;
        replacement.context_scope_revision += 1;
        replacement.leaf_sequence += 2;
        replacement.active_context.open_detail_block_id = None;
        state.apply_event(SessionEvent::RuntimeContextUpdated(
            crate::tui::events::RuntimeContextUpdatedEvent {
                context: replacement,
                disposition: crate::tui::events::RuntimeContextDisposition::ReplaceScope,
            },
        ));
        assert_eq!(
            state.active_context().open_detail,
            Some(ContextDetailTarget::Block("active-block".into()))
        );
        assert_eq!(state.dialog().map(|dialog| dialog.query.as_str()), Some(""));
        assert!(!state.dialog().is_some_and(|dialog| dialog.detail_focused));
        assert_eq!(
            state
                .dialog()
                .and_then(|dialog| dialog.selected_item())
                .map(|item| item.id.as_str()),
            Some("block:active-block")
        );
    }

    fn runtime_update(
        session_id: &str,
        branch_id: &str,
        revision: u64,
        leaf: u64,
        disposition: crate::tui::events::RuntimeContextDisposition,
    ) -> crate::tui::events::RuntimeContextUpdatedEvent {
        crate::tui::events::RuntimeContextUpdatedEvent {
            context: runtime_context(session_id, branch_id, revision, leaf),
            disposition,
        }
    }

    #[test]
    fn replace_scope_orders_same_session_by_revision() {
        let mut pane = ContextPaneState::default();
        pane.runtime_context = Some(runtime_context("session", "new", 4, 10));

        assert!(context_update_is_accepted(
            &pane,
            &runtime_update(
                "session",
                "other",
                5,
                1,
                crate::tui::events::RuntimeContextDisposition::ReplaceScope,
            ),
        ));
        assert!(!context_update_is_accepted(
            &pane,
            &runtime_update(
                "session",
                "other",
                4,
                99,
                crate::tui::events::RuntimeContextDisposition::ReplaceScope,
            ),
        ));
        assert!(!context_update_is_accepted(
            &pane,
            &runtime_update(
                "session",
                "other",
                3,
                99,
                crate::tui::events::RuntimeContextDisposition::ReplaceScope,
            ),
        ));
        assert!(context_update_is_accepted(
            &pane,
            &runtime_update(
                "other-session",
                "root",
                0,
                0,
                crate::tui::events::RuntimeContextDisposition::ReplaceScope,
            ),
        ));
    }

    #[test]
    fn advance_rejects_delayed_higher_leaf_from_old_scope() {
        let mut pane = ContextPaneState::default();
        pane.runtime_context = Some(runtime_context("session", "branch", 6, 10));

        assert!(!context_update_is_accepted(
            &pane,
            &runtime_update(
                "session",
                "branch",
                5,
                100,
                crate::tui::events::RuntimeContextDisposition::Advance,
            ),
        ));
    }

    fn option(label: &str) -> QuestionOption {
        QuestionOption {
            label: label.into(),
            description: format!("{label} option"),
        }
    }

    #[test]
    fn single_question_does_not_show_confirm_tab_and_submits_immediately() {
        let mut state = question_state(vec![QuestionSpec {
            question: "Choose one".into(),
            header: "Mode".into(),
            options: vec![option("Fast"), option("Safe")],
            multiple: false,
        }]);

        assert!(!state.show_confirm_tab());
        assert_eq!(state.total_tabs(), 1);
        assert_eq!(state.pick_option(1), QuestionAdvance::Submit);
        assert_eq!(state.questions[0].answers(), vec!["Safe".to_string()]);
    }

    #[test]
    fn single_select_clears_custom_text_and_advances() {
        let mut state = question_state(vec![
            QuestionSpec {
                question: "Choose one".into(),
                header: "Mode".into(),
                options: vec![option("Fast"), option("Safe")],
                multiple: false,
            },
            QuestionSpec {
                question: "Choose tone".into(),
                header: "Tone".into(),
                options: vec![option("Warm")],
                multiple: false,
            },
        ]);
        state.questions[0].custom_text = "Custom".into();
        state.questions[0].custom_cursor = 6;

        assert_eq!(state.pick_option(0), QuestionAdvance::Advanced);
        assert_eq!(state.questions[0].answers(), vec!["Fast".to_string()]);
        assert_eq!(state.questions[0].custom_text, "");
        assert_eq!(state.questions[0].custom_cursor, 0);
        assert_eq!(state.active_tab, 1);
    }

    #[test]
    fn multi_select_toggles_without_advancing_and_custom_coexists() {
        let mut state = question_state(vec![QuestionSpec {
            question: "Choose several".into(),
            header: "Features".into(),
            options: vec![option("Alpha"), option("Beta")],
            multiple: true,
        }]);

        assert!(state.show_confirm_tab());
        assert_eq!(state.total_tabs(), 2);
        assert_eq!(state.pick_option(0), QuestionAdvance::None);
        assert_eq!(state.pick_option(1), QuestionAdvance::None);
        state.questions[0].custom_text = "Gamma".into();

        assert_eq!(
            state.questions[0].answers(),
            vec!["Alpha".to_string(), "Beta".to_string(), "Gamma".to_string()]
        );

        assert_eq!(state.pick_option(0), QuestionAdvance::None);
        assert_eq!(
            state.questions[0].answers(),
            vec!["Beta".to_string(), "Gamma".to_string()]
        );
    }

    #[test]
    fn multi_select_custom_row_enters_edit_without_discarding_existing_custom() {
        let mut state = question_state(vec![QuestionSpec {
            question: "Choose several".into(),
            header: "Features".into(),
            options: vec![option("Alpha")],
            multiple: true,
        }]);
        state.active_row = 1;

        assert_eq!(state.activate_custom_row(), QuestionAdvance::Editing);
        assert!(state.editing_custom);
        state.questions[0].custom_edit_text = "Gamma".into();
        assert_eq!(state.commit_custom_answer(), QuestionAdvance::None);
        assert_eq!(state.questions[0].answers(), vec!["Gamma".to_string()]);

        assert_eq!(state.activate_custom_row(), QuestionAdvance::Editing);
        assert_eq!(state.questions[0].answers(), vec!["Gamma".to_string()]);
        assert_eq!(state.questions[0].custom_edit_text, "Gamma");
    }

    #[test]
    fn multi_question_custom_answer_advances_to_the_next_question() {
        let mut state = question_state(vec![
            QuestionSpec {
                question: "Choose several".into(),
                header: "Features".into(),
                options: vec![option("Alpha")],
                multiple: true,
            },
            QuestionSpec {
                question: "Choose a mode".into(),
                header: "Mode".into(),
                options: vec![option("Fast")],
                multiple: false,
            },
        ]);
        state.active_row = 1;
        state.begin_custom_edit();
        state.questions[0].custom_edit_text = "Gamma".into();

        assert_eq!(state.commit_custom_answer(), QuestionAdvance::Advanced);
        assert_eq!(state.active_tab, 1);
        assert_eq!(state.questions[0].answers(), vec!["Gamma".to_string()]);
    }

    #[test]
    fn custom_single_answer_clears_options_and_advances() {
        let mut state = question_state(vec![
            QuestionSpec {
                question: "Choose one".into(),
                header: "Mode".into(),
                options: vec![option("Fast"), option("Safe")],
                multiple: false,
            },
            QuestionSpec {
                question: "Choose tone".into(),
                header: "Tone".into(),
                options: vec![option("Warm")],
                multiple: false,
            },
        ]);
        state.questions[0].selected_labels.push("Fast".into());
        state.active_row = 2;
        assert_eq!(state.activate_custom_row(), QuestionAdvance::Editing);
        state.questions[0].custom_edit_text = "Tailored".into();
        state.questions[0].custom_edit_cursor = 8;

        assert_eq!(state.commit_custom_answer(), QuestionAdvance::Advanced);
        assert!(state.questions[0].selected_labels.is_empty());
        assert_eq!(state.questions[0].answers(), vec!["Tailored".to_string()]);
        assert_eq!(state.active_tab, 1);
    }

    #[test]
    fn unanswered_confirm_focuses_first_missing_question() {
        let mut state = question_state(vec![
            QuestionSpec {
                question: "First".into(),
                header: "One".into(),
                options: vec![option("A")],
                multiple: false,
            },
            QuestionSpec {
                question: "Second".into(),
                header: "Two".into(),
                options: vec![option("B")],
                multiple: false,
            },
        ]);
        state.pick_option(0);
        state.focus_tab(2);

        assert_eq!(state.first_unanswered_tab(), Some(1));
        assert!(state.is_confirm_tab());
    }

    #[test]
    fn tool_pending_updates_state_and_footer() {
        let mut state = TuiState::default();

        state.apply_event(SessionEvent::ToolPending(ToolPendingEvent::new(
            "call-pending",
            "edit__apply_patch",
        )));

        assert_eq!(state.phase, AppPhase::Running);
        assert_eq!(state.active_tool_call_id.as_deref(), Some("call-pending"));

        assert!(matches!(
            state.timeline.items().last(),
            Some(crate::tui::timeline::TimelineItem::Tool(tool))
                if tool.call_id == "call-pending"
                    && tool.status == crate::tui::timeline::ToolExecutionStatus::Pending
        ));
    }

    #[test]
    fn process_issue_keeps_running_phase_and_uses_error_toast() {
        let mut state = TuiState::default();
        state.phase = AppPhase::Running;

        state.apply_event(SessionEvent::ProcessIssue(ProcessIssueEvent {
            message: "Model stream interrupted".into(),
            detail: Some("Partial assistant output was preserved".into()),
            action: Some("Continuing with a fresh model iteration".into()),
        }));

        assert_eq!(state.phase, AppPhase::Running);
        assert!(matches!(state.toast(), Some(toast)
            if toast.message == "Model stream interrupted" && toast.kind == ToastKind::Error));
        assert!(state.timeline.items().is_empty());
    }

    #[test]
    fn tool_cancelled_updates_pending_tool_and_footer() {
        let mut state = TuiState::default();
        state.apply_event(SessionEvent::ToolPending(ToolPendingEvent::new(
            "call-pending",
            "edit__apply_patch",
        )));

        state.apply_event(SessionEvent::ToolCancelled(ToolCancelledEvent::new(
            "call-pending",
            "edit__apply_patch",
        )));

        assert_eq!(state.active_tool_call_id, None);

        assert!(matches!(
            state.timeline.items().last(),
            Some(crate::tui::timeline::TimelineItem::Tool(tool))
                if tool.call_id == "call-pending"
                    && tool.status == crate::tui::timeline::ToolExecutionStatus::Cancelled
        ));
    }

    #[test]
    fn markdown_link_click_does_not_toggle_its_tool_card() {
        let mut state = TuiState::default();
        state.last_transcript_area = ratatui::layout::Rect::new(0, 0, 8, 1);
        let mut document = crate::tui::transcript_render::Document::default();
        let block = document.add_source("docs");
        document.push_line(
            crate::tui::transcript_render::Line {
                spans: vec![
                    crate::tui::transcript_render::Span::source_with_interaction(
                        "docs",
                        ratatui::style::Style::default(),
                        crate::tui::transcript_render::SourceRange::new(block, 0, 4),
                        crate::tui::transcript_render::CopyJoin::Concat,
                        Some(crate::tui::transcript_render::Interaction::OpenUrl(
                            "https://example.test".into(),
                        )),
                    ),
                ],
            },
            crate::tui::transcript_render::Break::End,
        );
        state.transcript_render_cache.set_entries_for_test(vec![
            crate::tui::components::transcript::TranscriptRenderCacheEntry {
                revision: None,
                document,
            },
        ]);
        state
            .transcript_render_cache
            .set_row_metadata_for_test(vec![0], vec![1]);
        state
            .timeline
            .push_tool_started(crate::tui::events::ToolStartedEvent::new(
                "call-1",
                "shell__exec",
                "run",
            ));

        assert_eq!(state.transcript_click_target(0, 0), None);
    }

    #[test]
    fn per_tool_output_override_uses_global_preference_as_its_default() {
        let mut state = TuiState::default();
        state.tool_output_expanded = true;
        state.toggle_tool_output("call-1");
        assert_eq!(state.tool_output_overrides.get("call-1"), Some(&false));
        state.toggle_tool_output("call-1");
        assert_eq!(state.tool_output_overrides.get("call-1"), Some(&true));
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
    fn compaction_started_clears_stale_usage_and_marks_active() {
        let mut state = TuiState::default();
        state.set_token_usage(ModelTokenUsage {
            used_tokens: 1_000,
            context_window_tokens: 10_000,
            input_tokens: 900,
            output_tokens: 100,
            cached_tokens: 400,
            cache_report: None,
        });

        state.status_spinner_frame = 77;
        state.apply_event(SessionEvent::CompactionStarted);

        assert!(state.compaction_active);
        assert_eq!(state.compaction_animation_start_frame, 77);
        assert_eq!(state.model_token_usage, None);

        state.apply_event(SessionEvent::CompactionCommitted {
            summary: Some("compacted".into()),
        });
        assert!(!state.compaction_active);
    }

    #[test]
    fn compaction_failure_and_interrupt_clear_active_flag() {
        let mut state = TuiState::default();
        state.apply_event(SessionEvent::CompactionStarted);
        assert!(state.compaction_active);

        state.apply_event(SessionEvent::CompactionFailed);
        assert!(!state.compaction_active);

        state.apply_event(SessionEvent::CompactionStarted);
        state.apply_event(SessionEvent::Interrupted);
        assert!(!state.compaction_active);
    }

    #[test]
    fn permission_resolved_clears_active_tool_and_pending_permission() {
        let mut state = TuiState::default();
        let request = PermissionRequestEvent::new("call-1", "shell__exec", "run ls");

        state
            .set_pending_permission_projection(Some(PermissionView::from_request(request.clone())));
        state.apply_event(SessionEvent::PermissionRequested(request));
        assert_eq!(state.phase, AppPhase::WaitingForPermission);
        assert_eq!(state.active_tool_call_id.as_deref(), Some("call-1"));
        assert!(state.pending_permission.is_some());

        state.apply_event(SessionEvent::PermissionResolved(
            PermissionResolutionEvent::approved("call-1"),
        ));

        assert_eq!(state.phase, AppPhase::Running);
        assert_eq!(state.active_tool_call_id, None);
        assert!(state.pending_permission.is_none());

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
        state.apply_event(SessionEvent::ToolPending(ToolPendingEvent::new(
            "call-1",
            "shell__exec",
        )));
        state.set_pending_permission_projection(Some(PermissionView::from_request(
            PermissionRequestEvent::new("call-1", "shell__exec", "run ls"),
        )));
        state.apply_event(SessionEvent::PermissionRequested(
            PermissionRequestEvent::new("call-1", "shell__exec", "run ls"),
        ));

        state.apply_event(SessionEvent::Interrupted);

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
        state.apply_event(SessionEvent::ToolPending(ToolPendingEvent::new(
            "call-1",
            "shell__exec",
        )));
        state.apply_event(SessionEvent::Interrupted);

        state.apply_event(SessionEvent::ToolFinished(
            crate::tui::events::ToolFinishedEvent::new(
                "late-call",
                "fs__write",
                "fs__write completed",
                ToolOutcome::Success,
            ),
        ));

        assert_eq!(state.phase, AppPhase::Completed);
        assert_eq!(state.active_tool_call_id, None);

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
        state.apply_event(SessionEvent::ToolPending(ToolPendingEvent::new(
            "call-1",
            "shell__exec",
        )));
        state.apply_event(SessionEvent::Interrupted);

        state.apply_event(SessionEvent::ToolCancelled(ToolCancelledEvent::new(
            "late-call",
            "fs__write",
        )));

        assert_eq!(state.phase, AppPhase::Completed);
        assert_eq!(state.active_tool_call_id, None);

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
            1,
        );
        state.apply_child_session_event(
            "child-session",
            SessionEvent::ToolPending(ToolPendingEvent::new("call-1", "shell__exec")),
        );
        state.apply_child_session_event("child-session", SessionEvent::Interrupted);

        state.apply_child_session_event(
            "child-session",
            SessionEvent::ToolFinished(crate::tui::events::ToolFinishedEvent::new(
                "late-call",
                "fs__write",
                "fs__write completed",
                ToolOutcome::Success,
            )),
        );

        assert_eq!(state.phase, AppPhase::Completed);
        assert_eq!(state.active_tool_call_id, None);

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

        state.apply_event(SessionEvent::UserMessage(UserMessageEvent::new("hello")));

        assert_eq!(state.transcript_scroll_offset(), 4);
        assert!(!state.auto_scroll);
        assert_eq!(state.timeline.items().len(), 1);
    }

    #[test]
    fn toast_replaces_previous_message_and_resets_lifetime() {
        let mut state = TuiState::default();
        state.show_toast("Copied to clipboard", ToastKind::Success);
        state.apply_event(SessionEvent::Tick);
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

        state.apply_event(SessionEvent::Tick);
        assert!(state.toast().is_some());

        state.apply_event(SessionEvent::Tick);
        assert!(state.toast().is_none());
    }

    #[test]
    fn scroll_up_while_transcript_fits_viewport_keeps_auto_scroll() {
        let mut state = TuiState::default();
        state.last_transcript_area.height = 10;
        state.sync_transcript_viewport_rows(3);

        state.scroll_transcript_up(1);
        state.sync_transcript_viewport_rows(12);

        assert_eq!(state.transcript_scroll_offset(), 0);
        assert!(state.auto_scroll);
    }

    #[test]
    fn sync_transcript_viewport_rows_clamps_invalid_offset_and_restores_auto_scroll() {
        let mut state = TuiState::default();
        state.last_transcript_area.height = 5;
        state.sync_transcript_viewport_rows(10);
        state.scroll_transcript_up(4);
        assert_eq!(state.transcript_scroll_offset(), 4);
        assert!(!state.auto_scroll);

        state.last_transcript_area.height = 10;
        state.sync_transcript_viewport_rows(10);

        assert_eq!(state.transcript_scroll_offset(), 0);
        assert!(state.auto_scroll);
    }

    #[test]
    fn manual_transcript_viewport_tracks_new_rows_without_top_shift() {
        let mut state = TuiState::default();
        state.last_transcript_area.height = 5;
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
            1,
        );

        assert!(!state.slash_panel_is_open());
    }

    #[test]
    fn todo_events_update_latest_state_and_timeline() {
        let mut state = TuiState::default();

        state.apply_event(SessionEvent::AutoContinueChanged(
            AutoContinueChangedEvent::new(AutoContinueState { enabled: true }),
        ));
        assert!(state.latest_todo.is_none());
        assert_eq!(state.timeline.items().len(), 0);

        state.apply_event(SessionEvent::TodoSnapshot(TodoSnapshotEvent::new(vec![
            TodoItem {
                id: "t1".into(),
                content: "inspect".into(),
                status: TodoStatus::Pending,
            },
        ])));

        let todo = state.latest_todo.as_ref().expect("todo state exists");
        assert!(todo.auto_continue.enabled);
        assert_eq!(todo.items.len(), 1);
        assert!(matches!(
            state.timeline.items().last(),
            Some(crate::tui::timeline::TimelineItem::Todo(todo)) if todo.items.len() == 1
        ));

        state.apply_event(SessionEvent::UserMessage(UserMessageEvent::new(
            "next turn",
        )));
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
            1,
        );

        assert!(matches!(
            state.transcript_view,
            TranscriptViewState::Child {
                ref parent_session_id,
                ref child_session_id,
                ref agent_name,
                index: 2,
                total: 3,
                pool_ordinal: 1,
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

        state.apply_event(SessionEvent::Error(crate::tui::events::ErrorEvent::new(
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
            1,
        );

        assert_eq!(
            state
                .active_context()
                .tree
                .active_node_id()
                .map(|node_id| node_id.as_str()),
            Some("child-node")
        );
    }

    #[test]
    fn child_context_update_while_viewing_parent_updates_cached_child_context() {
        let child_records = vec![TranscriptRecord {
            session_id: "child-session".into(),
            sequence: 1,
            timestamp_ms: 0,
            context_branch_id: None,
            event: TranscriptEvent::AssistantMessage {
                content: "child response".into(),
            },
        }];
        let child_context_records = vec![
            TranscriptRecord {
                session_id: "child-session".into(),
                sequence: 2,
                timestamp_ms: 1,
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
                sequence: 3,
                timestamp_ms: 2,
                context_branch_id: None,
                event: TranscriptEvent::ContextNodeLifecycle {
                    node_id: "root".into(),
                    status: ContextNodeStatus::Inactive,
                },
            },
            TranscriptRecord {
                session_id: "child-session".into(),
                sequence: 4,
                timestamp_ms: 3,
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
            1,
        );
        state.restore_parent_timeline_view();
        assert_eq!(state.transcript_view, TranscriptViewState::Parent);

        let child_context = project_context_pane(&child_context_records).unwrap();
        state.apply_child_session_event(
            "child-session",
            SessionEvent::ContextTreeUpdated(ContextTreeUpdatedEvent {
                tree: child_context.tree,
            }),
        );

        let child_timeline = state
            .child_timeline
            .as_ref()
            .expect("child timeline cached");
        assert_eq!(
            child_timeline
                .context
                .tree
                .active_node_id()
                .map(|node_id| node_id.as_str()),
            Some("child-node")
        );
    }

    #[test]
    fn parent_context_update_while_viewing_child_updates_parent_context() {
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
            1,
        );
        assert_eq!(
            state
                .active_context()
                .tree
                .active_node_id()
                .map(|node_id| node_id.as_str()),
            Some("child-node")
        );

        let parent_context = project_context_pane(&parent_context_records).unwrap();
        state.apply_event(SessionEvent::ContextTreeUpdated(ContextTreeUpdatedEvent {
            tree: parent_context.tree,
        }));

        assert_eq!(
            state
                .context
                .tree
                .active_node_id()
                .map(|node_id| node_id.as_str()),
            Some("parent-node")
        );

        state.restore_parent_timeline_view();
        assert_eq!(
            state
                .active_context()
                .tree
                .active_node_id()
                .map(|node_id| node_id.as_str()),
            Some("parent-node")
        );
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
            1,
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
        state.apply_event(SessionEvent::PermissionRequested(request));
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
    fn missing_context_detail_target_shows_notice_and_closes_dialog() {
        let mut state = TuiState::default();
        state.open_context_detail(Some(ContextDetailTarget::Block("block-1".into())));
        state.open_dialog(DialogState::new(
            DialogKind::ContextDetail,
            "Detail · Current plan",
            None,
            vec![DialogItem::new(
                "detail",
                "Open detail",
                Some("Outline next steps".into()),
            )],
        ));

        state.apply_event(SessionEvent::ContextViewUpdated(ContextViewUpdatedEvent {
            projection: ContextViewProjection::default(),
        }));

        assert!(state.active_context().open_detail.is_none());
        assert!(state.dialog().is_none());
        assert_eq!(
            state.toast().map(|toast| toast.message.as_str()),
            Some("Context detail closed · Item no longer available")
        );
        assert!(state.timeline.items().is_empty());
    }

    #[test]
    fn restore_parent_view_preserves_pending_permission() {
        let mut state = TuiState::default();
        state.set_pending_permission_projection(Some(PermissionView::from_request(
            PermissionRequestEvent::new("call-1", "shell__exec", "run ls"),
        )));
        state.apply_event(SessionEvent::PermissionRequested(
            PermissionRequestEvent::new("call-1", "shell__exec", "run ls"),
        ));
        state.replace_child_timeline_from_records(
            &[],
            "parent-session",
            "child-session",
            "explorer",
            0,
            1,
            1,
        );

        state.restore_parent_timeline_view();

        assert_eq!(state.phase, AppPhase::WaitingForPermission);
        assert!(state.pending_permission.is_some());
        assert_eq!(state.active_tool_call_id.as_deref(), Some("call-1"));
    }

    #[test]
    fn matching_child_session_event_updates_child_timeline_only() {
        let mut state = TuiState::default();
        state.replace_child_timeline_from_records(
            &[],
            "parent-session",
            "child-session",
            "explorer",
            0,
            1,
            1,
        );

        state.apply_child_session_event(
            "child-session",
            SessionEvent::AssistantDelta(crate::tui::events::AssistantDeltaEvent::new("hello")),
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
        state.apply_event(SessionEvent::ToolStarted(
            crate::tui::events::ToolStartedEvent::new(
                "parent-call",
                "agent__explore",
                "inspect src/tui",
            ),
        ));

        state.apply_child_session_event_with_agent(
            "child-session",
            Some("explorer"),
            Some("parent-call"),
            SessionEvent::ToolStarted(crate::tui::events::ToolStartedEvent::new(
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
        state.apply_event(SessionEvent::ToolStarted(
            crate::tui::events::ToolStartedEvent::new(
                "parent-call",
                "agent__explore",
                "inspect src/tui",
            ),
        ));

        state.apply_child_session_event_with_agent(
            "child-session",
            Some("explorer"),
            Some("parent-call"),
            SessionEvent::ToolCancelled(ToolCancelledEvent::new("child-call", "shell__exec")),
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
        state.apply_event(SessionEvent::ToolStarted(
            crate::tui::events::ToolStartedEvent::new(
                "parent-call",
                "agent__fixer",
                "apply requested fix",
            ),
        ));

        let mut request =
            PermissionRequestEvent::new("perm-1", "shell__exec", "run cargo test --bin letcode");
        request.rationale = Some("validation".into());

        state.apply_child_session_event_with_agent(
            "child-session",
            Some("fixer"),
            Some("parent-call"),
            SessionEvent::PermissionRequested(request),
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
        let output = tool.output.as_deref().expect("live summary payload exists");
        assert!(output.contains("approval needed"), "{output}");
        assert!(output.contains("shell__exec"), "{output}");
        assert!(output.contains("run cargo test --bin letcode"), "{output}");
    }

    #[test]
    fn child_feedback_uses_shared_latest_wins_toast_in_parent_view() {
        let mut state = TuiState::default();
        state.apply_event(SessionEvent::ToolStarted(
            crate::tui::events::ToolStartedEvent::new(
                "parent-call",
                "agent__explore",
                "inspect src/tui",
            ),
        ));
        let parent_timeline = state.timeline.clone();

        for (message, kind, toast_kind) in [
            ("child info", NoticeKind::Info, ToastKind::Info),
            ("child success", NoticeKind::Success, ToastKind::Success),
            (
                "child recoverable error",
                NoticeKind::RecoverableError,
                ToastKind::Error,
            ),
        ] {
            state.apply_child_session_event(
                "child-session",
                SessionEvent::Notice(NoticeEvent::new(message, kind)),
            );
            let toast = state.toast().expect("child notice toast");
            assert_eq!(toast.message, message);
            assert_eq!(toast.kind, toast_kind);
            assert_eq!(toast.ticks_remaining(), ToastState::DEFAULT_TICKS);
        }

        state.apply_event(SessionEvent::Tick);
        state.apply_child_session_event(
            "child-session",
            SessionEvent::ProcessIssue(ProcessIssueEvent::new("child process issue")),
        );

        let toast = state.toast().expect("replacement child issue toast");
        assert_eq!(toast.message, "child process issue");
        assert_eq!(toast.kind, ToastKind::Error);
        assert_eq!(toast.ticks_remaining(), ToastState::DEFAULT_TICKS);
        assert_eq!(state.timeline, parent_timeline);
    }

    #[test]
    fn child_feedback_uses_shared_latest_wins_toast_in_child_view() {
        let mut state = TuiState::default();
        state.apply_event(SessionEvent::ToolStarted(
            crate::tui::events::ToolStartedEvent::new(
                "parent-call",
                "agent__explore",
                "inspect src/tui",
            ),
        ));
        state.replace_child_timeline_from_records(
            &[],
            "parent-session",
            "child-session",
            "explorer",
            0,
            1,
            1,
        );
        let parent_timeline = state.timeline.clone();

        state.apply_child_session_event(
            "child-session",
            SessionEvent::Notice(NoticeEvent::info("child info")),
        );
        state.apply_event(SessionEvent::Tick);
        state.apply_child_session_event(
            "child-session",
            SessionEvent::ProcessIssue(ProcessIssueEvent::new("child process issue")),
        );

        let toast = state.toast().expect("replacement child issue toast");
        assert_eq!(toast.message, "child process issue");
        assert_eq!(toast.kind, ToastKind::Error);
        assert_eq!(toast.ticks_remaining(), ToastState::DEFAULT_TICKS);
        assert!(state.active_timeline().items().is_empty());
        assert_eq!(state.timeline, parent_timeline);
    }

    #[test]
    fn non_matching_child_session_event_is_ignored() {
        let mut state = TuiState::default();
        state.replace_child_timeline_from_records(
            &[],
            "parent-session",
            "child-session",
            "explorer",
            0,
            1,
            1,
        );

        state.apply_child_session_event(
            "other-child",
            SessionEvent::AssistantDelta(crate::tui::events::AssistantDeltaEvent::new("hello")),
        );

        assert!(state.timeline.items().is_empty());
        assert!(state.active_timeline().items().is_empty());
    }

    #[test]
    fn composer_content_inserts_numbered_placeholders_at_attachment_positions() {
        let attachment = |id: &str| UserImageAttachment {
            id: id.into(),
            label: format!("{id}.png"),
            mime: "image/png".into(),
            data_url: "data:image/png;base64,AAAA".into(),
        };
        let mut state = TuiState::default();
        state.set_input("first second third");
        state.input_cursor = "first ".len();
        state.add_composer_attachment(attachment("one"));
        state.input_cursor = state.input_buffer.len() - "third".len();
        state.add_composer_attachment(attachment("two"));

        let content = state.composer_content();

        assert_eq!(content.text, "first [Image 1]second [Image 2]third");
        assert_eq!(
            content.parts(),
            vec![
                UserMessagePart::Text {
                    text: "first ".into(),
                },
                UserMessagePart::Text {
                    text: "[Image 1]".into(),
                },
                UserMessagePart::Image {
                    attachment: attachment("one"),
                },
                UserMessagePart::Text {
                    text: "second ".into(),
                },
                UserMessagePart::Text {
                    text: "[Image 2]".into(),
                },
                UserMessagePart::Image {
                    attachment: attachment("two"),
                },
                UserMessagePart::Text {
                    text: "third".into(),
                },
            ]
        );
    }

    #[test]
    fn set_composer_content_restores_tokens_without_duplicate_image_placeholders() {
        let attachment = UserImageAttachment {
            id: "image".into(),
            label: "image.png".into(),
            mime: "image/png".into(),
            data_url: "data:image/png;base64,AAAA".into(),
        };
        let content = UserMessageContent::from_parts(vec![
            UserMessagePart::Text {
                text: "before ".into(),
            },
            UserMessagePart::Text {
                text: "[Image 1]".into(),
            },
            UserMessagePart::Image {
                attachment: attachment.clone(),
            },
            UserMessagePart::Text {
                text: " after".into(),
            },
        ])
        .with_selected_skills(vec!["rust-audit".into()]);
        let mut state = TuiState::default();

        state.set_composer_content(content);

        assert_eq!(
            state.input_buffer,
            format!(
                "{}before {} after",
                COMPOSER_ATTACHMENT_MARKER, COMPOSER_ATTACHMENT_MARKER
            )
        );
        assert_eq!(state.composer_tokens.len(), 2);
        assert_eq!(state.composer_tokens[0].skill_name(), Some("rust-audit"));
        assert_eq!(state.composer_tokens[1].image(), Some(&attachment));
        assert_eq!(
            state.composer_content().parts(),
            vec![
                UserMessagePart::Text {
                    text: "before ".into(),
                },
                UserMessagePart::Text {
                    text: "[Image 1]".into(),
                },
                UserMessagePart::Image { attachment },
                UserMessagePart::Text {
                    text: " after".into(),
                },
            ]
        );
    }

    #[test]
    fn replacing_or_clearing_input_discards_inline_attachments() {
        let attachment = |id: &str| UserImageAttachment {
            id: id.into(),
            label: format!("{id}.png"),
            mime: "image/png".into(),
            data_url: "data:image/png;base64,AAAA".into(),
        };
        let mut state = TuiState::default();
        state.add_composer_attachment(attachment("first"));

        state.set_input("replacement");
        assert!(state.composer_tokens.is_empty());
        state.assert_composer_token_invariant();

        state.add_composer_attachment(attachment("second"));
        state.clear_input();
        assert!(state.composer_tokens.is_empty());
        state.assert_composer_token_invariant();
    }

    #[test]
    fn compaction_lifecycle_only_commits_after_acknowledged_terminal_event() {
        let mut state = TuiState::default();
        state.apply_event(SessionEvent::CompactionStarted);
        state.apply_event(SessionEvent::CompactionPreviewDelta {
            delta: "working summary".into(),
        });
        assert!(matches!(
            state.timeline.items(),
            [crate::tui::timeline::TimelineItem::Compaction(view)]
                if view.streaming && view.summary == "working summary"
        ));

        state.apply_event(SessionEvent::CompactionCommitted {
            summary: Some("working summary".into()),
        });
        assert!(matches!(
            state.timeline.items(),
            [crate::tui::timeline::TimelineItem::Compaction(view)]
                if !view.streaming && view.summary == "working summary"
        ));
    }

    #[test]
    fn compaction_non_success_and_terminal_events_clear_pending() {
        let terminals = [
            SessionEvent::CompactionNoProgress {
                blockers: vec!["no_historical_items".into()],
            },
            SessionEvent::CompactionFailed,
            SessionEvent::Interrupted,
            SessionEvent::Done,
            SessionEvent::Error(crate::tui::events::ErrorEvent::new("failed")),
        ];
        for terminal in terminals {
            let mut state = TuiState::default();
            state.apply_event(SessionEvent::CompactionStarted);
            state.apply_event(terminal);
            assert!(
                !state.timeline.items().iter().any(|item| matches!(
                    item,
                    crate::tui::timeline::TimelineItem::Compaction(view) if view.streaming
                )),
                "terminal event must clear streaming compaction"
            );
            assert!(
                !state.timeline.items().iter().any(|item| matches!(
                    item,
                    crate::tui::timeline::TimelineItem::Compaction(view) if !view.streaming
                )),
                "non-success must not create a committed compaction"
            );
        }

        let mut state = TuiState::default();
        state.apply_event(SessionEvent::CompactionStarted);
        state.apply_event(SessionEvent::CompactionNoProgress {
            blockers: vec!["no_safe_boundary".into()],
        });
        assert_eq!(
            state.toast().map(|toast| toast.message.as_str()),
            Some("Context limit reached; earlier context cannot be compacted safely yet.")
        );
    }
}
