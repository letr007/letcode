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
use crate::command::ThoughtsDisplayMode;
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

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct PendingComposerSettings {
    pub model: Option<(String, String)>,
    pub reasoning_effort: Option<String>,
    pub permission_mode: Option<String>,
}

impl PendingComposerSettings {
    pub fn clear(&mut self) {
        *self = Self::default();
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ComposerToken {
    Image(UserImageAttachment),
    Skill(String),
    PastedText(String),
}

impl ComposerToken {
    pub fn display_text(&self, image_index: usize) -> String {
        match self {
            Self::Image(_) => format!("[Image {}]", image_index + 1),
            Self::Skill(name) => format!("[Skill: {name}]"),
            Self::PastedText(text) => format!("[Pasted ~{} lines]", pasted_text_line_count(text)),
        }
    }

    #[cfg(test)]
    pub fn image(&self) -> Option<&UserImageAttachment> {
        match self {
            Self::Image(attachment) => Some(attachment),
            Self::Skill(_) | Self::PastedText(_) => None,
        }
    }

    #[cfg(test)]
    pub fn skill_name(&self) -> Option<&str> {
        match self {
            Self::Image(_) | Self::PastedText(_) => None,
            Self::Skill(name) => Some(name),
        }
    }
}

fn pasted_text_line_count(text: &str) -> usize {
    text.split('\n').count().max(1)
}

use anyhow::Result;
use std::collections::{HashMap, HashSet, VecDeque};

/// 文本选择范围
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TextSelection {
    pub start: SelectionAnchor,
    pub end: SelectionAnchor,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TranscriptClickTarget {
    OpenUrl(String),
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
    pub prompt_composition: Vec<crate::agent::PromptCompositionEntry>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DialogItem {
    pub id: String,
    pub label: String,
    pub detail: Option<String>,
    pub section: Option<String>,
    pub right_detail: Option<String>,
    pub checked: bool,
}

impl DialogItem {
    pub fn new(id: impl Into<String>, label: impl Into<String>, detail: Option<String>) -> Self {
        Self {
            id: id.into(),
            label: label.into(),
            detail,
            section: None,
            right_detail: None,
            checked: false,
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

    pub fn with_checked(mut self, checked: bool) -> Self {
        self.checked = checked;
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
    ThoughtsPicker,
    ThemePicker,
    FakePicker,
    LanguagePicker,
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

pub(crate) const MAX_CACHED_CHILD_TIMELINES: usize = 8;

#[derive(Debug, Clone, PartialEq, Eq)]
struct ChildSessionCacheSummary {
    phase: AppPhase,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct ChildTranscriptState {
    session_id: String,
    timeline: Timeline,
    model: Option<String>,
    record_count: usize,
    snapshot_loaded: bool,
    snapshot_dirty: bool,
    context: ContextPaneState,
    active_session: bool,
    latest_auto_continue: AutoContinueState,
    latest_todo: Option<TodoView>,
    retry: Option<RetryNoticeState>,
    phase: AppPhase,
    active_tool_call_id: Option<String>,
    pending_permission: Option<PermissionView>,
    model_token_usage: Option<ModelTokenUsage>,
    compaction_active: bool,
    compaction_animation_start_frame: usize,
    ignore_late_tool_events: bool,
}

impl ChildTranscriptState {
    fn empty(session_id: impl Into<String>) -> Self {
        Self {
            session_id: session_id.into(),
            timeline: Timeline::default(),
            model: None,
            record_count: 0,
            snapshot_loaded: false,
            snapshot_dirty: false,
            context: ContextPaneState::default(),
            active_session: true,
            latest_auto_continue: AutoContinueState::default(),
            latest_todo: None,
            retry: None,
            phase: AppPhase::Completed,
            active_tool_call_id: None,
            pending_permission: None,
            model_token_usage: None,
            compaction_active: false,
            compaction_animation_start_frame: 0,
            ignore_late_tool_events: false,
        }
    }

    fn replace_clean_snapshot(&mut self, records: &[TranscriptRecord], context: ContextPaneState) {
        self.timeline = Timeline::from_transcript_records(records);
        self.model = child_transcript_model(records);
        self.record_count = records.len();
        self.snapshot_loaded = true;
        self.snapshot_dirty = false;
        self.context = context;
    }

    fn from_snapshot(
        session_id: impl Into<String>,
        records: &[TranscriptRecord],
        context: ContextPaneState,
    ) -> Self {
        let mut state = Self::empty(session_id);
        state.replace_clean_snapshot(records, context);
        state
    }
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

impl PendingQuestionItem {
    /// Insert `text` at the custom-edit cursor (grapheme boundary is caller's
    /// responsibility; the insert is byte-based at the clamped cursor).
    pub(super) fn insert_custom_edit(&mut self, text: &str) {
        self.custom_edit_cursor = self.custom_edit_cursor.min(self.custom_edit_text.len());
        self.custom_edit_text
            .insert_str(self.custom_edit_cursor, text);
        self.custom_edit_cursor += text.len();
    }

    /// Delete the character just before the cursor.
    pub(super) fn backspace_custom_edit(&mut self) {
        if self.custom_edit_cursor == 0 {
            return;
        }
        let previous = self.custom_edit_text[..self.custom_edit_cursor]
            .char_indices()
            .last()
            .map(|(index, _)| index)
            .unwrap_or(0);
        self.custom_edit_text
            .drain(previous..self.custom_edit_cursor);
        self.custom_edit_cursor = previous;
    }

    /// Delete the character after the cursor.
    pub(super) fn delete_custom_edit(&mut self) {
        if self.custom_edit_cursor >= self.custom_edit_text.len() {
            return;
        }
        let next = self.custom_edit_text[self.custom_edit_cursor..]
            .char_indices()
            .nth(1)
            .map(|(index, _)| self.custom_edit_cursor + index)
            .unwrap_or(self.custom_edit_text.len());
        self.custom_edit_text.drain(self.custom_edit_cursor..next);
    }

    /// Move the cursor one character left.
    pub(super) fn move_custom_cursor_left(&mut self) {
        if self.custom_edit_cursor == 0 {
            return;
        }
        self.custom_edit_cursor = self.custom_edit_text[..self.custom_edit_cursor]
            .char_indices()
            .last()
            .map(|(index, _)| index)
            .unwrap_or(0);
    }

    /// Move the cursor one character right.
    pub(super) fn move_custom_cursor_right(&mut self) {
        if self.custom_edit_cursor >= self.custom_edit_text.len() {
            return;
        }
        self.custom_edit_cursor = self.custom_edit_text[self.custom_edit_cursor..]
            .char_indices()
            .nth(1)
            .map(|(index, _)| self.custom_edit_cursor + index)
            .unwrap_or(self.custom_edit_text.len());
    }

    /// Move the cursor to the start.
    pub(super) fn move_custom_cursor_home(&mut self) {
        self.custom_edit_cursor = 0;
    }

    /// Move the cursor to the end.
    pub(super) fn move_custom_cursor_end(&mut self) {
        self.custom_edit_cursor = self.custom_edit_text.len();
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PendingQuestionState {
    pub questions: Vec<PendingQuestionItem>,
    pub active_tab: usize,
    pub active_row: usize,
    pub editing_custom: bool,
    pub origin_label: Option<String>,
    pub confirm_scroll: usize,
    confirm_scroll_max: usize,
}

impl PendingQuestionState {
    pub fn new(request: QuestionRequest, origin_label: Option<String>) -> Self {
        Self {
            questions: request
                .questions
                .into_iter()
                .map(PendingQuestionItem::from_spec)
                .collect(),
            active_tab: 0,
            active_row: 0,
            editing_custom: false,
            origin_label,
            confirm_scroll: 0,
            confirm_scroll_max: 0,
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
            self.scroll_confirm_down(1);
            return;
        }
        let row_count = self.current_row_count();
        if row_count > 0 {
            self.active_row = (self.active_row + 1) % row_count;
        }
    }

    pub fn move_prev_row(&mut self) {
        if self.is_confirm_tab() {
            self.scroll_confirm_up(1);
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

    pub fn scroll_confirm_up(&mut self, amount: usize) {
        self.confirm_scroll = self.confirm_scroll.saturating_sub(amount);
    }

    pub fn scroll_confirm_down(&mut self, amount: usize) {
        self.confirm_scroll = self
            .confirm_scroll
            .saturating_add(amount)
            .min(self.confirm_scroll_max);
    }

    pub fn set_confirm_scroll_max(&mut self, max_scroll: usize) {
        self.confirm_scroll_max = max_scroll;
        self.confirm_scroll = self.confirm_scroll.min(max_scroll);
    }

    #[cfg(test)]
    pub fn confirm_scroll_max(&self) -> usize {
        self.confirm_scroll_max
    }

    fn reset_confirm_scroll(&mut self) {
        self.confirm_scroll = 0;
        self.confirm_scroll_max = 0;
    }

    pub fn move_next_tab(&mut self) {
        let total_tabs = self.total_tabs();
        if total_tabs == 0 {
            return;
        }
        self.editing_custom = false;
        self.active_tab = (self.active_tab + 1) % total_tabs;
        self.reset_confirm_scroll();
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
        self.reset_confirm_scroll();
        self.clamp_active_row();
    }

    pub fn pick_option(&mut self, option_index: usize) -> QuestionAdvance {
        let active_tab = self.active_tab;
        let questions_len = self.questions.len();
        let show_confirm = self.show_confirm_tab();
        self.reset_confirm_scroll();
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
        self.reset_confirm_scroll();
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
        self.reset_confirm_scroll();
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
            self.reset_confirm_scroll();
            QuestionAdvance::Advanced
        } else if show_confirm {
            self.active_tab = questions_len;
            self.active_row = 0;
            self.reset_confirm_scroll();
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
    child_timeline_cache_order: VecDeque<String>,
    child_session_summaries: HashMap<String, ChildSessionCacheSummary>,
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
    pub language: Option<crate::tui::i18n::Language>,
    /// Anchored bootstrap experiment active for the current session (composer badge).
    pub anchored_active: bool,
    pub model_token_usage: Option<ModelTokenUsage>,
    pub sidebar_model_token_usage: Option<ModelTokenUsage>,
    /// 上下文压缩进行中：footer 指示条改用开火车式往返扫描，隐藏过期的 token 数字。
    pub compaction_active: bool,
    /// 压缩动画相对于全局 spinner 的起始帧，确保每次都从左→右填充开始。
    pub compaction_animation_start_frame: usize,
    pub reasoning_effort_label: Option<String>,
    pub thoughts_display: ThoughtsDisplayMode,
    pub permission_mode_label: String,
    pub pending_composer_settings: PendingComposerSettings,
    pub session_id: Option<String>,
    pub git_branch: Option<String>,
    pub current_context_branch: String,
    pub active_tool_call_id: Option<String>,
    pub latest_auto_continue: AutoContinueState,
    pub latest_todo: Option<TodoView>,
    pub retry: Option<RetryNoticeState>,
    pub transcript_view: TranscriptViewState,
    pub transcript_scroll: usize,
    pub auto_scroll: bool,
    pub transcript_scrollbar_visible: bool,
    pub sidebar_hidden: bool,
    pub sidebar_forced_open: bool,
    pub sidebar_scroll: u16,
    pub sidebar_max_scroll: u16,
    pub sidebar_context_expanded: bool,
    pub sidebar_mcp_expanded: bool,
    pub sidebar_todos_expanded: bool,
    pub last_sidebar_area: ratatui::layout::Rect,
    pub last_sidebar_context_header: ratatui::layout::Rect,
    pub last_sidebar_mcp_header: ratatui::layout::Rect,
    pub last_sidebar_todos_header: ratatui::layout::Rect,
    pub last_terminal_width: u16,
    pub child_navigation_prefix: bool,
    pub child_navigation_prefix_ticks_remaining: u8,
    pub tool_output_expanded: bool,
    pub tool_output_overrides: HashMap<String, bool>,
    pub theme_id: String,
    pub custom_theme: Option<Theme>,
    pub fake_client: Option<crate::fake::FakeClient>,
    pub fake_installation_id: Option<String>,
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
    pub last_transcript_scroll_top: usize,
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
            child_timeline_cache_order: VecDeque::new(),
            child_session_summaries: HashMap::new(),
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
            language: None,
            anchored_active: false,
            model_token_usage: None,
            sidebar_model_token_usage: None,
            compaction_active: false,
            compaction_animation_start_frame: 0,
            reasoning_effort_label: None,
            thoughts_display: ThoughtsDisplayMode::default(),
            permission_mode_label: "default".into(),
            pending_composer_settings: PendingComposerSettings::default(),
            session_id: None,
            git_branch: None,
            current_context_branch: crate::transcript::ROOT_CONTEXT_BRANCH_ID.into(),
            active_tool_call_id: None,
            latest_auto_continue: AutoContinueState::default(),
            latest_todo: None,
            retry: None,
            transcript_view: TranscriptViewState::Parent,
            transcript_scroll: 0,
            auto_scroll: true,
            transcript_scrollbar_visible: true,
            sidebar_hidden: false,
            sidebar_forced_open: false,
            sidebar_scroll: 0,
            sidebar_max_scroll: 0,
            sidebar_context_expanded: true,
            sidebar_mcp_expanded: true,
            sidebar_todos_expanded: true,
            last_sidebar_area: ratatui::layout::Rect::default(),
            last_sidebar_context_header: ratatui::layout::Rect::default(),
            last_sidebar_mcp_header: ratatui::layout::Rect::default(),
            last_sidebar_todos_header: ratatui::layout::Rect::default(),
            last_terminal_width: 0,
            child_navigation_prefix: false,
            child_navigation_prefix_ticks_remaining: 0,
            tool_output_expanded: false,
            tool_output_overrides: HashMap::new(),
            theme_id: ThemeName::default().as_str().to_string(),
            custom_theme: None,
            fake_client: None,
            fake_installation_id: None,
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

    pub fn set_language(&mut self, language: Option<crate::tui::i18n::Language>) {
        let previous = self.language();
        self.language = language;
        if self.language() != previous {
            self.invalidate_transcript_cache();
        }
    }

    #[cfg(test)]
    pub fn effective_language(&self) -> crate::tui::i18n::Language {
        self.language()
    }

    pub fn language(&self) -> crate::tui::i18n::Language {
        self.language
            .unwrap_or_else(crate::tui::i18n::system_language)
    }

    pub fn translator(&self) -> crate::tui::i18n::Translator {
        crate::tui::i18n::Translator::new(self.language())
    }

    pub fn t(&self, key: &str) -> String {
        self.translator().t(key)
    }

    pub fn t_fmt(&self, key: &str, args: &[(&str, &str)]) -> String {
        self.translator().t_fmt(key, args)
    }

    pub fn set_model(&mut self, model_id: impl Into<String>, model_label: impl Into<String>) {
        self.model_id = model_id.into();
        self.model_label = model_label.into();
    }

    pub fn set_fast_mode_enabled(&mut self, enabled: bool) {
        self.fast_mode_enabled = enabled;
    }

    pub fn set_anchored_active(&mut self, active: bool) {
        self.anchored_active = active;
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
                prompt_composition: Vec::new(),
            });
        self.sidebar_model_token_usage = self.model_token_usage.clone();
    }

    pub fn set_reasoning_effort_label(&mut self, label: Option<String>) {
        self.reasoning_effort_label = label;
    }

    pub fn set_thoughts_display(&mut self, mode: ThoughtsDisplayMode) {
        if self.thoughts_display != mode {
            self.thoughts_display = mode;
            self.invalidate_transcript_cache();
            self.last_transcript_total_rows = None;
        }
    }

    pub fn set_pending_model(
        &mut self,
        model_id: impl Into<String>,
        model_label: impl Into<String>,
    ) {
        self.pending_composer_settings.model = Some((model_id.into(), model_label.into()));
    }

    pub fn set_pending_reasoning_effort(&mut self, label: impl Into<String>) {
        self.pending_composer_settings.reasoning_effort = Some(label.into());
    }

    pub fn set_pending_permission_mode(&mut self, label: impl Into<String>) {
        self.pending_composer_settings.permission_mode = Some(label.into());
    }

    pub fn clear_pending_model(&mut self) {
        self.pending_composer_settings.model = None;
    }

    pub fn clear_pending_model_if(&mut self, model_id: &str) {
        if self
            .pending_composer_settings
            .model
            .as_ref()
            .is_some_and(|(pending_id, _)| pending_id == model_id)
        {
            self.clear_pending_model();
        }
    }

    pub fn clear_pending_reasoning_effort(&mut self) {
        self.pending_composer_settings.reasoning_effort = None;
    }

    pub fn clear_pending_reasoning_effort_if(&mut self, label: &str) {
        if self.pending_composer_settings.reasoning_effort.as_deref() == Some(label) {
            self.clear_pending_reasoning_effort();
        }
    }

    pub fn clear_pending_permission_mode(&mut self) {
        self.pending_composer_settings.permission_mode = None;
    }

    pub fn clear_pending_permission_mode_if(&mut self, label: &str) {
        if self.pending_composer_settings.permission_mode.as_deref() == Some(label) {
            self.clear_pending_permission_mode();
        }
    }

    pub fn clear_pending_composer_settings(&mut self) {
        self.pending_composer_settings.clear();
    }

    pub fn theme(&self) -> Theme {
        if let Some(builtin) = ThemeName::parse(&self.theme_id) {
            Theme::for_name(builtin, self.status_spinner_frame)
        } else {
            self.custom_theme.unwrap_or_else(Theme::dark)
        }
    }

    pub fn transcript_theme(&self) -> Theme {
        if let Some(builtin) = ThemeName::parse(&self.theme_id) {
            Theme::for_name(builtin, 0)
        } else {
            self.custom_theme.unwrap_or_else(Theme::dark)
        }
    }

    pub fn set_theme_name(&mut self, theme_name: ThemeName) {
        self.set_active_theme(theme_name.as_str().to_string(), None);
    }

    pub fn set_fake_client(&mut self, client: Option<crate::fake::FakeClient>) {
        self.fake_client = client;
    }

    pub fn set_active_theme(&mut self, theme_id: String, custom_theme: Option<Theme>) {
        if self.theme_id != theme_id || self.custom_theme != custom_theme {
            self.theme_id = theme_id;
            self.custom_theme = custom_theme;
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

    pub fn set_sidebar_preference(&mut self, hidden: bool, forced_open: bool) {
        self.sidebar_hidden = hidden;
        self.sidebar_forced_open = !hidden && forced_open;
    }

    pub fn toggle_sidebar(&mut self) {
        if self.sidebar_visible(self.last_terminal_width) {
            self.sidebar_hidden = true;
            self.sidebar_forced_open = false;
        } else {
            self.sidebar_hidden = false;
            self.sidebar_forced_open = true;
        }
    }

    pub fn sidebar_visible(&self, terminal_width: u16) -> bool {
        !self.is_read_only_child_view()
            && !self.sidebar_hidden
            && (self.sidebar_forced_open || terminal_width > 120)
    }

    pub fn sync_sidebar_scroll(&mut self, total_rows: usize, viewport_rows: u16) {
        self.sidebar_max_scroll =
            u16::try_from(total_rows.saturating_sub(viewport_rows as usize)).unwrap_or(u16::MAX);
        self.sidebar_scroll = self.sidebar_scroll.min(self.sidebar_max_scroll);
    }

    pub fn scroll_sidebar_up(&mut self, rows: u16) {
        self.sidebar_scroll = self.sidebar_scroll.saturating_sub(rows);
    }

    pub fn scroll_sidebar_down(&mut self, rows: u16) {
        self.sidebar_scroll = self
            .sidebar_scroll
            .saturating_add(rows)
            .min(self.sidebar_max_scroll);
    }

    pub fn toggle_sidebar_context(&mut self) {
        toggle_sidebar_section(&mut self.sidebar_context_expanded, &mut self.sidebar_scroll);
    }

    pub fn toggle_sidebar_mcp(&mut self) {
        toggle_sidebar_section(&mut self.sidebar_mcp_expanded, &mut self.sidebar_scroll);
    }

    pub fn toggle_sidebar_todos(&mut self) {
        toggle_sidebar_section(&mut self.sidebar_todos_expanded, &mut self.sidebar_scroll);
    }

    #[cfg(test)]
    pub fn scroll_sidebar_to_bottom(&mut self) {
        self.sidebar_scroll = self.sidebar_max_scroll;
    }

    pub fn set_transcript_scrollbar_visible(&mut self, visible: bool) {
        if self.transcript_scrollbar_visible != visible {
            self.transcript_scrollbar_visible = visible;
            self.invalidate_transcript_cache();
            self.last_transcript_total_rows = None;
        }
    }

    pub fn set_token_usage(&mut self, usage: ModelTokenUsage) {
        self.sidebar_model_token_usage = Some(usage.clone());
        self.model_token_usage = Some(usage);
    }

    pub fn apply_live_token_usage(&mut self, usage: ModelTokenUsage) {
        self.model_token_usage = Some(usage);
    }

    pub fn commit_sidebar_token_usage(&mut self) {
        self.sidebar_model_token_usage = self.model_token_usage.clone();
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

    pub fn active_phase(&self) -> AppPhase {
        if self.is_read_only_child_view() {
            self.child_timeline
                .as_ref()
                .map(|state| state.phase)
                .unwrap_or(self.phase)
        } else {
            self.phase
        }
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

    pub fn active_sidebar_model_token_usage(&self) -> Option<&ModelTokenUsage> {
        if self.is_read_only_child_view() {
            self.child_timeline
                .as_ref()
                .and_then(|state| state.model_token_usage.as_ref())
        } else {
            self.sidebar_model_token_usage.as_ref()
        }
    }

    pub fn active_model_token_usage(&self) -> Option<&ModelTokenUsage> {
        if self.is_read_only_child_view() {
            self.child_timeline
                .as_ref()
                .and_then(|state| state.model_token_usage.as_ref())
        } else {
            self.model_token_usage.as_ref()
        }
    }

    pub fn merge_parent_prompt_composition(&self, usage: &mut TokenUsageEvent) {
        merge_prompt_composition(self.model_token_usage.as_ref(), usage);
    }

    pub fn merge_child_prompt_composition(&self, child_session_id: &str, event: &mut SessionEvent) {
        let SessionEvent::TokenUsage(usage) = event else {
            return;
        };
        let previous = self
            .child_timeline
            .as_ref()
            .filter(|child| child.session_id == child_session_id)
            .and_then(|child| child.model_token_usage.as_ref())
            .or_else(|| {
                self.child_timeline_cache
                    .get(child_session_id)
                    .and_then(|child| child.model_token_usage.as_ref())
            });
        merge_prompt_composition(previous, usage);
    }

    pub fn active_compaction_animation_start_frame(&self) -> Option<usize> {
        let (active, start_frame) = if self.is_read_only_child_view() {
            self.child_timeline
                .as_ref()
                .map(|state| {
                    (
                        state.compaction_active,
                        state.compaction_animation_start_frame,
                    )
                })
                .unwrap_or((false, 0))
        } else {
            (
                self.compaction_active,
                self.compaction_animation_start_frame,
            )
        };
        active.then_some(start_frame)
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

        u16::try_from(measure::max_scroll(rows, height)).unwrap_or(u16::MAX)
    }

    pub fn open_context_detail(&mut self, target: Option<ContextDetailTarget>) {
        if self.is_read_only_child_view()
            && let Some(child) = self.child_timeline.as_mut()
        {
            child.context.open_detail = target;
            return;
        }

        self.context.open_detail = target;
    }

    pub fn mark_session_active(&mut self) {
        self.active_session = true;
    }

    pub fn push_queued_user_message_preview(&mut self, submission: UserMessageSubmission) {
        if self.timeline.items().iter().any(|item| {
            matches!(
                item,
                TimelineItem::User(message)
                    if message.submission_id.as_deref() == Some(submission.id.as_str())
                        && message.queued
            )
        }) {
            return;
        }
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
                    Some(ComposerToken::PastedText(pasted)) => {
                        parts.push(UserMessagePart::Text {
                            text: pasted.clone(),
                        });
                    }
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

    pub fn add_composer_pasted_text(&mut self, text: String) {
        self.insert_composer_token(ComposerToken::PastedText(text));
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
    pub fn transcript_scroll_offset(&self) -> usize {
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

    pub fn scroll_transcript_up(&mut self, rows: usize) {
        self.transcript_scroll = self.transcript_scroll.saturating_add(rows);
        if let Some(total_rows) = self.last_transcript_total_rows {
            self.transcript_scroll = self.transcript_scroll.min(measure::max_scroll(
                total_rows,
                self.last_transcript_area.height,
            ));
        }
        self.auto_scroll = self.transcript_scroll == 0;
    }

    pub fn scroll_transcript_down(&mut self, rows: usize) {
        self.transcript_scroll = self.transcript_scroll.saturating_sub(rows);
        self.auto_scroll = self.transcript_scroll == 0;
    }

    pub fn scroll_transcript_to_bottom(&mut self) {
        self.transcript_scroll = 0;
        self.auto_scroll = true;
    }

    pub fn sync_transcript_viewport_rows_with_reflow(
        &mut self,
        total_rows: usize,
        width_reflowed: bool,
    ) {
        if !width_reflowed
            && !self.auto_scroll
            && let Some(previous_total_rows) = self.last_transcript_total_rows
            && total_rows > previous_total_rows
        {
            let delta = total_rows.saturating_sub(previous_total_rows);
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

    pub fn set_git_branch(&mut self, branch: Option<String>) {
        self.git_branch = branch;
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

        self.cache_active_child_timeline();
        self.active_session = true;
        self.timeline = timeline;
        self.context = context;
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
        let mut child_state = project_child_timeline_state(records)?;
        let child_session_id = child_session_id.into();
        child_state.session_id = child_session_id.clone();
        self.active_session = true;
        self.clear_input();
        self.close_dialog();
        self.reset_slash_panel();
        self.child_timeline = Some(child_state);
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
        let parent_session_id = parent_session_id.into();
        let child_session_id = child_session_id.into();
        let agent_name = agent_name.into();
        let mut context = ContextPaneState::default();
        apply_runtime_context(
            &mut context,
            runtime_context,
            crate::tui::events::RuntimeContextDisposition::ReplaceScope,
        );

        if let Some(active) = self.child_timeline.as_mut()
            && active.session_id == child_session_id
        {
            if active.snapshot_dirty {
                active.context = context;
                active.model = child_transcript_model(records);
            } else {
                active.replace_clean_snapshot(records, context);
            }
            self.active_session = true;
            self.clear_input();
            self.close_dialog();
            self.reset_slash_panel();
            self.retry = None;
            self.transcript_view = TranscriptViewState::Child {
                parent_session_id,
                child_session_id,
                agent_name,
                index,
                total,
                pool_ordinal,
            };
            self.scroll_transcript_to_bottom();
            self.invalidate_transcript_cache();
            self.last_transcript_total_rows = None;
            return Ok(());
        }

        self.child_timeline_cache_order
            .retain(|cached_id| cached_id != &child_session_id);
        self.child_session_summaries.remove(&child_session_id);
        let child_state = match self.child_timeline_cache.remove(&child_session_id) {
            Some(mut cached) => {
                if cached.snapshot_dirty {
                    cached.context = context;
                    cached.model = child_transcript_model(records);
                } else {
                    cached.replace_clean_snapshot(records, context);
                }
                cached
            }
            None => ChildTranscriptState::from_snapshot(child_session_id.clone(), records, context),
        };

        if let Some(active_child) = self.child_timeline.take() {
            self.cache_child_timeline(active_child);
        }
        self.child_timeline = Some(child_state);
        self.active_session = true;
        self.clear_input();
        self.close_dialog();
        self.reset_slash_panel();
        self.retry = None;
        self.transcript_view = TranscriptViewState::Child {
            parent_session_id,
            child_session_id,
            agent_name,
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
            snapshot_loaded: true,
            snapshot_dirty: false,
            context,
            active_session: true,
            latest_auto_continue: AutoContinueState::default(),
            latest_todo: None,
            retry: None,
            phase: AppPhase::Completed,
            active_tool_call_id: None,
            pending_permission: None,
            model_token_usage: None,
            compaction_active: false,
            compaction_animation_start_frame: 0,
            ignore_late_tool_events: false,
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
    pub fn child_view_has_unpersisted_projection(&self) -> bool {
        self.child_timeline
            .as_ref()
            .map(|child| child.snapshot_dirty)
            .unwrap_or(false)
    }

    #[cfg(test)]
    pub fn child_view_phase(&self) -> Option<AppPhase> {
        self.child_timeline.as_ref().map(|child| child.phase)
    }

    #[cfg(test)]
    pub fn set_child_view_phase_for_test(&mut self, phase: AppPhase) {
        if let Some(child) = self.child_timeline.as_mut() {
            child.phase = phase;
        }
    }

    #[cfg(test)]
    pub fn cached_child_phase(&self, child_session_id: &str) -> Option<AppPhase> {
        self.child_timeline_cache
            .get(child_session_id)
            .map(|child| child.phase)
            .or_else(|| {
                self.child_session_summaries
                    .get(child_session_id)
                    .map(|summary| summary.phase)
            })
    }

    #[cfg(test)]
    pub fn child_timeline_cache_contains(&self, child_session_id: &str) -> bool {
        self.child_timeline_cache.contains_key(child_session_id)
    }

    #[cfg(test)]
    pub fn child_timeline_cache_len(&self) -> usize {
        self.child_timeline_cache.len()
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

    fn touch_cached_child_timeline(&mut self, child_session_id: &str) {
        if self.child_timeline_cache.contains_key(child_session_id) {
            self.child_timeline_cache_order
                .retain(|cached_id| cached_id != child_session_id);
            self.child_timeline_cache_order
                .push_back(child_session_id.to_string());
        }
    }

    fn cache_child_timeline(&mut self, child: ChildTranscriptState) {
        let child_session_id = child.session_id.clone();
        self.child_session_summaries.remove(&child_session_id);
        self.child_timeline_cache_order
            .retain(|cached_id| cached_id != &child_session_id);
        self.child_timeline_cache_order
            .push_back(child_session_id.clone());
        self.child_timeline_cache.insert(child_session_id, child);
        while self.child_timeline_cache_order.len() > MAX_CACHED_CHILD_TIMELINES {
            let Some(candidate_index) = self.child_timeline_cache_order.iter().position(|id| {
                self.child_timeline_cache.get(id).is_some_and(|child| {
                    matches!(child.phase, AppPhase::Completed | AppPhase::Error)
                        && !child.snapshot_dirty
                })
            }) else {
                break;
            };
            let evicted = self
                .child_timeline_cache_order
                .remove(candidate_index)
                .expect("eviction candidate index came from cache order");
            if let Some(child) = self.child_timeline_cache.remove(&evicted) {
                self.child_session_summaries
                    .insert(evicted, ChildSessionCacheSummary { phase: child.phase });
            }
        }
    }

    pub fn cache_active_child_timeline(&mut self) {
        if let Some(active_child) = self.child_timeline.take() {
            self.cache_child_timeline(active_child);
        }
    }

    pub fn clear_child_timeline_cache(&mut self) {
        self.child_timeline_cache.clear();
        self.child_timeline_cache_order.clear();
        self.child_session_summaries.clear();
    }

    pub fn try_restore_parent_timeline_view_with_runtime_context(
        &mut self,
        records: &[TranscriptRecord],
        runtime_context: RuntimeActiveContext,
    ) -> Result<()> {
        validate_lifecycle_records(records, &runtime_context)?;
        apply_runtime_context(
            &mut self.context,
            runtime_context,
            crate::tui::events::RuntimeContextDisposition::ReplaceScope,
        );
        self.restore_parent_timeline_view();
        Ok(())
    }

    pub fn restore_parent_timeline_view(&mut self) {
        self.cache_active_child_timeline();
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
        self.sidebar_model_token_usage = None;
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
        let viewing_child = matches!(
            &self.transcript_view,
            TranscriptViewState::Child {
                child_session_id: active_child_session_id,
                ..
            } if active_child_session_id == child_session_id
        );

        match event {
            SessionEvent::Notice(notice) => {
                match notice.kind {
                    crate::tui::events::NoticeKind::Info if viewing_child => {
                        self.show_toast(notice.message, ToastKind::Info);
                    }
                    crate::tui::events::NoticeKind::Success if viewing_child => {
                        self.show_toast(notice.message, ToastKind::Success);
                    }
                    crate::tui::events::NoticeKind::RecoverableError => {
                        self.show_toast(
                            child_feedback_message(agent_name, child_session_id, &notice.message),
                            ToastKind::Error,
                        );
                    }
                    crate::tui::events::NoticeKind::Info
                    | crate::tui::events::NoticeKind::Success => {}
                }
                return;
            }
            SessionEvent::RetryScheduled(ref retry) => {
                let message = format!(
                    "{} · Retrying in {}s · attempt {} of {}",
                    retry.error, retry.delay_secs, retry.attempt, retry.max_attempts
                );
                self.show_toast(
                    child_feedback_message(agent_name, child_session_id, &message),
                    ToastKind::Error,
                );
            }
            SessionEvent::ProcessIssue(issue) => {
                self.show_toast(
                    child_feedback_message(agent_name, child_session_id, &issue.message),
                    ToastKind::Error,
                );
                return;
            }
            _ => {}
        }

        self.project_child_event_to_parent_subagent_tool(
            child_session_id,
            agent_name,
            parent_tool_call_id,
            &event,
        );

        let terminal_event = matches!(
            &event,
            SessionEvent::Error(_) | SessionEvent::Done | SessionEvent::Interrupted
        );
        if !self.child_event_targets_loaded_child(child_session_id) {
            if let Some(summary) = self.child_session_summaries.get_mut(child_session_id) {
                update_child_session_summary(summary, &event);
                return;
            }
            if matches!(
                &event,
                SessionEvent::Error(_) | SessionEvent::Done | SessionEvent::Interrupted
            ) {
                self.child_session_summaries.insert(
                    child_session_id.to_string(),
                    ChildSessionCacheSummary {
                        phase: child_phase_for_event(&event),
                    },
                );
                return;
            }
            self.cache_child_timeline(ChildTranscriptState::empty(child_session_id));
        } else if !viewing_child {
            self.touch_cached_child_timeline(child_session_id);
        }
        if !self.child_event_targets_loaded_child(child_session_id) {
            if let Some(summary) = self.child_session_summaries.get_mut(child_session_id) {
                update_child_session_summary(summary, &event);
            }
            return;
        }

        if self.apply_child_context_event(child_session_id, &event, viewing_child) {
            return;
        }

        let child_toast = {
            let TuiState {
                child_timeline,
                child_timeline_cache,
                status_spinner_frame,
                ..
            } = self;
            let child = if child_timeline
                .as_ref()
                .is_some_and(|child| child.session_id == child_session_id)
            {
                child_timeline
                    .as_mut()
                    .expect("matching active child timeline exists")
            } else {
                child_timeline_cache
                    .get_mut(child_session_id)
                    .expect("child timeline cache entry exists")
            };
            let mut child_toast = None;
            apply_event_to_child_transcript(child, event, status_spinner_frame, &mut child_toast);
            child_toast
        };

        if viewing_child {
            if child_toast.is_some() {
                self.toast = child_toast;
            }
            self.invalidate_transcript_cache();
            self.last_transcript_total_rows = None;
        } else if terminal_event
            && let Some(child) = self.child_timeline_cache.remove(child_session_id)
        {
            self.child_timeline_cache_order
                .retain(|cached_id| cached_id != child_session_id);
            self.child_session_summaries.insert(
                child_session_id.to_string(),
                ChildSessionCacheSummary { phase: child.phase },
            );
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
        let TuiState {
            child_timeline,
            child_timeline_cache,
            ..
        } = self;
        let Some(child) = (if child_timeline
            .as_ref()
            .is_some_and(|child| child.session_id == child_session_id)
        {
            child_timeline.as_mut()
        } else {
            child_timeline_cache.get_mut(child_session_id)
        }) else {
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
            SessionEvent::RuntimeContextUpdated(_) => true,
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
            if detail_closed && viewing_child {
                if matches!(
                    self.dialog.as_ref().map(|dialog| &dialog.kind),
                    Some(DialogKind::ContextDetail)
                ) {
                    self.close_dialog();
                }
                self.show_toast(
                    "Context detail closed · Item no longer available",
                    ToastKind::Info,
                );
            }
        }
        handled
    }

    fn child_event_targets_loaded_child(&self, child_session_id: &str) -> bool {
        self.child_timeline
            .as_ref()
            .is_some_and(|child| child.session_id == child_session_id)
            || self.child_timeline_cache.contains_key(child_session_id)
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

    pub fn update_background_subagent_result(
        &mut self,
        parent_tool_call_id: Option<&str>,
        result: &crate::subagent::SubagentRunSummary,
    ) {
        let Some(parent_tool_call_id) = parent_tool_call_id else {
            return;
        };
        if self
            .timeline
            .finish_background_subagent_tool(parent_tool_call_id, result)
        {
            self.transcript_render_cache.invalidate_row_metadata();
            self.last_transcript_total_rows = None;
        }
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
            self.transcript_render_cache.invalidate_row_metadata();
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

        let absolute_row = (terminal_row - area.y) as usize + self.last_transcript_scroll_top;
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
                if let Some(Interaction::OpenUrl(url)) = &span.interaction {
                    // Mouse capture owns the click; return the URL so the runtime
                    // can open it. Unsafe targets stay inert and still block tool-card toggles.
                    return crate::tui::transcript_ratatui::safe_hyperlink_url(url)
                        .then(|| TranscriptClickTarget::OpenUrl(url.clone()));
                }
                break;
            }
            visual_col = visual_col.saturating_add(span_width);
        }

        match self.active_timeline().items().get(item_index) {
            Some(TimelineItem::Tool(tool)) => {
                Some(TranscriptClickTarget::ToolCard(tool.call_id.clone()))
            }
            Some(TimelineItem::AutoReview(decision)) => Some(TranscriptClickTarget::ToolCard(
                format!("auto-review:{}", decision.call_id),
            )),
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
        let absolute_row = viewport_row as usize + self.last_transcript_scroll_top;

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
        let line = entry.document.lines.get(rendered_line_offset)?;

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

fn child_phase_for_event(event: &SessionEvent) -> AppPhase {
    match event {
        SessionEvent::Error(_) => AppPhase::Error,
        SessionEvent::Done | SessionEvent::Interrupted => AppPhase::Completed,
        SessionEvent::PermissionRequested(_) => AppPhase::WaitingForPermission,
        _ => AppPhase::Running,
    }
}

fn update_child_session_summary(summary: &mut ChildSessionCacheSummary, event: &SessionEvent) {
    if matches!(summary.phase, AppPhase::Completed | AppPhase::Error) {
        return;
    }
    summary.phase = child_phase_for_event(event);
}

fn child_feedback_message(
    agent_name: Option<&str>,
    child_session_id: &str,
    message: &str,
) -> String {
    let child = crate::tui::components::tool_card::truncate_display_width(child_session_id, 16);
    match agent_name.filter(|name| !name.is_empty()) {
        Some(agent_name) => format!("{agent_name} · {child}: {message}"),
        None => format!("{child}: {message}"),
    }
}

fn apply_event_to_child_transcript(
    child: &mut ChildTranscriptState,
    event: SessionEvent,
    status_spinner_frame: &mut usize,
    toast: &mut Option<ToastState>,
) {
    let terminal_event = matches!(
        &event,
        SessionEvent::Error(_) | SessionEvent::Done | SessionEvent::Interrupted
    );
    let timeline_revision = child.timeline.mutation_revision();

    match event {
        SessionEvent::PermissionRequested(request) => {
            child.phase = AppPhase::WaitingForPermission;
            child.active_tool_call_id = Some(request.call_id.clone());
            child.pending_permission = Some(PermissionView::from_request(request.clone()));
            child.timeline.push_permission_request(request);
        }
        SessionEvent::PermissionResolved(resolution) => {
            child.phase = AppPhase::Running;
            child.active_tool_call_id = None;
            child.pending_permission = None;
            child.timeline.resolve_permission(resolution);
        }
        event => {
            let accepts_tool_events = !child.ignore_late_tool_events;
            let mut child_quit_requested = false;
            apply_projected_session_event(
                EventProjection {
                    active_session: &mut child.active_session,
                    latest_auto_continue: &mut child.latest_auto_continue,
                    latest_todo: &mut child.latest_todo,
                    retry: &mut child.retry,
                    phase: &mut child.phase,
                    active_tool_call_id: &mut child.active_tool_call_id,
                    pending_permission: &mut child.pending_permission,
                    model_token_usage: &mut child.model_token_usage,
                    compaction_active: &mut child.compaction_active,
                    compaction_animation_start_frame: &mut child.compaction_animation_start_frame,
                    ignore_late_tool_events: &mut child.ignore_late_tool_events,
                    quit_requested: &mut child_quit_requested,
                    status_spinner_frame,
                    toast,
                    timeline: &mut child.timeline,
                    accepts_tool_events: true,
                }
                .with_tool_event_acceptance(accepts_tool_events),
                event,
            );
        }
    }

    if child.timeline.mutation_revision() != timeline_revision
        && (child.snapshot_loaded || !terminal_event)
    {
        child.snapshot_dirty = true;
    }
}

fn toggle_sidebar_section(expanded: &mut bool, scroll: &mut u16) {
    *expanded = !*expanded;
    *scroll = 0;
}

fn merge_prompt_composition(previous: Option<&ModelTokenUsage>, usage: &mut TokenUsageEvent) {
    let preserve_monotonic_context = !usage.prompt_composition.is_empty();
    let Some(previous) = previous else {
        return;
    };
    let previous = TokenUsageEvent::with_breakdown(
        previous.used_tokens,
        previous.context_window_tokens,
        previous.input_tokens,
        previous.output_tokens,
        previous.cached_tokens,
    )
    .with_cache_report(previous.cache_report.clone())
    .with_prompt_composition(previous.prompt_composition.clone());
    usage.merge_prompt_composition_from(&previous);
    if preserve_monotonic_context {
        usage.preserve_context_floor_from(&previous);
    }
}

impl From<TokenUsageEvent> for ModelTokenUsage {
    fn from(event: TokenUsageEvent) -> Self {
        Self {
            used_tokens: event.used_tokens,
            context_window_tokens: event.context_window_tokens,
            input_tokens: event.input_tokens,
            output_tokens: event.output_tokens,
            cached_tokens: event.cached_tokens,
            cache_report: event.cache_report,
            prompt_composition: event.prompt_composition,
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

    #[test]
    fn explicit_and_system_following_language_resolution_are_distinct() {
        let mut state = TuiState::default();
        assert_eq!(
            state.effective_language(),
            crate::tui::i18n::system_language()
        );
        state.set_language(Some(crate::tui::i18n::Language::ZhCn));
        assert_eq!(state.effective_language(), crate::tui::i18n::Language::ZhCn);
        state.set_language(None);
        assert_eq!(
            state.effective_language(),
            crate::tui::i18n::system_language()
        );
    }

    fn custom_edit_item(initial: &str) -> PendingQuestionItem {
        PendingQuestionItem {
            question: "q".into(),
            header: "h".into(),
            options: Vec::new(),
            multiple: false,
            selected_labels: Vec::new(),
            custom_text: String::new(),
            custom_cursor: 0,
            custom_edit_text: initial.into(),
            custom_edit_cursor: initial.len(),
        }
    }

    #[test]
    fn custom_edit_insert_backspace_delete_keep_unicode_intact() {
        let mut item = custom_edit_item("猫😺a");
        item.move_custom_cursor_left();
        item.move_custom_cursor_left();
        item.insert_custom_edit("喵");
        assert_eq!(item.custom_edit_text, "猫喵😺a");

        item.backspace_custom_edit();
        assert_eq!(item.custom_edit_text, "猫😺a");

        item.move_custom_cursor_home();
        item.delete_custom_edit();
        assert_eq!(item.custom_edit_text, "😺a");

        item.move_custom_cursor_end();
        item.insert_custom_edit("！");
        assert_eq!(item.custom_edit_text, "😺a！");
    }

    #[test]
    fn custom_edit_cursor_never_lands_inside_a_multibyte_char() {
        let mut item = custom_edit_item("😺😺");
        item.move_custom_cursor_left();
        // Cursor points at a char boundary; left again hits the first char.
        assert_eq!(item.custom_edit_cursor, "😺".len());
        item.move_custom_cursor_left();
        assert_eq!(item.custom_edit_cursor, 0);
        // Left at the start is a no-op.
        item.move_custom_cursor_left();
        assert_eq!(item.custom_edit_cursor, 0);

        item.move_custom_cursor_end();
        item.move_custom_cursor_right();
        assert_eq!(item.custom_edit_cursor, item.custom_edit_text.len());
    }

    #[test]
    fn custom_edit_backspace_and_delete_at_boundaries_are_noops() {
        let mut item = custom_edit_item("ab");
        item.move_custom_cursor_home();
        item.backspace_custom_edit();
        assert_eq!(item.custom_edit_text, "ab");
        item.move_custom_cursor_end();
        item.delete_custom_edit();
        assert_eq!(item.custom_edit_text, "ab");
    }

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
    fn auto_review_toggle_key_uses_call_id() {
        let mut state = TuiState::default();
        assert!(
            !state
                .tool_output_overrides
                .contains_key("auto-review:call-review")
        );

        state.toggle_tool_output("auto-review:call-review");
        assert_eq!(
            state.tool_output_overrides.get("auto-review:call-review"),
            Some(&true)
        );

        state.toggle_tool_output("auto-review:call-review");
        assert_eq!(
            state.tool_output_overrides.get("auto-review:call-review"),
            Some(&false)
        );
    }

    #[test]
    fn confirm_scroll_clamps_and_resets_on_tab_or_answer_changes() {
        let mut state = question_state(vec![
            QuestionSpec {
                question: "First".into(),
                header: "One".into(),
                options: vec![option("A")],
                multiple: true,
            },
            QuestionSpec {
                question: "Second".into(),
                header: "Two".into(),
                options: vec![option("B")],
                multiple: true,
            },
        ]);
        state.focus_tab(2);
        state.set_confirm_scroll_max(3);
        state.move_next_row();
        state.move_next_row();
        state.move_next_row();
        state.move_next_row();
        assert_eq!(state.confirm_scroll, 3);
        state.move_prev_row();
        assert_eq!(state.confirm_scroll, 2);
        state.focus_tab(0);
        assert_eq!(state.confirm_scroll, 0);
        state.focus_tab(2);
        state.set_confirm_scroll_max(3);
        state.move_next_row();
        state.pick_option(0);
        assert_eq!(state.confirm_scroll, 0);
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
    fn sidebar_defaults_to_wide_auto_and_can_be_forced_on_narrow_terminals() {
        let mut state = TuiState::default();
        assert!(state.sidebar_visible(121));
        assert!(!state.sidebar_visible(120));

        state.last_terminal_width = 100;
        state.toggle_sidebar();
        assert!(state.sidebar_visible(100));
        state.toggle_sidebar();
        assert!(!state.sidebar_visible(160));
    }

    #[test]
    fn composer_draft_keeps_boundary_skill_tokens_after_outer_text_trim() {
        let mut state = TuiState::default();
        state.add_composer_skill("rust-audit".into());
        state.input_buffer.push_str("  inspect  ");
        state.input_cursor = state.input_buffer.len();

        let mut content = state.composer_content();
        content.trim_outer_text();
        assert_eq!(content.text, "inspect");
        assert_eq!(content.selected_skills, vec!["rust-audit"]);
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
            prompt_composition: Vec::new(),
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

        // Child lifecycle state is local; the parent remains untouched.
        assert_eq!(state.phase, AppPhase::Idle);
        assert_eq!(state.active_tool_call_id, None);
        assert_eq!(state.child_view_phase(), Some(AppPhase::Completed));

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
    fn scroll_up_while_transcript_fits_viewport_keeps_auto_scroll() {
        let mut state = TuiState::default();
        state.last_transcript_area.height = 10;
        state.sync_transcript_viewport_rows_with_reflow(3, false);

        state.scroll_transcript_up(1);
        state.sync_transcript_viewport_rows_with_reflow(12, false);

        assert_eq!(state.transcript_scroll_offset(), 0);
        assert!(state.auto_scroll);
    }

    #[test]
    fn manual_history_offset_tracks_large_transcript_growth() {
        let mut state = TuiState::default();
        state.last_transcript_area.height = 30;
        state.sync_transcript_viewport_rows_with_reflow(60_000, false);
        state.scroll_transcript_up(59_970);
        assert_eq!(state.transcript_scroll_offset(), 59_970);
        assert!(!state.auto_scroll);

        state.sync_transcript_viewport_rows_with_reflow(90_000, false);
        assert_eq!(state.transcript_scroll_offset(), 89_970);
        state.scroll_transcript_up(1);
        assert_eq!(state.transcript_scroll_offset(), 89_970);
    }

    #[test]
    fn sync_transcript_viewport_rows_clamps_invalid_offset_and_restores_auto_scroll() {
        let mut state = TuiState::default();
        state.last_transcript_area.height = 5;
        state.sync_transcript_viewport_rows_with_reflow(10, false);
        state.scroll_transcript_up(4);
        assert_eq!(state.transcript_scroll_offset(), 4);
        assert!(!state.auto_scroll);

        state.last_transcript_area.height = 10;
        state.sync_transcript_viewport_rows_with_reflow(10, false);

        assert_eq!(state.transcript_scroll_offset(), 0);
        assert!(state.auto_scroll);
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
        let mut state =
            TuiState {
                phase: AppPhase::Running,
                active_tool_call_id: Some("call-2".into()),
                pending_permission: Some(PermissionView::from_request(
                    PermissionRequestEvent::new("call-2", "shell__exec", "run cargo test"),
                )),
                ..Default::default()
            };
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
    fn child_feedback_respects_view_ownership_and_labels_errors() {
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

        for (kind, message, expected_kind) in [
            (NoticeKind::Info, "child info", ToastKind::Info),
            (NoticeKind::Success, "child success", ToastKind::Success),
        ] {
            state.apply_child_session_event_with_agent(
                "child-session",
                Some("explorer"),
                None,
                SessionEvent::Notice(NoticeEvent::new(message, kind)),
            );
            assert_eq!(
                state
                    .toast()
                    .map(|toast| (toast.message.as_str(), toast.kind)),
                Some((message, expected_kind))
            );
        }

        for kind in [NoticeKind::Info, NoticeKind::Success] {
            state.show_toast("keep me", ToastKind::Success);
            state.apply_child_session_event_with_agent(
                "background-child",
                Some("fixer"),
                None,
                SessionEvent::Notice(NoticeEvent::new("background notice", kind)),
            );
            assert_eq!(
                state
                    .toast()
                    .map(|toast| (toast.message.as_str(), toast.kind)),
                Some(("keep me", ToastKind::Success))
            );
        }

        state.apply_child_session_event_with_agent(
            "background-child",
            None,
            None,
            SessionEvent::Notice(NoticeEvent::new(
                "child recoverable error",
                NoticeKind::RecoverableError,
            )),
        );
        assert_eq!(
            state
                .toast()
                .map(|toast| (toast.message.as_str(), toast.kind)),
            Some((
                "background-child: child recoverable error",
                ToastKind::Error
            ))
        );

        state.apply_child_session_event_with_agent(
            "background-child",
            Some("explorer"),
            None,
            SessionEvent::RetryScheduled(RetryLifecycleEvent {
                attempt: 2,
                max_attempts: 3,
                delay_secs: 4,
                error: "network request failed".into(),
            }),
        );
        assert_eq!(
            state
                .toast()
                .map(|toast| (toast.message.as_str(), toast.kind)),
            Some((
                "explorer · background-child: network request failed · Retrying in 4s · attempt 2 of 3",
                ToastKind::Error
            ))
        );
        assert_eq!(
            state
                .child_timeline_cache
                .get("background-child")
                .and_then(|child| child.retry.as_ref())
                .map(|retry| (retry.attempt, retry.max_attempts, retry.delay_secs)),
            Some((2, 3, 4))
        );

        state.apply_child_session_event_with_agent(
            "background-child-session-with-long-id",
            Some("fixer"),
            None,
            SessionEvent::ProcessIssue(ProcessIssueEvent::new("child process issue")),
        );
        let toast = state.toast().expect("background child issue toast");
        assert_eq!(toast.kind, ToastKind::Error);
        assert!(
            toast.message.starts_with("fixer · background-chil…:"),
            "{}",
            toast.message
        );
        assert!(
            toast.message.ends_with("child process issue"),
            "{}",
            toast.message
        );
        assert_eq!(toast.ticks_remaining(), ToastState::DEFAULT_TICKS);
    }

    #[test]
    fn parent_notice_still_uses_plain_message() {
        let mut state = TuiState::default();
        state.apply_event(SessionEvent::Notice(NoticeEvent::info("parent info")));
        assert_eq!(
            state
                .toast()
                .map(|toast| (toast.message.as_str(), toast.kind)),
            Some(("parent info", ToastKind::Info))
        );
    }

    #[test]
    fn terminal_events_seal_active_reasoning() {
        for terminal in [
            SessionEvent::Interrupted,
            SessionEvent::Done,
            SessionEvent::Error(crate::tui::events::ErrorEvent::new("failed")),
        ] {
            let start = std::time::Instant::now() - std::time::Duration::from_millis(250);
            let mut state = TuiState::default();
            state.apply_event(SessionEvent::ReasoningDelta(
                crate::tui::events::ReasoningDeltaEvent::at("reasoning-1", "working", start),
            ));
            state.apply_event(terminal);

            assert!(matches!(
                state.timeline.items().first(),
                Some(crate::tui::timeline::TimelineItem::Reasoning(reasoning))
                    if !reasoning.streaming
                        && reasoning.started_at.is_none()
                        && reasoning.duration_ms.is_some()
            ));
            let revision = state.timeline.item_revisions()[0];
            state.apply_event(SessionEvent::Tick);
            assert_eq!(state.timeline.item_revisions()[0], revision);
        }
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
