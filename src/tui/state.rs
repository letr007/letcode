use super::components::transcript::TranscriptRenderCache;
use super::events::{
    AppEvent, AutoContinueChangedEvent, PermissionDecision, PermissionRequestEvent,
    TodoSnapshotEvent, TokenUsageEvent, ToolOutcome, UserMessageEvent,
};
use super::measure;
use super::slash;
use super::timeline::{PermissionView, Timeline, TodoView};
use crate::agent::{AutoContinueState, ConversationMessage};
use crate::transcript::{
    TranscriptRecord, restore_latest_auto_continue_state, restore_latest_todo_snapshot,
};

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
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ModelTokenUsage {
    pub used_tokens: u64,
    pub context_window_tokens: u64,
}

impl Default for FooterStatus {
    fn default() -> Self {
        Self {
            summary: "Ready".into(),
            detail: Some("Ctrl-C or q to quit once keybindings are wired".into()),
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
    SessionPicker,
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
    pub timeline: Timeline,
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
    pub active_tool_call_id: Option<String>,
    pub latest_auto_continue: AutoContinueState,
    pub latest_todo: Option<TodoView>,
    pub footer_status: FooterStatus,
    pub transcript_scroll: u16,
    pub auto_scroll: bool,
    pub transcript_render_cache: TranscriptRenderCache,
    last_transcript_total_rows: Option<usize>,
    pub status_spinner_frame: usize,
    pub quit_requested: bool,
}

impl Default for TuiState {
    fn default() -> Self {
        Self {
            input_buffer: String::new(),
            timeline: Timeline::default(),
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
            active_tool_call_id: None,
            latest_auto_continue: AutoContinueState::default(),
            latest_todo: None,
            footer_status: FooterStatus::default(),
            transcript_scroll: 0,
            auto_scroll: true,
            transcript_render_cache: TranscriptRenderCache::default(),
            last_transcript_total_rows: None,
            status_spinner_frame: 0,
            quit_requested: false,
        }
    }
}

impl TuiState {
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
            });
    }

    pub fn set_reasoning_effort_label(&mut self, label: Option<String>) {
        self.reasoning_effort_label = label;
    }

    pub fn set_token_usage(&mut self, usage: ModelTokenUsage) {
        self.model_token_usage = Some(usage);
    }

    pub fn dialog(&self) -> Option<&DialogState> {
        self.dialog.as_ref()
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
        self.sync_input_phase();
        self.sync_slash_panel();
    }

    pub fn clear_input(&mut self) {
        self.input_buffer.clear();
        self.sync_input_phase();
        self.sync_slash_panel();
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

        self.phase = if self.input_buffer.is_empty() {
            AppPhase::Idle
        } else {
            AppPhase::Editing
        };
    }

    pub fn transcript_scroll_offset(&self) -> u16 {
        self.transcript_scroll
    }

    pub fn slash_panel_is_open(&self) -> bool {
        self.dialog.is_none()
            && self.pending_permission.is_none()
            && !self.slash_panel_dismissed
            && slash::slash_query(&self.input_buffer).is_some()
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

        let Some(query) = slash::slash_query(&self.input_buffer) else {
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

    pub fn set_provider_label(&mut self, label: impl Into<String>) {
        self.provider_label = label.into();
    }

    pub fn set_footer(&mut self, summary: impl Into<String>, detail: Option<String>) {
        self.footer_status = FooterStatus {
            summary: summary.into(),
            detail,
        };
    }

    pub fn replace_session_timeline(&mut self, messages: Vec<ConversationMessage>) {
        self.timeline = Timeline::from_conversation(messages);
        self.latest_auto_continue = AutoContinueState::default();
        self.latest_todo = None;
        self.reset_after_session_timeline_replace();
    }

    pub fn replace_session_timeline_from_records(&mut self, records: &[TranscriptRecord]) {
        self.timeline = Timeline::from_transcript_records(records);
        self.latest_auto_continue = restore_latest_auto_continue_state(records).unwrap_or_default();
        self.latest_todo = restore_latest_todo_snapshot(records).map(|items| TodoView {
            items,
            auto_continue: self.latest_auto_continue.clone(),
        });
        self.reset_after_session_timeline_replace();
    }

    fn reset_after_session_timeline_replace(&mut self) {
        self.pending_permission = None;
        self.active_tool_call_id = None;
        self.phase = AppPhase::Completed;
        self.model_token_usage = None;
        self.close_dialog();
        self.reset_slash_panel();
        self.scroll_transcript_to_bottom();
        self.transcript_render_cache.clear();
        self.last_transcript_total_rows = None;
    }

    pub fn apply_event(&mut self, event: AppEvent) {
        match event {
            AppEvent::Tick => {
                self.status_spinner_frame = self.status_spinner_frame.wrapping_add(1);
            }
            AppEvent::UserMessage(message) => self.on_user_message(message),
            AppEvent::ReasoningDelta(reasoning) => {
                self.phase = AppPhase::Running;
                self.timeline.push_reasoning_delta(reasoning);
                self.footer_status = FooterStatus::streaming();
            }
            AppEvent::ReasoningDone(reasoning) => {
                self.timeline
                    .finalize_reasoning(&reasoning.item_id, &reasoning.text);
            }
            AppEvent::AssistantDelta(delta) => {
                self.phase = AppPhase::Running;
                self.timeline.push_assistant_delta(delta);
                self.footer_status = FooterStatus::streaming();
            }
            AppEvent::AssistantDone { message_id } => {
                self.timeline
                    .finalize_assistant_message(message_id.as_deref());
                self.footer_status = FooterStatus::ready_for_next_prompt();
            }
            AppEvent::TokenUsage(usage) => self.set_token_usage(ModelTokenUsage::from(usage)),
            AppEvent::ToolStarted(tool) => {
                self.active_tool_call_id = Some(tool.call_id.clone());
                self.phase = AppPhase::Running;
                self.footer_status = FooterStatus::running_tool(&tool.name, &tool.summary);
                self.timeline.push_tool_started(tool);
            }
            AppEvent::ToolFinished(tool) => {
                if self.active_tool_call_id.as_deref() == Some(tool.call_id.as_str()) {
                    self.active_tool_call_id = None;
                }
                self.footer_status = match tool.outcome {
                    ToolOutcome::Success => FooterStatus::tool_finished(&tool.name, true),
                    ToolOutcome::Failure => FooterStatus::tool_finished(&tool.name, false),
                };
                self.timeline.push_tool_finished(tool);
            }
            AppEvent::TodoSnapshot(todo) => self.on_todo_snapshot(todo),
            AppEvent::AutoContinueChanged(event) => self.on_auto_continue_changed(event),
            AppEvent::PermissionRequested(request) => self.on_permission_requested(request),
            AppEvent::PermissionResolved(resolution) => {
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
                self.timeline.resolve_permission(resolution);
            }
            AppEvent::Interrupted => {
                self.phase = AppPhase::Completed;
                self.active_tool_call_id = None;
                self.pending_permission = None;
                self.footer_status = FooterStatus {
                    summary: "Interrupted".into(),
                    detail: Some("Current assistant turn stopped".into()),
                };
                self.timeline.push_notice("Interrupted by user");
            }
            AppEvent::Error(error) => {
                self.phase = AppPhase::Error;
                self.active_tool_call_id = None;
                self.footer_status = FooterStatus::error(&error.message);
                self.timeline.push_error(error);
            }
            AppEvent::Done => {
                self.phase = AppPhase::Completed;
                self.active_tool_call_id = None;
                self.footer_status = FooterStatus::ready_for_next_prompt();
            }
            AppEvent::Quit => {
                self.phase = AppPhase::Quitting;
                self.quit_requested = true;
                self.footer_status = FooterStatus {
                    summary: "Exiting".into(),
                    detail: None,
                };
            }
        }
    }

    fn on_user_message(&mut self, message: UserMessageEvent) {
        self.timeline.push_user_message(message);
        self.latest_auto_continue = AutoContinueState::default();
        self.latest_todo = None;
        self.phase = AppPhase::Running;
        self.active_tool_call_id = None;
        self.pending_permission = None;
        self.reset_slash_panel();
        self.footer_status = FooterStatus {
            summary: "Waiting for assistant".into(),
            detail: Some("Streaming output will appear in the timeline".into()),
        };
    }

    fn on_permission_requested(&mut self, request: PermissionRequestEvent) {
        self.phase = AppPhase::WaitingForPermission;
        self.active_tool_call_id = Some(request.call_id.clone());
        self.pending_permission = Some(PermissionView::from_request(request.clone()));
        self.slash_panel_dismissed = false;
        self.footer_status = FooterStatus {
            summary: format!("Permission required for {}", request.tool_name),
            detail: Some(request.summary.clone()),
        };
        self.timeline.push_permission_request(request);
    }

    fn on_todo_snapshot(&mut self, event: TodoSnapshotEvent) {
        let auto_continue = self.latest_auto_continue.clone();
        let todo_view = TodoView {
            items: event.items.clone(),
            auto_continue: auto_continue.clone(),
        };
        self.latest_todo = Some(todo_view);
        self.timeline.push_todo_snapshot(event);
        self.timeline
            .apply_auto_continue_changed(AutoContinueChangedEvent::new(auto_continue));
    }

    fn on_auto_continue_changed(&mut self, event: AutoContinueChangedEvent) {
        self.latest_auto_continue = event.state.clone();
        if let Some(todo) = self.latest_todo.as_mut() {
            todo.auto_continue = event.state.clone();
            self.timeline.apply_auto_continue_changed(event);
        }
    }
}

impl From<TokenUsageEvent> for ModelTokenUsage {
    fn from(event: TokenUsageEvent) -> Self {
        Self {
            used_tokens: event.used_tokens,
            context_window_tokens: event.context_window_tokens,
        }
    }
}

trait FooterStatusExt {
    fn streaming() -> Self;
    fn ready_for_next_prompt() -> Self;
    fn running_tool(tool_name: &str, summary: &str) -> Self;
    fn tool_finished(tool_name: &str, success: bool) -> Self;
    fn permission_resolved(approved: bool) -> Self;
    fn error(message: &str) -> Self;
}

impl FooterStatusExt for FooterStatus {
    fn streaming() -> Self {
        Self {
            summary: "Streaming response".into(),
            detail: Some("Assistant output is still arriving".into()),
        }
    }

    fn ready_for_next_prompt() -> Self {
        Self {
            summary: "Ready".into(),
            detail: Some("Enter a prompt when the runtime loop is wired".into()),
        }
    }

    fn running_tool(tool_name: &str, summary: &str) -> Self {
        Self {
            summary: format!("Running tool: {tool_name}"),
            detail: Some(summary.to_string()),
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
        }
    }

    fn error(message: &str) -> Self {
        Self {
            summary: "Error".into(),
            detail: Some(message.to_string()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::{AutoContinueState, TodoItem, TodoStatus};
    use crate::tui::events::{
        AppEvent, AutoContinueChangedEvent, PermissionResolutionEvent, TodoSnapshotEvent,
    };

    #[test]
    fn permission_resolved_clears_active_tool_and_pending_permission() {
        let mut state = TuiState::default();
        let request = PermissionRequestEvent::new("call-1", "shell__exec", "run ls");

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
}
