use super::components::transcript::TranscriptRenderCache;
use super::events::{
    AppEvent, AutoContinueChangedEvent, PermissionDecision, PermissionRequestEvent,
    PermissionResolutionEvent, TokenUsageEvent, ToolOutcome, UserMessageEvent,
};
use super::measure;
use super::slash;
use super::timeline::{PermissionView, Timeline, TodoView};
use crate::agent::{AutoContinueState, ConversationMessage};
use crate::transcript::{
    TranscriptEvent, TranscriptRecord, restore_latest_auto_continue_state,
    restore_latest_todo_snapshot,
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
    pub input_tokens: u64,
    pub output_tokens: u64,
    pub cached_tokens: u64,
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
    ReasoningPicker,
    SessionPicker,
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
    pub timeline: Timeline,
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
    pub quit_requested: bool,
    ignore_late_tool_events: bool,
}

impl Default for TuiState {
    fn default() -> Self {
        Self {
            input_buffer: String::new(),
            input_cursor: 0,
            timeline: Timeline::default(),
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
            quit_requested: false,
            ignore_late_tool_events: false,
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
            self.transcript_render_cache.clear();
            self.last_transcript_total_rows = None;
        }
    }

    pub fn set_transcript_scrollbar_visible(&mut self, visible: bool) {
        if self.transcript_scrollbar_visible != visible {
            self.transcript_scrollbar_visible = visible;
            self.transcript_render_cache.clear();
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

    pub fn mark_session_active(&mut self) {
        self.active_session = true;
    }

    pub fn push_queued_user_message_preview(&mut self, content: impl Into<String>) {
        self.active_session = true;
        self.timeline
            .push_user_message(UserMessageEvent::queued(content));
        self.reset_slash_panel();
    }

    pub fn activate_queued_user_message(&mut self, content: &str) -> bool {
        if !self.timeline.activate_first_queued_user_message(content) {
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
        self.sync_input_phase();
        self.sync_slash_panel();
    }

    pub fn clear_input(&mut self) {
        self.input_buffer.clear();
        self.input_cursor = 0;
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
        self.child_timeline = None;
        self.latest_auto_continue = AutoContinueState::default();
        self.latest_todo = None;
        self.transcript_view = TranscriptViewState::Parent;
        self.reset_after_session_timeline_replace();
    }

    pub fn replace_session_timeline_from_records(&mut self, records: &[TranscriptRecord]) {
        self.active_session = true;
        self.timeline = Timeline::from_transcript_records(records);
        self.child_timeline = None;
        self.latest_auto_continue = restore_latest_auto_continue_state(records).unwrap_or_default();
        self.latest_todo = restore_latest_todo_snapshot(records).map(|items| TodoView {
            items,
            auto_continue: self.latest_auto_continue.clone(),
        });
        self.transcript_view = TranscriptViewState::Parent;
        self.reset_after_session_timeline_replace();
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
        self.active_session = true;
        self.input_buffer.clear();
        self.input_cursor = 0;
        self.sync_input_phase();
        self.close_dialog();
        self.reset_slash_panel();
        self.replace_child_timeline_state(records);
        self.transcript_view = TranscriptViewState::Child {
            parent_session_id: parent_session_id.into(),
            child_session_id: child_session_id.into(),
            agent_name: agent_name.into(),
            index,
            total,
        };
        self.scroll_transcript_to_bottom();
        self.transcript_render_cache.clear();
        self.last_transcript_total_rows = None;
    }

    pub fn refresh_child_timeline_from_records(&mut self, records: &[TranscriptRecord]) {
        if !self.transcript_view.is_child() {
            return;
        }

        self.replace_child_timeline_state(records);
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
        self.transcript_render_cache.clear();
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
        self.transcript_render_cache.clear();
        self.last_transcript_total_rows = None;
    }

    fn replace_child_timeline_state(&mut self, records: &[TranscriptRecord]) {
        self.child_timeline = Some(ChildTranscriptState {
            timeline: Timeline::from_transcript_records(records),
            model: child_transcript_model(records),
            record_count: records.len(),
            live_streaming: false,
        });
        self.ignore_late_tool_events = false;
        self.transcript_render_cache.clear();
        self.last_transcript_total_rows = None;
        self.reproject_pending_permission();
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
        };
    }

    pub fn apply_child_app_event(&mut self, child_session_id: &str, event: AppEvent) {
        let viewing_child = matches!(
            &self.transcript_view,
            TranscriptViewState::Child {
                child_session_id: active_child_session_id,
                ..
            } if active_child_session_id == child_session_id
        );

        match event {
            AppEvent::PermissionRequested(request) => {
                self.apply_permission_requested_projection(&request);
                if viewing_child && let Some(child_timeline) = self.child_timeline.as_mut() {
                    child_timeline.timeline.push_permission_request(request);
                    child_timeline.live_streaming = true;
                    self.transcript_render_cache.clear();
                    self.last_transcript_total_rows = None;
                }
            }
            AppEvent::PermissionResolved(resolution) => {
                self.apply_permission_resolved_projection(&resolution);
                if viewing_child && let Some(child_timeline) = self.child_timeline.as_mut() {
                    child_timeline.timeline.resolve_permission(resolution);
                    child_timeline.live_streaming = true;
                    self.transcript_render_cache.clear();
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
                        timeline: &mut child_timeline.timeline,
                        live_streaming: None,
                        accepts_tool_events: true,
                    }
                    .with_live_streaming(&mut child_timeline.live_streaming)
                    .with_tool_event_acceptance(accepts_tool_events),
                    event,
                );

                self.transcript_render_cache.clear();
                self.last_transcript_total_rows = None;
            }
            _ => {}
        }
    }

    pub fn apply_event(&mut self, event: AppEvent) {
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
        };
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
        AppEvent::Interrupted => {
            *projection.phase = AppPhase::Completed;
            *projection.active_tool_call_id = None;
            *projection.pending_permission = None;
            *projection.ignore_late_tool_events = true;
            projection.timeline.cancel_active_tools();
            *projection.footer_status = FooterStatus {
                summary: "Interrupted".into(),
                detail: Some("Current assistant turn stopped".into()),
            };
            projection.timeline.push_notice("Interrupted by user");
        }
        AppEvent::Error(error) => {
            *projection.phase = AppPhase::Error;
            *projection.active_tool_call_id = None;
            *projection.pending_permission = None;
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
        AppEvent::Quit => {
            *projection.phase = AppPhase::Quitting;
            *projection.quit_requested = true;
            *projection.footer_status = FooterStatus {
                summary: "Exiting".into(),
                detail: None,
            };
        }
        AppEvent::PermissionRequested(_) => {}
    }

    if !terminal_event && let Some(live_streaming) = projection.live_streaming.as_deref_mut() {
        *live_streaming = true;
    }
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

    fn preparing_tool(tool_name: &str) -> Self {
        Self {
            summary: format!("Preparing tool: {tool_name}"),
            detail: Some("Tool input is still arriving".into()),
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
    use crate::transcript::{TranscriptEvent, TranscriptRecord};
    use crate::tui::events::{
        AppEvent, AutoContinueChangedEvent, PermissionResolutionEvent, TodoSnapshotEvent,
        ToolPendingEvent,
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
            event: TranscriptEvent::UserMessage {
                content: "parent prompt".into(),
            },
        }];
        let child_records = vec![TranscriptRecord {
            session_id: "child-session".into(),
            sequence: 1,
            timestamp_ms: 1,
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
    fn child_view_metadata_prefers_latest_model_change() {
        let records = vec![
            TranscriptRecord {
                session_id: "child-session".into(),
                sequence: 1,
                timestamp_ms: 0,
                event: TranscriptEvent::SessionStarted {
                    model: "gpt-5.5".into(),
                },
            },
            TranscriptRecord {
                session_id: "child-session".into(),
                sequence: 2,
                timestamp_ms: 1,
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
                event: TranscriptEvent::SessionStarted {
                    model: "gpt-test".into(),
                },
            },
            TranscriptRecord {
                session_id: "child-session".into(),
                sequence: 2,
                timestamp_ms: 1,
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
