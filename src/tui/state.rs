use super::events::{
    AppEvent, PermissionDecision, PermissionRequestEvent, ToolOutcome, UserMessageEvent,
};
use super::measure;
use super::slash;
use super::timeline::{PermissionView, Timeline};

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

impl Default for FooterStatus {
    fn default() -> Self {
        Self {
            summary: "Ready".into(),
            detail: Some("Ctrl-C or q to quit once keybindings are wired".into()),
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
    pub model_id: String,
    pub model_label: String,
    pub permission_mode_label: String,
    pub active_tool_call_id: Option<String>,
    pub footer_status: FooterStatus,
    pub transcript_scroll: u16,
    pub auto_scroll: bool,
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
            model_id: "pending-runtime-model".into(),
            model_label: "pending runtime model".into(),
            permission_mode_label: "default".into(),
            active_tool_call_id: None,
            footer_status: FooterStatus::default(),
            transcript_scroll: 0,
            auto_scroll: true,
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
        self.pending_permission.is_none()
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

    pub fn set_permission_mode_label(&mut self, label: impl Into<String>) {
        self.permission_mode_label = label.into();
    }

    pub fn set_footer(&mut self, summary: impl Into<String>, detail: Option<String>) {
        self.footer_status = FooterStatus {
            summary: summary.into(),
            detail,
        };
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
    use crate::tui::events::{AppEvent, PermissionResolutionEvent};

    #[test]
    fn permission_resolved_clears_active_tool_and_pending_permission() {
        let mut state = TuiState::default();
        let request = PermissionRequestEvent::new("call-1", "bash", "run ls");

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
}
