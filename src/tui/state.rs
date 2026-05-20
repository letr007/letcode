use super::events::{
    AppEvent, PermissionDecision, PermissionRequestEvent, ToolOutcome, UserMessageEvent,
};
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
    pub phase: AppPhase,
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
            phase: AppPhase::Idle,
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
    pub fn new(model_label: impl Into<String>, permission_mode_label: impl Into<String>) -> Self {
        Self {
            model_label: model_label.into(),
            permission_mode_label: permission_mode_label.into(),
            ..Self::default()
        }
    }

    pub fn set_input(&mut self, input: impl Into<String>) {
        self.input_buffer = input.into();
        self.sync_input_phase();
    }

    pub fn clear_input(&mut self) {
        self.input_buffer.clear();
        self.sync_input_phase();
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
        self.footer_status = FooterStatus {
            summary: "Waiting for assistant".into(),
            detail: Some("Streaming output will appear in the timeline".into()),
        };
    }

    fn on_permission_requested(&mut self, request: PermissionRequestEvent) {
        self.phase = AppPhase::WaitingForPermission;
        self.active_tool_call_id = Some(request.call_id.clone());
        self.pending_permission = Some(PermissionView::from_request(request.clone()));
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
}
