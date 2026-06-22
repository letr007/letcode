use crate::tui::state::{AppPhase, TuiState};

use super::InterruptRequest;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct ActiveTurnState {
    pub runner_turn_active: bool,
    pub queued_prompt_handoff_active: bool,
    pub runtime_pending_permission: bool,
    pub projected_pending_permission: bool,
    pub phase: AppPhase,
}

pub(crate) fn has_active_or_pending_runner_turn(state: ActiveTurnState) -> bool {
    state.runner_turn_active
        || state.queued_prompt_handoff_active
        || state.runtime_pending_permission
        || state.projected_pending_permission
        || matches!(
            state.phase,
            AppPhase::Running | AppPhase::WaitingForPermission
        )
}

pub(crate) fn active_turn_state(
    state: &TuiState,
    runner_turn_active: bool,
    queued_prompt_handoff_active: bool,
    runtime_pending_permission: bool,
) -> ActiveTurnState {
    ActiveTurnState {
        runner_turn_active,
        queued_prompt_handoff_active,
        runtime_pending_permission,
        projected_pending_permission: state.pending_permission.is_some(),
        phase: state.phase,
    }
}

pub(crate) fn build_interrupt_request(
    parent_tool_calls: Vec<(String, String)>,
    visible_child_session_id: Option<String>,
    child_view_has_live_stream: bool,
) -> InterruptRequest {
    InterruptRequest {
        parent_tool_calls,
        visible_child_session_id: visible_child_session_id.filter(|_| child_view_has_live_stream),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tui::state::AppPhase;

    #[test]
    fn active_turn_detection_considers_runtime_and_projection_inputs() {
        let idle = ActiveTurnState {
            runner_turn_active: false,
            queued_prompt_handoff_active: false,
            runtime_pending_permission: false,
            projected_pending_permission: false,
            phase: AppPhase::Idle,
        };
        assert!(!has_active_or_pending_runner_turn(idle));

        assert!(has_active_or_pending_runner_turn(ActiveTurnState {
            runner_turn_active: true,
            ..idle
        }));
        assert!(has_active_or_pending_runner_turn(ActiveTurnState {
            queued_prompt_handoff_active: true,
            ..idle
        }));
        assert!(has_active_or_pending_runner_turn(ActiveTurnState {
            runtime_pending_permission: true,
            ..idle
        }));
        assert!(has_active_or_pending_runner_turn(ActiveTurnState {
            projected_pending_permission: true,
            ..idle
        }));
        assert!(has_active_or_pending_runner_turn(ActiveTurnState {
            phase: AppPhase::Running,
            ..idle
        }));
        assert!(has_active_or_pending_runner_turn(ActiveTurnState {
            phase: AppPhase::WaitingForPermission,
            ..idle
        }));
    }

    #[test]
    fn interrupt_request_only_keeps_visible_child_when_live_streaming() {
        let request = build_interrupt_request(
            vec![("parent-call".into(), "shell__exec".into())],
            Some("child-session".into()),
            false,
        );
        assert_eq!(request.parent_tool_calls.len(), 1);
        assert_eq!(request.visible_child_session_id, None);

        let live_request = build_interrupt_request(
            vec![("parent-call".into(), "shell__exec".into())],
            Some("child-session".into()),
            true,
        );
        assert_eq!(
            live_request.visible_child_session_id.as_deref(),
            Some("child-session")
        );
    }
}
