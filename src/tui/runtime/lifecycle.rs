use crate::tui::state::{AppPhase, TuiState};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct ActiveTurnState {
    pub session_turn_active: bool,
    pub queued_prompt_handoff_active: bool,
    pub runtime_pending_permission: bool,
    pub projected_pending_permission: bool,
    pub phase: AppPhase,
}

pub(crate) fn has_active_or_pending_session_turn(state: ActiveTurnState) -> bool {
    state.session_turn_active
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
    session_turn_active: bool,
    queued_prompt_handoff_active: bool,
    runtime_pending_permission: bool,
) -> ActiveTurnState {
    ActiveTurnState {
        session_turn_active,
        queued_prompt_handoff_active,
        runtime_pending_permission,
        projected_pending_permission: state.pending_permission.is_some(),
        phase: state.phase,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tui::state::AppPhase;

    #[test]
    fn active_turn_detection_considers_runtime_and_projection_inputs() {
        let idle = ActiveTurnState {
            session_turn_active: false,
            queued_prompt_handoff_active: false,
            runtime_pending_permission: false,
            projected_pending_permission: false,
            phase: AppPhase::Idle,
        };
        assert!(!has_active_or_pending_session_turn(idle));

        assert!(has_active_or_pending_session_turn(ActiveTurnState {
            session_turn_active: true,
            ..idle
        }));
        assert!(has_active_or_pending_session_turn(ActiveTurnState {
            queued_prompt_handoff_active: true,
            ..idle
        }));
        assert!(has_active_or_pending_session_turn(ActiveTurnState {
            runtime_pending_permission: true,
            ..idle
        }));
        assert!(has_active_or_pending_session_turn(ActiveTurnState {
            projected_pending_permission: true,
            ..idle
        }));
        assert!(has_active_or_pending_session_turn(ActiveTurnState {
            phase: AppPhase::Running,
            ..idle
        }));
        assert!(has_active_or_pending_session_turn(ActiveTurnState {
            phase: AppPhase::WaitingForPermission,
            ..idle
        }));
    }
}
