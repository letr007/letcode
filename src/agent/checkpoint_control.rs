use crate::config::LogicalCheckpointConfig;
use std::sync::{Arc, Mutex};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LogicalCheckpointRequestOutcome {
    Queued,
    AlreadyQueued,
    Disabled,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum LogicalCheckpointRequestState {
    Idle,
    Pending(LogicalCheckpointRequestOwner),
    InFlight(LogicalCheckpointRequestOwner),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum LogicalCheckpointRequestOwner {
    Manual,
    Automatic { boundary_id: u64 },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct LogicalCheckpointLease {
    pub(super) run_id: u64,
    pub(super) request_id: u64,
    pub(super) ownership: LogicalCheckpointRequestOwner,
}

#[derive(Clone)]
pub struct LogicalCheckpointControl {
    pub(super) state: Arc<Mutex<LogicalCheckpointControlState>>,
}

#[derive(Debug)]
pub(super) struct LogicalCheckpointControlState {
    pub(super) enabled: bool,
    pub(super) request: LogicalCheckpointRequestState,
    pub(super) request_run_id: Option<u64>,
    pub(super) active_run_id: Option<u64>,
    pub(super) next_run_id: u64,
    pub(super) next_request_id: u64,
    pub(super) request_id: Option<u64>,
    pub(super) automatic_enabled: bool,
}

impl LogicalCheckpointControl {
    pub(super) fn disabled() -> Self {
        Self {
            state: Arc::new(Mutex::new(LogicalCheckpointControlState {
                enabled: false,
                request: LogicalCheckpointRequestState::Idle,
                request_run_id: None,
                active_run_id: None,
                next_run_id: 0,
                next_request_id: 0,
                request_id: None,
                automatic_enabled: false,
            })),
        }
    }

    pub fn request(&self) -> LogicalCheckpointRequestOutcome {
        let mut state = self
            .state
            .lock()
            .expect("logical checkpoint control poisoned");
        if !state.enabled {
            return LogicalCheckpointRequestOutcome::Disabled;
        }
        match state.request {
            LogicalCheckpointRequestState::Idle => {
                state.request =
                    LogicalCheckpointRequestState::Pending(LogicalCheckpointRequestOwner::Manual);
                state.request_run_id = state.active_run_id;
                state.request_id = Some(Self::next_request_id(&mut state));
                LogicalCheckpointRequestOutcome::Queued
            }
            LogicalCheckpointRequestState::Pending(LogicalCheckpointRequestOwner::Automatic {
                ..
            }) => {
                state.request =
                    LogicalCheckpointRequestState::Pending(LogicalCheckpointRequestOwner::Manual);
                state.request_id = Some(Self::next_request_id(&mut state));
                LogicalCheckpointRequestOutcome::Queued
            }
            LogicalCheckpointRequestState::Pending(_)
            | LogicalCheckpointRequestState::InFlight(_) => {
                LogicalCheckpointRequestOutcome::AlreadyQueued
            }
        }
    }

    pub(super) fn request_automatic(&self, boundary_id: u64) -> LogicalCheckpointRequestOutcome {
        let mut state = self
            .state
            .lock()
            .expect("logical checkpoint control poisoned");
        if !state.enabled || !state.automatic_enabled || state.active_run_id.is_none() {
            return LogicalCheckpointRequestOutcome::Disabled;
        }
        match state.request {
            LogicalCheckpointRequestState::Idle => {
                state.request = LogicalCheckpointRequestState::Pending(
                    LogicalCheckpointRequestOwner::Automatic { boundary_id },
                );
                state.request_run_id = state.active_run_id;
                state.request_id = Some(Self::next_request_id(&mut state));
                LogicalCheckpointRequestOutcome::Queued
            }
            LogicalCheckpointRequestState::Pending(_)
            | LogicalCheckpointRequestState::InFlight(_) => {
                LogicalCheckpointRequestOutcome::AlreadyQueued
            }
        }
    }

    fn next_request_id(state: &mut LogicalCheckpointControlState) -> u64 {
        state.next_request_id = state
            .next_request_id
            .checked_add(1)
            .expect("logical checkpoint request id overflow");
        state.next_request_id
    }

    pub(super) fn take_pending(&self) -> Option<LogicalCheckpointLease> {
        let mut state = self
            .state
            .lock()
            .expect("logical checkpoint control poisoned");
        if !state.enabled {
            state.request = LogicalCheckpointRequestState::Idle;
            state.request_run_id = None;
            return None;
        }
        match state.request {
            LogicalCheckpointRequestState::Pending(ownership) => {
                let run_id = state.request_run_id?;
                let request_id = state.request_id?;
                state.request = LogicalCheckpointRequestState::InFlight(ownership);
                Some(LogicalCheckpointLease {
                    run_id,
                    request_id,
                    ownership,
                })
            }
            LogicalCheckpointRequestState::Idle | LogicalCheckpointRequestState::InFlight(_) => {
                None
            }
        }
    }

    pub(super) fn clear(&self) {
        let mut state = self
            .state
            .lock()
            .expect("logical checkpoint control poisoned");
        state.request = LogicalCheckpointRequestState::Idle;
        state.request_run_id = None;
        state.request_id = None;
    }

    pub(super) fn clear_lease(&self, lease: LogicalCheckpointLease) {
        let mut state = self
            .state
            .lock()
            .expect("logical checkpoint control poisoned");
        if state.request_run_id == Some(lease.run_id) && state.request_id == Some(lease.request_id)
        {
            state.request = LogicalCheckpointRequestState::Idle;
            state.request_run_id = None;
            state.request_id = None;
        }
    }

    pub(super) fn begin_run(&self) -> LogicalCheckpointRunGuard {
        let run_id = {
            let mut state = self
                .state
                .lock()
                .expect("logical checkpoint control poisoned");
            state.next_run_id = state
                .next_run_id
                .checked_add(1)
                .expect("logical checkpoint run id overflow");
            let run_id = state.next_run_id;
            state.active_run_id = Some(run_id);
            // A request made immediately before the stream began belongs to this
            // run; later requests are tagged by request().
            if matches!(state.request, LogicalCheckpointRequestState::Pending(_))
                && state.request_run_id.is_none()
            {
                state.request_run_id = Some(run_id);
            }
            run_id
        };
        LogicalCheckpointRunGuard {
            control: self.clone(),
            run_id,
        }
    }

    pub(super) fn set_config(&self, config: LogicalCheckpointConfig) {
        let mut state = self
            .state
            .lock()
            .expect("logical checkpoint control poisoned");
        state.enabled = config.enabled;
        state.automatic_enabled = config.enabled && config.automatic;
        if !config.enabled {
            state.request = LogicalCheckpointRequestState::Idle;
            state.request_run_id = None;
            state.request_id = None;
        }
    }

    pub(super) fn set_enabled(&self, enabled: bool) {
        self.set_config(LogicalCheckpointConfig {
            enabled,
            ..LogicalCheckpointConfig::default()
        });
    }

    #[cfg(test)]
    pub(super) fn disabled_for_test() -> Self {
        Self::disabled()
    }
}

/// Owns a single streamed turn's checkpoint request.  It is deliberately held
/// across all awaits in a protocol stream, so cancellation cannot strand a
/// pending or in-flight request for the next turn.
pub(crate) struct LogicalCheckpointRunGuard {
    control: LogicalCheckpointControl,
    run_id: u64,
}

impl Drop for LogicalCheckpointRunGuard {
    fn drop(&mut self) {
        let mut state = self
            .control
            .state
            .lock()
            .expect("logical checkpoint control poisoned");
        if state.active_run_id == Some(self.run_id) {
            if state.request_run_id == Some(self.run_id) {
                state.request = LogicalCheckpointRequestState::Idle;
                state.request_run_id = None;
                state.request_id = None;
            }
            state.active_run_id = None;
        }
    }
}
