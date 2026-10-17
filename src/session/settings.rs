//! Session-owned idle settings mutations shared by TUI and line CLI.
//!
//! These helpers apply agent configuration changes and record transcript
//! provenance when the value actually changes. Frontends retain presentation
//! concerns (confirm dialogs, model catalog validation, status printing).

use std::sync::{Arc, Mutex};

use anyhow::{Result, anyhow};

use crate::agent::Agent;
use crate::config::ModelRoute;
use crate::permission::PermissionMode;
use crate::request_builder::ModelReasoningEffort;
use crate::transcript::TranscriptRecorder;

/// The result of a model apply that can disable Fast Mode before transcript
/// provenance is recorded.
#[derive(Debug)]
pub struct ModelApplyError {
    error: anyhow::Error,
    fast_mode_auto_disabled: bool,
}

impl ModelApplyError {
    pub(crate) fn new(error: anyhow::Error, fast_mode_auto_disabled: bool) -> Self {
        Self {
            error,
            fast_mode_auto_disabled,
        }
    }

    pub fn fast_mode_auto_disabled(&self) -> bool {
        self.fast_mode_auto_disabled
    }
}

impl std::fmt::Display for ModelApplyError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.error.fmt(formatter)
    }
}

impl std::error::Error for ModelApplyError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(self.error.as_ref())
    }
}

/// Apply a permission mode and record provenance when it changes.
pub fn apply_permission_mode(
    agent: &mut Agent,
    transcript: &Arc<Mutex<TranscriptRecorder>>,
    mode: PermissionMode,
) -> Result<()> {
    let previous = agent.permission_mode();
    agent.set_permission_mode(mode);
    if previous != mode {
        transcript
            .lock()
            .map_err(|_| anyhow!("transcript recorder poisoned"))?
            .record_permission_mode_changed(previous.to_string(), mode.to_string())?;
    }
    Ok(())
}

#[cfg(test)]
pub(crate) fn apply_model_route_with(
    agent: &mut Agent,
    transcript: &Arc<Mutex<TranscriptRecorder>>,
    route: ModelRoute,
    prepared_route: crate::agent::PreparedPrimaryRoute,
) -> std::result::Result<bool, ModelApplyError> {
    let previous_route = agent.route_display_name();
    let new_route = route.display_name();
    let route_changed = previous_route != new_route;
    let fast_mode_auto_disabled = agent
        .auto_disable_fast_mode_for_model(&route.model)
        .map_err(|error| ModelApplyError::new(error, false))?;

    if route_changed {
        transcript
            .lock()
            .map_err(|_| {
                ModelApplyError::new(
                    anyhow!("transcript recorder poisoned"),
                    fast_mode_auto_disabled,
                )
            })?
            .record_model_changed(previous_route, new_route)
            .map_err(|error| ModelApplyError::new(error, fast_mode_auto_disabled))?;
    }

    agent.apply_prepared_route(prepared_route);
    Ok(fast_mode_auto_disabled)
}

pub(crate) fn apply_model_route_with_authority(
    agent: &mut Agent,
    transcript: &Arc<Mutex<TranscriptRecorder>>,
    route: ModelRoute,
    resolved_route: std::sync::Arc<crate::model_runtime::ResolvedModelRoute>,
    prepared_route: crate::agent::PreparedPrimaryRoute,
) -> std::result::Result<bool, ModelApplyError> {
    let previous_route = agent.route_display_name();
    let new_route = route.display_name();
    let route_changed = previous_route != new_route;
    let fast_mode_auto_disabled = agent
        .auto_disable_fast_mode_for_model(&route.model)
        .map_err(|error| ModelApplyError::new(error, false))?;
    if route_changed {
        transcript
            .lock()
            .map_err(|_| {
                ModelApplyError::new(
                    anyhow!("transcript recorder poisoned"),
                    fast_mode_auto_disabled,
                )
            })?
            .record_model_changed(previous_route, new_route)
            .map_err(|error| ModelApplyError::new(error, fast_mode_auto_disabled))?;
    }
    agent.set_model_route_authority(route, resolved_route);
    agent.apply_prepared_route(prepared_route);
    Ok(fast_mode_auto_disabled)
}

/// Persist a primary-route selection before installing it. If the live route
/// cannot be applied afterwards, restore the previous persisted selection.
#[cfg(test)]
pub(crate) fn persist_and_apply_model_route_with(
    agent: &mut Agent,
    transcript: &Arc<Mutex<TranscriptRecorder>>,
    route: ModelRoute,
    prepared_route: crate::agent::PreparedPrimaryRoute,
    mut persist: impl FnMut(&ModelRoute) -> Result<()>,
) -> std::result::Result<bool, ModelApplyError> {
    let previous_route = agent.primary_route().cloned().ok_or_else(|| {
        ModelApplyError::new(anyhow!("current primary route is unavailable"), false)
    })?;
    let route_changed = previous_route != route;

    persist(&route).map_err(|error| {
        ModelApplyError::new(
            anyhow!(
                "failed to persist model route '{}': {error}",
                route.display_name()
            ),
            false,
        )
    })?;

    match apply_model_route_with(agent, transcript, route, prepared_route) {
        Ok(fast_mode_auto_disabled) => Ok(fast_mode_auto_disabled),
        Err(error) if !route_changed => Err(error),
        Err(error) => {
            if let Err(rollback_error) = persist(&previous_route) {
                return Err(ModelApplyError::new(
                    anyhow!(
                        "failed to apply model route after persisting it: {error}; failed to restore persisted route '{}': {rollback_error}",
                        previous_route.display_name()
                    ),
                    error.fast_mode_auto_disabled(),
                ));
            }
            Err(error)
        }
    }
}

/// Apply reasoning effort and record the session-local selection.
pub fn apply_reasoning_effort(
    agent: &mut Agent,
    transcript: &Arc<Mutex<TranscriptRecorder>>,
    effort: ModelReasoningEffort,
) -> Result<()> {
    let previous = agent.reasoning_effort();
    agent.set_reasoning_effort(effort.clone())?;
    if previous.as_ref() != Some(&effort) {
        transcript
            .lock()
            .map_err(|_| anyhow!("transcript recorder poisoned"))?
            .record_reasoning_effort_changed(agent.route_display_name(), effort)?;
    }
    Ok(())
}

#[cfg(test)]
#[path = "../../tests/internal/session/settings/integration_tests.rs"]
mod integration_tests;
