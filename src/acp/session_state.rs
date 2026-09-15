//! ACP view of the session mode and configuration options the engine owns.
//!
//! The engine is the source of truth for both: the frontend seeds this state
//! from the startup projection and the process catalog, then mirrors every
//! change the engine announces. A client therefore observes the settings the
//! engine runs with, whichever frontend changed them.

use agent_client_protocol::schema::v1::{
    ConfigOptionUpdate, CurrentModeUpdate, SessionConfigOption, SessionConfigOptionCategory,
    SessionConfigSelectOption, SessionMode, SessionModeState, SessionUpdate,
};

use crate::permission::PermissionMode;
use crate::request_builder::ModelReasoningEffort;
use crate::session::runner::{ModelCatalogEntry, ModelCatalogUpdatedEvent};

/// Configuration option ids exchanged with the client.
pub(super) const CONFIG_SESSION_MODE: &str = "mode";
pub(super) const CONFIG_MODEL: &str = "model";
pub(super) const CONFIG_REASONING_EFFORT: &str = "reasoning_effort";

/// A model route a client can select, with the reasoning levels the route
/// accepts.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelSummary {
    pub id: String,
    pub label: String,
    pub reasoning_efforts: Vec<ModelReasoningEffort>,
}

impl ModelSummary {
    pub fn new(
        id: impl Into<String>,
        label: impl Into<String>,
        reasoning_efforts: Vec<ModelReasoningEffort>,
    ) -> Self {
        Self {
            id: id.into(),
            label: label.into(),
            reasoning_efforts,
        }
    }

    fn from_catalog_entry(entry: &ModelCatalogEntry) -> Self {
        Self::new(
            entry.id.clone(),
            entry.label.clone(),
            entry
                .reasoning
                .efforts
                .iter()
                .map(|effort| parse_reasoning_effort(effort))
                .collect(),
        )
    }
}

/// Session settings the engine already holds when this frontend starts.
///
/// The engine announces catalog changes only when its configuration reloads,
/// so the process startup catalog seeds the client until then.
#[derive(Debug, Clone)]
pub struct SessionSettings {
    pub models: Vec<ModelSummary>,
    pub reasoning_effort: Option<ModelReasoningEffort>,
}

impl SessionSettings {
    pub fn new(models: Vec<ModelSummary>, reasoning_effort: Option<ModelReasoningEffort>) -> Self {
        Self {
            models,
            reasoning_effort,
        }
    }
}

/// Mirror of the engine's session mode and configuration options.
#[derive(Debug, Clone)]
pub(super) struct SessionState {
    mode: PermissionMode,
    model: String,
    models: Vec<ModelSummary>,
    reasoning_effort: Option<ModelReasoningEffort>,
}

impl SessionState {
    pub(super) fn new(mode_label: &str, model_id: &str, settings: &SessionSettings) -> Self {
        Self {
            // The engine labels its mode with `PermissionMode`'s own vocabulary.
            mode: PermissionMode::parse(mode_label).unwrap_or_default(),
            model: model_id.to_string(),
            models: settings.models.clone(),
            reasoning_effort: settings.reasoning_effort.clone(),
        }
    }

    /// The mode state a `session/new` response reports.
    pub(super) fn modes(&self) -> SessionModeState {
        SessionModeState::new(self.mode.as_str(), session_modes())
    }

    /// The complete configuration state a client receives.
    pub(super) fn config_options(&self) -> Vec<SessionConfigOption> {
        let mut options = vec![
            SessionConfigOption::select(
                CONFIG_SESSION_MODE,
                "Session mode",
                self.mode.as_str().to_string(),
                mode_options(),
            )
            .category(SessionConfigOptionCategory::Mode),
            SessionConfigOption::select(
                CONFIG_MODEL,
                "Model",
                self.model.clone(),
                self.model_options(),
            )
            .category(SessionConfigOptionCategory::Model),
        ];
        if let Some(reasoning) = self.reasoning_option() {
            options.push(reasoning);
        }
        options
    }

    pub(super) fn mode_update(&self) -> SessionUpdate {
        SessionUpdate::CurrentModeUpdate(CurrentModeUpdate::new(self.mode.as_str()))
    }

    pub(super) fn config_option_update(&self) -> SessionUpdate {
        SessionUpdate::ConfigOptionUpdate(ConfigOptionUpdate::new(self.config_options()))
    }

    /// Adopts the mode the engine announced. An announcement outside the
    /// engine's own mode vocabulary names no mode this frontend can report.
    pub(super) fn adopt_mode(&mut self, mode: &str) -> bool {
        let Some(mode) = PermissionMode::parse(mode) else {
            tracing::warn!(mode, "ignoring an unknown engine permission mode");
            return false;
        };
        self.mode = mode;
        true
    }

    /// Adopts the route the engine reported, and reports whether it changed.
    ///
    /// A route without selectable reasoning levels leaves the engine without a
    /// reasoning level, and the engine reports that change through the route
    /// rather than through an effort event.
    pub(super) fn adopt_model(&mut self, model_id: &str) -> bool {
        let changed = self.model != model_id;
        self.model = model_id.to_string();
        if self.active_model_efforts().is_empty() {
            self.reasoning_effort = None;
        }
        changed
    }

    pub(super) fn adopt_reasoning_effort(&mut self, effort: ModelReasoningEffort) {
        self.reasoning_effort = Some(effort);
    }

    pub(super) fn adopt_catalog(&mut self, catalog: &ModelCatalogUpdatedEvent) {
        self.models = catalog
            .models
            .iter()
            .map(ModelSummary::from_catalog_entry)
            .collect();
    }

    /// The reasoning levels the active route accepts. A route the catalog does
    /// not list reports none: what it accepts is unknown.
    fn active_model_efforts(&self) -> &[ModelReasoningEffort] {
        self.models
            .iter()
            .find(|model| model.id == self.model)
            .map_or(&[][..], |model| model.reasoning_efforts.as_slice())
    }

    fn model_options(&self) -> Vec<SessionConfigSelectOption> {
        select_options(
            &self.model,
            self.models
                .iter()
                .map(|model| (model.id.clone(), model.label.clone())),
        )
    }

    /// The reasoning option is offered while the engine holds a reasoning level
    /// and the active route can select one: otherwise there is nothing for a
    /// client to choose between.
    fn reasoning_option(&self) -> Option<SessionConfigOption> {
        let current = self.reasoning_effort.as_ref()?;
        let efforts = self.active_model_efforts();
        if efforts.is_empty() {
            return None;
        }
        let options = select_options(
            current.as_str(),
            efforts
                .iter()
                .map(|effort| (effort.as_str().to_string(), effort.as_str().to_string())),
        );
        Some(
            SessionConfigOption::select(
                CONFIG_REASONING_EFFORT,
                "Reasoning effort",
                current.as_str().to_string(),
                options,
            )
            .category(SessionConfigOptionCategory::ThoughtLevel),
        )
    }
}

/// The engine's permission modes are also the session mode vocabulary.
const PERMISSION_MODES: [PermissionMode; 4] = [
    PermissionMode::Safe,
    PermissionMode::Default,
    PermissionMode::Auto,
    PermissionMode::Yolo,
];

/// The modes a client can switch to.
fn session_modes() -> Vec<SessionMode> {
    PERMISSION_MODES
        .into_iter()
        .map(|mode| SessionMode::new(mode.as_str(), mode.as_str()))
        .collect()
}

/// The same modes as configuration option values.
fn mode_options() -> Vec<SessionConfigSelectOption> {
    PERMISSION_MODES
        .into_iter()
        .map(|mode| SessionConfigSelectOption::new(mode.as_str(), mode.as_str()))
        .collect()
}

/// Builds select options that always name `current`.
///
/// A configuration reload can drop a route or a reasoning level while the
/// running session keeps using it, so the active value is listed even when the
/// catalog no longer offers it.
fn select_options(
    current: &str,
    values: impl IntoIterator<Item = (String, String)>,
) -> Vec<SessionConfigSelectOption> {
    let mut options: Vec<SessionConfigSelectOption> = values
        .into_iter()
        .map(|(value, name)| SessionConfigSelectOption::new(value, name))
        .collect();
    if !options
        .iter()
        .any(|option| option.value.0.as_ref() == current)
    {
        options.insert(
            0,
            SessionConfigSelectOption::new(current.to_string(), current.to_string()),
        );
    }
    options
}

/// Reads a reasoning level from the engine's vocabulary. Levels come from
/// configured values, so an unlisted one is a provider-specific custom level.
pub(super) fn parse_reasoning_effort(value: &str) -> ModelReasoningEffort {
    crate::command::parse_reasoning_effort(value)
        .unwrap_or_else(|| ModelReasoningEffort::Custom(value.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::session::runner::{ModelCatalogReasoning, ModelCatalogUpdatedEvent};

    fn settings(
        models: Vec<ModelSummary>,
        effort: Option<ModelReasoningEffort>,
    ) -> SessionSettings {
        SessionSettings::new(models, effort)
    }

    fn model(id: &str, label: &str, efforts: &[ModelReasoningEffort]) -> ModelSummary {
        ModelSummary::new(id, label, efforts.to_vec())
    }

    fn select_options_of(option: &SessionConfigOption) -> Vec<String> {
        let agent_client_protocol::schema::v1::SessionConfigKind::Select(select) = &option.kind
        else {
            panic!("expected a select option");
        };
        let agent_client_protocol::schema::v1::SessionConfigSelectOptions::Ungrouped(values) =
            &select.options
        else {
            panic!("expected ungrouped select options");
        };
        values
            .iter()
            .map(|value| value.value.0.to_string())
            .collect()
    }

    fn current_value(option: &SessionConfigOption) -> String {
        let agent_client_protocol::schema::v1::SessionConfigKind::Select(select) = &option.kind
        else {
            panic!("expected a select option");
        };
        select.current_value.0.to_string()
    }

    #[test]
    fn new_session_state_reports_the_engine_mode_and_model() {
        let state = SessionState::new(
            "auto",
            "test/model",
            &settings(
                vec![model(
                    "test/model",
                    "Model",
                    &[ModelReasoningEffort::Low, ModelReasoningEffort::High],
                )],
                Some(ModelReasoningEffort::High),
            ),
        );

        let modes = state.modes();
        assert_eq!(modes.current_mode_id.0.to_string(), "auto");
        assert_eq!(
            modes
                .available_modes
                .iter()
                .map(|mode| mode.id.0.to_string())
                .collect::<Vec<_>>(),
            vec!["safe", "default", "auto", "yolo"]
        );

        let options = state.config_options();
        assert_eq!(
            options
                .iter()
                .map(|option| option.id.0.to_string())
                .collect::<Vec<_>>(),
            vec![CONFIG_SESSION_MODE, CONFIG_MODEL, CONFIG_REASONING_EFFORT]
        );
        assert_eq!(current_value(&options[0]), "auto");
        assert_eq!(current_value(&options[1]), "test/model");
    }

    #[test]
    fn config_options_omit_reasoning_without_an_engine_level() {
        let state = SessionState::new(
            "default",
            "test/model",
            &settings(
                vec![model("test/model", "Model", &[ModelReasoningEffort::Low])],
                None,
            ),
        );

        assert_eq!(
            state
                .config_options()
                .iter()
                .map(|option| option.id.0.to_string())
                .collect::<Vec<_>>(),
            vec![CONFIG_SESSION_MODE, CONFIG_MODEL]
        );
    }

    #[test]
    fn model_options_keep_the_active_route_when_the_catalog_drops_it() {
        let mut state = SessionState::new(
            "default",
            "test/retired",
            &settings(vec![model("test/retired", "Retired", &[])], None),
        );
        state.adopt_catalog(&ModelCatalogUpdatedEvent {
            models: vec![ModelCatalogEntry {
                id: "test/model".into(),
                label: "Model".into(),
                provider: "test".into(),
                context_window_tokens: None,
                reasoning: ModelCatalogReasoning {
                    effort: None,
                    efforts: Vec::new(),
                },
            }],
        });

        let options = state.config_options();
        assert_eq!(current_value(&options[1]), "test/retired");
        assert_eq!(
            select_options_of(&options[1]),
            vec!["test/retired", "test/model"]
        );
    }

    #[test]
    fn reasoning_options_list_the_active_models_levels() {
        let mut state = SessionState::new(
            "default",
            "test/model",
            &settings(
                vec![model(
                    "test/model",
                    "Model",
                    &[
                        ModelReasoningEffort::Low,
                        ModelReasoningEffort::Custom("turbo".into()),
                    ],
                )],
                Some(ModelReasoningEffort::Low),
            ),
        );

        assert_eq!(
            select_options_of(&state.config_options()[2]),
            vec!["low", "turbo"]
        );

        state.adopt_reasoning_effort(ModelReasoningEffort::High);
        let options = state.config_options();
        assert_eq!(current_value(&options[2]), "high");
        assert_eq!(
            select_options_of(&options[2]),
            vec!["high", "low", "turbo"],
            "a level the model does not list still names the active value"
        );
    }

    #[test]
    fn switching_to_a_route_without_levels_drops_the_reasoning_option() {
        let mut state = SessionState::new(
            "default",
            "test/model",
            &settings(
                vec![
                    model(
                        "test/model",
                        "Model",
                        &[ModelReasoningEffort::Low, ModelReasoningEffort::High],
                    ),
                    model("test/other", "Other", &[]),
                ],
                Some(ModelReasoningEffort::Low),
            ),
        );
        assert_eq!(
            state
                .config_options()
                .iter()
                .map(|option| option.id.0.to_string())
                .collect::<Vec<_>>(),
            vec![CONFIG_SESSION_MODE, CONFIG_MODEL, CONFIG_REASONING_EFFORT]
        );

        state.adopt_model("test/other");
        assert_eq!(
            state
                .config_options()
                .iter()
                .map(|option| option.id.0.to_string())
                .collect::<Vec<_>>(),
            vec![CONFIG_SESSION_MODE, CONFIG_MODEL]
        );
    }

    #[test]
    fn adopting_an_unexpressible_mode_keeps_the_current_state() {
        let mut state = SessionState::new(
            "default",
            "test/model",
            &settings(vec![model("test/model", "Model", &[])], None),
        );

        assert!(!state.adopt_mode("plan"));
        assert_eq!(state.modes().current_mode_id.0.to_string(), "default");
        assert!(state.adopt_mode("yolo"));
        assert_eq!(state.modes().current_mode_id.0.to_string(), "yolo");
    }
}
