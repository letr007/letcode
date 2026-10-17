use super::*;
use crate::session::runner::{ModelCatalogReasoning, ModelCatalogUpdatedEvent};

fn settings(models: Vec<ModelSummary>, effort: Option<ModelReasoningEffort>) -> SessionSettings {
    SessionSettings::new(models, effort)
}

fn model(id: &str, label: &str, efforts: &[ModelReasoningEffort]) -> ModelSummary {
    ModelSummary::new(id, label, efforts.to_vec())
}

fn select_options_of(option: &SessionConfigOption) -> Vec<String> {
    let agent_client_protocol::schema::v1::SessionConfigKind::Select(select) = &option.kind else {
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
    let agent_client_protocol::schema::v1::SessionConfigKind::Select(select) = &option.kind else {
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
