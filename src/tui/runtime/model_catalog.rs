//! Model and expert catalog types, independent of the runtime orchestrator.
//!
//! These describe the backend's available models/experts. They own no
//! `TuiRuntime` state and are constructed from the backend catalog (`main`
//! builds them for the runtime), so they live apart from the God-file body.

use crate::request_builder::ModelReasoningEffort;
use crate::session::runner::ModelCatalogEntry;

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct AvailableModel {
    pub(crate) id: String,
    pub(crate) label: String,
    pub(crate) provider: String,
    pub(crate) context_window_tokens: Option<u64>,
    pub(crate) reasoning_effort: Option<ModelReasoningEffort>,
    pub(crate) reasoning_efforts: Vec<ModelReasoningEffort>,
}

impl AvailableModel {
    pub(crate) fn from_catalog_entry(entry: &ModelCatalogEntry) -> Self {
        Self {
            id: entry.id.clone(),
            label: entry.label.clone(),
            provider: entry.provider.clone(),
            context_window_tokens: entry.context_window_tokens,
            reasoning_effort: entry
                .reasoning
                .effort
                .as_deref()
                .map(parse_catalog_reasoning_effort),
            reasoning_efforts: entry
                .reasoning
                .efforts
                .iter()
                .map(|effort| parse_catalog_reasoning_effort(effort))
                .collect(),
        }
    }

    #[cfg(test)]
    pub(crate) fn new(id: impl Into<String>, label: impl Into<String>) -> Self {
        let id = id.into();
        Self {
            provider: model_provider(&id),
            id,
            label: label.into(),
            context_window_tokens: None,
            reasoning_effort: None,
            reasoning_efforts: Vec::new(),
        }
    }

    #[cfg(test)]
    pub(crate) fn with_context_window(
        id: impl Into<String>,
        label: impl Into<String>,
        context_window_tokens: Option<u64>,
    ) -> Self {
        let id = id.into();
        Self {
            provider: model_provider(&id),
            id,
            label: label.into(),
            context_window_tokens,
            reasoning_effort: None,
            reasoning_efforts: Vec::new(),
        }
    }

    pub(crate) fn with_context_window_and_reasoning(
        id: impl Into<String>,
        label: impl Into<String>,
        context_window_tokens: Option<u64>,
        reasoning_effort: Option<ModelReasoningEffort>,
        reasoning_efforts: Vec<ModelReasoningEffort>,
    ) -> Self {
        let id = id.into();
        Self {
            provider: model_provider(&id),
            id,
            label: label.into(),
            context_window_tokens,
            reasoning_effort,
            reasoning_efforts,
        }
    }
}

pub(crate) fn parse_catalog_reasoning_effort(value: &str) -> ModelReasoningEffort {
    match value {
        "none" => ModelReasoningEffort::None,
        "minimal" => ModelReasoningEffort::Minimal,
        "low" => ModelReasoningEffort::Low,
        "medium" => ModelReasoningEffort::Medium,
        "high" => ModelReasoningEffort::High,
        "xhigh" => ModelReasoningEffort::Xhigh,
        "max" => ModelReasoningEffort::Max,
        other => ModelReasoningEffort::Custom(other.to_string()),
    }
}

fn model_provider(model_id: &str) -> String {
    model_id
        .split_once('/')
        .map(|(provider, _)| provider.to_string())
        .unwrap_or_default()
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct AvailableExpert {
    pub(crate) agent_name: String,
    pub(crate) route_id: String,
}
