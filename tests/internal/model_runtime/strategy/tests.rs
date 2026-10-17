use super::*;

#[test]
fn explicit_strategy_overrides_name_inference() {
    assert_eq!(
        ModelStrategyId::resolve(Some(ModelStrategyId::Default), "gpt-6-astra", None,),
        ModelStrategyId::Default
    );
    assert_eq!(
        ModelStrategyId::resolve(Some(ModelStrategyId::Astra), "other-model", None,),
        ModelStrategyId::Astra
    );
}

#[test]
fn astra_is_inferred_from_configured_or_wire_model_name() {
    assert_eq!(
        ModelStrategyId::resolve(None, "gpt-6-astra", None),
        ModelStrategyId::Astra
    );
    assert_eq!(
        ModelStrategyId::resolve(None, "friendly-name", Some("GPT-6-Astra-preview")),
        ModelStrategyId::Astra
    );
    assert_eq!(
        ModelStrategyId::resolve(None, "gpt-5.6", None),
        ModelStrategyId::Default
    );
}

#[test]
fn deepseek_is_inferred_from_configured_or_wire_model_name() {
    assert_eq!(
        ModelStrategyId::resolve(None, "deepseek-flash", None),
        ModelStrategyId::Deepseek
    );
    assert_eq!(
        ModelStrategyId::resolve(None, "friendly-name", Some("DeepSeek-V4")),
        ModelStrategyId::Deepseek
    );
    assert_eq!(
        ModelStrategyId::resolve(Some(ModelStrategyId::Default), "deepseek-flash", None),
        ModelStrategyId::Default
    );
    assert_eq!(
        ModelStrategyId::resolve(None, "gpt-5.6", None),
        ModelStrategyId::Default
    );
}

#[test]
fn deepseek_harness_is_interactive_only() {
    let message = ModelStrategyId::Deepseek
        .interactive_prelude_message()
        .expect("deepseek harness");
    assert_eq!(message.origin, PromptMessageOrigin::StaticPrelude);
    assert!(message.text.contains("DeepSeek 工作指引"));
}

#[test]
fn astra_reasoning_starts_at_low() {
    let mut metadata = ModelRequestMetadata {
        supports_reasoning: true,
        reasoning_effort: Some(ModelReasoningEffort::None),
        ..Default::default()
    };
    ModelStrategyId::Astra.normalize_request_metadata(&mut metadata);
    assert_eq!(metadata.reasoning_effort, Some(ModelReasoningEffort::Low));
    assert_eq!(
        metadata.reasoning_efforts,
        vec![
            ModelReasoningEffort::Low,
            ModelReasoningEffort::Medium,
            ModelReasoningEffort::High,
            ModelReasoningEffort::Xhigh,
            ModelReasoningEffort::Max,
        ]
    );
}

#[test]
fn astra_uses_the_first_explicit_effort_when_no_default_is_configured() {
    let mut metadata = ModelRequestMetadata {
        supports_reasoning: true,
        reasoning_efforts: vec![ModelReasoningEffort::High],
        ..Default::default()
    };
    ModelStrategyId::Astra.normalize_request_metadata(&mut metadata);
    assert_eq!(metadata.reasoning_effort, Some(ModelReasoningEffort::High));
    assert_eq!(metadata.reasoning_efforts, vec![ModelReasoningEffort::High]);
}

#[test]
fn astra_harness_is_interactive_only_and_default_is_empty() {
    assert!(
        ModelStrategyId::Default
            .interactive_prelude_message()
            .is_none()
    );
    let message = ModelStrategyId::Astra
        .interactive_prelude_message()
        .expect("astra harness");
    assert_eq!(message.origin, PromptMessageOrigin::StaticPrelude);
    assert!(message.text.contains("GPT-6 Astra 工作指引"));
    assert!(message.text.contains("自然段"));
}
