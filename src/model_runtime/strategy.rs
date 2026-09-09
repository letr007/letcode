use super::{ProviderFlavor, RuntimeGenerationDefaults};
#[cfg(test)]
use crate::request_builder::PromptMessageOrigin;
use crate::request_builder::{ModelReasoningEffort, ModelRequestMetadata, PromptMessage};
use serde::{Deserialize, Serialize};

const ASTRA_INTERACTIVE_HARNESS: &str = r#"GPT-6 Astra 工作指引：

当用户表达希望你采取行动时，应实际推进任务，并持续工作到目标完成。先利用现有上下文完成已经获得授权的只读、可撤销和本地准备工作；只有缺失信息会实质影响正确性、范围或不可逆操作时，才暂停询问。需要用户批准的最终外部或破坏性操作，应在其他准备工作完成并形成可审查结果后再请求批准。

严格遵循指令的真实权威与适用范围。若 AGENTS.md、Skill 或其他已加载指南导致你暂停、改变做法或无法完成任务，应指出具体来源和相关要求，并区分明文规定与自己的推断。不要把模糊、无关或仅供参考的指南扩大解释为新的阻塞条件。

根据任务规模选择协作和验证力度。并行工作能明显节省时间或提高质量时使用 subagent；简单任务直接完成。运行与改动风险相匹配的检查，测试通过后不要在没有新变化或新证据时机械扩大或重复验证。

默认使用清晰、连贯的自然段。仅在信息天然并列、存在明确步骤或确实需要比较时使用列表、标题或表格。避免套话、宣传式措辞、机械总结、过多粗体和装饰性 Markdown；直接说明事实、判断、操作和结果。

可独立发起且互不依赖的工具调用应尽量在同一批次并行；只有存在真实的数据或执行依赖时才串行调用。"#;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
#[serde(rename_all = "kebab-case")]
pub enum ModelStrategyId {
    #[default]
    Default,
    Astra,
}

impl ModelStrategyId {
    pub fn resolve(
        configured: Option<Self>,
        model_name: &str,
        model_override: Option<&str>,
    ) -> Self {
        configured.unwrap_or_else(|| {
            if is_astra_model(model_name) || model_override.is_some_and(is_astra_model) {
                Self::Astra
            } else {
                Self::Default
            }
        })
    }

    pub const fn default_protocol(self) -> &'static str {
        "responses"
    }

    pub const fn default_flavor(self) -> ProviderFlavor {
        ProviderFlavor::Standard
    }

    pub fn validate_binding(self, protocol: &str, flavor: ProviderFlavor) -> Result<(), String> {
        match self {
            Self::Default => Ok(()),
            Self::Astra
                if protocol == self.default_protocol() && flavor == self.default_flavor() =>
            {
                Ok(())
            }
            Self::Astra => Err(format!(
                "strategy 'astra' requires protocol '{}' and flavor '{}'",
                self.default_protocol(),
                self.default_flavor().as_str()
            )),
        }
    }

    pub fn validate_generation(
        self,
        generation: &RuntimeGenerationDefaults,
        reasoning_capability: bool,
        reasoning_generation: bool,
    ) -> Result<(), String> {
        if self == Self::Astra && reasoning_capability && !reasoning_generation {
            return Err(
                "strategy 'astra' requires capabilities.generation.reasoning when reasoning is enabled"
                    .into(),
            );
        }
        if self == Self::Astra
            && (generation.reasoning_effort.as_deref() == Some("none")
                || generation.reasoning_effort.as_deref() == Some("minimal")
                || generation
                    .reasoning_efforts
                    .iter()
                    .any(|effort| matches!(effort.as_str(), "none" | "minimal")))
        {
            return Err("strategy 'astra' supports reasoning effort 'low' or higher".into());
        }
        Ok(())
    }

    pub fn normalize_request_metadata(self, metadata: &mut ModelRequestMetadata) {
        if self != Self::Astra || !metadata.supports_reasoning {
            return;
        }

        let configured_efforts = !metadata.reasoning_efforts.is_empty();
        metadata.reasoning_efforts.retain(|effort| {
            !matches!(
                effort,
                ModelReasoningEffort::None | ModelReasoningEffort::Minimal
            )
        });
        if !configured_efforts {
            metadata.reasoning_efforts = vec![
                ModelReasoningEffort::Low,
                ModelReasoningEffort::Medium,
                ModelReasoningEffort::High,
                ModelReasoningEffort::Xhigh,
                ModelReasoningEffort::Max,
            ];
            if let Some(effort) = metadata.reasoning_effort.as_ref()
                && !matches!(
                    effort,
                    ModelReasoningEffort::None | ModelReasoningEffort::Minimal
                )
                && !metadata.reasoning_efforts.contains(effort)
            {
                metadata.reasoning_efforts.push(effort.clone());
            }
        }
        if !metadata
            .reasoning_effort
            .as_ref()
            .is_some_and(|effort| metadata.reasoning_efforts.contains(effort))
        {
            metadata.reasoning_effort = metadata.reasoning_efforts.first().cloned();
        }
    }

    pub fn interactive_prelude_message(self) -> Option<PromptMessage> {
        match self {
            Self::Default => None,
            Self::Astra => Some(PromptMessage::system(ASTRA_INTERACTIVE_HARNESS)),
        }
    }
}

fn is_astra_model(value: &str) -> bool {
    let value = value.to_ascii_lowercase();
    value.contains("gpt") && value.contains("astra")
}

#[cfg(test)]
mod tests {
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
}
