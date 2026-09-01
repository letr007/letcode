use super::{
    CacheIntent, ContentPart, ControlSegment, ControlSegmentKind, GenerationSettings, MessageRole,
    ModelMessage, ModelRequestInput, ReasoningEffort, ReasoningIntent, ReasoningSummary,
    StablePrefixMetadata, ToolDefinition, Verbosity,
};
use crate::request_builder::prompt_plan::{PromptPlan, PromptSegmentContent, PromptSegmentRole};
use crate::request_builder::{
    ModelReasoningEffort, ModelReasoningSummary, ModelRequestMetadata, ModelTextVerbosity, ToolSpec,
};
use base64::{Engine as _, engine::general_purpose::STANDARD};

pub(crate) fn model_request_from_prompt_plan(
    route: &super::ResolvedModelRoute,
    metadata: &ModelRequestMetadata,
    prompt_plan: &PromptPlan,
    tools: &[ToolSpec],
) -> Result<ModelRequestInput, String> {
    let mut request = ModelRequestInput::new(route.model_override.clone(), Vec::new());
    for segment in &prompt_plan.segments {
        match (&segment.role, &segment.content) {
            (PromptSegmentRole::System, PromptSegmentContent::Text { text }) => {
                request.segments.push(ControlSegment::system(text.clone()));
                request.segment_origins.push(segment.id.clone());
            }
            (PromptSegmentRole::Developer, PromptSegmentContent::Text { text }) => {
                request.segments.push(ControlSegment {
                    kind: ControlSegmentKind::Developer,
                    text: text.clone(),
                });
                request.segment_origins.push(segment.id.clone());
            }
            (PromptSegmentRole::User, PromptSegmentContent::Text { text }) => {
                request
                    .messages
                    .push(ModelMessage::text(MessageRole::User, text.clone()));
                request.message_origins.push(segment.id.clone());
            }
            (PromptSegmentRole::User, PromptSegmentContent::UserContent { content }) => {
                let mut parts = Vec::new();
                for part in content.parts() {
                    match part {
                        crate::user_content::UserMessagePart::Text { text } => {
                            parts.push(ContentPart::Text(text));
                        }
                        crate::user_content::UserMessagePart::Image { attachment } => {
                            let prefix = format!("data:{};base64,", attachment.mime);
                            let encoded =
                                attachment.data_url.strip_prefix(&prefix).ok_or_else(|| {
                                    format!("invalid image data URL for {}", attachment.id)
                                })?;
                            let data = STANDARD
                                .decode(encoded)
                                .map_err(|error| format!("invalid image data: {error}"))?;
                            parts.push(ContentPart::Image {
                                media_type: attachment.mime,
                                data,
                            });
                        }
                    }
                }
                request.messages.push(ModelMessage {
                    role: MessageRole::User,
                    content: parts,
                });
                request.message_origins.push(segment.id.clone());
            }
            (PromptSegmentRole::Assistant, PromptSegmentContent::Text { text }) => {
                request
                    .messages
                    .push(ModelMessage::text(MessageRole::Assistant, text.clone()));
                request.message_origins.push(segment.id.clone());
            }
            (
                PromptSegmentRole::Assistant,
                PromptSegmentContent::AssistantToolCalls {
                    text,
                    reasoning_content,
                    replay,
                    calls,
                },
            ) => {
                let mut parts = Vec::new();
                if let Some(reasoning) = reasoning_content
                    && (!reasoning.is_empty() || replay.is_some())
                {
                    parts.push(ContentPart::Reasoning {
                        item_id: format!("reasoning:{}", segment.id),
                        text: reasoning.clone(),
                        replay: replay.clone(),
                    });
                }
                if let Some(text) = text.clone().filter(|text| !text.is_empty()) {
                    parts.push(ContentPart::Text(text));
                }
                for call in calls {
                    let arguments = serde_json::from_str(&call.arguments_json)
                        .map_err(|error| format!("invalid tool arguments: {error}"))?;
                    parts.push(ContentPart::ToolCall {
                        id: call.call_id.clone(),
                        name: call.name.clone(),
                        arguments,
                    });
                }
                request.messages.push(ModelMessage {
                    role: MessageRole::Assistant,
                    content: parts,
                });
                request.message_origins.push(segment.id.clone());
            }
            (
                PromptSegmentRole::Tool,
                PromptSegmentContent::ToolOutput {
                    call_id,
                    output_json,
                    images,
                },
            ) => {
                let mut content = vec![ContentPart::Text(output_json.clone())];
                for image in images {
                    let prefix = format!("data:{};base64,", image.mime);
                    let encoded = image
                        .data_url
                        .strip_prefix(&prefix)
                        .ok_or_else(|| format!("invalid tool image data URL for {}", image.id))?;
                    content.push(ContentPart::Image {
                        media_type: image.mime.clone(),
                        data: STANDARD
                            .decode(encoded)
                            .map_err(|error| format!("invalid tool image data: {error}"))?,
                    });
                }
                request.messages.push(ModelMessage {
                    role: MessageRole::Tool,
                    content: vec![ContentPart::ToolResult {
                        id: call_id.clone(),
                        content,
                    }],
                });
                request.message_origins.push(segment.id.clone());
            }
            (role, content) => {
                return Err(format!(
                    "unsupported prompt segment role/content pair: {role:?}/{content:?}"
                ));
            }
        }
    }
    request.tools = tools
        .iter()
        .map(|tool| ToolDefinition {
            name: tool.name.clone(),
            description: tool.description.clone(),
            parameters: tool.parameters.clone(),
            strict: tool.strict,
        })
        .collect();
    request.generation = generation_settings(route, metadata)?;
    request.cache = CacheIntent {
        enabled: metadata.prompt_cache.enabled,
        namespace: metadata.prompt_cache.namespace.clone(),
        retention: metadata
            .prompt_cache
            .retention
            .map(|retention| match retention {
                crate::config::PromptCacheRetention::InMemory => super::CacheRetention::InMemory,
                crate::config::PromptCacheRetention::TwentyFourHours => {
                    super::CacheRetention::TwentyFourHours
                }
            }),
        stable_prefix: prompt_plan
            .stable_prefix_end
            .map(|index| StablePrefixMetadata {
                segment_count: index + 1,
                fingerprint: prompt_plan.stable_prefix_hash().map(str::to_owned),
            }),
    };
    Ok(request)
}

fn generation_settings(
    route: &super::ResolvedModelRoute,
    metadata: &ModelRequestMetadata,
) -> Result<GenerationSettings, String> {
    Ok(GenerationSettings {
        temperature: metadata.temperature,
        top_p: metadata.top_p,
        max_output_tokens: metadata
            .max_output_tokens
            .map(|value| u32::try_from(value).map_err(|_| "max_output_tokens exceeds u32"))
            .transpose()?,
        stop_sequences: Vec::new(),
        reasoning: ReasoningIntent {
            enabled: metadata.reasoning_effort.is_some()
                || metadata.reasoning_summary.is_some()
                || !matches!(
                    metadata.anthropic_thinking.mode,
                    crate::request_builder::AnthropicThinkingMode::Disabled
                ),
            effort: metadata.reasoning_effort.as_ref().map(reasoning_effort),
        },
        summary: metadata.reasoning_summary.map(|value| match value {
            ModelReasoningSummary::Auto => ReasoningSummary::Auto,
            ModelReasoningSummary::Concise => ReasoningSummary::Concise,
            ModelReasoningSummary::Detailed => ReasoningSummary::Detailed,
        }),
        verbosity: metadata.text_verbosity.map(|value| match value {
            ModelTextVerbosity::Low => Verbosity::Low,
            ModelTextVerbosity::Medium => Verbosity::Medium,
            ModelTextVerbosity::High => Verbosity::High,
        }),
        parallel_tool_calls: match route.protocol_id.as_str() {
            "responses" | "completions" if metadata.supports_tools => {
                Some(metadata.parallel_tool_calls)
            }
            _ => None,
        },
        priority_service: match route.protocol_id.as_str() {
            "responses" | "completions" if metadata.fast_mode => Some(true),
            _ => None,
        },
    })
}

fn reasoning_effort(value: &ModelReasoningEffort) -> ReasoningEffort {
    match value {
        ModelReasoningEffort::None => ReasoningEffort::None,
        ModelReasoningEffort::Minimal => ReasoningEffort::Minimal,
        ModelReasoningEffort::Low => ReasoningEffort::Low,
        ModelReasoningEffort::Medium => ReasoningEffort::Medium,
        ModelReasoningEffort::High => ReasoningEffort::High,
        ModelReasoningEffort::Xhigh => ReasoningEffort::Xhigh,
        ModelReasoningEffort::Max => ReasoningEffort::Max,
        ModelReasoningEffort::Custom(value) => ReasoningEffort::Custom(value.clone()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocol_frames::{ProtocolFrame, ProtocolToolCall};
    use crate::request_builder::prompt_plan::{PromptPlanBuildInput, build_prompt_plan};
    use crate::request_builder::{HistoryItem, PromptMessage};
    use crate::runtime_context::RuntimeSnapshot;

    #[test]
    fn projection_preserves_replay_tools_images_and_cache_boundary() {
        let replay = super::super::OpaqueReplayState::from_anthropic_thinking_blocks_json(
            r#"[{"type":"thinking","thinking":"plan","signature":"signed"}]"#,
        )
        .unwrap();
        let image =
            crate::user_content::UserImageAttachment::from_bytes("image", "image/png", b"png");
        let history = vec![
            HistoryItem::UserMessage {
                content: crate::user_content::UserMessageContent::new("hello", vec![image]),
            },
            HistoryItem::AssistantTurn {
                text: Some("working".into()),
                reasoning_content: Some("plan".into()),
                replay: Some(replay.clone()),
                calls: vec![ProtocolToolCall {
                    call_id: "call-1".into(),
                    name: "search".into(),
                    arguments_json: "{}".into(),
                }],
            },
            HistoryItem::ToolOutput {
                call_id: "call-1".into(),
                output_json: "{\"ok\":true}".into(),
                images: Vec::new(),
            },
        ];
        let frames = history
            .into_iter()
            .enumerate()
            .map(|(index, item)| ProtocolFrame::from_history_item(index, &item))
            .collect::<Vec<_>>();
        let plan = build_prompt_plan(PromptPlanBuildInput {
            model_id: "model",
            prelude: &[PromptMessage::system("system")],
            snapshot: &RuntimeSnapshot::new("test"),
            selected_frames: &frames,
            segment_order_offset: 0,
            protected_suffix_len: 1,
            evidence_message: None,
            selected_evidence_ids: &[],
        });
        let metadata = ModelRequestMetadata {
            supports_tools: true,
            supports_reasoning: true,
            reasoning_effort: Some(ModelReasoningEffort::High),
            max_output_tokens: Some(128),
            prompt_cache: crate::config::PromptCacheConfig {
                enabled: true,
                ..Default::default()
            },
            ..Default::default()
        };
        let route = crate::model_runtime::RuntimeConfig::from_toml(
            r#"active_provider = "vendor"
[providers.vendor]
protocol = "responses"
default_model = "model"
[providers.vendor.auth]
type = "none"
[providers.vendor.endpoints]
base_url = "https://example.invalid/v1"
[providers.vendor.models.model]
"#,
        )
        .unwrap()
        .resolve(&crate::model_runtime::ProtocolRegistry::builtins())
        .unwrap()
        .route("vendor", "model")
        .unwrap()
        .clone();
        let request = model_request_from_prompt_plan(
            &route,
            &metadata,
            &plan,
            &[ToolSpec {
                name: "search".into(),
                description: "search".into(),
                parameters: serde_json::json!({"type":"object"}),
                strict: true,
            }],
        )
        .unwrap();
        assert!(matches!(
            request.segments[0].kind,
            ControlSegmentKind::System
        ));
        assert!(request.messages.iter().any(|message| message.content.iter().any(
            |part| matches!(part, ContentPart::Reasoning { replay: Some(value), .. } if value == &replay)
        )));
        assert!(request.messages.iter().any(|message| {
            message
                .content
                .iter()
                .any(|part| matches!(part, ContentPart::ToolCall { id, .. } if id == "call-1"))
        }));
        assert_eq!(request.tools.len(), 1);
        assert_eq!(request.generation.max_output_tokens, Some(128));
    }
}
