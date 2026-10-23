use super::*;
use crate::protocol_frames::{ProtocolFrame, ProtocolToolCall};
use crate::request_builder::prompt_plan::{PromptPlanBuildInput, build_prompt_plan};
use crate::request_builder::{HistoryItem, PromptMessage};
use crate::runtime_context::RuntimeSnapshot;

fn replay_test_route(protocol: &str) -> super::super::ResolvedModelRoute {
    crate::model_runtime::RuntimeConfig::from_toml(&format!(
        r#"active_provider = "vendor"
[providers.vendor]
protocol = "{protocol}"
default_model = "model"
[providers.vendor.auth]
type = "none"
[providers.vendor.endpoints]
base_url = "https://example.invalid/v1"
[providers.vendor.models.model.capabilities]
tools = true
parallel_tool_calls = true
reasoning = true
input_images = true
[providers.vendor.models.model.capabilities.generation]
max_output_tokens = true
parallel_tool_calls = true
"#
    ))
    .unwrap()
    .resolve(&crate::model_runtime::ProtocolRegistry::builtins())
    .unwrap()
    .route("vendor", "model")
    .unwrap()
    .clone()
}

#[test]
fn history_replay_preserves_native_state_and_converts_foreign_messages() {
    use crate::model_runtime::{OpaqueReplayState, ReplayProducer};
    use serde_json::{Value, json};

    for source in ["responses", "anthropic"] {
        let source_route = replay_test_route(source);
        let replay = if source == "anthropic" {
            OpaqueReplayState::from_anthropic_thinking_blocks_json(
                r#"[{"type":"redacted_thinking","data":"native-opaque"}]"#,
            )
            .unwrap()
        } else {
            OpaqueReplayState::new(
                "responses.reasoning",
                1,
                ReplayProducer::new(
                    source_route.binding.replay_scope(),
                    source_route.binding.binding_identity(),
                ),
                json!({"type":"reasoning", "id":"rs_1", "encrypted_content":"native-opaque", "summary":[]}),
            )
        };
        let history = vec![
            HistoryItem::UserMessage {
                content: crate::user_content::UserMessageContent::new(
                    "question",
                    vec![crate::user_content::UserImageAttachment::from_bytes(
                        "image",
                        "image/png",
                        b"png",
                    )],
                ),
            },
            HistoryItem::AssistantTurn {
                text: None,
                reasoning_content: None,
                replay: Some(replay),
                calls: vec![],
            },
            HistoryItem::AssistantTurn {
                text: None,
                reasoning_content: Some("private reasoning".into()),
                replay: None,
                calls: vec![],
            },
            HistoryItem::AssistantTurn {
                text: Some("checking".into()),
                reasoning_content: None,
                replay: None,
                calls: ["call_a", "call_b"]
                    .into_iter()
                    .map(|id| ProtocolToolCall {
                        call_id: id.into(),
                        name: "search".into(),
                        arguments_json: "{}".into(),
                    })
                    .collect(),
            },
            HistoryItem::ToolOutput {
                call_id: "call_a".into(),
                output_json: "result a".into(),
                images: vec![],
            },
            HistoryItem::ToolOutput {
                call_id: "call_b".into(),
                output_json: "result b".into(),
                images: vec![],
            },
            HistoryItem::assistant("answer"),
            HistoryItem::user("continue"),
        ];
        let saved = serde_json::to_vec(&history).unwrap();
        let restored: Vec<HistoryItem> = serde_json::from_slice(&saved).unwrap();
        let frames = restored
            .iter()
            .enumerate()
            .map(|(index, item)| ProtocolFrame::from_history_item(index, item))
            .collect::<Vec<_>>();
        let plan = build_prompt_plan(PromptPlanBuildInput {
            model_id: "model",
            prelude: &[
                PromptMessage::system("system rules"),
                PromptMessage::developer("dynamic context"),
            ],
            snapshot: &RuntimeSnapshot::new("test"),
            selected_frames: &frames,
            segment_order_offset: 0,
            protected_suffix_len: 1,
            evidence_message: None,
            selected_evidence_ids: &[],
        });
        for target in ["responses", "anthropic", "completions"] {
            let route = replay_test_route(target);
            let request = model_request_from_prompt_plan(
                &route,
                &ModelRequestMetadata {
                    supports_tools: true,
                    ..Default::default()
                },
                &plan,
                &[],
            )
            .unwrap();
            let prepared = route.binding.prepare_request(&request).unwrap();
            let body: Value = serde_json::from_slice(&prepared.body).unwrap();
            let wire = body.to_string();
            assert_eq!(
                wire.contains("native-opaque"),
                source == target,
                "{source} -> {target}: {body}"
            );
            assert!(!wire.contains("private reasoning"));
            for text in [
                "question", "checking", "result a", "result b", "answer", "continue", "call_a",
                "call_b",
            ] {
                assert!(wire.contains(text), "{source} -> {target}: missing {text}");
            }
            if target == "anthropic" {
                assert!(
                    body["messages"]
                        .as_array()
                        .unwrap()
                        .iter()
                        .all(|message| { !message["content"].as_array().unwrap().is_empty() })
                );
                assert_eq!(body["system"][0]["text"], "system rules");
            } else if target == "completions" {
                assert!(body["messages"].as_array().unwrap().iter().all(|message| {
                    !message["content"].is_null() || message["tool_calls"].is_array()
                }));
                assert_eq!(body["messages"][0]["role"], "system");
            } else {
                assert_eq!(body["instructions"], "system rules");
            }
            let inspection = route
                .binding
                .inspect_prepared_request(&prepared, None)
                .unwrap();
            assert_eq!(
                inspection.prompt_units.len(),
                prepared.prompt_unit_origins.len()
            );
            assert!(inspection.prompt_units.iter().all(|unit| {
                !unit.semantic_segment_ids.is_empty()
                    && unit
                        .semantic_segment_ids
                        .iter()
                        .all(|id| plan.segments.iter().any(|segment| &segment.id == id))
            }));
        }
        assert_eq!(serde_json::to_vec(&restored).unwrap(), saved);
    }
}

#[test]
fn disabled_parallel_tool_calls_stay_off_the_wire() {
    use crate::request_builder::{TestRequestBuilderInput, build_test_request};
    use crate::user_content::UserMessageContent;

    let history = vec![HistoryItem::user_content(UserMessageContent::new(
        "hello",
        Vec::new(),
    ))];
    for protocol in ["responses", "completions"] {
        let route = replay_test_route(protocol);
        for enabled in [false, true] {
            let model = ModelRequestMetadata {
                supports_tools: true,
                parallel_tool_calls: enabled,
                ..Default::default()
            };
            let build = build_test_request(TestRequestBuilderInput {
                model_id: "model",
                model: model.clone(),
                prelude: &[],
                history: &history,
                evidence: &[],
                tools: &[],
                protected_start_index: 0,
            })
            .unwrap();
            let request =
                model_request_from_prompt_plan(&route, &model, &build.prompt_plan, &[]).unwrap();
            assert_eq!(
                request.generation.parallel_tool_calls,
                enabled.then_some(true),
                "{protocol} parallel_tool_calls={enabled}"
            );
        }
    }
}

#[test]
fn image_budget_and_wire_content_follow_route_capabilities() {
    use crate::request_builder::{TestRequestBuilderInput, build_test_request};
    use crate::user_content::{UserImageAttachment, UserMessageContent};

    let image = UserImageAttachment::from_bytes("frame.png", "image/png", b"png");
    let history = vec![
        HistoryItem::user_content(UserMessageContent::new("look", vec![image.clone()])),
        HistoryItem::AssistantTurn {
            text: None,
            reasoning_content: None,
            replay: None,
            calls: vec![ProtocolToolCall {
                call_id: "read-1".into(),
                name: "read".into(),
                arguments_json: "{}".into(),
            }],
        },
        HistoryItem::ToolOutput {
            call_id: "read-1".into(),
            output_json: "image metadata".into(),
            images: vec![image.clone()],
        },
    ];
    for protocol in ["responses", "completions", "anthropic"] {
        for input_images in [false, true] {
            // The Completions binding supports images in user messages,
            // but its tool result representation is text-only.
            let tool_image_modes: &[bool] = if protocol == "completions" {
                &[false]
            } else {
                &[false, true]
            };
            for &tool_result_images in tool_image_modes {
                let catalog = crate::model_runtime::RuntimeConfig::from_toml(&format!(
                    r#"
active_provider = "p"
[providers.p]
protocol = "{protocol}"
default_model = "m"
[providers.p.auth]
type = "none"
[providers.p.endpoints]
base_url = "https://example.invalid/v1"
[providers.p.models.m.capabilities]
tools = true
parallel_tool_calls = true
generation = {{ parallel_tool_calls = true }}
input_images = {input_images}
tool_result_images = {tool_result_images}
"#
                ))
                .unwrap()
                .resolve(&crate::model_runtime::ProtocolRegistry::builtins())
                .unwrap();
                let route = catalog.route("p", "m").unwrap();
                let model = ModelRequestMetadata {
                    context_window: Some(64_000),
                    supports_tools: true,
                    supports_input_images: route.capabilities.input_images,
                    supports_tool_result_images: route.capabilities.tool_result_images,
                    ..Default::default()
                };
                let build = build_test_request(TestRequestBuilderInput {
                    model_id: "m",
                    model: model.clone(),
                    prelude: &[],
                    history: &history,
                    evidence: &[],
                    tools: &[],
                    protected_start_index: 0,
                })
                .unwrap();
                let request =
                    model_request_from_prompt_plan(route, &model, &build.prompt_plan, &[]).unwrap();
                let user_parts = &request.messages[0].content;
                assert_eq!(
                    user_parts
                        .iter()
                        .any(|part| matches!(part, ContentPart::Image { .. })),
                    input_images
                );
                assert_eq!(user_parts.iter().any(|part| matches!(part, ContentPart::Text(text) if text.contains("[Image: frame.png]"))), !input_images);
                let ContentPart::ToolResult { content, .. } = &request.messages[2].content[0]
                else {
                    panic!("tool result")
                };
                assert_eq!(
                    content
                        .iter()
                        .any(|part| matches!(part, ContentPart::Image { .. })),
                    tool_result_images
                );
                assert_eq!(content.iter().any(|part| matches!(part, ContentPart::Text(text) if text.contains("[Image: frame.png]"))), !tool_result_images);
                route.binding.prepare_request(&request).unwrap_or_else(|error| panic!("{protocol} input_images={input_images} tool_result_images={tool_result_images}: {error:?}"));
                let visual_tokens = (u64::from(input_images) + u64::from(tool_result_images))
                    * image.visual_token_charge();
                assert!(build.budget.estimated_request_tokens >= visual_tokens);
                if visual_tokens == 0 {
                    assert!(build.budget.estimated_request_tokens < image.visual_token_charge());
                }

                // Estimation retains missing payloads for capable models; only
                // preparing a request decodes and rejects those attachments.
                for user_image in [false, true] {
                    let mut missing = history.clone();
                    if user_image {
                        let mut invalid = image.clone();
                        invalid.data_url.clear();
                        missing[0] = HistoryItem::user_content(UserMessageContent::new(
                            "look",
                            vec![invalid],
                        ));
                    } else if let HistoryItem::ToolOutput { images, .. } = &mut missing[2] {
                        images[0].data_url.clear();
                    }
                    let estimate = build_test_request(TestRequestBuilderInput {
                        model_id: "m",
                        model: model.clone(),
                        prelude: &[],
                        history: &missing,
                        evidence: &[],
                        tools: &[],
                        protected_start_index: 0,
                    })
                    .unwrap();
                    let result =
                        model_request_from_prompt_plan(route, &model, &estimate.prompt_plan, &[]);
                    if if user_image {
                        input_images
                    } else {
                        tool_result_images
                    } {
                        assert!(result.unwrap_err().contains("data URL"));
                    } else {
                        route.binding.prepare_request(&result.unwrap()).unwrap();
                    }
                }
            }
        }
    }
}

#[test]
fn projection_preserves_replay_tools_images_and_cache_boundary() {
    let replay = super::super::OpaqueReplayState::from_anthropic_thinking_blocks_json(
        r#"[{"type":"thinking","thinking":"plan","signature":"signed"}]"#,
    )
    .unwrap();
    let image = crate::user_content::UserImageAttachment::from_bytes("image", "image/png", b"png");
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

#[test]
fn skill_only_user_content_projects_to_a_text_marker() {
    let project = |content: crate::user_content::UserMessageContent| {
        let frames = vec![ProtocolFrame::from_history_item(
            0,
            &HistoryItem::UserMessage { content },
        )];
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
        model_request_from_prompt_plan(
            &replay_test_route("anthropic"),
            &ModelRequestMetadata::default(),
            &plan,
            &[],
        )
        .unwrap()
    };

    let request = project(
        crate::user_content::UserMessageContent::new("", vec![])
            .with_selected_skills(vec!["demo".into(), "second".into()]),
    );
    let user_messages = request
        .messages
        .iter()
        .filter(|message| message.role == MessageRole::User)
        .collect::<Vec<_>>();
    assert_eq!(user_messages.len(), 1);
    assert_eq!(
        user_messages[0].content,
        vec![ContentPart::Text("[Skill: demo] [Skill: second]".into())]
    );

    let request = project(crate::user_content::UserMessageContent::new("", vec![]));
    assert!(
        request
            .messages
            .iter()
            .all(|message| message.role != MessageRole::User)
    );
}
