use super::*;

#[test]
fn historian_preflight_uses_native_images_and_readable_text_in_all_protocols() {
    use crate::user_content::UserImageAttachment;
    use serde_json::{Value, json};
    let image = UserImageAttachment::from_bytes("user image", "image/png", &vec![37; 120_000]);
    let tool_image = UserImageAttachment::from_bytes("tool image", "image/png", b"tool-image-data");
    let replay = crate::model_runtime::OpaqueReplayState::from_anthropic_thinking_blocks_json(
        &json!([{"type":"redacted_thinking","data":"OPAQUE-ONLY".repeat(20_000)}]).to_string(),
    )
    .unwrap();
    let history = vec![
        HistoryItem::user_content(UserMessageContent::new("USER-TEXT", vec![image.clone()])),
        HistoryItem::AssistantTurn {
            text: Some("ASSISTANT-TEXT".into()),
            reasoning_content: Some("READABLE-REASONING".into()),
            replay: Some(replay),
            calls: vec![],
        },
        HistoryItem::ToolOutput {
            call_id: "image-call".into(),
            output_json: "TOOL-TEXT".into(),
            images: vec![tool_image.clone()],
        },
    ];
    let content = crate::historian::history_input(&history, &[], &[]);
    let prelude = [
        PromptMessage::system(crate::historian::HISTORIAN_PROMPT),
        PromptMessage::developer("DYNAMIC-CONTEXT"),
    ];
    for protocol in ["responses", "completions", "anthropic"] {
        for input_images in [true, false] {
            let catalog = crate::model_runtime::RuntimeConfig::from_toml(&format!(
                r#"active_provider="p"
[providers.p]
protocol="{protocol}"
default_model="m"
[providers.p.auth]
type="none"
[providers.p.endpoints]
base_url="http://127.0.0.1:1"
[providers.p.models.m.capabilities]
input_images={input_images}
generation={{max_output_tokens=true}}
[providers.p.models.m.generation]
max_output_tokens=128
"#
            ))
            .unwrap()
            .resolve(&crate::model_runtime::ProtocolRegistry::builtins())
            .unwrap();
            let route = catalog.route("p", "m").unwrap();
            let metadata = ModelRequestMetadata {
                context_window: Some(32_768),
                max_output_tokens: Some(128),
                supports_reasoning: false,
                supports_tools: false,
                parallel_tool_calls: false,
                ..Default::default()
            };
            let result = protocol_stream::prepare_resolved_oneshot_request(
                route,
                metadata.clone(),
                &prelude,
                &content,
                None,
            );
            if !input_images {
                let (build, input) = result.unwrap();
                let placeholders = UserMessageContent::from_parts(
                    content
                        .parts()
                        .into_iter()
                        .map(|part| match part {
                            crate::user_content::UserMessagePart::Image { attachment } => {
                                crate::user_content::UserMessagePart::Text {
                                    text: attachment.placeholder_summary(),
                                }
                            }
                            part => part,
                        })
                        .collect(),
                );
                let (text_build, _) = protocol_stream::prepare_resolved_oneshot_request(
                    route,
                    metadata,
                    &prelude,
                    &placeholders,
                    None,
                )
                .unwrap();
                assert_eq!(
                    build.budget.estimated_request_tokens,
                    text_build.budget.estimated_request_tokens
                );
                assert!(
                    input
                        .messages
                        .iter()
                        .all(|message| message.content.iter().all(|part| {
                            !matches!(part, crate::model_runtime::ContentPart::Image { .. })
                        }))
                );
                route.binding.prepare_request(&input).unwrap();
                continue;
            }
            let (build, input) = result.unwrap();
            assert!(!build.budget.truncated);
            assert!(build.budget.estimated_request_tokens < 20_000);
            assert!(
                build.budget.estimated_request_tokens
                    >= image.visual_token_charge() + tool_image.visual_token_charge()
            );
            let mut larger_history = history.clone();
            if let HistoryItem::UserMessage { content } = &mut larger_history[0] {
                *content = UserMessageContent::new(
                    "USER-TEXT",
                    vec![UserImageAttachment::from_bytes(
                        "user image",
                        "image/png",
                        &vec![38; 240_000],
                    )],
                );
            }
            let larger = crate::historian::history_input(&larger_history, &[], &[]);
            let (larger_build, _) = protocol_stream::prepare_resolved_oneshot_request(
                route,
                metadata.clone(),
                &prelude,
                &larger,
                None,
            )
            .unwrap();
            assert_eq!(
                build.budget.estimated_request_tokens, larger_build.budget.estimated_request_tokens,
                "encoded image bytes must not become text tokens"
            );
            let journal_text = json!({"new_messages":history}).to_string();
            assert!(
                protocol_stream::preflight_resolved_oneshot_text_request(
                    route,
                    metadata.clone(),
                    &prelude,
                    &journal_text
                )
                .is_err()
            );
            let prepared = route.binding.prepare_request(&input).unwrap();
            let body: Value = serde_json::from_slice(&prepared.body).unwrap();
            let messages = if protocol == "responses" {
                &body["input"]
            } else {
                &body["messages"]
            };
            let blocks: Vec<_> = messages
                .as_array()
                .unwrap()
                .iter()
                .filter(|message| message["role"] == "user")
                .flat_map(|message| message["content"].as_array().unwrap())
                .collect();
            let texts: Vec<_> = blocks
                .iter()
                .filter_map(|part| part["text"].as_str())
                .collect();
            let text = texts.join("\n");
            for marker in [
                "USER-TEXT",
                "ASSISTANT-TEXT",
                "READABLE-REASONING",
                "TOOL-TEXT",
            ] {
                assert!(text.contains(marker));
            }
            assert!(!text.contains("OPAQUE-ONLY"));
            assert!(!text.contains("data:image/png;base64,"));
            let payload: Value = serde_json::from_str(texts[0]).unwrap();
            assert_eq!(
                payload["new_messages"][0]["content"]["parts"][1]["attachment"]["attachment_index"],
                0
            );
            assert_eq!(
                payload["new_messages"][2]["content"]["images"][0]["attachment_index"],
                1
            );
            let images: Vec<_> = blocks
                .iter()
                .filter_map(|part| match protocol {
                    "responses" => part["image_url"].as_str().map(str::to_string),
                    "completions" => part["image_url"]["url"].as_str().map(str::to_string),
                    _ => part["source"]["data"]
                        .as_str()
                        .map(|data| format!("data:image/png;base64,{data}")),
                })
                .collect();
            assert_eq!(
                images,
                vec![image.data_url.clone(), tool_image.data_url.clone()]
            );
            assert_eq!(body.to_string().matches("You are Historian").count(), 1);
            assert!(!text.contains("You are Historian"));
            let system = match protocol {
                "responses" => body["instructions"].to_string(),
                "completions" => body["messages"][0]["content"].to_string(),
                _ => body["system"].to_string(),
            };
            assert!(system.contains("You are Historian"));
            assert!(!system.contains("USER-TEXT"));
            assert!(!system.contains("READABLE-REASONING"));
            if protocol != "anthropic" {
                assert!(!system.contains("DYNAMIC-CONTEXT"));
            }
            assert!(body.get("tools").is_none() || body["tools"].as_array().unwrap().is_empty());
        }
    }
}

#[test]
fn three_protocols_replay_frozen_slots_and_keep_baseline_before_delta() {
    for protocol in ["responses", "completions", "anthropic"] {
        let config = crate::model_runtime::RuntimeConfig::from_toml(&format!(
            r#"active_provider="p"
[providers.p]
protocol="{protocol}"
default_model="m"
[providers.p.auth]
type="none"
[providers.p.endpoints]
base_url="http://127.0.0.1:1"
[providers.p.models.m]
[providers.p.models.m.capabilities]
generation={{max_output_tokens=true}}
[providers.p.models.m.generation]
max_output_tokens=128
"#
        ))
        .unwrap()
        .resolve(&crate::model_runtime::ProtocolRegistry::builtins())
        .unwrap();
        let route = config.route("p", "m").unwrap();
        let model = ModelRequestMetadata {
            context_window: Some(8192),
            max_output_tokens: Some(128),
            supports_reasoning: false,
            supports_tools: false,
            ..Default::default()
        };
        let prepare = |delta: &str| {
            let history = vec![
                HistoryItem::context_summary("[Session history]\nFROZEN-BASELINE"),
                HistoryItem::context_summary(format!("[Session history]\n{delta}")),
                HistoryItem::user("new request"),
            ];
            let prelude = vec![
                PromptMessage::system("SYSTEM-AUTHORITY"),
                PromptMessage::developer("DYNAMIC-CONTEXT"),
            ];
            let build = crate::request_builder::build_test_request(
                crate::request_builder::TestRequestBuilderInput {
                    model_id: "m",
                    model: model.clone(),
                    prelude: &prelude,
                    history: &history,
                    protected_start_index: 2,
                    tools: &[],
                    evidence: &[],
                },
            )
            .unwrap();
            let input = crate::model_runtime::projection::model_request_from_prompt_plan(
                route,
                &model,
                &build.prompt_plan,
                &[],
            )
            .unwrap();
            let request = route.binding.prepare_request(&input).unwrap();
            String::from_utf8(request.body).unwrap()
        };
        let first = prepare("DELTA-ONE");
        assert_eq!(
            first,
            prepare("DELTA-ONE"),
            "{protocol}: defer must replay identically"
        );
        let second = prepare("DELTA-TWO");
        assert!(first.contains("FROZEN-BASELINE") && first.contains("DELTA-ONE"));
        assert_eq!(
            first.split("DELTA-ONE").next(),
            second.split("DELTA-TWO").next(),
            "{protocol}: soft update changed the preceding wire prefix"
        );
        let value: serde_json::Value = serde_json::from_str(&first).unwrap();
        match protocol {
            "responses" => {
                assert!(
                    value["instructions"]
                        .as_str()
                        .unwrap()
                        .contains("SYSTEM-AUTHORITY")
                );
                assert!(
                    !value["instructions"]
                        .as_str()
                        .unwrap()
                        .contains("FROZEN-BASELINE")
                );
            }
            "anthropic" => {
                assert!(value["system"].to_string().contains("SYSTEM-AUTHORITY"));
                assert!(!value["system"].to_string().contains("FROZEN-BASELINE"));
            }
            _ => assert_eq!(first.matches("SYSTEM-AUTHORITY").count(), 1),
        }
    }
}
