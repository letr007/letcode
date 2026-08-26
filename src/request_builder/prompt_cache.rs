use serde_json::Value;

use crate::config::{ApiProtocol, PromptCacheConfig};

use super::prompt_plan::{PromptPlan, PromptSegment};
use super::{CacheRequestFields, PromptCacheReport, ToolSpec, provider_serialization, sha256_hex};

pub(super) fn cache_request_fields(
    strategy: super::ProviderRequestStrategy,
    protocol: ApiProtocol,
    model_id: &str,
    config: &PromptCacheConfig,
    plan: &PromptPlan,
    tools: &[ToolSpec],
    supports_tools: bool,
    parallel_tool_calls: bool,
) -> CacheRequestFields {
    if !config.enabled || plan.cacheable_prefix_len() == 0 {
        return CacheRequestFields {
            key: None,
            retention: None,
        };
    }
    let namespace = config
        .namespace
        .as_deref()
        .expect("enabled prompt cache has normalized namespace");
    let key = routing_key(
        strategy,
        namespace,
        protocol,
        model_id,
        tools,
        supports_tools,
        parallel_tool_calls,
    );
    CacheRequestFields {
        key: Some(key),
        retention: (protocol == ApiProtocol::Responses)
            .then_some(config.retention)
            .flatten(),
    }
}

pub(super) fn prompt_cache_report(
    strategy: super::ProviderRequestStrategy,
    protocol: ApiProtocol,
    model_id: &str,
    config: &PromptCacheConfig,
    plan: &PromptPlan,
    tools: &[ToolSpec],
    supports_tools: bool,
    parallel_tool_calls: bool,
) -> PromptCacheReport {
    let prefix = plan.cacheable_prefix_len();
    if !config.enabled || prefix == 0 {
        return PromptCacheReport {
            local_prefix_segments: prefix,
            configured: config.enabled,
            ..Default::default()
        };
    }
    let namespace = config
        .namespace
        .as_deref()
        .expect("enabled prompt cache has normalized namespace");
    let canonical_input = canonical_cache_input(
        strategy,
        namespace,
        protocol,
        model_id,
        &plan.segments[..prefix],
        tools,
        supports_tools,
        parallel_tool_calls,
    );
    let routing_key = routing_key_from_canonical_input(&canonical_input);
    let local_prefix_fingerprint =
        format!("ppf-v2-{}", sha256_hex(&canonical_bytes(&canonical_input)));
    PromptCacheReport {
        local_prefix_segments: prefix,
        configured: true,
        hint_serialized: true,
        retention_sent: if protocol == ApiProtocol::Responses {
            config.retention
        } else {
            None
        },
        local_prefix_fingerprint: Some(local_prefix_fingerprint),
        routing_key: Some(routing_key),
    }
}

fn routing_key(
    strategy: super::ProviderRequestStrategy,
    namespace: &str,
    protocol: ApiProtocol,
    model_id: &str,
    tools: &[ToolSpec],
    supports_tools: bool,
    parallel_tool_calls: bool,
) -> String {
    let canonical_input = canonical_cache_input(
        strategy,
        namespace,
        protocol,
        model_id,
        &[],
        tools,
        supports_tools,
        parallel_tool_calls,
    );
    routing_key_from_canonical_input(&canonical_input)
}

pub(super) fn canonical_cache_input(
    strategy: super::ProviderRequestStrategy,
    namespace: &str,
    protocol: ApiProtocol,
    model_id: &str,
    prefix: &[PromptSegment],
    tools: &[ToolSpec],
    supports_tools: bool,
    parallel_tool_calls: bool,
) -> Value {
    let (items, provider_tools, parallel_tool_calls) = match protocol {
        ApiProtocol::Responses => (
            serde_json::to_value(
                prefix
                    .iter()
                    .flat_map(|segment| {
                        provider_serialization::prompt_segment_to_response_inputs(segment, strategy)
                    })
                    .collect::<Vec<_>>(),
            )
            .expect("response input is serializable"),
            serde_json::to_value(supports_tools.then(|| {
                tools
                    .iter()
                    .map(provider_serialization::tool_to_response_tool)
                    .collect::<Vec<_>>()
            }))
            .expect("response tools are serializable"),
            serde_json::to_value(supports_tools.then_some(parallel_tool_calls))
                .expect("parallel tool calls is serializable"),
        ),
        ApiProtocol::Anthropic => (
            serde_json::to_value(
                prefix
                    .iter()
                    .map(|segment| {
                        serde_json::json!({
                            "role": segment.role,
                            "text": segment.text,
                            "content": segment.content,
                        })
                    })
                    .collect::<Vec<_>>(),
            )
            .expect("anthropic canonical input is serializable"),
            serde_json::to_value(supports_tools.then(|| tools))
                .expect("anthropic tools are serializable"),
            Value::Null,
        ),
        ApiProtocol::Completions => (
            serde_json::to_value(
                prefix
                    .iter()
                    .map(provider_serialization::prompt_segment_to_chat_message)
                    .collect::<Vec<_>>(),
            )
            .expect("chat messages are serializable"),
            serde_json::to_value(supports_tools.then(|| {
                tools
                    .iter()
                    .map(provider_serialization::tool_to_chat_tool)
                    .collect::<Vec<_>>()
            }))
            .expect("chat tools are serializable"),
            serde_json::to_value(supports_tools.then_some(parallel_tool_calls))
                .expect("parallel tool calls is serializable"),
        ),
    };
    serde_json::json!({
        "namespace": namespace,
        "shape_version": 2,
        "protocol": protocol,
        "model": model_id,
        "items": items,
        "tools": provider_tools,
        "input_shape": { "parallel_tool_calls": parallel_tool_calls },
    })
}

fn routing_key_from_canonical_input(input: &Value) -> String {
    let Value::Object(values) = input else {
        unreachable!("canonical cache input is an object");
    };
    let routing_input = serde_json::json!({
        "namespace": values["namespace"],
        "shape_version": values["shape_version"],
        "protocol": values["protocol"],
        "model": values["model"],
        "tools": values["tools"],
        "input_shape": values["input_shape"],
    });
    format!(
        "lc-pc-v2-{}",
        &sha256_hex(&canonical_bytes(&routing_input))[..32]
    )
}

pub(super) fn canonical_bytes(value: &Value) -> Vec<u8> {
    fn append(out: &mut Vec<u8>, tag: u8, bytes: &[u8]) {
        out.push(tag);
        out.extend_from_slice(&(bytes.len() as u64).to_be_bytes());
        out.extend_from_slice(bytes);
    }
    fn visit(out: &mut Vec<u8>, value: &Value) {
        match value {
            Value::Null => append(out, b'n', b""),
            Value::Bool(value) => append(out, b'b', if *value { b"1" } else { b"0" }),
            Value::Number(value) => append(out, b'#', value.to_string().as_bytes()),
            Value::String(value) => append(out, b's', value.as_bytes()),
            Value::Array(values) => {
                append(out, b'[', &(values.len() as u64).to_be_bytes());
                for value in values {
                    visit(out, value);
                }
            }
            Value::Object(values) => {
                append(out, b'{', &(values.len() as u64).to_be_bytes());
                let mut keys = values.keys().collect::<Vec<_>>();
                keys.sort();
                for key in keys {
                    append(out, b'k', key.as_bytes());
                    visit(out, &values[key]);
                }
            }
        }
    }
    let mut out = Vec::new();
    visit(&mut out, value);
    out
}
