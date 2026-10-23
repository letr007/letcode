use super::{
    BindingFlavor, BindingIdentity, ContentPart, ControlSegmentKind, FailureKind, FailurePhase,
    GenerationSupport, MessageRole, ModelEvent, ModelFailure, ModelMessage, ModelRequestInput,
    ModelStreamDecoder, OpaqueReplayState, PreparedHttpRequest, PreparedPromptUnitInspection,
    PreparedRequestCacheInspection, PreparedRequestInspection, ProtocolAdapter, ProtocolBindInput,
    ProtocolBinding, ProtocolId, ReplayProducer, ReplayScope, RetryHint, TerminalStatus,
};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::BTreeMap;
use std::sync::Arc;

/// Default endpoint paths registered with the built-in protocol adapters.
pub(crate) fn default_endpoint_path(protocol: &str) -> Option<&'static str> {
    match protocol {
        "responses" => Some("/responses"),
        "completions" => Some("/chat/completions"),
        "anthropic" => Some("/messages"),
        "jev" => Some("/v1/systemone"),
        _ => None,
    }
}

/// All three built-in protocol adapters are executable.
pub(crate) fn builtins() -> impl IntoIterator<Item = Arc<dyn ProtocolAdapter>> {
    [
        Arc::new(ResponsesAdapter::new()) as Arc<dyn ProtocolAdapter>,
        Arc::new(CompletionsAdapter::new()) as Arc<dyn ProtocolAdapter>,
        Arc::new(AnthropicAdapter::new()) as Arc<dyn ProtocolAdapter>,
    ]
}

fn prepared_body(request: &PreparedHttpRequest) -> Result<Value, ModelFailure> {
    serde_json::from_slice(&request.body).map_err(|error| {
        ModelFailure::new(FailurePhase::Prepare, FailureKind::InvalidRequest)
            .with_code("prepared_request_inspection")
            .with_detail(error.to_string())
    })
}

fn prepared_inspection(
    request_shape: Value,
    prompt_units: Vec<Value>,
    prompt_unit_origins: &[Vec<String>],
    cache: PreparedRequestCacheInspection,
) -> Result<PreparedRequestInspection, ModelFailure> {
    let request_shape = serde_json::to_vec(&request_shape).map_err(|error| {
        ModelFailure::new(FailurePhase::Prepare, FailureKind::Internal)
            .with_code("prepared_request_shape")
            .with_detail(error.to_string())
    })?;
    if prompt_units.len() != prompt_unit_origins.len() {
        return Err(
            ModelFailure::new(FailurePhase::Prepare, FailureKind::Internal)
                .with_code("prepared_request_origin_mismatch"),
        );
    }
    let prompt_units = prompt_units
        .into_iter()
        .zip(prompt_unit_origins.iter())
        .map(|(unit, origins)| {
            let identity = serde_json::to_vec(&unit).map_err(|error| {
                ModelFailure::new(FailurePhase::Prepare, FailureKind::Internal)
                    .with_code("prepared_request_unit")
                    .with_detail(error.to_string())
            })?;
            Ok(PreparedPromptUnitInspection {
                identity,
                semantic_segment_ids: origins.clone(),
            })
        })
        .collect::<Result<Vec<_>, ModelFailure>>()?;
    Ok(PreparedRequestInspection {
        request_shape,
        prompt_units,
        cache,
    })
}

fn stable_request_fingerprint(
    stable_request: Option<&PreparedHttpRequest>,
    hint_serialized: bool,
) -> Option<String> {
    hint_serialized
        .then(|| stable_request.map(|request| crate::request_builder::sha256_hex(&request.body)))?
}

fn strip_anthropic_cache_markers(value: &mut Value) {
    match value {
        Value::Array(values) => values.iter_mut().for_each(strip_anthropic_cache_markers),
        Value::Object(fields) => {
            fields.remove("cache_control");
            fields.values_mut().for_each(strip_anthropic_cache_markers);
        }
        _ => {}
    }
}

fn contains_anthropic_cache_marker(value: &Value) -> bool {
    match value {
        Value::Array(values) => values.iter().any(contains_anthropic_cache_marker),
        Value::Object(fields) => {
            fields
                .get("cache_control")
                .is_some_and(|cache| cache.get("type").and_then(Value::as_str) == Some("ephemeral"))
                || fields.values().any(contains_anthropic_cache_marker)
        }
        _ => false,
    }
}

pub(crate) struct PlaceholderAdapter {
    protocol_id: ProtocolId,
}

impl PlaceholderAdapter {
    pub(crate) fn new(protocol_id: ProtocolId) -> Self {
        Self { protocol_id }
    }
}

impl ProtocolAdapter for PlaceholderAdapter {
    fn protocol_id(&self) -> ProtocolId {
        self.protocol_id.clone()
    }

    fn default_endpoint_path(&self) -> &str {
        default_endpoint_path(self.protocol_id.as_str()).unwrap_or("/")
    }

    fn bind(&self, input: ProtocolBindInput) -> Result<Arc<dyn ProtocolBinding>, ModelFailure> {
        Ok(Arc::new(ValidationOnlyBinding {
            identity: input.binding_identity,
            flavor: input.flavor,
        }))
    }
}

struct ValidationOnlyBinding {
    identity: BindingIdentity,
    flavor: BindingFlavor,
}

impl ProtocolBinding for ValidationOnlyBinding {
    fn binding_identity(&self) -> &BindingIdentity {
        &self.identity
    }

    fn flavor(&self) -> &BindingFlavor {
        &self.flavor
    }

    fn replay_scope(&self) -> ReplayScope {
        ReplayScope::Route
    }

    fn prepare_request(
        &self,
        _input: &ModelRequestInput,
    ) -> Result<PreparedHttpRequest, ModelFailure> {
        Err(
            ModelFailure::new(FailurePhase::Prepare, FailureKind::UnsupportedProtocol)
                .with_code("validation_only_adapter"),
        )
    }

    fn inspect_prepared_request(
        &self,
        _request: &PreparedHttpRequest,
        _stable_request: Option<&PreparedHttpRequest>,
    ) -> Result<PreparedRequestInspection, ModelFailure> {
        Err(
            ModelFailure::new(FailurePhase::Prepare, FailureKind::UnsupportedProtocol)
                .with_code("validation_only_adapter"),
        )
    }

    fn new_decoder(&self) -> Box<dyn ModelStreamDecoder> {
        Box::new(ValidationOnlyDecoder)
    }
}

struct ValidationOnlyDecoder;

impl ModelStreamDecoder for ValidationOnlyDecoder {
    fn push(&mut self, _chunk: &[u8]) -> Result<Vec<ModelEvent>, ModelFailure> {
        Err(
            ModelFailure::new(FailurePhase::Decode, FailureKind::UnsupportedProtocol)
                .with_code("validation_only_adapter"),
        )
    }

    fn finish(&mut self) -> Result<Vec<ModelEvent>, ModelFailure> {
        Err(
            ModelFailure::new(FailurePhase::Finish, FailureKind::UnsupportedProtocol)
                .with_code("validation_only_adapter"),
        )
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CompletionsFlavor {
    Standard,
    DeepSeek,
}

impl CompletionsFlavor {
    fn parse(flavor: &BindingFlavor) -> Option<Self> {
        match flavor.as_str() {
            "standard" => Some(Self::Standard),
            "deepseek" => Some(Self::DeepSeek),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ResponsesFlavor {
    Standard,
    DeepSeek,
}

impl ResponsesFlavor {
    fn parse(flavor: &BindingFlavor) -> Option<Self> {
        match flavor.as_str() {
            "standard" => Some(Self::Standard),
            "deepseek" => Some(Self::DeepSeek),
            _ => None,
        }
    }

    fn namespace(self) -> &'static str {
        match self {
            Self::Standard => "responses.reasoning",
            Self::DeepSeek => "responses.reasoning.deepseek",
        }
    }
}

pub(crate) struct AnthropicAdapter;

impl AnthropicAdapter {
    fn new() -> Self {
        Self
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct AnthropicSettings {
    #[serde(default)]
    anthropic_thinking: Option<AnthropicThinkingSettings>,
    #[serde(default)]
    anthropic_betas: Vec<String>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct AnthropicThinkingSettings {
    mode: String,
    #[serde(default)]
    budget_tokens: Option<u64>,
}

impl ProtocolAdapter for AnthropicAdapter {
    fn protocol_id(&self) -> ProtocolId {
        ProtocolId::new("anthropic").expect("built-in protocol id")
    }
    fn default_endpoint_path(&self) -> &str {
        "/messages"
    }
    fn bind(&self, input: ProtocolBindInput) -> Result<Arc<dyn ProtocolBinding>, ModelFailure> {
        if input.flavor.as_str() != "standard" {
            return Err(
                ModelFailure::new(FailurePhase::Bind, FailureKind::InvalidRequest)
                    .with_code("unsupported_anthropic_flavor"),
            );
        }
        let settings: AnthropicSettings = input
            .protocol_settings
            .value
            .clone()
            .try_into()
            .map_err(|error| {
                ModelFailure::new(FailurePhase::Bind, FailureKind::InvalidRequest)
                    .with_code("invalid_anthropic_settings")
                    .with_detail(error.to_string())
            })?;
        if settings.anthropic_betas.iter().any(|beta| {
            beta.trim().is_empty()
                || beta.trim() != beta
                || beta.len() > 128
                || beta.chars().any(|ch| ch.is_control() || ch == ',')
                || reqwest::header::HeaderValue::from_str(beta).is_err()
        }) || settings
            .anthropic_betas
            .iter()
            .collect::<std::collections::BTreeSet<_>>()
            .len()
            != settings.anthropic_betas.len()
        {
            return Err(
                ModelFailure::new(FailurePhase::Bind, FailureKind::InvalidRequest)
                    .with_code("invalid_anthropic_betas"),
            );
        }
        if let Some(thinking) = &settings.anthropic_thinking
            && (!matches!(thinking.mode.as_str(), "disabled" | "adaptive" | "budget")
                || (thinking.mode == "budget" && thinking.budget_tokens.unwrap_or(0) < 1024)
                || (thinking.mode != "budget" && thinking.budget_tokens.is_some())
                || (thinking.mode != "disabled"
                    && (!input.capabilities.reasoning || !input.generation_support.reasoning)))
        {
            return Err(
                ModelFailure::new(FailurePhase::Bind, FailureKind::InvalidRequest)
                    .with_code("invalid_anthropic_thinking"),
            );
        }
        Ok(Arc::new(AnthropicBinding {
            identity: input.binding_identity,
            flavor: input.flavor,
            endpoint: input.endpoint,
            capabilities: input.capabilities,
            generation_support: input.generation_support,
            settings,
        }))
    }
}

struct AnthropicBinding {
    identity: BindingIdentity,
    flavor: BindingFlavor,
    endpoint: String,
    capabilities: super::RouteCapabilities,
    generation_support: GenerationSupport,
    settings: AnthropicSettings,
}

impl ProtocolBinding for AnthropicBinding {
    fn binding_identity(&self) -> &BindingIdentity {
        &self.identity
    }
    fn flavor(&self) -> &BindingFlavor {
        &self.flavor
    }
    fn replay_scope(&self) -> ReplayScope {
        ReplayScope::Protocol
    }
    fn prepare_request(
        &self,
        input: &ModelRequestInput,
    ) -> Result<PreparedHttpRequest, ModelFailure> {
        if input.generation.max_output_tokens.is_some()
            && !self.generation_support.max_output_tokens
        {
            return Err(unsupported("max_output_tokens", "capability disabled"));
        }
        let max_tokens = input.generation.max_output_tokens.unwrap_or(1024);
        if let Some(thinking) = &self.settings.anthropic_thinking
            && thinking.mode == "budget"
            && thinking.budget_tokens.unwrap_or(0) >= max_tokens as u64
        {
            return Err(unsupported(
                "anthropic_thinking.budget_tokens",
                "must be less than max_tokens",
            ));
        }
        if input.cache.retention.is_some() {
            return Err(unsupported(
                "prompt_cache_retention",
                "Anthropic does not support retention",
            ));
        }
        require_generation(
            input.generation.temperature.is_some(),
            self.generation_support.temperature,
            "temperature",
        )?;
        require_generation(
            input.generation.top_p.is_some(),
            self.generation_support.top_p,
            "top_p",
        )?;
        require_generation(
            !input.generation.stop_sequences.is_empty(),
            self.generation_support.stop_sequences,
            "stop_sequences",
        )?;
        let reasoning_requested =
            input.generation.reasoning.enabled || input.generation.reasoning.effort.is_some();
        if reasoning_requested && self.settings.anthropic_thinking.is_none() {
            return Err(unsupported(
                "reasoning",
                "Anthropic reasoning requires anthropic_thinking settings",
            ));
        }
        if input.generation.parallel_tool_calls.is_some()
            || input.generation.priority_service.is_some()
            || input.generation.summary.is_some()
            || input.generation.verbosity.is_some()
        {
            return Err(unsupported(
                "generation",
                "unsupported Anthropic generation field",
            ));
        }
        if !input.tools.is_empty() && !self.capabilities.tools {
            return Err(unsupported("tools", "capability disabled"));
        }
        let mut system = input
            .segments
            .iter()
            .filter(|segment| {
                matches!(
                    segment.kind,
                    ControlSegmentKind::System | ControlSegmentKind::Developer
                )
            })
            .map(|segment| AnthropicBlock::Text {
                text: segment.text.clone(),
                cache_control: None,
            })
            .collect::<Vec<_>>();
        let mut messages = Vec::new();
        let mut message_indices = Vec::new();
        for (index, message) in input.messages.iter().enumerate() {
            let role = match message.role {
                MessageRole::User => "user",
                MessageRole::Assistant => "assistant",
                MessageRole::Tool => "user",
            };
            let mut content = Vec::new();
            let mut tool_calls = Vec::new();
            for part in &message.content {
                match part {
                    ContentPart::Text(text) => content.push(AnthropicBlock::Text {
                        text: text.clone(),
                        cache_control: None,
                    }),
                    ContentPart::Image { media_type, data } => {
                        if !self.capabilities.input_images {
                            return Err(unsupported("input_images", "capability disabled"));
                        }
                        content.push(AnthropicBlock::Image {
                            source: AnthropicImageSource {
                                source_type: "base64".into(),
                                media_type: media_type.clone(),
                                data: base64_encode(data),
                            },
                            cache_control: None,
                        });
                    }
                    ContentPart::ToolCall {
                        id,
                        name,
                        arguments,
                    } => tool_calls.push(AnthropicBlock::ToolUse {
                        id: id.clone(),
                        name: name.clone(),
                        input: arguments.clone(),
                        cache_control: None,
                    }),
                    ContentPart::ToolResult {
                        id,
                        content: result,
                    } => {
                        if result
                            .iter()
                            .any(|part| matches!(part, ContentPart::Image { .. }))
                            && !self.capabilities.tool_result_images
                        {
                            return Err(unsupported("tool_result_images", "capability disabled"));
                        }
                        let blocks = result
                            .iter()
                            .filter_map(|part| match part {
                                ContentPart::Text(text) => Some(AnthropicBlock::Text {
                                    text: text.clone(),
                                    cache_control: None,
                                }),
                                ContentPart::Image { media_type, data } => {
                                    if !self.capabilities.tool_result_images {
                                        return None;
                                    }
                                    Some(AnthropicBlock::Image {
                                        source: AnthropicImageSource {
                                            source_type: "base64".into(),
                                            media_type: media_type.clone(),
                                            data: base64_encode(data),
                                        },
                                        cache_control: None,
                                    })
                                }
                                _ => None,
                            })
                            .collect();
                        content.push(AnthropicBlock::ToolResult {
                            tool_use_id: id.clone(),
                            content: blocks,
                            is_error: None,
                            cache_control: None,
                        });
                    }
                    ContentPart::Reasoning {
                        replay: Some(replay),
                        ..
                    } if self.accepts_replay(replay) => {
                        content.extend(parse_anthropic_replay(&replay.payload)?);
                    }
                    ContentPart::Reasoning { .. } => {}
                }
            }
            content.extend(tool_calls);
            // Foreign reasoning has no Anthropic wire representation. Keep the
            // canonical history intact, but do not emit an empty assistant turn.
            if message.role == MessageRole::Assistant && content.is_empty() {
                continue;
            }
            message_indices.push(index);
            messages.push(AnthropicMessage { role, content });
        }
        if input.cache.enabled {
            let prefix_count = input
                .cache
                .stable_prefix
                .as_ref()
                .map(|prefix| prefix.segment_count)
                .unwrap_or(0);
            if prefix_count > 0 && !system.is_empty() {
                let system_index = prefix_count.min(system.len()) - 1;
                system[system_index].set_cache();
            }
            if prefix_count > system.len() {
                let message_index = prefix_count - system.len() - 1;
                if let Some(retained_index) = message_indices
                    .iter()
                    .rposition(|index| *index <= message_index)
                    && let Some(message) = messages.get_mut(retained_index)
                    && let Some(last) = message.content.last_mut()
                    && !last.set_cache()
                {
                    return Err(unsupported(
                        "prompt_cache",
                        "stable prefix ends at an Anthropic replay block",
                    ));
                }
            }
        }
        let settings = self.settings.anthropic_thinking.as_ref();
        if reasoning_requested && settings.is_some_and(|settings| settings.mode == "disabled") {
            return Err(unsupported(
                "reasoning",
                "Anthropic thinking is disabled for this route",
            ));
        }
        if input.generation.reasoning.effort.is_some()
            && settings.is_some_and(|settings| settings.mode != "adaptive")
        {
            return Err(unsupported(
                "reasoning_effort",
                "Anthropic reasoning effort requires adaptive thinking",
            ));
        }
        let thinking = settings.and_then(|value| match value.mode.as_str() {
            "adaptive" => Some(AnthropicThinking {
                kind: "adaptive",
                budget_tokens: None,
            }),
            "budget" => Some(AnthropicThinking {
                kind: "enabled",
                budget_tokens: value.budget_tokens,
            }),
            _ => None,
        });
        let mut tools = input
            .tools
            .iter()
            .map(|tool| AnthropicTool {
                name: tool.name.clone(),
                description: tool.description.clone(),
                input_schema: super::tool_schema::project(
                    &tool.parameters,
                    super::tool_schema::ToolSchemaDialect::Anthropic,
                ),
                strict: tool.strict,
                cache_control: None,
            })
            .collect::<Vec<_>>();
        if input.cache.enabled
            && let Some(last) = tools.last_mut()
        {
            last.cache_control = Some(AnthropicCacheControl { kind: "ephemeral" });
        }
        let effort = settings
            .filter(|settings| settings.mode == "adaptive")
            .map(|_| anthropic_reasoning_effort(input.generation.reasoning.effort.as_ref()))
            .transpose()?;
        let format = match input.generation.structured_output.as_ref() {
            None => None,
            Some(super::StructuredOutput::JsonSchema(schema)) => {
                require_declared_structured_output(
                    super::StructuredOutputSupport::JsonSchema,
                    self.generation_support.structured_output,
                )?;
                Some(AnthropicOutputFormat::JsonSchema {
                    schema: schema.schema.clone(),
                })
            }
            Some(super::StructuredOutput::JsonObject) => {
                return Err(unsupported(
                    "structured_output",
                    "Anthropic can express json_schema only",
                ));
            }
        };
        let output_config = (effort.is_some() || format.is_some())
            .then_some(AnthropicOutputConfig { effort, format });
        let request = AnthropicRequest {
            model: input.control.model.clone(),
            max_tokens,
            system,
            messages,
            tools,
            thinking,
            output_config,
            temperature: input.generation.temperature,
            top_p: input.generation.top_p,
            stop_sequences: input.generation.stop_sequences.clone(),
            stream: input.control.stream,
        };
        let body = serde_json::to_vec(&request).map_err(|error| {
            ModelFailure::new(FailurePhase::Prepare, FailureKind::Internal)
                .with_detail(error.to_string())
        })?;
        let mut headers = BTreeMap::new();
        headers.insert("accept".into(), "text/event-stream".into());
        headers.insert("content-type".into(), "application/json".into());
        headers.insert("anthropic-version".into(), "2023-06-01".into());
        if !self.settings.anthropic_betas.is_empty() {
            headers.insert(
                "anthropic-beta".into(),
                self.settings.anthropic_betas.join(","),
            );
        }
        Ok(PreparedHttpRequest {
            method: super::HttpMethod::Post,
            url: self.endpoint.clone(),
            protocol_headers: headers,
            body,
            prompt_unit_origins: input
                .segment_origins
                .iter()
                .chain(
                    message_indices
                        .iter()
                        .filter_map(|index| input.message_origins.get(*index)),
                )
                .map(|origin| vec![origin.clone()])
                .collect(),
        })
    }

    fn inspect_prepared_request(
        &self,
        request: &PreparedHttpRequest,
        stable_request: Option<&PreparedHttpRequest>,
    ) -> Result<PreparedRequestInspection, ModelFailure> {
        let mut body = prepared_body(request)?;
        let hint_serialized = contains_anthropic_cache_marker(&body);
        let fields = body.as_object_mut().ok_or_else(|| {
            ModelFailure::new(FailurePhase::Prepare, FailureKind::InvalidRequest)
                .with_code("prepared_request_shape")
        })?;
        let mut prompt_units = fields
            .remove("system")
            .and_then(|value| value.as_array().cloned())
            .unwrap_or_default();
        prompt_units.extend(
            fields
                .remove("messages")
                .and_then(|value| value.as_array().cloned())
                .unwrap_or_default(),
        );
        prompt_units
            .iter_mut()
            .for_each(strip_anthropic_cache_markers);
        strip_anthropic_cache_markers(&mut body);
        prepared_inspection(
            body,
            prompt_units,
            &request.prompt_unit_origins,
            PreparedRequestCacheInspection {
                hint_serialized,
                retention_sent: None,
                local_prefix_fingerprint: stable_request_fingerprint(
                    stable_request,
                    hint_serialized,
                ),
                routing_key: None,
            },
        )
    }

    fn new_decoder(&self) -> Box<dyn ModelStreamDecoder> {
        Box::new(AnthropicDecoder::new())
    }
}

impl AnthropicBinding {
    fn accepts_replay(&self, replay: &OpaqueReplayState) -> bool {
        replay.namespace == "anthropic.thinking_blocks"
            && replay.version == 1
            && replay.producer.scope == ReplayScope::Protocol
            && replay.producer.protocol_id.as_str() == "anthropic"
            && replay.producer.profile_identity.is_none()
            && replay.producer.route_identity.is_none()
    }
}

#[derive(Debug, Serialize)]
struct AnthropicRequest {
    model: String,
    max_tokens: u32,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    system: Vec<AnthropicBlock>,
    messages: Vec<AnthropicMessage>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    tools: Vec<AnthropicTool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    thinking: Option<AnthropicThinking>,
    #[serde(skip_serializing_if = "Option::is_none")]
    output_config: Option<AnthropicOutputConfig>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    top_p: Option<f32>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    stop_sequences: Vec<String>,
    stream: bool,
}
#[derive(Debug, Serialize)]
struct AnthropicThinking {
    #[serde(rename = "type")]
    kind: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    budget_tokens: Option<u64>,
}

#[derive(Debug, Serialize)]
struct AnthropicOutputConfig {
    #[serde(skip_serializing_if = "Option::is_none")]
    effort: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    format: Option<AnthropicOutputFormat>,
}

/// Anthropic carries the schema alone: the OpenAI name/strict envelope is absent.
#[derive(Debug, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum AnthropicOutputFormat {
    JsonSchema { schema: Value },
}
#[derive(Debug, Serialize)]
struct AnthropicMessage {
    role: &'static str,
    content: Vec<AnthropicBlock>,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
struct AnthropicImageSource {
    #[serde(rename = "type")]
    source_type: String,
    media_type: String,
    data: String,
}
#[derive(Debug, Serialize)]
struct AnthropicTool {
    name: String,
    description: String,
    input_schema: Value,
    strict: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    cache_control: Option<AnthropicCacheControl>,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
struct AnthropicCacheControl {
    #[serde(rename = "type")]
    kind: &'static str,
}
#[derive(Debug, Clone, Serialize)]
#[serde(tag = "type")]
enum AnthropicBlock {
    #[serde(rename = "text")]
    Text {
        text: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        cache_control: Option<AnthropicCacheControl>,
    },
    #[serde(rename = "image")]
    Image {
        source: AnthropicImageSource,
        #[serde(skip_serializing_if = "Option::is_none")]
        cache_control: Option<AnthropicCacheControl>,
    },
    #[serde(rename = "tool_use")]
    ToolUse {
        id: String,
        name: String,
        input: Value,
        #[serde(skip_serializing_if = "Option::is_none")]
        cache_control: Option<AnthropicCacheControl>,
    },
    #[serde(rename = "tool_result")]
    ToolResult {
        tool_use_id: String,
        content: Vec<AnthropicBlock>,
        #[serde(skip_serializing_if = "Option::is_none")]
        is_error: Option<bool>,
        #[serde(skip_serializing_if = "Option::is_none")]
        cache_control: Option<AnthropicCacheControl>,
    },
    #[serde(rename = "thinking")]
    Thinking {
        thinking: String,
        signature: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        cache_control: Option<AnthropicCacheControl>,
    },
    #[serde(rename = "redacted_thinking")]
    RedactedThinking {
        data: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        cache_control: Option<AnthropicCacheControl>,
    },
}
impl AnthropicBlock {
    fn set_cache(&mut self) -> bool {
        let cache = Some(AnthropicCacheControl { kind: "ephemeral" });
        match self {
            Self::Text { cache_control, .. }
            | Self::Image { cache_control, .. }
            | Self::ToolUse { cache_control, .. }
            | Self::ToolResult { cache_control, .. } => {
                *cache_control = cache;
                true
            }
            Self::Thinking { .. } | Self::RedactedThinking { .. } => false,
        }
    }
}

fn parse_anthropic_replay(payload: &Value) -> Result<Vec<AnthropicBlock>, ModelFailure> {
    let items = payload
        .as_array()
        .ok_or_else(|| unsupported("replay", "Anthropic replay payload must be an array"))?;
    items
        .iter()
        .map(|item| {
            let block_type = item.get("type").and_then(Value::as_str).unwrap_or_default();
            if !matches!(block_type, "thinking" | "redacted_thinking") {
                return Err(unsupported(
                    "replay",
                    "invalid Anthropic thinking replay block",
                ));
            }
            match block_type {
                "thinking" => {
                    let thinking = item
                        .get("thinking")
                        .and_then(Value::as_str)
                        .filter(|value| !value.is_empty())
                        .ok_or_else(|| unsupported("replay", "thinking text is required"))?;
                    let signature = item
                        .get("signature")
                        .and_then(Value::as_str)
                        .filter(|value| !value.is_empty())
                        .ok_or_else(|| unsupported("replay", "thinking signature is required"))?;
                    Ok(AnthropicBlock::Thinking {
                        thinking: thinking.to_owned(),
                        signature: signature.to_owned(),
                        cache_control: None,
                    })
                }
                "redacted_thinking" => {
                    let data = item
                        .get("data")
                        .and_then(Value::as_str)
                        .filter(|value| !value.is_empty())
                        .ok_or_else(|| unsupported("replay", "redacted data is required"))?;
                    Ok(AnthropicBlock::RedactedThinking {
                        data: data.to_owned(),
                        cache_control: None,
                    })
                }
                _ => unreachable!(),
            }
        })
        .collect()
}

struct AnthropicDecoder {
    framer: IncrementalSseFramer,
    blocks: BTreeMap<usize, AnthropicBlockState>,
    usage: Option<AnthropicUsage>,
    stop_reason: Option<String>,
    response_id: Option<String>,
    message_started: bool,
    message_stopped: bool,
    stop_announced: bool,
    finished: bool,
}

struct AnthropicBlockState {
    kind: String,
    id: Option<String>,
    name: Option<String>,
    text: String,
    signature: String,
    redacted_data: String,
    arguments: String,
    accepts_argument_deltas: bool,
    initial_empty_input: bool,
    done: bool,
}

impl AnthropicDecoder {
    fn new() -> Self {
        Self {
            framer: IncrementalSseFramer::new(),
            blocks: BTreeMap::new(),
            usage: None,
            stop_reason: None,
            response_id: None,
            message_started: false,
            message_stopped: false,
            stop_announced: false,
            finished: false,
        }
    }

    fn consume(&mut self, events: Vec<SseEvent>) -> Result<Vec<ModelEvent>, ModelFailure> {
        let mut output = Vec::new();
        for event in events {
            if self.finished {
                return Err(anthropic_invalid("event received after terminal"));
            }
            let parsed: AnthropicEvent = serde_json::from_str(&event.data).map_err(|error| {
                ModelFailure::new(FailurePhase::Decode, FailureKind::MalformedResponse)
                    .with_code("malformed_json")
                    .with_detail(error.to_string())
            })?;
            match parsed.event_type.as_str() {
                "ping" => {}
                "message_start" => {
                    if self.message_started {
                        return Err(anthropic_invalid("duplicate message_start"));
                    }
                    self.message_started = true;
                    if let Some(message) = parsed.message {
                        if let Some(id) = message.id.filter(|id| !id.is_empty()) {
                            self.response_id = Some(id.clone());
                            output.push(ModelEvent::ResponseMetadata { response_id: id });
                        }
                        if let Some(usage) = message.usage {
                            self.usage
                                .get_or_insert_with(AnthropicUsage::default)
                                .merge(usage);
                        }
                    }
                }
                "content_block_start" => {
                    if self.stop_announced {
                        return Err(anthropic_invalid("content block after stop reason"));
                    }
                    if !self.message_started {
                        return Err(anthropic_invalid(
                            "message_start is required before content blocks",
                        ));
                    }
                    let index = parsed
                        .index
                        .ok_or_else(|| anthropic_invalid("content block index is required"))?;
                    let block = parsed
                        .content_block
                        .ok_or_else(|| anthropic_invalid("content block is required"))?;
                    let kind = block.block_type.clone();
                    let initial_input = block.input.clone();
                    if kind == "tool_use"
                        && initial_input
                            .as_ref()
                            .is_some_and(|value| !value.is_object())
                    {
                        return Err(anthropic_invalid("tool input must be an object"));
                    }
                    let state = AnthropicBlockState {
                        kind: kind.clone(),
                        id: block.id.clone(),
                        name: block.name.clone(),
                        text: block
                            .text
                            .clone()
                            .or(block.thinking.clone())
                            .unwrap_or_default(),
                        signature: block.signature.clone().unwrap_or_default(),
                        redacted_data: block.data.clone().unwrap_or_default(),
                        arguments: initial_input
                            .as_ref()
                            .filter(|value| value.is_object())
                            .map(Value::to_string)
                            .unwrap_or_default(),
                        accepts_argument_deltas: initial_input.as_ref().is_none_or(|value| {
                            value.as_object().is_some_and(|object| object.is_empty())
                        }),
                        initial_empty_input: initial_input.as_ref().is_some_and(|value| {
                            value.as_object().is_some_and(|object| object.is_empty())
                        }),
                        done: false,
                    };
                    if self.blocks.insert(index, state).is_some() {
                        return Err(anthropic_invalid("duplicate content block"));
                    }
                    match kind.as_str() {
                        "text" => {
                            if let Some(text) = block.text.filter(|text| !text.is_empty()) {
                                output.push(ModelEvent::TextDelta { text });
                            }
                        }
                        "thinking" | "redacted_thinking" => {
                            let item_id = block.id.unwrap_or_else(|| format!("thinking-{index}"));
                            output.push(ModelEvent::ReasoningStarted {
                                item_id: item_id.clone(),
                            });
                            if let Some(text) = block.thinking.filter(|text| !text.is_empty()) {
                                output.push(ModelEvent::ReasoningDelta { item_id, text });
                            }
                        }
                        "tool_use" => {
                            let id = block
                                .id
                                .ok_or_else(|| anthropic_invalid("tool id is required"))?;
                            let name = block
                                .name
                                .ok_or_else(|| anthropic_invalid("tool name is required"))?;
                            output.push(ModelEvent::ToolStarted {
                                id: id.clone(),
                                name,
                            });
                            if let Some(input) = block.input.filter(|value| {
                                value.as_object().is_some_and(|object| !object.is_empty())
                            }) {
                                output.push(ModelEvent::ToolArgumentsDelta {
                                    id,
                                    delta: input.to_string(),
                                });
                            }
                        }
                        _ => return Err(anthropic_invalid("unknown content block type")),
                    }
                }
                "content_block_delta" => {
                    if self.stop_announced {
                        return Err(anthropic_invalid("content delta after stop reason"));
                    }
                    let index = parsed
                        .index
                        .ok_or_else(|| anthropic_invalid("content block index is required"))?;
                    let state = self
                        .blocks
                        .get_mut(&index)
                        .ok_or_else(|| anthropic_invalid("content block was not started"))?;
                    if state.done {
                        return Err(anthropic_invalid("content block is already complete"));
                    }
                    let delta = parsed
                        .delta
                        .ok_or_else(|| anthropic_invalid("content block delta is required"))?;
                    match delta.delta_type.as_str() {
                        "text_delta" => {
                            if state.kind != "text" {
                                return Err(anthropic_invalid("text delta kind mismatch"));
                            }
                            let text = delta.text.unwrap_or_default();
                            output.push(ModelEvent::TextDelta { text });
                        }
                        "thinking_delta" => {
                            if state.kind != "thinking" {
                                return Err(anthropic_invalid("thinking delta kind mismatch"));
                            }
                            let text = delta.thinking.unwrap_or_default();
                            state.text.push_str(&text);
                            output.push(ModelEvent::ReasoningDelta {
                                item_id: state
                                    .id
                                    .clone()
                                    .unwrap_or_else(|| format!("thinking-{index}")),
                                text,
                            });
                        }
                        "signature_delta" => {
                            if state.kind != "thinking" {
                                return Err(anthropic_invalid("signature delta kind mismatch"));
                            }
                            state
                                .signature
                                .push_str(&delta.signature.unwrap_or_default());
                        }
                        "input_json_delta" => {
                            if state.kind != "tool_use" {
                                return Err(anthropic_invalid("input json delta kind mismatch"));
                            }
                            if !state.accepts_argument_deltas {
                                return Err(anthropic_invalid(
                                    "tool input delta follows complete initial input",
                                ));
                            }
                            if state.initial_empty_input {
                                state.arguments.clear();
                                state.initial_empty_input = false;
                            }
                            let text = delta.partial_json.unwrap_or_default();
                            state.arguments.push_str(&text);
                            output.push(ModelEvent::ToolArgumentsDelta {
                                id: state
                                    .id
                                    .clone()
                                    .ok_or_else(|| anthropic_invalid("tool id is required"))?,
                                delta: text,
                            });
                        }
                        _ => return Err(anthropic_invalid("unknown content delta")),
                    }
                }
                "content_block_stop" => {
                    let index = parsed
                        .index
                        .ok_or_else(|| anthropic_invalid("content block index is required"))?;
                    if let Some(state) = self.blocks.get_mut(&index) {
                        if state.done {
                            return Err(anthropic_invalid("duplicate content block stop"));
                        }
                        state.done = true;
                        if state.kind == "thinking" || state.kind == "redacted_thinking" {
                            let item_id = state
                                .id
                                .clone()
                                .unwrap_or_else(|| format!("thinking-{index}"));
                            let replayable = if state.kind == "thinking" {
                                !state.text.is_empty()
                            } else {
                                !state.redacted_data.is_empty()
                            };
                            let block = if state.kind == "thinking" {
                                serde_json::json!({
                                    "type": "thinking",
                                    "thinking": state.text,
                                    "signature": state.signature,
                                })
                            } else {
                                serde_json::json!({
                                    "type": "redacted_thinking",
                                    "data": state.redacted_data,
                                })
                            };
                            let replay = replayable.then(|| {
                                OpaqueReplayState::new(
                                    "anthropic.thinking_blocks",
                                    1,
                                    ReplayProducer {
                                        scope: ReplayScope::Protocol,
                                        protocol_id: ProtocolId::new("anthropic")
                                            .expect("built-in protocol id"),
                                        profile_identity: None,
                                        route_identity: None,
                                    },
                                    Value::Array(vec![block]),
                                )
                            });
                            output.push(ModelEvent::ReasoningDone {
                                item_id,
                                text: state.text.clone(),
                                replay,
                            });
                        } else if state.kind == "tool_use" {
                            let arguments = if state.arguments.is_empty() {
                                "{}"
                            } else {
                                state.arguments.as_str()
                            };
                            let input: Value =
                                serde_json::from_str(arguments).map_err(|error| {
                                    ModelFailure::new(
                                        FailurePhase::Decode,
                                        FailureKind::MalformedResponse,
                                    )
                                    .with_code("malformed_tool_arguments")
                                    .with_detail(error.to_string())
                                })?;
                            output.push(ModelEvent::ToolDone {
                                id: state
                                    .id
                                    .clone()
                                    .ok_or_else(|| anthropic_invalid("tool id is required"))?,
                                name: state
                                    .name
                                    .clone()
                                    .ok_or_else(|| anthropic_invalid("tool name is required"))?,
                                arguments: input,
                            });
                        }
                    } else {
                        return Err(anthropic_invalid("content block was not started"));
                    }
                }
                "message_delta" => {
                    if !self.message_started || self.message_stopped || self.stop_announced {
                        return Err(anthropic_invalid("message_delta out of order"));
                    }
                    if self.blocks.values().any(|block| !block.done) {
                        return Err(anthropic_invalid(
                            "message_delta before content blocks completed",
                        ));
                    }
                    self.stop_announced = true;
                    if let Some(usage) = parsed.usage {
                        self.usage
                            .get_or_insert_with(AnthropicUsage::default)
                            .merge(usage);
                    }
                    let stop_reason = parsed
                        .stop_reason
                        .or_else(|| parsed.delta.and_then(|delta| delta.stop_reason));
                    if let Some(reason) = stop_reason {
                        if self.blocks.values().any(|block| !block.done) {
                            return Err(anthropic_invalid(
                                "stop reason before content blocks completed",
                            ));
                        }
                        if let Some(previous) = &self.stop_reason
                            && previous != &reason
                        {
                            return Err(anthropic_invalid("mismatched stop reasons"));
                        }
                        self.stop_reason = Some(reason);
                    }
                }
                "message_stop" => {
                    if !self.message_started || self.message_stopped {
                        return Err(anthropic_invalid("invalid message_stop"));
                    }
                    self.message_stopped = true;
                    self.finish_message(&mut output)?;
                }
                "error" => {
                    output.push(ModelEvent::Failure(anthropic_provider_error(parsed.error)));
                    self.finished = true;
                }
                _ => return Err(anthropic_invalid("unknown Anthropic lifecycle event")),
            }
        }
        Ok(output)
    }

    fn finish_message(&mut self, output: &mut Vec<ModelEvent>) -> Result<(), ModelFailure> {
        if !self.message_stopped {
            return Err(anthropic_invalid("message_stop is required"));
        }
        let reason = self
            .stop_reason
            .as_deref()
            .ok_or_else(|| anthropic_invalid("missing stop reason"))?;
        if self.blocks.values().any(|block| !block.done) {
            return Err(anthropic_invalid("incomplete content block"));
        }
        let has_tools = self.blocks.values().any(|block| block.kind == "tool_use");
        if (reason == "tool_use") != has_tools {
            return Err(anthropic_invalid("stop reason does not match tool blocks"));
        }
        let status = match reason {
            "end_turn" | "stop_sequence" => TerminalStatus::Completed,
            "tool_use" => TerminalStatus::ToolUse,
            "pause_turn" => TerminalStatus::Pause,
            "refusal" => TerminalStatus::Refusal,
            "max_tokens" | "model_context_window_exceeded" => TerminalStatus::Length,
            _ => return Err(anthropic_invalid("unknown stop reason")),
        };
        if let Some(usage) = &self.usage {
            output.push(ModelEvent::Usage {
                input_tokens: usage.input_tokens,
                output_tokens: usage.output_tokens,
                total_tokens: usage.input_tokens.saturating_add(usage.output_tokens),
                reasoning_tokens: None,
                cached_input_tokens: usage.cache_read_input_tokens,
            });
            if usage.cache_read_input_tokens.unwrap_or(0) > 0
                || usage.cache_creation_input_tokens.unwrap_or(0) > 0
                || usage.cache_write_input_tokens.unwrap_or(0) > 0
            {
                output.push(ModelEvent::Cache {
                    hit: usage.cache_read_input_tokens.unwrap_or(0) > 0,
                    read_tokens: usage.cache_read_input_tokens.unwrap_or(0),
                    write_tokens: usage
                        .cache_creation_input_tokens
                        .or(usage.cache_write_input_tokens)
                        .unwrap_or(0),
                });
            }
        }
        output.push(ModelEvent::Terminal { status });
        self.finished = true;
        Ok(())
    }
}

impl ModelStreamDecoder for AnthropicDecoder {
    fn push(&mut self, chunk: &[u8]) -> Result<Vec<ModelEvent>, ModelFailure> {
        let events = self
            .framer
            .push(chunk)
            .map_err(|_| anthropic_invalid("invalid_utf8"))?;
        self.consume(events)
    }
    fn finish(&mut self) -> Result<Vec<ModelEvent>, ModelFailure> {
        let events = self
            .framer
            .finish()
            .map_err(|_| anthropic_invalid("invalid_utf8"))?;
        let output = self.consume(events)?;
        if !self.finished {
            return Err(
                ModelFailure::new(FailurePhase::Finish, FailureKind::MalformedResponse)
                    .with_code("missing_message_stop")
                    .with_retry_hint(RetryHint::Retryable),
            );
        }
        Ok(output)
    }
}

fn anthropic_invalid(detail: &str) -> ModelFailure {
    ModelFailure::new(FailurePhase::Decode, FailureKind::MalformedResponse)
        .with_code("invalid_anthropic_event")
        .with_detail(detail)
}

#[derive(Debug, Deserialize, Default)]
struct AnthropicEvent {
    #[serde(rename = "type", default)]
    event_type: String,
    #[serde(default)]
    index: Option<usize>,
    #[serde(default)]
    message: Option<AnthropicMessageStart>,
    #[serde(default)]
    content_block: Option<AnthropicContentBlock>,
    #[serde(default)]
    delta: Option<AnthropicDelta>,
    #[serde(default)]
    usage: Option<AnthropicUsage>,
    #[serde(default)]
    stop_reason: Option<String>,
    #[serde(default)]
    error: Option<AnthropicError>,
}
#[derive(Debug, Deserialize, Default)]
struct AnthropicMessageStart {
    #[serde(default)]
    id: Option<String>,
    #[serde(default)]
    usage: Option<AnthropicUsage>,
}
#[derive(Debug, Deserialize, Default)]
struct AnthropicContentBlock {
    #[serde(rename = "type", default)]
    block_type: String,
    #[serde(default)]
    id: Option<String>,
    #[serde(default)]
    name: Option<String>,
    #[serde(default)]
    text: Option<String>,
    #[serde(default)]
    thinking: Option<String>,
    #[serde(default)]
    signature: Option<String>,
    #[serde(default)]
    data: Option<String>,
    #[serde(default)]
    input: Option<Value>,
}
#[derive(Debug, Deserialize, Default)]
struct AnthropicDelta {
    #[serde(rename = "type", default)]
    delta_type: String,
    #[serde(default)]
    stop_reason: Option<String>,
    #[serde(default)]
    text: Option<String>,
    #[serde(default)]
    thinking: Option<String>,
    #[serde(default)]
    signature: Option<String>,
    #[serde(default)]
    partial_json: Option<String>,
    #[serde(default)]
    data: Option<String>,
}
#[derive(Debug, Deserialize, Default)]
struct AnthropicUsage {
    #[serde(default)]
    input_tokens: u64,
    #[serde(default)]
    output_tokens: u64,
    #[serde(default)]
    cache_read_input_tokens: Option<u64>,
    #[serde(default)]
    cache_creation_input_tokens: Option<u64>,
    #[serde(default)]
    cache_write_input_tokens: Option<u64>,
}

impl AnthropicUsage {
    fn merge(&mut self, other: Self) {
        if other.input_tokens > 0 {
            self.input_tokens = other.input_tokens;
        }
        if other.output_tokens > 0 {
            self.output_tokens = other.output_tokens;
        }
        if other.cache_read_input_tokens.is_some() {
            self.cache_read_input_tokens = other.cache_read_input_tokens;
        }
        if other.cache_creation_input_tokens.is_some() {
            self.cache_creation_input_tokens = other.cache_creation_input_tokens;
        }
        if other.cache_write_input_tokens.is_some() {
            self.cache_write_input_tokens = other.cache_write_input_tokens;
        }
    }
}
#[derive(Debug, Deserialize, Default)]
struct AnthropicError {
    #[serde(default)]
    r#type: Option<String>,
    #[serde(default)]
    message: Option<String>,
}
fn anthropic_provider_error(error: Option<AnthropicError>) -> ModelFailure {
    let error = error.unwrap_or_default();
    let code = error.r#type.unwrap_or_else(|| "provider_error".into());
    let normalized = code.to_ascii_lowercase();
    let kind = if normalized.contains("rate") {
        FailureKind::RateLimited
    } else if normalized.contains("timeout") {
        FailureKind::Timeout
    } else if normalized.contains("auth") || normalized.contains("permission") {
        FailureKind::Authentication
    } else if normalized.contains("invalid") || normalized.contains("request_too_large") {
        FailureKind::InvalidRequest
    } else {
        FailureKind::Http
    };
    let retryable = matches!(kind, FailureKind::RateLimited | FailureKind::Timeout)
        || matches!(
            normalized.as_str(),
            "overloaded_error" | "server_error" | "api_error"
        );
    ModelFailure::new(FailurePhase::Decode, kind)
        .with_code(code)
        .with_retry_hint(if retryable {
            RetryHint::Retryable
        } else {
            RetryHint::Never
        })
        .with_detail(error.message.unwrap_or_default())
}

pub(crate) struct ResponsesAdapter;

impl ResponsesAdapter {
    fn new() -> Self {
        Self
    }
}

struct CompletionsDecoder {
    framer: IncrementalSseFramer,
    flavor: CompletionsFlavor,
    calls: BTreeMap<usize, CompletionCallState>,
    usage: Option<CompletionsUsage>,
    finish_reason: Option<String>,
    response_id: Option<String>,
    done: bool,
    finished_once: bool,
    reasoning_started: bool,
    reasoning_text: String,
    reasoning_done: bool,
    usage_emitted: bool,
}

struct CompletionCallState {
    id: String,
    name: String,
    arguments: String,
    done: bool,
    emitted: bool,
}

impl CompletionsDecoder {
    fn new(flavor: CompletionsFlavor) -> Self {
        Self {
            framer: IncrementalSseFramer::new(),
            flavor,
            calls: BTreeMap::new(),
            usage: None,
            finish_reason: None,
            response_id: None,
            done: false,
            finished_once: false,
            reasoning_started: false,
            reasoning_text: String::new(),
            reasoning_done: false,
            usage_emitted: false,
        }
    }

    fn consume(&mut self, events: Vec<SseEvent>) -> Result<Vec<ModelEvent>, ModelFailure> {
        let mut output = Vec::new();
        for event in events {
            if self.done {
                return Err(completions_invalid("event received after terminal"));
            }
            if event.data == "[DONE]" {
                if self.finish_reason.is_none() {
                    return Err(completions_invalid("DONE without finish reason"));
                }
                self.done = true;
                self.finish_stream(&mut output)?;
                continue;
            }
            let parsed: CompletionsStreamChunk =
                serde_json::from_str(&event.data).map_err(|error| {
                    ModelFailure::new(FailurePhase::Decode, FailureKind::MalformedResponse)
                        .with_code("malformed_json")
                        .with_detail(error.to_string())
                })?;
            if let Some(id) = parsed.id.filter(|id| !id.is_empty()) {
                if let Some(previous) = &self.response_id
                    && previous != &id
                {
                    return Err(completions_invalid("response id changed"));
                }
                if self.response_id.is_none() {
                    self.response_id = Some(id.clone());
                    output.push(ModelEvent::ResponseMetadata { response_id: id });
                }
            }
            if parsed.choices.iter().any(|choice| choice.index != 0) {
                return Err(completions_invalid("only choice index 0 is supported"));
            }
            if let Some(usage) = parsed.usage {
                self.usage = Some(usage);
            }
            for choice in parsed.choices {
                if let Some(content) = choice.delta.content {
                    output.push(ModelEvent::TextDelta { text: content });
                }
                if let Some(reasoning) = [
                    choice.delta.reasoning_content,
                    choice.delta.reasoning,
                    choice.delta.thinking,
                ]
                .into_iter()
                .flatten()
                .find_map(CompatibleReasoningDelta::into_text)
                {
                    if !self.reasoning_started {
                        self.reasoning_started = true;
                        output.push(ModelEvent::ReasoningStarted {
                            item_id: "reasoning".into(),
                        });
                    }
                    self.reasoning_text.push_str(&reasoning);
                    output.push(ModelEvent::ReasoningDelta {
                        item_id: "reasoning".into(),
                        text: reasoning,
                    });
                }
                if let Some(tool_calls) = choice.delta.tool_calls {
                    for fragment in tool_calls {
                        self.consume_tool(fragment, &mut output)?;
                    }
                }
                if let Some(reason) = choice.finish_reason {
                    if let Some(previous) = &self.finish_reason
                        && previous != &reason
                    {
                        return Err(completions_invalid("mismatched finish reasons"));
                    }
                    self.finish_reason = Some(reason);
                }
            }
        }
        Ok(output)
    }

    fn consume_tool(
        &mut self,
        fragment: CompletionToolFragment,
        output: &mut Vec<ModelEvent>,
    ) -> Result<(), ModelFailure> {
        let CompletionToolFragment {
            index,
            id,
            function,
        } = fragment;
        let (name, arguments) = function
            .map(|function| (function.name, function.arguments))
            .unwrap_or((None, None));
        let state = self
            .calls
            .entry(index)
            .or_insert_with(|| CompletionCallState {
                id: id.clone().unwrap_or_default(),
                name: name.clone().unwrap_or_default(),
                arguments: String::new(),
                done: false,
                emitted: false,
            });
        if let Some(id) = id {
            if !state.id.is_empty() && state.id != id {
                return Err(completions_invalid("tool call id changed"));
            }
            state.id = id;
        }
        if let Some(name) = name {
            if !state.name.is_empty() && state.name != name {
                return Err(completions_invalid("tool call name changed"));
            }
            state.name = name;
        }
        if let Some(arguments) = arguments.as_deref() {
            state.arguments.push_str(arguments);
        }
        if !state.id.is_empty() && !state.name.is_empty() && !state.emitted {
            state.emitted = true;
            output.push(ModelEvent::ToolStarted {
                id: state.id.clone(),
                name: state.name.clone(),
            });
            if !state.arguments.is_empty() {
                output.push(ModelEvent::ToolArgumentsDelta {
                    id: state.id.clone(),
                    delta: state.arguments.clone(),
                });
            }
        } else if state.emitted
            && let Some(arguments) = arguments
            && !arguments.is_empty()
        {
            output.push(ModelEvent::ToolArgumentsDelta {
                id: state.id.clone(),
                delta: arguments,
            });
        }
        Ok(())
    }

    fn finish_stream(&mut self, output: &mut Vec<ModelEvent>) -> Result<(), ModelFailure> {
        if self.finished_once {
            return Ok(());
        }
        let reason = self
            .finish_reason
            .as_deref()
            .ok_or_else(|| completions_invalid("missing finish reason"))?;
        match reason {
            "stop" => {
                if !self.calls.is_empty() {
                    return Err(completions_invalid("stop with tool calls"));
                }
                if self.reasoning_started && !self.reasoning_done {
                    self.reasoning_done = true;
                    output.push(ModelEvent::ReasoningDone {
                        item_id: "reasoning".into(),
                        text: self.reasoning_text.clone(),
                        replay: None,
                    });
                }
                output.push(ModelEvent::Terminal {
                    status: TerminalStatus::Completed,
                });
            }
            "tool_calls" | "function_call" => {
                if self.calls.values().any(|call| {
                    call.id.is_empty()
                        || call.name.is_empty()
                        || call.arguments.is_empty()
                        || serde_json::from_str::<Value>(&call.arguments).is_err()
                }) {
                    return Err(completions_invalid("incomplete tool call"));
                }
                for call in self.calls.values_mut() {
                    if !call.done {
                        let args = serde_json::from_str(&call.arguments)
                            .map_err(|_| completions_invalid("invalid tool arguments"))?;
                        call.done = true;
                        output.push(ModelEvent::ToolDone {
                            id: call.id.clone(),
                            name: call.name.clone(),
                            arguments: args,
                        });
                    }
                }
                if self.reasoning_started && !self.reasoning_done {
                    self.reasoning_done = true;
                    output.push(ModelEvent::ReasoningDone {
                        item_id: "reasoning".into(),
                        text: self.reasoning_text.clone(),
                        replay: None,
                    });
                }
                output.push(ModelEvent::Terminal {
                    status: TerminalStatus::ToolUse,
                });
            }
            "length" | "content_filter" | "refusal" => {
                return Err(completions_invalid("non-success finish reason"));
            }
            _ => return Err(completions_invalid("unknown finish reason")),
        }
        if !self.usage_emitted
            && let Some(usage) = &self.usage
        {
            output.push(ModelEvent::Usage {
                input_tokens: usage.prompt_tokens,
                output_tokens: usage.completion_tokens,
                total_tokens: usage.total_tokens,
                reasoning_tokens: usage
                    .completion_tokens_details
                    .as_ref()
                    .and_then(|d| d.reasoning_tokens),
                cached_input_tokens: usage
                    .prompt_tokens_details
                    .as_ref()
                    .and_then(|d| d.cached_tokens),
            });
            if let Some(details) = &usage.prompt_tokens_details {
                let read_tokens = details.cached_tokens.unwrap_or(0);
                let write_tokens = details.cache_write_tokens.unwrap_or(0);
                if read_tokens > 0 || write_tokens > 0 {
                    output.push(ModelEvent::Cache {
                        hit: read_tokens > 0,
                        read_tokens,
                        write_tokens,
                    });
                }
            }
            self.usage_emitted = true;
        }
        if let Some(index) = output
            .iter()
            .rposition(|event| matches!(event, ModelEvent::Terminal { .. }))
        {
            let terminal = output.remove(index);
            output.push(terminal);
        }
        self.finished_once = true;
        Ok(())
    }
}

impl ModelStreamDecoder for CompletionsDecoder {
    fn push(&mut self, chunk: &[u8]) -> Result<Vec<ModelEvent>, ModelFailure> {
        let events = self
            .framer
            .push(chunk)
            .map_err(|_| completions_invalid("invalid_utf8"))?;
        self.consume(events)
    }
    fn finish(&mut self) -> Result<Vec<ModelEvent>, ModelFailure> {
        let events = self
            .framer
            .finish()
            .map_err(|_| completions_invalid("invalid_utf8"))?;
        let output = self.consume(events)?;
        if self.done && self.finished_once {
            return Ok(output);
        }
        if self.done {
            return Ok(output);
        }
        Err(completions_incomplete("missing DONE marker"))
    }
}

fn completions_invalid(detail: &str) -> ModelFailure {
    ModelFailure::new(FailurePhase::Decode, FailureKind::MalformedResponse)
        .with_code("invalid_completions_event")
        .with_detail(detail)
}

fn completions_incomplete(detail: &str) -> ModelFailure {
    ModelFailure::new(FailurePhase::Decode, FailureKind::MalformedResponse)
        .with_code("incomplete_completions_stream")
        .with_retry_hint(RetryHint::Retryable)
        .with_detail(detail)
}

#[derive(Debug, Deserialize, Default)]
struct CompletionsStreamChunk {
    #[serde(default)]
    id: Option<String>,
    #[serde(default)]
    choices: Vec<CompletionsChoice>,
    #[serde(default)]
    usage: Option<CompletionsUsage>,
}
#[derive(Debug, Deserialize, Default)]
struct CompletionsChoice {
    index: usize,
    #[serde(default)]
    delta: CompletionsDelta,
    #[serde(default)]
    finish_reason: Option<String>,
}
#[derive(Debug, Deserialize, Default)]
struct CompletionsDelta {
    #[serde(default)]
    content: Option<String>,
    #[serde(default)]
    reasoning_content: Option<CompatibleReasoningDelta>,
    #[serde(default)]
    thinking: Option<CompatibleReasoningDelta>,
    #[serde(default)]
    reasoning: Option<CompatibleReasoningDelta>,
    #[serde(default)]
    tool_calls: Option<Vec<CompletionToolFragment>>,
}
#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum CompatibleReasoningDelta {
    Text(String),
    Object {
        content: Option<String>,
        text: Option<String>,
        summary: Option<String>,
    },
    Array(Vec<CompatibleReasoningDelta>),
}

impl CompatibleReasoningDelta {
    fn into_text(self) -> Option<String> {
        match self {
            Self::Text(text) => (!text.is_empty()).then_some(text),
            Self::Object {
                content,
                text,
                summary,
            } => content
                .or(text)
                .or(summary)
                .filter(|value| !value.is_empty()),
            Self::Array(parts) => {
                let text = parts
                    .into_iter()
                    .filter_map(Self::into_text)
                    .collect::<String>();
                (!text.is_empty()).then_some(text)
            }
        }
    }
}

#[derive(Debug, Deserialize, Default)]
struct CompletionToolFragment {
    index: usize,
    #[serde(default)]
    id: Option<String>,
    #[serde(default)]
    function: Option<CompletionFunctionFragment>,
}
#[derive(Debug, Deserialize, Default)]
struct CompletionFunctionFragment {
    #[serde(default)]
    name: Option<String>,
    #[serde(default)]
    arguments: Option<String>,
}
#[derive(Debug, Deserialize, Default)]
struct CompletionsUsage {
    prompt_tokens: u64,
    completion_tokens: u64,
    total_tokens: u64,
    #[serde(default)]
    prompt_tokens_details: Option<CompletionsTokenDetails>,
    #[serde(default)]
    completion_tokens_details: Option<CompletionsCompletionDetails>,
}

#[derive(Debug, Deserialize, Default)]
struct CompletionsCompletionDetails {
    #[serde(default)]
    reasoning_tokens: Option<u64>,
}
#[derive(Debug, Deserialize, Default)]
struct CompletionsTokenDetails {
    #[serde(default)]
    cached_tokens: Option<u64>,
    #[serde(default)]
    cache_write_tokens: Option<u64>,
}

impl ProtocolAdapter for CompletionsAdapter {
    fn protocol_id(&self) -> ProtocolId {
        ProtocolId::new("completions").expect("built-in protocol id is valid")
    }

    fn default_endpoint_path(&self) -> &str {
        "/chat/completions"
    }

    fn bind(&self, input: ProtocolBindInput) -> Result<Arc<dyn ProtocolBinding>, ModelFailure> {
        let flavor = CompletionsFlavor::parse(&input.flavor).ok_or_else(|| {
            ModelFailure::new(FailurePhase::Bind, FailureKind::InvalidRequest)
                .with_code("unsupported_completions_flavor")
        })?;
        Ok(Arc::new(CompletionsBinding {
            identity: input.binding_identity,
            flavor,
            flavor_id: input.flavor,
            endpoint: input.endpoint,
            capabilities: input.capabilities,
            generation_support: input.generation_support,
        }))
    }
}

pub(crate) struct CompletionsAdapter;

impl CompletionsAdapter {
    fn new() -> Self {
        Self
    }
}

struct CompletionsBinding {
    identity: BindingIdentity,
    flavor: CompletionsFlavor,
    flavor_id: BindingFlavor,
    endpoint: String,
    capabilities: super::RouteCapabilities,
    generation_support: GenerationSupport,
}

impl ProtocolBinding for CompletionsBinding {
    fn binding_identity(&self) -> &BindingIdentity {
        &self.identity
    }
    fn flavor(&self) -> &BindingFlavor {
        &self.flavor_id
    }
    fn replay_scope(&self) -> ReplayScope {
        ReplayScope::None
    }

    fn prepare_request(
        &self,
        input: &ModelRequestInput,
    ) -> Result<PreparedHttpRequest, ModelFailure> {
        let (mut request, message_origins) = CompletionsRequest::from_input(self, input)?;
        if self.flavor == CompletionsFlavor::Standard && input.cache.enabled {
            request.prompt_cache_key = Some(cache_key_for_completions(self, input, &request)?);
        }
        let body = serde_json::to_vec(&request).map_err(|error| {
            ModelFailure::new(FailurePhase::Prepare, FailureKind::Internal)
                .with_code("request_serialization")
                .with_detail(error.to_string())
        })?;
        let mut protocol_headers = BTreeMap::new();
        protocol_headers.insert("accept".into(), "text/event-stream".into());
        protocol_headers.insert("content-type".into(), "application/json".into());
        Ok(PreparedHttpRequest {
            method: super::HttpMethod::Post,
            url: self.endpoint.clone(),
            protocol_headers,
            body,
            prompt_unit_origins: input
                .segment_origins
                .iter()
                .map(|origin| vec![origin.clone()])
                .chain(message_origins)
                .collect(),
        })
    }

    fn inspect_prepared_request(
        &self,
        request: &PreparedHttpRequest,
        stable_request: Option<&PreparedHttpRequest>,
    ) -> Result<PreparedRequestInspection, ModelFailure> {
        let mut body = prepared_body(request)?;
        let fields = body.as_object_mut().ok_or_else(|| {
            ModelFailure::new(FailurePhase::Prepare, FailureKind::InvalidRequest)
                .with_code("prepared_request_shape")
        })?;
        let prompt_units = fields
            .remove("messages")
            .and_then(|value| value.as_array().cloned())
            .unwrap_or_default();
        let routing_key = fields
            .remove("prompt_cache_key")
            .and_then(|value| value.as_str().map(str::to_owned))
            .filter(|value| !value.is_empty());
        let hint_serialized = routing_key.is_some();
        prepared_inspection(
            body,
            prompt_units,
            &request.prompt_unit_origins,
            PreparedRequestCacheInspection {
                hint_serialized,
                retention_sent: None,
                local_prefix_fingerprint: stable_request_fingerprint(
                    stable_request,
                    hint_serialized,
                ),
                routing_key,
            },
        )
    }

    fn new_decoder(&self) -> Box<dyn ModelStreamDecoder> {
        Box::new(CompletionsDecoder::new(self.flavor))
    }
}

#[derive(Debug, Serialize)]
struct CompletionsRequest {
    model: String,
    messages: Vec<CompletionsMessage>,
    n: u8,
    stream: bool,
    stream_options: CompletionsStreamOptions,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_completion_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    stop: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    reasoning_effort: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    verbosity: Option<&'static str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    parallel_tool_calls: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    service_tier: Option<&'static str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    thinking: Option<DeepSeekThinking>,
    #[serde(skip_serializing_if = "Option::is_none")]
    prompt_cache_key: Option<String>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    tools: Vec<CompletionsTool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    response_format: Option<CompletionsResponseFormat>,
}

#[derive(Debug, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum CompletionsResponseFormat {
    JsonObject,
    JsonSchema { json_schema: CompletionsJsonSchema },
}

#[derive(Debug, Serialize)]
struct CompletionsJsonSchema {
    name: String,
    schema: Value,
    strict: bool,
}

#[derive(Debug, Serialize)]
struct CompletionsStreamOptions {
    include_usage: bool,
}

#[derive(Debug, Serialize)]
struct DeepSeekThinking {
    #[serde(rename = "type")]
    mode: &'static str,
}

#[derive(Debug, Serialize)]
struct CompletionsMessage {
    role: &'static str,
    content: Option<CompletionsMessageContent>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_calls: Option<Vec<CompletionsToolCall>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_call_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    reasoning_content: Option<String>,
}

#[derive(Debug, Serialize)]
#[serde(untagged)]
enum CompletionsMessageContent {
    Text(String),
    Parts(Vec<CompletionsContent>),
}

#[derive(Debug, Serialize)]
#[serde(tag = "type")]
enum CompletionsContent {
    #[serde(rename = "text")]
    Text { text: String },
    #[serde(rename = "image_url")]
    Image { image_url: CompletionsImageUrl },
}

#[derive(Debug, Serialize)]
struct CompletionsImageUrl {
    url: String,
}

#[derive(Debug, Serialize)]
struct CompletionsToolCall {
    id: String,
    #[serde(rename = "type")]
    call_type: &'static str,
    function: CompletionsFunction,
}

#[derive(Debug, Serialize)]
struct CompletionsFunction {
    name: String,
    arguments: String,
}

#[derive(Debug, Serialize)]
struct CompletionsTool {
    #[serde(rename = "type")]
    tool_type: &'static str,
    function: CompletionsToolFunction,
}

#[derive(Debug, Serialize)]
struct CompletionsToolFunction {
    name: String,
    description: String,
    parameters: Value,
    strict: bool,
}

impl CompletionsRequest {
    fn from_input(
        binding: &CompletionsBinding,
        input: &ModelRequestInput,
    ) -> Result<(Self, Vec<Vec<String>>), ModelFailure> {
        let mut messages = Vec::new();
        let mut message_origins = Vec::new();
        let mut pending_tool_result_images = Vec::new();
        let mut pending_tool_result_origins = Vec::new();
        for segment in &input.segments {
            let role = if binding.flavor == CompletionsFlavor::DeepSeek {
                "system"
            } else {
                match segment.kind {
                    ControlSegmentKind::System => "system",
                    ControlSegmentKind::Developer => "developer",
                }
            };
            messages.push(CompletionsMessage {
                role,
                content: Some(CompletionsMessageContent::Text(segment.text.clone())),
                tool_calls: None,
                tool_call_id: None,
                reasoning_content: None,
            });
        }
        for (index, message) in input.messages.iter().enumerate() {
            if message.role != MessageRole::Tool && !pending_tool_result_images.is_empty() {
                messages.push(CompletionsMessage {
                    role: "user",
                    content: Some(CompletionsMessageContent::Parts(std::mem::take(
                        &mut pending_tool_result_images,
                    ))),
                    tool_calls: None,
                    tool_call_id: None,
                    reasoning_content: None,
                });
                message_origins.push(std::mem::take(&mut pending_tool_result_origins));
            }
            let role = match message.role {
                MessageRole::User => "user",
                MessageRole::Assistant => "assistant",
                MessageRole::Tool => "tool",
            };
            let mut content = Vec::new();
            let mut calls = Vec::new();
            let mut tool_id = None;
            let mut reasoning = String::new();
            let pending_tool_result_images_before = pending_tool_result_images.len();
            for part in &message.content {
                match part {
                    ContentPart::Text(text) => {
                        content.push(CompletionsContent::Text { text: text.clone() })
                    }
                    ContentPart::Image { media_type, data } => {
                        if !binding.capabilities.input_images {
                            return Err(unsupported(
                                "input_images",
                                "input image capability is disabled",
                            ));
                        }
                        content.push(CompletionsContent::Image {
                            image_url: CompletionsImageUrl {
                                url: format!("data:{media_type};base64,{}", base64_encode(data)),
                            },
                        });
                    }
                    ContentPart::Reasoning { text, .. }
                        if binding.flavor == CompletionsFlavor::DeepSeek =>
                    {
                        reasoning.push_str(text)
                    }
                    ContentPart::ToolCall {
                        id,
                        name,
                        arguments,
                    } => calls.push(CompletionsToolCall {
                        id: id.clone(),
                        call_type: "function",
                        function: CompletionsFunction {
                            name: name.clone(),
                            arguments: serde_json::to_string(arguments).map_err(|error| {
                                unsupported("tool_arguments", &error.to_string())
                            })?,
                        },
                    }),
                    ContentPart::ToolResult {
                        id,
                        content: result_content,
                    } => {
                        tool_id = Some(id.clone());
                        for part in result_content {
                            match part {
                                ContentPart::Text(text) => {
                                    content.push(CompletionsContent::Text { text: text.clone() });
                                }
                                ContentPart::Image { media_type, data } => {
                                    if !binding.capabilities.tool_result_images {
                                        return Err(unsupported(
                                            "tool_result_images",
                                            "tool result image capability is disabled",
                                        ));
                                    }
                                    if !binding.capabilities.input_images {
                                        return Err(unsupported(
                                            "input_images",
                                            "tool result image translation requires input image capability",
                                        ));
                                    }
                                    pending_tool_result_images.push(CompletionsContent::Image {
                                        image_url: CompletionsImageUrl {
                                            url: format!(
                                                "data:{media_type};base64,{}",
                                                base64_encode(data)
                                            ),
                                        },
                                    });
                                }
                                _ => {}
                            }
                        }
                    }
                    _ => {}
                }
            }
            if message.role == MessageRole::Tool
                && pending_tool_result_images.len() > pending_tool_result_images_before
                && let Some(origin) = input.message_origins.get(index)
            {
                pending_tool_result_origins.push(origin.clone());
            }
            if message.role == MessageRole::Assistant && content.is_empty() && calls.is_empty() {
                // Completions providers require content or tool calls; reasoning_content
                // alone is not a message.
                tracing::warn!(
                    message_index = index,
                    model = %input.control.model,
                    "omitting assistant message without content or tool calls"
                );
                continue;
            }
            message_origins.push(
                input
                    .message_origins
                    .get(index)
                    .cloned()
                    .into_iter()
                    .collect(),
            );
            if binding.flavor == CompletionsFlavor::DeepSeek
                && message.role == MessageRole::Assistant
                && (!reasoning.is_empty() || !calls.is_empty())
            {
                messages.push(CompletionsMessage {
                    role,
                    content: if content.is_empty() {
                        None
                    } else if content.len() == 1 {
                        match content.into_iter().next().expect("one content item") {
                            CompletionsContent::Text { text } => {
                                Some(CompletionsMessageContent::Text(text))
                            }
                            item => Some(CompletionsMessageContent::Parts(vec![item])),
                        }
                    } else {
                        Some(CompletionsMessageContent::Parts(content))
                    },
                    tool_calls: (!calls.is_empty()).then_some(calls),
                    tool_call_id: tool_id,
                    reasoning_content: Some(reasoning),
                });
            } else {
                messages.push(CompletionsMessage {
                    role,
                    content: if content.is_empty() {
                        None
                    } else if content.len() == 1 {
                        match content.into_iter().next().expect("one content item") {
                            CompletionsContent::Text { text } => {
                                Some(CompletionsMessageContent::Text(text))
                            }
                            item => Some(CompletionsMessageContent::Parts(vec![item])),
                        }
                    } else {
                        Some(CompletionsMessageContent::Parts(content))
                    },
                    tool_calls: (!calls.is_empty()).then_some(calls),
                    tool_call_id: tool_id,
                    reasoning_content: None,
                });
            }
        }
        if !pending_tool_result_images.is_empty() {
            messages.push(CompletionsMessage {
                role: "user",
                content: Some(CompletionsMessageContent::Parts(pending_tool_result_images)),
                tool_calls: None,
                tool_call_id: None,
                reasoning_content: None,
            });
            message_origins.push(pending_tool_result_origins);
        }
        if !input.tools.is_empty() && !binding.capabilities.tools {
            return Err(unsupported("tools", "tools capability is disabled"));
        }
        let tools = input
            .tools
            .iter()
            .map(|tool| CompletionsTool {
                tool_type: "function",
                function: CompletionsToolFunction {
                    name: tool.name.clone(),
                    description: tool.description.clone(),
                    parameters: super::tool_schema::project(
                        &tool.parameters,
                        super::tool_schema::ToolSchemaDialect::OpenAiStrict,
                    ),
                    strict: tool.strict,
                },
            })
            .collect();
        let g = &input.generation;
        if !g.stop_sequences.is_empty()
            && (binding.flavor == CompletionsFlavor::DeepSeek
                || !binding.generation_support.stop_sequences)
        {
            return Err(unsupported(
                "stop_sequences",
                "stop sequence capability is disabled for this Completions route",
            ));
        }
        require_generation(
            g.temperature.is_some(),
            binding.generation_support.temperature,
            "temperature",
        )?;
        require_generation(g.top_p.is_some(), binding.generation_support.top_p, "top_p")?;
        require_generation(
            g.max_output_tokens.is_some(),
            binding.generation_support.max_output_tokens,
            "max_completion_tokens",
        )?;
        let reasoning_requested = g.reasoning.enabled || g.reasoning.effort.is_some();
        if reasoning_requested
            && (!binding.capabilities.reasoning || !binding.generation_support.reasoning)
        {
            return Err(unsupported(
                "reasoning_effort",
                "reasoning capability is disabled",
            ));
        }
        let parallel = g
            .parallel_tool_calls
            .map(|value| {
                if !binding.capabilities.parallel_tool_calls
                    || !binding.generation_support.parallel_tool_calls
                {
                    Err(unsupported(
                        "parallel_tool_calls",
                        "parallel tool-call capability is disabled",
                    ))
                } else {
                    Ok(value)
                }
            })
            .transpose()?;
        if g.verbosity.is_some()
            && (binding.flavor == CompletionsFlavor::DeepSeek
                || !binding.generation_support.text_verbosity)
        {
            return Err(unsupported("verbosity", "verbosity capability is disabled"));
        }
        let reasoning = g
            .reasoning
            .effort
            .as_ref()
            .filter(|_| binding.generation_support.reasoning)
            .map(|effort| reasoning_effort(Some(effort)));
        if g.reasoning.effort.is_some() && reasoning.is_none() {
            return Err(unsupported(
                "reasoning_effort",
                "reasoning capability is disabled",
            ));
        }
        if g.verbosity.is_some() && binding.flavor == CompletionsFlavor::DeepSeek {
            return Err(unsupported(
                "verbosity",
                "DeepSeek does not support verbosity",
            ));
        }
        if input.cache.enabled && binding.flavor == CompletionsFlavor::DeepSeek {
            return Err(unsupported(
                "cache",
                "DeepSeek does not support prompt cache",
            ));
        }
        if input.cache.retention.is_some() {
            return Err(unsupported(
                "prompt_cache_retention",
                "Completions does not support prompt cache retention",
            ));
        }
        if g.priority_service.is_some()
            && (binding.flavor == CompletionsFlavor::DeepSeek
                || !binding.capabilities.priority_service
                || !binding.generation_support.priority_service)
        {
            return Err(unsupported(
                "priority_service",
                "priority service capability is disabled",
            ));
        }
        let response_format = completions_response_format(
            g.structured_output.as_ref(),
            binding.generation_support.structured_output,
        )?;
        let request = Self {
            model: input.control.model.clone(),
            messages,
            n: 1,
            stream: input.control.stream,
            stream_options: CompletionsStreamOptions {
                include_usage: true,
            },
            temperature: g.temperature,
            top_p: g.top_p,
            max_completion_tokens: (binding.flavor == CompletionsFlavor::Standard)
                .then_some(g.max_output_tokens)
                .flatten(),
            max_tokens: (binding.flavor == CompletionsFlavor::DeepSeek)
                .then_some(g.max_output_tokens)
                .flatten(),
            stop: if binding.flavor == CompletionsFlavor::Standard {
                g.stop_sequences.clone()
            } else {
                Vec::new()
            },
            reasoning_effort: if binding.flavor == CompletionsFlavor::DeepSeek {
                g.reasoning
                    .effort
                    .as_ref()
                    .and_then(deepseek_reasoning_effort)
            } else {
                reasoning
            },
            verbosity: (binding.flavor == CompletionsFlavor::Standard)
                .then(|| g.verbosity.map(text_verbosity))
                .flatten(),
            parallel_tool_calls: parallel,
            service_tier: (binding.flavor == CompletionsFlavor::Standard
                && g.priority_service == Some(true))
            .then_some("priority"),
            thinking: (binding.flavor == CompletionsFlavor::DeepSeek && reasoning_requested)
                .then_some(DeepSeekThinking {
                    mode: if matches!(g.reasoning.effort, Some(super::ReasoningEffort::None)) {
                        "disabled"
                    } else {
                        "enabled"
                    },
                }),
            prompt_cache_key: None,
            response_format,
            tools,
        };
        Ok((request, message_origins))
    }
}

impl ProtocolAdapter for ResponsesAdapter {
    fn protocol_id(&self) -> ProtocolId {
        ProtocolId::new("responses").expect("built-in protocol id is valid")
    }

    fn default_endpoint_path(&self) -> &str {
        "/responses"
    }

    fn bind(&self, input: ProtocolBindInput) -> Result<Arc<dyn ProtocolBinding>, ModelFailure> {
        let flavor = ResponsesFlavor::parse(&input.flavor).ok_or_else(|| {
            ModelFailure::new(FailurePhase::Bind, FailureKind::InvalidRequest)
                .with_code("unsupported_responses_flavor")
                .with_detail(format!(
                    "unsupported Responses flavor: {}",
                    input.flavor.as_str()
                ))
        })?;
        Ok(Arc::new(ResponsesBinding {
            identity: input.binding_identity,
            flavor: input.flavor,
            profile: flavor,
            endpoint: input.endpoint,
            capabilities: input.capabilities,
            generation_support: input.generation_support,
            profile_settings: input.profile,
            protocol_settings: input.protocol_settings,
        }))
    }
}

struct ResponsesBinding {
    identity: BindingIdentity,
    flavor: BindingFlavor,
    profile: ResponsesFlavor,
    endpoint: String,
    capabilities: super::RouteCapabilities,
    generation_support: GenerationSupport,
    profile_settings: super::ProfileSettings,
    protocol_settings: super::ProtocolSettings,
}

impl ResponsesBinding {
    fn accepts_replay(&self, replay: &OpaqueReplayState) -> bool {
        replay.version == 1
            && replay.namespace == self.profile.namespace()
            && replay.producer.scope == self.replay_scope()
            && replay.producer.protocol_id == self.identity.protocol_id
            && replay.producer.profile_identity.as_ref() == Some(&self.identity.profile_identity)
            && replay.producer.route_identity.is_none()
            && self.capabilities.reasoning
    }
}

impl ProtocolBinding for ResponsesBinding {
    fn binding_identity(&self) -> &BindingIdentity {
        &self.identity
    }

    fn flavor(&self) -> &BindingFlavor {
        &self.flavor
    }

    fn replay_scope(&self) -> ReplayScope {
        // Reasoning items are provider/profile-owned and can be replayed on
        // another route in the same configured profile, but not cross-profile.
        ReplayScope::Profile
    }

    fn decode_http_error(
        &self,
        status: u16,
        body: &[u8],
        retry_hint: RetryHint,
    ) -> Option<ModelFailure> {
        let envelope = serde_json::from_slice::<ResponsesHttpErrorEnvelope>(body).ok()?;
        let ResponsesHttpErrorEnvelope {
            mut error,
            status: _,
            code,
            message,
        } = envelope;
        let effective_status = Some(Value::from(status));
        if error.is_none() {
            error = (code.is_some() || message.is_some() || effective_status.is_some()).then_some(
                ResponsesError {
                    status: effective_status.clone(),
                    code,
                    message,
                    ..ResponsesError::default()
                },
            );
        } else if let Some(value) = error.as_mut()
            && value.status.is_none()
        {
            value.status = effective_status;
        }
        let mut failure = error_failure(FailurePhase::Transport, error);
        failure.status = Some(status);
        let misalignment = failure.code.as_deref() == Some("misalignment_policy_violation");
        failure.kind = if misalignment {
            FailureKind::InvalidRequest
        } else {
            match status {
                401 | 403 => FailureKind::Authentication,
                429 => FailureKind::RateLimited,
                _ => FailureKind::Http,
            }
        };
        failure.retry_hint = if misalignment {
            RetryHint::Never
        } else if status == 429 || status >= 500 {
            retry_hint
        } else {
            RetryHint::Never
        };
        if !body.is_empty() {
            failure = failure.with_detail(String::from_utf8_lossy(body).into_owned());
        }
        Some(failure)
    }

    fn prepare_request(
        &self,
        input: &ModelRequestInput,
    ) -> Result<PreparedHttpRequest, ModelFailure> {
        let request = ResponsesRequest::from_input(self, input)?;
        let body = serde_json::to_vec(&request).map_err(|error| {
            ModelFailure::new(FailurePhase::Prepare, FailureKind::Internal)
                .with_code("request_serialization")
                .with_detail(error.to_string())
        })?;
        let mut protocol_headers = BTreeMap::new();
        protocol_headers.insert("accept".into(), "text/event-stream".into());
        protocol_headers.insert("content-type".into(), "application/json".into());
        Ok(PreparedHttpRequest {
            method: super::HttpMethod::Post,
            url: self.endpoint.clone(),
            protocol_headers,
            body,
            prompt_unit_origins: responses_prompt_unit_origins(self, input),
        })
    }

    fn websocket_frame(
        &self,
        request: &PreparedHttpRequest,
        previous_response_id: Option<&str>,
        incremental_prompt_unit_start: Option<usize>,
    ) -> Result<Vec<u8>, ModelFailure> {
        let mut body: Value = serde_json::from_slice(&request.body).map_err(|error| {
            ModelFailure::new(FailurePhase::Prepare, FailureKind::InvalidRequest)
                .with_code("websocket_frame_request")
                .with_detail(error.to_string())
        })?;
        let fields = body.as_object_mut().ok_or_else(|| {
            ModelFailure::new(FailurePhase::Prepare, FailureKind::InvalidRequest)
                .with_code("websocket_frame_shape")
        })?;
        if let Some(prompt_unit_start) = incremental_prompt_unit_start {
            let instructions_present = fields
                .get("instructions")
                .is_some_and(|instructions| !instructions.is_null());
            let input_start = prompt_unit_start
                .checked_sub(usize::from(instructions_present))
                .ok_or_else(|| {
                    ModelFailure::new(FailurePhase::Prepare, FailureKind::InvalidRequest)
                        .with_code("websocket_frame_input_boundary")
                })?;
            let input = fields
                .get_mut("input")
                .and_then(Value::as_array_mut)
                .ok_or_else(|| {
                    ModelFailure::new(FailurePhase::Prepare, FailureKind::InvalidRequest)
                        .with_code("websocket_frame_input")
                })?;
            if input_start > input.len() {
                return Err(
                    ModelFailure::new(FailurePhase::Prepare, FailureKind::InvalidRequest)
                        .with_code("websocket_frame_input_boundary"),
                );
            }
            *input = input[input_start..].to_vec();
        }
        fields.insert("type".into(), Value::String("response.create".into()));
        if let Some(previous_response_id) = previous_response_id {
            fields.insert(
                "previous_response_id".into(),
                Value::String(previous_response_id.to_owned()),
            );
        }
        serde_json::to_vec(&body).map_err(|error| {
            ModelFailure::new(FailurePhase::Prepare, FailureKind::Internal)
                .with_code("websocket_frame_serialization")
                .with_detail(error.to_string())
        })
    }

    fn inspect_prepared_request(
        &self,
        request: &PreparedHttpRequest,
        stable_request: Option<&PreparedHttpRequest>,
    ) -> Result<PreparedRequestInspection, ModelFailure> {
        let mut body = prepared_body(request)?;
        let fields = body.as_object_mut().ok_or_else(|| {
            ModelFailure::new(FailurePhase::Prepare, FailureKind::InvalidRequest)
                .with_code("prepared_request_shape")
        })?;
        let mut prompt_units = Vec::new();
        if let Some(instructions) = fields.remove("instructions")
            && !instructions.is_null()
        {
            prompt_units.push(serde_json::json!({
                "type": "instructions",
                "content": instructions,
            }));
        }
        prompt_units.extend(
            fields
                .remove("input")
                .and_then(|value| value.as_array().cloned())
                .unwrap_or_default(),
        );
        let routing_key = fields
            .remove("prompt_cache_key")
            .and_then(|value| value.as_str().map(str::to_owned))
            .filter(|value| !value.is_empty());
        let hint_serialized = routing_key.is_some();
        let retention_sent = fields
            .get("prompt_cache_retention")
            .and_then(Value::as_str)
            .and_then(|value| match value {
                "in_memory" => Some(super::CacheRetention::InMemory),
                "24h" => Some(super::CacheRetention::TwentyFourHours),
                _ => None,
            });
        prepared_inspection(
            body,
            prompt_units,
            &request.prompt_unit_origins,
            PreparedRequestCacheInspection {
                hint_serialized,
                retention_sent,
                local_prefix_fingerprint: stable_request_fingerprint(
                    stable_request,
                    hint_serialized,
                ),
                routing_key,
            },
        )
    }

    fn new_decoder(&self) -> Box<dyn ModelStreamDecoder> {
        Box::new(ResponsesDecoder::new(
            self.identity.clone(),
            self.profile,
            self.replay_scope(),
        ))
    }

    fn new_websocket_decoder(&self) -> Box<dyn ModelStreamDecoder> {
        Box::new(ResponsesWebSocketDecoder {
            inner: ResponsesDecoder::new(self.identity.clone(), self.profile, self.replay_scope()),
        })
    }
}

fn is_false(value: &bool) -> bool {
    !*value
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "snake_case")]
struct ResponsesRequest {
    model: String,
    stream: bool,
    input: Vec<ResponsesInputItem>,
    #[serde(skip_serializing_if = "Option::is_none")]
    instructions: Option<String>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    tools: Vec<ResponsesTool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_output_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    reasoning: Option<ResponsesReasoning>,
    #[serde(skip_serializing_if = "Option::is_none")]
    text: Option<ResponsesTextOptions>,
    #[serde(skip_serializing_if = "Option::is_none")]
    parallel_tool_calls: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    service_tier: Option<&'static str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    prompt_cache_key: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    prompt_cache_retention: Option<&'static str>,
}

#[derive(Debug, Serialize)]
#[serde(untagged)]
enum ResponsesInputItem {
    Message(ResponsesMessage),
    FunctionCall(ResponsesFunctionCall),
    FunctionCallOutput(ResponsesFunctionCallOutput),
    Reasoning(ResponsesReasoningItem),
    Opaque(Value),
}

#[derive(Debug, Serialize)]
struct ResponsesMessage {
    role: &'static str,
    content: Vec<ResponsesContentBlock>,
}

#[derive(Debug, Serialize)]
#[serde(tag = "type")]
enum ResponsesContentBlock {
    #[serde(rename = "input_text")]
    InputText { text: String },
    #[serde(rename = "input_image")]
    InputImage { image_url: String },
    #[serde(rename = "output_text")]
    OutputText { text: String },
}

#[derive(Debug, Serialize)]
struct ResponsesFunctionCall {
    #[serde(rename = "type")]
    item_type: &'static str,
    call_id: String,
    name: String,
    arguments: String,
    #[serde(rename = "async", skip_serializing_if = "is_false")]
    async_call: bool,
}

#[derive(Debug, Serialize)]
struct ResponsesFunctionCallOutput {
    #[serde(rename = "type")]
    item_type: &'static str,
    call_id: String,
    output: ResponsesFunctionCallOutputContent,
}

#[derive(Debug, Serialize)]
#[serde(untagged)]
enum ResponsesFunctionCallOutputContent {
    Text(String),
    Blocks(Vec<ResponsesContentBlock>),
}

#[derive(Debug, Serialize)]
struct ResponsesTool {
    #[serde(rename = "type")]
    tool_type: &'static str,
    name: String,
    description: String,
    parameters: Value,
    strict: bool,
    #[serde(rename = "async", skip_serializing_if = "is_false")]
    async_call: bool,
}

#[derive(Debug, Serialize)]
struct ResponsesReasoningItem {
    #[serde(rename = "type")]
    item_type: &'static str,
    id: String,
    summary: Vec<ResponsesReasoningSummary>,
    content: Vec<ResponsesReasoningContent>,
}

#[derive(Debug, Serialize)]
struct ResponsesReasoningSummary {
    #[serde(rename = "type")]
    block_type: &'static str,
    text: String,
}

#[derive(Debug, Serialize)]
struct ResponsesReasoningContent {
    #[serde(rename = "type")]
    block_type: &'static str,
    text: String,
}

#[derive(Debug, Serialize)]
struct ResponsesReasoning {
    effort: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    summary: Option<&'static str>,
}

#[derive(Debug, Serialize)]
struct ResponsesTextOptions {
    #[serde(skip_serializing_if = "Option::is_none")]
    verbosity: Option<&'static str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    format: Option<ResponsesTextFormat>,
}

#[derive(Debug, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum ResponsesTextFormat {
    JsonObject,
    JsonSchema {
        name: String,
        schema: Value,
        strict: bool,
    },
}

impl ResponsesRequest {
    fn from_input(
        binding: &ResponsesBinding,
        input: &ModelRequestInput,
    ) -> Result<Self, ModelFailure> {
        let mut instructions = Vec::new();
        for segment in &input.segments {
            if segment.kind == ControlSegmentKind::System {
                instructions.push(segment.text.clone());
            }
        }
        let instructions = (!instructions.is_empty()).then(|| instructions.join("\n\n"));

        let mut items = Vec::new();
        for segment in &input.segments {
            if segment.kind == ControlSegmentKind::Developer {
                items.push(ResponsesInputItem::Message(ResponsesMessage {
                    role: "developer",
                    content: vec![ResponsesContentBlock::InputText {
                        text: segment.text.clone(),
                    }],
                }));
            }
        }
        for message in &input.messages {
            append_message(binding, message, &mut items)?;
        }

        if !input.tools.is_empty() && !binding.capabilities.tools {
            return Err(unsupported("tools", "tools capability is disabled"));
        }
        let tools = input
            .tools
            .iter()
            .map(|tool| ResponsesTool {
                tool_type: "function",
                name: tool.name.clone(),
                description: tool.description.clone(),
                parameters: super::tool_schema::project(
                    &tool.parameters,
                    super::tool_schema::ToolSchemaDialect::OpenAiStrict,
                ),
                strict: tool.strict,
                async_call: tool.async_call,
            })
            .collect();

        let generation = &input.generation;
        require_generation(
            generation.temperature.is_some(),
            binding.generation_support.temperature,
            "temperature",
        )?;
        require_generation(
            generation.top_p.is_some(),
            binding.generation_support.top_p,
            "top_p",
        )?;
        require_generation(
            generation.max_output_tokens.is_some(),
            binding.generation_support.max_output_tokens,
            "max_output_tokens",
        )?;
        if !generation.stop_sequences.is_empty() {
            return Err(unsupported(
                "stop_sequences",
                "Responses does not support stop sequences",
            ));
        }
        let reasoning_requested = generation.reasoning.enabled
            || generation.reasoning.effort.is_some()
            || generation.summary.is_some();
        require_generation(
            reasoning_requested,
            binding.capabilities.reasoning && binding.generation_support.reasoning,
            "reasoning",
        )?;
        require_generation(
            generation.summary.is_some(),
            binding.generation_support.reasoning_summary,
            "reasoning_summary",
        )?;
        require_generation(
            generation.verbosity.is_some(),
            binding.generation_support.text_verbosity,
            "text_verbosity",
        )?;
        require_generation(
            generation.parallel_tool_calls == Some(true),
            binding.capabilities.parallel_tool_calls
                && binding.generation_support.parallel_tool_calls,
            "parallel_tool_calls",
        )?;
        require_generation(
            generation.priority_service == Some(true),
            binding.capabilities.priority_service && binding.generation_support.priority_service,
            "priority_service",
        )?;
        if input.cache.retention.is_some() && !input.cache.enabled {
            return Err(unsupported(
                "prompt_cache_retention",
                "retention requires prompt cache to be enabled",
            ));
        }

        let reasoning = if reasoning_requested {
            Some(ResponsesReasoning {
                effort: reasoning_effort(generation.reasoning.effort.as_ref()),
                summary: generation.summary.map(reasoning_summary),
            })
        } else {
            None
        };
        let format = responses_text_format(
            generation.structured_output.as_ref(),
            binding.generation_support.structured_output,
        )?;
        let text =
            (generation.verbosity.is_some() || format.is_some()).then(|| ResponsesTextOptions {
                verbosity: generation.verbosity.map(text_verbosity),
                format,
            });
        let prompt_cache_key = input.cache.enabled.then(|| cache_key(binding, input));
        let prompt_cache_retention = input.cache.retention.map(|retention| match retention {
            super::CacheRetention::InMemory => "in_memory",
            super::CacheRetention::TwentyFourHours => "24h",
        });
        Ok(Self {
            model: input.control.model.clone(),
            stream: input.control.stream,
            input: items,
            instructions,
            tools,
            temperature: generation.temperature,
            top_p: generation.top_p,
            max_output_tokens: generation.max_output_tokens,
            reasoning,
            text,
            parallel_tool_calls: generation.parallel_tool_calls,
            service_tier: (generation.priority_service == Some(true)).then_some("priority"),
            prompt_cache_key,
            prompt_cache_retention,
        })
    }
}

fn responses_prompt_unit_origins(
    binding: &ResponsesBinding,
    input: &ModelRequestInput,
) -> Vec<Vec<String>> {
    let system_origins = input
        .segments
        .iter()
        .zip(input.segment_origins.iter())
        .filter_map(|(segment, origin)| {
            (segment.kind == ControlSegmentKind::System).then_some(origin.clone())
        })
        .collect::<Vec<_>>();
    let mut origins = Vec::new();
    if !system_origins.is_empty() {
        origins.push(system_origins);
    }
    origins.extend(
        input
            .segments
            .iter()
            .zip(input.segment_origins.iter())
            .filter_map(|(segment, origin)| {
                (segment.kind == ControlSegmentKind::Developer).then_some(vec![origin.clone()])
            }),
    );
    for (message, origin) in input.messages.iter().zip(input.message_origins.iter()) {
        let mut count = 0usize;
        let has_visible = message
            .content
            .iter()
            .any(|part| matches!(part, ContentPart::Text(_) | ContentPart::Image { .. }));
        count += usize::from(has_visible);
        for part in &message.content {
            match part {
                ContentPart::Reasoning {
                    text,
                    replay: Some(replay),
                    ..
                } => {
                    count += usize::from(
                        binding.accepts_replay(replay)
                            || (binding.profile == ResponsesFlavor::DeepSeek && !text.is_empty()),
                    );
                }
                ContentPart::Reasoning {
                    text, replay: None, ..
                } => {
                    count += usize::from(
                        binding.profile == ResponsesFlavor::DeepSeek && !text.is_empty(),
                    );
                }
                ContentPart::ToolCall { .. } | ContentPart::ToolResult { .. } => count += 1,
                _ => {}
            }
        }
        origins.extend(std::iter::repeat_n(vec![origin.clone()], count));
    }
    origins
}

fn append_message(
    binding: &ResponsesBinding,
    message: &ModelMessage,
    items: &mut Vec<ResponsesInputItem>,
) -> Result<(), ModelFailure> {
    let role = match message.role {
        MessageRole::User => "user",
        MessageRole::Assistant => "assistant",
        MessageRole::Tool => "tool",
    };
    let mut text = Vec::new();
    let mut images = Vec::new();
    let mut replays = Vec::new();
    let mut tool_calls = Vec::new();
    let mut tool_outputs = Vec::new();
    let mut reasoning_items = Vec::new();
    for part in &message.content {
        match part {
            ContentPart::Text(value) => text.push(value.clone()),
            ContentPart::Image { media_type, data } => {
                if !binding.capabilities.input_images {
                    return Err(unsupported(
                        "input_images",
                        "input image capability is disabled",
                    ));
                }
                images.push(format!("data:{media_type};base64,{}", base64_encode(data)));
            }
            ContentPart::Reasoning {
                item_id,
                text,
                replay,
            } => {
                let replay_emitted = replay
                    .as_ref()
                    .filter(|replay| binding.accepts_replay(replay))
                    .map(|replay| {
                        replays.push(ResponsesInputItem::Opaque(replay.payload.clone()));
                    })
                    .is_some();
                if !replay_emitted
                    && binding.profile == ResponsesFlavor::DeepSeek
                    && !text.is_empty()
                {
                    reasoning_items.push(ResponsesInputItem::Reasoning(ResponsesReasoningItem {
                        item_type: "reasoning",
                        id: item_id.clone(),
                        summary: Vec::new(),
                        content: vec![ResponsesReasoningContent {
                            block_type: "reasoning_text",
                            text: text.clone(),
                        }],
                    }));
                }
            }
            ContentPart::ToolCall {
                id,
                name,
                arguments,
            } => tool_calls.push((id.clone(), name.clone(), arguments.clone())),
            ContentPart::ToolResult { id, content } => {
                if content
                    .iter()
                    .any(|part| matches!(part, ContentPart::Image { .. }))
                    && !binding.capabilities.tool_result_images
                {
                    return Err(unsupported(
                        "tool_result_images",
                        "tool result image capability is disabled",
                    ));
                }
                tool_outputs.push((id.clone(), content.clone()));
            }
        }
    }
    if !text.is_empty() || !images.is_empty() {
        let mut content = text
            .into_iter()
            .map(|text| match message.role {
                MessageRole::Assistant => ResponsesContentBlock::OutputText { text },
                MessageRole::User | MessageRole::Tool => ResponsesContentBlock::InputText { text },
            })
            .collect::<Vec<_>>();
        content.extend(
            images
                .into_iter()
                .map(|image_url| ResponsesContentBlock::InputImage { image_url }),
        );
        let role = if message.role == MessageRole::Tool {
            "user"
        } else {
            role
        };
        items.push(ResponsesInputItem::Message(ResponsesMessage {
            role,
            content,
        }));
    }
    // Replay items must precede function calls, including when a provider emits
    // the reasoning part and call part in separate internal message fragments.
    items.extend(replays);
    items.extend(reasoning_items);
    for (id, name, arguments) in tool_calls {
        let arguments = serde_json::to_string(&arguments).map_err(|error| {
            ModelFailure::new(FailurePhase::Prepare, FailureKind::InvalidRequest)
                .with_code("invalid_tool_arguments")
                .with_detail(error.to_string())
        })?;
        items.push(ResponsesInputItem::FunctionCall(ResponsesFunctionCall {
            item_type: "function_call",
            call_id: id,
            name,
            arguments,
            async_call: false,
        }));
    }
    for (id, content) in tool_outputs {
        let has_images = content
            .iter()
            .any(|part| matches!(part, ContentPart::Image { .. }));
        let output = if has_images {
            ResponsesFunctionCallOutputContent::Blocks(
                content
                    .into_iter()
                    .map(|part| match part {
                        ContentPart::Text(text) => Ok(ResponsesContentBlock::InputText { text }),
                        ContentPart::Image { media_type, data } => {
                            Ok(ResponsesContentBlock::InputImage {
                                image_url: format!(
                                    "data:{media_type};base64,{}",
                                    base64_encode(&data)
                                ),
                            })
                        }
                        _ => Err(unsupported(
                            "tool_output",
                            "unsupported nested tool output block",
                        )),
                    })
                    .collect::<Result<Vec<_>, _>>()?,
            )
        } else {
            let mut text = String::new();
            for part in content {
                match part {
                    ContentPart::Text(value) => text.push_str(&value),
                    _ => {
                        return Err(unsupported(
                            "tool_output",
                            "unsupported nested tool output block",
                        ));
                    }
                }
            }
            ResponsesFunctionCallOutputContent::Text(text)
        };
        items.push(ResponsesInputItem::FunctionCallOutput(
            ResponsesFunctionCallOutput {
                item_type: "function_call_output",
                call_id: id,
                output,
            },
        ));
    }
    Ok(())
}

fn require_generation(requested: bool, supported: bool, field: &str) -> Result<(), ModelFailure> {
    if requested && !supported {
        return Err(unsupported(
            field,
            "requested generation feature is not supported",
        ));
    }
    Ok(())
}

/// The route declares which structured-output contracts it supports; a request
/// asking for a contract the route did not declare fails in prepare.
fn require_declared_structured_output(
    requested: super::StructuredOutputSupport,
    supported: Option<super::StructuredOutputSupport>,
) -> Result<(), ModelFailure> {
    if supported.is_some_and(|supported| supported.supports(requested)) {
        return Ok(());
    }
    Err(unsupported(
        "structured_output",
        &format!("{} capability is disabled", requested.as_str()),
    ))
}

fn completions_response_format(
    requested: Option<&super::StructuredOutput>,
    supported: Option<super::StructuredOutputSupport>,
) -> Result<Option<CompletionsResponseFormat>, ModelFailure> {
    let Some(requested) = requested else {
        return Ok(None);
    };
    require_declared_structured_output(requested.kind(), supported)?;
    Ok(Some(match requested {
        super::StructuredOutput::JsonObject => CompletionsResponseFormat::JsonObject,
        super::StructuredOutput::JsonSchema(schema) => CompletionsResponseFormat::JsonSchema {
            json_schema: CompletionsJsonSchema {
                name: schema.name.clone(),
                schema: schema.schema.clone(),
                strict: schema.strict,
            },
        },
    }))
}

fn responses_text_format(
    requested: Option<&super::StructuredOutput>,
    supported: Option<super::StructuredOutputSupport>,
) -> Result<Option<ResponsesTextFormat>, ModelFailure> {
    let Some(requested) = requested else {
        return Ok(None);
    };
    require_declared_structured_output(requested.kind(), supported)?;
    Ok(Some(match requested {
        super::StructuredOutput::JsonObject => ResponsesTextFormat::JsonObject,
        super::StructuredOutput::JsonSchema(schema) => ResponsesTextFormat::JsonSchema {
            name: schema.name.clone(),
            schema: schema.schema.clone(),
            strict: schema.strict,
        },
    }))
}

fn unsupported(field: &str, detail: &str) -> ModelFailure {
    ModelFailure::new(FailurePhase::Prepare, FailureKind::InvalidRequest)
        .with_code("unsupported_request_field")
        .with_detail(format!("{field}: {detail}"))
}

fn deepseek_reasoning_effort(effort: &super::ReasoningEffort) -> Option<String> {
    match effort {
        super::ReasoningEffort::None => None,
        super::ReasoningEffort::Minimal | super::ReasoningEffort::Low => Some("low".into()),
        super::ReasoningEffort::Medium
        | super::ReasoningEffort::High
        | super::ReasoningEffort::Xhigh => Some("high".into()),
        super::ReasoningEffort::Max => Some("max".into()),
        super::ReasoningEffort::Custom(value) => Some(value.clone()),
    }
}

fn reasoning_effort(effort: Option<&super::ReasoningEffort>) -> String {
    match effort {
        Some(super::ReasoningEffort::Minimal) => "minimal".into(),
        Some(super::ReasoningEffort::Low) => "low".into(),
        Some(super::ReasoningEffort::Medium) => "medium".into(),
        Some(super::ReasoningEffort::High) => "high".into(),
        Some(super::ReasoningEffort::Xhigh) => "xhigh".into(),
        Some(super::ReasoningEffort::Max) => "max".into(),
        Some(super::ReasoningEffort::Custom(value)) => value.clone(),
        Some(super::ReasoningEffort::None) => "none".into(),
        None => "medium".into(),
    }
}

fn anthropic_reasoning_effort(
    effort: Option<&super::ReasoningEffort>,
) -> Result<String, ModelFailure> {
    match effort {
        Some(super::ReasoningEffort::None) => Err(unsupported(
            "reasoning_effort",
            "none is not supported by Anthropic adaptive thinking",
        )),
        Some(super::ReasoningEffort::Minimal) | Some(super::ReasoningEffort::Low) => {
            Ok("low".into())
        }
        Some(super::ReasoningEffort::Medium) => Ok("medium".into()),
        Some(super::ReasoningEffort::High) => Ok("high".into()),
        Some(super::ReasoningEffort::Xhigh) => Ok("xhigh".into()),
        Some(super::ReasoningEffort::Max) => Ok("max".into()),
        Some(super::ReasoningEffort::Custom(value))
            if matches!(value.as_str(), "low" | "medium" | "high" | "xhigh" | "max") =>
        {
            Ok(value.clone())
        }
        Some(super::ReasoningEffort::Custom(_)) => Err(unsupported(
            "reasoning_effort",
            "custom effort is not supported by Anthropic adaptive thinking",
        )),
        None => Ok("low".into()),
    }
}

fn reasoning_summary(summary: super::ReasoningSummary) -> &'static str {
    match summary {
        super::ReasoningSummary::Auto => "auto",
        super::ReasoningSummary::Concise => "concise",
        super::ReasoningSummary::Detailed => "detailed",
    }
}

fn text_verbosity(verbosity: super::Verbosity) -> &'static str {
    match verbosity {
        super::Verbosity::Low => "low",
        super::Verbosity::Medium => "medium",
        super::Verbosity::High => "high",
    }
}

fn cache_key_for_completions(
    binding: &CompletionsBinding,
    input: &ModelRequestInput,
    request: &CompletionsRequest,
) -> Result<String, ModelFailure> {
    #[derive(Serialize)]
    struct StableCacheShape<'a> {
        model: &'a str,
        tools: &'a [CompletionsTool],
        temperature: Option<f32>,
        top_p: Option<f32>,
        max_completion_tokens: Option<u32>,
        stop: &'a [String],
        reasoning_effort: &'a Option<String>,
        verbosity: Option<&'static str>,
        parallel_tool_calls: Option<bool>,
        service_tier: Option<&'static str>,
    }

    let namespace = input.cache.namespace.as_deref().unwrap_or("completions");
    let shape = StableCacheShape {
        model: &request.model,
        tools: &request.tools,
        temperature: request.temperature,
        top_p: request.top_p,
        max_completion_tokens: request.max_completion_tokens,
        stop: &request.stop,
        reasoning_effort: &request.reasoning_effort,
        verbosity: request.verbosity,
        parallel_tool_calls: request.parallel_tool_calls,
        service_tier: request.service_tier,
    };
    let wire = serde_json::to_vec(&shape).map_err(|error| {
        ModelFailure::new(FailurePhase::Prepare, FailureKind::Internal)
            .with_code("cache_identity_serialization")
            .with_detail(error.to_string())
    })?;
    let digest = crate::request_builder::sha256_hex(&wire);
    Ok(format!(
        "{namespace}:{}:{}:{digest}",
        binding.identity.profile_identity.as_str(),
        binding.identity.route_identity.as_str()
    ))
}

fn cache_key(binding: &ResponsesBinding, input: &ModelRequestInput) -> String {
    let namespace = input.cache.namespace.as_deref().unwrap_or("responses");
    format!(
        "{namespace}:{}:{}",
        binding.identity.profile_identity.as_str(),
        binding.identity.route_identity.as_str()
    )
}

fn base64_encode(data: &[u8]) -> String {
    const TABLE: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut output = String::with_capacity(data.len().div_ceil(3) * 4);
    for chunk in data.chunks(3) {
        let first = chunk[0] as u32;
        let second = chunk.get(1).copied().unwrap_or(0) as u32;
        let third = chunk.get(2).copied().unwrap_or(0) as u32;
        let value = (first << 16) | (second << 8) | third;
        output.push(TABLE[((value >> 18) & 63) as usize] as char);
        output.push(TABLE[((value >> 12) & 63) as usize] as char);
        output.push(if chunk.len() > 1 {
            TABLE[((value >> 6) & 63) as usize] as char
        } else {
            '='
        });
        output.push(if chunk.len() > 2 {
            TABLE[(value & 63) as usize] as char
        } else {
            '='
        });
    }
    output
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct SseEvent {
    pub(crate) event: Option<String>,
    pub(crate) data: String,
}

pub(crate) struct IncrementalSseFramer {
    buffer: Vec<u8>,
    event: Option<String>,
    data: Vec<String>,
}

impl IncrementalSseFramer {
    pub(crate) fn new() -> Self {
        Self {
            buffer: Vec::new(),
            event: None,
            data: Vec::new(),
        }
    }

    pub(crate) fn push(&mut self, chunk: &[u8]) -> Result<Vec<SseEvent>, std::str::Utf8Error> {
        self.buffer.extend_from_slice(chunk);
        self.drain_lines(false)
    }

    pub(crate) fn finish(&mut self) -> Result<Vec<SseEvent>, std::str::Utf8Error> {
        let mut events = self.drain_lines(true)?;
        if let Some(event) = self.dispatch() {
            events.push(event);
        }
        Ok(events)
    }

    fn drain_lines(&mut self, flush_partial: bool) -> Result<Vec<SseEvent>, std::str::Utf8Error> {
        let mut events = Vec::new();
        let mut consumed = 0usize;
        while let Some(relative) = self.buffer[consumed..]
            .iter()
            .position(|byte| *byte == b'\n' || *byte == b'\r')
        {
            let index = consumed + relative;
            let delimiter = self.buffer[index];
            if delimiter == b'\r' && index + 1 == self.buffer.len() && !flush_partial {
                break;
            }
            let line = self.buffer[consumed..index].to_vec();
            consumed = index + 1;
            if delimiter == b'\r' && self.buffer.get(consumed) == Some(&b'\n') {
                consumed += 1;
            }
            events.extend(self.process_line(&line)?);
        }
        if flush_partial && consumed < self.buffer.len() {
            let line = self.buffer[consumed..].to_vec();
            events.extend(self.process_line(&line)?);
            consumed = self.buffer.len();
        }
        self.buffer.drain(..consumed);
        Ok(events)
    }

    fn process_line(&mut self, line: &[u8]) -> Result<Vec<SseEvent>, std::str::Utf8Error> {
        let line = std::str::from_utf8(line)?;
        if line.is_empty() {
            return Ok(self.dispatch().into_iter().collect());
        }
        if line.starts_with(':') {
            return Ok(Vec::new());
        }
        let (field, value) = line.split_once(':').unwrap_or((line, ""));
        let value = value.strip_prefix(' ').unwrap_or(value);
        match field {
            "event" => self.event = Some(value.to_owned()),
            "data" => self.data.push(value.to_owned()),
            _ => {}
        }
        Ok(Vec::new())
    }

    fn dispatch(&mut self) -> Option<SseEvent> {
        if self.data.is_empty() {
            self.event = None;
            return None;
        }
        Some(SseEvent {
            event: self.event.take(),
            data: std::mem::take(&mut self.data).join("\n"),
        })
    }
}

#[derive(Debug, Deserialize, Serialize, Default)]
struct ResponsesStreamEvent {
    #[serde(rename = "type", default)]
    event_type: String,
    #[serde(default)]
    delta: Option<String>,
    #[serde(default)]
    text: Option<String>,
    #[serde(default)]
    item_id: Option<String>,
    #[serde(default)]
    output_index: Option<u64>,
    #[serde(default)]
    summary_index: Option<u64>,
    #[serde(default)]
    item: Option<ResponsesStreamItem>,
    #[serde(default)]
    response: Option<ResponsesStreamResponse>,
    #[serde(default)]
    arguments: Option<String>,
    #[serde(default)]
    name: Option<String>,
    #[serde(rename = "async", default)]
    async_call: Option<bool>,
    #[serde(default)]
    error: Option<ResponsesError>,
    #[serde(default)]
    code: Option<Value>,
    #[serde(default)]
    message: Option<Value>,
    #[serde(default, alias = "status_code")]
    status: Option<Value>,
}

#[derive(Debug, Deserialize, Serialize, Default)]
struct ResponsesStreamItem {
    #[serde(rename = "type", default)]
    item_type: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    call_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    name: Option<String>,
    #[serde(rename = "async", default, skip_serializing_if = "Option::is_none")]
    async_call: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    arguments: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    text: Option<String>,
    #[serde(default)]
    summary: Vec<ResponsesSummaryBlock>,
    #[serde(default)]
    content: Vec<ResponsesSummaryBlock>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    encrypted_content: Option<String>,
}

struct ToolState {
    call_id: String,
    name: String,
    arguments: String,
    arguments_done: bool,
    item_done: bool,
    final_arguments: Option<Value>,
    tool_done_emitted: bool,
}

#[derive(Debug, Deserialize, Serialize, Default)]
struct ResponsesSummaryBlock {
    #[serde(rename = "type", default)]
    block_type: String,
    #[serde(default)]
    text: Option<String>,
}

#[derive(Debug, Deserialize, Serialize, Default)]
struct ResponsesStreamResponse {
    #[serde(default)]
    id: Option<String>,
    #[serde(default)]
    status: Option<String>,
    #[serde(default)]
    usage: Option<ResponsesUsage>,
    #[serde(default)]
    error: Option<ResponsesError>,
}

#[derive(Debug, Deserialize, Serialize, Default)]
struct ResponsesUsage {
    #[serde(default)]
    input_tokens: u64,
    #[serde(default)]
    output_tokens: u64,
    #[serde(default)]
    total_tokens: u64,
    #[serde(default)]
    input_tokens_details: Option<ResponsesTokenDetails>,
    #[serde(default)]
    output_tokens_details: Option<ResponsesTokenDetails>,
    #[serde(default)]
    cache_read_input_tokens: Option<u64>,
    #[serde(default)]
    cache_creation_input_tokens: Option<u64>,
    #[serde(default)]
    cache_write_input_tokens: Option<u64>,
}

#[derive(Debug, Deserialize, Serialize, Default)]
struct ResponsesTokenDetails {
    #[serde(default)]
    cached_tokens: u64,
    #[serde(default)]
    cache_write_tokens: Option<u64>,
    reasoning_tokens: Option<u64>,
}

#[derive(Debug, Deserialize, Serialize, Default)]
struct ResponsesHttpErrorEnvelope {
    #[serde(default)]
    error: Option<ResponsesError>,
    #[serde(default, alias = "status_code")]
    status: Option<Value>,
    #[serde(default)]
    code: Option<Value>,
    #[serde(default)]
    message: Option<Value>,
}

#[derive(Debug, Deserialize, Serialize, Default)]
struct ResponsesError {
    #[serde(default, alias = "status_code")]
    status: Option<Value>,
    #[serde(rename = "type", default)]
    error_type: Option<Value>,
    #[serde(default)]
    code: Option<Value>,
    #[serde(default)]
    message: Option<Value>,
    #[serde(default)]
    param: Option<String>,
    #[serde(default)]
    detail: Option<Value>,
}

struct ResponsesDecoder {
    framer: IncrementalSseFramer,
    identity: BindingIdentity,
    flavor: ResponsesFlavor,
    replay_scope: ReplayScope,
    reasoning: BTreeMap<String, String>,
    reasoning_components: BTreeMap<(String, String), String>,
    reasoning_component_done: std::collections::BTreeSet<(String, String)>,
    reasoning_done: std::collections::BTreeSet<String>,
    tools: BTreeMap<String, ToolState>,
    response_id: Option<String>,
    terminal: bool,
}

impl ResponsesDecoder {
    fn new(identity: BindingIdentity, flavor: ResponsesFlavor, replay_scope: ReplayScope) -> Self {
        Self {
            framer: IncrementalSseFramer::new(),
            identity,
            flavor,
            replay_scope,
            reasoning: BTreeMap::new(),
            reasoning_components: BTreeMap::new(),
            reasoning_component_done: std::collections::BTreeSet::new(),
            reasoning_done: std::collections::BTreeSet::new(),
            tools: BTreeMap::new(),
            response_id: None,
            terminal: false,
        }
    }

    fn consume(&mut self, events: Vec<SseEvent>) -> Result<Vec<ModelEvent>, ModelFailure> {
        let mut output = Vec::new();
        for event in events {
            if event.data == "[DONE]" {
                continue;
            }
            let mut parsed: ResponsesStreamEvent =
                serde_json::from_str(&event.data).map_err(|error| {
                    ModelFailure::new(FailurePhase::Decode, FailureKind::MalformedResponse)
                        .with_code("malformed_json")
                        .with_detail(error.to_string())
                })?;
            if parsed.event_type.is_empty() {
                parsed.event_type = event.event.unwrap_or_default();
            }
            self.consume_event(parsed, &mut output)?;
        }
        Ok(output)
    }

    fn consume_event(
        &mut self,
        event: ResponsesStreamEvent,
        output: &mut Vec<ModelEvent>,
    ) -> Result<(), ModelFailure> {
        if let Some(response) = event.response.as_ref()
            && let Some(id) = response.id.as_ref().filter(|id| !id.is_empty())
        {
            if let Some(previous) = &self.response_id
                && previous != id
            {
                return Err(decode_invalid("response id changed"));
            }
            if self.response_id.is_none() {
                self.response_id = Some(id.clone());
                output.push(ModelEvent::ResponseMetadata {
                    response_id: id.clone(),
                });
            }
        }
        match event.event_type.as_str() {
            "response.output_item.added" => {
                if let Some(item) = event.item {
                    match item.item_type.as_str() {
                        "reasoning" => {
                            let id = item
                                .id
                                .or(event.item_id)
                                .ok_or_else(|| decode_invalid("reasoning item id is required"))?;
                            if self.reasoning.contains_key(&id) {
                                return Err(decode_invalid("duplicate reasoning item"));
                            }
                            self.reasoning.entry(id.clone()).or_default();
                            output.push(ModelEvent::ReasoningStarted { item_id: id });
                        }
                        "function_call" => {
                            let item_id = item.id.ok_or_else(|| {
                                decode_invalid("function call item id is required")
                            })?;
                            let call_id = item.call_id.ok_or_else(|| {
                                decode_invalid("function call call_id is required")
                            })?;
                            let name = item
                                .name
                                .ok_or_else(|| decode_invalid("function call name is required"))?;
                            if self.tools.contains_key(&item_id) {
                                return Err(decode_invalid("duplicate function call item"));
                            }
                            self.tools.insert(
                                item_id.clone(),
                                ToolState {
                                    call_id: call_id.clone(),
                                    name: name.clone(),
                                    arguments: String::new(),
                                    arguments_done: false,
                                    item_done: false,
                                    final_arguments: None,
                                    tool_done_emitted: false,
                                },
                            );
                            output.push(ModelEvent::ToolStarted { id: call_id, name });
                        }
                        _ => {}
                    }
                }
            }
            "response.reasoning_summary_text.delta" | "response.reasoning_text.delta" => {
                let id = event.item_id.unwrap_or_else(|| "reasoning".into());
                let text = event.delta.unwrap_or_default();
                if event.event_type.starts_with("response.reasoning_summary") {
                    let summary_index = event.summary_index.unwrap_or(0);
                    let component = format!("summary:{summary_index:020}");
                    self.reasoning_components
                        .entry((id.clone(), component))
                        .or_default()
                        .push_str(&text);
                    self.reasoning
                        .entry(id.clone())
                        .or_default()
                        .push_str(&text);
                    output.push(ModelEvent::ReasoningSummaryDelta {
                        item_id: id,
                        summary_index,
                        text,
                    });
                } else {
                    self.reasoning_components
                        .entry((id.clone(), "text".into()))
                        .or_default()
                        .push_str(&text);
                    self.reasoning
                        .entry(id.clone())
                        .or_default()
                        .push_str(&text);
                    output.push(ModelEvent::ReasoningDelta { item_id: id, text });
                }
            }
            "response.reasoning_summary_text.done" | "response.reasoning_text.done" => {
                let id = event
                    .item_id
                    .ok_or_else(|| decode_invalid("reasoning item id is required"))?;
                let value = event.text.or(event.delta).unwrap_or_default();
                let summary_index = event.summary_index.unwrap_or(0);
                let component = if event.event_type.starts_with("response.reasoning_summary") {
                    format!("summary:{summary_index:020}")
                } else {
                    "text".into()
                };
                let key = (id.clone(), component);
                if self.reasoning_component_done.contains(&key) {
                    return Ok(());
                }
                self.reasoning_components.insert(key.clone(), value.clone());
                self.reasoning_component_done.insert(key);
                if event.event_type.starts_with("response.reasoning_summary") {
                    output.push(ModelEvent::ReasoningSummaryDone {
                        item_id: id,
                        summary_index,
                        text: value,
                    });
                }
            }
            "response.output_text.delta" => {
                output.push(ModelEvent::TextDelta {
                    text: event.delta.unwrap_or_default(),
                });
            }
            "response.function_call_arguments.delta" => {
                let id = event
                    .item_id
                    .ok_or_else(|| decode_invalid("function call item id is required"))?;
                let delta = event
                    .delta
                    .ok_or_else(|| decode_invalid("function call argument delta is required"))?;
                let state = self
                    .tools
                    .get_mut(&id)
                    .ok_or_else(|| decode_invalid("function call was not started"))?;
                if state.item_done {
                    return Err(decode_invalid("function call item is already complete"));
                }
                state.arguments.push_str(&delta);
                output.push(ModelEvent::ToolArgumentsDelta {
                    id: state.call_id.clone(),
                    delta,
                });
            }
            "response.function_call_arguments.done" => {
                let id = event
                    .item_id
                    .ok_or_else(|| decode_invalid("function call item id is required"))?;
                let arguments = event
                    .arguments
                    .ok_or_else(|| decode_invalid("function call arguments are required"))?;
                self.finalize_arguments(&id, event.name.as_deref(), arguments, output)?;
            }
            "response.output_item.done" => {
                let item = event
                    .item
                    .ok_or_else(|| decode_invalid("output item is required"))?;
                match item.item_type.as_str() {
                    "function_call" => {
                        let id = item
                            .id
                            .ok_or_else(|| decode_invalid("function call item id is required"))?;
                        let state = self
                            .tools
                            .get(&id)
                            .ok_or_else(|| decode_invalid("function call was not started"))?;
                        if item.call_id.as_deref() != Some(state.call_id.as_str())
                            || item.name.as_deref() != Some(state.name.as_str())
                        {
                            return Err(decode_invalid("function call identity changed"));
                        }
                        self.complete_tool_item(
                            &id,
                            item.call_id.as_deref(),
                            item.name.as_deref(),
                            item.arguments,
                            output,
                        )?;
                    }
                    "reasoning" => {
                        let id = item
                            .id
                            .clone()
                            .ok_or_else(|| decode_invalid("reasoning item id is required"))?;
                        self.emit_reasoning_done(id, item.text.clone(), Some(item), output)?;
                    }
                    _ => {}
                }
            }
            "response.completed" => {
                if self.terminal {
                    return Err(decode_invalid("duplicate terminal event"));
                }
                let response = event
                    .response
                    .ok_or_else(|| decode_invalid("completed response is required"))?;
                let status = response
                    .status
                    .as_deref()
                    .ok_or_else(|| decode_invalid("completed response status is required"))?;
                let status = terminal_status(status)
                    .ok_or_else(|| decode_invalid("unknown response status"))?;
                if status == TerminalStatus::Completed
                    && (self.tools.values().any(|state| !state.item_done)
                        || self
                            .reasoning
                            .keys()
                            .any(|id| !self.reasoning_done.contains(id)))
                {
                    return Err(decode_invalid(
                        "completed response has incomplete output items",
                    ));
                }
                self.emit_usage(response.usage.as_ref(), output);
                output.push(ModelEvent::Terminal { status });
                self.terminal = true;
            }
            "response.incomplete" => {
                if self.terminal {
                    return Err(decode_invalid("duplicate terminal event"));
                }
                let response = event
                    .response
                    .ok_or_else(|| decode_invalid("incomplete response is required"))?;
                if response.status.as_deref() != Some("incomplete") {
                    return Err(decode_invalid(
                        "incomplete response status is contradictory",
                    ));
                }
                self.emit_usage(response.usage.as_ref(), output);
                output.push(ModelEvent::Terminal {
                    status: TerminalStatus::Incomplete,
                });
                self.terminal = true;
            }
            "response.failed" => {
                if self.terminal {
                    return Err(decode_invalid("duplicate terminal event"));
                }
                let response = event
                    .response
                    .ok_or_else(|| decode_invalid("failed response is required"))?;
                let failure = response_failure(FailurePhase::Decode, response.error);
                output.push(ModelEvent::Failure(failure));
                self.terminal = true;
            }
            "error" => {
                if self.terminal {
                    return Err(decode_invalid("duplicate terminal event"));
                }
                let status = event.status.clone();
                let error = event
                    .error
                    .map(|mut error| {
                        if error.status.is_none() {
                            error.status = status.clone();
                        }
                        error
                    })
                    .or_else(|| {
                        if event.code.is_some() || event.message.is_some() || event.status.is_some()
                        {
                            Some(ResponsesError {
                                status,
                                code: event.code,
                                message: event.message,
                                ..ResponsesError::default()
                            })
                        } else {
                            None
                        }
                    });
                output.push(ModelEvent::Failure(error_failure(
                    FailurePhase::Decode,
                    error,
                )));
                self.terminal = true;
            }
            _ => {}
        }
        Ok(())
    }

    fn emit_reasoning_done(
        &mut self,
        id: String,
        delta: Option<String>,
        item: Option<ResponsesStreamItem>,
        output: &mut Vec<ModelEvent>,
    ) -> Result<(), ModelFailure> {
        if self.reasoning_done.contains(&id) {
            return Ok(());
        }
        if item.is_none() {
            return Ok(());
        }
        self.reasoning_done.insert(id.clone());
        let authoritative_text = item.as_ref().map(|item| {
            item.summary
                .iter()
                .chain(item.content.iter())
                .filter_map(|block| block.text.as_deref())
                .filter(|text| !text.is_empty())
                .collect::<Vec<_>>()
                .join("\n\n")
        });
        let component_text = self
            .reasoning_components
            .iter()
            .filter(|((item_id, _), _)| item_id == &id)
            .map(|(_, text)| text.as_str())
            .filter(|text| !text.is_empty())
            .collect::<Vec<_>>()
            .join("\n\n");
        let text = authoritative_text
            .filter(|text| !text.is_empty())
            .or(delta.filter(|text| !text.is_empty()))
            .or_else(|| (!component_text.is_empty()).then_some(component_text))
            .or_else(|| self.reasoning.remove(&id))
            .unwrap_or_default();
        let payload = item
            .as_ref()
            .map(|value| serde_json::to_value(value).unwrap_or(Value::Null))
            .unwrap_or_else(|| {
                serde_json::json!({
                    "type": "reasoning",
                    "id": id,
                    "content": [{"type": "reasoning_text", "text": text}]
                })
            });
        output.push(ModelEvent::ReasoningDone {
            item_id: id,
            text,
            replay: Some(OpaqueReplayState::new(
                self.flavor.namespace(),
                1,
                ReplayProducer::new(self.replay_scope, &self.identity),
                payload,
            )),
        });
        Ok(())
    }

    fn parse_arguments(arguments: &str) -> Result<Value, ModelFailure> {
        if arguments.trim().is_empty() {
            return Err(decode_invalid("function call arguments are required"));
        }
        serde_json::from_str(arguments).map_err(|error| {
            ModelFailure::new(FailurePhase::Decode, FailureKind::MalformedResponse)
                .with_code("malformed_tool_arguments")
                .with_detail(error.to_string())
        })
    }

    fn finalize_arguments(
        &mut self,
        id: &str,
        name: Option<&str>,
        arguments: String,
        output: &mut Vec<ModelEvent>,
    ) -> Result<(), ModelFailure> {
        let value = Self::parse_arguments(&arguments)?;
        let state = self
            .tools
            .get_mut(id)
            .ok_or_else(|| decode_invalid("function call was not started"))?;
        if let Some(name) = name
            && name != state.name
        {
            return Err(decode_invalid("function call name changed"));
        }
        if state.arguments_done && state.final_arguments.as_ref() != Some(&value) {
            return Err(decode_invalid("function call arguments changed"));
        }
        state.arguments_done = true;
        state.final_arguments = Some(value.clone());
        state.arguments = arguments;
        if !state.tool_done_emitted {
            state.tool_done_emitted = true;
            output.push(ModelEvent::ToolDone {
                id: state.call_id.clone(),
                name: state.name.clone(),
                arguments: value,
            });
        }
        Ok(())
    }

    fn complete_tool_item(
        &mut self,
        id: &str,
        call_id: Option<&str>,
        name: Option<&str>,
        arguments: Option<String>,
        output: &mut Vec<ModelEvent>,
    ) -> Result<(), ModelFailure> {
        let state = self
            .tools
            .get(id)
            .ok_or_else(|| decode_invalid("function call was not started"))?;
        if call_id != Some(state.call_id.as_str()) || name != Some(state.name.as_str()) {
            return Err(decode_invalid("function call identity changed"));
        }
        let final_arguments = arguments.unwrap_or_else(|| state.arguments.clone());
        let value = Self::parse_arguments(&final_arguments)?;
        if state.arguments_done && state.final_arguments.as_ref() != Some(&value) {
            return Err(decode_invalid("function call arguments changed"));
        }
        let state = self.tools.get_mut(id).expect("validated function call");
        state.arguments_done = true;
        state.final_arguments = Some(value.clone());
        state.item_done = true;
        if !state.tool_done_emitted {
            state.tool_done_emitted = true;
            output.push(ModelEvent::ToolDone {
                id: state.call_id.clone(),
                name: state.name.clone(),
                arguments: value,
            });
        }
        Ok(())
    }

    fn emit_usage(&self, usage: Option<&ResponsesUsage>, output: &mut Vec<ModelEvent>) {
        let Some(usage) = usage else { return };
        let reasoning_tokens = usage
            .output_tokens_details
            .as_ref()
            .and_then(|details| details.reasoning_tokens);
        let cached_input_tokens = usage.cache_read_input_tokens.or_else(|| {
            usage
                .input_tokens_details
                .as_ref()
                .and_then(|details| (details.cached_tokens > 0).then_some(details.cached_tokens))
        });
        output.push(ModelEvent::Usage {
            input_tokens: usage.input_tokens,
            output_tokens: usage.output_tokens,
            total_tokens: if usage.total_tokens == 0 {
                usage.input_tokens + usage.output_tokens
            } else {
                usage.total_tokens
            },
            reasoning_tokens,
            cached_input_tokens,
        });
        let write_tokens = usage
            .cache_creation_input_tokens
            .or(usage.cache_write_input_tokens)
            .or_else(|| {
                usage
                    .input_tokens_details
                    .as_ref()
                    .and_then(|details| details.cache_write_tokens)
            })
            .unwrap_or(0);
        let read_tokens = cached_input_tokens.unwrap_or(0);
        if read_tokens > 0 || write_tokens > 0 {
            output.push(ModelEvent::Cache {
                hit: read_tokens > 0,
                read_tokens,
                write_tokens,
            });
        }
    }
}

struct ResponsesWebSocketDecoder {
    inner: ResponsesDecoder,
}

impl ModelStreamDecoder for ResponsesWebSocketDecoder {
    fn push(&mut self, chunk: &[u8]) -> Result<Vec<ModelEvent>, ModelFailure> {
        let event: ResponsesStreamEvent = serde_json::from_slice(chunk).map_err(|error| {
            ModelFailure::new(FailurePhase::Decode, FailureKind::MalformedResponse)
                .with_code("malformed_websocket_json")
                .with_detail(error.to_string())
        })?;
        if event.event_type.is_empty() {
            return Err(decode_invalid("websocket event type is required"));
        }
        let data = serde_json::to_string(&event).map_err(|error| {
            ModelFailure::new(FailurePhase::Decode, FailureKind::Internal)
                .with_code("websocket_event_serialization")
                .with_detail(error.to_string())
        })?;
        let framed = format!("event: {}\ndata: {}\n\n", event.event_type, data);
        self.inner.push(framed.as_bytes())
    }

    fn finish(&mut self) -> Result<Vec<ModelEvent>, ModelFailure> {
        self.inner.finish().map_err(|failure| {
            if failure.code.as_deref() == Some("missing_terminal_event") {
                failure.with_code("websocket_closed_before_terminal")
            } else {
                failure
            }
        })
    }
}

impl ModelStreamDecoder for ResponsesDecoder {
    fn push(&mut self, chunk: &[u8]) -> Result<Vec<ModelEvent>, ModelFailure> {
        let events = self.framer.push(chunk).map_err(|error| {
            ModelFailure::new(FailurePhase::Decode, FailureKind::MalformedResponse)
                .with_code("invalid_utf8")
                .with_detail(error.to_string())
        })?;
        self.consume(events)
    }

    fn finish(&mut self) -> Result<Vec<ModelEvent>, ModelFailure> {
        let events = self.framer.finish().map_err(|error| {
            ModelFailure::new(FailurePhase::Finish, FailureKind::MalformedResponse)
                .with_code("invalid_utf8")
                .with_detail(error.to_string())
        })?;
        let output = self.consume(events)?;
        if !self.terminal {
            return Err(
                ModelFailure::new(FailurePhase::Finish, FailureKind::MalformedResponse)
                    .with_code("missing_terminal_event")
                    .with_retry_hint(RetryHint::Retryable),
            );
        }
        Ok(output)
    }
}

fn terminal_status(status: &str) -> Option<TerminalStatus> {
    match status {
        "completed" => Some(TerminalStatus::Completed),
        "incomplete" => Some(TerminalStatus::Incomplete),
        "requires_action" | "tool_use" => Some(TerminalStatus::ToolUse),
        "length" => Some(TerminalStatus::Length),
        "content_filter" => Some(TerminalStatus::ContentFilter),
        "refusal" => Some(TerminalStatus::Refusal),
        "paused" => Some(TerminalStatus::Pause),
        _ => None,
    }
}

fn decode_invalid(detail: &str) -> ModelFailure {
    ModelFailure::new(FailurePhase::Decode, FailureKind::MalformedResponse)
        .with_code("invalid_event")
        .with_detail(detail)
}

fn response_failure(phase: FailurePhase, error: Option<ResponsesError>) -> ModelFailure {
    error_failure(phase, error)
}

fn response_error_string(value: Option<Value>, fallback: &str) -> String {
    match value {
        Some(Value::String(value)) => value,
        Some(Value::Number(value)) => value.to_string(),
        Some(Value::Bool(value)) => value.to_string(),
        Some(value) if !value.is_null() => value.to_string(),
        _ => fallback.to_owned(),
    }
}

fn response_error_status(value: &Value) -> Option<u16> {
    value
        .as_u64()
        .and_then(|value| u16::try_from(value).ok())
        .or_else(|| value.as_str().and_then(|value| value.parse().ok()))
}

fn error_failure(phase: FailurePhase, error: Option<ResponsesError>) -> ModelFailure {
    let error = error.unwrap_or_default();
    let code = response_error_string(error.code.or(error.error_type), "provider_error");
    let normalized = code.to_ascii_lowercase();
    let status = error
        .status
        .as_ref()
        .and_then(response_error_status)
        .or_else(|| {
            normalized
                .strip_prefix("http_")
                .and_then(|value| value.parse::<u16>().ok())
                .or_else(|| normalized.parse::<u16>().ok())
        });
    let kind = if normalized == "misalignment_policy_violation" {
        FailureKind::InvalidRequest
    } else if status == Some(429)
        || normalized.contains("rate_limit")
        || normalized.contains("too_many_requests")
    {
        FailureKind::RateLimited
    } else if status == Some(401)
        || status == Some(403)
        || normalized.contains("auth")
        || normalized.contains("api_key")
        || normalized.contains("unauthorized")
    {
        FailureKind::Authentication
    } else if normalized.contains("timeout") {
        FailureKind::Timeout
    } else {
        FailureKind::Http
    };
    let transient_code = matches!(
        normalized.as_str(),
        "server_error"
            | "internal_error"
            | "service_unavailable"
            | "temporarily_unavailable"
            | "overloaded"
            | "server_is_overloaded"
            | "bad_gateway"
            | "gateway_timeout"
            | "previous_response_not_found"
    );
    let retry_hint = match kind {
        FailureKind::InvalidRequest if normalized == "misalignment_policy_violation" => {
            RetryHint::Never
        }
        FailureKind::RateLimited | FailureKind::Timeout => RetryHint::Retryable,
        FailureKind::Http if status.is_some_and(|value| value >= 500) || transient_code => {
            RetryHint::Retryable
        }
        _ => RetryHint::Never,
    };
    let detail = error
        .message
        .map(|value| response_error_string(Some(value), ""))
        .filter(|value| !value.is_empty())
        .or(error.param)
        .or_else(|| error.detail.map(|value| value.to_string()))
        .unwrap_or_else(|| "provider reported a Responses error".into());
    let failure = ModelFailure::new(phase, kind)
        .with_code(code)
        .with_retry_hint(retry_hint)
        .with_detail(detail);
    match status {
        Some(status) => failure.with_status(status),
        None => failure,
    }
}

#[cfg(test)]
#[path = "../../tests/internal/model_runtime/adapters/tests.rs"]
mod tests;
