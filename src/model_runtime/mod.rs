//! Provider-neutral model runtime used by production turns and helper calls.
//!
//! Protocol implementations are supplied explicitly through immutable bindings;
//! this module has no provider SDK or global mutable registry.
//!
//! The contract surface includes adapter validation and introspection APIs that
//! are not all called by the binary entrypoint.
#![allow(dead_code)]

pub(crate) mod adapters;
pub(crate) mod decorator;
pub(crate) mod projection;
pub(crate) mod runtime;
pub(crate) mod websocket;

use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::BTreeMap;
use std::env;
use std::fmt;
use std::pin::Pin;
use std::str::FromStr;
use std::sync::Arc;
use std::time::Duration;

const MAX_DETAIL_CHARS: usize = 512 * 1024;
const MAX_FAILURE_CODE_BYTES: usize = 256;
const SENSITIVE_DETAIL_KEYS: &[&str] = &[
    "proxy_authorization",
    "authorization",
    "client_secret",
    "access_token",
    "refresh_token",
    "client_token",
    "api_token",
    "x_api_key",
    "api_key",
    "apikey",
    "id_token",
    "password",
    "passwd",
    "credentials",
    "credential",
    "secret",
    "token",
];
const SENSITIVE_DETAIL_LABELS: &[&str] = &[
    "proxy-authorization",
    "proxy_authorization",
    "proxy authorization",
    "authorization",
    "client-secret",
    "client_secret",
    "client secret",
    "access-token",
    "access_token",
    "access token",
    "refresh-token",
    "refresh_token",
    "refresh token",
    "client-token",
    "client_token",
    "client token",
    "api-token",
    "api_token",
    "api token",
    "x-api-key",
    "x_api_key",
    "x api key",
    "api-key",
    "api_key",
    "api key",
    "apikey",
    "id-token",
    "id_token",
    "id token",
    "password",
    "passwd",
    "credentials",
    "credential",
    "secret",
    "token",
];
const WHITESPACE_SEPARATED_DETAIL_LABELS: &[&str] = &[
    "proxy-authorization",
    "proxy_authorization",
    "proxy authorization",
    "authorization",
    "client-secret",
    "client_secret",
    "client secret",
    "access-token",
    "access_token",
    "access token",
    "refresh-token",
    "refresh_token",
    "refresh token",
    "client-token",
    "client_token",
    "client token",
    "api-token",
    "api_token",
    "api token",
    "x-api-key",
    "x_api_key",
    "x api key",
    "api-key",
    "api_key",
    "api key",
    "apikey",
    "id-token",
    "id_token",
    "id token",
    "password",
    "passwd",
];
const MAX_IDENTIFIER_BYTES: usize = 128;

/// A validated and extensible protocol identifier.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Ord, PartialOrd, Serialize, Deserialize)]
#[serde(try_from = "String")]
pub struct ProtocolId(String);

impl ProtocolId {
    pub fn new(value: impl AsRef<str>) -> Result<Self, ProtocolIdError> {
        let value = value.as_ref();
        validate_identifier(value, "protocol id")?;
        Ok(Self(value.to_owned()))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl TryFrom<String> for ProtocolId {
    type Error = ProtocolIdError;

    fn try_from(value: String) -> Result<Self, Self::Error> {
        Self::new(value)
    }
}

impl FromStr for ProtocolId {
    type Err = ProtocolIdError;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        Self::new(value)
    }
}

impl TryFrom<&str> for ProtocolId {
    type Error = ProtocolIdError;

    fn try_from(value: &str) -> Result<Self, Self::Error> {
        Self::new(value)
    }
}

impl fmt::Display for ProtocolId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ProtocolIdError {
    Empty,
    Invalid(String),
    TooLong,
}

impl fmt::Display for ProtocolIdError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Empty => f.write_str("identifier must not be empty"),
            Self::Invalid(label) => write!(f, "{label} contains invalid characters"),
            Self::TooLong => write!(f, "identifier exceeds {MAX_IDENTIFIER_BYTES} bytes"),
        }
    }
}

impl std::error::Error for ProtocolIdError {}

fn validate_identifier(value: &str, label: &str) -> Result<(), ProtocolIdError> {
    if value.is_empty() {
        return Err(ProtocolIdError::Empty);
    }
    if value.len() > MAX_IDENTIFIER_BYTES {
        return Err(ProtocolIdError::TooLong);
    }
    if value.trim() != value || value.chars().any(char::is_control) {
        return Err(ProtocolIdError::Invalid(label.to_owned()));
    }
    Ok(())
}

/// Identity of a profile selected by a route.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Ord, PartialOrd, Serialize, Deserialize)]
#[serde(try_from = "String")]
pub struct ProfileIdentity(String);

impl ProfileIdentity {
    pub fn new(value: impl AsRef<str>) -> Result<Self, ProtocolIdError> {
        let value = value.as_ref();
        validate_identifier(value, "profile identity")?;
        Ok(Self(value.to_owned()))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl TryFrom<String> for ProfileIdentity {
    type Error = ProtocolIdError;

    fn try_from(value: String) -> Result<Self, Self::Error> {
        Self::new(value)
    }
}

impl FromStr for ProfileIdentity {
    type Err = ProtocolIdError;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        Self::new(value)
    }
}

impl fmt::Display for ProfileIdentity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Identity of an adapter flavor selected for a route.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Ord, PartialOrd, Serialize, Deserialize)]
#[serde(try_from = "String")]
pub struct BindingFlavor(String);

impl BindingFlavor {
    pub fn new(value: impl AsRef<str>) -> Result<Self, ProtocolIdError> {
        let value = value.as_ref();
        validate_identifier(value, "binding flavor")?;
        Ok(Self(value.to_owned()))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl TryFrom<String> for BindingFlavor {
    type Error = ProtocolIdError;

    fn try_from(value: String) -> Result<Self, Self::Error> {
        Self::new(value)
    }
}

impl FromStr for BindingFlavor {
    type Err = ProtocolIdError;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        Self::new(value)
    }
}

/// Identity of a configured route within a profile.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Ord, PartialOrd, Serialize, Deserialize)]
#[serde(try_from = "String")]
pub struct RouteIdentity(String);

impl RouteIdentity {
    pub fn new(value: impl AsRef<str>) -> Result<Self, ProtocolIdError> {
        let value = value.as_ref();
        validate_identifier(value, "route identity")?;
        Ok(Self(value.to_owned()))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl TryFrom<String> for RouteIdentity {
    type Error = ProtocolIdError;

    fn try_from(value: String) -> Result<Self, Self::Error> {
        Self::new(value)
    }
}

impl FromStr for RouteIdentity {
    type Err = ProtocolIdError;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        Self::new(value)
    }
}

/// Stable identity used to decide whether adapter-owned replay data can be
/// reused by a target binding.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BindingIdentity {
    pub protocol_id: ProtocolId,
    pub profile_identity: ProfileIdentity,
    pub route_identity: RouteIdentity,
}

impl BindingIdentity {
    pub fn new(
        protocol_id: ProtocolId,
        profile_identity: ProfileIdentity,
        route_identity: RouteIdentity,
    ) -> Self {
        Self {
            protocol_id,
            profile_identity,
            route_identity,
        }
    }
}

/// Route-bound profile data passed while creating a binding.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct ProfileSettings {
    pub display_name: Option<String>,
    pub model: Option<String>,
}

/// Adapter-owned protocol settings preserved as typed TOML data.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(transparent)]
pub struct ProtocolSettings {
    pub value: toml::Value,
}

impl Default for ProtocolSettings {
    fn default() -> Self {
        Self {
            value: toml::Value::Table(toml::map::Map::new()),
        }
    }
}

impl ProtocolSettings {
    pub fn new(value: toml::Value) -> Self {
        Self { value }
    }

    /// Deserialize adapter-owned settings with the adapter's own strict schema.
    /// The runtime deliberately does not interpret vendor fields itself.
    pub fn deserialize_strict<T: DeserializeOwned>(&self) -> Result<T, toml::de::Error> {
        self.value.clone().try_into()
    }
}

/// Route capabilities that are known before a request is prepared.
/// All capabilities are default-off until a route explicitly advertises them.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct RouteCapabilities {
    pub tools: bool,
    pub parallel_tool_calls: bool,
    pub reasoning: bool,
    pub input_images: bool,
    pub tool_result_images: bool,
    pub prompt_cache: bool,
    pub priority_service: bool,
}

/// Generation features supported by a route.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct GenerationSupport {
    pub temperature: bool,
    pub top_p: bool,
    pub max_output_tokens: bool,
    pub stop_sequences: bool,
    pub reasoning: bool,
    pub reasoning_summary: bool,
    pub text_verbosity: bool,
    pub parallel_tool_calls: bool,
    pub priority_service: bool,
}

/// Input accepted by an adapter. Requests and replay data are intentionally
/// absent; they are supplied only to a route-bound binding at request time.
#[derive(Debug, Clone, PartialEq)]
pub struct ProtocolBindInput {
    pub binding_identity: BindingIdentity,
    pub endpoint: String,
    pub flavor: BindingFlavor,
    pub profile: ProfileSettings,
    pub protocol_settings: ProtocolSettings,
    pub capabilities: RouteCapabilities,
    pub generation_support: GenerationSupport,
}

impl ProtocolBindInput {
    pub fn new(
        binding_identity: BindingIdentity,
        endpoint: impl Into<String>,
        flavor: BindingFlavor,
        profile: ProfileSettings,
        protocol_settings: ProtocolSettings,
        capabilities: RouteCapabilities,
        generation_support: GenerationSupport,
    ) -> Self {
        Self {
            binding_identity,
            endpoint: endpoint.into(),
            flavor,
            profile,
            protocol_settings,
            capabilities,
            generation_support,
        }
    }
}

/// An immutable registry of explicitly injected protocol adapters.
#[derive(Clone)]
pub struct ProtocolRegistry {
    adapters: Arc<BTreeMap<ProtocolId, Arc<dyn ProtocolAdapter>>>,
}

impl ProtocolRegistry {
    pub fn new(
        adapters: impl IntoIterator<Item = Arc<dyn ProtocolAdapter>>,
    ) -> Result<Self, RegistryError> {
        let mut registered = BTreeMap::new();
        for adapter in adapters {
            let protocol_id = adapter.protocol_id();
            if registered.insert(protocol_id.clone(), adapter).is_some() {
                return Err(RegistryError::DuplicateProtocol(protocol_id));
            }
        }
        Ok(Self {
            adapters: Arc::new(registered),
        })
    }

    /// Returns the executable built-in Responses, Completions, and Anthropic adapters.
    pub fn builtins() -> Self {
        Self::new(adapters::builtins()).expect("built-in protocol ids are unique")
    }

    pub fn lookup(&self, protocol_id: &ProtocolId) -> Option<Arc<dyn ProtocolAdapter>> {
        self.adapters.get(protocol_id).cloned()
    }

    pub fn lookup_str(&self, value: &str) -> Result<Arc<dyn ProtocolAdapter>, LookupError> {
        let protocol_id = ProtocolId::new(value).map_err(LookupError::InvalidId)?;
        self.lookup(&protocol_id)
            .ok_or(LookupError::NotRegistered(protocol_id))
    }

    pub fn ids(&self) -> impl Iterator<Item = &ProtocolId> {
        self.adapters.keys()
    }

    pub fn len(&self) -> usize {
        self.adapters.len()
    }

    pub fn is_empty(&self) -> bool {
        self.adapters.is_empty()
    }
}

impl fmt::Debug for ProtocolRegistry {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ProtocolRegistry")
            .field("protocol_ids", &self.ids().collect::<Vec<_>>())
            .finish()
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RegistryError {
    DuplicateProtocol(ProtocolId),
}

impl fmt::Display for RegistryError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DuplicateProtocol(protocol_id) => {
                write!(f, "protocol adapter already registered: {protocol_id}")
            }
        }
    }
}

impl std::error::Error for RegistryError {}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LookupError {
    InvalidId(ProtocolIdError),
    NotRegistered(ProtocolId),
}

impl fmt::Display for LookupError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidId(error) => error.fmt(f),
            Self::NotRegistered(protocol_id) => {
                write!(f, "protocol adapter is not registered: {protocol_id}")
            }
        }
    }
}

impl std::error::Error for LookupError {}

/// Object-safe protocol adapter; implementations are injected by the caller.
pub trait ProtocolAdapter: Send + Sync {
    fn protocol_id(&self) -> ProtocolId;
    fn default_endpoint_path(&self) -> &str;
    fn bind(&self, input: ProtocolBindInput) -> Result<Arc<dyn ProtocolBinding>, ModelFailure>;
}

/// Protocol-owned binding. Semantic requests are prepared at call time and are
/// not stored in this route-bound object.
pub trait ProtocolBinding: Send + Sync {
    fn binding_identity(&self) -> &BindingIdentity;
    fn flavor(&self) -> &BindingFlavor;
    fn protocol_id(&self) -> &ProtocolId {
        &self.binding_identity().protocol_id
    }
    fn profile_identity(&self) -> &ProfileIdentity {
        &self.binding_identity().profile_identity
    }
    fn replay_scope(&self) -> ReplayScope;
    fn prepare_request(
        &self,
        input: &ModelRequestInput,
    ) -> Result<PreparedHttpRequest, ModelFailure>;
    fn websocket_frame(
        &self,
        _request: &PreparedHttpRequest,
        _previous_response_id: Option<&str>,
        _incremental_prompt_unit_start: Option<usize>,
    ) -> Result<Vec<u8>, ModelFailure> {
        Err(
            ModelFailure::new(FailurePhase::Prepare, FailureKind::InvalidRequest)
                .with_code("websocket_unsupported_protocol"),
        )
    }
    fn inspect_prepared_request(
        &self,
        request: &PreparedHttpRequest,
        stable_request: Option<&PreparedHttpRequest>,
    ) -> Result<PreparedRequestInspection, ModelFailure>;
    fn new_decoder(&self) -> Box<dyn ModelStreamDecoder>;
    fn new_websocket_decoder(&self) -> Box<dyn ModelStreamDecoder> {
        self.new_decoder()
    }

    fn accepts_replay_state(&self, state: &OpaqueReplayState) -> bool {
        self.replay_scope()
            .is_compatible_with(&state.producer, self.binding_identity())
    }
}

/// Incremental decoder for protocol response bytes.
pub trait ModelStreamDecoder: Send {
    fn push(&mut self, chunk: &[u8]) -> Result<Vec<ModelEvent>, ModelFailure>;
    fn finish(&mut self) -> Result<Vec<ModelEvent>, ModelFailure>;
}

/// Typed, provider-neutral semantic request input.
#[derive(Debug, Clone, PartialEq)]
pub struct ModelRequestInput {
    pub control: RequestControl,
    pub segments: Vec<ControlSegment>,
    pub segment_origins: Vec<String>,
    pub messages: Vec<ModelMessage>,
    pub message_origins: Vec<String>,
    pub tools: Vec<ToolDefinition>,
    pub generation: GenerationSettings,
    pub cache: CacheIntent,
}

impl ModelRequestInput {
    pub fn new(model: impl Into<String>, messages: Vec<ModelMessage>) -> Self {
        Self {
            control: RequestControl::new(model),
            segments: Vec::new(),
            segment_origins: Vec::new(),
            messages,
            message_origins: Vec::new(),
            tools: Vec::new(),
            generation: GenerationSettings::default(),
            cache: CacheIntent::default(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RequestControl {
    pub model: String,
    pub stream: bool,
}

impl RequestControl {
    pub fn new(model: impl Into<String>) -> Self {
        Self {
            model: model.into(),
            stream: true,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ControlSegment {
    pub kind: ControlSegmentKind,
    pub text: String,
}

impl ControlSegment {
    pub fn system(text: impl Into<String>) -> Self {
        Self {
            kind: ControlSegmentKind::System,
            text: text.into(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ControlSegmentKind {
    System,
    Developer,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelMessage {
    pub role: MessageRole,
    pub content: Vec<ContentPart>,
}

impl ModelMessage {
    pub fn text(role: MessageRole, text: impl Into<String>) -> Self {
        Self {
            role,
            content: vec![ContentPart::Text(text.into())],
        }
    }

    pub fn user_image(media_type: impl Into<String>, data: impl Into<Vec<u8>>) -> Self {
        Self {
            role: MessageRole::User,
            content: vec![ContentPart::Image {
                media_type: media_type.into(),
                data: data.into(),
            }],
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MessageRole {
    User,
    Assistant,
    Tool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ContentPart {
    Text(String),
    Image {
        media_type: String,
        data: Vec<u8>,
    },
    Reasoning {
        item_id: String,
        text: String,
        replay: Option<OpaqueReplayState>,
    },
    ToolCall {
        id: String,
        name: String,
        arguments: Value,
    },
    ToolResult {
        id: String,
        content: Vec<ContentPart>,
    },
}

pub type Content = ContentPart;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ToolDefinition {
    pub name: String,
    pub description: String,
    pub parameters: Value,
    pub strict: bool,
}

impl ToolDefinition {
    pub fn new(name: impl Into<String>, description: impl Into<String>, parameters: Value) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            parameters,
            strict: true,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct ToolParameters {
    pub schema: Value,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StablePrefixMetadata {
    pub segment_count: usize,
    pub fingerprint: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Default)]
pub struct GenerationSettings {
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub max_output_tokens: Option<u32>,
    pub stop_sequences: Vec<String>,
    pub reasoning: ReasoningIntent,
    pub summary: Option<ReasoningSummary>,
    pub verbosity: Option<Verbosity>,
    pub parallel_tool_calls: Option<bool>,
    pub priority_service: Option<bool>,
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct ReasoningIntent {
    pub enabled: bool,
    pub effort: Option<ReasoningEffort>,
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub enum ReasoningEffort {
    #[default]
    None,
    Minimal,
    Low,
    Medium,
    High,
    Xhigh,
    Max,
    Custom(String),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReasoningSummary {
    Auto,
    Concise,
    Detailed,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Verbosity {
    Low,
    Medium,
    High,
}

/// Request-side cache intent. The adapter derives any provider cache key from
/// the stable prefix and route-bound identity; callers do not supply a key.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct CacheIntent {
    pub enabled: bool,
    pub namespace: Option<String>,
    pub retention: Option<CacheRetention>,
    pub stable_prefix: Option<StablePrefixMetadata>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CacheRetention {
    InMemory,
    #[serde(rename = "24h")]
    TwentyFourHours,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PreparedHttpRequest {
    pub method: HttpMethod,
    pub url: String,
    pub protocol_headers: BTreeMap<String, String>,
    pub body: Vec<u8>,
    pub prompt_unit_origins: Vec<Vec<String>>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PreparedRequestInspection {
    pub request_shape: Vec<u8>,
    pub prompt_units: Vec<PreparedPromptUnitInspection>,
    pub cache: PreparedRequestCacheInspection,
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct PreparedPromptUnitInspection {
    pub identity: Vec<u8>,
    pub semantic_segment_ids: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct PreparedRequestCacheInspection {
    pub hint_serialized: bool,
    pub retention_sent: Option<CacheRetention>,
    pub local_prefix_fingerprint: Option<String>,
    pub routing_key: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HttpMethod {
    Get,
    Post,
}

pub struct TransportResponse {
    pub status: u16,
    pub headers: BTreeMap<String, String>,
    responses_websocket_events: bool,
    body: Pin<Box<dyn futures_util::Stream<Item = Result<Vec<u8>, ModelFailure>> + Send>>,
}

impl fmt::Debug for TransportResponse {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("TransportResponse")
            .field("status", &self.status)
            .field("headers", &self.headers)
            .field(
                "responses_websocket_events",
                &self.responses_websocket_events,
            )
            .finish_non_exhaustive()
    }
}

impl TransportResponse {
    pub fn retry_hint(&self) -> RetryHint {
        let mut headers = reqwest::header::HeaderMap::new();
        for (name, value) in &self.headers {
            if let (Ok(name), Ok(value)) = (
                reqwest::header::HeaderName::from_bytes(name.as_bytes()),
                reqwest::header::HeaderValue::from_str(value),
            ) {
                headers.insert(name, value);
            }
        }
        retry_hint_from_headers(&headers)
    }

    pub async fn next_chunk(&mut self) -> Option<Result<Vec<u8>, ModelFailure>> {
        use futures_util::StreamExt;
        self.body.next().await
    }

    pub(crate) fn uses_responses_websocket_events(&self) -> bool {
        self.responses_websocket_events
    }

    pub(crate) fn from_responses_websocket_stream<S>(
        status: u16,
        headers: BTreeMap<String, String>,
        stream: S,
    ) -> Self
    where
        S: futures_util::Stream<Item = Result<Vec<u8>, ModelFailure>> + Send + 'static,
    {
        Self {
            status,
            headers,
            responses_websocket_events: true,
            body: Box::pin(stream),
        }
    }

    #[cfg(test)]
    fn from_chunks(status: u16, headers: BTreeMap<String, String>, chunks: Vec<Vec<u8>>) -> Self {
        Self::from_results(status, headers, chunks.into_iter().map(Ok).collect())
    }

    #[cfg(test)]
    fn from_results(
        status: u16,
        headers: BTreeMap<String, String>,
        chunks: Vec<Result<Vec<u8>, ModelFailure>>,
    ) -> Self {
        let body = futures_util::stream::iter(chunks);
        Self {
            status,
            headers,
            responses_websocket_events: false,
            body: Box::pin(body),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ModelEvent {
    ReasoningStarted {
        item_id: String,
    },
    ReasoningDelta {
        item_id: String,
        text: String,
    },
    ReasoningSummaryDelta {
        item_id: String,
        summary_index: u64,
        text: String,
    },
    ReasoningSummaryDone {
        item_id: String,
        summary_index: u64,
        text: String,
    },
    ReasoningDone {
        item_id: String,
        text: String,
        replay: Option<OpaqueReplayState>,
    },
    TextDelta {
        text: String,
    },
    ToolStarted {
        id: String,
        name: String,
    },
    ToolArgumentsDelta {
        id: String,
        delta: String,
    },
    ToolDone {
        id: String,
        name: String,
        arguments: Value,
    },
    Usage {
        input_tokens: u64,
        output_tokens: u64,
        total_tokens: u64,
        reasoning_tokens: Option<u64>,
        cached_input_tokens: Option<u64>,
    },
    Cache {
        hit: bool,
        read_tokens: u64,
        write_tokens: u64,
    },
    ResponseMetadata {
        response_id: String,
    },
    Terminal {
        status: TerminalStatus,
    },
    Failure(ModelFailure),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TerminalStatus {
    Completed,
    ToolUse,
    Length,
    ContentFilter,
    Refusal,
    Pause,
    Incomplete,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FailurePhase {
    Bind,
    Prepare,
    Transport,
    Decode,
    Finish,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FailureKind {
    InvalidRequest,
    UnsupportedProtocol,
    Authentication,
    RateLimited,
    Timeout,
    Http,
    MalformedResponse,
    Internal,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RetryHint {
    Never,
    Retryable,
    RetryAfterSeconds(u64),
}

/// Structured failure metadata. Detail is bounded and credential-redacted.
/// Display remains concise; interactive callers can present `detail` separately.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelFailure {
    pub phase: FailurePhase,
    pub kind: FailureKind,
    pub status: Option<u16>,
    pub code: Option<String>,
    pub retry_hint: RetryHint,
    detail: String,
}

impl ModelFailure {
    pub fn new(phase: FailurePhase, kind: FailureKind) -> Self {
        Self {
            phase,
            kind,
            status: None,
            code: None,
            retry_hint: RetryHint::Never,
            detail: String::new(),
        }
    }

    pub fn with_status(mut self, status: u16) -> Self {
        self.status = Some(status);
        self
    }

    pub fn with_code(mut self, code: impl Into<String>) -> Self {
        self.code = Some(redact_code(code.into()));
        self
    }

    pub fn with_retry_hint(mut self, retry_hint: RetryHint) -> Self {
        self.retry_hint = retry_hint;
        self
    }

    pub fn with_detail(mut self, detail: impl Into<String>) -> Self {
        self.detail = redact_detail(detail.into(), &[]);
        self
    }

    pub fn with_detail_redacted(mut self, detail: impl Into<String>, secrets: &[&str]) -> Self {
        self.detail = redact_detail(detail.into(), secrets);
        self
    }

    pub fn detail(&self) -> &str {
        &self.detail
    }
}

fn redact_detail(detail: String, secrets: &[&str]) -> String {
    let (mut detail, structured_json) = redact_json_detail(detail);
    for secret in secrets.iter().filter(|secret| !secret.is_empty()) {
        let json_encoded = serde_json::to_string(secret)
            .ok()
            .and_then(|value| {
                value
                    .strip_prefix('"')?
                    .strip_suffix('"')
                    .map(str::to_owned)
            })
            .unwrap_or_default();
        for candidate in [
            (*secret).to_owned(),
            json_encoded,
            percent_encode(secret, false, false),
            percent_encode(secret, true, false),
            percent_encode(secret, false, true),
            percent_encode(secret, true, true),
        ]
        .into_iter()
        .filter(|candidate| !candidate.is_empty())
        {
            detail = detail.replace(candidate.as_str(), "[redacted]");
        }
    }
    let normalized = detail
        .replace("\r\n", "\n")
        .replace('\r', "\n")
        .chars()
        .map(|character| match character {
            '\n' | '\t' => character,
            character if character.is_control() => '�',
            character => character,
        })
        .collect::<String>();
    let mut redacted = if structured_json {
        normalized
    } else {
        redact_labeled_secrets(normalized)
    }
    .trim()
    .to_owned();
    if redacted.chars().count() > MAX_DETAIL_CHARS {
        redacted = redacted.chars().take(MAX_DETAIL_CHARS).collect();
        redacted.push_str("\n[provider detail truncated]");
    }
    redacted
}

fn redact_json_detail(detail: String) -> (String, bool) {
    let Ok(mut value) = serde_json::from_str::<Value>(&detail) else {
        return (detail, false);
    };
    redact_json_value(&mut value);
    (serde_json::to_string(&value).unwrap_or(detail), true)
}

fn redact_json_value(value: &mut Value) {
    match value {
        Value::Object(object) => {
            for (key, value) in object {
                if is_sensitive_detail_key(key) {
                    *value = Value::String("[redacted]".into());
                } else {
                    redact_json_value(value);
                }
            }
        }
        Value::Array(values) => values.iter_mut().for_each(redact_json_value),
        _ => {}
    }
}

fn is_sensitive_detail_key(key: &str) -> bool {
    let normalized = canonical_detail_name(key);
    SENSITIVE_DETAIL_KEYS.contains(&normalized.as_str())
}

fn canonical_detail_name(value: &str) -> String {
    let characters = value.trim().chars().collect::<Vec<_>>();
    let mut normalized = String::new();
    for (index, character) in characters.iter().copied().enumerate() {
        if !character.is_ascii_alphanumeric() {
            if !normalized.is_empty() && !normalized.ends_with('_') {
                normalized.push('_');
            }
            continue;
        }
        let previous = index.checked_sub(1).and_then(|index| characters.get(index));
        let next = characters.get(index + 1);
        let word_boundary = character.is_ascii_uppercase()
            && !normalized.is_empty()
            && !normalized.ends_with('_')
            && (previous.is_some_and(|character| character.is_ascii_lowercase())
                || previous.is_some_and(|character| character.is_ascii_uppercase())
                    && next.is_some_and(|character| character.is_ascii_lowercase()));
        if word_boundary {
            normalized.push('_');
        }
        normalized.push(character.to_ascii_lowercase());
    }
    normalized.trim_matches('_').to_owned()
}

fn redact_labeled_secrets(mut value: String) -> String {
    let mut scan_from = 0;
    while scan_from < value.len() {
        let lower = value.to_ascii_lowercase();
        let Some((_, value_start, whitespace_separated, authorization_label)) =
            SENSITIVE_DETAIL_LABELS
                .iter()
                .filter_map(|key| {
                    let mut search_from = scan_from;
                    while let Some(relative) = lower[search_from..].find(key) {
                        let start = search_from + relative;
                        let before = lower[..start].chars().next_back();
                        if before.is_some_and(|character| {
                            character.is_ascii_alphanumeric() || character == '_'
                        }) {
                            search_from = start + key.len();
                            continue;
                        }
                        let key_end = start + key.len();
                        let mut cursor = key_end;
                        while let Some(character) = lower[cursor..].chars().next() {
                            if matches!(character, '"' | '\'') {
                                cursor += character.len_utf8();
                            } else {
                                break;
                            }
                        }
                        if lower[cursor..].starts_with(':') || lower[cursor..].starts_with('=') {
                            let whitespace_separated =
                                WHITESPACE_SEPARATED_DETAIL_LABELS.contains(key);
                            let authorization_label = matches!(
                                *key,
                                "proxy-authorization"
                                    | "proxy_authorization"
                                    | "proxy authorization"
                                    | "authorization"
                            );
                            return Some((
                                start,
                                cursor + 1,
                                whitespace_separated,
                                authorization_label,
                            ));
                        }
                        if WHITESPACE_SEPARATED_DETAIL_LABELS.contains(key) {
                            let mut value_start = key_end;
                            while let Some(character) = lower[value_start..].chars().next() {
                                if character.is_ascii_whitespace() {
                                    value_start += character.len_utf8();
                                } else {
                                    break;
                                }
                            }
                            if value_start > key_end {
                                let authorization_label = matches!(
                                    *key,
                                    "proxy-authorization"
                                        | "proxy_authorization"
                                        | "proxy authorization"
                                        | "authorization"
                                );
                                return Some((start, value_start, true, authorization_label));
                            }
                        }
                        search_from = key_end;
                    }
                    None
                })
                .min_by_key(|(start, _, _, _)| *start)
        else {
            break;
        };

        let mut value_start = value_start;
        while let Some(character) = value[value_start..].chars().next() {
            if character.is_ascii_whitespace() {
                value_start += character.len_utf8();
            } else {
                break;
            }
        }
        let quote = value[value_start..]
            .chars()
            .next()
            .filter(|character| matches!(character, '"' | '\''));
        if let Some(quote) = quote {
            value_start += quote.len_utf8();
        }
        let value_end = if whitespace_separated && quote.is_none() {
            whitespace_secret_end(&value, value_start, authorization_label)
        } else {
            value[value_start..]
                .char_indices()
                .find_map(|(offset, character)| {
                    let quoted_end = quote.is_some_and(|quote| character == quote);
                    let unquoted_end = quote.is_none()
                        && matches!(character, '\n' | ',' | ';' | '&' | '<' | '>' | '}' | ']');
                    (quoted_end || unquoted_end).then_some(value_start + offset)
                })
                .unwrap_or(value.len())
        };
        value.replace_range(value_start..value_end, "[redacted]");
        scan_from = value_start + "[redacted]".len();
    }
    value
}

fn whitespace_secret_end(value: &str, start: usize, authorization_label: bool) -> usize {
    let line_end = value[start..]
        .char_indices()
        .find_map(|(offset, character)| (character == '\n').then_some(start + offset))
        .unwrap_or(value.len());
    if !authorization_label {
        return value[start..line_end]
            .char_indices()
            .find_map(|(offset, character)| {
                character.is_ascii_whitespace().then_some(start + offset)
            })
            .unwrap_or(line_end);
    }
    let first_end = value[start..]
        .char_indices()
        .find_map(|(offset, character)| {
            (character.is_ascii_whitespace()
                || matches!(character, ',' | ';' | '&' | '<' | '>' | '}' | ']'))
            .then_some(start + offset)
        })
        .unwrap_or(value.len());
    if !authorization_label
        || !matches!(
            value[start..first_end].to_ascii_lowercase().as_str(),
            "bearer" | "basic" | "token" | "apikey" | "api-key"
        )
    {
        return first_end;
    }

    let mut credential_start = first_end;
    while let Some(character) = value[credential_start..].chars().next() {
        if character.is_ascii_whitespace() {
            credential_start += character.len_utf8();
        } else {
            break;
        }
    }
    value[credential_start..]
        .char_indices()
        .find_map(|(offset, character)| {
            (character.is_ascii_whitespace()
                || matches!(character, ',' | ';' | '&' | '<' | '>' | '}' | ']'))
            .then_some(credential_start + offset)
        })
        .unwrap_or(value.len())
}

fn percent_encode(value: &str, form: bool, lowercase: bool) -> String {
    const UPPER_HEX: &[u8; 16] = b"0123456789ABCDEF";
    const LOWER_HEX: &[u8; 16] = b"0123456789abcdef";
    let hex = if lowercase { LOWER_HEX } else { UPPER_HEX };
    let mut encoded = String::new();
    for byte in value.bytes() {
        if byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'.' | b'_' | b'~') {
            encoded.push(byte as char);
        } else if form && byte == b' ' {
            encoded.push('+');
        } else {
            encoded.push('%');
            encoded.push(hex[(byte >> 4) as usize] as char);
            encoded.push(hex[(byte & 0x0f) as usize] as char);
        }
    }
    encoded
}

fn redact_code(code: String) -> String {
    let code = code.trim();
    if !code.is_empty()
        && code.len() <= MAX_FAILURE_CODE_BYTES
        && code.chars().all(|character| {
            character.is_ascii_alphanumeric() || matches!(character, '_' | '-' | '.')
        })
    {
        code.to_owned()
    } else {
        "[redacted]".to_owned()
    }
}

impl fmt::Display for ModelFailure {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "model {:?} failure during {:?}", self.kind, self.phase)?;
        if let Some(status) = self.status {
            write!(f, " (status {status})")?;
        }
        if let Some(code) = &self.code {
            write!(f, ", code {code}")?;
        }
        write!(f, ", retry hint {:?}", self.retry_hint)
    }
}

impl std::error::Error for ModelFailure {}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ReplayScope {
    None,
    Protocol,
    Profile,
    Route,
}

impl ReplayScope {
    pub fn is_compatible_with(self, producer: &ReplayProducer, target: &BindingIdentity) -> bool {
        if producer.scope != self {
            return false;
        }
        match self {
            Self::None => false,
            Self::Protocol => producer.protocol_id == target.protocol_id,
            Self::Profile => {
                producer.protocol_id == target.protocol_id
                    && producer.profile_identity.as_ref() == Some(&target.profile_identity)
            }
            Self::Route => {
                producer.protocol_id == target.protocol_id
                    && producer.profile_identity.as_ref() == Some(&target.profile_identity)
                    && producer.route_identity.as_ref() == Some(&target.route_identity)
            }
        }
    }
}

/// Identity of the binding that produced replay material.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ReplayProducer {
    pub scope: ReplayScope,
    pub protocol_id: ProtocolId,
    pub profile_identity: Option<ProfileIdentity>,
    pub route_identity: Option<RouteIdentity>,
}

impl ReplayProducer {
    pub fn new(scope: ReplayScope, binding: &BindingIdentity) -> Self {
        let (profile_identity, route_identity) = match scope {
            ReplayScope::None | ReplayScope::Protocol => (None, None),
            ReplayScope::Profile => (Some(binding.profile_identity.clone()), None),
            ReplayScope::Route => (
                Some(binding.profile_identity.clone()),
                Some(binding.route_identity.clone()),
            ),
        };
        Self {
            scope,
            protocol_id: binding.protocol_id.clone(),
            profile_identity,
            route_identity,
        }
    }
}

/// Opaque adapter-owned replay material with explicit provenance metadata.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OpaqueReplayState {
    pub namespace: String,
    pub version: u32,
    pub producer: ReplayProducer,
    pub payload: Value,
}

impl OpaqueReplayState {
    pub fn new(
        namespace: impl Into<String>,
        version: u32,
        producer: ReplayProducer,
        payload: Value,
    ) -> Self {
        Self {
            namespace: namespace.into(),
            version,
            producer,
            payload,
        }
    }

    pub fn from_anthropic_thinking_blocks_json(raw: &str) -> Option<Self> {
        let payload = serde_json::from_str(raw).ok()?;
        Some(Self::new(
            "anthropic.thinking_blocks",
            1,
            ReplayProducer {
                scope: ReplayScope::Protocol,
                protocol_id: ProtocolId::new("anthropic").ok()?,
                profile_identity: None,
                route_identity: None,
            },
            payload,
        ))
    }

    pub fn payload_json(&self) -> Option<String> {
        serde_json::to_string(&self.payload).ok()
    }
}

/// The only supported provider/model flavor during the staged runtime cutover.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Ord, PartialOrd, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum ProviderFlavor {
    Standard,
    Deepseek,
}

impl Default for ProviderFlavor {
    fn default() -> Self {
        Self::Standard
    }
}

impl ProviderFlavor {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Standard => "standard",
            Self::Deepseek => "deepseek",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum AuthScheme {
    Bearer,
    Header,
    Query,
    None,
}

pub type AuthMode = AuthScheme;

#[derive(Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RuntimeAuthConfig {
    #[serde(rename = "type")]
    pub scheme: AuthScheme,
    #[serde(default)]
    pub name: Option<String>,
    #[serde(default)]
    pub credential: Option<String>,
    #[serde(default)]
    pub credential_env: Option<String>,
}

impl fmt::Debug for RuntimeAuthConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("RuntimeAuthConfig")
            .field("scheme", &self.scheme)
            .field("name", &self.name)
            .field(
                "credential",
                &self.credential.as_ref().map(|_| "<redacted>"),
            )
            .field("credential_env", &self.credential_env)
            .finish()
    }
}

impl RuntimeAuthConfig {
    pub fn credential_value(&self, provider_name: &str) -> String {
        let default_env = format!(
            "{}_API_KEY",
            provider_name
                .chars()
                .map(|ch| {
                    if ch.is_ascii_alphanumeric() {
                        ch.to_ascii_uppercase()
                    } else {
                        '_'
                    }
                })
                .collect::<String>()
        );
        if let Some(name) = self.credential_env.as_deref() {
            return env::var(name)
                .ok()
                .filter(|value| !value.trim().is_empty())
                .unwrap_or_default();
        }
        env::var(default_env)
            .ok()
            .filter(|value| !value.trim().is_empty())
            .or_else(|| self.credential.clone())
            .unwrap_or_default()
    }

    fn snapshot(&self, provider_name: &str) -> Result<Self, RuntimeConfigError> {
        let credential = self.credential_value(provider_name);
        if !matches!(self.scheme, AuthScheme::None) && credential.trim().is_empty() {
            return Err(RuntimeConfigError::Missing(format!(
                "providers.{provider_name}.auth.credential"
            )));
        }
        Ok(Self {
            scheme: self.scheme,
            name: self.name.clone(),
            credential: Some(credential),
            credential_env: None,
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(deny_unknown_fields)]
pub struct RuntimeEndpointOverride {
    #[serde(default)]
    pub path: Option<String>,
    #[serde(default)]
    pub query: BTreeMap<String, String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RuntimeEndpoints {
    pub base_url: String,
    #[serde(default)]
    pub responses: RuntimeEndpointOverride,
    #[serde(default)]
    pub completions: RuntimeEndpointOverride,
    #[serde(default)]
    pub anthropic: RuntimeEndpointOverride,
}

impl RuntimeEndpoints {
    pub fn endpoint_for(
        &self,
        protocol: &ProtocolId,
        adapter_path: &str,
    ) -> Result<(String, BTreeMap<String, String>), RuntimeConfigError> {
        let base_url =
            reqwest::Url::parse(&self.base_url).map_err(|_| RuntimeConfigError::InvalidValue {
                field: "endpoints.base_url".into(),
                reason: "must be an absolute URL without query parameters".into(),
            })?;
        if base_url.query().is_some() || base_url.fragment().is_some() {
            return Err(RuntimeConfigError::InvalidValue {
                field: "endpoints.base_url".into(),
                reason: "must not contain query parameters or fragments".into(),
            });
        }
        let adapter_path = adapter_path.trim();
        if adapter_path.is_empty() {
            return Err(RuntimeConfigError::InvalidValue {
                field: format!("endpoints.{}.path", protocol),
                reason: "adapter default path must not be empty".into(),
            });
        }
        let override_config = match protocol.as_str() {
            "responses" => &self.responses,
            "completions" => &self.completions,
            "anthropic" => &self.anthropic,
            _ => {
                return Err(RuntimeConfigError::UnknownProtocol(protocol.to_string()));
            }
        };
        let path = override_config.path.as_deref().unwrap_or(adapter_path);
        if path.trim().is_empty() {
            return Err(RuntimeConfigError::InvalidValue {
                field: format!("endpoints.{}.path", protocol),
                reason: "must not be empty".into(),
            });
        }
        let base = self.base_url.trim_end_matches('/');
        let value = if path.starts_with("http://") || path.starts_with("https://") {
            path.to_owned()
        } else {
            format!("{base}/{}", path.trim_start_matches('/'))
        };
        if value.is_empty() {
            return Err(RuntimeConfigError::InvalidValue {
                field: "endpoints.base_url".into(),
                reason: "must not be empty".into(),
            });
        }
        let url = reqwest::Url::parse(&value).map_err(|_| RuntimeConfigError::InvalidValue {
            field: "endpoints.base_url".into(),
            reason: "must be an absolute URL".into(),
        })?;
        if url.query().is_some() || url.fragment().is_some() {
            return Err(RuntimeConfigError::InvalidValue {
                field: format!("endpoints.{}.path", protocol),
                reason: "must not contain query parameters or fragments; use query table".into(),
            });
        }
        if !matches!(url.scheme(), "http" | "https") {
            return Err(RuntimeConfigError::InvalidValue {
                field: "endpoints.base_url".into(),
                reason: "must use http or https".into(),
            });
        }
        for key in override_config.query.keys() {
            if key.trim().is_empty() {
                return Err(RuntimeConfigError::InvalidValue {
                    field: format!("endpoints.{}.query", protocol),
                    reason: "query keys must not be empty".into(),
                });
            }
        }
        Ok((value, override_config.query.clone()))
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RuntimeTransportConfig {
    #[serde(default = "default_connect_timeout_secs")]
    pub connect_timeout_secs: u64,
    #[serde(default = "default_true")]
    pub no_proxy_loopback: bool,
    #[serde(default)]
    pub websocket: bool,
}

impl Default for RuntimeTransportConfig {
    fn default() -> Self {
        Self {
            connect_timeout_secs: default_connect_timeout_secs(),
            no_proxy_loopback: true,
            websocket: false,
        }
    }
}

fn default_connect_timeout_secs() -> u64 {
    10
}

fn default_true() -> bool {
    true
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(deny_unknown_fields)]
pub struct RuntimeCapabilities {
    #[serde(default)]
    pub tools: bool,
    #[serde(default)]
    pub parallel_tool_calls: bool,
    #[serde(default)]
    pub reasoning: bool,
    #[serde(default)]
    pub input_images: bool,
    #[serde(default)]
    pub tool_result_images: bool,
    #[serde(default)]
    pub prompt_cache: bool,
    #[serde(default)]
    pub priority_service: bool,
    #[serde(default)]
    pub generation: RuntimeGenerationConfig,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(deny_unknown_fields)]
pub struct RuntimeGenerationConfig {
    #[serde(default)]
    pub temperature: bool,
    #[serde(default)]
    pub top_p: bool,
    #[serde(default)]
    pub max_output_tokens: bool,
    #[serde(default)]
    pub stop_sequences: bool,
    #[serde(default)]
    pub reasoning: bool,
    #[serde(default)]
    pub reasoning_summary: bool,
    #[serde(default)]
    pub text_verbosity: bool,
    #[serde(default)]
    pub parallel_tool_calls: bool,
    #[serde(default)]
    pub priority_service: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
#[serde(deny_unknown_fields)]
pub struct RuntimeGenerationDefaults {
    #[serde(default)]
    pub temperature: Option<f32>,
    #[serde(default)]
    pub top_p: Option<f32>,
    #[serde(default)]
    pub max_output_tokens: Option<u64>,
    #[serde(default)]
    pub reasoning_effort: Option<String>,
    #[serde(default)]
    pub reasoning_efforts: Vec<String>,
    #[serde(default)]
    pub reasoning_summary: Option<String>,
    #[serde(default)]
    pub text_verbosity: Option<String>,
    #[serde(default)]
    pub parallel_tool_calls: Option<bool>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(deny_unknown_fields)]
pub struct RuntimeCacheConfig {
    #[serde(default)]
    pub enabled: bool,
    #[serde(default)]
    pub retention: Option<CacheRetention>,
    #[serde(default)]
    pub namespace: Option<String>,
}

impl From<RuntimeCapabilities> for RouteCapabilities {
    fn from(value: RuntimeCapabilities) -> Self {
        Self {
            tools: value.tools,
            parallel_tool_calls: value.parallel_tool_calls,
            reasoning: value.reasoning,
            input_images: value.input_images,
            tool_result_images: value.tool_result_images,
            prompt_cache: value.prompt_cache,
            priority_service: value.priority_service,
        }
    }
}

impl From<RuntimeGenerationDefaults> for GenerationSupport {
    fn from(value: RuntimeGenerationDefaults) -> Self {
        Self {
            temperature: value.temperature.is_some(),
            top_p: value.top_p.is_some(),
            max_output_tokens: value.max_output_tokens.is_some(),
            stop_sequences: false,
            reasoning: value.reasoning_effort.is_some() || !value.reasoning_efforts.is_empty(),
            reasoning_summary: value.reasoning_summary.is_some(),
            text_verbosity: value.text_verbosity.is_some(),
            parallel_tool_calls: value.parallel_tool_calls.unwrap_or(false),
            priority_service: false,
        }
    }
}

impl From<RuntimeGenerationConfig> for GenerationSupport {
    fn from(value: RuntimeGenerationConfig) -> Self {
        Self {
            temperature: value.temperature,
            top_p: value.top_p,
            max_output_tokens: value.max_output_tokens,
            stop_sequences: value.stop_sequences,
            reasoning: value.reasoning,
            reasoning_summary: value.reasoning_summary,
            text_verbosity: value.text_verbosity,
            parallel_tool_calls: value.parallel_tool_calls,
            priority_service: value.priority_service,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RuntimeModelConfig {
    #[serde(default)]
    pub display: Option<String>,
    #[serde(default)]
    pub context_window: Option<u64>,
    #[serde(default)]
    pub effective_input_limit_tokens: Option<u64>,
    #[serde(default)]
    pub protocol: Option<String>,
    #[serde(default)]
    pub model_override: Option<String>,
    #[serde(default)]
    pub flavor: Option<ProviderFlavor>,
    #[serde(default)]
    pub capabilities: RuntimeCapabilities,
    #[serde(default)]
    pub generation: RuntimeGenerationDefaults,
    #[serde(default)]
    pub cache: RuntimeCacheConfig,
    #[serde(default)]
    pub protocol_settings: ProtocolSettings,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RuntimeProviderConfig {
    #[serde(default)]
    pub protocol: Option<String>,
    #[serde(default)]
    pub default_model: Option<String>,
    #[serde(default)]
    pub retry: Option<RuntimeRetryConfig>,
    #[serde(default)]
    pub flavor: ProviderFlavor,
    pub auth: RuntimeAuthConfig,
    pub endpoints: RuntimeEndpoints,
    #[serde(default)]
    pub transport: RuntimeTransportConfig,
    #[serde(default)]
    pub headers: BTreeMap<String, String>,
    #[serde(default)]
    pub query: BTreeMap<String, String>,
    #[serde(default)]
    pub models: BTreeMap<String, RuntimeModelConfig>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RuntimeConfig {
    pub active_provider: String,
    pub providers: BTreeMap<String, RuntimeProviderConfig>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RuntimeRetryConfig {
    pub enabled: bool,
    pub max_attempts: usize,
    pub max_recovery_attempts: usize,
    pub initial_delay_secs: u64,
    pub exponential_backoff: bool,
    pub backoff_multiplier: f32,
    pub jitter_secs: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RuntimeConfigError {
    Missing(String),
    InvalidValue {
        field: String,
        reason: String,
    },
    UnknownProtocol(String),
    DuplicateHeader(String),
    ReservedHeader(String),
    IncompatibleFlavor {
        flavor: ProviderFlavor,
        protocol: String,
    },
}

impl fmt::Display for RuntimeConfigError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Missing(field) => write!(f, "missing required config field: {field}"),
            Self::InvalidValue { field, reason } => write!(f, "invalid {field}: {reason}"),
            Self::UnknownProtocol(protocol) => write!(f, "unknown protocol: {protocol}"),
            Self::DuplicateHeader(header) => write!(f, "duplicate header: {header}"),
            Self::ReservedHeader(header) => {
                write!(f, "reserved header cannot be configured: {header}")
            }
            Self::IncompatibleFlavor { flavor, protocol } => {
                write!(
                    f,
                    "provider flavor {flavor:?} is not compatible with protocol {protocol}"
                )
            }
        }
    }
}

impl std::error::Error for RuntimeConfigError {}

impl RuntimeConfig {
    pub fn from_toml(value: &str) -> Result<Self, RuntimeConfigError> {
        let document: toml::Value =
            toml::from_str(value).map_err(|error| RuntimeConfigError::InvalidValue {
                field: "config".into(),
                reason: error.to_string(),
            })?;
        let table = document
            .as_table()
            .ok_or_else(|| RuntimeConfigError::InvalidValue {
                field: "config".into(),
                reason: "root must be a table".into(),
            })?;
        let mut runtime_table = toml::map::Map::new();
        let active_provider = table.get("active_provider").cloned().unwrap_or_else(|| {
            table
                .get("providers")
                .and_then(toml::Value::as_table)
                .and_then(|providers| providers.keys().next().cloned())
                .map(toml::Value::String)
                .unwrap_or_else(|| toml::Value::String(String::new()))
        });
        runtime_table.insert("active_provider".into(), active_provider);
        runtime_table.insert(
            "providers".into(),
            table
                .get("providers")
                .cloned()
                .ok_or_else(|| RuntimeConfigError::Missing("providers".into()))?,
        );
        let config: Self = toml::Value::Table(runtime_table)
            .try_into()
            .map_err(|error| RuntimeConfigError::InvalidValue {
                field: "providers".into(),
                reason: error.to_string(),
            })?;
        config.validate()
    }

    pub fn validate(self) -> Result<Self, RuntimeConfigError> {
        if self.providers.is_empty() {
            return Err(RuntimeConfigError::Missing("providers".into()));
        }
        if self.active_provider.trim().is_empty() {
            return Err(RuntimeConfigError::Missing("active_provider".into()));
        }
        self.providers.get(&self.active_provider).ok_or_else(|| {
            RuntimeConfigError::InvalidValue {
                field: "active_provider".into(),
                reason: "provider is not configured".into(),
            }
        })?;
        for (provider_name, provider) in &self.providers {
            if !matches!(
                provider.flavor,
                ProviderFlavor::Standard | ProviderFlavor::Deepseek
            ) {
                return Err(RuntimeConfigError::InvalidValue {
                    field: format!("providers.{provider_name}.flavor"),
                    reason: "unsupported provider flavor".into(),
                });
            }
            validate_config_identifier(provider_name).map_err(|reason| {
                RuntimeConfigError::InvalidValue {
                    field: format!("providers.{provider_name}"),
                    reason,
                }
            })?;
            validate_auth(provider_name, &provider.auth)?;
            if matches!(provider.auth.scheme, AuthScheme::Header) {
                let name = provider.auth.name.as_deref().unwrap_or_default();
                if provider
                    .headers
                    .keys()
                    .any(|key| key.eq_ignore_ascii_case(name))
                    || is_reserved_header(name) && !name.eq_ignore_ascii_case("x-api-key")
                {
                    return Err(RuntimeConfigError::InvalidValue {
                        field: format!("providers.{provider_name}.auth.name"),
                        reason: "collides with configured or reserved header".into(),
                    });
                }
            }
            if matches!(provider.auth.scheme, AuthScheme::Query) {
                let name = provider.auth.name.as_deref().unwrap_or_default();
                if provider
                    .query
                    .keys()
                    .any(|key| key.eq_ignore_ascii_case(name))
                    || [
                        "authorization",
                        "api_key",
                        "access_token",
                        "model",
                        "stream",
                    ]
                    .iter()
                    .any(|key| key.eq_ignore_ascii_case(name))
                {
                    return Err(RuntimeConfigError::InvalidValue {
                        field: format!("providers.{provider_name}.auth.name"),
                        reason: "collides with configured or reserved query".into(),
                    });
                }
            }
            validate_map(provider_name, "headers", &provider.headers)?;
            validate_map(provider_name, "query", &provider.query)?;
            validate_reserved_headers(provider_name, &provider.headers)?;
            validate_reserved_query(provider_name, &provider.query, &provider.auth)?;
            if provider.models.is_empty() {
                return Err(RuntimeConfigError::Missing(format!(
                    "providers.{provider_name}.models"
                )));
            }
            if let Some(retry) = &provider.retry {
                if retry.max_attempts == 0 || retry.max_recovery_attempts == 0 {
                    return Err(RuntimeConfigError::InvalidValue {
                        field: format!("providers.{provider_name}.retry"),
                        reason: "attempt limits must be greater than zero".into(),
                    });
                }
                if retry.max_attempts > 9000 || retry.max_recovery_attempts > 10 {
                    return Err(RuntimeConfigError::InvalidValue {
                        field: format!("providers.{provider_name}.retry"),
                        reason: "attempt limits exceed configured maximum".into(),
                    });
                }
                if retry.initial_delay_secs == 0
                    || !retry.backoff_multiplier.is_finite()
                    || !(1.0..=10.0).contains(&retry.backoff_multiplier)
                {
                    return Err(RuntimeConfigError::InvalidValue {
                        field: format!("providers.{provider_name}.retry"),
                        reason: "invalid delay or backoff multiplier".into(),
                    });
                }
            }
            if provider.transport.connect_timeout_secs == 0 {
                return Err(RuntimeConfigError::InvalidValue {
                    field: format!("providers.{provider_name}.transport.connect_timeout_secs"),
                    reason: "must be greater than zero".into(),
                });
            }
            if let Some(default_model) = provider.default_model.as_deref() {
                if default_model.trim().is_empty() {
                    return Err(RuntimeConfigError::InvalidValue {
                        field: format!("providers.{provider_name}.default_model"),
                        reason: "must not be empty".into(),
                    });
                }
                if !provider.models.contains_key(default_model) {
                    return Err(RuntimeConfigError::InvalidValue {
                        field: format!("providers.{provider_name}.default_model"),
                        reason: "model is not configured".into(),
                    });
                }
            }
            for (model_name, model) in &provider.models {
                let protocol = model
                    .protocol
                    .as_deref()
                    .or(provider.protocol.as_deref())
                    .unwrap_or("responses");
                let generation = &model.capabilities.generation;
                let effective_flavor = model.flavor.unwrap_or(provider.flavor);
                if effective_flavor == ProviderFlavor::Deepseek && protocol == "anthropic" {
                    return Err(RuntimeConfigError::IncompatibleFlavor {
                        flavor: effective_flavor,
                        protocol: protocol.to_owned(),
                    });
                }
                if model.generation.temperature.is_some() && !generation.temperature {
                    return Err(RuntimeConfigError::InvalidValue {
                        field: format!(
                            "providers.{provider_name}.models.{model_name}.generation.temperature"
                        ),
                        reason: "requires capabilities.generation.temperature = true".into(),
                    });
                }
                if model.generation.top_p.is_some() && !generation.top_p {
                    return Err(RuntimeConfigError::InvalidValue {
                        field: format!(
                            "providers.{provider_name}.models.{model_name}.generation.top_p"
                        ),
                        reason: "requires capabilities.generation.top_p = true".into(),
                    });
                }
                if model.generation.max_output_tokens.is_some() && !generation.max_output_tokens {
                    return Err(RuntimeConfigError::InvalidValue {
                        field: format!(
                            "providers.{provider_name}.models.{model_name}.generation.max_output_tokens"
                        ),
                        reason: "requires capabilities.generation.max_output_tokens = true".into(),
                    });
                }
                if model.effective_input_limit_tokens == Some(0) {
                    return Err(RuntimeConfigError::InvalidValue {
                        field: format!(
                            "providers.{provider_name}.models.{model_name}.effective_input_limit_tokens"
                        ),
                        reason: "must be greater than 0".into(),
                    });
                }
                if (model.generation.reasoning_effort.is_some()
                    || !model.generation.reasoning_efforts.is_empty())
                    && (!model.capabilities.reasoning || !generation.reasoning)
                {
                    return Err(RuntimeConfigError::InvalidValue {
                        field: format!(
                            "providers.{provider_name}.models.{model_name}.generation.reasoning_effort"
                        ),
                        reason: "requires reasoning capabilities".into(),
                    });
                }
                let protocol_id = ProtocolId::new(protocol)
                    .map_err(|_| RuntimeConfigError::UnknownProtocol(protocol.to_owned()))?;
                let adapter_path = adapters::default_endpoint_path(protocol_id.as_str())
                    .ok_or_else(|| RuntimeConfigError::UnknownProtocol(protocol_id.to_string()))?;
                let (_endpoint, endpoint_query) = provider
                    .endpoints
                    .endpoint_for(&protocol_id, adapter_path)?;
                validate_reserved_query(provider_name, &endpoint_query, &provider.auth)?;
                if matches!(provider.auth.scheme, AuthScheme::Query)
                    && endpoint_query.keys().any(|key| {
                        key.eq_ignore_ascii_case(provider.auth.name.as_deref().unwrap_or_default())
                    })
                {
                    return Err(RuntimeConfigError::InvalidValue {
                        field: format!("providers.{provider_name}.auth.name"),
                        reason: "collides with endpoint query".into(),
                    });
                }
                validate_protocol_settings(
                    provider_name,
                    model_name,
                    protocol,
                    &model.protocol_settings,
                )?;
                validate_generation_defaults(provider_name, model_name, protocol, model)?;
                if endpoint_query
                    .keys()
                    .any(|key| provider.query.contains_key(key))
                {
                    return Err(RuntimeConfigError::InvalidValue {
                        field: format!("providers.{provider_name}.models.{model_name}.query"),
                        reason: "endpoint query conflicts with provider query".into(),
                    });
                }
                validate_config_identifier(model_name).map_err(|reason| {
                    RuntimeConfigError::InvalidValue {
                        field: format!("providers.{provider_name}.models.{model_name}"),
                        reason,
                    }
                })?;
                if model
                    .model_override
                    .as_deref()
                    .is_some_and(|value| value.trim().is_empty())
                {
                    return Err(RuntimeConfigError::InvalidValue {
                        field: format!(
                            "providers.{provider_name}.models.{model_name}.model_override"
                        ),
                        reason: "must not be empty".into(),
                    });
                }

                if model.cache.enabled
                    && model.cache.namespace.as_deref().is_some_and(|value| {
                        value.trim().is_empty()
                            || value.len() > 64
                            || value.chars().any(char::is_control)
                    })
                {
                    return Err(RuntimeConfigError::InvalidValue {
                        field: format!(
                            "providers.{provider_name}.models.{model_name}.cache.namespace"
                        ),
                        reason:
                            "must be non-empty, at most 64 bytes, and contain no control characters"
                                .into(),
                    });
                }
                if !model.cache.enabled && model.cache.retention.is_some() {
                    return Err(RuntimeConfigError::InvalidValue {
                        field: format!(
                            "providers.{provider_name}.models.{model_name}.cache.retention"
                        ),
                        reason: "requires enabled = true".into(),
                    });
                }
                if model.cache.retention.is_some() && protocol != "responses" {
                    return Err(RuntimeConfigError::InvalidValue {
                        field: format!(
                            "providers.{provider_name}.models.{model_name}.cache.retention"
                        ),
                        reason: "is only supported for responses protocol".into(),
                    });
                }
                if !matches!(model.protocol_settings.value, toml::Value::Table(_)) {
                    return Err(RuntimeConfigError::InvalidValue {
                        field: format!(
                            "providers.{provider_name}.models.{model_name}.protocol_settings"
                        ),
                        reason: "must be a table".into(),
                    });
                }
            }
        }
        Ok(self)
    }

    pub fn resolve(
        &self,
        registry: &ProtocolRegistry,
    ) -> Result<ResolvedRuntimeCatalog, RuntimeConfigError> {
        let mut providers = BTreeMap::new();
        for (provider_name, provider) in &self.providers {
            providers.insert(
                provider_name.clone(),
                ResolvedProvider::new(provider_name, provider, registry)?,
            );
        }
        ResolvedRuntimeCatalog::new(self.active_provider.clone(), providers)
    }
}

fn validate_auth(provider: &str, auth: &RuntimeAuthConfig) -> Result<(), RuntimeConfigError> {
    if matches!(auth.scheme, AuthScheme::None) {
        if auth.credential.is_some() || auth.credential_env.is_some() {
            return Err(RuntimeConfigError::InvalidValue {
                field: format!("providers.{provider}.auth"),
                reason: "none auth must not configure credentials".into(),
            });
        }
        return Ok(());
    }
    if matches!(auth.scheme, AuthScheme::Header | AuthScheme::Query)
        && auth.name.as_deref().unwrap_or("").trim().is_empty()
    {
        return Err(RuntimeConfigError::Missing(format!(
            "providers.{provider}.auth.name"
        )));
    }
    if let Some(name) = auth.name.as_deref() {
        match auth.scheme {
            AuthScheme::Header
                if reqwest::header::HeaderName::from_bytes(name.as_bytes()).is_err() =>
            {
                return Err(RuntimeConfigError::InvalidValue {
                    field: format!("providers.{provider}.auth.name"),
                    reason: "must be a valid header name".into(),
                });
            }
            AuthScheme::Query if !is_safe_query_key(name) => {
                return Err(RuntimeConfigError::InvalidValue {
                    field: format!("providers.{provider}.auth.name"),
                    reason: "must be a safe query key".into(),
                });
            }
            _ => {}
        }
    }
    if let Some(credential) = auth.credential.as_deref()
        && (credential.contains('\n') || credential.contains('\r'))
    {
        return Err(RuntimeConfigError::InvalidValue {
            field: format!("providers.{provider}.auth.credential"),
            reason: "must not contain newlines".into(),
        });
    }
    if let Some(name) = &auth.credential_env {
        validate_config_identifier(name).map_err(|reason| RuntimeConfigError::InvalidValue {
            field: format!("providers.{provider}.auth.credential_env"),
            reason,
        })?;
    }
    Ok(())
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct AnthropicProtocolSettings {
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

fn validate_protocol_settings(
    provider: &str,
    model: &str,
    protocol: &str,
    settings: &ProtocolSettings,
) -> Result<(), RuntimeConfigError> {
    if protocol == "anthropic" {
        let parsed: AnthropicProtocolSettings =
            settings.value.clone().try_into().map_err(|error| {
                RuntimeConfigError::InvalidValue {
                    field: format!("providers.{provider}.models.{model}.protocol_settings"),
                    reason: error.to_string(),
                }
            })?;
        if let Some(thinking) = parsed.anthropic_thinking {
            if !matches!(thinking.mode.as_str(), "disabled" | "adaptive" | "budget") {
                return Err(RuntimeConfigError::InvalidValue {
                    field: format!(
                        "providers.{provider}.models.{model}.protocol_settings.anthropic_thinking.mode"
                    ),
                    reason: "must be disabled, adaptive, or budget".into(),
                });
            }
            if thinking.mode == "budget" && thinking.budget_tokens.unwrap_or(0) < 1024 {
                return Err(RuntimeConfigError::InvalidValue {
                    field: format!(
                        "providers.{provider}.models.{model}.protocol_settings.anthropic_thinking.budget_tokens"
                    ),
                    reason: "must be at least 1024 for budget thinking".into(),
                });
            }
            if thinking.mode != "budget" && thinking.budget_tokens.is_some() {
                return Err(RuntimeConfigError::InvalidValue {
                    field: format!(
                        "providers.{provider}.models.{model}.protocol_settings.anthropic_thinking.budget_tokens"
                    ),
                    reason: "is only valid for budget thinking".into(),
                });
            }
        }
        let mut seen_betas = std::collections::BTreeSet::new();
        for beta in parsed.anthropic_betas {
            if !seen_betas.insert(beta.clone()) {
                return Err(RuntimeConfigError::InvalidValue {
                    field: format!(
                        "providers.{provider}.models.{model}.protocol_settings.anthropic_betas"
                    ),
                    reason: "must not contain duplicate beta values".into(),
                });
            }
            if beta.trim().is_empty()
                || beta.trim() != beta
                || beta.len() > 128
                || beta.chars().any(|ch| ch.is_control() || ch == ',')
                || reqwest::header::HeaderValue::from_str(&beta).is_err()
            {
                return Err(RuntimeConfigError::InvalidValue {
                    field: format!(
                        "providers.{provider}.models.{model}.protocol_settings.anthropic_betas"
                    ),
                    reason: "entries must be valid HTTP header values".into(),
                });
            }
        }
        return Ok(());
    }
    let allowed: &[&str] = &[];
    if protocol != "responses" && protocol != "completions" {
        return Err(RuntimeConfigError::UnknownProtocol(protocol.to_owned()));
    }
    let Some(table) = settings.value.as_table() else {
        return Err(RuntimeConfigError::InvalidValue {
            field: format!("providers.{provider}.models.{model}.protocol_settings"),
            reason: "must be a table".into(),
        });
    };
    for key in table.keys() {
        if !allowed.contains(&key.as_str()) {
            return Err(RuntimeConfigError::InvalidValue {
                field: format!("providers.{provider}.models.{model}.protocol_settings.{key}"),
                reason: "unknown protocol setting".into(),
            });
        }
    }
    Ok(())
}

fn validate_generation_defaults(
    provider: &str,
    model: &str,
    protocol: &str,
    config: &RuntimeModelConfig,
) -> Result<(), RuntimeConfigError> {
    let path = |field: &str| format!("providers.{provider}.models.{model}.generation.{field}");
    let generation = &config.capabilities.generation;
    if protocol == "responses" && config.capabilities.generation.stop_sequences {
        return Err(RuntimeConfigError::InvalidValue {
            field: path("stop_sequences"),
            reason: "Responses does not support stop sequences".into(),
        });
    }
    if let Some(value) = config.generation.temperature {
        if !value.is_finite() || !(0.0..=2.0).contains(&value) || !generation.temperature {
            return Err(RuntimeConfigError::InvalidValue {
                field: path("temperature"),
                reason: "requires capability and must be between 0 and 2".into(),
            });
        }
    }
    if let Some(value) = config.generation.top_p {
        if !value.is_finite() || !(0.0..=1.0).contains(&value) || !generation.top_p {
            return Err(RuntimeConfigError::InvalidValue {
                field: path("top_p"),
                reason: "requires capability and must be between 0 and 1".into(),
            });
        }
    }
    if let Some(value) = config.generation.max_output_tokens {
        if value == 0 || value > u32::MAX as u64 || !generation.max_output_tokens {
            return Err(RuntimeConfigError::InvalidValue {
                field: path("max_output_tokens"),
                reason: "requires capability and must be between 1 and u32::MAX".into(),
            });
        }
    }
    if config.generation.reasoning_effort.is_some()
        || !config.generation.reasoning_efforts.is_empty()
    {
        if !config.capabilities.reasoning || !generation.reasoning {
            return Err(RuntimeConfigError::InvalidValue {
                field: path("reasoning_effort"),
                reason: "requires reasoning capabilities".into(),
            });
        }
        let valid_effort = |value: &str| {
            !value.trim().is_empty()
                && value.len() <= 64
                && value
                    .bytes()
                    .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_' | b'.'))
        };
        let mut seen = std::collections::BTreeSet::new();
        for value in &config.generation.reasoning_efforts {
            if !valid_effort(value) {
                return Err(RuntimeConfigError::InvalidValue {
                    field: path("reasoning_efforts"),
                    reason: "must contain valid reasoning identifiers".into(),
                });
            }
            if !seen.insert(value) {
                return Err(RuntimeConfigError::InvalidValue {
                    field: path("reasoning_efforts"),
                    reason: "must not contain duplicate efforts".into(),
                });
            }
        }
        if let Some(default_effort) = &config.generation.reasoning_effort {
            if !valid_effort(default_effort) {
                return Err(RuntimeConfigError::InvalidValue {
                    field: path("reasoning_effort"),
                    reason: "must be a valid reasoning identifier".into(),
                });
            }
        }
        if let Some(default_effort) = &config.generation.reasoning_effort
            && !config.generation.reasoning_efforts.is_empty()
            && !config.generation.reasoning_efforts.contains(default_effort)
        {
            return Err(RuntimeConfigError::InvalidValue {
                field: path("reasoning_efforts"),
                reason: "must include the configured reasoning_effort".into(),
            });
        }
    }
    if let Some(value) = config.generation.reasoning_summary.as_deref()
        && !matches!(value, "auto" | "concise" | "detailed")
    {
        return Err(RuntimeConfigError::InvalidValue {
            field: path("reasoning_summary"),
            reason: "must be auto, concise, or detailed".into(),
        });
    }
    if let Some(value) = config.generation.text_verbosity.as_deref()
        && !matches!(value, "low" | "medium" | "high")
    {
        return Err(RuntimeConfigError::InvalidValue {
            field: path("text_verbosity"),
            reason: "must be low, medium, or high".into(),
        });
    }
    if config.generation.reasoning_summary.is_some()
        && (protocol != "responses"
            || !config.capabilities.reasoning
            || !generation.reasoning_summary)
    {
        return Err(RuntimeConfigError::InvalidValue {
            field: path("reasoning_summary"),
            reason: "requires responses reasoning support".into(),
        });
    }
    if config.generation.text_verbosity.is_some()
        && (protocol != "responses" && protocol != "completions" || !generation.text_verbosity)
    {
        return Err(RuntimeConfigError::InvalidValue {
            field: path("text_verbosity"),
            reason: "requires completions or responses text verbosity support".into(),
        });
    }
    if config.generation.parallel_tool_calls == Some(true)
        && (!config.capabilities.parallel_tool_calls || !generation.parallel_tool_calls)
    {
        return Err(RuntimeConfigError::InvalidValue {
            field: path("parallel_tool_calls"),
            reason: "requires parallel tool-call capabilities".into(),
        });
    }
    Ok(())
}

fn validate_config_identifier(value: &str) -> Result<(), String> {
    if value.is_empty()
        || value.len() > 128
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'_' | b'-' | b'.'))
    {
        return Err("must be 1-128 ASCII letters, digits, '-', '_', or '.'".into());
    }
    Ok(())
}

fn validate_map(
    provider: &str,
    kind: &str,
    values: &BTreeMap<String, String>,
) -> Result<(), RuntimeConfigError> {
    for (key, value) in values {
        if kind == "headers" && reqwest::header::HeaderName::from_bytes(key.as_bytes()).is_err() {
            return Err(RuntimeConfigError::InvalidValue {
                field: format!("providers.{provider}.{kind}.{key}"),
                reason: "must be a valid header name".into(),
            });
        }
        if key.trim().is_empty() || value.contains('\n') || value.contains('\r') {
            return Err(RuntimeConfigError::InvalidValue {
                field: format!("providers.{provider}.{kind}.{key}"),
                reason: "must not be empty or contain newlines".into(),
            });
        }
    }
    Ok(())
}

fn validate_reserved_headers(
    _provider: &str,
    headers: &BTreeMap<String, String>,
) -> Result<(), RuntimeConfigError> {
    for key in headers.keys() {
        if is_reserved_header(key) {
            return Err(RuntimeConfigError::ReservedHeader(key.to_ascii_lowercase()));
        }
    }
    Ok(())
}

fn merge_query(
    provider_query: &BTreeMap<String, String>,
    endpoint_query: &BTreeMap<String, String>,
    provider: &str,
    model: &str,
) -> Result<BTreeMap<String, String>, RuntimeConfigError> {
    let mut merged = provider_query.clone();
    for (key, value) in endpoint_query {
        if let Some(existing) = merged.get(key) {
            return Err(RuntimeConfigError::InvalidValue {
                field: format!("providers.{provider}.models.{model}.query.{key}"),
                reason: format!("conflicts with provider query value '{existing}'"),
            });
        }
        merged.insert(key.clone(), value.clone());
    }
    Ok(merged)
}

fn validate_reserved_query(
    provider: &str,
    query: &BTreeMap<String, String>,
    auth: &RuntimeAuthConfig,
) -> Result<(), RuntimeConfigError> {
    for key in query.keys() {
        let key_lower = key.to_ascii_lowercase();
        if matches!(
            key_lower.as_str(),
            "authorization" | "api_key" | "access_token" | "model" | "stream"
        ) || (matches!(auth.scheme, AuthScheme::Query) && key_lower == "key")
        {
            return Err(RuntimeConfigError::InvalidValue {
                field: format!("providers.{provider}.query.{key}"),
                reason: "conflicts with a reserved request query parameter".into(),
            });
        }
    }
    Ok(())
}

#[derive(Debug, Clone)]
pub struct ProviderTransport {
    client: reqwest::Client,
    connect_timeout: Duration,
    no_proxy_loopback: bool,
    websocket: bool,
}

impl ProviderTransport {
    pub fn new(config: &RuntimeTransportConfig) -> Result<Self, ModelFailure> {
        Self::new_for_endpoint(config, None)
    }

    fn new_for_endpoint(
        config: &RuntimeTransportConfig,
        endpoint: Option<&str>,
    ) -> Result<Self, ModelFailure> {
        if config.connect_timeout_secs == 0 {
            return Err(
                ModelFailure::new(FailurePhase::Bind, FailureKind::InvalidRequest)
                    .with_code("invalid_connect_timeout"),
            );
        }
        let mut builder = reqwest::Client::builder()
            .connect_timeout(Duration::from_secs(config.connect_timeout_secs));
        if config.no_proxy_loopback
            && endpoint.is_some_and(|value| {
                reqwest::Url::parse(value).is_ok_and(|url| {
                    url.host_str().is_some_and(|host| {
                        host.eq_ignore_ascii_case("localhost")
                            || host == "127.0.0.1"
                            || host == "::1"
                    })
                })
            })
        {
            builder = builder.no_proxy();
        }
        let client = builder.build().map_err(|_| {
            ModelFailure::new(FailurePhase::Bind, FailureKind::Internal)
                .with_code("transport_build_failed")
        })?;
        Ok(Self {
            client,
            connect_timeout: Duration::from_secs(config.connect_timeout_secs),
            no_proxy_loopback: config.no_proxy_loopback,
            websocket: config.websocket,
        })
    }

    pub fn client(&self) -> reqwest::Client {
        self.client.clone()
    }

    pub(crate) async fn open_websocket(
        &self,
        protocol: &ProtocolId,
        request: reqwest::RequestBuilder,
        auth: &RuntimeAuthConfig,
    ) -> Result<websocket::TurnLocalWsSession, ModelFailure> {
        if protocol.as_str() != "responses" {
            return Err(
                ModelFailure::new(FailurePhase::Transport, FailureKind::InvalidRequest)
                    .with_code("websocket_unsupported_protocol"),
            );
        }
        if !self.websocket {
            return Err(
                ModelFailure::new(FailurePhase::Transport, FailureKind::InvalidRequest)
                    .with_code("websocket_disabled"),
            );
        }
        let request = request.build().map_err(|error| {
            ModelFailure::new(FailurePhase::Transport, FailureKind::InvalidRequest)
                .with_code("request_build_failed")
                .with_detail_redacted(
                    error.to_string(),
                    &[auth.credential.as_deref().unwrap_or_default()],
                )
        })?;
        let mut owned_secrets = vec![auth.credential.clone().unwrap_or_default()];
        owned_secrets.extend(
            request
                .url()
                .query_pairs()
                .map(|(_, value)| value.into_owned())
                .filter(|value| !value.is_empty()),
        );
        owned_secrets.extend(
            request
                .headers()
                .values()
                .filter_map(|value| value.to_str().ok())
                .filter(|value| value.len() > 1)
                .map(str::to_owned),
        );
        let secrets = owned_secrets.iter().map(String::as_str).collect::<Vec<_>>();
        websocket::TurnLocalWsSession::connect(
            self.client.clone(),
            request,
            self.connect_timeout,
            &secrets,
        )
        .await
    }

    pub async fn send(
        &self,
        request: reqwest::RequestBuilder,
        auth: &RuntimeAuthConfig,
    ) -> Result<TransportResponse, ModelFailure> {
        let request = request.build().map_err(|error| {
            ModelFailure::new(FailurePhase::Transport, FailureKind::InvalidRequest)
                .with_code("request_build_failed")
                .with_detail_redacted(
                    error.to_string(),
                    &[auth.credential.as_deref().unwrap_or_default()],
                )
        })?;
        let mut owned_secrets = vec![auth.credential.clone().unwrap_or_default()];
        owned_secrets.extend(
            request
                .url()
                .query_pairs()
                .map(|(_, value)| value.into_owned())
                .filter(|value| !value.is_empty()),
        );
        owned_secrets.extend(
            request
                .headers()
                .values()
                .filter_map(|value| value.to_str().ok())
                .filter(|value| value.len() > 1)
                .map(str::to_owned),
        );
        let secrets = owned_secrets.iter().map(String::as_str).collect::<Vec<_>>();
        self.send_request(request, &secrets).await
    }

    async fn send_redacted(
        &self,
        request: reqwest::RequestBuilder,
        secrets: &[&str],
    ) -> Result<TransportResponse, ModelFailure> {
        let request = request.build().map_err(|error| {
            ModelFailure::new(FailurePhase::Transport, FailureKind::InvalidRequest)
                .with_code("request_build_failed")
                .with_detail_redacted(error.to_string(), secrets)
        })?;
        self.send_request(request, secrets).await
    }

    async fn send_request(
        &self,
        request: reqwest::Request,
        secrets: &[&str],
    ) -> Result<TransportResponse, ModelFailure> {
        let response = self.client.execute(request).await.map_err(|error| {
            let kind = if error.is_timeout() {
                FailureKind::Timeout
            } else {
                FailureKind::Http
            };
            let retry_hint = if crate::retry::is_retryable_reqwest_error(&error) {
                RetryHint::Retryable
            } else {
                RetryHint::Never
            };
            let mut failure = ModelFailure::new(FailurePhase::Transport, kind)
                .with_code("request_failed")
                .with_retry_hint(retry_hint)
                .with_detail_redacted(error.to_string(), secrets);
            if let Some(status) = error.status() {
                failure = failure.with_status(status.as_u16());
            }
            failure
        })?;
        let status = response.status().as_u16();
        let response_retry_hint = retry_hint_from_headers(response.headers());
        let headers = response
            .headers()
            .iter()
            .filter_map(|(name, value)| {
                Some((name.as_str().to_owned(), value.to_str().ok()?.to_owned()))
            })
            .collect();
        let secrets = secrets
            .iter()
            .map(|secret| (*secret).to_owned())
            .collect::<Vec<_>>();
        use futures_util::StreamExt;
        let body = response.bytes_stream().map(move |result| {
            result.map(|chunk| chunk.to_vec()).map_err(|error| {
                let secret_refs = secrets.iter().map(String::as_str).collect::<Vec<_>>();
                ModelFailure::new(FailurePhase::Transport, FailureKind::Http)
                    .with_status(status)
                    .with_code("response_chunk_failed")
                    .with_retry_hint(response_retry_hint)
                    .with_detail_redacted(error.to_string(), &secret_refs)
            })
        });
        Ok(TransportResponse {
            status,
            headers,
            responses_websocket_events: false,
            body: Box::pin(body),
        })
    }

    pub fn request(
        &self,
        method: reqwest::Method,
        url: &str,
        provider_name: &str,
        auth: &RuntimeAuthConfig,
        headers: &BTreeMap<String, String>,
        query: &BTreeMap<String, String>,
    ) -> Result<reqwest::RequestBuilder, ModelFailure> {
        let request = self.client.request(method, url);
        self.apply_request(request, provider_name, auth, headers, query)
    }

    pub fn connect_timeout(&self) -> Duration {
        self.connect_timeout
    }

    pub fn no_proxy_loopback(&self) -> bool {
        self.no_proxy_loopback
    }

    pub fn websocket(&self) -> bool {
        self.websocket
    }

    pub fn apply_request(
        &self,
        request: reqwest::RequestBuilder,
        _provider_name: &str,
        auth: &RuntimeAuthConfig,
        headers: &BTreeMap<String, String>,
        query: &BTreeMap<String, String>,
    ) -> Result<reqwest::RequestBuilder, ModelFailure> {
        let request = request.build().map_err(|_| {
            ModelFailure::new(FailurePhase::Prepare, FailureKind::InvalidRequest)
                .with_code("request_build_failed")
        })?;
        let method = request.method().clone();
        let mut url = request.url().clone();
        let credential = auth.credential.as_deref().unwrap_or_default();
        {
            let mut pairs = url.query_pairs_mut();
            if matches!(auth.scheme, AuthScheme::Query) {
                let name = auth.name.as_deref().unwrap_or("api_key");
                if !is_safe_query_key(name) {
                    return Err(ModelFailure::new(
                        FailurePhase::Prepare,
                        FailureKind::InvalidRequest,
                    )
                    .with_code("invalid_query_key"));
                }
                pairs.append_pair(name, credential);
            }
            for (key, value) in query {
                if !is_safe_query_key(key) {
                    return Err(ModelFailure::new(
                        FailurePhase::Prepare,
                        FailureKind::InvalidRequest,
                    )
                    .with_code("invalid_query_key"));
                }
                pairs.append_pair(key, value);
            }
        }

        let mut request = self.client.request(method, url);
        match auth.scheme {
            AuthScheme::Bearer => {
                request = request.header(
                    reqwest::header::AUTHORIZATION,
                    format!("Bearer {credential}"),
                );
            }
            AuthScheme::Header => {
                request = request.header(auth.name.as_deref().unwrap_or("x-api-key"), credential);
            }
            AuthScheme::Query | AuthScheme::None => {}
        }
        for (key, value) in headers {
            if is_reserved_header(key) {
                return Err(
                    ModelFailure::new(FailurePhase::Prepare, FailureKind::InvalidRequest)
                        .with_code("reserved_header"),
                );
            }
            let name = reqwest::header::HeaderName::from_bytes(key.as_bytes()).map_err(|_| {
                ModelFailure::new(FailurePhase::Prepare, FailureKind::InvalidRequest)
                    .with_code("invalid_header_name")
            })?;
            let value = reqwest::header::HeaderValue::from_str(value).map_err(|_| {
                ModelFailure::new(FailurePhase::Prepare, FailureKind::InvalidRequest)
                    .with_code("invalid_header_value")
            })?;
            request = request.header(name, value);
        }
        Ok(request)
    }
}

fn is_safe_query_key(value: &str) -> bool {
    !value.is_empty()
        && value.len() <= 128
        && value
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'_' | b'-' | b'.'))
}

fn is_reserved_header(value: &str) -> bool {
    matches!(
        value.to_ascii_lowercase().as_str(),
        "authorization"
            | "x-api-key"
            | "content-type"
            | "content-length"
            | "host"
            | "accept"
            | "anthropic-version"
            | "anthropic-beta"
    )
}

#[derive(Clone)]
pub struct ResolvedProvider {
    pub name: String,
    pub flavor: ProviderFlavor,
    pub default_model: String,
    pub retry: Option<RuntimeRetryConfig>,
    pub auth: RuntimeAuthConfig,
    pub endpoint: String,
    pub endpoint_query: BTreeMap<String, String>,
    pub headers: BTreeMap<String, String>,
    pub query: BTreeMap<String, String>,
    pub transport: Arc<ProviderTransport>,
    pub models: BTreeMap<String, ResolvedModelRoute>,
}

impl ResolvedProvider {
    fn new(
        name: &str,
        config: &RuntimeProviderConfig,
        registry: &ProtocolRegistry,
    ) -> Result<Self, RuntimeConfigError> {
        let transport = Arc::new(
            ProviderTransport::new_for_endpoint(
                &config.transport,
                Some(&config.endpoints.base_url),
            )
            .map_err(|error| RuntimeConfigError::InvalidValue {
                field: format!("providers.{name}.transport"),
                reason: error.to_string(),
            })?,
        );
        let mut models = BTreeMap::new();
        let auth = config.auth.snapshot(name)?;
        for (model_name, model) in &config.models {
            let protocol_name = model
                .protocol
                .as_deref()
                .or(config.protocol.as_deref())
                .unwrap_or("responses");
            let protocol = ProtocolId::new(protocol_name)
                .map_err(|_| RuntimeConfigError::UnknownProtocol(protocol_name.to_owned()))?;
            if registry.lookup(&protocol).is_none() {
                return Err(RuntimeConfigError::UnknownProtocol(protocol.to_string()));
            }
            let adapter = registry
                .lookup(&protocol)
                .ok_or_else(|| RuntimeConfigError::UnknownProtocol(protocol.to_string()))?;
            let (endpoint, endpoint_query) = config
                .endpoints
                .endpoint_for(&protocol, adapter.default_endpoint_path())?;
            models.insert(
                model_name.clone(),
                ResolvedModelRoute {
                    provider: name.to_owned(),
                    model: model_name.clone(),
                    display: model.display.clone(),
                    context_window: model.context_window,
                    effective_input_limit_tokens: model.effective_input_limit_tokens,
                    model_override: model
                        .model_override
                        .clone()
                        .unwrap_or_else(|| model_name.clone()),
                    flavor: model.flavor.unwrap_or(config.flavor),
                    protocol_id: protocol.clone(),
                    endpoint: endpoint.clone(),
                    auth: auth.clone(),
                    headers: config.headers.clone(),
                    query: merge_query(&config.query, &endpoint_query, name, model_name)?,
                    capabilities: model.capabilities.clone().into(),
                    generation: model.capabilities.generation.clone().into(),
                    generation_defaults: model.generation.clone(),
                    cache: model.cache.clone(),
                    protocol_settings: model.protocol_settings.clone(),
                    retry: config.retry.clone(),
                    transport: transport.clone(),
                    binding: registry
                        .lookup(&protocol)
                        .expect("protocol was checked above")
                        .bind(ProtocolBindInput::new(
                            BindingIdentity::new(
                                protocol.clone(),
                                ProfileIdentity::new(name).map_err(|error| {
                                    RuntimeConfigError::InvalidValue {
                                        field: format!("providers.{name}"),
                                        reason: error.to_string(),
                                    }
                                })?,
                                RouteIdentity::new(format!("{name}/{model_name}")).map_err(
                                    |error| RuntimeConfigError::InvalidValue {
                                        field: format!("providers.{name}.models.{model_name}"),
                                        reason: error.to_string(),
                                    },
                                )?,
                            ),
                            endpoint.clone(),
                            BindingFlavor::new(model.flavor.unwrap_or(config.flavor).as_str())
                                .expect("valid flavor"),
                            ProfileSettings {
                                display_name: None,
                                model: Some(
                                    model
                                        .model_override
                                        .clone()
                                        .unwrap_or_else(|| model_name.clone()),
                                ),
                            },
                            model.protocol_settings.clone(),
                            model.capabilities.clone().into(),
                            model.capabilities.generation.clone().into(),
                        ))
                        .map_err(|error| RuntimeConfigError::InvalidValue {
                            field: format!("providers.{name}.models.{model_name}.binding"),
                            reason: error.to_string(),
                        })?,
                },
            );
        }
        Ok(Self {
            name: name.to_owned(),
            flavor: config.flavor,
            default_model: config.default_model.clone().unwrap_or_default(),
            retry: config.retry.clone(),
            auth: auth.clone(),
            endpoint: config.endpoints.base_url.clone(),
            endpoint_query: BTreeMap::new(),
            headers: config.headers.clone(),
            query: config.query.clone(),
            transport,
            models,
        })
    }
}

#[derive(Clone)]
pub struct ResolvedModelRoute {
    pub provider: String,
    pub model: String,
    pub display: Option<String>,
    pub context_window: Option<u64>,
    pub effective_input_limit_tokens: Option<u64>,
    pub model_override: String,
    pub flavor: ProviderFlavor,
    pub protocol_id: ProtocolId,
    pub endpoint: String,
    pub auth: RuntimeAuthConfig,
    pub headers: BTreeMap<String, String>,
    pub query: BTreeMap<String, String>,
    pub capabilities: RouteCapabilities,
    pub generation: GenerationSupport,
    pub generation_defaults: RuntimeGenerationDefaults,
    pub cache: RuntimeCacheConfig,
    pub protocol_settings: ProtocolSettings,
    pub retry: Option<RuntimeRetryConfig>,
    pub transport: Arc<ProviderTransport>,
    pub binding: Arc<dyn ProtocolBinding>,
}

impl ResolvedModelRoute {
    pub fn binding(&self) -> &Arc<dyn ProtocolBinding> {
        &self.binding
    }

    pub fn bind_input(&self) -> Result<ProtocolBindInput, RuntimeConfigError> {
        Ok(ProtocolBindInput::new(
            BindingIdentity::new(
                self.protocol_id.clone(),
                ProfileIdentity::new(self.provider.clone()).map_err(|error| {
                    RuntimeConfigError::InvalidValue {
                        field: "provider".into(),
                        reason: error.to_string(),
                    }
                })?,
                RouteIdentity::new(format!("{}/{}", self.provider, self.model)).map_err(
                    |error| RuntimeConfigError::InvalidValue {
                        field: "route".into(),
                        reason: error.to_string(),
                    },
                )?,
            ),
            self.endpoint.clone(),
            BindingFlavor::new(self.flavor.as_str()).expect("built-in flavor is valid"),
            ProfileSettings {
                display_name: None,
                model: Some(self.model_override.clone()),
            },
            self.protocol_settings.clone(),
            self.capabilities.clone(),
            self.generation.clone(),
        ))
    }
}

#[derive(Clone)]
pub struct ResolvedRuntimeCatalog {
    pub active_provider: String,
    pub providers: BTreeMap<String, ResolvedProvider>,
    pub fingerprint: RuntimeFingerprint,
}

impl fmt::Debug for ResolvedRuntimeCatalog {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ResolvedRuntimeCatalog")
            .field("active_provider", &self.active_provider)
            .field("providers", &self.providers.keys().collect::<Vec<_>>())
            .field("fingerprint", &self.fingerprint)
            .finish()
    }
}

impl ResolvedRuntimeCatalog {
    fn new(
        active_provider: String,
        providers: BTreeMap<String, ResolvedProvider>,
    ) -> Result<Self, RuntimeConfigError> {
        if providers.is_empty() {
            return Err(RuntimeConfigError::Missing("providers".into()));
        }
        if !providers.contains_key(&active_provider) {
            return Err(RuntimeConfigError::InvalidValue {
                field: "active_provider".into(),
                reason: "provider is not resolved".into(),
            });
        }
        let fingerprint = RuntimeFingerprint::from_resolved(&active_provider, &providers);
        Ok(Self {
            active_provider,
            providers,
            fingerprint,
        })
    }

    pub fn route(&self, provider: &str, model: &str) -> Option<&ResolvedModelRoute> {
        self.providers.get(provider)?.models.get(model)
    }

    pub fn fingerprint(&self) -> &RuntimeFingerprint {
        &self.fingerprint
    }
}

#[derive(Clone, PartialEq)]
pub struct RuntimeFingerprint {
    pub active_provider: String,
    pub providers: BTreeMap<String, ProviderFingerprint>,
}

impl fmt::Debug for RuntimeFingerprint {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("RuntimeFingerprint")
            .field("active_provider", &self.active_provider)
            .field("providers", &self.providers)
            .finish()
    }
}

#[derive(Clone, PartialEq)]
pub struct ProviderFingerprint {
    pub flavor: ProviderFlavor,
    pub default_model: String,
    pub auth_mode: AuthScheme,
    pub auth_name: Option<String>,
    pub credential_fingerprint: String,
    pub retry: Option<RuntimeRetryConfig>,
    pub endpoint: String,
    pub headers: BTreeMap<String, String>,
    pub query: BTreeMap<String, String>,
    pub connect_timeout_secs: u64,
    pub no_proxy_loopback: bool,
    pub websocket: bool,
    pub models: BTreeMap<String, ModelFingerprint>,
}

impl fmt::Debug for ProviderFingerprint {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ProviderFingerprint")
            .field("flavor", &self.flavor)
            .field("default_model", &self.default_model)
            .field("auth_mode", &self.auth_mode)
            .field("auth_name", &self.auth_name)
            .field("credential_fingerprint", &self.credential_fingerprint)
            .field("retry", &self.retry)
            .field("endpoint", &self.endpoint)
            .field("headers", &"<hashed>")
            .field("query", &"<hashed>")
            .field("connect_timeout_secs", &self.connect_timeout_secs)
            .field("no_proxy_loopback", &self.no_proxy_loopback)
            .field("websocket", &self.websocket)
            .field("models", &self.models)
            .finish()
    }
}

#[derive(Clone, PartialEq)]
pub struct ModelFingerprint {
    pub display: Option<String>,
    pub context_window: Option<u64>,
    pub effective_input_limit_tokens: Option<u64>,
    pub model_override: String,
    pub flavor: ProviderFlavor,
    pub protocol_id: ProtocolId,
    pub endpoint: String,
    pub headers: BTreeMap<String, String>,
    pub query: BTreeMap<String, String>,
    pub capabilities: RouteCapabilities,
    pub generation: GenerationSupport,
    pub generation_defaults: RuntimeGenerationDefaults,
    pub cache: RuntimeCacheConfig,
    pub protocol_settings: ProtocolSettings,
}

impl fmt::Debug for ModelFingerprint {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ModelFingerprint")
            .field("display", &self.display)
            .field("context_window", &self.context_window)
            .field(
                "effective_input_limit_tokens",
                &self.effective_input_limit_tokens,
            )
            .field("model_override", &self.model_override)
            .field("flavor", &self.flavor)
            .field("protocol_id", &self.protocol_id)
            .field("endpoint", &self.endpoint)
            .field("headers", &"<hashed>")
            .field("query", &"<hashed>")
            .field("capabilities", &self.capabilities)
            .field("generation", &self.generation)
            .field("generation_defaults", &self.generation_defaults)
            .field("cache", &self.cache)
            .field("protocol_settings", &"<redacted>")
            .finish()
    }
}

impl RuntimeFingerprint {
    fn from_resolved(
        active_provider: &str,
        providers: &BTreeMap<String, ResolvedProvider>,
    ) -> Self {
        Self {
            active_provider: active_provider.to_owned(),
            providers: providers
                .iter()
                .map(|(name, provider)| {
                    (
                        name.clone(),
                        ProviderFingerprint {
                            flavor: provider.flavor,
                            default_model: provider.default_model.clone(),
                            auth_mode: provider.auth.scheme,
                            auth_name: provider.auth.name.clone(),
                            credential_fingerprint: crate::request_builder::sha256_hex(
                                provider
                                    .auth
                                    .credential
                                    .as_deref()
                                    .unwrap_or_default()
                                    .as_bytes(),
                            ),
                            retry: provider.retry.clone(),
                            endpoint: provider.endpoint.clone(),
                            headers: provider.headers.clone(),
                            query: provider.query.clone(),
                            connect_timeout_secs: provider.transport.connect_timeout.as_secs(),
                            no_proxy_loopback: provider.transport.no_proxy_loopback,
                            websocket: provider.transport.websocket,
                            models: provider
                                .models
                                .iter()
                                .map(|(model, route)| {
                                    (
                                        model.clone(),
                                        ModelFingerprint {
                                            display: route.display.clone(),
                                            context_window: route.context_window,
                                            effective_input_limit_tokens: route
                                                .effective_input_limit_tokens,
                                            model_override: route.model_override.clone(),
                                            flavor: route.flavor,
                                            protocol_id: route.protocol_id.clone(),
                                            endpoint: route.endpoint.clone(),
                                            headers: route.headers.clone(),
                                            query: route.query.clone(),
                                            capabilities: route.capabilities.clone(),
                                            generation: route.generation.clone(),
                                            generation_defaults: route.generation_defaults.clone(),
                                            cache: route.cache.clone(),
                                            protocol_settings: route.protocol_settings.clone(),
                                        },
                                    )
                                })
                                .collect(),
                        },
                    )
                })
                .collect(),
        }
    }
}

/// Provider-neutral Retry-After conversion for model runtime failures.
pub fn retry_hint_from_headers(headers: &reqwest::header::HeaderMap) -> RetryHint {
    let Some(value) = headers
        .get(reqwest::header::RETRY_AFTER)
        .and_then(|value| value.to_str().ok())
    else {
        return RetryHint::Retryable;
    };
    if let Ok(seconds) = value.trim().parse::<u64>() {
        return RetryHint::RetryAfterSeconds(seconds);
    }
    if let Ok(at) = httpdate::parse_http_date(value.trim()) {
        if let Ok(delay) = at.duration_since(std::time::SystemTime::now()) {
            return RetryHint::RetryAfterSeconds(delay.as_secs());
        }
    }
    RetryHint::Retryable
}

#[cfg(test)]
mod tests {
    use super::*;

    fn binding(protocol: &str, profile: &str, route: &str) -> BindingIdentity {
        BindingIdentity::new(
            ProtocolId::new(protocol).expect("protocol"),
            ProfileIdentity::new(profile).expect("profile"),
            RouteIdentity::new(route).expect("route"),
        )
    }

    fn bind_input(identity: BindingIdentity) -> ProtocolBindInput {
        ProtocolBindInput::new(
            identity,
            "https://example.invalid/v1/responses",
            BindingFlavor::new("chat").expect("flavor"),
            ProfileSettings::default(),
            ProtocolSettings::default(),
            RouteCapabilities::default(),
            GenerationSupport::default(),
        )
    }

    #[test]
    fn default_route_capabilities_are_all_off() {
        let capabilities = RouteCapabilities::default();
        assert!(!capabilities.tools);
        assert!(!capabilities.parallel_tool_calls);
        assert!(!capabilities.reasoning);
        assert!(!capabilities.input_images);
        assert!(!capabilities.tool_result_images);
        assert!(!capabilities.prompt_cache);
        assert!(!capabilities.priority_service);
    }

    #[test]
    fn protocol_settings_preserve_adapter_owned_toml() {
        let value = toml::from_str::<toml::Value>(
            r#"vendor = { mode = "strict", retries = 2 }
flags = ["a", "b"]"#,
        )
        .expect("valid TOML value");
        let settings = ProtocolSettings::new(value.clone());
        assert_eq!(settings.value, value);
        assert!(matches!(settings.value, toml::Value::Table(_)));

        #[derive(Debug, Deserialize, PartialEq)]
        #[serde(deny_unknown_fields)]
        struct VendorSettings {
            mode: String,
            retries: u32,
        }
        let vendor: VendorSettings = settings
            .value
            .get("vendor")
            .cloned()
            .expect("vendor settings")
            .try_into()
            .expect("strict adapter settings");
        assert_eq!(vendor.mode, "strict");
        assert_eq!(vendor.retries, 2);
    }

    #[test]
    fn generation_supports_custom_reasoning_and_route_options() {
        let generation = GenerationSettings {
            reasoning: ReasoningIntent {
                enabled: true,
                effort: Some(ReasoningEffort::Custom("vendor-effort".into())),
            },
            summary: Some(ReasoningSummary::Detailed),
            verbosity: Some(Verbosity::High),
            parallel_tool_calls: Some(true),
            priority_service: Some(true),
            ..GenerationSettings::default()
        };
        assert_eq!(
            generation.reasoning.effort,
            Some(ReasoningEffort::Custom("vendor-effort".into()))
        );
        assert_eq!(generation.summary, Some(ReasoningSummary::Detailed));
        assert_eq!(generation.verbosity, Some(Verbosity::High));
        assert_eq!(generation.parallel_tool_calls, Some(true));
        assert_eq!(generation.priority_service, Some(true));
    }

    #[test]
    fn cache_intent_preserves_namespace_and_optional_retention() {
        let without_retention = CacheIntent {
            enabled: true,
            namespace: Some("cache-a".into()),
            retention: None,
            stable_prefix: None,
        };
        assert!(without_retention.enabled);
        assert_eq!(without_retention.namespace.as_deref(), Some("cache-a"));
        assert_eq!(without_retention.retention, None);

        let with_retention = CacheIntent {
            enabled: true,
            namespace: Some("cache-b".into()),
            retention: Some(CacheRetention::InMemory),
            stable_prefix: Some(StablePrefixMetadata {
                segment_count: 2,
                fingerprint: Some("prefix".into()),
            }),
        };
        assert_eq!(with_retention.retention, Some(CacheRetention::InMemory));
        assert_eq!(
            with_retention.stable_prefix.as_ref().unwrap().segment_count,
            2
        );
    }

    #[test]
    fn protocol_id_is_extensible_and_validated() {
        let custom = ProtocolId::new("vendor.protocol-v2").expect("custom id");
        assert_eq!(custom.as_str(), "vendor.protocol-v2");
        assert!(ProtocolId::new("").is_err());
        assert!(ProtocolId::new(" responses").is_err());
    }

    #[test]
    fn builtins_register_responses_and_validation_only_adapters() {
        let registry = ProtocolRegistry::builtins();
        assert_eq!(registry.len(), 3);
        let responses = registry.lookup_str("responses").expect("built-in adapter");
        let response_binding = responses
            .bind(ProtocolBindInput::new(
                binding("responses", "profile", "route"),
                "https://example.invalid/v1/responses",
                BindingFlavor::new("standard").expect("standard flavor"),
                ProfileSettings::default(),
                ProtocolSettings::default(),
                RouteCapabilities::default(),
                GenerationSupport::default(),
            ))
            .expect("Responses adapter binds supported flavor");
        assert_eq!(response_binding.protocol_id().as_str(), "responses");
        let model = ModelRequestInput::new("model", Vec::new());
        assert!(response_binding.prepare_request(&model).is_ok());

        let completions = registry
            .lookup_str("completions")
            .expect("built-in adapter");
        let completion_binding = completions
            .bind(ProtocolBindInput::new(
                binding("completions", "profile", "route"),
                "https://example.invalid/v1/chat/completions",
                BindingFlavor::new("standard").expect("standard flavor"),
                ProfileSettings::default(),
                ProtocolSettings::default(),
                RouteCapabilities::default(),
                GenerationSupport::default(),
            ))
            .expect("Completions adapter binds supported flavor");
        assert_eq!(completion_binding.protocol_id().as_str(), "completions");
        assert!(completion_binding.prepare_request(&model).is_ok());

        let anthropic = registry.lookup_str("anthropic").expect("built-in adapter");
        let anthropic_binding = anthropic
            .bind(ProtocolBindInput::new(
                BindingIdentity::new(
                    ProtocolId::new("anthropic").unwrap(),
                    ProfileIdentity::new("profile").unwrap(),
                    RouteIdentity::new("route").unwrap(),
                ),
                "https://example.invalid/v1/messages",
                BindingFlavor::new("standard").unwrap(),
                ProfileSettings::default(),
                ProtocolSettings::default(),
                RouteCapabilities::default(),
                GenerationSupport::default(),
            ))
            .expect("Anthropic adapter binds supported flavor");
        assert_eq!(anthropic_binding.protocol_id().as_str(), "anthropic");
        assert!(anthropic_binding.prepare_request(&model).is_ok());
    }

    #[test]
    fn registry_rejects_duplicate_protocol_ids_and_returns_adapter() {
        let id = ProtocolId::new("duplicate").expect("id");
        let adapters = [
            Arc::new(adapters::PlaceholderAdapter::new(id.clone())) as Arc<dyn ProtocolAdapter>,
            Arc::new(adapters::PlaceholderAdapter::new(id)) as Arc<dyn ProtocolAdapter>,
        ];
        assert!(matches!(
            ProtocolRegistry::new(adapters),
            Err(RegistryError::DuplicateProtocol(_))
        ));

        let registry = ProtocolRegistry::builtins();
        assert_eq!(
            registry
                .lookup_str("responses")
                .expect("registered adapter")
                .protocol_id()
                .as_str(),
            "responses"
        );
    }

    #[test]
    fn request_is_typed_and_has_no_conversation_id() {
        let request = ModelRequestInput::new(
            "model",
            vec![
                ModelMessage::user_image("image/png", [1, 2, 3]),
                ModelMessage {
                    role: MessageRole::Assistant,
                    content: vec![
                        ContentPart::Text("answer".into()),
                        ContentPart::Reasoning {
                            item_id: "r1".into(),
                            text: "thought".into(),
                            replay: None,
                        },
                        ContentPart::ToolCall {
                            id: "t1".into(),
                            name: "search".into(),
                            arguments: Value::Object(serde_json::Map::new()),
                        },
                    ],
                },
                ModelMessage {
                    role: MessageRole::Tool,
                    content: vec![ContentPart::ToolResult {
                        id: "t1".into(),
                        content: vec![ContentPart::Image {
                            media_type: "image/png".into(),
                            data: vec![4, 5],
                        }],
                    }],
                },
            ],
        );
        assert_eq!(request.messages.len(), 3);
        assert!(request.control.model == "model");
        assert!(request.cache.stable_prefix.is_none());
    }

    #[test]
    fn cache_retention_has_no_caller_supplied_key() {
        let cache = CacheIntent {
            enabled: true,
            namespace: Some("test-cache".into()),
            retention: Some(CacheRetention::TwentyFourHours),
            stable_prefix: Some(StablePrefixMetadata {
                segment_count: 0,
                fingerprint: Some("stable".into()),
            }),
        };
        assert_eq!(cache.retention, Some(CacheRetention::TwentyFourHours));
        assert!(cache.stable_prefix.is_some());
    }

    #[test]
    fn object_safe_decoder_is_incremental() {
        struct DecoderFixture;
        impl ModelStreamDecoder for DecoderFixture {
            fn push(&mut self, _chunk: &[u8]) -> Result<Vec<ModelEvent>, ModelFailure> {
                Ok(vec![ModelEvent::TextDelta {
                    text: "delta".into(),
                }])
            }

            fn finish(&mut self) -> Result<Vec<ModelEvent>, ModelFailure> {
                Ok(vec![ModelEvent::Terminal {
                    status: TerminalStatus::Completed,
                }])
            }
        }

        let mut decoder: Box<dyn ModelStreamDecoder> = Box::new(DecoderFixture);
        assert!(matches!(decoder.push(b"chunk"), Ok(events) if matches!(
            events.first(),
            Some(ModelEvent::TextDelta { text }) if text == "delta"
        )));
        assert!(matches!(decoder.finish(), Ok(events) if matches!(
            events.first(),
            Some(ModelEvent::Terminal { status: TerminalStatus::Completed })
        )));
    }

    #[test]
    fn replay_scope_uses_producer_identity_and_target_binding() {
        let target = binding("responses", "profile", "route");
        let same_profile_other_route = binding("responses", "profile", "other-route");
        let other_profile = binding("responses", "other-profile", "route");
        let protocol_producer = ReplayProducer::new(ReplayScope::Protocol, &target);
        let profile_producer = ReplayProducer::new(ReplayScope::Profile, &target);
        let route_producer = ReplayProducer::new(ReplayScope::Route, &target);

        assert!(ReplayScope::Protocol.is_compatible_with(&protocol_producer, &other_profile));
        assert!(
            ReplayScope::Profile.is_compatible_with(&profile_producer, &same_profile_other_route)
        );
        assert!(!ReplayScope::Profile.is_compatible_with(&profile_producer, &other_profile));
        assert!(ReplayScope::Route.is_compatible_with(&route_producer, &target));
        assert!(!ReplayScope::Route.is_compatible_with(&route_producer, &same_profile_other_route));
        assert!(!ReplayScope::None.is_compatible_with(&route_producer, &target));
    }

    #[test]
    fn opaque_replay_state_uses_typed_payload_and_producer() {
        let identity = binding("responses", "profile", "route");
        let state = OpaqueReplayState::new(
            "model-runtime",
            1,
            ReplayProducer::new(ReplayScope::Route, &identity),
            Value::Object(serde_json::Map::from_iter([
                ("cursor".into(), Value::from(3)),
                ("opaque".into(), Value::from(true)),
            ])),
        );
        assert_eq!(state.namespace, "model-runtime");
        assert_eq!(state.version, 1);
        assert_eq!(state.producer.scope, ReplayScope::Route);
        assert_eq!(state.payload["cursor"], Value::from(3));
    }

    #[test]
    fn model_events_cover_complete_lifecycle_and_terminal_statuses() {
        let events = [
            ModelEvent::ReasoningStarted {
                item_id: "r1".into(),
            },
            ModelEvent::ReasoningDelta {
                item_id: "r1".into(),
                text: "think".into(),
            },
            ModelEvent::ReasoningSummaryDelta {
                item_id: "r1".into(),
                summary_index: 0,
                text: "summary".into(),
            },
            ModelEvent::ReasoningSummaryDone {
                item_id: "r1".into(),
                summary_index: 0,
                text: "summary".into(),
            },
            ModelEvent::ReasoningDone {
                item_id: "r1".into(),
                text: "thought".into(),
                replay: None,
            },
            ModelEvent::ToolStarted {
                id: "t1".into(),
                name: "search".into(),
            },
            ModelEvent::ToolArgumentsDelta {
                id: "t1".into(),
                delta: "{\"q\":".into(),
            },
            ModelEvent::ToolDone {
                id: "t1".into(),
                name: "search".into(),
                arguments: Value::Object(serde_json::Map::new()),
            },
            ModelEvent::Usage {
                input_tokens: 1,
                output_tokens: 2,
                total_tokens: 3,
                reasoning_tokens: Some(1),
                cached_input_tokens: Some(4),
            },
            ModelEvent::Cache {
                hit: true,
                read_tokens: 4,
                write_tokens: 0,
            },
            ModelEvent::Terminal {
                status: TerminalStatus::Completed,
            },
            ModelEvent::Failure(ModelFailure::new(
                FailurePhase::Decode,
                FailureKind::MalformedResponse,
            )),
        ];
        assert_eq!(events.len(), 12);
        let statuses = [
            TerminalStatus::Completed,
            TerminalStatus::ToolUse,
            TerminalStatus::Length,
            TerminalStatus::ContentFilter,
            TerminalStatus::Refusal,
            TerminalStatus::Pause,
            TerminalStatus::Incomplete,
        ];
        assert_eq!(statuses.len(), 7);
    }

    #[test]
    fn model_failure_display_omits_bounded_redacted_detail() {
        let failure = ModelFailure::new(FailurePhase::Transport, FailureKind::Http)
            .with_status(500)
            .with_code("server_error")
            .with_detail("Authorization: bearer-secret response body");
        assert!(!failure.detail().contains("bearer-secret"));
        assert!(!failure.to_string().contains("bearer-secret"));
        assert!(failure.to_string().contains("status 500"));
    }

    #[test]
    fn model_failure_detail_preserves_unknown_provider_json_and_redacts_credentials() {
        let failure = ModelFailure::new(FailurePhase::Transport, FailureKind::Http)
            .with_detail_redacted(
                r#"{"error":{"unfamiliar":{"explanation":"unsupported input","request_fragment":{"model":"demo","prompt":"keep this diagnostic"}}},"apiKey":"response-secret","nested":[{"accessToken":"echoed-token"}],"clientSecret":"client-secret","proxyAuthorization":"proxy-secret","credential":{"value":"nested-secret"}}"#,
                &["response-secret", "echoed-token"],
            );
        assert!(failure.detail().contains("unsupported input"));
        assert!(failure.detail().contains("keep this diagnostic"));
        assert!(failure.detail().contains("\"unfamiliar\""));
        assert!(failure.detail().contains("\"apiKey\":\"[redacted]\""));
        assert!(failure.detail().contains("\"accessToken\":\"[redacted]\""));
        assert!(failure.detail().contains("\"clientSecret\":\"[redacted]\""));
        assert!(
            failure
                .detail()
                .contains("\"proxyAuthorization\":\"[redacted]\"")
        );
        assert!(failure.detail().contains("\"credential\":\"[redacted]\""));
        assert!(!failure.detail().contains("response-secret"));
        assert!(!failure.detail().contains("echoed-token"));
        assert!(!failure.detail().contains("client-secret"));
        assert!(!failure.detail().contains("proxy-secret"));
        assert!(!failure.detail().contains("nested-secret"));
    }

    #[test]
    fn model_failure_detail_redacts_whitespace_credentials_without_hiding_diagnostics() {
        let failure = ModelFailure::new(FailurePhase::Transport, FailureKind::Http).with_detail(
            "Authorization: Bearer colon-secret colon request rejected\nAuthorization Bearer abc123 request rejected\napi-key: colon-key colon endpoint unavailable\napi key key-value endpoint unavailable\naccess token token-value scope denied\ntoken count 123\nsecret explanation remains useful",
        );
        assert!(!failure.detail().contains("colon-secret"));
        assert!(!failure.detail().contains("abc123"));
        assert!(!failure.detail().contains("colon-key"));
        assert!(!failure.detail().contains("key-value"));
        assert!(!failure.detail().contains("token-value"));
        assert!(failure.detail().contains("colon request rejected"));
        assert!(failure.detail().contains("request rejected"));
        assert!(failure.detail().contains("colon endpoint unavailable"));
        assert!(failure.detail().contains("endpoint unavailable"));
        assert!(failure.detail().contains("scope denied"));
        assert!(failure.detail().contains("token count 123"));
        assert!(
            failure
                .detail()
                .contains("secret explanation remains useful")
        );
    }

    #[test]
    fn model_failure_detail_preserves_large_provider_diagnostics() {
        let diagnostic = "x".repeat(300 * 1024);
        let failure = ModelFailure::new(FailurePhase::Transport, FailureKind::Http)
            .with_detail(diagnostic.clone());
        assert_eq!(failure.detail(), diagnostic);
    }

    #[test]
    fn model_failure_detail_preserves_multiline_text_and_html() {
        let failure = ModelFailure::new(FailurePhase::Transport, FailureKind::Http).with_detail(
            "<html>\n<body>invalid field: input[2]</body>\n</html>\napi-key=provider-secret",
        );
        assert!(
            failure
                .detail()
                .contains("<body>invalid field: input[2]</body>")
        );
        assert!(failure.detail().contains('\n'));
        assert!(!failure.detail().contains("provider-secret"));
    }

    fn runtime_config(
        provider: &str,
        auth: &str,
        provider_extra: &str,
        model_extra: &str,
    ) -> String {
        let model_extra = model_extra
            .replace(
                "[capabilities]",
                &format!("[providers.{provider}.models.model.capabilities]"),
            )
            .replace(
                "[generation]",
                &format!("[providers.{provider}.models.model.generation]"),
            )
            .replace(
                "[protocol_settings]",
                &format!("[providers.{provider}.models.model.protocol_settings]"),
            );
        format!(
            r#"active_provider = "{provider}"
[providers.{provider}]
protocol = "responses"
default_model = "model"
{provider_extra}
[providers.{provider}.auth]
{auth}
[providers.{provider}.endpoints]
base_url = "https://example.invalid/v1"
[providers.{provider}.models.model]
{model_extra}
"#
        )
    }

    #[test]
    fn websocket_transport_defaults_off_and_is_resolved_into_fingerprint() {
        let default =
            RuntimeConfig::from_toml(&runtime_config("default-ws", "type = \"none\"", "", ""))
                .unwrap()
                .resolve(&ProtocolRegistry::builtins())
                .unwrap();
        assert!(
            !default
                .route("default-ws", "model")
                .unwrap()
                .transport
                .websocket()
        );
        assert!(!default.fingerprint().providers["default-ws"].websocket);

        let enabled = RuntimeConfig::from_toml(&runtime_config(
            "enabled-ws",
            "type = \"none\"",
            "[providers.enabled-ws.transport]\nwebsocket = true",
            "",
        ))
        .unwrap()
        .resolve(&ProtocolRegistry::builtins())
        .unwrap();
        assert!(
            enabled
                .route("enabled-ws", "model")
                .unwrap()
                .transport
                .websocket()
        );
        assert!(enabled.fingerprint().providers["enabled-ws"].websocket);
        assert_ne!(default.fingerprint(), enabled.fingerprint());
    }

    #[test]
    fn websocket_transport_allows_provider_mixed_protocols() {
        let config = runtime_config(
            "mixed-protocols",
            "type = \"none\"",
            "[providers.mixed-protocols.transport]\nwebsocket = true",
            "protocol = \"completions\"",
        );
        let resolved = RuntimeConfig::from_toml(&config)
            .unwrap()
            .resolve(&ProtocolRegistry::builtins())
            .unwrap();
        assert!(
            resolved
                .route("mixed-protocols", "model")
                .unwrap()
                .transport
                .websocket()
        );
    }

    #[test]
    fn bearer_auth_is_applied_by_the_real_request_builder() {
        let transport = ProviderTransport::new(&RuntimeTransportConfig::default()).unwrap();
        let auth = RuntimeAuthConfig {
            scheme: AuthScheme::Bearer,
            name: None,
            credential: Some("bearer-secret".into()),
            credential_env: None,
        };
        let headers = BTreeMap::from([("x-provider-version".into(), "2026-09-01".into())]);
        let query = BTreeMap::from([("region".into(), "global".into())]);
        let request = transport
            .request(
                reqwest::Method::POST,
                "https://example.invalid/v1/responses",
                "vendor",
                &auth,
                &headers,
                &query,
            )
            .unwrap()
            .build()
            .unwrap();
        assert_eq!(request.headers()["authorization"], "Bearer bearer-secret");
        assert_eq!(request.headers()["x-provider-version"], "2026-09-01");
        assert_eq!(
            request
                .url()
                .query_pairs()
                .find(|(key, _)| key == "region")
                .unwrap()
                .1,
            "global"
        );
    }

    #[test]
    fn named_header_auth_is_applied_by_the_real_request_builder() {
        let transport = ProviderTransport::new(&RuntimeTransportConfig::default()).unwrap();
        let auth = RuntimeAuthConfig {
            scheme: AuthScheme::Header,
            name: Some("x-vendor-key".into()),
            credential: Some("header-secret".into()),
            credential_env: None,
        };
        let headers = BTreeMap::from([("x-provider-version".into(), "2026-09-01".into())]);
        let query = BTreeMap::from([("region".into(), "global".into())]);
        let request = transport
            .request(
                reqwest::Method::POST,
                "https://example.invalid/v1/responses",
                "vendor",
                &auth,
                &headers,
                &query,
            )
            .unwrap()
            .build()
            .unwrap();
        assert_eq!(request.headers()["x-vendor-key"], "header-secret");
        assert_eq!(request.headers()["x-provider-version"], "2026-09-01");
        assert!(request.headers().get("authorization").is_none());
        assert_eq!(
            request
                .url()
                .query_pairs()
                .find(|(key, _)| key == "region")
                .unwrap()
                .1,
            "global"
        );
    }

    #[test]
    fn named_query_auth_is_applied_by_the_real_request_builder() {
        let transport = ProviderTransport::new(&RuntimeTransportConfig::default()).unwrap();
        let auth = RuntimeAuthConfig {
            scheme: AuthScheme::Query,
            name: Some("api-token".into()),
            credential: Some("query-secret".into()),
            credential_env: None,
        };
        let headers = BTreeMap::from([("x-provider-version".into(), "2026-09-01".into())]);
        let query = BTreeMap::from([("region".into(), "global".into())]);
        let request = transport
            .request(
                reqwest::Method::GET,
                "https://example.invalid/v1/responses",
                "vendor",
                &auth,
                &headers,
                &query,
            )
            .unwrap()
            .build()
            .unwrap();
        assert_eq!(request.headers()["x-provider-version"], "2026-09-01");
        assert_eq!(
            request
                .url()
                .query_pairs()
                .find(|(key, _)| key == "api-token")
                .unwrap()
                .1,
            "query-secret"
        );
        assert_eq!(
            request
                .url()
                .query_pairs()
                .find(|(key, _)| key == "region")
                .unwrap()
                .1,
            "global"
        );
    }

    #[test]
    fn provider_and_endpoint_query_conflicts_are_rejected() {
        let value = runtime_config(
            "vendor",
            "type = \"bearer\"\ncredential = \"secret\"",
            "query = { version = \"provider\" }",
            "",
        )
        .replace(
            "base_url = \"https://example.invalid/v1\"",
            "base_url = \"https://example.invalid/v1\"\n[providers.vendor.endpoints.responses]\nquery = { version = \"endpoint\" }",
        );
        let error = RuntimeConfig::from_toml(&value).unwrap_err();
        assert!(error.to_string().contains("conflicts with provider query"));
    }

    #[test]
    fn exact_generation_defaults_change_the_runtime_fingerprint() {
        let base = runtime_config(
            "vendor",
            "type = \"bearer\"\ncredential = \"secret\"",
            "",
            "[capabilities]\ngeneration = { max_output_tokens = true }\n[generation]\nmax_output_tokens = 128",
        );
        let changed = base.replace("max_output_tokens = 128", "max_output_tokens = 256");
        let first = RuntimeConfig::from_toml(&base)
            .unwrap()
            .resolve(&ProtocolRegistry::builtins())
            .unwrap();
        let second = RuntimeConfig::from_toml(&changed)
            .unwrap()
            .resolve(&ProtocolRegistry::builtins())
            .unwrap();
        assert_ne!(first.fingerprint(), second.fingerprint());
    }

    #[test]
    fn retry_validation_rejects_zero_and_out_of_range_attempts() {
        for retry in [
            "max_attempts = 0\nmax_recovery_attempts = 1\ninitial_delay_secs = 1\nbackoff_multiplier = 2.0",
            "max_attempts = 9001\nmax_recovery_attempts = 1\ninitial_delay_secs = 1\nbackoff_multiplier = 2.0",
        ] {
            let value = runtime_config(
                "vendor",
                "type = \"bearer\"\ncredential = \"secret\"",
                "",
                "",
            )
            .replace(
                "default_model = \"model\"",
                &format!("default_model = \"model\"\n[providers.vendor.retry]\n{retry}"),
            );
            assert!(RuntimeConfig::from_toml(&value).is_err(), "{retry}");
        }
    }

    #[test]
    fn malformed_and_unknown_protocol_settings_are_rejected() {
        let malformed = runtime_config(
            "vendor",
            "type = \"bearer\"\ncredential = \"secret\"",
            "",
            "protocol_settings = \"not-a-table\"",
        );
        assert!(RuntimeConfig::from_toml(&malformed).is_err());

        let unknown = runtime_config(
            "vendor",
            "type = \"bearer\"\ncredential = \"secret\"",
            "",
            "[protocol_settings]\nunknown = true",
        );
        let error = RuntimeConfig::from_toml(&unknown).unwrap_err();
        assert!(error.to_string().contains("unknown protocol setting"));
    }

    #[test]
    fn credential_debug_output_is_redacted_but_request_value_is_preserved() {
        let auth = RuntimeAuthConfig {
            scheme: AuthScheme::Bearer,
            name: None,
            credential: Some("credential-secret".into()),
            credential_env: None,
        };
        assert!(!format!("{auth:?}").contains("credential-secret"));
        assert_eq!(auth.credential_value("vendor"), "credential-secret");
    }

    #[test]
    fn credential_snapshot_is_immutable_after_environment_mutation() {
        let key = "LETcode_GROUP03A_SNAPSHOT";
        unsafe { std::env::set_var(key, "first-secret") };
        let value = runtime_config(
            "vendor",
            &format!("type = \"bearer\"\ncredential_env = \"{key}\""),
            "",
            "",
        );
        let config = RuntimeConfig::from_toml(&value).unwrap();
        let first = config.resolve(&ProtocolRegistry::builtins()).unwrap();
        unsafe { std::env::set_var(key, "second-secret") };
        let second = config.resolve(&ProtocolRegistry::builtins()).unwrap();
        assert_eq!(
            first
                .route("vendor", "model")
                .unwrap()
                .auth
                .credential
                .as_deref(),
            Some("first-secret")
        );
        assert_ne!(first.fingerprint(), second.fingerprint());
        unsafe { std::env::remove_var(key) };
    }

    #[test]
    fn default_provider_environment_credential_is_snapshotted() {
        let key = "GROUP03A_DEFAULT_ENV_ONLY";
        unsafe { std::env::set_var(key, "default-first") };
        let value = runtime_config(
            "default-env-vendor",
            &format!("type = \"bearer\"\ncredential_env = \"{key}\""),
            "",
            "",
        );
        let config = RuntimeConfig::from_toml(&value).unwrap();
        let first = config.resolve(&ProtocolRegistry::builtins()).unwrap();
        unsafe { std::env::set_var(key, "default-second") };
        assert_eq!(
            first
                .route("default-env-vendor", "model")
                .unwrap()
                .auth
                .credential
                .as_deref(),
            Some("default-first")
        );
        unsafe { std::env::remove_var(key) };
    }

    #[tokio::test]
    async fn transport_errors_redact_request_metadata() {
        let transport = ProviderTransport::new_for_endpoint(
            &RuntimeTransportConfig {
                connect_timeout_secs: 1,
                ..RuntimeTransportConfig::default()
            },
            Some("http://127.0.0.1:1"),
        )
        .unwrap();
        let auth = RuntimeAuthConfig {
            scheme: AuthScheme::Query,
            name: Some("api-token".into()),
            credential: Some("secret with / reserved?&=+".into()),
            credential_env: None,
        };
        let request = transport
            .request(
                reqwest::Method::GET,
                "http://127.0.0.1:1/unreachable",
                "vendor",
                &auth,
                &BTreeMap::from([(String::from("x-configured"), String::from("header secret"))]),
                &BTreeMap::from([(String::from("trace"), String::from("query value"))]),
            )
            .unwrap();
        let failure = transport.send(request, &auth).await.unwrap_err();
        assert_eq!(failure.status, None);
        assert_eq!(failure.retry_hint, RetryHint::Retryable);
        for representation in [
            "secret with / reserved?&=+",
            "secret%20with%20%2F%20reserved%3F%26%3D%2B",
            "secret+with+%2F+reserved%3F%26%3D%2B",
            "secret%20with%20%2f%20reserved%3f%26%3d%2b",
            "secret+with+%2f+reserved%3f%26%3d%2b",
        ] {
            assert!(!failure.detail().contains(representation));
            assert!(!format!("{failure:?}").contains(representation));
            assert!(!failure.to_string().contains(representation));
        }
    }

    #[tokio::test]
    async fn local_chunked_transport_response_preserves_boundary_metadata() {
        use std::io::Write;
        use std::net::TcpListener;
        use std::thread;

        let listener = TcpListener::bind(("127.0.0.1", 0)).unwrap();
        let address = listener.local_addr().unwrap();
        let server = thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            let mut request = [0_u8; 2048];
            let _ = std::io::Read::read(&mut stream, &mut request);
            for part in [
                b"HTTP/1.1 429 Too Many Requests\r\nRetry-After: 7\r\nX-Test: boundary\r\nTransfer-Encoding: chunked\r\n\r\n".as_slice(),
                b"5\r\nfirst\r\n".as_slice(),
                b"6\r\nsecond\r\n".as_slice(),
                b"0\r\n\r\n".as_slice(),
            ] {
                stream.write_all(part).unwrap();
                stream.flush().unwrap();
                thread::yield_now();
            }
        });
        let transport = ProviderTransport::new_for_endpoint(
            &RuntimeTransportConfig::default(),
            Some(&format!("http://{address}")),
        )
        .unwrap();
        let auth = RuntimeAuthConfig {
            scheme: AuthScheme::Query,
            name: Some("key".into()),
            credential: Some("secret".into()),
            credential_env: None,
        };
        let request = transport
            .request(
                reqwest::Method::GET,
                &format!("http://{address}/model"),
                "local",
                &auth,
                &BTreeMap::new(),
                &BTreeMap::from([(String::from("trace"), String::from("one"))]),
            )
            .unwrap();
        let mut response = transport.send(request, &auth).await.unwrap();
        assert_eq!(response.status, 429);
        assert_eq!(
            response.headers.get("x-test"),
            Some(&String::from("boundary"))
        );
        assert_eq!(response.retry_hint(), RetryHint::RetryAfterSeconds(7));
        let mut body = Vec::new();
        while let Some(chunk) = response.next_chunk().await {
            body.extend_from_slice(&chunk.unwrap());
        }
        assert_eq!(body, b"firstsecond");
        assert!(!body.windows(3).any(|window| window == b"5\r\n"));
        assert!(!body.windows(3).any(|window| window == b"6\r\n"));
        server.join().unwrap();
    }

    #[tokio::test]
    async fn truncated_response_chunk_preserves_status_and_retry_hint() {
        use std::io::Write;
        use std::net::TcpListener;
        use std::thread;

        let listener = TcpListener::bind(("127.0.0.1", 0)).unwrap();
        let address = listener.local_addr().unwrap();
        let server = thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            let mut request = [0_u8; 2048];
            let _ = std::io::Read::read(&mut stream, &mut request);
            stream
                .write_all(
                    b"HTTP/1.1 429 Too Many Requests\r\nRetry-After: 7\r\nContent-Length: 10\r\nConnection: close\r\n\r\nfirst",
                )
                .unwrap();
            stream.flush().unwrap();
        });
        let transport = ProviderTransport::new_for_endpoint(
            &RuntimeTransportConfig::default(),
            Some(&format!("http://{address}")),
        )
        .unwrap();
        let auth = RuntimeAuthConfig {
            scheme: AuthScheme::None,
            name: None,
            credential: None,
            credential_env: None,
        };
        let request = transport
            .request(
                reqwest::Method::GET,
                &format!("http://{address}/model"),
                "local",
                &auth,
                &BTreeMap::new(),
                &BTreeMap::new(),
            )
            .unwrap();
        let mut response = transport.send(request, &auth).await.unwrap();
        assert_eq!(response.status, 429);
        let mut failure = None;
        while let Some(chunk) = response.next_chunk().await {
            if let Err(error) = chunk {
                failure = Some(error);
                break;
            }
        }
        let failure = failure.expect("truncated body must fail while reading chunks");
        assert_eq!(failure.status, Some(429));
        assert_eq!(failure.retry_hint, RetryHint::RetryAfterSeconds(7));
        server.join().unwrap();
    }

    #[tokio::test]
    async fn transport_response_preserves_status_headers_and_chunks() {
        let mut headers = BTreeMap::new();
        headers.insert("retry-after".into(), "7".into());
        let mut response = TransportResponse::from_chunks(
            429,
            headers,
            vec![b"first".to_vec(), b"second".to_vec()],
        );
        assert_eq!(response.status, 429);
        assert_eq!(response.retry_hint(), RetryHint::RetryAfterSeconds(7));
        assert_eq!(response.next_chunk().await.unwrap().unwrap(), b"first");
        assert_eq!(response.next_chunk().await.unwrap().unwrap(), b"second");
        assert!(response.next_chunk().await.is_none());
    }
}
