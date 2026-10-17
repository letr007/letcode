use anyhow::{Context, Result, anyhow, bail};
use async_trait::async_trait;
use futures_util::stream::{self, StreamExt};
use indexmap::IndexMap;
use serde_json::{Value, json};
use std::process::Stdio;
use std::sync::Arc;
use tokio::io::{AsyncBufReadExt, AsyncReadExt, AsyncWriteExt, BufReader};
use tokio::process::{ChildStdin, ChildStdout, Command};
use tokio::sync::Mutex;
use tokio::time::{Duration, timeout};

use crate::config::{
    McpLocalServerConfig, McpRemoteServerConfig, McpServerConfig, McpTransportConfig,
};
use crate::permission::ToolPermissionClass;
use crate::process_tree::ManagedChild;
#[cfg(test)]
use crate::request_builder::ToolSpec;
use crate::tool::ToolHandler;

const MCP_PROTOCOL_VERSION: &str = "2025-06-18";
const MCP_DISCOVERY_CONCURRENCY: usize = 4;
const LOCAL_MCP_SHUTDOWN_TIMEOUT: Duration = Duration::from_millis(250);

/// Lightweight per-server state for a server-management UI.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct McpServerCatalogEntry {
    pub name: String,
    pub enabled: bool,
    pub status: McpServerStatus,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum McpServerStatus {
    Disabled,
    Online { tool_count: usize },
    Offline { message: String },
}

/// Discovery outcome for one configured server. Tool details are available only
/// to the runtime that registers executable MCP tools; catalog UIs should use
/// `server` rather than deriving display data from the tools. `session` is the
/// connection shared by `tools`; its owner closes it once the server goes away.
pub struct McpServerDiscovery {
    pub server: McpServerCatalogEntry,
    pub session: Option<McpSession>,
    pub tools: Vec<McpTool>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct McpToolCatalogEntry {
    pub name: String,
    pub description: String,
}

#[derive(Clone)]
pub struct McpTool {
    name: String,
    description: String,
    parameters: Value,
    server_name: String,
    tool_name: String,
    session: McpSession,
}

/// Discover every configured MCP server independently, preserving config order.
/// A server failure is represented in that server's status and does not discard
/// tools found from other servers. Disabled servers are never contacted.
pub async fn discover_servers(
    config: &IndexMap<String, McpServerConfig>,
) -> Vec<McpServerDiscovery> {
    let configured = config
        .iter()
        .enumerate()
        .map(|(index, (name, server))| (index, name.clone(), server.clone()))
        .collect::<Vec<_>>();
    let mut discovered = stream::iter(configured)
        .map(|(index, name, server)| async move {
            if !server.enabled {
                return (
                    index,
                    McpServerDiscovery {
                        server: McpServerCatalogEntry {
                            name,
                            enabled: false,
                            status: McpServerStatus::Disabled,
                        },
                        session: None,
                        tools: Vec::new(),
                    },
                );
            }
            let result = discover_server_tools(name.clone(), server).await;
            let (status, session, tools) = match result {
                Ok((session, tools)) => (
                    McpServerStatus::Online {
                        tool_count: tools.len(),
                    },
                    Some(session),
                    tools,
                ),
                Err(error) => (
                    McpServerStatus::Offline {
                        message: error.to_string(),
                    },
                    None,
                    Vec::new(),
                ),
            };
            (
                index,
                McpServerDiscovery {
                    server: McpServerCatalogEntry {
                        name,
                        enabled: true,
                        status,
                    },
                    session,
                    tools,
                },
            )
        })
        .buffer_unordered(MCP_DISCOVERY_CONCURRENCY)
        .collect::<Vec<_>>()
        .await;
    discovered.sort_by_key(|(index, _)| *index);
    discovered.into_iter().map(|(_, result)| result).collect()
}

async fn discover_server_tools(
    server_name: String,
    server_config: McpServerConfig,
) -> Result<(McpSession, Vec<McpTool>)> {
    let session = McpSession::new(server_config.transport, server_config.timeout_ms);
    let discovered = list_tools(&server_name, &session)
        .await
        .with_context(|| format!("failed to discover MCP tools from '{server_name}'"))?;
    let tools = discovered
        .into_iter()
        .map(|tool| McpTool::from_discovered(&server_name, session.clone(), tool))
        .collect::<Result<Vec<_>>>()?;
    Ok((session, tools))
}

#[async_trait]
impl ToolHandler for McpTool {
    fn name(&self) -> &str {
        &self.name
    }

    fn description(&self) -> &str {
        &self.description
    }

    fn parameters(&self) -> Value {
        self.parameters.clone()
    }

    fn strict(&self) -> bool {
        false
    }

    fn permission_class(&self) -> ToolPermissionClass {
        ToolPermissionClass::Read
    }

    fn parallelism(&self) -> crate::tool::ToolParallelism {
        crate::tool::ToolParallelism::Exclusive
    }

    async fn execute(&self, args: Value) -> Result<Value> {
        let result = self
            .session
            .request(
                "tools/call",
                json!({
                    "name": self.tool_name,
                    "arguments": args,
                }),
            )
            .await
            .with_context(|| {
                format!(
                    "MCP server '{}' failed tools/call '{}'",
                    self.server_name, self.tool_name
                )
            })?;

        if result
            .get("isError")
            .and_then(Value::as_bool)
            .unwrap_or(false)
        {
            bail!(
                "MCP tool '{}::{}' returned an error: {}",
                self.server_name,
                self.tool_name,
                mcp_content_text(&result)
            );
        }

        Ok(json!({
            "server": self.server_name,
            "tool": self.tool_name,
            "content": result.get("content").cloned().unwrap_or(Value::Array(Vec::new())),
        }))
    }
}

impl McpTool {
    pub fn catalog_entry(&self) -> McpToolCatalogEntry {
        McpToolCatalogEntry {
            name: self.tool_name.clone(),
            description: self.description.clone(),
        }
    }

    fn from_discovered(
        server_name: &str,
        session: McpSession,
        tool: DiscoveredTool,
    ) -> Result<Self> {
        let server_component = sanitize_tool_name_component(server_name);
        let tool_component = sanitize_tool_name_component(&tool.name);
        if server_component.is_empty() || tool_component.is_empty() {
            bail!("MCP tool names must contain at least one ASCII letter, digit, or underscore");
        }

        let description = match tool.description.trim() {
            "" => format!("MCP tool '{}' from server '{}'.", tool.name, server_name),
            description => format!("[MCP {server_name}] {description}"),
        };

        Ok(Self {
            name: format!("{server_component}__{tool_component}"),
            description,
            parameters: tool.input_schema,
            server_name: server_name.to_string(),
            tool_name: tool.name,
            session,
        })
    }
}

#[derive(Debug)]
struct DiscoveredTool {
    name: String,
    description: String,
    input_schema: Value,
}

async fn list_tools(server_name: &str, session: &McpSession) -> Result<Vec<DiscoveredTool>> {
    let mut cursor = None;
    let mut tools = Vec::new();
    loop {
        let mut params = serde_json::Map::new();
        if let Some(cursor) = cursor.take() {
            params.insert("cursor".into(), Value::String(cursor));
        }
        let result = session
            .request("tools/list", Value::Object(params))
            .await
            .with_context(|| format!("MCP server '{server_name}' failed tools/list"))?;
        let listed = result
            .get("tools")
            .and_then(Value::as_array)
            .ok_or_else(|| {
                anyhow!("MCP server '{server_name}' tools/list result is missing tools[]")
            })?;
        for tool in listed {
            tools.push(discovered_tool_from_value(server_name, tool)?);
        }

        cursor = result
            .get("nextCursor")
            .and_then(Value::as_str)
            .map(str::to_string);
        if cursor.is_none() {
            break;
        }
    }

    Ok(tools)
}

fn discovered_tool_from_value(server_name: &str, tool: &Value) -> Result<DiscoveredTool> {
    let name = tool
        .get("name")
        .and_then(Value::as_str)
        .ok_or_else(|| anyhow!("MCP server '{server_name}' returned a tool without name"))?
        .to_string();
    let description = tool
        .get("description")
        .and_then(Value::as_str)
        .or_else(|| tool.get("title").and_then(Value::as_str))
        .unwrap_or_default()
        .to_string();
    let input_schema = tool
        .get("inputSchema")
        .cloned()
        .unwrap_or_else(|| json!({ "type": "object", "properties": {} }));
    Ok(DiscoveredTool {
        name,
        description,
        input_schema,
    })
}

/// The connection to one configured MCP server, shared by every tool discovered
/// from it so the server keeps its session state across calls. A request that
/// leaves the transport unusable drops the connection, so the next request
/// reconnects; otherwise the session is reused until `shutdown` closes it.
#[derive(Clone)]
pub struct McpSession {
    inner: Arc<Mutex<McpSessionState>>,
}

struct McpSessionState {
    transport: McpTransportConfig,
    timeout_ms: u64,
    active: Option<ActiveSession>,
}

enum ActiveSession {
    Local(LocalMcpSession),
    Remote(RemoteMcpSession),
}

impl McpSession {
    fn new(transport: McpTransportConfig, timeout_ms: u64) -> Self {
        Self {
            inner: Arc::new(Mutex::new(McpSessionState {
                transport,
                timeout_ms,
                active: None,
            })),
        }
    }

    async fn request(&self, method: &str, params: Value) -> Result<Value> {
        let mut state = self.inner.lock().await;
        let mut active = match state.active.take() {
            Some(active) => active,
            None => {
                let transport = state.transport.clone();
                connect_session(&transport, state.timeout_ms).await?
            }
        };
        let outcome = active.request(method, params).await;
        if active.is_poisoned() {
            active.shutdown().await;
        } else {
            state.active = Some(active);
        }
        outcome
    }

    /// Closes the connection. Only stdio servers have a shutdown handshake.
    pub async fn shutdown(&self) {
        let mut state = self.inner.lock().await;
        if let Some(mut active) = state.active.take() {
            active.shutdown().await;
        }
    }
}

async fn connect_session(transport: &McpTransportConfig, timeout_ms: u64) -> Result<ActiveSession> {
    match transport {
        McpTransportConfig::Local(local) => {
            let mut session = LocalMcpSession::start(local, timeout_ms).await?;
            session.initialize().await?;
            Ok(ActiveSession::Local(session))
        }
        McpTransportConfig::Remote(remote) => {
            let mut session = RemoteMcpSession::new(remote, timeout_ms)?;
            session.initialize().await?;
            Ok(ActiveSession::Remote(session))
        }
    }
}

impl ActiveSession {
    async fn request(&mut self, method: &str, params: Value) -> Result<Value> {
        match self {
            Self::Local(session) => session.request(method, params).await,
            Self::Remote(session) => session.request(method, params).await,
        }
    }

    fn is_poisoned(&self) -> bool {
        match self {
            Self::Local(session) => session.is_poisoned(),
            Self::Remote(session) => session.is_poisoned(),
        }
    }

    async fn shutdown(&mut self) {
        if let Self::Local(session) = self {
            session.shutdown().await;
        }
    }
}

struct LocalMcpSession {
    child: Option<ManagedChild>,
    stdin: Option<ChildStdin>,
    stdout: BufReader<ChildStdout>,
    next_id: u64,
    timeout: Duration,
    poisoned: bool,
}

impl LocalMcpSession {
    async fn start(server: &McpLocalServerConfig, timeout_ms: u64) -> Result<Self> {
        let Some(program) = server.command.first() else {
            bail!("MCP local command cannot be empty");
        };
        let mut command = Command::new(program);
        command
            .args(server.command.iter().skip(1))
            .envs(server.environment.iter())
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());

        let mut child = ManagedChild::spawn(command)
            .with_context(|| format!("failed to start MCP server command '{program}'"))?;
        let stdin = child
            .take_stdin()
            .ok_or_else(|| anyhow!("failed to open MCP server stdin"))?;
        let stdout = child
            .take_stdout()
            .ok_or_else(|| anyhow!("failed to open MCP server stdout"))?;
        let stderr = child
            .take_stderr()
            .ok_or_else(|| anyhow!("failed to open MCP server stderr"))?;
        tokio::spawn(drain_mcp_stderr(stderr));

        Ok(Self {
            child: Some(child),
            stdin: Some(stdin),
            stdout: BufReader::new(stdout),
            next_id: 1,
            timeout: Duration::from_millis(timeout_ms),
            poisoned: false,
        })
    }

    async fn initialize(&mut self) -> Result<()> {
        self.request(
            "initialize",
            json!({
                "protocolVersion": MCP_PROTOCOL_VERSION,
                "capabilities": {},
                "clientInfo": {
                    "name": "letcode",
                    "version": env!("CARGO_PKG_VERSION"),
                },
            }),
        )
        .await?;
        self.notification(
            "notifications/initialized",
            Value::Object(Default::default()),
        )
        .await?;
        Ok(())
    }

    async fn request(&mut self, method: &str, params: Value) -> Result<Value> {
        let id = self.next_id;
        self.next_id += 1;
        if let Err(error) = self
            .write_json(json!({
                "jsonrpc": "2.0",
                "id": id,
                "method": method,
                "params": params,
            }))
            .await
        {
            self.poisoned = true;
            return Err(error);
        }
        self.read_response(id).await
    }

    async fn notification(&mut self, method: &str, params: Value) -> Result<()> {
        self.write_json(json!({
            "jsonrpc": "2.0",
            "method": method,
            "params": params,
        }))
        .await
    }

    async fn write_json(&mut self, message: Value) -> Result<()> {
        let mut line = serde_json::to_vec(&message)?;
        line.push(b'\n');
        let stdin = self
            .stdin
            .as_mut()
            .ok_or_else(|| anyhow!("MCP server stdin is closed"))?;
        timeout(self.timeout, stdin.write_all(&line))
            .await
            .context("timed out writing MCP message")??;
        timeout(self.timeout, stdin.flush())
            .await
            .context("timed out flushing MCP message")??;
        Ok(())
    }

    async fn read_response(&mut self, expected_id: u64) -> Result<Value> {
        loop {
            let mut line = String::new();
            let read = match timeout(self.timeout, self.stdout.read_line(&mut line)).await {
                Ok(Ok(read)) => read,
                Ok(Err(error)) => {
                    self.poisoned = true;
                    return Err(error.into());
                }
                Err(_) => return Err(anyhow!("timed out reading MCP response")),
            };
            if read == 0 {
                self.poisoned = true;
                bail!("MCP server closed stdout before responding to request {expected_id}");
            }
            let message: Value = match serde_json::from_str(line.trim_end()) {
                Ok(message) => message,
                Err(error) => {
                    self.poisoned = true;
                    return Err(error)
                        .with_context(|| format!("failed to parse MCP JSON-RPC line: {line:?}"));
                }
            };
            if message.get("id").and_then(Value::as_u64) != Some(expected_id) {
                continue;
            }
            if let Some(error) = message.get("error") {
                bail!("MCP request {expected_id} failed: {error}");
            }
            return message
                .get("result")
                .cloned()
                .ok_or_else(|| anyhow!("MCP response {expected_id} is missing result"));
        }
    }

    async fn shutdown(&mut self) {
        // EOF is the standard MCP stdio shutdown signal. If the server does not
        // exit promptly, terminate its owned process tree.
        self.stdin.take();
        let Some(mut child) = self.child.take() else {
            return;
        };
        let child_exited = matches!(
            timeout(LOCAL_MCP_SHUTDOWN_TIMEOUT, child.wait()).await,
            Ok(Ok(_))
        );
        if child_exited {
            child.disarm();
            return;
        }
        child.terminate_gracefully();
        let graceful_exit = matches!(
            timeout(LOCAL_MCP_SHUTDOWN_TIMEOUT, child.wait()).await,
            Ok(Ok(_))
        );
        if graceful_exit {
            child.disarm();
            return;
        }
        child.terminate();
        let child_reaped = matches!(
            timeout(LOCAL_MCP_SHUTDOWN_TIMEOUT, child.wait()).await,
            Ok(Ok(_))
        );
        if child_reaped {
            child.disarm();
        } else {
            child.spawn_reaper();
        }
    }

    fn is_poisoned(&self) -> bool {
        self.poisoned
    }
}

async fn drain_mcp_stderr(mut stderr: tokio::process::ChildStderr) {
    let mut buffer = [0_u8; 8192];
    loop {
        match stderr.read(&mut buffer).await {
            Ok(0) | Err(_) => break,
            Ok(_) => {}
        }
    }
}

impl Drop for LocalMcpSession {
    fn drop(&mut self) {
        self.stdin.take();
        // ManagedChild owns cancellation cleanup and asynchronous reaping.
        self.child.take();
    }
}

struct RemoteMcpSession {
    client: reqwest::Client,
    url: String,
    headers: IndexMap<String, String>,
    session_id: Option<String>,
    protocol_version: String,
    next_id: u64,
    poisoned: bool,
}

impl RemoteMcpSession {
    fn new(server: &McpRemoteServerConfig, timeout_ms: u64) -> Result<Self> {
        if server.oauth {
            bail!(
                "remote MCP OAuth is not supported yet; set oauth = false and provide headers for API-key based servers"
            );
        }

        let client = reqwest::Client::builder()
            .timeout(Duration::from_millis(timeout_ms))
            .build()
            .context("failed to build remote MCP HTTP client")?;
        Ok(Self {
            client,
            url: server.url.clone(),
            headers: server.headers.clone(),
            session_id: None,
            protocol_version: MCP_PROTOCOL_VERSION.to_string(),
            next_id: 1,
            poisoned: false,
        })
    }

    async fn initialize(&mut self) -> Result<()> {
        let result = self
            .request(
                "initialize",
                json!({
                    "protocolVersion": MCP_PROTOCOL_VERSION,
                    "capabilities": {},
                    "clientInfo": {
                        "name": "letcode",
                        "version": env!("CARGO_PKG_VERSION"),
                    },
                }),
            )
            .await?;
        if let Some(protocol_version) = result.get("protocolVersion").and_then(Value::as_str) {
            self.protocol_version = protocol_version.to_string();
        }
        self.notification(
            "notifications/initialized",
            Value::Object(Default::default()),
        )
        .await?;
        Ok(())
    }

    async fn request(&mut self, method: &str, params: Value) -> Result<Value> {
        let id = self.next_id;
        self.next_id += 1;
        let response = self
            .post(
                json!({
                "jsonrpc": "2.0",
                "id": id,
                "method": method,
                "params": params,
                }),
                Some(id),
            )
            .await?;
        parse_json_rpc_result(response, id)
    }

    async fn notification(&mut self, method: &str, params: Value) -> Result<()> {
        self.post(
            json!({
                "jsonrpc": "2.0",
                "method": method,
                "params": params,
            }),
            None,
        )
        .await?;
        Ok(())
    }

    async fn post(&mut self, message: Value, expected_id: Option<u64>) -> Result<Value> {
        let mut request = self
            .client
            .post(&self.url)
            .header("Accept", "application/json, text/event-stream")
            .header("MCP-Protocol-Version", &self.protocol_version)
            .json(&message);
        for (key, value) in &self.headers {
            request = request.header(key, value);
        }
        if let Some(session_id) = &self.session_id {
            request = request.header("Mcp-Session-Id", session_id);
        }

        let response = request
            .send()
            .await
            .with_context(|| format!("failed to POST MCP request to {}", self.url))?;
        if let Some(session_id) = response.headers().get("Mcp-Session-Id") {
            self.session_id = Some(
                session_id
                    .to_str()
                    .context("remote MCP session id header is not valid UTF-8")?
                    .to_string(),
            );
        }

        let status = response.status();
        let content_type = response
            .headers()
            .get(reqwest::header::CONTENT_TYPE)
            .and_then(|value| value.to_str().ok())
            .unwrap_or_default()
            .to_string();
        let body = response
            .text()
            .await
            .context("failed to read remote MCP response body")?;
        if !status.is_success() {
            if status == reqwest::StatusCode::NOT_FOUND {
                // The server no longer knows this session, so the next request
                // has to initialize a new one.
                self.poisoned = true;
            }
            bail!("remote MCP server returned HTTP {status}: {body}");
        }
        if body.trim().is_empty() {
            return Ok(Value::Null);
        }
        if content_type.contains("text/event-stream") {
            return parse_sse_json(&body, expected_id);
        }
        serde_json::from_str(&body)
            .with_context(|| format!("failed to parse remote MCP JSON response: {body}"))
    }

    fn is_poisoned(&self) -> bool {
        self.poisoned
    }
}

fn mcp_content_text(result: &Value) -> String {
    result
        .get("content")
        .and_then(Value::as_array)
        .map(|content| {
            content
                .iter()
                .filter_map(|part| part.get("text").and_then(Value::as_str))
                .collect::<Vec<_>>()
                .join("\n")
        })
        .filter(|text| !text.is_empty())
        .unwrap_or_else(|| result.to_string())
}

fn parse_json_rpc_result(message: Value, expected_id: u64) -> Result<Value> {
    if message.is_null() {
        bail!("remote MCP response is missing response id {expected_id}");
    }
    if let Some(items) = message.as_array() {
        let Some(item) = items
            .iter()
            .find(|item| item.get("id").and_then(Value::as_u64) == Some(expected_id))
        else {
            bail!("remote MCP response batch did not include response id {expected_id}");
        };
        return parse_json_rpc_result(item.clone(), expected_id);
    }
    if message.get("id").and_then(Value::as_u64) != Some(expected_id) {
        bail!("remote MCP response id did not match request id {expected_id}: {message}");
    }
    if let Some(error) = message.get("error") {
        bail!("MCP request {expected_id} failed: {error}");
    }
    message
        .get("result")
        .cloned()
        .ok_or_else(|| anyhow!("MCP response {expected_id} is missing result"))
}

fn parse_sse_json(body: &str, expected_id: Option<u64>) -> Result<Value> {
    let mut event_data = Vec::new();
    for line in body.lines() {
        let line = line.trim_end();
        if line.is_empty() {
            if let Some(message) = parse_sse_event(&event_data, expected_id)? {
                return Ok(message);
            }
            event_data.clear();
            continue;
        }
        if let Some(data) = line.strip_prefix("data:") {
            event_data.push(data.trim_start().to_string());
        }
    }
    if let Some(message) = parse_sse_event(&event_data, expected_id)? {
        return Ok(message);
    }
    if let Some(expected_id) = expected_id {
        bail!("remote MCP SSE stream did not include response id {expected_id}");
    }
    Ok(Value::Null)
}

fn parse_sse_event(event_data: &[String], expected_id: Option<u64>) -> Result<Option<Value>> {
    if event_data.is_empty() {
        return Ok(None);
    }
    let data = event_data.join("\n");
    if data.trim() == "[DONE]" {
        return Ok(None);
    }
    let message: Value = serde_json::from_str(&data)
        .with_context(|| format!("failed to parse remote MCP SSE data: {data}"))?;
    let Some(expected_id) = expected_id else {
        return Ok(Some(message));
    };
    if message.get("id").and_then(Value::as_u64) == Some(expected_id) {
        return Ok(Some(message));
    }
    if let Some(items) = message.as_array()
        && let Some(item) = items
            .iter()
            .find(|item| item.get("id").and_then(Value::as_u64) == Some(expected_id))
    {
        return Ok(Some(item.clone()));
    }
    Ok(None)
}

fn sanitize_tool_name_component(value: &str) -> String {
    let mut sanitized = String::new();
    let mut previous_was_underscore = false;
    for ch in value.chars() {
        let normalized = if ch.is_ascii_alphanumeric() || ch == '_' {
            Some(ch)
        } else {
            Some('_')
        };
        if let Some(ch) = normalized {
            if ch == '_' {
                if !previous_was_underscore && !sanitized.is_empty() {
                    sanitized.push('_');
                }
                previous_was_underscore = true;
            } else {
                sanitized.push(ch);
                previous_was_underscore = false;
            }
        }
    }
    sanitized.trim_matches('_').to_string()
}

#[cfg(test)]
#[path = "../tests/internal/mcp/integration_tests.rs"]
mod integration_tests;
