use anyhow::{Context, Result, anyhow, bail};
use async_trait::async_trait;
use futures_util::stream::{self, StreamExt};
use indexmap::IndexMap;
use serde_json::{Value, json};
use std::process::Stdio;
use tokio::io::{AsyncBufReadExt, AsyncReadExt, AsyncWriteExt, BufReader};
use tokio::process::{Child, ChildStdin, ChildStdout, Command};
use tokio::time::{Duration, timeout};

use crate::config::{
    McpLocalServerConfig, McpRemoteServerConfig, McpServerConfig, McpTransportConfig,
};
use crate::permission::ToolPermissionClass;
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
/// `server` rather than deriving display data from the tools.
pub struct McpServerDiscovery {
    pub server: McpServerCatalogEntry,
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
    transport: McpTransportConfig,
    timeout_ms: u64,
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
                        tools: Vec::new(),
                    },
                );
            }
            let result = discover_server_tools(name.clone(), server).await;
            let (status, tools) = match result {
                Ok(tools) => (
                    McpServerStatus::Online {
                        tool_count: tools.len(),
                    },
                    tools,
                ),
                Err(error) => (
                    McpServerStatus::Offline {
                        message: error.to_string(),
                    },
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
) -> Result<Vec<McpTool>> {
    let discovered = match &server_config.transport {
        McpTransportConfig::Local(local) => {
            list_local_tools(&server_name, local, server_config.timeout_ms)
                .await
                .with_context(|| format!("failed to discover MCP tools from '{server_name}'"))?
        }
        McpTransportConfig::Remote(remote) => {
            list_remote_tools(&server_name, remote, server_config.timeout_ms)
                .await
                .with_context(|| {
                    format!("failed to discover MCP tools from remote '{server_name}'")
                })?
        }
    };

    discovered
        .into_iter()
        .map(|tool| {
            McpTool::from_discovered(
                &server_name,
                server_config.transport.clone(),
                server_config.timeout_ms,
                tool,
            )
        })
        .collect()
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
        match &self.transport {
            McpTransportConfig::Local(local) => {
                call_local_tool(
                    &self.server_name,
                    local,
                    self.timeout_ms,
                    &self.tool_name,
                    args,
                )
                .await
            }
            McpTransportConfig::Remote(remote) => {
                call_remote_tool(
                    &self.server_name,
                    remote,
                    self.timeout_ms,
                    &self.tool_name,
                    args,
                )
                .await
            }
        }
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
        transport: McpTransportConfig,
        timeout_ms: u64,
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
            transport,
            timeout_ms,
        })
    }
}

#[derive(Debug)]
struct DiscoveredTool {
    name: String,
    description: String,
    input_schema: Value,
}

async fn list_local_tools(
    server_name: &str,
    server: &McpLocalServerConfig,
    timeout_ms: u64,
) -> Result<Vec<DiscoveredTool>> {
    let mut session = LocalMcpSession::start(server, timeout_ms).await?;
    let result = async {
        session.initialize().await?;

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
    .await;
    session.shutdown().await;
    result
}

async fn list_remote_tools(
    server_name: &str,
    server: &McpRemoteServerConfig,
    timeout_ms: u64,
) -> Result<Vec<DiscoveredTool>> {
    let mut session = RemoteMcpSession::new(server, timeout_ms)?;
    session.initialize().await?;

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
            .with_context(|| format!("remote MCP server '{server_name}' failed tools/list"))?;
        let listed = result
            .get("tools")
            .and_then(Value::as_array)
            .ok_or_else(|| {
                anyhow!("remote MCP server '{server_name}' tools/list result is missing tools[]")
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

async fn call_local_tool(
    server_name: &str,
    server: &McpLocalServerConfig,
    timeout_ms: u64,
    tool_name: &str,
    arguments: Value,
) -> Result<Value> {
    let mut session = LocalMcpSession::start(server, timeout_ms).await?;
    let result = async {
        session.initialize().await?;
        let result = session
            .request(
                "tools/call",
                json!({
                    "name": tool_name,
                    "arguments": arguments,
                }),
            )
            .await
            .with_context(|| {
                format!("MCP server '{server_name}' failed tools/call '{tool_name}'")
            })?;

        if result
            .get("isError")
            .and_then(Value::as_bool)
            .unwrap_or(false)
        {
            bail!(
                "MCP tool '{server_name}::{tool_name}' returned an error: {}",
                mcp_content_text(&result)
            );
        }

        Ok(json!({
            "server": server_name,
            "tool": tool_name,
            "content": result.get("content").cloned().unwrap_or(Value::Array(Vec::new())),
        }))
    }
    .await;
    session.shutdown().await;
    result
}

async fn call_remote_tool(
    server_name: &str,
    server: &McpRemoteServerConfig,
    timeout_ms: u64,
    tool_name: &str,
    arguments: Value,
) -> Result<Value> {
    let mut session = RemoteMcpSession::new(server, timeout_ms)?;
    session.initialize().await?;
    let result = session
        .request(
            "tools/call",
            json!({
                "name": tool_name,
                "arguments": arguments,
            }),
        )
        .await
        .with_context(|| {
            format!("remote MCP server '{server_name}' failed tools/call '{tool_name}'")
        })?;

    if result
        .get("isError")
        .and_then(Value::as_bool)
        .unwrap_or(false)
    {
        bail!(
            "MCP tool '{server_name}::{tool_name}' returned an error: {}",
            mcp_content_text(&result)
        );
    }

    Ok(json!({
        "server": server_name,
        "tool": tool_name,
        "content": result.get("content").cloned().unwrap_or(Value::Array(Vec::new())),
    }))
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

struct LocalMcpSession {
    child: Option<Child>,
    stdin: Option<ChildStdin>,
    stdout: BufReader<ChildStdout>,
    next_id: u64,
    timeout: Duration,
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

        #[cfg(unix)]
        unsafe {
            // Keep wrappers such as npx and their server descendants in an owned
            // group so session teardown can terminate all of them together.
            command.pre_exec(|| {
                if libc::setpgid(0, 0) == -1 {
                    return Err(std::io::Error::last_os_error());
                }
                Ok(())
            });
        }

        let mut child = command.spawn().with_context(|| {
            format!(
                "failed to start MCP server command '{}': {:?}",
                program, server.command
            )
        })?;
        let stdin = child
            .stdin
            .take()
            .ok_or_else(|| anyhow!("failed to open MCP server stdin"))?;
        let stdout = child
            .stdout
            .take()
            .ok_or_else(|| anyhow!("failed to open MCP server stdout"))?;
        let stderr = child
            .stderr
            .take()
            .ok_or_else(|| anyhow!("failed to open MCP server stderr"))?;
        tokio::spawn(drain_mcp_stderr(stderr));

        Ok(Self {
            child: Some(child),
            stdin: Some(stdin),
            stdout: BufReader::new(stdout),
            next_id: 1,
            timeout: Duration::from_millis(timeout_ms),
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
        self.write_json(json!({
            "jsonrpc": "2.0",
            "id": id,
            "method": method,
            "params": params,
        }))
        .await?;
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
            let read = timeout(self.timeout, self.stdout.read_line(&mut line))
                .await
                .context("timed out reading MCP response")??;
            if read == 0 {
                bail!("MCP server closed stdout before responding to request {expected_id}");
            }
            let message: Value = serde_json::from_str(line.trim_end())
                .with_context(|| format!("failed to parse MCP JSON-RPC line: {line:?}"))?;
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
        // EOF is the standard MCP stdio shutdown signal; MCP defines no
        // shutdown/exit RPC. Keep the child in the session while awaiting so
        // cancellation invokes Drop's hard process-group cleanup.
        self.stdin.take();
        if self.child.is_none() {
            return;
        }

        #[cfg(unix)]
        let process_group = self
            .child
            .as_ref()
            .and_then(|child| child.id().map(|pid| pid as i32));

        let child_exited = matches!(
            timeout(
                LOCAL_MCP_SHUTDOWN_TIMEOUT,
                self.child.as_mut().expect("child remains owned").wait(),
            )
            .await,
            Ok(Ok(_))
        );

        #[cfg(not(unix))]
        if child_exited {
            self.child.take();
            return;
        }

        #[cfg(unix)]
        if let Some(process_group) = process_group {
            if child_exited && !process_group_exists(process_group) {
                self.child.take();
                return;
            }
            unsafe {
                // The child owns this group, so SIGTERM reaches wrappers and
                // all server descendants without affecting the parent process.
                libc::kill(-process_group, libc::SIGTERM);
            }
            let child_exited = child_exited
                || matches!(
                    timeout(
                        LOCAL_MCP_SHUTDOWN_TIMEOUT,
                        self.child.as_mut().expect("child remains owned").wait(),
                    )
                    .await,
                    Ok(Ok(_))
                );
            if child_exited && !process_group_exists(process_group) {
                self.child.take();
                return;
            }
            unsafe {
                libc::kill(-process_group, libc::SIGKILL);
            }
        }

        // Direct-child escalation is required on non-Unix and covers a child
        // that exited while the Unix process-group signals were being sent.
        let _ = self
            .child
            .as_mut()
            .expect("child remains owned")
            .start_kill();
        let child_reaped = matches!(
            timeout(
                LOCAL_MCP_SHUTDOWN_TIMEOUT,
                self.child.as_mut().expect("child remains owned").wait(),
            )
            .await,
            Ok(Ok(_))
        );
        if !child_reaped {
            // The spawned task now owns reaping; disarm Drop immediately
            // before transferring ownership to it.
            let mut child = self.child.take().expect("child remains owned");
            tokio::spawn(async move {
                let _ = child.wait().await;
            });
        } else {
            self.child.take();
        }
    }
}

#[cfg(unix)]
fn process_group_exists(process_group: i32) -> bool {
    unsafe {
        libc::kill(-process_group, 0) == 0
            || std::io::Error::last_os_error().raw_os_error() != Some(libc::ESRCH)
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
        // Async callers use shutdown() for the normal EOF -> TERM -> KILL
        // lifecycle. Drop remains a nonblocking hard-cancellation and panic
        // fallback, where waiting would be unsafe.
        self.stdin.take();

        let Some(mut child) = self.child.take() else {
            return;
        };

        #[cfg(unix)]
        if let Some(process_group) = child.id().map(|pid| pid as i32) {
            unsafe {
                // Ignore ESRCH: the group may have exited between discovery and
                // teardown. A negative PID targets the entire process group.
                libc::kill(-process_group, libc::SIGKILL);
            }
        }

        // This is also the direct-child fallback on non-Unix platforms.
        let _ = child.start_kill();

        if let Ok(runtime) = tokio::runtime::Handle::try_current() {
            runtime.spawn(async move {
                let _ = child.wait().await;
            });
        }
    }
}

struct RemoteMcpSession {
    client: reqwest::Client,
    url: String,
    headers: IndexMap<String, String>,
    session_id: Option<String>,
    protocol_version: String,
    next_id: u64,
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
    if let Some(items) = message.as_array() {
        if let Some(item) = items
            .iter()
            .find(|item| item.get("id").and_then(Value::as_u64) == Some(expected_id))
        {
            return Ok(Some(item.clone()));
        }
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
mod tests {
    use super::*;
    use crate::tool::ToolRegistry;
    use std::fs;

    #[cfg(unix)]
    use std::os::unix::fs::PermissionsExt;
    #[cfg(unix)]
    use tokio::time::sleep;

    #[test]
    fn sanitizes_mcp_tool_name_components() {
        assert_eq!(sanitize_tool_name_component("context-7"), "context_7");
        assert_eq!(sanitize_tool_name_component(" filesystem "), "filesystem");
        assert_eq!(sanitize_tool_name_component("a::b///c"), "a_b_c");
    }

    #[test]
    fn maps_discovered_tool_to_prefixed_tool_spec() {
        let server = McpLocalServerConfig {
            command: vec!["server".into()],
            environment: IndexMap::new(),
        };
        let tool = McpTool::from_discovered(
            "context-7",
            McpTransportConfig::Local(server),
            5_000,
            DiscoveredTool {
                name: "get-library-docs".into(),
                description: "Fetch docs".into(),
                input_schema: json!({"type":"object","properties":{"id":{"type":"string"}}}),
            },
        )
        .expect("tool should map");

        let spec: ToolSpec = tool.spec();
        assert_eq!(spec.name, "context_7__get_library_docs");
        assert_eq!(spec.description, "[MCP context-7] Fetch docs");
        assert_eq!(
            spec.parameters,
            json!({"type":"object","properties":{"id":{"type":"string"}}})
        );
        assert!(!spec.strict);
        assert_eq!(tool.permission_class(), ToolPermissionClass::Read);
    }

    #[tokio::test]
    async fn mcp_tools_cannot_claim_reserved_context_tool_names() {
        let server = McpLocalServerConfig {
            command: vec!["server".into()],
            environment: IndexMap::new(),
        };
        let mut registry = ToolRegistry::new();

        for tool_name in ["checkpoint", "return"] {
            let tool = McpTool::from_discovered(
                "context",
                McpTransportConfig::Local(server.clone()),
                5_000,
                DiscoveredTool {
                    name: tool_name.into(),
                    description: "reserved-name probe".into(),
                    input_schema: json!({"type": "object"}),
                },
            )
            .expect("MCP tool should map before registration");
            let name = tool.name().to_string();

            assert_eq!(name, format!("context__{tool_name}"));
            assert!(registry.try_register(tool).is_err());
            assert!(!registry.specs().iter().any(|spec| spec.name == name));

            let result = registry.call(&name, json!({})).await;
            assert!(!result.ok);
            assert_eq!(
                result.error.expect("unknown tool error").message,
                format!("unknown tool: {name}")
            );
        }
    }

    #[test]
    fn parses_remote_sse_json_response() {
        let parsed = parse_sse_json(
            r#"event: message
data: {"jsonrpc":"2.0","method":"notifications/progress","params":{}}

event: message
data: {"jsonrpc":"2.0","id":2,"result":{"tools":[]}}

"#,
            Some(2),
        )
        .expect("sse should parse");

        assert_eq!(parsed["id"], 2);
    }

    #[tokio::test]
    async fn per_server_discovery_keeps_order_and_isolates_offline_servers() {
        let config = IndexMap::from([
            (
                "disabled".into(),
                McpServerConfig {
                    enabled: false,
                    timeout_ms: 1,
                    transport: McpTransportConfig::Local(McpLocalServerConfig {
                        command: vec!["definitely-not-started".into()],
                        environment: IndexMap::new(),
                    }),
                },
            ),
            (
                "offline".into(),
                McpServerConfig {
                    enabled: true,
                    timeout_ms: 1,
                    transport: McpTransportConfig::Local(McpLocalServerConfig {
                        command: vec!["definitely-not-a-command".into()],
                        environment: IndexMap::new(),
                    }),
                },
            ),
        ]);

        let servers = discover_servers(&config).await;
        assert_eq!(servers.len(), 2);
        assert_eq!(servers[0].server.name, "disabled");
        assert_eq!(servers[0].server.status, McpServerStatus::Disabled);
        assert!(servers[0].tools.is_empty());
        assert_eq!(servers[1].server.name, "offline");
        assert!(matches!(
            servers[1].server.status,
            McpServerStatus::Offline { .. }
        ));
        assert!(servers[1].tools.is_empty());
    }

    #[tokio::test]
    #[cfg(unix)]
    async fn local_stdio_discovery_closes_stdin_and_allows_clean_eof_exit() {
        let dir = std::env::temp_dir().join(format!(
            "letcode-mcp-test-{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("system time before unix epoch")
                .as_nanos()
        ));
        fs::create_dir_all(&dir).expect("temp dir should be created");
        let script = dir.join("server.sh");
        let eof_marker = dir.join("observed-eof");
        fs::write(
            &script,
            r#"#!/bin/sh
IFS= read -r line
printf '%s\n' '{"jsonrpc":"2.0","id":1,"result":{"protocolVersion":"2025-06-18","capabilities":{"tools":{}},"serverInfo":{"name":"fake","version":"1"}}}'
IFS= read -r line
IFS= read -r line
printf '%s\n' '{"jsonrpc":"2.0","id":2,"result":{"tools":[{"name":"lookup-docs","description":"Lookup docs","inputSchema":{"type":"object","properties":{"query":{"type":"string"}},"required":["query"]}}]}}'
IFS= read -r line
"#
            .to_string()
                + &format!("touch '{}'\n", eof_marker.display()),
        )
        .expect("script should be written");
        let mut permissions = fs::metadata(&script)
            .expect("metadata should load")
            .permissions();
        permissions.set_mode(0o755);
        fs::set_permissions(&script, permissions).expect("permissions should be set");

        let server = McpLocalServerConfig {
            command: vec![script.to_string_lossy().to_string()],
            environment: IndexMap::new(),
        };
        let tools = list_local_tools("fake", &server, 5_000)
            .await
            .expect("tools should be discovered");

        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0].name, "lookup-docs");
        assert_eq!(tools[0].description, "Lookup docs");
        assert_eq!(tools[0].input_schema["required"], json!(["query"]));
        assert!(
            eof_marker.exists(),
            "server should observe stdin EOF and exit"
        );
        let _ = fs::remove_dir_all(dir);
    }

    #[tokio::test]
    #[cfg(unix)]
    async fn local_stdio_tool_call_closes_stdin_and_allows_clean_eof_exit() {
        let dir = std::env::temp_dir().join(format!(
            "letcode-mcp-call-eof-{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("system time before unix epoch")
                .as_nanos()
        ));
        fs::create_dir_all(&dir).expect("temp dir should be created");
        let script = dir.join("server.sh");
        let eof_marker = dir.join("observed-eof");
        fs::write(
            &script,
            format!(
                r#"#!/bin/sh
IFS= read -r line
printf '%s\n' '{{"jsonrpc":"2.0","id":1,"result":{{"protocolVersion":"2025-06-18","capabilities":{{}},"serverInfo":{{"name":"fake","version":"1"}}}}}}'
IFS= read -r line
IFS= read -r line
printf '%s\n' '{{"jsonrpc":"2.0","id":2,"result":{{"content":[{{"type":"text","text":"done"}}]}}}}'
IFS= read -r line
touch '{}'
"#,
                eof_marker.display()
            ),
        )
        .expect("script should be written");
        let mut permissions = fs::metadata(&script)
            .expect("metadata should load")
            .permissions();
        permissions.set_mode(0o755);
        fs::set_permissions(&script, permissions).expect("permissions should be set");

        let server = McpLocalServerConfig {
            command: vec![script.to_string_lossy().to_string()],
            environment: IndexMap::new(),
        };
        let result = call_local_tool("fake", &server, 5_000, "lookup-docs", json!({}))
            .await
            .expect("tool call should succeed");

        assert_eq!(result["content"][0]["text"], "done");
        assert!(
            eof_marker.exists(),
            "server should observe stdin EOF and exit"
        );
        let _ = fs::remove_dir_all(dir);
    }

    #[cfg(unix)]
    async fn wait_for_process_exit(pid: i32) {
        timeout(Duration::from_secs(2), async {
            loop {
                let alive = unsafe { libc::kill(pid, 0) == 0 };
                if !alive {
                    return;
                }
                sleep(Duration::from_millis(10)).await;
            }
        })
        .await
        .unwrap_or_else(|_| panic!("timed out waiting for process {pid} to exit"));
    }

    #[tokio::test]
    #[cfg(unix)]
    async fn local_stdio_discovery_escalates_a_stubborn_server() {
        let dir = std::env::temp_dir().join(format!(
            "letcode-mcp-cleanup-{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("system time before unix epoch")
                .as_nanos()
        ));
        fs::create_dir_all(&dir).expect("temp dir should be created");
        let script = dir.join("server.sh");
        let pid_file = dir.join("server.pid");
        fs::write(
            &script,
            format!(
                r#"#!/bin/sh
echo $$ > '{}'
IFS= read -r line
printf '%s\n' '{{"jsonrpc":"2.0","id":1,"result":{{"protocolVersion":"2025-06-18","capabilities":{{}},"serverInfo":{{"name":"fake","version":"1"}}}}}}'
IFS= read -r line
IFS= read -r line
printf '%s\n' '{{"jsonrpc":"2.0","id":2,"result":{{"tools":[]}}}}'
while :; do sleep 1; done
"#,
                pid_file.display()
            ),
        )
        .expect("script should be written");
        let mut permissions = fs::metadata(&script)
            .expect("metadata should load")
            .permissions();
        permissions.set_mode(0o755);
        fs::set_permissions(&script, permissions).expect("permissions should be set");

        let server = McpLocalServerConfig {
            command: vec![script.to_string_lossy().to_string()],
            environment: IndexMap::new(),
        };
        assert!(
            list_local_tools("fake", &server, 5_000)
                .await
                .expect("tools should be discovered")
                .is_empty()
        );

        let pid = fs::read_to_string(&pid_file)
            .expect("server should record its pid")
            .trim()
            .parse()
            .expect("server pid should be numeric");
        wait_for_process_exit(pid).await;
        let _ = fs::remove_dir_all(dir);
    }

    #[tokio::test]
    #[cfg(unix)]
    async fn local_stdio_discovery_escalation_kills_stubborn_descendants_in_its_process_group() {
        let dir = std::env::temp_dir().join(format!(
            "letcode-mcp-process-group-{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("system time before unix epoch")
                .as_nanos()
        ));
        fs::create_dir_all(&dir).expect("temp dir should be created");
        let script = dir.join("server.sh");
        let release = dir.join("release");
        let marker = dir.join("descendant-marker");
        let descendant_pid = dir.join("descendant.pid");
        fs::write(
            &script,
            format!(
                r#"#!/bin/sh
(
  trap '' TERM
  while [ ! -e '{}' ]; do sleep 1; done
  touch '{}'
) &
echo $! > '{}'
IFS= read -r line
printf '%s\n' '{{"jsonrpc":"2.0","id":1,"result":{{"protocolVersion":"2025-06-18","capabilities":{{}},"serverInfo":{{"name":"fake","version":"1"}}}}}}'
IFS= read -r line
IFS= read -r line
printf '%s\n' '{{"jsonrpc":"2.0","id":2,"result":{{"tools":[]}}}}'
while :; do sleep 1; done
"#,
                release.display(),
                marker.display(),
                descendant_pid.display(),
            ),
        )
        .expect("script should be written");
        let mut permissions = fs::metadata(&script)
            .expect("metadata should load")
            .permissions();
        permissions.set_mode(0o755);
        fs::set_permissions(&script, permissions).expect("permissions should be set");

        let server = McpLocalServerConfig {
            command: vec![script.to_string_lossy().to_string()],
            environment: IndexMap::new(),
        };
        assert!(
            list_local_tools("fake", &server, 5_000)
                .await
                .expect("tools should be discovered")
                .is_empty()
        );

        let pid = fs::read_to_string(&descendant_pid)
            .expect("descendant should record its pid")
            .trim()
            .parse()
            .expect("descendant pid should be numeric");
        fs::write(&release, "go").expect("release side effect");
        wait_for_process_exit(pid).await;
        sleep(Duration::from_millis(100)).await;
        assert!(
            !marker.exists(),
            "descendant survived session teardown and performed a side effect"
        );
        let _ = fs::remove_dir_all(dir);
    }
}
