use super::*;
use crate::tool::ToolRegistry;
use std::fs;

#[cfg(unix)]
use std::os::unix::fs::PermissionsExt;
#[cfg(unix)]
use tokio::time::sleep;

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
            McpSession::new(McpTransportConfig::Local(server.clone()), 5_000),
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

#[cfg(unix)]
fn fake_local_server(name: &str, body: &str) -> (McpLocalServerConfig, std::path::PathBuf) {
    let dir = std::env::temp_dir().join(format!(
        "letcode-mcp-{name}-{}",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("system time before unix epoch")
            .as_nanos()
    ));
    fs::create_dir_all(&dir).expect("temp dir should be created");
    let script = dir.join("server.sh");
    fs::write(&script, format!("#!/bin/sh\n{body}")).expect("script should be written");
    let mut permissions = fs::metadata(&script)
        .expect("metadata should load")
        .permissions();
    permissions.set_mode(0o755);
    fs::set_permissions(&script, permissions).expect("permissions should be set");
    (
        McpLocalServerConfig {
            command: vec![script.to_string_lossy().to_string()],
            environment: IndexMap::new(),
        },
        dir,
    )
}

#[cfg(unix)]
#[tokio::test]
async fn local_stdio_tool_calls_reuse_one_session() {
    let (server, dir) = fake_local_server(
        "session-reuse",
        r#"
count=0
while IFS= read -r line; do
  id=$(printf '%s' "$line" | sed -n 's/.*"id":\([0-9][0-9]*\).*/\1/p')
  case "$line" in
    *'"method":"initialize"'*)
      printf '{"jsonrpc":"2.0","id":%s,"result":{"protocolVersion":"2025-06-18","capabilities":{"tools":{}},"serverInfo":{"name":"fake","version":"1"}}}\n' "$id"
      ;;
    *'"method":"tools/list"'*)
      printf '{"jsonrpc":"2.0","id":%s,"result":{"tools":[{"name":"bump","inputSchema":{"type":"object"}}]}}\n' "$id"
      ;;
    *'"method":"tools/call"'*)
      count=$((count + 1))
      printf '{"jsonrpc":"2.0","id":%s,"result":{"content":[{"type":"text","text":"%s"}]}}\n' "$id" "$count"
      ;;
  esac
done
"#,
    );
    let session = McpSession::new(McpTransportConfig::Local(server), 5_000);

    let first = session
        .request("tools/call", json!({"name": "bump", "arguments": {}}))
        .await
        .expect("first call should succeed");
    let second = session
        .request("tools/call", json!({"name": "bump", "arguments": {}}))
        .await
        .expect("second call should succeed");

    assert_eq!(first["content"][0]["text"], "1");
    assert_eq!(second["content"][0]["text"], "2");
    session.shutdown().await;
    let _ = fs::remove_dir_all(dir);
}

#[cfg(unix)]
#[tokio::test]
async fn local_stdio_session_reconnects_after_the_server_exits() {
    let (server, dir) = fake_local_server(
        "session-reconnect",
        r#"
while IFS= read -r line; do
  id=$(printf '%s' "$line" | sed -n 's/.*"id":\([0-9][0-9]*\).*/\1/p')
  case "$line" in
    *'"method":"initialize"'*)
      printf '{"jsonrpc":"2.0","id":%s,"result":{"protocolVersion":"2025-06-18","capabilities":{"tools":{}},"serverInfo":{"name":"fake","version":"1"}}}\n' "$id"
      ;;
    *'"method":"tools/call"'*)
      case "$line" in
        *'"die":true'*) exit 0 ;;
      esac
      printf '{"jsonrpc":"2.0","id":%s,"result":{"content":[{"type":"text","text":"alive"}]}}\n' "$id"
      ;;
  esac
done
"#,
    );
    let session = McpSession::new(McpTransportConfig::Local(server), 5_000);

    let error = session
        .request(
            "tools/call",
            json!({"name": "probe", "arguments": {"die": true}}),
        )
        .await
        .expect_err("a call whose server exits must fail");
    assert!(
        error.to_string().contains("closed stdout"),
        "unexpected error: {error}"
    );

    let recovered = session
        .request("tools/call", json!({"name": "probe", "arguments": {}}))
        .await
        .expect("the next call should reconnect");
    assert_eq!(recovered["content"][0]["text"], "alive");
    session.shutdown().await;
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
async fn remote_http_negotiates_protocol_before_sending_version_header() {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!("http://{}/mcp", listener.local_addr().unwrap());
    let server = tokio::spawn(async move {
        timeout(Duration::from_secs(10), async {
            for method in ["initialize", "notifications/initialized", "tools/list"] {
                let (socket, _) = listener.accept().await.unwrap();
                let mut socket = BufReader::new(socket);
                let mut line = String::new();
                socket.read_line(&mut line).await.unwrap();
                assert!(line.starts_with("POST /mcp HTTP/1.1"));
                let mut length = 0;
                let mut versions = Vec::new();
                let mut session_id = None;
                loop {
                    line.clear();
                    assert_ne!(socket.read_line(&mut line).await.unwrap(), 0);
                    if line == "\r\n" {
                        break;
                    }
                    let (name, value) = line.trim_end().split_once(':').unwrap();
                    if name.eq_ignore_ascii_case("content-length") {
                        length = value.trim().parse::<usize>().unwrap();
                    } else if name.eq_ignore_ascii_case("mcp-protocol-version") {
                        versions.push(value.trim().to_string());
                    } else if name.eq_ignore_ascii_case("mcp-session-id") {
                        session_id = Some(value.trim().to_string());
                    }
                }
                let mut body = vec![0; length];
                socket.read_exact(&mut body).await.unwrap();
                let request: Value = serde_json::from_slice(&body).unwrap();
                assert_eq!(request["method"], method);
                let (status, body) = if method == "initialize" {
                    assert!(versions.is_empty(), "unnegotiated header: {versions:?}");
                    assert!(session_id.is_none());
                    assert_eq!(request["params"]["protocolVersion"], MCP_PROTOCOL_VERSION);
                    (
                        "200 OK",
                        json!({
                            "jsonrpc": "2.0", "id": request["id"],
                            "result": {
                                "protocolVersion": "2025-11-25",
                                "capabilities": {"tools": {}},
                                "serverInfo": {"name": "test", "version": "1"}
                            }
                        }).to_string(),
                    )
                } else {
                    assert_eq!(versions, ["2025-11-25"]);
                    assert_eq!(session_id.as_deref(), Some("test-session"));
                    if method == "notifications/initialized" {
                        assert!(request.get("id").is_none());
                        ("202 Accepted", String::new())
                    } else {
                        (
                            "200 OK",
                            json!({
                                "jsonrpc": "2.0", "id": request["id"],
                                "result": {"tools": [{
                                    "name": "probe", "inputSchema": {"type": "object"}
                                }]}
                            }).to_string(),
                        )
                    }
                };
                let response = format!(
                    "HTTP/1.1 {status}\r\nContent-Type: application/json\r\nMcp-Session-Id: test-session\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                    body.len()
                );
                socket.get_mut().write_all(response.as_bytes()).await.unwrap();
            }
        })
        .await
        .expect("mock MCP server timed out");
    });
    let session = McpSession::new(
        McpTransportConfig::Remote(McpRemoteServerConfig {
            url,
            headers: IndexMap::new(),
            oauth: false,
        }),
        5_000,
    );
    let result = list_tools("test", &session).await;
    server.await.unwrap();
    let tools = result.expect("discovery should negotiate the protocol version");
    assert_eq!(tools.len(), 1);
    assert_eq!(tools[0].name, "probe");
    session.shutdown().await;
}

#[tokio::test]
#[ignore = "requires an explicitly supplied MCP_TEST_URL; only discovers tools"]
async fn remote_http_live_tool_discovery() {
    let url = std::env::var("MCP_TEST_URL").expect("set MCP_TEST_URL to the test server");
    let (session, tools) = discover_server_tools(
        "live-test".into(),
        McpServerConfig {
            enabled: true,
            timeout_ms: 10_000,
            transport: McpTransportConfig::Remote(McpRemoteServerConfig {
                url,
                headers: IndexMap::new(),
                oauth: false,
            }),
        },
    )
    .await
    .expect("live MCP discovery should succeed");
    assert!(!tools.is_empty());
    println!("Discovered {} tools", tools.len());
    session.shutdown().await;
}
