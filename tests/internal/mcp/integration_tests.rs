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
