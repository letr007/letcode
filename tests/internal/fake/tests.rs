use super::*;

fn context() -> CodexRequestContext {
    FakeIdentity::new("installation").turn_context(&FakeConfig::default(), None)
}

#[test]
fn fake_client_parse_supports_all_modes() {
    assert_eq!(FakeClient::parse("auto"), Some(FakeClient::Auto));
    assert_eq!(FakeClient::parse("codex"), Some(FakeClient::Codex));
    assert_eq!(FakeClient::parse("anthropic"), Some(FakeClient::Anthropic));
    assert_eq!(FakeClient::parse("other"), None);
}

#[test]
fn fake_modes_resolve_per_route_protocol() {
    let responses = crate::model_runtime::ProtocolId::new("responses").unwrap();
    let anthropic = crate::model_runtime::ProtocolId::new("anthropic").unwrap();
    let completions = crate::model_runtime::ProtocolId::new("completions").unwrap();

    assert_eq!(
        FakeClient::Auto.resolve(&responses),
        Some(FakeProfile::Codex)
    );
    assert_eq!(
        FakeClient::Auto.resolve(&anthropic),
        Some(FakeProfile::Anthropic)
    );
    assert_eq!(FakeClient::Auto.resolve(&completions), None);
    assert_eq!(
        FakeClient::Codex.resolve(&responses),
        Some(FakeProfile::Codex)
    );
    assert_eq!(FakeClient::Codex.resolve(&anthropic), None);
    assert_eq!(
        FakeClient::Anthropic.resolve(&anthropic),
        Some(FakeProfile::Anthropic)
    );
    assert_eq!(FakeClient::Anthropic.resolve(&responses), None);
    assert_eq!(FakeClient::Codex.resolve(&completions), None);
}

#[test]
fn synthetic_ids_match_the_shape_the_imitated_client_sends() {
    let session = synthetic_uuid();
    assert_eq!(session.as_bytes()[14], b'7', "session ids are version 7");
    let variant = session.as_bytes()[19] as char;
    assert!(
        ('8'..='b').contains(&variant),
        "variant bits are 10, got {variant}"
    );
    // The prefix is the current millisecond, so it must not read as zeros.
    assert!(!session.starts_with("0000"), "{session}");
    assert_ne!(session, synthetic_uuid());

    let installation = synthetic_installation_id();
    assert_eq!(
        installation.as_bytes()[14],
        b'4',
        "installation ids are version 4"
    );
    let variant = installation.as_bytes()[19] as char;
    assert!(('8'..='b').contains(&variant), "got {variant}");
    // Random, not a fixed value: two calls must differ.
    assert_ne!(installation, synthetic_installation_id());
}

#[test]
fn top_level_turns_carry_no_agent_name_unless_declared() {
    let identity = FakeIdentity::new(synthetic_installation_id());
    let derived = identity.turn_context(&crate::config::FakeConfig::default(), None);
    assert!(
        !derived
            .turn_metadata_json()
            .to_string()
            .contains("agent_name")
    );

    let mut declared = crate::config::FakeConfig::default();
    declared.identity.agent_name = Some("Hypatia".into());
    let named = identity.turn_context(&declared, None);
    assert_eq!(named.turn_metadata_json()["agent_name"], "Hypatia");
}

#[test]
fn codex_identity_uses_stable_session_ids_within_a_context() {
    let identity = FakeIdentity::new("installation");
    let context = identity.turn_context(&FakeConfig::default(), None);

    assert_eq!(context.session_id, context.thread_id);
    assert_eq!(context.window_id(), format!("{}:0", context.session_id));
    assert_eq!(context.installation_id, "installation");
}

#[test]
fn declared_values_win_over_derived_defaults() {
    let mut config = FakeConfig::default();
    config.codex.version = Some("9.9.9".into());
    config.codex.originator = Some("codex_cli_rs".into());
    config.codex.os = Some("Plan9".into());
    config.codex.arch = Some("vax".into());
    config.codex.terminal = Some("vt100/1".into());
    config.codex.beta_features = Some(vec!["one".into(), "two".into()]);
    config.clock.timezone = Some("Asia/Shanghai".into());
    config.clock.date = Some("1999-01-01".into());
    config.identity.agent_name = Some("unit-test".into());
    config.codex.sandbox = Some("seatbelt".into());
    config.codex.sandbox_mode = Some("read-only".into());
    config.codex.auto_review_enabled = Some(true);
    config.codex.node_repl_disabled = Some(true);
    config.codex.cwd = Some("/declared/cwd".into());
    config.codex.workspace = Some("/declared/ws".into());
    config.codex.shell = Some("fish".into());
    config.codex.git_commit_hash = Some("deadbeef".into());
    config.codex.git_remote_url = Some("https://example.test/repo.git".into());
    config.codex.git_has_changes = Some(true);
    config
        .codex
        .extra
        .insert("custom_key".into(), "custom".into());

    let context = FakeIdentity::new("installation").turn_context(&config, None);
    let headers = context.headers();
    let user_agent = headers
        .iter()
        .find(|(name, _)| name == "user-agent")
        .map(|(_, value)| value.as_str())
        .expect("user agent header");
    assert_eq!(user_agent, "codex_cli_rs/9.9.9 (Plan9; vax) vt100/1");
    assert_eq!(context.current_date.as_deref(), Some("1999-01-01"));
    assert_eq!(context.timezone, "Asia/Shanghai");

    let metadata = context.turn_metadata_json();
    assert_eq!(metadata["agent_name"], "unit-test");
    assert_eq!(metadata["sandbox"], "seatbelt");
    assert_eq!(metadata["sandbox_mode"], "read-only");
    assert_eq!(metadata["auto_review_enabled"], true);
    assert_eq!(metadata["node_repl_disabled"], true);
    assert_eq!(metadata["custom_key"], "custom");
    let workspace = &metadata["workspaces"]["/declared/ws"];
    assert_eq!(workspace["latest_git_commit_hash"], "deadbeef");
    assert_eq!(workspace["has_changes"], true);
    assert_eq!(
        workspace["associated_remote_urls"]["origin"],
        "https://example.test/repo.git"
    );

    let block = context.environment_context_text();
    assert!(block.starts_with("<environment_context>\n"));
    assert!(block.contains("  <cwd>/declared/cwd</cwd>\n"));
    assert!(block.contains("  <shell>fish</shell>\n"));
    assert!(block.contains("  <current_date>1999-01-01</current_date>\n"));
    assert!(block.contains("  <timezone>Asia/Shanghai</timezone>\n"));
    assert!(block.ends_with("</environment_context>"));
}

#[test]
fn environment_context_escapes_xml_text() {
    let mut config = FakeConfig::default();
    config.codex.cwd = Some("/a<b>&\"c\"".into());
    let context = FakeIdentity::new("installation").turn_context(&config, None);
    let block = context.environment_context_text();

    assert!(block.contains("/a&lt;b&gt;&amp;&quot;c&quot;"));
    assert!(!block.contains("/a<b>"));
}

#[test]
fn response_shape_preserves_prompt_and_tools() {
    let mut request = serde_json::json!({
        "model": "gpt-5.6-sol",
        "instructions": "core and workspace instructions",
        "input": [{"type": "message"}],
        "tools": [{"name": "fs__read"}],
        "reasoning": {"effort": "minimal", "summary": "concise"},
        "temperature": 0.2,
        "service_tier": "priority"
    });
    let context = context();
    apply_codex_response_shape(&mut request, &context);

    assert_eq!(request["model"], "gpt-5.6-sol");
    assert_eq!(request["instructions"], "core and workspace instructions");
    assert_eq!(request["parallel_tool_calls"], false);
    assert_eq!(request["tool_choice"], "auto");
    assert_eq!(request["store"], false);
    assert_eq!(request["reasoning"]["effort"], "minimal");
    assert_eq!(request["reasoning"]["summary"], "concise");
    assert_eq!(request["reasoning"]["context"], "all_turns");
    assert_eq!(request["prompt_cache_key"], context.session_id);
    assert!(request.get("temperature").is_none());
    assert!(request.get("service_tier").is_none());
    assert!(
        request["client_metadata"]["x-codex-installation-id"]
            .as_str()
            .is_some_and(|value| value == "installation")
    );
}

#[test]
fn runtime_context_item_is_reroled_without_changing_the_message_count() {
    let runtime_text = context().environment_context_text();
    let mut request = serde_json::json!({
        "model": "gpt",
        "instructions": "system",
        "input": [
            {"type": "message", "role": "developer", "content": [{"type": "input_text", "text": "skill prelude"}]},
            {"type": "message", "role": "developer", "content": [{"type": "input_text", "text": runtime_text}]},
            {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "hi"}]}
        ],
        "tools": []
    });
    let before = request["input"].as_array().unwrap().len();
    apply_codex_response_shape(&mut request, &context());

    let items = request["input"].as_array().unwrap();
    assert_eq!(items.len(), before, "message count must not change");
    assert_eq!(
        items[0]["role"], "developer",
        "other developer messages stay"
    );
    assert_eq!(
        items[1]["role"], "user",
        "runtime context becomes a user block"
    );
    assert_eq!(items[2]["role"], "user");
}

#[test]
fn unrelated_developer_messages_keep_their_role() {
    let mut request = serde_json::json!({
        "model": "gpt",
        "input": [
            {"type": "message", "role": "developer", "content": [{"type": "input_text", "text": "运行时上下文：\n- 当前日期：2026-09-11"}]}
        ]
    });
    apply_codex_response_shape(&mut request, &context());

    assert_eq!(request["input"][0]["role"], "developer");
}

#[test]
fn headers_expose_no_real_local_paths_unless_declared() {
    let mut config = FakeConfig::default();
    config.codex.workspace = Some("/workspace".into());
    config.codex.cwd = Some("/workspace".into());
    let context = FakeIdentity::new("installation").turn_context(&config, None);
    let headers = context.headers();
    let metadata = headers
        .iter()
        .find(|(name, _)| name == "x-codex-turn-metadata")
        .map(|(_, value)| value)
        .expect("turn metadata header");

    assert!(metadata.contains("/workspace"));
    assert!(!metadata.contains(r"/Users/"));
}

#[test]
fn claude_headers_report_the_client_profile() {
    let context = FakeIdentity::new("installation")
        .claude_turn_context(&FakeConfig::default(), PermissionMode::Auto);
    let headers = context.headers();
    let value = |name: &str| {
        headers
            .iter()
            .find(|(header, _)| header == name)
            .map(|(_, value)| value.as_str())
    };

    assert_eq!(value("accept"), Some("application/json"));
    assert_eq!(value("x-app"), Some("cli"));
    assert_eq!(value("x-stainless-lang"), Some("js"));
    assert_eq!(value("x-stainless-runtime"), Some("node"));
    assert_eq!(value("x-stainless-retry-count"), Some("0"));
    assert_eq!(
        value("anthropic-dangerous-direct-browser-access"),
        Some("true")
    );
    assert_eq!(
        value("user-agent"),
        Some(
            format!(
                "claude-cli/{} (external, cli)",
                super::claude_defaults::VERSION
            )
            .as_str()
        )
    );
    assert_eq!(
        value("x-claude-code-session-id"),
        Some(context.session_id.as_str())
    );
    assert!(
        value("anthropic-beta")
            .expect("beta header")
            .starts_with("claude-code-")
    );
    assert!(value("originator").is_none());
    assert!(value("x-codex-turn-metadata").is_none());
    // The client reads this id from responses; it never sends one.
    assert!(value("x-client-request-id").is_none());
}

#[test]
fn claude_client_blocks_name_the_reported_client() {
    let context = FakeIdentity::new("installation")
        .claude_turn_context(&FakeConfig::default(), PermissionMode::Auto);
    assert_eq!(
        context.client_system_blocks(),
        vec![
            format!(
                "x-anthropic-billing-header: cc_version={}.{}; cc_entrypoint={};",
                super::claude_defaults::VERSION,
                super::claude_defaults::BUILD,
                super::claude_defaults::ENTRYPOINT
            ),
            "You are a Claude agent, built on Anthropic's Claude Agent SDK.".to_string(),
        ]
    );

    let mut declared = FakeConfig::default();
    declared.claude.version = Some("9.9.9".into());
    declared.claude.build = Some("abc".into());
    let declared =
        FakeIdentity::new("installation").claude_turn_context(&declared, PermissionMode::Auto);
    assert!(declared.client_system_blocks()[0].contains("cc_version=9.9.9.abc;"));
}

#[test]
fn claude_body_reports_the_session_as_metadata_user_id() {
    let identity = FakeIdentity::new("installation");
    let context = identity.claude_turn_context(&FakeConfig::default(), PermissionMode::Auto);
    let mut request = serde_json::json!({
        "model": "claude",
        "messages": [{"role": "user", "content": "hi"}],
        "stream": true
    });
    apply_claude_body_shape(&mut request, &context);

    let user_id = request["metadata"]["user_id"].as_str().expect("user id");
    let decoded: Value = serde_json::from_str(user_id).expect("json blob");
    assert_eq!(decoded["device_id"], context.device_id.as_str());
    assert_eq!(decoded["session_id"], context.session_id.as_str());
    // The API-key client reports no account.
    assert_eq!(decoded["account_uuid"], "");
    assert_eq!(request["messages"][0]["content"], "hi");
    assert_eq!(
        request["context_management"],
        serde_json::json!({"edits": [{"type": "clear_thinking_20251015", "keep": "all"}]})
    );

    let mut declared = FakeConfig::default();
    declared.identity.account_uuid = Some("declared-account".into());
    let declared = identity.claude_turn_context(&declared, PermissionMode::Auto);
    assert_eq!(declared.account_uuid, "declared-account");
}

#[test]
fn claude_context_management_follows_its_beta() {
    let identity = FakeIdentity::new("installation");
    let mut declared = FakeConfig::default();
    declared.claude.betas = Some(vec!["claude-code-20250219".into()]);
    let context = identity.claude_turn_context(&declared, PermissionMode::Auto);
    let mut request = serde_json::json!({"model": "claude"});
    apply_claude_body_shape(&mut request, &context);
    assert!(request.get("context_management").is_none());
}

#[test]
fn claude_auto_mode_betas_follow_the_permission_posture() {
    let identity = FakeIdentity::new("installation");
    let active = identity.claude_turn_context(&FakeConfig::default(), PermissionMode::Auto);
    assert!(
        active
            .betas
            .iter()
            .any(|beta| beta == "dangerous-tool-use-2026-09-03")
    );

    let bypassed = identity.claude_turn_context(&FakeConfig::default(), PermissionMode::Yolo);
    assert!(
        !bypassed
            .betas
            .iter()
            .any(|beta| super::claude_defaults::AUTO_MODE_BETAS.contains(&beta.as_str()))
    );

    // A declared list is explicit, so it is reported verbatim.
    let mut declared = FakeConfig::default();
    declared.claude.betas = Some(vec!["afk-mode-2026-01-31".into()]);
    let declared = identity.claude_turn_context(&declared, PermissionMode::Yolo);
    assert_eq!(declared.betas, vec!["afk-mode-2026-01-31".to_string()]);
}

#[test]
fn claude_device_id_is_a_stable_digest_of_the_installation() {
    let identity = FakeIdentity::new("installation");
    let first = identity
        .claude_turn_context(&FakeConfig::default(), PermissionMode::Auto)
        .device_id;
    let second = identity
        .claude_turn_context(&FakeConfig::default(), PermissionMode::Auto)
        .device_id;
    assert_eq!(first, second);
    assert_eq!(first.len(), 64, "device ids are sha256 hex: {first}");
    assert!(first.chars().all(|digit| digit.is_ascii_hexdigit()));
    assert_ne!(
        first,
        FakeIdentity::new("other-installation")
            .claude_turn_context(&FakeConfig::default(), PermissionMode::Auto)
            .device_id
    );
}
