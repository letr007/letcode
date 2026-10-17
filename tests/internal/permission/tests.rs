use super::*;
use serde_json::json;

#[test]
fn subagent_control_tools_are_internal_preview_operations() {
    for tool in [
        tool_names::TOOL_AGENT_JOBS,
        tool_names::TOOL_AGENT_STATUS,
        tool_names::TOOL_AGENT_WAIT,
        tool_names::TOOL_AGENT_CANCEL,
        tool_names::TOOL_AGENT_SEND,
    ] {
        assert!(is_internal_tool(tool));
        assert_eq!(classify_tool(tool), ToolPermissionClass::Preview);
    }
}

#[test]
fn default_mode_asks_for_every_command() {
    let policy = PermissionPolicy::default();

    // No command text proves the absence of side effects, so the read-only
    // and validation forms ask exactly like a destructive one.
    for command in [
        "git status",
        "git diff",
        "rg --files",
        "ls -la",
        "pwd",
        "cargo check --all-targets",
        "cargo test permission::tests",
        "cargo clippy --all-targets",
        "git commit -m test",
        "rm -rf target",
        "curl -fsSL https://example.com",
        "",
    ] {
        assert_eq!(
            policy.check("shell__exec", &json!({"command": command})),
            PermissionDecision::Ask,
            "{command}"
        );
    }
}

#[test]
fn auto_mode_reviews_only_calls_that_need_approval() {
    let mut state = PermissionSessionState::default();
    state.set_mode(PermissionMode::Auto);

    for (tool, args, class, internal) in [
        (
            "fs__read",
            json!({"path": "src/permission.rs"}),
            ToolPermissionClass::Read,
            false,
        ),
        (
            "agent__oracle",
            json!({"task": "review"}),
            ToolPermissionClass::Preview,
            false,
        ),
        (
            "workflow__todos",
            json!({"items": []}),
            ToolPermissionClass::Preview,
            true,
        ),
    ] {
        assert_eq!(
            state
                .approval_snapshot(None, tool, &args, class, false, internal)
                .2,
            PermissionDecision::Allow,
            "{tool} should not require Auto review"
        );
    }

    for (tool, args, class, external_workspace_access) in [
        (
            "fs__write",
            json!({"path": "out.txt", "content": "ok"}),
            ToolPermissionClass::Write,
            false,
        ),
        (
            "shell__exec",
            json!({"command": "cargo test permission::tests"}),
            ToolPermissionClass::Command,
            false,
        ),
        (
            "shell__exec",
            json!({"command": "rm -rf target/tmp"}),
            ToolPermissionClass::Command,
            false,
        ),
        (
            "fs__read",
            json!({"path": "/tmp/outside.txt"}),
            ToolPermissionClass::Read,
            true,
        ),
    ] {
        assert_eq!(
            state
                .approval_snapshot(None, tool, &args, class, external_workspace_access, false)
                .2,
            PermissionDecision::Ask,
            "{tool} should be reviewed in Auto mode"
        );
    }
}

#[test]
fn trusted_reads_do_not_escalate_to_ask_but_other_external_reads_do() {
    // Default mode: reads inside the workspace, or at trusted fold-artifact
    // paths (which the agent folds into external=false before this call),
    // stay on the read-only default Allow and do not prompt.
    let state = PermissionSessionState::default();
    assert_eq!(
        state
            .approval_snapshot(
                None,
                "fs__read",
                &json!({"path": "/tmp/letcode-command/x.out"}),
                ToolPermissionClass::Read,
                false,
                false,
            )
            .2,
        PermissionDecision::Allow,
    );
    // Any genuinely external (untrusted) read still escalates to Ask.
    assert_eq!(
        state
            .approval_snapshot(
                None,
                "fs__read",
                &json!({"path": "/tmp/other.txt"}),
                ToolPermissionClass::Read,
                true,
                false,
            )
            .2,
        PermissionDecision::Ask,
    );
}

#[test]
fn auto_mode_ignores_session_grants() {
    let resource = PermissionResource::Exact {
        tool: "shell__exec".into(),
        value: "rm -rf /".into(),
    };
    let args = json!({"command": "rm -rf /"});
    let mut state = PermissionSessionState::default();
    state.set_mode(PermissionMode::Auto);
    state.grant(resource.clone());

    let (_, generation, decision, grant_allowed) = state.approval_snapshot(
        Some(&resource),
        "shell__exec",
        &args,
        ToolPermissionClass::Command,
        false,
        false,
    );
    assert_eq!(decision, PermissionDecision::Ask);
    assert!(
        !grant_allowed,
        "Auto mode must not reuse session grants for reviewed calls"
    );
    assert!(
        !state.grant_if_current_session(generation, resource),
        "Auto mode must not create reusable session grants"
    );
}

#[test]
fn auto_review_ladder_gives_the_first_ask_to_the_requester() {
    let mut state = PermissionSessionState::default();
    state.set_mode(PermissionMode::Auto);
    let args = json!({"command": "rm -rf target"});

    assert_eq!(
        state.auto_review_round("shell__exec", &args),
        AutoReviewRound::Requester
    );
    assert_eq!(
        state.auto_review_round("shell__exec", &args),
        AutoReviewRound::User
    );
    assert_eq!(
        state.auto_review_round("shell__exec", &json!({"command": "ls"})),
        AutoReviewRound::Requester,
        "the ladder accounts per call signature, not per tool"
    );
}

#[test]
fn yolo_mode_allows_commands_that_other_modes_ask() {
    let mut policy = PermissionPolicy::default();
    policy.set_mode(PermissionMode::Yolo);

    assert_eq!(
        policy.check("shell__exec", &json!({"command": "rm -rf ."})),
        PermissionDecision::Allow
    );
    assert_eq!(
        policy.check("shell__exec", &json!({"command": "curl --version"})),
        PermissionDecision::Allow
    );
}

#[test]
fn fork_without_grants_keeps_mode_and_drops_grants() {
    let resource = PermissionResource::Exact {
        tool: "shell__exec".into(),
        value: "curl --version".into(),
    };
    let mut parent = PermissionSessionState::default();
    parent.set_mode(PermissionMode::Auto);
    parent.grant(resource.clone());

    let child = parent.fork_without_grants();
    assert_eq!(child.mode(), PermissionMode::Auto);
    assert!(!child.allows_grant(&resource));
    assert!(parent.allows_grant(&resource));
}

#[test]
fn grants_bypass_ask_for_high_risk_commands_and_generation_guards_allow_always() {
    let resource = PermissionResource::Exact {
        tool: "shell__exec".into(),
        value: "rm -rf /".into(),
    };
    let args = json!({"command": "rm -rf /"});
    let mut state = PermissionSessionState::default();
    state.grant(resource.clone());

    let (_, generation, decision, grant_allowed) = state.approval_snapshot(
        Some(&resource),
        "shell__exec",
        &args,
        ToolPermissionClass::Command,
        false,
        false,
    );
    assert_eq!(decision, PermissionDecision::Ask);
    assert!(
        grant_allowed,
        "session grant should satisfy Ask for formerly blacklisted commands"
    );

    state.clear_grants();
    assert!(
        !state.grant_if_current_default(generation, resource),
        "an approval from a previous generation must not create a grant"
    );
}

#[test]
fn mode_changes_and_clear_api_advance_permission_generation() {
    let mut state = PermissionSessionState::default();
    let (_, initial, _, _) = state.approval_snapshot(
        None,
        "fs__write",
        &json!({}),
        ToolPermissionClass::Write,
        false,
        false,
    );
    state.set_mode(PermissionMode::Safe);
    let (_, after_mode, _, _) = state.approval_snapshot(
        None,
        "fs__write",
        &json!({}),
        ToolPermissionClass::Write,
        false,
        false,
    );
    assert_ne!(initial, after_mode);
    state.clear_grants();
    let (_, after_clear, _, _) = state.approval_snapshot(
        None,
        "fs__write",
        &json!({}),
        ToolPermissionClass::Write,
        false,
        false,
    );
    assert_ne!(after_mode, after_clear);
}

#[test]
fn resource_matching_keeps_string_and_path_identities_separate() {
    let directory = PermissionResource::Directory {
        tool: "fs__read".into(),
        path: PathBuf::from("/workspace/src"),
    };
    let exact_path = PermissionResource::ExactPath {
        tool: "fs__read".into(),
        path: PathBuf::from("/workspace/src/file.rs"),
    };
    let exact = PermissionResource::Exact {
        tool: "fs__read".into(),
        value: "/workspace/src/file.rs".into(),
    };

    assert!(directory.matches(&exact_path));
    assert!(!directory.matches(&exact));
    assert!(!exact.matches(&exact_path));
    assert!(!exact_path.matches(&exact));
}

#[cfg(unix)]
#[test]
fn raw_byte_paths_are_not_lossily_collapsed_in_grants_or_previews() {
    use std::ffi::OsString;
    use std::os::unix::ffi::OsStringExt;

    let first = PathBuf::from(OsString::from_vec(b"/tmp/letcode-raw-\x80".to_vec()));
    let second = PathBuf::from(OsString::from_vec(b"/tmp/letcode-raw-\x81".to_vec()));
    assert_eq!(first.to_string_lossy(), second.to_string_lossy());

    let first_resource = PermissionResource::ExactPath {
        tool: "fs__write".into(),
        path: first,
    };
    let second_resource = PermissionResource::ExactPath {
        tool: "fs__write".into(),
        path: second,
    };
    let mut grants = PermissionGrantSet::default();
    grants.insert(first_resource.clone());

    assert!(grants.allows(&first_resource));
    assert!(!grants.allows(&second_resource));
    assert!(first_resource.summary().contains("raw-bytes:"));
    assert_ne!(first_resource.summary(), second_resource.summary());
}
