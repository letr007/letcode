use super::{NormalizedSubagentInput, SubagentPathScope, normalize_subagent_input};
use serde_json::json;

#[test]
fn subagent_path_scope_from_input_empty_or_unresolvable() {
    let empty = NormalizedSubagentInput {
        objective: "x".into(),
        success_criteria: Vec::new(),
        allowed_paths: Vec::new(),
        forbidden_paths: Vec::new(),
        owned_paths: Vec::new(),
        timeout_secs: None,
        max_tool_calls: None,
        model: None,
        target_child_session_id: None,
        background: false,
    };
    assert!(
        SubagentPathScope::from_input(&empty)
            .expect("empty scope")
            .is_none()
    );

    let bad = NormalizedSubagentInput {
        owned_paths: vec!["/definitely/missing/letcode-scope-root".into()],
        ..empty
    };
    assert!(SubagentPathScope::from_input(&bad).is_err());
}

#[test]
fn normalize_subagent_input_rejects_missing_or_blank_objective() {
    let missing = normalize_subagent_input("agent__explore", &json!({}))
        .expect_err("missing objective should fail");
    assert!(
        missing
            .to_string()
            .contains("requires a non-empty 'task' or 'objective'")
    );

    let blank =
        normalize_subagent_input("agent__fixer", &json!({"task": "   ", "objective": null}))
            .expect_err("blank task should fail");
    assert!(
        blank
            .to_string()
            .contains("field 'task' must not be empty or whitespace")
    );
}

#[test]
fn writable_subagents_require_owned_paths_for_file_locking() {
    for tool in ["agent__fixer", "agent__general"] {
        for args in [
            json!({"task": "implement"}),
            json!({"task": "implement", "owned_paths": null}),
            json!({"task": "implement", "owned_paths": []}),
        ] {
            let missing = normalize_subagent_input(tool, &args)
                .expect_err("writable subagent without locks must fail");
            assert!(
                missing
                    .to_string()
                    .contains(&format!("{tool} requires non-empty owned_paths"))
            );
        }
        let input =
            normalize_subagent_input(tool, &json!({"task": "implement", "owned_paths": ["src"]}))
                .expect("writable subagent with owned paths normalizes");
        assert_eq!(input.owned_paths, vec!["src"]);
    }
}

#[test]
fn normalize_subagent_input_accepts_model_and_rejects_takeover_conflict() {
    let input = normalize_subagent_input(
        "agent__explore",
        &json!({"objective": "inspect", "model": "expert/special"}),
    )
    .expect("model override normalizes");
    assert_eq!(input.model.as_deref(), Some("expert/special"));
    assert!(!input.background);

    let background = normalize_subagent_input(
        "agent__explore",
        &json!({"objective": "inspect", "background": true}),
    )
    .expect("background flag normalizes");
    assert!(background.background);

    let invalid_background = normalize_subagent_input(
        "agent__explore",
        &json!({"objective": "inspect", "background": "yes"}),
    )
    .expect_err("non-boolean background should fail");
    assert!(
        invalid_background
            .to_string()
            .contains("field 'background' must be a boolean or null")
    );

    let error = normalize_subagent_input(
        "agent__explore",
        &json!({
            "objective": "inspect",
            "model": "expert/special",
            "target_child_session_id": "child-1"
        }),
    )
    .expect_err("override and takeover conflict");
    assert!(
        error
            .to_string()
            .contains("field 'model' cannot be combined with 'target_child_session_id'")
    );
}

#[test]
fn normalized_subagent_input_enforces_write_scope_paths() {
    for tool in ["agent__fixer", "agent__general"] {
        let input = normalize_subagent_input(
            tool,
            &json!({
                "objective": "fix",
                "allowed_paths": ["src"],
                "owned_paths": ["src/fixes"],
                "forbidden_paths": ["src/secrets"]
            }),
        )
        .expect("scope should normalize");

        assert!(input.has_write_scope());
        assert!(input.permits_write_path("src/fixes/mod.rs"));
        assert!(input.permits_write_path("src/lib.rs"));
        assert!(!input.permits_write_path("src/secrets/token.rs"));
        assert!(!input.permits_write_path("tests/outside.rs"));
    }
}
