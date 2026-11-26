use super::*;

fn test_model_config(protocol: crate::config::ApiProtocol) -> crate::config::ModelConfig {
    crate::config::ModelConfig {
        display_name: None,
        anthropic_thinking: Default::default(),
        anthropic_betas: Vec::new(),
        protocol,
        context_window: None,
        effective_input_limit_tokens: None,
        max_output_tokens: None,
        supports_tools: false,
        supports_input_images: false,
        supports_tool_result_images: false,
        supports_reasoning: false,
        reasoning_effort: None,
        reasoning_efforts: Vec::new(),
        reasoning_summary: None,
        text_verbosity: None,
        temperature: None,
        top_p: None,
        prompt_cache: crate::config::PromptCacheConfig::default(),
        parallel_tool_calls: false,
    }
}

fn test_provider(model: &str) -> crate::config::ProviderConfig {
    let protocol = crate::config::ApiProtocol::Completions;
    crate::config::ProviderConfig {
        base_url: "http://127.0.0.1:9/v1".into(),
        jev_endpoint: None,
        auth_mode: crate::config::ProviderAuthMode::ApiKey,
        api_key: "reviewer-key".into(),
        protocol,
        default_model: model.into(),
        retry: None,
        reviewer: None,
        models: indexmap::IndexMap::from([(model.into(), test_model_config(protocol))]),
    }
}

fn permission_request() -> PermissionRequest {
    PermissionRequest {
        call_id: Some("call-review".into()),
        tool: "shell__exec".into(),
        args: serde_json::json!({"command": "pwd"}),
        class: crate::permission::ToolPermissionClass::Command,
        summary: "Run a read-only command".into(),
        preview: Some("pwd".into()),
        can_allow_always: false,
        grant_summary: None,
    }
}

fn completed_summary(summary: &str) -> SubagentRunSummary {
    SubagentRunSummary {
        run_id: "r".into(),
        child_session_id: "c".into(),
        agent_name: "reviewer".into(),
        status: SubagentStatus::Completed,
        failure_kind: None,
        summary: summary.into(),
        structured_result: crate::subagent::StructuredSubagentResult {
            status: "completed".into(),
            summary: summary.into(),
            malformed: false,
            findings: Vec::new(),
            files_read: Vec::new(),
            files_changed: Vec::new(),
            commands_run: Vec::new(),
            validation: Vec::new(),
            blockers: Vec::new(),
            next_steps: Vec::new(),
            run_id: "r".into(),
            child_session_id: "c".into(),
            raw_excerpt: None,
        },
    }
}

#[test]
fn expert_review_prompt_carries_the_requesters_last_visible_message() {
    let mut parent = Agent::new("parent-model", 1, 1);
    parent.set_history_for_test(vec![
        crate::protocol_frames::ProtocolItem::user("run the test suite"),
        crate::protocol_frames::ProtocolItem::assistant("I will run the test suite next"),
    ]);

    let prompt = expert_review_prompt(&parent, &permission_request(), Some("run the test suite"));

    assert!(
        prompt.contains("Requester's most recent visible message:\nI will run the test suite next"),
        "{prompt}"
    );
}

#[test]
fn expert_review_prompt_omits_narration_when_the_requester_has_none() {
    let parent = Agent::new("parent-model", 1, 1);

    let prompt = expert_review_prompt(&parent, &permission_request(), None);

    assert!(!prompt.contains("most recent visible message"), "{prompt}");
}

#[test]
fn unparseable_output_fails_the_review() {
    let error = parse_reviewer_output(&completed_summary("I allow this"), None, true)
        .err()
        .expect("unparseable output must fail the review");
    assert!(
        error
            .to_string()
            .starts_with("auto-review returned unparseable output")
    );
}

#[test]
fn allow_always_is_downgraded_when_request_cannot_create_a_grant() {
    let parsed = parse_reviewer_output(
        &completed_summary(r#"{"decision":"allow_always","risk":"low","rationale":"safe"}"#),
        None,
        false,
    )
    .expect("parsed");
    assert_eq!(parsed.outcome, AutoReviewOutcome::AllowOnce);
}

#[test]
fn reviewer_decisions_map_to_the_three_outcomes() {
    let cases = [
        (
            r#"{"decision":"execute","risk":"low","rationale":"read only"}"#,
            AutoReviewOutcome::AllowOnce,
        ),
        (
            r#"{"decision":"ask_user","risk":"medium","rationale":"deletes files"}"#,
            AutoReviewOutcome::Ask,
        ),
        (
            r#"{"decision":"refuse","risk":"high","rationale":"contradicts the goal"}"#,
            AutoReviewOutcome::Deny,
        ),
    ];
    for (raw, expected) in cases {
        let parsed = parse_reviewer_output(&completed_summary(raw), None, false).expect("parsed");
        assert_eq!(parsed.outcome, expected, "{raw}");
    }
}

#[test]
fn full_reviewer_output_is_parsed_before_truncation_fallbacks() {
    let rationale = "safe ".repeat(80);
    let full_output =
        format!(r#"{{"decision":"allow_once","risk":"low","rationale":"{rationale}"}}"#);
    assert!(full_output.chars().count() > 240);

    let parsed = parse_reviewer_output(
        &completed_summary("truncated reviewer output"),
        Some(&full_output),
        true,
    )
    .expect("parsed");

    assert_eq!(parsed.outcome, AutoReviewOutcome::AllowOnce);
    assert_eq!(parsed.rationale, rationale);
}

static REVIEW_REDACT_TEST_LOCK: Mutex<()> = Mutex::new(());

struct ReviewRedactGlobals {
    active: Option<(bool, std::path::PathBuf, Arc<crate::redact::Redactor>)>,
}

impl Drop for ReviewRedactGlobals {
    fn drop(&mut self) {
        crate::redact::restore_active(self.active.clone());
    }
}

#[test]
#[ignore = "requires LETCODE_PII_MODEL_DIR pointing to the local PII model"]
fn review_prompt_is_masked_once_for_journal_and_oneshot_input() {
    let model_dir =
        std::env::var("LETCODE_PII_MODEL_DIR").expect("LETCODE_PII_MODEL_DIR is required");
    let _lock = REVIEW_REDACT_TEST_LOCK
        .lock()
        .unwrap_or_else(|error| error.into_inner());
    let config = tempfile::tempdir().expect("config dir");
    let model_target = config.path().join("models").join("astrlink-guard");
    std::fs::create_dir_all(&model_target).expect("model dir is created");
    for name in ["model_int8.onnx", "model_int8.onnx_data", "tokenizer.json"] {
        std::fs::copy(
            std::path::Path::new(&model_dir).join(name),
            model_target.join(name),
        )
        .expect("the model is copied");
    }
    let sessions = config.path().join("sessions");
    std::fs::create_dir_all(&sessions).expect("sessions dir is created");
    let _globals = ReviewRedactGlobals {
        active: crate::redact::active_snapshot(),
    };
    crate::redact::install(true, config.path(), &sessions);

    let transcript = Arc::new(Mutex::new(
        TranscriptRecorder::create(&sessions).expect("transcript"),
    ));
    let session_id = transcript
        .lock()
        .expect("transcript")
        .session_id()
        .to_string();
    let restore = crate::transcript::transcript_projection::project_runtime_restore_snapshot(
        session_id,
        read_records_allow_partial_tail(transcript.lock().expect("transcript").path())
            .expect("transcript records"),
        crate::transcript::transcript_projection::SessionContextCursor {
            branch_id: None,
            leaf_sequence: None,
        },
        &[],
    )
    .expect("the child session restore snapshot is projected");
    let mut agent = Agent::new("reviewer-model", 1, 1);
    agent
        .restore_runtime_snapshot(restore.snapshot)
        .expect("the agent is bound to the child session");
    let raw = "Contact person@example.com before running the command";

    let prepared = tokio::runtime::Runtime::new()
        .expect("runtime")
        .block_on(prepare_review_prompt(
            &mut agent,
            raw.to_string(),
            &transcript,
        ))
        .expect("the review prompt is prepared");

    assert!(prepared.contains("{|>EMAIL:"), "{prepared}");
    assert!(!prepared.contains("person@example.com"), "{prepared}");
    let records = read_records_allow_partial_tail(transcript.lock().expect("transcript").path())
        .expect("transcript records");
    let journal_prompt = records
        .iter()
        .rev()
        .find_map(|record| match &record.event {
            TranscriptEvent::UserMessage { content } => Some(content.text.clone()),
            _ => None,
        })
        .expect("the prompt is journaled");
    assert_eq!(journal_prompt, prepared);
    assert!(records.iter().any(|record| matches!(
        &record.event,
        TranscriptEvent::SessionRedaction { enabled: true }
    )));
}

mod integration_tests {
    use super::*;

    #[tokio::test]
    async fn missing_reviewer_route_credential_fails_before_starting_a_child() {
        let dir = std::env::temp_dir().join(format!(
            "letcode-auto-review-credential-{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("clock")
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).expect("sessions dir");
        let transcript = Arc::new(Mutex::new(
            TranscriptRecorder::create(&dir).expect("transcript"),
        ));
        let route = crate::config::ModelRoute::new("reviewer-provider", "reviewer-model");
        let providers =
            indexmap::IndexMap::from([(route.provider.clone(), test_provider(&route.model))]);
        let factory = Arc::new(
            crate::subagent::ExpertRouteFactory::new_with_policies(
                [(
                    REVIEWER_AGENT_NAME.into(),
                    Some(route.clone()),
                    Vec::new(),
                    None,
                )],
                &providers,
                &crate::config::RetryConfig::default(),
            )
            .expect("reviewer route factory"),
        );
        let mut parent = Agent::new("parent-model", 1, 1);
        parent.set_primary_route(crate::config::ModelRoute::new("primary", "parent-model"));
        parent.set_subagent_child_factory(factory);
        let reviewer = StickyAutoReviewer::new(
            SubagentPool::new(),
            dir.clone(),
            Arc::clone(&transcript),
            None,
            Arc::new(Mutex::new(indexmap::IndexMap::from([(
                route.display_name(),
                false,
            )]))),
            Arc::new(Mutex::new(indexmap::IndexMap::from([(
                route.provider.clone(),
                "Set REVIEWER_API_KEY".into(),
            )]))),
            "Set the provider API key".into(),
        );

        let error = reviewer
            .review(&parent, permission_request(), None)
            .await
            .expect_err("a missing reviewer credential must fail the review");

        assert!(
            error
                .to_string()
                .contains("reviewer-provider/reviewer-model")
        );
        assert!(error.to_string().contains("Set REVIEWER_API_KEY"));
        assert!(reviewer.sticky_child_id().is_empty());
        assert!(
            !crate::transcript::child_sessions_dir(&dir).exists(),
            "credential preflight must happen before child transcript creation"
        );
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn clearing_sticky_session_allows_new_reviewer_route_after_policy_change() {
        let dir = std::env::temp_dir().join(format!(
            "letcode-auto-review-route-change-{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("clock")
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).expect("sessions dir");
        let transcript = Arc::new(Mutex::new(
            TranscriptRecorder::create(&dir).expect("transcript"),
        ));
        let reviewer = StickyAutoReviewer::new(
            SubagentPool::new(),
            dir.clone(),
            Arc::clone(&transcript),
            None,
            Arc::new(Mutex::new(indexmap::IndexMap::new())),
            Arc::new(Mutex::new(indexmap::IndexMap::new())),
            "Set the provider API key".into(),
        );
        let child_dir = crate::transcript::child_sessions_dir(&dir);
        let mut child = TranscriptRecorder::create(&child_dir).expect("child transcript");
        let child_session_id = child.session_id().to_string();
        child
            .record_session_started("old-provider/reviewer-model")
            .expect("record child route");
        reviewer
            .inner
            .lock()
            .expect("reviewer state")
            .child_session_id = Some(child_session_id.clone());

        let new_route = crate::config::ModelRoute::new("new-provider", "reviewer-model");
        let providers = indexmap::IndexMap::from([(
            new_route.provider.clone(),
            test_provider(&new_route.model),
        )]);
        let factory = Arc::new(
            crate::subagent::ExpertRouteFactory::new_with_policies(
                [(
                    REVIEWER_AGENT_NAME.into(),
                    Some(new_route.clone()),
                    Vec::new(),
                    None,
                )],
                &providers,
                &crate::config::RetryConfig::default(),
            )
            .expect("reviewer route factory"),
        );
        let mut parent = Agent::new("parent-model", 1, 1);
        parent.set_primary_route(crate::config::ModelRoute::new("primary", "parent-model"));
        parent.set_subagent_child_factory(factory);

        let stale =
            reviewer.resolve_route(&parent, &AgentTemplate::reviewer(), Some(&child_session_id));
        assert!(
            stale
                .expect_err("old reviewer route must fail current policy")
                .to_string()
                .contains("cannot resume: recorded route 'old-provider/reviewer-model'")
        );

        reviewer.clear_sticky();
        assert_eq!(
            reviewer
                .resolve_route(&parent, &AgentTemplate::reviewer(), None)
                .expect("cleared reviewer should resolve the new default route"),
            new_route
        );
        let _ = std::fs::remove_dir_all(dir);
    }
}
