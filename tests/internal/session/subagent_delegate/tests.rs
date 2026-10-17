use super::*;

#[test]
fn capacity_result_reports_not_started_without_a_run_failure() {
    for tool in [
        crate::tool_names::TOOL_AGENT_EXPLORE,
        crate::tool_names::TOOL_AGENT_SEND,
    ] {
        let result = RunnerSubagentDelegate::capacity_result(
            tool,
            Some("explorer"),
            Some("child".into()),
            &crate::subagent::SubagentCapacityExceeded { limit: 1 },
        );
        assert!(!result.ok);
        assert!(result.error.unwrap().message.contains("no run was started"));
        let data = result.data.unwrap();
        assert_eq!(data["status"], "not_started");
        assert_eq!(data["started"], false);
        assert_eq!(data["active"], false);
        assert_eq!(data["accepted"], false);
        assert_eq!(data["max_concurrent"], 1);
        assert!(data.get("failure_kind").is_none());
    }
}

#[test]
fn hard_failure_result_preserves_compact_full_and_error_payloads() {
    let summary = "first line\n".to_string() + &"detail ".repeat(40);
    let result = RunnerSubagentDelegate::hard_failure_result(
        "agent__explore",
        "explorer",
        Some("child-123".to_string()),
        summary.clone(),
    );

    assert!(!result.ok);
    assert_eq!(result.tool, "agent__explore");
    assert_eq!(
        result.error.as_ref().map(|error| &error.message),
        Some(&summary)
    );
    assert_eq!(
        result.error.as_ref().map(|error| error.recoverable),
        Some(true)
    );

    let data = result.data.expect("hard failure data");
    assert_eq!(data["agent_name"], "explorer");
    assert_eq!(data["child_session_id"], "child-123");
    assert_eq!(data["status"], "failed");
    assert_eq!(data["failure_kind"], "hard");
    assert_eq!(data["summary"], compact_subagent_summary(&summary));
    assert_eq!(data["full_summary"], summary);
    assert_eq!(data["active"], false);
}

mod integration_tests {
    use super::*;

    #[tokio::test]
    async fn swarm_discovery_restores_tree_identity_and_keeps_run_history() {
        use crate::config::{SubagentMode, SubagentsConfig};
        use crate::subagent::SwarmMember;
        use crate::transcript::{child_sessions_dir, read_records};
        let dir = tempfile::tempdir().unwrap();
        let policy = SubagentsConfig {
            mode: SubagentMode::Swarm,
            ..Default::default()
        };
        let mut root = TranscriptRecorder::create(dir.path()).unwrap();
        root.record_session_started("gpt-test").unwrap();
        root.record_subagents_policy(&policy).unwrap();
        let root_id = root.session_id().to_string();
        let root_path = root.path().to_path_buf();

        fn record_run(
            parent: &mut TranscriptRecorder,
            child: &mut TranscriptRecorder,
            member: &SwarmMember,
            run: &str,
            status: &str,
        ) {
            let parent_id = parent.session_id().to_string();
            for recorder in [&mut *parent, &mut *child] {
                recorder
                    .record_subagent_started(
                        run,
                        &parent_id,
                        "turn-1",
                        &member.child_session_id,
                        &member.agent_name,
                        "private task",
                        member.pool_ordinal,
                    )
                    .unwrap();
            }
            child
                .record_subagent_activation(run.to_string(), member.clone())
                .unwrap();
            child
                .record_subagent_lifecycle(
                    run,
                    &parent_id,
                    "turn-1",
                    &member.agent_name,
                    status,
                    Some("private detail".into()),
                )
                .unwrap();
            parent
                .record_subagent_result_structured(
                    run,
                    &parent_id,
                    "turn-1",
                    &member.child_session_id,
                    &member.agent_name,
                    status,
                    "private summary",
                    None,
                )
                .unwrap();
        }
        let mut children: Vec<TranscriptRecorder> = Vec::new();
        let mut members: Vec<SwarmMember> = Vec::new();
        for ordinal in 1..=3 {
            let mut child = TranscriptRecorder::create(child_sessions_dir(dir.path())).unwrap();
            child.record_session_started("gpt-test").unwrap();
            child.record_subagents_policy(&policy).unwrap();
            let member = SwarmMember {
                root_session_id: root_id.clone(),
                parent_session_id: if ordinal == 3 {
                    members[0].child_session_id.clone()
                } else {
                    root_id.clone()
                },
                child_session_id: child.session_id().to_string(),
                depth: if ordinal == 3 { 2 } else { 1 },
                agent_name: "explorer".into(),
                pool_ordinal: ordinal,
            };
            if ordinal == 3 {
                record_run(
                    &mut children[0],
                    &mut child,
                    &member,
                    "grandchild-run",
                    "completed",
                );
            } else {
                record_run(
                    &mut root,
                    &mut child,
                    &member,
                    &format!("child-{ordinal}-run"),
                    "completed",
                );
            }
            members.push(member);
            children.push(child);
        }
        record_run(
            &mut root,
            &mut children[0],
            &members[0],
            "takeover-run",
            "failed",
        );
        let archived_id = members[2].child_session_id.clone();
        let archived_path =
            crate::transcript::archive_index::child_transcript_path(dir.path(), &archived_id);
        std::fs::create_dir_all(archived_path.parent().unwrap()).unwrap();
        let grandchild = children.pop().unwrap();
        let grandchild_path = grandchild.path().to_path_buf();
        drop(grandchild);
        std::fs::write(
            &archived_path,
            zstd::stream::encode_all(std::fs::File::open(&grandchild_path).unwrap(), 1).unwrap(),
        )
        .unwrap();
        std::fs::remove_file(grandchild_path).unwrap();
        use std::io::Write;
        let child_path = children[1].path().to_path_buf();
        drop(children);
        std::fs::OpenOptions::new()
            .append(true)
            .open(child_path)
            .unwrap()
            .write_all(b"{\"kind\":")
            .unwrap();

        let mut setup = TranscriptRecorder::create(child_sessions_dir(dir.path())).unwrap();
        setup.record_session_started("gpt-test").unwrap();
        setup.record_subagents_policy(&policy).unwrap();
        root.record_subagent_started(
            "setup-run",
            &root_id,
            "turn-1",
            setup.session_id(),
            "explorer",
            "private setup",
            99,
        )
        .unwrap();
        drop(setup);
        let runtime = SubagentPool::new();
        let transcript = Arc::new(Mutex::new(root));
        let delegate = Arc::new(RunnerSubagentDelegate {
            child_completions: None,
            runtime: runtime.clone(),
            sessions_dir: dir.path().to_path_buf(),
            transcript: transcript.clone(),
            event_tx: None,
            background_event_tx: None,
            child_live_text: None,
            background_child_started_tx: None,
            route_api_key_configured: Default::default(),
            provider_api_key_hints: Default::default(),
            api_key_hint: String::new(),
            background_control_tx: None,
        });
        for mode in [SubagentMode::Expert, SubagentMode::Swarm] {
            let mut agent = Agent::new("gpt-test", 2, 4);
            agent.set_subagents_policy(SubagentsConfig {
                mode,
                ..Default::default()
            });
            agent.set_subagent_delegate(delegate.clone());
            assert_eq!(
                agent
                    .tool_definitions_for_test()
                    .iter()
                    .any(|tool| tool.name == crate::tool_names::TOOL_AGENT_SEND),
                mode == SubagentMode::Swarm
            );
            let status = agent
                .tool_definitions_for_test()
                .into_iter()
                .find(|tool| tool.name == crate::tool_names::TOOL_AGENT_STATUS)
                .unwrap();
            assert_eq!(status.strict, mode == SubagentMode::Expert);
            if mode == SubagentMode::Expert {
                assert_eq!(
                    status.parameters,
                    json!({"type": "object", "properties": {"run_id": {"type": "string", "minLength": 1}}, "required": ["run_id"], "additionalProperties": false})
                );
            } else {
                assert!(status.parameters["properties"].get("session_id").is_some());
                assert_eq!(status.parameters["oneOf"].as_array().unwrap().len(), 2);
            }
        }
        let jobs = delegate
            .control(crate::tool_names::TOOL_AGENT_JOBS, &json!({}))
            .await
            .unwrap()
            .data
            .unwrap();
        assert_eq!(jobs["jobs"].as_array().unwrap().len(), 4);
        assert_eq!(jobs["jobs"][0]["session_id"], root_id);
        assert_eq!(jobs["jobs"][0]["role"], "root");
        assert_eq!(jobs["jobs"][0]["status"], "idle");
        for field in ["run_id", "child_session_id", "pool_ordinal"] {
            assert!(jobs["jobs"][0].get(field).is_none());
        }
        assert_eq!(jobs["jobs"][3]["depth"], 2);
        let child_records = read_records(crate::transcript::child_session_path(
            dir.path(),
            &members[0].child_session_id,
        ))
        .unwrap();
        assert_eq!(
            runtime
                .swarm_snapshot(dir.path(), &child_records)
                .unwrap()
                .latest_jobs()
                .len(),
            4
        );
        transcript
            .lock()
            .unwrap()
            .record_turn_started(crate::agent::TurnStartedEvent { turn_id: 1 })
            .unwrap();
        let active_root = delegate
            .control(
                crate::tool_names::TOOL_AGENT_STATUS,
                &json!({"session_id": root_id}),
            )
            .await
            .unwrap()
            .data
            .unwrap();
        assert_eq!(active_root["status"], "active");
        assert_eq!(active_root["active"], true);
        transcript
            .lock()
            .unwrap()
            .record_turn_interrupted(Some(1))
            .unwrap();
        assert!(!jobs.to_string().contains("private"));
        for field in ["summary", "full_summary", "structured_result", "task"] {
            assert!(!jobs.to_string().contains(field));
        }
        for (selector, expected) in [
            (json!({"run_id": "child-1-run"}), "completed"),
            (json!({"run_id": "takeover-run"}), "failed"),
            (json!({"session_id": members[0].child_session_id}), "failed"),
            (json!({"session_id": root_id}), "idle"),
            (json!({"session_id": archived_id}), "completed"),
        ] {
            let status = delegate
                .control(crate::tool_names::TOOL_AGENT_STATUS, &selector)
                .await
                .unwrap();
            assert!(status.ok);
            assert_eq!(status.data.unwrap()["status"], expected);
        }
        for args in [
            json!({}),
            json!({"run_id":"x", "session_id":"y"}),
            json!({"session_id":" "}),
            json!({"run_id":null}),
            json!({"other":"x"}),
        ] {
            assert!(
                delegate
                    .control(crate::tool_names::TOOL_AGENT_STATUS, &args)
                    .await
                    .is_err()
            );
        }
        let child_delegate = delegate
            .for_child(Arc::new(Mutex::new(
                TranscriptRecorder::open(
                    child_sessions_dir(dir.path()),
                    &members[0].child_session_id,
                )
                .unwrap(),
            )))
            .unwrap();
        for tool in [
            crate::tool_names::TOOL_AGENT_WAIT,
            crate::tool_names::TOOL_AGENT_CANCEL,
        ] {
            let result = child_delegate
                .control(tool, &json!({"run_id": "child-2-run"}))
                .await
                .unwrap();
            assert!(!result.ok);
        }
        drop(child_delegate);
        let old_records = read_records(&root_path).unwrap();
        assert_eq!(
            runtime
                .swarm_snapshot(dir.path(), &old_records)
                .unwrap()
                .members
                .len(),
            3
        );
        let mut parent = Agent::new("gpt-test", 2, 4);
        parent.set_subagents_policy(policy.clone());
        parent.set_subagent_delegate(delegate.clone());
        let invocation = || SubagentInvocation {
            prompt: "inspect".into(),
            model: None,
            parent_tool_call_id: None,
            input: crate::tool::NormalizedSubagentInput {
                objective: "inspect".into(),
                success_criteria: Vec::new(),
                allowed_paths: Vec::new(),
                forbidden_paths: Vec::new(),
                owned_paths: Vec::new(),
                timeout_secs: None,
                max_tool_calls: None,
                model: None,
                target_child_session_id: None,
                background: false,
            },
        };
        let started = runtime
            .clone()
            .start_named_governed(
                &parent,
                "explorer",
                invocation(),
                dir.path(),
                root_id.clone(),
                "turn-2".into(),
                Some(transcript.clone()),
                None,
            )
            .unwrap();
        assert_eq!(started.receipt().pool_ordinal, 4);
        let run_id = started.run_id().to_string();
        let snapshot = runtime.swarm_snapshot(dir.path(), &old_records).unwrap();
        assert!(runtime.active_jobs().iter().any(|job| job.run_id == run_id));
        assert_eq!(snapshot.members.len(), 4);
        runtime.cancel_run(&run_id);
        runtime.complete_started_run(started).await.unwrap();
        let records = read_records(&root_path).unwrap();
        let snapshot = runtime.swarm_snapshot(dir.path(), &records).unwrap();
        assert_eq!(snapshot.members.len(), 4);
        assert_eq!(
            SubagentPool::new()
                .swarm_snapshot(dir.path(), &records)
                .unwrap()
                .members,
            snapshot.members
        );

        let mut other = TranscriptRecorder::create(dir.path()).unwrap();
        other.record_subagents_policy(&policy).unwrap();
        let other_records = read_records(other.path()).unwrap();
        assert!(
            !delegate
                .control(
                    crate::tool_names::TOOL_AGENT_STATUS,
                    &json!({"session_id": other.session_id()})
                )
                .await
                .unwrap()
                .ok
        );
        assert_eq!(
            runtime
                .swarm_snapshot(dir.path(), &other_records)
                .unwrap()
                .latest_jobs()
                .len(),
            1
        );

        drop(parent);
        drop(delegate);
        drop(transcript);
        let prepared = crate::session::prepare_resume_package(dir.path(), &root_id).unwrap();
        drop(prepared);
        let child_path =
            crate::transcript::child_session_path(dir.path(), &members[0].child_session_id);
        let original = std::fs::read_to_string(&child_path).unwrap();
        let live_path = other.path().to_path_buf();
        let live_content = std::fs::read(&live_path).unwrap();
        for (field, value) in [
            ("root_session_id", json!("foreign-root")),
            ("parent_session_id", json!(members[1].child_session_id)),
            ("depth", json!(9)),
            ("agent_name", json!("oracle")),
            ("pool_ordinal", json!(2)),
        ] {
            let corrupted: String = original
                .lines()
                .map(|line| {
                    let mut record: serde_json::Value = serde_json::from_str(line).unwrap();
                    if record["kind"] == "subagent_activated" {
                        record["member"][field] = value.clone();
                    }
                    format!("{record}\n")
                })
                .collect();
            std::fs::write(&child_path, corrupted).unwrap();
            assert!(
                crate::session::prepare_resume_package(dir.path(), &root_id).is_err(),
                "{field}"
            );
            assert_eq!(
                crate::transcript::restore_subagents_policy(&read_records(&root_path).unwrap())
                    .unwrap(),
                policy
            );
            assert_eq!(std::fs::read(&live_path).unwrap(), live_content);
        }
        std::fs::write(child_path, original).unwrap();
    }
}
