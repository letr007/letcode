use super::*;

fn fixture(
    policy: crate::config::SubagentsConfig,
) -> (
    tempfile::TempDir,
    SubagentPool,
    Agent,
    Arc<Mutex<TranscriptRecorder>>,
) {
    let dir = tempfile::tempdir().unwrap();
    let runtime = SubagentPool::new();
    let mut parent = Agent::new("gpt-test", 2, 4);
    parent.set_primary_route(ModelRoute::new("test", "model"));
    parent.set_subagents_policy(policy.clone());
    let mut recorder = TranscriptRecorder::create(dir.path()).unwrap();
    recorder.record_session_started("test/model").unwrap();
    recorder.record_subagents_policy(&policy).unwrap();
    let recorder = Arc::new(Mutex::new(recorder));
    let (control, _controls) = tokio::sync::mpsc::unbounded_channel();
    crate::session::runner::AgentRunner::silent_with_transcript(recorder.clone())
        .with_subagent_runtime(
            runtime.clone(),
            dir.path().into(),
            Default::default(),
            Default::default(),
            String::new(),
            Some(control),
            None,
        )
        .install_subagent_delegate_for_test(&mut parent);
    parent.prepare_subagent_runtime().unwrap();
    (dir, runtime, parent, recorder)
}

fn invocation(args: Value) -> SubagentInvocation {
    let input = crate::tool::normalize_subagent_input("agent__explore", &args).unwrap();
    SubagentInvocation {
        prompt: input.objective.clone(),
        input,
        model: None,
        parent_tool_call_id: None,
    }
}

fn start(
    runtime: &SubagentPool,
    parent: &Agent,
    dir: &Path,
    recorder: &Arc<Mutex<TranscriptRecorder>>,
    args: Value,
) -> Result<StartedSubagentRun> {
    let parent_id = recorder.lock().unwrap().session_id().to_string();
    runtime.start_named_governed(
        parent,
        "explorer",
        invocation(args),
        dir,
        parent_id,
        "turn-1".into(),
        Some(recorder.clone()),
        None,
    )
}

fn resume(
    runtime: &SubagentPool,
    parent: &Agent,
    dir: &Path,
    recorder: &Arc<Mutex<TranscriptRecorder>>,
    target: &str,
) -> StartedSubagentRun {
    let parent_id = recorder.lock().unwrap().session_id().to_string();
    let template = AgentTemplate::explorer();
    let governance = SubagentRunGovernance::from_template_and_input(
        &template,
        invocation(serde_json::json!({"objective":"resume"})).input,
        None,
    );
    let run = runtime
        .start_run(
            parent,
            template,
            String::new(),
            governance,
            dir.into(),
            parent_id,
            "turn-resume".into(),
            Some(recorder.clone()),
            None,
            Some(target.into()),
            true,
        )
        .unwrap();
    let receipt = runtime.state.lock().unwrap().active_by_run[&run.run_id]
        .child
        .clone();
    StartedSubagentRun {
        run_id: run.run_id.clone(),
        receipt,
        run,
        task: String::new(),
    }
}

#[test]
fn shared_admission_reserves_the_last_slot_atomically() {
    let pool = SubagentPool::new();
    let policy = crate::config::SubagentsConfig {
        mode: crate::config::SubagentMode::Swarm,
        max_concurrent: 1,
        ..Default::default()
    };
    let barrier = std::sync::Barrier::new(2);
    let reservations = std::thread::scope(|scope| {
        let jobs: Vec<_> = (0..2)
            .map(|_| {
                let pool = &pool;
                let policy = &policy;
                let barrier = &barrier;
                scope.spawn(move || {
                    barrier.wait();
                    pool.reserve_governed_run(
                        RunPathAccess::Read(Vec::new()),
                        None,
                        Some(SwarmAdmission {
                            root_session_id: "root",
                            parent_session_id: "root",
                            depth: 1,
                            policy,
                            message_resume: false,
                        }),
                    )
                })
            })
            .collect();
        jobs.into_iter()
            .map(|job| job.join().unwrap())
            .collect::<Vec<_>>()
    });
    assert_eq!(
        reservations
            .iter()
            .filter(|reservation| reservation.is_ok())
            .count(),
        1
    );
    assert!(
        reservations
            .iter()
            .filter_map(|reservation| reservation.as_ref().err())
            .all(|error| error.is::<SubagentCapacityExceeded>())
    );
    let other_root = pool
        .reserve_governed_run(
            RunPathAccess::Read(Vec::new()),
            None,
            Some(SwarmAdmission {
                root_session_id: "other-root",
                parent_session_id: "other-root",
                depth: 1,
                policy: &policy,
                message_resume: false,
            }),
        )
        .unwrap();
    drop(other_root);
    drop(reservations);
    let reservation = pool
        .reserve_governed_run(
            RunPathAccess::Read(Vec::new()),
            None,
            Some(SwarmAdmission {
                root_session_id: "root",
                parent_session_id: "root",
                depth: 1,
                policy: &policy,
                message_resume: false,
            }),
        )
        .unwrap();
    drop(reservation);
    assert!(!pool.is_running());
}

mod integration_tests {
    use super::*;

    #[test]
    fn missing_owned_leaf_locks_conflict_only_with_overlapping_runs() {
        let (dir, runtime, parent, recorder) = fixture(crate::config::SubagentsConfig {
            mode: crate::config::SubagentMode::Swarm,
            ..Default::default()
        });
        let root = recorder.lock().unwrap().session_id().to_string();
        let leaf = dir.path().join("new.txt");
        let start_writer = |path: &Path| {
            runtime.start_named_governed(
                &parent,
                "fixer",
                invocation(serde_json::json!({"objective":"edit", "owned_paths":[path]})),
                dir.path(),
                root.clone(),
                "turn-1".into(),
                Some(recorder.clone()),
                None,
            )
        };
        let first = start_writer(&leaf).unwrap();
        let expected = dir.path().canonicalize().unwrap().join("new.txt");
        assert_eq!(
            first
                .run
                .child_agent
                .delegation_scope()
                .unwrap()
                .owned_roots(),
            &[expected]
        );
        for overlapping in [&leaf, dir.path()] {
            let error = start_writer(overlapping).err().unwrap();
            assert!(error.to_string().contains("path lock conflict"), "{error}");
        }
        let neighbor = start_writer(&dir.path().join("neighbor.txt")).unwrap();
        drop(neighbor);
        assert!(!leaf.exists());
        drop(first);
        drop(start_writer(&leaf).unwrap());
    }

    #[test]
    fn terminal_members_release_capacity_and_restore_the_frozen_root_limit() {
        let (dir, runtime, mut parent, recorder) = fixture(crate::config::SubagentsConfig {
            mode: crate::config::SubagentMode::Swarm,
            max_concurrent: 1,
            ..Default::default()
        });
        let mut invalid = invocation(serde_json::json!({"objective":"inspect"}));
        invalid.model = Some(ModelRoute::new("missing", "model"));
        let root = recorder.lock().unwrap().session_id().to_string();
        assert!(
            runtime
                .start_named_governed(
                    &parent,
                    "explorer",
                    invalid,
                    dir.path(),
                    root.clone(),
                    "turn-1".into(),
                    Some(recorder.clone()),
                    None
                )
                .is_err()
        );
        assert!(!runtime.is_running());
        assert!(
            runtime.state.lock().unwrap().swarms[&root]
                .members
                .is_empty()
        );
        let started = start(
            &runtime,
            &parent,
            dir.path(),
            &recorder,
            serde_json::json!({"objective":"inspect"}),
        )
        .unwrap();
        drop(started);
        parent.set_subagents_policy(crate::config::SubagentsConfig::default());
        for runtime in [runtime, SubagentPool::new()] {
            let started = start(
                &runtime,
                &parent,
                dir.path(),
                &recorder,
                serde_json::json!({"objective":"inspect"}),
            )
            .unwrap();
            let target = started.receipt.child_session_id.clone();
            assert_eq!(started.run.child_agent.subagents_policy().max_concurrent, 1);
            let error = start(
                &runtime,
                &parent,
                dir.path(),
                &recorder,
                serde_json::json!({"objective":"inspect"}),
            )
            .err()
            .unwrap();
            assert!(error.is::<SubagentCapacityExceeded>(), "{error}");
            assert!(runtime.state.lock().unwrap().swarms[&root].members.len() >= 2);
            drop(started);
            let restored = SubagentPool::new();
            let busy = start(
                &restored,
                &parent,
                dir.path(),
                &recorder,
                serde_json::json!({"objective":"inspect"}),
            )
            .unwrap();
            let error = start(
                &restored,
                &parent,
                dir.path(),
                &recorder,
                serde_json::json!({"objective":"resume", "target_child_session_id":target}),
            )
            .err()
            .unwrap();
            assert!(error.is::<SubagentCapacityExceeded>(), "{error}");
            drop(busy);
        }
    }

    #[tokio::test]
    async fn message_wakeups_share_capacity_with_starting_and_active_runs() {
        let (dir, runtime, parent, recorder) = fixture(crate::config::SubagentsConfig {
            mode: crate::config::SubagentMode::Swarm,
            max_concurrent: 1,
            ..Default::default()
        });
        let root = recorder.lock().unwrap().session_id().to_string();
        let dormant = start(
            &runtime,
            &parent,
            dir.path(),
            &recorder,
            serde_json::json!({"objective":"inspect"}),
        )
        .unwrap();
        let target = dormant.receipt.child_session_id.clone();
        runtime
            .complete_started_run_with_executor(dormant, |_, _, _, _, _, _| {
                Box::pin(async { Ok("done".into()) })
            })
            .await
            .unwrap();
        let busy = start(
            &runtime,
            &parent,
            dir.path(),
            &recorder,
            serde_json::json!({"objective":"inspect"}),
        )
        .unwrap();
        let records = read_records_allow_partial_tail(recorder.lock().unwrap().path()).unwrap();
        let error = runtime
            .enqueue_member_message(dir.path(), &records, &target, "hello")
            .unwrap_err();
        assert!(error.is::<SubagentCapacityExceeded>(), "{error}");
        assert!(!runtime.member_message_pending(&root, &target).unwrap());
        let active = busy.receipt.child_session_id.clone();
        assert_eq!(
            runtime
                .enqueue_member_message(dir.path(), &records, &active, "hello")
                .unwrap(),
            1
        );
        assert!(
            runtime
                .take_member_message(&root, &active)
                .unwrap()
                .is_some()
        );
        drop(busy);
        assert_eq!(
            runtime
                .enqueue_member_message(dir.path(), &records, &target, "first")
                .unwrap(),
            1
        );
        assert_eq!(
            runtime
                .enqueue_member_message(dir.path(), &records, &target, "second")
                .unwrap(),
            2
        );
        assert_eq!(runtime.state.lock().unwrap().swarm_occupancy(&root), 1);
        let waking = runtime
            .reserve_governed_run(
                RunPathAccess::Read(Vec::new()),
                Some(&target),
                Some(SwarmAdmission {
                    root_session_id: &root,
                    parent_session_id: &root,
                    depth: 1,
                    policy: parent.subagents_policy(),
                    message_resume: true,
                }),
            )
            .unwrap();
        assert_eq!(runtime.state.lock().unwrap().swarm_occupancy(&root), 1);
        let error = start(
            &runtime,
            &parent,
            dir.path(),
            &recorder,
            serde_json::json!({"objective":"inspect"}),
        )
        .err()
        .unwrap();
        assert!(error.is::<SubagentCapacityExceeded>(), "{error}");
        drop(waking);
        runtime.fail_member_messages(&root, &target, "message execution stopped", false);
        assert_eq!(runtime.state.lock().unwrap().swarm_occupancy(&root), 0);
    }

    #[tokio::test]
    async fn recursive_scope_and_activation_deadline_bound_descendants() {
        let (dir, runtime, parent, recorder) = fixture(crate::config::SubagentsConfig {
            mode: crate::config::SubagentMode::Swarm,
            max_depth: 2,
            run_timeout_secs: 10,
            ..Default::default()
        });
        let root_tools = parent.tool_definitions_for_test();
        for seconds in [0, u64::MAX] {
            let mut invalid = invocation(serde_json::json!({"objective":"inspect"}));
            invalid.input.timeout_secs = Some(seconds);
            let root = recorder.lock().unwrap().session_id().to_string();
            assert!(
                runtime
                    .start_named_governed(
                        &parent,
                        "explorer",
                        invalid,
                        dir.path(),
                        root,
                        "turn-1".into(),
                        Some(recorder.clone()),
                        None
                    )
                    .is_err()
            );
        }
        let root_id = recorder.lock().unwrap().session_id().to_string();
        let before = Instant::now();
        let first = start(&runtime, &parent, dir.path(), &recorder, serde_json::json!({"objective":"inspect", "allowed_paths":["src"], "forbidden_paths":["src/subagent"]})).unwrap();
        assert!(first.run.deadline.unwrap() >= before + Duration::from_secs(10));
        assert_eq!(first.run.timeout_secs, Some(10));
        let child_agent = &first.run.child_agent;
        let child_recorder = &first.run.child_transcript;
        let grandchild = start(
            &runtime,
            child_agent,
            dir.path(),
            child_recorder,
            serde_json::json!({"objective":"inspect"}),
        )
        .unwrap();
        let scope = grandchild.run.child_agent.delegation_scope().unwrap();
        let src = std::fs::canonicalize("src").unwrap();
        let forbidden = src.join("subagent/pool.rs");
        assert_eq!(scope.read_roots(), vec![src.clone()]);
        assert!(!scope.permits_read([forbidden.as_path()]));
        let narrowed = start(
            &runtime,
            child_agent,
            dir.path(),
            child_recorder,
            serde_json::json!({"objective":"inspect", "allowed_paths":["src/../src"]}),
        )
        .unwrap();
        let scope = narrowed.run.child_agent.delegation_scope().unwrap();
        assert_eq!(scope.read_roots(), vec![src]);
        assert!(!scope.permits_read([forbidden.as_path()]));
        drop(narrowed);
        for allowed in ["src/../Cargo.toml", "src/subagent"] {
            let error = start(
                &runtime,
                child_agent,
                dir.path(),
                child_recorder,
                serde_json::json!({"objective":"inspect", "allowed_paths":[allowed]}),
            )
            .err()
            .unwrap();
            assert!(
                error.to_string().contains("exceeds parent scope"),
                "{error}"
            );
        }
        let error = start(
            &runtime,
            &grandchild.run.child_agent,
            dir.path(),
            &grandchild.run.child_transcript,
            serde_json::json!({"objective":"inspect"}),
        )
        .err()
        .unwrap();
        assert!(error.to_string().contains("depth exhausted"), "{error}");
        let error = runtime
            .start_named_governed(
                child_agent,
                "fixer",
                invocation(serde_json::json!({"objective":"edit", "owned_paths":["src"]})),
                dir.path(),
                first.run.child_session_id.clone(),
                "turn-1".into(),
                Some(child_recorder.clone()),
                None,
            )
            .err()
            .unwrap();
        assert!(error.to_string().contains("read-only caller"), "{error}");
        let error = runtime
            .start_named_governed(
                &parent,
                "fixer",
                invocation(serde_json::json!({"objective":"edit", "owned_paths":["src"]})),
                dir.path(),
                root_id.clone(),
                "turn-1".into(),
                Some(recorder.clone()),
                None,
            )
            .err()
            .unwrap();
        assert!(error.to_string().contains("path lock conflict"), "{error}");
        let dormant = start(
            &runtime,
            child_agent,
            dir.path(),
            child_recorder,
            serde_json::json!({"objective":"dormant"}),
        )
        .unwrap();
        let dormant_id = dormant.run.child_session_id.clone();
        let waking = start(
            &runtime,
            child_agent,
            dir.path(),
            child_recorder,
            serde_json::json!({"objective":"waking"}),
        )
        .unwrap();
        let waking_id = waking.run.child_session_id.clone();
        runtime
            .complete_started_run_with_executor(waking, |_, _, _, _, _, _| {
                async { Ok("done".into()) }.boxed()
            })
            .await
            .unwrap();
        runtime
            .complete_started_run_with_executor(dormant, |_, _, _, _, _, _| {
                async { Ok("done".into()) }.boxed()
            })
            .await
            .unwrap();
        let timed_target = first.run.child_session_id.clone();
        runtime
            .complete_started_run_with_executor(first, |_, _, _, _, _, _| {
                async { Ok("handoff".into()) }.boxed()
            })
            .await
            .unwrap();
        assert!(
            !runtime.state.lock().unwrap().active_by_run[grandchild.run_id()]
                .cancellation_requested
        );
        let mut first = resume(&runtime, &parent, dir.path(), &recorder, &timed_target);
        let awakened = resume(
            &runtime,
            &first.run.child_agent,
            dir.path(),
            &first.run.child_transcript,
            &dormant_id,
        );
        assert!(
            runtime.state.lock().unwrap().active_by_run[awakened.run_id()]
                .parent_run_id
                .is_none()
        );
        let pending = runtime
            .reserve_governed_run(
                RunPathAccess::Read(Vec::new()),
                None,
                Some(SwarmAdmission {
                    root_session_id: &root_id,
                    parent_session_id: &first.run.child_session_id,
                    depth: 2,
                    policy: parent.subagents_policy(),
                    message_resume: false,
                }),
            )
            .unwrap();
        let root_records =
            crate::transcript::read_records(recorder.lock().unwrap().path()).unwrap();
        let timed_target = first.run.child_session_id.clone();
        let timed_path = first
            .run
            .child_transcript
            .lock()
            .unwrap()
            .path()
            .to_path_buf();
        assert_eq!(
            runtime
                .enqueue_member_message(
                    dir.path(),
                    &root_records,
                    &timed_target,
                    "pending old message"
                )
                .unwrap(),
            1
        );
        {
            let mut state = runtime.state.lock().unwrap();
            let mailbox = state
                .swarms
                .get_mut(&root_id)
                .unwrap()
                .mailboxes
                .get_mut(&waking_id)
                .unwrap();
            mailbox.waking = true;
            mailbox.pending.push_back("waking before timeout".into());
        }
        first.run.deadline = Some(Instant::now() - Duration::from_millis(1));
        let result = runtime
            .complete_started_run_with_executor(first, |_, _, _, _, _, _| {
                panic!("expired run must not enter executor")
            })
            .await
            .unwrap();
        assert_eq!(result.status, SubagentStatus::TimedOut);
        assert!(
            runtime.state.lock().unwrap().swarms[&root_id].mailboxes[&waking_id].wake_cancelled
        );
        let waking_admission = runtime.reserve_governed_run(
            RunPathAccess::Read(Vec::new()),
            Some(&waking_id),
            Some(SwarmAdmission {
                root_session_id: &root_id,
                parent_session_id: &timed_target,
                depth: 2,
                policy: parent.subagents_policy(),
                message_resume: true,
            }),
        );
        assert!(waking_admission.is_err());
        runtime.fail_member_messages(&root_id, &waking_id, "ancestor timed out", false);
        assert!(
            !runtime.state.lock().unwrap().swarms[&root_id].mailboxes[&waking_id].wake_cancelled
        );
        assert!(
            !runtime
                .member_message_pending(&root_id, &waking_id)
                .unwrap()
        );
        assert!(
            !runtime
                .member_message_pending(&root_id, &timed_target)
                .unwrap()
        );
        let timeout_records = crate::transcript::read_records(&timed_path).unwrap();
        assert!(timeout_records.iter().any(|record| matches!(&record.event, TranscriptEvent::Error { message } if message.contains("queued message") && message.contains("timed_out"))));
        assert!(!timeout_records.iter().any(|record| matches!(&record.event, TranscriptEvent::UserMessage { content } if content.text == "pending old message")));
        assert!(
            runtime.state.lock().unwrap().active_by_run[&pending.run_id].cancellation_requested
        );
        drop(pending);
        assert!(
            runtime.state.lock().unwrap().active_by_run[awakened.run_id()].cancellation_requested
        );
        let result = runtime
            .complete_started_run_with_executor(awakened, |_, _, _, _, _, _| {
                panic!("awakened descendant must be cancelled")
            })
            .await
            .unwrap();
        assert_eq!(result.status, SubagentStatus::Cancelled);
        let result = runtime
            .complete_started_run_with_executor(grandchild, |_, _, _, _, _, _| {
                panic!("cancelled descendant must not enter executor")
            })
            .await
            .unwrap();
        assert_eq!(result.status, SubagentStatus::Cancelled);
        assert!(!runtime.is_running());
        assert_eq!(
            runtime
                .enqueue_member_message(
                    dir.path(),
                    &root_records,
                    &timed_target,
                    "later explicit message"
                )
                .unwrap(),
            1
        );
        assert!(runtime.wait_until_idle().await);
        let resumed_records = crate::transcript::read_records(&timed_path).unwrap();
        assert_eq!(
            resumed_records
                .iter()
                .filter(|record| matches!(&record.event, TranscriptEvent::SubagentActivated { .. }))
                .count(),
            3
        );
        let delivered = format!(
            "[Swarm message from session_id={root_id} to session_id={timed_target}]\n\nlater explicit message"
        );
        assert!(resumed_records.iter().any(|record| matches!(&record.event, TranscriptEvent::UserMessage { content } if content.text == delivered)));
        assert!(!resumed_records.iter().any(|record| matches!(&record.event, TranscriptEvent::UserMessage { content } if content.text.ends_with("\n\npending old message"))));

        let mut running = start(
            &runtime,
            &parent,
            dir.path(),
            &recorder,
            serde_json::json!({"objective":"inspect"}),
        )
        .unwrap();
        tokio::time::pause();
        let deadline = Instant::now() + Duration::from_millis(100);
        running.run.deadline = Some(deadline);
        tokio::time::advance(Duration::from_millis(30)).await;
        let steps = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let observed = steps.clone();
        let execution =
            runtime.complete_started_run_with_executor(running, move |_, _, _, _, _, _| {
                async move {
                    observed.store(1, std::sync::atomic::Ordering::SeqCst);
                    tokio::time::sleep(Duration::from_millis(40)).await;
                    observed.store(2, std::sync::atomic::Ordering::SeqCst);
                    tokio::time::sleep(Duration::from_millis(40)).await;
                    observed.store(3, std::sync::atomic::Ordering::SeqCst);
                    Ok("done".into())
                }
                .boxed()
            });
        tokio::pin!(execution);
        assert!(futures_util::poll!(execution.as_mut()).is_pending());
        assert_eq!(steps.load(std::sync::atomic::Ordering::SeqCst), 1);
        tokio::time::advance(Duration::from_millis(41)).await;
        futures_util::future::poll_fn(|cx| {
            assert!(execution.as_mut().poll(cx).is_pending());
            if steps.load(std::sync::atomic::Ordering::SeqCst) == 2 {
                std::task::Poll::Ready(())
            } else {
                std::task::Poll::Pending
            }
        })
        .await;
        assert!(Instant::now() < deadline);
        let timeout_observation = deadline + Duration::from_millis(1);
        tokio::time::advance(timeout_observation - Instant::now()).await;
        let result = execution.await.unwrap();
        assert_eq!(result.status, SubagentStatus::TimedOut);
        assert_eq!(steps.load(std::sync::atomic::Ordering::SeqCst), 2);
        assert_eq!(Instant::now(), timeout_observation);
        tokio::time::resume();

        let stopped = start(
            &runtime,
            &parent,
            dir.path(),
            &recorder,
            serde_json::json!({"objective":"inspect"}),
        )
        .unwrap();
        let stopped_target = stopped.receipt().child_session_id.clone();
        let stopped_tools = stopped.run.child_agent.tool_definitions_for_test();
        runtime
            .enqueue_member_message(
                dir.path(),
                &root_records,
                &stopped_target,
                "queued before stop",
            )
            .unwrap();
        let starting = runtime
            .reserve_governed_run(
                RunPathAccess::Read(Vec::new()),
                None,
                Some(SwarmAdmission {
                    root_session_id: &root_id,
                    parent_session_id: &stopped_target,
                    depth: 2,
                    policy: parent.subagents_policy(),
                    message_resume: false,
                }),
            )
            .unwrap();
        assert!(runtime.cancel_run(stopped.run_id()));
        assert_eq!(
            stopped.run.child_agent.tool_definitions_for_test(),
            stopped_tools
        );
        assert!(
            runtime
                .resume_swarm(dir.path(), &root_records)
                .unwrap_err()
                .to_string()
                .contains("not settled")
        );
        let late_send = stopped
            .run
            .child_agent
            .execute_subagent_control_tool_for_test(
                "agent__send",
                &serde_json::json!({"session_id":root_id,"message":"late sender"}),
            )
            .await;
        assert!(!late_send.ok);
        assert!(
            runtime.state.lock().unwrap().active_by_run[&starting.run_id].cancellation_requested
        );
        assert!(
            !runtime
                .member_message_pending(&root_id, &stopped_target)
                .unwrap()
        );
        assert!(
            runtime
                .enqueue_member_message(dir.path(), &root_records, &stopped_target, "late")
                .is_err()
        );
        assert!(
            start(
                &runtime,
                &stopped.run.child_agent,
                dir.path(),
                &stopped.run.child_transcript,
                serde_json::json!({"objective":"late child"})
            )
            .is_err()
        );
        drop(starting);
        runtime
            .complete_started_run_with_executor(stopped, |_, _, _, _, _, _| {
                panic!("stopped member must not execute")
            })
            .await
            .unwrap();
        assert!(runtime.wait_until_idle().await);
        runtime.cancel_active();
        assert_eq!(parent.tool_definitions_for_test(), root_tools);
        let fresh = start(
            &runtime,
            &parent,
            dir.path(),
            &recorder,
            serde_json::json!({"objective":"inspect"}),
        )
        .unwrap();
        drop(fresh);

        let (dir, runtime, parent, recorder) = fixture(crate::config::SubagentsConfig::default());
        let expert = start(
            &runtime,
            &parent,
            dir.path(),
            &recorder,
            serde_json::json!({"objective":"inspect"}),
        )
        .unwrap();
        assert!(expert.run.timeout_secs.is_none());
        assert!(expert.run.deadline.is_none());
        drop(expert);
    }

    #[test]
    fn root_ordinals_are_shared_and_failed_activation_releases_reservation() {
        use std::io::Write;
        struct FailingSink {
            file: std::fs::File,
            stage: u8,
        }
        impl crate::transcript::JournalSink for FailingSink {
            fn write_all(&mut self, bytes: &[u8]) -> std::io::Result<()> {
                match self.stage {
                    0 => Err(std::io::Error::other("activation write failed")),
                    1 => {
                        self.file.write_all(&bytes[..bytes.len() / 2])?;
                        Err(std::io::Error::other("activation partial write failed"))
                    }
                    _ => self.file.write_all(bytes),
                }
            }
            fn flush(&mut self) -> std::io::Result<()> {
                if self.stage == 2 {
                    Err(std::io::Error::other("activation flush failed"))
                } else {
                    self.file.flush()
                }
            }
            fn sync_data(&mut self) -> std::io::Result<()> {
                Err(std::io::Error::other("activation sync failed"))
            }
        }
        let pool = SubagentPool::new();
        let ordinals = std::thread::scope(|scope| {
            let threads: Vec<_> = (0..8)
                .map(|_| {
                    let pool = pool.clone();
                    scope.spawn(move || pool.allocate_swarm_ordinal("root").unwrap())
                })
                .collect();
            threads
                .into_iter()
                .map(|thread| thread.join().unwrap())
                .collect::<std::collections::BTreeSet<_>>()
        });
        assert_eq!(ordinals, (1..=8).collect());
        assert_eq!(pool.allocate_swarm_ordinal("other-root").unwrap(), 1);
        for stage in 0..=4 {
            let dir = tempfile::tempdir().unwrap();
            let mut parent = TranscriptRecorder::create(dir.path()).unwrap();
            parent.record_session_started("test/model").unwrap();
            parent
                .record_subagents_policy(&crate::config::SubagentsConfig {
                    mode: crate::config::SubagentMode::Swarm,
                    ..Default::default()
                })
                .unwrap();
            let root_id = parent.session_id().to_string();
            let root_path = parent.path().to_path_buf();
            let mut recorder = TranscriptRecorder::create(child_sessions_dir(dir.path())).unwrap();
            recorder.record_session_started("test/model").unwrap();
            let member = crate::subagent::SwarmMember {
                root_session_id: root_id.clone(),
                parent_session_id: root_id.clone(),
                child_session_id: recorder.session_id().to_string(),
                depth: 1,
                agent_name: "explorer".into(),
                pool_ordinal: 1,
            };
            let path_access = RunPathAccess::Write(vec![dir.path().to_path_buf()]);
            let reservation = pool.reserve_run(path_access.clone(), None).unwrap();
            let run_id = reservation.run_id.clone();
            for journal in [&mut parent, &mut recorder] {
                journal
                    .record_subagent_started(
                        &run_id,
                        &root_id,
                        "turn-1",
                        &member.child_session_id,
                        "explorer",
                        "inspect",
                        1,
                    )
                    .unwrap();
            }
            recorder
                .record_subagent_lifecycle(&run_id, &root_id, "turn-1", "explorer", "running", None)
                .unwrap();
            let path = recorder.path().to_path_buf();
            recorder.replace_sink_for_test(Box::new(FailingSink {
                file: std::fs::OpenOptions::new()
                    .append(true)
                    .open(&path)
                    .unwrap(),
                stage: stage.min(3),
            }));
            if stage == 4 {
                parent.replace_sink_for_test(Box::new(FailingSink {
                    file: std::fs::OpenOptions::new()
                        .append(true)
                        .open(&root_path)
                        .unwrap(),
                    stage: 0,
                }));
            }
            let recorder = Arc::new(Mutex::new(recorder));
            let parent = Some(Arc::new(Mutex::new(parent)));
            let (cancel, _) = oneshot::channel();
            let summary = ChildSessionSummary {
                parent_session_id: root_id.clone(),
                parent_run_id: "turn-1".into(),
                child_session_id: member.child_session_id.clone(),
                agent_name: "explorer".into(),
                status: "running".into(),
                summary: String::new(),
                timestamp_ms: 0,
                pool_ordinal: 1,
            };
            let error = pool
                .activate_run(
                    reservation,
                    cancel,
                    summary,
                    &recorder,
                    &parent,
                    Some(member.clone()),
                    None,
                )
                .unwrap_err();
            let message = error.to_string();
            let expected = match stage {
                0 => "activation write failed",
                1 => "activation partial write failed",
                2 => "activation flush failed",
                _ => "activation sync failed",
            };
            assert!(message.contains(expected), "{message}");
            if stage == 4 {
                assert!(
                    message.contains("failed to record parent activation failure"),
                    "{message}"
                );
            }
            assert!(!pool.is_running());
            let next = pool.reserve_run(path_access, None).unwrap();
            drop(next);
            let result = pool.completed_result(&run_id).unwrap();
            assert_eq!(result.status, SubagentStatus::Failed);
            assert!(result.summary.contains(expected));
            let records = crate::transcript::read_records(&root_path).unwrap();
            for runtime in [pool.clone(), SubagentPool::new()] {
                let snapshot = runtime.swarm_snapshot(dir.path(), &records).unwrap();
                assert_eq!(snapshot.members.len(), usize::from(stage >= 2));
                assert!(
                    snapshot
                        .runs
                        .iter()
                        .all(|run| !run.active && run.status == "failed")
                );
                assert_eq!(snapshot.runs.len(), usize::from(stage >= 2));
            }
            let child_records = crate::transcript::read_records(&path).unwrap();
            assert!(child_records.iter().any(|record| matches!(&record.event,
                TranscriptEvent::SubagentLifecycle { run_id: recorded, status, detail: Some(detail), .. }
                if recorded == &run_id && status == "failed" && detail.contains(expected)
            )));
            if stage < 2 {
                assert!(
                    pool.state.lock().unwrap().swarms[&root_id]
                        .members
                        .is_empty()
                );
            }
        }
    }
}
