use super::*;
use crate::runtime_context::{RuntimeSource, SourceSpan};
use crate::transcript::PreparedLogicalCheckpoint;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct CommittedLogicalCheckpoint {
    pub protected_start_index: usize,
    pub owner: LogicalCheckpointRequestOwner,
}

/// Runs only at a completed tool-call batch.  The candidate is immutable until
/// its critical journal event has been acknowledged by the caller.
pub(super) async fn commit_pending_at_batch_boundary<C, E, Efut>(
    agent: &mut Agent<C>,
    protocol: ApiProtocol,
    turn_prelude: &[PromptMessage],
    _protected_start_index: usize,
    on_event: &mut E,
) -> Result<Option<CommittedLogicalCheckpoint>>
where
    C: Config + Clone,
    E: FnMut(AgentEvent) -> Efut,
    Efut: Future<Output = Result<()>>,
{
    commit_pending_at_boundary_with_automatic_token(
        agent,
        protocol,
        turn_prelude,
        _protected_start_index,
        None,
        on_event,
    )
    .await
}

/// The automatic token is supplied only by the immediate successor request
/// preparation.  Keeping this check beside the durable write prevents a stale
/// automatic request from being persisted at a later boundary.
pub(super) async fn commit_pending_at_boundary_with_automatic_token<C, E, Efut>(
    agent: &mut Agent<C>,
    protocol: ApiProtocol,
    turn_prelude: &[PromptMessage],
    _protected_start_index: usize,
    automatic_boundary_token: Option<u64>,
    on_event: &mut E,
) -> Result<Option<CommittedLogicalCheckpoint>>
where
    C: Config + Clone,
    E: FnMut(AgentEvent) -> Efut,
    Efut: Future<Output = Result<()>>,
{
    let Some(lease) = agent.logical_checkpoint_control.take_pending() else {
        return Ok(None);
    };

    if let LogicalCheckpointRequestOwner::Automatic { boundary_id } = lease.ownership
        && automatic_boundary_token != Some(boundary_id)
    {
        agent.logical_checkpoint_control.clear_lease(lease);
        bail!("automatic logical checkpoint owner does not match the current scheduler boundary");
    }

    let result = commit_pending(agent, protocol, turn_prelude, on_event).await;
    if result.is_ok() {}
    agent.logical_checkpoint_control.clear_lease(lease);
    result.map(|protected_start_index| {
        Some(CommittedLogicalCheckpoint {
            protected_start_index,
            owner: lease.ownership,
        })
    })
}

async fn commit_pending<C, E, Efut>(
    agent: &mut Agent<C>,
    protocol: ApiProtocol,
    turn_prelude: &[PromptMessage],
    on_event: &mut E,
) -> Result<usize>
where
    C: Config + Clone,
    E: FnMut(AgentEvent) -> Efut,
    Efut: Future<Output = Result<()>>,
{
    ensure!(
        agent.context_experiment_restore_point.is_none(),
        "logical checkpoint is unavailable during a context experiment"
    );
    ensure!(
        agent
            .context_scope_state
            .lock()
            .map_err(|_| anyhow!("context scope state poisoned"))?
            .active_experiment
            .is_none(),
        "logical checkpoint is unavailable during a context experiment"
    );
    let turn_id = agent.turn.turn_id;
    let segment_id = agent
        .runtime_snapshot
        .current_segment_id
        .ok_or_else(|| anyhow!("logical checkpoint requires a live segment"))?;
    ensure!(
        turn_id != 0 && agent.runtime_snapshot.current_turn_id == Some(turn_id),
        "logical checkpoint requires an active live turn"
    );
    ensure!(
        agent.turn.current_turn_start_index.is_some(),
        "logical checkpoint is unavailable for a restored live turn"
    );
    ensure!(
        agent.active_protocol() == protocol,
        "logical checkpoint protocol changed during the current turn"
    );
    crate::protocol_frames::validate_history_items_complete(
        &agent.history,
        agent.turn.current_turn_start_index,
    )
    .context("logical checkpoint requires a complete tool-call batch")?;

    let provider = agent.logical_checkpoint_candidate_provider.as_ref()
        .ok_or_else(|| anyhow!("logical checkpoint request reached a completed tool-call batch without a candidate provider"))?;
    let candidate =
        provider().context("failed to prepare transcript-backed logical checkpoint candidate")?;
    let prepared = validate_candidate(agent, candidate, turn_id, segment_id)?;

    // Validate the exact successor envelope with the actual model, protocol,
    // frozen evidence, prelude, and advertised tools before any durable write.
    let tools = agent.tool_definitions();
    let frozen = agent.turn.frozen_evidence.as_ref().map(|evidence| {
        crate::request_builder::FrozenEvidence {
            message: evidence.message.clone(),
            selected_ids: evidence.selected_ids.clone(),
        }
    });
    let policy = ProtectedContextPolicy::from_configured_reserve(
        None,
        effective_input_budget_tokens(agent.active_model_metadata(), &tools),
    );
    let build = build_request_with_policy(
        RequestBuilderInput {
            protocol,
            model_id: &agent.model,
            model: agent.active_model_metadata(),
            prelude: turn_prelude,
            snapshot: &prepared.snapshot,
            tools: &tools,
        },
        frozen.as_ref(),
        Some(policy),
    )
    .context("logical checkpoint successor request does not fit the active model budget")?;
    let classification = build.budget.request_classification();
    ensure!(
        classification.safe,
        "logical checkpoint successor request exceeds hard budget (request {}, hard limit {})",
        build.budget.estimated_request_tokens,
        classification.hard_request_limit,
    );

    on_event(AgentEvent::LogicalCheckpoint {
        expected_journal_frontier: prepared.expected_journal_frontier,
        expected_branch_id: prepared.expected_branch_id.clone(),
        event: prepared.event.clone(),
    })
    .await?;

    // Do not use scope restoration here: it deliberately resets ephemeral turn
    // state. Checkpoint replacement preserves it while installing the exact
    // acknowledged successor envelope.
    let protected_start_index = prepared.protected_start_index;
    agent.protocol_frames = prepared.protocol_frames;
    agent.history = prepared.history;
    agent.runtime_snapshot = prepared.snapshot;
    agent.turn.workflow = prepared.workflow;
    agent.turn.current_turn_start_index = Some(protected_start_index);
    agent.clear_active_epoch();
    Ok(protected_start_index)
}

struct PreparedCheckpointEnvelope {
    expected_journal_frontier: u64,
    expected_branch_id: String,
    event: crate::transcript::LogicalCheckpointEventV1,
    protocol_frames: Vec<crate::protocol_frames::ProtocolFrame>,
    history: Vec<HistoryItem>,
    snapshot: RuntimeSnapshot,
    workflow: WorkflowState,
    protected_start_index: usize,
}

fn validate_candidate<C: Config>(
    agent: &Agent<C>,
    candidate: PreparedLogicalCheckpoint,
    turn_id: u64,
    segment_id: u64,
) -> Result<PreparedCheckpointEnvelope> {
    let PreparedLogicalCheckpoint {
        expected_journal_frontier,
        expected_branch_id,
        event,
        projected_snapshot,
        projected_protocol_frames,
        projected_workflow,
    } = candidate;
    let expected_leaf = expected_journal_frontier
        .checked_add(1)
        .ok_or_else(|| anyhow!("logical checkpoint journal frontier overflow"))?;
    let snapshot_frames = projected_snapshot.active_protocol_frames();
    ensure!(
        projected_protocol_frames == snapshot_frames,
        "logical checkpoint protocol frames do not exactly match the projected snapshot"
    );
    let history = crate::protocol_frames::history_items_from_frames(&projected_protocol_frames);
    crate::protocol_frames::validate_history_items_complete(&history, Some(0))
        .context("logical checkpoint successor protocol is incomplete")?;
    validate_protocol_frame_correspondence(&projected_protocol_frames, &projected_snapshot)?;
    projected_snapshot.validate_references()?;
    ensure!(
        event.turn_id == turn_id,
        "logical checkpoint candidate targets a different turn"
    );
    ensure!(
        event.previous_segment_id == segment_id,
        "logical checkpoint candidate targets a stale segment"
    );
    ensure!(
        expected_branch_id == agent.runtime_snapshot.active_context.branch_id
            && expected_branch_id == projected_snapshot.active_context.branch_id,
        "logical checkpoint candidate targets a different branch"
    );
    ensure!(
        event.context_scope_revision == agent.runtime_snapshot.context_scope_revision
            && event.context_scope_revision == projected_snapshot.context_scope_revision,
        "logical checkpoint candidate has a stale context scope"
    );
    ensure!(
        projected_snapshot.leaf_sequence == Some(expected_leaf),
        "logical checkpoint successor has an invalid journal leaf"
    );
    ensure!(
        projected_snapshot.current_turn_id == Some(turn_id),
        "logical checkpoint successor lost the active turn"
    );
    ensure!(
        event.segment_id
            == segment_id
                .checked_add(1)
                .ok_or_else(|| anyhow!("logical checkpoint segment overflow"))?
            && projected_snapshot.current_segment_id
                == Some(
                    segment_id
                        .checked_add(1)
                        .ok_or_else(|| anyhow!("logical checkpoint segment overflow"))?
                ),
        "logical checkpoint successor has an invalid segment"
    );
    ensure!(
        projected_snapshot.latest_model.as_deref() == Some(agent.model.as_str()),
        "logical checkpoint candidate targets a different model"
    );
    let workflow = projected_workflow.unwrap_or_default();
    let workflow = WorkflowState {
        todos: workflow.todos,
        auto_continue: workflow.auto_continue,
    };
    ensure!(
        workflow == agent.turn.workflow,
        "logical checkpoint projected workflow does not exactly match the live workflow"
    );

    let summary = crate::transcript::render_checkpoint_v1(&event)?;
    let continuation = crate::transcript::render_checkpoint_continuation_v1(&event);
    let suffix = [
        crate::protocol_frames::ProtocolFrameItem::ContextSummary { text: summary },
        crate::protocol_frames::ProtocolFrameItem::InternalContinuation { text: continuation },
    ];
    let suffix_starts = history
        .windows(suffix.len())
        .enumerate()
        .filter_map(|(index, frames)| {
            (frames
                .iter()
                .map(protocol_frame_item_from_history_item)
                .eq(suffix.iter().cloned()))
            .then_some(index)
        })
        .collect::<Vec<_>>();
    ensure!(
        suffix_starts.len() == 1 && suffix_starts[0] + suffix.len() == history.len(),
        "logical checkpoint successor must end with one exact summary and continuation suffix"
    );
    let protected_start_index = suffix_starts[0];
    let suffix_frames = &projected_protocol_frames[protected_start_index..];
    for (frame, expected_source_id) in suffix_frames.iter().zip([
        format!("{}:summary", event.checkpoint_id),
        format!("{}:continuation", event.checkpoint_id),
    ]) {
        let provenance = frame
            .source_provenance
            .as_ref()
            .ok_or_else(|| anyhow!("logical checkpoint suffix frame has no provenance"))?;
        ensure!(
            provenance.source == RuntimeSource::Transcript
                && provenance.source_id.as_deref() == Some(expected_source_id.as_str())
                && provenance.source_span == Some(SourceSpan::new(expected_leaf, expected_leaf)?),
            "logical checkpoint suffix provenance does not match its event"
        );
    }
    let closure = event
        .covered_source_spans
        .iter()
        .map(|span| SourceSpan::new(span.start_sequence, span.end_sequence))
        .collect::<Result<Vec<_>>>()?;
    ensure!(
        closure.iter().all(|closed| projected_snapshot
            .compaction
            .retired_source_spans
            .iter()
            .any(|retired| retired.start_sequence <= closed.start_sequence
                && retired.end_sequence >= closed.end_sequence)),
        "logical checkpoint successor did not retire its closed source closure"
    );
    ensure!(
        projected_snapshot
            .frames
            .iter()
            .filter(
                |frame| frame.visibility == crate::runtime_context::FrameVisibility::Active
                    && frame.provenance.source == RuntimeSource::Transcript
            )
            .all(|frame| frame
                .provenance
                .source_span
                .is_none_or(|span| !closure.iter().any(|closed| span.overlaps(*closed)))),
        "logical checkpoint successor retains closed raw source material as active"
    );
    Ok(PreparedCheckpointEnvelope {
        expected_journal_frontier,
        expected_branch_id,
        event,
        protocol_frames: projected_protocol_frames,
        history,
        snapshot: projected_snapshot,
        workflow,
        protected_start_index,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use async_openai::config::OpenAIConfig;
    use serde_json::{Value, json};
    use std::collections::HashMap;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    struct OracleR2SuccessorLimitTool;

    #[async_trait::async_trait]
    impl crate::tool::ToolHandler for OracleR2SuccessorLimitTool {
        fn name(&self) -> &str {
            "oracle_r2_successor_limit"
        }

        fn description(&self) -> &str {
            "Advertised Oracle R2 successor-limit regression tool"
        }

        fn parameters(&self) -> Value {
            json!({
                "type": "object",
                "properties": {
                    "oracle_r2_payload": {
                        "type": "string",
                        "description": "oracle-r2-successor-limit ".repeat(500),
                    }
                },
                "required": ["oracle_r2_payload"],
                "additionalProperties": false,
            })
        }

        async fn execute(&self, _args: Value) -> Result<Value> {
            Ok(Value::Null)
        }
    }

    fn phase1c_checkpoint_agent() -> (Agent<OpenAIConfig>, Vec<PromptMessage>) {
        let client = Client::with_config(
            OpenAIConfig::new()
                .with_api_base("https://api.openai.com/v1")
                .with_api_key("test"),
        );
        let mut agent = Agent::new(client, "phase1c", 4, 4);
        agent.set_model_catalog(HashMap::from([(
            "phase1c".into(),
            ModelRequestMetadata {
                context_window: Some(32_000),
                max_output_tokens: Some(2_000),
                supports_tools: true,
                ..Default::default()
            },
        )]));
        agent.set_logical_checkpoint_config(LogicalCheckpointConfig {
            enabled: true,
            automatic: true,
            ..Default::default()
        });
        let prelude = agent.prepare_turn_prelude("continue phase 1C checkpoint");
        agent.history = vec![HistoryItem::user("current request")];
        agent.protocol_frames = crate::protocol_frames::history_items_to_frames(&agent.history);
        agent.turn.current_turn_start_index = Some(0);
        agent.runtime_snapshot.current_turn_id = Some(agent.turn.turn_id);
        agent.runtime_snapshot.current_segment_id = Some(1);
        agent.runtime_snapshot.leaf_sequence = Some(9);
        agent.runtime_snapshot.latest_model = Some(agent.model.clone());
        (agent, prelude)
    }

    fn prepared_checkpoint_for(agent: &Agent<OpenAIConfig>) -> PreparedLogicalCheckpoint {
        let previous_segment_id = agent
            .runtime_snapshot
            .current_segment_id
            .expect("live segment");
        let boundary_sequence = agent
            .runtime_snapshot
            .leaf_sequence
            .expect("journal frontier");
        let segment_id = previous_segment_id + 1;
        let leaf = boundary_sequence + 1;
        let event = crate::transcript::LogicalCheckpointEventV1 {
            schema_version: 1,
            checkpoint_id: "phase1c-checkpoint".into(),
            turn_id: agent.turn.turn_id,
            previous_segment_id,
            segment_id,
            previous_checkpoint_id: None,
            boundary_sequence,
            context_scope_revision: agent.runtime_snapshot.context_scope_revision,
            covered_source_spans: Vec::new(),
            retained_items: Vec::new(),
        };
        let summary = crate::transcript::render_checkpoint_v1(&event).expect("summary renders");
        let continuation = crate::transcript::render_checkpoint_continuation_v1(&event);
        let mut frames = vec![
            crate::protocol_frames::ProtocolFrame::derived(
                crate::protocol_frames::ProtocolFrameItem::ContextSummary { text: summary },
            ),
            crate::protocol_frames::ProtocolFrame::derived(
                crate::protocol_frames::ProtocolFrameItem::InternalContinuation {
                    text: continuation,
                },
            ),
        ];
        for (index, frame) in frames.iter_mut().enumerate() {
            frame.history_index = index;
        }
        for (frame, source_id) in frames.iter_mut().zip([
            format!("{}:summary", event.checkpoint_id),
            format!("{}:continuation", event.checkpoint_id),
        ]) {
            frame.source_provenance = Some(
                crate::runtime_context::RuntimeFrameProvenance::new(RuntimeSource::Transcript)
                    .with_source_id(&source_id)
                    .with_span(SourceSpan::new(leaf, leaf).expect("valid source span")),
            );
        }
        let mut snapshot =
            RuntimeSnapshot::new(agent.runtime_snapshot.active_context.branch_id.clone());
        snapshot.current_turn_id = Some(agent.turn.turn_id);
        snapshot.current_segment_id = Some(segment_id);
        snapshot.leaf_sequence = Some(leaf);
        snapshot.latest_model = Some(agent.model.clone());
        snapshot.frames = frames
            .iter()
            .enumerate()
            .map(|(index, frame)| {
                let mut runtime =
                    super::super::runtime_frame_from_protocol_frame(frame, index as u32);
                runtime.provenance = frame
                    .source_provenance
                    .clone()
                    .expect("checkpoint suffix has provenance");
                runtime
            })
            .collect();
        for (frame, runtime) in frames.iter_mut().zip(&snapshot.frames) {
            frame.runtime_frame_id = Some(runtime.id);
        }
        PreparedLogicalCheckpoint {
            expected_journal_frontier: boundary_sequence,
            expected_branch_id: agent.runtime_snapshot.active_context.branch_id.clone(),
            event,
            projected_snapshot: snapshot,
            projected_protocol_frames: frames,
            projected_workflow: Some(crate::transcript::CheckpointWorkflowProjection {
                todos: agent.turn.workflow.todos.clone(),
                auto_continue: agent.turn.workflow.auto_continue.clone(),
            }),
        }
    }

    fn install_active_epoch(
        agent: &mut Agent<OpenAIConfig>,
        protocol: ApiProtocol,
        prelude: &[PromptMessage],
    ) {
        let preview = agent
            .preview_active_epoch(protocol, prelude, &agent.tool_definitions())
            .expect("fixture can preview an active epoch");
        agent.active_epoch = Some(preview.epoch);
    }

    fn prospective_successor_budget(
        agent: &Agent<OpenAIConfig>,
        protocol: ApiProtocol,
        prelude: &[PromptMessage],
        candidate: &PreparedLogicalCheckpoint,
    ) -> Result<crate::request_builder::BudgetReport> {
        let tools = agent.tool_definitions_for_test();
        let policy = ProtectedContextPolicy::from_configured_reserve(
            None,
            effective_input_budget_tokens(agent.active_model_metadata(), &tools),
        );
        build_request_with_policy(
            RequestBuilderInput {
                protocol,
                model_id: &agent.model,
                model: agent.active_model_metadata(),
                prelude,
                snapshot: &candidate.projected_snapshot,
                tools: &tools,
            },
            None,
            Some(policy),
        )
        .map(|build| build.budget)
    }

    fn tune_oracle_r2_successor_limit(
        agent: &mut Agent<OpenAIConfig>,
        protocol: ApiProtocol,
        prelude: &[PromptMessage],
        candidate: &PreparedLogicalCheckpoint,
    ) -> crate::request_builder::BudgetReport {
        let metadata = agent
            .model_catalog
            .get_mut("phase1c")
            .expect("fixture metadata");
        metadata.effective_input_limit_tokens = Some(30_000);
        let baseline = prospective_successor_budget(agent, protocol, prelude, candidate)
            .expect("Oracle R2 baseline successor builds");
        let input_limit = baseline.estimated_request_tokens + 2_049;
        agent
            .model_catalog
            .get_mut("phase1c")
            .expect("fixture metadata")
            .effective_input_limit_tokens = Some(input_limit);
        prospective_successor_budget(agent, protocol, prelude, candidate)
            .expect("Oracle R2 tuned successor builds")
    }

    fn assert_live_state(
        agent: &Agent<OpenAIConfig>,
        history: &[HistoryItem],
        frames: &[crate::protocol_frames::ProtocolFrame],
        snapshot: &RuntimeSnapshot,
        workflow: &WorkflowState,
        start: Option<usize>,
        epoch: &Option<ActiveEpoch>,
    ) {
        assert_eq!(&agent.history, history);
        assert_eq!(&agent.protocol_frames, frames);
        assert_eq!(&agent.runtime_snapshot, snapshot);
        assert_eq!(&agent.turn.workflow, workflow);
        assert_eq!(agent.turn.current_turn_start_index, start);
        assert_eq!(&agent.active_epoch, epoch);
    }

    #[tokio::test]
    async fn checkpoint_legacy_writer_commits_exact_prepared_envelope_for_protocols() {
        for protocol in [ApiProtocol::Responses, ApiProtocol::Completions] {
            let (mut agent, prelude) = phase1c_checkpoint_agent();
            agent.set_model_protocols(HashMap::from([("phase1c".into(), protocol)]));
            install_active_epoch(&mut agent, protocol, &prelude);
            let candidate = prepared_checkpoint_for(&agent);
            let expected_history = crate::protocol_frames::history_items_from_frames(
                &candidate.projected_protocol_frames,
            );
            let expected_frames = candidate.projected_protocol_frames.clone();
            let expected_snapshot = candidate.projected_snapshot.clone();
            let expected_workflow = agent.turn.workflow.clone();
            agent
                .set_logical_checkpoint_candidate_provider(Arc::new(move || Ok(candidate.clone())));
            let run = agent.logical_checkpoint_control.begin_run();
            assert_eq!(
                agent.request_logical_checkpoint(),
                LogicalCheckpointRequestOutcome::Queued
            );
            let mut logical_events = 0;

            let committed = commit_pending_at_batch_boundary(
                &mut agent,
                protocol,
                &prelude,
                99,
                &mut |event| {
                    logical_events +=
                        usize::from(matches!(event, AgentEvent::LogicalCheckpoint { .. }));
                    std::future::ready(Ok(()))
                },
            )
            .await
            .expect("legacy checkpoint writer commits")
            .expect("pending checkpoint returns its owner and boundary");

            assert_eq!(logical_events, 1);
            assert_eq!(committed.protected_start_index, 0);
            assert_eq!(committed.owner, LogicalCheckpointRequestOwner::Manual);
            // commit_pending validates this exact prepared snapshot before the
            // acknowledgement and installs that same value afterwards. A
            // distinct post-install unsafe envelope cannot arise at this seam;
            // the protocol-stream lane owns the subsequent re-preview.
            assert_eq!(agent.history, expected_history);
            assert_eq!(agent.protocol_frames, expected_frames);
            assert_eq!(agent.runtime_snapshot, expected_snapshot);
            assert_eq!(agent.turn.workflow, expected_workflow);
            assert_eq!(agent.turn.current_turn_start_index, Some(0));
            assert!(agent.active_epoch.is_none());
            assert_eq!(agent.turn.automatic_checkpoint.commits, 0);
            assert_eq!(
                agent.request_logical_checkpoint(),
                LogicalCheckpointRequestOutcome::Queued,
                "successful transaction releases its lease"
            );
            drop(run);
        }
    }

    #[tokio::test]
    async fn checkpoint_legacy_writer_accepts_compatible_successor_above_input_budget() {
        let protocol = ApiProtocol::Responses;
        let (mut agent, prelude) = phase1c_checkpoint_agent();
        agent.set_model_protocols(HashMap::from([("phase1c".into(), protocol)]));
        agent.register_tool(OracleR2SuccessorLimitTool);
        let candidate = prepared_checkpoint_for(&agent);
        let budget = tune_oracle_r2_successor_limit(&mut agent, protocol, &prelude, &candidate);
        let classification = budget.request_classification();
        assert!(budget.estimated_tools_tokens > 2_048, "{budget:?}");
        assert!(
            budget.estimated_request_tokens > budget.input_budget_tokens,
            "{budget:?}"
        );
        assert!(
            classification.safe,
            "oracle successor must pass hard-limit admission: {budget:?}"
        );
        assert!(
            budget.estimated_request_tokens <= classification.hard_request_limit,
            "{budget:?}"
        );
        println!(
            "Oracle R2 budget: tools={}, input={}, request={}, hard_limit={}",
            budget.estimated_tools_tokens,
            budget.input_budget_tokens,
            budget.estimated_request_tokens,
            classification.hard_request_limit,
        );

        let expected_history =
            crate::protocol_frames::history_items_from_frames(&candidate.projected_protocol_frames);
        let expected_frames = candidate.projected_protocol_frames.clone();
        let expected_snapshot = candidate.projected_snapshot.clone();
        let expected_workflow = agent.turn.workflow.clone();
        agent.set_logical_checkpoint_candidate_provider(Arc::new(move || Ok(candidate.clone())));
        let run = agent.logical_checkpoint_control.begin_run();
        assert_eq!(
            agent.request_logical_checkpoint(),
            LogicalCheckpointRequestOutcome::Queued
        );
        let mut events = 0;

        let committed =
            commit_pending_at_batch_boundary(&mut agent, protocol, &prelude, 99, &mut |event| {
                events += usize::from(matches!(event, AgentEvent::LogicalCheckpoint { .. }));
                std::future::ready(Ok(()))
            })
            .await
            .expect("Oracle R2 compatible legacy envelope commits")
            .expect("pending checkpoint commits");

        assert_eq!(events, 1);
        assert_eq!(committed.protected_start_index, 0);
        assert_eq!(agent.history, expected_history);
        assert_eq!(agent.protocol_frames, expected_frames);
        assert_eq!(agent.runtime_snapshot, expected_snapshot);
        assert_eq!(agent.turn.workflow, expected_workflow);
        assert!(agent.active_epoch.is_none());
        assert_eq!(agent.turn.automatic_checkpoint.commits, 0);
        drop(run);
    }

    #[tokio::test]
    async fn phase1c_checkpoint_candidate_rejections_preserve_live_state_and_release_lease() {
        for case in ["provider", "lineage", "history-closure"] {
            let (mut agent, prelude) = phase1c_checkpoint_agent();
            install_active_epoch(&mut agent, ApiProtocol::Responses, &prelude);
            let mut candidate = prepared_checkpoint_for(&agent);
            let provider_calls = Arc::new(AtomicUsize::new(0));
            match case {
                "provider" => agent.set_logical_checkpoint_candidate_provider({
                    let provider_calls = Arc::clone(&provider_calls);
                    Arc::new(move || {
                        provider_calls.fetch_add(1, Ordering::SeqCst);
                        Err(anyhow!("candidate provider failed"))
                    })
                }),
                "lineage" => {
                    candidate.event.previous_segment_id += 1;
                    agent.set_logical_checkpoint_candidate_provider({
                        let provider_calls = Arc::clone(&provider_calls);
                        Arc::new(move || {
                            provider_calls.fetch_add(1, Ordering::SeqCst);
                            Ok(candidate.clone())
                        })
                    });
                }
                "history-closure" => {
                    candidate.event.covered_source_spans =
                        vec![crate::transcript::LogicalCheckpointSourceSpanV1 {
                            start_sequence: 1,
                            end_sequence: 1,
                        }];
                    agent.set_logical_checkpoint_candidate_provider({
                        let provider_calls = Arc::clone(&provider_calls);
                        Arc::new(move || {
                            provider_calls.fetch_add(1, Ordering::SeqCst);
                            Ok(candidate.clone())
                        })
                    });
                }
                _ => unreachable!(),
            }
            let history = agent.history.clone();
            let frames = agent.protocol_frames.clone();
            let snapshot = agent.runtime_snapshot.clone();
            let workflow = agent.turn.workflow.clone();
            let start = agent.turn.current_turn_start_index;
            let epoch = agent.active_epoch.clone();
            let boundary = agent.turn.automatic_checkpoint.begin_complete_boundary();
            agent.turn.automatic_checkpoint.mark_attempted(boundary);
            let scheduler = agent.turn.automatic_checkpoint.clone();
            let run = agent.logical_checkpoint_control.begin_run();
            assert_eq!(
                agent.logical_checkpoint_control.request_automatic(boundary),
                LogicalCheckpointRequestOutcome::Queued
            );
            let mut events = 0;

            commit_pending_at_boundary_with_automatic_token(
                &mut agent,
                ApiProtocol::Responses,
                &prelude,
                0,
                Some(boundary),
                &mut |event| {
                    events += usize::from(matches!(event, AgentEvent::LogicalCheckpoint { .. }));
                    std::future::ready(Ok(()))
                },
            )
            .await
            .expect_err(case);

            assert_eq!(provider_calls.load(Ordering::SeqCst), 1, "{case}");
            assert_eq!(events, 0, "{case}");
            assert_live_state(
                &agent, &history, &frames, &snapshot, &workflow, start, &epoch,
            );
            assert_eq!(agent.turn.automatic_checkpoint, scheduler, "{case}");
            assert_eq!(agent.turn.automatic_checkpoint.commits, 0);
            assert_eq!(
                agent
                    .logical_checkpoint_control
                    .request_automatic(boundary + 1),
                LogicalCheckpointRequestOutcome::Queued
            );
            drop(run);
        }
    }

    #[tokio::test]
    async fn phase1c_checkpoint_successor_preflight_rejections_precede_acknowledgement() {
        for case in ["build", "high-watermark"] {
            let (mut agent, prelude) = phase1c_checkpoint_agent();
            install_active_epoch(&mut agent, ApiProtocol::Responses, &prelude);
            if case == "build" {
                agent.set_model_catalog(HashMap::from([(
                    "phase1c".into(),
                    ModelRequestMetadata {
                        context_window: Some(1),
                        max_output_tokens: Some(1),
                        supports_tools: true,
                        ..Default::default()
                    },
                )]));
            } else {
                let metadata = agent
                    .model_catalog
                    .get_mut("phase1c")
                    .expect("fixture metadata");
                metadata.effective_input_limit_tokens = Some(1_280);
            }
            let candidate = prepared_checkpoint_for(&agent);
            agent
                .set_logical_checkpoint_candidate_provider(Arc::new(move || Ok(candidate.clone())));
            let history = agent.history.clone();
            let frames = agent.protocol_frames.clone();
            let snapshot = agent.runtime_snapshot.clone();
            let workflow = agent.turn.workflow.clone();
            let start = agent.turn.current_turn_start_index;
            let epoch = agent.active_epoch.clone();
            let boundary = agent.turn.automatic_checkpoint.begin_complete_boundary();
            agent.turn.automatic_checkpoint.mark_attempted(boundary);
            let scheduler = agent.turn.automatic_checkpoint.clone();
            let run = agent.logical_checkpoint_control.begin_run();
            assert_eq!(
                agent.logical_checkpoint_control.request_automatic(boundary),
                LogicalCheckpointRequestOutcome::Queued
            );
            let mut events = 0;

            commit_pending_at_boundary_with_automatic_token(
                &mut agent,
                ApiProtocol::Responses,
                &prelude,
                0,
                Some(boundary),
                &mut |event| {
                    events += usize::from(matches!(event, AgentEvent::LogicalCheckpoint { .. }));
                    std::future::ready(Ok(()))
                },
            )
            .await
            .expect_err(case);

            assert_eq!(events, 0, "{case}");
            assert_live_state(
                &agent, &history, &frames, &snapshot, &workflow, start, &epoch,
            );
            assert_eq!(agent.turn.automatic_checkpoint, scheduler, "{case}");
            assert_eq!(
                agent
                    .logical_checkpoint_control
                    .request_automatic(boundary + 1),
                LogicalCheckpointRequestOutcome::Queued
            );
            drop(run);
        }
    }

    #[tokio::test]
    async fn phase1c_checkpoint_acknowledgement_failure_preserves_state_and_releases_lease() {
        let (mut agent, prelude) = phase1c_checkpoint_agent();
        install_active_epoch(&mut agent, ApiProtocol::Responses, &prelude);
        let candidate = prepared_checkpoint_for(&agent);
        agent.set_logical_checkpoint_candidate_provider(Arc::new(move || Ok(candidate.clone())));
        let history = agent.history.clone();
        let frames = agent.protocol_frames.clone();
        let snapshot = agent.runtime_snapshot.clone();
        let workflow = agent.turn.workflow.clone();
        let start = agent.turn.current_turn_start_index;
        let epoch = agent.active_epoch.clone();
        let boundary = agent.turn.automatic_checkpoint.begin_complete_boundary();
        agent.turn.automatic_checkpoint.mark_attempted(boundary);
        let scheduler = agent.turn.automatic_checkpoint.clone();
        let run = agent.logical_checkpoint_control.begin_run();
        assert_eq!(
            agent.logical_checkpoint_control.request_automatic(boundary),
            LogicalCheckpointRequestOutcome::Queued
        );
        let mut events = 0;

        commit_pending_at_boundary_with_automatic_token(
            &mut agent,
            ApiProtocol::Responses,
            &prelude,
            0,
            Some(boundary),
            &mut |event| {
                events += usize::from(matches!(event, AgentEvent::LogicalCheckpoint { .. }));
                std::future::ready(Err(anyhow!("acknowledgement failed")))
            },
        )
        .await
        .expect_err("acknowledgement failure rejects before installation");

        assert_eq!(events, 1);
        assert_live_state(
            &agent, &history, &frames, &snapshot, &workflow, start, &epoch,
        );
        assert_eq!(agent.turn.automatic_checkpoint, scheduler);
        assert_eq!(agent.turn.automatic_checkpoint.commits, 0);
        assert_eq!(
            agent
                .logical_checkpoint_control
                .request_automatic(boundary + 1),
            LogicalCheckpointRequestOutcome::Queued
        );
        drop(run);
    }

    #[tokio::test]
    async fn phase1c_checkpoint_boundary_and_tool_history_rejections_skip_provider_and_event() {
        for case in ["boundary", "incomplete-tools"] {
            let (mut agent, prelude) = phase1c_checkpoint_agent();
            install_active_epoch(&mut agent, ApiProtocol::Responses, &prelude);
            if case == "incomplete-tools" {
                agent.history = vec![HistoryItem::AssistantToolCalls {
                    text: None,
                    calls: vec![crate::protocol_frames::ProtocolToolCall {
                        call_id: "unfinished".into(),
                        name: "tool".into(),
                        arguments_json: "{}".into(),
                    }],
                }];
                agent.protocol_frames =
                    crate::protocol_frames::history_items_to_frames(&agent.history);
            }
            let provider_calls = Arc::new(AtomicUsize::new(0));
            let candidate = prepared_checkpoint_for(&agent);
            agent.set_logical_checkpoint_candidate_provider({
                let provider_calls = Arc::clone(&provider_calls);
                Arc::new(move || {
                    provider_calls.fetch_add(1, Ordering::SeqCst);
                    Ok(candidate.clone())
                })
            });
            let history = agent.history.clone();
            let frames = agent.protocol_frames.clone();
            let snapshot = agent.runtime_snapshot.clone();
            let workflow = agent.turn.workflow.clone();
            let start = agent.turn.current_turn_start_index;
            let epoch = agent.active_epoch.clone();
            let boundary = agent.turn.automatic_checkpoint.begin_complete_boundary();
            agent.turn.automatic_checkpoint.mark_attempted(boundary);
            let scheduler = agent.turn.automatic_checkpoint.clone();
            let run = agent.logical_checkpoint_control.begin_run();
            assert_eq!(
                agent.logical_checkpoint_control.request_automatic(boundary),
                LogicalCheckpointRequestOutcome::Queued
            );
            let mut events = 0;
            let token = (case == "boundary")
                .then_some(boundary + 1)
                .or(Some(boundary));

            commit_pending_at_boundary_with_automatic_token(
                &mut agent,
                ApiProtocol::Responses,
                &prelude,
                0,
                token,
                &mut |event| {
                    events += usize::from(matches!(event, AgentEvent::LogicalCheckpoint { .. }));
                    std::future::ready(Ok(()))
                },
            )
            .await
            .expect_err(case);

            assert_eq!(provider_calls.load(Ordering::SeqCst), 0, "{case}");
            assert_eq!(events, 0, "{case}");
            assert_live_state(
                &agent, &history, &frames, &snapshot, &workflow, start, &epoch,
            );
            assert_eq!(agent.turn.automatic_checkpoint, scheduler, "{case}");
            assert_eq!(
                agent
                    .logical_checkpoint_control
                    .request_automatic(boundary + 1),
                LogicalCheckpointRequestOutcome::Queued
            );
            drop(run);
        }
    }
}
