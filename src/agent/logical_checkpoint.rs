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
    if result.is_ok() {
        agent
            .turn
            .automatic_checkpoint
            .mark_committed(lease.ownership);
    }
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
        agent.compaction_config.protected_reserve_tokens,
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
    ensure!(
        build.budget.estimated_request_tokens <= build.budget.input_budget_tokens,
        "logical checkpoint successor request exceeds the active input budget"
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
