use super::*;
use crate::protocol_frames::{ProtocolFrameItem, ToolCallGroupStatus, analyze_history_items};
use crate::runtime_context::{
    FrameVisibility, RuntimeFrameId, RuntimeSnapshot, RuntimeSource, SourceSpan,
};
use anyhow::ensure;
use futures_util::FutureExt;
use futures_util::future::BoxFuture;
use std::collections::{BTreeMap, BTreeSet};
use std::fmt;

#[derive(Debug)]
enum NoProgressSelection {
    NoHistoricalItems,
    NoSafeBoundary,
}

impl fmt::Display for NoProgressSelection {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::NoHistoricalItems => NO_HISTORICAL_ITEMS_FOR_COMPACTION,
            Self::NoSafeBoundary => NO_OLDER_ITEMS_AFTER_TAIL,
        })
    }
}

impl std::error::Error for NoProgressSelection {}

#[derive(Debug, Clone)]
pub(super) struct CompactionSelection {
    pub(super) previous_summary: Option<String>,
    pub(super) head_for_summary: Vec<HistoryItem>,
    pub(super) tail_items: Vec<HistoryItem>,
    pub(super) tail_start_index: usize,
    /// Every runtime frame retired by this compact (protocol prefix + dependents).
    pub(super) retired_frame_ids: Vec<RuntimeFrameId>,
    pub(super) retired_source_spans: Vec<SourceSpan>,
}

enum CompactionSelectionResult {
    Selected(CompactionSelection),
    NoProgress(CompactionNoProgress),
}

type EventCallback<'a> = dyn FnMut(AgentEvent) -> BoxFuture<'a, Result<()>> + Send + 'a;

enum PressureAdmissionError {
    /// Compaction installed but the successor still cannot fit the hard budget.
    /// History reclaim is kept so the session does not regress to the oversized
    /// pre-compact state.
    BudgetExhausted { detail: String },
    /// Protocol/runtime inconsistency after install; restore the checkpoint.
    Technical(anyhow::Error),
}

pub(super) struct PreparedRequestBuild {
    pub(super) protected_start_index: usize,
    pub(super) build: crate::request_builder::BuildResult,
    /// Retained through physical retries and committed at the send boundary.
    pub(super) epoch_preview: super::ActiveEpochPreview,
}

struct PreparedCompaction {
    retained_items: usize,
    event: ContextCompactionEvent,
    snapshot: RuntimeSnapshot,
    protocol_frames: Vec<crate::protocol_frames::ProtocolFrame>,
    history: Vec<HistoryItem>,
    current_turn_start_index: Option<usize>,
}

pub(super) async fn compact_session_stream_async<C, E, Efut, S, D>(
    agent: &mut Agent<C>,
    mut on_event: E,
    mut on_start: S,
    mut on_delta: D,
) -> Result<ManualCompactionOutcome>
where
    C: Config + Clone,
    E: FnMut(AgentEvent) -> Efut + Send,
    Efut: Future<Output = Result<()>> + Send,
    S: FnMut() -> Result<()> + Send,
    D: FnMut(&str) -> Result<()> + Send,
{
    let trigger = CompactionTrigger::Manual;
    on_event(AgentEvent::ContextCompactionStarted { trigger }).await?;
    if let Err(error) = on_start() {
        let _ = on_event(AgentEvent::ContextCompactionFailed { trigger }).await;
        return Err(error);
    }
    let mut on_event = |event| Box::pin(on_event(event)) as BoxFuture<'_, Result<()>>;
    attempt_compaction(agent, trigger, &mut on_event, Some(&mut on_delta)).await
}

/// Pressure uses the same candidate transaction as an explicit `/compact`.
/// The caller consumes its ephemeral frontier before entering this fallible
/// operation; this function therefore never mutates live state until the
/// durable callback has acknowledged the candidate.
pub(super) fn compact_for_request_pressure<'a, C, E, Efut>(
    agent: &'a mut Agent<C>,
    protocol: ApiProtocol,
    turn_prelude: &'a [PromptMessage],
    tool_definitions: &'a [crate::request_builder::ToolSpec],
    on_event: &'a mut E,
) -> BoxFuture<'a, Result<PreparedRequestBuild>>
where
    C: Config + Clone + 'a,
    E: FnMut(AgentEvent) -> Efut + Send + 'a,
    Efut: Future<Output = Result<()>> + Send + 'a,
{
    async move {
        let mut on_event = |event| Box::pin(on_event(event)) as BoxFuture<'_, Result<()>>;
        compact_for_request_pressure_with_callback(
            agent,
            protocol,
            turn_prelude,
            tool_definitions,
            &mut on_event,
        )
        .await
    }
    .boxed()
}

struct PreCompactionCheckpoint {
    snapshot: RuntimeSnapshot,
    protocol_frames: Vec<crate::protocol_frames::ProtocolFrame>,
    history: Vec<HistoryItem>,
    turn_start_index: Option<usize>,
    active_epoch: Option<super::ActiveEpoch>,
}

impl PreCompactionCheckpoint {
    fn capture<C: Config + Clone>(agent: &Agent<C>) -> Self {
        Self {
            snapshot: agent.runtime_snapshot.clone(),
            protocol_frames: agent.protocol_frames.clone(),
            history: agent.history.clone(),
            turn_start_index: agent.turn.current_turn_start_index,
            active_epoch: agent.active_epoch.clone(),
        }
    }

    fn restore_full<C: Config + Clone>(self, agent: &mut Agent<C>) {
        agent.runtime_snapshot = self.snapshot;
        agent.protocol_frames = self.protocol_frames;
        agent.history = self.history;
        agent.turn.current_turn_start_index = self.turn_start_index;
        agent.active_epoch = self.active_epoch;
    }

    fn restore_protocol_only<C: Config + Clone>(self, agent: &mut Agent<C>) {
        agent.runtime_snapshot = self.snapshot;
        agent.protocol_frames = self.protocol_frames;
    }
}

/// Shared select → summarize → prepare path for pressure and manual compact.
/// On success the agent still holds only the healed working snapshot; callers
/// decide when to install the prepared candidate and how to roll back.
async fn prepare_compaction_candidate<C>(
    agent: &mut Agent<C>,
    trigger: CompactionTrigger,
    on_event: &mut EventCallback<'_>,
    on_delta: Option<&mut (dyn FnMut(&str) -> Result<()> + Send + '_)>,
) -> Result<Result<(PreCompactionCheckpoint, PreparedCompaction), CompactionNoProgress>>
where
    C: Config + Clone,
{
    agent.refresh_runtime_snapshot_from_provider()?;
    validate_compaction_runtime_state(agent)?;
    let selection_config = aggressive_selection_config(&agent.compaction_config);
    let mut healed = healed_snapshot_for_selection(&agent.runtime_snapshot);
    // Pressure reclaim pins only the hard core so completed mid-turn tools stay
    // reclaimable. Manual compact uses the live snapshot pins as-is.
    if matches!(trigger, CompactionTrigger::RequestPressure) {
        protect_hard_core(
            &mut healed,
            agent.turn.current_turn_start_index,
        );
    }
    let selection = match select_compaction_attempt(&healed, &selection_config, trigger)? {
        CompactionSelectionResult::Selected(selection) => selection,
        CompactionSelectionResult::NoProgress(no_progress) => {
            return Ok(Err(no_progress));
        }
    };

    // Capture live state before promoting the healed working copy.
    let checkpoint = PreCompactionCheckpoint::capture(agent);
    agent.runtime_snapshot = healed;
    super::sync_protocol_frame_provenance_from_snapshot(
        &mut agent.protocol_frames,
        &agent.runtime_snapshot,
    );
    match compact_selected_context(agent, selection, on_event, on_delta).await {
        Ok(prepared) => Ok(Ok((checkpoint, prepared))),
        Err(error) => {
            // Roll heal promotion only; candidate was never installed.
            checkpoint.restore_protocol_only(agent);
            Err(error)
        }
    }
}

fn install_prepared_compaction<C: Config + Clone>(
    agent: &mut Agent<C>,
    prepared: &PreparedCompaction,
) -> Result<()> {
    // Compact replaces history; force cold epoch for successor admission.
    agent.commit_prepared_runtime_compaction(
        prepared.snapshot.clone(),
        prepared.protocol_frames.clone(),
        prepared.history.clone(),
        prepared.current_turn_start_index,
    )?;
    agent.clear_active_epoch();
    Ok(())
}

fn pressure_successor_request<C: Config + Clone>(
    agent: &mut Agent<C>,
    protocol: ApiProtocol,
    turn_prelude: &[PromptMessage],
    tool_definitions: &[crate::request_builder::ToolSpec],
) -> Result<PreparedRequestBuild> {
    let epoch_preview = agent.preview_active_epoch(protocol, turn_prelude, tool_definitions)?;
    Ok(PreparedRequestBuild {
        protected_start_index: agent
            .turn
            .current_turn_start_index
            .unwrap_or(agent.history.len()),
        build: epoch_preview.build.clone(),
        epoch_preview,
    })
}

async fn compact_for_request_pressure_with_callback<C>(
    agent: &mut Agent<C>,
    protocol: ApiProtocol,
    turn_prelude: &[PromptMessage],
    tool_definitions: &[crate::request_builder::ToolSpec],
    on_event: &mut EventCallback<'_>,
) -> Result<PreparedRequestBuild>
where
    C: Config + Clone,
{
    let trigger = CompactionTrigger::RequestPressure;
    on_event(AgentEvent::ContextCompactionStarted { trigger }).await?;
    let result = async {
        // OpenCode-style ladder:
        //   1) cheap prune large tool outputs (invalidates epoch when changed)
        //   2) hard-core protect + compact
        //   3) cold successor admission
        let pruned = emergency_prune_tool_outputs_for_pressure(agent)?;
        let prepared_result =
            prepare_compaction_candidate(agent, trigger, on_event, None).await?;
        let (checkpoint, prepared) = match prepared_result {
            Ok(pair) => pair,
            Err(no_progress) => {
                on_event(AgentEvent::ContextCompactionNoProgress(no_progress.clone())).await?;
                return pressure_successor_request(
                    agent,
                    protocol,
                    turn_prelude,
                    tool_definitions,
                )
                .map_err(|error| {
                    let labels = diagnostic_labels(&no_progress.blockers);
                    if pruned {
                        anyhow::anyhow!(
                            "request pressure has no compactable context after prune: {labels}; admission still fails: {error}"
                        )
                    } else {
                        anyhow::anyhow!(
                            "request pressure has no compactable context: {labels}; admission still fails: {error}"
                        )
                    }
                });
            }
        };

        // Install candidate so successor admission sees the compacted history.
        install_prepared_compaction(agent, &prepared)?;
        let admission = match pressure_successor_request(
            agent,
            protocol,
            turn_prelude,
            tool_definitions,
        ) {
            Ok(successor) => Ok(successor),
            // Compaction already reclaimed history. Do not roll that back just
            // because the first successor still misses the budget — prune the
            // new tail and retry once before treating this as a hard failure.
            Err(error) if is_recognized_request_budget_overflow(&error) => {
                let _ = emergency_prune_tool_outputs_for_pressure(agent);
                match pressure_successor_request(agent, protocol, turn_prelude, tool_definitions)
                {
                    Ok(successor) => Ok(successor),
                    Err(retry_error) => Err(PressureAdmissionError::BudgetExhausted {
                        detail: format!(
                            "request still over budget after compaction and prune: {retry_error} (first admission: {error})"
                        ),
                    }),
                }
            }
            Err(error) => Err(PressureAdmissionError::Technical(error)),
        };

        match admission {
            Ok(successor) => {
                if let Err(error) =
                    on_event(AgentEvent::ContextCompacted(prepared.event)).await
                {
                    // Journal rejected the durable compact record — roll back.
                    checkpoint.restore_full(agent);
                    return Err(error);
                }
                Ok(successor)
            }
            Err(PressureAdmissionError::BudgetExhausted { detail }) => {
                // Keep reclaimed history (+ any post-compact prune). Restoring
                // would put the session back into the pre-compact oversized state.
                Err(anyhow::anyhow!(detail))
            }
            Err(PressureAdmissionError::Technical(error)) => {
                checkpoint.restore_full(agent);
                Err(error)
            }
        }
    }
    .await;
    if result.is_err() {
        let _ = on_event(AgentEvent::ContextCompactionFailed { trigger }).await;
    }
    result
}

async fn attempt_compaction<C>(
    agent: &mut Agent<C>,
    trigger: CompactionTrigger,
    on_event: &mut EventCallback<'_>,
    on_delta: Option<&mut (dyn FnMut(&str) -> Result<()> + Send + '_)>,
) -> Result<CompactionAttemptOutcome>
where
    C: Config + Clone,
{
    let result = async {
        let (checkpoint, prepared) =
            match prepare_compaction_candidate(agent, trigger, on_event, on_delta).await? {
                Ok(pair) => pair,
                Err(no_progress) => {
                    on_event(AgentEvent::ContextCompactionNoProgress(no_progress.clone())).await?;
                    return Ok(CompactionAttemptOutcome::NoProgress(no_progress));
                }
            };

        // Manual path: durable callback first, then commit. Failed callback
        // rolls back heal without installing the candidate.
        if let Err(error) = on_event(AgentEvent::ContextCompacted(prepared.event.clone())).await {
            checkpoint.restore_protocol_only(agent);
            return Err(error);
        }
        agent.commit_prepared_runtime_compaction(
            prepared.snapshot,
            prepared.protocol_frames,
            prepared.history,
            prepared.current_turn_start_index,
        )?;
        Ok(CompactionAttemptOutcome::Compacted {
            retained_items: prepared.retained_items,
        })
    }
    .await;
    if result.is_err() {
        // The original technical error is authoritative. A diagnostic callback
        // failure must never replace it or cause a candidate install.
        let _ = on_event(AgentEvent::ContextCompactionFailed { trigger }).await;
    }
    result
}

fn diagnostic_labels(blockers: &[CompactionBlocker]) -> String {
    blockers
        .iter()
        .map(|blocker| blocker.label())
        .collect::<Vec<_>>()
        .join(",")
}

pub(super) async fn prepare_request_build<C, E, Efut>(
    agent: &mut Agent<C>,
    protocol: ApiProtocol,
    turn_prelude: &[PromptMessage],
    protected_start_index: usize,
    tool_definitions: &[crate::request_builder::ToolSpec],
    _on_event: &mut E,
) -> Result<PreparedRequestBuild>
where
    C: Config + Clone,
    E: FnMut(AgentEvent) -> Efut,
    Efut: Future<Output = Result<()>>,
{
    agent.refresh_runtime_snapshot_from_provider()?;
    let epoch_preview = agent.preview_active_epoch(protocol, turn_prelude, tool_definitions)?;
    Ok(PreparedRequestBuild {
        protected_start_index,
        build: epoch_preview.build.clone(),
        epoch_preview,
    })
}

/// Request-construction budget failures that a fresh logical checkpoint can
/// recover without treating malformed protocol or artifact errors as pressure.
pub(super) fn is_recognized_request_budget_overflow(error: &anyhow::Error) -> bool {
    error.chain().any(|cause| {
        let message = cause.to_string();
        message.starts_with("protected current context exceeds input budget:")
            || message == "final prompt and tools exceed selected input budget"
    })
}

pub(super) fn prune_old_tool_outputs<C: Config>(
    agent: &mut Agent<C>,
    preserve_recent_budget: u64,
) -> Result<()> {
    validate_compaction_runtime_state(agent)?;
    if !agent.compaction_config.prune {
        return Ok(());
    }

    // Prune is a live protocol edit: history is authority. Walk recent→old tool
    // outputs, keep a protect window of recent tokens (OpenCode-style), and stub
    // older large payloads. Incomplete tool groups and skill cards stay intact.
    super::ensure_active_protocol_source_spans(&mut agent.runtime_snapshot);
    super::sync_protocol_frame_provenance_from_snapshot(
        &mut agent.protocol_frames,
        &agent.runtime_snapshot,
    );

    // Caller-owned protect window (OpenCode keeps ~40k recent tool tokens).
    // `0` means no recent protect — prune every large eligible tool output.
    let protect_budget = preserve_recent_budget;
    let incomplete_call_ids = incomplete_tool_call_ids(&agent.history);
    let call_names = tool_output_names_by_frame_id(&agent.runtime_snapshot);
    let active = agent.runtime_snapshot.active_protocol_frames();
    ensure!(
        active.len() == agent.history.len(),
        "prune requires history and active protocol length to match ({} vs {})",
        agent.history.len(),
        active.len()
    );

    let mut kept_tool_tokens = 0u64;
    let mut changed = false;
    for index in (0..agent.history.len()).rev() {
        let Some(id) = active[index].runtime_frame_id else {
            continue;
        };
        let HistoryItem::ToolOutput {
            call_id,
            output_json,
            ..
        } = &agent.history[index]
        else {
            continue;
        };
        if incomplete_call_ids.contains(call_id) {
            continue;
        }
        let tool_name = call_names.get(&id).map(String::as_str);
        if tool_name.is_some_and(is_skill_tool_name)
            || output_json.contains(COMPACTION_PRUNED_MARKER)
        {
            continue;
        }
        let cost = estimate_history_item_tokens(&agent.history[index]);
        if kept_tool_tokens.saturating_add(cost) <= protect_budget {
            kept_tool_tokens = kept_tool_tokens.saturating_add(cost);
            continue;
        }
        if output_json.chars().count() < COMPACTION_PRUNE_MIN_OUTPUT_CHARS {
            continue;
        }
        let HistoryItem::ToolOutput { output_json, .. } = &mut agent.history[index] else {
            continue;
        };
        *output_json = build_pruned_tool_output_json(output_json, tool_name);
        changed = true;
    }

    if changed {
        agent.publish_history_to_protocol_mirrors()?;
        // Live prune rewrites protocol payloads (frame item content). Any warm
        // active-epoch prefix digest becomes stale; successor admission must cold
        // rebuild instead of treating the edit as a prefix mutation/reorder.
        // `emergency_prune_tool_outputs_for_pressure` calls this path directly and
        // would otherwise bypass `Agent::prune_old_tool_outputs`' clear.
        agent.clear_active_epoch();
    }
    Ok(())
}

/// Returns true when at least one large historical tool output was stubbed.
pub(super) fn emergency_prune_tool_outputs_for_pressure<C: Config>(
    agent: &mut Agent<C>,
) -> Result<bool> {
    if !agent.compaction_config.prune {
        return Ok(false);
    }
    let before = agent.history.clone();
    // Under emergency pressure, do not keep a large recent tool window. Mid-turn
    // completed tool mass is the common lock-up; stub every large eligible output
    // (incomplete tools / skill cards still skipped inside the pruner).
    prune_old_tool_outputs(agent, 0)?;
    Ok(agent.history != before)
}

fn incomplete_tool_call_ids(history: &[HistoryItem]) -> BTreeSet<String> {
    let Ok(transcript) = analyze_history_items(history, None) else {
        return BTreeSet::new();
    };
    transcript
        .tool_call_groups
        .iter()
        .filter(|group| group.status != ToolCallGroupStatus::Complete)
        .flat_map(|group| group.call_ids.iter().cloned())
        .collect()
}

async fn compact_selected_context<C>(
    agent: &mut Agent<C>,
    selection: CompactionSelection,
    on_event: &mut EventCallback<'_>,
    on_delta: Option<&mut (dyn FnMut(&str) -> Result<()> + Send + '_)>,
) -> Result<PreparedCompaction>
where
    C: Config + Clone,
{
    // Journal shape: summary + retired spans only. Typed derived_coverage is no
    // longer produced on the live path (legacy records may still carry it).
    let summary = crate::transcript::transcript_projection::sanitize_compaction_summary_body(
        &generate_context_summary(
            agent,
            selection.previous_summary.as_deref(),
            &selection.head_for_summary,
            on_event,
            on_delta,
        )
        .await?,
    );

    // Retire/summary on a working snapshot first (structure), then derive history
    // and prune large tool payloads on history (protocol authority). Finally
    // rebind pruned payloads back into the candidate snapshot mirror.
    let mut snapshot = agent.prepare_runtime_compaction_from_snapshot(
        &agent.runtime_snapshot,
        &selection,
        summary.clone(),
    )?;
    let current_turn_start_index =
        agent.rebased_current_turn_start_index_after_compaction(&selection, &mut snapshot)?;
    let protocol_frames = snapshot.active_protocol_frames();
    let mut history = crate::protocol_frames::history_items_from_frames(&protocol_frames);
    prune_history_tool_outputs(&mut history, &snapshot, &selection)?;
    crate::protocol_frames::analyze_history_items(&history, current_turn_start_index)?;
    super::rebind_active_protocol_from_history(&mut snapshot, &history)?;
    let protocol_frames = snapshot.active_protocol_frames();

    let event = ContextCompactionEvent {
        outcome: "succeeded".into(),
        summary,
        tail_start_index: selection.tail_start_index,
        original_history_items: 0,
        retained_history_items: 0,
        // The journal is cumulative. Persist exactly the closure state that was
        // applied to the candidate, so a later replay never has to infer which
        // earlier raw material this summary replaced.
        retired_source_spans: snapshot
            .compaction
            .retired_source_spans
            .iter()
            .map(|span| ContextCompactionSourceSpan {
                start_sequence: span.start_sequence,
                end_sequence: span.end_sequence,
            })
            .collect(),
        // New events do not persist identity bindings; resume rebinds from
        // protocol/history. Legacy events may still carry bindings for read-compat.
        frame_identity_bindings: Vec::new(),
        derived_coverage: None,
        detail: None,
    };
    Ok(PreparedCompaction {
        retained_items: 1 + selection.tail_items.len(),
        event,
        snapshot,
        protocol_frames,
        history,
        current_turn_start_index,
    })
}

async fn generate_context_summary<C>(
    agent: &Agent<C>,
    previous_summary: Option<&str>,
    head_for_summary: &[HistoryItem],
    on_event: &mut EventCallback<'_>,
    mut on_delta: Option<&mut (dyn FnMut(&str) -> Result<()> + Send + '_)>,
) -> Result<String>
where
    C: Config + Clone,
{
    // A summary is an implementation detail of a compaction transaction.  It
    // must never start another pressure-compaction transaction of its own.
    let mut summary_turn = TurnRuntimeState::default();
    summary_turn.pressure_compaction.suppress();
    let mut summary_agent = Agent {
        client: agent.client.clone(),
        model: agent.model.clone(),
        subagent_model_overrides: HashMap::new(),
        default_protocol: agent.default_protocol,
        model_protocols: agent.model_protocols.clone(),
        model_catalog: agent.model_catalog.clone(),
        prelude: vec![PromptMessage::developer(CONTEXT_COMPACTION_PRELUDE)],
        protocol_frames: Vec::new(),
        history: Vec::new(),
        runtime_snapshot: Agent::<C>::fresh_runtime_snapshot(&agent.model),
        tools: ToolRegistry::new(),
        skill_registry: None,
        skill_cards: Vec::new(),
        subagent_delegate: None,
        question_handler: None,
        permission_session: std::sync::Arc::new(std::sync::Mutex::new(
            PermissionSessionState::default(),
        )),
        compaction_config: CompactionConfig {
            ..CompactionConfig::default()
        },
        #[cfg(test)]
        automatic_checkpoint_policy: super::automatic_checkpoint::AutoCheckpointPolicy::from_config(
            LogicalCheckpointConfig::default(),
        ),
        retry_config: agent.retry_config.clone(),
        tool_timeout_secs: agent.tool_timeout_secs,
        turn: summary_turn,
        next_turn_id: 0,
        max_iterations: Some(1),
        max_tool_calls: Some(0),
        context_scope_state: Arc::new(std::sync::Mutex::new(ContextScopeState::default())),
        runtime_snapshot_provider: None,
        logical_checkpoint_candidate_provider: None,
        context_experiment_restore_point: None,
        logical_checkpoint_control: super::LogicalCheckpointControl {
            state: Arc::new(std::sync::Mutex::new(
                super::LogicalCheckpointControlState {
                    enabled: false,
                    request: super::LogicalCheckpointRequestState::Idle,
                    request_run_id: None,
                    active_run_id: None,
                    next_run_id: 0,
                    next_request_id: 0,
                    request_id: None,
                    automatic_enabled: false,
                },
            )),
        },
        logical_request_observations: super::LogicalRequestObservationTracker::default(),
        active_epoch: None,
        pressure_compaction_suppressed: true,
    };
    let prompt = render_compaction_prompt(
        previous_summary,
        head_for_summary,
        compaction_history_char_budget(agent.active_model_metadata()),
    );
    // Keep the nested stream callback independent of the outer event future.
    // Passing the outer callback through run_stream_async made its Send proof
    // recursively depend on the task which owns this compaction attempt.
    let (delta_tx, mut delta_rx) = tokio::sync::mpsc::unbounded_channel();
    let emit_tx = delta_tx.clone();
    drop(delta_tx);
    let summary = summary_agent
        .run_stream_async(
            &prompt,
            move |delta| {
                std::future::ready(
                    emit_tx
                        .send(delta.to_string())
                        .map_err(|_| anyhow::anyhow!("context compaction delta receiver closed")),
                )
            },
            |_| std::future::ready(Ok(())),
            |_| std::future::ready(Ok(PermissionApproval::Deny)),
        )
        .boxed();
    tokio::pin!(summary);
    let summary = loop {
        tokio::select! {
            result = &mut summary => break result?,
            Some(delta) = delta_rx.recv() => {
                if let Some(on_delta) = on_delta.as_deref_mut() {
                    on_delta(&delta)?;
                }
                on_event(AgentEvent::ContextCompactionDelta { delta }).await?;
            }
        }
    };
    while let Ok(delta) = delta_rx.try_recv() {
        if let Some(on_delta) = on_delta.as_deref_mut() {
            on_delta(&delta)?;
        }
        on_event(AgentEvent::ContextCompactionDelta { delta }).await?;
    }
    let trimmed = summary.trim();
    if trimmed.is_empty() {
        bail!("context compaction produced an empty summary")
    }
    Ok(trimmed.to_string())
}

/// Selects an aggressive compaction candidate without encoding normal lack of
/// progress in an error string. Validation and protocol inconsistencies still
/// use the outer `Result` and therefore remain technical failures.
fn select_compaction_attempt(
    snapshot: &RuntimeSnapshot,
    config: &CompactionConfig,
    trigger: CompactionTrigger,
) -> Result<CompactionSelectionResult> {
    match select_runtime_compaction_segments(snapshot, config, 0) {
        Ok(selection) => Ok(CompactionSelectionResult::Selected(selection)),
        Err(error) if is_nothing_to_compact_error(&error) => Ok(
            CompactionSelectionResult::NoProgress(CompactionNoProgress {
                trigger,
                blockers: compaction_blockers(snapshot)?,
            }),
        ),
        Err(error) => Err(error),
    }
}

fn compaction_blockers(snapshot: &RuntimeSnapshot) -> Result<Vec<CompactionBlocker>> {
    snapshot.validate_references()?;
    let frames = snapshot.active_protocol_frames();
    let items = frames
        .iter()
        .map(|frame| frame.to_history_item())
        .collect::<Vec<_>>();
    let transcript = analyze_history_items(&items, None)?;
    let base_start = frames
        .iter()
        .rposition(|frame| matches!(frame.item, ProtocolFrameItem::ContextSummary { .. }))
        .map(|index| index + 1)
        .unwrap_or(0);
    let mut blockers = BTreeSet::new();
    if base_start == frames.len() {
        blockers.insert(CompactionBlocker::NoHistoricalItems);
    }
    let protected = retirement_blocker_frame_ids(snapshot);
    let frame_by_id = snapshot
        .frames
        .iter()
        .map(|frame| (frame.id, frame))
        .collect::<BTreeMap<_, _>>();
    for frame in &frames[base_start..] {
        let id = frame
            .runtime_frame_id
            .expect("runtime protocol frames have ids");
        if protected.contains(&id) {
            blockers.insert(CompactionBlocker::ProtectedContext);
        }
        if frame_by_id[&id].provenance.source_span.is_none() {
            blockers.insert(CompactionBlocker::MissingSourceProvenance);
        }
    }
    if transcript
        .tool_call_groups
        .iter()
        .any(|group| group.status != ToolCallGroupStatus::Complete)
    {
        blockers.insert(CompactionBlocker::IncompleteToolGroup);
    }
    if blockers.is_empty() {
        blockers.insert(CompactionBlocker::NoSafeBoundary);
    }
    Ok(blockers.into_iter().collect())
}

/// History-first selection: protocol history frames drive the prefix; the runtime
/// snapshot is only the view that hosts those frames and dependent projections.
pub(super) fn select_runtime_compaction_segments(
    snapshot: &RuntimeSnapshot,
    config: &CompactionConfig,
    preserve_recent_budget: u64,
) -> Result<CompactionSelection> {
    // The request history is the only compaction authority. Runtime frames only
    // supply identity and durable source provenance for the selected prefix.
    snapshot.validate_references()?;
    let frames = snapshot.active_protocol_frames();
    let items = frames
        .iter()
        .map(|frame| frame.to_history_item())
        .collect::<Vec<_>>();
    let transcript = analyze_history_items(&items, None)?;
    let summary_index = frames
        .iter()
        .rposition(|frame| matches!(frame.item, ProtocolFrameItem::ContextSummary { .. }));
    let previous_summary = summary_index.and_then(|index| match &frames[index].item {
        ProtocolFrameItem::ContextSummary { text } => Some(text.clone()),
        _ => None,
    });
    let base_start = summary_index.map(|index| index + 1).unwrap_or(0);
    if base_start == frames.len() {
        return Err(NoProgressSelection::NoHistoricalItems.into());
    }

    let candidates = &items[base_start..];
    let turn_ranges = split_history_turn_ranges(candidates);
    let requested_tail_start = turn_ranges
        .iter()
        .rev()
        .take(config.tail_turns.min(turn_ranges.len()))
        .map(|(start, _)| *start)
        .min()
        .unwrap_or(candidates.len());
    let preserve_budget = config
        .preserve_recent_tokens
        .unwrap_or(preserve_recent_budget);
    let requested_tail_start = trim_tail_to_budget(
        candidates,
        &turn_ranges,
        requested_tail_start,
        preserve_budget,
    );
    let requested_end = base_start
        + crate::protocol_frames::canonical_compaction_boundary(candidates, requested_tail_start)?;

    let frame_by_id = snapshot
        .frames
        .iter()
        .map(|frame| (frame.id, frame))
        .collect::<BTreeMap<_, _>>();
    let protected_ids = retirement_blocker_frame_ids(snapshot);
    let has_incomplete_group = |end: usize| {
        transcript.tool_call_groups.iter().any(|group| {
            group.status != ToolCallGroupStatus::Complete
                && base_start <= group.assistant_index
                && group.assistant_index < end
        })
    };

    let tail_start_index = (base_start + 1..=requested_end)
        .rev()
        .find(|&end| {
            crate::protocol_frames::canonical_compaction_boundary(&items, end)
                .is_ok_and(|boundary| boundary == end)
                && !has_incomplete_group(end)
                && frames[base_start..end].iter().all(|frame| {
                    frame
                        .runtime_frame_id
                        .is_some_and(|id| retirable_frame(id, &frame_by_id, &protected_ids))
                })
                && {
                    let selected_ids = frames[base_start..end]
                        .iter()
                        .filter_map(|frame| frame.runtime_frame_id)
                        .collect::<BTreeSet<_>>();
                    let retired_spans = canonical_runtime_retired_closure(
                        selected_ids
                            .iter()
                            .filter_map(|id| frame_by_id[id].provenance.source_span)
                            .collect(),
                    );
                    retained_compaction_spans(
                        snapshot,
                        &selected_ids,
                        &retired_spans,
                    )
                    .is_ok_and(|retained_spans| {
                        retired_spans.iter().all(|retired| {
                            retained_spans
                                .iter()
                                .all(|retained| !spans_overlap(*retired, *retained))
                        })
                    })
                }
        })
        .ok_or(NoProgressSelection::NoSafeBoundary)?;

    let protocol_retired_ids = frames[base_start..tail_start_index]
        .iter()
        .filter_map(|frame| frame.runtime_frame_id)
        .collect::<BTreeSet<_>>();
    let retired_source_spans = canonical_runtime_retired_closure(
        protocol_retired_ids
            .iter()
            .filter_map(|id| frame_by_id[id].provenance.source_span)
            .collect(),
    );
    let mut retired_frame_ids = protocol_retired_ids;
    retired_frame_ids.extend(
        crate::transcript::transcript_projection::classify_compaction_closure(
            snapshot,
            &retired_source_spans,
        )
        .co_retired_frame_ids,
    );
    let first_protected_index = frames
        .iter()
        .position(|frame| {
            frame
                .runtime_frame_id
                .is_some_and(|id| protected_ids.contains(&id))
        })
        .unwrap_or(frames.len());
    Ok(CompactionSelection {
        previous_summary,
        head_for_summary: items[base_start..tail_start_index].to_vec(),
        tail_items: items[tail_start_index..first_protected_index].to_vec(),
        tail_start_index,
        retired_frame_ids: retired_frame_ids.into_iter().collect(),
        retired_source_spans,
    })
}

fn is_traceability_only(frame: &crate::runtime_context::RuntimeFrame) -> bool {
    matches!(frame.provenance.source, RuntimeSource::SummaryArtifact)
        || (frame.kind == crate::runtime_context::RuntimeFrameKind::PromptContributor
            && matches!(
                frame.provenance.label.as_deref(),
                Some("summary") | Some("evidence")
            ))
}

fn dependent_projection_ids(
    snapshot: &RuntimeSnapshot,
    retired_spans: &[SourceSpan],
) -> BTreeSet<RuntimeFrameId> {
    crate::transcript::transcript_projection::classify_compaction_closure(
        snapshot,
        retired_spans,
    )
    .co_retired_frame_ids
}

fn canonical_runtime_retired_closure(spans: Vec<SourceSpan>) -> Vec<SourceSpan> {
    crate::transcript::transcript_projection::canonical_retired_source_spans(
        spans
            .into_iter()
            .map(|span| ContextCompactionSourceSpan {
                start_sequence: span.start_sequence,
                end_sequence: span.end_sequence,
            })
            .collect(),
    )
    .into_iter()
    .map(|span| {
        SourceSpan::new(span.start_sequence, span.end_sequence)
            .expect("canonical source spans are valid")
    })
    .collect()
}

/// The one closure classification used by both selection and candidate apply.
/// A projection whose complete source is retired has no independent prompt
/// authority.  Summary provenance is traceability rather than request source.
pub(super) fn retained_compaction_spans(
    snapshot: &RuntimeSnapshot,
    protocol_retired_ids: &BTreeSet<RuntimeFrameId>,
    retired_spans: &[SourceSpan],
) -> Result<Vec<SourceSpan>> {
    snapshot.validate_references()?;
    let dependent_ids = dependent_projection_ids(snapshot, retired_spans);
    let co_retired = protocol_retired_ids
        .iter()
        .copied()
        .chain(dependent_ids.iter().copied())
        .collect::<BTreeSet<_>>();
    // Soft-retaining prompt contributors are not retirement pins. Only live
    // active/folded frame spans block overlapping retirements.
    Ok(snapshot
        .frames
        .iter()
        .filter(|frame| {
            matches!(
                frame.visibility,
                FrameVisibility::Active | FrameVisibility::Folded
            ) && !co_retired.contains(&frame.id)
                && !is_traceability_only(frame)
        })
        .filter_map(|frame| frame.provenance.source_span)
        .collect::<Vec<_>>())
}

fn aggressive_selection_config(base: &CompactionConfig) -> CompactionConfig {
    // Pressure / manual reclaim: retire as much history as is safe and prune
    // retained tool payloads in the same transaction.
    CompactionConfig {
        prune: true,
        tail_turns: 0,
        preserve_recent_tokens: Some(0),
        ..base.clone()
    }
}

fn prune_history_tool_outputs(
    history: &mut [HistoryItem],
    snapshot: &RuntimeSnapshot,
    selection: &CompactionSelection,
) -> Result<()> {
    // Always prune retained large tool outputs on compact. Session
    // `compaction.prune` only gates the standalone prune path.
    let retired_ids: BTreeSet<_> = selection.retired_frame_ids.iter().copied().collect();
    let call_names = tool_output_names_by_frame_id(snapshot);
    let active = snapshot.active_protocol_frames();
    ensure!(
        active.len() == history.len(),
        "prune history requires matching active protocol length"
    );
    let mut kept_tool_tokens = 0u64;
    // Walk recent→old like the snapshot pruner.
    for index in (0..history.len()).rev() {
        let Some(id) = active[index].runtime_frame_id else {
            continue;
        };
        if retired_ids.contains(&id) {
            continue;
        }
        let HistoryItem::ToolOutput { output_json, .. } = &history[index] else {
            continue;
        };
        let tool_name = call_names.get(&id).map(String::as_str);
        if tool_name.is_some_and(is_skill_tool_name)
            || output_json.contains(COMPACTION_PRUNED_MARKER)
        {
            continue;
        }
        let cost = estimate_history_item_tokens(&history[index]);
        if kept_tool_tokens.saturating_add(cost) <= COMPACTION_PRUNE_PROTECT_TOKENS {
            kept_tool_tokens = kept_tool_tokens.saturating_add(cost);
            continue;
        }
        if output_json.chars().count() >= COMPACTION_PRUNE_MIN_OUTPUT_CHARS {
            let HistoryItem::ToolOutput { output_json, .. } = &mut history[index] else {
                continue;
            };
            *output_json = build_pruned_tool_output_json(output_json, tool_name);
        }
    }
    Ok(())
}

fn validate_compaction_runtime_state<C: Config>(agent: &Agent<C>) -> Result<()> {
    // History is the protocol authority. Only structural completeness is
    // required before selection; no multi-copy payload equality checks.
    analyze_history_items(&agent.history, agent.turn.current_turn_start_index)?;
    agent.runtime_snapshot.validate_references()?;
    Ok(())
}

/// Working-copy heal for selection. Does not mutate live agent state.
fn healed_snapshot_for_selection(snapshot: &RuntimeSnapshot) -> RuntimeSnapshot {
    let mut healed = snapshot.clone();
    super::ensure_active_protocol_source_spans(&mut healed);
    healed
}

fn is_skill_tool_name(name: &str) -> bool {
    name == "skill"
        || name == tool_names::TOOL_SKILL
        || name.starts_with("skill__")
        || name.starts_with("skill/")
}

fn retirable_frame(
    id: RuntimeFrameId,
    frames: &BTreeMap<RuntimeFrameId, &crate::runtime_context::RuntimeFrame>,
    protected_ids: &BTreeSet<RuntimeFrameId>,
) -> bool {
    let frame = frames[&id];
    frame.visibility == FrameVisibility::Active
        && frame.provenance.source == RuntimeSource::Transcript
        && frame.provenance.source_span.is_some()
        && !protected_ids.contains(&id)
}

/// Compaction-only hard protection authority: explicit + turn pins.
/// Soft-retaining contributors do not participate.
///
/// Request-pressure paths pin only the emergency hard core on a working
/// snapshot before calling selection (see
/// `protect_hard_core`).
fn retirement_blocker_frame_ids(snapshot: &RuntimeSnapshot) -> BTreeSet<RuntimeFrameId> {
    // Hard protect only: explicit + turn pins.
    snapshot
        .compaction
        .explicit_protected_frame_ids
        .iter()
        .copied()
        .chain(
            snapshot
                .compaction
                .turn_protected_frame_ids
                .iter()
                .copied(),
        )
        .collect::<BTreeSet<_>>()
}

/// Hard core for pressure reclaim: keep only the current-turn opening
/// user/continuation and incomplete tool groups. Completed mid-turn tool mass
/// stays reclaimable so a long agent turn cannot hard-lock the session.
///
/// Overwrite `turn_protected_frame_ids`. Live snapshots may already mark the
/// whole active turn as turn-protected; leaving that set intact would re-block
/// every completed tool frame under pressure.
fn protect_hard_core(
    snapshot: &mut RuntimeSnapshot,
    current_turn_start_index: Option<usize>,
) {
    let frames = snapshot.active_protocol_frames();
    let items = frames
        .iter()
        .map(|frame| frame.to_history_item())
        .collect::<Vec<_>>();
    let start = current_turn_start_index
        .unwrap_or(frames.len())
        .min(frames.len());
    let mut hard_core = Vec::new();
    if start < frames.len() {
        // Opening user / internal-continuation of the active turn.
        if matches!(
            items[start],
            HistoryItem::UserMessage { .. } | HistoryItem::InternalContinuation { .. }
        ) {
            if let Some(id) = frames[start].runtime_frame_id {
                hard_core.push(id);
            }
        }
    }
    if let Ok(transcript) = analyze_history_items(&items, current_turn_start_index) {
        for group in transcript.tool_call_groups {
            if group.status == ToolCallGroupStatus::Complete {
                continue;
            }
            if let Some(id) = frames
                .get(group.assistant_index)
                .and_then(|frame| frame.runtime_frame_id)
            {
                hard_core.push(id);
            }
            for output_index in group.tool_output_indexes {
                if let Some(id) = frames
                    .get(output_index)
                    .and_then(|frame| frame.runtime_frame_id)
                {
                    hard_core.push(id);
                }
            }
        }
    }
    hard_core.sort();
    hard_core.dedup();
    // Replace turn protection with the hard core only. Prior explicit pins stay.
    snapshot.set_turn_protected_frame_ids(hard_core);
}

fn tool_output_names_by_frame_id(snapshot: &RuntimeSnapshot) -> BTreeMap<RuntimeFrameId, String> {
    let frames = snapshot.active_protocol_frames();
    let items = frames
        .iter()
        .map(|frame| frame.to_history_item())
        .collect::<Vec<_>>();
    let Ok(transcript) = analyze_history_items(&items, None) else {
        return BTreeMap::new();
    };
    let mut names = BTreeMap::new();
    for group in transcript.tool_call_groups {
        let ProtocolFrameItem::AssistantToolCalls { calls, .. } =
            &frames[group.assistant_index].item
        else {
            continue;
        };
        for output_index in group.tool_output_indexes {
            let ProtocolFrameItem::ToolOutput { call_id, .. } = &frames[output_index].item else {
                continue;
            };
            if let Some(call) = calls.iter().find(|call| call.call_id == *call_id)
                && let Some(id) = frames[output_index].runtime_frame_id
            {
                names.insert(id, call.name.clone());
            }
        }
    }
    names
}

fn spans_overlap(left: SourceSpan, right: SourceSpan) -> bool {
    left.overlaps(right)
}

#[cfg(test)]
pub(super) fn test_snapshot_for_history(history: &[HistoryItem]) -> RuntimeSnapshot {
    let mut snapshot = RuntimeSnapshot::new("test");
    for (index, item) in history.iter().enumerate() {
        let protocol_frame = crate::protocol_frames::ProtocolFrame::from_history_item(index, item);
        let mut frame = super::runtime_frame_from_protocol_frame(&protocol_frame, index as u32);
        frame.provenance = crate::runtime_context::RuntimeFrameProvenance::new(
            crate::runtime_context::RuntimeSource::Transcript,
        )
        .with_span(SourceSpan::new(index as u64 + 1, index as u64 + 1).expect("valid test span"));
        snapshot.push_frame(frame);
    }
    snapshot
}

#[cfg(test)]
pub(super) fn select_compaction_segments(
    history: &[HistoryItem],
    protected_start_index: usize,
    config: &CompactionConfig,
    preserve_recent_budget: u64,
) -> Result<CompactionSelection> {
    let mut snapshot = test_snapshot_for_history(history);
    // Compatibility tests express the current turn as an ordinal. Translate it
    // once into durable protection; production selection has no ordinal input.
    let mut protected = snapshot.compaction.protected_frame_ids.clone();
    protected.extend(
        snapshot.active_protocol_frames()[protected_start_index.min(history.len())..]
            .iter()
            .filter_map(|frame| frame.runtime_frame_id),
    );
    snapshot.set_protected_frame_ids(protected);
    select_runtime_compaction_segments(&snapshot, config, preserve_recent_budget)
}

fn is_nothing_to_compact_error(error: &anyhow::Error) -> bool {
    error.downcast_ref::<NoProgressSelection>().is_some()
}

fn split_history_turn_ranges(items: &[HistoryItem]) -> Vec<(usize, usize)> {
    let mut ranges = Vec::new();
    let mut current_start = 0usize;
    for (index, item) in items.iter().enumerate() {
        let starts_turn = matches!(
            item,
            HistoryItem::UserMessage { .. } | HistoryItem::InternalContinuation { .. }
        );
        if index > 0 && starts_turn {
            ranges.push((current_start, index));
            current_start = index;
        }
    }
    ranges.push((current_start, items.len()));
    ranges
}

fn trim_tail_to_budget(
    items: &[HistoryItem],
    turn_ranges: &[(usize, usize)],
    min_start: usize,
    preserve_budget: u64,
) -> usize {
    if preserve_budget == 0 || min_start >= items.len() {
        return items.len();
    }

    let relevant_turns = turn_ranges
        .iter()
        .copied()
        .filter(|(_, end)| *end > min_start)
        .collect::<Vec<_>>();
    let mut remaining_budget = preserve_budget;
    let mut tail_start = items.len();
    let mut kept_any = false;

    for (turn_start, turn_end) in relevant_turns.into_iter().rev() {
        let start = turn_start.max(min_start);
        if start >= turn_end {
            continue;
        }
        let turn_cost = items[start..turn_end]
            .iter()
            .map(estimate_history_item_tokens)
            .sum::<u64>();
        if turn_cost <= remaining_budget {
            tail_start = start;
            remaining_budget = remaining_budget.saturating_sub(turn_cost);
            kept_any = true;
            continue;
        }

        for split_start in start..turn_end {
            let slice_cost = items[split_start..turn_end]
                .iter()
                .map(estimate_history_item_tokens)
                .sum::<u64>();
            if slice_cost <= remaining_budget {
                tail_start = split_start;
                kept_any = true;
                break;
            }
        }
        break;
    }

    if kept_any { tail_start } else { items.len() }
}

pub(super) fn render_compaction_prompt(
    previous_summary: Option<&str>,
    head_for_summary: &[HistoryItem],
    history_char_budget: usize,
) -> String {
    let serialized_history =
        render_bounded_compaction_history(head_for_summary, history_char_budget);
    match previous_summary {
        Some(previous_summary) => {
            let previous_summary = truncate_for_compaction(
                previous_summary,
                history_char_budget.saturating_div(2).clamp(512, 8_000),
                "… [previous summary truncated for compaction]",
            );
            format!(
                "请根据以下内容更新已有锚定摘要。保留仍然正确且仍然重要的内容，删除已过时或被推翻的信息，并合并新的事实、约束、发现、已完成项、路径、工具、待办与可选下一步。输出必须仍遵循 prelude 的 Markdown section 结构。\n\n已有锚定摘要：\n{}\n\n需要并入的新历史：\n{}",
                previous_summary, serialized_history
            )
        }
        None => format!(
            "请根据以下会话历史生成新的锚定摘要，供后续轮次继续工作。使用 prelude 规定的 Markdown section 结构，覆盖 Goal、Instructions/Constraints、Discoveries、Accomplished、Relevant files/tools、Pending tasks 与 Optional next step。\n\n会话历史：\n{}",
            serialized_history
        ),
    }
}

pub(super) fn compaction_history_char_budget(model: ModelRequestMetadata) -> usize {
    let input_budget = effective_input_budget_tokens(model, &[]);
    input_budget
        .saturating_div(4)
        .clamp(256, 16_000)
        .saturating_mul(3)
        .try_into()
        .unwrap_or(COMPACTION_HISTORY_MAX_CHAR_BUDGET)
        .clamp(
            COMPACTION_HISTORY_MIN_CHAR_BUDGET,
            COMPACTION_HISTORY_MAX_CHAR_BUDGET,
        )
}

pub(super) fn default_preserve_recent_budget(input_budget: u64) -> u64 {
    if input_budget == 0 {
        return 0;
    }
    input_budget
        .saturating_div(4)
        .clamp(2_000, 8_000)
        .min(input_budget)
}

pub(super) fn describe_history_item(item: &HistoryItem) -> String {
    match item {
        HistoryItem::ContextSummary { text } => format!("摘要: {text}"),
        HistoryItem::UserMessage { content } => format!("用户: {}", content.display_text()),
        HistoryItem::InternalContinuation { text } => format!("继续执行指令: {text}"),
        HistoryItem::AssistantText { text } => format!("助手: {text}"),
        HistoryItem::AssistantToolCalls { text, calls } => format!(
            "助手工具调用{}: {}",
            text.as_deref()
                .map(|value| format!("（附言: {value}）"))
                .unwrap_or_default(),
            calls
                .iter()
                .map(|call| format!("{}({})", call.name, call.arguments_json))
                .collect::<Vec<_>>()
                .join(", ")
        ),
        HistoryItem::ToolOutput {
            call_id,
            output_json,
        } => format!(
            "工具输出 {call_id}: {}",
            render_tool_output_for_compaction(output_json)
        ),
    }
}

pub(super) fn render_bounded_compaction_history(
    items: &[HistoryItem],
    budget_chars: usize,
) -> String {
    let lines = items
        .iter()
        .enumerate()
        .map(|(index, item)| format!("{}. {}", index + 1, describe_history_item(item)))
        .collect::<Vec<_>>();
    let full = lines.join("\n");
    if full.chars().count() <= budget_chars {
        return full;
    }

    let marker_len = COMPACTION_HISTORY_TRUNCATION_MARKER.chars().count();
    if budget_chars <= marker_len + 1 {
        return truncate_for_compaction(
            COMPACTION_HISTORY_TRUNCATION_MARKER,
            budget_chars,
            COMPACTION_HISTORY_TRUNCATION_MARKER,
        );
    }

    let body_budget = budget_chars - marker_len - 1;
    let mut selected = Vec::new();
    let mut used = 0usize;
    for line in lines.iter().rev() {
        let line_len = line.chars().count();
        let separator = usize::from(!selected.is_empty());
        if used + separator + line_len <= body_budget {
            selected.push(line.clone());
            used += separator + line_len;
            continue;
        }
        if selected.is_empty() {
            selected.push(truncate_for_compaction(
                line,
                body_budget,
                COMPACTION_TOOL_OUTPUT_TRUNCATION_MARKER,
            ));
        }
        break;
    }
    selected.reverse();
    format!(
        "{}\n{}",
        COMPACTION_HISTORY_TRUNCATION_MARKER,
        selected.join("\n")
    )
}

fn render_tool_output_for_compaction(output_json: &str) -> String {
    let rendered = serde_json::from_str::<Value>(output_json)
        .ok()
        .map(sanitize_tool_output_value_for_compaction)
        .and_then(|value| serde_json::to_string(&value).ok())
        .unwrap_or_else(|| output_json.to_string());
    truncate_for_compaction(
        &rendered,
        COMPACTION_TOOL_OUTPUT_CHAR_CAP,
        COMPACTION_TOOL_OUTPUT_TRUNCATION_MARKER,
    )
}

fn sanitize_tool_output_value_for_compaction(mut value: Value) -> Value {
    strip_obvious_media_fields(&mut value, "$", false);
    value
}

fn strip_obvious_media_fields(value: &mut Value, path: &str, force_strip: bool) {
    match value {
        Value::Object(map) => {
            for (key, child) in map.iter_mut() {
                let child_path = format!("{path}.{key}");
                let should_strip =
                    force_strip || field_name_looks_media_like(key) || value_looks_blob_like(child);
                if should_strip {
                    let original_chars = child.to_string().chars().count();
                    *child = Value::String(format!(
                        "[stripped media/blob-like field at {child_path}; original size {original_chars} chars]"
                    ));
                    continue;
                }
                strip_obvious_media_fields(child, &child_path, false);
            }
        }
        Value::Array(items) => {
            for (index, child) in items.iter_mut().enumerate() {
                let child_path = format!("{path}[{index}]");
                let should_strip = force_strip || value_looks_blob_like(child);
                if should_strip {
                    let original_chars = child.to_string().chars().count();
                    *child = Value::String(format!(
                        "[stripped media/blob-like field at {child_path}; original size {original_chars} chars]"
                    ));
                    continue;
                }
                strip_obvious_media_fields(child, &child_path, false);
            }
        }
        _ => {}
    }
}

fn field_name_looks_media_like(key: &str) -> bool {
    let key = key.to_ascii_lowercase();
    [
        "image",
        "audio",
        "video",
        "screenshot",
        "thumbnail",
        "preview",
        "attachment",
        "blob",
        "base64",
        "binary",
        "data_uri",
        "dataurl",
    ]
    .iter()
    .any(|needle| key.contains(needle))
}

fn value_looks_blob_like(value: &Value) -> bool {
    let Some(text) = value.as_str() else {
        return false;
    };
    let trimmed = text.trim();
    trimmed.starts_with("data:")
        || trimmed.starts_with("blob:")
        || (trimmed.chars().count() >= 512 && looks_base64_like(trimmed))
}

fn looks_base64_like(text: &str) -> bool {
    let compact = text.trim();
    !compact.is_empty()
        && compact
            .chars()
            .all(|ch| ch.is_ascii_alphanumeric() || matches!(ch, '+' | '/' | '=' | '_' | '-'))
}

fn truncate_for_compaction(text: &str, max_chars: usize, marker: &str) -> String {
    let total_chars = text.chars().count();
    if total_chars <= max_chars {
        return text.to_string();
    }
    let marker_chars = marker.chars().count();
    if max_chars <= marker_chars {
        return marker.chars().take(max_chars).collect();
    }
    let keep = max_chars - marker_chars;
    let mut truncated = text.chars().take(keep).collect::<String>();
    truncated.push_str(marker);
    truncated
}

fn build_pruned_tool_output_json(output_json: &str, tool_name: Option<&str>) -> String {
    let original_chars = output_json.chars().count();
    let mut marker = serde_json::Map::new();
    marker.insert("pruned".into(), Value::Bool(true));
    marker.insert(
        "reason".into(),
        Value::String(COMPACTION_PRUNED_MARKER.to_string()),
    );
    marker.insert(
        "original_chars".into(),
        Value::Number(serde_json::Number::from(original_chars as u64)),
    );
    if let Some(tool_name) = tool_name {
        marker.insert("tool".into(), Value::String(tool_name.to_string()));
    }

    if serde_json::from_str::<Value>(output_json).is_err() {
        marker.insert("unparsed".into(), Value::Bool(true));
    }

    json!({ "_compaction": Value::Object(marker) }).to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime_context::{
        PromptContributorKind, PromptContributorPlaceholder, RuntimeFrame, RuntimeFrameIdSeed,
        RuntimeFrameKind, RuntimeFrameProvenance, RuntimeSource,
    };

    fn tool_call(call_id: &str) -> HistoryToolCall {
        HistoryToolCall {
            call_id: call_id.into(),
            name: "fs__read".into(),
            arguments_json: "{}".into(),
        }
    }

    fn snapshot_for(history: &[HistoryItem], missing_span_at: Option<usize>) -> RuntimeSnapshot {
        let mut snapshot = RuntimeSnapshot::new("main");
        for (index, item) in history.iter().enumerate() {
            let kind = match item {
                HistoryItem::ContextSummary { .. } => RuntimeFrameKind::Summary,
                HistoryItem::UserMessage { .. } => RuntimeFrameKind::User,
                HistoryItem::InternalContinuation { .. } => RuntimeFrameKind::Metadata,
                HistoryItem::AssistantText { .. } => RuntimeFrameKind::Assistant,
                HistoryItem::AssistantToolCalls { .. } => RuntimeFrameKind::ToolCall,
                HistoryItem::ToolOutput { .. } => RuntimeFrameKind::ToolOutput,
            };
            let source_span = (missing_span_at != Some(index))
                .then(|| SourceSpan::new(index as u64 + 1, index as u64 + 1).unwrap());
            let mut provenance = RuntimeFrameProvenance::new(RuntimeSource::Transcript);
            if let Some(source_span) = source_span {
                provenance = provenance.with_span(source_span);
            }
            let mut frame = RuntimeFrame::new(
                kind,
                FrameVisibility::Active,
                provenance,
                RuntimeFrameIdSeed {
                    frame_kind: kind,
                    source: RuntimeSource::Transcript,
                    ordinal: index as u32,
                    stable_key: "test",
                    source_span,
                },
            );
            frame.protocol =
                Some(crate::protocol_frames::ProtocolFrame::from_history_item(index, item).item);
            snapshot.push_frame(frame);
        }
        snapshot
    }

    fn compaction_config() -> CompactionConfig {
        CompactionConfig {
            tail_turns: 1,
            preserve_recent_tokens: Some(0),
            ..CompactionConfig::default()
        }
    }

    #[test]
    fn recognized_request_budget_overflow_is_narrow() {
        assert!(is_recognized_request_budget_overflow(&anyhow::anyhow!(
            "protected current context exceeds input budget: 10 > 9"
        )));
        assert!(is_recognized_request_budget_overflow(
            &anyhow::anyhow!("outer")
                .context("final prompt and tools exceed selected input budget")
        ));
        for message in [
            "final prompt and tools exceed selected input budget: malformed",
            "protocol frame validation failed",
            "serialization failed",
        ] {
            assert!(!is_recognized_request_budget_overflow(&anyhow::anyhow!(
                message
            )));
        }
    }

    #[test]
    fn selection_retires_complete_tool_call_group_and_merges_source_spans() {
        let history = vec![
            HistoryItem::user("old"),
            HistoryItem::AssistantToolCalls {
                text: None,
                calls: vec![tool_call("call-1")],
            },
            HistoryItem::ToolOutput {
                call_id: "call-1".into(),
                output_json: "{}".into(),
            },
            HistoryItem::assistant("done"),
            HistoryItem::user("current"),
        ];
        let selection = select_compaction_segments(&history, 4, &compaction_config(), 0)
            .expect("complete old group is eligible");

        assert_eq!(selection.head_for_summary, history[..4]);
        assert_eq!(selection.tail_items.len(), 0);
        assert_eq!(selection.retired_frame_ids.len(), 4);
        assert_eq!(
            selection.retired_source_spans,
            vec![SourceSpan::new(1, 4).unwrap()]
        );
    }

    #[test]
    fn selection_protects_incomplete_groups_and_source_less_frames() {
        let history = vec![
            HistoryItem::user("old"),
            HistoryItem::AssistantToolCalls {
                text: None,
                calls: vec![tool_call("call-1")],
            },
            HistoryItem::user("current"),
        ];
        let selection = select_compaction_segments(&history, 2, &compaction_config(), 0)
            .expect("the retirable prefix ends before incomplete group");
        assert_eq!(selection.head_for_summary, history[..1]);
    }

    #[test]
    fn emergency_hard_core_leaves_completed_mid_turn_tools_unprotected() {
        // Contiguous-prefix retirement cannot jump past a protected opening user.
        // Emergency hard-core still matters: completed mid-turn tools leave the
        // turn-protect set so prune (and pre-turn retirement) can reclaim mass.
        let history = vec![
            HistoryItem::user("active turn"),
            HistoryItem::AssistantToolCalls {
                text: None,
                calls: vec![tool_call("complete")],
            },
            HistoryItem::ToolOutput {
                call_id: "complete".into(),
                output_json: "{}".into(),
            },
            HistoryItem::AssistantToolCalls {
                text: None,
                calls: vec![tool_call("pending")],
            },
        ];
        let mut snapshot = snapshot_for(&history, None);
        let protocol = snapshot
            .active_protocol_frames()
            .iter()
            .filter_map(|frame| frame.runtime_frame_id)
            .collect::<Vec<_>>();
        snapshot.set_turn_protected_frame_ids(protocol.clone());
        protect_hard_core(&mut snapshot, Some(0));

        let turn = &snapshot.compaction.turn_protected_frame_ids;
        assert!(turn.contains(&protocol[0]), "opening user stays hard-protected");
        assert!(turn.contains(&protocol[3]), "incomplete tool call stays protected");
        assert!(
            !turn.contains(&protocol[1]) && !turn.contains(&protocol[2]),
            "completed mid-turn tools must leave turn protection for prune/reclaim"
        );

        // With the opening user protected, contiguous selection has no retirable prefix.
        let error = select_runtime_compaction_segments(
            &snapshot,
            &CompactionConfig {
                tail_turns: 0,
                preserve_recent_tokens: Some(0),
                ..CompactionConfig::default()
            },
            0,
        )
        .expect_err("cannot retire past protected opening user");
        assert!(is_nothing_to_compact_error(&error));
    }

    #[test]
    fn selection_protects_explicitly_protected_runtime_frames() {
        let history = vec![
            HistoryItem::user("old"),
            HistoryItem::assistant("answer"),
            HistoryItem::user("current"),
        ];
        let mut snapshot = snapshot_for(&history, None);
        snapshot.set_protected_frame_ids(vec![snapshot.frames[1].id, snapshot.frames[2].id]);

        let selection = select_runtime_compaction_segments(&snapshot, &compaction_config(), 0)
            .expect("the unprotected prefix remains eligible");

        assert_eq!(selection.head_for_summary, history[..1]);
        assert!(selection.tail_items.is_empty());
    }

    #[test]
    fn selection_does_not_retire_frame_with_turn_and_explicit_protection() {
        let history = vec![
            HistoryItem::user("old"),
            HistoryItem::assistant("explicitly retained"),
            HistoryItem::user("current"),
        ];
        let mut snapshot = snapshot_for(&history, None);
        let retained_id = snapshot.frames[1].id;
        // Protect the retained assistant + current user, leave the older user free.
        snapshot.set_turn_protected_frame_ids(vec![retained_id, snapshot.frames[2].id]);
        snapshot.set_protected_frame_ids(vec![retained_id]);

        let selection = select_runtime_compaction_segments(&snapshot, &compaction_config(), 0)
            .expect("the prefix before explicit protection remains eligible");

        assert_eq!(selection.head_for_summary, history[..1]);
        assert!(!selection.retired_frame_ids.contains(&retained_id));
    }

    #[test]
    fn selection_co_retires_fully_covered_folded_contributor_frame() {
        let history = vec![
            HistoryItem::user("old"),
            HistoryItem::assistant("answer"),
            HistoryItem::user("current"),
        ];
        let mut snapshot = snapshot_for(&history, None);
        let folded = RuntimeFrame::new(
            RuntimeFrameKind::FoldedOutput,
            FrameVisibility::Folded,
            RuntimeFrameProvenance::new(RuntimeSource::FoldedOutput)
                .with_span(SourceSpan::new(1, 1).unwrap()),
            RuntimeFrameIdSeed {
                frame_kind: RuntimeFrameKind::FoldedOutput,
                source: RuntimeSource::FoldedOutput,
                ordinal: 0,
                stable_key: "folded contributor",
                source_span: Some(SourceSpan::new(1, 1).unwrap()),
            },
        );
        let folded_id = folded.id;
        snapshot.push_frame(folded);
        snapshot.push_prompt_contributor(PromptContributorPlaceholder {
            contributor_id: "folded".into(),
            kind: PromptContributorKind::FoldedOutputSummary,
            label: None,
            provenance: RuntimeFrameProvenance::new(RuntimeSource::PromptContributor),
            retains_raw_sources: false,
            frame_ids: vec![folded_id],
            source_frame_ids: Vec::new(),
        });

        let selection = select_runtime_compaction_segments(&snapshot, &compaction_config(), 0)
            .expect("folded projection is dependent on the retired source");
        assert!(selection.retired_frame_ids.contains(&folded_id));

        snapshot.frames.last_mut().unwrap().provenance.source_span =
            Some(SourceSpan::new(99, 99).unwrap());
        assert!(select_runtime_compaction_segments(&snapshot, &compaction_config(), 0).is_ok());
    }

    #[test]
    fn ordinary_visible_context_frame_co_retires_without_typed_coverage() {
        let history = vec![
            HistoryItem::user("old"),
            HistoryItem::assistant("answer"),
            HistoryItem::user("current"),
        ];
        let mut snapshot = snapshot_for(&history, None);
        let context = RuntimeFrame::new(
            RuntimeFrameKind::ContextBlock,
            FrameVisibility::Active,
            RuntimeFrameProvenance::new(RuntimeSource::ContextView)
                .with_span(SourceSpan::new(1, 1).unwrap())
                .with_source_id("ordinary-visible"),
            RuntimeFrameIdSeed {
                frame_kind: RuntimeFrameKind::ContextBlock,
                source: RuntimeSource::ContextView,
                ordinal: 0,
                stable_key: "ordinary-visible",
                source_span: Some(SourceSpan::new(1, 1).unwrap()),
            },
        );
        let context_id = context.id;
        snapshot.push_frame(context);
        snapshot.push_prompt_contributor(PromptContributorPlaceholder {
            contributor_id: "context-view-active".into(),
            kind: PromptContributorKind::ContextMaterial,
            label: Some("Active context view".into()),
            provenance: RuntimeFrameProvenance::new(RuntimeSource::ContextView),
            retains_raw_sources: true,
            frame_ids: Vec::new(),
            source_frame_ids: Vec::new(),
        });
        snapshot.set_protected_frame_ids(vec![snapshot.frames[2].id]);

        let selection = select_runtime_compaction_segments(&snapshot, &compaction_config(), 0)
            .expect("ordinary projection co-retires");
        assert!(selection.retired_frame_ids.contains(&context_id));

        assert!(
            !snapshot
                .compaction
                .protected_frame_ids
                .contains(&context_id)
        );
    }

    #[test]
    fn hard_pinned_or_opened_context_frame_retains_its_source() {
        let history = vec![HistoryItem::user("old"), HistoryItem::user("current")];
        for authority in ["hard", "pinned", "opened"] {
            let mut snapshot = snapshot_for(&history, None);
            let context = RuntimeFrame::new(
                RuntimeFrameKind::ContextBlock,
                FrameVisibility::Active,
                RuntimeFrameProvenance::new(RuntimeSource::ContextView)
                    .with_span(SourceSpan::new(1, 1).unwrap())
                    .with_source_id(authority),
                RuntimeFrameIdSeed {
                    frame_kind: RuntimeFrameKind::ContextBlock,
                    source: RuntimeSource::ContextView,
                    ordinal: 0,
                    stable_key: authority,
                    source_span: Some(SourceSpan::new(1, 1).unwrap()),
                },
            );
            let context_id = context.id;
            snapshot.push_frame(context);
            snapshot.push_prompt_contributor(PromptContributorPlaceholder {
                contributor_id: format!("context-view-{authority}"),
                kind: PromptContributorKind::ContextMaterial,
                label: None,
                provenance: RuntimeFrameProvenance::new(RuntimeSource::ContextView),
                retains_raw_sources: true,
                frame_ids: vec![context_id],
                source_frame_ids: Vec::new(),
            });
            snapshot.set_protected_frame_ids(vec![snapshot.frames[1].id]);

            assert!(
                select_runtime_compaction_segments(&snapshot, &compaction_config(), 0).is_err()
            );
        }
    }

    #[test]
    fn standard_selection_retires_fully_covered_soft_retaining_context_material() {
        // Soft-retain materials no longer veto Standard selection; only hard protect does.
        let history = vec![
            HistoryItem::user("old user"),
            HistoryItem::assistant("old assistant"),
        ];
        let mut snapshot = snapshot_for(&history, None);
        let seed = RuntimeFrameIdSeed {
            frame_kind: RuntimeFrameKind::ContextBlock,
            source: RuntimeSource::ContextView,
            ordinal: 0,
            stable_key: "block-retaining-standard",
            source_span: Some(SourceSpan::new(1, 2).expect("span")),
        };
        let context_id = RuntimeFrameId::from_seed(&seed);
        snapshot.push_frame(RuntimeFrame::new(
            RuntimeFrameKind::ContextBlock,
            FrameVisibility::Active,
            RuntimeFrameProvenance::new(RuntimeSource::ContextView)
                .with_span(SourceSpan::new(1, 2).expect("span")),
            seed,
        ));
        snapshot.push_prompt_contributor(PromptContributorPlaceholder {
            contributor_id: "retaining-standard".into(),
            kind: PromptContributorKind::ContextMaterial,
            label: Some("retaining".into()),
            provenance: RuntimeFrameProvenance::new(RuntimeSource::ContextView)
                .with_span(SourceSpan::new(1, 2).expect("span")),
            retains_raw_sources: true,
            frame_ids: vec![context_id],
            source_frame_ids: Vec::new(),
        });
        snapshot.recompute_protected_frame_ids();

        let selection = select_runtime_compaction_segments(&snapshot, &CompactionConfig {
            tail_turns: 0,
            preserve_recent_tokens: Some(0),
            prune: true,
        }, 0)
        .expect("standard mode retires history despite soft retain");
        let protocol_ids = snapshot
            .active_protocol_frames()
            .iter()
            .filter_map(|frame| frame.runtime_frame_id)
            .collect::<Vec<_>>();
        for id in &protocol_ids {
            assert!(selection.retired_frame_ids.contains(id));
        }
    }


    #[test]
    fn selection_co_retires_fully_covered_retaining_context_material() {
        // Regression: soft-retaining materials used to join protected_frame_ids
        // and block co-retirement even when their whole source span was selected.
        let history = vec![
            HistoryItem::user("old user"),
            HistoryItem::assistant("old assistant"),
        ];
        let mut snapshot = snapshot_for(&history, None);
        let seed = RuntimeFrameIdSeed {
            frame_kind: RuntimeFrameKind::ContextBlock,
            source: RuntimeSource::ContextView,
            ordinal: 0,
            stable_key: "block-retaining",
            source_span: Some(SourceSpan::new(1, 1).expect("span")),
        };
        let context_id = RuntimeFrameId::from_seed(&seed);
        snapshot.push_frame(
            RuntimeFrame::new(
                RuntimeFrameKind::ContextBlock,
                FrameVisibility::Active,
                RuntimeFrameProvenance::new(RuntimeSource::ContextView)
                    .with_span(SourceSpan::new(1, 1).expect("span")),
                seed,
            )
            .with_summary("retaining block"),
        );
        snapshot.push_prompt_contributor(PromptContributorPlaceholder {
            contributor_id: "retaining-context".into(),
            kind: PromptContributorKind::ContextMaterial,
            label: Some("retaining".into()),
            provenance: RuntimeFrameProvenance::new(RuntimeSource::ContextView)
                .with_span(SourceSpan::new(1, 1).expect("span")),
            retains_raw_sources: true,
            frame_ids: vec![context_id],
            source_frame_ids: Vec::new(),
        });
        snapshot.recompute_protected_frame_ids();
        assert!(
            !snapshot
                .compaction
                .protected_frame_ids
                .contains(&context_id),
            "soft-retaining material must not join hard protected_frame_ids"
        );

        let selection = select_runtime_compaction_segments(
            &snapshot,
            &CompactionConfig {
                tail_turns: 0,
                preserve_recent_tokens: Some(0),
                prune: true,
            },
            0,
        )
        .expect("fully covered retaining material must co-retire with its source span");
        assert!(selection.retired_frame_ids.contains(&context_id));
        assert!(!selection.retired_frame_ids.is_empty());
    }

    #[test]
    fn selection_still_blocks_when_hard_protected_prefix_has_no_retirable_history() {
        // Hard protect on all protocol frames leaves no safe retirable prefix.
        let history = vec![
            HistoryItem::user("old user"),
            HistoryItem::assistant("old assistant"),
            HistoryItem::user("current user"),
        ];
        let mut snapshot = snapshot_for(&history, None);
        let protocol = snapshot
            .active_protocol_frames()
            .iter()
            .filter_map(|frame| frame.runtime_frame_id)
            .collect::<Vec<_>>();
        snapshot.set_protected_frame_ids(protocol.clone());
        snapshot.recompute_protected_frame_ids();

        let error = select_runtime_compaction_segments(
            &snapshot,
            &CompactionConfig {
                tail_turns: 0,
                preserve_recent_tokens: Some(0),
                prune: true,
            },
            0,
        )
        .expect_err("fully hard-protected history has no retirable prefix");
        assert!(is_nothing_to_compact_error(&error));
    }


    #[test]
    fn emergency_and_pressure_selection_ignore_soft_retaining_spans() {
        // Soft retain must not block RequestPressure or Emergency selection.
        let history = vec![
            HistoryItem::user("old user"),
            HistoryItem::assistant("old assistant"),
        ];
        let mut snapshot = snapshot_for(&history, None);
        let seed = RuntimeFrameIdSeed {
            frame_kind: RuntimeFrameKind::ContextBlock,
            source: RuntimeSource::ContextView,
            ordinal: 0,
            stable_key: "block-soft-pressure",
            source_span: Some(SourceSpan::new(1, 2).expect("span")),
        };
        let context_id = RuntimeFrameId::from_seed(&seed);
        snapshot.push_frame(RuntimeFrame::new(
            RuntimeFrameKind::ContextBlock,
            FrameVisibility::Active,
            RuntimeFrameProvenance::new(RuntimeSource::ContextView)
                .with_span(SourceSpan::new(1, 2).expect("span")),
            seed,
        ));
        snapshot.push_prompt_contributor(PromptContributorPlaceholder {
            contributor_id: "soft-wide".into(),
            kind: PromptContributorKind::ContextMaterial,
            label: Some("retaining".into()),
            provenance: RuntimeFrameProvenance::new(RuntimeSource::ContextView)
                .with_span(SourceSpan::new(1, 2).expect("span")),
            retains_raw_sources: true,
            frame_ids: vec![context_id],
            source_frame_ids: Vec::new(),
        });
        snapshot.recompute_protected_frame_ids();

        let selection = select_runtime_compaction_segments(
            &snapshot,
            &CompactionConfig {
                tail_turns: 0,
                preserve_recent_tokens: Some(0),
                prune: true,
            },
            0,
        )
        .expect("soft retain must not block selection");
        let protocol_ids = snapshot
            .active_protocol_frames()
            .iter()
            .filter_map(|frame| frame.runtime_frame_id)
            .collect::<Vec<_>>();
        for id in &protocol_ids {
            assert!(
                selection.retired_frame_ids.contains(id),
                "must retire protocol history"
            );
        }
    }


    #[test]
    fn retirement_blocker_excludes_soft_retain_frame_ids_in_all_modes() {
        let history = vec![
            HistoryItem::user("old user"),
            HistoryItem::assistant("old assistant"),
        ];
        let mut snapshot = snapshot_for(&history, None);
        let old_user_id = snapshot.frames[0].id;
        snapshot.push_prompt_contributor(PromptContributorPlaceholder {
            contributor_id: "soft-on-protocol".into(),
            kind: PromptContributorKind::ContextMaterial,
            label: Some("retaining".into()),
            provenance: RuntimeFrameProvenance::new(RuntimeSource::ContextView)
                .with_span(SourceSpan::new(1, 1).expect("span")),
            retains_raw_sources: true,
            frame_ids: vec![old_user_id],
            source_frame_ids: Vec::new(),
        });
        snapshot.recompute_protected_frame_ids();

                let blockers = retirement_blocker_frame_ids(&snapshot);
        assert!(
            !blockers.contains(&old_user_id),
            "soft retain must not pin protocol history"
        );
    }


    #[test]
    fn emergency_hard_core_clears_whole_turn_protection_but_keeps_user_and_incomplete() {
        // Contiguous-prefix selection cannot retire mid-turn tools past a protected
        // current user. Emergency instead narrows turn protection so completed
        // mid-turn tools are eligible for prune (and pre-turn history for retire).
        let history = vec![
            HistoryItem::user("older"),
            HistoryItem::assistant("older reply"),
            HistoryItem::user("current user"),
            HistoryItem::AssistantToolCalls {
                text: None,
                calls: vec![HistoryToolCall {
                    call_id: "done".into(),
                    name: "fs__read".into(),
                    arguments_json: "{}".into(),
                }],
            },
            HistoryItem::ToolOutput {
                call_id: "done".into(),
                output_json: "x".repeat(100),
            },
            HistoryItem::AssistantToolCalls {
                text: None,
                calls: vec![HistoryToolCall {
                    call_id: "pending".into(),
                    name: "shell__exec".into(),
                    arguments_json: "{}".into(),
                }],
            },
        ];
        let mut snapshot = snapshot_for(&history, None);
        let protocol = snapshot
            .active_protocol_frames()
            .iter()
            .filter_map(|frame| frame.runtime_frame_id)
            .collect::<Vec<_>>();
        // Simulate live turn protection covering the whole active turn.
        snapshot.set_turn_protected_frame_ids(protocol[2..].to_vec());
        protect_hard_core(&mut snapshot, Some(2));

        let turn = &snapshot.compaction.turn_protected_frame_ids;
        assert!(turn.contains(&protocol[2]), "current user stays hard-protected");
        assert!(turn.contains(&protocol[5]), "incomplete tool call stays protected");
        assert!(
            !turn.contains(&protocol[3]) && !turn.contains(&protocol[4]),
            "completed mid-turn tools must leave turn protection for prune/reclaim"
        );

        let selection = select_runtime_compaction_segments(
            &snapshot,
            &CompactionConfig {
                tail_turns: 0,
                preserve_recent_tokens: Some(0),
                prune: true,
            },
            0,
        )
        .expect("emergency still retires pre-turn history");
        assert!(selection.retired_frame_ids.contains(&protocol[0]));
        assert!(selection.retired_frame_ids.contains(&protocol[1]));
        assert!(!selection.retired_frame_ids.contains(&protocol[2]));
    }

    #[test]
    fn pressure_selection_retires_history_before_current_turn_despite_retaining_materials() {
        let history = vec![
            HistoryItem::user("old user"),
            HistoryItem::assistant("old assistant"),
            HistoryItem::user("current user"),
            HistoryItem::assistant("current assistant"),
        ];
        let mut snapshot = snapshot_for(&history, None);
        let seed = RuntimeFrameIdSeed {
            frame_kind: RuntimeFrameKind::ContextBlock,
            source: RuntimeSource::ContextView,
            ordinal: 0,
            stable_key: "block-old-retaining",
            source_span: Some(SourceSpan::new(1, 1).expect("span")),
        };
        let context_id = RuntimeFrameId::from_seed(&seed);
        snapshot.push_frame(
            RuntimeFrame::new(
                RuntimeFrameKind::ContextBlock,
                FrameVisibility::Active,
                RuntimeFrameProvenance::new(RuntimeSource::ContextView)
                    .with_span(SourceSpan::new(1, 1).expect("span")),
                seed,
            )
            .with_summary("old retaining block"),
        );
        snapshot.push_prompt_contributor(PromptContributorPlaceholder {
            contributor_id: "old-retaining".into(),
            kind: PromptContributorKind::ContextMaterial,
            label: Some("retaining".into()),
            provenance: RuntimeFrameProvenance::new(RuntimeSource::ContextView)
                .with_span(SourceSpan::new(1, 1).expect("span")),
            retains_raw_sources: true,
            frame_ids: vec![context_id],
            source_frame_ids: Vec::new(),
        });
        protect_hard_core(&mut snapshot, Some(2));

        let selection = select_runtime_compaction_segments(
            &snapshot,
            &CompactionConfig {
                tail_turns: 0,
                preserve_recent_tokens: Some(0),
                prune: true,
            },
            0,
        )
        .expect("pressure selection must retire pre-turn history with retaining materials");
        // Hard-core pins only the current-turn opening; protocol prefix still
        // retires and the current turn does not.
        let active_protocol = snapshot
            .active_protocol_frames()
            .iter()
            .filter_map(|frame| frame.runtime_frame_id)
            .collect::<Vec<_>>();
        assert!(selection.retired_frame_ids.contains(&active_protocol[0]));
        assert!(selection.retired_frame_ids.contains(&active_protocol[1]));
        assert!(!selection.retired_frame_ids.contains(&active_protocol[2]));
        assert!(!selection.retired_frame_ids.contains(&active_protocol[3]));
        assert!(selection.retired_frame_ids.len() >= 2);
    }

    #[test]
    fn selection_binds_history_to_transcript_frames_not_snapshot_positions() {
        let history = vec![
            HistoryItem::user("same user message"),
            HistoryItem::assistant("same assistant message"),
            HistoryItem::user("same user message"),
            HistoryItem::assistant("same assistant message"),
        ];
        let mut snapshot = snapshot_for(&history, None);
        let inserted_metadata = RuntimeFrame::new(
            RuntimeFrameKind::Metadata,
            FrameVisibility::Active,
            RuntimeFrameProvenance::new(RuntimeSource::SessionState),
            RuntimeFrameIdSeed {
                frame_kind: RuntimeFrameKind::Metadata,
                source: RuntimeSource::SessionState,
                ordinal: 0,
                stable_key: "inserted-runtime-metadata",
                source_span: None,
            },
        );
        snapshot.frames.insert(0, inserted_metadata);
        let selection = select_compaction_segments(&history, 3, &compaction_config(), 0)
            .expect("selection follows transcript frame identity");

        assert_eq!(selection.retired_frame_ids.len(), 3);
        assert_eq!(selection.head_for_summary, history[..3]);
        assert!(selection.tail_items.is_empty());
        assert_eq!(
            selection.retired_source_spans,
            vec![SourceSpan::new(1, 3).unwrap()]
        );
    }

    #[test]
    fn selection_keeps_safe_prefix_before_later_retained_span_overlap() {
        let history = vec![
            HistoryItem::user("old independent"),
            HistoryItem::assistant("old overlapping"),
            HistoryItem::user("current"),
        ];
        let mut snapshot = snapshot_for(&history, None);
        snapshot.frames[1].provenance.source_span = Some(SourceSpan::new(2, 3).unwrap());
        snapshot.set_protected_frame_ids(vec![snapshot.frames[2].id]);

        let selection = select_runtime_compaction_segments(&snapshot, &compaction_config(), 0)
            .expect("the independent prefix remains safe");

        assert_eq!(selection.head_for_summary, history[..1]);
        assert_eq!(selection.retired_frame_ids, vec![snapshot.frames[0].id]);
    }

    #[test]
    fn selection_rejects_contributor_span_overlap_but_preserves_prior_prefix() {
        let history = vec![
            HistoryItem::user("old independent"),
            HistoryItem::assistant("old overlapping"),
            HistoryItem::user("current"),
        ];
        let mut snapshot = snapshot_for(&history, None);
        snapshot.push_prompt_contributor(crate::runtime_context::PromptContributorPlaceholder {
            contributor_id: "retained-context".into(),
            kind: crate::runtime_context::PromptContributorKind::RuntimeContext,
            label: None,
            provenance: RuntimeFrameProvenance::new(RuntimeSource::ContextView)
                .with_span(SourceSpan::new(2, 2).unwrap()),
            retains_raw_sources: true,
            frame_ids: Vec::new(),
            source_frame_ids: Vec::new(),
        });
        snapshot.set_protected_frame_ids(vec![snapshot.frames[2].id]);

        let selection = select_runtime_compaction_segments(&snapshot, &compaction_config(), 0)
            .expect("soft-retain contributor spans do not veto overlapping history");

        // Soft retain is advisory for assembly only; overlapping assistant may retire.
        assert_eq!(selection.head_for_summary, history[..2]);
        assert!(selection.retired_frame_ids.contains(&snapshot.frames[0].id));
        assert!(selection.retired_frame_ids.contains(&snapshot.frames[1].id));
    }

    #[test]
    fn non_retaining_contributor_span_does_not_block_compaction() {
        let history = vec![
            HistoryItem::user("old independent"),
            HistoryItem::assistant("old overlapping"),
            HistoryItem::user("current"),
        ];
        let mut snapshot = snapshot_for(&history, None);
        snapshot.push_prompt_contributor(crate::runtime_context::PromptContributorPlaceholder {
            contributor_id: "dependent-context".into(),
            kind: crate::runtime_context::PromptContributorKind::RuntimeContext,
            label: None,
            provenance: RuntimeFrameProvenance::new(RuntimeSource::ContextView)
                .with_span(SourceSpan::new(2, 2).unwrap()),
            retains_raw_sources: false,
            frame_ids: Vec::new(),
            source_frame_ids: Vec::new(),
        });
        snapshot.set_protected_frame_ids(vec![snapshot.frames[2].id]);

        let selection = select_runtime_compaction_segments(&snapshot, &compaction_config(), 0)
            .expect("non-retaining contributor does not block its covered source");

        assert_eq!(selection.head_for_summary, history[..2]);
    }

    #[test]
    fn summary_history_is_exact_runtime_payload_projection_in_protocol_order() {
        let history = vec![
            HistoryItem::user("old"),
            HistoryItem::assistant("answer"),
            HistoryItem::user("current"),
        ];
        let mut snapshot = snapshot_for(&history, None);
        snapshot.set_protected_frame_ids(vec![snapshot.frames[2].id]);

        let selection = select_runtime_compaction_segments(&snapshot, &compaction_config(), 0)
            .expect("old protocol payloads are selected");
        let payloads = snapshot
            .active_protocol_frames()
            .into_iter()
            .filter(|frame| {
                selection
                    .retired_frame_ids
                    .contains(&frame.runtime_frame_id.unwrap())
            })
            .map(|frame| frame.to_history_item())
            .collect::<Vec<_>>();

        assert_eq!(selection.head_for_summary, payloads);
    }

    #[test]
    fn selection_rejects_malformed_protocol_before_terminal_summary_noop() {
        let history = vec![
            HistoryItem::context_summary("already compacted"),
            HistoryItem::ToolOutput {
                call_id: "orphan".into(),
                output_json: "{}".into(),
            },
        ];
        let snapshot = snapshot_for(&history, None);

        let error = select_runtime_compaction_segments(&snapshot, &compaction_config(), 0)
            .expect_err("orphan output is malformed even after a terminal summary");

        assert!(error.to_string().contains("orphan tool output"));
    }

    #[test]
    fn selection_rejects_duplicate_frame_ids_before_noop() {
        let history = vec![HistoryItem::context_summary("already compacted")];
        let mut snapshot = snapshot_for(&history, None);
        snapshot.frames.push(snapshot.frames[0].clone());

        let error = select_runtime_compaction_segments(&snapshot, &compaction_config(), 0)
            .expect_err("duplicate runtime IDs must not become a no-op");

        assert!(error.to_string().contains("duplicate frame id"));
    }

    #[test]
    fn selection_allows_well_formed_terminal_summary_noop() {
        let history = vec![HistoryItem::context_summary("already compacted")];
        let snapshot = snapshot_for(&history, None);

        let error = select_runtime_compaction_segments(&snapshot, &compaction_config(), 0)
            .expect_err("a terminal summary has no historical candidates");

        assert!(is_nothing_to_compact_error(&error));
    }

    #[test]
    fn healed_missing_protocol_span_becomes_retirable() {
        // Producer defect: active protocol history without source_span must not
        // permanently hard-fail compaction once a safe synthetic span exists.
        let history = vec![
            HistoryItem::user("old"),
            HistoryItem::assistant("answer"),
            HistoryItem::user("current"),
        ];
        let mut snapshot = snapshot_for(&history, Some(0));
        snapshot.set_protected_frame_ids(vec![snapshot.frames[2].id]);

        assert!(
            select_runtime_compaction_segments(&snapshot, &compaction_config(), 0).is_err(),
            "missing span still blocks before heal"
        );

        super::ensure_active_protocol_source_spans(&mut snapshot);
        let selection = select_runtime_compaction_segments(&snapshot, &compaction_config(), 0)
            .expect("healed span is retirable");
        assert!(
            !selection.retired_frame_ids.is_empty(),
            "heal must unlock a non-empty safe retirement prefix"
        );
        assert!(
            selection
                .head_for_summary
                .iter()
                .any(|item| matches!(item, HistoryItem::UserMessage { .. })),
            "healed prefix should include the previously unspannable user turn"
        );
    }

    #[test]
    fn pruned_tool_output_uses_explicit_structural_marker() {
        let pruned = build_pruned_tool_output_json(&"x".repeat(10_000), Some("skill"));
        let value: Value = serde_json::from_str(&pruned).expect("pruned output is JSON");

        assert_eq!(value["_compaction"]["pruned"], Value::Bool(true));
        assert_eq!(value["_compaction"]["tool"], Value::String("skill".into()));
        assert!(value.get("data").is_none());
    }
}
