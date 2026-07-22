use super::*;
use crate::protocol_frames::{ProtocolFrameItem, ToolCallGroupStatus, analyze_history_items};
use crate::runtime_context::{
    FrameVisibility, RuntimeFrameId, RuntimeSnapshot, RuntimeSource, SourceSpan,
};
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



fn restore_pre_compaction_agent_state<C>(
    agent: &mut Agent<C>,
    runtime_snapshot: RuntimeSnapshot,
    protocol_frames: Vec<crate::protocol_frames::ProtocolFrame>,
    history: Vec<HistoryItem>,
    current_turn_start_index: Option<usize>,
    active_epoch: Option<super::ActiveEpoch>,
) where
    C: Config + Clone,
{
    agent.runtime_snapshot = runtime_snapshot;
    agent.protocol_frames = protocol_frames;
    agent.history = history;
    agent.turn.current_turn_start_index = current_turn_start_index;
    agent.active_epoch = active_epoch;
}

/// Pressure compact: heal spans → select → summarize → commit candidate → rebuild request.
/// SoftUnsafe is not re-checked; only hard budget admission matters downstream.
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
        agent.refresh_runtime_snapshot_from_provider()?;
        validate_compaction_runtime_state(agent)?;
        let aggressive = CompactionConfig {
            tail_turns: 0,
            preserve_recent_tokens: Some(0),
            ..agent.compaction_config.clone()
        };
        let mut healed = healed_snapshot_for_selection(&agent.runtime_snapshot);
        protect_current_turn_for_pressure_selection(
            &mut healed,
            agent.turn.current_turn_start_index,
        );
        let selection = match select_compaction_attempt(&healed, &aggressive, trigger)? {
            CompactionSelectionResult::Selected(selection) => selection,
            CompactionSelectionResult::NoProgress(no_progress) => {
                on_event(AgentEvent::ContextCompactionNoProgress(no_progress.clone())).await?;
                bail!(
                    "request pressure has no compactable context: {}",
                    diagnostic_labels(&no_progress.blockers)
                );
            }
        };
        // Capture the pre-attempt live state first. Working-copy heal/protect
        // must never leak into rollback baselines.
        let previous_snapshot = agent.runtime_snapshot.clone();
        let previous_protocol_frames = agent.protocol_frames.clone();
        let previous_history = agent.history.clone();
        let previous_turn_start_index = agent.turn.current_turn_start_index;
        let previous_active_epoch = agent.active_epoch.clone();

        // Promote healed spans into the candidate transaction only after we know
        // selection made progress. Failure paths below still roll this back.
        agent.runtime_snapshot = healed;
        super::sync_protocol_frame_provenance_from_snapshot(
            &mut agent.protocol_frames,
            &agent.runtime_snapshot,
        );
        let prepared =
            match compact_selected_context(agent, selection, on_event, None).await {
                Ok(prepared) => prepared,
                Err(error) => {
                    restore_pre_compaction_agent_state(

                        agent,

                        previous_snapshot,

                        previous_protocol_frames,

                        previous_history,

                        previous_turn_start_index,

                        previous_active_epoch,

                    );
                    return Err(error);
                }
            };
        let PreparedCompaction {
            event,
            snapshot,
            protocol_frames,
            history,
            current_turn_start_index,
            ..
        } = prepared;

        agent.runtime_snapshot = snapshot;
        agent.protocol_frames = protocol_frames;
        agent.history = history;
        agent.turn.current_turn_start_index = current_turn_start_index;
        agent.active_epoch = None;

        let successor = (|| -> Result<PreparedRequestBuild> {
            let epoch_preview = agent.preview_active_epoch(protocol, turn_prelude, tool_definitions)?;
            Ok(PreparedRequestBuild {
                protected_start_index: agent
                    .turn
                    .current_turn_start_index
                    .unwrap_or(agent.history.len()),
                build: epoch_preview.build.clone(),
                epoch_preview,
            })
        })();

        let result = match successor {
            Ok(successor) => on_event(AgentEvent::ContextCompacted(event)).await.map(|_| successor),
            Err(error) => Err(error),
        };
        if result.is_err() {
            // Roll back candidate transaction, including any heal promotion.
            restore_pre_compaction_agent_state(

                agent,

                previous_snapshot,

                previous_protocol_frames,

                previous_history,

                previous_turn_start_index,

                previous_active_epoch,

            );
        }
        result
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
        agent.refresh_runtime_snapshot_from_provider()?;
        validate_compaction_runtime_state(agent)?;
        let aggressive = CompactionConfig {
            tail_turns: 0,
            preserve_recent_tokens: Some(0),
            ..agent.compaction_config.clone()
        };
        let healed = healed_snapshot_for_selection(&agent.runtime_snapshot);
        match select_compaction_attempt(&healed, &aggressive, trigger)? {
            CompactionSelectionResult::NoProgress(no_progress) => {
                on_event(AgentEvent::ContextCompactionNoProgress(no_progress.clone())).await?;
                Ok(CompactionAttemptOutcome::NoProgress(no_progress))
            }
            CompactionSelectionResult::Selected(selection) => {
                // Heal is applied only for selection identity. Live state stays
                // untouched until the durable ContextCompacted callback succeeds.
                let previous_snapshot = agent.runtime_snapshot.clone();
                let previous_protocol_frames = agent.protocol_frames.clone();
                agent.runtime_snapshot = healed;
                super::sync_protocol_frame_provenance_from_snapshot(
                    &mut agent.protocol_frames,
                    &agent.runtime_snapshot,
                );
                let prepared = match compact_selected_context(
                    agent,
                    selection,
                    on_event,
                    on_delta,
                )
                .await
                {
                    Ok(prepared) => prepared,
                    Err(error) => {
                        agent.runtime_snapshot = previous_snapshot;
                        agent.protocol_frames = previous_protocol_frames;
                        return Err(error);
                    }
                };
                if let Err(error) =
                    on_event(AgentEvent::ContextCompacted(prepared.event.clone())).await
                {
                    agent.runtime_snapshot = previous_snapshot;
                    agent.protocol_frames = previous_protocol_frames;
                    return Err(error);
                }
                agent.commit_prepared_runtime_compaction(
                    prepared.snapshot,
                    prepared.protocol_frames,
                    prepared.history,
                    prepared.current_turn_start_index,
                );
                Ok(CompactionAttemptOutcome::Compacted {
                    retained_items: prepared.retained_items,
                })
            }
        }
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

    // Prune intentionally mutates live state when it makes progress; heal first
    // so missing spans do not silently turn into a no-op.
    super::ensure_active_protocol_source_spans(&mut agent.runtime_snapshot);
    super::sync_protocol_frame_provenance_from_snapshot(
        &mut agent.protocol_frames,
        &agent.runtime_snapshot,
    );
    let selection = match select_runtime_compaction_segments(
        &agent.runtime_snapshot,
        &agent.compaction_config,
        preserve_recent_budget,
    ) {
        Ok(selection) => selection,
        Err(error) if is_nothing_to_compact_error(&error) => return Ok(()),
        Err(error) => return Err(error),
    };

    let protected_ids = selection
        .retired_frame_ids
        .iter()
        .copied()
        .collect::<BTreeSet<_>>();
    let call_names = tool_output_names_by_frame_id(&agent.runtime_snapshot);
    let mut snapshot = agent.runtime_snapshot.clone();
    let mut changed = false;
    for frame in &mut snapshot.frames {
        if !protected_ids.contains(&frame.id) {
            continue;
        }
        let Some(ProtocolFrameItem::ToolOutput {
            call_id: _,
            output_json,
        }) = frame.protocol.as_mut()
        else {
            continue;
        };
        if output_json.chars().count() < COMPACTION_PRUNE_MIN_OUTPUT_CHARS {
            continue;
        }
        if output_json.contains(COMPACTION_PRUNED_MARKER) {
            continue;
        }
        // Skill payloads are durable material sources; never structural-prune them.
        let tool_name = call_names.get(&frame.id).map(String::as_str);
        if tool_name.is_some_and(is_skill_tool_name) {
            continue;
        }
        *output_json = build_pruned_tool_output_json(output_json, tool_name);
        frame.summary = Some(output_json.clone());
        changed = true;
    }

    if changed {
        snapshot.validate_references()?;
        agent.runtime_snapshot = snapshot;
        agent.sync_protocol_caches_from_runtime_snapshot()?;
    }
    Ok(())
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
    let original_history_items = agent.history.len();
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

    // Pruning belongs to this candidate transaction.  Never prune the live
    // snapshot before the durable compaction record acknowledges it.
    let candidate = prune_tool_outputs_snapshot(agent, &agent.runtime_snapshot, &selection)?;
    let mut snapshot =
        agent.prepare_runtime_compaction_from_snapshot(
            &candidate,
            &selection,
            summary.clone(),
        )?;
    let current_turn_start_index =
        agent.rebased_current_turn_start_index_after_compaction(&selection, &mut snapshot)?;
    let protocol_frames = snapshot.active_protocol_frames();
    let history = crate::protocol_frames::history_items_from_frames(&protocol_frames);
    crate::protocol_frames::analyze_history_items(&history, current_turn_start_index)?;

    let event = ContextCompactionEvent {
        outcome: "succeeded".into(),
        summary,
        tail_start_index: selection.tail_start_index,
        original_history_items,
        retained_history_items: snapshot.active_history_items().len(),
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
    match select_runtime_compaction_segments_with_mode(
        snapshot,
        config,
        0,
        compaction_closure_mode_for_trigger(trigger),
    ) {
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

fn compaction_closure_mode_for_trigger(
    trigger: CompactionTrigger,
) -> crate::transcript::transcript_projection::CompactionClosureMode {
    match trigger {
        CompactionTrigger::RequestPressure => {
            crate::transcript::transcript_projection::CompactionClosureMode::RequestPressure
        }
        CompactionTrigger::Manual => {
            crate::transcript::transcript_projection::CompactionClosureMode::Standard
        }
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
    if snapshot.prompt_contributors.iter().any(|contributor| {
        contributor.retains_raw_sources
            && (contributor.provenance.source_span.is_some() || !contributor.frame_ids.is_empty())
    }) {
        blockers.insert(CompactionBlocker::RetainedSourceDependency);
    }
    if blockers.is_empty() {
        blockers.insert(CompactionBlocker::NoSafeBoundary);
    }
    Ok(blockers.into_iter().collect())
}

pub(super) fn select_runtime_compaction_segments(
    snapshot: &RuntimeSnapshot,
    config: &CompactionConfig,
    preserve_recent_budget: u64,
) -> Result<CompactionSelection> {
    select_runtime_compaction_segments_with_mode(
        snapshot,
        config,
        preserve_recent_budget,
        crate::transcript::transcript_projection::CompactionClosureMode::Standard,
    )
}

/// History-first selection: protocol history frames drive the prefix; the runtime
/// snapshot is only the view that hosts those frames and dependent projections.
pub(super) fn select_runtime_compaction_segments_with_mode(
    snapshot: &RuntimeSnapshot,
    config: &CompactionConfig,
    preserve_recent_budget: u64,
    closure_mode: crate::transcript::transcript_projection::CompactionClosureMode,
) -> Result<CompactionSelection> {
    // Runtime frame identity is the authority. Compatibility history is rendered
    // only after a prefix has been selected by those identities.
    // Validate before every no-op exit: a malformed snapshot is never a valid
    // "nothing to compact" result.
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
    let tail_turns = config.tail_turns.min(turn_ranges.len());
    let tail_candidate_start = turn_ranges
        .iter()
        .rev()
        .take(tail_turns)
        .map(|(start, _)| *start)
        .min()
        .unwrap_or(candidates.len());
    let preserve_budget = config
        .preserve_recent_tokens
        .unwrap_or(preserve_recent_budget);
    let tail_relative_start = trim_tail_to_valid_boundary(
        candidates,
        trim_tail_to_budget(
            candidates,
            &turn_ranges,
            tail_candidate_start,
            preserve_budget,
        ),
    );
    let requested_tail_start = base_start + tail_relative_start;
    let frame_by_id = snapshot
        .frames
        .iter()
        .map(|frame| (frame.id, frame))
        .collect::<BTreeMap<_, _>>();
    // Selection may retire completed active-turn material, but must retain all
    // independent protection authority. Keep this view separate from the
    // normalized request-protection union: a frame protected for both turn and
    // explicit/contributor reasons remains blocked.
    let protected_ids = retirement_blocker_frame_ids(snapshot);
    let mut unit_start_by_index = (0..frames.len()).collect::<Vec<_>>();
    let mut blocked_group_members = BTreeSet::new();
    for group in &transcript.tool_call_groups {
        let mut members = vec![group.assistant_index];
        members.extend(group.tool_output_indexes.iter().copied());
        let blocked = group.status != ToolCallGroupStatus::Complete
            || members.iter().any(|&index| {
                !retirable_frame(
                    frames[index]
                        .runtime_frame_id
                        .expect("runtime protocol frames have ids"),
                    &frame_by_id,
                    &protected_ids,
                )
            });
        for &index in &members {
            unit_start_by_index[index] = group.assistant_index;
            if blocked {
                blocked_group_members.insert(index);
            }
        }
    }
    let mut tail_start_index = requested_tail_start.min(frames.len());
    if tail_start_index < frames.len() {
        tail_start_index = unit_start_by_index[tail_start_index];
    }
    // Select the longest safe contiguous prefix. A later frame may overlap active
    // material, but that must not discard an earlier independent prefix.
    let requested_end = tail_start_index;
    tail_start_index = (base_start + 1..=requested_end)
        .rev()
        .find(|&end| {
            // Group boundaries and every selected member must be independently
            // retirable before its spans can be removed.
            if end < frames.len() && unit_start_by_index[end] != end {
                return false;
            }
            let selected_ids = frames[base_start..end]
                .iter()
                .map(|frame| {
                    frame
                        .runtime_frame_id
                        .expect("runtime protocol frames have ids")
                })
                .collect::<BTreeSet<_>>();
            if selected_ids.iter().any(|id| {
                !retirable_frame(*id, &frame_by_id, &protected_ids)
                    || frame_by_id[id].provenance.source_span.is_none()
            }) || (base_start..end).any(|index| blocked_group_members.contains(&index))
            {
                return false;
            }
            let retired_spans = canonical_runtime_retired_closure(
                selected_ids
                    .iter()
                    .filter_map(|id| frame_by_id[id].provenance.source_span)
                    .collect(),
            );
            let Ok(retained_spans) = retained_compaction_spans_with_mode(
                snapshot,
                &selected_ids,
                &retired_spans,
                closure_mode,
            ) else {
                return false;
            };
            retired_spans.iter().all(|retired| {
                retained_spans
                    .iter()
                    .all(|retained| !spans_overlap(*retired, *retained))
            })
        })
        .unwrap_or(base_start);
    let retired_ids = frames[base_start..tail_start_index]
        .iter()
        .map(|frame| {
            frame
                .runtime_frame_id
                .expect("runtime protocol frames have ids")
        })
        .collect::<Vec<_>>();
    if retired_ids.is_empty() {
        return Err(NoProgressSelection::NoSafeBoundary.into());
    }
    let retired_spans = canonical_runtime_retired_closure(
        retired_ids
            .iter()
            .flat_map(|id| frame_by_id[id].provenance.source_span)
            .collect(),
    );
    let closure = crate::transcript::transcript_projection::classify_compaction_closure_with_mode(
        snapshot,
        &retired_spans,
        closure_mode,
    );
    let retired_set = retired_ids.iter().copied().collect::<BTreeSet<_>>();
    let retained_spans = retained_compaction_spans_with_mode(
        snapshot,
        &retired_set,
        &retired_spans,
        closure_mode,
    )?;
    if retired_spans.iter().any(|retired| {
        retained_spans
            .iter()
            .any(|retained| spans_overlap(*retired, *retained))
    }) {
        return Err(NoProgressSelection::NoSafeBoundary.into());
    }
    let first_protected_index = frames
        .iter()
        .position(|frame| {
            frame
                .runtime_frame_id
                .is_some_and(|id| protected_ids.contains(&id))
        })
        .unwrap_or(frames.len());
    let head_for_summary = items[base_start..tail_start_index].to_vec();
    // The cache apply retains protected/current IDs directly; they are not raw
    // summary payload and therefore are deliberately absent from this adapter.
    let tail_items = items[tail_start_index..first_protected_index].to_vec();
    if head_for_summary.is_empty() {
        return Err(NoProgressSelection::NoSafeBoundary.into());
    }
    Ok(CompactionSelection {
        previous_summary,
        head_for_summary,
        tail_items,
        tail_start_index,
        // Protocol prefix + source-covered dependents (single retire set).
        // Single retire set: protocol prefix + classified dependents.
        retired_frame_ids: {
            let mut ids = retired_set;
            ids.extend(closure.co_retired_frame_ids.iter().copied());
            ids.into_iter().collect()
        },
        // This is the canonical raw closure, including journal records between
        // selected protocol sources. Selection, preparation, persistence, and
        // replay all use this same closure authority.
        retired_source_spans: retired_spans,
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
    dependent_projection_ids_with_mode(
        snapshot,
        retired_spans,
        crate::transcript::transcript_projection::CompactionClosureMode::Standard,
    )
}

fn dependent_projection_ids_with_mode(
    snapshot: &RuntimeSnapshot,
    retired_spans: &[SourceSpan],
    mode: crate::transcript::transcript_projection::CompactionClosureMode,
) -> BTreeSet<RuntimeFrameId> {
    crate::transcript::transcript_projection::classify_compaction_closure_with_mode(
        snapshot,
        retired_spans,
        mode,
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
    retained_compaction_spans_with_mode(
        snapshot,
        protocol_retired_ids,
        retired_spans,
        crate::transcript::transcript_projection::CompactionClosureMode::Standard,
    )
}

pub(super) fn retained_compaction_spans_with_mode(
    snapshot: &RuntimeSnapshot,
    protocol_retired_ids: &BTreeSet<RuntimeFrameId>,
    retired_spans: &[SourceSpan],
    mode: crate::transcript::transcript_projection::CompactionClosureMode,
) -> Result<Vec<SourceSpan>> {
    snapshot.validate_references()?;
    let dependent_ids = dependent_projection_ids_with_mode(snapshot, retired_spans, mode);
    let co_retired = protocol_retired_ids
        .iter()
        .copied()
        .chain(dependent_ids.iter().copied())
        .collect::<BTreeSet<_>>();
    let mut spans = snapshot
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
        .collect::<Vec<_>>();
    let frame_spans = snapshot
        .frames
        .iter()
        .map(|frame| (frame.id, frame.provenance.source_span))
        .collect::<BTreeMap<_, _>>();
    for contributor in &snapshot.prompt_contributors {
        if !contributor.retains_raw_sources {
            continue;
        }
        if contributor_is_traceability_only(contributor) {
            continue;
        }
        let references_co_retire = !contributor.frame_ids.is_empty()
            && contributor
                .frame_ids
                .iter()
                .all(|id| co_retired.contains(id));
        if references_co_retire {
            continue;
        }
        if let Some(span) = contributor.provenance.source_span {
            spans.push(span);
        }
        for id in &contributor.frame_ids {
            // validate_references above guarantees exact resolution.
            if let Some(Some(span)) = frame_spans.get(id) {
                spans.push(*span);
            }
        }
    }
    Ok(spans)
}

fn contributor_is_traceability_only(
    contributor: &crate::runtime_context::PromptContributorPlaceholder,
) -> bool {
    contributor.provenance.source == RuntimeSource::SummaryArtifact
        || matches!(
            contributor.provenance.label.as_deref(),
            Some("summary") | Some("evidence")
        )
}

fn prune_tool_outputs_snapshot<C: Config>(
    agent: &Agent<C>,
    source: &RuntimeSnapshot,
    selection: &CompactionSelection,
) -> Result<RuntimeSnapshot> {
    if !agent.compaction_config.prune {
        return Ok(source.clone());
    }
    let protected_ids = selection
        .retired_frame_ids
        .iter()
        .copied()
        .collect::<BTreeSet<_>>();
    let call_names = tool_output_names_by_frame_id(source);
    let mut snapshot = source.clone();
    for frame in &mut snapshot.frames {
        if !protected_ids.contains(&frame.id) {
            continue;
        }
        let Some(ProtocolFrameItem::ToolOutput { output_json, .. }) = frame.protocol.as_mut()
        else {
            continue;
        };
        if output_json.chars().count() < COMPACTION_PRUNE_MIN_OUTPUT_CHARS
            || output_json.contains(COMPACTION_PRUNED_MARKER)
        {
            continue;
        }
        *output_json = build_pruned_tool_output_json(
            output_json,
            call_names.get(&frame.id).map(String::as_str),
        );
        frame.summary = Some(output_json.clone());
    }
    snapshot.validate_references()?;
    Ok(snapshot)
}

/// Check all authoritative runtime/cache projections before pruning can decide
/// that no output needs changing. This happens before cloning or mutating the
/// snapshot so a failed validation is atomic.
fn validate_compaction_runtime_state<C: Config>(agent: &Agent<C>) -> Result<()> {
    // Keep validation side-effect free. Healing missing spans is applied only to
    // the selection working copy (or committed prune path) so failed compaction
    // attempts remain atomic w.r.t. live protocol/runtime caches.
    super::validate_runtime_snapshot_correspondence(&agent.history, &agent.runtime_snapshot)?;
    super::validate_protocol_frame_correspondence(&agent.protocol_frames, &agent.runtime_snapshot)?;
    analyze_history_items(&agent.history, agent.turn.current_turn_start_index)?;
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

/// Compaction-only protection authority. Turn protection keeps active epochs
/// stable for request construction, while explicit and raw-source-retaining
/// contributor protection prevents source retirement.
///
/// Request-pressure paths add the current-turn budget prefix as explicit
/// protection on a working snapshot before calling selection (see
/// `protect_current_turn_for_pressure_selection`).
fn retirement_blocker_frame_ids(snapshot: &RuntimeSnapshot) -> BTreeSet<RuntimeFrameId> {
    snapshot
        .compaction
        .explicit_protected_frame_ids
        .iter()
        .copied()
        .chain(
            snapshot
                .prompt_contributors
                .iter()
                .filter(|contributor| contributor.retains_raw_sources)
                .flat_map(|contributor| contributor.frame_ids.iter().copied()),
        )
        .collect()
}

/// Under request pressure, the active-turn budget prefix must remain unretirable.
/// Without this, healed source spans would allow summarizing the oversized
/// current message away and falsely "recover" a protected overflow.
fn protect_current_turn_for_pressure_selection(
    snapshot: &mut RuntimeSnapshot,
    current_turn_start_index: Option<usize>,
) {
    let frames = snapshot.active_protocol_frames();
    let start = current_turn_start_index.unwrap_or(frames.len()).min(frames.len());
    if start >= frames.len() {
        return;
    }
    let mut protected = snapshot.compaction.explicit_protected_frame_ids.clone();
    protected.extend(
        frames[start..]
            .iter()
            .filter_map(|frame| frame.runtime_frame_id),
    );
    protected.sort();
    protected.dedup();
    snapshot.set_protected_frame_ids(protected);
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

fn trim_tail_to_valid_boundary(items: &[HistoryItem], mut tail_start: usize) -> usize {
    while let Some(HistoryItem::ToolOutput { call_id, .. }) = items.get(tail_start) {
        if let Some(tool_call_index) = items[..tail_start].iter().rposition(|item| {
            matches!(
                item,
                HistoryItem::AssistantToolCalls { calls, .. }
                    if calls.iter().any(|call| call.call_id == *call_id)
            )
        }) {
            return tool_call_index;
        }
        tail_start += 1;
    }
    tail_start
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
                "请根据以下内容更新已有锚定摘要。保留仍然正确且仍然重要的内容，删除已过时或被推翻的信息，并合并新的事实、约束、决策、路径、命令、错误与后续动作。\n\n已有锚定摘要：\n{}\n\n需要并入的新历史：\n{}",
                previous_summary, serialized_history
            )
        }
        None => format!(
            "请根据以下会话历史生成新的锚定摘要，供后续轮次继续工作。摘要必须覆盖目标、约束、当前进展、关键决策、下一步、关键上下文与相关文件。\n\n会话历史：\n{}",
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
    fn selection_retires_completed_active_turn_prefix_without_splitting_incomplete_suffix() {
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
        // Request construction protects the entire active turn. Selection uses
        // its local retirement-blocker view, so the completed prefix remains
        // eligible while the incomplete group stays atomic.
        snapshot
            .set_turn_protected_frame_ids(snapshot.frames.iter().map(|frame| frame.id).collect());

        let selection = select_runtime_compaction_segments(
            &snapshot,
            &CompactionConfig {
                tail_turns: 0,
                preserve_recent_tokens: Some(0),
                ..CompactionConfig::default()
            },
            0,
        )
        .expect("completed active-turn prefix is eligible");

        assert_eq!(selection.head_for_summary, history[..3]);
        assert_eq!(selection.retired_frame_ids.len(), 3);
        assert_eq!(selection.tail_start_index, 3);
        assert!(!selection.retired_frame_ids.contains(&snapshot.frames[3].id));
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
        snapshot
            .set_turn_protected_frame_ids(snapshot.frames.iter().map(|frame| frame.id).collect());
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
    fn standard_selection_still_blocks_fully_covered_retaining_context_material() {
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
            contributor_id: "retaining-context-standard".into(),
            kind: PromptContributorKind::ContextMaterial,
            label: Some("retaining".into()),
            provenance: RuntimeFrameProvenance::new(RuntimeSource::ContextView)
                .with_span(SourceSpan::new(1, 1).expect("span")),
            retains_raw_sources: true,
            frame_ids: vec![context_id],
            source_frame_ids: Vec::new(),
        });
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
        .expect_err("standard mode must keep fully covered retaining materials");
        assert!(is_nothing_to_compact_error(&error));
    }

    #[test]
    fn selection_co_retires_fully_covered_retaining_context_material() {
        // Regression: retaining materials used to join protected_frame_ids and
        // block co-retirement even when their whole source span was selected,
        // producing request-pressure failures:
        // `protected_context,retained_source_dependency`.
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
            snapshot
                .compaction
                .protected_frame_ids
                .contains(&context_id),
            "retaining material still participates in protected_frame_ids"
        );

        let selection = select_runtime_compaction_segments_with_mode(
            &snapshot,
            &CompactionConfig {
                tail_turns: 0,
                preserve_recent_tokens: Some(0),
                prune: true,
            },
            0,
            crate::transcript::transcript_projection::CompactionClosureMode::RequestPressure,
        )
        .expect("fully covered retaining material must co-retire with its source span");
        assert!(selection.retired_frame_ids.contains(&context_id));
        assert!(!selection.retired_frame_ids.is_empty());
    }

    #[test]
    fn selection_blocks_incomplete_coverage_of_wide_retaining_material_span() {
        let history = vec![
            HistoryItem::user("old user"),
            HistoryItem::assistant("old assistant"),
        ];
        let mut snapshot = snapshot_for(&history, None);
        let seed = RuntimeFrameIdSeed {
            frame_kind: RuntimeFrameKind::ContextBlock,
            source: RuntimeSource::ContextView,
            ordinal: 0,
            stable_key: "block-retaining-wide",
            source_span: Some(SourceSpan::new(1, 3).expect("span")),
        };
        let context_id = RuntimeFrameId::from_seed(&seed);
        snapshot.push_frame(
            RuntimeFrame::new(
                RuntimeFrameKind::ContextBlock,
                FrameVisibility::Active,
                RuntimeFrameProvenance::new(RuntimeSource::ContextView)
                    .with_span(SourceSpan::new(1, 3).expect("span")),
                seed,
            )
            .with_summary("retaining wide block"),
        );
        snapshot.push_prompt_contributor(PromptContributorPlaceholder {
            contributor_id: "retaining-context-wide".into(),
            kind: PromptContributorKind::ContextMaterial,
            label: Some("retaining".into()),
            provenance: RuntimeFrameProvenance::new(RuntimeSource::ContextView)
                .with_span(SourceSpan::new(1, 3).expect("span")),
            retains_raw_sources: true,
            frame_ids: vec![context_id],
            source_frame_ids: Vec::new(),
        });
        snapshot.recompute_protected_frame_ids();

        let error = select_runtime_compaction_segments_with_mode(
            &snapshot,
            &CompactionConfig {
                tail_turns: 0,
                preserve_recent_tokens: Some(0),
                prune: true,
            },
            0,
            crate::transcript::transcript_projection::CompactionClosureMode::RequestPressure,
        )
        .expect_err("retaining dependency must block incomplete coverage");
        assert!(is_nothing_to_compact_error(&error));
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
        protect_current_turn_for_pressure_selection(&mut snapshot, Some(2));

        let selection = select_runtime_compaction_segments_with_mode(
            &snapshot,
            &CompactionConfig {
                tail_turns: 0,
                preserve_recent_tokens: Some(0),
                prune: true,
            },
            0,
            crate::transcript::transcript_projection::CompactionClosureMode::RequestPressure,
        )
        .expect("pressure selection must retire pre-turn history with retaining materials");
        // Single retire set may include co-retired contributor materials under
        // RequestPressure; protocol prefix still retires and the current turn does not.
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
            .expect("the contributor only blocks its overlapping candidate");

        assert_eq!(selection.head_for_summary, history[..1]);
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
