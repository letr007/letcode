use super::*;
use crate::protocol_frames::{ProtocolFrameItem, ToolCallGroupStatus, analyze_history_items};
use crate::runtime_context::{
    FrameVisibility, RuntimeFrameId, RuntimeSnapshot, RuntimeSource, SourceSpan,
};
use std::collections::{BTreeMap, BTreeSet};
use tokio::sync::mpsc;

#[derive(Debug, Clone)]
pub(super) struct CompactionSelection {
    pub(super) previous_summary: Option<String>,
    pub(super) head_for_summary: Vec<HistoryItem>,
    pub(super) tail_items: Vec<HistoryItem>,
    pub(super) tail_start_index: usize,
    pub(super) retired_frame_ids: Vec<RuntimeFrameId>,
    /// Raw context/folded projections derived wholly from retired protocol
    /// sources. They retire with their source instead of blocking its prefix.
    pub(super) dependent_frame_ids: Vec<RuntimeFrameId>,
    pub(super) retired_source_spans: Vec<SourceSpan>,
}

pub(super) struct PreparedRequestBuild {
    pub(super) protected_start_index: usize,
    pub(super) build: crate::request_builder::BuildResult,
}

struct PreparedCompaction {
    retained_items: usize,
    event: ContextCompactionEvent,
    snapshot: RuntimeSnapshot,
    protocol_frames: Vec<crate::protocol_frames::ProtocolFrame>,
    history: Vec<HistoryItem>,
}

pub(super) async fn compact_session_stream_async<C, E, Efut, S, D>(
    agent: &mut Agent<C>,
    mut on_event: E,
    mut on_start: S,
    mut on_delta: D,
) -> Result<ManualCompactionOutcome>
where
    C: Config + Clone,
    E: FnMut(AgentEvent) -> Efut,
    Efut: Future<Output = Result<()>>,
    S: FnMut() -> Result<()> + Send,
    D: FnMut(&str) -> Result<()> + Send,
{
    agent.refresh_runtime_snapshot_from_provider()?;
    validate_compaction_runtime_state(agent)?;
    let input_budget = effective_input_budget_tokens(agent.active_model_metadata(), &[]);
    let preserve_recent_budget = default_preserve_recent_budget(input_budget);
    let selection = match select_runtime_compaction_segments(
        &agent.runtime_snapshot,
        &agent.compaction_config,
        preserve_recent_budget,
    ) {
        Ok(selection) => selection,
        Err(error) if is_nothing_to_compact_error(&error) => {
            return Ok(ManualCompactionOutcome::NothingToCompact);
        }
        Err(error) => return Err(error),
    };
    on_start()?;
    let prepared = match compact_selected_context(agent, selection, Some(&mut on_delta)).await {
        Ok(result) => result,
        Err(error) => {
            emit_compaction_terminal_issue(&mut on_event, &error, false).await?;
            return Err(error);
        }
    };
    on_event(AgentEvent::ContextCompacted(prepared.event.clone())).await?;
    agent.commit_prepared_runtime_compaction(
        prepared.snapshot,
        prepared.protocol_frames,
        prepared.history,
    );
    Ok(ManualCompactionOutcome::Compacted {
        retained_items: prepared.retained_items,
    })
}

pub(super) async fn prepare_request_build<C, E, Efut>(
    agent: &mut Agent<C>,
    protocol: ApiProtocol,
    turn_prelude: &[PromptMessage],
    protected_start_index: usize,
    tool_definitions: &[crate::request_builder::ToolSpec],
    on_event: &mut E,
) -> Result<PreparedRequestBuild>
where
    C: Config + Clone,
    E: FnMut(AgentEvent) -> Efut,
    Efut: Future<Output = Result<()>>,
{
    // A provider projection can carry newly available folded-output metadata for
    // already-recorded tool output. Refresh before the first plan so the soft
    // reserve can fold it proactively; the overflow retry below remains a
    // safety net for metadata that arrives during request preparation.
    agent.refresh_runtime_snapshot_from_provider()?;
    let mut protected_start_index = protected_start_index;
    let compaction_enabled = agent.compaction_config.auto || agent.needs_compaction;

    let frozen_evidence = agent.turn.frozen_evidence.as_ref().map(|evidence| {
        crate::request_builder::FrozenEvidence {
            message: evidence.message.clone(),
            selected_ids: evidence.selected_ids.clone(),
        }
    });
    let policy = ProtectedContextPolicy::from_configured_reserve(
        agent.compaction_config.protected_reserve_tokens,
        effective_input_budget_tokens(agent.active_model_metadata(), tool_definitions),
    );
    let initial_build = match crate::request_builder::build_request_with_policy(
        RequestBuilderInput {
            protocol,
            model_id: &agent.model,
            model: agent.active_model_metadata(),
            prelude: turn_prelude,
            snapshot: &agent.runtime_snapshot,
            tools: tool_definitions,
        },
        frozen_evidence.as_ref(),
        Some(policy),
    ) {
        Ok(build) => build,
        Err(error) if is_protected_current_context_overflow(&error) => {
            // A just-finished ordinary tool call can have reached history before
            // its folded-output metadata reaches the runtime projection.
            agent.refresh_runtime_snapshot_from_provider()?;
            crate::request_builder::build_request_with_policy(
                RequestBuilderInput {
                    protocol,
                    model_id: &agent.model,
                    model: agent.active_model_metadata(),
                    prelude: turn_prelude,
                    snapshot: &agent.runtime_snapshot,
                    tools: tool_definitions,
                },
                frozen_evidence.as_ref(),
                Some(policy),
            )?
        }
        Err(error) => return Err(error),
    };

    if !compaction_enabled
        || protected_start_index == 0
        || !should_compact_for_build(agent, &initial_build.budget)
    {
        return Ok(PreparedRequestBuild {
            protected_start_index,
            build: initial_build,
        });
    }

    let preserve_recent_budget =
        default_preserve_recent_budget(initial_build.budget.input_budget_tokens);
    protected_start_index = match compact_context(agent, preserve_recent_budget, on_event).await {
        Ok(retained) => retained,
        Err(error) if is_nothing_to_compact_error(&error) => {
            return Ok(PreparedRequestBuild {
                protected_start_index,
                build: initial_build,
            });
        }
        Err(error) => return Err(error),
    };

    let final_build = crate::request_builder::build_request_with_policy(
        RequestBuilderInput {
            protocol,
            model_id: &agent.model,
            model: agent.active_model_metadata(),
            prelude: turn_prelude,
            snapshot: &agent.runtime_snapshot,
            tools: tool_definitions,
        },
        frozen_evidence.as_ref(),
        Some(policy),
    )?;

    Ok(PreparedRequestBuild {
        protected_start_index,
        build: final_build,
    })
}

pub(super) fn is_protected_current_context_overflow(error: &anyhow::Error) -> bool {
    error.chain().any(|cause| {
        cause
            .to_string()
            .starts_with("protected current context exceeds input budget:")
    })
}

pub(super) async fn preflight_compact_context<C, E, Efut>(
    agent: &mut Agent<C>,
    turn_prelude: &[PromptMessage],
    protected_start_index: usize,
    tool_definitions: &[crate::request_builder::ToolSpec],
    on_event: &mut E,
) -> Result<usize>
where
    C: Config + Clone,
    E: FnMut(AgentEvent) -> Efut,
    Efut: Future<Output = Result<()>>,
{
    prepare_request_build(
        agent,
        agent.active_protocol(),
        turn_prelude,
        protected_start_index,
        tool_definitions,
        on_event,
    )
    .await
    .map(|prepared| prepared.protected_start_index)
}

pub(super) fn should_compact_for_build<C: Config>(
    agent: &Agent<C>,
    budget: &crate::request_builder::BudgetReport,
) -> bool {
    agent.needs_compaction
        || budget.truncated
        || budget.estimated_request_tokens
            >= budget
                .input_budget_tokens
                .saturating_sub(compaction_reserved_tokens(
                    agent,
                    budget.input_budget_tokens,
                ))
}

fn compaction_reserved_tokens<C: Config>(agent: &Agent<C>, input_budget_tokens: u64) -> u64 {
    agent
        .compaction_config
        .reserved
        .unwrap_or_else(|| input_budget_tokens.saturating_div(10).clamp(256, 2_048))
}

pub(super) fn prune_old_tool_outputs<C: Config>(
    agent: &mut Agent<C>,
    preserve_recent_budget: u64,
) -> Result<()> {
    validate_compaction_runtime_state(agent)?;
    if !agent.compaction_config.prune {
        return Ok(());
    }

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

        *output_json = build_pruned_tool_output_json(
            output_json,
            call_names.get(&frame.id).map(String::as_str),
        );
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

pub(super) async fn compact_context<C, E, Efut>(
    agent: &mut Agent<C>,
    preserve_recent_budget: u64,
    on_event: &mut E,
) -> Result<usize>
where
    C: Config + Clone,
    E: FnMut(AgentEvent) -> Efut,
    Efut: Future<Output = Result<()>>,
{
    if agent.runtime_snapshot.active_protocol_frames().is_empty() {
        bail!("context compaction cannot summarize the protected current turn");
    }

    agent.refresh_runtime_snapshot_from_provider()?;
    validate_compaction_runtime_state(agent)?;
    let selection = select_runtime_compaction_segments(
        &agent.runtime_snapshot,
        &agent.compaction_config,
        preserve_recent_budget,
    )?;
    if selection.head_for_summary.is_empty() {
        bail!("context compaction could not select any historical items to summarize");
    }

    on_event(AgentEvent::ContextCompactionStarted).await?;

    let (delta_tx, mut delta_rx) = mpsc::unbounded_channel::<String>();
    let mut emit_delta = |delta: &str| {
        delta_tx
            .send(delta.to_string())
            .map_err(|_| anyhow!("context compaction delta receiver dropped"))?;
        Ok(())
    };
    let mut compaction = Box::pin(compact_selected_context(
        agent,
        selection,
        Some(&mut emit_delta),
    ));

    let result = loop {
        tokio::select! {
            result = &mut compaction => break result,
            maybe_delta = delta_rx.recv() => {
                match maybe_delta {
                    Some(delta) => {
                        on_event(AgentEvent::ContextCompactionDelta { delta }).await?;
                    }
                    None => continue,
                }
            }
        }
    };
    drop(compaction);

    while let Ok(delta) = delta_rx.try_recv() {
        on_event(AgentEvent::ContextCompactionDelta { delta }).await?;
    }

    let prepared = match result {
        Ok(result) => result,
        Err(error) => {
            emit_compaction_terminal_issue(on_event, &error, true).await?;
            return Err(error);
        }
    };
    on_event(AgentEvent::ContextCompacted(prepared.event.clone())).await?;
    agent.commit_prepared_runtime_compaction(
        prepared.snapshot,
        prepared.protocol_frames,
        prepared.history,
    );
    Ok(prepared.retained_items)
}

async fn compact_selected_context<C>(
    agent: &mut Agent<C>,
    selection: CompactionSelection,
    on_delta: Option<&mut (dyn FnMut(&str) -> Result<()> + Send + '_)>,
) -> Result<PreparedCompaction>
where
    C: Config + Clone,
{
    let original_history_items = agent.history.len();
    let summary = generate_context_summary(
        agent,
        selection.previous_summary.as_deref(),
        &selection.head_for_summary,
        on_delta,
    )
    .await?;

    // Pruning belongs to this candidate transaction.  Never prune the live
    // snapshot before the durable compaction record acknowledges it.
    let candidate = prune_tool_outputs_snapshot(agent, &agent.runtime_snapshot, &selection)?;
    let snapshot =
        agent.prepare_runtime_compaction_from_snapshot(&candidate, &selection, summary.clone())?;
    let protocol_frames = snapshot.active_protocol_frames();
    let history = crate::protocol_frames::history_items_from_frames(&protocol_frames);
    crate::protocol_frames::analyze_history_items(&history, agent.turn.current_turn_start_index)?;

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
        frame_identity_bindings:
            crate::transcript::transcript_projection::compaction_frame_identity_bindings(&snapshot),
        detail: None,
    };
    Ok(PreparedCompaction {
        retained_items: 1 + selection.tail_items.len(),
        event,
        snapshot,
        protocol_frames,
        history,
    })
}

async fn emit_compaction_terminal_issue<E, Efut>(
    on_event: &mut E,
    error: &anyhow::Error,
    continue_after_failure: bool,
) -> Result<()>
where
    E: FnMut(AgentEvent) -> Efut,
    Efut: Future<Output = Result<()>>,
{
    let detail = format!("{error:#}");
    let cancelled = is_compaction_cancelled_error(error);
    let message = if cancelled {
        "Context compaction cancelled"
    } else {
        "Context compaction failed"
    };
    let action = if continue_after_failure {
        "Continuing without compaction"
    } else {
        "Compaction did not complete"
    };
    on_event(AgentEvent::ModelStreamIssue {
        message: message.to_string(),
        detail: Some(detail),
        action: action.to_string(),
    })
    .await
}

fn is_compaction_cancelled_error(error: &anyhow::Error) -> bool {
    let message = error.to_string().to_ascii_lowercase();
    message.contains("cancelled") || message.contains("canceled")
}

async fn generate_context_summary<C: Config + Clone>(
    agent: &Agent<C>,
    previous_summary: Option<&str>,
    head_for_summary: &[HistoryItem],
    mut on_delta: Option<&mut (dyn FnMut(&str) -> Result<()> + Send + '_)>,
) -> Result<String> {
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
            auto: false,
            ..CompactionConfig::default()
        },
        automatic_checkpoint_policy: super::automatic_checkpoint::AutoCheckpointPolicy::from_config(
            LogicalCheckpointConfig::default(),
        ),
        retry_config: agent.retry_config.clone(),
        tool_timeout_secs: agent.tool_timeout_secs,
        needs_compaction: false,
        turn: TurnRuntimeState::default(),
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
    };
    let prompt = render_compaction_prompt(
        previous_summary,
        head_for_summary,
        compaction_history_char_budget(agent.active_model_metadata()),
    );
    let summary = Box::pin(summary_agent.run_stream_async(
        &prompt,
        |delta| {
            let result = if let Some(on_delta) = on_delta.as_deref_mut() {
                on_delta(delta)
            } else {
                Ok(())
            };
            std::future::ready(result)
        },
        |_| std::future::ready(Ok(())),
        |_| std::future::ready(Ok(PermissionApproval::Deny)),
    ))
    .await?;
    let trimmed = summary.trim();
    if trimmed.is_empty() {
        bail!("context compaction produced an empty summary")
    }
    Ok(trimmed.to_string())
}

pub(super) fn select_runtime_compaction_segments(
    snapshot: &RuntimeSnapshot,
    config: &CompactionConfig,
    preserve_recent_budget: u64,
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
        bail!(NO_HISTORICAL_ITEMS_FOR_COMPACTION);
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
    // `protected_frame_ids` is the normalized protection authority. It already
    // includes contributor `frame_ids`; source/traceability associations are
    // intentionally excluded so ordinary context-view blocks can co-retire.
    let protected_ids = snapshot
        .compaction
        .protected_frame_ids
        .iter()
        .copied()
        .collect::<BTreeSet<_>>();
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
            let Ok(retained_spans) =
                retained_compaction_spans(snapshot, &selected_ids, &retired_spans)
            else {
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
        bail!(NO_OLDER_ITEMS_AFTER_TAIL);
    }
    let retired_spans = canonical_runtime_retired_closure(
        retired_ids
            .iter()
            .flat_map(|id| frame_by_id[id].provenance.source_span)
            .collect(),
    );
    let dependent_ids = dependent_projection_ids(snapshot, &retired_spans);
    let retired_set = retired_ids.iter().copied().collect::<BTreeSet<_>>();
    let retained_spans = retained_compaction_spans(snapshot, &retired_set, &retired_spans)?;
    if retired_spans.iter().any(|retired| {
        retained_spans
            .iter()
            .any(|retained| spans_overlap(*retired, *retained))
    }) {
        bail!(NO_OLDER_ITEMS_AFTER_TAIL);
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
        bail!(NO_OLDER_ITEMS_AFTER_TAIL);
    }
    Ok(CompactionSelection {
        previous_summary,
        head_for_summary,
        tail_items,
        tail_start_index,
        retired_frame_ids: retired_ids,
        dependent_frame_ids: dependent_ids.into_iter().collect(),
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
    snapshot
        .frames
        .iter()
        .filter_map(|frame| {
            (matches!(
                frame.visibility,
                FrameVisibility::Active | FrameVisibility::Folded
            ) && !snapshot.compaction.protected_frame_ids.contains(&frame.id)
                && (matches!(
                    frame.provenance.source,
                    RuntimeSource::ContextView
                        | RuntimeSource::FoldedOutput
                        | RuntimeSource::Derived
                        | RuntimeSource::PromptContributor
                ) || (frame.provenance.source == RuntimeSource::Transcript
                    && frame.protocol.is_none()
                    && frame.kind != crate::runtime_context::RuntimeFrameKind::Metadata))
                && frame
                    .provenance
                    .source_span
                    .is_some_and(|span| span.covered_by_any(retired_spans)))
            .then_some(frame.id)
        })
        .collect()
}

fn canonical_runtime_retired_closure(spans: Vec<SourceSpan>) -> Vec<SourceSpan> {
    crate::transcript_projection::canonical_retired_source_spans(
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
        if contributor_is_traceability_only(contributor) {
            continue;
        }
        let references_co_retire = !contributor.frame_ids.is_empty()
            && contributor
                .frame_ids
                .iter()
                .all(|id| co_retired.contains(id));
        let dependent_span = !contributor.frame_ids.is_empty()
            && matches!(
                contributor.kind,
                crate::runtime_context::PromptContributorKind::RuntimeContext
                    | crate::runtime_context::PromptContributorKind::ContextMaterial
                    | crate::runtime_context::PromptContributorKind::ContextIndex
                    | crate::runtime_context::PromptContributorKind::FoldedOutputSummary
            )
            && contributor
                .provenance
                .source_span
                .is_some_and(|span| span.covered_by_any(retired_spans));
        if references_co_retire || dependent_span {
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
    super::validate_runtime_snapshot_correspondence(&agent.history, &agent.runtime_snapshot)?;
    super::validate_protocol_frame_correspondence(&agent.protocol_frames, &agent.runtime_snapshot)?;
    analyze_history_items(&agent.history, agent.turn.current_turn_start_index)?;
    Ok(())
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
    let message = error.to_string();
    message == NO_HISTORICAL_ITEMS_FOR_COMPACTION || message == NO_OLDER_ITEMS_AFTER_TAIL
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
    fn protected_context_overflow_matches_wrapped_budget_error_only() {
        let wrapped = anyhow::anyhow!(
            "protected current context exceeds input budget: protected/current context tokens (9) exceed budget (1)"
        )
        .context("refresh request state");
        assert!(is_protected_current_context_overflow(&wrapped));

        let unrelated = anyhow::anyhow!("protected current context exceeds input budgetary review");
        assert!(!is_protected_current_context_overflow(&unrelated));
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
            frame_ids: vec![folded_id],
            source_frame_ids: Vec::new(),
        });

        let selection = select_runtime_compaction_segments(&snapshot, &compaction_config(), 0)
            .expect("folded projection is dependent on the retired source");
        assert!(selection.dependent_frame_ids.contains(&folded_id));

        snapshot.frames.last_mut().unwrap().provenance.source_span =
            Some(SourceSpan::new(99, 99).unwrap());
        assert!(select_runtime_compaction_segments(&snapshot, &compaction_config(), 0).is_ok());
    }

    #[test]
    fn ordinary_visible_context_frame_does_not_block_and_co_retires_with_source() {
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
            frame_ids: Vec::new(),
            source_frame_ids: Vec::new(),
        });
        snapshot.set_protected_frame_ids(vec![snapshot.frames[2].id]);

        let selection = select_runtime_compaction_segments(&snapshot, &compaction_config(), 0)
            .expect("ordinary visible context must not retain historic source");

        assert_eq!(selection.head_for_summary, history[..2]);
        assert!(selection.dependent_frame_ids.contains(&context_id));
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
            frame_ids: Vec::new(),
            source_frame_ids: Vec::new(),
        });
        snapshot.set_protected_frame_ids(vec![snapshot.frames[2].id]);

        let selection = select_runtime_compaction_segments(&snapshot, &compaction_config(), 0)
            .expect("the contributor only blocks its overlapping candidate");

        assert_eq!(selection.head_for_summary, history[..1]);
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
    fn pruned_tool_output_uses_explicit_structural_marker() {
        let pruned = build_pruned_tool_output_json(&"x".repeat(10_000), Some("skill"));
        let value: Value = serde_json::from_str(&pruned).expect("pruned output is JSON");

        assert_eq!(value["_compaction"]["pruned"], Value::Bool(true));
        assert_eq!(value["_compaction"]["tool"], Value::String("skill".into()));
        assert!(value.get("data").is_none());
    }
}
