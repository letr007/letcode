use super::*;
#[cfg(test)]
use crate::protocol_frames::analyze_history_items;
use futures_util::FutureExt;
use futures_util::future::BoxFuture;

type EventCallback<'a> = dyn FnMut(AgentEvent) -> BoxFuture<'a, Result<()>> + Send + 'a;

#[cfg(test)]
enum PressureAdmissionError {
    /// Compaction installed but the successor still cannot fit the hard budget.
    /// History reclaim is kept so the session does not regress to the oversized
    /// pre-compact state.
    BudgetExhausted { detail: String },
    /// Protocol/runtime inconsistency after install. The acknowledged compact
    /// history remains authoritative and is intentionally not rolled back.
    Technical(anyhow::Error),
}

pub(super) struct PreparedRequestBuild {
    pub(super) protected_start_index: usize,
    pub(super) build: crate::request_builder::BuildResult,
    /// Retained through physical retries and committed at the send boundary.
    pub(super) epoch_preview: super::ActiveEpochPreview,
}

#[cfg(test)]
struct PreparedCompaction {
    event: ContextCompactionEvent,
    local_state: Option<PreparedLocalCompaction>,
}

#[cfg(test)]
struct PreparedLocalCompaction {
    current_turn_start_index: Option<usize>,
    runtime_snapshot: crate::runtime_context::RuntimeSnapshot,
}

pub(super) async fn compact_session_stream_async<E, Efut, S>(
    agent: &mut Agent,
    mut on_event: E,
    mut on_start: S,
) -> Result<ManualCompactionOutcome>
where
    E: FnMut(AgentEvent) -> Efut + Send,
    Efut: Future<Output = Result<()>> + Send,
    S: FnMut() -> Result<()> + Send,
{
    let trigger = CompactionTrigger::Manual;
    on_event(AgentEvent::ContextCompactionStarted { trigger }).await?;
    if let Err(error) = on_start() {
        let _ = on_event(AgentEvent::ContextCompactionFailed { trigger }).await;
        return Err(error);
    }
    let mut on_event = |event| Box::pin(on_event(event)) as BoxFuture<'_, Result<()>>;
    let result = attempt_compaction(agent, trigger, &mut on_event).await;
    if result.is_err() {
        let _ = on_event(AgentEvent::ContextCompactionFailed { trigger }).await;
    }
    result
}

/// Pressure uses the same compaction transaction as an explicit `/compact`.
/// The caller consumes its ephemeral frontier before entering this fallible
/// operation; this function therefore never mutates live state before the
/// durable callback acknowledges the compaction.
pub(super) fn compact_for_request_pressure<'a, E, Efut>(
    agent: &'a mut Agent,
    protocol: ApiProtocol,
    turn_prelude: &'a [PromptMessage],
    tool_definitions: &'a [crate::request_builder::ToolSpec],
    on_event: &'a mut E,
) -> BoxFuture<'a, Result<PreparedRequestBuild>>
where
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

/// Shared select → summarize → prepare path for pressure and manual compact.
#[cfg(test)]
async fn prepare_compaction(
    agent: &mut Agent,
    trigger: CompactionTrigger,
    on_event: &mut EventCallback<'_>,
) -> Result<Result<PreparedCompaction, CompactionNoProgress>>
where
{
    if agent.runtime_snapshot_provider.is_some() {
        agent.reload_runtime_snapshot_from_provider()?;
    }
    // History structure is validated once below and reused by cut planning.
    // Reload already analyzed with turn_start=None for install safety; this
    // pass uses the live turn cursor so incomplete tool groups are respected.
    agent.runtime_snapshot.validate_references()?;
    let live_history = agent.active_history_items();
    let live_protocol_frames = agent.active_protocol_frames();

    if live_history.is_empty() {
        return Ok(Err(CompactionNoProgress {
            trigger,
            blockers: vec![CompactionBlocker::NoHistoricalItems],
        }));
    }
    let history_transcript =
        analyze_history_items(&live_history, agent.turn.current_turn_start_index)?;
    let preserve_recent_tokens = match trigger {
        CompactionTrigger::Manual => 0,
        CompactionTrigger::RequestPressure => agent
            .compaction_config
            .preserve_recent_tokens
            .unwrap_or_else(|| {
                default_preserve_recent_budget(super::effective_input_budget_tokens(
                    agent.active_model_metadata(),
                    &agent.tool_definitions(),
                ))
            }),
    };
    let Some(planned_cut) = super::history_compact::plan_turn_cut_with_transcript(
        &live_history,
        agent.turn.current_turn_start_index,
        preserve_recent_tokens,
        &history_transcript,
    )?
    else {
        return Ok(Err(CompactionNoProgress {
            trigger,
            blockers: vec![CompactionBlocker::NoSafeBoundary],
        }));
    };
    let Some(cut) = fit_compaction_cut_to_summary_budget(
        agent,
        &live_history,
        &history_transcript,
        &planned_cut,
    )?
    else {
        return Ok(Err(CompactionNoProgress {
            trigger,
            blockers: vec![CompactionBlocker::ProtectedContext],
        }));
    };

    // Every item retired by this event is present in this summarization request.
    let summary = crate::transcript::transcript_projection::sanitize_compaction_summary_body(
        &generate_context_summary(
            agent,
            trigger,
            cut.previous_summary.as_deref(),
            &cut.prefix,
            cut.split_active_turn,
            on_event,
        )
        .await?,
    )?;

    if agent.runtime_snapshot_provider.is_some() {
        let first_kept_entry_id = live_protocol_frames
            .get(cut.cut_end)
            .map(|frame| {
                frame
                    .source_provenance
                    .as_ref()
                    .and_then(|provenance| provenance.source_id.as_deref())
                    .filter(|source_id| !source_id.trim().is_empty())
                    .map(str::to_owned)
                    .ok_or_else(|| {
                        anyhow::anyhow!("compaction is missing a stable first kept entry id")
                    })
            })
            .transpose()?;
        return Ok(Ok(PreparedCompaction {
            event: ContextCompactionEvent::succeeded_at(summary, first_kept_entry_id),
            local_state: None,
        }));
    }
    // Direct Agent callers have no append-only projection to acknowledge.
    // Keep their local install path compatible with a legacy-shaped event;
    // TranscriptRecorder rejects this shape, so it cannot enter production.
    let event = ContextCompactionEvent::succeeded(summary.clone(), cut.cut_end);
    let history =
        super::history_compact::compose_with_summary(&summary, &live_history, cut.cut_end)?;
    let current_turn_start_index = agent
        .turn
        .current_turn_start_index
        .and_then(|start| (start >= cut.cut_end).then(|| 1 + start.saturating_sub(cut.cut_end)));
    let transcript =
        crate::protocol_frames::analyze_history_items(&history, current_turn_start_index)?;
    let protocol_frames = compacted_protocol_frames(agent, &transcript.frames, &cut)?;
    let mut runtime_snapshot = agent.rebuilt_runtime_snapshot_from_protocol_frames(
        &protocol_frames,
        live_protocol_frames.len(),
        &live_history,
    )?;
    merge_non_protocol_runtime_metadata(&mut runtime_snapshot, &agent.runtime_snapshot);
    rebind_active_protocol_from_history(&mut runtime_snapshot, &history)?;
    let protected_start = current_turn_start_index
        .unwrap_or(history.len())
        .min(history.len());
    let mut protected_frame_ids = runtime_snapshot.active_protocol_frames()[protected_start..]
        .iter()
        .filter_map(|frame| frame.runtime_frame_id)
        .collect::<Vec<_>>();
    protected_frame_ids.sort();
    protected_frame_ids.dedup();
    runtime_snapshot.set_turn_protected_frame_ids(protected_frame_ids);
    runtime_snapshot.heal_references()?;

    Ok(Ok(PreparedCompaction {
        event,
        local_state: Some(PreparedLocalCompaction {
            current_turn_start_index,
            runtime_snapshot: runtime_snapshot.clone(),
        }),
    }))
}

#[cfg(test)]
fn compacted_protocol_frames(
    agent: &Agent,
    candidate_frames: &[crate::protocol_frames::ProtocolFrame],
    cut: &super::history_compact::TurnCut,
) -> Result<Vec<crate::protocol_frames::ProtocolFrame>> {
    let live_frames = agent.active_protocol_frames();
    let live_history = agent.active_history_items();
    anyhow::ensure!(
        live_frames.len() == live_history.len(),
        "cannot compact protocol identity: cached frames {} vs history {}",
        live_frames.len(),
        live_history.len()
    );

    let mut frames = candidate_frames.to_vec();
    let mut candidate_index = 1usize; // The compacted summary always gets a new identity.
    for old_index in cut.cut_end..live_frames.len() {
        inherit_protocol_identity(frames.get_mut(candidate_index), live_frames.get(old_index))?;
        candidate_index += 1;
    }
    anyhow::ensure!(
        candidate_index == frames.len(),
        "compacted protocol mapping produced {} frames for candidate length {}",
        candidate_index,
        frames.len()
    );
    Ok(frames)
}

#[cfg(test)]
fn inherit_protocol_identity(
    candidate: Option<&mut crate::protocol_frames::ProtocolFrame>,
    retained: Option<&crate::protocol_frames::ProtocolFrame>,
) -> Result<()> {
    let candidate =
        candidate.ok_or_else(|| anyhow::anyhow!("missing compacted candidate frame"))?;
    let retained = retained.ok_or_else(|| anyhow::anyhow!("missing retained protocol frame"))?;
    candidate.runtime_frame_id = retained.runtime_frame_id;
    candidate.source_provenance = retained.source_provenance.clone();
    Ok(())
}

#[cfg(test)]
fn install_prepared_compaction(agent: &mut Agent, prepared: PreparedLocalCompaction) {
    agent.turn.current_turn_start_index = prepared.current_turn_start_index;
    agent.runtime_snapshot = prepared.runtime_snapshot;
    agent.clear_active_epoch();
    agent.clear_provider_usage_anchor();
}

#[cfg(test)]
async fn commit_prepared_compaction(
    agent: &mut Agent,
    prepared: PreparedCompaction,
    on_event: &mut EventCallback<'_>,
) -> Result<()>
where
{
    on_event(AgentEvent::ContextCompacted(prepared.event)).await?;
    if agent.runtime_snapshot_provider.is_some() {
        agent.reload_runtime_snapshot_from_provider()?;
    } else {
        let local_state = prepared
            .local_state
            .ok_or_else(|| anyhow::anyhow!("local compaction is missing its prepared state"))?;
        install_prepared_compaction(agent, local_state);
    }
    Ok(())
}

fn pressure_successor_request(
    agent: &mut Agent,
    protocol: ApiProtocol,
    turn_prelude: &[PromptMessage],
    tool_definitions: &[crate::request_builder::ToolSpec],
) -> Result<PreparedRequestBuild> {
    let epoch_preview =
        match agent.prepare_active_epoch(protocol, turn_prelude, tool_definitions)? {
            super::ActiveEpochPreparation::Warm(preview) => preview,
            super::ActiveEpochPreparation::ColdRequired(_) => {
                agent.preview_active_epoch(protocol, turn_prelude, tool_definitions)?
            }
        };
    Ok(PreparedRequestBuild {
        protected_start_index: agent
            .turn
            .current_turn_start_index
            .unwrap_or(agent.active_history_items().len()),
        build: epoch_preview.build.clone(),
        epoch_preview,
    })
}

pub(super) async fn compact_for_request_pressure_with_callback(
    agent: &mut Agent,
    protocol: ApiProtocol,
    turn_prelude: &[PromptMessage],
    tool_definitions: &[crate::request_builder::ToolSpec],
    on_event: &mut EventCallback<'_>,
) -> Result<PreparedRequestBuild>
where
{
    let trigger = CompactionTrigger::RequestPressure;
    on_event(AgentEvent::ContextCompactionStarted { trigger }).await?;
    if agent.runtime_snapshot_provider.is_some() {
        // Each round must fold the existing archive or consume a raw prefix.
        // Never let ordinary request truncation silently discard history slots.
        let rounds = agent.active_protocol_frames().len().saturating_add(1);
        for round in 0..rounds {
            if round > 0 {
                on_event(AgentEvent::ContextCompactionStarted { trigger }).await?;
            }
            let progressed =
                match super::history_runtime::advance(agent, true, false, on_event).await {
                    Ok(value) => value,
                    Err(error) => {
                        let _ = on_event(AgentEvent::ContextCompactionFailed { trigger }).await;
                        return Err(error);
                    }
                };
            let next =
                match pressure_successor_request(agent, protocol, turn_prelude, tool_definitions) {
                    Ok(next) => next,
                    Err(error) if progressed && is_recognized_request_budget_overflow(&error) => {
                        // A smaller Historian may cover only part of an oversized
                        // active turn. Preserve that commit and consume the next
                        // complete prefix rather than stopping after the first one.
                        continue;
                    }
                    Err(error) => {
                        let _ = on_event(AgentEvent::ContextCompactionFailed { trigger }).await;
                        return Err(error);
                    }
                };
            if !next.build.budget.truncated {
                return Ok(next);
            }
            agent.history_budget_limit = Some(
                next.build
                    .budget
                    .input_budget_tokens
                    .saturating_sub(next.build.budget.estimated_prelude_tokens)
                    .saturating_sub(next.build.budget.estimated_protected_tokens)
                    .saturating_sub(next.build.budget.estimated_evidence_tokens),
            );
            if !progressed {
                on_event(AgentEvent::ContextCompactionNoProgress(
                    CompactionNoProgress {
                        trigger,
                        blockers: vec![CompactionBlocker::NoSafeBoundary],
                    },
                ))
                .await?;
                anyhow::bail!(
                    "history cannot fit without dropping unprocessed context; use a larger input window or Historian route"
                );
            }
        }
        on_event(AgentEvent::ContextCompactionFailed { trigger }).await?;
        anyhow::bail!("history compaction exhausted its source-bounded progress rounds");
    }

    #[cfg(not(test))]
    return Err(anyhow::anyhow!(
        "history compaction requires a persisted session"
    ));

    #[cfg(test)]
    async {
        // Build one durable compaction event. Tail pruning is part of the
        // append-only projection, never an unjournaled live mutation.
        let prepared_result = match prepare_compaction(agent, trigger, on_event).await {
            Ok(result) => result,
            Err(error) => {
                let _ = on_event(AgentEvent::ContextCompactionFailed { trigger }).await;
                return Err(error);
            }
        };
        let prepared = match prepared_result {
            Ok(prepared) => prepared,
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
                    anyhow::anyhow!(
                        "request pressure has no compactable context: {labels}; admission still fails: {error}"
                    )
                });
            }
        };

        // Persist before any live install. A rejected journal record leaves the
        // working history untouched, including on later admission failures.
        if let Err(error) = commit_prepared_compaction(agent, prepared, on_event).await {
            let _ = on_event(AgentEvent::ContextCompactionFailed { trigger }).await;
            return Err(error);
        }
        let admission = match pressure_successor_request(
            agent,
            protocol,
            turn_prelude,
            tool_definitions,
        ) {
            Ok(successor) => Ok(successor),
            Err(error) if is_recognized_request_budget_overflow(&error) => {
                Err(PressureAdmissionError::BudgetExhausted {
                    detail: format!("request still over budget after compaction: {error}"),
                })
            }
            Err(error) => Err(PressureAdmissionError::Technical(error)),
        };

        match admission {
            Ok(successor) => Ok(successor),
            Err(PressureAdmissionError::BudgetExhausted { detail }) => {
                // The durable record now exactly matches the live reclaimed
                // history, even though the successor cannot fit.
                Err(anyhow::anyhow!(detail))
            }
            Err(PressureAdmissionError::Technical(error)) => {
                // The durable compact record is already acknowledged. Keep its
                // installed history rather than restoring an unrecorded state.
                Err(error)
            }
        }
    }
    .await
}

async fn attempt_compaction(
    agent: &mut Agent,
    trigger: CompactionTrigger,
    on_event: &mut EventCallback<'_>,
) -> Result<CompactionAttemptOutcome>
where
{
    async {
        if agent.runtime_snapshot_provider.is_some() {
            if super::history_runtime::advance(
                agent,
                true,
                trigger == CompactionTrigger::Manual,
                on_event,
            )
            .await?
            {
                return Ok(CompactionAttemptOutcome::Compacted {
                    retained_items: agent.active_history_items().len(),
                });
            }
            let no_progress = CompactionNoProgress {
                trigger,
                blockers: vec![CompactionBlocker::NoSafeBoundary],
            };
            on_event(AgentEvent::ContextCompactionNoProgress(no_progress.clone())).await?;
            return Ok(CompactionAttemptOutcome::NoProgress(no_progress));
        }
        #[cfg(not(test))]
        anyhow::bail!("history compaction requires a persisted session");
        #[cfg(test)]
        {
            let prepared = match prepare_compaction(agent, trigger, on_event).await? {
                Ok(prepared) => prepared,
                Err(no_progress) => {
                    on_event(AgentEvent::ContextCompactionNoProgress(no_progress.clone())).await?;
                    return Ok(CompactionAttemptOutcome::NoProgress(no_progress));
                }
            };

            // The durable callback is the commit point. Provider-backed sessions
            // then reload their canonical projection before reporting retention.
            commit_prepared_compaction(agent, prepared, on_event).await?;
            Ok(CompactionAttemptOutcome::Compacted {
                retained_items: agent.active_history_items().len(),
            })
        }
    }
    .await
}

#[cfg(test)]
fn diagnostic_labels(blockers: &[CompactionBlocker]) -> String {
    blockers
        .iter()
        .map(|blocker| blocker.label())
        .collect::<Vec<_>>()
        .join(",")
}

pub(super) async fn prepare_request_build<E, Efut>(
    agent: &mut Agent,
    protocol: ApiProtocol,
    turn_prelude: &[PromptMessage],
    protected_start_index: usize,
    tool_definitions: &[crate::request_builder::ToolSpec],
    on_event: &mut E,
) -> Result<PreparedRequestBuild>
where
    E: FnMut(AgentEvent) -> Efut,
    Efut: Future<Output = Result<()>>,
{
    if agent.prepare_fast_mode_for_request()? {
        on_event(AgentEvent::FastModeChanged { enabled: false }).await?;
    }
    let epoch_preview =
        match agent.prepare_active_epoch(protocol, turn_prelude, tool_definitions)? {
            super::ActiveEpochPreparation::Warm(preview) => preview,
            super::ActiveEpochPreparation::ColdRequired(_) => {
                agent.preview_active_epoch(protocol, turn_prelude, tool_definitions)?
            }
        };
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

#[cfg(test)]
fn fit_compaction_cut_to_summary_budget(
    agent: &Agent,
    live_history: &[HistoryItem],
    transcript: &crate::protocol_frames::ProtocolTranscript,
    planned: &super::history_compact::TurnCut,
) -> Result<Option<super::history_compact::TurnCut>> {
    let base_start = planned.cut_end.saturating_sub(planned.prefix.len());
    let boundaries = (base_start + 1..=planned.cut_end)
        .filter(|boundary| {
            crate::protocol_frames::is_canonical_compaction_boundary(transcript, *boundary)
        })
        .collect::<Vec<_>>();
    if boundaries.is_empty() {
        return Ok(None);
    }

    let (serialized_history, item_ends) = render_compaction_history_with_item_ends(&planned.prefix);
    let prelude = [PromptMessage::system(CONTEXT_COMPACTION_PRELUDE)];
    let mut low = 0usize;
    let mut high = boundaries.len();
    let mut best = None;
    let mut last_error = None;

    while low < high {
        let middle = low + (high - low) / 2;
        let cut_end = boundaries[middle];
        let item_count = cut_end - base_start;
        let serialized_prefix = &serialized_history[..item_ends[item_count - 1]];
        let split_active_turn = agent
            .turn
            .current_turn_start_index
            .is_some_and(|index| index < cut_end);
        let prompt = render_compaction_prompt_with_serialized_history(
            planned.previous_summary.as_deref(),
            serialized_prefix,
            split_active_turn,
        );
        let Some(route) = agent.resolved_model_route() else {
            return Err(anyhow::anyhow!(
                "context compaction preflight requires an installed resolved model route"
            ));
        };
        match super::protocol_stream::preflight_resolved_oneshot_text_request(
            route,
            agent.active_model_metadata(),
            &prelude,
            &prompt,
        ) {
            Ok(build) => {
                let classification = build.budget.request_classification();
                if build.budget.truncated || !classification.safe {
                    last_error = Some(anyhow::anyhow!(
                        "final prompt and tools exceed selected input budget"
                    ));
                    high = middle;
                    continue;
                }
                best = Some((cut_end, split_active_turn));
                low = middle + 1;
            }
            Err(error) if is_recognized_request_budget_overflow(&error) => {
                last_error = Some(error);
                high = middle;
            }
            Err(error) => return Err(error),
        }
    }

    let Some((cut_end, split_active_turn)) = best else {
        if let Some(error) = last_error {
            tracing::debug!(error = %error, "no compaction prefix fits the summary request budget");
        }
        return Ok(None);
    };
    Ok(Some(super::history_compact::TurnCut {
        cut_end,
        prefix: live_history[base_start..cut_end].to_vec(),
        previous_summary: planned.previous_summary.clone(),
        split_active_turn,
    }))
}

#[cfg(test)]
async fn generate_context_summary(
    agent: &Agent,
    trigger: CompactionTrigger,
    previous_summary: Option<&str>,
    head_for_summary: &[HistoryItem],
    split_active_turn: bool,
    on_event: &mut EventCallback<'_>,
) -> Result<String>
where
{
    let prompt = render_compaction_prompt_with_serialized_history(
        previous_summary,
        &render_compaction_history(head_for_summary),
        split_active_turn,
    );
    // Narrow oneshot stream: no nested Agent turn, no tools, reasoning forced off.
    // Preview deltas and retry resets share one queue so a retry reset is
    // observed after the failed attempt's deltas and before the next attempt.
    // Retry waits for the reset callback to succeed before sending another request.
    let (event_tx, mut event_rx) = tokio::sync::mpsc::unbounded_channel::<(
        AgentEvent,
        Option<tokio::sync::oneshot::Sender<()>>,
    )>();
    let delta_tx = event_tx.clone();
    let retry_tx = event_tx.clone();
    drop(event_tx);
    let prelude = [PromptMessage::system(CONTEXT_COMPACTION_PRELUDE)];
    let route = agent.resolved_model_route().ok_or_else(|| {
        anyhow::anyhow!("context compaction requires an installed resolved model route")
    })?;
    let summary = super::protocol_stream::stream_resolved_oneshot_text_async(
        route,
        agent.active_model_metadata(),
        &prelude,
        &prompt,
        None,
        move |delta| {
            std::future::ready(
                delta_tx
                    .send((
                        AgentEvent::ContextCompactionDelta {
                            delta: delta.to_string(),
                        },
                        None,
                    ))
                    .map_err(|_| anyhow::anyhow!("context compaction event receiver closed")),
            )
        },
        move || {
            let retry_tx = retry_tx.clone();
            async move {
                let (ack_tx, ack_rx) = tokio::sync::oneshot::channel();
                retry_tx
                    .send((
                        AgentEvent::ContextCompactionStarted { trigger },
                        Some(ack_tx),
                    ))
                    .map_err(|_| anyhow::anyhow!("context compaction event receiver closed"))?;
                ack_rx
                    .await
                    .map_err(|_| anyhow::anyhow!("context compaction retry acknowledgement closed"))
            }
        },
    )
    .boxed();
    tokio::pin!(summary);
    let summary = loop {
        tokio::select! {
            result = &mut summary => break result?,
            Some((event, ack)) = event_rx.recv() => {
                on_event(event).await?;
                if let Some(ack) = ack {
                    let _ = ack.send(());
                }
            }
        }
    };
    while let Ok((event, ack)) = event_rx.try_recv() {
        on_event(event).await?;
        if let Some(ack) = ack {
            let _ = ack.send(());
        }
    }
    let trimmed = summary.trim();
    if trimmed.is_empty() {
        bail!("context compaction produced an empty summary")
    }
    Ok(trimmed.to_string())
}

#[cfg(test)]
fn render_compaction_prompt_with_serialized_history(
    previous_summary: Option<&str>,
    serialized_history: &str,
    split_active_turn: bool,
) -> String {
    let split_turn_instruction = split_active_turn.then_some(
        "本次是活动回合的前缀压缩：提供的历史（包括用户消息）都会退休并纳入摘要；仅 cut point 之后的原始尾部会被保留。",
    );
    let common = split_turn_instruction
        .map(|instruction| format!("\n\n{instruction}"))
        .unwrap_or_default();
    match previous_summary {
        Some(previous_summary) => {
            // Preserve the previous checkpoint intact. The canonical one-shot
            // request builder remains the input-budget authority and fails before
            // transport if the complete compaction prompt cannot fit.
            format!(
                "请根据以下内容更新已有执行检查点。保留仍正确的重要事实，删除过时或被推翻的信息，并合并新的事实、约束、决定、进度、验证、文件操作与精确下一步。输出必须仍遵循 prelude 的 Markdown section 结构。\n\n已有执行检查点：\n{}\n\n需要并入的新历史：\n{}{}",
                previous_summary, serialized_history, common
            )
        }
        None => format!(
            "请根据以下会话历史生成新的执行检查点，供后续轮次继续工作。使用 prelude 规定的 Markdown section 结构，覆盖 Progress、Key Decisions、Validation、File Operations、Next Steps 与 Critical Context。\n\n会话历史：\n{}{}",
            serialized_history, common
        ),
    }
}

// Pi keeps enough capacity to build and answer the checkpoint request before
// selecting its raw-message tail.  Keep the reserve explicit so effective input
// limits, rather than nominal context windows, bound the 20k target.
const COMPACTION_PREPARATION_RESERVE_TOKENS: u64 = 16_384;
const MIN_COMPACTION_PREPARATION_RESERVE_TOKENS: u64 = 2_048;

pub(super) fn default_preserve_recent_budget(input_budget: u64) -> u64 {
    // `input_budget` already excludes the model output and request safety
    // reserves. Keep a separate preparation reserve so a 20k raw tail is only
    // selected when the compaction prompt and its checkpoint reply can fit.
    input_budget
        .saturating_sub(
            COMPACTION_PREPARATION_RESERVE_TOKENS
                .min(input_budget.saturating_sub(MIN_COMPACTION_PREPARATION_RESERVE_TOKENS)),
        )
        .min(20_000)
}

#[cfg(test)]
pub(super) fn describe_history_item(item: &HistoryItem) -> String {
    match item {
        HistoryItem::ContextSummary { text } => format!("摘要: {text}"),
        HistoryItem::UserMessage { content } => format!("用户: {}", content.display_text()),
        HistoryItem::InternalContinuation { text } => format!("继续执行指令: {text}"),
        HistoryItem::AssistantTurn {
            text,
            reasoning_content,
            calls,
            ..
        } if calls.is_empty() => {
            let text = text.as_deref().unwrap_or_default();
            match reasoning_content {
                Some(reasoning) if !reasoning.is_empty() => {
                    format!("助手推理: {reasoning}\n助手: {text}")
                }
                _ => format!("助手: {text}"),
            }
        }
        HistoryItem::AssistantTurn { text, calls, .. } => format!(
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
            ..
        } => format!(
            "工具输出 {call_id}: {}",
            render_tool_output_for_compaction(output_json)
        ),
    }
}

#[cfg(test)]
pub(super) fn render_compaction_history(items: &[HistoryItem]) -> String {
    render_compaction_history_with_item_ends(items).0
}

#[cfg(test)]
fn render_compaction_history_with_item_ends(items: &[HistoryItem]) -> (String, Vec<usize>) {
    let mut rendered = String::new();
    let mut item_ends = Vec::with_capacity(items.len());
    for (index, item) in items.iter().enumerate() {
        if index > 0 {
            rendered.push('\n');
        }
        rendered.push_str(&format!("{}. {}", index + 1, describe_history_item(item)));
        item_ends.push(rendered.len());
    }
    (rendered, item_ends)
}

/// Cap on the raw JSON bytes we're willing to parse while summarizing a single
/// tool output. Outputs (read / grep / bash) can be many MB, but the compaction
/// prompt only ever surfaces the leading part — everything past this prefix is
/// guaranteed to be dropped by `COMPACTION_TOOL_OUTPUT_CHAR_CAP` anyway. Parse a
/// bounded prefix instead of the whole payload, matching how Pi / OpenCode trim
/// tool results before serializing them for the summarization request.
#[cfg(test)]
const TOOL_OUTPUT_PARSE_PREFIX_CAP: usize = 8 * 1024;

/// Take up to `max_chars` characters without scanning past the cap
/// (O(max_chars), not O(len)); truncation always lands on a char boundary.
#[cfg(test)]
fn bounded_take_chars(text: &str, max_chars: usize) -> &str {
    match text.char_indices().nth(max_chars) {
        Some((byte_idx, _)) => &text[..byte_idx],
        None => text,
    }
}

#[cfg(test)]
fn render_tool_output_for_compaction(output_json: &str) -> String {
    // 有界解析：只解析前 TOOL_OUTPUT_PARSE_PREFIX_CAP 字符，之后的内容必会被
    // COMPACTION_TOOL_OUTPUT_CHAR_CAP 截掉，无需为整个大 JSON 付出 O(原始体积) 的
    // 解析/序列化成本（对齐 Pi/OpenCode 先裁剪再处理的模式）。
    let bounded = bounded_take_chars(output_json, TOOL_OUTPUT_PARSE_PREFIX_CAP);
    let rendered = serde_json::from_str::<Value>(bounded)
        .ok()
        .map(sanitize_tool_output_value_for_compaction)
        .and_then(|value| serde_json::to_string(&value).ok())
        .unwrap_or_else(|| bounded.to_string());
    truncate_for_compaction(
        &rendered,
        COMPACTION_TOOL_OUTPUT_CHAR_CAP,
        COMPACTION_TOOL_OUTPUT_TRUNCATION_MARKER,
    )
}

#[cfg(test)]
fn sanitize_tool_output_value_for_compaction(mut value: Value) -> Value {
    strip_obvious_media_fields(&mut value, "$", false);
    value
}

#[cfg(test)]
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

#[cfg(test)]
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

#[cfg(test)]
fn value_looks_blob_like(value: &Value) -> bool {
    let Some(text) = value.as_str() else {
        return false;
    };
    let trimmed = text.trim();
    trimmed.starts_with("data:")
        || trimmed.starts_with("blob:")
        || (trimmed.chars().count() >= 512 && looks_base64_like(trimmed))
}

#[cfg(test)]
fn looks_base64_like(text: &str) -> bool {
    let compact = text.trim();
    !compact.is_empty()
        && compact
            .chars()
            .all(|ch| ch.is_ascii_alphanumeric() || matches!(ch, '+' | '/' | '=' | '_' | '-'))
}

#[cfg(test)]
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

#[cfg(test)]
#[path = "../../tests/internal/agent/compaction/transaction_tests.rs"]
mod transaction_tests;

#[cfg(test)]
#[path = "../../tests/internal/agent/compaction/integration_tests.rs"]
mod integration_tests;
