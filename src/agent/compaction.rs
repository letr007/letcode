use super::*;
use crate::protocol_frames::analyze_history_items;
use futures_util::FutureExt;
use futures_util::future::BoxFuture;
use std::collections::{BTreeMap, BTreeSet};

type EventCallback<'a> = dyn FnMut(AgentEvent) -> BoxFuture<'a, Result<()>> + Send + 'a;

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

struct PreparedCompaction {
    retained_items: usize,
    event: ContextCompactionEvent,
    history: Vec<HistoryItem>,
    current_turn_start_index: Option<usize>,
    protocol_frames: Vec<crate::protocol_frames::ProtocolFrame>,
    runtime_snapshot: crate::runtime_context::RuntimeSnapshot,
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
    let result = attempt_compaction(agent, trigger, &mut on_event, Some(&mut on_delta)).await;
    if result.is_err() {
        let _ = on_event(AgentEvent::ContextCompactionFailed { trigger }).await;
    }
    result
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

/// Shared select → summarize → prepare path for pressure and manual compact.
/// On success the agent still holds only the healed working snapshot; callers
/// decide when to install the prepared candidate and how to roll back.
async fn prepare_compaction_candidate<C>(
    agent: &mut Agent<C>,
    trigger: CompactionTrigger,
    on_event: &mut EventCallback<'_>,
    on_delta: Option<&mut (dyn FnMut(&str) -> Result<()> + Send + '_)>,
) -> Result<Result<PreparedCompaction, CompactionNoProgress>>
where
    C: Config + Clone,
{
    agent.refresh_runtime_snapshot_from_provider()?;
    validate_compaction_runtime_state(agent)?;

    if agent.history.is_empty() {
        return Ok(Err(CompactionNoProgress {
            trigger,
            blockers: vec![CompactionBlocker::NoHistoricalItems],
        }));
    }
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
    let Some(cut) = super::history_compact::plan_turn_cut(
        &agent.history,
        agent.turn.current_turn_start_index,
        preserve_recent_tokens,
    )?
    else {
        return Ok(Err(CompactionNoProgress {
            trigger,
            blockers: vec![CompactionBlocker::NoSafeBoundary],
        }));
    };

    // The compacted prefix is summarized independently from the retained tail.
    // In a split turn this is the explicit pi-style handoff boundary: the active
    // user message and its recent suffix are never supplied as retired history.
    let generated_summary =
        crate::transcript::transcript_projection::sanitize_compaction_summary_body(
            &generate_context_summary(
                agent,
                cut.previous_summary.as_deref(),
                &cut.prefix,
                cut.split_active_turn,
                on_event,
                on_delta,
            )
            .await?,
        );
    validate_checkpoint_sections(&generated_summary)?;
    let file_operations = cumulative_file_operations(cut.previous_summary.as_deref(), &cut.prefix)?;
    let summary = replace_checkpoint_file_operations(&generated_summary, &file_operations)?;
    validate_checkpoint_sections(&summary)?;
    let next_action = checkpoint_first_next_step(&summary)?;

    let split_turn_handoff = cut
        .split_active_turn
        .then(|| render_split_turn_handoff(&summary, &cut.prefix, &next_action))
        .transpose()?;
    let continuation = render_internal_continuation(&next_action, split_turn_handoff.as_deref());
    let event = ContextCompactionEvent::checkpointed(
        summary.clone(),
        cut.cut_end,
        CompactionCheckpoint {
            next_action,
            continuation: continuation.clone(),
            split_turn_handoff,
            file_operations,
        },
    );
    let history = super::history_compact::compose_with_summary(
        &summary,
        &agent.history,
        cut.cut_end,
        cut.preserved_user_index,
        Some(&continuation),
    )?;

    let current_turn_start_index = cut.preserved_user_index.map(|_| 1).or_else(|| {
        agent
            .turn
            .current_turn_start_index
            // The replacement summary and internal continuation precede a
            // retained active turn that did not split at its user message.
            .map(|start| 2 + start.saturating_sub(cut.cut_end))
    });
    // Validate and construct every live mirror before durable acknowledgement.
    // The post-ack install is then only infallible field replacement.
    let transcript =
        crate::protocol_frames::analyze_history_items(&history, current_turn_start_index)?;
    let protocol_frames = compacted_protocol_frames(agent, &transcript.frames, &cut, true)?;
    let mut runtime_snapshot = agent.rebuilt_runtime_snapshot_from_protocol_frames(
        &protocol_frames,
        agent.protocol_frames.len(),
        &agent.history,
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
    let protocol_frames = runtime_snapshot.active_protocol_frames();

    Ok(Ok(PreparedCompaction {
        retained_items: history.len(),
        event,
        history,
        current_turn_start_index,
        protocol_frames,
        runtime_snapshot,
    }))
}

fn compacted_protocol_frames<C: Config>(
    agent: &Agent<C>,
    candidate_frames: &[crate::protocol_frames::ProtocolFrame],
    cut: &super::history_compact::TurnCut,
    has_continuation: bool,
) -> Result<Vec<crate::protocol_frames::ProtocolFrame>> {
    anyhow::ensure!(
        agent.protocol_frames.len() == agent.history.len(),
        "cannot compact protocol identity: cached frames {} vs history {}",
        agent.protocol_frames.len(),
        agent.history.len()
    );

    let mut frames = candidate_frames.to_vec();
    let mut candidate_index = 1usize; // The compacted summary always gets a new identity.
    if let Some(old_index) = cut
        .preserved_user_index
        .filter(|index| *index < cut.cut_end)
    {
        inherit_protocol_identity(
            frames.get_mut(candidate_index),
            agent.protocol_frames.get(old_index),
        )?;
        candidate_index += 1;
    }
    if has_continuation {
        candidate_index += 1;
    }
    for old_index in cut.cut_end..agent.protocol_frames.len() {
        inherit_protocol_identity(
            frames.get_mut(candidate_index),
            agent.protocol_frames.get(old_index),
        )?;
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

fn install_prepared_compaction<C: Config + Clone>(
    agent: &mut Agent<C>,
    prepared: PreparedCompaction,
) {
    agent.history = prepared.history;
    agent.turn.current_turn_start_index = prepared.current_turn_start_index;
    agent.protocol_frames = prepared.protocol_frames;
    agent.runtime_snapshot = prepared.runtime_snapshot;
    agent.clear_active_epoch();
}

async fn commit_prepared_compaction<C>(
    agent: &mut Agent<C>,
    prepared: PreparedCompaction,
    on_event: &mut EventCallback<'_>,
) -> Result<()>
where
    C: Config + Clone,
{
    on_event(AgentEvent::ContextCompacted(prepared.event.clone())).await?;
    install_prepared_compaction(agent, prepared);
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
        // Build one durable compaction candidate. Tail pruning is part of that
        // candidate, never an unjournaled live mutation before or after it.
        let prepared_result = match prepare_compaction_candidate(agent, trigger, on_event, None).await {
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
    .await;
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
        let prepared =
            match prepare_compaction_candidate(agent, trigger, on_event, on_delta).await? {
                Ok(prepared) => prepared,
                Err(no_progress) => {
                    on_event(AgentEvent::ContextCompactionNoProgress(no_progress.clone())).await?;
                    return Ok(CompactionAttemptOutcome::NoProgress(no_progress));
                }
            };

        // Durable callback first, then install. Prepare does not mutate history,
        // so a failed callback needs no restore.
        let retained_items = prepared.retained_items;
        commit_prepared_compaction(agent, prepared, on_event).await?;
        Ok(CompactionAttemptOutcome::Compacted { retained_items })
    }
    .await;
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

async fn generate_context_summary<C>(
    agent: &Agent<C>,
    previous_summary: Option<&str>,
    head_for_summary: &[HistoryItem],
    split_active_turn: bool,
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
        retry_config: agent.retry_config.clone(),
        tool_timeout_secs: agent.tool_timeout_secs,
        turn: summary_turn,
        next_turn_id: 0,
        max_iterations: Some(1),
        max_tool_calls: Some(0),
        context_scope_state: Arc::new(std::sync::Mutex::new(ContextScopeState::default())),
        runtime_snapshot_provider: None,
        context_experiment_restore_point: None,
        logical_request_observations: super::LogicalRequestObservationTracker::default(),
        active_epoch: None,
        pressure_compaction_suppressed: true,
    };
    let prompt = render_compaction_prompt_with_workflow_facts(
        previous_summary,
        head_for_summary,
        compaction_history_char_budget(agent.active_model_metadata()),
        split_active_turn,
        &render_protected_workflow_facts(agent, head_for_summary)?,
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
    validate_checkpoint_sections(trimmed)?;
    Ok(trimmed.to_string())
}

fn render_protected_workflow_facts<C: Config>(
    agent: &Agent<C>,
    prefix: &[HistoryItem],
) -> Result<String> {
    let mut facts = Vec::new();
    let unfinished_todos = agent
        .turn
        .workflow
        .todos
        .iter()
        .filter(|todo| todo.status.is_unfinished())
        .map(|todo| format!("- todo {}: {} ({:?})", todo.id, todo.content, todo.status))
        .collect::<Vec<_>>();
    facts.push(if unfinished_todos.is_empty() {
        "待办：无未完成待办。".to_string()
    } else {
        format!("待办：\n{}", unfinished_todos.join("\n"))
    });

    let validation = if agent.turn.counters.validation_effects == 0 {
        "验证：尚未记录验证结果。".to_string()
    } else {
        format!(
            "验证：已记录 {} 项，其中失败 {} 项。",
            agent.turn.counters.validation_effects, agent.turn.counters.failed_validation_effects
        )
    };
    facts.push(validation);

    let decisions = agent
        .runtime_snapshot
        .evidence
        .iter()
        .filter(|evidence| {
            matches!(
                evidence.evidence_kind,
                crate::evidence::EvidenceKind::Decision
            ) || evidence
                .tags
                .iter()
                .any(|tag| matches!(tag.as_str(), "reconciled" | "subagent_reconciliation"))
        })
        .map(|evidence| format!("- {}: {}", evidence.title, evidence.summary))
        .collect::<Vec<_>>();
    facts.push(if decisions.is_empty() {
        "已解决问题与专家协调：无可用持久事实。".to_string()
    } else {
        format!("已解决问题与专家协调：\n{}", decisions.join("\n"))
    });

    let successful_questions = prefix
        .iter()
        .flat_map(|item| match item {
            HistoryItem::AssistantToolCalls { calls, .. } => calls
                .iter()
                .filter(|call| call.name == "question")
                .map(|call| call.call_id.as_str())
                .collect::<Vec<_>>(),
            _ => Vec::new(),
        })
        .collect::<BTreeSet<_>>();
    let answers = prefix
        .iter()
        .filter_map(|item| match item {
            HistoryItem::ToolOutput {
                call_id,
                output_json,
            } if successful_questions.contains(call_id.as_str()) => Some((call_id, output_json)),
            _ => None,
        })
        .filter_map(|(call_id, output_json)| {
            let output = serde_json::from_str::<Value>(output_json).ok()?;
            (output.get("ok").and_then(Value::as_bool) == Some(true)).then(|| {
                format!(
                    "- resolved question {call_id}: {}",
                    render_tool_output_for_compaction(output_json)
                )
            })
        })
        .collect::<Vec<_>>();
    facts.push(if answers.is_empty() {
        "已解决问题答案：无。".to_string()
    } else {
        format!("已解决问题答案：\n{}", answers.join("\n"))
    });

    Ok(facts.join("\n\n"))
}

fn validate_compaction_runtime_state<C: Config>(agent: &Agent<C>) -> Result<()> {
    // History is the protocol authority. Only structural completeness is
    // required before selection; no multi-copy payload equality checks.
    analyze_history_items(&agent.history, agent.turn.current_turn_start_index)?;
    agent.runtime_snapshot.validate_references()?;
    Ok(())
}

pub(super) fn render_compaction_prompt(
    previous_summary: Option<&str>,
    head_for_summary: &[HistoryItem],
    history_char_budget: usize,
) -> String {
    render_compaction_prompt_with_workflow_facts(
        previous_summary,
        head_for_summary,
        history_char_budget,
        false,
        "无",
    )
}

fn render_compaction_prompt_with_workflow_facts(
    previous_summary: Option<&str>,
    head_for_summary: &[HistoryItem],
    history_char_budget: usize,
    split_active_turn: bool,
    workflow_facts: &str,
) -> String {
    let serialized_history =
        render_bounded_compaction_history(head_for_summary, history_char_budget);
    let split_turn_instruction = split_active_turn.then_some(
        "本次是活动回合的前缀压缩：仅总结提供的已退休前缀；保留的用户消息和近期尾部不在本次摘要输入中。交接必须明确当前阶段、精确下一步、未完成工作和不得重复的决定。",
    );
    let common = format!(
        "\n\n受保护工作流事实（必须合并，不得被大工具输出挤出；缺失或无法解析的事实必须明确标为未知）：\n{}{}",
        workflow_facts,
        split_turn_instruction
            .map(|instruction| format!("\n\n{instruction}"))
            .unwrap_or_default(),
    );
    match previous_summary {
        Some(previous_summary) => {
            let previous_summary =
                prioritize_checkpoint(previous_summary, history_char_budget.clamp(512, 8_000));
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

pub(crate) fn render_internal_continuation(
    next_action: &str,
    split_turn_handoff: Option<&str>,
) -> String {
    let mut continuation = format!(
        "Continue from the Current Execution State in the checkpoint. Perform this exact first pending action now: {next_action}. Do not restart completed planning, re-ask resolved decisions, or retry rejected approaches."
    );
    if let Some(handoff) = split_turn_handoff {
        continuation.push_str("\n\nActive-turn handoff:\n");
        continuation.push_str(handoff);
    }
    continuation
}

pub(crate) fn render_split_turn_handoff(
    summary: &str,
    prefix: &[HistoryItem],
    next_action: &str,
) -> Result<String> {
    let in_progress = checkpoint_section(summary, "### In Progress")
        .ok_or_else(|| anyhow!("split-turn checkpoint is missing In Progress"))?;
    let decisions = checkpoint_section(summary, "## Key Decisions")
        .ok_or_else(|| anyhow!("split-turn checkpoint is missing Key Decisions"))?;
    let retired_calls = prefix
        .iter()
        .filter_map(|item| match item {
            HistoryItem::AssistantToolCalls { calls, .. } => Some(
                calls
                    .iter()
                    .map(|call| format!("{} ({})", call.call_id, call.name))
                    .collect::<Vec<_>>(),
            ),
            _ => None,
        })
        .flatten()
        .collect::<Vec<_>>();

    Ok(format!(
        "Current phase from the retired checkpoint prefix:\n{}\n\nExact next action from Next Steps:\n{}\n\nResolved or rejected decisions not to repeat:\n{}\n\nRetired completed tool calls: {}",
        in_progress.trim(),
        next_action,
        decisions.trim(),
        if retired_calls.is_empty() {
            "none recorded".to_string()
        } else {
            retired_calls.join(", ")
        }
    ))
}

fn cumulative_file_operations(
    previous_summary: Option<&str>,
    prefix: &[HistoryItem],
) -> Result<CompactionFileOperations> {
    let mut read_files = BTreeSet::new();
    let mut modified_files = BTreeSet::new();
    if let Some(summary) = previous_summary {
        // HistoryItem intentionally keeps legacy summaries as plain text. A
        // summary that does not carry the checkpoint schema predates file-op
        // metadata and contributes no cumulative paths.
        if let Ok(previous) = checkpoint_file_operations(summary) {
            read_files.extend(previous.read_files);
            modified_files.extend(previous.modified_files);
        }
    }

    let mut calls_by_id = BTreeMap::new();
    for item in prefix {
        if let HistoryItem::AssistantToolCalls { calls, .. } = item {
            for call in calls {
                calls_by_id.insert(call.call_id.as_str(), call);
            }
        }
    }
    for item in prefix {
        let HistoryItem::ToolOutput {
            call_id,
            output_json,
        } = item
        else {
            continue;
        };
        let Some(call) = calls_by_id.get(call_id.as_str()) else {
            continue;
        };
        let output = serde_json::from_str::<Value>(output_json).with_context(|| {
            format!(
                "cannot derive file operations: successful tool output for {} is malformed",
                call.call_id
            )
        })?;
        if output.get("ok").and_then(Value::as_bool) != Some(true) {
            continue;
        }
        let arguments = serde_json::from_str::<Value>(&call.arguments_json).with_context(|| {
            format!(
                "cannot derive file operations: tool arguments for {} are malformed",
                call.call_id
            )
        })?;
        let paths = json_path_values(&arguments);
        match call.name.as_str() {
            "fs__read" | "fs__list" | "search__rg" | "codegraph__codegraph_explore" => {
                read_files.extend(paths);
            }
            "edit__apply_patch" | "fs__write" | "fs__append" => {
                modified_files.extend(paths);
            }
            _ => {}
        }
    }
    read_files.retain(|path| !modified_files.contains(path));
    Ok(CompactionFileOperations {
        read_files: read_files.into_iter().collect(),
        modified_files: modified_files.into_iter().collect(),
    })
}

fn replace_checkpoint_file_operations(
    summary: &str,
    file_operations: &CompactionFileOperations,
) -> Result<String> {
    let read = render_file_operation_list(&file_operations.read_files);
    let modified = render_file_operation_list(&file_operations.modified_files);
    replace_checkpoint_section(
        &replace_checkpoint_section(summary, "### Read", &read)?,
        "### Modified",
        &modified,
    )
}

fn render_file_operation_list(paths: &[String]) -> String {
    if paths.is_empty() {
        "无".to_string()
    } else {
        paths
            .iter()
            .map(|path| format!("- {path}"))
            .collect::<Vec<_>>()
            .join("\n")
    }
}

fn replace_checkpoint_section(summary: &str, heading: &str, body: &str) -> Result<String> {
    let start = summary
        .find(heading)
        .ok_or_else(|| anyhow!("checkpoint is missing required section {heading}"))?;
    let body_start = start + heading.len();
    let tail = &summary[body_start..];
    let end = next_checkpoint_heading_offset(tail)
        .map(|offset| body_start + offset)
        .unwrap_or(summary.len());
    Ok(format!(
        "{}{}\n{}{}",
        &summary[..body_start],
        "\n",
        body,
        &summary[end..]
    ))
}

const CHECKPOINT_HEADINGS: [&str; 11] = [
    "## Progress",
    "### Done",
    "### In Progress",
    "### Blocked",
    "## Key Decisions",
    "## Validation",
    "## File Operations",
    "### Read",
    "### Modified",
    "## Next Steps",
    "## Critical Context",
];

pub(crate) fn validate_checkpoint_sections(summary: &str) -> Result<()> {
    let sections = checkpoint_sections(summary)?;
    for heading in [
        "### Done",
        "### In Progress",
        "### Blocked",
        "## Next Steps",
        "## Critical Context",
    ] {
        if sections[heading].trim().is_empty() {
            bail!("context compaction checkpoint section {heading} must not be empty");
        }
    }
    checkpoint_first_next_step(summary)?;
    Ok(())
}

pub(crate) fn checkpoint_first_next_step(summary: &str) -> Result<String> {
    let body = checkpoint_sections(summary)?["## Next Steps"].trim();
    let first = body
        .lines()
        .find_map(|line| line.trim().strip_prefix("- "))
        .map(str::trim)
        .filter(|step| !step.is_empty())
        .ok_or_else(|| anyhow!("checkpoint Next Steps must begin with '- <exact action>'"))?;
    Ok(first.to_string())
}

fn checkpoint_sections<'a>(summary: &'a str) -> Result<BTreeMap<&'static str, &'a str>> {
    let mut headings = Vec::with_capacity(CHECKPOINT_HEADINGS.len());
    let mut line_start = 0;
    let mut expected = 0;
    for line in summary.split_inclusive('\n') {
        let text = line.trim_end_matches('\n').trim_end_matches('\r');
        if text.starts_with("## ") || text.starts_with("### ") {
            let Some(expected_heading) = CHECKPOINT_HEADINGS.get(expected) else {
                bail!("context compaction checkpoint contains an unexpected section {text}");
            };
            if text != *expected_heading {
                bail!(
                    "context compaction checkpoint requires unique ordered section {expected_heading}; found '{text}'"
                );
            }
            headings.push((*expected_heading, line_start, line_start + text.len()));
            expected += 1;
        }
        line_start += line.len();
    }
    if expected != CHECKPOINT_HEADINGS.len() {
        bail!(
            "context compaction checkpoint is missing required section {}",
            CHECKPOINT_HEADINGS[expected]
        );
    }
    let mut sections = BTreeMap::new();
    for (index, (heading, _start, end)) in headings.iter().enumerate() {
        let body_start = end + usize::from(summary.as_bytes().get(*end) == Some(&b'\n'));
        let body_end = headings
            .get(index + 1)
            .map(|(_, start, _)| *start)
            .unwrap_or(summary.len());
        sections.insert(*heading, &summary[body_start..body_end]);
    }
    Ok(sections)
}

fn checkpoint_section<'a>(summary: &'a str, heading: &str) -> Option<&'a str> {
    let start = summary.find(heading)? + heading.len();
    let tail = &summary[start..];
    let end = next_checkpoint_heading_offset(tail).unwrap_or(tail.len());
    Some(&tail[..end])
}

fn next_checkpoint_heading_offset(tail: &str) -> Option<usize> {
    [tail.find("\n## "), tail.find("\n### ")]
        .into_iter()
        .flatten()
        .min()
}

pub(crate) fn checkpoint_file_operations(summary: &str) -> Result<CompactionFileOperations> {
    validate_checkpoint_sections(summary)?;
    let read_files = checkpoint_file_section(summary, "### Read")?;
    let modified_files = checkpoint_file_section(summary, "### Modified")?;
    let overlap = read_files
        .intersection(&modified_files)
        .cloned()
        .collect::<Vec<_>>();
    ensure!(
        overlap.is_empty(),
        "context compaction checkpoint cannot classify modified paths as read-only: {}",
        overlap.join(", ")
    );
    Ok(CompactionFileOperations {
        read_files: read_files.into_iter().collect(),
        modified_files: modified_files.into_iter().collect(),
    })
}

fn checkpoint_file_section(summary: &str, heading: &str) -> Result<BTreeSet<String>> {
    let body = checkpoint_section(summary, heading)
        .ok_or_else(|| anyhow!("checkpoint is missing required section {heading}"))?;
    body.lines()
        .filter(|line| !line.trim().is_empty())
        .map(|line| {
            let line = line.trim();
            if line == "无" {
                return Ok::<Option<String>, anyhow::Error>(None);
            }
            let path = line
                .strip_prefix("- ")
                .ok_or_else(|| {
                    anyhow!("checkpoint section {heading} must contain only '- <path>' lines")
                })?
                .trim();
            if path.is_empty() {
                bail!("checkpoint section {heading} contains an empty path");
            }
            if path == "无" {
                return Ok::<Option<String>, anyhow::Error>(None);
            }
            Ok::<Option<String>, anyhow::Error>(Some(path.to_string()))
        })
        .collect::<Result<Vec<_>>>()
        .map(|paths| paths.into_iter().flatten().collect())
}

fn json_path_values(value: &Value) -> BTreeSet<String> {
    let mut paths = BTreeSet::new();
    match value {
        Value::Object(object) => {
            for (key, value) in object {
                if matches!(key.as_str(), "path" | "file" | "file_path") {
                    if let Some(path) = value.as_str() {
                        paths.insert(path.to_string());
                    }
                }
                paths.extend(json_path_values(value));
            }
        }
        Value::Array(values) => {
            for value in values {
                paths.extend(json_path_values(value));
            }
        }
        _ => {}
    }
    paths
}

fn prioritize_checkpoint(summary: &str, max_chars: usize) -> String {
    if summary.chars().count() <= max_chars {
        return summary.to_string();
    }
    // Each section is selected independently. A verbose earlier section must
    // not starve the execution facts that make the next compaction actionable.
    const PRIORITY_HEADINGS: [&str; 6] = [
        "## Key Decisions",
        "### In Progress",
        "### Blocked",
        "## Validation",
        "## Next Steps",
        "## Critical Context",
    ];
    let present = PRIORITY_HEADINGS
        .into_iter()
        .filter_map(|heading| checkpoint_section(summary, heading).map(|body| (heading, body)))
        .collect::<Vec<_>>();
    if present.is_empty() {
        return truncate_for_compaction(
            summary,
            max_chars,
            "… [previous checkpoint reduced for compaction]",
        );
    }

    let separators = present.len().saturating_sub(1);
    let headings = present
        .iter()
        .map(|(heading, _)| heading.chars().count() + 1)
        .sum::<usize>();
    // Keep every critical section independently, including its last facts,
    // rather than letting an early section consume the entire update budget.
    let body_budget = max_chars.saturating_sub(headings + separators) / present.len();
    let reduced = present
        .into_iter()
        .map(|(heading, body)| {
            format!(
                "{heading}\n{}",
                truncate_checkpoint_section(body.trim(), body_budget.max(1))
            )
        })
        .collect::<Vec<_>>()
        .join("\n");
    truncate_for_compaction(
        &reduced,
        max_chars,
        "… [previous checkpoint reduced for compaction]",
    )
}

fn truncate_checkpoint_section(text: &str, max_chars: usize) -> String {
    if text.chars().count() <= max_chars {
        return text.to_string();
    }
    let marker = "… [section reduced]";
    if max_chars <= marker.chars().count() {
        return marker.chars().take(max_chars).collect();
    }
    let keep = max_chars - marker.chars().count();
    let head = keep / 2;
    let tail = keep.saturating_sub(head);
    format!(
        "{}{}{}",
        text.chars().take(head).collect::<String>(),
        marker,
        text.chars()
            .rev()
            .take(tail)
            .collect::<String>()
            .chars()
            .rev()
            .collect::<String>()
    )
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

#[cfg(test)]
mod transaction_tests {
    use super::*;
    use async_openai::{Client, config::OpenAIConfig};
    use std::sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    };

    fn test_agent() -> Agent<OpenAIConfig> {
        Agent::new(
            Client::with_config(
                OpenAIConfig::new()
                    .with_api_base("http://127.0.0.1:1")
                    .with_api_key("test"),
            ),
            "test-model",
            1,
            1,
        )
    }

    fn protocol_frames_with_spans(
        history: &[HistoryItem],
    ) -> Vec<crate::protocol_frames::ProtocolFrame> {
        crate::protocol_frames::analyze_history_items(history, None)
            .expect("history is valid")
            .frames
            .into_iter()
            .enumerate()
            .map(|(index, mut frame)| {
                let sequence = index as u64 + 1;
                frame.source_provenance = Some(
                    crate::runtime_context::RuntimeFrameProvenance::new(
                        crate::runtime_context::RuntimeSource::Transcript,
                    )
                    .with_span(
                        crate::runtime_context::SourceSpan::new(sequence, sequence)
                            .expect("singleton source span"),
                    ),
                );
                frame
            })
            .collect()
    }

    fn prepared(agent: &Agent<OpenAIConfig>, history: Vec<HistoryItem>) -> PreparedCompaction {
        let transcript = crate::protocol_frames::analyze_history_items(&history, None)
            .expect("candidate history is valid");
        let mut runtime_snapshot = agent
            .rebuilt_runtime_snapshot_from_protocol_frames(
                &transcript.frames,
                agent.protocol_frames.len(),
                &agent.history,
            )
            .expect("candidate snapshot");
        merge_non_protocol_runtime_metadata(&mut runtime_snapshot, &agent.runtime_snapshot);
        rebind_active_protocol_from_history(&mut runtime_snapshot, &history)
            .expect("candidate protocol binding");
        runtime_snapshot
            .heal_references()
            .expect("candidate references");
        let protocol_frames = runtime_snapshot.active_protocol_frames();
        PreparedCompaction {
            retained_items: history.len(),
            event: ContextCompactionEvent::succeeded("summary", 1),
            history,
            current_turn_start_index: None,
            protocol_frames,
            runtime_snapshot,
        }
    }

    fn checkpoint_with_files(read: &str, modified: &str) -> String {
        format!(
            "## Progress\n### Done\n- work\n### In Progress\n- continue\n### Blocked\n- none\n## Key Decisions\n- scope\n## Validation\n- pending\n## File Operations\n### Read\n- {read}\n### Modified\n- {modified}\n## Next Steps\n- continue\n## Critical Context\n- facts"
        )
    }

    #[test]
    fn cumulative_file_operations_upgrades_legacy_summary_and_prefers_modified() {
        let prefix = vec![
            HistoryItem::AssistantToolCalls {
                text: None,
                calls: vec![
                    HistoryToolCall {
                        call_id: "read-shared".into(),
                        name: "fs__read".into(),
                        arguments_json: r#"{"path":"src/shared.rs"}"#.into(),
                    },
                    HistoryToolCall {
                        call_id: "read-only".into(),
                        name: "fs__read".into(),
                        arguments_json: r#"{"path":"src/read_only.rs"}"#.into(),
                    },
                    HistoryToolCall {
                        call_id: "modify-shared".into(),
                        name: "edit__apply_patch".into(),
                        arguments_json: r#"{"edits":[{"path":"src/shared.rs"}]}"#.into(),
                    },
                ],
            },
            HistoryItem::ToolOutput {
                call_id: "read-shared".into(),
                output_json: r#"{"ok":true}"#.into(),
            },
            HistoryItem::ToolOutput {
                call_id: "read-only".into(),
                output_json: r#"{"ok":true}"#.into(),
            },
            HistoryItem::ToolOutput {
                call_id: "modify-shared".into(),
                output_json: r#"{"ok":true}"#.into(),
            },
        ];

        let operations = cumulative_file_operations(Some("legacy summary"), &prefix)
            .expect("legacy summaries have no structured file metadata");

        assert_eq!(operations.read_files, vec!["src/read_only.rs"]);
        assert_eq!(operations.modified_files, vec!["src/shared.rs"]);
    }

    #[test]
    fn checkpoint_file_operations_rejects_read_modified_overlap() {
        let checkpoint = checkpoint_with_files("src/shared.rs", "src/shared.rs");
        let error = checkpoint_file_operations(&checkpoint)
            .expect_err("modified files cannot remain classified as read-only");

        assert!(error.to_string().contains("src/shared.rs"));
    }

    #[tokio::test]
    async fn rejected_durable_compaction_does_not_change_live_history() {
        let mut agent = test_agent();
        agent.history = vec![HistoryItem::user("old"), HistoryItem::assistant("reply")];
        let original = agent.history.clone();
        let candidate = prepared(
            &agent,
            vec![
                HistoryItem::context_summary("summary"),
                HistoryItem::assistant("reply"),
            ],
        );
        let mut on_event = |_event| {
            Box::pin(async { Err(anyhow::anyhow!("durable append rejected")) })
                as BoxFuture<'_, Result<()>>
        };

        assert!(
            commit_prepared_compaction(&mut agent, candidate, &mut on_event)
                .await
                .is_err()
        );
        assert_eq!(agent.history, original);
    }

    #[tokio::test]
    async fn acknowledged_durable_compaction_installs_exact_candidate() {
        let mut agent = test_agent();
        agent.history = vec![HistoryItem::user("old"), HistoryItem::assistant("reply")];
        let expected = vec![
            HistoryItem::context_summary("summary"),
            HistoryItem::assistant("reply"),
        ];
        let candidate = prepared(&agent, expected.clone());
        let acknowledged = Arc::new(AtomicBool::new(false));
        let observed = acknowledged.clone();
        let mut on_event = move |event| {
            observed.store(
                matches!(event, AgentEvent::ContextCompacted(_)),
                Ordering::SeqCst,
            );
            Box::pin(async { Ok(()) }) as BoxFuture<'_, Result<()>>
        };

        commit_prepared_compaction(&mut agent, candidate, &mut on_event)
            .await
            .expect("durable acknowledgement installs candidate");

        assert!(acknowledged.load(Ordering::SeqCst));
        assert_eq!(agent.history, expected);
    }
}
