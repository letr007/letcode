use super::*;
use crate::protocol_frames::analyze_history_items;
use crate::runtime_context::RuntimeSnapshot;
use futures_util::FutureExt;
use futures_util::future::BoxFuture;

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

    let Some(cut) = super::history_compact::plan_turn_cut(
        &agent.history,
        agent.turn.current_turn_start_index,
    )?
    else {
        return Ok(Err(CompactionNoProgress {
            trigger,
            blockers: vec![CompactionBlocker::NoSafeBoundary],
        }));
    };

    let checkpoint = PreCompactionCheckpoint::capture(agent);
    let summary = crate::transcript::transcript_projection::sanitize_compaction_summary_body(
        &generate_context_summary(
            agent,
            cut.previous_summary.as_deref(),
            &cut.prefix,
            on_event,
            on_delta,
        )
        .await?,
    );

    let mut history =
        super::history_compact::compose_with_summary(&summary, &agent.history, cut.cut_end)?;
    if agent.compaction_config.prune {
        let _ = super::history_compact::stub_large_tool_outputs(&mut history, 0);
    }

    let current_turn_start_index = agent.turn.current_turn_start_index.and_then(|start| {
        if start >= cut.cut_end {
            Some(1 + (start - cut.cut_end))
        } else {
            None
        }
    });
    crate::protocol_frames::analyze_history_items(&history, current_turn_start_index)?;

    let event = ContextCompactionEvent {
        outcome: "succeeded".into(),
        summary: summary.clone(),
        tail_start_index: cut.cut_end,
        original_history_items: 0,
        retained_history_items: 0,
        retired_source_spans: Vec::new(),
        frame_identity_bindings: Vec::new(),
        derived_coverage: None,
        detail: None,
    };

    Ok(Ok((
        checkpoint,
        PreparedCompaction {
            retained_items: history.len(),
            event,
            snapshot: agent.runtime_snapshot.clone(),
            protocol_frames: Vec::new(),
            history,
            current_turn_start_index,
        },
    )))
}

fn install_prepared_compaction<C: Config + Clone>(
    agent: &mut Agent<C>,
    prepared: &PreparedCompaction,
) -> Result<()> {
    agent.install_history(
        prepared.history.clone(),
        prepared.current_turn_start_index,
    )
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
        install_prepared_compaction(agent, &prepared)?;
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
    if !agent.compaction_config.prune {
        return Ok(());
    }
    validate_compaction_runtime_state(agent)?;
    if !super::history_compact::stub_large_tool_outputs(
        &mut agent.history,
        preserve_recent_budget,
    ) {
        return Ok(());
    }
    agent.publish_history_to_protocol_mirrors()?;
    agent.clear_active_epoch();
    Ok(())
}

/// Returns true when at least one large historical tool output was stubbed.
pub(super) fn emergency_prune_tool_outputs_for_pressure<C: Config>(
    agent: &mut Agent<C>,
) -> Result<bool> {
    if !agent.compaction_config.prune {
        return Ok(false);
    }
    let mut history = agent.history.clone();
    if !super::history_compact::stub_large_tool_outputs(&mut history, 0) {
        return Ok(false);
    }
    let turn_start = agent.turn.current_turn_start_index;
    agent.install_history(history, turn_start)?;
    Ok(true)
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

