use super::*;

#[derive(Debug, Clone)]
pub(super) struct CompactionSelection {
    pub(super) previous_summary: Option<String>,
    pub(super) head_for_summary: Vec<HistoryItem>,
    pub(super) tail_items: Vec<HistoryItem>,
    pub(super) tail_start_index: usize,
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
    let protected_start_index = agent.history.len();
    if protected_start_index == 0 {
        return Ok(ManualCompactionOutcome::NothingToCompact);
    }
    let model = agent.active_model_metadata();
    let input_budget = model
        .context_window_tokens()
        .saturating_sub(model.output_reserve_tokens())
        .max(1);
    let preserve_recent_budget = default_preserve_recent_budget(input_budget);
    let selection = match select_compaction_segments(
        &agent.history,
        protected_start_index,
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
    compact_selected_context(
        agent,
        selection,
        protected_start_index,
        &mut on_event,
        Some(&mut on_delta),
    )
    .await
    .map(|retained_items| ManualCompactionOutcome::Compacted { retained_items })
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
    let preserve_recent_budget =
        default_preserve_recent_budget(agent.active_model_metadata().context_window_tokens());
    prune_old_tool_outputs(agent, protected_start_index, preserve_recent_budget);

    if !agent.compaction_config.auto && !agent.needs_compaction {
        return Ok(protected_start_index);
    }

    let build = build_request(RequestBuilderInput {
        protocol: agent.active_protocol(),
        model_id: &agent.model,
        model: agent.active_model_metadata(),
        prelude: turn_prelude,
        history: &agent.history,
        protected_start_index,
        tools: tool_definitions,
        evidence: &agent.evidence,
    })?;
    let should_compact = agent.needs_compaction
        || build.budget.truncated
        || build.budget.estimated_request_tokens
            >= build
                .budget
                .input_budget_tokens
                .saturating_sub(compaction_reserved_tokens(
                    agent,
                    build.budget.input_budget_tokens,
                ));
    if !should_compact {
        return Ok(protected_start_index);
    }

    let preserve_recent_budget = default_preserve_recent_budget(build.budget.input_budget_tokens);
    compact_context(
        agent,
        protected_start_index,
        preserve_recent_budget,
        on_event,
    )
    .await
}

fn compaction_reserved_tokens<C: Config>(agent: &Agent<C>, input_budget_tokens: u64) -> u64 {
    agent
        .compaction_config
        .reserved
        .unwrap_or_else(|| input_budget_tokens.saturating_div(10).clamp(256, 2_048))
}

pub(super) fn prune_old_tool_outputs<C: Config>(
    agent: &mut Agent<C>,
    protected_start_index: usize,
    preserve_recent_budget: u64,
) {
    if !agent.compaction_config.prune {
        return;
    }

    let Ok(selection) = select_compaction_segments(
        &agent.history,
        protected_start_index,
        &agent.compaction_config,
        preserve_recent_budget,
    ) else {
        return;
    };

    let protect_start = recent_token_protected_start(
        &agent.history[..protected_start_index],
        COMPACTION_PRUNE_PROTECT_TOKENS,
    );
    let prune_until = selection.tail_start_index.min(protect_start);

    let call_names = build_tool_call_name_index(&agent.history[..protected_start_index]);
    for item in agent.history[..prune_until].iter_mut() {
        let HistoryItem::ToolOutput {
            call_id,
            output_json,
        } = item
        else {
            continue;
        };
        if output_json.chars().count() < COMPACTION_PRUNE_MIN_OUTPUT_CHARS {
            continue;
        }
        if output_json.contains(COMPACTION_PRUNED_MARKER) {
            continue;
        }
        if call_names.get(call_id).is_some_and(|name| name == "skill") {
            continue;
        }

        *output_json =
            build_pruned_tool_output_json(output_json, call_names.get(call_id).map(String::as_str));
    }
}

async fn compact_context<C, E, Efut>(
    agent: &mut Agent<C>,
    protected_start_index: usize,
    preserve_recent_budget: u64,
    on_event: &mut E,
) -> Result<usize>
where
    C: Config + Clone,
    E: FnMut(AgentEvent) -> Efut,
    Efut: Future<Output = Result<()>>,
{
    let protected_start_index = protected_start_index.min(agent.history.len());
    if protected_start_index == 0 {
        bail!("context compaction cannot summarize the protected current turn");
    }

    let selection = select_compaction_segments(
        &agent.history,
        protected_start_index,
        &agent.compaction_config,
        preserve_recent_budget,
    )?;
    if selection.head_for_summary.is_empty() {
        bail!("context compaction could not select any historical items to summarize");
    }

    compact_selected_context(agent, selection, protected_start_index, on_event, None).await
}

async fn compact_selected_context<C, E, Efut>(
    agent: &mut Agent<C>,
    selection: CompactionSelection,
    protected_start_index: usize,
    on_event: &mut E,
    on_delta: Option<&mut (dyn FnMut(&str) -> Result<()> + Send + '_)>,
) -> Result<usize>
where
    C: Config + Clone,
    E: FnMut(AgentEvent) -> Efut,
    Efut: Future<Output = Result<()>>,
{
    let original_history_items = agent.history.len();
    let summary = generate_context_summary(
        agent,
        selection.previous_summary.as_deref(),
        &selection.head_for_summary,
        on_delta,
    )
    .await?;

    let mut retained_history_items = Vec::with_capacity(
        1 + selection.tail_items.len() + agent.history.len().saturating_sub(protected_start_index),
    );
    retained_history_items.push(HistoryItem::context_summary(summary.clone()));
    retained_history_items.extend(selection.tail_items.iter().cloned());
    retained_history_items.extend(agent.history[protected_start_index..].iter().cloned());
    agent.history = retained_history_items;
    agent.needs_compaction = false;

    let event = ContextCompactionEvent {
        summary,
        tail_start_index: selection.tail_start_index,
        original_history_items,
        retained_history_items: agent.history.len(),
    };
    on_event(AgentEvent::ContextCompacted(event)).await?;
    Ok(1 + selection.tail_items.len())
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
        history: Vec::new(),
        evidence: Vec::new(),
        tools: ToolRegistry::new(),
        skill_registry: None,
        skill_cards: Vec::new(),
        subagent_delegate: None,
        permission_policy: PermissionPolicy::default(),
        compaction_config: CompactionConfig {
            auto: false,
            ..CompactionConfig::default()
        },
        retry_config: agent.retry_config.clone(),
        needs_compaction: false,
        turn: TurnRuntimeState::default(),
        next_turn_id: 0,
        max_iterations: 1,
        max_tool_calls: 0,
        context_scope_state: Arc::new(std::sync::Mutex::new(ContextScopeState::default())),
        context_experiment_restore_point: None,
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
        |_| std::future::ready(Ok(false)),
    ))
    .await?;
    let trimmed = summary.trim();
    if trimmed.is_empty() {
        bail!("context compaction produced an empty summary")
    }
    Ok(trimmed.to_string())
}

pub(super) fn select_compaction_segments(
    history: &[HistoryItem],
    protected_start_index: usize,
    config: &CompactionConfig,
    preserve_recent_budget: u64,
) -> Result<CompactionSelection> {
    let protected_start_index = protected_start_index.min(history.len());
    let older = &history[..protected_start_index];
    let summary_index = older
        .iter()
        .rposition(|item| matches!(item, HistoryItem::ContextSummary { .. }));
    let previous_summary = summary_index.and_then(|index| match &older[index] {
        HistoryItem::ContextSummary { text } => Some(text.clone()),
        _ => None,
    });
    let base_start = summary_index.map(|index| index + 1).unwrap_or(0);
    let candidates = &older[base_start..];
    if candidates.is_empty() {
        bail!(NO_HISTORICAL_ITEMS_FOR_COMPACTION);
    }

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
    let tail_start_index = base_start + tail_relative_start;
    let head_for_summary = older[base_start..tail_start_index].to_vec();
    let tail_items = older[tail_start_index..protected_start_index].to_vec();
    if head_for_summary.is_empty() {
        bail!(NO_OLDER_ITEMS_AFTER_TAIL);
    }
    Ok(CompactionSelection {
        previous_summary,
        head_for_summary,
        tail_items,
        tail_start_index,
    })
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
    let input_budget = model
        .context_window_tokens()
        .saturating_sub(model.output_reserve_tokens())
        .max(1);
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

fn recent_token_protected_start(items: &[HistoryItem], protect_tokens: u64) -> usize {
    if protect_tokens == 0 {
        return items.len();
    }

    let mut remaining = protect_tokens;
    let mut start = items.len();
    for index in (0..items.len()).rev() {
        let cost = estimate_history_item_tokens(&items[index]);
        if cost > remaining {
            break;
        }
        start = index;
        remaining = remaining.saturating_sub(cost);
    }
    trim_tail_to_valid_boundary(items, start)
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
