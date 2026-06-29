use crate::context_tree::{ContextNodeId, ContextTreeOp, ContextTreeState};
use crate::agent::AutoContinueState;
use crate::evidence::EvidenceRecord;
use crate::request_builder::HistoryItem;
use crate::tool_format::format_tool_call;
use crate::transcript::{
    ChildSessionSummary, JobBoardEntry, TranscriptEvent, TranscriptRecord,
    ROOT_CONTEXT_BRANCH_ID,
};
use crate::tui::events::{
    AutoContinueChangedEvent, ErrorEvent, TodoSnapshotEvent, TokenUsageEvent, ToolFinishedEvent,
    ToolOutcome, ToolStartedEvent, UserMessageEvent,
};
use crate::tui::timeline::{
    COMPACTION_SEPARATOR_LABEL, MessageRole, PermissionPromptStatus, Timeline,
    restored_tool_summary,
};
use crate::user_content::UserMessageSubmission;
use crate::{agent::ConversationMessage, subagent::StructuredSubagentResult};
use anyhow::{Context, anyhow, ensure};
use std::collections::BTreeMap;
use std::path::Path;

/// Transcript restore intentionally differs from live projection in a few places:
/// - Permission decisions restore as terminal approved/denied permission items, not as pending prompts.
/// - Context compaction restores as separator + assistant summary + separator.
/// - Subagent lifecycle/result records are ignored in restored timelines.
/// - Turn audit and unknown transcript events are ignored during timeline restore.
pub(crate) fn timeline_from_transcript_records(records: &[TranscriptRecord]) -> Timeline {
    let mut projection = TranscriptTimelineProjection::default();
    for record in records {
        projection.apply_record(record);
    }
    projection.timeline
}

#[derive(Debug, Clone)]
pub(crate) struct SessionContextCursor {
    pub branch_id: Option<String>,
    pub leaf_sequence: Option<u64>,
}

#[derive(Debug, Clone)]
pub(crate) struct SessionRestoreSnapshot {
    pub session_id: String,
    pub branch_id: String,
    pub leaf_sequence: u64,
    pub records: Vec<TranscriptRecord>,
    pub messages: Vec<ConversationMessage>,
    pub history: Vec<HistoryItem>,
    pub evidence: Vec<EvidenceRecord>,
    pub latest_model: Option<String>,
    pub max_turn_id: u64,
    pub token_usage: Option<TokenUsageEvent>,
}

impl SessionRestoreSnapshot {
    pub(crate) fn evidence_count(&self) -> usize {
        self.evidence.len()
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ContextBranchInfo {
    pub branch_id: String,
    pub parent_branch_id: Option<String>,
    pub label: Option<String>,
    pub tip_sequence: u64,
    pub is_current: bool,
}

pub(crate) fn project_session_restore_snapshot(
    session_id: String,
    records: Vec<TranscriptRecord>,
    token_usage: Option<TokenUsageEvent>,
) -> anyhow::Result<SessionRestoreSnapshot> {
    build_session_context_snapshot(
        session_id,
        records,
        token_usage,
        SessionContextCursor {
            branch_id: None,
            leaf_sequence: None,
        },
    )
}

pub(crate) fn project_context_tree(records: &[TranscriptRecord]) -> anyhow::Result<ContextTreeState> {
    replay_context_tree(records)
}

pub(crate) fn replay_context_tree(records: &[TranscriptRecord]) -> anyhow::Result<ContextTreeState> {
    let mut ops = Vec::new();
    let mut saw_context_tree_metadata = false;

    for record in records {
        match &record.event {
            TranscriptEvent::ContextNodeCreated {
                node_id,
                parent_node_id,
                label,
                purpose,
                block_ref,
                source_ref,
            } => {
                saw_context_tree_metadata = true;
                let node_id = ContextNodeId::new(node_id.clone()).with_context(|| {
                    format!("invalid context node_id at transcript sequence {}", record.sequence)
                })?;
                let parent_node_id = parent_node_id
                    .as_ref()
                    .map(|value| ContextNodeId::new(value.clone()))
                    .transpose()
                    .with_context(|| {
                        format!(
                            "invalid parent context node_id at transcript sequence {}",
                            record.sequence
                        )
                    })?;
                ops.push(ContextTreeOp::CreateNode {
                    node_id,
                    parent_node_id,
                    label: label.clone(),
                    purpose: purpose.clone(),
                    block_ref: block_ref.clone(),
                    source_ref: source_ref.clone(),
                });
            }
            TranscriptEvent::ContextNodeLifecycle { node_id, status } => {
                saw_context_tree_metadata = true;
                ops.push(ContextTreeOp::SetNodeStatus {
                    node_id: ContextNodeId::new(node_id.clone()).with_context(|| {
                        format!("invalid context node_id at transcript sequence {}", record.sequence)
                    })?,
                    status: status.clone(),
                });
            }
            _ => {}
        }
    }

    if !saw_context_tree_metadata {
        return Ok(ContextTreeState::with_default_root());
    }

    ContextTreeState::replay(&ops)
}

pub(crate) fn list_context_branches(
    records: &[TranscriptRecord],
    current_branch_id: Option<&str>,
) -> anyhow::Result<Vec<ContextBranchInfo>> {
    let index = build_branch_index(records)?;
    let active_branch_id = resolve_active_branch_id(&index, current_branch_id);
    let mut branches = index
        .definitions
        .iter()
        .map(|(branch_id, definition)| {
            Ok(ContextBranchInfo {
                branch_id: branch_id.clone(),
                parent_branch_id: definition.parent_branch_id.clone(),
                label: definition.label.clone(),
                tip_sequence: index.branch_tip(branch_id)?,
                is_current: branch_id == &active_branch_id,
            })
        })
        .collect::<anyhow::Result<Vec<_>>>()?;
    branches.sort_by(|left, right| {
        (left.branch_id != ROOT_CONTEXT_BRANCH_ID)
            .cmp(&(right.branch_id != ROOT_CONTEXT_BRANCH_ID))
            .then_with(|| left.branch_id.cmp(&right.branch_id))
    });
    Ok(branches)
}

#[derive(Debug, Clone)]
struct BranchDefinition {
    parent_branch_id: Option<String>,
    base_sequence: u64,
    label: Option<String>,
}

#[derive(Debug, Clone)]
struct CheckoutState {
    branch_id: String,
    leaf_sequence: u64,
}

#[derive(Debug, Default)]
struct BranchIndex {
    definitions: BTreeMap<String, BranchDefinition>,
    latest_checkout: Option<CheckoutState>,
    branch_tips: BTreeMap<String, u64>,
}

#[derive(Debug)]
struct ResolvedBranchContext {
    branch_id: String,
    leaf_sequence: u64,
    records: Vec<TranscriptRecord>,
}

pub(crate) fn build_session_context_snapshot(
    session_id: String,
    records: Vec<TranscriptRecord>,
    token_usage: Option<TokenUsageEvent>,
    cursor: SessionContextCursor,
) -> anyhow::Result<SessionRestoreSnapshot> {
    let resolved = resolve_branch_context(records, cursor)?;
    let history = restore_session_history_projection(&resolved.records);
    let messages = history
        .clone()
        .into_iter()
        .filter_map(super::history_item_to_conversation_message)
        .collect();
    let evidence = crate::evidence::restore_evidence_records(&resolved.records)?;
    let latest_model = restore_latest_model_projection(&resolved.records);
    let max_turn_id = restore_max_turn_id_projection(&resolved.records);

    Ok(SessionRestoreSnapshot {
        session_id,
        branch_id: resolved.branch_id,
        leaf_sequence: resolved.leaf_sequence,
        records: resolved.records,
        messages,
        history,
        evidence,
        latest_model,
        max_turn_id,
        token_usage,
    })
}

fn resolve_branch_context(
    records: Vec<TranscriptRecord>,
    cursor: SessionContextCursor,
) -> anyhow::Result<ResolvedBranchContext> {
    let index = build_branch_index(&records)?;
    let branch_id = cursor
        .branch_id
        .unwrap_or_else(|| resolve_active_branch_id(&index, None));
    let default_checkout = index.latest_checkout.as_ref();
    let leaf_sequence = match cursor.leaf_sequence {
        Some(leaf_sequence) => leaf_sequence,
        None => match default_checkout {
            Some(checkout) if checkout.branch_id == branch_id => checkout.leaf_sequence,
            _ => index.branch_tip(&branch_id)?,
        },
    };

    let max_sequence = records
        .iter()
        .map(|record| record.sequence)
        .max()
        .unwrap_or(0);
    ensure!(
        leaf_sequence <= max_sequence || (leaf_sequence == 0 && max_sequence == 0),
        "session context leaf_sequence {leaf_sequence} exceeds max transcript sequence {max_sequence}"
    );

    let records = collect_branch_path_records(&records, &index, &branch_id, leaf_sequence)?;
    Ok(ResolvedBranchContext {
        branch_id,
        leaf_sequence,
        records,
    })
}

fn build_branch_index(records: &[TranscriptRecord]) -> anyhow::Result<BranchIndex> {
    let mut index = BranchIndex::default();
    index.definitions.insert(
        ROOT_CONTEXT_BRANCH_ID.to_string(),
        BranchDefinition {
            parent_branch_id: None,
            base_sequence: 0,
            label: None,
        },
    );
    index
        .branch_tips
        .insert(ROOT_CONTEXT_BRANCH_ID.to_string(), 0);

    for (position, record) in records.iter().enumerate() {
        match &record.event {
            TranscriptEvent::ContextBranchCreated {
                branch_id,
                parent_branch_id,
                base_sequence,
                label,
            } => {
                ensure!(
                    !index.definitions.contains_key(branch_id),
                    "duplicate context branch_id '{branch_id}'"
                );
                ensure!(
                    index.definitions.contains_key(parent_branch_id),
                    "missing parent context branch '{parent_branch_id}' for branch '{branch_id}'"
                );
                ensure!(
                    base_sequence_resolves_on_parent_path(
                        &records[..position],
                        &index,
                        parent_branch_id,
                        *base_sequence,
                    )?,
                    "context branch '{branch_id}' base_sequence {base_sequence} is not resolvable on parent branch '{parent_branch_id}'"
                );
                index.definitions.insert(
                    branch_id.clone(),
                    BranchDefinition {
                        parent_branch_id: Some(parent_branch_id.clone()),
                        base_sequence: *base_sequence,
                        label: label.clone(),
                    },
                );
                index.branch_tips.insert(branch_id.clone(), *base_sequence);
            }
            TranscriptEvent::ContextCheckout {
                branch_id,
                leaf_sequence,
            } => {
                ensure!(
                    index.definitions.contains_key(branch_id),
                    "unknown context branch '{branch_id}' in checkout metadata"
                );
                index.latest_checkout = Some(CheckoutState {
                    branch_id: branch_id.clone(),
                    leaf_sequence: *leaf_sequence,
                });
            }
            TranscriptEvent::ContextBranchSummary {
                branch_id,
                leaf_sequence,
                ..
            } => {
                ensure!(
                    index.definitions.contains_key(branch_id),
                    "unknown context branch '{branch_id}' in branch summary metadata"
                );
                let branch_tip = branch_tip_for_records(records, &index, branch_id)?;
                ensure!(
                    *leaf_sequence <= branch_tip,
                    "context branch summary leaf_sequence {leaf_sequence} exceeds tip {branch_tip} for branch '{branch_id}'"
                );
            }
            _ => {
                if record.event.is_context_branch_metadata() {
                    continue;
                }
                let effective_branch_id = effective_branch_id(record);
                ensure!(
                    index.definitions.contains_key(effective_branch_id),
                    "unknown context branch '{effective_branch_id}' in record scope at sequence {}",
                    record.sequence
                );
                index.branch_tips.insert(
                    effective_branch_id.to_string(),
                    branch_tip_for_records(records, &index, effective_branch_id)?,
                );
            }
        }
    }

    if let Some(checkout) = &index.latest_checkout {
        let branch_tip = index.branch_tip(&checkout.branch_id)?;
        ensure!(
            checkout.leaf_sequence <= branch_tip,
            "context checkout leaf_sequence {} exceeds tip {} for branch '{}'",
            checkout.leaf_sequence,
            branch_tip,
            checkout.branch_id
        );
    }

    Ok(index)
}

fn base_sequence_resolves_on_parent_path(
    records: &[TranscriptRecord],
    index: &BranchIndex,
    parent_branch_id: &str,
    base_sequence: u64,
) -> anyhow::Result<bool> {
    if base_sequence == 0 {
        return Ok(true);
    }

    if base_sequence > branch_tip_for_records(records, index, parent_branch_id)? {
        return Ok(false);
    }

    let path = collect_branch_path_records(records, index, parent_branch_id, base_sequence)?;
    Ok(path.iter().any(|record| record.sequence == base_sequence))
}

fn branch_tip_for_records(
    records: &[TranscriptRecord],
    index: &BranchIndex,
    branch_id: &str,
) -> anyhow::Result<u64> {
    let definition = index
        .definitions
        .get(branch_id)
        .ok_or_else(|| anyhow!("unknown context branch '{branch_id}'"))?;
    let local_tip = records
        .iter()
        .filter(|record| !record.event.is_context_branch_metadata())
        .filter(|record| effective_branch_id(record) == branch_id)
        .map(|record| record.sequence)
        .max()
        .unwrap_or(definition.base_sequence);
    Ok(local_tip.max(definition.base_sequence))
}

fn collect_branch_path_records(
    records: &[TranscriptRecord],
    index: &BranchIndex,
    branch_id: &str,
    leaf_sequence: u64,
) -> anyhow::Result<Vec<TranscriptRecord>> {
    let branch_tip = index.branch_tip(branch_id)?;
    ensure!(
        leaf_sequence <= branch_tip,
        "requested leaf_sequence {leaf_sequence} exceeds tip {branch_tip} for branch '{branch_id}'"
    );

    if branch_id == ROOT_CONTEXT_BRANCH_ID {
        return Ok(records
            .iter()
            .filter(|record| !record.event.is_context_branch_metadata())
            .filter(|record| effective_branch_id(record) == ROOT_CONTEXT_BRANCH_ID)
            .filter(|record| record.sequence <= leaf_sequence)
            .cloned()
            .collect());
    }

    let definition = index
        .definitions
        .get(branch_id)
        .ok_or_else(|| anyhow!("unknown context branch '{branch_id}'"))?;
    let parent_branch_id = definition
        .parent_branch_id
        .as_deref()
        .ok_or_else(|| anyhow!("context branch '{branch_id}' is missing a parent"))?;
    ensure!(
        leaf_sequence >= definition.base_sequence,
        "requested leaf_sequence {leaf_sequence} precedes base_sequence {} for branch '{branch_id}'",
        definition.base_sequence
    );

    let mut path = collect_branch_path_records(records, index, parent_branch_id, definition.base_sequence)?;
    path.extend(
        records
            .iter()
            .filter(|record| !record.event.is_context_branch_metadata())
            .filter(|record| effective_branch_id(record) == branch_id)
            .filter(|record| record.sequence <= leaf_sequence)
            .cloned(),
    );
    Ok(path)
}

fn effective_branch_id(record: &TranscriptRecord) -> &str {
    record
        .context_branch_id
        .as_deref()
        .unwrap_or(ROOT_CONTEXT_BRANCH_ID)
}

fn resolve_active_branch_id(index: &BranchIndex, current_branch_id: Option<&str>) -> String {
    current_branch_id
        .map(str::to_string)
        .or_else(|| index.latest_checkout.as_ref().map(|checkout| checkout.branch_id.clone()))
        .unwrap_or_else(|| ROOT_CONTEXT_BRANCH_ID.to_string())
}

impl BranchIndex {
    fn branch_tip(&self, branch_id: &str) -> anyhow::Result<u64> {
        self.branch_tips
            .get(branch_id)
            .copied()
            .ok_or_else(|| anyhow!("unknown context branch '{branch_id}'"))
    }
}

pub(crate) fn restore_session_history_projection(records: &[TranscriptRecord]) -> Vec<HistoryItem> {
    let mut history = Vec::new();
    for record in records {
        match &record.event {
            TranscriptEvent::ContextCompaction(event) => {
                let tail_start = event.tail_start_index.min(history.len());
                let mut compacted =
                    Vec::with_capacity(1 + history.len().saturating_sub(tail_start));
                compacted.push(HistoryItem::context_summary(event.summary.clone()));
                compacted.extend(history.drain(tail_start..));
                history = compacted;
            }
            TranscriptEvent::TurnInterrupted { .. } => {
                close_interrupted_turn(&mut history);
            }
            TranscriptEvent::TurnFinalized(event) if event.outcome == "interrupted" => {
                close_interrupted_turn(&mut history);
            }
            _ => super::append_history_item_from_transcript_record(&mut history, record),
        }
    }
    history
}

fn close_interrupted_turn(history: &mut Vec<HistoryItem>) {
    let Some(last_conversation_item) = history.iter().rfind(|item| {
        matches!(
            item,
            HistoryItem::UserMessage { .. }
                | HistoryItem::InternalContinuation { .. }
                | HistoryItem::AssistantText { .. }
                | HistoryItem::ContextSummary { .. }
        )
    }) else {
        return;
    };

    if matches!(
        last_conversation_item,
        HistoryItem::UserMessage { .. } | HistoryItem::InternalContinuation { .. }
    ) {
        history.push(HistoryItem::assistant(String::new()));
    }
}

pub(crate) fn restore_latest_model_projection(records: &[TranscriptRecord]) -> Option<String> {
    let mut model = None;
    for record in records {
        match &record.event {
            TranscriptEvent::SessionStarted { model: started } => model = Some(started.clone()),
            TranscriptEvent::ModelChanged { new_model, .. } => model = Some(new_model.clone()),
            _ => {}
        }
    }
    model
}

pub(crate) fn restore_max_turn_id_projection(records: &[TranscriptRecord]) -> u64 {
    records
        .iter()
        .filter_map(|record| match &record.event {
            TranscriptEvent::TurnStarted(event) => Some(event.turn_id),
            TranscriptEvent::ToolExecutionSummary(event) => Some(event.turn_id),
            TranscriptEvent::TurnFinalized(event) => Some(event.turn_id),
            TranscriptEvent::TurnInterrupted { turn_id } => *turn_id,
            _ => None,
        })
        .max()
        .unwrap_or(0)
}

pub(crate) fn project_child_session_summaries(
    child_dir: &Path,
    parent_records: &[TranscriptRecord],
) -> Vec<ChildSessionSummary> {
    let mut children = BTreeMap::new();

    for record in parent_records {
        if let TranscriptEvent::SubagentResult {
            parent_session_id,
            parent_run_id,
            child_session_id,
            agent_name,
            status,
            summary,
            ..
        } = &record.event
            && child_dir.join(format!("{child_session_id}.jsonl")).exists()
        {
            children.insert(
                child_session_id.clone(),
                ChildSessionSummary {
                    parent_session_id: parent_session_id.clone(),
                    parent_run_id: parent_run_id.clone(),
                    child_session_id: child_session_id.clone(),
                    agent_name: agent_name.clone(),
                    status: status.clone(),
                    summary: summary.clone(),
                    timestamp_ms: record.timestamp_ms,
                },
            );
        }
    }

    let mut children = children.into_values().collect::<Vec<_>>();
    children.sort_by(|left, right| {
        left.timestamp_ms
            .cmp(&right.timestamp_ms)
            .then_with(|| left.child_session_id.cmp(&right.child_session_id))
    });
    children
}

pub(crate) fn project_job_board(
    child_dir: &Path,
    parent_records: &[TranscriptRecord],
) -> anyhow::Result<Vec<JobBoardEntry>> {
    let mut jobs = BTreeMap::<String, JobBoardAccumulator>::new();

    for record in parent_records {
        match &record.event {
            TranscriptEvent::SubagentResult {
                run_id,
                child_session_id,
                agent_name,
                status,
                summary,
                ..
            } => {
                let entry = jobs.entry(run_id.clone()).or_default();
                entry.run_id = run_id.clone();
                entry.child_session_id = child_session_id.clone();
                entry.agent_name = agent_name.clone();
                entry.status = status.clone();
                entry.summary = summary.clone();
                entry.terminal = true;
                entry.active = false;
            }
            TranscriptEvent::Evidence {
                source:
                    crate::evidence::EvidenceSource::Subagent {
                        run_id,
                        child_session_id,
                        parent_tool,
                        ..
                    },
                summary,
                detail,
                tags,
                ..
            } => {
                let entry = jobs.entry(run_id.clone()).or_default();
                entry.run_id = run_id.clone();
                if entry.child_session_id.is_empty() {
                    entry.child_session_id = child_session_id.clone();
                }
                if entry.agent_name.is_empty() {
                    entry.agent_name = parent_tool.trim_start_matches("agent__").to_string();
                }
                if tags.iter().any(|tag| tag == "subagent_result") {
                    entry.summary = summary.clone();
                    if let Some(detail) = detail
                        && let Ok(structured) =
                            serde_json::from_str::<StructuredSubagentResult>(detail)
                    {
                        entry.malformed = structured.malformed;
                        entry.structured_status = Some(structured.status.clone());
                        if entry.status.is_empty() {
                            entry.status = structured.status;
                        }
                    }
                }
                if tags
                    .iter()
                    .any(|tag| tag == "subagent_reconciliation" || tag == "reconciled")
                {
                    entry.reconciled = true;
                }
            }
            _ => {}
        }
    }

    if child_dir.exists() {
        for entry in std::fs::read_dir(child_dir)? {
            let path = entry?.path();
            if path.extension().and_then(|ext| ext.to_str()) != Some("jsonl") {
                continue;
            }
            let child_session_id = match path.file_stem().and_then(|stem| stem.to_str()) {
                Some(value) => value.to_string(),
                None => continue,
            };
            let child_records = crate::transcript::read_records(&path)?;
            let latest = child_records
                .iter()
                .rev()
                .find_map(|record| match &record.event {
                    TranscriptEvent::SubagentLifecycle {
                        run_id,
                        agent_name,
                        status,
                        detail,
                        ..
                    } => Some((
                        run_id.clone(),
                        agent_name.clone(),
                        status.clone(),
                        detail.clone(),
                    )),
                    _ => None,
                });
            let Some((run_id, agent_name, status, detail)) = latest else {
                continue;
            };
            if status != "running" {
                continue;
            }
            let job = jobs.entry(run_id.clone()).or_default();
            if job.terminal {
                continue;
            }
            job.run_id = run_id;
            job.child_session_id = child_session_id;
            job.agent_name = agent_name;
            job.status = status;
            job.summary = detail.unwrap_or_else(|| "subagent running".into());
            job.active = true;
        }
    }

    let mut entries = jobs
        .into_values()
        .filter(|entry| !entry.run_id.is_empty())
        .map(|entry| {
            let reconciled = entry.terminal && entry.reconciled;
            let unreconciled = entry.terminal && !entry.reconciled;
            let reusable_eligible = reconciled
                && entry.status == "completed"
                && entry.structured_status.as_deref() == Some("completed")
                && !entry.malformed;
            JobBoardEntry {
                active: entry.active,
                unreconciled,
                reconciled,
                reusable_eligible,
                run_id: entry.run_id,
                child_session_id: entry.child_session_id,
                agent_name: entry.agent_name,
                status: entry.status,
                summary: entry.summary,
            }
        })
        .collect::<Vec<_>>();
    entries.sort_by(|a, b| a.run_id.cmp(&b.run_id));
    Ok(entries)
}

#[derive(Debug, Clone, Default)]
struct JobBoardAccumulator {
    run_id: String,
    child_session_id: String,
    agent_name: String,
    status: String,
    summary: String,
    active: bool,
    terminal: bool,
    reconciled: bool,
    malformed: bool,
    structured_status: Option<String>,
}

#[derive(Debug, Default)]
struct TranscriptTimelineProjection {
    timeline: Timeline,
    current_auto_continue: AutoContinueState,
}

impl TranscriptTimelineProjection {
    fn apply_record(&mut self, record: &TranscriptRecord) {
        match &record.event {
            TranscriptEvent::UserMessage { content } => {
                self.timeline
                    .push_user_message(UserMessageEvent::from_submission(
                        UserMessageSubmission::new(
                            format!("restored-user-message-{}", record.sequence),
                            content.clone(),
                        ),
                    ))
            }
            TranscriptEvent::AssistantMessage { content } => self
                .timeline
                .push_restored_message(MessageRole::Assistant, content.clone()),
            TranscriptEvent::ContextCompaction(event) => {
                self.timeline
                    .push_compaction_separator(COMPACTION_SEPARATOR_LABEL);
                self.timeline
                    .push_restored_message(MessageRole::Assistant, event.summary.clone());
                self.timeline
                    .push_compaction_separator(COMPACTION_SEPARATOR_LABEL);
            }
            TranscriptEvent::ReasoningMessage { content } => {
                self.timeline.push_restored_reasoning(
                    format!("restored-reasoning-{}", record.sequence),
                    content.clone(),
                );
            }
            TranscriptEvent::ContextExperimentReturned {
                branch_id,
                outcome,
                summary,
                next_action,
                had_writes,
                ..
            } => self.timeline.push_restored_message(
                MessageRole::Assistant,
                crate::transcript::format_context_experiment_return(
                    branch_id,
                    outcome,
                    summary,
                    next_action.as_deref(),
                    *had_writes,
                ),
            ),
            TranscriptEvent::ToolCallStarted {
                call_id,
                name,
                args,
            } => {
                self.timeline.push_tool_started(ToolStartedEvent {
                    call_id: call_id.clone(),
                    name: name.clone(),
                    summary: format_tool_call(name, args),
                    arguments: Some(args.to_string()),
                });
            }
            TranscriptEvent::ToolCallFinished {
                call_id,
                name,
                ok,
                output,
            } => {
                self.timeline.push_tool_finished(ToolFinishedEvent {
                    call_id: call_id.clone(),
                    name: name.clone(),
                    summary: restored_tool_summary(name, *ok),
                    outcome: if *ok {
                        ToolOutcome::Success
                    } else {
                        ToolOutcome::Failure
                    },
                    output: serde_json::to_value(output)
                        .ok()
                        .map(|value| value.to_string()),
                });
            }
            TranscriptEvent::ToolCallCancelled { call_id, name } => {
                self.timeline.cancel_tool(call_id, name);
            }
            TranscriptEvent::TodoSnapshot { items } => {
                self.timeline
                    .push_todo_snapshot(TodoSnapshotEvent::new(items.clone()));
                self.timeline
                    .apply_auto_continue_changed(AutoContinueChangedEvent::new(
                        self.current_auto_continue.clone(),
                    ));
            }
            TranscriptEvent::AutoContinueChanged { state } => {
                self.current_auto_continue = state.clone();
                self.timeline
                    .apply_auto_continue_changed(AutoContinueChangedEvent::new(state.clone()));
            }
            TranscriptEvent::PermissionDecision {
                call_id,
                tool,
                args,
                allowed,
                reason,
            } => {
                self.timeline.push_restored_permission_decision(
                    call_id.clone().unwrap_or_else(|| tool.clone()),
                    tool.clone(),
                    format_tool_call(tool, args),
                    Some(args.to_string()),
                    if *allowed {
                        PermissionPromptStatus::Approved
                    } else {
                        PermissionPromptStatus::Denied
                    },
                    reason.clone(),
                );
            }
            TranscriptEvent::Error { message } => {
                self.timeline.push_error(ErrorEvent::new(message.clone()));
            }
            TranscriptEvent::TurnInterrupted { .. } => {
                self.timeline.cancel_active_tools();
                self.timeline.push_notice("Interrupted by user");
            }
            TranscriptEvent::TurnFinalized(event) => {
                if event.outcome == "interrupted" {
                    self.timeline.cancel_active_tools();
                    self.timeline.push_notice("Interrupted by user");
                }
            }
            TranscriptEvent::SubagentResult { .. }
            | TranscriptEvent::SubagentLifecycle { .. }
            | TranscriptEvent::SessionStarted { .. }
            | TranscriptEvent::SessionTitle { .. }
            | TranscriptEvent::ContextBranchCreated { .. }
            | TranscriptEvent::ContextBranchSummary { .. }
            | TranscriptEvent::ContextCheckout { .. }
            | TranscriptEvent::ContextExperimentStarted { .. }
            | TranscriptEvent::ContextNodeCreated { .. }
            | TranscriptEvent::ContextNodeLifecycle { .. }
            | TranscriptEvent::ContextViewOperationMetadata { .. }
            | TranscriptEvent::ContextSummaryArtifactMetadata { .. }
            | TranscriptEvent::FoldedOutputMetadata { .. }
            | TranscriptEvent::TurnStarted(_)
            | TranscriptEvent::ModelChanged { .. }
            | TranscriptEvent::PermissionModeChanged { .. }
            | TranscriptEvent::AutoContinuationScheduled { .. }
            | TranscriptEvent::ValidationAdvisory(_)
            | TranscriptEvent::ToolExecutionSummary(_)
            | TranscriptEvent::Evidence { .. }
            | TranscriptEvent::Unknown => {}
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::ContextCompactionEvent;
    use crate::agent::{ToolExecutionSummaryEvent, TurnFinalizedEvent, TurnStartedEvent};
    use crate::context_tree::ContextNodeStatus;
    use crate::evidence::{EvidenceKind, EvidenceSource};
    use crate::tui::timeline::{TimelineItem, ToolExecutionStatus};
    use crate::user_content::{UserImageAttachment, UserMessageContent};
    use serde_json::json;

    fn record(event: TranscriptEvent) -> TranscriptRecord {
        record_at(1, event)
    }

    fn record_at(sequence: u64, event: TranscriptEvent) -> TranscriptRecord {
        TranscriptRecord {
            session_id: "s".into(),
            sequence,
            timestamp_ms: 0,
            context_branch_id: None,
            event,
        }
    }

    fn branch_record_at(sequence: u64, branch_id: &str, event: TranscriptEvent) -> TranscriptRecord {
        TranscriptRecord {
            session_id: "s".into(),
            sequence,
            timestamp_ms: 0,
            context_branch_id: Some(branch_id.into()),
            event,
        }
    }

    fn metadata_record_at(sequence: u64, event: TranscriptEvent) -> TranscriptRecord {
        TranscriptRecord {
            session_id: "s".into(),
            sequence,
            timestamp_ms: 0,
            context_branch_id: None,
            event,
        }
    }

    #[test]
    fn restored_permissions_are_terminal_not_pending_prompts() {
        let timeline =
            timeline_from_transcript_records(&[record(TranscriptEvent::PermissionDecision {
                call_id: Some("call-1".into()),
                tool: "shell__exec".into(),
                args: json!({"command": "cargo test"}),
                allowed: false,
                reason: Some("Denied by user from TUI permission prompt".into()),
            })]);

        assert!(matches!(
            timeline.items().first(),
            Some(TimelineItem::Permission(permission))
                if permission.status == PermissionPromptStatus::Denied
                    && permission.resolution_reason.as_deref() == Some("Denied by user from TUI permission prompt")
        ));
    }

    #[test]
    fn restored_compaction_uses_separator_summary_separator_shape() {
        let timeline = timeline_from_transcript_records(&[record(
            TranscriptEvent::ContextCompaction(ContextCompactionEvent {
                outcome: "succeeded".into(),
                summary: "Earlier context summary".into(),
                tail_start_index: 5,
                original_history_items: 11,
                retained_history_items: 3,
                detail: None,
            }),
        )]);

        assert_eq!(timeline.items().len(), 3);
        assert!(matches!(timeline.items()[0], TimelineItem::Notice(_)));
        assert!(matches!(timeline.items()[1], TimelineItem::Assistant(_)));
        assert!(matches!(timeline.items()[2], TimelineItem::Notice(_)));
    }

    #[test]
    fn restored_subagent_records_are_ignored_in_timeline_projection() {
        let timeline = timeline_from_transcript_records(&[
            record(TranscriptEvent::SubagentLifecycle {
                run_id: "run-1".into(),
                parent_session_id: "parent".into(),
                parent_run_id: "turn-1".into(),
                agent_name: "explorer".into(),
                status: "running".into(),
                detail: None,
            }),
            TranscriptRecord {
                session_id: "s".into(),
                sequence: 2,
                timestamp_ms: 1,
                context_branch_id: None,
                event: TranscriptEvent::SubagentResult {
                    run_id: "run-1".into(),
                    parent_session_id: "parent".into(),
                    parent_run_id: "turn-1".into(),
                    child_session_id: "child".into(),
                    agent_name: "explorer".into(),
                    status: "completed".into(),
                    summary: "done".into(),
                },
            },
        ]);

        assert!(timeline.items().is_empty());
    }

    #[test]
    fn restored_tool_events_keep_terminal_outcomes_without_live_pending_path() {
        let timeline = timeline_from_transcript_records(&[
            record(TranscriptEvent::ToolCallStarted {
                call_id: "call-1".into(),
                name: "shell__exec".into(),
                args: json!({"command": "sleep 10"}),
            }),
            TranscriptRecord {
                session_id: "s".into(),
                sequence: 2,
                timestamp_ms: 1,
                context_branch_id: None,
                event: TranscriptEvent::ToolCallCancelled {
                    call_id: "call-1".into(),
                    name: "shell__exec".into(),
                },
            },
        ]);

        assert!(matches!(
            timeline.items().first(),
            Some(TimelineItem::Tool(tool)) if tool.status == ToolExecutionStatus::Cancelled
        ));
        assert!(timeline.active_tool().is_none());
    }

    #[test]
    fn restored_user_messages_keep_image_attachments() {
        let timeline = timeline_from_transcript_records(&[record(TranscriptEvent::UserMessage {
            content: UserMessageContent::new(
                "inspect this",
                vec![UserImageAttachment {
                    id: "img-1".into(),
                    label: "screen.png".into(),
                    mime: "image/png".into(),
                    data_url: "data:image/png;base64,AAAA".into(),
                }],
            ),
        })]);

        assert!(matches!(
            timeline.items().first(),
            Some(TimelineItem::User(message))
                if message.text == "inspect this"
                    && message.attachments.len() == 1
                    && message.attachments[0].label == "screen.png"
        ));

        let history = restore_session_history_projection(&[record(TranscriptEvent::UserMessage {
            content: UserMessageContent::new(
                "inspect this",
                vec![UserImageAttachment {
                    id: "img-1".into(),
                    label: "screen.png".into(),
                    mime: "image/png".into(),
                    data_url: "data:image/png;base64,AAAA".into(),
                }],
            ),
        })]);
        assert!(matches!(
            history.first(),
            Some(HistoryItem::UserMessage { content })
                if content.attachments.len() == 1 && content.attachments[0].label == "screen.png"
        ));
    }

    #[test]
    fn replay_context_tree_uses_default_root_for_legacy_transcripts() {
        let tree = replay_context_tree(&[
            record_at(
                1,
                TranscriptEvent::SessionStarted {
                    model: "gpt-5".into(),
                },
            ),
            record_at(
                2,
                TranscriptEvent::UserMessage {
                    content: UserMessageContent::from("hello"),
                },
            ),
        ])
        .expect("replay legacy context tree");

        assert_eq!(tree.root_node_id().as_str(), "root");
        assert_eq!(tree.active_node_id().map(|id| id.as_str()), Some("root"));
        assert_eq!(tree.node_count(), 1);
    }

    #[test]
    fn replay_context_tree_reconstructs_valid_tree() {
        let tree = project_context_tree(&[
            metadata_record_at(
                1,
                TranscriptEvent::ContextNodeCreated {
                    node_id: "child".into(),
                    parent_node_id: Some("root".into()),
                    label: Some("Task branch".into()),
                    purpose: Some("Investigate session-level replay".into()),
                    block_ref: None,
                    source_ref: None,
                },
            ),
            metadata_record_at(
                2,
                TranscriptEvent::ContextNodeLifecycle {
                    node_id: "root".into(),
                    status: ContextNodeStatus::Inactive,
                },
            ),
            metadata_record_at(
                3,
                TranscriptEvent::ContextNodeLifecycle {
                    node_id: "child".into(),
                    status: ContextNodeStatus::Active,
                },
            ),
        ])
        .expect("replay valid context tree");

        let child = tree
            .node(&ContextNodeId::new("child").expect("node id"))
            .expect("child node exists");
        assert_eq!(child.parent_node_id.as_ref().map(|id| id.as_str()), Some("root"));
        assert_eq!(child.label.as_deref(), Some("Task branch"));
        assert_eq!(child.purpose.as_deref(), Some("Investigate session-level replay"));
        assert_eq!(tree.active_node_id().map(|id| id.as_str()), Some("child"));
        assert_eq!(tree.node_count(), 2);
    }

    #[test]
    fn replay_context_tree_rejects_unknown_parent() {
        let error = replay_context_tree(&[metadata_record_at(
            1,
            TranscriptEvent::ContextNodeCreated {
                node_id: "child".into(),
                parent_node_id: Some("missing".into()),
                label: None,
                purpose: None,
                block_ref: None,
                source_ref: None,
            },
        )])
        .expect_err("unknown parent should fail");

        assert!(error.to_string().contains("unknown parent context node 'missing'"));
    }

    #[test]
    fn replay_context_tree_rejects_duplicate_active_node() {
        let error = replay_context_tree(&[
            metadata_record_at(
                1,
                TranscriptEvent::ContextNodeCreated {
                    node_id: "child-a".into(),
                    parent_node_id: Some("root".into()),
                    label: None,
                    purpose: None,
                    block_ref: None,
                    source_ref: None,
                },
            ),
            metadata_record_at(
                2,
                TranscriptEvent::ContextNodeCreated {
                    node_id: "child-b".into(),
                    parent_node_id: Some("root".into()),
                    label: None,
                    purpose: None,
                    block_ref: None,
                    source_ref: None,
                },
            ),
            metadata_record_at(
                3,
                TranscriptEvent::ContextNodeLifecycle {
                    node_id: "root".into(),
                    status: ContextNodeStatus::Inactive,
                },
            ),
            metadata_record_at(
                4,
                TranscriptEvent::ContextNodeLifecycle {
                    node_id: "child-a".into(),
                    status: ContextNodeStatus::Active,
                },
            ),
            metadata_record_at(
                5,
                TranscriptEvent::ContextNodeLifecycle {
                    node_id: "child-b".into(),
                    status: ContextNodeStatus::Active,
                },
            ),
        ])
        .expect_err("duplicate active node should fail");

        assert!(error
            .to_string()
            .contains("cannot activate context node 'child-b' while 'child-a' is active"));
    }

    #[test]
    fn replay_context_tree_rejects_duplicate_node_with_second_parent() {
        let error = replay_context_tree(&[
            metadata_record_at(
                1,
                TranscriptEvent::ContextNodeCreated {
                    node_id: "parent-b".into(),
                    parent_node_id: Some("root".into()),
                    label: None,
                    purpose: None,
                    block_ref: None,
                    source_ref: None,
                },
            ),
            metadata_record_at(
                2,
                TranscriptEvent::ContextNodeCreated {
                    node_id: "child".into(),
                    parent_node_id: Some("root".into()),
                    label: None,
                    purpose: None,
                    block_ref: None,
                    source_ref: None,
                },
            ),
            metadata_record_at(
                3,
                TranscriptEvent::ContextNodeCreated {
                    node_id: "child".into(),
                    parent_node_id: Some("parent-b".into()),
                    label: None,
                    purpose: None,
                    block_ref: None,
                    source_ref: None,
                },
            ),
        ])
        .expect_err("duplicate node should fail");

        assert!(error.to_string().contains("duplicate context node_id 'child'"));
    }

    #[test]
    fn replay_context_tree_rejects_self_parent() {
        let error = replay_context_tree(&[metadata_record_at(
            1,
            TranscriptEvent::ContextNodeCreated {
                node_id: "self".into(),
                parent_node_id: Some("self".into()),
                label: None,
                purpose: None,
                block_ref: None,
                source_ref: None,
            },
        )])
        .expect_err("self parent should fail");

        assert!(error
            .to_string()
            .contains("context node 'self' cannot be its own parent"));
    }

    #[test]
    fn default_cursor_preserves_current_behavior() {
        let records = vec![
            record_at(
                1,
                TranscriptEvent::SessionStarted {
                    model: "gpt-5".into(),
                },
            ),
            record_at(
                2,
                TranscriptEvent::UserMessage {
                    content: UserMessageContent::from("hello"),
                },
            ),
            record_at(
                3,
                TranscriptEvent::AssistantMessage {
                    content: "hi".into(),
                },
            ),
        ];

        let expected = project_session_restore_snapshot("s".into(), records.clone(), None)
            .expect("default snapshot");
        let actual = build_session_context_snapshot(
            "s".into(),
            records,
            None,
            SessionContextCursor {
                branch_id: None,
                leaf_sequence: None,
            },
        )
        .expect("cursor snapshot");

        assert_eq!(actual.branch_id, ROOT_CONTEXT_BRANCH_ID);
        assert_eq!(actual.leaf_sequence, 3);
        assert_eq!(actual.records.len(), expected.records.len());
        assert_eq!(format!("{:?}", actual.messages), format!("{:?}", expected.messages));
        assert_eq!(actual.history, expected.history);
        assert_eq!(actual.evidence, expected.evidence);
        assert_eq!(actual.latest_model, expected.latest_model);
        assert_eq!(actual.max_turn_id, expected.max_turn_id);
    }

    #[test]
    fn explicit_leaf_truncates_future_records() {
        let records = vec![
            record_at(
                1,
                TranscriptEvent::UserMessage {
                    content: UserMessageContent::from("before"),
                },
            ),
            record_at(
                2,
                TranscriptEvent::AssistantMessage {
                    content: "visible".into(),
                },
            ),
            record_at(
                3,
                TranscriptEvent::AssistantMessage {
                    content: "hidden".into(),
                },
            ),
        ];

        let snapshot = build_session_context_snapshot(
            "s".into(),
            records,
            None,
            SessionContextCursor {
                branch_id: None,
                leaf_sequence: Some(2),
            },
        )
        .expect("snapshot truncated at leaf");

        assert_eq!(snapshot.leaf_sequence, 2);
        assert_eq!(snapshot.records.len(), 2);
        assert!(
            snapshot
                .messages
                .iter()
                .all(|message| message.content != "hidden")
        );
        assert!(matches!(
            snapshot.history.as_slice(),
            [HistoryItem::UserMessage { .. }, HistoryItem::AssistantText { text }] if text == "visible"
        ));
    }

    #[test]
    fn compaction_before_leaf_still_restores_context_summary() {
        let records = vec![
            record_at(
                1,
                TranscriptEvent::UserMessage {
                    content: UserMessageContent::from("before"),
                },
            ),
            record_at(
                2,
                TranscriptEvent::AssistantMessage {
                    content: "reply".into(),
                },
            ),
            record_at(
                3,
                TranscriptEvent::ContextCompaction(ContextCompactionEvent {
                    outcome: "succeeded".into(),
                    summary: "summary".into(),
                    tail_start_index: 1,
                    original_history_items: 2,
                    retained_history_items: 1,
                    detail: None,
                }),
            ),
            record_at(
                4,
                TranscriptEvent::AssistantMessage {
                    content: "after".into(),
                },
            ),
        ];

        let snapshot = build_session_context_snapshot(
            "s".into(),
            records,
            None,
            SessionContextCursor {
                branch_id: None,
                leaf_sequence: Some(4),
            },
        )
        .expect("snapshot with compaction");

        assert!(matches!(
            snapshot.history.first(),
            Some(HistoryItem::ContextSummary { text }) if text == "summary"
        ));
    }

    #[test]
    fn leaf_beyond_max_sequence_returns_error() {
        let records = vec![record_at(
            2,
            TranscriptEvent::UserMessage {
                content: UserMessageContent::from("hello"),
            },
        )];

        let error = build_session_context_snapshot(
            "s".into(),
            records,
            None,
            SessionContextCursor {
                branch_id: None,
                leaf_sequence: Some(3),
            },
        )
        .expect_err("leaf past max sequence should fail");

        assert!(
            error
                .to_string()
                .contains("leaf_sequence 3 exceeds max transcript sequence 2")
        );
    }

    #[test]
    fn max_turn_id_respects_leaf_cut() {
        let records = vec![
            record_at(
                1,
                TranscriptEvent::TurnStarted(TurnStartedEvent {
                    turn_id: 1,
                    intent: "task".into(),
                    directive: "do it".into(),
                    validation_reminder: String::new(),
                }),
            ),
            record_at(
                2,
                TranscriptEvent::ToolExecutionSummary(ToolExecutionSummaryEvent {
                    turn_id: 1,
                    call_id: "call-1".into(),
                    name: "shell__exec".into(),
                    status: "completed".into(),
                    rejection: None,
                    effect_kind: "read".into(),
                    primary_path: None,
                    command: None,
                }),
            ),
            record_at(
                3,
                TranscriptEvent::TurnFinalized(TurnFinalizedEvent {
                    turn_id: 1,
                    outcome: "completed".into(),
                    tool_call_count: 1,
                    continuation_count: 0,
                    write_effects: 0,
                    validation_effects: 0,
                    failed_validation_effects: 0,
                    validation_advisory_emitted: false,
                }),
            ),
            record_at(
                4,
                TranscriptEvent::TurnStarted(TurnStartedEvent {
                    turn_id: 7,
                    intent: "future".into(),
                    directive: "later".into(),
                    validation_reminder: String::new(),
                }),
            ),
            record_at(5, TranscriptEvent::TurnInterrupted { turn_id: Some(7) }),
        ];

        let snapshot = build_session_context_snapshot(
            "s".into(),
            records,
            None,
            SessionContextCursor {
                branch_id: None,
                leaf_sequence: Some(3),
            },
        )
        .expect("snapshot before future turn");

        assert_eq!(snapshot.max_turn_id, 1);
    }

    #[test]
    fn evidence_respects_leaf_cut() {
        let records = vec![
            record_at(
                1,
                TranscriptEvent::Evidence {
                    id: "ev-1".into(),
                    evidence_kind: EvidenceKind::Decision,
                    title: "one".into(),
                    summary: "visible".into(),
                    detail: None,
                    source: EvidenceSource::Transcript { sequence: 1 },
                    tags: vec![],
                },
            ),
            record_at(
                2,
                TranscriptEvent::Evidence {
                    id: "ev-2".into(),
                    evidence_kind: EvidenceKind::Decision,
                    title: "two".into(),
                    summary: "hidden".into(),
                    detail: None,
                    source: EvidenceSource::Transcript { sequence: 2 },
                    tags: vec![],
                },
            ),
        ];

        let snapshot = build_session_context_snapshot(
            "s".into(),
            records,
            None,
            SessionContextCursor {
                branch_id: None,
                leaf_sequence: Some(1),
            },
        )
        .expect("snapshot with evidence leaf");

        assert_eq!(snapshot.evidence.len(), 1);
        assert_eq!(snapshot.evidence[0].summary, "visible");
    }

    #[test]
    fn old_transcript_default_restore_still_matches_linear_behavior() {
        let records = vec![
            record_at(
                1,
                TranscriptEvent::SessionStarted {
                    model: "gpt-5".into(),
                },
            ),
            record_at(
                2,
                TranscriptEvent::UserMessage {
                    content: UserMessageContent::from("hello"),
                },
            ),
            record_at(
                3,
                TranscriptEvent::AssistantMessage {
                    content: "hi".into(),
                },
            ),
        ];

        let snapshot = project_session_restore_snapshot("s".into(), records, None)
            .expect("linear snapshot");

        assert_eq!(snapshot.branch_id, ROOT_CONTEXT_BRANCH_ID);
        assert_eq!(snapshot.leaf_sequence, 3);
        assert!(matches!(
            snapshot.history.as_slice(),
            [HistoryItem::UserMessage { .. }, HistoryItem::AssistantText { text }] if text == "hi"
        ));
    }

    #[test]
    fn explicit_branch_inherits_parent_prefix_and_excludes_parent_after_fork_base() {
        let records = vec![
            record_at(
                1,
                TranscriptEvent::UserMessage {
                    content: UserMessageContent::from("root-before"),
                },
            ),
            record_at(
                2,
                TranscriptEvent::AssistantMessage {
                    content: "root-at-fork".into(),
                },
            ),
            record_at(
                3,
                TranscriptEvent::ContextBranchCreated {
                    branch_id: "feature".into(),
                    parent_branch_id: ROOT_CONTEXT_BRANCH_ID.into(),
                    base_sequence: 2,
                    label: Some("feature".into()),
                },
            ),
            record_at(
                4,
                TranscriptEvent::AssistantMessage {
                    content: "root-after-fork".into(),
                },
            ),
            branch_record_at(
                5,
                "feature",
                TranscriptEvent::UserMessage {
                    content: UserMessageContent::from("child-only"),
                },
            ),
        ];

        let snapshot = build_session_context_snapshot(
            "s".into(),
            records,
            None,
            SessionContextCursor {
                branch_id: Some("feature".into()),
                leaf_sequence: None,
            },
        )
        .expect("branch snapshot");

        assert_eq!(snapshot.branch_id, "feature");
        assert_eq!(
            snapshot
                .records
                .iter()
                .map(|record| record.sequence)
                .collect::<Vec<_>>(),
            vec![1, 2, 5]
        );
        assert!(snapshot
            .messages
            .iter()
            .all(|message| message.content != "root-after-fork"));
    }

    #[test]
    fn latest_context_checkout_affects_default_branch_selection() {
        let records = vec![
            record_at(
                1,
                TranscriptEvent::UserMessage {
                    content: UserMessageContent::from("root-before"),
                },
            ),
            record_at(
                2,
                TranscriptEvent::ContextBranchCreated {
                    branch_id: "feature".into(),
                    parent_branch_id: ROOT_CONTEXT_BRANCH_ID.into(),
                    base_sequence: 1,
                    label: None,
                },
            ),
            branch_record_at(
                3,
                "feature",
                TranscriptEvent::AssistantMessage {
                    content: "branch-visible".into(),
                },
            ),
            record_at(
                4,
                TranscriptEvent::ContextCheckout {
                    branch_id: "feature".into(),
                    leaf_sequence: 3,
                },
            ),
            record_at(
                5,
                TranscriptEvent::AssistantMessage {
                    content: "root-later".into(),
                },
            ),
        ];

        let snapshot = project_session_restore_snapshot("s".into(), records, None)
            .expect("default restore uses latest checkout");

        assert_eq!(snapshot.branch_id, "feature");
        assert_eq!(snapshot.leaf_sequence, 3);
        assert_eq!(
            snapshot
                .records
                .iter()
                .map(|record| record.sequence)
                .collect::<Vec<_>>(),
            vec![1, 3]
        );
    }

    #[test]
    fn invalid_branch_resolution_errors_fail_fast() {
        let unknown_branch_error = build_session_context_snapshot(
            "s".into(),
            vec![record_at(
                1,
                TranscriptEvent::UserMessage {
                    content: UserMessageContent::from("hello"),
                },
            )],
            None,
            SessionContextCursor {
                branch_id: Some("missing".into()),
                leaf_sequence: None,
            },
        )
        .expect_err("unknown branch should fail");
        assert!(unknown_branch_error
            .to_string()
            .contains("unknown context branch 'missing'"));

        let invalid_base_error = build_session_context_snapshot(
            "s".into(),
            vec![
                record_at(
                    1,
                    TranscriptEvent::UserMessage {
                        content: UserMessageContent::from("root"),
                    },
                ),
                record_at(
                    2,
                    TranscriptEvent::ContextBranchCreated {
                        branch_id: "feature".into(),
                        parent_branch_id: ROOT_CONTEXT_BRANCH_ID.into(),
                        base_sequence: 9,
                        label: None,
                    },
                ),
            ],
            None,
            SessionContextCursor {
                branch_id: None,
                leaf_sequence: None,
            },
        )
        .expect_err("invalid base should fail");
        assert!(invalid_base_error
            .to_string()
            .contains("base_sequence 9 is not resolvable"));

        let leaf_beyond_tip_error = build_session_context_snapshot(
            "s".into(),
            vec![
                record_at(
                    1,
                    TranscriptEvent::UserMessage {
                        content: UserMessageContent::from("root"),
                    },
                ),
                record_at(
                    2,
                    TranscriptEvent::ContextBranchCreated {
                        branch_id: "feature".into(),
                        parent_branch_id: ROOT_CONTEXT_BRANCH_ID.into(),
                        base_sequence: 1,
                        label: None,
                    },
                ),
                branch_record_at(
                    3,
                    "feature",
                    TranscriptEvent::AssistantMessage {
                        content: "child".into(),
                    },
                ),
                record_at(
                    5,
                    TranscriptEvent::AssistantMessage {
                        content: "root-later".into(),
                    },
                ),
            ],
            None,
            SessionContextCursor {
                branch_id: Some("feature".into()),
                leaf_sequence: Some(4),
            },
        )
        .expect_err("leaf beyond tip should fail");
        assert!(leaf_beyond_tip_error
            .to_string()
            .contains("requested leaf_sequence 4 exceeds tip 3 for branch 'feature'"));
    }

    #[test]
    fn branch_local_compaction_replay_stays_branch_local() {
        let records = vec![
            record_at(
                1,
                TranscriptEvent::UserMessage {
                    content: UserMessageContent::from("root-a"),
                },
            ),
            record_at(
                2,
                TranscriptEvent::AssistantMessage {
                    content: "root-b".into(),
                },
            ),
            record_at(
                3,
                TranscriptEvent::ContextBranchCreated {
                    branch_id: "feature".into(),
                    parent_branch_id: ROOT_CONTEXT_BRANCH_ID.into(),
                    base_sequence: 2,
                    label: None,
                },
            ),
            branch_record_at(
                4,
                "feature",
                TranscriptEvent::UserMessage {
                    content: UserMessageContent::from("child-a"),
                },
            ),
            branch_record_at(
                5,
                "feature",
                TranscriptEvent::AssistantMessage {
                    content: "child-b".into(),
                },
            ),
            branch_record_at(
                6,
                "feature",
                TranscriptEvent::ContextCompaction(ContextCompactionEvent {
                    outcome: "succeeded".into(),
                    summary: "child-summary".into(),
                    tail_start_index: 2,
                    original_history_items: 4,
                    retained_history_items: 2,
                    detail: None,
                }),
            ),
            record_at(
                7,
                TranscriptEvent::AssistantMessage {
                    content: "root-after-fork".into(),
                },
            ),
        ];

        let snapshot = build_session_context_snapshot(
            "s".into(),
            records,
            None,
            SessionContextCursor {
                branch_id: Some("feature".into()),
                leaf_sequence: None,
            },
        )
        .expect("branch compaction snapshot");

        assert!(matches!(
            snapshot.history.as_slice(),
            [HistoryItem::ContextSummary { text }, HistoryItem::UserMessage { content }, HistoryItem::AssistantText { text: child_text }]
                if text == "child-summary" && content.display_text() == "child-a" && child_text == "child-b"
        ));
    }

    #[test]
    fn list_context_branches_marks_current_branch_and_labels() {
        let records = vec![
            record_at(
                1,
                TranscriptEvent::UserMessage {
                    content: UserMessageContent::from("root"),
                },
            ),
            record_at(
                2,
                TranscriptEvent::ContextBranchCreated {
                    branch_id: "feature-a".into(),
                    parent_branch_id: ROOT_CONTEXT_BRANCH_ID.into(),
                    base_sequence: 1,
                    label: Some("Feature A".into()),
                },
            ),
            branch_record_at(
                3,
                "feature-a",
                TranscriptEvent::AssistantMessage {
                    content: "child".into(),
                },
            ),
        ];

        let branches = list_context_branches(&records, Some("feature-a")).expect("branches");

        assert_eq!(branches.len(), 2);
        assert_eq!(branches[0].branch_id, ROOT_CONTEXT_BRANCH_ID);
        assert_eq!(branches[1].branch_id, "feature-a");
        assert_eq!(branches[1].label.as_deref(), Some("Feature A"));
        assert_eq!(branches[1].tip_sequence, 3);
        assert!(branches[1].is_current);
        assert!(!branches[0].is_current);
    }
}
