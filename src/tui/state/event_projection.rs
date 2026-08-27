//! SessionEvent → TuiState timeline/toast projection helpers.
//! Split out of `state` to keep the view-model file navigable.

use super::*;

pub(super) struct EventProjection<'a> {
    pub(super) active_session: &'a mut bool,
    pub(super) latest_auto_continue: &'a mut AutoContinueState,
    pub(super) latest_todo: &'a mut Option<TodoView>,
    pub(super) retry: &'a mut Option<RetryNoticeState>,
    pub(super) phase: &'a mut AppPhase,
    pub(super) active_tool_call_id: &'a mut Option<String>,
    pub(super) pending_permission: &'a mut Option<PermissionView>,
    pub(super) model_token_usage: &'a mut Option<ModelTokenUsage>,
    pub(super) compaction_active: &'a mut bool,
    pub(super) compaction_animation_start_frame: &'a mut usize,
    pub(super) ignore_late_tool_events: &'a mut bool,
    pub(super) quit_requested: &'a mut bool,
    pub(super) status_spinner_frame: &'a mut usize,
    pub(super) toast: &'a mut Option<ToastState>,
    pub(super) timeline: &'a mut Timeline,
    pub(super) accepts_tool_events: bool,
}

impl<'a> EventProjection<'a> {
    pub(super) fn with_tool_event_acceptance(mut self, accepts_tool_events: bool) -> Self {
        self.accepts_tool_events = accepts_tool_events;
        self
    }
}

pub(super) fn apply_projected_session_event(projection: EventProjection<'_>, event: SessionEvent) {
    match event {
        SessionEvent::Tick => {
            *projection.status_spinner_frame = projection.status_spinner_frame.wrapping_add(1);
            projection
                .timeline
                .tick_reasoning_elapsed(std::time::Instant::now());
            if let Some(retry) = projection.retry.as_mut() {
                // Sticky red countdown for the whole backoff window.
                retry.tick_frame();
                *projection.toast = Some(retry.sticky_toast());
            } else if projection.toast.as_mut().is_some_and(ToastState::tick) {
                *projection.toast = None;
            }
        }
        SessionEvent::UserMessage(message) => {
            *projection.active_session = true;
            projection.timeline.push_user_message(message);
            *projection.latest_auto_continue = AutoContinueState::default();
            *projection.latest_todo = None;
            *projection.retry = None;
            *projection.phase = AppPhase::Running;
            *projection.active_tool_call_id = None;
            *projection.pending_permission = None;
            *projection.ignore_late_tool_events = false;
        }
        SessionEvent::RetryScheduled(retry) => {
            *projection.phase = AppPhase::Running;
            let notice = RetryNoticeState::from_lifecycle(retry);
            *projection.toast = Some(notice.sticky_toast());
            *projection.retry = Some(notice);
        }
        SessionEvent::RetryStarted(_) => {
            *projection.retry = None;
            *projection.toast = None;
        }
        SessionEvent::ReasoningDelta(reasoning) => {
            *projection.phase = AppPhase::Running;
            projection.timeline.push_reasoning_delta(reasoning);
        }
        SessionEvent::ReasoningDone(reasoning) => {
            projection.timeline.finalize_reasoning(reasoning);
        }
        SessionEvent::AssistantDelta(delta) => {
            *projection.phase = AppPhase::Running;
            projection.timeline.push_assistant_delta(delta);
        }
        SessionEvent::AssistantDone { message_id } => {
            projection
                .timeline
                .finalize_assistant_message(message_id.as_deref());
        }
        SessionEvent::TokenUsage(usage) => {
            *projection.compaction_active = false;
            *projection.model_token_usage = Some(ModelTokenUsage::from(usage));
        }
        SessionEvent::ToolPending(tool) => {
            // Close any open assistant stream before tool cards so later
            // multi-iteration assistant text creates a new bubble after tools.
            projection.timeline.finalize_all_assistant_messages();
            if projection.accepts_tool_events
                && tool.name != crate::tool_names::TOOL_AGENT_WAIT
                && projection.timeline.push_tool_pending(tool.clone())
            {
                *projection.active_tool_call_id = Some(tool.call_id.clone());
                *projection.phase = AppPhase::Running;
            }
        }
        SessionEvent::ToolCancelled(tool) => {
            if projection.accepts_tool_events {
                if tool.name == crate::tool_names::TOOL_AGENT_WAIT
                    && projection
                        .timeline
                        .cancel_foreground_subagent_wait(&tool.call_id)
                {
                } else {
                    projection.timeline.cancel_tool(&tool.call_id, &tool.name);
                }
                if projection.active_tool_call_id.as_deref() == Some(tool.call_id.as_str()) {
                    *projection.active_tool_call_id = None;
                }
            }
        }
        SessionEvent::ToolStarted(tool) => {
            // ToolStarted may arrive without a prior pending event for some
            // protocols; still seal open assistant streams first.
            projection.timeline.finalize_all_assistant_messages();
            let started = if projection.accepts_tool_events
                && tool.name == crate::tool_names::TOOL_AGENT_WAIT
            {
                tool.arguments
                    .as_deref()
                    .and_then(|arguments| serde_json::from_str::<serde_json::Value>(arguments).ok())
                    .and_then(|args| {
                        args.get("run_id")
                            .and_then(serde_json::Value::as_str)
                            .map(str::to_string)
                    })
                    .is_some_and(|run_id| {
                        projection
                            .timeline
                            .begin_subagent_wait(&tool.call_id, &run_id)
                    })
                    || projection.timeline.push_tool_started(tool.clone())
            } else {
                projection.accepts_tool_events
                    && projection.timeline.push_tool_started(tool.clone())
            };
            if started {
                *projection.active_tool_call_id = Some(tool.call_id.clone());
                *projection.phase = AppPhase::Running;
            }
        }
        SessionEvent::ToolFinished(tool) => {
            if projection.accepts_tool_events
                && projection.timeline.push_tool_finished(tool.clone())
                && projection.active_tool_call_id.as_deref() == Some(tool.call_id.as_str())
            {
                *projection.active_tool_call_id = None;
            }
        }
        SessionEvent::ToolOutputDelta(delta) => {
            if projection.accepts_tool_events {
                projection.timeline.push_tool_output_delta(delta);
            }
        }
        SessionEvent::TodoSnapshot(todo) => {
            let auto_continue = projection.latest_auto_continue.clone();
            *projection.latest_todo = Some(TodoView {
                items: todo.items.clone(),
                auto_continue: auto_continue.clone(),
            });
            projection.timeline.push_todo_snapshot(todo);
            projection
                .timeline
                .apply_auto_continue_changed(AutoContinueChangedEvent::new(auto_continue));
        }
        SessionEvent::AutoContinueChanged(event) => {
            *projection.latest_auto_continue = event.state.clone();
            if let Some(todo) = projection.latest_todo.as_mut() {
                todo.auto_continue = event.state.clone();
                projection.timeline.apply_auto_continue_changed(event);
            }
        }
        SessionEvent::Notice(notice) => {
            let kind = match notice.kind {
                crate::tui::events::NoticeKind::Info => ToastKind::Info,
                crate::tui::events::NoticeKind::Success => ToastKind::Success,
                crate::tui::events::NoticeKind::RecoverableError => ToastKind::Error,
            };
            *projection.toast = Some(ToastState::new(
                notice.message,
                kind,
                ToastState::DEFAULT_TICKS,
            ));
        }
        SessionEvent::CompactionStarted => {
            // 指示条转为往返扫描动画；清掉过期的 token 数字，等压缩后新用量到达再恢复。
            *projection.model_token_usage = None;
            *projection.compaction_animation_start_frame = *projection.status_spinner_frame;
            *projection.compaction_active = true;
            projection.timeline.start_compaction();
        }
        SessionEvent::CompactionPreviewDelta { delta } => {
            projection.timeline.append_compaction_preview(&delta)
        }
        SessionEvent::CompactionCommitted { summary } => {
            *projection.compaction_active = false;
            match summary {
                Some(summary) => projection.timeline.commit_compaction_with_summary(summary),
                None => projection.timeline.finish_compaction(true),
            }
        }
        SessionEvent::CompactionNoProgress { blockers } => {
            *projection.compaction_active = false;
            projection.timeline.finish_compaction(false);
            let _ = blockers;
            *projection.toast = Some(ToastState::new(
                "Context limit reached; earlier context cannot be compacted safely yet.",
                ToastKind::Error,
                ToastState::DEFAULT_TICKS,
            ));
        }
        SessionEvent::CompactionFailed => {
            *projection.compaction_active = false;
            projection.timeline.finish_compaction(false);
            *projection.toast = Some(ToastState::new(
                "Context compaction failed",
                ToastKind::Error,
                ToastState::DEFAULT_TICKS,
            ));
        }
        SessionEvent::ProcessIssue(issue) => {
            *projection.phase = AppPhase::Running;
            *projection.toast = Some(ToastState::new(
                issue.message,
                ToastKind::Error,
                ToastState::DEFAULT_TICKS,
            ));
        }
        SessionEvent::Interrupted => {
            projection
                .timeline
                .seal_active_reasoning(std::time::Instant::now());
            *projection.retry = None;
            *projection.compaction_active = false;
            projection.timeline.finish_compaction(false);
            *projection.phase = AppPhase::Completed;
            *projection.active_tool_call_id = None;
            *projection.pending_permission = None;
            *projection.latest_auto_continue = AutoContinueState::default();
            *projection.latest_todo = None;
            *projection.ignore_late_tool_events = true;
            projection.timeline.cancel_foreground_subagent_waits();
            projection.timeline.cancel_active_tools();
            *projection.toast = Some(ToastState::new(
                "Interrupted by user",
                ToastKind::Info,
                ToastState::DEFAULT_TICKS,
            ));
        }
        SessionEvent::Error(error) => {
            projection
                .timeline
                .seal_active_reasoning(std::time::Instant::now());
            let had_retry = projection.retry.take().is_some();
            *projection.compaction_active = false;
            projection.timeline.finish_compaction(false);
            *projection.phase = AppPhase::Error;
            *projection.active_tool_call_id = None;
            *projection.pending_permission = None;
            *projection.latest_auto_continue = AutoContinueState::default();
            *projection.latest_todo = None;
            if had_retry {
                // Drop the sticky retry toast so it cannot outlive the failed turn.
                *projection.toast = None;
            }
            projection.timeline.push_error(error);
        }
        SessionEvent::Done => {
            projection
                .timeline
                .seal_active_reasoning(std::time::Instant::now());
            let had_retry = projection.retry.take().is_some();
            *projection.compaction_active = false;
            projection.timeline.finish_compaction(false);
            *projection.phase = AppPhase::Completed;
            *projection.active_tool_call_id = None;
            *projection.pending_permission = None;
            if had_retry {
                *projection.toast = None;
            }
        }
        SessionEvent::PermissionResolved(resolution) => {
            projection.timeline.resolve_permission(resolution);
        }
        SessionEvent::RuntimeContextUpdated(_)
        | SessionEvent::ContextTreeUpdated(_)
        | SessionEvent::ContextViewUpdated(_)
        | SessionEvent::ContextDetailOpened(_)
        | SessionEvent::ContextSummaryUpdated(_)
        | SessionEvent::SessionStarted { .. }
        | SessionEvent::SessionResumed { .. }
        | SessionEvent::SessionTokenUsage(_)
        | SessionEvent::ContextBranchChanged { .. }
        | SessionEvent::ToolBatchFinished => {}
        SessionEvent::Quit => {
            *projection.phase = AppPhase::Quitting;
            *projection.quit_requested = true;
        }
        SessionEvent::PermissionRequested(_) => {}
    }
}

pub(super) fn child_event_projection_payload(
    pending_permission: Option<&PermissionView>,
    event: &SessionEvent,
) -> Option<(String, String)> {
    match event {
        SessionEvent::ToolPending(tool) => Some((
            "preparing".into(),
            compact_child_projection_text(&format!("{} preparing input", tool.name)),
        )),
        SessionEvent::ToolCancelled(tool) => Some((
            "cancelled".into(),
            compact_child_projection_text(&format!("{} cancelled", tool.name)),
        )),
        SessionEvent::ToolStarted(tool) => Some((
            "running".into(),
            compact_child_projection_text(&child_tool_projection_summary(
                &tool.name,
                &tool.summary,
            )),
        )),
        SessionEvent::ToolFinished(tool) => Some((
            match tool.outcome {
                ToolOutcome::Success => "completed",
                ToolOutcome::Failure => "failed",
            }
            .into(),
            compact_child_projection_text(&child_tool_projection_summary(
                &tool.name,
                &tool.summary,
            )),
        )),
        SessionEvent::PermissionRequested(request) => Some((
            "approval".into(),
            compact_child_projection_text(&format!(
                "approval needed · {}",
                child_tool_projection_summary(&request.tool_name, &request.summary)
            )),
        )),
        SessionEvent::PermissionResolved(resolution) => {
            let subject = pending_permission
                .filter(|permission| permission.call_id == resolution.call_id)
                .map(|permission| {
                    child_tool_projection_summary(&permission.tool_name, &permission.summary)
                })
                .unwrap_or_else(|| "permission request".into());
            let status = match resolution.decision {
                PermissionDecision::Approved => "approved",
                PermissionDecision::Denied => "denied",
            };
            let summary = resolution
                .reason
                .as_deref()
                .map(str::trim)
                .filter(|reason| !reason.is_empty())
                .map(|reason| format!("approval {status} · {subject} · {reason}"))
                .unwrap_or_else(|| format!("approval {status} · {subject}"));
            Some((status.into(), compact_child_projection_text(&summary)))
        }
        SessionEvent::Error(error) => Some((
            "error".into(),
            compact_child_projection_text(&error.message),
        )),
        SessionEvent::Interrupted => {
            Some(("interrupted".into(), "child session interrupted".into()))
        }
        _ => None,
    }
}

pub(super) fn child_tool_projection_summary(name: &str, summary: &str) -> String {
    let summary = summary.trim();
    if summary.is_empty() {
        return name.to_string();
    }
    if summary.starts_with(name) {
        return summary.to_string();
    }
    format!("{name} — {summary}")
}

pub(super) fn compact_child_projection_text(text: &str) -> String {
    let single_line = text.split_whitespace().collect::<Vec<_>>().join(" ");
    let limit = 160;
    if single_line.chars().count() <= limit {
        return single_line;
    }

    let mut truncated = single_line.chars().take(limit).collect::<String>();
    truncated.push('…');
    truncated
}

#[cfg(test)]
pub(super) fn project_child_timeline_state(
    records: &[TranscriptRecord],
) -> Result<ChildTranscriptState> {
    Ok(ChildTranscriptState {
        session_id: records
            .first()
            .map(|record| record.session_id.clone())
            .unwrap_or_default(),
        timeline: Timeline::from_transcript_records(records),
        model: child_transcript_model(records),
        record_count: records.len(),
        snapshot_loaded: true,
        snapshot_dirty: false,
        context: project_context_pane(records)?,
        active_session: true,
        latest_auto_continue: AutoContinueState::default(),
        latest_todo: None,
        retry: None,
        phase: AppPhase::Completed,
        active_tool_call_id: None,
        pending_permission: None,
        model_token_usage: None,
        output_token_rate: None,
        compaction_active: false,
        compaction_animation_start_frame: 0,
        ignore_late_tool_events: false,
    })
}

#[cfg(test)]
pub(super) fn project_context_pane(records: &[TranscriptRecord]) -> Result<ContextPaneState> {
    let tree = transcript_projection::project_context_tree(records)?;
    let view = transcript_projection::project_context_view(records)?;
    let open_detail = view
        .view_state
        .open_detail_block_id()
        .map(|block_id| ContextDetailTarget::Block(block_id.as_str().to_string()));
    Ok(ContextPaneState {
        tree,
        view,
        runtime_context: None,
        open_detail,
    })
}

pub(super) fn context_update_is_accepted(
    pane: &ContextPaneState,
    update: &crate::tui::events::RuntimeContextUpdatedEvent,
) -> bool {
    let Some(cached) = &pane.runtime_context else {
        return true;
    };
    match update.disposition {
        crate::tui::events::RuntimeContextDisposition::ReplaceScope => {
            cached.session_id != update.context.session_id
                || update.context.context_scope_revision > cached.context_scope_revision
        }
        crate::tui::events::RuntimeContextDisposition::Advance => {
            cached.session_id == update.context.session_id
                && cached.active_context.branch_id == update.context.active_context.branch_id
                && cached.context_scope_revision == update.context.context_scope_revision
                && update.context.leaf_sequence >= cached.leaf_sequence
        }
    }
}

pub(super) fn apply_runtime_context(
    pane: &mut ContextPaneState,
    context: RuntimeActiveContext,
    disposition: crate::tui::events::RuntimeContextDisposition,
) {
    if disposition == crate::tui::events::RuntimeContextDisposition::ReplaceScope {
        // Inspection is local to a scope. Clear it before the new projection is
        // installed so a coincidental ID cannot carry it across a checkout.
        pane.open_detail = None;
    }
    pane.tree = context.context_tree.clone();
    pane.view = context.context_view.clone();
    pane.runtime_context = Some(context);
    if pane
        .open_detail
        .as_ref()
        .is_some_and(|target| !context_detail_target_exists(pane, target))
    {
        pane.open_detail = None;
    }
    // Provider detail initializes local inspection only after a replacement,
    // or when an advancing projection has no active local selection.
    if pane.open_detail.is_none() {
        pane.open_detail = pane
            .runtime_context
            .as_ref()
            .and_then(|payload| payload.active_context.open_detail_block_id.clone())
            .map(ContextDetailTarget::Block)
            .filter(|target| context_detail_target_exists(pane, target));
    }
}

pub(crate) fn context_dialog_items(context: &ContextPaneState) -> Vec<DialogItem> {
    let mut items = Vec::new();
    for node in context.tree.nodes() {
        if node.node_id == *context.tree.root_node_id() {
            continue;
        }
        let depth = context_node_depth(&context.tree, node.node_id.as_str());
        let indent = if depth == 0 {
            String::new()
        } else {
            format!("{}↳ ", "  ".repeat(depth.saturating_sub(1)))
        };
        items.push(
            DialogItem::new(
                format!("node:{}", node.node_id.as_str()),
                format!(
                    "{indent}{}",
                    node.label
                        .clone()
                        .unwrap_or_else(|| node.node_id.as_str().to_string())
                ),
                node.purpose.clone(),
            )
            .with_section("Nodes"),
        );
    }
    for (_, block) in context.view.provider_active_blocks() {
        items.push(
            DialogItem::new(
                format!("block:{}", block.block_id.as_str()),
                block.title.clone(),
                None,
            )
            .with_section("Blocks"),
        );
    }
    for artifact in &context.view.summary_artifacts {
        if context.view.provider_active_blocks().iter().any(|(_, block)| matches!(&block.source, ContextBlockSource::SummaryArtifact { artifact_id } if artifact_id == &artifact.artifact_id)) {
            items.push(DialogItem::new(format!("summary:{}", artifact.artifact_id), format!("Summary {}", artifact.artifact_id), Some(artifact.node_id.clone())).with_section("Summaries"));
        }
    }
    items
}

pub(super) fn rebuild_active_context_picker(
    state: &mut TuiState,
    disposition: crate::tui::events::RuntimeContextDisposition,
) {
    let items = context_dialog_items(state.active_context());
    let Some(dialog) = state
        .dialog
        .as_mut()
        .filter(|dialog| dialog.kind == DialogKind::ContextPicker)
    else {
        return;
    };
    let preserve = disposition == crate::tui::events::RuntimeContextDisposition::Advance;
    let selected_id = preserve
        .then(|| dialog.selected_item().map(|item| item.id.clone()))
        .flatten();
    let query = if preserve {
        dialog.query.clone()
    } else {
        Default::default()
    };
    let detail_focused = preserve && dialog.detail_focused;
    let detail_scroll = if preserve { dialog.detail_scroll } else { 0 };
    dialog.items = items;
    dialog.query = query;
    dialog.selected = selected_id
        .as_deref()
        .and_then(|id| {
            dialog
                .items
                .iter()
                .position(|item| item.id == id && dialog.item_matches_query(item))
        })
        .or_else(|| dialog.visible_items().next().map(|(index, _)| index))
        .unwrap_or(0);
    dialog.detail_focused = detail_focused;
    dialog.detail_scroll = detail_scroll.min(dialog.detail_scroll_max);
    state.sync_context_picker_preview();
}

pub(crate) fn context_detail_target_exists(
    context: &ContextPaneState,
    target: &ContextDetailTarget,
) -> bool {
    match target {
        ContextDetailTarget::Node(node_id) => context
            .tree
            .nodes()
            .any(|node| node.node_id.as_str() == node_id),
        ContextDetailTarget::Block(block_id) => context
            .view
            .provider_active_blocks()
            .iter()
            .any(|(_, block)| block.block_id.as_str() == block_id),
        ContextDetailTarget::Summary(artifact_id) => context.view.summary_artifacts.iter().any(|artifact| {
            artifact.artifact_id == *artifact_id && context.view.provider_active_blocks().iter().any(|(_, block)| {
                matches!(&block.source, ContextBlockSource::SummaryArtifact { artifact_id: source } if source == artifact_id)
            })
        }),
    }
}

pub(super) fn parse_context_dialog_target(id: &str) -> Option<ContextDetailTarget> {
    let (kind, value) = id.split_once(':')?;
    match kind {
        "node" => Some(ContextDetailTarget::Node(value.to_string())),
        "block" => Some(ContextDetailTarget::Block(value.to_string())),
        "summary" => Some(ContextDetailTarget::Summary(value.to_string())),
        _ => None,
    }
}

pub(super) fn sync_context_picker_preview_for(
    dialog: &mut DialogState,
    context: &mut ContextPaneState,
) {
    let mut first_available: Option<(usize, ContextDetailTarget)> = None;
    let mut selected_target: Option<(usize, ContextDetailTarget)> = None;

    for (index, item) in dialog.visible_items() {
        let Some(target) = parse_context_dialog_target(&item.id) else {
            continue;
        };
        if !context_detail_target_exists(context, &target) {
            continue;
        }
        if first_available.is_none() {
            first_available = Some((index, target.clone()));
        }
        if index == dialog.selected {
            selected_target = Some((index, target));
            break;
        }
    }

    if let Some((_, target)) = selected_target {
        context.open_detail = Some(target);
        return;
    }

    if let Some((index, target)) = first_available {
        dialog.selected = index;
        dialog.reset_detail_focus();
        context.open_detail = Some(target);
        return;
    }

    dialog.reset_detail_focus();
    context.open_detail = None;
}

pub(super) fn project_context_open_detail(
    context: &ContextPaneState,
    target: &ContextDetailTarget,
) -> Option<ContextOpenDetailView> {
    match target {
        ContextDetailTarget::Block(block_id) => {
            let block = context
                .view
                .blocks
                .iter()
                .find(|(candidate, _)| candidate.as_str() == block_id)
                .map(|(_, block)| block)?;
            if context.view.is_compacted(&block.block_id) {
                return None;
            }
            let mut badges = Vec::new();
            match context.view.view_state.status(&block.block_id) {
                Some(ContextViewStatus::Pinned) => badges.push("Pinned".into()),
                Some(ContextViewStatus::Archived) => badges.push("Archived".into()),
                Some(ContextViewStatus::Resolved) => badges.push("Resolved".into()),
                Some(ContextViewStatus::RemovedFromView) => return None,
                _ => {}
            }
            if block.is_protected() {
                badges.push("Protected".into());
            }
            let mut lines = vec![normalize_context_detail_text(&block.detail)];
            lines.extend(context_block_source_lines(block, &context.view));
            Some(ContextOpenDetailView {
                title: block.title.clone(),
                badges,
                lines,
            })
        }
        ContextDetailTarget::Summary(artifact_id) => {
            let artifact = context.view.open_summary_artifact(artifact_id)?;
            let mut lines = vec![normalize_context_detail_text(&artifact.summary)];
            if let Some(node_id) = artifact.source_node_id.as_deref() {
                lines.push(format!("Source · {node_id}"));
            }
            if let Some(block_id) = artifact.source_block_id.as_deref() {
                lines.push(format!("Block · {block_id}"));
            }
            Some(ContextOpenDetailView {
                title: format!("Summary {}", artifact.artifact_id),
                badges: vec!["Summary".into()],
                lines,
            })
        }
        ContextDetailTarget::Node(node_id) => {
            let node = context
                .tree
                .nodes()
                .find(|node| node.node_id.as_str() == node_id)?;
            let mut badges = Vec::new();
            if context.tree.active_node_id() == Some(&node.node_id) {
                badges.push("Active".into());
            }
            if node.status == ContextNodeStatus::Archived {
                badges.push("Archived".into());
            }
            let mut lines = Vec::new();
            if let Some(purpose) = node.purpose.as_deref() {
                lines.push(normalize_context_detail_text(purpose));
            }
            if let Some(source_ref) = node.source_ref.as_ref() {
                lines.push(match source_ref.source_id.as_deref() {
                    Some(source_id) => format!("Source · {}:{}", source_ref.source_kind, source_id),
                    None => format!("Source · {}", source_ref.source_kind),
                });
            }
            Some(ContextOpenDetailView {
                title: node
                    .label
                    .clone()
                    .unwrap_or_else(|| node.node_id.as_str().to_string()),
                badges,
                lines,
            })
        }
    }
}

pub(super) fn context_node_depth(tree: &ContextTreeState, node_id: &str) -> usize {
    let mut depth = 0usize;
    let mut current = tree
        .nodes()
        .find(|node| node.node_id.as_str() == node_id)
        .and_then(|node| node.parent_node_id.clone());
    while let Some(parent_id) = current {
        if parent_id == *tree.root_node_id() {
            break;
        }
        depth = depth.saturating_add(1);
        current = tree
            .node(&parent_id)
            .and_then(|node| node.parent_node_id.clone());
    }
    depth
}

pub(super) fn context_block_source_lines(
    block: &ContextBlock,
    view: &ContextViewProjection,
) -> Vec<String> {
    let mut lines = Vec::new();
    match &block.source {
        ContextBlockSource::TranscriptSpan {
            start_sequence,
            end_sequence,
        } => lines.push(format!(
            "Source · transcript @{}–@{}",
            start_sequence, end_sequence
        )),
        ContextBlockSource::SummaryArtifact { artifact_id } => {
            lines.push(format!("Source · summary {artifact_id}"));
            if let Some(artifact) = view.open_summary_artifact(artifact_id) {
                if let Some(node_id) = artifact.source_node_id.as_deref() {
                    lines.push(format!("Node · {node_id}"));
                }
                if let Some(source_block_id) = artifact.source_block_id.as_deref() {
                    lines.push(format!("Block · {source_block_id}"));
                }
            }
        }
    }
    lines
}

pub(super) fn normalize_context_detail_text(text: &str) -> String {
    text.split_whitespace().collect::<Vec<_>>().join(" ")
}

pub(super) fn child_transcript_model(records: &[TranscriptRecord]) -> Option<String> {
    let mut model = None;
    for record in records {
        match &record.event {
            TranscriptEvent::SessionStarted {
                model: session_model,
            } => model = Some(session_model.clone()),
            TranscriptEvent::ModelChanged { new_model, .. } => model = Some(new_model.clone()),
            _ => {}
        }
    }
    model
}
