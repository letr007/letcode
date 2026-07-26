use std::collections::BTreeSet;

use crate::context_view::{
    ContextBlock, ContextBlockKind, ContextBlockRetention, ContextBlockSource,
    ContextViewProjection, ContextViewStatus, ProtectedReason,
};
use crate::protocol_frames::{ProtocolFrame, ProtocolFrameItem};

use super::{HistoryAdapterProjection, HistoryItem, PromptMessage, PromptMessageOrigin};

pub(super) fn context_view_history_adapter(
    context_view: &ContextViewProjection,
    history: &[HistoryItem],
    protected_start_index: usize,
) -> HistoryAdapterProjection {
    let mut sections = HistoryAdapterProjection::default();
    let sorted_blocks = sorted_context_blocks(context_view);

    let protected_blocks = sorted_blocks
        .iter()
        .filter(|(id, block)| {
            !context_view.is_compacted(id) && block.is_protected() && !is_resolved(context_view, id)
        })
        .map(|(id, block)| (*id, *block))
        .collect::<Vec<_>>();
    if !protected_blocks.is_empty() {
        sections.prelude.push(PromptMessage::developer_with_origin(
            format!(
                "[Context: Hard Context]\n{}",
                protected_blocks
                    .iter()
                    .map(|(id, block)| format_protected_context_block_line(id, block))
                    .collect::<Vec<_>>()
                    .join("\n")
            ),
            PromptMessageOrigin::RuntimeContextView,
        ));
    }

    let protected_ids = protected_blocks
        .iter()
        .map(|(id, _)| id.as_str().to_string())
        .collect::<BTreeSet<_>>();
    let pinned_blocks = sorted_blocks
        .iter()
        .filter(|(id, block)| {
            !context_view.is_compacted(id)
                && is_pinned_visible(context_view, id)
                && !protected_ids.contains(id.as_str())
        })
        .map(|(id, block)| (*id, *block))
        .collect::<Vec<_>>();
    if !pinned_blocks.is_empty() {
        sections.prelude.push(PromptMessage::developer_with_origin(
            format!(
                "[Context: Pinned Context]\n{}",
                pinned_blocks
                    .iter()
                    .map(|(id, block)| format_context_block_line(id, block, false))
                    .collect::<Vec<_>>()
                    .join("\n")
            ),
            PromptMessageOrigin::RuntimeContextView,
        ));
    }

    if let Some(tail_section) = build_protected_tail_section(history, protected_start_index) {
        sections
            .history_prefix
            .push(ProtocolFrame::derived(ProtocolFrameItem::ContextSummary {
                text: tail_section,
            }));
    }

    let visible_index_blocks = sorted_blocks
        .iter()
        .filter(|(id, block)| include_in_context_index(context_view, id, block))
        .map(|(id, block)| format_context_block_line(id, block, false))
        .collect::<Vec<_>>();
    if !visible_index_blocks.is_empty() {
        sections
            .history_prefix
            .push(ProtocolFrame::derived(ProtocolFrameItem::ContextSummary {
                text: format!("[Context: Index]\n{}", visible_index_blocks.join("\n")),
            }));
    }

    let summaries = context_view
        .summary_artifacts
        .iter()
        .map(format_summary_artifact)
        .collect::<Vec<_>>();
    if !summaries.is_empty() {
        sections
            .history_prefix
            .push(ProtocolFrame::derived(ProtocolFrameItem::ContextSummary {
                text: format!("[Context: Summaries]\n{}", summaries.join("\n")),
            }));
    }

    if let Some(open_id) = context_view.view_state.open_detail_block_id()
        && let Some(block) = context_view.blocks.get(open_id)
        && !context_view.is_compacted(open_id)
        && view_status(context_view, open_id) != ContextViewStatus::RemovedFromView
        && !is_resolved(context_view, open_id)
    {
        sections
            .history_prefix
            .push(ProtocolFrame::derived(ProtocolFrameItem::ContextSummary {
                text: format!(
                    "[Context: Opened Details]\n{}\nDetail: {}",
                    format_context_block_line(open_id, block, false),
                    excerpt(&block.detail, 1200)
                ),
            }));
    }

    sections
}

fn is_opened(
    context_view: &ContextViewProjection,
    block_id: &crate::context_view::ContextBlockId,
) -> bool {
    context_view.view_state.open_detail_block_id() == Some(block_id)
}

fn view_status(
    context_view: &ContextViewProjection,
    block_id: &crate::context_view::ContextBlockId,
) -> ContextViewStatus {
    context_view
        .view_state
        .status(block_id)
        .unwrap_or(ContextViewStatus::Visible)
}

fn is_normally_visible(
    context_view: &ContextViewProjection,
    block_id: &crate::context_view::ContextBlockId,
) -> bool {
    matches!(
        view_status(context_view, block_id),
        ContextViewStatus::Visible | ContextViewStatus::Pinned
    )
}

fn is_resolved(
    context_view: &ContextViewProjection,
    block_id: &crate::context_view::ContextBlockId,
) -> bool {
    view_status(context_view, block_id) == ContextViewStatus::Resolved
}

fn is_pinned_visible(
    context_view: &ContextViewProjection,
    block_id: &crate::context_view::ContextBlockId,
) -> bool {
    view_status(context_view, block_id) == ContextViewStatus::Pinned
}

fn include_in_context_index(
    context_view: &ContextViewProjection,
    block_id: &crate::context_view::ContextBlockId,
    block: &ContextBlock,
) -> bool {
    if is_resolved(context_view, block_id) {
        return false;
    }
    if context_view.is_compacted(block_id) {
        return false;
    }
    if block.retention_class() == ContextBlockRetention::Debug {
        return is_pinned_visible(context_view, block_id) || is_opened(context_view, block_id);
    }
    block.is_protected() || is_normally_visible(context_view, block_id)
}

fn sorted_context_blocks(
    context_view: &ContextViewProjection,
) -> Vec<(&crate::context_view::ContextBlockId, &ContextBlock)> {
    let mut blocks = context_view.blocks.iter().collect::<Vec<_>>();
    blocks.sort_by(|(left_id, left), (right_id, right)| {
        left.source_start_sequence
            .or(left.available_sequence)
            .unwrap_or(u64::MAX)
            .cmp(
                &right
                    .source_start_sequence
                    .or(right.available_sequence)
                    .unwrap_or(u64::MAX),
            )
            .then_with(|| left_id.as_str().cmp(right_id.as_str()))
    });
    blocks
}

fn build_protected_tail_section(
    history: &[HistoryItem],
    protected_start_index: usize,
) -> Option<String> {
    let protected = &history[protected_start_index.min(history.len())..];
    let tail = protected.iter().rev().take(4).collect::<Vec<_>>();
    if tail.is_empty() {
        return None;
    }
    let mut lines = tail
        .into_iter()
        .rev()
        .map(format_history_item_line)
        .collect::<Vec<_>>();
    lines.retain(|line| !line.is_empty());
    if lines.is_empty() {
        None
    } else {
        Some(format!("[Context: Active Tail]\n{}", lines.join("\n")))
    }
}

fn format_history_item_line(item: &HistoryItem) -> String {
    match item {
        HistoryItem::ContextSummary { text } => format!("- summary: {}", excerpt(text, 240)),
        HistoryItem::UserMessage { content } => {
            format!("- user: {}", excerpt(&content.display_text(), 240))
        }
        HistoryItem::InternalContinuation { text } => {
            format!("- continuation: {}", excerpt(text, 240))
        }
        HistoryItem::AssistantText { text } => format!("- assistant: {}", excerpt(text, 240)),
        HistoryItem::AssistantToolCalls { calls, .. } => format!(
            "- tool_calls: {}",
            calls
                .iter()
                .map(|call| call.name.as_str())
                .collect::<Vec<_>>()
                .join(", ")
        ),
        HistoryItem::ToolOutput { call_id, .. } => format!("- tool_output: {call_id}"),
    }
}

fn format_context_block_line(
    block_id: &crate::context_view::ContextBlockId,
    block: &ContextBlock,
    include_protected: bool,
) -> String {
    let mut tags = Vec::new();
    tags.push(format!("id={}", block_id.as_str()));
    tags.push(format!("kind={}", context_block_kind_label(block.kind)));
    tags.push(format!(
        "retention={}",
        context_block_retention_label(block.retention_class())
    ));
    if include_protected && !block.protected_reasons.is_empty() {
        tags.push(format!(
            "protected={}",
            block
                .protected_reasons
                .iter()
                .map(|reason| protected_reason_label(*reason))
                .collect::<Vec<_>>()
                .join(",")
        ));
    }
    tags.push(format!("source={}", format_block_source(&block.source)));
    format!(
        "- [{}] {} :: {}",
        tags.join(" "),
        block.title,
        excerpt(&block.detail, 240)
    )
}

fn format_protected_context_block_line(
    block_id: &crate::context_view::ContextBlockId,
    block: &ContextBlock,
) -> String {
    let mut tags = Vec::new();
    tags.push(format!("id={}", block_id.as_str()));
    tags.push(format!("kind={}", context_block_kind_label(block.kind)));
    tags.push(format!(
        "retention={}",
        context_block_retention_label(block.retention_class())
    ));
    if !block.protected_reasons.is_empty() {
        tags.push(format!(
            "protected={}",
            block
                .protected_reasons
                .iter()
                .map(|reason| protected_reason_label(*reason))
                .collect::<Vec<_>>()
                .join(",")
        ));
    }
    tags.push(format!("source={}", format_block_source(&block.source)));
    format!("- [{}] {} :: {}", tags.join(" "), block.title, block.detail)
}

fn format_summary_artifact(artifact: &crate::context_view::SummaryArtifact) -> String {
    format!(
        "- id={} node={} version={} kind={} source_node={} source_block={} span={}..{} :: {}",
        artifact.artifact_id,
        artifact.node_id,
        artifact.version,
        artifact.artifact_kind,
        artifact.source_node_id.as_deref().unwrap_or("-"),
        artifact.source_block_id.as_deref().unwrap_or("-"),
        artifact
            .source_start_sequence
            .map(|value| value.to_string())
            .unwrap_or_else(|| "-".into()),
        artifact
            .source_end_sequence
            .map(|value| value.to_string())
            .unwrap_or_else(|| "-".into()),
        excerpt(&artifact.summary, 240)
    )
}

fn format_block_source(source: &ContextBlockSource) -> String {
    match source {
        ContextBlockSource::TranscriptSpan {
            start_sequence,
            end_sequence,
        } => format!("transcript:{start_sequence}..{end_sequence}"),
        ContextBlockSource::SummaryArtifact { artifact_id } => format!("summary:{artifact_id}"),
    }
}

fn context_block_kind_label(kind: ContextBlockKind) -> &'static str {
    match kind {
        ContextBlockKind::HardConstraint => "hard_constraint",
        ContextBlockKind::CurrentUserRequirement => "current_user_requirement",
        ContextBlockKind::UnresolvedError => "unresolved_error",
        ContextBlockKind::Permission => "permission",
        ContextBlockKind::FileWriteFact => "file_write_fact",
        ContextBlockKind::TestResult => "test_result",
        ContextBlockKind::CommitHash => "commit_hash",
        ContextBlockKind::ToolOutput => "tool_output",
        ContextBlockKind::Note => "note",
        ContextBlockKind::ReasoningNote => "reasoning_note",
    }
}

fn context_block_retention_label(retention: ContextBlockRetention) -> &'static str {
    match retention {
        ContextBlockRetention::Critical => "critical",
        ContextBlockRetention::Protected => "protected",
        ContextBlockRetention::Working => "working",
        ContextBlockRetention::Debug => "debug",
    }
}

fn protected_reason_label(reason: ProtectedReason) -> &'static str {
    match reason {
        ProtectedReason::HardConstraint => "hard_constraint",
        ProtectedReason::CurrentUserRequirement => "current_user_requirement",
        ProtectedReason::UnresolvedError => "unresolved_error",
        ProtectedReason::Permission => "permission",
        ProtectedReason::FileWriteFact => "file_write_fact",
        ProtectedReason::TestResult => "test_result",
        ProtectedReason::CommitHash => "commit_hash",
    }
}

fn excerpt(text: &str, max_chars: usize) -> String {
    let compact = text.split_whitespace().collect::<Vec<_>>().join(" ");
    let mut value = compact.chars().take(max_chars).collect::<String>();
    if compact.chars().count() > max_chars {
        value.push('…');
    }
    value
}
