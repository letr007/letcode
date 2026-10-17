use super::*;
use crate::agent::{ContextCompactionEvent, ToolExecutionSummaryEvent, ValidationAdvisory};
use crate::tool::ToolResult;
use crate::transcript::{TranscriptEvent, TranscriptRecord};
use crate::user_content::UserMessageContent;
use serde_json::json;

fn record_at(sequence: u64, event: TranscriptEvent) -> TranscriptRecord {
    TranscriptRecord {
        session_id: "s".into(),
        sequence,
        timestamp_ms: 0,
        context_branch_id: None,
        event,
    }
}

#[test]
fn protected_context_cannot_be_archived_or_removed() {
    let projection = project_context_view(&[record_at(
        1,
        TranscriptEvent::UserMessage {
            content: UserMessageContent::from("Do not commit; must keep tests passing"),
        },
    )])
    .expect("project context view");
    let block_id = ContextBlockId::new("block-seq-1-user-requirement").expect("block id");
    let block = projection
        .blocks
        .get(&block_id)
        .expect("protected block exists");
    assert!(block.is_protected());

    let archive_error = ContextViewState::replay(
        &projection.blocks,
        &[ContextViewOperation::Archive {
            block_id: block_id.clone(),
        }],
    )
    .expect_err("archive should fail");
    assert!(
        archive_error
            .to_string()
            .contains("cannot archive protected context block")
    );

    let remove_error = ContextViewState::replay(
        &projection.blocks,
        &[ContextViewOperation::RemoveFromView { block_id }],
    )
    .expect_err("remove should fail");
    assert!(
        remove_error
            .to_string()
            .contains("cannot remove_from_view protected context block")
    );
}

#[test]
fn resolve_marks_unresolved_error_without_mutating_raw_block() {
    let projection = project_context_view(&[
        record_at(
            1,
            TranscriptEvent::Error {
                message: "context view projection unavailable".into(),
            },
        ),
        record_at(
            2,
            TranscriptEvent::ContextViewOperationMetadata {
                operation: "resolve".into(),
                node_id: None,
                block_id: Some("block-seq-1-error".into()),
                detail: None,
            },
        ),
    ])
    .expect("resolved error projects");

    let block_id = ContextBlockId::new("block-seq-1-error").expect("id");
    let block = projection
        .blocks
        .get(&block_id)
        .expect("error block exists");
    assert_eq!(block.kind, ContextBlockKind::UnresolvedError);
    assert!(block.is_protected());
    assert_eq!(
        projection.view_state.status(&block_id),
        Some(ContextViewStatus::Resolved)
    );
}

#[test]
fn resolve_rejects_non_error_blocks() {
    let error = project_context_view(&[
        record_at(
            1,
            TranscriptEvent::AssistantMessage {
                content: "plain note".into(),
            },
        ),
        record_at(
            2,
            TranscriptEvent::ContextViewOperationMetadata {
                operation: "resolve".into(),
                node_id: None,
                block_id: Some("block-seq-1-note".into()),
                detail: None,
            },
        ),
    ])
    .expect_err("non-error resolve should fail");

    assert!(
        error
            .to_string()
            .contains("cannot resolve context block 'block-seq-1-note' with kind note")
    );
}

#[test]
fn replay_supports_pin_archive_remove_and_open_detail() {
    let projection = project_context_view(&[
        record_at(
            1,
            TranscriptEvent::AssistantMessage {
                content: "note one".into(),
            },
        ),
        record_at(
            2,
            TranscriptEvent::ContextViewOperationMetadata {
                operation: "pin".into(),
                node_id: None,
                block_id: Some("block-seq-1-note".into()),
                detail: None,
            },
        ),
        record_at(
            3,
            TranscriptEvent::ContextViewOperationMetadata {
                operation: "open_detail".into(),
                node_id: None,
                block_id: Some("block-seq-1-note".into()),
                detail: None,
            },
        ),
        record_at(
            4,
            TranscriptEvent::AssistantMessage {
                content: "note two".into(),
            },
        ),
        record_at(
            5,
            TranscriptEvent::ContextViewOperationMetadata {
                operation: "archive".into(),
                node_id: None,
                block_id: Some("block-seq-4-note".into()),
                detail: None,
            },
        ),
        record_at(
            6,
            TranscriptEvent::AssistantMessage {
                content: "note three".into(),
            },
        ),
        record_at(
            7,
            TranscriptEvent::ContextViewOperationMetadata {
                operation: "remove_from_view".into(),
                node_id: None,
                block_id: Some("block-seq-6-note".into()),
                detail: None,
            },
        ),
    ])
    .expect("project view state");

    assert_eq!(
        projection
            .view_state
            .status(&ContextBlockId::new("block-seq-1-note").expect("id")),
        Some(ContextViewStatus::Pinned)
    );
    assert_eq!(
        projection
            .view_state
            .status(&ContextBlockId::new("block-seq-4-note").expect("id")),
        Some(ContextViewStatus::Archived)
    );
    assert_eq!(
        projection
            .view_state
            .status(&ContextBlockId::new("block-seq-6-note").expect("id")),
        Some(ContextViewStatus::RemovedFromView)
    );
    assert_eq!(
        projection
            .view_state
            .open_detail_block_id()
            .map(ContextBlockId::as_str),
        Some("block-seq-1-note")
    );
}

#[test]
fn context_view_rejects_operation_targeting_future_block() {
    let error = project_context_view(&[
        record_at(
            1,
            TranscriptEvent::ContextViewOperationMetadata {
                operation: "pin".into(),
                node_id: None,
                block_id: Some("block-seq-2-note".into()),
                detail: None,
            },
        ),
        record_at(
            2,
            TranscriptEvent::AssistantMessage {
                content: "future note".into(),
            },
        ),
    ])
    .expect_err("future block operation should fail");

    assert!(
        error
            .to_string()
            .contains("targets future block 'block-seq-2-note' created at sequence 2")
    );
}

#[test]
fn file_write_fact_requires_active_context_node() {
    let error = project_context_view(&[
        record_at(
            1,
            TranscriptEvent::ContextNodeLifecycle {
                node_id: "root".into(),
                status: crate::context_tree::ContextNodeStatus::Inactive,
            },
        ),
        record_at(
            2,
            TranscriptEvent::ToolExecutionSummary(ToolExecutionSummaryEvent {
                turn_id: 1,
                call_id: "call-write".into(),
                name: "fs__write".into(),
                status: "completed".into(),
                rejection: None,
                effect_kind: "write".into(),
                primary_path: Some("src/lib.rs".into()),
                command: None,
            }),
        ),
    ])
    .expect_err("write fact without active context node should fail");

    assert!(
        error
            .to_string()
            .contains("file write fact at transcript sequence 2 has no active context node")
    );
}
