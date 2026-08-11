use super::*;
use crate::agent::{CompactionCheckpoint, CompactionFileOperations, ContextCompactionEvent};
use crate::agent::{ToolExecutionSummaryEvent, TurnStartedEvent, ValidationAdvisory};
use crate::context_tree::ContextNodeStatus;
use crate::context_view::{
    ContextBlock, ContextBlockId, ContextBlockKind, ContextBlockSource, ContextViewOperation,
    ContextViewProjection, ContextViewState, ProtectedReason,
};
use crate::evidence::{EvidenceKind, EvidenceSource};
use crate::protocol_frames::history_items_from_frames;
use crate::request_builder::HistoryToolCall;
use crate::runtime_context::{FrameVisibility, RuntimeFrameId, RuntimeFrameKind, SourceSpan};
use crate::tool::ToolResult;
use crate::transcript::LogicalCheckpointRetainedItemV1;
use crate::user_content::UserMessageContent;
use serde_json::json;
use std::collections::BTreeMap;

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

fn mixed_context_view(compacted: &[&str]) -> ContextViewProjection {
    let mut projection = ContextViewProjection::default();
    for sequence in 1..=3 {
        let block_id = ContextBlockId::new(format!("block-{sequence}")).expect("valid block");
        projection.blocks.insert(
            block_id.clone(),
            ContextBlock {
                block_id,
                node_id: None,
                kind: ContextBlockKind::ToolOutput,
                title: format!("block {sequence}"),
                detail: format!("detail {sequence}"),
                source: ContextBlockSource::TranscriptSpan {
                    start_sequence: sequence,
                    end_sequence: sequence,
                },
                source_start_sequence: Some(sequence),
                available_sequence: Some(sequence),
                protected_reasons: Vec::new(),
            },
        );
    }
    projection.compacted_block_ids = compacted
        .iter()
        .map(|id| ContextBlockId::new(*id).expect("valid block"))
        .collect();
    projection
}
fn frame_ids_by_source(
    snapshot: &RuntimeSnapshot,
    kind: RuntimeFrameKind,
) -> BTreeMap<String, RuntimeFrameId> {
    snapshot
        .frames
        .iter()
        .filter(|frame| frame.kind == kind)
        .map(|frame| {
            (
                frame.provenance.source_id.clone().expect("source id"),
                frame.id,
            )
        })
        .collect()
}

#[test]
fn context_view_contributor_protects_only_hard_pinned_or_opened_blocks() {
    let mut projection = mixed_context_view(&[]);
    let id = |value| ContextBlockId::new(value).expect("valid block id");
    projection
        .blocks
        .get_mut(&id("block-2"))
        .expect("hard block")
        .protected_reasons = vec![ProtectedReason::CurrentUserRequirement];
    projection.view_state = ContextViewState::replay(
        &projection.blocks,
        &[
            ContextViewOperation::Pin {
                block_id: id("block-3"),
            },
            ContextViewOperation::OpenDetail {
                block_id: id("block-1"),
            },
        ],
    )
    .expect("valid context view state");

    let snapshot = snapshot_for_context_view(&projection);
    let context_ids = frame_ids_by_source(&snapshot, RuntimeFrameKind::ContextBlock);
    let contributor = snapshot
        .prompt_contributors
        .iter()
        .find(|contributor| contributor.contributor_id == "context-view-active")
        .expect("context contributor");

    assert_eq!(
        contributor.frame_ids,
        vec![
            context_ids["block-1"],
            context_ids["block-2"],
            context_ids["block-3"]
        ]
    );
    snapshot
        .validate_references()
        .expect("retention references resolve");
}

#[test]
fn opened_archived_detail_without_visible_index_retains_its_source() {
    let mut projection = mixed_context_view(&[]);
    let id = |value| ContextBlockId::new(value).expect("valid block id");
    projection.view_state = ContextViewState::replay(
        &projection.blocks,
        &[
            ContextViewOperation::Archive {
                block_id: id("block-1"),
            },
            ContextViewOperation::Archive {
                block_id: id("block-2"),
            },
            ContextViewOperation::Archive {
                block_id: id("block-3"),
            },
            ContextViewOperation::OpenDetail {
                block_id: id("block-1"),
            },
        ],
    )
    .expect("archived block can be opened for detail");

    assert!(projection.provider_visible_block_ids().is_empty());
    assert_eq!(
        projection
            .provider_active_blocks()
            .iter()
            .map(|(block_id, _)| block_id.as_str())
            .collect::<Vec<_>>(),
        vec!["block-1"]
    );

    let mut opened = snapshot_for_context_view(&projection);
    let opened_context_id = frame_ids_by_source(&opened, RuntimeFrameKind::ContextBlock)["block-1"];
    let contributor = opened
        .prompt_contributors
        .iter()
        .find(|contributor| contributor.contributor_id == "context-view-active")
        .expect("opened detail has a context contributor");
    assert_eq!(contributor.frame_ids, vec![opened_context_id]);
    opened.recompute_protected_frame_ids();
    // Soft-retaining context contributors still surface source spans for prompt
    // assembly, but hard protect is explicit/turn only — opened detail must not
    // freeze protocol history via protected_frame_ids.
    assert!(
        !opened
            .compaction
            .protected_frame_ids
            .contains(&opened_context_id),
        "opened detail soft-retain must not hard-protect compaction sources"
    );
    assert!(
        opened
            .prompt_contributor_source_spans()
            .expect("opened source spans")
            .contains(&SourceSpan::new(1, 1).expect("valid source span"))
    );
    opened
        .validate_references()
        .expect("opened contributor references resolve");

    projection.view_state = ContextViewState::replay(
        &projection.blocks,
        &[
            ContextViewOperation::Archive {
                block_id: id("block-1"),
            },
            ContextViewOperation::Archive {
                block_id: id("block-2"),
            },
            ContextViewOperation::Archive {
                block_id: id("block-3"),
            },
        ],
    )
    .expect("cleared opened detail");
    let mut cleared = snapshot_for_context_view(&projection);
    assert!(
        cleared
            .prompt_contributors
            .iter()
            .all(|contributor| contributor.contributor_id != "context-view-active")
    );
    cleared.recompute_protected_frame_ids();
    assert!(
        !cleared
            .compaction
            .protected_frame_ids
            .contains(&opened_context_id),
        "cleared detail no longer blocks source co-retirement"
    );
    cleared
        .validate_references()
        .expect("cleared contributor references resolve");
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
    assert_eq!(
        child.parent_node_id.as_ref().map(|id| id.as_str()),
        Some("root")
    );
    assert_eq!(child.label.as_deref(), Some("Task branch"));
    assert_eq!(
        child.purpose.as_deref(),
        Some("Investigate session-level replay")
    );
    assert_eq!(tree.active_node_id().map(|id| id.as_str()), Some("child"));
    assert_eq!(tree.node_count(), 2);
}

#[test]
fn context_tree_active_branch_projection_excludes_abandoned_sibling_nodes() {
    let records = [
        record_at(
            1,
            TranscriptEvent::SessionStarted {
                model: "gpt-5".into(),
            },
        ),
        metadata_record_at(
            2,
            TranscriptEvent::ContextBranchCreated {
                branch_id: "active".into(),
                parent_branch_id: ROOT_CONTEXT_BRANCH_ID.into(),
                base_sequence: 1,
                label: None,
            },
        ),
        metadata_record_at(
            3,
            TranscriptEvent::ContextBranchCreated {
                branch_id: "abandoned".into(),
                parent_branch_id: ROOT_CONTEXT_BRANCH_ID.into(),
                base_sequence: 1,
                label: None,
            },
        ),
        branch_record_at(
            4,
            "active",
            TranscriptEvent::UserMessage {
                content: UserMessageContent::from("active branch"),
            },
        ),
        metadata_record_at(
            5,
            TranscriptEvent::ContextCheckout {
                branch_id: "active".into(),
                leaf_sequence: 4,
            },
        ),
        metadata_record_at(
            6,
            TranscriptEvent::ContextNodeCreated {
                node_id: "active-node".into(),
                parent_node_id: Some("root".into()),
                label: None,
                purpose: None,
                block_ref: None,
                source_ref: None,
            },
        ),
        metadata_record_at(
            7,
            TranscriptEvent::ContextNodeLifecycle {
                node_id: "root".into(),
                status: ContextNodeStatus::Inactive,
            },
        ),
        metadata_record_at(
            8,
            TranscriptEvent::ContextNodeLifecycle {
                node_id: "active-node".into(),
                status: ContextNodeStatus::Active,
            },
        ),
        metadata_record_at(
            9,
            TranscriptEvent::ContextCheckout {
                branch_id: "abandoned".into(),
                leaf_sequence: 1,
            },
        ),
        branch_record_at(
            10,
            "abandoned",
            TranscriptEvent::UserMessage {
                content: UserMessageContent::from("abandoned branch"),
            },
        ),
        metadata_record_at(
            11,
            TranscriptEvent::ContextNodeCreated {
                node_id: "active-node".into(),
                parent_node_id: Some("root".into()),
                label: Some("same id in abandoned scope".into()),
                purpose: None,
                block_ref: None,
                source_ref: None,
            },
        ),
        metadata_record_at(
            12,
            TranscriptEvent::ContextCheckout {
                branch_id: "active".into(),
                leaf_sequence: 4,
            },
        ),
    ];

    assert!(
        replay_context_tree(&records)
            .expect_err("global replay mixes checkout scopes")
            .to_string()
            .contains("duplicate context node_id 'active-node'")
    );

    let tree = project_context_tree_for_active_branch(&records, None)
        .expect("project active branch context tree");
    let active_node = tree
        .node(&ContextNodeId::new("active-node").expect("active node id"))
        .expect("active branch node");

    assert_eq!(tree.node_count(), 2);
    assert_eq!(active_node.label, None);
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

    assert!(
        error
            .to_string()
            .contains("unknown parent context node 'missing'")
    );
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

    assert!(
        error
            .to_string()
            .contains("cannot activate context node 'child-b' while 'child-a' is active")
    );
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

    assert!(
        error
            .to_string()
            .contains("duplicate context node_id 'child'")
    );
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

    assert!(
        error
            .to_string()
            .contains("context node 'self' cannot be its own parent")
    );
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
        records.clone(),
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
            TranscriptEvent::ContextCompaction(ContextCompactionEvent::succeeded("summary", 1)),
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
        records.clone(),
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
fn max_turn_id_is_global_across_restored_leaf_cuts() {
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
            TranscriptEvent::ContextBranchCreated {
                branch_id: "later".into(),
                parent_branch_id: ROOT_CONTEXT_BRANCH_ID.into(),
                base_sequence: 2,
                label: None,
            },
        ),
        branch_record_at(
            4,
            "later",
            TranscriptEvent::TurnStarted(TurnStartedEvent {
                turn_id: 7,
                intent: "future".into(),
                directive: "later".into(),
                validation_reminder: String::new(),
            }),
        ),
        branch_record_at(
            5,
            "later",
            TranscriptEvent::TurnInterrupted { turn_id: Some(7) },
        ),
    ];
    let records_for_snapshot = records.clone();

    let snapshot = build_session_context_snapshot(
        "s".into(),
        records_for_snapshot,
        SessionContextCursor {
            branch_id: None,
            leaf_sequence: Some(2),
        },
    )
    .expect("snapshot before future turn");

    assert_eq!(snapshot.max_turn_id, 7);

    let runtime = project_runtime_restore_snapshot(
        "s".into(),
        records,
        SessionContextCursor {
            branch_id: Some(ROOT_CONTEXT_BRANCH_ID.into()),
            leaf_sequence: Some(2),
        },
        &[],
    )
    .expect("root runtime snapshot before sibling turn");
    assert_eq!(runtime.max_turn_id, 7);
    assert_eq!(runtime.snapshot.current_turn_id, Some(1));
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
    assert!(
        snapshot
            .messages
            .iter()
            .all(|message| message.content != "root-after-fork")
    );
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

    let snapshot = project_session_restore_snapshot("s".into(), records)
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
        SessionContextCursor {
            branch_id: Some("missing".into()),
            leaf_sequence: None,
        },
    )
    .expect_err("unknown branch should fail");
    assert!(
        unknown_branch_error
            .to_string()
            .contains("unknown context branch 'missing'")
    );

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
        SessionContextCursor {
            branch_id: None,
            leaf_sequence: None,
        },
    )
    .expect_err("invalid base should fail");
    assert!(
        invalid_base_error
            .to_string()
            .contains("base_sequence 9 is not resolvable")
    );

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
        SessionContextCursor {
            branch_id: Some("feature".into()),
            leaf_sequence: Some(4),
        },
    )
    .expect_err("leaf beyond tip should fail");
    assert!(
        leaf_beyond_tip_error
            .to_string()
            .contains("requested leaf_sequence 4 exceeds tip 3 for branch 'feature'")
    );
}

#[test]
fn branch_summary_cannot_reference_a_future_branch_tip() {
    let records = vec![
        record_at(
            1,
            TranscriptEvent::UserMessage {
                content: UserMessageContent::from("root"),
            },
        ),
        metadata_record_at(
            2,
            TranscriptEvent::ContextBranchCreated {
                branch_id: "child".into(),
                parent_branch_id: ROOT_CONTEXT_BRANCH_ID.into(),
                base_sequence: 1,
                label: None,
            },
        ),
        metadata_record_at(
            3,
            TranscriptEvent::ContextBranchSummary {
                branch_id: "child".into(),
                leaf_sequence: 4,
                summary: "premature".into(),
            },
        ),
        branch_record_at(
            4,
            "child",
            TranscriptEvent::AssistantMessage {
                content: "future child content".into(),
            },
        ),
    ];

    let error = list_context_branches(&records, None)
        .expect_err("a summary must not reference a future branch tip");

    assert!(
        error
            .to_string()
            .contains("context branch summary leaf_sequence 4 exceeds tip 1 for branch 'child'")
    );
}

#[test]
fn session_protocol_frames_restore_history_compatibly() {
    let records = vec![
        record_at(
            1,
            TranscriptEvent::UserMessage {
                content: UserMessageContent::from("question"),
            },
        ),
        record_at(
            2,
            TranscriptEvent::AssistantMessage {
                content: "answer".into(),
            },
        ),
        record_at(
            3,
            TranscriptEvent::ToolCallStarted {
                call_id: "call-1".into(),
                name: "shell__exec".into(),
                args: json!({"command":"cargo check"}),
            },
        ),
        record_at(
            4,
            TranscriptEvent::ToolCallFinished {
                call_id: "call-1".into(),
                name: "shell__exec".into(),
                ok: true,
                output: ToolResult::ok("shell__exec", json!({"status": 0})),
            },
        ),
    ];

    let history = restore_session_history_projection(&records);
    let frames = restore_session_protocol_frames_projection(&records).expect("projection");

    assert_eq!(history_items_from_frames(&frames), history);
}

#[test]
fn legacy_incomplete_tool_group_is_removed_before_a_new_turn_is_appended() {
    let records = vec![
        record_at(
            1,
            TranscriptEvent::UserMessage {
                content: UserMessageContent::from("interrupted prompt"),
            },
        ),
        record_at(
            2,
            TranscriptEvent::ToolCallStarted {
                call_id: "call-1".into(),
                name: "fs__read".into(),
                args: json!({"path": "src/main.rs"}),
            },
        ),
    ];

    let mut restored = restore_session_history_projection(&records);
    assert_eq!(restored, vec![HistoryItem::user("interrupted prompt")]);

    restored.push(HistoryItem::user("new prompt"));
    let protocol = analyze_history_items(&restored, None).expect("new turn can be appended");
    assert!(!protocol.has_incomplete_tool_call_groups());
}

#[test]
fn legacy_cancelled_tool_call_restores_as_terminal_output() {
    let records = vec![
        record_at(
            1,
            TranscriptEvent::UserMessage {
                content: UserMessageContent::from("interrupted prompt"),
            },
        ),
        record_at(
            2,
            TranscriptEvent::ToolCallStarted {
                call_id: "call-1".into(),
                name: "fs__read".into(),
                args: json!({"path": "src/main.rs"}),
            },
        ),
        record_at(
            3,
            TranscriptEvent::ToolCallCancelled {
                call_id: "call-1".into(),
                name: "fs__read".into(),
            },
        ),
    ];

    let mut restored = restore_session_history_projection(&records);
    assert!(
        matches!(restored.last(), Some(HistoryItem::ToolOutput { call_id, output_json, .. })
        if call_id == "call-1" && output_json == r#"{"status":"cancelled","summary":"user cancelled"}"#)
    );
    restored.push(HistoryItem::user("new prompt"));
    crate::protocol_frames::validate_history_items_complete(&restored, None)
        .expect("cancelled legacy call is terminal");
}

#[test]
fn cancelled_durable_multi_call_batch_restores_every_terminal_output() {
    let calls = vec![
        HistoryToolCall {
            call_id: "call-1".into(),
            name: "fs__read".into(),
            arguments_json: "{}".into(),
        },
        HistoryToolCall {
            call_id: "call-2".into(),
            name: "fs__read".into(),
            arguments_json: "{}".into(),
        },
    ];
    let records = vec![
        record_at(
            1,
            TranscriptEvent::AssistantToolCallBatch {
                text: None,
                reasoning_content: None,
                calls,
            },
        ),
        record_at(
            2,
            TranscriptEvent::ToolCallCancelled {
                call_id: "call-1".into(),
                name: "fs__read".into(),
            },
        ),
    ];

    let restored = restore_session_history_projection(&records);
    assert!(
        matches!(restored.first(), Some(HistoryItem::AssistantToolCalls { calls, .. }) if calls.len() == 2)
    );
    assert_eq!(
        restored
            .iter()
            .filter(|item| matches!(item, HistoryItem::ToolOutput { .. }))
            .count(),
        2
    );
    crate::protocol_frames::validate_history_items_complete(&restored, None)
        .expect("all cancelled batch calls have terminal outputs");
}

#[test]
fn runtime_snapshot_projection_respects_branch_cursor() {
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
                label: Some("feature".into()),
            },
        ),
        branch_record_at(
            3,
            "feature",
            TranscriptEvent::AssistantMessage {
                content: "feature-only".into(),
            },
        ),
        record_at(
            4,
            TranscriptEvent::AssistantMessage {
                content: "root-after".into(),
            },
        ),
    ];

    let projected = project_runtime_restore_snapshot(
        "s".into(),
        records,
        SessionContextCursor {
            branch_id: Some("feature".into()),
            leaf_sequence: Some(3),
        },
        &[],
    )
    .expect("branch runtime snapshot");

    assert_eq!(projected.branch_id, "feature");
    assert_eq!(projected.leaf_sequence, 3);
    assert_eq!(projected.snapshot.active_context.branch_id, "feature");
    assert!(
        projected
            .snapshot
            .frames
            .iter()
            .any(|frame| frame.summary.as_deref() == Some("feature-only"))
    );
    assert!(
        projected
            .snapshot
            .frames
            .iter()
            .all(|frame| frame.summary.as_deref() != Some("root-after"))
    );
}

#[test]
fn runtime_snapshot_marks_entire_current_turn_as_protected() {
    let records = vec![
        record_at(
            1,
            TranscriptEvent::TurnStarted(TurnStartedEvent {
                turn_id: 7,
                intent: "chat".into(),
                directive: "answer".into(),
                validation_reminder: String::new(),
            }),
        ),
        record_at(
            2,
            TranscriptEvent::UserMessage {
                content: UserMessageContent::from("question"),
            },
        ),
        record_at(
            3,
            TranscriptEvent::ToolCallStarted {
                call_id: "call-1".into(),
                name: "fs__read".into(),
                args: json!({"path":"src/main.rs"}),
            },
        ),
    ];

    let projected = project_runtime_restore_snapshot(
        "s".into(),
        records,
        SessionContextCursor {
            branch_id: None,
            leaf_sequence: None,
        },
        &[],
    )
    .expect("runtime snapshot projection");

    let protected = &projected.snapshot.compaction.protected_frame_ids;
    assert_eq!(protected.len(), 2);
    assert!(projected.snapshot.frames.iter().any(|frame| {
        protected.contains(&frame.id) && frame.kind == RuntimeFrameKind::ToolCall
    }));
    assert!(
        projected
            .snapshot
            .frames
            .iter()
            .any(|frame| { protected.contains(&frame.id) && frame.kind == RuntimeFrameKind::User })
    );
}

#[test]
fn append_only_checkout_restore_scopes_branch_content_and_runtime_metadata() {
    let records = vec![
        record_at(
            1,
            TranscriptEvent::UserMessage {
                content: UserMessageContent::from("root-prefix"),
            },
        ),
        record_at(
            2,
            TranscriptEvent::AssistantMessage {
                content: "root-fork-base".into(),
            },
        ),
        metadata_record_at(
            3,
            TranscriptEvent::ContextBranchCreated {
                branch_id: "child".into(),
                parent_branch_id: ROOT_CONTEXT_BRANCH_ID.into(),
                base_sequence: 2,
                label: Some("child lane".into()),
            },
        ),
        metadata_record_at(
            4,
            TranscriptEvent::ContextCheckout {
                branch_id: "child".into(),
                leaf_sequence: 2,
            },
        ),
        branch_record_at(
            5,
            "child",
            TranscriptEvent::UserMessage {
                content: UserMessageContent::from("child-only"),
            },
        ),
        branch_record_at(
            6,
            "child",
            TranscriptEvent::AssistantMessage {
                content: "child-reply".into(),
            },
        ),
        metadata_record_at(
            7,
            TranscriptEvent::ContextNodeCreated {
                node_id: "child-node".into(),
                parent_node_id: Some("root".into()),
                label: Some("Child".into()),
                purpose: None,
                block_ref: None,
                source_ref: None,
            },
        ),
        metadata_record_at(
            8,
            TranscriptEvent::ContextNodeLifecycle {
                node_id: "root".into(),
                status: ContextNodeStatus::Inactive,
            },
        ),
        metadata_record_at(
            9,
            TranscriptEvent::ContextNodeLifecycle {
                node_id: "child-node".into(),
                status: ContextNodeStatus::Active,
            },
        ),
        metadata_record_at(
            10,
            TranscriptEvent::ContextViewOperationMetadata {
                operation: "pin".into(),
                node_id: Some("child-node".into()),
                block_id: Some("block-seq-6-note".into()),
                detail: None,
            },
        ),
        metadata_record_at(
            11,
            TranscriptEvent::ContextCheckout {
                branch_id: ROOT_CONTEXT_BRANCH_ID.into(),
                leaf_sequence: 2,
            },
        ),
        record_at(
            12,
            TranscriptEvent::AssistantMessage {
                content: "root-after-fork".into(),
            },
        ),
        metadata_record_at(
            13,
            TranscriptEvent::ContextNodeCreated {
                node_id: "root-after-node".into(),
                parent_node_id: Some("root".into()),
                label: Some("Root after fork".into()),
                purpose: None,
                block_ref: None,
                source_ref: None,
            },
        ),
        metadata_record_at(
            14,
            TranscriptEvent::ContextNodeLifecycle {
                node_id: "root".into(),
                status: ContextNodeStatus::Inactive,
            },
        ),
        metadata_record_at(
            15,
            TranscriptEvent::ContextNodeLifecycle {
                node_id: "root-after-node".into(),
                status: ContextNodeStatus::Active,
            },
        ),
        metadata_record_at(
            16,
            TranscriptEvent::ContextViewOperationMetadata {
                operation: "pin".into(),
                node_id: Some("root-after-node".into()),
                block_id: Some("block-seq-12-note".into()),
                detail: None,
            },
        ),
        metadata_record_at(
            17,
            TranscriptEvent::ContextCheckout {
                branch_id: "child".into(),
                leaf_sequence: 6,
            },
        ),
        metadata_record_at(
            18,
            TranscriptEvent::ContextNodeLifecycle {
                node_id: "root-after-node".into(),
                status: ContextNodeStatus::Inactive,
            },
        ),
        metadata_record_at(
            19,
            TranscriptEvent::ContextNodeLifecycle {
                node_id: "child-node".into(),
                status: ContextNodeStatus::Active,
            },
        ),
        metadata_record_at(
            20,
            TranscriptEvent::ContextViewOperationMetadata {
                operation: "open_detail".into(),
                node_id: Some("child-node".into()),
                block_id: Some("block-seq-6-note".into()),
                detail: None,
            },
        ),
    ];
    let original_input = serde_json::to_value(&records).expect("serialize input records");
    let project = |branch_id| {
        project_runtime_restore_snapshot(
            "s".into(),
            records.clone(),
            SessionContextCursor {
                branch_id,
                leaf_sequence: None,
            },
            &[],
        )
        .expect("project branch cursor")
    };
    let root = project(Some(ROOT_CONTEXT_BRANCH_ID.into()));
    let child = project(Some("child".into()));
    let latest = project(None);
    let contents = |projected: &RuntimeRestoreSnapshot| {
        projected
            .records
            .iter()
            .filter_map(|record| match &record.event {
                TranscriptEvent::UserMessage { content } => Some(content.display_text()),
                TranscriptEvent::AssistantMessage { content } => Some(content.clone()),
                _ => None,
            })
            .collect::<Vec<_>>()
    };

    assert_eq!(
        contents(&root),
        ["root-prefix", "root-fork-base", "root-after-fork"]
    );
    assert_eq!(
        contents(&child),
        ["root-prefix", "root-fork-base", "child-only", "child-reply"]
    );
    assert_eq!(latest.branch_id, "child");
    assert_eq!(contents(&latest), contents(&child));
    assert_eq!(
        child.snapshot.active_context.parent_branch_id.as_deref(),
        Some(ROOT_CONTEXT_BRANCH_ID)
    );
    assert_eq!(root.snapshot.context_scope_revision, 11);
    assert_eq!(child.snapshot.context_scope_revision, 17);
    assert_eq!(latest.snapshot.context_scope_revision, 17);
    assert_eq!(
        root.snapshot
            .context_tree
            .active_node_id()
            .map(|id| id.as_str()),
        Some("root-after-node")
    );
    assert_eq!(
        child
            .snapshot
            .context_tree
            .active_node_id()
            .map(|id| id.as_str()),
        Some("child-node")
    );
    assert_eq!(
        root.snapshot.active_context.pinned_block_ids,
        ["block-seq-12-note"]
    );
    assert_eq!(
        child
            .snapshot
            .active_context
            .open_detail_block_id
            .as_deref(),
        Some("block-seq-6-note")
    );
    assert_eq!(
        latest.snapshot.active_context.open_detail_block_id,
        child.snapshot.active_context.open_detail_block_id
    );
    assert_eq!(
        serde_json::to_value(&records).expect("serialize records after projection"),
        original_input
    );
}

fn checkpoint_candidate_fixture() -> (Vec<TranscriptRecord>, LogicalCheckpointEventV1) {
    let records = vec![
        record_at(
            1,
            TranscriptEvent::UserMessage {
                content: UserMessageContent::from("preserve the requirement"),
            },
        ),
        record_at(
            2,
            TranscriptEvent::TurnStarted(TurnStartedEvent {
                turn_id: 7,
                intent: "test".into(),
                directive: "retain the request".into(),
                validation_reminder: String::new(),
            }),
        ),
    ];
    let event = LogicalCheckpointEventV1 {
        schema_version: 1,
        checkpoint_id: "checkpoint-1".into(),
        turn_id: 7,
        previous_segment_id: 0,
        segment_id: 1,
        previous_checkpoint_id: None,
        boundary_sequence: 2,
        context_scope_revision: 0,
        covered_source_spans: vec![LogicalCheckpointSourceSpanV1 {
            start_sequence: 1,
            end_sequence: 1,
        }],
        retained_items: vec![LogicalCheckpointRetainedItemV1 {
            kind: crate::transcript::LogicalCheckpointRetainedKindV1::UserRequirement,
            title: "Requirement".into(),
            detail: "preserve the requirement".into(),
            audit_source: LogicalCheckpointAuditSourceV1::TranscriptSpan {
                start_sequence: 1,
                end_sequence: 1,
            },
        }],
    };
    (records, event)
}

fn non_root_checkpoint_journal() -> Vec<TranscriptRecord> {
    let mut records = vec![
        record_at(
            1,
            TranscriptEvent::UserMessage {
                content: UserMessageContent::from("root fork base"),
            },
        ),
        metadata_record_at(
            2,
            TranscriptEvent::ContextBranchCreated {
                branch_id: "child".into(),
                parent_branch_id: ROOT_CONTEXT_BRANCH_ID.into(),
                base_sequence: 1,
                label: None,
            },
        ),
        metadata_record_at(
            3,
            TranscriptEvent::ContextCheckout {
                branch_id: "child".into(),
                leaf_sequence: 1,
            },
        ),
        branch_record_at(
            4,
            "child",
            TranscriptEvent::UserMessage {
                content: UserMessageContent::from("child requirement"),
            },
        ),
        branch_record_at(
            5,
            "child",
            TranscriptEvent::TurnStarted(TurnStartedEvent {
                turn_id: 7,
                intent: "test".into(),
                directive: "retain the child request".into(),
                validation_reminder: String::new(),
            }),
        ),
    ];
    let checkpoint = prepare_logical_checkpoint_candidate("s", &records, "child".into(), 5)
        .expect("prepare child checkpoint");
    records.push(branch_record_at(
        6,
        "child",
        TranscriptEvent::LogicalCheckpoint(checkpoint),
    ));
    records
}

#[test]
fn non_root_checkpoint_journal_restores_without_re_resolving_selected_content() {
    let records = non_root_checkpoint_journal();

    let restored = project_runtime_restore_snapshot(
        "s".into(),
        records,
        SessionContextCursor {
            branch_id: Some("child".into()),
            leaf_sequence: None,
        },
        &[],
    )
    .expect("restore child checkpoint journal");

    assert_eq!(restored.branch_id, "child");
    assert_eq!(restored.snapshot.current_segment_id, Some(1));
}

#[test]
fn standalone_projection_rejects_branch_scoped_content_without_its_journal() {
    let records = non_root_checkpoint_journal();
    let selected_content = records
        .into_iter()
        .filter(|record| record.context_branch_id.as_deref() == Some("child"))
        .collect::<Vec<_>>();

    assert!(crate::context_view::project_context_view(&selected_content).is_err());
}

#[test]
fn logical_checkpoint_contract_rejects_the_regression_matrix_without_panicking() {
    let (records, valid) = checkpoint_candidate_fixture();
    validate_logical_checkpoint_candidate(
        "s",
        &records,
        Some(ROOT_CONTEXT_BRANCH_ID.into()),
        2,
        3,
        &valid,
    )
    .expect("fixture is a valid checkpoint candidate");

    let mut cases = Vec::new();
    for (name, mutate) in [
        (
            "schema",
            Box::new(|event: &mut LogicalCheckpointEventV1| event.schema_version = 2)
                as Box<dyn Fn(&mut LogicalCheckpointEventV1)>,
        ),
        (
            "id",
            Box::new(|event: &mut LogicalCheckpointEventV1| event.checkpoint_id = "bad id".into()),
        ),
        (
            "segment lineage",
            Box::new(|event: &mut LogicalCheckpointEventV1| event.segment_id = 3),
        ),
        (
            "previous checkpoint lineage",
            Box::new(|event: &mut LogicalCheckpointEventV1| {
                event.previous_checkpoint_id = Some("missing".into())
            }),
        ),
        (
            "boundary",
            Box::new(|event: &mut LogicalCheckpointEventV1| event.boundary_sequence = 3),
        ),
        (
            "scope revision",
            Box::new(|event: &mut LogicalCheckpointEventV1| event.context_scope_revision = 1),
        ),
        (
            "empty coverage",
            Box::new(|event: &mut LogicalCheckpointEventV1| event.covered_source_spans.clear()),
        ),
        (
            "inverted coverage",
            Box::new(|event: &mut LogicalCheckpointEventV1| {
                event.covered_source_spans[0] = LogicalCheckpointSourceSpanV1 {
                    start_sequence: 2,
                    end_sequence: 1,
                }
            }),
        ),
        (
            "coverage mismatch",
            Box::new(|event: &mut LogicalCheckpointEventV1| {
                event.covered_source_spans[0].end_sequence = 2
            }),
        ),
        (
            "audit outside closure",
            Box::new(|event: &mut LogicalCheckpointEventV1| {
                event.retained_items[0].audit_source =
                    LogicalCheckpointAuditSourceV1::TranscriptSpan {
                        start_sequence: 2,
                        end_sequence: 2,
                    }
            }),
        ),
        (
            "retained item size",
            Box::new(|event: &mut LogicalCheckpointEventV1| {
                event.retained_items[0].detail = "x".repeat(4097)
            }),
        ),
    ] {
        let mut event = valid.clone();
        mutate(&mut event);
        cases.push((name, event));
    }
    let mut duplicate = valid.clone();
    duplicate
        .retained_items
        .push(duplicate.retained_items[0].clone());
    cases.push(("duplicate retained item", duplicate));
    let mut too_many = valid.clone();
    too_many.retained_items = (0..65)
        .map(|index| LogicalCheckpointRetainedItemV1 {
            kind: crate::transcript::LogicalCheckpointRetainedKindV1::UserRequirement,
            title: format!("Requirement {index:02}"),
            detail: "retained".into(),
            audit_source: LogicalCheckpointAuditSourceV1::TranscriptSpan {
                start_sequence: 1,
                end_sequence: 1,
            },
        })
        .collect();
    cases.push(("retained item count", too_many));

    for (name, event) in cases {
        let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            validate_logical_checkpoint_candidate(
                "s",
                &records,
                Some(ROOT_CONTEXT_BRANCH_ID.into()),
                2,
                3,
                &event,
            )
        }));
        assert!(outcome.is_ok(), "{name} must not panic");
        assert!(
            outcome.expect("checked above").is_err(),
            "{name} must reject"
        );
    }
}

#[test]
fn public_restores_reject_malformed_checkpoint_before_projection() {
    let (mut records, mut event) = checkpoint_candidate_fixture();
    event.schema_version = 0;
    records.push(branch_record_at(
        3,
        ROOT_CONTEXT_BRANCH_ID,
        TranscriptEvent::LogicalCheckpoint(event),
    ));

    let history = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        crate::transcript::restore_session_history(&records)
    }));
    let compacted_messages = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        crate::transcript::restore_compacted_conversation_messages(&records)
    }));
    let messages = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        crate::transcript::restore_conversation_messages(&records)
    }));
    for outcome in [
        history.map(|result| result.is_err()),
        compacted_messages.map(|result| result.is_err()),
        messages.map(|result| result.is_err()),
    ] {
        assert!(outcome.expect("public restore must not panic"));
    }
    assert!(crate::transcript::restore_session_protocol_frames(&records).is_err());
    assert!(crate::transcript::restore_runtime_snapshot(&records).is_err());
}

#[test]
fn checkpoint_uses_the_content_boundary_but_keeps_metadata_at_the_journal_frontier() {
    let (mut records, mut checkpoint) = checkpoint_candidate_fixture();
    records.push(metadata_record_at(
        3,
        TranscriptEvent::FoldedOutputMetadata {
            node_id: None,
            output_id: "frontier-metadata".into(),
            output_kind: "metadata".into(),
            call_id: None,
            tool_name: None,
            stream: None,
            content: Some("metadata after content".into()),
            byte_count: None,
            line_count: None,
            truncated: None,
            shell_command: None,
            source_start_sequence: None,
            source_end_sequence: None,
            tool_ok: None,
            exit_status: None,
            provider_metadata: None,
            provider_fold_eligible: None,
        },
    ));
    checkpoint.boundary_sequence = 2;
    validate_logical_checkpoint_candidate(
        "s",
        &records,
        Some(ROOT_CONTEXT_BRANCH_ID.into()),
        3,
        4,
        &checkpoint,
    )
    .expect("metadata must not move the content boundary");
    records.push(branch_record_at(
        4,
        ROOT_CONTEXT_BRANCH_ID,
        TranscriptEvent::LogicalCheckpoint(checkpoint),
    ));

    let root = project_runtime_restore_snapshot(
        "s".into(),
        records.clone(),
        SessionContextCursor {
            branch_id: Some(ROOT_CONTEXT_BRANCH_ID.into()),
            leaf_sequence: None,
        },
        &[],
    )
    .expect("restore root checkpoint");
    let latest = project_runtime_restore_snapshot(
        "s".into(),
        records,
        SessionContextCursor {
            branch_id: None,
            leaf_sequence: None,
        },
        &[],
    )
    .expect("restore latest checkout");
    assert_eq!(root.snapshot.current_segment_id, Some(1));
    assert_eq!(root.snapshot.context_scope_revision, 0);
    assert_eq!(latest.branch_id, ROOT_CONTEXT_BRANCH_ID);
    assert!(root.snapshot.active_history_items().iter().any(|item| {
        matches!(item, HistoryItem::ContextSummary { text } if text.contains("checkpoint-1"))
    }));
}

#[test]
fn branch_matrix_resolves_root_parent_child_and_sibling_with_cursor_precedence() {
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
                branch_id: "parent".into(),
                parent_branch_id: ROOT_CONTEXT_BRANCH_ID.into(),
                base_sequence: 1,
                label: None,
            },
        ),
        branch_record_at(
            3,
            "parent",
            TranscriptEvent::AssistantMessage {
                content: "parent".into(),
            },
        ),
        record_at(
            4,
            TranscriptEvent::ContextBranchCreated {
                branch_id: "child".into(),
                parent_branch_id: "parent".into(),
                base_sequence: 3,
                label: None,
            },
        ),
        branch_record_at(
            5,
            "child",
            TranscriptEvent::AssistantMessage {
                content: "child".into(),
            },
        ),
        record_at(
            6,
            TranscriptEvent::ContextBranchCreated {
                branch_id: "sibling".into(),
                parent_branch_id: ROOT_CONTEXT_BRANCH_ID.into(),
                base_sequence: 1,
                label: None,
            },
        ),
        branch_record_at(
            7,
            "sibling",
            TranscriptEvent::AssistantMessage {
                content: "sibling".into(),
            },
        ),
        record_at(
            8,
            TranscriptEvent::ContextCheckout {
                branch_id: "sibling".into(),
                leaf_sequence: 7,
            },
        ),
    ];
    let visible = |branch_id: Option<&str>| {
        build_session_context_snapshot(
            "s".into(),
            records.clone(),
            SessionContextCursor {
                branch_id: branch_id.map(str::to_string),
                leaf_sequence: None,
            },
        )
        .expect("branch matrix projection")
        .records
        .into_iter()
        .map(|record| record.sequence)
        .collect::<Vec<_>>()
    };

    assert_eq!(visible(Some(ROOT_CONTEXT_BRANCH_ID)), vec![1]);
    assert_eq!(visible(Some("parent")), vec![1, 3]);
    assert_eq!(visible(Some("child")), vec![1, 3, 5]);
    assert_eq!(visible(Some("sibling")), vec![1, 7]);
    assert_eq!(visible(None), vec![1, 7]);
    // The explicit cursor is authoritative even after a later checkout.
    assert_eq!(visible(Some("child")), vec![1, 3, 5]);
}

#[test]
fn checkpoint_and_compaction_preserve_active_frames_and_retire_closed_artifacts() {
    let (mut records, checkpoint) = checkpoint_candidate_fixture();
    records.push(branch_record_at(
        3,
        ROOT_CONTEXT_BRANCH_ID,
        TranscriptEvent::LogicalCheckpoint(checkpoint),
    ));
    records.push(branch_record_at(
        4,
        ROOT_CONTEXT_BRANCH_ID,
        TranscriptEvent::AssistantMessage {
            content: "active segment".into(),
        },
    ));
    let restore = || {
        project_runtime_restore_snapshot(
            "s".into(),
            records.clone(),
            SessionContextCursor {
                branch_id: Some(ROOT_CONTEXT_BRANCH_ID.into()),
                leaf_sequence: None,
            },
            &[],
        )
        .expect("checkpoint projection")
    };
    let before = restore();
    let active_ids = before
        .snapshot
        .active_protocol_frames()
        .iter()
        .map(|frame| frame.runtime_frame_id)
        .collect::<Vec<_>>();
    assert!(before.snapshot.active_history_items().iter().any(
        |item| matches!(item, HistoryItem::AssistantText { text } if text == "active segment")
    ));
    assert_eq!(
        active_ids,
        restore()
            .snapshot
            .active_protocol_frames()
            .iter()
            .map(|frame| frame.runtime_frame_id)
            .collect::<Vec<_>>()
    );

    // A historical compaction remains retired when a later checkpoint closes
    // a new segment; raw retirement and logical retirement stay cumulative.
    let historical = vec![
        record_at(
            1,
            TranscriptEvent::UserMessage {
                content: UserMessageContent::from("old"),
            },
        ),
        record_at(
            2,
            TranscriptEvent::AssistantMessage {
                content: "old reply".into(),
            },
        ),
        record_at(
            3,
            TranscriptEvent::ContextCompaction(ContextCompactionEvent::succeeded("old summary", 1)),
        ),
    ];
    let history = restore_history_projection(&historical);
    assert!(history.iter().any(
        |entry| matches!(&entry.item, HistoryItem::ContextSummary { text } if text == "old summary")
    ));
}

#[test]
fn replay_derived_compaction_retires_context_and_evidence_without_persisted_spans() {
    let records = vec![
        record_at(
            1,
            TranscriptEvent::UserMessage {
                content: UserMessageContent::from("old request"),
            },
        ),
        record_at(
            2,
            TranscriptEvent::Evidence {
                id: "old-evidence".into(),
                evidence_kind: EvidenceKind::Decision,
                title: "old evidence".into(),
                summary: "old evidence summary".into(),
                detail: None,
                source: EvidenceSource::Transcript { sequence: 1 },
                tags: Vec::new(),
            },
        ),
        record_at(
            3,
            TranscriptEvent::AssistantMessage {
                content: "old reply".into(),
            },
        ),
        record_at(
            4,
            TranscriptEvent::ContextCompaction(ContextCompactionEvent::succeeded("old summary", 1)),
        ),
        record_at(
            5,
            TranscriptEvent::UserMessage {
                content: UserMessageContent::from("current request"),
            },
        ),
    ];

    let runtime = project_runtime_restore_snapshot(
        "s".into(),
        records,
        SessionContextCursor {
            branch_id: None,
            leaf_sequence: None,
        },
        &[],
    )
    .expect("runtime projection");

    assert!(runtime.snapshot.evidence.is_empty());
    assert!(
        !runtime
            .snapshot
            .prompt_contributors
            .iter()
            .any(|contributor| contributor.contributor_id == "evidence")
    );
    assert!(runtime.snapshot.frames.iter().any(|frame| {
        frame.kind == RuntimeFrameKind::ContextBlock
            && frame.visibility == crate::runtime_context::FrameVisibility::Retired
            && frame
                .provenance
                .source_span
                .is_some_and(|span| span.start_sequence == 1)
    }));
}

#[test]
fn old_transcripts_have_exact_history_and_runtime_projection_without_checkpoints() {
    let records = vec![
        record_at(
            1,
            TranscriptEvent::UserMessage {
                content: UserMessageContent::from("legacy request"),
            },
        ),
        record_at(
            2,
            TranscriptEvent::AssistantMessage {
                content: "legacy reply".into(),
            },
        ),
        record_at(
            3,
            TranscriptEvent::ToolCallStarted {
                call_id: "call".into(),
                name: crate::tool_names::TOOL_FS_READ.into(),
                args: json!({"path": "src/lib.rs"}),
            },
        ),
        record_at(
            4,
            TranscriptEvent::ToolCallFinished {
                call_id: "call".into(),
                name: crate::tool_names::TOOL_FS_READ.into(),
                ok: true,
                output: ToolResult::ok(crate::tool_names::TOOL_FS_READ, json!({"content": "ok"})),
            },
        ),
    ];
    let session = project_session_restore_snapshot("s".into(), records.clone())
        .expect("legacy session projection");
    let runtime = project_runtime_restore_snapshot(
        "s".into(),
        records.clone(),
        SessionContextCursor {
            branch_id: None,
            leaf_sequence: None,
        },
        &[],
    )
    .expect("legacy runtime projection");
    assert_eq!(
        session.history,
        restore_session_history_projection(&records)
    );
    assert_eq!(
        runtime
            .protocol_frames
            .iter()
            .map(|frame| frame.item.clone())
            .collect::<Vec<_>>(),
        history_items_to_frames(&session.history)
            .into_iter()
            .map(|frame| frame.item)
            .collect::<Vec<_>>()
    );
    assert_eq!(
        runtime.snapshot.active_history_items(),
        session.history.as_slice()
    );
    assert!(runtime.snapshot.current_segment_id.is_none());
}

#[test]
fn checkpoint_permission_and_execution_facts_rebind_to_finished_output() {
    use crate::transcript::LogicalCheckpointRetainedKindV1 as Kind;
    let groups = vec![CoveredCallGroup {
        call_id: "call-1".into(),
        name: "shell".into(),
        assistant_sequence: 3,
        finished_sequence: 5,
    }];
    let permission = record_at(
        4,
        TranscriptEvent::PermissionDecision {
            call_id: Some("call-1".into()),
            tool: "shell".into(),
            args: json!({}),
            allowed: true,
            reason: None,
            reviewer: None,
            approval: None,
            risk: None,
            reviewer_child_session_id: None,
        },
    );
    let write = record_at(
        7,
        TranscriptEvent::ToolExecutionSummary(ToolExecutionSummaryEvent {
            turn_id: 1,
            call_id: "call-1".into(),
            name: "shell".into(),
            status: "ok".into(),
            rejection: None,
            effect_kind: "write".into(),
            primary_path: Some("src/lib.rs".into()),
            command: None,
        }),
    );
    assert_eq!(
        validate_current_fact_provenance(
            &[permission],
            4,
            4,
            Kind::Permission,
            "permission",
            &groups,
            1,
        )
        .expect("permission must bind")
        .finished_sequence,
        5
    );
    assert_eq!(
        validate_current_fact_provenance(&[write], 7, 7, Kind::FileWriteFact, "write", &groups, 1,)
            .expect("write summary must bind")
            .finished_sequence,
        5
    );
}

#[test]
fn checkpoint_facts_with_missing_or_ambiguous_call_ids_fail() {
    use crate::transcript::LogicalCheckpointRetainedKindV1 as Kind;
    let records = vec![record_at(
        1,
        TranscriptEvent::PermissionDecision {
            call_id: None,
            tool: "shell".into(),
            args: json!({}),
            allowed: true,
            reason: None,
            reviewer: None,
            approval: None,
            risk: None,
            reviewer_child_session_id: None,
        },
    )];
    assert!(
        validate_current_fact_provenance(&records, 1, 1, Kind::Permission, "permission", &[], 1)
            .is_err()
    );
    let groups = vec![
        CoveredCallGroup {
            call_id: "dup".into(),
            name: "shell".into(),
            assistant_sequence: 2,
            finished_sequence: 3,
        },
        CoveredCallGroup {
            call_id: "dup".into(),
            name: "shell".into(),
            assistant_sequence: 4,
            finished_sequence: 5,
        },
    ];
    let records = vec![record_at(
        1,
        TranscriptEvent::PermissionDecision {
            call_id: Some("dup".into()),
            tool: "shell".into(),
            args: json!({}),
            allowed: true,
            reason: None,
            reviewer: None,
            approval: None,
            risk: None,
            reviewer_child_session_id: None,
        },
    )];
    assert!(
        validate_current_fact_provenance(
            &records,
            1,
            1,
            Kind::Permission,
            "permission",
            &groups,
            1,
        )
        .is_err()
    );
}

// Already at the fit shrink floor so the algorithm must drop rather than shrink.
const MIN_PRIORITY_DROP_TEXT: usize = 64;

fn split_checkpoint_records() -> Vec<TranscriptRecord> {
    vec![
        record_at(
            1,
            TranscriptEvent::UserMessage {
                content: UserMessageContent::from("current requirement"),
            },
        ),
        record_at(
            2,
            TranscriptEvent::TurnStarted(TurnStartedEvent {
                turn_id: 7,
                intent: "continue".into(),
                directive: "keep current requirement".into(),
                validation_reminder: String::new(),
            }),
        ),
        record_at(
            3,
            TranscriptEvent::AssistantMessage {
                content: "completed tool work".into(),
            },
        ),
        record_at(
            4,
            TranscriptEvent::ContextCompaction(ContextCompactionEvent::checkpointed(
                "legacy summary",
                2,
                CompactionCheckpoint {
                    next_action: "legacy action".into(),
                    continuation: "legacy continuation".into(),
                    split_turn_handoff: None,
                    file_operations: CompactionFileOperations::default(),
                },
            )),
        ),
    ]
}

#[test]
fn checkpointed_compaction_replays_internal_continuation_after_preserved_user() {
    let records = split_checkpoint_records();

    let history = restore_session_history_projection(&records);
    assert_eq!(history.len(), 3);
    assert!(matches!(&history[0], HistoryItem::ContextSummary { .. }));
    assert!(matches!(
        &history[1],
        HistoryItem::UserMessage { content } if content.display_text() == "current requirement"
    ));
    assert!(matches!(
        &history[2],
        HistoryItem::InternalContinuation { text } if text == "legacy continuation"
    ));

    let snapshot = crate::transcript::restore_runtime_snapshot(&records)
        .expect("checkpointed compaction snapshot restores");
    let continuation = snapshot
        .frames
        .iter()
        .find(|frame| frame.summary.as_deref() == Some("legacy continuation"))
        .expect("continuation frame exists");
    assert_eq!(
        continuation
            .provenance
            .source_span
            .expect("continuation has durable source")
            .start_sequence,
        4,
        "continuation provenance belongs to the compaction event, not retired history"
    );
}

#[test]
fn legacy_checkpointed_compaction_replays_tampered_continuation_without_rederivation() {
    let mut records = split_checkpoint_records();
    let TranscriptEvent::ContextCompaction(event) = &mut records.last_mut().unwrap().event else {
        panic!("last record is compaction");
    };
    event
        .checkpoint
        .as_mut()
        .expect("checkpoint")
        .continuation
        .push_str(" tampered");

    let snapshot = crate::transcript::restore_runtime_snapshot(&records)
        .expect("legacy checkpoint continuation replays as recorded");
    assert!(snapshot.frames.iter().any(|frame| {
        frame
            .summary
            .as_deref()
            .is_some_and(|summary| summary.ends_with("tampered"))
    }));
}

#[test]
fn legacy_checkpointed_split_compaction_replays_without_handoff_rederivation() {
    let mut records = split_checkpoint_records();
    let TranscriptEvent::ContextCompaction(event) = &mut records.last_mut().unwrap().event else {
        panic!("last record is compaction");
    };
    event
        .checkpoint
        .as_mut()
        .expect("checkpoint")
        .split_turn_handoff = None;

    crate::transcript::restore_runtime_snapshot(&records)
        .expect("legacy checkpoint without handoff replays as recorded");
}

#[test]
fn modern_active_turn_compaction_retires_current_user_with_prefix() {
    let records = vec![
        record_at(
            1,
            TranscriptEvent::UserMessage {
                content: UserMessageContent::from("old request"),
            },
        ),
        record_at(
            2,
            TranscriptEvent::AssistantMessage {
                content: "old reply".into(),
            },
        ),
        record_at(
            3,
            TranscriptEvent::UserMessage {
                content: UserMessageContent::from("current requirement"),
            },
        ),
        record_at(
            4,
            TranscriptEvent::TurnStarted(TurnStartedEvent {
                turn_id: 7,
                intent: "continue".into(),
                directive: "keep the current requirement".into(),
                validation_reminder: String::new(),
            }),
        ),
        record_at(
            5,
            TranscriptEvent::AssistantToolCallBatch {
                text: None,
                reasoning_content: None,
                calls: vec![HistoryToolCall {
                    call_id: "c1".into(),
                    name: "fs__read".into(),
                    arguments_json: "{}".into(),
                }],
            },
        ),
        record_at(
            6,
            TranscriptEvent::ToolCallFinished {
                call_id: "c1".into(),
                name: "fs__read".into(),
                ok: true,
                output: ToolResult::ok("fs__read", json!({"text": "done"})),
            },
        ),
        record_at(
            7,
            TranscriptEvent::ContextCompaction(ContextCompactionEvent::succeeded_at(
                "summary",
                Some("raw:5".into()),
            )),
        ),
    ];

    let history = restore_session_history_projection(&records);
    assert_eq!(history.len(), 3);
    assert!(matches!(
        &history[0],
        HistoryItem::ContextSummary { text } if text == "summary"
    ));
    assert!(matches!(
        &history[1],
        HistoryItem::AssistantToolCalls { .. }
    ));
    assert!(matches!(&history[2], HistoryItem::ToolOutput { .. }));
}

#[test]
fn branch_compaction_validates_against_selected_branch_scope() {
    let records = vec![
        record_at(
            1,
            TranscriptEvent::UserMessage {
                content: UserMessageContent::from("root"),
            },
        ),
        record_at(
            2,
            TranscriptEvent::AssistantMessage {
                content: "root-reply".into(),
            },
        ),
        record_at(
            3,
            TranscriptEvent::ContextBranchCreated {
                branch_id: "feature".into(),
                parent_branch_id: ROOT_CONTEXT_BRANCH_ID.into(),
                base_sequence: 1,
                label: None,
            },
        ),
        branch_record_at(
            4,
            "feature",
            TranscriptEvent::UserMessage {
                content: UserMessageContent::from("branch request"),
            },
        ),
        branch_record_at(
            5,
            "feature",
            TranscriptEvent::ContextCompaction(ContextCompactionEvent::succeeded_at(
                "branch summary",
                Some("raw:4".into()),
            )),
        ),
        branch_record_at(
            6,
            "feature",
            TranscriptEvent::AssistantMessage {
                content: "branch reply".into(),
            },
        ),
    ];

    let restored = project_runtime_restore_snapshot(
        "s".into(),
        records,
        SessionContextCursor {
            branch_id: Some("feature".into()),
            leaf_sequence: None,
        },
        &[],
    );
    assert!(
        restored.is_ok(),
        "branch compaction must validate against its branch scope, got: {restored:?}"
    );
}

#[test]
fn navigation_restore_validates_root_compaction_on_root_scope() {
    // Linear root journal with one modern compaction. Undo must validate the
    // compaction against the root scope where it was committed, not against the
    // new history branch created for the navigation (which postdates it).
    let records = vec![
        record_at(
            1,
            TranscriptEvent::UserMessage {
                content: UserMessageContent::from("a"),
            },
        ),
        record_at(
            2,
            TranscriptEvent::AssistantMessage {
                content: "a-reply".into(),
            },
        ),
        record_at(
            3,
            TranscriptEvent::UserMessage {
                content: UserMessageContent::from("b"),
            },
        ),
        record_at(
            4,
            TranscriptEvent::AssistantMessage {
                content: "b-reply".into(),
            },
        ),
        record_at(
            5,
            TranscriptEvent::ContextCompaction(ContextCompactionEvent::succeeded_at(
                "summary",
                Some("raw:3".into()),
            )),
        ),
        record_at(
            6,
            TranscriptEvent::UserMessage {
                content: UserMessageContent::from("c"),
            },
        ),
        record_at(
            7,
            TranscriptEvent::AssistantMessage {
                content: "c-reply".into(),
            },
        ),
        record_at(
            8,
            TranscriptEvent::UserMessage {
                content: UserMessageContent::from("d"),
            },
        ),
        record_at(
            9,
            TranscriptEvent::AssistantMessage {
                content: "d-reply".into(),
            },
        ),
    ];

    // Replicate navigate_undo: current = last visible entry, walk up the parent
    // chain to the User turn root, target = that User entry's parent sequence.
    let entries = project_session_history_tree(&records);
    let snapshot = build_session_context_snapshot(
        "s".into(),
        records.clone(),
        SessionContextCursor {
            branch_id: None,
            leaf_sequence: None,
        },
    )
    .expect("root session context builds");
    let visible: std::collections::BTreeSet<u64> =
        snapshot.records.iter().map(|r| r.sequence).collect();
    let current = entries
        .iter()
        .rev()
        .find(|entry| visible.contains(&entry.sequence))
        .map(|entry| entry.sequence)
        .expect("current session history entry");
    let mut turn_root = entries
        .iter()
        .find(|entry| entry.sequence == current)
        .expect("current history entry exists");
    while turn_root.kind != SessionHistoryEntryKind::User {
        let parent_id = turn_root
            .parent_id
            .as_deref()
            .expect("history parent is available");
        turn_root = entries
            .iter()
            .find(|entry| entry.id == parent_id)
            .expect("history parent exists");
    }
    let target = turn_root
        .parent_id
        .as_deref()
        .and_then(|id| id.strip_prefix("entry-"))
        .and_then(|sequence| sequence.parse::<u64>().ok())
        .unwrap_or(0);
    assert_eq!(target, 7, "undo target precedes the most recent user turn");
    assert!(
        target > 5,
        "target must include the compaction in the restore scope"
    );

    // Replicate navigate_history candidate records and restore validation.
    let branch_sequence = 10;
    let branch_id = format!("history-{branch_sequence}");
    let mut candidate = records.clone();
    let mut sequence = branch_sequence;
    for event in [
        TranscriptEvent::ContextBranchCreated {
            branch_id: branch_id.clone(),
            parent_branch_id: crate::transcript::ROOT_CONTEXT_BRANCH_ID.into(),
            base_sequence: target,
            label: None,
        },
        TranscriptEvent::ContextCheckout {
            branch_id: branch_id.clone(),
            leaf_sequence: target,
        },
        TranscriptEvent::HistoryNavigation {
            operation: crate::transcript::HistoryNavigationOperation::Undo,
            target_sequence: target,
            redo_stack: vec![current],
            redo_target_sequence: None,
        },
    ] {
        candidate.push(record_at(sequence, event));
        sequence += 1;
    }
    let restored = project_runtime_restore_snapshot(
        "s".into(),
        candidate,
        SessionContextCursor {
            branch_id: Some(branch_id.clone()),
            leaf_sequence: None,
        },
        &[],
    );
    assert!(
        restored.is_ok(),
        "undo navigation must validate the root compaction on root scope, got: {restored:?}"
    );
}
