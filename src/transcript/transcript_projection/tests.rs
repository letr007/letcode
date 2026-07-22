use super::*;
use crate::agent::{
    CONTEXT_COMPACTION_DERIVED_COVERAGE_VERSION, ContextCompactionDerivedCoverage,
    ContextCompactionDerivedCoverageItem, ContextCompactionDerivedKind, ContextCompactionEvent,
    ContextCompactionSourceSpan,
};
use crate::agent::{ToolExecutionSummaryEvent, TurnStartedEvent, ValidationAdvisory};
use crate::context_tree::ContextNodeStatus;
use crate::context_view::{
    ContextBlock, ContextBlockId, ContextBlockKind, ContextBlockSource, ContextViewOperation,
    ContextViewProjection, ContextViewState, FoldedOutputMetadata, ProtectedReason,
};
use crate::evidence::{EvidenceKind, EvidenceSource};
use crate::protocol_frames::history_items_from_frames;
use crate::request_builder::HistoryToolCall;
use crate::runtime_context::RuntimeFrameKind;
use crate::tool::ToolResult;
use crate::transcript::LogicalCheckpointRetainedItemV1;
use crate::user_content::UserMessageContent;
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

fn attach_modern_compaction_coverage(
    records: &[TranscriptRecord],
    event: &mut ContextCompactionEvent,
) {
    let resolved = resolve_branch_context(
        records.to_vec(),
        SessionContextCursor {
            branch_id: None,
            leaf_sequence: None,
        },
    )
    .expect("resolved branch");
    let snapshot =
        runtime_snapshot_from_resolved_context_unbound("s", records, &resolved, None, &[])
            .expect("pre-compaction runtime snapshot");
    let spans = derive_new_retired_source_spans(records, event.tail_start_index)
        .into_iter()
        .map(|span| SourceSpan::new(span.start_sequence, span.end_sequence).expect("valid span"))
        .collect::<Vec<_>>();
    let (_, coverage) =
        derive_modern_compaction_coverage(&snapshot, &spans).expect("deterministic coverage");
    event.derived_coverage = Some(coverage);
}

fn mixed_context_view(compacted: &[&str]) -> ContextViewProjection {
    let mut projection = ContextViewProjection::default();
    for (sequence, output_id) in [(1, "fold-1"), (2, "fold-2"), (3, "fold-3")] {
        let block_id = ContextBlockId::new(format!("block-{sequence}")).expect("valid block");
        projection.blocks.insert(
            block_id.clone(),
            ContextBlock {
                block_id,
                node_id: None,
                kind: ContextBlockKind::ToolOutput,
                title: format!("block {sequence}"),
                detail: format!("detail {sequence}"),
                source: ContextBlockSource::FoldedOutput {
                    output_id: output_id.into(),
                },
                source_start_sequence: Some(sequence),
                available_sequence: Some(sequence),
                protected_reasons: Vec::new(),
                folded_output_id: Some(output_id.into()),
            },
        );
        projection.folded_outputs.insert(
            output_id.into(),
            FoldedOutputMetadata {
                output_id: output_id.into(),
                node_id: None,
                output_kind: "tool_output".into(),
                call_id: None,
                tool_name: None,
                stream: None,
                content: String::new(),
                byte_count: 0,
                line_count: 0,
                truncated: false,
                shell_command: None,
                source_start_sequence: Some(sequence),
                source_end_sequence: Some(sequence),
                available_sequence: Some(sequence),
                tool_ok: None,
                exit_status: None,
                provider_metadata: None,
                provider_fold_eligible: true,
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
fn context_and_folded_frame_ids_ignore_mixed_retirement_visibility() {
    let live = snapshot_for_context_view(&mixed_context_view(&[]));
    let mixed = snapshot_for_context_view(&mixed_context_view(&["block-2"]));
    let repeated = snapshot_for_context_view(&mixed_context_view(&["block-1", "block-2"]));

    for kind in [
        RuntimeFrameKind::ContextBlock,
        RuntimeFrameKind::FoldedOutput,
    ] {
        let live_ids = frame_ids_by_source(&live, kind);
        assert_eq!(frame_ids_by_source(&mixed, kind), live_ids);
        assert_eq!(frame_ids_by_source(&repeated, kind), live_ids);
    }
    assert!(
        mixed
            .context_view
            .provider_visible_block_ids()
            .iter()
            .all(|id| id != "block-2")
    );
    assert!(mixed.frames.iter().any(|frame| {
        frame.provenance.source_id.as_deref() == Some("block-2")
            && frame.visibility == FrameVisibility::Retired
    }));
    assert!(mixed.frames.iter().any(|frame| {
        frame.provenance.source_id.as_deref() == Some("fold-2")
            && frame.visibility == FrameVisibility::Retired
    }));

    assert!(
        mixed
            .prompt_contributors
            .iter()
            .all(|contributor| contributor.contributor_id != "context-view-active")
    );
    let contributor_ids = |snapshot: &RuntimeSnapshot, contributor_id| {
        snapshot
            .prompt_contributors
            .iter()
            .find(|contributor| contributor.contributor_id == contributor_id)
            .map(|contributor| contributor.frame_ids.clone())
            .unwrap_or_default()
    };
    mixed
        .validate_references()
        .expect("contributor references resolve");

    let restored: RuntimeSnapshot =
        serde_json::from_str(&serde_json::to_string(&mixed).expect("persist compacted snapshot"))
            .expect("restore compacted snapshot");
    assert_eq!(
        frame_ids_by_source(&restored, RuntimeFrameKind::ContextBlock),
        frame_ids_by_source(&mixed, RuntimeFrameKind::ContextBlock)
    );
    assert_eq!(
        frame_ids_by_source(&restored, RuntimeFrameKind::FoldedOutput),
        frame_ids_by_source(&mixed, RuntimeFrameKind::FoldedOutput)
    );
    assert_eq!(
        contributor_ids(&restored, "context-view-active"),
        contributor_ids(&mixed, "context-view-active")
    );
    assert_eq!(
        contributor_ids(&restored, "folded-outputs"),
        contributor_ids(&mixed, "folded-outputs")
    );
    restored
        .validate_references()
        .expect("restored contributor references resolve");
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
    assert!(
        opened
            .compaction
            .protected_frame_ids
            .contains(&opened_context_id),
        "opened detail source remains protected during compaction"
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

    let expected =
        project_session_restore_snapshot("s".into(), records.clone()).expect("default snapshot");
    let actual = build_session_context_snapshot(
        "s".into(),
        records.clone(),
        SessionContextCursor {
            branch_id: None,
            leaf_sequence: None,
        },
    )
    .expect("cursor snapshot");

    assert_eq!(actual.branch_id, ROOT_CONTEXT_BRANCH_ID);
    assert_eq!(actual.leaf_sequence, 3);
    assert_eq!(actual.records.len(), expected.records.len());
    assert_eq!(
        format!("{:?}", actual.messages),
        format!("{:?}", expected.messages)
    );
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
fn phase2_artifacts_are_isolated_by_parent_child_sibling_and_future_leaf_cursors() {
    let large = "x".repeat(crate::context_view::DEFAULT_FOLDED_OUTPUT_THRESHOLD_BYTES + 1);
    let records = vec![
        record_at(
            1,
            TranscriptEvent::ToolCallStarted {
                call_id: "parent-file".into(),
                name: crate::tool_names::TOOL_FS_READ.into(),
                args: json!({}),
            },
        ),
        record_at(
            2,
            TranscriptEvent::ToolCallFinished {
                call_id: "parent-file".into(),
                name: crate::tool_names::TOOL_FS_READ.into(),
                ok: true,
                output: ToolResult::ok(crate::tool_names::TOOL_FS_READ, json!({"content":large})),
            },
        ),
        record_at(
            3,
            TranscriptEvent::ContextBranchCreated {
                branch_id: "child".into(),
                parent_branch_id: ROOT_CONTEXT_BRANCH_ID.into(),
                base_sequence: 2,
                label: None,
            },
        ),
        branch_record_at(
            4,
            "child",
            TranscriptEvent::ToolCallStarted {
                call_id: "child-search".into(),
                name: crate::tool_names::TOOL_SEARCH_RG.into(),
                args: json!({}),
            },
        ),
        branch_record_at(
            5,
            "child",
            TranscriptEvent::ToolCallFinished {
                call_id: "child-search".into(),
                name: crate::tool_names::TOOL_SEARCH_RG.into(),
                ok: true,
                output: ToolResult::ok(
                    crate::tool_names::TOOL_SEARCH_RG,
                    json!({"matches":[{"text":large}]}),
                ),
            },
        ),
        record_at(
            6,
            TranscriptEvent::ContextBranchCreated {
                branch_id: "sibling".into(),
                parent_branch_id: ROOT_CONTEXT_BRANCH_ID.into(),
                base_sequence: 2,
                label: None,
            },
        ),
        branch_record_at(
            7,
            "sibling",
            TranscriptEvent::ToolCallStarted {
                call_id: "sibling-mcp".into(),
                name: "mcp__call".into(),
                args: json!({}),
            },
        ),
        branch_record_at(
            8,
            "sibling",
            TranscriptEvent::ToolCallFinished {
                call_id: "sibling-mcp".into(),
                name: "mcp__call".into(),
                ok: true,
                output: ToolResult::ok(
                    "mcp__call",
                    json!({"server":"s","tool":"t","content":[{"type":"text","text":large}]}),
                ),
            },
        ),
        branch_record_at(
            9,
            "child",
            TranscriptEvent::ToolCallStarted {
                call_id: "future-child-search".into(),
                name: crate::tool_names::TOOL_SEARCH_RG.into(),
                args: json!({}),
            },
        ),
        branch_record_at(
            10,
            "child",
            TranscriptEvent::ToolCallFinished {
                call_id: "future-child-search".into(),
                name: crate::tool_names::TOOL_SEARCH_RG.into(),
                ok: true,
                output: ToolResult::ok(
                    crate::tool_names::TOOL_SEARCH_RG,
                    json!({"matches":[{"text":large}]}),
                ),
            },
        ),
    ];
    let artifact_ids = |branch_id: &str, leaf_sequence: u64| {
        project_runtime_restore_snapshot(
            "s".into(),
            records.clone(),
            SessionContextCursor {
                branch_id: Some(branch_id.into()),
                leaf_sequence: Some(leaf_sequence),
            },
            &[],
        )
        .expect("cursor projection")
        .snapshot
        .context_view
        .folded_outputs
        .keys()
        .cloned()
        .collect::<Vec<_>>()
    };
    assert_eq!(
        artifact_ids(ROOT_CONTEXT_BRANCH_ID, 2),
        vec![
            "folded-output-seq-2-content",
            "folded-output-seq-2-tool-result",
        ]
    );
    assert_eq!(
        artifact_ids("child", 5),
        vec![
            "folded-output-seq-2-content",
            "folded-output-seq-2-tool-result",
            "folded-output-seq-5-matches",
            "folded-output-seq-5-tool-result",
        ]
    );
    assert_eq!(
        artifact_ids("sibling", 8),
        vec![
            "folded-output-seq-2-content",
            "folded-output-seq-2-tool-result",
            "folded-output-seq-8-text",
            "folded-output-seq-8-tool-result",
        ]
    );
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
                retired_source_spans: Vec::new(),
                frame_identity_bindings: Vec::new(),
                derived_coverage: None,
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
fn compaction_projection_rejects_malformed_modern_fields_but_reads_legacy_shape() {
    let prefix = || {
        vec![
            record_at(
                1,
                TranscriptEvent::UserMessage {
                    content: UserMessageContent::from("old"),
                },
            ),
            record_at(
                2,
                TranscriptEvent::AssistantMessage {
                    content: "tail".into(),
                },
            ),
        ]
    };
    let project = |event| {
        let mut records = prefix();
        records.push(record_at(3, TranscriptEvent::ContextCompaction(event)));
        build_session_context_snapshot(
            "s".into(),
            records,
            SessionContextCursor {
                branch_id: None,
                leaf_sequence: None,
            },
        )
    };

    let legacy = project(ContextCompactionEvent {
        outcome: "succeeded".into(),
        summary: "legacy summary".into(),
        tail_start_index: 1,
        original_history_items: 2,
        retained_history_items: 1,
        retired_source_spans: Vec::new(),
        frame_identity_bindings: Vec::new(),
        derived_coverage: None,
        detail: None,
    })
    .expect("legacy span-less compaction remains readable");
    assert!(matches!(
        legacy.history.as_slice(),
        [HistoryItem::ContextSummary { text }, HistoryItem::AssistantText { text: tail }]
            if text == "legacy summary" && tail == "tail"
    ));

    let modern_spans = vec![ContextCompactionSourceSpan {
        start_sequence: 1,
        end_sequence: 1,
    }];
    let retained_error = project(ContextCompactionEvent {
        outcome: "succeeded".into(),
        summary: "summary".into(),
        tail_start_index: 1,
        original_history_items: 2,
        retained_history_items: 1,
        retired_source_spans: modern_spans.clone(),
        frame_identity_bindings: Vec::new(),
        derived_coverage: None,
        detail: None,
    })
    .expect_err("modern retained count must include the summary");
    assert!(
        retained_error
            .to_string()
            .contains("retained_history_items is inconsistent with summary and tail")
    );

    let original_count_error = project(ContextCompactionEvent {
        outcome: "succeeded".into(),
        summary: "summary".into(),
        tail_start_index: 1,
        original_history_items: 3,
        retained_history_items: 2,
        retired_source_spans: modern_spans.clone(),
        frame_identity_bindings: Vec::new(),
        derived_coverage: None,
        detail: None,
    })
    .expect_err("modern original count must match the visible branch history");
    assert!(
        original_count_error
            .to_string()
            .contains("original_history_items is inconsistent with visible history")
    );

    let tail_error = project(ContextCompactionEvent {
        outcome: "succeeded".into(),
        summary: "summary".into(),
        tail_start_index: 3,
        original_history_items: 2,
        retained_history_items: 0,
        retired_source_spans: modern_spans,
        frame_identity_bindings: Vec::new(),
        derived_coverage: None,
        detail: None,
    })
    .expect_err("modern tail index must be in the visible branch history");
    assert!(
        tail_error
            .to_string()
            .contains("tail_start_index exceeds original history")
    );
}

#[test]
fn legacy_spanless_compaction_retains_protocol_frames_with_historical_ids() {
    let prefix = vec![
        record_at(
            1,
            TranscriptEvent::UserMessage {
                content: UserMessageContent::from("old user"),
            },
        ),
        record_at(
            2,
            TranscriptEvent::AssistantMessage {
                content: "retained tail".into(),
            },
        ),
    ];
    let before_compaction = project_runtime_restore_snapshot(
        "s".into(),
        prefix.clone(),
        SessionContextCursor {
            branch_id: None,
            leaf_sequence: None,
        },
        &[],
    )
    .expect("prefix runtime projection");
    let old_user_id = before_compaction
        .snapshot
        .frames
        .iter()
        .find(|frame| {
            frame.kind == RuntimeFrameKind::User
                && frame.provenance.source_span.map(|span| span.start_sequence) == Some(1)
        })
        .expect("old user frame")
        .id;
    let tail_id = before_compaction
        .snapshot
        .frames
        .iter()
        .find(|frame| {
            frame.kind == RuntimeFrameKind::Assistant
                && frame.provenance.source_span.map(|span| span.start_sequence) == Some(2)
        })
        .expect("tail frame")
        .id;

    let mut records = prefix;
    records.push(record_at(
        3,
        TranscriptEvent::ContextCompaction(ContextCompactionEvent {
            outcome: "succeeded".into(),
            summary: "legacy summary".into(),
            tail_start_index: 1,
            original_history_items: 2,
            retained_history_items: 1,
            retired_source_spans: Vec::new(),
            frame_identity_bindings: Vec::new(),
            derived_coverage: None,
            detail: None,
        }),
    ));

    let restored = project_runtime_restore_snapshot(
        "s".into(),
        records,
        SessionContextCursor {
            branch_id: None,
            leaf_sequence: None,
        },
        &[],
    )
    .expect("legacy runtime projection");

    assert!(restored.snapshot.frames.iter().any(|frame| {
        frame.id == old_user_id
            && frame.visibility == FrameVisibility::Retired
            && frame.protocol.is_some()
    }));
    assert!(restored.snapshot.frames.iter().any(|frame| {
        frame.id == tail_id
            && frame.visibility == FrameVisibility::Active
            && frame.protocol.is_some()
    }));
    assert!(
        restored
            .snapshot
            .compaction
            .compacted_frame_ids
            .contains(&old_user_id)
    );
}

#[test]
fn modern_compaction_spans_must_equal_the_canonical_retired_source_closure() {
    let records = vec![
        record_at(
            1,
            TranscriptEvent::UserMessage {
                content: UserMessageContent::from("old user"),
            },
        ),
        record_at(
            2,
            TranscriptEvent::ReasoningMessage {
                content: "dependent raw record".into(),
            },
        ),
        record_at(
            3,
            TranscriptEvent::AssistantMessage {
                content: "old assistant".into(),
            },
        ),
        record_at(
            4,
            TranscriptEvent::AssistantMessage {
                content: "retained tail".into(),
            },
        ),
    ];
    let event = |retired_source_spans| {
        let mut event = ContextCompactionEvent {
            outcome: "succeeded".into(),
            summary: "summary".into(),
            tail_start_index: 2,
            original_history_items: 3,
            retained_history_items: 2,
            retired_source_spans,
            frame_identity_bindings: Vec::new(),
            derived_coverage: None,
            detail: None,
        };
        attach_modern_compaction_coverage(&records, &mut event);
        event
    };
    let canonical = vec![ContextCompactionSourceSpan {
        start_sequence: 1,
        end_sequence: 3,
    }];

    validate_context_compaction_event(&records, &event(canonical.clone()))
        .expect("the deterministic closure includes the dependent raw record");
    for invalid in [
        vec![ContextCompactionSourceSpan {
            start_sequence: 1,
            end_sequence: 4,
        }],
        vec![ContextCompactionSourceSpan {
            start_sequence: 1,
            end_sequence: 2,
        }],
        vec![
            ContextCompactionSourceSpan {
                start_sequence: 1,
                end_sequence: 1,
            },
            ContextCompactionSourceSpan {
                start_sequence: 2,
                end_sequence: 3,
            },
        ],
    ] {
        assert!(validate_context_compaction_event(&records, &event(invalid)).is_err());
    }
}

#[test]
fn compaction_binding_replay_rejects_prior_id_collision_before_current_remap() {
    let mut first_event = ContextCompactionEvent {
        outcome: "succeeded".into(),
        summary: "first summary".into(),
        tail_start_index: 1,
        original_history_items: 2,
        retained_history_items: 2,
        retired_source_spans: vec![ContextCompactionSourceSpan {
            start_sequence: 1,
            end_sequence: 1,
        }],
        frame_identity_bindings: Vec::new(),
        derived_coverage: None,
        detail: None,
    };
    let mut records = vec![
        record_at(
            1,
            TranscriptEvent::UserMessage {
                content: UserMessageContent::from("old user"),
            },
        ),
        record_at(
            2,
            TranscriptEvent::AssistantMessage {
                content: "retained reply".into(),
            },
        ),
    ];
    attach_modern_compaction_coverage(&records, &mut first_event);
    records.push(record_at(
        3,
        TranscriptEvent::ContextCompaction(first_event),
    ));
    records.push(record_at(
        4,
        TranscriptEvent::UserMessage {
            content: UserMessageContent::from("new user"),
        },
    ));
    let unbound_snapshot = |records: &[TranscriptRecord]| {
        let resolved = resolve_branch_context(
            records.to_vec(),
            SessionContextCursor {
                branch_id: None,
                leaf_sequence: None,
            },
        )
        .expect("resolved root branch");
        let latest_model = restore_latest_model_projection(&resolved.records);
        runtime_snapshot_from_resolved_context_unbound(
            "s",
            records,
            &resolved,
            latest_model.as_deref(),
            &[],
        )
        .expect("unbound runtime snapshot")
    };

    let first_snapshot = unbound_snapshot(&records[..3]);
    let mut first_bindings = compaction_frame_identity_bindings(&first_snapshot);
    let second_retired_source_spans = canonical_cumulative_retired_source_spans(
        vec![ContextCompactionSourceSpan {
            start_sequence: 1,
            end_sequence: 1,
        }],
        derive_new_retired_source_spans(&records, 3),
    );
    let mut second_event = ContextCompactionEvent {
        outcome: "succeeded".into(),
        summary: "second summary".into(),
        tail_start_index: 3,
        original_history_items: 3,
        retained_history_items: 1,
        retired_source_spans: second_retired_source_spans,
        frame_identity_bindings: Vec::new(),
        derived_coverage: None,
        detail: None,
    };
    attach_modern_compaction_coverage(&records, &mut second_event);
    records.push(record_at(
        5,
        TranscriptEvent::ContextCompaction(second_event),
    ));
    let second_snapshot = unbound_snapshot(&records);
    let new_user = second_snapshot
        .frames
        .iter()
        .find(|frame| {
            frame.provenance.source_span
                == Some(SourceSpan {
                    start_sequence: 4,
                    end_sequence: 4,
                })
        })
        .expect("new user frame");
    let summary_key = first_snapshot
        .frames
        .iter()
        .find(|frame| frame.provenance.source == RuntimeSource::SummaryArtifact)
        .expect("first summary frame")
        .durable_identity_key();
    let new_user_key = new_user.durable_identity_key();
    let colliding_id = new_user.id.as_u64();
    first_bindings
        .iter_mut()
        .find(|binding| binding.key == summary_key)
        .expect("first summary binding")
        .frame_id = colliding_id;

    let mut second_bindings = compaction_frame_identity_bindings(&second_snapshot);
    let remapped_new_user_id = u64::MAX;
    assert!(
        second_snapshot
            .frames
            .iter()
            .all(|frame| frame.id.as_u64() != remapped_new_user_id)
    );
    second_bindings
        .iter_mut()
        .find(|binding| binding.key == summary_key)
        .expect("second summary binding")
        .frame_id = colliding_id;
    second_bindings
        .iter_mut()
        .find(|binding| binding.key == new_user_key)
        .expect("new user binding")
        .frame_id = remapped_new_user_id;
    match &mut records[2].event {
        TranscriptEvent::ContextCompaction(event) => {
            event.frame_identity_bindings = first_bindings;
        }
        _ => unreachable!("first compaction record"),
    }
    match &mut records[4].event {
        TranscriptEvent::ContextCompaction(event) => {
            event.frame_identity_bindings = second_bindings;
        }
        _ => unreachable!("second compaction record"),
    }

    let error = project_runtime_restore_snapshot(
        "s".into(),
        records,
        SessionContextCursor {
            branch_id: None,
            leaf_sequence: None,
        },
        &[],
    )
    .expect_err("prior binding collision must fail before the current remap");
    assert!(
        error
            .to_string()
            .contains("runtime snapshot contains duplicate frame id")
    );
}

#[test]
fn derive_retired_source_spans_tracks_history_sources_before_tail() {
    let records = vec![
        record_at(
            1,
            TranscriptEvent::UserMessage {
                content: UserMessageContent::from("old user"),
            },
        ),
        record_at(
            2,
            TranscriptEvent::AssistantMessage {
                content: "old assistant".into(),
            },
        ),
        record_at(
            3,
            TranscriptEvent::ToolCallStarted {
                call_id: "call-1".into(),
                name: "shell__exec".into(),
                args: serde_json::json!({"command": "cargo test"}),
            },
        ),
        record_at(
            4,
            TranscriptEvent::ToolCallFinished {
                call_id: "call-1".into(),
                name: "shell__exec".into(),
                ok: true,
                output: crate::tool::ToolResult::ok("shell__exec", serde_json::json!({})),
            },
        ),
    ];

    let spans = derive_retired_source_spans(&records, 3);
    assert_eq!(spans.len(), 1);
    assert_eq!(spans[0].start_sequence, 1);
    assert_eq!(spans[0].end_sequence, 3);
}

#[test]
fn derive_retired_source_spans_covers_non_history_records_between_retired_sources() {
    let records = vec![
        record_at(
            1,
            TranscriptEvent::UserMessage {
                content: UserMessageContent::from("old user"),
            },
        ),
        record_at(
            2,
            TranscriptEvent::ReasoningMessage {
                content: "old reasoning".into(),
            },
        ),
        record_at(
            3,
            TranscriptEvent::AssistantMessage {
                content: "old assistant".into(),
            },
        ),
        record_at(
            4,
            TranscriptEvent::AssistantMessage {
                content: "retained tail".into(),
            },
        ),
    ];

    let spans = derive_retired_source_spans(&records, 2);
    assert_eq!(spans.len(), 1);
    assert_eq!(spans[0].start_sequence, 1);
    assert_eq!(spans[0].end_sequence, 3);
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

    let snapshot = project_session_restore_snapshot("s".into(), records).expect("linear snapshot");

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
                retired_source_spans: Vec::new(),
                frame_identity_bindings: Vec::new(),
                derived_coverage: None,
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

#[test]
fn runtime_snapshot_projection_collects_context_view_tree_evidence_and_compaction() {
    let mut records = vec![
        record_at(
            1,
            TranscriptEvent::SessionStarted {
                model: "gpt-5".into(),
            },
        ),
        metadata_record_at(
            2,
            TranscriptEvent::ContextNodeCreated {
                node_id: "child".into(),
                parent_node_id: Some("root".into()),
                label: Some("Task".into()),
                purpose: Some("Projection test".into()),
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
                node_id: "child".into(),
                status: ContextNodeStatus::Active,
            },
        ),
        record_at(
            5,
            TranscriptEvent::UserMessage {
                content: UserMessageContent::from("Do not skip tests"),
            },
        ),
        record_at(
            6,
            TranscriptEvent::ToolCallStarted {
                call_id: "call-1".into(),
                name: "shell__exec".into(),
                args: json!({"command":"cargo check"}),
            },
        ),
        record_at(
            7,
            TranscriptEvent::ToolCallFinished {
                call_id: "call-1".into(),
                name: "shell__exec".into(),
                ok: true,
                output: ToolResult::ok("shell__exec", json!({"stdout": "done", "status": 0})),
            },
        ),
        metadata_record_at(
            8,
            TranscriptEvent::FoldedOutputMetadata {
                node_id: Some("child".into()),
                output_id: "fold-1".into(),
                output_kind: "shell_output".into(),
                call_id: Some("call-1".into()),
                tool_name: Some("shell__exec".into()),
                stream: Some("stdout".into()),
                content: Some("done".into()),
                byte_count: Some(4),
                line_count: Some(1),
                truncated: Some(false),
                shell_command: Some("cargo check".into()),
                source_start_sequence: Some(7),
                source_end_sequence: Some(7),
                tool_ok: Some(true),
                exit_status: Some(0),
                provider_metadata: None,
                provider_fold_eligible: None,
            },
        ),
        metadata_record_at(
            9,
            TranscriptEvent::ContextViewOperationMetadata {
                operation: "pin".into(),
                node_id: Some("child".into()),
                block_id: Some("block-seq-7-folded-output-fold-1".into()),
                detail: None,
            },
        ),
        metadata_record_at(
            10,
            TranscriptEvent::ContextViewOperationMetadata {
                operation: "open_detail".into(),
                node_id: Some("child".into()),
                block_id: Some("block-seq-7-folded-output-fold-1".into()),
                detail: None,
            },
        ),
        record_at(
            11,
            TranscriptEvent::Evidence {
                id: "ev-1".into(),
                evidence_kind: EvidenceKind::Validation,
                title: "cargo check".into(),
                summary: "cargo check passed".into(),
                detail: None,
                source: EvidenceSource::Command {
                    command: "cargo check".into(),
                    status: Some(0),
                },
                tags: vec!["validation".into()],
            },
        ),
        metadata_record_at(
            12,
            TranscriptEvent::ContextSummaryArtifactMetadata {
                node_id: "child".into(),
                artifact_id: "sum-1".into(),
                artifact_kind: "summary".into(),
                version: Some(1),
                summary: Some("Condensed context".into()),
                source_node_id: Some("child".into()),
                source_block_id: None,
                source_start_sequence: Some(5),
                source_end_sequence: Some(7),
            },
        ),
        record_at(
            13,
            TranscriptEvent::ContextCompaction(ContextCompactionEvent {
                outcome: "succeeded".into(),
                summary: "Earlier context retired".into(),
                tail_start_index: 1,
                original_history_items: 3,
                retained_history_items: 3,
                retired_source_spans: vec![ContextCompactionSourceSpan {
                    start_sequence: 5,
                    end_sequence: 5,
                }],
                frame_identity_bindings: Vec::new(),
                derived_coverage: None,
                detail: None,
            }),
        ),
    ];

    let mut event = match records.pop().expect("compaction record").event {
        TranscriptEvent::ContextCompaction(event) => event,
        _ => panic!("expected context compaction"),
    };
    attach_modern_compaction_coverage(&records, &mut event);
    records.push(record_at(13, TranscriptEvent::ContextCompaction(event)));

    let child_sessions = vec![ChildSessionSummary {
        parent_session_id: "s".into(),
        parent_run_id: "run-1".into(),
        child_session_id: "child-session-1".into(),
        agent_name: "explorer".into(),
        status: "completed".into(),
        summary: "Looked up compile state".into(),
        timestamp_ms: 42,
    }];

    let projected = project_runtime_restore_snapshot(
        "s".into(),
        records,
        SessionContextCursor {
            branch_id: None,
            leaf_sequence: None,
        },
        &child_sessions,
    )
    .expect("runtime snapshot projection");

    assert_eq!(projected.snapshot.latest_model.as_deref(), Some("gpt-5"));
    assert_eq!(projected.snapshot.leaf_sequence, Some(13));
    assert_eq!(
        projected
            .snapshot
            .context_tree
            .active_node_id()
            .map(|id| id.as_str()),
        Some("child")
    );
    assert_eq!(projected.snapshot.evidence.len(), 1);
    assert_eq!(projected.snapshot.child_sessions.len(), 1);
    assert_eq!(projected.snapshot.folded_outputs.len(), 1);
    assert_eq!(projected.snapshot.compaction.retired_source_spans.len(), 1);
    assert_eq!(
        history_items_from_frames(&projected.protocol_frames),
        restore_session_history_projection(&projected.records)
    );
    assert_eq!(
        projected.protocol_frames,
        projected.snapshot.active_protocol_frames(),
        "runtime restore protocol frames must retain transcript frame identity"
    );
    let compaction_summary = projected
        .snapshot
        .frames
        .iter()
        .find(|frame| {
            matches!(
                frame.protocol,
                Some(crate::protocol_frames::ProtocolFrameItem::ContextSummary { .. })
            )
        })
        .expect("restored active compaction summary frame");
    assert_eq!(
        compaction_summary.provenance.source,
        RuntimeSource::SummaryArtifact
    );
    assert_eq!(compaction_summary.provenance.source_span, None);
    assert_eq!(
        projected
            .snapshot
            .active_context
            .open_detail_block_id
            .as_deref(),
        Some("block-seq-7-folded-output-fold-1")
    );
    assert_eq!(
        projected.snapshot.active_context.pinned_block_ids,
        vec!["block-seq-7-folded-output-fold-1".to_string()]
    );
    assert!(
        projected
            .snapshot
            .frames
            .iter()
            .any(|frame| frame.kind == RuntimeFrameKind::Summary
                && frame.summary.as_deref() == Some("Condensed context"))
    );
    assert!(
        projected
            .snapshot
            .prompt_contributors
            .iter()
            .any(|contributor| contributor.contributor_id == "folded-outputs")
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
        matches!(restored.last(), Some(HistoryItem::ToolOutput { call_id, output_json })
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
            TranscriptEvent::AssistantToolCallBatch { text: None, calls },
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
fn non_root_malformed_checkpoint_and_compaction_are_rejected() {
    let mut malformed_checkpoint = non_root_checkpoint_journal();
    let TranscriptEvent::LogicalCheckpoint(checkpoint) = &mut malformed_checkpoint[5].event else {
        panic!("child checkpoint record");
    };
    checkpoint.schema_version = 0;
    assert!(
        project_runtime_restore_snapshot(
            "s".into(),
            malformed_checkpoint,
            SessionContextCursor {
                branch_id: Some("child".into()),
                leaf_sequence: None,
            },
            &[],
        )
        .is_err()
    );

    let mut malformed_compaction = non_root_checkpoint_journal();
    malformed_compaction.pop();
    malformed_compaction.push(branch_record_at(
        6,
        "child",
        TranscriptEvent::ContextCompaction(ContextCompactionEvent {
            outcome: "succeeded".into(),
            summary: "modern summary".into(),
            tail_start_index: 0,
            original_history_items: 0,
            retained_history_items: 1,
            retired_source_spans: vec![ContextCompactionSourceSpan {
                start_sequence: 4,
                end_sequence: 4,
            }],
            frame_identity_bindings: Vec::new(),
            derived_coverage: None,
            detail: None,
        }),
    ));
    assert!(
        project_runtime_restore_snapshot(
            "s".into(),
            malformed_compaction,
            SessionContextCursor {
                branch_id: Some("child".into()),
                leaf_sequence: None,
            },
            &[],
        )
        .is_err()
    );
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
fn successful_compaction_validation_uses_the_event_branch_not_latest_checkout() {
    let records = vec![
        record_at(
            1,
            TranscriptEvent::UserMessage {
                content: UserMessageContent::from("root base"),
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
        metadata_record_at(
            4,
            TranscriptEvent::ContextCheckout {
                branch_id: ROOT_CONTEXT_BRANCH_ID.into(),
                leaf_sequence: 1,
            },
        ),
        branch_record_at(
            5,
            "child",
            TranscriptEvent::UserMessage {
                content: UserMessageContent::from("child old"),
            },
        ),
        branch_record_at(
            6,
            "child",
            TranscriptEvent::AssistantMessage {
                content: "child tail".into(),
            },
        ),
        branch_record_at(
            7,
            "child",
            TranscriptEvent::ContextCompaction(ContextCompactionEvent {
                outcome: "succeeded".into(),
                summary: "child summary".into(),
                tail_start_index: 2,
                original_history_items: 3,
                retained_history_items: 1,
                retired_source_spans: Vec::new(),
                frame_identity_bindings: Vec::new(),
                derived_coverage: None,
                detail: None,
            }),
        ),
    ];

    validate_successful_compactions(&records).expect("validate child compaction scope");
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
fn checkpoint_audits_accept_canonical_tool_and_explicit_artifacts_only_in_scope() {
    let (mut records, mut checkpoint) = checkpoint_candidate_fixture();
    let payload = "x".repeat(crate::context_view::DEFAULT_FOLDED_OUTPUT_THRESHOLD_BYTES + 1);
    records.push(metadata_record_at(
        3,
        TranscriptEvent::FoldedOutputMetadata {
            node_id: None,
            output_id: "explicit-output".into(),
            output_kind: "shell_output".into(),
            call_id: None,
            tool_name: None,
            stream: None,
            content: Some("metadata artifact".into()),
            byte_count: None,
            line_count: None,
            truncated: None,
            shell_command: None,
            source_start_sequence: Some(4),
            source_end_sequence: Some(5),
            tool_ok: None,
            exit_status: None,
            provider_metadata: None,
            provider_fold_eligible: None,
        },
    ));
    records.push(record_at(
        4,
        TranscriptEvent::AssistantToolCallBatch {
            text: None,
            calls: vec![HistoryToolCall {
                call_id: "read-1".into(),
                name: crate::tool_names::TOOL_FS_READ.into(),
                arguments_json: "{}".into(),
            }],
        },
    ));
    records.push(record_at(
        5,
        TranscriptEvent::ToolCallFinished {
            call_id: "read-1".into(),
            name: crate::tool_names::TOOL_FS_READ.into(),
            ok: true,
            output: ToolResult::ok(
                crate::tool_names::TOOL_FS_READ,
                json!({"content": payload, "path": "src/lib.rs"}),
            ),
        },
    ));
    checkpoint.boundary_sequence = 5;
    checkpoint.covered_source_spans = vec![
        LogicalCheckpointSourceSpanV1 {
            start_sequence: 1,
            end_sequence: 1,
        },
        LogicalCheckpointSourceSpanV1 {
            start_sequence: 4,
            end_sequence: 5,
        },
    ];
    checkpoint.retained_items = vec![LogicalCheckpointRetainedItemV1 {
        kind: crate::transcript::LogicalCheckpointRetainedKindV1::UserRequirement,
        title: "Canonical artifact".into(),
        detail: "large file read".into(),
        audit_source: LogicalCheckpointAuditSourceV1::FoldedOutputAudit {
            output_id: "folded-output-seq-5-content".into(),
            start_sequence: 5,
            end_sequence: 5,
        },
    }];
    validate_logical_checkpoint_candidate(
        "s",
        &records,
        Some(ROOT_CONTEXT_BRANCH_ID.into()),
        5,
        6,
        &checkpoint,
    )
    .expect("canonical ToolCallFinished artifact is accepted");

    let mut explicit = checkpoint.clone();
    explicit.retained_items[0].audit_source = LogicalCheckpointAuditSourceV1::FoldedOutputAudit {
        output_id: "explicit-output".into(),
        start_sequence: 4,
        end_sequence: 5,
    };
    validate_logical_checkpoint_candidate(
        "s",
        &records,
        Some(ROOT_CONTEXT_BRANCH_ID.into()),
        5,
        6,
        &explicit,
    )
    .expect("in-scope explicit metadata artifact is accepted");

    for (output_id, start, end) in [
        ("folded-output-seq-5-content", 1, 1),
        ("future-output", 4, 5),
    ] {
        let mut invalid = checkpoint.clone();
        invalid.retained_items[0].audit_source =
            LogicalCheckpointAuditSourceV1::FoldedOutputAudit {
                output_id: output_id.into(),
                start_sequence: start,
                end_sequence: end,
            };
        assert!(
            validate_logical_checkpoint_candidate(
                "s",
                &records,
                Some(ROOT_CONTEXT_BRANCH_ID.into()),
                5,
                6,
                &invalid,
            )
            .is_err(),
            "{output_id} with span {start}..={end} must reject"
        );
    }
}

#[test]
fn checkpoint_replay_keeps_logical_frame_identity_and_retires_closed_segments() {
    let (mut records, first) = checkpoint_candidate_fixture();
    records.push(branch_record_at(
        3,
        ROOT_CONTEXT_BRANCH_ID,
        TranscriptEvent::LogicalCheckpoint(first),
    ));
    records.push(branch_record_at(
        4,
        ROOT_CONTEXT_BRANCH_ID,
        TranscriptEvent::AssistantMessage {
            content: "second segment".into(),
        },
    ));
    let second = LogicalCheckpointEventV1 {
        schema_version: 1,
        checkpoint_id: "checkpoint-2".into(),
        turn_id: 7,
        previous_segment_id: 1,
        segment_id: 2,
        previous_checkpoint_id: Some("checkpoint-1".into()),
        boundary_sequence: 4,
        context_scope_revision: 0,
        covered_source_spans: vec![LogicalCheckpointSourceSpanV1 {
            start_sequence: 3,
            end_sequence: 4,
        }],
        retained_items: Vec::new(),
    };
    records.push(branch_record_at(
        5,
        ROOT_CONTEXT_BRANCH_ID,
        TranscriptEvent::LogicalCheckpoint(second),
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
        .expect("replay checkpoints")
    };
    let first_restore = restore();
    let second_restore = restore();
    assert_eq!(
        first_restore.protocol_frames,
        second_restore.protocol_frames
    );
    assert_eq!(
        first_restore.snapshot.active_protocol_frames(),
        second_restore.snapshot.active_protocol_frames()
    );
    assert_eq!(first_restore.snapshot.current_segment_id, Some(2));
    assert!(
        first_restore
            .snapshot
            .compaction
            .retired_source_spans
            .iter()
            .any(|span| span.start_sequence == 1 && span.end_sequence == 1)
    );
    assert!(
        first_restore
            .snapshot
            .compaction
            .retired_source_spans
            .iter()
            .any(|span| span.start_sequence == 3 && span.end_sequence == 4)
    );
    assert!(
        first_restore
            .snapshot
            .active_history_items()
            .iter()
            .all(|item| {
                !matches!(item, HistoryItem::AssistantText { text } if text == "second segment")
            })
    );
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
fn canonical_finished_artifacts_cover_file_search_mcp_and_shell_outputs() {
    let large = "x".repeat(crate::context_view::DEFAULT_FOLDED_OUTPUT_THRESHOLD_BYTES + 1);
    let cases = [
        (
            crate::tool_names::TOOL_FS_READ,
            json!({"content": large}),
            "content",
        ),
        (
            crate::tool_names::TOOL_SEARCH_RG,
            json!({"matches": [{"text": large}]}),
            "matches",
        ),
        (
            "mcp__github__search",
            json!({"server": "github", "tool": "search", "content": [{"type": "text", "text": large}]}),
            "text",
        ),
        (
            crate::tool_names::TOOL_SHELL_EXEC,
            json!({"status": 0, "stdout": large, "stdout_truncated": false}),
            "stdout",
        ),
    ];
    for (name, data, stream) in cases {
        let outputs = crate::context_view::restore_folded_outputs(
            &[
                record_at(
                    1,
                    TranscriptEvent::ToolCallStarted {
                        call_id: "call".into(),
                        name: name.into(),
                        args: json!({"command": "test"}),
                    },
                ),
                record_at(
                    2,
                    TranscriptEvent::ToolCallFinished {
                        call_id: "call".into(),
                        name: name.into(),
                        ok: true,
                        output: ToolResult::ok(name, data),
                    },
                ),
            ],
            crate::context_view::DEFAULT_FOLDED_OUTPUT_THRESHOLD_BYTES,
        )
        .expect("derive canonical output");
        let output = outputs
            .get(&format!("folded-output-seq-2-{stream}"))
            .expect("canonical output id");
        assert_eq!(output.tool_name.as_deref(), Some(name));
        assert_eq!(output.source_start_sequence, Some(2));
        assert_eq!(output.source_end_sequence, Some(2));
    }
}

#[test]
fn checkpoint_audit_rejects_duplicate_artifacts_and_accepts_scoped_multi_record_metadata() {
    let duplicate = vec![
        metadata_record_at(
            1,
            TranscriptEvent::FoldedOutputMetadata {
                node_id: None,
                output_id: "same".into(),
                output_kind: "text".into(),
                call_id: None,
                tool_name: None,
                stream: None,
                content: Some("one".into()),
                byte_count: None,
                line_count: None,
                truncated: None,
                shell_command: None,
                source_start_sequence: Some(1),
                source_end_sequence: Some(1),
                tool_ok: None,
                exit_status: None,
                provider_metadata: None,
                provider_fold_eligible: None,
            },
        ),
        metadata_record_at(
            2,
            TranscriptEvent::FoldedOutputMetadata {
                node_id: None,
                output_id: "same".into(),
                output_kind: "text".into(),
                call_id: None,
                tool_name: None,
                stream: None,
                content: Some("two".into()),
                byte_count: None,
                line_count: None,
                truncated: None,
                shell_command: None,
                source_start_sequence: Some(2),
                source_end_sequence: Some(2),
                tool_ok: None,
                exit_status: None,
                provider_metadata: None,
                provider_fold_eligible: None,
            },
        ),
    ];
    assert!(crate::context_view::restore_folded_outputs(&duplicate, 1).is_err());

    let (mut records, mut checkpoint) = checkpoint_candidate_fixture();
    records.push(metadata_record_at(
        3,
        TranscriptEvent::FoldedOutputMetadata {
            node_id: None,
            output_id: "scoped-output".into(),
            output_kind: "file_content".into(),
            call_id: Some("call".into()),
            tool_name: Some(crate::tool_names::TOOL_FS_READ.into()),
            stream: Some("content".into()),
            content: Some("small".into()),
            byte_count: None,
            line_count: None,
            truncated: None,
            shell_command: None,
            source_start_sequence: Some(4),
            source_end_sequence: Some(5),
            tool_ok: Some(true),
            exit_status: None,
            provider_metadata: Some(json!({"path":"src/lib.rs"})),
            provider_fold_eligible: None,
        },
    ));
    records.push(record_at(
        4,
        TranscriptEvent::AssistantToolCallBatch {
            text: None,
            calls: vec![HistoryToolCall {
                call_id: "call".into(),
                name: crate::tool_names::TOOL_FS_READ.into(),
                arguments_json: "{}".into(),
            }],
        },
    ));
    records.push(record_at(
        5,
        TranscriptEvent::ToolCallFinished {
            call_id: "call".into(),
            name: crate::tool_names::TOOL_FS_READ.into(),
            ok: true,
            output: ToolResult::ok(crate::tool_names::TOOL_FS_READ, json!({"content": "small"})),
        },
    ));
    checkpoint.boundary_sequence = 5;
    checkpoint.covered_source_spans = vec![
        LogicalCheckpointSourceSpanV1 {
            start_sequence: 1,
            end_sequence: 1,
        },
        LogicalCheckpointSourceSpanV1 {
            start_sequence: 4,
            end_sequence: 5,
        },
    ];
    checkpoint.retained_items = vec![LogicalCheckpointRetainedItemV1 {
        kind: crate::transcript::LogicalCheckpointRetainedKindV1::UserRequirement,
        title: "Scoped artifact".into(),
        detail: "metadata spans the completed call".into(),
        audit_source: LogicalCheckpointAuditSourceV1::FoldedOutputAudit {
            output_id: "scoped-output".into(),
            start_sequence: 4,
            end_sequence: 5,
        },
    }];
    validate_logical_checkpoint_candidate(
        "s",
        &records,
        Some(ROOT_CONTEXT_BRANCH_ID.into()),
        5,
        6,
        &checkpoint,
    )
    .expect("fully visible multi-record artifact is a valid audit source");
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
    assert!(
        before
            .snapshot
            .compaction
            .retired_source_spans
            .iter()
            .any(|span| span.start_sequence == 1 && span.end_sequence == 1)
    );
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
            TranscriptEvent::ContextCompaction(ContextCompactionEvent {
                outcome: "succeeded".into(),
                summary: "old summary".into(),
                tail_start_index: 1,
                original_history_items: 2,
                retained_history_items: 2,
                retired_source_spans: vec![ContextCompactionSourceSpan {
                    start_sequence: 1,
                    end_sequence: 1,
                }],
                frame_identity_bindings: Vec::new(),
                derived_coverage: None,
                detail: None,
            }),
        ),
    ];
    assert_eq!(
        restore_retired_source_spans_projection(&historical),
        vec![ContextCompactionSourceSpan {
            start_sequence: 1,
            end_sequence: 1
        }]
    );
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

#[test]
fn production_projection_compaction_retires_eligible_turn_and_replays() {
    let mut records = vec![
        record_at(
            1,
            TranscriptEvent::TurnStarted(TurnStartedEvent {
                turn_id: 1,
                intent: "implement".into(),
                directive: "write and validate".into(),
                validation_reminder: "run tests".into(),
            }),
        ),
        record_at(
            2,
            TranscriptEvent::UserMessage {
                content: UserMessageContent::from("historical requirement"),
            },
        ),
        record_at(
            3,
            TranscriptEvent::ToolCallStarted {
                call_id: "write-1".into(),
                name: "fs__write".into(),
                args: json!({"path":"src/lib.rs","content":"fn updated() {}"}),
            },
        ),
        record_at(
            4,
            TranscriptEvent::ToolCallFinished {
                call_id: "write-1".into(),
                name: "fs__write".into(),
                ok: true,
                output: ToolResult::ok(
                    "fs__write",
                    json!({"path":"src/lib.rs","bytes_written":15}),
                ),
            },
        ),
        record_at(
            5,
            TranscriptEvent::ToolExecutionSummary(ToolExecutionSummaryEvent {
                turn_id: 1,
                call_id: "write-1".into(),
                name: "fs__write".into(),
                status: "completed".into(),
                rejection: None,
                effect_kind: "write".into(),
                primary_path: Some("src/lib.rs".into()),
                command: None,
            }),
        ),
        record_at(
            6,
            TranscriptEvent::ToolCallStarted {
                call_id: "test-1".into(),
                name: "shell__exec".into(),
                args: json!({"command":"cargo test"}),
            },
        ),
        record_at(
            7,
            TranscriptEvent::ToolCallFinished {
                call_id: "test-1".into(),
                name: "shell__exec".into(),
                ok: true,
                output: ToolResult::ok(
                    "shell__exec",
                    json!({"status":0,"stdout":"test result: ok","stdout_truncated":false,"stderr":"","stderr_truncated":false}),
                ),
            },
        ),
        record_at(
            8,
            TranscriptEvent::ToolExecutionSummary(ToolExecutionSummaryEvent {
                turn_id: 1,
                call_id: "test-1".into(),
                name: "shell__exec".into(),
                status: "completed".into(),
                rejection: None,
                effect_kind: "validation".into(),
                primary_path: None,
                command: Some("cargo test".into()),
            }),
        ),
        record_at(
            9,
            TranscriptEvent::ValidationAdvisory(ValidationAdvisory {
                write_effects: 1,
                validation_effects: 1,
                failed_validation_effects: 0,
                message: "validation passed".into(),
            }),
        ),
        record_at(
            10,
            TranscriptEvent::Evidence {
                id: "evidence-validated".into(),
                evidence_kind: EvidenceKind::Validation,
                title: "cargo test".into(),
                summary: "cargo test passed".into(),
                detail: Some("all tests passed".into()),
                source: EvidenceSource::Command {
                    command: "cargo test".into(),
                    status: Some(0),
                },
                tags: vec!["validation".into()],
            },
        ),
        record_at(
            11,
            TranscriptEvent::AssistantMessage {
                content: "ordinary historical note".into(),
            },
        ),
        record_at(
            12,
            TranscriptEvent::FoldedOutputMetadata {
                node_id: None,
                output_id: "historical-fold".into(),
                output_kind: "shell_output".into(),
                call_id: Some("test-1".into()),
                tool_name: Some("shell__exec".into()),
                stream: Some("stdout".into()),
                content: Some("large historical output".repeat(500)),
                byte_count: None,
                line_count: None,
                truncated: Some(false),
                shell_command: Some("cargo test".into()),
                source_start_sequence: Some(7),
                source_end_sequence: Some(7),
                tool_ok: Some(true),
                exit_status: Some(0),
                provider_metadata: Some(
                    json!({"semantic_summary":"cargo test passed with all tests green"}),
                ),
                provider_fold_eligible: Some(true),
            },
        ),
        record_at(
            13,
            TranscriptEvent::AssistantMessage {
                content: "historical final assistant reply".into(),
            },
        ),
        record_at(
            14,
            TranscriptEvent::TurnStarted(TurnStartedEvent {
                turn_id: 2,
                intent: "continue".into(),
                directive: "keep current work".into(),
                validation_reminder: String::new(),
            }),
        ),
        record_at(
            15,
            TranscriptEvent::UserMessage {
                content: UserMessageContent::from("current requirement"),
            },
        ),
        record_at(
            16,
            TranscriptEvent::AssistantMessage {
                content: "pinned current note".into(),
            },
        ),
        record_at(
            17,
            TranscriptEvent::ContextViewOperationMetadata {
                operation: "pin".into(),
                node_id: None,
                block_id: Some("block-seq-16-note".into()),
                detail: None,
            },
        ),
        record_at(
            18,
            TranscriptEvent::AssistantMessage {
                content: "opened current note".into(),
            },
        ),
        record_at(
            19,
            TranscriptEvent::ContextViewOperationMetadata {
                operation: "open_detail".into(),
                node_id: None,
                block_id: Some("block-seq-18-note".into()),
                detail: None,
            },
        ),
        record_at(
            20,
            TranscriptEvent::ContextSummaryArtifactMetadata {
                node_id: "current".into(),
                artifact_id: "source-less-control".into(),
                artifact_kind: "summary".into(),
                version: Some(1),
                summary: Some("source-less retaining control".into()),
                source_node_id: None,
                source_block_id: None,
                source_start_sequence: None,
                source_end_sequence: None,
            },
        ),
        record_at(
            21,
            TranscriptEvent::ToolCallStarted {
                call_id: "incomplete-1".into(),
                name: "shell__exec".into(),
                args: json!({"command":"still running"}),
            },
        ),
    ];

    let resolved = resolve_branch_context(
        records.clone(),
        SessionContextCursor {
            branch_id: None,
            leaf_sequence: None,
        },
    )
    .expect("pre-event branch");
    assert!(
        !resolved.records.iter().any(|record| matches!(
            &record.event,
            TranscriptEvent::FoldedOutputMetadata { output_id, .. }
                if output_id == "historical-fold"
        )),
        "resolved branch content intentionally omits folded-output metadata"
    );
    let validation_records = runtime_projection_records(&records, &resolved);
    assert!(
        validation_records.iter().any(|record| matches!(
            &record.event,
            TranscriptEvent::FoldedOutputMetadata { output_id, .. }
                if output_id == "historical-fold"
        )),
        "runtime projection restores folded-output metadata for validation"
    );
    let pre_snapshot =
        runtime_snapshot_from_resolved_context_unbound("s", &records, &resolved, None, &[])
            .expect("pre-event runtime projection");
    let history = restore_history_projection(&records);
    let tail_start_index = history
        .iter()
        .position(|entry| {
            entry
                .source_spans
                .iter()
                .any(|span| span.start_sequence == 15)
        })
        .expect("current requirement history entry");
    let retired_source_spans = derive_new_retired_source_spans(&records, tail_start_index);
    let runtime_spans = retired_source_spans
        .iter()
        .map(|span| SourceSpan::new(span.start_sequence, span.end_sequence).expect("valid span"))
        .collect::<Vec<_>>();
    let (closure, coverage) = derive_modern_compaction_coverage(&pre_snapshot, &runtime_spans)
        .expect("exact modern coverage");
    let identities = coverage
        .items
        .iter()
        .map(|item| item.identity.as_str())
        .collect::<Vec<_>>();
    assert!(
        identities
            .iter()
            .any(|id| id.starts_with("context-block:block-seq-2-user-requirement"))
    );
    assert!(
        identities
            .iter()
            .any(|id| id.starts_with("evidence:evidence-validated"))
    );
    assert!(
        identities
            .iter()
            .any(|id| id == &"folded-output:historical-fold")
    );

    let mut event = ContextCompactionEvent {
        outcome: "succeeded".into(),
        summary: "historical turn compacted".into(),
        tail_start_index,
        original_history_items: history.len(),
        retained_history_items: 1 + history.len() - tail_start_index,
        retired_source_spans,
        frame_identity_bindings: Vec::new(),
        derived_coverage: Some(coverage.clone()),
        detail: None,
    };
    attach_modern_compaction_coverage(&records, &mut event);
    let validation_snapshot =
        runtime_snapshot_from_resolved_context_unbound("", &records, &resolved, None, &[])
            .expect("validation-equivalent runtime snapshot");
    let validation_spans = derive_new_retired_source_spans(&records, tail_start_index)
        .into_iter()
        .map(|span| SourceSpan::new(span.start_sequence, span.end_sequence).expect("valid span"))
        .collect::<Vec<_>>();
    let (_, validation_coverage) =
        derive_modern_compaction_coverage(&validation_snapshot, &validation_spans)
            .expect("validation-equivalent coverage");
    event.derived_coverage = Some(validation_coverage);
    assert!(!event.summary.contains(RETAINED_FACTS_MARKER));
    records.push(record_at(
        22,
        TranscriptEvent::ContextCompaction(event.clone()),
    ));
    let resolved_after_compaction = resolve_branch_context(
        records.clone(),
        SessionContextCursor {
            branch_id: None,
            leaf_sequence: None,
        },
    )
    .expect("post-compaction branch");
    let post_compaction_snapshot = runtime_snapshot_from_resolved_context_unbound(
        "s",
        &records,
        &resolved_after_compaction,
        None,
        &[],
    )
    .expect("post-compaction runtime snapshot");
    match &mut records.last_mut().expect("compaction record").event {
        TranscriptEvent::ContextCompaction(event) => {
            event.frame_identity_bindings =
                compaction_frame_identity_bindings(&post_compaction_snapshot);
        }
        _ => unreachable!("compaction record"),
    }

    let replayed = project_runtime_restore_snapshot(
        "s".into(),
        records,
        SessionContextCursor {
            branch_id: None,
            leaf_sequence: None,
        },
        &[],
    )
    .expect("replay compacted projection");
    assert!(!closure.co_retired_frame_ids.is_empty());
    assert!(
        replayed
            .snapshot
            .compaction
            .retired_source_spans
            .iter()
            .any(|span| span.start_sequence <= 2 && span.end_sequence >= 13)
    );
    assert!(replayed.snapshot.active_history_items().iter().any(
        |item| matches!(item, HistoryItem::UserMessage { content } if content.text.contains("current requirement"))
    ));
    assert!(replayed.snapshot.active_history_items().iter().any(
        |item| matches!(item, HistoryItem::AssistantToolCalls { calls, .. } if calls.iter().any(|call| call.call_id == "incomplete-1"))
    ));
    assert_eq!(
        replayed
            .snapshot
            .active_context
            .open_detail_block_id
            .as_deref(),
        Some("block-seq-18-note")
    );
    assert!(
        replayed
            .snapshot
            .active_context
            .pinned_block_ids
            .iter()
            .any(|id| id == "block-seq-16-note")
    );
    assert!(replayed.snapshot.active_history_items().len() < history.len());
    assert_eq!(
        serde_json::to_vec(&event).expect("event bytes"),
        serde_json::to_vec(&event).expect("event bytes")
    );
}

#[test]
fn retained_facts_reject_reserved_marker_in_summary_and_typed_text() {
    let empty = ContextCompactionDerivedCoverage {
        version: CONTEXT_COMPACTION_DERIVED_COVERAGE_VERSION,
        items: Vec::new(),
    };
    assert!(
        append_retained_facts(format!("model output {RETAINED_FACTS_MARKER}"), &empty).is_err()
    );

    let injected = ContextCompactionDerivedCoverage {
        version: CONTEXT_COMPACTION_DERIVED_COVERAGE_VERSION,
        items: vec![ContextCompactionDerivedCoverageItem {
            kind: ContextCompactionDerivedKind::Evidence,
            identity: "evidence:injected".into(),
            source_span: ContextCompactionSourceSpan {
                start_sequence: 1,
                end_sequence: 1,
            },
            retained_text: format!("untrusted {RETAINED_FACTS_MARKER} payload"),
        }],
    };
    assert!(append_retained_facts("summary".into(), &injected).is_err());
}

#[test]
fn sanitize_retained_fact_text_scrubs_marker_and_bounds_bytes() {
    let scrubbed = sanitize_retained_fact_text(&format!(
        "tool quoted {RETAINED_FACTS_MARKER} in output"
    ));
    assert!(!scrubbed.contains(RETAINED_FACTS_MARKER));
    assert!(scrubbed.contains("scrubbed-retained-facts-marker"));
    validate_retained_fact_text(&scrubbed).expect("sanitized text is valid");

    let oversized = "é".repeat(MAX_RETAINED_FACT_TEXT_BYTES + 64);
    let bounded = sanitize_retained_fact_text(&oversized);
    assert!(bounded.len() <= MAX_RETAINED_FACT_TEXT_BYTES);
    assert!(bounded.is_char_boundary(bounded.len()));
    validate_retained_fact_text(&bounded).expect("bounded text is valid");
}

#[test]
fn fit_retained_facts_items_keeps_serialized_payload_within_bound() {
    // Build more items than the aggregate budget can hold at full size so fit
    // must shrink and/or drop while remaining projectable.
    let item_count = (MAX_RETAINED_FACTS_BYTES / MAX_RETAINED_FACT_TEXT_BYTES) + 32;
    let items = (0..item_count)
        .map(|idx| ContextCompactionDerivedCoverageItem {
            kind: if idx % 5 == 0 {
                ContextCompactionDerivedKind::CurrentUserRequirement
            } else if idx % 5 == 1 {
                ContextCompactionDerivedKind::FileWriteFact
            } else if idx % 5 == 2 {
                ContextCompactionDerivedKind::TestResult
            } else if idx % 5 == 3 {
                ContextCompactionDerivedKind::Evidence
            } else {
                ContextCompactionDerivedKind::FoldedOutput
            },
            identity: format!("item:overflow-{idx:03}"),
            source_span: ContextCompactionSourceSpan {
                start_sequence: idx as u64 + 1,
                end_sequence: idx as u64 + 1,
            },
            retained_text: format!(
                "payload {idx} {RETAINED_FACTS_MARKER} {}",
                "x".repeat(MAX_RETAINED_FACT_TEXT_BYTES)
            ),
        })
        .collect::<Vec<_>>();

    let fitted = fit_retained_facts_items(items).expect("overflowing coverage is fitted");
    assert!(!fitted.is_empty());
    for item in &fitted {
        validate_retained_fact_text(&item.retained_text).expect("each fitted item is valid");
        assert!(!item.retained_text.contains(RETAINED_FACTS_MARKER));
    }
    let facts = serde_json::to_string(&fitted).expect("facts json");
    assert!(facts.len() <= MAX_RETAINED_FACTS_BYTES);

    let coverage = ContextCompactionDerivedCoverage {
        version: CONTEXT_COMPACTION_DERIVED_COVERAGE_VERSION,
        items: fitted,
    };
    append_retained_facts("historical turn compacted".into(), &coverage)
        .expect("fitted coverage projects into a bounded retained-facts suffix");
}

// Already at the fit shrink floor so the algorithm must drop rather than shrink.
const MIN_PRIORITY_DROP_TEXT: usize = 64;

#[test]
fn fit_retained_facts_prefers_dropping_folded_output_before_requirements() {
    let requirement = ContextCompactionDerivedCoverageItem {
        kind: ContextCompactionDerivedKind::CurrentUserRequirement,
        identity: "context-block:requirement".into(),
        source_span: ContextCompactionSourceSpan {
            start_sequence: 1,
            end_sequence: 1,
        },
        retained_text: "x".repeat(MIN_PRIORITY_DROP_TEXT),
    };
    // Already-minimal texts: enough bulk that aggregate JSON exceeds the budget
    // and priority-based dropping is required.
    let bulk_count = (MAX_RETAINED_FACTS_BYTES / (MIN_PRIORITY_DROP_TEXT + 96)) + 128;
    let mut items = vec![requirement];
    for idx in 0..bulk_count {
        items.push(ContextCompactionDerivedCoverageItem {
            kind: ContextCompactionDerivedKind::FoldedOutput,
            identity: format!("folded-output:bulk-{idx:04}"),
            source_span: ContextCompactionSourceSpan {
                start_sequence: 10 + idx as u64,
                end_sequence: 10 + idx as u64,
            },
            retained_text: "y".repeat(MIN_PRIORITY_DROP_TEXT),
        });
    }
    let fitted = fit_retained_facts_items(items).expect("priority fit succeeds");
    assert!(
        fitted
            .iter()
            .any(|item| item.identity == "context-block:requirement"),
        "user requirements must outrank folded bulk when dropping"
    );
    assert!(
        fitted
            .iter()
            .filter(|item| item.kind == ContextCompactionDerivedKind::FoldedOutput)
            .count()
            < bulk_count,
        "some folded bulk items should be dropped under pressure"
    );
    assert!(serde_json::to_string(&fitted).expect("json").len() <= MAX_RETAINED_FACTS_BYTES);
}

#[test]
fn project_compaction_summary_scrubs_marker_in_model_body() {
    let coverage = ContextCompactionDerivedCoverage {
        version: CONTEXT_COMPACTION_DERIVED_COVERAGE_VERSION,
        items: vec![ContextCompactionDerivedCoverageItem {
            kind: ContextCompactionDerivedKind::Evidence,
            identity: "evidence:clean".into(),
            source_span: ContextCompactionSourceSpan {
                start_sequence: 1,
                end_sequence: 1,
            },
            retained_text: "kept fact".into(),
        }],
    };
    let projected = project_compaction_summary(
        &format!("model mentioned {RETAINED_FACTS_MARKER} by accident"),
        Some(&coverage),
    )
    .expect("quoted markers in model summaries are scrubbed");
    // Modern path keeps typed coverage off the visible summary body.
    assert!(!projected.contains(RETAINED_FACTS_DELIMITER));
    assert!(!projected.contains(RETAINED_FACTS_MARKER));
    assert!(projected.contains("scrubbed-retained-facts-marker"));
    assert!(!projected.contains("kept fact"));
}

fn folded_output_for_semantic_test(
    content: String,
    truncated: bool,
    provider_metadata: Option<serde_json::Value>,
) -> FoldedOutputMetadata {
    FoldedOutputMetadata {
        output_id: "semantic-test".into(),
        node_id: None,
        output_kind: "file_content".into(),
        call_id: None,
        tool_name: None,
        stream: None,
        byte_count: content.len(),
        line_count: 1,
        content,
        truncated,
        shell_command: None,
        source_start_sequence: Some(1),
        source_end_sequence: Some(1),
        available_sequence: Some(1),
        tool_ok: Some(true),
        exit_status: None,
        provider_metadata,
        provider_fold_eligible: true,
    }
}

#[test]
fn folded_output_coverage_uses_bounded_raw_excerpt_instead_of_semantic_summary() {
    let output = folded_output_for_semantic_test(
        "é".repeat(MAX_RETAINED_FACT_TEXT_BYTES),
        true,
        Some(json!({"path":"src/lib.rs","semantic_summary":"untrusted summary"})),
    );

    let excerpt = deterministic_folded_output_retained_text(&output)
        .expect("complete raw output retains a deterministic excerpt");
    assert_eq!(
        excerpt,
        deterministic_folded_output_retained_text(&output)
            .expect("the raw excerpt is deterministic")
    );
    assert!(excerpt.starts_with("[bounded raw folded-output excerpt; not a semantic summary]"));
    assert!(!excerpt.contains("untrusted summary"));
    assert!(excerpt.len() <= MAX_RETAINED_FACT_TEXT_BYTES);
    assert!(excerpt.is_char_boundary(excerpt.len()));
}

#[test]
fn folded_output_coverage_accepts_every_trusted_output_kind() {
    for output_kind in [
        "text",
        "tool_result",
        "shell_output",
        "tool_output",
        "file_content",
        "search_matches",
        "mcp_text",
    ] {
        let mut output = folded_output_for_semantic_test(
            "x".repeat(MAX_RETAINED_FACT_TEXT_BYTES + 1),
            true,
            None,
        );
        output.output_kind = output_kind.into();
        assert!(deterministic_folded_output_retained_text(&output).is_ok());
    }
}

#[test]
fn folded_output_coverage_rejects_incomplete_or_unknown_raw_artifacts() {
    let mut incomplete = folded_output_for_semantic_test("raw".into(), false, None);
    incomplete.provider_fold_eligible = false;
    assert!(deterministic_folded_output_retained_text(&incomplete).is_err());

    let mut unknown_kind = folded_output_for_semantic_test("raw".into(), false, None);
    unknown_kind.output_kind = "unknown_output".into();
    assert!(deterministic_folded_output_retained_text(&unknown_kind).is_err());
}

#[test]
fn legacy_and_uncovered_compactions_reject_bare_retained_facts_marker() {
    let records = vec![
        record_at(
            1,
            TranscriptEvent::UserMessage {
                content: UserMessageContent::from("old"),
            },
        ),
        record_at(
            2,
            TranscriptEvent::AssistantMessage {
                content: "tail".into(),
            },
        ),
    ];
    let event = ContextCompactionEvent {
        outcome: "succeeded".into(),
        summary: format!("legacy summary {RETAINED_FACTS_MARKER}"),
        tail_start_index: 1,
        original_history_items: 2,
        retained_history_items: 1,
        retired_source_spans: Vec::new(),
        frame_identity_bindings: Vec::new(),
        derived_coverage: None,
        detail: None,
    };
    assert!(validate_context_compaction_event(&records, &event).is_err());
}

fn modern_compaction_tamper_fixture() -> Vec<TranscriptRecord> {
    let mut records = vec![
        record_at(
            1,
            TranscriptEvent::TurnStarted(TurnStartedEvent {
                turn_id: 1,
                intent: "implement".into(),
                directive: "write and validate".into(),
                validation_reminder: String::new(),
            }),
        ),
        record_at(
            2,
            TranscriptEvent::UserMessage {
                content: UserMessageContent::from("historical requirement"),
            },
        ),
        record_at(
            3,
            TranscriptEvent::ToolCallStarted {
                call_id: "write-1".into(),
                name: "fs__write".into(),
                args: json!({"path":"src/lib.rs","content":"fn updated() {}"}),
            },
        ),
        record_at(
            4,
            TranscriptEvent::ToolCallFinished {
                call_id: "write-1".into(),
                name: "fs__write".into(),
                ok: true,
                output: ToolResult::ok(
                    "fs__write",
                    json!({"path":"src/lib.rs","bytes_written":15}),
                ),
            },
        ),
        record_at(
            5,
            TranscriptEvent::ToolExecutionSummary(ToolExecutionSummaryEvent {
                turn_id: 1,
                call_id: "write-1".into(),
                name: "fs__write".into(),
                status: "completed".into(),
                rejection: None,
                effect_kind: "write".into(),
                primary_path: Some("src/lib.rs".into()),
                command: None,
            }),
        ),
        record_at(
            6,
            TranscriptEvent::ToolCallStarted {
                call_id: "test-1".into(),
                name: "shell__exec".into(),
                args: json!({"command":"cargo test"}),
            },
        ),
        record_at(
            7,
            TranscriptEvent::ToolCallFinished {
                call_id: "test-1".into(),
                name: "shell__exec".into(),
                ok: true,
                output: ToolResult::ok("shell__exec", json!({"status":0,"stdout":"ok"})),
            },
        ),
        record_at(
            8,
            TranscriptEvent::ToolExecutionSummary(ToolExecutionSummaryEvent {
                turn_id: 1,
                call_id: "test-1".into(),
                name: "shell__exec".into(),
                status: "completed".into(),
                rejection: None,
                effect_kind: "validation".into(),
                primary_path: None,
                command: Some("cargo test".into()),
            }),
        ),
        record_at(
            9,
            TranscriptEvent::ValidationAdvisory(ValidationAdvisory {
                write_effects: 1,
                validation_effects: 1,
                failed_validation_effects: 0,
                message: "validation passed".into(),
            }),
        ),
        record_at(
            10,
            TranscriptEvent::Evidence {
                id: "evidence-validated".into(),
                evidence_kind: EvidenceKind::Validation,
                title: "cargo test".into(),
                summary: "cargo test passed".into(),
                detail: None,
                source: EvidenceSource::Command {
                    command: "cargo test".into(),
                    status: Some(0),
                },
                tags: vec![],
            },
        ),
        record_at(
            11,
            TranscriptEvent::FoldedOutputMetadata {
                node_id: None,
                output_id: "historical-fold".into(),
                output_kind: "shell_output".into(),
                call_id: Some("test-1".into()),
                tool_name: Some("shell__exec".into()),
                stream: Some("stdout".into()),
                content: Some("large historical output".repeat(500)),
                byte_count: None,
                line_count: None,
                truncated: Some(false),
                shell_command: Some("cargo test".into()),
                source_start_sequence: Some(7),
                source_end_sequence: Some(7),
                tool_ok: Some(true),
                exit_status: Some(0),
                provider_metadata: Some(json!({"semantic_summary":"cargo test passed"})),
                provider_fold_eligible: Some(true),
            },
        ),
        record_at(
            12,
            TranscriptEvent::AssistantMessage {
                content: "historical reply".into(),
            },
        ),
        record_at(
            13,
            TranscriptEvent::UserMessage {
                content: UserMessageContent::from("current requirement"),
            },
        ),
    ];
    let history = restore_history_projection(&records);
    let tail_start_index = history
        .iter()
        .position(|entry| {
            entry
                .source_spans
                .iter()
                .any(|span| span.start_sequence == 13)
        })
        .expect("current requirement history entry");
    let mut event = ContextCompactionEvent {
        outcome: "succeeded".into(),
        summary: "historical turn compacted".into(),
        tail_start_index,
        original_history_items: history.len(),
        retained_history_items: 1 + history.len() - tail_start_index,
        retired_source_spans: derive_new_retired_source_spans(&records, tail_start_index),
        frame_identity_bindings: Vec::new(),
        derived_coverage: None,
        detail: None,
    };
    attach_modern_compaction_coverage(&records, &mut event);
    assert!(
        event
            .derived_coverage
            .as_ref()
            .expect("modern coverage")
            .items
            .len()
            >= 2,
        "fixture needs multiple canonical derived coverage items"
    );
    records.push(record_at(14, TranscriptEvent::ContextCompaction(event)));
    records
}

fn non_root_modern_compaction_fixture() -> Vec<TranscriptRecord> {
    let mut records = modern_compaction_tamper_fixture();
    records.pop(); // Reuse the historical material, but compact it from a child branch.
    records.push(metadata_record_at(
        14,
        TranscriptEvent::ContextBranchCreated {
            branch_id: "child".into(),
            parent_branch_id: ROOT_CONTEXT_BRANCH_ID.into(),
            base_sequence: 13,
            label: None,
        },
    ));
    records.push(metadata_record_at(
        15,
        TranscriptEvent::ContextCheckout {
            branch_id: "child".into(),
            leaf_sequence: 13,
        },
    ));
    records.push(branch_record_at(
        16,
        "child",
        TranscriptEvent::UserMessage {
            content: UserMessageContent::from("current child requirement"),
        },
    ));

    let history = restore_history_projection(&records);
    let tail_start_index = history
        .iter()
        .position(|entry| {
            entry
                .source_spans
                .iter()
                .any(|span| span.start_sequence == 16)
        })
        .expect("current child history entry");
    let mut event = ContextCompactionEvent {
        outcome: "succeeded".into(),
        summary: "child historical turn compacted".into(),
        tail_start_index,
        original_history_items: history.len(),
        retained_history_items: 1 + history.len() - tail_start_index,
        retired_source_spans: derive_new_retired_source_spans(&records, tail_start_index),
        frame_identity_bindings: Vec::new(),
        derived_coverage: None,
        detail: None,
    };
    attach_modern_compaction_coverage(&records, &mut event);
    assert!(
        !event
            .derived_coverage
            .as_ref()
            .expect("modern coverage")
            .items
            .is_empty(),
        "fixture needs canonical non-root derived coverage"
    );

    let mut candidate_records = records.clone();
    candidate_records.push(branch_record_at(
        17,
        "child",
        TranscriptEvent::ContextCompaction(event.clone()),
    ));
    let candidate = project_runtime_restore_snapshot(
        "s".into(),
        candidate_records,
        SessionContextCursor {
            branch_id: Some("child".into()),
            leaf_sequence: Some(17),
        },
        &[],
    )
    .expect("unbound non-root compaction candidate");
    event.frame_identity_bindings = compaction_frame_identity_bindings(&candidate.snapshot);
    assert!(
        !event.frame_identity_bindings.is_empty(),
        "fixture needs canonical non-root frame bindings"
    );
    records.push(branch_record_at(
        17,
        "child",
        TranscriptEvent::ContextCompaction(event),
    ));
    records
}

#[test]
fn non_root_modern_compaction_rejects_structurally_invalid_retained_text() {
    let valid = non_root_modern_compaction_fixture();
    project_session_restore_snapshot("s".into(), valid.clone())
        .expect("valid non-root modern compaction restores");

    let mut poisoned = valid.clone();
    {
        let TranscriptEvent::ContextCompaction(event) =
            &mut poisoned.last_mut().expect("compaction record").event
        else {
            panic!("compaction record");
        };
        event.derived_coverage.as_mut().expect("coverage").items[0]
            .retained_text = format!("tampered {RETAINED_FACTS_MARKER}");
    }
    assert!(
        project_session_restore_snapshot("s".into(), poisoned.clone()).is_err(),
        "reserved marker in coverage text must fail"
    );
    assert!(
        project_runtime_restore_snapshot(
            "s".into(),
            poisoned,
            SessionContextCursor {
                branch_id: Some("child".into()),
                leaf_sequence: None,
            },
            &[],
        )
        .is_err()
    );

    // Benign coverage text drift is accepted after exact re-derive was removed.
    let mut drifted = valid;
    {
        let TranscriptEvent::ContextCompaction(event) =
            &mut drifted.last_mut().expect("compaction record").event
        else {
            panic!("compaction record");
        };
        event.derived_coverage.as_mut().expect("coverage").items[0]
            .retained_text
            .push_str(" benign drift");
    }
    project_session_restore_snapshot("s".into(), drifted)
        .expect("benign coverage drift must not fail restore");
}


#[test]
fn modern_compaction_rejects_only_structurally_invalid_coverage() {
    // Exact re-derive matching is gone. Restore only fails closed on structural
    // poison (reserved markers / oversize text), not on benign coverage drift.
    let valid = modern_compaction_tamper_fixture();
    project_session_restore_snapshot("s".into(), valid.clone())
        .expect("valid modern compaction restores");

    let mut poisoned = valid.clone();
    {
        let TranscriptEvent::ContextCompaction(event) =
            &mut poisoned.last_mut().expect("event").event
        else {
            panic!("fixture ends with compaction");
        };
        event.derived_coverage.as_mut().expect("coverage").items[0]
            .retained_text = format!("bad {RETAINED_FACTS_MARKER}");
    }
    assert!(
        project_session_restore_snapshot("s".into(), poisoned).is_err(),
        "reserved marker in retained_text must still fail closed"
    );

    let mut bare_marker = valid;
    {
        let TranscriptEvent::ContextCompaction(event) =
            &mut bare_marker.last_mut().expect("event").event
        else {
            panic!("fixture ends with compaction");
        };
        event.summary = format!("historical turn compacted {RETAINED_FACTS_MARKER}");
    }
    assert!(
        project_session_restore_snapshot("s".into(), bare_marker).is_err(),
        "bare marker in summary without delimiter must still fail closed"
    );
}
