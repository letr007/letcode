use crate::context_tree::{ContextNodeId, ContextTreeOp, ContextTreeState};
use crate::transcript::{TranscriptEvent, TranscriptRecord};
use anyhow::Context;

pub(crate) fn project_context_tree(
    records: &[TranscriptRecord],
) -> anyhow::Result<ContextTreeState> {
    replay_context_tree(records)
}

pub(crate) fn replay_context_tree(
    records: &[TranscriptRecord],
) -> anyhow::Result<ContextTreeState> {
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
                    format!(
                        "invalid context node_id at transcript sequence {}",
                        record.sequence
                    )
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
                        format!(
                            "invalid context node_id at transcript sequence {}",
                            record.sequence
                        )
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
