use anyhow::{Result, anyhow, ensure};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

pub(crate) const DEFAULT_ROOT_CONTEXT_NODE_ID: &str = "root";

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub(crate) struct ContextNodeId(String);

impl ContextNodeId {
    pub(crate) fn new(value: impl Into<String>) -> Result<Self> {
        let value = value.into();
        let trimmed = value.trim();
        ensure!(!trimmed.is_empty(), "context node_id must not be empty");
        Ok(Self(trimmed.to_string()))
    }

    pub(crate) fn root() -> Self {
        Self(DEFAULT_ROOT_CONTEXT_NODE_ID.to_string())
    }

    pub(crate) fn as_str(&self) -> &str {
        &self.0
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct ContextBlockRef {
    pub block_id: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct ContextSourceRef {
    pub source_kind: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source_id: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum ContextNodeStatus {
    Active,
    Inactive,
    Archived,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct ContextNodeRecord {
    pub node_id: ContextNodeId,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parent_node_id: Option<ContextNodeId>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub label: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub purpose: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub block_ref: Option<ContextBlockRef>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source_ref: Option<ContextSourceRef>,
    pub status: ContextNodeStatus,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum ContextTreeOp {
    CreateNode {
        node_id: ContextNodeId,
        parent_node_id: Option<ContextNodeId>,
        label: Option<String>,
        purpose: Option<String>,
        block_ref: Option<ContextBlockRef>,
        source_ref: Option<ContextSourceRef>,
    },
    SetNodeStatus {
        node_id: ContextNodeId,
        status: ContextNodeStatus,
    },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct ContextTreeState {
    root_node_id: ContextNodeId,
    active_node_id: Option<ContextNodeId>,
    nodes: BTreeMap<ContextNodeId, ContextNodeRecord>,
}

impl ContextTreeState {
    pub(crate) fn with_default_root() -> Self {
        let root_node_id = ContextNodeId::root();
        let root = ContextNodeRecord {
            node_id: root_node_id.clone(),
            parent_node_id: None,
            label: None,
            purpose: None,
            block_ref: None,
            source_ref: None,
            status: ContextNodeStatus::Active,
        };
        let mut nodes = BTreeMap::new();
        nodes.insert(root_node_id.clone(), root);
        Self {
            root_node_id: root_node_id.clone(),
            active_node_id: Some(root_node_id),
            nodes,
        }
    }

    pub(crate) fn replay(ops: &[ContextTreeOp]) -> Result<Self> {
        let mut state = Self::with_default_root();
        for op in ops {
            state.apply(op)?;
        }
        Ok(state)
    }

    pub(crate) fn apply(&mut self, op: &ContextTreeOp) -> Result<()> {
        match op {
            ContextTreeOp::CreateNode {
                node_id,
                parent_node_id,
                label,
                purpose,
                block_ref,
                source_ref,
            } => {
                ensure!(
                    !self.nodes.contains_key(node_id),
                    "duplicate context node_id '{}'",
                    node_id.as_str()
                );
                let parent_node_id = parent_node_id
                    .clone()
                    .ok_or_else(|| anyhow!("context node '{}' is missing a parent", node_id.as_str()))?;
                ensure!(
                    parent_node_id != *node_id,
                    "context node '{}' cannot be its own parent",
                    node_id.as_str()
                );
                ensure!(
                    self.nodes.contains_key(&parent_node_id),
                    "unknown parent context node '{}' for node '{}'",
                    parent_node_id.as_str(),
                    node_id.as_str()
                );

                self.nodes.insert(
                    node_id.clone(),
                    ContextNodeRecord {
                        node_id: node_id.clone(),
                        parent_node_id: Some(parent_node_id),
                        label: label.clone(),
                        purpose: purpose.clone(),
                        block_ref: block_ref.clone(),
                        source_ref: source_ref.clone(),
                        status: ContextNodeStatus::Inactive,
                    },
                );
            }
            ContextTreeOp::SetNodeStatus { node_id, status } => {
                let node = self.nodes.get_mut(node_id).ok_or_else(|| {
                    anyhow!("unknown context node '{}' in lifecycle event", node_id.as_str())
                })?;

                match status {
                    ContextNodeStatus::Active => {
                        if let Some(active_node_id) = &self.active_node_id {
                            ensure!(
                                active_node_id == node_id,
                                "cannot activate context node '{}' while '{}' is active",
                                node_id.as_str(),
                                active_node_id.as_str()
                            );
                        }
                        self.active_node_id = Some(node_id.clone());
                    }
                    ContextNodeStatus::Inactive | ContextNodeStatus::Archived => {
                        if self.active_node_id.as_ref() == Some(node_id) {
                            self.active_node_id = None;
                        }
                    }
                }

                node.status = status.clone();
            }
        }
        Ok(())
    }

    pub(crate) fn root_node_id(&self) -> &ContextNodeId {
        &self.root_node_id
    }

    pub(crate) fn active_node_id(&self) -> Option<&ContextNodeId> {
        self.active_node_id.as_ref()
    }

    pub(crate) fn node(&self, node_id: &ContextNodeId) -> Option<&ContextNodeRecord> {
        self.nodes.get(node_id)
    }

    pub(crate) fn node_count(&self) -> usize {
        self.nodes.len()
    }
}
