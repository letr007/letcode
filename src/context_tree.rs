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
                let parent_node_id = parent_node_id.clone().ok_or_else(|| {
                    anyhow!("context node '{}' is missing a parent", node_id.as_str())
                })?;
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
                ensure!(
                    self.nodes
                        .get(&parent_node_id)
                        .is_some_and(|parent| parent.status != ContextNodeStatus::Archived),
                    "cannot create context node '{}' under archived parent '{}'",
                    node_id.as_str(),
                    parent_node_id.as_str()
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
                let current_status = self
                    .nodes
                    .get(node_id)
                    .map(|node| node.status.clone())
                    .ok_or_else(|| {
                        anyhow!(
                            "unknown context node '{}' in lifecycle event",
                            node_id.as_str()
                        )
                    })?;

                ensure!(
                    !(current_status == ContextNodeStatus::Archived
                        && *status != ContextNodeStatus::Archived),
                    "cannot change status for archived context node '{}'",
                    node_id.as_str()
                );

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

                let node = self.nodes.get_mut(node_id).expect("validated context node");
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

    pub(crate) fn nodes(&self) -> impl Iterator<Item = &ContextNodeRecord> {
        self.nodes.values()
    }

    #[cfg(test)]
    pub(crate) fn node_count(&self) -> usize {
        self.nodes.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn create_node_rejects_unknown_parent() {
        let mut tree = ContextTreeState::with_default_root();

        let error = tree
            .apply(&ContextTreeOp::CreateNode {
                node_id: ContextNodeId::new("child").expect("node id"),
                parent_node_id: Some(ContextNodeId::new("missing").expect("parent id")),
                label: None,
                purpose: None,
                block_ref: None,
                source_ref: None,
            })
            .expect_err("unknown parent should fail");

        assert!(
            error
                .to_string()
                .contains("unknown parent context node 'missing'")
        );
    }

    #[test]
    fn create_node_rejects_duplicate_node_id() {
        let mut tree = ContextTreeState::with_default_root();
        let child_id = ContextNodeId::new("child").expect("node id");

        tree.apply(&ContextTreeOp::CreateNode {
            node_id: child_id.clone(),
            parent_node_id: Some(ContextNodeId::root()),
            label: None,
            purpose: None,
            block_ref: None,
            source_ref: None,
        })
        .expect("create child");

        let error = tree
            .apply(&ContextTreeOp::CreateNode {
                node_id: child_id,
                parent_node_id: Some(ContextNodeId::root()),
                label: None,
                purpose: None,
                block_ref: None,
                source_ref: None,
            })
            .expect_err("duplicate node should fail");

        assert!(
            error
                .to_string()
                .contains("duplicate context node_id 'child'")
        );
    }

    #[test]
    fn activate_rejects_when_another_node_is_active() {
        let mut tree = ContextTreeState::with_default_root();
        let child_a = ContextNodeId::new("child-a").expect("child a");
        let child_b = ContextNodeId::new("child-b").expect("child b");

        for node_id in [child_a.clone(), child_b.clone()] {
            tree.apply(&ContextTreeOp::CreateNode {
                node_id,
                parent_node_id: Some(ContextNodeId::root()),
                label: None,
                purpose: None,
                block_ref: None,
                source_ref: None,
            })
            .expect("create child");
        }

        tree.apply(&ContextTreeOp::SetNodeStatus {
            node_id: ContextNodeId::root(),
            status: ContextNodeStatus::Inactive,
        })
        .expect("suspend root");
        tree.apply(&ContextTreeOp::SetNodeStatus {
            node_id: child_a,
            status: ContextNodeStatus::Active,
        })
        .expect("activate child a");

        let error = tree
            .apply(&ContextTreeOp::SetNodeStatus {
                node_id: child_b,
                status: ContextNodeStatus::Active,
            })
            .expect_err("second active node should fail");

        assert!(
            error
                .to_string()
                .contains("cannot activate context node 'child-b' while 'child-a' is active")
        );
    }

    #[test]
    fn archived_node_rejects_lifecycle_changes() {
        let mut tree = ContextTreeState::with_default_root();
        let child = ContextNodeId::new("child").expect("child");

        tree.apply(&ContextTreeOp::CreateNode {
            node_id: child.clone(),
            parent_node_id: Some(ContextNodeId::root()),
            label: None,
            purpose: None,
            block_ref: None,
            source_ref: None,
        })
        .expect("create child");
        tree.apply(&ContextTreeOp::SetNodeStatus {
            node_id: ContextNodeId::root(),
            status: ContextNodeStatus::Inactive,
        })
        .expect("suspend root");
        tree.apply(&ContextTreeOp::SetNodeStatus {
            node_id: child.clone(),
            status: ContextNodeStatus::Active,
        })
        .expect("activate child");
        tree.apply(&ContextTreeOp::SetNodeStatus {
            node_id: child.clone(),
            status: ContextNodeStatus::Archived,
        })
        .expect("archive child");

        let error = tree
            .apply(&ContextTreeOp::SetNodeStatus {
                node_id: child,
                status: ContextNodeStatus::Inactive,
            })
            .expect_err("archived node should reject further lifecycle changes");

        assert!(
            error
                .to_string()
                .contains("cannot change status for archived context node 'child'")
        );
    }

    #[test]
    fn archived_parent_rejects_new_children() {
        let mut tree = ContextTreeState::with_default_root();
        let parent = ContextNodeId::new("parent").expect("parent");

        tree.apply(&ContextTreeOp::CreateNode {
            node_id: parent.clone(),
            parent_node_id: Some(ContextNodeId::root()),
            label: None,
            purpose: None,
            block_ref: None,
            source_ref: None,
        })
        .expect("create parent");
        tree.apply(&ContextTreeOp::SetNodeStatus {
            node_id: parent.clone(),
            status: ContextNodeStatus::Archived,
        })
        .expect("archive parent");

        let error = tree
            .apply(&ContextTreeOp::CreateNode {
                node_id: ContextNodeId::new("child").expect("child"),
                parent_node_id: Some(parent),
                label: None,
                purpose: None,
                block_ref: None,
                source_ref: None,
            })
            .expect_err("archived parent should fail");

        assert!(
            error
                .to_string()
                .contains("cannot create context node 'child' under archived parent 'parent'")
        );
    }
}
