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
