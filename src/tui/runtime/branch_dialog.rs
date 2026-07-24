use super::{DialogItem, transcript_projection};

pub(super) fn branch_dialog_items(
    branches: &[transcript_projection::ContextBranchInfo],
) -> Vec<DialogItem> {
    fn push_children(
        out: &mut Vec<DialogItem>,
        branches: &[transcript_projection::ContextBranchInfo],
        parent: Option<&str>,
        depth: usize,
    ) {
        let mut children = branches
            .iter()
            .filter(|branch| branch.parent_branch_id.as_deref() == parent)
            .collect::<Vec<_>>();
        children.sort_by(|left, right| left.branch_id.cmp(&right.branch_id));

        for branch in children {
            let indent = if depth == 0 {
                String::new()
            } else {
                format!("{}↳ ", "  ".repeat(depth.saturating_sub(1)))
            };
            let mut label = format!("{indent}{}", branch.branch_id);
            if branch.is_current {
                label.push_str(" • current");
            }
            let mut item = DialogItem::new(branch.branch_id.clone(), label, branch.label.clone())
                .with_right_detail(format!("@{}", branch.tip_sequence));
            if depth == 0 {
                item = item.with_section("Context branches");
            }
            out.push(item);
            push_children(out, branches, Some(branch.branch_id.as_str()), depth + 1);
        }
    }

    let mut items = Vec::new();
    push_children(
        &mut items,
        branches,
        Some(crate::transcript::ROOT_CONTEXT_BRANCH_ID),
        1,
    );

    // Include root/main first with no indent.
    if let Some(root) = branches
        .iter()
        .find(|branch| branch.branch_id == crate::transcript::ROOT_CONTEXT_BRANCH_ID)
    {
        let root_item = DialogItem::new(
            root.branch_id.clone(),
            if root.is_current {
                format!("{} • current", root.branch_id)
            } else {
                root.branch_id.clone()
            },
            root.label.clone(),
        )
        .with_section("Context branches")
        .with_right_detail(format!("@{}", root.tip_sequence));
        items.insert(0, root_item);
    }

    items
}
