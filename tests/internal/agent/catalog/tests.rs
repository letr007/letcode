use super::*;

#[test]
fn writable_templates_match_catalog_contracts() {
    for entry in SUBAGENT_CATALOG {
        let template = AgentTemplate::from_name(entry.agent_name).expect("catalog template");
        let writable = matches!(entry.agent_name, "fixer" | "general");
        let contract = template.capability_contract();
        assert_eq!(template.can_write, writable);
        assert_eq!(entry.read_only, !writable);
        assert_eq!(contract.can_write, writable);
        assert!(!contract.can_delegate);
        assert_eq!(contract.permission_mode, PermissionMode::Default);
        assert_eq!(
            contract.tool_scope,
            if writable {
                ToolScope::FullAccess
            } else {
                ToolScope::ReadOnlyExplorer
            }
        );
        assert_eq!(
            crate::permission::classify_tool(entry.tool_name),
            if writable {
                crate::permission::ToolPermissionClass::Write
            } else {
                crate::permission::ToolPermissionClass::Preview
            }
        );
    }
}

#[test]
fn reviewer_evidence_parent_tool_uses_system_prefix() {
    assert_eq!(
        subagent_evidence_parent_tool("reviewer").as_deref(),
        Some("system__reviewer")
    );
    assert_eq!(
        subagent_evidence_parent_tool("explorer").as_deref(),
        Some(tool_names::TOOL_AGENT_EXPLORE)
    );
    assert!(subagent_evidence_parent_tool("nosuch").is_none());
}
