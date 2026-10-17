use super::*;

#[test]
fn metadata_lists_supported_experts_in_canonical_order() {
    let names = supported_agent_names().collect::<Vec<_>>();
    assert_eq!(
        names,
        vec![
            "explorer",
            "fixer",
            "oracle",
            "designer",
            "librarian",
            "general",
            "reviewer",
            "historian",
        ]
    );
    assert!(is_system_expert("reviewer"));
    assert!(
        !DELEGATION_EXPERTS
            .iter()
            .any(|expert| expert.agent_name == "reviewer")
    );
}
