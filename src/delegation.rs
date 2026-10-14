#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DelegationMetadata {
    pub agent_name: &'static str,
}

pub const DELEGATION_EXPERTS: &[DelegationMetadata] = &[
    DelegationMetadata {
        agent_name: "explorer",
    },
    DelegationMetadata {
        agent_name: "fixer",
    },
    DelegationMetadata {
        agent_name: "oracle",
    },
    DelegationMetadata {
        agent_name: "designer",
    },
    DelegationMetadata {
        agent_name: "librarian",
    },
    DelegationMetadata {
        agent_name: "general",
    },
];

pub const SYSTEM_EXPERTS: &[&str] = &["reviewer", "historian"];

pub fn supported_agent_names() -> impl Iterator<Item = &'static str> {
    DELEGATION_EXPERTS
        .iter()
        .map(|expert| expert.agent_name)
        .chain(SYSTEM_EXPERTS.iter().copied())
}

#[cfg(test)]
pub fn is_system_expert(agent_name: &str) -> bool {
    SYSTEM_EXPERTS.contains(&agent_name)
}

#[cfg(test)]
mod tests {
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
}
