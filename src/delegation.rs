#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DelegationMetadata {
    pub agent_name: &'static str,
    pub command: &'static str,
    pub insert_text: &'static str,
    pub description: &'static str,
    pub usage: &'static str,
}

pub const DELEGATION_EXPERTS: &[DelegationMetadata] = &[
    DelegationMetadata {
        agent_name: "explorer",
        command: "@explorer",
        insert_text: "@explorer ",
        description: "Delegate a read-only exploration task",
        usage: "@explorer <task>",
    },
    DelegationMetadata {
        agent_name: "fixer",
        command: "@fixer",
        insert_text: "@fixer ",
        description: "Delegate an implementation or repair task",
        usage: "@fixer <task>",
    },
    DelegationMetadata {
        agent_name: "oracle",
        command: "@oracle",
        insert_text: "@oracle ",
        description: "Delegate a review or audit task",
        usage: "@oracle <task>",
    },
    DelegationMetadata {
        agent_name: "designer",
        command: "@designer",
        insert_text: "@designer ",
        description: "Delegate UX or design-oriented work",
        usage: "@designer <task>",
    },
    DelegationMetadata {
        agent_name: "librarian",
        command: "@librarian",
        insert_text: "@librarian ",
        description: "Delegate documentation or reference gathering",
        usage: "@librarian <task>",
    },
    DelegationMetadata {
        agent_name: "general",
        command: "@general",
        insert_text: "@general ",
        description: "Delegate general-purpose task execution",
        usage: "@general <task>",
    },
];

pub fn supported_agent_names() -> impl Iterator<Item = &'static str> {
    DELEGATION_EXPERTS.iter().map(|expert| expert.agent_name)
}

pub fn find_expert(agent_name: &str) -> Option<&'static DelegationMetadata> {
    DELEGATION_EXPERTS
        .iter()
        .find(|expert| expert.agent_name == agent_name)
}

pub fn delegation_help_summary() -> String {
    DELEGATION_EXPERTS
        .iter()
        .map(|expert| expert.usage)
        .collect::<Vec<_>>()
        .join(", ")
}

pub fn delegation_usage_list() -> String {
    let experts = DELEGATION_EXPERTS
        .iter()
        .map(|expert| expert.agent_name)
        .collect::<Vec<_>>()
        .join("|");
    format!("@<{experts}> <task>")
}

pub fn unknown_expert_error(agent_name: &str) -> String {
    let mut experts = DELEGATION_EXPERTS
        .iter()
        .map(|expert| expert.command)
        .collect::<Vec<_>>();
    let last = experts.pop().unwrap_or("@general");
    let experts = if experts.is_empty() {
        last.to_string()
    } else {
        format!("{}, or {last}", experts.join(", "))
    };
    format!("Unknown expert: @{agent_name}. Use {experts}.")
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
                "general"
            ]
        );
    }
}
