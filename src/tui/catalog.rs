use crate::{mcp::McpToolCard, skills::SkillCard, tui::state::DialogItem};

pub fn mcp_dialog_items(tools: &[McpToolCard]) -> Vec<DialogItem> {
    tools
        .iter()
        .map(|tool| {
            DialogItem::new(
                tool.registered_name.clone(),
                tool.name.clone(),
                Some(format!("{} · {}", tool.server, tool.source)),
            )
            .with_right_detail(tool.description.clone())
            .with_inspect_detail(format!(
                "Description\n{}\n\nServer\n{}\n\nSource\n{}\n\nRegistered name\n{}\n\nParameters\n{}",
                tool.description,
                tool.server,
                tool.source,
                tool.registered_name,
                serde_json::to_string_pretty(&tool.parameters)
                    .unwrap_or_else(|_| tool.parameters.to_string())
            ))
        })
        .collect()
}

pub fn skill_dialog_items(skills: &[SkillCard]) -> Vec<DialogItem> {
    skills
        .iter()
        .map(|skill| {
            let path = skill.path.display().to_string();
            DialogItem::new(
                skill.name.clone(),
                skill.name.clone(),
                Some(format!("{} · {}", skill.location, path)),
            )
            .with_right_detail(skill.description.clone())
            .with_inspect_detail(format!(
                "Description\n{}\n\nLocation\n{}\n\nPath\n{}",
                skill.description, skill.location, path
            ))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use std::path::PathBuf;

    #[test]
    fn maps_mcp_cards_to_searchable_inspectable_items() {
        let items = mcp_dialog_items(&[McpToolCard {
            name: "lookup-docs".into(),
            registered_name: "docs__lookup_docs".into(),
            description: "Find documentation".into(),
            server: "docs".into(),
            source: "Remote · https://example.test/mcp".into(),
            parameters: json!({"type":"object","required":["query"]}),
        }]);

        assert_eq!(items[0].label, "lookup-docs");
        assert!(items[0].detail.as_deref().unwrap().contains("docs"));
        assert!(
            items[0]
                .inspect_detail
                .as_deref()
                .unwrap()
                .contains("\"query\"")
        );
    }

    #[test]
    fn maps_skill_cards_with_location_and_path() {
        let items = skill_dialog_items(&[SkillCard {
            name: "rust-audit".into(),
            description: "Review Rust code".into(),
            location: ".agents/skills".into(),
            path: PathBuf::from("/repo/.agents/skills/rust-audit/SKILL.md"),
        }]);

        assert!(
            items[0]
                .detail
                .as_deref()
                .unwrap()
                .contains(".agents/skills")
        );
        assert!(
            items[0]
                .inspect_detail
                .as_deref()
                .unwrap()
                .contains("SKILL.md")
        );
    }
}
