use crate::{
    mcp::{McpServerCatalogEntry, McpServerStatus},
    skills::SkillCard,
    tui::state::DialogItem,
};

pub fn mcp_dialog_items(
    servers: &[McpServerCatalogEntry],
    updating: &std::collections::HashSet<String>,
) -> Vec<DialogItem> {
    servers
        .iter()
        .map(|server| {
            let status = if updating.contains(&server.name) {
                "◌ Updating".into()
            } else {
                match &server.status {
                    McpServerStatus::Disabled => "○ Disabled".into(),
                    McpServerStatus::Online { tool_count } => {
                        format!("● Online · {tool_count} tools")
                    }
                    McpServerStatus::Offline { .. } => "● Offline".into(),
                }
            };
            DialogItem::new(server.name.clone(), server.name.clone(), None)
                .with_right_detail(status)
        })
        .collect()
}

pub fn skill_dialog_items(skills: &[SkillCard]) -> Vec<DialogItem> {
    skills
        .iter()
        .map(|skill| DialogItem::new(skill.name.clone(), skill.name.clone(), None))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn maps_mcp_servers_to_compact_status_rows() {
        let items = mcp_dialog_items(
            &[McpServerCatalogEntry {
                name: "docs".into(),
                enabled: true,
                status: McpServerStatus::Online { tool_count: 2 },
            }],
            &std::collections::HashSet::new(),
        );

        assert_eq!(items[0].label, "docs");
        assert_eq!(items[0].right_detail.as_deref(), Some("● Online · 2 tools"));
    }

    #[test]
    fn maps_skill_cards_with_location_and_path() {
        let items = skill_dialog_items(&[SkillCard {
            name: "rust-audit".into(),
            description: "Review Rust code".into(),
            location: ".agents/skills".into(),
            path: std::path::PathBuf::from("/repo/.agents/skills/rust-audit/SKILL.md"),
        }]);

        assert_eq!(items[0].label, "rust-audit");
        assert!(items[0].detail.is_none());
    }
}
