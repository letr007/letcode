use crate::{
    mcp::{McpServerCatalogEntry, McpServerStatus},
    skills::SkillCard,
    tui::{i18n::Language, state::DialogItem},
};

pub fn mcp_dialog_items(
    servers: &[McpServerCatalogEntry],
    updating: &std::collections::HashSet<String>,
    language: Language,
) -> Vec<DialogItem> {
    let translator = crate::tui::i18n::Translator::new(language);
    servers
        .iter()
        .map(|server| {
            let status = if updating.contains(&server.name) {
                format!("◌ {}", translator.t("status.updating"))
            } else {
                match &server.status {
                    McpServerStatus::Disabled => format!("○ {}", translator.t("status.disabled")),
                    McpServerStatus::Online { tool_count } => translator.t_fmt(
                        "status.mcp_online_tools",
                        &[("count", &tool_count.to_string())],
                    ),
                    McpServerStatus::Offline { .. } => {
                        format!("● {}", translator.t("status.offline"))
                    }
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

pub fn mcp_tool_dialog_items(tools: &[crate::mcp::McpToolCatalogEntry]) -> Vec<DialogItem> {
    tools
        .iter()
        .map(|tool| {
            DialogItem::new(
                tool.name.clone(),
                tool.name.clone(),
                (!tool.description.is_empty()).then(|| tool.description.clone()),
            )
        })
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
            Language::En,
        );

        assert_eq!(items[0].label, "docs");
        assert_eq!(items[0].right_detail.as_deref(), Some("● Online · 2 tools"));
    }

    #[test]
    fn maps_language_picker_entries_with_endonyms_and_current_ids() {
        let items = [
            DialogItem::new("en", "English", None),
            DialogItem::new("zh-CN", "简体中文", None),
        ];
        assert_eq!(
            items
                .iter()
                .map(|item| item.id.as_str())
                .collect::<Vec<_>>(),
            vec!["en", "zh-CN"]
        );
        assert_eq!(items[0].label, "English");
        assert_eq!(items[1].label, "简体中文");
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
