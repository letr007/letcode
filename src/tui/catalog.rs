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
#[path = "../../tests/internal/tui/catalog/tests.rs"]
mod tests;
