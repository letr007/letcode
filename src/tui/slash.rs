use crate::command::command_metadata;
use crate::delegation::DELEGATION_EXPERTS;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SlashCommandEntry {
    pub command: &'static str,
    pub insert_text: &'static str,
    pub description: &'static str,
}

pub const MAX_VISIBLE_SLASH_COMMANDS: usize = 5;

pub fn slash_commands() -> Vec<SlashCommandEntry> {
    command_metadata()
        .iter()
        .filter(|command| command.visible_in_slash)
        .map(|command| SlashCommandEntry {
            command: command.name,
            insert_text: command.insert_text,
            description: command.description,
        })
        .collect()
}

pub fn slash_query(input: &str) -> Option<String> {
    let trimmed_start = input.trim_start();
    if !trimmed_start.starts_with('/') {
        return None;
    }

    Some(trimmed_start.trim_end().to_string())
}

pub fn expert_query(input: &str) -> Option<String> {
    let trimmed_start = input.trim_start();
    if !trimmed_start.starts_with('@') {
        return None;
    }

    Some(trimmed_start.trim_end().to_string())
}

pub fn completion_query(input: &str) -> Option<String> {
    slash_query(input).or_else(|| expert_query(input))
}

pub fn matching_slash_commands(input: &str) -> Vec<SlashCommandEntry> {
    let Some(query) = slash_query(input) else {
        return Vec::new();
    };

    slash_commands()
        .into_iter()
        .filter(|entry| entry.command.starts_with(query.as_str()))
        .collect()
}

pub fn matching_expert_commands(input: &str) -> Vec<SlashCommandEntry> {
    let Some(query) = expert_query(input) else {
        return Vec::new();
    };

    DELEGATION_EXPERTS
        .iter()
        .map(|expert| SlashCommandEntry {
            command: expert.command,
            insert_text: expert.insert_text,
            description: expert.description,
        })
        .filter(|entry| entry.command.starts_with(query.as_str()))
        .collect()
}

pub fn matching_completion_commands(input: &str) -> Vec<SlashCommandEntry> {
    if input.trim_start().starts_with('@') {
        matching_expert_commands(input)
    } else {
        matching_slash_commands(input)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matching_slash_commands_excludes_removed_subagent_commands() {
        assert!(matching_slash_commands("/exp").is_empty());
        assert!(matching_slash_commands("/fix").is_empty());
    }

    #[test]
    fn slash_registry_includes_child_navigation_commands() {
        let commands = slash_commands()
            .into_iter()
            .map(|entry| entry.command)
            .collect::<Vec<_>>();

        assert!(commands.contains(&"/child"));
        assert!(commands.contains(&"/children"));
        assert!(commands.contains(&"/parent"));
    }
}
