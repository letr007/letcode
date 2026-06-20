use crate::command::command_metadata;

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

pub fn matching_slash_commands(input: &str) -> Vec<SlashCommandEntry> {
    let Some(query) = slash_query(input) else {
        return Vec::new();
    };

    slash_commands()
        .into_iter()
        .filter(|entry| entry.command.starts_with(query.as_str()))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn slash_query_requires_leading_slash() {
        assert_eq!(slash_query("/help"), Some("/help".into()));
        assert_eq!(
            slash_query("   /permission s"),
            Some("/permission s".into())
        );
        assert_eq!(slash_query("hello"), None);
    }

    #[test]
    fn matching_slash_commands_filters_by_prefix() {
        let matches = matching_slash_commands("/permission s");
        assert!(matches.is_empty());
    }

    #[test]
    fn matching_slash_commands_includes_reasoning() {
        let matches = matching_slash_commands("/rea");
        assert_eq!(matches.len(), 1);
        assert_eq!(matches[0].command, "/reasoning");
    }

    #[test]
    fn matching_slash_commands_includes_explore() {
        let matches = matching_slash_commands("/exp");
        assert_eq!(matches.len(), 1);
        assert_eq!(matches[0].command, "/explore");
    }

    #[test]
    fn matching_slash_commands_includes_fixer() {
        let matches = matching_slash_commands("/fix");
        assert_eq!(matches.len(), 1);
        assert_eq!(matches[0].command, "/fixer");
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

    #[test]
    fn slash_registry_includes_tool_output_command() {
        let commands = slash_commands()
            .into_iter()
            .map(|entry| entry.command)
            .collect::<Vec<_>>();

        assert!(commands.contains(&"/tool-output"));
        assert!(commands.contains(&"/scrollbar"));
    }

    #[test]
    fn slash_registry_uses_shared_metadata_filter() {
        let commands = slash_commands()
            .into_iter()
            .map(|entry| entry.command)
            .collect::<Vec<_>>();

        assert!(commands.contains(&"/quit"));
        assert!(!commands.contains(&"/perm"));
        assert!(!commands.contains(&"/think"));
    }
}
