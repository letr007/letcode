#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SlashCommandEntry {
    pub command: &'static str,
    pub insert_text: &'static str,
    pub description: &'static str,
}

pub const MAX_VISIBLE_SLASH_COMMANDS: usize = 5;

const SLASH_COMMANDS: &[SlashCommandEntry] = &[
    SlashCommandEntry {
        command: "/help",
        insert_text: "/help",
        description: "Show available local commands",
    },
    SlashCommandEntry {
        command: "/exit",
        insert_text: "/exit",
        description: "Exit the TUI session",
    },
    SlashCommandEntry {
        command: "/quit",
        insert_text: "/quit",
        description: "Exit the TUI session",
    },
    SlashCommandEntry {
        command: "/permission",
        insert_text: "/permission ",
        description: "Show or switch permission mode",
    },
    SlashCommandEntry {
        command: "/model",
        insert_text: "/model ",
        description: "Show or switch the active model",
    },
    SlashCommandEntry {
        command: "/reasoning",
        insert_text: "/reasoning ",
        description: "Show or switch reasoning effort",
    },
    SlashCommandEntry {
        command: "/resume",
        insert_text: "/resume ",
        description: "Resume a previous session",
    },
    SlashCommandEntry {
        command: "/new",
        insert_text: "/new",
        description: "Start a new session",
    },
    SlashCommandEntry {
        command: "/explore",
        insert_text: "/explore ",
        description: "Run read-only explorer subagent",
    },
    SlashCommandEntry {
        command: "/child",
        insert_text: "/child",
        description: "View child subagent transcript",
    },
    SlashCommandEntry {
        command: "/children",
        insert_text: "/children",
        description: "View child subagent transcripts",
    },
    SlashCommandEntry {
        command: "/parent",
        insert_text: "/parent",
        description: "Return to parent transcript",
    },
];

pub fn slash_commands() -> &'static [SlashCommandEntry] {
    SLASH_COMMANDS
}

pub fn slash_query(input: &str) -> Option<String> {
    let trimmed_start = input.trim_start();
    if !trimmed_start.starts_with('/') {
        return None;
    }

    Some(trimmed_start.trim_end().to_string())
}

pub fn matching_slash_commands(input: &str) -> Vec<&'static SlashCommandEntry> {
    let Some(query) = slash_query(input) else {
        return Vec::new();
    };

    slash_commands()
        .iter()
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
    fn slash_registry_includes_child_navigation_commands() {
        let commands = slash_commands()
            .iter()
            .map(|entry| entry.command)
            .collect::<Vec<_>>();

        assert!(commands.contains(&"/child"));
        assert!(commands.contains(&"/children"));
        assert!(commands.contains(&"/parent"));
    }
}
