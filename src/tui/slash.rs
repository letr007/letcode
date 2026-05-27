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
        description: "Show current permission mode",
    },
    SlashCommandEntry {
        command: "/permission safe",
        insert_text: "/permission safe",
        description: "Ask before risky tools",
    },
    SlashCommandEntry {
        command: "/permission default",
        insert_text: "/permission default",
        description: "Restore normal permission behavior",
    },
    SlashCommandEntry {
        command: "/permission solo --yes",
        insert_text: "/permission solo --yes",
        description: "Fully enable write and command tools",
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
        assert_eq!(matches.len(), 2);
        assert_eq!(matches[0].command, "/permission safe");
        assert_eq!(matches[1].command, "/permission solo --yes");
    }
}
