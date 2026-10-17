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
    assert!(!commands.contains(&"/children"));
    assert!(!commands.contains(&"/parent"));
}
