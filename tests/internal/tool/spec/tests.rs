use super::*;
use crate::permission::ToolPermissionClass;
use crate::tool::ToolRegistry;

#[test]
fn default_tools_all_declare_a_class() {
    let registry = ToolRegistry::default_tools();
    for spec in registry.specs() {
        assert!(
            tool_class(&spec.name).is_some(),
            "tool '{}' has no ToolClass entry",
            spec.name
        );
    }
}

#[test]
fn git_is_read_only_and_explorer_visible() {
    let class = tool_class(tool_names::TOOL_GIT).unwrap();
    assert_eq!(class.permission, ToolPermissionClass::Read);
    assert!(class.explorer_allowed);
    assert_eq!(class.effect, ToolEffectKind::Read);
}

#[test]
fn memory_recall_is_read_only_but_not_explorer_visible() {
    let class = tool_class(tool_names::TOOL_MEMORY_RECALL).unwrap();
    assert_eq!(class.permission, ToolPermissionClass::Read);
    assert!(!class.explorer_allowed);
}

#[test]
fn shell_exec_is_a_command_tool() {
    let class = tool_class(tool_names::TOOL_SHELL_EXEC).unwrap();
    assert_eq!(class.permission, ToolPermissionClass::Command);
    assert_eq!(class.effect, ToolEffectKind::Command);
}

#[test]
fn unknown_tools_have_no_class() {
    assert!(tool_class("does__not_exist").is_none());
}
