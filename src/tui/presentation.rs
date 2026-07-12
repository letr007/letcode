use serde_json::Value;

use crate::agent::is_subagent_tool_name;
use crate::tool_format::format_tool_call;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ToolPresentation {
    Hidden,
    Inline,
    CompactCard,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ToolPresentationStatus {
    Pending,
    Running,
    Succeeded,
    Failed,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ToolPresentationContext {
    pub name: String,
    pub status: ToolPresentationStatus,
    pub args: Option<Value>,
    pub output: Option<Value>,
}

impl ToolPresentationContext {
    pub fn new(name: impl Into<String>, status: ToolPresentationStatus) -> Self {
        Self {
            name: name.into(),
            status,
            args: None,
            output: None,
        }
    }

    pub fn summary(&self) -> String {
        self.args
            .as_ref()
            .map(|args| format_tool_call(&self.name, args))
            .unwrap_or_else(|| self.name.clone())
    }
}

/// Render-facing context for TUI timeline tool items.
///
/// The TUI currently stores tool arguments/output as already-formatted text in the
/// timeline view models (not structured JSON). This context lets PresentationPolicy
/// own the single source of truth for whether a tool should be shown and how much
/// detail is reasonable by default.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ToolTextPresentationContext {
    pub name: String,
    pub status: ToolPresentationStatus,
    pub arguments: Option<String>,
    pub output: Option<String>,
}

impl ToolTextPresentationContext {
    pub fn new(name: impl Into<String>, status: ToolPresentationStatus) -> Self {
        Self {
            name: name.into(),
            status,
            arguments: None,
            output: None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct PresentationPolicy;

impl PresentationPolicy {
    pub fn tool_presentation(&self, context: &ToolPresentationContext) -> ToolPresentation {
        tool_presentation_impl(&context.name, context.status, is_quiet_success(context))
    }

    pub fn tool_presentation_text(
        &self,
        context: &ToolTextPresentationContext,
    ) -> ToolPresentation {
        tool_presentation_impl(
            &context.name,
            context.status,
            is_quiet_success_text(context.output.as_deref()),
        )
    }
}

fn tool_presentation_impl(
    tool_name: &str,
    status: ToolPresentationStatus,
    is_quiet_success: bool,
) -> ToolPresentation {
    use crate::permission::ToolPermissionClass;

    // A completed user question is part of the conversation's durable decision trail.
    // It must remain visible even when a generic low-risk tool result would be quiet.
    if tool_name == crate::tool_names::TOOL_QUESTION {
        return ToolPresentation::CompactCard;
    }

    if is_workflow_control_tool(tool_name) {
        return match status {
            ToolPresentationStatus::Pending | ToolPresentationStatus::Running => {
                ToolPresentation::CompactCard
            }
            ToolPresentationStatus::Succeeded => ToolPresentation::Hidden,
            ToolPresentationStatus::Failed => ToolPresentation::CompactCard,
        };
    }

    let class = crate::permission::classify_tool(tool_name);

    if is_subagent_tool_name(tool_name) {
        return ToolPresentation::CompactCard;
    }

    match status {
        ToolPresentationStatus::Pending => ToolPresentation::CompactCard,
        ToolPresentationStatus::Running => match class {
            ToolPermissionClass::Read | ToolPermissionClass::Preview => ToolPresentation::Inline,
            ToolPermissionClass::Write
            | ToolPermissionClass::Command
            | ToolPermissionClass::Unknown => ToolPresentation::CompactCard,
        },
        ToolPresentationStatus::Succeeded => {
            // Safety/audit trail: never hide write/command/unknown tool executions.
            // Quiet success hiding is only allowed for low-risk read/preview tools.
            if is_quiet_success {
                match class {
                    ToolPermissionClass::Read | ToolPermissionClass::Preview => {
                        ToolPresentation::Hidden
                    }
                    ToolPermissionClass::Write
                    | ToolPermissionClass::Command
                    | ToolPermissionClass::Unknown => ToolPresentation::CompactCard,
                }
            } else {
                match class {
                    ToolPermissionClass::Read | ToolPermissionClass::Preview => {
                        ToolPresentation::Inline
                    }
                    ToolPermissionClass::Write
                    | ToolPermissionClass::Command
                    | ToolPermissionClass::Unknown => ToolPresentation::CompactCard,
                }
            }
        }
        ToolPresentationStatus::Failed => ToolPresentation::CompactCard,
    }
}

fn is_workflow_control_tool(tool_name: &str) -> bool {
    matches!(
        tool_name,
        "workflow__todos" | "workflow__auto_continue" | "context__checkpoint" | "context__return"
    )
}

fn is_quiet_success(context: &ToolPresentationContext) -> bool {
    if context.status != ToolPresentationStatus::Succeeded {
        return false;
    }

    let Some(output) = context.output.as_ref() else {
        return true;
    };

    output.is_null()
        || output.as_object().is_some_and(|obj| obj.is_empty())
        || output
            .get("result")
            .and_then(Value::as_str)
            .is_some_and(str::is_empty)
}

fn is_quiet_success_text(output: Option<&str>) -> bool {
    let Some(output) = output else {
        return true;
    };
    output.trim().is_empty()
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn read_tool_running_is_inline() {
        let policy = PresentationPolicy;
        let mut context = ToolPresentationContext::new("fs__read", ToolPresentationStatus::Running);
        context.args = Some(json!({"path": "src/main.rs"}));

        assert_eq!(context.summary(), "fs__read src/main.rs");
        assert_eq!(policy.tool_presentation(&context), ToolPresentation::Inline);
    }

    #[test]
    fn completed_question_is_never_hidden_as_a_quiet_success() {
        let policy = PresentationPolicy;
        let context =
            ToolTextPresentationContext::new("question", ToolPresentationStatus::Succeeded);

        assert_eq!(
            policy.tool_presentation_text(&context),
            ToolPresentation::CompactCard
        );
    }

    #[test]
    fn command_tool_success_is_compact_card() {
        let policy = PresentationPolicy;
        let mut context =
            ToolPresentationContext::new("shell__exec", ToolPresentationStatus::Succeeded);
        context.args = Some(json!({"command": "cargo check"}));
        context.output = Some(json!({"stdout": "ok"}));

        assert_eq!(
            policy.tool_presentation(&context),
            ToolPresentation::CompactCard
        );
    }

    #[test]
    fn quiet_success_can_be_hidden() {
        let policy = PresentationPolicy;
        let mut context =
            ToolPresentationContext::new("util__echo", ToolPresentationStatus::Succeeded);
        context.output = Some(json!({}));

        assert_eq!(policy.tool_presentation(&context), ToolPresentation::Hidden);
    }

    #[test]
    fn quiet_success_write_like_tools_are_never_hidden() {
        let policy = PresentationPolicy;
        let mut context =
            ToolPresentationContext::new("fs__write", ToolPresentationStatus::Succeeded);
        context.output = Some(json!({}));
        assert_eq!(
            policy.tool_presentation(&context),
            ToolPresentation::CompactCard
        );

        let mut unknown =
            ToolPresentationContext::new("some_unknown_tool", ToolPresentationStatus::Succeeded);
        unknown.output = Some(json!({}));
        assert_eq!(
            policy.tool_presentation(&unknown),
            ToolPresentation::CompactCard
        );
    }

    #[test]
    fn failures_are_never_hidden() {
        let policy = PresentationPolicy;
        let mut context = ToolPresentationContext::new("fs__read", ToolPresentationStatus::Failed);
        context.output = Some(json!({"error": "not found"}));

        assert_eq!(
            policy.tool_presentation(&context),
            ToolPresentation::CompactCard
        );
    }

    #[test]
    fn quiet_success_text_can_be_hidden() {
        let policy = PresentationPolicy;
        let mut context =
            ToolTextPresentationContext::new("util__echo", ToolPresentationStatus::Succeeded);
        context.output = Some("\n".into());
        assert_eq!(
            policy.tool_presentation_text(&context),
            ToolPresentation::Hidden
        );
    }

    #[test]
    fn quiet_success_text_write_like_tools_are_never_hidden() {
        let policy = PresentationPolicy;
        let mut ctx =
            ToolTextPresentationContext::new("shell__exec", ToolPresentationStatus::Succeeded);
        ctx.output = Some("\n".into());
        assert_eq!(
            policy.tool_presentation_text(&ctx),
            ToolPresentation::CompactCard
        );
    }

    #[test]
    fn workflow_control_tools_follow_pending_running_finished_visibility() {
        let policy = PresentationPolicy;

        for (status, expected) in [
            (
                ToolPresentationStatus::Pending,
                ToolPresentation::CompactCard,
            ),
            (
                ToolPresentationStatus::Running,
                ToolPresentation::CompactCard,
            ),
            (ToolPresentationStatus::Succeeded, ToolPresentation::Hidden),
            (
                ToolPresentationStatus::Failed,
                ToolPresentation::CompactCard,
            ),
        ] {
            for tool_name in [
                "workflow__todos",
                "workflow__auto_continue",
                "context__checkpoint",
            ] {
                let context = ToolPresentationContext::new(tool_name, status);
                assert_eq!(policy.tool_presentation(&context), expected);

                let text_context = ToolTextPresentationContext::new(tool_name, status);
                assert_eq!(policy.tool_presentation_text(&text_context), expected);
            }
        }
    }

    #[test]
    fn subagent_tools_always_use_compact_cards_for_status_surfaces() {
        let policy = PresentationPolicy;

        for status in [
            ToolPresentationStatus::Pending,
            ToolPresentationStatus::Running,
            ToolPresentationStatus::Succeeded,
            ToolPresentationStatus::Failed,
        ] {
            let context = ToolPresentationContext::new("agent__fixer", status);
            assert_eq!(
                policy.tool_presentation(&context),
                ToolPresentation::CompactCard
            );
        }
    }
}
