use crate::agent::is_subagent_tool_name;
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
        return ToolPresentation::CompactCard;
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
    matches!(tool_name, "workflow__todos" | "workflow__auto_continue")
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

}
