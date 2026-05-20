use serde_json::Value;

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

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct PresentationPolicy;

impl PresentationPolicy {
    pub fn tool_presentation(&self, context: &ToolPresentationContext) -> ToolPresentation {
        let class = crate::permission::classify_tool(&context.name);

        match context.status {
            ToolPresentationStatus::Pending => ToolPresentation::CompactCard,
            ToolPresentationStatus::Running => match class {
                crate::permission::ToolPermissionClass::Read
                | crate::permission::ToolPermissionClass::Preview => ToolPresentation::Inline,
                crate::permission::ToolPermissionClass::Write
                | crate::permission::ToolPermissionClass::Command
                | crate::permission::ToolPermissionClass::Unknown => ToolPresentation::CompactCard,
            },
            ToolPresentationStatus::Succeeded => {
                if is_quiet_success(context) {
                    ToolPresentation::Hidden
                } else {
                    match class {
                        crate::permission::ToolPermissionClass::Read
                        | crate::permission::ToolPermissionClass::Preview => {
                            ToolPresentation::Inline
                        }
                        crate::permission::ToolPermissionClass::Write
                        | crate::permission::ToolPermissionClass::Command
                        | crate::permission::ToolPermissionClass::Unknown => {
                            ToolPresentation::CompactCard
                        }
                    }
                }
            }
            ToolPresentationStatus::Failed => ToolPresentation::CompactCard,
        }
    }
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

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn read_tool_running_is_inline() {
        let policy = PresentationPolicy;
        let mut context =
            ToolPresentationContext::new("read_file", ToolPresentationStatus::Running);
        context.args = Some(json!({"path": "src/main.rs"}));

        assert_eq!(context.summary(), "read_file src/main.rs");
        assert_eq!(policy.tool_presentation(&context), ToolPresentation::Inline);
    }

    #[test]
    fn command_tool_success_is_compact_card() {
        let policy = PresentationPolicy;
        let mut context =
            ToolPresentationContext::new("run_command", ToolPresentationStatus::Succeeded);
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
        let mut context = ToolPresentationContext::new("echo", ToolPresentationStatus::Succeeded);
        context.output = Some(json!({}));

        assert_eq!(policy.tool_presentation(&context), ToolPresentation::Hidden);
    }

    #[test]
    fn failures_are_never_hidden() {
        let policy = PresentationPolicy;
        let mut context = ToolPresentationContext::new("read_file", ToolPresentationStatus::Failed);
        context.output = Some(json!({"error": "not found"}));

        assert_eq!(
            policy.tool_presentation(&context),
            ToolPresentation::CompactCard
        );
    }
}
