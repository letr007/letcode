use super::*;
use async_trait::async_trait;
use serde_json::{Value, json};

struct DynamicTool(&'static str);

#[async_trait]
impl ToolHandler for DynamicTool {
    fn name(&self) -> &str {
        self.0
    }

    fn description(&self) -> &str {
        "dynamic test tool"
    }

    fn parameters(&self) -> Value {
        json!({"type": "object"})
    }

    async fn execute(&self, _args: Value) -> Result<Value> {
        Ok(json!({"executed": true}))
    }
}

#[tokio::test]
async fn rejects_reserved_context_tool_names_from_dynamic_registration() {
    let mut registry = ToolRegistry::new();

    for name in RESERVED_DYNAMIC_TOOL_NAMES {
        let error = registry
            .try_register(DynamicTool(name))
            .expect_err("reserved tool registration must fail");
        assert_eq!(
            error.to_string(),
            format!("tool '{name}' is reserved and cannot be dynamically registered")
        );
        assert!(!registry.contains(name));
        assert!(!registry.specs().iter().any(|spec| spec.name == name));

        let result = registry.call(name, json!({})).await;
        assert!(!result.ok);
        assert_eq!(
            result.error.expect("unknown tool error").message,
            format!("unknown tool: {name}")
        );
    }

    registry
        .try_register(DynamicTool("example__dynamic"))
        .expect("unreserved dynamic tool registration must succeed");
    assert!(registry.contains("example__dynamic"));
}

#[test]
fn registration_rejects_an_existing_parallel_override_for_exclusive_tool() {
    let mut registry = ToolRegistry::new();
    registry
        .set_parallelism_overrides([("example__dynamic".to_string(), ToolParallelism::Parallel)])
        .expect("unknown tool override remains pending until registration");

    let error = registry
        .try_register(DynamicTool("example__dynamic"))
        .expect_err("late registration must apply the existing override constraint");
    assert!(error.to_string().contains("does not declare parallel"));
    assert!(!registry.contains("example__dynamic"));
}
