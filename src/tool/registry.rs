use anyhow::{Result, bail};
use serde_json::Value;
use std::collections::BTreeMap;
use std::sync::Arc;
use tracing::{debug, warn};

use super::{ToolExecutionContext, ToolHandler, ToolOutputEmitter, ToolOutputStream, ToolResult};
use crate::permission::{ToolPermissionClass, ToolScope, classify_tool};
use crate::request_builder::ToolSpec;
// Removed context-control tools remain reserved so MCP/dynamic tools cannot
// revive their names.
const RESERVED_DYNAMIC_TOOL_NAMES: [&str; 2] = ["context__checkpoint", "context__return"];

#[derive(Clone, Default)]
pub struct ToolRegistry {
    tools: BTreeMap<String, Arc<dyn ToolHandler>>,
    scope: ToolScope,
}

impl ToolRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn register<T>(&mut self, tool: T)
    where
        T: ToolHandler + 'static,
    {
        let name = tool.name().to_string();
        self.tools.insert(name, Arc::new(tool));
    }

    pub fn scoped(&self, scope: ToolScope) -> Self {
        Self {
            tools: self.tools.clone(),
            scope,
        }
    }

    pub fn without_tools(mut self, names: &[&str]) -> Self {
        for name in names {
            self.tools.remove(*name);
        }
        self
    }

    /// Remove a dynamically registered tool. Returns whether it was present.
    pub fn remove(&mut self, name: &str) -> bool {
        self.tools.remove(name).is_some()
    }

    pub fn scope(&self) -> ToolScope {
        self.scope
    }

    pub fn try_register<T>(&mut self, tool: T) -> Result<()>
    where
        T: ToolHandler + 'static,
    {
        let name = tool.name().to_string();
        if RESERVED_DYNAMIC_TOOL_NAMES.contains(&name.as_str()) {
            bail!("tool '{name}' is reserved and cannot be dynamically registered");
        }
        if self.tools.contains_key(&name) {
            bail!("tool '{name}' is already registered");
        }
        self.tools.insert(name, Arc::new(tool));
        Ok(())
    }

    pub fn specs(&self) -> Vec<ToolSpec> {
        self.tools
            .values()
            .filter(|tool| self.scope.allows_tool(tool.name()))
            .map(|tool| tool.spec())
            .collect()
    }

    pub fn permission_class(&self, name: &str) -> ToolPermissionClass {
        self.tools
            .get(name)
            .map(|tool| tool.permission_class())
            .unwrap_or_else(|| classify_tool(name))
    }

    pub fn contains(&self, name: &str) -> bool {
        self.tools.contains_key(name)
    }

    pub async fn call(&self, name: &str, args: Value) -> ToolResult {
        self.call_with_context(name, args, ToolExecutionContext::default())
            .await
    }

    pub async fn call_with_context(
        &self,
        name: &str,
        args: Value,
        context: ToolExecutionContext,
    ) -> ToolResult {
        let mut emit = |_stream: ToolOutputStream, _chunk: String| Ok(());
        self.call_streaming(name, args, context, &mut emit).await
    }

    pub async fn call_streaming(
        &self,
        name: &str,
        args: Value,
        context: ToolExecutionContext,
        emit: ToolOutputEmitter<'_>,
    ) -> ToolResult {
        debug!(tool_name = %name, args = %args, "calling tool");

        if !self.scope.allows_tool(name) {
            warn!(tool_name = %name, scope = %self.scope, "tool rejected by scope");
            return ToolResult::err(name, self.scope.rejection_message(name));
        }

        let Some(tool) = self.tools.get(name) else {
            warn!(tool_name = %name, "unknown tool requested");
            return ToolResult::err(name, format!("unknown tool: {name}"));
        };

        match tool.execute_streaming(args, context, emit).await {
            Ok(data) => ToolResult::ok(name, data),
            Err(err) => {
                warn!(tool_name = %name, error = %err, "tool execution failed");
                ToolResult::err(name, err.to_string())
            }
        }
    }
}

#[cfg(test)]
mod tests {
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
}
