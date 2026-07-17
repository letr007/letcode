use anyhow::{Result, bail};
use serde_json::Value;
use std::collections::BTreeMap;
use std::sync::Arc;
use tracing::{debug, warn};

use super::{ToolExecutionContext, ToolHandler, ToolOutputEmitter, ToolOutputStream, ToolResult};
use crate::permission::{ToolPermissionClass, ToolScope, classify_tool};
use crate::request_builder::ToolSpec;

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

    pub fn scope(&self) -> ToolScope {
        self.scope
    }

    pub fn try_register<T>(&mut self, tool: T) -> Result<()>
    where
        T: ToolHandler + 'static,
    {
        let name = tool.name().to_string();
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
