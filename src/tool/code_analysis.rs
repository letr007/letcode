use anyhow::Result;
use async_trait::async_trait;
use serde_json::{Value, json};

use super::args::{optional_usize, required_string};
use super::{
    ToolExecutionContext, ToolHandler, ToolParallelism, ToolRegistry,
};
use crate::code_analysis::{AstReplacePreviewRequest, AstSearchRequest, CodeAnalysisRegistry};

pub(super) fn register(registry: &mut ToolRegistry) {
    registry.register(AstSearchTool);
    registry.register(AstReplacePreviewTool);
}

struct AstSearchTool;

#[async_trait]
impl ToolHandler for AstSearchTool {
    fn name(&self) -> &'static str {
        "code__ast_search"
    }

    fn description(&self) -> &'static str {
        "Search code with a language-agnostic AST-aware pattern using the configured code analysis backend. Currently uses ast-grep CLI when available. Patterns are code, not regex, and can use metavariables like $A or $$$ARGS. This tool does not modify files."
    }

    fn parallelism(&self) -> ToolParallelism {
        ToolParallelism::Parallel
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Workspace-relative file or directory path to search, e.g. src or ."
                },
                "language": {
                    "type": "string",
                    "description": "Language name/alias for ast-grep, e.g. rust, typescript, python, go; use auto to infer from file extensions"
                },
                "pattern": {
                    "type": "string",
                    "description": "AST pattern written as valid code, e.g. self.tools.call($NAME, $ARGS).await"
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum matches to return, capped at 1000"
                }
            },
            "required": ["path", "language", "pattern", "max_results"],
            "additionalProperties": false
        })
    }

    async fn execute(&self, args: Value) -> Result<Value> {
        self.execute_with_context(args, ToolExecutionContext::default())
            .await
    }

    async fn execute_with_context(
        &self,
        args: Value,
        context: ToolExecutionContext,
    ) -> Result<Value> {
        CodeAnalysisRegistry::default_backends()
            .ast_search(AstSearchRequest {
                path: required_string(&args, "path")?.to_string(),
                language: Some(required_string(&args, "language")?.to_string()),
                pattern: required_string(&args, "pattern")?.to_string(),
                max_results: optional_usize(&args, "max_results").unwrap_or(100),
                allow_outside_workspace: context.allow_outside_workspace,
            })
            .await
    }
}

struct AstReplacePreviewTool;

#[async_trait]
impl ToolHandler for AstReplacePreviewTool {
    fn name(&self) -> &'static str {
        "code__ast_replace_preview"
    }

    fn description(&self) -> &'static str {
        "Preview an AST-aware rewrite with the configured code analysis backend. This returns a diff preview only and does not write files. Use edit__apply_patch for audited edits."
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Workspace-relative file or directory path to preview rewrites in"
                },
                "language": {
                    "type": "string",
                    "description": "Language name/alias for ast-grep, or auto to infer from file extensions"
                },
                "pattern": {
                    "type": "string",
                    "description": "AST pattern written as valid code, e.g. console.log($MSG)"
                },
                "rewrite": {
                    "type": "string",
                    "description": "Rewrite pattern, e.g. logger.info($MSG)"
                }
            },
            "required": ["path", "language", "pattern", "rewrite"],
            "additionalProperties": false
        })
    }

    async fn execute(&self, args: Value) -> Result<Value> {
        self.execute_with_context(args, ToolExecutionContext::default())
            .await
    }

    async fn execute_with_context(
        &self,
        args: Value,
        context: ToolExecutionContext,
    ) -> Result<Value> {
        CodeAnalysisRegistry::default_backends()
            .ast_replace_preview(AstReplacePreviewRequest {
                path: required_string(&args, "path")?.to_string(),
                language: Some(required_string(&args, "language")?.to_string()),
                pattern: required_string(&args, "pattern")?.to_string(),
                rewrite: required_string(&args, "rewrite")?.to_string(),
                allow_outside_workspace: context.allow_outside_workspace,
            })
            .await
    }
}
