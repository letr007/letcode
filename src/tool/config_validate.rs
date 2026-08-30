use anyhow::Result;
use async_trait::async_trait;
use serde_json::{Value, json};

use super::args::optional_string;
use super::{ToolHandler, ToolParallelism, ToolRegistry};
use crate::config::{self, validate_config_file};
use crate::permission::ToolPermissionClass;
use crate::tool_names;

pub(super) fn register(registry: &mut ToolRegistry) {
    registry.register(ConfigValidateTool);
}

struct ConfigValidateTool;

#[async_trait]
impl ToolHandler for ConfigValidateTool {
    fn name(&self) -> &'static str {
        tool_names::TOOL_CONFIG_VALIDATE
    }

    fn description(&self) -> &'static str {
        "Validate a letcode.toml with the same parser used at startup and hot-reload. Call after editing letcode configuration. Returns valid=true with a summary, or valid=false with the parse/validation error so you can fix the file and retry."
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "path": {
                    "type": ["string", "null"],
                    "description": "Absolute path to letcode.toml. Defaults to ~/.config/letcode/letcode.toml"
                }
            },
            "required": ["path"],
            "additionalProperties": false
        })
    }

    fn permission_class(&self) -> ToolPermissionClass {
        ToolPermissionClass::Read
    }

    fn parallelism(&self) -> ToolParallelism {
        ToolParallelism::Parallel
    }

    async fn execute(&self, args: Value) -> Result<Value> {
        let path = match optional_string(&args, "path").filter(|value| !value.trim().is_empty()) {
            Some(path) => std::path::PathBuf::from(path),
            None => config::default_config_path()?,
        };
        let report = validate_config_file(&path);
        Ok(serde_json::to_value(report)?)
    }
}
