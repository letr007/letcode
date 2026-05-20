use serde_json::Value;

pub fn format_tool_call(name: &str, args: &Value) -> String {
    match name {
        "list_dir" | "read_file" | "write_file" | "append_file" | "mkdir" => args
            .get("path")
            .and_then(Value::as_str)
            .map(|path| format!("{name} {path}"))
            .unwrap_or_else(|| format!("{name} {args}")),
        "rg" => {
            let pattern = args.get("pattern").and_then(Value::as_str).unwrap_or("");
            let path = args.get("path").and_then(Value::as_str).unwrap_or(".");
            format!("rg {:?} in {}", truncate_label(pattern, 60), path)
        }
        "git_status" => "git status".to_string(),
        "git_diff" => {
            let staged = args.get("staged").and_then(Value::as_bool).unwrap_or(false);
            let path = args.get("path").and_then(Value::as_str).unwrap_or("");
            let staged_flag = if staged { " --cached" } else { "" };
            format!("git diff{} {}", staged_flag, path)
                .trim()
                .to_string()
        }
        "git_log" => {
            let max_count = args.get("max_count").and_then(Value::as_u64).unwrap_or(10);
            format!("git log -{}", max_count)
        }
        "apply_patch" => {
            let edits = args
                .get("edits")
                .and_then(Value::as_array)
                .map(Vec::len)
                .unwrap_or(0);
            format!(
                "apply_patch {} edit{}",
                edits,
                if edits == 1 { "" } else { "s" }
            )
        }
        "ast_search" => {
            let pattern = args.get("pattern").and_then(Value::as_str).unwrap_or("");
            let path = args.get("path").and_then(Value::as_str).unwrap_or(".");
            format!("ast_search {:?} in {}", truncate_label(pattern, 60), path)
        }
        "ast_replace_preview" => {
            let pattern = args.get("pattern").and_then(Value::as_str).unwrap_or("");
            let path = args.get("path").and_then(Value::as_str).unwrap_or(".");
            format!(
                "ast_replace_preview {:?} in {}",
                truncate_label(pattern, 60),
                path
            )
        }
        "run_command" => {
            let command = args.get("command").and_then(Value::as_str).unwrap_or("");
            format!("run_command {}", truncate_label(command, 120))
        }
        "echo" => args
            .get("text")
            .and_then(Value::as_str)
            .map(|text| format!("echo {:?}", truncate_label(text, 60)))
            .unwrap_or_else(|| format!("echo {args}")),
        _ => format!("{name} {args}"),
    }
}

pub fn truncate_label(text: &str, max_chars: usize) -> String {
    let mut truncated = text.chars().take(max_chars).collect::<String>();
    if text.chars().count() > max_chars {
        truncated.push('…');
    }
    truncated
}

#[cfg(test)]
mod tests {
    use super::{format_tool_call, truncate_label};
    use serde_json::json;

    #[test]
    fn formats_rg_with_truncated_pattern_and_path() {
        let pattern = "a".repeat(61);
        let args = json!({ "pattern": pattern, "path": "src" });

        assert_eq!(
            format_tool_call("rg", &args),
            format!("rg {:?} in src", format!("{}…", "a".repeat(60)))
        );
    }

    #[test]
    fn formats_run_command_with_120_char_truncation() {
        let command = "x".repeat(121);
        let args = json!({ "command": command });

        assert_eq!(
            format_tool_call("run_command", &args),
            format!("run_command {}…", "x".repeat(120))
        );
    }

    #[test]
    fn formats_file_path_tools_using_path() {
        let args = json!({ "path": "src/main.rs" });

        assert_eq!(
            format_tool_call("read_file", &args),
            "read_file src/main.rs"
        );
    }

    #[test]
    fn formats_apply_patch_with_pluralization() {
        assert_eq!(
            format_tool_call("apply_patch", &json!({ "edits": [{}, {}] })),
            "apply_patch 2 edits"
        );
        assert_eq!(
            format_tool_call("apply_patch", &json!({ "edits": [{}] })),
            "apply_patch 1 edit"
        );
    }

    #[test]
    fn formats_unknown_tools_with_fallback() {
        let args = json!({ "flag": true });

        assert_eq!(
            format_tool_call("custom_tool", &args),
            "custom_tool {\"flag\":true}"
        );
    }

    #[test]
    fn truncates_labels_with_ellipsis_when_needed() {
        assert_eq!(truncate_label("hello", 10), "hello");
        assert_eq!(truncate_label("abcdef", 3), "abc…");
    }
}
