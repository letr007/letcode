use serde_json::Value;

pub fn format_tool_call(name: &str, args: &Value) -> String {
    match name {
        "fs__list" | "fs__read" | "fs__write" | "fs__append" | "fs__mkdir" => args
            .get("path")
            .and_then(Value::as_str)
            .map(|path| format!("{name} {path}"))
            .unwrap_or_else(|| format!("{name} {args}")),
        "search__rg" => {
            let pattern = args.get("pattern").and_then(Value::as_str).unwrap_or("");
            let path = args.get("path").and_then(Value::as_str).unwrap_or(".");
            format!("search__rg {:?} in {}", truncate_label(pattern, 60), path)
        }
        "git__status" => "git status".to_string(),
        "git__diff" => {
            let staged = args.get("staged").and_then(Value::as_bool).unwrap_or(false);
            let path = args.get("path").and_then(Value::as_str).unwrap_or("");
            let staged_flag = if staged { " --cached" } else { "" };
            format!("git diff{} {}", staged_flag, path)
                .trim()
                .to_string()
        }
        "git__log" => {
            let max_count = args.get("max_count").and_then(Value::as_u64).unwrap_or(10);
            format!("git log -{}", max_count)
        }
        "skill" => args
            .get("name")
            .and_then(Value::as_str)
            .map(|name| format!("Skill {:?}", truncate_label(name, 60)))
            .unwrap_or_else(|| "Skill".to_string()),
        "edit__apply_patch" => {
            let edits = args
                .get("edits")
                .and_then(Value::as_array)
                .map(Vec::len)
                .unwrap_or(0);
            format!(
                "edit__apply_patch {} edit{}",
                edits,
                if edits == 1 { "" } else { "s" }
            )
        }
        "code__ast_search" => {
            let pattern = args.get("pattern").and_then(Value::as_str).unwrap_or("");
            let path = args.get("path").and_then(Value::as_str).unwrap_or(".");
            format!(
                "code__ast_search {:?} in {}",
                truncate_label(pattern, 60),
                path
            )
        }
        "code__ast_replace_preview" => {
            let pattern = args.get("pattern").and_then(Value::as_str).unwrap_or("");
            let path = args.get("path").and_then(Value::as_str).unwrap_or(".");
            format!(
                "code__ast_replace_preview {:?} in {}",
                truncate_label(pattern, 60),
                path
            )
        }
        "shell__exec" => {
            let command = args.get("command").and_then(Value::as_str).unwrap_or("");
            format!("shell__exec {}", truncate_label(command, 120))
        }
        "context__list" => "context__list".to_string(),
        "context__search" => args
            .get("query")
            .and_then(Value::as_str)
            .map(|query| format!("context__search {:?}", truncate_label(query, 60)))
            .unwrap_or_else(|| "context__search".to_string()),
        "context__grep" => {
            let ref_id = args.get("ref_id").and_then(Value::as_str).unwrap_or("?");
            let query = args.get("query").and_then(Value::as_str).unwrap_or("");
            format!(
                "context__grep {} {:?}",
                truncate_label(ref_id, 80),
                truncate_label(query, 60)
            )
        }
        "context__open" => {
            let ref_type = args.get("ref_type").and_then(Value::as_str).unwrap_or("?");
            let ref_id = args.get("ref_id").and_then(Value::as_str).unwrap_or("?");
            format!("context__open {ref_type} {}", truncate_label(ref_id, 80))
        }
        "context__summarize" => {
            let artifact_id = args
                .get("artifact_id")
                .and_then(Value::as_str)
                .unwrap_or("?");
            format!("context__summarize {}", truncate_label(artifact_id, 80))
        }
        "context__pin" | "context__archive" | "context__remove" | "context__resolve" => args
            .get("block_id")
            .and_then(Value::as_str)
            .map(|block_id| format!("{name} {}", truncate_label(block_id, 80)))
            .unwrap_or_else(|| name.to_string()),
        "util__echo" => args
            .get("text")
            .and_then(Value::as_str)
            .map(|text| format!("util__echo {:?}", truncate_label(text, 60)))
            .unwrap_or_else(|| format!("util__echo {args}")),
        "question" => args
            .get("questions")
            .and_then(Value::as_array)
            .map(|questions| {
                format!(
                    "question {} question{}",
                    questions.len(),
                    if questions.len() == 1 { "" } else { "s" }
                )
            })
            .unwrap_or_else(|| "question".to_string()),
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
            format_tool_call("search__rg", &args),
            format!("search__rg {:?} in src", format!("{}…", "a".repeat(60)))
        );
    }

    #[test]
    fn formats_run_command_with_120_char_truncation() {
        let command = "x".repeat(121);
        let args = json!({ "command": command });

        assert_eq!(
            format_tool_call("shell__exec", &args),
            format!("shell__exec {}…", "x".repeat(120))
        );
    }

    #[test]
    fn formats_file_path_tools_using_path() {
        let args = json!({ "path": "src/main.rs" });

        assert_eq!(format_tool_call("fs__read", &args), "fs__read src/main.rs");
    }

    #[test]
    fn formats_apply_patch_with_pluralization() {
        assert_eq!(
            format_tool_call("edit__apply_patch", &json!({ "edits": [{}, {}] })),
            "edit__apply_patch 2 edits"
        );
        assert_eq!(
            format_tool_call("edit__apply_patch", &json!({ "edits": [{}] })),
            "edit__apply_patch 1 edit"
        );
    }

    #[test]
    fn formats_skill_load_with_skill_name() {
        assert_eq!(
            format_tool_call("skill", &json!({ "name": "git" })),
            "Skill \"git\""
        );
        assert_eq!(format_tool_call("skill", &json!({})), "Skill");
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
    fn formats_question_tool_concisely() {
        assert_eq!(
            format_tool_call(
                "question",
                &json!({"questions":[{"header":"One"},{"header":"Two"}]})
            ),
            "question 2 questions"
        );
    }

    #[test]
    fn formats_context_tools_concisely() {
        assert_eq!(
            format_tool_call("context__list", &json!({})),
            "context__list"
        );
        assert_eq!(
            format_tool_call(
                "context__grep",
                &json!({"ref_id":"folded-output-seq-2-stdout","query":"needle"})
            ),
            "context__grep folded-output-seq-2-stdout \"needle\""
        );
        assert_eq!(
            format_tool_call(
                "context__open",
                &json!({"ref_type":"block","ref_id":"block-seq-1-note"})
            ),
            "context__open block block-seq-1-note"
        );
    }

    #[test]
    fn truncates_labels_with_ellipsis_when_needed() {
        assert_eq!(truncate_label("hello", 10), "hello");
        assert_eq!(truncate_label("abcdef", 3), "abc…");
    }
}
