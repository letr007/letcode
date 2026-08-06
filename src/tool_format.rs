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
        "web__fetch" => args
            .get("url")
            .and_then(Value::as_str)
            .map(|url| format!("web__fetch {}", truncate_label(url, 120)))
            .unwrap_or_else(|| "web__fetch".to_string()),
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
