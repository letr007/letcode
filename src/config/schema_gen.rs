//! Injects schema `default`/`enum` values from their Rust authorities into
//! `letcode.schema.json`.
//!
//! The JSON file keeps its own layout: only members whose value differs from the
//! Rust authority are rewritten, and a missing `default` is inserted next to the
//! existing members. Structure and descriptions are left untouched.

use std::path::Path;

use anyhow::{Context, Result, anyhow, bail};
use serde::Serialize;
use serde_json::{Value, json};

use super::{ApiProtocol, PromptCacheRetention, RawMcpServerKind, SubagentMode, SubagentsConfig};
use crate::model_runtime::strategy::ModelStrategyId;
use crate::model_runtime::{
    AuthScheme, ProviderFlavor, ProviderReviewerBackend, StructuredOutputSupport,
};
use crate::permission::PermissionMode;
use crate::request_builder::{ModelReasoningSummary, ModelTextVerbosity};
use crate::session::archive::SessionArchiveConfig;
use crate::tool::ToolParallelism;

/// Repository-relative path of the published configuration schema.
pub(crate) const SCHEMA_FILE: &str = "letcode.schema.json";

enum Injection {
    Default(Value),
    Enum(Vec<String>),
}

struct Target {
    pointer: &'static [&'static str],
    injection: Injection,
}

fn targets() -> Result<Vec<Target>> {
    let archive = SessionArchiveConfig::default();
    let subagents = SubagentsConfig::default();
    Ok(vec![
        Target {
            pointer: &["definitions", "subagents", "properties", "mode"],
            injection: Injection::Enum(wire_values(&[SubagentMode::Expert, SubagentMode::Swarm])?),
        },
        Target {
            pointer: &["definitions", "subagents", "properties", "mode"],
            injection: Injection::Default(wire(&subagents.mode)?),
        },
        Target {
            pointer: &["definitions", "subagents", "properties", "max_depth"],
            injection: Injection::Default(json!(subagents.max_depth)),
        },
        Target {
            pointer: &["definitions", "subagents", "properties", "max_concurrent"],
            injection: Injection::Default(json!(subagents.max_concurrent)),
        },
        Target {
            pointer: &[
                "definitions",
                "subagents",
                "properties",
                "max_pending_messages",
            ],
            injection: Injection::Default(json!(subagents.max_pending_messages)),
        },
        Target {
            pointer: &[
                "definitions",
                "subagents",
                "properties",
                "max_message_bytes",
            ],
            injection: Injection::Default(json!(subagents.max_message_bytes)),
        },
        Target {
            pointer: &["definitions", "subagents", "properties", "run_timeout_secs"],
            injection: Injection::Default(json!(subagents.run_timeout_secs)),
        },
        Target {
            pointer: &["definitions", "protocol"],
            // An omitted protocol resolves to completions at load time.
            injection: Injection::Default(wire(&ApiProtocol::Completions)?),
        },
        Target {
            pointer: &["definitions", "protocol"],
            injection: Injection::Enum(wire_values(&[
                ApiProtocol::Responses,
                ApiProtocol::Completions,
                ApiProtocol::Anthropic,
            ])?),
        },
        Target {
            pointer: &["definitions", "permissions", "properties", "mode"],
            injection: Injection::Enum(permission_mode_values()?),
        },
        Target {
            pointer: &[
                "definitions",
                "tools",
                "properties",
                "parallelism",
                "additionalProperties",
            ],
            injection: Injection::Enum(wire_values(&[
                ToolParallelism::Parallel,
                ToolParallelism::Exclusive,
            ])?),
        },
        Target {
            pointer: &["definitions", "mcpServer", "properties", "type"],
            injection: Injection::Enum(wire_values(&[
                RawMcpServerKind::Local,
                RawMcpServerKind::Remote,
            ])?),
        },
        Target {
            pointer: &["definitions", "flavor"],
            injection: Injection::Enum(wire_values(&[
                ProviderFlavor::Standard,
                ProviderFlavor::Deepseek,
            ])?),
        },
        Target {
            pointer: &["definitions", "auth", "properties", "type"],
            injection: Injection::Enum(wire_values(&[
                AuthScheme::Bearer,
                AuthScheme::Header,
                AuthScheme::Query,
                AuthScheme::None,
            ])?),
        },
        Target {
            pointer: &["definitions", "strategy"],
            injection: Injection::Enum(wire_values(&[
                ModelStrategyId::Default,
                ModelStrategyId::Astra,
                ModelStrategyId::Deepseek,
            ])?),
        },
        Target {
            pointer: &["definitions", "cache", "properties", "retention"],
            injection: Injection::Enum(wire_values(&[
                PromptCacheRetention::InMemory,
                PromptCacheRetention::TwentyFourHours,
            ])?),
        },
        Target {
            pointer: &[
                "definitions",
                "generationCapabilities",
                "properties",
                "structured_output",
            ],
            injection: Injection::Enum(wire_values(&[
                StructuredOutputSupport::JsonObject,
                StructuredOutputSupport::JsonSchema,
            ])?),
        },
        Target {
            pointer: &[
                "definitions",
                "generation",
                "properties",
                "reasoning_summary",
            ],
            injection: Injection::Enum(wire_values(&[
                ModelReasoningSummary::Auto,
                ModelReasoningSummary::Concise,
                ModelReasoningSummary::Detailed,
            ])?),
        },
        Target {
            pointer: &["definitions", "generation", "properties", "text_verbosity"],
            injection: Injection::Enum(wire_values(&[
                ModelTextVerbosity::Low,
                ModelTextVerbosity::Medium,
                ModelTextVerbosity::High,
            ])?),
        },
        Target {
            pointer: &["definitions", "provider", "properties", "reviewer"],
            injection: Injection::Enum(wire_values(&[ProviderReviewerBackend::Jev])?),
        },
        Target {
            pointer: &["definitions", "model", "properties", "context_window"],
            injection: Injection::Default(json!(
                crate::request_builder::DEFAULT_FALLBACK_CONTEXT_WINDOW_TOKENS
            )),
        },
        Target {
            pointer: &["definitions", "global", "properties", "tool_timeout_secs"],
            injection: Injection::Default(json!(super::DEFAULT_TOOL_TIMEOUT_SECS)),
        },
        Target {
            pointer: &[
                "definitions",
                "global",
                "properties",
                "session_archive",
                "properties",
                "enabled",
            ],
            injection: Injection::Default(json!(archive.enabled)),
        },
        Target {
            pointer: &[
                "definitions",
                "global",
                "properties",
                "session_archive",
                "properties",
                "older_than_days",
            ],
            injection: Injection::Default(json!(archive.older_than_days)),
        },
    ])
}

fn wire<T: Serialize>(value: &T) -> Result<Value> {
    Ok(serde_json::to_value(value)?)
}

fn wire_values<T: Serialize>(variants: &[T]) -> Result<Vec<String>> {
    variants
        .iter()
        .map(|variant| {
            serde_json::to_value(variant)?
                .as_str()
                .map(str::to_string)
                .ok_or_else(|| anyhow!("enum variant does not serialize to a JSON string"))
        })
        .collect()
}

/// `solo` is a serde alias of `yolo`; the schema lists it so editors keep accepting it.
fn permission_mode_values() -> Result<Vec<String>> {
    let mut values = wire_values(&[
        PermissionMode::Safe,
        PermissionMode::Default,
        PermissionMode::Auto,
        PermissionMode::Yolo,
    ])?;
    values.push("solo".to_string());
    Ok(values)
}

fn apply_all(original: &str) -> Result<(String, Vec<String>)> {
    let mut text = original.to_string();
    let mut changed = Vec::new();
    for target in targets()? {
        let next = inject(&text, target.pointer, &target.injection)?;
        if next != text {
            changed.push(target.pointer.join("/"));
            text = next;
        }
    }
    Ok((text, changed))
}

fn inject(text: &str, pointer: &[&str], injection: &Injection) -> Result<String> {
    let object = resolve(text, pointer)?;
    match injection {
        Injection::Default(value) => set_member(text, &object, "default", value),
        Injection::Enum(values) => set_member(
            text,
            &object,
            "enum",
            &Value::Array(values.iter().cloned().map(Value::String).collect()),
        ),
    }
}

fn set_member(text: &str, object: &Object, key: &str, value: &Value) -> Result<String> {
    let Some(member) = object.member(key) else {
        return Ok(insert_member(text, object, key, value));
    };
    let current: Value = serde_json::from_str(&text[member.value_start..member.value_end])
        .with_context(|| format!("failed to parse existing `{key}` value"))?;
    if current == *value {
        return Ok(text.to_string());
    }
    let mut out = String::with_capacity(text.len());
    out.push_str(&text[..member.value_start]);
    out.push_str(&serde_json::to_string(value)?);
    out.push_str(&text[member.value_end..]);
    Ok(out)
}

fn insert_member(text: &str, object: &Object, key: &str, value: &Value) -> String {
    let rendered = serde_json::to_string(value).expect("schema values serialize");
    let multiline = text[object.open..=object.close].contains('\n');
    let anchor = object
        .member("description")
        .unwrap_or_else(|| object.members.last().expect("schema object has members"));
    let before_description = object.member("description").is_some();

    let (offset, insertion) = if multiline {
        let line_start = text[..anchor.key_start]
            .rfind('\n')
            .map_or(0, |index| index + 1);
        let indent = &text[line_start..anchor.key_start];
        if before_description {
            (line_start, format!("{indent}\"{key}\": {rendered},\n"))
        } else {
            (
                anchor.value_end,
                format!(",\n{indent}\"{key}\": {rendered}"),
            )
        }
    } else if before_description {
        (anchor.key_start, format!("\"{key}\": {rendered}, "))
    } else {
        (anchor.value_end, format!(", \"{key}\": {rendered}"))
    };

    let mut out = String::with_capacity(text.len() + insertion.len());
    out.push_str(&text[..offset]);
    out.push_str(&insertion);
    out.push_str(&text[offset..]);
    out
}

struct Member {
    key: String,
    key_start: usize,
    value_start: usize,
    value_end: usize,
}

struct Object {
    open: usize,
    close: usize,
    members: Vec<Member>,
}

impl Object {
    fn member(&self, key: &str) -> Option<&Member> {
        self.members.iter().find(|member| member.key == key)
    }
}

fn resolve(text: &str, pointer: &[&str]) -> Result<Object> {
    let bytes = text.as_bytes();
    let root_open = text
        .find('{')
        .ok_or_else(|| anyhow!("schema has no root object"))?;
    let mut object = parse_object(bytes, root_open)?;
    for segment in pointer {
        let member = object
            .member(segment)
            .ok_or_else(|| anyhow!("schema path not found: {segment}"))?;
        if bytes.get(member.value_start) != Some(&b'{') {
            bail!("schema path segment `{segment}` does not name an object");
        }
        object = parse_object(bytes, member.value_start)?;
    }
    Ok(object)
}

fn parse_object(bytes: &[u8], open: usize) -> Result<Object> {
    let mut index = open + 1;
    let mut members = Vec::new();
    loop {
        index = skip_ws(bytes, index);
        match bytes.get(index) {
            Some(b'}') => {
                return Ok(Object {
                    open,
                    close: index,
                    members,
                });
            }
            Some(b'"') => {}
            _ => bail!("malformed schema object at byte {index}"),
        }
        let key_start = index;
        let key_end = scan_string(bytes, index);
        let key = std::str::from_utf8(&bytes[key_start + 1..key_end.saturating_sub(1)])
            .context("schema key is not valid UTF-8")?
            .to_string();
        index = skip_ws(bytes, key_end);
        if bytes.get(index) != Some(&b':') {
            bail!("malformed schema member at byte {index}");
        }
        index = skip_ws(bytes, index + 1);
        let value_start = index;
        let value_end = scan_value(bytes, index)?;
        members.push(Member {
            key,
            key_start,
            value_start,
            value_end,
        });
        index = skip_ws(bytes, value_end);
        match bytes.get(index) {
            Some(b',') => index += 1,
            Some(b'}') => {
                return Ok(Object {
                    open,
                    close: index,
                    members,
                });
            }
            _ => bail!("malformed schema object at byte {index}"),
        }
    }
}

fn scan_string(bytes: &[u8], start: usize) -> usize {
    let mut index = start + 1;
    while index < bytes.len() {
        match bytes[index] {
            b'\\' => index += 2,
            b'"' => return index + 1,
            _ => index += 1,
        }
    }
    bytes.len()
}

fn scan_value(bytes: &[u8], start: usize) -> Result<usize> {
    let index = skip_ws(bytes, start);
    match bytes.get(index) {
        Some(b'"') => Ok(scan_string(bytes, index)),
        Some(open @ (b'{' | b'[')) => {
            let close = if *open == b'{' { b'}' } else { b']' };
            let mut depth = 0usize;
            let mut cursor = index;
            while cursor < bytes.len() {
                match bytes[cursor] {
                    b'"' => {
                        cursor = scan_string(bytes, cursor);
                        continue;
                    }
                    byte if byte == *open => depth += 1,
                    byte if byte == close => {
                        depth -= 1;
                        if depth == 0 {
                            return Ok(cursor + 1);
                        }
                    }
                    _ => {}
                }
                cursor += 1;
            }
            bail!("unterminated schema value at byte {index}")
        }
        Some(_) => {
            let mut cursor = index;
            while cursor < bytes.len()
                && !matches!(
                    bytes[cursor],
                    b',' | b'}' | b']' | b' ' | b'\t' | b'\n' | b'\r'
                )
            {
                cursor += 1;
            }
            Ok(cursor)
        }
        None => bail!("unexpected end of schema"),
    }
}

fn skip_ws(bytes: &[u8], start: usize) -> usize {
    let mut index = start;
    while index < bytes.len() && matches!(bytes[index], b' ' | b'\t' | b'\n' | b'\r') {
        index += 1;
    }
    index
}

/// Rewrites `path` from the Rust authorities; check mode only reports drift.
pub(crate) fn generate_schema(path: &Path, check: bool) -> Result<()> {
    let original = std::fs::read_to_string(path)
        .with_context(|| format!("failed to read {}", path.display()))?;
    let (rendered, changed) = apply_all(&original)?;
    if changed.is_empty() {
        println!("schema is up to date: {}", path.display());
        return Ok(());
    }
    if check {
        bail!(
            "schema is out of date: {} (drifted: {})",
            path.display(),
            changed.join(", ")
        );
    }
    std::fs::write(path, rendered)
        .with_context(|| format!("failed to write {}", path.display()))?;
    println!(
        "updated {} (drifted: {})",
        path.display(),
        changed.join(", ")
    );
    Ok(())
}

#[cfg(test)]
const SCHEMA_JSON: &str = include_str!("../../letcode.schema.json");

#[cfg(test)]
#[path = "../../tests/internal/config/schema_gen/tests.rs"]
mod tests;
