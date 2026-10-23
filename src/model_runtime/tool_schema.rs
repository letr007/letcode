use serde_json::{Map, Value};

#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum ToolSchemaDialect {
    Anthropic,
    OpenAiStrict,
}

/// Projects a tool schema onto what a protocol can express natively. Features the
/// protocol cannot encode degrade into the property description, which the model
/// still reads; the tool implementation owns enforcement.
pub(crate) fn project(schema: &Value, dialect: ToolSchemaDialect) -> Value {
    let Value::Object(mut projected) = project_node(schema, dialect, true) else {
        return empty_object_schema();
    };
    projected.insert("type".into(), Value::String("object".into()));
    projected
        .entry("properties")
        .or_insert_with(|| Value::Object(Map::new()));
    projected.insert("additionalProperties".into(), Value::Bool(false));
    Value::Object(projected)
}

fn project_node(node: &Value, dialect: ToolSchemaDialect, root: bool) -> Value {
    let Value::Object(map) = node else {
        return node.clone();
    };
    let mut projected = Map::new();
    let mut notes = Vec::new();
    for (key, value) in map {
        match key.as_str() {
            "oneOf" => notes.push(describe_composition(key, value)),
            "anyOf" | "allOf" if root => notes.push(describe_composition(key, value)),
            "minLength" | "maxLength" | "minimum" | "maximum" | "exclusiveMinimum"
            | "exclusiveMaximum" | "multipleOf" | "maxItems"
                if dialect == ToolSchemaDialect::Anthropic =>
            {
                notes.push(format!("{key}: {value}"));
            }
            "minItems"
                if dialect == ToolSchemaDialect::Anthropic
                    && !matches!(value.as_u64(), Some(0 | 1)) =>
            {
                notes.push(format!("{key}: {value}"));
            }
            "properties" => {
                let value = match value {
                    Value::Object(properties) => Value::Object(
                        properties
                            .iter()
                            .map(|(name, child)| {
                                (name.clone(), project_node(child, dialect, false))
                            })
                            .collect(),
                    ),
                    other => other.clone(),
                };
                projected.insert(key.clone(), value);
            }
            "items" => {
                projected.insert(key.clone(), project_node(value, dialect, false));
            }
            _ => {
                projected.insert(key.clone(), value.clone());
            }
        }
    }
    if !notes.is_empty() {
        let existing = projected
            .get("description")
            .and_then(Value::as_str)
            .unwrap_or_default();
        let constraints = notes.join("; ");
        let description = if existing.is_empty() {
            constraints
        } else {
            format!("{existing} ({constraints})")
        };
        projected.insert("description".into(), Value::String(description));
    }
    Value::Object(projected)
}

fn describe_composition(key: &str, value: &Value) -> String {
    let Value::Array(branches) = value else {
        return format!("{key}: {value}");
    };
    let branches = branches
        .iter()
        .map(|branch| {
            let Value::Object(branch) = branch else {
                return branch.to_string();
            };
            let mut parts = Vec::new();
            if let Some(required) = branch.get("required").and_then(Value::as_array) {
                let names = required
                    .iter()
                    .filter_map(Value::as_str)
                    .collect::<Vec<_>>()
                    .join(", ");
                if !names.is_empty() {
                    parts.push(format!("required {names}"));
                }
            }
            if let Some(kind) = branch.get("type").and_then(Value::as_str) {
                parts.push(kind.to_string());
            }
            if parts.is_empty() {
                Value::Object(branch.clone()).to_string()
            } else {
                parts.join(" ")
            }
        })
        .collect::<Vec<_>>()
        .join(" | ");
    let note = format!("{key}: {branches}");
    note.chars().take(240).collect()
}

fn empty_object_schema() -> Value {
    let mut map = Map::new();
    map.insert("type".into(), Value::String("object".into()));
    map.insert("properties".into(), Value::Object(Map::new()));
    map.insert("additionalProperties".into(), Value::Bool(false));
    Value::Object(map)
}
