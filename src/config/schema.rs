use serde_json::Value;
use std::sync::OnceLock;

const SCHEMA_JSON: &str = include_str!("../../letcode.schema.json");

fn schema() -> Option<&'static Value> {
    static SCHEMA: OnceLock<Option<Value>> = OnceLock::new();
    SCHEMA
        .get_or_init(|| serde_json::from_str(SCHEMA_JSON).ok())
        .as_ref()
}

/// The bundled schema's description for a configuration path, together with a
/// stable key path that replaces provider, model, and server names with `*`.
///
/// Leaf keys without their own description fall back to the nearest documented
/// ancestor, so `capabilities.tools` still explains what capabilities are.
pub fn field_schema(path: &[&str]) -> Option<(String, String)> {
    let root = schema()?;
    let mut node = root;
    let mut fallback: Option<String> = None;
    let mut key_path: Vec<String> = Vec::new();
    for segment in path {
        if let Some(description) = node.get("description").and_then(Value::as_str) {
            fallback = Some(description.to_string());
        }
        let resolved = deref(root, node);
        let (child, dynamic) = if let Some(child) = resolved
            .get("properties")
            .and_then(|properties| properties.get(segment))
        {
            (child, false)
        } else if let Some(child) = resolved.get("additionalProperties") {
            (child, true)
        } else {
            return None;
        };
        key_path.push(if dynamic {
            "*".to_string()
        } else {
            segment.to_string()
        });
        node = deref(root, child);
    }
    let description = node
        .get("description")
        .and_then(Value::as_str)
        .map(str::to_string)
        .or(fallback)?;
    Some((description, key_path.join("/")))
}

/// The allowed string values for an enumerated configuration path.
pub fn field_enum(path: &[&str]) -> Option<Vec<String>> {
    let root = schema()?;
    let mut node = root;
    for segment in path {
        node = child_node(root, node, segment)?;
    }
    let mut values = node
        .get("enum")
        .and_then(Value::as_array)?
        .iter()
        .filter_map(|value| value.as_str().map(str::to_string))
        .collect::<Vec<_>>();
    // `solo` is a historical alias of `yolo`; the editor offers the canonical value.
    if path.len() == 2 && path[0] == "permissions" && path[1] == "mode" {
        values.retain(|value| value != "solo");
    }
    if values.is_empty() {
        None
    } else {
        Some(values)
    }
}

/// Every documented configuration path, keyed like `field_schema`.
#[cfg(test)]
pub fn schema_keys() -> Vec<String> {
    let Some(root) = schema() else {
        return Vec::new();
    };
    let mut keys = Vec::new();
    collect_keys(root, root, &mut Vec::new(), &mut keys);
    keys.sort();
    keys
}

#[cfg(test)]
fn collect_keys(root: &Value, node: &Value, path: &mut Vec<String>, keys: &mut Vec<String>) {
    let node = deref(root, node);
    if !path.is_empty() && node.get("description").is_some() {
        keys.push(path.join("/"));
    }
    if let Some(properties) = node.get("properties").and_then(Value::as_object) {
        for (name, child) in properties {
            path.push(name.clone());
            collect_keys(root, child, path, keys);
            path.pop();
        }
    } else if let Some(child) = node.get("additionalProperties") {
        path.push("*".to_string());
        collect_keys(root, child, path, keys);
        path.pop();
    }
}

fn child_node<'a>(root: &'a Value, node: &'a Value, key: &str) -> Option<&'a Value> {
    let node = deref(root, node);
    if let Some(child) = node
        .get("properties")
        .and_then(|properties| properties.get(key))
    {
        return Some(deref(root, child));
    }
    // providers.<name>, models.<id>, and mcp.<name> are keyed maps.
    node.get("additionalProperties").map(|child| deref(root, child))
}

fn deref<'a>(root: &'a Value, node: &'a Value) -> &'a Value {
    let Some(reference) = node.get("$ref").and_then(Value::as_str) else {
        return node;
    };
    let Some(pointer) = reference.strip_prefix("#/") else {
        return node;
    };
    let mut current = root;
    for segment in pointer.split('/') {
        let Some(next) = current.get(segment) else {
            return node;
        };
        current = next;
    }
    current
}
