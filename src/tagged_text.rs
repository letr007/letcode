//! Exact tag delimiters with literal, unescaped field text.

pub(crate) fn recover_blocks<'a>(text: &'a str, tag: &str) -> Vec<&'a str> {
    let open = format!("<{tag}>");
    let close = format!("</{tag}>");
    let mut blocks = Vec::new();
    let mut cursor = 0;
    while let Some(offset) = text[cursor..].find(&open) {
        let start = cursor + offset + open.len();
        let rest = &text[start..];
        let body = bounded(rest, &close, &[open.as_str()]);
        blocks.push(body);
        if body.len() == rest.len() {
            break;
        }
        cursor = start + body.len();
    }
    blocks
}

pub(crate) fn bounded<'a>(rest: &'a str, close: &str, boundaries: &[&str]) -> &'a str {
    let mut end = rest.len();
    if let Some(position) = rest.find(close) {
        end = end.min(position);
    }
    for boundary in boundaries {
        if let Some(position) = rest.find(boundary) {
            end = end.min(position);
        }
    }
    &rest[..end]
}
