use base64::{Engine as _, engine::general_purpose::STANDARD};
use serde::{Deserialize, Deserializer, Serialize};

const IMAGE_PATCH_SIZE_PIXELS: u64 = 32;
/// A bounded charge for attachments whose dimensions cannot be inspected. This
/// avoids treating an opaque provider auto-detail image as free without
/// coupling the request builder to a model-specific image-pricing catalog.
const UNKNOWN_IMAGE_VISUAL_TOKENS: u64 = 4_096;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct UserImageAttachment {
    pub id: String,
    pub label: String,
    pub mime: String,
    pub data_url: String,
}

impl UserImageAttachment {
    pub fn placeholder_summary(&self) -> String {
        format!("[Image: {}]", self.label)
    }

    pub fn prompt_plan_placeholder(&self) -> String {
        format!(
            "[ImageAttachment id={} label={} mime={} content_hash={}]",
            self.id,
            self.label,
            self.mime,
            stable_hash64(&self.data_url)
        )
    }

    /// Estimates provider visual input from PNG dimensions, not data-URL
    /// transport size. Clipboard images are PNG data URLs, so decoding only
    /// the fixed PNG signature and IHDR prefix avoids expanding their payload.
    pub fn visual_token_charge(&self) -> u64 {
        png_dimensions(&self.data_url)
            .map(|(width, height)| {
                width
                    .div_ceil(IMAGE_PATCH_SIZE_PIXELS)
                    .saturating_mul(height.div_ceil(IMAGE_PATCH_SIZE_PIXELS))
            })
            .filter(|tokens| *tokens > 0)
            .unwrap_or(UNKNOWN_IMAGE_VISUAL_TOKENS)
    }
}

fn png_dimensions(data_url: &str) -> Option<(u64, u64)> {
    const PNG_SIGNATURE: [u8; 8] = [137, 80, 78, 71, 13, 10, 26, 10];
    // PNG signature, IHDR chunk length/type, and the width and height.
    const PNG_HEADER_BYTES: usize = 24;
    const PNG_HEADER_BASE64_BYTES: usize = 32;

    let encoded = data_url.strip_prefix("data:image/png;base64,")?;
    let header = STANDARD
        .decode(encoded.get(..PNG_HEADER_BASE64_BYTES)?)
        .ok()?;
    if header.len() != PNG_HEADER_BYTES
        || header[..8] != PNG_SIGNATURE
        || header[12..16] != *b"IHDR"
    {
        return None;
    }

    let width = u32::from_be_bytes(header[16..20].try_into().ok()?);
    let height = u32::from_be_bytes(header[20..24].try_into().ok()?);
    (width > 0 && height > 0).then_some((u64::from(width), u64::from(height)))
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum UserMessagePart {
    Text { text: String },
    Image { attachment: UserImageAttachment },
}

#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize)]
pub struct UserMessageContent {
    #[serde(skip_serializing)]
    pub text: String,
    #[serde(skip_serializing)]
    pub attachments: Vec<UserImageAttachment>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub parts: Vec<UserMessagePart>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub selected_skills: Vec<String>,
}

impl<'de> Deserialize<'de> for UserMessageContent {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        #[derive(Deserialize)]
        struct StructuredUserMessageContent {
            #[serde(default)]
            text: String,
            #[serde(default)]
            attachments: Vec<UserImageAttachment>,
            #[serde(default)]
            parts: Vec<UserMessagePart>,
            #[serde(default)]
            selected_skills: Vec<String>,
        }

        #[derive(Deserialize)]
        #[serde(untagged)]
        enum Repr {
            Text(String),
            Structured(StructuredUserMessageContent),
        }

        match Repr::deserialize(deserializer)? {
            Repr::Text(text) => Ok(UserMessageContent::new(text, Vec::new())),
            Repr::Structured(content) if !content.parts.is_empty() => {
                Ok(UserMessageContent::from_parts(content.parts)
                    .with_selected_skills(content.selected_skills))
            }
            Repr::Structured(content) => {
                Ok(UserMessageContent::new(content.text, content.attachments)
                    .with_selected_skills(content.selected_skills))
            }
        }
    }
}

impl UserMessageContent {
    pub fn new(text: impl Into<String>, attachments: Vec<UserImageAttachment>) -> Self {
        let text = text.into();
        let mut parts = Vec::with_capacity(1 + attachments.len());
        if !text.is_empty() {
            parts.push(UserMessagePart::Text { text: text.clone() });
        }
        parts.extend(
            attachments
                .iter()
                .cloned()
                .map(|attachment| UserMessagePart::Image { attachment }),
        );
        Self {
            text,
            attachments,
            parts,
            selected_skills: Vec::new(),
        }
    }

    pub fn from_parts(parts: Vec<UserMessagePart>) -> Self {
        let text = parts
            .iter()
            .filter_map(|part| match part {
                UserMessagePart::Text { text } => Some(text.as_str()),
                UserMessagePart::Image { .. } => None,
            })
            .collect::<String>();
        let attachments = parts
            .iter()
            .filter_map(|part| match part {
                UserMessagePart::Text { .. } => None,
                UserMessagePart::Image { attachment } => Some(attachment.clone()),
            })
            .collect();
        Self {
            text,
            attachments,
            parts,
            selected_skills: Vec::new(),
        }
    }

    pub fn with_selected_skills(mut self, selected_skills: Vec<String>) -> Self {
        self.selected_skills = selected_skills;
        self
    }

    pub fn is_empty(&self) -> bool {
        self.selected_skills.is_empty()
            && self.parts().iter().all(
                |part| matches!(part, UserMessagePart::Text { text } if text.trim().is_empty()),
            )
    }

    pub fn trim_outer_text(&mut self) {
        let mut parts = self.parts();
        if let Some(UserMessagePart::Text { text }) = parts.first_mut() {
            *text = text.trim_start().to_string();
        }
        if let Some(UserMessagePart::Text { text }) = parts.last_mut() {
            *text = text.trim_end().to_string();
        }
        parts.retain(|part| !matches!(part, UserMessagePart::Text { text } if text.is_empty()));
        let selected_skills = std::mem::take(&mut self.selected_skills);
        *self = Self::from_parts(parts).with_selected_skills(selected_skills);
    }

    pub fn parts(&self) -> Vec<UserMessagePart> {
        if !self.parts.is_empty() {
            return self.parts.clone();
        }

        let mut parts = Vec::with_capacity(1 + self.attachments.len());
        if !self.text.is_empty() {
            parts.push(UserMessagePart::Text {
                text: self.text.clone(),
            });
        }
        parts.extend(
            self.attachments
                .iter()
                .cloned()
                .map(|attachment| UserMessagePart::Image { attachment }),
        );
        parts
    }

    pub fn display_text(&self) -> String {
        self.selected_skills
            .iter()
            .map(|name| format!("[Skill: {name}]"))
            .chain(self.parts().into_iter().map(|part| match part {
                UserMessagePart::Text { text } => text,
                UserMessagePart::Image { attachment } => attachment.placeholder_summary(),
            }))
            .collect::<Vec<_>>()
            .join("\n")
    }

    pub fn prompt_plan_text(&self) -> String {
        self.parts()
            .into_iter()
            .map(|part| match part {
                UserMessagePart::Text { text } => text,
                UserMessagePart::Image { attachment } => attachment.prompt_plan_placeholder(),
            })
            .collect::<Vec<_>>()
            .join("\n")
    }
}

fn stable_hash64(input: &str) -> String {
    const OFFSET_BASIS: u64 = 0xcbf29ce484222325;
    const PRIME: u64 = 0x00000100000001b3;

    let mut hash = OFFSET_BASIS;
    for byte in input.as_bytes() {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(PRIME);
    }

    format!("{:016x}", hash)
}

impl From<&str> for UserMessageContent {
    fn from(value: &str) -> Self {
        Self::new(value, Vec::new())
    }
}

impl From<String> for UserMessageContent {
    fn from(value: String) -> Self {
        Self::new(value, Vec::new())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserMessageSubmission {
    pub id: String,
    pub content: UserMessageContent,
}

impl UserMessageSubmission {
    pub fn new(id: impl Into<String>, content: UserMessageContent) -> Self {
        Self {
            id: id.into(),
            content,
        }
    }

    pub fn text(&self) -> &str {
        &self.content.text
    }
}

impl PartialEq for UserMessageSubmission {
    fn eq(&self, other: &Self) -> bool {
        self.content == other.content
    }
}

impl Eq for UserMessageSubmission {}

impl PartialEq<String> for UserMessageSubmission {
    fn eq(&self, other: &String) -> bool {
        self.text() == other
    }
}

impl PartialEq<&str> for UserMessageSubmission {
    fn eq(&self, other: &&str) -> bool {
        self.text() == *other
    }
}

impl From<&str> for UserMessageSubmission {
    fn from(value: &str) -> Self {
        Self::new(
            format!("test-submission-{value}"),
            UserMessageContent::from(value),
        )
    }
}

impl From<String> for UserMessageSubmission {
    fn from(value: String) -> Self {
        let id = format!("test-submission-{value}");
        Self::new(id, UserMessageContent::from(value))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn image(id: &str) -> UserImageAttachment {
        UserImageAttachment {
            id: id.into(),
            label: format!("{id}.png"),
            mime: "image/png".into(),
            data_url: "data:image/png;base64,AAAA".into(),
        }
    }

    #[test]
    fn serialization_persists_only_canonical_ordered_parts() {
        let content = UserMessageContent::new("before", vec![image("one")])
            .with_selected_skills(vec!["rust-audit".into()]);
        let json = serde_json::to_value(&content).expect("content serializes");

        assert!(json.get("text").is_none());
        assert!(json.get("attachments").is_none());
        assert_eq!(json["parts"][1]["attachment"]["id"], "one");
        assert_eq!(json["selected_skills"][0], "rust-audit");
        assert_eq!(
            serde_json::from_value::<UserMessageContent>(json)
                .expect("canonical content deserializes"),
            content
        );
    }

    #[test]
    fn structured_parts_win_over_conflicting_legacy_fields() {
        let content = serde_json::from_value::<UserMessageContent>(serde_json::json!({
            "text": "legacy text",
            "attachments": [image("legacy")],
            "parts": [
                {"kind": "text", "text": "canonical"},
                {"kind": "image", "attachment": image("canonical")}
            ]
        }))
        .expect("structured content deserializes");

        assert_eq!(content.text, "canonical");
        assert_eq!(content.attachments[0].id, "canonical");
    }
}
