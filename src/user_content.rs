use serde::{Deserialize, Deserializer, Serialize};

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
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum UserMessagePart {
    Text { text: String },
    Image { attachment: UserImageAttachment },
}

#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize)]
pub struct UserMessageContent {
    pub text: String,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub attachments: Vec<UserImageAttachment>,
}

impl<'de> Deserialize<'de> for UserMessageContent {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        #[derive(Deserialize)]
        struct StructuredUserMessageContent {
            text: String,
            #[serde(default)]
            attachments: Vec<UserImageAttachment>,
        }

        #[derive(Deserialize)]
        #[serde(untagged)]
        enum Repr {
            Text(String),
            Structured(StructuredUserMessageContent),
        }

        match Repr::deserialize(deserializer)? {
            Repr::Text(text) => Ok(UserMessageContent::new(text, Vec::new())),
            Repr::Structured(content) => {
                Ok(UserMessageContent::new(content.text, content.attachments))
            }
        }
    }
}

impl UserMessageContent {
    pub fn new(text: impl Into<String>, attachments: Vec<UserImageAttachment>) -> Self {
        Self {
            text: text.into(),
            attachments,
        }
    }

    pub fn is_empty(&self) -> bool {
        self.text.trim().is_empty() && self.attachments.is_empty()
    }

    pub fn parts(&self) -> Vec<UserMessagePart> {
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
        let mut lines = Vec::new();
        if !self.text.is_empty() {
            lines.push(self.text.clone());
        }
        lines.extend(
            self.attachments
                .iter()
                .map(UserImageAttachment::placeholder_summary),
        );
        lines.join("\n")
    }
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
