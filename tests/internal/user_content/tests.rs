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
fn image_attachment_from_bytes_builds_stable_data_url_descriptor() {
    let first = UserImageAttachment::from_bytes("pixel.png", "image/png", b"bytes");
    let second = UserImageAttachment::from_bytes("pixel.png", "image/png", b"bytes");

    assert_eq!(first, second);
    assert_eq!(first.label, "pixel.png");
    assert_eq!(first.mime, "image/png");
    assert!(first.id.starts_with("image-"));
    assert_eq!(first.data_url, "data:image/png;base64,Ynl0ZXM=");
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
        serde_json::from_value::<UserMessageContent>(json).expect("canonical content deserializes"),
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
