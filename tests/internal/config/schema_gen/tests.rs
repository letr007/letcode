use super::*;

#[test]
fn committed_schema_matches_rust_authorities() {
    let rendered = apply_all(SCHEMA_JSON).expect("render schema").0;
    assert_eq!(rendered, SCHEMA_JSON);
}

#[test]
fn subagents_editor_schema_exposes_modes_and_defaults() {
    assert_eq!(
        super::super::field_enum(&["subagents", "mode"]),
        Some(vec!["expert".to_string(), "swarm".to_string()])
    );
    assert_eq!(
        super::super::default_value(&["subagents", "mode"])
            .unwrap()
            .as_str(),
        Some("expert")
    );
    let schema: Value = serde_json::from_str(SCHEMA_JSON).unwrap();
    assert_eq!(
        schema["properties"]["subagents"]["$ref"],
        "#/definitions/subagents"
    );
    assert_eq!(
        schema["definitions"]["subagents"]["additionalProperties"],
        false
    );
    assert_eq!(
        schema["definitions"]["subagents"]["not"],
        json!({ "required": ["max_concurrent", "max_descendants"] })
    );
    for (field, expected) in [
        ("max_depth", 3),
        ("max_concurrent", 16),
        ("max_pending_messages", 16),
        ("max_message_bytes", 16384),
        ("run_timeout_secs", 600),
    ] {
        assert_eq!(
            super::super::default_value(&["subagents", field])
                .unwrap()
                .as_integer(),
            Some(expected),
            "{field}"
        );
        assert_eq!(
            schema["definitions"]["subagents"]["properties"][field]["minimum"],
            1
        );
        assert_eq!(
            schema["definitions"]["subagents"]["properties"][field]["type"],
            "integer"
        );
    }
}

#[test]
fn inserts_default_before_description_and_appends_inline() {
    let multiline = "{\n  \"a\": {\n    \"type\": \"integer\",\n    \"description\": \"x\"\n  }\n}";
    assert_eq!(
        inject(multiline, &["a"], &Injection::Default(json!(60))).expect("inject"),
        "{\n  \"a\": {\n    \"type\": \"integer\",\n    \"default\": 60,\n    \"description\": \"x\"\n  }\n}"
    );

    let inline = "{ \"a\": { \"type\": \"integer\", \"minimum\": 1 } }";
    assert_eq!(
        inject(inline, &["a"], &Injection::Default(json!(262144))).expect("inject"),
        "{ \"a\": { \"type\": \"integer\", \"minimum\": 1, \"default\": 262144 } }"
    );
}

#[test]
fn leaves_matching_members_untouched() {
    let text = "{\n  \"a\": {\n    \"enum\": [\"x\", \"y\"]\n  }\n}";
    assert_eq!(
        inject(
            text,
            &["a"],
            &Injection::Enum(vec!["x".to_string(), "y".to_string()])
        )
        .expect("inject"),
        text
    );
}
