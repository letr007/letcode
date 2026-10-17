use super::*;

#[test]
fn envelope_fields_are_top_level_keys_not_payload_text() {
    for line in [
        r#"{"payload":{"schema_version":2,"journal_schema_version":2,"journal_entry":"transaction_commit"}}"#,
        r#"{"payload":[{"schema_version":2}],"content":"journal_entry\" : \"schema_version"}"#,
        r#"{"content":"schema_version","other":"journal_schema_version"}"#,
    ] {
        assert_eq!(journal_fields(line), JournalFields::default());
    }
    let line = format!(
        r#"{{"schema_version":1,"payload":"{}","journal_schema_version" : 2,"journal_entry":"transaction_commit"}}"#,
        "large payload ".repeat(4096)
    );
    assert_eq!(
        journal_fields(&line),
        JournalFields {
            schema_version: true,
            journal_schema_version: true,
            journal_entry: true,
        }
    );
}

#[test]
fn envelope_versions_preserve_legacy_and_payload_schema_selection() {
    let record = r#""session_id":"s","sequence":1,"timestamp_ms":18446744073709551616,"kind":"future_audit_event""#;
    let metadata = r#""event_id":"s:1","scope":"global","base_revision":0,"resulting_revision":1"#;
    let legacy = format!("{{{record}}}");
    let ParsedJournalLine::Record(entry) = parse_journal_line(&legacy).unwrap() else {
        panic!("expected a legacy record");
    };
    assert!(entry.envelope.is_none());
    for (fields, expected_version) in [
        (r#""schema_version":1"#, 1),
        (r#""schema_version":2"#, 2),
        (r#""schema_version":99,"journal_schema_version":2"#, 2),
        (r#""journal_schema_version":2,"schema_version":99"#, 2),
    ] {
        let line = format!("{{{record},{metadata},{fields}}}");
        let ParsedJournalLine::Record(entry) = parse_journal_line(&line).unwrap() else {
            panic!("expected an enveloped record");
        };
        let envelope = entry.envelope.unwrap();
        assert_eq!(envelope.schema_version, expected_version);
        assert_eq!(entry.record.timestamp_ms, 18446744073709551616);
        assert_eq!(envelope.record.timestamp_ms, entry.record.timestamp_ms);
    }
}

#[test]
fn field_detection_does_not_accept_invalid_json_or_envelope_metadata() {
    for line in [
        r#"{"schema_version":2,"content":"unterminated}"#,
        r#"{"schema_version":2}"#,
        r#"{"journal_entry":"transaction_commit","schema_version":2}"#,
        r#"{"session_id":"s","sequence":1,"timestamp_ms":0,"kind":"user_message"}"#,
        r#"{"session_id":"s","sequence":1,"timestamp_ms":0,"kind":"future_audit_event","schema_version":99,"event_id":"s:1","scope":"global","base_revision":0,"resulting_revision":1}"#,
    ] {
        assert!(parse_journal_line(line).is_err(), "accepted {line}");
    }
}
