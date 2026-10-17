use super::*;

#[test]
fn history_input_keeps_readable_sources_and_native_image_occurrences() {
    let image = UserImageAttachment::from_bytes("screenshot", "image/png", b"image-data");
    let opaque = "OPAQUE-TRANSPORT-ONLY".repeat(10_000);
    let replay = crate::model_runtime::OpaqueReplayState::from_anthropic_thinking_blocks_json(
        &json!([{"type":"redacted_thinking","data":opaque}]).to_string(),
    )
    .unwrap();
    let output = r#"{"content":"literal encrypted_content and data_url keys are source text"}"#;
    let history = vec![
        ProtocolItem::user_content(
            UserMessageContent::from_parts(vec![
                UserMessagePart::Text {
                    text: "before image".into(),
                },
                UserMessagePart::Image {
                    attachment: image.clone(),
                },
                UserMessagePart::Text {
                    text: "after image".into(),
                },
            ])
            .with_selected_skills(vec!["ui-review".into()]),
        ),
        ProtocolItem::AssistantTurn {
            text: Some("visible answer".into()),
            reasoning_content: Some("readable reasoning".into()),
            replay: Some(replay),
            calls: vec![crate::protocol_frames::ProtocolToolCall {
                call_id: "read-1".into(),
                name: "fs__read".into(),
                arguments_json: r#"{"path":"view.png"}"#.into(),
            }],
        },
        ProtocolItem::ToolOutput {
            call_id: "read-1".into(),
            output_json: output.into(),
            images: vec![image.clone()],
        },
        ProtocolItem::internal_continuation("continue checking"),
        ProtocolItem::context_summary("existing summary"),
    ];
    let original = serde_json::to_vec(&history).unwrap();
    let references = vec![json!({"title":"reference"})];
    let facts = vec![json!({"id":"f1","text":"project fact"})];
    let input = history_input(&history, &references, &facts);
    assert_eq!(serde_json::to_vec(&history).unwrap(), original);
    assert_eq!(input.attachments, vec![image.clone(), image.clone()]);
    assert!(!input.text.contains("OPAQUE-TRANSPORT-ONLY"));
    assert!(!input.text.contains(&image.data_url));
    let value: Value = serde_json::from_str(&input.text).unwrap();
    assert_eq!(value["source_count"], 5);
    assert_eq!(value["references"], json!(references));
    assert_eq!(value["project_facts"], json!(facts));
    let messages = value["new_messages"].as_array().unwrap();
    for (index, message) in messages.iter().enumerate() {
        assert_eq!(message["index"], index);
    }
    let user = &messages[0]["content"];
    assert_eq!(user["selected_skills"], json!(["ui-review"]));
    assert_eq!(user["parts"][0]["text"], "before image");
    assert_eq!(user["parts"][1]["attachment"]["attachment_index"], 0);
    assert_eq!(user["parts"][2]["text"], "after image");
    let assistant = &messages[1]["content"];
    assert_eq!(assistant["text"], "visible answer");
    assert_eq!(assistant["reasoning_content"], "readable reasoning");
    assert_eq!(assistant["calls"][0]["call_id"], "read-1");
    assert_eq!(
        assistant["calls"][0]["arguments_json"],
        r#"{"path":"view.png"}"#
    );
    assert!(assistant.get("replay").is_none());
    assert_eq!(messages[2]["content"]["output_json"], output);
    assert_eq!(messages[2]["content"]["images"][0]["attachment_index"], 1);
    assert_eq!(messages[3]["content"]["text"], "continue checking");
    assert_eq!(messages[4]["content"]["text"], "existing summary");
}

#[test]
fn three_tiers_are_bound_to_host_sources() {
    let publication =
        parse_publication("p", &["raw:1".into(), "raw:2".into()], &tagged_episode(1)).unwrap();
    assert_eq!(publication.source_ids, vec!["raw:1"]);
    assert_eq!(publication.compartments[0].id, "p:c0");
    // A first cut beyond the supplied sources leaves nothing usable.
    assert!(parse_publication("p", &["raw:1".into()], &tagged_episode(2)).is_err());
}

fn tagged_response(cuts: &[usize]) -> String {
    cuts.iter()
        .map(|end| tagged_episode(*end))
        .collect::<Vec<_>>()
        .join("\n")
}

#[test]
fn episode_cuts_define_the_covered_prefix() {
    let sources: Vec<_> = (0..4).map(|i| format!("raw:{i}")).collect();
    for cut in [3usize, 4] {
        let publication = parse_publication("p", &sources, &tagged_response(&[2, cut])).unwrap();
        assert_eq!(publication.source_ids, sources[..cut]);
        assert_eq!(publication.compartments[0].source_ids, sources[..2]);
        assert_eq!(publication.compartments[1].source_ids, sources[2..cut]);
    }
    assert!(
        parse_publication("p", &sources, &tagged_response(&[]))
            .unwrap_err()
            .to_string()
            .contains("historian produced no completed work")
    );
}

/// The first block that is unusable, or whose cut does not advance, ends the
/// accepted prefix instead of failing the whole response. Only a response with
/// no usable episode is a failure.
#[test]
fn a_broken_or_regressing_block_ends_the_accepted_prefix() {
    let sources: Vec<_> = (0..4).map(|i| format!("raw:{i}")).collect();
    let broken = format!("{}\n<episode><end>4</end><title>", tagged_episode(2));
    let publication = parse_publication("p", &sources, &broken).unwrap();
    assert_eq!(publication.source_ids, sources[..2]);
    assert_eq!(publication.compartments.len(), 1);

    for cuts in [vec![2, 1, 3], vec![2, 9], vec![2, 2]] {
        let publication = parse_publication("p", &sources, &tagged_response(&cuts)).unwrap();
        assert_eq!(publication.source_ids, sources[..2], "{cuts:?}");
    }

    assert!(parse_publication("p", &sources, "<episode><end>4</end>").is_err());
}

/// A block that parses but breaks the episode contract ends the prefix like a
/// broken block: the episodes before it stay usable instead of voiding the
/// whole response.
#[test]
fn an_unusable_episode_ends_the_accepted_prefix() {
    let sources: Vec<_> = (0..4).map(|i| format!("raw:{i}")).collect();
    for unusable in [
        "<episode><end>3</end><title></title><importance>60</importance><detailed>d</detailed><compact>c</compact><anchor>a</anchor></episode>",
        "<episode><end>3</end><title>T</title><importance>0</importance><detailed>d</detailed><compact>c</compact><anchor>a</anchor></episode>",
        "<episode><end>3</end><title>T</title><importance>60</importance><detailed> </detailed><compact>c</compact><anchor>a</anchor></episode>",
        "<episode><end>3</end><title>T</title><importance>900</importance><detailed>d</detailed><compact>c</compact><anchor>a</anchor></episode>",
    ] {
        let text = format!("{}\n{unusable}", tagged_episode(2));
        let publication = parse_publication("p", &sources, &text).unwrap();
        assert_eq!(publication.source_ids, sources[..2], "{unusable}");
        assert_eq!(publication.compartments.len(), 1, "{unusable}");
        assert!(
            parse_publication("p", &sources, unusable).is_err(),
            "{unusable}"
        );
    }
}

#[test]
fn failure_classes_separate_size_rejections_from_broken_responses() {
    let overflow = "model Http failure during Decode, code context_too_large, retry hint Never";
    let overloaded =
        "model Http failure during Decode, code server_is_overloaded, retry hint Never";
    let contract = "historian produced no completed work: line 3 is not an episode";
    assert!(is_context_overflow(overflow));
    assert_eq!(classify_failure(overflow), FailureClass::Overflow);
    assert_eq!(classify_failure(overloaded), FailureClass::Transport);
    assert_eq!(classify_failure(contract), FailureClass::Contract);
    assert!(FailureClass::Overflow.shrinks());
    assert!(FailureClass::Contract.shrinks());
    assert!(!FailureClass::Transport.shrinks());
}

#[test]
fn raw_tail_keeps_the_end_of_a_rejected_response() {
    let long = format!("{}MARK", "汉".repeat(RAW_TAIL_CHARS + 10));
    let tail = raw_tail(&long);
    assert!(tail.starts_with('…'));
    assert!(tail.ends_with("MARK"));
    assert_eq!(tail.trim_start_matches('…').chars().count(), RAW_TAIL_CHARS);
    assert_eq!(
        tail.chars().filter(|c| *c == '汉').count(),
        RAW_TAIL_CHARS - 4
    );
    assert_eq!(raw_tail("short"), "short");
}

#[test]
fn raw_excerpt_is_bounded_without_truncating_characters() {
    let long: String = "汉".repeat(RAW_EXCERPT_CHARS + 10);
    let excerpt = raw_excerpt(&long);
    assert!(excerpt.ends_with('…'));
    assert_eq!(
        excerpt.chars().filter(|c| *c == '汉').count(),
        RAW_EXCERPT_CHARS
    );
    assert_eq!(raw_excerpt("short"), "short");
}

#[test]
fn a_repair_request_keeps_the_chunk_and_quotes_the_rejected_tail() {
    let chunk = history_input(
        &[ProtocolItem::user_content(UserMessageContent::new(
            "chunk",
            Vec::new(),
        ))],
        &[],
        &[],
    );
    let repair = repair_input(&chunk, "first\nBROKEN", "line 2 is not an episode");
    assert_eq!(repair.attachments, chunk.attachments);
    assert!(repair.text.starts_with(&chunk.text));
    assert!(repair.text.contains("line 2 is not an episode"));
    assert!(repair.text.contains("BROKEN"));
}

fn tagged_episode(end: usize) -> String {
    format!(
        "<episode><end>{end}</end><title>Parser</title><importance>60</importance><detailed>Detailed</detailed><compact>Compact</compact><anchor>Parser</anchor></episode>"
    )
}

#[test]
fn tag_values_end_at_the_nearest_close_tag_next_tag_or_end() {
    assert_eq!(recover_value("<end>2</end>", "end").as_deref(), Some("2"));
    assert_eq!(
        recover_value("<end>2<title>x</title>", "end").as_deref(),
        Some("2")
    );
    assert_eq!(recover_value("<end>2", "end").as_deref(), Some("2"));
    assert_eq!(
        recover_value("<title>a<b</title>", "title").as_deref(),
        Some("a<b")
    );
    assert_eq!(
        recover_value("<anchor></anchor>", "anchor").as_deref(),
        Some("")
    );
    assert_eq!(recover_value("<end>2</end>", "title"), None);
}

#[test]
fn a_missing_close_tag_ends_at_the_next_wrapper() {
    let text = format!(
        "{}{}",
        tagged_episode(2).replace("</episode>", ""),
        tagged_episode(4)
    );
    let blocks = recover_blocks(&text, EPISODE_TAG);
    assert_eq!(blocks.len(), 2);
    assert!(blocks[0].contains("<end>2</end>"));
    assert!(blocks[1].contains("<end>4</end>"));
}

#[test]
fn the_recovery_record_names_the_tags_that_appeared() {
    let block = "<episode><end>2</end><title>T</title><detailed>D</detailed><compact>C</compact><anchor></anchor></episode>";
    let record = recover_episode(block);
    assert_eq!(
        record.appeared(),
        vec!["end", "title", "detailed", "compact", "anchor"]
    );
    assert_eq!(record.value("importance"), None);
    assert_eq!(record.value("anchor"), Some(""));
    assert_eq!(record.value("title"), Some("T"));
}

#[test]
fn a_block_missing_a_tag_is_unusable_and_names_it() {
    let sources: Vec<_> = (0..2).map(|i| format!("raw:{i}")).collect();
    let block = "<episode><end>2</end><title>T</title><importance>60</importance><detailed>D</detailed><compact>C</compact></episode>";
    let error = parse_publication("p", &sources, block)
        .unwrap_err()
        .to_string();
    assert!(error.contains("missing tags: anchor"), "{error}");
}

#[test]
fn the_tag_parser_survives_arbitrary_input() {
    let sources: Vec<_> = (0..2).map(|i| format!("raw:{i}")).collect();
    for text in [
        "",
        "<episode>",
        "</episode>",
        "<episode></episode>",
        "<episode><end>",
        "<episode><end>1",
        "<episode><end>1</end>",
        "<episode><title>a<b</title></episode>",
        "<episode><end>2</end><title>汉</title></episode>",
        "<episode><episode><episode>",
        "<episode><end>2</end></episode><episode><end>4</end></episode>",
        "<episode><end>99999999999999999999</end></episode>",
    ] {
        let _ = parse_publication("p", &sources, text);
        let _ = recover_episode(text);
        let _ = recover_blocks(text, EPISODE_TAG);
    }
}
