use super::*;

const SAMPLE_REQUEST: &str = "Approve or deny this tool permission request.\n\
         \n\
         User goal:\nExecute commands to test auto-approval\n\
         \n\
         Tool: shell__exec\n\
         Class: command\n\
         Summary: shell__exec python3 -c \"...\"\n\
         Preview: (none)\n\
         can_allow_always: false\n\
         Arguments:\n{\n  \"command\": \"python3\"\n}\n\
         \n\
         Reply with ONLY JSON:\n\
         {\"decision\":\"execute|ask_user|refuse\",\"risk\":\"low|medium|high\",\"rationale\":\"...\"}\n";

#[test]
fn parses_review_request_fields() {
    let card = parse_review_request(SAMPLE_REQUEST).expect("request");
    assert_eq!(card.tool, "shell__exec");
    assert_eq!(card.class.as_deref(), Some("command"));
    assert!(!card.can_allow_always);
    assert_eq!(
        card.goal.as_deref(),
        Some("Execute commands to test auto-approval")
    );
}

#[test]
fn ask_user_decision_is_kept_as_its_own_card() {
    let card = parse_review_decision(
        r#"{"decision":"ask_user","risk":"medium","rationale":"needs the user"}"#,
    )
    .expect("decision");
    assert_eq!(card.decision, "ask_user");
    assert_eq!(card.risk.as_deref(), Some("medium"));
}

#[test]
fn invalid_allow_always_decision_is_rendered_as_allow_once() {
    let card =
        parse_review_decision(r#"{"decision":"allow_always","risk":"low","rationale":"safe"}"#)
            .expect("decision");
    assert_eq!(card.decision, "allow_once");
}

#[test]
fn parses_decision_json_from_assistant_text() {
    let card = parse_review_decision(
        "here\n{\"decision\":\"allow_once\",\"risk\":\"low\",\"rationale\":\"safe https fetch\"}\n",
    )
    .expect("decision");
    assert_eq!(card.decision, "allow_once");
    assert_eq!(card.risk.as_deref(), Some("low"));
    assert!(card.rationale.contains("safe https"));
}
