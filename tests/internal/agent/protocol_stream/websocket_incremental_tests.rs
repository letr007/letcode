use super::*;
use crate::model_runtime::{PreparedPromptUnitInspection, PreparedRequestInspection};
use crate::request_builder::prompt_plan::{
    PromptPlan, PromptSegment, PromptSegmentContent, PromptSegmentProtection,
    PromptSegmentRetention, PromptSegmentRole, PromptSegmentSource, PromptSegmentStability,
    PromptTokenEstimate,
};
use crate::runtime_context::{
    PromptContributorKind, RuntimeFrameProvenance, RuntimeSource, SourceSpan,
};

fn inspection(units: Vec<(&str, &[&str])>) -> PreparedRequestInspection {
    PreparedRequestInspection {
        request_shape: vec![1],
        prompt_units: units
            .into_iter()
            .map(|(kind, ids)| PreparedPromptUnitInspection {
                identity: serde_json::json!({"type": kind}).to_string().into_bytes(),
                semantic_segment_ids: ids.iter().map(|id| (*id).to_string()).collect(),
            })
            .collect(),
        cache: Default::default(),
    }
}

fn segment(
    id: &str,
    role: PromptSegmentRole,
    source_span: Option<SourceSpan>,
    source_label: Option<&str>,
    content: PromptSegmentContent,
) -> PromptSegment {
    PromptSegment {
        id: id.into(),
        order: 0,
        role,
        contributor_id: id.into(),
        source: PromptSegmentSource {
            order: 0,
            contributor_kind: PromptContributorKind::CurrentTurn,
            provenance: RuntimeFrameProvenance {
                source: RuntimeSource::Transcript,
                label: None,
                source_span,
                source_id: None,
            },
            source_key: None,
            source_label: source_label.map(str::to_owned),
        },
        stability: PromptSegmentStability::Volatile,
        retention: PromptSegmentRetention::Required,
        protection: PromptSegmentProtection::default(),
        cache: crate::request_builder::prompt_plan::PromptCacheMetadata {
            cache_eligible: false,
            boundary: None,
            prefix_hash: None,
        },
        tokens: PromptTokenEstimate {
            estimated_input_tokens: None,
            budget_input_tokens: None,
            actual_input_tokens: None,
        },
        text: id.into(),
        content,
    }
}

fn plan_with_assistant_and_tool(
    assistant_span: Option<SourceSpan>,
    tool_span: Option<SourceSpan>,
) -> PromptPlan {
    PromptPlan {
        model_id: "test".into(),
        contributors: Vec::new(),
        segments: vec![
            segment(
                "assistant",
                PromptSegmentRole::Assistant,
                assistant_span,
                None,
                PromptSegmentContent::AssistantToolCalls {
                    text: Some("calling lookup".into()),
                    reasoning_content: None,
                    replay: None,
                    calls: vec![crate::request_builder::HistoryToolCall {
                        call_id: "call".into(),
                        name: "lookup".into(),
                        arguments_json: "{}".into(),
                    }],
                },
            ),
            segment(
                "tool",
                PromptSegmentRole::Tool,
                tool_span,
                None,
                PromptSegmentContent::ToolOutput {
                    call_id: "call".into(),
                    output_json: "{}".into(),
                    images: Vec::new(),
                },
            ),
        ],
        stable_prefix_end: None,
        kernel_end_exclusive: 0,
        envelope_end_exclusive: 0,
    }
}

fn previous_snapshot() -> WsRequestSnapshot {
    WsRequestSnapshot {
        inspection: inspection(vec![("instructions", &["system"]), ("message", &["user"])]),
        assistant_frontier: 10,
    }
}

#[test]
fn websocket_incremental_boundary_accepts_only_new_tool_output() {
    let current = inspection(vec![
        ("instructions", &["system"]),
        ("message", &["user"]),
        ("message", &["assistant"]),
        ("message", &["tool"]),
    ]);
    assert_eq!(
        websocket_incremental_prompt_unit_start(
            &current,
            &plan_with_assistant_and_tool(
                Some(SourceSpan::new(10, 10).unwrap()),
                Some(SourceSpan::new(11, 12).unwrap()),
            ),
            &previous_snapshot(),
        ),
        Some(3)
    );
}

#[test]
fn websocket_incremental_boundary_skips_repeated_assistant_units() {
    let current = inspection(vec![
        ("instructions", &["system"]),
        ("message", &["user"]),
        ("message", &["assistant"]),
        ("message", &["assistant"]),
        ("message", &["tool"]),
    ]);
    assert_eq!(
        websocket_incremental_prompt_unit_start(
            &current,
            &plan_with_assistant_and_tool(
                Some(SourceSpan::new(10, 10).unwrap()),
                Some(SourceSpan::new(11, 12).unwrap()),
            ),
            &previous_snapshot(),
        ),
        Some(4)
    );
}

#[test]
fn websocket_incremental_boundary_resets_for_unsafe_prefix_or_missing_span() {
    let unsafe_prefix = inspection(vec![
        ("instructions", &["changed"]),
        ("message", &["user"]),
        ("message", &["assistant"]),
        ("message", &["tool"]),
    ]);
    assert_eq!(
        websocket_incremental_prompt_unit_start(
            &unsafe_prefix,
            &plan_with_assistant_and_tool(
                Some(SourceSpan::new(10, 10).unwrap()),
                Some(SourceSpan::new(11, 12).unwrap()),
            ),
            &previous_snapshot(),
        ),
        None
    );

    let current = inspection(vec![
        ("instructions", &["system"]),
        ("message", &["user"]),
        ("message", &["assistant"]),
        ("message", &["tool"]),
    ]);
    assert_eq!(
        websocket_incremental_prompt_unit_start(
            &current,
            &plan_with_assistant_and_tool(Some(SourceSpan::new(10, 10).unwrap()), None,),
            &previous_snapshot(),
        ),
        None
    );

    let crossing = inspection(vec![
        ("instructions", &["system"]),
        ("message", &["user"]),
        ("message", &["assistant"]),
        ("message", &["tool"]),
    ]);
    assert_eq!(
        websocket_incremental_prompt_unit_start(
            &crossing,
            &plan_with_assistant_and_tool(
                Some(SourceSpan::new(10, 11).unwrap()),
                Some(SourceSpan::new(12, 13).unwrap()),
            ),
            &previous_snapshot(),
        ),
        None
    );

    let old_after_new = inspection(vec![
        ("instructions", &["system"]),
        ("message", &["user"]),
        ("message", &["assistant"]),
        ("message", &["tool"]),
        ("message", &["assistant"]),
    ]);
    assert_eq!(
        websocket_incremental_prompt_unit_start(
            &old_after_new,
            &plan_with_assistant_and_tool(
                Some(SourceSpan::new(10, 10).unwrap()),
                Some(SourceSpan::new(11, 12).unwrap()),
            ),
            &previous_snapshot(),
        ),
        None
    );

    let multi_origin = inspection(vec![
        ("instructions", &["system"]),
        ("message", &["user"]),
        ("message", &["assistant", "other"]),
        ("message", &["tool"]),
    ]);
    assert_eq!(
        websocket_incremental_prompt_unit_start(
            &multi_origin,
            &plan_with_assistant_and_tool(
                Some(SourceSpan::new(10, 10).unwrap()),
                Some(SourceSpan::new(11, 12).unwrap()),
            ),
            &previous_snapshot(),
        ),
        None
    );
}
