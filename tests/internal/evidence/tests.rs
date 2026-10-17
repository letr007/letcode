use super::*;
use crate::agent::{
    ToolEffectKind, ToolEffects, ToolExecutionRecord, ToolExecutionRejection, ToolExecutionStatus,
};
use crate::permission::ToolPermissionClass;
use crate::transcript::{TranscriptEvent, TranscriptRecord};
use serde_json::json;

fn file_record(id: &str, path: &str, sequence: u64) -> EvidenceRecord {
    EvidenceRecord {
        id: id.into(),
        sequence,
        timestamp_ms: 0,
        evidence_kind: EvidenceKind::FileExcerpt,
        title: format!("read {path}"),
        summary: format!("read {path}"),
        detail: Some("old contents".into()),
        source: EvidenceSource::File {
            path: path.into(),
            start_line: Some(1),
            end_line: Some(2),
        },
        tags: vec![path.into()],
    }
}

fn tool_execution_record(
    tool_name: &str,
    arguments: Option<Value>,
    output: ToolResult,
    effects: ToolEffects,
) -> ToolExecutionRecord {
    ToolExecutionRecord {
        call_id: format!("call-{tool_name}"),
        tool_name: tool_name.into(),
        arguments,
        permission_class: ToolPermissionClass::Unknown,
        status: if output.ok {
            ToolExecutionStatus::Executed
        } else {
            ToolExecutionStatus::Rejected
        },
        rejection: (!output.ok).then_some(ToolExecutionRejection::PermissionDeniedByPolicy),
        output,
        effects,
    }
}

fn transcript_evidence_record(id: &str, sequence: u64) -> TranscriptRecord {
    TranscriptRecord {
        session_id: "s".into(),
        sequence,
        timestamp_ms: sequence as u128,
        context_branch_id: None,
        event: TranscriptEvent::Evidence {
            id: id.into(),
            evidence_kind: EvidenceKind::Decision,
            title: format!("title-{sequence}"),
            summary: format!("summary-{sequence}"),
            detail: None,
            source: EvidenceSource::Transcript { sequence },
            tags: vec![],
        },
    }
}

#[test]
fn apply_patch_change_tags_make_older_file_evidence_stale() {
    let old_file = file_record("ev-old", "src/lib.rs", 1);
    let change = EvidenceRecord {
        id: "ev-change".into(),
        sequence: 2,
        timestamp_ms: 0,
        evidence_kind: EvidenceKind::Change,
        title: "apply patch".into(),
        summary: "changed src/lib.rs".into(),
        detail: None,
        source: EvidenceSource::ToolCall {
            call_id: "call-patch".into(),
            tool: "edit__apply_patch".into(),
        },
        tags: vec!["src/lib.rs".into()],
    };

    let (message, ids, _) = evidence_context_message(&[old_file, change], "src/lib.rs", 1024);
    let message = message.expect("evidence selected");
    assert!(ids.contains(&"ev-change".to_string()), "{message}");
    assert!(!ids.contains(&"ev-old".to_string()), "{message}");
}

#[test]
fn duplicate_legacy_agent_ids_restore_with_deterministic_rewrite() {
    let records = vec![
        transcript_evidence_record("ev-agent-000002", 2),
        transcript_evidence_record("ev-agent-000002", 7),
        transcript_evidence_record("ev-agent-000002", 9),
    ];

    let restored = restore_evidence_records(&records).expect("legacy duplicates should restore");

    assert_eq!(restored.len(), 3);
    assert_eq!(restored[0].id, "ev-agent-000002");
    assert_eq!(restored[1].id, "ev-agent-000002-seq-7");
    assert_eq!(restored[2].id, "ev-agent-000002-seq-9");
}

#[test]
fn duplicate_non_legacy_ids_still_fail_restore() {
    let records = vec![
        transcript_evidence_record("ev-custom-1", 1),
        transcript_evidence_record("ev-custom-1", 2),
    ];

    let error = restore_evidence_records(&records).expect_err("non-legacy duplicate should fail");
    assert!(
        error
            .to_string()
            .contains("duplicate evidence id: ev-custom-1")
    );
}
