use super::*;

#[test]
fn source_span_rejects_inverted_ranges() {
    let error = SourceSpan::new(9, 3).expect_err("inverted spans must fail");

    assert!(
        error
            .to_string()
            .contains("source span start_sequence must be <= end_sequence")
    );
}

fn prompt_contributor(retains_raw_sources: bool) -> PromptContributorPlaceholder {
    PromptContributorPlaceholder {
        contributor_id: "test-contributor".into(),
        kind: PromptContributorKind::RuntimeContext,
        label: None,
        provenance: RuntimeFrameProvenance::new(RuntimeSource::ContextView)
            .with_span(SourceSpan::new(3, 3).expect("valid source span")),
        retains_raw_sources,
        frame_ids: Vec::new(),
        source_frame_ids: Vec::new(),
    }
}

#[test]
fn workflow_state_serde_is_backward_compatible_and_omits_empty_state() {
    let empty = RuntimeSnapshot::new("main");
    let empty_json = serde_json::to_value(&empty).expect("serialize empty snapshot");
    assert!(empty_json.get("workflow").is_none());

    let legacy: RuntimeSnapshot = serde_json::from_value(empty_json)
        .expect("legacy snapshot without workflow remains readable");
    assert!(legacy.workflow.is_empty());

    let mut populated = RuntimeSnapshot::new("main");
    populated.workflow.todos = vec![crate::agent::TodoItem {
        id: "todo".into(),
        content: "persist workflow".into(),
        status: crate::agent::TodoStatus::InProgress,
    }];
    populated.workflow.auto_continue.enabled = true;
    let populated_json = serde_json::to_value(&populated).expect("serialize workflow snapshot");
    let restored: RuntimeSnapshot =
        serde_json::from_value(populated_json).expect("restore workflow snapshot");
    assert_eq!(restored.workflow, populated.workflow);
}

#[test]
fn snapshot_reference_validation_rejects_duplicate_ids() {
    let mut snapshot = RuntimeSnapshot::new("main");
    let frame = RuntimeFrame::new(
        RuntimeFrameKind::Metadata,
        FrameVisibility::Active,
        RuntimeFrameProvenance::new(RuntimeSource::Derived),
        RuntimeFrameIdSeed {
            frame_kind: RuntimeFrameKind::Metadata,
            source: RuntimeSource::Derived,
            ordinal: 0,
            stable_key: "one",
            source_span: None,
        },
    );
    snapshot.push_frame(frame.clone());
    snapshot.push_frame(frame);
    assert!(snapshot.validate_structure().is_err());
    assert!(snapshot.validate_references().is_err());
}

#[test]
fn heal_references_scrubs_dangling_compaction_and_contributor_ids() {
    let mut snapshot = RuntimeSnapshot::new("main");
    let frame = RuntimeFrame::new(
        RuntimeFrameKind::Metadata,
        FrameVisibility::Active,
        RuntimeFrameProvenance::new(RuntimeSource::Derived),
        RuntimeFrameIdSeed {
            frame_kind: RuntimeFrameKind::Metadata,
            source: RuntimeSource::Derived,
            ordinal: 0,
            stable_key: "live",
            source_span: None,
        },
    );
    let live_id = frame.id;
    let ghost = RuntimeFrameId::from_persisted(0xdead_beef);
    snapshot.push_frame(frame);
    snapshot.compaction.protected_frame_ids.push(ghost);
    snapshot
        .compaction
        .compacted_frame_ids
        .extend([live_id, ghost]);
    snapshot
        .prompt_contributors
        .push(PromptContributorPlaceholder {
            contributor_id: "c1".into(),
            kind: PromptContributorKind::RuntimeContext,
            label: None,
            provenance: RuntimeFrameProvenance::new(RuntimeSource::Derived),
            retains_raw_sources: true,
            frame_ids: vec![ghost],
            source_frame_ids: vec![live_id, ghost],
        });

    assert!(!snapshot.dangling_reference_report().is_empty());
    // Non-fatal on shared validate path.
    snapshot
        .validate_references()
        .expect("dangling must not fail-close");

    let report = snapshot.heal_references().expect("heal");
    assert!(!report.is_empty());
    assert!(snapshot.dangling_reference_report().is_empty());
    assert_eq!(
        snapshot.compaction.protected_frame_ids,
        Vec::<RuntimeFrameId>::new()
    );
    assert_eq!(snapshot.compaction.compacted_frame_ids, vec![live_id]);
    assert_eq!(
        snapshot.prompt_contributors[0].frame_ids,
        Vec::<RuntimeFrameId>::new()
    );
    assert_eq!(
        snapshot.prompt_contributors[0].source_frame_ids,
        vec![live_id]
    );
}

#[test]
fn replace_frames_scrubs_stale_reference_sets() {
    let mut snapshot = RuntimeSnapshot::new("main");
    let keep = RuntimeFrame::new(
        RuntimeFrameKind::User,
        FrameVisibility::Active,
        RuntimeFrameProvenance::new(RuntimeSource::Transcript),
        RuntimeFrameIdSeed {
            frame_kind: RuntimeFrameKind::User,
            source: RuntimeSource::Transcript,
            ordinal: 0,
            stable_key: "keep",
            source_span: None,
        },
    );
    let drop = RuntimeFrame::new(
        RuntimeFrameKind::User,
        FrameVisibility::Active,
        RuntimeFrameProvenance::new(RuntimeSource::Transcript),
        RuntimeFrameIdSeed {
            frame_kind: RuntimeFrameKind::User,
            source: RuntimeSource::Transcript,
            ordinal: 1,
            stable_key: "drop",
            source_span: None,
        },
    );
    let drop_id = drop.id;
    let keep_id = keep.id;
    snapshot.push_frame(keep.clone());
    snapshot.push_frame(drop);
    snapshot.compaction.compacted_frame_ids.push(drop_id);
    snapshot.compaction.protected_frame_ids.push(drop_id);
    snapshot.replace_frames(vec![keep]);
    assert!(snapshot.dangling_reference_report().is_empty());
    assert!(!snapshot.compaction.compacted_frame_ids.contains(&drop_id));
    assert!(snapshot.frames.iter().any(|f| f.id == keep_id));
}
