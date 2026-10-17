use super::*;
use std::fs;
use tempfile::tempdir;

mod integration_tests {
    use super::*;

    /// The file scan and the projection over records must agree.
    #[test]
    fn the_file_scan_agrees_with_the_projection_over_records() {
        let temp = tempdir().unwrap();
        let child_dir = temp.path().join("children");
        let mut parent = crate::transcript::TranscriptRecorder::create(temp.path()).unwrap();
        let parent_session_id = parent.session_id().to_string();

        // Two children with a transcript, one of them still running, and a third
        // whose record names a transcript that was never written.
        let mut recorded_children = Vec::new();
        for _ in 0..2 {
            let mut child = crate::transcript::TranscriptRecorder::create(&child_dir).unwrap();
            recorded_children.push(child.session_id().to_string());
            child.record_user_message("child transcript").unwrap();
        }
        let missing_child = "child-without-transcript".to_string();

        fn start(
            parent: &mut crate::transcript::TranscriptRecorder,
            run_id: &str,
            parent_session_id: &str,
            child_session_id: &str,
            ordinal: u32,
        ) {
            parent
                .record_subagent_started(
                    run_id,
                    parent_session_id.to_string(),
                    "turn-1",
                    child_session_id,
                    "explorer",
                    "inspect child",
                    ordinal,
                )
                .unwrap();
        }

        start(
            &mut parent,
            "run-1",
            &parent_session_id,
            &recorded_children[0],
            1,
        );
        parent
            .record_subagent_result_structured(
                "run-1".to_string(),
                parent_session_id.clone(),
                "turn-1".to_string(),
                recorded_children[0].clone(),
                "explorer".to_string(),
                "completed".to_string(),
                "finished".to_string(),
                None,
            )
            .unwrap();
        start(
            &mut parent,
            "run-2",
            &parent_session_id,
            &recorded_children[1],
            2,
        );
        start(&mut parent, "run-3", &parent_session_id, &missing_child, 3);
        let parent_path = parent.path().to_path_buf();
        drop(parent);

        let records = crate::transcript::read_records(&parent_path).unwrap();
        let from_records = project_child_session_summaries(&child_dir, &records);
        let from_file = project_child_session_summaries_from_file(&child_dir, &parent_path)
            .expect("scan the parent journal");

        assert_eq!(from_records.len(), 2, "only children with a transcript");
        assert_eq!(from_records, from_file);
    }
}

#[test]
fn child_summary_file_scan_ignores_large_unrelated_records() {
    let temp = tempdir().unwrap();
    let child_dir = temp.path().join("children");
    fs::create_dir(&child_dir).unwrap();
    fs::write(child_dir.join("child.jsonl"), "").unwrap();
    let parent_path = temp.path().join("parent.jsonl");
    fs::write(
            &parent_path,
            concat!(
                "{\"session_id\":\"parent\",\"sequence\":1,\"timestamp_ms\":1,\"kind\":\"tool_call_finished\",\"output\":{\"data\":\"large\"}}\n",
                "{\"session_id\":\"parent\",\"sequence\":2,\"timestamp_ms\":2,\"kind\":\"subagent_started\",\"parent_session_id\":\"parent\",\"parent_run_id\":\"run\",\"child_session_id\":\"child\",\"agent_name\":\"explorer\",\"summary\":\"started\",\"pool_ordinal\":1}\n",
                "{\"session_id\":\"parent\",\"sequence\":3,\"timestamp_ms\":3,\"kind\":\"subagent_result\",\"parent_session_id\":\"parent\",\"parent_run_id\":\"run\",\"child_session_id\":\"child\",\"agent_name\":\"explorer\",\"status\":\"done\",\"summary\":\"finished\"}\n",
            ),
        )
        .unwrap();

    let children = project_child_session_summaries_from_file(&child_dir, &parent_path).unwrap();
    assert_eq!(children.len(), 1);
    assert_eq!(children[0].status, "done");
    assert_eq!(children[0].summary, "finished");
}
