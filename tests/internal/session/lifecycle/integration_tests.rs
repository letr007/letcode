use super::*;
use std::fs;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use crate::session::archive::{ArchiveBudget, SessionArchiveConfig, run_archive_pass};
use crate::transcript::{archive_index, child_sessions_dir};

const SECONDS_PER_DAY: u64 = 24 * 60 * 60;

fn write_session(base_dir: &Path, model: &str) -> String {
    let mut recorder = TranscriptRecorder::create(base_dir).unwrap();
    let session_id = recorder.session_id().to_string();
    recorder.record_session_started(model).unwrap();
    recorder.record_user_message("question").unwrap();
    recorder.record_assistant_message("answer").unwrap();
    session_id
}

/// A parent transcript with one child session, both aged past the archive
/// threshold so the next pass compresses the whole family.
fn write_aged_family(sessions_dir: &Path) -> (String, String) {
    let child_id = write_session(&child_sessions_dir(sessions_dir), "child-model");
    let parent_id = write_session(sessions_dir, "parent-model");
    let mut recorder = TranscriptRecorder::open_existing(sessions_dir, &parent_id).unwrap();
    recorder
        .record_subagent_started(
            "run-1", &parent_id, "run-1", &child_id, "explorer", "started", 1,
        )
        .unwrap();
    drop(recorder);

    let modified = SystemTime::now() - Duration::from_secs(8 * SECONDS_PER_DAY);
    for path in [
        sessions_dir.join(format!("{parent_id}.jsonl")),
        child_sessions_dir(sessions_dir).join(format!("{child_id}.jsonl")),
    ] {
        let file = fs::OpenOptions::new().write(true).open(path).unwrap();
        file.set_times(fs::FileTimes::new().set_modified(modified))
            .unwrap();
    }
    (parent_id, child_id)
}

fn archive_family(sessions_dir: &Path) -> String {
    let (parent_id, _) = write_aged_family(sessions_dir);
    let report = run_archive_pass(
        sessions_dir,
        SessionArchiveConfig::default(),
        ArchiveBudget::default(),
    )
    .unwrap();
    assert_eq!(report.archived_sessions, vec![parent_id.clone()]);
    assert!(!sessions_dir.join(format!("{parent_id}.jsonl")).exists());
    parent_id
}

#[test]
fn an_archived_family_resolves_by_id_and_prefix_then_resumes() {
    let dir = tempfile::tempdir().unwrap();
    let parent_id = archive_family(dir.path());

    assert_eq!(
        resolve_session_prefix(dir.path(), &parent_id).unwrap(),
        parent_id
    );
    let prefix = &parent_id[..8];
    assert_eq!(
        resolve_session_prefix(dir.path(), prefix).unwrap(),
        parent_id
    );

    let prepared = crate::session::prepare_resume_package(dir.path(), &parent_id).unwrap();

    assert_eq!(prepared.session_id, parent_id);
    assert!(dir.path().join(format!("{parent_id}.jsonl")).is_file());
    assert!(archive_index::archived_ids(dir.path()).unwrap().is_empty());
}

#[test]
fn an_unknown_query_still_reports_not_found_next_to_archived_sessions() {
    let dir = tempfile::tempdir().unwrap();
    archive_family(dir.path());

    let error = resolve_session_prefix(dir.path(), "no-such-session").unwrap_err();

    assert!(
        matches!(error, ResolveSessionError::NotFound { .. }),
        "expected NotFound, got {error:?}"
    );
}

#[test]
fn a_prefix_matching_two_archived_sessions_is_ambiguous() {
    let dir = tempfile::tempdir().unwrap();
    let parent_id = archive_family(dir.path());
    // The same transcript under a neighbouring id: an exact id that is also
    // a prefix of another archived session must stay ambiguous.
    let sibling_id = format!("{parent_id}-sibling");
    let archived = fs::read(archive_index::transcript_path(dir.path(), &parent_id)).unwrap();
    fs::write(
        archive_index::transcript_path(dir.path(), &sibling_id),
        archived,
    )
    .unwrap();

    let error = resolve_session_prefix(dir.path(), &parent_id).unwrap_err();
    let matches = match error {
        ResolveSessionError::Ambiguous { matches, .. } => matches,
        other => panic!("expected Ambiguous, got {other}"),
    };

    let mut matches = matches;
    matches.sort();
    assert_eq!(matches, vec![parent_id, sibling_id]);
}
