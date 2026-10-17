use super::*;
use crate::transcript::TranscriptRecorder;

fn write_session(base_dir: &Path, model: &str) -> (String, PathBuf) {
    let mut recorder = TranscriptRecorder::create(base_dir).unwrap();
    let session_id = recorder.session_id().to_string();
    recorder.record_session_started(model).unwrap();
    recorder.record_user_message("question").unwrap();
    recorder.record_assistant_message("answer").unwrap();
    let path = recorder.path().to_path_buf();
    drop(recorder);
    (session_id, path)
}

fn write_family(sessions_dir: &Path) -> (String, PathBuf, String, PathBuf) {
    let (child_id, child_path) = write_session(&child_sessions_dir(sessions_dir), "child-model");
    let (parent_id, parent_path) = write_session(sessions_dir, "parent-model");
    let mut recorder = TranscriptRecorder::open_existing(sessions_dir, &parent_id).unwrap();
    recorder
        .record_subagent_started(
            "run-1", &parent_id, "run-1", &child_id, "explorer", "started", 1,
        )
        .unwrap();
    drop(recorder);
    (parent_id, parent_path, child_id, child_path)
}

fn age(path: &Path, days: u64) {
    let modified = SystemTime::now() - Duration::from_secs(days * SECONDS_PER_DAY);
    let file = fs::OpenOptions::new().write(true).open(path).unwrap();
    file.set_times(fs::FileTimes::new().set_modified(modified))
        .unwrap();
}

fn pass(sessions_dir: &Path) -> ArchiveRunReport {
    run_archive_pass(
        sessions_dir,
        SessionArchiveConfig::default(),
        ArchiveBudget::default(),
    )
    .unwrap()
}

/// Publish a two-member family's archive files the way an archive pass does,
/// so a caller that already holds the family locks can stand in for a pass
/// that has not committed yet.
fn publish_family_archive(sessions_dir: &Path, parent_id: &str, child_id: &str) {
    archive_index::ensure_archive_subdirs(sessions_dir).unwrap();
    for (transcript, archive) in [
        (
            sessions_dir.join(format!("{parent_id}.jsonl")),
            archive_index::transcript_path(sessions_dir, parent_id),
        ),
        (
            child_sessions_dir(sessions_dir).join(format!("{child_id}.jsonl")),
            archive_index::child_transcript_path(sessions_dir, child_id),
        ),
    ] {
        let staging = temporary_path(&archive);
        compress_transcript(&transcript, &staging).unwrap();
        fs::rename(&staging, &archive).unwrap();
        fs::remove_file(&transcript).unwrap();
    }
}

#[test]
fn deleting_a_live_family_removes_transcripts_artifacts_and_rows() {
    let dir = tempfile::tempdir().unwrap();
    let sessions_dir = dir.path();
    let (parent_id, parent_path, _child_id, child_path) = write_family(sessions_dir);
    let parent_artifacts = sessions_dir.join("artifacts").join(&parent_id);
    fs::create_dir_all(&parent_artifacts).unwrap();
    fs::write(parent_artifacts.join("blob.txt"), b"x").unwrap();

    let removed = delete_session_family(sessions_dir, &parent_id).unwrap();

    assert_eq!(removed, 2);
    assert!(!parent_path.exists());
    assert!(!child_path.exists());
    assert!(!parent_artifacts.exists());
    assert!(list_sessions(sessions_dir).unwrap().is_empty());
}

#[test]
fn deleting_an_archived_family_removes_archive_and_rows() {
    let dir = tempfile::tempdir().unwrap();
    let sessions_dir = dir.path();
    let (parent_id, parent_path, child_id, child_path) = write_family(sessions_dir);
    age(&parent_path, 8);
    age(&child_path, 8);
    pass(sessions_dir);
    assert!(archive_index::rows(sessions_dir).contains_key(&parent_id));

    let removed = delete_session_family(sessions_dir, &parent_id).unwrap();

    assert_eq!(removed, 2);
    assert!(!archive_index::transcript_path(sessions_dir, &parent_id).exists());
    assert!(!archive_index::child_transcript_path(sessions_dir, &child_id).exists());
    assert!(!archive_index::rows(sessions_dir).contains_key(&parent_id));
    assert!(list_sessions(sessions_dir).unwrap().is_empty());
}

#[test]
fn a_locked_member_keeps_the_whole_family() {
    let dir = tempfile::tempdir().unwrap();
    let sessions_dir = dir.path();
    let (parent_id, parent_path, _child_id, child_path) = write_family(sessions_dir);
    let _lock = TranscriptWriterLock::try_acquire(&parent_path).unwrap();

    let error = delete_session_family(sessions_dir, &parent_id).unwrap_err();

    assert!(
        format!("{error:#}").contains(&parent_id),
        "error: {error:#}"
    );
    assert!(parent_path.is_file());
    assert!(child_path.is_file());
}

#[test]
fn idle_family_archives_and_restores_byte_identical() {
    let dir = tempfile::tempdir().unwrap();
    let (parent_id, parent_path, child_id, child_path) = write_family(dir.path());
    let parent_before = fs::read(&parent_path).unwrap();
    let child_before = fs::read(&child_path).unwrap();
    age(&parent_path, 8);
    age(&child_path, 8);

    let report = pass(dir.path());

    assert_eq!(report.archived_sessions, vec![parent_id.clone()]);
    assert_eq!(report.archived_transcripts, 2);
    assert!(report.original_bytes > report.archived_bytes);
    assert!(!parent_path.exists());
    assert!(!child_path.exists());
    assert!(archive_index::transcript_path(dir.path(), &parent_id).is_file());
    assert!(archive_index::child_transcript_path(dir.path(), &child_id).is_file());
    assert!(archive_index::rows(dir.path()).contains_key(&parent_id));

    let stats = restore_session_if_archived(dir.path(), &parent_id)
        .unwrap()
        .expect("session was archived");

    assert_eq!(stats.transcripts, 2);
    assert_eq!(stats.original_bytes, report.original_bytes);
    assert_eq!(fs::read(&parent_path).unwrap(), parent_before);
    assert_eq!(fs::read(&child_path).unwrap(), child_before);
    assert!(!archive_index::transcript_path(dir.path(), &parent_id).exists());
    assert!(!archive_index::child_transcript_path(dir.path(), &child_id).exists());
    assert!(archive_index::rows(dir.path()).is_empty());
}

#[test]
fn recent_sessions_are_left_alone() {
    let dir = tempfile::tempdir().unwrap();
    let (parent_id, parent_path, _, child_path) = write_family(dir.path());

    let report = pass(dir.path());

    assert!(report.archived_sessions.is_empty());
    assert!(parent_path.exists());
    assert!(child_path.exists());
    assert!(archive_index::archived_ids(dir.path()).unwrap().is_empty());
    assert!(!archive_index::rows(dir.path()).contains_key(&parent_id));
}

#[test]
fn a_locked_member_cancels_the_whole_family() {
    let dir = tempfile::tempdir().unwrap();
    let (parent_id, parent_path, _, child_path) = write_family(dir.path());
    age(&parent_path, 8);
    age(&child_path, 8);
    let _child_lock = TranscriptWriterLock::try_acquire(&child_path)
        .unwrap()
        .expect("child transcript lock");

    let report = pass(dir.path());

    assert_eq!(report.skipped_locked, 1);
    assert!(report.archived_sessions.is_empty());
    assert!(parent_path.exists());
    assert!(child_path.exists());
    assert!(archive_index::archived_ids(dir.path()).unwrap().is_empty());
    assert!(!archive_index::rows(dir.path()).contains_key(&parent_id));
}

#[test]
fn a_damaged_member_keeps_every_archive_and_restores_nothing() {
    let dir = tempfile::tempdir().unwrap();
    let (parent_id, parent_path, child_id, child_path) = write_family(dir.path());
    age(&parent_path, 8);
    age(&child_path, 8);
    pass(dir.path());

    let child_archive = archive_index::child_transcript_path(dir.path(), &child_id);
    fs::write(&child_archive, b"not a zstd frame").unwrap();

    let error = restore_session_if_archived(dir.path(), &parent_id).unwrap_err();

    assert!(format!("{error:#}").contains(&child_id), "error: {error:#}");
    assert!(
        !parent_path.exists(),
        "a partially restored family must not land"
    );
    assert!(archive_index::transcript_path(dir.path(), &parent_id).is_file());
    assert!(child_archive.is_file());
    assert!(archive_index::rows(dir.path()).contains_key(&parent_id));
    let staging = staging_root(dir.path()).join(format!("{parent_id}-{}", std::process::id()));
    assert!(!staging.exists());
}

#[test]
fn a_conflicting_artifacts_target_keeps_every_archive_file() {
    let dir = tempfile::tempdir().unwrap();
    let (parent_id, parent_path, child_id, child_path) = write_family(dir.path());
    let artifact_dir = dir.path().join("artifacts").join(&parent_id);
    fs::create_dir_all(&artifact_dir).unwrap();
    fs::write(artifact_dir.join("abcd.txt"), b"retained output").unwrap();
    age(&parent_path, 8);
    age(&child_path, 8);
    pass(dir.path());

    // Something else occupies the restore target.
    let artifact_target = dir.path().join("artifacts").join(&parent_id);
    assert!(!artifact_target.exists());
    fs::write(&artifact_target, b"not a directory").unwrap();

    let error = restore_session_if_archived(dir.path(), &parent_id).unwrap_err();

    assert!(
        format!("{error:#}").contains("is not a directory"),
        "error: {error:#}"
    );
    assert!(
        !parent_path.exists(),
        "nothing may land when the family cannot be fully restored"
    );
    assert!(archive_index::transcript_path(dir.path(), &parent_id).is_file());
    assert!(archive_index::child_transcript_path(dir.path(), &child_id).is_file());
    assert!(archive_index::artifacts_path(dir.path(), &parent_id).is_file());
}

#[test]
fn artifacts_are_archived_and_restored_with_the_family() {
    let dir = tempfile::tempdir().unwrap();
    let (parent_id, parent_path, _, child_path) = write_family(dir.path());
    let artifact_dir = dir.path().join("artifacts").join(&parent_id);
    fs::create_dir_all(&artifact_dir).unwrap();
    let artifact = artifact_dir.join("abcd.txt");
    fs::write(&artifact, b"retained output").unwrap();
    age(&parent_path, 8);
    age(&child_path, 8);

    pass(dir.path());

    assert!(!artifact_dir.exists());
    assert!(archive_index::artifacts_path(dir.path(), &parent_id).is_file());

    restore_session_if_archived(dir.path(), &parent_id)
        .unwrap()
        .expect("session was archived");

    assert_eq!(fs::read(&artifact).unwrap(), b"retained output");
    assert!(!archive_index::artifacts_path(dir.path(), &parent_id).exists());
}

#[test]
fn repeated_failures_are_reported_once_the_attempt_limit_is_reached() {
    let dir = tempfile::tempdir().unwrap();
    let broken = dir.path().join("broken.jsonl");
    fs::write(&broken, b"{\"kind\":\"subagent_started\" oops}\n").unwrap();
    age(&broken, 8);

    for _ in 1..MAX_ARCHIVE_ATTEMPTS {
        let report = pass(dir.path());
        assert_eq!(report.failed, vec!["broken".to_string()]);
        assert!(report.anomalies.is_empty());
    }

    let report = pass(dir.path());

    assert_eq!(report.anomalies, vec!["broken".to_string()]);
    assert!(broken.is_file(), "a failing session is never modified");
}

#[test]
fn the_byte_budget_stops_the_pass_between_families() {
    let dir = tempfile::tempdir().unwrap();
    let (first_id, first_path, _, first_child) = write_family(dir.path());
    let (second_id, second_path, _, second_child) = write_family(dir.path());
    age(&first_path, 9);
    age(&first_child, 9);
    age(&second_path, 8);
    age(&second_child, 8);

    let report = run_archive_pass(
        dir.path(),
        SessionArchiveConfig::default(),
        ArchiveBudget {
            max_bytes: 1,
            max_duration: Duration::from_secs(30),
        },
    )
    .unwrap();

    assert_eq!(report.archived_sessions, vec![first_id]);
    assert!(!first_path.exists());
    assert!(second_path.exists());
    assert!(second_child.exists());
    assert!(!archive_index::rows(dir.path()).contains_key(&second_id));
}

#[test]
fn restore_is_a_noop_for_a_live_session() {
    let dir = tempfile::tempdir().unwrap();
    let (parent_id, parent_path, _, _) = write_family(dir.path());

    assert!(
        restore_session_if_archived(dir.path(), &parent_id)
            .unwrap()
            .is_none()
    );
    assert!(parent_path.exists());
}

#[test]
fn a_lost_index_row_is_rebuilt_from_the_archived_transcript() {
    let dir = tempfile::tempdir().unwrap();
    let (parent_id, parent_path, _, child_path) = write_family(dir.path());
    age(&parent_path, 8);
    age(&child_path, 8);
    pass(dir.path());
    fs::remove_file(archive_index::archive_dir(dir.path()).join("index.json")).unwrap();

    let summaries = archived_summaries(dir.path()).unwrap();

    assert_eq!(summaries.len(), 1);
    assert_eq!(summaries[0].session_id, parent_id);
    assert!(summaries[0].record_count > 0);
    assert!(archive_index::rows(dir.path()).contains_key(&parent_id));
}

#[test]
fn an_archive_copy_of_a_live_session_is_removed() {
    let dir = tempfile::tempdir().unwrap();
    let (parent_id, parent_path, _, child_path) = write_family(dir.path());
    age(&parent_path, 8);
    age(&child_path, 8);
    pass(dir.path());
    // A restore that crashed after the transcript came back leaves both copies.
    fs::write(
        &parent_path,
        b"{\"kind\":\"session_started\",\"model\":\"m\"}\n",
    )
    .unwrap();

    pass(dir.path());

    assert!(parent_path.is_file());
    assert!(!archive_index::transcript_path(dir.path(), &parent_id).exists());
    assert!(!archive_index::rows(dir.path()).contains_key(&parent_id));
}

#[test]
fn merged_listing_holds_an_archived_and_a_live_session_once() {
    let dir = tempfile::tempdir().unwrap();
    let (archived_id, archived_path, _, child_path) = write_family(dir.path());
    let parent_bytes = fs::read(&archived_path).unwrap();
    age(&archived_path, 8);
    age(&child_path, 8);
    pass(dir.path());

    let archived_records = archive_index::rows(dir.path())[&archived_id].record_count;

    // Bring the archived parent back as a live transcript with twice as many
    // records, so the live copy is distinguishable from the archived row.
    let mut doubled = parent_bytes.clone();
    doubled.extend_from_slice(&parent_bytes);
    fs::write(&archived_path, doubled).unwrap();
    let (live_id, live_path) = write_session(dir.path(), "live-model");

    let summaries = merged_session_summaries(dir.path()).unwrap();

    let ids = summaries
        .iter()
        .map(|summary| summary.session_id.clone())
        .collect::<BTreeSet<_>>();
    assert_eq!(ids, BTreeSet::from([archived_id.clone(), live_id.clone()]));
    let recreated = summaries
        .iter()
        .find(|summary| summary.session_id == archived_id)
        .expect("the archived session is listed");
    assert_eq!(
        recreated.record_count,
        archived_records * 2,
        "the live copy wins over the archived row"
    );
    assert!(
        summaries
            .iter()
            .any(|summary| summary.session_id == live_id && summary.record_count > 0)
    );
    assert!(live_path.is_file());
    assert!(archive_index::transcript_path(dir.path(), &archived_id).is_file());
}

#[test]
fn resume_restores_an_archived_family() {
    let dir = tempfile::tempdir().unwrap();
    let (parent_id, parent_path, child_id, child_path) = write_family(dir.path());
    age(&parent_path, 8);
    age(&child_path, 8);
    pass(dir.path());

    let stats = restore_session_family_for_resume(dir.path(), &parent_id)
        .unwrap()
        .expect("archived family is restored");

    assert_eq!(stats.transcripts, 2);
    assert!(parent_path.is_file());
    assert!(child_path.is_file());
    assert!(!archive_index::transcript_path(dir.path(), &parent_id).exists());
    assert!(!archive_index::child_transcript_path(dir.path(), &child_id).exists());
}

#[test]
fn resume_of_a_live_session_skips_archive_work() {
    let dir = tempfile::tempdir().unwrap();
    let (parent_id, parent_path, _, _) = write_family(dir.path());

    assert!(
        restore_session_family_for_resume(dir.path(), &parent_id)
            .unwrap()
            .is_none()
    );
    assert!(parent_path.is_file());
}

#[test]
fn resume_waits_out_a_transient_archive_lock() {
    let dir = tempfile::tempdir().unwrap();
    let (parent_id, parent_path, _, child_path) = write_family(dir.path());
    age(&parent_path, 8);
    age(&child_path, 8);
    pass(dir.path());

    let locked_path = parent_path.clone();
    let (holding_tx, holding_rx) = std::sync::mpsc::channel();
    let holder = std::thread::spawn(move || {
        let lock = TranscriptWriterLock::try_acquire(&locked_path)
            .unwrap()
            .expect("archive lock");
        holding_tx.send(()).unwrap();
        sleep(Duration::from_millis(50));
        drop(lock);
    });
    holding_rx.recv().unwrap();

    let stats = restore_session_family_for_resume_with(
        dir.path(),
        &parent_id,
        50,
        Duration::from_millis(10),
    )
    .unwrap_or_else(|error| panic!("restore failed: {error:#}"))
    .expect("resume outlasts the transient lock");
    holder.join().unwrap();

    assert_eq!(stats.transcripts, 2);
    assert!(parent_path.is_file());
}

#[test]
fn resume_waits_out_an_archive_that_has_not_published_yet() {
    let dir = tempfile::tempdir().unwrap();
    let (parent_id, parent_path, child_id, child_path) = write_family(dir.path());
    age(&parent_path, 8);
    age(&child_path, 8);

    // An archive pass locks the whole family before its archive files become
    // visible, so the resume starts against a session that still looks live.
    let sessions_dir = dir.path().to_path_buf();
    let locked_parent = parent_path.clone();
    let locked_child = child_path.clone();
    let archived_parent = parent_id.clone();
    let archived_child = child_id.clone();
    let (holding_tx, holding_rx) = std::sync::mpsc::channel();
    let holder = std::thread::spawn(move || {
        let _parent_lock = TranscriptWriterLock::try_acquire(&locked_parent)
            .unwrap()
            .expect("parent transcript lock");
        let _child_lock = TranscriptWriterLock::try_acquire(&locked_child)
            .unwrap()
            .expect("child transcript lock");
        holding_tx.send(()).unwrap();
        sleep(Duration::from_millis(100));
        publish_family_archive(&sessions_dir, &archived_parent, &archived_child);
    });
    holding_rx.recv().unwrap();

    let stats = restore_session_family_for_resume_with(
        dir.path(),
        &parent_id,
        100,
        Duration::from_millis(10),
    )
    .unwrap_or_else(|error| panic!("resume failed: {error:#}"))
    .expect("the resume outlasts the pass and restores the family");
    holder.join().unwrap();

    assert_eq!(stats.transcripts, 2);
    assert!(parent_path.is_file());
    assert!(child_path.is_file());
    assert!(!archive_index::transcript_path(dir.path(), &parent_id).exists());
    assert!(!archive_index::child_transcript_path(dir.path(), &child_id).exists());
}

#[test]
fn an_overlapping_pass_keeps_archives_a_family_is_still_committing() {
    let dir = tempfile::tempdir().unwrap();
    let (parent_id, parent_path, child_id, child_path) = write_family(dir.path());
    let parent_before = fs::read(&parent_path).unwrap();
    let child_before = fs::read(&child_path).unwrap();
    age(&parent_path, 8);
    age(&child_path, 8);

    // A pass that has published its archive files while holding the family
    // locks, and has not removed the live transcripts yet: both copies exist.
    let parent_lock = TranscriptWriterLock::try_acquire(&parent_path)
        .unwrap()
        .expect("parent transcript lock");
    let child_lock = TranscriptWriterLock::try_acquire(&child_path)
        .unwrap()
        .expect("child transcript lock");
    archive_index::ensure_archive_subdirs(dir.path()).unwrap();
    let publish = |transcript: &Path, archive: &Path| {
        let staging = temporary_path(archive);
        compress_transcript(transcript, &staging).unwrap();
        fs::rename(&staging, archive).unwrap();
    };
    publish(
        &parent_path,
        &archive_index::transcript_path(dir.path(), &parent_id),
    );
    publish(
        &child_path,
        &archive_index::child_transcript_path(dir.path(), &child_id),
    );

    pass(dir.path());

    assert!(archive_index::transcript_path(dir.path(), &parent_id).is_file());
    assert!(archive_index::child_transcript_path(dir.path(), &child_id).is_file());
    assert_eq!(fs::read(&parent_path).unwrap(), parent_before);
    assert_eq!(fs::read(&child_path).unwrap(), child_before);

    drop(parent_lock);
    drop(child_lock);

    restore_session_family_for_resume(dir.path(), &parent_id).unwrap();

    assert_eq!(fs::read(&parent_path).unwrap(), parent_before);
    assert_eq!(fs::read(&child_path).unwrap(), child_before);
    assert!(!archive_index::transcript_path(dir.path(), &parent_id).exists());
    assert!(!archive_index::child_transcript_path(dir.path(), &child_id).exists());
}

#[test]
fn resume_finishes_a_restore_that_landed_only_the_parent() {
    let dir = tempfile::tempdir().unwrap();
    let (parent_id, parent_path, child_id, child_path) = write_family(dir.path());
    let child_before = fs::read(&child_path).unwrap();
    age(&parent_path, 8);
    age(&child_path, 8);
    pass(dir.path());

    // A restore that stopped after landing the parent: the parent transcript
    // is back, while its own archive copy and the child are still there.
    decompress_transcript(
        &archive_index::transcript_path(dir.path(), &parent_id),
        &parent_path,
    )
    .unwrap();
    let parent_before = fs::read(&parent_path).unwrap();

    let stats = restore_session_family_for_resume(dir.path(), &parent_id)
        .unwrap()
        .expect("the parked child is restored");

    assert_eq!(stats.transcripts, 1, "only the child was still archived");
    assert_eq!(
        fs::read(&parent_path).unwrap(),
        parent_before,
        "the live parent is left untouched"
    );
    assert_eq!(fs::read(&child_path).unwrap(), child_before);
    assert!(!archive_index::transcript_path(dir.path(), &parent_id).exists());
    assert!(!archive_index::child_transcript_path(dir.path(), &child_id).exists());
    assert!(archive_index::rows(dir.path()).is_empty());

    assert!(
        restore_session_family_for_resume(dir.path(), &parent_id)
            .unwrap()
            .is_none(),
        "a finished family has nothing left to do"
    );
    assert_eq!(fs::read(&parent_path).unwrap(), parent_before);
    assert_eq!(fs::read(&child_path).unwrap(), child_before);
}

#[test]
fn a_family_locked_by_an_archive_is_reported_as_busy() {
    let dir = tempfile::tempdir().unwrap();
    let (parent_id, parent_path, _, child_path) = write_family(dir.path());
    age(&parent_path, 8);
    age(&child_path, 8);
    pass(dir.path());
    let _lock = TranscriptWriterLock::try_acquire(&parent_path)
        .unwrap()
        .expect("archive lock");

    let error =
        restore_session_family_for_resume_with(dir.path(), &parent_id, 3, Duration::from_millis(1))
            .expect_err("a held family lock is reported instead of waiting forever");

    assert!(
        format!("{error:#}").contains(&parent_id),
        "error: {error:#}"
    );
    assert!(!parent_path.exists());
    assert!(archive_index::transcript_path(dir.path(), &parent_id).is_file());
}
