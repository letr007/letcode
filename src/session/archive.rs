//! Age-based archiving of idle session transcripts.
//!
//! A session family — the parent transcript, every descendant transcript under
//! `children/`, and their retained artifacts — is archived as one unit: all
//! members are locked for the whole operation, compressed into temporary names
//! first, and only committed once every member succeeded. A failure therefore
//! leaves the family untouched and is retried on a later pass.
//!
//! Restore runs synchronously before a resume: members that are live again are
//! skipped and the rest are decompressed into a staging directory, parsed, and
//! only then committed, so the archive files are removed only after every
//! restored member is back in place. Landing nothing when a member cannot be
//! restored keeps the call retryable, which is what lets a later resume finish a
//! family whose earlier restore stopped halfway.

use std::collections::{BTreeSet, VecDeque};
use std::fs::{self, File};
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};
use std::thread::sleep;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result, anyhow, ensure};
use serde::Deserialize;
use tracing::warn;

use crate::transcript::archive_index::{self, ArchivedSession};
use crate::transcript::{
    SessionSummary, TranscriptEvent, TranscriptRecord, TranscriptWriterLock, child_sessions_dir,
    list_sessions, read_records_allow_partial_tail, summarize_session_reader,
};

/// Consecutive failures after which a session is reported as anomalous.
pub(crate) const MAX_ARCHIVE_ATTEMPTS: u32 = 3;

const ZSTD_LEVEL: i32 = 9;
const STAGING_DIR: &str = ".restore";
const STALE_STAGING_AGE: Duration = Duration::from_secs(24 * 60 * 60);
const ARCHIVING_TMP_SUFFIX: &str = ".tmp";
const SECONDS_PER_DAY: u64 = 24 * 60 * 60;
/// How long a resume waits for an in-flight archive pass to release the family
/// lock before reporting the session as busy.
const RESUME_LOCK_ATTEMPTS: usize = 5;
const RESUME_LOCK_RETRY_DELAY: Duration = Duration::from_millis(250);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct SessionArchiveConfig {
    pub(crate) enabled: bool,
    pub(crate) older_than_days: u64,
}

impl Default for SessionArchiveConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            older_than_days: 7,
        }
    }
}

/// Work a single pass may spend before returning, so a background pass stays
/// quiet even when the sessions directory holds gigabytes of history.
#[derive(Debug, Clone, Copy)]
pub(crate) struct ArchiveBudget {
    pub(crate) max_bytes: u64,
    pub(crate) max_duration: Duration,
}

impl Default for ArchiveBudget {
    fn default() -> Self {
        Self {
            max_bytes: 512 * 1024 * 1024,
            max_duration: Duration::from_secs(30),
        }
    }
}

#[derive(Debug, Default)]
pub(crate) struct ArchiveRunReport {
    pub(crate) archived_sessions: Vec<String>,
    pub(crate) archived_transcripts: usize,
    pub(crate) original_bytes: u64,
    pub(crate) archived_bytes: u64,
    pub(crate) skipped_recent: usize,
    pub(crate) skipped_locked: usize,
    pub(crate) failed: Vec<String>,
    /// Sessions whose failure count reached [`MAX_ARCHIVE_ATTEMPTS`] in this pass.
    pub(crate) anomalies: Vec<String>,
}

#[derive(Debug)]
pub(crate) struct RestoreStats {
    pub(crate) transcripts: usize,
    pub(crate) original_bytes: u64,
}

#[derive(Debug)]
struct FamilyMember {
    session_id: String,
    transcript: PathBuf,
}

#[derive(Debug)]
struct FamilyStats {
    transcripts: usize,
    original_bytes: u64,
    archived_bytes: u64,
}

enum FamilyArchiveError {
    Locked,
    Io(anyhow::Error),
}

impl From<anyhow::Error> for FamilyArchiveError {
    fn from(error: anyhow::Error) -> Self {
        Self::Io(error)
    }
}

#[derive(Debug)]
struct RestoreMember {
    /// Live transcript this member is restored to.
    target: PathBuf,
    staged: PathBuf,
    archived: PathBuf,
}

#[derive(Debug)]
struct RestorePlan {
    /// Members the archive still holds the only copy of.
    members: Vec<RestoreMember>,
    artifacts: Vec<RestoreArtifacts>,
    /// Archive copies of members that are live again: a restore that stopped
    /// between landing a member and dropping its archive leaves both copies, and
    /// the live transcript is the one that counts.
    stale_archives: Vec<PathBuf>,
    original_bytes: u64,
}

#[derive(Debug)]
struct RestoreArtifacts {
    target: PathBuf,
    staged: PathBuf,
    archived: PathBuf,
}

/// Archive idle session families, oldest first, within `budget`.
pub(crate) fn run_archive_pass(
    sessions_dir: &Path,
    config: SessionArchiveConfig,
    budget: ArchiveBudget,
) -> Result<ArchiveRunReport> {
    let mut report = ArchiveRunReport::default();
    if !config.enabled {
        return Ok(report);
    }

    remove_stale_archives(sessions_dir)?;
    cleanup_stale_staging(sessions_dir);

    let cutoff = cutoff_time(config.older_than_days);
    let deadline = Instant::now() + budget.max_duration;
    let mut spent_bytes = 0_u64;

    for (session_id, _) in idle_session_candidates(sessions_dir, cutoff)? {
        if spent_bytes >= budget.max_bytes || Instant::now() >= deadline {
            break;
        }

        let members = match collect_family_members(sessions_dir, &session_id) {
            Ok(members) => members,
            Err(error) => {
                record_failure(&mut report, sessions_dir, &session_id, error)?;
                continue;
            }
        };
        if !family_is_idle(&members, cutoff) {
            report.skipped_recent += 1;
            continue;
        }

        match archive_family(sessions_dir, &session_id, &members) {
            Ok(stats) => {
                report.archived_sessions.push(session_id);
                report.archived_transcripts += stats.transcripts;
                report.original_bytes += stats.original_bytes;
                report.archived_bytes += stats.archived_bytes;
                spent_bytes = spent_bytes.saturating_add(stats.original_bytes);
            }
            Err(FamilyArchiveError::Locked) => report.skipped_locked += 1,
            Err(FamilyArchiveError::Io(error)) => {
                record_failure(&mut report, sessions_dir, &session_id, error)?;
            }
        }
    }

    Ok(report)
}

/// Restore the family of `session_id` when any member still lives only in the
/// archive.
///
/// Restoration is incremental and idempotent: members that are live again are
/// skipped, so a resume finishes a family whose earlier restore stopped after
/// some of its members were back on disk.
///
/// Returns `Ok(None)` when no member of the family is archived, and leaves the
/// archive files untouched on any failure.
pub(crate) fn restore_session_if_archived(
    sessions_dir: &Path,
    session_id: &str,
) -> Result<Option<RestoreStats>> {
    let staging = staging_root(sessions_dir).join(format!("{session_id}-{}", std::process::id()));

    let restored =
        plan_family_restore(sessions_dir, session_id, &staging).and_then(|plan| match plan {
            Some(plan) => commit_restore(sessions_dir, session_id, plan).map(Some),
            None => Ok(None),
        });

    if let Err(error) = fs::remove_dir_all(&staging)
        && error.kind() != std::io::ErrorKind::NotFound
    {
        warn!(path = %staging.display(), error = %error, "failed to remove restore staging");
    }

    restored
}

/// Live sessions plus archived ones, newest first, each id listed once.
pub(crate) fn merged_session_summaries(sessions_dir: &Path) -> Result<Vec<SessionSummary>> {
    let mut summaries = list_sessions(sessions_dir)?;
    let mut listed: BTreeSet<String> = summaries
        .iter()
        .map(|summary| summary.session_id.clone())
        .collect();

    for summary in archived_summaries(sessions_dir)? {
        if listed.insert(summary.session_id.clone()) {
            summaries.push(summary);
        }
    }

    summaries.sort_by_key(|summary| summary.last_timestamp_ms.unwrap_or(0));
    summaries.reverse();
    Ok(summaries)
}

/// Listing summaries for archived sessions, rebuilt from the transcripts when a
/// cached row is missing so a lost row never hides an archived session.
pub(crate) fn archived_summaries(sessions_dir: &Path) -> Result<Vec<SessionSummary>> {
    let ids = archive_index::archived_ids(sessions_dir)?;
    let rows = archive_index::rows(sessions_dir);
    let mut summaries = Vec::with_capacity(ids.len());

    for session_id in ids {
        if let Some(row) = rows.get(&session_id) {
            summaries.push(row.summary(session_id));
            continue;
        }

        match reconcile_row(sessions_dir, &session_id) {
            Ok(Some(row)) => {
                summaries.push(row.summary(session_id.clone()));
                if let Err(error) = archive_index::upsert(sessions_dir, &session_id, row) {
                    warn!(session_id = %session_id, error = %error, "failed to cache archived session row");
                }
            }
            Ok(None) => {}
            Err(error) => {
                // Still list the session: the failure belongs to the resume path,
                // which reports it with the real cause.
                warn!(session_id = %session_id, error = %error, "failed to summarize archived session");
                summaries.push(SessionSummary {
                    session_id,
                    record_count: 0,
                    first_timestamp_ms: None,
                    last_timestamp_ms: None,
                    model: None,
                    title: None,
                    last_user_summary: None,
                    last_assistant_summary: None,
                });
            }
        }
    }

    summaries.sort_by_key(|summary| summary.last_timestamp_ms.unwrap_or(0));
    summaries.reverse();
    Ok(summaries)
}

/// Restore the family of `session_id` when a resume needs it, waiting out an
/// archive pass that currently holds the family's writer lock.
///
/// Returns `Ok(None)` when the family is live by the time the lock is free, and
/// reports a family that is still archived as busy.
pub(crate) fn restore_session_family_for_resume(
    sessions_dir: &Path,
    session_id: &str,
) -> Result<Option<RestoreStats>> {
    restore_session_family_for_resume_with(
        sessions_dir,
        session_id,
        RESUME_LOCK_ATTEMPTS,
        RESUME_LOCK_RETRY_DELAY,
    )
}

fn restore_session_family_for_resume_with(
    sessions_dir: &Path,
    session_id: &str,
    attempts: usize,
    delay: Duration,
) -> Result<Option<RestoreStats>> {
    let transcript = sessions_dir.join(format!("{session_id}.jsonl"));

    // The lock excludes an archive pass that is mid-family, and any second
    // resume of the same archived family, so the staging directory and the
    // commit stay single-writer. A pass holds the lock from before its archive
    // files become visible, so a resume that arrives in that window finds a
    // session that still looks live and has to wait for the pass to finish.
    for attempt in 0..attempts {
        if let Some(_lock) = TranscriptWriterLock::try_acquire(&transcript)? {
            return restore_session_if_archived(sessions_dir, session_id);
        }
        if attempt + 1 == attempts {
            break;
        }
        sleep(delay);
    }

    // A transcript that exists only in the archive can only be locked by a
    // process archiving or restoring it, so the retry budget is all a resume can
    // offer. A live transcript belongs to whoever holds the lock, and the
    // caller's own open reports that.
    if session_is_archived(sessions_dir, session_id) {
        return Err(anyhow!(
            "session {session_id} is busy: another process is archiving or restoring it; retry the resume in a moment"
        ));
    }
    Ok(None)
}

/// Whether the session currently lives in the archive rather than on disk.
pub(crate) fn session_is_archived(sessions_dir: &Path, session_id: &str) -> bool {
    !sessions_dir.join(format!("{session_id}.jsonl")).is_file()
        && archive_index::transcript_path(sessions_dir, session_id).is_file()
}

fn record_failure(
    report: &mut ArchiveRunReport,
    sessions_dir: &Path,
    session_id: &str,
    error: anyhow::Error,
) -> Result<()> {
    warn!(session_id = %session_id, error = %error, "session archiving failed");
    let attempts = archive_index::bump_attempts(sessions_dir, session_id)?;
    report.failed.push(session_id.to_string());
    if attempts == MAX_ARCHIVE_ATTEMPTS {
        report.anomalies.push(session_id.to_string());
    }
    Ok(())
}

fn cutoff_time(older_than_days: u64) -> SystemTime {
    let age = Duration::from_secs(older_than_days.saturating_mul(SECONDS_PER_DAY));
    SystemTime::now().checked_sub(age).unwrap_or(UNIX_EPOCH)
}

fn idle_session_candidates(
    sessions_dir: &Path,
    cutoff: SystemTime,
) -> Result<Vec<(String, SystemTime)>> {
    let entries = match fs::read_dir(sessions_dir) {
        Ok(entries) => entries,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
        Err(error) => {
            return Err(error).with_context(|| {
                format!(
                    "failed to read sessions directory {}",
                    sessions_dir.display()
                )
            });
        }
    };

    let mut candidates = Vec::new();
    for entry in entries {
        let entry = entry?;
        let path = entry.path();
        if path.extension().and_then(|extension| extension.to_str()) != Some("jsonl") {
            continue;
        }
        let Some(session_id) = path.file_stem().and_then(|stem| stem.to_str()) else {
            continue;
        };
        let modified = entry.metadata()?.modified().unwrap_or(UNIX_EPOCH);
        if modified <= cutoff {
            candidates.push((session_id.to_string(), modified));
        }
    }
    candidates.sort_by_key(|(_, modified)| *modified);
    Ok(candidates)
}

/// Parent plus every descendant transcript, discovered through the recorded
/// subagent boundaries. Nested children share the flat `children/` directory.
fn collect_family_members(
    sessions_dir: &Path,
    parent_session_id: &str,
) -> Result<Vec<FamilyMember>> {
    let child_dir = child_sessions_dir(sessions_dir);
    let parent_transcript = sessions_dir.join(format!("{parent_session_id}.jsonl"));
    let mut members = vec![FamilyMember {
        session_id: parent_session_id.to_string(),
        transcript: parent_transcript.clone(),
    }];
    let mut seen: BTreeSet<String> = BTreeSet::from([parent_session_id.to_string()]);
    let mut queue = VecDeque::from([parent_transcript]);

    while let Some(transcript) = queue.pop_front() {
        for child_id in child_session_ids(&transcript)? {
            if !seen.insert(child_id.clone()) {
                continue;
            }
            let child_transcript = child_dir.join(format!("{child_id}.jsonl"));
            if !child_transcript.is_file() {
                continue;
            }
            members.push(FamilyMember {
                session_id: child_id,
                transcript: child_transcript.clone(),
            });
            queue.push_back(child_transcript);
        }
    }

    Ok(members)
}

fn family_is_idle(members: &[FamilyMember], cutoff: SystemTime) -> bool {
    members.iter().all(|member| {
        fs::metadata(&member.transcript)
            .and_then(|metadata| metadata.modified())
            .is_ok_and(|modified| modified <= cutoff)
    })
}

fn archive_family(
    sessions_dir: &Path,
    parent_session_id: &str,
    members: &[FamilyMember],
) -> std::result::Result<FamilyStats, FamilyArchiveError> {
    archive_index::ensure_archive_subdirs(sessions_dir)?;

    // Hold every member's writer lock for the whole operation: a session that
    // starts writing mid-archive would otherwise append records that the commit
    // drops when it removes the original transcript.
    let mut _locks = Vec::with_capacity(members.len());
    for member in members {
        match TranscriptWriterLock::try_acquire(&member.transcript) {
            Ok(Some(lock)) => _locks.push(lock),
            Ok(None) => return Err(FamilyArchiveError::Locked),
            Err(error) => {
                return Err(FamilyArchiveError::Io(error.context(format!(
                    "failed to lock transcript {}",
                    member.transcript.display()
                ))));
            }
        }
    }

    let parent_transcript = members
        .first()
        .ok_or_else(|| anyhow!("session family has no parent transcript"))?
        .transcript
        .clone();
    let parent_summary = summarize_transcript(&parent_transcript, parent_session_id)?;

    let mut staged = Vec::with_capacity(members.len());
    let mut staged_artifacts = Vec::new();
    let mut original_bytes = 0_u64;
    let mut archived_bytes = 0_u64;

    let outcome = (|| -> Result<()> {
        for member in members {
            let archive_path =
                family_transcript_archive(sessions_dir, parent_session_id, &member.session_id);
            let staging_path = temporary_path(&archive_path);
            let (member_original, member_archived) =
                compress_transcript(&member.transcript, &staging_path).with_context(|| {
                    format!("failed to archive {}", member.transcript.display())
                })?;
            original_bytes = original_bytes.saturating_add(member_original);
            archived_bytes = archived_bytes.saturating_add(member_archived);
            staged.push((staging_path, archive_path));

            let artifacts_dir = sessions_dir.join("artifacts").join(&member.session_id);
            if artifacts_dir.is_dir() {
                let archive_path = archive_index::artifacts_path(sessions_dir, &member.session_id);
                let staging_path = temporary_path(&archive_path);
                archive_artifacts(&artifacts_dir, &staging_path).with_context(|| {
                    format!("failed to archive artifacts of {}", member.session_id)
                })?;
                staged_artifacts.push((staging_path, archive_path));
            }
        }

        // Commit only after every member compressed successfully.
        for (staging_path, archive_path) in &staged {
            fs::rename(staging_path, archive_path).with_context(|| {
                format!(
                    "failed to publish archived transcript {}",
                    archive_path.display()
                )
            })?;
        }
        for (staging_path, archive_path) in &staged_artifacts {
            fs::rename(staging_path, archive_path).with_context(|| {
                format!(
                    "failed to publish archived artifacts {}",
                    archive_path.display()
                )
            })?;
        }
        Ok(())
    })();

    if let Err(error) = outcome {
        for (staging_path, _) in staged.iter().chain(staged_artifacts.iter()) {
            if let Err(cleanup) = fs::remove_file(staging_path)
                && cleanup.kind() != std::io::ErrorKind::NotFound
            {
                warn!(path = %staging_path.display(), error = %cleanup, "failed to remove staging file");
            }
        }
        return Err(FamilyArchiveError::Io(error));
    }

    // The archived copies are durable now. A failure to drop an original leaves a
    // duplicate that the next pass removes (the live transcript wins), so these
    // are reported rather than turned into a family failure.
    for member in members {
        if let Err(error) = fs::remove_file(&member.transcript)
            && error.kind() != std::io::ErrorKind::NotFound
        {
            warn!(path = %member.transcript.display(), error = %error, "failed to remove archived transcript");
        }
        let artifacts_dir = sessions_dir.join("artifacts").join(&member.session_id);
        if artifacts_dir.is_dir()
            && let Err(error) = fs::remove_dir_all(&artifacts_dir)
        {
            warn!(path = %artifacts_dir.display(), error = %error, "failed to remove archived artifacts");
        }
    }

    match parent_summary {
        Some(summary) => {
            let row = ArchivedSession {
                original_bytes,
                archived_bytes,
                record_count: summary.record_count,
                first_timestamp_ms: summary.first_timestamp_ms,
                last_timestamp_ms: summary.last_timestamp_ms,
                model: summary.model,
                title: summary.title,
                last_user_summary: summary.last_user_summary,
                last_assistant_summary: summary.last_assistant_summary,
                archived_at_ms: now_ms(),
            };
            archive_index::upsert(sessions_dir, parent_session_id, row)?;
        }
        // A transcript without session content stays unlisted, exactly as the live
        // sidecar index treats it, but still counts as archived for this pass.
        None => archive_index::clear_attempts(sessions_dir, parent_session_id)?,
    }

    Ok(FamilyStats {
        transcripts: members.len(),
        original_bytes,
        archived_bytes,
    })
}

/// Stage every member that still lives only in the archive, walking the parent
/// and its descendants through whichever copy of each transcript exists.
///
/// Returns `Ok(None)` when nothing in the family needs restoring and nothing was
/// written, including the case of a member that is live while its archive copy
/// is still around: the live transcript is authoritative, and the leftover
/// archive file is dropped with the rest of the plan.
fn plan_family_restore(
    sessions_dir: &Path,
    session_id: &str,
    staging: &Path,
) -> Result<Option<RestorePlan>> {
    // A live member is only worth reading for its children while some child is
    // archived at all; otherwise a plain resume walks no transcript.
    let child_archives_exist = !archive_index::archived_child_ids(sessions_dir)?.is_empty();

    let mut plan = RestorePlan {
        members: Vec::new(),
        artifacts: Vec::new(),
        stale_archives: Vec::new(),
        original_bytes: 0,
    };
    let mut seen: BTreeSet<String> = BTreeSet::new();
    let mut queue = VecDeque::from([session_id.to_string()]);
    let mut staging_ready = false;

    while let Some(member_id) = queue.pop_front() {
        if !seen.insert(member_id.clone()) {
            continue;
        }

        let live = family_live_transcript(sessions_dir, session_id, &member_id);
        let archived = family_transcript_archive(sessions_dir, session_id, &member_id);
        let artifacts_archive = archive_index::artifacts_path(sessions_dir, &member_id);

        if !archived.is_file() {
            // Already live, or gone entirely: nothing to land here, but its
            // descendants may still be in the archive.
            if live.is_file() {
                enqueue_live_children(&live, child_archives_exist, &mut queue)?;
            }
            continue;
        }

        if live.is_file() {
            // Restored by an earlier call that stopped before dropping the
            // archive, or left behind by an archive that failed to remove its
            // original: the live transcript is the one that counts.
            plan.stale_archives.push(archived);
            if artifacts_archive.is_file() {
                plan.stale_archives.push(artifacts_archive);
            }
            enqueue_live_children(&live, child_archives_exist, &mut queue)?;
            continue;
        }

        if !staging_ready {
            if staging.exists() {
                fs::remove_dir_all(staging)
                    .with_context(|| format!("failed to clear staging {}", staging.display()))?;
            }
            fs::create_dir_all(staging)
                .with_context(|| format!("failed to create staging {}", staging.display()))?;
            staging_ready = true;
        }

        let staged = staging.join(format!("{member_id}.jsonl"));
        let bytes = decompress_transcript(&archived, &staged).with_context(|| {
            format!(
                "failed to restore archived transcript {}",
                archived.display()
            )
        })?;
        plan.original_bytes = plan.original_bytes.saturating_add(bytes);

        // Parse before anything is committed: a damaged archive must fail here,
        // while every archive file is still in place.
        let records = read_records_allow_partial_tail(&staged).with_context(|| {
            format!("archived transcript for session {member_id} is not readable")
        })?;
        queue.extend(child_session_ids_from_records(&records));

        if artifacts_archive.is_file() {
            let staged_artifacts = staging.join("artifacts").join(&member_id);
            extract_artifacts(&artifacts_archive, &staged_artifacts).with_context(|| {
                format!(
                    "failed to restore archived artifacts {}",
                    artifacts_archive.display()
                )
            })?;
            plan.artifacts.push(RestoreArtifacts {
                target: sessions_dir.join("artifacts").join(&member_id),
                staged: staged_artifacts,
                archived: artifacts_archive,
            });
        }

        plan.members.push(RestoreMember {
            target: live,
            staged,
            archived,
        });
    }

    if plan.members.is_empty() && plan.stale_archives.is_empty() {
        return Ok(None);
    }

    // Check every landing target before a single file moves: a conflict has to
    // leave the whole family in the archive rather than half of it on disk.
    for member in &plan.members {
        ensure!(
            !member.target.exists(),
            "restore target {} already exists",
            member.target.display()
        );
    }
    for artifacts in &plan.artifacts {
        ensure!(
            !artifacts.target.exists() || artifacts.target.is_dir(),
            "archived artifacts target {} is not a directory",
            artifacts.target.display()
        );
    }

    Ok(Some(plan))
}

/// Queue the recorded children of a live member, which are the only descendants
/// an incremental restore can still find in the archive.
fn enqueue_live_children(
    live: &Path,
    child_archives_exist: bool,
    queue: &mut VecDeque<String>,
) -> Result<()> {
    if !child_archives_exist {
        return Ok(());
    }
    let records = read_records_allow_partial_tail(live)
        .with_context(|| format!("failed to read transcript {}", live.display()))?;
    queue.extend(child_session_ids_from_records(&records));
    Ok(())
}

fn commit_restore(
    sessions_dir: &Path,
    session_id: &str,
    plan: RestorePlan,
) -> Result<RestoreStats> {
    let child_dir = child_sessions_dir(sessions_dir);
    fs::create_dir_all(&child_dir)
        .with_context(|| format!("failed to create {}", child_dir.display()))?;

    let artifacts_root = sessions_dir.join("artifacts");

    // Land every member before deleting anything: a failure at any point here
    // must leave all archive files in place so the restore stays retryable.
    for artifacts in &plan.artifacts {
        if artifacts.target.is_dir() {
            // Content-addressed artifacts only ever gain files, so a directory
            // that is already present is authoritative.
            continue;
        }
        fs::create_dir_all(&artifacts_root)
            .with_context(|| format!("failed to create {}", artifacts_root.display()))?;
        fs::rename(&artifacts.staged, &artifacts.target).with_context(|| {
            format!("failed to restore artifacts {}", artifacts.target.display())
        })?;
    }

    for member in &plan.members {
        fs::rename(&member.staged, &member.target)
            .with_context(|| format!("failed to restore transcript {}", member.target.display()))?;
    }

    // The family is in place; only now drop the archive copies, including those
    // of members that were already live.
    for member in &plan.members {
        remove_archive_file(&member.archived);
    }
    for artifacts in &plan.artifacts {
        remove_archive_file(&artifacts.archived);
    }
    for path in &plan.stale_archives {
        remove_archive_file(path);
    }
    archive_index::remove(sessions_dir, session_id)?;

    Ok(RestoreStats {
        transcripts: plan.members.len(),
        original_bytes: plan.original_bytes,
    })
}

fn remove_archive_file(path: &Path) {
    if let Err(error) = fs::remove_file(path)
        && error.kind() != std::io::ErrorKind::NotFound
    {
        warn!(path = %path.display(), error = %error, "failed to remove archive file");
    }
}

/// Drop archive copies whose session is live again: the live transcript is
/// authoritative, and listing must never show the same session twice.
///
/// The writer lock is what an archive pass holds from before it publishes the
/// archive until it has removed the live transcript. Taking it here keeps this
/// cleanup from deleting an archive a concurrent pass is about to finish, which
/// would leave the session with neither copy. A locked session is retried by
/// the next pass.
fn remove_stale_archives(sessions_dir: &Path) -> Result<()> {
    for session_id in archive_index::archived_ids(sessions_dir)? {
        let live = sessions_dir.join(format!("{session_id}.jsonl"));
        if !live.is_file() {
            continue;
        }
        let Some(_lock) = TranscriptWriterLock::try_acquire(&live)? else {
            continue;
        };
        if !live.is_file() {
            continue;
        }
        remove_archive_file(&archive_index::transcript_path(sessions_dir, &session_id));
        remove_archive_file(&archive_index::artifacts_path(sessions_dir, &session_id));
        archive_index::remove(sessions_dir, &session_id)?;
    }

    let child_dir = child_sessions_dir(sessions_dir);
    for child_id in archive_index::archived_child_ids(sessions_dir)? {
        let live = child_dir.join(format!("{child_id}.jsonl"));
        if !live.is_file() {
            continue;
        }
        let Some(_lock) = TranscriptWriterLock::try_acquire(&live)? else {
            continue;
        };
        if !live.is_file() {
            continue;
        }
        remove_archive_file(&archive_index::child_transcript_path(
            sessions_dir,
            &child_id,
        ));
        remove_archive_file(&archive_index::artifacts_path(sessions_dir, &child_id));
    }
    Ok(())
}

fn cleanup_stale_staging(sessions_dir: &Path) {
    let root = staging_root(sessions_dir);
    let Ok(entries) = fs::read_dir(&root) else {
        return;
    };
    for entry in entries.flatten() {
        let Ok(modified) = entry.metadata().and_then(|metadata| metadata.modified()) else {
            continue;
        };
        if SystemTime::now()
            .duration_since(modified)
            .unwrap_or_default()
            < STALE_STAGING_AGE
        {
            continue;
        }
        let path = entry.path();
        if let Err(error) = fs::remove_dir_all(&path) {
            warn!(path = %path.display(), error = %error, "failed to remove stale restore staging");
        }
    }
}

fn staging_root(sessions_dir: &Path) -> PathBuf {
    archive_index::archive_dir(sessions_dir).join(STAGING_DIR)
}

/// Live transcript of a family member: the parent sits in the sessions
/// directory, every descendant below `children/`.
fn family_live_transcript(
    sessions_dir: &Path,
    parent_session_id: &str,
    session_id: &str,
) -> PathBuf {
    if session_id == parent_session_id {
        sessions_dir.join(format!("{session_id}.jsonl"))
    } else {
        child_sessions_dir(sessions_dir).join(format!("{session_id}.jsonl"))
    }
}

/// Children are archived below `archive/children/`, mirroring the live layout.
fn family_transcript_archive(
    sessions_dir: &Path,
    parent_session_id: &str,
    session_id: &str,
) -> PathBuf {
    if session_id == parent_session_id {
        archive_index::transcript_path(sessions_dir, session_id)
    } else {
        archive_index::child_transcript_path(sessions_dir, session_id)
    }
}

fn temporary_path(target: &Path) -> PathBuf {
    let mut name = target.as_os_str().to_os_string();
    name.push(ARCHIVING_TMP_SUFFIX);
    PathBuf::from(name)
}

fn compress_transcript(source: &Path, target: &Path) -> Result<(u64, u64)> {
    let original_bytes = fs::metadata(source)
        .with_context(|| format!("failed to stat transcript {}", source.display()))?
        .len();
    let input = File::open(source)
        .with_context(|| format!("failed to read transcript {}", source.display()))?;
    let output = File::create(target)
        .with_context(|| format!("failed to create archived transcript {}", target.display()))?;

    let mut reader = BufReader::new(input);
    let mut encoder = zstd::stream::write::Encoder::new(BufWriter::new(output), ZSTD_LEVEL)?;
    std::io::copy(&mut reader, &mut encoder)
        .with_context(|| format!("failed to compress transcript {}", source.display()))?;
    let mut writer = encoder.finish()?;
    writer.flush()?;
    let file = writer
        .into_inner()
        .map_err(|error| anyhow!("failed to flush {}: {error}", target.display()))?;
    file.sync_all()?;
    drop(file);

    let archived_bytes = fs::metadata(target)
        .with_context(|| format!("failed to stat archived transcript {}", target.display()))?
        .len();
    Ok((original_bytes, archived_bytes))
}

fn decompress_transcript(source: &Path, target: &Path) -> Result<u64> {
    let input = File::open(source)
        .with_context(|| format!("failed to read archived transcript {}", source.display()))?;
    let mut decoder = zstd::stream::read::Decoder::new(BufReader::new(input))?;
    let output = File::create(target)
        .with_context(|| format!("failed to create restored transcript {}", target.display()))?;
    let mut writer = BufWriter::new(output);
    let bytes = std::io::copy(&mut decoder, &mut writer)?;
    writer.flush()?;
    writer
        .into_inner()
        .map_err(|error| anyhow!("failed to flush {}: {error}", target.display()))?
        .sync_all()?;
    Ok(bytes)
}

fn archive_artifacts(source_dir: &Path, target: &Path) -> Result<()> {
    let output = File::create(target)
        .with_context(|| format!("failed to create archived artifacts {}", target.display()))?;
    let encoder = zstd::stream::write::Encoder::new(BufWriter::new(output), ZSTD_LEVEL)?;
    let mut builder = tar::Builder::new(encoder);
    builder.follow_symlinks(false);
    builder
        .append_dir_all(".", source_dir)
        .with_context(|| format!("failed to pack artifacts {}", source_dir.display()))?;
    let encoder = builder.into_inner()?;
    let mut writer = encoder.finish()?;
    writer.flush()?;
    let file = writer
        .into_inner()
        .map_err(|error| anyhow!("failed to flush {}: {error}", target.display()))?;
    file.sync_all()?;
    Ok(())
}

fn extract_artifacts(source: &Path, target: &Path) -> Result<()> {
    fs::create_dir_all(target)
        .with_context(|| format!("failed to create artifacts staging {}", target.display()))?;
    let input = File::open(source)
        .with_context(|| format!("failed to read archived artifacts {}", source.display()))?;
    let mut archive = tar::Archive::new(zstd::stream::read::Decoder::new(BufReader::new(input))?);
    for entry in archive.entries()? {
        let mut entry = entry?;
        let path = entry.path()?.into_owned();
        ensure!(
            entry.unpack_in(target)?,
            "archived artifact entry escapes its directory: {}",
            path.display()
        );
    }
    Ok(())
}

fn summarize_transcript(path: &Path, session_id: &str) -> Result<Option<SessionSummary>> {
    let file = File::open(path)
        .with_context(|| format!("failed to read transcript {}", path.display()))?;
    summarize_session_reader(BufReader::new(file), session_id.to_string(), path)
}

fn reconcile_row(sessions_dir: &Path, session_id: &str) -> Result<Option<ArchivedSession>> {
    let path = archive_index::transcript_path(sessions_dir, session_id);
    let archived_bytes = fs::metadata(&path)
        .with_context(|| format!("failed to stat archived transcript {}", path.display()))?
        .len();
    let input = File::open(&path)
        .with_context(|| format!("failed to read archived transcript {}", path.display()))?;
    let decoder = zstd::stream::read::Decoder::new(BufReader::new(input))?;
    let Some(summary) =
        summarize_session_reader(BufReader::new(decoder), session_id.to_string(), &path)?
    else {
        return Ok(None);
    };

    Ok(Some(ArchivedSession {
        archived_bytes,
        record_count: summary.record_count,
        first_timestamp_ms: summary.first_timestamp_ms,
        last_timestamp_ms: summary.last_timestamp_ms,
        model: summary.model,
        title: summary.title,
        last_user_summary: summary.last_user_summary,
        last_assistant_summary: summary.last_assistant_summary,
        archived_at_ms: now_ms(),
        ..ArchivedSession::default()
    }))
}

#[derive(Deserialize)]
struct ChildSessionLine {
    kind: String,
    #[serde(default)]
    child_session_id: Option<String>,
}

fn child_session_ids(transcript: &Path) -> Result<Vec<String>> {
    let file = File::open(transcript)
        .with_context(|| format!("failed to read transcript {}", transcript.display()))?;
    let mut ids = Vec::new();
    for line in BufReader::new(file).lines() {
        let line =
            line.with_context(|| format!("failed to read transcript {}", transcript.display()))?;
        if !line.contains("\"subagent_started\"") {
            continue;
        }
        let parsed: ChildSessionLine = serde_json::from_str(&line).with_context(|| {
            format!(
                "failed to parse subagent record in {}",
                transcript.display()
            )
        })?;
        if parsed.kind != "subagent_started" {
            continue;
        }
        if let Some(child_id) = parsed.child_session_id
            && !ids.contains(&child_id)
        {
            ids.push(child_id);
        }
    }
    Ok(ids)
}

fn child_session_ids_from_records(records: &[TranscriptRecord]) -> Vec<String> {
    let mut ids = Vec::new();
    for record in records {
        if let TranscriptEvent::SubagentStarted {
            child_session_id, ..
        } = &record.event
            && !ids.contains(child_session_id)
        {
            ids.push(child_session_id.clone());
        }
    }
    ids
}

fn now_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis()
}

#[cfg(test)]
mod tests {
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
        let (child_id, child_path) =
            write_session(&child_sessions_dir(sessions_dir), "child-model");
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

        let error = restore_session_family_for_resume_with(
            dir.path(),
            &parent_id,
            3,
            Duration::from_millis(1),
        )
        .expect_err("a held family lock is reported instead of waiting forever");

        assert!(
            format!("{error:#}").contains(&parent_id),
            "error: {error:#}"
        );
        assert!(!parent_path.exists());
        assert!(archive_index::transcript_path(dir.path(), &parent_id).is_file());
    }
}
