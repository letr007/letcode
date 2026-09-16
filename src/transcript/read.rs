//! Shapes for reading a session journal.
//!
//! The journal file is the only source of transcript records: nothing here keeps
//! decoded records, and every shape reads the file it is asked about. Call sites
//! pick the narrowest shape that answers their question, so a reader does not
//! decode the whole journal to answer a question about part of it.

use std::fs;
use std::path::Path;

use anyhow::{Context, Result};

use super::journal::{ParsedJournalLine, parse_journal_line, transaction_fields};
use super::{TranscriptRecord, read_records};

/// Whether any record in the journal satisfies `predicate`, stopping at the
/// first one that does.
///
/// Records are released in journal order and the answer is decided from the
/// records read so far, so the lines after a match are neither parsed nor
/// validated. When nothing matches, the answer comes from [`read_records`]: only
/// a full read settles that a journal has no matching record, and it is also
/// what reports a journal that does not validate.
///
/// A journal that batches records into a transaction is read whole as well. Its
/// records are released at the commit line, so answering from one of them before
/// that would decide on a record the full read may never release.
pub(crate) fn any_record_where(
    path: impl AsRef<Path>,
    mut predicate: impl FnMut(&TranscriptRecord) -> bool,
) -> Result<bool> {
    let path = path.as_ref();
    let content = fs::read_to_string(path)
        .with_context(|| format!("failed to read transcript {}", path.display()))?;

    for (index, line) in content.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        let parsed = parse_journal_line(line).with_context(|| {
            format!(
                "failed to parse line {} from transcript {}",
                index + 1,
                path.display()
            )
        })?;
        let entry = match parsed {
            ParsedJournalLine::Record(entry) => entry,
            ParsedJournalLine::Commit(_) => break,
        };
        if transaction_fields(&entry.envelope)?.is_some() {
            break;
        }
        if predicate(&entry.record) {
            return Ok(true);
        }
    }

    read_records(path).map(|records| records.iter().any(predicate))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use std::path::PathBuf;
    use std::time::{SystemTime, UNIX_EPOCH};

    use crate::transcript::{TranscriptEvent, TranscriptRecorder};

    fn temp_dir() -> PathBuf {
        std::env::temp_dir().join(format!(
            "letcode-transcript-read-test-{}",
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .expect("system time is valid")
                .as_nanos()
        ))
    }

    /// A journal whose matching record is not its last record, so the scan can
    /// answer before the end.
    fn journal_with_two_prompts() -> (PathBuf, PathBuf) {
        let dir = temp_dir();
        let mut recorder = TranscriptRecorder::create(&dir).expect("create transcript");
        recorder
            .record_session_started("test/model")
            .expect("record session start");
        recorder
            .record_user_message("first")
            .expect("record prompt");
        recorder
            .record_user_message("second")
            .expect("record prompt");
        let path = recorder.path().to_path_buf();
        drop(recorder);
        (dir, path)
    }

    fn is_user_message(record: &TranscriptRecord) -> bool {
        matches!(&record.event, TranscriptEvent::UserMessage { .. })
    }

    #[test]
    fn answers_from_a_record_before_the_end_of_the_journal() {
        let (dir, path) = journal_with_two_prompts();
        let total = read_records(&path).expect("read the journal").len();
        let mut seen = 0usize;

        let matched = any_record_where(&path, |record| {
            seen += 1;
            is_user_message(record)
        })
        .expect("scan the journal");

        assert!(matched);
        assert!(
            seen < total,
            "stopped early: read {seen} of {total} records"
        );
        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn falls_back_to_the_full_read_when_nothing_matches() {
        let (dir, path) = journal_with_two_prompts();
        assert!(!any_record_where(&path, |_| false).expect("scan the journal"));
        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn reports_a_missing_journal_like_the_full_read() {
        let dir = temp_dir();
        let path = dir.join("absent.jsonl");
        assert!(any_record_where(&path, |_| true).is_err());
        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn a_match_before_a_torn_tail_answers_and_a_missing_match_still_reports_it() {
        let (dir, path) = journal_with_two_prompts();
        let mut file = fs::OpenOptions::new()
            .append(true)
            .open(&path)
            .expect("open the journal");
        file.write_all(br#"{"schema_version":2,"event_i"#)
            .expect("write a torn record");
        drop(file);

        assert!(
            any_record_where(&path, is_user_message).expect("scan the journal"),
            "the answer is reached before the torn tail"
        );
        assert!(
            any_record_where(&path, |_| false).is_err(),
            "a full read reports the torn tail"
        );
        let _ = fs::remove_dir_all(dir);
    }
}
