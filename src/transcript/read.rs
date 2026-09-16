//! Shapes for reading a session journal.
//!
//! The journal file is the only source of transcript records: nothing here keeps
//! decoded records, and every shape reads the file it is asked about. Call sites
//! pick the narrowest shape that answers their question, so a reader does not
//! decode the whole journal to answer a question about part of it.

use std::fs;
use std::io::{BufRead, BufReader};
use std::path::Path;

use anyhow::{Context, Result};

use super::journal::{ParsedJournalLine, parse_journal_line, transaction_fields};
use super::{TranscriptRecord, read_records};

/// Whether any record in the journal satisfies `predicate`, stopping at the
/// first one that does.
///
/// The journal is read line by line and the scan stops at the first match, so
/// neither the rest of the file nor the records in it are touched. When nothing
/// matches, the answer comes from [`read_records`]: only a full read settles
/// that a journal has no matching record, and it is also what reports a journal
/// that does not validate.
///
/// Two kinds of line end the scan without an answer, because neither can be
/// judged on its own:
///
/// - A record that belongs to a transaction. Its records are released at the
///   commit line, so answering from one of them would decide on a record the
///   full read may never release.
/// - A commit. Reaching one means a transaction came first.
pub(crate) fn any_record_where(
    path: impl AsRef<Path>,
    mut predicate: impl FnMut(&TranscriptRecord) -> bool,
) -> Result<bool> {
    let path = path.as_ref();
    let file = fs::File::open(path)
        .with_context(|| format!("failed to read transcript {}", path.display()))?;
    let mut reader = BufReader::new(file);
    let mut line = String::new();
    let mut number = 0usize;

    loop {
        line.clear();
        if reader
            .read_line(&mut line)
            .with_context(|| format!("failed to read transcript {}", path.display()))?
            == 0
        {
            break;
        }
        number += 1;
        let record_line = line.trim_end_matches(['\n', '\r']);
        if record_line.trim().is_empty() {
            continue;
        }
        let parsed = parse_journal_line(record_line).with_context(|| {
            format!(
                "failed to parse line {} from transcript {}",
                number,
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

    use crate::transcript::{
        TranscriptEvent, TranscriptRecorder, read_records_allow_partial_tail,
        record_is_session_content, record_is_session_title, record_is_user_message,
    };

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

    /// Measurement harness, not part of the suite: it reads journals named in the
    /// environment, compares both reads for agreement, and reports their cost.
    #[test]
    #[ignore = "measurement harness: set LETCODE_BENCH_JOURNALS to colon-separated paths"]
    fn ab_measure() {
        use std::time::Instant;

        let list = std::env::var("LETCODE_BENCH_JOURNALS").expect("LETCODE_BENCH_JOURNALS");
        let subjects: Vec<(&str, Box<dyn Fn(&TranscriptRecord) -> bool>)> = vec![
            ("session content", Box::new(record_is_session_content)),
            (
                "prompt or title",
                Box::new(|record: &TranscriptRecord| {
                    record_is_user_message(record) || record_is_session_title(record)
                }),
            ),
        ];
        let median = |mut times: Vec<std::time::Duration>| {
            times.sort();
            times[times.len() / 2].as_secs_f64() * 1000.0
        };

        for raw in list.split(':') {
            let path = PathBuf::from(raw);
            if !path.is_file() {
                println!("skip {raw}: not a file");
                continue;
            }
            let size = fs::metadata(&path).expect("stat the journal").len();
            let warm = read_records(&path).expect("warm read");
            println!(
                "\n### {} — {:.1} MB, {} records",
                path.file_name().unwrap_or_default().to_string_lossy(),
                size as f64 / 1048576.0,
                warm.len()
            );
            drop(warm);

            for (name, predicate) in &subjects {
                let mut full = Vec::new();
                let mut shaped = Vec::new();
                let mut answers = (false, false);
                for round in 0..3 {
                    for shaped_first in [round % 2 == 0, round % 2 != 0] {
                        if shaped_first {
                            let start = Instant::now();
                            answers.1 = any_record_where(&path, |record| predicate(record))
                                .expect("shaped");
                            shaped.push(start.elapsed());
                        } else {
                            let start = Instant::now();
                            answers.0 = read_records(&path).expect("full").iter().any(predicate);
                            full.push(start.elapsed());
                        }
                    }
                }
                assert_eq!(answers.0, answers.1, "{name}: the two reads disagree");
                let (full, shaped) = (median(full), median(shaped));
                println!(
                    "  {name:<16} full {full:8.1} ms   shaped {shaped:8.2} ms   {:>8.0}x",
                    full / shaped.max(f64::MIN_POSITIVE)
                );
            }

            // What the child view poll no longer pays on each tick: the full
            // parent read against the two stats that replaced it.
            let (mut full, mut stats) = (Vec::new(), Vec::new());
            for _ in 0..3 {
                let start = Instant::now();
                let _ = read_records(&path).expect("full read");
                full.push(start.elapsed());
                let start = Instant::now();
                for _ in 0..2 {
                    let metadata = fs::metadata(&path).expect("stat the journal");
                    let _ = (metadata.len(), metadata.modified());
                }
                stats.push(start.elapsed());
            }
            let (full, stats) = (median(full), median(stats));
            println!(
                "  {:<16} full {full:8.1} ms   stats  {stats:8.3} ms   {:>8.0}x",
                "poll gate",
                full / stats.max(f64::MIN_POSITIVE)
            );
        }
    }

    /// Measurement harness for the project memory pass, which used to read every
    /// source it tracks on each idle tick. Takes a file listing journal paths, one
    /// per line, in LETCODE_BENCH_SOURCES.
    #[test]
    #[ignore = "measurement harness: set LETCODE_BENCH_SOURCES to a list of journal paths"]
    fn worker_tick_measure() {
        use std::time::Instant;

        let list = std::env::var("LETCODE_BENCH_SOURCES").expect("LETCODE_BENCH_SOURCES");
        let paths: Vec<PathBuf> = fs::read_to_string(&list)
            .expect("read the source list")
            .lines()
            .map(str::trim)
            .filter(|line| !line.is_empty())
            .map(PathBuf::from)
            .filter(|path| path.is_file())
            .collect();

        let start = Instant::now();
        let mut records = 0usize;
        let mut bytes = 0u64;
        for path in &paths {
            bytes += fs::metadata(path).expect("stat the journal").len();
            records += read_records_allow_partial_tail(path)
                .expect("read the source")
                .len();
        }
        let full = start.elapsed();

        let start = Instant::now();
        for path in &paths {
            let _ = path.is_file();
        }
        let gated = start.elapsed();

        println!(
            "\n### project memory pass — {} sources, {:.1} MB, {records} records",
            paths.len(),
            bytes as f64 / 1048576.0
        );
        println!("  full read  {full:?}");
        println!(
            "  gate only  {gated:?}   {:>8.0}x",
            full.as_secs_f64() / gated.as_secs_f64().max(f64::MIN_POSITIVE)
        );
    }
}
