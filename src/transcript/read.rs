//! Shapes for reading a session journal.
//!
//! The journal file is the only source of transcript records: no shape keeps
//! decoded records, and each one reads the file it is asked about.

use std::fs;
use std::io::{BufRead, BufReader};
use std::path::Path;

use anyhow::{Context, Result};

use super::journal::{ParsedJournalLine, parse_journal_line, transaction_fields};
use super::{TranscriptRecord, read_records};

/// Whether any record satisfies `predicate`, stopping at the first match.
///
/// A match is decided from the records read so far, so nothing after it is read
/// or validated. A journal with no match is read in full instead: only that
/// settles the answer, and it is also what reports a journal that does not
/// validate. `predicate` sees the records already scanned a second time then,
/// because that read goes over the journal again.
///
/// A transaction ends the scan without an answer: its records are released at
/// the commit line, so deciding on one of them would use a record the full read
/// may never release.
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
#[path = "../../tests/internal/transcript/read/tests.rs"]
mod tests;
