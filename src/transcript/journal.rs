use anyhow::{Context, Result, anyhow, ensure};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::fs;
use std::io::{self, Write};
use std::path::Path;

use super::model::{TranscriptEvent, TranscriptFileFingerprint, TranscriptRecord};

pub const JOURNAL_SCHEMA_VERSION: u32 = 1;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum JournalScope {
    Global,
    Branch,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JournalRecordV1 {
    pub(crate) schema_version: u32,
    pub(crate) event_id: String,
    pub(crate) scope: JournalScope,
    pub(crate) base_revision: u64,
    pub(crate) resulting_revision: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) transaction_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) transaction_index: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) transaction_count: Option<usize>,
    #[serde(flatten)]
    pub(crate) record: TranscriptRecord,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JournalTransactionCommitV1 {
    pub(crate) schema_version: u32,
    pub(crate) journal_entry: String,
    pub(crate) transaction_id: String,
    pub(crate) transaction_count: usize,
    pub(crate) base_revision: u64,
    pub(crate) resulting_revision: u64,
    pub(crate) payload_length: usize,
    pub(crate) payload_digest: String,
}

pub(crate) const JOURNAL_TRANSACTION_COMMIT: &str = "transaction_commit";

pub trait JournalSink: Send {
    fn write_all(&mut self, bytes: &[u8]) -> io::Result<()>;
    fn flush(&mut self) -> io::Result<()>;
    fn sync_data(&mut self) -> io::Result<()>;
}

pub struct FileJournalSink(pub(crate) std::fs::File);

impl JournalSink for FileJournalSink {
    fn write_all(&mut self, bytes: &[u8]) -> io::Result<()> {
        self.0.write_all(bytes)
    }

    fn flush(&mut self) -> io::Result<()> {
        self.0.flush()
    }

    fn sync_data(&mut self) -> io::Result<()> {
        self.0.sync_data()
    }
}

/// Logical checkpoint payloads own their frozen `schema_version` field. Keep
/// the existing journal field for every other event, but use a distinct outer
/// name for this one flattened payload so JSON never contains duplicate keys.
pub fn serialize_journal_record(envelope: &JournalRecordV1) -> Result<Vec<u8>> {
    if !matches!(envelope.record.event, TranscriptEvent::LogicalCheckpoint(_)) {
        return Ok(serde_json::to_vec(envelope)?);
    }
    let mut value = serde_json::to_value(&envelope.record)?;
    let object = value
        .as_object_mut()
        .ok_or_else(|| anyhow!("serialized transcript record is not an object"))?;
    object.insert(
        "journal_schema_version".into(),
        Value::from(envelope.schema_version),
    );
    object.insert("event_id".into(), Value::from(envelope.event_id.clone()));
    object.insert("scope".into(), serde_json::to_value(envelope.scope)?);
    object.insert("base_revision".into(), Value::from(envelope.base_revision));
    object.insert(
        "resulting_revision".into(),
        Value::from(envelope.resulting_revision),
    );
    if let Some(value) = &envelope.transaction_id {
        object.insert("transaction_id".into(), Value::from(value.clone()));
    }
    if let Some(value) = envelope.transaction_index {
        object.insert("transaction_index".into(), Value::from(value));
    }
    if let Some(value) = envelope.transaction_count {
        object.insert("transaction_count".into(), Value::from(value));
    }
    Ok(serde_json::to_vec(&value)?)
}

pub(crate) fn journal_payload_digest(bytes: &[u8]) -> String {
    // A deterministic corruption guard, not a cryptographic integrity mechanism.
    let mut hash = 0xcbf29ce484222325u64;
    for byte in bytes {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x100000001b3);
    }
    format!("{hash:016x}")
}

#[allow(dead_code)]
pub fn read_records(path: impl AsRef<Path>) -> Result<Vec<TranscriptRecord>> {
    read_records_inner(path, false)
}

pub fn read_records_with_fingerprint(
    path: impl AsRef<Path>,
) -> Result<(Vec<TranscriptRecord>, TranscriptFileFingerprint)> {
    let path = path.as_ref();
    let content = fs::read_to_string(path)
        .with_context(|| format!("failed to read transcript {}", path.display()))?;
    let records = parse_records_content(path, &content, false)?;
    Ok((records, transcript_file_fingerprint(&content)))
}

pub fn transcript_file_fingerprint(content: &str) -> TranscriptFileFingerprint {
    TranscriptFileFingerprint {
        content_len: content.len(),
        content_digest: journal_payload_digest(content.as_bytes()),
    }
}

pub fn content_tail_is_uncommitted_transaction(path: &Path, content: &str) -> Result<bool> {
    Ok(scan_transcript_content(path, content)?.has_uncommitted_transaction_tail)
}

pub fn read_records_allow_partial_tail(path: impl AsRef<Path>) -> Result<Vec<TranscriptRecord>> {
    read_records_inner(path, true)
}

pub fn read_records_inner(
    path: impl AsRef<Path>,
    allow_partial_tail: bool,
) -> Result<Vec<TranscriptRecord>> {
    let path = path.as_ref();
    let content = fs::read_to_string(path)
        .with_context(|| format!("failed to read transcript {}", path.display()))?;
    parse_records_content(path, &content, allow_partial_tail)
}

pub fn parse_records_content(
    path: &Path,
    content: &str,
    allow_partial_tail: bool,
) -> Result<Vec<TranscriptRecord>> {
    let has_complete_tail = content.ends_with('\n');
    let mut last_non_empty_line = None;
    for (index, line) in content.lines().enumerate() {
        if !line.trim().is_empty() {
            last_non_empty_line = Some(index);
        }
    }

    let mut records = Vec::new();
    let mut pending_transaction: Option<PendingTransaction> = None;
    for (index, line) in content.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }

        match parse_journal_line(line) {
            Ok(ParsedJournalLine::Record(entry)) => {
                let transaction = transaction_fields(&entry.v1)?;
                match transaction {
                    Some((transaction_id, transaction_index, transaction_count)) => {
                        ensure!(
                            transaction_count > 0,
                            "transcript transaction count must be positive"
                        );
                        let pending =
                            pending_transaction.get_or_insert_with(|| PendingTransaction {
                                transaction_id: transaction_id.clone(),
                                transaction_count,
                                base_revision: entry.v1.as_ref().unwrap().base_revision,
                                payload: Vec::new(),
                                entries: Vec::new(),
                            });
                        ensure!(
                            pending.transaction_id == transaction_id,
                            "transcript transaction is interrupted by a different transaction"
                        );
                        ensure!(
                            pending.transaction_count == transaction_count,
                            "transcript transaction count changes mid-transaction"
                        );
                        ensure!(
                            transaction_index == pending.entries.len(),
                            "transcript transaction records are not contiguous"
                        );
                        ensure!(
                            transaction_index < transaction_count,
                            "transcript transaction index exceeds its count"
                        );
                        pending
                            .payload
                            .extend(serialize_journal_record(entry.v1.as_ref().unwrap())?);
                        pending.payload.push(b'\n');
                        pending.entries.push(entry);
                    }
                    None => {
                        ensure!(
                            pending_transaction.is_none(),
                            "transcript transaction is missing its commit marker before another record"
                        );
                        records.push(entry);
                    }
                }
            }
            Ok(ParsedJournalLine::Commit(commit)) => {
                let pending = pending_transaction
                    .take()
                    .ok_or_else(|| anyhow!("transcript transaction commit has no records"))?;
                ensure!(
                    commit.schema_version == JOURNAL_SCHEMA_VERSION,
                    "unsupported transcript journal schema version {}",
                    commit.schema_version
                );
                ensure!(
                    commit.journal_entry == JOURNAL_TRANSACTION_COMMIT,
                    "unknown transcript journal entry '{}'",
                    commit.journal_entry
                );
                ensure!(
                    commit.transaction_id == pending.transaction_id,
                    "transcript transaction commit id does not match records"
                );
                ensure!(
                    commit.transaction_count == pending.transaction_count
                        && pending.entries.len() == pending.transaction_count,
                    "transcript transaction commit count does not match records"
                );
                ensure!(
                    commit.base_revision == pending.base_revision,
                    "transcript transaction commit base revision does not match records"
                );
                let last_payload_revision = pending
                    .entries
                    .last()
                    .and_then(|entry| entry.v1.as_ref())
                    .ok_or_else(|| anyhow!("transcript transaction commit has no payload records"))?
                    .resulting_revision;
                ensure!(
                    commit.resulting_revision == last_payload_revision,
                    "transcript transaction commit resulting revision does not match payload records"
                );
                ensure!(
                    commit.resulting_revision
                        == commit.base_revision + u64::try_from(commit.transaction_count).unwrap(),
                    "transcript transaction commit revision does not match count"
                );
                ensure!(
                    commit.payload_length == pending.payload.len()
                        && commit.payload_digest == journal_payload_digest(&pending.payload),
                    "transcript transaction commit payload does not match records"
                );
                records.extend(pending.entries);
            }
            Err(error)
                if allow_partial_tail
                    && !has_complete_tail
                    && Some(index) == last_non_empty_line =>
            {
                tracing::debug!(
                    transcript = %path.display(),
                    line = index + 1,
                    error = %error,
                    "ignored incomplete transcript tail while reading live transcript"
                );
                break;
            }
            Err(error) => {
                return Err(error).with_context(|| {
                    format!(
                        "failed to parse line {} from transcript {}",
                        index + 1,
                        path.display()
                    )
                });
            }
        }
    }

    // A complete but uncommitted transaction can only be the physical tail.
    // It is deliberately invisible to projections and recovery.
    validate_journal_entries(&records)?;
    Ok(records.into_iter().map(|entry| entry.record).collect())
}

#[cfg(test)]
pub(crate) fn transcript_records_match(
    current: &[TranscriptRecord],
    expected: &[TranscriptRecord],
) -> Result<bool> {
    if current.len() != expected.len() {
        return Ok(false);
    }
    for (current, expected) in current.iter().zip(expected) {
        if serde_json::to_vec(current)? != serde_json::to_vec(expected)? {
            return Ok(false);
        }
    }
    Ok(true)
}

pub(crate) struct TranscriptContentState {
    has_uncommitted_transaction_tail: bool,
}

pub fn scan_transcript_content(path: &Path, content: &str) -> Result<TranscriptContentState> {
    let mut pending_transaction: Option<PendingTransaction> = None;

    for (index, line) in content.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        match parse_journal_line(line).with_context(|| {
            format!(
                "failed to parse line {} from transcript {}",
                index + 1,
                path.display()
            )
        })? {
            ParsedJournalLine::Record(entry) => match transaction_fields(&entry.v1)? {
                Some((transaction_id, transaction_index, transaction_count)) => {
                    ensure!(
                        transaction_count > 0,
                        "transcript transaction count must be positive"
                    );
                    let pending = pending_transaction.get_or_insert_with(|| PendingTransaction {
                        transaction_id: transaction_id.clone(),
                        transaction_count,
                        base_revision: entry.v1.as_ref().unwrap().base_revision,
                        payload: Vec::new(),
                        entries: Vec::new(),
                    });
                    ensure!(
                        pending.transaction_id == transaction_id,
                        "transcript transaction is interrupted by a different transaction"
                    );
                    ensure!(
                        pending.transaction_count == transaction_count,
                        "transcript transaction count changes mid-transaction"
                    );
                    ensure!(
                        transaction_index == pending.entries.len(),
                        "transcript transaction records are not contiguous"
                    );
                    ensure!(
                        transaction_index < transaction_count,
                        "transcript transaction index exceeds its count"
                    );
                    pending
                        .payload
                        .extend(serialize_journal_record(entry.v1.as_ref().unwrap())?);
                    pending.payload.push(b'\n');
                    pending.entries.push(entry);
                }
                None => ensure!(
                    pending_transaction.is_none(),
                    "transcript transaction is missing its commit marker before another record"
                ),
            },
            ParsedJournalLine::Commit(commit) => {
                let pending = pending_transaction
                    .take()
                    .ok_or_else(|| anyhow!("transcript transaction commit has no records"))?;
                ensure!(
                    commit.schema_version == JOURNAL_SCHEMA_VERSION,
                    "unsupported transcript journal schema version {}",
                    commit.schema_version
                );
                ensure!(
                    commit.journal_entry == JOURNAL_TRANSACTION_COMMIT,
                    "unknown transcript journal entry '{}'",
                    commit.journal_entry
                );
                ensure!(
                    commit.transaction_id == pending.transaction_id,
                    "transcript transaction commit id does not match records"
                );
                ensure!(
                    commit.transaction_count == pending.transaction_count
                        && pending.entries.len() == pending.transaction_count,
                    "transcript transaction commit count does not match records"
                );
                ensure!(
                    commit.base_revision == pending.base_revision,
                    "transcript transaction commit base revision does not match records"
                );
                let last_payload_revision = pending
                    .entries
                    .last()
                    .and_then(|entry| entry.v1.as_ref())
                    .ok_or_else(|| anyhow!("transcript transaction commit has no payload records"))?
                    .resulting_revision;
                ensure!(
                    commit.resulting_revision == last_payload_revision,
                    "transcript transaction commit resulting revision does not match payload records"
                );
                ensure!(
                    commit.resulting_revision
                        == commit.base_revision + u64::try_from(commit.transaction_count).unwrap(),
                    "transcript transaction commit revision does not match count"
                );
                ensure!(
                    commit.payload_length == pending.payload.len()
                        && commit.payload_digest == journal_payload_digest(&pending.payload),
                    "transcript transaction commit payload does not match records"
                );
            }
        }
    }

    Ok(TranscriptContentState {
        has_uncommitted_transaction_tail: pending_transaction.is_some(),
    })
}

#[derive(Debug)]
pub(crate) struct JournalEntry {
    pub(crate) record: TranscriptRecord,
    pub(crate) v1: Option<JournalRecordV1>,
}

pub(crate) struct PendingTransaction {
    pub(crate) transaction_id: String,
    pub(crate) transaction_count: usize,
    pub(crate) base_revision: u64,
    payload: Vec<u8>,
    entries: Vec<JournalEntry>,
}

pub enum ParsedJournalLine {
    Record(JournalEntry),
    Commit(JournalTransactionCommitV1),
}

pub fn parse_journal_line(line: &str) -> Result<ParsedJournalLine> {
    if has_top_level_json_field(line, "journal_entry") {
        return Ok(ParsedJournalLine::Commit(serde_json::from_str(line)?));
    }
    if has_top_level_json_field(line, "journal_schema_version")
        || has_top_level_json_field(line, "schema_version")
    {
        let v1 = parse_journal_v1(line)?;
        ensure!(
            v1.schema_version == JOURNAL_SCHEMA_VERSION,
            "unsupported transcript journal schema version {}",
            v1.schema_version
        );
        Ok(ParsedJournalLine::Record(JournalEntry {
            record: v1.record.clone(),
            v1: Some(v1),
        }))
    } else {
        Ok(ParsedJournalLine::Record(JournalEntry {
            record: serde_json::from_str(line)?,
            v1: None,
        }))
    }
}

fn has_top_level_json_field(line: &str, field: &str) -> bool {
    let bytes = line.as_bytes();
    let mut depth = 0usize;
    let mut index = 0usize;
    while index < bytes.len() {
        match bytes[index] {
            b'{' | b'[' => depth += 1,
            b'}' | b']' => depth = depth.saturating_sub(1),
            b'"' => {
                let start = index + 1;
                index += 1;
                while index < bytes.len() {
                    if bytes[index] == b'\\' {
                        index += 2;
                    } else if bytes[index] == b'"' {
                        break;
                    } else {
                        index += 1;
                    }
                }
                if index >= bytes.len() {
                    return false;
                }
                let mut next = index + 1;
                while next < bytes.len() && bytes[next].is_ascii_whitespace() {
                    next += 1;
                }
                if depth == 1
                    && bytes[start..index] == *field.as_bytes()
                    && bytes.get(next) == Some(&b':')
                {
                    return true;
                }
            }
            _ => {}
        }
        index += 1;
    }
    false
}

fn parse_journal_v1(line: &str) -> Result<JournalRecordV1> {
    #[derive(Deserialize)]
    struct JournalMetadata {
        #[serde(rename = "journal_schema_version")]
        schema_version: u32,
        event_id: String,
        scope: JournalScope,
        base_revision: u64,
        resulting_revision: u64,
        #[serde(default)]
        transaction_id: Option<String>,
        #[serde(default)]
        transaction_index: Option<usize>,
        #[serde(default)]
        transaction_count: Option<usize>,
        timestamp_ms: u128,
    }

    let metadata: JournalMetadata = if has_top_level_json_field(line, "journal_schema_version") {
        serde_json::from_str(line)?
    } else {
        #[derive(Deserialize)]
        struct LegacyJournalMetadata {
            schema_version: u32,
            event_id: String,
            scope: JournalScope,
            base_revision: u64,
            resulting_revision: u64,
            #[serde(default)]
            transaction_id: Option<String>,
            #[serde(default)]
            transaction_index: Option<usize>,
            #[serde(default)]
            transaction_count: Option<usize>,
            timestamp_ms: u128,
        }
        let legacy: LegacyJournalMetadata = serde_json::from_str(line)?;
        JournalMetadata {
            schema_version: legacy.schema_version,
            event_id: legacy.event_id,
            scope: legacy.scope,
            base_revision: legacy.base_revision,
            resulting_revision: legacy.resulting_revision,
            transaction_id: legacy.transaction_id,
            transaction_index: legacy.transaction_index,
            transaction_count: legacy.transaction_count,
            timestamp_ms: legacy.timestamp_ms,
        }
    };
    let record: TranscriptRecord = serde_json::from_str(line)?;
    ensure!(
        metadata.timestamp_ms == record.timestamp_ms,
        "transcript v1 timestamp metadata is inconsistent"
    );
    Ok(JournalRecordV1 {
        schema_version: metadata.schema_version,
        event_id: metadata.event_id,
        scope: metadata.scope,
        base_revision: metadata.base_revision,
        resulting_revision: metadata.resulting_revision,
        transaction_id: metadata.transaction_id,
        transaction_index: metadata.transaction_index,
        transaction_count: metadata.transaction_count,
        record,
    })
}

pub fn transaction_fields(v1: &Option<JournalRecordV1>) -> Result<Option<(String, usize, usize)>> {
    let Some(v1) = v1 else { return Ok(None) };
    match (
        &v1.transaction_id,
        v1.transaction_index,
        v1.transaction_count,
    ) {
        (None, None, None) => Ok(None),
        (Some(id), Some(index), Some(count)) => Ok(Some((id.clone(), index, count))),
        _ => Err(anyhow!(
            "transcript transaction fields must be present together"
        )),
    }
}

pub fn validate_journal_entries(entries: &[JournalEntry]) -> Result<()> {
    let mut session_id = None;
    let mut previous_sequence = None;
    let mut previous_revision = None;
    let mut saw_v1 = false;
    let mut event_ids = std::collections::BTreeSet::new();

    for entry in entries {
        if let Some(expected) = &session_id {
            ensure!(
                entry.record.session_id == *expected,
                "transcript contains records from multiple sessions"
            );
        } else {
            session_id = Some(entry.record.session_id.clone());
        }
        if let Some(previous) = previous_sequence {
            ensure!(
                entry.record.sequence > previous,
                "transcript sequence must be strictly increasing"
            );
        }
        previous_sequence = Some(entry.record.sequence);

        match &entry.v1 {
            Some(v1) => {
                saw_v1 = true;
                ensure!(
                    v1.event_id == format!("{}:{}", entry.record.session_id, entry.record.sequence),
                    "transcript v1 event_id does not match record identity"
                );
                ensure!(
                    event_ids.insert(v1.event_id.as_str()),
                    "transcript v1 event_id must be unique"
                );
                ensure!(
                    v1.scope == journal_scope_for(&entry.record),
                    "transcript v1 scope does not match context_branch_id"
                );
                ensure!(
                    v1.resulting_revision == v1.base_revision + 1,
                    "transcript v1 revisions must be consecutive"
                );
                ensure!(
                    v1.resulting_revision == entry.record.sequence,
                    "transcript v1 resulting_revision must equal sequence"
                );
                let expected_base = previous_revision.unwrap_or(0);
                ensure!(
                    v1.base_revision == expected_base,
                    "transcript v1 base_revision is not continuous"
                );
                previous_revision = Some(v1.resulting_revision);
            }
            None => {
                ensure!(!saw_v1, "legacy transcript record cannot follow v1 records");
                previous_revision = Some(entry.record.sequence);
            }
        }
    }
    Ok(())
}

pub fn journal_scope_for(record: &TranscriptRecord) -> JournalScope {
    if record.context_branch_id.is_some() {
        JournalScope::Branch
    } else {
        JournalScope::Global
    }
}
