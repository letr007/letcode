use std::collections::HashMap;
use std::io::{BufRead, Write};
use std::path::{Path, PathBuf};

use super::Span;

const MAX_BYTES: u64 = 32 << 20;

pub(crate) const KINDS: [&str; 10] = [
    "email",
    "phone",
    "account",
    "payment_card",
    "ip_address",
    "url",
    "common_secret",
    "private_address",
    "private_date",
    "private_person",
];

pub(crate) struct DiskCache {
    path: PathBuf,
    file: Option<std::fs::File>,
}

impl DiskCache {
    pub(crate) fn open(dir: &Path) -> Self {
        Self {
            path: dir.join("cache").join("redact-spans.tsv"),
            file: None,
        }
    }

    /// Drops the file when it outgrows its budget, so one long session cannot
    /// pin an unbounded cache on disk.
    pub(crate) fn load(&mut self) -> HashMap<[u8; 32], Vec<Span>> {
        let Ok(metadata) = std::fs::metadata(&self.path) else {
            return HashMap::new();
        };
        if metadata.len() > MAX_BYTES {
            let _ = std::fs::remove_file(&self.path);
            return HashMap::new();
        }
        let Ok(file) = std::fs::File::open(&self.path) else {
            return HashMap::new();
        };
        let mut entries = HashMap::new();
        for line in std::io::BufReader::new(file).lines().map_while(Result::ok) {
            if let Some((key, spans)) = parse(&line) {
                entries.insert(key, spans);
            }
        }
        entries
    }

    pub(crate) fn append(&mut self, key: &[u8; 32], spans: &[Span]) {
        if self.file.is_none() {
            if let Some(parent) = self.path.parent() {
                let _ = std::fs::create_dir_all(parent);
            }
            self.file = std::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(&self.path)
                .ok();
        }
        let Some(file) = self.file.as_mut() else {
            return;
        };
        let line = format!("{}\t{}\n", hex(key), encode(spans));
        let _ = file.write_all(line.as_bytes());
    }
}

fn encode(spans: &[Span]) -> String {
    spans
        .iter()
        .map(|span| {
            format!(
                "{}:{}:{}:{:.4}",
                span.start, span.end, span.kind, span.score
            )
        })
        .collect::<Vec<_>>()
        .join(",")
}

fn parse(line: &str) -> Option<([u8; 32], Vec<Span>)> {
    let (key, body) = line.split_once('\t')?;
    let key = unhex(key)?;
    let mut spans = Vec::new();
    for item in body.split(',').filter(|item| !item.is_empty()) {
        let mut parts = item.splitn(4, ':');
        let start = parts.next()?.parse().ok()?;
        let end = parts.next()?.parse().ok()?;
        let name = parts.next()?;
        let kind = KINDS.iter().find(|kind| **kind == name)?;
        let score = parts.next()?.parse().ok()?;
        spans.push(Span {
            start,
            end,
            kind,
            score,
        });
    }
    Some((key, spans))
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}

fn unhex(text: &str) -> Option<[u8; 32]> {
    if text.len() != 64 {
        return None;
    }
    let mut out = [0u8; 32];
    for (index, chunk) in text.as_bytes().chunks(2).enumerate() {
        out[index] = u8::from_str_radix(std::str::from_utf8(chunk).ok()?, 16).ok()?;
    }
    Some(out)
}
