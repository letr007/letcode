use std::collections::HashMap;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use sha2::{Digest, Sha256};

use super::Model;

pub(crate) fn kind_of(label: &str) -> &'static str {
    match label {
        "email" => "EMAIL",
        "phone" => "PHONE",
        "account" => "ACCOUNT",
        "payment_card" => "CARD",
        "ip_address" => "IP",
        "url" => "URL",
        "common_secret" => "SECRET",
        "private_address" => "ADDRESS",
        "private_date" => "DATE",
        "private_person" => "PERSON",
        _ => "OTHER",
    }
}

pub(crate) struct Masker {
    model: Model,
    key: [u8; 32],
    values: HashMap<String, String>,
    placeholders: HashMap<(String, String), String>,
}

impl Masker {
    pub(crate) fn open(dir: &Path, key_path: &Path) -> Result<Self> {
        Ok(Self {
            model: Model::load(dir)?,
            key: load_or_create_key(key_path)?,
            values: HashMap::new(),
            placeholders: HashMap::new(),
        })
    }

    pub(crate) fn mask(&mut self, text: &str) -> Result<String> {
        let spans = self.model.detect(text)?;
        if spans.is_empty() {
            return Ok(text.to_string());
        }
        let mut out = String::with_capacity(text.len());
        let mut at = 0;
        for span in &spans {
            if span.start < at || span.end > text.len() {
                continue;
            }
            let placeholder = self.placeholder(kind_of(span.kind), &text[span.start..span.end]);
            out.push_str(&text[at..span.start]);
            out.push_str(&placeholder);
            at = span.end;
        }
        out.push_str(&text[at..]);
        Ok(out)
    }

    pub(crate) fn restore(&self, text: &str) -> String {
        restore_values(&self.values, text)
    }

    fn placeholder(&mut self, kind: &str, value: &str) -> String {
        let id = (kind.to_string(), value.to_string());
        if let Some(existing) = self.placeholders.get(&id) {
            return existing.clone();
        }
        let placeholder = format!("{{{{{kind}_{}}}}}", tag(&self.key, kind, value));
        self.placeholders.insert(id, placeholder.clone());
        self.values.insert(placeholder.clone(), value.to_string());
        placeholder
    }
}

pub(crate) fn tag(key: &[u8; 32], kind: &str, value: &str) -> String {
    let mut data = Vec::with_capacity(kind.len() + value.len() + 1);
    data.extend_from_slice(kind.as_bytes());
    data.push(0);
    data.extend_from_slice(value.as_bytes());
    base32(&hmac_sha256(key, &data))[..8].to_string()
}

pub(crate) fn restore_values(values: &HashMap<String, String>, text: &str) -> String {
    if values.is_empty() || !text.contains("{{") {
        return text.to_string();
    }
    let mut out = String::with_capacity(text.len());
    let mut rest = text;
    while let Some(open) = rest.find("{{") {
        out.push_str(&rest[..open]);
        let after = &rest[open + 2..];
        let Some(close) = after.find("}}") else {
            out.push_str(&rest[open..]);
            return out;
        };
        let name = &after[..close];
        match values.get(&format!("{{{{{name}}}}}")) {
            Some(value) => {
                out.push_str(value);
                rest = &after[close + 2..];
            }
            None => {
                out.push_str("{{");
                rest = after;
            }
        }
    }
    out.push_str(rest);
    out
}

fn load_or_create_key(path: &Path) -> Result<[u8; 32]> {
    if let Ok(bytes) = std::fs::read(path) {
        let Ok(key) = <[u8; 32]>::try_from(bytes.as_slice()) else {
            bail!("{} is not a 32 byte key", path.display());
        };
        return Ok(key);
    }
    let mut key = [0u8; 32];
    getrandom::fill(&mut key).context("reading random bytes")?;
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("creating {}", parent.display()))?;
    }
    write_private(path, &key)?;
    Ok(key)
}

fn write_private(path: &Path, bytes: &[u8]) -> Result<()> {
    use std::io::Write;
    let mut options = std::fs::OpenOptions::new();
    options.write(true).create_new(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(0o600);
    }
    options
        .open(path)
        .with_context(|| format!("writing {}", path.display()))?
        .write_all(bytes)?;
    Ok(())
}

pub(crate) fn key_path(config_dir: &Path) -> PathBuf {
    config_dir.join("redact.key")
}

fn hmac_sha256(key: &[u8], data: &[u8]) -> [u8; 32] {
    let mut padded = [0u8; 64];
    if key.len() > 64 {
        padded[..32].copy_from_slice(&Sha256::digest(key));
    } else {
        padded[..key.len()].copy_from_slice(key);
    }
    let (mut inner, mut outer) = ([0u8; 64], [0u8; 64]);
    for i in 0..64 {
        inner[i] = padded[i] ^ 0x36;
        outer[i] = padded[i] ^ 0x5c;
    }
    let mut hasher = Sha256::new();
    hasher.update(inner);
    hasher.update(data);
    let inner = hasher.finalize();
    let mut hasher = Sha256::new();
    hasher.update(outer);
    hasher.update(inner);
    hasher.finalize().into()
}

fn base32(bytes: &[u8]) -> String {
    const ALPHABET: &[u8; 32] = b"abcdefghijklmnopqrstuvwxyz234567";
    let mut out = String::with_capacity(bytes.len() * 8 / 5 + 1);
    let mut buffer = 0u32;
    let mut bits = 0u32;
    for &byte in bytes {
        buffer = (buffer << 8) | u32::from(byte);
        bits += 8;
        while bits >= 5 {
            bits -= 5;
            out.push(ALPHABET[((buffer >> bits) & 31) as usize] as char);
        }
    }
    if bits > 0 {
        out.push(ALPHABET[((buffer << (5 - bits)) & 31) as usize] as char);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn one_value_keeps_one_placeholder_and_comes_back() {
        let key = [7u8; 32];
        assert_eq!(
            tag(&key, "SECRET", "sk-test-123"),
            tag(&key, "SECRET", "sk-test-123")
        );
        assert_ne!(tag(&key, "SECRET", "one"), tag(&key, "SECRET", "two"));

        let name = format!("{{{{SECRET_{}}}}}", tag(&key, "SECRET", "sk-test-123"));
        let values = HashMap::from([(name.clone(), "sk-test-123".to_string())]);
        let text = format!("auth {name}, again {name} and {{{{SECRET_unknown}}}}");
        assert_eq!(
            restore_values(&values, &text),
            "auth sk-test-123, again sk-test-123 and {{SECRET_unknown}}"
        );
    }
}
