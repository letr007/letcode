mod cache;
mod fetch;
mod forward;
mod kernels;
mod masker;
mod onnx;
mod rules;

use std::path::Path;

use anyhow::Result;
use tokenizers::Tokenizer;

pub(crate) const LABELS: [&str; 21] = [
    "O",
    "B-email",
    "I-email",
    "B-phone",
    "I-phone",
    "B-account",
    "I-account",
    "B-payment_card",
    "I-payment_card",
    "B-ip_address",
    "I-ip_address",
    "B-url",
    "I-url",
    "B-common_secret",
    "I-common_secret",
    "B-private_address",
    "I-private_address",
    "B-private_date",
    "I-private_date",
    "B-private_person",
    "I-private_person",
];

const MAX_POS: usize = 512;
const OVERLAP: usize = 128;
const WINDOW: usize = MAX_POS - 2;

// The model's recommended minimum.
pub(crate) const MIN_SCORE: f32 = 0.8;

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Span {
    pub(crate) start: usize,
    pub(crate) end: usize,
    pub(crate) kind: &'static str,
    pub(crate) score: f32,
}

struct SpanCache {
    spans: std::collections::HashMap<[u8; 32], Vec<Span>>,
    order: std::collections::VecDeque<([u8; 32], usize)>,
    bytes: usize,
    disk: Option<cache::DiskCache>,
}

impl SpanCache {
    fn open(dir: Option<&Path>) -> Self {
        let mut disk = dir.map(cache::DiskCache::open);
        let spans = disk
            .as_mut()
            .map(cache::DiskCache::load)
            .unwrap_or_default();
        let order = spans
            .iter()
            .map(|(key, value)| (*key, value.len()))
            .collect();
        Self {
            spans,
            order,
            bytes: 0,
            disk,
        }
    }

    fn take(&mut self, key: &[u8; 32]) -> Option<Vec<Span>> {
        let spans = self.spans.get(key)?.clone();
        if let Some(index) = self.order.iter().position(|(entry, _)| entry == key)
            && let Some(entry) = self.order.remove(index)
        {
            self.order.push_back(entry);
        }
        Some(spans)
    }

    fn insert(&mut self, key: [u8; 32], spans: Vec<Span>, len: usize) {
        if let Some(disk) = self.disk.as_mut() {
            disk.append(&key, &spans);
        }
        if self.spans.insert(key, spans).is_none() {
            self.order.push_back((key, len));
            self.bytes += len;
        }
        while self.bytes > CACHE_BUDGET {
            let Some((old, len)) = self.order.pop_front() else {
                break;
            };
            if self.spans.remove(&old).is_some() {
                self.bytes -= len;
            }
        }
    }
}

const CACHE_BUDGET: usize = 256 << 20;

pub(crate) struct Model {
    weights: forward::Weights,
    tokenizer: Tokenizer,
    cache: std::sync::Mutex<SpanCache>,
}

impl Model {
    pub(crate) fn load(dir: &Path, cache_dir: Option<&Path>) -> Result<Self> {
        let graph = onnx::read(&dir.join("model_int8.onnx"))?;
        let weights = forward::Weights::load(&graph)?;
        let tokenizer = Tokenizer::from_file(dir.join("tokenizer.json"))
            .map_err(|e| anyhow::anyhow!("reading tokenizer.json: {e}"))?;
        Ok(Self {
            weights,
            tokenizer,
            cache: std::sync::Mutex::new(SpanCache::open(cache_dir)),
        })
    }

    pub(crate) fn detect_tracked(&self, text: &str, cancel: &AtomicBool) -> Result<Vec<Span>> {
        use sha2::Digest;
        let digest: [u8; 32] = sha2::Sha256::digest(text.as_bytes()).into();
        if let Ok(mut cache) = self.cache.lock()
            && let Some(spans) = cache.take(&digest)
        {
            return Ok(spans);
        }
        let mut spans = self.classify(text, cancel)?;
        spans.extend(rules::scan(text));
        let spans = merge(spans);
        if let Ok(mut cache) = self.cache.lock() {
            cache.insert(digest, spans.clone(), text.len());
        }
        Ok(spans)
    }

    fn classify(&self, text: &str, cancel: &AtomicBool) -> Result<Vec<Span>> {
        let encoding = self
            .tokenizer
            .encode(text, false)
            .map_err(|e| anyhow::anyhow!("tokenizing: {e}"))?;
        let ids: Vec<usize> = encoding.get_ids().iter().map(|&id| id as usize).collect();
        let offsets = encoding.get_offsets();
        let n = ids.len();
        if n == 0 {
            return Ok(Vec::new());
        }

        let mut best = vec![-1i32; n];
        let mut label = vec![0usize; n];
        let mut prob = vec![0f32; n];

        let mut start = 0usize;
        loop {
            check_cancel(cancel)?;
            let end = (start + WINDOW).min(n);
            let mut seq = Vec::with_capacity(end - start + 2);
            seq.push(0);
            seq.extend_from_slice(&ids[start..end]);
            seq.push(2);
            let types = vec![0usize; seq.len()];
            let logits = self.weights.logits(&seq, &types)?;
            for i in 0..(end - start) {
                let margin = i.min(end - start - 1 - i) as i32;
                let at = start + i;
                if margin <= best[at] {
                    continue;
                }
                let row = &logits[(i + 1) * forward::CLASSES..(i + 2) * forward::CLASSES];
                let (arg, p) = argmax(row);
                best[at] = margin;
                label[at] = arg;
                prob[at] = p;
            }
            if end >= n {
                break;
            }
            start = end - OVERLAP;
        }

        let mut spans: Vec<Span> = Vec::new();
        let mut current: Option<usize> = None;
        for at in 0..n {
            let (s, e) = offsets[at];
            if s == e {
                continue;
            }
            let tag = LABELS[label[at]];
            if tag == "O" {
                current = None;
                continue;
            }
            let kind = &tag[2..];
            match current {
                Some(i) if tag.starts_with('I') && spans[i].kind == kind => {
                    spans[i].end = e;
                    spans[i].score = spans[i].score.min(prob[at]);
                }
                _ => {
                    spans.push(Span {
                        start: s,
                        end: e,
                        kind,
                        score: prob[at],
                    });
                    current = Some(spans.len() - 1);
                }
            }
        }
        spans.retain(|s| s.score >= MIN_SCORE && s.start < s.end && s.end <= text.len());
        Ok(spans)
    }
}

fn merge(mut spans: Vec<Span>) -> Vec<Span> {
    spans.sort_by(|a, b| a.start.cmp(&b.start).then(b.end.cmp(&a.end)));
    let mut out: Vec<Span> = Vec::with_capacity(spans.len());
    for span in spans {
        if out.last().is_some_and(|last| span.start < last.end) {
            continue;
        }
        out.push(span);
    }
    out
}

fn argmax(row: &[f32]) -> (usize, f32) {
    let mut best = 0usize;
    let mut max = f32::NEG_INFINITY;
    for (i, &v) in row.iter().enumerate() {
        if v > max {
            max = v;
            best = i;
        }
    }
    let mut sum = 0f32;
    for &v in row {
        sum += (v - max).exp();
    }
    (best, 1.0 / sum)
}

use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use crate::model_runtime::{ContentPart, ModelRequestInput};
pub(crate) use masker::Masker;

pub(crate) type RestoreMap = std::collections::HashMap<String, String>;

pub(crate) struct Ready {
    pub(crate) model: Model,
    pub(crate) key: [u8; 32],
}

pub(crate) struct Redactor {
    dir: PathBuf,
    key: PathBuf,
    ready: Mutex<Option<Arc<Ready>>>,
    init: Mutex<()>,
    started: AtomicBool,
}

impl Redactor {
    fn new(config_dir: &Path, preload: bool) -> Arc<Self> {
        let redactor = Arc::new(Self {
            dir: config_dir.join("models").join("astrlink-guard"),
            key: masker::key_path(config_dir),
            ready: Mutex::new(None),
            init: Mutex::new(()),
            started: AtomicBool::new(false),
        });
        if preload {
            redactor.preload();
        }
        redactor
    }

    fn preload(self: &Arc<Self>) {
        if self.started.swap(true, Ordering::SeqCst) {
            return;
        }
        let worker = Arc::clone(self);
        std::thread::spawn(move || {
            if let Err(error) = worker.ensure() {
                tracing::warn!("masking model unavailable: {error}");
            }
        });
    }

    pub(crate) fn masker(&self) -> Result<Masker> {
        self.ensure()?;
        let ready = self
            .ready
            .lock()
            .map_err(|_| anyhow::anyhow!("the masking model lock is poisoned"))?
            .clone()
            .ok_or_else(|| anyhow::anyhow!("the masking model is unavailable"))?;
        Ok(Masker::new(ready))
    }

    fn ready(&self) -> bool {
        self.ready.lock().is_ok_and(|slot| slot.is_some())
    }

    fn ensure(&self) -> Result<()> {
        if self.ready() {
            return Ok(());
        }
        let _guard = self
            .init
            .lock()
            .map_err(|_| anyhow::anyhow!("the masking model lock is poisoned"))?;
        if self.ready() {
            return Ok(());
        }
        let result = std::thread::scope(|scope| {
            scope
                .spawn(|| -> Result<()> {
                    if !fetch::present(&self.dir) {
                        fetch::fetch_missing(&self.dir)?;
                    }
                    let model = Model::load(&self.dir, self.key.parent())?;
                    let key = masker::load_or_create_key(&self.key)?;
                    *self
                        .ready
                        .lock()
                        .map_err(|_| anyhow::anyhow!("the masking model lock is poisoned"))? =
                        Some(Arc::new(Ready { model, key }));
                    Ok(())
                })
                .join()
        });
        match result {
            Ok(Ok(())) => Ok(()),
            Ok(Err(error)) => Err(error),
            Err(_) => anyhow::bail!("the masking model initializer panicked"),
        }
    }
}

/// Each download carries a token so a stale download cannot clear the progress of a newer one.
static PREPARING: Mutex<Option<(u64, u64, u64)>> = Mutex::new(None);
static NEXT_TOKEN: AtomicU64 = AtomicU64::new(1);

pub(crate) fn preparing() -> Option<(u64, u64)> {
    PREPARING
        .lock()
        .ok()
        .and_then(|state| state.map(|(_, done, total)| (done, total)))
}

pub(crate) struct Progress {
    token: u64,
}

impl Progress {
    pub(crate) fn start(total: u64) -> Self {
        let token = NEXT_TOKEN.fetch_add(1, Ordering::SeqCst);
        if let Ok(mut state) = PREPARING.lock() {
            *state = Some((token, 0, total));
        }
        Self { token }
    }

    pub(crate) fn advance(&self, done: u64) {
        if let Ok(mut state) = PREPARING.lock()
            && let Some((token, _, total)) = *state
            && token == self.token
        {
            *state = Some((token, done, total));
        }
    }
}

impl Drop for Progress {
    fn drop(&mut self) {
        if let Ok(mut state) = PREPARING.lock()
            && state.is_some_and(|(token, _, _)| token == self.token)
        {
            *state = None;
        }
    }
}

static ACTIVE: Mutex<Option<(bool, PathBuf, Arc<Redactor>)>> = Mutex::new(None);

/// Applies the switch, including after a configuration reload. A reload that
/// leaves the switch and the directory alone keeps the running instance.
pub(crate) fn install(enabled: bool, config_dir: &Path) {
    let Ok(mut slot) = ACTIVE.lock() else {
        return;
    };
    if let Some((current_enabled, current_dir, redactor)) = slot.as_mut()
        && *current_dir == config_dir
    {
        *current_enabled = enabled;
        if enabled {
            redactor.preload();
        }
        return;
    }
    *slot = Some((
        enabled,
        config_dir.to_path_buf(),
        Redactor::new(config_dir, enabled),
    ));
}

pub(crate) fn active() -> Option<Arc<Redactor>> {
    ACTIVE.lock().ok().and_then(|slot| {
        slot.as_ref()
            .filter(|(enabled, _, _)| *enabled)
            .map(|(_, _, redactor)| Arc::clone(redactor))
    })
}

pub(crate) fn installed() -> Option<Arc<Redactor>> {
    ACTIVE
        .lock()
        .ok()
        .and_then(|slot| slot.as_ref().map(|(_, _, redactor)| Arc::clone(redactor)))
}

pub(crate) fn mask_input(
    redactor: &Redactor,
    input: &mut ModelRequestInput,
    eligible: &std::collections::HashSet<String>,
    cancel: &AtomicBool,
) -> Result<Option<Arc<RestoreMap>>> {
    if eligible.is_empty() {
        return Ok(None);
    }
    anyhow::ensure!(
        input.segment_origins.len() == input.segments.len(),
        "segment origins do not match segments"
    );
    anyhow::ensure!(
        input.message_origins.len() == input.messages.len(),
        "message origins do not match messages"
    );
    let mut masker = redactor.masker()?;
    for (index, segment) in input.segments.iter_mut().enumerate() {
        if !is_eligible(input.segment_origins.get(index), eligible) {
            continue;
        }
        check_cancel(cancel)?;
        segment.text = masker.mask_tracked(&segment.text, cancel)?;
    }
    for (index, message) in input.messages.iter_mut().enumerate() {
        if !is_eligible(input.message_origins.get(index), eligible) {
            continue;
        }
        check_cancel(cancel)?;
        mask_parts(&mut masker, &mut message.content, cancel)?;
    }
    Ok(Some(masker.finish()))
}

fn is_eligible(origin: Option<&String>, eligible: &std::collections::HashSet<String>) -> bool {
    origin.is_some_and(|origin| eligible.contains(origin))
}

fn check_cancel(cancel: &AtomicBool) -> Result<()> {
    if cancel.load(Ordering::Relaxed) {
        anyhow::bail!("redaction cancelled");
    }
    Ok(())
}

fn mask_parts(masker: &mut Masker, parts: &mut [ContentPart], cancel: &AtomicBool) -> Result<()> {
    for part in parts {
        check_cancel(cancel)?;
        match part {
            ContentPart::Text(text) => *text = masker.mask_tracked(text, cancel)?,
            ContentPart::Reasoning { text, .. } => *text = masker.mask_tracked(text, cancel)?,
            ContentPart::ToolCall { arguments, .. } => masker.mask_value(arguments, cancel)?,
            ContentPart::ToolResult { content, .. } => mask_parts(masker, content, cancel)?,
            ContentPart::Image { .. } => {}
        }
    }
    Ok(())
}

pub(crate) fn mask_value(redactor: &Redactor, value: &mut serde_json::Value) -> Result<()> {
    let cancel = AtomicBool::new(false);
    let mut masker = redactor.masker()?;
    masker.mask_value(value, &cancel)
}

pub(crate) fn restore_text(values: &RestoreMap, text: &str) -> String {
    masker::restore_values(values, text)
}

pub(crate) fn restore_value(values: &RestoreMap, value: &mut serde_json::Value) {
    match value {
        serde_json::Value::String(text) => *text = masker::restore_values(values, text),
        serde_json::Value::Array(items) => {
            for item in items {
                restore_value(values, item);
            }
        }
        serde_json::Value::Object(map) => {
            for item in map.values_mut() {
                restore_value(values, item);
            }
        }
        _ => {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model_runtime::{MessageRole, ModelMessage};
    use std::collections::HashSet;

    #[test]
    fn matches_the_reference_spans() {
        let Ok(dir) = std::env::var("LETCODE_PII_MODEL_DIR") else {
            return;
        };
        let model = Model::load(Path::new(&dir), None).expect("the model loads");
        let cases: Vec<Case> =
            serde_json::from_str(include_str!("../../tests/fixtures/pii_guard_spans.json"))
                .expect("the fixture parses");
        let cancel = AtomicBool::new(false);
        for case in cases {
            let spans = model
                .detect_tracked(&case.text, &cancel)
                .expect("detection runs");
            let got: Vec<(&str, &str, f32)> = spans
                .iter()
                .map(|s| (s.kind, &case.text[s.start..s.end], s.score))
                .collect();
            let want: Vec<(&str, &str, f32)> = case
                .spans
                .iter()
                .map(|s| (s.kind.as_str(), s.text.as_str(), s.score))
                .collect();
            assert_eq!(
                got.iter().map(|(k, t, _)| (*k, *t)).collect::<Vec<_>>(),
                want.iter().map(|(k, t, _)| (*k, *t)).collect::<Vec<_>>(),
                "spans for {:?}",
                case.text
            );
            for ((_, _, got), (_, _, want)) in got.iter().zip(want.iter()) {
                assert!(
                    (got - want).abs() < 0.01,
                    "score {got:.4} against {want:.4} for {:?}",
                    case.text
                );
            }
        }
    }

    #[test]
    fn masking_removes_the_values_and_puts_them_back() {
        let Ok(dir) = std::env::var("LETCODE_PII_MODEL_DIR") else {
            return;
        };
        let cache_dir =
            std::env::temp_dir().join(format!("letcode-redact-key-{}", std::process::id()));
        std::fs::create_dir_all(&cache_dir).expect("the temp dir is created");
        let key_path = cache_dir.join("key");
        let _ = std::fs::remove_file(&key_path);
        let ready = Arc::new(Ready {
            model: Model::load(Path::new(&dir), None).expect("the model loads"),
            key: masker::load_or_create_key(&key_path).expect("the key loads"),
        });
        let mut masker = Masker::new(ready);
        let cancel = AtomicBool::new(false);
        let text = "联系张伟 13800138000，邮箱 zhang@example.com，key sk-test-secret";
        let masked = masker.mask_tracked(text, &cancel).expect("masking runs");
        assert_ne!(masked, text);
        let values = masker.finish();
        assert_eq!(restore_text(&values, &masked), text);
        let _ = std::fs::remove_dir_all(&cache_dir);
    }

    #[test]
    fn masks_only_eligible_messages() {
        let Ok(dir) = std::env::var("LETCODE_PII_MODEL_DIR") else {
            return;
        };
        let cache_dir =
            std::env::temp_dir().join(format!("letcode-redact-input-{}", std::process::id()));
        std::fs::create_dir_all(&cache_dir).expect("the temp dir is created");
        let key_path = cache_dir.join("key");
        let _ = std::fs::remove_file(&key_path);
        let ready = Arc::new(Ready {
            model: Model::load(Path::new(&dir), None).expect("the model loads"),
            key: masker::load_or_create_key(&key_path).expect("the key loads"),
        });
        let redactor = Redactor {
            dir: PathBuf::from(&dir),
            key: key_path,
            ready: Mutex::new(Some(ready)),
            init: Mutex::new(()),
            started: AtomicBool::new(false),
        };
        let history = "联系张伟 13800138000";
        let current = "邮箱 zhang@example.com";
        let mut input = ModelRequestInput::new(
            "test",
            vec![
                ModelMessage::text(MessageRole::User, history),
                ModelMessage::text(MessageRole::Assistant, "收到"),
                ModelMessage::text(MessageRole::User, current),
            ],
        );
        input.message_origins = vec!["old".into(), "assistant".into(), "current".into()];
        let eligible = HashSet::from(["current".to_string()]);
        let cancel = AtomicBool::new(false);
        let values = mask_input(&redactor, &mut input, &eligible, &cancel)
            .expect("masking runs")
            .expect("current message is masked");
        let texts: Vec<&str> = input
            .messages
            .iter()
            .map(|message| match message.content.first() {
                Some(ContentPart::Text(text)) => text.as_str(),
                _ => panic!("text message"),
            })
            .collect();
        assert_eq!(texts[0], history);
        assert_eq!(texts[1], "收到");
        assert_ne!(texts[2], current);
        assert_eq!(restore_text(&values, texts[2]), current);
        let _ = std::fs::remove_dir_all(&cache_dir);
    }

    #[derive(serde::Deserialize)]
    struct Case {
        text: String,
        spans: Vec<Expected>,
    }

    #[derive(serde::Deserialize)]
    struct Expected {
        kind: String,
        text: String,
        score: f32,
    }
}
