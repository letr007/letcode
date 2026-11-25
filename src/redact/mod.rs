mod fetch;
mod forward;
mod kernels;
mod map;
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

pub(crate) struct Model {
    weights: forward::Weights,
    tokenizer: Tokenizer,
}

impl Model {
    pub(crate) fn load(dir: &Path) -> Result<Self> {
        let graph = onnx::read(&dir.join("model_int8.onnx"))?;
        let weights = forward::Weights::load(&graph)?;
        let tokenizer = Tokenizer::from_file(dir.join("tokenizer.json"))
            .map_err(|e| anyhow::anyhow!("reading tokenizer.json: {e}"))?;
        Ok(Self { weights, tokenizer })
    }

    pub(crate) fn detect(&self, text: &str, cancel: &AtomicBool) -> Result<Vec<Span>> {
        let mut spans = self.classify(text, cancel)?;
        spans.extend(rules::scan(text));
        Ok(merge(spans))
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

pub(crate) use masker::Masker;
pub(crate) use masker::split_placeholder;

pub(crate) type RestoreMap = std::collections::HashMap<String, String>;

pub(crate) struct Ready {
    pub(crate) model: Model,
    pub(crate) key: [u8; 32],
}

pub(crate) struct Redactor {
    dir: PathBuf,
    key: PathBuf,
    sessions: PathBuf,
    maps: Mutex<std::collections::HashMap<String, Arc<Mutex<map::Map>>>>,
    ready: Mutex<Option<Arc<Ready>>>,
    init: Mutex<()>,
    started: AtomicBool,
}

impl Redactor {
    fn new(config_dir: &Path, sessions: &Path, preload: bool) -> Arc<Self> {
        let redactor = Arc::new(Self {
            dir: config_dir.join("models").join("astrlink-guard"),
            key: masker::key_path(config_dir),
            sessions: sessions.to_path_buf(),
            maps: Mutex::new(std::collections::HashMap::new()),
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

    fn map(&self, session_id: &str) -> Option<Arc<Mutex<map::Map>>> {
        if session_id.is_empty() {
            return None;
        }
        let mut maps = self.maps.lock().ok()?;
        Some(Arc::clone(
            maps.entry(session_id.to_string()).or_insert_with(|| {
                Arc::new(Mutex::new(map::Map::open(&self.sessions, session_id)))
            }),
        ))
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
                    let _loading = Progress::loading();
                    let model = Model::load(&self.dir)?;
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Preparing {
    Downloading { done: u64, total: u64 },
    Loading,
}

// Each preparation carries a token so a stale one cannot clear the state of a newer one.
static PREPARING: Mutex<Option<(u64, Preparing)>> = Mutex::new(None);
static NEXT_TOKEN: AtomicU64 = AtomicU64::new(1);

pub(crate) fn preparing() -> Option<Preparing> {
    PREPARING
        .lock()
        .ok()
        .and_then(|state| state.map(|(_, state)| state))
}

pub(crate) struct Progress {
    token: u64,
}

impl Progress {
    fn begin(state: Preparing) -> Self {
        let token = NEXT_TOKEN.fetch_add(1, Ordering::SeqCst);
        if let Ok(mut slot) = PREPARING.lock() {
            *slot = Some((token, state));
        }
        Self { token }
    }

    pub(crate) fn start(total: u64) -> Self {
        Self::begin(Preparing::Downloading { done: 0, total })
    }

    pub(crate) fn loading() -> Self {
        Self::begin(Preparing::Loading)
    }

    pub(crate) fn advance(&self, done: u64) {
        if let Ok(mut state) = PREPARING.lock()
            && let Some((token, Preparing::Downloading { total, .. })) = *state
            && token == self.token
        {
            *state = Some((token, Preparing::Downloading { done, total }));
        }
    }
}

impl Drop for Progress {
    fn drop(&mut self) {
        if let Ok(mut state) = PREPARING.lock()
            && state.is_some_and(|(token, _)| token == self.token)
        {
            *state = None;
        }
    }
}

static ACTIVE: Mutex<Option<(bool, PathBuf, Arc<Redactor>)>> = Mutex::new(None);

/// Applies the switch, including after a configuration reload. A reload that
/// leaves the switch and the directory alone keeps the running instance.
pub(crate) fn install(enabled: bool, config_dir: &Path, sessions_dir: &Path) {
    let Ok(mut slot) = ACTIVE.lock() else {
        return;
    };
    if let Some((current_enabled, current_dir, redactor)) = slot.as_mut()
        && *current_dir == config_dir
        && redactor.sessions == sessions_dir
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
        Redactor::new(config_dir, sessions_dir, enabled),
    ));
}

#[cfg(test)]
pub(crate) fn active_snapshot() -> Option<(bool, PathBuf, Arc<Redactor>)> {
    ACTIVE.lock().ok().and_then(|slot| slot.clone())
}

#[cfg(test)]
pub(crate) fn restore_active(snapshot: Option<(bool, PathBuf, Arc<Redactor>)>) {
    if let Ok(mut slot) = ACTIVE.lock() {
        *slot = snapshot;
    }
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

fn check_cancel(cancel: &AtomicBool) -> Result<()> {
    if cancel.load(Ordering::Relaxed) {
        anyhow::bail!("redaction cancelled");
    }
    Ok(())
}

pub(crate) fn mask_text(
    redactor: &Redactor,
    session_id: Option<&str>,
    text: &str,
) -> Result<String> {
    let Some(map) = session_id.and_then(|session_id| redactor.map(session_id)) else {
        return Ok(text.to_string());
    };
    let mut masker = redactor.masker()?;
    let cancel = AtomicBool::new(false);
    let masked = masker.mask_tracked(text, &cancel)?;
    map.lock()
        .map_err(|_| anyhow::anyhow!("the session map lock is poisoned"))?
        .add(&masker.finish())?;
    Ok(masked)
}

pub(crate) fn mask_json(
    redactor: &Redactor,
    session_id: Option<&str>,
    value: &mut serde_json::Value,
) -> Result<()> {
    let Some(map) = session_id.and_then(|session_id| redactor.map(session_id)) else {
        return Ok(());
    };
    let mut masker = redactor.masker()?;
    let cancel = AtomicBool::new(false);
    masker.mask_value(value, &cancel)?;
    map.lock()
        .map_err(|_| anyhow::anyhow!("the session map lock is poisoned"))?
        .add(&masker.finish())
}

pub(crate) fn mask_value(
    redactor: &Redactor,
    value: &mut serde_json::Value,
) -> Result<Arc<RestoreMap>> {
    let cancel = AtomicBool::new(false);
    let mut masker = redactor.masker()?;
    masker.mask_value(value, &cancel)?;
    Ok(masker.finish())
}

pub(crate) fn restore_session_text(session_id: &str, text: &str) -> String {
    crate::redact::installed()
        .and_then(|redactor| session_restore_map(&redactor, Some(session_id)))
        .map(|values| restore_text(&values, text))
        .unwrap_or_else(|| text.to_string())
}

pub(crate) fn session_restore_map(
    redactor: &Redactor,
    session_id: Option<&str>,
) -> Option<Arc<RestoreMap>> {
    let map = redactor.map(session_id?)?;
    let values = map.lock().ok()?.values().clone();
    (!values.is_empty()).then(|| Arc::new(values))
}

pub(crate) fn mask_item(
    redactor: &Redactor,
    item: &mut crate::request_builder::HistoryItem,
    session_id: Option<&str>,
) -> Result<()> {
    use crate::user_content::UserMessagePart;

    let Some(map) = session_id.and_then(|session_id| redactor.map(session_id)) else {
        return Ok(());
    };
    let mut masker = redactor.masker()?;
    let cancel = AtomicBool::new(false);
    match item {
        crate::request_builder::HistoryItem::ContextSummary { text }
        | crate::request_builder::HistoryItem::InternalContinuation { text } => {
            *text = masker.mask_tracked(text, &cancel)?;
        }
        crate::request_builder::HistoryItem::UserMessage { content } => {
            let skills = content.selected_skills.clone();
            let mut parts = Vec::with_capacity(content.parts().len());
            for part in content.parts() {
                match part {
                    UserMessagePart::Text { mut text } => {
                        text = masker.mask_tracked(&text, &cancel)?;
                        parts.push(UserMessagePart::Text { text });
                    }
                    part => parts.push(part),
                }
            }
            *content = crate::user_content::UserMessageContent::from_parts(parts)
                .with_selected_skills(skills);
        }
        crate::request_builder::HistoryItem::AssistantTurn {
            text,
            reasoning_content,
            calls,
            ..
        } => {
            if let Some(text) = text {
                *text = masker.mask_tracked(text, &cancel)?;
            }
            if let Some(reasoning) = reasoning_content {
                *reasoning = masker.mask_tracked(reasoning, &cancel)?;
            }
            for call in calls {
                let mut arguments = serde_json::from_str(&call.arguments_json)?;
                masker.mask_value(&mut arguments, &cancel)?;
                call.arguments_json = serde_json::to_string(&arguments)?;
            }
        }
        crate::request_builder::HistoryItem::ToolOutput { output_json, .. } => {
            let mut output = serde_json::from_str(output_json)?;
            masker.mask_value(&mut output, &cancel)?;
            *output_json = serde_json::to_string(&output)?;
        }
    }
    map.lock()
        .map_err(|_| anyhow::anyhow!("the session map lock is poisoned"))?
        .add(&masker.finish())
}

pub(crate) fn restore_text(values: &RestoreMap, text: &str) -> String {
    masker::restore_values(values, text)
}

pub(crate) fn restore_messages(
    values: &RestoreMap,
    messages: &mut [crate::agent::ConversationMessage],
) {
    for message in messages {
        message.content = restore_text(values, &message.content);
    }
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

    #[test]
    fn matches_the_reference_spans() {
        let Ok(dir) = std::env::var("LETCODE_PII_MODEL_DIR") else {
            return;
        };
        let model = Model::load(Path::new(&dir)).expect("the model loads");
        let cases: Vec<Case> =
            serde_json::from_str(include_str!("../../tests/fixtures/pii_guard_spans.json"))
                .expect("the fixture parses");
        let cancel = AtomicBool::new(false);
        for case in cases {
            let spans = model.detect(&case.text, &cancel).expect("detection runs");
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
            model: Model::load(Path::new(&dir)).expect("the model loads"),
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
