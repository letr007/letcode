mod fetch;
mod forward;
mod kernels;
mod masker;
mod onnx;

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

    pub(crate) fn detect(&self, text: &str) -> Result<Vec<Span>> {
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

#[cfg(test)]
mod tests {
    use super::*;

    // Reference spans from the same model read by ONNX Runtime; texts and values are synthetic.
    #[test]
    fn matches_the_reference_spans() {
        let Ok(dir) = std::env::var("LETCODE_PII_MODEL_DIR") else {
            return;
        };
        let model = Model::load(Path::new(&dir)).expect("the model loads");
        let cases: Vec<Case> =
            serde_json::from_str(include_str!("../../tests/fixtures/pii_guard_spans.json"))
                .expect("the fixture parses");
        for case in cases {
            let spans = model.detect(&case.text).expect("detection runs");
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

use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

use crate::model_runtime::{ContentPart, ModelRequestInput};
use masker::Masker;

pub(crate) struct Redactor {
    dir: PathBuf,
    key: PathBuf,
    masker: Mutex<Option<Masker>>,
    fetching: AtomicBool,
}

impl Redactor {
    fn new(config_dir: &Path) -> Arc<Self> {
        let redactor = Arc::new(Self {
            dir: config_dir.join("models").join("astrlink-guard"),
            key: masker::key_path(config_dir),
            masker: Mutex::new(None),
            fetching: AtomicBool::new(false),
        });
        let worker = Arc::clone(&redactor);
        std::thread::spawn(move || worker.ensure());
        redactor
    }

    pub(crate) fn mask(&self, text: &str) -> String {
        if self.ready().is_none() {
            self.ensure();
        }
        let Ok(mut slot) = self.masker.lock() else {
            return text.to_string();
        };
        match slot.as_mut() {
            Some(masker) => masker.mask(text).unwrap_or_else(|_| text.to_string()),
            None => text.to_string(),
        }
    }

    pub(crate) fn restore(&self, text: &str) -> String {
        match self.masker.lock() {
            Ok(slot) => slot
                .as_ref()
                .map_or_else(|| text.to_string(), |m| m.restore(text)),
            Err(_) => text.to_string(),
        }
    }

    fn ready(&self) -> Option<()> {
        self.masker
            .lock()
            .ok()
            .and_then(|slot| slot.as_ref().map(|_| ()))
    }

    fn ensure(&self) {
        if self.ready().is_some() {
            return;
        }
        if !fetch::present(&self.dir) {
            if self.fetching.swap(true, Ordering::SeqCst) {
                return;
            }
            if let Err(error) = fetch::fetch_missing(&self.dir) {
                tracing::warn!("masking model unavailable: {error}");
            }
            self.fetching.store(false, Ordering::SeqCst);
        }
        let Ok(mut slot) = self.masker.lock() else {
            return;
        };
        if slot.is_none() {
            match Masker::open(&self.dir, &self.key) {
                Ok(masker) => *slot = Some(masker),
                Err(error) => tracing::warn!("masking model did not load: {error}"),
            }
        }
    }
}

static ACTIVE: OnceLock<Option<Arc<Redactor>>> = OnceLock::new();

pub(crate) fn install(enabled: bool, config_dir: &Path) {
    let _ = ACTIVE.set(enabled.then(|| Redactor::new(config_dir)));
}

pub(crate) fn active() -> Option<&'static Arc<Redactor>> {
    ACTIVE.get().and_then(Option::as_ref)
}

pub(crate) fn mask_input(redactor: &Redactor, input: &mut ModelRequestInput) {
    for segment in &mut input.segments {
        segment.text = redactor.mask(&segment.text);
    }
    for message in &mut input.messages {
        mask_parts(redactor, &mut message.content);
    }
}

fn mask_parts(redactor: &Redactor, parts: &mut [ContentPart]) {
    for part in parts {
        match part {
            ContentPart::Text(text) => *text = redactor.mask(text),
            ContentPart::Reasoning { text, .. } => *text = redactor.mask(text),
            ContentPart::ToolCall { arguments, .. } => mask_value(redactor, arguments),
            ContentPart::ToolResult { content, .. } => mask_parts(redactor, content),
            ContentPart::Image { .. } => {}
        }
    }
}

pub(crate) fn mask_value(redactor: &Redactor, value: &mut serde_json::Value) {
    match value {
        serde_json::Value::String(text) => *text = redactor.mask(text),
        serde_json::Value::Array(items) => {
            for item in items {
                mask_value(redactor, item);
            }
        }
        serde_json::Value::Object(map) => {
            for item in map.values_mut() {
                mask_value(redactor, item);
            }
        }
        _ => {}
    }
}

pub(crate) fn restore_value(redactor: &Redactor, value: &mut serde_json::Value) {
    match value {
        serde_json::Value::String(text) => *text = redactor.restore(text),
        serde_json::Value::Array(items) => {
            for item in items {
                restore_value(redactor, item);
            }
        }
        serde_json::Value::Object(map) => {
            for item in map.values_mut() {
                restore_value(redactor, item);
            }
        }
        _ => {}
    }
}
