use std::collections::HashMap;
use std::sync::OnceLock;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Language {
    En,
    ZhCn,
}

impl Language {
    pub const fn id(self) -> &'static str {
        match self {
            Self::En => "en",
            Self::ZhCn => "zh-CN",
        }
    }

    pub fn parse(value: &str) -> Option<Self> {
        let normalized = value.trim().replace('_', "-").to_ascii_lowercase();
        match normalized.as_str() {
            "en" | "en-us" | "en-gb" => Some(Self::En),
            "zh" | "zh-cn" => Some(Self::ZhCn),
            _ => None,
        }
    }
}

pub(crate) fn language_from_locale(locale: Option<&str>) -> Language {
    match locale.map(str::trim).filter(|locale| !locale.is_empty()) {
        Some(locale) if locale.to_ascii_lowercase().starts_with("zh") => Language::ZhCn,
        Some(locale) if locale.to_ascii_lowercase().starts_with("en") => Language::En,
        _ => Language::En,
    }
}

pub fn system_language() -> Language {
    language_from_locale(sys_locale::get_locale().as_deref())
}

const EN: &str = include_str!("../../locales/en.toml");
const ZH_CN: &str = include_str!("../../locales/zh-CN.toml");

#[derive(Debug, Clone)]
pub struct Translator {
    language: Language,
    english: &'static HashMap<String, String>,
    current: &'static HashMap<String, String>,
}

impl Translator {
    pub fn new(language: Language) -> Self {
        let english = english_catalog();
        let current = if language == Language::En {
            english
        } else {
            zh_cn_catalog()
        };
        Self {
            language,
            english,
            current,
        }
    }

    /// The translation for `key` in the active language, falling back to English.
    pub fn lookup(&self, key: &str) -> Option<String> {
        self.current
            .get(key)
            .or_else(|| self.english.get(key))
            .cloned()
    }

    pub fn t(&self, key: &str) -> String {
        if let Some(value) = self.current.get(key) {
            return value.clone();
        }
        if let Some(value) = self.english.get(key) {
            tracing::warn!(
                key,
                language = self.language.id(),
                "missing TUI translation; using English fallback"
            );
            return value.clone();
        }
        tracing::warn!(key, "missing TUI translation key");
        key.to_string()
    }

    pub fn t_fmt(&self, key: &str, args: &[(&str, &str)]) -> String {
        substitute(self.t(key), args)
    }

    pub fn model_failure_summary(&self, failure: &crate::model_runtime::ModelFailure) -> String {
        use crate::model_runtime::{FailureKind, FailurePhase};
        let key = match failure.code.as_deref() {
            Some("max_iterations") => "error.model_iteration_limit",
            Some("max_tool_calls") => "error.model_tool_limit",
            _ => match failure.kind {
                FailureKind::InvalidRequest => "error.model_invalid_request",
                FailureKind::UnsupportedProtocol => "error.model_unsupported_protocol",
                FailureKind::Authentication => "error.model_authentication",
                FailureKind::RateLimited => "error.model_rate_limited",
                FailureKind::Timeout => "error.model_timeout",
                FailureKind::Http if failure.status.is_some_and(|status| status >= 500) => {
                    "error.model_unavailable"
                }
                FailureKind::Http
                    if failure.status.is_none() && failure.phase == FailurePhase::Transport =>
                {
                    "error.model_connection"
                }
                FailureKind::Http => "error.model_http",
                FailureKind::MalformedResponse => "error.model_malformed_response",
                FailureKind::Internal => "error.model_internal",
            },
        };
        let reason = self.t(key);
        match failure.status {
            Some(status) => self.t_fmt(
                "error.with_status",
                &[("reason", &reason), ("status", &status.to_string())],
            ),
            None => reason,
        }
    }

    pub fn model_failure_message(&self, failure: &crate::model_runtime::ModelFailure) -> String {
        use crate::model_runtime::FailureKind;
        let reason = self.model_failure_summary(failure);
        let stopped = match failure.retry_exhausted_attempts {
            Some(attempts) => self.t_fmt(
                "error.retry_exhausted",
                &[("attempts", &attempts.to_string())],
            ),
            None => self.t("error.request_stopped"),
        };
        let action_key = match failure.code.as_deref() {
            Some("max_iterations" | "max_tool_calls") => "error.action_continue",
            _ => match failure.kind {
                FailureKind::Authentication => "error.action_authentication",
                FailureKind::RateLimited => "error.action_rate_limited",
                FailureKind::Http
                    if failure
                        .status
                        .is_some_and(|status| (400..500).contains(&status) && status != 408) =>
                {
                    "error.action_configuration"
                }
                FailureKind::Timeout | FailureKind::Http => "error.action_connection",
                FailureKind::InvalidRequest | FailureKind::UnsupportedProtocol => {
                    "error.action_configuration"
                }
                FailureKind::MalformedResponse | FailureKind::Internal => "error.action_details",
            },
        };
        format!("{reason}\n{stopped}\n{}", self.t(action_key))
    }

    /// [`Self::lookup`] with `{name}` placeholders replaced by `args`.
    pub fn lookup_fmt(&self, key: &str, args: &[(String, String)]) -> Option<String> {
        let args: Vec<(&str, &str)> = args
            .iter()
            .map(|(name, value)| (name.as_str(), value.as_str()))
            .collect();
        self.lookup(key).map(|value| substitute(value, &args))
    }
}

fn substitute(value: String, args: &[(&str, &str)]) -> String {
    args.iter().fold(value, |value, (name, replacement)| {
        value.replace(&format!("{{{name}}}"), replacement)
    })
}

fn english_catalog() -> &'static HashMap<String, String> {
    static CATALOG: OnceLock<HashMap<String, String>> = OnceLock::new();
    CATALOG.get_or_init(|| parse_catalog(EN))
}

fn zh_cn_catalog() -> &'static HashMap<String, String> {
    static CATALOG: OnceLock<HashMap<String, String>> = OnceLock::new();
    CATALOG.get_or_init(|| parse_catalog(ZH_CN))
}

fn parse_catalog(source: &str) -> HashMap<String, String> {
    let value = source
        .parse::<toml::Value>()
        .expect("embedded translation catalog must be valid TOML");
    let mut catalog = HashMap::new();
    flatten_table(&value, "", &mut catalog);
    catalog
}

fn flatten_table(value: &toml::Value, prefix: &str, catalog: &mut HashMap<String, String>) {
    let toml::Value::Table(table) = value else {
        return;
    };
    for (key, value) in table {
        let full_key = if prefix.is_empty() {
            key.clone()
        } else {
            format!("{prefix}.{key}")
        };
        match value {
            toml::Value::String(text) => {
                catalog.insert(full_key, text.clone());
            }
            toml::Value::Table(_) => flatten_table(value, &full_key, catalog),
            _ => {}
        }
    }
}

#[cfg(test)]
#[path = "../../tests/internal/tui/i18n/tests.rs"]
mod tests;
