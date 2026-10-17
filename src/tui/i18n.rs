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
