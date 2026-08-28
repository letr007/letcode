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
        args.iter().fold(self.t(key), |value, (name, replacement)| {
            value.replace(&format!("{{{name}}}"), replacement)
        })
    }
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
mod tests {
    use super::*;

    #[test]
    fn language_normalizes_supported_ids() {
        assert_eq!(Language::parse("en"), Some(Language::En));
        assert_eq!(Language::parse("en-US"), Some(Language::En));
        assert_eq!(Language::parse("zh"), Some(Language::ZhCn));
        assert_eq!(Language::parse("zh_CN"), Some(Language::ZhCn));
        assert_eq!(Language::parse("zh-CN"), Some(Language::ZhCn));
        assert_eq!(Language::parse("fr"), None);
    }

    #[test]
    fn system_locale_resolution_maps_supported_and_unsupported_locales() {
        assert_eq!(language_from_locale(Some("zh-TW")), Language::ZhCn);
        assert_eq!(language_from_locale(Some("en-GB")), Language::En);
        assert_eq!(language_from_locale(Some("fr-FR")), Language::En);
        assert_eq!(language_from_locale(None), Language::En);
    }

    #[test]
    fn translator_interpolates_and_falls_back_to_english() {
        let translator = Translator::new(Language::ZhCn);
        assert_eq!(translator.t("command.help"), "显示可用的本地命令");
        assert_eq!(translator.t("unknown.key"), "unknown.key");
    }

    #[test]
    fn english_fallback_emits_a_warning() {
        use std::sync::{Arc, Mutex};
        use tracing_subscriber::fmt::MakeWriter;

        #[derive(Clone)]
        struct Writer(Arc<Mutex<Vec<u8>>>);
        impl<'a> MakeWriter<'a> for Writer {
            type Writer = Capture;
            fn make_writer(&'a self) -> Self::Writer {
                Capture(self.0.clone())
            }
        }
        struct Capture(Arc<Mutex<Vec<u8>>>);
        impl std::io::Write for Capture {
            fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
                self.0.lock().unwrap().extend_from_slice(bytes);
                Ok(bytes.len())
            }
            fn flush(&mut self) -> std::io::Result<()> {
                Ok(())
            }
        }

        let output = Arc::new(Mutex::new(Vec::new()));
        let subscriber = tracing_subscriber::fmt()
            .with_ansi(false)
            .with_writer(Writer(output.clone()))
            .finish();
        tracing::subscriber::with_default(subscriber, || {
            assert_eq!(
                Translator::new(Language::ZhCn).t("parse.only_english"),
                "English only"
            );
        });
        let output = String::from_utf8(output.lock().unwrap().clone()).unwrap();
        assert!(output.contains("missing TUI translation"), "{output}");
    }

    #[test]
    fn nested_catalogs_flatten_to_dotted_keys() {
        let catalog = parse_catalog("[parse]\n[parse.inner]\nmessage = \"ok\"\n");
        assert_eq!(catalog.get("parse.inner.message"), Some(&"ok".to_string()));
    }
}
