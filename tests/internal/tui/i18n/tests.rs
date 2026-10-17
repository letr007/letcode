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
fn every_notice_id_has_a_translation() {
    let catalog = zh_cn_catalog();
    for message_id in crate::session::NoticeId::ALL {
        assert!(
            catalog.contains_key(message_id.key()),
            "missing translation for {}",
            message_id.key()
        );
    }
}

#[test]
fn translator_interpolates_and_falls_back_to_english() {
    let translator = Translator::new(Language::ZhCn);
    assert_eq!(translator.t("command.help"), "查看命令与使用说明");
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

/// English-only by design: the fallback test needs a key the active
/// catalog does not carry.
const ENGLISH_ONLY_KEYS: &[&str] = &["parse.only_english"];

#[test]
fn chinese_catalog_covers_every_schema_description() {
    let missing = crate::config::schema_keys()
        .into_iter()
        .filter(|key| {
            zh_cn_catalog()
                .get(&format!("config.schema.{key}"))
                .is_none()
        })
        .collect::<Vec<_>>();
    assert!(missing.is_empty(), "{missing:?}");
}

#[test]
fn chinese_catalog_covers_every_english_key() {
    let english = english_catalog();
    let chinese = zh_cn_catalog();
    let mut missing = english
        .keys()
        .filter(|key| !ENGLISH_ONLY_KEYS.contains(&key.as_str()))
        .filter(|key| !chinese.contains_key(*key))
        .collect::<Vec<_>>();
    missing.sort();
    assert!(missing.is_empty(), "zh-CN is missing keys: {missing:?}");
}

#[test]
fn nested_catalogs_flatten_to_dotted_keys() {
    let catalog = parse_catalog("[parse]\n[parse.inner]\nmessage = \"ok\"\n");
    assert_eq!(catalog.get("parse.inner.message"), Some(&"ok".to_string()));
}
