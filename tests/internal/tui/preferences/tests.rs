use super::*;

#[test]
fn legacy_preferences_default_to_automatic_sidebar() {
    let loaded: TuiPreferences = serde_json::from_str(
        r#"{"tool_output_expanded":false,"transcript_scrollbar_visible":true,"theme":"dark"}"#,
    )
    .expect("legacy preferences deserialize");
    assert!(!loaded.sidebar_hidden);
    assert!(!loaded.sidebar_forced_open);
}

#[test]
fn invalid_persisted_language_falls_back_to_system_locale() {
    let prefs: TuiPreferences = serde_json::from_str(
        r#"{"tool_output_expanded":false,"transcript_scrollbar_visible":true,"theme":"dark","language":"fr"}"#,
    )
    .expect("invalid language preference deserializes");
    assert_eq!(prefs.explicit_language(), None);
}

#[test]
fn legacy_preferences_default_to_full_thoughts_display() {
    let loaded: TuiPreferences = serde_json::from_str(
        r#"{"tool_output_expanded":true,"transcript_scrollbar_visible":false,"theme":"dark"}"#,
    )
    .expect("legacy preferences deserialize");
    assert_eq!(loaded.thoughts_display, ThoughtsDisplayMode::Full);
    assert_eq!(loaded.tools_display, ToolsDisplayMode::Detailed);
}

#[test]
fn fake_installation_id_is_stable_once_created() {
    let mut prefs = TuiPreferences::default();
    assert_eq!(prefs.fake_installation_id, None);

    let first = prefs.ensure_fake_installation_id();
    let second = prefs.ensure_fake_installation_id();
    assert_eq!(first, second);
    assert_eq!(prefs.fake_installation_id, Some(first));
}

#[test]
fn non_uuid_installation_ids_are_replaced() {
    let mut prefs = TuiPreferences {
        fake_installation_id: Some("letcode".to_string()),
        ..TuiPreferences::default()
    };
    let generated = prefs.ensure_fake_installation_id();
    assert!(is_uuid_shaped(&generated), "{generated}");
    assert_eq!(generated.as_bytes()[14], b'4', "version 4");
    assert_eq!(prefs.fake_installation_id, Some(generated));

    // An existing UUID is kept verbatim.
    let kept = "00000000-0000-4000-8000-000000000000";
    let mut existing = TuiPreferences {
        fake_installation_id: Some(kept.to_string()),
        ..TuiPreferences::default()
    };
    assert_eq!(existing.ensure_fake_installation_id(), kept);
}

#[test]
fn custom_theme_id_round_trips() {
    let prefs = TuiPreferences {
        tool_output_expanded: false,
        transcript_scrollbar_visible: true,
        sidebar_hidden: false,
        sidebar_forced_open: true,
        theme: "sunset".into(),
        thoughts_display: ThoughtsDisplayMode::Compact,
        tools_display: ToolsDisplayMode::Detailed,
        language: None,
        fake_installation_id: None,
        fast_mode: false,
    };
    let json = serde_json::to_string(&prefs).expect("serialize");
    let loaded: TuiPreferences = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(loaded.theme, "sunset");
}

mod integration_tests {
    use super::*;

    #[test]
    fn preferences_round_trip() {
        let base = std::env::temp_dir().join(format!(
            "letcode-tui-preferences-test-{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("time ok")
                .as_nanos()
        ));

        let prefs = TuiPreferences {
            tool_output_expanded: true,
            transcript_scrollbar_visible: false,
            sidebar_hidden: true,
            sidebar_forced_open: false,
            theme: "forest".into(),
            thoughts_display: ThoughtsDisplayMode::Titles,
            tools_display: ToolsDisplayMode::Compact,
            language: Some("zh-CN".into()),
            fake_installation_id: Some("fake-installation".into()),
            fast_mode: true,
        };
        TuiPreferences::update_in_dir(&base, |current| *current = prefs.clone())
            .expect("save preferences");

        let loaded = TuiPreferences::load_from_dir(&base);
        assert_eq!(loaded, prefs);
    }

    #[test]
    fn replacement_failures_preserve_unique_sources() {
        let dir = tempfile::tempdir().unwrap();
        let path = preferences_path(dir.path());
        fs::create_dir(&path).unwrap();
        let mut failed_sources = std::collections::BTreeMap::new();

        for theme in ["forest", "sunset"] {
            TuiPreferences::update_in_dir(dir.path(), |prefs| prefs.theme = theme.into())
                .unwrap_err();
            let sources: std::collections::BTreeMap<_, _> = fs::read_dir(dir.path())
                .unwrap()
                .map(|entry| entry.unwrap().path())
                .filter(|source| source.extension().and_then(|ext| ext.to_str()) == Some("tmp"))
                .map(|source| {
                    let bytes = fs::read(&source).unwrap();
                    (source, bytes)
                })
                .collect();
            assert_eq!(sources.len(), failed_sources.len() + 1);
            for (source, bytes) in &failed_sources {
                assert_eq!(sources.get(source), Some(bytes));
            }
            let expected = serde_json::to_vec_pretty(&TuiPreferences {
                theme: theme.into(),
                ..TuiPreferences::default()
            })
            .unwrap();
            assert!(sources.values().any(|bytes| bytes == &expected));
            failed_sources = sources;
        }

        fs::remove_dir(&path).unwrap();
        TuiPreferences::update_in_dir(dir.path(), |prefs| prefs.theme = "forest".into()).unwrap();
        assert_eq!(TuiPreferences::load_from_dir(dir.path()).theme, "forest");
        for (source, bytes) in &failed_sources {
            assert_eq!(&fs::read(source).unwrap(), bytes);
        }
        assert_eq!(
            fs::read_dir(dir.path())
                .unwrap()
                .map(|entry| entry.unwrap().path())
                .filter(|source| source.extension().and_then(|ext| ext.to_str()) == Some("tmp"))
                .count(),
            failed_sources.len()
        );
    }

    #[test]
    fn malformed_preferences_fall_back_to_default() {
        let base = std::env::temp_dir().join(format!(
            "letcode-tui-preferences-bad-test-{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("time ok")
                .as_nanos()
        ));
        fs::create_dir_all(&base).expect("create prefs dir");
        fs::write(preferences_path(&base), "{not valid json").expect("write malformed prefs");

        assert_eq!(
            TuiPreferences::load_from_dir(&base),
            TuiPreferences::default()
        );
    }

    #[test]
    fn field_updates_preserve_other_process_changes() {
        let base = std::env::temp_dir().join(format!(
            "letcode-tui-preferences-update-test-{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("time ok")
                .as_nanos()
        ));
        std::thread::scope(|scope| {
            scope.spawn(|| {
                TuiPreferences::update_in_dir(&base, |prefs| {
                    prefs.theme = "forest".into();
                })
                .expect("write theme");
            });
            scope.spawn(|| {
                TuiPreferences::update_in_dir(&base, |prefs| {
                    prefs.language = Some("zh-CN".into());
                })
                .expect("write language");
            });
        });

        let loaded = TuiPreferences::load_from_dir(&base);
        assert_eq!(loaded.theme, "forest");
        assert_eq!(loaded.language.as_deref(), Some("zh-CN"));
    }
}
