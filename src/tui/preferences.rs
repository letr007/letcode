use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use anyhow::Result;
use serde::{Deserialize, Serialize};

use crate::command::{ThemeName, ThoughtsDisplayMode, ToolsDisplayMode};
use crate::tui::i18n::Language;

const TUI_PREFERENCES_FILE: &str = "tui-preferences.json";
static PREFERENCES_WRITE_SEQUENCE: AtomicU64 = AtomicU64::new(0);

/// Whether a persisted value has the shape of a UUID: five groups of hex digits
/// separated by dashes, `8-4-4-4-12`.
fn is_uuid_shaped(value: &str) -> bool {
    let bytes = value.as_bytes();
    bytes.len() == 36
        && bytes.iter().enumerate().all(|(index, byte)| {
            if matches!(index, 8 | 13 | 18 | 23) {
                *byte == b'-'
            } else {
                byte.is_ascii_hexdigit()
            }
        })
}

fn default_theme_id() -> String {
    ThemeName::default().as_str().to_string()
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TuiPreferences {
    #[serde(default)]
    pub tool_output_expanded: bool,
    #[serde(default = "default_transcript_scrollbar_visible")]
    pub transcript_scrollbar_visible: bool,
    #[serde(default)]
    pub sidebar_hidden: bool,
    #[serde(default)]
    pub sidebar_forced_open: bool,
    #[serde(default = "default_theme_id")]
    pub theme: String,
    #[serde(default)]
    pub thoughts_display: ThoughtsDisplayMode,
    #[serde(default)]
    pub tools_display: ToolsDisplayMode,
    #[serde(default)]
    pub language: Option<String>,
    #[serde(default)]
    pub fake_installation_id: Option<String>,
}

impl Default for TuiPreferences {
    fn default() -> Self {
        Self {
            tool_output_expanded: false,
            transcript_scrollbar_visible: default_transcript_scrollbar_visible(),
            sidebar_hidden: false,
            sidebar_forced_open: false,
            theme: default_theme_id(),
            thoughts_display: ThoughtsDisplayMode::default(),
            tools_display: ToolsDisplayMode::default(),
            language: None,
            fake_installation_id: None,
        }
    }
}

const fn default_transcript_scrollbar_visible() -> bool {
    true
}

impl TuiPreferences {
    /// Returns a stable synthetic installation id, creating one only when the
    /// fake pipeline needs it.
    pub fn ensure_fake_installation_id(&mut self) -> String {
        if let Some(id) = self.fake_installation_id.as_deref()
            && is_uuid_shaped(id)
        {
            return id.to_string();
        }
        // An installation id is a version 4 UUID. A persisted value that is not
        // one cannot have come from a real client, so it is replaced rather than
        // sent as this client's identity.
        let id = crate::fake::synthetic_installation_id();
        self.fake_installation_id = Some(id.clone());
        id
    }

    pub fn load_from_dir(config_dir: &Path) -> Self {
        let path = preferences_path(config_dir);
        fs::read_to_string(path)
            .ok()
            .and_then(|text| serde_json::from_str::<Self>(&text).ok())
            .unwrap_or_default()
    }

    pub fn explicit_language(&self) -> Option<Language> {
        match self.language.as_deref() {
            None => None,
            Some(value) => match Language::parse(value) {
                Some(language) => Some(language),
                None => {
                    tracing::warn!(
                        value,
                        "unsupported persisted TUI language; using system locale"
                    );
                    None
                }
            },
        }
    }

    pub fn update_in_dir(config_dir: &Path, update: impl FnOnce(&mut Self)) -> Result<()> {
        let path = preferences_path(config_dir);
        fs::create_dir_all(config_dir)?;
        let config_target = fs::canonicalize(config_dir)?.join(
            path.file_name()
                .ok_or_else(|| anyhow::anyhow!("preferences path has no file name"))?,
        );
        let _lock = crate::config::acquire_config_lock(&config_target)?;
        let mut current = fs::read_to_string(&path)
            .ok()
            .and_then(|text| serde_json::from_str::<Self>(&text).ok())
            .unwrap_or_default();
        update(&mut current);
        let json = serde_json::to_vec_pretty(&current)?;
        let parent = path.parent().unwrap_or_else(|| Path::new("."));
        let file_name = path
            .file_name()
            .ok_or_else(|| anyhow::anyhow!("preferences path has no file name"))?;
        let temp_path = parent.join(format!(
            ".{}.{}.{}.tmp",
            file_name.to_string_lossy(),
            std::process::id(),
            PREFERENCES_WRITE_SEQUENCE.fetch_add(1, Ordering::Relaxed)
        ));
        let write_result = (|| -> Result<()> {
            let mut temp = OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(&temp_path)?;
            temp.write_all(&json)?;
            temp.sync_all()?;
            drop(temp);
            crate::config::replace_file(&temp_path, &path)
        })();
        if write_result.is_err() {
            let _ = fs::remove_file(&temp_path);
        }
        write_result
    }
}

pub fn preferences_path(config_dir: &Path) -> PathBuf {
    config_dir.join(TUI_PREFERENCES_FILE)
}

#[cfg(test)]
mod tests {
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
        };
        TuiPreferences::update_in_dir(&base, |current| *current = prefs.clone())
            .expect("save preferences");

        let loaded = TuiPreferences::load_from_dir(&base);
        assert_eq!(loaded, prefs);
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
        };
        let json = serde_json::to_string(&prefs).expect("serialize");
        let loaded: TuiPreferences = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(loaded.theme, "sunset");
    }
}
