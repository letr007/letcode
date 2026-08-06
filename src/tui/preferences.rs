use std::fs;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::command::ThemeName;

const TUI_PREFERENCES_FILE: &str = "tui-preferences.json";

fn default_theme_id() -> String {
    ThemeName::default().as_str().to_string()
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TuiPreferences {
    #[serde(default)]
    pub tool_output_expanded: bool,
    #[serde(default = "default_transcript_scrollbar_visible")]
    pub transcript_scrollbar_visible: bool,
    #[serde(default = "default_theme_id")]
    pub theme: String,
}

impl Default for TuiPreferences {
    fn default() -> Self {
        Self {
            tool_output_expanded: false,
            transcript_scrollbar_visible: default_transcript_scrollbar_visible(),
            theme: default_theme_id(),
        }
    }
}

const fn default_transcript_scrollbar_visible() -> bool {
    true
}

impl TuiPreferences {
    pub fn load_from_dir(config_dir: &Path) -> Self {
        let path = preferences_path(config_dir);
        fs::read_to_string(path)
            .ok()
            .and_then(|text| serde_json::from_str::<Self>(&text).ok())
            .unwrap_or_default()
    }

    pub fn save_to_dir(&self, config_dir: &Path) -> anyhow::Result<()> {
        fs::create_dir_all(config_dir)?;
        let path = preferences_path(config_dir);
        let json = serde_json::to_string_pretty(self)?;
        fs::write(path, json)?;
        Ok(())
    }
}

pub fn preferences_path(config_dir: &Path) -> PathBuf {
    config_dir.join(TUI_PREFERENCES_FILE)
}

#[cfg(test)]
mod tests {
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
            theme: "forest".into(),
        };
        prefs.save_to_dir(&base).expect("save preferences");

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
    fn custom_theme_id_round_trips() {
        let prefs = TuiPreferences {
            tool_output_expanded: false,
            transcript_scrollbar_visible: true,
            theme: "sunset".into(),
        };
        let json = serde_json::to_string(&prefs).expect("serialize");
        let loaded: TuiPreferences = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(loaded.theme, "sunset");
    }
}
