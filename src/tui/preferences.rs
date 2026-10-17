use std::fs;
use std::io::Write;
#[cfg(unix)]
use std::os::unix::fs::PermissionsExt as _;
use std::path::{Path, PathBuf};

use anyhow::Result;
use serde::{Deserialize, Serialize};

use crate::command::{ThemeName, ThoughtsDisplayMode, ToolsDisplayMode};
use crate::tui::i18n::Language;

const TUI_PREFERENCES_FILE: &str = "tui-preferences.json";

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
    #[serde(default)]
    pub fast_mode: bool,
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
            fast_mode: false,
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
        let prefix = format!(".{}.", file_name.to_string_lossy());
        let mut builder = tempfile::Builder::new();
        builder.prefix(&prefix).suffix(".tmp");
        #[cfg(unix)]
        builder.permissions(fs::Permissions::from_mode(0o666));
        let mut temp = builder.tempfile_in(parent)?;
        temp.write_all(&json)?;
        temp.as_file().sync_all()?;
        let temp_path = temp.into_temp_path().keep()?;
        crate::config::replace_file(&temp_path, &path)
    }
}

pub fn preferences_path(config_dir: &Path) -> PathBuf {
    config_dir.join(TUI_PREFERENCES_FILE)
}

#[cfg(test)]
#[path = "../../tests/internal/tui/preferences/tests.rs"]
mod tests;
