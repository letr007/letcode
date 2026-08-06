use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use ratatui::style::Color;
use serde::Deserialize;

use super::theme::{Theme, ThemeName};

const THEMES_DIR: &str = "themes";

const BUNDLED_THEMES: &[(&str, &str)] = &[
    ("ocean", include_str!("../../themes/ocean.toml")),
    ("forest", include_str!("../../themes/forest.toml")),
    ("rose", include_str!("../../themes/rose.toml")),
    ("tokyonight", include_str!("../../themes/tokyonight.toml")),
];

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CustomThemeInfo {
    pub id: String,
    pub path: PathBuf,
    pub label: String,
    pub description: Option<String>,
}

#[derive(Debug, Deserialize)]
struct ThemeFile {
    #[serde(default)]
    label: Option<String>,
    #[serde(default)]
    description: Option<String>,
    #[serde(default)]
    root_bg: Option<String>,
    #[serde(default)]
    surface_bg: Option<String>,
    #[serde(default)]
    element_bg: Option<String>,
    #[serde(default)]
    elevated_bg: Option<String>,
    #[serde(default)]
    border: Option<String>,
    #[serde(default)]
    text: Option<String>,
    #[serde(default)]
    muted_text: Option<String>,
    #[serde(default)]
    dim_text: Option<String>,
    #[serde(default)]
    accent: Option<String>,
    #[serde(default)]
    assistant: Option<String>,
    #[serde(default)]
    user: Option<String>,
    #[serde(default)]
    success: Option<String>,
    #[serde(default)]
    warning: Option<String>,
    #[serde(default)]
    error: Option<String>,
    #[serde(default)]
    approval: Option<String>,
    #[serde(default)]
    notice: Option<String>,
    #[serde(default)]
    diff_add_bg: Option<String>,
    #[serde(default)]
    diff_delete_bg: Option<String>,
    #[serde(default)]
    diff_hunk_bg: Option<String>,
}

pub fn themes_dir(preferences_dir: &Path) -> PathBuf {
    preferences_dir.join(THEMES_DIR)
}

pub fn theme_file_path(preferences_dir: &Path, id: &str) -> PathBuf {
    themes_dir(preferences_dir).join(format!("{id}.toml"))
}

pub fn is_reserved_theme_id(id: &str) -> bool {
    ThemeName::parse(id).is_some()
}

pub fn normalize_theme_id(value: &str) -> Option<String> {
    let trimmed = value.trim();
    if trimmed.is_empty() {
        return None;
    }
    if let Some(builtin) = ThemeName::parse(trimmed) {
        return Some(builtin.as_str().to_string());
    }
    let id = trimmed.to_ascii_lowercase();
    let id = match id.as_str() {
        "tokyo-night" | "tokyo_night" => "tokyonight".to_string(),
        _ => id,
    };
    if !id
        .chars()
        .all(|ch| ch.is_ascii_alphanumeric() || ch == '-' || ch == '_')
    {
        return None;
    }
    Some(id)
}

/// Seed shipped `themes/*.toml` if missing.
pub fn ensure_bundled_themes(preferences_dir: &Path) {
    let dir = themes_dir(preferences_dir);
    if let Err(error) = fs::create_dir_all(&dir) {
        tracing::warn!(%error, path = %dir.display(), "failed to create themes directory");
        return;
    }
    for (id, contents) in BUNDLED_THEMES {
        let path = dir.join(format!("{id}.toml"));
        if path.exists() {
            continue;
        }
        if let Err(error) = fs::write(&path, contents) {
            tracing::warn!(%error, path = %path.display(), "failed to seed bundled theme");
        }
    }
}

pub fn discover_custom_themes(preferences_dir: &Path) -> Vec<CustomThemeInfo> {
    let dir = themes_dir(preferences_dir);
    let Ok(entries) = fs::read_dir(&dir) else {
        return Vec::new();
    };

    let mut themes = Vec::new();
    for entry in entries.flatten() {
        let path = entry.path();
        if path.extension().and_then(|ext| ext.to_str()) != Some("toml") {
            continue;
        }
        let Some(stem) = path.file_stem().and_then(|stem| stem.to_str()) else {
            continue;
        };
        let Some(id) = normalize_theme_id(stem) else {
            continue;
        };
        if is_reserved_theme_id(&id) {
            tracing::warn!(
                path = %path.display(),
                "ignoring theme file that shadows a builtin theme name"
            );
            continue;
        }
        match load_theme_file(&path) {
            Ok((file, _)) => {
                let label = file
                    .label
                    .filter(|value| !value.trim().is_empty())
                    .unwrap_or_else(|| id.clone());
                let description = file
                    .description
                    .map(|value| value.trim().to_string())
                    .filter(|value| !value.is_empty())
                    .or_else(|| Some(format!("Custom theme ({})", path.display())));
                themes.push(CustomThemeInfo {
                    id,
                    path,
                    label,
                    description,
                });
            }
            Err(error) => {
                tracing::warn!(path = %path.display(), %error, "skipping invalid custom theme");
            }
        }
    }
    themes.sort_by(|left, right| left.id.cmp(&right.id));
    themes
}

pub fn load_custom_theme(preferences_dir: &Path, id: &str) -> Result<Theme> {
    let id = normalize_theme_id(id).context("invalid theme id")?;
    if is_reserved_theme_id(&id) {
        bail!("'{id}' is a builtin theme");
    }
    let path = theme_file_path(preferences_dir, &id);
    let (_, theme) = load_theme_file(&path)
        .with_context(|| format!("failed to load theme '{}'", path.display()))?;
    Ok(theme)
}

fn load_theme_file(path: &Path) -> Result<(ThemeFile, Theme)> {
    let text = fs::read_to_string(path)
        .with_context(|| format!("failed to read theme file {}", path.display()))?;
    let file: ThemeFile = toml::from_str(&text)
        .with_context(|| format!("failed to parse theme file {}", path.display()))?;
    let theme = file.to_theme()?;
    Ok((file, theme))
}

impl ThemeFile {
    fn to_theme(&self) -> Result<Theme> {
        let base = Theme::dark();
        Ok(Theme {
            root_bg: parse_optional_color(self.root_bg.as_deref(), "root_bg")?
                .unwrap_or(base.root_bg),
            surface_bg: parse_optional_color(self.surface_bg.as_deref(), "surface_bg")?
                .unwrap_or(base.surface_bg),
            element_bg: parse_optional_color(self.element_bg.as_deref(), "element_bg")?
                .unwrap_or(base.element_bg),
            elevated_bg: parse_optional_color(self.elevated_bg.as_deref(), "elevated_bg")?
                .unwrap_or(base.elevated_bg),
            border: parse_optional_color(self.border.as_deref(), "border")?.unwrap_or(base.border),
            text: parse_optional_color(self.text.as_deref(), "text")?.unwrap_or(base.text),
            muted_text: parse_optional_color(self.muted_text.as_deref(), "muted_text")?
                .unwrap_or(base.muted_text),
            dim_text: parse_optional_color(self.dim_text.as_deref(), "dim_text")?
                .unwrap_or(base.dim_text),
            accent: parse_optional_color(self.accent.as_deref(), "accent")?.unwrap_or(base.accent),
            assistant: parse_optional_color(self.assistant.as_deref(), "assistant")?
                .unwrap_or(base.assistant),
            user: parse_optional_color(self.user.as_deref(), "user")?.unwrap_or(base.user),
            success: parse_optional_color(self.success.as_deref(), "success")?
                .unwrap_or(base.success),
            warning: parse_optional_color(self.warning.as_deref(), "warning")?
                .unwrap_or(base.warning),
            error: parse_optional_color(self.error.as_deref(), "error")?.unwrap_or(base.error),
            approval: parse_optional_color(self.approval.as_deref(), "approval")?
                .unwrap_or(base.approval),
            notice: parse_optional_color(self.notice.as_deref(), "notice")?.unwrap_or(base.notice),
            diff_add_bg: parse_optional_color(self.diff_add_bg.as_deref(), "diff_add_bg")?
                .unwrap_or(base.diff_add_bg),
            diff_delete_bg: parse_optional_color(self.diff_delete_bg.as_deref(), "diff_delete_bg")?
                .unwrap_or(base.diff_delete_bg),
            diff_hunk_bg: parse_optional_color(self.diff_hunk_bg.as_deref(), "diff_hunk_bg")?
                .unwrap_or(base.diff_hunk_bg),
        })
    }
}

fn parse_optional_color(value: Option<&str>, field: &str) -> Result<Option<Color>> {
    match value {
        None => Ok(None),
        Some(raw) => parse_hex_color(raw)
            .with_context(|| format!("invalid color for {field}: {raw}"))
            .map(Some),
    }
}

fn parse_hex_color(value: &str) -> Result<Color> {
    let raw = value.trim();
    let hex = raw
        .strip_prefix('#')
        .ok_or_else(|| anyhow::anyhow!("expected #RRGGBB or #RGB"))?;
    let (r, g, b) = match hex.len() {
        3 => {
            let r = u8::from_str_radix(&hex[0..1], 16)? * 17;
            let g = u8::from_str_radix(&hex[1..2], 16)? * 17;
            let b = u8::from_str_radix(&hex[2..3], 16)? * 17;
            (r, g, b)
        }
        6 => {
            let r = u8::from_str_radix(&hex[0..2], 16)?;
            let g = u8::from_str_radix(&hex[2..4], 16)?;
            let b = u8::from_str_radix(&hex[4..6], 16)?;
            (r, g, b)
        }
        _ => bail!("expected #RRGGBB or #RGB"),
    };
    Ok(Color::Rgb(r, g, b))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_prefs_dir(name: &str) -> PathBuf {
        let base = std::env::temp_dir().join(format!(
            "letcode-theme-file-{name}-{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("time ok")
                .as_nanos()
        ));
        fs::create_dir_all(themes_dir(&base)).expect("create themes dir");
        base
    }

    #[test]
    fn loads_partial_theme_with_dark_defaults() {
        let prefs = temp_prefs_dir("partial");
        let path = theme_file_path(&prefs, "sunset");
        fs::write(
            &path,
            r##"
label = "Sunset"
accent = "#ff6600"
error = "#f00"
"##,
        )
        .expect("write theme");

        let theme = load_custom_theme(&prefs, "sunset").expect("load theme");
        assert_eq!(theme.accent, Color::Rgb(255, 102, 0));
        assert_eq!(theme.error, Color::Rgb(255, 0, 0));
        assert_eq!(theme.root_bg, Theme::dark().root_bg);
    }

    #[test]
    fn discover_skips_builtin_shadow_and_sorts() {
        let prefs = temp_prefs_dir("discover");
        fs::write(
            theme_file_path(&prefs, "zebra"),
            r##"label = "Zebra"
accent = "#112233"
"##,
        )
        .expect("write zebra");
        fs::write(
            theme_file_path(&prefs, "amber"),
            r##"label = "Amber"
accent = "#445566"
"##,
        )
        .expect("write amber");
        fs::write(
            theme_file_path(&prefs, "dark"),
            r##"label = "Nope"
accent = "#000000"
"##,
        )
        .expect("write dark shadow");

        let themes = discover_custom_themes(&prefs);
        assert_eq!(
            themes
                .iter()
                .map(|theme| theme.id.as_str())
                .collect::<Vec<_>>(),
            vec!["amber", "zebra"]
        );
        assert_eq!(themes[0].label, "Amber");
    }

    #[test]
    fn reserved_and_normalize_helpers() {
        assert!(is_reserved_theme_id("dark"));
        assert!(is_reserved_theme_id("rainbow"));
        assert!(!is_reserved_theme_id("tokyonight"));
        assert!(!is_reserved_theme_id("sunset"));
        assert_eq!(
            normalize_theme_id("Tokyo-Night").as_deref(),
            Some("tokyonight")
        );
        assert_eq!(normalize_theme_id("My_Theme").as_deref(), Some("my_theme"));
        assert_eq!(normalize_theme_id("bad theme"), None);
    }

    #[test]
    fn ensure_bundled_themes_seeds_missing_only() {
        let prefs = temp_prefs_dir("bundled");
        ensure_bundled_themes(&prefs);

        let ocean_path = theme_file_path(&prefs, "ocean");
        assert!(ocean_path.exists());
        assert_eq!(
            load_custom_theme(&prefs, "ocean")
                .expect("load ocean")
                .accent,
            Color::Rgb(0x3e, 0xbe, 0xd3)
        );

        fs::write(&ocean_path, r##"accent = "#010203""##).expect("edit ocean");
        ensure_bundled_themes(&prefs);
        assert_eq!(
            load_custom_theme(&prefs, "ocean")
                .expect("reload ocean")
                .accent,
            Color::Rgb(0x01, 0x02, 0x03)
        );

        let ids = discover_custom_themes(&prefs)
            .into_iter()
            .map(|theme| theme.id)
            .collect::<Vec<_>>();
        assert_eq!(ids, vec!["forest", "ocean", "rose", "tokyonight"]);
    }
}
