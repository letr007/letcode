use anyhow::{Context, Result};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

#[derive(Debug)]
pub struct FastMode {
    config_path: PathBuf,
    state: Mutex<bool>,
}

pub(crate) struct PreparedFastModeDisable {
    mode: Arc<FastMode>,
}

impl FastMode {
    pub fn load(config_path: impl Into<PathBuf>, enabled: bool) -> Arc<Self> {
        Arc::new(Self {
            config_path: config_path.into(),
            state: Mutex::new(enabled),
        })
    }

    pub fn enabled(&self) -> bool {
        *self.state.lock().expect("Fast Mode state poisoned")
    }

    pub fn toggle(&self, model_id: &str) -> Result<FastModeToggle> {
        let mut state = self.state.lock().expect("Fast Mode state poisoned");
        let enabled = !*state;
        if enabled && !is_fast_capable_model(model_id) {
            return Ok(FastModeToggle::Unavailable);
        }
        self.set_enabled_locked(&mut state, enabled)?;
        Ok(if enabled {
            FastModeToggle::Enabled
        } else {
            FastModeToggle::Disabled
        })
    }

    pub fn auto_disable_for_model(&self, model_id: &str) -> Result<bool> {
        let mut state = self.state.lock().expect("Fast Mode state poisoned");
        if *state && !is_fast_capable_model(model_id) {
            self.set_enabled_locked(&mut state, false)?;
            return Ok(true);
        }
        Ok(false)
    }

    pub(crate) fn prepare_auto_disable_for_model(
        self: &Arc<Self>,
        model_id: &str,
    ) -> Result<Option<PreparedFastModeDisable>> {
        let state = self.state.lock().expect("Fast Mode state poisoned");
        if !*state || is_fast_capable_model(model_id) {
            return Ok(None);
        }
        crate::config::validate_fast_mode_update(&self.config_path, false)?;
        Ok(Some(PreparedFastModeDisable {
            mode: Arc::clone(self),
        }))
    }

    fn set_enabled_locked(&self, current: &mut bool, enabled: bool) -> Result<()> {
        crate::config::persist_fast_mode_enabled(&self.config_path, enabled).with_context(
            || {
                format!(
                    "failed to persist Fast Mode state in {}",
                    self.config_path.display()
                )
            },
        )?;
        *current = enabled;
        Ok(())
    }
}

impl PreparedFastModeDisable {
    pub(crate) fn commit(self) -> Result<()> {
        let mut state = self.mode.state.lock().expect("Fast Mode state poisoned");
        crate::config::persist_fast_mode_enabled(&self.mode.config_path, false)?;
        *state = false;
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FastModeToggle {
    Enabled,
    Disabled,
    Unavailable,
}

pub fn is_fast_capable_model(model_id: &str) -> bool {
    let id = model_id.trim().to_ascii_lowercase();
    id.starts_with("gpt-") && !id.starts_with("gpt-image")
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn temp_config(name: &str, fast_mode: Option<bool>) -> PathBuf {
        let directory = std::env::temp_dir().join(format!(
            "letcode-fast-mode-{name}-{}",
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .expect("time ok")
                .as_nanos()
        ));
        fs::create_dir_all(&directory).expect("create config directory");
        let path = directory.join("letcode.toml");
        let fast_mode = fast_mode
            .map(|enabled| format!("fast_mode = {enabled}\n"))
            .unwrap_or_default();
        fs::write(
            &path,
            format!(
                "{fast_mode}active_provider = \"primary\"\n\
                 [providers.primary]\n\
                 base_url = \"https://primary.invalid/v1\"\n\
                 api_key = \"primary-key\"\n\
                 protocol = \"responses\"\n\
                 [providers.primary.models.gpt-5]\n"
            ),
        )
        .expect("write config");
        path
    }

    #[test]
    fn omitted_fast_mode_defaults_disabled_and_toggle_persists_in_main_config() {
        let path = temp_config("persist", None);
        let mode = FastMode::load(&path, false);
        assert!(!mode.enabled());
        assert_eq!(
            mode.toggle("gpt-5.5").expect("enable fast mode"),
            FastModeToggle::Enabled
        );
        assert!(
            crate::config::AppConfig::load_from_path(&path)
                .expect("reload config state")
                .fast_mode_enabled
        );
        assert!(
            !path.with_file_name("fast-mode.json").exists(),
            "Fast Mode must not create a separate state file"
        );
    }

    #[test]
    fn unsupported_model_auto_disables_main_config_state() {
        let path = temp_config("auto-disable", Some(true));
        let mode = FastMode::load(&path, true);
        assert!(mode.auto_disable_for_model("claude-4").expect("disable"));
        assert!(
            !crate::config::AppConfig::load_from_path(&path)
                .expect("reload state")
                .fast_mode_enabled
        );
    }
}
