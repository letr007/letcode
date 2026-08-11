use anyhow::Result;
use std::sync::{Arc, Mutex};

#[derive(Debug)]
pub struct FastMode {
    state: Mutex<bool>,
}

pub(crate) struct PreparedFastModeDisable {
    mode: Arc<FastMode>,
}

impl FastMode {
    pub fn load(_config_path: impl Into<std::path::PathBuf>, enabled: bool) -> Arc<Self> {
        Arc::new(Self {
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
        *state = enabled;
        Ok(if enabled {
            FastModeToggle::Enabled
        } else {
            FastModeToggle::Disabled
        })
    }

    pub fn auto_disable_for_model(&self, model_id: &str) -> Result<bool> {
        let mut state = self.state.lock().expect("Fast Mode state poisoned");
        if *state && !is_fast_capable_model(model_id) {
            *state = false;
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
        Ok(Some(PreparedFastModeDisable {
            mode: Arc::clone(self),
        }))
    }
}

impl PreparedFastModeDisable {
    pub(crate) fn commit(self) -> Result<()> {
        let mut state = self.mode.state.lock().expect("Fast Mode state poisoned");
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
    use std::path::PathBuf;
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
    fn omitted_fast_mode_defaults_disabled_and_toggle_does_not_change_config() {
        let path = temp_config("toggle", None);
        let before = fs::read_to_string(&path).expect("read config before toggle");
        let mode = FastMode::load(&path, false);
        assert!(!mode.enabled());
        assert_eq!(
            mode.toggle("gpt-5.5").expect("enable fast mode"),
            FastModeToggle::Enabled
        );
        assert!(mode.enabled());
        assert_eq!(
            fs::read_to_string(&path).expect("read config after toggle"),
            before
        );
    }

    #[test]
    fn unsupported_model_auto_disables_without_changing_config() {
        let path = temp_config("auto-disable", Some(true));
        let before = fs::read_to_string(&path).expect("read config before auto-disable");
        let mode = FastMode::load(&path, true);
        assert!(mode.auto_disable_for_model("claude-4").expect("disable"));
        assert!(!mode.enabled());
        assert_eq!(
            fs::read_to_string(&path).expect("read config after auto-disable"),
            before
        );
    }
}
