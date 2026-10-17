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

    pub(crate) fn fork(&self) -> Arc<Self> {
        Arc::new(Self {
            state: Mutex::new(self.enabled()),
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
    pub(crate) fn commit(self) {
        let mut state = self.mode.state.lock().expect("Fast Mode state poisoned");
        *state = false;
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
    #[test]
    fn toggle_updates_session_fast_mode() {
        let mode = FastMode::load("letcode.toml", false);
        assert!(!mode.enabled());
        assert_eq!(
            mode.toggle("gpt-5.5").expect("enable fast mode"),
            FastModeToggle::Enabled
        );
        assert!(mode.enabled());
    }

    #[test]
    fn unsupported_model_disables_session_fast_mode() {
        let mode = FastMode::load("letcode.toml", true);
        assert!(mode.auto_disable_for_model("claude-4").expect("disable"));
        assert!(!mode.enabled());
    }
}
