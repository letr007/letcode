use anyhow::Result;
use std::sync::{Arc, Mutex};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FastModeTier {
    Fast,
    Ultrafast,
}

impl FastModeTier {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Fast => "fast",
            Self::Ultrafast => "ultrafast",
        }
    }

    pub fn parse(value: &str) -> Option<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "fast" => Some(Self::Fast),
            "ultrafast" => Some(Self::Ultrafast),
            _ => None,
        }
    }
}

#[derive(Debug)]
pub struct FastMode {
    state: Mutex<Option<FastModeTier>>,
}

pub(crate) struct PreparedFastModeDisable {
    mode: Arc<FastMode>,
}

impl FastMode {
    pub fn load(_config_path: impl Into<std::path::PathBuf>, enabled: bool) -> Arc<Self> {
        Arc::new(Self {
            state: Mutex::new(enabled.then_some(FastModeTier::Fast)),
        })
    }

    pub(crate) fn fork(&self) -> Arc<Self> {
        Arc::new(Self {
            state: Mutex::new(self.tier()),
        })
    }

    #[cfg(test)]
    pub fn enabled(&self) -> bool {
        self.tier().is_some()
    }

    pub fn tier(&self) -> Option<FastModeTier> {
        *self.state.lock().expect("Fast Mode state poisoned")
    }

    pub fn set(&self, model_id: &str, tier: Option<FastModeTier>) -> Result<bool> {
        if let Some(tier) = tier
            && !is_tier_capable_model(model_id, tier)
        {
            return Ok(false);
        }
        *self.state.lock().expect("Fast Mode state poisoned") = tier;
        Ok(true)
    }

    pub fn toggle(&self, model_id: &str) -> Result<FastModeToggle> {
        let mut state = self.state.lock().expect("Fast Mode state poisoned");
        let enabled = state.is_none();
        if enabled && !is_fast_capable_model(model_id) {
            return Ok(FastModeToggle::Unavailable);
        }
        *state = enabled.then_some(FastModeTier::Fast);
        Ok(if enabled {
            FastModeToggle::Enabled
        } else {
            FastModeToggle::Disabled
        })
    }

    pub fn auto_disable_for_model(&self, model_id: &str) -> Result<bool> {
        let mut state = self.state.lock().expect("Fast Mode state poisoned");
        if state.is_some_and(|tier| !is_tier_capable_model(model_id, tier)) {
            *state = None;
            return Ok(true);
        }
        Ok(false)
    }

    pub(crate) fn prepare_auto_disable_for_model(
        self: &Arc<Self>,
        model_id: &str,
    ) -> Result<Option<PreparedFastModeDisable>> {
        let state = self.state.lock().expect("Fast Mode state poisoned");
        if state.is_none_or(|tier| is_tier_capable_model(model_id, tier)) {
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
        *state = None;
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FastModeToggle {
    Enabled,
    Disabled,
    Unavailable,
}

pub fn is_tier_capable_model(model_id: &str, tier: FastModeTier) -> bool {
    match tier {
        FastModeTier::Fast => is_fast_capable_model(model_id),
        FastModeTier::Ultrafast => is_ultrafast_capable_model(model_id),
    }
}

pub fn is_ultrafast_capable_model(model_id: &str) -> bool {
    matches!(
        model_id.trim().to_ascii_lowercase().as_str(),
        "gpt-6.2-astra" | "gpt-6.2-sol" | "gpt-6-astra" | "gpt-6.1-sol"
    )
}

pub fn is_fast_capable_model(model_id: &str) -> bool {
    let id = model_id.trim().to_ascii_lowercase();
    id.starts_with("gpt-") && !id.starts_with("gpt-image")
}

#[cfg(test)]
#[path = "../tests/internal/fast_mode/tests.rs"]
mod tests;
