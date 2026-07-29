use anyhow::{Context, Result, anyhow, bail};
use serde::{Deserialize, Serialize};
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

const FAST_MODE_STATE_FILE: &str = "fast-mode.json";
const FAST_MODE_STATE_VERSION: u8 = 1;
static FAST_MODE_WRITE_SEQUENCE: AtomicU64 = AtomicU64::new(0);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FastModeState {
    pub version: u8,
    pub enabled: bool,
}

impl Default for FastModeState {
    fn default() -> Self {
        Self {
            version: FAST_MODE_STATE_VERSION,
            enabled: false,
        }
    }
}

impl FastModeState {
    fn validate(self) -> Result<Self> {
        if self.version != FAST_MODE_STATE_VERSION {
            bail!(
                "invalid Fast Mode state: version must be {FAST_MODE_STATE_VERSION}, got {}",
                self.version
            );
        }
        Ok(self)
    }
}

#[derive(Debug)]
pub struct FastMode {
    config_dir: PathBuf,
    state: Mutex<FastModeState>,
}

impl FastMode {
    pub fn load(config_dir: impl Into<PathBuf>) -> Result<Arc<Self>> {
        let config_dir = config_dir.into();
        let state = load_state(&state_path(&config_dir))?;
        Ok(Arc::new(Self {
            config_dir,
            state: Mutex::new(state),
        }))
    }

    pub fn enabled(&self) -> bool {
        self.state.lock().expect("Fast Mode state poisoned").enabled
    }

    pub fn toggle(&self, model_id: &str) -> Result<FastModeToggle> {
        let mut state = self.state.lock().expect("Fast Mode state poisoned");
        let enabled = !state.enabled;
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
        if state.enabled && !is_fast_capable_model(model_id) {
            self.set_enabled_locked(&mut state, false)?;
            return Ok(true);
        }
        Ok(false)
    }

    fn set_enabled_locked(&self, current: &mut FastModeState, enabled: bool) -> Result<()> {
        let state = FastModeState {
            version: FAST_MODE_STATE_VERSION,
            enabled,
        };
        write_state(&state_path(&self.config_dir), state)?;
        *current = state;
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

fn state_path(config_dir: &Path) -> PathBuf {
    config_dir.join(FAST_MODE_STATE_FILE)
}

fn load_state(path: &Path) -> Result<FastModeState> {
    match fs::read_to_string(path) {
        Ok(contents) => serde_json::from_str::<FastModeState>(&contents)
            .context(
                "invalid Fast Mode state: exactly {\"version\":1,\"enabled\":boolean} is required",
            )?
            .validate(),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(FastModeState::default()),
        Err(error) => {
            Err(error).with_context(|| format!("failed to read Fast Mode state {}", path.display()))
        }
    }
}

fn write_state(path: &Path, state: FastModeState) -> Result<()> {
    let parent = path.parent().unwrap_or_else(|| Path::new("."));
    fs::create_dir_all(parent).with_context(|| {
        format!(
            "failed to create Fast Mode config directory {}",
            parent.display()
        )
    })?;
    let contents = format!("{}\n", serde_json::to_string(&state)?);
    let temp_path = parent.join(format!(
        ".{}.{}.{}.tmp",
        path.file_name()
            .ok_or_else(|| anyhow!("Fast Mode state path has no file name: {}", path.display()))?
            .to_string_lossy(),
        std::process::id(),
        FAST_MODE_WRITE_SEQUENCE.fetch_add(1, Ordering::Relaxed),
    ));
    let result = (|| -> Result<()> {
        let mut temp = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&temp_path)
            .with_context(|| {
                format!(
                    "failed to create temporary Fast Mode state {}",
                    temp_path.display()
                )
            })?;
        temp.write_all(contents.as_bytes()).with_context(|| {
            format!(
                "failed to write temporary Fast Mode state {}",
                temp_path.display()
            )
        })?;
        temp.sync_all().with_context(|| {
            format!(
                "failed to sync temporary Fast Mode state {}",
                temp_path.display()
            )
        })?;
        fs::rename(&temp_path, path).with_context(|| {
            format!(
                "failed to atomically replace Fast Mode state {} with {}",
                path.display(),
                temp_path.display()
            )
        })?;
        Ok(())
    })();
    if result.is_err() {
        let _ = fs::remove_file(&temp_path);
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_dir(name: &str) -> PathBuf {
        std::env::temp_dir().join(format!(
            "letcode-fast-mode-{name}-{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("time ok")
                .as_nanos()
        ))
    }

    #[test]
    fn missing_state_defaults_disabled_and_toggle_persists() {
        let dir = temp_dir("persist");
        let mode = FastMode::load(&dir).expect("load missing state");
        assert!(!mode.enabled());
        assert_eq!(
            mode.toggle("gpt-5.5").expect("enable fast mode"),
            FastModeToggle::Enabled
        );
        assert!(FastMode::load(&dir).expect("reload state").enabled());
    }

    #[test]
    fn malformed_state_fails_instead_of_defaulting() {
        let dir = temp_dir("malformed");
        fs::create_dir_all(&dir).expect("create state dir");
        fs::write(
            state_path(&dir),
            "{\"version\":1,\"enabled\":true,\"extra\":false}",
        )
        .expect("write malformed state");
        assert!(FastMode::load(&dir).is_err());
    }

    #[test]
    fn gpt_models_except_image_are_fast_capable() {
        assert!(is_fast_capable_model("GPT-5.5"));
        assert!(is_fast_capable_model("gpt-4.1-mini"));
        assert!(!is_fast_capable_model("gpt-image-1"));
        assert!(!is_fast_capable_model("claude-4"));
    }

    #[test]
    fn unsupported_model_auto_disables_persisted_state() {
        let dir = temp_dir("auto-disable");
        let mode = FastMode::load(&dir).expect("load state");
        mode.toggle("gpt-5.5").expect("enable");
        assert!(mode.auto_disable_for_model("claude-4").expect("disable"));
        assert!(!FastMode::load(&dir).expect("reload state").enabled());
    }

    #[test]
    fn concurrent_transitions_keep_disk_and_memory_in_sync() {
        use std::thread;

        let dir = temp_dir("concurrent-transitions");
        let mode = FastMode::load(&dir).expect("load state");
        let mut workers = Vec::new();
        for _ in 0..16 {
            let mode = Arc::clone(&mode);
            workers.push(thread::spawn(move || {
                for _ in 0..32 {
                    mode.toggle("gpt-5.5").expect("toggle fast mode");
                    mode.auto_disable_for_model("claude-4")
                        .expect("auto-disable fast mode");
                }
            }));
        }
        for worker in workers {
            worker.join().expect("worker completes");
        }

        let persisted = FastMode::load(&dir).expect("reload state");
        assert_eq!(mode.enabled(), persisted.enabled());
    }
}
