use std::collections::HashMap;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};

pub(crate) type Values = HashMap<String, String>;

pub(crate) struct Map {
    path: PathBuf,
    values: Values,
}

impl Map {
    pub(crate) fn open(sessions_dir: &Path, session_id: &str) -> Self {
        let path = sessions_dir.join("maps").join(format!("{session_id}.json"));
        let values = std::fs::read(&path)
            .ok()
            .and_then(|bytes| serde_json::from_slice(&bytes).ok())
            .unwrap_or_default();
        Self { path, values }
    }

    pub(crate) fn values(&self) -> &Values {
        &self.values
    }

    pub(crate) fn add(&mut self, values: &Values) -> Result<()> {
        let mut changed = false;
        for (placeholder, value) in values {
            if self.values.get(placeholder) != Some(value) {
                self.values.insert(placeholder.clone(), value.clone());
                changed = true;
            }
        }
        if changed {
            self.write()?;
        }
        Ok(())
    }

    fn write(&self) -> Result<()> {
        let parent = self
            .path
            .parent()
            .context("the session map has no directory")?;
        std::fs::create_dir_all(parent)
            .with_context(|| format!("creating {}", parent.display()))?;
        let temporary = self.path.with_extension("json.tmp");
        write_private(&temporary, &serde_json::to_vec(&self.values)?)
            .with_context(|| format!("writing {}", temporary.display()))?;
        std::fs::rename(&temporary, &self.path)
            .with_context(|| format!("replacing {}", self.path.display()))
    }
}

fn write_private(path: &Path, bytes: &[u8]) -> std::io::Result<()> {
    let mut options = std::fs::OpenOptions::new();
    options.write(true).create(true).truncate(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(0o600);
    }
    use std::io::Write;
    options.open(path)?.write_all(bytes)
}
