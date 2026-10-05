use anyhow::{Context, Result, bail};
use std::path::Path;

use crate::config::AppConfig;

/// Offer an interactive repair loop when the configuration file fails to load.
///
/// The TUI has not taken over the terminal yet, so an editor can run directly.
pub(crate) fn run(config_path: &Path, error: &anyhow::Error) -> Result<AppConfig> {
    let mut message = format!("{error:#}");
    loop {
        eprintln!("Configuration is invalid:\n{message}\n");
        eprintln!(
            "Press Enter to open {} in $EDITOR, or type q to quit:",
            config_path.display()
        );
        let mut answer = String::new();
        let read = std::io::stdin()
            .read_line(&mut answer)
            .context("failed to read the repair prompt")?;
        if read == 0 || answer.trim().eq_ignore_ascii_case("q") {
            bail!("configuration is invalid: {message}");
        }
        open_in_editor(config_path)?;
        match AppConfig::load_from_path(config_path) {
            Ok(config) => return Ok(config),
            Err(next) => message = format!("{next:#}"),
        }
    }
}

fn open_in_editor(config_path: &Path) -> Result<()> {
    let editor = std::env::var("VISUAL")
        .or_else(|_| std::env::var("EDITOR"))
        .unwrap_or_else(|_| "vi".to_string());
    let status = std::process::Command::new(&editor)
        .arg(config_path)
        .status()
        .with_context(|| format!("failed to launch editor {editor}"))?;
    if !status.success() {
        bail!("editor {editor} exited with {status}");
    }
    Ok(())
}
