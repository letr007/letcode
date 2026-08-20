use anyhow::{Context, Result, bail};
use self_update::update::ReleaseUpdate;

const REPO_OWNER: &str = "letr007";
const REPO_NAME: &str = "letcode";
const BIN_NAME: &str = "letcode";
const CURRENT_VERSION: &str = env!("CARGO_PKG_VERSION");

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct UpdateTarget {
    target: &'static str,
    archive_binary: &'static str,
}

pub fn run(check_only: bool) -> Result<()> {
    let target = current_target()?;
    if check_only {
        check(target)
    } else {
        install(target)
    }
}

fn check(target: UpdateTarget) -> Result<()> {
    let updater = build_updater(target, false)?;
    let release = updater
        .get_latest_release()
        .context("failed to query the latest GitHub release")?;
    let latest = release.version.as_str();
    let available = self_update::version::bump_is_greater(CURRENT_VERSION, latest)
        .with_context(|| format!("failed to compare versions {CURRENT_VERSION} and {latest}"))?;

    println!("Current version: {CURRENT_VERSION}");
    println!("Latest version: {latest}");
    if available {
        println!("Update available: run `letcode update`");
    } else {
        println!("letcode is up to date");
    }
    Ok(())
}

fn install(target: UpdateTarget) -> Result<()> {
    let status = build_updater(target, true)?
        .update()
        .context("failed to update letcode")?;

    if status.updated() {
        println!("letcode was updated to {}", status.version());
    } else {
        println!("letcode {} is already up to date", status.version());
    }
    Ok(())
}

fn build_updater(
    target: UpdateTarget,
    show_download_progress: bool,
) -> Result<Box<dyn ReleaseUpdate>> {
    let mut builder = self_update::backends::github::Update::configure();
    builder
        .repo_owner(REPO_OWNER)
        .repo_name(REPO_NAME)
        .bin_name(BIN_NAME)
        .target(target.target)
        .identifier(target.target)
        .bin_path_in_archive(target.archive_binary)
        .show_download_progress(show_download_progress)
        .show_output(show_download_progress)
        .current_version(CURRENT_VERSION);
    builder.build().context("failed to configure updater")
}

fn current_target() -> Result<UpdateTarget> {
    target_for(std::env::consts::OS, std::env::consts::ARCH)
}

fn target_for(os: &str, arch: &str) -> Result<UpdateTarget> {
    match (os, arch) {
        ("macos", "aarch64") => Ok(UpdateTarget {
            target: "aarch64-apple-darwin",
            archive_binary: "letcode-{{ version }}-{{ target }}/letcode",
        }),
        ("windows", "x86_64") => Ok(UpdateTarget {
            target: "x86_64-pc-windows-msvc",
            archive_binary: "letcode-{{ version }}-{{ target }}/letcode.exe",
        }),
        _ => bail!("self-update is not supported on {os}/{arch}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn release_targets_match_the_packaging_workflow() {
        assert_eq!(
            target_for("macos", "aarch64").expect("macOS ARM64 should be supported"),
            UpdateTarget {
                target: "aarch64-apple-darwin",
                archive_binary: "letcode-{{ version }}-{{ target }}/letcode",
            }
        );
        assert_eq!(
            target_for("windows", "x86_64").expect("Windows x64 should be supported"),
            UpdateTarget {
                target: "x86_64-pc-windows-msvc",
                archive_binary: "letcode-{{ version }}-{{ target }}/letcode.exe",
            }
        );
    }

    #[test]
    fn unsupported_targets_fail_explicitly() {
        let error = target_for("linux", "x86_64").expect_err("Linux is not released yet");
        assert_eq!(
            error.to_string(),
            "self-update is not supported on linux/x86_64"
        );
    }
}
