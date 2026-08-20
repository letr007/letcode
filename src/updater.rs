use std::io::{self, IsTerminal, Read, Write};
use std::path::{Path, PathBuf};
use std::time::Duration;

use anyhow::{Context, Result, anyhow, bail, ensure};
use reqwest::blocking::Client;
use serde::Deserialize;
use sha2::{Digest, Sha256};

const REPO_OWNER: &str = "letr007";
const REPO_NAME: &str = "letcode";
const CURRENT_VERSION: &str = env!("CARGO_PKG_VERSION");
const API_TIMEOUT: Duration = Duration::from_secs(10);
const DOWNLOAD_TIMEOUT: Duration = Duration::from_secs(120);
const MAX_RELEASE_BYTES: u64 = 128 * 1024 * 1024;
const LABEL_WIDTH: usize = 14;

const RESET: &str = "\x1b[0m";
const BOLD: &str = "\x1b[1m";
const DIM: &str = "\x1b[2m";
const CYAN: &str = "\x1b[38;5;81m";
const BLUE: &str = "\x1b[38;5;75m";
const GREEN: &str = "\x1b[38;5;114m";
const AMBER: &str = "\x1b[38;5;179m";
const MAGENTA: &str = "\x1b[38;5;176m";

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AvailableUpdate {
    pub current_version: String,
    pub latest_version: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct UpdateTarget {
    target: &'static str,
    display: &'static str,
    extension: &'static str,
    archive_binary: &'static str,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct UpdatePlan {
    current_version: String,
    latest_version: String,
    target: UpdateTarget,
    asset_name: String,
    download_url: String,
    size: u64,
    sha256: String,
}

#[derive(Debug, Deserialize)]
struct GithubRelease {
    tag_name: String,
    #[serde(default)]
    draft: bool,
    #[serde(default)]
    prerelease: bool,
    assets: Vec<GithubAsset>,
}

#[derive(Debug, Deserialize)]
struct GithubAsset {
    name: String,
    browser_download_url: String,
    size: u64,
    digest: Option<String>,
}

struct TerminalUi {
    color: bool,
    dynamic: bool,
}

impl TerminalUi {
    fn stdout() -> Self {
        Self {
            color: io::stdout().is_terminal(),
            dynamic: io::stdout().is_terminal(),
        }
    }

    fn paint(&self, value: &str, codes: &[&str]) -> String {
        if self.color {
            format!("{}{value}{RESET}", codes.concat())
        } else {
            value.to_string()
        }
    }

    fn heading(&self, section: &str) {
        println!();
        println!(
            "{}{}",
            self.paint("letcode", &[BOLD, CYAN]),
            self.paint(&format!(" / {section}"), &[DIM])
        );
        println!("{}", self.paint(&"─".repeat(56), &[DIM]));
    }

    fn field(&self, label: &str, value: impl std::fmt::Display) {
        print!("{}", self.paint(&format!("{label:<LABEL_WIDTH$}"), &[DIM]));
        println!("{value}");
    }

    fn success(&self, label: &str, detail: &str) {
        if self.dynamic {
            print!("\r\x1b[2K");
        }
        print!("{} {label}", self.paint("✓", &[GREEN]));
        if !detail.is_empty() {
            print!("  {}", self.paint(detail, &[DIM]));
        }
        println!();
    }

    fn progress(&self, downloaded: u64, total: u64) -> io::Result<()> {
        if !self.dynamic || total == 0 {
            return Ok(());
        }
        let percent = ((downloaded.saturating_mul(100)) / total).min(100);
        let width = 24usize;
        let filled = ((percent as usize * width) / 100).min(width);
        let bar = if filled == width {
            "━".repeat(width)
        } else if filled == 0 {
            format!("╺{}", "─".repeat(width - 1))
        } else {
            format!("{}╾{}", "━".repeat(filled), "─".repeat(width - filled - 1))
        };
        print!(
            "\r\x1b[2K{} Downloading  {}  {}  {}",
            self.paint("◐", &[AMBER]),
            self.paint(&bar, &[CYAN]),
            self.paint(&format!("{percent:>3}%"), &[AMBER]),
            format_bytes(downloaded),
        );
        io::stdout().flush()
    }
}

pub fn run(check_only: bool) -> Result<()> {
    let target = current_target()?;
    let plan = fetch_update_plan(target)?;
    if check_only {
        show_check(plan.as_ref());
        return Ok(());
    }
    let Some(plan) = plan else {
        let ui = TerminalUi::stdout();
        ui.heading("update");
        ui.field("Current", ui.paint(CURRENT_VERSION, &[GREEN]));
        ui.field("Latest", ui.paint(CURRENT_VERSION, &[GREEN]));
        ui.field("Status", ui.paint("up to date", &[GREEN]));
        return Ok(());
    };
    install(&plan)
}

pub fn available_update() -> Result<Option<AvailableUpdate>> {
    let target = current_target()?;
    Ok(fetch_update_plan(target)?.map(|plan| AvailableUpdate {
        current_version: plan.current_version,
        latest_version: plan.latest_version,
    }))
}

fn show_check(plan: Option<&UpdatePlan>) {
    let ui = TerminalUi::stdout();
    ui.heading("update check");
    match plan {
        Some(plan) => {
            ui.field("Current", &plan.current_version);
            ui.field("Latest", ui.paint(&plan.latest_version, &[BOLD, GREEN]));
            ui.field("Status", ui.paint("update available", &[AMBER]));
            println!();
            println!("Run {} to install.", ui.paint("`letcode update`", &[CYAN]));
        }
        None => {
            ui.field("Current", ui.paint(CURRENT_VERSION, &[GREEN]));
            ui.field("Latest", ui.paint(CURRENT_VERSION, &[GREEN]));
            ui.field("Status", ui.paint("up to date", &[GREEN]));
        }
    }
}

fn install(plan: &UpdatePlan) -> Result<()> {
    let ui = TerminalUi::stdout();
    ui.heading("update");
    println!("{}", ui.paint("UPDATE AVAILABLE", &[BOLD, AMBER]));
    println!();
    ui.field("Current", &plan.current_version);
    ui.field(
        "Release",
        format!(
            "{}  {}  {}",
            ui.paint(&plan.current_version, &[DIM]),
            ui.paint("→", &[CYAN]),
            ui.paint(&plan.latest_version, &[BOLD, GREEN])
        ),
    );
    ui.field("Target", ui.paint(plan.target.display, &[MAGENTA]));
    ui.field("Package", format_bytes(plan.size));
    ui.field("Asset", ui.paint(&plan.asset_name, &[BLUE]));
    println!();

    if !confirm_install(&plan.latest_version)? {
        println!();
        println!("{}", ui.paint(&"-".repeat(56), &[DIM]));
        println!("{}", ui.paint("UPDATE CANCELLED", &[BOLD, AMBER]));
        println!();
        ui.field("Version", format!("remains {}", plan.current_version));
        return Ok(());
    }

    println!();
    println!(
        "{}",
        ui.paint(&format!("Installing {}", plan.latest_version), &[BOLD])
    );
    println!();

    let install_path = std::env::current_exe().context("failed to locate current executable")?;
    ensure_install_location_writable(&install_path)?;
    ui.success("Install location writable", "current executable");

    let temp_dir = tempfile::Builder::new()
        .prefix("letcode-update-")
        .tempdir_in(
            install_path
                .parent()
                .ok_or_else(|| anyhow!("current executable has no parent directory"))?,
        )
        .context("failed to create update staging directory")?;
    let archive_path = temp_dir.path().join(&plan.asset_name);
    download_release(plan, &archive_path, &ui)?;
    verify_sha256(&archive_path, &plan.sha256)?;
    ui.success("Release digest verified", "SHA-256");

    let archive_binary = render_archive_binary(plan.target.archive_binary, plan);
    self_update::Extract::from_source(&archive_path)
        .extract_file(temp_dir.path(), &archive_binary)
        .context("failed to extract release executable")?;
    let new_executable = temp_dir.path().join(&archive_binary);
    ensure!(
        new_executable.is_file(),
        "release executable was not extracted"
    );
    ui.success("Executable prepared", plan.target.display);

    self_update::self_replace::self_replace(&new_executable)
        .context("failed to replace current executable")?;
    ui.success("Binary replaced", "atomic install");

    println!();
    println!("{}", ui.paint(&"-".repeat(56), &[DIM]));
    println!("{}", ui.paint("UPDATE COMPLETE", &[BOLD, GREEN]));
    println!();
    ui.field(
        "Installed",
        format!(
            "{}  {}  {}",
            ui.paint(&plan.current_version, &[DIM]),
            ui.paint("→", &[CYAN]),
            ui.paint(&plan.latest_version, &[BOLD, GREEN])
        ),
    );
    ui.field("Target", plan.target.display);
    Ok(())
}

fn fetch_update_plan(target: UpdateTarget) -> Result<Option<UpdatePlan>> {
    let url = format!("https://api.github.com/repos/{REPO_OWNER}/{REPO_NAME}/releases/latest");
    let release = Client::builder()
        .timeout(API_TIMEOUT)
        .user_agent(format!("letcode/{CURRENT_VERSION}"))
        .build()
        .context("failed to build GitHub client")?
        .get(url)
        .send()
        .context("failed to query the latest GitHub release")?
        .error_for_status()
        .context("GitHub release request failed")?
        .json::<GithubRelease>()
        .context("failed to parse GitHub release")?;
    plan_from_release(release, target)
}

fn plan_from_release(release: GithubRelease, target: UpdateTarget) -> Result<Option<UpdatePlan>> {
    ensure!(!release.draft, "latest GitHub release is a draft");
    ensure!(!release.prerelease, "latest GitHub release is a prerelease");
    let latest_version = release.tag_name.trim_start_matches('v').to_string();
    if !self_update::version::bump_is_greater(CURRENT_VERSION, &latest_version).with_context(
        || format!("failed to compare versions {CURRENT_VERSION} and {latest_version}"),
    )? {
        return Ok(None);
    }

    let expected_name = asset_name(&latest_version, target);
    let asset = release
        .assets
        .into_iter()
        .find(|asset| asset.name == expected_name)
        .with_context(|| format!("release {latest_version} has no asset named {expected_name}"))?;
    ensure!(asset.size > 0, "release asset is empty");
    ensure!(
        asset.size <= MAX_RELEASE_BYTES,
        "release asset exceeds 128 MiB limit"
    );
    let digest = asset
        .digest
        .with_context(|| format!("release asset {} has no digest", asset.name))?;
    let sha256 = parse_sha256_digest(&digest)?;

    Ok(Some(UpdatePlan {
        current_version: CURRENT_VERSION.to_string(),
        latest_version,
        target,
        asset_name: asset.name,
        download_url: asset.browser_download_url,
        size: asset.size,
        sha256,
    }))
}

fn download_release(plan: &UpdatePlan, path: &Path, ui: &TerminalUi) -> Result<()> {
    let mut response = Client::builder()
        .timeout(DOWNLOAD_TIMEOUT)
        .user_agent(format!("letcode/{CURRENT_VERSION}"))
        .build()
        .context("failed to build release download client")?
        .get(&plan.download_url)
        .send()
        .context("failed to download release asset")?
        .error_for_status()
        .context("release asset download failed")?;
    if let Some(length) = response.content_length() {
        ensure!(
            length <= MAX_RELEASE_BYTES,
            "release download exceeds 128 MiB limit"
        );
    }

    let mut output = std::fs::File::create(path).context("failed to create release archive")?;
    let mut buffer = [0u8; 64 * 1024];
    let mut downloaded = 0u64;
    loop {
        let read = response
            .read(&mut buffer)
            .context("failed while reading release download")?;
        if read == 0 {
            break;
        }
        downloaded = downloaded.saturating_add(read as u64);
        ensure!(
            downloaded <= MAX_RELEASE_BYTES,
            "release download exceeds 128 MiB limit"
        );
        output
            .write_all(&buffer[..read])
            .context("failed while writing release archive")?;
        ui.progress(downloaded, plan.size)?;
    }
    output.flush().context("failed to flush release archive")?;
    ensure!(downloaded == plan.size, "release download size mismatch");
    ui.success("Package downloaded", &format_bytes(downloaded));
    Ok(())
}

fn verify_sha256(path: &Path, expected: &str) -> Result<()> {
    let mut file = std::fs::File::open(path).context("failed to open release archive")?;
    let mut digest = Sha256::new();
    let mut buffer = [0u8; 64 * 1024];
    loop {
        let read = file
            .read(&mut buffer)
            .context("failed to hash release archive")?;
        if read == 0 {
            break;
        }
        digest.update(&buffer[..read]);
    }
    let actual = format!("{:x}", digest.finalize());
    ensure!(actual == expected, "release SHA-256 digest mismatch");
    Ok(())
}

fn confirm_install(version: &str) -> Result<bool> {
    print!("Install release {version}? [y/N] ");
    io::stdout().flush()?;
    let mut answer = String::new();
    io::stdin().read_line(&mut answer)?;
    Ok(matches!(
        answer.trim().to_ascii_lowercase().as_str(),
        "y" | "yes"
    ))
}

fn ensure_install_location_writable(path: &Path) -> Result<()> {
    let parent = path
        .parent()
        .ok_or_else(|| anyhow!("current executable has no parent directory"))?;
    ensure!(
        parent.is_dir(),
        "current executable directory does not exist"
    );
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        ensure!(
            parent.metadata()?.permissions().mode() & 0o200 != 0,
            "current executable directory is not writable"
        );
    }
    Ok(())
}

fn render_archive_binary(template: &str, plan: &UpdatePlan) -> PathBuf {
    PathBuf::from(
        template
            .replace("{{ version }}", &plan.latest_version)
            .replace("{{ target }}", plan.target.target),
    )
}

fn parse_sha256_digest(value: &str) -> Result<String> {
    let digest = value
        .strip_prefix("sha256:")
        .context("release asset digest is not SHA-256")?;
    ensure!(
        digest.len() == 64 && digest.bytes().all(|byte| byte.is_ascii_hexdigit()),
        "release asset SHA-256 digest is invalid"
    );
    Ok(digest.to_ascii_lowercase())
}

fn asset_name(version: &str, target: UpdateTarget) -> String {
    format!("letcode-{version}-{}{}", target.target, target.extension)
}

fn format_bytes(bytes: u64) -> String {
    const MIB: f64 = 1024.0 * 1024.0;
    format!("{:.2} MiB", bytes as f64 / MIB)
}

fn current_target() -> Result<UpdateTarget> {
    target_for(std::env::consts::OS, std::env::consts::ARCH)
}

fn target_for(os: &str, arch: &str) -> Result<UpdateTarget> {
    match (os, arch) {
        ("macos", "aarch64") => Ok(UpdateTarget {
            target: "aarch64-apple-darwin",
            display: "macOS · Apple Silicon",
            extension: ".tar.gz",
            archive_binary: "letcode-{{ version }}-{{ target }}/letcode",
        }),
        ("windows", "x86_64") => Ok(UpdateTarget {
            target: "x86_64-pc-windows-msvc",
            display: "Windows · x86-64",
            extension: ".zip",
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
        let macos = target_for("macos", "aarch64").expect("macOS ARM64 should be supported");
        assert_eq!(
            asset_name("1.2.3", macos),
            "letcode-1.2.3-aarch64-apple-darwin.tar.gz"
        );
        assert_eq!(
            macos.archive_binary,
            "letcode-{{ version }}-{{ target }}/letcode"
        );
        let windows = target_for("windows", "x86_64").expect("Windows x64 should be supported");
        assert_eq!(
            asset_name("1.2.3", windows),
            "letcode-1.2.3-x86_64-pc-windows-msvc.zip"
        );
    }

    #[test]
    fn sha256_digest_requires_exact_algorithm_and_length() {
        assert_eq!(
            parse_sha256_digest(&format!("sha256:{}", "a".repeat(64))).expect("valid digest"),
            "a".repeat(64)
        );
        assert!(parse_sha256_digest(&format!("sha512:{}", "a".repeat(64))).is_err());
        assert!(parse_sha256_digest("sha256:abcd").is_err());
    }

    #[test]
    fn release_plan_requires_the_exact_asset_and_github_digest() {
        let target = target_for("macos", "aarch64").expect("supported target");
        let asset = asset_name("99.0.0", target);
        let plan = plan_from_release(
            GithubRelease {
                tag_name: "v99.0.0".into(),
                draft: false,
                prerelease: false,
                assets: vec![GithubAsset {
                    name: asset.clone(),
                    browser_download_url: "https://example.invalid/release".into(),
                    size: 42,
                    digest: Some(format!("sha256:{}", "b".repeat(64))),
                }],
            },
            target,
        )
        .expect("release should produce a plan")
        .expect("release should be newer");

        assert_eq!(plan.asset_name, asset);
        assert_eq!(plan.sha256, "b".repeat(64));
    }

    #[test]
    fn archive_binary_template_uses_release_version_and_target() {
        let target = target_for("macos", "aarch64").expect("supported target");
        let plan = UpdatePlan {
            current_version: "0.5.2".into(),
            latest_version: "0.5.3".into(),
            target,
            asset_name: asset_name("0.5.3", target),
            download_url: "https://example.invalid/release".into(),
            size: 1,
            sha256: "a".repeat(64),
        };
        assert_eq!(
            render_archive_binary(target.archive_binary, &plan),
            PathBuf::from("letcode-0.5.3-aarch64-apple-darwin/letcode")
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
