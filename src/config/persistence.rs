use anyhow::{Context, Result, anyhow, bail};
use std::fs;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use toml_edit::{Array, DocumentMut, Item, Table, value};

#[cfg(unix)]
use std::os::fd::AsRawFd;
#[cfg(unix)]
use std::os::unix::fs::{MetadataExt, OpenOptionsExt};

use super::{
    AppConfig, McpServerConfig, ModelRoute, RawAppConfig, RawMcpServerConfig,
    build_mcp_server_config,
};

static CONFIG_WRITE_SEQUENCE: AtomicU64 = AtomicU64::new(0);

/// Persist one expert's complete provider-qualified allowlist.
pub fn persist_expert_allowed_models(
    config_path: &Path,
    agent_name: &str,
    routes: &[ModelRoute],
) -> Result<()> {
    if !crate::delegation::supported_agent_names().any(|name| name == agent_name) {
        bail!("unknown expert '{agent_name}'");
    }

    persist_config_document(config_path, "expert allowed models", true, |document| {
        let agents = document["agents"].or_insert(Item::Table(Table::new()));
        let agents = agents
            .as_table_mut()
            .ok_or_else(|| anyhow!("config [agents] entry is not a table"))?;
        let agent = agents
            .entry(agent_name)
            .or_insert(Item::Table(Table::new()))
            .as_table_mut()
            .ok_or_else(|| anyhow!("agents.{agent_name} is not a configured table"))?;
        let mut allowed_models = Array::default();
        for route in routes {
            allowed_models.push(route.display_name());
        }
        agent.insert("allowed_models", Item::Value(allowed_models.into()));
        Ok(())
    })
}

fn persist_config_document(
    config_path: &Path,
    update_name: &str,
    require_full_config: bool,
    edit: impl FnOnce(&mut DocumentMut) -> Result<()>,
) -> Result<()> {
    // Resolve before editing so a config symlink remains a symlink while its
    // existing target is atomically replaced.
    let config_target = fs::canonicalize(config_path)
        .with_context(|| format!("failed to resolve config file {}", config_path.display()))?;
    // The lock lives beside the canonical target, so it survives the target's
    // atomic replacement and serializes all cooperating letcode writers.
    let _lock = acquire_config_lock(&config_target)?;
    let mut config_file = fs::File::open(&config_target)
        .with_context(|| format!("failed to open config file {}", config_target.display()))?;
    let original_metadata = config_file
        .metadata()
        .with_context(|| format!("failed to stat config file {}", config_target.display()))?;
    let mut config_text = String::new();
    config_file
        .read_to_string(&mut config_text)
        .with_context(|| format!("failed to read config file {}", config_target.display()))?;
    let mut document = config_text
        .parse::<DocumentMut>()
        .with_context(|| format!("failed to parse config file {}", config_target.display()))?;
    edit(&mut document)?;

    let updated_config = document.to_string();
    validate_updated_config_document(
        &config_target,
        &updated_config,
        update_name,
        require_full_config,
    )?;
    atomic_write_config_with_source(
        &config_target,
        &config_text,
        &original_metadata,
        updated_config.as_bytes(),
        Some(&config_file),
    )
}

fn validate_updated_config_document(
    config_path: &Path,
    config_text: &str,
    update_name: &str,
    require_full_config: bool,
) -> Result<()> {
    let raw: RawAppConfig = toml::from_str(config_text)
        .with_context(|| format!("failed to parse config file {}", config_path.display()))?;

    if raw.providers.is_empty() && !require_full_config {
        return Ok(());
    }

    AppConfig::load_from_str_at_path(config_path, config_text)
        .with_context(|| format!("failed to validate updated {update_name} configuration"))?;
    Ok(())
}

/// Persist one configured MCP server's enabled state without rewriting unrelated
/// configuration content.
pub fn persist_mcp_server_enabled(
    config_path: &Path,
    server_name: &str,
    enabled: bool,
) -> Result<McpServerConfig> {
    let mut persisted_server = None;
    persist_config_document(config_path, "MCP server state", false, |document| {
        let mcp = document
            .get_mut("mcp")
            .and_then(Item::as_table_like_mut)
            .ok_or_else(|| anyhow!("config does not define an [mcp] table"))?;
        let server = mcp
            .get_mut(server_name)
            .and_then(Item::as_table_like_mut)
            .ok_or_else(|| anyhow!("MCP server '{server_name}' is not a configured table"))?;
        server.insert("enabled", value(enabled));

        let raw_document = toml::from_str::<toml::Value>(&document.to_string())
            .context("failed to parse updated MCP server configuration")?;
        let raw_server = raw_document
            .get("mcp")
            .and_then(toml::Value::as_table)
            .and_then(|mcp| mcp.get(server_name))
            .cloned()
            .ok_or_else(|| anyhow!("updated MCP server '{server_name}' is missing"))?
            .try_into::<RawMcpServerConfig>()
            .with_context(|| format!("failed to parse MCP server '{server_name}'"))?;
        persisted_server = Some(build_mcp_server_config(server_name, raw_server)?.1);
        Ok(())
    })?;
    Ok(persisted_server.expect("MCP server persistence validates and stores the server"))
}

#[cfg(test)]
pub(super) fn atomic_write_config(
    config_path: &Path,
    original_contents: &str,
    original_metadata: &fs::Metadata,
    contents: &[u8],
) -> Result<()> {
    atomic_write_config_with_source(
        config_path,
        original_contents,
        original_metadata,
        contents,
        None,
    )
}

fn atomic_write_config_with_source(
    config_path: &Path,
    original_contents: &str,
    original_metadata: &fs::Metadata,
    contents: &[u8],
    source_file: Option<&fs::File>,
) -> Result<()> {
    let parent = config_path.parent().unwrap_or_else(|| Path::new("."));
    let file_name = config_path
        .file_name()
        .ok_or_else(|| anyhow!("config path has no file name: {}", config_path.display()))?;
    let temp_path = parent.join(format!(
        ".{}.{}.{}.tmp",
        file_name.to_string_lossy(),
        std::process::id(),
        CONFIG_WRITE_SEQUENCE.fetch_add(1, Ordering::Relaxed)
    ));
    let write_result = (|| -> Result<()> {
        let mut temp = create_config_temp_file(&temp_path, original_metadata)?;
        temp.write_all(contents).with_context(|| {
            format!(
                "failed to write temporary config file {}",
                temp_path.display()
            )
        })?;
        temp.sync_all().with_context(|| {
            format!(
                "failed to sync temporary config file {}",
                temp_path.display()
            )
        })?;
        drop(temp);
        revalidate_config_source(
            config_path,
            original_contents,
            original_metadata,
            source_file,
        )?;
        replace_file(&temp_path, config_path).with_context(|| {
            format!(
                "failed to atomically replace config file {} with {}",
                config_path.display(),
                temp_path.display()
            )
        })?;
        Ok(())
    })();
    if write_result.is_err() && temp_path.exists() {
        let _ = fs::remove_file(&temp_path);
    }
    write_result
}

fn create_config_temp_file(temp_path: &Path, source_metadata: &fs::Metadata) -> Result<fs::File> {
    let mut options = fs::OpenOptions::new();
    options.write(true).create_new(true);
    #[cfg(unix)]
    {
        // `mode` is applied by the creation syscall, before the path can be
        // observed. The umask may make it more restrictive, never broader.
        options.mode(source_metadata.mode() & 0o777);
    }
    let temp = options.open(temp_path).with_context(|| {
        format!(
            "failed to create temporary config file {}",
            temp_path.display()
        )
    })?;
    // Restore the exact original permissions when a restrictive umask changed
    // them. This occurs only after safe initial creation.
    temp.set_permissions(source_metadata.permissions())
        .with_context(|| {
            format!(
                "failed to preserve config permissions for {}",
                temp_path.display()
            )
        })?;
    Ok(temp)
}

fn revalidate_config_source(
    config_path: &Path,
    original_contents: &str,
    original_metadata: &fs::Metadata,
    source_file: Option<&fs::File>,
) -> Result<()> {
    let current_metadata = fs::metadata(config_path)
        .with_context(|| format!("failed to restat config file {}", config_path.display()))?;
    let current_contents = fs::read_to_string(config_path)
        .with_context(|| format!("failed to reread config file {}", config_path.display()))?;
    if current_contents != original_contents
        || !config_metadata_matches(original_metadata, &current_metadata)
        || source_file.is_some_and(|source| !config_source_identity_matches(source, config_path))
    {
        bail!(
            "config file {} changed while updating configuration; refusing to overwrite it",
            config_path.display()
        );
    }
    Ok(())
}

fn config_source_identity_matches(source: &fs::File, path: &Path) -> bool {
    #[cfg(windows)]
    {
        use std::os::windows::io::AsRawHandle;
        use windows_sys::Win32::Storage::FileSystem::{
            BY_HANDLE_FILE_INFORMATION, GetFileInformationByHandle,
        };
        fn identity(handle: std::os::windows::io::RawHandle) -> Option<(u32, u64)> {
            let mut information = BY_HANDLE_FILE_INFORMATION::default();
            let ok = unsafe { GetFileInformationByHandle(handle.cast(), &mut information) };
            (ok != 0).then_some((
                information.dwVolumeSerialNumber,
                ((information.nFileIndexHigh as u64) << 32) | information.nFileIndexLow as u64,
            ))
        }
        let Ok(current) = fs::File::open(path) else {
            return false;
        };
        match (
            identity(source.as_raw_handle()),
            identity(current.as_raw_handle()),
        ) {
            (Some(source), Some(current)) => source == current,
            _ => false,
        }
    }
    #[cfg(not(windows))]
    {
        let _ = (source, path);
        true
    }
}

fn config_metadata_matches(expected: &fs::Metadata, current: &fs::Metadata) -> bool {
    if expected.len() != current.len() || expected.modified().ok() != current.modified().ok() {
        return false;
    }
    #[cfg(unix)]
    {
        expected.dev() == current.dev() && expected.ino() == current.ino()
    }
    #[cfg(not(unix))]
    {
        true
    }
}

pub(super) struct ConfigLock {
    _file: fs::File,
}

pub(crate) fn replace_file(source: &Path, destination: &Path) -> Result<()> {
    match fs::symlink_metadata(destination) {
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            return fs::rename(source, destination).context("failed to install new file");
        }
        Err(error) => return Err(error).context("failed to inspect replacement destination"),
        Ok(_) => {}
    }
    #[cfg(windows)]
    {
        use std::os::windows::ffi::OsStrExt;
        use windows_sys::Win32::Storage::FileSystem::ReplaceFileW;
        let source = source
            .as_os_str()
            .encode_wide()
            .chain(std::iter::once(0))
            .collect::<Vec<_>>();
        let destination = destination
            .as_os_str()
            .encode_wide()
            .chain(std::iter::once(0))
            .collect::<Vec<_>>();
        let ok = unsafe {
            ReplaceFileW(
                destination.as_ptr(),
                source.as_ptr(),
                std::ptr::null(),
                0,
                std::ptr::null(),
                std::ptr::null(),
            )
        };
        if ok == 0 {
            return Err(std::io::Error::last_os_error()).context("failed to replace file");
        }
        Ok(())
    }
    #[cfg(not(windows))]
    {
        fs::rename(source, destination).context("failed to replace file")
    }
}

pub(super) fn acquire_config_lock(config_target: &Path) -> Result<ConfigLock> {
    let lock_path = config_lock_path(config_target)?;
    let file = open_config_lock_file(&lock_path)?;
    lock_file(&file)?;
    Ok(ConfigLock { _file: file })
}

pub(super) fn config_lock_path(config_target: &Path) -> Result<PathBuf> {
    let parent = config_target.parent().unwrap_or_else(|| Path::new("."));
    let file_name = config_target
        .file_name()
        .ok_or_else(|| anyhow!("config path has no file name: {}", config_target.display()))?;
    Ok(parent.join(format!(".{}.lock", file_name.to_string_lossy())))
}

pub(super) fn open_config_lock_file(lock_path: &Path) -> Result<fs::File> {
    #[cfg(unix)]
    {
        let mut create = fs::OpenOptions::new();
        create
            .read(true)
            .write(true)
            .create_new(true)
            .mode(0o600)
            .custom_flags(libc::O_NOFOLLOW);
        match create.open(lock_path) {
            Ok(file) => return Ok(file),
            Err(error) if error.kind() != std::io::ErrorKind::AlreadyExists => {
                return Err(error).with_context(|| {
                    format!("failed to create config lock file {}", lock_path.display())
                });
            }
            Err(_) => {}
        }

        let mut open = fs::OpenOptions::new();
        open.read(true).write(true).custom_flags(libc::O_NOFOLLOW);
        let file = open
            .open(lock_path)
            .with_context(|| format!("failed to open config lock file {}", lock_path.display()))?;
        if !file
            .metadata()
            .with_context(|| format!("failed to stat config lock file {}", lock_path.display()))?
            .is_file()
        {
            bail!(
                "config lock path is not a regular file: {}",
                lock_path.display()
            );
        }
        Ok(file)
    }

    #[cfg(not(unix))]
    {
        fs::OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .open(lock_path)
            .with_context(|| format!("failed to open config lock file {}", lock_path.display()))
    }
}

#[cfg(unix)]
fn lock_file(file: &fs::File) -> Result<()> {
    lock_file_with_mode(file, libc::LOCK_EX)
}

#[cfg(unix)]
fn lock_file_shared(file: &fs::File) -> Result<()> {
    lock_file_with_mode(file, libc::LOCK_SH)
}

#[cfg(unix)]
fn lock_file_with_mode(file: &fs::File, mode: i32) -> Result<()> {
    let result = unsafe { libc::flock(file.as_raw_fd(), mode) };
    if result != 0 {
        return Err(std::io::Error::last_os_error()).context("failed to lock config lock file");
    }
    Ok(())
}

#[cfg(windows)]
fn lock_file(file: &fs::File) -> Result<()> {
    fs4::fs_std::FileExt::lock_exclusive(file).context("failed to lock config lock file")
}

#[cfg(windows)]
fn lock_file_shared(file: &fs::File) -> Result<()> {
    fs4::fs_std::FileExt::lock_shared(file).context("failed to acquire shared config lock")
}

#[cfg(not(any(unix, windows)))]
fn lock_file(_file: &fs::File) -> Result<()> {
    Ok(())
}

#[cfg(not(any(unix, windows)))]
fn lock_file_shared(_file: &fs::File) -> Result<()> {
    Ok(())
}

pub(super) fn acquire_config_read_lock(config_target: &Path) -> Result<ConfigLock> {
    let lock_path = config_lock_path(config_target)?;
    let file = open_config_lock_file(&lock_path)?;
    lock_file_shared(&file)?;
    Ok(ConfigLock { _file: file })
}
