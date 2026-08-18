//! ApplyPatch secure batch authorization and execution.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use std::sync::Arc;
#[cfg(unix)]
use std::{
    fs::File,
    os::fd::{AsRawFd, FromRawFd, OwnedFd},
};

use anyhow::{Context, Result, anyhow, bail};
use serde_json::{Value, json};

use crate::permission::{PermissionResource, path_preview};

use super::args::required_string;
use super::paths::{join_workspace_path, workspace_root};
use super::{ExternalWorkspaceAccess, ToolExecutionContext};

/// An opaque, authorization-time binding for an ApplyPatch batch.
#[derive(Clone)]
pub(crate) struct PreparedApplyPatch {
    inner: Arc<PreparedApplyPatchInner>,
}

impl std::fmt::Debug for PreparedApplyPatch {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PreparedApplyPatch")
            .field("binding", &"<prepared_apply_patch>")
            .finish()
    }
}


const APPLY_PATCH_CHANGED: &str = "apply patch target changed after authorization";
const APPLY_PATCH_MODIFIED: &str = "apply patch target was concurrently modified";

struct ParsedApplyPatchEdit {
    path: String,
    find: String,
    replace: String,
    replace_all: bool,
}

struct PreparedApplyPatchInner {
    workspace_root: PathBuf,
    edit_paths: Vec<String>,
    edit_targets: Vec<PathBuf>,
    targets: BTreeMap<PathBuf, PreparedApplyPatchTarget>,
    #[cfg(unix)]
    parents: BTreeMap<PathBuf, PreparedApplyPatchParent>,
    #[cfg(test)]
    hook: std::sync::Mutex<Option<(ApplyPatchWorkerPoint, Box<dyn FnOnce() + Send>)>>,
}

#[cfg(test)]
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum ApplyPatchWorkerPoint {
    BeforeStagingValidation,
    BeforeBatchPrecommit,
    BeforeCommitOpen { path: PathBuf, ordinal: usize },
}

#[cfg(unix)]
struct PreparedApplyPatchParent {
    fd: OwnedFd,
    dev: u64,
    ino: u64,
}

#[cfg(unix)]
struct PreparedApplyPatchTarget {
    leaf: std::ffi::OsString,
    parent: PathBuf,
    fd: OwnedFd,
    dev: u64,
    ino: u64,
    size: i64,
    mtime: (i64, i64),
    ctime: (i64, i64),
    nlink: u64,
}

#[cfg(not(unix))]
struct PreparedApplyPatchTarget;

fn parse_apply_patch(args: &Value) -> Result<Vec<ParsedApplyPatchEdit>> {
    let edits = args
        .get("edits")
        .and_then(Value::as_array)
        .ok_or_else(|| anyhow!("missing or invalid array argument: edits"))?;
    if edits.is_empty() {
        bail!("edits cannot be empty");
    }
    edits
        .iter()
        .enumerate()
        .map(|(index, edit)| {
            let path = required_string(edit, "path")?.to_owned();
            let find = required_string(edit, "find")?.to_owned();
            let replace = required_string(edit, "replace")?.to_owned();
            let replace_all = edit
                .get("replace_all")
                .and_then(Value::as_bool)
                .ok_or_else(|| {
                    anyhow!("missing or invalid boolean argument: edits[{index}].replace_all")
                })?;
            if find.is_empty() {
                bail!("edits[{index}].find cannot be empty");
            }
            Ok(ParsedApplyPatchEdit {
                path,
                find,
                replace,
                replace_all,
            })
        })
        .collect()
}

pub(crate) fn prepare_apply_patch_targets(args: &Value) -> Result<PreparedApplyPatch> {
    let edits = parse_apply_patch(args)?;
    prepare_apply_patch_edits(&edits)
}

#[cfg(unix)]
fn prepare_apply_patch_edits(edits: &[ParsedApplyPatchEdit]) -> Result<PreparedApplyPatch> {
    use std::os::unix::{ffi::OsStrExt, fs::MetadataExt};

    let workspace_root = workspace_root()?;
    let edit_paths = edits.iter().map(|edit| edit.path.clone()).collect();
    let mut edit_targets = Vec::with_capacity(edits.len());
    let mut canonical = BTreeSet::new();
    for edit in edits {
        let candidate = join_workspace_path(&workspace_root, &edit.path);
        let target = candidate
            .canonicalize()
            .with_context(|| format!("path does not exist: {}", candidate.display()))?;
        edit_targets.push(target.clone());
        canonical.insert(target);
    }
    if canonical.len() > 64 {
        bail!("apply patch accepts at most 64 unique target files");
    }

    let mut parents = BTreeMap::new();
    let mut targets = BTreeMap::new();
    let mut identities = BTreeSet::new();
    for destination in canonical {
        let parent = destination
            .parent()
            .ok_or_else(|| anyhow!("path has no parent: {}", destination.display()))?
            .to_path_buf();
        let leaf = destination
            .file_name()
            .ok_or_else(|| anyhow!("path has no file name: {}", destination.display()))?
            .to_os_string();
        if !parents.contains_key(&parent) {
            let parent_c = std::ffi::CString::new(parent.as_os_str().as_bytes())?;
            let fd = unsafe {
                libc::open(
                    parent_c.as_ptr(),
                    libc::O_RDONLY | libc::O_DIRECTORY | libc::O_CLOEXEC | libc::O_NOFOLLOW,
                )
            };
            if fd < 0 {
                return Err(std::io::Error::last_os_error()).with_context(|| {
                    format!("failed to open parent directory {}", parent.display())
                });
            }
            let fd = unsafe { OwnedFd::from_raw_fd(fd) };
            let metadata = File::from(fd.try_clone().map_err(|error| {
                apply_patch_io_error(&destination, "clone parent directory anchor", error)
            })?)
            .metadata()
            .map_err(|error| {
                apply_patch_io_error(&destination, "inspect parent directory", error)
            })?;
            parents.insert(
                parent.clone(),
                PreparedApplyPatchParent {
                    fd,
                    dev: metadata.dev(),
                    ino: metadata.ino(),
                },
            );
        }
        let anchor = parents.get(&parent).expect("inserted parent");
        let leaf_c = std::ffi::CString::new(leaf.as_bytes())?;
        let fd = unsafe {
            libc::openat(
                anchor.fd.as_raw_fd(),
                leaf_c.as_ptr(),
                libc::O_RDONLY | libc::O_NONBLOCK | libc::O_CLOEXEC | libc::O_NOFOLLOW,
            )
        };
        if fd < 0 {
            return Err(std::io::Error::last_os_error())
                .with_context(|| format!("failed to open file {}", destination.display()));
        }
        let fd = unsafe { OwnedFd::from_raw_fd(fd) };
        let metadata = File::from(
            fd.try_clone()
                .map_err(|error| apply_patch_io_error(&destination, "clone file anchor", error))?,
        )
        .metadata()
        .map_err(|error| apply_patch_io_error(&destination, "inspect file", error))?;
        if !metadata.is_file() {
            bail!("edits path is not a file: {}", destination.display());
        }
        if !identities.insert((metadata.dev(), metadata.ino())) {
            bail!("apply patch targets alias the same existing file");
        }
        targets.insert(
            destination.clone(),
            PreparedApplyPatchTarget {
                leaf,
                parent,
                fd,
                dev: metadata.dev(),
                ino: metadata.ino(),
                size: metadata.size() as i64,
                mtime: (metadata.mtime(), metadata.mtime_nsec()),
                ctime: (metadata.ctime(), metadata.ctime_nsec()),
                nlink: metadata.nlink(),
            },
        );
    }
    Ok(PreparedApplyPatch {
        inner: Arc::new(PreparedApplyPatchInner {
            workspace_root,
            edit_paths,
            edit_targets,
            targets,
            parents,
            #[cfg(test)]
            hook: std::sync::Mutex::new(None),
        }),
    })
}

#[cfg(not(unix))]
fn prepare_apply_patch_edits(_edits: &[ParsedApplyPatchEdit]) -> Result<PreparedApplyPatch> {
    bail!("secure apply patch authorization is unsupported on this platform")
}

impl PreparedApplyPatch {
    pub(crate) fn target_paths(&self) -> impl Iterator<Item = &Path> + '_ {
        self.inner.targets.keys().map(PathBuf::as_path)
    }

    pub(crate) fn external_workspace_access(&self) -> Option<ExternalWorkspaceAccess> {
        let paths: Vec<_> = self
            .inner
            .targets
            .keys()
            .filter(|path| !path.starts_with(&self.inner.workspace_root))
            .map(|path| path_preview(path))
            .collect();
        (!paths.is_empty()).then_some(ExternalWorkspaceAccess { paths })
    }

    pub(crate) fn permission_resource(&self, tool: &str) -> PermissionResource {
        PermissionResource::PatchTargets {
            tool: tool.into(),
            paths: self.inner.targets.keys().cloned().collect(),
        }
    }

    #[cfg(test)]
    pub(crate) fn set_worker_hook(
        &self,
        point: ApplyPatchWorkerPoint,
        hook: impl FnOnce() + Send + 'static,
    ) {
        *self.inner.hook.lock().expect("apply patch hook poisoned") = Some((point, Box::new(hook)));
    }

    #[cfg(test)]
    fn run_worker_hook(&self, point: ApplyPatchWorkerPoint) {
        let hook = {
            let mut hook = self.inner.hook.lock().expect("apply patch hook poisoned");
            hook.as_ref()
                .is_some_and(|(expected, _)| expected == &point)
                .then(|| hook.take().expect("hook checked").1)
        };
        if let Some(hook) = hook {
            hook();
        }
    }
}

fn prepared_apply_patch_for_execution(
    args: &Value,
    context: &ToolExecutionContext,
) -> Result<(Vec<ParsedApplyPatchEdit>, PreparedApplyPatch)> {
    let edits = parse_apply_patch(args)?;
    let prepared = match context.prepared_apply_patch() {
        Some(prepared) => prepared.clone(),
        None => prepare_apply_patch_edits(&edits)?,
    };
    if !context.allow_outside_workspace && prepared.external_workspace_access().is_some() {
        bail!("path is outside workspace");
    }
    validate_apply_patch_mapping(&prepared, &edits)?;
    Ok((edits, prepared))
}

fn validate_apply_patch_mapping(
    prepared: &PreparedApplyPatch,
    edits: &[ParsedApplyPatchEdit],
) -> Result<()> {
    if edits.len() != prepared.inner.edit_targets.len()
        || edits
            .iter()
            .map(|edit| &edit.path)
            .ne(prepared.inner.edit_paths.iter())
    {
        bail!(APPLY_PATCH_CHANGED);
    }
    for expected in &prepared.inner.edit_targets {
        #[cfg(unix)]
        {
            use std::os::unix::fs::MetadataExt;
            let target = prepared
                .inner
                .targets
                .get(expected)
                .expect("prepared target");
            let parent = prepared
                .inner
                .parents
                .get(&target.parent)
                .expect("prepared parent");
            let metadata = std::fs::metadata(&target.parent).map_err(|error| {
                apply_patch_open_error(expected, "inspect parent directory", error)
            })?;
            if !metadata.is_dir() || metadata.dev() != parent.dev || metadata.ino() != parent.ino {
                bail!(APPLY_PATCH_CHANGED);
            }
            let leaf_metadata = std::fs::metadata(expected)
                .map_err(|error| apply_patch_open_error(expected, "inspect file", error))?;
            if !apply_patch_metadata_matches(target, &leaf_metadata) {
                bail!(APPLY_PATCH_CHANGED);
            }
        }
    }
    Ok(())
}

#[cfg(unix)]
fn apply_patch_open_error(path: &Path, action: &str, error: std::io::Error) -> anyhow::Error {
    match error.raw_os_error() {
        Some(libc::ENOENT | libc::ENOTDIR | libc::ELOOP) => anyhow!(APPLY_PATCH_CHANGED),
        _ => anyhow!(error).context(format!("failed to {action} {}", path.display())),
    }
}

#[cfg(unix)]
fn apply_patch_io_error(path: &Path, action: &str, error: std::io::Error) -> anyhow::Error {
    anyhow!(error).context(format!("failed to {action} {}", path.display()))
}

#[cfg(unix)]
fn apply_patch_metadata_matches(
    target: &PreparedApplyPatchTarget,
    metadata: &std::fs::Metadata,
) -> bool {
    use std::os::unix::fs::MetadataExt;
    metadata.is_file() && metadata.dev() == target.dev && metadata.ino() == target.ino
}

#[cfg(unix)]
fn apply_patch_version_matches(
    target: &PreparedApplyPatchTarget,
    metadata: &std::fs::Metadata,
) -> bool {
    use std::os::unix::fs::MetadataExt;
    metadata.size() as i64 == target.size
        && (metadata.mtime(), metadata.mtime_nsec()) == target.mtime
        && (metadata.ctime(), metadata.ctime_nsec()) == target.ctime
        && metadata.nlink() == target.nlink
}

#[cfg(unix)]
fn apply_patch_open(prepared: &PreparedApplyPatch, path: &Path, writable: bool) -> Result<File> {
    use std::os::unix::ffi::OsStrExt;
    let target = prepared.inner.targets.get(path).expect("prepared target");
    let parent = prepared
        .inner
        .parents
        .get(&target.parent)
        .expect("prepared parent");
    let parent_fd = parent
        .fd
        .try_clone()
        .map_err(|error| apply_patch_io_error(path, "clone parent directory anchor", error))?;
    let parent_metadata = File::from(parent_fd)
        .metadata()
        .map_err(|error| apply_patch_io_error(path, "inspect parent directory", error))?;
    use std::os::unix::fs::MetadataExt;
    if parent_metadata.dev() != parent.dev || parent_metadata.ino() != parent.ino {
        bail!(APPLY_PATCH_CHANGED);
    }
    let leaf = std::ffi::CString::new(target.leaf.as_bytes())?;
    let flags = if writable {
        libc::O_RDWR
    } else {
        libc::O_RDONLY
    } | libc::O_NONBLOCK
        | libc::O_CLOEXEC
        | libc::O_NOFOLLOW;
    let fd = unsafe { libc::openat(parent.fd.as_raw_fd(), leaf.as_ptr(), flags) };
    if fd < 0 {
        let open_error = std::io::Error::last_os_error();
        return Err(apply_patch_open_failure(
            parent,
            target,
            path,
            if writable {
                "open for writing"
            } else {
                "open for reading"
            },
            open_error,
        ));
    }
    let file = unsafe { File::from_raw_fd(fd) };
    let metadata = file
        .metadata()
        .map_err(|error| apply_patch_io_error(path, "inspect file", error))?;
    if !apply_patch_metadata_matches(target, &metadata) {
        bail!(APPLY_PATCH_CHANGED);
    }
    if !apply_patch_version_matches(target, &metadata) {
        bail!(APPLY_PATCH_MODIFIED);
    }
    Ok(file)
}

#[cfg(unix)]
fn apply_patch_open_failure(
    parent: &PreparedApplyPatchParent,
    target: &PreparedApplyPatchTarget,
    path: &Path,
    action: &str,
    open_error: std::io::Error,
) -> anyhow::Error {
    use std::os::unix::ffi::OsStrExt;

    let leaf = match std::ffi::CString::new(target.leaf.as_bytes()) {
        Ok(leaf) => leaf,
        Err(_) => return apply_patch_io_error(path, action, open_error),
    };
    let mut stat = std::mem::MaybeUninit::<libc::stat>::uninit();
    let result = unsafe {
        libc::fstatat(
            parent.fd.as_raw_fd(),
            leaf.as_ptr(),
            stat.as_mut_ptr(),
            libc::AT_SYMLINK_NOFOLLOW,
        )
    };
    if result < 0 {
        return match std::io::Error::last_os_error().raw_os_error() {
            Some(libc::ENOENT | libc::ENOTDIR | libc::ELOOP) => anyhow!(APPLY_PATCH_CHANGED),
            _ => apply_patch_io_error(path, action, open_error),
        };
    }
    let stat = unsafe { stat.assume_init() };
    let regular = (stat.st_mode & libc::S_IFMT) == libc::S_IFREG;
    if !regular || stat.st_dev as u64 != target.dev || stat.st_ino != target.ino {
        anyhow!(APPLY_PATCH_CHANGED)
    } else {
        apply_patch_io_error(path, action, open_error)
    }
}

#[cfg(unix)]
fn apply_patch_read(
    file: &mut File,
    target: &PreparedApplyPatchTarget,
    path: &Path,
) -> Result<Vec<u8>> {
    use std::io::{Read, Seek, SeekFrom};
    let before = file
        .metadata()
        .map_err(|error| apply_patch_io_error(path, "inspect file", error))?;
    if !apply_patch_metadata_matches(target, &before) {
        bail!(APPLY_PATCH_CHANGED);
    }
    if !apply_patch_version_matches(target, &before) {
        bail!(APPLY_PATCH_MODIFIED);
    }
    file.seek(SeekFrom::Start(0))
        .map_err(|error| apply_patch_io_error(path, "seek file", error))?;
    let mut bytes = Vec::new();
    file.read_to_end(&mut bytes)
        .map_err(|error| apply_patch_io_error(path, "read file", error))?;
    let after = file
        .metadata()
        .map_err(|error| apply_patch_io_error(path, "inspect file", error))?;
    if !apply_patch_metadata_matches(target, &after) {
        bail!(APPLY_PATCH_CHANGED);
    }
    if !apply_patch_version_matches(target, &after) {
        bail!(APPLY_PATCH_MODIFIED);
    }
    Ok(bytes)
}

pub(crate) async fn apply_patch(args: Value, context: ToolExecutionContext) -> Result<Value> {
    tokio::task::spawn_blocking(move || apply_patch_worker(args, context)).await?
}

#[cfg(unix)]
fn apply_patch_worker(args: Value, context: ToolExecutionContext) -> Result<Value> {
    use std::io::{Seek, SeekFrom, Write};
    let (edits, prepared) = prepared_apply_patch_for_execution(&args, &context)?;
    let mut contents = BTreeMap::new();
    let mut originals = BTreeMap::new();
    for path in prepared.inner.targets.keys() {
        let target = prepared.inner.targets.get(path).expect("prepared target");
        let mut file = File::from(
            target
                .fd
                .try_clone()
                .map_err(|error| apply_patch_io_error(path, "open approved file", error))?,
        );
        let bytes = apply_patch_read(&mut file, target, path)?;
        let content = String::from_utf8(bytes)
            .with_context(|| format!("failed to read UTF-8 file {}", path.display()))?;
        originals.insert(path.clone(), content.clone());
        contents.insert(path.clone(), content);
    }
    #[cfg(test)]
    prepared.run_worker_hook(ApplyPatchWorkerPoint::BeforeStagingValidation);
    let mut results = Vec::with_capacity(edits.len());
    for (index, edit) in edits.iter().enumerate() {
        let path = &prepared.inner.edit_targets[index];
        let content = contents.get_mut(path).expect("staged content");
        let matches = content.matches(&edit.find).count();
        let label = bound_display(&prepared.inner.workspace_root, path);
        if matches == 0 {
            bail!("edits[{index}] did not match any text in {label}");
        }
        if !edit.replace_all && matches != 1 {
            bail!(
                "edits[{index}] matched {matches} occurrences in {label}; provide more context or set replace_all=true"
            );
        }
        *content = if edit.replace_all {
            content.replace(&edit.find, &edit.replace)
        } else {
            content.replacen(&edit.find, &edit.replace, 1)
        };
        results
            .push(json!({"path": label, "replacements": matches, "replace_all": edit.replace_all}));
    }
    validate_apply_patch_mapping(&prepared, &edits)?;
    #[cfg(test)]
    prepared.run_worker_hook(ApplyPatchWorkerPoint::BeforeBatchPrecommit);
    for path in prepared.inner.targets.keys() {
        let target = prepared.inner.targets.get(path).expect("prepared target");
        let mut file = apply_patch_open(&prepared, path, false)?;
        let bytes = apply_patch_read(&mut file, target, path)?;
        if bytes != originals.get(path).expect("staged original").as_bytes() {
            bail!(APPLY_PATCH_MODIFIED);
        }
    }
    for (ordinal, (path, content)) in contents.iter().enumerate() {
        #[cfg(test)]
        prepared.run_worker_hook(ApplyPatchWorkerPoint::BeforeCommitOpen {
            path: path.clone(),
            ordinal,
        });
        #[cfg(not(test))]
        let _ = ordinal;
        let target = prepared.inner.targets.get(path).expect("prepared target");
        let mut file = apply_patch_open(&prepared, path, true)?;
        let current = apply_patch_read(&mut file, target, path)?;
        if current != originals.get(path).expect("staged original").as_bytes() {
            bail!(APPLY_PATCH_MODIFIED);
        }
        file.set_len(0)
            .map_err(|error| apply_patch_io_error(path, "write patched file", error))?;
        file.seek(SeekFrom::Start(0))
            .map_err(|error| apply_patch_io_error(path, "seek file", error))?;
        file.write_all(content.as_bytes())
            .map_err(|error| apply_patch_io_error(path, "write patched file", error))?;
        file.flush()
            .map_err(|error| apply_patch_io_error(path, "flush file", error))?;
        file.seek(SeekFrom::Start(0))
            .map_err(|error| apply_patch_io_error(path, "seek file", error))?;
        let mut verified = Vec::new();
        std::io::Read::read_to_end(&mut file, &mut verified)
            .map_err(|error| apply_patch_io_error(path, "verify file", error))?;
        if verified != content.as_bytes() {
            bail!("failed to write patched file {}", path.display());
        }
    }
    Ok(json!({"files_changed": contents.len(), "edits_applied": edits.len(), "edits": results}))
}

#[cfg(not(unix))]
fn apply_patch_worker(args: Value, context: ToolExecutionContext) -> Result<Value> {
    let _ = prepared_apply_patch_for_execution(&args, &context)?;
    bail!("secure apply patch authorization is unsupported on this platform")
}

fn bound_display(root: &Path, path: &Path) -> String {
    path.strip_prefix(root)
        .unwrap_or(path)
        .to_string_lossy()
        .to_string()
}

