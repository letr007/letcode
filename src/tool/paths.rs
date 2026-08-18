//! Workspace-scoped path resolution and safety checks shared across tools.

use std::path::{Component, Path, PathBuf};

use anyhow::{Context, Result, bail};

use super::ToolExecutionContext;

pub(crate) fn workspace_root() -> Result<PathBuf> {
    std::env::current_dir()?
        .canonicalize()
        .context("failed to canonicalize current workspace")
}

pub(crate) fn outside_existing_workspace_path(path: &str) -> Option<String> {
    let root = workspace_root().ok()?;
    let candidate = join_workspace_path(&root, path);

    if let Ok(canonical) = candidate.canonicalize() {
        return outside_workspace_label(&root, &canonical);
    }

    syntactic_outside_workspace_label(&root, path, &candidate)
}

pub(crate) fn outside_new_workspace_path(path: &str) -> Option<String> {
    let root = workspace_root().ok()?;
    let candidate = join_workspace_path(&root, path);

    if let Some(canonical_ancestor) = canonical_existing_ancestor(&candidate)
        && let Some(label) = outside_workspace_label(&root, &canonical_ancestor)
    {
        return Some(label);
    }

    syntactic_outside_workspace_label(&root, path, &candidate)
}

pub(crate) fn canonical_existing_ancestor(path: &Path) -> Option<PathBuf> {
    let mut current = Some(path);
    while let Some(candidate) = current {
        if let Ok(canonical) = candidate.canonicalize() {
            return Some(canonical);
        }
        current = candidate.parent();
    }
    None
}

pub(crate) fn outside_workspace_label(root: &Path, path: &Path) -> Option<String> {
    (!path.starts_with(root)).then(|| path.display().to_string())
}

pub(crate) fn syntactic_outside_workspace_label(
    root: &Path,
    raw_path: &str,
    candidate: &Path,
) -> Option<String> {
    let raw_path = Path::new(raw_path);
    if raw_path.is_absolute() {
        return outside_workspace_label(root, raw_path);
    }

    if relative_path_escapes_workspace(raw_path) {
        return Some(candidate.display().to_string());
    }

    None
}

pub(crate) fn relative_path_escapes_workspace(path: &Path) -> bool {
    let mut depth = 0usize;
    for component in path.components() {
        match component {
            Component::Normal(_) => depth = depth.saturating_add(1),
            Component::CurDir => {}
            Component::ParentDir => {
                if depth == 0 {
                    return true;
                }
                depth -= 1;
            }
            Component::RootDir | Component::Prefix(_) => return true,
        }
    }
    false
}

pub(crate) fn existing_workspace_path(
    path: &str,
    context: &ToolExecutionContext,
) -> Result<PathBuf> {
    let root = workspace_root()?;
    let candidate = join_workspace_path(&root, path);
    let canonical = candidate
        .canonicalize()
        .with_context(|| format!("path does not exist: {}", candidate.display()))?;
    if !context.allow_outside_workspace {
        ensure_inside_workspace(&root, &canonical)?;
    }
    Ok(canonical)
}

pub(crate) fn new_workspace_path(path: &str, context: &ToolExecutionContext) -> Result<PathBuf> {
    let root = workspace_root()?;
    if context.allow_outside_workspace {
        if Path::new(path).as_os_str().is_empty() {
            bail!("path cannot be empty");
        }
        let candidate = join_workspace_path(&root, path);
        return Ok(candidate);
    }
    let relative = safe_relative_path_arg(path)?;
    let candidate = root.join(relative);
    if let Some(canonical_ancestor) = canonical_existing_ancestor(&candidate) {
        ensure_inside_workspace(&root, &canonical_ancestor)?;
    }
    Ok(candidate)
}

pub(crate) fn safe_relative_path_arg(path: &str) -> Result<String> {
    let path = Path::new(path);
    if path.is_absolute() {
        bail!("absolute paths are not allowed here: {}", path.display());
    }

    let mut normalized = PathBuf::new();
    for component in path.components() {
        match component {
            Component::Normal(part) => normalized.push(part),
            Component::CurDir => {}
            Component::ParentDir | Component::RootDir | Component::Prefix(_) => {
                bail!("parent traversal is not allowed: {}", path.display());
            }
        }
    }

    if normalized.as_os_str().is_empty() {
        bail!("path cannot be empty");
    }

    Ok(normalized.to_string_lossy().to_string())
}

pub(crate) fn join_workspace_path(root: &Path, path: &str) -> PathBuf {
    let path = Path::new(path);
    if path.is_absolute() {
        path.to_path_buf()
    } else {
        root.join(path)
    }
}

pub(crate) fn ensure_inside_workspace(root: &Path, path: &Path) -> Result<()> {
    if !path.starts_with(root) {
        bail!(
            "path is outside workspace: {} (workspace: {})",
            path.display(),
            root.display()
        );
    }
    Ok(())
}

pub(crate) fn display_workspace_relative(path: &Path) -> Result<String> {
    let root = workspace_root()?;
    let absolute = if path.exists() {
        path.canonicalize()?
    } else {
        path.to_path_buf()
    };

    Ok(absolute
        .strip_prefix(root)
        .unwrap_or(&absolute)
        .to_string_lossy()
        .to_string())
}

pub(crate) fn canonical_existing_path(path: &str) -> Option<PathBuf> {
    join_workspace_path(&workspace_root().ok()?, path)
        .canonicalize()
        .ok()
}

pub(crate) fn canonical_destination_path(path: &str) -> Option<PathBuf> {
    let root = workspace_root().ok()?;
    let candidate = join_workspace_path(&root, path);
    if let Ok(path) = candidate.canonicalize() {
        return Some(path);
    }
    let parent = candidate.parent()?.canonicalize().ok()?;
    Some(parent.join(candidate.file_name()?))
}