//! Shared helpers for folding large tool output to a local temp artifact.
//!
//! Tools whose output can grow large (web__fetch, shell__exec) persist the full
//! body to a deterministic temp file and return only a short preview inline.
//! The model can then retrieve or search the full output on demand with
//! fs__read / search__rg via the returned `local_path`.

use anyhow::{Context, Result};

/// Bodies larger than this are folded to a local artifact instead of being
/// returned inline.
pub const FOLD_THRESHOLD_BYTES: usize = 64 * 1024;
/// Subdirectory names (under the OS temp dir) used for folded tool output.
pub const FETCH_ARTIFACT_DIR: &str = "letcode-fetch";
pub const COMMAND_ARTIFACT_DIR: &str = "letcode-command";
pub const SEARCH_ARTIFACT_DIR: &str = "letcode-search";
/// Number of characters kept inline when a body is folded.
pub const FOLD_PREVIEW_CHARS: usize = 8 * 1024;

pub fn fold_preview(content: &str, max_chars: usize) -> String {
    content.chars().take(max_chars).collect()
}

/// Deterministic artifact name derived from the body content. Identical bodies
/// reuse the same file, so repeated outputs do not accumulate duplicates.
///
/// FNV-1a is used instead of std's DefaultHasher because the latter's internal
/// algorithm is not stable across rustc releases, which would silently break
/// cross-version deduplication.
fn artifact_file_name(body: &[u8], ext: &str) -> String {
    const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
    const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;
    let mut hash = FNV_OFFSET;
    for &byte in body {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(FNV_PRIME);
    }
    format!("{hash:016x}.{ext}")
}

/// Persist `body` under `{temp_dir}/{dir_name}/{hash}.{ext}` and return the
/// absolute path. Writing is idempotent for identical bodies.
pub async fn write_artifact(dir_name: &str, body: &[u8], ext: &str) -> Result<String> {
    let dir = std::env::temp_dir().join(dir_name);
    tokio::fs::create_dir_all(&dir)
        .await
        .with_context(|| format!("failed to create artifact dir {}", dir.display()))?;
    let path = dir.join(artifact_file_name(body, ext));
    tokio::fs::write(&path, body)
        .await
        .with_context(|| format!("failed to write artifact {}", path.display()))?;
    Ok(path.to_string_lossy().to_string())
}

/// Whether `path` resolves inside one of letcode's own fold-artifact directories
/// under the OS temp dir. Reads of these are trusted read-only access: they are
/// temp, content-addressed outputs written by the tools themselves, so they do
/// not warrant a permission prompt.
///
/// `path` is compared against a canonicalized temp root so symlinked temp dirs
/// (e.g. `/var` → `/private/var` on macOS) match the canonical paths produced by
/// `external_workspace_access_for_tool`. Unresolvable cases fall back to
/// untrusted (safe default).
pub fn is_trusted_artifact_path(path: &std::path::Path) -> bool {
    if is_retained_artifact_path(path) {
        return true;
    }
    let Ok(root) = std::env::temp_dir().canonicalize() else {
        return false;
    };
    // Canonicalize when possible (paths from `access.paths` already canonical);
    // otherwise fall back to the raw path so a syntax-level prefix check still
    // works for the entirely artificial/missing paths exercised in tests.
    let path = path.canonicalize().unwrap_or_else(|_| path.to_path_buf());
    [
        FETCH_ARTIFACT_DIR,
        COMMAND_ARTIFACT_DIR,
        SEARCH_ARTIFACT_DIR,
    ]
    .into_iter()
    .any(|dir| path.starts_with(root.join(dir)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    #[test]
    fn preview_keeps_only_leading_chars() {
        let content = "abcdefghij";
        assert_eq!(fold_preview(content, 4), "abcd");
        assert_eq!(fold_preview(content, 100), content);
        assert_eq!(fold_preview("中文内容", 2), "中文");
    }

    #[test]
    fn artifact_names_are_content_addressed() {
        let name = artifact_file_name(b"hello", "txt");
        assert_eq!(name, artifact_file_name(b"hello", "txt"));
        assert_ne!(name, artifact_file_name(b"hello!", "txt"));
        assert!(name.ends_with(".txt"));
        assert_ne!(
            artifact_file_name(b"x", "out"),
            artifact_file_name(b"x", "err")
        );
    }

    #[test]
    fn trusted_artifact_path_covers_only_our_temp_subdirs() {
        let root = std::env::temp_dir()
            .canonicalize()
            .unwrap_or_else(|_| std::env::temp_dir());
        let command = root.join(COMMAND_ARTIFACT_DIR).join("stream-1-2.out");
        let fetch = root.join(FETCH_ARTIFACT_DIR).join("a.txt");
        let search = root.join(SEARCH_ARTIFACT_DIR).join("deadbeef.txt");
        assert!(is_trusted_artifact_path(Path::new(&command)));
        assert!(is_trusted_artifact_path(Path::new(&fetch)));
        assert!(is_trusted_artifact_path(Path::new(&search)));
        // Sibling / unrelated paths and the temp root itself are not trusted.
        assert!(!is_trusted_artifact_path(Path::new(&root)));
        assert!(!is_trusted_artifact_path(Path::new(
            &root.join("letcode-outside-tool-read-123.txt")
        )));
        assert!(!is_trusted_artifact_path(Path::new(
            &root.join("letcode-command-other")
        )));
        assert!(!is_trusted_artifact_path(Path::new(
            "/nope/letcode-command/x"
        )));
    }

    #[tokio::test]
    async fn write_artifact_is_idempotent_and_round_trips() {
        let body = b"payload content";
        let first = write_artifact("letcode-test-fold", body, "txt")
            .await
            .unwrap();
        let second = write_artifact("letcode-test-fold", body, "txt")
            .await
            .unwrap();
        assert_eq!(first, second);
        let written = tokio::fs::read(&first).await.unwrap();
        assert_eq!(written, body);
        let _ = tokio::fs::remove_file(&first).await;
        let _ = tokio::fs::remove_dir(std::env::temp_dir().join("letcode-test-fold")).await;
    }
}

/// Promote built-in folded outputs before their tool result enters the journal.
/// The content-addressed file is durable before its reference is returned.
pub(crate) async fn retain_session_outputs(
    tool_name: &str,
    output: &mut super::ToolResult,
    session_id: &str,
) -> Result<()> {
    if !matches!(tool_name, "shell__exec" | "web__fetch" | "search__rg") {
        return Ok(());
    }
    anyhow::ensure!(
        !session_id.is_empty()
            && session_id
                .chars()
                .all(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '_')),
        "invalid artifact session id"
    );
    let Some(data) = output
        .data
        .as_mut()
        .and_then(serde_json::Value::as_object_mut)
    else {
        return Ok(());
    };
    if !["folded", "stdout_folded", "stderr_folded"]
        .iter()
        .any(|key| data.get(*key).and_then(serde_json::Value::as_bool) == Some(true))
    {
        return Ok(());
    }
    let root = crate::memory::configured_memory_sessions_dir()?
        .join("artifacts")
        .join(session_id);
    for (flag, key) in [
        ("folded", "local_path"),
        ("stdout_folded", "stdout_local_path"),
        ("stderr_folded", "stderr_local_path"),
    ] {
        if data.get(flag).and_then(serde_json::Value::as_bool) != Some(true) {
            continue;
        }
        let path = data
            .get(key)
            .and_then(serde_json::Value::as_str)
            .ok_or_else(|| anyhow::anyhow!("folded output has no original path"))?;
        let path = std::path::PathBuf::from(path);
        anyhow::ensure!(
            is_trusted_artifact_path(&path),
            "folded output is not a built-in artifact"
        );
        let directory = root.clone();
        let retained =
            tokio::task::spawn_blocking(move || persist_artifact(&path, &directory)).await??;
        data.insert(key.into(), serde_json::json!(retained));
        data.insert(format!("{key}_retained"), serde_json::json!(true));
    }
    Ok(())
}

fn persist_artifact(source: &std::path::Path, directory: &std::path::Path) -> Result<String> {
    use sha2::{Digest, Sha256};
    use std::io::{Read, Write};
    std::fs::create_dir_all(directory)?;
    let mut input = std::fs::File::open(source)?;
    let mut temp = tempfile::NamedTempFile::new_in(directory)?;
    let mut hash = Sha256::new();
    let mut buffer = [0u8; 16384];
    loop {
        let count = input.read(&mut buffer)?;
        if count == 0 {
            break;
        }
        hash.update(&buffer[..count]);
        temp.write_all(&buffer[..count])?;
    }
    temp.as_file().sync_all()?;
    let filename = format!("{:x}.txt", hash.finalize());
    let destination = directory.join(filename);
    match temp.persist_noclobber(&destination) {
        Ok(_) => {}
        Err(error) if error.error.kind() == std::io::ErrorKind::AlreadyExists => {
            // A hash name is not itself proof that an externally edited file is intact.
            let mut existing = std::fs::File::open(&destination)?;
            let mut actual = Sha256::new();
            loop {
                let count = existing.read(&mut buffer)?;
                if count == 0 {
                    break;
                }
                actual.update(&buffer[..count]);
            }
            anyhow::ensure!(
                format!("{:x}.txt", actual.finalize())
                    == destination.file_name().unwrap().to_string_lossy(),
                "retained artifact integrity mismatch"
            );
        }
        Err(error) => return Err(error.error.into()),
    }
    #[cfg(unix)]
    std::fs::File::open(directory)?.sync_all()?;
    Ok(destination.to_string_lossy().into_owned())
}

#[cfg(test)]
mod retained_tests {
    use super::*;
    #[test]
    fn retained_body_survives_removal_of_temporary_original() {
        let root = tempfile::tempdir().unwrap();
        let source = root.path().join("temporary.txt");
        std::fs::write(&source, "complete output").unwrap();
        let path = persist_artifact(&source, &root.path().join("session")).unwrap();
        assert_eq!(
            path,
            persist_artifact(&source, &root.path().join("session")).unwrap()
        );
        std::fs::remove_file(source).unwrap();
        assert_eq!(std::fs::read_to_string(path).unwrap(), "complete output");
    }
}

pub(crate) fn is_retained_artifact_path(path: &std::path::Path) -> bool {
    let Ok(root) = crate::memory::configured_memory_sessions_dir()
        .and_then(|p| Ok(p.join("artifacts").canonicalize()?))
    else {
        return false;
    };
    let Ok(path) = path.canonicalize() else {
        return false;
    };
    let Some(name) = path
        .file_name()
        .and_then(|s| s.to_str())
        .and_then(|s| s.strip_suffix(".txt"))
    else {
        return false;
    };
    name.len() == 64
        && name.bytes().all(|c| c.is_ascii_hexdigit())
        && path.parent().and_then(std::path::Path::parent) == Some(root.as_path())
}
