use anyhow::{Context, Result, anyhow, bail};
use async_trait::async_trait;
use serde_json::{Value, json};
use std::path::{Component, Path, PathBuf};
use std::process::Stdio;
use std::sync::Arc;
use tokio::process::Command;
use tokio::time::{Duration, timeout};
use tracing::debug;

const AST_GREP_TIMEOUT_SECS: u64 = 30;
const MAX_PREVIEW_BYTES: usize = 64 * 1024;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AnalysisCapability {
    AstSearch,
    AstReplacePreview,
}

#[derive(Debug, Clone)]
pub struct AstSearchRequest {
    pub path: String,
    pub language: Option<String>,
    pub pattern: String,
    pub max_results: usize,
    pub allow_outside_workspace: bool,
}

#[derive(Debug, Clone)]
pub struct AstReplacePreviewRequest {
    pub path: String,
    pub language: Option<String>,
    pub pattern: String,
    pub rewrite: String,
    pub allow_outside_workspace: bool,
}

#[async_trait]
pub trait CodeAnalysisBackend: Send + Sync {
    fn name(&self) -> &'static str;

    fn supports(&self, capability: AnalysisCapability) -> bool;

    async fn ast_search(&self, _request: AstSearchRequest) -> Result<Value> {
        bail!("backend {} does not support ast_search", self.name())
    }

    async fn ast_replace_preview(&self, _request: AstReplacePreviewRequest) -> Result<Value> {
        bail!(
            "backend {} does not support ast_replace_preview",
            self.name()
        )
    }
}

#[derive(Default, Clone)]
pub struct CodeAnalysisRegistry {
    backends: Vec<Arc<dyn CodeAnalysisBackend>>,
}

impl CodeAnalysisRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn default_backends() -> Self {
        let mut registry = Self::new();
        registry.register(AstGrepCliBackend);
        registry
    }

    pub fn register<B>(&mut self, backend: B)
    where
        B: CodeAnalysisBackend + 'static,
    {
        self.backends.push(Arc::new(backend));
    }

    pub async fn ast_search(&self, request: AstSearchRequest) -> Result<Value> {
        self.backend_for(AnalysisCapability::AstSearch)?
            .ast_search(request)
            .await
    }

    pub async fn ast_replace_preview(&self, request: AstReplacePreviewRequest) -> Result<Value> {
        self.backend_for(AnalysisCapability::AstReplacePreview)?
            .ast_replace_preview(request)
            .await
    }

    fn backend_for(&self, capability: AnalysisCapability) -> Result<Arc<dyn CodeAnalysisBackend>> {
        self.backends
            .iter()
            .find(|backend| backend.supports(capability))
            .cloned()
            .ok_or_else(|| anyhow!("no code analysis backend supports {:?}", capability))
    }
}

struct AstGrepCliBackend;

#[async_trait]
impl CodeAnalysisBackend for AstGrepCliBackend {
    fn name(&self) -> &'static str {
        "ast-grep-cli"
    }

    fn supports(&self, capability: AnalysisCapability) -> bool {
        matches!(
            capability,
            AnalysisCapability::AstSearch | AnalysisCapability::AstReplacePreview
        )
    }

    async fn ast_search(&self, request: AstSearchRequest) -> Result<Value> {
        let root = workspace_root()?;
        let path = analysis_path(&root, &request.path, request.allow_outside_workspace)?;
        let max_results = request.max_results.clamp(1, 1000);

        let mut args = vec![
            "run".to_string(),
            "--pattern".to_string(),
            request.pattern.clone(),
            "--json=stream".to_string(),
        ];
        push_lang_args(&mut args, request.language.as_deref());
        args.push(path.clone());

        let output = run_ast_grep(&root, &args).await?;
        let mut matches = Vec::new();
        let mut parse_errors = Vec::new();
        let mut truncated = output.stdout_truncated;

        for line in output.stdout.lines() {
            if matches.len() >= max_results {
                truncated = true;
                break;
            }

            match serde_json::from_str::<Value>(line) {
                Ok(value) => matches.push(value),
                Err(err) => parse_errors.push(json!({
                    "line": line,
                    "error": err.to_string(),
                })),
            }
        }

        Ok(json!({
            "backend": self.name(),
            "path": path,
            "language": normalized_language(request.language.as_deref()),
            "pattern": request.pattern,
            "matches": matches,
            "match_count": matches.len(),
            "truncated": truncated,
            "status": output.status,
            "success": output.success,
            "stderr": output.stderr,
            "stderr_truncated": output.stderr_truncated,
            "parse_errors": parse_errors,
        }))
    }

    async fn ast_replace_preview(&self, request: AstReplacePreviewRequest) -> Result<Value> {
        let root = workspace_root()?;
        let path = analysis_path(&root, &request.path, request.allow_outside_workspace)?;

        let mut args = vec![
            "run".to_string(),
            "--pattern".to_string(),
            request.pattern.clone(),
            "--rewrite".to_string(),
            request.rewrite.clone(),
        ];
        push_lang_args(&mut args, request.language.as_deref());
        args.push(path.clone());

        let output = run_ast_grep(&root, &args).await?;

        Ok(json!({
            "backend": self.name(),
            "path": path,
            "language": normalized_language(request.language.as_deref()),
            "pattern": request.pattern,
            "rewrite": request.rewrite,
            "diff_preview": output.stdout,
            "diff_preview_truncated": output.stdout_truncated,
            "stderr": output.stderr,
            "stderr_truncated": output.stderr_truncated,
            "status": output.status,
            "success": output.success,
            "applied": false,
            "note": "Preview only. This tool does not write files. Use edit__apply_patch for audited edits.",
        }))
    }
}

struct CommandOutput {
    status: Option<i32>,
    success: bool,
    stdout: String,
    stdout_truncated: bool,
    stderr: String,
    stderr_truncated: bool,
}

async fn run_ast_grep(root: &Path, args: &[String]) -> Result<CommandOutput> {
    debug!(args = ?args, "running ast-grep");

    let output = match timeout(
        Duration::from_secs(AST_GREP_TIMEOUT_SECS),
        Command::new("ast-grep")
            .args(args)
            .current_dir(root)
            .stdin(Stdio::null())
            .output(),
    )
    .await
    {
        Ok(output) => output.with_context(
            || "failed to run ast-grep. Install ast-grep CLI or use search__rg/fs__read",
        )?,
        Err(_) => bail!("ast-grep timed out after {AST_GREP_TIMEOUT_SECS}s"),
    };

    let stdout = truncate_utf8(&String::from_utf8_lossy(&output.stdout), MAX_PREVIEW_BYTES);
    let stderr = truncate_utf8(&String::from_utf8_lossy(&output.stderr), MAX_PREVIEW_BYTES);

    Ok(CommandOutput {
        status: output.status.code(),
        success: output.status.success(),
        stdout: stdout.text,
        stdout_truncated: stdout.truncated,
        stderr: stderr.text,
        stderr_truncated: stderr.truncated,
    })
}

fn push_lang_args(args: &mut Vec<String>, language: Option<&str>) {
    let Some(language) = language else {
        return;
    };

    let language = language.trim();
    if language.is_empty() || language.eq_ignore_ascii_case("auto") {
        return;
    }

    args.push("--lang".to_string());
    args.push(language.to_string());
}

fn normalized_language(language: Option<&str>) -> String {
    language
        .filter(|language| !language.trim().is_empty())
        .unwrap_or("auto")
        .to_string()
}

fn workspace_root() -> Result<PathBuf> {
    std::env::current_dir()?
        .canonicalize()
        .context("failed to canonicalize current workspace")
}

fn workspace_relative_path(root: &Path, path: &str) -> Result<String> {
    let path = safe_relative_path(path)?;
    let absolute = root.join(&path);
    if absolute.exists() {
        let canonical = absolute.canonicalize()?;
        if !canonical.starts_with(root) {
            bail!("path is outside workspace: {}", canonical.display());
        }
    }
    Ok(path.to_string_lossy().to_string())
}

fn analysis_path(root: &Path, path: &str, allow_outside_workspace: bool) -> Result<String> {
    if !allow_outside_workspace {
        return workspace_relative_path(root, path);
    }

    let path = Path::new(path);
    let candidate = if path.is_absolute() {
        path.to_path_buf()
    } else {
        root.join(path)
    };
    let absolute = if candidate.exists() {
        candidate.canonicalize()?
    } else {
        candidate
    };
    Ok(absolute.to_string_lossy().to_string())
}

fn safe_relative_path(path: &str) -> Result<PathBuf> {
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
        Ok(PathBuf::from("."))
    } else {
        Ok(normalized)
    }
}

struct TruncatedText {
    text: String,
    truncated: bool,
}

fn truncate_utf8(text: &str, max_bytes: usize) -> TruncatedText {
    if text.len() <= max_bytes {
        return TruncatedText {
            text: text.to_string(),
            truncated: false,
        };
    }

    let mut end = max_bytes;
    while !text.is_char_boundary(end) {
        end -= 1;
    }

    TruncatedText {
        text: text[..end].to_string(),
        truncated: true,
    }
}
