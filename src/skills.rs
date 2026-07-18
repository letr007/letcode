use anyhow::{Context, Result, anyhow, bail};
use async_trait::async_trait;
use serde_json::{Value, json};
use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tracing::warn;

use crate::permission::ToolPermissionClass;
use crate::protocol_frames::ProtocolFrameItem;
use crate::runtime_context::{
    FrameVisibility, PromptContributorKind, PromptContributorPlaceholder, RuntimeFrame,
    RuntimeFrameIdSeed, RuntimeFrameKind, RuntimeFrameProvenance, RuntimePromptPayload,
    RuntimePromptRole, RuntimeSnapshot, RuntimeSource,
};
use crate::tool::ToolHandler;

const SKILL_FILE_NAME: &str = "SKILL.md";
const MAX_SKILL_FILE_SAMPLES: usize = 32;
const MAX_SKILL_FILE_DEPTH: usize = 4;
const MAX_SKILL_MD_BYTES: u64 = 1024 * 1024;
const MAX_SKILL_RESOURCE_BYTES: u64 = MAX_SKILL_MD_BYTES;
const MAX_SKILL_NAME_CHARS: usize = 64;

/// Render the explicit marker used to select a skill for the next turn.
pub fn format_manual_skill_marker(name: &str) -> Result<String> {
    validate_skill_name(name)?;
    Ok(format!("@skill({name})"))
}

/// Extract explicit `@skill(name)` selections in input order.
///
/// Only explicit marker starts are interpreted, so ordinary text is unchanged.
/// Repeated names are deduplicated while preserving the first occurrence.
pub fn parse_manual_skill_markers(input: &str) -> Result<Vec<String>> {
    const PREFIX: &str = "@skill(";
    let mut names = Vec::new();
    let mut seen = BTreeSet::new();
    let mut remainder = input;
    while let Some(start) = remainder.find(PREFIX) {
        remainder = &remainder[start + PREFIX.len()..];
        let end = remainder
            .find(')')
            .ok_or_else(|| anyhow!("malformed skill marker: missing ')' after @skill("))?;
        let name = &remainder[..end];
        validate_skill_name(name)
            .with_context(|| format!("invalid skill marker @skill({name})"))?;
        if seen.insert(name.to_string()) {
            names.push(name.to_string());
        }
        remainder = &remainder[end + 1..];
    }
    Ok(names)
}

/// Extract persisted successful skill material without consulting the registry.
/// `None` means this was not a successful `skill` result.
pub(crate) fn parse_persisted_skill_output(output_json: &str) -> Result<Option<(String, String)>> {
    let Ok(result) = serde_json::from_str::<crate::tool::ToolResult>(output_json) else {
        return Ok(None);
    };
    if !result.ok || result.tool != "skill" {
        return Ok(None);
    }
    let Some(data) = result.data.as_ref() else {
        return Ok(None);
    };
    let (Some(name), Some(content)) = (
        data.get("name").and_then(Value::as_str),
        data.get("content").and_then(Value::as_str),
    ) else {
        return Ok(None);
    };
    Ok(Some((name.to_owned(), content.to_owned())))
}

/// Whether this is the explicit replacement shape written by compaction.
///
/// This intentionally does not treat invalid JSON or a textual mention of the
/// marker as pruned output: only an already-materialized skill may survive the
/// structural compaction replacement.
fn is_compaction_pruned_tool_output(output_json: &str) -> bool {
    serde_json::from_str::<Value>(output_json)
        .ok()
        .and_then(|value| {
            value
                .get("_compaction")
                .and_then(Value::as_object)
                .and_then(|marker| marker.get("pruned"))
                .and_then(Value::as_bool)
        })
        == Some(true)
}

/// Rebuild detached exact skill material from persisted protocol output.
pub(crate) fn reconcile_loaded_skill_material(snapshot: &mut RuntimeSnapshot) -> Result<()> {
    let mut skill_calls = BTreeSet::new();
    for frame in &snapshot.frames {
        if let Some(ProtocolFrameItem::AssistantToolCalls { calls, .. }) = &frame.protocol {
            skill_calls.extend(
                calls
                    .iter()
                    .filter(|call| call.name == "skill")
                    .map(|call| call.call_id.clone()),
            );
        }
    }
    let mut occurrences = snapshot
        .frames
        .iter()
        .enumerate()
        .filter_map(|(snapshot_position, frame)| {
            let ProtocolFrameItem::ToolOutput {
                call_id,
                output_json,
            } = frame.protocol.as_ref()?
            else {
                return None;
            };
            skill_calls.contains(call_id).then_some((
                snapshot_position,
                frame.id,
                call_id.clone(),
                output_json.clone(),
                frame.provenance.clone(),
            ))
        })
        .collect::<Vec<_>>();
    // Compaction stores active and retired frames in separate partitions. The
    // transcript source span is the canonical occurrence order across both.
    // Live frames without a span have no transcript chronology, so retain the
    // snapshot/protocol order instead of incorrectly treating their hash IDs
    // as chronology.
    occurrences.sort_by_key(|(snapshot_position, source_id, _, _, provenance)| {
        match provenance.source_span {
            Some(span) => (false, span.start_sequence, span.end_sequence, 0, *source_id),
            None => (true, 0, 0, *snapshot_position, *source_id),
        }
    });
    let mut wanted = BTreeSet::new();
    let mut wanted_frame_ids = BTreeSet::new();
    let mut ordinal: u32 = 0;
    for (_snapshot_position, source_id, call_id, output_json, _source_provenance) in occurrences {
        let contributor_id = format!("skill-material:{call_id}");
        let existing = snapshot
            .prompt_contributors
            .iter()
            .position(|c| c.contributor_id == contributor_id);
        let parsed = parse_persisted_skill_output(&output_json)?;
        if parsed.is_none() && is_compaction_pruned_tool_output(&output_json) {
            if let Some(index) = existing {
                // Compaction deliberately replaces the source body. Its
                // detached material is already authoritative, including the
                // label, payload, and source anchor, so retain it untouched.
                let contributor = &snapshot.prompt_contributors[index];
                wanted.insert(contributor_id);
                wanted_frame_ids.extend(contributor.frame_ids.iter().copied());
                ordinal = ordinal.saturating_add(1);
                continue;
            }
        }
        let Some((name, content)) = parsed else {
            continue;
        };
        wanted.insert(contributor_id.clone());
        let detached_id = RuntimeFrame::new(
            RuntimeFrameKind::PromptContributor,
            FrameVisibility::Active,
            RuntimeFrameProvenance::new(RuntimeSource::PromptContributor)
                .with_source_id(format!("skill-material:{call_id}")),
            RuntimeFrameIdSeed {
                frame_kind: RuntimeFrameKind::PromptContributor,
                source: RuntimeSource::PromptContributor,
                ordinal,
                stable_key: &contributor_id,
                source_span: None,
            },
        )
        .id;
        wanted_frame_ids.insert(detached_id);
        if let Some(frame) = snapshot
            .frames
            .iter_mut()
            .find(|frame| frame.id == detached_id)
        {
            frame.visibility = FrameVisibility::Active;
            frame.summary = Some(format!("Skill: {name}"));
            frame.prompt_payload = Some(RuntimePromptPayload {
                role: RuntimePromptRole::Developer,
                text: content.clone(),
            });
        } else {
            snapshot.push_frame(
                RuntimeFrame::new(
                    RuntimeFrameKind::PromptContributor,
                    FrameVisibility::Active,
                    RuntimeFrameProvenance::new(RuntimeSource::PromptContributor)
                        .with_source_id(format!("skill-material:{call_id}")),
                    RuntimeFrameIdSeed {
                        frame_kind: RuntimeFrameKind::PromptContributor,
                        source: RuntimeSource::PromptContributor,
                        ordinal,
                        stable_key: &contributor_id,
                        source_span: None,
                    },
                )
                .with_summary(format!("Skill: {name}"))
                .with_prompt_payload(RuntimePromptPayload {
                    role: RuntimePromptRole::Developer,
                    text: content,
                }),
            );
        }
        let contributor = PromptContributorPlaceholder {
            contributor_id: contributor_id.clone(),
            kind: PromptContributorKind::SkillMaterial,
            label: Some(name),
            provenance: RuntimeFrameProvenance::new(RuntimeSource::PromptContributor)
                .with_source_id(format!("skill-call:{call_id}")),
            frame_ids: vec![detached_id],
            source_frame_ids: vec![source_id],
        };
        if let Some(index) = existing {
            snapshot.prompt_contributors[index] = contributor;
        } else {
            snapshot.push_prompt_contributor(contributor);
        }
        ordinal = ordinal.saturating_add(1);
    }
    let stale = snapshot
        .prompt_contributors
        .iter()
        .filter(|c| {
            c.contributor_id.starts_with("skill-material:") && !wanted.contains(&c.contributor_id)
        })
        .flat_map(|c| c.frame_ids.iter().copied())
        .collect::<BTreeSet<_>>();
    snapshot.prompt_contributors.retain(|c| {
        !c.contributor_id.starts_with("skill-material:") || wanted.contains(&c.contributor_id)
    });
    snapshot.frames.retain(|frame| {
        !stale.contains(&frame.id)
            && (!frame
                .provenance
                .source_id
                .as_deref()
                .is_some_and(|id| id.starts_with("skill-material:"))
                || wanted_frame_ids.contains(&frame.id))
    });
    snapshot.recompute_protected_frame_ids();
    snapshot.validate_references()
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SkillCard {
    pub name: String,
    pub description: String,
    pub location: String,
    pub path: PathBuf,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SkillEntry {
    pub name: String,
    pub description: String,
    pub body: String,
    pub content: String,
    pub location: String,
    pub path: PathBuf,
    pub base_dir: PathBuf,
}

#[derive(Debug, Clone, Default)]
pub struct SkillRegistry {
    skills: BTreeMap<String, SkillEntry>,
}

impl SkillRegistry {
    pub fn load(config_dir: &Path, workspace_root: &Path) -> Result<Self> {
        Self::load_from_roots(skill_roots(config_dir, workspace_root, true))
    }

    fn load_from_roots(roots: Vec<(PathBuf, String)>) -> Result<Self> {
        let mut entries = Vec::new();
        for (root, location) in roots {
            entries.extend(discover_skill_entries(&root, &location)?);
        }

        Self::from_entries(entries)
    }

    pub fn from_entries(entries: Vec<SkillEntry>) -> Result<Self> {
        let mut skills = BTreeMap::new();
        for entry in entries {
            if let Some(existing) = skills.insert(entry.name.clone(), entry.clone()) {
                warn!(
                    skill = %entry.name,
                    previous = %existing.path.display(),
                    replacement = %entry.path.display(),
                    "replacing lower-priority skill with later discovered skill"
                );
            }
        }
        Ok(Self { skills })
    }

    pub fn is_empty(&self) -> bool {
        self.skills.is_empty()
    }

    pub fn get(&self, name: &str) -> Option<&SkillEntry> {
        self.skills.get(name).or_else(|| {
            let normalized = normalize_skill_name(name);
            self.skills.get(&normalized)
        })
    }

    pub fn selected_entries<'a>(&'a self, names: &[String]) -> Result<Vec<&'a SkillEntry>> {
        names
            .iter()
            .map(|name| {
                self.get(name)
                    .ok_or_else(|| anyhow!("unknown selected skill: {name}"))
            })
            .collect()
    }

    pub fn cards(&self) -> Vec<SkillCard> {
        self.skills
            .values()
            .map(|entry| SkillCard {
                name: entry.name.clone(),
                description: entry.description.clone(),
                location: entry.location.clone(),
                path: entry.path.clone(),
            })
            .collect()
    }

    pub fn list_resource_paths(&self, name: &str) -> Result<Vec<String>> {
        let entry = self
            .get(name)
            .ok_or_else(|| anyhow!("unknown skill: {name}"))?;
        list_relative_resource_files(&entry.base_dir)
    }

    pub fn read_resource(&self, name: &str, path: &str) -> Result<String> {
        let entry = self
            .get(name)
            .ok_or_else(|| anyhow!("unknown skill: {name}"))?;
        let resource_path = resolve_skill_resource_path(&entry.base_dir, path)?;
        read_utf8_resource_file(&resource_path)
    }
}

pub struct SkillTool {
    registry: Arc<SkillRegistry>,
}

impl SkillTool {
    pub fn new(registry: Arc<SkillRegistry>) -> Self {
        Self { registry }
    }
}

pub struct SkillResourceListTool {
    registry: Arc<SkillRegistry>,
}

impl SkillResourceListTool {
    pub fn new(registry: Arc<SkillRegistry>) -> Self {
        Self { registry }
    }
}

pub struct SkillResourceReadTool {
    registry: Arc<SkillRegistry>,
}

impl SkillResourceReadTool {
    pub fn new(registry: Arc<SkillRegistry>) -> Self {
        Self { registry }
    }
}

#[async_trait]
impl ToolHandler for SkillTool {
    fn name(&self) -> &str {
        "skill"
    }

    fn description(&self) -> &str {
        "Load a registered local skill by name and return its full SKILL.md content plus metadata."
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Registered skill name from the injected skill cards"
                }
            },
            "required": ["name"],
            "additionalProperties": false
        })
    }

    fn permission_class(&self) -> ToolPermissionClass {
        ToolPermissionClass::Read
    }

    async fn execute(&self, args: Value) -> Result<Value> {
        let name = args
            .get("name")
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .ok_or_else(|| anyhow!("missing or invalid string argument: name"))?;
        let entry = self
            .registry
            .get(name)
            .ok_or_else(|| anyhow!("unknown skill: {name}"))?;

        Ok(json!({
            "name": entry.name.clone(),
            "description": entry.description.clone(),
            "content": entry.content.clone(),
            "base_dir": entry.base_dir.display().to_string(),
            "location": entry.location.clone(),
            "path": entry.path.display().to_string(),
            "files": sample_relative_files(&entry.base_dir, MAX_SKILL_FILE_SAMPLES)?,
        }))
    }
}

#[async_trait]
impl ToolHandler for SkillResourceListTool {
    fn name(&self) -> &str {
        "skill__resource_list"
    }

    fn description(&self) -> &str {
        "List relative resource file paths for a registered local skill."
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Registered skill name from the injected skill cards"
                }
            },
            "required": ["name"],
            "additionalProperties": false
        })
    }

    fn permission_class(&self) -> ToolPermissionClass {
        ToolPermissionClass::Read
    }

    async fn execute(&self, args: Value) -> Result<Value> {
        let name = args
            .get("name")
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .ok_or_else(|| anyhow!("missing or invalid string argument: name"))?;

        let files = self.registry.list_resource_paths(name)?;
        let resolved_name = self
            .registry
            .get(name)
            .map(|entry| entry.name.clone())
            .ok_or_else(|| anyhow!("unknown skill: {name}"))?;

        Ok(json!({
            "name": resolved_name,
            "files": files,
        }))
    }
}

#[async_trait]
impl ToolHandler for SkillResourceReadTool {
    fn name(&self) -> &str {
        "skill__resource_read"
    }

    fn description(&self) -> &str {
        "Read a UTF-8 resource file from a registered local skill by relative path."
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Registered skill name from the injected skill cards"
                },
                "path": {
                    "type": "string",
                    "description": "Relative resource path within the skill directory"
                }
            },
            "required": ["name", "path"],
            "additionalProperties": false
        })
    }

    fn permission_class(&self) -> ToolPermissionClass {
        ToolPermissionClass::Read
    }

    async fn execute(&self, args: Value) -> Result<Value> {
        let name = args
            .get("name")
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .ok_or_else(|| anyhow!("missing or invalid string argument: name"))?;
        let path = args
            .get("path")
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .ok_or_else(|| anyhow!("missing or invalid string argument: path"))?;

        let content = self.registry.read_resource(name, path)?;
        let resolved_name = self
            .registry
            .get(name)
            .map(|entry| entry.name.clone())
            .ok_or_else(|| anyhow!("unknown skill: {name}"))?;

        Ok(json!({
            "name": resolved_name,
            "path": path,
            "content": content,
        }))
    }
}

fn discover_skill_entries(root: &Path, location: &str) -> Result<Vec<SkillEntry>> {
    if !root.exists() {
        return Ok(Vec::new());
    }
    if !root.is_dir() {
        bail!("skills root is not a directory: {}", root.display());
    }

    let mut skill_dirs = fs::read_dir(root)
        .with_context(|| format!("failed to read skills directory {}", root.display()))?
        .collect::<std::io::Result<Vec<_>>>()
        .with_context(|| format!("failed to enumerate skills directory {}", root.display()))?;
    skill_dirs.sort_by_key(|entry| entry.file_name());

    let mut skills = Vec::new();
    for entry in skill_dirs {
        let skill_dir = entry.path();
        if !skill_dir.is_dir() {
            continue;
        }

        let skill_file = skill_dir.join(SKILL_FILE_NAME);
        if !skill_file.exists() {
            continue;
        }

        skills.push(parse_skill_entry(&skill_file, location)?);
    }

    Ok(skills)
}

fn parse_skill_entry(skill_file: &Path, location: &str) -> Result<SkillEntry> {
    let metadata = fs::symlink_metadata(skill_file)
        .with_context(|| format!("failed to stat {}", skill_file.display()))?;
    if metadata.file_type().is_symlink() || !metadata.is_file() {
        bail!(
            "skill file must be a regular file: {}",
            skill_file.display()
        );
    }
    let size = metadata.len();
    if size > MAX_SKILL_MD_BYTES {
        bail!(
            "skill file {} is too large: {} bytes exceeds {} bytes",
            skill_file.display(),
            size,
            MAX_SKILL_MD_BYTES
        );
    }

    let content = fs::read_to_string(skill_file)
        .with_context(|| format!("failed to read {}", skill_file.display()))?;
    let parsed = parse_skill_markdown(&content)
        .with_context(|| format!("invalid skill file {}", skill_file.display()))?;

    let base_dir = skill_file
        .parent()
        .ok_or_else(|| {
            anyhow!(
                "skill file has no parent directory: {}",
                skill_file.display()
            )
        })?
        .to_path_buf();
    let dir_name = base_dir
        .file_name()
        .and_then(|name| name.to_str())
        .ok_or_else(|| {
            anyhow!(
                "skill directory name is not valid UTF-8: {}",
                base_dir.display()
            )
        })?;
    validate_skill_name(&parsed.name)?;
    if dir_name != parsed.name {
        bail!(
            "skill directory '{}' must match name '{}' for {}",
            dir_name,
            parsed.name,
            skill_file.display()
        );
    }

    Ok(SkillEntry {
        name: parsed.name,
        description: parsed.description,
        body: parsed.body,
        content,
        location: location.to_string(),
        path: skill_file.to_path_buf(),
        base_dir,
    })
}

#[derive(Debug)]
struct ParsedSkill {
    name: String,
    description: String,
    body: String,
}

fn parse_skill_markdown(content: &str) -> Result<ParsedSkill> {
    let all_lines = content.lines().collect::<Vec<_>>();
    if all_lines.first().copied() != Some("---") {
        bail!("missing opening frontmatter delimiter '---'");
    }

    let mut name = None;
    let mut description = None;
    let mut body_start = None;

    for (index, line) in all_lines.iter().enumerate().skip(1) {
        if *line == "---" {
            body_start = Some(index + 1);
            break;
        }
    }

    let body_start =
        body_start.ok_or_else(|| anyhow!("missing closing frontmatter delimiter '---'"))?;
    let frontmatter_end = body_start - 1;
    let mut index = 1;
    while index < frontmatter_end {
        let line = all_lines[index];
        if line.trim().is_empty() {
            index += 1;
            continue;
        }

        let Some((key, raw_value)) = line.split_once(':') else {
            if line.chars().next().is_some_and(char::is_whitespace)
                || line.trim_start().starts_with('-')
            {
                index += 1;
                continue;
            }
            bail!("invalid frontmatter line '{}': expected key: value", line);
        };
        let key = key.trim();
        let raw_value = raw_value.trim();
        let (value, next_index) =
            parse_frontmatter_value(raw_value, &all_lines, index + 1, frontmatter_end);
        match key {
            "name" => name = Some(value),
            "description" => description = Some(value),
            _ => {}
        }
        index = next_index;
    }

    let name = name
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
        .ok_or_else(|| anyhow!("missing frontmatter field 'name'"))?;
    let description = description
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
        .ok_or_else(|| anyhow!("missing frontmatter field 'description'"))?;
    let body = all_lines[body_start..].join("\n");

    Ok(ParsedSkill {
        name,
        description,
        body,
    })
}

fn parse_frontmatter_value(
    raw_value: &str,
    all_lines: &[&str],
    mut next_index: usize,
    frontmatter_end: usize,
) -> (String, usize) {
    let is_folded_block = matches!(raw_value, ">" | ">-");
    let is_literal_block = matches!(raw_value, "|" | "|-");
    if !is_folded_block && !is_literal_block {
        return (parse_scalar(raw_value), next_index);
    }

    let mut parts = Vec::new();
    while next_index < frontmatter_end {
        let line = all_lines[next_index];
        if !line.trim().is_empty()
            && !line.chars().next().is_some_and(char::is_whitespace)
            && !line.trim_start().starts_with('-')
        {
            break;
        }
        if !line.trim().is_empty() {
            parts.push(line.trim().to_string());
        }
        next_index += 1;
    }

    let value = if is_folded_block {
        parts.join(" ")
    } else {
        parts.join("\n")
    };
    (value, next_index)
}

fn parse_scalar(value: &str) -> String {
    let stripped = value.trim();
    if stripped.len() >= 2 {
        let first = stripped.chars().next().unwrap_or_default();
        let last = stripped.chars().last().unwrap_or_default();
        if (first == '"' && last == '"') || (first == '\'' && last == '\'') {
            return stripped[1..stripped.len() - 1].trim().to_string();
        }
    }
    stripped.to_string()
}

pub fn normalize_skill_name(name: &str) -> String {
    let mut normalized = String::new();
    let mut previous_was_sep = false;

    for ch in name.chars() {
        if ch.is_ascii_alphanumeric() {
            normalized.push(ch.to_ascii_lowercase());
            previous_was_sep = false;
        } else if !previous_was_sep && !normalized.is_empty() {
            normalized.push('-');
            previous_was_sep = true;
        }
    }

    normalized.trim_matches('-').to_string()
}

fn validate_skill_name(name: &str) -> Result<()> {
    let len = name.chars().count();
    if len == 0 || len > MAX_SKILL_NAME_CHARS {
        bail!("skill name must be 1-{MAX_SKILL_NAME_CHARS} characters");
    }
    let mut previous_was_dash = false;
    for (index, ch) in name.chars().enumerate() {
        match ch {
            'a'..='z' | '0'..='9' => previous_was_dash = false,
            '-' => {
                if index == 0 || previous_was_dash {
                    bail!(
                        "skill name must use lowercase kebab-case without leading, trailing, or repeated dashes"
                    );
                }
                previous_was_dash = true;
            }
            _ => bail!("skill name must use lowercase kebab-case"),
        }
    }
    if previous_was_dash {
        bail!(
            "skill name must use lowercase kebab-case without leading, trailing, or repeated dashes"
        );
    }
    Ok(())
}

fn skill_roots(
    config_dir: &Path,
    workspace_root: &Path,
    include_user_roots: bool,
) -> Vec<(PathBuf, String)> {
    let mut roots = Vec::new();
    let mut seen = BTreeSet::new();

    if include_user_roots {
        if let Some(home) = std::env::var_os("HOME").map(PathBuf::from) {
            push_skill_root(
                &mut roots,
                &mut seen,
                home.join(".config").join("opencode").join("skills"),
                "~/.config/opencode/skills",
            );
            push_skill_root(
                &mut roots,
                &mut seen,
                home.join(".agents").join("skills"),
                "~/.agents/skills",
            );
            push_skill_root(
                &mut roots,
                &mut seen,
                home.join(".claude").join("skills"),
                "~/.claude/skills",
            );
        }
    }

    if !include_user_roots {
        push_skill_root(
            &mut roots,
            &mut seen,
            config_dir.join("skills"),
            "letcode config skills",
        );
    } else {
        push_skill_root(
            &mut roots,
            &mut seen,
            config_dir.join("skills"),
            "letcode config skills",
        );
    }

    for ancestor in workspace_skill_ancestors(workspace_root).into_iter().rev() {
        push_skill_root(
            &mut roots,
            &mut seen,
            ancestor.join(".agents").join("skills"),
            ".agents/skills",
        );
        push_skill_root(
            &mut roots,
            &mut seen,
            ancestor.join(".claude").join("skills"),
            ".claude/skills",
        );
        push_skill_root(
            &mut roots,
            &mut seen,
            ancestor.join(".opencode").join("skills"),
            ".opencode/skills",
        );
        push_skill_root(
            &mut roots,
            &mut seen,
            ancestor.join(".letcode").join("skills"),
            ".letcode/skills",
        );
    }

    roots
}

fn workspace_skill_ancestors(workspace_root: &Path) -> Vec<&Path> {
    let mut ancestors = Vec::new();
    for ancestor in workspace_root.ancestors() {
        ancestors.push(ancestor);
        if ancestor.join(".git").exists() {
            break;
        }
    }
    if ancestors
        .last()
        .is_some_and(|ancestor| ancestor.join(".git").exists())
    {
        ancestors
    } else {
        vec![workspace_root]
    }
}

fn push_skill_root(
    roots: &mut Vec<(PathBuf, String)>,
    seen: &mut BTreeSet<PathBuf>,
    root: PathBuf,
    location: &str,
) {
    if seen.insert(root.clone()) {
        roots.push((root, location.to_string()));
    }
}

fn sample_relative_files(base_dir: &Path, limit: usize) -> Result<Vec<String>> {
    let mut files = Vec::new();
    collect_relative_files(base_dir, base_dir, &mut files, limit, 0)?;
    files.sort();
    Ok(files)
}

fn list_relative_resource_files(base_dir: &Path) -> Result<Vec<String>> {
    let mut files = Vec::new();
    collect_relative_resource_files(base_dir, base_dir, &mut files)?;
    files.sort();
    Ok(files)
}

fn collect_relative_files(
    base_dir: &Path,
    current: &Path,
    files: &mut Vec<String>,
    limit: usize,
    depth: usize,
) -> Result<()> {
    if files.len() >= limit || depth > MAX_SKILL_FILE_DEPTH {
        return Ok(());
    }
    let mut entries = fs::read_dir(current)
        .with_context(|| format!("failed to read skill directory {}", current.display()))?
        .collect::<std::io::Result<Vec<_>>>()
        .with_context(|| format!("failed to enumerate skill directory {}", current.display()))?;
    entries.sort_by_key(|entry| entry.file_name());

    for entry in entries {
        if files.len() >= limit {
            break;
        }
        let path = entry.path();
        let metadata = fs::symlink_metadata(&path)
            .with_context(|| format!("failed to inspect skill path {}", path.display()))?;
        if metadata.file_type().is_symlink() {
            continue;
        }
        if metadata.is_dir() {
            collect_relative_files(base_dir, &path, files, limit, depth + 1)?;
        } else if metadata.is_file() {
            let relative = path
                .strip_prefix(base_dir)
                .with_context(|| format!("failed to make relative path for {}", path.display()))?;
            if relative == Path::new(SKILL_FILE_NAME) {
                continue;
            }
            files.push(relative.to_string_lossy().to_string());
        }
    }

    Ok(())
}

fn collect_relative_resource_files(
    base_dir: &Path,
    current: &Path,
    files: &mut Vec<String>,
) -> Result<()> {
    let mut entries = fs::read_dir(current)
        .with_context(|| format!("failed to read skill directory {}", current.display()))?
        .collect::<std::io::Result<Vec<_>>>()
        .with_context(|| format!("failed to enumerate skill directory {}", current.display()))?;
    entries.sort_by_key(|entry| entry.file_name());

    for entry in entries {
        let path = entry.path();
        let metadata = fs::symlink_metadata(&path)
            .with_context(|| format!("failed to inspect skill path {}", path.display()))?;
        if metadata.file_type().is_symlink() {
            continue;
        }
        if metadata.is_dir() {
            collect_relative_resource_files(base_dir, &path, files)?;
            continue;
        }
        if !metadata.is_file() {
            continue;
        }

        let relative = path
            .strip_prefix(base_dir)
            .with_context(|| format!("failed to make relative path for {}", path.display()))?;
        if relative == Path::new(SKILL_FILE_NAME) {
            continue;
        }
        files.push(relative.to_string_lossy().to_string());
    }

    Ok(())
}

fn resolve_skill_resource_path(base_dir: &Path, raw_path: &str) -> Result<PathBuf> {
    let candidate = Path::new(raw_path);
    if candidate.as_os_str().is_empty() {
        bail!("resource path must not be empty");
    }
    if candidate.is_absolute() {
        bail!("resource path must be relative");
    }
    if candidate == Path::new(SKILL_FILE_NAME) {
        bail!("resource path must not reference {SKILL_FILE_NAME}");
    }

    let mut resolved = base_dir.to_path_buf();
    let mut components = candidate.components().peekable();
    while let Some(component) = components.next() {
        match component {
            std::path::Component::Normal(part) => {
                resolved.push(part);
                let metadata = fs::symlink_metadata(&resolved).with_context(|| {
                    format!("failed to inspect skill resource {}", resolved.display())
                })?;
                if metadata.file_type().is_symlink() {
                    bail!("resource path must not traverse symlinks: {raw_path}");
                }
                if components.peek().is_some() {
                    if !metadata.is_dir() {
                        bail!("resource path component is not a directory: {raw_path}");
                    }
                } else if !metadata.is_file() {
                    bail!("skill resource must be a regular file: {raw_path}");
                }
            }
            std::path::Component::CurDir => {
                bail!("resource path must not contain '.' components");
            }
            std::path::Component::ParentDir => {
                bail!("resource path must not contain '..' components");
            }
            std::path::Component::RootDir | std::path::Component::Prefix(_) => {
                bail!("resource path must be relative");
            }
        }
    }

    Ok(resolved)
}

fn read_utf8_resource_file(path: &Path) -> Result<String> {
    let metadata = fs::symlink_metadata(path)
        .with_context(|| format!("failed to inspect skill resource {}", path.display()))?;
    if metadata.file_type().is_symlink() || !metadata.is_file() {
        bail!("skill resource must be a regular file: {}", path.display());
    }
    let size = metadata.len();
    if size > MAX_SKILL_RESOURCE_BYTES {
        bail!(
            "skill resource {} is too large: {} bytes exceeds {} bytes",
            path.display(),
            size,
            MAX_SKILL_RESOURCE_BYTES
        );
    }

    fs::read_to_string(path).with_context(|| format!("failed to read {} as UTF-8", path.display()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::request_builder::HistoryToolCall;
    use crate::runtime_context::SourceSpan;
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::time::{SystemTime, UNIX_EPOCH};

    static TEMP_DIR_COUNTER: AtomicU64 = AtomicU64::new(0);

    struct TempDir {
        path: PathBuf,
    }

    impl TempDir {
        fn new() -> Self {
            let unique = format!(
                "letcode-skills-{}-{}-{}",
                std::process::id(),
                TEMP_DIR_COUNTER.fetch_add(1, Ordering::Relaxed),
                SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .expect("system time")
                    .as_nanos()
            );
            let path = std::env::temp_dir().join(unique);
            fs::create_dir_all(&path).expect("create temp dir");
            Self { path }
        }

        fn path(&self) -> &Path {
            &self.path
        }
    }

    impl Drop for TempDir {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.path);
        }
    }

    fn write_skill(base: &Path, root: &str, dir: &str, content: &str) -> PathBuf {
        let skill_dir = base.join(root).join(dir);
        fs::create_dir_all(&skill_dir).expect("create skill dir");
        let path = skill_dir.join(SKILL_FILE_NAME);
        fs::write(&path, content).expect("write skill file");
        path
    }

    fn load_test_registry(config_dir: &Path, workspace_root: &Path) -> Result<SkillRegistry> {
        SkillRegistry::load_from_roots(skill_roots(config_dir, workspace_root, false))
    }

    #[test]
    fn persisted_skill_parser_ignores_malformed_and_failed_outputs() {
        for output in [
            "not json",
            r#"{"ok":true,"tool":"skill","data":{"name":3,"content":"body"}}"#,
            r#"{"ok":true,"tool":"skill","data":{"name":"skill"}}"#,
            r#"{"ok":false,"tool":"skill","data":{"name":"skill","content":"body"}}"#,
            r#"{"ok":true,"tool":"other","data":{"name":"skill","content":"body"}}"#,
        ] {
            assert_eq!(
                parse_persisted_skill_output(output).expect("parser is tolerant"),
                None
            );
        }
        assert_eq!(
            parse_persisted_skill_output(
                r#"{"ok":true,"tool":"skill","data":{"name":"skill","content":"body"}}"#
            )
            .expect("valid output parses"),
            Some(("skill".into(), "body".into()))
        );
    }

    #[test]
    fn formats_and_parses_manual_skill_markers_in_stable_deduplicated_order() {
        assert_eq!(
            format_manual_skill_marker("rust-audit").expect("valid marker"),
            "@skill(rust-audit)"
        );
        assert_eq!(
            parse_manual_skill_markers(
                "Use @skill(rust-audit), then @skill(git), then @skill(rust-audit)."
            )
            .expect("markers parse"),
            vec!["rust-audit", "git"]
        );
        assert!(
            parse_manual_skill_markers("@skill()")
                .expect_err("empty explicit marker fails")
                .to_string()
                .contains("invalid skill marker")
        );
        assert!(
            parse_manual_skill_markers("@skill(rust-audit")
                .expect_err("unclosed explicit marker fails")
                .to_string()
                .contains("missing ')'")
        );
        assert!(
            parse_manual_skill_markers("ordinary @skill text stays ordinary")
                .expect("ordinary text")
                .is_empty()
        );
    }

    fn skill_call_frame(call_id: &str, sequence: u64, visibility: FrameVisibility) -> RuntimeFrame {
        let span = SourceSpan::new(sequence, sequence).expect("valid source span");
        RuntimeFrame::new(
            RuntimeFrameKind::ToolCall,
            visibility,
            RuntimeFrameProvenance::new(RuntimeSource::Transcript).with_span(span),
            RuntimeFrameIdSeed {
                frame_kind: RuntimeFrameKind::ToolCall,
                source: RuntimeSource::Transcript,
                ordinal: sequence as u32,
                stable_key: call_id,
                source_span: Some(span),
            },
        )
        .with_protocol(ProtocolFrameItem::AssistantToolCalls {
            text: None,
            calls: vec![HistoryToolCall {
                call_id: call_id.into(),
                name: "skill".into(),
                arguments_json: "{}".into(),
            }],
        })
    }

    fn skill_output_frame(
        call_id: &str,
        output_json: &str,
        sequence: u64,
        visibility: FrameVisibility,
    ) -> RuntimeFrame {
        let span = SourceSpan::new(sequence, sequence).expect("valid source span");
        RuntimeFrame::new(
            RuntimeFrameKind::ToolOutput,
            visibility,
            RuntimeFrameProvenance::new(RuntimeSource::Transcript).with_span(span),
            RuntimeFrameIdSeed {
                frame_kind: RuntimeFrameKind::ToolOutput,
                source: RuntimeSource::Transcript,
                ordinal: sequence as u32,
                stable_key: call_id,
                source_span: Some(span),
            },
        )
        .with_protocol(ProtocolFrameItem::ToolOutput {
            call_id: call_id.into(),
            output_json: output_json.into(),
        })
    }

    fn unspanned_skill_output_frame(
        call_id: &str,
        output_json: &str,
        stable_key: &str,
        ordinal: u32,
        visibility: FrameVisibility,
    ) -> RuntimeFrame {
        RuntimeFrame::new(
            RuntimeFrameKind::ToolOutput,
            visibility,
            RuntimeFrameProvenance::new(RuntimeSource::Transcript),
            RuntimeFrameIdSeed {
                frame_kind: RuntimeFrameKind::ToolOutput,
                source: RuntimeSource::Transcript,
                ordinal,
                stable_key,
                source_span: None,
            },
        )
        .with_protocol(ProtocolFrameItem::ToolOutput {
            call_id: call_id.into(),
            output_json: output_json.into(),
        })
    }

    #[test]
    fn reconciliation_skips_malformed_skill_occurrence_and_keeps_later_valid_material() {
        let mut snapshot = RuntimeSnapshot::new("main");
        snapshot.push_frame(skill_call_frame("malformed", 1, FrameVisibility::Retired));
        snapshot.push_frame(skill_output_frame(
            "malformed",
            "not json",
            2,
            FrameVisibility::Retired,
        ));
        snapshot.push_frame(skill_call_frame("valid", 3, FrameVisibility::Active));
        snapshot.push_frame(skill_output_frame(
            "valid",
            r#"{"ok":true,"tool":"skill","data":{"name":"persisted-only","content":"exact persisted body"}}"#,
            4,
            FrameVisibility::Active,
        ));

        reconcile_loaded_skill_material(&mut snapshot).expect("reconciliation succeeds");

        assert_eq!(snapshot.prompt_contributors.len(), 1);
        let contributor = &snapshot.prompt_contributors[0];
        assert_eq!(contributor.contributor_id, "skill-material:valid");
        assert_eq!(contributor.label.as_deref(), Some("persisted-only"));
        let material = snapshot
            .frames
            .iter()
            .find(|frame| frame.id == contributor.frame_ids[0])
            .expect("detached material frame");
        assert_eq!(
            material
                .prompt_payload
                .as_ref()
                .map(|payload| payload.text.as_str()),
            Some("exact persisted body")
        );
    }

    #[test]
    fn reconciliation_overwrites_stale_detached_skill_material_from_persisted_tool_result() {
        let mut snapshot = RuntimeSnapshot::new("main");
        snapshot.push_frame(skill_call_frame("load", 1, FrameVisibility::Active));
        snapshot.push_frame(skill_output_frame(
            "load",
            r#"{"ok":true,"tool":"skill","data":{"name":"authoritative","content":"authoritative body"}}"#,
            2,
            FrameVisibility::Active,
        ));
        reconcile_loaded_skill_material(&mut snapshot).expect("initial reconciliation");

        snapshot.prompt_contributors[0].label = Some("stale label".into());
        let detached_id = snapshot.prompt_contributors[0].frame_ids[0];
        let detached = snapshot
            .frames
            .iter_mut()
            .find(|frame| frame.id == detached_id)
            .expect("detached material frame");
        detached.summary = Some("Skill: stale summary".into());
        detached.prompt_payload = Some(RuntimePromptPayload {
            role: RuntimePromptRole::System,
            text: "stale body".into(),
        });

        reconcile_loaded_skill_material(&mut snapshot).expect("repeat reconciliation");

        assert_eq!(
            snapshot.prompt_contributors[0].label.as_deref(),
            Some("authoritative")
        );
        let detached = snapshot
            .frames
            .iter()
            .find(|frame| frame.id == detached_id)
            .expect("detached material frame");
        assert_eq!(detached.summary.as_deref(), Some("Skill: authoritative"));
        assert_eq!(
            detached.prompt_payload.as_ref(),
            Some(&RuntimePromptPayload {
                role: RuntimePromptRole::Developer,
                text: "authoritative body".into(),
            })
        );
    }

    #[test]
    fn reconciliation_uses_persisted_content_and_source_span_order_without_skill_files() {
        let mut snapshot = RuntimeSnapshot::new("main");
        snapshot.push_frame(skill_call_frame("later", 30, FrameVisibility::Active));
        snapshot.push_frame(skill_output_frame(
            "later",
            r#"{"ok":true,"tool":"skill","data":{"name":"later","content":"later persisted body"}}"#,
            31,
            FrameVisibility::Active,
        ));
        snapshot.push_frame(skill_call_frame("earlier", 10, FrameVisibility::Retired));
        snapshot.push_frame(skill_output_frame(
            "earlier",
            r#"{"ok":true,"tool":"skill","data":{"name":"earlier","content":"earlier persisted body"}}"#,
            11,
            FrameVisibility::Retired,
        ));

        reconcile_loaded_skill_material(&mut snapshot).expect("reconciliation succeeds");

        assert_eq!(
            snapshot
                .prompt_contributors
                .iter()
                .map(|contributor| contributor.contributor_id.as_str())
                .collect::<Vec<_>>(),
            vec!["skill-material:earlier", "skill-material:later"]
        );
        assert!(snapshot.frames.iter().any(|frame| {
            frame
                .prompt_payload
                .as_ref()
                .is_some_and(|payload| payload.text == "earlier persisted body")
        }));
    }

    #[test]
    fn reconciliation_orders_unspanned_live_skill_outputs_by_snapshot_position() {
        let mut snapshot = RuntimeSnapshot::new("main");
        let output = |name: &str| {
            format!(
                r#"{{"ok":true,"tool":"skill","data":{{"name":"{name}","content":"{name} body"}}}}"#
            )
        };
        let mut outputs = vec![
            unspanned_skill_output_frame(
                "first",
                &output("first"),
                "first-id",
                0,
                FrameVisibility::Active,
            ),
            unspanned_skill_output_frame(
                "second",
                &output("second"),
                "second-id",
                1,
                FrameVisibility::Active,
            ),
            unspanned_skill_output_frame(
                "third",
                &output("third"),
                "third-id",
                2,
                FrameVisibility::Active,
            ),
        ];
        // Deliberately make protocol insertion order disagree with ID order.
        outputs.sort_by_key(|frame| std::cmp::Reverse(frame.id));
        let expected_calls = outputs
            .iter()
            .filter_map(|frame| match frame.protocol.as_ref() {
                Some(ProtocolFrameItem::ToolOutput { call_id, .. }) => Some(call_id.clone()),
                _ => None,
            })
            .collect::<Vec<_>>();
        assert_ne!(
            expected_calls,
            ["first", "second", "third"],
            "test insertion order differs from RuntimeFrameId order"
        );
        for call_id in &expected_calls {
            snapshot.push_frame(skill_call_frame(call_id, 10, FrameVisibility::Active));
        }
        for frame in outputs {
            snapshot.push_frame(frame);
        }

        reconcile_loaded_skill_material(&mut snapshot).expect("initial reconciliation");
        let expected_contributors = expected_calls
            .iter()
            .map(|call_id| format!("skill-material:{call_id}"))
            .collect::<Vec<_>>();
        assert_eq!(
            snapshot
                .prompt_contributors
                .iter()
                .map(|contributor| contributor.contributor_id.clone())
                .collect::<Vec<_>>(),
            expected_contributors
        );
        let detached_ids = snapshot
            .prompt_contributors
            .iter()
            .map(|contributor| contributor.frame_ids.clone())
            .collect::<Vec<_>>();

        reconcile_loaded_skill_material(&mut snapshot).expect("repeat reconciliation");
        assert_eq!(
            snapshot
                .prompt_contributors
                .iter()
                .map(|contributor| contributor.contributor_id.clone())
                .collect::<Vec<_>>(),
            expected_contributors
        );
        assert_eq!(
            snapshot
                .prompt_contributors
                .iter()
                .map(|contributor| contributor.frame_ids.clone())
                .collect::<Vec<_>>(),
            detached_ids
        );

        for frame in &mut snapshot.frames {
            if matches!(frame.protocol, Some(ProtocolFrameItem::ToolOutput { .. })) {
                frame.visibility = FrameVisibility::Retired;
            }
        }
        reconcile_loaded_skill_material(&mut snapshot).expect("reconciliation after retirement");
        assert_eq!(
            snapshot
                .prompt_contributors
                .iter()
                .map(|contributor| contributor.contributor_id.clone())
                .collect::<Vec<_>>(),
            expected_contributors
        );
        assert_eq!(
            snapshot
                .prompt_contributors
                .iter()
                .map(|contributor| contributor.frame_ids.clone())
                .collect::<Vec<_>>(),
            detached_ids
        );
    }

    #[test]
    fn reconciliation_preserves_material_only_for_structurally_pruned_output() {
        let mut snapshot = RuntimeSnapshot::new("main");
        snapshot.push_frame(skill_call_frame("load", 1, FrameVisibility::Active));
        snapshot.push_frame(skill_output_frame(
            "load",
            r#"{"ok":true,"tool":"skill","data":{"name":"exact-name","content":"exact persisted body"}}"#,
            2,
            FrameVisibility::Active,
        ));
        reconcile_loaded_skill_material(&mut snapshot).expect("initial reconciliation");
        let original_contributor = snapshot.prompt_contributors[0].clone();
        let detached_id = original_contributor.frame_ids[0];
        let source = snapshot
            .frames
            .iter_mut()
            .find(|frame| frame.id == original_contributor.source_frame_ids[0])
            .expect("source output");
        let ProtocolFrameItem::ToolOutput { output_json, .. } = source.protocol.as_mut().unwrap()
        else {
            panic!("source is a tool output");
        };
        *output_json = r#"{"_compaction":{"pruned":true,"reason":"tool output pruned by compaction.prune","original_chars":9999,"tool":"skill"}}"#.into();

        reconcile_loaded_skill_material(&mut snapshot).expect("pruned reconciliation");
        assert_eq!(
            snapshot.prompt_contributors,
            vec![original_contributor.clone()]
        );
        assert_eq!(
            snapshot
                .frames
                .iter()
                .find(|frame| frame.id == detached_id)
                .and_then(|frame| frame.prompt_payload.as_ref())
                .map(|payload| payload.text.as_str()),
            Some("exact persisted body")
        );

        let source = snapshot
            .frames
            .iter_mut()
            .find(|frame| frame.id == original_contributor.source_frame_ids[0])
            .expect("source output");
        let ProtocolFrameItem::ToolOutput { output_json, .. } = source.protocol.as_mut().unwrap()
        else {
            panic!("source is a tool output");
        };
        *output_json = "not a structural compaction marker".into();
        reconcile_loaded_skill_material(&mut snapshot).expect("malformed reconciliation");
        assert!(snapshot.prompt_contributors.is_empty());
        assert!(snapshot.frames.iter().all(|frame| frame.id != detached_id));
    }

    #[test]
    fn parses_skill_markdown_frontmatter_and_body() {
        let parsed = parse_skill_markdown(
            "---\nname: rust-audit\ndescription: \"Inspect Rust code\"\nignored: value\n---\n# Heading\nUse this skill.\n",
        )
        .expect("skill parses");

        assert_eq!(parsed.name, "rust-audit");
        assert_eq!(parsed.description, "Inspect Rust code");
        assert_eq!(parsed.body, "# Heading\nUse this skill.");
    }

    #[test]
    fn parses_folded_frontmatter_description() {
        let parsed = parse_skill_markdown(
            "---\nname: git\ndescription: >-\n  Use for git workflows\n  including commits and PRs.\nmetadata:\n  area: vcs\n---\n# Git\n",
        )
        .expect("skill parses");

        assert_eq!(
            parsed.description,
            "Use for git workflows including commits and PRs."
        );
    }

    #[test]
    fn parses_long_skill_description_without_truncation() {
        let long_description = "Use this skill for detailed workflows. ".repeat(80);
        let content = format!(
            "---\nname: complex-skill\ndescription: >-\n  {}\n---\n# Complex\n",
            long_description
        );
        let parsed = parse_skill_markdown(&content).expect("long description parses");

        assert_eq!(parsed.description, long_description.trim());
    }

    #[test]
    fn parser_rejects_invalid_skill_markdown() {
        let error = parse_skill_markdown("name: nope\ndescription: missing frontmatter\n")
            .expect_err("invalid skill should fail");
        assert!(
            error
                .to_string()
                .contains("missing opening frontmatter delimiter")
        );
    }

    #[test]
    fn registry_rejects_non_kebab_case_skill_name() {
        let temp = TempDir::new();
        write_skill(
            temp.path(),
            "config/skills",
            "rust-audit",
            "---\nname: Rust Audit\ndescription: Inspect Rust code\n---\n# Rust\n",
        );

        let error = load_test_registry(&temp.path().join("config"), temp.path())
            .expect_err("invalid name should fail");
        assert!(error.to_string().contains("lowercase kebab-case"));
    }

    #[test]
    fn registry_rejects_oversized_skill_file() {
        let temp = TempDir::new();
        let large_body = "x".repeat(MAX_SKILL_MD_BYTES as usize + 1);
        write_skill(
            temp.path(),
            "config/skills",
            "large-skill",
            &format!("---\nname: large-skill\ndescription: Large\n---\n{large_body}"),
        );

        let error = load_test_registry(&temp.path().join("config"), temp.path())
            .expect_err("oversized skill should fail");
        assert!(error.to_string().contains("is too large"));
    }

    #[test]
    fn registry_discovers_config_and_workspace_skills() {
        let temp = TempDir::new();
        write_skill(
            temp.path(),
            "config/skills",
            "rust-audit",
            "---\nname: rust-audit\ndescription: Inspect Rust code\n---\n# Rust\n",
        );
        write_skill(
            temp.path(),
            ".letcode/skills",
            "project-skill",
            "---\nname: project-skill\ndescription: Project-local helper\n---\n# Project\n",
        );

        let registry =
            load_test_registry(&temp.path().join("config"), temp.path()).expect("registry loads");
        let cards = registry.cards();

        assert_eq!(cards.len(), 2);
        assert!(cards.iter().any(|card| card.name == "rust-audit"));
        assert!(cards.iter().any(|card| card.location == ".letcode/skills"));
    }

    #[test]
    fn registry_returns_empty_when_skills_directory_is_missing() {
        let temp = TempDir::new();
        let registry = load_test_registry(&temp.path().join("config"), temp.path())
            .expect("registry loads without skills dir");
        assert!(registry.is_empty());
    }

    #[test]
    fn registry_discovers_parent_opencode_skills() {
        let temp = TempDir::new();
        fs::create_dir_all(temp.path().join(".git")).expect("create repo marker");
        write_skill(
            temp.path(),
            ".opencode/skills",
            "repo-skill",
            "---\nname: repo-skill\ndescription: Repo helper\n---\n# Repo\n",
        );
        let nested = temp.path().join("src/module");
        fs::create_dir_all(&nested).expect("create nested workspace");

        let registry = load_test_registry(&temp.path().join("config"), &nested)
            .expect("registry loads parent skill");
        assert!(
            registry
                .cards()
                .iter()
                .any(|card| card.name == "repo-skill")
        );
    }

    #[test]
    fn registry_does_not_discover_skills_above_repo_root() {
        let temp = TempDir::new();
        let outer = temp.path().join("outer");
        let repo = outer.join("repo");
        fs::create_dir_all(repo.join(".git")).expect("create repo marker");
        write_skill(
            &outer,
            ".opencode/skills",
            "upper-skill",
            "---\nname: upper-skill\ndescription: Upper helper\n---\n# Upper\n",
        );
        let nested = repo.join("src/module");
        fs::create_dir_all(&nested).expect("create nested workspace");

        let registry = load_test_registry(&temp.path().join("config"), &nested)
            .expect("registry loads without upper skill");
        assert!(
            !registry
                .cards()
                .iter()
                .any(|card| card.name == "upper-skill")
        );
    }

    #[test]
    fn later_discovered_skill_overrides_earlier_skill() {
        let registry = SkillRegistry::from_entries(vec![
            SkillEntry {
                name: "same-skill".into(),
                description: "old".into(),
                body: "old".into(),
                content: "old".into(),
                location: "old".into(),
                path: PathBuf::from("/old/SKILL.md"),
                base_dir: PathBuf::from("/old"),
            },
            SkillEntry {
                name: "same-skill".into(),
                description: "new".into(),
                body: "new".into(),
                content: "new".into(),
                location: "new".into(),
                path: PathBuf::from("/new/SKILL.md"),
                base_dir: PathBuf::from("/new"),
            },
        ])
        .expect("registry allows precedence override");

        assert_eq!(
            registry.get("same-skill").expect("skill").description,
            "new"
        );
    }

    #[test]
    fn registry_rejects_directory_name_mismatch() {
        let temp = TempDir::new();
        let skill_path = write_skill(
            temp.path(),
            "config/skills",
            "wrong-name",
            "---\nname: correct-name\ndescription: Desc\n---\nBody\n",
        );

        let error = load_test_registry(&temp.path().join("config"), temp.path())
            .expect_err("mismatch should fail");
        assert!(error.to_string().contains("must match name 'correct-name'"));
        assert!(
            error
                .to_string()
                .contains(&skill_path.display().to_string())
        );
    }

    #[tokio::test]
    async fn skill_tool_returns_full_skill_content_and_sampled_files() {
        let temp = TempDir::new();
        let skill_dir = temp.path().join("config/skills/rust-audit");
        fs::create_dir_all(skill_dir.join("notes")).expect("create nested dirs");
        fs::write(
            skill_dir.join(SKILL_FILE_NAME),
            "---\nname: rust-audit\ndescription: Inspect Rust code\n---\n# Rust\nRead the code.\n",
        )
        .expect("write skill file");
        fs::write(skill_dir.join("notes/context.txt"), "context").expect("write sample file");

        let registry = Arc::new(
            load_test_registry(&temp.path().join("config"), temp.path()).expect("registry"),
        );
        let tool = SkillTool::new(registry);
        let result = tool
            .execute(json!({"name": "rust-audit"}))
            .await
            .expect("skill loads");

        assert_eq!(result["name"], json!("rust-audit"));
        assert_eq!(result["description"], json!("Inspect Rust code"));
        assert!(
            result["content"]
                .as_str()
                .expect("content str")
                .contains("# Rust")
        );
        assert!(
            !result["files"]
                .as_array()
                .expect("files array")
                .iter()
                .any(|value| value == "SKILL.md")
        );
        assert!(
            result["files"]
                .as_array()
                .expect("files array")
                .iter()
                .any(|value| value == "notes/context.txt")
        );
    }

    #[tokio::test]
    async fn skill_tool_rejects_unknown_skill() {
        let registry = Arc::new(SkillRegistry::default());
        let tool = SkillTool::new(registry);
        let error = tool
            .execute(json!({"name": "missing"}))
            .await
            .expect_err("unknown skill should fail");

        assert!(error.to_string().contains("unknown skill: missing"));
    }

    #[tokio::test]
    async fn skill_resource_list_tool_lists_sorted_regular_relative_files() {
        let temp = TempDir::new();
        let skill_dir = temp.path().join("config/skills/rust-audit");
        fs::create_dir_all(skill_dir.join("docs")).expect("create docs dir");
        fs::write(
            skill_dir.join(SKILL_FILE_NAME),
            "---\nname: rust-audit\ndescription: Inspect Rust code\n---\n# Rust\n",
        )
        .expect("write skill file");
        fs::write(skill_dir.join("z-last.txt"), "z").expect("write z file");
        fs::write(skill_dir.join("docs/a-first.txt"), "a").expect("write a file");

        let registry = Arc::new(
            load_test_registry(&temp.path().join("config"), temp.path()).expect("registry"),
        );
        let tool = SkillResourceListTool::new(registry);
        let result = tool
            .execute(json!({"name": "rust-audit"}))
            .await
            .expect("resource list loads");

        assert_eq!(result["name"], json!("rust-audit"));
        assert_eq!(result["files"], json!(["docs/a-first.txt", "z-last.txt"]));
    }

    #[tokio::test]
    async fn skill_resource_read_tool_rejects_absolute_and_parent_paths() {
        let temp = TempDir::new();
        let skill_dir = temp.path().join("config/skills/rust-audit");
        fs::create_dir_all(skill_dir.join("docs")).expect("create docs dir");
        fs::write(
            skill_dir.join(SKILL_FILE_NAME),
            "---\nname: rust-audit\ndescription: Inspect Rust code\n---\n# Rust\n",
        )
        .expect("write skill file");
        fs::write(skill_dir.join("docs/guide.txt"), "hello").expect("write guide");

        let registry = Arc::new(
            load_test_registry(&temp.path().join("config"), temp.path()).expect("registry"),
        );
        let tool = SkillResourceReadTool::new(registry);

        let absolute_error = tool
            .execute(json!({"name": "rust-audit", "path": "/tmp/nope.txt"}))
            .await
            .expect_err("absolute path should fail");
        assert!(absolute_error.to_string().contains("must be relative"));

        let traversal_error = tool
            .execute(json!({"name": "rust-audit", "path": "../secret.txt"}))
            .await
            .expect_err("traversal path should fail");
        assert!(
            traversal_error
                .to_string()
                .contains("must not contain '..'")
        );
    }

    #[tokio::test]
    async fn skill_resource_read_tool_reads_utf8_resource_content() {
        let temp = TempDir::new();
        let skill_dir = temp.path().join("config/skills/rust-audit");
        fs::create_dir_all(skill_dir.join("docs")).expect("create docs dir");
        fs::write(
            skill_dir.join(SKILL_FILE_NAME),
            "---\nname: rust-audit\ndescription: Inspect Rust code\n---\n# Rust\n",
        )
        .expect("write skill file");
        fs::write(skill_dir.join("docs/guide.txt"), "hello").expect("write guide");

        let registry = Arc::new(
            load_test_registry(&temp.path().join("config"), temp.path()).expect("registry"),
        );
        let tool = SkillResourceReadTool::new(registry);
        let result = tool
            .execute(json!({"name": "rust-audit", "path": "docs/guide.txt"}))
            .await
            .expect("resource read succeeds");

        assert_eq!(result["name"], json!("rust-audit"));
        assert_eq!(result["path"], json!("docs/guide.txt"));
        assert_eq!(result["content"], json!("hello"));
    }

    #[test]
    fn samples_skip_symlinked_directories() {
        let temp = TempDir::new();
        let base = temp.path().join("skill");
        let outside = temp.path().join("outside");
        fs::create_dir_all(&base).expect("create base");
        fs::create_dir_all(&outside).expect("create outside");
        fs::write(base.join("note.txt"), "note").expect("write note");
        fs::write(outside.join("secret.txt"), "secret").expect("write outside");

        #[cfg(unix)]
        std::os::unix::fs::symlink(&outside, base.join("linked")).expect("create symlink");

        let files = sample_relative_files(&base, MAX_SKILL_FILE_SAMPLES).expect("sample files");
        assert!(files.iter().any(|file| file == "note.txt"));
        assert!(!files.iter().any(|file| file.contains("secret.txt")));
    }
}
