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
    .with_protocol(ProtocolFrameItem::AssistantTurn {
        text: None,
        reasoning_content: None,
        replay: None,
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
        images: Vec::new(),
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
        images: Vec::new(),
    })
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

    for name in [
        "customize-letcode",
        "git",
        "verification-planning",
        "worktrees",
        "simplify",
    ] {
        let card = cards
            .iter()
            .find(|card| card.name == name)
            .unwrap_or_else(|| panic!("missing built-in skill {name}"));
        assert_eq!(card.location, BUILTIN_SKILL_LOCATION);
    }
    assert!(cards.iter().any(|card| card.name == "rust-audit"));
    assert!(cards.iter().any(|card| card.location == ".letcode/skills"));
    assert_eq!(cards.len(), 7);
}

#[test]
fn disk_skill_overrides_same_named_builtin() {
    let temp = TempDir::new();
    write_skill(
        temp.path(),
        "config/skills",
        "git",
        "---\nname: git\ndescription: Local Git workflow\n---\n# Local Git\n",
    );

    let registry =
        load_test_registry(&temp.path().join("config"), temp.path()).expect("registry loads");
    let git = registry.get("git").expect("git skill");

    assert_eq!(git.location, "letcode config skills");
    assert_eq!(git.description, "Local Git workflow");
    assert_eq!(git.body, "# Local Git");
}

#[test]
fn registry_rejects_directory_name_mismatch() {
    let temp = TempDir::new();
    write_skill(
        temp.path(),
        "config/skills",
        "wrong-name",
        "---\nname: correct-name\ndescription: Desc\n---\nBody\n",
    );

    let error = load_test_registry(&temp.path().join("config"), temp.path())
        .expect_err("mismatch should fail");
    assert!(error.to_string().contains("must match name 'correct-name'"));
    let rendered = error.to_string();
    assert!(rendered.ends_with(&format!(
        "wrong-name{}{}",
        std::path::MAIN_SEPARATOR,
        SKILL_FILE_NAME
    )));
}

mod integration_tests {
    use super::*;

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
}
