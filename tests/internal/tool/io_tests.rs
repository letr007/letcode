use super::apply_patch::ApplyPatchWorkerPoint;
use super::{
    ToolExecutionContext, ToolRegistry, external_workspace_access_for_tool,
    permission_resource_for_tool, prepare_apply_patch_targets, prepare_writable_leaf,
    secure_write_writable_leaf,
};
use crate::permission::ToolScope;
use crate::skills::{SkillEntry, SkillRegistry, SkillTool};
use base64::Engine as _;
use serde_json::{Value, json};
use std::path::PathBuf;
use std::sync::Arc;

async fn call_workflow_todos(items: serde_json::Value) -> crate::tool::ToolResult {
    ToolRegistry::default_tools()
        .call("workflow__todos", json!({"items": items}))
        .await
}

mod reads {
    use super::*;

    mod integration_tests {
        use super::*;

        #[tokio::test]
        async fn external_workspace_read_requires_explicit_grant() {
            let outside_path = std::env::temp_dir().join(format!(
                "letcode-outside-tool-read-{}.txt",
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .expect("system time")
                    .as_nanos()
            ));
            std::fs::write(&outside_path, "outside\n").expect("write outside fixture");
            let args = json!({
                "path": outside_path.to_string_lossy(),
                "offset": 1,
                "limit": 10,
            });

            let access = external_workspace_access_for_tool("fs__read", &args)
                .expect("outside path should require approval");
            assert_eq!(access.paths.len(), 1);

            let registry = ToolRegistry::default_tools();
            let denied = registry.call("fs__read", args.clone()).await;
            assert!(!denied.ok);
            assert!(
                denied
                    .error
                    .as_ref()
                    .is_some_and(|error| error.message.contains("outside workspace")),
                "{:?}",
                denied.error
            );

            let allowed = registry
                .call_with_context(
                    "fs__read",
                    args,
                    ToolExecutionContext::outside_workspace_granted(),
                )
                .await;
            assert!(allowed.ok, "{:?}", allowed.error);
            assert!(
                allowed
                    .data
                    .as_ref()
                    .and_then(|data| data.get("content"))
                    .and_then(serde_json::Value::as_str)
                    .is_some_and(|content| content.contains("outside"))
            );

            let _ = std::fs::remove_file(outside_path);
        }

        #[tokio::test]
        async fn fs_read_text_output_remains_compatible_and_has_no_images() {
            let path = std::env::temp_dir().join(format!(
                "letcode-read-text-{}.txt",
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .expect("system time")
                    .as_nanos()
            ));
            std::fs::write(&path, "alpha\nbeta\n").expect("write text fixture");

            let output = ToolRegistry::default_tools()
                .call_with_context(
                    "fs__read",
                    json!({
                        "path": path.to_string_lossy(),
                        "offset": 2,
                        "limit": 1,
                    }),
                    ToolExecutionContext::outside_workspace_granted(),
                )
                .await;

            assert!(output.ok, "{:?}", output.error);
            assert!(output.images.is_empty());
            assert_eq!(
                output
                    .data
                    .as_ref()
                    .and_then(|data| data.get("content"))
                    .and_then(Value::as_str),
                Some("beta\n")
            );
            assert_eq!(
                output
                    .data
                    .as_ref()
                    .and_then(|data| data.get("start_line"))
                    .and_then(Value::as_u64),
                Some(2)
            );

            let _ = std::fs::remove_file(path);
        }

        #[tokio::test]
        async fn fs_read_returns_supported_images_as_multimodal_content() {
            let path = std::env::temp_dir().join(format!(
                "letcode-read-image-{}.png",
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .expect("system time")
                    .as_nanos()
            ));
            let png = base64::engine::general_purpose::STANDARD
        .decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAusB9Wl2n3sAAAAASUVORK5CYII=")
        .expect("decode png fixture");
            std::fs::write(&path, &png).expect("write png fixture");

            let output = ToolRegistry::default_tools()
                .call_with_context(
                    "fs__read",
                    json!({
                        "path": path.to_string_lossy(),
                        "offset": 1,
                        "limit": 10,
                    }),
                    {
                        let mut context = ToolExecutionContext::outside_workspace_granted();
                        context.allow_tool_result_images = true;
                        context
                    },
                )
                .await;

            assert!(output.ok, "{:?}", output.error);
            assert_eq!(output.images.len(), 1);
            assert_eq!(output.images[0].mime, "image/png");
            assert!(
                output.images[0]
                    .data_url
                    .starts_with("data:image/png;base64,")
            );
            assert_eq!(
                output
                    .data
                    .as_ref()
                    .and_then(|data| data.get("kind"))
                    .and_then(Value::as_str),
                Some("image")
            );
            assert!(
                output
                    .data
                    .as_ref()
                    .is_some_and(|data| !data.to_string().contains("iVBOR"))
            );

            let _ = std::fs::remove_file(path);
        }

        #[cfg(unix)]
        #[tokio::test]
        async fn writable_leaf_symlinks_require_grants_and_preserve_the_resolved_leaf_identity() {
            use std::os::unix::fs::symlink;

            let unique = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("system time")
                .as_nanos();
            let outside =
                std::env::temp_dir().join(format!("letcode-writable-leaf-target-{unique}"));
            let links =
                PathBuf::from("target").join(format!("letcode-writable-leaf-links-{unique}"));
            std::fs::create_dir_all(&outside).expect("create outside fixture");
            std::fs::create_dir_all(&links).expect("create links fixture");

            for (name, target, expected) in [
                ("existing", outside.join("existing.txt"), "written appended"),
                ("dangling", outside.join("dangling.txt"), "written appended"),
            ] {
                let target = outside
                    .canonicalize()
                    .expect("canonical outside fixture")
                    .join(target.file_name().expect("target leaf"));
                if name == "existing" {
                    std::fs::write(&target, "old").expect("create existing target");
                }
                let link = links.join(name);
                symlink(&target, &link).expect("create leaf symlink");
                let args = json!({"path": link.to_string_lossy(), "content": "written"});
                let prepared = prepare_writable_leaf(args["path"].as_str().expect("path"))
                    .expect("prepare resolved leaf");
                assert_eq!(prepared.destination, target);
                assert_eq!(
                    prepared.permission_resource("fs__write"),
                    permission_resource_for_tool("fs__write", &args).expect("resource")
                );
                assert!(
                    prepared
                        .external_workspace_access()
                        .expect("external access")
                        .preview()
                        .contains(outside.to_string_lossy().as_ref())
                );

                let registry = ToolRegistry::default_tools();
                let denied = registry.call("fs__write", args.clone()).await;
                assert!(!denied.ok);
                if name == "existing" {
                    assert_eq!(
                        std::fs::read_to_string(&target).expect("read unchanged target"),
                        "old"
                    );
                } else {
                    assert!(!target.exists());
                }

                let written = registry
                    .call_with_context(
                        "fs__write",
                        args,
                        ToolExecutionContext::outside_workspace_granted(),
                    )
                    .await;
                assert!(written.ok, "{:?}", written.error);
                let appended = registry
                    .call_with_context(
                        "fs__append",
                        json!({"path": link.to_string_lossy(), "content": " appended"}),
                        ToolExecutionContext::outside_workspace_granted(),
                    )
                    .await;
                assert!(appended.ok, "{:?}", appended.error);
                assert_eq!(
                    std::fs::read_to_string(&target).expect("read target"),
                    expected
                );
                std::fs::remove_file(link).expect("remove link");
            }

            let _ = std::fs::remove_dir_all(links);
            let _ = std::fs::remove_dir_all(outside);
        }
    }
}

#[cfg(unix)]
#[test]
fn writable_leaf_loop_and_missing_parent_fail_before_execution() {
    use std::os::unix::fs::symlink;

    let unique = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .expect("system time")
        .as_nanos();
    std::fs::create_dir_all("target").expect("create target fixture");
    let first = PathBuf::from("target").join(format!("letcode-loop-a-{unique}"));
    let second = PathBuf::from("target").join(format!("letcode-loop-b-{unique}"));
    symlink(second.file_name().expect("second sibling"), &first).expect("create first loop link");
    symlink(first.file_name().expect("first sibling"), &second).expect("create second loop link");
    let loop_error =
        prepare_writable_leaf(&first.to_string_lossy()).expect_err("a -> b -> a must loop");
    assert!(loop_error.to_string().contains("too many symlink levels"));
    let missing_error = prepare_writable_leaf(&format!("target/letcode-missing-{unique}/child"))
        .expect_err("missing ultimate parent must fail independently");
    assert!(
        missing_error
            .to_string()
            .contains("parent directory does not exist")
    );
    let _ = std::fs::remove_file(first);
    let _ = std::fs::remove_file(second);
}

#[cfg(unix)]
#[tokio::test]
async fn secure_writable_leaf_openat_nofollow_rejects_leaf_swapped_after_preparation() {
    use std::os::unix::fs::symlink;

    let unique = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let parent = PathBuf::from("target").join(format!("letcode-nofollow-{unique}"));
    let leaf = parent.join("leaf");
    let target = parent.join("target");
    std::fs::create_dir_all(&parent).unwrap();
    std::fs::write(&leaf, "authorized regular leaf").unwrap();
    std::fs::write(&target, "must stay unchanged").unwrap();
    let prepared = prepare_writable_leaf(leaf.to_str().unwrap()).expect("authorize regular leaf");
    std::fs::remove_file(&leaf).unwrap();
    symlink(&target, &leaf).unwrap();

    let error = secure_write_writable_leaf(&prepared, b"must not write", false)
        .await
        .expect_err("openat O_NOFOLLOW must reject the replacement symlink");
    assert_eq!(
        error.to_string(),
        "writable destination changed after authorization"
    );
    assert_eq!(
        std::fs::read_to_string(&target).unwrap(),
        "must stay unchanged"
    );
    let _ = std::fs::remove_dir_all(parent);
}

mod writes {
    use super::*;

    mod integration_tests {
        use super::*;

        #[cfg(unix)]
        #[tokio::test]
        async fn prepared_writable_leaf_rejects_parent_replacement_before_writing() {
            let unique = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("system time")
                .as_nanos();
            let parent = PathBuf::from("target").join(format!("letcode-parent-replace-{unique}"));
            let retired = PathBuf::from("target").join(format!("letcode-parent-retired-{unique}"));
            let raw = parent.join("leaf.txt");
            std::fs::create_dir_all(&parent).expect("create initial parent");
            let prepared = prepare_writable_leaf(raw.to_str().expect("UTF-8 fixture path"))
                .expect("prepare writable leaf");
            std::fs::rename(&parent, &retired).expect("replace authorized parent");
            std::fs::create_dir(&parent).expect("create replacement parent");
            let mut context = ToolExecutionContext::default();
            context.attach_prepared_writable_leaf(prepared);

            let result = ToolRegistry::default_tools()
                .call_with_context(
                    "fs__write",
                    json!({"path": raw, "content": "must not write"}),
                    context,
                )
                .await;
            assert!(!result.ok);
            assert_eq!(
                result.error.as_ref().map(|error| error.message.as_str()),
                Some("writable destination changed after authorization")
            );
            assert!(!parent.join("leaf.txt").exists());
            assert!(!retired.join("leaf.txt").exists());
            let _ = std::fs::remove_dir_all(parent);
            let _ = std::fs::remove_dir_all(retired);
        }

        #[cfg(windows)]
        #[tokio::test]
        async fn windows_write_and_append_tools_modify_regular_files() {
            let unique = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("system time")
                .as_nanos();
            let path = PathBuf::from("target").join(format!("letcode-windows-write-{unique}.txt"));
            std::fs::create_dir_all("target").expect("create target directory");
            std::fs::write(&path, "old").expect("write existing file");
            let registry = ToolRegistry::default_tools();

            let written = registry
                .call(
                    "fs__write",
                    json!({"path": path.to_string_lossy(), "content": "first"}),
                )
                .await;
            assert!(written.ok, "{:?}", written.error);
            let appended = registry
                .call(
                    "fs__append",
                    json!({"path": path.to_string_lossy(), "content": " second"}),
                )
                .await;
            assert!(appended.ok, "{:?}", appended.error);
            assert_eq!(std::fs::read_to_string(&path).unwrap(), "first second");
            let _ = std::fs::remove_file(path);
        }

        #[cfg(windows)]
        #[tokio::test]
        async fn windows_write_tool_creates_missing_file() {
            let unique = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("system time")
                .as_nanos();
            let path = PathBuf::from("target").join(format!("letcode-windows-new-{unique}.txt"));
            std::fs::create_dir_all("target").expect("create target directory");

            let written = ToolRegistry::default_tools()
                .call(
                    "fs__write",
                    json!({"path": path.to_string_lossy(), "content": "created"}),
                )
                .await;

            assert!(written.ok, "{:?}", written.error);
            assert_eq!(std::fs::read_to_string(&path).unwrap(), "created");
            let _ = std::fs::remove_file(path);
        }
    }
}

#[cfg(windows)]
#[tokio::test]
async fn windows_missing_writable_leaf_rejects_intervening_file_creation() {
    let unique = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .expect("system time")
        .as_nanos();
    let parent = PathBuf::from("target").join(format!("letcode-windows-create-{unique}"));
    let leaf = parent.join("leaf.txt");
    std::fs::create_dir_all(&parent).unwrap();
    let prepared = prepare_writable_leaf(leaf.to_str().unwrap()).unwrap();
    std::fs::write(&leaf, "must stay unchanged").unwrap();

    let error = secure_write_writable_leaf(&prepared, b"must not write", false)
        .await
        .expect_err("intervening file creation must be rejected");
    assert_eq!(
        error.to_string(),
        "writable destination changed after authorization"
    );
    assert_eq!(
        std::fs::read_to_string(&leaf).unwrap(),
        "must stay unchanged"
    );
    drop(prepared);
    let _ = std::fs::remove_dir_all(parent);
}

#[cfg(windows)]
#[tokio::test]
async fn windows_existing_writable_leaf_prevents_path_replacement() {
    let unique = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .expect("system time")
        .as_nanos();
    let parent = PathBuf::from("target").join(format!("letcode-windows-anchor-{unique}"));
    let leaf = parent.join("leaf.txt");
    let replacement = parent.join("replacement.txt");
    std::fs::create_dir_all(&parent).unwrap();
    std::fs::write(&leaf, "original").unwrap();
    std::fs::write(&replacement, "replacement").unwrap();
    let prepared = prepare_writable_leaf(leaf.to_str().unwrap()).unwrap();

    assert!(std::fs::remove_file(&leaf).is_err());
    assert!(std::fs::rename(&replacement, &leaf).is_err());
    secure_write_writable_leaf(&prepared, b"updated", false)
        .await
        .expect("authorized file remains writable");
    assert_eq!(std::fs::read_to_string(&leaf).unwrap(), "updated");
    assert_eq!(
        std::fs::read_to_string(&replacement).unwrap(),
        "replacement"
    );
    drop(prepared);
    let _ = std::fs::remove_dir_all(parent);
}

#[cfg(windows)]
#[test]
fn windows_writable_leaf_rejects_existing_symlink() {
    use std::os::windows::fs::symlink_file;

    let unique = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .expect("system time")
        .as_nanos();
    let parent = PathBuf::from("target").join(format!("letcode-windows-link-{unique}"));
    let leaf = parent.join("leaf.txt");
    let target = parent.join("target.txt");
    std::fs::create_dir_all(&parent).unwrap();
    std::fs::write(&target, "unchanged").unwrap();
    symlink_file(&target, &leaf).expect("create fixture symlink");

    let error = prepare_writable_leaf(leaf.to_str().unwrap())
        .expect_err("link destination must be rejected");
    assert!(error.to_string().contains("cannot be a link"));
    assert_eq!(std::fs::read_to_string(&target).unwrap(), "unchanged");
    let _ = std::fs::remove_dir_all(parent);
}

mod integration_tests {
    use super::*;

    #[tokio::test]
    async fn unknown_tool_scope_rejection_precedes_lookup() {
        let name = "missing__tool";
        let read_only = ToolRegistry::default_tools().scoped(ToolScope::ReadOnlyExplorer);
        let rejected = read_only.call(name, json!({})).await;
        assert_eq!(
            rejected.error.as_ref().expect("scope error").message,
            "tool 'missing__tool' is not allowed in read_only_explorer scope"
        );

        let unknown = ToolRegistry::default_tools()
            .scoped(ToolScope::FullAccess)
            .call(name, json!({}))
            .await;
        assert_eq!(
            unknown.error.as_ref().expect("unknown tool error").message,
            "unknown tool: missing__tool"
        );
    }

    #[tokio::test]
    async fn workflow_todos_rejects_duplicate_ids() {
        let output = call_workflow_todos(json!([
            {"id": "todo-1", "content": "first", "status": "pending"},
            {"id": "todo-1", "content": "second", "status": "completed"}
        ]))
        .await;

        assert!(!output.ok);
        assert!(
            output
                .error
                .as_ref()
                .expect("todo error")
                .message
                .contains("duplicate id 'todo-1'")
        );
    }

    #[tokio::test]
    async fn workflow_todos_allows_multiple_in_progress_items() {
        let output = call_workflow_todos(json!([
            {"id": "todo-1", "content": "first", "status": "in_progress"},
            {"id": "todo-2", "content": "second", "status": "in_progress"}
        ]))
        .await;

        assert!(output.ok);
    }

    #[tokio::test]
    async fn read_only_explorer_scope_filters_specs_and_rejects_calls() {
        let registry = SkillRegistry::from_entries(vec![SkillEntry {
            name: "rust-audit".into(),
            description: "Inspect Rust code".into(),
            body: "# Body".into(),
            content: "---\nname: rust-audit\ndescription: Inspect Rust code\n---\n# Body\n".into(),
            location: "config".into(),
            path: PathBuf::from("/tmp/rust-audit/SKILL.md"),
            base_dir: PathBuf::from("/tmp/rust-audit"),
        }])
        .expect("skill registry");
        let mut tools = ToolRegistry::default_tools();
        tools
            .try_register(SkillTool::new(Arc::new(registry)))
            .expect("register skill tool");
        let tools = tools.scoped(ToolScope::ReadOnlyExplorer);
        let specs = tools.specs();

        assert!(specs.iter().any(|spec| spec.name == "fs__read"));
        assert!(specs.iter().any(|spec| spec.name == "skill"));
        assert!(!specs.iter().any(|spec| spec.name == "agent__explore"));
        assert!(!specs.iter().any(|spec| spec.name == "agent__fixer"));
        assert!(!specs.iter().any(|spec| spec.name == "fs__write"));
        assert!(!specs.iter().any(|spec| spec.name == "workflow__todos"));
        assert!(!specs.iter().any(|spec| spec.name == "memory__recall"));
        assert!(!specs.iter().any(|spec| spec.name == "context__checkpoint"));
        assert!(!specs.iter().any(|spec| spec.name == "context__return"));

        let output = tools
            .call("fs__write", json!({"path": "src/lib.rs", "content": "x"}))
            .await;
        assert!(!output.ok);
        assert_eq!(
            output.error.as_ref().expect("scope error").message,
            "tool 'fs__write' is not allowed in read_only_explorer scope"
        );

        let output = tools
            .call("agent__explore", json!({"task": "inspect"}))
            .await;
        assert!(!output.ok);
        assert_eq!(
            output.error.as_ref().expect("scope error").message,
            "tool 'agent__explore' is not allowed in read_only_explorer scope"
        );

        let output = tools
            .call("agent__fixer", json!({"task": "implement"}))
            .await;
        assert!(!output.ok);
        assert_eq!(
            output.error.as_ref().expect("scope error").message,
            "tool 'agent__fixer' is not allowed in read_only_explorer scope"
        );
    }

    #[cfg(windows)]
    #[tokio::test]
    async fn windows_apply_patch_updates_existing_file() {
        let unique = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("system time")
            .as_nanos();
        let path = PathBuf::from("target").join(format!("letcode-windows-patch-{unique}.txt"));
        std::fs::create_dir_all("target").unwrap();
        std::fs::write(&path, "before\n").unwrap();

        let output = ToolRegistry::default_tools()
            .call(
                "edit__apply_patch",
                json!({"edits": [{
                    "path": path.to_string_lossy(),
                    "find": "before",
                    "replace": "after",
                    "replace_all": false
                }]}),
            )
            .await;

        assert!(output.ok, "{:?}", output.error);
        assert_eq!(std::fs::read_to_string(&path).unwrap(), "after\n");
        let _ = std::fs::remove_file(path);
    }

    #[tokio::test]
    async fn apply_patch_validation_failure_leaves_all_files_unchanged() {
        let unique = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("system time")
            .as_nanos();
        let fixture =
            PathBuf::from("target").join(format!("letcode-apply-patch-validation-{unique}"));
        let first = fixture.join("first.txt");
        let second = fixture.join("second.txt");
        std::fs::create_dir_all(&fixture).expect("create fixture directory");
        std::fs::write(&first, "first original").expect("write first fixture");
        std::fs::write(&second, "second original").expect("write second fixture");

        let result = ToolRegistry::default_tools()
            .call(
                "edit__apply_patch",
                json!({
                    "edits": [
                        {
                            "path": first,
                            "find": "first original",
                            "replace": "first changed",
                            "replace_all": false
                        },
                        {
                            "path": second,
                            "find": "missing text",
                            "replace": "second changed",
                            "replace_all": false
                        }
                    ]
                }),
            )
            .await;

        assert!(!result.ok, "{result:?}");
        assert_eq!(std::fs::read_to_string(&first).unwrap(), "first original");
        assert_eq!(std::fs::read_to_string(&second).unwrap(), "second original");
        let _ = std::fs::remove_dir_all(fixture);
    }

    #[cfg(unix)]
    #[test]
    fn apply_patch_preparation_rejects_more_than_64_canonical_targets_before_anchors() {
        let fixture =
            std::env::temp_dir().join(format!("letcode-apply-patch-cap-{}", std::process::id()));
        std::fs::create_dir_all(&fixture).expect("fixture directory");
        let edits: Vec<_> = (0..65)
            .map(|index| {
                let path = fixture.join(format!("{index}.txt"));
                std::fs::write(&path, "old").expect("fixture file");
                json!({"path": path, "find": "old", "replace": "new", "replace_all": false})
            })
            .collect();
        let error =
            prepare_apply_patch_targets(&json!({"edits": edits})).expect_err("cap must reject");
        assert_eq!(
            error.to_string(),
            "apply patch accepts at most 64 unique target files"
        );
        let _ = std::fs::remove_dir_all(fixture);
    }

    #[cfg(unix)]
    #[test]
    fn apply_patch_preparation_rejects_batch_hardlink_aliases() {
        let fixture = std::env::temp_dir().join(format!(
            "letcode-apply-patch-hardlink-{}",
            std::process::id()
        ));
        std::fs::create_dir_all(&fixture).expect("fixture directory");
        let first = fixture.join("first.txt");
        let second = fixture.join("second.txt");
        std::fs::write(&first, "old").expect("fixture file");
        std::fs::hard_link(&first, &second).expect("hardlink");
        let error = prepare_apply_patch_targets(&json!({"edits": [
            {"path": first, "find": "old", "replace": "new", "replace_all": false},
            {"path": second, "find": "old", "replace": "new", "replace_all": false}
        ]}))
        .expect_err("hardlink aliases must reject");
        assert_eq!(
            error.to_string(),
            "apply patch targets alias the same existing file"
        );
        let _ = std::fs::remove_dir_all(fixture);
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn apply_patch_bound_execution_rejects_replaced_leaf_without_writing() {
        let fixture = std::env::temp_dir().join(format!(
            "letcode-apply-patch-replacement-{}",
            std::process::id()
        ));
        std::fs::create_dir_all(&fixture).expect("fixture directory");
        let file = fixture.join("file.txt");
        let replacement = fixture.join("replacement.txt");
        std::fs::write(&file, "old").expect("fixture file");
        let args = json!({"edits": [{"path": file, "find": "old", "replace": "new", "replace_all": false}]});
        let prepared = prepare_apply_patch_targets(&args).expect("prepare binding");
        std::fs::write(&replacement, "replacement").expect("replacement file");
        std::fs::rename(&replacement, &file).expect("replace leaf");
        let mut context = ToolExecutionContext::outside_workspace_granted();
        context.attach_prepared_apply_patch(prepared);
        let result = ToolRegistry::default_tools()
            .call_with_context("edit__apply_patch", args, context)
            .await;
        assert_eq!(
            result.error.expect("error").message,
            "apply patch target changed after authorization"
        );
        assert_eq!(std::fs::read_to_string(&file).unwrap(), "replacement");
        let _ = std::fs::remove_dir_all(fixture);
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn apply_patch_attached_binding_rejects_unresolvable_rebind_without_preparing_it() {
        let fixture = std::env::temp_dir().join(format!(
            "letcode-apply-patch-lazy-binding-{}",
            std::process::id()
        ));
        std::fs::create_dir_all(&fixture).expect("fixture directory");
        let file = fixture.join("file.txt");
        std::fs::write(&file, "old").expect("fixture file");
        let authorized = json!({"edits": [{"path": file, "find": "old", "replace": "new", "replace_all": false}]});
        let prepared = prepare_apply_patch_targets(&authorized).expect("prepare binding");
        let mut context = ToolExecutionContext::outside_workspace_granted();
        context.attach_prepared_apply_patch(prepared);
        let result = ToolRegistry::default_tools()
            .call_with_context(
                "edit__apply_patch",
                json!({"edits": [{
                    "path": fixture.join("missing").join("target.txt"),
                    "find": "old",
                    "replace": "new",
                    "replace_all": false
                }]}),
                context,
            )
            .await;
        assert_eq!(
            result.error.expect("error").message,
            "apply patch target changed after authorization"
        );
        assert_eq!(std::fs::read_to_string(&file).unwrap(), "old");
        let _ = std::fs::remove_dir_all(fixture);
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn apply_patch_bound_execution_rejects_fifo_replacement_without_blocking_or_writing() {
        use std::ffi::CString;
        use std::os::unix::ffi::OsStrExt;
        use std::os::unix::fs::FileTypeExt;

        let fixture = std::env::temp_dir().join(format!(
            "letcode-apply-patch-fifo-replacement-{}",
            std::process::id()
        ));
        std::fs::create_dir_all(&fixture).expect("fixture directory");
        let file = fixture.join("file.txt");
        std::fs::write(&file, "old").expect("fixture file");
        let args = json!({"edits": [{"path": file, "find": "old", "replace": "new", "replace_all": false}]});
        let prepared = prepare_apply_patch_targets(&args).expect("prepare binding");
        std::fs::remove_file(&file).expect("remove approved leaf");
        let fifo = CString::new(file.as_os_str().as_bytes()).expect("fifo path");
        assert_eq!(
            unsafe { libc::mkfifo(fifo.as_ptr(), 0o600) },
            0,
            "create fifo"
        );
        let mut context = ToolExecutionContext::outside_workspace_granted();
        context.attach_prepared_apply_patch(prepared);
        let result = ToolRegistry::default_tools()
            .call_with_context("edit__apply_patch", args, context)
            .await;
        assert_eq!(
            result.error.expect("error").message,
            "apply patch target changed after authorization"
        );
        assert!(
            std::fs::metadata(&file)
                .expect("fifo metadata")
                .file_type()
                .is_fifo()
        );
        let _ = std::fs::remove_dir_all(fixture);
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn apply_patch_direct_context_requires_external_grant_and_supports_mixed_targets() {
        let unique = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("system time")
            .as_nanos();
        let internal =
            PathBuf::from("target").join(format!("letcode-apply-patch-internal-{unique}"));
        let external = std::env::temp_dir().join(format!("letcode-apply-patch-external-{unique}"));
        std::fs::write(&internal, "old").expect("internal fixture");
        std::fs::write(&external, "old").expect("external fixture");
        let args = json!({"edits": [
            {"path": internal, "find": "old", "replace": "internal", "replace_all": false},
            {"path": external, "find": "old", "replace": "external", "replace_all": false}
        ]});

        let denied = ToolRegistry::default_tools()
            .call("edit__apply_patch", args.clone())
            .await;
        assert!(!denied.ok);
        assert_eq!(std::fs::read_to_string(&internal).unwrap(), "old");
        assert_eq!(std::fs::read_to_string(&external).unwrap(), "old");

        let granted = ToolRegistry::default_tools()
            .call_with_context(
                "edit__apply_patch",
                args,
                ToolExecutionContext::outside_workspace_granted(),
            )
            .await;
        assert!(granted.ok, "{granted:?}");
        assert_eq!(std::fs::read_to_string(&internal).unwrap(), "internal");
        assert_eq!(std::fs::read_to_string(&external).unwrap(), "external");
        let _ = std::fs::remove_file(internal);
        let _ = std::fs::remove_file(external);
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn apply_patch_staging_and_precommit_mutations_leave_every_target_unchanged_by_patch() {
        for point in [
            ApplyPatchWorkerPoint::BeforeStagingValidation,
            ApplyPatchWorkerPoint::BeforeBatchPrecommit,
        ] {
            let fixture = std::env::temp_dir().join(format!(
                "letcode-apply-patch-barrier-{}-{:?}",
                std::process::id(),
                point
            ));
            std::fs::create_dir_all(&fixture).unwrap();
            let first = fixture.join("first.txt");
            let second = fixture.join("second.txt");
            std::fs::write(&first, "first old").unwrap();
            std::fs::write(&second, "second old").unwrap();
            let args = json!({"edits": [
                {"path": first, "find": "old", "replace": "new", "replace_all": false},
                {"path": second, "find": "old", "replace": "new", "replace_all": false}
            ]});
            let prepared = prepare_apply_patch_targets(&args).unwrap();
            let mutate = second.clone();
            prepared.set_worker_hook(point, move || {
                std::fs::write(mutate, "second mutated").unwrap();
            });
            let mut context = ToolExecutionContext::outside_workspace_granted();
            context.attach_prepared_apply_patch(prepared);
            let result = ToolRegistry::default_tools()
                .call_with_context("edit__apply_patch", args, context)
                .await;
            assert!(!result.ok);
            assert_eq!(std::fs::read_to_string(&first).unwrap(), "first old");
            assert_eq!(std::fs::read_to_string(&second).unwrap(), "second mutated");
            let _ = std::fs::remove_dir_all(fixture);
        }
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn apply_patch_commit_rechecks_nonregular_replacements_and_preserves_prior_commit_on_io_error()
     {
        use std::ffi::CString;
        use std::os::unix::{ffi::OsStrExt, fs::PermissionsExt};

        let fixture =
            std::env::temp_dir().join(format!("letcode-apply-patch-commit-{}", std::process::id()));
        std::fs::create_dir_all(&fixture).unwrap();
        let fifo = fixture.join("fifo.txt");
        std::fs::write(&fifo, "old").unwrap();
        let fifo_args = json!({"edits": [{"path": fifo, "find": "old", "replace": "new", "replace_all": false}]});
        let prepared = prepare_apply_patch_targets(&fifo_args).unwrap();
        let fifo_for_hook = fifo.clone();
        prepared.set_worker_hook(
            ApplyPatchWorkerPoint::BeforeCommitOpen {
                path: fifo.canonicalize().unwrap(),
                ordinal: 0,
            },
            move || {
                std::fs::remove_file(&fifo_for_hook).unwrap();
                let path = CString::new(fifo_for_hook.as_os_str().as_bytes()).unwrap();
                assert_eq!(unsafe { libc::mkfifo(path.as_ptr(), 0o600) }, 0);
            },
        );
        let mut context = ToolExecutionContext::outside_workspace_granted();
        context.attach_prepared_apply_patch(prepared);
        let result = ToolRegistry::default_tools()
            .call_with_context("edit__apply_patch", fifo_args, context)
            .await;
        assert_eq!(
            result.error.unwrap().message,
            "apply patch target changed after authorization"
        );

        let directory = fixture.join("directory.txt");
        std::fs::write(&directory, "old").unwrap();
        let directory_args = json!({"edits": [{"path": directory, "find": "old", "replace": "new", "replace_all": false}]});
        let prepared = prepare_apply_patch_targets(&directory_args).unwrap();
        let directory_for_hook = directory.clone();
        prepared.set_worker_hook(
            ApplyPatchWorkerPoint::BeforeCommitOpen {
                path: directory.canonicalize().unwrap(),
                ordinal: 0,
            },
            move || {
                std::fs::remove_file(&directory_for_hook).unwrap();
                std::fs::create_dir(&directory_for_hook).unwrap();
            },
        );
        let mut context = ToolExecutionContext::outside_workspace_granted();
        context.attach_prepared_apply_patch(prepared);
        let result = ToolRegistry::default_tools()
            .call_with_context("edit__apply_patch", directory_args, context)
            .await;
        assert_eq!(
            result.error.unwrap().message,
            "apply patch target changed after authorization"
        );

        let first = fixture.join("a-first.txt");
        let later = fixture.join("b-later.txt");
        std::fs::write(&first, "old").unwrap();
        std::fs::write(&later, "old").unwrap();
        let args = json!({"edits": [
            {"path": first, "find": "old", "replace": "new", "replace_all": false},
            {"path": later, "find": "old", "replace": "new", "replace_all": false}
        ]});
        let prepared = prepare_apply_patch_targets(&args).unwrap();
        let later_for_hook = later.clone();
        prepared.set_worker_hook(
            ApplyPatchWorkerPoint::BeforeCommitOpen {
                path: later.canonicalize().unwrap(),
                ordinal: 1,
            },
            move || {
                std::fs::set_permissions(&later_for_hook, std::fs::Permissions::from_mode(0o000))
                    .unwrap();
            },
        );
        let mut context = ToolExecutionContext::outside_workspace_granted();
        context.attach_prepared_apply_patch(prepared);
        let result = ToolRegistry::default_tools()
            .call_with_context("edit__apply_patch", args, context)
            .await;
        let error = result.error.expect("later I/O error").message;
        assert!(error.contains(later.to_string_lossy().as_ref()), "{error}");
        assert_ne!(error, "apply patch target changed after authorization");
        assert_eq!(std::fs::read_to_string(&first).unwrap(), "new");
        std::fs::set_permissions(&later, std::fs::Permissions::from_mode(0o600)).unwrap();
        assert_eq!(std::fs::read_to_string(&later).unwrap(), "old");
        let _ = std::fs::remove_dir_all(fixture);
    }
}
