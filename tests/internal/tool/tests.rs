use super::{NormalizedSubagentInput, SubagentPathScope, normalize_subagent_input};
use serde_json::json;

#[test]
fn subagent_path_scope_authorizes_missing_roots() {
    let empty = NormalizedSubagentInput {
        objective: "x".into(),
        success_criteria: Vec::new(),
        allowed_paths: Vec::new(),
        forbidden_paths: Vec::new(),
        owned_paths: Vec::new(),
        timeout_secs: None,
        max_tool_calls: None,
        model: None,
        target_child_session_id: None,
        background: false,
    };
    assert!(
        SubagentPathScope::from_input(&empty)
            .expect("empty scope")
            .is_none()
    );

    let dir = tempfile::tempdir().unwrap();
    let root = dir.path().canonicalize().unwrap();
    let readable = root.join("readable");
    let leaf = root.join("new.txt");
    let tree = root.join("new-tree");
    let forbidden = tree.join("private");
    let input = NormalizedSubagentInput {
        allowed_paths: vec![readable.to_string_lossy().into_owned()],
        owned_paths: vec![
            leaf.to_string_lossy().into_owned(),
            tree.to_string_lossy().into_owned(),
        ],
        forbidden_paths: vec![forbidden.to_string_lossy().into_owned()],
        ..empty
    };
    let scope = SubagentPathScope::from_input(&input).unwrap().unwrap();
    assert!(scope.permits_read([readable.join("child").as_path()]));
    assert!(!scope.permits_write([readable.join("child").as_path()]));
    for (path, permitted) in [(&leaf, true), (&root.join("neighbor.txt"), false)] {
        let path = path.to_str().unwrap();
        let prepared = super::prepare_writable_leaf(path).unwrap();
        assert_eq!(
            super::delegation_scope_denial(
                &scope,
                "fs__write",
                &json!({"path": path}),
                Some(&prepared),
                None,
            )
            .is_none(),
            permitted
        );
    }
    for (path, permitted) in [
        (tree.join("child"), true),
        (forbidden.join("child"), false),
        (root.join("neighbor-tree"), false),
    ] {
        assert_eq!(
            super::delegation_scope_denial(
                &scope,
                "fs__mkdir",
                &json!({"path": path}),
                None,
                None,
            )
            .is_none(),
            permitted
        );
    }
    assert!(!scope.permits_read([forbidden.as_path()]));
    assert!(!scope.permits_write([forbidden.as_path()]));
    assert!(!leaf.exists());
    assert!(!tree.exists());
}

#[test]
fn subagent_path_scope_narrows_missing_roots_within_parent() {
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path().canonicalize().unwrap();
    let owned = root.join("owned");
    let readable = root.join("readable");
    let forbidden = owned.join("child/private");
    let input = normalize_subagent_input(
        "agent__fixer",
        &json!({
            "objective": "edit",
            "allowed_paths": [readable],
            "owned_paths": [owned],
            "forbidden_paths": [forbidden],
        }),
    )
    .unwrap();
    let parent = SubagentPathScope::from_input(&input).unwrap().unwrap();
    let child_input = normalize_subagent_input(
        "agent__fixer",
        &json!({"objective": "edit", "owned_paths": [owned.join("child")]}),
    )
    .unwrap();
    let child = SubagentPathScope::from_input(&child_input)
        .unwrap()
        .unwrap()
        .narrowed_from(&parent)
        .unwrap();
    assert!(child.permits_write([owned.join("child/new.txt").as_path()]));
    assert!(!child.permits_write([owned.join("neighbor.txt").as_path()]));
    assert!(!child.permits_read([forbidden.as_path()]));
    for args in [
        json!({"allowed_paths": [root.join("outside")]}),
        json!({"owned_paths": [root.join("outside")]}),
        json!({"owned_paths": [readable.join("new.txt")]}),
        json!({"owned_paths": [forbidden.join("new.txt")]}),
    ] {
        let mut args = args;
        args["objective"] = json!("inspect");
        let input = normalize_subagent_input("agent__explore", &args).unwrap();
        let error = SubagentPathScope::from_input(&input)
            .unwrap()
            .unwrap()
            .narrowed_from(&parent)
            .unwrap_err();
        assert!(error.to_string().contains("exceeds parent scope"));
    }
}

#[test]
fn subagent_path_scope_uses_platform_path_identity() {
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path().canonicalize().unwrap();
    let file = tempfile::NamedTempFile::new_in(dir.path()).unwrap();
    let canonical_file = file.path().canonicalize().unwrap();
    for (path, windows_target) in [
        (dir.path().join("missing/../outside"), root.join("outside")),
        (file.path().join("child"), canonical_file.join("child")),
        (
            std::path::PathBuf::from(format!("{}/", file.path().display())),
            canonical_file.clone(),
        ),
        (file.path().join("."), canonical_file),
        (
            dir.path().join("missing/./../outside/."),
            root.join("outside"),
        ),
    ] {
        let canonical =
            std::fs::canonicalize(&path).map_err(|error| (error.kind(), error.raw_os_error()));
        let metadata = std::fs::symlink_metadata(&path)
            .map(|metadata| metadata.is_dir())
            .map_err(|error| (error.kind(), error.raw_os_error()));
        let expected = cfg!(windows).then_some(windows_target);
        for field in ["allowed_paths", "owned_paths", "forbidden_paths"] {
            let mut args = json!({"objective": "inspect"});
            if field == "forbidden_paths" {
                args["allowed_paths"] = json!([root]);
                args["owned_paths"] = json!([root]);
            }
            args[field] = json!([path]);
            let input = normalize_subagent_input("agent__explore", &args).unwrap();
            let result = SubagentPathScope::from_input(&input);
            if let Some(target) = &expected {
                let scope = result
                    .unwrap_or_else(|error| {
                        panic!(
                            "{field} {path:?}: canonicalize={canonical:?}, metadata={metadata:?}, scope={error:#}"
                        )
                    })
                    .unwrap();
                for target in [target.clone(), target.join("child")] {
                    assert_eq!(
                        scope.permits_read([target.as_path()]),
                        field != "forbidden_paths"
                    );
                    assert_eq!(
                        scope.permits_write([target.as_path()]),
                        field == "owned_paths"
                    );
                }
                let neighbor = root.join("neighbor");
                assert_eq!(
                    scope.permits_read([neighbor.as_path()]),
                    field == "forbidden_paths"
                );
                assert_eq!(
                    scope.permits_write([neighbor.as_path()]),
                    field == "forbidden_paths"
                );
                match field {
                    "allowed_paths" => {
                        assert_eq!(scope.read_roots(), std::slice::from_ref(target));
                    }
                    "owned_paths" => {
                        assert_eq!(scope.owned_roots(), std::slice::from_ref(target));
                    }
                    _ => {}
                }
            } else {
                assert!(
                    result.is_err(),
                    "{field} {path:?}: canonicalize={canonical:?}, metadata={metadata:?}, scope={result:?}"
                );
                assert!(
                    result
                        .unwrap_err()
                        .to_string()
                        .starts_with("delegated path scope root cannot be resolved:")
                );
            }
        }
    }
}

#[cfg(unix)]
#[test]
fn subagent_path_scope_resolves_symlink_targets_with_missing_components() {
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path().canonicalize().unwrap();
    let real = root.join("real/inner");
    std::fs::create_dir_all(&real).unwrap();
    let alias = root.join("alias");
    std::os::unix::fs::symlink(&real, &alias).unwrap();
    let dangling = root.join("dangling");
    let target = root.join("new.txt");
    std::os::unix::fs::symlink("new.txt", &dangling).unwrap();
    let absolute = root.join("absolute");
    std::os::unix::fs::symlink(&target, &absolute).unwrap();
    let forbidden_input = normalize_subagent_input(
        "agent__fixer",
        &json!({"objective": "edit", "owned_paths": [root], "forbidden_paths": [dangling]}),
    )
    .unwrap();
    let forbidden_scope = SubagentPathScope::from_input(&forbidden_input)
        .unwrap()
        .unwrap();
    let owned_input = normalize_subagent_input(
        "agent__fixer",
        &json!({"objective": "edit", "owned_paths": [dangling]}),
    )
    .unwrap();
    let owned_scope = SubagentPathScope::from_input(&owned_input)
        .unwrap()
        .unwrap();
    for path in [&dangling, &absolute, &target] {
        let path = path.to_str().unwrap();
        let prepared = super::prepare_writable_leaf(path).unwrap();
        assert_eq!(prepared.destination(), target);
        assert!(owned_scope.permits_write([prepared.destination()]));
        let denial = super::delegation_scope_denial(
            &forbidden_scope,
            "fs__write",
            &json!({"path": path}),
            Some(&prepared),
            None,
        )
        .expect("symlink target is forbidden");
        assert!(denial.contains("is forbidden"));
    }
    let directory_alias = root.join("directory-alias");
    let directory_target = root.join("new-dir");
    std::os::unix::fs::symlink("new-dir", &directory_alias).unwrap();
    for (index, link_target) in ["directory-alias/", "directory-alias/."]
        .into_iter()
        .enumerate()
    {
        let link = root.join(format!("directory-chain-{index}"));
        std::os::unix::fs::symlink(link_target, &link).unwrap();
    }
    for (relative, expected) in [
        ("directory-alias/", directory_target.clone()),
        ("directory-alias/.", directory_target.clone()),
        ("./directory-alias//./", directory_target.clone()),
        (
            "directory-alias//./child//.",
            directory_target.join("child"),
        ),
        ("directory-chain-0", directory_target.clone()),
        ("directory-chain-1", directory_target.clone()),
    ] {
        let path = format!("{}/{relative}", root.display());
        let input = normalize_subagent_input(
            "agent__fixer",
            &json!({"objective": "edit", "owned_paths": [root], "forbidden_paths": [path]}),
        )
        .unwrap();
        let scope = SubagentPathScope::from_input(&input).unwrap().unwrap();
        let denial = super::delegation_scope_denial(
            &scope,
            "fs__mkdir",
            &json!({"path": expected}),
            None,
            None,
        )
        .expect("symlink directory target is forbidden");
        assert!(denial.contains("is forbidden"));
        let input = normalize_subagent_input(
            "agent__fixer",
            &json!({"objective": "edit", "owned_paths": [path]}),
        )
        .unwrap();
        let scope = SubagentPathScope::from_input(&input).unwrap().unwrap();
        assert_eq!(scope.owned_roots(), std::slice::from_ref(&expected));
    }
    let nested = root.join("nested-alias");
    std::os::unix::fs::symlink("missing/parent", &nested).unwrap();
    for (path, expected) in [
        (alias.clone(), real.clone()),
        (alias.join("new"), real.join("new")),
        (alias.join("../new"), root.join("real/new")),
        (dangling, target.clone()),
        (absolute, target),
        (nested.join("child"), root.join("missing/parent/child")),
    ] {
        let input = normalize_subagent_input(
            "agent__fixer",
            &json!({"objective": "edit", "owned_paths": [path]}),
        )
        .unwrap();
        let scope = SubagentPathScope::from_input(&input).unwrap().unwrap();
        assert_eq!(scope.owned_roots(), std::slice::from_ref(&expected));
        assert!(scope.permits_write([expected.join("child").as_path()]));
        assert!(!scope.permits_write([root.join("new/child").as_path()]));
    }
    let cycle = root.join("cycle");
    std::os::unix::fs::symlink("cycle", &cycle).unwrap();
    let input = normalize_subagent_input(
        "agent__fixer",
        &json!({"objective": "edit", "owned_paths": [cycle]}),
    )
    .unwrap();
    let error = SubagentPathScope::from_input(&input).unwrap_err();
    assert!(format!("{error:#}").contains("too many symlink levels"));
}

#[test]
fn normalize_subagent_input_rejects_missing_or_blank_objective() {
    let missing = normalize_subagent_input("agent__explore", &json!({}))
        .expect_err("missing objective should fail");
    assert!(
        missing
            .to_string()
            .contains("requires a non-empty 'task' or 'objective'")
    );

    let blank =
        normalize_subagent_input("agent__fixer", &json!({"task": "   ", "objective": null}))
            .expect_err("blank task should fail");
    assert!(
        blank
            .to_string()
            .contains("field 'task' must not be empty or whitespace")
    );
}

#[test]
fn writable_subagents_require_owned_paths_for_file_locking() {
    for tool in ["agent__fixer", "agent__general"] {
        for args in [
            json!({"task": "implement"}),
            json!({"task": "implement", "owned_paths": null}),
            json!({"task": "implement", "owned_paths": []}),
        ] {
            let missing = normalize_subagent_input(tool, &args)
                .expect_err("writable subagent without locks must fail");
            assert!(
                missing
                    .to_string()
                    .contains(&format!("{tool} requires non-empty owned_paths"))
            );
        }
        let input =
            normalize_subagent_input(tool, &json!({"task": "implement", "owned_paths": ["src"]}))
                .expect("writable subagent with owned paths normalizes");
        assert_eq!(input.owned_paths, vec!["src"]);
    }
}

#[test]
fn normalize_subagent_input_accepts_model_and_rejects_takeover_conflict() {
    let input = normalize_subagent_input(
        "agent__explore",
        &json!({"objective": "inspect", "model": "expert/special"}),
    )
    .expect("model override normalizes");
    assert_eq!(input.model.as_deref(), Some("expert/special"));
    assert!(!input.background);

    let background = normalize_subagent_input(
        "agent__explore",
        &json!({"objective": "inspect", "background": true}),
    )
    .expect("background flag normalizes");
    assert!(background.background);

    let invalid_background = normalize_subagent_input(
        "agent__explore",
        &json!({"objective": "inspect", "background": "yes"}),
    )
    .expect_err("non-boolean background should fail");
    assert!(
        invalid_background
            .to_string()
            .contains("field 'background' must be a boolean or null")
    );

    let error = normalize_subagent_input(
        "agent__explore",
        &json!({
            "objective": "inspect",
            "model": "expert/special",
            "target_child_session_id": "child-1"
        }),
    )
    .expect_err("override and takeover conflict");
    assert!(
        error
            .to_string()
            .contains("field 'model' cannot be combined with 'target_child_session_id'")
    );
}

#[test]
fn normalized_subagent_input_enforces_write_scope_paths() {
    for tool in ["agent__fixer", "agent__general"] {
        let input = normalize_subagent_input(
            tool,
            &json!({
                "objective": "fix",
                "allowed_paths": ["src"],
                "owned_paths": ["src/fixes"],
                "forbidden_paths": ["src/secrets"]
            }),
        )
        .expect("scope should normalize");

        assert!(input.has_write_scope());
        assert!(input.permits_write_path("src/fixes/mod.rs"));
        assert!(input.permits_write_path("src/lib.rs"));
        assert!(!input.permits_write_path("src/secrets/token.rs"));
        assert!(!input.permits_write_path("tests/outside.rs"));
    }
}
