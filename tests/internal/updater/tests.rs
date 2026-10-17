use super::*;

#[test]
fn release_targets_match_the_packaging_workflow() {
    for (os, arch, expected_asset) in [
        (
            "linux",
            "x86_64",
            "letcode-1.2.3-x86_64-unknown-linux-gnu.tar.gz",
        ),
        (
            "linux",
            "aarch64",
            "letcode-1.2.3-aarch64-unknown-linux-gnu.tar.gz",
        ),
        (
            "macos",
            "x86_64",
            "letcode-1.2.3-x86_64-apple-darwin.tar.gz",
        ),
        (
            "macos",
            "aarch64",
            "letcode-1.2.3-aarch64-apple-darwin.tar.gz",
        ),
        (
            "windows",
            "x86_64",
            "letcode-1.2.3-x86_64-pc-windows-msvc.zip",
        ),
        (
            "windows",
            "aarch64",
            "letcode-1.2.3-aarch64-pc-windows-msvc.zip",
        ),
    ] {
        let target = target_for(os, arch).expect("release target should be supported");
        assert_eq!(asset_name("1.2.3", target), expected_asset);
    }
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
    let error = target_for("freebsd", "x86_64").expect_err("FreeBSD is not released yet");
    assert_eq!(
        error.to_string(),
        "self-update is not supported on freebsd/x86_64"
    );
}
