use super::*;
#[test]
fn toggle_updates_session_fast_mode() {
    let mode = FastMode::load("letcode.toml", false);
    assert!(!mode.enabled());
    assert_eq!(
        mode.toggle("gpt-5.5").expect("enable fast mode"),
        FastModeToggle::Enabled
    );
    assert!(mode.enabled());
}

#[test]
fn unsupported_model_disables_session_fast_mode() {
    let mode = FastMode::load("letcode.toml", true);
    assert!(mode.auto_disable_for_model("claude-4").expect("disable"));
    assert!(!mode.enabled());
}

#[test]
fn ultrafast_is_limited_to_supported_models() {
    let mode = FastMode::load("letcode.toml", false);
    assert!(
        !mode
            .set("gpt-5.5", FastModeTier::Ultrafast)
            .expect("set tier")
    );
    assert!(
        mode.set("gpt-6-astra", FastModeTier::Ultrafast)
            .expect("set tier")
    );
    assert_eq!(mode.tier(), Some(FastModeTier::Ultrafast));
    assert!(
        mode.auto_disable_for_model("gpt-5.5")
            .expect("disable unsupported tier")
    );
    assert_eq!(mode.tier(), None);
}
