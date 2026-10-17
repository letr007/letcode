use super::*;

fn temp_prefs_dir(name: &str) -> PathBuf {
    let base = std::env::temp_dir().join(format!(
        "letcode-theme-file-{name}-{}",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("time ok")
            .as_nanos()
    ));
    fs::create_dir_all(themes_dir(&base)).expect("create themes dir");
    base
}

#[test]
fn default_leaves_a_slot_to_the_terminal() {
    let prefs = temp_prefs_dir("default-color");
    fs::write(
        theme_file_path(&prefs, "translucent"),
        r##"
                root_bg = "default"
                element_bg = "Default"
                accent = "#010203"
            "##,
    )
    .expect("write theme");

    let theme = load_custom_theme(&prefs, "translucent").expect("load theme");

    assert_eq!(theme.root_bg, Color::Reset);
    assert_eq!(theme.element_bg, Color::Reset);
    assert_eq!(theme.accent, Color::Rgb(0x01, 0x02, 0x03));
    assert!(!theme.paints_panels());
}

#[test]
fn discovered_ids_load_the_discovered_file() {
    let prefs = temp_prefs_dir("discover-roundtrip");
    fs::write(
        theme_file_path(&prefs, "tokyo-night"),
        r##"label = "Mine"
accent = "#010203""##,
    )
    .expect("write theme");

    let ids = discover_custom_themes(&prefs)
        .into_iter()
        .map(|theme| theme.id)
        .collect::<Vec<_>>();
    assert_eq!(ids, vec!["tokyo-night"]);

    let theme = load_custom_theme(&prefs, "tokyo-night").expect("load discovered theme");
    assert_eq!(theme.accent, Color::Rgb(0x01, 0x02, 0x03));
}

#[test]
fn unknown_color_values_name_their_field() {
    let prefs = temp_prefs_dir("bad-color");
    fs::write(
        theme_file_path(&prefs, "translucent"),
        r##"muted_text = "defaults""##,
    )
    .expect("write theme");

    let error = load_custom_theme(&prefs, "translucent").expect_err("reject unknown value");

    assert!(format!("{error:#}").contains("muted_text"), "{error:#}");
}

#[test]
fn ensure_bundled_themes_seeds_missing_only() {
    let prefs = temp_prefs_dir("bundled");
    ensure_bundled_themes(&prefs);

    let anthropic_path = theme_file_path(&prefs, "anthropic");
    assert!(anthropic_path.exists());
    assert_eq!(
        load_custom_theme(&prefs, "anthropic")
            .expect("load anthropic")
            .accent,
        Color::Rgb(0xd9, 0x77, 0x57)
    );

    fs::write(&anthropic_path, r##"accent = "#010203""##).expect("edit anthropic");
    ensure_bundled_themes(&prefs);
    assert_eq!(
        load_custom_theme(&prefs, "anthropic")
            .expect("reload anthropic")
            .accent,
        Color::Rgb(0x01, 0x02, 0x03)
    );

    let ids = discover_custom_themes(&prefs)
        .into_iter()
        .map(|theme| theme.id)
        .collect::<Vec<_>>();
    assert_eq!(
        ids,
        vec![
            "anthropic",
            "catppuccin-frappe",
            "catppuccin-latte",
            "catppuccin-macchiato",
            "catppuccin-mocha",
            "github-dark",
            "github-light",
            "gruvbox-light",
            "matrix",
            "tokyonight-day",
            "tokyonight-moon",
            "tokyonight-night",
            "tokyonight-storm",
        ]
    );
}
