use super::{LAYOUT_END, LAYOUT_START, NAMED_END, NAMED_START, PROTECTED_SPACE};

#[test]
fn arrays_require_valid_column_specs_and_render_rules() {
    let rendered = super::render_text(
        r"\begin{array}{|l|c|r|} a & b & c \\ \hline d & e & f \\ g & h & i \end{array}",
        true,
    )
    .expect("array");
    assert!(rendered.contains("│ a"));
    assert!(rendered.contains("──"));
    assert!(rendered.contains("│ c │"));
    assert!(
        super::render_text(
            r"\begin{array}{c|ccc} r_1 & 1 & 0 & 2 \\ r_2 & 0 & 1 & 3 \end{array}",
            true,
        )
        .is_some()
    );
    assert!(super::render_text(r"\begin{array}{ccc} a & b & c \end{array}", true).is_some());
    assert!(super::render_text(r"\begin{array}{|l||c} a & b \end{array}", true).is_none());
    assert!(super::render_text(r"\begin{array}{lq} a & b \end{array}", true).is_none());
    assert!(super::render_text(r"\begin{array} a & b \end{array}", true).is_none());
    assert!(super::render_text(r"\begin{array}{lc} a & b \end{array}", true).is_some());
    assert!(super::render_text(r"\begin{array}{l} a & b \end{array}", true).is_none());
    assert!(super::render_text(r"\begin{array}{l} \hline \end{array}", true).is_none());
    assert!(super::render_text(r"\begin{array}{l} \hlinefoo x \end{array}", true).is_none());
}

#[test]
fn extended_arrows_require_upper_labels_and_keep_optional_lower_labels() {
    assert_eq!(
        super::render_text(r"A \xrightarrow{f} B", false).as_deref(),
        Some("A ─f→ B")
    );
    assert_eq!(
        super::render_text(r"A \xleftarrow[g]{h} B", false).as_deref(),
        Some("A ─h (g)← B")
    );
    assert_eq!(
        super::render_text(r"A \xleftarrow[\alpha] {\beta} B", false).as_deref(),
        Some("A ─β (α)← B")
    );
    assert!(super::render_text(r"A \xrightarrow B", false).is_none());
    assert!(super::render_text(r"A \xrightarrow{} B", false).is_none());
    assert!(super::render_text(r"A \xrightarrow[] {f} B", false).is_none());
    assert!(super::render_text(r"A \xrightarrow[g{f} B", false).is_none());
}

#[test]
fn latex_space_command_is_not_a_matrix_row_separator() {
    assert_eq!(super::render_text(r"a\ b", false).as_deref(), Some("a b"));
}

#[test]
fn malformed_optional_arguments_and_internal_sentinels_fail_closed() {
    assert!(super::render_text(r"\sqrt[3{x}", false).is_none());
    for sentinel in [
        LAYOUT_START,
        LAYOUT_END,
        PROTECTED_SPACE,
        NAMED_START,
        NAMED_END,
    ] {
        let source = format!("x{sentinel}y");
        assert!(super::render_text(&source, false).is_none(), "{source:?}");
    }
}

#[test]
fn unicode_width_and_resource_limits_are_enforced() {
    let layout = super::render("变量🙂", false).expect("unicode math");
    assert_eq!(layout.width, crate::tui::measure::display_width("变量🙂"));
    assert!(
        super::render_text(
            &"x".repeat(super::MathLimits::default().max_source_chars + 1),
            false
        )
        .is_none()
    );

    let too_many_rows = format!(
        r"\begin{{matrix}}{}\end{{matrix}}",
        (0..=super::MathLimits::default().max_rows)
            .map(|_| "x")
            .collect::<Vec<_>>()
            .join(r"\\")
    );
    assert!(super::render(&too_many_rows, true).is_none());

    let too_wide = "x".repeat(257);
    assert!(super::render(&too_wide, false).is_none());
}
