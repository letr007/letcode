use super::*;

#[test]
fn light_scheme_selects_a_light_syntax_theme() {
    let themes = ThemeSet::load_defaults();

    assert!(
        themes.themes.contains_key("InspiredGitHub"),
        "available: {:?}",
        themes.themes.keys().collect::<Vec<_>>()
    );
    assert_ne!(syntect_theme(true).name, syntect_theme(false).name);
}

fn rendered(markdown: &str, width: usize) -> Vec<Line<'static>> {
    render_markdown(markdown, Theme::dark(), MarkdownRenderOptions::new(width))
}

#[test]
fn markdown_links_retain_open_url_interaction_through_wrapping() {
    let document = render_markdown_document(
        "[a long link label](https://example.test/path)",
        Theme::dark(),
        MarkdownRenderOptions::new(4),
    );
    assert!(document.validate());
    let links = document
        .lines
        .iter()
        .flat_map(|line| &line.spans)
        .filter_map(|span| span.interaction.as_ref())
        .collect::<Vec<_>>();
    assert!(links.iter().all(|interaction| matches!(
        interaction,
        crate::tui::transcript_render::Interaction::OpenUrl(url)
            if url == "https://example.test/path"
    )));
    assert!(links.len() > 1, "{document:?}");
    assert!(
        document
            .lines
            .iter()
            .flat_map(|line| &line.spans)
            .filter(|span| span.text.contains("link") || span.text.contains("label"))
            .all(|span| span.interaction.is_some())
    );
    assert!(
        document
            .lines
            .iter()
            .flat_map(|line| &line.spans)
            .filter(|span| span.text.contains('<') || span.text.contains("example"))
            .all(|span| span.interaction.is_none()),
        "destination suffix must not be jumpable: {document:?}"
    );
}

#[test]
fn legacy_bridge_preserves_markdown_visuals() {
    let lines = rendered("# Title\n- **item** with `code`", 80);
    let text = lines
        .iter()
        .map(Line::to_string)
        .collect::<Vec<_>>()
        .join("\n");
    assert!(text.contains("▌ Title"), "{text}");
    assert!(text.contains("• item with code"), "{text}");
    assert!(!text.contains("**item**"), "{text}");
    assert!(!text.contains('`'), "{text}");
}

#[test]
fn code_blocks_preserve_card_layout_and_spacing() {
    let lines = rendered("before\n\n```rust\nlet x = 1;\n```\n\nafter", 24);
    let text = lines
        .iter()
        .map(Line::to_string)
        .collect::<Vec<_>>()
        .join("\n");

    assert_eq!(lines[0].to_string(), "before");
    assert_eq!(lines[1].to_string(), "");
    assert!(lines[2].to_string().contains("╭─ rust"), "{text}");
    assert!(
        lines
            .iter()
            .any(|line| line.to_string().contains("│ let x = 1;")),
        "{text}"
    );
    assert!(
        lines.iter().any(|line| line.to_string().contains('╰')),
        "{text}"
    );
    assert_eq!(lines[lines.len() - 2].to_string(), "");
    assert_eq!(lines.last().expect("after line").to_string(), "after");
    for line in &lines[2..=4] {
        assert_eq!(display_width(&line.to_string()), 24, "{line:?}");
    }
}

#[test]
fn code_blocks_apply_syntax_highlighting_and_plain_fallback() {
    let theme = Theme::dark();
    let highlighted = rendered("```rust\nlet x = 1;\n```", 32);
    let body = highlighted
        .iter()
        .find(|line| line.to_string().contains("let x"))
        .expect("highlighted code body line");
    assert!(
        body.spans.iter().any(|span| {
            !span.content.trim().is_empty()
                && span.content.as_ref() != "│ "
                && span.style.bg == Some(theme.element_bg)
                && span.style.fg != Some(theme.text)
        }),
        "{body:?}"
    );

    let plain = rendered("```definitely-not-a-language\nlet x = 1;\n```", 32);
    let body = plain
        .iter()
        .find(|line| line.to_string().contains("let x"))
        .expect("plain code body line");
    assert!(
        body.spans.iter().any(|span| {
            span.content.as_ref() == "let x = 1;"
                && span.style.fg == Some(theme.text)
                && span.style.bg == Some(theme.element_bg)
        }),
        "{body:?}"
    );
}

#[test]
fn latex_parenthesis_and_bracket_delimiters_render_outside_code() {
    let markdown = r"inline \(e^{i\pi}+1=0\).

\[\int_{-\infty}^{+\infty} e^{-x^2}\,dx = \sqrt{\pi}\]

`\(code\)`

```text
\[code block\]
```";
    let document =
        render_markdown_document(markdown, Theme::dark(), MarkdownRenderOptions::new(80));
    let text = document
        .lines
        .iter()
        .map(|line| render_span_text(&line.spans))
        .collect::<Vec<_>>()
        .join("\n");
    assert!(text.contains("e^(iπ)+1 = 0"), "{text}");
    assert!(text.contains('∫'), "{text}");
    assert!(text.contains("√π"), "{text}");
    assert!(text.contains(r"\(code\)"), "{text}");
    assert!(text.contains(r"\[code block\]"), "{text}");
    assert!(document.validate(), "{document:?}");
}

#[test]
fn math_delimiter_normalization_requires_pairs_and_respects_code_boundaries() {
    let markdown = "paired \\(x\\)\nunclosed \\(x\nescaped \\\\(x\\\\)\n`\\(inline code\\)`\n````text\n\\[fenced code\\]\n```\n\\(still fenced\\)\n````\nmath \\(y\\)";
    assert_eq!(
        normalize_markdown_math_delimiters(markdown),
        "paired $x$\nunclosed \\(x\nescaped \\\\(x\\\\)\n`\\(inline code\\)`\n````text\n\\[fenced code\\]\n```\n\\(still fenced\\)\n````\nmath $y$"
    );
    assert_eq!(
        normalize_markdown_math_delimiters("`unclosed \\(x\\)"),
        "`unclosed $x$"
    );
    assert_eq!(
        normalize_markdown_math_delimiters("mismatched \\(x\\]"),
        "mismatched \\(x\\]"
    );
    assert_eq!(
        normalize_markdown_math_delimiters("unclosed \\(x\nlater \\)"),
        "unclosed \\(x\nlater \\)"
    );
    assert_eq!(
        normalize_markdown_math_delimiters("unclosed \\[x\n\nlater \\]"),
        "unclosed \\[x\n\nlater \\]"
    );
    assert_eq!(
        normalize_markdown_math_delimiters("display \\[x +\n y\\]"),
        "display $$x +\n y$$"
    );
}

#[test]
fn mermaid_dag_and_cycle_render_without_code_fallback() {
    let supported = render_markdown_document(
        "```mermaid\ngraph TD\nA[Start]\nB[Finish]\nA --> B\n```",
        Theme::dark(),
        MarkdownRenderOptions::new(32),
    );
    let supported_text = supported
        .lines
        .iter()
        .map(|line| render_span_text(&line.spans))
        .collect::<Vec<_>>()
        .join("\n");
    assert!(supported_text.contains("Start"), "{supported_text}");
    assert!(supported_text.contains("Finish"), "{supported_text}");
    // 二维画布：节点框化，垂直箭头连接。
    assert!(supported_text.contains('╭'), "{supported_text}");
    assert!(supported_text.contains('│'), "{supported_text}");
    assert!(supported_text.contains('▼'), "{supported_text}");
    assert!(!supported_text.contains("╭─ mermaid"), "{supported_text}");
    assert!(supported.validate(), "{supported:?}");

    let cycle = rendered("```mermaid\ngraph TD\nA --> B\nB --> A\n```", 32);
    let cycle_text = cycle
        .iter()
        .map(Line::to_string)
        .collect::<Vec<_>>()
        .join("\n");
    assert!(cycle_text.contains("A"), "{cycle_text}");
    assert!(cycle_text.contains("B"), "{cycle_text}");
    assert!(!cycle_text.contains("╭─ mermaid"), "{cycle_text}");
}

#[test]
fn mermaid_flowchart_labels_and_sequence_diagram_render() {
    let flowchart = render_markdown_document(
        "```mermaid\nflowchart TD\nA[开始] --> B{渲染类型}\nB -->|LaTeX| C[解析数学公式]\nB -->|Mermaid| D[解析流程图]\n```",
        Theme::dark(),
        MarkdownRenderOptions::new(80),
    );
    let flowchart_text = flowchart
        .lines
        .iter()
        .map(|line| render_span_text(&line.spans))
        .collect::<Vec<_>>()
        .join("\n");
    assert!(flowchart_text.contains("渲染类型"), "{flowchart_text}");
    assert!(flowchart_text.contains("LaTeX"), "{flowchart_text}");
    assert!(flowchart_text.contains("Mermaid"), "{flowchart_text}");
    assert!(!flowchart_text.contains("╭─ mermaid"), "{flowchart_text}");
    assert!(flowchart.validate(), "{flowchart:?}");

    let sequence = render_markdown_document(
        "```mermaid\nsequenceDiagram\nparticipant U as 用户\nparticipant C as 客户端\nparticipant R as 渲染器\nU->>C: 提交 Markdown\nC->>R: 发送 LaTeX / Mermaid 内容\nR-->>C: 返回渲染结果\n```",
        Theme::dark(),
        MarkdownRenderOptions::new(80),
    );
    let sequence_text = sequence
        .lines
        .iter()
        .map(|line| render_span_text(&line.spans))
        .collect::<Vec<_>>()
        .join("\n");
    assert!(sequence_text.contains("用户"), "{sequence_text}");
    assert!(sequence_text.contains("客户端"), "{sequence_text}");
    assert!(sequence_text.contains("渲染器"), "{sequence_text}");
    assert!(
        sequence_text
            .lines()
            .any(|line| line.contains("提交 Markdown") && line.contains('▶')),
        "{sequence_text}"
    );
    assert!(
        sequence_text
            .lines()
            .any(|line| line.contains("返回渲染结果") && line.contains('◀')),
        "{sequence_text}"
    );
    assert!(!sequence_text.contains("╭─ mermaid"), "{sequence_text}");
    assert!(sequence.validate(), "{sequence:?}");

    let structured = rendered(
        "```mermaid\nsequenceDiagram\nparticipant A as A\nloop retry\nA->>A: again\nend\n```",
        80,
    );
    let structured_text = structured
        .iter()
        .map(Line::to_string)
        .collect::<Vec<_>>()
        .join("\n");
    assert!(!structured_text.contains("╭─ mermaid"), "{structured_text}");
    assert!(structured_text.contains("loop retry"), "{structured_text}");
    assert!(structured_text.contains("◀┘again"), "{structured_text}");

    let too_wide = rendered(
        "```mermaid\nsequenceDiagram\nparticipant A as A\nparticipant B as B\nA->>B: a message that cannot fit\n```",
        12,
    );
    let too_wide_text = too_wide
        .iter()
        .map(Line::to_string)
        .collect::<Vec<_>>()
        .join("\n");
    assert!(too_wide_text.contains("╭─ mermaid"), "{too_wide_text}");
}

#[test]
fn mermaid_mindmap_and_timeline_render_and_fallback_cleanly() {
    for (markdown, expected) in [
        (
            "```mermaid\nmindmap\n根🙂\n  branch[标签✅]\n    leaf((详情🚀))\n```",
            "详情🚀",
        ),
        (
            "```mermaid\ntimeline\n    title 发布计划🙂\n    section 客户端✅\n    2004 : 开始🚀 : 完成\n         : 验收\n```",
            "验收",
        ),
    ] {
        let document =
            render_markdown_document(markdown, Theme::dark(), MarkdownRenderOptions::new(60));
        let text = document
            .lines
            .iter()
            .map(|line| render_span_text(&line.spans))
            .collect::<Vec<_>>()
            .join("\n");
        assert!(text.contains(expected), "{text}");
        assert!(!text.contains("╭─ mermaid"), "{text}");
        assert!(document.validate(), "{document:?}");
    }
    let fallback = rendered("```mermaid\nmindmap\nroot\n  child:::class\n```", 60);
    let text = fallback
        .iter()
        .map(Line::to_string)
        .collect::<Vec<_>>()
        .join("\n");
    assert!(text.contains("╭─ mermaid"), "{text}");

    let narrow = rendered(
        "```mermaid\ntimeline\n    2004 : This event is too wide\n```",
        16,
    );
    let text = narrow
        .iter()
        .map(Line::to_string)
        .collect::<Vec<_>>()
        .join("\n");
    assert!(text.contains("╭─ mermaid"), "{text}");
}

#[test]
fn common_mermaid_families_render_and_fallback_cleanly() {
    let cases = [
        (
            "```mermaid\nflowchart LR\nA[letcode<br>binary] --> B[composition root]\n```",
            "binary",
        ),
        (
            "```mermaid\nclassDiagram\nclass User {\n  +name String\n}\nclass Service\nUser --> Service : calls\n```",
            "calls",
        ),
        (
            "```mermaid\nerDiagram\nUSER {\n  string name\n}\nORDER {\n  int id\n}\nUSER ||--o{ ORDER : creates\n```",
            "creates",
        ),
        (
            "```mermaid\ngantt\ntitle Release plan\ndateFormat YYYY-MM-DD\nsection Client\nImplement renderer : active, render, 2026-08-09, 3d\n```",
            "Implement renderer",
        ),
        (
            "```mermaid\nstateDiagram-v2\nstate \"Waiting\" as Waiting\nstate Done\n[*] --> Waiting : start\nWaiting --> Done : finish\nDone --> [*]\n```",
            "finish",
        ),
        (
            "```mermaid\nmindmap\nRoot\n  Client\n    Renderer\n```",
            "Renderer",
        ),
        (
            "```mermaid\ntimeline\n    title Release plan\n    section Client\n    2026 : Implement renderer : Ship\n```",
            "Ship",
        ),
        (
            "```mermaid\nsequenceDiagram\nparticipant 用户\nparticipant 前端\n用户->>前端: 点击登录\n前端-->>用户: 登录成功\n```",
            "登录成功",
        ),
        (
            "```mermaid\npie title Language share\n\"TypeScript\" : 45\n\"Rust\" : 30\n\"Python\" : 15\n\"Other\" : 10\n```",
            "█ TypeScript  45%",
        ),
        (
            "```mermaid\njourney\ntitle Delivery\nsection Build\nImplement renderer: 5: Agent\nVerify output: 4: User\n```",
            "●●●●○",
        ),
        (
            "```mermaid\ngitGraph\ncommit id: start\nbranch feature\ncommit id: implement\ncheckout main\nmerge feature\n```",
            "merge feature",
        ),
        (
            "```mermaid\nquadrantChart\ntitle Technology choice\nx-axis Low cost --> High cost\ny-axis Low return --> High return\nquadrant-1 Invest\nquadrant-2 Evaluate\nquadrant-3 Avoid\nquadrant-4 Trial\nPostgreSQL: [0.4, 0.85]\n```",
            "PostgreSQL",
        ),
    ];
    for (markdown, expected) in cases {
        let document =
            render_markdown_document(markdown, Theme::dark(), MarkdownRenderOptions::new(80));
        let text = document
            .lines
            .iter()
            .map(|line| render_span_text(&line.spans))
            .collect::<Vec<_>>()
            .join("\n");
        assert!(text.contains(expected), "{text}");
        assert!(!text.contains("╭─ mermaid"), "{text}");
        assert!(document.validate(), "{document:?}");
    }

    for markdown in [
        "```mermaid\npie\ntitle unsupported\n```",
        "```mermaid\nclassDiagram\nclass A {\n  +x\n```",
        "```mermaid\nstateDiagram\ndirection LR\nA --> B\n```",
        "```mermaid\nmindmap\nroot\n  child:::class\n```",
        "```mermaid\ntimeline\n    2026 : event :\n```",
        "```mermaid\njourney\nsection Build\nInvalid score: 8: Agent\n```",
        "```mermaid\ngitGraph\ncheckout missing\ncommit\n```",
        "```mermaid\nquadrantChart\nx-axis low --> high\ny-axis low --> high\nquadrant-1 one\nA: [0.5, 0.5]\n```",
    ] {
        let text = rendered(markdown, 80)
            .iter()
            .map(Line::to_string)
            .collect::<Vec<_>>()
            .join("\n");
        assert!(text.contains("╭─ mermaid"), "{text}");
    }

    let narrow = rendered(
        "```mermaid\ngantt\ntitle A release plan that cannot fit\n```",
        12,
    )
    .iter()
    .map(Line::to_string)
    .collect::<Vec<_>>()
    .join("\n");
    assert!(narrow.contains("╭─ mermaid"), "{narrow}");
}

#[test]
fn mermaid_waits_for_the_closing_fence_before_rendering() {
    let open = render_markdown_document(
        "```mermaid\nflowchart TD\nA[Start] --> B[End]\n",
        Theme::dark(),
        MarkdownRenderOptions::new(80),
    );
    let open_text = open
        .lines
        .iter()
        .map(|line| render_span_text(&line.spans))
        .collect::<Vec<_>>()
        .join("\n");
    assert!(open_text.contains("╭─ mermaid"), "{open_text}");
    assert!(!open_text.contains('▼'), "{open_text}");

    let closed = render_markdown_document(
        "```mermaid\nflowchart TD\nA[Start] --> B[End]\n```",
        Theme::dark(),
        MarkdownRenderOptions::new(80),
    );
    let closed_text = closed
        .lines
        .iter()
        .map(|line| render_span_text(&line.spans))
        .collect::<Vec<_>>()
        .join("\n");
    assert!(!closed_text.contains("╭─ mermaid"), "{closed_text}");
    assert!(closed_text.contains('▼'), "{closed_text}");
}

#[test]
fn indented_code_with_mermaid_fence_text_does_not_shift_closure_state() {
    let source = concat!(
        "    ```mermaid\n",
        "    flowchart TD\n",
        "    A --> B\n",
        "    ```\n",
        "```mermaid\n",
        "flowchart TD\n",
        "C --> D\n",
    );
    let document = render_markdown_document(source, Theme::dark(), MarkdownRenderOptions::new(80));
    let text = document
        .lines
        .iter()
        .map(|line| render_span_text(&line.spans))
        .collect::<Vec<_>>()
        .join("\n");
    assert!(text.contains("╭─ code"), "{text}");
    assert!(text.contains("╭─ mermaid"), "{text}");
    assert!(!text.contains('▼'), "{text}");
}

#[test]
fn mermaid_fence_closure_tracks_blockquotes_and_lists() {
    for source in [
        "> ```mermaid\n> flowchart TD\n> A --> B\n> ```\n",
        "- ```mermaid\n  flowchart TD\n  A --> B\n  ```\n",
        "10. ```mermaid\n    flowchart TD\n    A --> B\n    ```\n",
        "> - ```mermaid\n>   flowchart TD\n>   A --> B\n>   ```\n",
        "- outer\n    - ```mermaid\n      flowchart TD\n      A --> B\n      ```\n",
    ] {
        let document =
            render_markdown_document(source, Theme::dark(), MarkdownRenderOptions::new(80));
        let text = document
            .lines
            .iter()
            .map(|line| render_span_text(&line.spans))
            .collect::<Vec<_>>()
            .join("\n");
        assert!(!text.contains("╭─ mermaid"), "{text}");
        assert!(text.contains("A"), "{text}");
        assert!(text.contains("B"), "{text}");
    }
}

#[test]
fn mermaid_waits_for_matching_tilde_and_long_backtick_fences() {
    for source in [
        "~~~mermaid\nflowchart TD\nA --> B\n",
        "````mermaid\nflowchart TD\nA --> B\n```\n",
    ] {
        let document =
            render_markdown_document(source, Theme::dark(), MarkdownRenderOptions::new(80));
        let text = document
            .lines
            .iter()
            .map(|line| render_span_text(&line.spans))
            .collect::<Vec<_>>()
            .join("\n");
        assert!(text.contains("╭─ mermaid"), "{text}");
    }
}

#[test]
fn mermaid_supports_directions_shapes_edge_styles_comments_and_subgraphs() {
    let source = concat!(
        "```mermaid\n",
        // 方向别名 + 注释 + 分号多语句
        "flowchart BT\n",
        "%% a comment line\n",
        "subgraph SG[服务层]\n",
        "A[开始] --> B[(数据库)]\n",
        "B --> C((缓存)); C ==> D{{结果}}\n",
        "end\n",
        "D --- E[/中间/]\n",
        "E -.-> F[结束]\n",
        "```",
    );
    let document = render_markdown_document(source, Theme::dark(), MarkdownRenderOptions::new(80));
    assert!(document.validate(), "{document:?}");
    let text = document
        .lines
        .iter()
        .map(|line| render_span_text(&line.spans))
        .collect::<Vec<_>>()
        .join("\n");
    assert!(!text.contains("╭─ mermaid"), "{text}");
    assert!(text.contains("开始"), "{text}");
    assert!(text.contains("数据库"), "{text}");
    assert!(text.contains("缓存"), "{text}");
    assert!(text.contains("结果"), "{text}");
    assert!(text.contains("中间"), "{text}");
    assert!(text.contains("结束"), "{text}");
    // 二维画布：节点框化 + 垂直连线。
    assert!(text.contains('╭'), "{text}");
    assert!(text.contains('│'), "{text}");
    assert!(text.contains('▼'), "{text}");
}

#[test]
fn mermaid_edge_text_labels_and_more_directions_render() {
    let source = concat!(
        "```mermaid\n",
        "graph RL\n",
        "A -- text --> B\n",
        "C ==>|thick| D\n",
        "E -. dashed .-> F\n",
        "```",
    );
    let document = render_markdown_document(source, Theme::dark(), MarkdownRenderOptions::new(80));
    assert!(document.validate(), "{document:?}");
    let text = document
        .lines
        .iter()
        .map(|line| render_span_text(&line.spans))
        .collect::<Vec<_>>()
        .join("\n");
    assert!(!text.contains("╭─ mermaid"), "{text}");
    assert!(text.contains("text"), "{text}");
    assert!(text.contains("thick"), "{text}");
    assert!(text.contains("dashed"), "{text}");
}

#[test]
fn mermaid_supports_tb_bu_and_lr_aliases() {
    for dir in ["TB", "TD", "BT", "LR", "RL"] {
        let document = render_markdown_document(
            &format!("```mermaid\ngraph {dir}\nA[Start] --> B[End]\n```"),
            Theme::dark(),
            MarkdownRenderOptions::new(40),
        );
        assert!(document.validate(), "{document:?}");
        let text = document
            .lines
            .iter()
            .map(|line| render_span_text(&line.spans))
            .collect::<Vec<_>>()
            .join("\n");
        assert!(text.contains("Start"), "{dir}: {text}");
        assert!(text.contains("End"), "{dir}: {text}");
        assert!(!text.contains("╭─ mermaid"), "{dir}: {text}");
        // 垂直与横向布局都保留方向对应的连线。
        if matches!(dir, "TB" | "TD" | "BT") {
            assert!(text.contains('╭'), "{dir}: {text}");
            assert!(text.contains('│'), "{dir}: {text}");
        } else if dir == "LR" {
            assert!(text.contains('▶'), "{dir}: {text}");
        } else {
            assert!(text.contains('◀'), "{dir}: {text}");
        }
    }
}

#[test]
fn math_renders_unicode_ast_and_falls_back_atomically_when_too_wide() {
    let supported = rendered("inline $\\alpha^2$ and $$x_1$$", 32);
    let supported_text = supported
        .iter()
        .map(Line::to_string)
        .collect::<Vec<_>>()
        .join("\n");
    assert!(supported_text.contains("α²"), "{supported_text}");
    assert!(supported_text.contains("x₁"), "{supported_text}");

    let exact_width = render_markdown_document(
        "$\\alpha^2$",
        Theme::dark(),
        MarkdownRenderOptions::new(display_width("α²")),
    );
    assert_eq!(
        exact_width
            .lines
            .iter()
            .map(|line| render_span_text(&line.spans))
            .collect::<String>(),
        "α²"
    );
    assert!(
        exact_width
            .lines
            .iter()
            .flat_map(|line| &line.spans)
            .all(|span| span.copy_mode == crate::tui::transcript_render::CopyMode::Atomic),
        "{exact_width:?}"
    );

    let too_wide = rendered("$\\frac{1}{x}$", 2);
    let too_wide_text = too_wide
        .iter()
        .map(Line::to_string)
        .collect::<Vec<_>>()
        .join("");
    assert!(too_wide_text.contains("\\frac{1}{x}"), "{too_wide_text}");
}

#[test]
fn math_box_layout_supports_frac_sqrt_integral_matrices_and_scripts() {
    let document = render_markdown_document(
        "$$\\frac{\\sqrt{x}}{2} + \\int_0^1 f(x) dx$$\n\n$$\\begin{pmatrix}a&b\\\\c&d\\end{pmatrix}$$",
        Theme::dark(),
        MarkdownRenderOptions::new(64),
    );
    let text = document
        .lines
        .iter()
        .map(|line| render_span_text(&line.spans))
        .collect::<Vec<_>>()
        .join("\n");
    assert!(text.contains('─'), "{text}");
    assert!(text.contains('√'), "{text}");
    assert!(text.contains('∫'), "{text}");
    assert!(text.contains("⎛ a │ b ⎞"), "{text}");
    assert!(text.contains("⎝ c │ d ⎠"), "{text}");
    assert!(document.validate(), "{document:?}");
}

#[test]
fn transformed_mermaid_output_has_exact_label_ranges_and_isolated_nodes() {
    let prefixed = render_markdown_document(
        "> ```mermaid\n> graph LR\n> A[One]\n> B[Two]\n> C[Alone]\n> A --> B\n> ```",
        Theme::dark(),
        MarkdownRenderOptions::new(40),
    );
    let prefixed_text = prefixed
        .lines
        .iter()
        .map(|line| render_span_text(&line.spans))
        .collect::<Vec<_>>();
    let rendered_prefixed_lines = prefixed_text
        .iter()
        .filter(|text| text.contains("One") || text.contains("Two") || text.contains("Alone"))
        .collect::<Vec<_>>();
    assert!(rendered_prefixed_lines.len() >= 2);
    for label in ["One", "Two", "Alone"] {
        assert!(
            rendered_prefixed_lines
                .iter()
                .any(|line| line.contains(label)),
            "missing {label}: {prefixed_text:?}"
        );
    }
    assert!(
        rendered_prefixed_lines
            .iter()
            .all(|line| line.starts_with("│ "))
    );

    let markdown = "```mermaid\ngraph LR\nA[One]\nB[Two]\nC[Alone]\nA --> B\n```";
    let document =
        render_markdown_document(markdown, Theme::dark(), MarkdownRenderOptions::new(40));
    assert!(document.validate(), "{document:?}");
    assert!(
        document
            .lines
            .iter()
            .flat_map(|line| &line.spans)
            .any(|span| span.text == "Alone")
    );
    let source = &document.source_blocks[0].source;
    for span in document.lines.iter().flat_map(|line| &line.spans) {
        if let Some(range) = span.source {
            assert_eq!(
                source
                    .chars()
                    .skip(range.start)
                    .take(range.end - range.start)
                    .collect::<String>(),
                span.text
            );
        }
    }

    let sequence_source =
        "sequenceDiagram\nparticipant A as A\nparticipant B as 接收方\nA->>B: A sends\n";
    let sequence = render_markdown_document(
        &format!("```mermaid\n{sequence_source}```"),
        Theme::dark(),
        MarkdownRenderOptions::new(40),
    );
    assert!(sequence.validate(), "{sequence:?}");
    let sequence_block = sequence
        .source_blocks
        .iter()
        .position(|block| block.source == sequence_source)
        .expect("sequence source block");
    for span in sequence.lines.iter().flat_map(|line| &line.spans) {
        let Some(range) = span.source else {
            continue;
        };
        if range.block_index != sequence_block {
            continue;
        }
        assert_eq!(
            sequence.source_blocks[range.block_index]
                .source
                .chars()
                .skip(range.start)
                .take(range.end - range.start)
                .collect::<String>(),
            span.text
        );
    }
}

#[test]
fn transformed_math_has_atomic_full_source_mapping_and_width_fallback() {
    let transformed = render_markdown_document(
        "The value is $$\\frac{1}{x}$$.",
        Theme::dark(),
        MarkdownRenderOptions::new(40),
    );
    assert!(transformed.validate(), "{transformed:?}");
    let math_span = transformed
        .lines
        .iter()
        .flat_map(|line| &line.spans)
        .find(|span| span.text.contains('─'))
        .expect("transformed fraction span");
    assert_eq!(
        math_span.copy_mode,
        crate::tui::transcript_render::CopyMode::Atomic
    );
    let range = math_span.source.expect("atomic source range");
    assert_eq!(
        transformed.source_blocks[range.block_index].source,
        "\\frac{1}{x}"
    );
    assert_eq!(range.start, 0);
    assert_eq!(range.end, "\\frac{1}{x}".chars().count());

    let fallback = render_markdown_document(
        "$\\frac{1}{x}$",
        Theme::dark(),
        MarkdownRenderOptions::new(2),
    );
    assert!(fallback.validate(), "{fallback:?}");
    let fallback_text = fallback
        .lines
        .iter()
        .map(|line| render_span_text(&line.spans))
        .collect::<String>();
    assert!(fallback_text.contains("\\frac{1}{x}"), "{fallback_text}");
    assert!(
        fallback
            .lines
            .iter()
            .flat_map(|line| &line.spans)
            .any(|span| span.copy_mode == crate::tui::transcript_render::CopyMode::Atomic)
    );
}

#[test]
fn display_math_accepts_normalized_multiline_input_and_rejects_malformed_groups() {
    assert!(crate::tui::math::render("\\int _{0}\n ^{1} f(x) + \\frac{1}{x}", true).is_some());
    let document = render_markdown_document(
        "$$\\int_{0}^{1} f(x)\\frac{1}{x}$$",
        Theme::dark(),
        MarkdownRenderOptions::new(64),
    );
    let text = document
        .lines
        .iter()
        .map(|line| render_span_text(&line.spans))
        .collect::<Vec<_>>()
        .join("\n");
    assert!(text.contains('∫'), "{text}");
    assert!(text.contains('─'), "{text}");
    assert!(document.validate(), "{document:?}");

    let malformed = render_markdown_document(
        "$\\frac{1}{x$",
        Theme::dark(),
        MarkdownRenderOptions::new(64),
    );
    let fallback = malformed
        .lines
        .iter()
        .map(|line| render_span_text(&line.spans))
        .collect::<String>();
    assert!(fallback.contains("\\frac{1}{x"), "{fallback}");
    assert!(malformed.validate(), "{malformed:?}");
}

#[test]
fn latex_official_corpus_inline_and_malformed_cases() {
    let cases = [
        (r"\mathbb{C}^3 \to \mathbb{C}^3", "ℂ³ → ℂ³"),
        (
            r"\{3x+2y,\; 27x^2-4z-1,\; x(x-1)(x+1)\} \quad\Rightarrow\quad x \in \{0, \pm 1\},",
            "{3x+2y, 27x²-4z-1, x(x-1)(x+1)} ⇒ x ∈ {0, ± 1},",
        ),
        (r"F_1 = -\frac{1}{4x^2}.", "F₁ = -1/(4x²)."),
        (r"\mathbb{C}^*", "ℂ^*"),
        (
            r"s \mapsto (s,\, -\tfrac{3}{2s},\, \tfrac{13}{2s^2})",
            "s ↦ (s, -3/(2s), 13/(2s²))",
        ),
        (
            r"\boxed{1\ \text{milliwatt per square metre}}",
            "[1 milliwatt per square metre]",
        ),
        (
            r"\pi(2.5\ \text{km})^2 = 19.6\ \text{km}^2",
            "π(2.5 km)² = 19.6 km²",
        ),
        (
            r"\det\!\left(\frac{\partial(F_1,F_2,F_3)}{\partial(x,y,z)}\right)=-2.",
            "det((∂(F₁,F₂,F₃))/(∂(x,y,z))) = -2.",
        ),
        (r"e^{i\pi}+1=0", "e^(iπ)+1 = 0"),
        (
            r"x=\frac{-b\pm\sqrt{b^2-4ac}}{2a}",
            "x = (-b±√(b²-4ac))/(2a)",
        ),
        (
            r"\int_0^\infty e^{-x^2}\,dx=\frac{\sqrt{\pi}}{2}",
            "∫₀^∞ e^(-x²) dx = (√π)/2",
        ),
        (
            r"\sum_{n=1}^{\infty}\frac{1}{n^2}=\frac{\pi^2}{6}",
            "∑ₙ₌₁^∞1/(n²) = π²/6",
        ),
        (r"\lim_{x\to 0}\frac{\sin x}{x}=1", "lim[x→0] (sin x)/x = 1"),
        (
            r"\sqrt[2]{x}+\sqrt[3]{x}+\sqrt[4]{x}+\sqrt[n]{x}+\sqrt[k]{x+1}",
            "√x+∛x+∜x+ⁿ√x+ᵏ√(x+1)",
        ),
        (
            r"\acute{x}+\grave{y}+\widehat{xyz}+\overrightarrow{AB}",
            "x́+ỳ+widehat(xyz)+overrightarrow(AB)",
        ),
        (
            r"\textnormal{hello}+\mbox{world}+\boldsymbol{x}",
            "hello+world+x",
        ),
        (r"A\not\subseteq B,\quad x\not\in X", "A ⊈ B, x ∉ X"),
        (
            r"\lvert{x}\rvert+\lVert{v}\rVert+\left.\frac{dy}{dx}\right|_{x=0}",
            "|x|+‖v‖+dy/(dx)|ₓ₌₀",
        ),
        (
            r"\operatorname*{arg\,max}_{x\in X} f(x)",
            "arg max[x∈X] f(x)",
        ),
        (r"a\bmod n,\quad a\equiv b\pmod n", "a mod n, a ≡ b (mod n)"),
        (
            r"\overset{!}{=}+\underset{n}{x}+\stackrel{def}{=}",
            "=^!+xₙ+=ᵈᵉᶠ",
        ),
    ];
    for (source, expected) in cases {
        assert_eq!(
            crate::tui::math::render_text(source, false).as_deref(),
            Some(expected),
            "{source}"
        );
    }
    for source in [
        r"x + \unknown{y}",
        r"\frac{1}{x",
        "x}",
        r"\begin{matrix}1 & 2",
        r"x\\",
    ] {
        assert!(
            crate::tui::math::render_text(source, false).is_none(),
            "{source}"
        );
    }
}

#[test]
fn latex_official_corpus_display_golden_layouts() {
    let cases = [
        (r"\sum_{i=0}^n x_i", " n\n ∑  xᵢ\ni=0"),
        (r"\min_{x\in X} f(x)", "min f(x)\nx∈X"),
        (
            r"\operatorname*{arg\,max}_{x\in X} f(x)",
            "arg max f(x)\n  x∈X",
        ),
        (
            r"x=\frac{-b\pm\sqrt{b^2-4ac}}{2a}",
            "    -b±√(b²-4ac)\nx = ────────────\n         2a",
        ),
        (r"\frac{x^2+1}{x-1}", "x²+1\n────\nx-1"),
        (
            r"\begin{cases}a & x<0 \\ b & x=0 \\ c & x>0\end{cases}",
            "⎧ a if x < 0\n⎨ b if x = 0\n⎩ c if x > 0",
        ),
        (
            r"\begin{pmatrix}1&200\\3000&4\end{pmatrix}",
            "⎛ 1    │ 200 ⎞\n⎝ 3000 │ 4   ⎠",
        ),
    ];
    for (source, expected) in cases {
        assert_eq!(
            crate::tui::math::render_text(source, true).as_deref(),
            Some(expected),
            "{source}"
        );
    }
}

#[test]
fn ordered_lists_blockquotes_tasks_and_links_keep_visual_semantics() {
    let lines = rendered("> 1. [x] done\n> 2. [ ] [docs](https://example.com)", 80);
    let text = lines
        .iter()
        .map(Line::to_string)
        .collect::<Vec<_>>()
        .join("\n");
    assert!(text.contains("│ 1. ☑ done"), "{text}");
    assert!(text.contains("│ 2. ☐ docs <https://example.com>"), "{text}");
}

#[test]
fn tables_keep_headers_values_separator_and_wide_cells() {
    let md = "| 套件 | 结果 |\n| --- | --- |\n| runtime_context::tests | 12 passed (含 heal / replace_frames) |\n| runtime::tests | 184 passed |";
    let lines = rendered(md, 80);
    let text = lines
        .iter()
        .map(Line::to_string)
        .collect::<Vec<_>>()
        .join("\n");
    assert!(text.contains("套件"), "{text}");
    assert!(text.contains('┼'), "{text}");
    assert!(
        text.contains("12 passed (含 heal / replace_frames)"),
        "{text}"
    );
}

fn document_text(document: &Document<Style>) -> String {
    document
        .lines
        .iter()
        .map(|line| {
            line.spans
                .iter()
                .map(|span| span.text.as_str())
                .collect::<String>()
        })
        .collect::<Vec<_>>()
        .join("\n")
}

#[test]
fn table_cells_wrap_instead_of_dropping_overflow() {
    let markdown =
        "| 名称 | 说明 |\n| --- | --- |\n| 组件 | 这是一个非常非常长的说明文字用来观察换行行为 |";
    let document =
        render_markdown_document(markdown, Theme::dark(), MarkdownRenderOptions::new(24));
    assert!(document.validate(), "{document:?}");
    let text = document_text(&document);
    assert!(text.contains("这是一个非常非常"), "{text}");
    assert!(text.contains("观察换行行为"), "{text}");
}

#[test]
fn streaming_markdown_matches_full_render() {
    let samples: Vec<Vec<&str>> = vec![
        vec![
            "first paragraph\n\n",
            "second paragraph\n\n",
            "third paragraph\n\n",
        ],
        vec!["intro with `code`\n\n", "more `inline` text\n\n"],
        vec!["opening `tick\n\n", "closing` tick\n\n", "done\n\n"],
        vec![
            "see [alpha]\n\n",
            "[alpha]: https://example.test\n\n",
            "after\n\n",
        ],
        vec!["- one\n- two\n\n", "after list\n\n"],
        vec!["```rust\nlet x = 1;\n```\n\n", "after code\n\n"],
        vec!["> quote\n\n", "after quote\n\n"],
        vec!["math \\(x+1\\)\n\n", "more math \\[y\\]\n\n"],
        vec!["text\n\n", "unclosed `tick here\n\n", "still open\n\n"],
        vec!["# heading\n\n", "body\n\n"],
        vec!["| a | b |\n| - | - |\n| 1 | 2 |\n\n", "after table\n\n"],
        vec!["- outer\n  - inner\n\n", "after nested list\n\n"],
    ];
    for chunks in samples {
        let theme = Theme::dark();
        let options = MarkdownRenderOptions::new(40);
        let mut renderer = StreamingMarkdownRenderer::new(theme, options);
        let mut text = String::new();
        for chunk in chunks {
            text.push_str(chunk);
            let incremental = renderer.render(&text);
            let full = render_markdown_document(&text, theme, options);
            assert_eq!(incremental, full, "text = {text:?}");
        }
    }
}

#[test]
fn streaming_markdown_advances_checkpoint_for_plain_blocks() {
    let theme = Theme::dark();
    let options = MarkdownRenderOptions::new(40);
    let mut renderer = StreamingMarkdownRenderer::new(theme, options);
    let mut text = String::new();
    for chunk in ["one\n\n", "two\n\n", "three\n\n"] {
        text.push_str(chunk);
        renderer.render(&text);
    }
    assert!(renderer.checkpoint() > 0, "checkpoint stayed at zero");
}

#[test]
fn streaming_markdown_holds_checkpoint_for_open_code_span_and_definition() {
    let theme = Theme::dark();
    let options = MarkdownRenderOptions::new(40);

    let mut renderer = StreamingMarkdownRenderer::new(theme, options);
    renderer.render("line with `open tick\n\n");
    assert_eq!(
        renderer.checkpoint(),
        0,
        "an open code span must not finalize a block"
    );

    let mut renderer = StreamingMarkdownRenderer::new(theme, options);
    renderer.render("see [alpha]\n\n");
    assert!(renderer.checkpoint() > 0);
    renderer.render("see [alpha]\n\n[alpha]: https://example.test\n\n");
    assert_eq!(
        renderer.checkpoint(),
        0,
        "a reference definition must reset the checkpoint"
    );
}

#[test]
fn squeezed_columns_still_keep_one_grapheme() {
    let markdown = "| 一 | 二 | 三 | 四 | 五 | 六 | 七 | 八 |\n| --- | --- | --- | --- | --- | --- | --- | --- |\n| 甲 | 乙 | 丙 | 丁 | 戊 | 己 | 庚 | 辛 |";
    let document =
        render_markdown_document(markdown, Theme::dark(), MarkdownRenderOptions::new(30));
    assert!(document.validate(), "{document:?}");
    let text = document_text(&document);
    for glyph in [
        "一", "二", "三", "四", "五", "六", "七", "八", "甲", "乙", "丙", "丁", "戊", "己", "庚",
        "辛",
    ] {
        assert!(text.contains(glyph), "missing {glyph} in {text}");
    }
}
