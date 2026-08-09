//! Pure Mermaid ownership, rendering, and provenance tests.

#[cfg(test)]
mod tests {
    use super::super::MermaidRender;
    use super::super::{MermaidSourceSpan, flowchart, flowchart_ir as ir, flowchart_parser};

    fn source_slice(source: &str, start: usize, end: usize) -> String {
        source.chars().skip(start).take(end - start).collect()
    }

    fn assert_spans_exact(source: &str, rendered: &MermaidRender) {
        for line in &rendered.lines {
            for span in line {
                if let Some(MermaidSourceSpan { start, end }) = span.source {
                    assert_eq!(source_slice(source, start, end), span.text);
                }
            }
        }
    }

    fn rendered_text(source: &str, width: usize) -> String {
        flowchart::render(source, width)
            .unwrap()
            .into_iter()
            .map(|line| line.into_iter().map(|span| span.text).collect::<String>())
            .collect::<Vec<_>>()
            .join("\n")
    }

    #[test]
    fn mermaid_diamond_renders_fork_and_merge_without_crossing_boxes() {
        let source = concat!(
            "graph TD\n",
            "A[开始] -->|是| B[分支1]\n",
            "A -->|否| C[分支2]\n",
            "B --> D[汇合]\n",
            "C --> D[汇合]\n",
        );
        let text = rendered_text(source, 40);
        for label in ["开始", "分支1", "分支2", "汇合", "是", "否"] {
            assert!(text.contains(label), "missing {label}: {text}");
        }
        assert!(
            text.lines()
                .any(|line| line.contains("┌") && line.contains("┴") && line.contains("┐")),
            "{text}"
        );
        assert!(
            text.lines()
                .any(|line| line.contains("└") && line.contains("┬") && line.contains("┘")),
            "{text}"
        );
    }

    #[test]
    fn mermaid_crosslayer_edge_routes_around_boxes() {
        let source = concat!(
            "graph TD\n",
            "A[是否有数据] -->|否| E[分析]\n",
            "A -->|是| B[接入数据源]\n",
            "B --> C[清洗]\n",
            "C --> E\n",
            "E --> F[可视化]\n",
        );
        let text = rendered_text(source, 40);
        for label in [
            "是否有数据",
            "接入数据源",
            "清洗",
            "分析",
            "可视化",
            "是",
            "否",
        ] {
            assert!(text.contains(label), "missing {label}:\n{text}");
        }
        assert!(
            text.lines()
                .any(|line| line.contains('├') && line.contains('┘')),
            "{text}"
        );
    }

    #[test]
    fn mermaid_branch_and_diamond_layout_render() {
        let source = concat!(
            "graph TD\n",
            "A[开始] --> B{需要处理?}\n",
            "B -->|是| C[处理]\n",
            "B -->|否| D[跳过]\n",
        );
        let text = rendered_text(source, 40);
        for label in [
            "开始",
            "需要处理?",
            "处理",
            "跳过",
            "是",
            "否",
            "│ 开始 │",
            "│ 处理 │",
            "│ 跳过 │",
        ] {
            assert!(text.contains(label), "missing {label}: {text}");
        }
    }

    #[test]
    fn mermaid_parser_accepts_edge_text_and_pipe_labels() {
        let cases = [
            ("graph RL\nA -- text --> B\n", "text"),
            ("graph RL\nC ==>|thick| D\n", "thick"),
            ("graph RL\nE -. dashed .-> F\n", "dashed"),
            ("graph RL\nG -- x --> H\n", "x"),
        ];
        for (source, expected) in cases {
            let graph = flowchart_parser::parse(source).unwrap();
            assert_eq!(graph.edges.len(), 1);
            assert_eq!(
                graph.edges[0]
                    .label
                    .as_ref()
                    .map(|label| label.text.as_str()),
                Some(expected)
            );
        }
        let source = concat!(
            "graph RL\n",
            "A -- text --> B\n",
            "C ==>|thick| D\n",
            "E -. dashed .-> F\n"
        );
        let graph = flowchart_parser::parse(source).unwrap();
        assert_eq!(graph.edges.len(), 3);
        assert_eq!(graph.edges[1].style, ir::MermaidEdgeStyle::Thick);
        assert!(graph.edges[1].arrow);
        assert_eq!(graph.edges[2].style, ir::MermaidEdgeStyle::Dashed);
        assert!(graph.edges[2].arrow);
    }

    #[test]
    fn multi_layer_crossing_threshold_preserves_linear_fallback() {
        let source = concat!(
            "graph TD\n",
            "A --> F\n",
            "B --> E\n",
            "C --> D\n",
            "D --> G\n",
            "E --> G\n",
            "F --> G\n",
            "G --> H\n",
        );
        let text = rendered_text(source, 80);
        assert!(!text.contains('╭'), "expected linear fallback:\n{text}");
        assert!(text.lines().all(|line| line.starts_with("↓ ")), "{text}");
    }

    #[test]
    fn flowchart_facade_spans_are_exact_for_td_canvas_and_lr_linear_output() {
        for source in [
            concat!("graph TD\n", "A[\"开始🙂\"] -->|通过 ✅| B[\"结束\"]\n"),
            concat!(
                "graph LR\n",
                "A[\"开始🙂\"] -->|通过 ✅| B[\"结束\"]\n",
                "I[孤立节点]\n"
            ),
        ] {
            let rendered = super::super::render(source, 80).unwrap();
            assert_spans_exact(source, &rendered);
        }
    }

    #[test]
    fn parser_rejects_malformed_subgraph_boundaries_and_asymmetric_quotes() {
        for source in [
            "graph TD\nsubgraph\nA --> B\n",
            "graph TD\nend\nA --> B\n",
            "graph TD\nsubgraph group\nA --> B\nend\nend\n",
            "graph TD\nA[\"unmatched] --> B\n",
            "graph TD\nA[unmatched\"] --> B\n",
        ] {
            assert!(
                flowchart_parser::parse(source).is_none(),
                "accepted {source:?}"
            );
        }
    }

    #[test]
    fn connector_tokens_are_longest_first() {
        let graph = flowchart_parser::parse("graph LR\nA ---> B\n").unwrap();
        assert_eq!(graph.edges.len(), 1);
        assert!(graph.edges[0].arrow);
        assert_eq!(graph.edges[0].style, ir::MermaidEdgeStyle::Solid);
    }

    #[test]
    fn parser_rejects_unsupported_extended_labeled_connectors() {
        for source in [
            "graph LR\nA -- label ----> B\n",
            "graph LR\nA == label ===> B\n",
            "graph LR\nA -.. label .-> B\n",
            "graph LR\nA -. label ..-> B\n",
        ] {
            assert!(
                flowchart_parser::parse(source).is_none(),
                "accepted {source:?}"
            );
        }
    }

    #[test]
    fn parser_expands_chained_links_and_preserves_connector_metadata() {
        let source = "graph LR\nA -- text --> B -. dashed .-> C ==>|thick| D\n";
        let graph = flowchart_parser::parse(source).unwrap();
        assert_eq!(graph.edges.len(), 3);
        assert_eq!(
            (graph.edges[0].from.as_str(), graph.edges[0].to.as_str()),
            ("A", "B")
        );
        assert_eq!(
            (graph.edges[1].from.as_str(), graph.edges[1].to.as_str()),
            ("B", "C")
        );
        assert_eq!(
            (graph.edges[2].from.as_str(), graph.edges[2].to.as_str()),
            ("C", "D")
        );
        assert_eq!(graph.edges[0].style, ir::MermaidEdgeStyle::Solid);
        assert_eq!(graph.edges[1].style, ir::MermaidEdgeStyle::Dashed);
        assert_eq!(graph.edges[2].style, ir::MermaidEdgeStyle::Thick);
        assert!(graph.edges.iter().all(|edge| edge.arrow));
        assert_eq!(graph.edges[0].label.as_ref().unwrap().text, "text");
        assert_eq!(graph.edges[1].label.as_ref().unwrap().text, "dashed");
        assert_eq!(graph.edges[2].label.as_ref().unwrap().text, "thick");
    }

    #[test]
    fn parser_expands_endpoint_groups_cartesian_and_mixed() {
        let graph = flowchart_parser::parse("graph LR\na --> b & c --> d\n").unwrap();
        assert_eq!(graph.edges.len(), 4);
        assert_eq!(
            graph
                .edges
                .iter()
                .map(|edge| (edge.from.as_str(), edge.to.as_str()))
                .collect::<Vec<_>>(),
            vec![("a", "b"), ("a", "c"), ("b", "d"), ("c", "d")]
        );

        let graph = flowchart_parser::parse("graph LR\nA & B --> C & D\n").unwrap();
        assert_eq!(graph.edges.len(), 4);
        assert_eq!(
            graph
                .edges
                .iter()
                .map(|edge| (edge.from.as_str(), edge.to.as_str()))
                .collect::<Vec<_>>(),
            vec![("A", "C"), ("A", "D"), ("B", "C"), ("B", "D")]
        );
        assert!(graph.edges.iter().all(|edge| edge.label.is_none()));
    }

    #[test]
    fn parser_applies_each_connector_label_to_every_expanded_edge() {
        let graph = flowchart_parser::parse("graph LR\nA & B -- link --> C & D\n").unwrap();
        assert_eq!(graph.edges.len(), 4);
        assert!(
            graph.edges.iter().all(|edge| {
                edge.label.as_ref().map(|label| label.text.as_str()) == Some("link")
            })
        );
    }

    #[test]
    fn parser_last_explicit_definition_wins_and_bare_reference_preserves_it() {
        let source = concat!(
            "graph LR\n",
            "A[first] --> B\n",
            "A[second] --> C\n",
            "A --> D\n",
        );
        let graph = flowchart_parser::parse(source).unwrap();
        let node = graph.nodes.get("A").unwrap();
        assert_eq!(node.label, "second");
        let second_start = source
            .match_indices("second")
            .next()
            .map(|(index, _)| source[..index].chars().count())
            .unwrap();
        assert_eq!(node.start, second_start);
        assert_eq!(node.end, node.start + "second".chars().count());
        assert_eq!(node.shape, ir::MermaidShape::Rectangle);
    }

    #[test]
    fn parser_later_explicit_definition_replaces_bare_default_inside_groups_and_chains() {
        let source = concat!(
            "graph LR\n",
            "A & B --> C\n",
            "A[defined] & B --> D --> E\n",
        );
        let graph = flowchart_parser::parse(source).unwrap();
        assert_eq!(graph.nodes.get("A").unwrap().label, "defined");
        assert_eq!(graph.edges.len(), 5);
        assert!(
            graph
                .edges
                .iter()
                .any(|edge| edge.from == "D" && edge.to == "E")
        );
    }

    #[test]
    fn facade_uses_the_winning_definition_source_span() {
        let source = concat!(
            "graph LR\n",
            "A[first🙂] --> B\n",
            "A[second🙂] --> C\n",
            "A --> D\n",
        );
        let rendered = super::super::render(source, 120).unwrap();
        assert_spans_exact(source, &rendered);
        let a_spans = rendered
            .lines
            .iter()
            .flatten()
            .filter(|span| span.text == "second🙂")
            .collect::<Vec<_>>();
        assert!(!a_spans.is_empty());
        let start = source
            .match_indices("second🙂")
            .next()
            .map(|(index, _)| source[..index].chars().count())
            .expect("winning definition");
        assert!(a_spans.iter().all(|span| {
            span.source
                == Some(super::super::MermaidSourceSpan::new(
                    start,
                    start + "second🙂".chars().count(),
                ))
        }));
    }

    #[test]
    fn parser_statement_failure_does_not_commit_partial_group() {
        assert!(flowchart_parser::parse("graph LR\nA & --> B\n").is_none());
        assert!(flowchart_parser::parse("graph LR\nA --> B &\n").is_none());
        assert!(flowchart_parser::parse("graph LR\nA --> B -->\n").is_none());
    }

    #[test]
    fn parser_rejects_expansion_over_limit() {
        let left = (0..9)
            .map(|i| format!("A{i}"))
            .collect::<Vec<_>>()
            .join(" & ");
        let right = (0..8)
            .map(|i| format!("B{i}"))
            .collect::<Vec<_>>()
            .join(" & ");
        let source = format!("graph LR\n{left} --> {right}\n");
        assert!(flowchart_parser::parse(&source).is_none());
    }

    #[test]
    fn parser_limits_compact_expansion_without_rejecting_explicit_edges() {
        let edges = (0..65)
            .map(|i| format!("A{i} --> B{i}"))
            .collect::<Vec<_>>()
            .join("\n");
        let source = format!("graph LR\n{edges}\n");
        assert_eq!(flowchart_parser::parse(&source).unwrap().edges.len(), 65);
    }

    #[test]
    fn flowchart_facade_spans_are_exact_for_chains_and_groups() {
        let source = "graph LR\nA[开始🙂] -- 通过 ✅ --> B & C[结束] --> D\n";
        let rendered = super::super::render(source, 120).unwrap();
        assert_spans_exact(source, &rendered);
        for text in ["开始🙂", "结束", "通过 ✅"] {
            assert!(
                rendered
                    .lines
                    .iter()
                    .flatten()
                    .any(|span| span.text == text),
                "missing {text}"
            );
        }
    }

    #[test]
    fn sequence_has_separate_parser_and_renderer_ownership() {
        let source = concat!(
            "sequenceDiagram\n",
            "participant A as 开始\n",
            "participant B as 结束\n",
            "A->>B: 消息🙂\n"
        );
        let rendered = super::super::render(source, 80).unwrap();
        assert_spans_exact(source, &rendered);
        let text = rendered
            .lines
            .into_iter()
            .flatten()
            .map(|span| span.text)
            .collect::<String>();
        assert!(text.contains("开始"));
        assert!(text.contains("结束"));
        assert!(text.contains("消息🙂"));
    }

    #[test]
    fn sequence_supports_structured_blocks_and_nested_unicode_provenance() {
        let source = concat!(
            "sequenceDiagram\n",
            "participant A as 开始🙂\n",
            "participant B as 结束✅\n",
            "loop 重复🔁\n",
            "A->>B: 外层🙂\n",
            "alt 条件一✅\n",
            "A-->>B: 真分支\n",
            "else 条件二🌀\n",
            "opt 可选🌟\n",
            "A->>B: 嵌套\n",
            "end\n",
            "end\n",
            "end\n",
        );
        let rendered = super::super::render(source, 80).unwrap();
        assert_spans_exact(source, &rendered);
        let text = rendered
            .lines
            .iter()
            .flatten()
            .map(|span| span.text.as_str())
            .collect::<String>();
        for label in [
            "loop",
            "重复🔁",
            "alt",
            "条件一✅",
            "else",
            "条件二🌀",
            "opt",
            "可选🌟",
        ] {
            assert!(text.contains(label), "missing {label}: {text}");
        }
    }

    #[test]
    fn sequence_structures_fail_closed_and_honor_width() {
        let base = "sequenceDiagram\nparticipant A as A\nparticipant B as B\nA->>B: m\n";
        for suffix in [
            "end\n",
            "else\nA->>B: m\nend\n",
            "loop\nA->>B: m\nend\n",
            "alt x\nA->>B: m\nelse\nelse y\nA->>B: n\nend\n",
            "opt x\nend\n",
            "loop x\nA->>B: m\n",
            "when x\nA->>B: m\nend\n",
        ] {
            let source = format!("{base}{suffix}");
            assert!(
                super::super::render(&source, 80).is_none(),
                "accepted {source:?}"
            );
        }
        let bare_else = concat!(
            "sequenceDiagram\n",
            "participant A as A\n",
            "participant B as B\n",
            "alt primary\n",
            "A->>B: first\n",
            "else\n",
            "A->>B: second\n",
            "end\n",
        );
        assert!(super::super::render(bare_else, 80).is_some());

        let duplicate_else = concat!(
            "sequenceDiagram\n",
            "participant A as A\n",
            "participant B as B\n",
            "alt primary\n",
            "A->>B: first\n",
            "else\n",
            "A->>B: second\n",
            "else fallback\n",
            "A->>B: third\n",
            "end\n",
        );
        assert!(super::super::render(duplicate_else, 80).is_none());
        assert!(super::super::render(base, 1).is_none());
    }
}
