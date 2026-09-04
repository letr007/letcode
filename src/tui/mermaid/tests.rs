//! Pure Mermaid ownership, rendering, and provenance tests.

#[cfg(test)]
mod tests {
    use super::super::flowchart::{mermaid_crossings, mermaid_layers};
    use super::super::{
        MermaidRender, MermaidSourceSpan, flowchart, flowchart_ir as ir, flowchart_parser,
        render_cache, render_uncached,
    };

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

    fn assert_spans_valid(source: &str, rendered: &MermaidRender) {
        for line in &rendered.lines {
            for span in line {
                if let Some(MermaidSourceSpan { start, end }) = span.source {
                    let original = source_slice(source, start, end);
                    assert!(!original.is_empty());
                    assert!(span.atomic || original == span.text, "{span:?}");
                    assert!(!span.text.contains(['\n', '\r']), "{span:?}");
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

    fn facade_text(source: &str, width: usize) -> String {
        super::super::render(source, width)
            .unwrap()
            .lines
            .into_iter()
            .map(|line| line.into_iter().map(|span| span.text).collect::<String>())
            .collect::<Vec<_>>()
            .join("\n")
    }

    #[test]
    fn render_cache_reuses_completed_diagrams_by_source_and_width() {
        let source = "flowchart TD\nA[Start] --> B[End]\n";
        let expected = render_uncached(source, 80).expect("render diagram");
        render_cache()
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .insert(source, 80, expected.clone());

        assert_eq!(
            render_cache()
                .lock()
                .unwrap_or_else(|error| error.into_inner())
                .get(source, 80),
            Some(expected)
        );
        assert!(
            render_cache()
                .lock()
                .unwrap_or_else(|error| error.into_inner())
                .get(source, 79)
                .is_none()
        );
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
    fn flowchart_horizontal_grouped_fanout_uses_one_corridor() {
        for (header, arrow) in [("graph LR", '▶'), ("graph RL", '◀')] {
            let source = format!(
                "{header}\nA[a] --> B[b]\nA --> C[c]\nA --> D[d]\nA --> E[e]\nA --> F[f]\nA --> G[g]\nA --> H[h]\n"
            );
            let text = rendered_text(&source, 40);
            assert!(text.contains('╭'), "expected canvas:\n{text}");
            assert_eq!(
                text.matches(arrow).count(),
                7,
                "missing routed edges:\n{text}"
            );
        }
    }

    #[test]
    fn flowchart_br_labels_render_as_dynamic_multiline_nodes() {
        for separator in ["<br>", "<br/>", "<br />"] {
            let source = format!(
                "flowchart LR\nA[letcode{separator}binary] -->|build{separator}ready| B[composition{separator}root]\n"
            );
            let rendered = super::super::render(&source, 80).expect("multiline flowchart");
            assert_spans_valid(&source, &rendered);
            let rows = rendered
                .lines
                .iter()
                .map(|line| {
                    line.iter()
                        .map(|span| span.text.as_str())
                        .collect::<String>()
                })
                .collect::<Vec<_>>();
            let text = rows.join("\n");
            assert!(!text.contains("<br"), "{text}");
            assert!(
                text.contains("build / ready"),
                "edge label did not retain both lines:\n{text}"
            );
            let letcode_row = rows.iter().position(|row| row.contains("letcode")).unwrap();
            let binary_row = rows.iter().position(|row| row.contains("binary")).unwrap();
            let composition_row = rows
                .iter()
                .position(|row| row.contains("composition"))
                .unwrap();
            let root_row = rows.iter().position(|row| row.contains("root")).unwrap();
            assert_eq!(
                binary_row,
                letcode_row + 1,
                "first label was not split into adjacent rows:\n{text}"
            );
            assert_eq!(
                root_row,
                composition_row + 1,
                "second label was not split into adjacent rows:\n{text}"
            );
            for row in [
                &rows[letcode_row],
                &rows[binary_row],
                &rows[composition_row],
                &rows[root_row],
            ] {
                assert!(
                    row.contains('│'),
                    "multiline label escaped its node box: {row}"
                );
            }
        }
    }

    #[test]
    fn flowchart_math_labels_transform_atomically_and_preserve_wrapper_spans() {
        let source = concat!(
            "flowchart TD\n",
            "A[$$E=mc^2$$] -->|integral $$\\int_0^1 x^2 dx$$| B[Done]\n",
        );
        let graph = flowchart_parser::parse(source).unwrap();
        let node = graph.nodes.get("A").unwrap();
        assert!(node.atomic);
        assert_eq!(source_slice(source, node.start, node.end), "$$E=mc^2$$");
        assert!(!node.label.contains('$'));
        let label = graph.edges[0].label.as_ref().unwrap();
        assert!(label.atomic);
        assert_eq!(
            source_slice(source, label.start, label.end),
            "integral $$\\int_0^1 x^2 dx$$"
        );
        assert!(!label.text.contains('$'));

        let rendered = super::super::render(source, 80).unwrap();
        let text = rendered
            .lines
            .iter()
            .map(|line| {
                line.iter()
                    .map(|span| span.text.as_str())
                    .collect::<String>()
            })
            .collect::<Vec<_>>()
            .join("\n");
        for span in rendered
            .lines
            .iter()
            .flatten()
            .filter(|span| span.text == node.label || span.text == label.text)
        {
            assert!(span.atomic);
            let span_source =
                source_slice(source, span.source.unwrap().start, span.source.unwrap().end);
            assert!(span_source == "$$E=mc^2$$" || span_source == "integral $$\\int_0^1 x^2 dx$$");
        }
        let ordinary = rendered
            .lines
            .iter()
            .flatten()
            .find(|span| span.text == "Done")
            .unwrap();
        assert!(!ordinary.atomic);
        assert!(text.contains(&node.label));
        assert!(text.contains(&label.text));

        let too_wide = concat!(
            "flowchart TD\n",
            "A[a] -->|$$abcdefghijklmnopqrstuvwxyz$$| B[b]\n",
        );
        assert!(super::super::render(too_wide, 20).is_none());
    }

    #[test]
    fn flowchart_math_labels_fail_closed_for_malformed_or_multiline_latex() {
        for source in [
            "flowchart TD\nA[$$E=mc^2] --> B[Done]\n",
            "flowchart TD\nA[foo$$E=mc^2] --> B[Done]\n",
            r#"flowchart TD
A[$$\begin{aligned}x\\ny\end{aligned}$$] --> B[Done]
"#,
            r#"flowchart TD
A[$$\not_a_supported_command$$] --> B[Done]
"#,
        ] {
            assert!(
                flowchart_parser::parse(source).is_none(),
                "accepted {source:?}"
            );
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
    fn flowchart_crossings_are_counted_per_corridor() {
        let source = concat!(
            "graph TD\n",
            "A --> D\n",
            "B --> C\n",
            "C --> F\n",
            "D --> E\n",
        );
        let graph = flowchart_parser::parse(source).unwrap();
        let layers = mermaid_layers(&graph).unwrap();
        assert_eq!(mermaid_crossings(&graph, &layers), 2);
    }

    #[test]
    fn multi_layer_crossings_fall_back_when_routes_cannot_be_drawn_unambiguously() {
        let source = concat!(
            "graph TD\n",
            "A --> J\n",
            "B --> I\n",
            "C --> H\n",
            "D --> G\n",
            "E --> F\n",
            "F --> K\n",
            "G --> K\n",
            "H --> K\n",
            "I --> K\n",
            "J --> K\n",
            "K --> L\n",
        );
        let text = rendered_text(source, 100);
        assert!(!text.contains('╭'), "expected linear fallback:\n{text}");
        assert!(text.lines().all(|line| line.starts_with("↓ ")), "{text}");
    }

    #[test]
    fn flowchart_lr_branch_labels_use_separate_tracks_and_preserve_spans() {
        for header in ["flowchart LR", "flowchart RL"] {
            let source = format!(
                "{header}\nA[开始] -->|是| B[处理]\nA -->|否| C[跳过]\nB --> D[汇合]\nC --> D\n"
            );
            let rendered = super::super::render(&source, 120).unwrap();
            assert_spans_exact(&source, &rendered);
            let text = rendered
                .lines
                .iter()
                .map(|line| {
                    line.iter()
                        .map(|span| span.text.as_str())
                        .collect::<String>()
                })
                .collect::<Vec<_>>();
            assert!(
                text.iter().any(|line| line.contains('╭')),
                "expected canvas for {header}: {text:?}"
            );
            assert!(text.iter().any(|line| line.contains("是")));
            assert!(text.iter().any(|line| line.contains("否")));
            assert!(
                text.iter()
                    .filter(|line| line.contains("是") || line.contains("否"))
                    .count()
                    >= 2,
                "{header}: {text:?}"
            );
            assert!(
                text.iter().all(|line| {
                    let labels = ["是", "否"]
                        .iter()
                        .filter(|label| line.contains(**label))
                        .count();
                    labels < 2
                }),
                "{header}: {text:?}"
            );
        }
    }

    #[test]
    fn flowchart_lr_service_fanout_avoids_ambiguous_crossings() {
        for header in ["flowchart LR", "flowchart RL"] {
            let source = format!(
                "{header}\nB[浏览器] --> G[API 网关]\nM[移动端] --> G\nG --> A[身份认证]\nG --> O[订单服务]\nA --> C[缓存]\nO --> D[主数据库]\nO --> I[库存服务]\nO --> P[支付服务]\nO --> Q[消息队列]\n"
            );
            let rendered = super::super::render(&source, 160).unwrap();
            assert_spans_exact(&source, &rendered);
            let text = facade_text(&source, 160);
            for label in [
                "浏览器",
                "移动端",
                "API 网关",
                "身份认证",
                "订单服务",
                "缓存",
                "主数据库",
                "库存服务",
                "支付服务",
                "消息队列",
            ] {
                assert!(text.contains(label), "missing {label}:\n{text}");
            }
            assert!(text.contains('╭'), "expected canvas:\n{text}");
        }
    }

    #[test]
    fn flowchart_lr_mixed_grouped_corridors_stay_collision_free() {
        for header in ["flowchart LR", "flowchart RL"] {
            let source =
                format!("{header}\nS[入口] --> A[Alpha]\nS --> B[Beta]\nA --> T[出口]\nB --> T\n");
            let rendered = super::super::render(&source, 120).unwrap();
            assert_spans_exact(&source, &rendered);
            let text = facade_text(&source, 120);
            assert!(text.contains('╭'), "expected canvas:\n{text}");
        }
    }

    #[test]
    fn flowchart_lr_fanin_uses_shared_target_beam_with_same_row_member() {
        for (header, arrow) in [("flowchart LR", '▶'), ("flowchart RL", '◀')] {
            let source = format!("{header}\nA[Alpha] --> T[出口]\nB[Beta] --> T\nC[Gamma] --> T\n");
            let rendered = super::super::render(&source, 100).unwrap();
            assert_spans_exact(&source, &rendered);
            let text = facade_text(&source, 100);
            assert!(text.contains('╭'), "expected canvas:\n{text}");
            assert!(text.contains('┼'), "missing target-beam junction:\n{text}");
            assert!(text.contains(arrow), "missing target arrow:\n{text}");
        }
    }

    #[test]
    fn flowchart_lr_edge_in_source_and_target_groups_falls_back_atomically() {
        for header in ["flowchart LR", "flowchart RL"] {
            let source =
                format!("{header}\nS[Source] --> A[Alpha]\nS --> B[Beta]\nC[Client] --> B\n");
            let rendered = super::super::render(&source, 100).unwrap();
            assert_spans_exact(&source, &rendered);
            let text = facade_text(&source, 100);
            assert!(!text.contains('╭'), "expected linear fallback:\n{text}");
            assert_eq!(
                text.lines().count(),
                3,
                "expected complete fallback:\n{text}"
            );
            assert!(text.lines().all(|line| line.contains('▶')), "{text}");
        }
    }

    #[test]
    fn flowchart_lr_renders_boxed_nodes_and_horizontal_routes() {
        let source = concat!(
            "flowchart LR\n",
            "A[接收请求] --> B{校验通过?}\n",
            "B -->|是| C[处理业务]\n",
            "B -->|否| D[返回错误]\n",
            "C --> E[返回结果]\n",
        );
        let text = facade_text(source, 100);
        for label in ["接收请求", "校验通过?", "处理业务", "返回错误", "返回结果"]
        {
            assert!(text.contains(label), "missing {label}:\n{text}");
        }
        assert!(text.lines().count() >= 4, "{text}");
        assert!(text.contains('▶'), "{text}");
    }

    #[test]
    fn flowchart_td_uses_filled_triangle_arrows() {
        let source = concat!("flowchart TD\n", "A[开始] --> B[结束]\n");
        let text = facade_text(source, 40);
        assert!(text.contains('▼'), "{text}");
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
    fn parser_rejects_unknown_subgraph_direction() {
        let source = "flowchart TD\nsubgraph A\ndirection SIDEWAYS\nA --> B\nend\n";
        assert!(flowchart_parser::parse(source).is_none());
    }

    #[test]
    fn parser_accepts_nested_subgraphs_local_directions_cycles_and_hyphenated_ids() {
        let source = concat!(
            "flowchart TD\n",
            "subgraph TOP[Platform]\n",
            "direction LR\n",
            "subgraph API[API Layer]\n",
            "direction TB\n",
            "api-gateway[Gateway] --> worker-1[Worker]\n",
            "end\n",
            "worker-1 --> api-gateway\n",
            "end\n",
        );
        let graph = flowchart_parser::parse(source).expect("supported flowchart syntax");
        assert!(graph.nodes.contains_key("api-gateway"));
        assert!(graph.nodes.contains_key("worker-1"));
        assert_eq!(graph.edges.len(), 2);

        let rendered = super::super::render(source, 80).expect("cycle uses linear fallback");
        assert_spans_exact(source, &rendered);
        let text = rendered
            .lines
            .iter()
            .map(|line| {
                line.iter()
                    .map(|span| span.text.as_str())
                    .collect::<String>()
            })
            .collect::<Vec<_>>()
            .join("\n");
        assert!(text.contains("Gateway"), "{text}");
        assert!(text.contains("Worker"), "{text}");
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
    fn mindmap_and_timeline_render_with_exact_unicode_provenance() {
        let mindmap = concat!(
            "mindmap\n",
            "%% supported comment\n",
            "根🙂\n",
            "  branch[标签✅]\n",
            "    leaf((详情🚀))\n",
            "  节点[方形]\n",
            "  cloud)云朵(\n",
            "  bang))重点((\n",
            "  other(\"圆形\")\n",
        );
        let rendered = super::super::render(mindmap, 60).unwrap();
        assert_spans_exact(mindmap, &rendered);
        let text = rendered
            .lines
            .iter()
            .map(|line| {
                line.iter()
                    .map(|span| span.text.as_str())
                    .collect::<String>()
            })
            .collect::<Vec<_>>()
            .join("\n");
        assert!(text.contains("根🙂"), "{text}");
        assert!(text.contains("标签✅"), "{text}");
        assert!(text.contains("详情🚀"), "{text}");
        assert!(text.contains("方形"), "{text}");
        assert!(text.contains("云朵"), "{text}");
        assert!(text.contains("重点"), "{text}");
        assert!(
            text.lines()
                .any(|line| line.contains("根🙂") && line.contains('┼')),
            "{text}"
        );
        assert!(
            text.lines()
                .any(|line| line.contains("┌──") && line.contains("标签✅")),
            "{text}"
        );
        assert!(
            text.lines()
                .any(|line| line.contains("└──") && line.contains("圆形")),
            "{text}"
        );

        let timeline = concat!(
            "timeline LR\n",
            "    %% supported comment\n",
            "    title 发布计划🙂\n",
            "\n",
            "    section 客户端✅\n",
            "    2004: 开始🚀 : 完成于 10:30\n",
            "         : 验收 : 发布\n",
        );
        let rendered = super::super::render(timeline, 60).unwrap();
        assert_spans_exact(timeline, &rendered);
        let text = rendered
            .lines
            .iter()
            .map(|line| {
                line.iter()
                    .map(|span| span.text.as_str())
                    .collect::<String>()
            })
            .collect::<Vec<_>>()
            .join("\n");
        assert!(text.contains("发布计划🙂"), "{text}");
        assert!(text.contains("10:30"), "{text}");
        assert!(text.contains("验收"), "{text}");
        assert!(text.contains("发布"), "{text}");
        assert!(
            text.lines()
                .any(|line| line.contains('┌') && line.contains('┐')),
            "{text}"
        );
        assert!(
            text.lines()
                .any(|line| line.chars().filter(|ch| *ch == '┬').count() >= 1),
            "{text}"
        );
        assert!(
            text.lines()
                .any(|line| line.chars().filter(|ch| *ch == '┴').count() >= 1),
            "{text}"
        );
        assert!(
            text.lines()
                .any(|line| line.contains("2004") && !line.contains("开始🚀")),
            "{text}"
        );

        let multi_period = concat!(
            "timeline\n",
            "    2024 : Alpha : Beta\n",
            "    2025 : Ship\n",
        );
        let rendered = super::super::render(multi_period, 50).unwrap();
        assert_spans_exact(multi_period, &rendered);
        let text = facade_text(multi_period, 50);
        assert!(
            text.lines().any(|line| line.matches('┬').count() == 2),
            "{text}"
        );
        assert!(
            text.lines().any(|line| line.matches('┴').count() == 2),
            "{text}"
        );
        assert!(
            text.lines()
                .any(|line| line.contains("2024") && line.contains("2025")),
            "{text}"
        );

        let trailing_colon = "timeline\n    2004 : event:\n";
        let rendered = super::super::render(trailing_colon, 40).unwrap();
        assert_spans_exact(trailing_colon, &rendered);
        assert!(facade_text(trailing_colon, 40).contains("event:"));
    }

    #[test]
    fn mindmap_and_timeline_fail_closed_for_unsupported_or_narrow_input() {
        for source in [
            "mindmap\nroot\n  child:::class\n",
            "mindmap\nroot\n  `markdown`\n",
            "mindmap\nroot\n  icon: fa fa-home\n",
            "mindmap\nroot\n  child[broken)\n",
            "timeline\n\ttitle invalid\n",
            "timeline\n    title invalid \n",
            "timeline\n    section phase\n    title late\n    2004 : event\n",
            "timeline\n    section Phase A : Event\n    2004 : event\n",
            "timeline\n    2004 : event :\n",
            "timeline\n    2004 : event <br> more\n",
            "timeline\n    2004:Event without separator whitespace\n",
            "timeline\n    2004 : event\n         :Continuation without whitespace\n",
            "timeline TD\n    2004 : unsupported direction\n",
        ] {
            assert!(
                super::super::render(source, 80).is_none(),
                "accepted {source:?}"
            );
        }
        assert!(super::super::render("mindmap\nVeryLongRoot\n", 4).is_none());
        assert!(super::super::render("timeline\n    2004 : This event is too wide\n", 8).is_none());
        for source in ["mindmapx\nroot\n", "timelineDiagram\n2026 : event\n"] {
            assert!(super::super::render(source, 80).is_none());
        }
    }

    #[test]
    fn common_diagram_families_render_with_exact_unicode_provenance() {
        let cases = [
            concat!(
                "classDiagram\n",
                "class 用户 {\n",
                "  +名称🙂: String\n",
                "}\n",
                "class 服务\n",
                "用户 --> 服务 : 调用🚀\n",
            ),
            concat!(
                "erDiagram\n",
                "用户 {\n",
                "  string 名称🙂\n",
                "}\n",
                "订单 {\n",
                "  int 编号✅\n",
                "}\n",
                "用户 ||--o{ 订单 : 创建🚀\n",
            ),
            concat!(
                "gantt\n",
                "title 发布计划🙂\n",
                "dateFormat YYYY-MM-DD\n",
                "section 客户端✅\n",
                "实现渲染🚀 : active, render, 2026-08-09, 3d\n",
            ),
            concat!(
                "stateDiagram-v2\n",
                "state \"等待🙂\" as Waiting\n",
                "state \"完成✅\" as Done\n",
                "[*] --> Waiting : 启动🚀\n",
                "Waiting --> Done : 完成\n",
                "Done --> [*]\n",
            ),
            concat!(
                "mindmap\n",
                "根🙂\n",
                "  branch[标签✅]\n",
                "    leaf((详情🚀))\n",
            ),
            concat!(
                "timeline\n",
                "    title 发布计划🙂\n",
                "    section 客户端✅\n",
                "    2004 : 开始🚀 : 完成\n",
                "         : 验收\n",
            ),
            concat!(
                "pie showData title 语言占比🙂\n",
                "\"Rust🚀\" : 60\n",
                "\"其他✅\" : 40\n",
            ),
        ];
        for source in cases {
            let rendered = super::super::render(source, 120).unwrap_or_else(|| panic!("{source}"));
            assert_spans_exact(source, &rendered);
        }
    }

    #[test]
    fn common_diagram_families_fail_closed_and_honor_width() {
        let malformed = [
            "classDiagram\nclass A {\n  +x\n",
            "classDiagram\nclass A\nA --> Missing\n",
            "erDiagram\nA {\n  string id\n}\nA ||--o{ B : owns\n",
            "erDiagram\nA {\n  string id\n}\nB {\n  string id\n}\nA broken B : owns\n",
            "gantt\ntitle x\nTask : active, id, someday, later\n",
            "gantt\nexcludes weekends\n",
            "stateDiagram\nstate \"Missing end as A\n",
            "stateDiagram\ndirection LR\nA --> B\n",
            "stateDiagram\nstate A {\nA --> B\n",
            "mindmap\nroot\n  child:::class\n",
            "timeline\n    section Phase A : Event\n    2004 : event\n",
            "journey\nsection Experience\nTask: 0: User\n",
            "gitGraph\nmerge missing\ncommit\n",
            "pie\ntitle unsupported\n",
            "pie\nTypeScript : 45\n",
            "pie\n\"TypeScript\" : 0\n",
            "pie\n\"TypeScript\" : 45\n\"TypeScript\" : 30\n",
        ];
        for source in malformed {
            assert!(
                super::super::render(source, 120).is_none(),
                "accepted {source:?}"
            );
        }

        let narrow = [
            "classDiagram\nclass VeryLongClassName\n",
            "erDiagram\nA {\n  string very_long_attribute\n}\nB {\n  int id\n}\nA ||--|| B : owns\n",
            "gantt\ntitle Very long release plan\n",
            "stateDiagram\nVeryLongStateName --> OtherState\n",
            "mindmap\nVeryLongRoot\n",
            "timeline\n    2004 : This event is too wide\n",
            "journey\nsection Experience\nA very long journey task: 5: User\n",
            "gitGraph\ncommit id: a-very-long-commit-label\n",
            "pie title This title is too long\n\"A\" : 1\n",
        ];
        for source in narrow {
            assert!(super::super::render(source, 4).is_none(), "fit {source:?}");
        }
    }

    #[test]
    fn common_diagram_families_reject_noncanonical_headers_and_resource_overflow() {
        for source in [
            " classDiagram\nclass A\n",
            "erDiagram \nA {\n  string id\n}\nB {\n  string id\n}\nA ||--|| B : owns\n",
            "\tgantt\ntitle release\n",
            "stateDiagram-v2  \nA --> B\n",
            " mindmap\nroot\n",
            "timeline \n2004 : event\n",
            " journey\nsection Experience\nTask: 5: User\n",
            "gitGraph \ncommit\n",
            " pie\n\"A\" : 1\n",
            "pieChart\n\"A\" : 1\n",
        ] {
            assert!(
                super::super::render(source, 120).is_none(),
                "accepted {source:?}"
            );
        }

        let too_many_lines = format!(
            "classDiagram\n{}",
            "%% comment\n".repeat(super::super::MAX_SOURCE_LINES)
        );
        assert!(super::super::render(&too_many_lines, 120).is_none());

        let too_many_chars = format!(
            "gantt\ntitle {}\n",
            "x".repeat(super::super::MAX_SOURCE_CHARS)
        );
        assert!(super::super::render(&too_many_chars, usize::MAX).is_none());
    }

    #[test]
    fn class_colon_members_and_state_composites_render() {
        let class = concat!(
            "classDiagram\n",
            "class User\n",
            "User : +name String\n",
            "class Service\n",
            "User ..> Service : uses\n",
        );
        let class_rendered = super::super::render(class, 80).unwrap();
        assert_spans_exact(class, &class_rendered);
        assert!(
            class_rendered
                .lines
                .iter()
                .flatten()
                .any(|span| span.text == "+name String")
        );

        let state = concat!(
            "stateDiagram\n",
            "state Parent {\n",
            "  state Child\n",
            "  [*] --> Child\n",
            "}\n",
        );
        let state_rendered = super::super::render(state, 80).unwrap();
        assert_spans_exact(state, &state_rendered);
        let text = state_rendered
            .lines
            .iter()
            .map(|line| {
                line.iter()
                    .map(|span| span.text.as_str())
                    .collect::<String>()
            })
            .collect::<Vec<_>>()
            .join("\n");
        assert!(text.contains("Parent"), "{text}");
        assert!(text.contains("Child"), "{text}");
    }

    #[test]
    fn sequence_renders_participant_lifelines_and_bidirectional_messages() {
        let source = concat!(
            "sequenceDiagram\n",
            "participant Client as Client\n",
            "participant Server as Server\n",
            "Client->>Server: request\n",
            "Server-->>Client: response\n",
        );
        let rendered = super::super::render(source, 80).unwrap();
        assert_spans_exact(source, &rendered);
        let text = facade_text(source, 80);
        let lines = text.lines().collect::<Vec<_>>();
        assert!(
            lines[0].contains("Client") && lines[0].contains("Server"),
            "{text}"
        );
        assert!(text.matches('│').count() >= 4, "{text}");
        assert!(
            text.lines()
                .any(|line| line.contains("request") && line.contains('▶')),
            "{text}"
        );
        assert!(
            text.lines()
                .any(|line| line.contains("response") && line.contains('◀')),
            "{text}"
        );
    }

    #[test]
    fn sequence_accepts_bare_unicode_participants_and_renders_login_flow() {
        let source = concat!(
            "sequenceDiagram\n",
            "participant 用户\n",
            "participant 前端\n",
            "participant 服务器\n",
            "participant 数据库\n",
            "用户->>前端: 点击登录\n",
            "前端->>服务器: 发送登录请求\n",
            "服务器->>数据库: 校验账号密码\n",
            "数据库-->>服务器: 返回结果\n",
            "服务器-->>前端: 返回 token\n",
            "前端-->>用户: 登录成功\n",
        );
        let rendered = super::super::render(source, 100).expect("unicode sequence diagram");
        assert_spans_exact(source, &rendered);
        let text = facade_text(source, 100);
        let rows = text.lines().collect::<Vec<_>>();
        let header = rows.first().unwrap();
        let participant_positions = ["用户", "前端", "服务器", "数据库"]
            .map(|participant| header.find(participant).unwrap());
        assert!(
            participant_positions
                .windows(2)
                .all(|pair| pair[0] < pair[1]),
            "participant order changed:\n{text}"
        );
        for (message, arrow) in [
            ("点击登录", '▶'),
            ("发送登录请求", '▶'),
            ("校验账号密码", '▶'),
            ("返回结果", '◀'),
            ("返回 token", '◀'),
            ("登录成功", '◀'),
        ] {
            assert!(
                rows.iter()
                    .any(|row| row.contains(message) && row.contains(arrow)),
                "message {message:?} lost its direction:\n{text}"
            );
        }
    }

    #[test]
    fn sequence_math_labels_render_atomically_with_full_source_spans() {
        let source = concat!(
            "sequenceDiagram\n",
            "participant A as $$\\alpha$$\n",
            "participant B as Beta\n",
            "A->>B: Solve: $$\\sqrt{2+2}$$\n",
            "alt $$x^2$$\n",
            "B-->>A: Answer: $$2$$\n",
            "else try $$x_1$$ again\n",
            "B-->>A: plain\n",
            "end\n",
        );
        let rendered = super::super::render(source, 80).unwrap();
        let transformed = rendered
            .lines
            .iter()
            .flatten()
            .filter(|span| {
                span.source.is_some_and(|range| {
                    source_slice(source, range.start, range.end).contains("$$")
                })
            })
            .collect::<Vec<_>>();
        assert!(!transformed.is_empty());
        assert!(transformed.iter().all(|span| span.atomic));
        for span in transformed {
            let range = span.source.unwrap();
            let original = source_slice(source, range.start, range.end);
            assert!(original.contains("$$"), "{original:?}");
            assert!(!span.text.contains('$'), "{:?}", span.text);
        }
        let text = facade_text(source, 80);
        assert!(text.contains('α'), "{text}");
        assert!(text.contains("Solve: √(2+2)"), "{text}");
        assert!(text.contains("x²"), "{text}");
        assert!(text.contains("Answer: 2"), "{text}");
        assert!(text.contains("try x₁ again"), "{text}");

        let linear_source = concat!(
            "sequenceDiagram\n",
            "participant A as A\n",
            "A->>A: plain\n",
        );
        let linear = super::super::render(linear_source, 80).unwrap();
        let plain = linear
            .lines
            .iter()
            .flatten()
            .find(|span| span.text == "plain")
            .unwrap();
        assert!(!plain.atomic);
        assert_eq!(
            source_slice(
                linear_source,
                plain.source.unwrap().start,
                plain.source.unwrap().end,
            ),
            "plain"
        );
    }

    #[test]
    fn sequence_math_labels_fail_closed_for_malformed_latex() {
        for source in [
            "sequenceDiagram\nparticipant A as $$\\alpha\nparticipant B as B\nA->>B: x\n",
            "sequenceDiagram\nparticipant A as A\nparticipant B as B\nA->>B: empty $$$$\n",
            "sequenceDiagram\nparticipant A as A\nparticipant B as B\nA->>B: $$\\not_a_supported_command$$\n",
            "sequenceDiagram\nparticipant A as A\nparticipant B as B\nA->>B: $$\\begin{aligned}x\\\\y\\end{aligned}$$\n",
        ] {
            assert!(
                super::super::render(source, 80).is_none(),
                "accepted {source:?}"
            );
        }
    }

    #[test]
    fn sequence_supports_actor_and_autonumber() {
        let source = concat!(
            "sequenceDiagram\n",
            "autonumber\n",
            "actor U as 用户\n",
            "participant S as 服务\n",
            "U->>S: 请求\n",
            "S-->>U: 响应\n",
        );
        let rendered = super::super::render(source, 80).unwrap();
        assert_spans_exact(source, &rendered);
        let text = facade_text(source, 80);
        assert!(text.contains("用户") && text.contains("服务"), "{text}");
        assert!(
            text.lines()
                .any(|line| line.contains("1 ") && line.contains("请求")),
            "{text}"
        );
        assert!(
            text.lines()
                .any(|line| line.contains("2 ") && line.contains("响应")),
            "{text}"
        );
    }

    #[test]
    fn sequence_supports_activation_shorthand_in_nested_checkout_flow() {
        let source = concat!(
            "sequenceDiagram\n",
            "autonumber\n",
            "actor U as 用户\n",
            "participant W as Web\n",
            "participant G as 网关\n",
            "participant O as 订单服务\n",
            "participant I as 库存服务\n",
            "participant P as 支付服务\n",
            "participant D as 数据库\n",
            "U->>W: 点击购买\n",
            "W->>+G: 提交订单\n",
            "G->>+O: 创建订单\n",
            "O->>+I: 检查库存\n",
            "alt 库存充足\n",
            "I-->>-O: 锁定库存\n",
            "O->>+P: 发起支付\n",
            "alt 支付成功\n",
            "P-->>-O: 支付成功\n",
            "O->>D: 保存已支付订单\n",
            "D-->>O: 保存成功\n",
            "O-->>-G: 返回订单\n",
            "G-->>-W: 订单创建成功\n",
            "W-->>U: 显示支付成功\n",
            "else 支付失败\n",
            "P-->>-O: 支付失败\n",
            "O->>I: 释放库存\n",
            "O-->>-G: 返回支付失败\n",
            "G-->>-W: 显示错误\n",
            "W-->>U: 支付失败\n",
            "end\n",
            "else 库存不足\n",
            "I-->>-O: 库存不足\n",
            "O-->>-G: 返回库存不足\n",
            "G-->>-W: 显示错误\n",
            "W-->>U: 商品库存不足\n",
            "end\n",
        );
        let rendered = super::super::render(source, 80).unwrap();
        assert_spans_exact(source, &rendered);
        let text = facade_text(source, 80);
        assert!(
            text.contains("提交订单") && text.contains("商品库存不足"),
            "{text}"
        );
        assert!(text.lines().any(|line| line.contains("20 ")), "{text}");
    }

    #[test]
    fn sequence_autonumber_is_visible_in_linear_fallback() {
        let source = concat!(
            "sequenceDiagram\n",
            "actor U as 用户\n",
            "autonumber\n",
            "U->>U: 自己\n",
        );
        let rendered = super::super::render(source, 80).unwrap();
        assert_spans_exact(source, &rendered);
        let text = facade_text(source, 80);
        assert!(text.contains("1 ") && text.contains("自己"), "{text}");
    }

    #[test]
    fn sequence_rejects_misplaced_or_duplicate_autonumber_and_actor_declarations() {
        for source in [
            "sequenceDiagram\nautonumber\nautonumber\nparticipant A as A\nparticipant B as B\nA->>B: x\n",
            "sequenceDiagram\nparticipant A as A\nparticipant B as B\nA->>B: x\nautonumber\n",
            "sequenceDiagram\nparticipant A as A\nparticipant B as B\nalt branch\nautonumber\nA->>B: x\nend\n",
            "sequenceDiagram\nparticipant用户\nparticipant B\n用户->>B: x\n",
            "sequenceDiagram\nparticipant A/B\nparticipant B\nA/B->>B: x\n",
            "sequenceDiagram\nparticipant A as A\nparticipant B as B\nA->>++B: x\n",
        ] {
            assert!(super::super::render(source, 80).is_none(), "{source}");
        }
    }

    #[test]
    fn er_renders_entities_as_tables_with_relationships() {
        let source = concat!(
            "erDiagram\n",
            "USER {\n  int id PK\n  string name\n}\n",
            "ORDER {\n  int id PK\n  int user_id FK\n}\n",
            "USER ||--o{ ORDER : places\n",
        );
        let rendered = super::super::render(source, 80).unwrap();
        assert_spans_exact(source, &rendered);
        let text = facade_text(source, 80);
        assert!(text.contains("┌") && text.contains("├"), "{text}");
        assert!(
            text.contains("int id PK") && text.contains("int user_id FK"),
            "{text}"
        );
        assert!(text.contains("places"), "{text}");
    }

    #[test]
    fn er_screenshot_topology_uses_entity_boxes_and_relationship_routes() {
        let source = concat!(
            "erDiagram\n",
            "USER {\n  int id PK\n  string name\n}\n",
            "ORDER {\n  int id PK\n  int user_id FK\n}\n",
            "ORDER_ITEM {\n  int order_id FK\n  int product_id FK\n  int quantity\n}\n",
            "PRODUCT {\n  int id PK\n  string name\n}\n",
            "USER ||--o{ ORDER : places\n",
            "ORDER ||--|{ ORDER_ITEM : contains\n",
            "PRODUCT ||--o{ ORDER_ITEM : references\n",
        );
        let rendered = super::super::render(source, 120).unwrap();
        assert_spans_exact(source, &rendered);
        let text = facade_text(source, 120);
        assert!(text.contains('┌') && text.contains('└'), "{text}");
        assert!(
            text.lines().any(|line| line.contains("│ int id PK")),
            "{text}"
        );
        assert!(
            text.contains("places") && text.contains("contains") && text.contains("references"),
            "{text}"
        );
        assert!(
            !text.lines().any(|line| line.starts_with("USER||")),
            "{text}"
        );
    }

    #[test]
    fn er_cycles_keep_entity_tables_and_relationship_routes() {
        let source = concat!(
            "erDiagram\n",
            "USER {\n  int id PK\n}\n",
            "TEAM {\n  int id PK\n}\n",
            "USER ||--o{ TEAM : joins\n",
            "TEAM ||--o{ USER : includes\n",
        );
        let rendered = super::super::render(source, 80).expect("cyclic ER diagram");
        assert_spans_exact(source, &rendered);
        let text = facade_text(source, 80);
        let user_row = text
            .lines()
            .position(|line| line.contains("USER") && line.contains('│'))
            .unwrap();
        let team_row = text
            .lines()
            .position(|line| line.contains("TEAM") && line.contains('│'))
            .unwrap();
        let joins_row = text
            .lines()
            .position(|line| line.contains("joins"))
            .unwrap();
        let includes_row = text
            .lines()
            .position(|line| line.contains("includes"))
            .unwrap();
        assert!(
            user_row < team_row,
            "entities lost their stable vertical layout:\n{text}"
        );
        assert!(
            joins_row > user_row && joins_row < team_row,
            "joins relation is not routed between its entities:\n{text}"
        );
        assert!(
            includes_row > user_row && includes_row < team_row,
            "includes relation is not routed between its entities:\n{text}"
        );
        assert!(
            text.lines().any(|line| line.contains("│ int id PK │")),
            "entity attributes are no longer table rows:\n{text}"
        );
        assert!(
            !text.lines().any(|line| line.starts_with("USER||")),
            "cycle fell back to linear Mermaid source:\n{text}"
        );
    }

    #[test]
    fn er_empty_entity_self_relation_uses_readable_linear_fallback() {
        let source = concat!("erDiagram\n", "NODE {\n}\n", "NODE ||--o{ NODE : parent\n",);
        let text = facade_text(source, 40);
        let relation = text.lines().find(|row| row.contains("parent")).unwrap();
        assert!(
            relation.contains("NODE||--o{NODE : parent"),
            "self relation disappeared when both graphical ports collapsed:\n{text}"
        );
    }

    #[test]
    fn er_relaxed_routes_go_around_intermediate_entity_tables() {
        let source = concat!(
            "erDiagram\n",
            "A {\n  int id PK\n}\n",
            "B {\n  int id PK\n}\n",
            "C {\n  int id PK\n}\n",
            "A ||--o{ C : forward\n",
            "C ||--o{ A : return\n",
        );
        let text = facade_text(source, 100);
        let rows = text.lines().collect::<Vec<_>>();
        let b_header = rows
            .iter()
            .position(|row| row.contains('B') && row.contains('│'))
            .unwrap();
        for expected in [
            "┌───────────┐",
            "│     B     │",
            "├───────────┤",
            "│ int id PK │",
            "└───────────┘",
        ] {
            assert!(
                rows[b_header - 1..=b_header + 3]
                    .iter()
                    .any(|row| row.contains(expected)),
                "intermediate B table was crossed or damaged at {expected:?}:\n{text}"
            );
        }
        assert!(
            text.contains("forward") && text.contains("return"),
            "{text}"
        );
    }

    #[test]
    fn er_routes_multiple_parents_into_order_item_without_crossing_tables() {
        let source = concat!(
            "erDiagram\n",
            "USER {\n  int id PK\n}\n",
            "PRODUCT {\n  int id PK\n}\n",
            "ORDER {\n  int id PK\n}\n",
            "ORDER_ITEM {\n  int id PK\n}\n",
            "USER ||--o{ ORDER : places\n",
            "PRODUCT ||--o{ ORDER : contains\n",
            "ORDER ||--|{ ORDER_ITEM : includes\n",
        );
        let rendered = super::super::render(source, 120).unwrap();
        assert_spans_exact(source, &rendered);
        let text = facade_text(source, 120);
        for label in [
            "USER",
            "PRODUCT",
            "ORDER",
            "ORDER_ITEM",
            "places",
            "contains",
            "includes",
            "int id PK",
        ] {
            assert!(text.contains(label), "missing {label}:\n{text}");
        }
        let relation_lines = text
            .lines()
            .filter(|line| {
                line.contains("places") || line.contains("contains") || line.contains("includes")
            })
            .collect::<Vec<_>>();
        assert_eq!(relation_lines.len(), 3, "{text}");
        assert!(text.matches('1').count() >= 2, "{text}");
        assert!(text.matches('◇').count() >= 2, "{text}");
        assert!(text.matches('┤').count() >= 1, "{text}");
        assert!(
            text.lines().any(|line| line.contains("│ int id PK │")),
            "{text}"
        );
    }

    #[test]
    fn state_implicit_endpoint_scope_is_registered_and_cross_scope_fails_closed() {
        let source = concat!(
            "stateDiagram-v2\n",
            "state Parent {\n",
            "  [*] --> Child\n",
            "}\n",
            "Child --> Outside\n",
        );
        assert!(super::super::render(source, 120).is_none());
    }

    #[test]
    fn mermaid_layout_invariants_fail_closed_at_capacity_boundaries() {
        let er = concat!(
            "erDiagram\n",
            "A {\n  int id\n}\n",
            "B {\n  int id\n}\n",
            "C {\n  int id\n}\n",
            "A ||--|| B : first\n",
            "A ||--|| C : second\n",
        );
        assert!(super::super::render(er, 80).is_some());
        let flow = concat!(
            "flowchart LR\n",
            "A[a] --> B[b]\n",
            "A --> C[c]\n",
            "A --> D[d]\n",
            "A --> E[e]\n",
        );
        assert!(super::super::render(flow, 80).is_some());

        let cross_scope = concat!(
            "stateDiagram-v2\n",
            "state Parent {\n",
            "  [*] --> Child\n",
            "}\n",
            "Child --> Outside\n",
        );
        assert!(super::super::render(cross_scope, 120).is_none());
        let nested = concat!(
            "stateDiagram-v2\n",
            "state Outer {\n",
            "  state Inner {\n",
            "    [*] --> Leaf\n",
            "  }\n",
            "}\n",
        );
        let nested_text = facade_text(nested, 120);
        assert!(!nested_text.contains('╭'), "{nested_text}");
    }

    #[test]
    fn class_renders_compartment_boxes_and_inheritance_connector() {
        let source = concat!(
            "classDiagram\n",
            "class Animal {\n",
            "  +name: str\n",
            "  +speak()\n",
            "}\n",
            "class Dog\n",
            "Animal <|-- Dog\n",
        );
        let rendered = super::super::render(source, 80).unwrap();
        assert_spans_exact(source, &rendered);
        let text = facade_text(source, 80);
        assert!(text.contains("┌") && text.contains("┐"), "{text}");
        assert!(text.matches('├').count() >= 2, "{text}");
        assert!(text.contains("△"), "{text}");
        assert!(
            text.find("Animal").unwrap() < text.find("Dog").unwrap(),
            "{text}"
        );
    }

    #[test]
    fn class_accepts_common_cardinality_and_annotation_syntax() {
        let source = concat!(
            "classDiagram\n",
            "class User {\n",
            "  +String id\n",
            "  +createOrder() Order\n",
            "}\n",
            "class Order {\n  +String id\n}\n",
            "User \"1\" --> \"*\" Order : creates\n",
        );
        let rendered = super::super::render(source, 80).unwrap();
        assert_spans_exact(source, &rendered);
        let text = facade_text(source, 80);
        assert!(text.contains("User") && text.contains("Order"), "{text}");
        assert!(text.contains("createOrder"), "{text}");
    }

    #[test]
    fn state_renders_start_and_end_icons_with_boxed_states() {
        let source = concat!(
            "stateDiagram-v2\n",
            "state \"等待\" as Waiting\n",
            "state \"完成\" as Done\n",
            "[*] --> Waiting : start\n",
            "Waiting --> Done : finish\n",
            "Done --> [*]\n",
        );
        let rendered = super::super::render(source, 80).unwrap();
        assert_spans_exact(source, &rendered);
        let text = facade_text(source, 80);
        assert!(text.contains('●'), "{text}");
        assert!(text.contains('◎'), "{text}");
        assert!(
            text.contains("│ 等待  │") && text.contains("│ 完成  │"),
            "{text}"
        );
        assert!(text.contains('▼'), "{text}");
    }

    #[test]
    fn state_cycles_route_back_to_the_existing_boxed_state() {
        let source = concat!(
            "stateDiagram-v2\n",
            "state \"等待输入\" as Idle\n",
            "state \"模型推理\" as Thinking\n",
            "state \"执行工具\" as Tool\n",
            "state \"输出回复\" as Reply\n",
            "[*] --> Idle\n",
            "Idle --> Thinking : 提交任务\n",
            "Thinking --> Tool : 请求调用工具\n",
            "Tool --> Thinking : 返回工具结果\n",
            "Thinking --> Reply : 最终答案\n",
            "Reply --> [*] : 输出完成\n",
        );
        let rendered = super::super::render(source, 80).expect("cyclic state diagram");
        assert_spans_exact(source, &rendered);
        let text = facade_text(source, 80);
        let rows = text.lines().collect::<Vec<_>>();
        let thinking_row = rows
            .iter()
            .position(|row| row.contains("模型推理") && row.contains('│'))
            .unwrap();
        let tool_row = rows
            .iter()
            .position(|row| row.contains("执行工具") && row.contains('│'))
            .unwrap();
        let feedback_label_row = rows
            .iter()
            .position(|row| row.contains("返回工具结果"))
            .unwrap();
        for transition in [
            "提交任务",
            "请求调用工具",
            "返回工具结果",
            "最终答案",
            "输出完成",
        ] {
            assert!(
                text.contains(transition),
                "missing transition {transition}:\n{text}"
            );
        }
        assert!(
            thinking_row < tool_row,
            "main state chain is not laid out top to bottom:\n{text}"
        );
        assert!(
            feedback_label_row > thinking_row && feedback_label_row < tool_row,
            "feedback label is detached from the loop corridor:\n{text}"
        );
        assert_eq!(
            text.matches("模型推理").count(),
            1,
            "target state was repeated as an ordinary hint:\n{text}"
        );
        assert!(
            rows[thinking_row].contains('◀') || rows[thinking_row].contains('▶'),
            "feedback edge does not enter the existing Thinking box:\n{text}"
        );
        assert!(
            !text.contains("[ 模型推理 ]"),
            "cycle fell back to boxed transition text:\n{text}"
        );
    }

    #[test]
    fn state_accepts_chinese_composites_declared_after_transitions() {
        let source = concat!(
            "stateDiagram-v2\n",
            "[*] --> 待支付\n",
            "待支付 --> 支付处理中 : 提交支付\n",
            "state 支付处理中 {\n",
            "  [*] --> 验证订单\n",
            "  验证订单 --> 支付成功 : 验证通过\n",
            "  支付成功 --> [*]\n",
            "}\n",
            "支付处理中 --> 已支付 : 支付成功\n",
        );
        let rendered = super::super::render(source, 100).unwrap();
        assert_spans_exact(source, &rendered);
        let text = facade_text(source, 100);
        assert!(text.contains('●') && text.contains('◎'), "{text}");
        assert!(text.contains("支付处理中"), "{text}");
        assert!(text.contains("┌") || text.contains("[ "), "{text}");
        assert!(text.contains("验证订单"), "{text}");
    }

    #[test]
    fn state_screenshot_topology_boxes_states_with_independent_endpoints() {
        let source = concat!(
            "stateDiagram-v2\n",
            "[*] --> 待支付\n",
            "待支付 --> 已支付 : 支付成功\n",
            "待支付 --> 已取消 : 超时\n",
            "已支付 --> 配送中 : 发货\n",
            "配送中 --> 已完成 : 签收\n",
            "已取消 --> [*]\n",
            "已完成 --> [*]\n",
        );
        let rendered = super::super::render(source, 120).unwrap();
        assert_spans_exact(source, &rendered);
        let text = facade_text(source, 120);
        assert!(text.contains('┌') && text.contains('└'), "{text}");
        assert!(text.matches('◎').count() >= 2, "{text}");
        assert!(text.contains('▼'), "{text}");
        assert!(
            !text.lines().any(|line| line.contains("[ 待支付 ]")),
            "{text}"
        );
        for line in text.lines() {
            for label in ["支付成功", "超时", "发货", "签收"] {
                if line.contains(label) {
                    assert!(
                        !line.contains('┌') && !line.contains('└') && !line.contains('▼'),
                        "label overlaps state geometry: {line}"
                    );
                }
            }
        }
    }

    #[test]
    fn gantt_screenshot_schedule_renders_timeline_bars() {
        let source = concat!(
            "gantt\n",
            "title 项目排期\n",
            "dateFormat YYYY-MM-DD\n",
            "section 设计\n",
            "需求分析 : done, req, 2026-08-01, 3d\n",
            "原型设计 : active, ui, after req, 4d\n",
            "section 开发\n",
            "功能开发 : crit, dev, after ui, 7d\n",
            "测试验收 : test, after dev, 3d\n",
        );
        let rendered = super::super::render(source, 120).unwrap();
        assert_spans_exact(source, &rendered);
        let text = facade_text(source, 120);
        let rows = text.lines().collect::<Vec<_>>();
        let requirement = rows.iter().find(|row| row.starts_with("需求分析")).unwrap();
        let prototype = rows.iter().find(|row| row.starts_with("原型设计")).unwrap();
        let development = rows.iter().find(|row| row.starts_with("功能开发")).unwrap();
        let acceptance = rows.iter().find(|row| row.starts_with("测试验收")).unwrap();
        assert!(
            requirement.find('━').unwrap() < prototype.find('▶').unwrap()
                && prototype.find('▶').unwrap() < development.find('!').unwrap()
                && development.find('!').unwrap() < acceptance.find('━').unwrap(),
            "dependent task bars are not ordered on the timeline:\n{text}"
        );
        assert!(
            requirement.contains('✓'),
            "done task lost its completion marker:\n{text}"
        );
        assert!(
            [requirement, prototype, development, acceptance]
                .iter()
                .all(|row| row.matches('┊').count() <= 6),
            "timeline reference lines are dense again:\n{text}"
        );
        assert!(text.contains("08-01") && text.contains("08-18"), "{text}");
        assert!(
            rows.iter().any(|row| row.trim_end() == "── 设计")
                && rows.iter().any(|row| row.trim_end() == "── 开发"),
            "sections are not concise group labels:\n{text}"
        );
        assert!(
            rows.first()
                .is_some_and(|row| row.trim_end() == "╭─ 项目排期 ─╮"),
            "title still expands into a full-width rule:\n{text}"
        );
        assert!(
            !text.contains("after req") && !text.contains("after ui"),
            "{text}"
        );
        assert!(!text.contains("done") && !text.contains("active"), "{text}");
    }

    #[test]
    fn gantt_scales_long_schedules_into_available_width() {
        let source = concat!(
            "gantt\n",
            "title 跨年度产品基础设施交付路线图\n",
            "dateFormat YYYY-MM-DD\n",
            "section 产品与基础设施协同交付\n",
            "基础设施 : done, infra, 2026-01-01, 120d\n",
            "产品交付 : active, release, after infra, 120d\n",
        );
        let rendered = super::super::render(source, 64).unwrap();
        assert_spans_exact(source, &rendered);
        let text = facade_text(source, 64);
        assert!(text.contains("01-01") && text.contains("08-29"), "{text}");
        assert!(
            text.lines().all(|line| line.chars().count() <= 64),
            "{text}"
        );
        assert!(!text.contains("2026-01-01, 120d"), "{text}");
        let first_task = text
            .lines()
            .find(|line| line.starts_with("基础设施"))
            .unwrap();
        assert!(first_task.matches('━').count() >= 20, "{text}");
        assert!(first_task.contains('✓'), "{text}");
    }

    #[test]
    fn gantt_rejects_semantically_invalid_dependencies_before_fallback() {
        for source in [
            concat!(
                "gantt\n",
                "dateFormat YYYY-MM-DD\n",
                "任务 : task, after missing, 2d\n",
            ),
            concat!(
                "gantt\n",
                "dateFormat YYYY-MM-DD\n",
                "任务一 : a, after b, 2d\n",
                "任务二 : b, after a, 2d\n",
            ),
            concat!(
                "gantt\n",
                "dateFormat YYYY-MM-DD\n",
                "任务一 : same, 2026-08-01, 2d\n",
                "任务二 : same, 2026-08-03, 2d\n",
            ),
        ] {
            assert!(
                super::super::render(source, 120).is_none(),
                "accepted {source:?}"
            );
        }
    }

    #[test]
    fn gantt_unsupported_canvas_semantics_use_exact_linear_fallback() {
        for source in [
            concat!(
                "gantt\n",
                "dateFormat YYYY-MM-DD\n",
                "快速任务 : fast, 2026-08-01, 12h\n",
            ),
            concat!(
                "gantt\n",
                "dateFormat DD/MM/YYYY\n",
                "任务 : task, 2026-08-01, 2d\n",
            ),
        ] {
            let rendered = super::super::render(source, 120).unwrap();
            assert_spans_exact(source, &rendered);
            let text = facade_text(source, 120);
            assert!(text.contains("2026-08-01"), "{text}");
            assert!(!text.contains("08-03"), "expected linear fallback:\n{text}");
        }
    }

    #[test]
    fn gantt_axis_format_keeps_the_graphical_timeline() {
        for (axis_format, start, end) in [
            ("%m-%d", "08-01", "08-03"),
            ("%Y-%m-%d", "2026-08-01", "2026-08-03"),
        ] {
            let source = format!(
                "gantt\ndateFormat YYYY-MM-DD\naxisFormat {axis_format}\nsection 计划\n交付 : task, 2026-08-01, 2d\n"
            );
            let rendered = super::super::render(&source, 120).unwrap();
            assert_spans_exact(&source, &rendered);
            let text = facade_text(&source, 120);
            let rows = text.lines().collect::<Vec<_>>();
            assert!(
                rows.first().is_some_and(|row| row.trim_end() == "── 计划"),
                "section is not a concise group label:\n{text}"
            );
            let task_row = rows.iter().position(|row| row.starts_with("交付")).unwrap();
            let axis_row = rows
                .iter()
                .position(|row| row.contains(start) && row.contains(end))
                .unwrap();
            assert!(
                task_row < axis_row,
                "task bar must appear above the date axis:\n{text}"
            );
            assert!(
                rows[task_row].contains("━━━"),
                "delivery task lost its scheduled bar:\n{text}"
            );
            assert!(
                !text.contains("axisFormat") && !text.contains("2026-08-01, 2d"),
                "graphical rendering fell back to Mermaid source:\n{text}"
            );
        }

        let narrow = concat!(
            "gantt\n",
            "dateFormat YYYY-MM-DD\n",
            "axisFormat %Y-%m-%d\n",
            "section 计划\n",
            "交付 : task, 2026-08-01, 2d\n",
        );
        let narrow_text = facade_text(narrow, 27);
        assert!(
            narrow_text.contains("axisFormat %Y-%m-%d") && narrow_text.contains("2026-08-01, 2d"),
            "long-date axis silently dropped an endpoint instead of using the exact fallback:\n{narrow_text}"
        );
    }

    #[test]
    fn gantt_milestone_uses_a_single_distinct_event_marker() {
        let source = concat!(
            "gantt\n",
            "dateFormat YYYY-MM-DD\n",
            "section 发布\n",
            "正式发布 : milestone, release, 2026-09-28, 0d\n",
        );
        let text = facade_text(source, 60);
        let release = text
            .lines()
            .find(|row| row.starts_with("正式发布"))
            .unwrap();
        assert_eq!(release.matches('◆').count(), 1, "{text}");
        assert!(
            !release.contains('━'),
            "milestone was rendered as a duration bar:\n{text}"
        );
        assert!(
            text.lines().any(|row| row.trim_end() == "── 发布"),
            "{text}"
        );

        let dependency_source = concat!(
            "gantt\n",
            "dateFormat YYYY-MM-DD\n",
            "section 发布\n",
            "正式发布 : milestone, release, 2026-09-28, 0d\n",
            "发布观察 : observe, after release, 2d\n",
        );
        let dependency_text = facade_text(dependency_source, 60);
        let release_row = dependency_text
            .lines()
            .find(|row| row.starts_with("正式发布"))
            .unwrap();
        let observation_row = dependency_text
            .lines()
            .find(|row| row.starts_with("发布观察"))
            .unwrap();
        assert_eq!(
            release_row.find('◆'),
            observation_row.find('━'),
            "a task after a zero-day milestone started one day late:\n{dependency_text}"
        );

        let terminal_source = concat!(
            "gantt\n",
            "dateFormat YYYY-MM-DD\n",
            "section 发布\n",
            "准备发布 : prep, 2026-09-27, 1d\n",
            "正式发布 : milestone, release, 2026-09-28, 0d\n",
        );
        let terminal_text = facade_text(terminal_source, 60);
        let terminal_release = terminal_text
            .lines()
            .find(|row| row.starts_with("正式发布"))
            .unwrap();
        assert_eq!(
            terminal_release.matches('◆').count(),
            1,
            "milestone at the timeline's right boundary disappeared:\n{terminal_text}"
        );
    }

    #[test]
    fn pie_renders_a_segmented_proportion_bar_and_readable_legend() {
        let source = concat!(
            "pie showData title 语言使用占比\n",
            "\"TypeScript\" : 45\n",
            "\"Rust\" : 30\n",
            "\"Python\" : 15\n",
            "\"其他\" : 10\n",
        );
        let rendered = super::super::render(source, 64).expect("pie proportion bar");
        assert_spans_exact(source, &rendered);
        let text = facade_text(source, 64);
        let rows = text.lines().collect::<Vec<_>>();
        assert_eq!(rows.first().copied(), Some("语言使用占比"), "{text}");
        let bar = rows[2];
        let segment_counts = ['█', '▓', '▒', '░'].map(|glyph| bar.matches(glyph).count());
        assert_eq!(segment_counts.iter().sum::<usize>(), 64, "{text}");
        assert!(
            segment_counts.windows(2).all(|pair| pair[0] > pair[1]),
            "segment lengths do not reflect descending values:\n{text}"
        );
        for legend in [
            "█ TypeScript  45% · 45",
            "▓ Rust        30% · 30",
            "▒ Python      15% · 15",
            "░ 其他        10% · 10",
        ] {
            assert!(
                rows.iter().any(|row| row == &legend),
                "missing legend {legend:?}:\n{text}"
            );
        }

        let without_data = concat!(
            "pie title 语言使用占比\n",
            "\"TypeScript\" : 45\n",
            "\"Rust\" : 30\n",
            "\"Python\" : 15\n",
            "\"其他\" : 10\n",
        );
        let text = facade_text(without_data, 40);
        assert!(text.contains("█ TypeScript  45%"), "{text}");
        assert!(
            !text.contains("· 45"),
            "showData=false leaked raw values:\n{text}"
        );

        let decimals = concat!(
            "pie title Decimal values\n",
            "\"Ready\" : 12.5\n",
            "\"Remaining\" : 87.5\n",
        );
        let decimal_text = facade_text(decimals, 40);
        assert!(decimal_text.contains("Ready      12.5%"), "{decimal_text}");
        assert!(decimal_text.contains("Remaining  87.5%"), "{decimal_text}");
        for invalid in ["1e3", "+1", ".5", "5."] {
            let source = format!("pie\n\"A\" : {invalid}\n\"B\" : 1\n");
            assert!(
                super::super::render(&source, 40).is_none(),
                "accepted non-decimal pie value {invalid:?}"
            );
        }
    }

    #[test]
    fn state_composite_renders_root_and_inner_orthogonal_layout_with_exact_spans() {
        let source = concat!(
            "stateDiagram-v2\n",
            "[*] --> 待支付\n",
            "待支付 --> 支付处理中 : 提交支付\n",
            "state 支付处理中 {\n",
            "  [*] --> 验证订单\n",
            "  验证订单 --> 支付成功 : 验证通过\n",
            "  支付成功 --> [*]\n",
            "}\n",
            "支付处理中 --> 已支付 : 支付完成\n",
            "已支付 --> [*]\n",
        );
        let rendered = super::super::render(source, 120).unwrap();
        assert_spans_exact(source, &rendered);
        let text = facade_text(source, 120);
        for label in ["待支付", "支付处理中", "验证订单", "支付成功", "已支付"] {
            assert!(text.contains(label), "missing {label}:\n{text}");
        }
        assert!(text.contains('┌') && text.contains('└'), "{text}");
        assert!(text.contains('▼'), "{text}");
    }

    #[test]
    fn journey_renders_section_lanes_connected_tasks_and_score_marks() {
        let source = concat!(
            "journey\n",
            "title 完成编程任务\n",
            "section 发起任务\n",
            "打开项目: 5: 用户\n",
            "描述需求: 4: 用户\n",
            "section Agent 执行\n",
            "分析代码: 4: Agent\n",
            "运行验证: 5: Agent\n",
        );
        let rendered = super::super::render(source, 120).expect("journey diagram");
        assert_spans_valid(source, &rendered);
        let text = facade_text(source, 120);
        let rows = text.lines().collect::<Vec<_>>();
        assert!(
            rows.first().is_some_and(|row| row.contains("完成编程任务")),
            "journey title is missing:\n{text}"
        );
        let first_section = rows
            .iter()
            .position(|row| row.starts_with("╭─ 发起任务"))
            .unwrap();
        let first_tasks = rows
            .iter()
            .position(|row| {
                row.contains("打开项目") && row.contains("─▶─") && row.contains("描述需求")
            })
            .unwrap();
        let second_section = rows
            .iter()
            .position(|row| row.starts_with("╭─ Agent 执行"))
            .unwrap();
        let second_tasks = rows
            .iter()
            .position(|row| {
                row.contains("分析代码") && row.contains("─▶─") && row.contains("运行验证")
            })
            .unwrap();
        assert!(
            first_section < first_tasks
                && first_tasks < second_section
                && second_section < second_tasks,
            "journey stages or task order are incorrect:\n{text}"
        );
        assert!(
            rows[first_tasks].contains("打开项目 ●●●●● 用户"),
            "first task lost score or actor context:\n{text}"
        );
        assert!(
            rows[first_tasks].contains("描述需求 ●●●●○ 用户"),
            "second task lost score or actor context:\n{text}"
        );
        assert!(
            !text.contains("score=") && !text.contains("participants="),
            "debug-style field names leaked into the chart:\n{text}"
        );

        let compact = super::super::render(source, 32).expect("compact journey diagram");
        assert_spans_valid(source, &compact);
        let compact_text = compact
            .lines
            .iter()
            .map(|line| {
                line.iter()
                    .map(|span| span.text.as_str())
                    .collect::<String>()
            })
            .collect::<Vec<_>>()
            .join("\n");
        let compact_rows = compact_text.lines().collect::<Vec<_>>();
        let open_project = compact_rows
            .iter()
            .position(|row| row.contains("打开项目"))
            .unwrap();
        let describe_task = compact_rows
            .iter()
            .position(|row| row.contains("描述需求"))
            .unwrap();
        assert!(
            open_project < describe_task,
            "compact journey did not preserve task order:\n{compact_text}"
        );
        assert!(
            compact_rows[open_project + 2].contains('▼'),
            "compact journey did not connect vertically stacked tasks:\n{compact_text}"
        );

        let long_section = "journey\nsection A deliberately long delivery phase\nShip: 5: Agent\n";
        let long_rendered = super::super::render(long_section, 60).expect("long journey section");
        let long_rows = long_rendered
            .lines
            .iter()
            .map(|line| {
                line.iter()
                    .map(|span| span.text.as_str())
                    .collect::<String>()
            })
            .collect::<Vec<_>>();
        let widths = long_rows
            .iter()
            .map(|row| crate::tui::measure::display_width(row))
            .collect::<std::collections::HashSet<_>>();
        assert_eq!(
            widths.len(),
            1,
            "section border and task rows have different widths: {long_rows:?}"
        );

        let stacked_source = "journey\nsection X\nT: 5: A\nU: 4: B\n";
        let stacked = super::super::render(stacked_source, 16).expect("stacked journey");
        let stacked_rows = stacked
            .lines
            .iter()
            .map(|line| {
                line.iter()
                    .map(|span| span.text.as_str())
                    .collect::<String>()
            })
            .collect::<Vec<_>>();
        assert!(
            stacked_rows
                .iter()
                .all(|row| crate::tui::measure::display_width(row) == 16),
            "stacked journey rows do not share the section width: {stacked_rows:?}"
        );
        for task in ["T", "U"] {
            let row = stacked_rows.iter().find(|row| row.contains(task)).unwrap();
            assert!(
                row.ends_with("││"),
                "stacked task {task} lost its inner or outer right border: {row:?}"
            );
        }
    }

    #[test]
    fn gitgraph_renders_left_side_tracks_and_right_side_event_labels() {
        let source = concat!(
            "gitGraph\n",
            "commit id: \"初始化项目\"\n",
            "commit id: \"实现核心功能\"\n",
            "branch   feature\n",
            "checkout   feature\n",
            "commit id: \"开发新功能\"\n",
            "commit id: \"补充测试\"\n",
            "checkout  main\n",
            "commit id: \"修复线上问题\"\n",
            "merge   feature\n",
            "commit id: \"发布版本\"\n",
        );
        let rendered = super::super::render(source, 100).expect("gitGraph diagram");
        assert_spans_exact(source, &rendered);
        let text = facade_text(source, 100);
        let rows = text.lines().collect::<Vec<_>>();
        let header = rows.first().unwrap();
        let main_header = header.find("main").unwrap();
        let feature_header = header.find("feature").unwrap();
        assert!(
            main_header < feature_header,
            "branch headers are not aligned with their left-side lanes:\n{text}"
        );
        let initial = rows
            .iter()
            .position(|row| row.contains("commit 初始化项目"))
            .unwrap();
        let feature_lane = feature_header + "feature".len() / 2;
        assert_eq!(
            rows[initial].chars().nth(feature_lane),
            Some(' '),
            "feature lane is visible before the branch is created:\n{text}"
        );
        let feature_commit = rows
            .iter()
            .position(|row| row.contains("commit 开发新功能"))
            .unwrap();
        let main_checkout = rows
            .iter()
            .position(|row| row.contains("checkout main"))
            .unwrap();
        let hotfix = rows
            .iter()
            .position(|row| row.contains("commit 修复线上问题"))
            .unwrap();
        let merge = rows
            .iter()
            .position(|row| row.contains("merge feature"))
            .unwrap();
        let release = rows
            .iter()
            .position(|row| row.contains("commit 发布版本"))
            .unwrap();
        assert!(
            initial < feature_commit
                && feature_commit < main_checkout
                && main_checkout < hotfix
                && hotfix < merge
                && merge < release,
            "git event order changed:\n{text}"
        );
        assert!(
            rows[feature_commit].find('●').unwrap() > rows[initial].find('●').unwrap(),
            "feature commit was not placed on its branch lane:\n{text}"
        );
        assert!(
            rows[merge].contains("●◀") || rows[merge].contains("▶●"),
            "merge row does not point from the branch back to main:\n{text}"
        );
    }

    #[test]
    fn transformed_labels_keep_their_syntactic_source_occurrence() {
        let state_source = concat!("stateDiagram-v2\n", "state \"state\" as S\n", "[*] --> S\n",);
        let state = super::super::render(state_source, 40).unwrap();
        let state_label = state
            .lines
            .iter()
            .flatten()
            .find(|span| span.text == "state")
            .unwrap();
        let quoted_state_byte = state_source.find("\"state\"").unwrap() + 1;
        assert_eq!(
            state_label.source.unwrap().start,
            state_source[..quoted_state_byte].chars().count()
        );

        let er_source = concat!(
            "erDiagram\n",
            "ITEM {\n  int id PK\n}\n",
            "ITEM ||--o{ ITEM : x\n",
        );
        let er = super::super::render(er_source, 18).unwrap();
        let mut item_starts = er
            .lines
            .iter()
            .flatten()
            .filter(|span| span.text == "ITEM")
            .map(|span| span.source.unwrap().start)
            .collect::<Vec<_>>();
        item_starts.sort_unstable();
        item_starts.dedup();
        let expected_item_starts = er_source
            .match_indices("ITEM")
            .map(|(byte, _)| er_source[..byte].chars().count())
            .collect::<Vec<_>>();
        assert_eq!(item_starts, expected_item_starts);

        let git_source = concat!(
            "gitGraph\n",
            "commit id: commit\n",
            "branch   branch\n",
            "commit id: work\n",
        );
        let git = super::super::render(git_source, 60).unwrap();
        let commit_id = git
            .lines
            .iter()
            .flatten()
            .find(|span| span.text == "commit")
            .unwrap();
        let expected_commit = git_source
            .match_indices("commit")
            .nth(1)
            .map(|(byte, _)| git_source[..byte].chars().count())
            .unwrap();
        assert_eq!(commit_id.source.unwrap().start, expected_commit);
        let branch_name = git
            .lines
            .iter()
            .flatten()
            .find(|span| span.text == "branch")
            .unwrap();
        let expected_branch = git_source
            .match_indices("branch")
            .nth(1)
            .map(|(byte, _)| git_source[..byte].chars().count())
            .unwrap();
        assert_eq!(branch_name.source.unwrap().start, expected_branch);
    }

    #[test]
    fn journey_and_gitgraph_reject_invalid_semantics() {
        for source in [
            "journey\nsection 体验\n任务: 6: 用户\n",
            "journey\n任务: 5: 用户\n",
            "gitGraph\ncheckout missing\ncommit\n",
            "gitGraph\nbranch feature\nbranch feature\ncommit\n",
            "gitGraph\ncommit id: same\ncommit id: same\n",
        ] {
            assert!(
                super::super::render(source, 100).is_none(),
                "accepted {source:?}"
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
