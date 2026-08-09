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
    fn flowchart_horizontal_corridor_capacity_falls_back() {
        let source = concat!(
            "graph LR\n",
            "A[a] --> B[b]\n",
            "A --> C[c]\n",
            "A --> D[d]\n",
            "A --> E[e]\n",
            "A --> F[f]\n",
            "A --> G[g]\n",
            "A --> H[h]\n",
        );
        let text = rendered_text(source, 40);
        assert!(!text.contains('╭'), "expected fallback:\n{text}");
        assert!(text.lines().all(|line| !line.contains('┌')), "{text}");
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
        assert!(text.matches('╭').count() >= 4, "{text}");
        assert!(text.contains('▶'), "{text}");
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
            "pie\ntitle unsupported\n",
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
            text.contains("│ 等待 │") && text.contains("│ 完成 │"),
            "{text}"
        );
        assert!(text.contains('▼'), "{text}");
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
        for marker in ['=', '#', '!', '-'] {
            assert!(text.contains(marker), "missing {marker}:\n{text}");
        }
        assert!(text.contains("08-01") && text.contains("08-18"), "{text}");
        assert!(
            text.contains("section 设计") && text.contains("section 开发"),
            "{text}"
        );
    }

    #[test]
    fn gantt_axis_format_uses_linear_fallback_with_exact_spans() {
        let source = concat!(
            "gantt\n",
            "dateFormat YYYY-MM-DD\n",
            "axisFormat %Y-%m-%d\n",
            "section 计划\n",
            "交付 : task, 2026-08-01, 2d\n",
        );
        let rendered = super::super::render(source, 120).unwrap();
        assert_spans_exact(source, &rendered);
        let text = facade_text(source, 120);
        assert!(text.contains("axisFormat %Y-%m-%d"), "{text}");
        assert_eq!(text.lines().count(), 4, "expected linear fallback:\n{text}");
        assert!(!text.contains("08-03"), "expected linear fallback:\n{text}");
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
