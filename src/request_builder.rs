use async_openai::types::responses::{CreateResponse, InputItem, Reasoning, Tool};

/// Minimal per-model request metadata required for request construction.
///
/// This is intentionally OpenAI-Responses-specific and is used for:
/// - context-window budgeting (context_window)
/// - reserving conservative completion space (max_output_tokens)
/// - capability gating (supports_tools / supports_reasoning)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct ModelRequestMetadata {
    /// Model context window in tokens.
    pub context_window: Option<u64>,
    /// Optional known max output tokens for the model.
    pub max_output_tokens: Option<u64>,
    pub supports_tools: bool,
    pub supports_reasoning: bool,
}

impl ModelRequestMetadata {
    pub fn context_window_tokens(self) -> u64 {
        self.context_window
            .filter(|v| *v > 0)
            .unwrap_or(DEFAULT_FALLBACK_CONTEXT_WINDOW_TOKENS)
            .max(MIN_CONTEXT_WINDOW_TOKENS)
    }

    pub fn output_reserve_tokens(self) -> u64 {
        self.max_output_tokens
            .filter(|v| *v > 0)
            .unwrap_or(DEFAULT_FALLBACK_OUTPUT_RESERVE_TOKENS)
            .max(MIN_OUTPUT_RESERVE_TOKENS)
    }
}

/// Reserved seam for future fixed system/developer prompt blocks.
/// Currently unused by the application, but kept explicit so we don't have to
/// re-inline request construction in Agent later.
#[derive(Debug, Clone, Default)]
pub struct RequestPrelude {
    pub items: Vec<InputItem>,
}

#[derive(Debug, Clone)]
pub struct RequestBuilderInput<'a> {
    pub model_id: &'a str,
    pub model: ModelRequestMetadata,

    /// Full mutable history is owned by Agent; builder receives an immutable slice.
    pub history: &'a [InputItem],

    /// Index into `history` from which items are considered "protected" and must
    /// always be included (current turn and any items appended after it).
    pub protected_start_index: usize,

    pub tools: &'a [Tool],
    pub prelude: RequestPrelude,

    /// Optional reasoning request settings.
    /// This is applied only when `model.supports_reasoning` is true.
    pub reasoning: Option<Reasoning>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BudgetReport {
    pub context_window_tokens: u64,
    pub input_budget_tokens: u64,
    pub estimated_request_tokens: u64,
    pub estimated_prelude_tokens: u64,
    pub estimated_protected_tokens: u64,
    pub estimated_retained_history_tokens: u64,
    pub estimated_tools_tokens: u64,
    pub original_history_items: usize,
    pub retained_history_items: usize,
    pub dropped_history_items: usize,
    pub truncated: bool,
}

#[derive(Debug, Clone)]
pub struct BuildResult {
    pub request: CreateResponse,
    pub budget: BudgetReport,
}

const MIN_CONTEXT_WINDOW_TOKENS: u64 = 1024;
const DEFAULT_FALLBACK_CONTEXT_WINDOW_TOKENS: u64 = 8 * 1024;
const MIN_OUTPUT_RESERVE_TOKENS: u64 = 128;
const DEFAULT_FALLBACK_OUTPUT_RESERVE_TOKENS: u64 = 1024;

/// Extra reserve for request framing, tool schemas, and estimator slack.
const SAFETY_OVERHEAD_TOKENS: u64 = 256;

/// Deterministic, conservative-ish estimator.
///
/// We avoid tokenizers for now; estimate from JSON size and apply a constant
/// factor plus a per-item overhead.
fn estimate_item_tokens(item: &InputItem) -> u64 {
    // JSON size is stable/deterministic for our serde configuration.
    // Use a conservative ratio (3 chars/token) and add per-item overhead.
    let json_len = serde_json::to_string(item).map(|s| s.len()).unwrap_or(0);
    let base = (json_len as u64 + 2) / 3;
    base.saturating_add(8)
}

fn estimate_items_tokens(items: &[InputItem]) -> u64 {
    items.iter().map(estimate_item_tokens).sum()
}

fn estimate_tools_tokens(tools: &[Tool]) -> u64 {
    if tools.is_empty() {
        return 0;
    }
    let json_len = serde_json::to_string(tools).map(|s| s.len()).unwrap_or(0);
    let base = (json_len as u64 + 2) / 3;
    // Small overhead for array framing.
    base.saturating_add(16)
}

fn clamp_protected_start(len: usize, protected_start_index: usize) -> usize {
    protected_start_index.min(len)
}

pub fn build_create_response(input: RequestBuilderInput<'_>) -> BuildResult {
    let history_len = input.history.len();
    let protected_start = clamp_protected_start(history_len, input.protected_start_index);

    let (older, protected) = input.history.split_at(protected_start);
    let prelude_tokens = estimate_items_tokens(&input.prelude.items);
    let protected_tokens = estimate_items_tokens(protected);

    let context_window = input.model.context_window_tokens();
    let output_reserve = input.model.output_reserve_tokens();
    let tools_tokens = if input.model.supports_tools {
        estimate_tools_tokens(input.tools)
    } else {
        0
    };
    let input_budget = context_window
        .saturating_sub(output_reserve)
        .saturating_sub(SAFETY_OVERHEAD_TOKENS)
        .saturating_sub(tools_tokens)
        .max(1);

    // Budget only applies to history + prelude. Tool schemas are not included in
    // this estimate; SAFETY_OVERHEAD_TOKENS is intended to cover them somewhat.
    //
    // Policy: always keep prelude + protected items, then add as much older
    // history as fits from newest backwards, dropping oldest items first.
    let mut retained_older: Vec<InputItem> = Vec::new();
    let mut retained_older_tokens: u64 = 0;

    // If protected items already exceed budget, drop all older history.
    // Still send protected content to preserve current turn deterministically.
    if prelude_tokens.saturating_add(protected_tokens) < input_budget {
        // Add older items from newest to oldest until budget is reached.
        for item in older.iter().rev() {
            let cost = estimate_item_tokens(item);
            let next = prelude_tokens
                .saturating_add(protected_tokens)
                .saturating_add(retained_older_tokens)
                .saturating_add(cost);
            if next > input_budget {
                // Oldest-to-newest truncation: once we can't fit an older item,
                // we stop. This keeps a contiguous suffix of the older history.
                break;
            }
            retained_older.push(item.clone());
            retained_older_tokens = retained_older_tokens.saturating_add(cost);
        }

        // We built newest->oldest; restore chronological order.
        retained_older.reverse();
    }

    let retained_history_items = retained_older.len() + protected.len();
    let dropped_history_items = history_len.saturating_sub(retained_history_items);
    let truncated = dropped_history_items > 0;

    let mut final_input_items: Vec<InputItem> =
        Vec::with_capacity(input.prelude.items.len() + retained_older.len() + protected.len());
    final_input_items.extend(input.prelude.items);
    final_input_items.extend(retained_older.iter().cloned());
    final_input_items.extend(protected.iter().cloned());

    let estimated_request_tokens = prelude_tokens
        .saturating_add(protected_tokens)
        .saturating_add(retained_older_tokens)
        .saturating_add(tools_tokens);

    let tools = if input.model.supports_tools {
        Some(input.tools.to_vec())
    } else {
        None
    };
    let parallel_tool_calls = if input.model.supports_tools {
        Some(false)
    } else {
        None
    };

    let reasoning = if input.model.supports_reasoning {
        input.reasoning
    } else {
        None
    };

    let request = CreateResponse {
        model: Some(input.model_id.to_string()),
        input: final_input_items.into(),
        previous_response_id: None,
        tools,
        parallel_tool_calls,
        reasoning,
        ..Default::default()
    };

    BuildResult {
        request,
        budget: BudgetReport {
            context_window_tokens: context_window,
            input_budget_tokens: input_budget,
            estimated_request_tokens,
            estimated_prelude_tokens: prelude_tokens,
            estimated_protected_tokens: protected_tokens,
            estimated_retained_history_tokens: retained_older_tokens,
            estimated_tools_tokens: tools_tokens,
            original_history_items: history_len,
            retained_history_items,
            dropped_history_items,
            truncated,
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use async_openai::types::responses::{
        EasyInputContent, EasyInputMessage, FunctionTool, MessageType, ReasoningSummary, Role, Tool,
    };
    use serde_json::json;

    fn msg(role: Role, content: &str) -> InputItem {
        InputItem::EasyMessage(EasyInputMessage {
            r#type: MessageType::Message,
            role,
            content: EasyInputContent::Text(content.to_string()),
            phase: None,
        })
    }

    #[test]
    fn keeps_all_history_when_budget_large_enough() {
        let history = vec![
            msg(Role::User, "u1"),
            msg(Role::Assistant, "a1"),
            msg(Role::User, "u2"),
        ];
        let input = RequestBuilderInput {
            model_id: "gpt-test",
            model: ModelRequestMetadata {
                context_window: Some(128_000),
                max_output_tokens: Some(1024),
                supports_tools: true,
                supports_reasoning: true,
            },
            history: &history,
            protected_start_index: 2,
            tools: &[],
            prelude: RequestPrelude::default(),
            reasoning: None,
        };

        let result = build_create_response(input);
        assert!(!result.budget.truncated);
        assert_eq!(result.budget.dropped_history_items, 0);
        assert_eq!(result.budget.retained_history_items, 3);
    }

    #[test]
    fn truncates_oldest_history_first_preserving_protected_items() {
        let long = "x".repeat(5000);
        let history = vec![
            msg(Role::User, "old-1"),
            msg(Role::Assistant, &long),
            msg(Role::User, "current"),
            msg(Role::Assistant, "tool-ish"),
        ];

        let input = RequestBuilderInput {
            model_id: "gpt-test",
            model: ModelRequestMetadata {
                context_window: Some(1200),
                max_output_tokens: Some(256),
                supports_tools: true,
                supports_reasoning: true,
            },
            history: &history,
            protected_start_index: 2,
            tools: &[],
            prelude: RequestPrelude::default(),
            reasoning: None,
        };

        let result = build_create_response(input);
        assert!(result.budget.truncated);
        // Must keep protected (index 2..)
        let request_json = serde_json::to_string(&result.request).expect("request serializes");
        assert!(request_json.contains("current"));
        assert!(request_json.contains("tool-ish"));
    }

    #[test]
    fn missing_or_zero_context_window_uses_safe_defaults_and_does_not_panic() {
        let history = vec![msg(Role::User, "hello")];
        for context_window in [None, Some(0), Some(1)] {
            let input = RequestBuilderInput {
                model_id: "gpt-test",
                model: ModelRequestMetadata {
                    context_window,
                    max_output_tokens: None,
                    supports_tools: true,
                    supports_reasoning: true,
                },
                history: &history,
                protected_start_index: 0,
                tools: &[],
                prelude: RequestPrelude::default(),
                reasoning: None,
            };
            let result = build_create_response(input);
            assert!(result.budget.context_window_tokens >= MIN_CONTEXT_WINDOW_TOKENS);
        }
    }

    #[test]
    fn supports_tools_false_omits_tools_fields() {
        let history = vec![msg(Role::User, "hello")];
        let input = RequestBuilderInput {
            model_id: "gpt-test",
            model: ModelRequestMetadata {
                context_window: Some(8192),
                max_output_tokens: Some(256),
                supports_tools: false,
                supports_reasoning: true,
            },
            history: &history,
            protected_start_index: 0,
            tools: &[],
            prelude: RequestPrelude::default(),
            reasoning: None,
        };
        let result = build_create_response(input);
        assert!(result.request.tools.is_none());
        assert!(result.request.parallel_tool_calls.is_none());
    }

    #[test]
    fn supports_tools_true_includes_tools_fields_even_when_empty() {
        let history = vec![msg(Role::User, "hello")];
        let input = RequestBuilderInput {
            model_id: "gpt-test",
            model: ModelRequestMetadata {
                context_window: Some(8192),
                max_output_tokens: Some(256),
                supports_tools: true,
                supports_reasoning: true,
            },
            history: &history,
            protected_start_index: 0,
            tools: &[],
            prelude: RequestPrelude::default(),
            reasoning: None,
        };
        let result = build_create_response(input);
        assert_eq!(result.request.tools.as_ref().map(Vec::len), Some(0));
        assert_eq!(result.request.parallel_tool_calls, Some(false));
    }

    #[test]
    fn changing_context_window_changes_truncation_outcome() {
        let long = "x".repeat(10_000);
        let history = vec![
            msg(Role::User, &long),
            msg(Role::Assistant, &long),
            msg(Role::User, "current"),
        ];

        let small = build_create_response(RequestBuilderInput {
            model_id: "gpt-test",
            model: ModelRequestMetadata {
                context_window: Some(2048),
                max_output_tokens: Some(512),
                supports_tools: true,
                supports_reasoning: true,
            },
            history: &history,
            protected_start_index: 2,
            tools: &[],
            prelude: RequestPrelude::default(),
            reasoning: None,
        });

        let large = build_create_response(RequestBuilderInput {
            model_id: "gpt-test",
            model: ModelRequestMetadata {
                context_window: Some(128_000),
                max_output_tokens: Some(512),
                supports_tools: true,
                supports_reasoning: true,
            },
            history: &history,
            protected_start_index: 2,
            tools: &[],
            prelude: RequestPrelude::default(),
            reasoning: None,
        });

        assert!(small.budget.truncated);
        assert!(!large.budget.truncated);
    }

    #[test]
    fn tool_schema_size_counts_toward_budget() {
        let long = "x".repeat(6000);
        let history = vec![
            msg(Role::User, "old"),
            msg(Role::Assistant, &long),
            msg(Role::User, "current"),
        ];

        let tools = vec![Tool::Function(FunctionTool {
            name: "big_tool".into(),
            description: Some("big".into()),
            parameters: Some(json!({
                "type": "object",
                "properties": {
                    "payload": { "type": "string", "description": "x".repeat(8000) }
                },
                "required": ["payload"],
                "additionalProperties": false
            })),
            strict: Some(true),
            defer_loading: None,
        })];

        let without_tools = build_create_response(RequestBuilderInput {
            model_id: "gpt-test",
            model: ModelRequestMetadata {
                context_window: Some(4096),
                max_output_tokens: Some(512),
                supports_tools: true,
                supports_reasoning: true,
            },
            history: &history,
            protected_start_index: 2,
            tools: &[],
            prelude: RequestPrelude::default(),
            reasoning: None,
        });
        let with_tools = build_create_response(RequestBuilderInput {
            model_id: "gpt-test",
            model: ModelRequestMetadata {
                context_window: Some(4096),
                max_output_tokens: Some(512),
                supports_tools: true,
                supports_reasoning: true,
            },
            history: &history,
            protected_start_index: 2,
            tools: &tools,
            prelude: RequestPrelude::default(),
            reasoning: None,
        });

        assert!(with_tools.budget.estimated_tools_tokens > 0);
        // With the same history and budget, including large tool schemas should not retain
        // *more* history than the no-tools case.
        assert!(
            with_tools.budget.retained_history_items <= without_tools.budget.retained_history_items
        );
    }

    #[test]
    fn protected_items_exceed_budget_drops_all_older_history_but_keeps_protected() {
        let huge = "y".repeat(30_000);
        let history = vec![
            msg(Role::User, "old-1"),
            msg(Role::Assistant, "old-2"),
            msg(Role::User, &huge),
        ];

        let result = build_create_response(RequestBuilderInput {
            model_id: "gpt-test",
            model: ModelRequestMetadata {
                context_window: Some(1024),
                max_output_tokens: Some(256),
                supports_tools: true,
                supports_reasoning: true,
            },
            history: &history,
            protected_start_index: 2,
            tools: &[],
            prelude: RequestPrelude::default(),
            reasoning: None,
        });

        assert!(result.budget.truncated);
        assert_eq!(result.budget.retained_history_items, 1);
        let request_json = serde_json::to_string(&result.request).expect("request serializes");
        assert!(request_json.contains(&"y".repeat(100))); // protected content present
    }

    #[test]
    fn supports_reasoning_gates_reasoning_field() {
        let history = vec![msg(Role::User, "hello")];
        let reasoning = Reasoning {
            effort: None,
            summary: Some(ReasoningSummary::Auto),
        };

        let enabled = build_create_response(RequestBuilderInput {
            model_id: "gpt-test",
            model: ModelRequestMetadata {
                context_window: Some(8192),
                max_output_tokens: Some(256),
                supports_tools: true,
                supports_reasoning: true,
            },
            history: &history,
            protected_start_index: 0,
            tools: &[],
            prelude: RequestPrelude::default(),
            reasoning: Some(reasoning.clone()),
        });
        assert!(enabled.request.reasoning.is_some());

        let disabled = build_create_response(RequestBuilderInput {
            model_id: "gpt-test",
            model: ModelRequestMetadata {
                context_window: Some(8192),
                max_output_tokens: Some(256),
                supports_tools: true,
                supports_reasoning: false,
            },
            history: &history,
            protected_start_index: 0,
            tools: &[],
            prelude: RequestPrelude::default(),
            reasoning: Some(reasoning),
        });
        assert!(disabled.request.reasoning.is_none());
    }
}
