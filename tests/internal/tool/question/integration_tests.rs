use super::super::{QuestionRequest, QuestionResponse, ToolExecutionContext, ToolRegistry};
use crate::tool_names;
use serde_json::json;
use std::sync::Arc;

#[tokio::test]
async fn question_tool_rejects_unanswered_or_invalid_single_select_responses() {
    let registry = ToolRegistry::default_tools();
    let args = json!({
        "questions": [{
            "question": "Choose mode",
            "header": "Mode",
            "options": [{"label": "Fast", "description": "Fast path"}],
            "multiple": false
        }]
    });

    let unanswered = registry
        .call_with_context(
            tool_names::TOOL_QUESTION,
            args.clone(),
            ToolExecutionContext {
                question_handler: Some(Arc::new(|_| {
                    Box::pin(async {
                        Ok(QuestionResponse {
                            answers: vec![vec![]],
                        })
                    })
                })),
                ..ToolExecutionContext::default()
            },
        )
        .await;
    assert!(!unanswered.ok);

    let invalid_single = registry
        .call_with_context(
            tool_names::TOOL_QUESTION,
            args,
            ToolExecutionContext {
                question_handler: Some(Arc::new(|_| {
                    Box::pin(async {
                        Ok(QuestionResponse {
                            answers: vec![vec!["Fast".into(), "Custom".into()]],
                        })
                    })
                })),
                ..ToolExecutionContext::default()
            },
        )
        .await;
    assert!(!invalid_single.ok);
}
