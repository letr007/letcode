use anyhow::{Context, Result, bail};
use async_trait::async_trait;
use serde_json::{Value, json};

use super::{QuestionRequest, QuestionResponse, ToolExecutionContext, ToolHandler, ToolRegistry};

const MAX_QUESTION_HEADER_CHARS: usize = 30;
const MAX_QUESTION_OPTIONS: usize = 9;

pub(super) fn register(registry: &mut ToolRegistry) {
    registry.register(QuestionTool);
}

struct QuestionTool;

#[async_trait]
impl ToolHandler for QuestionTool {
    fn name(&self) -> &'static str {
        crate::tool_names::TOOL_QUESTION
    }

    fn description(&self) -> &'static str {
        "Ask the user one or more clarifying questions and wait for their answers."
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "questions": {
                    "type": "array",
                    "minItems": 1,
                    "description": "Questions to ask the user in order.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "question": {
                                "type": "string",
                                "description": "Complete question shown to the user"
                            },
                            "header": {
                                "type": "string",
                                "maxLength": MAX_QUESTION_HEADER_CHARS,
                                "description": "Short label for the question"
                            },
                            "options": {
                                "type": "array",
                                "minItems": 1,
                                "maxItems": MAX_QUESTION_OPTIONS,
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "label": {
                                            "type": "string",
                                            "description": "Short option label"
                                        },
                                        "description": {
                                            "type": "string",
                                            "description": "Option description"
                                        }
                                    },
                                    "required": ["label", "description"],
                                    "additionalProperties": false
                                }
                            },
                            "multiple": {
                                "type": "boolean",
                                "description": "Whether the user may select multiple answers"
                            }
                        },
                        "required": ["question", "header", "options", "multiple"],
                        "additionalProperties": false
                    }
                }
            },
            "required": ["questions"],
            "additionalProperties": false
        })
    }

    async fn execute(&self, _args: Value) -> Result<Value> {
        bail!("question tool requires an interactive runtime")
    }

    async fn execute_with_context(
        &self,
        args: Value,
        context: ToolExecutionContext,
    ) -> Result<Value> {
        let request: QuestionRequest =
            serde_json::from_value(args).context("invalid question tool arguments")?;
        validate_question_request(&request)?;

        let Some(handler) = context.question_handler else {
            bail!("question tool requires an interactive runtime with question handling enabled");
        };

        let response = handler(request.clone()).await?;
        validate_question_response(&request, &response)?;

        Ok(json!({
            "answers": response.answers,
            "message": render_question_answers(&request, &response),
        }))
    }
}

fn validate_question_request(request: &QuestionRequest) -> Result<()> {
    if request.questions.is_empty() {
        bail!("question tool requires at least one question");
    }

    for (question_index, question) in request.questions.iter().enumerate() {
        if question.question.trim().is_empty() {
            bail!("questions[{question_index}].question must not be empty");
        }
        if question.header.trim().is_empty() {
            bail!("questions[{question_index}].header must not be empty");
        }
        if question.header.chars().count() > MAX_QUESTION_HEADER_CHARS {
            bail!(
                "questions[{question_index}].header exceeds {MAX_QUESTION_HEADER_CHARS} characters"
            );
        }
        if question.options.is_empty() {
            bail!("questions[{question_index}].options must contain at least one option");
        }
        if question.options.len() > MAX_QUESTION_OPTIONS {
            bail!(
                "questions[{question_index}].options accepts at most {MAX_QUESTION_OPTIONS} items"
            );
        }
        for (option_index, option) in question.options.iter().enumerate() {
            if option.label.trim().is_empty() {
                bail!(
                    "questions[{question_index}].options[{option_index}].label must not be empty"
                );
            }
            if option.description.trim().is_empty() {
                bail!(
                    "questions[{question_index}].options[{option_index}].description must not be empty"
                );
            }
        }
    }

    Ok(())
}

fn validate_question_response(
    request: &QuestionRequest,
    response: &QuestionResponse,
) -> Result<()> {
    if response.answers.len() != request.questions.len() {
        bail!(
            "question runtime returned {} answers for {} questions",
            response.answers.len(),
            request.questions.len()
        );
    }

    for (question_index, (question, answers)) in request
        .questions
        .iter()
        .zip(response.answers.iter())
        .enumerate()
    {
        if answers.is_empty() {
            bail!("questions[{question_index}] was left unanswered");
        }
        if !question.multiple && answers.len() > 1 {
            bail!("questions[{question_index}] accepts at most one answer");
        }
        for (answer_index, answer) in answers.iter().enumerate() {
            if answer.trim().is_empty() {
                bail!("questions[{question_index}].answers[{answer_index}] must not be empty");
            }
        }
    }

    Ok(())
}

fn render_question_answers(request: &QuestionRequest, response: &QuestionResponse) -> String {
    let mut lines = vec!["User has answered your questions:".to_string()];
    for (question, answers) in request.questions.iter().zip(response.answers.iter()) {
        let rendered = if answers.is_empty() {
            "Unanswered".to_string()
        } else {
            answers.join(", ")
        };
        lines.push(format!("- {}: {}", question.header, rendered));
    }
    lines.join("\n")
}

#[cfg(test)]
mod tests {
    use super::super::{QuestionRequest, QuestionResponse, ToolExecutionContext, ToolRegistry};
    use crate::tool_names;
    use serde_json::json;
    use std::sync::Arc;

    #[tokio::test]
    async fn question_tool_uses_interactive_callback() {
        let registry = ToolRegistry::default_tools();
        let context = ToolExecutionContext {
            question_handler: Some(Arc::new(|request: QuestionRequest| {
                Box::pin(async move {
                    assert_eq!(request.questions.len(), 1);
                    Ok(QuestionResponse {
                        answers: vec![vec!["Fast".into()]],
                    })
                })
            })),
            ..ToolExecutionContext::default()
        };

        let output = registry
            .call_with_context(
                tool_names::TOOL_QUESTION,
                json!({
                    "questions": [{
                        "question": "Choose mode",
                        "header": "Mode",
                        "options": [{"label": "Fast", "description": "Fast path"}],
                        "multiple": false
                    }]
                }),
                context,
            )
            .await;

        assert!(output.ok, "{:?}", output.error);
        let data = output.data.expect("question result data");
        assert_eq!(data["answers"], json!([["Fast"]]));
        assert!(
            data["message"]
                .as_str()
                .is_some_and(|message| message.contains("User has answered your questions"))
        );
    }

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
}
