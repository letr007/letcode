use anyhow::Result;
use serde_json::Value;
use std::future::Future;
use std::time::Duration;
use tracing::Instrument;

use super::*;
use crate::langfuse_trace;

pub(super) async fn execute_tool_call<C, E, A, Efut, Afut>(
    agent: &mut Agent<C>,
    call: &HistoryToolCall,
    on_event: &mut E,
    approve: &mut A,
) -> Result<ToolExecutionRecord>
where
    C: Config,
    E: FnMut(AgentEvent) -> Efut,
    A: FnMut(PermissionRequest) -> Afut,
    Efut: Future<Output = Result<()>>,
    Afut: Future<Output = Result<PermissionApproval>>,
{
    let span = langfuse_trace::tool_span(
        agent.turn.turn_id,
        &call.name,
        &call.call_id,
        call.arguments_json.len(),
    );
    let result = async {
        let record = match serde_json::from_str::<Value>(&call.arguments_json) {
            Ok(args) => execute_with_arguments(agent, call, args, on_event, approve).await?,
            Err(err) => invalid_json_record(agent, call, err, on_event).await?,
        };

        agent.record_tool_effects(&record);
        Agent::<C>::emit_audit_event(
            on_event,
            AgentEvent::ToolExecutionSummary(agent.tool_execution_summary_event(&record)),
            "tool_execution_summary",
        )
        .await;
        Ok(record)
    }
    .instrument(span.clone())
    .await;
    langfuse_trace::finish_tool_span(&span, &result);
    drop(span);
    result
}

async fn execute_with_arguments<C, E, A, Efut, Afut>(
    agent: &mut Agent<C>,
    call: &HistoryToolCall,
    args: Value,
    on_event: &mut E,
    approve: &mut A,
) -> Result<ToolExecutionRecord>
where
    C: Config,
    E: FnMut(AgentEvent) -> Efut,
    A: FnMut(PermissionRequest) -> Afut,
    Efut: Future<Output = Result<()>>,
    Afut: Future<Output = Result<PermissionApproval>>,
{
    let directive = agent.turn.policy.directive;
    let permission_class = permission_class_for_tool_call(&agent.tools, &call.name);

    if !is_executable_tool(agent, &call.name) {
        let output = ToolResult::err(
            &call.name,
            format!("unknown or unavailable tool: {}", call.name),
        );
        let record = ToolExecutionRecord::new(
            call,
            Some(args),
            permission_class,
            directive,
            ToolExecutionStatus::Rejected,
            None,
            output,
        );
        emit_finished(on_event, call, &record).await?;
        return Ok(record);
    }

    if matches!(
        call.name.as_str(),
        tool_names::TOOL_CONTEXT_CHECKPOINT | tool_names::TOOL_CONTEXT_RETURN
    ) && let Err(error) = agent.validate_context_control_tool(&call.name)
    {
        let output = ToolResult::err(&call.name, error.to_string());
        let record = ToolExecutionRecord::new(
            call,
            Some(args),
            permission_class,
            directive,
            ToolExecutionStatus::Rejected,
            None,
            output,
        );
        emit_finished(on_event, call, &record).await?;
        return Ok(record);
    }

    if !agent.tools.scope().allows_tool(&call.name) {
        let output = ToolResult::err(
            &call.name,
            agent.tools.scope().rejection_message(&call.name),
        );
        let record = ToolExecutionRecord::new(
            call,
            Some(args),
            permission_class,
            directive,
            ToolExecutionStatus::Rejected,
            Some(ToolExecutionRejection::ToolScopeDenied),
            output,
        );
        emit_finished(on_event, call, &record).await?;
        return Ok(record);
    }

    if let Some(message) =
        restricted_by_directive_with_class(&call.name, &args, permission_class, directive)
    {
        let output = ToolResult::err(&call.name, message);
        let record = ToolExecutionRecord::new(
            call,
            Some(args),
            permission_class,
            directive,
            ToolExecutionStatus::Rejected,
            Some(ToolExecutionRejection::DirectiveBlocked),
            output,
        );
        emit_finished(on_event, call, &record).await?;
        return Ok(record);
    }

    let external_workspace_access = external_workspace_access_for_tool(&call.name, &args);
    let resource = crate::tool::permission_resource_for_tool(&call.name, &args);
    let (mode, permission_generation, permission_decision, grant_allowed) = {
        let state = agent
            .permission_session
            .lock()
            .map_err(|_| anyhow::anyhow!("permission session poisoned"))?;
        state.approval_snapshot(
            resource.as_ref(),
            &call.name,
            &args,
            permission_class,
            directive,
            external_workspace_access.is_some(),
            crate::permission::is_internal_tool(&call.name),
        )
    };
    let permission_decision = if mode == PermissionMode::Default && grant_allowed {
        PermissionDecision::Allow
    } else {
        permission_decision
    };
    let mut approval = None;
    let should_execute = match permission_decision {
        PermissionDecision::Allow => true,
        PermissionDecision::Ask => {
            let can_allow_always = mode == PermissionMode::Default && resource.is_some();
            let result = approve(PermissionRequest {
                call_id: Some(call.call_id.clone()),
                tool: call.name.clone(),
                args: args.clone(),
                class: permission_class,
                summary: format_tool_call(&call.name, &args),
                preview: external_workspace_access
                    .as_ref()
                    .map(|access| access.preview()),
                can_allow_always,
                grant_summary: can_allow_always
                    .then(|| resource.as_ref().expect("resource checked").summary()),
            })
            .await?;
            approval = Some(result);
            result.allowed()
        }
        PermissionDecision::Deny => false,
    };

    if matches!(approval, Some(PermissionApproval::AllowAlways)) {
        if let Some(resource) = resource {
            agent
                .permission_session
                .lock()
                .map_err(|_| anyhow::anyhow!("permission session poisoned"))?
                .grant_if_current_default(permission_generation, resource);
        }
    }

    if should_execute {
        on_event(AgentEvent::ToolCallStarted {
            call_id: call.call_id.clone(),
            name: call.name.clone(),
            args: args.clone(),
        })
        .await?;

        let mut output = if is_subagent_tool_name(&call.name) {
            agent.execute_subagent_tool(&call.name, &args).await
        } else {
            let context = match agent
                .tool_execution_context_for(&call.name, external_workspace_access.is_some())
            {
                Ok(context) => context,
                Err(error) => {
                    let output = ToolResult::err(&call.name, error.to_string());
                    let record = ToolExecutionRecord::new(
                        call,
                        Some(args),
                        permission_class,
                        directive,
                        ToolExecutionStatus::Rejected,
                        None,
                        output,
                    );
                    emit_finished(on_event, call, &record).await?;
                    return Ok(record);
                }
            };
            let (delta_tx, mut delta_rx) = tokio::sync::mpsc::unbounded_channel();
            let timeout_secs = non_shell_tool_timeout_secs(agent, &call.name);
            let output = {
                let emit_tx = delta_tx.clone();
                drop(delta_tx);
                let mut emit = move |stream, chunk| {
                    emit_tx
                        .send((stream, chunk))
                        .map_err(|_| anyhow::anyhow!("tool output receiver closed"))
                };
                let output =
                    agent
                        .tools
                        .call_streaming(&call.name, args.clone(), context, &mut emit);
                tokio::pin!(output);
                if let Some(timeout_secs) = timeout_secs {
                    let timeout_sleep = tokio::time::sleep(Duration::from_secs(timeout_secs));
                    tokio::pin!(timeout_sleep);
                    let result = loop {
                        tokio::select! {
                            output = &mut output => break Ok(output),
                            Some((stream, chunk)) = delta_rx.recv() => {
                                on_event(AgentEvent::ToolOutputDelta {
                                    call_id: call.call_id.clone(),
                                    stream,
                                    chunk,
                                })
                                .await?;
                            }
                            _ = &mut timeout_sleep => break Err(timeout_secs),
                        }
                    };
                    match result {
                        Ok(output) => Ok(output),
                        Err(timeout_secs) => Err(timeout_secs),
                    }
                } else {
                    let output = loop {
                        tokio::select! {
                            output = &mut output => break output,
                            Some((stream, chunk)) = delta_rx.recv() => {
                                on_event(AgentEvent::ToolOutputDelta {
                                    call_id: call.call_id.clone(),
                                    stream,
                                    chunk,
                                })
                                .await?;
                            }
                        }
                    };
                    Ok(output)
                }
            };
            let output = match output {
                Ok(output) => output,
                Err(timeout_secs) => {
                    on_event(AgentEvent::ToolCallCancelled {
                        call_id: call.call_id.clone(),
                        name: call.name.clone(),
                    })
                    .await?;
                    while let Some((stream, chunk)) = delta_rx.recv().await {
                        on_event(AgentEvent::ToolOutputDelta {
                            call_id: call.call_id.clone(),
                            stream,
                            chunk,
                        })
                        .await?;
                    }
                    let output = timed_out_tool_result(&call.name, timeout_secs);
                    on_event(AgentEvent::ToolCallFinished {
                        call_id: call.call_id.clone(),
                        name: call.name.clone(),
                        ok: false,
                        output: output.clone(),
                    })
                    .await?;
                    return Ok(ToolExecutionRecord::new(
                        call,
                        Some(args),
                        permission_class,
                        directive,
                        ToolExecutionStatus::TimedOut,
                        None,
                        output,
                    ));
                }
            };
            while let Some((stream, chunk)) = delta_rx.recv().await {
                on_event(AgentEvent::ToolOutputDelta {
                    call_id: call.call_id.clone(),
                    stream,
                    chunk,
                })
                .await?;
            }
            output
        };

        if output.ok && call.name == tool_names::TOOL_CONTEXT_RETURN {
            let writes_observed = agent
                .context_scope_state
                .lock()
                .map_err(|_| anyhow::anyhow!("context scope state poisoned"))?
                .active_experiment
                .as_ref()
                .is_some_and(|experiment| experiment.writes_observed);
            if writes_observed {
                if let Some(data) = output.data.as_mut() {
                    data["warning"] =
                        Value::String("Context restored, files were NOT reverted".to_string());
                    if let Some(message) = data.get("message").and_then(Value::as_str) {
                        data["message"] = Value::String(format!(
                            "{message} Context restored, files were NOT reverted."
                        ));
                    }
                }
            }
        }

        if output.ok {
            agent
                .apply_control_tool_state(&call.name, &args, on_event)
                .await?;
        }

        on_event(AgentEvent::ToolCallFinished {
            call_id: call.call_id.clone(),
            name: call.name.clone(),
            ok: output.ok,
            output: output.clone(),
        })
        .await?;

        Ok(ToolExecutionRecord::new(
            call,
            Some(args),
            permission_class,
            directive,
            ToolExecutionStatus::Executed,
            None,
            output,
        ))
    } else {
        let output = if matches!(permission_decision, PermissionDecision::Deny) {
            ToolResult::err(&call.name, "permission denied by current mode")
        } else {
            ToolResult::err(&call.name, "user denied permission")
        };
        let rejection = if matches!(permission_decision, PermissionDecision::Deny) {
            ToolExecutionRejection::PermissionDeniedByPolicy
        } else {
            ToolExecutionRejection::PermissionDeniedByUser
        };
        let record = ToolExecutionRecord::new(
            call,
            Some(args),
            permission_class,
            directive,
            ToolExecutionStatus::Rejected,
            Some(rejection),
            output,
        );
        emit_finished(on_event, call, &record).await?;
        Ok(record)
    }
}

fn non_shell_tool_timeout_secs<C: Config>(agent: &Agent<C>, tool_name: &str) -> Option<u64> {
    (tool_name != tool_names::TOOL_SHELL_EXEC && !is_subagent_tool_name(tool_name))
        .then_some(agent.tool_timeout_secs)
        .flatten()
}

fn timed_out_tool_result(tool_name: &str, timeout_secs: u64) -> ToolResult {
    ToolResult::err_with_data(
        tool_name,
        format!("tool timed out after {timeout_secs}s"),
        serde_json::json!({
            "status": "timed_out",
            "timeout_secs": timeout_secs,
        }),
    )
}

async fn invalid_json_record<C, E, Efut>(
    agent: &Agent<C>,
    call: &HistoryToolCall,
    err: serde_json::Error,
    on_event: &mut E,
) -> Result<ToolExecutionRecord>
where
    C: Config,
    E: FnMut(AgentEvent) -> Efut,
    Efut: Future<Output = Result<()>>,
{
    warn!(
        tool_name = %call.name,
        call_id = %call.call_id,
        error = %err,
        raw_arguments = %call.arguments_json,
        "invalid tool call JSON arguments"
    );
    let output = ToolResult::err(
        &call.name,
        format!(
            "invalid JSON arguments: {err}; raw: {}",
            call.arguments_json
        ),
    );
    on_event(AgentEvent::ToolCallFinished {
        call_id: call.call_id.clone(),
        name: call.name.clone(),
        ok: false,
        output: output.clone(),
    })
    .await?;
    Ok(ToolExecutionRecord::new(
        call,
        None,
        permission_class_for_tool_call(&agent.tools, &call.name),
        agent.turn.policy.directive,
        ToolExecutionStatus::Rejected,
        Some(ToolExecutionRejection::InvalidJsonArguments),
        output,
    ))
}

async fn emit_finished<E, Efut>(
    on_event: &mut E,
    call: &HistoryToolCall,
    record: &ToolExecutionRecord,
) -> Result<()>
where
    E: FnMut(AgentEvent) -> Efut,
    Efut: Future<Output = Result<()>>,
{
    on_event(AgentEvent::ToolCallFinished {
        call_id: call.call_id.clone(),
        name: call.name.clone(),
        ok: record.output.ok,
        output: record.output.clone(),
    })
    .await
}
