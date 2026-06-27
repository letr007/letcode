use anyhow::Result;
use serde_json::Value;
use std::future::Future;

use super::*;

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
    Afut: Future<Output = Result<bool>>,
{
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
    Afut: Future<Output = Result<bool>>,
{
    let directive = agent.turn.policy.directive;
    let permission_class = permission_class_for_tool_call(&agent.tools, &call.name);

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
    let base_permission_decision = agent.permission_policy.check_class_with_directive(
        &call.name,
        &args,
        permission_class,
        directive,
    );
    let permission_decision = match (base_permission_decision, external_workspace_access.as_ref()) {
        (PermissionDecision::Allow, Some(_)) => PermissionDecision::Ask,
        (decision, _) => decision,
    };
    let should_execute = if is_workflow_control_tool(&call.name) {
        true
    } else {
        match permission_decision {
            PermissionDecision::Allow => true,
            PermissionDecision::Ask => {
                approve(PermissionRequest {
                    call_id: Some(call.call_id.clone()),
                    tool: call.name.clone(),
                    args: args.clone(),
                    class: permission_class,
                    summary: format_tool_call(&call.name, &args),
                    preview: external_workspace_access
                        .as_ref()
                        .map(|access| access.preview()),
                })
                .await?
            }
            PermissionDecision::Deny => false,
        }
    };

    if should_execute {
        on_event(AgentEvent::ToolCallStarted {
            call_id: call.call_id.clone(),
            name: call.name.clone(),
            args: args.clone(),
        })
        .await?;

        let mut output = if is_subagent_tool_name(&call.name) {
            agent.execute_subagent_tool(&call.name, &args).await
        } else if external_workspace_access.is_some() {
            agent
                .tools
                .call_with_context(
                    &call.name,
                    args.clone(),
                    ToolExecutionContext::outside_workspace_granted(),
                )
                .await
        } else {
            agent.tools.call(&call.name, args.clone()).await
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
                    data["warning"] = Value::String(
                        "Context restored, files were NOT reverted".to_string(),
                    );
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

        if output.ok && call.name == tool_names::TOOL_CONTEXT_CHECKPOINT {
            agent.finalize_context_checkpoint_after_recording()?;
        }
        if output.ok && call.name == tool_names::TOOL_CONTEXT_RETURN {
            agent.finalize_context_return_after_recording(&output)?;
        }

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
