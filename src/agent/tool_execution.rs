use anyhow::Result;
use futures_util::future::join_all;
use serde_json::Value;
use std::future::Future;
use std::sync::Mutex;
use std::time::Duration;
use tracing::Instrument;

use super::*;
use crate::langfuse_trace;

pub(super) async fn execute_tool_call<E, A, Efut, Afut>(
    agent: &mut Agent,
    call: &HistoryToolCall,
    on_event: &mut E,
    approve: &mut A,
) -> Result<ToolExecutionRecord>
where
    E: FnMut(AgentEvent) -> Efut,
    A: FnMut(PermissionRequest) -> Afut,
    Efut: Future<Output = Result<()>>,
    Afut: Future<Output = Result<PermissionApproval>>,
{
    // Anchored bootstrap alias resolution: the model may have called an alias
    // name (bash / str_replace_editor). Resolve once here so permissions,
    // execution, and records all use the real tool name.
    let resolved_call = {
        let mut resolved = call.clone();
        resolved.name = agent.resolve_tool_alias(&call.name);
        resolved
    };
    let call = &resolved_call;
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
        Agent::emit_audit_event(
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

pub(super) async fn execute_parallel_tool_call_batch<E, Efut>(
    agent: &mut Agent,
    calls: &[HistoryToolCall],
    on_event: &mut E,
) -> Result<Vec<ParallelBatchRecord>>
where
    E: FnMut(AgentEvent) -> Efut,
    Efut: Future<Output = Result<()>>,
{
    let mut prepared = Vec::with_capacity(calls.len());
    for call in calls {
        // Resolve anchored-bootstrap aliases so preflight checks run against
        // the real tool name (permissions, parallelism, scope).
        let call = {
            let mut resolved = call.clone();
            resolved.name = agent.resolve_tool_alias(&call.name);
            resolved
        };
        let args = serde_json::from_str::<Value>(&call.arguments_json)
            .map_err(|error| anyhow::anyhow!("parallel tool preflight changed: {error}"))?;
        let permission_class = permission_class_for_tool_call(&agent.tools, &call.name);
        if agent.tools.parallelism(&call.name) != ToolParallelism::Parallel
            || !is_executable_tool(agent, &call.name)
            || !agent.tools.scope().allows_tool(&call.name)
            || restricted_by_directive_with_class(
                &call.name,
                &args,
                permission_class,
                agent.turn.policy.directive,
            )
            .is_some()
            || external_workspace_access_for_tool(&call.name, &args).is_some()
        {
            return Err(anyhow::anyhow!(
                "parallel tool preflight changed for '{}'",
                call.name
            ));
        }
        let context = agent.tool_execution_context_for(&call.name, false)?;
        prepared.push((call.clone(), args, permission_class, context));
    }

    let permission_generation = {
        let state = agent
            .permission_session
            .lock()
            .map_err(|_| anyhow::anyhow!("permission session poisoned"))?;
        let mut permission_generation = None;
        for (call, args, permission_class, _) in &prepared {
            let (mode, generation, decision, grant_allowed) = state.approval_snapshot(
                crate::tool::permission_resource_for_tool(&call.name, args).as_ref(),
                &call.name,
                args,
                *permission_class,
                agent.turn.policy.directive,
                false,
                crate::permission::is_internal_tool(&call.name),
            );
            let decision = if mode.supports_session_grants() && grant_allowed {
                PermissionDecision::Allow
            } else {
                decision
            };
            if decision != PermissionDecision::Allow {
                return Err(anyhow::anyhow!(
                    "parallel tool '{}' unexpectedly requires approval",
                    call.name
                ));
            }
            match permission_generation {
                Some(expected) if expected != generation => {
                    return Err(anyhow::anyhow!("parallel permission preflight changed"));
                }
                None => permission_generation = Some(generation),
                _ => {}
            }
        }
        permission_generation
    };

    // Do not expose any call as started until every call in this batch has
    // completed structural and permission preflight successfully.
    for (index, (call, args, _, _)) in prepared.iter().enumerate() {
        if let Err(error) = on_event(AgentEvent::ToolCallStarted {
            call_id: call.call_id.clone(),
            name: call.name.clone(),
            args: args.clone(),
        })
        .await
        {
            cancel_parallel_calls_best_effort(
                prepared[..=index]
                    .iter()
                    .map(|(call, _, _, _)| (call.call_id.clone(), call.name.clone()))
                    .collect(),
                on_event,
            )
            .await;
            return Err(error);
        }
    }

    let changed_permission_call = {
        let state = agent
            .permission_session
            .lock()
            .map_err(|_| anyhow::anyhow!("permission session poisoned"))?;
        prepared
            .iter()
            .find_map(|(call, args, permission_class, _)| {
                let (mode, generation, decision, grant_allowed) = state.approval_snapshot(
                    crate::tool::permission_resource_for_tool(&call.name, args).as_ref(),
                    &call.name,
                    args,
                    *permission_class,
                    agent.turn.policy.directive,
                    false,
                    crate::permission::is_internal_tool(&call.name),
                );
                let decision = if mode.supports_session_grants() && grant_allowed {
                    PermissionDecision::Allow
                } else {
                    decision
                };
                (Some(generation) != permission_generation || decision != PermissionDecision::Allow)
                    .then(|| call.name.clone())
            })
    };
    if let Some(call_name) = changed_permission_call {
        cancel_parallel_calls_best_effort(
            prepared
                .iter()
                .map(|(call, _, _, _)| (call.call_id.clone(), call.name.clone()))
                .collect(),
            on_event,
        )
        .await;
        return Err(anyhow::anyhow!(
            "parallel permission preflight changed for '{call_name}'"
        ));
    }

    let tools = agent.tools.clone();
    let timeout_secs = agent.tool_timeout_secs;
    let directive = agent.turn.policy.directive;
    let turn_id = agent.turn.turn_id;
    let records = join_all(
        prepared
            .into_iter()
            .map(|(call, args, permission_class, context)| {
                let tools = tools.clone();
                let completion = ToolSpanCompletion::new(langfuse_trace::tool_span(
                    turn_id,
                    &call.name,
                    &call.call_id,
                    call.arguments_json.len(),
                ));
                let span = completion.span();
                async move {
                    let tool_timeout = non_shell_tool_timeout_secs(timeout_secs, &call.name);
                    let output = if let Some(timeout_secs) = tool_timeout {
                        match tokio::time::timeout(
                            Duration::from_secs(timeout_secs),
                            tools.call_with_context(&call.name, args.clone(), context),
                        )
                        .await
                        {
                            Ok(output) => output,
                            Err(_) => timed_out_tool_result(&call.name, timeout_secs),
                        }
                    } else {
                        tools
                            .call_with_context(&call.name, args.clone(), context)
                            .await
                    };
                    let status = if output
                        .data
                        .as_ref()
                        .and_then(|data| data.get("status"))
                        .and_then(Value::as_str)
                        == Some("timed_out")
                    {
                        ToolExecutionStatus::TimedOut
                    } else {
                        ToolExecutionStatus::Executed
                    };
                    ParallelBatchRecord {
                        record: ToolExecutionRecord::new(
                            &call,
                            Some(args),
                            permission_class,
                            directive,
                            status,
                            None,
                            output,
                        ),
                        completion,
                    }
                }
                .instrument(span)
            }),
    )
    .await;

    Ok(records)
}

pub(super) async fn cancel_parallel_calls_best_effort<E, Efut>(
    calls: Vec<(String, String)>,
    on_event: &mut E,
) where
    E: FnMut(AgentEvent) -> Efut,
    Efut: Future<Output = Result<()>>,
{
    for (call_id, name) in calls {
        let _ = on_event(AgentEvent::ToolCallCancelled { call_id, name }).await;
    }
}

pub(super) async fn finalize_parallel_tool_call<E, Efut>(
    agent: &mut Agent,
    record: &ToolExecutionRecord,
    on_event: &mut E,
) -> Result<()>
where
    E: FnMut(AgentEvent) -> Efut,
    Efut: Future<Output = Result<()>>,
{
    if record.status == ToolExecutionStatus::TimedOut {
        on_event(AgentEvent::ToolCallCancelled {
            call_id: record.call_id.clone(),
            name: record.tool_name.clone(),
        })
        .await?;
    }
    on_event(AgentEvent::ToolCallFinished {
        call_id: record.call_id.clone(),
        name: record.tool_name.clone(),
        ok: record.output.ok,
        output: record.output.clone(),
    })
    .await?;
    agent.record_tool_effects(record);
    Agent::emit_audit_event(
        on_event,
        AgentEvent::ToolExecutionSummary(agent.tool_execution_summary_event(record)),
        "tool_execution_summary",
    )
    .await;
    Ok(())
}

enum SubagentPreflight {
    Admitted(Value),
    Rejected(ToolExecutionRecord),
}

struct ToolSpanCompletion {
    span: Mutex<Option<tracing::Span>>,
}

impl ToolSpanCompletion {
    fn new(span: tracing::Span) -> Self {
        Self {
            span: Mutex::new(Some(span)),
        }
    }

    fn span(&self) -> tracing::Span {
        self.span
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .as_ref()
            .expect("unfinished subagent span is present")
            .clone()
    }

    fn finish(&self, result: Result<ToolExecutionRecord>) {
        let span = self
            .span
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .take();
        if let Some(span) = span {
            langfuse_trace::finish_tool_span(&span, &result);
        }
    }

    fn finish_error(&self, message: &'static str) {
        self.finish(Err(anyhow::anyhow!(message)));
    }
}

impl Drop for ToolSpanCompletion {
    fn drop(&mut self) {
        let span = self
            .span
            .get_mut()
            .unwrap_or_else(|error| error.into_inner())
            .take();
        if let Some(span) = span {
            langfuse_trace::finish_tool_span(
                &span,
                &Err(anyhow::anyhow!(
                    "subagent batch span dropped before reconciliation completed"
                )),
            );
        }
    }
}

pub(super) struct ParallelBatchRecord {
    pub(super) record: ToolExecutionRecord,
    completion: ToolSpanCompletion,
}

impl ParallelBatchRecord {
    pub(super) fn span(&self) -> tracing::Span {
        self.completion.span()
    }

    pub(super) fn finish(&self, reconciliation: &Result<()>) {
        match reconciliation {
            Ok(()) => self.completion.finish(Ok(self.record.clone())),
            Err(_) => self
                .completion
                .finish_error("parallel tool batch reconciliation failed"),
        }
    }
}

pub(super) struct SubagentBatchRecord {
    pub(super) record: ToolExecutionRecord,
    completion: ToolSpanCompletion,
}

impl SubagentBatchRecord {
    pub(super) fn span(&self) -> tracing::Span {
        self.completion.span()
    }

    /// Completes the span only after this record has finished its model-ordered
    /// lifecycle, history, evidence, and cancellation reconciliation.
    pub(super) fn finish(&self, reconciliation: &Result<()>) {
        match reconciliation {
            Ok(()) => self.completion.finish(Ok(self.record.clone())),
            Err(_) => self
                .completion
                .finish_error("subagent batch reconciliation failed"),
        }
    }
}

fn finish_subagent_batch_spans_error(preflight: &[(ToolSpanCompletion, SubagentPreflight)]) {
    for (completion, _) in preflight {
        completion.finish_error("subagent batch execution failed");
    }
}

/// Preflights a contiguous model-ordered subagent batch, then polls admitted
/// delegate futures together without retaining mutable Agent or callback borrows.
pub(super) async fn execute_subagent_tool_call_batch<E, A, Efut, Afut>(
    agent: &mut Agent,
    calls: &[HistoryToolCall],
    on_event: &mut E,
    approve: &mut A,
) -> Result<Vec<SubagentBatchRecord>>
where
    E: FnMut(AgentEvent) -> Efut,
    A: FnMut(PermissionRequest) -> Afut,
    Efut: Future<Output = Result<()>>,
    Afut: Future<Output = Result<PermissionApproval>>,
{
    let mut preflight = Vec::with_capacity(calls.len());
    for call in calls {
        let completion = ToolSpanCompletion::new(langfuse_trace::tool_span(
            agent.turn.turn_id,
            &call.name,
            &call.call_id,
            call.arguments_json.len(),
        ));
        let entry = match preflight_subagent_tool_call(agent, call, approve).await {
            Ok(entry) => entry,
            Err(error) => {
                preflight.push((
                    completion,
                    SubagentPreflight::Rejected(ToolExecutionRecord::new(
                        call,
                        None,
                        permission_class_for_tool_call(&agent.tools, &call.name),
                        agent.turn.policy.directive,
                        ToolExecutionStatus::Rejected,
                        None,
                        ToolResult::err(&call.name, error.to_string()),
                    )),
                ));
                finish_subagent_batch_spans_error(&preflight);
                return Err(error);
            }
        };
        preflight.push((completion, entry));
    }

    // Do not expose an admitted call as started until every call in this
    // concurrent group has completed preflight successfully.
    for ((_, entry), call) in preflight.iter().zip(calls) {
        if let SubagentPreflight::Admitted(args) = entry
            && let Err(error) = on_event(AgentEvent::ToolCallStarted {
                call_id: call.call_id.clone(),
                name: call.name.clone(),
                args: args.clone(),
            })
            .await
        {
            finish_subagent_batch_spans_error(&preflight);
            return Err(error);
        }
    }

    // All admitted futures have only shared Agent borrows. They are fully
    // consumed before state and callbacks are mutably borrowed for reconciliation.
    let outputs = {
        let futures = preflight
            .iter()
            .enumerate()
            .filter_map(|(index, (completion, entry))| {
                let SubagentPreflight::Admitted(args) = entry else {
                    return None;
                };
                let call = &calls[index];
                Some(
                    async {
                        agent
                            .execute_subagent_tool_for_call(
                                &call.name,
                                args,
                                Some(call.call_id.clone()),
                            )
                            .await
                    }
                    .instrument(completion.span()),
                )
            });
        join_all(futures).await
    };

    let mut outputs = outputs.into_iter();
    let mut records = Vec::with_capacity(calls.len());
    for ((completion, entry), call) in preflight.into_iter().zip(calls) {
        let record = match entry {
            SubagentPreflight::Admitted(args) => {
                let output = outputs.next().expect("admitted output is present");
                ToolExecutionRecord::new(
                    call,
                    Some(args),
                    permission_class_for_tool_call(&agent.tools, &call.name),
                    agent.turn.policy.directive,
                    ToolExecutionStatus::Executed,
                    None,
                    output,
                )
            }
            SubagentPreflight::Rejected(record) => record,
        };
        records.push(SubagentBatchRecord { record, completion });
    }

    Ok(records)
}

/// Finalizes one completed subagent record in model order. Callers must record
/// its history, token projection, evidence, and cancellation before finalizing
/// the next record.
pub(super) async fn finalize_subagent_tool_call<E, Efut>(
    agent: &mut Agent,
    call: &HistoryToolCall,
    record: &ToolExecutionRecord,
    on_event: &mut E,
) -> Result<()>
where
    E: FnMut(AgentEvent) -> Efut,
    Efut: Future<Output = Result<()>>,
{
    emit_finished(on_event, call, record).await?;
    agent.record_tool_effects(record);
    Agent::emit_audit_event(
        on_event,
        AgentEvent::ToolExecutionSummary(agent.tool_execution_summary_event(record)),
        "tool_execution_summary",
    )
    .await;
    Ok(())
}

async fn preflight_subagent_tool_call<A, Afut>(
    agent: &mut Agent,
    call: &HistoryToolCall,
    approve: &mut A,
) -> Result<SubagentPreflight>
where
    A: FnMut(PermissionRequest) -> Afut,
    Afut: Future<Output = Result<PermissionApproval>>,
{
    let args = match serde_json::from_str::<Value>(&call.arguments_json) {
        Ok(args) => args,
        Err(error) => {
            warn!(
                tool_name = %call.name,
                call_id = %call.call_id,
                error = %error,
                raw_arguments = %call.arguments_json,
                "invalid tool call JSON arguments"
            );
            let output = ToolResult::err(
                &call.name,
                format!(
                    "invalid JSON arguments: {error}; raw: {}",
                    call.arguments_json
                ),
            );
            return Ok(SubagentPreflight::Rejected(ToolExecutionRecord::new(
                call,
                None,
                permission_class_for_tool_call(&agent.tools, &call.name),
                agent.turn.policy.directive,
                ToolExecutionStatus::Rejected,
                Some(ToolExecutionRejection::InvalidJsonArguments),
                output,
            )));
        }
    };
    let directive = agent.turn.policy.directive;
    let permission_class = permission_class_for_tool_call(&agent.tools, &call.name);
    if !is_executable_tool(agent, &call.name) {
        let record = ToolExecutionRecord::new(
            call,
            Some(args),
            permission_class,
            directive,
            ToolExecutionStatus::Rejected,
            None,
            ToolResult::err(
                &call.name,
                format!("unknown or unavailable tool: {}", call.name),
            ),
        );
        return Ok(SubagentPreflight::Rejected(record));
    }
    if !agent.tools.scope().allows_tool(&call.name) {
        let record = ToolExecutionRecord::new(
            call,
            Some(args),
            permission_class,
            directive,
            ToolExecutionStatus::Rejected,
            Some(ToolExecutionRejection::ToolScopeDenied),
            ToolResult::err(
                &call.name,
                agent.tools.scope().rejection_message(&call.name),
            ),
        );
        return Ok(SubagentPreflight::Rejected(record));
    }
    if agent.permission_mode() != PermissionMode::Auto
        && let Some(message) =
            restricted_by_directive_with_class(&call.name, &args, permission_class, directive)
    {
        let record = ToolExecutionRecord::new(
            call,
            Some(args),
            permission_class,
            directive,
            ToolExecutionStatus::Rejected,
            Some(ToolExecutionRejection::DirectiveBlocked),
            ToolResult::err(&call.name, message),
        );
        return Ok(SubagentPreflight::Rejected(record));
    }
    let (mode, _generation, decision, grant_allowed) = {
        let state = agent
            .permission_session
            .lock()
            .map_err(|_| anyhow::anyhow!("permission session poisoned"))?;
        state.approval_snapshot(
            None,
            &call.name,
            &args,
            permission_class,
            directive,
            false,
            crate::permission::is_internal_tool(&call.name),
        )
    };
    let decision = if mode.supports_session_grants() && grant_allowed {
        PermissionDecision::Allow
    } else {
        decision
    };
    let mut auto_deny_reason = None;
    let allowed = match decision {
        PermissionDecision::Allow => true,
        PermissionDecision::Ask => {
            let request = PermissionRequest {
                call_id: Some(call.call_id.clone()),
                tool: call.name.clone(),
                args: args.clone(),
                class: permission_class,
                directive,
                summary: format_tool_call(&call.name, &args),
                preview: None,
                can_allow_always: false,
                grant_summary: None,
            };
            let approval = if mode == PermissionMode::Auto {
                let resolution = agent.resolve_auto_permission(request, None).await?;
                if !resolution.approval.allowed() {
                    auto_deny_reason = Some(resolution.reason);
                }
                resolution.approval
            } else {
                approve(request).await?
            };
            approval.allowed()
        }
        PermissionDecision::Deny => false,
    };
    if !allowed {
        let message = if matches!(decision, PermissionDecision::Deny) {
            "permission denied by current mode".to_string()
        } else if let Some(reason) = auto_deny_reason {
            format!("auto-review denied permission: {reason}")
        } else {
            "user denied permission".to_string()
        };
        let record = ToolExecutionRecord::new(
            call,
            Some(args),
            permission_class,
            directive,
            ToolExecutionStatus::Rejected,
            Some(if matches!(decision, PermissionDecision::Deny) {
                ToolExecutionRejection::PermissionDeniedByPolicy
            } else {
                ToolExecutionRejection::PermissionDeniedByUser
            }),
            ToolResult::err(&call.name, message),
        );
        return Ok(SubagentPreflight::Rejected(record));
    }
    Ok(SubagentPreflight::Admitted(args))
}

async fn execute_with_arguments<E, A, Efut, Afut>(
    agent: &mut Agent,
    call: &HistoryToolCall,
    args: Value,
    on_event: &mut E,
    approve: &mut A,
) -> Result<ToolExecutionRecord>
where
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

    if !agent.tools.scope().allows_tool(&call.name) && !is_subagent_control_tool_name(&call.name) {
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

    if agent.permission_mode() != PermissionMode::Auto
        && let Some(message) =
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

    let prepared_writable_leaf = if matches!(
        call.name.as_str(),
        tool_names::TOOL_FS_WRITE | tool_names::TOOL_FS_APPEND
    ) && args.get("path").and_then(Value::as_str).is_some()
        && args.get("content").and_then(Value::as_str).is_some()
    {
        let path = args.get("path").and_then(Value::as_str).expect("checked");
        match crate::tool::prepare_writable_leaf(path) {
            Ok(prepared) => Some(prepared),
            Err(error) => {
                let record = ToolExecutionRecord::new(
                    call,
                    Some(args),
                    permission_class,
                    directive,
                    ToolExecutionStatus::Rejected,
                    None,
                    ToolResult::err(&call.name, error.to_string()),
                );
                emit_finished(on_event, call, &record).await?;
                return Ok(record);
            }
        }
    } else {
        None
    };
    let prepared_apply_patch = if call.name == tool_names::TOOL_EDIT_APPLY_PATCH {
        let prepare_args = args.clone();
        match tokio::task::spawn_blocking(move || {
            crate::tool::prepare_apply_patch_targets(&prepare_args)
        })
        .await?
        {
            Ok(prepared) => Some(prepared),
            Err(error) => {
                let record = ToolExecutionRecord::new(
                    call,
                    Some(args),
                    permission_class,
                    directive,
                    ToolExecutionStatus::Rejected,
                    None,
                    ToolResult::err(&call.name, error.to_string()),
                );
                emit_finished(on_event, call, &record).await?;
                return Ok(record);
            }
        }
    } else {
        None
    };

    let mut delegation_scope_authorized = false;
    if let Some(scope) = agent.subagent_path_scope.as_deref() {
        if let Some(message) = crate::tool::delegation_scope_denial(
            scope,
            &call.name,
            &args,
            prepared_writable_leaf.as_ref(),
            prepared_apply_patch.as_ref(),
        ) {
            let record = ToolExecutionRecord::new(
                call,
                Some(args),
                permission_class,
                directive,
                ToolExecutionStatus::Rejected,
                Some(ToolExecutionRejection::DelegationScopeDenied),
                ToolResult::err(&call.name, message),
            );
            emit_finished(on_event, call, &record).await?;
            return Ok(record);
        }
        if is_delegation_path_scoped_tool(&call.name) {
            delegation_scope_authorized = true;
        }
    }

    let (external_workspace_access, resource) = if let Some(prepared) = &prepared_apply_patch {
        (
            prepared.external_workspace_access(),
            Some(prepared.permission_resource(&call.name)),
        )
    } else {
        (
            prepared_writable_leaf
                .as_ref()
                .and_then(|prepared| prepared.external_workspace_access())
                .or_else(|| {
                    prepared_writable_leaf
                        .is_none()
                        .then(|| external_workspace_access_for_tool(&call.name, &args))
                        .flatten()
                }),
            prepared_writable_leaf
                .as_ref()
                .map(|prepared| prepared.permission_resource(&call.name))
                .or_else(|| crate::tool::permission_resource_for_tool(&call.name, &args)),
        )
    };
    // Reads of letcode's own fold-artifact dirs (folded tool output) are trusted
    // read-only access: they reach workspace-external temp paths but should not
    // prompt. Anything else external still needs approval.
    let needs_external_approval = external_workspace_access.as_ref().is_some_and(|access| {
        !access.paths.is_empty()
            && !access
                .paths
                .iter()
                .all(|path| crate::tool::is_trusted_artifact_path(std::path::Path::new(path)))
    });
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
            needs_external_approval,
            crate::permission::is_internal_tool(&call.name),
        )
    };
    let permission_decision = match permission_decision {
        PermissionDecision::Deny => PermissionDecision::Deny,
        PermissionDecision::Allow => PermissionDecision::Allow,
        PermissionDecision::Ask
            if delegation_scope_authorized || (mode.supports_session_grants() && grant_allowed) =>
        {
            PermissionDecision::Allow
        }
        other => other,
    };
    let mut approval = None;
    let mut auto_deny_reason = None;
    let should_execute = match permission_decision {
        PermissionDecision::Allow => true,
        PermissionDecision::Ask => {
            let can_allow_always = mode.supports_session_grants() && resource.is_some();
            let request = PermissionRequest {
                call_id: Some(call.call_id.clone()),
                tool: call.name.clone(),
                args: args.clone(),
                class: permission_class,
                directive,
                summary: format_tool_call(&call.name, &args),
                preview: external_workspace_access
                    .as_ref()
                    .map(|access| access.preview()),
                can_allow_always,
                grant_summary: can_allow_always
                    .then(|| resource.as_ref().expect("resource checked").summary()),
            };
            let result = if mode == PermissionMode::Auto {
                let resolution = agent.resolve_auto_permission(request, None).await?;
                if !resolution.approval.allowed() {
                    auto_deny_reason = Some(resolution.reason);
                }
                resolution.approval
            } else {
                approve(request).await?
            };
            approval = Some(result);
            result.allowed()
        }
        PermissionDecision::Deny => false,
    };

    if matches!(approval, Some(PermissionApproval::AllowAlways))
        && let Some(resource) = resource
    {
        agent
            .permission_session
            .lock()
            .map_err(|_| anyhow::anyhow!("permission session poisoned"))?
            .grant_if_current_session(permission_generation, resource);
    }

    if should_execute {
        on_event(AgentEvent::ToolCallStarted {
            call_id: call.call_id.clone(),
            name: call.name.clone(),
            args: args.clone(),
        })
        .await?;

        let output =
            if is_subagent_tool_name(&call.name) || is_subagent_control_tool_name(&call.name) {
                agent
                    .execute_subagent_tool_for_call(&call.name, &args, Some(call.call_id.clone()))
                    .await
            } else {
                let mut context = match agent
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
                if let Some(prepared) = prepared_writable_leaf {
                    context.attach_prepared_writable_leaf(prepared);
                }
                if let Some(prepared) = prepared_apply_patch {
                    context.attach_prepared_apply_patch(prepared);
                }
                let (delta_tx, mut delta_rx) = tokio::sync::mpsc::unbounded_channel();
                let timeout_secs = non_shell_tool_timeout_secs(agent.tool_timeout_secs, &call.name);
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

                        loop {
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
        } else if let Some(reason) = auto_deny_reason {
            ToolResult::err(
                &call.name,
                format!("auto-review denied permission: {reason}"),
            )
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

fn non_shell_tool_timeout_secs(tool_timeout_secs: Option<u64>, tool_name: &str) -> Option<u64> {
    (tool_name != tool_names::TOOL_SHELL_EXEC
        && tool_name != tool_names::TOOL_QUESTION
        && !is_subagent_tool_name(tool_name)
        && !is_subagent_control_tool_name(tool_name))
    .then_some(tool_timeout_secs)
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

async fn invalid_json_record<E, Efut>(
    agent: &Agent,
    call: &HistoryToolCall,
    err: serde_json::Error,
    on_event: &mut E,
) -> Result<ToolExecutionRecord>
where
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn question_is_exempt_from_global_tool_timeout() {
        assert_eq!(
            non_shell_tool_timeout_secs(Some(60), tool_names::TOOL_QUESTION),
            None
        );
    }

    #[test]
    fn ordinary_non_shell_tool_inherits_global_tool_timeout() {
        assert_eq!(
            non_shell_tool_timeout_secs(Some(60), tool_names::TOOL_FS_READ),
            Some(60)
        );
    }
}
