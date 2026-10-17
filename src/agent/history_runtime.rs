//! Incremental Historian orchestration at request boundaries.
use super::*;
use crate::context_history::{
    HistoryApplication, HistoryCompartment, HistorySelection, HistoryTier, select_tiers,
};
use crate::session::historian::HistoryWork;

const HISTORIAN_CHUNK_MIN_TOKENS: u64 = 20_000;
const HISTORIAN_CHUNK_MAX_TOKENS: u64 = 50_000;

/// Tokens one historian call may read. A summarizer call has to stay small
/// enough that one response covers the whole chunk, so the budget comes from the
/// historian route's own window: the producer's window says what the route
/// accepts, not what the model can answer in one pass.
///
/// The floor keeps a region that one response can already cover from being split
/// into several calls: a source image alone is charged several thousand tokens,
/// so a small chunk is a handful of messages and pays a model round trip each.
/// The ceiling keeps the derived share from growing without bound on a route
/// whose window is far larger than any single response needs to be.
fn historian_chunk_tokens(helper: &Agent) -> u64 {
    let budget =
        effective_input_budget_tokens(helper.active_model_metadata(), &helper.tool_definitions());
    (budget / 4).clamp(HISTORIAN_CHUNK_MIN_TOKENS, HISTORIAN_CHUNK_MAX_TOKENS)
}

// Live appends retain process-local frame identities for request replay, but
// their provisional spans are not journal coordinates. History work and retained
// boundaries must come from the recorder's selected projection, without replacing
// the live request state merely to prepare a background task.
fn persisted_history_snapshot(agent: &Agent) -> Result<RuntimeSnapshot> {
    let provider = agent
        .runtime_snapshot_provider
        .as_ref()
        .ok_or_else(|| anyhow!("history requires a persisted runtime snapshot provider"))?;
    let snapshot = provider()?;
    anyhow::ensure!(
        snapshot.session_id == agent.runtime_snapshot.session_id
            && snapshot.active_context.branch_id == agent.runtime_snapshot.active_context.branch_id
            && snapshot.context_scope_revision == agent.runtime_snapshot.context_scope_revision,
        "history source scope changed before preparation: active session={:?}, branch={}, revision={}; persisted session={:?}, branch={}, revision={}",
        agent.runtime_snapshot.session_id,
        agent.runtime_snapshot.active_context.branch_id,
        agent.runtime_snapshot.context_scope_revision,
        snapshot.session_id,
        snapshot.active_context.branch_id,
        snapshot.context_scope_revision
    );
    Ok(snapshot)
}

pub(super) fn work(
    agent: &Agent,
    manual: bool,
    pass_limit: Option<usize>,
) -> Result<Option<HistoryWork>> {
    let snapshot = persisted_history_snapshot(agent)?;
    let frames = snapshot.active_protocol_frames();
    let history = crate::protocol_frames::history_items_from_frames(&frames);
    let current_turn_start_index = Agent::current_turn_start_index_for_snapshot(&snapshot);
    let analysis =
        crate::protocol_frames::analyze_history_items(&history, current_turn_start_index)?;
    let budget =
        effective_input_budget_tokens(agent.active_model_metadata(), &agent.tool_definitions());
    let preserve = if manual {
        0
    } else {
        agent
            .compaction_config
            .preserve_recent_tokens
            .unwrap_or_else(|| compaction::default_preserve_recent_budget(budget))
    };
    let Some(cut) = history_compact::plan_turn_cut_with_transcript(
        &history,
        current_turn_start_index,
        preserve,
        &analysis,
    )?
    else {
        return Ok(None);
    };
    let start = cut.cut_end - cut.prefix.len();
    let helper = AgentFactory::create_child_with_route_and_max_tool_calls(
        agent,
        &AgentTemplate::historian(),
        None,
        false,
        Some(0),
    )?;
    let route = helper
        .resolved_model_route()
        .ok_or_else(|| anyhow!("historian requires a resolved route"))?;
    let session_id = snapshot
        .session_id
        .clone()
        .ok_or_else(|| anyhow!("historian requires a persisted session"))?;
    let archive = &snapshot.history_archive;
    let references: Vec<_> = archive
        .publication_order
        .iter()
        .rev()
        .filter_map(|id| archive.publications.get(id))
        .flat_map(|p| p.compartments.iter().rev())
        .take(6)
        .map(|c| serde_json::json!({"title":c.title,"compact":c.compact,"importance":c.importance}))
        .collect();
    let mut facts = Vec::new();
    let mut fact_tokens = 0;
    let external_fact_ids = Vec::new();
    for evidence in snapshot
        .evidence
        .iter()
        .rev()
        .filter(|e| e.tags.iter().any(|t| t == "historian_fact"))
    {
        let item = serde_json::json!({"id":evidence.id.strip_prefix("fact:").unwrap_or(&evidence.id),"text":evidence.summary});
        let cost = (item.to_string().len() as u64).div_ceil(3);
        if fact_tokens + cost <= crate::request_builder::evidence_budget_tokens(budget) {
            facts.push(item);
            fact_tokens += cost;
        }
    }
    let mut end = cut.cut_end;
    let chunk_tokens = historian_chunk_tokens(&helper);
    let mut tokens = 0u64;
    let mut requested = start;
    for (offset, item) in history[start..cut.cut_end].iter().enumerate() {
        tokens = tokens.saturating_add(crate::request_builder::estimate_history_item_tokens(item));
        // A single message that exceeds the budget is still sent whole: the host
        // cannot cut inside a message.
        if tokens > chunk_tokens && offset > 0 {
            break;
        }
        requested = start + offset + 1;
    }
    if requested > start {
        let capped = crate::protocol_frames::canonical_compaction_boundary_with_transcript(
            &analysis, requested,
        )?;
        if capped > start && capped < end {
            end = capped;
        }
    }
    // A provider rejection for an oversized request is authoritative: no later
    // local preflight can prove the window is large enough, so bound the prefix
    // by the last length this session has not yet seen rejected.
    if let Some(limit) = pass_limit {
        let requested = start.saturating_add(limit).min(end);
        let capped = crate::protocol_frames::canonical_compaction_boundary_with_transcript(
            &analysis, requested,
        )?;
        if capped > start {
            end = capped;
        }
    }
    // Use the configured Historian route's actual preparation budget. Never
    // retire a message which did not fit into the producer input.
    let mut prelude = helper.prelude.clone();
    super::prepend_fake_claude_client_blocks(&helper, &route.protocol_id, &mut prelude);
    while end > start {
        let source_ids = frames[start..end]
            .iter()
            .map(|f| {
                f.source_provenance
                    .as_ref()
                    .and_then(|p| p.source_id.clone())
                    .ok_or_else(|| anyhow!("historian source has no durable identity"))
            })
            .collect::<Result<Vec<_>>>()?;
        let input = crate::historian::history_input(&history[start..end], &references, &facts);
        match protocol_stream::prepare_resolved_oneshot_request(
            route,
            helper.active_model_metadata(),
            &prelude,
            &input,
            None,
        ) {
            Ok(_) => {
                let identity = format!(
                    "{}:{}:{}:{}",
                    session_id,
                    snapshot.active_context.branch_id,
                    snapshot.context_scope_revision,
                    source_ids.join("|")
                );
                let id = crate::request_builder::sha256_hex(identity.as_bytes()).to_string();
                return Ok(Some(HistoryWork {
                    id,
                    session_id,
                    branch_id: snapshot.active_context.branch_id.clone(),
                    revision: snapshot.context_scope_revision,
                    source_ids,
                    input,
                    project_path: crate::tool::workspace_root_for_subagent_lock()?
                        .to_string_lossy()
                        .into_owned(),
                    external_fact_ids: external_fact_ids.clone(),
                }));
            }
            Err(error)
                if compaction::is_recognized_request_budget_overflow(&error)
                    || error.to_string().contains("input budget") =>
            {
                end = crate::protocol_frames::canonical_compaction_boundary_with_transcript(
                    &analysis,
                    end - 1,
                )?;
            }
            Err(error) => return Err(error),
        }
    }
    Ok(None)
}

fn pending_application(agent: &Agent, hard: bool) -> Result<Option<HistoryApplication>> {
    let snapshot = persisted_history_snapshot(agent)?;
    let archive = &snapshot.history_archive;
    let frames = snapshot.active_protocol_frames();
    let raw: Vec<_> = frames
        .iter()
        .filter(|f| !matches!(f.item, HistoryItem::ContextSummary { .. }))
        .collect();
    let mut offset = 0;
    let mut publication_ids = Vec::new();
    let mut new_compartments = Vec::new();
    while let Some(first) = raw
        .get(offset)
        .and_then(|f| f.source_provenance.as_ref())
        .and_then(|p| p.source_id.as_ref())
    {
        let Some(p) = archive
            .publications
            .values()
            .find(|p| !archive.applied_ids.contains(&p.id) && p.source_ids.first() == Some(first))
        else {
            break;
        };
        let actual: Vec<_> = raw
            .iter()
            .skip(offset)
            .take(p.source_ids.len())
            .filter_map(|f| {
                f.source_provenance
                    .as_ref()
                    .and_then(|p| p.source_id.as_ref())
            })
            .collect();
        anyhow::ensure!(
            actual.iter().copied().eq(p.source_ids.iter()),
            "pending historian source no longer matches the active prefix"
        );
        offset += p.source_ids.len();
        publication_ids.push(p.id.clone());
        new_compartments.extend(p.compartments.iter().cloned());
    }
    if publication_ids.is_empty() && (!hard || archive.application.is_none()) {
        return Ok(None);
    }
    let first_kept_entry_id = raw
        .get(offset)
        .and_then(|f| f.source_provenance.as_ref())
        .and_then(|p| p.source_id.clone());
    let input_budget =
        effective_input_budget_tokens(agent.active_model_metadata(), &agent.tool_definitions());
    let history_budget = (input_budget / 5)
        .min(agent.history_budget_limit.unwrap_or(u64::MAX))
        .saturating_sub(128);
    let previous = archive.application.as_ref();
    let legacy_summary = match previous {
        Some(a) => a.legacy_summary.clone(),
        None => {
            let legacy = frames
                .iter()
                .filter_map(|f| match &f.item {
                    HistoryItem::ContextSummary { text } => Some(text.as_str()),
                    _ => None,
                })
                .collect::<Vec<_>>()
                .join("\n\n");
            (!legacy.is_empty()).then_some(legacy)
        }
    };
    let mut baseline = previous.map(|a| a.baseline.clone()).unwrap_or_default();
    let mut delta = previous.map(|a| a.delta.clone()).unwrap_or_default();
    delta.extend(new_compartments.iter().map(|c| HistorySelection {
        compartment_id: c.id.clone(),
        tier: HistoryTier::Detailed,
    }));
    let mut baseline_fact_ids = previous
        .map(|a| a.baseline_fact_ids.clone())
        .unwrap_or_default();
    let mut delta_fact_ids = previous
        .map(|a| a.delta_fact_ids.clone())
        .unwrap_or_default();
    let mut withdrawn_fact_ids = previous
        .map(|a| a.withdrawn_fact_ids.clone())
        .unwrap_or_default();
    for id in &publication_ids {
        let publication = &archive.publications[id];
        delta_fact_ids.extend(publication.facts.iter().map(|f| f.id.clone()));
        withdrawn_fact_ids.extend(publication.withdrawn_fact_ids.iter().cloned());
    }
    let delta_tokens = (archive.render(&delta).len() as u64).div_ceil(3)
        + (archive.render_facts(&delta_fact_ids).len() as u64).div_ceil(3);
    if hard || previous.is_none() || delta_tokens > history_budget / 5 {
        baseline_fact_ids.clear();
        let mut fact_cost = 0;
        for id in archive.effective_fact_ids(&publication_ids) {
            let cost = (archive.render_facts(std::slice::from_ref(&id)).len() as u64).div_ceil(3);
            if fact_cost + cost
                <= crate::request_builder::evidence_budget_tokens(input_budget).min(history_budget)
            {
                fact_cost += cost;
                baseline_fact_ids.push(id);
            }
        }
        delta_fact_ids.clear();
        withdrawn_fact_ids.clear();
        // Re-materialize from the full applied archive, including episodes
        // previously omitted at a smaller budget. Never decay a decay result.
        let compartments: Vec<HistoryCompartment> = archive
            .publication_order
            .iter()
            .filter(|id| archive.applied_ids.contains(*id) || publication_ids.contains(*id))
            .filter_map(|id| archive.publications.get(id))
            .flat_map(|p| p.compartments.iter().cloned())
            .collect();
        baseline = select_tiers(
            &compartments,
            history_budget.saturating_sub(fact_cost).saturating_sub(
                legacy_summary
                    .as_ref()
                    .map(|s| (s.len() as u64).div_ceil(3))
                    .unwrap_or(0),
            ),
        );
        delta.clear();
    }
    if publication_ids.is_empty()
        && previous.is_some_and(|p| {
            p.baseline == baseline
                && p.delta == delta
                && p.baseline_fact_ids == baseline_fact_ids
                && p.delta_fact_ids == delta_fact_ids
                && p.withdrawn_fact_ids == withdrawn_fact_ids
        })
    {
        return Ok(None);
    }
    Ok(Some(HistoryApplication {
        publication_ids,
        baseline,
        delta,
        baseline_fact_ids,
        delta_fact_ids,
        withdrawn_fact_ids,
        first_kept_entry_id,
        legacy_summary,
    }))
}

async fn apply_history<E, Efut>(
    agent: &mut Agent,
    application: HistoryApplication,
    blocking: bool,
    on_event: &mut E,
) -> Result<()>
where
    E: FnMut(AgentEvent) -> Efut + Send + ?Sized,
    Efut: Future<Output = Result<()>> + Send,
{
    on_event(AgentEvent::HistoryApplied {
        application,
        revision: agent.runtime_snapshot.context_scope_revision,
        blocking,
    })
    .await?;
    agent.reload_runtime_snapshot_from_provider()?;
    agent.clear_active_epoch();
    agent.clear_provider_usage_anchor();
    // The retired conversation is replaced by summary text. Workflow state is
    // snapshot state rather than transcript text, so restate it for the model;
    // without this the todos the session is working through disappear here.
    if let Some(text) = agent.runtime_snapshot.workflow.render_compaction_reminder()
        && !matches!(
            agent.active_history_items().last(),
            Some(HistoryItem::InternalContinuation { text: last }) if last == &text
        )
    {
        agent.append_history_item(HistoryItem::internal_continuation(text.clone()))?;
        on_event(AgentEvent::InternalContinuation {
            text,
            source: crate::transcript::InternalContinuationSource::CompactionReminder,
        })
        .await?;
    }
    Ok(())
}

pub(super) async fn advance<E, Efut>(
    agent: &mut Agent,
    blocking: bool,
    manual: bool,
    on_event: &mut E,
) -> Result<bool>
where
    E: FnMut(AgentEvent) -> Efut + Send + ?Sized,
    Efut: Future<Output = Result<()>> + Send,
{
    if agent.runtime_snapshot_provider.is_none() {
        return Ok(false);
    }
    let runtime = agent.historian_runtime.clone();
    if let Some(runtime) = &runtime {
        let snapshot = &agent.runtime_snapshot;
        match runtime.poll(
            snapshot.session_id.as_deref().unwrap_or(""),
            &snapshot.active_context.branch_id,
            snapshot.context_scope_revision,
        ) {
            Ok(Some(publication)) => {
                agent.reload_runtime_snapshot_from_provider()?;
                agent.last_historian_work = Some(publication.id);
            }
            Ok(None) => {}
            Err(error) => {
                tracing::warn!(error=%error,"historian unavailable; raw context remains active")
            }
        }
    }
    let input_budget =
        effective_input_budget_tokens(agent.active_model_metadata(), &agent.tool_definitions());
    let raw_tokens: u64 = agent
        .active_history_items()
        .iter()
        .map(crate::request_builder::estimate_history_item_tokens)
        .sum();
    let execute = blocking || raw_tokens >= input_budget.saturating_mul(65) / 100;
    if execute && let Some(application) = pending_application(agent, blocking)? {
        apply_history(agent, application, blocking, on_event).await?;
        return Ok(true);
    }
    if !execute {
        return Ok(false);
    }
    if let Some(runtime) = &runtime {
        if !blocking {
            if let Some(failure) = runtime.failure()
                && failure.class.shrinks()
            {
                runtime.reduce_pass_limit(failure.source_len);
            }
            if !runtime.is_running() {
                let Some(work) = work(agent, manual, runtime.pass_limit())? else {
                    return Ok(false);
                };
                let first_source_id = work.source_ids.first().map(String::as_str);
                if agent.last_historian_work.as_ref() == Some(&work.id)
                    || runtime.exhausted(first_source_id)
                {
                    return Ok(false);
                }
                runtime.start(agent, work)?;
            }
            return Ok(false);
        }
        let session = agent
            .runtime_snapshot
            .session_id
            .clone()
            .unwrap_or_default();
        let branch = agent.runtime_snapshot.active_context.branch_id.clone();
        let revision = agent.runtime_snapshot.context_scope_revision;
        // A provider rejection for an oversized request is authoritative: the
        // declared window cannot be trusted, so shrink the prefix and retry
        // within this bounded preparation round rather than failing the turn.
        let mut shrink_rounds = 0u8;
        loop {
            if !runtime.is_running() {
                if let Some(failure) = runtime.failure()
                    && failure.class.shrinks()
                {
                    runtime.reduce_pass_limit(failure.source_len);
                }
                let Some(work) = work(agent, manual, runtime.pass_limit())? else {
                    return Ok(false);
                };
                if runtime.exhausted(work.source_ids.first().map(String::as_str)) {
                    let detail = runtime
                        .failure()
                        .map(|failure| failure.message)
                        .unwrap_or_else(|| "unknown failure".to_string());
                    anyhow::bail!("historian history preparation failed: {detail}");
                }
                runtime.start(agent, work)?;
            }
            let outcome = tokio::time::timeout(std::time::Duration::from_secs(610), async {
                loop {
                    if let Some(publication) = runtime.poll(&session, &branch, revision)? {
                        return Ok::<_, anyhow::Error>(publication);
                    }
                    if !runtime.is_running() {
                        anyhow::bail!("historian stopped before publishing");
                    }
                    tokio::time::sleep(std::time::Duration::from_millis(25)).await;
                }
            })
            .await
            .map_err(|_| anyhow!("historian completion timed out"))?;
            match outcome {
                Ok(publication) => {
                    agent.last_historian_work = Some(publication.id);
                    break;
                }
                Err(error) => {
                    let shrinkable = runtime
                        .failure()
                        .is_some_and(|failure| failure.class.shrinks());
                    if shrinkable && shrink_rounds < 6 {
                        if let Some(failure) = runtime.failure() {
                            runtime.reduce_pass_limit(failure.source_len);
                        }
                        shrink_rounds += 1;
                        continue;
                    }
                    return Err(error);
                }
            }
        }
        agent.reload_runtime_snapshot_from_provider()?;
    } else {
        // Child-session and direct runner paths use the same producer contract
        // synchronously, without constructing recursive Historian sessions.
        let Some(work) = work(agent, manual, None)? else {
            return Ok(false);
        };
        let helper = AgentFactory::create_child_with_route_and_max_tool_calls(
            agent,
            &AgentTemplate::historian(),
            None,
            false,
            Some(0),
        )?;
        let crate::historian::HistorianRun {
            mut publication, ..
        } = crate::historian::run_historian_with_repair(
            &helper,
            &work.id,
            &work.source_ids,
            &work.input,
            |_: &str| std::future::ready(Ok::<(), crate::model_runtime::ModelFailure>(())),
            |_| {},
        )
        .await?;
        publication.project_path = Some(work.project_path.clone());
        publication.external_fact_ids = work.external_fact_ids.clone();
        let publication_id = publication.id.clone();
        on_event(AgentEvent::HistoryPublished {
            publication,
            revision: work.revision,
        })
        .await?;
        agent.last_historian_work = Some(publication_id);
        agent.reload_runtime_snapshot_from_provider()?;
    }
    if let Some(application) = pending_application(agent, true)? {
        apply_history(agent, application, blocking, on_event).await?;
        return Ok(true);
    }
    Ok(false)
}

#[cfg(test)]
#[path = "../../tests/internal/agent/history_runtime/tests.rs"]
mod tests;

#[cfg(test)]
#[path = "../../tests/internal/agent/history_runtime/wire_tests.rs"]
mod wire_tests;

#[cfg(test)]
#[path = "../../tests/internal/agent/history_runtime/live_source_tests.rs"]
mod live_source_tests;
