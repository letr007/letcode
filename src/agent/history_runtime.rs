//! Incremental Historian orchestration at request boundaries.
use super::*;
use crate::context_history::{
    HistoryApplication, HistoryCompartment, HistorySelection, HistoryTier, select_tiers,
};
use crate::session::historian::HistoryWork;

pub(super) fn work(agent: &Agent, manual: bool) -> Result<Option<HistoryWork>> {
    let history = agent.active_history_items();
    let frames = agent.active_protocol_frames();
    let analysis = crate::protocol_frames::analyze_history_items(
        &history,
        agent.turn.current_turn_start_index,
    )?;
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
        agent.turn.current_turn_start_index,
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
    let session_id = agent
        .runtime_snapshot
        .session_id
        .clone()
        .ok_or_else(|| anyhow!("historian requires a persisted session"))?;
    let archive = &agent.runtime_snapshot.history_archive;
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
    let mut external_fact_ids = Vec::new();
    for evidence in agent
        .runtime_snapshot
        .evidence
        .iter()
        .rev()
        .filter(|e| e.tags.iter().any(|t| t == "historian_fact"))
        .chain(agent.turn.recalled_project_facts.iter())
    {
        let item = serde_json::json!({"id":evidence.id.strip_prefix("fact:").unwrap_or(&evidence.id),"text":evidence.summary});
        let cost = (item.to_string().len() as u64).div_ceil(3);
        if fact_tokens + cost <= crate::request_builder::evidence_budget_tokens(budget) {
            facts.push(item);
            if evidence.tags.iter().any(|t| t == "recalled_project_fact") {
                external_fact_ids.push(evidence.id.clone());
            }
            fact_tokens += cost;
        }
    }
    // Use the configured Historian route's actual preparation budget. Never
    // retire a message which did not fit into the producer input.
    let mut end = cut.cut_end;
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
        let new_messages: Vec<_> = history[start..end]
            .iter()
            .enumerate()
            .map(|(index, item)| serde_json::json!({"index":index,"content":item}))
            .collect();
        let prompt = serde_json::json!({"references":references,"project_facts":facts,"new_messages":new_messages}).to_string();
        match protocol_stream::preflight_resolved_oneshot_text_request(
            route,
            helper.active_model_metadata(),
            &helper.prelude,
            &prompt,
        ) {
            Ok(_) => {
                let identity = format!(
                    "{}:{}:{}:{}",
                    session_id,
                    agent.runtime_snapshot.active_context.branch_id,
                    agent.runtime_snapshot.context_scope_revision,
                    source_ids.join("|")
                );
                let id = format!(
                    "{}",
                    crate::request_builder::sha256_hex(identity.as_bytes())
                );
                return Ok(Some(HistoryWork {
                    id,
                    session_id,
                    branch_id: agent.runtime_snapshot.active_context.branch_id.clone(),
                    revision: agent.runtime_snapshot.context_scope_revision,
                    source_ids,
                    prompt,
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
    let archive = &agent.runtime_snapshot.history_archive;
    let frames = agent.active_protocol_frames();
    let raw: Vec<_> = frames
        .iter()
        .filter(|f| !matches!(f.item, HistoryItem::ContextSummary { .. }))
        .collect();
    let mut offset = 0;
    let mut publication_ids = Vec::new();
    let mut new_compartments = Vec::new();
    loop {
        let Some(first) = raw
            .get(offset)
            .and_then(|f| f.source_provenance.as_ref())
            .and_then(|p| p.source_id.as_ref())
        else {
            break;
        };
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

fn external_retraction_application(agent: &Agent) -> Option<HistoryApplication> {
    let archive = &agent.runtime_snapshot.history_archive;
    let previous = archive.application.as_ref()?;
    let known: std::collections::BTreeSet<_> = archive
        .applied_ids
        .iter()
        .filter_map(|id| archive.publications.get(id))
        .flat_map(|p| p.facts.iter().map(|f| &f.id))
        .collect();
    let revoked: std::collections::BTreeSet<String> = agent
        .turn
        .recalled_project_facts
        .iter()
        .flat_map(|f| f.tags.iter())
        .filter_map(|t| t.strip_prefix("withdraw_local_fact:"))
        .filter(|id| known.contains(&id.to_string()) && !archive.withdrawn_fact_ids.contains(*id))
        .map(str::to_string)
        .collect();
    if revoked.is_empty() {
        return None;
    }
    let mut application = previous.clone();
    application.publication_ids.clear();
    application
        .baseline_fact_ids
        .retain(|id| !revoked.contains(id));
    application
        .delta_fact_ids
        .retain(|id| !revoked.contains(id));
    application.withdrawn_fact_ids.extend(revoked);
    application.withdrawn_fact_ids.sort();
    application.withdrawn_fact_ids.dedup();
    application.first_kept_entry_id = agent
        .active_protocol_frames()
        .iter()
        .find(|f| !matches!(f.item, HistoryItem::ContextSummary { .. }))
        .and_then(|f| f.source_provenance.as_ref())
        .and_then(|p| p.source_id.clone());
    Some(application)
}

fn reconcile_recalled_facts(agent: &mut Agent) {
    let archive = &agent.runtime_snapshot.history_archive;
    let retired: std::collections::HashSet<_> = archive
        .applied_ids
        .iter()
        .filter_map(|id| archive.publications.get(id))
        .flat_map(|p| {
            p.withdrawn_fact_ids
                .iter()
                .chain(p.facts.iter().flat_map(|f| f.supersedes.iter()))
        })
        .collect();
    let before = agent.turn.recalled_project_facts.len();
    agent
        .turn
        .recalled_project_facts
        .retain(|fact| !retired.contains(&fact.id));
    if before != agent.turn.recalled_project_facts.len() {
        agent.turn.frozen_evidence = None;
    }
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
    if let Some(application) = external_retraction_application(agent) {
        on_event(AgentEvent::HistoryApplied {
            application,
            revision: agent.runtime_snapshot.context_scope_revision,
            blocking,
        })
        .await?;
        agent.reload_runtime_snapshot_from_provider()?;
        agent.turn.frozen_evidence = None;
        agent.clear_active_epoch();
        agent.clear_provider_usage_anchor();
        return Ok(true);
    }
    let runtime = agent.historian_runtime.clone();
    if let Some(runtime) = &runtime {
        let snapshot = &agent.runtime_snapshot;
        match runtime.poll(
            snapshot.session_id.as_deref().unwrap_or(""),
            &snapshot.active_context.branch_id,
            snapshot.context_scope_revision,
        ) {
            Ok(Some(_)) => agent.reload_runtime_snapshot_from_provider()?,
            Ok(None) => {}
            Err(error) if blocking => return Err(error),
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
        on_event(AgentEvent::HistoryApplied {
            application,
            revision: agent.runtime_snapshot.context_scope_revision,
            blocking,
        })
        .await?;
        agent.reload_runtime_snapshot_from_provider()?;
        reconcile_recalled_facts(agent);
        agent.clear_active_epoch();
        agent.clear_provider_usage_anchor();
        return Ok(true);
    }
    if !execute {
        return Ok(false);
    }
    if let Some(runtime) = &runtime {
        if !runtime.is_running() {
            let Some(work) = work(agent, manual)? else {
                return Ok(false);
            };
            if !blocking && agent.last_historian_work.as_ref() == Some(&work.id) {
                return Ok(false);
            }
            agent.last_historian_work = Some(work.id.clone());
            runtime.start(agent, work)?;
        }
        if !blocking {
            return Ok(false);
        }
        let session = agent
            .runtime_snapshot
            .session_id
            .clone()
            .unwrap_or_default();
        let branch = agent.runtime_snapshot.active_context.branch_id.clone();
        let revision = agent.runtime_snapshot.context_scope_revision;
        tokio::time::timeout(std::time::Duration::from_secs(610), async {
            loop {
                if runtime.poll(&session, &branch, revision)?.is_some() {
                    return Ok::<_, anyhow::Error>(());
                }
                if !runtime.is_running() {
                    anyhow::bail!("historian stopped before publishing");
                }
                tokio::time::sleep(std::time::Duration::from_millis(25)).await;
            }
        })
        .await
        .map_err(|_| anyhow!("historian completion timed out"))??;
        agent.reload_runtime_snapshot_from_provider()?;
    } else {
        // Child-session and direct runner paths use the same producer contract
        // synchronously, without constructing recursive Historian sessions.
        let Some(work) = work(agent, manual)? else {
            return Ok(false);
        };
        let helper = AgentFactory::create_child_with_route_and_max_tool_calls(
            agent,
            &AgentTemplate::historian(),
            None,
            false,
            Some(0),
        )?;
        let raw = helper.run_resolved_text_oneshot(&work.prompt).await?;
        let mut publication =
            crate::historian::parse_publication(&work.id, &work.source_ids, &raw)?;
        publication.project_path = Some(work.project_path.clone());
        publication.external_fact_ids = work.external_fact_ids.clone();
        on_event(AgentEvent::HistoryPublished {
            publication,
            revision: work.revision,
        })
        .await?;
        agent.reload_runtime_snapshot_from_provider()?;
    }
    if let Some(application) = pending_application(agent, true)? {
        on_event(AgentEvent::HistoryApplied {
            application,
            revision: agent.runtime_snapshot.context_scope_revision,
            blocking,
        })
        .await?;
        agent.reload_runtime_snapshot_from_provider()?;
        reconcile_recalled_facts(agent);
        agent.clear_active_epoch();
        agent.clear_provider_usage_anchor();
        return Ok(true);
    }
    Ok(false)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::transcript::{TranscriptEvent, TranscriptRecorder};
    use std::sync::{Arc, Mutex};
    use tokio::io::{AsyncReadExt, AsyncWriteExt};

    #[tokio::test]
    async fn background_publication_preserves_new_work_and_uses_native_instructions() {
        tokio::time::timeout(std::time::Duration::from_secs(15),async {
            let listener=tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
            let address=listener.local_addr().unwrap();
            let second_accepted=Arc::new(tokio::sync::Notify::new());
            let release_second=Arc::new(tokio::sync::Notify::new());
            let accepted=second_accepted.clone();
            let release=release_second.clone();
            let server=tokio::spawn(async move {
              for attempt in 0..2 {
                let (mut stream,_)=listener.accept().await.unwrap();
                let mut bytes=Vec::new();
                let mut buffer=[0u8;4096];
                let (header_end,length)=loop {
                    let n=stream.read(&mut buffer).await.unwrap(); assert!(n>0); bytes.extend_from_slice(&buffer[..n]);
                    if let Some(at)=bytes.windows(4).position(|p|p==b"\r\n\r\n") {
                        let end=at+4;
                        let headers=std::str::from_utf8(&bytes[..end]).unwrap();
                        let length=headers.lines().find_map(|line| {
                            let (key,value)=line.split_once(':')?;
                            key.eq_ignore_ascii_case("content-length").then(||value.trim().parse::<usize>().unwrap())
                        }).unwrap();
                        break (end,length);
                    }
                };
                while bytes.len()<header_end+length {
                    let n=stream.read(&mut buffer).await.unwrap(); assert!(n>0);bytes.extend_from_slice(&buffer[..n]);
                }
                let request:serde_json::Value=serde_json::from_slice(&bytes[header_end..header_end+length]).unwrap();
                let instructions=request["instructions"].as_str().unwrap();
                assert_eq!(instructions.matches("You are Historian").count(),1);
                assert!(!request["input"].to_string().contains("You are Historian"));
                assert!(request.get("tools").is_none() || request["tools"].as_array().unwrap().is_empty());
                if attempt==0 { assert!(request["input"].to_string().contains("EXACT-TAIL-CONSTRAINT")); }
                if attempt==1 { accepted.notify_one(); release.notified().await; }
                tokio::time::sleep(std::time::Duration::from_millis(100)).await;
                let output=serde_json::json!({"compartments":[{"start":0,"end":if attempt==0 {4} else {1},"title":"Parser fix","importance":70,"detailed":"Fixed UTF-8 parser offsets","compact":"Parser offset fix","anchor":"UTF-8 parser"}],"facts":[],"unprocessed_from":null}).to_string();
                let delta=serde_json::json!({"type":"response.output_text.delta","delta":output});
                let terminal=serde_json::json!({"type":"response.completed","response":{"status":"completed","usage":{"input_tokens":200,"output_tokens":60,"total_tokens":260}}});
                let body=format!("data: {delta}\n\ndata: {terminal}\n\n");
                let result=stream.write_all(format!("HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",body.len()).as_bytes()).await;
                if attempt==0 { result.unwrap(); }
              }
            });
            let config=crate::model_runtime::RuntimeConfig::from_toml(&format!(r#"active_provider="test"
[providers.test]
protocol="responses"
default_model="test-model"
[providers.test.auth]
type="none"
[providers.test.endpoints]
base_url="http://{address}"
[providers.test.retry]
enabled=false
max_attempts=1
max_recovery_attempts=1
initial_delay_secs=1
exponential_backoff=false
backoff_multiplier=1.0
jitter_secs=0
[providers.test.models."test-model"]
[providers.test.models."test-model".capabilities]
generation={{max_output_tokens=true}}
[providers.test.models."test-model".generation]
max_output_tokens=4096
"#)).unwrap().resolve(&crate::model_runtime::ProtocolRegistry::builtins()).unwrap();
            let route=config.route("test","test-model").unwrap().clone();
            let mut agent=Agent::new("test-model",5,0);
            agent.set_primary_route(crate::config::ModelRoute::new("test","test-model"));
            agent.set_resolved_model_route(Some(Arc::new(route)));
            let directory=tempfile::tempdir().unwrap();
            let mut recorder=TranscriptRecorder::create(directory.path()).unwrap();
            recorder.record_session_started("test/test-model").unwrap();
            recorder.record_user_message("Fix parser").unwrap();
            recorder.record_assistant_tool_call_batch(None,None,None,vec![HistoryToolCall { call_id:"read-1".into(),name:"fs__read".into(),arguments_json:r#"{"path":"parser.rs"}"#.into() }]).unwrap();
            recorder.record_tool_call_finished("read-1","fs__read",true,crate::tool::ToolResult::ok("fs__read",serde_json::json!({"content":format!("{} EXACT-TAIL-CONSTRAINT","detail ".repeat(500))}))).unwrap();
            recorder.record_assistant_message("Fixed UTF-8 parser offsets").unwrap();
            let recorder=Arc::new(Mutex::new(recorder));
            let projected=recorder.clone();
            agent.set_runtime_snapshot_provider(Arc::new(move || {
                let recorder=projected.lock().unwrap();
                let records=crate::transcript::read_records(recorder.path())?;
                Ok(crate::transcript::transcript_projection::project_runtime_restore_snapshot(
                    recorder.session_id().to_string(),records,
                    crate::transcript::transcript_projection::SessionContextCursor {branch_id:None,leaf_sequence:None},&[],
                )?.snapshot)
            }));
            agent.reload_runtime_snapshot_from_provider().unwrap();
            let (tx,_rx)=tokio::sync::mpsc::unbounded_channel();
            let pool=crate::subagent::SubagentPool::new();
            let runtime=Arc::new(crate::session::historian::HistorianRuntime::new(pool.clone(),directory.path().into(),recorder.clone(),tx));
            agent.historian_runtime=Some(runtime.clone());
            runtime.start(&agent,work(&agent,true).unwrap().unwrap()).unwrap();
            assert!(runtime.is_running());
            recorder.lock().unwrap().record_user_message("New work during history preparation").unwrap();
            let observed=recorder.clone();
            let mut events=move |event| {
                let result=crate::agent_event_journal::persist_agent_event(&mut observed.lock().unwrap(),&event).map(|_|());
                std::future::ready(result)
            };
            assert!(advance(&mut agent,true,true,&mut events).await.unwrap());
            assert!(agent.active_history_items().iter().any(|item|matches!(item,HistoryItem::UserMessage { content } if content.text.contains("New work"))));
            let records=crate::transcript::read_records(recorder.lock().unwrap().path()).unwrap();
            assert_eq!(records.iter().filter(|r|matches!(r.event,TranscriptEvent::HistoryPublished(_))).count(),1);
            assert_eq!(records.iter().filter(|r|matches!(r.event,TranscriptEvent::HistoryApplied(_))).count(),1);
            let child_id = records.iter().find_map(|record| match &record.event {
                TranscriptEvent::SubagentStarted { child_session_id, agent_name, .. } if agent_name == "historian" => Some(child_session_id),
                _ => None,
            }).unwrap();
            let child_records = crate::transcript::read_child_session_records(directory.path(), child_id).unwrap();
            let report = crate::transcript::restore_session_history(&child_records).unwrap().into_iter().find_map(|item| {
                if let HistoryItem::AssistantTurn { text: Some(content), .. } = item { serde_json::from_str::<crate::historian::HistorianReport>(&content).ok() } else { None }
            }).expect("persisted structured Historian report");
            assert_eq!(report.publication.compartments.len(), 1);
            assert_eq!(report.source_session_id, recorder.lock().unwrap().session_id());
            assert!(matches!(report.usage.last(), Some(crate::historian::UsageUpdate::Usage { input_tokens:200, output_tokens:60, .. })));
            assert!(!records.iter().any(|r|matches!(r.event,TranscriptEvent::ContextCompaction(_))));
            let restored=crate::transcript::restore_session_history(&records).unwrap();
            assert!(restored.iter().any(|item|matches!(item,HistoryItem::UserMessage {content} if content.text.contains("New work"))));
            assert!(!crate::evidence::restore_evidence_records(&records).unwrap().iter().any(|e| e.tags.iter().any(|t|t=="historian")), "internal work reports must not become project facts");
            agent.history_budget_limit=Some(128);
            let small=pending_application(&agent,true).unwrap().unwrap();
            assert!(small.publication_ids.is_empty() && small.baseline.is_empty());
            events(AgentEvent::HistoryApplied {application:small,revision:0,blocking:false}).await.unwrap();
            agent.reload_runtime_snapshot_from_provider().unwrap();
            agent.history_budget_limit=None;
            let restored_selection=pending_application(&agent,true).unwrap().unwrap();
            assert_eq!(restored_selection.baseline.len(),1,"a larger budget may restore archived tiers without an LLM call");
            events(AgentEvent::HistoryApplied {application:restored_selection,revision:0,blocking:false}).await.unwrap();
            agent.reload_runtime_snapshot_from_provider().unwrap();
            runtime.start(&agent,work(&agent,true).unwrap().unwrap()).unwrap();
            second_accepted.notified().await;
            runtime.cancel();
            release_second.notify_one();
            server.await.unwrap();
            while pool.is_running() { tokio::time::sleep(std::time::Duration::from_millis(10)).await; }
            let cancelled_records=crate::transcript::read_records(recorder.lock().unwrap().path()).unwrap();
            assert_eq!(cancelled_records.iter().filter(|r|matches!(r.event,TranscriptEvent::HistoryPublished(_))).count(),1,"cancelled work must not publish late");
        }).await.expect("historian test timed out");
    }
}

#[cfg(test)]
mod wire_tests {
    use super::*;
    #[test]
    fn three_protocols_replay_frozen_slots_and_keep_baseline_before_delta() {
        for protocol in ["responses", "completions", "anthropic"] {
            let config = crate::model_runtime::RuntimeConfig::from_toml(&format!(
                r#"active_provider="p"
[providers.p]
protocol="{protocol}"
default_model="m"
[providers.p.auth]
type="none"
[providers.p.endpoints]
base_url="http://127.0.0.1:1"
[providers.p.models.m]
[providers.p.models.m.capabilities]
generation={{max_output_tokens=true}}
[providers.p.models.m.generation]
max_output_tokens=128
"#
            ))
            .unwrap()
            .resolve(&crate::model_runtime::ProtocolRegistry::builtins())
            .unwrap();
            let route = config.route("p", "m").unwrap();
            let model = ModelRequestMetadata {
                context_window: Some(8192),
                max_output_tokens: Some(128),
                supports_reasoning: false,
                supports_tools: false,
                ..Default::default()
            };
            let recalled = crate::evidence::EvidenceRecord {
                id: "other:fact:f".into(),
                sequence: 1,
                timestamp_ms: 0,
                evidence_kind: crate::evidence::EvidenceKind::Decision,
                title: "Project memory".into(),
                summary: "RECALLED-FACT".into(),
                detail: None,
                source: crate::evidence::EvidenceSource::Session {
                    session_id: "other".into(),
                    branch_id: "root".into(),
                    entry_id: "evidence:fact:f".into(),
                },
                tags: vec!["recalled_project_fact".into()],
            };
            let prepare = |delta: &str| {
                let history = vec![
                    HistoryItem::context_summary("[Session history]\nFROZEN-BASELINE"),
                    HistoryItem::context_summary(format!("[Session history]\n{delta}")),
                    HistoryItem::user("new request"),
                ];
                let prelude = vec![
                    PromptMessage::system("SYSTEM-AUTHORITY"),
                    PromptMessage::developer("DYNAMIC-CONTEXT"),
                ];
                let build = crate::request_builder::build_test_request(
                    crate::request_builder::TestRequestBuilderInput {
                        model_id: "m",
                        model: model.clone(),
                        prelude: &prelude,
                        history: &history,
                        protected_start_index: 2,
                        tools: &[],
                        evidence: std::slice::from_ref(&recalled),
                    },
                )
                .unwrap();
                let input = crate::model_runtime::projection::model_request_from_prompt_plan(
                    route,
                    &model,
                    &build.prompt_plan,
                    &[],
                )
                .unwrap();
                let request = route.binding.prepare_request(&input).unwrap();
                String::from_utf8(request.body).unwrap()
            };
            let first = prepare("DELTA-ONE");
            assert_eq!(
                first,
                prepare("DELTA-ONE"),
                "{protocol}: defer must replay identically"
            );
            let second = prepare("DELTA-TWO");
            assert!(first.contains("FROZEN-BASELINE") && first.contains("DELTA-ONE"));
            assert_eq!(
                first.split("DELTA-ONE").next(),
                second.split("DELTA-TWO").next(),
                "{protocol}: soft update changed the preceding wire prefix"
            );
            let value: serde_json::Value = serde_json::from_str(&first).unwrap();
            match protocol {
                "responses" => {
                    assert!(
                        value["instructions"]
                            .as_str()
                            .unwrap()
                            .contains("SYSTEM-AUTHORITY")
                    );
                    assert!(
                        !value["instructions"]
                            .as_str()
                            .unwrap()
                            .contains("FROZEN-BASELINE")
                    );
                }
                "anthropic" => {
                    assert!(value["system"].to_string().contains("SYSTEM-AUTHORITY"));
                    assert!(!value["system"].to_string().contains("FROZEN-BASELINE"));
                    assert!(!value["system"].to_string().contains("RECALLED-FACT"));
                }
                _ => assert_eq!(first.matches("SYSTEM-AUTHORITY").count(), 1),
            }
        }
    }
}

#[cfg(test)]
mod retraction_tests {
    use super::*;
    #[tokio::test]
    async fn resuming_fact_owner_applies_external_retraction_without_rewriting_history() {
        let root = tempfile::tempdir().unwrap();
        crate::memory::project_fact_tests::write_session(root.path(), "a", "project", vec![]);
        crate::memory::project_fact_tests::write_session(
            root.path(),
            "b",
            "project",
            vec!["a:fact:a-f".into()],
        );
        let records = crate::transcript::read_records(root.path().join("a.jsonl")).unwrap();
        let snapshot = crate::transcript::transcript_projection::project_runtime_restore_snapshot(
            "a".into(),
            records.clone(),
            crate::transcript::transcript_projection::SessionContextCursor {
                branch_id: None,
                leaf_sequence: None,
            },
            &[],
        )
        .unwrap()
        .snapshot;
        let mut agent = Agent::new("m", 1, 0);
        agent.runtime_snapshot = snapshot;
        agent.turn.recalled_project_facts =
            crate::memory::recall_project_facts(root.path(), "project", "a").unwrap();
        let application = external_retraction_application(&agent).unwrap();
        assert!(!application.baseline_fact_ids.contains(&"a-f".to_string()));
        assert!(application.withdrawn_fact_ids.contains(&"a-f".to_string()));
        let mut recorder =
            crate::transcript::TranscriptRecorder::open_existing(root.path(), "a").unwrap();
        let path = recorder.path().to_path_buf();
        agent.set_runtime_snapshot_provider(Arc::new(move || {
            Ok(
                crate::transcript::transcript_projection::project_runtime_restore_snapshot(
                    "a".into(),
                    crate::transcript::read_records(&path)?,
                    crate::transcript::transcript_projection::SessionContextCursor {
                        branch_id: None,
                        leaf_sequence: None,
                    },
                    &[],
                )?
                .snapshot,
            )
        }));
        let mut events = |event| {
            std::future::ready(
                crate::agent_event_journal::persist_agent_event(&mut recorder, &event).map(|_| ()),
            )
        };
        assert!(
            advance(&mut agent, false, false, &mut events)
                .await
                .unwrap(),
            "reconcile even below the background threshold"
        );
        assert!(
            external_retraction_application(&agent).is_none(),
            "the same retraction is not applied twice"
        );
        let hard = pending_application(&agent, true).unwrap().unwrap();
        assert!(hard.baseline_fact_ids.is_empty() && hard.delta_fact_ids.is_empty());
        events(AgentEvent::HistoryApplied {
            application: hard,
            revision: agent.runtime_snapshot.context_scope_revision,
            blocking: false,
        })
        .await
        .unwrap();
        agent.reload_runtime_snapshot_from_provider().unwrap();
        let records = crate::transcript::read_records(root.path().join("a.jsonl")).unwrap();
        let restored = &agent.runtime_snapshot;
        let third_session =
            crate::memory::recall_project_facts(root.path(), "project", "third").unwrap();
        assert_eq!(third_session.len(), 1);
        assert_eq!(third_session[0].id, "b:fact:b-f");
        assert!(restored.evidence.is_empty());
        assert!(restored.history_archive.effective_fact_ids(&[]).is_empty());
        assert!(
            matches!(&records[1].event,crate::transcript::TranscriptEvent::HistoryPublished(p) if p.facts[0].text=="a design")
        );
        assert!(crate::transcript::restore_session_history(&records).unwrap().iter().all(|item|!matches!(item,HistoryItem::ContextSummary {text} if text.contains("a design"))));
    }
}
