//! Incremental Historian orchestration at request boundaries.
use super::*;
use crate::context_history::{
    HistoryApplication, HistoryCompartment, HistorySelection, HistoryTier, select_tiers,
};
use crate::session::historian::HistoryWork;

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

pub(super) fn work(agent: &Agent, manual: bool) -> Result<Option<HistoryWork>> {
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
        let input = crate::historian::history_input(&history[start..end], &references, &facts);
        match protocol_stream::prepare_resolved_oneshot_request(
            route,
            helper.active_model_metadata(),
            &helper.prelude,
            &input,
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
        apply_history(agent, application, blocking, on_event).await?;
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
                if let Some(publication) = runtime.poll(&session, &branch, revision)? {
                    agent.last_historian_work = Some(publication.id);
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
        let (raw, _) = helper.run_historian(&work.input).await?;
        let mut publication =
            crate::historian::parse_publication(&work.id, &work.source_ids, &raw)?;
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
              for attempt in 0..3 {
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
                let input = request["input"].as_array().unwrap().iter()
                    .filter_map(|message| message["content"].as_array()).flatten()
                    .filter_map(|part| part["text"].as_str())
                    .find_map(|text| serde_json::from_str::<serde_json::Value>(text).ok().filter(|value| value.get("new_messages").is_some())).unwrap();
                assert_eq!(input["source_count"].as_u64().unwrap() as usize, input["new_messages"].as_array().unwrap().len());
                if attempt==0 {
                    assert!(input.to_string().contains("EXACT-TAIL-CONSTRAINT"));
                    assert!(!input.to_string().contains("data:image/png;base64,"));
                    assert!(!input.to_string().contains("OPAQUE-REPLAY"));
                    assert!(input.to_string().contains("READABLE-REASONING"));
                    let images: Vec<_> = request["input"].as_array().unwrap().iter()
                        .flat_map(|message| message["content"].as_array().into_iter().flatten())
                        .filter_map(|part| part["image_url"].as_str()).collect();
                    assert_eq!(images, vec![
                        "data:image/png;base64,dXNlci1pbWFnZQ==",
                        "data:image/png;base64,dG9vbC1pbWFnZQ==",
                    ]);
                }
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
input_images=true
generation={{max_output_tokens=true}}
[providers.test.models."test-model".generation]
max_output_tokens=4096
"#)).unwrap().resolve(&crate::model_runtime::ProtocolRegistry::builtins()).unwrap();
            let route=config.route("test","test-model").unwrap().clone();
            let mut agent=Agent::new("test-model",5,0);
            agent.set_model_catalog(std::collections::HashMap::from([("test-model".into(), ModelRequestMetadata {
                context_window: Some(32_768), max_output_tokens: Some(4096), ..Default::default()
            })]));
            agent.set_primary_route(crate::config::ModelRoute::new("test","test-model"));
            agent.set_resolved_model_route(Some(Arc::new(route)));
            let directory=tempfile::tempdir().unwrap();
            let mut recorder=TranscriptRecorder::create(directory.path()).unwrap();
            recorder.record_session_started("test/test-model").unwrap();
            recorder.record_user_message_content(UserMessageContent::new("Fix parser", vec![
                crate::user_content::UserImageAttachment::from_bytes("user image", "image/png", b"user-image")
            ])).unwrap();
            let replay = r#"[{"type":"redacted_thinking","data":"OPAQUE-REPLAY"}]"#.to_string();
            recorder.record_assistant_tool_call_batch(None,Some("READABLE-REASONING".into()),Some(replay),vec![HistoryToolCall { call_id:"read-1".into(),name:"fs__read".into(),arguments_json:r#"{"path":"parser.rs"}"#.into() }]).unwrap();
            recorder.record_tool_call_finished("read-1","fs__read",true,crate::tool::ToolResult::ok("fs__read",serde_json::json!({"content":format!("{} EXACT-TAIL-CONSTRAINT","detail ".repeat(500))})).with_images(vec![
                crate::user_content::UserImageAttachment::from_bytes("tool image", "image/png", b"tool-image")
            ])).unwrap();
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
            let first_work = work(&agent, true).unwrap().unwrap();
            let first_work_id = first_work.id.clone();
            runtime.start(&agent, first_work).unwrap();
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
            assert_eq!(agent.last_historian_work.as_deref(), Some(first_work_id.as_str()));
            let child_id = records.iter().find_map(|record| match &record.event {
                TranscriptEvent::SubagentStarted { child_session_id, agent_name, .. } if agent_name == "historian" => Some(child_session_id),
                _ => None,
            }).unwrap();
            let child_records = crate::transcript::read_child_session_records(directory.path(), child_id).unwrap();
            for record in records.iter().chain(&child_records) {
                match &record.event {
                    TranscriptEvent::SubagentStarted { agent_name, summary, .. } if agent_name == "historian" => {
                        assert_eq!(summary, "Organize 4 history items");
                    }
                    TranscriptEvent::SubagentLifecycle { agent_name, detail: Some(detail), .. } if agent_name == "historian" => {
                        assert!(!detail.contains("EXACT-TAIL-CONSTRAINT"));
                        assert!(!detail.contains("new_messages"));
                    }
                    _ => {}
                }
            }
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
            let second_work = work(&agent, true).unwrap().unwrap();
            let second_work_id = second_work.id.clone();
            let observed = recorder.clone();
            let mut events = move |event| {
                let result = crate::agent_event_journal::persist_agent_event(
                    &mut observed.lock().unwrap(),
                    &event,
                )
                .map(|_| ());
                std::future::ready(result)
            };
            let mut advancing = Box::pin(advance(&mut agent, true, true, &mut events));
            let result = tokio::select! {
                result = &mut advancing => result,
                _ = second_accepted.notified() => {
                    runtime.cancel();
                    release_second.notify_one();
                    (&mut advancing).await
                }
            };
            drop(advancing);
            assert!(result.is_err(), "cancellation must not publish the work");
            while pool.is_running() {
                tokio::time::sleep(std::time::Duration::from_millis(10)).await;
            }
            assert_ne!(
                agent.last_historian_work.as_deref(),
                Some(second_work_id.as_str()),
                "cancelled work must not be recorded as successful"
            );

            let retry_work = work(&agent, true).unwrap().unwrap();
            assert_eq!(retry_work.id, second_work_id);
            let observed = recorder.clone();
            let mut events = move |event| {
                let result = crate::agent_event_journal::persist_agent_event(
                    &mut observed.lock().unwrap(),
                    &event,
                )
                .map(|_| ());
                std::future::ready(result)
            };
            assert!(advance(&mut agent, true, true, &mut events).await.unwrap());
            assert_eq!(
                agent.last_historian_work.as_deref(),
                Some(second_work_id.as_str()),
                "only the successful retry may record the work"
            );
            server.await.unwrap();
            let cancelled_records=crate::transcript::read_records(recorder.lock().unwrap().path()).unwrap();
            assert_eq!(cancelled_records.iter().filter(|r|matches!(r.event,TranscriptEvent::HistoryPublished(_))).count(),2,"cancelled work must not publish late");
        }).await.expect("historian test timed out");
    }
}

#[cfg(test)]
mod wire_tests {
    use super::*;

    #[test]
    fn historian_preflight_uses_native_images_and_readable_text_in_all_protocols() {
        use crate::user_content::UserImageAttachment;
        use serde_json::{Value, json};
        let image = UserImageAttachment::from_bytes("user image", "image/png", &vec![37; 120_000]);
        let tool_image =
            UserImageAttachment::from_bytes("tool image", "image/png", b"tool-image-data");
        let replay = crate::model_runtime::OpaqueReplayState::from_anthropic_thinking_blocks_json(
            &json!([{"type":"redacted_thinking","data":"OPAQUE-ONLY".repeat(20_000)}]).to_string(),
        )
        .unwrap();
        let history = vec![
            HistoryItem::user_content(UserMessageContent::new("USER-TEXT", vec![image.clone()])),
            HistoryItem::AssistantTurn {
                text: Some("ASSISTANT-TEXT".into()),
                reasoning_content: Some("READABLE-REASONING".into()),
                replay: Some(replay),
                calls: vec![],
            },
            HistoryItem::ToolOutput {
                call_id: "image-call".into(),
                output_json: "TOOL-TEXT".into(),
                images: vec![tool_image.clone()],
            },
        ];
        let content = crate::historian::history_input(&history, &[], &[]);
        let prelude = [
            PromptMessage::system(crate::historian::HISTORIAN_PROMPT),
            PromptMessage::developer("DYNAMIC-CONTEXT"),
        ];
        for protocol in ["responses", "completions", "anthropic"] {
            for input_images in [true, false] {
                let catalog = crate::model_runtime::RuntimeConfig::from_toml(&format!(
                    r#"active_provider="p"
[providers.p]
protocol="{protocol}"
default_model="m"
[providers.p.auth]
type="none"
[providers.p.endpoints]
base_url="http://127.0.0.1:1"
[providers.p.models.m.capabilities]
input_images={input_images}
generation={{max_output_tokens=true}}
[providers.p.models.m.generation]
max_output_tokens=128
"#
                ))
                .unwrap()
                .resolve(&crate::model_runtime::ProtocolRegistry::builtins())
                .unwrap();
                let route = catalog.route("p", "m").unwrap();
                let metadata = ModelRequestMetadata {
                    context_window: Some(32_768),
                    max_output_tokens: Some(128),
                    supports_reasoning: false,
                    supports_tools: false,
                    parallel_tool_calls: false,
                    ..Default::default()
                };
                let result = protocol_stream::prepare_resolved_oneshot_request(
                    route,
                    metadata.clone(),
                    &prelude,
                    &content,
                );
                if !input_images {
                    assert!(
                        result
                            .unwrap_err()
                            .to_string()
                            .contains("unsupported_request_field")
                    );
                    continue;
                }
                let (build, input) = result.unwrap();
                assert!(!build.budget.truncated);
                assert!(build.budget.estimated_request_tokens < 20_000);
                assert!(
                    build.budget.estimated_request_tokens
                        >= image.visual_token_charge() + tool_image.visual_token_charge()
                );
                let mut larger_history = history.clone();
                if let HistoryItem::UserMessage { content } = &mut larger_history[0] {
                    *content = UserMessageContent::new(
                        "USER-TEXT",
                        vec![UserImageAttachment::from_bytes(
                            "user image",
                            "image/png",
                            &vec![38; 240_000],
                        )],
                    );
                }
                let larger = crate::historian::history_input(&larger_history, &[], &[]);
                let (larger_build, _) = protocol_stream::prepare_resolved_oneshot_request(
                    route,
                    metadata.clone(),
                    &prelude,
                    &larger,
                )
                .unwrap();
                assert_eq!(
                    build.budget.estimated_request_tokens,
                    larger_build.budget.estimated_request_tokens,
                    "encoded image bytes must not become text tokens"
                );
                let journal_text = json!({"new_messages":history}).to_string();
                assert!(
                    protocol_stream::preflight_resolved_oneshot_text_request(
                        route,
                        metadata.clone(),
                        &prelude,
                        &journal_text
                    )
                    .is_err()
                );
                let prepared = route.binding.prepare_request(&input).unwrap();
                let body: Value = serde_json::from_slice(&prepared.body).unwrap();
                let messages = if protocol == "responses" {
                    &body["input"]
                } else {
                    &body["messages"]
                };
                let blocks: Vec<_> = messages
                    .as_array()
                    .unwrap()
                    .iter()
                    .filter(|message| message["role"] == "user")
                    .flat_map(|message| message["content"].as_array().unwrap())
                    .collect();
                let texts: Vec<_> = blocks
                    .iter()
                    .filter_map(|part| part["text"].as_str())
                    .collect();
                let text = texts.join("\n");
                for marker in [
                    "USER-TEXT",
                    "ASSISTANT-TEXT",
                    "READABLE-REASONING",
                    "TOOL-TEXT",
                ] {
                    assert!(text.contains(marker));
                }
                assert!(!text.contains("OPAQUE-ONLY"));
                assert!(!text.contains("data:image/png;base64,"));
                let payload: Value = serde_json::from_str(texts[0]).unwrap();
                assert_eq!(
                    payload["new_messages"][0]["content"]["parts"][1]["attachment"]["attachment_index"],
                    0
                );
                assert_eq!(
                    payload["new_messages"][2]["content"]["images"][0]["attachment_index"],
                    1
                );
                let images: Vec<_> = blocks
                    .iter()
                    .filter_map(|part| match protocol {
                        "responses" => part["image_url"].as_str().map(str::to_string),
                        "completions" => part["image_url"]["url"].as_str().map(str::to_string),
                        _ => part["source"]["data"]
                            .as_str()
                            .map(|data| format!("data:image/png;base64,{data}")),
                    })
                    .collect();
                assert_eq!(
                    images,
                    vec![image.data_url.clone(), tool_image.data_url.clone()]
                );
                assert_eq!(body.to_string().matches("You are Historian").count(), 1);
                assert!(!text.contains("You are Historian"));
                let system = match protocol {
                    "responses" => body["instructions"].to_string(),
                    "completions" => body["messages"][0]["content"].to_string(),
                    _ => body["system"].to_string(),
                };
                assert!(system.contains("You are Historian"));
                assert!(!system.contains("USER-TEXT"));
                assert!(!system.contains("READABLE-REASONING"));
                if protocol != "anthropic" {
                    assert!(!system.contains("DYNAMIC-CONTEXT"));
                }
                assert!(
                    body.get("tools").is_none() || body["tools"].as_array().unwrap().is_empty()
                );
            }
        }
    }

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
                        evidence: &[],
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
                }
                _ => assert_eq!(first.matches("SYSTEM-AUTHORITY").count(), 1),
            }
        }
    }
}

#[cfg(test)]
mod live_source_tests {
    use super::*;
    use crate::transcript::{TranscriptEvent, TranscriptRecorder};
    use std::sync::{Arc, Mutex};

    fn agent_with_recorder(root: &std::path::Path) -> (Agent, Arc<Mutex<TranscriptRecorder>>) {
        let config = crate::model_runtime::RuntimeConfig::from_toml(
            r#"
active_provider="test"
[providers.test]
protocol="responses"
default_model="m"
[providers.test.auth]
type="none"
[providers.test.endpoints]
base_url="http://127.0.0.1:1"
[providers.test.models.m]
[providers.test.models.m.capabilities]
generation={max_output_tokens=true}
[providers.test.models.m.generation]
max_output_tokens=128
"#,
        )
        .unwrap()
        .resolve(&crate::model_runtime::ProtocolRegistry::builtins())
        .unwrap();
        let mut agent = Agent::new("m", 5, 0);
        agent.set_primary_route(crate::config::ModelRoute::new("test", "m"));
        agent.set_resolved_model_route(Some(Arc::new(config.route("test", "m").unwrap().clone())));
        let mut recorder = TranscriptRecorder::create(root).unwrap();
        recorder.record_session_started("test/m").unwrap();
        let recorder = Arc::new(Mutex::new(recorder));
        let projected = recorder.clone();
        agent.set_runtime_snapshot_provider(Arc::new(move || {
            let recorder = projected.lock().unwrap();
            Ok(
                crate::transcript::transcript_projection::project_runtime_restore_snapshot(
                    recorder.session_id().into(),
                    crate::transcript::read_records(recorder.path())?,
                    crate::transcript::transcript_projection::SessionContextCursor {
                        branch_id: None,
                        leaf_sequence: None,
                    },
                    &[],
                )?
                .snapshot,
            )
        }));
        agent.reload_runtime_snapshot_from_provider().unwrap();
        (agent, recorder)
    }

    // Mirrors the live append/persist order, including non-protocol journal rows
    // between results. A synthetic live span must never become a raw journal ID.
    fn append_live_group(
        agent: &mut Agent,
        recorder: &Arc<Mutex<TranscriptRecorder>>,
        prefix: &str,
        count: usize,
    ) {
        let calls: Vec<_> = (0..count)
            .map(|i| HistoryToolCall {
                call_id: format!("{prefix}-{i}"),
                name: "fs__read".into(),
                arguments_json: r#"{"path":"source.rs"}"#.into(),
            })
            .collect();
        agent
            .append_history_item(HistoryItem::AssistantTurn {
                text: None,
                reasoning_content: None,
                replay: None,
                calls: calls.clone(),
            })
            .unwrap();
        recorder
            .lock()
            .unwrap()
            .record_assistant_tool_call_batch(None, None, None, calls.clone())
            .unwrap();
        for call in &calls {
            recorder
                .lock()
                .unwrap()
                .record_tool_call_started(
                    &call.call_id,
                    &call.name,
                    serde_json::json!({"path":"source.rs"}),
                )
                .unwrap();
        }
        for call in &calls {
            let output = crate::tool::ToolResult::ok(
                "fs__read",
                serde_json::json!({"content":format!("unique body {}",call.call_id)}),
            );
            recorder
                .lock()
                .unwrap()
                .record_tool_call_finished(&call.call_id, &call.name, true, output.clone())
                .unwrap();
            agent
                .append_history_item(HistoryItem::ToolOutput {
                    call_id: call.call_id.clone(),
                    output_json: serde_json::to_string(&output).unwrap(),
                    images: vec![],
                })
                .unwrap();
        }
    }

    #[tokio::test]
    async fn persisted_sources_prepare_and_apply_while_live_appends_keep_request_identity() {
        let root = tempfile::tempdir().unwrap();
        let (mut agent, recorder) = agent_with_recorder(root.path());
        recorder
            .lock()
            .unwrap()
            .record_user_message("Inspect source")
            .unwrap();
        recorder
            .lock()
            .unwrap()
            .record_turn_started(TurnStartedEvent {
                turn_id: 1,
                intent: "inspect".into(),
                directive: String::new(),
                validation_reminder: String::new(),
            })
            .unwrap();
        agent.turn.current_turn_start_index = Some(0);
        agent
            .append_history_item(HistoryItem::user("Inspect source"))
            .unwrap();
        append_live_group(&mut agent, &recorder, "first", 8);
        assert!(
            agent.active_protocol_frames().iter().all(|f| f
                .source_provenance
                .as_ref()
                .unwrap()
                .source_id
                .is_none())
        );
        let before = agent.runtime_snapshot.clone();
        let frontier = agent.protocol_append_state.frontier_token();
        let generation = agent.protocol_append_state.generation();
        let prepared = work(&agent, true).unwrap().unwrap();
        assert_eq!(
            agent.runtime_snapshot, before,
            "background preparation must not install a different runtime"
        );
        assert_eq!(agent.protocol_append_state.frontier_token(), frontier);
        assert_eq!(agent.protocol_append_state.generation(), generation);
        let canonical = persisted_history_snapshot(&agent).unwrap();
        let expected: Vec<_> = canonical
            .active_protocol_frames()
            .iter()
            .map(|f| {
                f.source_provenance
                    .as_ref()
                    .unwrap()
                    .source_id
                    .clone()
                    .unwrap()
            })
            .collect();
        assert_eq!(prepared.source_ids, expected);
        assert_eq!(prepared.source_ids.len(), 10);
        assert!(prepared.input.text.contains("unique body first-7"));
        let mut publication = crate::historian::parse_publication(&prepared.id,&prepared.source_ids,&serde_json::json!({
            "compartments":[{"start":0,"end":10,"title":"Inspected sources","importance":70,"detailed":"Detailed result","compact":"Result","anchor":"Sources"}],
            "facts":[],"unprocessed_from":null
        }).to_string()).unwrap();
        publication.project_path = Some(prepared.project_path);
        recorder
            .lock()
            .unwrap()
            .record_history_event(
                TranscriptEvent::HistoryPublished(publication),
                prepared.revision,
            )
            .unwrap();
        append_live_group(&mut agent, &recorder, "second", 2);
        let canonical = persisted_history_snapshot(&agent).unwrap();
        let tail = canonical.active_protocol_frames()[10..].to_vec();
        let application = pending_application(&agent, true).unwrap().unwrap();
        assert_eq!(
            application.first_kept_entry_id,
            tail[0].source_provenance.as_ref().unwrap().source_id
        );
        recorder
            .lock()
            .unwrap()
            .record_history_event(
                TranscriptEvent::HistoryApplied(application),
                prepared.revision,
            )
            .unwrap();
        agent.reload_runtime_snapshot_from_provider().unwrap();
        assert_eq!(
            agent.active_history_items().last(),
            Some(&tail.last().unwrap().item)
        );
        let second = work(&agent, true).unwrap().unwrap();
        let publication = crate::historian::parse_publication(&second.id, &second.source_ids,
            &serde_json::json!({"compartments":[{"start":0,"end":second.source_ids.len(),"title":"Second group","importance":70,"detailed":"Second result","compact":"Result","anchor":"Second"}],"facts":[],"unprocessed_from":null}).to_string()).unwrap();
        recorder
            .lock()
            .unwrap()
            .record_history_event(
                TranscriptEvent::HistoryPublished(publication),
                second.revision,
            )
            .unwrap();
        let second_application = pending_application(&agent, true).unwrap().unwrap();
        assert!(second_application.first_kept_entry_id.is_none());
        recorder
            .lock()
            .unwrap()
            .record_history_event(
                TranscriptEvent::HistoryApplied(second_application),
                second.revision,
            )
            .unwrap();
        agent.reload_runtime_snapshot_from_provider().unwrap();
        append_live_group(&mut agent, &recorder, "third", 1);
        assert!(
            agent
                .active_protocol_frames()
                .iter()
                .filter(|f| !matches!(f.item, HistoryItem::ContextSummary { .. }))
                .all(|f| f.source_provenance.as_ref().unwrap().source_id.is_none())
        );
        let expected_tail: Vec<_> = persisted_history_snapshot(&agent)
            .unwrap()
            .active_protocol_frames()
            .into_iter()
            .filter(|f| !matches!(f.item, HistoryItem::ContextSummary { .. }))
            .map(|f| f.item)
            .collect();
        // Reapplying the frozen selection must retain the persisted raw boundary.
        let application = pending_application(&agent, true).unwrap();
        let observed = recorder.clone();
        let mut events = move |event| {
            std::future::ready(
                crate::agent_event_journal::persist_agent_event(
                    &mut observed.lock().unwrap(),
                    &event,
                )
                .map(|_| ()),
            )
        };
        if let Some(application) = application {
            apply_history(&mut agent, application, false, &mut events)
                .await
                .unwrap();
        }
        let actual_tail: Vec<_> = agent
            .active_history_items()
            .into_iter()
            .filter(|item| !matches!(item, HistoryItem::ContextSummary { .. }))
            .collect();
        assert_eq!(actual_tail, expected_tail);
    }

    #[test]
    fn historian_sources_reject_a_different_persisted_scope() {
        let root = tempfile::tempdir().unwrap();
        let (mut agent, _recorder) = agent_with_recorder(root.path());
        let original = agent.runtime_snapshot.clone();
        for field in ["session", "branch", "revision"] {
            agent.runtime_snapshot = original.clone();
            match field {
                "session" => agent.runtime_snapshot.session_id = None,
                "branch" => agent.runtime_snapshot.active_context.branch_id = "other".into(),
                _ => agent.runtime_snapshot.context_scope_revision += 1,
            }
            let error = work(&agent, true)
                .err()
                .expect("scope mismatch")
                .to_string();
            assert!(error.contains("scope changed"), "{field}: {error}");
            assert!(error.contains("active session="), "{error}");
            assert!(error.contains("persisted session="), "{error}");
        }
    }
}
