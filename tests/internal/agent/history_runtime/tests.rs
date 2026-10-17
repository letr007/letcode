use super::*;
use crate::transcript::{TranscriptEvent, TranscriptRecorder};
use std::sync::{Arc, Mutex};
use tokio::io::{AsyncReadExt, AsyncWriteExt};

mod integration_tests {
    use super::*;

    #[tokio::test]
    async fn background_publication_preserves_new_work_and_uses_native_instructions() {
        tokio::time::timeout(std::time::Duration::from_secs(120),async {
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
                let output=format!("<episode><end>{}</end><title>Parser fix</title><importance>70</importance><detailed>Fixed UTF-8 parser offsets</detailed><compact>Parser offset fix</compact><anchor>UTF-8 parser</anchor></episode>", if attempt==0 {4} else {1});
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
            let first_work = work(&agent, true, None).unwrap().unwrap();
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
            let exchange = child_records.iter().find_map(|record| match &record.event {
                TranscriptEvent::HistorianExchange(exchange) => Some(exchange.clone()),
                _ => None,
            }).expect("an accepted publication records the exchange");
            assert_eq!(exchange.outcome, "prepared");
            assert_eq!(exchange.source_count, 4);
            assert!(exchange.payload_bytes > 0);
            assert_eq!(exchange.input_tokens, Some(200));
            assert_eq!(exchange.output_tokens, Some(60));
            assert!(exchange.response.is_none());
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
            let second_work = work(&agent, true, None).unwrap().unwrap();
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

            let retry_work = work(&agent, true, None).unwrap().unwrap();
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
