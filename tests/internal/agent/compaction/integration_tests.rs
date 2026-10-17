use super::*;
use crate::request_builder::HistoryToolCall;
use crate::transcript::{TranscriptEvent, TranscriptRecorder};
use std::sync::{Arc, Mutex};
use tokio::io::{AsyncReadExt, AsyncWriteExt};

#[tokio::test]
async fn pressure_consumes_multiple_route_bounded_prefixes() {
    tokio::time::timeout(std::time::Duration::from_secs(120), async {
            let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
            let address = listener.local_addr().unwrap();
            let server = tokio::spawn(async move {
                // At most one producer call per source frame; abort on admission.
                for _ in 0..25 {
                    let (mut stream, _) = listener.accept().await.unwrap();
                    let mut bytes = Vec::new();
                    let mut buffer = [0; 8192];
                    let (header_end, length) = loop {
                        let n = stream.read(&mut buffer).await.unwrap();
                        assert!(n > 0);
                        bytes.extend_from_slice(&buffer[..n]);
                        if let Some(end) = bytes.windows(4).position(|w| w == b"\r\n\r\n") {
                            let headers = String::from_utf8_lossy(&bytes[..end]).to_lowercase();
                            let length: usize = headers.lines().find_map(|line| line.strip_prefix("content-length:")).unwrap().trim().parse().unwrap();
                            break (end + 4, length);
                        }
                    };
                    while bytes.len() < header_end + length {
                        let n = stream.read(&mut buffer).await.unwrap();
                        assert!(n > 0);
                        bytes.extend_from_slice(&buffer[..n]);
                    }
                    let request: serde_json::Value = serde_json::from_slice(&bytes[header_end..header_end + length]).unwrap();
                    let prompt = request["input"].as_array().unwrap().iter().flat_map(|item| item["content"].as_array().into_iter().flatten())
                        .filter_map(|part| part["text"].as_str())
                        .find_map(|text| serde_json::from_str::<serde_json::Value>(text).ok().filter(|value| value.get("new_messages").is_some())).unwrap();
                    let source_messages = prompt["new_messages"].as_array().unwrap();
                    let has_user_image = source_messages.iter().any(|message| message["content"]["kind"] == "user_message");
                    let image_count = request["input"].as_array().unwrap().iter()
                        .flat_map(|item| item["content"].as_array().into_iter().flatten())
                        .filter(|part| part["type"] == "input_image").count();
                    assert_eq!(image_count, usize::from(has_user_image));
                    assert!(!prompt.to_string().contains("data:image/png;base64,"));
                    let count = source_messages.len();
                    let output = format!("<episode><end>{count}</end><title>Read sources</title><importance>70</importance><detailed>Read source files</detailed><compact>Read sources</compact><anchor>Sources</anchor></episode>");
                    let delta = serde_json::json!({"type":"response.output_text.delta","delta":output});
                    let terminal = serde_json::json!({"type":"response.completed","response":{"status":"completed"}});
                    let body = format!("data: {delta}\n\ndata: {terminal}\n\n");
                    stream.write_all(format!("HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",body.len()).as_bytes()).await.unwrap();
                }
            });
            let config = crate::model_runtime::RuntimeConfig::from_toml(&format!(r#"active_provider="test"
[providers.test]
protocol="responses"
default_model="m"
[providers.test.auth]
type="none"
[providers.test.endpoints]
base_url="http://{address}"
[providers.test.models.m]
[providers.test.models.m.capabilities]
input_images=true
generation={{max_output_tokens=true}}
[providers.test.models.m.generation]
max_output_tokens=128
"#)).unwrap().resolve(&crate::model_runtime::ProtocolRegistry::builtins()).unwrap();
            let mut agent = Agent::new("m", 5, 0);
            agent.set_primary_route(crate::config::ModelRoute::new("test", "m"));
            agent.set_resolved_model_route(Some(Arc::new(config.route("test", "m").unwrap().clone())));
            agent.set_model_catalog(std::collections::HashMap::from([("m".into(), ModelRequestMetadata {
                context_window: Some(12_000), effective_input_limit_tokens: Some(8_000), max_output_tokens: Some(128), ..Default::default()
            })]));
            let directory = tempfile::tempdir().unwrap();
            let mut recorder = TranscriptRecorder::create(directory.path()).unwrap();
            recorder.record_session_started("test/m").unwrap();
            recorder.record_turn_started(crate::agent::TurnStartedEvent { turn_id: 1 }).unwrap();
            recorder.record_user_message_content(crate::user_content::UserMessageContent::new("Read the source files", vec![
                crate::user_content::UserImageAttachment {
                    id: "source-image".into(), label: "source image".into(), mime: "image/png".into(),
                    data_url: "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+jRZkAAAAASUVORK5CYII=".into(),
                }
            ])).unwrap();
            for index in 0..12 {
                let id = format!("read-{index}");
                recorder.record_assistant_tool_call_batch(None, None, None, vec![HistoryToolCall {call_id:id.clone(),name:"fs__read".into(),arguments_json:r#"{"path":"source.rs"}"#.into()}]).unwrap();
                recorder.record_tool_call_finished(&id,"fs__read",true,crate::tool::ToolResult::ok("fs__read",serde_json::json!({"content":"source detail ".repeat(600)}))).unwrap();
            }
            let recorder = Arc::new(Mutex::new(recorder));
            let projected = recorder.clone();
            agent.set_runtime_snapshot_provider(Arc::new(move || {
                let recorder = projected.lock().unwrap();
                Ok(crate::transcript::transcript_projection::project_runtime_restore_snapshot(recorder.session_id().into(),crate::transcript::read_records(recorder.path())?,crate::transcript::transcript_projection::SessionContextCursor {branch_id:None,leaf_sequence:None},&[])?.snapshot)
            }));
            agent.reload_runtime_snapshot_from_provider().unwrap();
            agent.turn.current_turn_start_index = Some(0);
            assert!(pressure_successor_request(&mut agent, ApiProtocol::Responses, &[], &[]).is_err());
            let observed = recorder.clone();
            let mut events = move |event| {
                let result = crate::agent_event_journal::persist_agent_event(&mut observed.lock().unwrap(), &event).map(|_| ());
                Box::pin(std::future::ready(result)) as BoxFuture<'_, Result<()>>
            };
            let successor = compact_for_request_pressure_with_callback(&mut agent, ApiProtocol::Responses, &[], &[], &mut events).await;
            server.abort();
            let successor = successor.unwrap();
            assert!(!successor.build.budget.truncated);
            let records = crate::transcript::read_records(recorder.lock().unwrap().path()).unwrap();
            assert!(records.iter().filter(|r| matches!(r.event, TranscriptEvent::HistoryPublished(_))).count() >= 2);
            assert!(!records.iter().any(|r| matches!(r.event, TranscriptEvent::ContextCompaction(_))));
            let first_apply = records.iter().position(|r| matches!(r.event, TranscriptEvent::HistoryApplied(_))).unwrap();
            agent.runtime_snapshot = crate::transcript::transcript_projection::project_runtime_restore_snapshot(records[0].session_id.clone(), records[..=first_apply].to_vec(), crate::transcript::transcript_projection::SessionContextCursor {branch_id:None,leaf_sequence:None}, &[]).unwrap().snapshot;
            agent.clear_active_epoch();
            let first_successor = pressure_successor_request(&mut agent, ApiProtocol::Responses, &[], &[]).err().expect("the first partial publication must still exceed the protected budget");
            assert!(is_recognized_request_budget_overflow(&first_successor));
        }).await.expect("pressure test timed out");
}
