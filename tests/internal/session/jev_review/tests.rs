use super::*;
use crate::agent::{ConfiguredPrimaryRouteFactory, PrimaryRouteFactory};
use crate::tui::components::reviewer_cards;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpListener;

/// Serves one response and hands back the requests it received.
async fn endpoint(
    status: &str,
    body: &str,
) -> (String, Arc<Mutex<Vec<String>>>, tokio::task::JoinHandle<()>) {
    serving_endpoint(status, body, 1).await
}

/// Serves the same response to each of `requests` connections.
async fn serving_endpoint(
    status: &str,
    body: &str,
    requests: usize,
) -> (String, Arc<Mutex<Vec<String>>>, tokio::task::JoinHandle<()>) {
    let listener = TcpListener::bind("127.0.0.1:0")
        .await
        .expect("listener should bind");
    let address = listener.local_addr().expect("listener address");
    let seen: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));
    let captured = Arc::clone(&seen);
    let status = status.to_string();
    let body = body.to_string();
    let server = tokio::spawn(async move {
        for _ in 0..requests {
            serve_one(&listener, &captured, &status, &body).await;
        }
    });
    (format!("http://{address}"), seen, server)
}

/// Reads one request, records it, and answers with `status` and `body`.
async fn serve_one(
    listener: &TcpListener,
    captured: &Arc<Mutex<Vec<String>>>,
    status: &str,
    body: &str,
) {
    let (mut socket, _) = listener.accept().await.expect("connection");
    let mut request = Vec::new();
    let mut buffer = [0u8; 4096];
    loop {
        let read = socket.read(&mut buffer).await.expect("request read");
        if read == 0 {
            break;
        }
        request.extend_from_slice(&buffer[..read]);
        let text = String::from_utf8_lossy(&request);
        if let Some(headers_end) = text.find("\r\n\r\n") {
            let length = text
                .lines()
                .find_map(|line| {
                    line.strip_prefix("content-length: ")
                        .or_else(|| line.strip_prefix("Content-Length: "))
                })
                .and_then(|value| value.trim().parse::<usize>().ok())
                .unwrap_or(0);
            if request.len() >= headers_end + 4 + length {
                break;
            }
        }
    }
    captured
        .lock()
        .expect("capture lock")
        .push(String::from_utf8_lossy(&request).to_string());
    let response = format!(
        "HTTP/1.1 {status}\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{body}",
        body.len()
    );
    socket
        .write_all(response.as_bytes())
        .await
        .expect("response write");
    let _ = socket.shutdown().await;
}

fn review_config(base_url: String, credential: &str) -> JevReviewConfig {
    let endpoint = format!("{}/v1/systemone", base_url.trim_end_matches('/'));
    JevReviewConfig {
        provider: "typesafe".into(),
        base_url,
        endpoint,
        model: "jev-latest".into(),
        credential: credential.into(),
        timeout_secs: 5,
    }
}

fn permission_request() -> PermissionRequest {
    PermissionRequest {
        call_id: Some("call-review".into()),
        tool: "shell__exec".into(),
        args: json!({"command": "pwd"}),
        class: crate::permission::ToolPermissionClass::Command,
        summary: "Run a read-only command".into(),
        preview: Some("pwd".into()),
        can_allow_always: false,
        grant_summary: None,
    }
}

fn client() -> reqwest::Client {
    reqwest::Client::builder()
        .timeout(Duration::from_secs(5))
        .no_proxy()
        .build()
        .expect("client")
}

#[test]
fn truncate_marks_what_it_cut() {
    assert_eq!(truncate("abc", 3), "abc");
    assert_eq!(truncate("abcd", 3), "ab…");
}

#[test]
fn loopback_endpoints_cover_ipv6_and_named_hosts() {
    assert!(is_loopback_endpoint("http://[::1]:8080"));
    assert!(is_loopback_endpoint("http://[0:0:0:0:0:0:0:1]:8080"));
    assert!(is_loopback_endpoint("http://127.0.0.1:1234"));
    assert!(is_loopback_endpoint("http://localhost:3000"));
    assert!(!is_loopback_endpoint("http://[::2]:8080"));
    assert!(!is_loopback_endpoint("https://example.invalid/v1"));
}

mod integration_tests {
    use super::*;

    #[tokio::test]
    async fn ask_sends_the_frozen_question_and_reads_the_choice() {
        let (base_url, seen, server) = endpoint(
            "200 OK",
            r#"{"answers":{"approval":{"type":"choice","choice":"ask_user","confidence":0.32,"probabilities":{"execute":0.45,"ask_user":0.55,"refuse":0.0}}}}"#,
        )
        .await;
        let config = review_config(base_url, "test-key");
        let answer = ask_jev(&client(), &config, json!({"tool": "shell__exec"}))
            .await
            .expect("answer");

        assert_eq!(answer.choice, "ask_user");
        assert_eq!(outcome_for(&answer.choice), Some(AutoReviewOutcome::Ask));
        let rationale = describe(&answer);
        assert!(rationale.contains("ask_user"), "{rationale}");
        assert!(rationale.contains("0.32"), "{rationale}");

        {
            let requests = seen.lock().expect("capture lock");
            let request = requests.first().expect("one request");
            assert!(request.starts_with("POST /v1/systemone "), "{request}");
            assert!(
                request.contains("authorization: Bearer test-key"),
                "{request}"
            );
            assert!(request.contains("\"model\":\"jev-latest\""), "{request}");
            assert!(request.contains("\"type\":\"choice\""), "{request}");
            for (outcome, _) in REVIEW_OUTCOME_CRITERIA {
                assert!(request.contains(outcome), "{request}");
            }
        }
        server.await.expect("server");
    }

    #[tokio::test]
    async fn ask_leaves_an_unknown_decision_unmapped() {
        let (base_url, _, server) = endpoint(
            "200 OK",
            r#"{"answers":{"approval":{"choice":"maybe","confidence":0.4}}}"#,
        )
        .await;
        let config = review_config(base_url, "test-key");
        let answer = ask_jev(&client(), &config, json!({}))
            .await
            .expect("answer");

        assert_eq!(answer.choice, "maybe");
        assert_eq!(outcome_for(&answer.choice), None);
        server.await.expect("server");
    }

    #[tokio::test]
    async fn ask_reports_a_rejected_credential() {
        let (base_url, _, server) = endpoint("401 Unauthorized", "{\"error\":\"nope\"}").await;
        let config = review_config(base_url, "bad-key");
        let error = ask_jev(&client(), &config, json!({}))
            .await
            .expect_err("401 should fail");

        assert!(format!("{error:#}").contains("401"), "{error:#}");
        server.await.expect("server");
    }

    #[tokio::test]
    async fn ask_reports_a_malformed_body() {
        let (base_url, _, server) = endpoint("200 OK", "not json").await;
        let config = review_config(base_url, "test-key");
        let error = ask_jev(&client(), &config, json!({}))
            .await
            .expect_err("malformed body should fail");

        assert!(format!("{error:#}").contains("malformed body"), "{error:#}");
        server.await.expect("server");
    }

    #[tokio::test]
    async fn ask_without_a_credential_never_reaches_the_endpoint() {
        let (base_url, seen, server) = endpoint("200 OK", "{}").await;
        let config = review_config(base_url, "");
        let error = ask_jev(&client(), &config, json!({}))
            .await
            .expect_err("missing credential should fail");

        assert!(
            format!("{error:#}").contains("[providers.typesafe.auth]"),
            "{error:#}"
        );
        assert!(seen.lock().expect("capture lock").is_empty());
        server.abort();
    }

    #[tokio::test]
    async fn review_records_the_exchange_in_a_reviewer_child_session() {
        tokio::time::timeout(Duration::from_secs(30), async {
            let directory = tempfile::tempdir().expect("config dir");
            let config_path = directory.path().join("letcode.toml");
            std::fs::write(
                &config_path,
                r#"
active_provider = "test"
[providers.test]
protocol = "responses"
default_model = "current"
[providers.test.auth]
type = "bearer"
credential = "test-key"
[providers.test.endpoints]
base_url = "http://127.0.0.1:1"
[providers.test.models.current]
"#,
            )
            .expect("write config");
            let config = crate::config::AppConfig::load_from_path(&config_path).expect("config");

            let sessions = tempfile::tempdir().expect("sessions dir");
            let transcript = Arc::new(Mutex::new(
                TranscriptRecorder::create(sessions.path()).expect("parent transcript"),
            ));
            let route = crate::config::ModelRoute::new("test", "current");
            transcript
                .lock()
                .expect("transcript")
                .record_session_started(route.display_name())
                .expect("record session started");

            let mut parent = Agent::new("current", 1, 1);
            let primary_factory = Arc::new(ConfiguredPrimaryRouteFactory::new_with_runtime_catalog(
                config.providers.clone(),
                config.global.retry.clone(),
                config.runtime_catalog.clone(),
            ));
            parent.apply_prepared_route(
                primary_factory
                    .prepare_route(route.clone())
                    .expect("prepare primary route"),
            );
            parent.set_primary_route_factory(primary_factory);
            parent.set_subagent_child_factory(Arc::new(
                crate::subagent::ExpertRouteFactory::new_with_policies(
                    [("reviewer".to_string(), None, Vec::new(), None)],
                    &config.providers,
                    &config.global.retry,
                )
                .expect("reviewer route factory"),
            ));

            let (base_url, seen, server) = serving_endpoint(
                "200 OK",
                r#"{"answers":{"approval":{"choice":"execute","confidence":0.93,"probabilities":{"execute":0.93,"ask_user":0.05,"refuse":0.02}}}}"#,
                2,
            )
            .await;
            let reviewer = JevReviewer::new(
                review_config(base_url, "test-key"),
                Arc::clone(&transcript),
                None,
                SubagentPool::new(),
                sessions.path().to_path_buf(),
            )
            .expect("reviewer");

            let first = reviewer
                .review(&parent, permission_request(), None)
                .await
                .expect("first review");
            assert_eq!(first.outcome, AutoReviewOutcome::AllowOnce);

            let child_session_id = reviewer.sticky_child_id().expect("reviewer child session");
            let child_path = crate::transcript::child_sessions_dir(sessions.path())
                .join(format!("{child_session_id}.jsonl"));
            let child_records =
                read_records_allow_partial_tail(&child_path).expect("child records");
            let request_card = child_records
                .iter()
                .find_map(|record| match &record.event {
                    TranscriptEvent::UserMessage { content, .. } => {
                        reviewer_cards::parse_review_request(&content.text)
                    }
                    _ => None,
                })
                .expect("request card");
            assert_eq!(request_card.tool, "shell__exec");
            assert_eq!(request_card.class.as_deref(), Some("command"));
            let decision_card = child_records
                .iter()
                .find_map(|record| match &record.event {
                    TranscriptEvent::AssistantTurn(turn) => turn
                        .text
                        .as_deref()
                        .and_then(reviewer_cards::parse_review_decision),
                    _ => None,
                })
                .expect("decision card");
            assert_eq!(decision_card.decision, "allow_once");
            assert!(decision_card.rationale.contains("jev: execute"));

            let parent_path = transcript.lock().expect("transcript").path().to_path_buf();
            let decision = read_records_allow_partial_tail(&parent_path)
                .expect("parent records")
                .into_iter()
                .find_map(|record| match record.event {
                    TranscriptEvent::PermissionDecision {
                        reviewer,
                        reviewer_child_session_id,
                        ..
                    } => Some((reviewer, reviewer_child_session_id)),
                    _ => None,
                })
                .expect("parent permission decision");
            assert_eq!(decision.0.as_deref(), Some("auto"));
            assert_eq!(decision.1.as_deref(), Some(child_session_id.as_str()));

            let second = reviewer
                .review(&parent, permission_request(), None)
                .await
                .expect("second review");
            assert_eq!(second.outcome, AutoReviewOutcome::AllowOnce);
            assert_eq!(
                reviewer.sticky_child_id().as_deref(),
                Some(child_session_id.as_str())
            );
            let child_records =
                read_records_allow_partial_tail(&child_path).expect("child records");
            assert_eq!(
                child_records
                    .iter()
                    .filter(|record| matches!(
                        &record.event,
                        TranscriptEvent::UserMessage { .. }
                    ))
                    .count(),
                2,
                "the second review must reuse the reviewer child session"
            );

            server.await.expect("server");
            assert_eq!(seen.lock().expect("capture lock").len(), 2);
        })
        .await
        .expect("review flow timed out");
    }
}
