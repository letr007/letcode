use std::path::PathBuf;
use std::process::{Output, Stdio};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use serde_json::{Value, json};
use tempfile::TempDir;
use tokio::io::{AsyncRead, AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};
use tokio::process::{Child, ChildStdin, Command};
use tokio::sync::Notify;
use tokio::task::{JoinHandle, JoinSet};
use tokio::time::timeout;

const PROCESS_TIMEOUT: Duration = Duration::from_secs(45);
const IO_TIMEOUT: Duration = Duration::from_secs(10);
const MAX_BYTES: usize = 2 * 1024 * 1024;
const PROMPT: &str = "Read fixture.txt using fs__read and report the result.";
const FOLLOWUP: &str = "Continue the saved file discussion.";
const FILE_CONTENT: &str = "cli-process-file-evidence-7319\n";
const ANSWER: &str = "The file was read successfully.";
const RESUMED_ANSWER: &str = "The saved file discussion was resumed.";
const CALL_ID: &str = "read-fixture-call";
const MODEL_ERROR: &str = "scripted_invalid_request";
const PEER_PROMPT: &str = "Send the peer a greeting and finish this turn.";
const PEER_GREETING: &str = "peer-greeting-evidence-5821";
const PEER_REPLY: &str = "peer-reply-evidence-9043";
const PEER_SENT: &str = "The greeting was written; this turn is complete.";
const PEER_REPLIED: &str = "The reply was written.";
const PEER_ANSWER: &str = "The peer reply was received.";
const DISCOVER_CALL: &str = "peer-discover-call";
const SEND_CALL: &str = "peer-greeting-call";
const REPLY_CALL: &str = "peer-reply-call";

type Requests = Arc<Mutex<Vec<Value>>>;

struct Sandbox {
    _root: TempDir,
    home: PathBuf,
    workspace: PathBuf,
    config: PathBuf,
    sessions: PathBuf,
    temp: PathBuf,
}

impl Sandbox {
    fn new() -> Self {
        let root = tempfile::tempdir().unwrap();
        let home = root.path().join("home");
        let workspace = root.path().join("workspace");
        let config = home.join(".config/letcode/letcode.toml");
        let sessions = root.path().join("sessions");
        let temp = root.path().join("tmp");
        for path in [
            &home,
            &workspace,
            config.parent().unwrap(),
            &sessions,
            &temp,
        ] {
            std::fs::create_dir_all(path).unwrap();
        }
        std::fs::create_dir(workspace.join(".git")).unwrap();
        std::fs::write(workspace.join(".env"), "").unwrap();
        Self {
            _root: root,
            home,
            workspace,
            config,
            sessions,
            temp,
        }
    }

    fn configure(&self, server: &ModelServer) {
        let config = format!(
            r#"active_provider = "local"

[global]
sessions_dir = {sessions}
log_file = {log}
max_iterations = 6
max_tool_calls = 4

[global.retry]
enabled = false

[permissions]
mode = "default"

[providers.local]
protocol = "responses"
default_model = "fixture-model"
flavor = "standard"

[providers.local.auth]
type = "bearer"
credential = "dummy-cli-test-key"

[providers.local.endpoints]
base_url = "http://{address}/v1"

[providers.local.models.fixture-model.capabilities]
tools = true
"#,
            sessions = json!(self.sessions.to_str().unwrap()),
            log = json!(self.home.join("letcode.log").to_str().unwrap()),
            address = server.address,
        );
        std::fs::write(&self.config, config).unwrap();
    }

    fn command(&self, args: &[&str]) -> Command {
        let mut command = Command::new(env!("CARGO_BIN_EXE_letcode"));
        command.env_clear();
        for key in [
            "PATH",
            "SystemRoot",
            "WINDIR",
            "SystemDrive",
            "COMSPEC",
            "PATHEXT",
        ] {
            if let Some(value) = std::env::var_os(key) {
                command.env(key, value);
            }
        }
        command
            .env("HOME", &self.home)
            .env("USERPROFILE", &self.home)
            .env("XDG_CONFIG_HOME", self.home.join(".config"))
            .env("XDG_CACHE_HOME", self.home.join(".cache"))
            .env("XDG_DATA_HOME", self.home.join(".local/share"))
            .env("XDG_STATE_HOME", self.home.join(".local/state"))
            .env("APPDATA", self.home.join("AppData/Roaming"))
            .env("LOCALAPPDATA", self.home.join("AppData/Local"))
            .env("TEMP", &self.temp)
            .env("TMP", &self.temp)
            .env("TMPDIR", &self.temp)
            .env("LANG", "C")
            .env("LC_ALL", "C")
            .current_dir(&self.workspace)
            .args(args)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .kill_on_drop(true);
        command
    }

    async fn run(&self, args: &[&str], input: &str, requests: Option<&Requests>) -> Output {
        let mut child = self.command(args).spawn().unwrap();
        let mut stdin = child.stdin.take().unwrap();
        let mut stdout = child.stdout.take().unwrap().take(MAX_BYTES as u64);
        let mut stderr = child.stderr.take().unwrap().take(MAX_BYTES as u64);
        let mut out = Vec::new();
        let mut err = Vec::new();
        let result = timeout(PROCESS_TIMEOUT, async {
            let (written, status, read_out, read_err) = tokio::join!(
                async {
                    stdin.write_all(input.as_bytes()).await?;
                    stdin.shutdown().await
                },
                child.wait(),
                stdout.read_to_end(&mut out),
                stderr.read_to_end(&mut err),
            );
            written?;
            read_out?;
            read_err?;
            status
        })
        .await;
        let diagnostics = || {
            format!(
                "args={args:?}\nstdout={}\nstderr={}\nrequests={:#?}",
                String::from_utf8_lossy(&out),
                String::from_utf8_lossy(&err),
                requests.map(|requests| requests.lock().unwrap().clone()),
            )
        };
        let status = match result {
            Ok(Ok(status)) => status,
            other => {
                let _ = child.start_kill();
                let _ = timeout(IO_TIMEOUT, child.wait()).await;
                panic!(
                    "CLI process failed or exceeded {PROCESS_TIMEOUT:?}: {other:?}\n{}",
                    diagnostics()
                );
            }
        };
        assert!(
            out.len() < MAX_BYTES && err.len() < MAX_BYTES,
            "output limit exceeded\n{}",
            diagnostics()
        );
        Output {
            status,
            stdout: out,
            stderr: err,
        }
    }

    async fn peers(&self, expected: usize) -> Vec<Value> {
        let directory = self.home.join(".config/letcode/peers");
        timeout(IO_TIMEOUT, async {
            loop {
                let peers: Vec<Value> = if directory.exists() {
                    std::fs::read_dir(&directory)
                        .unwrap()
                        .map(|entry| entry.unwrap().path())
                        .filter(|path| {
                            path.extension()
                                .is_some_and(|extension| extension == "json")
                        })
                        .map(|path| serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap())
                        .collect()
                } else {
                    Vec::new()
                };
                if peers.len() == expected {
                    return peers;
                }
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
        })
        .await
        .expect("peer registration deadline exceeded")
    }

    fn journal(&self, session_id: &str) -> (PathBuf, Vec<u8>, Vec<Value>) {
        let mut pending = vec![self.sessions.clone()];
        let mut matches = Vec::new();
        let mut entries = 0;
        while let Some(dir) = pending.pop() {
            for entry in std::fs::read_dir(dir).unwrap() {
                entries += 1;
                assert!(entries <= 128, "unexpected session directory size");
                let entry = entry.unwrap();
                let path = entry.path();
                if entry.file_type().unwrap().is_dir() {
                    pending.push(path);
                } else if path
                    .extension()
                    .is_some_and(|extension| extension == "jsonl")
                {
                    let bytes = std::fs::read(&path).unwrap();
                    let records: Vec<Value> = std::str::from_utf8(&bytes)
                        .unwrap()
                        .lines()
                        .map(|line| serde_json::from_str(line).unwrap())
                        .collect();
                    if records
                        .first()
                        .is_some_and(|record| record["session_id"] == session_id)
                    {
                        assert!(
                            records
                                .iter()
                                .all(|record| record["session_id"] == session_id)
                        );
                        assert!(
                            records
                                .windows(2)
                                .all(|pair| pair[0]["sequence"].as_u64().unwrap()
                                    < pair[1]["sequence"].as_u64().unwrap())
                        );
                        matches.push((path, bytes, records));
                    }
                }
            }
        }
        assert_eq!(matches.len(), 1, "expected one journal for {session_id}");
        matches.pop().unwrap()
    }
}

struct Repl {
    child: Child,
    stdin: ChildStdin,
    stdout: Arc<Mutex<Vec<u8>>>,
    out_task: JoinHandle<Vec<u8>>,
    err_task: JoinHandle<Vec<u8>>,
}

impl Repl {
    fn start(sandbox: &Sandbox) -> Self {
        let mut child = sandbox.command(&["--cli"]).spawn().unwrap();
        let stdin = child.stdin.take().unwrap();
        let stdout = Arc::new(Mutex::new(Vec::new()));
        let out_task = tokio::spawn(capture_output(child.stdout.take().unwrap(), stdout.clone()));
        let err_task = tokio::spawn(capture_output(
            child.stderr.take().unwrap(),
            Arc::new(Mutex::new(Vec::new())),
        ));
        Self {
            child,
            stdin,
            stdout,
            out_task,
            err_task,
        }
    }

    async fn wait_for(&self, text: &str) {
        timeout(IO_TIMEOUT, async {
            loop {
                if String::from_utf8_lossy(&self.stdout.lock().unwrap()).contains(text) {
                    return;
                }
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
        })
        .await
        .unwrap_or_else(|_| {
            panic!(
                "missing {text:?} in CLI stdout: {}",
                String::from_utf8_lossy(&self.stdout.lock().unwrap())
            )
        });
    }

    async fn exit(mut self) -> Output {
        self.stdin.write_all(b"/exit\n").await.unwrap();
        let status = timeout(IO_TIMEOUT, self.child.wait())
            .await
            .unwrap()
            .unwrap();
        Output {
            status,
            stdout: self.out_task.await.unwrap(),
            stderr: self.err_task.await.unwrap(),
        }
    }
}

async fn capture_output(
    mut reader: impl AsyncRead + Unpin,
    captured: Arc<Mutex<Vec<u8>>>,
) -> Vec<u8> {
    timeout(PROCESS_TIMEOUT, async {
        let mut chunk = [0; 4096];
        loop {
            let read = reader.read(&mut chunk).await.unwrap();
            if read == 0 {
                return captured.lock().unwrap().clone();
            }
            let mut output = captured.lock().unwrap();
            output.extend_from_slice(&chunk[..read]);
            assert!(output.len() < MAX_BYTES, "CLI output limit exceeded");
        }
    })
    .await
    .expect("CLI output deadline exceeded")
}

#[derive(Clone)]
enum ModelScript {
    File { fail: bool },
    Peer { reply_allowed: Arc<Notify> },
}

struct ModelServer {
    address: std::net::SocketAddr,
    requests: Requests,
    errors: Arc<Mutex<Vec<String>>>,
    task: JoinHandle<()>,
}

impl ModelServer {
    async fn start(fail: bool) -> Self {
        Self::scripted(ModelScript::File { fail }).await
    }

    async fn scripted(script: ModelScript) -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let requests = Arc::new(Mutex::new(Vec::new()));
        let errors = Arc::new(Mutex::new(Vec::new()));
        let seen = requests.clone();
        let failures = errors.clone();
        let task = tokio::spawn(async move {
            let mut connections = JoinSet::new();
            let serve = async {
                loop {
                    tokio::select! {
                        accepted = listener.accept() => {
                            let (stream, _) = accepted.unwrap();
                            let seen = seen.clone();
                            let failures = failures.clone();
                            let script = script.clone();
                            connections.spawn(async move {
                                match timeout(IO_TIMEOUT, serve_request(stream, &seen, &script)).await {
                                    Ok(Ok(())) => {}
                                    other => failures.lock().unwrap().push(format!("model connection failed: {other:?}")),
                                }
                            });
                        }
                        Some(joined) = connections.join_next(), if !connections.is_empty() => {
                            if let Err(error) = joined {
                                failures.lock().unwrap().push(format!("model task failed: {error}"));
                            }
                        }
                    }
                }
            };
            if timeout(Duration::from_secs(120), serve).await.is_err() {
                failures
                    .lock()
                    .unwrap()
                    .push("model server deadline exceeded".into());
            }
        });
        Self {
            address,
            requests,
            errors,
            task,
        }
    }

    fn main_requests(&self) -> Vec<Value> {
        assert!(
            self.errors.lock().unwrap().is_empty(),
            "server errors: {:?}; requests: {:?}",
            self.errors.lock().unwrap(),
            self.requests.lock().unwrap()
        );
        self.requests
            .lock()
            .unwrap()
            .iter()
            .filter(|request| has_tools(request))
            .cloned()
            .collect()
    }
}

impl Drop for ModelServer {
    fn drop(&mut self) {
        self.task.abort();
    }
}

fn has_tools(request: &Value) -> bool {
    request["tools"]
        .as_array()
        .is_some_and(|tools| !tools.is_empty())
}

async fn serve_request(
    mut stream: TcpStream,
    requests: &Requests,
    script: &ModelScript,
) -> Result<(), String> {
    let mut bytes = Vec::new();
    let mut chunk = [0; 4096];
    let (header_end, content_length) = loop {
        let read = stream
            .read(&mut chunk)
            .await
            .map_err(|error| error.to_string())?;
        if read == 0 {
            return Err("request ended before headers".into());
        }
        bytes.extend_from_slice(&chunk[..read]);
        if bytes.len() > MAX_BYTES {
            return Err("request limit exceeded".into());
        }
        if let Some(end) = bytes.windows(4).position(|part| part == b"\r\n\r\n") {
            let headers = std::str::from_utf8(&bytes[..end]).map_err(|error| error.to_string())?;
            let request_line = headers.lines().next().ok_or("missing request line")?;
            let parts: Vec<_> = request_line.split_whitespace().collect();
            if parts.len() != 3
                || parts[0] != "POST"
                || parts[1].split('?').next() != Some("/v1/responses")
                || parts[2] != "HTTP/1.1"
            {
                return Err(format!("unexpected request target: {request_line}"));
            }
            if !headers
                .lines()
                .any(|line| line.eq_ignore_ascii_case("authorization: Bearer dummy-cli-test-key"))
            {
                return Err("missing dummy bearer credential".into());
            }
            let length = headers
                .lines()
                .find_map(|line| {
                    let (name, value) = line.split_once(':')?;
                    name.eq_ignore_ascii_case("content-length")
                        .then(|| value.trim().parse::<usize>())
                })
                .ok_or("missing content length")?
                .map_err(|error| error.to_string())?;
            if length > MAX_BYTES {
                return Err("request body limit exceeded".into());
            }
            break (end + 4, length);
        }
    };
    while bytes.len() < header_end + content_length {
        let read = stream
            .read(&mut chunk)
            .await
            .map_err(|error| error.to_string())?;
        if read == 0 {
            return Err("request ended before body".into());
        }
        bytes.extend_from_slice(&chunk[..read]);
    }
    let request: Value = serde_json::from_slice(&bytes[header_end..header_end + content_length])
        .map_err(|error| error.to_string())?;
    requests.lock().unwrap().push(request.clone());
    let (status, content_type, body) = if matches!(script, ModelScript::File { fail: true })
        && has_tools(&request)
    {
        (
            "400 Bad Request",
            "application/json",
            json!({"error": {"message": MODEL_ERROR, "type": "invalid_request_error"}}).to_string(),
        )
    } else {
        let input = request["input"].to_string();
        let output = if !has_tools(&request) {
            message("File discussion")
        } else if let ModelScript::Peer { reply_allowed } = script {
            peer_response(&request, reply_allowed).await?
        } else if input.contains(FOLLOWUP) {
            message(RESUMED_ANSWER)
        } else if tool_output(&request).is_some() {
            message(ANSWER)
        } else if input.contains(PROMPT) {
            json!({"type": "function_call", "id": "fc_fixture", "call_id": CALL_ID, "name": "fs__read", "arguments": json!({"path":"fixture.txt", "offset":1, "limit":10}).to_string()})
        } else {
            return Err(format!("unexpected model input: {input}"));
        };
        let response = json!({"id":"resp_fixture", "status":"completed", "output":[output], "usage":{"input_tokens":50,"output_tokens":10,"total_tokens":60}});
        if request["stream"] == true {
            let mut events = Vec::new();
            if output["type"] == "function_call" {
                events.push(
                    json!({"type":"response.output_item.added", "output_index":0, "item":output}),
                );
                events.push(
                    json!({"type":"response.output_item.done", "output_index":0, "item":output}),
                );
            } else {
                events.push(json!({"type":"response.output_text.delta", "delta":output["content"][0]["text"]}));
            }
            events.push(json!({"type":"response.completed", "response":response}));
            let body = events
                .iter()
                .map(|event| format!("data: {event}\n\n"))
                .collect::<String>();
            ("200 OK", "text/event-stream", body)
        } else {
            ("200 OK", "application/json", response.to_string())
        }
    };
    let response = format!(
        "HTTP/1.1 {status}\r\nContent-Type: {content_type}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
        body.len()
    );
    stream
        .write_all(response.as_bytes())
        .await
        .map_err(|error| error.to_string())?;
    stream.shutdown().await.map_err(|error| error.to_string())
}

fn latest_user(request: &Value) -> (&Value, &[Value]) {
    let input = request["input"].as_array().unwrap();
    let index = input
        .iter()
        .rposition(|item| item["role"] == "user")
        .unwrap();
    (&input[index], &input[index + 1..])
}

fn user_text(user: &Value) -> String {
    user["content"]
        .as_array()
        .unwrap()
        .iter()
        .filter_map(|part| part["text"].as_str())
        .collect::<Vec<_>>()
        .join("\n")
}

async fn peer_response(request: &Value, reply_allowed: &Notify) -> Result<Value, String> {
    let (user, turn) = latest_user(request);
    if let Some(output) = turn
        .iter()
        .rev()
        .find(|item| item["type"] == "function_call_output")
    {
        let result: Value = serde_json::from_str(output["output"].as_str().unwrap()).unwrap();
        if result["ok"] != true {
            return Err(format!("peer tool failed: {result}"));
        }
        return match output["call_id"].as_str().unwrap() {
            DISCOVER_CALL => {
                let peers = result["data"]["peers"].as_array().unwrap();
                assert_eq!(peers.len(), 1, "{result}");
                assert_ne!(peers[0]["address"], result["data"]["self"]);
                Ok(peer_call(
                    SEND_CALL,
                    "peer__send",
                    json!({
                        "to": peers[0]["address"], "message": PEER_GREETING,
                    }),
                ))
            }
            SEND_CALL | REPLY_CALL => {
                assert_eq!(result["data"]["status"], "written", "{result}");
                Ok(message(if output["call_id"] == SEND_CALL {
                    PEER_SENT
                } else {
                    PEER_REPLIED
                }))
            }
            _ => Err(format!("unexpected peer tool output: {output}")),
        };
    }
    let text = user_text(user);
    if text == PEER_PROMPT {
        Ok(peer_call(DISCOVER_CALL, "peer__discover", json!({})))
    } else if text.contains(PEER_GREETING) {
        let address = peer_source(&text);
        reply_allowed.notified().await;
        Ok(peer_call(
            REPLY_CALL,
            "peer__send",
            json!({
                "to": address, "message": PEER_REPLY,
            }),
        ))
    } else if text.contains(PEER_REPLY) {
        Ok(message(PEER_ANSWER))
    } else {
        Err(format!("unexpected peer model input: {text}"))
    }
}

fn peer_source(text: &str) -> Value {
    let source = text.strip_prefix("[Peer message from ").unwrap();
    let (source, _) = source
        .split_once(". The source is routing metadata.]\n")
        .unwrap();
    serde_json::from_str(source).unwrap()
}

fn peer_call(call_id: &str, name: &str, arguments: Value) -> Value {
    json!({"type":"function_call", "id":format!("fc_{call_id}"), "call_id":call_id, "name":name, "arguments":arguments.to_string()})
}

fn message(text: &str) -> Value {
    json!({"type":"message", "id":"msg_fixture", "role":"assistant", "status":"completed", "content":[{"type":"output_text", "text":text, "annotations":[]}]})
}

fn tool_output(request: &Value) -> Option<&Value> {
    request["input"]
        .as_array()?
        .iter()
        .find(|item| item["type"] == "function_call_output" && item["call_id"] == CALL_ID)
}

fn assert_file_output(request: &Value) {
    let output = tool_output(request).expect("model request must contain the real fs__read result");
    let output: Value = serde_json::from_str(output["output"].as_str().unwrap()).unwrap();
    assert_eq!(output["data"]["content"], FILE_CONTENT, "{output:#}");
}

fn json_stdout(output: &Output, requests: &Requests) -> Value {
    serde_json::from_slice(&output.stdout).unwrap_or_else(|error| {
        panic!(
            "invalid stdout JSON: {error}; status={}\nstdout={}\nstderr={}\nrequests={:#?}",
            output.status,
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr),
            requests.lock().unwrap()
        );
    })
}

fn assert_success(output: &Output) {
    assert!(
        output.status.success(),
        "status={}\nstdout={}\nstderr={}",
        output.status,
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
}

#[tokio::test]
async fn entrypoints_report_version_and_reject_invalid_arguments_and_config() {
    let sandbox = Sandbox::new();
    let version = sandbox.run(&["--version"], "", None).await;
    assert_success(&version);
    assert_eq!(
        String::from_utf8(version.stdout).unwrap(),
        format!("letcode {}\n", env!("CARGO_PKG_VERSION"))
    );
    assert!(version.stderr.is_empty());
    for (args, expected) in [
        (vec!["--invalid-cli-test-flag"], "unknown flag"),
        (vec!["-p"], "requires a prompt value"),
        (vec!["--json"], "requires --prompt/-p"),
    ] {
        let output = sandbox.run(&args, "", None).await;
        assert!(!output.status.success());
        assert!(output.stdout.is_empty());
        assert!(
            String::from_utf8_lossy(&output.stderr).contains(expected),
            "{output:?}"
        );
    }
    std::fs::write(&sandbox.config, "active_provider = [").unwrap();
    let invalid = sandbox.run(&["-p", "hello", "--json"], "", None).await;
    assert!(!invalid.status.success());
    assert!(invalid.stdout.is_empty());
    assert!(
        String::from_utf8_lossy(&invalid.stderr).contains("TOML"),
        "{invalid:?}"
    );
    let invalid = sandbox
        .run(
            &["config", "validate", sandbox.config.to_str().unwrap()],
            "",
            None,
        )
        .await;
    assert!(!invalid.status.success());
    assert!(invalid.stdout.is_empty());
    assert!(
        String::from_utf8_lossy(&invalid.stderr).contains("invalid:"),
        "{invalid:?}"
    );
    assert_eq!(std::fs::read_dir(&sandbox.sessions).unwrap().count(), 0);
}

#[tokio::test]
async fn one_shot_tool_result_is_persisted_and_resumed_by_another_process() {
    let sandbox = Sandbox::new();
    let server = ModelServer::start(false).await;
    sandbox.configure(&server);
    std::fs::write(sandbox.workspace.join("fixture.txt"), FILE_CONTENT).unwrap();
    let output = sandbox
        .run(&["-p", PROMPT, "--json"], "", Some(&server.requests))
        .await;
    assert_success(&output);
    let result = json_stdout(&output, &server.requests);
    assert_eq!(result["ok"], true);
    assert_eq!(result["response"], ANSWER);
    let session_id = result["session_id"]
        .as_str()
        .filter(|id| !id.is_empty())
        .unwrap();
    let requests = server.main_requests();
    assert_eq!(requests.len(), 2, "{requests:#?}");
    assert!(requests[0]["input"].to_string().contains(PROMPT));
    assert_file_output(&requests[1]);
    let (path, original, records) = sandbox.journal(session_id);
    assert!(
        records
            .iter()
            .any(|record| record["kind"] == "user_message" && record.to_string().contains(PROMPT))
    );
    let tool = records
        .iter()
        .find(|record| record["kind"] == "tool_call_finished" && record["call_id"] == CALL_ID)
        .unwrap();
    assert_eq!(tool["name"], "fs__read");
    assert_eq!(tool["ok"], true);
    assert_eq!(tool["output"]["data"]["content"], FILE_CONTENT);
    assert!(
        records
            .iter()
            .any(|record| record["kind"] == "assistant_turn" && record["text"] == ANSWER)
    );

    let input = format!("/resume {session_id}\n{FOLLOWUP}\n/exit\n");
    let resumed = sandbox
        .run(&["--cli"], &input, Some(&server.requests))
        .await;
    assert_success(&resumed);
    let stdout = String::from_utf8_lossy(&resumed.stdout);
    assert!(
        stdout.contains(&format!("resumed session {session_id}")),
        "{stdout}"
    );
    assert!(stdout.contains(RESUMED_ANSWER), "{stdout}");
    let requests = server.main_requests();
    assert_eq!(requests.len(), 3, "{requests:#?}");
    let history = requests[2]["input"].to_string();
    for text in [PROMPT, ANSWER, FOLLOWUP] {
        assert!(
            history.contains(text),
            "missing {text:?} in restored history: {history}"
        );
    }
    assert_file_output(&requests[2]);
    let (resumed_path, appended, records) = sandbox.journal(session_id);
    assert_eq!(resumed_path, path);
    assert!(appended.starts_with(&original));
    assert!(appended.len() > original.len());
    assert!(records.iter().any(|record| record["kind"] == "user_message" && record.to_string().contains(FOLLOWUP)));
    assert!(
        records
            .iter()
            .any(|record| record["kind"] == "assistant_turn" && record["text"] == RESUMED_ANSWER)
    );
}

#[tokio::test]
async fn model_rejection_returns_failure_json_and_persists_error() {
    let sandbox = Sandbox::new();
    let server = ModelServer::start(true).await;
    sandbox.configure(&server);
    let config = std::fs::read_to_string(&sandbox.config).unwrap();
    std::fs::write(
        &sandbox.config,
        config.replace("enabled = false", "enabled = true\nmax_attempts = 2"),
    )
    .unwrap();
    let output = sandbox
        .run(&["-p", PROMPT, "--json"], "", Some(&server.requests))
        .await;
    assert!(!output.status.success(), "{output:?}");
    let result = json_stdout(&output, &server.requests);
    assert_eq!(result["ok"], false);
    assert_eq!(result["response"], "");
    assert!(
        result["error"].as_str().unwrap().contains(MODEL_ERROR),
        "{result:#}"
    );
    assert!(
        String::from_utf8_lossy(&output.stderr).contains(MODEL_ERROR),
        "{output:?}"
    );
    let session_id = result["session_id"]
        .as_str()
        .filter(|id| !id.is_empty())
        .unwrap();
    let (_, _, records) = sandbox.journal(session_id);
    assert!(
        records.iter().any(|record| record["kind"] == "error"
            && record["message"].as_str().unwrap().contains(MODEL_ERROR)),
        "{records:#?}"
    );
    let requests = server.main_requests();
    assert_eq!(
        requests.len(),
        1,
        "non-retryable HTTP 400 was retried: {requests:#?}"
    );
}

#[tokio::test]
async fn repl_peers_exchange_messages_and_remove_registrations_on_exit() {
    let sandbox = Sandbox::new();
    let reply_allowed = Arc::new(Notify::new());
    let server = ModelServer::scripted(ModelScript::Peer {
        reply_allowed: reply_allowed.clone(),
    })
    .await;
    sandbox.configure(&server);
    timeout(PROCESS_TIMEOUT, async {
        let mut a = Repl::start(&sandbox);
        let a_peer = sandbox.peers(1).await.pop().unwrap();
        let b = Repl::start(&sandbox);
        let peers = sandbox.peers(2).await;
        let b_peer = peers
            .iter()
            .find(|peer| peer["address"] != a_peer["address"])
            .unwrap()
            .clone();

        a.stdin
            .write_all(format!("{PEER_PROMPT}\n").as_bytes())
            .await
            .unwrap();
        a.wait_for(PEER_SENT).await;
        let requests = server.main_requests();
        assert!(
            requests.iter().any(|request| {
                request["input"].as_array().unwrap().iter().any(|item| {
                    item["type"] == "function_call_output" && item["call_id"] == SEND_CALL
                })
            }),
            "{requests:#?}"
        );
        assert!(
            !requests
                .iter()
                .any(|request| { user_text(latest_user(request).0).contains(PEER_REPLY) }),
            "A's initial turn must finish before B is allowed to reply: {requests:#?}"
        );

        reply_allowed.notify_one();
        a.wait_for(PEER_ANSWER).await;
        b.wait_for(PEER_REPLIED).await;
        let requests = server.main_requests();
        for (payload, source) in [(PEER_GREETING, &a_peer), (PEER_REPLY, &b_peer)] {
            let request = requests
                .iter()
                .find(|request| user_text(latest_user(request).0).contains(payload))
                .unwrap_or_else(|| panic!("missing delivered {payload:?}: {requests:#?}"));
            assert_eq!(
                peer_source(&user_text(latest_user(request).0)),
                source["address"]
            );
        }
        for (call_id, name, destination, payload) in [
            (SEND_CALL, "peer__send", &b_peer, PEER_GREETING),
            (REPLY_CALL, "peer__send", &a_peer, PEER_REPLY),
        ] {
            let call = requests
                .iter()
                .flat_map(|request| request["input"].as_array().unwrap())
                .find(|item| item["type"] == "function_call" && item["call_id"] == call_id)
                .unwrap_or_else(|| panic!("missing {call_id}: {requests:#?}"));
            assert_eq!(call["name"], name);
            let arguments: Value =
                serde_json::from_str(call["arguments"].as_str().unwrap()).unwrap();
            assert_eq!(arguments["to"], destination["address"]);
            assert_eq!(arguments["message"], payload);
        }

        let (a_output, b_output) = tokio::join!(a.exit(), b.exit());
        assert_success(&a_output);
        assert_success(&b_output);
        assert!(String::from_utf8_lossy(&a_output.stdout).contains(PEER_ANSWER));
        assert!(String::from_utf8_lossy(&b_output.stdout).contains(PEER_REPLIED));
        assert!(sandbox.peers(0).await.is_empty());
        for (peer, payload, answer) in [
            (&a_peer, PEER_REPLY, PEER_ANSWER),
            (&b_peer, PEER_GREETING, PEER_REPLIED),
        ] {
            let (_, _, records) = sandbox.journal(peer["address"]["session_id"].as_str().unwrap());
            assert!(
                records.iter().any(|record| {
                    record["kind"] == "user_message" && record.to_string().contains(payload)
                }),
                "{records:#?}"
            );
            assert!(
                records.iter().any(|record| {
                    record["kind"] == "assistant_turn" && record["text"] == answer
                }),
                "{records:#?}"
            );
        }
        server.main_requests();
    })
    .await
    .expect("peer round trip deadline exceeded");
}
