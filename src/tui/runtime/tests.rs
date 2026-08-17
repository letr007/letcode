use super::*;
use crate::agent::{Agent, SubagentInvocation};
use crate::agent::{
    AutoContinueState, CacheUsageReport, TodoItem, TodoStatus, TokenUsageEstimate,
    TurnFinalizedEvent, TurnStartedEvent,
};
use crate::config::CompactionConfig;
use crate::context_tree::{ContextNodeId, ContextTreeOp, ContextTreeState};
use crate::context_view::{
    ContextBlock, ContextBlockId, ContextBlockKind, ContextBlockSource, ContextViewProjection,
};
use crate::request_builder::{HistoryItem, ModelRequestMetadata};
use crate::runtime_context::RuntimeActiveContext;
use crate::session::AgentRunner;
use crate::session::engine::{
    ActiveSessionOperation, InterruptRequest, ManualCompactionOperation, SessionEngineCommand,
    SessionEngineControl, derive_interrupt_request, enqueue_deferred_command,
    flush_parked_commands, initial_session_metadata, manual_compaction_session_token_usage,
    next_idle_session_command, park_active_turn_command, record_interrupt_transcript,
    rehydrate_agent_from_transcript, run_manual_compaction, select_active_session_operation,
    select_manual_compaction_operation, send_subagent_interrupted, wait_for_subagent_cancel_settle,
};
use crate::session::restore::restored_session_token_usage;
use crate::session::runner::{ModelCatalogEntry, ModelCatalogReasoning, ModelCatalogUpdatedEvent};
use crate::session::{
    PermissionResponse, RunnerPermissionRequest, SessionTransportEvent, TokenUsageEvent,
};
use crate::subagent::SubagentPool;
use crate::transcript::sync_recorder_branch;
use crate::transcript::{
    ROOT_CONTEXT_BRANCH_ID, TranscriptEvent, TranscriptRecord, TranscriptRecorder, read_records,
};
use crate::tui::events::NoticeEvent;
use crate::tui::runtime::session_cleanup::{empty_session_path, remove_current_empty_session};
use crate::tui::{
    AppPhase, AssistantDeltaEvent, PermissionDecision, PermissionRequestEvent,
    PermissionResolutionEvent, SessionEvent, TimelineItem, ToolFinishedEvent, ToolOutcome,
    ToolStartedEvent, UserMessageEvent,
};
use async_openai::{Client, config::OpenAIConfig};
use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
use ratatui::{Terminal, backend::TestBackend};
use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpListener;
use tokio::sync::{Notify, mpsc, oneshot};
use tokio::task::JoinHandle;
use tokio::time::timeout;

fn event_context(session_id: &str, leaf_sequence: u64) -> RuntimeActiveContext {
    let mut snapshot =
        crate::runtime_context::RuntimeSnapshot::new(crate::transcript::ROOT_CONTEXT_BRANCH_ID)
            .with_session_id(session_id)
            .with_leaf_sequence(leaf_sequence);
    snapshot.active_context.active_node_id = snapshot
        .context_tree
        .active_node_id()
        .map(|node_id| node_id.as_str().to_string());
    RuntimeActiveContext::try_from(&snapshot).expect("test runtime context")
}

fn cache_report(actual_cached_tokens: Option<u64>) -> CacheUsageReport {
    CacheUsageReport {
        configured: true,
        hint_serialized: true,
        retention_sent: None,
        stable_prefix_segments: 2,
        stable_prompt_tokens: 400,
        volatile_prompt_tokens: 60,
        cacheable_prefix_tokens: 350,
        stable_after_boundary_tokens: 50,
        local_prefix_fingerprint: Some("prefix-a".into()),
        routing_key: Some("route-a".into()),
        actual_cached_tokens,
    }
}

fn sample_context_state() -> crate::tui::state::ContextPaneState {
    let tree = ContextTreeState::replay(&[ContextTreeOp::CreateNode {
        node_id: ContextNodeId::new("node-1").expect("node id"),
        parent_node_id: Some(ContextNodeId::root()),
        label: Some("Active task".into()),
        purpose: Some("Track current work".into()),
        block_ref: None,
        source_ref: None,
    }])
    .expect("tree");
    let mut blocks = BTreeMap::new();
    let block_id = ContextBlockId::new("block-1").expect("block id");
    blocks.insert(
        block_id.clone(),
        ContextBlock {
            block_id,
            node_id: Some("node-1".into()),
            kind: ContextBlockKind::Note,
            title: "Current plan".into(),
            detail: "Outline next steps".into(),
            source: ContextBlockSource::TranscriptSpan {
                start_sequence: 1,
                end_sequence: 2,
            },
            source_start_sequence: Some(1),
            available_sequence: Some(2),
            protected_reasons: Vec::new(),
        },
    );

    crate::tui::state::ContextPaneState {
        tree,
        view: ContextViewProjection {
            blocks,
            ..ContextViewProjection::default()
        },
        runtime_context: None,
        open_detail: None,
    }
}

fn sample_question_request(multiple: bool) -> crate::tool::QuestionRequest {
    crate::tool::QuestionRequest {
        questions: vec![crate::tool::QuestionSpec {
            question: if multiple {
                "Choose several".into()
            } else {
                "Choose one".into()
            },
            header: "Mode".into(),
            options: vec![
                crate::tool::QuestionOption {
                    label: "Fast".into(),
                    description: "Fast path".into(),
                },
                crate::tool::QuestionOption {
                    label: "Safe".into(),
                    description: "Safe path".into(),
                },
            ],
            multiple,
        }],
    }
}

fn sample_multi_question_request() -> crate::tool::QuestionRequest {
    crate::tool::QuestionRequest {
        questions: vec![
            crate::tool::QuestionSpec {
                question: "Choose one".into(),
                header: "Mode".into(),
                options: vec![crate::tool::QuestionOption {
                    label: "Fast".into(),
                    description: "Fast path".into(),
                }],
                multiple: false,
            },
            crate::tool::QuestionSpec {
                question: "Choose tone".into(),
                header: "Tone".into(),
                options: vec![crate::tool::QuestionOption {
                    label: "Warm".into(),
                    description: "Warm path".into(),
                }],
                multiple: false,
            },
        ],
    }
}

fn runtime_with_experts(available_experts: Vec<AvailableExpert>) -> TuiRuntime {
    static NEXT_RUNTIME_DIR: AtomicU64 = AtomicU64::new(0);

    let (_tx, rx) = mpsc::unbounded_channel();
    let base = std::env::temp_dir().join(format!(
        "letcode-tui-runtime-test-{}-{}",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("time ok")
            .as_nanos(),
        NEXT_RUNTIME_DIR.fetch_add(1, Ordering::Relaxed)
    ));
    TuiRuntime::new(
        TuiState::new("gpt-5.5", "GPT-5.5", "default"),
        rx,
        vec![AvailableModel::with_context_window_and_reasoning(
            "gpt-5.5",
            "GPT-5.5",
            None,
            None,
            vec![
                ModelReasoningEffort::None,
                ModelReasoningEffort::Minimal,
                ModelReasoningEffort::Low,
                ModelReasoningEffort::Medium,
                ModelReasoningEffort::High,
                ModelReasoningEffort::Xhigh,
            ],
        )],
        available_experts,
        std::env::temp_dir(),
        base,
    )
}

fn runtime() -> TuiRuntime {
    runtime_with_experts(Vec::new())
}

fn render_runtime_transcript(runtime: &mut TuiRuntime) {
    let backend = TestBackend::new(120, 40);
    let mut terminal = Terminal::new(backend).expect("create test terminal");
    terminal
        .draw(|frame| crate::tui::render::render(frame, runtime.state_mut()))
        .expect("render transcript");
}

#[test]
fn zero_distance_drag_does_not_swallow_auto_review_toggle() {
    let mut runtime = runtime();
    runtime.apply_session_transport_event(SessionTransportEvent::PermissionResolved(
        PermissionResolutionEvent {
            call_id: "call-review".into(),
            decision: PermissionDecision::Denied,
            reason: Some("unsafe command".into()),
            tool_name: Some("shell__exec".into()),
            summary: Some("run command".into()),
            origin_label: Some("reviewer".into()),
            approval: Some("deny".into()),
            risk: Some("high".into()),
            reviewer_child_session_id: Some("reviewer-child".into()),
        },
    ));
    runtime
        .state_mut()
        .toggle_tool_output("auto-review:call-review");
    render_runtime_transcript(&mut runtime);

    let state = runtime.state();
    let item_row = state.transcript_render_cache.row_starts()[0] + 1;
    let area = state.last_transcript_area;
    let visible_item_row =
        u16::try_from(item_row.saturating_sub(state.last_transcript_scroll_top as usize))
            .expect("auto-review row fits in terminal coordinates");
    let row = area.y + visible_item_row;
    let col = area.x + 3;
    assert!(row < area.bottom());

    runtime.handle_selection_start(col, row);
    assert!(runtime.state().selection_in_progress);
    runtime.handle_selection_drag(col, row);
    assert!(!runtime.state().selection_dragged);
    runtime.handle_selection_end(col, row, false);

    assert_eq!(
        runtime
            .state()
            .tool_output_overrides
            .get("auto-review:call-review"),
        Some(&false)
    );

    render_runtime_transcript(&mut runtime);
    let text = runtime.state().transcript_render_cache.entries()[0]
        .document
        .lines
        .iter()
        .flat_map(|line| line.spans.iter())
        .map(|span| span.text.as_str())
        .collect::<String>();
    assert!(text.contains(" · expand"));
    assert!(!text.contains("unsafe command"));
}

#[test]
fn model_catalog_update_refreshes_open_picker_and_notifies_once_per_absence() {
    let mut runtime = runtime();
    let mut dialog = DialogState::new(
        DialogKind::ModelPicker,
        "Select model",
        None,
        vec![
            DialogItem::new("gpt-5.5", "GPT-5.5", None),
            DialogItem::new("gpt-4", "GPT-4", None),
        ],
    );
    dialog.query = "gpt".into();
    dialog.selected = 0;
    runtime.state_mut().open_dialog(dialog);

    let removed = ModelCatalogUpdatedEvent {
        models: vec![ModelCatalogEntry {
            id: "gpt-4".into(),
            label: "GPT-4 refreshed".into(),
            provider: "openai".into(),
            context_window_tokens: Some(128_000),
            reasoning: ModelCatalogReasoning {
                effort: None,
                efforts: Vec::new(),
            },
        }],
    };
    runtime.apply_session_transport_event(SessionTransportEvent::ModelCatalogUpdated(removed));

    assert_eq!(runtime.state().model_id, "gpt-5.5");
    assert_eq!(runtime.available_models().len(), 1);
    assert_eq!(runtime.available_models()[0].label, "GPT-4 refreshed");
    assert_eq!(
        runtime.state().toast().map(|toast| toast.message.as_str()),
        Some("Current model is no longer available: gpt-5.5")
    );
    assert_eq!(
        runtime.state().dialog().map(|dialog| dialog.query.as_str()),
        Some("gpt")
    );
    assert_eq!(
        runtime
            .state()
            .dialog()
            .and_then(|dialog| dialog.selected_item())
            .map(|item| item.id.as_str()),
        Some("gpt-4")
    );

    let first_toast = runtime.state().toast().cloned();
    runtime.apply_session_transport_event(SessionTransportEvent::ModelCatalogUpdated(
        ModelCatalogUpdatedEvent {
            models: vec![ModelCatalogEntry {
                id: "gpt-4".into(),
                label: "GPT-4 refreshed".into(),
                provider: "openai".into(),
                context_window_tokens: Some(128_000),
                reasoning: ModelCatalogReasoning {
                    effort: None,
                    efforts: Vec::new(),
                },
            }],
        },
    ));
    assert_eq!(runtime.state().toast().cloned(), first_toast);

    runtime.state_mut().toast = None;
    runtime.apply_session_transport_event(SessionTransportEvent::ModelCatalogUpdated(
        ModelCatalogUpdatedEvent {
            models: vec![
                ModelCatalogEntry {
                    id: "gpt-5.5".into(),
                    label: "GPT-5.5 updated".into(),
                    provider: "openai".into(),
                    context_window_tokens: Some(200_000),
                    reasoning: ModelCatalogReasoning {
                        effort: Some("high".into()),
                        efforts: vec!["none".into(), "high".into()],
                    },
                },
                ModelCatalogEntry {
                    id: "gpt-4".into(),
                    label: "GPT-4 refreshed".into(),
                    provider: "openai".into(),
                    context_window_tokens: Some(128_000),
                    reasoning: ModelCatalogReasoning {
                        effort: None,
                        efforts: Vec::new(),
                    },
                },
            ],
        },
    ));

    assert!(runtime.state().toast().is_none());
    assert_eq!(runtime.state().model_id, "gpt-5.5");
    assert_eq!(
        runtime
            .state()
            .dialog()
            .and_then(|dialog| dialog.selected_item())
            .map(|item| item.id.as_str()),
        Some("gpt-4")
    );
    assert_eq!(
        runtime.available_models()[0].context_window_tokens,
        Some(200_000)
    );
    assert_eq!(
        runtime.available_models()[0].reasoning_effort,
        Some(ModelReasoningEffort::High)
    );

    runtime.apply_session_transport_event(SessionTransportEvent::ModelCatalogUpdated(
        ModelCatalogUpdatedEvent {
            models: vec![ModelCatalogEntry {
                id: "gpt-4".into(),
                label: "GPT-4 refreshed".into(),
                provider: "openai".into(),
                context_window_tokens: Some(128_000),
                reasoning: ModelCatalogReasoning {
                    effort: None,
                    efforts: Vec::new(),
                },
            }],
        },
    ));
    assert!(runtime.state().toast().is_some());
}

#[test]
fn clipboard_paste_prefers_image_only_in_composer_context() {
    assert_eq!(
        choose_clipboard_paste(ClipboardPasteContext::Composer, true, true),
        ClipboardPasteChoice::Image
    );
    assert_eq!(
        choose_clipboard_paste(ClipboardPasteContext::Composer, false, true),
        ClipboardPasteChoice::Image
    );
    assert_eq!(
        choose_clipboard_paste(ClipboardPasteContext::Composer, true, false),
        ClipboardPasteChoice::Text
    );
    assert_eq!(
        choose_clipboard_paste(ClipboardPasteContext::Composer, false, false),
        ClipboardPasteChoice::None
    );
}

#[test]
fn clipboard_paste_dialog_is_text_first() {
    assert_eq!(
        choose_clipboard_paste(ClipboardPasteContext::Dialog, true, true),
        ClipboardPasteChoice::Text
    );
    assert_eq!(
        choose_clipboard_paste(ClipboardPasteContext::Dialog, false, true),
        ClipboardPasteChoice::None
    );
}

#[test]
fn clipboard_paste_pending_question_is_text_first() {
    assert_eq!(
        choose_clipboard_paste(ClipboardPasteContext::Question, true, true),
        ClipboardPasteChoice::Text
    );
    assert_eq!(
        choose_clipboard_paste(ClipboardPasteContext::Question, false, true),
        ClipboardPasteChoice::None
    );
}

#[test]
fn clipboard_paste_permission_is_text_first() {
    assert_eq!(
        choose_clipboard_paste(ClipboardPasteContext::Permission, true, true),
        ClipboardPasteChoice::Text
    );
    assert_eq!(
        choose_clipboard_paste(ClipboardPasteContext::Permission, false, true),
        ClipboardPasteChoice::None
    );
}

fn test_agent() -> Agent<OpenAIConfig> {
    Agent::new(
        Client::with_config(
            OpenAIConfig::new()
                .with_api_base("https://api.openai.com/v1")
                .with_api_key("test"),
        ),
        "gpt-test",
        4,
        4,
    )
}

#[test]
fn mcp_toggle_is_deferred_while_a_turn_is_running() {
    let mut runtime = runtime();
    runtime
        .state_mut()
        .set_mcp_servers(vec![crate::mcp::McpServerCatalogEntry {
            name: "docs".into(),
            enabled: true,
            status: crate::mcp::McpServerStatus::Online { tool_count: 1 },
        }]);
    runtime.show_mcp_dialog().expect("opens picker");
    runtime.session_turn_active = true;

    let command = runtime
        .handle_input_action(InputAction::DialogToggle)
        .expect("toggle is deferred");

    assert_eq!(
        command,
        Some(RuntimeCommand::ToggleMcpServer("docs".into()))
    );
    assert!(!runtime.state().mcp_updating.contains("docs"));
}

#[test]
fn mcp_toggle_is_rejected_while_the_server_is_updating() {
    let mut runtime = runtime();
    runtime
        .state_mut()
        .set_mcp_servers(vec![crate::mcp::McpServerCatalogEntry {
            name: "docs".into(),
            enabled: true,
            status: crate::mcp::McpServerStatus::Online { tool_count: 1 },
        }]);
    runtime.show_mcp_dialog().expect("opens picker");

    let first_command = runtime
        .handle_input_action(InputAction::DialogToggle)
        .expect("toggle starts");
    let second_command = runtime
        .handle_input_action(InputAction::DialogToggle)
        .expect("duplicate toggle is rejected");

    assert_eq!(
        first_command,
        Some(RuntimeCommand::ToggleMcpServer("docs".into()))
    );
    assert_eq!(second_command, None);
    assert!(runtime.state().mcp_updating.contains("docs"));
    assert!(matches!(
        runtime.state().dialog(),
        Some(dialog) if dialog.kind == DialogKind::McpPicker
    ));
}

#[test]
fn submit_question_with_dropped_receiver_clears_ui_without_error() {
    let mut runtime = runtime();
    let (tx, rx) = oneshot::channel();
    drop(rx);
    runtime.apply_session_transport_event(SessionTransportEvent::QuestionRequested {
        request: sample_question_request(false),
        handle: RunnerQuestionRequest::new(tx),
    });

    assert!(
        runtime
            .handle_input_action(InputAction::QuestionPickOption(1))
            .is_ok()
    );
    assert!(runtime.state().pending_question.is_none());
    assert!(runtime.pending_question_handle.is_none());
}

#[tokio::test]
async fn cancel_question_delivers_cancellation() {
    let mut runtime = runtime();
    let (tx, rx) = oneshot::channel();
    runtime.apply_session_transport_event(SessionTransportEvent::QuestionRequested {
        request: sample_question_request(false),
        handle: RunnerQuestionRequest::new(tx),
    });

    runtime
        .handle_input_action(InputAction::QuestionCancel)
        .expect("cancel succeeds");

    assert_eq!(
        rx.await.expect("cancellation received"),
        Err("question dismissed by user".into())
    );
    assert!(runtime.state().pending_question.is_none());
}

#[test]
fn history_tree_selection_restores_user_content_and_targets_the_parent() {
    let sessions_dir = std::env::temp_dir().join(format!(
        "letcode-history-selection-test-{}",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("time ok")
            .as_nanos()
    ));
    std::fs::create_dir_all(&sessions_dir).expect("sessions directory");
    let mut recorder = TranscriptRecorder::create(&sessions_dir).expect("recorder");
    let session_id = recorder.session_id().to_string();
    recorder.record_user_message("first").expect("first user");
    recorder
        .record_assistant_message("first answer")
        .expect("first answer");
    recorder.record_user_message("second").expect("second user");
    drop(recorder);
    let transcript = Arc::new(StdMutex::new(
        TranscriptRecorder::open_existing(&sessions_dir, &session_id).expect("open recorder"),
    ));
    let mut state = TuiState::new("gpt-5.5", "GPT-5.5", "default");
    let (initial_session_id, _) = initial_session_metadata(&transcript).expect("load metadata");
    state.session_id = Some(initial_session_id);
    assert_eq!(state.session_id.as_deref(), Some(session_id.as_str()));
    let (_tx, rx) = mpsc::unbounded_channel();
    let mut runtime = TuiRuntime::new(
        state,
        rx,
        Vec::new(),
        Vec::new(),
        sessions_dir,
        std::env::temp_dir(),
    );

    let path = runtime.sessions_dir.join(format!("{session_id}.jsonl"));
    let records = read_records(&path).expect("records");
    runtime.apply_session_transport_event(SessionTransportEvent::SessionHistoryLoaded {
        entries: transcript_projection::project_session_history_tree(&records),
    });
    let dialog = runtime.state().dialog().expect("history dialog");
    assert_eq!(dialog.kind, DialogKind::HistoryTree);
    assert_eq!(dialog.selected_item().expect("selected").id, "entry-3");

    let command = runtime
        .handle_dialog_accept()
        .expect("accept user selection");
    assert_eq!(
        command,
        Some(RuntimeCommand::NavigateHistory {
            target_entry_id: "entry-2".into(),
        })
    );
    assert_eq!(runtime.state().input_buffer, "second");

    runtime.apply_session_transport_event(SessionTransportEvent::SessionHistoryLoaded {
        entries: transcript_projection::project_session_history_tree(&records),
    });
    runtime
        .state_mut()
        .dialog_mut()
        .expect("history dialog")
        .selected = 1;
    let command = runtime
        .handle_dialog_accept()
        .expect("accept assistant selection");
    assert_eq!(
        command,
        Some(RuntimeCommand::NavigateHistory {
            target_entry_id: "entry-2".into(),
        })
    );
}

#[test]
fn history_tree_acceptance_rechecks_pending_running_queued_question_and_inflight_state() {
    for state in ["running", "permission", "queued", "question", "inflight"] {
        let mut runtime = runtime();
        runtime.state_mut().open_dialog(DialogState::new(
            DialogKind::HistoryTree,
            "Session history",
            None,
            vec![DialogItem::new("entry-1", "You: first", None)],
        ));
        match state {
            "running" => runtime.session_turn_active = true,
            "permission" => {
                let (reply_tx, _reply_rx) = oneshot::channel();
                runtime.apply_session_transport_event(SessionTransportEvent::PermissionRequested {
                    event: PermissionRequestEvent::new("call-1", "shell__exec", "Run command"),
                    handle: RunnerPermissionRequest::new(reply_tx),
                });
            }
            "queued" => runtime
                .queued_prompts
                .push_back(UserMessageSubmission::from("queued")),
            "question" => {
                let (reply_tx, _reply_rx) = oneshot::channel();
                runtime.apply_session_transport_event(SessionTransportEvent::QuestionRequested {
                    request: sample_question_request(false),
                    handle: RunnerQuestionRequest::new(reply_tx),
                });
            }
            "inflight" => runtime
                .queued_prompt_lifecycle
                .dispatch(UserMessageSubmission::from("inflight")),
            _ => unreachable!(),
        }

        assert_eq!(
            runtime
                .handle_dialog_accept()
                .expect("accept history dialog"),
            None,
            "{state} state must reject navigation"
        );
        assert!(!runtime.state().dialog_is_open());
    }
}

#[test]
fn navigation_commands_are_blocked_for_running_pending_and_queued_turns() {
    for state in ["running", "pending", "queued"] {
        let mut runtime = runtime();
        match state {
            "running" => runtime.session_turn_active = true,
            "pending" => {
                let (reply_tx, _reply_rx) = oneshot::channel();
                runtime.apply_session_transport_event(SessionTransportEvent::PermissionRequested {
                    event: PermissionRequestEvent::new("call-1", "shell__exec", "Run command"),
                    handle: RunnerPermissionRequest::new(reply_tx),
                });
            }
            "queued" => runtime
                .queued_prompt_lifecycle
                .dispatch(UserMessageSubmission::from("queued")),
            _ => unreachable!(),
        }
        runtime.state_mut().set_input("/undo");
        assert_eq!(
            runtime
                .handle_input_action(InputAction::Submit)
                .expect("submit"),
            None,
            "{state} turn must block undo"
        );
        assert_eq!(runtime.state().input_buffer, "/undo");
    }
}

#[test]
fn matching_child_stream_event_updates_child_view_without_touching_parent() {
    let mut runtime = runtime();
    runtime.state_mut().replace_child_timeline_from_records(
        &[],
        "parent-session",
        "child-session",
        "explorer",
        0,
        1,
        1,
    );

    runtime.apply_session_transport_event(SessionTransportEvent::ChildSessionEvent {
        child_session_id: "child-session".into(),
        agent_name: None,
        parent_tool_call_id: None,
        event: SessionEvent::AssistantDelta(crate::tui::events::AssistantDeltaEvent::new("hello")),
    });

    assert!(runtime.state().timeline.items().is_empty());
    assert!(matches!(
        runtime.state().active_timeline().items().last(),
        Some(crate::tui::TimelineItem::Assistant(message)) if message.text == "hello"
    ));
}

#[test]
fn non_matching_child_stream_event_does_not_mutate_current_view() {
    let mut runtime = runtime();
    runtime.state_mut().replace_child_timeline_from_records(
        &[],
        "parent-session",
        "child-session",
        "explorer",
        0,
        1,
        1,
    );

    runtime.apply_session_transport_event(SessionTransportEvent::ChildSessionEvent {
        child_session_id: "other-child".into(),
        agent_name: None,
        parent_tool_call_id: None,
        event: SessionEvent::AssistantDelta(crate::tui::events::AssistantDeltaEvent::new("hello")),
    });

    assert!(runtime.state().timeline.items().is_empty());
    assert!(runtime.state().active_timeline().items().is_empty());
}

#[test]
fn child_interrupted_event_updates_child_view_without_touching_parent() {
    let mut runtime = runtime();
    runtime.state_mut().replace_child_timeline_from_records(
        &[],
        "parent-session",
        "child-session",
        "explorer",
        0,
        1,
        1,
    );

    runtime.apply_session_transport_event(SessionTransportEvent::ChildSessionEvent {
        child_session_id: "child-session".into(),
        agent_name: None,
        parent_tool_call_id: None,
        event: SessionEvent::Interrupted,
    });

    assert!(runtime.state().timeline.items().is_empty());
    assert_eq!(
        runtime.state().toast().map(|toast| toast.message.as_str()),
        Some("Interrupted by user")
    );
}

#[test]
fn running_turn_opens_session_setting_dialogs() {
    for command_text in ["/model", "/agents", "/permission", "/reasoning"] {
        let mut runtime = runtime();
        runtime.session_turn_active = true;
        runtime.state_mut().phase = AppPhase::Running;
        runtime.state_mut().set_input(command_text);

        let command = runtime
            .handle_input_action(InputAction::Submit)
            .expect("dialog command succeeds");

        assert_eq!(command, None, "{command_text}");
        assert!(runtime.state().dialog_is_open(), "{command_text}");
        assert_eq!(runtime.state().input_buffer, "", "{command_text}");
    }
}

#[test]
fn thoughts_command_opens_picker_and_persists_selected_mode() {
    let mut runtime = runtime();
    runtime.state_mut().set_input("/thoughts");

    assert_eq!(
        runtime
            .handle_input_action(InputAction::Submit)
            .expect("thoughts picker opens"),
        None
    );
    assert!(matches!(
        runtime.state().dialog(),
        Some(dialog)
            if dialog.kind == DialogKind::ThoughtsPicker
                && dialog.selected_item().map(|item| item.id.as_str()) == Some("full")
    ));

    runtime
        .handle_input_action(InputAction::DialogPrev)
        .expect("select titles");
    runtime
        .handle_input_action(InputAction::DialogAccept)
        .expect("accept thoughts display");

    assert_eq!(
        runtime.state().thoughts_display,
        ThoughtsDisplayMode::Titles
    );
    assert!(!runtime.state().dialog_is_open());
    assert_eq!(
        TuiPreferences::load_from_dir(&runtime.preferences_dir).thoughts_display,
        ThoughtsDisplayMode::Titles
    );
}

#[test]
fn thoughts_command_is_available_during_active_turn_and_child_view() {
    let mut runtime = runtime();
    runtime.session_turn_active = true;
    runtime.state_mut().phase = AppPhase::Running;
    runtime.state_mut().set_input("/thoughts compact");

    assert_eq!(
        runtime
            .handle_input_action(InputAction::Submit)
            .expect("active turn accepts local display command"),
        None
    );
    assert_eq!(
        runtime.state().thoughts_display,
        ThoughtsDisplayMode::Compact
    );

    runtime.state_mut().replace_child_timeline_from_records(
        &[],
        "parent-session",
        "child-session",
        "explorer",
        0,
        1,
        1,
    );
    runtime.state_mut().set_input("/thoughts full");
    assert_eq!(
        runtime
            .handle_input_action(InputAction::Submit)
            .expect("child view accepts local display command"),
        None
    );
    assert_eq!(runtime.state().thoughts_display, ThoughtsDisplayMode::Full);
}

#[test]
fn thoughts_numeric_alias_persists_and_loads_for_a_new_runtime() {
    let mut runtime = runtime();
    runtime.state_mut().set_input("/thoughts 1");
    runtime
        .handle_input_action(InputAction::Submit)
        .expect("numeric thoughts command succeeds");
    assert_eq!(
        runtime.state().thoughts_display,
        ThoughtsDisplayMode::Compact
    );

    let loaded = TuiPreferences::load_from_dir(&runtime.preferences_dir);
    let (_tx, rx) = mpsc::unbounded_channel();
    let mut state = TuiState::new("gpt-5.5", "GPT-5.5", "default");
    state.set_thoughts_display(loaded.thoughts_display);
    let restarted = TuiRuntime::new(
        state,
        rx,
        Vec::new(),
        Vec::new(),
        std::env::temp_dir(),
        runtime.preferences_dir.clone(),
    );

    assert_eq!(
        restarted.state().thoughts_display,
        ThoughtsDisplayMode::Compact
    );
}

#[test]
fn reasoning_shortcut_waits_for_backend_confirmation() {
    let mut runtime = runtime();
    let previous = runtime.state().reasoning_effort_label.clone();

    assert!(matches!(
        runtime
            .handle_input_action(InputAction::CycleReasoningEffort)
            .expect("reasoning shortcut succeeds"),
        Some(RuntimeCommand::SetReasoningEffort(_))
    ));
    assert_eq!(runtime.state().reasoning_effort_label, previous);
}

#[tokio::test]
async fn running_turn_projects_session_setting_until_backend_confirmation() {
    let mut runtime = runtime();
    runtime.session_turn_active = true;
    runtime.state_mut().phase = AppPhase::Running;
    runtime.state_mut().set_input("/permission safe");

    let command = runtime
        .handle_input_action(InputAction::Submit)
        .expect("permission setting is accepted")
        .expect("permission setting is dispatched");
    assert_eq!(
        command,
        RuntimeCommand::SetPermissionMode(crate::permission::PermissionMode::Safe)
    );

    let (mut engine, ingress, _egress) = SessionEngine::new();
    command_dispatch::dispatch_command(&mut runtime, command, &ingress, true);
    assert!(matches!(
        engine.recv_control().await,
        Some(SessionEngineControl::Command(
            SessionEngineCommand::SetPermissionMode(crate::permission::PermissionMode::Safe)
        ))
    ));
    assert_eq!(runtime.state().permission_mode_label, "default");
    assert_eq!(
        runtime
            .state()
            .pending_composer_settings
            .permission_mode
            .as_deref(),
        Some("safe")
    );

    runtime.apply_session_transport_event(SessionTransportEvent::PermissionModeChanged {
        mode: "safe".into(),
    });
    assert_eq!(runtime.state().permission_mode_label, "safe");
    assert_eq!(
        runtime.state().pending_composer_settings.permission_mode,
        None
    );
}

#[test]
fn setting_failure_clears_only_the_matching_pending_projection() {
    let mut runtime = runtime();
    runtime.state_mut().set_pending_model("p/new", "New");
    runtime.state_mut().set_pending_reasoning_effort("high");
    runtime.state_mut().set_pending_permission_mode("safe");

    runtime.apply_session_transport_event(SessionTransportEvent::SettingChangeFailed {
        command: crate::session::SessionCommand::SetPermissionMode(
            crate::permission::PermissionMode::Safe,
        ),
    });

    assert_eq!(
        runtime.state().pending_composer_settings.permission_mode,
        None
    );
    assert!(runtime.state().pending_composer_settings.model.is_some());
    assert_eq!(
        runtime
            .state()
            .pending_composer_settings
            .reasoning_effort
            .as_deref(),
        Some("high")
    );
}

#[test]
fn stale_setting_result_does_not_clear_a_newer_pending_projection() {
    let mut runtime = runtime();
    runtime.state_mut().set_pending_permission_mode("auto");

    runtime.apply_session_transport_event(SessionTransportEvent::PermissionModeChanged {
        mode: "safe".into(),
    });
    assert_eq!(
        runtime
            .state()
            .pending_composer_settings
            .permission_mode
            .as_deref(),
        Some("auto")
    );

    runtime.apply_session_transport_event(SessionTransportEvent::SettingChangeFailed {
        command: crate::session::SessionCommand::SetPermissionMode(
            crate::permission::PermissionMode::Safe,
        ),
    });
    assert_eq!(
        runtime
            .state()
            .pending_composer_settings
            .permission_mode
            .as_deref(),
        Some("auto")
    );
}

#[test]
fn parent_view_refresh_preserves_pending_setting_projection() {
    let mut runtime = runtime();
    runtime.state_mut().set_pending_reasoning_effort("high");

    runtime.apply_session_transport_event(SessionTransportEvent::ParentSessionViewed {
        session_id: "parent-session".into(),
        branch_id: ROOT_CONTEXT_BRANCH_ID.into(),
        records: vec![],
        model_id: None,
        token_usage: None,
        runtime_context: event_context("parent-session", 1),
    });

    assert_eq!(
        runtime
            .state()
            .pending_composer_settings
            .reasoning_effort
            .as_deref(),
        Some("high")
    );
}

#[test]
fn parent_view_refresh_without_usage_preserves_known_parent_usage() {
    let mut runtime = runtime();
    runtime
        .state_mut()
        .set_token_usage(TokenUsageEvent::with_breakdown(700, 1_000, 600, 100, 0).into());
    runtime.apply_session_transport_event(SessionTransportEvent::ChildSessionViewed {
        parent_session_id: "parent-session".into(),
        child_session_id: "child-session".into(),
        agent_name: "explorer".into(),
        index: 0,
        total: 1,
        pool_ordinal: 1,
        records: vec![],
        runtime_context: event_context("child-session", 1),
    });

    runtime.apply_session_transport_event(SessionTransportEvent::ParentSessionViewed {
        session_id: "parent-session".into(),
        branch_id: ROOT_CONTEXT_BRANCH_ID.into(),
        records: vec![],
        model_id: None,
        token_usage: None,
        runtime_context: event_context("parent-session", 1),
    });

    assert_eq!(
        runtime
            .state()
            .active_model_token_usage()
            .map(|usage| (usage.used_tokens, usage.context_window_tokens)),
        Some((700, 1_000))
    );
}

#[test]
fn parent_view_refresh_preserves_confirmed_reasoning_effort() {
    let mut runtime = runtime();
    runtime
        .state_mut()
        .set_reasoning_effort_label(Some("high".into()));

    runtime.apply_session_transport_event(SessionTransportEvent::ParentSessionViewed {
        session_id: "parent-session".into(),
        branch_id: ROOT_CONTEXT_BRANCH_ID.into(),
        records: vec![],
        model_id: Some("gpt-5.5".into()),
        token_usage: None,
        runtime_context: event_context("parent-session", 1),
    });

    assert_eq!(
        runtime.state().reasoning_effort_label.as_deref(),
        Some("high")
    );
}

#[test]
fn running_turn_blocks_exit_and_quit_commands() {
    for command_text in ["exit", "quit", "/exit", "/quit"] {
        let mut runtime = runtime();
        runtime.state_mut().phase = AppPhase::Running;
        runtime.state_mut().set_input(command_text);

        let command = runtime
            .handle_input_action(InputAction::Submit)
            .expect("command is blocked while running");

        assert_eq!(command, None, "{command_text}");
        assert!(!runtime.state().quit_requested, "{command_text}");
        assert_eq!(runtime.queued_prompts.len(), 0, "{command_text}");
        assert!(runtime.submitted_prompts().is_empty(), "{command_text}");
        assert_eq!(runtime.state().input_buffer, command_text, "{command_text}");
    }
}

#[test]
fn double_escape_confirms_running_turn_interrupt() {
    let mut runtime = runtime();
    runtime.state.phase = AppPhase::Running;

    let first = runtime
        .handle_input_action(InputAction::Interrupt)
        .expect("first interrupt hint succeeds");
    assert_eq!(first, None);

    let second = runtime
        .handle_input_action(InputAction::Interrupt)
        .expect("second interrupt returns command");
    assert_eq!(second, Some(RuntimeCommand::Interrupt));
}

#[test]
fn interrupt_confirmation_survives_tick() {
    let mut runtime = runtime();
    runtime.state.phase = AppPhase::Running;

    runtime
        .handle_input_action(InputAction::Interrupt)
        .expect("first interrupt hint succeeds");
    runtime
        .handle_input_action(InputAction::Tick)
        .expect("tick succeeds");

    let second = runtime
        .handle_input_action(InputAction::Interrupt)
        .expect("second interrupt still confirms");
    assert_eq!(second, Some(RuntimeCommand::Interrupt));
}

#[test]
fn interrupt_uses_runner_active_after_non_terminal_error() {
    let mut runtime = runtime();
    runtime.session_turn_active = true;
    runtime.state.phase = AppPhase::Running;
    runtime.state_mut().set_input("follow up");
    runtime
        .handle_input_action(InputAction::Submit)
        .expect("queue succeeds");

    runtime.apply_session_transport_event(SessionTransportEvent::Error(ErrorEvent::new(
        "failed to view child transcript",
    )));
    assert_eq!(runtime.state().phase, AppPhase::Error);

    let first = runtime
        .handle_input_action(InputAction::Interrupt)
        .expect("first interrupt hint succeeds");
    assert_eq!(first, None);

    let second = runtime
        .handle_input_action(InputAction::Interrupt)
        .expect("second interrupt returns command");
    assert_eq!(second, Some(RuntimeCommand::Interrupt));
}

#[test]
fn interrupted_session_transport_event_returns_to_prompt_ready_state() {
    let mut runtime = runtime();
    runtime.state.phase = AppPhase::Running;

    runtime.apply_session_transport_event(SessionTransportEvent::Interrupted);

    assert_eq!(runtime.state().phase, AppPhase::Completed);

    assert!(runtime.state().pending_permission.is_none());
}

#[tokio::test]
async fn interrupted_cancels_parent_question_and_clears_local_state() {
    let mut runtime = runtime();
    let (tx, rx) = oneshot::channel();
    runtime.apply_session_transport_event(SessionTransportEvent::QuestionRequested {
        request: sample_question_request(false),
        handle: RunnerQuestionRequest::new(tx),
    });

    runtime.apply_session_transport_event(SessionTransportEvent::Interrupted);

    assert!(runtime.state().pending_question.is_none());
    assert!(runtime.pending_question_handle.is_none());
    assert!(runtime.pending_question_child_session_id.is_none());
    assert_eq!(
        rx.await.expect("cancellation received"),
        Err("question cancelled because the turn was interrupted".into())
    );
}

#[test]
fn interrupted_clears_question_when_receiver_was_dropped() {
    let mut runtime = runtime();
    let (tx, rx) = oneshot::channel();
    drop(rx);
    runtime.apply_session_transport_event(SessionTransportEvent::QuestionRequested {
        request: sample_question_request(false),
        handle: RunnerQuestionRequest::new(tx),
    });

    runtime.apply_session_transport_event(SessionTransportEvent::Interrupted);

    assert!(runtime.state().pending_question.is_none());
    assert!(runtime.pending_question_handle.is_none());
    assert!(runtime.pending_question_child_session_id.is_none());
}

#[tokio::test]
async fn interrupted_cancels_child_question_and_clears_local_state() {
    let mut runtime = runtime();
    let (tx, rx) = oneshot::channel();
    runtime.apply_session_transport_event(SessionTransportEvent::ChildQuestionRequested {
        child_session_id: "child-1".into(),
        request: sample_question_request(false),
        handle: RunnerQuestionRequest::new(tx),
    });

    runtime.apply_session_transport_event(SessionTransportEvent::Interrupted);

    assert!(runtime.state().pending_question.is_none());
    assert!(runtime.pending_question_handle.is_none());
    assert!(runtime.pending_question_child_session_id.is_none());
    assert_eq!(
        rx.await.expect("cancellation received"),
        Err("question cancelled because the turn was interrupted".into())
    );
}

#[tokio::test]
async fn child_terminal_event_cancels_matching_question_and_clears_local_state() {
    for event in [
        SessionEvent::Done,
        SessionEvent::Interrupted,
        SessionEvent::Error(ErrorEvent::new("child stopped")),
    ] {
        let mut runtime = runtime();
        let (tx, rx) = oneshot::channel();
        runtime.apply_session_transport_event(SessionTransportEvent::ChildQuestionRequested {
            child_session_id: "child-1".into(),
            request: sample_question_request(false),
            handle: RunnerQuestionRequest::new(tx),
        });

        runtime.apply_session_transport_event(SessionTransportEvent::ChildSessionEvent {
            child_session_id: "child-1".into(),
            agent_name: Some("explorer".into()),
            parent_tool_call_id: Some("parent-call".into()),
            event,
        });

        assert!(runtime.state().pending_question.is_none());
        assert!(runtime.pending_question_handle.is_none());
        assert!(runtime.pending_question_child_session_id.is_none());
        assert_eq!(
            rx.await.expect("cancellation received"),
            Err("question cancelled because the child session stopped".into())
        );
    }
}

#[test]
fn error_preserves_question_while_done_clears_parent_question() {
    let mut runtime = runtime();
    let (tx, mut rx) = oneshot::channel();
    runtime.apply_session_transport_event(SessionTransportEvent::QuestionRequested {
        request: sample_question_request(false),
        handle: RunnerQuestionRequest::new(tx),
    });

    runtime.apply_session_transport_event(SessionTransportEvent::Error(ErrorEvent::new(
        "turn failed",
    )));

    assert!(runtime.state().pending_question.is_some());
    assert!(matches!(
        rx.try_recv(),
        Err(oneshot::error::TryRecvError::Empty)
    ));

    runtime.apply_session_transport_event(SessionTransportEvent::Done);

    assert!(runtime.state().pending_question.is_none());
    assert!(runtime.pending_question_handle.is_none());
    assert!(matches!(
        rx.try_recv(),
        Ok(Err(reason)) if reason == "question cancelled because the turn ended"
    ));
}

#[test]
fn interrupt_rehydrates_agent_from_transcript() {
    let base_dir = std::env::temp_dir().join(format!(
        "letcode-tui-runtime-interrupt-rehydrate-test-{}",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("time ok")
            .as_nanos()
    ));
    let recorder = TranscriptRecorder::create(&base_dir).expect("create recorder");
    let transcript = Arc::new(StdMutex::new(recorder));

    {
        let mut recorder = transcript.lock().expect("lock recorder");
        recorder
            .record_user_message("unfinished")
            .expect("record user message");
        recorder
            .record_turn_interrupted(Some(1))
            .expect("record turn interrupted");
    }

    let mut agent = test_agent();
    agent
        .restore_session_history(vec![HistoryItem::user("stale dangling")], Vec::new(), 0)
        .expect("seed stale history");

    rehydrate_agent_from_transcript(&mut agent, &transcript).expect("rehydrate agent");

    assert!(matches!(
        agent.history_for_test(),
        [HistoryItem::UserMessage { content }, HistoryItem::AssistantText { text: assistant_text }]
            if content.text == "unfinished" && assistant_text.is_empty()
    ));
}

#[test]
fn resumed_session_usage_uses_target_agent_and_drops_response_accounting() {
    let mut old_agent = test_agent();
    old_agent
        .restore_session_history(
            vec![HistoryItem::user("old session ".repeat(2_000))],
            Vec::new(),
            99,
        )
        .expect("restore old session");
    let old_usage = old_agent.session_token_usage().expect("old usage");

    let mut target_agent = test_agent();
    target_agent.set_model("target-model");
    target_agent
        .restore_session_history(vec![HistoryItem::user("target session")], Vec::new(), 2)
        .expect("restore target session");
    let target_frames = target_agent.protocol_frames_for_test().to_vec();
    let target_snapshot = target_agent.runtime_snapshot_for_test().clone();
    let expected_usage =
        restored_session_token_usage(&target_agent, target_agent.model(), &target_snapshot)
            .expect("target usage");
    let prepared_usage =
        restored_session_token_usage(&old_agent, target_agent.model(), &target_snapshot)
            .expect("prepare target usage");

    old_agent
        .restore_new_session_runtime_snapshot(target_frames, target_snapshot.clone(), 2)
        .expect("install target session");
    old_agent.set_model("target-model");
    let resumed_usage =
        restored_session_token_usage(&old_agent, old_agent.model(), &target_snapshot)
            .expect("resumed usage");

    assert_eq!(
        old_agent.runtime_snapshot_for_test().current_turn_id,
        Some(2)
    );
    assert_ne!(old_usage.used_tokens, expected_usage.used_tokens);
    assert_eq!(prepared_usage, expected_usage);
    assert_eq!(resumed_usage, expected_usage);
    assert_eq!(resumed_usage.output_tokens, 0);
    assert_eq!(resumed_usage.cached_tokens, 0);
    assert_eq!(resumed_usage.cache_report, None);
}

#[test]
fn failed_candidate_usage_preserves_agent_and_recorder() {
    let sessions_dir = std::env::temp_dir().join(format!(
        "letcode-tui-runtime-candidate-usage-test-{}",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("time ok")
            .as_nanos()
    ));
    let recorder = TranscriptRecorder::create(&sessions_dir).expect("create recorder");
    let recorder_id = recorder.session_id().to_string();
    let recorder_path = recorder.path().to_path_buf();
    let mut agent = test_agent();
    agent
        .restore_session_history(vec![HistoryItem::user("old session")], Vec::new(), 7)
        .expect("restore old session");
    let model = agent.model().to_string();
    let history = agent.history_for_test().to_vec();
    let runtime_snapshot = agent.runtime_snapshot_for_test().clone();
    let invalid_metadata = crate::request_builder::ModelRequestMetadata {
        effective_input_limit_tokens: Some(0),
        ..Default::default()
    };
    agent.set_model_catalog(std::collections::HashMap::from([(
        String::from("invalid-model"),
        invalid_metadata,
    )]));
    let target_snapshot = test_agent().runtime_snapshot_for_test().clone();

    assert!(restored_session_token_usage(&agent, "invalid-model", &target_snapshot).is_err());

    assert_eq!(agent.model(), model);
    assert_eq!(agent.history_for_test(), history.as_slice());
    assert_eq!(agent.runtime_snapshot_for_test(), &runtime_snapshot);
    assert_eq!(recorder.session_id(), recorder_id);
    assert_eq!(recorder.path(), recorder_path);
}

#[test]
fn permission_prompt_requires_double_esc_to_interrupt() {
    let mut runtime = runtime();
    runtime.session_turn_active = true;
    runtime
        .state_mut()
        .apply_event(SessionEvent::PermissionRequested(
            crate::tui::events::PermissionRequestEvent::new("call-1", "shell__exec", "run ls"),
        ));

    let first = runtime
        .handle_input_action(InputAction::Interrupt)
        .expect("first esc succeeds");
    assert_eq!(first, None);

    assert!(runtime.state().pending_permission.is_some());

    let second = runtime
        .handle_input_action(InputAction::Interrupt)
        .expect("second esc succeeds");
    assert_eq!(second, Some(RuntimeCommand::Interrupt));
}

#[test]
fn slash_subagent_interrupt_terminalizes_parent_runtime_from_parent_view() {
    let (tx, rx) = mpsc::unbounded_channel();
    let mut runtime = TuiRuntime::new(
        TuiState::default(),
        rx,
        vec![AvailableModel::new("gpt-5.5", "GPT-5.5")],
        Vec::new(),
        std::env::temp_dir(),
        std::env::temp_dir(),
    );
    runtime.session_turn_active = true;
    runtime.state_mut().phase = AppPhase::Running;

    send_subagent_interrupted(&tx, Some("child-session".into()));
    runtime.try_drain_session_events();

    assert!(!runtime.session_turn_active);
    assert_eq!(runtime.state().phase, AppPhase::Completed);
}

#[test]
fn slash_subagent_interrupt_terminalizes_parent_runtime_from_child_view() {
    let (tx, rx) = mpsc::unbounded_channel();
    let mut runtime = TuiRuntime::new(
        TuiState::default(),
        rx,
        vec![AvailableModel::new("gpt-5.5", "GPT-5.5")],
        Vec::new(),
        std::env::temp_dir(),
        std::env::temp_dir(),
    );
    runtime.session_turn_active = true;
    runtime.state_mut().phase = AppPhase::Running;
    runtime.state_mut().replace_child_timeline_from_records(
        &[],
        "parent-session",
        "child-session",
        "explorer",
        0,
        1,
        1,
    );

    send_subagent_interrupted(&tx, Some("child-session".into()));
    runtime.try_drain_session_events();

    assert!(!runtime.session_turn_active);
    assert_eq!(runtime.state().phase, AppPhase::Completed);
    assert_eq!(
        runtime.state().toast().map(|toast| toast.message.as_str()),
        Some("Interrupted by user")
    );
}

#[test]
fn parent_interrupt_while_viewing_child_closes_child_active_tools() {
    let (tx, rx) = mpsc::unbounded_channel();
    let mut runtime = TuiRuntime::new(
        TuiState::default(),
        rx,
        vec![AvailableModel::new("gpt-5.5", "GPT-5.5")],
        Vec::new(),
        std::env::temp_dir(),
        std::env::temp_dir(),
    );
    runtime.session_turn_active = true;
    runtime.state_mut().phase = AppPhase::Running;
    runtime.state_mut().replace_child_timeline_from_records(
        &[],
        "parent-session",
        "child-session",
        "explorer",
        0,
        1,
        1,
    );
    runtime.state_mut().apply_child_session_event(
        "child-session",
        SessionEvent::ToolPending(crate::tui::events::ToolPendingEvent::new(
            "child-call",
            "fs__write",
        )),
    );

    send_subagent_interrupted(&tx, Some("child-session".into()));
    runtime.try_drain_session_events();

    assert!(!runtime.session_turn_active);
    assert!(matches!(
        runtime.state().active_timeline().items().iter().find_map(|item| match item {
            crate::tui::TimelineItem::Tool(tool) => Some(tool),
            _ => None,
        }),
        Some(tool) if tool.status == crate::tui::timeline::ToolExecutionStatus::Cancelled
    ));
}

#[test]
fn child_transcript_view_blocks_parent_mutating_submit_paths() {
    for input in [
        "ask the parent agent",
        "@explorer inspect src/agent.rs",
        "@fixer wire agent__fixer tool",
        "@oracle review src/main.rs",
        "/new",
        "/resume abc123",
        "/model gpt-5.5-mini",
        "/permission safe",
    ] {
        let mut runtime = runtime();
        runtime.state_mut().replace_child_timeline_from_records(
            &[],
            "parent-session",
            "child-session",
            "explorer",
            0,
            1,
            1,
        );
        runtime.state_mut().set_input(input);

        let command = runtime
            .handle_input_action(InputAction::Submit)
            .expect("submit succeeds");

        assert_eq!(command, None, "{input}");
        assert!(runtime.submitted_prompts().is_empty(), "{input}");
    }
}

#[test]
fn child_view_navigation_and_read_only_actions_do_not_show_toasts() {
    let mut runtime = runtime();
    runtime.state_mut().replace_child_timeline_from_records(
        &[],
        "parent-session",
        "child-session",
        "explorer",
        0,
        1,
        1,
    );

    assert_eq!(
        runtime
            .handle_input_action(InputAction::CycleReasoningEffort)
            .expect("read-only reasoning action"),
        None
    );
    assert!(runtime.state().toast().is_none());

    runtime
        .handle_input_action(InputAction::ChildPrefix)
        .expect("child navigation prefix");
    assert!(runtime.state().toast().is_none());
    for _ in 0..CHILD_NAVIGATION_PREFIX_TIMEOUT_TICKS {
        runtime
            .handle_input_action(InputAction::Tick)
            .expect("prefix timeout tick");
    }
    assert!(!runtime.state().child_navigation_prefix);
    assert!(runtime.state().toast().is_none());

    runtime.state_mut().set_input("/model gpt-5.5");
    assert_eq!(
        runtime
            .handle_input_action(InputAction::Submit)
            .expect("read-only command"),
        None
    );
    assert!(runtime.state().toast().is_none());

    assert_eq!(
        runtime
            .handle_input_action(InputAction::ChildParent)
            .expect("return to parent"),
        Some(RuntimeCommand::ViewParent)
    );
    assert!(runtime.state().toast().is_none());
    assert!(!runtime.state().transcript_view.is_child());
}

#[test]
fn child_view_transport_navigation_does_not_show_toasts() {
    let mut runtime = runtime();
    runtime.apply_session_transport_event(SessionTransportEvent::ChildSessionViewed {
        parent_session_id: "parent-session".into(),
        child_session_id: "child-session".into(),
        agent_name: "explorer".into(),
        index: 0,
        total: 1,
        pool_ordinal: 1,
        records: vec![],
        runtime_context: event_context("child-session", 1),
    });
    assert!(runtime.state().toast().is_none());

    runtime.apply_session_transport_event(SessionTransportEvent::ParentSessionViewed {
        session_id: "parent-session".into(),
        branch_id: ROOT_CONTEXT_BRANCH_ID.into(),
        records: vec![],
        model_id: None,
        token_usage: None,
        runtime_context: event_context("parent-session", 1),
    });
    assert!(runtime.state().toast().is_none());
}

#[test]
fn running_turn_queues_plain_prompts() {
    let mut runtime = runtime();
    runtime
        .state_mut()
        .show_toast("stale notice", ToastKind::Info);
    runtime.state_mut().phase = AppPhase::Running;
    runtime.state_mut().set_input("follow up");

    let command = runtime
        .handle_input_action(InputAction::Submit)
        .expect("submit succeeds");

    assert_eq!(command, None);
    assert!(runtime.state().input_buffer.is_empty());
    assert_eq!(runtime.state().phase, AppPhase::Running);
    assert_eq!(runtime.submitted_prompts(), &["follow up".to_string()]);
    assert_eq!(runtime.queued_prompts.len(), 1);
    assert!(runtime
            .state()
            .timeline
            .items()
            .iter()
            .any(|item| matches!(item, TimelineItem::User(message) if message.text == "follow up" && message.queued)));
    assert!(runtime.state().toast().is_none());
}

#[test]
fn running_turn_preserves_selected_skills_in_queued_prompt() {
    let mut runtime = runtime();
    runtime.state_mut().phase = AppPhase::Running;
    runtime.state_mut().set_input("follow up");
    assert!(runtime.state_mut().add_composer_skill("rust-audit".into()));

    let command = runtime
        .handle_input_action(InputAction::Submit)
        .expect("submit succeeds");

    assert_eq!(command, None);
    assert_eq!(
        runtime.queued_prompts[0].content.selected_skills,
        vec!["rust-audit"]
    );
    assert!(runtime.state().composer_tokens.is_empty());
}

#[test]
fn running_turn_rejects_delegate_commands_without_queueing() {
    let mut runtime = runtime();
    runtime.state_mut().phase = AppPhase::Running;
    runtime.state_mut().set_input("@fixer fix failing test");

    let command = runtime
        .handle_input_action(InputAction::Submit)
        .expect("submit succeeds");

    assert_eq!(command, None);
    assert_eq!(runtime.queued_prompts.len(), 0);
    assert!(runtime.submitted_prompts().is_empty());
    assert_eq!(runtime.state().input_buffer, "@fixer fix failing test");

    assert!(
        !runtime
            .state()
            .timeline
            .items()
            .iter()
            .any(|item| matches!(item, TimelineItem::Delegation(_)))
    );
}

#[test]
fn queued_prompt_ack_requires_dispatched_prompt() {
    let mut runtime = runtime();
    runtime.session_turn_active = true;
    runtime.state_mut().phase = AppPhase::Running;
    runtime.state_mut().set_input("same");

    runtime
        .handle_input_action(InputAction::Submit)
        .expect("queue succeeds");

    runtime.apply_session_transport_event(SessionTransportEvent::UserMessage(
        UserMessageEvent::new("same"),
    ));

    assert_eq!(
        runtime.queued_prompts.iter().cloned().collect::<Vec<_>>(),
        vec!["same".to_string()]
    );
    assert_eq!(
        runtime
            .state()
            .timeline
            .items()
            .iter()
            .filter(|item| matches!(item, TimelineItem::User(message) if message.text == "same"))
            .count(),
        2
    );
    assert!(runtime
            .state()
            .timeline
            .items()
            .iter()
            .any(|item| matches!(item, TimelineItem::User(message) if message.text == "same" && message.queued)));
}

#[test]
fn queued_prompt_preview_does_not_reset_active_turn_state_until_ack() {
    let mut runtime = runtime();
    runtime.session_turn_active = true;
    runtime.state_mut().phase = AppPhase::Running;
    runtime.state_mut().active_tool_call_id = Some("tool-1".into());
    runtime.state_mut().latest_todo = Some(crate::tui::timeline::TodoView {
        items: vec![TodoItem {
            id: "todo-1".into(),
            content: "keep working".into(),
            status: TodoStatus::InProgress,
        }],
        auto_continue: AutoContinueState { enabled: true },
    });
    runtime.state_mut().set_input("follow up");

    runtime
        .handle_input_action(InputAction::Submit)
        .expect("queue succeeds");

    assert_eq!(
        runtime.state().active_tool_call_id.as_deref(),
        Some("tool-1")
    );
    assert!(runtime.state().latest_todo.is_some());

    runtime.apply_session_transport_event(SessionTransportEvent::Done);
    let Some(RuntimeCommand::SubmitPrompt(submission)) = runtime.take_next_queued_prompt_command()
    else {
        panic!("expected queued submit command");
    };
    runtime.apply_session_transport_event(SessionTransportEvent::UserMessage(
        UserMessageEvent::from_submission(submission),
    ));

    assert_eq!(runtime.state().active_tool_call_id, None);
    assert!(runtime.state().latest_todo.is_none());
    assert!(matches!(
        runtime.state().timeline.items().last(),
        Some(TimelineItem::User(message)) if message.text == "follow up" && !message.queued
    ));
}

#[test]
fn queued_prompt_survives_parent_view_navigation() {
    let mut runtime = runtime();
    runtime.session_turn_active = true;
    runtime.state_mut().phase = AppPhase::Running;
    runtime.state_mut().set_input("follow up with attachment");
    runtime
        .state_mut()
        .add_composer_attachment(UserImageAttachment {
            id: "img-queued".into(),
            label: "screen.png".into(),
            mime: "image/png".into(),
            data_url: "data:image/png;base64,AAAA".into(),
        });

    runtime
        .handle_input_action(InputAction::Submit)
        .expect("queue succeeds");

    assert_eq!(runtime.queued_prompts.len(), 1);
    assert_eq!(runtime.queued_prompts[0].content.attachments.len(), 1);

    // Parent view navigation reprojects the parent timeline from transcript
    // records; it must not reset the queued submission or its preview.
    runtime.apply_session_transport_event(SessionTransportEvent::ParentSessionViewed {
        session_id: "session-1".into(),
        branch_id: "root".into(),
        records: Vec::new(),
        model_id: Some("gpt-5.5".into()),
        token_usage: None,
        runtime_context: event_context("session-1", 0),
    });

    assert_eq!(
        runtime.queued_prompts.len(),
        1,
        "queued prompt must survive parent view navigation"
    );
    assert_eq!(
        runtime.queued_prompts[0].content.attachments[0].id,
        "img-queued"
    );
    assert!(!runtime.queued_prompt_lifecycle.has_inflight_handoff());
    assert!(runtime.session_turn_active);
    assert!(runtime.state().timeline.items().iter().any(|item| {
        matches!(
            item,
            TimelineItem::User(message)
                if message.queued
                    && message.attachments.first().is_some_and(|attachment| {
                        attachment.id == "img-queued"
                    })
        )
    }));

    // The queued prompt remains dispatchable once the running turn finishes.
    runtime.apply_session_transport_event(SessionTransportEvent::Done);
    let Some(RuntimeCommand::SubmitPrompt(submission)) = runtime.take_next_queued_prompt_command()
    else {
        panic!("expected queued submit command after parent view navigation");
    };
    assert_eq!(submission.content.attachments[0].id, "img-queued");
}

#[test]
fn parent_view_navigation_preserves_inflight_handoff_and_ack_activates_republished_preview() {
    let mut runtime = runtime();
    runtime.session_turn_active = true;
    runtime.state_mut().phase = AppPhase::Running;
    runtime.state_mut().set_input("follow up");

    runtime
        .handle_input_action(InputAction::Submit)
        .expect("queue succeeds");

    // Dispatch the queued prompt; the handoff is in flight (no ack yet).
    runtime.apply_session_transport_event(SessionTransportEvent::Done);
    let Some(RuntimeCommand::SubmitPrompt(submission)) = runtime.take_next_queued_prompt_command()
    else {
        panic!("expected queued submit command");
    };
    assert!(runtime.queued_prompt_lifecycle.has_inflight_handoff());

    // Parent view navigation must preserve the in-flight handoff and
    // republish the preview for the still-unacked submission.
    runtime.apply_session_transport_event(SessionTransportEvent::ParentSessionViewed {
        session_id: "session-1".into(),
        branch_id: "root".into(),
        records: Vec::new(),
        model_id: Some("gpt-5.5".into()),
        token_usage: None,
        runtime_context: event_context("session-1", 0),
    });

    assert!(runtime.queued_prompt_lifecycle.has_inflight_handoff());
    assert_eq!(runtime.queued_prompts.len(), 1);
    assert!(runtime.state().timeline.items().iter().any(|item| {
        matches!(
            item,
            TimelineItem::User(message) if message.text == "follow up" && message.queued
        )
    }));

    // The late ack activates the republished preview instead of duplicating
    // the message, and the handoff resolves.
    runtime.apply_session_transport_event(SessionTransportEvent::UserMessage(
        UserMessageEvent::from_submission(submission),
    ));
    assert!(!runtime.queued_prompt_lifecycle.has_inflight_handoff());
    assert!(runtime.queued_prompts.is_empty());
    assert_eq!(
        runtime
            .state()
            .timeline
            .items()
            .iter()
            .filter(
                |item| matches!(item, TimelineItem::User(message) if message.text == "follow up")
            )
            .count(),
        1,
        "republished preview must be activated, not duplicated"
    );
}

#[test]
fn parent_view_navigation_preserves_pending_permission_projection() {
    let mut runtime = runtime();
    runtime.session_turn_active = true;
    runtime.state_mut().phase = AppPhase::Running;
    let (reply_tx, _reply_rx) = oneshot::channel();
    runtime.apply_session_transport_event(SessionTransportEvent::PermissionRequested {
        event: PermissionRequestEvent::new("call-1", "shell__exec", "Run command"),
        handle: RunnerPermissionRequest::new(reply_tx),
    });
    assert!(runtime.state().pending_permission.is_some());
    assert_eq!(runtime.state().phase, AppPhase::WaitingForPermission);

    // Parent view navigation reprojects the parent timeline; the pending
    // permission request must remain visible and resolvable.
    runtime.apply_session_transport_event(SessionTransportEvent::ParentSessionViewed {
        session_id: "session-1".into(),
        branch_id: "root".into(),
        records: Vec::new(),
        model_id: Some("gpt-5.5".into()),
        token_usage: None,
        runtime_context: event_context("session-1", 0),
    });

    assert!(
        runtime.state().pending_permission.is_some(),
        "pending permission must survive parent view navigation"
    );
    assert_eq!(runtime.state().phase, AppPhase::WaitingForPermission);
    assert!(runtime.permission_lifecycle.is_pending());
}

#[test]
fn parent_view_navigation_preserves_pending_question_projection() {
    for child_origin in [false, true] {
        let mut runtime = runtime();
        runtime.session_turn_active = true;
        runtime.state_mut().phase = AppPhase::Running;
        let (tx, mut rx) = oneshot::channel();
        if child_origin {
            runtime.apply_session_transport_event(SessionTransportEvent::ChildQuestionRequested {
                child_session_id: "child-1".into(),
                request: sample_question_request(false),
                handle: RunnerQuestionRequest::new(tx),
            });
        } else {
            runtime.apply_session_transport_event(SessionTransportEvent::QuestionRequested {
                request: sample_question_request(false),
                handle: RunnerQuestionRequest::new(tx),
            });
        }
        assert!(runtime.state().pending_question.is_some());
        assert_eq!(runtime.state().phase, AppPhase::WaitingForPermission);

        // Parent view navigation reprojects the parent timeline; the
        // unanswered question dialog must remain visible and answerable.
        runtime.apply_session_transport_event(SessionTransportEvent::ParentSessionViewed {
            session_id: "session-1".into(),
            branch_id: "root".into(),
            records: Vec::new(),
            model_id: Some("gpt-5.5".into()),
            token_usage: None,
            runtime_context: event_context("session-1", 0),
        });

        assert!(
            runtime.state().pending_question.is_some(),
            "pending question must survive parent view navigation (child_origin={child_origin})"
        );
        assert_eq!(runtime.state().phase, AppPhase::WaitingForPermission);

        // The preserved dialog remains answerable through its live handle.
        runtime
            .handle_input_action(InputAction::QuestionPickOption(1))
            .expect("pick succeeds");
        assert!(runtime.state().pending_question.is_none());
        assert_eq!(
            rx.try_recv(),
            Ok(Ok(crate::tool::QuestionResponse {
                answers: vec![vec!["Fast".into()]],
            })),
            "answer must reach the engine after parent view navigation (child_origin={child_origin})"
        );
    }
}

#[test]
fn parent_view_navigation_projection_failure_restores_pending_question() {
    let mut runtime = runtime();
    runtime.session_turn_active = true;
    runtime.state_mut().phase = AppPhase::Running;
    let (tx, _rx) = oneshot::channel();
    runtime.apply_session_transport_event(SessionTransportEvent::QuestionRequested {
        request: sample_question_request(false),
        handle: RunnerQuestionRequest::new(tx),
    });
    assert!(runtime.state().pending_question.is_some());

    // Records whose session id conflicts with the projected runtime context
    // make the timeline projection fail; the pending question must be
    // restored untouched and the error surfaced.
    let conflicting_records = vec![TranscriptRecord {
        session_id: "other-session".into(),
        sequence: 1,
        timestamp_ms: 0,
        context_branch_id: None,
        event: TranscriptEvent::SessionStarted {
            model: "gpt-test".into(),
        },
    }];
    runtime.apply_session_transport_event(SessionTransportEvent::ParentSessionViewed {
        session_id: "session-1".into(),
        branch_id: "root".into(),
        records: conflicting_records,
        model_id: Some("gpt-5.5".into()),
        token_usage: None,
        runtime_context: event_context("session-1", 0),
    });

    assert!(
        runtime.state().pending_question.is_some(),
        "pending question must be restored when the projection fails"
    );
    assert!(runtime.pending_question_handle.is_some());
    assert!(matches!(
        runtime.state().toast(),
        Some(toast) if toast.message.contains("Context projection failed") && toast.kind == ToastKind::Error
    ));
}

#[test]
fn dispatched_queued_prompt_failure_before_ack_clears_handoff_without_redispatch() {
    let mut runtime = runtime();
    runtime.state_mut().phase = AppPhase::Running;
    runtime.state_mut().set_input("follow up");
    runtime
        .handle_input_action(InputAction::Submit)
        .expect("queue succeeds");

    runtime.apply_session_transport_event(SessionTransportEvent::Done);
    let Some(RuntimeCommand::SubmitPrompt(submission)) = runtime.take_next_queued_prompt_command()
    else {
        panic!("expected queued submit command");
    };
    runtime.apply_session_transport_event(SessionTransportEvent::QueuedPromptAccepted {
        prompt: submission,
    });

    runtime.apply_session_transport_event(SessionTransportEvent::Error(ErrorEvent::new(
        "missing API key",
    )));
    runtime.apply_session_transport_event(SessionTransportEvent::Done);

    assert!(runtime.queued_prompts.is_empty());
    assert_eq!(runtime.queued_prompt_lifecycle.dispatched_prompt(), None);
    assert!(!runtime.queued_prompt_lifecycle.has_inflight_handoff());
    assert!(!runtime.queued_prompt_lifecycle.failed_after_accept());
    assert_eq!(runtime.take_next_queued_prompt_command(), None);
    assert!(!runtime
            .state()
            .timeline
            .items()
            .iter()
            .any(|item| matches!(item, TimelineItem::User(message) if message.text == "follow up" && message.queued)));

    runtime.state_mut().set_input("after failure");
    assert_eq!(
        runtime.handle_input_action(InputAction::Submit).unwrap(),
        Some(RuntimeCommand::SubmitPrompt("after failure".into()))
    );
}

#[test]
fn old_error_done_before_queued_prompt_accept_does_not_consume_handoff() {
    let mut runtime = runtime();
    runtime.session_turn_active = true;
    runtime.state_mut().phase = AppPhase::Running;
    runtime.state_mut().set_input("follow up 1");
    runtime
        .handle_input_action(InputAction::Submit)
        .expect("first queue succeeds");
    runtime.state_mut().set_input("follow up 2");
    runtime
        .handle_input_action(InputAction::Submit)
        .expect("second queue succeeds");

    runtime.apply_session_transport_event(SessionTransportEvent::Done);
    let Some(RuntimeCommand::SubmitPrompt(first_submission)) =
        runtime.take_next_queued_prompt_command()
    else {
        panic!("expected first queued submit command");
    };

    runtime.apply_session_transport_event(SessionTransportEvent::Error(ErrorEvent::new(
        "old turn failed",
    )));
    runtime.apply_session_transport_event(SessionTransportEvent::ToolBatchFinished);

    assert_eq!(runtime.take_next_queued_prompt_command(), None);
    assert_eq!(
        runtime.queued_prompts.iter().cloned().collect::<Vec<_>>(),
        vec!["follow up 1".to_string(), "follow up 2".to_string()]
    );
    assert_eq!(
        runtime.queued_prompt_lifecycle.dispatched_prompt(),
        Some("follow up 1")
    );
    assert!(runtime.queued_prompt_lifecycle.has_inflight_handoff());
    assert!(!runtime.queued_prompt_lifecycle.is_accepted());
    assert!(!runtime.queued_prompt_lifecycle.failed_after_accept());

    runtime.apply_session_transport_event(SessionTransportEvent::QueuedPromptAccepted {
        prompt: first_submission.clone(),
    });
    runtime.apply_session_transport_event(SessionTransportEvent::UserMessage(
        UserMessageEvent::from_submission(first_submission),
    ));

    assert_eq!(
        runtime.queued_prompts.iter().cloned().collect::<Vec<_>>(),
        vec!["follow up 2".to_string()]
    );
    assert_eq!(runtime.queued_prompt_lifecycle.dispatched_prompt(), None);
    assert!(!runtime.queued_prompt_lifecycle.has_inflight_handoff());
    assert!(!runtime.queued_prompt_lifecycle.is_accepted());
    assert!(runtime.state().timeline.items().iter().any(
            |item| matches!(item, TimelineItem::User(message) if message.text == "follow up 1" && !message.queued)
        ));
}

#[test]
fn old_done_before_queued_prompt_ack_does_not_dispatch_next_prompt() {
    let mut runtime = runtime();
    runtime.session_turn_active = true;
    runtime.state_mut().phase = AppPhase::Running;
    runtime.state_mut().set_input("follow up 1");
    runtime
        .handle_input_action(InputAction::Submit)
        .expect("first queue succeeds");
    runtime.state_mut().set_input("follow up 2");
    runtime
        .handle_input_action(InputAction::Submit)
        .expect("second queue succeeds");

    runtime.apply_session_transport_event(SessionTransportEvent::Done);

    let Some(RuntimeCommand::SubmitPrompt(first_submission)) =
        runtime.take_next_queued_prompt_command()
    else {
        panic!("expected first queued submit command");
    };
    runtime.apply_session_transport_event(SessionTransportEvent::Done);

    assert_eq!(runtime.take_next_queued_prompt_command(), None);
    assert_eq!(
        runtime.queued_prompts.iter().cloned().collect::<Vec<_>>(),
        vec!["follow up 1".to_string(), "follow up 2".to_string()]
    );
    assert_eq!(
        runtime.queued_prompt_lifecycle.dispatched_prompt(),
        Some("follow up 1")
    );
    assert!(runtime.queued_prompt_lifecycle.has_inflight_handoff());
    assert!(runtime.state().timeline.items().iter().any(
            |item| matches!(item, TimelineItem::User(message) if message.text == "follow up 1" && message.queued)
        ));

    runtime.apply_session_transport_event(SessionTransportEvent::UserMessage(
        UserMessageEvent::from_submission(first_submission),
    ));

    assert_eq!(
        runtime.queued_prompts.iter().cloned().collect::<Vec<_>>(),
        vec!["follow up 2".to_string()]
    );
    assert_eq!(runtime.queued_prompt_lifecycle.dispatched_prompt(), None);
    assert!(!runtime.queued_prompt_lifecycle.has_inflight_handoff());
    assert_eq!(runtime.take_next_queued_prompt_command(), None);
    assert!(runtime.state().timeline.items().iter().any(
            |item| matches!(item, TimelineItem::User(message) if message.text == "follow up 1" && !message.queued)
        ));
}

#[test]
fn manual_submit_during_queued_handoff_is_queued_behind_pending_prompt() {
    let mut runtime = runtime();
    runtime.session_turn_active = true;
    runtime.state_mut().phase = AppPhase::Running;
    runtime.state_mut().set_input("follow up 1");
    runtime
        .handle_input_action(InputAction::Submit)
        .expect("queue succeeds");

    runtime.apply_session_transport_event(SessionTransportEvent::Done);
    assert_eq!(
        runtime.take_next_queued_prompt_command(),
        Some(RuntimeCommand::SubmitPrompt("follow up 1".into()))
    );
    runtime.apply_session_transport_event(SessionTransportEvent::Done);

    runtime.state_mut().set_input("manual follow up");
    let command = runtime
        .handle_input_action(InputAction::Submit)
        .expect("manual submit queues");

    assert_eq!(command, None);
    assert_eq!(
        runtime.queued_prompts.iter().cloned().collect::<Vec<_>>(),
        vec!["follow up 1".to_string(), "manual follow up".to_string()]
    );
    assert_eq!(
        runtime.queued_prompt_lifecycle.dispatched_prompt(),
        Some("follow up 1")
    );
    assert!(runtime.queued_prompt_lifecycle.has_inflight_handoff());
    assert!(runtime.state().timeline.items().iter().any(
            |item| matches!(item, TimelineItem::User(message) if message.text == "manual follow up" && message.queued)
        ));
}

#[test]
fn queued_prompt_dispatches_after_turn_done() {
    let mut runtime = runtime();
    runtime.state_mut().phase = AppPhase::Running;
    runtime.state_mut().set_input("follow up");
    runtime
        .handle_input_action(InputAction::Submit)
        .expect("queue succeeds");

    assert_eq!(runtime.take_next_queued_prompt_command(), None);
    runtime.apply_session_transport_event(SessionTransportEvent::Done);

    let command = runtime.take_next_queued_prompt_command();

    let Some(RuntimeCommand::SubmitPrompt(submission)) = command else {
        panic!("expected queued submit command");
    };
    assert_eq!(runtime.queued_prompts.len(), 1);
    assert_eq!(runtime.state().phase, AppPhase::Running);
    assert!(matches!(
        runtime.state().timeline.items().last(),
        Some(TimelineItem::User(message)) if message.text == "follow up" && message.queued
    ));

    runtime.apply_session_transport_event(SessionTransportEvent::UserMessage(
        UserMessageEvent::from_submission(submission),
    ));

    assert_eq!(runtime.queued_prompts.len(), 0);
    assert!(matches!(
        runtime.state().timeline.items().last(),
        Some(TimelineItem::User(message)) if message.text == "follow up" && !message.queued
    ));
    assert_eq!(
        runtime
            .state()
            .timeline
            .items()
            .iter()
            .filter(
                |item| matches!(item, TimelineItem::User(message) if message.text == "follow up")
            )
            .count(),
        1
    );
}

#[test]
fn queued_prompts_become_history_on_interruption() {
    let mut runtime = runtime();
    runtime.state_mut().phase = AppPhase::Running;
    runtime.state_mut().set_input("follow up");
    runtime
        .handle_input_action(InputAction::Submit)
        .expect("queue succeeds");

    runtime.apply_session_transport_event(SessionTransportEvent::Interrupted);

    assert!(runtime.queued_prompts.is_empty());
    assert_eq!(runtime.take_next_queued_prompt_command(), None);
    assert!(runtime.state().timeline.items().iter().any(
            |item| matches!(item, TimelineItem::User(message) if message.text == "follow up" && !message.queued)
        ));
}

#[test]
fn interrupted_session_transport_event_clears_inflight_queued_prompt_handoff_state() {
    let mut runtime = runtime();
    runtime.session_turn_active = true;
    runtime.state_mut().phase = AppPhase::Running;
    runtime.state_mut().set_input("follow up");
    runtime
        .handle_input_action(InputAction::Submit)
        .expect("queue succeeds");

    runtime.apply_session_transport_event(SessionTransportEvent::Done);
    let Some(RuntimeCommand::SubmitPrompt(submission)) = runtime.take_next_queued_prompt_command()
    else {
        panic!("expected queued submit command");
    };
    runtime.apply_session_transport_event(SessionTransportEvent::QueuedPromptAccepted {
        prompt: submission,
    });

    runtime.apply_session_transport_event(SessionTransportEvent::Interrupted);

    assert!(runtime.queued_prompts.is_empty());
    assert_eq!(runtime.queued_prompt_lifecycle.dispatched_prompt(), None);
    assert!(!runtime.queued_prompt_lifecycle.has_inflight_handoff());
    assert!(!runtime.queued_prompt_lifecycle.is_accepted());
    assert!(!runtime.queued_prompt_lifecycle.failed_after_accept());
    assert_eq!(runtime.take_next_queued_prompt_command(), None);
    assert!(runtime.state().timeline.items().iter().any(
            |item| matches!(item, TimelineItem::User(message) if message.text == "follow up" && !message.queued)
        ));
}

#[test]
fn queued_prompt_accept_does_not_consume_history_until_user_message_arrives() {
    let mut runtime = runtime();
    runtime.session_turn_active = true;
    runtime.state_mut().phase = AppPhase::Running;
    runtime.state_mut().set_input("follow up");
    runtime
        .handle_input_action(InputAction::Submit)
        .expect("queue succeeds");

    runtime.apply_session_transport_event(SessionTransportEvent::Done);
    let Some(RuntimeCommand::SubmitPrompt(submission)) = runtime.take_next_queued_prompt_command()
    else {
        panic!("expected queued submit command");
    };

    runtime.apply_session_transport_event(SessionTransportEvent::QueuedPromptAccepted {
        prompt: submission.clone(),
    });

    assert_eq!(
        runtime.queued_prompts.iter().cloned().collect::<Vec<_>>(),
        vec!["follow up".to_string()]
    );
    assert_eq!(
        runtime.queued_prompt_lifecycle.dispatched_prompt(),
        Some("follow up")
    );
    assert!(runtime.queued_prompt_lifecycle.has_inflight_handoff());
    assert!(runtime.queued_prompt_lifecycle.is_accepted());
    assert!(runtime.state().timeline.items().iter().any(
            |item| matches!(item, TimelineItem::User(message) if message.text == "follow up" && message.queued)
        ));

    runtime.apply_session_transport_event(SessionTransportEvent::UserMessage(
        UserMessageEvent::from_submission(submission),
    ));

    assert!(runtime.queued_prompts.is_empty());
    assert_eq!(runtime.queued_prompt_lifecycle.dispatched_prompt(), None);
    assert!(!runtime.queued_prompt_lifecycle.has_inflight_handoff());
    assert!(!runtime.queued_prompt_lifecycle.is_accepted());
    assert!(runtime.state().timeline.items().iter().any(
            |item| matches!(item, TimelineItem::User(message) if message.text == "follow up" && !message.queued)
        ));
}

#[test]
fn queued_prompt_does_not_dispatch_after_single_tool_finished() {
    let mut runtime = runtime();
    runtime.session_turn_active = true;
    runtime.state_mut().phase = AppPhase::Running;
    runtime.state_mut().set_input("follow up");
    runtime
        .handle_input_action(InputAction::Submit)
        .expect("queue succeeds");

    runtime.apply_session_transport_event(SessionTransportEvent::ToolFinished(
        ToolFinishedEvent::new("tool-1", "fs__read", "read completed", ToolOutcome::Success),
    ));

    assert_eq!(runtime.take_next_queued_prompt_command(), None);
    assert_eq!(runtime.queued_prompt_lifecycle.dispatched_prompt(), None);
    assert!(!runtime.queued_prompt_lifecycle.has_inflight_handoff());
}

#[test]
fn queued_prompt_dispatches_after_tool_batch_finished() {
    let mut runtime = runtime();
    runtime.session_turn_active = true;
    runtime.state_mut().phase = AppPhase::Running;
    runtime.state_mut().set_input("follow up");
    runtime
        .handle_input_action(InputAction::Submit)
        .expect("queue succeeds");

    runtime.apply_session_transport_event(SessionTransportEvent::ToolFinished(
        ToolFinishedEvent::new("tool-1", "fs__read", "read completed", ToolOutcome::Success),
    ));
    assert_eq!(runtime.take_next_queued_prompt_command(), None);

    runtime.apply_session_transport_event(SessionTransportEvent::ToolBatchFinished);

    assert_eq!(
        runtime.take_next_queued_prompt_command(),
        Some(RuntimeCommand::SubmitPrompt("follow up".into()))
    );
    assert_eq!(
        runtime.queued_prompt_lifecycle.dispatched_prompt(),
        Some("follow up")
    );
}

#[test]
fn non_terminal_error_does_not_drop_or_dispatch_queued_prompt() {
    let mut runtime = runtime();
    runtime.state_mut().phase = AppPhase::Running;
    runtime.state_mut().set_input("follow up");
    runtime
        .handle_input_action(InputAction::Submit)
        .expect("queue succeeds");

    runtime.apply_session_transport_event(SessionTransportEvent::Error(ErrorEvent::new(
        "failed to view child transcript",
    )));

    assert_eq!(runtime.queued_prompts.len(), 1);
    assert_eq!(runtime.take_next_queued_prompt_command(), None);
    assert!(runtime
            .state()
            .timeline
            .items()
            .iter()
            .any(|item| matches!(item, TimelineItem::User(message) if message.text == "follow up" && message.queued)));
}

#[test]
fn prompt_after_non_terminal_error_still_queues_until_done() {
    let mut runtime = runtime();
    runtime.state_mut().phase = AppPhase::Running;
    runtime.session_turn_active = true;
    runtime.state_mut().set_input("follow up 1");
    runtime
        .handle_input_action(InputAction::Submit)
        .expect("first queue succeeds");

    runtime.apply_session_transport_event(SessionTransportEvent::Error(ErrorEvent::new(
        "failed to view child transcript",
    )));
    runtime.state_mut().set_input("follow up 2");
    let command = runtime
        .handle_input_action(InputAction::Submit)
        .expect("second queue succeeds");

    assert_eq!(command, None);
    assert_eq!(
        runtime.queued_prompts.iter().cloned().collect::<Vec<_>>(),
        vec!["follow up 1".to_string(), "follow up 2".to_string()]
    );

    runtime.apply_session_transport_event(SessionTransportEvent::Done);

    assert_eq!(
        runtime.take_next_queued_prompt_command(),
        Some(RuntimeCommand::SubmitPrompt("follow up 1".into()))
    );
    assert_eq!(
        runtime.queued_prompts.iter().cloned().collect::<Vec<_>>(),
        vec!["follow up 1".to_string(), "follow up 2".to_string()]
    );
    assert!(matches!(
        runtime.state().timeline.items().iter().find(|item| matches!(item, TimelineItem::User(message) if message.text == "follow up 1")),
        Some(TimelineItem::User(message)) if message.queued
    ));
}

#[test]
fn session_transport_events_continue_updating_parent_timeline_while_viewing_child() {
    let mut runtime = runtime();
    runtime
        .state_mut()
        .replace_session_timeline_from_records(&[TranscriptRecord {
            session_id: "parent-session".into(),
            sequence: 1,
            timestamp_ms: 0,
            context_branch_id: None,
            event: TranscriptEvent::UserMessage {
                content: "parent prompt".into(),
            },
        }]);
    runtime.state_mut().replace_child_timeline_from_records(
        &[TranscriptRecord {
            session_id: "child-session".into(),
            sequence: 1,
            timestamp_ms: 1,
            context_branch_id: None,
            event: TranscriptEvent::AssistantMessage {
                content: "child response".into(),
            },
        }],
        "parent-session",
        "child-session",
        "explorer",
        0,
        1,
        1,
    );

    runtime.apply_session_transport_event(SessionTransportEvent::ToolStarted(
        crate::tui::events::ToolStartedEvent {
            call_id: "call-1".into(),
            name: "shell__exec".into(),
            summary: "run ls".into(),
            arguments: Some("ls".into()),
        },
    ));
    runtime.apply_session_transport_event(SessionTransportEvent::Done);

    assert!(matches!(
        runtime.state().active_timeline().items(),
        [crate::tui::TimelineItem::Assistant(message)] if message.text == "child response"
    ));
    assert!(matches!(
        runtime.state().timeline.items().last(),
        Some(crate::tui::TimelineItem::Tool(tool)) if tool.call_id == "call-1"
    ));

    runtime.state_mut().restore_parent_timeline_view();

    assert!(matches!(
        runtime.state().active_timeline().items().first(),
        Some(crate::tui::TimelineItem::User(message)) if message.text == "parent prompt"
    ));
    assert!(matches!(
        runtime.state().active_timeline().items().last(),
        Some(crate::tui::TimelineItem::Tool(tool)) if tool.call_id == "call-1"
    ));
}

#[test]
fn tree_dispatches_session_history_and_retired_branch_commands_are_invalid() {
    let mut runtime = runtime();
    runtime.state_mut().set_input("/branches");
    assert_eq!(
        runtime
            .handle_input_action(InputAction::Submit)
            .expect("branches command"),
        None
    );

    runtime.state_mut().set_input("/tree");
    assert_eq!(
        runtime
            .handle_input_action(InputAction::Submit)
            .expect("tree command"),
        Some(RuntimeCommand::ShowHistoryTree)
    );
}

#[test]
fn recorder_current_branch_updates_on_resume_checkout_and_new_session() {
    let sessions_dir = std::env::temp_dir().join(format!(
        "letcode-branch-sync-test-{}",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("time ok")
            .as_nanos()
    ));
    std::fs::create_dir_all(&sessions_dir).expect("create sessions dir");
    let mut recorder = TranscriptRecorder::create(&sessions_dir).expect("create recorder");
    recorder
        .record_session_started("gpt-test")
        .expect("session started");
    recorder.record_user_message("root").expect("root message");
    recorder
        .record_context_branch_created(
            "feature",
            crate::transcript::ROOT_CONTEXT_BRANCH_ID,
            2,
            None,
        )
        .expect("branch created");
    recorder.set_current_context_branch_id(Some("feature".into()));
    recorder
        .record_assistant_message("branch reply")
        .expect("branch message");
    recorder
        .record_context_checkout("feature", 4)
        .expect("checkout metadata");
    let session_id = recorder.session_id().to_string();
    let path = recorder.path().to_path_buf();

    let records = read_records(&path).expect("read records");
    let snapshot =
        transcript_projection::project_session_restore_snapshot(session_id.clone(), records)
            .expect("snapshot");
    let mut reopened =
        TranscriptRecorder::open_existing(&sessions_dir, &session_id).expect("reopen recorder");
    sync_recorder_branch(&mut reopened, &snapshot.branch_id);
    assert_eq!(reopened.current_context_branch_id(), Some("feature"));

    sync_recorder_branch(&mut reopened, crate::transcript::ROOT_CONTEXT_BRANCH_ID);
    assert_eq!(reopened.current_context_branch_id(), None);

    let mut fresh = TranscriptRecorder::create(&sessions_dir).expect("fresh recorder");
    fresh.set_current_context_branch_id(Some("temp".into()));
    fresh.set_current_context_branch_id(None);
    assert_eq!(fresh.current_context_branch_id(), None);
}

#[test]
fn tick_does_not_clobber_live_child_stream_with_disk_records() {
    let sessions_dir = std::env::temp_dir().join(format!(
        "letcode-tui-child-live-refresh-{}",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("time ok")
            .as_nanos()
    ));
    let mut parent = TranscriptRecorder::create(&sessions_dir).expect("create parent");
    let child_dir = crate::transcript::child_sessions_dir(&sessions_dir);
    let mut child = TranscriptRecorder::create(&child_dir).expect("create child");
    let parent_session_id = parent.session_id().to_string();
    let child_session_id = child.session_id().to_string();

    child
        .record_session_started("gpt-child")
        .expect("record child start");
    parent
        .record_subagent_result(
            "run-1",
            &parent_session_id,
            "turn-1",
            &child_session_id,
            "explorer",
            "running",
            "inspecting",
        )
        .expect("record child result");

    let (_tx, rx) = mpsc::unbounded_channel();
    let mut runtime = TuiRuntime::new(
        TuiState::default(),
        rx,
        vec![AvailableModel::new("gpt-5.5", "GPT-5.5")],
        Vec::new(),
        sessions_dir.clone(),
        std::env::temp_dir(),
    );
    runtime.state_mut().replace_child_timeline_from_records(
        &[TranscriptRecord {
            session_id: child_session_id.clone(),
            sequence: 1,
            timestamp_ms: 0,
            context_branch_id: None,
            event: TranscriptEvent::SessionStarted {
                model: "gpt-child".into(),
            },
        }],
        parent_session_id,
        child_session_id.clone(),
        "explorer",
        0,
        1,
        1,
    );
    runtime.apply_session_transport_event(SessionTransportEvent::ChildSessionEvent {
        child_session_id: child_session_id.clone(),
        agent_name: None,
        parent_tool_call_id: None,
        event: SessionEvent::AssistantDelta(crate::tui::events::AssistantDeltaEvent::new(
            "partial stream",
        )),
    });

    child
        .record_tool_call_started("call-1", "shell__exec", serde_json::json!({}))
        .expect("record child tool start");

    runtime
        .handle_input_action(InputAction::Tick)
        .expect("tick succeeds");

    let metadata = runtime
        .state()
        .child_view_metadata()
        .expect("child metadata");
    assert_eq!(metadata.record_count, 1);
    assert!(matches!(
        runtime.state().active_timeline().items().last(),
        Some(crate::tui::TimelineItem::Assistant(message)) if message.text == "partial stream"
    ));
}

#[test]
fn remove_current_empty_session_deletes_session_started_only_transcript() {
    let sessions_dir = std::env::temp_dir().join(format!(
        "letcode-tui-remove-current-empty-session-{}",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("system clock before unix epoch")
            .as_nanos()
    ));
    let mut recorder = TranscriptRecorder::create(&sessions_dir).expect("create recorder");
    recorder
        .record_session_started("gpt-test")
        .expect("record session started");
    let path = recorder.path().to_path_buf();
    let recorder = Arc::new(StdMutex::new(recorder));

    assert!(remove_current_empty_session(&recorder).expect("remove empty session"));
    assert!(!path.exists());
}

#[test]
fn session_resumed_event_replaces_timeline_not_appends() {
    let mut runtime = runtime();
    runtime.session_resume_pending = true;
    runtime
        .state_mut()
        .timeline
        .push_assistant_delta(AssistantDeltaEvent::new("current session notice"));

    runtime.apply_session_transport_event(SessionTransportEvent::SessionResumed {
        session_id: "session-1".into(),
        branch_id: crate::transcript::ROOT_CONTEXT_BRANCH_ID.into(),
        messages: vec![crate::agent::ConversationMessage {
            role: crate::agent::ConversationRole::User,
            content: "old prompt".into(),
        }],
        records: vec![crate::transcript::TranscriptRecord {
            session_id: "session-1".into(),
            sequence: 1,
            timestamp_ms: 0,
            context_branch_id: None,
            event: crate::transcript::TranscriptEvent::UserMessage {
                content: "old prompt".into(),
            },
        }],
        evidence_count: 2,
        model_id: None,
        token_usage: None,
        runtime_context: event_context("session-1", 1),
        expert_models: indexmap::IndexMap::new(),
    });

    assert!(matches!(
        runtime.state().timeline.items().first(),
        Some(crate::tui::TimelineItem::User(message)) if message.text == "old prompt"
    ));

    assert!(runtime.state().active_session);
    assert!(!runtime.session_resume_pending);
}

#[test]
fn session_error_clears_pending_idle_resume() {
    let mut runtime = runtime();
    runtime.session_resume_pending = true;

    runtime.apply_session_transport_event(SessionTransportEvent::Error(ErrorEvent::new(
        "resume failed",
    )));

    assert!(!runtime.session_resume_pending);
    assert!(runtime.state().timeline.items().iter().any(|item| matches!(
        item,
        TimelineItem::Error(error) if error.message == "resume failed"
    )));
}

#[test]
fn session_resume_projection_failure_clears_pending_and_reports_error() {
    let mut runtime = runtime();
    runtime.session_resume_pending = true;

    runtime.apply_session_transport_event(SessionTransportEvent::SessionResumed {
        session_id: "session-1".into(),
        branch_id: crate::transcript::ROOT_CONTEXT_BRANCH_ID.into(),
        messages: Vec::new(),
        records: vec![TranscriptRecord {
            session_id: "different-session".into(),
            sequence: 1,
            timestamp_ms: 0,
            context_branch_id: None,
            event: TranscriptEvent::UserMessage {
                content: "old prompt".into(),
            },
        }],
        evidence_count: 0,
        model_id: None,
        token_usage: None,
        runtime_context: event_context("session-1", 1),
        expert_models: indexmap::IndexMap::new(),
    });

    assert!(!runtime.session_resume_pending);
    assert!(runtime.state().toast.as_ref().is_some_and(|toast| {
        toast.kind == ToastKind::Error && toast.message.starts_with("Context projection failed:")
    }));
}

#[test]
fn session_resumed_event_restores_recorded_model() {
    let (_tx, rx) = mpsc::unbounded_channel();
    let mut runtime = TuiRuntime::new(
        TuiState::new("gpt-5.5", "GPT-5.5", "default"),
        rx,
        vec![
            AvailableModel::with_context_window("gpt-5.5", "GPT-5.5", Some(128_000)),
            AvailableModel::with_context_window("gpt-5.5-mini", "GPT-5.5 Mini", Some(64_000)),
        ],
        Vec::new(),
        std::env::temp_dir(),
        std::env::temp_dir(),
    );

    runtime.apply_session_transport_event(SessionTransportEvent::SessionResumed {
        session_id: "session-1".into(),
        branch_id: "feature-a".into(),
        messages: Vec::new(),
        records: vec![crate::transcript::TranscriptRecord {
            session_id: "session-1".into(),
            sequence: 1,
            timestamp_ms: 0,
            context_branch_id: None,
            event: crate::transcript::TranscriptEvent::ContextBranchCreated {
                branch_id: "feature-a".into(),
                parent_branch_id: crate::transcript::ROOT_CONTEXT_BRANCH_ID.into(),
                base_sequence: 0,
                label: None,
            },
        }],
        evidence_count: 0,
        model_id: Some("gpt-5.5-mini".into()),
        token_usage: Some(TokenUsageEvent::new(12_345, 64_000)),
        runtime_context: event_context("session-1", 1),
        expert_models: indexmap::IndexMap::new(),
    });

    assert_eq!(runtime.state().model_id, "gpt-5.5-mini");
    assert_eq!(runtime.state().current_context_branch, "feature-a");
    assert_eq!(runtime.state().model_label, "GPT-5.5 Mini");
    assert_eq!(
        runtime
            .state()
            .model_token_usage
            .as_ref()
            .map(|usage| usage.context_window_tokens),
        Some(64_000)
    );
    assert_eq!(
        runtime
            .state()
            .model_token_usage
            .as_ref()
            .map(|usage| usage.used_tokens),
        Some(12_345)
    );
    assert_eq!(
        runtime
            .state()
            .model_token_usage
            .as_ref()
            .and_then(|usage| usage.cache_report.as_ref()),
        None
    );
}

#[test]
fn session_resume_expert_snapshot_clears_previous_session_route() {
    let mut runtime = runtime_with_experts(vec![
        AvailableExpert {
            agent_name: "explorer".into(),
            route_id: "p/previous".into(),
        },
        AvailableExpert {
            agent_name: "reviewer".into(),
            route_id: "p/previous-reviewer".into(),
        },
    ]);

    runtime.apply_session_transport_event(SessionTransportEvent::SessionResumed {
        session_id: "session-1".into(),
        branch_id: crate::transcript::ROOT_CONTEXT_BRANCH_ID.into(),
        messages: Vec::new(),
        records: Vec::new(),
        evidence_count: 0,
        model_id: Some("p/main".into()),
        token_usage: None,
        runtime_context: event_context("session-1", 1),
        expert_models: indexmap::IndexMap::from([("explorer".into(), "p/session".into())]),
    });

    assert_eq!(runtime.available_experts[0].route_id, "p/session");
    assert_eq!(runtime.available_experts[1].route_id, "p/main");
}

#[test]
fn session_started_event_uses_complete_expert_snapshot() {
    let mut runtime = runtime_with_experts(vec![AvailableExpert {
        agent_name: "explorer".into(),
        route_id: "p/previous".into(),
    }]);

    runtime.apply_session_transport_event(SessionTransportEvent::SessionStarted {
        session_id: "new-session".into(),
        records: Vec::new(),
        runtime_context: event_context("new-session", 1),
        expert_models: indexmap::IndexMap::from([("explorer".into(), "p/new".into())]),
    });

    assert_eq!(runtime.available_experts[0].route_id, "p/new");
}

#[test]
fn session_started_event_clears_timeline_for_new_session() {
    let mut runtime = runtime();
    runtime
        .state_mut()
        .timeline
        .push_assistant_delta(AssistantDeltaEvent::new("current session notice"));

    runtime.apply_session_transport_event(SessionTransportEvent::SessionStarted {
        session_id: "new-session".into(),
        records: Vec::new(),
        runtime_context: event_context("new-session", 1),
        expert_models: indexmap::IndexMap::new(),
    });

    assert_eq!(runtime.state().timeline.items().len(), 0);
    assert!(!runtime.state().active_session);
    assert!(runtime.state().show_dashboard());
    assert_eq!(runtime.session_title, None);
}

#[test]
fn invalid_lifecycle_timeline_does_not_clear_parent_permission() {
    let mut runtime = runtime();
    let (tx, _rx) = oneshot::channel();
    runtime.apply_session_transport_event(SessionTransportEvent::PermissionRequested {
        event: PermissionRequestEvent::new("call-1", "shell__exec", "cargo test"),
        handle: RunnerPermissionRequest::new(tx),
    });
    let timeline_len = runtime.state().timeline.items().len();

    runtime.apply_session_transport_event(SessionTransportEvent::SessionResumed {
        session_id: "new-session".into(),
        branch_id: crate::transcript::ROOT_CONTEXT_BRANCH_ID.into(),
        messages: Vec::new(),
        records: vec![TranscriptRecord {
            session_id: "wrong-session".into(),
            sequence: 1,
            timestamp_ms: 0,
            context_branch_id: None,
            event: TranscriptEvent::UserMessage {
                content: "malformed lifecycle timeline".into(),
            },
        }],
        evidence_count: 0,
        model_id: None,
        token_usage: None,
        runtime_context: event_context("new-session", 1),
        expert_models: indexmap::IndexMap::new(),
    });

    assert!(runtime.pending_permission_handle().is_some());
    assert_eq!(runtime.state().phase, AppPhase::WaitingForPermission);
    assert_eq!(runtime.state().timeline.items().len(), timeline_len);
}

#[test]
fn runner_permission_events_update_state_and_handle() {
    let mut runtime = runtime();
    let (tx, _rx) = oneshot::channel();
    let handle = RunnerPermissionRequest::new(tx);
    runtime
        .state_mut()
        .show_toast("stale notice", ToastKind::Info);

    runtime.apply_session_transport_event(SessionTransportEvent::PermissionRequested {
        event: PermissionRequestEvent::new("call-1", "shell__exec", "cargo test"),
        handle: handle.clone(),
    });

    assert_eq!(runtime.state().phase, AppPhase::WaitingForPermission);
    assert!(runtime.pending_permission_handle().is_some());
    assert!(runtime.state().toast().is_none());

    runtime.apply_session_transport_event(SessionTransportEvent::PermissionResolved(
        PermissionResolutionEvent::approved("call-1"),
    ));

    assert!(runtime.pending_permission_handle().is_none());
    assert_eq!(runtime.state().pending_permission, None);
    let permission = runtime
        .state()
        .timeline
        .items()
        .iter()
        .find_map(|item| match item {
            crate::tui::TimelineItem::Permission(permission) => Some(permission),
            _ => None,
        })
        .expect("permission item exists");
    assert_eq!(
        permission.status,
        crate::tui::PermissionPromptStatus::Approved
    );
}

#[test]
fn child_session_viewed_does_not_clear_runtime_pending_permission() {
    let mut runtime = runtime();
    let (tx, _rx) = oneshot::channel();
    let handle = RunnerPermissionRequest::new(tx);

    runtime.apply_session_transport_event(SessionTransportEvent::ChildPermissionRequested {
        child_session_id: "child-session".into(),
        agent_name: Some("explorer".into()),
        parent_tool_call_id: Some("parent-call".into()),
        event: PermissionRequestEvent::new("call-1", "shell__exec", "cargo test"),
        handle,
    });
    runtime.apply_session_transport_event(SessionTransportEvent::ChildSessionViewed {
        parent_session_id: "parent-session".into(),
        child_session_id: "child-session".into(),
        agent_name: "explorer".into(),
        index: 0,
        total: 1,
        pool_ordinal: 1,
        records: vec![],
        runtime_context: event_context("child-session", 1),
    });

    assert!(runtime.pending_permission_handle().is_some());
    assert!(runtime.state().pending_permission.is_some());
    assert_eq!(runtime.state().phase, AppPhase::WaitingForPermission);
}

#[tokio::test]
async fn approve_and_deny_actions_respond_through_pending_handle() {
    let mut approve_runtime = runtime();
    let (approve_tx, approve_rx) = oneshot::channel();
    approve_runtime
        .permission_lifecycle
        .begin_parent(
            PermissionRequestEvent::new("call-a", "shell__exec", "ls"),
            RunnerPermissionRequest::new(approve_tx),
        )
        .expect("seed pending parent permission");
    approve_runtime.reproject_pending_permission();

    approve_runtime
        .handle_input_action(InputAction::ApprovePermission)
        .expect("approve succeeds");
    assert_eq!(
        approve_rx.await.expect("approval received"),
        PermissionResponse::AllowOnce
    );
    assert!(approve_runtime.pending_permission_handle().is_none());

    let mut deny_runtime = runtime();
    let (deny_tx, deny_rx) = oneshot::channel();
    deny_runtime
        .permission_lifecycle
        .begin_parent(
            PermissionRequestEvent::new("call-b", "shell__exec", "rm"),
            RunnerPermissionRequest::new(deny_tx),
        )
        .expect("seed pending parent permission");
    deny_runtime.reproject_pending_permission();

    deny_runtime
        .handle_input_action(InputAction::DenyPermission)
        .expect("deny succeeds");
    assert_eq!(
        deny_rx.await.expect("denial received"),
        PermissionResponse::Deny
    );
    assert!(deny_runtime.pending_permission_handle().is_none());
}

#[tokio::test]
async fn child_permission_request_survives_view_switch_and_can_be_approved() {
    let mut runtime = runtime();
    let (tx, rx) = oneshot::channel();

    runtime.apply_session_transport_event(SessionTransportEvent::ChildPermissionRequested {
        child_session_id: "child-session".into(),
        agent_name: Some("explorer".into()),
        parent_tool_call_id: Some("parent-call".into()),
        event: PermissionRequestEvent::new("call-1", "shell__exec", "cargo test"),
        handle: RunnerPermissionRequest::new(tx),
    });
    runtime.state_mut().restore_parent_timeline_view();

    runtime
        .handle_input_action(InputAction::ApprovePermission)
        .expect("approve succeeds");

    assert_eq!(
        rx.await.expect("approval received"),
        PermissionResponse::AllowOnce
    );
    assert!(runtime.pending_permission_handle().is_none());
    assert!(runtime.state().pending_permission.is_some());
}

#[test]
fn child_terminal_event_clears_matching_permission_handle() {
    for event in [
        SessionEvent::Done,
        SessionEvent::Interrupted,
        SessionEvent::Error(ErrorEvent::new("child stopped")),
    ] {
        let mut runtime = runtime();
        let (tx, _rx) = oneshot::channel();
        runtime.apply_session_transport_event(SessionTransportEvent::ChildPermissionRequested {
            child_session_id: "child-session".into(),
            agent_name: Some("explorer".into()),
            parent_tool_call_id: Some("parent-call".into()),
            event: PermissionRequestEvent::new("call-1", "shell__exec", "cargo test"),
            handle: RunnerPermissionRequest::new(tx),
        });

        runtime.apply_session_transport_event(SessionTransportEvent::ChildSessionEvent {
            child_session_id: "child-session".into(),
            agent_name: Some("explorer".into()),
            parent_tool_call_id: Some("parent-call".into()),
            event,
        });

        assert!(runtime.pending_permission_handle().is_none());
        assert!(runtime.state().pending_permission.is_none());
    }
}

#[test]
fn non_terminal_session_error_does_not_clear_pending_permission() {
    let mut runtime = runtime();
    let (tx, _rx) = oneshot::channel();
    runtime.apply_session_transport_event(SessionTransportEvent::PermissionRequested {
        event: PermissionRequestEvent::new("call-1", "shell__exec", "cargo test"),
        handle: RunnerPermissionRequest::new(tx),
    });

    runtime.apply_session_transport_event(SessionTransportEvent::Error(ErrorEvent::new(
        "failed to view child transcript",
    )));

    assert!(runtime.pending_permission_handle().is_some());
    assert!(runtime.state().pending_permission.is_some());
    assert_eq!(runtime.state().phase, AppPhase::WaitingForPermission);
}

#[tokio::test]
async fn second_permission_request_is_denied_without_replacing_active_one() {
    let mut runtime = runtime();
    let (first_tx, _first_rx) = oneshot::channel();
    let (second_tx, second_rx) = oneshot::channel();

    runtime.apply_session_transport_event(SessionTransportEvent::PermissionRequested {
        event: PermissionRequestEvent::new("call-1", "shell__exec", "cargo test"),
        handle: RunnerPermissionRequest::new(first_tx),
    });
    runtime.apply_session_transport_event(SessionTransportEvent::PermissionRequested {
        event: PermissionRequestEvent::new("call-2", "fs__write", "write file"),
        handle: RunnerPermissionRequest::new(second_tx),
    });

    assert_eq!(
        second_rx.await.expect("denial received"),
        PermissionResponse::Deny
    );
    assert_eq!(
        runtime
            .state()
            .pending_permission
            .as_ref()
            .map(|permission| permission.call_id.as_str()),
        Some("call-1")
    );
    assert_eq!(runtime.permission_lifecycle.child_session_id(), None);

    runtime.apply_session_transport_event(SessionTransportEvent::PermissionResolved(
        PermissionResolutionEvent::denied("call-2", None),
    ));
    runtime.apply_session_transport_event(SessionTransportEvent::ChildSessionEvent {
        child_session_id: "other-child".into(),
        agent_name: None,
        parent_tool_call_id: None,
        event: SessionEvent::Interrupted,
    });

    assert!(runtime.pending_permission_handle().is_some());
    assert_eq!(
        runtime
            .state()
            .pending_permission
            .as_ref()
            .map(|permission| permission.call_id.as_str()),
        Some("call-1")
    );
}

#[test]
fn draining_session_transport_events_is_bounded_so_input_can_make_progress() {
    let (tx, rx) = mpsc::unbounded_channel();
    let mut runtime = TuiRuntime::new(
        TuiState::default(),
        rx,
        vec![AvailableModel::new("gpt-5.5", "GPT-5.5")],
        Vec::new(),
        std::env::temp_dir(),
        std::env::temp_dir(),
    );
    for index in 0..300 {
        tx.send(SessionTransportEvent::UserMessage(UserMessageEvent::new(
            format!("message-{index}"),
        )))
        .expect("queue session transport event");
    }

    runtime.try_drain_session_events();

    assert!(runtime.session_transport_rx.try_recv().is_ok());
    assert_eq!(
        runtime
            .handle_input_action(InputAction::Interrupt)
            .expect("input is processed after bounded drain"),
        None
    );
}

#[test]
fn resumed_session_restores_latest_todo_state_from_records() {
    let mut runtime = runtime();
    let records = vec![
        TranscriptRecord {
            session_id: "s".into(),
            sequence: 1,
            timestamp_ms: 0,
            context_branch_id: None,
            event: TranscriptEvent::AutoContinueChanged {
                state: AutoContinueState { enabled: true },
            },
        },
        TranscriptRecord {
            session_id: "s".into(),
            sequence: 2,
            timestamp_ms: 1,
            context_branch_id: None,
            event: TranscriptEvent::TodoSnapshot {
                items: vec![TodoItem {
                    id: "t1".into(),
                    content: "inspect".into(),
                    status: TodoStatus::InProgress,
                }],
            },
        },
    ];

    runtime.apply_session_transport_event(SessionTransportEvent::SessionResumed {
        session_id: "s".into(),
        branch_id: crate::transcript::ROOT_CONTEXT_BRANCH_ID.into(),
        messages: Vec::new(),
        records,
        evidence_count: 0,
        model_id: None,
        token_usage: None,
        runtime_context: event_context("s", 2),
        expert_models: indexmap::IndexMap::new(),
    });

    let todo = runtime
        .state()
        .latest_todo
        .as_ref()
        .expect("todo state restored");
    assert_eq!(todo.items.len(), 1);
    assert_eq!(todo.items[0].status, TodoStatus::InProgress);
    assert!(todo.auto_continue.enabled);
}

const SESSION_ENGINE_INTEGRATION_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(3);

enum ControlledSseResponse {
    Blocked(String),
    Immediate(String),
}

struct ControlledSseServer {
    base_url: String,
    requests: mpsc::UnboundedReceiver<usize>,
    release: Arc<Notify>,
    task: JoinHandle<()>,
}

impl ControlledSseServer {
    async fn expect_request(&mut self, expected: usize) {
        let request = timeout(SESSION_ENGINE_INTEGRATION_TIMEOUT, self.requests.recv())
            .await
            .expect("timed out waiting for SSE request")
            .expect("SSE server stopped before the expected request");
        assert_eq!(request, expected);
    }

    async fn finish(self) {
        self.release.notify_waiters();
        self.release.notify_one();
        timeout(SESSION_ENGINE_INTEGRATION_TIMEOUT, self.task)
            .await
            .expect("SSE server should finish")
            .expect("SSE server task should not panic");
    }

    async fn abort(self) {
        self.task.abort();
        let _ = self.task.await;
    }
}

fn complete_http_request_len(request: &[u8]) -> Option<usize> {
    let header_end = request
        .windows(4)
        .position(|window| window == b"\r\n\r\n")?;
    let headers =
        std::str::from_utf8(&request[..header_end]).expect("test client sends UTF-8 HTTP headers");
    let content_length = headers
        .lines()
        .find_map(|header| {
            header
                .split_once(':')
                .filter(|(name, _)| name.eq_ignore_ascii_case("content-length"))
                .map(|(_, value)| {
                    value
                        .trim()
                        .parse::<usize>()
                        .expect("test client sends a numeric content length")
                })
        })
        .unwrap_or(0);
    Some(header_end + 4 + content_length)
}

async fn read_complete_http_request(socket: &mut tokio::net::TcpStream) {
    let mut request = Vec::new();
    loop {
        if complete_http_request_len(&request).is_some_and(|length| request.len() >= length) {
            return;
        }
        let read = socket
            .read_buf(&mut request)
            .await
            .expect("server reads request");
        assert_ne!(read, 0, "test client closed before completing its request");
    }
}

async fn spawn_controlled_sse_server(responses: Vec<ControlledSseResponse>) -> ControlledSseServer {
    let listener = TcpListener::bind("127.0.0.1:0")
        .await
        .expect("test server should bind");
    let address = listener
        .local_addr()
        .expect("test server has local address");
    let (request_tx, request_rx) = mpsc::unbounded_channel();
    let release = Arc::new(Notify::new());
    let server_release = Arc::clone(&release);
    let task = tokio::spawn(async move {
        for (index, response) in responses.into_iter().enumerate() {
            let (mut socket, _) = listener.accept().await.expect("server accepts request");
            read_complete_http_request(&mut socket).await;
            match response {
                ControlledSseResponse::Blocked(body) => {
                    socket
                            .write_all(
                                b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nConnection: close\r\n\r\n",
                            )
                            .await
                            .expect("server writes SSE headers");
                    socket.flush().await.expect("server flushes SSE headers");
                    let _ = request_tx.send(index);
                    server_release.notified().await;
                    let _ = socket.write_all(body.as_bytes()).await;
                    let _ = socket.shutdown().await;
                }
                ControlledSseResponse::Immediate(body) => {
                    let _ = request_tx.send(index);
                    let response = format!(
                        "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                        body.len()
                    );
                    socket
                        .write_all(response.as_bytes())
                        .await
                        .expect("server writes SSE response");
                    socket.shutdown().await.expect("server closes SSE response");
                }
            }
        }
    });
    ControlledSseServer {
        base_url: format!("http://{address}"),
        requests: request_rx,
        release,
        task,
    }
}

fn compaction_checkpoint(next_step: &str) -> String {
    format!(
        "## Progress\n### Done\n- completed work\n### In Progress\n- continue execution\n### Blocked\n- 无\n## Key Decisions\n- resolved scope\n## Validation\n- pending\n## File Operations\n### Read\n- 无\n### Modified\n- 无\n## Next Steps\n- {next_step}\n## Critical Context\n- durable workflow facts"
    )
}

fn responses_sse_tool_call_body(name: &str, arguments: serde_json::Value) -> String {
    let response = serde_json::json!({
        "type": "response.completed", "sequence_number": 1,
        "response": {
            "id": "r-tool-call", "object": "response", "created_at": 1,
            "status": "completed", "background": false, "error": null,
            "incomplete_details": null, "instructions": null, "max_output_tokens": null,
            "model": "m1", "output": [{
                "type": "function_call", "id": "fc-test", "call_id": "call-test",
                "name": name, "arguments": arguments.to_string(), "status": "completed"
            }],
            "parallel_tool_calls": true, "previous_response_id": null, "reasoning": {},
            "store": true, "temperature": 1, "text": {"format": {"type": "text"}},
            "tool_choice": "auto", "tools": [], "top_p": 1, "truncation": "disabled",
            "usage": {
                "input_tokens": 1, "input_tokens_details": {"cached_tokens": 0},
                "output_tokens": 1, "output_tokens_details": {"reasoning_tokens": 0},
                "total_tokens": 2
            },
            "user": null, "metadata": {}
        }
    });
    let response = serde_json::to_string(&response).expect("SSE response serializes");
    format!("data: {response}\n\ndata: [DONE]\n\n")
}

fn responses_sse_body(text: &str) -> String {
    let response = serde_json::json!({
        "type": "response.completed", "sequence_number": 1,
        "response": {
            "id": "r-test", "object": "response", "created_at": 1,
            "status": "completed", "background": false, "error": null,
            "incomplete_details": null, "instructions": null, "max_output_tokens": null,
            "model": "m1", "output": [{
                "type": "message", "id": "m-test", "status": "completed", "role": "assistant",
                "content": [{"type": "output_text", "text": text, "annotations": []}]
            }],
            "parallel_tool_calls": true, "previous_response_id": null, "reasoning": {},
            "store": true, "temperature": 1, "text": {"format": {"type": "text"}},
            "tool_choice": "auto", "tools": [], "top_p": 1, "truncation": "disabled",
            "usage": {
                "input_tokens": 1, "input_tokens_details": {"cached_tokens": 0},
                "output_tokens": 1, "output_tokens_details": {"reasoning_tokens": 0},
                "total_tokens": 2
            },
            "user": null, "metadata": {}
        }
    });
    let response = serde_json::to_string(&response).expect("SSE response serializes");
    format!("data: {response}\n\ndata: [DONE]\n\n")
}

fn test_transcript(
    name: &str,
    history: Vec<(String, String)>,
) -> (PathBuf, Arc<StdMutex<TranscriptRecorder>>) {
    let sessions_dir = std::env::temp_dir().join(format!(
        "letcode-tui-runner-interrupt-{name}-{}",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("clock")
            .as_nanos()
    ));
    let mut recorder = TranscriptRecorder::create(&sessions_dir).expect("create transcript");
    recorder
        .record_session_started("m1")
        .expect("record session start");
    // Avoid a title-generation side request in tests that exercise a prompt.
    recorder
        .record_session_title("runner interrupt test")
        .expect("record title");
    for (user, assistant) in history {
        recorder
            .record_user_message(user)
            .expect("record user message");
        recorder
            .record_assistant_message(assistant)
            .expect("record assistant message");
    }
    (sessions_dir, Arc::new(StdMutex::new(recorder)))
}

fn integration_agent(base_url: String, m1_input_limit_tokens: u64) -> Agent<OpenAIConfig> {
    integration_agent_with_tools(base_url, m1_input_limit_tokens, false)
}

fn integration_agent_with_tools(
    base_url: String,
    m1_input_limit_tokens: u64,
    supports_tools: bool,
) -> Agent<OpenAIConfig> {
    let client = Client::with_config(
        OpenAIConfig::new()
            .with_api_base(base_url.clone())
            .with_api_key("test"),
    );
    let mut agent = Agent::new(client, "m1", 4, 4);
    let metadata = |input_limit_tokens: u64| ModelRequestMetadata {
        context_window: Some(input_limit_tokens.saturating_add(1_000)),
        effective_input_limit_tokens: Some(input_limit_tokens),
        max_output_tokens: Some(128),
        supports_tools,
        supports_reasoning: false,
        ..Default::default()
    };
    agent.set_model_catalog(HashMap::from([
        ("m1".into(), metadata(m1_input_limit_tokens)),
        ("m2".into(), metadata(100_000)),
    ]));
    agent.set_compaction_config(CompactionConfig {
        preserve_recent_tokens: Some(0),
    });
    if supports_tools {
        let provider = crate::config::ProviderConfig {
            base_url,
            api_key: "test".into(),
            protocol: crate::config::ApiProtocol::Responses,
            default_model: "m1".into(),
            retry: None,
            models: indexmap::IndexMap::from([(
                "m1".into(),
                crate::config::ModelConfig {
                    display_name: None,
                    protocol: crate::config::ApiProtocol::Responses,
                    context_window: Some(m1_input_limit_tokens.saturating_add(1_000)),
                    effective_input_limit_tokens: Some(m1_input_limit_tokens),
                    max_output_tokens: Some(128),
                    supports_tools: true,
                    supports_reasoning: false,
                    reasoning_effort: None,
                    reasoning_efforts: Vec::new(),
                    reasoning_summary: None,
                    text_verbosity: None,
                    temperature: None,
                    top_p: None,
                    prompt_cache: crate::config::PromptCacheConfig::default(),
                    parallel_tool_calls: false,
                },
            )]),
        };
        let factory = crate::subagent::ExpertRouteFactory::new_with_policies(
            [(
                "explorer".into(),
                Some(crate::config::ModelRoute::new("test", "m1")),
                Vec::new(),
            )],
            &indexmap::IndexMap::from([("test".into(), provider)]),
            &crate::config::RetryConfig::default(),
        )
        .expect("integration expert factory");
        agent.set_primary_route(crate::config::ModelRoute::new("test", "m1"));
        agent.set_subagent_child_factory(Arc::new(factory));
    }
    agent
}

fn test_interrupt() -> InterruptRequest {
    InterruptRequest {
        parent_tool_calls: Vec::new(),
        visible_child_session_id: None,
    }
}

fn turn_started(turn_id: u64) -> TurnStartedEvent {
    TurnStartedEvent {
        turn_id,
        intent: "test".into(),
        directive: "test turn lifecycle".into(),
        validation_reminder: String::new(),
    }
}

struct SessionExecutorHarness {
    ingress: Option<crate::session::SessionEngineIngress>,
    event_rx: mpsc::UnboundedReceiver<SessionTransportEvent>,
    task: JoinHandle<Agent<OpenAIConfig>>,
}

impl SessionExecutorHarness {
    fn ingress(&self) -> &crate::session::SessionEngineIngress {
        self.ingress
            .as_ref()
            .expect("session engine ingress remains connected")
    }

    fn disconnect_ingress(&mut self) {
        self.ingress.take();
    }

    fn send_command(&self, command: SessionEngineCommand) -> Result<()> {
        self.ingress()
            .submit_transitional(command)
            .map_err(Into::into)
    }

    fn send_interrupt(&self, _interrupt: InterruptRequest) -> Result<()> {
        self.ingress().request_interrupt().map_err(Into::into)
    }

    fn shutdown(&self) -> Result<()> {
        self.ingress().shutdown().map_err(Into::into)
    }
}

struct SessionExecutorPollGate {
    ready: oneshot::Sender<()>,
    release: oneshot::Receiver<()>,
}

fn start_session_executor_harness(
    agent: Agent<OpenAIConfig>,
    transcript: Arc<StdMutex<TranscriptRecorder>>,
    sessions_dir: PathBuf,
) -> SessionExecutorHarness {
    start_session_executor_harness_with_poll_gate(agent, transcript, sessions_dir, None)
}

fn start_session_executor_harness_with_poll_gate(
    agent: Agent<OpenAIConfig>,
    transcript: Arc<StdMutex<TranscriptRecorder>>,
    sessions_dir: PathBuf,
    poll_gate: Option<SessionExecutorPollGate>,
) -> SessionExecutorHarness {
    let (engine, ingress, egress) = SessionEngine::new();
    let event_rx = egress.into_receiver();
    let (control_rx, session_transport_tx) = engine.into_session_executor_parts();
    let task = tokio::spawn(test_session_executor_loop(
        agent,
        transcript,
        sessions_dir,
        session_transport_tx,
        control_rx,
        poll_gate,
    ));
    SessionExecutorHarness {
        ingress: Some(ingress),
        event_rx,
        task,
    }
}

async fn start_paused_session_executor_harness(
    agent: Agent<OpenAIConfig>,
    transcript: Arc<StdMutex<TranscriptRecorder>>,
    sessions_dir: PathBuf,
) -> (SessionExecutorHarness, oneshot::Sender<()>) {
    let (ready_tx, ready_rx) = oneshot::channel();
    let (release_tx, release_rx) = oneshot::channel();
    let harness = start_session_executor_harness_with_poll_gate(
        agent,
        transcript,
        sessions_dir,
        Some(SessionExecutorPollGate {
            ready: ready_tx,
            release: release_rx,
        }),
    );
    timeout(SESSION_ENGINE_INTEGRATION_TIMEOUT, ready_rx)
        .await
        .expect("session executor did not reach the control poll gate")
        .expect("session executor dropped the control poll gate");
    (harness, release_tx)
}

async fn test_session_executor_loop(
    mut agent: Agent<OpenAIConfig>,
    transcript: Arc<StdMutex<TranscriptRecorder>>,
    sessions_dir: PathBuf,
    session_transport_tx: mpsc::UnboundedSender<SessionTransportEvent>,
    mut control_rx: mpsc::UnboundedReceiver<SessionEngineControl>,
    poll_gate: Option<SessionExecutorPollGate>,
) -> Agent<OpenAIConfig> {
    let subagent_runtime = SubagentPool::new();
    let mut deferred_commands = VecDeque::new();

    if let Some(SessionExecutorPollGate { ready, release }) = poll_gate {
        ready.send(()).expect("test releases the control poll gate");
        release
            .await
            .expect("test retains the control poll gate release sender");
    }

    loop {
        let Some(command) =
            next_idle_session_command(&mut control_rx, &mut deferred_commands).await
        else {
            break;
        };

        match command {
            SessionEngineCommand::SetExpertModel {
                agent_name,
                model_id,
            } => {
                let _ = session_transport_tx.send(SessionTransportEvent::ExpertModelChanged {
                    agent_name,
                    model_id,
                });
            }
            SessionEngineCommand::Compact => {
                let shutdown = run_manual_compaction(
                    &mut agent,
                    &transcript,
                    &session_transport_tx,
                    &mut control_rx,
                    &mut deferred_commands,
                )
                .await;
                if shutdown {
                    deferred_commands.clear();
                    break;
                }
            }
            SessionEngineCommand::Prompt(prompt) => {
                let _ = session_transport_tx.send(SessionTransportEvent::QueuedPromptAccepted {
                    prompt: prompt.clone(),
                });
                let (runner_event_tx, mut runner_event_rx) = mpsc::unbounded_channel();
                let route_api_key_configured =
                    indexmap::IndexMap::from([(agent.route_display_name(), true)]);
                let runner = AgentRunner::<OpenAIConfig>::with_transcript(
                    runner_event_tx,
                    Arc::clone(&transcript),
                )
                .with_subagent_runtime(
                    subagent_runtime.clone(),
                    sessions_dir.clone(),
                    indexmap::IndexMap::new(),
                    route_api_key_configured,
                    indexmap::IndexMap::new(),
                    String::new(),
                );
                let (interrupted, shutdown) = {
                    let run = runner.run_prompt(&mut agent, prompt);
                    tokio::pin!(run);
                    loop {
                        match crate::session::engine::select_active_session_operation_with_events(
                            &mut control_rx,
                            &mut deferred_commands,
                            run.as_mut(),
                            Some(&mut runner_event_rx),
                        )
                        .await
                        {
                            outcome @ (ActiveSessionOperation::Interrupted
                            | ActiveSessionOperation::Shutdown) => {
                                let interrupt =
                                    derive_interrupt_request(&transcript, &subagent_runtime);
                                let is_shutdown =
                                    matches!(outcome, ActiveSessionOperation::Shutdown);
                                let settle_shutdown = if subagent_runtime.is_running() {
                                    subagent_runtime.cancel_active();
                                    wait_for_subagent_cancel_settle(
                                        &mut control_rx,
                                        &mut deferred_commands,
                                        run.as_mut(),
                                        &subagent_runtime,
                                    )
                                    .await
                                } else {
                                    false
                                };
                                break (Some(interrupt), is_shutdown || settle_shutdown);
                            }
                            ActiveSessionOperation::RunnerEvent(SessionTransportEvent::Done) => {
                                // Runner completion is internal until its future
                                // settles and the executor emits authoritative Done.
                            }
                            ActiveSessionOperation::RunnerEvent(event) => {
                                let _ = session_transport_tx.send(event);
                            }
                            ActiveSessionOperation::Completed(_) => {
                                crate::session::engine::forward_queued_runner_events(
                                    &mut runner_event_rx,
                                    &session_transport_tx,
                                );
                                let _ = session_transport_tx.send(SessionTransportEvent::Done);
                                break (None, false);
                            }
                            ActiveSessionOperation::Command(Some(
                                SessionEngineCommand::Prompt(prompt),
                            )) => {
                                deferred_commands.push_front(SessionEngineCommand::Prompt(prompt));
                                let _ = session_transport_tx.send(
                                    SessionTransportEvent::AssistantDone { message_id: None },
                                );
                                break (None, false);
                            }
                            ActiveSessionOperation::Command(Some(command)) => {
                                deferred_commands.push_front(command);
                                break (None, false);
                            }
                            ActiveSessionOperation::Command(None) => break (None, false),
                        }
                    }
                };
                if let Some(interrupt) = interrupted {
                    subagent_runtime.cancel_active();
                    record_interrupt_transcript(&transcript, &interrupt);
                    let _ = rehydrate_agent_from_transcript(&mut agent, &transcript);
                    send_subagent_interrupted(
                        &session_transport_tx,
                        interrupt.visible_child_session_id,
                    );
                }
                if shutdown {
                    deferred_commands.clear();
                    break;
                }
            }
            SessionEngineCommand::DelegateSubagent { agent_name, task } => {
                let parent_session_id = transcript
                    .lock()
                    .expect("lock transcript")
                    .session_id()
                    .to_string();
                let input = crate::tool::normalize_subagent_input(
                    &format!("agent__{agent_name}"),
                    &serde_json::json!({ "task": task }),
                )
                .expect("delegate input normalizes");
                let invocation = SubagentInvocation {
                    prompt: input.objective.clone(),
                    input,
                    model: None,
                    parent_tool_call_id: None,
                };
                let (interrupted, child_started, interrupted_child_session_id, shutdown) = {
                    let delegate = subagent_runtime.run_named_governed(
                        &agent,
                        &agent_name,
                        invocation,
                        sessions_dir.clone(),
                        parent_session_id,
                        "runner-harness".into(),
                        Some(Arc::clone(&transcript)),
                        Some(crate::session::subagent_event_sender(
                            session_transport_tx.clone(),
                        )),
                    );
                    tokio::pin!(delegate);
                    let mut interrupted = false;
                    let mut child_started = false;
                    let mut interrupted_child_session_id = None;
                    let mut shutdown = false;
                    match select_active_session_operation(
                        &mut control_rx,
                        &mut deferred_commands,
                        delegate.as_mut(),
                    )
                    .await
                    {
                        outcome @ (ActiveSessionOperation::Interrupted
                        | ActiveSessionOperation::Shutdown) => {
                            shutdown = matches!(outcome, ActiveSessionOperation::Shutdown);
                            let interrupt =
                                derive_interrupt_request(&transcript, &subagent_runtime);
                            child_started = subagent_runtime.is_running();
                            interrupted = true;
                            interrupted_child_session_id =
                                interrupt.visible_child_session_id.clone();
                            if child_started {
                                subagent_runtime.cancel_active();
                            }
                            record_interrupt_transcript(&transcript, &interrupt);
                            if child_started {
                                let _ = delegate.await;
                            }
                        }
                        ActiveSessionOperation::Completed(result) => match result {
                            Ok(_) => {
                                let _ = session_transport_tx.send(SessionTransportEvent::Done);
                            }
                            Err(error) => {
                                let _ = session_transport_tx.send(SessionTransportEvent::Error(
                                    ErrorEvent::new(format!("{error:#}")),
                                ));
                                let _ = session_transport_tx.send(SessionTransportEvent::Done);
                            }
                        },
                        ActiveSessionOperation::RunnerEvent(_) => {
                            unreachable!("event-aware selection is not used for delegates")
                        }
                        ActiveSessionOperation::Command(Some(command)) => {
                            deferred_commands.push_front(command);
                        }
                        ActiveSessionOperation::Command(None) => {}
                    }
                    (
                        interrupted,
                        child_started,
                        interrupted_child_session_id,
                        shutdown,
                    )
                };
                if interrupted {
                    if child_started {
                        let _ = rehydrate_agent_from_transcript(&mut agent, &transcript);
                    }
                    send_subagent_interrupted(&session_transport_tx, interrupted_child_session_id);
                }
                if shutdown {
                    deferred_commands.clear();
                    break;
                }
            }
            #[cfg(test)]
            SessionEngineCommand::InspectHistory(reply) => {
                let _ = reply.send(agent.history_for_test().to_vec());
            }
            _ => {}
        }
    }
    agent
}

async fn session_transport_events_until_terminal(
    harness: &mut SessionExecutorHarness,
) -> Vec<SessionTransportEvent> {
    let mut events = Vec::new();
    loop {
        let event = timeout(SESSION_ENGINE_INTEGRATION_TIMEOUT, harness.event_rx.recv())
            .await
            .expect("timed out waiting for session transport event")
            .expect("session transport event channel closed before terminal event");
        let terminal = matches!(
            event,
            SessionTransportEvent::Done | SessionTransportEvent::Interrupted
        );
        events.push(event);
        if terminal {
            return events;
        }
    }
}

async fn assert_no_session_transport_event(
    harness: &mut SessionExecutorHarness,
    window: std::time::Duration,
) {
    match timeout(window, harness.event_rx.recv()).await {
        Err(_) | Ok(None) => {}
        Ok(Some(event)) => panic!("unexpected late session transport event: {event:?}"),
    }
}

async fn session_transport_events_until_handoff(
    harness: &mut SessionExecutorHarness,
) -> Vec<SessionTransportEvent> {
    let mut events = Vec::new();
    loop {
        let event = timeout(SESSION_ENGINE_INTEGRATION_TIMEOUT, harness.event_rx.recv())
            .await
            .expect("timed out waiting for session handoff event")
            .expect("session transport event channel closed before handoff");
        let handoff = matches!(
            event,
            SessionTransportEvent::AssistantDone { message_id: None }
        );
        events.push(event);
        if handoff {
            return events;
        }
    }
}

async fn session_transport_events_until_compaction_committed(
    harness: &mut SessionExecutorHarness,
) -> Vec<SessionTransportEvent> {
    let mut events = Vec::new();
    loop {
        let event = timeout(SESSION_ENGINE_INTEGRATION_TIMEOUT, harness.event_rx.recv())
            .await
            .expect("timed out waiting for compaction event")
            .expect("session transport event channel closed before compaction commit");
        let committed = matches!(event, SessionTransportEvent::CompactionCommitted { .. });
        events.push(event);
        if committed {
            return events;
        }
    }
}

async fn inspect_session_history(harness: &SessionExecutorHarness) -> Vec<HistoryItem> {
    let (reply_tx, reply_rx) = oneshot::channel();
    harness
        .send_command(SessionEngineCommand::InspectHistory(reply_tx))
        .expect("session executor accepts history inspection");
    timeout(SESSION_ENGINE_INTEGRATION_TIMEOUT, reply_rx)
        .await
        .expect("timed out waiting for history inspection")
        .expect("session executor dropped history inspection reply")
}

async fn finish_session_executor_harness(harness: SessionExecutorHarness) -> Agent<OpenAIConfig> {
    let SessionExecutorHarness {
        ingress,
        event_rx,
        task,
    } = harness;
    drop(ingress);
    drop(event_rx);
    timeout(SESSION_ENGINE_INTEGRATION_TIMEOUT, task)
        .await
        .expect("session executor harness should stop")
        .expect("session executor harness task should not panic")
}

fn records(transcript: &Arc<StdMutex<TranscriptRecorder>>) -> Vec<TranscriptRecord> {
    let recorder = transcript.lock().expect("lock transcript");
    read_records(recorder.path()).expect("read transcript")
}

fn project_terminal_runtime(events: &[SessionTransportEvent]) -> TuiRuntime {
    let mut projected = runtime();
    projected.session_turn_active = true;
    projected.state_mut().phase = AppPhase::Running;
    for event in events {
        projected.apply_session_transport_event(event.clone());
    }
    projected
}

fn terminal_count(events: &[SessionTransportEvent]) -> usize {
    events
        .iter()
        .filter(|event| {
            matches!(
                event,
                SessionTransportEvent::Done | SessionTransportEvent::Interrupted
            )
        })
        .count()
}

#[tokio::test]
async fn session_prompt_auto_continue_lifecycle_has_one_public_done() {
    let mut server = spawn_controlled_sse_server(vec![
        ControlledSseResponse::Immediate(responses_sse_tool_call_body(
            "workflow__auto_continue",
            serde_json::json!({ "enabled": true }),
        )),
        ControlledSseResponse::Blocked(responses_sse_tool_call_body(
            "workflow__todos",
            serde_json::json!({
                "items": [{
                    "id": "middle",
                    "content": "blocked middle iteration",
                    "status": "blocked"
                }]
            }),
        )),
        ControlledSseResponse::Immediate(responses_sse_tool_call_body(
            "workflow__auto_continue",
            serde_json::json!({ "enabled": false }),
        )),
        ControlledSseResponse::Immediate(responses_sse_body("prompt completed")),
    ])
    .await;
    let (sessions_dir, transcript) = test_transcript("auto-continue-lifecycle", Vec::new());
    let agent = integration_agent_with_tools(server.base_url.clone(), 32_000, true);
    let mut harness = start_session_executor_harness(agent, Arc::clone(&transcript), sessions_dir);

    harness
        .send_command(SessionEngineCommand::Prompt(UserMessageSubmission::new(
            "auto-continue-lifecycle",
            crate::user_content::UserMessageContent::new("complete the task", Vec::new()),
        )))
        .expect("session executor accepts lifecycle prompt");
    server.expect_request(0).await;
    server.expect_request(1).await;

    tokio::task::yield_now().await;
    let mut events = Vec::new();
    while let Ok(event) = harness.event_rx.try_recv() {
        events.push(event);
    }
    assert!(
        !events
            .iter()
            .any(|event| matches!(event, SessionTransportEvent::Done)),
        "runner-internal completion must not become public Done: {events:?}"
    );

    let blocked_deadline = std::time::Instant::now() + std::time::Duration::from_millis(100);
    while std::time::Instant::now() < blocked_deadline {
        let remaining = blocked_deadline.saturating_duration_since(std::time::Instant::now());
        match timeout(remaining, harness.event_rx.recv()).await {
            Ok(Some(event)) => {
                assert!(
                    !matches!(event, SessionTransportEvent::Done),
                    "public Done appeared while the middle provider iteration was blocked"
                );
                events.push(event);
            }
            Ok(None) | Err(_) => break,
        }
    }

    let midpoint_runtime = project_terminal_runtime(&events);
    assert!(midpoint_runtime.session_turn_active);
    assert_eq!(midpoint_runtime.state().phase, AppPhase::Running);

    server.release.notify_one();
    events.extend(session_transport_events_until_terminal(&mut harness).await);

    assert_eq!(
        events
            .iter()
            .filter(|event| matches!(event, SessionTransportEvent::Done))
            .count(),
        1,
        "the prompt must expose exactly one public Done: {events:?}"
    );
    assert!(
        events
            .iter()
            .any(|event| matches!(event, SessionTransportEvent::AutoContinueChanged(_)))
    );

    let final_runtime = project_terminal_runtime(&events);
    assert!(!final_runtime.session_turn_active);
    assert_eq!(final_runtime.state().phase, AppPhase::Completed);

    let _ = finish_session_executor_harness(harness).await;
    server.finish().await;
}

#[tokio::test]
async fn session_control_fifo_delegate_then_interrupt_before_first_poll_drops_unstarted_child() {
    let mut server = spawn_controlled_sse_server(vec![ControlledSseResponse::Immediate(
        responses_sse_body("reusable child slot"),
    )])
    .await;
    let (sessions_dir, transcript) = test_transcript("fifo-delegate-before-interrupt", Vec::new());
    let agent = integration_agent(server.base_url.clone(), 32_000);
    let (mut harness, release) =
        start_paused_session_executor_harness(agent, Arc::clone(&transcript), sessions_dir.clone())
            .await;

    harness
        .send_command(SessionEngineCommand::DelegateSubagent {
            agent_name: "explorer".into(),
            task: "must not start".into(),
        })
        .expect("session executor accepts delegate command");
    harness
        .send_interrupt(test_interrupt())
        .expect("session executor accepts delegate cancellation");
    release
        .send(())
        .expect("release the session executor after both FIFO controls are queued");

    let interrupted_events = session_transport_events_until_terminal(&mut harness).await;
    assert!(matches!(
        interrupted_events.last(),
        Some(SessionTransportEvent::Interrupted)
    ));
    assert_eq!(
        interrupted_events
            .iter()
            .filter(|event| matches!(event, SessionTransportEvent::Interrupted))
            .count(),
        1
    );
    assert_eq!(terminal_count(&interrupted_events), 1);
    assert!(!interrupted_events.iter().any(|event| matches!(
        event,
        SessionTransportEvent::Done | SessionTransportEvent::ChildSessionEvent { .. }
    )));
    assert!(matches!(
        server.requests.try_recv(),
        Err(mpsc::error::TryRecvError::Empty)
    ));
    let pre_start_records = records(&transcript);
    assert!(
        !pre_start_records
            .iter()
            .any(|record| matches!(record.event, TranscriptEvent::SubagentLifecycle { .. }))
    );
    assert!(!crate::transcript::child_sessions_dir(&sessions_dir).exists());

    harness
        .send_command(SessionEngineCommand::DelegateSubagent {
            agent_name: "explorer".into(),
            task: "prove the child slot is reusable".into(),
        })
        .expect("session executor accepts follow-up delegate");
    server.expect_request(0).await;
    let follow_up_events = session_transport_events_until_terminal(&mut harness).await;
    assert!(matches!(
        follow_up_events.last(),
        Some(SessionTransportEvent::Done)
    ));
    assert_eq!(terminal_count(&follow_up_events), 1);

    let _ = finish_session_executor_harness(harness).await;
    server.finish().await;
}

#[tokio::test]
async fn session_control_fifo_command_then_interrupt_before_first_poll_interrupts_prompt_and_runs_next_command()
 {
    let mut server = spawn_controlled_sse_server(vec![ControlledSseResponse::Immediate(
        responses_sse_body("next prompt completed"),
    )])
    .await;
    let (sessions_dir, transcript) = test_transcript("fifo-prompt-before-interrupt", Vec::new());
    let agent = integration_agent(server.base_url.clone(), 32_000);
    let (mut harness, release) =
        start_paused_session_executor_harness(agent, Arc::clone(&transcript), sessions_dir).await;
    let mut dispatch_runtime = runtime();

    command_dispatch::dispatch_command(
        &mut dispatch_runtime,
        RuntimeCommand::SubmitPrompt(UserMessageSubmission::new(
            "cancelled-before-start",
            crate::user_content::UserMessageContent::new("must not reach the provider", Vec::new()),
        )),
        harness.ingress(),
        true,
    );
    command_dispatch::dispatch_command(
        &mut dispatch_runtime,
        RuntimeCommand::Interrupt,
        harness.ingress(),
        true,
    );
    release
        .send(())
        .expect("release the session executor after both FIFO controls are queued");

    let interrupted_events = session_transport_events_until_terminal(&mut harness).await;
    assert!(matches!(
        interrupted_events.last(),
        Some(SessionTransportEvent::Interrupted)
    ));
    assert_eq!(terminal_count(&interrupted_events), 1);
    assert!(
        !interrupted_events
            .iter()
            .any(|event| matches!(event, SessionTransportEvent::Done))
    );
    assert!(matches!(
        server.requests.try_recv(),
        Err(mpsc::error::TryRecvError::Empty)
    ));

    command_dispatch::dispatch_command(
        &mut dispatch_runtime,
        RuntimeCommand::SubmitPrompt(UserMessageSubmission::new(
            "follow-up",
            crate::user_content::UserMessageContent::new("the next command still runs", Vec::new()),
        )),
        harness.ingress(),
        true,
    );
    server.expect_request(0).await;
    let follow_up_events = session_transport_events_until_terminal(&mut harness).await;
    assert!(matches!(
        follow_up_events.last(),
        Some(SessionTransportEvent::Done)
    ));
    assert!(
        !follow_up_events
            .iter()
            .any(|event| matches!(event, SessionTransportEvent::Interrupted))
    );

    let _ = finish_session_executor_harness(harness).await;
    server.finish().await;
}

#[tokio::test]
async fn session_control_fifo_prompt_then_interrupt_before_first_poll_does_not_interrupt_finalized_turn()
 {
    let mut server = spawn_controlled_sse_server(vec![ControlledSseResponse::Immediate(
        responses_sse_body("this request must not run"),
    )])
    .await;
    let (sessions_dir, transcript) = test_transcript("fifo-prompt-finalized-history", Vec::new());
    {
        let mut recorder = transcript.lock().expect("lock transcript");
        recorder
            .record_user_message("completed request")
            .expect("record prior user message");
        recorder
            .record_turn_started(TurnStartedEvent {
                turn_id: 41,
                intent: "lightweight".into(),
                directive: "complete the prior request".into(),
                validation_reminder: "".into(),
            })
            .expect("record prior turn start");
        recorder
            .record_assistant_message("completed reply")
            .expect("record prior assistant message");
        recorder
            .record_turn_finalized(TurnFinalizedEvent {
                turn_id: 41,
                outcome: "completed".into(),
                tool_call_count: 0,
                continuation_count: 0,
                write_effects: 0,
                validation_effects: 0,
                failed_validation_effects: 0,
                validation_advisory_emitted: false,
            })
            .expect("record prior turn finalization");
    }
    let history_before = records(&transcript);
    let agent = integration_agent(server.base_url.clone(), 32_000);
    let (mut harness, release) =
        start_paused_session_executor_harness(agent, Arc::clone(&transcript), sessions_dir).await;

    harness
        .send_command(SessionEngineCommand::Prompt(UserMessageSubmission::new(
            "cancelled-before-start",
            crate::user_content::UserMessageContent::new("must not reach the provider", Vec::new()),
        )))
        .expect("session executor accepts prompt command");
    harness
        .send_interrupt(test_interrupt())
        .expect("session executor accepts prompt cancellation");
    release
        .send(())
        .expect("release the session executor after both FIFO controls are queued");

    let interrupted_events = session_transport_events_until_terminal(&mut harness).await;
    assert!(matches!(
        interrupted_events.last(),
        Some(SessionTransportEvent::Interrupted)
    ));
    assert_eq!(
        interrupted_events
            .iter()
            .filter(|event| matches!(event, SessionTransportEvent::Interrupted))
            .count(),
        1
    );
    assert_eq!(terminal_count(&interrupted_events), 1);
    assert!(matches!(
        server.requests.try_recv(),
        Err(mpsc::error::TryRecvError::Empty)
    ));

    let history_after = records(&transcript);
    assert_eq!(history_after.len(), history_before.len());
    assert_eq!(
        history_after
            .iter()
            .filter(|record| matches!(record.event, TranscriptEvent::UserMessage { .. }))
            .count(),
        1
    );
    assert_eq!(
        history_after
            .iter()
            .filter(|record| matches!(record.event, TranscriptEvent::TurnStarted(_)))
            .count(),
        1
    );
    assert!(!history_after.iter().any(|record| matches!(
        record.event,
        TranscriptEvent::TurnInterrupted { turn_id: Some(41) }
    )));
    assert!(
        !history_after
            .iter()
            .any(|record| matches!(record.event, TranscriptEvent::TurnInterrupted { .. }))
    );

    let _ = finish_session_executor_harness(harness).await;
    server.abort().await;
}

#[tokio::test]
async fn session_control_fifo_interrupt_then_command_before_first_poll_discards_idle_interrupt() {
    let mut server = spawn_controlled_sse_server(vec![ControlledSseResponse::Immediate(
        responses_sse_body("idle interrupt does not poison this prompt"),
    )])
    .await;
    let (sessions_dir, transcript) = test_transcript("fifo-interrupt-before-prompt", Vec::new());
    let agent = integration_agent(server.base_url.clone(), 32_000);
    let (mut harness, release) =
        start_paused_session_executor_harness(agent, Arc::clone(&transcript), sessions_dir).await;
    let mut dispatch_runtime = runtime();

    command_dispatch::dispatch_command(
        &mut dispatch_runtime,
        RuntimeCommand::Interrupt,
        harness.ingress(),
        true,
    );
    command_dispatch::dispatch_command(
        &mut dispatch_runtime,
        RuntimeCommand::SubmitPrompt(UserMessageSubmission::new(
            "after-idle-interrupt",
            crate::user_content::UserMessageContent::new("this prompt must run", Vec::new()),
        )),
        harness.ingress(),
        true,
    );
    release
        .send(())
        .expect("release the session executor after both FIFO controls are queued");

    server.expect_request(0).await;
    let events = session_transport_events_until_terminal(&mut harness).await;
    assert!(matches!(events.last(), Some(SessionTransportEvent::Done)));
    assert!(
        !events
            .iter()
            .any(|event| matches!(event, SessionTransportEvent::Interrupted))
    );
    assert_eq!(terminal_count(&events), 1);

    let _ = finish_session_executor_harness(harness).await;
    server.finish().await;
}

#[tokio::test]
async fn session_control_fifo_command_then_interrupt_before_first_poll_cancels_manual_compaction_without_provider_request()
 {
    let mut server = spawn_controlled_sse_server(vec![ControlledSseResponse::Immediate(
        responses_sse_body("summary that must not be requested"),
    )])
    .await;
    let (sessions_dir, transcript) = test_transcript(
        "fifo-compact-before-interrupt",
        vec![("older request".into(), "older reply".into())],
    );
    let mut agent = integration_agent(server.base_url.clone(), 32_000);
    rehydrate_agent_from_transcript(&mut agent, &transcript)
        .expect("seed compaction history from transcript");
    let (mut harness, release) =
        start_paused_session_executor_harness(agent, Arc::clone(&transcript), sessions_dir).await;
    let mut dispatch_runtime = runtime();

    command_dispatch::dispatch_command(
        &mut dispatch_runtime,
        RuntimeCommand::Compact,
        harness.ingress(),
        true,
    );
    command_dispatch::dispatch_command(
        &mut dispatch_runtime,
        RuntimeCommand::Interrupt,
        harness.ingress(),
        true,
    );
    release
        .send(())
        .expect("release the session executor after both FIFO controls are queued");

    let events = session_transport_events_until_terminal(&mut harness).await;
    assert!(matches!(events.last(), Some(SessionTransportEvent::Done)));
    assert!(
        events
            .iter()
            .any(|event| matches!(event, SessionTransportEvent::CompactionFailed))
    );
    assert!(
        !events
            .iter()
            .any(|event| matches!(event, SessionTransportEvent::CompactionCommitted { .. }))
    );
    assert!(
        !events
            .iter()
            .any(|event| matches!(event, SessionTransportEvent::Interrupted))
    );
    assert_eq!(terminal_count(&events), 1);
    assert!(matches!(
        server.requests.try_recv(),
        Err(mpsc::error::TryRecvError::Empty)
    ));
    assert!(
        !records(&transcript)
            .iter()
            .any(|record| matches!(record.event, TranscriptEvent::ContextCompaction(_)))
    );

    let _ = finish_session_executor_harness(harness).await;
    server.abort().await;
}

#[tokio::test]
async fn session_manual_compaction_cancel_before_persistence_rehydrates_and_drains_stale_cancel() {
    let mut server = spawn_controlled_sse_server(vec![
        ControlledSseResponse::Immediate(responses_sse_body("completed older turn")),
        ControlledSseResponse::Blocked(responses_sse_body("summary that must not persist")),
        ControlledSseResponse::Immediate(responses_sse_body("follow-up survives")),
    ])
    .await;
    let (sessions_dir, transcript) = test_transcript("manual-cancel", Vec::new());
    let agent = integration_agent(server.base_url.clone(), 32_000);
    let mut harness = start_session_executor_harness(agent, Arc::clone(&transcript), sessions_dir);

    harness
        .send_command(SessionEngineCommand::Prompt(UserMessageSubmission::new(
            "completed-turn",
            crate::user_content::UserMessageContent::new("complete an older turn", Vec::new()),
        )))
        .expect("session executor accepts completed prompt");
    server.expect_request(0).await;
    let completed_events = session_transport_events_until_terminal(&mut harness).await;
    assert!(matches!(
        completed_events.last(),
        Some(SessionTransportEvent::Done)
    ));
    let durable_history = inspect_session_history(&harness).await;
    assert!(
        records(&transcript)
            .iter()
            .any(|record| matches!(record.event, TranscriptEvent::TurnFinalized(_)))
    );

    harness
        .send_command(SessionEngineCommand::Compact)
        .expect("session executor accepts manual compaction");
    server.expect_request(1).await;
    let (queued_history_tx, queued_history_rx) = oneshot::channel();
    harness
        .send_command(SessionEngineCommand::InspectHistory(queued_history_tx))
        .expect("session executor queues a command behind manual compaction");
    harness
        .send_interrupt(test_interrupt())
        .expect("session executor accepts compaction cancellation");
    server.release.notify_one();
    let cancelled_events = session_transport_events_until_terminal(&mut harness).await;

    assert!(
        cancelled_events
            .iter()
            .any(|event| matches!(event, SessionTransportEvent::CompactionStarted))
    );
    assert!(
        cancelled_events
            .iter()
            .any(|event| matches!(event, SessionTransportEvent::CompactionFailed))
    );
    assert!(
        !cancelled_events
            .iter()
            .any(|event| matches!(event, SessionTransportEvent::CompactionCommitted { .. }))
    );
    assert!(
        !cancelled_events
            .iter()
            .any(|event| matches!(event, SessionTransportEvent::Interrupted))
    );
    assert_eq!(terminal_count(&cancelled_events), 1);

    let projected = project_terminal_runtime(&cancelled_events);
    assert!(!projected.session_turn_active);
    assert!(projected.state().pending_question.is_none());
    assert!(projected.state().pending_permission.is_none());
    assert_eq!(projected.state().phase, AppPhase::Completed);
    let queued_history = timeout(SESSION_ENGINE_INTEGRATION_TIMEOUT, queued_history_rx)
        .await
        .expect("queued command is processed after manual compaction")
        .expect("session executor keeps the queued command reply sender");
    assert_eq!(queued_history, durable_history);

    let durable_records = records(&transcript);
    assert!(
        !durable_records
            .iter()
            .any(|record| matches!(record.event, TranscriptEvent::ContextCompaction(_)))
    );
    assert!(
        !durable_records
            .iter()
            .any(|record| matches!(record.event, TranscriptEvent::TurnInterrupted { .. }))
    );

    // A second cancellation arrives while idle. The session executor must consume it
    // before accepting the next operation rather than poisoning that prompt.
    harness
        .send_interrupt(test_interrupt())
        .expect("session executor accepts stale cancellation");
    harness
        .send_command(SessionEngineCommand::Prompt(UserMessageSubmission::new(
            "follow-up",
            crate::user_content::UserMessageContent::new("follow up after compact", Vec::new()),
        )))
        .expect("session executor accepts follow-up prompt");
    server.expect_request(2).await;
    let follow_up_events = session_transport_events_until_terminal(&mut harness).await;
    assert!(matches!(
        follow_up_events.last(),
        Some(SessionTransportEvent::Done)
    ));
    assert!(
        !follow_up_events
            .iter()
            .any(|event| matches!(event, SessionTransportEvent::Interrupted))
    );

    let agent = finish_session_executor_harness(harness).await;
    assert!(agent.history_for_test().iter().any(|item| {
        matches!(item, HistoryItem::AssistantText { text } if text == "follow-up survives")
    }));
    server.finish().await;
}

#[tokio::test]
async fn session_manual_compaction_refreshes_session_token_usage_after_commit() {
    let mut server = spawn_controlled_sse_server(vec![ControlledSseResponse::Immediate(
        responses_sse_body(&compaction_checkpoint("durable summary")),
    )])
    .await;
    // Short one-liners are comparable to the durable summary length, so seed
    // enough historical bulk that a successful compact must reduce tokens.
    let bulky_user = "older request ".repeat(120);
    let bulky_assistant = "older reply ".repeat(120);
    let (sessions_dir, transcript) = test_transcript(
        "manual-token-refresh",
        vec![
            (bulky_user.clone(), bulky_assistant.clone()),
            (bulky_user, bulky_assistant),
        ],
    );
    let mut agent = integration_agent(server.base_url.clone(), 32_000);
    rehydrate_agent_from_transcript(&mut agent, &transcript).expect("seed agent from transcript");
    let before = manual_compaction_session_token_usage(&agent).expect("initial token usage");
    let mut harness = start_session_executor_harness(agent, Arc::clone(&transcript), sessions_dir);

    harness
        .send_command(SessionEngineCommand::Compact)
        .expect("session executor accepts manual compaction");
    server.expect_request(0).await;
    let events = session_transport_events_until_terminal(&mut harness).await;

    let committed_index = events
        .iter()
        .position(|event| matches!(event, SessionTransportEvent::CompactionCommitted { .. }))
        .expect("compaction committed event");
    let (usage_index, usage) = events
        .iter()
        .enumerate()
        .find_map(|(index, event)| match event {
            SessionTransportEvent::SessionTokenUsage(usage) => Some((index, usage)),
            _ => None,
        })
        .expect("session token usage event");
    let context_index = events
        .iter()
        .position(|event| matches!(event, SessionTransportEvent::RuntimeContextUpdated(_)))
        .expect("runtime context event");
    let done_index = events
        .iter()
        .position(|event| matches!(event, SessionTransportEvent::Done))
        .expect("done event");
    assert!(committed_index < usage_index);
    assert!(usage_index < context_index);
    assert!(context_index < done_index);
    assert!(usage.used_tokens < before.used_tokens);
    assert!(usage.input_tokens < before.input_tokens);
    assert_eq!(usage.output_tokens, 0);
    assert_eq!(usage.cached_tokens, 0);
    assert_eq!(usage.cache_report, None);

    let _ = finish_session_executor_harness(harness).await;
    server.finish().await;
}

#[tokio::test]
async fn session_manual_compaction_persistence_wins_over_late_cancel() {
    let mut server = spawn_controlled_sse_server(vec![ControlledSseResponse::Immediate(
        responses_sse_body(&compaction_checkpoint("durable summary")),
    )])
    .await;
    let (sessions_dir, transcript) = test_transcript(
        "manual-persistence-wins",
        vec![("older request".into(), "older reply".into())],
    );
    let mut agent = integration_agent(server.base_url.clone(), 32_000);
    rehydrate_agent_from_transcript(&mut agent, &transcript).expect("seed agent from transcript");
    let mut harness = start_session_executor_harness(agent, Arc::clone(&transcript), sessions_dir);

    harness
        .send_command(SessionEngineCommand::Compact)
        .expect("session executor accepts manual compaction");
    server.expect_request(0).await;
    let mut events = session_transport_events_until_compaction_committed(&mut harness).await;

    assert!(
        records(&transcript)
            .iter()
            .any(|record| matches!(record.event, TranscriptEvent::ContextCompaction(_)))
    );
    harness
        .send_interrupt(test_interrupt())
        .expect("session executor accepts late cancellation");
    let committed_history = inspect_session_history(&harness).await;
    events.extend(session_transport_events_until_terminal(&mut harness).await);

    assert!(
        events
            .iter()
            .any(|event| matches!(event, SessionTransportEvent::CompactionCommitted { .. }))
    );
    assert!(!events.iter().any(|event| matches!(
        event,
        SessionTransportEvent::CompactionFailed | SessionTransportEvent::Error(_)
    )));
    assert!(
        !events
            .iter()
            .any(|event| matches!(event, SessionTransportEvent::Interrupted))
    );
    assert_eq!(terminal_count(&events), 1);
    let committed_index = events
        .iter()
        .position(|event| matches!(event, SessionTransportEvent::CompactionCommitted { .. }))
        .expect("compaction committed event");
    let usage_index = events
        .iter()
        .position(|event| matches!(event, SessionTransportEvent::SessionTokenUsage(_)))
        .expect("rehydrated token usage event");
    let context_index = events
        .iter()
        .position(|event| matches!(event, SessionTransportEvent::RuntimeContextUpdated(_)))
        .expect("runtime context event");
    let done_index = events
        .iter()
        .position(|event| matches!(event, SessionTransportEvent::Done))
        .expect("done event");
    assert!(committed_index < usage_index);
    assert!(usage_index < context_index);
    assert!(context_index < done_index);
    assert!(
        committed_history
            .iter()
            .any(|item| matches!(item, HistoryItem::ContextSummary { .. }))
    );

    let mut restored = integration_agent(server.base_url.clone(), 32_000);
    rehydrate_agent_from_transcript(&mut restored, &transcript)
        .expect("rehydrate committed compaction");
    assert_eq!(restored.history_for_test(), committed_history.as_slice());

    let _ = finish_session_executor_harness(harness).await;
    server.finish().await;
}

#[tokio::test]
async fn session_pressure_compaction_cancel_interrupts_enclosing_prompt_without_stale_cancel() {
    let mut server = spawn_controlled_sse_server(vec![
        ControlledSseResponse::Blocked(responses_sse_body("pressure summary")),
        ControlledSseResponse::Immediate(responses_sse_body("next prompt completed")),
    ])
    .await;
    let history = (0..24)
        .map(|index| {
            (
                format!("older request {index}: {}", "x".repeat(1_000)),
                format!("older reply {index}: {}", "y".repeat(1_000)),
            )
        })
        .collect();
    let (sessions_dir, transcript) = test_transcript("pressure-cancel", history);
    let mut agent = integration_agent(server.base_url.clone(), 8_000);
    rehydrate_agent_from_transcript(&mut agent, &transcript).expect("seed pressure history");
    agent.install_provider_usage_anchor_for_test(TokenUsageEstimate {
        used_tokens: 8_000,
        context_window_tokens: 8_000,
        input_tokens: 8_000,
        output_tokens: 0,
        cached_tokens: 0,
    });
    let mut harness = start_session_executor_harness(agent, Arc::clone(&transcript), sessions_dir);

    harness
        .send_command(SessionEngineCommand::Prompt(UserMessageSubmission::new(
            "pressure-prompt",
            crate::user_content::UserMessageContent::new("current pressure prompt", Vec::new()),
        )))
        .expect("session executor accepts pressure prompt");
    server.expect_request(0).await;
    harness
        .send_interrupt(test_interrupt())
        .expect("session executor accepts prompt cancellation");
    server.release.notify_one();
    let interrupted_events = session_transport_events_until_terminal(&mut harness).await;

    assert!(
        interrupted_events
            .iter()
            .any(|event| matches!(event, SessionTransportEvent::CompactionStarted))
    );
    assert!(matches!(
        interrupted_events.last(),
        Some(SessionTransportEvent::Interrupted)
    ));
    assert!(
        !interrupted_events
            .iter()
            .any(|event| matches!(event, SessionTransportEvent::CompactionCommitted { .. }))
    );
    assert_eq!(terminal_count(&interrupted_events), 1);
    let interrupted_records = records(&transcript);
    assert!(
        !interrupted_records
            .iter()
            .any(|record| matches!(record.event, TranscriptEvent::ContextCompaction(_)))
    );
    assert_eq!(
        interrupted_records
            .iter()
            .filter(|record| matches!(record.event, TranscriptEvent::TurnInterrupted { .. }))
            .count(),
        1
    );

    harness
        .send_interrupt(test_interrupt())
        .expect("session executor accepts stale cancellation");
    let _ = inspect_session_history(&harness).await;
    harness
        .send_command(SessionEngineCommand::Prompt(UserMessageSubmission::new(
            "post-pressure",
            crate::user_content::UserMessageContent::new("prompt after cancellation", Vec::new()),
        )))
        .expect("session executor accepts next prompt");
    server.expect_request(1).await;
    let follow_up_events = session_transport_events_until_terminal(&mut harness).await;
    assert!(matches!(
        follow_up_events.last(),
        Some(SessionTransportEvent::Done)
    ));
    assert!(
        !follow_up_events
            .iter()
            .any(|event| matches!(event, SessionTransportEvent::Interrupted))
    );

    let _ = finish_session_executor_harness(harness).await;
    server.finish().await;
}

#[tokio::test]
async fn queued_prompt_handoff_drops_late_runner_events() {
    let mut server = spawn_controlled_sse_server(vec![
        ControlledSseResponse::Blocked(responses_sse_body("late first response")),
        ControlledSseResponse::Immediate(responses_sse_body("second response")),
    ])
    .await;
    let (sessions_dir, transcript) = test_transcript("queued-prompt-handoff", Vec::new());
    let agent = integration_agent(server.base_url.clone(), 32_000);
    let mut harness = start_session_executor_harness(agent, Arc::clone(&transcript), sessions_dir);

    harness
        .send_command(SessionEngineCommand::Prompt(UserMessageSubmission::new(
            "first-prompt",
            crate::user_content::UserMessageContent::new("first", Vec::new()),
        )))
        .expect("session executor accepts first prompt");
    server.expect_request(0).await;
    harness
        .send_command(SessionEngineCommand::Prompt(UserMessageSubmission::new(
            "second-prompt",
            crate::user_content::UserMessageContent::new("second", Vec::new()),
        )))
        .expect("session executor accepts queued prompt");

    let handoff_events = session_transport_events_until_handoff(&mut harness).await;
    assert!(matches!(
        handoff_events.last(),
        Some(SessionTransportEvent::AssistantDone { .. })
    ));
    assert!(!handoff_events.iter().any(|event| matches!(
        event,
        SessionTransportEvent::Done | SessionTransportEvent::Interrupted
    )));

    server.release.notify_one();
    server.expect_request(1).await;
    let second_events = session_transport_events_until_terminal(&mut harness).await;
    assert!(matches!(
        second_events.last(),
        Some(SessionTransportEvent::Done)
    ));
    assert_eq!(
        second_events
            .iter()
            .filter(|event| matches!(event, SessionTransportEvent::Done))
            .count(),
        1,
        "unexpected events: {second_events:?}"
    );
    assert!(!second_events.iter().any(|event| matches!(
        event,
        SessionTransportEvent::AssistantDelta(delta) if delta.delta.contains("late first response")
    )));
    assert_no_session_transport_event(&mut harness, std::time::Duration::from_millis(100)).await;

    let _ = finish_session_executor_harness(harness).await;
    server.finish().await;
}

#[tokio::test]
async fn session_prompt_ingress_disconnect_interrupts_active_turn() {
    let mut server = spawn_controlled_sse_server(vec![ControlledSseResponse::Blocked(
        responses_sse_body("prompt interrupted by disconnect"),
    )])
    .await;
    let (sessions_dir, transcript) = test_transcript("prompt-ingress-disconnect", Vec::new());
    let agent = integration_agent(server.base_url.clone(), 32_000);
    let mut harness = start_session_executor_harness(agent, Arc::clone(&transcript), sessions_dir);

    harness
        .send_command(SessionEngineCommand::Prompt(UserMessageSubmission::new(
            "disconnect-prompt",
            crate::user_content::UserMessageContent::new("must be interrupted", Vec::new()),
        )))
        .expect("session executor accepts prompt");
    server.expect_request(0).await;
    transcript
        .lock()
        .expect("lock transcript")
        .record_tool_call_started("parent-tool", "shell__exec", serde_json::json!({}))
        .expect("record active parent tool");
    harness
        .send_command(SessionEngineCommand::Prompt(UserMessageSubmission::new(
            "deferred-after-disconnect",
            crate::user_content::UserMessageContent::new(
                "must not execute after disconnect",
                Vec::new(),
            ),
        )))
        .expect("session executor accepts deferred prompt");
    harness.disconnect_ingress();
    server.release.notify_one();

    let events = session_transport_events_until_terminal(&mut harness).await;
    assert!(matches!(
        events.last(),
        Some(SessionTransportEvent::Interrupted)
    ));
    assert_eq!(terminal_count(&events), 1);
    assert_no_session_transport_event(&mut harness, std::time::Duration::from_millis(100)).await;
    assert!(records(&transcript).iter().any(|record| matches!(
        &record.event,
        TranscriptEvent::ToolCallCancelled { call_id, name }
            if call_id == "parent-tool" && name == "shell__exec"
    )));
    assert!(
        records(&transcript)
            .iter()
            .any(|record| matches!(record.event, TranscriptEvent::TurnInterrupted { .. }))
    );

    let agent = finish_session_executor_harness(harness).await;
    assert!(agent.history_for_test().iter().all(|item| match item {
        HistoryItem::UserMessage { content } => {
            content.text != "must not execute after disconnect"
        }
        _ => true,
    }));
    server.finish().await;
}

#[tokio::test]
async fn session_prompt_tool_call_interrupt_settles_active_child() {
    let mut server = spawn_controlled_sse_server(vec![
        ControlledSseResponse::Immediate(responses_sse_tool_call_body(
            "agent__explore",
            serde_json::json!({ "task": "inspect the blocked child" }),
        )),
        ControlledSseResponse::Blocked(responses_sse_body("child must not complete")),
    ])
    .await;
    let (sessions_dir, transcript) = test_transcript("tool-call-child-interrupt", Vec::new());
    let agent = integration_agent_with_tools(server.base_url.clone(), 32_000, true);
    let mut harness =
        start_session_executor_harness(agent, Arc::clone(&transcript), sessions_dir.clone());

    harness
        .send_command(SessionEngineCommand::Prompt(UserMessageSubmission::new(
            "parent-tool-call",
            crate::user_content::UserMessageContent::new("delegate the investigation", Vec::new()),
        )))
        .expect("session executor accepts prompt");
    server.expect_request(0).await;
    server.expect_request(1).await;
    harness
        .send_interrupt(test_interrupt())
        .expect("session executor accepts interrupt");
    let events = session_transport_events_until_terminal(&mut harness).await;

    assert!(
        matches!(events.last(), Some(SessionTransportEvent::Interrupted)),
        "unexpected events: {events:?}"
    );
    assert_eq!(
        events
            .iter()
            .filter(|event| matches!(event, SessionTransportEvent::Interrupted))
            .count(),
        1,
        "unexpected events: {events:?}"
    );
    assert_eq!(terminal_count(&events), 1, "unexpected events: {events:?}");
    assert!(
        !events
            .iter()
            .any(|event| matches!(event, SessionTransportEvent::Done)),
        "an explicit parent interrupt must not expose Done: {events:?}"
    );
    assert_no_session_transport_event(&mut harness, std::time::Duration::from_millis(100)).await;
    assert!(crate::transcript::child_sessions_dir(&sessions_dir).exists());
    assert!(records(&transcript).iter().any(|record| matches!(
        &record.event,
        TranscriptEvent::SubagentResult { agent_name, status, .. }
            if agent_name == "explorer" && status == "cancelled"
    )));
    assert!(
        records(&transcript)
            .iter()
            .any(|record| matches!(record.event, TranscriptEvent::TurnInterrupted { .. }))
    );

    let _ = finish_session_executor_harness(harness).await;
    server.finish().await;
}

#[tokio::test]
async fn session_delegate_ingress_disconnect_interrupts_active_child() {
    let mut server = spawn_controlled_sse_server(vec![ControlledSseResponse::Blocked(
        responses_sse_body("delegate interrupted by disconnect"),
    )])
    .await;
    let (sessions_dir, transcript) = test_transcript("delegate-ingress-disconnect", Vec::new());
    let agent = integration_agent(server.base_url.clone(), 32_000);
    let mut harness =
        start_session_executor_harness(agent, Arc::clone(&transcript), sessions_dir.clone());

    harness
        .send_command(SessionEngineCommand::DelegateSubagent {
            agent_name: "explorer".into(),
            task: "must be interrupted".into(),
        })
        .expect("session executor accepts delegate");
    server.expect_request(0).await;
    harness
        .send_command(SessionEngineCommand::Prompt(UserMessageSubmission::new(
            "deferred-after-disconnect",
            crate::user_content::UserMessageContent::new(
                "must not execute after disconnect",
                Vec::new(),
            ),
        )))
        .expect("session executor accepts deferred prompt");
    harness.disconnect_ingress();
    server.release.notify_one();

    let events = session_transport_events_until_terminal(&mut harness).await;
    assert!(matches!(
        events.last(),
        Some(SessionTransportEvent::Interrupted)
    ));
    assert_eq!(terminal_count(&events), 1);
    assert!(crate::transcript::child_sessions_dir(&sessions_dir).exists());

    let agent = finish_session_executor_harness(harness).await;
    assert!(agent.history_for_test().iter().all(|item| match item {
        HistoryItem::UserMessage { content } => {
            content.text != "must not execute after disconnect"
        }
        _ => true,
    }));
    server.finish().await;
}

#[tokio::test]
async fn session_manual_compaction_ingress_disconnect_rehydrates_without_persistence() {
    let mut server = spawn_controlled_sse_server(vec![
        ControlledSseResponse::Immediate(responses_sse_body("completed older turn")),
        ControlledSseResponse::Blocked(responses_sse_body("summary that must not persist")),
    ])
    .await;
    let (sessions_dir, transcript) = test_transcript("compaction-ingress-disconnect", Vec::new());
    let agent = integration_agent(server.base_url.clone(), 32_000);
    let mut harness = start_session_executor_harness(agent, Arc::clone(&transcript), sessions_dir);

    harness
        .send_command(SessionEngineCommand::Prompt(UserMessageSubmission::new(
            "completed-turn",
            crate::user_content::UserMessageContent::new("complete an older turn", Vec::new()),
        )))
        .expect("session executor accepts completed prompt");
    server.expect_request(0).await;
    let completed_events = session_transport_events_until_terminal(&mut harness).await;
    assert!(matches!(
        completed_events.last(),
        Some(SessionTransportEvent::Done)
    ));
    let durable_history = inspect_session_history(&harness).await;

    harness
        .send_command(SessionEngineCommand::Compact)
        .expect("session executor accepts manual compaction");
    server.expect_request(1).await;
    harness
        .send_command(SessionEngineCommand::Prompt(UserMessageSubmission::new(
            "deferred-after-disconnect",
            crate::user_content::UserMessageContent::new(
                "must not execute after disconnect",
                Vec::new(),
            ),
        )))
        .expect("session executor accepts deferred prompt");
    harness.disconnect_ingress();
    server.release.notify_one();

    let events = session_transport_events_until_terminal(&mut harness).await;
    assert!(
        events
            .iter()
            .any(|event| matches!(event, SessionTransportEvent::CompactionFailed))
    );
    assert!(
        !events
            .iter()
            .any(|event| matches!(event, SessionTransportEvent::CompactionCommitted { .. }))
    );
    assert_eq!(terminal_count(&events), 1);
    assert!(
        !records(&transcript)
            .iter()
            .any(|record| matches!(record.event, TranscriptEvent::ContextCompaction(_)))
    );

    let agent = finish_session_executor_harness(harness).await;
    assert_eq!(agent.history_for_test(), durable_history.as_slice());
    assert!(agent.history_for_test().iter().all(|item| match item {
        HistoryItem::UserMessage { content } => {
            content.text != "must not execute after disconnect"
        }
        _ => true,
    }));
    server.finish().await;
}

#[tokio::test]
async fn session_idle_shutdown_stops_session_executor_loop() {
    let (sessions_dir, transcript) = test_transcript("idle-shutdown", Vec::new());
    let harness = start_session_executor_harness(test_agent(), transcript, sessions_dir);
    harness.shutdown().expect("request shutdown");
    let _ = timeout(SESSION_ENGINE_INTEGRATION_TIMEOUT, harness.task)
        .await
        .expect("session executor stops after idle shutdown")
        .expect("session executor task does not panic");
}

#[tokio::test]
async fn session_active_interrupt_then_shutdown_drops_deferred_prompt() {
    let mut server = spawn_controlled_sse_server(vec![ControlledSseResponse::Blocked(
        responses_sse_body("prompt interrupted by shutdown"),
    )])
    .await;
    let (sessions_dir, transcript) = test_transcript("active-shutdown", Vec::new());
    let agent = integration_agent(server.base_url.clone(), 32_000);
    let mut harness = start_session_executor_harness(agent, Arc::clone(&transcript), sessions_dir);
    harness
        .send_command(SessionEngineCommand::Prompt(UserMessageSubmission::new(
            "first",
            crate::user_content::UserMessageContent::new("first", Vec::new()),
        )))
        .expect("queue active prompt");
    server.expect_request(0).await;
    harness
        .send_command(SessionEngineCommand::Prompt(UserMessageSubmission::new(
            "deferred",
            crate::user_content::UserMessageContent::new("must not execute", Vec::new()),
        )))
        .expect("queue deferred prompt");
    harness
        .send_interrupt(test_interrupt())
        .expect("request interrupt");
    harness.shutdown().expect("request shutdown");
    server.release.notify_one();

    let events = session_transport_events_until_terminal(&mut harness).await;
    assert!(matches!(
        events.last(),
        Some(SessionTransportEvent::Interrupted)
    ));
    let agent = finish_session_executor_harness(harness).await;
    assert!(agent.history_for_test().iter().all(|item| match item {
        HistoryItem::UserMessage { content } => content.text != "must not execute",
        _ => true,
    }));
    server.finish().await;
}

#[tokio::test]
async fn active_operation_forwards_runner_events_but_suppresses_done() {
    let (control_tx, mut control_rx) = mpsc::unbounded_channel();
    let (runner_tx, mut runner_rx) = mpsc::unbounded_channel();
    let mut deferred_commands = VecDeque::new();
    let pending_operation = std::future::pending::<()>();
    tokio::pin!(pending_operation);

    runner_tx
        .send(SessionTransportEvent::Done)
        .expect("runner event sender remains open");
    assert!(matches!(
        crate::session::engine::select_active_session_operation_with_events(
            &mut control_rx,
            &mut deferred_commands,
            pending_operation.as_mut(),
            Some(&mut runner_rx),
        )
        .await,
        ActiveSessionOperation::RunnerEvent(SessionTransportEvent::Done)
    ));

    control_tx
        .send(SessionEngineControl::Interrupt)
        .expect("queue interrupt");
    runner_tx
        .send(SessionTransportEvent::AssistantDone { message_id: None })
        .expect("runner event sender remains open");
    assert!(matches!(
        crate::session::engine::select_active_session_operation_with_events(
            &mut control_rx,
            &mut deferred_commands,
            pending_operation.as_mut(),
            Some(&mut runner_rx),
        )
        .await,
        ActiveSessionOperation::Interrupted
    ));
    assert!(matches!(
        runner_rx.try_recv(),
        Ok(SessionTransportEvent::AssistantDone { .. })
    ));
}

#[tokio::test]
async fn completed_operation_forwards_queued_events_before_done() {
    let (_control_tx, mut control_rx) = mpsc::unbounded_channel();
    let (runner_tx, mut runner_rx) = mpsc::unbounded_channel();
    let (session_tx, mut session_rx) = mpsc::unbounded_channel();
    let mut deferred_commands = VecDeque::new();
    let operation = async {
        runner_tx
            .send(SessionTransportEvent::AssistantDone { message_id: None })
            .expect("runner event receiver remains open");
        runner_tx
            .send(SessionTransportEvent::Error(ErrorEvent::new(
                "request failed",
            )))
            .expect("runner event receiver remains open");
        runner_tx
            .send(SessionTransportEvent::Done)
            .expect("runner event receiver remains open");
    };
    tokio::pin!(operation);

    assert!(matches!(
        crate::session::engine::select_active_session_operation_with_events(
            &mut control_rx,
            &mut deferred_commands,
            operation.as_mut(),
            Some(&mut runner_rx),
        )
        .await,
        ActiveSessionOperation::Completed(())
    ));
    crate::session::engine::forward_queued_runner_events(&mut runner_rx, &session_tx);
    session_tx
        .send(SessionTransportEvent::Done)
        .expect("session event receiver remains open");

    assert!(matches!(
        session_rx.recv().await,
        Some(SessionTransportEvent::AssistantDone { .. })
    ));
    assert!(matches!(
        session_rx.recv().await,
        Some(SessionTransportEvent::Error(_))
    ));
    assert!(matches!(
        session_rx.recv().await,
        Some(SessionTransportEvent::Done)
    ));
    assert!(session_rx.try_recv().is_err());
}

#[test]
fn deferred_settings_use_last_write_wins_per_category() {
    let mut commands = VecDeque::new();
    enqueue_deferred_command(
        &mut commands,
        SessionEngineCommand::SetModel("first".into()),
    );
    enqueue_deferred_command(
        &mut commands,
        SessionEngineCommand::SetPermissionMode(crate::permission::PermissionMode::Safe),
    );
    enqueue_deferred_command(&mut commands, SessionEngineCommand::SetModel("last".into()));
    enqueue_deferred_command(
        &mut commands,
        SessionEngineCommand::SetExpertModel {
            agent_name: "explorer".into(),
            model_id: "explorer-first".into(),
        },
    );
    enqueue_deferred_command(
        &mut commands,
        SessionEngineCommand::SetExpertModel {
            agent_name: "reviewer".into(),
            model_id: "reviewer-only".into(),
        },
    );
    enqueue_deferred_command(
        &mut commands,
        SessionEngineCommand::SetExpertModel {
            agent_name: "explorer".into(),
            model_id: "explorer-last".into(),
        },
    );

    assert!(matches!(
        commands.pop_front(),
        Some(SessionEngineCommand::SetPermissionMode(
            crate::permission::PermissionMode::Safe
        ))
    ));
    assert!(matches!(
        commands.pop_front(),
        Some(SessionEngineCommand::SetModel(model)) if model == "last"
    ));
    assert!(matches!(
        commands.pop_front(),
        Some(SessionEngineCommand::SetExpertModel { agent_name, model_id })
            if agent_name == "reviewer" && model_id == "reviewer-only"
    ));
    assert!(matches!(
        commands.pop_front(),
        Some(SessionEngineCommand::SetExpertModel { agent_name, model_id })
            if agent_name == "explorer" && model_id == "explorer-last"
    ));
    assert!(commands.is_empty());
}

#[test]
fn last_write_wins_does_not_cross_relative_command_barriers() {
    let mut commands = VecDeque::new();
    enqueue_deferred_command(
        &mut commands,
        SessionEngineCommand::SetModel("first".into()),
    );
    enqueue_deferred_command(&mut commands, SessionEngineCommand::ToggleFastMode);
    enqueue_deferred_command(&mut commands, SessionEngineCommand::SetModel("last".into()));

    assert!(matches!(
        commands.pop_front(),
        Some(SessionEngineCommand::SetModel(model)) if model == "first"
    ));
    assert!(matches!(
        commands.pop_front(),
        Some(SessionEngineCommand::ToggleFastMode)
    ));
    assert!(matches!(
        commands.pop_front(),
        Some(SessionEngineCommand::SetModel(model)) if model == "last"
    ));
    assert!(commands.is_empty());
}

#[test]
fn last_write_wins_does_not_cross_prompt_barriers() {
    let mut commands = VecDeque::new();
    enqueue_deferred_command(
        &mut commands,
        SessionEngineCommand::SetModel("first".into()),
    );
    enqueue_deferred_command(
        &mut commands,
        SessionEngineCommand::Prompt(UserMessageSubmission::new(
            "prompt",
            crate::user_content::UserMessageContent::from("prompt"),
        )),
    );
    enqueue_deferred_command(&mut commands, SessionEngineCommand::SetModel("last".into()));

    assert!(matches!(
        commands.pop_front(),
        Some(SessionEngineCommand::SetModel(model)) if model == "first"
    ));
    assert!(matches!(
        commands.pop_front(),
        Some(SessionEngineCommand::Prompt(prompt)) if prompt.id == "prompt"
    ));
    assert!(matches!(
        commands.pop_front(),
        Some(SessionEngineCommand::SetModel(model)) if model == "last"
    ));
    assert!(commands.is_empty());
}

#[test]
fn reasoning_last_write_wins_does_not_cross_toggle_barriers() {
    let mut commands = VecDeque::new();
    enqueue_deferred_command(
        &mut commands,
        SessionEngineCommand::SetReasoningEffort(crate::request_builder::ModelReasoningEffort::Low),
    );
    enqueue_deferred_command(&mut commands, SessionEngineCommand::ToggleFastMode);
    enqueue_deferred_command(
        &mut commands,
        SessionEngineCommand::SetReasoningEffort(
            crate::request_builder::ModelReasoningEffort::High,
        ),
    );

    assert!(matches!(
        commands.pop_front(),
        Some(SessionEngineCommand::SetReasoningEffort(
            crate::request_builder::ModelReasoningEffort::Low
        ))
    ));
    assert!(matches!(
        commands.pop_front(),
        Some(SessionEngineCommand::ToggleFastMode)
    ));
    assert!(matches!(
        commands.pop_front(),
        Some(SessionEngineCommand::SetReasoningEffort(
            crate::request_builder::ModelReasoningEffort::High
        ))
    ));
    assert!(commands.is_empty());
}

#[test]
fn deferred_toggles_retain_each_operation() {
    let mut commands = VecDeque::new();
    enqueue_deferred_command(&mut commands, SessionEngineCommand::ToggleFastMode);
    enqueue_deferred_command(&mut commands, SessionEngineCommand::ToggleFastMode);
    enqueue_deferred_command(
        &mut commands,
        SessionEngineCommand::ToggleMcpServer("docs".into()),
    );
    enqueue_deferred_command(
        &mut commands,
        SessionEngineCommand::ToggleMcpServer("docs".into()),
    );

    assert_eq!(commands.len(), 4);
    assert!(matches!(
        commands.pop_front(),
        Some(SessionEngineCommand::ToggleFastMode)
    ));
    assert!(matches!(
        commands.pop_front(),
        Some(SessionEngineCommand::ToggleFastMode)
    ));
    assert!(matches!(
        commands.pop_front(),
        Some(SessionEngineCommand::ToggleMcpServer(server)) if server == "docs"
    ));
    assert!(matches!(
        commands.pop_front(),
        Some(SessionEngineCommand::ToggleMcpServer(server)) if server == "docs"
    ));
}

#[test]
fn parked_commands_stay_before_later_deferred_commands() {
    let (session_tx, mut session_rx) = mpsc::unbounded_channel();
    let mut parked_commands = VecDeque::new();
    let mut deferred_commands = VecDeque::from([SessionEngineCommand::SetPermissionMode(
        crate::permission::PermissionMode::Auto,
    )]);

    park_active_turn_command(
        &mut parked_commands,
        SessionEngineCommand::SetModel("earlier".into()),
        &session_tx,
    );
    flush_parked_commands(&mut deferred_commands, &mut parked_commands);

    assert!(matches!(
        deferred_commands.pop_front(),
        Some(SessionEngineCommand::SetModel(model)) if model == "earlier"
    ));
    assert!(matches!(
        deferred_commands.pop_front(),
        Some(SessionEngineCommand::SetPermissionMode(
            crate::permission::PermissionMode::Auto
        ))
    ));
    assert!(matches!(
        session_rx.try_recv(),
        Ok(SessionTransportEvent::Notice(notice))
            if notice.message == "Change queued for after the current turn"
    ));
}

#[test]
fn active_turn_reject_is_not_parked_by_engine() {
    let (session_tx, mut session_rx) = mpsc::unbounded_channel();
    let mut parked_commands = VecDeque::new();

    crate::session::engine::handle_active_turn_command(
        SessionEngineCommand::NewSession,
        &mut parked_commands,
        &session_tx,
    );

    assert!(parked_commands.is_empty());
    assert!(matches!(
        session_rx.try_recv(),
        Ok(SessionTransportEvent::Notice(notice)) if notice.message == "Turn still running"
    ));
}

#[tokio::test]
async fn queued_shutdown_interrupts_active_session_operation() {
    let (control_tx, mut control_rx) = mpsc::unbounded_channel();
    let mut deferred_commands = VecDeque::new();
    control_tx
        .send(SessionEngineControl::Shutdown)
        .expect("queue shutdown");
    let pending_operation = std::future::pending::<()>();
    tokio::pin!(pending_operation);

    assert!(matches!(
        select_active_session_operation(
            &mut control_rx,
            &mut deferred_commands,
            pending_operation.as_mut(),
        )
        .await,
        ActiveSessionOperation::Shutdown
    ));
}

#[tokio::test]
async fn queued_interrupt_then_shutdown_returns_shutdown() {
    let (control_tx, mut control_rx) = mpsc::unbounded_channel();
    let mut deferred_commands = VecDeque::new();
    control_tx
        .send(SessionEngineControl::Interrupt)
        .expect("queue interrupt");
    control_tx
        .send(SessionEngineControl::Shutdown)
        .expect("queue shutdown");
    let pending_operation = std::future::pending::<()>();
    tokio::pin!(pending_operation);

    assert!(matches!(
        select_active_session_operation(
            &mut control_rx,
            &mut deferred_commands,
            pending_operation.as_mut(),
        )
        .await,
        ActiveSessionOperation::Shutdown
    ));
}

#[tokio::test]
async fn queued_interrupt_then_shutdown_stops_manual_compaction() {
    let (control_tx, mut control_rx) = mpsc::unbounded_channel();
    let mut deferred_commands = VecDeque::new();
    control_tx
        .send(SessionEngineControl::Interrupt)
        .expect("queue interrupt");
    control_tx
        .send(SessionEngineControl::Shutdown)
        .expect("queue shutdown");
    let pending_operation = std::future::pending::<()>();
    tokio::pin!(pending_operation);

    assert!(matches!(
        select_manual_compaction_operation(
            &mut control_rx,
            &mut deferred_commands,
            pending_operation.as_mut(),
        )
        .await,
        ManualCompactionOperation::Shutdown
    ));
}

#[tokio::test]
async fn queued_interrupt_then_disconnect_returns_shutdown() {
    let (control_tx, mut control_rx) = mpsc::unbounded_channel();
    let mut deferred_commands = VecDeque::new();
    control_tx
        .send(SessionEngineControl::Interrupt)
        .expect("queue interrupt");
    drop(control_tx);
    let pending_operation = std::future::pending::<()>();
    tokio::pin!(pending_operation);

    assert!(matches!(
        select_active_session_operation(
            &mut control_rx,
            &mut deferred_commands,
            pending_operation.as_mut(),
        )
        .await,
        ActiveSessionOperation::Shutdown
    ));
}

#[tokio::test]
async fn idle_shutdown_prevents_deferred_command_dispatch() {
    let (control_tx, mut control_rx) = mpsc::unbounded_channel();
    let mut deferred_commands =
        VecDeque::from([SessionEngineCommand::Prompt(UserMessageSubmission::new(
            "deferred",
            crate::user_content::UserMessageContent::from("deferred"),
        ))]);
    control_tx
        .send(SessionEngineControl::Shutdown)
        .expect("queue shutdown");

    assert!(
        next_idle_session_command(&mut control_rx, &mut deferred_commands)
            .await
            .is_none()
    );
    assert_eq!(deferred_commands.len(), 1);
}

#[tokio::test]
async fn idle_disconnect_prevents_deferred_command_dispatch() {
    let (control_tx, mut control_rx) = mpsc::unbounded_channel();
    let mut deferred_commands =
        VecDeque::from([SessionEngineCommand::Prompt(UserMessageSubmission::new(
            "deferred",
            crate::user_content::UserMessageContent::from("deferred"),
        ))]);
    drop(control_tx);

    assert!(
        next_idle_session_command(&mut control_rx, &mut deferred_commands)
            .await
            .is_none()
    );
    assert_eq!(deferred_commands.len(), 1);
}

#[tokio::test]
async fn disconnected_control_ingress_interrupts_active_session_operation() {
    let (control_tx, mut control_rx) = mpsc::unbounded_channel::<SessionEngineControl>();
    let mut deferred_commands = VecDeque::new();
    drop(control_tx);
    let pending_operation = std::future::pending::<()>();
    tokio::pin!(pending_operation);

    assert!(matches!(
        select_active_session_operation(
            &mut control_rx,
            &mut deferred_commands,
            pending_operation.as_mut(),
        )
        .await,
        ActiveSessionOperation::Shutdown
    ));
}

#[tokio::test]
async fn session_delegate_cancel_prioritizes_interrupt_and_reuses_child_slot() {
    let (race_control_tx, mut race_control_rx) = mpsc::unbounded_channel();
    let mut deferred_commands = VecDeque::new();
    race_control_tx
        .send(SessionEngineControl::Interrupt)
        .expect("queue simultaneous cancellation");
    let ready_delegate = std::future::ready(());
    tokio::pin!(ready_delegate);
    assert!(matches!(
        select_active_session_operation(
            &mut race_control_rx,
            &mut deferred_commands,
            ready_delegate.as_mut(),
        )
        .await,
        ActiveSessionOperation::Interrupted
    ));

    let mut server = spawn_controlled_sse_server(vec![
        ControlledSseResponse::Blocked(responses_sse_body("cancelled child response")),
        ControlledSseResponse::Immediate(responses_sse_body("second child response")),
    ])
    .await;
    let (sessions_dir, transcript) = test_transcript("delegate-cancel", Vec::new());
    let agent = integration_agent(server.base_url.clone(), 32_000);
    let mut harness = start_session_executor_harness(agent, Arc::clone(&transcript), sessions_dir);

    harness
        .send_command(SessionEngineCommand::DelegateSubagent {
            agent_name: "explorer".into(),
            task: "wait for cancellation".into(),
        })
        .expect("session executor accepts first delegate");
    server.expect_request(0).await;
    harness
        .send_interrupt(test_interrupt())
        .expect("session executor accepts delegate cancellation");
    server.release.notify_one();
    let cancelled_events = session_transport_events_until_terminal(&mut harness).await;
    assert_eq!(
        cancelled_events
            .iter()
            .filter(|event| matches!(event, SessionTransportEvent::Interrupted))
            .count(),
        1
    );
    assert_eq!(terminal_count(&cancelled_events), 1);

    harness
        .send_command(SessionEngineCommand::DelegateSubagent {
            agent_name: "explorer".into(),
            task: "prove the slot is reusable".into(),
        })
        .expect("session executor accepts second delegate");
    server.expect_request(1).await;
    let second_events = session_transport_events_until_terminal(&mut harness).await;
    assert!(matches!(
        second_events.last(),
        Some(SessionTransportEvent::Done)
    ));
    assert_eq!(terminal_count(&second_events), 1);

    let _ = finish_session_executor_harness(harness).await;
    server.finish().await;
}

#[tokio::test]
async fn session_delegate_interrupt_notifies_active_child() {
    let mut server = spawn_controlled_sse_server(vec![ControlledSseResponse::Blocked(
        responses_sse_body("cancelled child response"),
    )])
    .await;
    let (sessions_dir, transcript) = test_transcript("delegate-child-interrupt", Vec::new());
    let agent = integration_agent(server.base_url.clone(), 32_000);
    let mut harness = start_session_executor_harness(agent, Arc::clone(&transcript), sessions_dir);

    harness
        .send_command(SessionEngineCommand::DelegateSubagent {
            agent_name: "explorer".into(),
            task: "wait for interruption".into(),
        })
        .expect("session executor accepts delegate");
    server.expect_request(0).await;
    harness
        .send_interrupt(test_interrupt())
        .expect("session executor accepts interrupt");
    server.release.notify_one();

    let events = session_transport_events_until_terminal(&mut harness).await;
    assert!(events.iter().any(|event| matches!(
        event,
        SessionTransportEvent::ChildSessionEvent {
            event: SessionEvent::Interrupted,
            ..
        }
    )));
    assert!(matches!(
        events.last(),
        Some(SessionTransportEvent::Interrupted)
    ));

    let _ = finish_session_executor_harness(harness).await;
    server.finish().await;
}

#[tokio::test]
async fn session_interrupt_records_the_unmatched_started_turn() {
    let mut server = spawn_controlled_sse_server(vec![ControlledSseResponse::Blocked(
        responses_sse_body("cancelled response"),
    )])
    .await;
    let (sessions_dir, transcript) = test_transcript("started-turn-interrupt", Vec::new());
    let agent = integration_agent(server.base_url.clone(), 32_000);
    let mut harness = start_session_executor_harness(agent, Arc::clone(&transcript), sessions_dir);

    harness
        .send_command(SessionEngineCommand::Prompt(UserMessageSubmission::new(
            "started-prompt",
            crate::user_content::UserMessageContent::new("wait for interruption", Vec::new()),
        )))
        .expect("session executor accepts prompt");
    server.expect_request(0).await;
    let started_turn_id = records(&transcript)
        .iter()
        .find_map(|record| match &record.event {
            TranscriptEvent::TurnStarted(event) => Some(event.turn_id),
            _ => None,
        })
        .expect("provider request follows a recorded turn start");
    transcript
        .lock()
        .expect("lock transcript")
        .record_tool_call_started("active-call", "shell__exec", serde_json::json!({}))
        .expect("record active tool call");

    harness
        .send_interrupt(test_interrupt())
        .expect("session executor accepts prompt cancellation");
    let interrupted_events = session_transport_events_until_terminal(&mut harness).await;
    assert!(matches!(
        interrupted_events.last(),
        Some(SessionTransportEvent::Interrupted)
    ));
    assert_eq!(
        interrupted_events
            .iter()
            .filter(|event| matches!(event, SessionTransportEvent::Interrupted))
            .count(),
        1
    );
    assert_eq!(terminal_count(&interrupted_events), 1);

    let interrupted_turn_ids = records(&transcript)
        .iter()
        .filter_map(|record| match &record.event {
            TranscriptEvent::TurnInterrupted {
                turn_id: Some(turn_id),
            } => Some(*turn_id),
            _ => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(interrupted_turn_ids, vec![started_turn_id]);
    assert!(records(&transcript).iter().any(|record| matches!(
        &record.event,
        TranscriptEvent::ToolCallCancelled { call_id, name }
            if call_id == "active-call" && name == "shell__exec"
    )));

    server.release.notify_one();
    let _ = finish_session_executor_harness(harness).await;
    server.finish().await;
}

#[test]
fn session_title_base_sequence_remains_resolvable_for_interrupt_branch_scope() {
    let (_, transcript) = test_transcript("title-base-sequence-interrupt", Vec::new());
    let mut recorder = transcript.lock().expect("lock transcript");
    let title_sequence = read_records(recorder.path())
        .expect("read transcript records")
        .into_iter()
        .find(|record| matches!(record.event, TranscriptEvent::SessionTitle { .. }))
        .expect("session title exists")
        .sequence;

    recorder
        .record_context_branch_created("branch-a", ROOT_CONTEXT_BRANCH_ID, title_sequence, None)
        .expect("title sequence resolves on root branch");
}

#[test]
fn record_interrupt_transcript_scopes_active_turn_to_recorder_branch() {
    let (_, transcript) = test_transcript("branch-scoped-interrupt", Vec::new());
    {
        let mut recorder = transcript.lock().expect("lock transcript");
        let root_leaf_sequence = read_records(recorder.path())
            .expect("read root records")
            .last()
            .expect("session metadata exists")
            .sequence;
        recorder
            .record_context_branch_created(
                "branch-a",
                ROOT_CONTEXT_BRANCH_ID,
                root_leaf_sequence,
                None,
            )
            .expect("create branch A");
        recorder
            .record_context_checkout("branch-a", root_leaf_sequence)
            .expect("checkout branch A");
        recorder.set_current_context_branch_id(Some("branch-a".into()));
        recorder
            .record_turn_started(turn_started(41))
            .expect("start branch A turn");

        recorder
            .record_context_branch_created(
                "branch-b",
                ROOT_CONTEXT_BRANCH_ID,
                root_leaf_sequence,
                None,
            )
            .expect("create branch B");
        recorder
            .record_context_checkout("branch-b", root_leaf_sequence)
            .expect("checkout branch B");
        recorder.set_current_context_branch_id(Some("branch-b".into()));
        recorder
            .record_turn_started(turn_started(42))
            .expect("start branch B turn");

        recorder.set_current_context_branch_id(Some("branch-a".into()));
    }

    record_interrupt_transcript(
        &transcript,
        &InterruptRequest {
            parent_tool_calls: vec![("call-a".into(), "shell__exec".into())],
            visible_child_session_id: None,
        },
    );

    let after = records(&transcript);
    let interruptions = after
        .iter()
        .filter(|record| matches!(record.event, TranscriptEvent::TurnInterrupted { .. }))
        .collect::<Vec<_>>();
    assert_eq!(interruptions.len(), 1);
    assert!(matches!(
        &interruptions[0].event,
        TranscriptEvent::TurnInterrupted { turn_id: Some(41) }
    ));
    assert_eq!(
        interruptions[0].context_branch_id.as_deref(),
        Some("branch-a")
    );
    assert!(after.iter().any(|record| {
        matches!(
            &record.event,
            TranscriptEvent::ToolCallCancelled { call_id, name }
                if call_id == "call-a" && name == "shell__exec"
        ) && record.context_branch_id.as_deref() == Some("branch-a")
    }));
}

#[test]
fn record_interrupt_transcript_pre_start_cancellation_does_not_interrupt_sibling_turn() {
    let (_, transcript) = test_transcript("branch-pre-start-cancel", Vec::new());
    {
        let mut recorder = transcript.lock().expect("lock transcript");
        let root_leaf_sequence = read_records(recorder.path())
            .expect("read root records")
            .last()
            .expect("session metadata exists")
            .sequence;
        recorder
            .record_context_branch_created(
                "branch-a",
                ROOT_CONTEXT_BRANCH_ID,
                root_leaf_sequence,
                None,
            )
            .expect("create branch A");
        recorder
            .record_context_checkout("branch-a", root_leaf_sequence)
            .expect("checkout branch A");

        recorder
            .record_context_branch_created(
                "branch-b",
                ROOT_CONTEXT_BRANCH_ID,
                root_leaf_sequence,
                None,
            )
            .expect("create branch B");
        recorder
            .record_context_checkout("branch-b", root_leaf_sequence)
            .expect("checkout branch B");
        recorder.set_current_context_branch_id(Some("branch-b".into()));
        recorder
            .record_turn_started(turn_started(52))
            .expect("start branch B turn");

        recorder.set_current_context_branch_id(Some("branch-a".into()));
    }

    record_interrupt_transcript(
        &transcript,
        &InterruptRequest {
            parent_tool_calls: vec![("call-a".into(), "shell__exec".into())],
            visible_child_session_id: None,
        },
    );

    let after = records(&transcript);
    assert!(
        !after
            .iter()
            .any(|record| matches!(record.event, TranscriptEvent::TurnInterrupted { .. }))
    );
    assert!(after.iter().any(|record| {
        matches!(
            &record.event,
            TranscriptEvent::ToolCallCancelled { call_id, name }
                if call_id == "call-a" && name == "shell__exec"
        ) && record.context_branch_id.as_deref() == Some("branch-a")
    }));
}

#[test]
fn record_interrupt_transcript_normalizes_root_branch() {
    let (_, transcript) = test_transcript("root-active-turn-interrupt", Vec::new());
    {
        let mut recorder = transcript.lock().expect("lock transcript");
        assert_eq!(recorder.current_context_branch_id(), None);
        recorder
            .record_turn_started(turn_started(61))
            .expect("start root turn");
    }

    record_interrupt_transcript(&transcript, &test_interrupt());

    let interruptions = records(&transcript)
        .into_iter()
        .filter(|record| matches!(record.event, TranscriptEvent::TurnInterrupted { .. }))
        .collect::<Vec<_>>();
    assert_eq!(interruptions.len(), 1);
    assert!(matches!(
        &interruptions[0].event,
        TranscriptEvent::TurnInterrupted { turn_id: Some(61) }
    ));
    assert_eq!(interruptions[0].context_branch_id, None);
}

#[test]
fn record_interrupt_transcript_fails_closed_when_branch_projection_cannot_resolve() {
    let (_, transcript) = test_transcript("unresolvable-branch-interrupt", Vec::new());
    {
        let mut recorder = transcript.lock().expect("lock transcript");
        recorder
            .record_turn_started(turn_started(71))
            .expect("start root turn");
        recorder.set_current_context_branch_id(Some("missing-branch".into()));
    }

    let before = serde_json::to_value(records(&transcript)).expect("serialize transcript");

    record_interrupt_transcript(
        &transcript,
        &InterruptRequest {
            parent_tool_calls: vec![("call-missing".into(), "shell__exec".into())],
            visible_child_session_id: None,
        },
    );

    let after = serde_json::to_value(records(&transcript)).expect("serialize transcript");
    assert_eq!(after, before);
}

#[tokio::test]
async fn error_phase_double_escape_dispatches_to_a_live_session_control_stream() {
    let mut server = spawn_controlled_sse_server(vec![ControlledSseResponse::Blocked(
        responses_sse_body("blocked normal prompt"),
    )])
    .await;
    let (sessions_dir, transcript) = test_transcript("error-phase-escape", Vec::new());
    let agent = integration_agent(server.base_url.clone(), 32_000);
    let mut harness = start_session_executor_harness(agent, Arc::clone(&transcript), sessions_dir);
    harness
        .send_command(SessionEngineCommand::Prompt(UserMessageSubmission::new(
            "blocked-prompt",
            crate::user_content::UserMessageContent::new("hold this prompt", Vec::new()),
        )))
        .expect("session executor accepts blocked prompt");
    server.expect_request(0).await;

    let (_event_tx, event_rx) = mpsc::unbounded_channel();
    let mut runtime = TuiRuntime::new(
        TuiState::default(),
        event_rx,
        vec![AvailableModel::new("m1", "M1")],
        Vec::new(),
        std::env::temp_dir(),
        std::env::temp_dir(),
    );
    runtime.session_turn_active = true;
    runtime.state_mut().phase = AppPhase::Error;
    let escape = KeyEvent::new(KeyCode::Esc, KeyModifiers::NONE);

    let first = runtime
        .handle_input_action(map_key_event(runtime.state(), escape))
        .expect("first escape is accepted");
    assert_eq!(first, None);
    let second = runtime
        .handle_input_action(map_key_event(runtime.state(), escape))
        .expect("second escape is accepted")
        .expect("second escape requests interruption");
    assert!(matches!(second, RuntimeCommand::Interrupt));
    command_dispatch::dispatch_command(&mut runtime, second, harness.ingress(), true);

    server.release.notify_one();
    let events = session_transport_events_until_terminal(&mut harness).await;
    for event in &events {
        runtime.apply_session_transport_event(event.clone());
    }
    assert!(matches!(
        events.last(),
        Some(SessionTransportEvent::Interrupted)
    ));
    assert!(!runtime.session_turn_active);
    assert_eq!(runtime.state().phase, AppPhase::Completed);

    let _ = finish_session_executor_harness(harness).await;
    server.finish().await;
}

#[test]
fn repeated_child_view_projection_does_not_reset_live_child_state() {
    let mut runtime = runtime();
    let event = SessionTransportEvent::ChildSessionViewed {
        parent_session_id: "parent-session".into(),
        child_session_id: "child-session".into(),
        agent_name: "explorer".into(),
        index: 0,
        total: 1,
        pool_ordinal: 1,
        records: vec![],
        runtime_context: event_context("child-session", 1),
    };

    runtime.apply_session_transport_event(event.clone());
    runtime.state_mut().apply_child_session_event(
        "child-session",
        SessionEvent::AssistantDelta(AssistantDeltaEvent::new("live child output")),
    );
    runtime.apply_session_transport_event(event);

    assert!(runtime.state().child_view_has_unpersisted_projection());
}

#[test]
fn assistant_typewriter_reveals_text_across_frames() {
    let mut runtime = runtime();
    runtime.consume_session_transport_event(SessionTransportEvent::AssistantDelta(
        AssistantDeltaEvent::new("abcdefghijklmnopqrst"),
    ));

    assert!(runtime.state().timeline.items().is_empty());
    runtime.advance_assistant_typewriter_by(TUI_FRAME_POLL_INTERVAL);
    let first = match runtime.state().timeline.items().last() {
        Some(TimelineItem::Assistant(message)) => message.text.clone(),
        other => panic!("expected assistant message, got {other:?}"),
    };
    assert!(!first.is_empty());
    assert!(first.len() < 20, "first frame revealed {first:?}");

    runtime.advance_assistant_typewriter_by(TUI_FRAME_POLL_INTERVAL);
    let second = match runtime.state().timeline.items().last() {
        Some(TimelineItem::Assistant(message)) => message.text.clone(),
        other => panic!("expected assistant message, got {other:?}"),
    };
    assert!(second.len() > first.len());
    assert!(second.len() < 20, "second frame revealed {second:?}");
}

#[test]
fn assistant_typewriter_preserves_grapheme_clusters_split_across_deltas() {
    let now = Instant::now();
    let stream = AssistantDeltaStream {
        child_session_id: None,
        parent_tool_call_id: None,
        message_id: None,
    };
    let mut typewriter = AssistantTypewriter::new(stream, None, now);
    typewriter.push("e", now);
    typewriter.push("\u{301}x", now + Duration::from_millis(5));

    assert_eq!(
        typewriter.take_frame(now + Duration::from_millis(10), false),
        "e\u{301}"
    );
    assert_eq!(typewriter.pending, "x");
}

#[test]
fn assistant_typewriter_preserves_zwj_sequence_split_across_deltas() {
    let now = Instant::now();
    let stream = AssistantDeltaStream {
        child_session_id: None,
        parent_tool_call_id: None,
        message_id: None,
    };
    let mut typewriter = AssistantTypewriter::new(stream, None, now);
    typewriter.push("👩", now);
    typewriter.push("‍💻x", now + Duration::from_millis(5));

    assert_eq!(
        typewriter.take_frame(now + Duration::from_millis(10), false),
        "👩‍💻"
    );
    assert_eq!(typewriter.pending, "x");
}

#[test]
fn assistant_typewriter_preserves_grapheme_clusters() {
    let mut runtime = runtime();
    runtime.consume_session_transport_event(SessionTransportEvent::AssistantDelta(
        AssistantDeltaEvent::new("👨‍👩‍👧‍👦e\u{301}x"),
    ));

    runtime.advance_assistant_typewriter_by(Duration::from_millis(17));
    assert!(matches!(
        runtime.state().timeline.items().last(),
        Some(TimelineItem::Assistant(message)) if message.text == "👨‍👩‍👧‍👦"
    ));

    runtime.advance_assistant_typewriter_by(Duration::from_millis(17));
    assert!(matches!(
        runtime.state().timeline.items().last(),
        Some(TimelineItem::Assistant(message)) if message.text == "👨‍👩‍👧‍👦e\u{301}"
    ));
}

#[test]
fn assistant_typewriter_does_not_bank_budget_between_deltas() {
    let now = Instant::now();
    let stream = AssistantDeltaStream {
        child_session_id: None,
        parent_tool_call_id: None,
        message_id: None,
    };
    let mut typewriter = AssistantTypewriter::new(stream, None, now);
    typewriter.push("ab", now);
    assert_eq!(
        typewriter.take_frame(now + Duration::from_millis(100), false),
        "ab"
    );
    assert!(typewriter.pending.is_empty());

    assert_eq!(
        typewriter.take_frame(now + Duration::from_millis(600), false),
        ""
    );
    typewriter.push("cd", now + Duration::from_millis(600));
    assert_eq!(
        typewriter.take_frame(now + Duration::from_millis(610), false),
        ""
    );
    assert_eq!(typewriter.pending, "cd");
}

#[test]
fn assistant_typewriter_keeps_live_stream_state_between_deltas() {
    let mut runtime = runtime();
    runtime.consume_session_transport_event(SessionTransportEvent::AssistantDelta(
        AssistantDeltaEvent::new("ab"),
    ));
    runtime.advance_assistant_typewriter_by(Duration::from_millis(100));

    assert!(runtime.assistant_typewriter.is_some());
    runtime.advance_assistant_typewriter_by(Duration::from_millis(500));
    let before = match runtime.state().timeline.items().last() {
        Some(TimelineItem::Assistant(message)) => message.text.clone(),
        other => panic!("expected assistant message, got {other:?}"),
    };
    runtime.consume_session_transport_event(SessionTransportEvent::AssistantDelta(
        AssistantDeltaEvent::new("cd"),
    ));
    runtime.advance_assistant_typewriter_by(Duration::from_millis(10));

    let after = match runtime.state().timeline.items().last() {
        Some(TimelineItem::Assistant(message)) => message.text.as_str(),
        other => panic!("expected assistant message, got {other:?}"),
    };
    let pending = runtime
        .assistant_typewriter
        .as_ref()
        .map(|typewriter| typewriter.pending.as_str())
        .expect("typewriter remains active");
    assert_eq!(format!("{after}{pending}"), "abcd");
    assert!(
        after.len() <= before.len() + 1,
        "released {after:?} after {before:?}"
    );
}

#[test]
fn assistant_done_waits_for_typewriter_to_drain() {
    let mut runtime = runtime();
    runtime.consume_session_transport_event(SessionTransportEvent::AssistantDelta(
        AssistantDeltaEvent::new("smooth output"),
    ));
    runtime
        .consume_session_transport_event(SessionTransportEvent::AssistantDone { message_id: None });

    assert!(runtime.state().timeline.items().is_empty());
    assert_eq!(runtime.deferred_session_events.len(), 1);

    for _ in 0..8 {
        runtime.advance_assistant_typewriter_by(TUI_FRAME_POLL_INTERVAL);
    }

    assert!(runtime.assistant_typewriter.is_none());
    assert!(runtime.deferred_session_events.is_empty());
    assert!(matches!(
        runtime.state().timeline.items().last(),
        Some(TimelineItem::Assistant(message))
            if message.text == "smooth output" && !message.streaming
    ));
}

#[test]
fn interrupted_preempts_pending_typewriter_output() {
    let mut runtime = runtime();
    runtime.session_turn_active = true;
    runtime.state_mut().phase = AppPhase::Running;
    runtime.consume_session_transport_event(SessionTransportEvent::AssistantDelta(
        AssistantDeltaEvent::new("pending output"),
    ));

    runtime.consume_session_transport_event(SessionTransportEvent::Interrupted);

    assert!(runtime.assistant_typewriter.is_none());
    assert!(runtime.deferred_session_events.is_empty());
    assert!(!runtime.session_turn_active);
    assert_eq!(runtime.state().phase, AppPhase::Completed);
    assert!(matches!(
        runtime.state().timeline.items().last(),
        Some(TimelineItem::Assistant(message)) if message.text == "pending output"
    ));
}

#[test]
fn child_view_preempts_pending_typewriter_output() {
    let mut runtime = runtime();
    runtime.consume_session_transport_event(SessionTransportEvent::AssistantDelta(
        AssistantDeltaEvent::new("parent pending output"),
    ));

    runtime.consume_session_transport_event(SessionTransportEvent::ChildSessionViewed {
        parent_session_id: "parent-session".into(),
        child_session_id: "child-session".into(),
        agent_name: "explorer".into(),
        index: 0,
        total: 1,
        pool_ordinal: 1,
        records: vec![],
        runtime_context: event_context("child-session", 1),
    });

    assert!(runtime.assistant_typewriter.is_none());
    assert!(runtime.deferred_session_events.is_empty());
    assert!(runtime.state().is_read_only_child_view());
    assert_eq!(
        runtime
            .state()
            .child_view_metadata()
            .map(|metadata| metadata.child_session_id),
        Some("child-session".into())
    );
}

#[test]
fn stale_presented_frame_is_not_applied_after_timeline_replacement() {
    let mut state = TuiState::default();
    state.apply_event(SessionEvent::UserMessage(UserMessageEvent::new("old")));
    let mut rendered = state.clone();
    rendered.last_transcript_area = ratatui::layout::Rect::new(1, 2, 30, 12);

    state.timeline = crate::tui::timeline::Timeline::new();
    state.apply_event(SessionEvent::UserMessage(UserMessageEvent::new("new")));
    state.last_transcript_area = ratatui::layout::Rect::default();
    let _ = state.apply_presented_frame(&rendered);

    assert_eq!(state.last_transcript_area, ratatui::layout::Rect::default());
}

struct BlockingTestDrawer {
    started: std::sync::mpsc::Sender<()>,
    release: std::sync::mpsc::Receiver<()>,
}

impl RuntimeDrawer for BlockingTestDrawer {
    fn draw(&mut self, _state: &mut TuiState) -> std::io::Result<()> {
        let _ = self.started.send(());
        self.release
            .recv()
            .map_err(|_| std::io::Error::other("test render release dropped"))
    }
}

#[tokio::test]
async fn double_escape_dispatches_while_render_is_blocked() {
    let mut runtime = runtime();
    runtime.session_turn_active = true;
    runtime.state_mut().phase = AppPhase::Running;
    let presented_state = runtime.state().clone();
    let (mut engine, ingress, _egress) = crate::session::SessionEngine::new();
    let (started_tx, started_rx) = std::sync::mpsc::channel();
    let (release_tx, release_rx) = std::sync::mpsc::channel();
    let mut renderer = AsyncRenderWorker::spawn(BlockingTestDrawer {
        started: started_tx,
        release: release_rx,
    });
    renderer
        .submit("test".into(), runtime.state().clone())
        .expect("submit blocking render");
    started_rx
        .recv_timeout(Duration::from_secs(1))
        .expect("render starts");
    let escape = KeyEvent::new(KeyCode::Esc, KeyModifiers::NONE);

    handle_terminal_event(&mut runtime, &presented_state, Event::Key(escape), &ingress)
        .expect("first escape is handled");
    handle_terminal_event(&mut runtime, &presented_state, Event::Key(escape), &ingress)
        .expect("second escape is handled");

    assert!(matches!(
        engine.recv_control().await,
        Some(SessionEngineControl::Interrupt)
    ));
    release_tx.send(()).expect("release render");
    assert!(
        renderer
            .recv_timeout(Duration::from_secs(1))
            .expect("receive render result")
            .is_some()
    );
}

#[tokio::test]
async fn terminal_event_handler_preserves_double_escape_interrupt_fifo() {
    let mut runtime = runtime();
    runtime.session_turn_active = true;
    runtime.state_mut().phase = AppPhase::Running;
    let (mut engine, ingress, _egress) = crate::session::SessionEngine::new();
    let escape = KeyEvent::new(KeyCode::Esc, KeyModifiers::NONE);

    let presented_state = runtime.state().clone();
    handle_terminal_event(&mut runtime, &presented_state, Event::Key(escape), &ingress)
        .expect("first escape is handled");
    assert!(matches!(
        engine.try_recv_control(),
        Err(tokio::sync::mpsc::error::TryRecvError::Empty)
    ));

    handle_terminal_event(&mut runtime, &presented_state, Event::Key(escape), &ingress)
        .expect("second escape is handled");
    assert!(matches!(
        engine.recv_control().await,
        Some(SessionEngineControl::Interrupt)
    ));
}

#[test]
fn render_feedback_does_not_overwrite_newer_business_state() {
    let mut runtime = runtime();
    runtime.state_mut().set_input("newer input");
    runtime.state_mut().phase = AppPhase::Running;
    let mut rendered = runtime.state().clone();
    rendered.set_input("stale input");
    rendered.phase = AppPhase::Idle;
    rendered.last_transcript_area = ratatui::layout::Rect::new(1, 2, 30, 12);
    rendered.last_transcript_scroll_top = 4;

    runtime.state_mut().apply_render_feedback(&rendered);

    assert_eq!(runtime.state().input_buffer, "newer input");
    assert_eq!(runtime.state().phase, AppPhase::Running);
    assert_eq!(
        runtime.state().last_transcript_area,
        rendered.last_transcript_area
    );
    assert_eq!(runtime.state().last_transcript_scroll_top, 4);
}

#[test]
fn deferred_events_keep_later_deltas_behind_the_barrier() {
    let mut runtime = runtime();
    runtime.consume_session_transport_event(SessionTransportEvent::AssistantDelta(
        AssistantDeltaEvent::new("before"),
    ));
    runtime
        .consume_session_transport_event(SessionTransportEvent::AssistantDone { message_id: None });
    runtime.consume_session_transport_event(SessionTransportEvent::AssistantDelta(
        AssistantDeltaEvent::new("after"),
    ));

    runtime.flush_assistant_typewriter();

    assert_eq!(runtime.state().timeline.items().len(), 2);
    assert!(matches!(
        &runtime.state().timeline.items()[0],
        TimelineItem::Assistant(message) if message.text == "before" && !message.streaming
    ));
    assert!(matches!(
        &runtime.state().timeline.items()[1],
        TimelineItem::Assistant(message) if message.text == "after" && message.streaming
    ));
}

#[test]
fn assistant_typewriter_tracks_delta_arrival_rate() {
    let now = Instant::now();
    let mut typewriter = AssistantTypewriter::new(
        AssistantDeltaStream {
            child_session_id: None,
            parent_tool_call_id: None,
            message_id: None,
        },
        None,
        now,
    );
    typewriter.push("a", now);
    typewriter.push("abcdefghij", now + Duration::from_millis(20));

    assert!(
        typewriter.graphemes_per_second > ASSISTANT_TYPEWRITER_INITIAL_RATE,
        "rate = {}",
        typewriter.graphemes_per_second
    );
    assert!(typewriter.graphemes_per_second <= ASSISTANT_TYPEWRITER_MAX_RATE);
}

#[test]
fn child_delta_flush_before_parent_view_preserves_child_projection() {
    let (tx, rx) = mpsc::unbounded_channel();
    let mut runtime = TuiRuntime::new(
        TuiState::default(),
        rx,
        Vec::new(),
        Vec::new(),
        std::env::temp_dir(),
        std::env::temp_dir(),
    );

    tx.send(SessionTransportEvent::ChildSessionViewed {
        parent_session_id: "parent-session".into(),
        child_session_id: "child-session".into(),
        agent_name: "explorer".into(),
        index: 0,
        total: 1,
        pool_ordinal: 1,
        records: vec![],
        runtime_context: event_context("child-session", 1),
    })
    .expect("queue initial child view");
    runtime.try_drain_session_events();

    tx.send(SessionTransportEvent::ChildSessionEvent {
        child_session_id: "child-session".into(),
        agent_name: Some("explorer".into()),
        parent_tool_call_id: Some("parent-tool".into()),
        event: SessionEvent::AssistantDelta(AssistantDeltaEvent::new("child live delta")),
    })
    .expect("queue child delta");
    tx.send(SessionTransportEvent::ParentSessionViewed {
        session_id: "parent-session".into(),
        branch_id: ROOT_CONTEXT_BRANCH_ID.into(),
        records: vec![],
        model_id: None,
        token_usage: None,
        runtime_context: event_context("parent-session", 1),
    })
    .expect("queue parent view");
    runtime.try_drain_session_events();
    runtime.flush_assistant_typewriter();

    tx.send(SessionTransportEvent::ChildSessionViewed {
        parent_session_id: "parent-session".into(),
        child_session_id: "child-session".into(),
        agent_name: "explorer".into(),
        index: 0,
        total: 1,
        pool_ordinal: 1,
        records: vec![],
        runtime_context: event_context("child-session", 1),
    })
    .expect("queue child view after parent view");
    runtime.try_drain_session_events();
    render_runtime_transcript(&mut runtime);

    assert!(matches!(
        runtime.state().active_timeline().items().last(),
        Some(TimelineItem::Assistant(message)) if message.text == "child live delta"
    ));
}

#[test]
fn child_interrupt_does_not_drop_parent_tool_terminal_state_after_parent_view() {
    let (tx, rx) = mpsc::unbounded_channel();
    let mut runtime = TuiRuntime::new(
        TuiState::default(),
        rx,
        Vec::new(),
        Vec::new(),
        std::env::temp_dir(),
        std::env::temp_dir(),
    );

    runtime.apply_session_transport_event(SessionTransportEvent::ToolStarted(
        ToolStartedEvent::new("parent-tool", "shell__exec", "run command"),
    ));
    tx.send(SessionTransportEvent::ChildSessionViewed {
        parent_session_id: "parent-session".into(),
        child_session_id: "child-session".into(),
        agent_name: "explorer".into(),
        index: 0,
        total: 1,
        pool_ordinal: 1,
        records: vec![],
        runtime_context: event_context("child-session", 1),
    })
    .expect("queue child view");
    tx.send(SessionTransportEvent::ChildSessionEvent {
        child_session_id: "child-session".into(),
        agent_name: Some("explorer".into()),
        parent_tool_call_id: Some("parent-tool".into()),
        event: SessionEvent::Interrupted,
    })
    .expect("queue child interrupt");
    tx.send(SessionTransportEvent::ToolFinished(ToolFinishedEvent::new(
        "parent-tool",
        "shell__exec",
        "command completed",
        ToolOutcome::Success,
    )))
    .expect("queue parent tool finish while child view is active");
    runtime.try_drain_session_events();
    render_runtime_transcript(&mut runtime);

    tx.send(SessionTransportEvent::ParentSessionViewed {
        session_id: "parent-session".into(),
        branch_id: ROOT_CONTEXT_BRANCH_ID.into(),
        records: vec![TranscriptRecord {
            session_id: "parent-session".into(),
            sequence: 1,
            timestamp_ms: 0,
            context_branch_id: None,
            event: TranscriptEvent::ToolCallStarted {
                call_id: "parent-tool".into(),
                name: "shell__exec".into(),
                args: serde_json::json!({ "command": "run command" }),
            },
        }],
        model_id: None,
        token_usage: None,
        runtime_context: event_context("parent-session", 1),
    })
    .expect("queue parent view after child interruption and parent finish");
    runtime.try_drain_session_events();
    render_runtime_transcript(&mut runtime);

    assert!(matches!(
        runtime.state().active_timeline().items().iter().find_map(|item| {
            match item {
                TimelineItem::Tool(tool) if tool.call_id == "parent-tool" => Some(tool),
                _ => None,
            }
        }),
        Some(tool) if tool.status == crate::tui::timeline::ToolExecutionStatus::Succeeded
    ));
}

#[test]
fn parent_view_navigation_with_live_parent_restores_parent_view() {
    let mut runtime = runtime();
    runtime.session_turn_active = true;
    runtime.state_mut().phase = AppPhase::Running;
    runtime.apply_session_transport_event(SessionTransportEvent::ChildSessionViewed {
        parent_session_id: "parent-session".into(),
        child_session_id: "child-session".into(),
        agent_name: "explorer".into(),
        index: 0,
        total: 1,
        pool_ordinal: 1,
        records: vec![],
        runtime_context: event_context("child-session", 1),
    });

    runtime.apply_session_transport_event(SessionTransportEvent::ParentSessionViewed {
        session_id: "parent-session".into(),
        branch_id: ROOT_CONTEXT_BRANCH_ID.into(),
        records: vec![],
        model_id: None,
        token_usage: None,
        runtime_context: event_context("parent-session", 1),
    });
    render_runtime_transcript(&mut runtime);

    assert!(!runtime.state().transcript_view.is_child());
    assert!(
        runtime
            .state()
            .child_timeline_cache_contains("child-session")
    );
}

#[test]
fn background_child_event_updates_cached_child_lifecycle() {
    let mut runtime = runtime();
    runtime.apply_session_transport_event(SessionTransportEvent::ChildSessionEvent {
        child_session_id: "background-child".into(),
        agent_name: Some("fixer".into()),
        parent_tool_call_id: None,
        event: SessionEvent::Error(crate::tui::events::ErrorEvent::new("failed")),
    });

    assert_eq!(runtime.state().phase, AppPhase::Idle);
    assert_eq!(
        runtime.state().cached_child_phase("background-child"),
        Some(AppPhase::Error)
    );
}

#[test]
fn background_child_terminal_preserves_parent_interrupt_confirmation() {
    let mut runtime = runtime();
    runtime.session_turn_active = true;
    runtime.state_mut().phase = AppPhase::Running;
    assert_eq!(
        runtime
            .handle_input_action(InputAction::Interrupt)
            .expect("first interrupt hint succeeds"),
        None
    );

    runtime.apply_session_transport_event(SessionTransportEvent::ChildSessionEvent {
        child_session_id: "background-child".into(),
        agent_name: Some("fixer".into()),
        parent_tool_call_id: None,
        event: SessionEvent::Done,
    });

    assert_eq!(
        runtime
            .handle_input_action(InputAction::Interrupt)
            .expect("child terminal does not reset parent confirmation"),
        Some(RuntimeCommand::Interrupt)
    );
}

#[test]
fn unseen_child_terminal_then_first_view_loads_snapshot_history() {
    let mut runtime = runtime();
    runtime.apply_session_transport_event(SessionTransportEvent::ChildSessionEvent {
        child_session_id: "background-child".into(),
        agent_name: Some("fixer".into()),
        parent_tool_call_id: None,
        event: SessionEvent::Done,
    });

    runtime.apply_session_transport_event(SessionTransportEvent::ChildSessionViewed {
        parent_session_id: "parent-session".into(),
        child_session_id: "background-child".into(),
        agent_name: "fixer".into(),
        index: 0,
        total: 1,
        pool_ordinal: 1,
        records: vec![TranscriptRecord {
            session_id: "background-child".into(),
            sequence: 1,
            timestamp_ms: 0,
            context_branch_id: None,
            event: TranscriptEvent::AssistantMessage {
                content: "persisted child history".into(),
            },
        }],
        runtime_context: event_context("background-child", 1),
    });
    render_runtime_transcript(&mut runtime);

    assert!(matches!(
        runtime.state().active_timeline().items().last(),
        Some(TimelineItem::Assistant(message)) if message.text == "persisted child history"
    ));
    assert_eq!(
        runtime
            .state()
            .child_view_metadata()
            .map(|metadata| metadata.record_count),
        Some(1)
    );
}

#[test]
fn child_context_update_does_not_block_later_snapshot_growth() {
    let mut runtime = runtime();
    runtime.apply_session_transport_event(SessionTransportEvent::ChildSessionViewed {
        parent_session_id: "parent-session".into(),
        child_session_id: "child-session".into(),
        agent_name: "explorer".into(),
        index: 0,
        total: 1,
        pool_ordinal: 1,
        records: vec![],
        runtime_context: event_context("child-session", 1),
    });
    runtime.apply_session_transport_event(SessionTransportEvent::ChildSessionEvent {
        child_session_id: "child-session".into(),
        agent_name: Some("explorer".into()),
        parent_tool_call_id: None,
        event: SessionEvent::RuntimeContextUpdated(
            crate::tui::events::RuntimeContextUpdatedEvent {
                context: event_context("child-session", 2),
                disposition: crate::tui::events::RuntimeContextDisposition::Advance,
            },
        ),
    });

    runtime.apply_session_transport_event(SessionTransportEvent::ChildSessionViewed {
        parent_session_id: "parent-session".into(),
        child_session_id: "child-session".into(),
        agent_name: "explorer".into(),
        index: 0,
        total: 1,
        pool_ordinal: 1,
        records: vec![TranscriptRecord {
            session_id: "child-session".into(),
            sequence: 1,
            timestamp_ms: 0,
            context_branch_id: None,
            event: TranscriptEvent::AssistantMessage {
                content: "snapshot after context update".into(),
            },
        }],
        runtime_context: event_context("child-session", 2),
    });
    render_runtime_transcript(&mut runtime);

    assert!(matches!(
        runtime.state().active_timeline().items().last(),
        Some(TimelineItem::Assistant(message)) if message.text == "snapshot after context update"
    ));
}

#[test]
fn child_terminal_snapshot_refresh_preserves_canonical_projection() {
    let mut runtime = runtime();
    runtime.apply_session_transport_event(SessionTransportEvent::ChildSessionViewed {
        parent_session_id: "parent-session".into(),
        child_session_id: "child-session".into(),
        agent_name: "explorer".into(),
        index: 0,
        total: 1,
        pool_ordinal: 1,
        records: vec![],
        runtime_context: event_context("child-session", 1),
    });
    runtime.apply_session_transport_event(SessionTransportEvent::ChildSessionEvent {
        child_session_id: "child-session".into(),
        agent_name: Some("explorer".into()),
        parent_tool_call_id: None,
        event: SessionEvent::AssistantDelta(AssistantDeltaEvent::new("final live output")),
    });
    runtime.apply_session_transport_event(SessionTransportEvent::ChildSessionEvent {
        child_session_id: "child-session".into(),
        agent_name: Some("explorer".into()),
        parent_tool_call_id: None,
        event: SessionEvent::Done,
    });
    runtime.apply_session_transport_event(SessionTransportEvent::ChildSessionViewed {
        parent_session_id: "parent-session".into(),
        child_session_id: "child-session".into(),
        agent_name: "explorer".into(),
        index: 0,
        total: 1,
        pool_ordinal: 1,
        records: vec![TranscriptRecord {
            session_id: "child-session".into(),
            sequence: 1,
            timestamp_ms: 0,
            context_branch_id: None,
            event: TranscriptEvent::SessionStarted {
                model: "gpt-child".into(),
            },
        }],
        runtime_context: event_context("child-session", 1),
    });
    render_runtime_transcript(&mut runtime);

    assert!(matches!(
        runtime.state().active_timeline().items().last(),
        Some(TimelineItem::Assistant(message)) if message.text == "final live output"
    ));
}

#[test]
fn child_view_snapshot_growth_preserves_unpersisted_live_delta() {
    let (tx, rx) = mpsc::unbounded_channel();
    let mut runtime = TuiRuntime::new(
        TuiState::default(),
        rx,
        Vec::new(),
        Vec::new(),
        std::env::temp_dir(),
        std::env::temp_dir(),
    );

    tx.send(SessionTransportEvent::ChildSessionViewed {
        parent_session_id: "parent-session".into(),
        child_session_id: "child-session".into(),
        agent_name: "explorer".into(),
        index: 0,
        total: 1,
        pool_ordinal: 1,
        records: vec![],
        runtime_context: event_context("child-session", 1),
    })
    .expect("queue initial child view");
    tx.send(SessionTransportEvent::ChildSessionEvent {
        child_session_id: "child-session".into(),
        agent_name: Some("explorer".into()),
        parent_tool_call_id: None,
        event: SessionEvent::AssistantDelta(AssistantDeltaEvent::new("unpersisted live delta")),
    })
    .expect("queue child delta");
    tx.send(SessionTransportEvent::ChildSessionViewed {
        parent_session_id: "parent-session".into(),
        child_session_id: "child-session".into(),
        agent_name: "explorer".into(),
        index: 0,
        total: 1,
        pool_ordinal: 1,
        records: vec![TranscriptRecord {
            session_id: "child-session".into(),
            sequence: 1,
            timestamp_ms: 0,
            context_branch_id: None,
            event: TranscriptEvent::SessionStarted {
                model: "gpt-child".into(),
            },
        }],
        runtime_context: event_context("child-session", 1),
    })
    .expect("queue growing child snapshot");
    runtime.try_drain_session_events();
    runtime.flush_assistant_typewriter();
    render_runtime_transcript(&mut runtime);

    assert!(matches!(
        runtime.state().active_timeline().items().last(),
        Some(TimelineItem::Assistant(message)) if message.text == "unpersisted live delta"
    ));
}

#[test]
fn closed_session_transport_event_stream_terminalizes_running_state() {
    let (event_tx, event_rx) = mpsc::unbounded_channel();
    let mut runtime = TuiRuntime::new(
        TuiState::default(),
        event_rx,
        vec![AvailableModel::new("m1", "M1")],
        Vec::new(),
        std::env::temp_dir(),
        std::env::temp_dir(),
    );
    runtime.session_turn_active = true;
    runtime.state_mut().phase = AppPhase::Running;
    drop(event_tx);

    runtime.try_drain_session_events();

    assert!(!runtime.session_turn_active);
    assert_eq!(runtime.state().phase, AppPhase::Completed);
    assert!(runtime.state().timeline.items().iter().any(|item| matches!(
            item,
            TimelineItem::Error(error) if error.message == "TUI session event stream closed unexpectedly"
        )));
}

#[test]
fn closed_session_transport_event_stream_reports_pending_idle_resume() {
    let (event_tx, event_rx) = mpsc::unbounded_channel();
    let mut runtime = TuiRuntime::new(
        TuiState::default(),
        event_rx,
        vec![AvailableModel::new("m1", "M1")],
        Vec::new(),
        std::env::temp_dir(),
        std::env::temp_dir(),
    );
    runtime.session_resume_pending = true;
    drop(event_tx);

    runtime.try_drain_session_events();

    assert!(!runtime.session_resume_pending);
    assert_eq!(runtime.state().phase, AppPhase::Completed);
    assert!(runtime.state().timeline.items().iter().any(|item| matches!(
            item,
            TimelineItem::Error(error) if error.message == "TUI session event stream closed unexpectedly"
        )));
}

#[test]
fn bounded_session_transport_event_drain_keeps_double_escape_cancel_dispatch_fair() {
    let (session_transport_tx, session_transport_rx) = mpsc::unbounded_channel();
    let mut runtime = TuiRuntime::new(
        TuiState::default(),
        session_transport_rx,
        vec![AvailableModel::new("m1", "M1")],
        Vec::new(),
        std::env::temp_dir(),
        std::env::temp_dir(),
    );
    runtime.session_turn_active = true;
    runtime.state_mut().phase = AppPhase::Error;
    for index in 0..512 {
        session_transport_tx
            .send(SessionTransportEvent::AssistantDelta(
                AssistantDeltaEvent::new(format!("flood-{index}")),
            ))
            .expect("queue session transport flood event");
    }

    runtime.try_drain_session_events();
    assert!(
        runtime.session_transport_rx.try_recv().is_ok(),
        "drain stays bounded"
    );

    let (mut engine, ingress, _egress) = SessionEngine::new();
    let escape = KeyEvent::new(KeyCode::Esc, KeyModifiers::NONE);
    assert_eq!(
        runtime
            .handle_input_action(map_key_event(runtime.state(), escape))
            .expect("first escape is accepted"),
        None
    );
    let command = runtime
        .handle_input_action(map_key_event(runtime.state(), escape))
        .expect("second escape is accepted")
        .expect("second escape requests interruption");
    command_dispatch::dispatch_command(&mut runtime, command, &ingress, true);
    assert!(
        matches!(
            engine.try_recv_control(),
            Ok(SessionEngineControl::Interrupt)
        ),
        "interrupt dispatch makes progress after flood"
    );
}
