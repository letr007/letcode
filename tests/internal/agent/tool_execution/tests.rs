use super::*;
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};
use tokio::sync::Notify;

fn test_parallel_batch_record(call: &HistoryToolCall) -> ParallelBatchRecord {
    ParallelBatchRecord {
        record: ToolExecutionRecord {
            call_id: call.call_id.clone(),
            tool_name: call.name.clone(),
            arguments: Some(serde_json::json!({})),
            permission_class: crate::permission::ToolPermissionClass::Read,
            status: ToolExecutionStatus::Executed,
            rejection: None,
            output: ToolResult::ok(&call.name, serde_json::json!({})),
            effects: ToolEffects {
                kind: ToolEffectKind::Read,
                primary_path: None,
                edited_paths: Vec::new(),
                command: None,
            },
        },
        completion: ToolSpanCompletion::new(tracing::Span::none()),
    }
}

#[tokio::test]
async fn dropping_parallel_join_future_aborts_the_tool_task() {
    let started = Arc::new(Notify::new());
    let release = Arc::new(Notify::new());
    let completed = Arc::new(AtomicBool::new(false));
    let call = HistoryToolCall {
        call_id: "call-1".into(),
        name: "test-tool".into(),
        arguments_json: "{}".into(),
    };
    let task_started = Arc::clone(&started);
    let task_release = Arc::clone(&release);
    let task_completed = Arc::clone(&completed);
    let task_call = call.clone();
    let task = tokio::spawn(async move {
        task_started.notify_one();
        task_release.notified().await;
        task_completed.store(true, Ordering::Release);
        test_parallel_batch_record(&task_call)
    });
    let parallel = ParallelToolCall {
        call,
        args: serde_json::json!({}),
        permission_class: crate::permission::ToolPermissionClass::Read,
        turn_id: 0,
        task: Some(task),
    };
    started.notified().await;

    let mut join = Box::pin(parallel.join());
    let pending = std::future::poll_fn(|cx| match join.as_mut().poll(cx) {
        std::task::Poll::Pending => std::task::Poll::Ready(()),
        std::task::Poll::Ready(_) => panic!("tool task unexpectedly completed"),
    });
    pending.await;
    drop(join);
    release.notify_one();
    for _ in 0..3 {
        tokio::task::yield_now().await;
    }
    assert!(!completed.load(Ordering::Acquire));
}

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
