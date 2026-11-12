use super::*;
use std::sync::atomic::AtomicBool;

fn log(task_id: &str) -> Arc<Mutex<LogState>> {
    OutputSink::new(task_id.to_string()).state()
}

#[test]
fn completing_before_registration_rejects_the_start_notice() {
    let task_id = allocate_id();
    assert!(
        complete(
            &task_id,
            TaskState::Exited { code: Some(0) },
            5,
            0,
            None,
            String::new(),
            &AtomicBool::new(true),
        )
        .is_none()
    );

    let mut cancel = Some(oneshot::channel().0);
    let outcome = register(
        &task_id,
        Some("background-test"),
        "echo hi",
        None,
        log(&task_id),
        &mut cancel,
    );
    assert_eq!(outcome, RegisterOutcome::AlreadyCompleted);
}
