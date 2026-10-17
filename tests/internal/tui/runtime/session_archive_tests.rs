use super::TuiRuntime;
use crate::session::archive::ArchiveRunReport;
use crate::tui::TuiState;
use crate::tui::state::ToastKind;
use std::time::{SystemTime, UNIX_EPOCH};
use tokio::sync::mpsc;

fn temp_dir(label: &str) -> std::path::PathBuf {
    std::env::temp_dir().join(format!(
        "letcode-session-archive-{label}-{}",
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("time")
            .as_nanos()
    ))
}

fn runtime() -> TuiRuntime {
    let (_tx, rx) = mpsc::unbounded_channel();
    TuiRuntime::new(
        TuiState::default(),
        rx,
        Vec::new(),
        Vec::new(),
        temp_dir("sessions"),
        temp_dir("preferences"),
    )
}

fn deliver_report(runtime: &mut TuiRuntime, report: ArchiveRunReport) {
    let (tx, rx) = mpsc::unbounded_channel();
    runtime.archive_pass_rx = Some(rx);
    tx.send(Ok(report)).expect("archive report should send");
}

#[test]
fn archive_anomalies_surface_once_as_an_error_toast() {
    let mut runtime = runtime();
    runtime
        .state_mut()
        .set_language(Some(crate::tui::i18n::Language::En));
    deliver_report(
        &mut runtime,
        ArchiveRunReport {
            anomalies: vec!["broken".into()],
            ..ArchiveRunReport::default()
        },
    );

    runtime.poll_session_archive_pass();

    let toast = runtime.state().toast().expect("anomaly toast");
    assert_eq!(toast.kind, ToastKind::Error);
    assert_eq!(
        toast.message,
        "Session archiving keeps failing for broken; see the log for details"
    );
    assert!(runtime.archive_pass_rx.is_none());
}

#[test]
fn archive_anomalies_count_the_remaining_sessions() {
    let mut runtime = runtime();
    runtime
        .state_mut()
        .set_language(Some(crate::tui::i18n::Language::En));
    deliver_report(
        &mut runtime,
        ArchiveRunReport {
            anomalies: vec!["first".into(), "second".into(), "third".into()],
            ..ArchiveRunReport::default()
        },
    );

    runtime.poll_session_archive_pass();

    let toast = runtime.state().toast().expect("anomaly toast");
    assert_eq!(toast.kind, ToastKind::Error);
    assert_eq!(
        toast.message,
        "Session archiving keeps failing for first and 2 more sessions; see the log for details"
    );
}

#[test]
fn a_successful_archive_pass_leaves_the_ui_unchanged() {
    let mut runtime = runtime();
    deliver_report(
        &mut runtime,
        ArchiveRunReport {
            archived_sessions: vec!["idle".into()],
            ..ArchiveRunReport::default()
        },
    );

    runtime.poll_session_archive_pass();

    assert!(runtime.state().toast().is_none());
    assert!(runtime.archive_pass_rx.is_none());
}

#[test]
fn a_failed_archive_pass_stays_out_of_the_ui() {
    let mut runtime = runtime();
    let (tx, rx) = mpsc::unbounded_channel();
    runtime.archive_pass_rx = Some(rx);
    tx.send(Err(anyhow::anyhow!("unreadable sessions directory")))
        .expect("archive failure should send");

    runtime.poll_session_archive_pass();

    assert!(runtime.state().toast().is_none());
    assert!(runtime.archive_pass_rx.is_none());
}
