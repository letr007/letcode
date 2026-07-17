use super::*;
use crate::agent_event_journal::persist_agent_event;
use crate::context_tree::{ContextNodeStatus, ContextTreeState};
use crate::context_view::{
    ContextBlock, ContextBlockId, ContextBlockKind, ContextBlockSource, ContextViewProjection,
    FoldedOutputMetadata,
};
use crate::protocol_frames::ProtocolFrameItem;
use crate::request_builder::{TestRequestBuilderInput, build_test_request};
use crate::runtime_context::{
    PromptContributorKind, PromptContributorPlaceholder, RuntimeChildSession, RuntimeFrameIdSeed,
    RuntimeFrameKind, RuntimeFrameProvenance, RuntimeSource, SourceSpan,
};
use crate::transcript::transcript_projection::{project_context_tree, project_context_view};
use crate::transcript::{
    ROOT_CONTEXT_BRANCH_ID, TranscriptEvent, TranscriptRecord, TranscriptRecorder, read_records,
    restore_runtime_snapshot, restore_session_history,
};
use async_openai::config::OpenAIConfig;
use async_trait::async_trait;
use serde_json::json;
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpListener;
use tokio::task::JoinHandle;
use tokio::time::{Duration, sleep};

fn observed_units(digests: &[&str]) -> crate::request_builder::LogicalRequestObservation {
    crate::request_builder::LogicalRequestObservation {
        cohort: crate::request_builder::LogicalRequestCohort {
            request_shape_digest: "same-request-shape".into(),
        },
        units: digests
            .iter()
            .map(|digest| crate::request_builder::LogicalRequestUnit {
                category: crate::request_builder::LogicalRequestUnitCategory::User,
                estimated_tokens: 1,
                byte_count: 4,
                digest: (*digest).into(),
            })
            .collect(),
    }
}

#[test]
fn adjacent_lcp_distinguishes_equal_append_middle_and_removed_suffix() {
    let mut tracker = LogicalRequestObservationTracker::default();
    tracker.commit(observed_units(&["a", "b"]));

    let exact = tracker.preview(observed_units(&["a", "b"]));
    assert_eq!(exact.lcp_units, 2);
    assert_eq!(exact.first_breaker, None);

    let append = tracker.preview(observed_units(&["a", "b", "c"]));
    assert_eq!(append.lcp_units, 2);
    assert_eq!(append.first_breaker, None);

    let middle = tracker.preview(observed_units(&["a", "changed", "c"]));
    assert_eq!(middle.lcp_units, 1);
    assert_eq!(
        middle.first_breaker,
        Some(crate::request_builder::LogicalRequestBreaker::CurrentUnit(
            crate::request_builder::LogicalRequestUnitCategory::User
        ))
    );

    let shrink = tracker.preview(observed_units(&["a"]));
    assert_eq!(shrink.lcp_units, 1);
    assert_eq!(
        shrink.first_breaker,
        Some(crate::request_builder::LogicalRequestBreaker::RemovedSuffix)
    );
}

#[test]
fn observation_preview_does_not_advance_the_baseline() {
    let mut tracker = LogicalRequestObservationTracker::default();
    let preview = tracker.preview(observed_units(&["a"]));
    assert!(!preview.cohort_comparable);
    // An unsent preview must leave the next preview without a baseline.
    assert!(!tracker.preview(observed_units(&["a"])).cohort_comparable);
    tracker.commit(observed_units(&["a"]));
    assert!(tracker.preview(observed_units(&["a"])).cohort_comparable);
}

#[test]
fn logical_checkpoint_disable_clears_pending_before_take() {
    let control = LogicalCheckpointControl::disabled_for_test();
    control.set_enabled(true);
    assert_eq!(control.request(), LogicalCheckpointRequestOutcome::Queued);
    control.set_enabled(false);
    assert!(control.take_pending().is_none());
    assert_eq!(control.request(), LogicalCheckpointRequestOutcome::Disabled);
}

#[test]
fn logical_checkpoint_request_and_disable_are_serialized() {
    let control = LogicalCheckpointControl::disabled_for_test();
    control.set_enabled(true);
    let requester = control.clone();
    let disabler = control.clone();
    let barrier = Arc::new(std::sync::Barrier::new(3));
    let request_barrier = Arc::clone(&barrier);
    let disable_barrier = Arc::clone(&barrier);
    let request = std::thread::spawn(move || {
        request_barrier.wait();
        requester.request()
    });
    let disable = std::thread::spawn(move || {
        disable_barrier.wait();
        disabler.set_enabled(false);
    });
    barrier.wait();
    let _ = request.join().expect("request thread");
    disable.join().expect("disable thread");
    assert!(control.take_pending().is_none());
    assert_eq!(control.request(), LogicalCheckpointRequestOutcome::Disabled);
}

#[test]
fn logical_checkpoint_run_guard_clears_pending_and_in_flight_requests_on_drop() {
    let control = LogicalCheckpointControl::disabled_for_test();
    control.set_enabled(true);

    let pending_run = control.begin_run();
    assert_eq!(control.request(), LogicalCheckpointRequestOutcome::Queued);
    drop(pending_run);
    assert_eq!(control.request(), LogicalCheckpointRequestOutcome::Queued);
    control.clear();

    let in_flight_run = control.begin_run();
    assert_eq!(control.request(), LogicalCheckpointRequestOutcome::Queued);
    assert!(control.take_pending().is_some());
    drop(in_flight_run);
    assert_eq!(control.request(), LogicalCheckpointRequestOutcome::Queued);
}

#[test]
fn logical_checkpoint_lease_does_not_clear_a_later_run_request() {
    let control = LogicalCheckpointControl::disabled_for_test();
    control.set_enabled(true);
    let first_run = control.begin_run();
    assert_eq!(control.request(), LogicalCheckpointRequestOutcome::Queued);
    let first_lease = control.take_pending().expect("first lease");
    control.clear_lease(first_lease);
    drop(first_run);

    let second_run = control.begin_run();
    assert_eq!(control.request(), LogicalCheckpointRequestOutcome::Queued);
    control.clear_lease(first_lease);
    assert_eq!(
        control.request(),
        LogicalCheckpointRequestOutcome::AlreadyQueued
    );
    drop(second_run);
}

#[test]
fn automatic_checkpoint_request_requires_active_run_and_never_displaces_manual() {
    let control = LogicalCheckpointControl::disabled_for_test();
    control.set_config(LogicalCheckpointConfig {
        enabled: true,
        automatic: true,
        ..Default::default()
    });
    assert_eq!(
        control.request_automatic(1),
        LogicalCheckpointRequestOutcome::Disabled
    );
    let run = control.begin_run();
    assert_eq!(
        control.request_automatic(1),
        LogicalCheckpointRequestOutcome::Queued
    );
    assert_eq!(control.request(), LogicalCheckpointRequestOutcome::Queued);
    let lease = control.take_pending().expect("manual lease");
    assert_eq!(lease.ownership, LogicalCheckpointRequestOwner::Manual);
    assert_eq!(
        control.request_automatic(2),
        LogicalCheckpointRequestOutcome::AlreadyQueued
    );
    control.clear_lease(lease);
    drop(run);
}

#[test]
fn automatic_checkpoint_lease_identity_prevents_same_run_aba_clear() {
    let control = LogicalCheckpointControl::disabled_for_test();
    control.set_config(LogicalCheckpointConfig {
        enabled: true,
        automatic: true,
        ..Default::default()
    });
    let run = control.begin_run();
    assert_eq!(
        control.request_automatic(1),
        LogicalCheckpointRequestOutcome::Queued
    );
    let first = control.take_pending().expect("first lease");
    control.clear_lease(first);
    assert_eq!(
        control.request_automatic(2),
        LogicalCheckpointRequestOutcome::Queued
    );
    let second = control.take_pending().expect("second lease");
    assert_ne!(first.request_id, second.request_id);
    control.clear_lease(first);
    assert_eq!(
        control.request(),
        LogicalCheckpointRequestOutcome::AlreadyQueued
    );
    control.clear_lease(second);
    drop(run);
}

#[test]
fn automatic_scheduler_state_is_ephemeral_and_resets_for_the_next_turn() {
    let mut state = AutomaticCheckpointSchedulerState::default();
    assert!(state.armed);
    assert!(!state.view().boundary_available);
    assert_eq!(state.begin_complete_boundary(), 1);
    state.mark_attempted(1);
    assert!(state.view().boundary_attempted);
    state.rearm();
    state.mark_committed(LogicalCheckpointRequestOwner::Automatic { boundary_id: 1 });
    state.suppress();
    state.reset_for_turn_end();
    assert_eq!(state, AutomaticCheckpointSchedulerState::default());
}

#[test]
fn automatic_scheduler_new_boundary_allows_trigger_after_disarmed_boundary() {
    let mut state = AutomaticCheckpointSchedulerState::default();
    state.begin_complete_boundary();
    state.mark_attempted(1);
    state.rearm();
    assert!(state.view().boundary_attempted);
    assert_eq!(state.begin_complete_boundary(), 2);
    assert!(state.view().boundary_available);
    assert!(!state.view().boundary_attempted);
    assert!(!state.view().boundary_consumed);
}

#[test]
fn committed_checkpoint_consumes_and_disarms_but_only_automatic_counts() {
    let mut state = AutomaticCheckpointSchedulerState::default();
    state.begin_complete_boundary();
    state.mark_committed(LogicalCheckpointRequestOwner::Manual);
    assert!(!state.armed);
    assert!(state.view().boundary_consumed);
    assert_eq!(state.commits, 0);

    state.begin_complete_boundary();
    state.mark_committed(LogicalCheckpointRequestOwner::Automatic { boundary_id: 2 });
    assert_eq!(state.commits, 1);
}

#[test]
fn logical_checkpoint_config_preserves_nondefault_automatic_policy_for_children() {
    let mut parent = test_agent();
    let config = LogicalCheckpointConfig {
        enabled: true,
        automatic: true,
        max_automatic_per_turn: 3,
    };
    parent.set_logical_checkpoint_config(config);
    assert_eq!(
        parent.automatic_checkpoint_policy,
        automatic_checkpoint::AutoCheckpointPolicy::from_config(config)
    );

    let child = AgentFactory::create_child(&parent, &AgentTemplate::explorer());
    assert_eq!(
        child.automatic_checkpoint_policy,
        parent.automatic_checkpoint_policy
    );
}

fn checkpoint_test_agent() -> (Agent<OpenAIConfig>, Vec<PromptMessage>) {
    let mut agent = test_agent();
    agent.set_model_catalog(HashMap::from([(
        "m1".into(),
        ModelRequestMetadata {
            context_window: Some(32_000),
            max_output_tokens: Some(2_000),
            supports_tools: true,
            ..Default::default()
        },
    )]));
    agent.set_logical_checkpoint_config(LogicalCheckpointConfig {
        enabled: true,
        ..Default::default()
    });
    let prelude = agent.prepare_turn_prelude("continue the active turn");
    agent.history = vec![HistoryItem::user("current request")];
    agent.turn.current_turn_start_index = Some(0);
    agent.runtime_snapshot.current_turn_id = Some(agent.turn.turn_id);
    agent.runtime_snapshot.current_segment_id = Some(1);
    agent.runtime_snapshot.leaf_sequence = Some(9);
    agent.runtime_snapshot.latest_model = Some(agent.model.clone());
    (agent, prelude)
}

fn prepared_checkpoint_for(
    agent: &Agent<OpenAIConfig>,
) -> crate::transcript::PreparedLogicalCheckpoint {
    prepared_checkpoint_for_lineage(agent, "checkpoint-test", None)
}

fn prepared_checkpoint_for_lineage(
    agent: &Agent<OpenAIConfig>,
    checkpoint_id: &str,
    previous_checkpoint_id: Option<&str>,
) -> crate::transcript::PreparedLogicalCheckpoint {
    let previous_segment_id = agent
        .runtime_snapshot
        .current_segment_id
        .expect("checkpoint fixture has a live segment");
    let boundary_sequence = agent
        .runtime_snapshot
        .leaf_sequence
        .expect("checkpoint fixture has a journal frontier");
    let segment_id = previous_segment_id + 1;
    let leaf = boundary_sequence + 1;
    let event = LogicalCheckpointEventV1 {
        schema_version: 1,
        checkpoint_id: checkpoint_id.into(),
        turn_id: agent.turn.turn_id,
        previous_segment_id,
        segment_id,
        previous_checkpoint_id: previous_checkpoint_id.map(str::to_string),
        boundary_sequence,
        context_scope_revision: agent.runtime_snapshot.context_scope_revision,
        covered_source_spans: Vec::new(),
        retained_items: Vec::new(),
    };
    let summary = crate::transcript::render_checkpoint_v1(&event).expect("summary renders");
    let continuation = crate::transcript::render_checkpoint_continuation_v1(&event);
    let mut frames = vec![
        crate::protocol_frames::ProtocolFrame::derived(ProtocolFrameItem::ContextSummary {
            text: summary,
        }),
        crate::protocol_frames::ProtocolFrame::derived(ProtocolFrameItem::InternalContinuation {
            text: continuation,
        }),
    ];
    for (index, frame) in frames.iter_mut().enumerate() {
        frame.history_index = index;
    }
    for (frame, source_id) in frames.iter_mut().zip([
        format!("{checkpoint_id}:summary"),
        format!("{checkpoint_id}:continuation"),
    ]) {
        frame.source_provenance = Some(
            RuntimeFrameProvenance::new(RuntimeSource::Transcript)
                .with_source_id(&source_id)
                .with_span(SourceSpan::new(leaf, leaf).expect("valid source span")),
        );
    }
    let mut snapshot =
        RuntimeSnapshot::new(agent.runtime_snapshot.active_context.branch_id.clone());
    snapshot.current_turn_id = Some(agent.turn.turn_id);
    snapshot.current_segment_id = Some(segment_id);
    snapshot.leaf_sequence = Some(leaf);
    snapshot.latest_model = Some(agent.model.clone());
    snapshot.frames = frames
        .iter()
        .enumerate()
        .map(|(index, frame)| {
            let mut runtime = runtime_frame_from_protocol_frame(frame, index as u32);
            runtime.provenance = frame
                .source_provenance
                .clone()
                .expect("checkpoint suffix has provenance");
            runtime
        })
        .collect();
    for (frame, runtime) in frames.iter_mut().zip(&snapshot.frames) {
        frame.runtime_frame_id = Some(runtime.id);
    }
    crate::transcript::PreparedLogicalCheckpoint {
        expected_journal_frontier: boundary_sequence,
        expected_branch_id: agent.runtime_snapshot.active_context.branch_id.clone(),
        event,
        projected_snapshot: snapshot,
        projected_protocol_frames: frames,
        projected_workflow: Some(crate::transcript::CheckpointWorkflowProjection {
            todos: agent.turn.workflow.todos.clone(),
            auto_continue: agent.turn.workflow.auto_continue.clone(),
        }),
    }
}

#[tokio::test]
async fn logical_checkpoint_missing_provider_rejects_and_clears_its_request() {
    let (mut agent, prelude) = checkpoint_test_agent();
    let run = agent.logical_checkpoint_control.begin_run();
    assert_eq!(
        agent.request_logical_checkpoint(),
        LogicalCheckpointRequestOutcome::Queued
    );

    let error = logical_checkpoint::commit_pending_at_batch_boundary(
        &mut agent,
        ApiProtocol::Responses,
        &prelude,
        0,
        &mut |_| std::future::ready(Ok(())),
    )
    .await
    .expect_err("missing provider must reject the pending checkpoint");

    assert!(error.to_string().contains("without a candidate provider"));
    assert_eq!(
        agent.request_logical_checkpoint(),
        LogicalCheckpointRequestOutcome::Queued
    );
    drop(run);
}

#[tokio::test]
async fn logical_checkpoint_callback_failure_preserves_live_envelope_and_clears_request() {
    let (mut agent, prelude) = checkpoint_test_agent();
    agent.turn.automatic_checkpoint.begin_complete_boundary();
    let candidate = prepared_checkpoint_for(&agent);
    agent.set_logical_checkpoint_candidate_provider(Arc::new(move || Ok(candidate.clone())));
    let history_before = agent.history.clone();
    let frames_before = agent.protocol_frames.clone();
    let snapshot_before = agent.runtime_snapshot.clone();
    let workflow_before = agent.turn.workflow.clone();
    let start_before = agent.turn.current_turn_start_index;
    let run = agent.logical_checkpoint_control.begin_run();
    assert_eq!(
        agent.request_logical_checkpoint(),
        LogicalCheckpointRequestOutcome::Queued
    );

    let error = logical_checkpoint::commit_pending_at_batch_boundary(
        &mut agent,
        ApiProtocol::Responses,
        &prelude,
        0,
        &mut |_| std::future::ready(Err(anyhow!("durable callback failed"))),
    )
    .await
    .expect_err("durable acknowledgement failure must reject before installation");

    assert!(error.to_string().contains("durable callback failed"));
    assert_eq!(agent.history, history_before);
    assert_eq!(agent.protocol_frames, frames_before);
    assert_eq!(agent.runtime_snapshot, snapshot_before);
    assert_eq!(agent.turn.workflow, workflow_before);
    assert_eq!(agent.turn.current_turn_start_index, start_before);
    assert!(agent.turn.automatic_checkpoint.armed);
    assert_eq!(agent.turn.automatic_checkpoint.commits, 0);
    assert_eq!(
        agent.request_logical_checkpoint(),
        LogicalCheckpointRequestOutcome::Queued
    );
    drop(run);
}

#[tokio::test]
async fn automatic_checkpoint_ack_failure_preserves_envelope_and_releases_lease() {
    let (mut agent, prelude) = checkpoint_test_agent();
    agent.set_logical_checkpoint_config(LogicalCheckpointConfig {
        enabled: true,
        automatic: true,
        ..Default::default()
    });
    let boundary = agent.turn.automatic_checkpoint.begin_complete_boundary();
    agent.turn.automatic_checkpoint.mark_attempted(boundary);
    let scheduler_before = agent.turn.automatic_checkpoint.clone();
    let candidate = prepared_checkpoint_for(&agent);
    agent.set_logical_checkpoint_candidate_provider(Arc::new(move || Ok(candidate.clone())));
    let history_before = agent.history.clone();
    let frames_before = agent.protocol_frames.clone();
    let snapshot_before = agent.runtime_snapshot.clone();
    let workflow_before = agent.turn.workflow.clone();
    let run = agent.logical_checkpoint_control.begin_run();
    assert_eq!(
        agent.logical_checkpoint_control.request_automatic(boundary),
        LogicalCheckpointRequestOutcome::Queued
    );

    logical_checkpoint::commit_pending_at_boundary_with_automatic_token(
        &mut agent,
        ApiProtocol::Responses,
        &prelude,
        0,
        Some(boundary),
        &mut |_| std::future::ready(Err(anyhow!("durable automatic acknowledgement failed"))),
    )
    .await
    .expect_err("failed automatic acknowledgement must not install its candidate");

    assert_eq!(agent.history, history_before);
    assert_eq!(agent.protocol_frames, frames_before);
    assert_eq!(agent.runtime_snapshot, snapshot_before);
    assert_eq!(agent.turn.workflow, workflow_before);
    assert_eq!(agent.turn.automatic_checkpoint, scheduler_before);
    assert_eq!(
        agent
            .logical_checkpoint_control
            .request_automatic(boundary + 1),
        LogicalCheckpointRequestOutcome::Queued,
        "the failed lease is cleared rather than stranding the scheduler"
    );
    drop(run);
}

#[tokio::test]
async fn automatic_checkpoint_owner_mismatch_rejects_without_consuming_manual_budget_or_lease() {
    let (mut agent, prelude) = checkpoint_test_agent();
    agent.set_logical_checkpoint_config(LogicalCheckpointConfig {
        enabled: true,
        automatic: true,
        max_automatic_per_turn: 1,
        ..Default::default()
    });
    let candidate = prepared_checkpoint_for(&agent);
    agent.set_logical_checkpoint_candidate_provider(Arc::new(move || Ok(candidate.clone())));
    let run = agent.logical_checkpoint_control.begin_run();
    assert_eq!(
        agent.logical_checkpoint_control.request_automatic(7),
        LogicalCheckpointRequestOutcome::Queued
    );

    let mut logical_events = 0;
    let error = logical_checkpoint::commit_pending_at_boundary_with_automatic_token(
        &mut agent,
        ApiProtocol::Responses,
        &prelude,
        0,
        Some(8),
        &mut |event| {
            logical_events += usize::from(matches!(event, AgentEvent::LogicalCheckpoint { .. }));
            std::future::ready(Ok(()))
        },
    )
    .await
    .expect_err("a stale automatic boundary must not commit");

    assert!(error.to_string().contains("does not match"));
    assert_eq!(
        logical_events, 0,
        "a stale automatic owner is never persisted"
    );
    assert_eq!(agent.turn.automatic_checkpoint.commits, 0);
    assert_eq!(
        agent.logical_checkpoint_control.request_automatic(9),
        LogicalCheckpointRequestOutcome::Queued,
        "the mismatched lease was cleaned up"
    );
    drop(run);
}

#[tokio::test]
async fn logical_checkpoint_success_installs_exact_prepared_envelope_after_acknowledgement() {
    let (mut agent, prelude) = checkpoint_test_agent();
    agent.turn.automatic_checkpoint.begin_complete_boundary();
    let candidate = prepared_checkpoint_for(&agent);
    let expected_history =
        crate::protocol_frames::history_items_from_frames(&candidate.projected_protocol_frames);
    let expected_snapshot = candidate.projected_snapshot.clone();
    let expected_frames = candidate.projected_protocol_frames.clone();
    agent.set_logical_checkpoint_candidate_provider(Arc::new(move || Ok(candidate.clone())));
    let frozen = FrozenTurnEvidence {
        message: Some("frozen evidence".into()),
        selected_ids: vec!["e-1".into()],
    };
    agent.turn.frozen_evidence = Some(frozen.clone());
    let run = agent.logical_checkpoint_control.begin_run();
    assert_eq!(
        agent.request_logical_checkpoint(),
        LogicalCheckpointRequestOutcome::Queued
    );
    let mut acknowledged = false;

    let protected_start = logical_checkpoint::commit_pending_at_batch_boundary(
        &mut agent,
        ApiProtocol::Responses,
        &prelude,
        0,
        &mut |event| {
            assert!(matches!(event, AgentEvent::LogicalCheckpoint { .. }));
            acknowledged = true;
            std::future::ready(Ok(()))
        },
    )
    .await
    .expect("checkpoint commits")
    .expect("prepared successor supplies protected start");

    assert!(acknowledged);
    assert_eq!(protected_start.protected_start_index, 0);
    assert_eq!(protected_start.owner, LogicalCheckpointRequestOwner::Manual);
    assert!(agent.turn.automatic_checkpoint.view().boundary_consumed);
    assert!(!agent.turn.automatic_checkpoint.armed);
    assert_eq!(agent.turn.automatic_checkpoint.commits, 0);
    assert_eq!(agent.history, expected_history);
    assert_eq!(agent.protocol_frames, expected_frames);
    assert_eq!(agent.runtime_snapshot, expected_snapshot);
    assert_eq!(agent.turn.current_turn_start_index, Some(0));
    assert_eq!(agent.turn.frozen_evidence, Some(frozen));
    assert_eq!(
        agent.runtime_snapshot.current_turn_id,
        Some(agent.turn.turn_id)
    );
    assert_eq!(agent.runtime_snapshot.current_segment_id, Some(2));
    drop(run);
}

fn checkpoint_recorder(name: &str) -> Arc<Mutex<TranscriptRecorder>> {
    let directory = std::env::temp_dir().join(format!(
        "letcode-phase3b-{name}-{}",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("clock")
            .as_nanos()
    ));
    Arc::new(Mutex::new(
        TranscriptRecorder::create(directory).expect("create checkpoint recorder"),
    ))
}

/// Builds a live envelope and its journal predecessor using the same event
/// persistence boundary as the interactive runners.
fn transcript_backed_checkpoint_agent(
    recorder: &Arc<Mutex<TranscriptRecorder>>,
) -> (Agent<OpenAIConfig>, Vec<PromptMessage>) {
    let (mut agent, prelude) = checkpoint_test_agent();
    let checkpoint_recorder = Arc::clone(recorder);
    let mut recorder_guard = recorder.lock().expect("checkpoint recorder lock");
    recorder_guard
        .record_session_started("m1")
        .expect("session started");
    recorder_guard
        .record_user_message("current request")
        .expect("user message");
    recorder_guard
        .record_turn_started(agent.turn_started_event())
        .expect("turn started");
    recorder_guard
        .record_assistant_message("working")
        .expect("assistant message");
    agent.history.push(HistoryItem::assistant("working"));
    agent.protocol_frames = crate::protocol_frames::history_items_to_frames(&agent.history);
    agent.runtime_snapshot.current_segment_id = Some(0);
    agent.runtime_snapshot.leaf_sequence = Some(
        read_records(recorder_guard.path())
            .expect("read checkpoint predecessor")
            .last()
            .expect("checkpoint predecessor record")
            .sequence,
    );
    agent.runtime_snapshot.latest_model = Some(agent.model.clone());
    drop(recorder_guard);
    agent.set_logical_checkpoint_candidate_provider(Arc::new(move || {
        checkpoint_recorder
            .lock()
            .map_err(|_| anyhow!("checkpoint recorder poisoned"))?
            .prepare_logical_checkpoint()
    }));
    (agent, prelude)
}

#[tokio::test]
async fn phase3b_transcript_checkpoint_acknowledgement_replays_the_installed_successor() {
    let recorder = checkpoint_recorder("acknowledged-successor");
    let (mut agent, prelude) = transcript_backed_checkpoint_agent(&recorder);
    let run = agent.logical_checkpoint_control.begin_run();
    assert_eq!(
        agent.request_logical_checkpoint(),
        LogicalCheckpointRequestOutcome::Queued
    );

    logical_checkpoint::commit_pending_at_batch_boundary(
        &mut agent,
        ApiProtocol::Responses,
        &prelude,
        0,
        &mut |event| {
            let recorder = Arc::clone(&recorder);
            async move {
                persist_agent_event(
                    &mut recorder.lock().expect("checkpoint recorder lock"),
                    &event,
                )
                .map(|_| ())
            }
        },
    )
    .await
    .expect("transcript acknowledgement commits checkpoint");

    let records = {
        let recorder = recorder.lock().expect("checkpoint recorder lock");
        read_records(recorder.path()).expect("read committed transcript")
    };
    assert!(matches!(
        records.last().map(|record| &record.event),
        Some(TranscriptEvent::LogicalCheckpoint(_))
    ));
    let replay = restore_runtime_snapshot(&records).expect("replay checkpoint successor");
    assert_eq!(agent.runtime_snapshot, replay);
    assert_eq!(
        agent.history,
        restore_session_history(&records).expect("replay checkpoint history")
    );
    let replay_protocol =
        crate::transcript::transcript_projection::project_runtime_restore_snapshot(
            recorder
                .lock()
                .expect("checkpoint recorder lock")
                .session_id()
                .to_string(),
            records.clone(),
            crate::transcript::transcript_projection::SessionContextCursor {
                branch_id: Some(ROOT_CONTEXT_BRANCH_ID.into()),
                leaf_sequence: records.last().map(|record| record.sequence),
            },
            &[],
        )
        .expect("replay checkpoint protocol");
    assert_eq!(agent.protocol_frames, replay_protocol.protocol_frames);
    drop(run);
}

#[tokio::test]
async fn phase3b_journal_frontier_race_rejects_without_record_or_installation() {
    let recorder = checkpoint_recorder("frontier-race");
    let (mut agent, prelude) = transcript_backed_checkpoint_agent(&recorder);
    let before_history = agent.history.clone();
    let before_frames = agent.protocol_frames.clone();
    let before_snapshot = agent.runtime_snapshot.clone();
    let run = agent.logical_checkpoint_control.begin_run();
    assert_eq!(
        agent.request_logical_checkpoint(),
        LogicalCheckpointRequestOutcome::Queued
    );

    let error = logical_checkpoint::commit_pending_at_batch_boundary(
        &mut agent,
        ApiProtocol::Responses,
        &prelude,
        0,
        &mut |event| {
            let recorder = Arc::clone(&recorder);
            async move {
                let mut recorder = recorder.lock().expect("checkpoint recorder lock");
                recorder
                    .record_assistant_message("racing writer")
                    .expect("race write");
                persist_agent_event(&mut recorder, &event).map(|_| ())
            }
        },
    )
    .await
    .expect_err("stale recorder frontier rejects acknowledgement");

    assert!(error.to_string().contains("stale"));
    assert_eq!(agent.history, before_history);
    assert_eq!(agent.protocol_frames, before_frames);
    assert_eq!(agent.runtime_snapshot, before_snapshot);
    let records = {
        let recorder = recorder.lock().expect("checkpoint recorder lock");
        read_records(recorder.path()).expect("read transcript")
    };
    assert!(
        !records
            .iter()
            .any(|record| matches!(record.event, TranscriptEvent::LogicalCheckpoint(_)))
    );
    drop(run);
}

#[tokio::test]
async fn phase3b_recorder_cursor_branch_change_after_preparation_rejects_without_installing() {
    let recorder = checkpoint_recorder("cursor-branch-race");
    let (mut agent, prelude) = transcript_backed_checkpoint_agent(&recorder);
    let before_history = agent.history.clone();
    let before_frames = agent.protocol_frames.clone();
    let before_snapshot = agent.runtime_snapshot.clone();
    let run = agent.logical_checkpoint_control.begin_run();
    assert_eq!(
        agent.request_logical_checkpoint(),
        LogicalCheckpointRequestOutcome::Queued
    );

    let error = logical_checkpoint::commit_pending_at_batch_boundary(
        &mut agent,
        ApiProtocol::Responses,
        &prelude,
        0,
        &mut |event| {
            let recorder = Arc::clone(&recorder);
            async move {
                let mut recorder = recorder.lock().expect("checkpoint recorder lock");
                // Moving only the recorder cursor must invalidate the prepared
                // branch envelope even though no journal record is appended.
                recorder.set_current_context_branch_id(Some("other-branch".into()));
                persist_agent_event(&mut recorder, &event).map(|_| ())
            }
        },
    )
    .await
    .expect_err("a cursor branch change must reject the prepared candidate");

    assert!(error.to_string().contains("expected branch"));
    assert_eq!(agent.history, before_history);
    assert_eq!(agent.protocol_frames, before_frames);
    assert_eq!(agent.runtime_snapshot, before_snapshot);
    let records = {
        let recorder = recorder.lock().expect("checkpoint recorder lock");
        read_records(recorder.path()).expect("read transcript")
    };
    assert!(
        !records
            .iter()
            .any(|record| matches!(record.event, TranscriptEvent::LogicalCheckpoint(_)))
    );
    drop(run);
}

fn assert_request_telemetry_is_terminal_once(events: &[LlmRequestTelemetry]) {
    let mut terminals = HashMap::new();
    for event in events {
        match event.phase {
            LlmRequestTelemetryPhase::Prepared => {
                assert!(
                    terminals
                        .insert((event.logical_request_id.as_str(), event.attempt), None,)
                        .is_none(),
                    "duplicate prepared event for physical request"
                );
            }
            LlmRequestTelemetryPhase::Completed
            | LlmRequestTelemetryPhase::Failed
            | LlmRequestTelemetryPhase::Interrupted => {
                let key = (event.logical_request_id.as_str(), event.attempt);
                let terminal = terminals
                    .get_mut(&key)
                    .expect("terminal event without prepared event");
                assert!(
                    terminal.replace(event.phase).is_none(),
                    "duplicate terminal event"
                );
            }
        }
    }
    assert!(
        terminals.values().all(Option::is_some),
        "prepared event without terminal event"
    );
}

fn test_skill_registry() -> Arc<SkillRegistry> {
    Arc::new(
        SkillRegistry::from_entries(vec![crate::skills::SkillEntry {
            name: "rust-audit".into(),
            description: "Inspect Rust code".into(),
            body: "# Private body".into(),
            content: "---\nname: rust-audit\ndescription: Inspect Rust code\n---\n# Private body\n"
                .into(),
            location: ".letcode/skills".into(),
            path: PathBuf::from("/workspace/.letcode/skills/rust-audit/SKILL.md"),
            base_dir: PathBuf::from("/workspace/.letcode/skills/rust-audit"),
        }])
        .expect("skill registry"),
    )
}

fn test_agent() -> Agent<OpenAIConfig> {
    let client = Client::with_config(
        OpenAIConfig::new()
            .with_api_base("https://api.openai.com/v1")
            .with_api_key("test"),
    );
    Agent::new(client, "m1", 4, 4)
}

fn active_epoch_agent(history: Vec<HistoryItem>) -> Agent<OpenAIConfig> {
    let mut agent = test_agent();
    agent.history = history;
    agent.runtime_snapshot = runtime_snapshot_for_history("active-epoch", &agent.history);
    agent.runtime_snapshot.current_turn_id = Some(1);
    agent.protocol_frames = agent.runtime_snapshot.active_protocol_frames();
    agent.turn.turn_id = 1;
    agent
}

fn replace_active_epoch_history(agent: &mut Agent<OpenAIConfig>, history: Vec<HistoryItem>) {
    agent.history = history;
    agent.runtime_snapshot = runtime_snapshot_for_history("active-epoch", &agent.history);
    agent.runtime_snapshot.current_turn_id = Some(agent.turn.turn_id);
    agent.protocol_frames = agent.runtime_snapshot.active_protocol_frames();
}

fn active_epoch_tools() -> Vec<crate::request_builder::ToolSpec> {
    vec![crate::request_builder::ToolSpec {
        name: "lookup".into(),
        description: "Lookup a value".into(),
        parameters: json!({"type": "object", "properties": {"key": {"type": "string"}}}),
        strict: true,
    }]
}

fn active_epoch_history_with_complete_tool_group() -> Vec<HistoryItem> {
    vec![
        HistoryItem::user("seed"),
        HistoryItem::AssistantToolCalls {
            text: Some("calling tools".into()),
            calls: vec![
                crate::protocol_frames::ProtocolToolCall {
                    call_id: "call-1".into(),
                    name: "lookup".into(),
                    arguments_json: r#"{"key":"one"}"#.into(),
                },
                crate::protocol_frames::ProtocolToolCall {
                    call_id: "call-2".into(),
                    name: "lookup".into(),
                    arguments_json: r#"{"key":"two"}"#.into(),
                },
            ],
        },
        HistoryItem::ToolOutput {
            call_id: "call-1".into(),
            output_json: r#"{"value":1}"#.into(),
        },
        HistoryItem::ToolOutput {
            call_id: "call-2".into(),
            output_json: r#"{"value":2}"#.into(),
        },
    ]
}

#[test]
fn active_epoch_appends_complete_groups_for_both_protocols() {
    for protocol in [ApiProtocol::Responses, ApiProtocol::Completions] {
        let tools = active_epoch_tools();
        let mut agent = active_epoch_agent(vec![HistoryItem::user("seed")]);
        let cold = agent
            .preview_active_epoch(protocol, &[], &tools)
            .expect("cold preview");
        assert!(matches!(cold.transition, ActiveEpochTransition::Cold));
        let cold_units = cold.epoch.observation.units.clone();
        let cold_cacheable_prefix = cold.build.budget.plan_cacheable_prefix_tokens;
        agent.commit_active_epoch(cold);

        replace_active_epoch_history(&mut agent, active_epoch_history_with_complete_tool_group());
        let append = agent
            .preview_active_epoch(protocol, &[], &tools)
            .expect("complete tool group appends");
        assert!(matches!(
            append.transition,
            ActiveEpochTransition::Append { added: 3 }
        ));
        assert_eq!(
            &append.epoch.observation.units[..cold_units.len()],
            cold_units.as_slice(),
            "{protocol:?} preserves the exact provider-unit prefix"
        );
        let prefix_len = agent
            .active_epoch
            .as_ref()
            .expect("committed epoch")
            .committed_plan
            .segments
            .len();
        assert!(
            append.epoch.committed_plan.segments[..prefix_len]
                .iter()
                .all(|segment| segment.stability
                    == crate::request_builder::prompt_plan::PromptSegmentStability::Stable)
        );
        assert!(
            append.epoch.committed_plan.segments[prefix_len..]
                .iter()
                .all(|segment| segment.stability
                    == crate::request_builder::prompt_plan::PromptSegmentStability::Volatile)
        );
        assert!(
            append.build.budget.plan_cacheable_prefix_tokens > cold_cacheable_prefix,
            "{protocol:?} has a cacheable prefix after append"
        );
        agent.commit_active_epoch(append);

        let repeat = agent
            .preview_active_epoch(protocol, &[], &tools)
            .expect("exact repeat appends nothing");
        assert!(matches!(
            repeat.transition,
            ActiveEpochTransition::Append { added: 0 }
        ));
    }
}

#[test]
fn active_epoch_rejects_non_append_changes_without_advancing() {
    let tools = active_epoch_tools();
    for mutation in ["mutated", "truncated"] {
        let mut agent = active_epoch_agent(vec![HistoryItem::user("seed")]);
        let preview = agent
            .preview_active_epoch(ApiProtocol::Responses, &[], &tools)
            .expect("cold preview");
        agent.commit_active_epoch(preview);
        let before = agent.active_epoch.clone();
        if mutation == "mutated" {
            agent.protocol_frames[0].item = ProtocolFrameItem::UserMessage {
                content: crate::user_content::UserMessageContent::new("changed", Vec::new()),
            };
        } else {
            agent.protocol_frames.clear();
        }
        assert!(
            agent
                .preview_active_epoch(ApiProtocol::Responses, &[], &tools)
                .is_err()
        );
        assert_eq!(agent.active_epoch, before);
    }

    for item in [
        HistoryItem::AssistantToolCalls {
            text: None,
            calls: vec![crate::protocol_frames::ProtocolToolCall {
                call_id: "call-1".into(),
                name: "lookup".into(),
                arguments_json: "{}".into(),
            }],
        },
        HistoryItem::ToolOutput {
            call_id: "orphan".into(),
            output_json: "{}".into(),
        },
    ] {
        let mut agent = active_epoch_agent(vec![HistoryItem::user("seed")]);
        let preview = agent
            .preview_active_epoch(ApiProtocol::Responses, &[], &tools)
            .expect("cold preview");
        agent.commit_active_epoch(preview);
        let before = agent.active_epoch.clone();
        let mut history = agent.history.clone();
        history.push(item);
        replace_active_epoch_history(&mut agent, history);
        assert!(
            agent
                .preview_active_epoch(ApiProtocol::Responses, &[], &tools)
                .is_err()
        );
        assert_eq!(agent.active_epoch, before);
    }
}

#[test]
fn active_epoch_cold_rebuilds_on_provider_identity_changes() {
    let tools = active_epoch_tools();
    let mut agent = active_epoch_agent(vec![HistoryItem::user("seed")]);
    let mut prelude = vec![PromptMessage::developer("kernel")];
    let initial = agent
        .preview_active_epoch(ApiProtocol::Responses, &prelude, &tools)
        .expect("initial preview");
    agent.commit_active_epoch(initial);

    let protocol = agent
        .preview_active_epoch(ApiProtocol::Completions, &prelude, &tools)
        .expect("protocol change rebuilds cold");
    assert!(matches!(protocol.transition, ActiveEpochTransition::Cold));
    agent.commit_active_epoch(protocol);

    let mut changed_tools = tools.clone();
    changed_tools[0].parameters = json!({
        "type": "object",
        "properties": {"key": {"type": "string"}, "limit": {"type": "integer"}}
    });
    let shape = agent
        .preview_active_epoch(ApiProtocol::Completions, &prelude, &changed_tools)
        .expect("tool schema change rebuilds cold");
    assert!(matches!(shape.transition, ActiveEpochTransition::Cold));
    agent.commit_active_epoch(shape);

    prelude.push(PromptMessage::developer("changed kernel"));
    let kernel = agent
        .preview_active_epoch(ApiProtocol::Completions, &prelude, &changed_tools)
        .expect("kernel change rebuilds cold");
    assert!(matches!(kernel.transition, ActiveEpochTransition::Cold));
    agent.commit_active_epoch(kernel);

    agent.turn.frozen_evidence = Some(FrozenTurnEvidence {
        message: Some("changed envelope evidence".into()),
        selected_ids: vec!["active-epoch".into()],
    });
    let envelope = agent
        .preview_active_epoch(ApiProtocol::Completions, &prelude, &changed_tools)
        .expect("evidence change rebuilds cold");
    assert!(matches!(envelope.transition, ActiveEpochTransition::Cold));
}

#[test]
fn active_epoch_resets_at_lifecycle_boundaries() {
    let tools = active_epoch_tools();
    let mut agent = active_epoch_agent(vec![HistoryItem::user("seed")]);
    let preview = agent
        .preview_active_epoch(ApiProtocol::Responses, &[], &tools)
        .expect("cold preview");
    agent.commit_active_epoch(preview);
    agent.prepare_turn_prelude("next turn");
    assert!(agent.active_epoch.is_none());
    assert!(matches!(
        agent
            .preview_active_epoch(ApiProtocol::Responses, &[], &tools)
            .expect("new turn preview")
            .transition,
        ActiveEpochTransition::Cold
    ));

    let frames = agent.protocol_frames.clone();
    let snapshot = agent.runtime_snapshot.clone();
    agent.commit_active_epoch(
        agent
            .preview_active_epoch(ApiProtocol::Responses, &[], &tools)
            .expect("preview before restore"),
    );
    agent
        .restore_runtime_snapshot(frames, snapshot)
        .expect("restore succeeds");
    assert!(agent.active_epoch.is_none());
    assert!(matches!(
        agent
            .preview_active_epoch(ApiProtocol::Responses, &[], &tools)
            .expect("restored preview")
            .transition,
        ActiveEpochTransition::Cold
    ));
}

fn test_evidence(id: &str, sequence: u64) -> EvidenceRecord {
    EvidenceRecord {
        id: id.into(),
        sequence,
        timestamp_ms: 0,
        evidence_kind: crate::evidence::EvidenceKind::Decision,
        title: format!("evidence {id}"),
        summary: format!("summary {id}"),
        detail: None,
        source: EvidenceSource::Transcript { sequence },
        tags: Vec::new(),
    }
}

fn runtime_frames_for_history(history: &[HistoryItem]) -> Vec<RuntimeFrame> {
    crate::protocol_frames::history_items_to_frames(history)
        .iter()
        .enumerate()
        .map(|(ordinal, frame)| runtime_frame_from_protocol_frame(frame, ordinal as u32))
        .collect()
}

fn runtime_snapshot_for_history(
    branch_id: impl Into<String>,
    history: &[HistoryItem],
) -> RuntimeSnapshot {
    let mut snapshot = RuntimeSnapshot::new(branch_id);
    snapshot.frames = runtime_frames_for_history(history);
    snapshot
}

#[test]
fn runtime_compaction_applies_repeatedly_with_cumulative_ids_and_retained_frames() {
    let mut agent = test_agent();
    let history = vec![
        HistoryItem::user("first"),
        HistoryItem::assistant("second"),
        HistoryItem::user("retained"),
    ];
    agent.replace_history(history).expect("valid history");
    agent.runtime_snapshot = compaction::test_snapshot_for_history(&agent.history);
    let first_id = agent.runtime_snapshot.frames[0].id;
    let second_id = agent.runtime_snapshot.frames[1].id;
    let retained_id = agent.runtime_snapshot.frames[2].id;
    let first_span = agent.runtime_snapshot.frames[0]
        .provenance
        .source_span
        .unwrap();
    let second_span = agent.runtime_snapshot.frames[1]
        .provenance
        .source_span
        .unwrap();
    let first = compaction::CompactionSelection {
        previous_summary: None,
        head_for_summary: vec![HistoryItem::user("first")],
        tail_items: Vec::new(),
        tail_start_index: 1,
        retired_frame_ids: vec![first_id],
        dependent_frame_ids: Vec::new(),
        retired_source_spans: vec![first_span],
    };
    agent
        .apply_runtime_compaction(&first, "first summary".into())
        .expect("first apply succeeds");
    let summary_id = agent.runtime_snapshot.frames[0].id;
    let second = compaction::CompactionSelection {
        previous_summary: Some("first summary".into()),
        head_for_summary: vec![HistoryItem::assistant("second")],
        tail_items: Vec::new(),
        tail_start_index: 1,
        retired_frame_ids: vec![second_id],
        dependent_frame_ids: Vec::new(),
        retired_source_spans: vec![second_span],
    };
    agent
        .apply_runtime_compaction(&second, "second summary".into())
        .expect("second apply succeeds");

    assert_eq!(
        agent.runtime_snapshot.compaction.compacted_frame_ids,
        vec![first_id, second_id]
    );
    assert_eq!(agent.runtime_snapshot.frames[0].id, summary_id);
    assert!(
        agent
            .runtime_snapshot
            .frames
            .iter()
            .any(|frame| frame.id == retained_id && frame.visibility == FrameVisibility::Active)
    );
}

#[test]
fn runtime_compaction_overlap_failure_is_atomic() {
    let mut agent = test_agent();
    let history = vec![HistoryItem::user("old"), HistoryItem::user("retained")];
    agent.replace_history(history).expect("valid history");
    agent.runtime_snapshot = compaction::test_snapshot_for_history(&agent.history);
    let before = agent.runtime_snapshot.clone();
    let invalid = compaction::CompactionSelection {
        previous_summary: None,
        head_for_summary: vec![HistoryItem::user("old")],
        tail_items: Vec::new(),
        tail_start_index: 1,
        retired_frame_ids: vec![before.frames[0].id],
        dependent_frame_ids: Vec::new(),
        retired_source_spans: vec![SourceSpan::new(1, 2).unwrap()],
    };

    assert!(
        agent
            .apply_runtime_compaction(&invalid, "summary".into())
            .is_err()
    );
    assert_eq!(agent.runtime_snapshot, before);
}

#[test]
fn runtime_snapshot_provider_refresh_retains_durable_metadata() {
    let mut agent = test_agent();
    let history = vec![HistoryItem::user("current")];
    agent
        .replace_history(history.clone())
        .expect("valid history");
    agent.runtime_snapshot = compaction::test_snapshot_for_history(&history);
    let session_frame = RuntimeFrame::new(
        RuntimeFrameKind::Metadata,
        FrameVisibility::Active,
        RuntimeFrameProvenance::new(RuntimeSource::SessionState),
        RuntimeFrameIdSeed {
            frame_kind: RuntimeFrameKind::Metadata,
            source: RuntimeSource::SessionState,
            ordinal: 0,
            stable_key: "durable-session-state",
            source_span: None,
        },
    );
    let session_frame_id = session_frame.id;
    agent.runtime_snapshot.push_frame(session_frame);
    agent
        .runtime_snapshot
        .push_child_session(RuntimeChildSession {
            parent_run_id: "parent".into(),
            child_session_id: "child".into(),
            agent_name: "explorer".into(),
            status: "completed".into(),
            summary: "retained".into(),
            timestamp_ms: 1,
        });
    agent
        .runtime_snapshot
        .push_prompt_contributor(PromptContributorPlaceholder {
            contributor_id: "contributor".into(),
            kind: PromptContributorKind::RuntimeContext,
            label: Some("retained".into()),
            provenance: RuntimeFrameProvenance::new(RuntimeSource::ContextView),
            frame_ids: Vec::new(),
            source_frame_ids: Vec::new(),
        });
    let projected = runtime_snapshot_for_history("main", &history);
    agent.set_runtime_snapshot_provider(Arc::new(move || Ok(projected.clone())));

    agent
        .refresh_runtime_snapshot_from_provider()
        .expect("refresh succeeds");

    assert!(
        agent
            .runtime_snapshot
            .frames
            .iter()
            .any(|frame| frame.id == session_frame_id)
    );
    assert_eq!(
        agent.runtime_snapshot.child_sessions[0].child_session_id,
        "child"
    );
    assert_eq!(
        agent.runtime_snapshot.prompt_contributors[0].contributor_id,
        "contributor"
    );
}

#[test]
fn runtime_snapshot_provider_refresh_accepts_empty_context_projection() {
    let mut agent = test_agent();
    let history = vec![HistoryItem::user("current")];
    agent
        .replace_history(history.clone())
        .expect("valid history");
    agent.runtime_snapshot = runtime_snapshot_for_history("main", &history);
    let records = vec![transcript_record(
        1,
        TranscriptEvent::AssistantMessage {
            content: "stale context".into(),
        },
    )];
    agent
        .runtime_snapshot
        .set_context_view(project_context_view(&records).expect("context view"));
    agent
        .runtime_snapshot
        .set_context_tree(project_context_tree(&records).expect("context tree"));
    let projected = runtime_snapshot_for_history("main", &history);
    agent.set_runtime_snapshot_provider(Arc::new(move || Ok(projected.clone())));

    agent
        .refresh_runtime_snapshot_from_provider()
        .expect("refresh succeeds");

    assert_eq!(
        agent.runtime_snapshot.context_view,
        ContextViewProjection::default()
    );
    assert_eq!(
        agent.runtime_snapshot.context_tree,
        ContextTreeState::with_default_root()
    );
}

#[test]
fn evidence_has_one_runtime_snapshot_authority_and_failed_candidates_are_atomic() {
    let mut agent = test_agent();
    agent
        .add_evidence(test_evidence("ev-live", 1))
        .expect("add live evidence");
    assert_eq!(agent.evidence(), agent.runtime_snapshot.evidence.as_slice());

    let before = agent.runtime_snapshot.evidence.clone();
    assert!(agent.add_evidence(test_evidence("ev-live", 2)).is_err());
    assert_eq!(agent.runtime_snapshot.evidence, before);

    assert!(
        agent
            .restore_evidence(vec![
                test_evidence("ev-duplicate", 2),
                test_evidence("ev-duplicate", 3)
            ])
            .is_err()
    );
    assert_eq!(agent.runtime_snapshot.evidence, before);

    assert!(
        agent
            .restore_session_history(
                Vec::new(),
                vec![
                    test_evidence("ev-duplicate", 2),
                    test_evidence("ev-duplicate", 3),
                ],
                0,
            )
            .is_err()
    );
    assert_eq!(agent.runtime_snapshot.evidence, before);

    let mut invalid_restore = RuntimeSnapshot::new(ROOT_CONTEXT_BRANCH_ID);
    invalid_restore.set_evidence(vec![
        test_evidence("ev-duplicate", 2),
        test_evidence("ev-duplicate", 3),
    ]);
    assert!(
        agent
            .restore_runtime_snapshot(Vec::new(), invalid_restore)
            .is_err()
    );
    assert_eq!(agent.runtime_snapshot.evidence, before);

    let mut replacement = RuntimeSnapshot::new(ROOT_CONTEXT_BRANCH_ID);
    replacement.set_evidence(vec![test_evidence("ev-provider", 4)]);
    agent.set_runtime_snapshot_provider(Arc::new(move || Ok(replacement.clone())));
    agent
        .replace_runtime_snapshot_from_provider()
        .expect("replace provider snapshot");
    assert_eq!(agent.evidence(), agent.runtime_snapshot.evidence.as_slice());
    assert_eq!(agent.evidence()[0].id, "ev-provider");

    let before = agent.runtime_snapshot.evidence.clone();
    let mut invalid = RuntimeSnapshot::new(ROOT_CONTEXT_BRANCH_ID);
    invalid.set_evidence(vec![
        test_evidence("ev-duplicate", 5),
        test_evidence("ev-duplicate", 6),
    ]);
    agent.set_runtime_snapshot_provider(Arc::new(move || Ok(invalid.clone())));
    assert!(agent.replace_runtime_snapshot_from_provider().is_err());
    assert_eq!(agent.runtime_snapshot.evidence, before);
}

fn transcript_record(sequence: u64, event: TranscriptEvent) -> TranscriptRecord {
    TranscriptRecord {
        session_id: "s".into(),
        sequence,
        timestamp_ms: 0,
        context_branch_id: None,
        event,
    }
}

#[tokio::test]
async fn runtime_snapshot_allows_context_tool_execution_without_context_provider() {
    let mut agent = test_agent();
    let records = vec![
        transcript_record(
            1,
            TranscriptEvent::UserMessage {
                content: crate::user_content::UserMessageContent::from("append-only requirement"),
            },
        ),
        transcript_record(
            2,
            TranscriptEvent::ContextNodeCreated {
                node_id: "node-a".into(),
                parent_node_id: Some("root".into()),
                label: Some("node a".into()),
                purpose: Some("tool snapshot".into()),
                block_ref: None,
                source_ref: None,
            },
        ),
        transcript_record(
            3,
            TranscriptEvent::ContextNodeLifecycle {
                node_id: "root".into(),
                status: ContextNodeStatus::Inactive,
            },
        ),
        transcript_record(
            4,
            TranscriptEvent::ContextNodeLifecycle {
                node_id: "node-a".into(),
                status: ContextNodeStatus::Active,
            },
        ),
    ];
    agent
        .runtime_snapshot
        .set_context_view(project_context_view(&records).expect("context view"));
    agent
        .runtime_snapshot
        .set_context_tree(project_context_tree(&records).expect("context tree"));

    let call = HistoryToolCall {
        call_id: "call-1".into(),
        name: tool_names::TOOL_CONTEXT_LIST.into(),
        arguments_json: json!({"include_archived":false,"include_removed":false,"limit":null})
            .to_string(),
    };

    let record = tool_execution::execute_tool_call(
        &mut agent,
        &call,
        &mut |_| async { Ok(()) },
        &mut |_| async { Ok(PermissionApproval::Deny) },
    )
    .await
    .expect("context tool executes with injected snapshots");

    assert!(record.output.ok, "{:?}", record.output);
    let nodes = record
        .output
        .data
        .as_ref()
        .and_then(|data| data.get("nodes"))
        .and_then(Value::as_array)
        .expect("nodes array");
    assert!(nodes.iter().any(|node| node["ref_id"] == "node-a"));
}

#[tokio::test]
async fn non_context_tool_execution_does_not_require_snapshot_provider() {
    let mut agent = test_agent();
    let call = HistoryToolCall {
        call_id: "call-echo".into(),
        name: tool_names::TOOL_UTIL_ECHO.into(),
        arguments_json: json!({"text":"hello"}).to_string(),
    };

    let record = tool_execution::execute_tool_call(
        &mut agent,
        &call,
        &mut |_| async { Ok(()) },
        &mut |_| async { Ok(PermissionApproval::Deny) },
    )
    .await
    .expect("non-context tool executes without snapshots");

    assert!(record.output.ok, "{:?}", record.output);
}

#[tokio::test]
async fn context_tool_execution_without_snapshot_provider_uses_runtime_snapshot() {
    let mut agent = test_agent();
    let call = HistoryToolCall {
        call_id: "call-context".into(),
        name: tool_names::TOOL_CONTEXT_LIST.into(),
        arguments_json: json!({"include_archived":false,"include_removed":false,"limit":null})
            .to_string(),
    };

    let record = tool_execution::execute_tool_call(
        &mut agent,
        &call,
        &mut |_| async { Ok(()) },
        &mut |_| async { Ok(PermissionApproval::Deny) },
    )
    .await
    .expect("context tool execution is returned as record");

    assert!(record.output.ok, "{:?}", record.output);
}

#[test]
fn agent_iteration_limit_allows_tool_budget_plus_final_round() {
    let client = Client::with_config(
        OpenAIConfig::new()
            .with_api_base("https://api.openai.com/v1")
            .with_api_key("test"),
    );
    let agent = Agent::new(client, "m1", 64, 128);

    assert_eq!(agent.max_tool_calls_limit(), Some(128));
    assert_eq!(agent.max_iterations_limit(), Some(64));
}

#[test]
fn agent_iteration_limit_preserves_larger_configured_limit() {
    let client = Client::with_config(
        OpenAIConfig::new()
            .with_api_base("https://api.openai.com/v1")
            .with_api_key("test"),
    );
    let agent = Agent::new(client, "m1", 200, 128);

    assert_eq!(agent.max_iterations_limit(), Some(200));
}

#[test]
fn agent_limits_are_unbounded_by_default_when_omitted() {
    let client = Client::with_config(
        OpenAIConfig::new()
            .with_api_base("https://api.openai.com/v1")
            .with_api_key("test"),
    );
    let agent = Agent::new(client, "m1", None, None);

    assert_eq!(agent.max_iterations_limit(), None);
    assert_eq!(agent.max_tool_calls_limit(), None);
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

async fn spawn_chat_completion_server(
    responses: Vec<&'static str>,
) -> (String, Arc<AtomicUsize>, JoinHandle<()>) {
    let listener = TcpListener::bind("127.0.0.1:0")
        .await
        .expect("test server should bind");
    let addr = listener.local_addr().expect("test server has local addr");
    let count = Arc::new(AtomicUsize::new(0));
    let server_count = count.clone();
    let handle = tokio::spawn(async move {
        for response in responses {
            let (mut socket, _) = listener.accept().await.expect("server accepts request");
            // A single read can stop anywhere in the headers or request body.  Serving
            // the next scripted response at that point lets a connection close race the
            // client's upload, which made response sequencing intermittent under a busy
            // serial suite.  Consume the complete request before advancing the script.
            read_complete_http_request(&mut socket).await;
            server_count.fetch_add(1, Ordering::SeqCst);
            socket
                .write_all(response.as_bytes())
                .await
                .expect("server writes response");
            socket.shutdown().await.expect("server closes response");
        }
    });
    (format!("http://{addr}"), count, handle)
}

fn sse_response(body: String) -> &'static str {
    Box::leak(
            format!(
                "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                body.len()
            )
            .into_boxed_str(),
        )
}

fn responses_tool_batch_sse(calls: Vec<serde_json::Value>) -> &'static str {
    let response = json!({
        "type": "response.completed", "sequence_number": 1,
        "response": {
            "id": "r-tools", "object": "response", "created_at": 1,
            "status": "completed", "background": false, "error": null,
            "incomplete_details": null, "instructions": null, "max_output_tokens": null,
            "model": "m1", "output": calls, "parallel_tool_calls": true,
            "previous_response_id": null, "reasoning": {}, "store": true,
            "temperature": 1, "text": {"format": {"type": "text"}},
            "tool_choice": "auto", "tools": [], "top_p": 1,
            "truncation": "disabled",
            "usage": {"input_tokens": 1, "input_tokens_details": {"cached_tokens": 0}, "output_tokens": 1, "output_tokens_details": {"reasoning_tokens": 0}, "total_tokens": 2},
            "user": null, "metadata": {}
        }
    });
    let response = serde_json::to_string(&response).expect("response serializes");
    sse_response(format!("data: {response}\n\ndata: [DONE]\n\n"))
}

fn chat_tool_batch_sse(name: &str, call_id: &str, arguments: String) -> &'static str {
    let response = json!({
        "choices": [{
            "index": 0,
            "delta": {
                "tool_calls": [{
                    "index": 0,
                    "id": call_id,
                    "type": "function",
                    "function": {"name": name, "arguments": arguments}
                }]
            },
            "finish_reason": "tool_calls"
        }]
    });
    let response = serde_json::to_string(&response).expect("response serializes");
    sse_response(format!("data: {response}\n\ndata: [DONE]\n\n"))
}

fn responses_final_sse(text: &str) -> &'static str {
    let response = json!({
        "type": "response.completed", "sequence_number": 1,
        "response": {
            "id": "r-final", "object": "response", "created_at": 1,
            "status": "completed", "background": false, "error": null,
            "incomplete_details": null, "instructions": null, "max_output_tokens": null,
            "model": "m1", "output": [{"type": "message", "id": "m1", "status": "completed", "role": "assistant", "content": [{"type": "output_text", "text": text, "annotations": []}]}],
            "parallel_tool_calls": true, "previous_response_id": null, "reasoning": {},
            "store": true, "temperature": 1, "text": {"format": {"type": "text"}},
            "tool_choice": "auto", "tools": [], "top_p": 1, "truncation": "disabled",
            "usage": {"input_tokens": 1, "input_tokens_details": {"cached_tokens": 0}, "output_tokens": 1, "output_tokens_details": {"reasoning_tokens": 0}, "total_tokens": 2},
            "user": null, "metadata": {}
        }
    });
    let response = serde_json::to_string(&response).expect("response serializes");
    sse_response(format!("data: {response}\n\ndata: [DONE]\n\n"))
}

fn checkpoint_stream_agent(base_url: String, protocol: ApiProtocol) -> Agent<OpenAIConfig> {
    let client = Client::with_config(
        OpenAIConfig::new()
            .with_api_base(base_url)
            .with_api_key("test"),
    );
    let mut agent = Agent::new(client, "m1", 4, 4);
    agent.set_default_protocol(protocol);
    agent.set_model_catalog(HashMap::from([(
        "m1".into(),
        ModelRequestMetadata {
            context_window: Some(32_000),
            max_output_tokens: Some(2_000),
            supports_tools: true,
            ..Default::default()
        },
    )]));
    agent.set_logical_checkpoint_config(LogicalCheckpointConfig {
        enabled: true,
        ..Default::default()
    });
    // The stream owns turn 1; seed the live segment that it continues.
    agent.turn.turn_id = 1;
    agent.runtime_snapshot.current_turn_id = Some(1);
    agent.runtime_snapshot.current_segment_id = Some(1);
    agent.runtime_snapshot.leaf_sequence = Some(9);
    agent.runtime_snapshot.latest_model = Some("m1".into());
    let candidate = prepared_checkpoint_for(&agent);
    agent.set_logical_checkpoint_candidate_provider(Arc::new(move || Ok(candidate.clone())));
    agent
}

async fn assert_checkpointed_tool_stream(protocol: ApiProtocol) {
    let response_body = r#"data: {"type":"response.completed","sequence_number":1,"response":{"id":"r-tools","object":"response","created_at":1,"status":"completed","background":false,"error":null,"incomplete_details":null,"instructions":null,"max_output_tokens":null,"model":"m1","output":[{"type":"function_call","id":"f1","call_id":"one","name":"test__replay_guard","arguments":"{}","status":"completed"},{"type":"function_call","id":"f2","call_id":"two","name":"test__replay_guard","arguments":"{}","status":"completed"}],"parallel_tool_calls":true,"previous_response_id":null,"reasoning":{},"store":true,"temperature":1,"text":{"format":{"type":"text"}},"tool_choice":"auto","tools":[],"top_p":1,"truncation":"disabled","usage":{"input_tokens":1,"input_tokens_details":{"cached_tokens":0},"output_tokens":1,"output_tokens_details":{"reasoning_tokens":0},"total_tokens":2},"user":null,"metadata":{}}}

data: [DONE]

"#;
    let response_final = r#"data: {"type":"response.completed","sequence_number":1,"response":{"id":"r-final","object":"response","created_at":1,"status":"completed","background":false,"error":null,"incomplete_details":null,"instructions":null,"max_output_tokens":null,"model":"m1","output":[{"type":"message","id":"m1","status":"completed","role":"assistant","content":[{"type":"output_text","text":"done","annotations":[]}]}],"parallel_tool_calls":true,"previous_response_id":null,"reasoning":{},"store":true,"temperature":1,"text":{"format":{"type":"text"}},"tool_choice":"auto","tools":[],"top_p":1,"truncation":"disabled","usage":{"input_tokens":1,"input_tokens_details":{"cached_tokens":0},"output_tokens":1,"output_tokens_details":{"reasoning_tokens":0},"total_tokens":2},"user":null,"metadata":{}}}

data: [DONE]

"#;
    let chat_tools = r#"data: {"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"one","type":"function","function":{"name":"test__replay_guard","arguments":"{}"}},{"index":1,"id":"two","type":"function","function":{"name":"test__replay_guard","arguments":"{}"}}]},"finish_reason":"tool_calls"}]}

data: [DONE]

"#;
    let chat_final = r#"data: {"choices":[{"index":0,"delta":{"content":"done"},"finish_reason":"stop"}]}

data: [DONE]

"#;
    let bodies = match protocol {
        ApiProtocol::Responses => vec![
            sse_response(response_body.into()),
            sse_response(response_final.into()),
        ],
        ApiProtocol::Completions => vec![
            sse_response(chat_tools.into()),
            sse_response(chat_final.into()),
        ],
    };
    let (base_url, requests, server) = spawn_chat_completion_server(bodies).await;
    let mut agent = checkpoint_stream_agent(base_url, protocol);
    let executions = Arc::new(AtomicUsize::new(0));
    agent.register_tool(ReplayGuardTool(executions.clone()));
    assert_eq!(
        agent.request_logical_checkpoint(),
        LogicalCheckpointRequestOutcome::Queued
    );
    let mut events = Vec::new();
    let result = agent
        .run_stream_async(
            "continue",
            |_| std::future::ready(Ok(())),
            |event| {
                events.push(match event {
                    AgentEvent::AssistantToolCallBatch { .. } => "batch",
                    AgentEvent::ToolCallBatchFinished => "finished",
                    AgentEvent::LogicalCheckpoint { .. } => "checkpoint",
                    AgentEvent::AssistantMessage { .. } => "final",
                    AgentEvent::ModelStreamIssue { .. } => "recovery",
                    _ => "other",
                });
                std::future::ready(Ok(()))
            },
            |_| std::future::ready(Ok(PermissionApproval::AllowOnce)),
        )
        .await
        .expect("tool batch and successor request should complete");

    assert_eq!(result, "done");
    assert_eq!(requests.load(Ordering::SeqCst), 2);
    assert_eq!(
        executions.load(Ordering::SeqCst),
        2,
        "each tool runs once, never on the successor request"
    );
    assert_eq!(
        events
            .iter()
            .filter(|event| **event == "checkpoint")
            .count(),
        1
    );
    let batch = events
        .iter()
        .position(|event| *event == "batch")
        .expect("tool batch");
    let finished = events
        .iter()
        .position(|event| *event == "finished")
        .expect("batch finished");
    let checkpoint = events
        .iter()
        .position(|event| *event == "checkpoint")
        .expect("checkpoint");
    let final_message = events
        .iter()
        .position(|event| *event == "final")
        .expect("final reply");
    assert!(
        batch < finished && finished < checkpoint && checkpoint < final_message,
        "{events:?}"
    );
    assert!(!events[..finished].contains(&"checkpoint"));
    assert!(!events.contains(&"recovery"));
    // Finalization clears the per-turn start marker, while the checkpointed
    // successor remains in the same advanced segment.
    assert_eq!(agent.turn.current_turn_start_index, None);
    assert_eq!(agent.runtime_snapshot.current_segment_id, Some(2));
    server.await.expect("server task should finish");
}

#[tokio::test]
async fn logical_checkpoint_responses_stream_commits_only_after_multi_tool_batch() {
    assert_checkpointed_tool_stream(ApiProtocol::Responses).await;
}

#[tokio::test]
async fn logical_checkpoint_chat_stream_commits_only_after_multi_tool_batch() {
    assert_checkpointed_tool_stream(ApiProtocol::Completions).await;
}

#[tokio::test]
async fn phase3b_live_checkpoint_preserves_101_complete_tool_pairs_without_reexecution() {
    let calls = (0..101)
        .map(|index| {
            json!({
                "type": "function_call",
                "id": format!("f-{index}"),
                "call_id": format!("call-{index}"),
                "name": "test__replay_guard",
                "arguments": "{}",
                "status": "completed"
            })
        })
        .collect::<Vec<_>>();
    let tools = json!({
        "type": "response.completed",
        "sequence_number": 1,
        "response": {
            "id": "r-101-tools", "object": "response", "created_at": 1,
            "status": "completed", "background": false, "error": null,
            "incomplete_details": null, "instructions": null,
            "max_output_tokens": null, "model": "m1", "output": calls,
            "parallel_tool_calls": true, "previous_response_id": null,
            "reasoning": {}, "store": true, "temperature": 1,
            "text": {"format": {"type": "text"}}, "tool_choice": "auto",
            "tools": [], "top_p": 1, "truncation": "disabled",
            "usage": {"input_tokens": 1, "input_tokens_details": {"cached_tokens": 0}, "output_tokens": 1, "output_tokens_details": {"reasoning_tokens": 0}, "total_tokens": 2},
            "user": null, "metadata": {}
        }
    });
    let final_reply = r#"data: {"type":"response.completed","sequence_number":1,"response":{"id":"r-101-final","object":"response","created_at":1,"status":"completed","background":false,"error":null,"incomplete_details":null,"instructions":null,"max_output_tokens":null,"model":"m1","output":[{"type":"message","id":"m1","status":"completed","role":"assistant","content":[{"type":"output_text","text":"done","annotations":[]}]}],"parallel_tool_calls":true,"previous_response_id":null,"reasoning":{},"store":true,"temperature":1,"text":{"format":{"type":"text"}},"tool_choice":"auto","tools":[],"top_p":1,"truncation":"disabled","usage":{"input_tokens":1,"input_tokens_details":{"cached_tokens":0},"output_tokens":1,"output_tokens_details":{"reasoning_tokens":0},"total_tokens":2},"user":null,"metadata":{}}}

data: [DONE]

"#;
    let tool_reply = format!("data: {}\n\ndata: [DONE]\n\n", tools);
    let (base_url, requests, server) = spawn_chat_completion_server(vec![
        sse_response(tool_reply),
        sse_response(final_reply.into()),
    ])
    .await;
    let mut agent = checkpoint_stream_agent(base_url, ApiProtocol::Responses);
    agent.max_tool_calls = Some(128);
    agent.max_iterations = Some(4);
    let executions = Arc::new(AtomicUsize::new(0));
    agent.register_tool(ReplayGuardTool(Arc::clone(&executions)));
    assert_eq!(
        agent.request_logical_checkpoint(),
        LogicalCheckpointRequestOutcome::Queued
    );
    let mut checkpoint_count = 0;
    let mut observed_pair_count = 0;
    let result = agent
        .run_stream_async(
            "continue",
            |_| std::future::ready(Ok(())),
            |event| {
                if let AgentEvent::AssistantToolCallBatch { ref calls, .. } = event {
                    observed_pair_count += calls.len();
                }
                if matches!(event, AgentEvent::LogicalCheckpoint { .. }) {
                    checkpoint_count += 1;
                }
                std::future::ready(Ok(()))
            },
            |_| std::future::ready(Ok(PermissionApproval::AllowOnce)),
        )
        .await
        .expect("101 complete tool pairs must produce a valid successor request");

    assert_eq!(result, "done");
    assert_eq!(requests.load(Ordering::SeqCst), 2);
    assert_eq!(observed_pair_count, 101);
    assert_eq!(executions.load(Ordering::SeqCst), 101);
    assert_eq!(checkpoint_count, 1);
    assert_eq!(agent.runtime_snapshot.current_segment_id, Some(2));
    assert!(crate::protocol_frames::validate_history_items_complete(&agent.history, None).is_ok());
    server.await.expect("server task should finish");
}

#[tokio::test]
async fn phase3b_live_turn_commits_two_successor_segments_after_distinct_batches() {
    let first_batch = vec![json!({
        "type":"function_call", "id":"f1", "call_id":"one",
        "name":"test__replay_guard", "arguments":"{}", "status":"completed"
    })];
    let second_batch = vec![json!({
        "type":"function_call", "id":"f2", "call_id":"two",
        "name":"test__replay_guard", "arguments":"{}", "status":"completed"
    })];
    let (base_url, requests, server) = spawn_chat_completion_server(vec![
        responses_tool_batch_sse(first_batch),
        responses_tool_batch_sse(second_batch),
        responses_final_sse("done"),
    ])
    .await;
    let mut agent = checkpoint_stream_agent(base_url, ApiProtocol::Responses);
    let first = prepared_checkpoint_for_lineage(&agent, "checkpoint-live-1", None);
    let mut after_first = checkpoint_stream_agent("http://unused".into(), ApiProtocol::Responses);
    after_first.runtime_snapshot.current_segment_id = Some(2);
    after_first.runtime_snapshot.leaf_sequence = Some(10);
    let second = prepared_checkpoint_for_lineage(
        &after_first,
        "checkpoint-live-2",
        Some("checkpoint-live-1"),
    );
    let candidates = Arc::new(vec![first, second]);
    let candidate_index = Arc::new(AtomicUsize::new(0));
    agent.set_logical_checkpoint_candidate_provider({
        let candidates = Arc::clone(&candidates);
        let candidate_index = Arc::clone(&candidate_index);
        Arc::new(move || {
            candidates
                .get(candidate_index.fetch_add(1, Ordering::SeqCst))
                .cloned()
                .ok_or_else(|| anyhow!("unexpected third checkpoint"))
        })
    });
    let executions = Arc::new(AtomicUsize::new(0));
    agent.register_tool(ReplayGuardTool(Arc::clone(&executions)));
    assert_eq!(
        agent.request_logical_checkpoint(),
        LogicalCheckpointRequestOutcome::Queued
    );
    let checkpoint_control = agent.logical_checkpoint_control.clone();
    let mut checkpoints = Vec::new();
    let mut batches = 0;
    let result = agent
        .run_stream_async(
            "continue",
            |_| std::future::ready(Ok(())),
            |event| {
                if matches!(event, AgentEvent::AssistantToolCallBatch { .. }) {
                    batches += 1;
                    if batches == 2 {
                        assert_eq!(
                            checkpoint_control.request(),
                            LogicalCheckpointRequestOutcome::Queued
                        );
                    }
                }
                if let AgentEvent::LogicalCheckpoint { event, .. } = event {
                    checkpoints.push((
                        event.previous_segment_id,
                        event.segment_id,
                        event.previous_checkpoint_id,
                    ));
                }
                std::future::ready(Ok(()))
            },
            |_| std::future::ready(Ok(PermissionApproval::AllowOnce)),
        )
        .await
        .expect("two completed batches commit two successors");

    assert_eq!(result, "done");
    assert_eq!(requests.load(Ordering::SeqCst), 3);
    assert_eq!(executions.load(Ordering::SeqCst), 2);
    assert_eq!(
        checkpoints,
        vec![(1, 2, None), (2, 3, Some("checkpoint-live-1".into()))]
    );
    assert_eq!(agent.runtime_snapshot.current_segment_id, Some(3));
    assert!(crate::protocol_frames::validate_history_items_complete(&agent.history, None).is_ok());
    server.await.expect("server task should finish");
}

#[tokio::test]
async fn phase3b_live_workflow_controls_are_preserved_by_successor_checkpoint() {
    let calls = vec![
        json!({"type":"function_call", "id":"todos", "call_id":"todos", "name":"workflow__todos", "arguments":"{\"items\":[{\"id\":\"t1\",\"content\":\"ship\",\"status\":\"pending\"}]}", "status":"completed"}),
        json!({"type":"function_call", "id":"continue", "call_id":"continue", "name":"workflow__auto_continue", "arguments":"{\"enabled\":true,\"max_continuations\":2}", "status":"completed"}),
        json!({"type":"function_call", "id":"reset", "call_id":"reset", "name":"workflow__todos", "arguments":"{\"items\":[]}", "status":"completed"}),
    ];
    let (base_url, requests, server) = spawn_chat_completion_server(vec![
        responses_tool_batch_sse(calls),
        responses_final_sse("done"),
    ])
    .await;
    let mut agent = checkpoint_stream_agent(base_url, ApiProtocol::Responses);
    let mut candidate = prepared_checkpoint_for(&agent);
    candidate.projected_workflow = Some(crate::transcript::CheckpointWorkflowProjection {
        todos: Vec::new(),
        auto_continue: AutoContinueState {
            enabled: true,
            max_continuations: 2,
        },
    });
    agent.set_logical_checkpoint_candidate_provider(Arc::new(move || Ok(candidate.clone())));
    assert_eq!(
        agent.request_logical_checkpoint(),
        LogicalCheckpointRequestOutcome::Queued
    );
    let result = agent
        .run_stream_async(
            "continue",
            |_| std::future::ready(Ok(())),
            |_| std::future::ready(Ok(())),
            |_| std::future::ready(Ok(PermissionApproval::AllowOnce)),
        )
        .await
        .expect("workflow successor request succeeds");

    assert_eq!(result, "done");
    assert_eq!(requests.load(Ordering::SeqCst), 2);
    let expected = WorkflowState {
        todos: Vec::new(),
        auto_continue: AutoContinueState {
            enabled: true,
            max_continuations: 2,
        },
    };
    assert_eq!(agent.turn.workflow, expected);
    server.await.expect("server task should finish");
}

#[tokio::test]
async fn phase3b_live_checkpoint_eligibility_rejections_are_nonpersistent_and_release_leases() {
    type Configure =
        fn(&mut Agent<OpenAIConfig>, &mut crate::transcript::PreparedLogicalCheckpoint);
    let cases: Vec<(&str, Configure)> = vec![
        ("active context experiment", |agent, _| {
            agent.set_context_scope_state(Arc::new(std::sync::Mutex::new(ContextScopeState {
                active_experiment: Some(ActiveContextExperiment {
                    branch_id: "experiment".into(),
                    parent_branch_id: ROOT_CONTEXT_BRANCH_ID.into(),
                    base_sequence: 9,
                    writes_observed: false,
                }),
            })));
        }),
        ("restored unavailable live turn", |agent, _| {
            agent.turn.current_turn_start_index = None;
        }),
        ("model protocol mismatch", |agent, _| {
            agent.set_model_protocols(HashMap::from([("m1".into(), ApiProtocol::Completions)]));
        }),
        ("scope mismatch", |_, candidate| {
            candidate.event.context_scope_revision += 1;
        }),
        ("workflow mismatch", |_, candidate| {
            candidate.projected_workflow = Some(crate::transcript::CheckpointWorkflowProjection {
                todos: vec![TodoItem {
                    id: "wrong".into(),
                    content: "wrong".into(),
                    status: TodoStatus::Pending,
                }],
                auto_continue: AutoContinueState::default(),
            });
        }),
        ("prospective request overflow", |agent, _| {
            agent.set_model_catalog(HashMap::from([(
                "m1".into(),
                ModelRequestMetadata {
                    context_window: Some(1),
                    max_output_tokens: Some(1),
                    supports_tools: true,
                    ..Default::default()
                },
            )]));
        }),
    ];

    for (name, configure) in cases {
        let (mut agent, prelude) = checkpoint_test_agent();
        let mut candidate = prepared_checkpoint_for(&agent);
        configure(&mut agent, &mut candidate);
        agent.set_logical_checkpoint_candidate_provider(Arc::new(move || Ok(candidate.clone())));
        let history = agent.history.clone();
        let frames = agent.protocol_frames.clone();
        let snapshot = agent.runtime_snapshot.clone();
        let workflow = agent.turn.workflow.clone();
        let run = agent.logical_checkpoint_control.begin_run();
        assert_eq!(
            agent.request_logical_checkpoint(),
            LogicalCheckpointRequestOutcome::Queued
        );
        let mut logical_events = 0;
        logical_checkpoint::commit_pending_at_batch_boundary(
            &mut agent,
            ApiProtocol::Responses,
            &prelude,
            0,
            &mut |event| {
                logical_events +=
                    usize::from(matches!(event, AgentEvent::LogicalCheckpoint { .. }));
                std::future::ready(Ok(()))
            },
        )
        .await
        .expect_err(name);
        assert_eq!(logical_events, 0, "{name} must not persist a logical event");
        assert_eq!(agent.history, history, "{name}");
        assert_eq!(agent.protocol_frames, frames, "{name}");
        assert_eq!(agent.runtime_snapshot, snapshot, "{name}");
        assert_eq!(agent.turn.workflow, workflow, "{name}");
        assert_eq!(
            agent.request_logical_checkpoint(),
            LogicalCheckpointRequestOutcome::Queued,
            "{name} lease"
        );
        drop(run);
    }
}

async fn assert_cancelled_stream_releases_checkpoint_lease(
    protocol: ApiProtocol,
    cancel_at_checkpoint: bool,
) {
    let response_tools = r#"data: {"type":"response.completed","sequence_number":1,"response":{"id":"r-tools","object":"response","created_at":1,"status":"completed","background":false,"error":null,"incomplete_details":null,"instructions":null,"max_output_tokens":null,"model":"m1","output":[{"type":"function_call","id":"f1","call_id":"one","name":"test__replay_guard","arguments":"{}","status":"completed"}],"parallel_tool_calls":true,"previous_response_id":null,"reasoning":{},"store":true,"temperature":1,"text":{"format":{"type":"text"}},"tool_choice":"auto","tools":[],"top_p":1,"truncation":"disabled","usage":{"input_tokens":1,"input_tokens_details":{"cached_tokens":0},"output_tokens":1,"output_tokens_details":{"reasoning_tokens":0},"total_tokens":2},"user":null,"metadata":{}}}

data: [DONE]

"#;
    let chat_tools = r#"data: {"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"one","type":"function","function":{"name":"test__replay_guard","arguments":"{}"}}]},"finish_reason":"tool_calls"}]}

data: [DONE]

"#;
    let response_final = r#"data: {"type":"response.completed","sequence_number":1,"response":{"id":"r-final","object":"response","created_at":1,"status":"completed","background":false,"error":null,"incomplete_details":null,"instructions":null,"max_output_tokens":null,"model":"m1","output":[{"type":"message","id":"m1","status":"completed","role":"assistant","content":[{"type":"output_text","text":"clean","annotations":[]}]}],"parallel_tool_calls":true,"previous_response_id":null,"reasoning":{},"store":true,"temperature":1,"text":{"format":{"type":"text"}},"tool_choice":"auto","tools":[],"top_p":1,"truncation":"disabled","usage":{"input_tokens":1,"input_tokens_details":{"cached_tokens":0},"output_tokens":1,"output_tokens_details":{"reasoning_tokens":0},"total_tokens":2},"user":null,"metadata":{}}}

data: [DONE]

"#;
    let chat_final = r#"data: {"choices":[{"index":0,"delta":{"content":"clean"},"finish_reason":"stop"}]}

data: [DONE]

"#;
    let bodies = match protocol {
        ApiProtocol::Responses => vec![
            sse_response(response_tools.into()),
            sse_response(response_final.into()),
        ],
        ApiProtocol::Completions => vec![
            sse_response(chat_tools.into()),
            sse_response(chat_final.into()),
        ],
    };
    let (base_url, requests, server) = spawn_chat_completion_server(bodies).await;
    let mut agent = checkpoint_stream_agent(base_url, protocol);
    let executions = Arc::new(AtomicUsize::new(0));
    agent.register_tool(ReplayGuardTool(Arc::clone(&executions)));
    assert_eq!(
        agent.request_logical_checkpoint(),
        LogicalCheckpointRequestOutcome::Queued
    );
    let entered = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let entered_event = Arc::clone(&entered);
    {
        let first = agent.run_stream_async(
            "cancelled turn",
            |_| std::future::ready(Ok(())),
            move |event| {
                let stop = if cancel_at_checkpoint {
                    matches!(event, AgentEvent::LogicalCheckpoint { .. })
                } else {
                    matches!(event, AgentEvent::ToolCallBatchFinished)
                };
                let entered = Arc::clone(&entered_event);
                async move {
                    if stop {
                        entered.store(true, Ordering::SeqCst);
                        std::future::pending::<()>().await;
                    }
                    Ok(())
                }
            },
            |_| std::future::ready(Ok(PermissionApproval::AllowOnce)),
        );
        tokio::pin!(first);
        for _ in 0..50 {
            if entered.load(Ordering::SeqCst) {
                break;
            }
            tokio::select! {
                _ = &mut first => panic!("cancelled stream completed before its cancellation point"),
                _ = sleep(Duration::from_millis(10)) => {}
            }
        }
        assert!(
            entered.load(Ordering::SeqCst),
            "stream did not reach cancellation point"
        );
    }
    assert_eq!(
        agent.request_logical_checkpoint(),
        LogicalCheckpointRequestOutcome::Queued
    );

    let clean = agent
        .run_stream_async(
            "clean turn",
            |_| std::future::ready(Ok(())),
            |_| std::future::ready(Ok(())),
            |_| std::future::ready(Ok(PermissionApproval::AllowOnce)),
        )
        .await
        .expect("the successor run must not inherit the cancelled checkpoint lease");
    assert_eq!(clean, "clean");
    assert_eq!(requests.load(Ordering::SeqCst), 2);
    assert_eq!(executions.load(Ordering::SeqCst), 1);
    server.await.expect("server task should finish");
}

#[tokio::test]
async fn phase3b_actual_responses_and_chat_stream_cancellation_releases_clean_next_run() {
    for protocol in [ApiProtocol::Responses, ApiProtocol::Completions] {
        assert_cancelled_stream_releases_checkpoint_lease(protocol, false).await;
        assert_cancelled_stream_releases_checkpoint_lease(protocol, true).await;
    }
}

#[tokio::test]
async fn logical_checkpoint_rejection_matrix_is_nonpersistent_and_releases_its_lease() {
    type CandidateMutation = fn(&mut crate::transcript::PreparedLogicalCheckpoint);
    let cases: Vec<(&str, CandidateMutation)> = vec![
        ("stale frontier", |candidate| {
            candidate.expected_journal_frontier = 8
        }),
        ("frontier overflow", |candidate| {
            candidate.expected_journal_frontier = u64::MAX
        }),
        ("model", |candidate| {
            candidate.projected_snapshot.latest_model = Some("other".into())
        }),
        ("scope", |candidate| {
            candidate.event.context_scope_revision += 1
        }),
        ("branch", |candidate| {
            candidate.expected_branch_id = "other-branch".into();
            candidate.projected_snapshot.active_context.branch_id = "other-branch".into();
        }),
        ("leaf", |candidate| {
            candidate.projected_snapshot.leaf_sequence = Some(11)
        }),
        ("workflow", |candidate| {
            candidate.projected_workflow = Some(crate::transcript::CheckpointWorkflowProjection {
                todos: vec![TodoItem {
                    id: "different".into(),
                    content: "different".into(),
                    status: TodoStatus::Pending,
                }],
                auto_continue: AutoContinueState::default(),
            })
        }),
        ("suffix", |candidate| {
            candidate.projected_protocol_frames.pop();
        }),
    ];

    for (name, mutate) in cases {
        let (mut agent, prelude) = checkpoint_test_agent();
        let mut candidate = prepared_checkpoint_for(&agent);
        mutate(&mut candidate);
        agent.set_logical_checkpoint_candidate_provider(Arc::new(move || Ok(candidate.clone())));
        let history = agent.history.clone();
        let frames = agent.protocol_frames.clone();
        let snapshot = agent.runtime_snapshot.clone();
        let workflow = agent.turn.workflow.clone();
        let run = agent.logical_checkpoint_control.begin_run();
        assert_eq!(
            agent.request_logical_checkpoint(),
            LogicalCheckpointRequestOutcome::Queued
        );
        let mut persisted = false;
        let error = logical_checkpoint::commit_pending_at_batch_boundary(
            &mut agent,
            ApiProtocol::Responses,
            &prelude,
            0,
            &mut |event| {
                persisted |= matches!(event, AgentEvent::LogicalCheckpoint { .. });
                std::future::ready(Ok(()))
            },
        )
        .await
        .expect_err(name);
        assert!(!error.to_string().is_empty(), "{name}");
        assert!(!persisted, "{name} must not emit a logical event");
        assert_eq!(agent.history, history, "{name}");
        assert_eq!(agent.protocol_frames, frames, "{name}");
        assert_eq!(agent.runtime_snapshot, snapshot, "{name}");
        assert_eq!(agent.turn.workflow, workflow, "{name}");
        assert_eq!(
            agent.request_logical_checkpoint(),
            LogicalCheckpointRequestOutcome::Queued,
            "{name} lease"
        );
        drop(run);
    }
}

#[tokio::test]
async fn logical_checkpoint_disabled_and_unrequested_controls_are_exact_noops() {
    for enabled in [false, true] {
        let (mut agent, prelude) = checkpoint_test_agent();
        agent.set_logical_checkpoint_config(LogicalCheckpointConfig {
            enabled,
            ..Default::default()
        });
        let candidates = Arc::new(AtomicUsize::new(0));
        let candidate_count = Arc::clone(&candidates);
        agent.set_logical_checkpoint_candidate_provider(Arc::new(move || {
            candidate_count.fetch_add(1, Ordering::SeqCst);
            unreachable!("a no-op control must not ask for a checkpoint candidate")
        }));
        let history = agent.history.clone();
        let frames = agent.protocol_frames.clone();
        let snapshot = agent.runtime_snapshot.clone();
        let workflow = agent.turn.workflow.clone();
        let run = agent.logical_checkpoint_control.begin_run();
        if !enabled {
            assert_eq!(
                agent.request_logical_checkpoint(),
                LogicalCheckpointRequestOutcome::Disabled
            );
        }
        let result = logical_checkpoint::commit_pending_at_batch_boundary(
            &mut agent,
            ApiProtocol::Responses,
            &prelude,
            0,
            &mut |_| std::future::ready(Err(anyhow!("a no-op must emit no event"))),
        )
        .await
        .expect("disabled or unrequested control is a no-op");
        assert_eq!(result, None);
        assert_eq!(candidates.load(Ordering::SeqCst), 0);
        assert_eq!(agent.history, history);
        assert_eq!(agent.protocol_frames, frames);
        assert_eq!(agent.runtime_snapshot, snapshot);
        assert_eq!(agent.turn.workflow, workflow);
        drop(run);
    }
}

fn test_retry_config() -> RetryConfig {
    RetryConfig {
        enabled: true,
        max_attempts: 3,
        initial_delay_ms: 1,
        max_delay_ms: 5,
        backoff_multiplier: 2.0,
        jitter_ms: 0,
    }
}

fn test_tool_call(name: &str, arguments_json: &str) -> HistoryToolCall {
    HistoryToolCall {
        call_id: format!("call-{name}"),
        name: name.into(),
        arguments_json: arguments_json.into(),
    }
}

fn test_execution_record(tool_name: &str, output: ToolResult) -> ToolExecutionRecord {
    ToolExecutionRecord {
        call_id: format!("call-{tool_name}"),
        tool_name: tool_name.into(),
        arguments: Some(json!({})),
        permission_class: crate::permission::ToolPermissionClass::Read,
        directive: ExecutionDirective::None,
        status: ToolExecutionStatus::Executed,
        rejection: None,
        output,
        effects: ToolEffects {
            kind: ToolEffectKind::Read,
            primary_path: None,
            edited_paths: vec![],
            command: None,
        },
    }
}

struct SleepTool;

#[async_trait]
impl ToolHandler for SleepTool {
    fn name(&self) -> &str {
        "test__sleep"
    }

    fn description(&self) -> &str {
        "sleep test tool"
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": false
        })
    }

    async fn execute(&self, _args: Value) -> Result<Value> {
        sleep(Duration::from_millis(1_100)).await;
        Ok(json!({"done": true}))
    }
}

struct ReplayGuardTool(Arc<AtomicUsize>);

#[async_trait]
impl ToolHandler for ReplayGuardTool {
    fn name(&self) -> &str {
        "test__replay_guard"
    }

    fn description(&self) -> &str {
        "counts executions"
    }

    fn parameters(&self) -> Value {
        json!({"type": "object", "properties": {}, "additionalProperties": false})
    }

    async fn execute(&self, _args: Value) -> Result<Value> {
        self.0.fetch_add(1, Ordering::SeqCst);
        Ok(json!({"executed": true}))
    }
}

/// Makes the completed tool batch exceed the protected tail budget while
/// leaving the checkpoint successor small.  This exercises the actual
/// request-preparation path rather than injecting a scheduler decision.
struct ProtectedOverflowTool(Arc<AtomicUsize>);

#[async_trait]
impl ToolHandler for ProtectedOverflowTool {
    fn name(&self) -> &str {
        "test__protected_overflow"
    }

    fn description(&self) -> &str {
        "produces a protected-tail overflow"
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {"payload": {"type": "string"}},
            "additionalProperties": false
        })
    }

    async fn execute(&self, _args: Value) -> Result<Value> {
        self.0.fetch_add(1, Ordering::SeqCst);
        Ok(json!({"executed": true}))
    }
}

/// Leaves the protected tail below the hard limit, but above the automatic
/// high watermark.  This makes the stream exercise the post-fold soft path
/// instead of the hard-overflow recovery path.
struct SoftPressureTool(Arc<AtomicUsize>);

#[async_trait]
impl ToolHandler for SoftPressureTool {
    fn name(&self) -> &str {
        "test__soft_pressure"
    }

    fn description(&self) -> &str {
        "produces soft protected-tail pressure"
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {"payload": {"type": "string"}},
            "additionalProperties": false
        })
    }

    async fn execute(&self, _args: Value) -> Result<Value> {
        self.0.fetch_add(1, Ordering::SeqCst);
        Ok(json!({"executed": true}))
    }
}

fn tool_call_arguments(payload_size: usize) -> String {
    serde_json::to_string(&json!({"payload": "x ".repeat(payload_size)}))
        .expect("tool arguments serialize")
}

async fn assert_automatic_soft_pressure_checkpoint(protocol: ApiProtocol) {
    let arguments = tool_call_arguments(45_000);
    let responses_tools = responses_tool_batch_sse(vec![json!({
        "type": "function_call", "id": "f1", "call_id": "one",
        "name": "test__soft_pressure", "arguments": arguments, "status": "completed"
    })]);
    let chat_tools = chat_tool_batch_sse("test__soft_pressure", "one", tool_call_arguments(45_000));
    let bodies = match protocol {
            ApiProtocol::Responses => vec![responses_tools, responses_final_sse("done")],
            ApiProtocol::Completions => vec![
                chat_tools,
                sse_response("data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"done\"},\"finish_reason\":\"stop\"}]}\n\ndata: [DONE]\n\n".into()),
            ],
        };
    let (base_url, requests, server) = spawn_chat_completion_server(bodies).await;
    let mut agent = checkpoint_stream_agent(base_url, protocol);
    agent.set_model_catalog(HashMap::from([(
        "m1".into(),
        ModelRequestMetadata {
            context_window: Some(32_000),
            max_output_tokens: Some(128),
            supports_tools: true,
            ..Default::default()
        },
    )]));
    agent.set_logical_checkpoint_config(LogicalCheckpointConfig {
        enabled: true,
        automatic: true,
        max_automatic_per_turn: 1,
    });
    let candidate = prepared_checkpoint_for(&agent);
    let candidates = Arc::new(AtomicUsize::new(0));
    agent.set_logical_checkpoint_candidate_provider({
        let candidates = Arc::clone(&candidates);
        Arc::new(move || {
            candidates.fetch_add(1, Ordering::SeqCst);
            Ok(candidate.clone())
        })
    });
    let executions = Arc::new(AtomicUsize::new(0));
    agent.register_tool(SoftPressureTool(Arc::clone(&executions)));
    let mut checkpoints = 0;
    let mut request_telemetry = 0;
    let answer = agent
        .run_stream_async(
            "continue",
            |_| std::future::ready(Ok(())),
            |event| {
                match event {
                    AgentEvent::LogicalCheckpoint { .. } => checkpoints += 1,
                    AgentEvent::LlmRequestTelemetry(_) => request_telemetry += 1,
                    _ => {}
                }
                std::future::ready(Ok(()))
            },
            |_| std::future::ready(Ok(PermissionApproval::AllowOnce)),
        )
        .await
        .expect("soft pressure checkpoint rebuilds the successor request");

    assert_eq!(answer, "done");
    assert_eq!(requests.load(Ordering::SeqCst), 2);
    assert_eq!(
        executions.load(Ordering::SeqCst),
        1,
        "tool batch is not replayed"
    );
    assert_eq!(candidates.load(Ordering::SeqCst), 1);
    assert_eq!(checkpoints, 1);
    assert_eq!(
        request_telemetry, 4,
        "the discarded pre-checkpoint build emits neither request metadata nor telemetry"
    );
    assert_eq!(agent.runtime_snapshot.current_segment_id, Some(2));
    server.await.expect("server task should finish");
}

#[tokio::test]
async fn phase3c_actual_responses_and_chat_fixed_high_watermark_rebuild_once_without_replay() {
    for protocol in [ApiProtocol::Responses, ApiProtocol::Completions] {
        assert_automatic_soft_pressure_checkpoint(protocol).await;
    }
}

async fn assert_automatic_hard_overflow_checkpoint(protocol: ApiProtocol) {
    let arguments = tool_call_arguments(50_000);
    let response_tools = responses_tool_batch_sse(vec![json!({
        "type": "function_call", "id": "f1", "call_id": "one",
        "name": "test__protected_overflow", "arguments": arguments, "status": "completed"
    })]);
    let chat_tools = chat_tool_batch_sse(
        "test__protected_overflow",
        "one",
        tool_call_arguments(50_000),
    );
    let bodies = match protocol {
            ApiProtocol::Responses => vec![
                response_tools,
                responses_final_sse("done"),
            ],
            ApiProtocol::Completions => vec![
                chat_tools,
                sse_response("data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"done\"},\"finish_reason\":\"stop\"}]}\n\ndata: [DONE]\n\n".into()),
            ],
        };
    let (base_url, requests, server) = spawn_chat_completion_server(bodies).await;
    let mut agent = checkpoint_stream_agent(base_url, protocol);
    agent.set_model_catalog(HashMap::from([(
        "m1".into(),
        ModelRequestMetadata {
            context_window: Some(32_000),
            max_output_tokens: Some(128),
            supports_tools: true,
            ..Default::default()
        },
    )]));
    agent.set_logical_checkpoint_config(LogicalCheckpointConfig {
        enabled: true,
        automatic: true,
        max_automatic_per_turn: 1,
        ..Default::default()
    });
    let candidate = prepared_checkpoint_for(&agent);
    let candidates = Arc::new(AtomicUsize::new(0));
    agent.set_logical_checkpoint_candidate_provider({
        let candidates = Arc::clone(&candidates);
        Arc::new(move || {
            candidates.fetch_add(1, Ordering::SeqCst);
            Ok(candidate.clone())
        })
    });
    let executions = Arc::new(AtomicUsize::new(0));
    agent.register_tool(ProtectedOverflowTool(Arc::clone(&executions)));
    let mut checkpoints = 0;
    let mut prepared = 0;
    let result = agent
        .run_stream_async(
            "continue",
            |_| std::future::ready(Ok(())),
            |event| {
                match event {
                    AgentEvent::LogicalCheckpoint { .. } => checkpoints += 1,
                    AgentEvent::LlmRequestTelemetry(_) => prepared += 1,
                    _ => {}
                }
                std::future::ready(Ok(()))
            },
            |_| std::future::ready(Ok(PermissionApproval::AllowOnce)),
        )
        .await
        .expect("automatic checkpoint rebuilds the protected overflow");

    assert_eq!(result, "done");
    assert_eq!(
        requests.load(Ordering::SeqCst),
        2,
        "one request per iteration"
    );
    assert_eq!(
        executions.load(Ordering::SeqCst),
        1,
        "the batch never replays"
    );
    assert_eq!(candidates.load(Ordering::SeqCst), 1);
    assert_eq!(checkpoints, 1, "one automatic boundary event");
    assert_eq!(
        prepared, 4,
        "only the two provider requests emit prepared and terminal telemetry"
    );
    assert_eq!(agent.runtime_snapshot.current_segment_id, Some(2));
    server.await.expect("server task should finish");
}

#[tokio::test]
async fn phase3c_actual_responses_and_chat_hard_overflow_rebuild_once_without_replay() {
    for protocol in [ApiProtocol::Responses, ApiProtocol::Completions] {
        assert_automatic_hard_overflow_checkpoint(protocol).await;
    }
}

#[tokio::test]
async fn phase3c_actual_two_automatic_boundaries_rearm_below_low_and_commit_successors() {
    let arguments = tool_call_arguments(45_000);
    let first_batch = vec![json!({
        "type": "function_call", "id": "f1", "call_id": "one",
        "name": "test__soft_pressure", "arguments": arguments, "status": "completed"
    })];
    let second_batch = vec![json!({
        "type": "function_call", "id": "f2", "call_id": "two",
        "name": "test__soft_pressure", "arguments": tool_call_arguments(45_000), "status": "completed"
    })];
    let (base_url, requests, server) = spawn_chat_completion_server(vec![
        responses_tool_batch_sse(first_batch),
        responses_tool_batch_sse(second_batch),
        responses_final_sse("done"),
    ])
    .await;
    let mut agent = checkpoint_stream_agent(base_url, ApiProtocol::Responses);
    agent.set_model_catalog(HashMap::from([(
        "m1".into(),
        ModelRequestMetadata {
            context_window: Some(32_000),
            max_output_tokens: Some(128),
            supports_tools: true,
            ..Default::default()
        },
    )]));
    agent.set_logical_checkpoint_config(LogicalCheckpointConfig {
        enabled: true,
        automatic: true,
        max_automatic_per_turn: 2,
    });
    let first = prepared_checkpoint_for_lineage(&agent, "automatic-1", None);
    let mut successor = checkpoint_stream_agent("http://unused".into(), ApiProtocol::Responses);
    successor.runtime_snapshot.current_segment_id = Some(2);
    successor.runtime_snapshot.leaf_sequence = Some(10);
    let second = prepared_checkpoint_for_lineage(&successor, "automatic-2", Some("automatic-1"));
    let candidates = Arc::new(vec![first, second]);
    let candidate_index = Arc::new(AtomicUsize::new(0));
    agent.set_logical_checkpoint_candidate_provider({
        let candidates = Arc::clone(&candidates);
        let candidate_index = Arc::clone(&candidate_index);
        Arc::new(move || {
            candidates
                .get(candidate_index.fetch_add(1, Ordering::SeqCst))
                .cloned()
                .ok_or_else(|| anyhow!("unexpected third automatic checkpoint"))
        })
    });
    let executions = Arc::new(AtomicUsize::new(0));
    agent.register_tool(SoftPressureTool(Arc::clone(&executions)));
    let mut checkpoints = Vec::new();
    let mut request_telemetry = 0;
    let result = agent
        .run_stream_async(
            "continue",
            |_| std::future::ready(Ok(())),
            |event| {
                match event {
                    AgentEvent::LogicalCheckpoint { event, .. } => checkpoints.push((
                        event.previous_segment_id,
                        event.segment_id,
                        event.previous_checkpoint_id,
                    )),
                    AgentEvent::LlmRequestTelemetry(_) => request_telemetry += 1,
                    _ => {}
                }
                std::future::ready(Ok(()))
            },
            |_| std::future::ready(Ok(PermissionApproval::AllowOnce)),
        )
        .await
        .expect("each boundary automatically commits its rearmed successor");

    assert_eq!(result, "done");
    assert_eq!(requests.load(Ordering::SeqCst), 3);
    assert_eq!(executions.load(Ordering::SeqCst), 2);
    assert_eq!(candidate_index.load(Ordering::SeqCst), 2);
    assert_eq!(
        checkpoints,
        vec![(1, 2, None), (2, 3, Some("automatic-1".into()))]
    );
    assert_eq!(
        request_telemetry, 6,
        "only three provider requests are observable"
    );
    assert_eq!(agent.runtime_snapshot.current_segment_id, Some(3));
    server.await.expect("server task should finish");
}

#[tokio::test]
async fn phase3c_actual_automatic_checkpoint_cancellation_never_installs_a_successor() {
    let arguments = tool_call_arguments(45_000);
    let (base_url, requests, server) = spawn_chat_completion_server(vec![
        responses_tool_batch_sse(vec![json!({
            "type": "function_call", "id": "f1", "call_id": "one",
            "name": "test__soft_pressure", "arguments": arguments, "status": "completed"
        })]),
        responses_final_sse("clean"),
    ])
    .await;
    let mut agent = checkpoint_stream_agent(base_url, ApiProtocol::Responses);
    agent.set_model_catalog(HashMap::from([(
        "m1".into(),
        ModelRequestMetadata {
            context_window: Some(32_000),
            max_output_tokens: Some(128),
            supports_tools: true,
            ..Default::default()
        },
    )]));
    agent.set_logical_checkpoint_config(LogicalCheckpointConfig {
        enabled: true,
        automatic: true,
        max_automatic_per_turn: 1,
    });
    agent.register_tool(SoftPressureTool(Arc::new(AtomicUsize::new(0))));
    let entered = Arc::new(std::sync::atomic::AtomicBool::new(false));
    {
        let entered_event = Arc::clone(&entered);
        let stream = agent.run_stream_async(
            "cancel automatic checkpoint",
            |_| std::future::ready(Ok(())),
            move |event| {
                let entered = Arc::clone(&entered_event);
                async move {
                    if matches!(event, AgentEvent::LogicalCheckpoint { .. }) {
                        entered.store(true, Ordering::SeqCst);
                        std::future::pending::<()>().await;
                    }
                    Ok(())
                }
            },
            |_| std::future::ready(Ok(PermissionApproval::AllowOnce)),
        );
        tokio::pin!(stream);
        for _ in 0..50 {
            if entered.load(Ordering::SeqCst) {
                break;
            }
            tokio::select! {
                _ = &mut stream => panic!("stream completed before automatic callback cancellation"),
                _ = sleep(Duration::from_millis(10)) => {}
            }
        }
        assert!(
            entered.load(Ordering::SeqCst),
            "automatic checkpoint callback was not reached"
        );
    }
    assert_eq!(agent.runtime_snapshot.current_segment_id, Some(1));
    agent
        .replace_history(Vec::new())
        .expect("the cancelled stream history can be cleared before the clean run");

    let result = agent
        .run_stream_async(
            "clean next turn",
            |_| std::future::ready(Ok(())),
            |_| std::future::ready(Ok(())),
            |_| std::future::ready(Ok(PermissionApproval::AllowOnce)),
        )
        .await
        .expect("the clean run must not inherit a cancelled automatic checkpoint");
    assert_eq!(result, "clean");
    assert_eq!(requests.load(Ordering::SeqCst), 2);
    assert_eq!(
        agent.turn.automatic_checkpoint,
        AutomaticCheckpointSchedulerState::default()
    );
    server.await.expect("server task should finish");
}

#[test]
fn completed_tool_output_projection_and_restore_never_reexecutes_handler() {
    let records = vec![
        TranscriptRecord {
            session_id: "s".into(),
            sequence: 1,
            timestamp_ms: 0,
            context_branch_id: None,
            event: TranscriptEvent::UserMessage {
                content: crate::user_content::UserMessageContent::from("resume"),
            },
        },
        TranscriptRecord {
            session_id: "s".into(),
            sequence: 2,
            timestamp_ms: 0,
            context_branch_id: None,
            event: TranscriptEvent::AssistantToolCallBatch {
                text: None,
                calls: vec![HistoryToolCall {
                    call_id: "finished".into(),
                    name: "test__replay_guard".into(),
                    arguments_json: "{}".into(),
                }],
            },
        },
        TranscriptRecord {
            session_id: "s".into(),
            sequence: 3,
            timestamp_ms: 0,
            context_branch_id: None,
            event: TranscriptEvent::ToolCallFinished {
                call_id: "finished".into(),
                name: "test__replay_guard".into(),
                ok: true,
                output: ToolResult::ok("test__replay_guard", json!({"persisted": true})),
            },
        },
    ];
    let projected = crate::transcript::transcript_projection::project_runtime_restore_snapshot(
        "s".into(),
        records,
        crate::transcript::transcript_projection::SessionContextCursor {
            branch_id: None,
            leaf_sequence: None,
        },
        &[],
    )
    .expect("project completed output");
    let executions = Arc::new(AtomicUsize::new(0));
    let mut agent = test_agent();
    agent.register_tool(ReplayGuardTool(executions.clone()));
    agent
        .restore_runtime_snapshot(projected.protocol_frames, projected.snapshot)
        .expect("restore persisted output");
    assert_eq!(executions.load(Ordering::SeqCst), 0);
}

#[test]
fn context_checkpoint_cannot_nest_inside_active_experiment() {
    let mut agent = test_agent();
    agent.set_context_scope_state(Arc::new(std::sync::Mutex::new(ContextScopeState {
        active_experiment: Some(ActiveContextExperiment {
            branch_id: "branch-1".into(),
            parent_branch_id: ROOT_CONTEXT_BRANCH_ID.into(),
            base_sequence: 4,
            writes_observed: false,
        }),
    })));

    let error = agent
        .validate_context_control_tool(tool_names::TOOL_CONTEXT_CHECKPOINT)
        .expect_err("nested checkpoint should fail");

    assert!(
        error
            .to_string()
            .contains("cannot start a nested experiment")
    );
}

#[test]
fn context_return_requires_active_experiment() {
    let agent = test_agent();
    let error = agent
        .validate_context_control_tool(tool_names::TOOL_CONTEXT_RETURN)
        .expect_err("return without active experiment should fail");

    assert!(
        error
            .to_string()
            .contains("requires an active context experiment")
    );
}

#[tokio::test]
async fn active_context_experiment_blocks_normal_turn_finalization() {
    let mut agent = test_agent();
    agent.set_context_scope_state(Arc::new(std::sync::Mutex::new(ContextScopeState {
        active_experiment: Some(ActiveContextExperiment {
            branch_id: "branch-1".into(),
            parent_branch_id: ROOT_CONTEXT_BRANCH_ID.into(),
            base_sequence: 4,
            writes_observed: false,
        }),
    })));

    let mut events = Vec::new();
    let error = agent
        .continue_or_finalize_no_tool_reply(
            &mut |event| {
                events.push(event);
                std::future::ready(Ok(()))
            },
            0,
            &mut 0,
        )
        .await
        .expect_err("active experiment should fail closed");

    assert!(
        error
            .to_string()
            .contains("cannot finalize turn while context experiment 'branch-1' is active")
    );
    assert!(
        !events
            .iter()
            .any(|event| matches!(event, AgentEvent::TurnFinalized(_)))
    );
}

#[tokio::test]
async fn non_shell_tool_timeout_emits_cancelled_and_timed_out_terminal_events() {
    let mut agent = test_agent();
    agent.set_tool_timeout_secs(Some(1));
    agent
        .try_register_tool(SleepTool)
        .expect("register sleep tool");

    let call = test_tool_call("test__sleep", "{}");
    let mut events = Vec::new();
    let record = agent
        .execute_tool_call(
            &call,
            &mut |event| {
                events.push(event);
                std::future::ready(Ok(()))
            },
            &mut |_| std::future::ready(Ok(PermissionApproval::AllowOnce)),
        )
        .await
        .expect("tool call should return timeout record");

    assert_eq!(record.status, ToolExecutionStatus::TimedOut);
    assert!(!record.output.ok);
    assert_eq!(
        record
            .output
            .data
            .as_ref()
            .and_then(|data| data.get("status"))
            .and_then(Value::as_str),
        Some("timed_out")
    );
    assert!(events.iter().any(|event| matches!(
        event,
        AgentEvent::ToolCallCancelled { call_id, name }
            if call_id == "call-test__sleep" && name == "test__sleep"
    )));
    assert!(events.iter().any(|event| matches!(
        event,
        AgentEvent::ToolCallFinished { ok, output, .. }
            if !ok
                && output
                    .data
                    .as_ref()
                    .and_then(|data| data.get("status"))
                    .and_then(Value::as_str)
                    == Some("timed_out")
    )));
}

#[test]
fn write_effects_mark_active_context_experiment_before_transcript_audit_replay() {
    let mut agent = test_agent();
    let scope = ActiveContextExperiment {
        branch_id: "branch-1".into(),
        parent_branch_id: ROOT_CONTEXT_BRANCH_ID.into(),
        base_sequence: 4,
        writes_observed: false,
    };
    agent.set_context_scope_state(Arc::new(std::sync::Mutex::new(ContextScopeState {
        active_experiment: Some(scope.clone()),
    })));
    agent.set_runtime_snapshot_provider(Arc::new(|| Ok(RuntimeSnapshot::new("branch-1"))));
    agent.set_context_experiment_restore_point(
        scope,
        Vec::new(),
        RuntimeSnapshot::new(ROOT_CONTEXT_BRANCH_ID),
    );

    let mut record = test_execution_record(
        "fs__write",
        ToolResult::ok("fs__write", json!({"path": "src/lib.rs"})),
    );
    record.effects.kind = ToolEffectKind::Write;
    record.effects.primary_path = Some("src/lib.rs".into());

    agent.record_tool_effects(&record);

    assert!(
        agent
            .context_scope_state
            .lock()
            .expect("scope state lock")
            .active_experiment
            .as_ref()
            .is_some_and(|experiment| experiment.writes_observed)
    );
    assert!(
        agent
            .context_experiment_restore_point
            .as_ref()
            .is_some_and(|restore| restore.scope.writes_observed)
    );
}

#[test]
fn evidence_ids_remain_unique_after_restoring_older_evidence_snapshot() {
    let mut agent = test_agent();
    let first = agent
        .remember_tool_evidence(&test_execution_record(
            "fs__read",
            ToolResult::ok("fs__read", json!({"content": "one"})),
        ))
        .expect("first evidence");
    let older_snapshot = agent.runtime_snapshot.evidence.clone();

    let second = agent
        .remember_tool_evidence(&test_execution_record(
            "fs__read",
            ToolResult::ok("fs__read", json!({"content": "two"})),
        ))
        .expect("second evidence");
    assert_ne!(first.id, second.id);

    agent
        .restore_session_history(agent.history.clone(), older_snapshot, agent.next_turn_id)
        .expect("restore older evidence snapshot");

    let third = agent
        .remember_tool_evidence(&test_execution_record(
            "fs__read",
            ToolResult::ok("fs__read", json!({"content": "three"})),
        ))
        .expect("third evidence after restore");

    assert_ne!(first.id, third.id);
    assert_ne!(second.id, third.id);
}

fn large_tool_output_json(field: &str) -> String {
    json!({field: "line ".repeat((COMPACTION_TOOL_OUTPUT_CHAR_CAP + 500) / 5)}).to_string()
}

fn prunable_tool_output_json(field: &str) -> String {
    json!({field: "line ".repeat((COMPACTION_PRUNE_MIN_OUTPUT_CHARS + 1_000) / 5)}).to_string()
}

fn prune_protect_padding() -> String {
    "padding ".repeat(18_000)
}

struct StaticSubagentDelegate {
    result: ToolResult,
}

struct CapturingSubagentDelegate {
    result: ToolResult,
    explorer_tasks: Arc<std::sync::Mutex<Vec<String>>>,
    fixer_tasks: Arc<std::sync::Mutex<Vec<String>>>,
}

impl SubagentDelegate<OpenAIConfig> for StaticSubagentDelegate {
    fn run_named<'a>(
        &'a self,
        _parent: &'a Agent<OpenAIConfig>,
        _agent_name: &'a str,
        _invocation: SubagentInvocation,
    ) -> Pin<Box<dyn Future<Output = Result<ToolResult>> + Send + 'a>> {
        let result = self.result.clone();
        Box::pin(async move { Ok(result) })
    }
}

impl SubagentDelegate<OpenAIConfig> for CapturingSubagentDelegate {
    fn run_named<'a>(
        &'a self,
        _parent: &'a Agent<OpenAIConfig>,
        agent_name: &'a str,
        invocation: SubagentInvocation,
    ) -> Pin<Box<dyn Future<Output = Result<ToolResult>> + Send + 'a>> {
        match agent_name {
            "fixer" => self
                .fixer_tasks
                .lock()
                .expect("fixer capture lock")
                .push(invocation.prompt),
            _ => self
                .explorer_tasks
                .lock()
                .expect("explorer capture lock")
                .push(invocation.prompt),
        }
        let result = self.result.clone();
        Box::pin(async move { Ok(result) })
    }
}

fn static_delegate(result: ToolResult) -> Arc<dyn SubagentDelegate<OpenAIConfig>> {
    Arc::new(StaticSubagentDelegate { result })
}

fn capturing_delegate(
    result: ToolResult,
) -> (
    Arc<dyn SubagentDelegate<OpenAIConfig>>,
    Arc<std::sync::Mutex<Vec<String>>>,
    Arc<std::sync::Mutex<Vec<String>>>,
) {
    let explorer_tasks = Arc::new(std::sync::Mutex::new(Vec::new()));
    let fixer_tasks = Arc::new(std::sync::Mutex::new(Vec::new()));
    (
        Arc::new(CapturingSubagentDelegate {
            result,
            explorer_tasks: Arc::clone(&explorer_tasks),
            fixer_tasks: Arc::clone(&fixer_tasks),
        }),
        explorer_tasks,
        fixer_tasks,
    )
}

#[test]
fn tool_effects_classify_read_write_validation_command_diagnostic_and_workflow_control() {
    let read = ToolEffects::derive(
        "fs__read",
        Some(&json!({"path": "src/lib.rs"})),
        &ToolResult::ok(
            "fs__read",
            json!({"path": "src/lib.rs", "content": "fn main() {}"}),
        ),
    );
    assert_eq!(read.kind, ToolEffectKind::Read);
    assert_eq!(read.primary_path.as_deref(), Some("src/lib.rs"));
    assert!(read.edited_paths.is_empty());
    assert_eq!(read.command, None);

    let write = ToolEffects::derive(
        "edit__apply_patch",
        None,
        &ToolResult::ok(
            "edit__apply_patch",
            json!({"edits": [{"path": "src/lib.rs"}, {"path": "src/agent.rs"}]}),
        ),
    );
    assert_eq!(write.kind, ToolEffectKind::Write);
    assert_eq!(write.edited_paths, vec!["src/lib.rs", "src/agent.rs"]);

    let validation = ToolEffects::derive(
        "shell__exec",
        Some(&json!({"command": "cargo test transcript"})),
        &ToolResult::ok(
            "shell__exec",
            json!({"command": "cargo test transcript", "status": 0}),
        ),
    );
    assert_eq!(validation.kind, ToolEffectKind::Validation);
    assert_eq!(validation.command.as_deref(), Some("cargo test transcript"));

    let failed_validation = ToolEffects::derive(
        "shell__exec",
        Some(&json!({"command": "cargo test transcript"})),
        &ToolResult::ok(
            "shell__exec",
            json!({"command": "cargo test transcript", "status": 101, "success": false}),
        ),
    );
    assert_eq!(failed_validation.kind, ToolEffectKind::Diagnostic);
    assert_eq!(
        failed_validation.command.as_deref(),
        Some("cargo test transcript")
    );

    let contradictory_failed_validation = ToolEffects::derive(
        "shell__exec",
        Some(&json!({"command": "cargo test transcript"})),
        &ToolResult::ok(
            "shell__exec",
            json!({"command": "cargo test transcript", "status": 101, "success": true}),
        ),
    );
    assert_eq!(
        contradictory_failed_validation.kind,
        ToolEffectKind::Diagnostic
    );

    let checkout = ToolEffects::derive(
        "shell__exec",
        Some(&json!({"command": "git checkout main"})),
        &ToolResult::ok(
            "shell__exec",
            json!({"command": "git checkout main", "status": 0, "success": true}),
        ),
    );
    assert_eq!(checkout.kind, ToolEffectKind::Command);

    let command = ToolEffects::derive(
        "shell__exec",
        Some(&json!({"command": "ls src"})),
        &ToolResult::ok("shell__exec", json!({"command": "ls src", "status": 0})),
    );
    assert_eq!(command.kind, ToolEffectKind::Command);
    assert_eq!(command.command.as_deref(), Some("ls src"));

    let diagnostic = ToolEffects::derive(
        "shell__exec",
        Some(&json!({"command": "cargo test agent::tests::tool"})),
        &ToolResult::err("shell__exec", "command failed"),
    );
    assert_eq!(diagnostic.kind, ToolEffectKind::Diagnostic);
    assert_eq!(
        diagnostic.command.as_deref(),
        Some("cargo test agent::tests::tool")
    );

    let workflow = ToolEffects::derive(
        "workflow__todos",
        Some(&json!({"items": [{"id": "t1", "content": "x", "status": "pending"}]})),
        &ToolResult::ok("workflow__todos", json!({"ok": true})),
    );
    assert_eq!(workflow.kind, ToolEffectKind::WorkflowControl);
}

#[test]
fn agent_tool_definitions_hide_subagent_tools_until_delegate_is_installed() {
    let mut agent = test_agent();
    let specs = agent.tool_definitions();
    assert!(
        specs
            .iter()
            .any(|spec| spec.name == tool_names::TOOL_AGENT_RECONCILE)
    );
    for name in [
        "agent__explore",
        "agent__fixer",
        "agent__oracle",
        "agent__designer",
        "agent__librarian",
        "agent__general",
    ] {
        assert!(
            !specs.iter().any(|spec| spec.name == name),
            "{name} should be hidden"
        );
    }

    agent.set_subagent_delegate(static_delegate(ToolResult::ok(
        "agent__explore",
        json!({
            "run_id": "run-1",
            "child_session_id": "child-session",
            "agent_name": "explorer",
            "status": "completed",
            "summary": "done",
        }),
    )));

    let specs = agent.tool_definitions();
    for name in [
        "agent__explore",
        "agent__fixer",
        "agent__oracle",
        "agent__designer",
        "agent__librarian",
        "agent__general",
    ] {
        assert!(
            specs.iter().any(|spec| spec.name == name),
            "{name} should be exposed"
        );
    }
}

#[test]
fn agent_templates_expose_capability_contracts() {
    let explorer = AgentTemplate::explorer().capability_contract();
    assert_eq!(explorer.name, "explorer");
    assert_eq!(explorer.tool_scope, ToolScope::ReadOnlyExplorer);
    assert_eq!(explorer.permission_mode, PermissionMode::Default);
    assert!(!explorer.can_write);
    assert!(!explorer.can_delegate);
    assert_eq!(explorer.default_max_tool_calls, None);
    assert!(explorer.input_expectations.contains("task 或 objective"));
    assert!(explorer.expected_result_shape.contains("run_id"));

    let fixer = AgentTemplate::fixer().capability_contract();
    assert_eq!(fixer.name, "fixer");
    assert_eq!(fixer.tool_scope, ToolScope::FullAccess);
    assert!(fixer.can_write);
    assert!(!fixer.can_delegate);
    assert_eq!(fixer.default_max_tool_calls, None);

    let readonly_names = ["oracle", "designer", "librarian", "general"];
    for name in readonly_names {
        let template = AgentTemplate::from_name(name).expect("known template");
        let contract = template.capability_contract();
        assert_eq!(contract.name, name);
        assert_eq!(contract.tool_scope, ToolScope::ReadOnlyExplorer);
        assert!(!contract.can_write);
        assert!(!contract.can_delegate);
    }
}

#[test]
fn child_agent_applies_runtime_max_tool_call_override() {
    let agent = test_agent();
    let child =
        AgentFactory::create_child_with_max_tool_calls(&agent, &AgentTemplate::fixer(), Some(1));

    assert_eq!(child.max_tool_calls_limit(), Some(1));
    let error = child
        .ensure_tool_call_budget(0, 2)
        .expect_err("tool-call budget should be enforced");
    assert!(error.to_string().contains("too many tool calls"));
}

#[test]
fn child_agent_retains_explicit_tool_call_override_with_unbounded_parent() {
    let client = Client::with_config(
        OpenAIConfig::new()
            .with_api_base("https://api.openai.com/v1")
            .with_api_key("test"),
    );
    let agent = Agent::new(client, "m1", None, None);
    let child =
        AgentFactory::create_child_with_max_tool_calls(&agent, &AgentTemplate::fixer(), Some(2));

    assert_eq!(child.max_tool_calls_limit(), Some(2));
    child
        .ensure_tool_call_budget(0, 2)
        .expect("budget edge should pass");
    let error = child
        .ensure_tool_call_budget(0, 3)
        .expect_err("explicit child budget should still be enforced");
    assert!(error.to_string().contains("max 2"));
}

#[test]
fn child_agent_inherits_parent_tool_call_limit_without_template_budget() {
    let agent = test_agent();
    let child = AgentFactory::create_child(&agent, &AgentTemplate::fixer());

    assert_eq!(child.max_tool_calls_limit(), agent.max_tool_calls_limit());
}

#[tokio::test]
async fn subagent_tool_execution_normalizes_bounded_input_before_delegation() {
    let mut agent = test_agent();
    let (delegate, explorer_tasks, fixer_tasks) = capturing_delegate(ToolResult::ok(
        "agent__fixer",
        json!({
            "run_id": "run-1",
            "child_session_id": "child-session",
            "agent_name": "fixer",
            "status": "completed",
            "summary": "done"
        }),
    ));
    agent.set_subagent_delegate(delegate);

    let output = agent
        .execute_subagent_tool(
            "agent__fixer",
            &json!({
                "objective": "Implement contract",
                "success_criteria": ["tests pass"],
                "allowed_paths": ["src/agent.rs"],
                "owned_paths": ["src/tool.rs"],
                "timeout_secs": 30,
                "max_tool_calls": 5
            }),
        )
        .await;

    assert!(output.ok);
    assert!(explorer_tasks.lock().expect("explorer tasks").is_empty());
    let fixer_tasks = fixer_tasks.lock().expect("fixer tasks");
    assert_eq!(fixer_tasks.len(), 1);
    let prompt = &fixer_tasks[0];
    assert!(prompt.contains("Objective: Implement contract"));
    assert!(prompt.contains("Success criteria:"));
    assert!(prompt.contains("Allowed paths: src/agent.rs"));
    assert!(prompt.contains("Owned paths: src/tool.rs"));
    assert!(prompt.contains("Execution bounds: timeout_secs=30, max_tool_calls=5"));
    assert!(prompt.contains("do not recursively delegate"));
}

#[tokio::test]
async fn subagent_tool_execution_supports_legacy_task_only_input() {
    let mut agent = test_agent();
    let (delegate, explorer_tasks, _fixer_tasks) = capturing_delegate(ToolResult::ok(
        "agent__explore",
        json!({
            "run_id": "run-1",
            "child_session_id": "child-session",
            "agent_name": "explorer",
            "status": "completed",
            "summary": "done"
        }),
    ));
    agent.set_subagent_delegate(delegate);

    let output = agent
        .execute_subagent_tool(
            "agent__explore",
            &json!({"task": "inspect src/subagent.rs"}),
        )
        .await;

    assert!(output.ok);
    let explorer_tasks = explorer_tasks.lock().expect("explorer tasks");
    assert_eq!(explorer_tasks.len(), 1);
    assert!(explorer_tasks[0].contains("Objective: inspect src/subagent.rs"));
    assert!(explorer_tasks[0].contains("Mode: read-only exploration only."));
}

#[tokio::test]
async fn readonly_expert_subagent_tool_execution_routes_through_generic_delegate() {
    let mut agent = test_agent();
    let (delegate, explorer_tasks, fixer_tasks) = capturing_delegate(ToolResult::ok(
        "agent__oracle",
        json!({
            "run_id": "run-1",
            "child_session_id": "child-session",
            "agent_name": "oracle",
            "status": "completed",
            "summary": "done"
        }),
    ));
    agent.set_subagent_delegate(delegate);

    let output = agent
        .execute_subagent_tool("agent__oracle", &json!({"task": "analyze failure mode"}))
        .await;

    assert!(output.ok);
    assert!(fixer_tasks.lock().expect("fixer tasks").is_empty());
    let readonly_tasks = explorer_tasks.lock().expect("readonly tasks");
    assert_eq!(readonly_tasks.len(), 1);
    assert!(readonly_tasks[0].contains("Objective: analyze failure mode"));
}

#[tokio::test]
async fn subagent_tool_execution_returns_validation_error_for_missing_objective() {
    let agent = test_agent();

    let output = agent
        .execute_subagent_tool("agent__explore", &json!({}))
        .await;

    assert!(!output.ok);
    assert!(
        output
            .error
            .as_ref()
            .expect("validation error")
            .message
            .contains("requires a non-empty 'task' or 'objective'")
    );
}

#[test]
fn remembered_subagent_evidence_carries_parent_turn_provenance() {
    let mut agent = test_agent();
    agent.turn.turn_id = 42;
    let record = ToolExecutionRecord {
        call_id: "call-agent__explore".into(),
        tool_name: "agent__explore".into(),
        arguments: Some(json!({"task": "inspect"})),
        permission_class: crate::permission::ToolPermissionClass::Preview,
        directive: crate::permission::ExecutionDirective::None,
        status: ToolExecutionStatus::Executed,
        rejection: None,
        output: ToolResult::ok(
            "agent__explore",
            json!({
                "run_id": "run-1",
                "child_session_id": "child-1",
                "status": "completed",
                "summary": "done"
            }),
        ),
        effects: ToolEffects {
            kind: ToolEffectKind::Read,
            primary_path: None,
            edited_paths: vec![],
            command: None,
        },
    };

    let evidence = agent
        .remember_tool_evidence(&record)
        .expect("subagent evidence should be recorded");

    match evidence.source {
        EvidenceSource::Subagent {
            run_id,
            child_session_id,
            parent_tool,
            parent_turn_id,
            ..
        } => {
            assert_eq!(run_id, "run-1");
            assert_eq!(child_session_id, "child-1");
            assert_eq!(parent_tool, "agent__explore");
            assert_eq!(parent_turn_id.as_deref(), Some("turn-42"));
        }
        other => panic!("unexpected evidence source: {other:?}"),
    }
}

#[tokio::test]
async fn cancelled_agent_explore_records_tool_output_before_interrupting_turn() {
    let mut agent = test_agent();
    agent.set_subagent_delegate(static_delegate(ToolResult::err_with_data(
        "agent__explore",
        "explorer cancelled",
        json!({
            "run_id": "run-1",
            "child_session_id": "child-session",
            "agent_name": "explorer",
            "status": "cancelled",
            "summary": "explorer cancelled",
        }),
    )));
    let call = test_tool_call("agent__explore", r#"{"task":"inspect"}"#);
    agent
        .append_assistant_tool_calls("", std::slice::from_ref(&call))
        .expect("assistant tool calls should append");
    let mut events = Vec::new();

    let error = agent
        .execute_tool_call_and_record(
            &call,
            &mut |event| {
                events.push(event);
                std::future::ready(Ok(()))
            },
            &mut |_| std::future::ready(Ok(PermissionApproval::AllowOnce)),
        )
        .await
        .expect_err("cancelled explorer interrupts the turn after recording output");

    assert!(error.to_string().contains("agent__explore cancelled"));
    assert!(matches!(
        agent.history.last(),
        Some(HistoryItem::ToolOutput {
            call_id,
            output_json,
        }) if call_id == "call-agent__explore"
            && output_json.contains("cancelled")
            && output_json.contains("child-session")
    ));
    assert!(events.iter().any(|event| matches!(
        event,
        AgentEvent::ToolCallFinished {
            name,
            ok: false,
            output,
            ..
        } if name == "agent__explore"
            && output
                .data
                .as_ref()
                .and_then(|data| data.get("status"))
                .and_then(Value::as_str)
                == Some("cancelled")
    )));
}

#[tokio::test]
async fn cancelled_agent_fixer_records_tool_output_before_interrupting_turn() {
    let mut agent = test_agent();
    agent.set_subagent_delegate(static_delegate(ToolResult::err_with_data(
        "agent__fixer",
        "fixer cancelled",
        json!({
            "run_id": "run-1",
            "child_session_id": "child-session",
            "agent_name": "fixer",
            "status": "cancelled",
            "summary": "fixer cancelled",
        }),
    )));
    let call = test_tool_call("agent__fixer", r#"{"task":"apply requested fix"}"#);
    agent
        .append_assistant_tool_calls("", std::slice::from_ref(&call))
        .expect("assistant tool calls should append");
    let mut events = Vec::new();

    let error = agent
        .execute_tool_call_and_record(
            &call,
            &mut |event| {
                events.push(event);
                std::future::ready(Ok(()))
            },
            &mut |_| std::future::ready(Ok(PermissionApproval::AllowOnce)),
        )
        .await
        .expect_err("cancelled fixer interrupts the turn after recording output");

    assert!(error.to_string().contains("agent__fixer cancelled"));
    assert!(matches!(
        agent.history.last(),
        Some(HistoryItem::ToolOutput {
            call_id,
            output_json,
        }) if call_id == "call-agent__fixer"
            && output_json.contains("cancelled")
            && output_json.contains("child-session")
    ));
    assert!(events.iter().any(|event| matches!(
        event,
        AgentEvent::ToolCallFinished {
            name,
            ok: false,
            output,
            ..
        } if name == "agent__fixer"
            && output
                .data
                .as_ref()
                .and_then(|data| data.get("status"))
                .and_then(Value::as_str)
                == Some("cancelled")
    )));
}

#[tokio::test]
async fn delegated_structured_subagent_results_surface_in_next_turn_prelude() {
    let mut agent = test_agent();
    agent.prepare_turn_prelude("Delegate implementation work");
    agent.set_subagent_delegate(static_delegate(ToolResult::ok(
        "agent__fixer",
        json!({
            "run_id": "run-structured-1",
            "child_session_id": "child-structured-1",
            "agent_name": "fixer",
            "status": "completed",
            "summary": "implemented bounded fix",
            "structured_result": {
                "status": "completed",
                "summary": "implemented bounded fix",
                "malformed": false,
                "findings": [],
                "files_read": ["src/agent.rs"],
                "files_changed": ["src/agent.rs"],
                "commands_run": ["cargo test subagent --quiet"],
                "validation": ["cargo test subagent --quiet passed"],
                "blockers": [],
                "next_steps": ["reconcile in parent turn"],
                "run_id": "run-structured-1",
                "child_session_id": "child-structured-1"
            }
        }),
    )));

    let call = test_tool_call("agent__fixer", r#"{"task":"implement bounded fix"}"#);
    agent
        .append_assistant_tool_calls("", std::slice::from_ref(&call))
        .expect("assistant tool calls should append");
    let mut events = Vec::new();
    agent
        .execute_tool_call_and_record(
            &call,
            &mut |event| {
                events.push(event);
                std::future::ready(Ok(()))
            },
            &mut |_| std::future::ready(Ok(PermissionApproval::AllowOnce)),
        )
        .await
        .expect("subagent tool execution should succeed");

    assert!(events.iter().any(|event| matches!(
        event,
        AgentEvent::EvidenceRecorded(record)
            if record.tags.iter().any(|tag| tag == "subagent_result")
                && record.tags.iter().any(|tag| tag == "unreconciled")
    )));

    let jobs = agent.pending_subagent_jobs();
    assert_eq!(jobs.len(), 1);
    assert_eq!(jobs[0].agent_name, "fixer");
    assert_eq!(jobs[0].run_id, "run-structured-1");
    assert_eq!(jobs[0].child_session_id, "child-structured-1");
    assert_eq!(jobs[0].summary, "implemented bounded fix");

    let prelude = agent.prepare_turn_prelude("Reconcile child work");
    assert!(prelude.iter().any(|message| {
        message.text.contains("Pending child subagent results")
            && message.text.contains("run-structured-1")
            && message.text.contains("implemented bounded fix")
            && message.text.contains("child-structured-1")
            && message.text.contains("child_session_id=child-structured-1")
            && message.text.contains("child transcript navigation")
    }));
}

#[test]
fn model_switch_uses_new_metadata_for_next_request_build() {
    let client = Client::with_config(
        OpenAIConfig::new()
            .with_api_base("https://api.openai.com/v1")
            .with_api_key("test"),
    );
    let mut agent = Agent::new(client, "m1", 1, 1);

    let mut catalog = HashMap::new();
    catalog.insert(
        "m1".to_string(),
        ModelRequestMetadata {
            context_window: Some(2048),
            max_output_tokens: Some(256),
            supports_tools: true,
            supports_reasoning: false,
            ..Default::default()
        },
    );
    catalog.insert(
        "m2".to_string(),
        ModelRequestMetadata {
            context_window: Some(128_000),
            max_output_tokens: Some(256),
            supports_tools: true,
            supports_reasoning: false,
            ..Default::default()
        },
    );
    agent.set_model_catalog(catalog);

    // Simulate first user message.
    agent
        .append_history_item(HistoryItem::user("hello"))
        .expect("history append succeeds");
    let history = agent.history_items();
    let b1 = build_test_request(TestRequestBuilderInput {
        protocol: ApiProtocol::Responses,
        model_id: agent.model(),
        model: agent.active_model_metadata(),
        prelude: &agent.prelude,
        history: &history,
        protected_start_index: history.len().saturating_sub(1),
        tools: &[],
        evidence: &[],
        history_adapter: None,
        context_view: None,
    })
    .expect("request builds");
    assert_eq!(b1.budget.context_window_tokens, 2048.max(1024));

    // Switch model and build again.
    agent.set_model("m2");
    let history = agent.history_items();
    let b2 = build_test_request(TestRequestBuilderInput {
        protocol: ApiProtocol::Responses,
        model_id: agent.model(),
        model: agent.active_model_metadata(),
        prelude: &agent.prelude,
        history: &history,
        protected_start_index: history.len().saturating_sub(1),
        tools: &[],
        evidence: &[],
        history_adapter: None,
        context_view: None,
    })
    .expect("request builds");
    assert!(b2.budget.context_window_tokens > b1.budget.context_window_tokens);
}

#[test]
fn inline_reasoning_extractor_splits_think_tags_from_visible_text() {
    let mut extractor = InlineReasoningExtractor::new("r-1");

    let mut parts = extractor.push("hello <thi");
    parts.extend(extractor.push("nk>plan</think> answer"));
    parts.extend(extractor.finish());

    assert_eq!(
        parts,
        vec![
            StreamTextPart::Visible("hello ".into()),
            StreamTextPart::ReasoningDelta {
                item_id: "r-1".into(),
                delta: "plan".into(),
            },
            StreamTextPart::ReasoningDone {
                item_id: "r-1".into(),
                text: "plan".into(),
            },
            StreamTextPart::Visible(" answer".into()),
        ]
    );
}

#[test]
fn compatible_chat_delta_reads_native_reasoning_fields() {
    for (field, expected) in [
        ("reasoning_content", "plan"),
        ("reasoning", "think"),
        ("thinking", "ponder"),
    ] {
        let raw = serde_json::json!({
            "content": null,
            field: expected,
        });
        let delta: CompatibleChatCompletionStreamResponseDelta =
            serde_json::from_value(raw).expect("delta deserializes");

        assert_eq!(delta.reasoning_delta().as_deref(), Some(expected));
    }
}

#[test]
fn protocol_frames_remain_authoritative_for_history_cache() {
    let mut agent = test_agent();
    agent
        .append_history_item(HistoryItem::user("hello"))
        .expect("user append succeeds");
    agent
        .append_history_item(HistoryItem::AssistantToolCalls {
            text: Some("working".into()),
            calls: vec![test_tool_call("fs__read", r#"{"path":"src/main.rs"}"#)],
        })
        .expect("tool call append succeeds");
    agent
        .append_history_item(HistoryItem::ToolOutput {
            call_id: "call-fs__read".into(),
            output_json: r#"{"ok":true}"#.into(),
        })
        .expect("tool output append succeeds");

    assert_eq!(
        crate::protocol_frames::history_items_from_frames(agent.protocol_frames_for_test()),
        agent.history_for_test()
    );
    assert_eq!(
        agent.runtime_snapshot.frames.len(),
        agent.protocol_frames_for_test().len()
    );
}

#[test]
fn append_history_item_is_atomic_when_protocol_validation_fails() {
    let mut agent = test_agent();
    agent
        .append_history_item(HistoryItem::user("hello"))
        .expect("user append succeeds");

    let history_before = agent.history.clone();
    let frames_before = agent.protocol_frames.clone();
    let snapshot_before = agent.runtime_snapshot.clone();

    let error = agent
        .append_history_item(HistoryItem::ToolOutput {
            call_id: "call-orphan".into(),
            output_json: "{}".into(),
        })
        .expect_err("orphan tool output must fail");

    assert!(error.to_string().contains("orphan tool output"));
    assert_eq!(agent.history, history_before);
    assert_eq!(agent.protocol_frames, frames_before);
    assert_eq!(agent.runtime_snapshot, snapshot_before);
}

#[test]
fn compatible_chat_delta_reads_object_and_array_reasoning() {
    let raw = serde_json::json!({
        "reasoning_content": [
            {"text": "step "},
            {"content": "one"}
        ]
    });
    let delta: CompatibleChatCompletionStreamResponseDelta =
        serde_json::from_value(raw).expect("delta deserializes");

    assert_eq!(delta.reasoning_delta().as_deref(), Some("step one"));
}

#[test]
fn compatible_chat_stream_accepts_terminal_chunk_without_delta() {
    let raw = serde_json::json!({
        "id": "chatcmpl-test",
        "object": "chat.completion.chunk",
        "created": 1780856440_u64,
        "model": "gpt-5.5",
        "choices": [{
            "index": 0,
            "finish_reason": "stop"
        }],
        "usage": {
            "prompt_tokens": 3060,
            "completion_tokens": 25,
            "total_tokens": 3085
        }
    });

    let response: CompatibleChatCompletionStreamResponse =
        serde_json::from_value(raw).expect("terminal chunk deserializes");

    assert_eq!(response.choices.len(), 1);
    assert!(response.choices[0].delta.is_none());
    assert_eq!(response.choices[0].finish_reason, Some(FinishReason::Stop));
    assert_eq!(
        response.usage.as_ref().map(|usage| usage.prompt_tokens),
        Some(3060)
    );
    assert_eq!(
        response.usage.as_ref().map(|usage| usage.completion_tokens),
        Some(25)
    );
}

#[test]
fn sse_parser_drains_data_events_and_done_marker() {
    let mut buffer = String::new();
    append_sse_chunk(&mut buffer, b"data: {\"choices\":[]}\n\ndata: [DONE]\n\n");

    assert_eq!(
        drain_sse_data_events(&mut buffer),
        vec![Some(r#"{"choices":[]}"#.into()), None]
    );
    assert!(buffer.is_empty());
}

#[test]
fn ignores_non_terminal_lifecycle_events_missing_model() {
    for event_type in ["response.created", "response.in_progress"] {
        let raw = serde_json::json!({
            "type": event_type,
            "sequence_number": 1,
            "response": {
                "id": "resp_test",
                "object": "response",
                "created_at": 1780765723_u64,
                "status": "in_progress",
                "background": false,
                "error": null,
                "output": []
            }
        });
        assert!(
            is_ignorable_response_lifecycle_event(&raw),
            "{event_type} should be ignored"
        );
    }
}

#[test]
fn does_not_ignore_other_stream_deserialize_errors() {
    let raw = serde_json::json!({
        "type": "response.completed",
        "sequence_number": 1,
        "response": {
            "id": "resp_test",
            "object": "response",
            "created_at": 1780765723_u64,
            "status": "completed",
            "background": false,
            "error": null,
            "output": []
        }
    });
    assert!(!is_ignorable_response_lifecycle_event(&raw));
}

#[test]
fn projects_provider_reasoning_efforts_without_mutating_completed_events() {
    for effort in ["max", "provider-ultra"] {
        let raw = serde_json::json!({
            "type": "response.completed", "sequence_number": 1,
            "response": {
                "id": "resp_test", "object": "response", "created_at": 1780765723_u64,
                "status": "completed", "background": false, "error": null,
                "incomplete_details": null, "instructions": null, "max_output_tokens": null,
                "model": "m1", "output": [], "parallel_tool_calls": true,
                "previous_response_id": null, "reasoning": {"effort": effort}, "store": true,
                "temperature": 1, "text": {"format": {"type": "text"}}, "tool_choice": "auto",
                "tools": [], "top_p": 1, "truncation": "disabled",
                "usage": {"input_tokens": 5, "input_tokens_details": {"cached_tokens": 0}, "output_tokens": 3, "output_tokens_details": {"reasoning_tokens": 0}, "total_tokens": 8},
                "user": null, "metadata": {}
            }
        });
        let event = project_response_stream_event(&raw).expect("completed event should project");
        let ResponseStreamEvent::ResponseCompleted(event) = event else {
            panic!("expected completion")
        };
        assert_eq!(event.response.id, "resp_test");
        assert!(event.response.output.is_empty());
        assert_eq!(event.response.usage.expect("usage").total_tokens, 8);
        assert_eq!(raw["response"]["reasoning"]["effort"], effort);
    }
}

#[test]
fn rejects_malformed_completed_events_after_projection() {
    let raw = serde_json::json!({"type": "response.completed", "response": {"reasoning": {"effort": "max"}}});
    assert!(project_response_stream_event(&raw).is_err());
}

#[test]
fn compact_indexed_chat_tool_calls_does_not_synthesize_missing_indices() {
    let mut indexed = BTreeMap::new();
    let mut call = ChatCompletionMessageToolCall::default();
    call.id = "call-1".into();
    call.function.name = "fs__write".into();
    call.function.arguments = r#"{"path":"a.txt","content":"ok"}"#.into();
    indexed.insert(1, call);

    let compacted = compact_indexed_chat_tool_calls(indexed);

    assert_eq!(compacted.len(), 1);
    assert_eq!(compacted[0].id, "call-1");
    assert_eq!(compacted[0].function.name, "fs__write");
    validate_chat_tool_calls(&compacted).expect("valid sparse-index tool call");
}

#[test]
fn chat_tool_call_chunk_empty_name_does_not_overwrite_real_name() {
    let mut indexed = BTreeMap::new();
    for raw in [
        serde_json::json!({
            "index": 0,
            "id": "call-1",
            "type": "function",
            "function": {"name": "fs__write", "arguments": ""}
        }),
        serde_json::json!({
            "index": 0,
            "function": {"name": "", "arguments": "{\"path\":"}
        }),
        serde_json::json!({
            "index": 0,
            "function": {"name": "", "arguments": "\"a.txt\",\"content\":\"ok\"}"}
        }),
    ] {
        let chunk: ChatCompletionMessageToolCallChunk =
            serde_json::from_value(raw).expect("chunk deserializes");
        merge_chat_tool_call_chunk(&mut indexed, chunk);
    }

    let compacted = compact_indexed_chat_tool_calls(indexed);

    assert_eq!(compacted.len(), 1);
    assert_eq!(compacted[0].id, "call-1");
    assert_eq!(compacted[0].function.name, "fs__write");
    assert_eq!(
        compacted[0].function.arguments,
        r#"{"path":"a.txt","content":"ok"}"#
    );
    validate_chat_tool_calls(&compacted).expect("valid streamed tool call");
}

#[test]
fn classifies_lightweight_and_engineering_turns() {
    assert_eq!(
        classify_turn_intent("Explain how Rust ownership works."),
        TurnIntent::Lightweight
    );
    assert_eq!(
        classify_turn_intent("Explain what this function does."),
        TurnIntent::Lightweight
    );
    assert_eq!(
        classify_turn_intent(
            "Fix the failing tests in src/agent.rs and update the implementation."
        ),
        TurnIntent::Engineering
    );
}

#[test]
fn prepare_turn_prelude_assigns_incrementing_turn_ids() {
    let client = Client::with_config(
        OpenAIConfig::new()
            .with_api_base("https://api.openai.com/v1")
            .with_api_key("test"),
    );
    let mut agent = Agent::new(client, "m1", 1, 1);

    agent.prepare_turn_prelude("first turn");
    assert_eq!(agent.current_turn_id(), 1);

    agent.prepare_turn_prelude("second turn");
    assert_eq!(agent.current_turn_id(), 2);
}

#[test]
fn restore_session_context_seeds_next_turn_id() {
    let mut agent = test_agent();

    agent
        .restore_session_context(Vec::new(), Vec::new(), 7)
        .expect("restore session context");
    agent.prepare_turn_prelude("resumed turn");

    assert_eq!(agent.current_turn_id(), 8);
}

#[test]
fn candidate_session_usage_failure_leaves_live_agent_unchanged() {
    let mut agent = test_agent();
    agent
        .restore_session_history(vec![HistoryItem::user("old session")], Vec::new(), 7)
        .expect("restore old session");
    agent.prepare_turn_prelude("active turn");
    let mut invalid_metadata = ModelRequestMetadata::default();
    invalid_metadata.effective_input_limit_tokens = Some(0);
    agent.set_model_catalog(HashMap::from([(
        String::from("invalid-model"),
        invalid_metadata,
    )]));
    let target_snapshot = runtime_snapshot_for_history(
        ROOT_CONTEXT_BRANCH_ID,
        &[HistoryItem::user("target session")],
    );
    let model = agent.model.clone();
    let history = agent.history.clone();
    let protocol_frames = agent.protocol_frames.clone();
    let runtime_snapshot = agent.runtime_snapshot.clone();
    let turn_id = agent.current_turn_id();
    let next_turn_id = agent.next_turn_id;

    let error = agent
        .candidate_session_token_usage("invalid-model", &target_snapshot)
        .expect_err("invalid target metadata must fail");

    assert!(
        error
            .to_string()
            .contains("effective_input_limit_tokens must be greater than 0")
    );
    assert_eq!(agent.model, model);
    assert_eq!(agent.history, history);
    assert_eq!(agent.protocol_frames, protocol_frames);
    assert_eq!(agent.runtime_snapshot, runtime_snapshot);
    assert_eq!(agent.current_turn_id(), turn_id);
    assert_eq!(agent.next_turn_id, next_turn_id);
}

#[test]
fn restore_runtime_snapshot_keeps_projected_runtime_state_authoritative() {
    let mut agent = test_agent();
    let history = vec![
        HistoryItem::user("resume question"),
        HistoryItem::assistant("resume answer"),
    ];
    let frames = crate::protocol_frames::history_items_to_frames(&history);
    let mut snapshot = RuntimeSnapshot::new("feature")
        .with_session_id("session-1")
        .with_latest_model("m1")
        .with_leaf_sequence(12)
        .with_current_turn_id(7);
    snapshot.frames = runtime_frames_for_history(&history);
    snapshot.set_evidence(vec![EvidenceRecord {
        id: "evidence-1".into(),
        sequence: 1,
        timestamp_ms: 1,
        evidence_kind: crate::evidence::EvidenceKind::Decision,
        title: "Restored evidence".into(),
        summary: "restored evidence".into(),
        detail: None,
        source: EvidenceSource::Transcript { sequence: 1 },
        tags: Vec::new(),
    }]);

    agent
        .restore_runtime_snapshot(frames.clone(), snapshot.clone())
        .expect("restore runtime snapshot");

    assert_eq!(
        agent.protocol_frames_for_test(),
        snapshot.active_protocol_frames().as_slice()
    );
    assert_eq!(
        agent.history_for_test(),
        crate::protocol_frames::history_items_from_frames(&frames).as_slice()
    );
    assert_eq!(agent.runtime_snapshot_for_test(), &snapshot);
    assert_eq!(agent.evidence(), snapshot.evidence.as_slice());
    agent.prepare_turn_prelude("continued turn");
    assert_eq!(agent.current_turn_id(), 8);
}

#[test]
fn compatibility_rebuilds_preserve_restored_turn_id_without_an_active_turn() {
    let mut agent = test_agent();
    let snapshot = RuntimeSnapshot::new(ROOT_CONTEXT_BRANCH_ID).with_current_turn_id(7);
    agent
        .restore_runtime_snapshot(Vec::new(), snapshot)
        .expect("restore runtime snapshot");
    agent.compaction_config.prune = true;
    agent.compaction_config.tail_turns = 1;

    agent
        .replace_history(vec![
            HistoryItem::user("older turn"),
            HistoryItem::AssistantToolCalls {
                text: None,
                calls: vec![HistoryToolCall {
                    call_id: "call-read".into(),
                    name: "fs__read".into(),
                    arguments_json: r#"{"path":"src/lib.rs"}"#.into(),
                }],
            },
            HistoryItem::ToolOutput {
                call_id: "call-read".into(),
                output_json: prunable_tool_output_json("stdout"),
            },
            HistoryItem::assistant(prune_protect_padding()),
            HistoryItem::user("recent turn"),
            HistoryItem::assistant("recent reply"),
            HistoryItem::user("current turn"),
        ])
        .expect("compatibility replacement succeeds");
    assert_eq!(agent.runtime_snapshot_for_test().current_turn_id, Some(7));

    agent
        .prune_old_tool_outputs(4_000)
        .expect("pruning succeeds");
    assert_eq!(agent.runtime_snapshot_for_test().current_turn_id, Some(7));

    agent
        .append_history_item(HistoryItem::context_summary("restored summary"))
        .expect("summary append succeeds");
    assert_eq!(agent.runtime_snapshot_for_test().current_turn_id, Some(7));
}

#[test]
fn new_session_reset_discards_restored_runtime_metadata() {
    let mut agent = test_agent();
    let history = vec![HistoryItem::user("old prompt")];
    let mut snapshot = RuntimeSnapshot::new("feature")
        .with_session_id("old-session")
        .with_latest_model("old-model")
        .with_leaf_sequence(12)
        .with_current_turn_id(7);
    snapshot.frames = runtime_frames_for_history(&history);
    agent
        .restore_runtime_snapshot(
            crate::protocol_frames::history_items_to_frames(&history),
            snapshot,
        )
        .expect("restore runtime snapshot");

    agent.reset_for_new_session();

    assert!(agent.history_for_test().is_empty());
    assert!(agent.protocol_frames_for_test().is_empty());
    assert!(agent.evidence().is_empty());
    assert_eq!(
        agent.runtime_snapshot_for_test(),
        &RuntimeSnapshot::new(ROOT_CONTEXT_BRANCH_ID).with_latest_model("m1")
    );
    agent.prepare_turn_prelude("fresh turn");
    assert_eq!(agent.current_turn_id(), 1);
}

#[test]
fn compaction_selection_preserves_recent_tail_and_reuses_previous_summary() {
    let history = vec![
        HistoryItem::context_summary("旧摘要"),
        HistoryItem::user("turn-1 user"),
        HistoryItem::assistant("turn-1 assistant"),
        HistoryItem::user("turn-2 user"),
        HistoryItem::assistant("turn-2 assistant"),
        HistoryItem::user("current user"),
    ];

    let selection = select_compaction_segments(
        &history,
        5,
        &CompactionConfig {
            tail_turns: 1,
            ..CompactionConfig::default()
        },
        4_000,
    )
    .expect("selection succeeds");

    assert_eq!(selection.previous_summary.as_deref(), Some("旧摘要"));
    assert_eq!(selection.head_for_summary.len(), 4);
    assert!(selection.tail_items.is_empty());
    assert_eq!(selection.tail_start_index, 5);
}

#[tokio::test]
async fn manual_compaction_noops_when_history_is_empty() {
    let mut agent = test_agent();

    let outcome = agent
        .compact_session_async(|_| async { Ok(()) })
        .await
        .expect("manual compaction should not fail");

    assert_eq!(outcome, ManualCompactionOutcome::NothingToCompact);
}

#[tokio::test]
async fn manual_compaction_noops_when_only_recent_tail_exists() {
    let mut agent = test_agent();
    agent
        .replace_history(vec![
            HistoryItem::user("short prompt"),
            HistoryItem::assistant("reply"),
        ])
        .expect("history replace succeeds");

    let outcome = agent
        .compact_session_async(|_| async { Ok(()) })
        .await
        .expect("manual compaction should not fail");

    assert_eq!(outcome, ManualCompactionOutcome::NothingToCompact);
    assert_eq!(
        agent.history,
        vec![
            HistoryItem::user("short prompt"),
            HistoryItem::assistant("reply")
        ]
    );
}

#[test]
fn compaction_selection_never_summarizes_protected_current_turn() {
    let history = vec![
        HistoryItem::user("old user"),
        HistoryItem::assistant("old assistant"),
        HistoryItem::user("current user"),
        HistoryItem::assistant("current assistant"),
    ];

    let selection = select_compaction_segments(
        &history,
        2,
        &CompactionConfig {
            tail_turns: 0,
            preserve_recent_tokens: Some(0),
            ..CompactionConfig::default()
        },
        0,
    )
    .expect("selection succeeds");

    assert_eq!(selection.head_for_summary.len(), 2);
    assert!(selection.tail_items.is_empty());
    assert_eq!(
        &history[2..],
        &[
            HistoryItem::user("current user"),
            HistoryItem::assistant("current assistant")
        ]
    );
}

#[test]
fn latest_item_over_budget_does_not_force_tail_retention() {
    let history = vec![
        HistoryItem::user("older user"),
        HistoryItem::assistant("older assistant"),
        HistoryItem::user("x".repeat(15_000)),
    ];

    let selection = select_compaction_segments(
        &history,
        3,
        &CompactionConfig {
            tail_turns: 1,
            ..CompactionConfig::default()
        },
        10,
    )
    .expect("selection succeeds");

    assert!(selection.tail_items.is_empty());
    assert_eq!(selection.head_for_summary.len(), 3);
    assert_eq!(selection.tail_start_index, 3);
}

#[test]
fn oversized_latest_turn_can_keep_suffix_that_fits_budget() {
    let suffix = HistoryItem::assistant("small suffix");
    let history = vec![
        HistoryItem::user("older user"),
        HistoryItem::assistant("older assistant"),
        HistoryItem::user("x".repeat(15_000)),
        suffix.clone(),
    ];

    let selection = select_compaction_segments(
        &history,
        4,
        &CompactionConfig {
            tail_turns: 1,
            ..CompactionConfig::default()
        },
        estimate_history_item_tokens(&suffix),
    )
    .expect("selection succeeds");

    assert_eq!(selection.tail_items, vec![suffix]);
    assert_eq!(selection.head_for_summary.len(), 3);
    assert_eq!(selection.tail_start_index, 3);
}

#[test]
fn compaction_tail_does_not_start_with_orphan_tool_output() {
    let tool_output = HistoryItem::ToolOutput {
        call_id: "call-read".into(),
        output_json: r#"{"ok":true}"#.into(),
    };
    let history = vec![
        HistoryItem::user("older user"),
        HistoryItem::assistant("older assistant"),
        HistoryItem::user("inspect file"),
        HistoryItem::AssistantToolCalls {
            text: None,
            calls: vec![test_tool_call("read", r#"{"path":"src/main.rs"}"#)],
        },
        tool_output.clone(),
    ];

    let selection = select_compaction_segments(
        &history,
        history.len(),
        &CompactionConfig {
            tail_turns: 1,
            ..CompactionConfig::default()
        },
        estimate_history_item_tokens(&tool_output),
    )
    .expect("selection succeeds");

    assert!(matches!(
        selection.tail_items.first(),
        Some(HistoryItem::AssistantToolCalls { .. })
    ));
    assert!(matches!(
        selection.tail_items.get(1),
        Some(HistoryItem::ToolOutput { call_id, .. }) if call_id == "call-read"
    ));
}

#[test]
fn default_preserve_recent_budget_uses_quarter_clamped_range() {
    assert_eq!(default_preserve_recent_budget(1_000), 1_000);
    assert_eq!(default_preserve_recent_budget(12_000), 3_000);
    assert_eq!(default_preserve_recent_budget(100_000), 8_000);
}

#[test]
fn compaction_history_char_budget_scales_with_model_window() {
    let small = compaction_history_char_budget(ModelRequestMetadata {
        context_window: Some(1_024),
        max_output_tokens: Some(128),
        ..ModelRequestMetadata::default()
    });
    let large = compaction_history_char_budget(ModelRequestMetadata {
        context_window: Some(128_000),
        max_output_tokens: Some(4_096),
        ..ModelRequestMetadata::default()
    });

    assert!(small <= 1_000);
    assert!(large > small);
    assert!(large <= COMPACTION_HISTORY_MAX_CHAR_BUDGET);
}

#[test]
fn compaction_history_char_budget_uses_effective_input_limit() {
    let uncapped = compaction_history_char_budget(ModelRequestMetadata {
        context_window: Some(128_000),
        max_output_tokens: Some(4_096),
        ..ModelRequestMetadata::default()
    });
    let capped = compaction_history_char_budget(ModelRequestMetadata {
        context_window: Some(128_000),
        effective_input_limit_tokens: Some(4_000),
        max_output_tokens: Some(4_096),
        ..ModelRequestMetadata::default()
    });

    assert!(capped < uncapped);
    assert!(capped <= 4_000);
}

#[test]
fn render_compaction_prompt_distinguishes_initial_and_incremental_modes() {
    let items = vec![HistoryItem::user("修复 src/agent.rs")];

    let initial = render_compaction_prompt(None, &items, 16_000);
    assert!(initial.contains("生成新的锚定摘要"));
    assert!(!initial.contains("更新已有锚定摘要"));

    let incremental = render_compaction_prompt(Some("已有摘要"), &items, 16_000);
    assert!(incremental.contains("更新已有锚定摘要"));
    assert!(incremental.contains("删除已过时或被推翻的信息"));
}

#[test]
fn render_compaction_tool_output_caps_large_payloads() {
    let rendered = describe_history_item(&HistoryItem::ToolOutput {
        call_id: "call-big".into(),
        output_json: large_tool_output_json("stdout"),
    });

    assert!(rendered.contains(COMPACTION_TOOL_OUTPUT_TRUNCATION_MARKER));
    assert!(rendered.chars().count() < 2_200);
}

#[test]
fn render_compaction_tool_output_strips_media_like_fields() {
    let base64 = "A".repeat(3_000);
    let rendered = describe_history_item(&HistoryItem::ToolOutput {
        call_id: "call-media".into(),
        output_json: json!({
            "image_base64": base64,
            "preview_url": "blob:https://example.invalid/123",
            "stdout": "kept text"
        })
        .to_string(),
    });

    assert!(rendered.contains("stripped media/blob-like field"));
    assert!(rendered.contains("kept text"));
    assert!(!rendered.contains("blob:https://example.invalid/123"));
    assert!(!rendered.contains(&"A".repeat(128)));
}

#[test]
fn render_compaction_prompt_applies_total_history_cap() {
    let items = (0..20)
        .map(|index| HistoryItem::ToolOutput {
            call_id: format!("call-{index}"),
            output_json: large_tool_output_json("stdout"),
        })
        .collect::<Vec<_>>();

    let rendered = render_bounded_compaction_history(&items, 4_000);

    assert!(rendered.contains(COMPACTION_HISTORY_TRUNCATION_MARKER));
    assert!(rendered.chars().count() <= 4_000);
    assert!(rendered.contains("call-19"));
    assert!(!rendered.contains("call-0"));
}

#[test]
fn prune_old_tool_outputs_protects_source_less_payloads() {
    let mut agent = test_agent();
    agent.compaction_config.prune = true;
    agent.compaction_config.tail_turns = 1;
    let prunable_output = prunable_tool_output_json("stdout");
    agent
        .replace_history(vec![
            HistoryItem::user("older turn"),
            HistoryItem::AssistantToolCalls {
                text: None,
                calls: vec![HistoryToolCall {
                    call_id: "call-read".into(),
                    name: "fs__read".into(),
                    arguments_json: r#"{"path":"src/lib.rs"}"#.into(),
                }],
            },
            HistoryItem::ToolOutput {
                call_id: "call-read".into(),
                output_json: prunable_output.clone(),
            },
            HistoryItem::assistant(prune_protect_padding()),
            HistoryItem::user("recent turn"),
            HistoryItem::assistant("recent reply"),
            HistoryItem::user("current turn"),
        ])
        .expect("history replace succeeds");

    agent
        .prune_old_tool_outputs(4_000)
        .expect("pruning succeeds");

    let HistoryItem::ToolOutput { output_json, .. } = &agent.history[2] else {
        panic!("expected tool output");
    };
    assert_eq!(output_json, &prunable_output);
    assert_eq!(
        crate::protocol_frames::history_items_from_frames(agent.protocol_frames_for_test()),
        agent.history_for_test()
    );
    assert_eq!(
        agent.runtime_snapshot.frames.len(),
        agent.protocol_frames_for_test().len()
    );
}

#[test]
fn prune_old_tool_outputs_skips_recent_and_skill_payloads() {
    let mut agent = test_agent();
    agent.compaction_config.prune = true;
    agent.compaction_config.tail_turns = 1;
    let skill_output = prunable_tool_output_json("result");
    let recent_output = prunable_tool_output_json("stdout");
    agent
        .replace_history(vec![
            HistoryItem::user("older turn"),
            HistoryItem::AssistantToolCalls {
                text: None,
                calls: vec![HistoryToolCall {
                    call_id: "call-skill".into(),
                    name: "skill".into(),
                    arguments_json: r#"{"name":"rust-audit"}"#.into(),
                }],
            },
            HistoryItem::ToolOutput {
                call_id: "call-skill".into(),
                output_json: skill_output.clone(),
            },
            HistoryItem::assistant(prune_protect_padding()),
            HistoryItem::user("recent turn"),
            HistoryItem::AssistantToolCalls {
                text: None,
                calls: vec![HistoryToolCall {
                    call_id: "call-recent".into(),
                    name: "fs__read".into(),
                    arguments_json: r#"{"path":"src/main.rs"}"#.into(),
                }],
            },
            HistoryItem::ToolOutput {
                call_id: "call-recent".into(),
                output_json: recent_output.clone(),
            },
            HistoryItem::user("current turn"),
        ])
        .expect("history replace succeeds");

    agent
        .prune_old_tool_outputs(4_000)
        .expect("pruning succeeds");

    let HistoryItem::ToolOutput {
        output_json: skill_after,
        ..
    } = &agent.history[2]
    else {
        panic!("expected skill tool output");
    };
    let HistoryItem::ToolOutput {
        output_json: recent_after,
        ..
    } = &agent.history[6]
    else {
        panic!("expected recent tool output");
    };
    assert_eq!(skill_after, &skill_output);
    assert_eq!(recent_after, &recent_output);
    assert_eq!(
        crate::protocol_frames::history_items_from_frames(agent.protocol_frames_for_test()),
        agent.history_for_test()
    );
    assert_eq!(
        agent.runtime_snapshot.frames.len(),
        agent.protocol_frames_for_test().len()
    );
}

#[test]
fn context_checkpoint_restore_point_keeps_complete_tool_call_group() {
    let mut agent = test_agent();
    let call = test_tool_call(
        tool_names::TOOL_CONTEXT_CHECKPOINT,
        r#"{"label":"alt","reason":"try alternative approach"}"#,
    );
    agent
        .append_assistant_tool_calls("", std::slice::from_ref(&call))
        .expect("checkpoint-only tool batch should append");
    agent
        .append_history_item(HistoryItem::ToolOutput {
            call_id: call.call_id.clone(),
            output_json: json!({
                "label": "alt",
                "reason": "try alternative approach"
            })
            .to_string(),
        })
        .expect("tool output append succeeds");
    let scope = ActiveContextExperiment {
        branch_id: "branch-1".into(),
        parent_branch_id: ROOT_CONTEXT_BRANCH_ID.into(),
        base_sequence: 4,
        writes_observed: false,
    };
    agent.set_context_scope_state(Arc::new(std::sync::Mutex::new(ContextScopeState {
        active_experiment: Some(scope.clone()),
    })));

    agent.set_runtime_snapshot_provider(Arc::new(|| Ok(RuntimeSnapshot::new("branch-1"))));

    agent
        .finalize_context_checkpoint_after_recording()
        .expect("checkpoint finalize succeeds");

    let restore = agent
        .context_experiment_restore_point
        .as_ref()
        .expect("restore point stored");
    let restore_history =
        crate::protocol_frames::history_items_from_frames(&restore.protocol_frames);
    crate::protocol_frames::validate_history_items_complete(&restore_history, None)
        .expect("restore history remains protocol-complete");
    assert!(matches!(
        restore_history.last(),
        Some(HistoryItem::ToolOutput { call_id, .. }) if call_id == &call.call_id
    ));
    assert_eq!(
        restore.runtime_snapshot.evidence,
        agent.runtime_snapshot.evidence
    );
    assert_eq!(
        restore.runtime_snapshot.current_turn_id,
        Some(agent.next_turn_id)
    );
}

#[tokio::test]
async fn context_return_records_output_before_restoring_parent_context() {
    let mut agent = test_agent();
    agent
        .append_history_item(HistoryItem::user("hello"))
        .expect("seed history");

    let restore_history = agent.history.clone();
    let restore_turn_id = agent.next_turn_id;
    let scope = ActiveContextExperiment {
        branch_id: "branch-1".into(),
        parent_branch_id: ROOT_CONTEXT_BRANCH_ID.into(),
        base_sequence: 1,
        writes_observed: false,
    };
    agent.set_context_scope_state(Arc::new(std::sync::Mutex::new(ContextScopeState {
        active_experiment: Some(scope.clone()),
    })));
    agent.set_context_experiment_restore_point(
        scope,
        crate::protocol_frames::history_items_to_frames(&restore_history),
        runtime_snapshot_for_history(ROOT_CONTEXT_BRANCH_ID, &restore_history)
            .with_current_turn_id(restore_turn_id),
    );
    let returned_summary =
        HistoryItem::context_summary(crate::transcript::format_context_experiment_return(
            "branch-1",
            "useful",
            "Found the issue",
            Some("Apply fix"),
            false,
        ));
    let parent_history = vec![restore_history[0].clone(), returned_summary];
    agent.set_runtime_snapshot_provider(Arc::new(move || {
        Ok(runtime_snapshot_for_history(
            ROOT_CONTEXT_BRANCH_ID,
            &parent_history,
        ))
    }));

    let call = test_tool_call(
        tool_names::TOOL_CONTEXT_RETURN,
        r#"{"outcome":"useful","summary":"Found the issue","next_action":"Apply fix"}"#,
    );
    agent
        .append_assistant_tool_calls("", std::slice::from_ref(&call))
        .expect("return tool call should append");

    agent
        .execute_tool_call_and_record(&call, &mut |_| std::future::ready(Ok(())), &mut |_| {
            std::future::ready(Ok(PermissionApproval::AllowOnce))
        })
        .await
        .expect("context return should record before restoring");

    crate::protocol_frames::validate_history_items_complete(&agent.history, None)
        .expect("restored history remains protocol-complete");
    assert_eq!(agent.history.len(), 2);
    assert!(matches!(&agent.history[0], HistoryItem::UserMessage { .. }));
    assert!(matches!(
        &agent.history[1],
        HistoryItem::ContextSummary { .. }
    ));
    assert!(agent.history.iter().all(
        |item| !matches!(item, HistoryItem::ToolOutput { call_id, .. } if call_id == &call.call_id)
    ));
    assert!(agent.context_experiment_restore_point.is_none());
}

#[test]
fn context_checkpoint_batched_with_other_tool_call_fails_before_history_mutation() {
    let mut agent = test_agent();
    let calls = vec![
        test_tool_call(
            tool_names::TOOL_CONTEXT_CHECKPOINT,
            r#"{"label":"alt","reason":"try alternative approach"}"#,
        ),
        test_tool_call("fs__read", r#"{"path":"src/main.rs"}"#),
    ];
    let history_before = agent.history.clone();
    let frames_before = agent.protocol_frames.clone();
    let snapshot_before = agent.runtime_snapshot.clone();

    let error = agent
        .append_assistant_tool_calls("", &calls)
        .expect_err("batched checkpoint must fail before history mutation");

    assert!(error.to_string().contains(
            "context__checkpoint cannot be batched with other tool calls in the same assistant tool-call group"
        ));
    assert_eq!(agent.history, history_before);
    assert_eq!(agent.protocol_frames, frames_before);
    assert_eq!(agent.runtime_snapshot, snapshot_before);
}

#[test]
fn context_return_batched_with_sibling_fails_before_history_mutation() {
    let mut agent = test_agent();
    let calls = vec![
        test_tool_call(
            tool_names::TOOL_CONTEXT_RETURN,
            r#"{"outcome":"useful","summary":"done","next_action":null}"#,
        ),
        test_tool_call("fs__read", r#"{"path":"src/main.rs"}"#),
    ];
    let history_before = agent.history.clone();
    let frames_before = agent.protocol_frames.clone();
    let snapshot_before = agent.runtime_snapshot.clone();

    let error = agent
        .append_assistant_tool_calls("", &calls)
        .expect_err("batched context__return must fail before history mutation");

    assert!(error.to_string().contains(
            "context__return cannot be batched with other tool calls in the same assistant tool-call group"
        ));
    assert_eq!(agent.history, history_before);
    assert_eq!(agent.protocol_frames, frames_before);
    assert_eq!(agent.runtime_snapshot, snapshot_before);
}

#[tokio::test]
async fn ordinary_request_build_uses_installed_runtime_snapshot_only() {
    let mut agent = test_agent();
    agent.history = vec![HistoryItem::user("EXTERNAL-TRANSCRIPT-CONTENT")];
    agent.runtime_snapshot = runtime_snapshot_for_history(
        ROOT_CONTEXT_BRANCH_ID,
        &[HistoryItem::user("INSTALLED-RUNTIME-SNAPSHOT-CONTENT")],
    );
    let mut on_event = |_| std::future::ready(Ok(()));

    let prepared = compaction::prepare_request_build(
        &mut agent,
        ApiProtocol::Responses,
        &[],
        0,
        &[],
        &mut on_event,
    )
    .await
    .expect("ordinary request builds from the installed runtime snapshot");

    let request = match prepared.build.request {
        BuiltRequest::Responses(request) => serde_json::to_value(request),
        BuiltRequest::ResponsesCompatible(request) => Ok(request),
        BuiltRequest::Completions(_) | BuiltRequest::CompletionsCompatible(_) => {
            panic!("expected responses request")
        }
    };
    let request = request.expect("request serializes");
    let json = serde_json::to_string(&request).expect("request serializes");
    assert!(json.contains("INSTALLED-RUNTIME-SNAPSHOT-CONTENT"));
    assert!(!json.contains("EXTERNAL-TRANSCRIPT-CONTENT"));
}

#[tokio::test]
async fn chat_stream_creation_failure_includes_request_budget_diagnostic() {
    let listener = TcpListener::bind("127.0.0.1:0")
        .await
        .expect("test port should bind");
    let addr = listener.local_addr().expect("test listener has local addr");
    drop(listener);
    let client = Client::with_config(
        OpenAIConfig::new()
            .with_api_base(format!("http://{addr}"))
            .with_api_key("test"),
    );
    let mut agent = Agent::new(client, "m1", 4, 4);
    agent.set_model_catalog(HashMap::from([(
        "m1".into(),
        ModelRequestMetadata {
            context_window: Some(32_000),
            effective_input_limit_tokens: Some(4_096),
            max_output_tokens: Some(2_000),
            supports_tools: false,
            supports_reasoning: false,
            ..Default::default()
        },
    )]));
    let mut retry_config = test_retry_config();
    retry_config.enabled = false;
    agent.set_retry_config(retry_config);

    let error = agent
        .run_oai_comp_stream_async(
            "hello",
            |_| std::future::ready(Ok(())),
            |_| std::future::ready(Ok(())),
            |_| std::future::ready(Ok(PermissionApproval::AllowOnce)),
        )
        .await
        .expect_err("stream creation should fail");
    let message = format!("{error:#}");

    assert!(
        message.contains("failed to create streamed chat completion"),
        "{message}"
    );
    assert!(message.contains("model=m1"), "{message}");
    assert!(message.contains("estimated_request_tokens="), "{message}");
    assert!(message.contains("input_budget_tokens=4096"), "{message}");
    assert!(
        message.contains("effective_input_limit_tokens=4096"),
        "{message}"
    );
    assert!(message.contains("protected_tokens="), "{message}");
}

#[tokio::test]
async fn compatible_chat_stream_sends_one_physical_request() {
    let (base_url, request_count, server) = spawn_chat_completion_server(vec![
            "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nContent-Length: 14\r\nConnection: close\r\n\r\ndata: [DONE]\n\n",
        ])
        .await;
    let client = Client::with_config(
        OpenAIConfig::new()
            .with_api_base(base_url)
            .with_api_key("test"),
    );
    let request = json!({"model": "m1", "stream": true, "messages": []});
    let response = send_compatible_chat_completion_stream(&client, &request)
        .await
        .expect("request should succeed");

    assert_eq!(response.status(), reqwest::StatusCode::OK);
    assert_eq!(request_count.load(Ordering::SeqCst), 1);
    server.await.expect("server task should finish");
}

#[tokio::test]
async fn responses_completion_over_tool_call_budget_emits_completed_telemetry() {
    let body = r#"data: {"type":"response.completed","sequence_number":1,"response":{"id":"resp-over-budget","object":"response","created_at":1780856440,"status":"completed","background":false,"error":null,"incomplete_details":null,"instructions":null,"max_output_tokens":null,"model":"m1","output":[{"type":"function_call","id":"fc-1","call_id":"call-1","name":"first","arguments":"{}","status":"completed"},{"type":"function_call","id":"fc-2","call_id":"call-2","name":"second","arguments":"{}","status":"completed"}],"parallel_tool_calls":true,"previous_response_id":null,"reasoning":{},"store":true,"temperature":1,"text":{"format":{"type":"text"}},"tool_choice":"auto","tools":[],"top_p":1,"truncation":"disabled","usage":{"input_tokens":5,"input_tokens_details":{"cached_tokens":0},"output_tokens":3,"output_tokens_details":{"reasoning_tokens":0},"total_tokens":8},"user":null,"metadata":{}}}

data: [DONE]

"#;
    let response = Box::leak(
            format!(
                "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                body.len()
            )
            .into_boxed_str(),
        );
    let (base_url, request_count, server) = spawn_chat_completion_server(vec![response]).await;
    let client = Client::with_config(
        OpenAIConfig::new()
            .with_api_base(base_url)
            .with_api_key("test"),
    );
    let mut agent = Agent::new(client, "m1", 4, 1);
    agent.set_model_catalog(HashMap::from([(
        "m1".into(),
        ModelRequestMetadata {
            context_window: Some(32_000),
            max_output_tokens: Some(2_000),
            supports_tools: true,
            supports_reasoning: false,
            ..Default::default()
        },
    )]));
    let mut audit_telemetry = Vec::new();

    let error = agent
        .run_stream_async(
            "hello",
            |_| std::future::ready(Ok(())),
            |event| {
                if let AgentEvent::LlmRequestTelemetry(telemetry) = event {
                    audit_telemetry.push(telemetry);
                }
                std::future::ready(Ok(()))
            },
            |_| std::future::ready(Ok(PermissionApproval::AllowOnce)),
        )
        .await
        .expect_err("over-budget tool calls should fail locally");

    assert!(error.to_string().contains("too many tool calls"));
    assert_eq!(
        audit_telemetry
            .iter()
            .map(|telemetry| telemetry.phase)
            .collect::<Vec<_>>(),
        vec![
            LlmRequestTelemetryPhase::Prepared,
            LlmRequestTelemetryPhase::Completed,
        ]
    );
    let completed = audit_telemetry
        .iter()
        .find(|telemetry| telemetry.phase == LlmRequestTelemetryPhase::Completed)
        .expect("provider completion telemetry");
    assert_eq!(
        completed.provider_response_id.as_deref(),
        Some("resp-over-budget")
    );
    assert_eq!(
        completed.usage.as_ref().map(|usage| usage.used_tokens),
        Some(8)
    );
    assert_request_telemetry_is_terminal_once(&audit_telemetry);
    assert_eq!(request_count.load(Ordering::SeqCst), 1);
    server.await.expect("server task should finish");
}

#[tokio::test]
async fn responses_stream_recovers_from_malformed_event_after_visible_delta() {
    let first_body = r#"data: {"type":"response.output_text.delta","sequence_number":1,"item_id":"msg-1","output_index":0,"content_index":0,"delta":"partial "}

data: {"type":"response.completed","response":{"reasoning":{"effort":"max"}}}

"#;
    let second_body = r#"data: {"type":"response.completed","sequence_number":1,"response":{"id":"resp-recovered","object":"response","created_at":1780856440,"status":"completed","background":false,"error":null,"incomplete_details":null,"instructions":null,"max_output_tokens":null,"model":"m1","output":[{"type":"message","id":"msg-2","status":"completed","role":"assistant","content":[{"type":"output_text","text":"continued","annotations":[]}]}],"parallel_tool_calls":true,"previous_response_id":null,"reasoning":{},"store":true,"temperature":1,"text":{"format":{"type":"text"}},"tool_choice":"auto","tools":[],"top_p":1,"truncation":"disabled","usage":{"input_tokens":5,"input_tokens_details":{"cached_tokens":0},"output_tokens":3,"output_tokens_details":{"reasoning_tokens":0},"total_tokens":8},"user":null,"metadata":{}}}

data: [DONE]

"#;
    let first_response = Box::leak(
            format!(
                "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{first_body}",
                first_body.len()
            )
            .into_boxed_str(),
        );
    let second_response = Box::leak(
            format!(
                "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{second_body}",
                second_body.len()
            )
            .into_boxed_str(),
        );
    let (base_url, request_count, server) =
        spawn_chat_completion_server(vec![first_response, second_response]).await;
    let client = Client::with_config(
        OpenAIConfig::new()
            .with_api_base(base_url)
            .with_api_key("test"),
    );
    let mut agent = Agent::new(client, "m1", 4, 4);
    agent.set_model_catalog(HashMap::from([(
        "m1".into(),
        ModelRequestMetadata {
            context_window: Some(32_000),
            max_output_tokens: Some(2_000),
            supports_tools: false,
            supports_reasoning: false,
            ..Default::default()
        },
    )]));
    agent.set_retry_config(test_retry_config());
    let mut deltas = Vec::new();
    let mut stream_issues = Vec::new();
    let mut audit_telemetry = Vec::new();

    let result = agent
        .run_stream_async(
            "hello",
            |delta| {
                deltas.push(delta.to_string());
                std::future::ready(Ok(()))
            },
            |event| {
                match event {
                    AgentEvent::ModelStreamIssue { message, .. } => stream_issues.push(message),
                    AgentEvent::LlmRequestTelemetry(telemetry) => audit_telemetry.push(telemetry),
                    _ => {}
                }
                std::future::ready(Ok(()))
            },
            |_| std::future::ready(Ok(PermissionApproval::AllowOnce)),
        )
        .await
        .expect("malformed event after output should continue with a fresh iteration");

    assert_eq!(result, "partial continued");
    assert_eq!(deltas, vec!["partial "]);
    assert_eq!(stream_issues, vec!["Model stream interrupted"]);
    assert_eq!(
        audit_telemetry
            .iter()
            .map(|telemetry| (telemetry.phase, telemetry.error_class))
            .collect::<Vec<_>>(),
        vec![
            (LlmRequestTelemetryPhase::Prepared, None),
            (
                LlmRequestTelemetryPhase::Interrupted,
                Some(LlmRequestErrorClass::ProtocolValidation),
            ),
            (LlmRequestTelemetryPhase::Prepared, None),
            (LlmRequestTelemetryPhase::Completed, None),
        ]
    );
    assert_request_telemetry_is_terminal_once(&audit_telemetry);
    assert_eq!(request_count.load(Ordering::SeqCst), 2);
    server.await.expect("server task should finish");
}

#[tokio::test]
async fn compatible_chat_stream_retries_read_error_before_visible_output() {
    let body = r#"data: {"choices":[{"index":0,"delta":{"content":"ok"},"finish_reason":"stop"}]}

data: [DONE]

"#;
    let first_response = "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nContent-Length: 100\r\nConnection: close\r\n\r\n";
    let second_response = Box::leak(
            format!(
                "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                body.len()
            )
            .into_boxed_str(),
        );
    let (base_url, request_count, server) =
        spawn_chat_completion_server(vec![first_response, second_response]).await;
    let client = Client::with_config(
        OpenAIConfig::new()
            .with_api_base(base_url)
            .with_api_key("test"),
    );
    let mut agent = Agent::new(client, "m1", 4, 4);
    agent.set_model_catalog(HashMap::from([(
        "m1".into(),
        ModelRequestMetadata {
            context_window: Some(32_000),
            max_output_tokens: Some(2_000),
            supports_tools: true,
            supports_reasoning: false,
            ..Default::default()
        },
    )]));
    agent.set_retry_config(test_retry_config());
    let mut deltas = Vec::new();
    let mut audit_telemetry = Vec::new();

    let result = agent
        .run_oai_comp_stream_async(
            "hello",
            |delta| {
                deltas.push(delta.to_string());
                std::future::ready(Ok(()))
            },
            |event| {
                if let AgentEvent::LlmRequestTelemetry(telemetry) = event {
                    audit_telemetry.push(telemetry);
                }
                std::future::ready(Ok(()))
            },
            |_| std::future::ready(Ok(PermissionApproval::AllowOnce)),
        )
        .await
        .expect("pre-output stream read failure should retry");

    assert_eq!(result, "ok");
    assert_eq!(deltas, vec!["ok"]);
    assert_eq!(
        audit_telemetry
            .iter()
            .map(|telemetry| (telemetry.phase, telemetry.attempt))
            .collect::<Vec<_>>(),
        vec![
            (LlmRequestTelemetryPhase::Prepared, 1),
            (LlmRequestTelemetryPhase::Failed, 1),
            (LlmRequestTelemetryPhase::Prepared, 2),
            (LlmRequestTelemetryPhase::Completed, 2),
        ]
    );
    assert_request_telemetry_is_terminal_once(&audit_telemetry);
    assert_eq!(request_count.load(Ordering::SeqCst), 2);
    server.await.expect("server task should finish");
}

#[tokio::test]
async fn compatible_chat_stream_continues_after_streamed_usage_event() {
    let first_body = r#"data: {"choices":[],"usage":{"prompt_tokens":5,"completion_tokens":0,"total_tokens":5}}

"#;
    let second_body = r#"data: {"choices":[{"index":0,"delta":{"content":"ok"},"finish_reason":"stop"}]}

data: [DONE]

"#;
    let first_response = Box::leak(
            format!(
                "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nContent-Length: 1000\r\nConnection: close\r\n\r\n{first_body}"
            )
            .into_boxed_str(),
        );
    let second_response = Box::leak(
            format!(
                "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{second_body}",
                second_body.len()
            )
            .into_boxed_str(),
        );
    let (base_url, request_count, server) =
        spawn_chat_completion_server(vec![first_response, second_response]).await;
    let client = Client::with_config(
        OpenAIConfig::new()
            .with_api_base(base_url)
            .with_api_key("test"),
    );
    let mut agent = Agent::new(client, "m1", 4, 4);
    agent.set_model_catalog(HashMap::from([(
        "m1".into(),
        ModelRequestMetadata {
            context_window: Some(32_000),
            max_output_tokens: Some(2_000),
            supports_tools: true,
            supports_reasoning: false,
            ..Default::default()
        },
    )]));
    agent.set_retry_config(test_retry_config());
    let mut deltas = Vec::new();
    let mut usage_events = Vec::new();
    let mut stream_issues = Vec::new();

    let result = agent
        .run_oai_comp_stream_async(
            "hello",
            |delta| {
                deltas.push(delta.to_string());
                std::future::ready(Ok(()))
            },
            |event| {
                match event {
                    AgentEvent::TokenUsageUpdated { used_tokens, .. } => {
                        usage_events.push(used_tokens);
                    }
                    AgentEvent::ModelStreamIssue { message, .. } => {
                        stream_issues.push(message);
                    }
                    _ => {}
                }
                std::future::ready(Ok(()))
            },
            |_| std::future::ready(Ok(PermissionApproval::AllowOnce)),
        )
        .await
        .expect("stream read failure after usage event should continue with a fresh iteration");

    assert_eq!(result, "ok");
    assert_eq!(deltas, vec!["ok"]);
    assert_eq!(stream_issues, vec!["Model stream interrupted"]);
    assert!(
        usage_events.contains(&5),
        "missing streamed usage event: {usage_events:?}"
    );
    assert_eq!(request_count.load(Ordering::SeqCst), 2);
    server.await.expect("server task should finish");
}

#[tokio::test]
async fn compatible_chat_stream_retries_incomplete_json_event_before_visible_output() {
    let first_body = "data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"partial\"}\n\n";
    let second_body = r#"data: {"choices":[{"index":0,"delta":{"content":"ok"},"finish_reason":"stop"}]}

data: [DONE]

"#;
    let first_response = Box::leak(
            format!(
                "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nConnection: close\r\n\r\n{first_body}"
            )
            .into_boxed_str(),
        );
    let second_response = Box::leak(
            format!(
                "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{second_body}",
                second_body.len()
            )
            .into_boxed_str(),
        );
    let (base_url, request_count, server) =
        spawn_chat_completion_server(vec![first_response, second_response]).await;
    let client = Client::with_config(
        OpenAIConfig::new()
            .with_api_base(base_url)
            .with_api_key("test"),
    );
    let mut agent = Agent::new(client, "m1", 4, 4);
    agent.set_model_catalog(HashMap::from([(
        "m1".into(),
        ModelRequestMetadata {
            context_window: Some(32_000),
            max_output_tokens: Some(2_000),
            supports_tools: true,
            supports_reasoning: false,
            ..Default::default()
        },
    )]));
    agent.set_retry_config(test_retry_config());
    let mut deltas = Vec::new();

    let result = agent
        .run_oai_comp_stream_async(
            "hello",
            |delta| {
                deltas.push(delta.to_string());
                std::future::ready(Ok(()))
            },
            |_| std::future::ready(Ok(())),
            |_| std::future::ready(Ok(PermissionApproval::AllowOnce)),
        )
        .await
        .expect("incomplete pre-output json event should retry");

    assert_eq!(result, "ok");
    assert_eq!(deltas, vec!["ok"]);
    assert_eq!(request_count.load(Ordering::SeqCst), 2);
    server.await.expect("server task should finish");
}

#[tokio::test]
async fn compatible_chat_stream_shares_retry_budget_across_creation_and_read() {
    let body = r#"data: {"choices":[{"index":0,"delta":{"content":"too-late"},"finish_reason":"stop"}]}

data: [DONE]

"#;
    let third_response = Box::leak(
            format!(
                "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                body.len()
            )
            .into_boxed_str(),
        );
    let (base_url, request_count, server) = spawn_chat_completion_server(vec![
            "HTTP/1.1 500 Internal Server Error\r\nContent-Length: 9\r\nConnection: close\r\n\r\ntransient",
            "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nContent-Length: 100\r\nConnection: close\r\n\r\n",
            third_response,
        ])
        .await;
    let client = Client::with_config(
        OpenAIConfig::new()
            .with_api_base(base_url)
            .with_api_key("test"),
    );
    let mut retry_config = test_retry_config();
    retry_config.max_attempts = 2;
    let mut agent = Agent::new(client, "m1", 4, 4);
    agent.set_model_catalog(HashMap::from([(
        "m1".into(),
        ModelRequestMetadata {
            context_window: Some(32_000),
            max_output_tokens: Some(2_000),
            supports_tools: true,
            supports_reasoning: false,
            ..Default::default()
        },
    )]));
    agent.set_retry_config(retry_config);
    let mut deltas = Vec::new();

    let error = agent
        .run_oai_comp_stream_async(
            "hello",
            |delta| {
                deltas.push(delta.to_string());
                std::future::ready(Ok(()))
            },
            |_| std::future::ready(Ok(())),
            |_| std::future::ready(Ok(PermissionApproval::AllowOnce)),
        )
        .await
        .expect_err("read retry should not exceed shared max_attempts budget");

    assert!(
        !error.to_string().trim().is_empty(),
        "unexpected empty error message: {error:?}"
    );
    assert!(deltas.is_empty());
    assert_eq!(request_count.load(Ordering::SeqCst), 2);
    server.abort();
}

#[tokio::test]
async fn compatible_chat_stream_recovers_read_error_after_visible_text() {
    let body = r#"data: {"choices":[{"index":0,"delta":{"content":"partial"},"finish_reason":null}]}

"#;
    let first_response = Box::leak(
            format!(
                "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nContent-Length: 1000\r\nConnection: close\r\n\r\n{body}"
            )
            .into_boxed_str(),
        );
    let second_body = r#"data: {"choices":[{"index":0,"delta":{"content":" continuation"},"finish_reason":"stop"}]}

data: [DONE]

"#;
    let second_response = Box::leak(
            format!(
                "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{second_body}",
                second_body.len()
            )
            .into_boxed_str(),
        );
    let (base_url, request_count, server) =
        spawn_chat_completion_server(vec![first_response, second_response]).await;
    let client = Client::with_config(
        OpenAIConfig::new()
            .with_api_base(base_url)
            .with_api_key("test"),
    );
    let mut agent = Agent::new(client, "m1", 4, 4);
    agent.set_model_catalog(HashMap::from([(
        "m1".into(),
        ModelRequestMetadata {
            context_window: Some(32_000),
            max_output_tokens: Some(2_000),
            supports_tools: true,
            supports_reasoning: false,
            ..Default::default()
        },
    )]));
    agent.set_retry_config(test_retry_config());
    let mut deltas = Vec::new();
    let mut stream_issues = Vec::new();
    let mut finalized_outcomes = Vec::new();

    let result = agent
        .run_oai_comp_stream_async(
            "hello",
            |delta| {
                deltas.push(delta.to_string());
                std::future::ready(Ok(()))
            },
            |event| {
                match event {
                    AgentEvent::TurnFinalized(event) => finalized_outcomes.push(event.outcome),
                    AgentEvent::ModelStreamIssue { message, .. } => stream_issues.push(message),
                    _ => {}
                }
                std::future::ready(Ok(()))
            },
            |_| std::future::ready(Ok(PermissionApproval::AllowOnce)),
        )
        .await
        .expect("post-output stream read failure should continue with a fresh iteration");

    let expected = "partial continuation".to_string();
    assert_eq!(result, expected);
    assert_eq!(deltas, vec!["partial", " continuation"]);
    assert_eq!(stream_issues, vec!["Model stream interrupted"]);
    assert_eq!(finalized_outcomes, vec!["completed"]);
    assert_eq!(request_count.load(Ordering::SeqCst), 2);
    assert!(
        agent
            .history
            .iter()
            .any(|item| matches!(item, HistoryItem::AssistantText { text } if text == "partial"))
    );
    server.await.expect("server task should finish");
}

#[tokio::test]
async fn compatible_chat_stream_recovers_missing_finish_reason_after_visible_text() {
    let body = r#"data: {"choices":[{"index":0,"delta":{"content":"partial"},"finish_reason":null}]}

data: [DONE]

"#;
    let first_response = Box::leak(
            format!(
                "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                body.len()
            )
            .into_boxed_str(),
        );
    let second_body = r#"data: {"choices":[{"index":0,"delta":{"content":" continuation"},"finish_reason":"stop"}]}

data: [DONE]

"#;
    let second_response = Box::leak(
            format!(
                "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{second_body}",
                second_body.len()
            )
            .into_boxed_str(),
        );
    let (base_url, request_count, server) =
        spawn_chat_completion_server(vec![first_response, second_response]).await;
    let client = Client::with_config(
        OpenAIConfig::new()
            .with_api_base(base_url)
            .with_api_key("test"),
    );
    let mut agent = Agent::new(client, "m1", 4, 4);
    agent.set_model_catalog(HashMap::from([(
        "m1".into(),
        ModelRequestMetadata {
            context_window: Some(32_000),
            max_output_tokens: Some(2_000),
            supports_tools: true,
            supports_reasoning: false,
            ..Default::default()
        },
    )]));
    agent.set_retry_config(test_retry_config());

    let result = agent
        .run_oai_comp_stream_async(
            "hello",
            |_| std::future::ready(Ok(())),
            |_| std::future::ready(Ok(())),
            |_| std::future::ready(Ok(PermissionApproval::AllowOnce)),
        )
        .await
        .expect("missing finish_reason after visible text should continue next iteration");

    let expected = "partial continuation".to_string();
    assert_eq!(result, expected);
    assert!(matches!(
        agent.history.first(),
        Some(HistoryItem::UserMessage { .. })
    ));
    assert!(
        agent
            .history
            .iter()
            .any(|item| matches!(item, HistoryItem::AssistantText { text } if text == "partial"))
    );
    assert_eq!(request_count.load(Ordering::SeqCst), 2);
    server.await.expect("server task should finish");
}

#[tokio::test]
async fn compatible_chat_stream_cancels_pending_tool_call_on_invalid_finish_reason() {
    let first_body = r#"data: {"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call-interrupted","type":"function","function":{"name":"shell__exec","arguments":""}}]},"finish_reason":"stop"}]}

data: [DONE]

"#;
    let second_body = r#"data: {"choices":[{"index":0,"delta":{"content":"ok"},"finish_reason":"stop"}]}

data: [DONE]

"#;
    let first_response = Box::leak(
            format!(
                "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{first_body}",
                first_body.len()
            )
            .into_boxed_str(),
        );
    let second_response = Box::leak(
            format!(
                "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{second_body}",
                second_body.len()
            )
            .into_boxed_str(),
        );
    let (base_url, request_count, server) =
        spawn_chat_completion_server(vec![first_response, second_response]).await;
    let client = Client::with_config(
        OpenAIConfig::new()
            .with_api_base(base_url)
            .with_api_key("test"),
    );
    let mut agent = Agent::new(client, "m1", 4, 4);
    agent.set_model_catalog(HashMap::from([(
        "m1".into(),
        ModelRequestMetadata {
            context_window: Some(32_000),
            max_output_tokens: Some(2_000),
            supports_tools: true,
            supports_reasoning: false,
            ..Default::default()
        },
    )]));
    agent.set_retry_config(test_retry_config());
    let mut cancelled_calls = Vec::new();
    let mut started_calls = Vec::new();
    let mut finished_calls = Vec::new();
    let mut stream_issues = Vec::new();

    let result = agent
        .run_oai_comp_stream_async(
            "hello",
            |_| std::future::ready(Ok(())),
            |event| {
                match event {
                    AgentEvent::ToolCallCancelled { call_id, name } => {
                        cancelled_calls.push((call_id, name));
                    }
                    AgentEvent::ToolCallStarted { call_id, .. } => started_calls.push(call_id),
                    AgentEvent::ToolCallFinished { call_id, .. } => finished_calls.push(call_id),
                    AgentEvent::ModelStreamIssue { message, .. } => stream_issues.push(message),
                    _ => {}
                }
                std::future::ready(Ok(()))
            },
            |_| std::future::ready(Ok(PermissionApproval::AllowOnce)),
        )
        .await
        .expect("invalid finish reason with pending tool should cancel and continue");

    assert_eq!(result, "ok");
    assert_eq!(
        cancelled_calls,
        vec![("call-interrupted".to_string(), "shell__exec".to_string())]
    );
    assert!(started_calls.is_empty());
    assert!(finished_calls.is_empty());
    assert_eq!(stream_issues, vec!["Model stream interrupted"]);
    assert!(!agent.history.iter().any(|item| matches!(
        item,
        HistoryItem::AssistantToolCalls { calls, .. }
            if calls.iter().any(|call| call.call_id == "call-interrupted")
    )));
    assert_eq!(request_count.load(Ordering::SeqCst), 2);
    server.await.expect("server task should finish");
}

#[tokio::test]
async fn compatible_chat_stream_cancels_pending_tool_call_before_terminal_finish_error() {
    let body = r#"data: {"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call-filtered","type":"function","function":{"name":"shell__exec","arguments":""}}]},"finish_reason":"content_filter"}]}

data: [DONE]

"#;
    let response = Box::leak(
            format!(
                "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                body.len()
            )
            .into_boxed_str(),
        );
    let (base_url, request_count, server) = spawn_chat_completion_server(vec![response]).await;
    let client = Client::with_config(
        OpenAIConfig::new()
            .with_api_base(base_url)
            .with_api_key("test"),
    );
    let mut agent = Agent::new(client, "m1", 4, 4);
    agent.set_model_catalog(HashMap::from([(
        "m1".into(),
        ModelRequestMetadata {
            context_window: Some(32_000),
            max_output_tokens: Some(2_000),
            supports_tools: true,
            supports_reasoning: false,
            ..Default::default()
        },
    )]));
    agent.set_retry_config(test_retry_config());
    let mut cancelled_calls = Vec::new();
    let mut started_calls = Vec::new();

    let error = agent
        .run_oai_comp_stream_async(
            "hello",
            |_| std::future::ready(Ok(())),
            |event| {
                match event {
                    AgentEvent::ToolCallCancelled { call_id, name } => {
                        cancelled_calls.push((call_id, name));
                    }
                    AgentEvent::ToolCallStarted { call_id, .. } => started_calls.push(call_id),
                    _ => {}
                }
                std::future::ready(Ok(()))
            },
            |_| std::future::ready(Ok(PermissionApproval::AllowOnce)),
        )
        .await
        .expect_err("content_filter should remain terminal after cancelling pending tool");

    assert!(error.to_string().contains("finish_reason=content_filter"));
    assert_eq!(
        cancelled_calls,
        vec![("call-filtered".to_string(), "shell__exec".to_string())]
    );
    assert!(started_calls.is_empty());
    assert_eq!(request_count.load(Ordering::SeqCst), 1);
    server.await.expect("server task should finish");
}

#[tokio::test]
async fn compatible_chat_stream_does_not_recover_terminal_finish_reason_errors() {
    for (finish_reason, expected_error) in [
        ("length", "finish_reason=length"),
        ("content_filter", "finish_reason=content_filter"),
    ] {
        let body = format!(
            r#"data: {{"choices":[{{"index":0,"delta":{{"content":"partial"}},"finish_reason":"{finish_reason}"}}]}}

data: [DONE]

"#
        );
        let response = Box::leak(
                format!(
                    "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                    body.len()
                )
                .into_boxed_str(),
            );
        let (base_url, request_count, server) = spawn_chat_completion_server(vec![response]).await;
        let client = Client::with_config(
            OpenAIConfig::new()
                .with_api_base(base_url)
                .with_api_key("test"),
        );
        let mut agent = Agent::new(client, "m1", 4, 4);
        agent.set_model_catalog(HashMap::from([(
            "m1".into(),
            ModelRequestMetadata {
                context_window: Some(32_000),
                max_output_tokens: Some(2_000),
                supports_tools: true,
                supports_reasoning: false,
                ..Default::default()
            },
        )]));
        agent.set_retry_config(test_retry_config());
        let mut deltas = Vec::new();

        let error = agent
            .run_oai_comp_stream_async(
                "hello",
                |delta| {
                    deltas.push(delta.to_string());
                    std::future::ready(Ok(()))
                },
                |_| std::future::ready(Ok(())),
                |_| std::future::ready(Ok(PermissionApproval::AllowOnce)),
            )
            .await
            .expect_err("terminal finish_reason errors should fail explicitly");

        assert!(
            error.to_string().contains(expected_error),
            "unexpected error for {finish_reason}: {error:?}"
        );
        assert_eq!(deltas, vec!["partial"]);
        assert_eq!(request_count.load(Ordering::SeqCst), 1);
        assert!(
            !agent
                .history
                .iter()
                .any(|item| matches!(item, HistoryItem::AssistantText { .. }))
        );
        server.await.expect("server task should finish");
    }
}

#[tokio::test]
async fn compatible_chat_stream_does_not_retry_bad_request() {
    let (base_url, request_count, server) = spawn_chat_completion_server(vec![
        "HTTP/1.1 400 Bad Request\r\nContent-Length: 11\r\nConnection: close\r\n\r\nbad request",
    ])
    .await;
    let client = Client::with_config(
        OpenAIConfig::new()
            .with_api_base(base_url)
            .with_api_key("test"),
    );
    let request = json!({"model": "m1", "stream": true, "messages": []});
    let error = send_compatible_chat_completion_stream(&client, &request)
        .await
        .expect_err("400 should fail fast");

    assert!(
        error
            .to_string()
            .contains("chat completions request failed with status 400 Bad Request")
    );
    assert_eq!(request_count.load(Ordering::SeqCst), 1);
    server.await.expect("server task should finish");
}

#[test]
fn auto_continue_defaults_to_disabled() {
    let agent = test_agent();

    assert_eq!(agent.auto_continue(), &AutoContinueState::default());
    assert!(agent.todos().is_empty());
}

#[tokio::test]
async fn workflow_auto_continue_tool_enables_bounded_state() {
    let mut agent = test_agent();
    let call = HistoryToolCall {
        call_id: "call-auto".into(),
        name: "workflow__auto_continue".into(),
        arguments_json: r#"{"enabled":true,"max_continuations":2}"#.into(),
    };

    let record = agent
        .execute_tool_call(&call, &mut |_| std::future::ready(Ok(())), &mut |_| {
            std::future::ready(Ok(PermissionApproval::AllowOnce))
        })
        .await
        .expect("control tool should succeed");

    assert!(record.output.ok);
    assert_eq!(agent.auto_continue().enabled, true);
    assert_eq!(agent.auto_continue().max_continuations, 2);
}

#[tokio::test]
async fn execute_tool_call_records_success_status_effects_and_started_finished_events() {
    let mut agent = test_agent();
    let call = test_tool_call(
        "workflow__todos",
        r#"{"items":[{"id":"t1","content":"first","status":"pending"}]}"#,
    );
    let mut events = Vec::new();

    let record = agent
        .execute_tool_call(
            &call,
            &mut |event| {
                events.push(event);
                std::future::ready(Ok(()))
            },
            &mut |_| std::future::ready(Ok(PermissionApproval::AllowOnce)),
        )
        .await
        .expect("tool call should succeed");

    assert_eq!(record.status, ToolExecutionStatus::Executed);
    assert_eq!(record.rejection, None);
    assert!(record.output.ok);
    assert_eq!(record.effects.kind, ToolEffectKind::WorkflowControl);
    assert!(matches!(
        events.as_slice(),
        [
            AgentEvent::ToolCallStarted { .. },
            AgentEvent::TodoSnapshotUpdated { .. },
            AgentEvent::ToolCallFinished { ok: true, .. },
            AgentEvent::ToolExecutionSummary(ToolExecutionSummaryEvent {
                status,
                effect_kind,
                ..
            })
        ] if status == "executed" && effect_kind == "workflow_control"
    ));
}

#[tokio::test]
async fn workflow_todos_tool_updates_todo_state() {
    let mut agent = test_agent();
    let call = HistoryToolCall {
            call_id: "call-todos".into(),
            name: "workflow__todos".into(),
            arguments_json: r#"{"items":[{"id":"t1","content":"first","status":"pending"},{"id":"t2","content":"done","status":"completed"}]}"#.into(),
        };

    agent
        .execute_tool_call(&call, &mut |_| std::future::ready(Ok(())), &mut |_| {
            std::future::ready(Ok(PermissionApproval::AllowOnce))
        })
        .await
        .expect("todo control tool should succeed");

    assert_eq!(agent.todos().len(), 2);
    assert_eq!(agent.todos()[0].status, TodoStatus::Pending);
    assert_eq!(agent.todos()[1].status, TodoStatus::Completed);
}

#[tokio::test]
async fn workflow_todos_event_failure_does_not_mutate_state() {
    let mut agent = test_agent();
    let previous = vec![TodoItem {
        id: "old".into(),
        content: "old task".into(),
        status: TodoStatus::InProgress,
    }];
    agent.turn.workflow.todos = previous.clone();
    let args = json!({
        "items": [{"id":"new","content":"new task","status":"pending"}]
    });

    let result = agent
        .apply_control_tool_state("workflow__todos", &args, &mut |_| {
            std::future::ready(Err(anyhow!("event sink failed")))
        })
        .await;

    assert!(result.is_err());
    assert_eq!(agent.todos(), previous.as_slice());
}

#[tokio::test]
async fn workflow_auto_continue_event_failure_does_not_mutate_state() {
    let mut agent = test_agent();
    let previous = AutoContinueState {
        enabled: false,
        max_continuations: 3,
    };
    agent.turn.workflow.auto_continue = previous.clone();
    let args = json!({"enabled": true, "max_continuations": 5});

    let result = agent
        .apply_control_tool_state("workflow__auto_continue", &args, &mut |_| {
            std::future::ready(Err(anyhow!("event sink failed")))
        })
        .await;

    assert!(result.is_err());
    assert_eq!(agent.auto_continue(), &previous);
}

fn writable_call(
    call_id: &str,
    tool: &str,
    path: &std::path::Path,
    content: &str,
) -> HistoryToolCall {
    HistoryToolCall {
        call_id: call_id.into(),
        name: tool.into(),
        arguments_json: json!({"path": path, "content": content}).to_string(),
    }
}

fn writable_event_phases(events: &[AgentEvent]) -> Vec<bool> {
    events
        .iter()
        .filter_map(|event| match event {
            AgentEvent::ToolCallStarted { .. } => Some(true),
            AgentEvent::ToolCallFinished { ok, .. } => Some(*ok),
            _ => None,
        })
        .collect()
}

fn apply_patch_call(call_id: &str, edits: Vec<Value>) -> HistoryToolCall {
    HistoryToolCall {
        call_id: call_id.into(),
        name: "edit__apply_patch".into(),
        arguments_json: json!({"edits": edits}).to_string(),
    }
}

fn apply_patch_edit(path: &std::path::Path, find: &str, replace: &str) -> Value {
    json!({
        "path": path,
        "find": find,
        "replace": replace,
        "replace_all": false,
    })
}

#[cfg(unix)]
struct UnixWritableFixture {
    external: PathBuf,
    workspace: PathBuf,
}

#[cfg(unix)]
impl UnixWritableFixture {
    fn new(name: &str) -> Self {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system time")
            .as_nanos();
        let external = std::env::temp_dir().join(format!("letcode-agent-{name}-{unique}"));
        let workspace = PathBuf::from("target").join(format!("letcode-agent-{name}-{unique}"));
        std::fs::create_dir_all(&external).expect("create external fixture");
        std::fs::create_dir_all(&workspace).expect("create workspace fixture");
        Self {
            external,
            workspace,
        }
    }
}

#[cfg(unix)]
impl Drop for UnixWritableFixture {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.workspace);
        let _ = std::fs::remove_dir_all(&self.external);
    }
}

#[cfg(unix)]
#[tokio::test]
async fn apply_patch_permission_modes_authorize_external_and_mixed_batches() {
    let fixture = UnixWritableFixture::new("apply-patch-mode-matrix");
    for (mode, batch) in [
        (PermissionMode::Default, "external"),
        (PermissionMode::Default, "mixed"),
        (PermissionMode::Safe, "external"),
        (PermissionMode::Safe, "mixed"),
        (PermissionMode::Solo, "external"),
        (PermissionMode::Solo, "mixed"),
    ] {
        let external = fixture
            .external
            .join(format!("{mode:?}-{batch}-external.txt"));
        let internal = fixture
            .workspace
            .join(format!("{mode:?}-{batch}-internal.txt"));
        std::fs::write(&external, "old external").expect("seed external target");
        std::fs::write(&internal, "old internal").expect("seed internal target");
        let mut edits = vec![apply_patch_edit(&external, "old", "new")];
        if batch == "mixed" {
            edits.push(apply_patch_edit(&internal, "old", "new"));
        }
        let call = apply_patch_call(&format!("apply-patch-{mode:?}-{batch}"), edits);
        let expected_preview = format!(
            "Outside-workspace access requested:\n- {}",
            external
                .canonicalize()
                .expect("canonical external target")
                .display()
        );
        let expected_targets = if batch == "mixed" { 2 } else { 1 };
        let mut agent = test_agent();
        agent.set_permission_mode(mode);
        let mut approvals = 0;
        let mut events = Vec::new();
        let record = agent
            .execute_tool_call(
                &call,
                &mut |event| {
                    events.push(event);
                    std::future::ready(Ok(()))
                },
                &mut |request| {
                    approvals += 1;
                    assert_eq!(request.preview.as_deref(), Some(expected_preview.as_str()));
                    assert_eq!(request.can_allow_always, mode == PermissionMode::Default);
                    assert_eq!(
                        request.grant_summary.as_deref(),
                        (mode == PermissionMode::Default)
                            .then(|| {
                                format!("edit__apply_patch: {expected_targets} target path(s)")
                            })
                            .as_deref()
                    );
                    std::future::ready(Ok(PermissionApproval::AllowOnce))
                },
            )
            .await
            .expect("approved patch executes");

        assert_eq!(approvals, usize::from(mode != PermissionMode::Solo));
        assert_eq!(writable_event_phases(&events), vec![true, true]);
        assert_eq!(record.status, ToolExecutionStatus::Executed);
        assert_eq!(record.rejection, None);
        assert!(
            record.output.ok,
            "{mode:?} {batch}: {:?}",
            record.output.error
        );
        assert_eq!(std::fs::read_to_string(&external).unwrap(), "new external");
        if batch == "mixed" {
            assert_eq!(std::fs::read_to_string(&internal).unwrap(), "new internal");
        }
    }
}

#[cfg(unix)]
#[tokio::test]
async fn apply_patch_allow_always_grants_exact_targets_and_prepares_each_call() {
    let fixture = UnixWritableFixture::new("apply-patch-allow-always");
    let first = fixture.external.join("first.txt");
    let second = fixture.external.join("second.txt");
    let third = fixture.external.join("third.txt");
    for path in [&first, &second, &third] {
        std::fs::write(path, "old").expect("seed target");
    }
    let mut agent = test_agent();
    let mut approvals = 0;
    let mut previews = Vec::new();
    let mut grant_summaries = Vec::new();

    let first_record = agent
        .execute_tool_call(
            &apply_patch_call(
                "apply-patch-grant-first",
                vec![
                    apply_patch_edit(&first, "old", "first"),
                    apply_patch_edit(&second, "old", "first"),
                ],
            ),
            &mut |_| std::future::ready(Ok(())),
            &mut |request| {
                approvals += 1;
                previews.push(request.preview.expect("external preview"));
                grant_summaries.push(request.grant_summary.expect("target-set grant"));
                std::future::ready(Ok(PermissionApproval::AllowAlways))
            },
        )
        .await
        .expect("first target set executes");
    assert!(first_record.output.ok, "{:?}", first_record.output.error);

    let replacement = fixture.external.join("first-replacement.txt");
    std::fs::write(&replacement, "replacement").expect("seed replacement inode");
    std::fs::rename(&replacement, &first).expect("replace first target inode");
    let repeated = agent
        .execute_tool_call(
            &apply_patch_call(
                "apply-patch-grant-repeated",
                vec![
                    apply_patch_edit(&first, "replacement", "second"),
                    apply_patch_edit(&second, "first", "second"),
                ],
            ),
            &mut |_| std::future::ready(Ok(())),
            &mut |_| {
                approvals += 1;
                std::future::ready(Ok(PermissionApproval::Deny))
            },
        )
        .await
        .expect("matching target set reuses grant and prepares afresh");
    assert!(repeated.output.ok, "{:?}", repeated.output.error);
    assert_eq!(approvals, 1, "identical canonical target set reuses grant");
    assert_eq!(std::fs::read_to_string(&first).unwrap(), "second");
    assert_eq!(std::fs::read_to_string(&second).unwrap(), "second");

    let distinct = agent
        .execute_tool_call(
            &apply_patch_call(
                "apply-patch-grant-distinct",
                vec![
                    apply_patch_edit(&first, "second", "third"),
                    apply_patch_edit(&third, "old", "third"),
                ],
            ),
            &mut |_| std::future::ready(Ok(())),
            &mut |request| {
                approvals += 1;
                previews.push(request.preview.expect("external preview"));
                grant_summaries.push(request.grant_summary.expect("target-set grant"));
                std::future::ready(Ok(PermissionApproval::AllowOnce))
            },
        )
        .await
        .expect("distinct target set requests approval");
    assert!(distinct.output.ok, "{:?}", distinct.output.error);
    assert_eq!(approvals, 2);
    assert_eq!(
        grant_summaries,
        vec!["edit__apply_patch: 2 target path(s)"; 2]
    );
    assert_ne!(
        previews[0], previews[1],
        "distinct canonical target set previews differ"
    );
    assert_eq!(std::fs::read_to_string(&third).unwrap(), "third");
}

#[cfg(unix)]
#[tokio::test]
async fn apply_patch_approval_and_started_rebinding_fail_closed() {
    use std::os::unix::fs::symlink;

    for timing in ["approval", "started"] {
        for scenario in [
            "parent",
            "leaf",
            "inside-outside",
            "outside-inside",
            "missing",
        ] {
            let fixture = UnixWritableFixture::new(&format!("apply-patch-{timing}-{scenario}"));
            let raw = match scenario {
                "inside-outside" => fixture.workspace.join("target.txt"),
                "parent" => fixture.external.join("parent").join("target.txt"),
                _ => fixture.external.join("target.txt"),
            };
            std::fs::create_dir_all(raw.parent().expect("target parent"))
                .expect("create target parent");
            std::fs::write(&raw, "authorized").expect("seed authorized target");
            let inside = fixture.workspace.join("inside.txt");
            let outside = fixture.external.join("outside.txt");
            std::fs::write(&inside, "inside").expect("seed inside alternate");
            std::fs::write(&outside, "outside").expect("seed outside alternate");
            let parent_stash = fixture.external.join("parent-stash");
            let missing = fixture.external.join("missing").join("target.txt");
            let call = apply_patch_call(
                &format!("apply-patch-{timing}-{scenario}"),
                vec![apply_patch_edit(&raw, "authorized", "must not write")],
            );
            let mutate = || match scenario {
                "parent" => {
                    let parent = raw.parent().unwrap();
                    std::fs::rename(parent, &parent_stash).expect("replace authorized parent");
                    std::fs::create_dir(parent).expect("create replacement parent");
                    std::fs::write(&raw, "replacement").expect("seed replacement target");
                }
                "leaf" => {
                    let replacement = fixture.external.join("replacement.txt");
                    std::fs::write(&replacement, "replacement").expect("seed replacement inode");
                    std::fs::rename(&replacement, &raw).expect("replace authorized leaf");
                }
                "inside-outside" => {
                    std::fs::remove_file(&raw).expect("remove inside target");
                    symlink(&outside, &raw).expect("rebind inside path outside");
                }
                "outside-inside" => {
                    std::fs::remove_file(&raw).expect("remove outside target");
                    symlink(&inside, &raw).expect("rebind outside path inside");
                }
                "missing" => {
                    std::fs::remove_file(&raw).expect("remove authorized target");
                    symlink(&missing, &raw).expect("rebind to missing target");
                }
                _ => unreachable!(),
            };
            let mut agent = test_agent();
            let mut events = Vec::new();
            let mut approvals = 0;
            let record = agent
                .execute_tool_call(
                    &call,
                    &mut |event| {
                        if timing == "started"
                            && matches!(event, AgentEvent::ToolCallStarted { .. })
                        {
                            mutate();
                        }
                        events.push(event);
                        std::future::ready(Ok(()))
                    },
                    &mut |_| {
                        approvals += 1;
                        if timing == "approval" {
                            mutate();
                        }
                        std::future::ready(Ok(PermissionApproval::AllowOnce))
                    },
                )
                .await
                .expect("security failure is recorded");

            assert_eq!(approvals, 1, "{timing} {scenario}");
            assert_eq!(
                writable_event_phases(&events),
                vec![true, false],
                "{timing} {scenario}"
            );
            assert_eq!(
                record.status,
                ToolExecutionStatus::Executed,
                "{timing} {scenario}"
            );
            assert_eq!(record.rejection, None, "{timing} {scenario}");
            assert_eq!(
                record
                    .output
                    .error
                    .as_ref()
                    .map(|error| error.message.as_str()),
                Some("apply patch target changed after authorization"),
                "{timing} {scenario}"
            );
            assert_eq!(std::fs::read_to_string(&inside).unwrap(), "inside");
            assert_eq!(std::fs::read_to_string(&outside).unwrap(), "outside");
            if scenario == "parent" {
                assert_eq!(
                    std::fs::read_to_string(parent_stash.join("target.txt")).unwrap(),
                    "authorized"
                );
                assert_eq!(std::fs::read_to_string(&raw).unwrap(), "replacement");
            } else if scenario == "leaf" {
                assert_eq!(std::fs::read_to_string(&raw).unwrap(), "replacement");
            }
        }
    }
}

#[cfg(unix)]
#[tokio::test]
async fn apply_patch_target_cap_fails_before_approval_or_started() {
    let fixture = UnixWritableFixture::new("apply-patch-target-cap");
    let edits = (0..65)
        .map(|index| {
            let path = fixture.external.join(format!("{index}.txt"));
            std::fs::write(&path, "old").expect("seed target");
            apply_patch_edit(&path, "old", "new")
        })
        .collect();
    let call = apply_patch_call("apply-patch-target-cap", edits);
    let mut agent = test_agent();
    let mut approvals = 0;
    let mut events = Vec::new();
    let record = agent
        .execute_tool_call(
            &call,
            &mut |event| {
                events.push(event);
                std::future::ready(Ok(()))
            },
            &mut |_| {
                approvals += 1;
                std::future::ready(Ok(PermissionApproval::AllowOnce))
            },
        )
        .await
        .expect("prepare failure is recorded");
    assert_eq!(approvals, 0);
    assert_eq!(writable_event_phases(&events), vec![false]);
    assert_eq!(record.status, ToolExecutionStatus::Rejected);
    assert_eq!(record.rejection, None);
    assert_eq!(
        record
            .output
            .error
            .as_ref()
            .map(|error| error.message.as_str()),
        Some("apply patch accepts at most 64 unique target files")
    );
    for index in 0..65 {
        assert_eq!(
            std::fs::read_to_string(fixture.external.join(format!("{index}.txt"))).unwrap(),
            "old"
        );
    }
}

#[cfg(not(unix))]
#[tokio::test]
async fn apply_patch_unsupported_platform_fails_before_approval_or_started() {
    let call = apply_patch_call(
        "apply-patch-unsupported-platform",
        vec![apply_patch_edit(
            std::path::Path::new("Cargo.toml"),
            "[package]",
            "[package]",
        )],
    );
    let mut agent = test_agent();
    let mut approvals = 0;
    let mut events = Vec::new();
    let record = agent
        .execute_tool_call(
            &call,
            &mut |event| {
                events.push(event);
                std::future::ready(Ok(()))
            },
            &mut |_| {
                approvals += 1;
                std::future::ready(Ok(PermissionApproval::AllowOnce))
            },
        )
        .await
        .expect("unsupported prepare failure is recorded");
    assert_eq!(approvals, 0);
    assert_eq!(writable_event_phases(&events), vec![false]);
    assert_eq!(record.status, ToolExecutionStatus::Rejected);
    assert_eq!(record.rejection, None);
    assert_eq!(
        record
            .output
            .error
            .as_ref()
            .map(|error| error.message.as_str()),
        Some("secure apply patch authorization is unsupported on this platform")
    );
}

#[cfg(unix)]
#[tokio::test]
async fn default_allow_always_writable_grants_use_raw_path_identity() {
    use std::os::unix::ffi::OsStringExt;
    use std::os::unix::fs::symlink;

    let fixture = UnixWritableFixture::new("raw-grant-identity");
    let target_a = fixture
        .external
        .join(std::ffi::OsString::from_vec(b"same-lossy-\x80".to_vec()));
    let target_b = fixture
        .external
        .join(std::ffi::OsString::from_vec(b"same-lossy-\x81".to_vec()));
    let workspace_a = fixture.workspace.join("a");
    let workspace_b = fixture.workspace.join("b");
    // Keep the user-supplied paths UTF-8 while resolving the leaf through a
    // symlink. APFS permits these dangling symlinks even though it rejects a
    // non-UTF-8 leaf at open time. Linux filesystems that permit those leaves
    // exercise the successful write path below without changing this fixture.
    symlink(&target_a, &workspace_a).expect("link target A");
    symlink(&target_b, &workspace_b).expect("link target B");

    let call_a = writable_call("raw-a", "fs__write", &workspace_a, "A");
    let call_a_again = writable_call("raw-a-again", "fs__write", &workspace_a, "A again");
    let call_b = writable_call("raw-b", "fs__write", &workspace_b, "B");
    let mut previews = Vec::new();
    let mut grant_summaries = Vec::new();
    let mut agent_a = test_agent();

    let first = agent_a
        .execute_tool_call(
            &call_a,
            &mut |_| std::future::ready(Ok(())),
            &mut |request| {
                previews.push(request.preview.expect("external preview"));
                grant_summaries.push(request.grant_summary.expect("grant summary"));
                assert!(request.can_allow_always);
                std::future::ready(Ok(PermissionApproval::AllowAlways))
            },
        )
        .await
        .expect("A is approved");
    assert_eq!(first.status, ToolExecutionStatus::Executed);

    let mut repeated_approval_requested = false;
    let repeated = agent_a
        .execute_tool_call(
            &call_a_again,
            &mut |_| std::future::ready(Ok(())),
            &mut |_| {
                repeated_approval_requested = true;
                std::future::ready(Ok(PermissionApproval::Deny))
            },
        )
        .await
        .expect("A grant executes");
    assert_eq!(repeated.status, ToolExecutionStatus::Executed);
    assert!(
        !repeated_approval_requested,
        "matching raw-byte grant must not request approval"
    );

    let denied_b = agent_a
        .execute_tool_call(
            &call_b,
            &mut |_| std::future::ready(Ok(())),
            &mut |request| {
                previews.push(request.preview.expect("external preview"));
                grant_summaries.push(request.grant_summary.expect("grant summary"));
                std::future::ready(Ok(PermissionApproval::Deny))
            },
        )
        .await
        .expect("B denial is recorded");
    assert_eq!(denied_b.status, ToolExecutionStatus::Rejected);
    assert_eq!(
        denied_b.rejection,
        Some(ToolExecutionRejection::PermissionDeniedByUser)
    );

    assert_eq!(previews.len(), 2);
    assert_eq!(grant_summaries.len(), 2);
    assert_ne!(
        previews[0], previews[1],
        "raw-byte preview markers distinguish A and B"
    );
    assert_ne!(
        grant_summaries[0], grant_summaries[1],
        "raw-byte grant markers distinguish A and B"
    );
    if cfg!(target_os = "linux") {
        assert!(
            first.output.ok,
            "Linux raw-byte leaf should be writable: {:?}",
            first.output.error
        );
        assert!(
            repeated.output.ok,
            "Linux raw-byte leaf should be writable: {:?}",
            repeated.output.error
        );
        assert_eq!(std::fs::read(&target_a).expect("A was written"), b"A again");
        assert!(
            !target_b.exists(),
            "denied B must not create its raw-byte leaf"
        );
    }
}

#[cfg(unix)]
#[tokio::test]
async fn writable_inside_to_outside_rebind_is_a_post_started_security_failure() {
    use std::os::unix::fs::symlink;

    let fixture = UnixWritableFixture::new("inside-outside-rebind");
    let raw = fixture.workspace.join("inside.txt");
    let outside = fixture.external.join("outside.txt");
    std::fs::write(&raw, "inside").expect("create inside target");
    let call = writable_call("inside-outside", "fs__write", &raw, "must not write");
    let mut agent = test_agent();
    let mut events = Vec::new();
    let record = agent
        .execute_tool_call(
            &call,
            &mut |event| {
                if matches!(event, AgentEvent::ToolCallStarted { .. }) {
                    std::fs::remove_file(&raw).expect("remove inside target");
                    symlink(&outside, &raw).expect("rebind to external leaf");
                }
                events.push(event);
                std::future::ready(Ok(()))
            },
            &mut |_| std::future::ready(Ok(PermissionApproval::AllowOnce)),
        )
        .await
        .expect("security failure is recorded");

    assert_eq!(writable_event_phases(&events), vec![true, false]);
    assert_eq!(record.status, ToolExecutionStatus::Executed);
    assert_eq!(record.rejection, None);
    assert_eq!(
        record
            .output
            .error
            .as_ref()
            .map(|error| error.message.as_str()),
        Some("writable destination changed after authorization")
    );
    assert!(
        !raw.exists(),
        "rebinding must not recreate the original target"
    );
    assert!(!outside.exists(), "rebinding must not write outside");
}

#[cfg(unix)]
#[tokio::test]
async fn writable_unresolvable_rebind_is_a_post_started_security_failure() {
    use std::os::unix::fs::symlink;

    let fixture = UnixWritableFixture::new("unresolvable-rebind");
    let raw = fixture.workspace.join("authorized.txt");
    let missing = fixture.external.join("missing-parent").join("outside.txt");
    std::fs::write(&raw, "original").expect("create authorized target");
    let call = writable_call("unresolvable-rebind", "fs__write", &raw, "must not write");
    let mut agent = test_agent();
    let mut events = Vec::new();
    let record = agent
        .execute_tool_call(
            &call,
            &mut |event| {
                if matches!(event, AgentEvent::ToolCallStarted { .. }) {
                    std::fs::remove_file(&raw).expect("remove authorized target");
                    symlink(&missing, &raw).expect("rebind to missing parent");
                }
                events.push(event);
                std::future::ready(Ok(()))
            },
            &mut |_| std::future::ready(Ok(PermissionApproval::AllowOnce)),
        )
        .await
        .expect("security failure is recorded");

    assert_eq!(writable_event_phases(&events), vec![true, false]);
    assert_eq!(record.status, ToolExecutionStatus::Executed);
    assert_eq!(record.rejection, None);
    assert_eq!(
        record
            .output
            .error
            .as_ref()
            .map(|error| error.message.as_str()),
        Some("writable destination changed after authorization")
    );
    assert!(
        !raw.exists(),
        "unresolvable rebind must not recreate original"
    );
    assert!(
        !missing.exists(),
        "unresolvable rebind must not write elsewhere"
    );
}

#[cfg(unix)]
#[tokio::test]
async fn writable_tools_cover_allow_once_and_solo_external_execution_events() {
    let fixture = UnixWritableFixture::new("mode-events");
    for (tool, mode) in [
        ("fs__write", PermissionMode::Default),
        ("fs__write", PermissionMode::Safe),
        ("fs__append", PermissionMode::Default),
        ("fs__append", PermissionMode::Safe),
    ] {
        let path = fixture.external.join(format!("allow-once-{tool}-{mode:?}"));
        if tool == "fs__append" {
            std::fs::write(&path, "seed").expect("seed append target");
        }
        let call = writable_call(
            &format!("allow-once-{tool}-{mode:?}"),
            tool,
            &path,
            "approved",
        );
        let mut agent = test_agent();
        agent.set_permission_mode(mode);
        let mut approvals = 0;
        let mut events = Vec::new();
        let record = agent
            .execute_tool_call(
                &call,
                &mut |event| {
                    events.push(event);
                    std::future::ready(Ok(()))
                },
                &mut |request| {
                    approvals += 1;
                    assert_eq!(request.can_allow_always, mode == PermissionMode::Default);
                    std::future::ready(Ok(PermissionApproval::AllowOnce))
                },
            )
            .await
            .expect("allow-once execution");
        assert_eq!(approvals, 1, "{tool} {mode:?}");
        assert_eq!(
            writable_event_phases(&events),
            vec![true, true],
            "{tool} {mode:?}"
        );
        assert_eq!(record.status, ToolExecutionStatus::Executed);
        assert_eq!(record.rejection, None);
        assert_eq!(
            std::fs::read_to_string(&path).expect("written target"),
            if tool == "fs__append" {
                "seedapproved"
            } else {
                "approved"
            },
            "{tool} {mode:?} effect"
        );
    }

    for tool in ["fs__write", "fs__append"] {
        let path = fixture.external.join(format!("solo-{tool}"));
        if tool == "fs__append" {
            std::fs::write(&path, "seed").expect("seed append target");
        }
        let call = writable_call(&format!("solo-{tool}"), tool, &path, "solo");
        let mut agent = test_agent();
        agent.set_permission_mode(PermissionMode::Solo);
        let mut approvals = 0;
        let mut events = Vec::new();
        let record = agent
            .execute_tool_call(
                &call,
                &mut |event| {
                    events.push(event);
                    std::future::ready(Ok(()))
                },
                &mut |_| {
                    approvals += 1;
                    std::future::ready(Ok(PermissionApproval::Deny))
                },
            )
            .await
            .expect("solo execution");
        assert_eq!(approvals, 0, "{tool}");
        assert_eq!(writable_event_phases(&events), vec![true, true], "{tool}");
        assert!(record.output.ok, "{tool}: {:?}", record.output.error);
        assert_eq!(
            std::fs::read_to_string(&path).expect("solo target"),
            if tool == "fs__append" {
                "seedsolo"
            } else {
                "solo"
            },
            "{tool} solo effect"
        );
    }
}

#[cfg(unix)]
#[tokio::test]
async fn writable_tools_cover_user_and_directive_denials_without_starting() {
    let fixture = UnixWritableFixture::new("denial-events");
    for tool in ["fs__write", "fs__append"] {
        for (denial, modes) in [
            ("user", &[PermissionMode::Default, PermissionMode::Safe][..]),
            ("directive", &[PermissionMode::Default][..]),
        ] {
            for mode in modes {
                let path = fixture.external.join(format!("{denial}-{tool}"));
                let call = writable_call(&format!("{denial}-{tool}"), tool, &path, "denied");
                let mut agent = test_agent();
                agent.set_permission_mode(*mode);
                if denial == "directive" {
                    agent.turn = TurnRuntimeState::new(
                        1,
                        WorkflowTurnState::from_user_input("Read-only: inspect and report only."),
                    );
                }
                let mut approvals = 0;
                let mut events = Vec::new();
                let record = agent
                    .execute_tool_call(
                        &call,
                        &mut |event| {
                            events.push(event);
                            std::future::ready(Ok(()))
                        },
                        &mut |_| {
                            approvals += 1;
                            std::future::ready(Ok(PermissionApproval::Deny))
                        },
                    )
                    .await
                    .expect("denial is recorded");
                assert_eq!(
                    writable_event_phases(&events),
                    vec![false],
                    "{tool} {denial} {mode:?}"
                );
                assert!(
                    !events
                        .iter()
                        .any(|event| matches!(event, AgentEvent::ToolCallStarted { .. })),
                    "{tool} {denial} {mode:?} must not start"
                );
                assert!(
                    events.iter().any(|event| matches!(
                        event,
                        AgentEvent::ToolCallFinished { ok: false, .. }
                    )),
                    "{tool} {denial} {mode:?} must finish unsuccessfully"
                );
                assert_eq!(record.status, ToolExecutionStatus::Rejected);
                assert_eq!(
                    record.rejection,
                    Some(if denial == "user" {
                        ToolExecutionRejection::PermissionDeniedByUser
                    } else {
                        ToolExecutionRejection::DirectiveBlocked
                    }),
                    "{tool} {denial} {mode:?} rejection"
                );
                assert_eq!(
                    approvals,
                    usize::from(denial == "user"),
                    "{tool} {denial} {mode:?}"
                );
                assert!(!path.exists(), "{tool} {denial} {mode:?} must not write");
            }
        }
    }
}

#[cfg(unix)]
#[tokio::test]
async fn writable_tools_cover_post_started_rebind_failures() {
    use std::os::unix::fs::symlink;

    let fixture = UnixWritableFixture::new("post-started-events");
    for tool in ["fs__write", "fs__append"] {
        let original = fixture.external.join(format!("original-{tool}"));
        let rebound = fixture.external.join(format!("rebound-{tool}"));
        let raw = fixture.workspace.join(format!("raw-{tool}"));
        std::fs::write(&original, "original").expect("create original target");
        std::fs::write(&rebound, "rebound").expect("create rebound target");
        symlink(&original, &raw).expect("create initial leaf link");
        let call = writable_call(
            &format!("post-started-{tool}"),
            tool,
            &raw,
            "must not write",
        );
        let mut agent = test_agent();
        let mut events = Vec::new();
        let record = agent
            .execute_tool_call(
                &call,
                &mut |event| {
                    if matches!(event, AgentEvent::ToolCallStarted { .. }) {
                        std::fs::remove_file(&raw).expect("remove initial leaf link");
                        symlink(&rebound, &raw).expect("rebind leaf link");
                    }
                    events.push(event);
                    std::future::ready(Ok(()))
                },
                &mut |_| std::future::ready(Ok(PermissionApproval::AllowOnce)),
            )
            .await
            .expect("security failure is recorded");
        assert_eq!(writable_event_phases(&events), vec![true, false], "{tool}");
        assert_eq!(record.status, ToolExecutionStatus::Executed, "{tool}");
        assert_eq!(record.rejection, None, "{tool}");
        assert_eq!(
            record
                .output
                .error
                .as_ref()
                .map(|error| error.message.as_str()),
            Some("writable destination changed after authorization"),
            "{tool}"
        );
        assert_eq!(
            std::fs::read_to_string(&original).unwrap(),
            "original",
            "{tool}"
        );
        assert_eq!(
            std::fs::read_to_string(&rebound).unwrap(),
            "rebound",
            "{tool}"
        );
    }
}

#[tokio::test]
async fn solo_external_workspace_read_executes_without_approval() {
    let mut agent = test_agent();
    agent.set_permission_mode(PermissionMode::Solo);
    let outside_path = std::env::temp_dir().join(format!(
        "letcode-outside-agent-read-{}.txt",
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system time")
            .as_nanos()
    ));
    std::fs::write(&outside_path, "outside\n").expect("write outside fixture");
    let outside = outside_path.to_string_lossy().to_string();
    let call = HistoryToolCall {
        call_id: "call-outside-read".into(),
        name: "fs__read".into(),
        arguments_json: json!({"path": outside, "offset": 1, "limit": 10}).to_string(),
    };
    let mut permission_requests = Vec::new();

    let record = agent
        .execute_tool_call(&call, &mut |_| std::future::ready(Ok(())), &mut |request| {
            permission_requests.push(request);
            std::future::ready(Ok(PermissionApproval::AllowOnce))
        })
        .await
        .expect("outside read should execute in solo mode");

    assert!(record.output.ok, "{:?}", record.output.error);
    assert!(permission_requests.is_empty());
    assert!(
        record
            .output
            .data
            .as_ref()
            .and_then(|data| data.get("content"))
            .and_then(Value::as_str)
            .is_some_and(|content| content.contains("outside"))
    );

    let _ = std::fs::remove_file(outside_path);
}

#[tokio::test]
async fn default_external_workspace_read_allow_always_grants_only_matching_resource() {
    let mut agent = test_agent();
    let fixture_root = std::env::temp_dir().join(format!(
        "letcode-external-read-grant-{}",
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system time")
            .as_nanos()
    ));
    let first_directory = fixture_root.join("first");
    let second_directory = fixture_root.join("second");
    std::fs::create_dir_all(&first_directory).expect("create first fixture directory");
    std::fs::create_dir_all(&second_directory).expect("create second fixture directory");
    let first_path = first_directory.join("read.txt");
    let second_path = second_directory.join("read.txt");
    std::fs::write(&first_path, "first\n").expect("write first fixture");
    std::fs::write(&second_path, "second\n").expect("write second fixture");

    let first_call = HistoryToolCall {
        call_id: "call-external-read-first".into(),
        name: "fs__read".into(),
        arguments_json: json!({"path": first_path, "offset": 1, "limit": 10}).to_string(),
    };
    let repeated_call = HistoryToolCall {
        call_id: "call-external-read-repeated".into(),
        name: "fs__read".into(),
        arguments_json: first_call.arguments_json.clone(),
    };
    let other_directory_call = HistoryToolCall {
        call_id: "call-external-read-other".into(),
        name: "fs__read".into(),
        arguments_json: json!({"path": second_path, "offset": 1, "limit": 10}).to_string(),
    };
    let mut approval_requests = 0;

    let first = agent
        .execute_tool_call(
            &first_call,
            &mut |_| std::future::ready(Ok(())),
            &mut |request| {
                approval_requests += 1;
                assert!(request.can_allow_always);
                std::future::ready(Ok(PermissionApproval::AllowAlways))
            },
        )
        .await
        .expect("first external read should execute after approval");
    assert!(first.output.ok);
    assert_eq!(approval_requests, 1);

    let repeated = agent
        .execute_tool_call(
            &repeated_call,
            &mut |_| std::future::ready(Ok(())),
            &mut |_| {
                approval_requests += 1;
                std::future::ready(Ok(PermissionApproval::Deny))
            },
        )
        .await
        .expect("matching grant should execute without approval");
    assert!(repeated.output.ok);
    assert_eq!(approval_requests, 1, "matching grant must bypass approval");

    let other_directory = agent
        .execute_tool_call(
            &other_directory_call,
            &mut |_| std::future::ready(Ok(())),
            &mut |_| {
                approval_requests += 1;
                std::future::ready(Ok(PermissionApproval::Deny))
            },
        )
        .await
        .expect("other external directory should request approval");
    assert_eq!(approval_requests, 2);
    assert_eq!(other_directory.status, ToolExecutionStatus::Rejected);
    assert_eq!(
        other_directory.rejection,
        Some(ToolExecutionRejection::PermissionDeniedByUser)
    );

    let _ = std::fs::remove_dir_all(fixture_root);
}

#[tokio::test]
async fn matching_grant_does_not_override_base_policy_denial() {
    let mut agent = test_agent();
    let command = "rm -rf letcode-test-target";
    agent
        .permission_session
        .lock()
        .expect("permission session")
        .grant(crate::permission::PermissionResource::Exact {
            tool: "shell__exec".into(),
            value: command.into(),
        });
    let call = HistoryToolCall {
        call_id: "call-granted-policy-denial".into(),
        name: "shell__exec".into(),
        arguments_json: json!({"command": command}).to_string(),
    };
    let mut approval_requested = false;

    let record = agent
        .execute_tool_call(&call, &mut |_| std::future::ready(Ok(())), &mut |_| {
            approval_requested = true;
            std::future::ready(Ok(PermissionApproval::AllowOnce))
        })
        .await
        .expect("policy denial should produce a rejection record");

    assert!(!approval_requested);
    assert_eq!(record.status, ToolExecutionStatus::Rejected);
    assert_eq!(
        record.rejection,
        Some(ToolExecutionRejection::PermissionDeniedByPolicy)
    );
}

#[tokio::test]
async fn solo_external_workspace_write_executes_without_approval() {
    let mut agent = test_agent();
    agent.set_permission_mode(PermissionMode::Solo);
    let outside_path = std::env::temp_dir().join(format!(
        "letcode-outside-agent-denied-write-{}.txt",
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system time")
            .as_nanos()
    ));
    let outside = outside_path.to_string_lossy().to_string();
    let call = HistoryToolCall {
        call_id: "call-outside-write-denied".into(),
        name: "fs__write".into(),
        arguments_json: json!({"path": outside, "content": "denied"}).to_string(),
    };
    let mut permission_requests = Vec::new();
    let mut events = Vec::new();

    let record = agent
        .execute_tool_call(
            &call,
            &mut |event| {
                events.push(event);
                std::future::ready(Ok(()))
            },
            &mut |request| {
                permission_requests.push(request);
                std::future::ready(Ok(PermissionApproval::Deny))
            },
        )
        .await
        .expect("outside write should execute in solo mode");

    assert!(permission_requests.is_empty());
    assert_eq!(record.status, ToolExecutionStatus::Executed);
    assert!(record.output.ok);
    assert!(outside_path.exists());
    assert!(
        events
            .iter()
            .any(|event| matches!(event, AgentEvent::ToolCallFinished { ok: true, .. }))
    );
    let _ = std::fs::remove_file(outside_path);
}

#[tokio::test]
async fn default_and_safe_allow_once_authorize_external_writes() {
    for mode in [PermissionMode::Default, PermissionMode::Safe] {
        let mut agent = test_agent();
        agent.set_permission_mode(mode);
        let outside_path = std::env::temp_dir().join(format!(
            "letcode-external-write-allow-once-{mode:?}-{}",
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .expect("system time")
                .as_nanos()
        ));
        let call = HistoryToolCall {
            call_id: format!("call-external-write-allow-once-{mode:?}"),
            name: "fs__write".into(),
            arguments_json: json!({"path": outside_path, "content": "approved"}).to_string(),
        };
        let mut approvals = 0;
        let mut events = Vec::new();
        let record = agent
            .execute_tool_call(
                &call,
                &mut |event| {
                    events.push(event);
                    std::future::ready(Ok(()))
                },
                &mut |_| {
                    approvals += 1;
                    std::future::ready(Ok(PermissionApproval::AllowOnce))
                },
            )
            .await
            .expect("approved write should execute");
        assert_eq!(approvals, 1);
        assert_eq!(record.status, ToolExecutionStatus::Executed);
        assert!(record.output.ok, "{:?}", record.output.error);
        assert!(
            events
                .iter()
                .any(|event| matches!(event, AgentEvent::ToolCallStarted { .. }))
        );
        assert!(
            events
                .iter()
                .any(|event| matches!(event, AgentEvent::ToolCallFinished { ok: true, .. }))
        );
        let _ = std::fs::remove_file(outside_path);
    }
}

#[cfg(unix)]
#[tokio::test]
async fn default_allow_always_external_write_grant_is_cleared_at_top_level_session_boundary() {
    let fixture = UnixWritableFixture::new("allow-always-session-boundary");
    let path = fixture.external.join("written.txt");
    let first_call = writable_call("allow-always-first", "fs__write", &path, "first");
    let repeated_call = writable_call("allow-always-repeated", "fs__write", &path, "second");
    let after_reset_call = writable_call("allow-always-after-reset", "fs__write", &path, "denied");
    let mut agent = test_agent();
    let mut approvals = 0;

    let first = agent
        .execute_tool_call(
            &first_call,
            &mut |_| std::future::ready(Ok(())),
            &mut |request| {
                approvals += 1;
                assert!(request.can_allow_always);
                std::future::ready(Ok(PermissionApproval::AllowAlways))
            },
        )
        .await
        .expect("first AllowAlways write executes");
    assert!(first.output.ok, "{:?}", first.output.error);
    assert_eq!(approvals, 1);
    assert_eq!(
        std::fs::read_to_string(&path).expect("first write"),
        "first"
    );

    let repeated = agent
        .execute_tool_call(
            &repeated_call,
            &mut |_| std::future::ready(Ok(())),
            &mut |_| {
                approvals += 1;
                std::future::ready(Ok(PermissionApproval::Deny))
            },
        )
        .await
        .expect("matching session grant executes");
    assert!(repeated.output.ok, "{:?}", repeated.output.error);
    assert_eq!(approvals, 1, "matching destination must reuse the grant");
    assert_eq!(
        std::fs::read_to_string(&path).expect("repeated write"),
        "second"
    );

    // This is the production top-level session transition, not a replacement
    // Agent instance: it must discard the grant held by this Agent.
    agent.reset_for_new_session();
    let after_reset = agent
        .execute_tool_call(
            &after_reset_call,
            &mut |_| std::future::ready(Ok(())),
            &mut |request| {
                approvals += 1;
                assert!(request.can_allow_always);
                std::future::ready(Ok(PermissionApproval::Deny))
            },
        )
        .await
        .expect("post-reset denial is recorded");
    assert_eq!(approvals, 2, "reset session must request approval again");
    assert_eq!(after_reset.status, ToolExecutionStatus::Rejected);
    assert_eq!(
        after_reset.rejection,
        Some(ToolExecutionRejection::PermissionDeniedByUser)
    );
    assert_eq!(
        std::fs::read_to_string(&path).expect("denial has no write"),
        "second"
    );
}

#[cfg(unix)]
#[tokio::test]
async fn approved_external_write_rebinding_is_rejected_after_started() {
    use std::os::unix::fs::symlink;

    let unique = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system time")
        .as_nanos();
    let first = std::env::temp_dir().join(format!("letcode-rebind-first-{unique}"));
    let second = std::env::temp_dir().join(format!("letcode-rebind-second-{unique}"));
    let link = PathBuf::from("target").join(format!("letcode-rebind-link-{unique}"));
    std::fs::create_dir_all(&first).expect("create first fixture");
    std::fs::create_dir_all(&second).expect("create second fixture");
    std::fs::create_dir_all("target").expect("create target fixture");
    symlink(&first, &link).expect("create initial link");
    let raw = link.join("target.txt");

    let mut agent = test_agent();
    let call = HistoryToolCall {
        call_id: "call-rebound-external-write".into(),
        name: "fs__write".into(),
        arguments_json: json!({"path": raw, "content": "must not write"}).to_string(),
    };
    let mut events = Vec::new();
    let record = agent
        .execute_tool_call(
            &call,
            &mut |event| {
                events.push(event);
                std::future::ready(Ok(()))
            },
            &mut |_| {
                std::fs::remove_file(&link).expect("remove initial link");
                symlink(&second, &link).expect("rebind link");
                std::future::ready(Ok(PermissionApproval::AllowOnce))
            },
        )
        .await
        .expect("rebound write should produce a tool record");

    assert_eq!(record.status, ToolExecutionStatus::Executed);
    assert_eq!(record.rejection, None);
    assert!(!record.output.ok);
    assert!(record.output.error.as_ref().is_some_and(|error| {
        error.message == "writable destination changed after authorization"
    }));
    assert!(
        events
            .iter()
            .any(|event| matches!(event, AgentEvent::ToolCallStarted { .. }))
    );
    assert!(
        events
            .iter()
            .any(|event| matches!(event, AgentEvent::ToolCallFinished { ok: false, .. }))
    );
    assert!(!first.join("target.txt").exists());
    assert!(!second.join("target.txt").exists());
    let _ = std::fs::remove_file(link);
    let _ = std::fs::remove_dir_all(first);
    let _ = std::fs::remove_dir_all(second);
}

#[cfg(unix)]
#[tokio::test]
async fn started_callback_rebinding_is_rejected_without_writing_either_destination() {
    use std::os::unix::fs::symlink;

    let unique = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system time")
        .as_nanos();
    let first = std::env::temp_dir().join(format!("letcode-started-rebind-first-{unique}"));
    let second = std::env::temp_dir().join(format!("letcode-started-rebind-second-{unique}"));
    let link = PathBuf::from("target").join(format!("letcode-started-rebind-link-{unique}"));
    std::fs::create_dir_all(&first).expect("create first fixture");
    std::fs::create_dir_all(&second).expect("create second fixture");
    std::fs::create_dir_all("target").expect("create target fixture");
    symlink(&first, &link).expect("create initial link");
    let raw = link.join("target.txt");

    let mut agent = test_agent();
    let call = HistoryToolCall {
        call_id: "call-started-rebound-external-write".into(),
        name: "fs__write".into(),
        arguments_json: json!({"path": raw, "content": "must not write"}).to_string(),
    };
    let mut events = Vec::new();
    let record = agent
        .execute_tool_call(
            &call,
            &mut |event| {
                if matches!(event, AgentEvent::ToolCallStarted { .. }) {
                    std::fs::remove_file(&link).expect("remove initial link");
                    symlink(&second, &link).expect("rebind link");
                }
                events.push(event);
                std::future::ready(Ok(()))
            },
            &mut |_| std::future::ready(Ok(PermissionApproval::AllowOnce)),
        )
        .await
        .expect("rebound write should produce a tool record");

    assert_eq!(record.status, ToolExecutionStatus::Executed);
    assert_eq!(record.rejection, None);
    assert!(!record.output.ok);
    assert_eq!(
        record
            .output
            .error
            .as_ref()
            .map(|error| error.message.as_str()),
        Some("writable destination changed after authorization")
    );
    assert!(
        events
            .iter()
            .any(|event| matches!(event, AgentEvent::ToolCallStarted { .. }))
    );
    assert!(
        events
            .iter()
            .any(|event| matches!(event, AgentEvent::ToolCallFinished { ok: false, .. }))
    );
    assert!(!first.join("target.txt").exists());
    assert!(!second.join("target.txt").exists());
    let _ = std::fs::remove_file(link);
    let _ = std::fs::remove_dir_all(first);
    let _ = std::fs::remove_dir_all(second);
}

#[cfg(unix)]
#[tokio::test]
async fn leaf_symlink_callback_rebinding_is_post_started_security_failure() {
    use std::os::unix::fs::symlink;

    let unique = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let first = std::env::temp_dir().join(format!("letcode-leaf-callback-a-{unique}"));
    let second = std::env::temp_dir().join(format!("letcode-leaf-callback-b-{unique}"));
    let link = PathBuf::from("target").join(format!("letcode-leaf-callback-{unique}"));
    std::fs::create_dir_all("target").unwrap();
    std::fs::write(&first, "A").unwrap();
    std::fs::write(&second, "B").unwrap();
    symlink(&first, &link).unwrap();
    let call = HistoryToolCall {
        call_id: "call-leaf-callback-rebind".into(),
        name: "fs__write".into(),
        arguments_json: json!({"path": link, "content": "must not write"}).to_string(),
    };
    let mut agent = test_agent();
    let mut events = Vec::new();
    let record = agent
        .execute_tool_call(
            &call,
            &mut |event| {
                events.push(event);
                std::future::ready(Ok(()))
            },
            &mut |_| {
                std::fs::remove_file(&link).unwrap();
                symlink(&second, &link).unwrap();
                std::future::ready(Ok(PermissionApproval::AllowOnce))
            },
        )
        .await
        .expect("security failure is a completed execution record");

    assert_eq!(record.status, ToolExecutionStatus::Executed);
    assert_eq!(record.rejection, None);
    assert!(!record.output.ok);
    assert_eq!(
        record
            .output
            .error
            .as_ref()
            .map(|error| error.message.as_str()),
        Some("writable destination changed after authorization")
    );
    assert!(
        events
            .iter()
            .any(|event| matches!(event, AgentEvent::ToolCallStarted { .. }))
    );
    assert!(
        events
            .iter()
            .any(|event| matches!(event, AgentEvent::ToolCallFinished { ok: false, .. }))
    );
    assert_eq!(std::fs::read_to_string(&first).unwrap(), "A");
    assert_eq!(std::fs::read_to_string(&second).unwrap(), "B");
    let _ = std::fs::remove_file(link);
    let _ = std::fs::remove_file(first);
    let _ = std::fs::remove_file(second);
}

#[tokio::test]
async fn writable_leaf_preparation_failure_skips_approval_and_started_event() {
    let unique = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let call = test_tool_call(
        "fs__write",
        &json!({"path": format!("target/missing-parent-{unique}/leaf"), "content": "x"})
            .to_string(),
    );
    let mut agent = test_agent();
    let mut approvals = 0;
    let mut events = Vec::new();
    let record = agent
        .execute_tool_call(
            &call,
            &mut |event| {
                events.push(event);
                std::future::ready(Ok(()))
            },
            &mut |_| {
                approvals += 1;
                std::future::ready(Ok(PermissionApproval::AllowOnce))
            },
        )
        .await
        .expect("preparation failure becomes a rejection record");

    assert_eq!(approvals, 0);
    assert_eq!(record.status, ToolExecutionStatus::Rejected);
    assert_eq!(record.rejection, None);
    assert!(
        record
            .output
            .error
            .as_ref()
            .unwrap()
            .message
            .contains("parent directory does not exist")
    );
    assert!(
        events
            .iter()
            .any(|event| matches!(event, AgentEvent::ToolCallFinished { ok: false, .. }))
    );
    assert!(
        !events
            .iter()
            .any(|event| matches!(event, AgentEvent::ToolCallStarted { .. }))
    );
}

#[tokio::test]
async fn solo_mode_executes_commands_that_default_mode_denies_by_policy() {
    let mut agent = test_agent();
    agent.set_permission_mode(PermissionMode::Solo);
    let call = HistoryToolCall {
        call_id: "call-solo-deny-risk".into(),
        name: "shell__exec".into(),
        arguments_json: json!({"command": "curl --version"}).to_string(),
    };
    let mut approval_requested = false;

    let record = agent
        .execute_tool_call(&call, &mut |_| std::future::ready(Ok(())), &mut |_| {
            approval_requested = true;
            std::future::ready(Ok(PermissionApproval::Deny))
        })
        .await
        .expect("solo mode should execute command without asking");

    assert!(!approval_requested);
    assert_eq!(record.status, ToolExecutionStatus::Executed);
    assert_ne!(
        record.rejection,
        Some(ToolExecutionRejection::PermissionDeniedByPolicy)
    );
}

#[tokio::test]
async fn unfinished_todos_trigger_bounded_internal_continuation() {
    let mut agent = test_agent();
    agent.prepare_turn_prelude("implement a feature");
    let turn_id = agent.current_turn_id();
    agent.turn.workflow.auto_continue = AutoContinueState {
        enabled: true,
        max_continuations: 2,
    };
    agent.turn.workflow.todos = vec![TodoItem {
        id: "t1".into(),
        content: "keep going".into(),
        status: TodoStatus::InProgress,
    }];
    let mut continuation_count = 0;
    let mut events = Vec::new();

    let should_continue = agent
        .continue_after_no_tool_reply(
            &mut |event| {
                events.push(event);
                std::future::ready(Ok(()))
            },
            &mut continuation_count,
        )
        .await
        .expect("continuation decision succeeds");

    assert!(should_continue);
    assert_eq!(continuation_count, 1);
    assert_eq!(agent.current_turn_id(), turn_id);
    assert!(matches!(
        agent.history.last(),
        Some(HistoryItem::InternalContinuation { .. })
    ));
    assert!(matches!(
        events.as_slice(),
        [
            AgentEvent::InternalContinuation {
                source: crate::transcript::InternalContinuationSource::AutoContinue,
                ..
            },
            AgentEvent::AutoContinuationScheduled {
                continuation_count: 1,
                remaining_unfinished: 1,
            }
        ]
    ));
}

#[tokio::test]
async fn auto_continue_stops_when_todos_do_not_progress() {
    let mut agent = test_agent();
    agent.turn.workflow.auto_continue = AutoContinueState {
        enabled: true,
        max_continuations: 3,
    };
    agent.turn.workflow.todos = vec![TodoItem {
        id: "t1".into(),
        content: "still pending".into(),
        status: TodoStatus::Pending,
    }];
    let mut continuation_count = 0;

    assert!(
        agent
            .continue_after_no_tool_reply(
                &mut |_| std::future::ready(Ok(())),
                &mut continuation_count
            )
            .await
            .expect("first continuation should proceed")
    );

    let error = agent
        .continue_after_no_tool_reply(&mut |_| std::future::ready(Ok(())), &mut continuation_count)
        .await
        .expect_err("unchanged todo snapshot should stop");

    assert!(error.to_string().contains("no todo progress"));
    assert_eq!(continuation_count, 1);
}

#[tokio::test]
async fn completed_or_blocked_todos_stop_auto_continuation() {
    let mut agent = test_agent();
    agent.turn.workflow.auto_continue.enabled = true;
    let mut continuation_count = 0;

    agent.turn.workflow.todos = vec![TodoItem {
        id: "done".into(),
        content: "done".into(),
        status: TodoStatus::Completed,
    }];
    assert!(
        !agent
            .continue_after_no_tool_reply(
                &mut |_| std::future::ready(Ok(())),
                &mut continuation_count
            )
            .await
            .expect("completed todos should stop")
    );

    agent.turn.workflow.todos = vec![TodoItem {
        id: "blocked".into(),
        content: "blocked".into(),
        status: TodoStatus::Blocked,
    }];
    assert!(
        !agent
            .continue_after_no_tool_reply(
                &mut |_| std::future::ready(Ok(())),
                &mut continuation_count
            )
            .await
            .expect("blocked todos should stop")
    );
}

#[tokio::test]
async fn continuation_bound_is_runtime_enforced() {
    let mut agent = test_agent();
    agent.turn.workflow.auto_continue = AutoContinueState {
        enabled: true,
        max_continuations: 1,
    };
    agent.turn.workflow.todos = vec![TodoItem {
        id: "t1".into(),
        content: "still pending".into(),
        status: TodoStatus::Pending,
    }];
    let mut continuation_count = 1;

    let error = agent
        .continue_after_no_tool_reply(&mut |_| std::future::ready(Ok(())), &mut continuation_count)
        .await
        .expect_err("limit should fail fast");

    assert!(error.to_string().contains("auto-continue limit reached"));
    assert_eq!(continuation_count, 1);
}

#[test]
fn engineering_turn_prelude_adds_workflow_context_and_validation_reminder() {
    let client = Client::with_config(
        OpenAIConfig::new()
            .with_api_base("https://api.openai.com/v1")
            .with_api_key("test"),
    );
    let mut agent = Agent::new(client, "m1", 1, 1);

    let turn_prelude =
        agent.prepare_turn_prelude("Implement the fix in src/agent.rs and run cargo test.");

    assert_eq!(agent.current_turn().intent, TurnIntent::Engineering);
    assert_eq!(agent.current_turn().directive, ExecutionDirective::None);
    assert_eq!(turn_prelude.len(), agent.prelude.len() + 2);
    let runtime_message = &turn_prelude[turn_prelude.len() - 2];
    assert_eq!(
        runtime_message.role,
        crate::request_builder::PromptRole::Developer
    );
    assert!(runtime_message.text.contains("Runtime context"));
    assert!(runtime_message.text.contains("Current date:"));
    assert!(runtime_message.text.contains("Timezone:"));
    assert!(!runtime_message.text.contains("Current time:"));
    let workflow_message = &turn_prelude[turn_prelude.len() - 1];
    assert_eq!(
        workflow_message.role,
        crate::request_builder::PromptRole::Developer
    );
    assert!(workflow_message.text.contains("engineering workflow task"));
    assert!(workflow_message.text.contains("Delegate bounded work"));
    assert!(workflow_message.text.contains("context hygiene"));
    assert!(workflow_message.text.contains("targeted validation"));
}

#[test]
fn lightweight_turn_prelude_adds_only_runtime_context() {
    let client = Client::with_config(
        OpenAIConfig::new()
            .with_api_base("https://api.openai.com/v1")
            .with_api_key("test"),
    );
    let mut agent = Agent::new(client, "m1", 1, 1);

    let turn_prelude = agent.prepare_turn_prelude("Summarize what this tool does.");

    assert_eq!(agent.current_turn().intent, TurnIntent::Lightweight);
    assert_eq!(turn_prelude.len(), agent.prelude.len() + 1);
    assert_eq!(
        &turn_prelude[..agent.prelude.len()],
        agent.prelude.as_slice()
    );
    let runtime_message = turn_prelude.last().expect("runtime context present");
    assert_eq!(
        runtime_message.role,
        crate::request_builder::PromptRole::Developer
    );
    assert!(runtime_message.text.contains("Runtime context"));
    assert!(runtime_message.text.contains("Current date:"));
    assert!(runtime_message.text.contains("Timezone:"));
    assert!(!runtime_message.text.contains("Current time:"));
}

#[test]
fn normalize_session_title_trims_and_strips_wrapping_quotes() {
    assert_eq!(
        normalize_session_title("  \"Fix startup crash in CI\"  ").expect("normalize title"),
        "Fix startup crash in CI"
    );
    assert_eq!(
        normalize_session_title("`Debug flaky transcript tests`\nextra").expect("normalize title"),
        "Debug flaky transcript tests"
    );
}

#[test]
fn session_title_agent_has_no_tools_or_history() {
    let mut agent = test_agent();
    agent.restore_transcript_messages(vec![ConversationMessage {
        role: ConversationRole::User,
        content: "existing conversation".into(),
    }]);
    let title_agent = agent.session_title_agent();

    assert!(title_agent.history.is_empty());
    assert!(title_agent.runtime_snapshot.evidence.is_empty());
    assert!(title_agent.tools.specs().is_empty());
    assert_eq!(title_agent.model(), agent.model());
}

#[test]
fn turn_prelude_injects_skill_cards_without_skill_body() {
    let mut agent = test_agent();
    agent
        .register_skill_registry(test_skill_registry())
        .expect("register skill registry");

    let turn_prelude = agent.prepare_turn_prelude("Summarize the available tools.");
    let skill_message = turn_prelude
        .iter()
        .find(|message| message.text.contains("Available local skills:"))
        .expect("skill prelude message present");

    assert!(
        skill_message
            .text
            .contains("Load relevant skills with the `skill` tool when needed.")
    );
    assert!(
        skill_message
            .text
            .contains("rust-audit — Inspect Rust code")
    );
    assert!(skill_message.text.contains("source: .letcode/skills"));
    assert!(
        !skill_message
            .text
            .contains("/workspace/.letcode/skills/rust-audit/SKILL.md")
    );
    assert!(!skill_message.text.contains("# Private body"));
    assert!(
        skill_message
            .text
            .contains("Skills do not change permissions or expand tool scope.")
    );
}

#[test]
fn register_skill_registry_registers_skill_resource_tools() {
    let mut agent = test_agent();
    agent
        .register_skill_registry(test_skill_registry())
        .expect("register skill registry");

    let specs = agent.tool_definitions();
    for name in ["skill", "skill__resource_list", "skill__resource_read"] {
        assert!(
            specs.iter().any(|spec| spec.name == name),
            "{name} should be registered"
        );
    }
}

#[test]
fn empty_skill_registry_does_not_register_skill_tool_or_prelude() {
    let mut agent = test_agent();
    agent
        .register_skill_registry(Arc::new(SkillRegistry::default()))
        .expect("register empty skill registry");

    assert!(!agent.tool_definitions().iter().any(|spec| {
        matches!(
            spec.name.as_str(),
            "skill" | "skill__resource_list" | "skill__resource_read"
        )
    }));
    let turn_prelude = agent.prepare_turn_prelude("Summarize this project.");
    assert!(
        !turn_prelude
            .iter()
            .any(|message| message.text.contains("Available local skills:"))
    );
}

#[test]
fn runtime_context_message_contains_date_and_timezone_only() {
    let message = runtime_context_message_from_parts("2026-06-18", "Asia/Shanghai");

    assert_eq!(message.role, crate::request_builder::PromptRole::Developer);
    assert!(message.text.contains("Runtime context:"));
    assert!(message.text.contains("Current date: 2026-06-18"));
    assert!(message.text.contains("Timezone: Asia/Shanghai"));
    assert!(!message.text.contains("Current time:"));
    assert!(!message.text.contains("09:43"));
}

#[test]
fn utc_date_from_unix_days_formats_calendar_dates() {
    assert_eq!(utc_date_from_unix_days(0), "1970-01-01");
    assert_eq!(utc_date_from_unix_days(20_622), "2026-06-18");
}

#[test]
fn detects_explicit_execution_directives() {
    assert_eq!(
        detect_execution_directive("Read-only: inspect src/permission.rs and summarize it."),
        ExecutionDirective::ReadOnly
    );
    assert_eq!(
        detect_execution_directive("Plan only. Do not edit anything yet."),
        ExecutionDirective::PlanOnly
    );
    assert_eq!(
        detect_execution_directive("Analyze only and explain the failure."),
        ExecutionDirective::AnalyzeOnly
    );
    assert_eq!(
        detect_execution_directive("Please investigate, but do not edit files."),
        ExecutionDirective::DoNotEdit
    );
}

#[tokio::test]
async fn execute_tool_call_blocks_write_tools_under_read_only_directive() {
    let client = Client::with_config(
        OpenAIConfig::new()
            .with_api_base("https://api.openai.com/v1")
            .with_api_key("test"),
    );
    let mut agent = Agent::new(client, "m1", 1, 1);
    agent.turn = TurnRuntimeState::new(
        1,
        WorkflowTurnState::from_user_input("Read-only: inspect and report only."),
    );

    let call = HistoryToolCall {
        call_id: "call-1".into(),
        name: "fs__write".into(),
        arguments_json: r#"{"path":"a.txt","content":"x"}"#.into(),
    };
    let mut events = Vec::new();

    let record = agent
        .execute_tool_call(
            &call,
            &mut |event| {
                events.push(event);
                std::future::ready(Ok(()))
            },
            &mut |_| std::future::ready(Ok(PermissionApproval::AllowOnce)),
        )
        .await
        .expect("tool call should complete with visible error");

    assert!(!record.output.ok);
    assert!(
        record
            .output
            .error
            .as_ref()
            .expect("error payload")
            .message
            .contains("read_only directive")
    );
    assert!(matches!(
        events.as_slice(),
        [
            AgentEvent::ToolCallFinished { .. },
            AgentEvent::ToolExecutionSummary(ToolExecutionSummaryEvent {
                status,
                rejection: Some(rejection),
                effect_kind,
                ..
            })
        ] if status == "rejected"
                && rejection == "directive_blocked"
                && effect_kind == "diagnostic"
    ));
    assert_eq!(record.status, ToolExecutionStatus::Rejected);
    assert_eq!(
        record.rejection,
        Some(ToolExecutionRejection::DirectiveBlocked)
    );
    assert_eq!(record.effects.kind, ToolEffectKind::Diagnostic);
}

#[tokio::test]
async fn execute_tool_call_blocks_non_read_only_commands_under_read_only_directive() {
    let client = Client::with_config(
        OpenAIConfig::new()
            .with_api_base("https://api.openai.com/v1")
            .with_api_key("test"),
    );
    let mut agent = Agent::new(client, "m1", 1, 1);
    agent.turn = TurnRuntimeState::new(
        1,
        WorkflowTurnState::from_user_input("Read only. Analyze and report."),
    );

    let call = HistoryToolCall {
        call_id: "call-2".into(),
        name: "shell__exec".into(),
        arguments_json: r#"{"command":"cargo test permission::tests"}"#.into(),
    };

    let record = agent
        .execute_tool_call(&call, &mut |_| std::future::ready(Ok(())), &mut |_| {
            std::future::ready(Ok(PermissionApproval::AllowOnce))
        })
        .await
        .expect("tool call should complete with visible error");

    assert!(!record.output.ok);
    assert!(
        record
            .output
            .error
            .as_ref()
            .expect("error payload")
            .message
            .contains("not read-only compatible")
    );
}

#[tokio::test]
async fn execute_tool_call_emits_finished_event_for_policy_denial() {
    let client = Client::with_config(
        OpenAIConfig::new()
            .with_api_base("https://api.openai.com/v1")
            .with_api_key("test"),
    );
    let mut agent = Agent::new(client, "m1", 1, 1);
    let call = HistoryToolCall {
        call_id: "call-denied".into(),
        name: "shell__exec".into(),
        arguments_json: r#"{"command":"rm -rf target"}"#.into(),
    };
    let mut events = Vec::new();

    let record = agent
        .execute_tool_call(
            &call,
            &mut |event| {
                events.push(event);
                std::future::ready(Ok(()))
            },
            &mut |_| std::future::ready(Ok(PermissionApproval::AllowOnce)),
        )
        .await
        .expect("policy denial should be reported as tool output");

    assert!(!record.output.ok);
    assert!(matches!(
        events.as_slice(),
        [
            AgentEvent::ToolCallFinished { ok: false, .. },
            AgentEvent::ToolExecutionSummary(ToolExecutionSummaryEvent {
                status,
                rejection: Some(rejection),
                effect_kind,
                ..
            })
        ] if status == "rejected"
                && rejection == "permission_denied_by_policy"
                && effect_kind == "diagnostic"
    ));
    assert_eq!(record.status, ToolExecutionStatus::Rejected);
    assert_eq!(
        record.rejection,
        Some(ToolExecutionRejection::PermissionDeniedByPolicy)
    );
    assert_eq!(record.effects.kind, ToolEffectKind::Diagnostic);
}

#[tokio::test]
async fn execute_tool_call_invalid_json_emits_finished_event_and_records_rejection() {
    let mut agent = test_agent();
    let call = test_tool_call("fs__write", r#"{"path":"a.txt","content": }"#);
    let mut events = Vec::new();

    let record = agent
        .execute_tool_call(
            &call,
            &mut |event| {
                events.push(event);
                std::future::ready(Ok(()))
            },
            &mut |_| std::future::ready(Ok(PermissionApproval::AllowOnce)),
        )
        .await
        .expect("invalid json should still produce a record");

    assert_eq!(record.status, ToolExecutionStatus::Rejected);
    assert_eq!(
        record.rejection,
        Some(ToolExecutionRejection::InvalidJsonArguments)
    );
    assert!(!record.output.ok);
    assert_eq!(record.arguments, None);
    assert_eq!(record.effects.kind, ToolEffectKind::Diagnostic);
    assert!(matches!(
        events.as_slice(),
        [
            AgentEvent::ToolCallFinished { ok: false, .. },
            AgentEvent::ToolExecutionSummary(ToolExecutionSummaryEvent {
                status,
                rejection: Some(rejection),
                effect_kind,
                ..
            })
        ] if status == "rejected"
            && rejection == "invalid_json_arguments"
            && effect_kind == "diagnostic"
    ));
}

#[tokio::test]
async fn audit_event_failures_do_not_fail_tool_execution() {
    let mut agent = test_agent();
    let call = test_tool_call("fs__write", r#"{"path":"a.txt","content": }"#);
    let mut event_count = 0;

    let record = agent
        .execute_tool_call(
            &call,
            &mut |event| {
                assert!(matches!(
                    event,
                    AgentEvent::ToolCallFinished { .. } | AgentEvent::ToolExecutionSummary(_)
                ));
                event_count += 1;
                if matches!(event, AgentEvent::ToolExecutionSummary(_)) {
                    std::future::ready(Err(anyhow!("audit sink failed")))
                } else {
                    std::future::ready(Ok(()))
                }
            },
            &mut |_| std::future::ready(Ok(PermissionApproval::AllowOnce)),
        )
        .await
        .expect("audit failure should not fail tool execution");

    assert_eq!(event_count, 2);
    assert_eq!(record.status, ToolExecutionStatus::Rejected);
    assert_eq!(
        record.rejection,
        Some(ToolExecutionRejection::InvalidJsonArguments)
    );
}

#[test]
fn pending_validation_advisory_only_emits_for_write_without_validation() {
    let mut agent = test_agent();
    assert!(agent.pending_validation_advisory().is_none());

    agent.turn.counters.write_effects = 1;
    let advisory = agent
        .pending_validation_advisory()
        .expect("write without validation should emit advisory");
    assert_eq!(advisory.write_effects, 1);
    assert_eq!(advisory.validation_effects, 0);
    assert_eq!(advisory.failed_validation_effects, 0);
    assert!(advisory.message.contains("without running validation"));

    agent.turn.counters.failed_validation_effects = 1;
    let advisory = agent
        .pending_validation_advisory()
        .expect("failed validation should emit advisory");
    assert_eq!(advisory.write_effects, 1);
    assert_eq!(advisory.validation_effects, 0);
    assert_eq!(advisory.failed_validation_effects, 1);
    assert!(advisory.message.contains("validation ran but failed"));

    agent.turn.counters.validation_effects = 1;
    let advisory = agent
        .pending_validation_advisory()
        .expect("failed validation should continue to emit advisory");
    assert_eq!(advisory.validation_effects, 1);
    assert_eq!(advisory.failed_validation_effects, 1);
    assert!(advisory.message.contains("validation ran but failed"));
}

#[test]
fn pending_validation_advisory_includes_child_write_and_validation_failures() {
    let mut agent = test_agent();
    agent.turn.counters.child_write_effects = 2;
    let advisory = agent
        .pending_validation_advisory()
        .expect("child writes without validation should emit advisory");
    assert_eq!(advisory.write_effects, 2);
    assert_eq!(advisory.validation_effects, 0);
    assert!(advisory.message.contains("delegated child work"));

    agent.turn.counters.child_validation_effects = 1;
    agent.turn.counters.child_failed_validation_effects = 1;
    let advisory = agent
        .pending_validation_advisory()
        .expect("child validation failures should emit advisory");
    assert_eq!(advisory.failed_validation_effects, 1);
    assert!(advisory.message.contains("validation failed"));
}

#[test]
fn prepare_turn_prelude_includes_unreconciled_subagent_results() {
    let mut agent = test_agent();
    agent
        .add_evidence(
            EvidenceDraft {
                id: Some("ev-1".into()),
                evidence_kind: crate::evidence::EvidenceKind::Decision,
                title: "subagent result".into(),
                summary: "child completed".into(),
                detail: Some(
                    serde_json::to_string(&crate::subagent::StructuredSubagentResult {
                        status: "completed".into(),
                        summary: "child completed".into(),
                        malformed: false,
                        findings: vec![],
                        files_read: vec![],
                        files_changed: vec![],
                        commands_run: vec![],
                        validation: vec![],
                        blockers: vec![],
                        next_steps: vec![],
                        run_id: "run-1".into(),
                        child_session_id: "child-1".into(),
                        raw_excerpt: None,
                    })
                    .expect("serialize structured result"),
                ),
                source: EvidenceSource::Subagent {
                    run_id: "run-1".into(),
                    child_session_id: "child-1".into(),
                    source_session_id: "child-1".into(),
                    parent_tool: "agent__explore".into(),
                    parent_turn_id: Some("turn-1".into()),
                    parent_session_id: None,
                },
                tags: vec![
                    "explorer".into(),
                    "subagent_result".into(),
                    "unreconciled".into(),
                ],
            }
            .into_record("ev-1".into(), 1, 0)
            .expect("build evidence"),
        )
        .expect("add evidence");

    let prelude = agent.prepare_turn_prelude("Implement next step");
    assert!(prelude.iter().any(|message| {
        message.text.contains("Pending child subagent results")
            && message.text.contains("agent__reconcile")
            && message.text.contains("run-1")
            && message.text.contains("child completed")
            && message.text.contains("child_session_id=child-1")
            && message.text.contains("child transcript navigation")
    }));
}

#[test]
fn pending_subagent_jobs_clear_after_live_reconciliation_evidence() {
    let mut agent = test_agent();
    agent
        .add_evidence(
            EvidenceDraft {
                id: Some("ev-result".into()),
                evidence_kind: crate::evidence::EvidenceKind::Decision,
                title: "subagent result".into(),
                summary: "child completed".into(),
                detail: Some(
                    serde_json::to_string(&crate::subagent::StructuredSubagentResult {
                        status: "completed".into(),
                        summary: "child completed".into(),
                        malformed: false,
                        findings: vec![],
                        files_read: vec![],
                        files_changed: vec![],
                        commands_run: vec![],
                        validation: vec![],
                        blockers: vec![],
                        next_steps: vec![],
                        run_id: "run-1".into(),
                        child_session_id: "child-1".into(),
                        raw_excerpt: None,
                    })
                    .expect("serialize structured result"),
                ),
                source: EvidenceSource::Subagent {
                    run_id: "run-1".into(),
                    child_session_id: "child-1".into(),
                    source_session_id: "child-1".into(),
                    parent_tool: "agent__explore".into(),
                    parent_turn_id: Some("turn-1".into()),
                    parent_session_id: None,
                },
                tags: vec![
                    "explorer".into(),
                    "subagent_result".into(),
                    "unreconciled".into(),
                ],
            }
            .into_record("ev-result".into(), 1, 0)
            .expect("build evidence"),
        )
        .expect("add result evidence");
    assert_eq!(agent.pending_subagent_jobs().len(), 1);

    let record = ToolExecutionRecord::new(
        &test_tool_call(
            tool_names::TOOL_AGENT_RECONCILE,
            r#"{"run_id":"run-1","child_session_id":"child-1","agent_name":"explorer","decision":"accepted","summary":"accepted child result"}"#,
        ),
        Some(json!({
            "run_id": "run-1",
            "child_session_id": "child-1",
            "agent_name": "explorer",
            "decision": "accepted",
            "summary": "accepted child result"
        })),
        crate::permission::ToolPermissionClass::Preview,
        ExecutionDirective::None,
        ToolExecutionStatus::Executed,
        None,
        ToolResult::ok(
            tool_names::TOOL_AGENT_RECONCILE,
            json!({
                "run_id": "run-1",
                "child_session_id": "child-1",
                "agent_name": "explorer",
                "decision": "accepted",
                "summary": "accepted child result",
                "reconciled": true,
                "pending_recording": true
            }),
        ),
    );
    agent.record_tool_effects(&record);
    let evidence = agent
        .remember_tool_evidence(&record)
        .expect("record live reconciliation evidence");
    assert!(
        evidence
            .tags
            .iter()
            .any(|tag| tag == "subagent_reconciliation")
    );
    assert!(evidence.tags.iter().any(|tag| tag == "reconciled"));
    assert_eq!(agent.pending_subagent_jobs().len(), 0);
}

#[test]
fn default_prelude_and_engineering_guidance_frame_non_trivial_work_as_orchestration() {
    assert!(DEFAULT_AGENT_PRELUDE.contains("workflow manager first"));
    assert!(
        DEFAULT_AGENT_PRELUDE
            .contains("Direct execution is for trivial, single-file, clearly bounded work")
    );
    assert!(ENGINEERING_WORKFLOW_PRELUDE.contains("specialist lane is needed"));
    assert!(ENGINEERING_WORKFLOW_PRELUDE.contains("explorer for broad or unknown code search"));
    assert!(ENGINEERING_WORKFLOW_PRELUDE.contains("prefer completed or reconciled sessions"));
    assert!(ENGINEERING_WORKFLOW_PRELUDE.contains("Never reuse cancelled or errored sessions"));
    assert!(ENGINEERING_WORKFLOW_PRELUDE.contains("only one active subagent"));
    assert!(ENGINEERING_WORKFLOW_PRELUDE.contains("Delegates do not queue"));
    let mut agent = test_agent();
    let prelude = agent.prepare_turn_prelude("Implement a non-trivial feature with validation");
    assert!(
        prelude
            .iter()
            .any(|message| message.text.contains("workflow manager first"))
    );
    assert!(
        prelude
            .iter()
            .any(|message| message.text.contains("specialist lane is needed"))
    );
    assert!(prelude.iter().any(|message| {
        message
            .text
            .contains("explorer for broad or unknown code search")
    }));
}

#[tokio::test]
async fn finalization_does_not_auto_reconcile_unreconciled_subagent_jobs() {
    let mut agent = test_agent();
    agent.prepare_turn_prelude("Follow up on child work");
    agent
        .add_evidence(
            EvidenceDraft {
                id: Some("ev-1".into()),
                evidence_kind: crate::evidence::EvidenceKind::Decision,
                title: "subagent result".into(),
                summary: "child completed".into(),
                detail: Some(
                    serde_json::to_string(&crate::subagent::StructuredSubagentResult {
                        status: "completed".into(),
                        summary: "child completed".into(),
                        malformed: false,
                        findings: vec![],
                        files_read: vec![],
                        files_changed: vec![],
                        commands_run: vec![],
                        validation: vec![],
                        blockers: vec![],
                        next_steps: vec![],
                        run_id: "run-1".into(),
                        child_session_id: "child-1".into(),
                        raw_excerpt: None,
                    })
                    .expect("serialize structured result"),
                ),
                source: EvidenceSource::Subagent {
                    run_id: "run-1".into(),
                    child_session_id: "child-1".into(),
                    source_session_id: "child-1".into(),
                    parent_tool: "agent__explore".into(),
                    parent_turn_id: Some("turn-1".into()),
                    parent_session_id: None,
                },
                tags: vec![
                    "explorer".into(),
                    "subagent_result".into(),
                    "unreconciled".into(),
                ],
            }
            .into_record("ev-1".into(), 1, 0)
            .expect("build evidence"),
        )
        .expect("add evidence");

    let mut events = Vec::new();
    let continued = agent
        .continue_or_finalize_no_tool_reply(
            &mut |event| {
                events.push(event);
                std::future::ready(Ok(()))
            },
            0,
            &mut 0,
        )
        .await
        .expect("finalization succeeds");

    assert!(!continued);
    assert!(!events.iter().any(|event| matches!(
        event,
        AgentEvent::EvidenceRecorded(record)
            if record.tags.iter().any(|tag| tag == "subagent_reconciliation")
    )));
    assert_eq!(agent.pending_subagent_jobs().len(), 1);
}

#[test]
fn child_validation_classification_ignores_not_run_and_counts_object_failures() {
    let (ran, failed) = classify_child_validation_entries(&[
        "cargo test not_run".into(),
        "cargo fmt passed".into(),
        "cargo test failed".into(),
    ]);

    assert_eq!(ran, 2);
    assert_eq!(failed, 1);
}

#[test]
fn turn_lifecycle_events_capture_expected_snapshot_fields() {
    let client = Client::with_config(
        OpenAIConfig::new()
            .with_api_base("https://api.openai.com/v1")
            .with_api_key("test"),
    );
    let mut agent = Agent::new(client, "m1", 1, 1);

    agent.prepare_turn_prelude("Implement fix in src/agent.rs and run cargo test transcript");
    agent.turn.counters.write_effects = 2;
    agent.turn.counters.validation_effects = 1;
    agent.turn.counters.failed_validation_effects = 0;

    let started = agent.turn_started_event();
    assert_eq!(started.turn_id, 1);
    assert_eq!(started.intent, "engineering");
    assert_eq!(started.directive, "none");
    assert_eq!(started.validation_reminder, "targeted");

    let finalized = agent.turn_finalized_event("completed", 3, 1, true);
    assert_eq!(finalized.turn_id, 1);
    assert_eq!(finalized.outcome, "completed");
    assert_eq!(finalized.tool_call_count, 3);
    assert_eq!(finalized.continuation_count, 1);
    assert_eq!(finalized.write_effects, 2);
    assert_eq!(finalized.validation_effects, 1);
    assert!(finalized.validation_advisory_emitted);
}

#[test]
fn tool_execution_summary_event_omits_full_output_and_captures_audit_fields() {
    let mut agent = test_agent();
    agent.prepare_turn_prelude("Implement fix");
    let record = ToolExecutionRecord::new(
        &test_tool_call("shell__exec", r#"{"command":"cargo test transcript"}"#),
        Some(json!({"command": "cargo test transcript", "path": "src/agent.rs"})),
        crate::permission::ToolPermissionClass::Command,
        ExecutionDirective::None,
        ToolExecutionStatus::Executed,
        None,
        ToolResult::ok(
            "shell__exec",
            json!({"command": "cargo test transcript", "status": 0, "stdout": "lots"}),
        ),
    );

    let summary = agent.tool_execution_summary_event(&record);
    assert_eq!(summary.turn_id, 1);
    assert_eq!(summary.call_id, "call-shell__exec");
    assert_eq!(summary.name, "shell__exec");
    assert_eq!(summary.status, "executed");
    assert_eq!(summary.effect_kind, "validation");
    assert_eq!(summary.primary_path.as_deref(), Some("src/agent.rs"));
    assert_eq!(summary.command.as_deref(), Some("cargo test transcript"));
    assert_eq!(summary.rejection, None);
}
