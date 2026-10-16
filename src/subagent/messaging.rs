use super::*;
use std::collections::VecDeque;

pub(crate) type MessageDelegateFactory = Arc<
    dyn Fn(
            SubagentPool,
            Arc<Mutex<TranscriptRecorder>>,
        ) -> Result<Arc<dyn crate::agent::SubagentDelegate>>
        + Send
        + Sync,
>;

pub(crate) struct MessageHost {
    pub seed: Mutex<Agent>,
    pub sessions_dir: PathBuf,
    pub root_transcript: Arc<Mutex<TranscriptRecorder>>,
    pub delegate: MessageDelegateFactory,
    pub event_sender: Option<SubagentEventSender>,
    pub wake_root: Arc<dyn Fn(String) -> Result<()> + Send + Sync>,
}

#[derive(Default)]
pub(super) struct Mailbox {
    pub pending: VecDeque<String>,
    pub waking: bool,
    pub wake_cancelled: bool,
    pub stopped: bool,
    pub execution_sender: Option<Option<SubagentEventSender>>,
    pub transcript: std::sync::Weak<Mutex<TranscriptRecorder>>,
}

impl SubagentPool {
    pub(crate) fn swarm_generation(&self, session_id: &str) -> Option<u64> {
        self.state.lock().ok().and_then(|state| {
            state
                .swarms
                .iter()
                .find(|(root, authority)| {
                    root.as_str() == session_id || authority.members.contains_key(session_id)
                })
                .map(|(_, authority)| authority.generation)
        })
    }

    pub(crate) fn resume_swarm(
        &self,
        sessions_dir: &Path,
        records: &[crate::transcript::TranscriptRecord],
    ) -> Result<()> {
        let snapshot = self.swarm_snapshot(sessions_dir, records)?;
        let root = &snapshot.root.session_id;
        let mut state = self
            .state
            .lock()
            .map_err(|_| anyhow!("subagent pool lock poisoned"))?;
        anyhow::ensure!(
            !state
                .active_by_run
                .values()
                .any(|slot| slot.swarm_root.as_ref() == Some(root)),
            "swarm runs have not settled before resume"
        );
        let authority = state.swarms.get_mut(root).expect("restored authority");
        anyhow::ensure!(
            !authority.mailboxes.values().any(|mailbox| mailbox.waking),
            "swarm message owners have not settled before resume"
        );
        authority.generation = authority
            .generation
            .checked_add(1)
            .ok_or_else(|| anyhow!("swarm generation exhausted"))?;
        authority.mailboxes.clear();
        authority.message_host = None;
        Ok(())
    }

    pub(crate) fn install_message_host(
        &self,
        records: &[crate::transcript::TranscriptRecord],
        host: MessageHost,
    ) -> Result<()> {
        let snapshot = self.swarm_snapshot(&host.sessions_dir, records)?;
        anyhow::ensure!(
            super::super::swarm::caller_identity(records)?.is_none(),
            "message host must belong to the root"
        );
        let mut state = self
            .state
            .lock()
            .map_err(|_| anyhow!("subagent pool lock poisoned"))?;
        let authority = state
            .swarms
            .get_mut(&snapshot.root.session_id)
            .expect("restored authority");
        authority.message_host = Some(Arc::new(host));
        Ok(())
    }

    pub(crate) fn enqueue_member_message(
        &self,
        sessions_dir: &Path,
        records: &[crate::transcript::TranscriptRecord],
        target: &str,
        message: &str,
    ) -> Result<usize> {
        let snapshot = self.swarm_snapshot(sessions_dir, records)?;
        let root = snapshot.root.session_id.clone();
        let caller = records
            .first()
            .ok_or_else(|| anyhow!("missing sender session"))?
            .session_id
            .as_str();
        let jobs = snapshot.latest_jobs();
        anyhow::ensure!(
            !jobs
                .iter()
                .any(|job| job.session_id == caller && job.status == "cancelled"),
            "sender is cancelled"
        );
        let target_job = jobs
            .into_iter()
            .find(|job| job.session_id == target)
            .ok_or_else(|| anyhow!("unknown or cross-root swarm session_id: {target}"))?;
        anyhow::ensure!(!message.trim().is_empty(), "message must not be empty");
        anyhow::ensure!(
            message.len() <= snapshot.policy.max_message_bytes,
            "message exceeds max_message_bytes"
        );
        let (host, wake, pending) = {
            let mut state = self
                .state
                .lock()
                .map_err(|_| anyhow!("subagent pool lock poisoned"))?;
            anyhow::ensure!(
                !state.swarm_scope_is_stopping(&root, caller),
                "sender is stopping"
            );
            anyhow::ensure!(
                !state.swarm_scope_is_stopping(&root, target),
                "target is stopping"
            );
            let active = state.active_by_run.values().find(|slot| {
                slot.child.child_session_id == target
                    || slot.takeover_child_session_id.as_deref() == Some(target)
            });
            anyhow::ensure!(
                active.is_none_or(|slot| !slot.cancellation_requested),
                "target is stopping"
            );
            let active = active.is_some();
            let occupied = state.swarm_occupancy(&root);
            let authority = state.swarms.get_mut(&root).expect("restored authority");
            anyhow::ensure!(
                !authority
                    .mailboxes
                    .get(caller)
                    .is_some_and(|mailbox| mailbox.stopped),
                "sender is cancelled or stopping"
            );
            let host = authority
                .message_host
                .clone()
                .ok_or_else(|| anyhow!("swarm message execution is unavailable in this runtime"))?;
            let mailbox = authority.mailboxes.entry(target.to_string()).or_default();
            anyhow::ensure!(
                !mailbox.stopped && target_job.status != "cancelled",
                "target is cancelled or stopping"
            );
            anyhow::ensure!(
                mailbox.pending.len() < snapshot.policy.max_pending_messages,
                "target message queue is full"
            );
            let wake = !active && !mailbox.waking;
            if wake && target != root && occupied >= snapshot.policy.max_concurrent {
                return Err(SubagentCapacityExceeded {
                    limit: snapshot.policy.max_concurrent,
                }
                .into());
            }
            mailbox.pending.push_back(format!(
                "[Swarm message from session_id={caller} to session_id={target}]\n\n{message}"
            ));
            if wake {
                mailbox.waking = true;
                mailbox.wake_cancelled = false;
            }
            (host, wake, mailbox.pending.len())
        };
        self.changed.notify_waiters();
        if wake {
            self.wake_message_target(root, target.to_string(), host);
        }
        Ok(pending)
    }

    fn wake_message_target(&self, root: String, target: String, host: Arc<MessageHost>) {
        if target == root {
            if let Err(error) = (host.wake_root)(root.clone()) {
                self.fail_member_messages(
                    &root,
                    &target,
                    &format!("root message dispatch failed: {error}"),
                    false,
                );
            }
            return;
        }
        let runtime = self.clone();
        tokio::spawn(async move {
            let started = runtime.start_message_run(&root, &target, &host);
            match started {
                Ok(started) => {
                    if let Err(error) = runtime.complete_started_run(started).await {
                        runtime.fail_member_messages(
                            &root,
                            &target,
                            &format!("member message execution failed: {error}"),
                            false,
                        );
                    }
                }
                Err(error) => {
                    runtime.fail_member_messages(
                        &root,
                        &target,
                        &format!("member message startup failed: {error:#}"),
                        false,
                    );
                }
            }
        });
    }

    fn start_message_run(
        &self,
        root: &str,
        target: &str,
        host: &MessageHost,
    ) -> Result<StartedSubagentRun> {
        let records = super::super::swarm::read_session(&host.sessions_dir, target, true)?;
        let member = super::super::swarm::caller_identity(&records)?
            .ok_or_else(|| anyhow!("message target has no swarm identity"))?;
        anyhow::ensure!(
            member.root_session_id == root,
            "message target root mismatch"
        );
        let parent_records = super::super::swarm::read_session(
            &host.sessions_dir,
            &member.parent_session_id,
            member.parent_session_id != root,
        )?;
        let parent_transcript = if member.parent_session_id == root {
            host.root_transcript.clone()
        } else {
            self.member_transcript(root, &member.parent_session_id, &host.sessions_dir)?
        };
        let seed = host
            .seed
            .lock()
            .map_err(|_| anyhow!("message launch configuration poisoned"))?;
        let parent_role = super::super::swarm::caller_identity(&parent_records)?;
        let mut parent_agent = if let Some(parent_member) = &parent_role {
            let template = AgentTemplate::from_name(&parent_member.agent_name)
                .ok_or_else(|| anyhow!("unknown original parent role"))?
                .for_mode(crate::config::SubagentMode::Swarm);
            AgentFactory::create_child_with_route_and_max_tool_calls(
                &seed, &template, None, false, None,
            )?
        } else {
            AgentFactory::launch_seed(&seed)
        };
        drop(seed);
        let policy = self
            .state
            .lock()
            .map_err(|_| anyhow!("subagent pool lock poisoned"))?
            .swarms[root]
            .policy
            .clone()
            .expect("frozen policy");
        parent_agent.set_subagents_policy(policy);
        parent_agent.set_subagent_path_scope(
            self.restored_member_scope(&host.sessions_dir, &member.parent_session_id, root)?
                .map(Arc::new),
        );
        parent_agent
            .set_subagent_delegate((host.delegate)(self.clone(), parent_transcript.clone())?);
        let original = records
            .iter()
            .find_map(|record| match &record.event {
                TranscriptEvent::SubagentDelegation { input, .. } => Some(input.clone()),
                _ => None,
            })
            .ok_or_else(|| anyhow!("missing original delegation"))?;
        let mut input = original;
        input.target_child_session_id = Some(target.to_string());
        input.model = None;
        input.timeout_secs = None;
        input.background = false;
        let template = AgentTemplate::from_name(&member.agent_name)
            .ok_or_else(|| anyhow!("unknown member role"))?;
        let governance = SubagentRunGovernance::from_template_and_input(&template, input, None);
        let event_sender = self
            .state
            .lock()
            .map_err(|_| anyhow!("subagent pool lock poisoned"))?
            .swarms[root]
            .mailboxes
            .get(target)
            .and_then(|mailbox| mailbox.execution_sender.clone())
            .unwrap_or_else(|| host.event_sender.clone());
        let run = self.start_run(
            &parent_agent,
            template,
            String::new(),
            governance,
            host.sessions_dir.clone(),
            member.parent_session_id,
            generate_run_id(),
            Some(parent_transcript),
            event_sender,
            Some(target.to_string()),
            true,
        )?;
        let receipt = self
            .state
            .lock()
            .map_err(|_| anyhow!("subagent pool lock poisoned"))?
            .active_by_run[&run.run_id]
            .child
            .clone();
        Ok(StartedSubagentRun {
            run_id: run.run_id.clone(),
            receipt,
            run,
            task: String::new(),
        })
    }

    pub(super) fn member_transcript(
        &self,
        root: &str,
        target: &str,
        sessions_dir: &Path,
    ) -> Result<Arc<Mutex<TranscriptRecorder>>> {
        let mut state = self
            .state
            .lock()
            .map_err(|_| anyhow!("subagent pool lock poisoned"))?;
        let mailbox = state
            .swarms
            .get_mut(root)
            .ok_or_else(|| anyhow!("unknown swarm root"))?
            .mailboxes
            .entry(target.to_string())
            .or_default();
        if let Some(transcript) = mailbox.transcript.upgrade() {
            return Ok(transcript);
        }
        let transcript = Arc::new(Mutex::new(TranscriptRecorder::open(
            child_sessions_dir(sessions_dir),
            target,
        )?));
        mailbox.transcript = Arc::downgrade(&transcript);
        Ok(transcript)
    }

    fn restored_member_scope(
        &self,
        sessions_dir: &Path,
        target: &str,
        root: &str,
    ) -> Result<Option<SubagentPathScope>> {
        let mut chain = Vec::new();
        let mut current = target.to_string();
        while current != root {
            let records = super::super::swarm::read_session(sessions_dir, &current, true)?;
            let member = super::super::swarm::caller_identity(&records)?
                .ok_or_else(|| anyhow!("missing scope ancestry"))?;
            let input = records
                .iter()
                .find_map(|record| match &record.event {
                    TranscriptEvent::SubagentDelegation { input, .. } => Some(input),
                    _ => None,
                })
                .ok_or_else(|| anyhow!("missing original scope"))?;
            chain.push(SubagentPathScope::from_input(input)?);
            current = member.parent_session_id;
        }
        let mut scope = None;
        for requested in chain.into_iter().rev() {
            scope = match (requested, scope) {
                (Some(requested), Some(parent)) => Some(requested.narrowed_from(&parent)?),
                (Some(requested), None) => Some(requested),
                (None, parent) => parent,
            };
        }
        Ok(scope)
    }

    pub(crate) fn member_message_pending(&self, root: &str, target: &str) -> Result<bool> {
        let state = self
            .state
            .lock()
            .map_err(|_| anyhow!("subagent pool lock poisoned"))?;
        Ok(state
            .swarms
            .get(root)
            .and_then(|authority| authority.mailboxes.get(target))
            .is_some_and(|mailbox| !mailbox.stopped && !mailbox.pending.is_empty()))
    }

    pub(crate) fn take_member_message(&self, root: &str, target: &str) -> Result<Option<String>> {
        let mut state = self
            .state
            .lock()
            .map_err(|_| anyhow!("subagent pool lock poisoned"))?;
        let Some(authority) = state.swarms.get_mut(root) else {
            return Ok(None);
        };
        let Some(mailbox) = authority
            .mailboxes
            .get_mut(target)
            .filter(|mailbox| !mailbox.stopped)
        else {
            return Ok(None);
        };
        mailbox.waking = false;
        Ok(mailbox.pending.pop_front())
    }

    pub(crate) fn member_scope_stopped(&self, session_id: &str) -> bool {
        self.state.lock().is_ok_and(|state| {
            state.swarms.values().any(|authority| {
                authority
                    .mailboxes
                    .get(session_id)
                    .is_some_and(|mailbox| mailbox.stopped)
            })
        })
    }

    pub(crate) fn close_swarm_sessions(&self) {
        self.cancel_active();
        if let Ok(mut state) = self.state.lock() {
            for (root, authority) in &mut state.swarms {
                authority.mailboxes.entry(root.clone()).or_default().stopped = true;
                authority.message_host = None;
            }
        }
    }

    pub(super) fn stop_swarm_members(&self, run_id: Option<&str>) {
        let targets = {
            let Ok(mut state) = self.state.lock() else {
                return;
            };
            let scope = run_id
                .and_then(|run| state.active_by_run.get(run))
                .and_then(|slot| {
                    slot.swarm_root
                        .clone()
                        .map(|root| (root, slot.child.child_session_id.clone()))
                });
            let mut targets = Vec::new();
            for (root, authority) in &mut state.swarms {
                if run_id.is_some()
                    && scope
                        .as_ref()
                        .is_none_or(|(scope_root, _)| scope_root != root)
                {
                    continue;
                }
                let mut pending = scope
                    .as_ref()
                    .map_or_else(|| vec![root.clone()], |(_, session)| vec![session.clone()]);
                while let Some(target) = pending.pop() {
                    let mailbox = authority.mailboxes.entry(target.clone()).or_default();
                    if &target == root {
                        mailbox.waking = false;
                    } else {
                        mailbox.stopped = true;
                    }
                    mailbox.execution_sender = None;
                    targets.push((root.clone(), target.clone()));
                    pending.extend(
                        authority
                            .members
                            .values()
                            .filter(|member| member.parent_session_id == target)
                            .map(|member| member.child_session_id.clone()),
                    );
                }
            }
            for slot in state.active_by_run.values_mut() {
                if slot.swarm_root.as_ref().is_some_and(|root| {
                    targets.iter().any(|(target_root, target)| {
                        target_root == root
                            && (target == &slot.child.child_session_id
                                || slot.takeover_child_session_id.as_ref() == Some(target))
                    })
                }) {
                    slot.cancellation_requested = true;
                    if let Some(cancel) = slot.cancel.take() {
                        let _ = cancel.send(());
                    }
                }
            }
            if let Some(run_id) = run_id {
                cancel_descendants(&mut state, run_id);
            }
            targets
        };
        for (root, target) in &targets {
            self.fail_member_messages(root, target, "scope stopped", target != root);
        }
    }

    pub(crate) fn fail_member_messages(
        &self,
        root: &str,
        target: &str,
        reason: &str,
        stopped: bool,
    ) -> usize {
        let (host, count, release_owner) = {
            let Ok(mut state) = self.state.lock() else {
                return 0;
            };
            let active = state.active_by_run.values().any(|slot| {
                slot.child.child_session_id == target
                    || slot.takeover_child_session_id.as_deref() == Some(target)
            });
            let Some(authority) = state.swarms.get_mut(root) else {
                return 0;
            };
            let mailbox = authority.mailboxes.entry(target.to_string()).or_default();
            let count = mailbox.pending.len();
            mailbox.pending.clear();
            mailbox.stopped |= stopped;
            (authority.message_host.clone(), count, !active && !stopped)
        };
        self.changed.notify_waiters();
        if count == 0 {
            if release_owner {
                self.release_message_owner(root, target);
            }
            return 0;
        }
        if let Some(host) = host {
            let message = format!("{count} queued message(s) for {target} failed: {reason}");
            let result = (|| -> Result<()> {
                let recorder = if target == root {
                    host.root_transcript.clone()
                } else {
                    self.member_transcript(root, target, &host.sessions_dir)?
                };
                recorder
                    .lock()
                    .map_err(|_| anyhow!("transcript recorder poisoned"))?
                    .record_error(message.clone())
            })();
            emit_error(&host.event_sender, message.clone());
            if let Err(error) = result {
                let detail = format!("{message}; failed to record target error: {error:#}");
                emit_error(&host.event_sender, detail.clone());
                if target != root {
                    let root_result = host
                        .root_transcript
                        .lock()
                        .map_err(|_| anyhow!("root transcript recorder poisoned"))
                        .and_then(|mut recorder| recorder.record_error(detail));
                    if let Err(error) = root_result {
                        tracing::error!(%error, session_id = target, "failed to record queued message failure in the root");
                    }
                }
            }
        }
        if release_owner {
            self.release_message_owner(root, target);
        }
        count
    }

    pub(super) fn settle_message_run(&self, run_id: &str, summary: &SubagentRunSummary) -> usize {
        let root = self.state.lock().ok().and_then(|state| {
            state
                .active_by_run
                .get(run_id)
                .and_then(|slot| slot.swarm_root.clone())
        });
        let Some(root) = root else {
            return 0;
        };
        if summary.status != SubagentStatus::Completed {
            if let Ok(mut state) = self.state.lock()
                && let Some(slot) = state.active_by_run.get_mut(run_id)
            {
                slot.cancellation_requested = true;
                cancel_descendants(&mut state, run_id);
            }
            return self.fail_member_messages(
                &root,
                &summary.child_session_id,
                summary.status.as_str(),
                summary.status == SubagentStatus::Cancelled,
            );
        }
        0
    }

    pub(super) fn release_message_owner(&self, root: &str, target: &str) {
        let host = {
            let Ok(mut state) = self.state.lock() else {
                return;
            };
            let Some(authority) = state.swarms.get_mut(root) else {
                return;
            };
            let Some(mailbox) = authority.mailboxes.get_mut(target) else {
                return;
            };
            mailbox.wake_cancelled = false;
            mailbox.waking = !mailbox.stopped && !mailbox.pending.is_empty();
            if mailbox.waking {
                authority.message_host.clone()
            } else {
                None
            }
        };
        self.changed.notify_waiters();
        if let Some(host) = host {
            self.wake_message_target(root.to_string(), target.to_string(), host);
        }
    }
}
