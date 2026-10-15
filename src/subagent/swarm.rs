use std::collections::{BTreeMap, HashSet};
use std::path::Path;

use anyhow::{Result, anyhow, ensure};
use serde::{Deserialize, Serialize};

use crate::transcript::{TranscriptEvent, TranscriptRecord};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SwarmMember {
    pub root_session_id: String,
    pub parent_session_id: String,
    pub child_session_id: String,
    pub depth: u32,
    pub agent_name: String,
    pub pool_ordinal: u32,
}

#[derive(Debug, Clone, Serialize)]
pub(crate) struct SwarmJob {
    pub session_id: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub child_session_id: Option<String>,
    pub root_session_id: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parent_session_id: Option<String>,
    pub depth: u32,
    pub role: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub pool_ordinal: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub run_id: Option<String>,
    pub status: String,
    pub active: bool,
}

impl SwarmJob {
    pub(crate) fn member(
        member: &SwarmMember,
        run_id: String,
        status: String,
        active: bool,
    ) -> Self {
        Self {
            session_id: member.child_session_id.clone(),
            child_session_id: Some(member.child_session_id.clone()),
            root_session_id: member.root_session_id.clone(),
            parent_session_id: Some(member.parent_session_id.clone()),
            depth: member.depth,
            role: member.agent_name.clone(),
            pool_ordinal: Some(member.pool_ordinal),
            run_id: Some(run_id),
            status,
            active,
        }
    }
}

pub(crate) struct SwarmSnapshot {
    pub policy: crate::config::SubagentsConfig,
    pub root: SwarmJob,
    pub members: BTreeMap<String, SwarmMember>,
    pub runs: Vec<SwarmJob>,
}

impl SwarmSnapshot {
    pub(crate) fn latest_jobs(&self) -> Vec<SwarmJob> {
        let mut jobs = BTreeMap::new();
        for run in &self.runs {
            jobs.insert(run.session_id.clone(), run.clone());
        }
        let mut jobs: Vec<_> = jobs.into_values().collect();
        jobs.sort_by_key(|job| job.pool_ordinal);
        jobs.insert(0, self.root.clone());
        jobs
    }
}

pub(crate) fn caller_identity(records: &[TranscriptRecord]) -> Result<Option<SwarmMember>> {
    let mut identity = None;
    for record in records {
        if let TranscriptEvent::SubagentActivated { member, .. } = &record.event {
            ensure!(
                member.child_session_id == record.session_id,
                "swarm activation session mismatch"
            );
            ensure!(
                identity.as_ref().is_none_or(|old| old == member),
                "conflicting swarm member identity"
            );
            identity = Some(member.clone());
        }
    }
    Ok(identity)
}

pub(super) fn read_session(
    sessions_dir: &Path,
    session_id: &str,
    child: bool,
) -> Result<Vec<TranscriptRecord>> {
    let live = if child {
        crate::transcript::child_session_path(sessions_dir, session_id)
    } else {
        sessions_dir.join(format!("{session_id}.jsonl"))
    };
    if live.exists() {
        return crate::transcript::read_records_allow_partial_tail(live);
    }
    let archived = if child {
        crate::transcript::archive_index::child_transcript_path(sessions_dir, session_id)
    } else {
        crate::transcript::archive_index::transcript_path(sessions_dir, session_id)
    };
    let file = std::fs::File::open(&archived)?;
    let content = zstd::stream::decode_all(file)?;
    crate::transcript::parse_records_content(&archived, std::str::from_utf8(&content)?, true)
}

pub(crate) fn project(
    sessions_dir: &Path,
    caller_records: &[TranscriptRecord],
) -> Result<SwarmSnapshot> {
    let caller = caller_records
        .first()
        .ok_or_else(|| anyhow!("missing swarm session records"))?
        .session_id
        .clone();
    let identity = caller_identity(caller_records)?;
    let root_id = identity
        .as_ref()
        .map_or(caller.as_str(), |member| member.root_session_id.as_str());
    let root_records = if root_id == caller {
        caller_records.to_vec()
    } else {
        read_session(sessions_dir, root_id, false)?
    };
    ensure!(
        caller_identity(&root_records)?.is_none(),
        "swarm root is a member"
    );
    ensure!(
        crate::transcript::restore_subagents_policy(&root_records)?.mode
            == crate::config::SubagentMode::Swarm,
        "swarm root policy mismatch"
    );
    let mut root_active = false;
    for record in &root_records {
        match &record.event {
            TranscriptEvent::TurnStarted(_) => root_active = true,
            TranscriptEvent::TurnFinalized(_) | TranscriptEvent::TurnInterrupted { .. } => {
                root_active = false
            }
            _ => {}
        }
    }
    let mut snapshot = SwarmSnapshot {
        policy: crate::transcript::restore_subagents_policy(&root_records)?,
        root: SwarmJob {
            session_id: root_id.to_string(),
            child_session_id: None,
            root_session_id: root_id.to_string(),
            parent_session_id: None,
            depth: 0,
            role: "root".into(),
            pool_ordinal: None,
            run_id: None,
            status: if root_active { "active" } else { "idle" }.into(),
            active: root_active,
        },
        members: BTreeMap::new(),
        runs: Vec::new(),
    };
    let mut pending = vec![(root_id.to_string(), 0u32, root_records)];
    let mut ordinals = HashSet::new();
    let mut run_ids = HashSet::new();
    while let Some((parent_id, depth, records)) = pending.pop() {
        ensure!(
            records.iter().all(|record| record.session_id == parent_id),
            "swarm transcript session mismatch"
        );
        let mut children = BTreeMap::<String, Vec<_>>::new();
        for record in &records {
            if let TranscriptEvent::SubagentStarted {
                child_session_id,
                parent_session_id,
                ..
            } = &record.event
                && parent_session_id == &parent_id
            {
                children
                    .entry(child_session_id.clone())
                    .or_default()
                    .push(record);
            }
        }
        for (child_id, starts) in children {
            ensure!(
                child_id != root_id && !snapshot.members.contains_key(&child_id),
                "swarm member has conflicting parent or cycle"
            );
            let child_records = read_session(sessions_dir, &child_id, true)?;
            let Some(member) = caller_identity(&child_records)? else {
                continue;
            };
            ensure!(
                member.root_session_id == root_id
                    && member.parent_session_id == parent_id
                    && member.child_session_id == child_id
                    && member.depth
                        == depth
                            .checked_add(1)
                            .ok_or_else(|| anyhow!("swarm depth overflow"))?
                    && member.pool_ordinal > 0,
                "swarm member ancestry mismatch"
            );
            ensure!(
                ordinals.insert(member.pool_ordinal),
                "duplicate swarm pool ordinal"
            );
            let mut activated = HashSet::new();
            for record in &child_records {
                if let TranscriptEvent::SubagentActivated { run_id, .. } = &record.event {
                    ensure!(
                        activated.insert(run_id.clone()),
                        "duplicate swarm activation run"
                    );
                }
            }
            for start in starts {
                let TranscriptEvent::SubagentStarted {
                    run_id,
                    agent_name,
                    pool_ordinal,
                    ..
                } = &start.event
                else {
                    unreachable!()
                };
                ensure!(
                    agent_name == &member.agent_name && *pool_ordinal == member.pool_ordinal,
                    "swarm member role or ordinal mismatch"
                );
                if !activated.remove(run_id) {
                    continue;
                }
                ensure!(run_ids.insert(run_id.clone()), "duplicate swarm run id");
                let child_started = child_records.iter().any(|record| match &record.event {
                    TranscriptEvent::SubagentStarted {
                        run_id: child_run,
                        parent_session_id,
                        child_session_id,
                        agent_name,
                        pool_ordinal,
                        ..
                    } => {
                        child_run == run_id
                            && parent_session_id == &parent_id
                            && child_session_id == &child_id
                            && agent_name == &member.agent_name
                            && *pool_ordinal == member.pool_ordinal
                    }
                    _ => false,
                });
                ensure!(child_started, "swarm activation has no child start");
                let mut status = "running".to_string();
                let mut active = true;
                for record in &child_records {
                    if let TranscriptEvent::SubagentLifecycle {
                        run_id: recorded,
                        parent_session_id,
                        agent_name,
                        status: next,
                        ..
                    } = &record.event
                        && recorded == run_id
                    {
                        ensure!(
                            parent_session_id == &parent_id && agent_name == &member.agent_name,
                            "swarm lifecycle identity mismatch"
                        );
                        status = next.clone();
                        active = status == "running";
                    }
                }
                for record in &records {
                    if let TranscriptEvent::SubagentResult {
                        run_id: recorded,
                        parent_session_id,
                        child_session_id,
                        agent_name,
                        status: next,
                        ..
                    } = &record.event
                        && recorded == run_id
                    {
                        ensure!(
                            parent_session_id == &parent_id
                                && child_session_id == &child_id
                                && agent_name == &member.agent_name,
                            "swarm result identity mismatch"
                        );
                        status = next.clone();
                        active = false;
                    }
                }
                snapshot
                    .runs
                    .push(SwarmJob::member(&member, run_id.clone(), status, active));
            }
            ensure!(activated.is_empty(), "swarm activation has no parent start");
            snapshot.members.insert(child_id.clone(), member.clone());
            pending.push((child_id, member.depth, child_records));
        }
    }
    if let Some(identity) = identity {
        ensure!(
            snapshot.members.get(&caller) == Some(&identity),
            "swarm caller is not a root member"
        );
    } else {
        ensure!(
            !caller_records
                .iter()
                .any(|record| matches!(record.event, TranscriptEvent::SubagentLifecycle { .. })),
            "swarm child is not activated"
        );
    }
    Ok(snapshot)
}
