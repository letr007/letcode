use crate::subagent::StructuredSubagentResult;
use crate::transcript::{ChildSessionSummary, JobBoardEntry, TranscriptEvent, TranscriptRecord};
use std::collections::BTreeMap;
use std::path::Path;

pub(crate) fn project_child_session_summaries(
    child_dir: &Path,
    parent_records: &[TranscriptRecord],
) -> Vec<ChildSessionSummary> {
    let mut children = BTreeMap::new();

    for record in parent_records {
        if let TranscriptEvent::SubagentResult {
            parent_session_id,
            parent_run_id,
            child_session_id,
            agent_name,
            status,
            summary,
            ..
        } = &record.event
            && child_dir.join(format!("{child_session_id}.jsonl")).exists()
        {
            children.insert(
                child_session_id.clone(),
                ChildSessionSummary {
                    parent_session_id: parent_session_id.clone(),
                    parent_run_id: parent_run_id.clone(),
                    child_session_id: child_session_id.clone(),
                    agent_name: agent_name.clone(),
                    status: status.clone(),
                    summary: summary.clone(),
                    timestamp_ms: record.timestamp_ms,
                },
            );
        }
    }

    let mut children = children.into_values().collect::<Vec<_>>();
    children.sort_by(|left, right| {
        left.timestamp_ms
            .cmp(&right.timestamp_ms)
            .then_with(|| left.child_session_id.cmp(&right.child_session_id))
    });
    children
}

pub(crate) fn project_job_board(
    child_dir: &Path,
    parent_records: &[TranscriptRecord],
) -> anyhow::Result<Vec<JobBoardEntry>> {
    let mut jobs = BTreeMap::<String, JobBoardAccumulator>::new();

    for record in parent_records {
        match &record.event {
            TranscriptEvent::SubagentResult {
                run_id,
                child_session_id,
                agent_name,
                status,
                summary,
                ..
            } => {
                let entry = jobs.entry(run_id.clone()).or_default();
                entry.run_id = run_id.clone();
                entry.child_session_id = child_session_id.clone();
                entry.agent_name = agent_name.clone();
                entry.status = status.clone();
                entry.summary = summary.clone();
                entry.terminal = true;
                entry.active = false;
            }
            TranscriptEvent::Evidence {
                source:
                    crate::evidence::EvidenceSource::Subagent {
                        run_id,
                        child_session_id,
                        parent_tool,
                        ..
                    },
                summary,
                detail,
                tags,
                ..
            } => {
                let entry = jobs.entry(run_id.clone()).or_default();
                entry.run_id = run_id.clone();
                if entry.child_session_id.is_empty() {
                    entry.child_session_id = child_session_id.clone();
                }
                if entry.agent_name.is_empty() {
                    entry.agent_name = parent_tool.trim_start_matches("agent__").to_string();
                }
                if tags.iter().any(|tag| tag == "subagent_result") {
                    entry.summary = summary.clone();
                    if let Some(detail) = detail
                        && let Ok(structured) =
                            serde_json::from_str::<StructuredSubagentResult>(detail)
                    {
                        entry.malformed = structured.malformed;
                        entry.structured_status = Some(structured.status.clone());
                        if entry.status.is_empty() {
                            entry.status = structured.status;
                        }
                    }
                }
                if tags
                    .iter()
                    .any(|tag| tag == "subagent_reconciliation" || tag == "reconciled")
                {
                    entry.reconciled = true;
                }
            }
            _ => {}
        }
    }

    if child_dir.exists() {
        for entry in std::fs::read_dir(child_dir)? {
            let path = entry?.path();
            if path.extension().and_then(|ext| ext.to_str()) != Some("jsonl") {
                continue;
            }
            let child_session_id = match path.file_stem().and_then(|stem| stem.to_str()) {
                Some(value) => value.to_string(),
                None => continue,
            };
            let child_records = crate::transcript::read_records_allow_partial_tail(&path)?;
            let latest = child_records
                .iter()
                .rev()
                .find_map(|record| match &record.event {
                    TranscriptEvent::SubagentLifecycle {
                        run_id,
                        agent_name,
                        status,
                        detail,
                        ..
                    } => Some((
                        run_id.clone(),
                        agent_name.clone(),
                        status.clone(),
                        detail.clone(),
                    )),
                    _ => None,
                });
            let Some((run_id, agent_name, status, detail)) = latest else {
                continue;
            };
            if status != "running" {
                continue;
            }
            let job = jobs.entry(run_id.clone()).or_default();
            if job.terminal {
                continue;
            }
            job.run_id = run_id;
            job.child_session_id = child_session_id;
            job.agent_name = agent_name;
            job.status = status;
            job.summary = detail.unwrap_or_else(|| "subagent running".into());
            job.active = true;
        }
    }

    let mut entries = jobs
        .into_values()
        .filter(|entry| !entry.run_id.is_empty())
        .map(|entry| {
            let reconciled = entry.terminal && entry.reconciled;
            let unreconciled = entry.terminal && !entry.reconciled;
            let reusable_eligible = reconciled
                && entry.status == "completed"
                && entry.structured_status.as_deref() == Some("completed")
                && !entry.malformed;
            JobBoardEntry {
                active: entry.active,
                unreconciled,
                reconciled,
                reusable_eligible,
                run_id: entry.run_id,
                child_session_id: entry.child_session_id,
                agent_name: entry.agent_name,
                status: entry.status,
                summary: entry.summary,
            }
        })
        .collect::<Vec<_>>();
    entries.sort_by(|a, b| a.run_id.cmp(&b.run_id));
    Ok(entries)
}

#[derive(Debug, Clone, Default)]
struct JobBoardAccumulator {
    run_id: String,
    child_session_id: String,
    agent_name: String,
    status: String,
    summary: String,
    active: bool,
    terminal: bool,
    reconciled: bool,
    malformed: bool,
    structured_status: Option<String>,
}
