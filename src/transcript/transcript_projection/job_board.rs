use crate::transcript::{ChildSessionSummary, TranscriptEvent, TranscriptRecord};
use std::collections::BTreeMap;
use std::path::Path;

pub(crate) fn project_child_session_summaries(
    child_dir: &Path,
    parent_records: &[TranscriptRecord],
) -> Vec<ChildSessionSummary> {
    let mut children = BTreeMap::new();

    for record in parent_records {
        match &record.event {
            TranscriptEvent::SubagentStarted {
                parent_session_id,
                parent_run_id,
                child_session_id,
                agent_name,
                summary,
                pool_ordinal,
                ..
            } if parent_session_id == &record.session_id
                && child_dir.join(format!("{child_session_id}.jsonl")).exists() =>
            {
                children
                    .entry(child_session_id.clone())
                    .or_insert_with(|| ChildSessionSummary {
                        parent_session_id: parent_session_id.clone(),
                        parent_run_id: parent_run_id.clone(),
                        child_session_id: child_session_id.clone(),
                        agent_name: agent_name.clone(),
                        status: "running".into(),
                        summary: summary.clone(),
                        timestamp_ms: record.timestamp_ms,
                        pool_ordinal: *pool_ordinal,
                    });
            }
            TranscriptEvent::SubagentResult {
                parent_session_id,
                parent_run_id,
                child_session_id,
                agent_name,
                status,
                summary,
                ..
            } if parent_session_id == &record.session_id
                && child_dir.join(format!("{child_session_id}.jsonl")).exists() =>
            {
                let child = children.entry(child_session_id.clone()).or_insert_with(|| {
                    ChildSessionSummary {
                        parent_session_id: parent_session_id.clone(),
                        parent_run_id: parent_run_id.clone(),
                        child_session_id: child_session_id.clone(),
                        agent_name: agent_name.clone(),
                        status: status.clone(),
                        summary: summary.clone(),
                        timestamp_ms: record.timestamp_ms,
                        pool_ordinal: 0,
                    }
                });
                child.status = status.clone();
                child.summary = summary.clone();
            }
            _ => {}
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

#[cfg(test)]
pub(crate) fn project_job_board(
    child_dir: &Path,
    parent_records: &[TranscriptRecord],
) -> anyhow::Result<Vec<crate::transcript::JobBoardEntry>> {
    use crate::subagent::StructuredSubagentResult;
    use crate::transcript::{JobBoardEntry, read_records_allow_partial_tail};
    let mut jobs = BTreeMap::<String, JobBoardAccumulator>::new();

    for record in parent_records {
        match &record.event {
            TranscriptEvent::SubagentStarted {
                run_id,
                parent_session_id,
                child_session_id,
                agent_name,
                summary,
                pool_ordinal: _,
                ..
            } if parent_session_id == &record.session_id
                && child_dir.join(format!("{child_session_id}.jsonl")).exists() =>
            {
                let entry = jobs.entry(run_id.clone()).or_default();
                entry.run_id = run_id.clone();
                entry.child_session_id = child_session_id.clone();
                entry.agent_name = agent_name.clone();
                entry.status = "running".into();
                entry.summary = summary.clone();
                entry.active = true;
            }
            TranscriptEvent::SubagentResult {
                run_id,
                parent_session_id,
                child_session_id,
                agent_name,
                status,
                summary,
                ..
            } if parent_session_id == &record.session_id => {
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
                    entry.agent_name = parent_tool
                        .strip_prefix("agent__")
                        .or_else(|| parent_tool.strip_prefix("system__"))
                        .unwrap_or(parent_tool)
                        .to_string();
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

    let mut jobs = jobs.into_values().collect::<Vec<_>>();
    for entry in &mut jobs {
        if entry.active {
            hydrate_active_job_from_child_transcript(child_dir, entry)?;
        }
    }

    let mut entries = jobs
        .into_iter()
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

#[cfg(test)]
fn hydrate_active_job_from_child_transcript(
    child_dir: &Path,
    entry: &mut JobBoardAccumulator,
) -> anyhow::Result<()> {
    use crate::transcript::read_records_allow_partial_tail;
    let child_records = read_records_allow_partial_tail(
        child_dir.join(format!("{}.jsonl", entry.child_session_id)),
    )?;

    for record in child_records {
        let TranscriptEvent::SubagentLifecycle {
            run_id,
            status,
            detail,
            ..
        } = record.event
        else {
            continue;
        };
        if run_id != entry.run_id {
            continue;
        }

        entry.status = status.clone();
        if let Some(detail) = detail {
            entry.summary = detail;
        }
        if is_terminal_subagent_status(&status) {
            entry.terminal = true;
            entry.active = false;
        }
    }

    Ok(())
}

#[cfg(test)]
fn is_terminal_subagent_status(status: &str) -> bool {
    matches!(
        status,
        "completed" | "failed" | "budget_exhausted" | "cancelled" | "timed_out"
    )
}

#[cfg(test)]
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
