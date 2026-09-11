use crate::transcript::{ChildSessionSummary, TranscriptEvent, TranscriptRecord};
use anyhow::Result;
use std::collections::BTreeMap;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

pub(crate) fn project_child_session_summaries(
    child_dir: &Path,
    parent_records: &[TranscriptRecord],
) -> Vec<ChildSessionSummary> {
    let mut children = BTreeMap::new();
    let owned_children = parent_records
        .iter()
        .filter_map(|record| match &record.event {
            TranscriptEvent::SubagentStarted {
                parent_session_id,
                child_session_id,
                ..
            } if parent_session_id == &record.session_id
                && child_dir.join(format!("{child_session_id}.jsonl")).exists() =>
            {
                Some(child_session_id.clone())
            }
            _ => None,
        })
        .collect::<std::collections::HashSet<_>>();

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
                let child = children.entry(child_session_id.clone()).or_insert_with(|| {
                    ChildSessionSummary {
                        parent_session_id: parent_session_id.clone(),
                        parent_run_id: parent_run_id.clone(),
                        child_session_id: child_session_id.clone(),
                        agent_name: agent_name.clone(),
                        status: "running".into(),
                        summary: summary.clone(),
                        timestamp_ms: record.timestamp_ms,
                        pool_ordinal: *pool_ordinal,
                    }
                });
                child.parent_run_id = parent_run_id.clone();
                child.agent_name = agent_name.clone();
                child.status = "running".into();
                child.summary = summary.clone();
                child.timestamp_ms = record.timestamp_ms;
                child.pool_ordinal = *pool_ordinal;
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
                && owned_children.contains(child_session_id)
                && child_dir.join(format!("{child_session_id}.jsonl")).exists() =>
            {
                let child = children
                    .get_mut(child_session_id)
                    .expect("owned child was inserted by SubagentStarted");
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

/// Project child summaries without decoding the parent transcript payloads.
///
/// Child navigation only needs a small set of top-level fields. The fast kind
/// check avoids asking serde to walk large tool-result payloads for unrelated
/// records.
pub(crate) fn project_child_session_summaries_from_file(
    child_dir: &Path,
    parent_path: &Path,
) -> Result<Vec<ChildSessionSummary>> {
    #[derive(serde::Deserialize, Default)]
    struct ChildProjectionRecord {
        #[serde(default)]
        session_id: Option<String>,
        #[serde(default)]
        timestamp_ms: Option<u128>,
        #[serde(default)]
        kind: Option<String>,
        #[serde(default)]
        parent_session_id: Option<String>,
        #[serde(default)]
        parent_run_id: Option<String>,
        #[serde(default)]
        child_session_id: Option<String>,
        #[serde(default)]
        agent_name: Option<String>,
        #[serde(default)]
        summary: Option<String>,
        #[serde(default)]
        status: Option<String>,
        #[serde(default)]
        pool_ordinal: Option<u32>,
    }

    let file = File::open(parent_path)?;
    let reader = BufReader::new(file);
    let mut children = BTreeMap::new();
    for line in reader.lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        // Most transcript records contain large arbitrary JSON payloads. Only
        // deserialize lines that can describe a child lifecycle event; parsing
        // every line into a typed struct still makes serde walk those payloads.
        if !line.contains("\"kind\":\"subagent_started\"")
            && !line.contains("\"kind\":\"subagent_result\"")
        {
            continue;
        }
        let record: ChildProjectionRecord = serde_json::from_str(&line)?;
        let (Some(kind), Some(session_id), Some(child_session_id)) = (
            record.kind.as_deref(),
            record.session_id.as_deref(),
            record.child_session_id.as_deref(),
        ) else {
            continue;
        };
        if record.parent_session_id.as_deref() != Some(session_id)
            || !child_dir.join(format!("{child_session_id}.jsonl")).exists()
        {
            continue;
        }
        match kind {
            "subagent_started" => {
                let child = children
                    .entry(child_session_id.to_string())
                    .or_insert_with(|| ChildSessionSummary {
                        parent_session_id: session_id.to_string(),
                        parent_run_id: record.parent_run_id.clone().unwrap_or_default(),
                        child_session_id: child_session_id.to_string(),
                        agent_name: record.agent_name.clone().unwrap_or_default(),
                        status: "running".into(),
                        summary: record.summary.clone().unwrap_or_default(),
                        timestamp_ms: record.timestamp_ms.unwrap_or_default(),
                        pool_ordinal: record.pool_ordinal.unwrap_or_default(),
                    });
                child.parent_run_id = record.parent_run_id.unwrap_or_default();
                child.agent_name = record.agent_name.unwrap_or_default();
                child.status = "running".into();
                child.summary = record.summary.unwrap_or_default();
                child.timestamp_ms = record.timestamp_ms.unwrap_or_default();
                child.pool_ordinal = record.pool_ordinal.unwrap_or_default();
            }
            "subagent_result" => {
                if let Some(child) = children.get_mut(child_session_id) {
                    child.status = record.status.unwrap_or_default();
                    child.summary = record.summary.unwrap_or_default();
                }
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
    Ok(children)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::tempdir;

    #[test]
    fn child_summary_file_scan_ignores_large_unrelated_records() {
        let temp = tempdir().unwrap();
        let child_dir = temp.path().join("children");
        fs::create_dir(&child_dir).unwrap();
        fs::write(child_dir.join("child.jsonl"), "").unwrap();
        let parent_path = temp.path().join("parent.jsonl");
        fs::write(
            &parent_path,
            concat!(
                "{\"session_id\":\"parent\",\"sequence\":1,\"timestamp_ms\":1,\"kind\":\"tool_call_finished\",\"output\":{\"data\":\"large\"}}\n",
                "{\"session_id\":\"parent\",\"sequence\":2,\"timestamp_ms\":2,\"kind\":\"subagent_started\",\"parent_session_id\":\"parent\",\"parent_run_id\":\"run\",\"child_session_id\":\"child\",\"agent_name\":\"explorer\",\"summary\":\"started\",\"pool_ordinal\":1}\n",
                "{\"session_id\":\"parent\",\"sequence\":3,\"timestamp_ms\":3,\"kind\":\"subagent_result\",\"parent_session_id\":\"parent\",\"parent_run_id\":\"run\",\"child_session_id\":\"child\",\"agent_name\":\"explorer\",\"status\":\"done\",\"summary\":\"finished\"}\n",
            ),
        )
        .unwrap();

        let children = project_child_session_summaries_from_file(&child_dir, &parent_path).unwrap();
        assert_eq!(children.len(), 1);
        assert_eq!(children[0].status, "done");
        assert_eq!(children[0].summary, "finished");
    }
}

pub(crate) fn project_job_board(
    child_dir: &Path,
    parent_records: &[TranscriptRecord],
) -> anyhow::Result<Vec<crate::transcript::JobBoardEntry>> {
    use crate::subagent::StructuredSubagentResult;
    use crate::transcript::JobBoardEntry;
    let mut jobs = BTreeMap::<String, JobBoardAccumulator>::new();
    let owned_runs = parent_records
        .iter()
        .filter_map(|record| match &record.event {
            TranscriptEvent::SubagentStarted {
                run_id,
                parent_session_id,
                child_session_id,
                ..
            } if parent_session_id == &record.session_id
                && child_dir.join(format!("{child_session_id}.jsonl")).exists() =>
            {
                Some(run_id.clone())
            }
            _ => None,
        })
        .collect::<std::collections::HashSet<_>>();

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
            } if parent_session_id == &record.session_id
                && owned_runs.contains(run_id)
                && child_dir.join(format!("{child_session_id}.jsonl")).exists() =>
            {
                let entry = jobs.entry(run_id.clone()).or_default();
                entry.run_id = run_id.clone();
                entry.child_session_id = child_session_id.clone();
                entry.agent_name = agent_name.clone();
                entry.status = status.clone();
                entry.summary = summary.clone();
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
            } if owned_runs.contains(run_id) => {
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
                        && entry.status.is_empty()
                    {
                        entry.status = structured.status;
                    }
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
        .map(|entry| JobBoardEntry {
            active: entry.active,
            run_id: entry.run_id,
            child_session_id: entry.child_session_id,
            agent_name: entry.agent_name,
            status: entry.status,
            summary: entry.summary,
        })
        .collect::<Vec<_>>();
    entries.sort_by(|a, b| a.run_id.cmp(&b.run_id));
    Ok(entries)
}

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
            entry.active = false;
        }
    }

    Ok(())
}

fn is_terminal_subagent_status(status: &str) -> bool {
    matches!(
        status,
        "completed" | "failed" | "budget_exhausted" | "cancelled" | "timed_out"
    )
}

#[derive(Debug, Clone, Default)]
struct JobBoardAccumulator {
    run_id: String,
    child_session_id: String,
    agent_name: String,
    status: String,
    summary: String,
    active: bool,
}
