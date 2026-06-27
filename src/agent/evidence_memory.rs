use anyhow::Result;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use super::*;

static NEXT_AGENT_EVIDENCE_ID: AtomicU64 = AtomicU64::new(1);

pub(super) fn remember_tool_evidence<C: Config>(
    agent: &mut Agent<C>,
    record: &ToolExecutionRecord,
) -> Result<EvidenceRecord> {
    let mut draft = EvidenceDraft::from_tool_execution_record(record);
    if let EvidenceSource::Subagent { parent_turn_id, .. } = &mut draft.source {
        *parent_turn_id = Some(format!("turn-{}", agent.turn.turn_id));
    }
    let sequence = next_evidence_sequence(agent);
    let id = draft.id.clone().unwrap_or_else(next_agent_evidence_id);
    let record = draft.into_record(id, sequence, 0)?;
    agent.add_evidence(record.clone())?;
    Ok(record)
}

pub(super) fn next_evidence_sequence<C: Config>(agent: &Agent<C>) -> u64 {
    agent
        .evidence
        .iter()
        .map(|record| record.sequence)
        .max()
        .unwrap_or(0)
        .saturating_add(1)
}

fn next_agent_evidence_id() -> String {
    let timestamp_ns = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system clock is before the Unix epoch")
        .as_nanos();
    let counter = NEXT_AGENT_EVIDENCE_ID.fetch_add(1, Ordering::Relaxed);
    format!("ev-agent-{timestamp_ns}-{counter}")
}
