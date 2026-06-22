use anyhow::Result;

use super::*;

pub(super) fn remember_tool_evidence<C: Config>(
    agent: &mut Agent<C>,
    record: &ToolExecutionRecord,
) -> Result<EvidenceRecord> {
    let mut draft = EvidenceDraft::from_tool_execution_record(record);
    if let EvidenceSource::Subagent { parent_turn_id, .. } = &mut draft.source {
        *parent_turn_id = Some(format!("turn-{}", agent.turn.turn_id));
    }
    let sequence = next_evidence_sequence(agent);
    let id = draft
        .id
        .clone()
        .unwrap_or_else(|| format!("ev-agent-{sequence:06}"));
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
