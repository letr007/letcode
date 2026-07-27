use super::{ContextBranchInfo, ResolvedBranchContext, SessionContextCursor};
use crate::transcript::{ROOT_CONTEXT_BRANCH_ID, TranscriptEvent, TranscriptRecord};
use anyhow::{anyhow, ensure};
use std::collections::BTreeMap;

#[derive(Debug, Clone)]
struct BranchDefinition {
    parent_branch_id: Option<String>,
    base_sequence: u64,
    label: Option<String>,
}

#[derive(Debug, Clone)]
struct CheckoutState {
    branch_id: String,
    leaf_sequence: u64,
}

#[derive(Debug, Default)]
pub(super) struct BranchIndex {
    definitions: BTreeMap<String, BranchDefinition>,
    latest_checkout: Option<CheckoutState>,
    branch_tips: BTreeMap<String, u64>,
}

pub(crate) fn list_context_branches(
    records: &[TranscriptRecord],
    current_branch_id: Option<&str>,
) -> anyhow::Result<Vec<ContextBranchInfo>> {
    let index = build_branch_index(records)?;
    let active_branch_id = resolve_active_branch_id(&index, current_branch_id);
    let mut branches = index
        .definitions
        .iter()
        .map(|(branch_id, definition)| {
            Ok(ContextBranchInfo {
                branch_id: branch_id.clone(),
                parent_branch_id: definition.parent_branch_id.clone(),
                label: definition.label.clone(),
                tip_sequence: index.branch_tip(branch_id)?,
                is_current: branch_id == &active_branch_id,
            })
        })
        .collect::<anyhow::Result<Vec<_>>>()?;
    branches.sort_by(|left, right| {
        (left.branch_id != ROOT_CONTEXT_BRANCH_ID)
            .cmp(&(right.branch_id != ROOT_CONTEXT_BRANCH_ID))
            .then_with(|| left.branch_id.cmp(&right.branch_id))
    });
    Ok(branches)
}

pub(super) fn resolve_branch_context(
    records: Vec<TranscriptRecord>,
    cursor: SessionContextCursor,
) -> anyhow::Result<ResolvedBranchContext> {
    let index = build_branch_index(&records)?;
    let branch_id = cursor
        .branch_id
        .unwrap_or_else(|| resolve_active_branch_id(&index, None));
    let leaf_sequence = match cursor.leaf_sequence {
        Some(leaf_sequence) => leaf_sequence,
        // A checkout chooses the active branch/scope, not a permanently frozen
        // content leaf. Explicit cursors are the sole way to request a cut.
        None => index.branch_tip(&branch_id)?,
    };

    let max_sequence = records
        .iter()
        .map(|record| record.sequence)
        .max()
        .unwrap_or(0);
    ensure!(
        leaf_sequence <= max_sequence || (leaf_sequence == 0 && max_sequence == 0),
        "session context leaf_sequence {leaf_sequence} exceeds max transcript sequence {max_sequence}"
    );

    // An explicit cursor retains the latest scope interval for its branch even
    // when a later checkout selected a different branch.
    let scope_checkout_sequence = records.iter().rev().find_map(|record| {
        matches!(
            &record.event,
            TranscriptEvent::ContextCheckout {
                branch_id: checkout_branch_id,
                ..
            } if checkout_branch_id == &branch_id
        )
        .then_some(record.sequence)
    });
    let records = collect_branch_path_records(&records, &index, &branch_id, leaf_sequence)?;
    Ok(ResolvedBranchContext {
        branch_id,
        leaf_sequence,
        scope_checkout_sequence,
        records,
    })
}

pub(super) fn build_branch_index(records: &[TranscriptRecord]) -> anyhow::Result<BranchIndex> {
    let mut index = BranchIndex::default();
    index.definitions.insert(
        ROOT_CONTEXT_BRANCH_ID.to_string(),
        BranchDefinition {
            parent_branch_id: None,
            base_sequence: 0,
            label: None,
        },
    );
    index
        .branch_tips
        .insert(ROOT_CONTEXT_BRANCH_ID.to_string(), 0);

    for (position, record) in records.iter().enumerate() {
        match &record.event {
            TranscriptEvent::ContextBranchCreated {
                branch_id,
                parent_branch_id,
                base_sequence,
                label,
            } => {
                ensure!(
                    !index.definitions.contains_key(branch_id),
                    "duplicate context branch_id '{branch_id}'"
                );
                ensure!(
                    index.definitions.contains_key(parent_branch_id),
                    "missing parent context branch '{parent_branch_id}' for branch '{branch_id}'"
                );
                ensure!(
                    base_sequence_resolves_on_parent_path(
                        &records[..position],
                        &index,
                        parent_branch_id,
                        *base_sequence,
                    )?,
                    "context branch '{branch_id}' base_sequence {base_sequence} is not resolvable on parent branch '{parent_branch_id}'"
                );
                index.definitions.insert(
                    branch_id.clone(),
                    BranchDefinition {
                        parent_branch_id: Some(parent_branch_id.clone()),
                        base_sequence: *base_sequence,
                        label: label.clone(),
                    },
                );
                index.branch_tips.insert(branch_id.clone(), *base_sequence);
            }
            TranscriptEvent::ContextCheckout {
                branch_id,
                leaf_sequence,
            } => {
                ensure!(
                    index.definitions.contains_key(branch_id),
                    "unknown context branch '{branch_id}' in checkout metadata"
                );
                index.latest_checkout = Some(CheckoutState {
                    branch_id: branch_id.clone(),
                    leaf_sequence: *leaf_sequence,
                });
            }
            TranscriptEvent::ContextBranchSummary {
                branch_id,
                leaf_sequence,
                ..
            } => {
                ensure!(
                    index.definitions.contains_key(branch_id),
                    "unknown context branch '{branch_id}' in branch summary metadata"
                );
                let branch_tip = index.branch_tip(branch_id)?;
                ensure!(
                    *leaf_sequence <= branch_tip,
                    "context branch summary leaf_sequence {leaf_sequence} exceeds tip {branch_tip} for branch '{branch_id}'"
                );
            }
            _ => {
                if record.event.is_context_branch_metadata() {
                    continue;
                }
                let effective_branch_id = effective_branch_id(record);
                ensure!(
                    index.definitions.contains_key(effective_branch_id),
                    "unknown context branch '{effective_branch_id}' in record scope at sequence {}",
                    record.sequence
                );
                let branch_tip = index.branch_tip(effective_branch_id)?;
                index.branch_tips.insert(
                    effective_branch_id.to_string(),
                    branch_tip.max(record.sequence),
                );
            }
        }
    }

    if let Some(checkout) = &index.latest_checkout {
        let branch_tip = index.branch_tip(&checkout.branch_id)?;
        ensure!(
            checkout.leaf_sequence <= branch_tip,
            "context checkout leaf_sequence {} exceeds tip {} for branch '{}'",
            checkout.leaf_sequence,
            branch_tip,
            checkout.branch_id
        );
    }

    Ok(index)
}

fn base_sequence_resolves_on_parent_path(
    records: &[TranscriptRecord],
    index: &BranchIndex,
    parent_branch_id: &str,
    base_sequence: u64,
) -> anyhow::Result<bool> {
    if base_sequence == 0 {
        return Ok(true);
    }

    if base_sequence > branch_tip_for_records(records, index, parent_branch_id)? {
        return Ok(false);
    }

    let path = collect_branch_path_records(records, index, parent_branch_id, base_sequence)?;
    Ok(path.iter().any(|record| record.sequence == base_sequence))
}

pub(super) fn branch_tip_for_records(
    records: &[TranscriptRecord],
    index: &BranchIndex,
    branch_id: &str,
) -> anyhow::Result<u64> {
    let definition = index
        .definitions
        .get(branch_id)
        .ok_or_else(|| anyhow!("unknown context branch '{branch_id}'"))?;
    let local_tip = records
        .iter()
        .filter(|record| !record.event.is_context_branch_metadata())
        .filter(|record| effective_branch_id(record) == branch_id)
        .map(|record| record.sequence)
        .max()
        .unwrap_or(definition.base_sequence);
    Ok(local_tip.max(definition.base_sequence))
}

pub(super) fn collect_branch_path_records(
    records: &[TranscriptRecord],
    index: &BranchIndex,
    branch_id: &str,
    leaf_sequence: u64,
) -> anyhow::Result<Vec<TranscriptRecord>> {
    let branch_tip = index.branch_tip(branch_id)?;
    ensure!(
        leaf_sequence <= branch_tip,
        "requested leaf_sequence {leaf_sequence} exceeds tip {branch_tip} for branch '{branch_id}'"
    );

    if branch_id == ROOT_CONTEXT_BRANCH_ID {
        return Ok(records
            .iter()
            .filter(|record| !record.event.is_context_branch_metadata())
            .filter(|record| effective_branch_id(record) == ROOT_CONTEXT_BRANCH_ID)
            .filter(|record| record.sequence <= leaf_sequence)
            .cloned()
            .collect());
    }

    let definition = index
        .definitions
        .get(branch_id)
        .ok_or_else(|| anyhow!("unknown context branch '{branch_id}'"))?;
    let parent_branch_id = definition
        .parent_branch_id
        .as_deref()
        .ok_or_else(|| anyhow!("context branch '{branch_id}' is missing a parent"))?;
    ensure!(
        leaf_sequence >= definition.base_sequence,
        "requested leaf_sequence {leaf_sequence} precedes base_sequence {} for branch '{branch_id}'",
        definition.base_sequence
    );

    let mut path =
        collect_branch_path_records(records, index, parent_branch_id, definition.base_sequence)?;
    path.extend(
        records
            .iter()
            .filter(|record| !record.event.is_context_branch_metadata())
            .filter(|record| effective_branch_id(record) == branch_id)
            .filter(|record| record.sequence <= leaf_sequence)
            .cloned(),
    );
    Ok(path)
}

fn effective_branch_id(record: &TranscriptRecord) -> &str {
    record
        .context_branch_id
        .as_deref()
        .unwrap_or(ROOT_CONTEXT_BRANCH_ID)
}

pub(super) fn resolve_active_branch_id(
    index: &BranchIndex,
    current_branch_id: Option<&str>,
) -> String {
    current_branch_id
        .map(str::to_string)
        .or_else(|| {
            index
                .latest_checkout
                .as_ref()
                .map(|checkout| checkout.branch_id.clone())
        })
        .unwrap_or_else(|| ROOT_CONTEXT_BRANCH_ID.to_string())
}

/// Resolves the effective checkout branch from this journal prefix. Content is
/// deliberately resolved separately at the branch's actual content boundary.
pub(crate) fn effective_branch_id_at_frontier(
    records: &[TranscriptRecord],
) -> anyhow::Result<String> {
    Ok(resolve_active_branch_id(
        &build_branch_index(records)?,
        None,
    ))
}

pub(super) fn branch_parent_id(
    records: &[TranscriptRecord],
    branch_id: &str,
) -> anyhow::Result<Option<String>> {
    let index = build_branch_index(records)?;
    Ok(index
        .definitions
        .get(branch_id)
        .and_then(|definition| definition.parent_branch_id.clone()))
}

impl BranchIndex {
    fn branch_tip(&self, branch_id: &str) -> anyhow::Result<u64> {
        self.branch_tips
            .get(branch_id)
            .copied()
            .ok_or_else(|| anyhow!("unknown context branch '{branch_id}'"))
    }
}
