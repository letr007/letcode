//! Incremental history artifacts. The transcript owns publications and selections;
//! rendering never rewrites a stored paraphrase or treats an archive as deletion.
use anyhow::{Result, ensure};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum HistoryTier {
    Detailed,
    Compact,
    Anchor,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct HistoryCompartment {
    pub id: String,
    pub title: String,
    pub source_ids: Vec<String>,
    pub importance: u8,
    pub detailed: String,
    pub compact: String,
    pub anchor: String,
}

impl HistoryCompartment {
    pub fn text(&self, tier: HistoryTier) -> &str {
        match tier {
            HistoryTier::Detailed => &self.detailed,
            HistoryTier::Compact => &self.compact,
            HistoryTier::Anchor => &self.anchor,
        }
    }

    pub(crate) fn validate(&self) -> Result<()> {
        ensure!(
            !self.id.trim().is_empty(),
            "history compartment id is empty"
        );
        ensure!(
            !self.title.trim().is_empty(),
            "history compartment title is empty"
        );
        ensure!(
            (1..=100).contains(&self.importance),
            "history importance must be 1..100"
        );
        ensure!(
            !self.source_ids.is_empty(),
            "history compartment has no source"
        );
        ensure!(
            !self.detailed.trim().is_empty() && !self.compact.trim().is_empty(),
            "history requires detailed and compact paraphrases"
        );
        // An empty anchor is meaningful: the title alone can identify the episode.
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum HistoryFactCategory {
    ProjectRules,
    Architecture,
    Constraints,
    ConfigValues,
    Naming,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct HistoryFact {
    pub id: String,
    pub category: HistoryFactCategory,
    pub text: String,
    pub source_ids: Vec<String>,
    #[serde(default)]
    pub supersedes: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct HistoryPublication {
    pub id: String,
    #[serde(default)]
    pub project_path: Option<String>,
    #[serde(default)]
    pub external_fact_ids: Vec<String>,
    pub source_ids: Vec<String>,
    pub compartments: Vec<HistoryCompartment>,
    #[serde(default)]
    pub facts: Vec<HistoryFact>,
    #[serde(default)]
    pub withdrawn_fact_ids: Vec<String>,
}

impl HistoryPublication {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            !self.id.trim().is_empty(),
            "history publication id is empty"
        );
        ensure!(
            !self.compartments.is_empty(),
            "history publication has no compartments"
        );
        let mut ids = BTreeSet::new();
        let mut covered = Vec::new();
        for compartment in &self.compartments {
            compartment.validate()?;
            ensure!(ids.insert(&compartment.id), "duplicate compartment id");
            covered.extend(compartment.source_ids.iter().cloned());
        }
        ensure!(
            covered == self.source_ids,
            "history publication coverage is not ordered and complete"
        );
        let unique: BTreeSet<_> = covered.iter().collect();
        ensure!(
            unique.len() == covered.len(),
            "history source is covered twice"
        );
        let mut fact_ids = BTreeSet::new();
        for fact in &self.facts {
            ensure!(
                !fact.id.trim().is_empty() && fact_ids.insert(&fact.id),
                "duplicate or empty fact id"
            );
            ensure!(!fact.text.trim().is_empty(), "history fact text is empty");
            ensure!(
                !fact.source_ids.is_empty() && fact.source_ids.iter().all(|id| unique.contains(id)),
                "history fact cites a source outside its publication"
            );
            ensure!(
                !fact.supersedes.contains(&fact.id),
                "history fact supersedes itself"
            );
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct HistorySelection {
    pub compartment_id: String,
    pub tier: HistoryTier,
}

/// Frozen selections, not provider cache state. Missing compartments are archived
/// but remain available through their original publication and source references.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct HistoryApplication {
    pub publication_ids: Vec<String>,
    pub baseline: Vec<HistorySelection>,
    pub delta: Vec<HistorySelection>,
    #[serde(default)]
    pub baseline_fact_ids: Vec<String>,
    #[serde(default)]
    pub delta_fact_ids: Vec<String>,
    #[serde(default)]
    pub withdrawn_fact_ids: Vec<String>,
    pub first_kept_entry_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub legacy_summary: Option<String>,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct HistoryArchive {
    pub publications: BTreeMap<String, HistoryPublication>,
    #[serde(default)]
    pub publication_order: Vec<String>,
    pub applied_ids: BTreeSet<String>,
    #[serde(default)]
    pub withdrawn_fact_ids: BTreeSet<String>,
    pub application: Option<HistoryApplication>,
}

impl HistoryArchive {
    pub fn from_records(records: &[crate::transcript::TranscriptRecord]) -> Result<Self> {
        let mut archive = Self::default();
        for record in records {
            match &record.event {
                crate::transcript::TranscriptEvent::HistoryPublished(p) => {
                    archive.publish(p.clone())?
                }
                crate::transcript::TranscriptEvent::HistoryApplied(a) => {
                    archive.apply(a.clone())?
                }
                _ => {}
            }
        }
        Ok(archive)
    }

    pub fn publish(&mut self, publication: HistoryPublication) -> Result<()> {
        publication.validate()?;
        ensure!(
            !self.publications.contains_key(&publication.id),
            "duplicate history publication"
        );
        let old_ids: BTreeSet<_> = self
            .publications
            .values()
            .flat_map(|p| p.compartments.iter().map(|c| &c.id))
            .collect();
        let old_sources: BTreeSet<_> = self
            .publications
            .values()
            .flat_map(|p| p.source_ids.iter())
            .collect();
        ensure!(
            publication
                .compartments
                .iter()
                .all(|c| !old_ids.contains(&c.id)),
            "duplicate history compartment"
        );
        ensure!(
            publication
                .source_ids
                .iter()
                .all(|id| !old_sources.contains(id)),
            "history source already published"
        );
        let old_facts: BTreeSet<_> = self
            .publications
            .values()
            .flat_map(|p| p.facts.iter().map(|f| &f.id))
            .collect();
        ensure!(
            publication
                .withdrawn_fact_ids
                .iter()
                .all(|id| old_facts.contains(id) || publication.external_fact_ids.contains(id)),
            "withdrawing an unknown history fact"
        );
        for fact in &publication.facts {
            ensure!(!old_facts.contains(&fact.id), "duplicate history fact");
            ensure!(
                fact.supersedes
                    .iter()
                    .all(|id| old_facts.contains(id) || publication.external_fact_ids.contains(id)),
                "history fact revises an unknown fact"
            );
        }
        self.publication_order.push(publication.id.clone());
        self.publications
            .insert(publication.id.clone(), publication);
        Ok(())
    }

    pub fn apply(&mut self, application: HistoryApplication) -> Result<()> {
        let mut applied = self.applied_ids.clone();
        for id in &application.publication_ids {
            ensure!(
                self.publications.contains_key(id),
                "history application references unpublished work"
            );
            ensure!(
                applied.insert(id.clone()),
                "history publication already applied"
            );
        }
        let allowed: BTreeSet<_> = applied
            .iter()
            .filter_map(|id| self.publications.get(id))
            .flat_map(|p| p.compartments.iter().map(|c| &c.id))
            .collect();
        let mut selected = BTreeSet::new();
        for selection in application.baseline.iter().chain(&application.delta) {
            ensure!(
                allowed.contains(&selection.compartment_id),
                "history selection references unapplied compartment"
            );
            ensure!(
                selected.insert(&selection.compartment_id),
                "history selection is duplicated"
            );
        }
        let allowed_facts: BTreeSet<_> = applied
            .iter()
            .filter_map(|id| self.publications.get(id))
            .flat_map(|p| p.facts.iter().map(|f| &f.id))
            .collect();
        let mut seen_facts = BTreeSet::new();
        for id in application
            .baseline_fact_ids
            .iter()
            .chain(&application.delta_fact_ids)
        {
            ensure!(
                allowed_facts.contains(id) && seen_facts.insert(id),
                "history fact selection is unknown or duplicated"
            );
        }
        ensure!(
            application
                .withdrawn_fact_ids
                .iter()
                .all(|id| allowed_facts.contains(id)
                    || applied
                        .iter()
                        .filter_map(|p| self.publications.get(p))
                        .any(|p| p.external_fact_ids.contains(id))),
            "history withdrawal references an unknown fact"
        );
        self.withdrawn_fact_ids
            .extend(application.withdrawn_fact_ids.iter().cloned());
        self.applied_ids = applied;
        self.application = Some(application);
        Ok(())
    }

    pub fn compartment(&self, id: &str) -> Option<&HistoryCompartment> {
        self.publications
            .values()
            .flat_map(|p| &p.compartments)
            .find(|c| c.id == id)
    }

    pub fn render_facts(&self, ids: &[String]) -> String {
        ids.iter()
            .filter_map(|id| {
                self.publications
                    .values()
                    .flat_map(|p| p.facts.iter())
                    .find(|f| &f.id == id)
            })
            .map(|f| {
                format!(
                    "[evidence:fact:{}] {}{}",
                    f.id,
                    f.text,
                    if f.supersedes.is_empty() {
                        String::new()
                    } else {
                        format!(" (supersedes: {})", f.supersedes.join(", "))
                    }
                )
            })
            .collect::<Vec<_>>()
            .join("\n")
    }

    pub fn effective_fact_ids(&self, additional_publications: &[String]) -> Vec<String> {
        let selected: Vec<_> = self
            .publications
            .iter()
            .filter(|(id, _)| {
                self.applied_ids.contains(*id) || additional_publications.contains(*id)
            })
            .map(|(_, p)| p)
            .collect();
        let removed: BTreeSet<_> = selected
            .iter()
            .flat_map(|p| {
                p.withdrawn_fact_ids
                    .iter()
                    .chain(p.facts.iter().flat_map(|f| f.supersedes.iter()))
            })
            .collect();
        selected
            .iter()
            .flat_map(|p| p.facts.iter())
            .filter(|f| !removed.contains(&f.id) && !self.withdrawn_fact_ids.contains(&f.id))
            .map(|f| f.id.clone())
            .collect()
    }

    pub fn render(&self, selections: &[HistorySelection]) -> String {
        selections
            .iter()
            .filter_map(|s| {
                self.compartment(&s.compartment_id)
                    .map(|c| format!("[compartment:{}] {}\n{}", c.id, c.title, c.text(s.tier)))
            })
            .collect::<Vec<_>>()
            .join("\n\n")
    }
}

/// MC's age/importance decay model, with costs measured from these actual three
/// paraphrases instead of importing its four-tier empirical token constants.
/// Call only at a materialization boundary; persist and replay the selection.
pub fn select_tiers(
    compartments: &[HistoryCompartment],
    budget_tokens: u64,
) -> Vec<HistorySelection> {
    let cost = |c: &HistoryCompartment, tier| {
        ((c.title.len() + c.id.len() + c.text(tier).len()) as u64).div_ceil(3) + 12
    };
    let mut result: Vec<_> = compartments
        .iter()
        .enumerate()
        .map(|(i, c)| {
            let age = compartments.len().saturating_sub(i + 1) as f64;
            let half_life = 24.0 * 2_f64.powf((f64::from(c.importance) - 50.0) / 25.0);
            let normalized_age = age / half_life;
            let detailed = cost(c, HistoryTier::Detailed).max(1) as f64;
            let compact = cost(c, HistoryTier::Compact).max(1) as f64;
            let anchor = cost(c, HistoryTier::Anchor).max(1) as f64;
            let tier = if normalized_age < (detailed / compact.max(anchor)).max(1.0).ln() / 2.0 {
                HistoryTier::Detailed
            } else if normalized_age < (detailed / anchor).max(1.0).ln() / 2.0 {
                HistoryTier::Compact
            } else {
                HistoryTier::Anchor
            };
            HistorySelection {
                compartment_id: c.id.clone(),
                tier,
            }
        })
        .collect();
    let mut used: u64 = result
        .iter()
        .zip(compartments)
        .map(|(s, c)| cost(c, s.tier))
        .sum();
    // Oldest-first budget demotion, with archive as a visibility decision. Each
    // selection moves a finite number of times, including with a zero budget.
    for tier in [HistoryTier::Compact, HistoryTier::Anchor] {
        for (s, c) in result.iter_mut().zip(compartments) {
            if used <= budget_tokens {
                break;
            }
            let old = cost(c, s.tier);
            let next = cost(c, tier);
            if next < old {
                used -= old - next;
                s.tier = tier;
            }
        }
    }
    let mut start = 0;
    while used > budget_tokens && start < result.len() {
        used = used.saturating_sub(cost(&compartments[start], result[start].tier));
        start += 1;
    }
    result.drain(..start);
    result
}

#[cfg(test)]
#[path = "../tests/internal/context_history/tests.rs"]
mod tests;

#[derive(Debug, Clone)]
pub struct HistoryCursor {
    pub session_id: String,
    pub branch_id: String,
    pub leaf_sequence: Option<u64>,
}

pub fn read_sources(cursor: &HistoryCursor) -> Result<Vec<crate::transcript::TranscriptRecord>> {
    ensure!(
        !cursor.session_id.is_empty()
            && cursor
                .session_id
                .chars()
                .all(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '_')),
        "invalid history session id"
    );
    let root = crate::memory::configured_memory_sessions_dir()?;
    let path = root.join(format!("{}.jsonl", cursor.session_id));
    let records = if path.is_file() {
        crate::transcript::read_records(&path)?
    } else {
        crate::transcript::read_child_session_records(&root, &cursor.session_id)?
    };
    crate::transcript::transcript_projection::selected_source_records(
        &records,
        crate::transcript::transcript_projection::SessionContextCursor {
            branch_id: Some(cursor.branch_id.clone()),
            leaf_sequence: cursor.leaf_sequence,
        },
    )
}

#[derive(Debug, Clone, Serialize)]
pub struct HistorySearchEntry {
    pub id: String,
    pub kind: &'static str,
    pub text: String,
    pub source_ids: Vec<String>,
}

pub fn source_entries(
    records: &[crate::transcript::TranscriptRecord],
) -> Result<Vec<HistorySearchEntry>> {
    use crate::transcript::TranscriptEvent;
    let mut entries = Vec::new();
    for record in records {
        let included = matches!(
            record.event,
            TranscriptEvent::UserMessage { .. }
                | TranscriptEvent::AssistantTurn(_)
                | TranscriptEvent::AssistantMessage { .. }
                | TranscriptEvent::AssistantToolCallBatch { .. }
                | TranscriptEvent::ToolCallFinished { .. }
                | TranscriptEvent::ToolCallStarted { .. }
                | TranscriptEvent::ToolCallCancelled { .. }
        );
        if included {
            let id = format!("raw:{}", record.sequence);
            entries.push(HistorySearchEntry {
                source_ids: vec![id.clone()],
                id,
                kind: "transcript",
                text: serde_json::to_string(&record.event)?,
            });
        }
        if let TranscriptEvent::HistoryPublished(publication) = &record.event {
            for c in &publication.compartments {
                entries.push(HistorySearchEntry {
                    id: format!("compartment:{}", c.id),
                    kind: "compartment",
                    text: format!("{}\n{}", c.title, c.detailed),
                    source_ids: c.source_ids.clone(),
                });
            }
        }
    }
    for (id, item) in crate::transcript::transcript_projection::raw_protocol_source_entries(records)
    {
        if let Some(existing) = entries.iter_mut().find(|e| e.id == id) {
            // A canonical entry can merge consecutive legacy tool starts. Its
            // identity covers the whole protocol group, not only the first row.
            existing.text = serde_json::to_string(&item)?;
        } else {
            entries.push(HistorySearchEntry {
                id: id.clone(),
                kind: "transcript",
                text: serde_json::to_string(&item)?,
                source_ids: vec![id],
            });
        }
    }
    let evidence = crate::evidence::restore_evidence_records(records)?;
    let active_facts: BTreeSet<_> = evidence
        .iter()
        .filter(|e| e.tags.iter().any(|t| t == "historian_fact"))
        .map(|e| e.id.as_str())
        .collect();
    for record in records {
        if let crate::transcript::TranscriptEvent::HistoryPublished(p) = &record.event {
            for f in &p.facts {
                let fact_id = format!("fact:{}", f.id);
                entries.push(HistorySearchEntry {
                    id: format!("evidence:{fact_id}"),
                    kind: "fact",
                    source_ids: f.source_ids.clone(),
                    text: format!(
                        "{:?}\n{}\nEffective in this view: {}",
                        f.category,
                        f.text,
                        active_facts.contains(fact_id.as_str())
                    ),
                });
            }
        }
    }
    for e in evidence {
        if e.tags.iter().any(|t| t == "historian_fact") {
            continue;
        }
        let source_ids = match e.source {
            crate::evidence::EvidenceSource::Transcript { sequence } => {
                vec![format!("raw:{sequence}")]
            }
            _ => vec![],
        };
        entries.push(HistorySearchEntry {
            id: format!("evidence:{}", e.id),
            kind: "evidence",
            text: format!(
                "{}\n{}\n{}",
                e.title,
                e.summary,
                e.detail.unwrap_or_default()
            ),
            source_ids,
        });
    }
    Ok(entries)
}

#[cfg(test)]
#[path = "../tests/internal/context_history/integration_tests.rs"]
mod integration_tests;
