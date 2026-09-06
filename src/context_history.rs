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

    fn validate(&self) -> Result<()> {
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
mod tests {
    use super::*;
    fn compartment(id: &str) -> HistoryCompartment {
        HistoryCompartment {
            id: id.into(),
            title: "Investigated parser".into(),
            source_ids: vec![format!("raw:{id}")],
            importance: 70,
            detailed: "details ".repeat(100),
            compact: "result ".repeat(20),
            anchor: "parser fixed".into(),
        }
    }
    fn publication() -> HistoryPublication {
        let c = compartment("1");
        HistoryPublication {
            id: "pub1".into(),
            project_path: None,
            external_fact_ids: vec![],
            source_ids: c.source_ids.clone(),
            compartments: vec![c],
            facts: vec![],
            withdrawn_fact_ids: vec![],
        }
    }
    #[test]
    fn publication_requires_exact_nonoverlapping_coverage() {
        let mut p = publication();
        p.validate().unwrap();
        p.source_ids.push("raw:missing".into());
        assert!(p.validate().is_err());
    }
    #[test]
    fn published_work_is_not_applied_and_archive_does_not_delete_it() {
        let mut archive = HistoryArchive::default();
        archive.publish(publication()).unwrap();
        assert!(archive.applied_ids.is_empty());
        archive
            .apply(HistoryApplication {
                publication_ids: vec!["pub1".into()],
                baseline: vec![],
                delta: vec![],
                baseline_fact_ids: vec![],
                delta_fact_ids: vec![],
                withdrawn_fact_ids: vec![],
                first_kept_entry_id: None,
                legacy_summary: None,
            })
            .unwrap();
        assert!(archive.compartment("1").is_some());
        assert!(archive.render(&[]).is_empty());
    }
    #[test]
    fn application_cannot_reference_unpublished_or_duplicate_work() {
        let mut archive = HistoryArchive::default();
        let app = HistoryApplication {
            publication_ids: vec!["pub1".into()],
            baseline: vec![],
            delta: vec![],
            baseline_fact_ids: vec![],
            delta_fact_ids: vec![],
            withdrawn_fact_ids: vec![],
            first_kept_entry_id: None,
            legacy_summary: None,
        };
        assert!(archive.apply(app.clone()).is_err());
        archive.publish(publication()).unwrap();
        archive.apply(app.clone()).unwrap();
        assert!(archive.apply(app).is_err());
    }
    #[test]
    fn selections_are_deterministic_and_zero_budget_archives() {
        let c = vec![compartment("1"), compartment("2")];
        assert_eq!(select_tiers(&c, 200), select_tiers(&c, 200));
        assert!(select_tiers(&c, 0).is_empty());
        assert_eq!(select_tiers(&c, 10000).len(), 2);
    }
}

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
    let active_facts: BTreeSet<_> = crate::evidence::restore_evidence_records(records)?
        .into_iter()
        .filter(|e| e.tags.iter().any(|t| t == "historian_fact"))
        .map(|e| e.id)
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
                        active_facts.contains(&fact_id)
                    ),
                });
            }
        }
    }
    for e in crate::evidence::restore_evidence_records(records)? {
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
mod history_integration_tests {
    use super::*;
    use crate::request_builder::HistoryItem;
    use crate::transcript::{TranscriptEvent as E, TranscriptRecord, restore_session_history};
    fn record(sequence: u64, event: E) -> TranscriptRecord {
        TranscriptRecord {
            session_id: "history-test".into(),
            sequence,
            timestamp_ms: 0,
            context_branch_id: None,
            event,
        }
    }
    fn fixture() -> (
        Vec<TranscriptRecord>,
        HistoryPublication,
        HistoryApplication,
    ) {
        let records = vec![
            record(
                1,
                E::UserMessage {
                    content: "Find the parser failure".into(),
                },
            ),
            record(
                2,
                E::AssistantMessage {
                    content: "UTF-8 boundary caused the failure".into(),
                },
            ),
            record(
                3,
                E::UserMessage {
                    content: "Continue with the fix".into(),
                },
            ),
        ];
        let source_ids = vec!["raw:1".into(), "raw:2".into()];
        let p = HistoryPublication {
            id: "p1".into(),
            project_path: None,
            external_fact_ids: vec![],
            withdrawn_fact_ids: vec![],
            source_ids: source_ids.clone(),
            compartments: vec![HistoryCompartment {
                id: "c1".into(),
                title: "Parser failure".into(),
                source_ids: source_ids.clone(),
                importance: 80,
                detailed: "UTF-8 boundary caused the failure".into(),
                compact: "Parser boundary error".into(),
                anchor: "UTF-8".into(),
            }],
            facts: vec![HistoryFact {
                id: "f1".into(),
                category: HistoryFactCategory::Constraints,
                text: "Parser offsets are byte offsets".into(),
                source_ids,
                supersedes: vec![],
            }],
        };
        let a = HistoryApplication {
            publication_ids: vec!["p1".into()],
            baseline: vec![HistorySelection {
                compartment_id: "c1".into(),
                tier: HistoryTier::Compact,
            }],
            delta: vec![],
            baseline_fact_ids: vec![],
            delta_fact_ids: vec![],
            withdrawn_fact_ids: vec![],
            first_kept_entry_id: Some("raw:3".into()),
            legacy_summary: None,
        };
        (records, p, a)
    }
    #[test]
    fn history_publication_and_application_replay_separately() {
        let (mut records, p, a) = fixture();
        records.push(record(4, E::HistoryPublished(p)));
        assert_eq!(restore_session_history(&records).unwrap().len(), 3);
        records.push(record(5, E::HistoryApplied(a)));
        let active = restore_session_history(&records).unwrap();
        assert!(
            matches!(&active[0],HistoryItem::ContextSummary {text} if text.contains("Parser boundary error"))
        );
        assert_eq!(active.len(), 2);
        let entries = source_entries(&records).unwrap();
        assert!(
            entries
                .iter()
                .any(|e| e.id == "raw:2" && e.text.contains("UTF-8 boundary"))
        );
        assert!(entries.iter().any(|e| e.id == "evidence:fact:f1"));
    }
    #[test]
    fn history_cannot_retire_uncovered_tail_or_publish_unknown_source() {
        let (mut records, p, mut a) = fixture();
        records.push(record(4, E::HistoryPublished(p)));
        a.first_kept_entry_id = None;
        records.push(record(5, E::HistoryApplied(a)));
        assert!(restore_session_history(&records).is_err());
        let (mut records, mut p, _) = fixture();
        p.source_ids = vec!["raw:99".into()];
        p.compartments[0].source_ids = p.source_ids.clone();
        p.facts.clear();
        records.push(record(4, E::HistoryPublished(p)));
        assert!(restore_session_history(&records).is_err());
    }
    #[test]
    fn history_branch_scope_cannot_read_sibling_sources() {
        let (mut records, p, _) = fixture();
        records.push(record(
            4,
            E::ContextBranchCreated {
                branch_id: "other".into(),
                parent_branch_id: crate::transcript::ROOT_CONTEXT_BRANCH_ID.into(),
                base_sequence: 2,
                label: None,
            },
        ));
        let mut sibling = record(
            5,
            E::UserMessage {
                content: "sibling only".into(),
            },
        );
        sibling.context_branch_id = Some("other".into());
        records.push(sibling);
        let selected = crate::transcript::transcript_projection::selected_source_records(
            &records,
            crate::transcript::transcript_projection::SessionContextCursor {
                branch_id: Some(crate::transcript::ROOT_CONTEXT_BRANCH_ID.into()),
                leaf_sequence: None,
            },
        )
        .unwrap();
        assert!(
            !source_entries(&selected)
                .unwrap()
                .iter()
                .any(|e| e.text.contains("sibling only"))
        );
        let mut invalid = p;
        invalid.source_ids = vec!["raw:5".into()];
        invalid.compartments[0].source_ids = invalid.source_ids.clone();
        invalid.facts.clear();
        records.push(record(6, E::HistoryPublished(invalid)));
        assert!(restore_session_history(&records).is_err());
    }
    #[test]
    fn canonical_legacy_tool_source_includes_every_call_in_its_group() {
        let records = vec![
            record(
                1,
                E::ToolCallStarted {
                    call_id: "a".into(),
                    name: "fs__read".into(),
                    args: serde_json::json!({"path":"first.rs"}),
                },
            ),
            record(
                2,
                E::ToolCallStarted {
                    call_id: "b".into(),
                    name: "fs__read".into(),
                    args: serde_json::json!({"path":"second.rs"}),
                },
            ),
            record(
                3,
                E::ToolCallFinished {
                    call_id: "a".into(),
                    name: "fs__read".into(),
                    ok: true,
                    output: crate::tool::ToolResult::ok(
                        "fs__read",
                        serde_json::json!({"content":"first"}),
                    ),
                },
            ),
            record(
                4,
                E::ToolCallFinished {
                    call_id: "b".into(),
                    name: "fs__read".into(),
                    ok: true,
                    output: crate::tool::ToolResult::ok(
                        "fs__read",
                        serde_json::json!({"content":"second"}),
                    ),
                },
            ),
        ];
        let entries = source_entries(&records).unwrap();
        let group = entries.iter().find(|e| e.id == "raw:1").unwrap();
        assert!(group.text.contains("first.rs"));
        assert!(group.text.contains("second.rs"));
    }

    #[test]
    fn independent_facts_from_one_publication_are_not_deduplicated() {
        let (mut records, mut p, mut a) = fixture();
        a.baseline_fact_ids = vec!["f1".into(), "f2".into()];
        let mut second = p.facts[0].clone();
        second.id = "f2".into();
        second.text = "Parser input is UTF-8".into();
        p.facts.push(second);
        records.push(record(4, E::HistoryPublished(p)));
        records.push(record(5, E::HistoryApplied(a)));
        let facts = crate::evidence::restore_evidence_records(&records).unwrap();
        let (message, ids, _) = crate::evidence::evidence_context_message(&facts, "Parser", 3000);
        assert!(
            message.is_none() && ids.is_empty(),
            "facts are rendered by the frozen history slots, not injected twice"
        );
        let rendered = restore_session_history(&records).unwrap();
        assert!(
            matches!(&rendered[0],HistoryItem::ContextSummary {text} if text.contains("Parser offsets are byte offsets") && text.contains("Parser input is UTF-8"))
        );
    }

    #[test]
    fn memory_projects_facts_from_one_selected_branch() {
        let (mut records, p, a) = fixture();
        records.push(record(4, E::HistoryPublished(p.clone())));
        records.push(record(5, E::HistoryApplied(a.clone())));
        records.push(record(
            6,
            E::ContextBranchCreated {
                branch_id: "alternative".into(),
                parent_branch_id: crate::transcript::ROOT_CONTEXT_BRANCH_ID.into(),
                base_sequence: 3,
                label: None,
            },
        ));
        let mut other = p;
        other.facts[0].text = "Sibling alternative constraint".into();
        let mut published = record(7, E::HistoryPublished(other));
        published.context_branch_id = Some("alternative".into());
        records.push(published);
        let mut applied = record(8, E::HistoryApplied(a));
        applied.context_branch_id = Some("alternative".into());
        records.push(applied);
        records.push(record(
            9,
            E::ContextCheckout {
                branch_id: crate::transcript::ROOT_CONTEXT_BRANCH_ID.into(),
                leaf_sequence: 5,
            },
        ));
        let memories = crate::memory::project_memory_objects("history-test", &records).unwrap();
        assert_eq!(memories.len(), 1);
        assert_eq!(memories[0].summary, "Parser offsets are byte offsets");
        assert_eq!(
            memories[0].branch_id.as_deref(),
            Some(crate::transcript::ROOT_CONTEXT_BRANCH_ID)
        );
    }

    #[test]
    fn legacy_bootstrap_cannot_be_dropped_by_an_application() {
        let (mut records, mut p, mut a) = fixture();
        records.push(record(
            4,
            E::ContextCompaction(crate::agent::ContextCompactionEvent::succeeded_at(
                "legacy decision",
                Some("raw:3".into()),
            )),
        ));
        p.source_ids = vec!["raw:3".into()];
        p.compartments[0].source_ids = p.source_ids.clone();
        p.facts.clear();
        records.push(record(5, E::HistoryPublished(p)));
        a.first_kept_entry_id = None;
        records.push(record(6, E::HistoryApplied(a.clone())));
        assert!(restore_session_history(&records).is_err());
        a.legacy_summary = Some("legacy decision".into());
        records.last_mut().unwrap().event = E::HistoryApplied(a);
        let history = restore_session_history(&records).unwrap();
        assert!(
            matches!(&history[0], HistoryItem::ContextSummary { text } if text.contains("legacy decision"))
        );
    }

    #[test]
    fn facts_revise_projection_without_deleting_original_observation() {
        let (mut records, p, a) = fixture();
        records.push(record(4, E::HistoryPublished(p.clone())));
        assert!(
            crate::evidence::restore_evidence_records(&records)
                .unwrap()
                .is_empty()
        );
        records.push(record(5, E::HistoryApplied(a)));
        let mut next = p;
        next.id = "p2".into();
        next.source_ids = vec!["raw:3".into()];
        next.compartments[0].id = "c2".into();
        next.compartments[0].source_ids = next.source_ids.clone();
        next.facts[0].id = "f2".into();
        next.facts[0].text = "Corrected parser constraint".into();
        next.facts[0].source_ids = next.source_ids.clone();
        next.facts[0].supersedes = vec!["f1".into()];
        records.push(record(6, E::HistoryPublished(next)));
        assert_eq!(
            crate::evidence::restore_evidence_records(&records).unwrap()[0].id,
            "fact:f1"
        );
        records.push(record(
            7,
            E::HistoryApplied(HistoryApplication {
                publication_ids: vec!["p2".into()],
                baseline: vec![],
                delta: vec![],
                baseline_fact_ids: vec![],
                delta_fact_ids: vec![],
                withdrawn_fact_ids: vec![],
                first_kept_entry_id: None,
                legacy_summary: None,
            }),
        ));
        restore_session_history(&records).unwrap();
        let archived = source_entries(&records).unwrap();
        let previous = archived
            .iter()
            .find(|e| e.id == "evidence:fact:f1")
            .unwrap();
        assert_eq!(previous.source_ids, vec!["raw:1", "raw:2"]);
        assert!(previous.text.contains("false"));
        let facts = crate::evidence::restore_evidence_records(&records).unwrap();
        assert_eq!(facts.len(), 1);
        assert_eq!(facts[0].id, "fact:f2");
        assert!(matches!(&records[3].event,E::HistoryPublished(p) if p.facts[0].id=="f1"));
    }
}
