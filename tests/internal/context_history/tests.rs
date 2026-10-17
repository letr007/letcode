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
