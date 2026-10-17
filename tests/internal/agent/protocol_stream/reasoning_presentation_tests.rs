use super::*;

#[test]
fn summary_suppresses_only_the_duplicate_canonical_item() {
    let summary_items = BTreeSet::from(["r1".to_string()]);
    let reasoning_text = BTreeMap::from([("r1".to_string(), "raw reasoning".to_string())]);
    assert_eq!(
        reasoning_done_presentation_text(
            "r1",
            "summary\n\nraw reasoning",
            &summary_items,
            &reasoning_text,
        ),
        Some("raw reasoning")
    );
    assert_eq!(
        reasoning_done_presentation_text(
            "r2",
            "summary only",
            &BTreeSet::from(["r2".to_string()]),
            &BTreeMap::new(),
        ),
        None
    );
    assert_eq!(
        reasoning_done_presentation_text(
            "r3",
            "plain reasoning",
            &BTreeSet::new(),
            &BTreeMap::new(),
        ),
        Some("plain reasoning")
    );
}
