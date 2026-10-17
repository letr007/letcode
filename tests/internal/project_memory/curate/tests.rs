use super::*;
use crate::project_memory::store::MemoryRecord;

fn memory(id: &str, title: &str) -> MemoryRecord {
    MemoryRecord {
        id: id.into(),
        kind: "diagnostic".into(),
        title: title.into(),
        summary: format!("{title} 的正文"),
        status: "useful".into(),
        session_id: "s".into(),
        branch_id: "main".into(),
        source_ids: vec!["raw:2".into()],
        paths: vec![],
        created_at_ms: 1,
        state: "active".into(),
        recall_count: 0,
        last_recalled_at: None,
        curated_at_ms: None,
    }
}

fn batch(memories: &[MemoryRecord]) -> CurationBatch {
    prepare_batch(memories).expect("batch")
}

#[test]
fn duplicate_groups_pair_titles_that_differ_only_in_punctuation() {
    let memories = vec![
        memory("a", "LCD 丝印方向不能作为字符方向判据"),
        memory("b", "LCD 丝印方向不能作为字符方向判据"),
        memory("c", "lcd丝印方向不能作为字符方向判据。"),
        memory("d", "审批提示改用左右方向键"),
    ];
    let groups = duplicate_groups(&memories);
    assert_eq!(groups, vec![vec![0, 1, 2]]);
}

#[test]
fn duplicate_groups_ignore_short_containment() {
    let memories = vec![memory("a", "缓存"), memory("b", "缓存失效策略")];
    assert!(duplicate_groups(&memories).is_empty());
    let memories = vec![
        memory("a", "缓存失效策略"),
        memory("b", "缓存失效策略与回源"),
    ];
    assert_eq!(duplicate_groups(&memories), vec![vec![0, 1]]);
}

#[test]
fn prepare_batch_returns_none_without_duplicates() {
    let memories = vec![memory("a", "甲"), memory("b", "乙")];
    assert!(prepare_batch(&memories).is_none());
}

#[test]
fn parse_curation_rejects_ids_outside_the_batch() {
    let memories = vec![
        memory("a", "缓存失效策略"),
        memory("b", "缓存失效策略与回源"),
    ];
    let batch = batch(&memories);
    assert!(
        parse_curation(
            r#"{"merges":[{"retire":"memory:other","into":"a"}],"rewrites":[]}"#,
            &batch
        )
        .is_err()
    );
    assert!(
        parse_curation(
            r#"{"merges":[{"retire":"b","into":"memory:other"}],"rewrites":[]}"#,
            &batch
        )
        .is_err()
    );
    assert!(
        parse_curation(
            r#"{"merges":[{"retire":"b","into":"b"}],"rewrites":[]}"#,
            &batch
        )
        .is_err()
    );
    assert!(
        parse_curation(
            r#"{"merges":[],"rewrites":[{"id":"memory:other","title":"x","summary":"y"}]}"#,
            &batch
        )
        .is_err()
    );
}

#[test]
fn parse_curation_rejects_chain_merges_and_emptying_a_group() {
    let memories = vec![
        memory("a", "缓存失效策略"),
        memory("b", "缓存失效策略与回源"),
        memory("c", "缓存失效策略与回源细节"),
    ];
    let batch = batch(&memories);
    // b 既要被退休、又要当覆盖者，链式合并必须拒绝。
    assert!(
        parse_curation(
            r#"{"merges":[{"retire":"a","into":"b"},{"retire":"b","into":"c"}],"rewrites":[]}"#,
            &batch
        )
        .is_err()
    );
    // 同一目标被退休两次必须拒绝。
    assert!(
        parse_curation(
            r#"{"merges":[{"retire":"a","into":"c"},{"retire":"a","into":"c"}],"rewrites":[]}"#,
            &batch
        )
        .is_err()
    );
    let update = parse_curation(
            r#"{"merges":[{"retire":"a","into":"c"}],"rewrites":[{"id":"c","title":"缓存失效策略与回源细节","summary":"合并后的正文"}]}"#,
            &batch,
        )
        .unwrap();
    assert_eq!(update.merges.len(), 1);
    assert_eq!(update.merges[0].into, "c");
    assert_eq!(update.rewrites.len(), 1);
}
