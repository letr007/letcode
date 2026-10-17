use super::*;

fn cache_report() -> CacheUsageReport {
    CacheUsageReport {
        configured: true,
        hint_serialized: true,
        retention_sent: None,
        stable_prefix_segments: 2,
        stable_prompt_tokens: 120,
        volatile_prompt_tokens: 30,
        cacheable_prefix_tokens: 100,
        stable_after_boundary_tokens: 20,
        local_prefix_fingerprint: Some("fp-123".into()),
        routing_key: Some("route-123".into()),
        actual_cached_tokens: None,
    }
}

#[test]
fn cache_metadata_projection_preserves_absent_and_zero_boundaries() {
    let no_prefix = CacheUsageReport {
        stable_prefix_segments: 0,
        local_prefix_fingerprint: None,
        routing_key: None,
        ..cache_report()
    };
    let absent = cache_metadata_projection(&no_prefix, None);
    assert_eq!(absent.first_volatile_index, None);
    assert_eq!(absent.local_prefix_fingerprint, None);
    assert_eq!(absent.routing_key, None);

    let volatile_first = cache_metadata_projection(&no_prefix, Some(0));
    assert_eq!(volatile_first.first_volatile_index, Some(0));
    assert_eq!(volatile_first.local_prefix_fingerprint, None);
    assert_eq!(volatile_first.routing_key, None);

    let stable_prefix = cache_metadata_projection(&cache_report(), Some(2));
    assert_eq!(stable_prefix.stable_prefix_segments, 2);
    assert_eq!(stable_prefix.first_volatile_index, Some(2));
    assert_eq!(
        stable_prefix.local_prefix_fingerprint.as_deref(),
        Some("fp-123")
    );
    assert_eq!(stable_prefix.routing_key.as_deref(), Some("route-123"));
}

#[test]
fn cache_metadata_projection_keeps_plan_and_actual_usage_separate() {
    let prepared = cache_metadata_projection(&cache_report(), Some(2));
    let actual = cache_metadata_projection(&cache_report().with_actual_cached_tokens(77), Some(2));
    assert_eq!(prepared.actual_cached_tokens, None);
    assert_eq!(actual.actual_cached_tokens, Some(77));
    assert_eq!(prepared.stable_prompt_tokens, actual.stable_prompt_tokens);
    assert_eq!(
        prepared.cacheable_prefix_tokens,
        actual.cacheable_prefix_tokens
    );

    let values = actual.values();
    assert!(
        values
            .values()
            .all(|value| matches!(value, Value::Bool(_) | Value::Number(_) | Value::String(_)))
    );
}
