use super::TrackAllocator;

#[test]
fn reserve_is_atomic_for_contiguous_spans() {
    let mut allocator = TrackAllocator::default();
    assert_eq!(allocator.reserve(2, 0, 3, 2), Some(2));
    assert_eq!(allocator.reserve(1, 0, 3, 2), Some(0));
    assert_eq!(allocator.reserve(1, 0, 3, 2), None);
    assert_eq!(allocator.allocate(3, 0, 3), None);
}
