use std::collections::HashSet;

use super::{
    HorizontalRouteOwner, HorizontalRouteSegment, horizontal_routes_do_not_conflict,
    vertical_trunks_may_share,
};

#[test]
fn horizontal_route_ownership_allows_same_net_junctions_and_rejects_foreign_contacts() {
    let owner = HorizontalRouteOwner::Source("S");
    let mut segments = vec![
        HorizontalRouteSegment {
            from: (2, 0),
            to: (2, 2),
            owner,
        },
        HorizontalRouteSegment {
            from: (0, 1),
            to: (2, 1),
            owner,
        },
    ];
    assert!(horizontal_routes_do_not_conflict(
        &segments,
        &HashSet::new(),
        None,
    ));

    segments.push(HorizontalRouteSegment {
        from: (2, 1),
        to: (4, 1),
        owner: HorizontalRouteOwner::Edge(0),
    });
    assert!(!horizontal_routes_do_not_conflict(
        &segments,
        &HashSet::new(),
        None,
    ));
}

#[test]
fn independent_fanout_and_fanin_do_not_share_a_trunk() {
    assert!(!vertical_trunks_may_share(
        HorizontalRouteOwner::Source("fanout"),
        HorizontalRouteOwner::Target("fanin"),
        None,
    ));
    assert!(vertical_trunks_may_share(
        HorizontalRouteOwner::Source("same"),
        HorizontalRouteOwner::Target("same"),
        None,
    ));
}
