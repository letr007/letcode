use super::{PlacedEntity, port_col};

#[test]
fn full_port_capacity_uses_each_interior_column_once() {
    let placed = PlacedEntity {
        row: 0,
        col: 4,
        width: 7,
        height: 3,
    };
    let ports = (0..5)
        .map(|ordinal| port_col(placed, ordinal, 5).unwrap())
        .collect::<Vec<_>>();
    assert_eq!(ports, vec![5, 6, 7, 8, 9]);
}
