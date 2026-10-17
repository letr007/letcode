use super::*;

fn geometry(position: usize) -> ScrollbarGeometry {
    ScrollbarGeometry::new(4, 100, 10, position).expect("scrollable")
}

fn backdrop() -> Color {
    Color::Rgb(18, 18, 18)
}

#[test]
fn hidden_when_the_transcript_fits() {
    assert!(ScrollbarGeometry::new(10, 10, 10, 0).is_none());
    assert!(ScrollbarGeometry::new(10, 4, 10, 0).is_none());
    assert!(ScrollbarGeometry::new(0, 100, 10, 0).is_none());
    assert!(ScrollbarGeometry::new(10, 100, 10, 0).is_some());
}

#[test]
fn thumb_stays_one_cell_long_and_travels_the_whole_track() {
    let top = geometry(0);
    assert_eq!((top.thumb_start_eighths(), top.thumb_end_eighths()), (0, 8));

    let bottom = geometry(90);
    assert_eq!(
        (bottom.thumb_start_eighths(), bottom.thumb_end_eighths()),
        (24, 32)
    );
}

#[test]
fn thumb_lands_on_eighth_boundaries() {
    let middle = geometry(45);
    assert_eq!(
        (middle.thumb_start_eighths(), middle.thumb_end_eighths()),
        (12, 20)
    );
    assert!(!middle.contains_row(0));
    assert!(middle.contains_row(1));
    assert!(middle.contains_row(2));
    assert!(!middle.contains_row(3));
}

#[test]
fn thumb_start_round_trips_through_the_position() {
    for start in 0..=24 {
        let position = geometry(0).position_for_thumb_start(start);
        assert_eq!(geometry(position).thumb_start_eighths(), start);
    }
}

#[test]
fn grabbing_the_thumb_keeps_it_under_the_pointer() {
    let geometry = geometry(45);
    assert_eq!(geometry.thumb_start_eighths(), 12);

    let top_edge = geometry.grab_offset(1);
    assert_eq!(top_edge, 0);
    assert_eq!(geometry.position_for_grab(1, top_edge), 45);

    let centred = geometry.grab_offset(2);
    assert_eq!(centred, 8);
    assert_eq!(geometry.position_for_grab(0, centred), 0);
    assert_eq!(geometry.position_for_grab(9, centred), 90);
}

#[test]
fn cells_render_blocks_only_where_the_thumb_is() {
    let theme = Theme::dark();

    let top = geometry(0);
    assert_eq!(top.cell(0, theme, backdrop()).0, '█');
    assert_eq!(top.cell(1, theme, backdrop()).0, '│');
    assert_eq!(top.cell(1, theme, backdrop()).1.fg, Some(theme.border));

    let middle = geometry(45);
    assert_eq!(middle.cell(0, theme, backdrop()).0, '│');
    assert_eq!(middle.cell(1, theme, backdrop()).0, '▄');
    assert_eq!(
        middle.cell(1, theme, backdrop()).1.fg,
        Some(theme.muted_text)
    );
    assert_eq!(middle.cell(2, theme, backdrop()).0, '▄');
    assert_eq!(middle.cell(2, theme, backdrop()).1.fg, Some(backdrop()));
    assert_eq!(
        middle.cell(2, theme, backdrop()).1.bg,
        Some(theme.muted_text)
    );
    assert_eq!(middle.cell(3, theme, backdrop()).0, '│');
}

#[test]
fn every_cell_of_a_tall_track_stays_within_the_thumb_bounds() {
    let theme = Theme::dark();
    let geometry = ScrollbarGeometry::new(40, 1000, 40, 500).expect("scrollable");
    let (start, end) = (geometry.thumb_start_eighths(), geometry.thumb_end_eighths());
    for row in 0..40 {
        let (symbol, _) = geometry.cell(row, theme, backdrop());
        let row_start = row * EIGHTHS;
        let row_end = row_start + EIGHTHS;
        let filled = end > row_start && start < row_end;
        assert_eq!(filled, symbol != '│', "row {row} symbol {symbol}");
    }
    assert!(geometry.contains_row(start / EIGHTHS));
    assert!(geometry.contains_row((end - 1) / EIGHTHS));
}
