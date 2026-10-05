use ratatui::style::{Color, Style};

use crate::tui::theme::Theme;

const EIGHTHS: usize = 8;
const FULL_BLOCK: char = '█';
const TRACK_SYMBOL: char = '│';
const LOWER_BLOCKS: [char; EIGHTHS] = ['▁', '▂', '▃', '▄', '▅', '▆', '▇', '█'];

pub(crate) struct ScrollbarGeometry {
    track_rows: usize,
    total_rows: usize,
    viewport_rows: usize,
    position: usize,
}

impl ScrollbarGeometry {
    pub(crate) fn new(
        track_rows: usize,
        total_rows: usize,
        viewport_rows: usize,
        position: usize,
    ) -> Option<Self> {
        (track_rows > 0 && total_rows > viewport_rows).then_some(Self {
            track_rows,
            total_rows,
            viewport_rows,
            position,
        })
    }

    pub(crate) fn thumb_start_eighths(&self) -> usize {
        let travel = self.travel_eighths();
        if travel == 0 {
            return 0;
        }
        let position = self.position.min(self.max_scroll());
        (position * travel + self.max_scroll() / 2) / self.max_scroll()
    }

    pub(crate) fn thumb_end_eighths(&self) -> usize {
        self.thumb_start_eighths() + self.thumb_length_eighths()
    }

    pub(crate) fn contains_row(&self, row: usize) -> bool {
        self.thumb_end_eighths() > row * EIGHTHS && self.thumb_start_eighths() < (row + 1) * EIGHTHS
    }

    pub(crate) fn grab_offset(&self, track_row: usize) -> usize {
        self.row_centre_eighths(track_row)
            .saturating_sub(self.thumb_start_eighths())
    }

    pub(crate) fn position_for_grab(&self, track_row: usize, grab_eighths: usize) -> usize {
        let start = self
            .row_centre_eighths(track_row)
            .saturating_sub(grab_eighths);
        self.position_for_thumb_start(start)
    }

    pub(crate) fn position_for_thumb_start(&self, start_eighths: usize) -> usize {
        let travel = self.travel_eighths();
        if travel == 0 {
            return 0;
        }
        let position = (start_eighths.min(travel) * self.max_scroll() + travel / 2) / travel;
        position.min(self.max_scroll())
    }

    pub(crate) fn cell(&self, row: usize, theme: Theme, backdrop: Color) -> (char, Style) {
        let cell_start = row * EIGHTHS;
        let cell_end = cell_start + EIGHTHS;
        let start = self.thumb_start_eighths().clamp(cell_start, cell_end) - cell_start;
        let end = self.thumb_end_eighths().clamp(cell_start, cell_end) - cell_start;
        if end <= start {
            return (
                TRACK_SYMBOL,
                Style::default().fg(theme.border).bg(theme.root_bg),
            );
        }
        let thumb = Style::default().fg(theme.muted_text).bg(backdrop);
        if start == 0 && end == EIGHTHS {
            return (FULL_BLOCK, thumb);
        }
        if start == 0 {
            return (
                LOWER_BLOCKS[EIGHTHS - end - 1],
                Style::default().fg(backdrop).bg(theme.muted_text),
            );
        }
        if end == EIGHTHS {
            return (LOWER_BLOCKS[EIGHTHS - start - 1], thumb);
        }
        (FULL_BLOCK, thumb)
    }

    fn row_centre_eighths(&self, track_row: usize) -> usize {
        track_row * EIGHTHS + EIGHTHS / 2
    }

    fn max_scroll(&self) -> usize {
        self.total_rows - self.viewport_rows
    }

    fn track_eighths(&self) -> usize {
        self.track_rows * EIGHTHS
    }

    fn thumb_length_eighths(&self) -> usize {
        (self.track_eighths() * self.viewport_rows / self.total_rows)
            .clamp(EIGHTHS, self.track_eighths())
    }

    fn travel_eighths(&self) -> usize {
        self.track_eighths() - self.thumb_length_eighths()
    }
}

#[cfg(test)]
mod tests {
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
}
