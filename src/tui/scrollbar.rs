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
#[path = "../../tests/internal/tui/scrollbar/tests.rs"]
mod tests;
