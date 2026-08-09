//! Generic Mermaid display canvas primitives.

use super::MermaidSourceSpan;

/// A source-backed label placed on a display grid.
pub(crate) struct MermaidCanvasLabel {
    pub(crate) row: usize,
    pub(crate) col: usize,
    pub(crate) text: String,
    pub(crate) source: MermaidSourceSpan,
}

/// One terminal display column.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum MermaidCell {
    Empty,
    Char(char),
    Wide,
}

/// A display grid plus source-backed label positions.
pub(crate) struct MermaidCanvas {
    pub(crate) rows: Vec<Vec<MermaidCell>>,
    pub(crate) labels: Vec<MermaidCanvasLabel>,
}

fn mermaid_char_width(ch: char) -> usize {
    if (ch as u32) > 0x2E7F { 2 } else { 1 }
}

impl MermaidCanvas {
    pub(crate) fn ensure_row(&mut self, row: usize, cols: usize) {
        while self.rows.len() <= row {
            self.rows.push(Vec::new());
        }
        let line = self.rows.get_mut(row).unwrap();
        if line.len() < cols {
            line.resize(cols, MermaidCell::Empty);
        }
    }

    pub(crate) fn blit(&mut self, row: usize, col: usize, text: &str) {
        for (i, line) in text.lines().enumerate() {
            let target = row + i;
            let mut c = col;
            for ch in line.chars() {
                let w = mermaid_char_width(ch);
                self.put(c, target, ch);
                c += w;
            }
        }
    }

    pub(crate) fn put(&mut self, col: usize, row: usize, ch: char) {
        self.ensure_row(row, col + mermaid_char_width(ch));
        let line = self.rows.get_mut(row).unwrap();
        if col < line.len() && line[col] == MermaidCell::Wide {
            return;
        }
        line[col] = MermaidCell::Char(ch);
        if mermaid_char_width(ch) == 2 && col + 1 < line.len() {
            line[col + 1] = MermaidCell::Wide;
        }
    }

    #[cfg(test)]
    pub(crate) fn to_rows(&self) -> Vec<String> {
        self.rows
            .iter()
            .map(|cells| {
                let mut s = String::new();
                for cell in cells {
                    match cell {
                        MermaidCell::Empty => s.push(' '),
                        MermaidCell::Wide => {}
                        MermaidCell::Char(ch) => s.push(*ch),
                    }
                }
                s
            })
            .collect()
    }
}
