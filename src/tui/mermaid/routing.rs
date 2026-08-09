//! Shared orthogonal routing primitives for Mermaid diagrams.

use std::{
    collections::HashMap,
    ops::{Deref, DerefMut},
};

pub(crate) const ROUTE_UP: u8 = 1;
pub(crate) const ROUTE_RIGHT: u8 = 2;
pub(crate) const ROUTE_DOWN: u8 = 4;
pub(crate) const ROUTE_LEFT: u8 = 8;

#[derive(Debug, Default)]
pub(crate) struct RouteGrid {
    cells: HashMap<(usize, usize), u8>,
}

impl Deref for RouteGrid {
    type Target = HashMap<(usize, usize), u8>;

    fn deref(&self) -> &Self::Target {
        &self.cells
    }
}

impl DerefMut for RouteGrid {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.cells
    }
}

impl RouteGrid {
    pub(crate) fn new() -> Self {
        Self::default()
    }

    pub(crate) fn connect(&mut self, from: (usize, usize), to: (usize, usize)) {
        if from == to {
            return;
        }
        debug_assert!(from.0 == to.0 || from.1 == to.1);
        if from.1 == to.1 {
            let (lo, hi) = if from.0 < to.0 {
                (from.0, to.0)
            } else {
                (to.0, from.0)
            };
            for col in lo..hi {
                *self.cells.entry((col, from.1)).or_default() |= ROUTE_RIGHT;
                *self.cells.entry((col + 1, from.1)).or_default() |= ROUTE_LEFT;
            }
        } else {
            let (lo, hi) = if from.1 < to.1 {
                (from.1, to.1)
            } else {
                (to.1, from.1)
            };
            for row in lo..hi {
                *self.cells.entry((from.0, row)).or_default() |= ROUTE_DOWN;
                *self.cells.entry((from.0, row + 1)).or_default() |= ROUTE_UP;
            }
        }
    }
}

#[derive(Debug, Default)]
pub(crate) struct TrackAllocator {
    used: Vec<usize>,
}

impl TrackAllocator {
    pub(crate) fn allocate(&mut self, preferred: usize, min: usize, max: usize) -> Option<usize> {
        self.reserve(preferred, min, max, 1)
    }

    pub(crate) fn reserve(
        &mut self,
        preferred: usize,
        min: usize,
        max: usize,
        len: usize,
    ) -> Option<usize> {
        if len == 0 || min > max || len - 1 > max - min {
            return None;
        }
        let end = max - (len - 1);
        let preferred = preferred.clamp(min, end);
        for distance in 0..=end - min {
            let candidates = if distance == 0 {
                [Some(preferred), None]
            } else {
                [
                    preferred
                        .checked_sub(distance)
                        .filter(|value| *value >= min),
                    preferred
                        .checked_add(distance)
                        .filter(|value| *value <= end),
                ]
            };
            for start in candidates.into_iter().flatten() {
                let span = start..start + len;
                if span.clone().all(|value| !self.used.contains(&value)) {
                    self.used.extend(span);
                    return Some(start);
                }
            }
        }
        None
    }
}

pub(crate) fn route_glyph(mask: u8) -> char {
    match mask {
        1 | 4 | 5 => '│',
        2 | 8 | 10 => '─',
        6 => '┌',
        12 => '┐',
        3 => '└',
        9 => '┘',
        7 => '├',
        13 => '┤',
        14 => '┬',
        11 => '┴',
        15 => '┼',
        _ => '┼',
    }
}

#[cfg(test)]
mod tests {
    use super::TrackAllocator;

    #[test]
    fn reserve_is_atomic_for_contiguous_spans() {
        let mut allocator = TrackAllocator::default();
        assert_eq!(allocator.reserve(2, 0, 3, 2), Some(2));
        assert_eq!(allocator.reserve(1, 0, 3, 2), Some(0));
        assert_eq!(allocator.reserve(1, 0, 3, 2), None);
        assert_eq!(allocator.allocate(3, 0, 3), None);
    }
}
