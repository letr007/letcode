use std::collections::VecDeque;

/// 一个点列对应一个速率采样；与 footer 的 10 格宽（20 列）一致。
pub const OUTPUT_RATE_GRAPH_COLUMNS: usize = 20;

const MAX_LEVEL: usize = 4;

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct OutputRateGraph {
    samples: VecDeque<Option<u64>>,
    view_id: Option<String>,
}

impl OutputRateGraph {
    pub fn push(&mut self, view_id: Option<&str>, rate: Option<u64>) {
        if self.view_id.as_deref() != view_id {
            *self = Self {
                view_id: view_id.map(str::to_owned),
                ..Self::default()
            };
        }
        self.samples.push_back(rate);
        while self.samples.len() > OUTPUT_RATE_GRAPH_COLUMNS {
            self.samples.pop_front();
        }
    }

    pub fn clear(&mut self) {
        *self = Self::default();
    }

    pub fn latest_sample(&self) -> Option<Option<u64>> {
        self.samples.back().copied()
    }

    pub fn has_rate(&self) -> bool {
        self.samples.iter().any(Option::is_some)
    }

    pub fn levels(&self, columns: usize) -> Vec<usize> {
        let low = self.samples.iter().flatten().copied().min().unwrap_or(0);
        let high = self.samples.iter().flatten().copied().max().unwrap_or(0);
        let mut levels = vec![1; columns];
        let offset = columns.saturating_sub(self.samples.len());
        for (index, rate) in self.samples.iter().enumerate() {
            if let Some(level) = levels.get_mut(offset + index) {
                *level = rate.map_or(1, |rate| level_for(rate, low, high));
            }
        }
        levels
    }
}

fn level_for(rate: u64, low: u64, high: u64) -> usize {
    if high <= low {
        return 1;
    }
    let position = rate.saturating_sub(low) as f64 / (high - low) as f64;
    1 + (position.clamp(0.0, 1.0) * (MAX_LEVEL - 1) as f64).round() as usize
}

#[cfg(test)]
#[path = "../../../tests/internal/tui/state/output_rate_graph/tests.rs"]
mod tests;
