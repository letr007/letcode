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
    /// `None` 表示该秒没有活跃输出：只画底部基线，且不参与自适应量程。
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

    /// 最新一列：外层 `None` 表示还没有任何采样；内层 `None` 表示静默。
    pub fn latest_sample(&self) -> Option<Option<u64>> {
        self.samples.back().copied()
    }

    /// 窗口内是否出现过真实速率；用于决定速率区域是否显示。
    pub fn has_rate(&self) -> bool {
        self.samples.iter().any(Option::is_some)
    }

    /// 每个点列的高度档位（1..=4）。空闲列为底部一点，样本不足时左侧同样补底部一点。
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
mod tests {
    use super::*;

    #[test]
    fn window_keeps_only_the_most_recent_samples() {
        let mut graph = OutputRateGraph::default();
        for rate in 1..=25 {
            graph.push(None, Some(rate));
        }

        assert_eq!(graph.samples.len(), OUTPUT_RATE_GRAPH_COLUMNS);
        assert_eq!(graph.samples.front(), Some(&Some(6)));
        assert_eq!(graph.samples.back(), Some(&Some(25)));
    }

    #[test]
    fn levels_span_the_active_window_range() {
        let mut graph = OutputRateGraph::default();
        for rate in [0, 40, 80] {
            graph.push(None, Some(rate));
        }

        assert_eq!(graph.levels(4), vec![1, 1, 3, 4]);
    }

    #[test]
    fn idle_samples_render_baseline_without_widening_the_scale() {
        let mut graph = OutputRateGraph::default();
        for rate in [None, Some(600), Some(604), None] {
            graph.push(None, rate);
        }

        assert_eq!(graph.levels(4), vec![1, 1, 4, 1]);
    }

    #[test]
    fn latest_sample_reports_the_newest_column() {
        let mut graph = OutputRateGraph::default();

        assert_eq!(graph.latest_sample(), None);

        graph.push(None, Some(60));
        assert_eq!(graph.latest_sample(), Some(Some(60)));

        graph.push(None, None);
        assert_eq!(graph.latest_sample(), Some(None));
    }

    #[test]
    fn flat_window_renders_the_bottom_level() {
        let mut graph = OutputRateGraph::default();
        for _ in 0..5 {
            graph.push(None, Some(0));
        }

        assert_eq!(graph.levels(3), vec![1, 1, 1]);
    }

    #[test]
    fn view_change_clears_history() {
        let mut graph = OutputRateGraph::default();
        graph.push(None, Some(60));
        graph.push(Some("child"), Some(60));

        assert_eq!(graph.samples.len(), 1);
    }
}
