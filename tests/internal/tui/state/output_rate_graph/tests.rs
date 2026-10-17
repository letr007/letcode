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
