use super::MermaidSourceSpan;

#[derive(Debug, Clone, PartialEq)]
pub(super) struct Label {
    pub(super) text: String,
    pub(super) span: MermaidSourceSpan,
}

#[derive(Debug, Clone, PartialEq)]
pub(super) struct Axis {
    pub(super) left: Label,
    pub(super) right: Label,
}

#[derive(Debug, Clone, PartialEq)]
pub(super) struct Point {
    pub(super) label: Label,
    pub(super) x: f64,
    pub(super) y: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub(super) struct Diagram {
    pub(super) title: Option<Label>,
    pub(super) x_axis: Axis,
    pub(super) y_axis: Axis,
    pub(super) quadrants: [Label; 4],
    pub(super) points: Vec<Point>,
}
