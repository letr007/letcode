use super::MermaidSourceSpan;

#[derive(Debug, Clone, PartialEq)]
pub(super) struct Label {
    pub(super) text: String,
    pub(super) span: MermaidSourceSpan,
}

#[derive(Debug, Clone, PartialEq)]
pub(super) struct Slice {
    pub(super) label: Label,
    pub(super) value: f64,
    pub(super) raw_value: Label,
}

#[derive(Debug, Clone, PartialEq)]
pub(super) struct Diagram {
    pub(super) title: Option<Label>,
    pub(super) show_data: bool,
    pub(super) slices: Vec<Slice>,
}
