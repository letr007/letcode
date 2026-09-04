use super::MermaidSourceSpan;

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct Label {
    pub(super) text: String,
    pub(super) span: MermaidSourceSpan,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct Task {
    pub(super) label: Label,
    pub(super) score: Label,
    pub(super) participants: Vec<Label>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct Section {
    pub(super) label: Label,
    pub(super) tasks: Vec<Task>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct Diagram {
    pub(super) title: Option<Label>,
    pub(super) sections: Vec<Section>,
}
