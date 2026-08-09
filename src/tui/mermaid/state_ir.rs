use super::MermaidSourceSpan;

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct Label {
    pub(super) text: String,
    pub(super) span: MermaidSourceSpan,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct State {
    pub(super) label: Label,
    pub(super) composite: bool,
    pub(super) depth: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct Transition {
    pub(super) from: Label,
    pub(super) to: Label,
    pub(super) label: Option<Label>,
    pub(super) depth: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) enum Item {
    State(State),
    Transition(Transition),
    Close(usize),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct Diagram {
    pub(super) items: Vec<Item>,
}
