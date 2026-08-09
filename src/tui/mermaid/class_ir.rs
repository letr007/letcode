use super::MermaidSourceSpan;

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct Label {
    pub(super) text: String,
    pub(super) span: MermaidSourceSpan,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct Class {
    pub(super) name: Label,
    pub(super) members: Vec<Label>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct Relation {
    pub(super) from: Label,
    pub(super) to: Label,
    pub(super) label: Option<Label>,
    pub(super) connector: &'static str,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct Diagram {
    pub(super) classes: Vec<Class>,
    pub(super) relations: Vec<Relation>,
}
