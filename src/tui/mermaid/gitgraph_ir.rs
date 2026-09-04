use super::MermaidSourceSpan;

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct Label {
    pub(super) text: String,
    pub(super) span: MermaidSourceSpan,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct Branch {
    pub(super) name: String,
    pub(super) span: Option<MermaidSourceSpan>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct Commit {
    pub(super) id: Option<Label>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) enum Item {
    Commit(Commit),
    Branch(Label),
    Checkout(Label),
    Merge(Label),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct Diagram {
    pub(super) branches: Vec<Branch>,
    pub(super) items: Vec<Item>,
}
