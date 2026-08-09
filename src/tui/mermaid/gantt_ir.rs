use super::MermaidSourceSpan;

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct Label {
    pub(super) text: String,
    pub(super) span: MermaidSourceSpan,
}
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct Config {
    pub(super) key: &'static str,
    pub(super) value: Label,
}
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct Task {
    pub(super) name: Label,
    pub(super) status: Option<Label>,
    pub(super) id: Option<Label>,
    pub(super) timing: Label,
}
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) enum Item {
    Config(Config),
    Section(Label),
    Task(Task),
}
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct Diagram {
    pub(super) items: Vec<Item>,
}
