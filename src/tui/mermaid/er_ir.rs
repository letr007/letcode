use super::MermaidSourceSpan;

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct Label {
    pub(super) text: String,
    pub(super) span: MermaidSourceSpan,
}
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct Entity {
    pub(super) name: Label,
    pub(super) attributes: Vec<Label>,
}
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct Relation {
    pub(super) from: Label,
    pub(super) from_cardinality: Label,
    pub(super) connector: &'static str,
    pub(super) to_cardinality: Label,
    pub(super) to: Label,
    pub(super) label: Label,
}
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct Diagram {
    pub(super) entities: Vec<Entity>,
    pub(super) relations: Vec<Relation>,
}
