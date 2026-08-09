//! Sequence-specific Mermaid intermediate representation.

use std::collections::HashMap;

#[derive(Debug, Clone)]
pub(crate) struct MermaidNode {
    pub(crate) label: String,
    pub(crate) start: usize,
    pub(crate) end: usize,
}

#[derive(Debug, Clone)]
pub(crate) struct MermaidLabel {
    pub(crate) text: String,
    pub(crate) start: usize,
    pub(crate) end: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum MermaidBlockKind {
    Loop,
    Alt,
    Opt,
}

impl MermaidBlockKind {
    pub(crate) const fn keyword(self) -> &'static str {
        match self {
            Self::Loop => "loop",
            Self::Alt => "alt",
            Self::Opt => "opt",
        }
    }
}

#[derive(Debug)]
pub(crate) struct MermaidSequence {
    pub(crate) participants: HashMap<String, MermaidNode>,
    pub(crate) items: Vec<MermaidSequenceItem>,
}

#[derive(Debug)]
pub(crate) enum MermaidSequenceItem {
    Message(MermaidMessage),
    Block(MermaidBlock),
}

#[derive(Debug)]
pub(crate) struct MermaidBlock {
    pub(crate) kind: MermaidBlockKind,
    pub(crate) label: MermaidLabel,
    pub(crate) branches: Vec<MermaidBranch>,
}

#[derive(Debug)]
pub(crate) struct MermaidBranch {
    pub(crate) label: Option<MermaidLabel>,
    pub(crate) items: Vec<MermaidSequenceItem>,
}

#[derive(Debug)]
pub(crate) struct MermaidMessage {
    pub(crate) from: String,
    pub(crate) to: String,
    pub(crate) label: MermaidLabel,
    pub(crate) dashed: bool,
}
