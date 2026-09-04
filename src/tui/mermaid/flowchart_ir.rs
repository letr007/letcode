//! Flowchart-specific Mermaid intermediate representation.

use std::collections::HashMap;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum MermaidDirection {
    Td,
    Bu,
    Lr,
    Rl,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum MermaidEdgeStyle {
    Solid,
    Dashed,
    Thick,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum MermaidShape {
    Rectangle,
    RoundRect,
    Diamond,
    Hexagon,
    Cylinder,
    Circle,
    Subroutine,
    Parallelogram,
    Stadium,
}

#[derive(Debug)]
pub(crate) struct MermaidGraph {
    pub(crate) direction: MermaidDirection,
    pub(crate) nodes: HashMap<String, MermaidNode>,
    pub(crate) edges: Vec<MermaidEdge>,
}

#[derive(Debug, Clone)]
pub(crate) struct MermaidNode {
    pub(crate) label: String,
    pub(crate) start: usize,
    pub(crate) end: usize,
    pub(crate) atomic: bool,
    pub(crate) shape: MermaidShape,
}

#[derive(Debug, Clone)]
pub(crate) struct MermaidLabel {
    pub(crate) text: String,
    pub(crate) start: usize,
    pub(crate) end: usize,
    pub(crate) atomic: bool,
}

#[derive(Debug)]
pub(crate) struct MermaidEdge {
    pub(crate) from: String,
    pub(crate) to: String,
    pub(crate) label: Option<MermaidLabel>,
    pub(crate) style: MermaidEdgeStyle,
    pub(crate) arrow: bool,
    pub(crate) reverse_arrow: bool,
}
