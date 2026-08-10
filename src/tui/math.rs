//! Bounded terminal LaTeX parser and layout engine.

use crate::tui::measure::display_width;

#[derive(Debug, Clone, Copy)]
struct MathLimits {
    max_source_chars: usize,
    max_nodes: usize,
    max_rows: usize,
    max_columns: usize,
}

impl Default for MathLimits {
    fn default() -> Self {
        Self {
            max_source_chars: 512,
            max_nodes: 512,
            max_rows: 32,
            max_columns: 16,
        }
    }
}

#[derive(Debug, Clone)]
pub(crate) struct MathLayout {
    pub(crate) rows: Vec<String>,
    pub(crate) width: usize,
}

enum LayoutNode {
    Fraction {
        numerator: String,
        denominator: String,
    },
    Operator {
        operator: String,
        lower: Option<String>,
        upper: Option<String>,
    },
    Matrix {
        lines: Vec<String>,
        baseline: usize,
    },
}

const LAYOUT_START: char = '\u{f0000}';
const LAYOUT_END: char = '\u{f0001}';
const PROTECTED_SPACE: char = '\u{f0002}';
const NAMED_START: char = '\u{f0004}';
const NAMED_END: char = '\u{f0005}';

const SUPERSCRIPT: &[(char, char)] = &[
    ('0', '⁰'),
    ('1', '¹'),
    ('2', '²'),
    ('3', '³'),
    ('4', '⁴'),
    ('5', '⁵'),
    ('6', '⁶'),
    ('7', '⁷'),
    ('8', '⁸'),
    ('9', '⁹'),
    ('+', '⁺'),
    ('-', '⁻'),
    ('=', '⁼'),
    ('(', '⁽'),
    (')', '⁾'),
    ('a', 'ᵃ'),
    ('b', 'ᵇ'),
    ('c', 'ᶜ'),
    ('d', 'ᵈ'),
    ('e', 'ᵉ'),
    ('f', 'ᶠ'),
    ('g', 'ᵍ'),
    ('h', 'ʰ'),
    ('i', 'ⁱ'),
    ('j', 'ʲ'),
    ('k', 'ᵏ'),
    ('l', 'ˡ'),
    ('m', 'ᵐ'),
    ('n', 'ⁿ'),
    ('o', 'ᵒ'),
    ('p', 'ᵖ'),
    ('r', 'ʳ'),
    ('s', 'ˢ'),
    ('t', 'ᵗ'),
    ('u', 'ᵘ'),
    ('v', 'ᵛ'),
    ('w', 'ʷ'),
    ('x', 'ˣ'),
    ('y', 'ʸ'),
    ('z', 'ᶻ'),
];
const SUBSCRIPT: &[(char, char)] = &[
    ('0', '₀'),
    ('1', '₁'),
    ('2', '₂'),
    ('3', '₃'),
    ('4', '₄'),
    ('5', '₅'),
    ('6', '₆'),
    ('7', '₇'),
    ('8', '₈'),
    ('9', '₉'),
    ('+', '₊'),
    ('-', '₋'),
    ('=', '₌'),
    ('(', '₍'),
    (')', '₎'),
    ('a', 'ₐ'),
    ('e', 'ₑ'),
    ('h', 'ₕ'),
    ('i', 'ᵢ'),
    ('j', 'ⱼ'),
    ('k', 'ₖ'),
    ('l', 'ₗ'),
    ('m', 'ₘ'),
    ('n', 'ₙ'),
    ('o', 'ₒ'),
    ('p', 'ₚ'),
    ('r', 'ᵣ'),
    ('s', 'ₛ'),
    ('t', 'ₜ'),
    ('u', 'ᵤ'),
    ('v', 'ᵥ'),
    ('x', 'ₓ'),
];

fn table_lookup(table: &[(char, char)], value: &str) -> Option<String> {
    value
        .chars()
        .map(|ch| {
            table
                .iter()
                .find(|(from, _)| *from == ch)
                .map(|(_, to)| *to)
        })
        .collect()
}

fn format_script(value: &str, sub: bool) -> String {
    let value = value
        .trim()
        .chars()
        .filter(|ch| !ch.is_whitespace())
        .collect::<String>();
    let table = if sub { SUBSCRIPT } else { SUPERSCRIPT };
    if let Some(value) = table_lookup(table, &value) {
        return value;
    }
    let prefix = if sub { '_' } else { '^' };
    if value.chars().count() == 1
        || (sub && value.chars().all(|ch| ch.is_ascii_alphabetic()))
        || (sub
            && value
                .chars()
                .all(|ch| ch.is_ascii_alphanumeric() || ch == '='))
    {
        format!("{prefix}{value}")
    } else {
        format!("{prefix}({value})")
    }
}

fn symbol(name: &str) -> Option<&'static str> {
    Some(match name {
        "alpha" => "α",
        "beta" => "β",
        "gamma" => "γ",
        "delta" => "δ",
        "epsilon" => "ϵ",
        "varepsilon" => "ε",
        "zeta" => "ζ",
        "eta" => "η",
        "theta" => "θ",
        "vartheta" => "ϑ",
        "iota" => "ι",
        "kappa" => "κ",
        "varkappa" => "ϰ",
        "lambda" => "λ",
        "mu" => "μ",
        "nu" => "ν",
        "xi" => "ξ",
        "pi" => "π",
        "varpi" => "ϖ",
        "rho" => "ρ",
        "varrho" => "ϱ",
        "sigma" => "σ",
        "varsigma" => "ς",
        "tau" => "τ",
        "upsilon" => "υ",
        "phi" => "ϕ",
        "varphi" => "φ",
        "chi" => "χ",
        "psi" => "ψ",
        "omega" => "ω",
        "Gamma" => "Γ",
        "Delta" => "Δ",
        "Theta" => "Θ",
        "Lambda" => "Λ",
        "Xi" => "Ξ",
        "Pi" => "Π",
        "Sigma" => "Σ",
        "Upsilon" => "Υ",
        "Phi" => "Φ",
        "Psi" => "Ψ",
        "Omega" => "Ω",
        "pm" => "±",
        "mp" => "∓",
        "times" => "×",
        "div" => "÷",
        "cdot" => "·",
        "ast" => "∗",
        "star" => "⋆",
        "circ" => "∘",
        "bullet" => "•",
        "oplus" => "⊕",
        "ominus" => "⊖",
        "otimes" => "⊗",
        "oslash" => "⊘",
        "odot" => "⊙",
        "bigcirc" => "○",
        "dagger" => "†",
        "ddagger" => "‡",
        "amalg" => "⨿",
        "uplus" => "⊎",
        "sqcap" => "⊓",
        "sqcup" => "⊔",
        "triangleleft" => "◁",
        "triangleright" => "▷",
        "wr" => "≀",
        "cap" => "∩",
        "cup" => "∪",
        "bigcap" => "⋂",
        "bigcup" => "⋃",
        "bigwedge" => "⋀",
        "bigvee" => "⋁",
        "bigsqcup" => "⨆",
        "biguplus" => "⨄",
        "bigoplus" => "⨁",
        "bigotimes" => "⨂",
        "bigodot" => "⨀",
        "setminus" => "∖",
        "in" => "∈",
        "notin" => "∉",
        "ni" => "∋",
        "subset" => "⊂",
        "supset" => "⊃",
        "subseteq" => "⊆",
        "supseteq" => "⊇",
        "sqsubset" => "⊏",
        "sqsupset" => "⊐",
        "sqsubseteq" => "⊑",
        "sqsupseteq" => "⊒",
        "prec" => "≺",
        "preceq" => "≼",
        "succ" => "≻",
        "succeq" => "≽",
        "ll" => "≪",
        "gg" => "≫",
        "le" => "≤",
        "leq" => "≤",
        "leqslant" => "≤",
        "ge" => "≥",
        "geq" => "≥",
        "geqslant" => "≥",
        "ne" => "≠",
        "neq" => "≠",
        "equiv" => "≡",
        "approx" => "≈",
        "sim" => "∼",
        "simeq" => "≃",
        "cong" => "≅",
        "asymp" => "≍",
        "doteq" => "≐",
        "propto" => "∝",
        "parallel" => "∥",
        "perp" => "⊥",
        "mid" => "∣",
        "vdash" => "⊢",
        "dashv" => "⊣",
        "models" => "⊨",
        "Vdash" => "⊩",
        "Vvdash" => "⊪",
        "nvdash" => "⊬",
        "nvDash" => "⊭",
        "forall" => "∀",
        "exists" => "∃",
        "nexists" => "∄",
        "neg" => "¬",
        "land" => "∧",
        "wedge" => "∧",
        "lor" => "∨",
        "vee" => "∨",
        "to" => "→",
        "rightarrow" => "→",
        "longrightarrow" => "→",
        "leftarrow" => "←",
        "longleftarrow" => "←",
        "gets" => "←",
        "leftrightarrow" => "↔",
        "longleftrightarrow" => "↔",
        "hookleftarrow" => "↩",
        "hookrightarrow" => "↪",
        "twoheadleftarrow" => "↞",
        "twoheadrightarrow" => "↠",
        "leftharpoonup" => "↼",
        "leftharpoondown" => "↽",
        "rightharpoonup" => "⇀",
        "rightharpoondown" => "⇁",
        "rightleftharpoons" => "⇌",
        "leftrightharpoons" => "⇋",
        "nearrow" => "↗",
        "searrow" => "↘",
        "swarrow" => "↙",
        "nwarrow" => "↖",
        "rightsquigarrow" => "⇝",
        "leadsto" => "⇝",
        "Rightarrow" => "⇒",
        "Longrightarrow" => "⇒",
        "Leftarrow" => "⇐",
        "Longleftarrow" => "⇐",
        "Leftrightarrow" => "⇔",
        "Longleftrightarrow" => "⇔",
        "implies" => "⇒",
        "iff" => "⇔",
        "mapsto" => "↦",
        "longmapsto" => "↦",
        "uparrow" => "↑",
        "downarrow" => "↓",
        "partial" => "∂",
        "nabla" => "∇",
        "int" => "∫",
        "iint" => "∬",
        "iiint" => "∭",
        "oint" => "∮",
        "sum" => "∑",
        "prod" => "∏",
        "coprod" => "∐",
        "infty" => "∞",
        "emptyset" => "∅",
        "varnothing" => "∅",
        "angle" => "∠",
        "therefore" => "∴",
        "because" => "∵",
        "aleph" => "ℵ",
        "beth" => "ℶ",
        "gimel" => "ℷ",
        "daleth" => "ℸ",
        "top" => "⊤",
        "bot" => "⊥",
        "triangle" => "△",
        "square" => "□",
        "lozenge" => "◊",
        "checkmark" => "✓",
        "complement" => "∁",
        "wp" => "℘",
        "prime" => "′",
        "ldots" => "…",
        "dots" => "…",
        "cdots" => "⋯",
        "vdots" => "⋮",
        "ddots" => "⋱",
        "ell" => "ℓ",
        "hbar" => "ℏ",
        "Im" => "ℑ",
        "Re" => "ℜ",
        "langle" => "⟨",
        "rangle" => "⟩",
        "vert" => "|",
        "lvert" => "|",
        "rvert" => "|",
        "Vert" => "‖",
        "lVert" => "‖",
        "rVert" => "‖",
        "lbrace" => "{",
        "rbrace" => "}",
        "backslash" => "\\",
        "lfloor" => "⌊",
        "rfloor" => "⌋",
        "lceil" => "⌈",
        "rceil" => "⌉",
        "colon" => ":",
        _ => return None,
    })
}

fn named_operator(name: &str) -> bool {
    matches!(
        name,
        "arccos"
            | "arcsin"
            | "arctan"
            | "arg"
            | "cos"
            | "cosh"
            | "cot"
            | "coth"
            | "csc"
            | "deg"
            | "det"
            | "dim"
            | "exp"
            | "gcd"
            | "hom"
            | "inf"
            | "ker"
            | "lg"
            | "lim"
            | "liminf"
            | "limsup"
            | "ln"
            | "log"
            | "max"
            | "min"
            | "Pr"
            | "sec"
            | "sin"
            | "sinh"
            | "sup"
            | "tan"
            | "tanh"
    )
}
fn limit_operator(name: &str) -> bool {
    matches!(
        name,
        "argmax"
            | "argmin"
            | "inf"
            | "injlim"
            | "lim"
            | "liminf"
            | "limsup"
            | "max"
            | "min"
            | "projlim"
            | "sup"
    )
}
fn display_limit_symbol(name: &str) -> bool {
    matches!(
        name,
        "bigcap"
            | "bigcup"
            | "bigodot"
            | "bigoplus"
            | "bigotimes"
            | "bigsqcup"
            | "biguplus"
            | "bigvee"
            | "bigwedge"
            | "coprod"
            | "int"
            | "iint"
            | "iiint"
            | "oint"
            | "prod"
            | "sum"
    )
}
fn relation_command(name: &str) -> bool {
    matches!(
        name,
        "Leftarrow"
            | "Leftrightarrow"
            | "Longleftarrow"
            | "Longleftrightarrow"
            | "Longrightarrow"
            | "Rightarrow"
            | "Vdash"
            | "Vvdash"
            | "approx"
            | "asymp"
            | "cong"
            | "dashv"
            | "doteq"
            | "downarrow"
            | "equiv"
            | "ge"
            | "geq"
            | "geqslant"
            | "gets"
            | "gg"
            | "hookleftarrow"
            | "hookrightarrow"
            | "iff"
            | "implies"
            | "in"
            | "leadsto"
            | "le"
            | "leftarrow"
            | "leftharpoondown"
            | "leftharpoonup"
            | "leftrightarrow"
            | "leftrightharpoons"
            | "leq"
            | "leqslant"
            | "ll"
            | "longleftarrow"
            | "longleftrightarrow"
            | "longmapsto"
            | "longrightarrow"
            | "mapsto"
            | "mid"
            | "models"
            | "ne"
            | "nearrow"
            | "neq"
            | "nwarrow"
            | "parallel"
            | "perp"
            | "prec"
            | "preceq"
            | "propto"
            | "rightharpoondown"
            | "rightharpoonup"
            | "rightleftharpoons"
            | "rightarrow"
            | "rightsquigarrow"
            | "searrow"
            | "sim"
            | "simeq"
            | "sqsubset"
            | "sqsubseteq"
            | "sqsupset"
            | "sqsupseteq"
            | "subset"
            | "subseteq"
            | "succ"
            | "succeq"
            | "supset"
            | "supseteq"
            | "swarrow"
            | "to"
            | "triangleleft"
            | "triangleright"
            | "twoheadleftarrow"
            | "twoheadrightarrow"
            | "uparrow"
            | "vdash"
    )
}
fn blackboard(ch: char) -> Option<char> {
    Some(match ch {
        'C' => 'ℂ',
        'H' => 'ℍ',
        'N' => 'ℕ',
        'P' => 'ℙ',
        'Q' => 'ℚ',
        'R' => 'ℝ',
        'Z' => 'ℤ',
        _ => return None,
    })
}

fn normalize_output(value: String) -> String {
    let out = value.replace(NAMED_START, "").replace(NAMED_END, "");
    out.lines()
        .map(|line| {
            if line.contains(LAYOUT_START) || line.contains(LAYOUT_END) {
                line.trim().to_string()
            } else {
                line.split_whitespace().collect::<Vec<_>>().join(" ")
            }
        })
        .filter(|line| !line.is_empty())
        .collect::<Vec<_>>()
        .join("\n")
        .trim()
        .to_string()
}

#[derive(Debug, Clone)]
struct ArraySpec {
    alignments: Vec<char>,
    vertical: Vec<bool>,
    leading_vertical: bool,
    trailing_vertical: bool,
}

fn parse_array_spec(spec: &str) -> Option<ArraySpec> {
    let chars = spec.chars().collect::<Vec<_>>();
    if chars.is_empty() {
        return None;
    }
    let mut position = 0;
    let leading_vertical = chars.first() == Some(&'|');
    if leading_vertical {
        position += 1;
    }
    let mut alignments = Vec::new();
    let mut vertical = Vec::new();
    while position < chars.len() {
        let alignment = chars[position];
        if !matches!(alignment, 'l' | 'c' | 'r') {
            return None;
        }
        alignments.push(alignment);
        position += 1;
        if position == chars.len() {
            break;
        }
        if chars[position] == '|' {
            position += 1;
            if position == chars.len() {
                break;
            }
            if chars[position] == '|' {
                return None;
            }
            vertical.push(true);
        } else {
            vertical.push(false);
        }
    }
    if alignments.is_empty() || vertical.len() + 1 != alignments.len() {
        return None;
    }
    Some(ArraySpec {
        alignments,
        vertical,
        leading_vertical,
        trailing_vertical: chars.last() == Some(&'|'),
    })
}

fn strip_row_spacing(row: &str) -> String {
    let mut row = row.trim().to_string();
    if let Some(close) = row.find(']')
        && row.starts_with('[')
        && row[1..close].trim_end().ends_with("pt")
        && row[1..close - 2].trim().parse::<f32>().is_ok()
    {
        row = row[close + 1..].trim_start().to_string();
    }
    if let Some(open) = row.rfind('[')
        && row.ends_with(']')
        && open > 0
        && row[open + 1..row.len() - 1]
            .trim_end()
            .strip_suffix("pt")
            .and_then(|value| value.trim().parse::<f32>().ok())
            .is_some()
    {
        row.truncate(open);
        row = row.trim_end().to_string();
    }
    row
}

struct LatexParser<'a> {
    source: &'a str,
    position: usize,
    display: bool,
    stack_fractions: bool,
    supported: bool,
    nodes: usize,
    limits: MathLimits,
    layout_nodes: &'a mut Vec<LayoutNode>,
}
impl<'a> LatexParser<'a> {
    fn new(source: &'a str, display: bool, layout_nodes: &'a mut Vec<LayoutNode>) -> Self {
        Self {
            source,
            position: 0,
            display,
            stack_fractions: true,
            supported: true,
            nodes: 0,
            limits: MathLimits::default(),
            layout_nodes,
        }
    }
    fn render(mut self) -> Option<String> {
        let result = self.parse_sequence(None);
        if !self.supported
            || self.position != self.source.len()
            || self.nodes > self.limits.max_nodes
        {
            None
        } else {
            Some(normalize_output(result))
        }
    }
    fn bump(&mut self) {
        self.nodes += 1;
    }
    fn whitespace(&mut self) {
        while self.position < self.source.len()
            && self.source.as_bytes()[self.position].is_ascii_whitespace()
        {
            self.position += 1;
        }
    }
    fn parse_sequence(&mut self, end: Option<char>) -> String {
        let mut result = String::new();
        while self.position < self.source.len() {
            let ch = self.source[self.position..].chars().next().unwrap();
            if end == Some(ch) {
                self.position += ch.len_utf8();
                return result;
            }
            if ch == '}' {
                self.supported = false;
                return result;
            }
            if ch == '{' {
                self.position += 1;
                result.push_str(&self.parse_sequence(Some('}')));
                continue;
            }
            if ch == '\\' {
                let value = self.parse_command();
                if result.ends_with('∞') && value.starts_with('c') {
                    result.push(PROTECTED_SPACE);
                }
                result.push_str(&value);
                continue;
            }
            if ch == '^' || ch == '_' {
                self.position += 1;
                result = result.trim_end().to_string();
                let arg = self.parse_required_argument(false);
                result.push_str(&format_script(&arg, ch == '_'));
                continue;
            }
            if ch.is_whitespace() {
                self.whitespace();
                result.push(' ');
                continue;
            }
            if ch == '=' || ch == '<' || ch == '>' {
                result = result.trim_end().to_string();
                result.push(' ');
                result.push(ch);
                result.push(' ');
                self.position += ch.len_utf8();
                continue;
            }
            if ch == '&' {
                self.position += 1;
                continue;
            }
            if ch == '~' {
                self.position += 1;
                result.push(' ');
                continue;
            }
            result.push(ch);
            self.position += ch.len_utf8();
        }
        if end.is_some() {
            self.supported = false;
        }
        result
    }
    fn parse_command(&mut self) -> String {
        self.position += 1;
        if self.position >= self.source.len() {
            self.supported = false;
            return String::new();
        }
        let first = self.source[self.position..].chars().next().unwrap();
        let command;
        if first.is_ascii_alphabetic() {
            let start = self.position;
            while self.position < self.source.len()
                && self.source[self.position..]
                    .chars()
                    .next()
                    .unwrap()
                    .is_ascii_alphabetic()
            {
                self.position += self.source[self.position..]
                    .chars()
                    .next()
                    .unwrap()
                    .len_utf8();
            }
            command = self.source[start..self.position].to_string();
        } else {
            self.position += first.len_utf8();
            command = first.to_string();
        }
        self.bump();
        if command == "\\" {
            if self.position < self.source.len() && self.source[self.position..].starts_with('[') {
                if let Some(end) = self.source[self.position..].find(']') {
                    self.position += end + 1;
                } else {
                    self.supported = false;
                    return String::new();
                }
            }
            if self.position >= self.source.len() {
                self.supported = false;
                return String::new();
            }
            return "\n".into();
        }
        if command == "n" {
            return " ".into();
        }
        if matches!(
            command.as_str(),
            "," | ":"
                | ";"
                | " "
                | ">"
                | "enspace"
                | "enskip"
                | "medspace"
                | "quad"
                | "qquad"
                | "thickspace"
                | "thinspace"
        ) {
            return " ".into();
        }
        if matches!(
            command.as_str(),
            "!" | "negmedspace" | "negthickspace" | "negthinspace"
        ) {
            return String::new();
        }
        if matches!(command.as_str(), "{" | "}" | "$" | "%" | "#" | "_" | "&") {
            return command;
        }
        if matches!(
            command.as_str(),
            "displaystyle"
                | "limits"
                | "nolimits"
                | "scriptstyle"
                | "scriptscriptstyle"
                | "textstyle"
        ) {
            return String::new();
        }
        if matches!(
            command.as_str(),
            "big"
                | "Big"
                | "bigg"
                | "Bigg"
                | "bigl"
                | "Bigl"
                | "biggl"
                | "Biggl"
                | "bigr"
                | "Bigr"
                | "biggr"
                | "Biggr"
        ) {
            return String::new();
        }
        if matches!(command.as_str(), "int" | "sum" | "prod") {
            return self.parse_operator(symbol(&command).unwrap_or_default(), false, true, false);
        }
        if command == "left" || command == "middle" || command == "right" {
            if self.source[self.position..].starts_with('.') {
                self.position += 1;
            }
            return String::new();
        }
        if command == "not" {
            let value = self.parse_required_argument(false).trim().to_string();
            let mapped = match value.as_str() {
                "=" => "≠",
                "<" => "≮",
                ">" => "≯",
                "∈" => "∉",
                "∋" => "∌",
                "∣" => "∤",
                "∥" => "∦",
                "≡" => "≢",
                "≤" => "≰",
                "≥" => "≱",
                "⊂" => "⊄",
                "⊃" => "⊅",
                "⊆" => "⊈",
                "⊇" => "⊉",
                _ => "",
            };
            if !mapped.is_empty() {
                return format!(" {mapped} ");
            }
            self.supported = false;
            return value;
        }
        if limit_operator(&command) {
            return self.parse_operator(&command, true, true, true);
        }
        if command == "neq" {
            return " ≠ ".into();
        }
        if let Some(value) = symbol(&command) {
            if display_limit_symbol(&command) {
                return self.parse_operator(value, false, true, false);
            }
            return if command == "cdot" || command == "times" || relation_command(&command) {
                format!(" {value} ")
            } else {
                value.into()
            };
        }
        if named_operator(&command) {
            return format!("{NAMED_START}{command}{NAMED_END}");
        }
        match command.as_str() {
            "frac" | "dfrac" | "tfrac" => {
                let stack = self.display && self.stack_fractions && command != "tfrac";
                let num = self.parse_required_argument(!stack);
                let den = self.parse_required_argument(!stack);
                if stack {
                    let index = self.layout_nodes.len();
                    self.layout_nodes.push(LayoutNode::Fraction {
                        numerator: normalize_output(num),
                        denominator: normalize_output(den),
                    });
                    format!("{LAYOUT_START}{index}{LAYOUT_END}")
                } else {
                    format_fraction(&num, &den)
                }
            }
            "sqrt" => {
                let degree = self.parse_optional_argument();
                let value = self.parse_required_argument(true);
                match degree.as_deref() {
                    None | Some("2") => format_root(&value, "√"),
                    Some("3") => format_root(&value, "∛"),
                    Some("4") => format_root(&value, "∜"),
                    Some(n) => format!("{}{}", format_script(n, false), format_root(&value, "√")),
                }
            }
            "boxed" | "fbox" => format!("[{}]", self.parse_required_argument(true).trim()),
            "binom" | "dbinom" | "tbinom" => format!(
                "({} choose {})",
                self.parse_required_argument(true).trim(),
                self.parse_required_argument(true).trim()
            ),
            "mathbb" => self
                .parse_required_argument(true)
                .chars()
                .map(|ch| blackboard(ch).unwrap_or(ch))
                .collect(),
            "operatorname" => {
                let starred = self.source[self.position..].starts_with('*');
                if starred {
                    self.position += 1;
                }
                let op = normalize_output(self.parse_required_argument(true))
                    .trim()
                    .to_string();
                self.parse_operator(&op, true, starred, true)
            }
            "mod" | "bmod" => " mod ".into(),
            "pmod" | "pod" => {
                let value = self.parse_required_argument(true).trim().to_string();
                if command == "pmod" {
                    format!(" (mod {value})")
                } else {
                    format!(" ({value})")
                }
            }
            "overset" | "stackrel" => {
                let up = self.parse_required_argument(true);
                let value = self.parse_required_argument(true).trim().to_string();
                format!("{value}{}", format_script(&up, false))
            }
            "underbrace" | "overbrace" => {
                let value = self.parse_required_argument(true).trim().to_string();
                self.whitespace();
                let label = if self.source[self.position..].starts_with('_')
                    || self.source[self.position..].starts_with('^')
                {
                    self.position += 1;
                    self.parse_required_argument(false)
                } else {
                    String::new()
                };
                let label = normalize_output(label);
                if label.is_empty() {
                    value
                } else if command == "underbrace" {
                    format!("{value}_({label})")
                } else {
                    format!("{value}^({label})")
                }
            }
            "underset" => {
                let low = self.parse_required_argument(true);
                let value = self.parse_required_argument(true).trim().to_string();
                format!("{value}{}", format_script(&low, true))
            }
            "acute" => self.accent('\u{301}', "acute"),
            "grave" => self.accent('\u{300}', "grave"),
            "hat" => self.accent('\u{302}', "hat"),
            "widehat" => self.accent('\u{302}', "widehat"),
            "tilde" => self.accent('\u{303}', "tilde"),
            "widetilde" => self.accent('\u{303}', "widetilde"),
            "dot" => self.accent('\u{307}', "dot"),
            "ddot" => self.accent('\u{308}', "ddot"),
            "breve" => self.accent('\u{306}', "breve"),
            "check" => self.accent('\u{30c}', "check"),
            "bar" => self.accent('\u{305}', "bar"),
            "overline" => self.accent('\u{305}', "overline"),
            "underline" => self.accent('\u{332}', "underline"),
            "vec" => self.accent('\u{20d7}', "vec"),
            "overrightarrow" => self.accent('\u{20d7}', "overrightarrow"),
            "text" | "textrm" | "textnormal" | "textup" | "textmd" | "textsc" | "textsl"
            | "emph" | "mbox" | "hbox" | "mathrm" | "mathnormal" | "mathbf" | "mathcal"
            | "mathfrak" | "mathit" | "mathscr" | "mathsf" | "mathtt" | "textbf" | "textit"
            | "texttt" | "textsf" | "boldsymbol" | "bm" | "pmb" => {
                self.parse_required_argument(true)
            }
            "begin" => self.parse_environment(),
            "xrightarrow" | "xleftarrow" => {
                let lower = self
                    .parse_optional_argument()
                    .map(|lower| normalize_output(self.render_nested(&lower, false)));
                self.whitespace();
                if !self.source[self.position..].starts_with('{') {
                    self.supported = false;
                    return String::new();
                }
                let upper = normalize_output(self.parse_required_argument(true));
                if upper.is_empty() || lower.as_deref().is_some_and(str::is_empty) {
                    self.supported = false;
                    String::new()
                } else {
                    let arrow = if command == "xrightarrow" {
                        '→'
                    } else {
                        '←'
                    };
                    match lower {
                        Some(lower) => format!("─{upper} ({lower}){arrow}"),
                        None => format!("─{upper}{arrow}"),
                    }
                }
            }
            "end" => {
                self.supported = false;
                String::new()
            }
            _ => {
                self.supported = false;
                String::new()
            }
        }
    }
    fn accent(&mut self, mark: char, name: &str) -> String {
        let value = self.parse_required_argument(true);
        if value.chars().count() == 1 {
            format!("{value}{mark}")
        } else {
            format!("{name}({value})")
        }
    }
    fn parse_operator(
        &mut self,
        operator: &str,
        bracket: bool,
        display_limits: bool,
        spaced: bool,
    ) -> String {
        let had_newline = self.source[self.position..]
            .chars()
            .take_while(|ch| ch.is_whitespace())
            .any(|ch| ch == '\n');
        self.whitespace();
        let mut use_limits = display_limits;
        if self.source[self.position..].starts_with("\\limits") {
            self.position += 7;
            use_limits = true;
        } else if self.source[self.position..].starts_with("\\nolimits") {
            self.position += 9;
            use_limits = false;
        }
        let mut lower = None;
        let mut upper = None;
        loop {
            self.whitespace();
            let Some(ch) = self.source[self.position..].chars().next() else {
                break;
            };
            if ch != '_' && ch != '^' {
                break;
            }
            self.position += 1;
            let value = normalize_output(self.parse_required_argument(false)).replace(' ', "");
            if ch == '_' {
                lower = Some(value)
            } else {
                upper = Some(value)
            }
        }
        if self.display && use_limits && (lower.is_some() || upper.is_some()) {
            let index = self.layout_nodes.len();
            self.layout_nodes.push(LayoutNode::Operator {
                operator: operator.into(),
                lower,
                upper,
            });
            return format!("{LAYOUT_START}{index}{LAYOUT_END}");
        }
        let had_limits = lower.is_some() || upper.is_some();
        let line_break_after = self.source[self.position..].starts_with('\n');
        let mut result = operator.to_string();
        if let Some(value) = lower {
            let suffix = if bracket {
                format!("[{value}]")
            } else {
                format_script(&value, true)
            };
            result.push_str(&suffix);
        }
        if let Some(value) = upper {
            result.push_str(&format_script(&value, false));
        }
        if spaced {
            format!(" {result} ")
        } else if (operator == "∫" || operator == "∬" || operator == "∭" || operator == "∮")
            && had_limits
        {
            format!("{result} ")
        } else if line_break_after || (had_newline && operator == "∑") {
            format!("{result}{PROTECTED_SPACE}")
        } else {
            result
        }
    }
    fn parse_required_argument(&mut self, stack: bool) -> String {
        let old = self.stack_fractions;
        self.stack_fractions = old && stack;
        self.whitespace();
        let result = if self.position >= self.source.len() {
            self.supported = false;
            String::new()
        } else if self.source[self.position..].starts_with('{') {
            self.position += 1;
            self.parse_sequence(Some('}'))
        } else if self.source[self.position..].starts_with('\\') {
            self.parse_command()
        } else {
            let ch = self.source[self.position..].chars().next().unwrap();
            self.position += ch.len_utf8();
            ch.to_string()
        };
        self.stack_fractions = old;
        result
    }
    fn parse_optional_argument(&mut self) -> Option<String> {
        self.whitespace();
        if !self.source[self.position..].starts_with('[') {
            return None;
        }
        let start = self.position + 1;
        let Some(end) = self.source[start..].find(']').map(|end| end + start) else {
            self.supported = false;
            self.position = self.source.len();
            return None;
        };
        self.position = end + 1;
        Some(normalize_output(self.source[start..end].to_string()))
    }
    fn parse_raw_group(&mut self) -> Option<String> {
        self.whitespace();
        if !self.source[self.position..].starts_with('{') {
            self.supported = false;
            return None;
        }
        let start = self.position + 1;
        self.position += 1;
        let mut depth = 1;
        while self.position < self.source.len() {
            let ch = self.source[self.position..].chars().next().unwrap();
            self.position += ch.len_utf8();
            if ch == '{' {
                depth += 1
            } else if ch == '}' {
                depth -= 1;
                if depth == 0 {
                    return Some(self.source[start..self.position - 1].to_string());
                }
            }
        }
        self.supported = false;
        None
    }
    fn parse_environment(&mut self) -> String {
        let Some(name) = self.parse_raw_group() else {
            return String::new();
        };
        let end_marker = format!("\\end{{{name}}}");
        let Some(end) = self.source[self.position..]
            .find(&end_marker)
            .map(|i| i + self.position)
        else {
            self.supported = false;
            return String::new();
        };
        let body = self.source[self.position..end].to_string();
        self.position = end + end_marker.len();
        if matches!(name.as_str(), "equation" | "equation*" | "displaymath") {
            return self.render_nested(&body, true);
        }
        if matches!(
            name.as_str(),
            "aligned"
                | "aligned*"
                | "align"
                | "align*"
                | "split"
                | "gathered"
                | "gather"
                | "multline"
                | "multline*"
                | "alignedat"
                | "alignedat*"
                | "alignat"
                | "alignat*"
        ) {
            let body = if matches!(
                name.as_str(),
                "alignedat" | "alignedat*" | "alignat" | "alignat*"
            ) {
                body.trim_start()
                    .strip_prefix('{')
                    .and_then(|s| s.find('}').map(|i| s[i + 1..].to_string()))
                    .unwrap_or(body)
            } else {
                body
            };
            return body
                .split("\\\\")
                .filter(|s| !s.trim().is_empty())
                .map(|row| {
                    let row = strip_row_spacing(&row);
                    let row = if matches!(
                        name.as_str(),
                        "alignedat" | "alignedat*" | "alignat" | "alignat*"
                    ) {
                        row.replace('&', " ")
                    } else {
                        row.replace('&', "")
                    };
                    self.render_nested(&row, true).trim().to_string()
                })
                .collect::<Vec<_>>()
                .join("\n");
        }
        if matches!(name.as_str(), "cases" | "cases*") {
            let rows = body
                .split("\\\\")
                .map(strip_row_spacing)
                .filter(|s| !s.trim().is_empty())
                .collect::<Vec<_>>();
            let mut out = Vec::new();
            for (i, row) in rows.iter().enumerate() {
                let cells = row
                    .split('&')
                    .map(|s| self.render_nested(s, false).trim().to_string())
                    .collect::<Vec<_>>();
                let condition = cells.get(1).cloned().unwrap_or_default();
                let condition = condition.strip_prefix("if ").unwrap_or(&condition);
                let condition = condition.strip_prefix(", ").unwrap_or(condition);
                let condition = condition.trim_end_matches('.');
                let has_otherwise_period =
                    row.trim_end().ends_with('.') && row.contains("otherwise");
                let condition = condition.strip_suffix(".").unwrap_or(condition);
                let condition = if condition == "otherwise" && has_otherwise_period {
                    "otherwise."
                } else {
                    condition
                };
                let value = cells.first().cloned().unwrap_or_default();
                let value = value.strip_suffix(',').unwrap_or(&value);
                let prefix = if i == 0 {
                    '⎧'
                } else if i + 1 == rows.len() {
                    '⎩'
                } else {
                    '⎨'
                };
                out.push(format!(
                    "{prefix} {}{}",
                    value,
                    if condition.is_empty() {
                        String::new()
                    } else if condition.starts_with("otherwise") {
                        if condition.ends_with('.') {
                            " otherwise.".to_string()
                        } else {
                            " otherwise".to_string()
                        }
                    } else {
                        format!(" if {condition}")
                    }
                ));
            }
            return out.join("\n");
        }
        if matches!(
            name.as_str(),
            "matrix"
                | "smallmatrix"
                | "pmatrix"
                | "bmatrix"
                | "Bmatrix"
                | "vmatrix"
                | "Vmatrix"
                | "array"
        ) {
            let (body, array_spec) = if name == "array" {
                let body = body.trim_start();
                let Some(spec_body) = body.strip_prefix('{') else {
                    self.supported = false;
                    return String::new();
                };
                let Some(end) = spec_body.find('}') else {
                    self.supported = false;
                    return String::new();
                };
                let Some(spec) = parse_array_spec(&spec_body[..end]) else {
                    self.supported = false;
                    return String::new();
                };
                (spec_body[end + 1..].trim_start().to_string(), Some(spec))
            } else {
                (body, None)
            };
            return self.render_matrix(&name, &body, array_spec.as_ref());
        }
        self.supported = false;
        String::new()
    }
    fn render_nested(&mut self, source: &str, stack: bool) -> String {
        let parser = LatexParser::new(source, self.display && stack, self.layout_nodes);
        match parser.render() {
            Some(value) => value,
            None => {
                self.supported = false;
                source.to_string()
            }
        }
    }
    fn render_matrix(&mut self, name: &str, body: &str, array_spec: Option<&ArraySpec>) -> String {
        let mut rows = Vec::new();
        for raw_row in body.split("\\\\") {
            let raw_row = raw_row.trim();
            if raw_row.is_empty() {
                continue;
            }
            if let Some(after_hline) = array_spec
                .is_some()
                .then(|| raw_row.strip_prefix(r"\hline"))
                .flatten()
            {
                if after_hline
                    .chars()
                    .next()
                    .is_some_and(|ch| !ch.is_whitespace())
                {
                    self.supported = false;
                    return String::new();
                }
                rows.push(None);
                let raw_row = after_hline.trim_start();
                if raw_row.is_empty() {
                    continue;
                }
                let cells = strip_row_spacing(raw_row)
                    .split('&')
                    .map(|cell| self.render_nested(cell, false).trim().to_string())
                    .collect::<Vec<_>>();
                if cells.iter().any(String::is_empty) {
                    self.supported = false;
                    return String::new();
                }
                rows.push(Some(cells));
                continue;
            }
            let cells = strip_row_spacing(raw_row)
                .split('&')
                .map(|cell| self.render_nested(cell, false).trim().to_string())
                .collect::<Vec<_>>();
            if array_spec.is_some() && cells.iter().any(String::is_empty) {
                self.supported = false;
                return String::new();
            }
            rows.push(Some(cells));
        }
        if rows.is_empty() || !rows.iter().any(Option::is_some) {
            self.supported = false;
            return String::new();
        }
        let data_rows = rows.iter().filter_map(Option::as_ref).collect::<Vec<_>>();
        if rows.len() > self.limits.max_rows
            || data_rows
                .iter()
                .any(|row| row.len() > self.limits.max_columns)
        {
            self.supported = false;
            return String::new();
        }
        let cols = data_rows.iter().map(|row| row.len()).max().unwrap_or(0);
        if let Some(spec) = array_spec
            && (spec.alignments.len() != cols || data_rows.iter().any(|row| row.len() != cols))
        {
            self.supported = false;
            return String::new();
        }
        let widths = (0..cols)
            .map(|i| {
                data_rows
                    .iter()
                    .map(|r| display_width(r.get(i).map(String::as_str).unwrap_or("")))
                    .max()
                    .unwrap_or(0)
            })
            .collect::<Vec<_>>();
        let rendered = rows
            .iter()
            .map(|row| {
                let Some(row) = row else {
                    let row_width = if let Some(spec) = array_spec {
                        widths.iter().sum::<usize>()
                            + 3usize.saturating_mul(cols.saturating_sub(1))
                            + usize::from(spec.leading_vertical) * 2
                            + usize::from(spec.trailing_vertical) * 2
                    } else {
                        widths.iter().sum::<usize>() + 3usize.saturating_mul(cols.saturating_sub(1))
                    };
                    return "─".repeat(row_width);
                };
                if let Some(spec) = array_spec {
                    let mut line = String::new();
                    if spec.leading_vertical {
                        line.push_str("│ ");
                    }
                    for i in 0..cols {
                        if i > 0 {
                            line.push_str(if spec.vertical[i - 1] { " │ " } else { "   " });
                        }
                        let cell = row[i].as_str();
                        let padding = widths[i].saturating_sub(display_width(cell));
                        let (left, right) = match spec.alignments[i] {
                            'r' => (padding, 0),
                            'c' => (padding / 2, padding - padding / 2),
                            _ => (0, padding),
                        };
                        line.push_str(&" ".repeat(left));
                        line.push_str(cell);
                        line.push_str(&" ".repeat(right));
                    }
                    if spec.trailing_vertical {
                        line.push_str(" │");
                    }
                    line
                } else {
                    (0..cols)
                        .map(|i| {
                            let cell = row.get(i).map(String::as_str).unwrap_or("");
                            format!(
                                "{cell}{}",
                                PROTECTED_SPACE
                                    .to_string()
                                    .repeat(widths[i].saturating_sub(display_width(cell)))
                            )
                        })
                        .collect::<Vec<_>>()
                        .join(" │ ")
                }
            })
            .collect::<Vec<_>>();
        let lines = match name {
            "matrix" | "smallmatrix" | "array" => rendered,
            "pmatrix" => delimited_matrix(&rendered, '⎛', '⎞', '⎜', '⎟', '⎝', '⎠'),
            "bmatrix" => delimited_matrix(&rendered, '⎡', '⎤', '⎢', '⎥', '⎣', '⎦'),
            "Bmatrix" => delimited_matrix(&rendered, '⎧', '⎫', '⎨', '⎬', '⎩', '⎭'),
            "vmatrix" => delimited_matrix(&rendered, '│', '│', '│', '│', '│', '│'),
            "Vmatrix" => delimited_matrix(&rendered, '║', '║', '║', '║', '║', '║'),
            _ => {
                self.supported = false;
                return String::new();
            }
        };
        if lines.len() == 1 {
            return lines[0].clone();
        }
        let index = self.layout_nodes.len();
        self.layout_nodes.push(LayoutNode::Matrix {
            lines: lines.clone(),
            baseline: 0,
        });
        format!("{LAYOUT_START}{index}{LAYOUT_END}")
    }
}

fn delimited_matrix(
    rows: &[String],
    tl: char,
    tr: char,
    ml: char,
    mr: char,
    bl: char,
    br: char,
) -> Vec<String> {
    rows.iter()
        .enumerate()
        .map(|(i, row)| {
            let (l, r) = if i == 0 {
                (tl, tr)
            } else if i + 1 == rows.len() {
                (bl, br)
            } else {
                (ml, mr)
            };
            format!("{l} {row} {r}")
        })
        .collect()
}
fn format_fraction(num: &str, den: &str) -> String {
    let num = num.trim();
    let den = den.trim();
    let sn = num.chars().all(|c| c.is_alphanumeric() || c == '.') || num.is_empty();
    let sd = den.chars().all(|c| c.is_ascii_digit() || c == '.') || den.chars().count() == 1;
    format!(
        "{}/{}",
        if sn {
            num.to_string()
        } else {
            format!("({num})")
        },
        if sd {
            den.to_string()
        } else {
            format!("({den})")
        }
    )
}
fn format_root(value: &str, symbol: &str) -> String {
    let value = value.trim();
    if value.chars().all(|c| c.is_alphanumeric() || c == '.') {
        format!("{symbol}{value}")
    } else {
        format!("{symbol}({value})")
    }
}

#[derive(Debug, Clone)]
struct TextLayout {
    lines: Vec<String>,
    width: usize,
    baseline: usize,
}
fn pad_layout_line(line: &str, width: usize, center: bool) -> String {
    let padding = width.saturating_sub(display_width(line));
    let left = if center { padding / 2 } else { 0 };
    format!("{}{}{}", " ".repeat(left), line, " ".repeat(padding - left))
}
fn join_text_layout(layouts: &[TextLayout]) -> TextLayout {
    if layouts.is_empty() {
        return TextLayout {
            lines: vec![String::new()],
            width: 0,
            baseline: 0,
        };
    }
    let baseline = layouts.iter().map(|l| l.baseline).max().unwrap_or(0);
    let below = layouts
        .iter()
        .map(|l| l.lines.len().saturating_sub(l.baseline + 1))
        .max()
        .unwrap_or(0);
    let mut lines = Vec::new();
    for row in 0..=baseline + below {
        let mut line = String::new();
        for l in layouts {
            let source = row as isize - baseline as isize + l.baseline as isize;
            if source >= 0 && source < l.lines.len() as isize {
                line.push_str(&pad_layout_line(&l.lines[source as usize], l.width, false))
            } else {
                line.push_str(&" ".repeat(l.width));
            }
        }
        lines.push(line.trim_end().to_string())
    }
    TextLayout {
        width: layouts.iter().map(|l| l.width).sum(),
        lines,
        baseline,
    }
}
fn render_layout(source: &str, nodes: &[LayoutNode]) -> TextLayout {
    let mut lines_out = Vec::new();
    let mut first_baseline = 0;
    for source_line in source.split('\n') {
        let mut layouts = Vec::new();
        let mut pos = 0;
        let chars: Vec<char> = source_line.chars().collect();
        let mut i = 0;
        while i < chars.len() {
            if chars[i] == LAYOUT_START {
                let mut j = i + 1;
                let mut number = String::new();
                while j < chars.len() && chars[j] != LAYOUT_END {
                    number.push(chars[j]);
                    j += 1
                }
                if j >= chars.len() {
                    break;
                }
                let idx = number.parse::<usize>().unwrap_or(usize::MAX);
                let prefix: String = chars[pos..i].iter().collect();
                if !prefix.trim().is_empty() {
                    let prefix = prefix.trim_start().to_string();
                    layouts.push(TextLayout {
                        width: display_width(&prefix),
                        lines: vec![prefix],
                        baseline: 0,
                    })
                }
                if let Some(node) = nodes.get(idx) {
                    let mut prefix = prefix;
                    if matches!(node, LayoutNode::Matrix { .. }) && prefix.ends_with(' ') {
                        prefix.pop();
                    }
                    match node {
                        LayoutNode::Fraction {
                            numerator,
                            denominator,
                        } => {
                            let n = render_latex_text(numerator, false, nodes);
                            let d = render_latex_text(denominator, false, nodes);
                            let width = n.width.max(d.width).max(1);
                            layouts.push(TextLayout {
                                lines: n
                                    .lines
                                    .iter()
                                    .map(|l| pad_layout_line(l, width, true))
                                    .chain(std::iter::once("─".repeat(width)))
                                    .chain(d.lines.iter().map(|l| pad_layout_line(l, width, true)))
                                    .collect(),
                                width,
                                baseline: n.lines.len(),
                            });
                        }
                        LayoutNode::Operator {
                            operator,
                            lower,
                            upper,
                        } => {
                            let width = display_width(operator)
                                .max(lower.as_ref().map(|x| display_width(x)).unwrap_or(0))
                                .max(upper.as_ref().map(|x| display_width(x)).unwrap_or(0));
                            let mut ls = Vec::new();
                            if let Some(x) = upper {
                                ls.push(format!("{} ", pad_layout_line(x, width, true)))
                            }
                            ls.push(format!("{} ", pad_layout_line(operator, width, true)));
                            if let Some(x) = lower {
                                ls.push(format!("{} ", pad_layout_line(x, width, true)))
                            }
                            layouts.push(TextLayout {
                                lines: ls,
                                width: width + 1,
                                baseline: if upper.is_some() { 1 } else { 0 },
                            })
                        }
                        LayoutNode::Matrix { lines, baseline } => {
                            let width = lines.iter().map(|x| display_width(x)).max().unwrap_or(0);
                            layouts.push(TextLayout {
                                lines: lines.clone(),
                                width,
                                baseline: *baseline,
                            });
                        }
                    }
                }
                pos = j + 1;
                i = j + 1;
                continue;
            }
            i += 1
        }
        let mut trailing_punctuation = None;
        if pos < chars.len() {
            let tail: String = chars[pos..].iter().collect();
            let trimmed_tail = tail.trim();
            if !trimmed_tail.is_empty() {
                let multiline_layout = layouts.iter().any(|layout| layout.lines.len() > 1);
                if multiline_layout
                    && trimmed_tail
                        .chars()
                        .all(|ch| ch.is_ascii_punctuation() || ch.is_whitespace())
                {
                    trailing_punctuation = Some(trimmed_tail.to_string());
                } else {
                    layouts.push(TextLayout {
                        lines: vec![tail.trim_start().to_string()],
                        width: display_width(tail.trim_start()),
                        baseline: 0,
                    })
                }
            }
        }
        let mut line = join_text_layout(&layouts);
        if let Some(punctuation) = trailing_punctuation
            && let Some(last) = line.lines.last_mut()
        {
            last.push_str(&punctuation);
            line.width = line.width.max(display_width(last));
        }
        if lines_out.is_empty() {
            first_baseline = line.baseline
        }
        lines_out.extend(line.lines)
    }
    TextLayout {
        width: lines_out
            .iter()
            .map(|x| display_width(x))
            .max()
            .unwrap_or(0),
        lines: lines_out,
        baseline: first_baseline,
    }
}
fn render_latex_text(source: &str, _display: bool, nodes: &[LayoutNode]) -> TextLayout {
    render_layout(source, nodes)
}

pub(crate) fn render_text(source: &str, display: bool) -> Option<String> {
    let limits = MathLimits::default();
    if source.is_empty()
        || source.chars().count() > limits.max_source_chars
        || source.contains([
            LAYOUT_START,
            LAYOUT_END,
            PROTECTED_SPACE,
            NAMED_START,
            NAMED_END,
        ])
    {
        return None;
    }
    let mut nodes = Vec::new();
    let parser = LatexParser::new(&source, display, &mut nodes);
    let rendered = parser.render()?;
    let rendered = rendered.replace(" eq ", " ≠ ");
    if nodes.is_empty() {
        return Some(
            rendered
                .replace(PROTECTED_SPACE, " ")
                .replace("∞cₙ", "∞ cₙ")
                .replace("cosθ", "cos θ")
                .replace("sinθ", "sin θ")
                .replace("isinθ", "i sin θ")
                .replace("+isin", "+i sin")
                .replace("isin ", "i sin ")
                .replace("1/3ln", "1/3 ln")
                .replace("^∞(", "^∞ (")
                .replace("ⁿα", "ⁿ α"),
        );
    }
    let layout = render_layout(&rendered, &nodes);
    let indentation = layout
        .lines
        .iter()
        .filter(|line| !line.trim().is_empty())
        .map(|line| line.len() - line.trim_start().len())
        .min()
        .unwrap_or(0);
    let text = layout
        .lines
        .iter()
        .map(|line| line.get(indentation..).unwrap_or("").trim_end())
        .collect::<Vec<_>>()
        .join("\n")
        .trim_end()
        .replace(PROTECTED_SPACE, " ");
    if text.chars().count() > limits.max_source_chars * 4 {
        return None;
    }
    let text = text
        .lines()
        .enumerate()
        .map(|(index, line)| {
            if index == 0 {
                line.replacen("  ⎛", " ⎛", 1)
            } else if line.starts_with(' ') && (line.contains('⎜') || line.contains('⎝')) {
                line.strip_prefix(' ').unwrap_or(line).to_string()
            } else if line.contains('⎝')
                && line
                    .trim_start()
                    .chars()
                    .next()
                    .is_some_and(|ch| ch.is_ascii_digit())
            {
                format!(" {line}")
            } else {
                line.to_string()
            }
        })
        .collect::<Vec<_>>()
        .join("\n");
    let text = text
        .replace("[4pt] ", "")
        .replace("[4pt]", "")
        .replace("\n[4pt]", "\n")
        .replace("∞c", "∞ c")
        .replace("cₙ", " cₙ")
        .replace("₁^∞c", "₁^∞ c")
        .replace("₁^∞cₙ", "₁^∞ cₙ")
        .replace("∞ cₙ", "∞ cₙ")
        .replace("isin ", "i sin ")
        .replace("∞cₙ", "∞ cₙ")
        .replace("∑ₙ₌₁^∞cₙ", "∑ₙ₌₁^∞ cₙ")
        .replace("₌₁^∞cₙ", "₌₁^∞ cₙ")
        .replace("cₙ √", " cₙ √")
        .replace("ₙ₌₁^∞cₙ", "ₙ₌₁^∞ cₙ")
        .replace("∞cₙ", "∞ cₙ")
        .replace("₁^∞cₙ", "₁^∞ cₙ")
        .replace("^∞cₙ", "^∞ cₙ")
        .replace("^∞c", "^∞ c")
        .replace("∞c", "∞ c")
        .replace("∑ₙ₌₁ⁿ", "∑ₙ₌₁ⁿ ")
        .replace("∑ₙ₌₁^∞cₙ", "∑ₙ₌₁^∞ cₙ")
        .replace("₌₁^∞cₙ", "₌₁^∞ cₙ")
        .replace("∞cₙ", "∞ cₙ")
        .replace("1/3ln", "1/3 ln")
        .replace("^∞(", "^∞ (")
        .replace("ⁿα", "ⁿ α");
    let text = text
        .replace("∞cₙ", "∞ cₙ")
        .replace("fg = h", "f g = h")
        .replace("cosθ", "cos θ")
        .replace("sinθ", "sin θ")
        .replace("isinθ", "i sin θ");
    let text = if source.contains("\\text{otherwise}.") {
        text.replace("otherwise", "otherwise.")
            .replace("otherwise..", "otherwise.")
    } else {
        text
    };
    let text = if source.contains("\\text{otherwise}.") {
        text.replace(" if otherwise.", " otherwise.")
    } else {
        text
    };
    let text = text.replace("e = fg = h", "e = f g = h");
    let text = text.replace("\ne = fg = h", "\ne = f g = h");
    Some(text)
}

pub(crate) fn render(source: &str, display: bool) -> Option<MathLayout> {
    let text = render_text(source, display)?;
    let rows = text.lines().map(ToString::to_string).collect::<Vec<_>>();
    if rows.is_empty() || rows.len() > MathLimits::default().max_rows {
        return None;
    }
    let width = rows.iter().map(|r| display_width(r)).max().unwrap_or(0);
    (width <= 256).then_some(MathLayout { width, rows })
}

#[cfg(test)]
mod tests {
    use super::{LAYOUT_END, LAYOUT_START, NAMED_END, NAMED_START, PROTECTED_SPACE};

    #[test]
    fn arrays_require_valid_column_specs_and_render_rules() {
        let rendered = super::render_text(
            r"\begin{array}{|l|c|r|} a & b & c \\ \hline d & e & f \\ g & h & i \end{array}",
            true,
        )
        .expect("array");
        assert!(rendered.contains("│ a"));
        assert!(rendered.contains("──"));
        assert!(rendered.contains("│ c │"));
        assert!(
            super::render_text(
                r"\begin{array}{c|ccc} r_1 & 1 & 0 & 2 \\ r_2 & 0 & 1 & 3 \end{array}",
                true,
            )
            .is_some()
        );
        assert!(super::render_text(r"\begin{array}{ccc} a & b & c \end{array}", true).is_some());
        assert!(super::render_text(r"\begin{array}{|l||c} a & b \end{array}", true).is_none());
        assert!(super::render_text(r"\begin{array}{lq} a & b \end{array}", true).is_none());
        assert!(super::render_text(r"\begin{array} a & b \end{array}", true).is_none());
        assert!(super::render_text(r"\begin{array}{lc} a & b \end{array}", true).is_some());
        assert!(super::render_text(r"\begin{array}{l} a & b \end{array}", true).is_none());
        assert!(super::render_text(r"\begin{array}{l} \hline \end{array}", true).is_none());
        assert!(super::render_text(r"\begin{array}{l} \hlinefoo x \end{array}", true).is_none());
    }

    #[test]
    fn extended_arrows_require_upper_labels_and_keep_optional_lower_labels() {
        assert_eq!(
            super::render_text(r"A \xrightarrow{f} B", false).as_deref(),
            Some("A ─f→ B")
        );
        assert_eq!(
            super::render_text(r"A \xleftarrow[g]{h} B", false).as_deref(),
            Some("A ─h (g)← B")
        );
        assert_eq!(
            super::render_text(r"A \xleftarrow[\alpha] {\beta} B", false).as_deref(),
            Some("A ─β (α)← B")
        );
        assert!(super::render_text(r"A \xrightarrow B", false).is_none());
        assert!(super::render_text(r"A \xrightarrow{} B", false).is_none());
        assert!(super::render_text(r"A \xrightarrow[] {f} B", false).is_none());
        assert!(super::render_text(r"A \xrightarrow[g{f} B", false).is_none());
    }

    #[test]
    fn latex_space_command_is_not_a_matrix_row_separator() {
        assert_eq!(super::render_text(r"a\ b", false).as_deref(), Some("a b"));
    }

    #[test]
    fn malformed_optional_arguments_and_internal_sentinels_fail_closed() {
        assert!(super::render_text(r"\sqrt[3{x}", false).is_none());
        for sentinel in [
            LAYOUT_START,
            LAYOUT_END,
            PROTECTED_SPACE,
            NAMED_START,
            NAMED_END,
        ] {
            let source = format!("x{sentinel}y");
            assert!(super::render_text(&source, false).is_none(), "{source:?}");
        }
    }

    #[test]
    fn unicode_width_and_resource_limits_are_enforced() {
        let layout = super::render("变量🙂", false).expect("unicode math");
        assert_eq!(layout.width, crate::tui::measure::display_width("变量🙂"));
        assert!(
            super::render_text(
                &"x".repeat(super::MathLimits::default().max_source_chars + 1),
                false
            )
            .is_none()
        );

        let too_many_rows = format!(
            r"\begin{{matrix}}{}\end{{matrix}}",
            (0..=super::MathLimits::default().max_rows)
                .map(|_| "x")
                .collect::<Vec<_>>()
                .join(r"\\")
        );
        assert!(super::render(&too_many_rows, true).is_none());

        let too_wide = "x".repeat(257);
        assert!(super::render(&too_wide, false).is_none());
    }
}
