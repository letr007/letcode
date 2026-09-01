//! Anchored bootstrap experiment: keep the FIRST model request on a
//! Minimal-aligned tool pair (alias names `bash` / `str_replace_editor`) with no
//! injected context, then restore the full catalog and regular injections after
//! the session's first durable promotion signal.
//!
//! Design decisions:
//! - Persona stays Minimal for the whole session (dsh-anch behavior) — no jump.
//! - The promoted phase restores the FULL catalog and injections with zero
//!   residue: alias names only exist while the phase is not promoted.
//! - A compaction falls back to the controlled phase until a NEW durable signal.
//!
//! The module is deliberately self-contained: agent pipeline code only stores an
//! `Option<AnchoredBootstrap>` and calls one of the four methods below.

use crate::config::{AnchoredBootstrapConfig, PromoteOn};
use crate::protocol_frames::ProtocolItem as HistoryItem;
use crate::request_builder::{PromptMessage, ToolSpec};
use crate::tool_names::{TOOL_EDIT_APPLY_PATCH, TOOL_SHELL_EXEC};

/// Alias tool names exposed to the model during bootstrap/fallback. Kept
/// byte-identical to dsh-anchored-standard's Minimal bootstrap pair.
pub const ALIAS_BASH: &str = "bash";
pub const ALIAS_STR_REPLACE_EDITOR: &str = "str_replace_editor";

/// The Minimal preset persona (DeepSeek Harness official Minimal preset text,
/// reused verbatim by dsh-anchored-standard). Replaces the letcode persona for
/// the whole experiment session — no persona jump between phases.
pub const MINIMAL_PERSONA: &str = "You are a helpful software engineer assistant.";

/// Session phase derived from durable history.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AnchoredPhase {
    /// First request, or a compaction happened and no new signal exists past
    /// its boundary.
    Bootstrap,
    /// A durable assistant-side signal exists (and none since the last
    /// compaction, when one exists).
    Promoted,
    /// A compaction happened and no new signal exists past its boundary.
    CompactedFallback,
}

impl AnchoredPhase {
    pub(crate) const fn as_str(self) -> &'static str {
        match self {
            Self::Bootstrap => "bootstrap",
            Self::Promoted => "promoted",
            Self::CompactedFallback => "compacted-fallback",
        }
    }
}

/// Per-agent experiment state. Constructed only when the experiment is enabled;
/// every pipeline hook gates on [`AnchoredBootstrap::enabled_for`] so switching
/// models off the whitelist naturally bypasses the experiment.
pub struct AnchoredBootstrap {
    config: AnchoredBootstrapConfig,
}

impl AnchoredBootstrap {
    /// Some when the experiment is enabled; otherwise None.
    pub fn from_config(config: &AnchoredBootstrapConfig) -> Option<Self> {
        config.enabled.then(|| Self {
            config: config.clone(),
        })
    }

    /// Whether the current model is on the whitelist. Evaluated at each hook so
    /// a mid-session model switch off the whitelist bypasses the experiment.
    pub fn enabled_for(&self, model: &str) -> bool {
        self.config
            .models
            .iter()
            .any(|candidate| candidate == model)
    }

    /// Core work set exposed after a compaction (alias pair always present).
    pub(crate) fn compaction_tools(&self) -> &[String] {
        &self.config.compaction_tools
    }

    fn has_signal(&self, history: &[HistoryItem]) -> bool {
        let mode = self.config.promote_on;
        history.iter().any(|item| match (mode, item) {
            (PromoteOn::ToolCall, HistoryItem::ToolOutput { .. }) => true,
            (
                PromoteOn::AssistantMessage,
                HistoryItem::AssistantTurn {
                    text: Some(text), ..
                },
            ) => !text.is_empty(),
            (PromoteOn::AssistantMessage, HistoryItem::AssistantTurn { .. }) => true,
            (PromoteOn::Either, HistoryItem::ToolOutput { .. }) => true,
            (
                PromoteOn::Either,
                HistoryItem::AssistantTurn {
                    text: Some(text), ..
                },
            ) => !text.is_empty(),
            (PromoteOn::Either, HistoryItem::AssistantTurn { .. }) => true,
            _ => false,
        })
    }

    /// Derive the session phase from durable history. Pure function: no
    /// process state, resume/reload keep it by construction.
    pub fn phase(&self, history: &[HistoryItem]) -> AnchoredPhase {
        let boundary = history
            .iter()
            .rposition(|item| matches!(item, HistoryItem::ContextSummary { .. }));
        match boundary {
            Some(boundary) if !self.has_signal(&history[boundary..]) => {
                AnchoredPhase::CompactedFallback
            }
            _ if self.has_signal(history) => AnchoredPhase::Promoted,
            _ => AnchoredPhase::Bootstrap,
        }
    }

    /// Narrow an assembled catalog to this phase's tool set. Alias names are
    /// API-visible only; execution resolves them back via
    /// [`Self::resolve_tool_name`].
    pub fn tool_catalog(&self, phase: &AnchoredPhase, full: Vec<ToolSpec>) -> Vec<ToolSpec> {
        match phase {
            AnchoredPhase::Promoted => full,
            AnchoredPhase::Bootstrap => alias_pair(full),
            AnchoredPhase::CompactedFallback => {
                let mut out = alias_pair(full.clone());
                out.extend(full.into_iter().filter(|spec| {
                    // Never duplicate the alias pair even if compaction_tools
                    // lists the real tools they map to.
                    spec.name != TOOL_SHELL_EXEC
                        && spec.name != TOOL_EDIT_APPLY_PATCH
                        && self.config.compaction_tools.contains(&spec.name)
                }));
                out
            }
        }
    }

    /// Resolve an alias tool name to the real registry name. Pure name mapping:
    /// the caller ([`crate::agent::Agent::resolve_tool_alias`]) gates it on the
    /// request-bound phase so aliases from promoted requests — or same-named
    /// MCP tools — are never hijacked.
    pub fn resolve_tool_name(&self, name: &str) -> String {
        match name {
            ALIAS_BASH => TOOL_SHELL_EXEC.to_string(),
            ALIAS_STR_REPLACE_EDITOR => TOOL_EDIT_APPLY_PATCH.to_string(),
            _ => name.to_string(),
        }
    }

    /// Assemble the turn prelude for this phase. The persona (first base
    /// message) is replaced with the Minimal persona for the whole session;
    /// bootstrap/fallback keeps ONLY the persona plus manual skill gestures.
    pub fn prelude(
        &self,
        phase: &AnchoredPhase,
        base: &[PromptMessage],
        runtime: Option<PromptMessage>,
        skill: Option<PromptMessage>,
        workflow: Option<PromptMessage>,
        manual: &[PromptMessage],
    ) -> Vec<PromptMessage> {
        let mut persona = base
            .first()
            .cloned()
            .unwrap_or_else(|| PromptMessage::system(MINIMAL_PERSONA));
        persona.text = MINIMAL_PERSONA.to_string();

        match phase {
            AnchoredPhase::Promoted => {
                // Full restoration: the AGENTS.md chain (appended after the
                // persona by load_instruction_file) and every injection return.
                let mut out = vec![persona];
                out.extend(base.iter().skip(1).cloned());
                // Message order matches the regular (non-experiment) path:
                // persona, AGENTS.md chain, runtime, skill catalog, manual
                // skill material, workflow.
                if let Some(runtime) = runtime {
                    out.push(runtime);
                }
                if let Some(skill) = skill {
                    out.push(skill);
                }
                out.extend(manual.iter().cloned());
                if let Some(workflow) = workflow {
                    out.push(workflow);
                }
                out
            }
            AnchoredPhase::Bootstrap | AnchoredPhase::CompactedFallback => {
                let mut out = vec![persona];
                // User-initiated skill gestures are never stripped.
                out.extend(manual.iter().cloned());
                out
            }
        }
    }
}

/// Extract the bootstrap alias pair (`bash` + `str_replace_editor`) from a full
/// catalog, keeping the real tools' schemas but renaming them to the alias
/// names. Missing real tools simply yield a smaller pair (composition drift
/// must not brick a request).
fn alias_pair(full: Vec<ToolSpec>) -> Vec<ToolSpec> {
    let mut out = Vec::with_capacity(2);
    for spec in full {
        match spec.name.as_str() {
            TOOL_SHELL_EXEC => out.push(ToolSpec {
                name: ALIAS_BASH.to_string(),
                ..spec
            }),
            TOOL_EDIT_APPLY_PATCH => out.push(ToolSpec {
                name: ALIAS_STR_REPLACE_EDITOR.to_string(),
                ..spec
            }),
            _ => {}
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::PromoteOn;
    use serde_json::json;

    fn config(promote_on: PromoteOn) -> AnchoredBootstrapConfig {
        AnchoredBootstrapConfig {
            enabled: true,
            models: vec!["deepseek-v4-pro".into()],
            promote_on,
            compaction_tools: vec!["fs__read".into(), "workflow__todos".into()],
        }
    }

    fn spec(name: &str) -> ToolSpec {
        ToolSpec {
            name: name.to_string(),
            description: String::new(),
            parameters: json!({}),
            strict: true,
        }
    }

    fn full_catalog() -> Vec<ToolSpec> {
        vec![
            spec("shell__exec"),
            spec("edit__apply_patch"),
            spec("fs__read"),
            spec("fs__write"),
            spec("workflow__todos"),
            spec("search__rg"),
        ]
    }

    fn tool_output() -> HistoryItem {
        HistoryItem::ToolOutput {
            call_id: "call-1".into(),
            output_json: "{}".into(),
            images: vec![],
        }
    }

    fn tool_calls() -> HistoryItem {
        HistoryItem::AssistantTurn {
            text: Some("calling".into()),
            reasoning_content: None,
            replay: None,
            calls: vec![],
        }
    }

    #[test]
    fn phase_empty_or_user_only_is_bootstrap() {
        let anchored = AnchoredBootstrap::from_config(&config(PromoteOn::Either)).unwrap();
        assert_eq!(anchored.phase(&[]), AnchoredPhase::Bootstrap);
        assert_eq!(
            anchored.phase(&[HistoryItem::user("hello")]),
            AnchoredPhase::Bootstrap
        );
    }

    #[test]
    fn phase_promotes_on_assistant_signal() {
        let anchored = AnchoredBootstrap::from_config(&config(PromoteOn::Either)).unwrap();
        assert_eq!(
            anchored.phase(&[HistoryItem::user("hi"), HistoryItem::assistant("ok")]),
            AnchoredPhase::Promoted
        );
        assert_eq!(
            anchored.phase(&[HistoryItem::user("hi"), tool_calls()]),
            AnchoredPhase::Promoted
        );
        assert_eq!(
            anchored.phase(&[HistoryItem::user("hi"), tool_output()]),
            AnchoredPhase::Promoted
        );
    }

    #[test]
    fn phase_compaction_falls_back_until_new_signal() {
        let anchored = AnchoredBootstrap::from_config(&config(PromoteOn::Either)).unwrap();
        let compacted = vec![
            HistoryItem::user("old"),
            HistoryItem::assistant("old reply"),
            HistoryItem::context_summary("compressed"),
            HistoryItem::user("continue"),
        ];
        assert_eq!(anchored.phase(&compacted), AnchoredPhase::CompactedFallback);

        let re_promoted = vec![
            HistoryItem::user("old"),
            HistoryItem::assistant("old reply"),
            HistoryItem::context_summary("compressed"),
            HistoryItem::user("continue"),
            HistoryItem::assistant("new reply"),
        ];
        assert_eq!(anchored.phase(&re_promoted), AnchoredPhase::Promoted);
    }

    #[test]
    fn phase_respects_promote_on_mode() {
        let tool_only = AnchoredBootstrap::from_config(&config(PromoteOn::ToolCall)).unwrap();
        // Pure-text first reply does not promote under tool-call.
        assert_eq!(
            tool_only.phase(&[HistoryItem::user("hi"), HistoryItem::assistant("ok")]),
            AnchoredPhase::Bootstrap
        );
        assert_eq!(
            tool_only.phase(&[HistoryItem::user("hi"), tool_output()]),
            AnchoredPhase::Promoted
        );

        let message_only =
            AnchoredBootstrap::from_config(&config(PromoteOn::AssistantMessage)).unwrap();
        assert_eq!(
            message_only.phase(&[HistoryItem::user("hi"), tool_output()]),
            AnchoredPhase::Bootstrap
        );
        assert_eq!(
            message_only.phase(&[HistoryItem::user("hi"), HistoryItem::assistant("ok")]),
            AnchoredPhase::Promoted
        );
    }

    #[test]
    fn tool_catalog_bootstrap_exposes_alias_pair() {
        let anchored = AnchoredBootstrap::from_config(&config(PromoteOn::Either)).unwrap();
        let catalog = anchored.tool_catalog(&AnchoredPhase::Bootstrap, full_catalog());
        let names: Vec<_> = catalog.iter().map(|spec| spec.name.as_str()).collect();
        assert_eq!(names, vec![ALIAS_BASH, ALIAS_STR_REPLACE_EDITOR]);
    }

    #[test]
    fn tool_catalog_fallback_adds_compaction_tools() {
        let anchored = AnchoredBootstrap::from_config(&config(PromoteOn::Either)).unwrap();
        let catalog = anchored.tool_catalog(&AnchoredPhase::CompactedFallback, full_catalog());
        let names: Vec<_> = catalog.iter().map(|spec| spec.name.as_str()).collect();
        assert_eq!(
            names,
            vec![
                ALIAS_BASH,
                ALIAS_STR_REPLACE_EDITOR,
                "fs__read",
                "workflow__todos"
            ]
        );
    }

    #[test]
    fn tool_catalog_fallback_never_duplicates_alias_pair() {
        let mut cfg = config(PromoteOn::Either);
        // Adversarial config listing the real tools the aliases map to.
        cfg.compaction_tools = vec!["shell__exec".into(), "edit__apply_patch".into()];
        let anchored = AnchoredBootstrap::from_config(&cfg).unwrap();
        let catalog = anchored.tool_catalog(&AnchoredPhase::CompactedFallback, full_catalog());
        let names: Vec<_> = catalog.iter().map(|spec| spec.name.as_str()).collect();
        assert_eq!(names, vec![ALIAS_BASH, ALIAS_STR_REPLACE_EDITOR]);
    }

    #[test]
    fn tool_catalog_promoted_returns_full_untouched() {
        let anchored = AnchoredBootstrap::from_config(&config(PromoteOn::Either)).unwrap();
        let full = full_catalog();
        let catalog = anchored.tool_catalog(&AnchoredPhase::Promoted, full.clone());
        assert_eq!(catalog, full);
    }

    #[test]
    fn alias_pair_tolerates_missing_real_tools() {
        let anchored = AnchoredBootstrap::from_config(&config(PromoteOn::Either)).unwrap();
        let catalog = anchored.tool_catalog(&AnchoredPhase::Bootstrap, vec![spec("fs__read")]);
        assert!(catalog.is_empty());
    }

    #[test]
    fn resolve_tool_name_maps_aliases_unconditionally() {
        let anchored = AnchoredBootstrap::from_config(&config(PromoteOn::Either)).unwrap();
        assert_eq!(anchored.resolve_tool_name(ALIAS_BASH), "shell__exec");
        assert_eq!(
            anchored.resolve_tool_name(ALIAS_STR_REPLACE_EDITOR),
            "edit__apply_patch"
        );
        assert_eq!(anchored.resolve_tool_name("fs__read"), "fs__read");
    }

    fn persona_base() -> Vec<PromptMessage> {
        vec![
            PromptMessage::system("你是运行在本地仓库中的编程代理。"),
            PromptMessage::system("来自 /tmp/AGENTS.md 的指令：\nworkspace rules"),
        ]
    }

    #[test]
    fn prelude_bootstrap_keeps_only_minimal_persona_and_manual() {
        let anchored = AnchoredBootstrap::from_config(&config(PromoteOn::Either)).unwrap();
        let manual = vec![PromptMessage::developer("manual skill material")];
        let prelude = anchored.prelude(
            &AnchoredPhase::Bootstrap,
            &persona_base(),
            Some(PromptMessage::developer("runtime")),
            Some(PromptMessage::developer("skill catalog")),
            Some(PromptMessage::developer("workflow")),
            &manual,
        );
        let texts: Vec<_> = prelude
            .iter()
            .map(|message| message.text.as_str())
            .collect();
        assert_eq!(texts, vec![MINIMAL_PERSONA, "manual skill material"]);
    }

    #[test]
    fn prelude_promoted_restores_injections_with_minimal_persona() {
        let anchored = AnchoredBootstrap::from_config(&config(PromoteOn::Either)).unwrap();
        let manual = vec![PromptMessage::developer("manual skill material")];
        let prelude = anchored.prelude(
            &AnchoredPhase::Promoted,
            &persona_base(),
            Some(PromptMessage::developer("runtime")),
            Some(PromptMessage::developer("skill catalog")),
            Some(PromptMessage::developer("workflow")),
            &manual,
        );
        let texts: Vec<_> = prelude
            .iter()
            .map(|message| message.text.as_str())
            .collect();
        assert_eq!(
            texts,
            vec![
                MINIMAL_PERSONA,
                "来自 /tmp/AGENTS.md 的指令：\nworkspace rules",
                "runtime",
                "skill catalog",
                "manual skill material",
                "workflow",
            ]
        );
    }

    #[test]
    fn from_config_and_enabled_for_whitelist() {
        assert!(AnchoredBootstrap::from_config(&config(PromoteOn::Either)).is_some());
        let disabled = AnchoredBootstrapConfig {
            enabled: false,
            ..config(PromoteOn::Either)
        };
        assert!(AnchoredBootstrap::from_config(&disabled).is_none());
        let anchored = AnchoredBootstrap::from_config(&config(PromoteOn::Either)).unwrap();
        assert!(anchored.enabled_for("deepseek-v4-pro"));
        assert!(!anchored.enabled_for("gpt-5.6-sol"));
    }
}
