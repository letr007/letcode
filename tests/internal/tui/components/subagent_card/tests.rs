use super::*;
use crate::tui::i18n::Language;

#[test]
fn every_reported_subagent_status_is_localized() {
    let translator = Translator::new(Language::ZhCn);
    let terminal = [
        crate::subagent::SubagentStatus::Running,
        crate::subagent::SubagentStatus::Completed,
        crate::subagent::SubagentStatus::Failed,
        crate::subagent::SubagentStatus::BudgetExhausted,
        crate::subagent::SubagentStatus::Cancelled,
        crate::subagent::SubagentStatus::TimedOut,
    ]
    .map(|status| status.as_str());
    let projected = [
        "preparing",
        "approval",
        "approved",
        "denied",
        "error",
        "interrupted",
    ];
    for raw in terminal.into_iter().chain(projected) {
        let text = subagent_status_text(raw, &translator);
        assert_ne!(text, raw, "{raw} is not localized");
    }
}

#[test]
fn unknown_subagent_status_passes_through() {
    let translator = Translator::new(Language::ZhCn);
    assert_eq!(
        subagent_status_text("changes_requested", &translator),
        "changes_requested"
    );
}

#[test]
fn every_state_flag_is_localized() {
    let translator = Translator::new(Language::ZhCn);
    for flag in ["background", "active", "malformed", "hard", "logical"] {
        assert_ne!(subagent_state_flag_label(flag, &translator), flag);
    }
}
