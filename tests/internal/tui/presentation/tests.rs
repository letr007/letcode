use super::*;
#[test]
fn completed_question_is_never_hidden_as_a_quiet_success() {
    let policy = PresentationPolicy;
    let context = ToolTextPresentationContext::new("question", ToolPresentationStatus::Succeeded);

    assert_eq!(
        policy.tool_presentation_text(&context),
        ToolPresentation::CompactCard
    );
}

#[test]
fn peer_tools_use_cards_for_all_statuses() {
    let policy = PresentationPolicy;
    for name in [
        crate::tool_names::TOOL_PEER_SEND,
        crate::tool_names::TOOL_PEER_DISCOVER,
    ] {
        for status in [
            ToolPresentationStatus::Pending,
            ToolPresentationStatus::Running,
            ToolPresentationStatus::Succeeded,
            ToolPresentationStatus::Failed,
        ] {
            let mut context = ToolTextPresentationContext::new(name, status);
            for output in [None, Some("{\"status\":\"written\"}".into())] {
                context.output = output;
                assert_eq!(
                    policy.tool_presentation_text(&context),
                    ToolPresentation::CompactCard,
                    "{name} {status:?}"
                );
            }
        }
    }
}

#[test]
fn quiet_success_text_write_like_tools_are_never_hidden() {
    let policy = PresentationPolicy;
    let mut ctx =
        ToolTextPresentationContext::new("shell__exec", ToolPresentationStatus::Succeeded);
    ctx.output = Some("\n".into());
    assert_eq!(
        policy.tool_presentation_text(&ctx),
        ToolPresentation::CompactCard
    );
}
