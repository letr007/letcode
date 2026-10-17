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
