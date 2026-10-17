use super::*;

#[test]
fn fake_set_maps_to_backend_and_show_stays_local() {
    for client in [
        crate::fake::FakeClient::Auto,
        crate::fake::FakeClient::Codex,
        crate::fake::FakeClient::Anthropic,
    ] {
        assert_eq!(
            SessionCommand::from_command_intent(crate::command::CommandIntent::Fake(
                crate::command::FakeCommand::Set(Some(client))
            )),
            Some(SessionCommand::SetFakeClient(Some(client)))
        );
    }
    assert_eq!(
        SessionCommand::from_command_intent(crate::command::CommandIntent::Fake(
            crate::command::FakeCommand::Show
        )),
        None
    );
    assert_eq!(
        SessionCommand::SetFakeClient(Some(crate::fake::FakeClient::Codex))
            .active_turn_disposition(),
        crate::session::ActiveTurnCommandDisposition::Defer
    );
}
