use super::*;
#[test]
fn historian_uses_the_existing_optional_expert_route_shape() {
    let raw: RawAgentsConfig = toml::from_str(
        "[historian]\nprovider=\"p\"\nmodel=\"small\"\nallowed_models=[\"p/small\"]",
    )
    .unwrap();
    let historian = raw.historian.unwrap();
    assert_eq!(historian.provider.as_deref(), Some("p"));
    assert_eq!(historian.model.as_deref(), Some("small"));
    assert!(AgentsConfig::default().config_for("historian").is_some());
    assert!(AgentsConfig::default().route_for("historian").is_none());
    assert!(crate::delegation::is_system_expert("historian"));
    assert!(
        !crate::delegation::DELEGATION_EXPERTS
            .iter()
            .any(|expert| expert.agent_name == "historian")
    );
}
