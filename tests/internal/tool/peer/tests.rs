use super::*;
use crate::agent::{AgentFactory, AgentTemplate};
use crate::peer::PeerRuntime;

#[tokio::test]
async fn main_session_tools_discover_and_send_without_a_reply() {
    let directory = tempfile::tempdir().unwrap();
    let first = PeerRuntime::start(directory.path(), None, "first".into(), None, |_| {})
        .await
        .unwrap();
    let (messages, mut received) = tokio::sync::mpsc::unbounded_channel();
    let second = PeerRuntime::start(
        directory.path(),
        None,
        "second".into(),
        Some("Review".into()),
        move |message| {
            messages.send(message).unwrap();
        },
    )
    .await
    .unwrap();
    let discover = PeerDiscoverTool(first.client())
        .execute(json!({}))
        .await
        .unwrap();
    assert_eq!(discover["peers"].as_array().unwrap().len(), 1);
    assert_eq!(discover["peers"][0]["address"]["session_id"], "second");
    let result = PeerSendTool(first.client())
        .execute(json!({
            "to": second.client().info().address,
            "message": "Review the interface"
        }))
        .await
        .unwrap();
    assert_eq!(result["status"], "written");
    let message = tokio::time::timeout(std::time::Duration::from_secs(2), received.recv())
        .await
        .unwrap()
        .unwrap();
    assert_eq!(message.from, first.client().info().address);
    assert_eq!(message.message, "Review the interface");

    let mut parent = Agent::new("model", None, None);
    register(&mut parent, first.client()).unwrap();
    let child = AgentFactory::create_child(&parent, &AgentTemplate::general());
    let tools = child.tool_definitions_for_test();
    assert!(
        !tools
            .iter()
            .any(|tool| tool.name == tool_names::TOOL_PEER_DISCOVER)
    );
    assert!(
        !tools
            .iter()
            .any(|tool| tool.name == tool_names::TOOL_PEER_SEND)
    );
}
