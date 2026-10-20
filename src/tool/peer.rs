use anyhow::Result;
use async_trait::async_trait;
use serde::Deserialize;
use serde_json::{Value, json};

use super::{ToolHandler, ToolParallelism};
use crate::agent::Agent;
use crate::peer::{PeerAddress, PeerClient};
use crate::tool_names;

pub(crate) fn register(agent: &mut Agent, client: PeerClient) -> Result<()> {
    agent.try_register_tool(PeerDiscoverTool(client.clone()))?;
    agent.try_register_tool(PeerSendTool(client))
}

struct PeerDiscoverTool(PeerClient);
struct PeerSendTool(PeerClient);

#[async_trait]
impl ToolHandler for PeerDiscoverTool {
    fn name(&self) -> &'static str {
        tool_names::TOOL_PEER_DISCOVER
    }

    fn description(&self) -> &'static str {
        "Discover other registered local letcode main sessions sharing this configuration directory. Returns their reply addresses, workspaces and titles. Registrations may be stale; sending can fail."
    }

    fn parameters(&self) -> Value {
        json!({"type": "object", "properties": {}, "required": [], "additionalProperties": false})
    }

    fn parallelism(&self) -> ToolParallelism {
        ToolParallelism::Parallel
    }

    async fn execute(&self, _args: Value) -> Result<Value> {
        let own = self.0.info();
        let mut peers = self.0.discover()?;
        peers.retain(|peer| peer.address.instance_id != own.address.instance_id);
        peers.sort_by_key(|peer| peer.workspace != own.workspace);
        Ok(json!({"self": own.address, "peers": peers}))
    }
}

#[derive(Deserialize)]
struct SendArguments {
    to: PeerAddress,
    message: String,
}

#[async_trait]
impl ToolHandler for PeerSendTool {
    fn name(&self) -> &'static str {
        tool_names::TOOL_PEER_SEND
    }

    fn description(&self) -> &'static str {
        "Send a text message to another local letcode main session over a single short connection. Use its address from peer__discover or an incoming message. Returns after the local write, without delivery confirmation or waiting for a reply. The recipient can reply with peer__send; continue your work or finish normally."
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "to": {
                    "type": "object",
                    "properties": {
                        "instance_id": {"type": "string"},
                        "session_id": {"type": "string"},
                        "generation": {"type": "integer", "minimum": 0}
                    },
                    "required": ["instance_id", "session_id", "generation"],
                    "additionalProperties": false
                },
                "message": {"type": "string", "description": "Text to send; identify the requested work and whether a reply is needed"}
            },
            "required": ["to", "message"],
            "additionalProperties": false
        })
    }

    async fn execute(&self, args: Value) -> Result<Value> {
        let args: SendArguments = serde_json::from_value(args)?;
        self.0.send(args.to, args.message).await?;
        Ok(json!({"status": "written"}))
    }
}

#[cfg(test)]
#[path = "../../tests/internal/tool/peer/tests.rs"]
mod tests;
