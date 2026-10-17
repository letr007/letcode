mod pool;
mod result;
mod route_factory;
pub(crate) mod swarm;
pub use swarm::SwarmMember;

pub(crate) use pool::generate_run_id as generate_message_id;
pub(crate) use pool::{MessageDelegateFactory, MessageHost, SubagentCapacityExceeded};
pub use pool::{SubagentJob, SubagentPool, SubagentRunGovernance};
pub(crate) use result::REPORT_PROMPT;
pub use result::{
    StructuredSubagentResult, SubagentFailureKind, SubagentRunSummary, SubagentStatus,
    try_parse_structured_subagent_result,
};
pub use route_factory::ExpertRouteFactory;

#[cfg(test)]
use crate::agent::{Agent, AgentFactory, AgentTemplate, PrimaryRouteFactory, SubagentChildFactory};
#[cfg(test)]
use crate::config::{ModelRoute, ProviderConfig, RetryConfig};
#[cfg(test)]
use crate::tool::NormalizedSubagentInput;
#[cfg(test)]
use crate::transcript::TranscriptRecorder;
#[cfg(test)]
use anyhow::Result;
#[cfg(test)]
use futures_util::FutureExt;
#[cfg(test)]
use pool::generate_run_id;
#[cfg(test)]
use result::{build_completed_summary, build_runtime_summary, classify_failure_status};
#[cfg(test)]
use std::path::PathBuf;
#[cfg(test)]
use std::sync::{Arc, Mutex};

#[cfg(test)]
#[path = "../tests/internal/subagent/tests.rs"]
mod tests;
