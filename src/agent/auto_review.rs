use std::future::Future;
use std::pin::Pin;

use anyhow::Result;
use async_openai::config::Config;

use crate::agent::Agent;
use crate::permission::{PermissionApproval, PermissionRequest};

#[derive(Debug, Clone)]
pub struct AutoReviewResolution {
    pub approval: PermissionApproval,
    pub reason: String,
    #[allow(dead_code)]
    pub risk: Option<String>,
    #[allow(dead_code)]
    pub approval_label: &'static str,
    #[allow(dead_code)]
    pub reviewer_child_session_id: String,
}

pub trait AutoReviewService<C: Config>: Send + Sync {
    fn review<'a>(
        &'a self,
        parent: &'a Agent<C>,
        request: PermissionRequest,
        user_goal: Option<String>,
    ) -> Pin<Box<dyn Future<Output = Result<AutoReviewResolution>> + Send + 'a>>;

    fn clear_sticky(&self);

    /// Reset per-turn counters while keeping the sticky reviewer child session.
    fn begin_turn(&self) {
        // Default: no-op for mock services.
    }
}
