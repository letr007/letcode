use std::future::Future;
use std::pin::Pin;

use crate::agent::Agent;
use crate::permission::{PermissionApproval, PermissionRequest};
use anyhow::Result;

#[derive(Debug, Clone)]
pub struct AutoReviewResolution {
    pub approval: PermissionApproval,
    pub reason: String,
}

pub trait AutoReviewService: Send + Sync {
    fn review<'a>(
        &'a self,
        parent: &'a Agent,
        request: PermissionRequest,
        user_goal: Option<String>,
    ) -> Pin<Box<dyn Future<Output = Result<AutoReviewResolution>> + Send + 'a>>;

    fn clear_sticky(&self);
}
