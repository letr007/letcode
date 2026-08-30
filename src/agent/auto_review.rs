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
}

pub trait AutoReviewService<C: Config>: Send + Sync {
    fn review<'a>(
        &'a self,
        parent: &'a Agent<C>,
        request: PermissionRequest,
        user_goal: Option<String>,
    ) -> Pin<Box<dyn Future<Output = Result<AutoReviewResolution>> + Send + 'a>>;

    fn clear_sticky(&self);
}
