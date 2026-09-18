use std::future::Future;
use std::pin::Pin;

use crate::agent::Agent;
use crate::permission::{PermissionApproval, PermissionRequest};
use anyhow::Result;

/// The standing question an approval backend answers, verbatim.
pub(crate) const REVIEW_INSTRUCTIONS: &str = "Agent 在用户授权下于本项目内工作。用户通常只给出粗略目标。判断这次工具调用应当：execute（直接执行）、ask_user（先让执行方补充说明）、还是 refuse（拒绝执行）。state 没有把问题说清楚时选 ask_user。";

/// Criteria per outcome, shared so the expert route and a dedicated backend
/// cannot drift apart in wording.
pub(crate) const REVIEW_OUTCOME_CRITERIA: [(&str, &str); 3] = [
    (
        "execute",
        "普通的、可预测的工作，后果从调用本身就能看清，且不超出用户目标。可以无人值守地执行；出错也能被发现且容易撤销。",
    ),
    (
        "ask_user",
        "具有破坏性或难以撤销、越出项目范围、涉及凭据或机密、删除或发布数据，或安全性取决于 state 里没有给出的事实。需要执行方补充说明它为什么属于当前目标；补充后仍不明确的，才由用户决定。",
    ),
    (
        "refuse",
        "与用户目标相矛盾，或会造成用户不可能合理预期、且难以撤销的破坏。",
    ),
];

/// The decision vocabulary both backends answer in.
pub(crate) const REVIEW_DECISION_VOCABULARY: &str = "execute|ask_user|refuse";

/// What the reviewer decided about a call, or that the decision belongs to the user.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AutoReviewOutcome {
    AllowOnce,
    AllowAlways,
    /// The reviewer leaves the call to the user, who answers the permission request.
    Ask,
    Deny,
}

impl AutoReviewOutcome {
    /// The approval to apply, or `None` when the user has to answer.
    pub fn approval(self) -> Option<PermissionApproval> {
        match self {
            Self::AllowOnce => Some(PermissionApproval::AllowOnce),
            Self::AllowAlways => Some(PermissionApproval::AllowAlways),
            Self::Ask => None,
            Self::Deny => Some(PermissionApproval::Deny),
        }
    }

    /// Label recorded with a resolved decision.
    pub fn label(self) -> &'static str {
        match self {
            Self::AllowOnce => "once",
            Self::AllowAlways => "always",
            Self::Ask => "ask",
            Self::Deny => "deny",
        }
    }
}

#[derive(Debug, Clone)]
pub struct AutoReviewResolution {
    pub outcome: AutoReviewOutcome,
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
