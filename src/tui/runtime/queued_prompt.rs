use crate::user_content::UserMessageSubmission;

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum QueuedPromptDoneDisposition {
    ReadyForNextDispatch,
    PreserveInFlight,
    ConsumeFailedAcceptedPrompt(UserMessageSubmission),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum QueuedPromptLifecycle {
    Idle { dispatch_ready: bool },
    InFlight(QueuedPromptHandoff),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct QueuedPromptHandoff {
    prompt: UserMessageSubmission,
    accepted: bool,
    failed_after_accept: bool,
}

impl QueuedPromptLifecycle {
    pub(crate) fn idle(dispatch_ready: bool) -> Self {
        Self::Idle { dispatch_ready }
    }

    pub(crate) fn is_dispatch_ready(&self) -> bool {
        matches!(
            self,
            Self::Idle {
                dispatch_ready: true
            }
        )
    }

    pub(crate) fn has_inflight_handoff(&self) -> bool {
        matches!(self, Self::InFlight(_))
    }

    pub(crate) fn dispatched_prompt(&self) -> Option<&str> {
        match self {
            Self::Idle { .. } => None,
            Self::InFlight(handoff) => Some(handoff.prompt.text()),
        }
    }

    pub(crate) fn dispatched_submission_id(&self) -> Option<&str> {
        match self {
            Self::Idle { .. } => None,
            Self::InFlight(handoff) => Some(handoff.prompt.id.as_str()),
        }
    }

    pub(crate) fn is_accepted(&self) -> bool {
        matches!(
            self,
            Self::InFlight(QueuedPromptHandoff { accepted: true, .. })
        )
    }

    pub(crate) fn failed_after_accept(&self) -> bool {
        matches!(
            self,
            Self::InFlight(QueuedPromptHandoff {
                failed_after_accept: true,
                ..
            })
        )
    }

    pub(crate) fn mark_dispatch_ready(&mut self) {
        if let Self::Idle { dispatch_ready } = self {
            *dispatch_ready = true;
        }
    }

    pub(crate) fn clear_dispatch_ready(&mut self) {
        if let Self::Idle { dispatch_ready } = self {
            *dispatch_ready = false;
        }
    }

    pub(crate) fn dispatch(&mut self, prompt: UserMessageSubmission) {
        *self = Self::InFlight(QueuedPromptHandoff {
            prompt,
            accepted: false,
            failed_after_accept: false,
        });
    }

    pub(crate) fn accept(&mut self, submission_id: &str) {
        if let Self::InFlight(handoff) = self
            && handoff.prompt.id == submission_id
        {
            handoff.accepted = true;
        }
    }

    pub(crate) fn record_error(&mut self) {
        if let Self::InFlight(handoff) = self
            && handoff.accepted
        {
            handoff.failed_after_accept = true;
        }
    }

    pub(crate) fn done_disposition(&self) -> QueuedPromptDoneDisposition {
        match self {
            Self::Idle { .. } => QueuedPromptDoneDisposition::ReadyForNextDispatch,
            Self::InFlight(handoff) if handoff.accepted && handoff.failed_after_accept => {
                QueuedPromptDoneDisposition::ConsumeFailedAcceptedPrompt(handoff.prompt.clone())
            }
            Self::InFlight(_) => QueuedPromptDoneDisposition::PreserveInFlight,
        }
    }

    pub(crate) fn resolve_user_message(&mut self, submission_id: &str) -> bool {
        match self {
            Self::InFlight(handoff) if handoff.prompt.id == submission_id => {
                *self = Self::idle(false);
                true
            }
            _ => false,
        }
    }

    pub(crate) fn reset(&mut self) {
        *self = Self::default();
    }
}

impl Default for QueuedPromptLifecycle {
    fn default() -> Self {
        Self::Idle {
            dispatch_ready: false,
        }
    }
}
