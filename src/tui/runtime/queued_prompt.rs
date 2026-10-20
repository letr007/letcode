use crate::user_content::UserMessageSubmission;

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(crate) struct QueuedPromptLifecycle {
    accepted: Option<UserMessageSubmission>,
}

impl QueuedPromptLifecycle {
    pub(crate) fn accepted_submission_id(&self) -> Option<&str> {
        self.accepted.as_ref().map(|prompt| prompt.id.as_str())
    }

    pub(crate) fn accept(&mut self, prompt: UserMessageSubmission) {
        self.accepted = Some(prompt);
    }

    pub(crate) fn take(&mut self) -> Option<UserMessageSubmission> {
        self.accepted.take()
    }

    pub(crate) fn resolve_user_message(&mut self, submission_id: &str) {
        if self.accepted_submission_id() == Some(submission_id) {
            self.accepted = None;
        }
    }

    pub(crate) fn reset(&mut self) {
        self.accepted = None;
    }
}
