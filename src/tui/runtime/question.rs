use super::*;

impl TuiRuntime {
    pub(super) fn begin_pending_question(
        &mut self,
        request: crate::tool::QuestionRequest,
        handle: RunnerQuestionRequest,
        child_session_id: Option<String>,
    ) -> Result<()> {
        if self.state.pending_question.is_some() || self.permission_lifecycle.is_pending() {
            return Err(anyhow!("interactive request already pending"));
        }

        let origin_label = child_session_id
            .as_ref()
            .map(|_| "Child question".to_string());
        self.state.pending_question = Some(PendingQuestionState::new(request, origin_label));
        self.pending_question_handle = Some(handle);
        self.pending_question_child_session_id = child_session_id;
        self.state.phase = crate::tui::state::AppPhase::WaitingForPermission;
        self.state.toast = None;
        Ok(())
    }

    pub(super) fn clear_pending_question(&mut self) {
        self.state.pending_question = None;
        self.pending_question_handle = None;
        self.pending_question_child_session_id = None;
        if matches!(
            self.state.phase,
            crate::tui::state::AppPhase::WaitingForPermission
        ) {
            self.state.phase = crate::tui::state::AppPhase::Running;
        }
        self.state.sync_input_phase();
    }

    pub(super) fn is_stale_question_interaction(error: &anyhow::Error) -> bool {
        matches!(
            error.to_string().as_str(),
            "question response receiver dropped" | "question request already resolved"
        )
    }

    pub(super) fn cancel_pending_question(&mut self, reason: impl Into<String>) -> Result<()> {
        let reason = reason.into();
        let handle = self.pending_question_handle.take();
        self.clear_pending_question();
        if let Some(handle) = handle
            && let Err(error) = handle.cancel(reason.clone())
        {
            if Self::is_stale_question_interaction(&error) {
                tracing::warn!(error = %error, "ignored stale question cancellation");
            } else {
                return Err(error);
            }
        }
        Ok(())
    }

    pub(super) fn cancel_pending_question_if_parent(&mut self, reason: &str) {
        if self.pending_question_child_session_id.is_none()
            && (self.state.pending_question.is_some() || self.pending_question_handle.is_some())
        {
            let _ = self.cancel_pending_question(reason);
        }
    }

    pub(super) fn submit_pending_question(&mut self) -> Result<()> {
        let unanswered_tab = self
            .state
            .pending_question
            .as_ref()
            .and_then(PendingQuestionState::first_unanswered_tab);

        if let Some(tab_index) = unanswered_tab {
            if let Some(question) = self.state.pending_question.as_mut() {
                question.focus_tab(tab_index);
            }
            self.state.show_toast(
                self.state.t("runtime.answer_all_questions"),
                ToastKind::Info,
            );
            return Ok(());
        }

        let Some(question) = self.state.pending_question.as_ref() else {
            self.state
                .show_toast(self.state.t("runtime.no_question_pending"), ToastKind::Info);
            return Ok(());
        };
        if question.has_invalid_single_response() {
            self.state
                .show_toast(self.state.t("runtime.single_select_only"), ToastKind::Info);
            return Ok(());
        }

        let response = question.build_response();
        let handle = self.pending_question_handle.take();
        self.state.toast = None;
        self.clear_pending_question();
        if let Some(handle) = handle
            && let Err(error) = handle.answer(response)
        {
            if Self::is_stale_question_interaction(&error) {
                tracing::warn!(error = %error, "ignored stale question answer");
            } else {
                return Err(error);
            }
            return Ok(());
        }
        Ok(())
    }

    pub(super) fn insert_pending_question_text(&mut self, text: &str) {
        if let Some(question) = self
            .state
            .pending_question
            .as_mut()
            .filter(|question| question.editing_custom)
            .and_then(PendingQuestionState::current_question_mut)
        {
            question.insert_custom_edit(text);
        }
    }

    pub(super) fn backspace_pending_question_text(&mut self) {
        if let Some(question) = self
            .state
            .pending_question
            .as_mut()
            .and_then(PendingQuestionState::current_question_mut)
        {
            question.backspace_custom_edit();
        }
    }

    pub(super) fn delete_pending_question_text(&mut self) {
        if let Some(question) = self
            .state
            .pending_question
            .as_mut()
            .and_then(PendingQuestionState::current_question_mut)
        {
            question.delete_custom_edit();
        }
    }

    pub(super) fn move_pending_question_cursor_left(&mut self) {
        if let Some(question) = self
            .state
            .pending_question
            .as_mut()
            .and_then(PendingQuestionState::current_question_mut)
        {
            question.move_custom_cursor_left();
        }
    }

    pub(super) fn move_pending_question_cursor_right(&mut self) {
        if let Some(question) = self
            .state
            .pending_question
            .as_mut()
            .and_then(PendingQuestionState::current_question_mut)
        {
            question.move_custom_cursor_right();
        }
    }

    pub(super) fn move_pending_question_cursor_home(&mut self) {
        if let Some(question) = self
            .state
            .pending_question
            .as_mut()
            .and_then(PendingQuestionState::current_question_mut)
        {
            question.move_custom_cursor_home();
        }
    }

    pub(super) fn move_pending_question_cursor_end(&mut self) {
        if let Some(question) = self
            .state
            .pending_question
            .as_mut()
            .and_then(PendingQuestionState::current_question_mut)
        {
            question.move_custom_cursor_end();
        }
    }
}
