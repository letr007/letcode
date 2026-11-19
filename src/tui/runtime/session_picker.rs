use super::*;

impl TuiRuntime {
    pub(super) fn session_rows(
        &self,
        sessions: &[SessionSummary],
        scope: SessionPickerScope,
    ) -> Vec<DialogItem> {
        session_dialog_items(
            sessions,
            self.workspace_key.as_deref(),
            scope,
            &self.state.t("dialog.session_unassigned"),
        )
    }

    pub(super) fn toggle_session_scope(&mut self) {
        let Some(scope) = self
            .state
            .dialog()
            .filter(|dialog| dialog.kind == DialogKind::SessionPicker)
            .map(|dialog| dialog.session_scope.toggled())
        else {
            return;
        };
        let items = self.session_rows(&self.session_summaries, scope);
        if let Some(dialog) = self.state.dialog_mut() {
            dialog.session_scope = scope;
            dialog.replace_items(items);
        }
    }

    pub(super) fn show_resume_dialog(&mut self) -> Result<Option<SubmittedCommand>> {
        if self.session_list_rx.is_some() {
            return Ok(Some(SubmittedCommand::LocalOnly));
        }

        let (tx, rx) = mpsc::unbounded_channel();
        self.session_list_rx = Some(rx);
        let sessions_dir = self.sessions_dir.clone();
        // ponytail: std thread + try_recv keeps the frame loop free without a
        // dedicated async worker type; swap to spawn_blocking if we already hold
        // a Handle in more places.
        std::thread::spawn(move || {
            let _ = tx.send(merged_session_summaries(&sessions_dir));
        });
        Ok(Some(SubmittedCommand::LocalOnly))
    }

    pub(super) fn show_rename_dialog(&mut self) -> Result<Option<SubmittedCommand>> {
        let mut dialog = DialogState::new(
            DialogKind::SessionRename,
            self.state.t("dialog.rename_title"),
            None,
            Vec::new(),
        );
        dialog.query = self.session_title.clone().unwrap_or_default();
        self.state.open_dialog(dialog);
        Ok(Some(SubmittedCommand::LocalOnly))
    }

    pub(super) fn show_session_delete_confirm(&mut self) -> Result<Option<RuntimeCommand>> {
        let Some((session_id, label)) = self.state.dialog().and_then(|dialog| {
            if dialog.kind != DialogKind::SessionPicker {
                return None;
            }
            dialog
                .selected_item()
                .map(|item| (item.id.clone(), item.label.clone()))
        }) else {
            return Ok(None);
        };

        let title = self.state.t("dialog.session_delete_title");
        let description = self
            .state
            .t_fmt("dialog.session_delete_body", &[("session", &label)]);
        let confirm_label = self.state.t("dialog.session_delete_confirm");
        let cancel_label = self.state.t("ui.cancel");
        let mut dialog = DialogState::new(
            DialogKind::SessionDeleteConfirm,
            title,
            Some(description),
            vec![
                DialogItem::new(session_id, confirm_label, None),
                DialogItem::new(SESSION_DELETE_CANCEL_ID, cancel_label, None),
            ],
        );
        dialog.selected = 1;
        self.state.open_dialog(dialog);
        Ok(None)
    }
    pub(super) fn show_child_members_dialog(&mut self) -> Result<()> {
        let Some(root_session_id) = self.state.session_id.as_deref() else {
            return Ok(());
        };
        let members = crate::session::child_view::project_session_members(
            &self.sessions_dir,
            root_session_id,
        )?;
        let mut items = vec![DialogItem::new(
            root_session_id,
            self.state.t("runtime.root_session"),
            Some(root_session_id.to_string()),
        )];
        items.extend(members.into_iter().map(|member| {
            DialogItem::new(
                member.child_session_id.clone(),
                format!("#{} {}", member.pool_ordinal, member.agent_name),
                Some(format!("{} · {}", member.status, member.child_session_id)),
            )
        }));
        let mut dialog = DialogState::new(
            DialogKind::ChildPicker,
            self.state.t("runtime.child_members"),
            Some(self.state.t("runtime.child_members_hint")),
            items,
        );
        if let Some(child_session_id) = child_navigation_anchor(&self.state) {
            dialog.selected = dialog
                .items
                .iter()
                .position(|item| item.id == child_session_id)
                .unwrap_or(0);
        }
        self.state.open_dialog(dialog);
        Ok(())
    }

    pub(super) fn open_history_tree_dialog(
        &mut self,
        entries: &[transcript_projection::SessionHistoryEntry],
    ) {
        if entries.is_empty() {
            self.push_command_notice(self.state.t("runtime.no_transcript_entries"));
            return;
        }

        let mut dialog = DialogState::new(
            DialogKind::HistoryTree,
            self.state.t("runtime.session_history"),
            Some(self.state.t("runtime.select_entry")),
            history_tree_dialog_items(entries),
        );
        dialog.selected = dialog.items.len().saturating_sub(1);
        self.state.open_dialog(dialog);
    }

    pub(super) fn show_thoughts_dialog(&mut self) -> Result<Option<SubmittedCommand>> {
        let mut dialog = DialogState::new(
            DialogKind::ThoughtsPicker,
            self.state.t("runtime.thinking_display_title"),
            Some(self.state.t("runtime.thinking_display_description")),
            vec![
                DialogItem::new(
                    ThoughtsDisplayMode::Compact.as_str(),
                    self.state.t("runtime.thinking_level_compact"),
                    Some(self.state.t("runtime.thinking_level_compact_desc")),
                ),
                DialogItem::new(
                    ThoughtsDisplayMode::Titles.as_str(),
                    self.state.t("runtime.thinking_level_titles"),
                    Some(self.state.t("runtime.thinking_level_titles_desc")),
                ),
                DialogItem::new(
                    ThoughtsDisplayMode::Scroll.as_str(),
                    self.state.t("runtime.thinking_level_scroll"),
                    Some(self.state.t("runtime.thinking_level_scroll_desc")),
                ),
                DialogItem::new(
                    ThoughtsDisplayMode::Full.as_str(),
                    self.state.t("runtime.thinking_level_full"),
                    Some(self.state.t("runtime.thinking_level_full_desc")),
                ),
            ],
        );
        dialog.selected = match self.state.thoughts_display {
            ThoughtsDisplayMode::Compact => 0,
            ThoughtsDisplayMode::Titles => 1,
            ThoughtsDisplayMode::Scroll => 2,
            ThoughtsDisplayMode::Full => 3,
        };
        self.state.open_dialog(dialog);
        Ok(Some(SubmittedCommand::LocalOnly))
    }

    pub(super) fn show_tools_dialog(&mut self) -> Result<Option<SubmittedCommand>> {
        let mut dialog = DialogState::new(
            DialogKind::ToolsPicker,
            self.state.t("runtime.tools_display_title"),
            Some(self.state.t("runtime.tools_display_description")),
            vec![
                DialogItem::new(
                    ToolsDisplayMode::Compact.as_str(),
                    self.state.t("runtime.tools_level_compact"),
                    Some(self.state.t("runtime.tools_level_compact_desc")),
                ),
                DialogItem::new(
                    ToolsDisplayMode::Detailed.as_str(),
                    self.state.t("runtime.tools_level_detailed"),
                    Some(self.state.t("runtime.tools_level_detailed_desc")),
                ),
            ],
        );
        dialog.selected = match self.state.tools_display {
            ToolsDisplayMode::Compact => 0,
            ToolsDisplayMode::Detailed => 1,
        };
        self.state.open_dialog(dialog);
        Ok(Some(SubmittedCommand::LocalOnly))
    }

    pub(super) fn show_reasoning_dialog(&mut self) -> Result<Option<SubmittedCommand>> {
        let efforts = self.active_reasoning_efforts();
        if efforts.is_empty() {
            self.push_command_notice(self.state.t("runtime.no_configurable_reasoning"));
            return Ok(Some(SubmittedCommand::LocalOnly));
        }
        let mut dialog = DialogState::new(
            DialogKind::ReasoningPicker,
            self.state.t("runtime.reasoning_title"),
            Some(self.state.t("runtime.reasoning_description")),
            reasoning_dialog_items(&efforts),
        );
        dialog.selected =
            reasoning_dialog_selected_index(&efforts, self.current_reasoning_effort());
        self.state.open_dialog(dialog);
        Ok(Some(SubmittedCommand::LocalOnly))
    }

    pub(super) fn show_context_dialog(&mut self) -> Result<Option<SubmittedCommand>> {
        let items = crate::tui::state::context_dialog_items(self.state.active_context());
        if items.is_empty() {
            self.push_command_notice(self.state.t("runtime.no_context_details"));
            return Ok(Some(SubmittedCommand::LocalOnly));
        }
        let mut dialog = DialogState::new(
            DialogKind::ContextPicker,
            self.state.t("runtime.context"),
            None,
            items,
        );
        select_active_context_item(
            &mut dialog,
            self.state.active_context().open_detail.as_ref(),
        );
        self.state.open_dialog(dialog);
        self.state.sync_context_picker_preview();
        Ok(Some(SubmittedCommand::LocalOnly))
    }

    pub(super) fn show_mcp_dialog(&mut self) -> Result<Option<SubmittedCommand>> {
        self.show_mcp_dialog_with_state(String::new(), None);
        Ok(Some(SubmittedCommand::LocalOnly))
    }

    pub(super) fn show_mcp_dialog_with_state(
        &mut self,
        query: String,
        selected_server: Option<String>,
    ) {
        let description = mcp_discovery_description(self.state.mcp_discovery);
        let mut dialog = DialogState::new(
            DialogKind::McpPicker,
            self.state.t("runtime.mcp_servers"),
            description,
            mcp_dialog_items(
                &self.state.mcp_servers,
                &self.state.mcp_updating,
                self.state.language(),
            ),
        );
        dialog.query = query;
        if let Some(server_name) = selected_server
            && let Some(index) = dialog.items.iter().position(|item| item.id == server_name)
        {
            dialog.selected = index;
        }
        self.state.open_dialog(dialog);
    }

    pub(super) fn refresh_open_mcp_dialog(&mut self) {
        let items = mcp_dialog_items(
            &self.state.mcp_servers,
            &self.state.mcp_updating,
            self.state.language(),
        );
        let description = mcp_discovery_description(self.state.mcp_discovery);
        let Some(dialog) = self
            .state
            .dialog_mut()
            .filter(|dialog| dialog.kind == DialogKind::McpPicker)
        else {
            return;
        };

        let selected_id = dialog
            .items
            .get(dialog.selected)
            .map(|item| item.id.clone());
        dialog.items = items;
        dialog.description = description;
        dialog.selected = selected_id
            .as_deref()
            .and_then(|id| dialog.items.iter().position(|item| item.id == id))
            .unwrap_or_else(|| dialog.selected.min(dialog.items.len().saturating_sub(1)));
    }

    pub(super) fn show_skill_dialog(&mut self) -> Result<Option<SubmittedCommand>> {
        self.state.open_dialog(DialogState::new(
            DialogKind::SkillPicker,
            self.state.t("ui.skills_title"),
            None,
            skill_dialog_items(&self.state.skill_cards),
        ));
        Ok(Some(SubmittedCommand::LocalOnly))
    }

    pub(super) fn show_mcp_tools_dialog(&mut self, server_name: String) {
        let (primary_query, primary_selected_server) = self
            .state
            .dialog()
            .filter(|dialog| dialog.kind == DialogKind::McpPicker)
            .map(|dialog| {
                (
                    dialog.query.clone(),
                    dialog.selected_item().map(|item| item.id.clone()),
                )
            })
            .unwrap_or_default();
        let tools = self
            .state
            .mcp_server_tools
            .get(&server_name)
            .cloned()
            .unwrap_or_default();
        let description = self.mcp_tools_dialog_description(&server_name);
        let mut dialog = DialogState::new(
            DialogKind::McpToolsPicker,
            self.state
                .t_fmt("runtime.mcp_tools", &[("server", &server_name)]),
            description,
            mcp_tool_dialog_items(&tools),
        );
        dialog.mcp_server_name = Some(server_name);
        dialog.mcp_primary_query = Some(primary_query);
        dialog.mcp_primary_selected_server = primary_selected_server;
        self.state.open_dialog(dialog);
    }

    pub(super) fn mcp_tools_dialog_description(&self, server_name: &str) -> Option<String> {
        self.state
            .mcp_servers
            .iter()
            .find(|server| server.name == server_name)
            .map(|server| match &server.status {
                mcp::McpServerStatus::Disabled => self.state.t("runtime.mcp_disabled"),
                mcp::McpServerStatus::Online { tool_count } => self
                    .state
                    .t_fmt("runtime.mcp_online", &[("count", &tool_count.to_string())]),
                mcp::McpServerStatus::Offline { .. } => self.state.t("runtime.mcp_offline"),
            })
    }

    pub(super) fn refresh_open_mcp_tools_dialog(&mut self, server_name: &str) {
        let tools = self
            .state
            .mcp_server_tools
            .get(server_name)
            .cloned()
            .unwrap_or_default();
        let description = self.mcp_tools_dialog_description(server_name);
        let Some(dialog) = self.state.dialog_mut().filter(|dialog| {
            dialog.kind == DialogKind::McpToolsPicker
                && dialog.mcp_server_name.as_deref() == Some(server_name)
        }) else {
            return;
        };

        let selected_id = dialog.selected_item().map(|item| item.id.clone());
        dialog.items = mcp_tool_dialog_items(&tools);
        dialog.description = description;
        dialog.selected = selected_id
            .as_deref()
            .and_then(|id| dialog.items.iter().position(|item| item.id == id))
            .unwrap_or_else(|| dialog.selected.min(dialog.items.len().saturating_sub(1)));
    }

    pub(super) fn handle_mcp_toggle(&mut self) -> Result<Option<RuntimeCommand>> {
        let Some(server_name) = self
            .state
            .dialog()
            .filter(|dialog| dialog.kind == DialogKind::McpPicker)
            .and_then(|dialog| dialog.selected_item())
            .map(|item| item.id.clone())
        else {
            return Ok(None);
        };
        if self.state.mcp_updating.contains(&server_name) {
            self.show_toast(
                self.state.t("runtime.mcp_update_progress"),
                ToastKind::Error,
            );
            return Ok(None);
        }
        if !self.engine_turn_is_active() {
            self.state
                .set_mcp_server_updating(server_name.clone(), true);
            self.refresh_open_mcp_dialog();
        }
        Ok(Some(RuntimeCommand::ToggleMcpServer(server_name)))
    }
}
