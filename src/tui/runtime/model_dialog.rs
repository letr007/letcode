use super::*;

impl TuiRuntime {
    pub(super) fn apply_model_catalog_update(&mut self, catalog: &ModelCatalogUpdatedEvent) {
        self.available_models = catalog
            .models
            .iter()
            .map(AvailableModel::from_catalog_entry)
            .collect();

        let items = self.model_dialog_items();
        let Some(dialog) = self.state.dialog_mut() else {
            return;
        };
        if !matches!(
            dialog.kind,
            DialogKind::ModelPicker | DialogKind::ExpertModelPicker(_)
        ) {
            return;
        }
        let selected_id = dialog.selected_item().map(|item| item.id.clone());
        let checked_ids = matches!(dialog.kind, DialogKind::ExpertModelPicker(_)).then(|| {
            dialog
                .items
                .iter()
                .filter(|item| item.checked)
                .map(|item| item.id.clone())
                .collect::<std::collections::HashSet<_>>()
        });
        let query = dialog.query.clone();
        let old_selected = dialog.selected;
        dialog.items = items;
        if let Some(checked_ids) = checked_ids {
            for item in &mut dialog.items {
                item.checked = checked_ids.contains(&item.id);
            }
        }
        dialog.query = query;
        dialog.selected = selected_id
            .and_then(|id| dialog.items.iter().position(|item| item.id == id))
            .unwrap_or_else(|| old_selected.min(dialog.items.len().saturating_sub(1)));
    }

    pub(super) fn model_dialog_items(&self) -> Vec<DialogItem> {
        self.available_models
            .iter()
            .map(|model| {
                DialogItem::new(model.id.clone(), model.label.clone(), None)
                    .with_section(model.provider.clone())
            })
            .collect()
    }

    pub(super) fn show_model_dialog(&mut self) -> Result<Option<SubmittedCommand>> {
        let items = self.model_dialog_items();
        let mut dialog = DialogState::new(
            DialogKind::ModelPicker,
            self.state.t("runtime.select_model"),
            None,
            items,
        );
        let selected_model_id = self
            .state
            .pending_composer_settings
            .model
            .as_ref()
            .map(|(model_id, _)| model_id.as_str())
            .unwrap_or(&self.state.model_id);
        if let Some(index) = self
            .available_models
            .iter()
            .position(|model| model.id == selected_model_id)
        {
            dialog.selected = index;
        }
        self.state.open_dialog(dialog);
        Ok(Some(SubmittedCommand::LocalOnly))
    }

    pub(super) fn show_agents_dialog(&mut self) -> Result<Option<SubmittedCommand>> {
        self.show_agents_dialog_with_state(String::new(), None);
        Ok(Some(SubmittedCommand::LocalOnly))
    }

    pub(super) fn expert_allowed_models<'a>(&'a self, expert: &'a AvailableExpert) -> &'a [String] {
        self.pending_expert_allowed_models
            .get(&expert.agent_name)
            .map(Vec::as_slice)
            .unwrap_or(expert.allowed_models.as_slice())
    }

    pub(super) fn sync_open_expert_model_picker(&mut self, agent_name: &str) {
        let Some(allowed_models) = self
            .available_experts
            .iter()
            .find(|expert| expert.agent_name == agent_name)
            .map(|expert| self.expert_allowed_models(expert).to_vec())
        else {
            return;
        };
        if let Some(dialog) = self.state.dialog_mut().filter(|dialog| {
            matches!(
                &dialog.kind,
                DialogKind::ExpertModelPicker(open_agent) if open_agent == agent_name
            )
        }) {
            for item in &mut dialog.items {
                item.checked = allowed_models.contains(&item.id);
            }
        }
    }

    /// Rebuilds an open expert list so its route summaries track the experts' live routes.
    pub(super) fn refresh_open_agent_picker(&mut self) {
        let Some((query, selected_agent)) = self
            .state
            .dialog()
            .filter(|dialog| dialog.kind == DialogKind::AgentPicker)
            .map(|dialog| {
                (
                    dialog.query.clone(),
                    dialog.selected_item().map(|item| item.id.clone()),
                )
            })
        else {
            return;
        };
        self.show_agents_dialog_with_state(query, selected_agent);
    }

    pub(super) fn show_agents_dialog_with_state(
        &mut self,
        query: String,
        selected_agent: Option<String>,
    ) {
        let items = self
            .available_experts
            .iter()
            .map(|expert| {
                DialogItem::new(expert.agent_name.clone(), expert.agent_name.clone(), None)
                    .with_section(self.state.t("ui.experts"))
                    .with_right_detail(expert.model_summary(self.expert_allowed_models(expert)))
            })
            .collect();
        let mut dialog = DialogState::new(
            DialogKind::AgentPicker,
            self.state.t("ui.expert_models"),
            None,
            items,
        );
        dialog.query = query;
        if let Some(agent_name) = selected_agent
            && let Some(index) = dialog.items.iter().position(|item| item.id == agent_name)
        {
            dialog.selected = index;
        }
        self.state.open_dialog(dialog);
    }

    pub(super) fn show_expert_model_dialog(&mut self, agent_name: String) {
        let (primary_query, primary_selected_agent) = self
            .state
            .dialog()
            .filter(|dialog| dialog.kind == DialogKind::AgentPicker)
            .map(|dialog| {
                (
                    dialog.query.clone(),
                    dialog.selected_item().map(|item| item.id.clone()),
                )
            })
            .unwrap_or_default();
        let expert = self
            .available_experts
            .iter()
            .find(|expert| expert.agent_name == agent_name);
        let current_route = expert
            .map(|expert| expert.route_id.clone())
            .unwrap_or_else(|| self.state.model_id.clone());
        let allowed_models = expert
            .map(|expert| self.expert_allowed_models(expert).to_vec())
            .unwrap_or_default();
        let mut dialog = DialogState::new(
            DialogKind::ExpertModelPicker(agent_name),
            self.state.t("runtime.select_expert_model"),
            None,
            self.model_dialog_items()
                .into_iter()
                .map(|item| {
                    let checked = allowed_models.contains(&item.id);
                    item.with_checked(checked)
                })
                .collect(),
        );
        dialog.expert_primary_query = Some(primary_query);
        dialog.expert_primary_selected_agent = primary_selected_agent;
        if let Some(index) = dialog
            .items
            .iter()
            .position(|item| item.id == current_route)
        {
            dialog.selected = index;
        }
        self.state.open_dialog(dialog);
    }

    pub(super) fn apply_restored_model(&mut self, model_id: String) {
        if let Some(model) = self
            .available_models
            .iter()
            .find(|model| model.id == model_id)
            .cloned()
        {
            self.state.set_model(model.id.clone(), model.label.clone());
            self.state
                .set_model_context_window(model.context_window_tokens);
            self.state
                .set_reasoning_effort_label(Some(reasoning_effort_status_label(
                    model.reasoning_effort,
                )));
        } else {
            self.state.set_model(model_id.clone(), model_id);
            self.state.set_model_context_window(None);
            self.state.set_reasoning_effort_label(None);
        }
    }

    pub(super) fn handle_model_selection(
        &mut self,
        model_id: String,
    ) -> Result<Option<SubmittedCommand>> {
        let Some(model) = self
            .available_models
            .iter()
            .find(|model| model.id == model_id)
            .cloned()
        else {
            let available = self
                .available_models
                .iter()
                .map(|model| model.id.as_str())
                .collect::<Vec<_>>()
                .join(", ");
            self.push_command_notice(self.state.t_fmt(
                "runtime.unknown_model",
                &[("model", &model_id), ("available", &available)],
            ));
            return Ok(Some(SubmittedCommand::LocalOnly));
        };

        Ok(Some(SubmittedCommand::Runtime(RuntimeCommand::SetModel(
            model.id,
        ))))
    }
}
