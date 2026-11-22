use super::*;

impl TuiRuntime {
    pub(super) fn config_dialog_items(&self) -> (Vec<DialogItem>, Vec<ConfigFieldRef>) {
        let Some(document) = self.config_draft.as_ref() else {
            return (Vec::new(), Vec::new());
        };
        let query = self
            .state
            .dialog()
            .map(|dialog| dialog.query.trim().to_string())
            .unwrap_or_default();
        if !query.is_empty() {
            return self.config_search_items(document, &query);
        }
        let Some(dialog) = self.state.dialog() else {
            return self.config_overview_items(document);
        };
        match dialog.config_tab {
            ConfigTab::Overview if dialog.config_path.is_empty() => {
                self.config_overview_items(document)
            }
            ConfigTab::Advanced if dialog.config_path.is_empty() => {
                self.config_advanced_items(document)
            }
            ConfigTab::Providers if dialog.config_path.len() == 2 => {
                self.config_provider_items(document, &dialog.config_path)
            }
            ConfigTab::Mcp if dialog.config_path.len() == 2 => {
                self.config_mcp_items(document, &dialog.config_path)
            }
            _ => self.config_level_items(document, &dialog.config_path),
        }
    }

    pub(super) fn config_tab_path(tab: ConfigTab) -> Vec<String> {
        tab.root_path()
            .iter()
            .map(|part| part.to_string())
            .collect()
    }

    pub(super) fn config_overview_items(
        &self,
        document: &toml_edit::DocumentMut,
    ) -> (Vec<DialogItem>, Vec<ConfigFieldRef>) {
        let (mut items, fields) = self.config_filtered_root_items(
            document,
            &[
                "active_provider",
                "permissions",
                "subagents",
                "providers",
                "agents",
                "mcp",
            ],
        );
        if let Some(item) = items.iter_mut().find(|item| item.id == "active_provider")
            && let Some(provider) = item.detail.clone()
            && let Some(model) = Self::config_active_model(document)
        {
            item.detail = Some(format!("{provider} · {model}"));
        }
        if let Some(item) = items
            .iter_mut()
            .find(|item| item.id == "mcp" || item.id == "new/mcp")
            && let Some(summary) = self.config_mcp_summary(document)
        {
            item.detail = Some(summary);
        }
        for (index, item) in items.iter_mut().enumerate() {
            item.section = Some(
                self.state
                    .t(if index < 3 {
                        "config.section_common"
                    } else {
                        "config.section_manage"
                    })
                    .to_string(),
            );
        }
        (items, fields)
    }

    pub(super) fn config_advanced_items(
        &self,
        document: &toml_edit::DocumentMut,
    ) -> (Vec<DialogItem>, Vec<ConfigFieldRef>) {
        self.config_filtered_root_items(
            document,
            &["permissions", "global", "subagents", "tools", "fake"],
        )
    }

    pub(super) fn config_filtered_root_items(
        &self,
        document: &toml_edit::DocumentMut,
        wanted: &[&str],
    ) -> (Vec<DialogItem>, Vec<ConfigFieldRef>) {
        let (items, fields) = self.config_level_items(document, &[]);
        let mut items = items.into_iter().map(Some).collect::<Vec<_>>();
        let mut fields = fields.into_iter().map(Some).collect::<Vec<_>>();
        let mut picked_items = Vec::new();
        let mut picked_fields = Vec::new();
        for name in wanted {
            let add_id = format!("new/{name}");
            let target = if items
                .iter()
                .any(|item| item.as_ref().is_some_and(|item| item.id == add_id))
            {
                add_id
            } else {
                (*name).to_string()
            };
            if let Some(index) = items
                .iter()
                .position(|item| item.as_ref().is_some_and(|item| item.id == target))
            {
                picked_items.push(items[index].take().expect("item"));
                picked_fields.push(fields[index].take().expect("field"));
            }
        }
        (picked_items, picked_fields)
    }

    pub(super) fn config_provider_items(
        &self,
        document: &toml_edit::DocumentMut,
        level: &[String],
    ) -> (Vec<DialogItem>, Vec<ConfigFieldRef>) {
        let (items, fields) = self.config_level_items(document, level);
        let mut rows = items.into_iter().zip(fields).collect::<Vec<_>>();
        rows.sort_by_key(|(item, _)| Self::config_provider_field_rank(&item.id));
        for (item, _) in &mut rows {
            let (section, _) = Self::config_provider_field_rank(&item.id);
            item.section = Some(self.state.t(section).to_string());
        }
        rows.into_iter().unzip()
    }

    pub(super) fn config_mcp_items(
        &self,
        document: &toml_edit::DocumentMut,
        level: &[String],
    ) -> (Vec<DialogItem>, Vec<ConfigFieldRef>) {
        let (items, fields) = self.config_level_items(document, level);
        let mut rows = items.into_iter().zip(fields).collect::<Vec<_>>();
        rows.sort_by_key(|(item, _)| Self::config_mcp_field_rank(&item.id));
        for (item, _) in &mut rows {
            let (section, _) = Self::config_mcp_field_rank(&item.id);
            item.section = Some(self.state.t(section).to_string());
        }
        rows.into_iter().unzip()
    }

    pub(super) fn config_mcp_field_rank(id: &str) -> (&'static str, u8) {
        match id.rsplit('/').next().unwrap_or_default() {
            "type" | "enabled" | "timeout" => ("config.section_basic", 0),
            "local" | "remote" => ("config.section_connection", 1),
            "command" | "environment" | "env" | "url" | "headers" | "oauth" => {
                ("config.section_connection", 2)
            }
            _ => ("config.section_other", 3),
        }
    }

    pub(super) fn config_provider_field_rank(id: &str) -> (&'static str, u8) {
        match id.rsplit('/').next().unwrap_or_default() {
            "protocol" | "default_model" | "flavor" => ("config.section_basic", 0),
            "auth" => ("config.section_auth", 1),
            "endpoints" | "transport" | "headers" | "query" => ("config.section_connection", 2),
            "models" => ("config.section_models", 3),
            "retry" => ("config.section_retry", 4),
            _ => ("config.section_other", 5),
        }
    }

    /// Rows for what is there, plus what the schema allows to add.
    pub(super) fn config_level_items(
        &self,
        document: &toml_edit::DocumentMut,
        level: &[String],
    ) -> (Vec<DialogItem>, Vec<ConfigFieldRef>) {
        let Some(table) = Self::config_table(document, level) else {
            let level_refs: Vec<&str> = level.iter().map(String::as_str).collect();
            if crate::config::table_accepts_entries(&level_refs) {
                let name = level.last().cloned().unwrap_or_default();
                let item = DialogItem::new(
                    format!("new/{}", level.join("/")),
                    self.state.t_fmt("config.new_entry", &[("table", &name)]),
                    None,
                )
                .with_description(Self::config_description(&self.state, &level_refs));
                return (vec![item], vec![ConfigFieldRef::NewEntry(level.join("/"))]);
            }
            if level.len() == 1 {
                let path = level.to_vec();
                let item = DialogItem::new(path.join("/"), path[0].clone(), None)
                    .with_description(Self::config_description(&self.state, &level_refs))
                    .with_right_detail("＋");
                return (vec![item], vec![ConfigFieldRef::NewField(path)]);
            }
            return (Vec::new(), Vec::new());
        };
        let baseline = self.config_baseline_document();
        let mut items = Vec::new();
        let mut fields = Vec::new();
        let mut present: Vec<String> = Vec::new();
        for (key, item) in table.iter() {
            present.push(key.to_string());
            let path: Vec<String> = level
                .iter()
                .cloned()
                .chain(std::iter::once(key.to_string()))
                .collect();
            if !Self::config_entry_visible(document, &path) {
                continue;
            }
            let path_refs: Vec<&str> = path.iter().map(String::as_str).collect();
            match item {
                toml_edit::Item::Table(_) => {
                    let row = DialogItem::new(
                        path.join("/"),
                        key.to_string(),
                        self.config_table_detail(document, &path),
                    )
                    .with_description(Self::config_description(&self.state, &path_refs))
                    .with_right_detail("›");
                    items.push(row);
                    fields.push(ConfigFieldRef::Table(path));
                }
                toml_edit::Item::Value(value) => {
                    let entry = crate::config::leaf_config_entry(&path, value);
                    let field = if entry.kind == crate::config::ConfigEntryKind::Array {
                        ConfigFieldRef::List(path.clone())
                    } else if Self::config_field_values(document, &path).is_some() {
                        ConfigFieldRef::Choice(path.clone())
                    } else {
                        ConfigFieldRef::Field(path.clone())
                    };
                    let affordance = Self::config_affordance(&entry, &field);
                    let row = DialogItem::new(path.join("/"), entry.label, Some(entry.display))
                        .with_description(Self::config_description(&self.state, &path_refs))
                        .with_modified(Self::config_field_modified(
                            document,
                            baseline.as_ref(),
                            &path,
                        ))
                        .with_right_detail(affordance);
                    items.push(row);
                    fields.push(field);
                }
                _ => {}
            }
        }
        let level_refs: Vec<&str> = level.iter().map(String::as_str).collect();
        for name in crate::config::schema_properties(&level_refs) {
            if present.contains(&name) {
                continue;
            }
            let path: Vec<String> = level
                .iter()
                .cloned()
                .chain(std::iter::once(name.clone()))
                .collect();
            if !Self::config_entry_visible(document, &path) {
                continue;
            }
            let path_refs: Vec<&str> = path.iter().map(String::as_str).collect();
            let field = if Self::config_field_values(document, &path).is_some() {
                if crate::config::field_is_array(&path_refs) {
                    ConfigFieldRef::List(path.clone())
                } else {
                    ConfigFieldRef::Choice(path.clone())
                }
            } else {
                ConfigFieldRef::NewField(path.clone())
            };
            let affordance = match field {
                ConfigFieldRef::Choice(_) | ConfigFieldRef::List(_) => "▸",
                _ => "＋",
            };
            let detail = Self::config_effective_missing_value(document, &path);
            let row = DialogItem::new(path.join("/"), name, detail)
                .with_description(Self::config_description(&self.state, &path_refs))
                .with_right_detail(affordance);
            items.push(row);
            fields.push(field);
        }
        let level_refs: Vec<&str> = level.iter().map(String::as_str).collect();
        if level.is_empty() {
            for table in crate::config::entry_tables(&[]) {
                if document.get(&table).is_some() {
                    continue;
                }
                let row = DialogItem::new(
                    format!("new/{table}"),
                    self.state.t_fmt("config.new_entry", &[("table", &table)]),
                    None,
                )
                .with_description(Self::config_description(&self.state, &[table.as_str()]));
                items.push(row);
                fields.push(ConfigFieldRef::NewEntry(table));
            }
        } else if crate::config::table_accepts_entries(&level_refs) {
            let name = level.last().cloned().unwrap_or_default();
            let row = DialogItem::new(
                format!("new/{}", level.join("/")),
                self.state.t_fmt("config.new_entry", &[("table", &name)]),
                None,
            )
            .with_description(Self::config_description(&self.state, &level_refs));
            items.push(row);
            fields.push(ConfigFieldRef::NewEntry(level.join("/")));
        }
        (items, fields)
    }

    /// Every matching leaf, flattened across tables, so search stays fast.
    pub(super) fn config_search_items(
        &self,
        document: &toml_edit::DocumentMut,
        query: &str,
    ) -> (Vec<DialogItem>, Vec<ConfigFieldRef>) {
        let needle = query.to_lowercase();
        let baseline = self.config_baseline_document();
        let mut items = Vec::new();
        let mut fields = Vec::new();
        for entry in crate::config::config_entries_in(document) {
            let modified = Self::config_field_modified(document, baseline.as_ref(), &entry.path);
            if !Self::config_entry_visible(document, &entry.path)
                || !entry.path.join(".").to_lowercase().contains(&needle)
            {
                continue;
            }
            let path: Vec<&str> = entry.path.iter().map(String::as_str).collect();
            let field = if entry.kind == crate::config::ConfigEntryKind::Array {
                ConfigFieldRef::List(entry.path.clone())
            } else if Self::config_field_values(document, &entry.path).is_some() {
                ConfigFieldRef::Choice(entry.path.clone())
            } else {
                ConfigFieldRef::Field(entry.path.clone())
            };
            let affordance = Self::config_affordance(&entry, &field);
            let section = entry.path.first().cloned().unwrap_or_default();
            items.push(
                DialogItem::new(entry.path.join("."), entry.label, Some(entry.display))
                    .with_description(crate::config::field_schema(&path).map(|(fallback, key)| {
                        self.state
                            .t_opt(&format!("config.schema.{key}"))
                            .unwrap_or(fallback)
                    }))
                    .with_modified(modified)
                    .with_section(section)
                    .with_right_detail(affordance),
            );
            fields.push(field);
        }
        (items, fields)
    }

    pub(super) fn refresh_config_search(&mut self) {
        if self
            .state
            .dialog()
            .is_some_and(|dialog| dialog.kind == DialogKind::ConfigEditor)
        {
            self.refresh_config_dialog();
            self.sync_config_dialog_description();
        }
    }

    pub(super) fn config_table<'a>(
        document: &'a toml_edit::DocumentMut,
        level: &[String],
    ) -> Option<&'a dyn toml_edit::TableLike> {
        let mut table: &dyn toml_edit::TableLike = document.as_table();
        for segment in level {
            table = table
                .get(segment)
                .and_then(toml_edit::Item::as_table_like)?;
        }
        Some(table)
    }

    pub(super) fn config_description(state: &TuiState, path: &[&str]) -> Option<String> {
        crate::config::field_schema(path).map(|(fallback, key)| {
            state
                .t_opt(&format!("config.schema.{key}"))
                .unwrap_or(fallback)
        })
    }

    pub(super) fn config_affordance(
        entry: &crate::config::ConfigEntry,
        field: &ConfigFieldRef,
    ) -> String {
        if entry.kind == crate::config::ConfigEntryKind::Bool {
            return if entry.display == "true" {
                "[x]"
            } else {
                "[ ]"
            }
            .to_string();
        }
        match field {
            ConfigFieldRef::Choice(_) | ConfigFieldRef::List(_) => "▸".to_string(),
            _ => String::new(),
        }
    }

    pub(super) fn config_baseline_document(&self) -> Option<toml_edit::DocumentMut> {
        self.config_draft_baseline.as_deref()?.parse().ok()
    }

    pub(super) fn config_value_signature(
        document: &toml_edit::DocumentMut,
        path: &[String],
    ) -> Option<String> {
        let refs: Vec<&str> = path.iter().map(String::as_str).collect();
        if let Some((kind, display)) = crate::config::config_value_in(document, &refs) {
            return Some(format!("{kind:?}\u{1f}{display}"));
        }
        crate::config::config_array_in(document, &refs)
            .map(|values| format!("array\u{1f}{}", values.join("\u{1f}")))
    }

    pub(super) fn config_field_modified(
        document: &toml_edit::DocumentMut,
        baseline: Option<&toml_edit::DocumentMut>,
        path: &[String],
    ) -> bool {
        let Some(baseline) = baseline else {
            return false;
        };
        Self::config_value_signature(document, path) != Self::config_value_signature(baseline, path)
    }

    pub(super) fn config_child_table_count(
        document: &toml_edit::DocumentMut,
        path: &[String],
    ) -> usize {
        Self::config_table(document, path)
            .map(|table| {
                table
                    .iter()
                    .filter(|(_, item)| item.as_table_like().is_some())
                    .count()
            })
            .unwrap_or(0)
    }

    pub(super) fn config_table_detail(
        &self,
        document: &toml_edit::DocumentMut,
        path: &[String],
    ) -> Option<String> {
        let head = path.first().map(String::as_str);
        let name = path.get(1).map(String::as_str);
        match (head, path.len()) {
            (Some("providers"), 1) => {
                Some(self.config_count_text(Self::config_child_table_count(document, path)))
            }
            (Some("providers"), 2) => {
                let name = name?;
                let protocol =
                    crate::config::config_value_in(document, &["providers", name, "protocol"])
                        .map(|(_, value)| value);
                let default_model =
                    crate::config::config_value_in(document, &["providers", name, "default_model"])
                        .map(|(_, value)| value);
                let models = Self::config_child_table_count(
                    document,
                    &[
                        "providers".to_string(),
                        name.to_string(),
                        "models".to_string(),
                    ],
                );
                let auth = Self::config_text(document, &["providers", name, "auth", "type"]);
                let mut parts = Vec::new();
                if let Some(protocol) = protocol {
                    parts.push(protocol);
                }
                if let Some(default_model) = default_model {
                    parts.push(default_model);
                }
                if let Some(auth) = auth {
                    parts.push(auth);
                }
                if models > 0 {
                    parts.push(
                        self.state
                            .t_fmt("config.model_count", &[("count", &models.to_string())]),
                    );
                }
                (!parts.is_empty()).then(|| parts.join(" · "))
            }
            (Some("providers"), 3) if path.get(2).is_some_and(|segment| segment == "models") => {
                Some(self.config_count_text(Self::config_child_table_count(document, path)))
            }
            (Some("providers"), 4) => {
                let model = path.get(3)?;
                let display = crate::config::config_value_in(
                    document,
                    &["providers", name?, "models", model, "display"],
                )
                .map(|(_, value)| value);
                let strategy = crate::config::config_value_in(
                    document,
                    &["providers", name?, "models", model, "strategy"],
                )
                .map(|(_, value)| value);
                let mut parts = Vec::new();
                if let Some(display) = display {
                    parts.push(display);
                }
                if let Some(strategy) = strategy {
                    parts.push(strategy);
                }
                (!parts.is_empty()).then(|| parts.join(" · "))
            }
            (Some("agents"), 1) => {
                Some(self.config_count_text(Self::config_child_table_count(document, path)))
            }
            (Some("agents"), 2) => {
                let name = name?;
                let provider =
                    crate::config::config_value_in(document, &["agents", name, "provider"])
                        .map(|(_, value)| value);
                let model = crate::config::config_value_in(document, &["agents", name, "model"])
                    .map(|(_, value)| value);
                let route = match (provider, model) {
                    (Some(provider), Some(model)) => Some(format!("{provider}/{model}")),
                    _ => crate::config::config_array_in(
                        document,
                        &["agents", name, "allowed_models"],
                    )
                    .and_then(|values| values.into_iter().next())
                    .or_else(|| Some(self.state.t("config.follow_session"))),
                };
                let backend =
                    Self::config_text(document, &["agents", name, "backend"]).or_else(|| {
                        Self::config_effective_missing_value(
                            document,
                            &[
                                "agents".to_string(),
                                name.to_string(),
                                "backend".to_string(),
                            ],
                        )
                    });
                match (route, backend) {
                    (Some(route), Some(backend)) if backend != "chat" => {
                        Some(format!("{route} · {backend}"))
                    }
                    (Some(route), _) => Some(route),
                    (None, Some(backend)) if backend != "chat" => Some(backend),
                    _ => None,
                }
            }
            (Some("mcp"), 1) => {
                Some(self.config_count_text(Self::config_child_table_count(document, path)))
            }
            (Some("mcp"), 2) => {
                let name = name?;
                let local = Self::config_table(
                    document,
                    &["mcp".to_string(), name.to_string(), "local".to_string()],
                );
                let remote = Self::config_table(
                    document,
                    &["mcp".to_string(), name.to_string(), "remote".to_string()],
                );
                let kind = if local.is_some() {
                    "local".to_string()
                } else if remote.is_some() {
                    "remote".to_string()
                } else {
                    crate::config::config_value_in(document, &["mcp", name, "type"])
                        .map(|(_, value)| value)
                        .unwrap_or_else(|| "local".to_string())
                };
                let enabled = crate::config::config_value_in(document, &["mcp", name, "enabled"])
                    .map(|(_, value)| value)
                    .unwrap_or_else(|| "true".to_string());
                let state = if enabled == "true" {
                    self.state.t("config.enabled")
                } else {
                    self.state.t("config.disabled")
                };
                let url = if kind == "remote" {
                    Self::config_text(document, &["mcp", name, "remote", "url"])
                        .or_else(|| Self::config_text(document, &["mcp", name, "url"]))
                } else {
                    None
                };
                let mut parts = vec![kind, state];
                if let Some(url) = url {
                    parts.push(url);
                }
                Some(parts.join(" · "))
            }
            (Some("permissions"), 1) => {
                crate::config::config_value_in(document, &["permissions", "mode"])
                    .map(|(_, value)| value)
            }
            (Some("subagents"), 1) => {
                crate::config::config_value_in(document, &["subagents", "mode"])
                    .map(|(_, value)| value)
            }
            _ => None,
        }
    }

    pub(super) fn config_count_text(&self, count: usize) -> String {
        self.state
            .t_fmt("config.count", &[("count", &count.to_string())])
    }

    pub(super) fn config_active_model(document: &toml_edit::DocumentMut) -> Option<String> {
        let provider = Self::config_text(document, &["active_provider"])?;
        Self::config_text(document, &["providers", &provider, "default_model"])
    }

    pub(super) fn config_mcp_summary(&self, document: &toml_edit::DocumentMut) -> Option<String> {
        let table = Self::config_table(document, &["mcp".to_string()])?;
        let mut total = 0usize;
        let mut enabled = 0usize;
        for (name, item) in table.iter() {
            if item.as_table_like().is_none() {
                continue;
            }
            total += 1;
            if Self::config_bool(document, &["mcp", name, "enabled"]).unwrap_or(true) {
                enabled += 1;
            }
        }
        (total > 0).then(|| {
            self.state.t_fmt(
                "config.mcp_summary",
                &[
                    ("total", &total.to_string()),
                    ("enabled", &enabled.to_string()),
                ],
            )
        })
    }

    pub(super) fn persist_fast_mode(&self, enabled: bool) {
        if let Err(error) = crate::tui::preferences::TuiPreferences::update_in_dir(
            &self.preferences_dir,
            |preferences| {
                preferences.fast_mode = enabled;
                if !enabled {
                    preferences.fast_mode_tier = None;
                }
            },
        ) {
            tracing::warn!(%error, "failed to persist fast mode");
        }
    }

    pub(super) fn persist_fast_mode_tier(&self, tier: Option<String>) {
        if let Err(error) = crate::tui::preferences::TuiPreferences::update_in_dir(
            &self.preferences_dir,
            |preferences| {
                preferences.fast_mode = tier.is_some();
                preferences.fast_mode_tier = tier;
            },
        ) {
            tracing::warn!(%error, "failed to persist fast mode tier");
        }
    }

    pub(super) fn persist_redact_switch(&self, enabled: bool) -> anyhow::Result<()> {
        let config_path = self
            .config_path
            .as_deref()
            .ok_or_else(|| anyhow::anyhow!("configuration path is unavailable"))?;
        let text = std::fs::read_to_string(config_path)?;
        let mut document = text.parse::<toml_edit::DocumentMut>()?;
        crate::config::set_config_scalar(
            &mut document,
            &["global", "redact"],
            crate::config::ConfigScalar::Bool(enabled),
        )?;
        crate::config::save_config_document(config_path, &document)?;
        let config_dir = crate::config::resolve_config_target(config_path)?
            .parent()
            .map(std::path::Path::to_path_buf)
            .ok_or_else(|| anyhow::anyhow!("configuration file has no directory"))?;
        crate::redact::install(enabled, &config_dir);
        Ok(())
    }

    pub(super) fn load_config_draft(&mut self) {
        let text = self
            .config_path
            .as_deref()
            .and_then(|config_path| std::fs::read_to_string(config_path).ok());
        self.config_draft = text
            .as_deref()
            .and_then(|text| text.parse::<toml_edit::DocumentMut>().ok());
        self.config_draft_baseline = text;
    }

    pub(super) fn edit_config_document(
        &mut self,
        edit: impl FnOnce(&mut toml_edit::DocumentMut) -> Result<()>,
    ) -> Result<()> {
        let Some(mut document) = self.config_draft.take() else {
            anyhow::bail!("configuration draft is unavailable");
        };
        let result = edit(&mut document);
        self.config_draft = Some(document);
        if result.is_ok()
            && let Some(dialog) = self.state.dialog_mut()
        {
            dialog.config_dirty = true;
            dialog.config_error = None;
        }
        result
    }

    pub(super) fn save_config_draft(&mut self) -> bool {
        let Some(config_path) = self.config_path.clone() else {
            self.state
                .show_toast(self.state.t("config.unavailable"), ToastKind::Error);
            return false;
        };
        let Some(document) = self.config_draft.as_ref() else {
            return false;
        };
        let saved_text = document.to_string();
        let externally_changed = std::fs::read_to_string(&config_path).ok().as_deref()
            != self.config_draft_baseline.as_deref();
        match crate::config::save_config_document(&config_path, document) {
            Ok(()) => {
                self.config_draft_baseline = Some(saved_text);
                if let Some(dialog) = self.state.dialog_mut() {
                    dialog.config_dirty = false;
                    dialog.config_error = None;
                }
                let key = if externally_changed {
                    "config.saved_over_external"
                } else {
                    "config.saved_value"
                };
                self.state.show_toast(self.state.t(key), ToastKind::Info);
                self.refresh_config_dialog();
                self.sync_config_dialog_description();
                true
            }
            Err(error) => {
                self.report_config_save_failure(error);
                false
            }
        }
    }

    /// Fields whose siblings make them inapplicable stay out of the panel unless they
    /// already hold a value; keeping those visible avoids trapping invalid configuration.
    pub(super) fn config_entry_visible(document: &toml_edit::DocumentMut, path: &[String]) -> bool {
        if path.len() == 1 && path[0] == "fast_mode" {
            return false;
        }
        if path.len() == 3 && path[0] == "providers" && path[2] == "reviewer" {
            return false;
        }
        if path.len() >= 6 && path[4] == "capabilities" && path[5] == "generation" {
            return false;
        }
        Self::config_field_applicable(document, path) || Self::config_path_present(document, path)
    }

    pub(super) fn config_field_applicable(
        document: &toml_edit::DocumentMut,
        path: &[String],
    ) -> bool {
        if path.len() == 3 && path[0] == "agents" && path[2] == "backend" {
            return path[1] == "reviewer";
        }
        if path.len() == 2 && path[0] == "subagents" {
            let swarm_only = matches!(
                path[1].as_str(),
                "max_depth"
                    | "max_concurrent"
                    | "max_descendants"
                    | "max_pending_messages"
                    | "max_message_bytes"
                    | "run_timeout_secs"
            );
            if swarm_only {
                return Self::config_text(document, &["subagents", "mode"]).as_deref()
                    == Some("swarm");
            }
        }
        if path.len() == 3
            && path[0] == "global"
            && path[1] == "session_archive"
            && path[2] == "older_than_days"
        {
            return Self::config_bool(document, &["global", "session_archive", "enabled"])
                .unwrap_or(true);
        }
        if path.len() == 3 && path[0] == "global" && path[1] == "retry" {
            return Self::config_retry_field_applicable(document, None, &path[2]);
        }
        if path.len() == 4 && path[0] == "providers" && path[2] == "retry" {
            return Self::config_retry_field_applicable(document, Some(&path[1]), &path[3]);
        }
        if path.len() == 4 && path[0] == "providers" && path[2] == "auth" {
            let kind = Self::config_text(document, &["providers", &path[1], "auth", "type"]);
            return match path[3].as_str() {
                "name" => matches!(kind.as_deref(), Some("header" | "query")),
                "credential" | "credential_env" => kind.as_deref() != Some("none"),
                _ => true,
            };
        }
        if path[0] == "mcp" && (path.len() == 3 || path.len() == 4) {
            let kind = Self::config_text(document, &["mcp", &path[1], "type"]);
            let local = Self::config_table(
                document,
                &["mcp".to_string(), path[1].clone(), "local".to_string()],
            );
            let remote = Self::config_table(
                document,
                &["mcp".to_string(), path[1].clone(), "remote".to_string()],
            );
            let nested = local.is_some() || remote.is_some();
            let flat_local =
                crate::config::config_array_in(document, &["mcp", &path[1], "command"])
                    .is_some_and(|values| !values.is_empty())
                    || Self::config_table(
                        document,
                        &[
                            "mcp".to_string(),
                            path[1].clone(),
                            "environment".to_string(),
                        ],
                    )
                    .is_some_and(|table| table.iter().next().is_some());
            let flat_remote = Self::config_text(document, &["mcp", &path[1], "url"]).is_some()
                || Self::config_table(
                    document,
                    &["mcp".to_string(), path[1].clone(), "headers".to_string()],
                )
                .is_some_and(|table| table.iter().next().is_some())
                || Self::config_bool(document, &["mcp", &path[1], "oauth"]).unwrap_or(false);
            if path.len() == 4 {
                return match path[2].as_str() {
                    "local" => local.is_some() && remote.is_none(),
                    "remote" => remote.is_some() && local.is_none(),
                    _ => true,
                };
            }
            match path[2].as_str() {
                "type" => return !nested,
                "local" => {
                    return if nested {
                        local.is_some() && remote.is_none()
                    } else {
                        !flat_local && kind.as_deref() == Some("local")
                    };
                }
                "remote" => {
                    return if nested {
                        remote.is_some() && local.is_none()
                    } else {
                        !flat_remote && kind.as_deref() == Some("remote")
                    };
                }
                "command" | "environment" | "env" => {
                    return !nested && kind.as_deref() != Some("remote");
                }
                "url" | "headers" | "oauth" => {
                    return !nested && kind.as_deref() == Some("remote");
                }
                _ => {}
            }
        }
        if let Some((provider, model)) = Self::config_model_scope(path) {
            if path.len() == 6 && path[4] == "capabilities" {
                return Self::config_capability_applicable(document, provider, model, &path[5]);
            }
            if path.len() == 6 && path[4] == "generation" {
                return Self::config_generation_field_applicable(
                    document, provider, model, &path[5],
                );
            }
            if path.len() == 5 && path[4] == "generation" {
                return Self::config_generation_field_applicable(
                    document,
                    provider,
                    model,
                    "temperature",
                ) || [
                    "top_p",
                    "max_output_tokens",
                    "stop_sequences",
                    "reasoning_effort",
                    "reasoning_efforts",
                    "reasoning_summary",
                    "text_verbosity",
                    "parallel_tool_calls",
                    "async_tools",
                ]
                .iter()
                .any(|field| {
                    Self::config_generation_field_applicable(document, provider, model, field)
                });
            }
            if path.len() == 6 && path[4] == "transport" && path[5] == "websocket" {
                let protocol = Self::config_model_protocol(document, provider, model);
                return Self::config_protocol_allows(protocol.as_deref(), &["responses"]);
            }
            if path.len() == 5 && path[4] == "transport" {
                let protocol = Self::config_model_protocol(document, provider, model);
                return Self::config_protocol_allows(protocol.as_deref(), &["responses"]);
            }
            if path.len() == 6 && path[4] == "cache" {
                let cache_enabled = Self::config_bool(
                    document,
                    &["providers", provider, "models", model, "cache", "enabled"],
                )
                .unwrap_or(false);
                return match path[5].as_str() {
                    "namespace" => cache_enabled,
                    "retention" => {
                        let protocol = Self::config_model_protocol(document, provider, model);
                        cache_enabled
                            && Self::config_protocol_allows(protocol.as_deref(), &["responses"])
                    }
                    _ => true,
                };
            }
            if path.len() == 5 && path[4] == "protocol_settings" {
                let protocol = Self::config_model_protocol(document, provider, model);
                return protocol.as_deref() == Some("anthropic");
            }
        }
        true
    }

    pub(super) fn config_effective_missing_value(
        document: &toml_edit::DocumentMut,
        path: &[String],
    ) -> Option<String> {
        if path.len() != 3 || path[0] != "agents" || path[1] != "reviewer" || path[2] != "backend" {
            return None;
        }
        let provider =
            Self::config_text(document, &["agents", "reviewer", "provider"]).or_else(|| {
                crate::config::config_array_in(document, &["agents", "reviewer", "allowed_models"])
                    .and_then(|routes| routes.first().cloned())
                    .and_then(|route| {
                        route
                            .split_once('/')
                            .map(|(provider, _)| provider.to_string())
                    })
            })?;
        (Self::config_text(document, &["providers", &provider, "reviewer"]).as_deref()
            == Some("jev"))
        .then(|| "jev".to_string())
    }

    pub(super) fn config_path_present(document: &toml_edit::DocumentMut, path: &[String]) -> bool {
        let refs: Vec<&str> = path.iter().map(String::as_str).collect();
        crate::config::config_value_in(document, &refs).is_some()
            || crate::config::config_array_in(document, &refs).is_some()
            || Self::config_table(document, path).is_some()
    }

    pub(super) fn config_bool(document: &toml_edit::DocumentMut, path: &[&str]) -> Option<bool> {
        crate::config::config_value_in(document, path).and_then(|value| {
            (value.0 == crate::config::ConfigEntryKind::Bool).then_some(value.1 == "true")
        })
    }

    pub(super) fn config_text(document: &toml_edit::DocumentMut, path: &[&str]) -> Option<String> {
        crate::config::config_value_in(document, path).map(|value| value.1)
    }

    pub(super) fn config_model_scope(path: &[String]) -> Option<(&str, &str)> {
        (path.len() >= 5 && path[0] == "providers" && path[2] == "models")
            .then(|| (path[1].as_str(), path[3].as_str()))
    }

    pub(super) fn config_model_protocol(
        document: &toml_edit::DocumentMut,
        provider: &str,
        model: &str,
    ) -> Option<String> {
        Self::config_text(
            document,
            &["providers", provider, "models", model, "protocol"],
        )
        .or_else(|| Self::config_text(document, &["providers", provider, "protocol"]))
    }

    pub(super) fn config_protocol_allows(protocol: Option<&str>, allowed: &[&str]) -> bool {
        protocol.is_none_or(|protocol| allowed.contains(&protocol))
    }

    pub(super) fn config_retry_field_applicable(
        document: &toml_edit::DocumentMut,
        provider: Option<&str>,
        field: &str,
    ) -> bool {
        let global_enabled =
            Self::config_bool(document, &["global", "retry", "enabled"]).unwrap_or(true);
        let global_backoff =
            Self::config_bool(document, &["global", "retry", "exponential_backoff"])
                .unwrap_or(true);
        let (enabled, backoff) = if let Some(provider) = provider {
            (
                Self::config_bool(document, &["providers", provider, "retry", "enabled"])
                    .unwrap_or(global_enabled),
                Self::config_bool(
                    document,
                    &["providers", provider, "retry", "exponential_backoff"],
                )
                .unwrap_or(global_backoff),
            )
        } else {
            (global_enabled, global_backoff)
        };
        match field {
            "enabled" => true,
            "backoff_multiplier" => enabled && backoff,
            _ => enabled,
        }
    }

    pub(super) fn config_capability_applicable(
        document: &toml_edit::DocumentMut,
        provider: &str,
        model: &str,
        field: &str,
    ) -> bool {
        let tools = Self::config_bool(
            document,
            &[
                "providers",
                provider,
                "models",
                model,
                "capabilities",
                "tools",
            ],
        )
        .unwrap_or(false);
        match field {
            "parallel_tool_calls" | "tool_result_images" => tools,
            _ => true,
        }
    }

    pub(super) fn config_generation_field_applicable(
        document: &toml_edit::DocumentMut,
        provider: &str,
        model: &str,
        field: &str,
    ) -> bool {
        let caps = |name: &str| {
            Self::config_bool(
                document,
                &["providers", provider, "models", model, "capabilities", name],
            )
            .unwrap_or(false)
        };
        let protocol = Self::config_model_protocol(document, provider, model);
        match field {
            "reasoning_effort" | "reasoning_efforts" => caps("reasoning"),
            "reasoning_summary" => {
                caps("reasoning")
                    && Self::config_protocol_allows(protocol.as_deref(), &["responses"])
            }
            "text_verbosity" => {
                Self::config_protocol_allows(protocol.as_deref(), &["responses", "completions"])
            }
            "parallel_tool_calls" => caps("parallel_tool_calls"),
            "async_tools" => {
                let strategy = Self::config_text(
                    document,
                    &["providers", provider, "models", model, "strategy"],
                );
                caps("tools")
                    && strategy
                        .as_deref()
                        .is_none_or(|strategy| strategy == "astra")
                    && Self::config_protocol_allows(protocol.as_deref(), &["responses"])
            }
            _ => true,
        }
    }

    pub(super) fn config_field_values(
        document: &toml_edit::DocumentMut,
        path: &[String],
    ) -> Option<Vec<String>> {
        if path.len() == 1 && path[0] == "active_provider" {
            return Self::config_table_names(document, &["providers"]);
        }
        if path.len() == 3 && path[0] == "providers" && path[2] == "default_model" {
            return Self::config_table_names(document, &["providers", &path[1], "models"]);
        }
        if (path.len() == 3 && path[0] == "agents" && path[2] == "reasoning_effort")
            || (path.len() == 6
                && path[0] == "providers"
                && path[2] == "models"
                && path[4] == "generation"
                && path[5] == "reasoning_effort")
        {
            return Some(
                crate::config::REASONING_EFFORTS
                    .iter()
                    .map(|value| value.to_string())
                    .collect(),
            );
        }
        if path.len() == 3 && path[0] == "agents" && path[2] == "allowed_models" {
            return Some(
                Self::config_table_names(document, &["providers"])?
                    .into_iter()
                    .flat_map(|provider| {
                        Self::config_table_names(document, &["providers", &provider, "models"])
                            .unwrap_or_default()
                            .into_iter()
                            .map(move |model| format!("{provider}/{model}"))
                    })
                    .collect(),
            );
        }
        if path.len() == 3 && path[0] == "agents" {
            match path[2].as_str() {
                "provider" => return Self::config_table_names(document, &["providers"]),
                "model" => {
                    let agent_provider = crate::config::config_value_in(
                        document,
                        &["agents", path[1].as_str(), "provider"],
                    )
                    .or_else(|| crate::config::config_value_in(document, &["active_provider"]))
                    .map(|(_, value)| value)
                    .or_else(|| {
                        Self::config_table_names(document, &["providers"])?
                            .into_iter()
                            .next()
                    })?;
                    return Self::config_table_names(
                        document,
                        &["providers", &agent_provider, "models"],
                    );
                }
                _ => {}
            }
        }
        let path_refs: Vec<&str> = path.iter().map(String::as_str).collect();
        crate::config::field_enum(&path_refs)
    }

    pub(super) fn config_table_names(
        document: &toml_edit::DocumentMut,
        path: &[&str],
    ) -> Option<Vec<String>> {
        let path: Vec<String> = path.iter().map(|segment| segment.to_string()).collect();
        Some(
            Self::config_table(document, &path)?
                .iter()
                .filter(|(_, item)| item.as_table_like().is_some())
                .map(|(name, _)| name.to_string())
                .collect(),
        )
    }

    pub(super) fn expanded_field(&self) -> Option<ConfigFieldRef> {
        let dialog = self.state.dialog()?;
        dialog.config_fields.get(dialog.config_expanded?).cloned()
    }

    pub(super) fn selected_config_field(&self) -> Option<ConfigFieldRef> {
        let dialog = self.state.dialog()?;
        dialog.config_fields.get(dialog.selected).cloned()
    }

    pub(super) fn expanded_index(&self) -> Option<usize> {
        self.state
            .dialog()
            .and_then(|dialog| dialog.config_expanded)
    }

    pub(super) fn expand_config_field(&mut self, index: usize) {
        let Some(field) = self
            .state
            .dialog()
            .and_then(|dialog| dialog.config_fields.get(index).cloned())
        else {
            return;
        };
        let (ConfigFieldRef::List(path) | ConfigFieldRef::Choice(path)) = &field else {
            return;
        };
        let Some(document) = self.config_draft.as_ref() else {
            return;
        };
        let path_refs: Vec<&str> = path.iter().map(String::as_str).collect();
        let array = crate::config::config_array_in(document, &path_refs);
        let values = match &array {
            Some(values) => values.clone(),
            None => {
                let Some(values) = Self::config_field_values(document, path) else {
                    return;
                };
                values
            }
        };
        let current = self
            .state
            .dialog()
            .and_then(|dialog| dialog.selected_item())
            .and_then(|item| item.detail.clone());
        let mut items = values
            .iter()
            .map(|value| DialogItem::new(value.clone(), value.clone(), None))
            .collect::<Vec<_>>();
        if matches!(field, ConfigFieldRef::Choice(_))
            && Self::config_field_allows_custom(document, path)
        {
            items.push(DialogItem::new(
                CONFIG_CUSTOM_CHOICE.to_string(),
                self.state.t("config.custom_value"),
                None,
            ));
        }
        let selected = current
            .and_then(|current| items.iter().position(|item| item.label == current))
            .unwrap_or(0);
        if let Some(dialog) = self.state.dialog_mut() {
            dialog.config_expanded = Some(index);
            dialog.config_detail_items = items;
            dialog.config_detail_selected = selected;
            dialog.config_detail_target = None;
        }
    }

    pub(super) fn collapse_config_field(&mut self) {
        if let Some(dialog) = self.state.dialog_mut() {
            dialog.config_expanded = None;
            dialog.config_detail_items.clear();
            dialog.config_detail_selected = 0;
            dialog.config_detail_target = None;
        }
    }

    /// A field whose schema is an open string still accepts hand-written values.
    pub(super) fn config_field_allows_custom(
        document: &toml_edit::DocumentMut,
        path: &[String],
    ) -> bool {
        let path_refs: Vec<&str> = path.iter().map(String::as_str).collect();
        crate::config::field_enum(&path_refs).is_none()
            && Self::config_field_values(document, path).is_some()
    }

    pub(super) fn config_list_has_choices(&self, path: &[String]) -> bool {
        self.config_draft
            .as_ref()
            .is_some_and(|document| Self::config_field_values(document, path).is_some())
    }

    pub(super) fn show_config_element_choices(&mut self, path: &[String], index: usize) {
        let Some(document) = self.config_draft.as_ref() else {
            return;
        };
        let Some(values) = Self::config_field_values(document, path) else {
            return;
        };
        let items = values
            .iter()
            .map(|value| DialogItem::new(value.clone(), value.clone(), None))
            .collect::<Vec<_>>();
        if let Some(dialog) = self.state.dialog_mut() {
            dialog.config_detail_items = items;
            dialog.config_detail_selected = 0;
            dialog.config_detail_target = Some(index);
        }
    }

    pub(super) fn reopen_config_detail(&mut self, index: usize, selected: usize) {
        self.collapse_config_field();
        self.expand_config_field(index);
        if let Some(dialog) = self.state.dialog_mut() {
            let last = dialog.config_detail_items.len().saturating_sub(1);
            dialog.config_detail_selected = selected.min(last);
        }
    }

    pub(super) fn accept_config_detail(&mut self) {
        let Some(field) = self.expanded_field() else {
            return;
        };
        let Some(dialog) = self.state.dialog() else {
            return;
        };
        let selected = dialog.config_detail_selected;
        let target = dialog.config_detail_target;
        let Some(item) = dialog.config_detail_items.get(selected).cloned() else {
            return;
        };
        let value = item.label.clone();
        match field {
            ConfigFieldRef::List(path) => match target {
                Some(index) => self.commit_config_list_item(&path, index, &value),
                None if self.config_list_has_choices(&path) => {
                    self.show_config_element_choices(&path, selected)
                }
                None => self.begin_config_list_item_edit(ConfigFieldRef::ListItem(path, selected)),
            },
            ConfigFieldRef::Choice(path) => {
                if item.id == CONFIG_CUSTOM_CHOICE {
                    self.collapse_config_field();
                    self.begin_config_edit(&path);
                } else {
                    self.apply_config_choice(&path, &value);
                }
            }
            _ => {}
        }
    }

    pub(super) fn apply_config_choice(&mut self, path: &[String], value: &str) {
        let path_refs: Vec<&str> = path.iter().map(String::as_str).collect();
        let result = self.edit_config_document(|document| {
            crate::config::set_config_scalar(
                document,
                &path_refs,
                crate::config::ConfigScalar::String(value.to_string()),
            )
        });
        match result {
            Ok(()) => {
                self.collapse_config_field();
                self.refresh_config_dialog();
                self.state
                    .show_toast(self.state.t("config.unsaved"), ToastKind::Info);
            }
            Err(error) => self.report_config_save_failure(error),
        }
    }

    pub(super) fn append_config_list_item(&mut self) {
        let Some(ConfigFieldRef::List(path)) = self.expanded_field() else {
            return;
        };
        let path_refs: Vec<&str> = path.iter().map(String::as_str).collect();
        let next = self
            .config_draft
            .as_ref()
            .and_then(|document| crate::config::config_array_in(document, &path_refs))
            .map(|values| values.len())
            .unwrap_or(0);
        if self.config_list_has_choices(&path) {
            self.show_config_element_choices(&path, next);
            return;
        }
        let field = ConfigFieldRef::ListItem(path, next);
        if let Some(dialog) = self.state.dialog_mut() {
            dialog
                .config_detail_items
                .push(DialogItem::new(String::new(), String::new(), None));
            dialog.config_detail_selected = dialog.config_detail_items.len() - 1;
        }
        self.state.config_edit = Some(crate::tui::state::ConfigEditState {
            field,
            cursor: 0,
            buffer: String::new(),
        });
    }

    pub(super) fn remove_config_list_item(&mut self) {
        let Some(ConfigFieldRef::List(path)) = self.expanded_field() else {
            return;
        };
        let Some(index) = self
            .state
            .dialog()
            .map(|dialog| dialog.config_detail_selected)
        else {
            return;
        };
        let path_refs: Vec<&str> = path.iter().map(String::as_str).collect();
        let Some(mut values) = self
            .config_draft
            .as_ref()
            .and_then(|document| crate::config::config_array_in(document, &path_refs))
        else {
            return;
        };
        if index >= values.len() {
            return;
        }
        values.remove(index);
        let result = self.edit_config_document(|document| {
            crate::config::set_config_scalar(
                document,
                &path_refs,
                crate::config::ConfigScalar::Array(values),
            )
        });
        match result {
            Ok(()) => {
                if let Some(expanded) = self.expanded_index() {
                    self.reopen_config_detail(expanded, index);
                }
                self.state
                    .show_toast(self.state.t("config.unsaved"), ToastKind::Info);
            }
            Err(error) => self.report_config_save_failure(error),
        }
    }

    pub(super) fn begin_config_list_item_edit(&mut self, field: ConfigFieldRef) {
        let ConfigFieldRef::ListItem(path, index) = &field else {
            return;
        };
        let path_refs: Vec<&str> = path.iter().map(String::as_str).collect();
        let buffer = self
            .config_draft
            .as_ref()
            .and_then(|document| crate::config::config_array_in(document, &path_refs))
            .and_then(|values| values.get(*index).cloned())
            .unwrap_or_default();
        self.state.config_edit = Some(crate::tui::state::ConfigEditState {
            field,
            cursor: buffer.len(),
            buffer,
        });
    }

    pub(super) fn commit_config_list_item(&mut self, path: &[String], index: usize, value: &str) {
        let path_refs: Vec<&str> = path.iter().map(String::as_str).collect();
        let Some(mut values) = self
            .config_draft
            .as_ref()
            .and_then(|document| crate::config::config_array_in(document, &path_refs))
        else {
            return;
        };
        let selected = if index >= values.len() {
            values.push(value.to_string());
            values.len() - 1
        } else {
            values[index] = value.to_string();
            index
        };
        let result = self.edit_config_document(|document| {
            crate::config::set_config_scalar(
                document,
                &path_refs,
                crate::config::ConfigScalar::Array(values),
            )
        });
        match result {
            Ok(()) => {
                if let Some(expanded) = self.expanded_index() {
                    self.reopen_config_detail(expanded, selected);
                }
                self.state
                    .show_toast(self.state.t("config.unsaved"), ToastKind::Info);
            }
            Err(error) => self.report_config_save_failure(error),
        }
    }

    pub(super) fn commit_config_new_entry(&mut self, table: &str, name: &str) {
        if name.is_empty() {
            return;
        }
        let path: Vec<String> = table
            .split('/')
            .map(str::to_string)
            .chain(std::iter::once(name.to_string()))
            .collect();
        self.commit_config_new_item(&path);
    }

    pub(super) fn commit_config_new_field(&mut self, path: &[String]) {
        self.commit_config_new_item(path);
    }

    pub(super) fn commit_config_new_item(&mut self, path: &[String]) {
        let path_refs: Vec<&str> = path.iter().map(String::as_str).collect();
        let item = crate::config::default_value(&path_refs)
            .unwrap_or_else(|| toml_edit::Item::Table(toml_edit::Table::new()));
        let result = self.edit_config_document(|document| {
            crate::config::set_config_item(document, &path_refs, item)
        });
        match result {
            Ok(()) => {
                self.refresh_config_dialog();
                self.state
                    .show_toast(self.state.t("config.unsaved"), ToastKind::Info);
            }
            Err(error) => self.report_config_save_failure(error),
        }
    }

    pub(super) fn sync_config_dialog_description(&mut self) {
        let description = self
            .state
            .dialog()
            .filter(|dialog| dialog.kind == DialogKind::ConfigEditor)
            .and_then(|dialog| {
                let item = dialog.selected_item()?;
                let field = dialog.config_fields.get(dialog.selected);
                self.config_detail_text(field, item)
            });
        if let Some(dialog) = self.state.dialog_mut()
            && dialog.kind == DialogKind::ConfigEditor
        {
            dialog.description = description;
        }
    }

    pub(super) fn config_detail_text(
        &self,
        field: Option<&ConfigFieldRef>,
        item: &DialogItem,
    ) -> Option<String> {
        let mut lines = Vec::new();
        if let Some(description) = &item.description {
            lines.push(description.clone());
        }
        if let Some(path) = field.and_then(ConfigFieldRef::path) {
            let refs: Vec<&str> = path.iter().map(String::as_str).collect();
            let mut meta = Vec::new();
            if let Some(default) = crate::config::default_display(&refs) {
                meta.push(format!("{} {default}", self.state.t("config.default")));
            }
            if let Some((kind, _)) = self
                .config_draft
                .as_ref()
                .and_then(|document| crate::config::config_value_in(document, &refs))
            {
                meta.push(self.state.t(Self::config_kind_key(kind)).to_string());
            }
            if item.modified {
                meta.push(self.state.t("config.modified"));
            }
            if !meta.is_empty() {
                lines.push(meta.join(" · "));
            }
        }
        (!lines.is_empty()).then(|| lines.join("\n"))
    }

    pub(super) fn config_kind_key(kind: crate::config::ConfigEntryKind) -> &'static str {
        match kind {
            crate::config::ConfigEntryKind::Text | crate::config::ConfigEntryKind::ReadOnly => {
                "config.kind_text"
            }
            crate::config::ConfigEntryKind::Integer => "config.kind_integer",
            crate::config::ConfigEntryKind::Float => "config.kind_float",
            crate::config::ConfigEntryKind::Bool => "config.kind_bool",
            crate::config::ConfigEntryKind::Array => "config.kind_array",
        }
    }

    pub(super) fn handle_config_editor_accept(&mut self) {
        if let Some(selected) = self
            .state
            .dialog()
            .and_then(|dialog| dialog.config_close_selected)
        {
            match selected {
                0 => {
                    if self.save_config_draft() {
                        self.state.close_dialog();
                        self.config_draft = None;
                    } else if let Some(dialog) = self.state.dialog_mut() {
                        dialog.config_close_selected = None;
                    }
                }
                1 => {
                    self.state.close_dialog();
                    self.config_draft = None;
                    self.state
                        .show_toast(self.state.t("config.discarded"), ToastKind::Info);
                }
                _ => {
                    if let Some(dialog) = self.state.dialog_mut() {
                        dialog.config_close_selected = None;
                    }
                }
            }
            return;
        }
        if self.expanded_index().is_some() {
            self.accept_config_detail();
            return;
        }
        let Some(dialog) = self.state.dialog() else {
            return;
        };
        let index = dialog.selected;
        let detail = dialog.selected_item().and_then(|item| item.detail.clone());
        let Some(field) = dialog.config_fields.get(index).cloned() else {
            return;
        };
        match field {
            ConfigFieldRef::NewEntry(table) => {
                self.state.config_edit = Some(crate::tui::state::ConfigEditState {
                    field: ConfigFieldRef::NewEntry(table),
                    cursor: 0,
                    buffer: String::new(),
                });
            }
            ConfigFieldRef::Choice(_) | ConfigFieldRef::List(_) => self.expand_config_field(index),
            ConfigFieldRef::Field(path) => {
                let path_refs: Vec<&str> = path.iter().map(String::as_str).collect();
                let kind = self
                    .config_draft
                    .as_ref()
                    .and_then(|document| crate::config::config_value_in(document, &path_refs))
                    .map(|(kind, _)| kind);
                match kind {
                    Some(crate::config::ConfigEntryKind::Bool) => {
                        self.toggle_config_bool(&path, detail.as_deref() == Some("true"))
                    }
                    Some(crate::config::ConfigEntryKind::Text)
                    | Some(crate::config::ConfigEntryKind::Integer)
                    | Some(crate::config::ConfigEntryKind::Float) => self.begin_config_edit(&path),
                    _ => {}
                }
            }
            ConfigFieldRef::NewField(path) => self.commit_config_new_field(&path),
            ConfigFieldRef::Table(path) => self.enter_config_table(path),
            ConfigFieldRef::ListItem(..) => {}
        }
    }

    pub(super) fn enter_config_table(&mut self, path: Vec<String>) {
        if let Some(dialog) = self.state.dialog_mut() {
            let target = ConfigTab::for_path(&path);
            let switching = target.is_some_and(|tab| tab != dialog.config_tab);
            let old_index = dialog.config_tab.index();
            dialog.config_tab_paths[old_index] = dialog.config_path.clone();
            dialog.config_tab_selected[old_index] = dialog.selected;
            if switching {
                dialog.config_selected.clear();
            } else {
                dialog.config_selected.push(dialog.selected);
            }
            if let Some(tab) = target {
                dialog.config_tab = tab;
            }
            dialog.config_path = path;
            dialog.selected = 0;
            dialog.query.clear();
            dialog.config_searching = false;
            let index = dialog.config_tab.index();
            dialog.config_tab_paths[index] = dialog.config_path.clone();
            dialog.config_tab_selected[index] = 0;
        }
        self.refresh_config_dialog();
        self.sync_config_dialog_description();
    }

    pub(super) fn leave_config_table(&mut self) -> bool {
        let Some(dialog) = self.state.dialog_mut() else {
            return false;
        };
        let tab_root = Self::config_tab_path(dialog.config_tab);
        if dialog.config_tab != ConfigTab::Overview && dialog.config_path == tab_root {
            let old_index = dialog.config_tab.index();
            dialog.config_tab_paths[old_index] = dialog.config_path.clone();
            dialog.config_tab_selected[old_index] = dialog.selected;
            dialog.config_tab = ConfigTab::Overview;
            let index = dialog.config_tab.index();
            dialog.config_path = dialog.config_tab_paths[index].clone();
            dialog.selected = dialog.config_tab_selected[index];
            dialog.config_selected.clear();
            dialog.query.clear();
            dialog.config_searching = false;
            self.refresh_config_dialog();
            self.sync_config_dialog_description();
            return true;
        }
        if dialog.config_path.is_empty() {
            return false;
        }
        dialog.config_path.pop();
        if dialog.config_path.len() < tab_root.len() {
            dialog.config_path = tab_root;
        }
        dialog.selected = dialog.config_selected.pop().unwrap_or(0);
        let index = dialog.config_tab.index();
        dialog.config_tab_paths[index] = dialog.config_path.clone();
        dialog.config_tab_selected[index] = dialog.selected;
        self.refresh_config_dialog();
        self.sync_config_dialog_description();
        true
    }

    pub(super) fn select_config_tab(&mut self, tab: ConfigTab) {
        if self
            .state
            .dialog()
            .is_some_and(|dialog| dialog.config_tab == tab)
        {
            return;
        }
        if let Some(dialog) = self.state.dialog_mut() {
            let old_index = dialog.config_tab.index();
            dialog.config_tab_paths[old_index] = dialog.config_path.clone();
            dialog.config_tab_selected[old_index] = dialog.selected;
            let index = tab.index();
            dialog.config_tab = tab;
            let mut path = dialog.config_tab_paths[index].clone();
            if path.is_empty() && !tab.root_path().is_empty() {
                path = Self::config_tab_path(tab);
            }
            dialog.config_path = path;
            dialog.selected = dialog.config_tab_selected[index];
            dialog.config_selected.clear();
            dialog.query.clear();
            dialog.config_searching = false;
            dialog.config_expanded = None;
            dialog.config_detail_items.clear();
            dialog.config_detail_selected = 0;
            dialog.config_detail_target = None;
            let index = dialog.config_tab.index();
            dialog.config_tab_paths[index] = dialog.config_path.clone();
            dialog.config_tab_selected[index] = dialog.selected;
        }
        self.refresh_config_dialog();
        self.sync_config_dialog_description();
    }

    pub(super) fn cycle_config_tab(&mut self, forward: bool) {
        let Some(dialog) = self.state.dialog() else {
            return;
        };
        let tab = if forward {
            dialog.config_tab.next()
        } else {
            dialog.config_tab.previous()
        };
        self.select_config_tab(tab);
    }

    pub(super) fn start_config_search(&mut self) {
        if let Some(dialog) = self.state.dialog_mut() {
            dialog.config_searching = true;
        }
        self.refresh_config_dialog();
        self.sync_config_dialog_description();
    }

    pub(super) fn cancel_config_search(&mut self) -> bool {
        let Some(dialog) = self.state.dialog_mut() else {
            return false;
        };
        if !dialog.config_searching && dialog.query.is_empty() {
            return false;
        }
        dialog.config_searching = false;
        dialog.query.clear();
        self.refresh_config_dialog();
        self.sync_config_dialog_description();
        true
    }

    pub(super) fn reset_config_field(&mut self) {
        let Some(field) = self.selected_config_field() else {
            return;
        };
        let Some(path) = field.path().map(<[String]>::to_vec) else {
            return;
        };
        let refs: Vec<&str> = path.iter().map(String::as_str).collect();
        if crate::config::default_display(&refs).is_none() {
            return;
        }
        let Some(document) = self.config_draft.as_ref() else {
            return;
        };
        let present = crate::config::config_value_in(document, &refs).is_some()
            || crate::config::config_array_in(document, &refs).is_some();
        if !present {
            return;
        }
        let result = self
            .edit_config_document(|document| crate::config::remove_config_table(document, &refs));
        if result.is_ok() {
            self.refresh_config_dialog();
            self.sync_config_dialog_description();
            self.state
                .show_toast(self.state.t("config.reset_value"), ToastKind::Info);
        } else if let Err(error) = result {
            self.report_config_save_failure(error);
        }
    }

    pub(super) fn remove_config_table(&mut self, path: Vec<String>) {
        let path_refs: Vec<&str> = path.iter().map(String::as_str).collect();
        let result = self.edit_config_document(|document| {
            crate::config::remove_config_table(document, &path_refs)
        });
        match result {
            Ok(()) => {
                self.refresh_config_dialog();
                self.state
                    .show_toast(self.state.t("config.unsaved"), ToastKind::Info);
            }
            Err(error) => self.report_config_save_failure(error),
        }
    }

    pub(super) fn toggle_config_bool(&mut self, segments: &[String], current: bool) {
        let path: Vec<&str> = segments.iter().map(String::as_str).collect();
        let result = self.edit_config_document(|document| {
            crate::config::set_config_scalar(
                document,
                &path,
                crate::config::ConfigScalar::Bool(!current),
            )
        });
        match result {
            Ok(()) => {
                self.refresh_config_dialog();
                self.state
                    .show_toast(self.state.t("config.unsaved"), ToastKind::Info);
            }
            Err(error) => self.report_config_save_failure(error),
        }
    }

    pub(super) fn begin_config_edit(&mut self, path: &[String]) {
        let path_refs: Vec<&str> = path.iter().map(String::as_str).collect();
        let buffer = self
            .config_draft
            .as_ref()
            .and_then(|document| crate::config::config_value_in(document, &path_refs))
            .map(|(_, text)| text)
            .unwrap_or_default();
        self.state.config_edit = Some(crate::tui::state::ConfigEditState {
            field: ConfigFieldRef::Field(path.to_vec()),
            cursor: buffer.len(),
            buffer,
        });
    }

    pub(super) fn commit_config_edit(&mut self) {
        let Some(edit) = self.state.config_edit.clone() else {
            return;
        };
        let value = edit.buffer.trim().to_string();
        match edit.field {
            ConfigFieldRef::NewEntry(table) => {
                self.state.config_edit = None;
                self.commit_config_new_entry(&table, &value);
            }
            ConfigFieldRef::ListItem(path, index) => {
                self.state.config_edit = None;
                self.commit_config_list_item(&path, index, &value);
            }
            ConfigFieldRef::NewField(path) => {
                self.state.config_edit = None;
                self.commit_config_new_field(&path);
            }
            ConfigFieldRef::Field(path) => {
                let path_refs: Vec<&str> = path.iter().map(String::as_str).collect();
                let kind = self
                    .config_draft
                    .as_ref()
                    .and_then(|document| crate::config::config_value_in(document, &path_refs))
                    .map(|(kind, _)| kind)
                    .unwrap_or(crate::config::ConfigEntryKind::Text);
                let scalar = match kind {
                    crate::config::ConfigEntryKind::Integer => match value.parse::<i64>() {
                        Ok(number) => crate::config::ConfigScalar::Integer(number),
                        Err(_) => {
                            let message = self.state.t("config.invalid_number");
                            if let Some(dialog) = self.state.dialog_mut() {
                                dialog.config_error = Some(message.clone());
                            }
                            self.state.show_toast(message, ToastKind::Error);
                            return;
                        }
                    },
                    crate::config::ConfigEntryKind::Float => match value.parse::<f64>() {
                        Ok(number) => crate::config::ConfigScalar::Float(number),
                        Err(_) => {
                            let message = self.state.t("config.invalid_float");
                            if let Some(dialog) = self.state.dialog_mut() {
                                dialog.config_error = Some(message.clone());
                            }
                            self.state.show_toast(message, ToastKind::Error);
                            return;
                        }
                    },
                    _ => crate::config::ConfigScalar::String(value.clone()),
                };
                let result = self.edit_config_document(|document| {
                    crate::config::set_config_scalar(document, &path_refs, scalar)
                });
                match result {
                    Ok(()) => {
                        self.state.config_edit = None;
                        self.refresh_config_dialog();
                        self.state
                            .show_toast(self.state.t("config.unsaved"), ToastKind::Info);
                    }
                    Err(error) => self.report_config_save_failure(error),
                }
            }
            ConfigFieldRef::Choice(_) | ConfigFieldRef::List(_) | ConfigFieldRef::Table(_) => {}
        }
    }

    pub(super) fn report_config_save_failure(&mut self, error: anyhow::Error) {
        tracing::warn!(%error, "failed to persist configuration change");
        let error_text = format!("{error:#}");
        if let Some(dialog) = self.state.dialog_mut() {
            dialog.config_error = Some(error_text.clone());
        }
        self.state.show_toast(
            self.state
                .t_fmt("config.save_failed", &[("error", &error_text)]),
            ToastKind::Error,
        );
    }

    pub(super) fn refresh_config_dialog(&mut self) {
        let (items, fields) = self.config_dialog_items();
        if let Some(dialog) = self.state.dialog_mut() {
            let selected = dialog.selected.min(items.len().saturating_sub(1));
            dialog.items = items;
            dialog.config_fields = fields;
            dialog.selected = selected;
        }
    }
    pub(super) fn show_config_dialog(&mut self) -> Result<Option<SubmittedCommand>> {
        self.load_config_draft();
        let (items, fields) = self.config_dialog_items();
        let mut dialog = DialogState::new(
            DialogKind::ConfigEditor,
            self.state.t("config.title"),
            None,
            items,
        );
        dialog.config_fields = fields;
        dialog.config_tab = ConfigTab::Overview;
        dialog.config_searching = false;
        dialog.selected = 0;
        self.state.open_dialog(dialog);
        self.sync_config_dialog_description();
        Ok(Some(SubmittedCommand::LocalOnly))
    }
}
