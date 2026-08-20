//! Session engine configuration reload support.

use super::*;

pub(crate) fn reload_has_runtime_delta(
    providers_runtime_unchanged: bool,
    maps_unchanged: bool,
    settings_unchanged: bool,
    current_provider_runtime_unchanged: bool,
    catalog_unchanged: bool,
    new_session_default_unchanged: bool,
) -> bool {
    !(providers_runtime_unchanged
        && maps_unchanged
        && settings_unchanged
        && current_provider_runtime_unchanged
        && catalog_unchanged
        && new_session_default_unchanged)
}

pub(crate) fn model_catalog_updated_event(config: &AppConfig) -> ModelCatalogUpdatedEvent {
    ModelCatalogUpdatedEvent {
        models: config
            .providers
            .iter()
            .flat_map(|(provider_name, provider)| {
                provider.models.iter().map(move |(model_id, model)| {
                    let metadata = model.request_metadata();
                    ModelCatalogEntry {
                        id: ModelRoute::new(provider_name, model_id).display_name(),
                        label: provider.model_label(model_id),
                        provider: provider_name.clone(),
                        context_window_tokens: model.context_window,
                        reasoning: ModelCatalogReasoning {
                            effort: model
                                .reasoning_effort
                                .as_ref()
                                .map(|effort| effort.as_str().to_string()),
                            efforts: metadata
                                .selectable_reasoning_efforts()
                                .into_iter()
                                .map(|effort| effort.as_str().to_string())
                                .collect(),
                        },
                    }
                })
            })
            .collect(),
    }
}

pub(crate) fn apply_config_reload(
    agent: &mut Agent<async_openai::config::OpenAIConfig>,
    config_path: &std::path::Path,
    model_routes: &mut indexmap::IndexMap<String, ModelRoute>,
    route_api_key_configured: &mut indexmap::IndexMap<String, bool>,
    expert_model_routes: &mut indexmap::IndexMap<String, ModelRoute>,
    new_session_default_expert_routes: &mut indexmap::IndexMap<String, ModelRoute>,
    expert_allowed_models: &mut indexmap::IndexMap<String, Vec<ModelRoute>>,
    legacy_expert_models: &mut indexmap::IndexMap<String, String>,
    providers: &mut indexmap::IndexMap<String, ProviderConfig>,
    global_retry: &mut RetryConfig,
    provider_api_key_hints: &mut indexmap::IndexMap<String, String>,
    new_session_default_route: &mut ModelRoute,
    event_tx: &mpsc::UnboundedSender<SessionTransportEvent>,
) -> Result<()> {
    let config = AppConfig::load_from_path(config_path)?;
    let catalog_event = model_catalog_updated_event(&config);
    let previous_active_route = agent
        .primary_route()
        .cloned()
        .ok_or_else(|| anyhow!("active agent route is unavailable during configuration reload"))?;
    let next_new_session_default_route = config.active_route();
    config.resolve_route(&next_new_session_default_route)?;
    let current_route_available = config.resolve_route(&previous_active_route).is_ok();
    let primary_factory =
        ConfiguredPrimaryRouteFactory::new(config.providers.clone(), config.global.retry.clone());

    let next_model_routes = config
        .providers
        .iter()
        .flat_map(|(provider_name, provider)| {
            provider.models.keys().map(move |model| {
                let route = ModelRoute::new(provider_name, model);
                (route.display_name(), route)
            })
        })
        .collect::<indexmap::IndexMap<_, _>>();
    let next_route_api_key_configured = config
        .providers
        .iter()
        .flat_map(|(provider_name, provider)| {
            provider.models.keys().map(move |model| {
                let route = ModelRoute::new(provider_name, model);
                (route.display_name(), !provider.api_key.trim().is_empty())
            })
        })
        .collect::<indexmap::IndexMap<_, _>>();
    let next_new_session_default_expert_routes = crate::delegation::supported_agent_names()
        .filter_map(|name| {
            config
                .model_route_for(name)
                .cloned()
                .map(|route| (name.to_string(), route))
        })
        .collect::<indexmap::IndexMap<_, _>>();
    let next_expert_allowed_models = crate::delegation::supported_agent_names()
        .map(|name| {
            (
                name.to_string(),
                config
                    .agents
                    .allowed_models_for(name)
                    .unwrap_or_default()
                    .to_vec(),
            )
        })
        .collect::<indexmap::IndexMap<_, _>>();
    let next_legacy_expert_models = crate::delegation::supported_agent_names()
        .filter(|name| config.agents.follows_active_provider(name))
        .filter_map(|name| {
            config
                .model_route_for(name)
                .map(|route| (name.to_string(), route.model.clone()))
        })
        .collect::<indexmap::IndexMap<_, _>>();
    let next_provider_api_key_hints = config
        .providers
        .keys()
        .map(|name| {
            (
                name.clone(),
                format!(
                    "Set providers.{name}.api_key in {} or set {}.",
                    config.config_path.display(),
                    crate::config::provider_api_key_env_var(name)
                ),
            )
        })
        .collect::<indexmap::IndexMap<_, _>>();
    let mut session_providers = config.providers.clone();
    for route in expert_model_routes.values() {
        if config.resolve_route(route).is_ok() {
            continue;
        }
        let provider = providers.get(&route.provider).ok_or_else(|| {
            anyhow!(
                "current expert route provider '{}' is unavailable during configuration reload",
                route.provider
            )
        })?;
        if !provider.has_model(&route.model) {
            bail!(
                "current expert route '{}' is unavailable during configuration reload",
                route.display_name()
            );
        }
        session_providers
            .entry(route.provider.clone())
            .or_insert_with(|| provider.clone());
        if let Some(session_provider) = session_providers.get_mut(&route.provider)
            && !session_provider.has_model(&route.model)
        {
            let model = provider.models.get(&route.model).cloned().ok_or_else(|| {
                anyhow!(
                    "current expert route '{}' is unavailable during configuration reload",
                    route.display_name()
                )
            })?;
            session_provider.models.insert(route.model.clone(), model);
        }
    }
    let expert_factory = crate::subagent::ExpertRouteFactory::new_with_policies(
        crate::delegation::supported_agent_names().map(|name| {
            (
                name.to_string(),
                expert_model_routes.get(name).cloned(),
                next_expert_allowed_models
                    .get(name)
                    .cloned()
                    .unwrap_or_default(),
            )
        }),
        &session_providers,
        &config.global.retry,
    )?;
    let next_global_retry = config.global.retry.clone();
    let current_provider = current_route_available
        .then(|| config.providers.get(&previous_active_route.provider))
        .flatten();
    let next_agent_retry = current_provider
        .and_then(|provider| provider.retry.clone())
        .unwrap_or_else(|| agent.retry_config().clone());
    let next_parallelism = config
        .tools
        .parallelism
        .iter()
        .map(|(name, mode)| (name.clone(), *mode))
        .collect::<std::collections::BTreeMap<_, _>>();

    let providers_runtime_unchanged = providers_runtime_eq(providers, &session_providers);
    let maps_unchanged = *model_routes == next_model_routes
        && *route_api_key_configured == next_route_api_key_configured
        && *new_session_default_expert_routes == next_new_session_default_expert_routes
        && *expert_allowed_models == next_expert_allowed_models
        && *legacy_expert_models == next_legacy_expert_models
        && *provider_api_key_hints == next_provider_api_key_hints
        && *global_retry == next_global_retry;
    let settings_unchanged = agent.compaction_config() == &config.global.compaction
        && agent.tool_timeout_secs() == config.global.tool_timeout_secs
        && agent.retry_config() == &next_agent_retry
        && agent.tool_parallelism_overrides() == &next_parallelism;
    let previous_provider = providers.get(&previous_active_route.provider);
    let current_provider_runtime_unchanged = current_provider.is_none_or(|provider| {
        previous_provider.is_some_and(|previous| {
            previous.api_key == provider.api_key
                && previous.base_url == provider.base_url
                && previous.protocol == provider.protocol
        })
    });
    let next_model_protocols = current_provider.map(|provider| {
        provider
            .models
            .iter()
            .map(|(id, model)| (id.clone(), model.protocol))
            .collect::<HashMap<_, _>>()
    });
    let next_model_catalog = current_provider.map(|provider| {
        provider
            .models
            .iter()
            .map(|(id, model)| (id.clone(), model.request_metadata()))
            .collect::<HashMap<_, _>>()
    });
    let catalog_unchanged = current_provider.is_none_or(|provider| {
        agent.default_protocol() == provider.protocol
            && next_model_protocols
                .as_ref()
                .is_some_and(|protocols| agent.model_protocols() == protocols)
            && next_model_catalog
                .as_ref()
                .is_some_and(|catalog| agent.model_catalog() == catalog)
    });

    // Global config writes for non-reloadable fields (for example MCP enabled state)
    // and duplicate watcher events land here with no runtime delta. Stay silent.
    let new_session_default_unchanged =
        *new_session_default_route == next_new_session_default_route;
    if !reload_has_runtime_delta(
        providers_runtime_unchanged,
        maps_unchanged,
        settings_unchanged,
        current_provider_runtime_unchanged,
        catalog_unchanged,
        new_session_default_unchanged,
    ) {
        return Ok(());
    }

    // Fallible mutator first; remaining updates below are infallible.
    agent.set_tool_parallelism(next_parallelism)?;
    if agent.compaction_config() != &config.global.compaction {
        agent.set_compaction_config(config.global.compaction.clone());
    }
    if agent.tool_timeout_secs() != config.global.tool_timeout_secs {
        agent.set_tool_timeout_secs(config.global.tool_timeout_secs);
    }
    if agent.retry_config() != &next_agent_retry {
        agent.set_retry_config(next_agent_retry);
    }
    agent.set_primary_route_factory(Arc::new(primary_factory));
    agent.set_subagent_child_factory(Arc::new(expert_factory));
    if current_route_available && (!current_provider_runtime_unchanged || !catalog_unchanged) {
        let prepared = agent.prepare_primary_route(previous_active_route.clone())?;
        prepared.into_install().apply(agent);
    } else if !current_route_available {
        let _ = event_tx.send(SessionTransportEvent::Notice(NoticeEvent::info(format!(
            "Current model '{}' is no longer in the configured model catalog; this session will keep using its existing route until you switch models or start a new session",
            previous_active_route.display_name()
        ))));
    }

    *model_routes = next_model_routes;
    *route_api_key_configured = next_route_api_key_configured;
    for route in expert_model_routes.values() {
        let retained_credential = providers
            .get(&route.provider)
            .is_some_and(|provider| !provider.api_key.trim().is_empty());
        route_api_key_configured
            .entry(route.display_name())
            .or_insert(retained_credential);
    }
    *new_session_default_expert_routes = next_new_session_default_expert_routes;
    let changed_expert_allowed_models = next_expert_allowed_models
        .iter()
        .filter(|(name, routes)| {
            expert_allowed_models
                .get(*name)
                .map(Vec::as_slice)
                .unwrap_or_default()
                != routes.as_slice()
        })
        .map(|(name, routes)| {
            (
                name.clone(),
                routes
                    .iter()
                    .map(ModelRoute::display_name)
                    .collect::<Vec<_>>(),
            )
        })
        .collect::<Vec<_>>();
    *expert_allowed_models = next_expert_allowed_models;
    *legacy_expert_models = next_legacy_expert_models;
    *provider_api_key_hints = next_provider_api_key_hints;
    *providers = session_providers;
    *global_retry = next_global_retry;
    *new_session_default_route = next_new_session_default_route;
    let _ = event_tx.send(SessionTransportEvent::ModelCatalogUpdated(catalog_event));
    for (agent_name, model_ids) in changed_expert_allowed_models {
        let _ = event_tx.send(SessionTransportEvent::ExpertAllowedModelsChanged {
            agent_name,
            model_ids,
        });
    }
    let _ = event_tx.send(SessionTransportEvent::Notice(NoticeEvent::info(
        "configuration reloaded (supported runtime fields only; MCP, permissions, Fast Mode, max_iterations/max_tool_calls unchanged)",
    )));
    Ok(())
}

/// Compare reloadable provider fields, ignoring `default_model` which is often
/// rewritten by in-session model switches that already updated the live agent.
fn providers_runtime_eq(
    left: &indexmap::IndexMap<String, ProviderConfig>,
    right: &indexmap::IndexMap<String, ProviderConfig>,
) -> bool {
    if left.len() != right.len() {
        return false;
    }
    left.iter().all(|(name, left_provider)| {
        right.get(name).is_some_and(|right_provider| {
            left_provider.base_url == right_provider.base_url
                && left_provider.api_key == right_provider.api_key
                && left_provider.protocol == right_provider.protocol
                && left_provider.retry == right_provider.retry
                && left_provider.models == right_provider.models
        })
    })
}

pub(crate) fn create_config_watcher(
    config_path: &std::path::Path,
    reload_tx: mpsc::UnboundedSender<()>,
) -> Result<RecommendedWatcher> {
    let target = std::fs::canonicalize(config_path).unwrap_or_else(|_| config_path.to_path_buf());
    let watch_dir = target
        .parent()
        .unwrap_or_else(|| std::path::Path::new("."))
        .to_path_buf();
    let mut watcher = notify::recommended_watcher(move |event: notify::Result<Event>| {
        let Ok(event) = event else {
            // Transient watcher errors should not force a reload storm.
            return;
        };
        if matches!(
            event.kind,
            EventKind::Create(_) | EventKind::Modify(_) | EventKind::Remove(_)
        ) && event
            .paths
            .iter()
            .any(|path| path.file_name() == target.file_name())
        {
            let _ = reload_tx.send(());
        }
    })?;
    watcher.watch(&watch_dir, RecursiveMode::NonRecursive)?;
    Ok(watcher)
}

pub(crate) fn route_has_api_key(
    route_api_key_configured: &indexmap::IndexMap<String, bool>,
    route_display_name: &str,
) -> bool {
    route_api_key_configured
        .get(route_display_name)
        .copied()
        .unwrap_or(false)
}

pub(crate) fn route_api_key_hint(
    route_display_name: &str,
    provider_api_key_hints: &indexmap::IndexMap<String, String>,
    fallback_hint: &str,
) -> String {
    let provider = route_display_name
        .split_once('/')
        .map(|(provider, _)| provider)
        .unwrap_or("selected");
    provider_api_key_hints
        .get(provider)
        .cloned()
        .unwrap_or_else(|| fallback_hint.to_string())
}

pub(crate) fn active_route_has_api_key(
    agent: &Agent<async_openai::config::OpenAIConfig>,
    route_api_key_configured: &indexmap::IndexMap<String, bool>,
) -> bool {
    route_api_key_configured
        .get(&agent.route_display_name())
        .copied()
        .unwrap_or(true)
}

pub(crate) fn config_default_expert_routes_for_primary(
    configured_expert_routes: &indexmap::IndexMap<String, ModelRoute>,
    legacy_expert_models: &indexmap::IndexMap<String, String>,
    primary_route: &ModelRoute,
) -> indexmap::IndexMap<String, ModelRoute> {
    let mut routes = configured_expert_routes.clone();
    for (agent_name, model) in legacy_expert_models {
        routes.insert(
            agent_name.clone(),
            ModelRoute::new(primary_route.provider.clone(), model.clone()),
        );
    }
    routes
}

pub(crate) fn reviewer_policy_changed(
    previous_primary_route: Option<&ModelRoute>,
    current_primary_route: Option<&ModelRoute>,
    previous_expert_model_routes: &indexmap::IndexMap<String, ModelRoute>,
    current_expert_model_routes: &indexmap::IndexMap<String, ModelRoute>,
    previous_expert_allowed_models: &indexmap::IndexMap<String, Vec<ModelRoute>>,
    current_expert_allowed_models: &indexmap::IndexMap<String, Vec<ModelRoute>>,
) -> bool {
    previous_primary_route != current_primary_route
        || previous_expert_model_routes.get("reviewer")
            != current_expert_model_routes.get("reviewer")
        || previous_expert_allowed_models.get("reviewer")
            != current_expert_allowed_models.get("reviewer")
}

pub(crate) fn expert_routes_after_primary_switch(
    expert_model_routes: &indexmap::IndexMap<String, ModelRoute>,
    legacy_expert_models: &indexmap::IndexMap<String, String>,
    providers: &indexmap::IndexMap<String, ProviderConfig>,
    primary_route: &ModelRoute,
) -> Result<indexmap::IndexMap<String, ModelRoute>> {
    let mut routes = expert_model_routes.clone();
    let provider = providers.get(&primary_route.provider).ok_or_else(|| {
        anyhow!(
            "provider '{}' is not configured for expert route updates",
            primary_route.provider
        )
    })?;
    for (agent_name, model) in legacy_expert_models {
        if !provider.has_model(model) {
            bail!(
                "expert '{agent_name}' model '{model}' is not configured for provider '{}'",
                primary_route.provider
            );
        }
        routes.insert(
            agent_name.clone(),
            ModelRoute::new(primary_route.provider.clone(), model.clone()),
        );
    }
    Ok(routes)
}

#[cfg(test)]
mod expert_route_switch_tests {
    use super::*;

    fn provider(models: &[&str]) -> ProviderConfig {
        ProviderConfig {
            base_url: "http://127.0.0.1:9/v1".into(),
            api_key: "key".into(),
            protocol: crate::config::ApiProtocol::Completions,
            default_model: models.first().copied().unwrap_or_default().into(),
            retry: None,
            models: models
                .iter()
                .map(|model| {
                    (
                        (*model).to_string(),
                        crate::config::ModelConfig {
                            display_name: None,
                            protocol: crate::config::ApiProtocol::Completions,
                            context_window: None,
                            effective_input_limit_tokens: None,
                            max_output_tokens: None,
                            supports_tools: false,
                            supports_reasoning: false,
                            reasoning_effort: None,
                            reasoning_efforts: Vec::new(),
                            reasoning_summary: None,
                            text_verbosity: None,
                            temperature: None,
                            top_p: None,
                            prompt_cache: crate::config::PromptCacheConfig::default(),
                            parallel_tool_calls: false,
                        },
                    )
                })
                .collect(),
        }
    }

    #[test]
    fn default_route_only_reload_is_not_treated_as_noop() {
        assert!(reload_has_runtime_delta(
            true, true, true, true, true, false
        ));
        assert!(!reload_has_runtime_delta(
            true, true, true, true, true, true
        ));
    }

    #[test]
    fn inherited_reviewer_policy_changes_when_primary_route_changes() {
        let old_primary = ModelRoute::new("old", "shared");
        let new_primary = ModelRoute::new("new", "shared");
        let expert_routes = indexmap::IndexMap::new();
        let allowed_models = indexmap::IndexMap::new();

        assert!(reviewer_policy_changed(
            Some(&old_primary),
            Some(&new_primary),
            &expert_routes,
            &expert_routes,
            &allowed_models,
            &allowed_models,
        ));
    }

    #[test]
    fn configured_expert_defaults_do_not_inherit_current_session_overrides() {
        let configured = indexmap::IndexMap::from([(
            "reviewer".into(),
            ModelRoute::new("configured", "reviewer"),
        )]);
        let legacy = indexmap::IndexMap::from([("explorer".into(), "legacy".into())]);
        let primary = ModelRoute::new("next", "primary");

        let routes = config_default_expert_routes_for_primary(&configured, &legacy, &primary);

        assert_eq!(
            routes.get("reviewer"),
            Some(&ModelRoute::new("configured", "reviewer"))
        );
        assert_eq!(
            routes.get("explorer"),
            Some(&ModelRoute::new("next", "legacy"))
        );
        assert!(!routes.contains_key("fixer"));
    }

    #[test]
    fn unrelated_expert_reload_does_not_change_reviewer_policy() {
        let primary = ModelRoute::new("primary", "shared");
        let previous_routes = indexmap::IndexMap::from([(
            "explorer".into(),
            ModelRoute::new("primary", "old-explorer"),
        )]);
        let current_routes = indexmap::IndexMap::from([(
            "explorer".into(),
            ModelRoute::new("primary", "new-explorer"),
        )]);
        let allowed_models = indexmap::IndexMap::new();

        assert!(!reviewer_policy_changed(
            Some(&primary),
            Some(&primary),
            &previous_routes,
            &current_routes,
            &allowed_models,
            &allowed_models,
        ));
    }

    #[test]
    fn primary_switch_rejects_missing_legacy_expert_model_without_mutating_routes() {
        let routes = indexmap::IndexMap::from([(
            "reviewer".into(),
            ModelRoute::new("fixed", "reviewer-model"),
        )]);
        let legacy = indexmap::IndexMap::from([("explorer".into(), "legacy-model".into())]);
        let providers = indexmap::IndexMap::from([("next".into(), provider(&["primary-model"]))]);

        let error = expert_routes_after_primary_switch(
            &routes,
            &legacy,
            &providers,
            &ModelRoute::new("next", "primary-model"),
        )
        .expect_err("missing follows-active-provider model must fail before switching");

        assert!(error.to_string().contains(
            "expert 'explorer' model 'legacy-model' is not configured for provider 'next'"
        ));
        assert_eq!(
            routes.get("reviewer"),
            Some(&ModelRoute::new("fixed", "reviewer-model"))
        );
    }
}
