//! Session engine configuration reload support.

use super::*;
use crate::agent::PrimaryRouteFactory;

pub(crate) fn reload_has_runtime_delta(
    runtime_fingerprint_unchanged: bool,
    maps_unchanged: bool,
    settings_unchanged: bool,
    current_route_runtime_unchanged: bool,
    catalog_unchanged: bool,
    new_session_default_unchanged: bool,
) -> bool {
    !(runtime_fingerprint_unchanged
        && maps_unchanged
        && settings_unchanged
        && current_route_runtime_unchanged
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
                            effort: metadata
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
    agent: &mut Agent,
    config_path: &std::path::Path,
    model_routes: &mut indexmap::IndexMap<String, ModelRoute>,
    route_api_key_configured: &mut indexmap::IndexMap<String, bool>,
    expert_default_routes: &mut indexmap::IndexMap<String, ModelRoute>,
    expert_allowed_models: &mut indexmap::IndexMap<String, Vec<ModelRoute>>,
    providers: &mut indexmap::IndexMap<String, ProviderConfig>,
    global_retry: &mut RetryConfig,
    provider_api_key_hints: &mut indexmap::IndexMap<String, String>,
    new_session_default_route: &mut ModelRoute,
    runtime_catalog: &mut crate::model_runtime::ResolvedRuntimeCatalog,
    event_tx: &mpsc::UnboundedSender<SessionTransportEvent>,
) -> Result<()> {
    let config = AppConfig::load_from_path(config_path)?;
    let catalog_event = model_catalog_updated_event(&config);
    let next_runtime_catalog = config.runtime_catalog.clone();
    let next_runtime_fingerprint = next_runtime_catalog.fingerprint().clone();
    let previous_active_route = agent
        .primary_route()
        .cloned()
        .ok_or_else(|| anyhow!("active agent route is unavailable during configuration reload"))?;
    let next_new_session_default_route = config.active_route();
    config.resolve_route(&next_new_session_default_route)?;
    let current_route_available = config.resolve_route(&previous_active_route).is_ok();
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
    let mut next_route_api_key_configured = config
        .providers
        .iter()
        .flat_map(|(provider_name, provider)| {
            provider.models.keys().map(move |model| {
                let route = ModelRoute::new(provider_name, model);
                (route.display_name(), !provider.api_key.trim().is_empty())
            })
        })
        .collect::<indexmap::IndexMap<_, _>>();
    if !current_route_available {
        let configured = providers
            .get(&previous_active_route.provider)
            .is_some_and(|provider| !provider.api_key.trim().is_empty());
        next_route_api_key_configured.insert(previous_active_route.display_name(), configured);
    }
    let next_expert_default_routes = crate::delegation::supported_agent_names()
        .filter_map(|name| {
            config
                .expert_route_for(name)
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
    let next_provider_api_key_hints = config
        .providers
        .keys()
        .map(|name| {
            (
                name.clone(),
                format!(
                    "Set [providers.{name}.auth].credential in {} or set {} environment variable.",
                    config.config_path.display(),
                    crate::config::provider_api_key_env_var(name)
                ),
            )
        })
        .collect::<indexmap::IndexMap<_, _>>();
    let mut session_providers = config.providers.clone();
    if !current_route_available {
        let provider = providers
            .get(&previous_active_route.provider)
            .ok_or_else(|| {
                anyhow!(
                    "current route provider '{}' is unavailable during configuration reload",
                    previous_active_route.provider
                )
            })?;
        if !provider.has_model(&previous_active_route.model) {
            bail!(
                "current route '{}' is unavailable during configuration reload",
                previous_active_route.display_name()
            );
        }
        session_providers
            .entry(previous_active_route.provider.clone())
            .or_insert_with(|| provider.clone());
        if let Some(session_provider) = session_providers.get_mut(&previous_active_route.provider)
            && !session_provider.has_model(&previous_active_route.model)
        {
            let model = provider
                .models
                .get(&previous_active_route.model)
                .cloned()
                .ok_or_else(|| {
                    anyhow!(
                        "current route '{}' is unavailable during configuration reload",
                        previous_active_route.display_name()
                    )
                })?;
            session_provider
                .models
                .insert(previous_active_route.model.clone(), model);
        }
    }
    let primary_session_providers = session_providers.clone();
    let primary_factory = ConfiguredPrimaryRouteFactory::new_with_runtime_catalog(
        primary_session_providers,
        config.global.retry.clone(),
        config.runtime_catalog.clone(),
    );
    let prepared_current_route = if current_route_available {
        Some(primary_factory.prepare_route(previous_active_route.clone())?)
    } else {
        None
    };
    let expert_factory = crate::subagent::ExpertRouteFactory::new_with_policies(
        crate::delegation::supported_agent_names().map(|name| {
            (
                name.to_string(),
                next_expert_default_routes.get(name).cloned(),
                next_expert_allowed_models
                    .get(name)
                    .cloned()
                    .unwrap_or_default(),
            )
        }),
        &config.providers,
        &config.global.retry,
    )?
    .with_runtime_catalog(config.runtime_catalog.clone());
    let next_global_retry = config.global.retry.clone();
    let current_provider = current_route_available
        .then(|| config.providers.get(&previous_active_route.provider))
        .flatten();
    let next_agent_retry = if let Some(provider) = current_provider {
        provider
            .retry
            .clone()
            .unwrap_or_else(|| next_global_retry.clone())
    } else {
        agent.retry_config().clone()
    };
    let next_parallelism = config
        .tools
        .parallelism
        .iter()
        .map(|(name, mode)| (name.clone(), *mode))
        .collect::<std::collections::BTreeMap<_, _>>();

    let runtime_fingerprint_unchanged = runtime_catalog.fingerprint() == &next_runtime_fingerprint;
    let maps_unchanged = *model_routes == next_model_routes
        && *route_api_key_configured == next_route_api_key_configured
        && *expert_default_routes == next_expert_default_routes
        && *expert_allowed_models == next_expert_allowed_models
        && *provider_api_key_hints == next_provider_api_key_hints
        && *global_retry == next_global_retry;
    let settings_unchanged = agent.compaction_config() == &config.global.compaction
        && agent.tool_timeout_secs() == config.global.tool_timeout_secs
        && agent.retry_config() == &next_agent_retry
        && agent.tool_parallelism_overrides() == &next_parallelism
        && agent.fake_config() == &config.fake;
    let current_route_runtime_unchanged = route_runtime_fingerprint_eq(
        runtime_catalog.fingerprint(),
        &next_runtime_fingerprint,
        &previous_active_route,
    );
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
        runtime_fingerprint_unchanged,
        maps_unchanged,
        settings_unchanged,
        current_route_runtime_unchanged,
        catalog_unchanged,
        new_session_default_unchanged,
    ) {
        if agent.resolved_runtime_catalog().is_none() {
            agent.set_resolved_runtime_catalog(Some(runtime_catalog.clone()));
        }
        return Ok(());
    }

    // Intentionally perform the only remaining fallible mutation first; all later
    // agent mutations are infallible for these already validated inputs.
    agent.set_tool_parallelism(next_parallelism)?;
    if agent.fake_config() != &config.fake {
        agent.set_fake_config(config.fake.clone());
    }
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
    if current_route_available && (!current_route_runtime_unchanged || !catalog_unchanged) {
        if let Some(prepared) = prepared_current_route {
            prepared.into_install().apply(agent);
        }
        if agent
            .fake_client()
            .is_some_and(|client| !client.supports_protocol(agent.active_protocol()))
        {
            agent.set_fake_client(None)?;
            let _ = event_tx.send(SessionTransportEvent::FakeClientChanged { client: None });
            let _ = event_tx.send(SessionTransportEvent::Notice(NoticeEvent::info(
                "Fake mode disabled: unsupported by the reloaded model protocol",
            )));
        }
    } else if !current_route_available {
        let _ = event_tx.send(SessionTransportEvent::Notice(NoticeEvent::info(format!(
            "Current model '{}' is no longer in the configured model catalog; this session will keep using its existing route until you switch models or start a new session",
            previous_active_route.display_name()
        ))));
    }

    *model_routes = next_model_routes;
    *route_api_key_configured = next_route_api_key_configured;
    if !current_route_available {
        let retained_credential = providers
            .get(&previous_active_route.provider)
            .is_some_and(|provider| !provider.api_key.trim().is_empty());
        route_api_key_configured
            .entry(previous_active_route.display_name())
            .or_insert(retained_credential);
    }
    *expert_default_routes = next_expert_default_routes;
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
    *provider_api_key_hints = next_provider_api_key_hints;
    *providers = session_providers;
    *global_retry = next_global_retry;
    *new_session_default_route = next_new_session_default_route;
    agent.set_resolved_runtime_catalog(Some(next_runtime_catalog.clone()));
    *runtime_catalog = next_runtime_catalog;
    let _ = event_tx.send(SessionTransportEvent::ModelCatalogUpdated(catalog_event));
    for (agent_name, model_ids) in changed_expert_allowed_models {
        let _ = event_tx.send(SessionTransportEvent::ExpertAllowedModelsChanged {
            agent_name,
            model_ids,
        });
    }
    Ok(())
}

pub(crate) fn route_runtime_fingerprint_eq(
    left: &crate::model_runtime::RuntimeFingerprint,
    right: &crate::model_runtime::RuntimeFingerprint,
    route: &ModelRoute,
) -> bool {
    let left_provider = left.providers.get(&route.provider);
    let right_provider = right.providers.get(&route.provider);
    if left_provider.is_none() && right_provider.is_none() {
        return true;
    }
    let (Some(left_provider), Some(right_provider)) = (left_provider, right_provider) else {
        return false;
    };
    left_provider.flavor == right_provider.flavor
        && left_provider.auth_mode == right_provider.auth_mode
        && left_provider.auth_name == right_provider.auth_name
        && left_provider.credential_fingerprint == right_provider.credential_fingerprint
        && left_provider.retry == right_provider.retry
        && left_provider.endpoint == right_provider.endpoint
        && left_provider.headers == right_provider.headers
        && left_provider.query == right_provider.query
        && left_provider.connect_timeout_secs == right_provider.connect_timeout_secs
        && left_provider.no_proxy_loopback == right_provider.no_proxy_loopback
        && left_provider.models.get(&route.model) == right_provider.models.get(&route.model)
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
    agent: &Agent,
    route_api_key_configured: &indexmap::IndexMap<String, bool>,
) -> bool {
    route_api_key_configured
        .get(&agent.route_display_name())
        .copied()
        .unwrap_or(true)
}

pub(crate) fn reviewer_policy_changed(
    previous_expert_default_routes: &indexmap::IndexMap<String, ModelRoute>,
    current_expert_default_routes: &indexmap::IndexMap<String, ModelRoute>,
    previous_expert_allowed_models: &indexmap::IndexMap<String, Vec<ModelRoute>>,
    current_expert_allowed_models: &indexmap::IndexMap<String, Vec<ModelRoute>>,
) -> bool {
    previous_expert_default_routes.get("reviewer") != current_expert_default_routes.get("reviewer")
        || previous_expert_allowed_models.get("reviewer")
            != current_expert_allowed_models.get("reviewer")
}

#[cfg(test)]
mod expert_route_switch_tests {
    use super::*;

    fn provider(models: &[&str]) -> ProviderConfig {
        ProviderConfig {
            base_url: "http://127.0.0.1:9/v1".into(),
            jev_endpoint: None,
            auth_mode: crate::config::ProviderAuthMode::ApiKey,
            api_key: "key".into(),
            protocol: crate::config::ApiProtocol::Completions,
            default_model: models.first().copied().unwrap_or_default().into(),
            retry: None,
            reviewer: None,
            models: models
                .iter()
                .map(|model| {
                    (
                        (*model).to_string(),
                        crate::config::ModelConfig {
                            display_name: None,
                            protocol: crate::config::ApiProtocol::Completions,
                            anthropic_thinking: Default::default(),
                            anthropic_betas: Vec::new(),
                            context_window: None,
                            effective_input_limit_tokens: None,
                            max_output_tokens: None,
                            supports_tools: false,
                            supports_input_images: false,
                            supports_tool_result_images: false,
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
    fn route_fingerprint_detects_websocket_transport_changes() {
        let config = |websocket: bool| {
            let websocket = if websocket {
                "[providers.vendor.models.model.transport]\nwebsocket = true\n"
            } else {
                ""
            };
            crate::model_runtime::RuntimeConfig::from_toml(&format!(
                r#"active_provider = "vendor"
[providers.vendor]
protocol = "responses"
default_model = "model"
[providers.vendor.auth]
type = "none"
[providers.vendor.endpoints]
base_url = "https://example.invalid/v1"
[providers.vendor.models.model]
{websocket}[providers.vendor.models.other]
"#
            ))
            .unwrap()
            .resolve(&crate::model_runtime::ProtocolRegistry::builtins())
            .unwrap()
            .fingerprint()
            .clone()
        };
        let disabled = config(false);
        let enabled = config(true);
        assert!(!route_runtime_fingerprint_eq(
            &disabled,
            &enabled,
            &ModelRoute::new("vendor", "model"),
        ));
        assert!(route_runtime_fingerprint_eq(
            &disabled,
            &enabled,
            &ModelRoute::new("vendor", "other"),
        ));
    }

    #[test]
    fn unrelated_expert_reload_does_not_change_reviewer_policy() {
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
            &previous_routes,
            &current_routes,
            &allowed_models,
            &allowed_models,
        ));
    }
}
