use anyhow::{Result, anyhow, bail};
use std::collections::HashMap;

use crate::agent::{Agent, AgentFactory, AgentTemplate, PrimaryRouteFactory, SubagentChildFactory};
use crate::config::{ModelRoute, ProviderConfig, RetryConfig};
use crate::model_runtime::ResolvedRuntimeCatalog;

#[derive(Clone)]
pub struct ExpertRouteFactory {
    policies: HashMap<String, ExpertRoutePolicy>,
    providers: indexmap::IndexMap<String, ProviderConfig>,
    global_retry: RetryConfig,
    runtime_catalog: Option<ResolvedRuntimeCatalog>,
}

#[derive(Clone)]
struct ExpertRoutePolicy {
    default_route: Option<ModelRoute>,
    allowed_models: Vec<ModelRoute>,
}

impl ExpertRouteFactory {
    fn prepare_route(&self, route: ModelRoute) -> Result<crate::agent::PreparedPrimaryRoute> {
        self.prepare_route_with_runtime_route(route, None)
    }

    fn prepare_route_with_runtime_route(
        &self,
        route: ModelRoute,
        inherited_runtime_route: Option<std::sync::Arc<crate::model_runtime::ResolvedModelRoute>>,
    ) -> Result<crate::agent::PreparedPrimaryRoute> {
        let provider = self.providers.get(&route.provider).ok_or_else(|| {
            anyhow!(
                "child route provider '{}' is not configured",
                route.provider
            )
        })?;
        if !provider.has_model(&route.model) {
            bail!(
                "child route provider '{}' model '{}' is not configured",
                route.provider,
                route.model
            );
        }
        let runtime_route = inherited_runtime_route.or_else(|| {
            self.runtime_catalog
                .as_ref()
                .and_then(|catalog| catalog.route(&route.provider, &route.model))
                .cloned()
                .map(std::sync::Arc::new)
        });
        let prepared = crate::agent::PreparedPrimaryRoute::new_with_runtime_route(
            route.clone(),
            provider.protocol,
            provider
                .models
                .iter()
                .map(|(id, model)| (id.clone(), model.protocol))
                .collect(),
            provider
                .models
                .iter()
                .map(|(id, model)| (id.clone(), model.request_metadata()))
                .collect(),
            provider
                .retry
                .clone()
                .unwrap_or_else(|| self.global_retry.clone()),
            runtime_route,
        );
        Ok(match self.runtime_catalog.clone() {
            Some(catalog) => prepared.with_runtime_catalog(catalog),
            None => prepared,
        })
    }

    #[allow(dead_code)]
    pub fn new(
        routes: impl IntoIterator<Item = (String, ModelRoute)>,
        providers: &indexmap::IndexMap<String, ProviderConfig>,
        global_retry: &RetryConfig,
    ) -> Result<Self> {
        Self::new_with_policies(
            routes
                .into_iter()
                .map(|(name, route)| (name, Some(route), Vec::new())),
            providers,
            global_retry,
        )
    }

    pub fn new_with_policies(
        policies: impl IntoIterator<Item = (String, Option<ModelRoute>, Vec<ModelRoute>)>,
        providers: &indexmap::IndexMap<String, ProviderConfig>,
        global_retry: &RetryConfig,
    ) -> Result<Self> {
        let mut prepared = HashMap::new();
        for (agent_name, default_route, allowed_models) in policies {
            if let Some(route) = &default_route {
                Self::validate_configured_route(providers, &agent_name, "default", route)?;
            }
            for route in &allowed_models {
                Self::validate_configured_route(providers, &agent_name, "allowed", route)?;
            }
            prepared.insert(
                agent_name,
                ExpertRoutePolicy {
                    default_route,
                    allowed_models,
                },
            );
        }
        Ok(Self {
            policies: prepared,
            providers: providers.clone(),
            global_retry: global_retry.clone(),
            runtime_catalog: None,
        })
    }

    pub fn with_runtime_catalog(mut self, runtime_catalog: ResolvedRuntimeCatalog) -> Self {
        self.runtime_catalog = Some(runtime_catalog);
        self
    }

    fn validate_configured_route(
        providers: &indexmap::IndexMap<String, ProviderConfig>,
        agent_name: &str,
        kind: &str,
        route: &ModelRoute,
    ) -> Result<()> {
        let provider = providers.get(&route.provider).ok_or_else(|| {
            anyhow!(
                "expert {kind} route for '{agent_name}' references unknown provider '{}'",
                route.provider
            )
        })?;
        if !provider.has_model(&route.model) {
            bail!(
                "expert {kind} route for '{agent_name}' references unknown model '{}' under provider '{}'",
                route.model,
                route.provider
            );
        }
        Ok(())
    }
}

impl PrimaryRouteFactory for ExpertRouteFactory {
    fn prepare_route(&self, route: ModelRoute) -> Result<crate::agent::PreparedPrimaryRoute> {
        self.prepare_route(route)
    }
}

impl SubagentChildFactory for ExpertRouteFactory {
    fn resolve_route(
        &self,
        parent: &Agent,
        template: &AgentTemplate,
        requested_route: Option<&ModelRoute>,
        takeover: bool,
    ) -> Result<ModelRoute> {
        let policy = self
            .policies
            .get(&template.name)
            .ok_or_else(|| anyhow!("no route policy configured for expert '{}'", template.name))?;
        if let Some(route) = requested_route {
            if takeover && parent.prepare_primary_route(route.clone()).is_ok() {
                return Ok(route.clone());
            }
            let effective_default = policy
                .default_route
                .as_ref()
                .or_else(|| parent.primary_route());
            let allowed = policy.allowed_models.iter().any(|allowed| allowed == route);
            let default_takeover = takeover && effective_default == Some(route);
            if !allowed && !default_takeover {
                let action = if takeover { "historical" } else { "requested" };
                bail!(
                    "{action} model route '{}' is not allowed for expert '{}'",
                    route.display_name(),
                    template.name
                );
            }
            parent.prepare_primary_route(route.clone())?;
            return Ok(route.clone());
        }
        if takeover {
            bail!("takeover requires a recorded provider/model route");
        }
        policy
            .default_route
            .clone()
            .or_else(|| parent.primary_route().cloned())
            .ok_or_else(|| {
                anyhow!(
                    "parent model route is unavailable for expert '{}'",
                    template.name
                )
            })
    }

    fn create_child(
        &self,
        parent: &Agent,
        template: &AgentTemplate,
        route: &ModelRoute,
        max_tool_calls_override: Option<usize>,
    ) -> Result<Agent> {
        let prepared = parent.prepare_primary_route(route.clone())?;
        Ok(
            AgentFactory::create_prepared_routed_child_with_max_tool_calls(
                parent,
                template,
                prepared,
                max_tool_calls_override,
            ),
        )
    }
}
