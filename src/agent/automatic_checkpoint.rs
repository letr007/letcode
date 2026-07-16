use crate::config::LogicalCheckpointConfig;
use crate::request_builder::BudgetReport;

pub(super) const EMERGENCY_RESERVE_TOKENS: u64 = 2_048;

/// Fixed request admission for the canonical prompt path.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct RequestBudgetClassification {
    pub prompt_limit: u64,
    pub reserve: u64,
    pub hard_request_limit: u64,
    pub high_watermark: u64,
    pub safe: bool,
}

pub(super) fn classify_request_budget(budget: &BudgetReport) -> RequestBudgetClassification {
    let prompt_limit = budget.input_budget_tokens;
    let reserve = EMERGENCY_RESERVE_TOKENS.min(prompt_limit);
    let hard_request_limit = prompt_limit.saturating_add(budget.estimated_tools_tokens);
    let high_watermark = prompt_limit
        .saturating_sub(reserve)
        .saturating_add(budget.estimated_tools_tokens);
    RequestBudgetClassification {
        prompt_limit,
        reserve,
        hard_request_limit,
        high_watermark,
        safe: !budget.truncated && budget.estimated_request_tokens < high_watermark,
    }
}

/// Configuration normalized for the pure automatic checkpoint scheduler.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct AutoCheckpointPolicy {
    pub enabled: bool,
    pub max_automatic_per_turn: u8,
}

impl AutoCheckpointPolicy {
    pub(super) fn from_config(config: LogicalCheckpointConfig) -> Self {
        Self {
            enabled: config.enabled && config.automatic,
            max_automatic_per_turn: config.max_automatic_per_turn,
        }
    }

    /// Retains the established operational authorization and fresh-boundary gates.
    pub(super) fn authorizes_checkpoint(self, state: AutoCheckpointSchedulerView) -> bool {
        self.enabled
            && !state.suppressed
            && state.armed
            && state.boundary_available
            && !state.boundary_consumed
            && !state.boundary_attempted
            && state.automatic_commits < self.max_automatic_per_turn
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct AutoCheckpointSchedulerView {
    pub armed: bool,
    pub automatic_commits: u8,
    pub boundary_available: bool,
    pub boundary_consumed: bool,
    pub boundary_attempted: bool,
    pub suppressed: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn policy() -> AutoCheckpointPolicy {
        AutoCheckpointPolicy {
            enabled: true,
            max_automatic_per_turn: 2,
        }
    }

    fn view() -> AutoCheckpointSchedulerView {
        AutoCheckpointSchedulerView {
            armed: true,
            automatic_commits: 0,
            boundary_available: true,
            boundary_consumed: false,
            boundary_attempted: false,
            suppressed: false,
        }
    }
    fn budget(input_budget_tokens: u64, estimated_request_tokens: u64) -> BudgetReport {
        BudgetReport {
            context_window_tokens: 0,
            input_budget_tokens,
            estimated_request_tokens,
            estimated_prelude_tokens: 0,
            estimated_protected_tokens: 0,
            protected_safe_ceiling_tokens: 0,
            protected_reserve_tokens: 0,
            estimated_foldable_protected_tokens: 0,
            estimated_provider_folded_protected_tokens: 0,
            estimated_unaddressable_protected_tokens: 0,
            provider_folded_output_count: 0,
            estimated_retained_history_tokens: 0,
            estimated_tools_tokens: 0,
            estimated_evidence_tokens: 0,
            estimated_required_fallback_tokens: 0,
            original_history_items: 0,
            retained_history_items: 0,
            dropped_history_items: 0,
            selected_evidence_items: 0,
            dropped_evidence_items: 0,
            truncated: false,
            plan_total_prompt_tokens: 0,
            plan_stable_prompt_tokens: 0,
            plan_volatile_prompt_tokens: 0,
            plan_cacheable_prefix_tokens: 0,
            plan_stable_after_boundary_tokens: 0,
        }
    }

    #[test]
    fn request_budget_uses_fixed_reserve_and_strict_high_watermark() {
        let mut budget = BudgetReport {
            estimated_tools_tokens: 500,
            ..budget(10_000, 8_451)
        };
        let classified = classify_request_budget(&budget);
        assert_eq!(classified.reserve, 2_048);
        assert_eq!(classified.high_watermark, 8_452);
        assert!(classified.safe);
        budget.estimated_request_tokens = classified.high_watermark;
        assert!(!classify_request_budget(&budget).safe, "equality is unsafe");
        budget.estimated_request_tokens += 1;
        assert!(!classify_request_budget(&budget).safe, "above is unsafe");
        budget.estimated_request_tokens = 1;
        budget.truncated = true;
        assert!(
            !classify_request_budget(&budget).safe,
            "truncation is unsafe"
        );
        budget.input_budget_tokens = 100;
        budget.truncated = false;
        budget.estimated_tools_tokens = 0;
        let small = classify_request_budget(&budget);
        assert_eq!(small.reserve, 100);
        assert_eq!(small.high_watermark, 0);
        assert_eq!(small.hard_request_limit, 100);
    }

    #[test]
    fn authorization_requires_enabled_automatic_configuration_and_available_boundary() {
        let disabled = AutoCheckpointPolicy::from_config(LogicalCheckpointConfig::default());
        assert!(!disabled.authorizes_checkpoint(view()));

        let mut config = LogicalCheckpointConfig {
            enabled: true,
            automatic: false,
            ..Default::default()
        };
        assert!(!AutoCheckpointPolicy::from_config(config).authorizes_checkpoint(view()));
        config.automatic = true;
        let policy = AutoCheckpointPolicy::from_config(config);
        assert!(policy.authorizes_checkpoint(view()));

        for blocked in [
            AutoCheckpointSchedulerView {
                armed: false,
                ..view()
            },
            AutoCheckpointSchedulerView {
                boundary_available: false,
                ..view()
            },
            AutoCheckpointSchedulerView {
                boundary_consumed: true,
                ..view()
            },
            AutoCheckpointSchedulerView {
                boundary_attempted: true,
                ..view()
            },
            AutoCheckpointSchedulerView {
                suppressed: true,
                ..view()
            },
        ] {
            assert!(!policy.authorizes_checkpoint(blocked));
        }
    }

    #[test]
    fn authorization_respects_maximum_automatic_checkpoints_per_turn() {
        let policy = policy();
        assert!(policy.authorizes_checkpoint(view()));
        assert!(!policy.authorizes_checkpoint(AutoCheckpointSchedulerView {
            automatic_commits: policy.max_automatic_per_turn,
            ..view()
        }));
    }
}
