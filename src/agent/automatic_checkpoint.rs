#[cfg(test)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct AutoCheckpointPolicy {
    pub enabled: bool,
    pub max_automatic_per_turn: u8,
}
#[cfg(test)]
impl AutoCheckpointPolicy {
    pub(super) fn from_config(config: crate::config::LogicalCheckpointConfig) -> Self {
        Self {
            enabled: config.enabled && config.automatic,
            max_automatic_per_turn: config.max_automatic_per_turn,
        }
    }
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
#[cfg(test)]
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
    use crate::request_builder::BudgetReport;

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
        let classified = budget.request_classification();
        assert_eq!(classified.reserve, 2_048);
        assert_eq!(classified.high_watermark, 8_452);
        assert!(classified.safe);
        budget.estimated_request_tokens = classified.high_watermark;
        assert!(!budget.request_classification().safe, "equality is unsafe");
        budget.estimated_request_tokens += 1;
        assert!(!budget.request_classification().safe, "above is unsafe");
        budget.estimated_request_tokens = 1;
        budget.truncated = true;
        assert!(
            !budget.request_classification().safe,
            "truncation is unsafe"
        );
        budget.input_budget_tokens = 100;
        budget.truncated = false;
        budget.estimated_tools_tokens = 0;
        let small = budget.request_classification();
        assert_eq!(small.reserve, 100);
        assert_eq!(small.high_watermark, 0);
        assert_eq!(small.hard_request_limit, 100);
    }
}
