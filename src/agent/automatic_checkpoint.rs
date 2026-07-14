use crate::config::LogicalCheckpointConfig;
use crate::request_builder::BudgetReport;

/// Configuration normalized for the pure automatic checkpoint scheduler.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct AutoCheckpointPolicy {
    pub enabled: bool,
    pub trigger_reserve_percent: u8,
    pub max_automatic_per_turn: u8,
}

impl AutoCheckpointPolicy {
    pub(super) fn from_config(config: LogicalCheckpointConfig) -> Self {
        Self {
            enabled: config.enabled && config.automatic,
            trigger_reserve_percent: config.trigger_reserve_percent,
            max_automatic_per_turn: config.max_automatic_per_turn,
        }
    }

    fn high_watermark(self, pressure: AutoCheckpointPressure) -> u64 {
        let increment = pressure
            .reserve_tokens
            .saturating_mul(u64::from(self.trigger_reserve_percent))
            .saturating_add(99)
            / 100;
        pressure.safe_ceiling_tokens.saturating_add(increment)
    }

    pub(super) fn decide(
        self,
        pressure: Option<AutoCheckpointPressure>,
        hard_protected_overflow: bool,
        state: AutoCheckpointSchedulerView,
    ) -> AutoCheckpointDecision {
        if !self.enabled || state.suppressed {
            return AutoCheckpointDecision::Suppress;
        }
        if let Some(pressure) = pressure
            && pressure.final_protected_tokens <= pressure.safe_ceiling_tokens
        {
            return AutoCheckpointDecision::Rearm;
        }
        if !state.armed
            || !state.boundary_available
            || state.boundary_consumed
            || state.boundary_attempted
            || state.automatic_commits >= self.max_automatic_per_turn
        {
            return AutoCheckpointDecision::Suppress;
        }
        if hard_protected_overflow {
            return AutoCheckpointDecision::Trigger(AutoCheckpointTrigger::HardProtectedOverflow);
        }
        let Some(pressure) = pressure else {
            return AutoCheckpointDecision::Suppress;
        };
        if pressure.reserve_tokens == 0 {
            return AutoCheckpointDecision::Suppress;
        }
        if pressure.final_protected_tokens >= self.high_watermark(pressure) {
            AutoCheckpointDecision::Trigger(AutoCheckpointTrigger::SoftPressure)
        } else {
            AutoCheckpointDecision::Suppress
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct AutoCheckpointPressure {
    pub safe_ceiling_tokens: u64,
    pub reserve_tokens: u64,
    pub final_protected_tokens: u64,
}

impl AutoCheckpointPressure {
    pub(super) fn from_budget(budget: &BudgetReport) -> Self {
        Self {
            safe_ceiling_tokens: budget.protected_safe_ceiling_tokens,
            reserve_tokens: budget.protected_reserve_tokens,
            final_protected_tokens: budget.estimated_protected_tokens,
        }
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum AutoCheckpointTrigger {
    SoftPressure,
    HardProtectedOverflow,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum AutoCheckpointDecision {
    Trigger(AutoCheckpointTrigger),
    Rearm,
    Suppress,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn policy(percent: u8) -> AutoCheckpointPolicy {
        AutoCheckpointPolicy {
            enabled: true,
            trigger_reserve_percent: percent,
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
    fn pressure(protected: u64) -> AutoCheckpointPressure {
        AutoCheckpointPressure {
            safe_ceiling_tokens: 1_000,
            reserve_tokens: 101,
            final_protected_tokens: protected,
        }
    }

    #[test]
    fn soft_threshold_uses_ceiling_rounding_and_only_final_protected_pressure() {
        assert_eq!(
            policy(50).decide(Some(pressure(1_050)), false, view()),
            AutoCheckpointDecision::Suppress
        );
        assert_eq!(
            policy(50).decide(Some(pressure(1_051)), false, view()),
            AutoCheckpointDecision::Trigger(AutoCheckpointTrigger::SoftPressure)
        );
        let mut irrelevant = pressure(1_051);
        irrelevant.final_protected_tokens = 1_000;
        assert_eq!(
            policy(50).decide(Some(irrelevant), false, view()),
            AutoCheckpointDecision::Rearm
        );
    }

    #[test]
    fn hysteresis_rearms_only_at_or_below_low_watermark() {
        assert_eq!(
            policy(50).decide(Some(pressure(1_001)), false, view()),
            AutoCheckpointDecision::Suppress
        );
        assert_eq!(
            policy(50).decide(Some(pressure(1_000)), false, view()),
            AutoCheckpointDecision::Rearm
        );
    }

    #[test]
    fn hard_override_needs_no_budget_but_obeys_the_same_scheduler_gates() {
        assert_eq!(
            policy(50).decide(None, true, view()),
            AutoCheckpointDecision::Trigger(AutoCheckpointTrigger::HardProtectedOverflow)
        );
        let mut absent = view();
        absent.boundary_available = false;
        assert_eq!(
            policy(50).decide(None, true, absent),
            AutoCheckpointDecision::Suppress
        );
        let mut blocked = view();
        blocked.boundary_consumed = true;
        assert_eq!(
            policy(50).decide(None, true, blocked),
            AutoCheckpointDecision::Suppress
        );
    }

    #[test]
    fn zero_reserve_and_scheduler_limits_suppress_automatic_requests() {
        let zero = AutoCheckpointPressure {
            reserve_tokens: 0,
            ..pressure(9_000)
        };
        assert_eq!(
            policy(50).decide(Some(zero), false, view()),
            AutoCheckpointDecision::Suppress
        );
        let mut capped = view();
        capped.automatic_commits = 2;
        assert_eq!(
            policy(50).decide(Some(pressure(9_000)), false, capped),
            AutoCheckpointDecision::Suppress
        );
        let mut suppressed = view();
        suppressed.suppressed = true;
        assert_eq!(
            policy(50).decide(Some(pressure(9_000)), false, suppressed),
            AutoCheckpointDecision::Suppress
        );
    }

    #[test]
    fn low_pressure_rearms_disarmed_or_consumed_scheduler_without_reusing_boundary() {
        let mut disarmed = view();
        disarmed.armed = false;
        assert_eq!(
            policy(50).decide(Some(pressure(1_000)), false, disarmed),
            AutoCheckpointDecision::Rearm
        );

        let mut consumed = disarmed;
        consumed.boundary_consumed = true;
        assert_eq!(
            policy(50).decide(Some(pressure(1_000)), false, consumed),
            AutoCheckpointDecision::Rearm
        );
        consumed.armed = true;
        assert_eq!(
            policy(50).decide(Some(pressure(9_000)), false, consumed),
            AutoCheckpointDecision::Suppress
        );
    }

    #[test]
    fn trigger_requires_an_available_unconsumed_unattempted_boundary() {
        for blocked in [
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
        ] {
            assert_eq!(
                policy(50).decide(Some(pressure(9_000)), false, blocked),
                AutoCheckpointDecision::Suppress
            );
        }
    }

    #[test]
    fn rebuilt_successor_rearms_only_at_low_and_needs_a_fresh_boundary_to_trigger_again() {
        let mut after_checkpoint = view();
        after_checkpoint.armed = false;
        after_checkpoint.boundary_available = false;
        after_checkpoint.boundary_consumed = true;

        assert_eq!(
            policy(50).decide(Some(pressure(1_000)), false, after_checkpoint),
            AutoCheckpointDecision::Rearm,
            "a successor at the low watermark immediately rearms"
        );
        assert_eq!(
            policy(50).decide(Some(pressure(1_001)), false, after_checkpoint),
            AutoCheckpointDecision::Suppress,
            "a successor above low remains disarmed"
        );

        let next_boundary = AutoCheckpointSchedulerView {
            armed: true,
            boundary_available: true,
            boundary_consumed: false,
            boundary_attempted: false,
            ..after_checkpoint
        };
        assert_eq!(
            policy(50).decide(Some(pressure(9_000)), false, next_boundary),
            AutoCheckpointDecision::Trigger(AutoCheckpointTrigger::SoftPressure),
            "only a new completed batch can use the rearmed scheduler"
        );
    }

    #[test]
    fn disabled_default_and_missing_pressure_never_create_soft_automatic_work() {
        let disabled = AutoCheckpointPolicy::from_config(LogicalCheckpointConfig::default());
        assert_eq!(
            disabled.decide(Some(pressure(9_000)), false, view()),
            AutoCheckpointDecision::Suppress
        );
        assert_eq!(
            policy(50).decide(None, false, view()),
            AutoCheckpointDecision::Suppress
        );
    }
}
