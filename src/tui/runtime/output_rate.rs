use super::*;

impl TuiRuntime {
    pub(super) fn observe_output_rate_transport_event(
        &mut self,
        event: &SessionTransportEvent,
        now: Instant,
    ) {
        match event {
            SessionTransportEvent::UserMessage(_) => {
                self.output_rate_samples
                    .retain(|sample| sample.child_session_id.is_some());
            }
            SessionTransportEvent::ChildSessionEvent {
                child_session_id,
                event: SessionEvent::UserMessage(_),
                ..
            } => {
                self.output_rate_samples
                    .retain(|sample| sample.child_session_id.as_deref() != Some(child_session_id));
            }
            SessionTransportEvent::AssistantDelta(delta) => {
                self.update_output_rate_sample(None, &delta.delta, now);
            }
            SessionTransportEvent::ReasoningDelta(delta) => {
                self.update_output_rate_sample(None, &delta.delta, now);
            }
            SessionTransportEvent::ToolPending(tool) => {
                self.update_output_rate_sample(None, &tool.name, now);
            }
            SessionTransportEvent::ToolStarted(tool) => {
                if let Some(arguments) = tool.arguments.as_deref() {
                    self.update_existing_output_rate_sample(None, arguments, now);
                }
            }
            SessionTransportEvent::ChildSessionEvent {
                child_session_id,
                event: SessionEvent::AssistantDelta(delta),
                ..
            } => {
                self.update_output_rate_sample(Some(child_session_id), &delta.delta, now);
            }
            SessionTransportEvent::ChildSessionEvent {
                child_session_id,
                event: SessionEvent::ReasoningDelta(delta),
                ..
            } => {
                self.update_output_rate_sample(Some(child_session_id), &delta.delta, now);
            }
            SessionTransportEvent::ChildSessionEvent {
                child_session_id,
                event: SessionEvent::ToolPending(tool),
                ..
            } => {
                self.update_output_rate_sample(Some(child_session_id), &tool.name, now);
            }
            SessionTransportEvent::ChildSessionEvent {
                child_session_id,
                event: SessionEvent::ToolStarted(tool),
                ..
            } => {
                if let Some(arguments) = tool.arguments.as_deref() {
                    self.update_existing_output_rate_sample(Some(child_session_id), arguments, now);
                }
            }
            _ => self.finish_output_rate_for_transport_event(event, now),
        }
    }

    pub(super) fn sample_output_rate_graph(&mut self, now: Instant) {
        const SAMPLE_INTERVAL: Duration = Duration::from_millis(500);
        if self
            .last_output_rate_graph_sample_at
            .is_some_and(|last| now.saturating_duration_since(last) < SAMPLE_INTERVAL)
        {
            return;
        }
        self.last_output_rate_graph_sample_at = Some(now);
        let rate = if self.session_turn_active {
            self.state.active_output_token_rate()
        } else {
            None
        };
        self.state.push_output_rate_graph_sample(rate);
    }

    pub(super) fn set_output_token_rate_for_session(
        &mut self,
        child_session_id: Option<&str>,
        rate: Option<u64>,
    ) {
        if let Some(child_session_id) = child_session_id {
            self.state
                .set_child_output_token_rate(child_session_id, rate);
        } else {
            self.state.set_output_token_rate(rate);
        }
    }

    pub(super) fn update_output_rate_sample(
        &mut self,
        child_session_id: Option<&str>,
        delta: &str,
        now: Instant,
    ) {
        let sample_index = self
            .output_rate_samples
            .iter()
            .position(|sample| sample.child_session_id.as_deref() == child_session_id)
            .unwrap_or_else(|| {
                let displayed_rate = if let Some(child_session_id) = child_session_id {
                    self.state
                        .child_output_token_rate(child_session_id)
                        .map(|rate| rate as f64)
                } else {
                    self.state.output_token_rate.map(|rate| rate as f64)
                };
                self.output_rate_samples.push(OutputRateSample {
                    child_session_id: child_session_id.map(str::to_owned),
                    started_at: now,
                    streamed_bytes: 0,
                    displayed_rate,
                    last_display_at: None,
                });
                self.output_rate_samples.len() - 1
            });
        self.update_output_rate_sample_at(sample_index, delta, now);
    }

    pub(super) fn update_existing_output_rate_sample(
        &mut self,
        child_session_id: Option<&str>,
        delta: &str,
        now: Instant,
    ) {
        if let Some(sample_index) = self
            .output_rate_samples
            .iter()
            .position(|sample| sample.child_session_id.as_deref() == child_session_id)
        {
            self.update_output_rate_sample_at(sample_index, delta, now);
        }
    }

    pub(super) fn update_output_rate_sample_at(
        &mut self,
        sample_index: usize,
        delta: &str,
        now: Instant,
    ) {
        const MIN_SAMPLE_DURATION: Duration = Duration::from_millis(500);
        const MIN_DISPLAY_INTERVAL: Duration = Duration::from_millis(250);
        const RATE_SMOOTHING: f64 = 0.25;
        const MAX_RATE_INCREASE_FACTOR: f64 = 2.0;

        let sample = &mut self.output_rate_samples[sample_index];
        sample.streamed_bytes = sample.streamed_bytes.saturating_add(delta.len() as u64);
        let elapsed = now.saturating_duration_since(sample.started_at);
        if elapsed < MIN_SAMPLE_DURATION
            || sample.last_display_at.is_some_and(|last_display_at| {
                now.saturating_duration_since(last_display_at) < MIN_DISPLAY_INTERVAL
            })
        {
            return;
        }
        let estimated_tokens = sample.streamed_bytes.div_ceil(3);
        let instantaneous_rate = estimated_tokens as f64 / elapsed.as_secs_f64();
        let displayed_rate = match sample.displayed_rate {
            Some(previous) => {
                let smoothed =
                    previous * (1.0 - RATE_SMOOTHING) + instantaneous_rate * RATE_SMOOTHING;
                smoothed.min(previous * MAX_RATE_INCREASE_FACTOR)
            }
            None => instantaneous_rate,
        };
        sample.displayed_rate = Some(displayed_rate);
        sample.last_display_at = Some(now);
        let child_session_id = sample.child_session_id.clone();
        self.set_output_token_rate_for_session(
            child_session_id.as_deref(),
            Some(displayed_rate.round() as u64),
        );
    }

    pub(super) fn finish_output_rate_for_transport_event(
        &mut self,
        event: &SessionTransportEvent,
        now: Instant,
    ) {
        match event {
            SessionTransportEvent::TokenUsage(usage) if usage.output_tokens > 0 => {
                self.finish_output_rate_sample(None, usage.output_tokens, now);
            }
            SessionTransportEvent::ChildSessionEvent {
                child_session_id,
                event: SessionEvent::TokenUsage(usage),
                ..
            } if usage.output_tokens > 0 => {
                self.finish_output_rate_sample(Some(child_session_id), usage.output_tokens, now);
            }
            _ => {}
        }
    }

    pub(super) fn finish_output_rate_sample(
        &mut self,
        child_session_id: Option<&str>,
        output_tokens: u64,
        now: Instant,
    ) {
        let Some(sample_index) = self
            .output_rate_samples
            .iter()
            .position(|sample| sample.child_session_id.as_deref() == child_session_id)
        else {
            return;
        };
        let sample = self.output_rate_samples.remove(sample_index);
        let elapsed = now.saturating_duration_since(sample.started_at);
        if elapsed.is_zero() {
            return;
        }
        let rate = (output_tokens as f64 / elapsed.as_secs_f64()).round() as u64;
        self.set_output_token_rate_for_session(sample.child_session_id.as_deref(), Some(rate));
    }
}
