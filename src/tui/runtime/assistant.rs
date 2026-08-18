//! Assistant-delta streaming ("typewriter") plumbing, isolated from the
//! runtime orchestrator.
//!
//! Owns the grapheme-buffered typewriter that paces assistant deltas onto the
//! screen, plus the helpers that slice assistant deltas out of (and back into)
//! `SessionTransportEvent`. None of this touches `TuiRuntime` state, so it
//! lives apart from the God-file body.

use std::time::{Duration, Instant};

use unicode_segmentation::UnicodeSegmentation;

use crate::session::{SessionEvent, SessionTransportEvent};
use crate::tui::events::AssistantDeltaEvent;

pub(crate) const ASSISTANT_TYPEWRITER_INITIAL_RATE: f64 = 60.0;
const ASSISTANT_TYPEWRITER_MIN_RATE: f64 = 24.0;
pub(crate) const ASSISTANT_TYPEWRITER_MAX_RATE: f64 = 360.0;
const ASSISTANT_TYPEWRITER_RATE_SMOOTHING: f64 = 0.2;
const ASSISTANT_TYPEWRITER_CATCHUP_WINDOW: Duration = Duration::from_millis(132);

pub(crate) fn assistant_delta_parts(
    event: &SessionTransportEvent,
) -> Option<(AssistantDeltaStream, Option<String>, String)> {
    match event {
        SessionTransportEvent::AssistantDelta(delta) => Some((
            AssistantDeltaStream {
                child_session_id: None,
                parent_tool_call_id: None,
                message_id: delta.message_id.clone(),
            },
            None,
            delta.delta.clone(),
        )),
        SessionTransportEvent::ChildSessionEvent {
            child_session_id,
            agent_name,
            parent_tool_call_id,
            event: SessionEvent::AssistantDelta(delta),
        } => Some((
            AssistantDeltaStream {
                child_session_id: Some(child_session_id.clone()),
                parent_tool_call_id: parent_tool_call_id.clone(),
                message_id: delta.message_id.clone(),
            },
            agent_name.clone(),
            delta.delta.clone(),
        )),
        _ => None,
    }
}

pub(crate) fn assistant_delta_event(
    stream: &AssistantDeltaStream,
    agent_name: &Option<String>,
    delta: String,
) -> SessionTransportEvent {
    let delta = match &stream.message_id {
        Some(message_id) => AssistantDeltaEvent::with_message_id(message_id, delta),
        None => AssistantDeltaEvent::new(delta),
    };
    match &stream.child_session_id {
        Some(child_session_id) => SessionTransportEvent::ChildSessionEvent {
            child_session_id: child_session_id.clone(),
            agent_name: agent_name.clone(),
            parent_tool_call_id: stream.parent_tool_call_id.clone(),
            event: SessionEvent::AssistantDelta(delta),
        },
        None => SessionTransportEvent::AssistantDelta(delta),
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct AssistantDeltaStream {
    pub(crate) child_session_id: Option<String>,
    pub(crate) parent_tool_call_id: Option<String>,
    pub(crate) message_id: Option<String>,
}

#[derive(Debug)]
pub(crate) struct AssistantTypewriter {
    pub(crate) stream: AssistantDeltaStream,
    pub(crate) agent_name: Option<String>,
    pub(crate) pending: String,
    display_budget: f64,
    pub(crate) graphemes_per_second: f64,
    last_delta_at: Option<Instant>,
    pub(crate) last_frame_at: Instant,
}

impl AssistantTypewriter {
    pub(crate) fn new(
        stream: AssistantDeltaStream,
        agent_name: Option<String>,
        now: Instant,
    ) -> Self {
        Self {
            stream,
            agent_name,
            pending: String::new(),
            display_budget: 0.0,
            graphemes_per_second: ASSISTANT_TYPEWRITER_INITIAL_RATE,
            last_delta_at: None,
            last_frame_at: now,
        }
    }

    pub(crate) fn push(&mut self, delta: &str, now: Instant) {
        if delta.is_empty() {
            return;
        }
        if self.pending.is_empty() {
            self.display_budget = 0.0;
            self.last_frame_at = now;
        }
        self.pending.push_str(delta);
        let grapheme_count = UnicodeSegmentation::graphemes(delta, true).count();
        if grapheme_count == 0 {
            return;
        }
        if let Some(last_delta_at) = self.last_delta_at {
            let elapsed = now.saturating_duration_since(last_delta_at);
            if !elapsed.is_zero() {
                let sample_rate = grapheme_count as f64 / elapsed.as_secs_f64();
                let sample_rate =
                    sample_rate.clamp(ASSISTANT_TYPEWRITER_MIN_RATE, ASSISTANT_TYPEWRITER_MAX_RATE);
                self.graphemes_per_second = self.graphemes_per_second
                    * (1.0 - ASSISTANT_TYPEWRITER_RATE_SMOOTHING)
                    + sample_rate * ASSISTANT_TYPEWRITER_RATE_SMOOTHING;
            }
        }
        self.last_delta_at = Some(now);
    }

    pub(crate) fn take_frame(&mut self, now: Instant, catch_up: bool) -> String {
        let elapsed = now.saturating_duration_since(self.last_frame_at);
        self.last_frame_at = now;

        let pending_graphemes = self.pending_graphemes();
        if pending_graphemes == 0 {
            self.display_budget = 0.0;
            return String::new();
        }
        self.display_budget += self.graphemes_per_second * elapsed.as_secs_f64();
        if catch_up {
            let catchup_rate =
                pending_graphemes as f64 / ASSISTANT_TYPEWRITER_CATCHUP_WINDOW.as_secs_f64();
            let frame_rate = self
                .graphemes_per_second
                .max(catchup_rate)
                .min(ASSISTANT_TYPEWRITER_MAX_RATE);
            self.display_budget += (frame_rate - self.graphemes_per_second) * elapsed.as_secs_f64();
        }

        let count = self.display_budget.floor() as usize;
        if count == 0 {
            return String::new();
        }
        let count = count.min(pending_graphemes);
        let released = take_grapheme_prefix(&mut self.pending, count);
        self.display_budget -=
            UnicodeSegmentation::graphemes(released.as_str(), true).count() as f64;
        released
    }

    pub(crate) fn pending_graphemes(&self) -> usize {
        UnicodeSegmentation::graphemes(self.pending.as_str(), true).count()
    }
}

fn take_grapheme_prefix(text: &mut String, count: usize) -> String {
    if count == 0 || text.is_empty() {
        return String::new();
    }
    let mut split_at = UnicodeSegmentation::grapheme_indices(text.as_str(), true)
        .nth(count)
        .map(|(index, _)| index)
        .unwrap_or(text.len());
    while split_at < text.len() {
        let mut remainder = text[split_at..].chars();
        let Some(character) = remainder.next() else {
            break;
        };
        let continuation = is_grapheme_continuation(character)
            || character == '\u{200d}' && remainder.next().is_some();
        if !continuation {
            break;
        }
        split_at += character.len_utf8();
        if character == '\u{200d}'
            && let Some(joined) = text[split_at..].chars().next()
        {
            split_at += joined.len_utf8();
        }
    }
    let tail = text.split_off(split_at);
    std::mem::replace(text, tail)
}

fn is_grapheme_continuation(character: char) -> bool {
    matches!(character, '\u{200d}' | '\u{fe0e}' | '\u{fe0f}')
        || ('\u{0300}'..='\u{036f}').contains(&character)
        || ('\u{1ab0}'..='\u{1aff}').contains(&character)
        || ('\u{1dc0}'..='\u{1dff}').contains(&character)
        || ('\u{20d0}'..='\u{20ff}').contains(&character)
        || ('\u{fe20}'..='\u{fe2f}').contains(&character)
        || ('\u{1f3fb}'..='\u{1f3ff}').contains(&character)
}
