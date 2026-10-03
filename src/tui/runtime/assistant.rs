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

/// The stream an event closes, if it closes one.
pub(crate) fn assistant_stream_end(event: &SessionTransportEvent) -> Option<AssistantDeltaStream> {
    match event {
        SessionTransportEvent::AssistantDone { message_id } => Some(AssistantDeltaStream {
            child_session_id: None,
            parent_tool_call_id: None,
            message_id: message_id.clone(),
        }),
        SessionTransportEvent::ChildSessionEvent {
            child_session_id,
            parent_tool_call_id,
            event: SessionEvent::AssistantDone { message_id, .. },
            ..
        } => Some(AssistantDeltaStream {
            child_session_id: Some(child_session_id.clone()),
            parent_tool_call_id: parent_tool_call_id.clone(),
            message_id: message_id.clone(),
        }),
        _ => None,
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
    pending_start: usize,
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
            pending_start: 0,
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
        if self.pending_text().is_empty() {
            self.pending.clear();
            self.pending_start = 0;
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

    pub(crate) fn take_frame(&mut self, now: Instant) -> String {
        let elapsed = now.saturating_duration_since(self.last_frame_at);
        self.last_frame_at = now;

        let pending_graphemes = self.pending_graphemes();
        if pending_graphemes == 0 {
            self.display_budget = 0.0;
            return String::new();
        }
        self.display_budget += self.graphemes_per_second * elapsed.as_secs_f64();

        // The animation smooths output; it may not let display fall behind.
        let keep = self.graphemes_per_second * ASSISTANT_TYPEWRITER_CATCHUP_WINDOW.as_secs_f64();
        let release = self
            .display_budget
            .max(pending_graphemes as f64 - keep)
            .max(0.0);
        let count = (release.floor() as usize).min(pending_graphemes);
        if count == 0 {
            return String::new();
        }
        let count = count.min(pending_graphemes);
        let split_at = grapheme_prefix_len(self.pending_text(), count);
        let start = self.pending_start;
        let released = self.pending[start..start + split_at].to_owned();
        self.pending_start += split_at;
        self.compact_pending();
        self.display_budget = (self.display_budget
            - UnicodeSegmentation::graphemes(released.as_str(), true).count() as f64)
            .max(0.0);
        released
    }

    pub(crate) fn pending_text(&self) -> &str {
        &self.pending[self.pending_start..]
    }

    pub(crate) fn take_pending(&mut self) -> String {
        let pending_start = self.pending_start;
        self.pending_start = 0;
        let pending = std::mem::take(&mut self.pending);
        pending[pending_start..].to_owned()
    }

    pub(crate) fn pending_graphemes(&self) -> usize {
        UnicodeSegmentation::graphemes(self.pending_text(), true).count()
    }

    fn compact_pending(&mut self) {
        if self.pending_start == self.pending.len() {
            self.pending.clear();
            self.pending_start = 0;
        } else if self.pending_start >= 4096 && self.pending_start >= self.pending.len() / 2 {
            self.pending.drain(..self.pending_start);
            self.pending_start = 0;
        }
    }
}

fn grapheme_prefix_len(text: &str, count: usize) -> usize {
    if count == 0 || text.is_empty() {
        return 0;
    }
    let mut split_at = UnicodeSegmentation::grapheme_indices(text, true)
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
    split_at
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
