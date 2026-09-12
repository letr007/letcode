use crate::protocol_frames::{ProtocolFrame, ProtocolFrameItem};
use crate::request_builder::ModelRequestMetadata;
use crate::runtime_context::{FrameVisibility, RuntimeFrame, RuntimeSnapshot};
use crate::user_content::{UserMessageContent, UserMessagePart};

pub(super) fn provider_visible_protocol_frames(snapshot: &RuntimeSnapshot) -> Vec<ProtocolFrame> {
    snapshot
        .frames
        .iter()
        .filter(|frame| frame_is_provider_visible(snapshot, frame))
        .filter_map(|frame| {
            frame.protocol.clone().map(|item| ProtocolFrame {
                runtime_frame_id: Some(frame.id),
                source_provenance: Some(frame.provenance.clone()),
                history_index: 0,
                item,
            })
        })
        .enumerate()
        .map(|(history_index, mut frame)| {
            frame.history_index = history_index;
            frame
        })
        .collect()
}

pub(crate) fn project_model_images(frames: &mut [ProtocolFrame], model: &ModelRequestMetadata) {
    for frame in frames {
        match &mut frame.item {
            ProtocolFrameItem::UserMessage { content }
                if !model.supports_input_images
                    && content
                        .parts()
                        .iter()
                        .any(|part| matches!(part, UserMessagePart::Image { .. })) =>
            {
                let selected_skills = content.selected_skills.clone();
                let parts = content
                    .parts()
                    .into_iter()
                    .map(|part| match part {
                        UserMessagePart::Text { text } => UserMessagePart::Text { text },
                        UserMessagePart::Image { attachment } => UserMessagePart::Text {
                            text: attachment.placeholder_summary(),
                        },
                    })
                    .collect();
                *content =
                    UserMessageContent::from_parts(parts).with_selected_skills(selected_skills);
            }
            ProtocolFrameItem::ToolOutput {
                output_json,
                images,
                ..
            } if !model.supports_tool_result_images => {
                for image in images.iter() {
                    output_json.push('\n');
                    output_json.push_str(&image.placeholder_summary());
                }
                images.clear();
            }
            _ => {}
        }
    }
}

fn frame_is_provider_visible(snapshot: &RuntimeSnapshot, frame: &RuntimeFrame) -> bool {
    frame.visibility == FrameVisibility::Active
        && !snapshot.compaction.compacted_frame_ids.contains(&frame.id)
        && frame.provenance.source_span.is_none_or(|span| {
            !snapshot
                .compaction
                .retired_source_spans
                .iter()
                .any(|retired| retired.overlaps(span))
        })
}

pub(super) fn protected_start_index_for_snapshot(
    snapshot: &RuntimeSnapshot,
    frames: &[ProtocolFrame],
) -> usize {
    frames
        .iter()
        .position(|frame| {
            frame
                .runtime_frame_id
                .is_some_and(|id| snapshot.compaction.protected_frame_ids.contains(&id))
        })
        .unwrap_or(frames.len())
}
