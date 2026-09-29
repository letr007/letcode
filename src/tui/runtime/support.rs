//! Runtime support statics, constants, and tiny helpers, independent of the
//! orchestrator. These hold no `TuiRuntime` state, so they live apart from the
//! God-file body.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use crate::transcript::{TranscriptEvent, TranscriptRecord};
use crate::tui::state::McpDiscoveryState;
use crate::user_content::UserImageAttachment;

pub(crate) const TERMINAL_TITLE_TICKS_PER_FRAME: usize = 3;
const TERMINAL_TITLE_APP_NAME: &str = "LetCode";
const TERMINAL_TITLE_SPINNER: [char; 10] = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏'];
const MCP_DISCOVERY_LOADING_DESCRIPTION: &str = "Discovering MCP servers";
const MCP_DISCOVERY_UNAVAILABLE_DESCRIPTION: &str = "MCP discovery unavailable";
static NEXT_SUBMISSION_ID: AtomicU64 = AtomicU64::new(1);
static NEXT_ATTACHMENT_ID: AtomicU64 = AtomicU64::new(1);

pub(crate) fn mcp_discovery_description(discovery: McpDiscoveryState) -> Option<String> {
    match discovery {
        McpDiscoveryState::Loading => Some(MCP_DISCOVERY_LOADING_DESCRIPTION.into()),
        McpDiscoveryState::Ready => None,
        McpDiscoveryState::Unavailable => Some(MCP_DISCOVERY_UNAVAILABLE_DESCRIPTION.into()),
    }
}

pub(crate) fn next_submission_id() -> String {
    format!(
        "user-submission-{}",
        NEXT_SUBMISSION_ID.fetch_add(1, Ordering::Relaxed)
    )
}

pub(crate) fn next_attachment_id() -> String {
    format!(
        "user-attachment-{}",
        NEXT_ATTACHMENT_ID.fetch_add(1, Ordering::Relaxed)
    )
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ClipboardPasteContext {
    Composer,
    Dialog,
    Question,
    Permission,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ClipboardPasteChoice {
    Image,
    Text,
    None,
}

pub(crate) fn choose_clipboard_paste(
    context: ClipboardPasteContext,
    has_text: bool,
    has_image: bool,
) -> ClipboardPasteChoice {
    if matches!(context, ClipboardPasteContext::Composer) && has_image {
        ClipboardPasteChoice::Image
    } else if has_text {
        ClipboardPasteChoice::Text
    } else {
        ClipboardPasteChoice::None
    }
}

pub(crate) fn clipboard_image_attachments(paths: &[PathBuf]) -> Vec<UserImageAttachment> {
    paths
        .iter()
        .filter_map(|path| clipboard_image_attachment(path))
        .collect()
}

fn clipboard_image_attachment(path: &Path) -> Option<UserImageAttachment> {
    let mime = crate::tool::supported_image_mime(path)?;
    let metadata = std::fs::metadata(path).ok()?;
    if !metadata.is_file() || metadata.len() > crate::tool::MAX_READ_IMAGE_BYTES {
        return None;
    }
    let bytes = std::fs::read(path).ok()?;
    let label = path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("clipboard")
        .to_string();
    Some(UserImageAttachment::from_bytes(label, mime, &bytes))
}

pub(crate) fn session_title_from_records(records: &[TranscriptRecord]) -> Option<String> {
    records.iter().rev().find_map(|record| match &record.event {
        TranscriptEvent::SessionTitle { title } => Some(title.clone()),
        _ => None,
    })
}

pub(crate) fn format_terminal_title(
    session_title: Option<&str>,
    spinner_frame: Option<usize>,
) -> String {
    let title = match session_title.filter(|title| !title.trim().is_empty()) {
        Some(title) => format!("{TERMINAL_TITLE_APP_NAME} | {title}"),
        None => TERMINAL_TITLE_APP_NAME.to_string(),
    };
    match spinner_frame {
        Some(frame) => format!(
            "{} {title}",
            TERMINAL_TITLE_SPINNER[frame % TERMINAL_TITLE_SPINNER.len()]
        ),
        None => title,
    }
}
