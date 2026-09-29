use std::collections::HashMap;
use std::path::{Component, Path};

use super::DialogItem;
use crate::transcript::SessionSummary;
use crate::tui::state::SessionPickerScope;

pub(super) fn session_dialog_items(
    sessions: &[SessionSummary],
    current_workspace: Option<&str>,
    scope: SessionPickerScope,
    unassigned_label: &str,
) -> Vec<DialogItem> {
    match scope {
        SessionPickerScope::Workspace => {
            let Some(current) = current_workspace else {
                return Vec::new();
            };
            sessions
                .iter()
                .filter(|session| session.workspace.as_deref() == Some(current))
                .map(date_section_item)
                .collect()
        }
        SessionPickerScope::All => {
            let groups = project_groups(sessions, current_workspace);
            let labels = project_labels(&groups);
            let mut items = Vec::with_capacity(sessions.len());
            for (root, members) in groups {
                let label = match &root {
                    Some(root) => labels.get(root).cloned().unwrap_or_else(|| root.clone()),
                    None => unassigned_label.to_string(),
                };
                items.extend(members.into_iter().map(|session| {
                    session_item(session, label.clone(), session_stamp_label(session))
                }));
            }
            items
        }
    }
}

fn project_groups<'a>(
    sessions: &'a [SessionSummary],
    current_workspace: Option<&str>,
) -> Vec<(Option<String>, Vec<&'a SessionSummary>)> {
    let mut groups: Vec<(Option<String>, Vec<&'a SessionSummary>)> = Vec::new();
    let mut index: HashMap<Option<&str>, usize> = HashMap::new();
    for session in sessions {
        let root = session.workspace.as_deref();
        let position = match index.get(&root) {
            Some(position) => *position,
            None => {
                groups.push((root.map(str::to_string), Vec::new()));
                index.insert(root, groups.len() - 1);
                groups.len() - 1
            }
        };
        groups[position].1.push(session);
    }

    groups.sort_by_key(|(root, members)| {
        let current = root.as_deref() == current_workspace;
        let recent = members
            .iter()
            .filter_map(|session| session.last_timestamp_ms)
            .max()
            .unwrap_or(0);
        (root.is_none(), !current, std::cmp::Reverse(recent))
    });
    groups
}

fn project_labels(groups: &[(Option<String>, Vec<&SessionSummary>)]) -> HashMap<String, String> {
    let roots = groups
        .iter()
        .filter_map(|(root, _)| root.as_deref())
        .collect::<Vec<_>>();
    let mut short_counts: HashMap<String, usize> = HashMap::new();
    for root in &roots {
        *short_counts
            .entry(trailing_components(root, 1))
            .or_default() += 1;
    }

    roots
        .into_iter()
        .map(|root| {
            let short = trailing_components(root, 1);
            let label = if short_counts[&short] == 1 {
                short
            } else {
                trailing_components(root, 2)
            };
            (root.to_string(), label)
        })
        .collect()
}

fn trailing_components(root: &str, depth: usize) -> String {
    let mut parts = Path::new(root)
        .components()
        .filter_map(|component| match component {
            Component::Normal(part) => Some(part.to_string_lossy().into_owned()),
            _ => None,
        })
        .collect::<Vec<_>>();
    if parts.len() > depth {
        parts.drain(..parts.len() - depth);
    }
    parts.join("/")
}

fn session_label(session: &SessionSummary) -> String {
    session
        .title
        .clone()
        .or_else(|| session.last_user_summary.clone())
        .or_else(|| session.last_assistant_summary.clone())
        .unwrap_or_else(|| "empty session".into())
}

fn session_item(session: &SessionSummary, section: String, right_detail: String) -> DialogItem {
    DialogItem::new(
        session.session_id.clone(),
        session_label(session),
        Some(session.session_id.clone()),
    )
    .with_section(section)
    .with_right_detail(right_detail)
}

fn date_section_item(session: &SessionSummary) -> DialogItem {
    let timestamp_ms = session.last_timestamp_ms.or(session.first_timestamp_ms);
    let section = timestamp_ms
        .map(session_section_label)
        .unwrap_or_else(|| "Unknown date".into());
    let right_detail = timestamp_ms
        .map(session_time_label)
        .unwrap_or_else(|| "--:--".into());

    session_item(session, section, right_detail)
}

fn session_stamp_label(session: &SessionSummary) -> String {
    let Some(timestamp_ms) = session.last_timestamp_ms.or(session.first_timestamp_ms) else {
        return "--:--".into();
    };
    let (year, month, day) = utc_date_parts(timestamp_ms);
    let (current_year, current_month, current_day) = utc_date_parts(unix_timestamp_ms_for_tui());
    if (year, month, day) == (current_year, current_month, current_day) {
        return session_time_label(timestamp_ms);
    }
    if year == current_year {
        return format!("{} {day:02}", month_name(month));
    }
    format!("{} {day:02} {year}", month_name(month))
}

fn session_section_label(timestamp_ms: u128) -> String {
    let (year, month, day) = utc_date_parts(timestamp_ms);
    let today = utc_date_parts(unix_timestamp_ms_for_tui());
    if (year, month, day) == today {
        return "Today".into();
    }

    let weekday = weekday_name(year, month, day);
    let month = month_name(month);
    format!("{weekday} {month} {day:02} {year}")
}

fn session_time_label(timestamp_ms: u128) -> String {
    let total_seconds = (timestamp_ms / 1_000) as u64;
    let seconds_in_day = total_seconds % 86_400;
    let hour = seconds_in_day / 3_600;
    let minute = (seconds_in_day % 3_600) / 60;
    let suffix = if hour < 12 { "AM" } else { "PM" };
    let display_hour = match hour % 12 {
        0 => 12,
        hour => hour,
    };
    format!("{display_hour}:{minute:02} {suffix}")
}

fn utc_date_parts(timestamp_ms: u128) -> (i32, u32, u32) {
    let days = (timestamp_ms / 1_000 / 86_400) as i64;
    civil_from_days(days)
}

fn civil_from_days(days_since_unix_epoch: i64) -> (i32, u32, u32) {
    let z = days_since_unix_epoch + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1_460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let day = doy - (153 * mp + 2) / 5 + 1;
    let month = mp + if mp < 10 { 3 } else { -9 };
    let year = y + if month <= 2 { 1 } else { 0 };
    (year as i32, month as u32, day as u32)
}

fn weekday_name(year: i32, month: u32, day: u32) -> &'static str {
    let mut month = month as i32;
    let mut year = year;
    if month < 3 {
        month += 12;
        year -= 1;
    }
    let k = year % 100;
    let j = year / 100;
    let h = (day as i32 + (13 * (month + 1)) / 5 + k + k / 4 + j / 4 + 5 * j) % 7;
    match h {
        0 => "Sat",
        1 => "Sun",
        2 => "Mon",
        3 => "Tue",
        4 => "Wed",
        5 => "Thu",
        _ => "Fri",
    }
}

fn month_name(month: u32) -> &'static str {
    match month {
        1 => "Jan",
        2 => "Feb",
        3 => "Mar",
        4 => "Apr",
        5 => "May",
        6 => "Jun",
        7 => "Jul",
        8 => "Aug",
        9 => "Sep",
        10 => "Oct",
        11 => "Nov",
        _ => "Dec",
    }
}

fn unix_timestamp_ms_for_tui() -> u128 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn summary(id: &str, last_timestamp_ms: u128, workspace: Option<&str>) -> SessionSummary {
        SessionSummary {
            session_id: id.into(),
            record_count: 1,
            first_timestamp_ms: Some(last_timestamp_ms),
            last_timestamp_ms: Some(last_timestamp_ms),
            model: Some("test/model".into()),
            title: Some(format!("title {id}")),
            last_user_summary: None,
            last_assistant_summary: None,
            workspace: workspace.map(str::to_string),
        }
    }

    fn ids(items: &[DialogItem]) -> Vec<&str> {
        items.iter().map(|item| item.id.as_str()).collect()
    }

    fn sections(items: &[DialogItem]) -> Vec<&str> {
        items
            .iter()
            .map(|item| item.section.as_deref().unwrap_or_default())
            .collect()
    }

    #[test]
    fn workspace_scope_lists_only_the_running_workspace() {
        let sessions = vec![
            summary("current", 300, Some("/work/letcode")),
            summary("other", 200, Some("/work/other")),
            summary("unrecorded", 100, None),
        ];

        let items = session_dialog_items(
            &sessions,
            Some("/work/letcode"),
            SessionPickerScope::Workspace,
            "Unknown project",
        );

        assert_eq!(ids(&items), ["current"]);
    }

    #[test]
    fn every_workspace_scope_leads_with_the_running_project() {
        let sessions = vec![
            summary("newest", 400, Some("/work/other")),
            summary("unrecorded", 300, None),
            summary("current", 200, Some("/work/letcode")),
            summary("older", 100, Some("/work/letcode")),
        ];

        let items = session_dialog_items(
            &sessions,
            Some("/work/letcode"),
            SessionPickerScope::All,
            "Unknown project",
        );

        assert_eq!(ids(&items), ["current", "older", "newest", "unrecorded"]);
        assert_eq!(
            sections(&items),
            ["letcode", "letcode", "other", "Unknown project"]
        );
    }

    #[test]
    fn project_headings_widen_when_directory_names_collide() {
        let sessions = vec![
            summary("work", 200, Some("/work/api")),
            summary("oss", 100, Some("/oss/api")),
        ];

        let items = session_dialog_items(
            &sessions,
            Some("/elsewhere"),
            SessionPickerScope::All,
            "Unknown project",
        );

        assert_eq!(sections(&items), ["work/api", "oss/api"]);
    }
}
