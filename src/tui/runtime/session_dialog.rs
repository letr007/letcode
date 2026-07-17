use super::DialogItem;
use crate::transcript::SessionSummary;

pub(super) fn session_dialog_item(session: &SessionSummary) -> DialogItem {
    let label = session
        .title
        .clone()
        .or_else(|| session.last_user_summary.clone())
        .or_else(|| session.last_assistant_summary.clone())
        .unwrap_or_else(|| "empty session".into());
    let timestamp_ms = session.last_timestamp_ms.or(session.first_timestamp_ms);
    let section = timestamp_ms
        .map(session_section_label)
        .unwrap_or_else(|| "Unknown date".into());
    let right_detail = timestamp_ms
        .map(session_time_label)
        .unwrap_or_else(|| "--:--".into());

    DialogItem::new(
        session.session_id.clone(),
        label,
        Some(session.session_id.clone()),
    )
    .with_section(section)
    .with_right_detail(right_detail)
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
