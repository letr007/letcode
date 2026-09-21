//! 启动期探测终端默认背景色（OSC 11）。
//!
//! 只在 raw mode 已开、输入读取尚未开始的窗口里跑一次：crossterm 0.28 没有把字节还回事件流
//! 的接口，探针窗口里读到的按键无法归还，所以写查询前先确认没有待读输入，解析到回包就停。
//! Windows 走原生控制台事件，没有可复用的字节窗口，因此不探测。

use std::time::Duration;

pub type Rgb = (u8, u8, u8);

const QUERY: &[u8] = b"\x1b]11;?\x1b\\";
const REPLY_PREFIX: &[u8] = b"\x1b]11;";
const TIMEOUT: Duration = Duration::from_millis(150);
const MAX_BYTES: usize = 256;

#[cfg(unix)]
pub fn query_background() -> Option<Rgb> {
    use std::io::Write;
    use std::os::fd::AsRawFd;

    if matches!(crossterm::event::poll(Duration::ZERO), Ok(true)) {
        return None;
    }

    let fd = std::io::stdin().as_raw_fd();
    if unsafe { libc::isatty(fd) } != 1 {
        return None;
    }

    let mut stdout = std::io::stdout();
    stdout.write_all(QUERY).ok()?;
    stdout.flush().ok()?;

    let deadline = std::time::Instant::now() + TIMEOUT;
    let mut buffer = Vec::with_capacity(MAX_BYTES);
    while let Some(remaining) = deadline.checked_duration_since(std::time::Instant::now()) {
        if !readable(fd, remaining) {
            break;
        }
        let mut chunk = [0_u8; 64];
        let count = unsafe { libc::read(fd, chunk.as_mut_ptr().cast(), chunk.len()) };
        if count <= 0 {
            break;
        }
        buffer.extend_from_slice(&chunk[..count as usize]);
        if let Some(rgb) = parse_reply(&buffer) {
            return Some(rgb);
        }
        if buffer.len() >= MAX_BYTES {
            break;
        }
    }
    None
}

#[cfg(not(unix))]
pub fn query_background() -> Option<Rgb> {
    None
}

#[cfg(unix)]
fn readable(fd: libc::c_int, timeout: Duration) -> bool {
    let mut descriptor = libc::pollfd {
        fd,
        events: libc::POLLIN,
        revents: 0,
    };
    let millis = timeout.as_millis().min(i32::MAX as u128) as libc::c_int;
    let result = unsafe { libc::poll(&mut descriptor, 1, millis) };
    result > 0 && descriptor.revents & libc::POLLIN != 0
}

#[cfg(any(unix, test))]
fn parse_reply(bytes: &[u8]) -> Option<Rgb> {
    let start = bytes
        .windows(REPLY_PREFIX.len())
        .position(|window| window == REPLY_PREFIX)?
        + REPLY_PREFIX.len();
    let payload = &bytes[start..];
    let end = osc_payload_end(payload)?;
    parse_color_spec(std::str::from_utf8(&payload[..end]).ok()?)
}

#[cfg(any(unix, test))]
fn osc_payload_end(bytes: &[u8]) -> Option<usize> {
    let mut index = 0;
    while index < bytes.len() {
        match bytes[index] {
            0x07 => return Some(index),
            0x1b if bytes.get(index + 1) == Some(&b'\\') => return Some(index),
            _ => index += 1,
        }
    }
    None
}

#[cfg(any(unix, test))]
fn parse_color_spec(spec: &str) -> Option<Rgb> {
    let (kind, values) = spec.trim().split_once(':')?;
    let kind = kind.to_ascii_lowercase();
    if kind != "rgb" && kind != "rgba" {
        return None;
    }

    let mut parts = values.split('/');
    let red = parse_component(parts.next()?)?;
    let green = parse_component(parts.next()?)?;
    let blue = parse_component(parts.next()?)?;
    if kind == "rgba" {
        parse_component(parts.next()?)?;
    }
    parts.next().is_none().then_some((red, green, blue))
}

/// 分量是 1–4 位十六进制，按自身位宽归一到 8 位。
#[cfg(any(unix, test))]
fn parse_component(component: &str) -> Option<u8> {
    if !(1..=4).contains(&component.len()) {
        return None;
    }
    let value = u32::from(u16::from_str_radix(component, 16).ok()?);
    let maximum = (1_u32 << (component.len() * 4)) - 1;
    Some((value * u32::from(u8::MAX) / maximum) as u8)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_bel_and_st_replies() {
        assert_eq!(
            parse_reply(b"\x1b]11;rgb:1a1b/1a1b/2626\x1b\\"),
            Some((26, 26, 38))
        );
        assert_eq!(parse_reply(b"\x1b]11;rgb:11/11/11\x07"), Some((17, 17, 17)));
    }

    #[test]
    fn parses_component_widths_and_alpha() {
        assert_eq!(parse_color_spec("rgb:f/e/d"), Some((255, 238, 221)));
        assert_eq!(parse_color_spec("rgb:00/80/ff"), Some((0, 128, 255)));
        assert_eq!(
            parse_color_spec("rgba:1111/1111/1111/ffff"),
            Some((17, 17, 17))
        );
    }

    #[test]
    fn rejects_incomplete_or_malformed_replies() {
        assert_eq!(parse_reply(b"\x1b]10;rgb:eeee/eeee/eeee\x1b\\"), None);
        assert_eq!(parse_reply(b"\x1b]11;rgb:1111/1111/1111"), None);
        assert_eq!(parse_color_spec("rgb:fffff/0/0"), None);
        assert_eq!(parse_color_spec("rgb:11/11"), None);
        assert_eq!(parse_color_spec("rgb:11/11/11/11"), None);
        assert_eq!(parse_color_spec("rgb:gg/11/11"), None);
        assert_eq!(parse_color_spec("nope:11/11/11"), None);
    }

    #[test]
    fn finds_the_reply_after_unrelated_bytes() {
        assert_eq!(
            parse_reply(b"xy\x1b]11;rgb:0000/0000/0000\x07z"),
            Some((0, 0, 0))
        );
    }
}
