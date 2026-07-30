use std::collections::HashSet;
use std::future::Future;
use std::net::{IpAddr, Ipv4Addr, Ipv6Addr, SocketAddr};
use std::time::Duration;

use anyhow::{Context, Result, bail};
use async_trait::async_trait;
use reqwest::header::{CONTENT_TYPE, LOCATION};
use serde_json::{Value, json};
use tokio::net::lookup_host;

use super::{ToolHandler, ToolParallelism, ToolRegistry};
use crate::tool_names;

const MAX_URL_CHARS: usize = 8_192;
const MAX_RESPONSE_BYTES: usize = 1024 * 1024;
const MAX_REDIRECTS: usize = 5;
const CONNECT_TIMEOUT: Duration = Duration::from_secs(5);
const REQUEST_TIMEOUT: Duration = Duration::from_secs(20);
const TOTAL_TIMEOUT: Duration = Duration::from_secs(30);

pub(super) fn register(registry: &mut ToolRegistry) {
    registry.register(WebFetchTool);
}

struct WebFetchTool;

#[async_trait]
impl ToolHandler for WebFetchTool {
    fn name(&self) -> &'static str {
        tool_names::TOOL_WEB_FETCH
    }

    fn description(&self) -> &'static str {
        "Fetch text content from a public HTTP or HTTPS URL. Redirects, response size, timeouts, and private-network access are restricted."
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "minLength": 1,
                    "maxLength": MAX_URL_CHARS,
                    "description": "Public HTTP or HTTPS URL to fetch"
                }
            },
            "required": ["url"],
            "additionalProperties": false
        })
    }

    fn parallelism(&self) -> ToolParallelism {
        ToolParallelism::Parallel
    }

    async fn execute(&self, args: Value) -> Result<Value> {
        let raw_url = args
            .get("url")
            .and_then(Value::as_str)
            .context("web__fetch requires string field 'url'")?;
        if raw_url.chars().count() > MAX_URL_CHARS {
            bail!("web__fetch URL exceeds {MAX_URL_CHARS} characters");
        }

        fetch_public_url(raw_url).await
    }
}

async fn fetch_public_url(raw_url: &str) -> Result<Value> {
    fetch_with_timeout(fetch_public_url_inner(raw_url), TOTAL_TIMEOUT).await
}

async fn fetch_with_timeout<F, T>(future: F, timeout: Duration) -> Result<T>
where
    F: Future<Output = Result<T>>,
{
    tokio::time::timeout(timeout, future)
        .await
        .with_context(|| format!("web__fetch exceeded the total {timeout:?} timeout"))?
}

async fn fetch_public_url_inner(raw_url: &str) -> Result<Value> {
    let mut current = parse_public_url(raw_url)?;
    let requested_url = current.to_string();
    let mut redirects = Vec::new();
    let mut visited = HashSet::from([requested_url.clone()]);

    loop {
        let client = client_for_url(&current).await?;
        let mut response = client
            .get(current.clone())
            .send()
            .await
            .with_context(|| format!("failed to fetch {current}"))?;
        let status = response.status();

        if status.is_redirection() {
            if redirects.len() >= MAX_REDIRECTS {
                bail!("web__fetch exceeded {MAX_REDIRECTS} redirects");
            }
            let location = response
                .headers()
                .get(LOCATION)
                .context("web__fetch redirect response is missing Location header")?
                .to_str()
                .context("web__fetch redirect Location header is not valid text")?;
            let next = current
                .join(location)
                .context("web__fetch redirect Location is not a valid URL")?;
            let next = validate_public_url(next)?;
            let next_url = next.to_string();
            if !visited.insert(next_url.clone()) {
                bail!("web__fetch detected a redirect loop at {next_url}");
            }
            redirects.push(json!({
                "status": status.as_u16(),
                "from": current.to_string(),
                "to": next_url,
            }));
            current = next;
            continue;
        }

        if !status.is_success() {
            bail!("web__fetch received HTTP {status} for {current}");
        }

        let content_type = response
            .headers()
            .get(CONTENT_TYPE)
            .map(|value| value.to_str())
            .transpose()
            .context("web__fetch Content-Type header is not valid text")?
            .map(str::to_string);
        if !supported_content_type(content_type.as_deref()) {
            bail!(
                "web__fetch does not support Content-Type {}",
                content_type.as_deref().unwrap_or("<missing>")
            );
        }

        let content_length_exceeds_limit = response
            .content_length()
            .is_some_and(|length| length > MAX_RESPONSE_BYTES as u64);
        let (body, stream_truncated) = read_limited_body(&mut response).await?;
        let content_bytes = body.len();
        let (content, output_truncated) = content_for_output(&body);
        let truncated = content_length_exceeds_limit || stream_truncated || output_truncated;

        return Ok(json!({
            "requested_url": requested_url,
            "final_url": current.to_string(),
            "status": status.as_u16(),
            "content_type": content_type,
            "content": content,
            "content_bytes": content_bytes,
            "truncated": truncated,
            "redirects": redirects,
        }));
    }
}

fn parse_public_url(raw_url: &str) -> Result<reqwest::Url> {
    let trimmed = raw_url.trim();
    if trimmed.is_empty() {
        bail!("web__fetch URL must not be empty");
    }
    let url = reqwest::Url::parse(trimmed).context("web__fetch requires a valid URL")?;
    validate_public_url(url)
}

fn validate_public_url(mut url: reqwest::Url) -> Result<reqwest::Url> {
    if !matches!(url.scheme(), "http" | "https") {
        bail!("web__fetch only supports http and https URLs");
    }
    if !url.username().is_empty() || url.password().is_some() {
        bail!("web__fetch does not allow credentials in URLs");
    }
    let host = url.host_str().context("web__fetch URL requires a host")?;
    if host.trim().is_empty() {
        bail!("web__fetch URL requires a host");
    }
    if let Ok(ip) = host.trim_matches(['[', ']']).parse::<IpAddr>() {
        ensure_public_ip(ip)?;
    }
    url.set_fragment(None);
    Ok(url)
}

async fn client_for_url(url: &reqwest::Url) -> Result<reqwest::Client> {
    let host = url.host_str().context("web__fetch URL requires a host")?;
    let port = url
        .port_or_known_default()
        .context("web__fetch URL has no usable port")?;
    let parsed_ip = host.trim_matches(['[', ']']).parse::<IpAddr>().ok();

    let mut builder = reqwest::Client::builder()
        .connect_timeout(CONNECT_TIMEOUT)
        .timeout(REQUEST_TIMEOUT)
        .redirect(reqwest::redirect::Policy::none())
        .no_proxy()
        .user_agent(concat!("letcode/", env!("CARGO_PKG_VERSION")));

    if parsed_ip.is_none() {
        let addresses = resolve_public_addresses(host, port).await?;
        builder = builder.resolve_to_addrs(host, &addresses);
    }

    builder
        .build()
        .context("failed to build web__fetch HTTP client")
}

async fn resolve_public_addresses(host: &str, port: u16) -> Result<Vec<SocketAddr>> {
    let mut addresses = lookup_host((host, port))
        .await
        .with_context(|| format!("failed to resolve web__fetch host {host}"))?
        .collect::<Vec<_>>();
    addresses.sort_unstable();
    addresses.dedup();
    if addresses.is_empty() {
        bail!("web__fetch host {host} resolved to no addresses");
    }
    for address in &addresses {
        ensure_public_ip(address.ip())
            .with_context(|| format!("web__fetch host {host} resolved to blocked address"))?;
    }
    Ok(addresses)
}

fn ensure_public_ip(ip: IpAddr) -> Result<()> {
    let allowed = match ip {
        IpAddr::V4(ip) => is_public_ipv4(ip),
        IpAddr::V6(ip) => is_public_ipv6(ip),
    };
    if allowed {
        Ok(())
    } else {
        bail!("web__fetch blocks non-public address {ip}")
    }
}

fn is_public_ipv4(ip: Ipv4Addr) -> bool {
    let value = u32::from(ip);
    !in_ipv4_cidr(value, [0, 0, 0, 0], 8)
        && !in_ipv4_cidr(value, [10, 0, 0, 0], 8)
        && !in_ipv4_cidr(value, [100, 64, 0, 0], 10)
        && !in_ipv4_cidr(value, [127, 0, 0, 0], 8)
        && !in_ipv4_cidr(value, [169, 254, 0, 0], 16)
        && !in_ipv4_cidr(value, [172, 16, 0, 0], 12)
        && !in_ipv4_cidr(value, [192, 0, 0, 0], 24)
        && !in_ipv4_cidr(value, [192, 0, 2, 0], 24)
        && !in_ipv4_cidr(value, [192, 168, 0, 0], 16)
        && !in_ipv4_cidr(value, [198, 18, 0, 0], 15)
        && !in_ipv4_cidr(value, [198, 51, 100, 0], 24)
        && !in_ipv4_cidr(value, [203, 0, 113, 0], 24)
        && !in_ipv4_cidr(value, [224, 0, 0, 0], 4)
        && !in_ipv4_cidr(value, [240, 0, 0, 0], 4)
}

fn in_ipv4_cidr(value: u32, network: [u8; 4], prefix: u32) -> bool {
    let mask = if prefix == 0 {
        0
    } else {
        u32::MAX << (32 - prefix)
    };
    value & mask == u32::from(Ipv4Addr::from(network)) & mask
}

fn is_public_ipv6(ip: Ipv6Addr) -> bool {
    if let Some(ipv4) = ip.to_ipv4_mapped() {
        return is_public_ipv4(ipv4);
    }
    let value = u128::from(ip);
    in_ipv6_cidr(value, Ipv6Addr::new(0x2000, 0, 0, 0, 0, 0, 0, 0), 3)
        && !in_ipv6_cidr(value, Ipv6Addr::new(0x2001, 0, 0, 0, 0, 0, 0, 0), 32)
        && !in_ipv6_cidr(value, Ipv6Addr::new(0x2001, 0x0002, 0, 0, 0, 0, 0, 0), 48)
        && !in_ipv6_cidr(value, Ipv6Addr::new(0x2001, 0x0db8, 0, 0, 0, 0, 0, 0), 32)
        && !in_ipv6_cidr(value, Ipv6Addr::new(0x2001, 0x0010, 0, 0, 0, 0, 0, 0), 28)
        && !in_ipv6_cidr(value, Ipv6Addr::new(0x2001, 0x0020, 0, 0, 0, 0, 0, 0), 28)
        && !in_ipv6_cidr(value, Ipv6Addr::new(0x2002, 0, 0, 0, 0, 0, 0, 0), 16)
}

fn in_ipv6_cidr(value: u128, network: Ipv6Addr, prefix: u32) -> bool {
    let mask = if prefix == 0 {
        0
    } else {
        u128::MAX << (128 - prefix)
    };
    value & mask == u128::from(network) & mask
}

fn supported_content_type(content_type: Option<&str>) -> bool {
    let Some(content_type) = content_type else {
        return true;
    };
    let media_type = content_type
        .split(';')
        .next()
        .unwrap_or_default()
        .trim()
        .to_ascii_lowercase();
    media_type.starts_with("text/")
        || media_type == "application/json"
        || media_type.ends_with("+json")
        || media_type == "application/xml"
        || media_type.ends_with("+xml")
        || matches!(
            media_type.as_str(),
            "application/javascript" | "application/x-javascript"
        )
}

fn content_for_output(body: &[u8]) -> (String, bool) {
    let content = String::from_utf8_lossy(body);
    let mut output = String::new();
    let mut json_bytes = 0;
    for character in content.chars() {
        let escaped_bytes = match character {
            '\"' | '\\' | '\u{0008}' | '\t' | '\n' | '\u{000c}' | '\r' => 2,
            '\u{0000}'..='\u{001f}' => 6,
            _ => character.len_utf8(),
        };
        if json_bytes + escaped_bytes > MAX_RESPONSE_BYTES {
            return (output, true);
        }
        output.push(character);
        json_bytes += escaped_bytes;
    }
    (output, false)
}

async fn read_limited_body(response: &mut reqwest::Response) -> Result<(Vec<u8>, bool)> {
    let mut body = Vec::new();
    while let Some(chunk) = response
        .chunk()
        .await
        .context("failed while reading web__fetch response body")?
    {
        let remaining = MAX_RESPONSE_BYTES.saturating_sub(body.len());
        if chunk.len() > remaining {
            body.extend_from_slice(&chunk[..remaining]);
            return Ok((body, true));
        }
        body.extend_from_slice(&chunk);
        if body.len() == MAX_RESPONSE_BYTES {
            if response
                .chunk()
                .await
                .context("failed while checking web__fetch response limit")?
                .is_some()
            {
                return Ok((body, true));
            }
            return Ok((body, false));
        }
    }
    Ok((body, false))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn accepts_public_http_urls_and_strips_fragments() {
        let url = parse_public_url("https://example.com/docs?q=1#section").expect("public URL");
        assert_eq!(url.as_str(), "https://example.com/docs?q=1");
    }

    #[test]
    fn rejects_non_http_credentials_and_private_literal_hosts() {
        for url in [
            "file:///etc/passwd",
            "https://user:pass@example.com/",
            "http://127.0.0.1/",
            "http://10.0.0.1/",
            "http://169.254.169.254/latest/meta-data/",
            "http://[::1]/",
            "http://[fc00::1]/",
            "http://[2001:db8::1]/",
        ] {
            assert!(
                parse_public_url(url).is_err(),
                "URL should be blocked: {url}"
            );
        }
    }

    #[test]
    fn public_ip_classification_covers_ipv4_and_ipv6_boundaries() {
        for ip in ["1.1.1.1", "8.8.8.8", "2606:4700:4700::1111"] {
            assert!(ensure_public_ip(ip.parse().unwrap()).is_ok(), "{ip}");
        }
        for ip in [
            "0.1.2.3",
            "100.64.0.1",
            "172.31.255.255",
            "192.0.2.1",
            "198.19.0.1",
            "224.0.0.1",
            "255.255.255.255",
            "::",
            "::1",
            "::ffff:127.0.0.1",
            "2001::1",
            "2001:20::1",
            "2001:db8::1",
            "2002:7f00:1::",
        ] {
            assert!(ensure_public_ip(ip.parse().unwrap()).is_err(), "{ip}");
        }
    }

    #[test]
    fn resolves_relative_redirects_before_revalidation() {
        let base = parse_public_url("https://example.com/a/b").unwrap();
        let next = validate_public_url(base.join("../c?q=1#ignored").unwrap()).unwrap();
        assert_eq!(next.as_str(), "https://example.com/c?q=1");
    }

    #[test]
    fn output_content_is_bounded_after_utf8_and_json_escaping() {
        let invalid_utf8 = vec![0xff; MAX_RESPONSE_BYTES];
        let (content, truncated) = content_for_output(&invalid_utf8);
        assert!(truncated);
        assert!(serde_json::to_string(&content).unwrap().len() <= MAX_RESPONSE_BYTES + 2);

        let control_bytes = vec![0; MAX_RESPONSE_BYTES];
        let (content, truncated) = content_for_output(&control_bytes);
        assert!(truncated);
        assert!(serde_json::to_string(&content).unwrap().len() <= MAX_RESPONSE_BYTES + 2);
    }

    #[tokio::test]
    async fn total_timeout_bounds_the_whole_fetch_future() {
        let result = fetch_with_timeout(
            async {
                tokio::time::sleep(Duration::from_millis(50)).await;
                Ok(())
            },
            Duration::from_millis(1),
        )
        .await;

        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("web__fetch exceeded the total 1ms timeout")
        );
    }

    #[test]
    fn content_type_filter_accepts_text_json_and_xml_only() {
        for content_type in [
            None,
            Some("text/html; charset=utf-8"),
            Some("application/json"),
            Some("application/problem+json"),
            Some("application/rss+xml"),
        ] {
            assert!(supported_content_type(content_type), "{content_type:?}");
        }
        for content_type in ["image/png", "application/pdf", "application/octet-stream"] {
            assert!(
                !supported_content_type(Some(content_type)),
                "{content_type}"
            );
        }
    }
}
