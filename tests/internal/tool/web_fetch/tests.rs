use super::*;

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
