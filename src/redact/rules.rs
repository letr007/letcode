use std::sync::OnceLock;

use regex::Regex;

use super::Span;

struct Rule {
    regex: Regex,
    kind: &'static str,
}

fn rule(pattern: &str, kind: &'static str) -> Rule {
    Rule {
        regex: Regex::new(pattern).expect("a built-in rule compiles"),
        kind,
    }
}

fn rules() -> &'static [Rule] {
    static RULES: OnceLock<Vec<Rule>> = OnceLock::new();
    RULES.get_or_init(|| {
        vec![
            rule(
                r"-----BEGIN [A-Z ]*PRIVATE KEY-----[\s\S]*?-----END [A-Z ]*PRIVATE KEY-----",
                "common_secret",
            ),
            rule(r"\bsk-[A-Za-z0-9_-]{16,}", "common_secret"),
            rule(r"\bAKIA[0-9A-Z]{16}\b", "common_secret"),
            rule(r"\b(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,})", "common_secret"),
            rule(r"\bglpat-[A-Za-z0-9_-]{20,}", "common_secret"),
            rule(r"\bxox[abposr]-[A-Za-z0-9-]{10,}", "common_secret"),
            rule(r"\bAIza[0-9A-Za-z_-]{35}", "common_secret"),
            rule(
                r"\beyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}",
                "common_secret",
            ),
            rule(
                r#"(?i)(?:password|passwd|pwd|secret|token|api[_-]?key)\s*[:=]\s*("[^"]*"|'[^']*'|[^\s,;'"]{6,})"#,
                "common_secret",
            ),
            rule(r"(?i)authorization:\s*\S+\s+(\S{12,})", "common_secret"),
            rule(r"\b[a-z][a-z0-9+.-]*://[^/\s:@]+:([^/\s@]+)@", "common_secret"),
            rule(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b", "email"),
            rule(r"\b1[3-9]\d{9}\b", "phone"),
            rule(
                r"\b[1-9]\d{5}(?:18|19|20)\d{2}(?:0[1-9]|1[0-2])(?:0[1-9]|[12]\d|3[01])\d{3}[\dXx]\b",
                "account",
            ),
            rule(r"\b(?:\d{13,17}|\d{19})\b", "payment_card"),
        ]
    })
}

pub(crate) fn scan(text: &str) -> Vec<Span> {
    let mut spans = Vec::new();
    for rule in rules() {
        for captures in rule.regex.captures_iter(text) {
            let matched = match captures.get(1) {
                Some(group) => group,
                None => captures.get(0).expect("a match has a whole group"),
            };
            if !accepted(rule.kind, matched.as_str()) {
                continue;
            }
            spans.push(Span {
                start: matched.start(),
                end: matched.end(),
                kind: rule.kind,
                score: 1.0,
            });
        }
    }
    spans.sort_by(|a, b| a.start.cmp(&b.start).then(b.end.cmp(&a.end)));
    spans.dedup_by(|a, b| a.start == b.start && a.end == b.end);
    spans
}

fn accepted(kind: &str, value: &str) -> bool {
    match kind {
        "account" => id_card_checksum(value),
        "payment_card" => luhn(value),
        _ => true,
    }
}

fn luhn(value: &str) -> bool {
    let digits: Vec<u32> = value.chars().filter_map(|c| c.to_digit(10)).collect();
    if digits.len() < 13 {
        return false;
    }
    let sum: u32 = digits
        .iter()
        .rev()
        .enumerate()
        .map(|(index, &digit)| {
            if index % 2 == 1 {
                let doubled = digit * 2;
                if doubled > 9 { doubled - 9 } else { doubled }
            } else {
                digit
            }
        })
        .sum();
    sum.is_multiple_of(10)
}

fn id_card_checksum(value: &str) -> bool {
    let bytes = value.as_bytes();
    if bytes.len() != 18 {
        return false;
    }
    const WEIGHTS: [u32; 17] = [7, 9, 10, 5, 8, 4, 2, 1, 6, 3, 7, 9, 10, 5, 8, 4, 2];
    const CHECK: [u8; 11] = *b"10X98765432";
    let mut sum = 0u32;
    for (index, weight) in WEIGHTS.iter().enumerate() {
        let Some(digit) = bytes[index].checked_sub(b'0') else {
            return false;
        };
        if digit > 9 {
            return false;
        }
        sum += u32::from(digit) * weight;
    }
    CHECK[(sum % 11) as usize] == bytes[17].to_ascii_uppercase()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn kinds(text: &str) -> Vec<(&'static str, &str)> {
        let mut found: Vec<(&'static str, &str)> = scan(text)
            .into_iter()
            .map(|span| (span.kind, &text[span.start..span.end]))
            .collect();
        found.sort();
        found
    }

    #[test]
    fn finds_secrets_and_leaves_ordinary_text_alone() {
        assert_eq!(
            kinds("OPENAI_API_KEY=sk-proj-abcdefghijklmnopqrstuvwxyz0123456789"),
            vec![(
                "common_secret",
                "sk-proj-abcdefghijklmnopqrstuvwxyz0123456789"
            )]
        );
        assert_eq!(
            kinds("aws AKIAIOSFODNN7EXAMPLE and gh token ghp_1234567890abcdefghijklmnopqrstuvwx"),
            vec![
                ("common_secret", "AKIAIOSFODNN7EXAMPLE"),
                ("common_secret", "ghp_1234567890abcdefghijklmnopqrstuvwx"),
            ]
        );
        assert_eq!(
            kinds("password=hunter2-not-real next line\nversion 1.2.3 commit 3f9a1c2"),
            vec![("common_secret", "hunter2-not-real")]
        );
        assert!(kinds("order 202409301234567890, total 128.00, build 4.8.2").is_empty());
        assert!(kinds("card 4111111111111112").is_empty());
        assert_eq!(
            kinds("id 11010519491231002X and card 4111111111111111"),
            vec![
                ("account", "11010519491231002X"),
                ("payment_card", "4111111111111111")
            ]
        );
    }
}
