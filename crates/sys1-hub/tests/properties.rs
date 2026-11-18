//! Deterministic property tests for reference parsing and cache keys.

use std::collections::HashSet;
use sys1_hub::{ModelRef, default_cache_key};

/// Small deterministic generator so failures reproduce without a dependency.
struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Self(seed)
    }

    fn next(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        self.0
    }

    fn pick<'a>(&mut self, values: &'a [&'a str]) -> &'a str {
        values[(self.next() as usize) % values.len()]
    }

    fn text(&mut self, max_len: usize) -> String {
        const ALPHABET: &[&str] = &[
            "a", "Z", "0", "-", "_", ".", "/", "\\", ":", "@", "#", " ", "é", "中", "😀", "..", "|",
        ];
        let len = (self.next() as usize) % (max_len + 1);
        (0..len).map(|_| self.pick(ALPHABET)).collect()
    }
}

#[test]
fn cache_keys_are_filesystem_safe_and_stable() {
    let mut rng = Rng::new(0x5eed);
    let mut seen = HashSet::new();
    for _ in 0..2000 {
        let reference = ModelRef {
            scheme: rng.text(6),
            location: rng.text(40),
            revision: Some(rng.text(12)),
            subfolder: Some(rng.text(12)),
        };
        let key = default_cache_key(&reference);
        assert_eq!(key, default_cache_key(&reference), "{reference:?}");
        assert!(!key.is_empty() && key.len() <= 61, "{key:?}");
        assert!(!key.starts_with('-') && !key.ends_with('-'), "{key:?}");
        assert!(
            key.chars()
                .all(|ch| ch.is_ascii_lowercase() || ch.is_ascii_digit() || ch == '-'),
            "{key:?} for {reference:?}"
        );
        assert!(!key.contains("--"), "{key:?}");
        assert!(seen.insert(key), "two generated references collided");
    }
}

#[test]
fn repository_references_round_trip() {
    let mut rng = Rng::new(0xc0ffee);
    const PARTS: &[&str] = &["owner", "repo", "main", "v1.0", "sub/dir", "naïve", "a-b_c"];
    for _ in 0..1000 {
        let scheme = if rng.next().is_multiple_of(2) {
            "hf"
        } else {
            "modelscope"
        };
        let owner = rng.pick(PARTS);
        let repo = rng.pick(PARTS);
        let revision = rng.pick(PARTS);
        let subfolder = rng.pick(PARTS);
        let text = format!("{scheme}:{owner}/{repo}@{revision}#{subfolder}");
        let reference: ModelRef = text.parse().expect("generated reference parses");
        assert_eq!(reference.scheme, scheme);
        assert_eq!(reference.location, format!("{owner}/{repo}"));
        assert_eq!(reference.revision.as_deref(), Some(revision));
        assert_eq!(reference.subfolder.as_deref(), Some(subfolder));
        assert_eq!(reference.raw_key(), text);
    }
}
