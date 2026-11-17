use std::cmp::Ordering;

pub(crate) const TYPES: [&str; 3] = ["choice", "score", "noul"];

/// Which confidence formula is exposed in the official answer shape.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ConfidenceMode {
    /// `(K * p_max - 1) / (K - 1)`, matching ollaya's TypeSafe profile.
    #[default]
    TypeSafe,
    /// Laya's normalized entropy: `1 - H(p) / ln(K)`.
    LayaNative,
}

pub(crate) fn bucket(kind: usize, count: usize) -> String {
    let size = if count <= 2 {
        "2"
    } else if count <= 5 {
        "3-5"
    } else if count <= 10 {
        "6-10"
    } else {
        "11+"
    };
    format!("{}:{size}", TYPES[kind])
}

pub(crate) fn probability(logits: &[f32], temperature: f32) -> Vec<f32> {
    if logits.is_empty() {
        return Vec::new();
    }
    let temperature = if temperature.is_finite() && temperature > 0.0 {
        temperature.clamp(0.5, 5.0)
    } else {
        1.0
    };
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max) / temperature;
    if !max.is_finite() {
        let uniform = 1.0 / logits.len() as f32;
        return vec![uniform; logits.len()];
    }
    let mut values: Vec<f32> = logits
        .iter()
        .map(|value| (value / temperature - max).exp())
        .collect();
    let total: f32 = values.iter().sum();
    if total > 0.0 {
        values.iter_mut().for_each(|value| *value /= total);
    }
    values
}

pub(crate) fn laya_confidence(probabilities: &[f32]) -> f64 {
    if probabilities.len() < 2 {
        return 1.0;
    }
    let entropy: f64 = probabilities
        .iter()
        .map(|value| {
            let value = *value as f64;
            -value * value.max(1e-12).ln()
        })
        .sum();
    (1.0 - entropy / (probabilities.len() as f64).ln()).clamp(0.0, 1.0)
}

pub(crate) fn typesafe_confidence(probabilities: &[f32]) -> f64 {
    let count = probabilities.len();
    if count < 2 {
        return 1.0;
    }
    let max = probabilities
        .iter()
        .copied()
        .fold(f32::NEG_INFINITY, f32::max) as f64;
    ((count as f64 * max - 1.0) / (count as f64 - 1.0)).clamp(0.0, 1.0)
}

pub(crate) fn confidence(mode: ConfidenceMode, probabilities: &[f32]) -> f64 {
    match mode {
        ConfidenceMode::TypeSafe => typesafe_confidence(probabilities),
        ConfidenceMode::LayaNative => laya_confidence(probabilities),
    }
}

pub(crate) fn argmax(values: &[f32]) -> usize {
    let mut best = 0;
    for (index, value) in values.iter().enumerate() {
        if value.total_cmp(&values[best]) == Ordering::Greater {
            best = index;
        }
    }
    best
}

/// CPython `round(value, 4)`-compatible rounding.
///
/// Rust's fixed-precision float formatting is correctly rounded and uses
/// ties-to-even, which matches CPython's decimal `round` behavior.
pub(crate) fn round4(value: f64) -> f64 {
    if !value.is_finite() {
        return value;
    }
    format!("{value:.4}").parse::<f64>().unwrap_or(value)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn buckets_match_laya() {
        assert_eq!(bucket(0, 2), "choice:2");
        assert_eq!(bucket(1, 4), "score:3-5");
        assert_eq!(bucket(2, 11), "noul:11+");
    }

    #[test]
    fn softmax_is_stable_and_normalized() {
        let probabilities = probability(&[1.0, 2.0, 3.0], 1.0);
        let total: f32 = probabilities.iter().sum();
        assert!((total - 1.0).abs() < 1e-6);
        assert!(probabilities[2] > probabilities[1]);
        assert!(probabilities[1] > probabilities[0]);
    }

    #[test]
    fn confidence_formulas() {
        let probabilities = [0.8, 0.1, 0.1];
        assert!((typesafe_confidence(&probabilities) - 0.7).abs() < 1e-6);
        let native = laya_confidence(&probabilities);
        assert!(native > 0.0 && native < 1.0);
        assert_eq!(typesafe_confidence(&[1.0]), 1.0);
        assert_eq!(laya_confidence(&[1.0]), 1.0);
    }

    #[test]
    fn argmax_keeps_the_first_tie() {
        assert_eq!(argmax(&[1.0, 2.0, 2.0]), 1);
    }

    #[test]
    fn round4_matches_cpython() {
        let cases = [
            (0.00005, 0.0001),
            (0.00015, 0.0001),
            (0.00025, 0.0003),
            (0.00035, 0.0003),
            (1.23445, 1.2345),
            (1.23455, 1.2346),
            (2.675, 2.675),
            (2.67505, 2.6751),
            (-0.00005, -0.0001),
            (-0.00015, -0.0001),
            (0.99995, 1.0),
            (0.12345, 0.1235),
            (1.0 / 3.0, 0.3333),
        ];
        for (input, expected) in cases {
            assert_eq!(round4(input), expected, "round4({input})");
        }
        assert_eq!(round4(-0.0).to_bits(), (-0.0_f64).to_bits());
    }
}
