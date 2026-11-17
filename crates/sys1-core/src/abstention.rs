//! Opt-in abstention gating over typed answers.
//!
//! This mirrors the official Laya `min_confidence` policy: a scalar threshold
//! applies to every answer, while a map resolves one threshold per
//! question-type/option-count bucket. The answer payload itself is left
//! untouched; the gate result is reported alongside it.

use crate::DecisionOutput;
use crate::model::ModelError;
use indexmap::IndexMap;
use serde::{Deserialize, Serialize};
use sys1_protocol::Answer;

/// Per-call prediction controls. The default performs no abstention.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct PredictionOptions {
    pub min_confidence: Option<MinConfidence>,
}

/// Scalar or per-bucket abstention thresholds.
#[derive(Clone, Debug, PartialEq)]
pub enum MinConfidence {
    /// One threshold for every answer.
    Scalar(f64),
    /// Thresholds keyed by `choice:3-5`, `score:11+`, `noul:2`, or `default`.
    Buckets(IndexMap<String, f64>),
}

/// The gate's decision for one answer.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum AbstentionState {
    Passed,
    Abstained,
    Unevaluated,
}

/// Why an answer passed, abstained, or could not be evaluated.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Abstention {
    pub state: AbstentionState,
    pub low_confidence: bool,
    pub threshold: f64,
}

impl Abstention {
    pub fn passed(threshold: f64) -> Self {
        Self {
            state: AbstentionState::Passed,
            low_confidence: false,
            threshold,
        }
    }

    pub fn abstained(threshold: f64) -> Self {
        Self {
            state: AbstentionState::Abstained,
            low_confidence: true,
            threshold,
        }
    }

    pub fn unevaluated(threshold: f64) -> Self {
        Self {
            state: AbstentionState::Unevaluated,
            low_confidence: false,
            threshold,
        }
    }
}

impl MinConfidence {
    /// Validate and build a scalar threshold.
    pub fn scalar(value: f64) -> Result<Self, ModelError> {
        validate_threshold("min_confidence", value)?;
        Ok(Self::Scalar(value))
    }

    /// Validate and build a per-bucket threshold map.
    pub fn buckets(values: IndexMap<String, f64>) -> Result<Self, ModelError> {
        if values.is_empty() {
            return Err(ModelError::invalid_request(
                "a min_confidence map must not be empty",
            ));
        }
        for (bucket, value) in &values {
            if bucket != DEFAULT_BUCKET_KEY && !BUCKET_KEYS.contains(&bucket.as_str()) {
                return Err(ModelError::invalid_request(format!(
                    "min_confidence bucket {bucket:?} must be one of {BUCKET_KEYS:?} or \
                     {DEFAULT_BUCKET_KEY:?}"
                )));
            }
            validate_threshold(&format!("min_confidence[{bucket:?}]"), *value)?;
        }
        Ok(Self::Buckets(values))
    }

    pub fn validate(&self) -> Result<(), ModelError> {
        match self {
            Self::Scalar(value) => validate_threshold("min_confidence", *value),
            Self::Buckets(values) => {
                Self::buckets(values.clone())?;
                Ok(())
            }
        }
    }

    /// The threshold that applies to `answer`.
    ///
    /// A bucket map falls back to its `default` entry, then to `0.0` (gate
    /// nothing) for an unconfigured bucket.
    pub fn threshold_for(&self, answer: &Answer) -> f64 {
        match self {
            Self::Scalar(value) => *value,
            Self::Buckets(values) => bucket_key(answer)
                .and_then(|bucket| values.get(&bucket).copied())
                .or_else(|| values.get(DEFAULT_BUCKET_KEY).copied())
                .unwrap_or(0.0),
        }
    }
}

/// Apply the gate to every answer in `output`.
///
/// `output.answers` is unchanged. `output.abstentions` is replaced with one
/// entry per answer; an empty map means no gate ran.
pub fn apply_abstention(
    output: &mut DecisionOutput,
    policy: &MinConfidence,
) -> Result<(), ModelError> {
    policy.validate()?;
    let mut abstentions = IndexMap::with_capacity(output.answers.len());
    for (id, answer) in output.answers.iter() {
        let threshold = policy.threshold_for(answer);
        let abstention = match answer_confidence(answer) {
            Some(confidence) if confidence < threshold => Abstention::abstained(threshold),
            Some(_) => Abstention::passed(threshold),
            None => Abstention::unevaluated(threshold),
        };
        abstentions.insert(id.clone(), abstention);
    }
    output.abstentions = abstentions;
    Ok(())
}

/// `answer_confidence`: max probability, falling back to the reported
/// confidence when probabilities are absent. `None` means there is no usable
/// number to gate on.
pub fn answer_confidence(answer: &Answer) -> Option<f64> {
    match answer {
        Answer::Choice(answer) => max_probability(&answer.probabilities, answer.confidence),
        Answer::Score(answer) => max_probability(&answer.probabilities, answer.confidence),
        Answer::Noul(answer) => finite(answer.noul),
        Answer::Box(answer) => answer.confidence.and_then(finite),
    }
}

fn max_probability(probabilities: &IndexMap<String, f64>, fallback: f64) -> Option<f64> {
    let max = probabilities
        .values()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max);
    if max.is_finite() {
        Some(max)
    } else {
        finite(fallback)
    }
}

fn finite(value: f64) -> Option<f64> {
    value.is_finite().then_some(value)
}

/// The bucket name for an answer, or `None` for a type the official map does
/// not name (box).
pub fn bucket_key(answer: &Answer) -> Option<String> {
    let (kind, count) = match answer {
        Answer::Choice(answer) => {
            if answer.probabilities.is_empty() {
                return None;
            }
            ("choice", answer.probabilities.len())
        }
        Answer::Score(answer) => {
            if answer.probabilities.is_empty() {
                return None;
            }
            ("score", answer.probabilities.len())
        }
        Answer::Noul(_) => ("noul", 2),
        Answer::Box(_) => return None,
    };
    Some(format!("{kind}:{}", bucket_size(count)))
}

fn bucket_size(count: usize) -> &'static str {
    if count <= 2 {
        "2"
    } else if count <= 5 {
        "3-5"
    } else if count <= 10 {
        "6-10"
    } else {
        "11+"
    }
}

/// The one non-bucket key a map may carry.
pub const DEFAULT_BUCKET_KEY: &str = "default";

const BUCKET_KEYS: [&str; 12] = [
    "choice:2",
    "choice:3-5",
    "choice:6-10",
    "choice:11+",
    "score:2",
    "score:3-5",
    "score:6-10",
    "score:11+",
    "noul:2",
    "noul:3-5",
    "noul:6-10",
    "noul:11+",
];

fn validate_threshold(label: &str, value: f64) -> Result<(), ModelError> {
    if !value.is_finite() || !(0.0..=1.0).contains(&value) {
        return Err(ModelError::invalid_request(format!(
            "{label} must be a finite float in [0.0, 1.0], got {value}"
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Answers, DecisionOutput};
    use sys1_protocol::{ChoiceAnswer, NoulAnswer, ScoreAnswer, Usage};

    fn output() -> DecisionOutput {
        let answers = IndexMap::from([
            ("billing".to_owned(), Answer::Noul(NoulAnswer { noul: 0.7 })),
            (
                "tone".to_owned(),
                Answer::Choice(ChoiceAnswer {
                    choice: "calm".to_owned(),
                    confidence: 0.2,
                    probabilities: IndexMap::from([
                        ("angry".to_owned(), 0.4),
                        ("calm".to_owned(), 0.6),
                    ]),
                }),
            ),
            (
                "urgency".to_owned(),
                Answer::Score(ScoreAnswer {
                    score: 1.6,
                    confidence: 0.1,
                    legend: IndexMap::new(),
                    probabilities: IndexMap::from([
                        ("0".to_owned(), 0.1),
                        ("1".to_owned(), 0.2),
                        ("2".to_owned(), 0.7),
                    ]),
                }),
            ),
        ]);
        DecisionOutput::new(Answers::new(answers), Usage::new(1, 0))
    }

    #[test]
    fn scalar_gate_flags_answers_below_the_threshold() {
        let mut output = output();
        apply_abstention(&mut output, &MinConfidence::Scalar(0.65)).expect("gate");
        assert_eq!(
            output.abstention("billing").expect("billing").state,
            AbstentionState::Passed
        );
        assert_eq!(
            output.abstention("tone").expect("tone").state,
            AbstentionState::Abstained
        );
        assert!(output.abstention("tone").expect("tone").low_confidence);
        assert_eq!(output.abstention("tone").expect("tone").threshold, 0.65);
        assert_eq!(
            output.abstention("urgency").expect("urgency").state,
            AbstentionState::Passed
        );
        // The payload itself is untouched.
        assert_eq!(output.answers.noul("billing").expect("noul").noul, 0.7);
    }

    #[test]
    fn bucket_map_uses_the_answer_bucket_and_default() {
        let policy = MinConfidence::buckets(IndexMap::from([
            ("choice:2".to_owned(), 0.8),
            ("default".to_owned(), 0.5),
        ]))
        .expect("policy");
        assert_eq!(
            policy.threshold_for(output().answers.get("tone").expect("tone")),
            0.8
        );
        assert_eq!(
            policy.threshold_for(output().answers.get("urgency").expect("urgency")),
            0.5
        );
        assert_eq!(
            policy.threshold_for(output().answers.get("billing").expect("billing")),
            0.5
        );
    }

    #[test]
    fn zero_runs_the_gate_without_abstaining() {
        let mut output = output();
        apply_abstention(&mut output, &MinConfidence::Scalar(0.0)).expect("gate");
        for id in ["billing", "tone", "urgency"] {
            let abstention = output.abstention(id).expect("gate ran");
            assert_eq!(abstention.state, AbstentionState::Passed);
            assert_eq!(abstention.threshold, 0.0);
        }
    }

    #[test]
    fn rejects_invalid_thresholds_and_buckets() {
        assert!(MinConfidence::scalar(-0.1).is_err());
        assert!(MinConfidence::scalar(1.1).is_err());
        assert!(MinConfidence::scalar(f64::NAN).is_err());
        assert!(MinConfidence::buckets(IndexMap::from([("bogus".to_owned(), 0.5)])).is_err());
        assert!(MinConfidence::buckets(IndexMap::new()).is_err());
    }

    #[test]
    fn missing_confidence_is_unevaluated() {
        let mut output = output();
        output.answers = Answers::new(IndexMap::from([(
            "tone".to_owned(),
            Answer::Choice(ChoiceAnswer {
                choice: "calm".to_owned(),
                confidence: f64::NAN,
                probabilities: IndexMap::new(),
            }),
        )]));
        apply_abstention(&mut output, &MinConfidence::Scalar(0.5)).expect("gate");
        assert_eq!(
            output.abstention("tone").expect("tone").state,
            AbstentionState::Unevaluated
        );
    }
}
