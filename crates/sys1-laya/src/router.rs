use crate::model::{LayaModel, LayaOptions};
use crate::prepare::render_state;
use indexmap::IndexMap;
use std::collections::{HashMap, HashSet};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use sys1_core::{DecisionModel, DecisionOutput, ModelError, Prediction, Questions};
use sys1_protocol::State;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum Variant {
    English,
    Multilingual,
    TypedDecisions,
}

/// Laya's automatic router across the three published checkpoints.
///
/// Selection is deliberately simple in v1: typed-decisions question-id
/// signatures win, then mostly-ASCII text with English function words selects
/// English, and everything else falls back to multilingual.
pub struct LayaRouter {
    english: PathBuf,
    multilingual: PathBuf,
    typed_decisions: PathBuf,
    options: LayaOptions,
    loaded: Mutex<IndexMap<Variant, Arc<LayaModel>>>,
    max_loaded: usize,
}

impl LayaRouter {
    /// `root` is the Laya repository root: it must contain `multilingual/` and
    /// `typed-decisions/` subdirectories beside the English checkpoint.
    pub fn new(root: impl Into<PathBuf>, options: LayaOptions) -> Self {
        let root = root.into();
        Self::with_paths(
            root.clone(),
            root.join("multilingual"),
            root.join("typed-decisions"),
            options,
        )
    }

    pub fn with_paths(
        english: impl Into<PathBuf>,
        multilingual: impl Into<PathBuf>,
        typed_decisions: impl Into<PathBuf>,
        options: LayaOptions,
    ) -> Self {
        Self {
            english: english.into(),
            multilingual: multilingual.into(),
            typed_decisions: typed_decisions.into(),
            options,
            loaded: Mutex::new(IndexMap::new()),
            max_loaded: 2,
        }
    }

    fn select(&self, state: &State, questions: &Questions) -> Variant {
        const TYPED_SETS: [&[&str]; 4] = [
            &["action", "needs_review", "outcome", "risk", "urgency"],
            &["action", "category", "churn_risk", "needs_human", "urgency"],
            &[
                "discrepancy_severity",
                "disposition",
                "duplicate",
                "matches_order",
                "urgency",
            ],
            &[
                "credential_compromise",
                "disposition",
                "severity",
                "true_positive",
                "urgency",
            ],
        ];

        let ids: HashSet<&str> = questions.iter().map(|(id, _)| id.as_str()).collect();
        if TYPED_SETS
            .iter()
            .any(|set| set.len() == ids.len() && set.iter().all(|id| ids.contains(id)))
        {
            return Variant::TypedDecisions;
        }

        let variant = if looks_english(&render_state(state)) {
            Variant::English
        } else {
            Variant::Multilingual
        };
        tracing::info!(?variant, "laya router selected checkpoint");
        variant
    }

    fn load(&self, variant: Variant) -> Result<Arc<LayaModel>, ModelError> {
        let mut loaded = self
            .loaded
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if let Some(model) = loaded.get(&variant) {
            return Ok(model.clone());
        }

        let path = match variant {
            Variant::English => &self.english,
            Variant::Multilingual => &self.multilingual,
            Variant::TypedDecisions => &self.typed_decisions,
        };
        tracing::info!(?variant, path = %path.display(), "loading Laya checkpoint");
        let model = Arc::new(LayaModel::load_dir(path, self.options.clone())?);
        loaded.insert(variant, model.clone());

        while loaded.len() > self.max_loaded {
            let Some(oldest) = loaded.keys().next().copied() else {
                break;
            };
            if oldest == variant {
                break;
            }
            loaded.shift_remove(&oldest);
        }

        Ok(model)
    }
}

impl DecisionModel for LayaRouter {
    fn predict(&self, state: &State, questions: &Questions) -> Result<DecisionOutput, ModelError> {
        let variant = self.select(state, questions);
        self.load(variant)?.predict(state, questions)
    }

    fn supports_cross_request_batching(&self) -> bool {
        !self.options.device.is_cpu()
    }

    fn predict_batch(
        &self,
        requests: &[Prediction<'_>],
    ) -> Vec<Result<DecisionOutput, ModelError>> {
        let mut slots: Vec<Option<Result<DecisionOutput, ModelError>>> =
            (0..requests.len()).map(|_| None).collect();
        let mut groups: HashMap<Variant, Vec<usize>> = HashMap::new();
        for (index, request) in requests.iter().enumerate() {
            groups
                .entry(self.select(request.state, request.questions))
                .or_default()
                .push(index);
        }

        for (variant, indices) in groups {
            let model = match self.load(variant) {
                Ok(model) => model,
                Err(error) => {
                    for index in indices {
                        slots[index] = Some(Err(error.clone()));
                    }
                    continue;
                }
            };
            let predictions: Vec<Prediction<'_>> = indices
                .iter()
                .map(|&index| Prediction::new(requests[index].state, requests[index].questions))
                .collect();
            for (index, output) in indices.into_iter().zip(model.predict_batch(&predictions)) {
                slots[index] = Some(output);
            }
        }

        slots
            .into_iter()
            .map(|slot| slot.expect("every request slot is filled"))
            .collect()
    }
}

fn looks_english(text: &str) -> bool {
    if text.trim().is_empty() {
        return false;
    }
    let letters = text.chars().filter(|ch| ch.is_alphabetic()).count();
    if letters == 0 {
        return false;
    }
    let ascii = text
        .chars()
        .filter(|ch| ch.is_alphabetic() && ch.is_ascii())
        .count();
    if ascii * 5 < letters * 4 {
        return false;
    }

    const STOPWORDS: [&str; 12] = [
        "the", "is", "are", "please", "this", "that", "you", "your", "my", "account", "refund",
        "message",
    ];
    text.split_whitespace().any(|word| {
        let word = word
            .trim_matches(|ch: char| !ch.is_alphabetic())
            .to_ascii_lowercase();
        STOPWORDS.contains(&word.as_str())
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn english_detection_uses_function_words() {
        assert!(looks_english(
            "Please refund the duplicate charge on my account."
        ));
        assert!(!looks_english("Bonjour, je voudrais un remboursement."));
        assert!(!looks_english("退款してください"));
        assert!(!looks_english(""));
    }
}
