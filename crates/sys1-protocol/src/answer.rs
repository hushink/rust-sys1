use crate::content::Content;
use indexmap::IndexMap;
use serde::{Deserialize, Serialize};

/// An answer whose type matches the corresponding question.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum Answer {
    Noul(NoulAnswer),
    Choice(ChoiceAnswer),
    Score(ScoreAnswer),
    Box(BoxAnswer),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum AnswerType {
    Noul,
    Choice,
    Score,
    Box,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct NoulAnswer {
    /// Probability of a yes answer or a true statement, from 0 to 1.
    pub noul: f64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ChoiceAnswer {
    pub choice: String,
    pub confidence: f64,
    pub probabilities: IndexMap<String, f64>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ScoreAnswer {
    pub score: f64,
    pub confidence: f64,
    pub legend: IndexMap<String, Content>,
    pub probabilities: IndexMap<String, f64>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct BoxAnswer {
    #[serde(rename = "box")]
    pub box_: BoxCoordinates,
    /// Null when the model does not provide confidence.
    pub confidence: Option<f64>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct BoxCoordinates {
    pub top: f64,
    pub bottom: f64,
    pub left: f64,
    pub right: f64,
}

impl Answer {
    pub fn answer_type(&self) -> AnswerType {
        match self {
            Self::Noul(_) => AnswerType::Noul,
            Self::Choice(_) => AnswerType::Choice,
            Self::Score(_) => AnswerType::Score,
            Self::Box(_) => AnswerType::Box,
        }
    }
}

impl NoulAnswer {
    pub fn new(noul: f64) -> Self {
        Self { noul }
    }
}

impl ChoiceAnswer {
    pub fn new(
        choice: impl Into<String>,
        confidence: f64,
        probabilities: IndexMap<String, f64>,
    ) -> Self {
        Self {
            choice: choice.into(),
            confidence,
            probabilities,
        }
    }
}

impl ScoreAnswer {
    pub fn new(
        score: f64,
        confidence: f64,
        legend: IndexMap<String, Content>,
        probabilities: IndexMap<String, f64>,
    ) -> Self {
        Self {
            score,
            confidence,
            legend,
            probabilities,
        }
    }
}

impl BoxAnswer {
    pub fn new(box_: BoxCoordinates, confidence: Option<f64>) -> Self {
        Self { box_, confidence }
    }
}

impl BoxCoordinates {
    pub fn new(top: f64, bottom: f64, left: f64, right: f64) -> Self {
        Self {
            top,
            bottom,
            left,
            right,
        }
    }
}
