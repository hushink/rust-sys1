use crate::answer::Answer;
use indexmap::IndexMap;
use serde::{Deserialize, Serialize};

/// A successful System One response.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SystemOneResponse {
    pub model: String,
    pub answers: IndexMap<String, Answer>,
    pub usage: Usage,
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Usage {
    pub input_tokens: u64,
    pub output_tokens: u64,
}

impl SystemOneResponse {
    pub fn new(model: impl Into<String>, answers: IndexMap<String, Answer>, usage: Usage) -> Self {
        Self {
            model: model.into(),
            answers,
            usage,
        }
    }
}

impl Usage {
    pub fn new(input_tokens: u64, output_tokens: u64) -> Self {
        Self {
            input_tokens,
            output_tokens,
        }
    }
}
