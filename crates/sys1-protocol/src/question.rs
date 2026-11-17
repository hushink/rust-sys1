use crate::content::{Content, deserialize_optional_content};
use indexmap::IndexMap;
use serde::{Deserialize, Serialize};

/// A question about the supplied state.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum Question {
    Noul(NoulQuestion),
    Choice(ChoiceQuestion),
    Score(ScoreQuestion),
    Box(BoxQuestion),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum QuestionType {
    Noul,
    Choice,
    Score,
    Box,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct NoulQuestion {
    #[serde(
        default,
        deserialize_with = "deserialize_optional_content",
        skip_serializing_if = "Option::is_none"
    )]
    pub instructions: Option<Content>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub criteria: Option<NoulCriteria>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct NoulCriteria {
    #[serde(
        rename = "true",
        default,
        deserialize_with = "deserialize_optional_content",
        skip_serializing_if = "Option::is_none"
    )]
    pub is_true: Option<Content>,
    #[serde(
        rename = "false",
        default,
        deserialize_with = "deserialize_optional_content",
        skip_serializing_if = "Option::is_none"
    )]
    pub is_false: Option<Content>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ChoiceQuestion {
    #[serde(
        default,
        deserialize_with = "deserialize_optional_content",
        skip_serializing_if = "Option::is_none"
    )]
    pub instructions: Option<Content>,
    pub criteria: IndexMap<String, Content>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ScoreQuestion {
    #[serde(
        default,
        deserialize_with = "deserialize_optional_content",
        skip_serializing_if = "Option::is_none"
    )]
    pub instructions: Option<Content>,
    pub criteria: Vec<Content>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct BoxQuestion {
    #[serde(
        default,
        deserialize_with = "deserialize_optional_content",
        skip_serializing_if = "Option::is_none"
    )]
    pub instructions: Option<Content>,
}

impl Question {
    pub fn question_type(&self) -> QuestionType {
        match self {
            Self::Noul(_) => QuestionType::Noul,
            Self::Choice(_) => QuestionType::Choice,
            Self::Score(_) => QuestionType::Score,
            Self::Box(_) => QuestionType::Box,
        }
    }

    pub fn instructions(&self) -> Option<&Content> {
        match self {
            Self::Noul(question) => question.instructions.as_ref(),
            Self::Choice(question) => question.instructions.as_ref(),
            Self::Score(question) => question.instructions.as_ref(),
            Self::Box(question) => question.instructions.as_ref(),
        }
    }
}

impl NoulQuestion {
    pub fn new(instructions: impl Into<Content>) -> Self {
        Self {
            instructions: Some(instructions.into()),
            criteria: None,
        }
    }

    pub fn with_criteria(mut self, criteria: NoulCriteria) -> Self {
        self.criteria = Some(criteria);
        self
    }
}

impl NoulCriteria {
    pub fn new(is_true: impl Into<Content>, is_false: impl Into<Content>) -> Self {
        Self {
            is_true: Some(is_true.into()),
            is_false: Some(is_false.into()),
        }
    }
}

impl ChoiceQuestion {
    pub fn new(criteria: IndexMap<String, Content>) -> Self {
        Self {
            instructions: None,
            criteria,
        }
    }

    pub fn with_instructions(mut self, instructions: impl Into<Content>) -> Self {
        self.instructions = Some(instructions.into());
        self
    }
}

impl ScoreQuestion {
    pub fn new(criteria: Vec<Content>) -> Self {
        Self {
            instructions: None,
            criteria,
        }
    }

    pub fn with_instructions(mut self, instructions: impl Into<Content>) -> Self {
        self.instructions = Some(instructions.into());
        self
    }
}

impl BoxQuestion {
    pub fn new(instructions: impl Into<Content>) -> Self {
        Self {
            instructions: Some(instructions.into()),
        }
    }
}
