use crate::error::{LocSegment, ValidationError, ValidationReport, error_type};
use crate::question::Question;
use crate::state::State;
use indexmap::IndexMap;
use serde::{Deserialize, Serialize};

/// Request body for `POST /v1/systemone`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SystemOneRequest {
    pub model: String,
    pub state: State,
    pub questions: IndexMap<String, Question>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub assets: Option<IndexMap<String, Asset>>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Asset {
    #[serde(rename = "type")]
    pub kind: AssetKind,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub format: Option<AssetFormat>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub data_base64: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub part: Option<String>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum AssetKind {
    Image,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum AssetFormat {
    Png,
    Jpg,
    Jpeg,
}

impl SystemOneRequest {
    pub fn new(
        model: impl Into<String>,
        state: State,
        questions: IndexMap<String, Question>,
    ) -> Self {
        Self {
            model: model.into(),
            state,
            questions,
            assets: None,
        }
    }

    /// Validate the official request constraints that are not expressible in
    /// Rust's type system.
    pub fn validate(&self) -> Result<(), ValidationReport> {
        let mut report = ValidationReport::new();

        if self.model.trim().is_empty() {
            report.push(ValidationError::new(
                [LocSegment::key("body"), LocSegment::key("model")],
                "String should have at least 1 character",
                error_type::STRING_TOO_SHORT,
            ));
        }

        if let Err(questions) = validate_questions(&self.questions) {
            report.extend(questions);
        }

        if let Some(assets) = &self.assets {
            if assets.is_empty() {
                report.push(ValidationError::new(
                    [LocSegment::key("body"), LocSegment::key("assets")],
                    "Dictionary should have at least 1 item",
                    error_type::TOO_SHORT,
                ));
            }

            for (asset_id, asset) in assets {
                let has_part = asset.part.as_deref().is_some_and(|part| !part.is_empty());
                let has_inline = asset.format.is_some() && asset.data_base64.is_some();
                if !has_part && !has_inline {
                    report.push(ValidationError::new(
                        [
                            LocSegment::key("body"),
                            LocSegment::key("assets"),
                            LocSegment::key(asset_id),
                        ],
                        "Asset requires either `part` or both `format` and `data_base64`",
                        error_type::VALUE_ERROR,
                    ));
                }
            }
        }

        if report.is_empty() {
            Ok(())
        } else {
            Err(report)
        }
    }
}

/// Validate question counts and criteria limits shared by the HTTP server and
/// the in-process inference API.
pub fn validate_questions(questions: &IndexMap<String, Question>) -> Result<(), ValidationReport> {
    let mut report = ValidationReport::new();

    if questions.is_empty() {
        report.push(ValidationError::new(
            [LocSegment::key("body"), LocSegment::key("questions")],
            "Dictionary should have at least 1 item",
            error_type::TOO_SHORT,
        ));
    }

    for (question_id, question) in questions {
        match question {
            Question::Choice(choice) => {
                if choice.criteria.is_empty() {
                    report.push(ValidationError::new(
                        [
                            LocSegment::key("body"),
                            LocSegment::key("questions"),
                            LocSegment::key(question_id),
                            LocSegment::key("criteria"),
                        ],
                        "Dictionary should have at least 1 item",
                        error_type::TOO_SHORT,
                    ));
                } else if choice.criteria.len() > 255 {
                    report.push(ValidationError::new(
                        [
                            LocSegment::key("body"),
                            LocSegment::key("questions"),
                            LocSegment::key(question_id),
                            LocSegment::key("criteria"),
                        ],
                        "Dictionary should have at most 255 items",
                        error_type::TOO_LONG,
                    ));
                }
            }
            Question::Score(score) => {
                if score.criteria.is_empty() {
                    report.push(ValidationError::new(
                        [
                            LocSegment::key("body"),
                            LocSegment::key("questions"),
                            LocSegment::key(question_id),
                            LocSegment::key("criteria"),
                        ],
                        "List should have at least 1 item",
                        error_type::TOO_SHORT,
                    ));
                } else if score.criteria.len() > 10 {
                    report.push(ValidationError::new(
                        [
                            LocSegment::key("body"),
                            LocSegment::key("questions"),
                            LocSegment::key(question_id),
                            LocSegment::key("criteria"),
                        ],
                        "List should have at most 10 items",
                        error_type::TOO_LONG,
                    ));
                }
            }
            Question::Noul(_) | Question::Box(_) => {}
        }
    }

    if report.is_empty() {
        Ok(())
    } else {
        Err(report)
    }
}

impl Asset {
    pub fn inline(format: AssetFormat, data_base64: impl Into<String>) -> Self {
        Self {
            kind: AssetKind::Image,
            format: Some(format),
            data_base64: Some(data_base64.into()),
            part: None,
        }
    }

    pub fn multipart(part: impl Into<String>) -> Self {
        Self {
            kind: AssetKind::Image,
            format: None,
            data_base64: None,
            part: Some(part.into()),
        }
    }
}
