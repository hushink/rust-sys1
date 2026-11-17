use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

/// A field path segment in a validation error.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum LocSegment {
    Key(String),
    Index(usize),
}

impl LocSegment {
    pub fn key(value: impl Into<String>) -> Self {
        Self::Key(value.into())
    }

    pub fn index(value: usize) -> Self {
        Self::Index(value)
    }
}

impl From<&str> for LocSegment {
    fn from(value: &str) -> Self {
        Self::key(value)
    }
}

impl From<String> for LocSegment {
    fn from(value: String) -> Self {
        Self::key(value)
    }
}

impl From<usize> for LocSegment {
    fn from(value: usize) -> Self {
        Self::index(value)
    }
}

/// A request validation error at a specific field or array element.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ValidationError {
    pub loc: Vec<LocSegment>,
    pub msg: String,
    #[serde(rename = "type")]
    pub kind: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub input: Option<Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ctx: Option<Map<String, Value>>,
}

impl ValidationError {
    pub fn new(
        loc: impl IntoIterator<Item = LocSegment>,
        msg: impl Into<String>,
        kind: impl Into<String>,
    ) -> Self {
        Self {
            loc: loc.into_iter().collect(),
            msg: msg.into(),
            kind: kind.into(),
            input: None,
            ctx: None,
        }
    }

    pub fn with_input(mut self, input: impl Into<Value>) -> Self {
        self.input = Some(input.into());
        self
    }

    pub fn with_ctx(mut self, ctx: Map<String, Value>) -> Self {
        self.ctx = Some(ctx);
        self
    }
}

/// A collection of validation errors returned as HTTP 422.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ValidationReport {
    errors: Vec<ValidationError>,
}

impl ValidationReport {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn single(error: ValidationError) -> Self {
        Self {
            errors: vec![error],
        }
    }

    pub fn push(&mut self, error: ValidationError) {
        self.errors.push(error);
    }

    pub fn is_empty(&self) -> bool {
        self.errors.is_empty()
    }

    pub fn errors(&self) -> &[ValidationError] {
        &self.errors
    }

    pub fn into_errors(self) -> Vec<ValidationError> {
        self.errors
    }

    pub fn extend(&mut self, other: ValidationReport) {
        self.errors.extend(other.errors);
    }

    pub fn into_response(self) -> ErrorResponse {
        ErrorResponse::validation(self.errors)
    }
}

impl From<ValidationError> for ValidationReport {
    fn from(error: ValidationError) -> Self {
        Self::single(error)
    }
}

impl From<Vec<ValidationError>> for ValidationReport {
    fn from(errors: Vec<ValidationError>) -> Self {
        Self { errors }
    }
}

impl std::fmt::Display for ValidationReport {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "{} validation error(s)", self.errors.len())
    }
}

impl std::error::Error for ValidationReport {}

/// The official error envelope: `{"detail": ...}`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ErrorResponse {
    pub detail: ErrorDetail,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ErrorDetail {
    Validation(Vec<ValidationError>),
    Message(MessageDetail),
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum MessageDetail {
    Text(String),
    Object(Map<String, Value>),
}

impl ErrorResponse {
    pub fn bad_request(message: impl Into<String>) -> Self {
        Self {
            detail: ErrorDetail::Message(MessageDetail::Text(message.into())),
        }
    }

    pub fn bad_request_object(detail: Map<String, Value>) -> Self {
        Self {
            detail: ErrorDetail::Message(MessageDetail::Object(detail)),
        }
    }

    pub fn validation(errors: Vec<ValidationError>) -> Self {
        Self {
            detail: ErrorDetail::Validation(errors),
        }
    }
}

/// Machine-readable `ValidationError.type` values used by this crate.
///
/// The official schema treats this field as an open string.
pub mod error_type {
    pub const MISSING: &str = "missing";
    pub const JSON_INVALID: &str = "json_invalid";
    pub const TYPE_ERROR: &str = "type_error";
    pub const STATE_TYPE: &str = "state_type";
    pub const JSON_TYPE: &str = "json_type";
    pub const UNION_TAG_INVALID: &str = "union_tag_invalid";
    pub const TOO_SHORT: &str = "too_short";
    pub const TOO_LONG: &str = "too_long";
    pub const TOO_MANY_OPTIONS: &str = "too_many_options";
    pub const VALUE_ERROR: &str = "value_error";
    pub const STRING_TOO_SHORT: &str = "string_too_short";
    pub const UNSUPPORTED: &str = "unsupported";
}
