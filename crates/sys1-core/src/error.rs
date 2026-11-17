use crate::model::ModelError;
use sys1_protocol::{AnswerType, ValidationReport};
use thiserror::Error;

/// Registry and in-process inference errors.
#[derive(Debug, Error)]
pub enum Error {
    #[error("unknown model: {model}")]
    ModelNotFound { model: String },
    #[error("model `{model}` failed: {source}")]
    Model {
        model: String,
        #[source]
        source: ModelError,
    },
    #[error(transparent)]
    Validation(#[from] ValidationReport),
}

/// A typed answer lookup failure.
#[derive(Clone, Debug, Error, PartialEq, Eq)]
pub enum AnswerError {
    #[error("answer `{id}` is missing")]
    Missing { id: String },
    #[error("answer `{id}` has type {actual:?}, expected {expected:?}")]
    WrongType {
        id: String,
        expected: AnswerType,
        actual: AnswerType,
    },
}
