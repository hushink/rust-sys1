use axum::http::StatusCode;
use sys1_core::{Error as RegistryError, ModelError};
use sys1_protocol::{ErrorResponse, LocSegment, ValidationError, error_type};

pub(crate) struct ApiError {
    pub status: StatusCode,
    pub body: ErrorResponse,
}

impl ApiError {
    pub fn new(status: StatusCode, body: ErrorResponse) -> Self {
        Self { status, body }
    }

    pub fn bad_request(message: impl Into<String>) -> Self {
        Self::new(StatusCode::BAD_REQUEST, ErrorResponse::bad_request(message))
    }

    pub fn validation(errors: Vec<ValidationError>) -> Self {
        Self::new(
            StatusCode::UNPROCESSABLE_ENTITY,
            ErrorResponse::validation(errors),
        )
    }

    pub fn unauthorized() -> Self {
        Self::new(
            StatusCode::UNAUTHORIZED,
            ErrorResponse::bad_request("invalid or missing API key"),
        )
    }

    pub fn payload_too_large(message: impl Into<String>) -> Self {
        Self::new(
            StatusCode::PAYLOAD_TOO_LARGE,
            ErrorResponse::bad_request(message),
        )
    }

    pub fn unsupported_encoding(message: impl Into<String>) -> Self {
        Self::new(
            StatusCode::UNSUPPORTED_MEDIA_TYPE,
            ErrorResponse::bad_request(message),
        )
    }

    pub fn internal(message: impl Into<String>) -> Self {
        Self::new(
            StatusCode::INTERNAL_SERVER_ERROR,
            ErrorResponse::bad_request(message),
        )
    }
}

pub(crate) fn map_registry_error(error: RegistryError) -> ApiError {
    match error {
        RegistryError::Validation(report) => ApiError::validation(report.into_errors()),
        RegistryError::ModelNotFound { model } => {
            ApiError::bad_request(format!("unknown model: {model}"))
        }
        RegistryError::Model { source, .. } => map_model_error(source),
    }
}

fn map_model_error(error: ModelError) -> ApiError {
    match error {
        ModelError::InvalidRequest { message } => {
            if message.starts_with("max_tokens_exceeded") {
                ApiError::bad_request(message)
            } else {
                ApiError::validation(vec![ValidationError::new(
                    [LocSegment::key("body")],
                    message,
                    error_type::VALUE_ERROR,
                )])
            }
        }
        ModelError::ContextLengthExceeded { message } => ApiError::bad_request(message),
        ModelError::Unsupported { message } => ApiError::validation(vec![ValidationError::new(
            [LocSegment::key("body")],
            message,
            error_type::UNSUPPORTED,
        )]),
        ModelError::Load { message } => ApiError::internal(message),
        ModelError::Inference { message } => ApiError::internal(message),
    }
}
