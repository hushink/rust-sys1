use crate::error::{LocSegment, ValidationError, ValidationReport, error_type};
use crate::request::SystemOneRequest;
use serde_json::Value;
use serde_path_to_error::Segment;

/// Parse and validate a `POST /v1/systemone` request body.
///
/// Deserialization errors are converted into the official 422 `detail[]`
/// shape, with a best-effort field path.
pub fn parse_request_slice(bytes: &[u8]) -> Result<SystemOneRequest, ValidationReport> {
    let mut deserializer = serde_json::Deserializer::from_slice(bytes);
    let parsed: Result<SystemOneRequest, _> = serde_path_to_error::deserialize(&mut deserializer);

    match parsed {
        Ok(request) => match request.validate() {
            Ok(()) => Ok(request),
            Err(report) => Err(report),
        },
        Err(error) => {
            let mut loc = vec![LocSegment::key("body")];
            for segment in error.path().iter() {
                loc.push(match segment {
                    Segment::Map { key } => LocSegment::key(key),
                    Segment::Seq { index } => LocSegment::index(*index),
                    Segment::Enum { variant } => LocSegment::key(variant),
                    Segment::Unknown => LocSegment::key("<unknown>"),
                });
            }

            let inner = error.into_inner();
            if loc.len() == 1
                && let Some(field) = missing_field_name(&inner)
            {
                loc.push(LocSegment::key(field));
            }

            let kind = classify(&inner);
            let message = inner.to_string();
            Err(ValidationReport::single(ValidationError::new(
                loc, message, kind,
            )))
        }
    }
}

/// Convenience wrapper for tests and callers that already have a JSON value.
pub fn parse_request_value(value: Value) -> Result<SystemOneRequest, ValidationReport> {
    let bytes = serde_json::to_vec(&value).expect("serde_json::Value is always serializable");
    parse_request_slice(&bytes)
}

fn classify(error: &serde_json::Error) -> &'static str {
    let message = error.to_string();
    if error.is_syntax() || error.is_eof() {
        return error_type::JSON_INVALID;
    }
    if message.contains("missing field") {
        return error_type::MISSING;
    }
    if message.contains("unknown variant") || message.contains("unknown tag") {
        return error_type::UNION_TAG_INVALID;
    }
    if message.contains("state must be") {
        return error_type::STATE_TYPE;
    }
    if message.contains("content must be") {
        return error_type::JSON_TYPE;
    }
    if message.contains("invalid type") {
        return error_type::TYPE_ERROR;
    }
    error_type::VALUE_ERROR
}

fn missing_field_name(error: &serde_json::Error) -> Option<String> {
    let message = error.to_string();
    let marker = "missing field `";
    let start = message.find(marker)? + marker.len();
    let rest = &message[start..];
    let end = rest.find('`')?;
    Some(rest[..end].to_owned())
}
