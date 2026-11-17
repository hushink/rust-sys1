use crate::error::ApiError;
use axum::body::{Bytes, to_bytes};
use axum::extract::Request;
use axum::http::{HeaderMap, header::CONTENT_ENCODING};
use flate2::read::GzDecoder;
use ruzstd::decoding::StreamingDecoder;
use std::io::Read;

pub(crate) async fn read_body(
    request: Request,
    max_body_bytes: usize,
) -> Result<(HeaderMap, Bytes), ApiError> {
    let (parts, body) = request.into_parts();
    let headers = parts.headers;
    let bytes = to_bytes(body, max_body_bytes.saturating_add(1))
        .await
        .map_err(|error| {
            ApiError::payload_too_large(format!(
                "request body exceeds {max_body_bytes} bytes: {error}"
            ))
        })?;
    if bytes.len() > max_body_bytes {
        return Err(ApiError::payload_too_large(format!(
            "request body exceeds {max_body_bytes} bytes"
        )));
    }

    let encoding = headers
        .get(CONTENT_ENCODING)
        .and_then(|value| value.to_str().ok())
        .map(|value| value.trim().to_ascii_lowercase());

    let decoded = match encoding.as_deref() {
        None | Some("") | Some("identity") => bytes,
        Some("gzip") | Some("x-gzip") => {
            let mut decoder = GzDecoder::new(bytes.as_ref());
            let mut decoded = Vec::new();
            decoder
                .read_to_end(&mut decoded)
                .map_err(|error| ApiError::bad_request(format!("malformed gzip body: {error}")))?;
            if decoded.len() > max_body_bytes {
                return Err(ApiError::payload_too_large(format!(
                    "decompressed request body exceeds {max_body_bytes} bytes"
                )));
            }
            Bytes::from(decoded)
        }
        Some("zstd") => {
            let decoder = StreamingDecoder::new(bytes.as_ref())
                .map_err(|error| ApiError::bad_request(format!("malformed zstd body: {error}")))?;
            let mut decoded = Vec::new();
            decoder
                .take(max_body_bytes as u64 + 1)
                .read_to_end(&mut decoded)
                .map_err(|error| ApiError::bad_request(format!("malformed zstd body: {error}")))?;
            if decoded.len() > max_body_bytes {
                return Err(ApiError::payload_too_large(format!(
                    "decompressed request body exceeds {max_body_bytes} bytes"
                )));
            }
            Bytes::from(decoded)
        }
        Some(other) => {
            return Err(ApiError::unsupported_encoding(format!(
                "unsupported Content-Encoding: {other}"
            )));
        }
    };

    Ok((headers, decoded))
}
