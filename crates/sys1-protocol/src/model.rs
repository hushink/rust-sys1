use serde::{Deserialize, Serialize};

/// Response body for `GET /v1/models`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ModelMetadataList {
    pub models: Vec<ModelMetadata>,
}

/// A model name or alias available to the caller.
///
/// All three fields are required by the official schema.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ModelMetadata {
    pub name: String,
    pub description: String,
    /// Release date formatted as `YYYY-MM-DD`.
    pub release_date: String,
}

impl ModelMetadataList {
    pub fn new(models: Vec<ModelMetadata>) -> Self {
        Self { models }
    }
}

impl ModelMetadata {
    pub fn new(
        name: impl Into<String>,
        description: impl Into<String>,
        release_date: impl Into<String>,
    ) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            release_date: release_date.into(),
        }
    }
}

/// Loose `YYYY-MM-DD` shape check. Calendar validity is intentionally left to
/// the caller; the official API only documents the format.
pub fn is_valid_release_date(value: &str) -> bool {
    let bytes = value.as_bytes();
    bytes.len() == 10
        && bytes[4] == b'-'
        && bytes[7] == b'-'
        && bytes
            .iter()
            .enumerate()
            .all(|(index, byte)| index == 4 || index == 7 || byte.is_ascii_digit())
}
