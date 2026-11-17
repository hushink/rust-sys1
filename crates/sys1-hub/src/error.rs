use std::path::PathBuf;
use sys1_core::ManifestError;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum HubError {
    #[error("invalid model reference: {0}")]
    InvalidReference(String),
    #[error("no model source for scheme `{0}`")]
    SourceNotFound(String),
    #[error("I/O error at {path}: {source}")]
    Io {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("HTTP error for {url}: {message}")]
    Http { url: String, message: String },
    #[error("HTTP {status} for {url}: {message}")]
    HttpStatus {
        url: String,
        status: u16,
        message: String,
    },
    #[error("invalid JSON from {url}: {message}")]
    Json { url: String, message: String },
    #[error("invalid data from {url}: {message}")]
    InvalidData { url: String, message: String },
    #[error("invalid URL {url}: {message}")]
    Url { url: String, message: String },
    #[error(transparent)]
    Manifest(#[from] ManifestError),
    #[error("hash mismatch for {path}: expected {expected}, got {actual}")]
    HashMismatch {
        path: PathBuf,
        expected: String,
        actual: String,
    },
    #[error("size mismatch for {path}: expected {expected}, got {actual}")]
    SizeMismatch {
        path: PathBuf,
        expected: u64,
        actual: u64,
    },
    #[error("offline mode: {0}")]
    Offline(String),
    #[error("unsupported: {0}")]
    Unsupported(String),
}

impl HubError {
    pub(crate) fn io(path: impl Into<PathBuf>, source: std::io::Error) -> Self {
        Self::Io {
            path: path.into(),
            source,
        }
    }
}
