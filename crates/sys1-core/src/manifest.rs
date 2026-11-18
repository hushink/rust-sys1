use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::path::{Component, Path};
use sys1_protocol::{ModelMetadata, is_valid_release_date};
use thiserror::Error;

/// A model bundle manifest.
///
/// Manifests are produced by a `ModelSource` (local, Hugging Face,
/// ModelScope, or a custom HTTP center) and consumed by the registry.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ModelManifest {
    pub schema_version: u32,
    pub name: String,
    pub description: String,
    /// Release date formatted as `YYYY-MM-DD`.
    pub release_date: String,
    pub family: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub variant: Option<String>,
    /// Inference backend identifier, e.g. `candle` or `onnx`.
    pub backend: String,
    /// Weight/artifact format, e.g. `safetensors` or `onnx`.
    pub format: String,
    pub version: String,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub aliases: Vec<String>,
    pub files: Vec<ModelFile>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ModelFile {
    /// Repository-relative, forward-slash path.
    pub path: String,
    /// Absolute URL, a URL relative to the manifest, or empty to resolve
    /// `path` against the manifest URL.
    #[serde(default)]
    pub url: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sha256: Option<String>,
    /// Git blob hash for sources (Hugging Face) that do not expose sha256 for
    /// non-LFS files.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub git_sha1: Option<String>,
    pub size: u64,
}

#[derive(Clone, Debug, Error, PartialEq, Eq)]
pub enum ManifestError {
    #[error("unsupported schema_version: {0}")]
    SchemaVersion(u32),
    #[error("field `{0}` must not be empty")]
    Empty(&'static str),
    #[error("invalid release_date: {0}")]
    ReleaseDate(String),
    #[error("manifest must contain at least one file")]
    NoFiles,
    #[error("duplicate file path: {0}")]
    DuplicatePath(String),
    #[error("file path must be relative, use forward slashes, and contain no `..`: {0}")]
    InvalidPath(String),
    #[error("invalid sha256 for `{path}`: {value}")]
    InvalidSha256 { path: String, value: String },
    #[error("invalid git sha1 for `{path}`: {value}")]
    InvalidGitSha1 { path: String, value: String },
    #[error("file size must be greater than zero: {0}")]
    ZeroSize(String),
    #[error("duplicate model name: {0}")]
    DuplicateName(String),
    #[error("duplicate alias: {0}")]
    DuplicateAlias(String),
    #[error("alias must not equal its model name: {0}")]
    AliasEqualsName(String),
    #[error("alias conflicts with a model name: {0}")]
    AliasConflict(String),
}

impl ModelManifest {
    pub fn validate(&self) -> Result<(), ManifestError> {
        if self.schema_version != 1 {
            return Err(ManifestError::SchemaVersion(self.schema_version));
        }

        for (field, value) in [
            ("name", self.name.as_str()),
            ("description", self.description.as_str()),
            ("family", self.family.as_str()),
            ("backend", self.backend.as_str()),
            ("format", self.format.as_str()),
            ("version", self.version.as_str()),
        ] {
            if value.trim().is_empty() {
                return Err(ManifestError::Empty(field));
            }
        }

        if !is_valid_release_date(&self.release_date) {
            return Err(ManifestError::ReleaseDate(self.release_date.clone()));
        }

        if self.files.is_empty() {
            return Err(ManifestError::NoFiles);
        }

        let mut paths = HashSet::new();
        for file in &self.files {
            if !is_valid_relative_path(&file.path) {
                return Err(ManifestError::InvalidPath(file.path.clone()));
            }
            if !paths.insert(file.path.clone()) {
                return Err(ManifestError::DuplicatePath(file.path.clone()));
            }
            if file.size == 0 {
                return Err(ManifestError::ZeroSize(file.path.clone()));
            }
            if let Some(value) = &file.sha256
                && !is_hex(value, 64)
            {
                return Err(ManifestError::InvalidSha256 {
                    path: file.path.clone(),
                    value: value.clone(),
                });
            }
            if let Some(value) = &file.git_sha1
                && !is_hex(value, 40)
            {
                return Err(ManifestError::InvalidGitSha1 {
                    path: file.path.clone(),
                    value: value.clone(),
                });
            }
        }

        let mut aliases = HashSet::new();
        for alias in &self.aliases {
            if alias.trim().is_empty() {
                return Err(ManifestError::Empty("aliases"));
            }
            if alias == &self.name {
                return Err(ManifestError::AliasEqualsName(alias.clone()));
            }
            if !aliases.insert(alias) {
                return Err(ManifestError::DuplicateAlias(alias.clone()));
            }
        }

        Ok(())
    }

    pub fn metadata(&self) -> ModelMetadata {
        ModelMetadata::new(
            self.name.clone(),
            self.description.clone(),
            self.release_date.clone(),
        )
    }
}

fn is_valid_relative_path(value: &str) -> bool {
    if value.is_empty() || value.contains('\\') || value.contains('\0') {
        return false;
    }
    let path = Path::new(value);
    if path.is_absolute() {
        return false;
    }
    path.components()
        .all(|component| matches!(component, Component::Normal(_) | Component::CurDir))
}

fn is_hex(value: &str, length: usize) -> bool {
    value.len() == length && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}
