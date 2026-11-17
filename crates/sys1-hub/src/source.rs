use crate::{HubConfig, HubError, ModelRef};
use sha2::{Digest, Sha256};
use std::path::{Path, PathBuf};
use sys1_core::{ModelFile, ModelManifest};

pub struct SourceContext<'a> {
    pub client: &'a reqwest::blocking::Client,
    pub config: &'a HubConfig,
}

/// A source of model manifests and files.
///
/// Built-in implementations cover local paths, Hugging Face, ModelScope, and
/// static HTTP centers. Custom centers implement this trait and are passed to
/// [`Hub::with_sources`](crate::Hub::with_sources).
pub trait ModelSource: Send + Sync {
    fn scheme(&self) -> &'static str;

    fn matches(&self, reference: &ModelRef) -> bool {
        reference.scheme == self.scheme()
    }

    fn resolve(
        &self,
        reference: &ModelRef,
        context: &SourceContext<'_>,
    ) -> Result<ModelManifest, HubError>;

    fn fetch(
        &self,
        reference: &ModelRef,
        file: &ModelFile,
        destination: &Path,
        context: &SourceContext<'_>,
    ) -> Result<(), HubError>;

    /// Return the local path for a file when the source does not need copying
    /// into the cache.
    fn local_file(&self, reference: &ModelRef, file: &ModelFile) -> Option<PathBuf> {
        let _ = (reference, file);
        None
    }

    /// Return the source directory when the model already lives on disk.
    fn local_directory(&self, reference: &ModelRef) -> Option<PathBuf> {
        let _ = reference;
        None
    }

    fn cache_key(&self, reference: &ModelRef) -> String {
        default_cache_key(reference)
    }
}

/// Stable, filesystem-safe cache key for a reference.
pub fn default_cache_key(reference: &ModelRef) -> String {
    let raw = reference.raw_key();
    let digest = Sha256::digest(raw.as_bytes());
    let hash = hex::encode(digest);
    format!("{}-{}", sanitize(&raw), &hash[..12])
}

fn sanitize(value: &str) -> String {
    let mut out = String::new();
    for ch in value.chars() {
        if ch.is_ascii_alphanumeric() {
            out.push(ch.to_ascii_lowercase());
        } else if !out.ends_with('-') {
            out.push('-');
        }
    }
    let out = out.trim_matches('-');
    let out = out.chars().take(48).collect::<String>();
    if out.is_empty() {
        "model".to_owned()
    } else {
        out
    }
}

pub(crate) fn infer_format(files: &[ModelFile]) -> (String, String) {
    if files.iter().any(|file| file.path.ends_with(".onnx")) {
        ("onnx".to_owned(), "onnx".to_owned())
    } else if files.iter().any(|file| file.path.ends_with(".safetensors")) {
        ("candle".to_owned(), "safetensors".to_owned())
    } else if files.iter().any(|file| file.path.ends_with(".gguf")) {
        ("llama.cpp".to_owned(), "gguf".to_owned())
    } else {
        ("unknown".to_owned(), "unknown".to_owned())
    }
}

pub(crate) fn model_name_from_repo(repo: &str, subfolder: Option<&str>) -> String {
    let repo_name = repo.rsplit('/').next().unwrap_or(repo);
    match subfolder {
        Some(subfolder) if !subfolder.is_empty() => {
            format!("{repo_name}-{}", subfolder.replace('/', "-"))
        }
        _ => repo_name.to_owned(),
    }
}
