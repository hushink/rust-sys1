use crate::reference::file_url_to_path;
use crate::{HubError, ModelRef, ModelSource, SourceContext, download};
use std::fs;
use std::path::{Path, PathBuf};
use sys1_core::{ModelFile, ModelManifest};

/// A model directory (or a single `model.json`) on the local filesystem.
pub struct LocalSource;

impl ModelSource for LocalSource {
    fn scheme(&self) -> &'static str {
        "local"
    }

    fn resolve(
        &self,
        reference: &ModelRef,
        _context: &SourceContext<'_>,
    ) -> Result<ModelManifest, HubError> {
        let (manifest_path, _) = manifest_path_and_dir(&reference.location)?;
        let bytes =
            fs::read(&manifest_path).map_err(|error| HubError::io(&manifest_path, error))?;
        let manifest: ModelManifest =
            serde_json::from_slice(&bytes).map_err(|error| HubError::Json {
                url: manifest_path.to_string_lossy().into_owned(),
                message: error.to_string(),
            })?;
        manifest.validate()?;
        Ok(manifest)
    }

    fn fetch(
        &self,
        reference: &ModelRef,
        file: &ModelFile,
        destination: &Path,
        _context: &SourceContext<'_>,
    ) -> Result<(), HubError> {
        let source = self
            .local_file(reference, file)
            .ok_or_else(|| HubError::Unsupported("local model path is not readable".to_owned()))?;
        download::copy_local(
            &source,
            destination,
            Some(file.size),
            file.sha256.as_deref(),
        )
    }

    fn local_file(&self, reference: &ModelRef, file: &ModelFile) -> Option<PathBuf> {
        let (_, base) = manifest_path_and_dir(&reference.location).ok()?;
        resolve_local_file(&base, file)
    }

    fn local_directory(&self, reference: &ModelRef) -> Option<PathBuf> {
        manifest_path_and_dir(&reference.location)
            .ok()
            .map(|(_, directory)| directory)
    }
}

fn manifest_path_and_dir(location: &str) -> Result<(PathBuf, PathBuf), HubError> {
    let path = PathBuf::from(location);
    if path.is_dir() {
        return Ok((path.join("model.json"), path));
    }
    if path.is_file() {
        let directory = path
            .parent()
            .map(Path::to_path_buf)
            .unwrap_or_else(|| PathBuf::from("."));
        return Ok((path, directory));
    }
    Err(HubError::io(
        &path,
        std::io::Error::new(std::io::ErrorKind::NotFound, "model path does not exist"),
    ))
}

fn resolve_local_file(base: &Path, file: &ModelFile) -> Option<PathBuf> {
    if file.url.trim().is_empty() {
        return Some(base.join(&file.path));
    }
    if let Some(rest) = file.url.strip_prefix("file://") {
        return file_url_to_path(rest).ok();
    }
    let path = PathBuf::from(&file.url);
    if path.is_absolute() {
        Some(path)
    } else {
        Some(base.join(path))
    }
}
