use std::path::Path;
use std::sync::Arc;
use sys1_hub::Hub;
use sys1_server::{CacheManager, CachedFile, CachedModel};

/// Exposes the hub's on-disk model cache through the server admin API.
pub struct HubCacheManager {
    hub: Arc<Hub>,
}

impl HubCacheManager {
    pub fn new(hub: Arc<Hub>) -> Self {
        Self { hub }
    }
}

impl CacheManager for HubCacheManager {
    fn root(&self) -> String {
        display_path(&self.hub.config().cache_dir)
    }

    fn list(&self) -> Result<Vec<CachedModel>, String> {
        let models = self.hub.list_cached().map_err(|error| error.to_string())?;
        Ok(models
            .into_iter()
            .map(|pulled| {
                let key = pulled
                    .directory
                    .file_name()
                    .and_then(|name| name.to_str())
                    .unwrap_or_default()
                    .to_owned();
                let files = pulled
                    .manifest
                    .files
                    .iter()
                    .map(|file| CachedFile {
                        path: file.path.clone(),
                        size: file.size,
                    })
                    .collect::<Vec<_>>();
                let size_bytes = files.iter().map(|file| file.size).sum();
                CachedModel {
                    key,
                    name: pulled.manifest.name,
                    directory: display_path(&pulled.directory),
                    size_bytes,
                    files,
                }
            })
            .collect())
    }

    fn remove(&self, key: &str) -> Result<bool, String> {
        self.hub
            .remove_cached(key)
            .map_err(|error| error.to_string())
    }
}

fn display_path(path: &Path) -> String {
    let text = path.display().to_string();
    if let Some(rest) = text.strip_prefix(r"\\?\UNC\") {
        format!(r"\\{rest}")
    } else if let Some(rest) = text.strip_prefix(r"\\?\") {
        rest.to_owned()
    } else {
        text
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strips_windows_verbatim_prefixes() {
        assert_eq!(display_path(Path::new(r"\\?\D:\models")), r"D:\models");
        assert_eq!(
            display_path(Path::new(r"\\?\UNC\server\share")),
            r"\\server\share"
        );
        assert_eq!(display_path(Path::new("/models")), "/models");
    }
}
