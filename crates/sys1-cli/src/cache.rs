use std::path::Path;
use std::sync::Arc;
use std::time::UNIX_EPOCH;
use sys1_hub::Hub;
use sys1_server::{CacheManager, CacheRemoveError, CacheUsage, CachedFile, CachedModel};

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
        let entries = self
            .hub
            .cache_entries()
            .map_err(|error| error.to_string())?;
        let mut models = entries
            .into_iter()
            .map(|entry| CachedModel {
                key: entry.key,
                name: entry.name,
                directory: display_path(&entry.directory),
                size_bytes: entry.size_bytes,
                last_used_unix: entry
                    .last_used
                    .duration_since(UNIX_EPOCH)
                    .map(|duration| duration.as_secs())
                    .unwrap_or(0),
                evictable: entry.evictable,
                files: entry
                    .files
                    .into_iter()
                    .map(|file| CachedFile {
                        path: file.path,
                        size: file.size,
                    })
                    .collect(),
            })
            .collect::<Vec<_>>();
        models.sort_by(|left, right| left.name.cmp(&right.name));
        Ok(models)
    }

    fn usage(&self) -> Result<CacheUsage, String> {
        let usage = self.hub.cache_usage().map_err(|error| error.to_string())?;
        Ok(CacheUsage {
            limit_bytes: usage.limit_bytes,
            used_bytes: usage.used_bytes,
            free_bytes: usage.free_bytes,
            reclaimable_bytes: usage.reclaimable_bytes,
            model_count: usage.model_count,
            resident_count: usage.resident_count,
        })
    }

    fn remove(&self, key: &str) -> Result<bool, CacheRemoveError> {
        match self.hub.remove_cached(key) {
            Ok(removed) => Ok(removed),
            Err(
                error @ (sys1_hub::HubError::CacheBusy { .. }
                | sys1_hub::HubError::CacheResident { .. }),
            ) => Err(CacheRemoveError::conflict(error.to_string())),
            Err(error) => Err(CacheRemoveError::other(error.to_string())),
        }
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
