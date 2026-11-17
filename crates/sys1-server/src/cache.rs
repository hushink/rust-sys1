use serde::Serialize;

/// A cached model on disk, as reported by `GET /api/cache`.
#[derive(Clone, Debug, Serialize)]
pub struct CachedModel {
    /// Stable cache key (the model directory name under `models/`).
    pub key: String,
    pub name: String,
    pub directory: String,
    pub size_bytes: u64,
    pub files: Vec<CachedFile>,
}

#[derive(Clone, Debug, Serialize)]
pub struct CachedFile {
    pub path: String,
    pub size: u64,
}

/// Disk cache management injected by the embedding application.
///
/// `sys1-server` intentionally does not depend on `sys1-hub`; the CLI
/// implements this trait with its hub. Cache routes are registered only when
/// admin endpoints are enabled and a manager is present.
pub trait CacheManager: Send + Sync {
    /// Cache root, for display in `GET /api/cache`.
    fn root(&self) -> String;

    /// Cached models, sorted by name.
    fn list(&self) -> Result<Vec<CachedModel>, String>;

    /// Remove one cache key. Returns whether it existed.
    fn remove(&self, key: &str) -> Result<bool, String>;
}
