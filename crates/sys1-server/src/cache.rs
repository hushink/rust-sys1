use serde::Serialize;

/// A cached model on disk, as reported by `GET /api/cache`.
#[derive(Clone, Debug, Serialize)]
pub struct CachedModel {
    /// Stable cache key (the model directory name under `models/`).
    pub key: String,
    pub name: String,
    pub directory: String,
    pub size_bytes: u64,
    /// Last materialization time in Unix seconds; `0` when unknown.
    pub last_used_unix: u64,
    /// `false` when the model is resident or a pull/removal is active.
    pub evictable: bool,
    pub files: Vec<CachedFile>,
}

#[derive(Clone, Debug, Serialize)]
pub struct CachedFile {
    pub path: String,
    pub size: u64,
}

/// Error returned by [`CacheManager::remove`].
#[derive(Clone, Debug)]
pub enum CacheRemoveError {
    /// The entry cannot be removed right now (resident model or active pull).
    Conflict(String),
    /// Any other filesystem or manager error.
    Other(String),
}

impl CacheRemoveError {
    pub fn conflict(message: impl Into<String>) -> Self {
        Self::Conflict(message.into())
    }

    pub fn other(message: impl Into<String>) -> Self {
        Self::Other(message.into())
    }

    pub fn message(&self) -> &str {
        match self {
            Self::Conflict(message) | Self::Other(message) => message,
        }
    }
}

impl std::fmt::Display for CacheRemoveError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(self.message())
    }
}

impl std::error::Error for CacheRemoveError {}

/// Aggregate disk-cache usage reported by `GET /api/cache`.
#[derive(Clone, Copy, Debug, Default, Serialize)]
pub struct CacheUsage {
    /// Configured quota, or `null` when unlimited.
    pub limit_bytes: Option<u64>,
    pub used_bytes: u64,
    /// `limit_bytes - used_bytes`, or `null` when unlimited.
    pub free_bytes: Option<u64>,
    /// Bytes that could be freed by evicting non-resident entries.
    pub reclaimable_bytes: u64,
    pub model_count: usize,
    pub resident_count: usize,
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

    /// Aggregate usage and quota headroom. The default derives totals from
    /// [`CacheManager::list`]; managers that track partial downloads or a
    /// quota should override it.
    fn usage(&self) -> Result<CacheUsage, String> {
        let models = self.list()?;
        let used_bytes = models.iter().map(|model| model.size_bytes).sum();
        let reclaimable_bytes = models
            .iter()
            .filter(|model| model.evictable)
            .map(|model| model.size_bytes)
            .sum();
        Ok(CacheUsage {
            limit_bytes: None,
            used_bytes,
            free_bytes: None,
            reclaimable_bytes,
            model_count: models.len(),
            resident_count: models.iter().filter(|model| !model.evictable).count(),
        })
    }

    /// Remove one cache key. Returns whether it existed.
    ///
    /// Return [`CacheRemoveError::Conflict`] for a resident model or an active
    /// pull; the HTTP layer maps that to `409`.
    fn remove(&self, key: &str) -> Result<bool, CacheRemoveError>;
}
