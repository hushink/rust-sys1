use serde::Serialize;

/// Result of a safe incremental config reload.
#[derive(Clone, Debug, Serialize)]
pub struct ConfigReload {
    pub config_path: Option<String>,
    pub added: Vec<String>,
    pub removed: Vec<String>,
    pub unchanged: Vec<String>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ReloadError {
    /// Changes touch settings that cannot be applied to a running process.
    RestartRequired(Vec<String>),
    /// Parsing, I/O, or model resolution failed. No changes were applied.
    Failed(String),
}

/// Embedded reload hook for `POST /api/config/reload`.
///
/// Implementations decide which parts of the config can be applied safely.
/// The CLI applies `[models]` additions/removals and rejects changes to hub or
/// server settings; other embedders can use their own policy.
pub trait ConfigReloader: Send + Sync {
    fn reload(&self) -> Result<ConfigReload, ReloadError>;
}
