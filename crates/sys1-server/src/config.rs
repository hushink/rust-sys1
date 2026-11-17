use std::time::Duration;

/// Cross-request batching settings.
#[derive(Clone, Debug)]
pub struct BatchConfig {
    /// Maximum number of requests in one model forward pass.
    pub max_batch: usize,
    /// How long to wait for additional requests before running a batch.
    pub max_wait: Duration,
}

impl Default for BatchConfig {
    fn default() -> Self {
        Self {
            max_batch: 32,
            max_wait: Duration::from_millis(2),
        }
    }
}

/// HTTP server configuration.
#[derive(Clone, Debug)]
pub struct ServerConfig {
    /// When set, `/v1/*` requires `Authorization: Bearer <api_key>`.
    pub api_key: Option<String>,
    /// Maximum compressed and decompressed request body size.
    pub max_body_bytes: usize,
    /// Cross-request batching. `None` disables it.
    pub batching: Option<BatchConfig>,
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            api_key: None,
            max_body_bytes: 32 * 1024 * 1024,
            batching: Some(BatchConfig::default()),
        }
    }
}
