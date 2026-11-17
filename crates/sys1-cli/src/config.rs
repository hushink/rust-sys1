use anyhow::{Context, bail};
use indexmap::IndexMap;
use serde::Deserialize;
use std::fs;
use std::net::SocketAddr;
use std::path::{Path, PathBuf};

const CONFIG_FILE: &str = "sys1.toml";

/// The optional `sys1.toml` file. Every field is optional; CLI arguments take
/// precedence over this file.
#[derive(Clone, Debug, Default, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FileConfig {
    #[serde(default)]
    pub hub: HubSection,
    #[serde(default)]
    pub server: ServerSection,
    /// `name = "local:/path"`, `hf:owner/repo`, or any supported `ModelRef`.
    #[serde(default)]
    pub models: IndexMap<String, String>,
}

#[derive(Clone, Debug, Default, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HubSection {
    /// Model download/cache directory. Relative paths resolve from the config file.
    #[serde(alias = "cache")]
    pub cache_dir: Option<PathBuf>,
    /// Maximum bytes kept under the model cache. Omit or use `0` for unlimited.
    pub max_cache_bytes: Option<u64>,
    pub offline: Option<bool>,
    pub hf_endpoint: Option<String>,
    pub modelscope_endpoint: Option<String>,
    pub hf_token: Option<String>,
    pub modelscope_token: Option<String>,
    pub http_token: Option<String>,
    pub timeout_secs: Option<u64>,
}

#[derive(Clone, Debug, Default, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ServerSection {
    pub bind: Option<SocketAddr>,
    pub api_key: Option<String>,
    pub device: Option<String>,
    pub dtype: Option<String>,
    pub max_len: Option<usize>,
    pub laya_confidence: Option<String>,
    pub batch_max: Option<usize>,
    pub batch_wait_ms: Option<u64>,
    pub max_attention_elements: Option<usize>,
    /// `false` disables cross-request batching.
    pub batch: Option<bool>,
    pub tf32: Option<bool>,
    /// `false` uses eager attention for local layers.
    pub local_window: Option<bool>,
    /// `false` disables the `/api/models` management endpoints (default true).
    pub admin: Option<bool>,
    /// `false` disables `/metrics` and `/api/metrics` (default true for the CLI).
    pub metrics: Option<bool>,
    /// Models to load before serving. Names may be aliases from `[models]`.
    #[serde(default)]
    pub preload: Vec<String>,
    /// `tracing` filter, e.g. `info` or `sys1=debug,tower_http=info`.
    pub log_level: Option<String>,
    /// `text` or `json`.
    pub log_format: Option<String>,
    /// Append logs to this file instead of stdout. Relative to this config.
    pub log_file: Option<PathBuf>,
    /// CORS allowlist. Omit for permissive; `[]` disables CORS.
    pub cors_origins: Option<Vec<String>>,
}

#[derive(Clone, Debug, Default)]
pub struct LoadedConfig {
    pub path: Option<PathBuf>,
    pub value: FileConfig,
}

impl LoadedConfig {
    pub fn model_specs(&self) -> Vec<String> {
        self.value
            .models
            .iter()
            .map(|(name, reference)| format!("{name}={reference}"))
            .collect()
    }

    /// Let `sys1 run laya-latest` use `[models].laya-latest` from the config.
    pub fn model_reference(&self, reference: &str) -> String {
        self.value
            .models
            .get(reference)
            .cloned()
            .unwrap_or_else(|| reference.to_owned())
    }
}

/// Load `--config`, `SYS1_CONFIG`, or `./sys1.toml` (in that order).
pub fn load(explicit: Option<&Path>) -> anyhow::Result<LoadedConfig> {
    let explicit = explicit
        .map(Path::to_path_buf)
        .or_else(|| std::env::var_os("SYS1_CONFIG").map(PathBuf::from));
    let path = match explicit {
        Some(path) => {
            if !path.is_file() {
                bail!("config file {} does not exist", path.display());
            }
            Some(path)
        }
        None => {
            let default = PathBuf::from(CONFIG_FILE);
            default.is_file().then_some(default)
        }
    };

    let Some(path) = path else {
        return Ok(LoadedConfig::default());
    };

    let path = fs::canonicalize(&path)
        .with_context(|| format!("resolve config file {}", path.display()))?;
    let base_dir = path
        .parent()
        .map(Path::to_path_buf)
        .unwrap_or_else(|| PathBuf::from("."));
    let text = fs::read_to_string(&path)
        .with_context(|| format!("read config file {}", path.display()))?;
    let mut value: FileConfig =
        toml::from_str(&text).with_context(|| format!("parse config file {}", path.display()))?;

    if let Some(cache) = value.hub.cache_dir.take() {
        value.hub.cache_dir = Some(resolve_path(&base_dir, &cache));
    }
    if let Some(log_file) = value.server.log_file.take() {
        value.server.log_file = Some(resolve_path(&base_dir, &log_file));
    }
    for reference in value.models.values_mut() {
        *reference = resolve_reference(&base_dir, reference);
    }

    Ok(LoadedConfig {
        path: Some(path),
        value,
    })
}

fn resolve_path(base: &Path, path: &Path) -> PathBuf {
    let path = expand_home(path);
    if path.is_absolute() {
        path
    } else {
        base.join(path)
    }
}

fn resolve_reference(base: &Path, reference: &str) -> String {
    let trimmed = reference.trim();
    let Some(location) = trimmed.strip_prefix("local:") else {
        return trimmed.to_owned();
    };
    let path = resolve_path(base, Path::new(location));
    format!("local:{}", path.display())
}

fn expand_home(path: &Path) -> PathBuf {
    let Some(text) = path.to_str() else {
        return path.to_path_buf();
    };
    let Some(rest) = text.strip_prefix("~/").or_else(|| text.strip_prefix("~\\")) else {
        return path.to_path_buf();
    };
    std::env::var_os("USERPROFILE")
        .or_else(|| std::env::var_os("HOME"))
        .map(PathBuf::from)
        .map(|home| home.join(rest))
        .unwrap_or_else(|| path.to_path_buf())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolves_relative_cache_and_local_models() {
        let base = Path::new("C:/models/config");
        let mut models = IndexMap::new();
        models.insert("laya".to_owned(), "local:laya".to_owned());
        models.insert("remote".to_owned(), "hf:owner/repo".to_owned());
        assert_eq!(
            resolve_reference(base, &models["laya"]),
            format!("local:{}", base.join("laya").display())
        );
        assert_eq!(resolve_reference(base, &models["remote"]), "hf:owner/repo");
        assert_eq!(
            resolve_path(base, Path::new("cache")),
            PathBuf::from("C:/models/config/cache")
        );
    }

    #[test]
    fn parses_example_config() {
        let text = r#"
            [hub]
            cache_dir = "D:/models/sys1"
            max_cache_bytes = 34359738368

            [server]
            bind = "0.0.0.0:8080"
            batch_max = 64
            batch_wait_ms = 1
            local_window = true
            metrics = false
            preload = ["laya-latest"]
            log_level = "debug"
            log_format = "json"
            log_file = "logs/sys1.log"
            cors_origins = ["https://app.example"]

            [models]
            laya-latest = "local:D:/models/laya"
        "#;
        let config: FileConfig = toml::from_str(text).expect("config");
        assert_eq!(config.hub.max_cache_bytes, Some(34_359_738_368));
        assert_eq!(config.server.batch_max, Some(64));
        assert_eq!(config.server.metrics, Some(false));
        assert_eq!(config.server.preload, vec!["laya-latest".to_owned()]);
        assert_eq!(config.server.log_level.as_deref(), Some("debug"));
        assert_eq!(config.server.log_format.as_deref(), Some("json"));
        assert_eq!(
            config.server.log_file.as_deref(),
            Some(std::path::Path::new("logs/sys1.log"))
        );
        assert_eq!(
            config.server.cors_origins.as_deref(),
            Some(&["https://app.example".to_owned()][..])
        );
        assert_eq!(
            config.models.get("laya-latest").map(String::as_str),
            Some("local:D:/models/laya")
        );
    }
}
