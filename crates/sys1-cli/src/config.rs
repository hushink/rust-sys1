use anyhow::{Context, bail};
use indexmap::IndexMap;
use serde::Deserialize;
use std::fs;
use std::net::SocketAddr;
use std::path::{Path, PathBuf};
use sys1_families::{
    DecimaFamilyOptions, DecisionFamilyOptions, FamilyOptions, KevFamilyOptions, MaxLenOptions,
};

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
    #[serde(default)]
    pub families: FamiliesSection,
    /// `name = "local:/path"`, `hf:owner/repo`, or any supported `ModelRef`.
    /// A table form adds per-model family overrides:
    ///
    /// ```toml
    /// [models.nli-small]
    /// reference = "hf:owner/nli"
    /// [models.nli-small.families]
    /// max_batch = 8
    /// ```
    #[serde(default)]
    pub models: IndexMap<String, ModelEntry>,
}

/// A `[models]` entry: either a bare reference string or a table with
/// per-model settings.
#[derive(Clone, Debug, PartialEq, Deserialize)]
#[serde(untagged)]
pub enum ModelEntry {
    Reference(String),
    Detailed(ModelDetail),
}

#[derive(Clone, Debug, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ModelDetail {
    pub reference: String,
    #[serde(default)]
    pub families: FamiliesSection,
}

impl ModelEntry {
    pub fn reference(&self) -> &str {
        match self {
            Self::Reference(reference) => reference,
            Self::Detailed(detail) => &detail.reference,
        }
    }

    /// Merge this entry's overrides onto the global family options.
    pub fn families(&self, global: &FamilyOptions) -> FamilyOptions {
        match self {
            Self::Reference(_) => global.clone(),
            Self::Detailed(detail) => merge_families(global, &detail.families),
        }
    }
}

fn merge_families(global: &FamilyOptions, overrides: &FamiliesSection) -> FamilyOptions {
    let overrides = overrides.to_options();
    FamilyOptions {
        max_batch: overrides.max_batch.or(global.max_batch),
        qwen3guard: MaxLenOptions {
            max_len: overrides.qwen3guard.max_len.or(global.qwen3guard.max_len),
        },
        von: MaxLenOptions {
            max_len: overrides.von.max_len.or(global.von.max_len),
        },
        clef: MaxLenOptions {
            max_len: overrides.clef.max_len.or(global.clef.max_len),
        },
        decima: DecimaFamilyOptions {
            choice_fit: overrides.decima.choice_fit.or(global.decima.choice_fit),
        },
        kev: KevFamilyOptions {
            base_dir: overrides
                .kev
                .base_dir
                .or_else(|| global.kev.base_dir.clone()),
            temperature: overrides.kev.temperature.or(global.kev.temperature),
        },
        decision: DecisionFamilyOptions {
            temperature: overrides
                .decision
                .temperature
                .or(global.decision.temperature),
        },
    }
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
    /// Maximum time per download attempt. Large files resume from the partial
    /// file, so this bounds stalls rather than total transfer time.
    pub download_timeout_secs: Option<u64>,
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
    pub laya_auto_task_detection: Option<bool>,
    pub laya_router_default: Option<String>,
    pub laya_lang_guess: Option<String>,
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

/// Per-family settings for the candle model families.
#[derive(Clone, Debug, Default, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FamiliesSection {
    /// Cross-request batch size for encoder families (default 32).
    pub max_batch: Option<usize>,
    #[serde(default)]
    pub qwen3guard: MaxLenSection,
    #[serde(default)]
    pub von: MaxLenSection,
    #[serde(default)]
    pub clef: MaxLenSection,
    #[serde(default)]
    pub decima: DecimaSection,
    #[serde(default)]
    pub kev: KevSection,
    #[serde(default)]
    pub decision: DecisionSection,
}

#[derive(Clone, Debug, Default, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MaxLenSection {
    pub max_len: Option<usize>,
}

#[derive(Clone, Debug, Default, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DecimaSection {
    /// Override `choice_fit` from `decima.json`.
    pub choice_fit: Option<bool>,
}

#[derive(Clone, Debug, Default, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct KevSection {
    /// Qwen3.5 base directory. Relative paths resolve from this config file.
    pub base_dir: Option<PathBuf>,
    /// Pointer-logit temperature; defaults to `kev.json` or `head.pt`.
    pub temperature: Option<f32>,
}

#[derive(Clone, Debug, Default, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DecisionSection {
    /// Endpoint-logit temperature; defaults to the checkpoint calibration.
    pub temperature: Option<f32>,
}

impl FamiliesSection {
    pub fn to_options(&self) -> FamilyOptions {
        FamilyOptions {
            max_batch: self.max_batch,
            qwen3guard: MaxLenOptions {
                max_len: self.qwen3guard.max_len,
            },
            von: MaxLenOptions {
                max_len: self.von.max_len,
            },
            clef: MaxLenOptions {
                max_len: self.clef.max_len,
            },
            decima: DecimaFamilyOptions {
                choice_fit: self.decima.choice_fit,
            },
            kev: KevFamilyOptions {
                base_dir: self.kev.base_dir.clone(),
                temperature: self.kev.temperature,
            },
            decision: DecisionFamilyOptions {
                temperature: self.decision.temperature,
            },
        }
    }
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
            .map(|(name, entry)| format!("{name}={}", entry.reference()))
            .collect()
    }

    /// Let `sys1 run laya-latest` use `[models].laya-latest` from the config.
    pub fn model_reference(&self, reference: &str) -> String {
        self.value
            .models
            .get(reference)
            .map(|entry| entry.reference().to_owned())
            .unwrap_or_else(|| reference.to_owned())
    }

    /// Global family options with the entry's per-model overrides applied.
    pub fn families_for(&self, name: &str, global: &FamilyOptions) -> FamilyOptions {
        self.value
            .models
            .get(name)
            .map(|entry| entry.families(global))
            .unwrap_or_else(|| global.clone())
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
    if let Some(base) = value.families.kev.base_dir.take() {
        value.families.kev.base_dir = Some(resolve_path(&base_dir, &base));
    }
    for entry in value.models.values_mut() {
        match entry {
            ModelEntry::Reference(reference) => {
                *reference = resolve_reference(&base_dir, reference);
            }
            ModelEntry::Detailed(detail) => {
                detail.reference = resolve_reference(&base_dir, &detail.reference);
                if let Some(base) = detail.families.kev.base_dir.take() {
                    detail.families.kev.base_dir = Some(resolve_path(&base_dir, &base));
                }
            }
        }
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
            timeout_secs = 30
            download_timeout_secs = 600

            [server]
            bind = "0.0.0.0:8080"
            batch_max = 64
            batch_wait_ms = 1
            local_window = true
            metrics = false
            laya_auto_task_detection = true
            laya_router_default = "typed-decisions"
            laya_lang_guess = "en"
            preload = ["laya-latest"]
            log_level = "debug"
            log_format = "json"
            log_file = "logs/sys1.log"
            cors_origins = ["https://app.example"]

            [families]
            max_batch = 16

            [families.qwen3guard]
            max_len = 32768

            [families.von]
            max_len = 4096

            [families.clef]
            max_len = 8192

            [families.decima]
            choice_fit = true

            [families.kev]
            base_dir = "D:/models/qwen-base"
            temperature = 2.351

            [families.decision]
            temperature = 1.0

            [models]
            laya-latest = "local:D:/models/laya"

            [models.nli-small]
            reference = "hf:owner/nli"
            [models.nli-small.families]
            max_batch = 8
            [models.nli-small.families.decima]
            choice_fit = false
        "#;
        let config: FileConfig = toml::from_str(text).expect("config");
        assert_eq!(config.hub.max_cache_bytes, Some(34_359_738_368));
        assert_eq!(config.hub.timeout_secs, Some(30));
        assert_eq!(config.hub.download_timeout_secs, Some(600));
        assert_eq!(config.server.batch_max, Some(64));
        assert_eq!(config.server.metrics, Some(false));
        assert_eq!(config.server.laya_auto_task_detection, Some(true));
        assert_eq!(
            config.server.laya_router_default.as_deref(),
            Some("typed-decisions")
        );
        assert_eq!(config.server.laya_lang_guess.as_deref(), Some("en"));
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
            config
                .models
                .get("laya-latest")
                .map(|entry| entry.reference()),
            Some("local:D:/models/laya")
        );
        assert_eq!(
            config
                .models
                .get("nli-small")
                .map(|entry| entry.reference()),
            Some("hf:owner/nli")
        );

        let families = config.families.to_options();
        assert_eq!(families.max_batch, Some(16));
        assert_eq!(families.qwen3guard.max_len, Some(32_768));
        assert_eq!(families.von.max_len, Some(4096));
        assert_eq!(families.clef.max_len, Some(8192));
        assert_eq!(families.decima.choice_fit, Some(true));
        assert_eq!(
            families.kev.base_dir.as_deref(),
            Some(Path::new("D:/models/qwen-base"))
        );
        assert_eq!(families.kev.temperature, Some(2.351));
        assert_eq!(families.decision.temperature, Some(1.0));

        let merged = config.models["nli-small"].families(&families);
        assert_eq!(merged.max_batch, Some(8));
        assert_eq!(merged.decima.choice_fit, Some(false));
        assert_eq!(merged.qwen3guard.max_len, Some(32_768));
        assert_eq!(merged.kev.temperature, Some(2.351));
        assert_eq!(merged.decision.temperature, Some(1.0));
    }

    #[test]
    fn resolves_relative_kev_base() {
        let base = Path::new("C:/models/config");
        let mut config = FamiliesSection::default();
        config.kev.base_dir = Some(PathBuf::from("qwen-base"));
        let resolved = resolve_path(base, config.kev.base_dir.as_deref().unwrap());
        assert_eq!(resolved, PathBuf::from("C:/models/config/qwen-base"));
    }
}
