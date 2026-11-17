use crate::calibration::bucket;
use serde::Deserialize;
use std::collections::HashMap;
use std::fs;
use std::path::Path;

#[derive(Clone, Debug, Deserialize)]
pub(crate) struct LayaConfig {
    pub max_len: usize,
    pub head_max_len: usize,
    pub temperature: Vec<f32>,
    #[serde(default)]
    pub temperature_by_options: HashMap<String, f32>,
    #[serde(default = "default_head_layers")]
    pub head_layers: usize,
}

fn default_head_layers() -> usize {
    2
}

impl LayaConfig {
    pub fn load(path: &Path) -> Result<Self, String> {
        let bytes = fs::read(path)
            .map_err(|error| format!("failed to read {}: {error}", path.display()))?;
        let config: Self = serde_json::from_slice(&bytes)
            .map_err(|error| format!("failed to parse {}: {error}", path.display()))?;
        if config.temperature.len() != 3 {
            return Err(format!(
                "{} must contain exactly three temperature values",
                path.display()
            ));
        }
        if config.head_layers == 0 {
            return Err("head_layers must be greater than zero".to_owned());
        }
        Ok(config)
    }

    pub fn temperature_for(&self, kind: usize, count: usize) -> f32 {
        let bucket = bucket(kind, count);
        let temperature = self
            .temperature_by_options
            .get(&bucket)
            .copied()
            .unwrap_or_else(|| self.temperature.get(kind).copied().unwrap_or(1.0));
        if temperature.is_finite() {
            temperature.clamp(0.5, 5.0)
        } else {
            1.0
        }
    }
}
