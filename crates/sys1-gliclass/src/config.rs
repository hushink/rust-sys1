use serde::Deserialize;
use std::fs;
use std::path::Path;
use sys1_modernbert::ModernBertConfig;

fn default_max_len() -> usize {
    1024
}

#[derive(Deserialize)]
struct GliclassFile {
    encoder_config: ModernBertConfig,
    #[serde(default)]
    architecture_type: String,
    class_token_index: u32,
    text_token_index: u32,
    #[serde(default)]
    #[allow(dead_code)]
    example_token_index: u32,
    class_token_pooling: String,
    #[serde(default)]
    use_segment_embeddings: bool,
    #[serde(default)]
    prompt_first: bool,
    pad_token_id: u32,
    #[serde(default)]
    hidden_size: Option<usize>,
    #[serde(default)]
    scorer_type: String,
    #[serde(default)]
    use_calibrator: bool,
    #[serde(default)]
    normalize_features: bool,
    #[serde(default)]
    use_lstm: bool,
    #[serde(default)]
    max_len: Option<usize>,
}

/// Configuration for the ModernBERT-backed GLiClass uni-encoder.
#[derive(Clone, Debug)]
pub struct GliclassConfig {
    pub encoder: ModernBertConfig,
    pub class_token_index: u32,
    pub text_token_index: u32,
    pub pad: u32,
    pub max_len: usize,
    pub hidden_size: usize,
}

impl GliclassConfig {
    pub fn load(directory: &Path) -> Result<Self, String> {
        let path = directory.join("config.json");
        let bytes = fs::read(&path).map_err(|error| format!("read {}: {error}", path.display()))?;
        let file: GliclassFile = serde_json::from_slice(&bytes)
            .map_err(|error| format!("parse {}: {error}", path.display()))?;
        if file.architecture_type != "uni-encoder" {
            return Err(format!(
                "unsupported GLiClass architecture_type {:?}; only uni-encoder is implemented",
                file.architecture_type
            ));
        }
        if file.class_token_pooling != "average" {
            return Err(format!(
                "unsupported class_token_pooling {:?}; only average is implemented",
                file.class_token_pooling
            ));
        }
        if !file.use_segment_embeddings || !file.prompt_first {
            return Err(
                "unsupported GLiClass layout: use_segment_embeddings and prompt_first must be true"
                    .to_owned(),
            );
        }
        if file.scorer_type != "mlp" {
            return Err(format!(
                "unsupported scorer_type {:?}; only mlp is implemented",
                file.scorer_type
            ));
        }
        if file.use_calibrator {
            return Err("GLiClass calibrator is not implemented".to_owned());
        }
        if file.normalize_features || file.use_lstm {
            return Err("GLiClass normalize_features/use_lstm are not implemented".to_owned());
        }
        let hidden_size = file.hidden_size.unwrap_or(file.encoder_config.hidden_size);
        if hidden_size != file.encoder_config.hidden_size {
            return Err(format!(
                "GLiClass projector width {hidden_size} differs from encoder hidden_size {}",
                file.encoder_config.hidden_size
            ));
        }
        let max_len = file.max_len.unwrap_or_else(default_max_len);
        if max_len < 8 || max_len > file.encoder_config.max_position_embeddings {
            return Err(format!(
                "max_len {max_len} is outside 8..={}",
                file.encoder_config.max_position_embeddings
            ));
        }
        Ok(Self {
            encoder: file.encoder_config,
            class_token_index: file.class_token_index,
            text_token_index: file.text_token_index,
            pad: file.pad_token_id,
            max_len,
            hidden_size,
        })
    }
}
