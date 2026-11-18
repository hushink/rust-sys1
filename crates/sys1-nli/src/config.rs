use crate::deberta::DebertaConfig;
use serde::Deserialize;
use std::fs;
use std::path::Path;
use sys1_modernbert::ModernBertConfig;

fn default_max_len() -> usize {
    512
}

/// Hypothesis templates for the ModernBERT zeroshot v2.0 checkpoints.
#[derive(Clone, Debug, Deserialize)]
pub struct NliTemplates {
    pub choice_with_description: String,
    pub choice_without_description: String,
    pub score: String,
    pub noul_pair: String,
    pub noul_single: String,
}

impl Default for NliTemplates {
    fn default() -> Self {
        Self::modernbert()
    }
}

impl NliTemplates {
    /// The canonical ModernBERT zeroshot-v2.0 templates.
    pub fn modernbert() -> Self {
        Self {
            choice_with_description: "{description}".to_owned(),
            choice_without_description: "This text is about {label}.".to_owned(),
            score: "{description}".to_owned(),
            noul_pair: "{description}".to_owned(),
            noul_single: "{instructions}".to_owned(),
        }
    }

    /// The DeBERTa-v3 zeroshot-v2.0 templates.
    pub fn deberta() -> Self {
        let answer = "The answer to \"{instructions}\" is: ";
        Self {
            choice_with_description: format!("{answer}{{option}}"),
            choice_without_description: format!("{answer}{{option}}"),
            score: format!("{answer}{{description}}"),
            noul_pair: "{description}".to_owned(),
            noul_single: "{instructions}".to_owned(),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Pooling {
    Mean,
    Cls,
}

#[derive(Deserialize)]
struct NliFile {
    #[serde(flatten)]
    encoder: ModernBertConfig,
    #[serde(default = "default_max_len")]
    max_len: usize,
    #[serde(default)]
    classifier_pooling: Option<String>,
    cls_token_id: Option<u32>,
    sep_token_id: Option<u32>,
    pad_token_id: Option<u32>,
}

/// The encoder behind an NLI checkpoint.
#[derive(Clone, Debug)]
pub enum NliEncoderConfig {
    Modern(ModernBertConfig),
    Deberta(Box<DebertaConfig>),
}

impl NliEncoderConfig {
    pub fn hidden_size(&self) -> usize {
        match self {
            Self::Modern(config) => config.hidden_size,
            Self::Deberta(config) => config.hidden_size(),
        }
    }

    pub fn max_position_embeddings(&self) -> usize {
        match self {
            Self::Modern(config) => config.max_position_embeddings(),
            Self::Deberta(config) => config.max_position_embeddings(),
        }
    }

    pub fn layer_norm_eps(&self) -> f64 {
        match self {
            Self::Modern(config) => config.layer_norm_eps(),
            Self::Deberta(config) => config.layer_norm_eps(),
        }
    }

    pub fn is_deberta(&self) -> bool {
        matches!(self, Self::Deberta(_))
    }
}

/// The NLI model configuration: encoder plus pair-classification settings.
#[derive(Clone, Debug)]
pub struct NliConfig {
    pub encoder: NliEncoderConfig,
    pub max_len: usize,
    pub cls: u32,
    pub sep: u32,
    pub pad: u32,
    pub entailment: usize,
    pub not_entailment: usize,
    pub pooling: Pooling,
    pub templates: NliTemplates,
}

impl NliConfig {
    pub fn load(directory: &Path) -> Result<Self, String> {
        let path = directory.join("config.json");
        let bytes = fs::read(&path).map_err(|error| format!("read {}: {error}", path.display()))?;
        let value: serde_json::Value = serde_json::from_slice(&bytes)
            .map_err(|error| format!("parse {}: {error}", path.display()))?;
        let (entailment, not_entailment) = labels(&value)?;
        let model_type = value
            .get("model_type")
            .and_then(serde_json::Value::as_str)
            .unwrap_or_default();

        if model_type == "deberta-v2" {
            let config: DebertaConfig = serde_json::from_value(value.clone())
                .map_err(|error| format!("parse DeBERTa config: {error}"))?;
            config.validate().map_err(|error| error.to_string())?;
            let max_len = value
                .get("max_len")
                .and_then(serde_json::Value::as_u64)
                .map(|length| length as usize)
                .unwrap_or(512);
            if max_len < 8 || max_len > config.max_position_embeddings() {
                return Err(format!(
                    "max_len {max_len} is outside 8..={}",
                    config.max_position_embeddings()
                ));
            }
            let pad = config.pad_token_id.unwrap_or(0);
            return Ok(Self {
                encoder: NliEncoderConfig::Deberta(Box::new(config)),
                max_len,
                cls: 1,
                sep: 2,
                pad,
                entailment,
                not_entailment,
                pooling: Pooling::Cls,
                templates: NliTemplates::deberta(),
            });
        }

        let file: NliFile = serde_json::from_value(value)
            .map_err(|error| format!("parse {}: {error}", path.display()))?;
        let cls = file
            .cls_token_id
            .ok_or_else(|| "config.json has no cls_token_id".to_owned())?;
        let sep = file
            .sep_token_id
            .ok_or_else(|| "config.json has no sep_token_id".to_owned())?;
        let pad = file
            .pad_token_id
            .ok_or_else(|| "config.json has no pad_token_id".to_owned())?;
        let pooling = match file.classifier_pooling.as_deref() {
            None | Some("mean") => Pooling::Mean,
            Some("cls") => Pooling::Cls,
            Some(other) => return Err(format!("unsupported classifier_pooling {other:?}")),
        };
        if file.max_len < 8 {
            return Err("config max_len must be at least 8".to_owned());
        }

        Ok(Self {
            encoder: NliEncoderConfig::Modern(file.encoder),
            max_len: file.max_len,
            cls,
            sep,
            pad,
            entailment,
            not_entailment,
            pooling,
            templates: NliTemplates::modernbert(),
        })
    }
}

fn labels(value: &serde_json::Value) -> Result<(usize, usize), String> {
    let mut entailment = 0;
    let mut not_entailment = 1;
    if let Some(map) = value.get("label2id").and_then(serde_json::Value::as_object) {
        for (label, index) in map {
            let index = index
                .as_u64()
                .ok_or_else(|| format!("label2id[{label:?}] is not an index"))?
                as usize;
            match label.as_str() {
                "entailment" => entailment = index,
                "not_entailment" => not_entailment = index,
                _ => {}
            }
        }
    }
    if let Some(map) = value.get("id2label").and_then(serde_json::Value::as_object) {
        for (index, label) in map {
            let index: usize = index
                .parse()
                .map_err(|_| format!("config.json id2label key {index:?} is not an index"))?;
            match label.as_str() {
                Some("entailment") => entailment = index,
                Some("not_entailment") => not_entailment = index,
                _ => {}
            }
        }
    }
    if entailment == not_entailment {
        return Err("config.json maps entailment and not_entailment to the same class".to_owned());
    }
    Ok((entailment, not_entailment))
}
