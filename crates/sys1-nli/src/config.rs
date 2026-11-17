use serde::Deserialize;
use std::collections::HashMap;
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
        Self {
            choice_with_description: "{description}".to_owned(),
            choice_without_description: "This text is about {label}.".to_owned(),
            score: "{description}".to_owned(),
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
    #[serde(default)]
    label2id: HashMap<String, usize>,
    #[serde(default)]
    id2label: HashMap<String, String>,
}

/// The NLI model configuration: the ModernBERT encoder plus pair-classification
/// settings.
#[derive(Clone, Debug)]
pub struct NliConfig {
    pub encoder: ModernBertConfig,
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
        let file: NliFile = serde_json::from_slice(&bytes)
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

        let mut entailment = 0;
        let mut not_entailment = 1;
        for (label, index) in &file.label2id {
            if label == "entailment" {
                entailment = *index;
            } else if label == "not_entailment" {
                not_entailment = *index;
            }
        }
        for (index, label) in &file.id2label {
            let index: usize = index
                .parse()
                .map_err(|_| format!("config.json id2label key {index:?} is not an index"))?;
            if label == "entailment" {
                entailment = index;
            } else if label == "not_entailment" {
                not_entailment = index;
            }
        }
        if entailment == not_entailment {
            return Err(
                "config.json maps entailment and not_entailment to the same class".to_owned(),
            );
        }

        let pooling = match file.classifier_pooling.as_deref() {
            None | Some("mean") => Pooling::Mean,
            Some("cls") => Pooling::Cls,
            Some(other) => return Err(format!("unsupported classifier_pooling {other:?}")),
        };
        if file.max_len < 8 {
            return Err("config max_len must be at least 8".to_owned());
        }

        Ok(Self {
            encoder: file.encoder,
            max_len: file.max_len,
            cls,
            sep,
            pad,
            entailment,
            not_entailment,
            pooling,
            templates: NliTemplates::default(),
        })
    }
}
