use serde::Deserialize;
use serde_json::Value;
use std::fs;
use std::path::{Path, PathBuf};
use tokenizers::Tokenizer;

pub(crate) struct LayaTokenizer {
    inner: Tokenizer,
    pub pad_id: u32,
    pub cls_id: u32,
    pub sep_id: u32,
    pub mask_id: u32,
}

#[derive(Default, Deserialize)]
struct TokenizerConfig {
    #[serde(default)]
    cls_token: Option<Value>,
    #[serde(default)]
    sep_token: Option<Value>,
    #[serde(default)]
    pad_token: Option<Value>,
    #[serde(default)]
    mask_token: Option<Value>,
}

impl LayaTokenizer {
    pub fn load(directory: &Path) -> Result<Self, String> {
        let path = tokenizer_path(directory)?;
        let inner = Tokenizer::from_file(&path)
            .map_err(|error| format!("failed to load tokenizer {}: {error}", path.display()))?;
        let config = read_config(&path);

        let pad_id = resolve_special(
            &inner,
            explicit(&config.pad_token),
            &["[PAD]", "<pad>"],
            "pad",
        )?;
        let cls_id = resolve_special(
            &inner,
            explicit(&config.cls_token),
            &["[CLS]", "<bos>", "<s>"],
            "cls",
        )?;
        let sep_id = resolve_special(
            &inner,
            explicit(&config.sep_token),
            &["[SEP]", "<eos>", "</s>"],
            "sep",
        )?;
        let mask_id = resolve_special(
            &inner,
            explicit(&config.mask_token),
            &["[MASK]", "<mask>"],
            "mask",
        )?;

        Ok(Self {
            inner,
            pad_id,
            cls_id,
            sep_id,
            mask_id,
        })
    }

    pub fn encode(&self, text: &str) -> Result<Vec<u32>, String> {
        self.inner
            .encode(text, false)
            .map(|encoding| encoding.get_ids().to_vec())
            .map_err(|error| error.to_string())
    }
}

fn resolve_special(
    tokenizer: &Tokenizer,
    explicit: Option<String>,
    candidates: &[&str],
    label: &str,
) -> Result<u32, String> {
    if let Some(token) = explicit
        && let Some(id) = tokenizer.token_to_id(&token)
    {
        return Ok(id);
    }
    candidates
        .iter()
        .find_map(|token| tokenizer.token_to_id(token))
        .ok_or_else(|| format!("tokenizer is missing a {label} token"))
}

fn read_config(tokenizer_path: &Path) -> TokenizerConfig {
    let candidates = [
        tokenizer_path
            .parent()
            .map(|parent| parent.join("tokenizer_config.json")),
        tokenizer_path
            .parent()
            .and_then(Path::parent)
            .map(|parent| parent.join("tokenizer_config.json")),
    ];
    for path in candidates.into_iter().flatten() {
        if path.is_file()
            && let Ok(bytes) = fs::read(&path)
            && let Ok(config) = serde_json::from_slice(&bytes)
        {
            return config;
        }
    }
    TokenizerConfig::default()
}

fn explicit(value: &Option<Value>) -> Option<String> {
    match value {
        Some(Value::String(value)) => Some(value.clone()),
        Some(Value::Object(map)) => map
            .get("content")
            .and_then(Value::as_str)
            .map(str::to_owned),
        _ => None,
    }
}

fn tokenizer_path(directory: &Path) -> Result<PathBuf, String> {
    let candidates = [
        directory.join("tokenizer").join("tokenizer.json"),
        directory.join("tokenizer").join("tokenizer.v1.json"),
        directory.join("tokenizer.json"),
    ];
    candidates
        .into_iter()
        .find(|path| path.is_file())
        .ok_or_else(|| {
            format!(
                "no tokenizer.json found under {} or its tokenizer/ subdirectory",
                directory.display()
            )
        })
}
