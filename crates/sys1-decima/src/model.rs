use crate::bert::{BertConfig, BertEncoder};
use candle_core::{DType, Device, Tensor};
use candle_nn::{Activation, LayerNorm, Linear, Module, VarBuilder, layer_norm, linear};
use indexmap::IndexMap;
use serde::Deserialize;
use std::path::Path;
use sys1_core::{
    Answers, DecisionModel, DecisionOutput, ModelError, Prediction, Questions, choice_answer,
    python_json, softmax, strip,
};
use sys1_modernbert::{Encoder, ModernBertConfig};
use sys1_protocol::{Answer, Content, NoulCriteria, Question, State, Usage};
use tokenizers::Tokenizer;
use unicode_normalization::UnicodeNormalization;

/// Runtime options for Decima.
#[derive(Clone)]
pub struct DecimaOptions {
    pub device: Device,
    pub dtype: DType,
    pub max_batch: usize,
    /// Override the `choice_fit` setting from `decima.json`.
    pub choice_fit: Option<bool>,
}

impl Default for DecimaOptions {
    fn default() -> Self {
        Self {
            device: Device::Cpu,
            dtype: DType::F32,
            max_batch: 32,
            choice_fit: None,
        }
    }
}

#[derive(Clone, Debug, Deserialize)]
pub struct DecimaConfig {
    #[serde(default)]
    pub backbone: String,
    #[serde(default = "default_scorer_layers")]
    pub scorer_layers: usize,
    #[serde(default = "default_heads")]
    pub heads: usize,
    #[serde(default = "default_ffn")]
    pub ffn: usize,
    #[serde(default)]
    pub max_state_tokens: usize,
    #[serde(default)]
    pub max_choice_tokens: usize,
    #[serde(default = "default_temperature")]
    pub temperature: f64,
    #[serde(default)]
    pub question_in_state: bool,
    #[serde(default = "default_true")]
    pub question_in_choices: bool,
    #[serde(default)]
    pub choice_fit: bool,
    #[serde(default)]
    pub state_prefix: String,
    #[serde(default)]
    pub choice_prefix: String,
    #[serde(default)]
    pub limits: Limits,
}

#[derive(Clone, Debug, Deserialize)]
pub struct Limits {
    #[serde(default = "default_questions_limit")]
    pub questions: [usize; 2],
    #[serde(default = "default_choices_limit")]
    pub choices: [usize; 2],
    #[serde(default = "default_levels_limit")]
    pub levels: [usize; 2],
}

impl Default for Limits {
    fn default() -> Self {
        Self {
            questions: default_questions_limit(),
            choices: default_choices_limit(),
            levels: default_levels_limit(),
        }
    }
}

fn default_scorer_layers() -> usize {
    2
}
fn default_heads() -> usize {
    6
}
fn default_ffn() -> usize {
    1536
}
fn default_temperature() -> f64 {
    1.0
}
fn default_true() -> bool {
    true
}
fn default_questions_limit() -> [usize; 2] {
    [1, 256]
}
fn default_choices_limit() -> [usize; 2] {
    [2, 255]
}
fn default_levels_limit() -> [usize; 2] {
    [2, 10]
}

struct Attention {
    q: Linear,
    k: Linear,
    v: Linear,
    o: Linear,
    heads: usize,
    head_dim: usize,
}

impl Attention {
    fn load(vb: VarBuilder, dim: usize, heads: usize) -> Result<Self, ModelError> {
        let load = |name: &str| -> Result<Linear, ModelError> {
            linear(dim, dim, vb.pp(name)).map_err(|error| ModelError::load(error.to_string()))
        };
        Ok(Self {
            q: load("q")?,
            k: load("k")?,
            v: load("v")?,
            o: load("o")?,
            heads,
            head_dim: dim / heads,
        })
    }

    fn forward(
        &self,
        queries: &Tensor,
        memory: &Tensor,
        memory_pad: &Tensor,
    ) -> Result<Tensor, ModelError> {
        let (rows, query_len, dim) = queries.dims3().map_err(inference_error)?;
        let memory_len = memory.dim(1).map_err(inference_error)?;
        let q = queries
            .apply(&self.q)
            .map_err(inference_error)?
            .reshape((rows, query_len, self.heads, self.head_dim))
            .map_err(inference_error)?
            .transpose(1, 2)
            .map_err(inference_error)?;
        let k = memory
            .apply(&self.k)
            .map_err(inference_error)?
            .reshape((rows, memory_len, self.heads, self.head_dim))
            .map_err(inference_error)?
            .transpose(1, 2)
            .map_err(inference_error)?;
        let v = memory
            .apply(&self.v)
            .map_err(inference_error)?
            .reshape((rows, memory_len, self.heads, self.head_dim))
            .map_err(inference_error)?
            .transpose(1, 2)
            .map_err(inference_error)?;
        let scores = q
            .contiguous()
            .map_err(inference_error)?
            .matmul(
                &k.transpose(2, 3)
                    .map_err(inference_error)?
                    .contiguous()
                    .map_err(inference_error)?,
            )
            .map_err(inference_error)?
            .affine((self.head_dim as f64).powf(-0.5), 0.0)
            .map_err(inference_error)?;
        let bias = memory_pad
            .affine(-1.0, 1.0)
            .map_err(inference_error)?
            .affine(-1e4, 0.0)
            .map_err(inference_error)?
            .unsqueeze(1)
            .map_err(inference_error)?
            .unsqueeze(2)
            .map_err(inference_error)?;
        let scores = scores.broadcast_add(&bias).map_err(inference_error)?;
        let dtype = q.dtype();
        let probabilities = if matches!(dtype, DType::F16 | DType::BF16) {
            candle_nn::ops::softmax(
                &scores.to_dtype(DType::F32).map_err(inference_error)?,
                candle_core::D::Minus1,
            )
            .map_err(inference_error)?
            .to_dtype(dtype)
            .map_err(inference_error)?
        } else {
            candle_nn::ops::softmax(&scores, candle_core::D::Minus1).map_err(inference_error)?
        };
        probabilities
            .matmul(&v.contiguous().map_err(inference_error)?)
            .map_err(inference_error)?
            .transpose(1, 2)
            .map_err(inference_error)?
            .reshape((rows, query_len, dim))
            .map_err(inference_error)?
            .apply(&self.o)
            .map_err(inference_error)
    }
}

struct ScorerLayer {
    self_attn: Attention,
    cross_attn: Attention,
    ff1: Linear,
    ff2: Linear,
    n1: LayerNorm,
    n2: LayerNorm,
    n3: LayerNorm,
    activation: Activation,
}

impl ScorerLayer {
    fn load(vb: VarBuilder, dim: usize, cfg: &DecimaConfig) -> Result<Self, ModelError> {
        let norm = |name: &str| {
            layer_norm(dim, 1e-5, vb.pp(name)).map_err(|error| ModelError::load(error.to_string()))
        };
        Ok(Self {
            self_attn: Attention::load(vb.pp("self_attn"), dim, cfg.heads)?,
            cross_attn: Attention::load(vb.pp("cross_attn"), dim, cfg.heads)?,
            ff1: linear(dim, cfg.ffn, vb.pp("ff").pp("0"))
                .map_err(|error| ModelError::load(error.to_string()))?,
            ff2: linear(cfg.ffn, dim, vb.pp("ff").pp("3"))
                .map_err(|error| ModelError::load(error.to_string()))?,
            n1: norm("n1")?,
            n2: norm("n2")?,
            n3: norm("n3")?,
            activation: Activation::Gelu,
        })
    }

    fn forward(
        &self,
        x: &Tensor,
        x_pad: &Tensor,
        memory: &Tensor,
        memory_pad: &Tensor,
    ) -> Result<Tensor, ModelError> {
        let h = x.apply(&self.n1).map_err(inference_error)?;
        let x = x
            .add(&self.self_attn.forward(&h, &h, x_pad)?)
            .map_err(inference_error)?;
        let h = x.apply(&self.n2).map_err(inference_error)?;
        let x = x
            .add(&self.cross_attn.forward(&h, memory, memory_pad)?)
            .map_err(inference_error)?;
        let h = x.apply(&self.n3).map_err(inference_error)?;
        let ff = self
            .activation
            .forward(&h.apply(&self.ff1).map_err(inference_error)?)
            .map_err(inference_error)?;
        x.add(&ff.apply(&self.ff2).map_err(inference_error)?)
            .map_err(inference_error)
    }
}

struct Scorer {
    layers: Vec<ScorerLayer>,
    out_norm: LayerNorm,
    score: Linear,
    ord_g: Linear,
    ord_gap: Linear,
    sim_scale: f32,
    sim_center: f32,
}

impl Scorer {
    fn load(vb: VarBuilder, dim: usize, cfg: &DecimaConfig) -> Result<Self, ModelError> {
        let layers = (0..cfg.scorer_layers)
            .map(|index| ScorerLayer::load(vb.pp(format!("layers.{index}")), dim, cfg))
            .collect::<Result<_, _>>()?;
        let scalar = |name: &str| {
            vb.get_unchecked(name)
                .and_then(|tensor| tensor.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>())
                .map(|values| values[0])
                .map_err(|error| ModelError::load(format!("{name}: {error}")))
        };
        Ok(Self {
            layers,
            out_norm: layer_norm(dim, 1e-5, vb.pp("out_norm"))
                .map_err(|error| ModelError::load(error.to_string()))?,
            score: linear(dim, 1, vb.pp("score"))
                .map_err(|error| ModelError::load(error.to_string()))?,
            ord_g: linear(dim, 1, vb.pp("ord_g"))
                .map_err(|error| ModelError::load(error.to_string()))?,
            ord_gap: linear(dim, 1, vb.pp("ord_gap"))
                .map_err(|error| ModelError::load(error.to_string()))?,
            sim_scale: scalar("sim_scale")?,
            sim_center: scalar("sim_center")?,
        })
    }

    fn forward(
        &self,
        state_hidden: &Tensor,
        state_mask: &Tensor,
        option_hidden: &Tensor,
        option_mask: &Tensor,
    ) -> Result<(Tensor, Tensor), ModelError> {
        let rows = option_hidden.dim(0).map_err(inference_error)?;
        let state_len = state_hidden.dim(1).map_err(inference_error)?;
        let dim = state_hidden.dim(2).map_err(inference_error)?;
        let memory = state_hidden
            .broadcast_as((rows, state_len, dim))
            .map_err(inference_error)?;
        let memory_pad = state_mask
            .broadcast_as((rows, state_len))
            .map_err(inference_error)?;
        let option_pad = option_mask.affine(-1.0, 1.0).map_err(inference_error)?;
        let mut x = option_hidden.clone();
        for layer in &self.layers {
            x = layer.forward(&x, &option_pad, &memory, &memory_pad)?;
        }
        let z = masked_mean(
            &x.apply(&self.out_norm).map_err(inference_error)?,
            option_mask,
        )?;
        let score = self
            .score
            .forward(&z)
            .map_err(inference_error)?
            .squeeze(1)
            .map_err(inference_error)?;
        let mean_state = masked_mean(state_hidden, state_mask)?;
        let mean_option = masked_mean(option_hidden, option_mask)?;
        let cosine = cosine_similarity(&mean_state, &mean_option)?;
        let sim = cosine
            .affine(
                f64::from(self.sim_scale),
                -f64::from(self.sim_scale) * f64::from(self.sim_center),
            )
            .map_err(inference_error)?;
        Ok((score.add(&sim).map_err(inference_error)?, z))
    }

    fn projections(&self, z: &Tensor) -> Result<(Vec<f32>, Vec<f32>), ModelError> {
        let project = |linear: &Linear| -> Result<Vec<f32>, ModelError> {
            linear
                .forward(z)
                .map_err(inference_error)?
                .squeeze(1)
                .map_err(inference_error)?
                .to_dtype(DType::F32)
                .map_err(inference_error)?
                .to_vec1::<f32>()
                .map_err(inference_error)
        };
        Ok((project(&self.ord_g)?, project(&self.ord_gap)?))
    }
}

fn masked_mean(values: &Tensor, mask: &Tensor) -> Result<Tensor, ModelError> {
    let (rows, _length, _dim) = values.dims3().map_err(inference_error)?;
    let count = mask
        .sum(1)
        .map_err(inference_error)?
        .reshape((rows, 1))
        .map_err(inference_error)?
        .clamp(1.0, 1e9)
        .map_err(inference_error)?;
    let summed = values
        .broadcast_mul(&mask.unsqueeze(2).map_err(inference_error)?)
        .map_err(inference_error)?
        .sum(1)
        .map_err(inference_error)?;
    summed.broadcast_div(&count).map_err(inference_error)
}

fn cosine_similarity(a: &Tensor, b: &Tensor) -> Result<Tensor, ModelError> {
    let normalize = |values: &Tensor| -> Result<Tensor, ModelError> {
        let norm = values
            .sqr()
            .map_err(inference_error)?
            .sum_keepdim(1)
            .map_err(inference_error)?
            .sqrt()
            .map_err(inference_error)?
            .clamp(1e-8, 1e30)
            .map_err(inference_error)?;
        values.broadcast_div(&norm).map_err(inference_error)
    };
    let a = normalize(a)?;
    let b = normalize(b)?;
    a.broadcast_mul(&b)
        .map_err(inference_error)?
        .sum(1)
        .map_err(inference_error)
}

fn softplus(value: f64) -> f64 {
    if value > 20.0 {
        value
    } else {
        value.exp().ln_1p()
    }
}

fn sigmoid(value: f64) -> f64 {
    1.0 / (1.0 + (-value).exp())
}

fn log_softmax(values: &[f64]) -> Vec<f64> {
    let max = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let sum = max
        + values
            .iter()
            .map(|value| (value - max).exp())
            .sum::<f64>()
            .ln();
    values.iter().map(|value| value - sum).collect()
}

/// Decima's cumulative-link ordinal head.
fn ordinal_log_probs(scores: &[f32], g: &[f32], gap: &[f32], temperature: f64) -> Vec<f64> {
    let k = scores.len();
    if k == 0 {
        return Vec::new();
    }
    let tempered: Vec<f64> = scores
        .iter()
        .map(|value| f64::from(*value) / temperature)
        .collect();
    let logits = log_softmax(&tempered);
    let expected = logits
        .iter()
        .enumerate()
        .map(|(index, log_probability)| log_probability.exp() * index as f64)
        .sum::<f64>()
        - (k as f64 - 1.0) / 2.0;
    let latent = g.iter().map(|value| f64::from(*value)).sum::<f64>() / k as f64 + expected;
    let gaps: Vec<f64> = gap
        .iter()
        .map(|value| softplus(f64::from(*value)) + 1e-3)
        .collect();
    let half = gaps.iter().sum::<f64>() / 2.0;
    let mut above = Vec::with_capacity(k - 1);
    let mut theta = 0.0;
    for gap in &gaps[..k - 1] {
        theta += gap;
        above.push(sigmoid(latent - (theta - half)));
    }
    let levels: Vec<f64> = (0..k)
        .map(|index| {
            let high = if index == 0 { 1.0 } else { above[index - 1] };
            let low = if index + 1 == k { 0.0 } else { above[index] };
            (high - low).max(1e-7).ln()
        })
        .collect();
    log_softmax(&levels)
}

struct Row {
    ids: Vec<u32>,
    truncated: bool,
}

struct QuestionPlan {
    id: String,
    kind: QuestionKind,
    state: Row,
    options: Vec<Row>,
    labels: Vec<String>,
    legend: IndexMap<String, Content>,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum QuestionKind {
    Choice,
    Score,
    Noul,
}

type QuestionScores = (Vec<f32>, Vec<f32>, Vec<f32>);

enum EncoderKind {
    Modern(Encoder),
    Bert(BertEncoder),
}

impl EncoderKind {
    fn forward(
        &self,
        ids: &Tensor,
        mask: &Tensor,
        has_padding: bool,
    ) -> Result<Tensor, ModelError> {
        match self {
            Self::Modern(encoder) => encoder
                .forward(ids, mask, has_padding)
                .map_err(inference_error),
            Self::Bert(encoder) => encoder.forward(ids, mask),
        }
    }
}

/// A Decima late-interaction model.
pub struct DecimaModel {
    tokenizer: Tokenizer,
    encoder: EncoderKind,
    scorer: Scorer,
    config: DecimaConfig,
    bos: u32,
    eos: u32,
    device: Device,
    dtype: DType,
}

impl DecimaModel {
    pub fn load_dir(directory: &Path, options: DecimaOptions) -> Result<Self, ModelError> {
        let checkpoint = if directory.join("pytorch/decima.json").is_file() {
            directory.join("pytorch")
        } else {
            directory.to_path_buf()
        };
        let config_path = checkpoint.join("decima.json");
        let mut config: DecimaConfig =
            serde_json::from_slice(&std::fs::read(&config_path).map_err(|error| {
                ModelError::load(format!("read {}: {error}", config_path.display()))
            })?)
            .map_err(|error| {
                ModelError::load(format!("parse {}: {error}", config_path.display()))
            })?;
        if let Some(choice_fit) = options.choice_fit {
            config.choice_fit = choice_fit;
        }
        if config.max_state_tokens < 3 || config.max_choice_tokens < 3 {
            return Err(ModelError::load("Decima token budgets are too small"));
        }
        let encoder_dir = checkpoint.join("encoder");
        let config_path = encoder_dir.join("config.json");
        let config_bytes = std::fs::read(&config_path).map_err(|error| {
            ModelError::load(format!("read {}: {error}", config_path.display()))
        })?;
        let raw: serde_json::Value = serde_json::from_slice(&config_bytes).map_err(|error| {
            ModelError::load(format!("parse {}: {error}", config_path.display()))
        })?;
        let is_bert = raw
            .get("model_type")
            .and_then(serde_json::Value::as_str)
            .is_some_and(|model_type| model_type.eq_ignore_ascii_case("bert"))
            || raw
                .get("architectures")
                .and_then(serde_json::Value::as_array)
                .is_some_and(|architectures| {
                    architectures.iter().any(|value| {
                        value
                            .as_str()
                            .is_some_and(|name| name.contains("BertModel"))
                    })
                });
        let tokenizer_path = [
            encoder_dir.join("tokenizer.json"),
            checkpoint.join("tokenizer.json"),
        ]
        .into_iter()
        .find(|path| path.is_file())
        .ok_or_else(|| ModelError::load("Decima checkpoint has no tokenizer.json"))?;
        let mut tokenizer = Tokenizer::from_file(&tokenizer_path)
            .map_err(|error| ModelError::load(format!("load tokenizer: {error}")))?;
        tokenizer
            .with_truncation(None)
            .map_err(|error| ModelError::load(format!("disable truncation: {error}")))?;
        let _ = tokenizer.with_padding(None);

        let encoder_tensors =
            candle_core::safetensors::load(encoder_dir.join("model.safetensors"), &options.device)
                .map_err(|error| ModelError::load(format!("load encoder: {error}")))?;
        let encoder_vb = VarBuilder::from_tensors(encoder_tensors, options.dtype, &options.device);
        let (encoder, hidden_size, bos, eos) = if is_bert {
            let bert_config = BertConfig::from_slice(&config_bytes)
                .map_err(|error| ModelError::load(format!("parse BERT config: {error}")))?;
            if config.max_state_tokens > bert_config.max_position_embeddings {
                return Err(ModelError::load(format!(
                    "Decima state budget {} exceeds the BERT encoder's {} positions",
                    config.max_state_tokens, bert_config.max_position_embeddings
                )));
            }
            let hidden_size = bert_config.hidden_size;
            let bos = bert_config.cls_token_id.unwrap_or(0);
            let eos = bert_config.sep_token_id.unwrap_or(2);
            (
                EncoderKind::Bert(BertEncoder::load(encoder_vb, &bert_config)?),
                hidden_size,
                bos,
                eos,
            )
        } else {
            let mut encoder_config = ModernBertConfig::from_slice(&config_bytes)
                .map_err(|error| ModelError::load(format!("parse encoder config: {error}")))?;
            if encoder_config.max_position_embeddings < config.max_state_tokens {
                encoder_config.max_position_embeddings = config.max_state_tokens;
            }
            let prefix = ["model.", "", "encoder.", "encoder.model."]
                .into_iter()
                .find(|prefix| {
                    encoder_vb.contains_tensor(&format!("{prefix}embeddings.tok_embeddings.weight"))
                })
                .ok_or_else(|| ModelError::load("Decima encoder has no token embeddings"))?;
            let hidden_size = encoder_config.hidden_size;
            let bos = encoder_config
                .bos_token_id
                .or(encoder_config.cls_token_id)
                .unwrap_or(2);
            let eos = encoder_config
                .eos_token_id
                .or(encoder_config.sep_token_id)
                .unwrap_or(1);
            let encoder = Encoder::load(encoder_vb, &encoder_config, options.dtype, false, prefix)
                .map_err(|error| ModelError::load(error.to_string()))?;
            (EncoderKind::Modern(encoder), hidden_size, bos, eos)
        };

        let head_path = checkpoint.join("head.safetensors");
        let head_tensors = candle_core::safetensors::load(&head_path, &options.device)
            .map_err(|error| ModelError::load(format!("load head: {error}")))?;
        let head_vb = VarBuilder::from_tensors(head_tensors, options.dtype, &options.device);
        let scorer = Scorer::load(head_vb, hidden_size, &config)?;
        Ok(Self {
            tokenizer,
            encoder,
            scorer,
            config,
            bos,
            eos,
            device: options.device,
            dtype: options.dtype,
        })
    }

    fn encode(&self, text: &str) -> Result<Vec<u32>, ModelError> {
        Ok(self
            .tokenizer
            .encode(text, false)
            .map_err(|error| ModelError::inference(format!("tokenize: {error}")))?
            .get_ids()
            .to_vec())
    }

    fn normalize(&self, text: &str) -> String {
        strip(&text.nfc().collect::<String>()).to_owned()
    }

    fn row(&self, text: &str, max_len: usize) -> Result<Row, ModelError> {
        let body = self.encode(text)?;
        let keep = body.len().min(max_len.saturating_sub(2));
        let mut ids = Vec::with_capacity(keep + 2);
        ids.push(self.bos);
        ids.extend_from_slice(&body[..keep]);
        ids.push(self.eos);
        Ok(Row {
            ids,
            truncated: keep < body.len(),
        })
    }

    fn text(&self, value: Option<&Content>) -> String {
        match value {
            None | Some(Content::Null) => String::new(),
            Some(Content::String(text)) => text.clone(),
            Some(content) => python_json(&sys1_core::content_to_value(content), false),
        }
    }

    fn question_text(&self, id: &str, question: &Question) -> (String, Vec<String>, QuestionKind) {
        let instructions = question
            .instructions()
            .map(|value| self.text(Some(value)))
            .unwrap_or_else(|| id.to_owned());
        match question {
            Question::Choice(choice) => {
                let options = choice
                    .criteria
                    .iter()
                    .map(|(label, value)| {
                        let description = self.text(Some(value));
                        if strip(&description).is_empty() {
                            label.clone()
                        } else {
                            format!("{label}: {description}")
                        }
                    })
                    .collect();
                (instructions, options, QuestionKind::Choice)
            }
            Question::Score(score) => (
                instructions,
                score
                    .criteria
                    .iter()
                    .map(|value| self.text(Some(value)))
                    .collect(),
                QuestionKind::Score,
            ),
            Question::Noul(noul) => {
                let mut body = instructions;
                if let Some(NoulCriteria { is_true, is_false }) = &noul.criteria {
                    let t = strip(&self.text(is_true.as_ref())).to_owned();
                    let f = strip(&self.text(is_false.as_ref())).to_owned();
                    if !t.is_empty() || !f.is_empty() {
                        body.push_str(&format!(
                            "\nTrue if: {}\nFalse if: {}",
                            if t.is_empty() { "—" } else { &t },
                            if f.is_empty() { "—" } else { &f }
                        ));
                    }
                }
                (
                    body,
                    vec!["yes".to_owned(), "no".to_owned()],
                    QuestionKind::Noul,
                )
            }
            Question::Box(_) => (instructions, Vec::new(), QuestionKind::Choice),
        }
    }

    fn option_texts(
        &self,
        question: &str,
        options: &[String],
        question_in_options: bool,
    ) -> Result<Vec<String>, ModelError> {
        let prefix = &self.config.choice_prefix;
        let max_len = self.config.max_choice_tokens;
        let question_in_options = question_in_options && !question.is_empty();
        let question_text = if question_in_options { question } else { "" };
        let full: Vec<String> = options
            .iter()
            .map(|option| {
                format!(
                    "{prefix}{}",
                    self.normalize(strip(&format!("{question_text} {option}")))
                )
            })
            .collect();
        if !self.config.choice_fit {
            let mut rows = std::collections::HashSet::new();
            for text in &full {
                rows.insert(self.row(text, max_len)?.ids);
            }
            if rows.len() == full.len() {
                return Ok(full);
            }
        }
        let question = if question_in_options {
            self.normalize(question)
        } else {
            String::new()
        };
        let words: Vec<&str> = question.split_whitespace().collect();
        let fitted = |head: usize, option: &str| -> String {
            format!("{prefix}{} … {option}", words[..head].join(" "))
        };
        let mut out = Vec::with_capacity(options.len());
        for (option, full) in options.iter().zip(full) {
            if self.encode(&full)?.len() + 2 <= max_len {
                out.push(full);
                continue;
            }
            let option = self.normalize(option);
            let (mut low, mut high) = (0usize, words.len());
            while low < high {
                let middle = (low + high).div_ceil(2);
                if self.encode(&fitted(middle, &option))?.len() + 2 <= max_len {
                    low = middle;
                } else {
                    high = middle - 1;
                }
            }
            out.push(if low > 0 {
                fitted(low, &option)
            } else {
                format!("{prefix}{option}")
            });
        }
        Ok(out)
    }

    fn plan(&self, state: &str, questions: &Questions) -> Result<Vec<QuestionPlan>, ModelError> {
        if !(self.config.limits.questions[0]..=self.config.limits.questions[1])
            .contains(&questions.len())
        {
            return Err(ModelError::invalid_request(format!(
                "Decima takes {} to {} questions",
                self.config.limits.questions[0], self.config.limits.questions[1]
            )));
        }
        let mut plans = Vec::with_capacity(questions.len());
        for (id, question) in questions.iter() {
            if matches!(question, Question::Box(_)) {
                return Err(ModelError::invalid_request(
                    "Decima does not answer box questions",
                ));
            }
            let (body, options, kind) = self.question_text(id, question);
            let option_count = options.len();
            let max_options = match kind {
                QuestionKind::Choice => self.config.limits.choices[1],
                QuestionKind::Score => self.config.limits.levels[1],
                QuestionKind::Noul => 2,
            };
            let min_options = match kind {
                QuestionKind::Choice => self.config.limits.choices[0],
                QuestionKind::Score => self.config.limits.levels[0],
                QuestionKind::Noul => 1,
            };
            if option_count < min_options || option_count > max_options {
                return Err(ModelError::invalid_request(format!(
                    "question {id:?}: this model takes {min_options}..{max_options} options, got {option_count}"
                )));
            }
            let state_text = if self.config.question_in_state && !body.is_empty() {
                format!("{body}\n{state}")
            } else {
                state.to_owned()
            };
            let state_row = self.row(
                &format!(
                    "{}{}",
                    self.config.state_prefix,
                    self.normalize(&state_text)
                ),
                self.config.max_state_tokens,
            )?;
            if state_row.truncated {
                return Err(ModelError::context_length_exceeded(format!(
                    "question {id:?}: the state row exceeds {} tokens",
                    self.config.max_state_tokens
                )));
            }
            let option_texts =
                self.option_texts(&body, &options, self.config.question_in_choices)?;
            let option_rows = option_texts
                .iter()
                .map(|text| self.row(text, self.config.max_choice_tokens))
                .collect::<Result<Vec<_>, _>>()?;
            let (labels, legend) = match question {
                Question::Choice(choice) => {
                    (choice.criteria.keys().cloned().collect(), IndexMap::new())
                }
                Question::Score(score) => (
                    (0..score.criteria.len())
                        .map(|index| index.to_string())
                        .collect(),
                    score
                        .criteria
                        .iter()
                        .enumerate()
                        .map(|(index, value)| (index.to_string(), value.clone()))
                        .collect(),
                ),
                Question::Noul(_) => (Vec::new(), IndexMap::new()),
                Question::Box(_) => unreachable!(),
            };
            plans.push(QuestionPlan {
                id: id.clone(),
                kind,
                state: state_row,
                options: option_rows,
                labels,
                legend,
            });
        }
        Ok(plans)
    }

    fn forward_question(&self, plan: &QuestionPlan) -> Result<QuestionScores, ModelError> {
        let state_ids = Tensor::from_vec(
            plan.state.ids.clone(),
            (1, plan.state.ids.len()),
            &self.device,
        )
        .map_err(inference_error)?;
        let state_mask = Tensor::ones((1, plan.state.ids.len()), self.dtype, &self.device)
            .map_err(inference_error)?;
        let state_hidden = self.encoder.forward(&state_ids, &state_mask, false)?;
        let mut scores = Vec::with_capacity(plan.options.len());
        let mut latent = Vec::with_capacity(plan.options.len());
        for option in &plan.options {
            let ids = Tensor::from_vec(option.ids.clone(), (1, option.ids.len()), &self.device)
                .map_err(inference_error)?;
            let mask = Tensor::ones((1, option.ids.len()), self.dtype, &self.device)
                .map_err(inference_error)?;
            let hidden = self.encoder.forward(&ids, &mask, false)?;
            let (score, z) = self
                .scorer
                .forward(&state_hidden, &state_mask, &hidden, &mask)?;
            scores.push(
                score
                    .to_dtype(DType::F32)
                    .map_err(inference_error)?
                    .to_vec1::<f32>()
                    .map_err(inference_error)?[0],
            );
            latent.push(z);
        }
        let z = Tensor::cat(&latent.iter().collect::<Vec<_>>(), 0).map_err(inference_error)?;
        let (g, gap) = self.scorer.projections(&z)?;
        Ok((scores, g, gap))
    }

    fn answer(&self, plan: &QuestionPlan) -> Result<Answer, ModelError> {
        let (scores, g, gap) = self.forward_question(plan)?;
        match plan.kind {
            QuestionKind::Choice => {
                let probabilities = softmax(&scores, self.config.temperature as f32);
                Ok(choice_answer(&plan.labels, &probabilities))
            }
            QuestionKind::Noul => {
                let probabilities = softmax(&scores, self.config.temperature as f32);
                Ok(sys1_core::noul_answer(probabilities[1], probabilities[0]))
            }
            QuestionKind::Score => {
                let log_probs = ordinal_log_probs(&scores, &g, &gap, self.config.temperature);
                let probabilities: Vec<f32> =
                    log_probs.iter().map(|value| value.exp() as f32).collect();
                Ok(sys1_core::score_answer(&plan.legend, &probabilities))
            }
        }
    }
}

impl DecisionModel for DecimaModel {
    fn predict(&self, state: &State, questions: &Questions) -> Result<DecisionOutput, ModelError> {
        if questions.is_empty() {
            return Err(ModelError::invalid_request("questions must not be empty"));
        }
        let state_text = sys1_core::render_state(state);
        let plans = self.plan(&state_text, questions)?;
        let mut answers = IndexMap::with_capacity(plans.len());
        for plan in &plans {
            answers.insert(plan.id.clone(), self.answer(plan)?);
        }
        let input_tokens = plans
            .iter()
            .map(|plan| {
                plan.state.ids.len() as u64
                    + plan
                        .options
                        .iter()
                        .map(|row| row.ids.len() as u64)
                        .sum::<u64>()
            })
            .sum();
        Ok(DecisionOutput::new(
            Answers::new(answers),
            Usage::new(input_tokens, 0),
        ))
    }

    fn supports_cross_request_batching(&self) -> bool {
        false
    }

    fn predict_batch(
        &self,
        requests: &[Prediction<'_>],
    ) -> Vec<Result<DecisionOutput, ModelError>> {
        requests
            .iter()
            .map(|request| self.predict(request.state, request.questions))
            .collect()
    }
}

fn inference_error(error: candle_core::Error) -> ModelError {
    ModelError::inference(error.to_string())
}
