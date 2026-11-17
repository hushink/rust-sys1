use candle_core::{D, DType, Device, Tensor};
use candle_nn::{LayerNorm, Linear, Module, VarBuilder, layer_norm, linear};
use indexmap::IndexMap;
use serde::Deserialize;
use serde_json::{Map, Value};
use std::path::Path;
use sys1_core::{
    Answers, DecisionOutput, ModelError, Prediction, Questions, canonical_json, choice_answer,
    content_to_value, noul_answer, score_answer, softmax,
};
use sys1_protocol::{Answer, Content, Question, State, Usage};
use sys1_qwen::{Qwen35, Qwen35Config, WeightStore, detect_prefix, safetensors_files};
use tokenizers::Tokenizer;

const SYSTEM_PROMPT: &str = "Read the complete state and schema. Decide every field jointly. Each answer must be exactly one of that field's allowed options.";
const QUESTION_TYPES: [&str; 3] = ["noul", "choice", "score"];
const NOUL_TRUE: &str = "The proposition is true or the answer is yes.";
const NOUL_FALSE: &str = "The proposition is false or the answer is no.";
const DEFAULT_MAX_LEN: usize = 4096;

/// Runtime options for Clef.
#[derive(Clone)]
pub struct ClefOptions {
    pub device: Device,
    pub dtype: DType,
    pub max_len: Option<usize>,
}

impl Default for ClefOptions {
    fn default() -> Self {
        Self {
            device: Device::Cpu,
            dtype: DType::F32,
            max_len: None,
        }
    }
}

#[derive(Clone, Debug, Deserialize)]
struct HeadConfig {
    hidden_size: usize,
    width: usize,
    routing_layers: usize,
    layers: usize,
    heads: usize,
    feedforward: usize,
}

struct Mha {
    q: Linear,
    k: Linear,
    v: Linear,
    out: Linear,
    heads: usize,
    head_dim: usize,
}

impl Mha {
    fn load(vb: VarBuilder, width: usize, heads: usize) -> Result<Self, ModelError> {
        let weight = vb
            .get((3 * width, width), "in_proj_weight")
            .map_err(|error| ModelError::load(error.to_string()))?;
        let bias = vb
            .get(3 * width, "in_proj_bias")
            .map_err(|error| ModelError::load(error.to_string()))?;
        let slice = |offset: usize| -> Result<Linear, ModelError> {
            let weight = weight
                .narrow(0, offset, width)
                .map_err(|error| ModelError::load(error.to_string()))?
                .contiguous()
                .map_err(|error| ModelError::load(error.to_string()))?;
            let bias = bias
                .narrow(0, offset, width)
                .map_err(|error| ModelError::load(error.to_string()))?
                .contiguous()
                .map_err(|error| ModelError::load(error.to_string()))?;
            let bias = Some(
                bias.to_dtype(weight.dtype())
                    .map_err(|error| ModelError::load(error.to_string()))?,
            );
            Ok(Linear::new(weight, bias))
        };
        Ok(Self {
            q: slice(0)?,
            k: slice(width)?,
            v: slice(2 * width)?,
            out: linear(width, width, vb.pp("out_proj"))
                .map_err(|error| ModelError::load(error.to_string()))?,
            heads,
            head_dim: width / heads,
        })
    }

    fn forward(
        &self,
        queries: &Tensor,
        keys: &Tensor,
        values: &Tensor,
        causal: bool,
    ) -> Result<Tensor, ModelError> {
        let (batch, query_len, width) = queries.dims3().map_err(inference_error)?;
        let key_len = keys.dim(1).map_err(inference_error)?;
        let q = self
            .q
            .forward(queries)
            .map_err(inference_error)?
            .reshape((batch, query_len, self.heads, self.head_dim))
            .map_err(inference_error)?
            .transpose(1, 2)
            .map_err(inference_error)?;
        let k = self
            .k
            .forward(keys)
            .map_err(inference_error)?
            .reshape((batch, key_len, self.heads, self.head_dim))
            .map_err(inference_error)?
            .transpose(1, 2)
            .map_err(inference_error)?;
        let v = self
            .v
            .forward(values)
            .map_err(inference_error)?
            .reshape((batch, key_len, self.heads, self.head_dim))
            .map_err(inference_error)?
            .transpose(1, 2)
            .map_err(inference_error)?;
        let mut scores = q
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
        if causal {
            let mask = Tensor::tril2(query_len, DType::F32, scores.device())
                .map_err(inference_error)?
                .affine(1e9, -1e9)
                .map_err(inference_error)?
                .to_dtype(scores.dtype())
                .map_err(inference_error)?
                .reshape((1, 1, query_len, query_len))
                .map_err(inference_error)?;
            scores = scores.broadcast_add(&mask).map_err(inference_error)?;
        }
        let dtype = q.dtype();
        let probabilities = if matches!(dtype, DType::F16 | DType::BF16) {
            candle_nn::ops::softmax(
                &scores.to_dtype(DType::F32).map_err(inference_error)?,
                D::Minus1,
            )
            .map_err(inference_error)?
            .to_dtype(dtype)
            .map_err(inference_error)?
        } else {
            candle_nn::ops::softmax(&scores, D::Minus1).map_err(inference_error)?
        };
        probabilities
            .matmul(&v.contiguous().map_err(inference_error)?)
            .map_err(inference_error)?
            .transpose(1, 2)
            .map_err(inference_error)?
            .reshape((batch, query_len, width))
            .map_err(inference_error)?
            .apply(&self.out)
            .map_err(inference_error)
    }
}

struct EvidenceLayer {
    query_norm: LayerNorm,
    memory_norm: LayerNorm,
    attention: Mha,
    feedforward_norm: LayerNorm,
    ff1: Linear,
    ff2: Linear,
}

impl EvidenceLayer {
    fn load(vb: VarBuilder, cfg: &HeadConfig) -> Result<Self, ModelError> {
        let width = cfg.width;
        Ok(Self {
            query_norm: layer_norm(width, 1e-5, vb.pp("query_norm"))
                .map_err(|error| ModelError::load(error.to_string()))?,
            memory_norm: layer_norm(width, 1e-5, vb.pp("memory_norm"))
                .map_err(|error| ModelError::load(error.to_string()))?,
            attention: Mha::load(vb.pp("attention"), width, cfg.heads)?,
            feedforward_norm: layer_norm(width, 1e-5, vb.pp("feedforward_norm"))
                .map_err(|error| ModelError::load(error.to_string()))?,
            ff1: linear(width, cfg.feedforward, vb.pp("feedforward").pp("0"))
                .map_err(|error| ModelError::load(error.to_string()))?,
            ff2: linear(cfg.feedforward, width, vb.pp("feedforward").pp("3"))
                .map_err(|error| ModelError::load(error.to_string()))?,
        })
    }

    fn forward(&self, xs: &Tensor, memory: &Tensor) -> Result<Tensor, ModelError> {
        let queries = xs.apply(&self.query_norm).map_err(inference_error)?;
        let memory = memory.apply(&self.memory_norm).map_err(inference_error)?;
        let xs = xs
            .add(&self.attention.forward(&queries, &memory, &memory, false)?)
            .map_err(inference_error)?;
        let feed = self
            .ff1
            .forward(&xs.apply(&self.feedforward_norm).map_err(inference_error)?)
            .map_err(inference_error)?
            .gelu_erf()
            .map_err(inference_error)?;
        xs.add(&self.ff2.forward(&feed).map_err(inference_error)?)
            .map_err(inference_error)
    }
}

struct DecoderLayer {
    norm1: LayerNorm,
    self_attn: Mha,
    norm2: LayerNorm,
    cross_attn: Mha,
    norm3: LayerNorm,
    linear1: Linear,
    linear2: Linear,
}

impl DecoderLayer {
    fn load(vb: VarBuilder, cfg: &HeadConfig) -> Result<Self, ModelError> {
        Ok(Self {
            norm1: layer_norm(cfg.width, 1e-5, vb.pp("norm1"))
                .map_err(|error| ModelError::load(error.to_string()))?,
            self_attn: Mha::load(vb.pp("self_attn"), cfg.width, cfg.heads)?,
            norm2: layer_norm(cfg.width, 1e-5, vb.pp("norm2"))
                .map_err(|error| ModelError::load(error.to_string()))?,
            cross_attn: Mha::load(vb.pp("multihead_attn"), cfg.width, cfg.heads)?,
            norm3: layer_norm(cfg.width, 1e-5, vb.pp("norm3"))
                .map_err(|error| ModelError::load(error.to_string()))?,
            linear1: linear(cfg.width, cfg.feedforward, vb.pp("linear1"))
                .map_err(|error| ModelError::load(error.to_string()))?,
            linear2: linear(cfg.feedforward, cfg.width, vb.pp("linear2"))
                .map_err(|error| ModelError::load(error.to_string()))?,
        })
    }

    fn forward(&self, xs: &Tensor, memory: &Tensor) -> Result<Tensor, ModelError> {
        let normed = xs.apply(&self.norm1).map_err(inference_error)?;
        let xs = xs
            .add(&self.self_attn.forward(&normed, &normed, &normed, true)?)
            .map_err(inference_error)?;
        let normed = xs.apply(&self.norm2).map_err(inference_error)?;
        let xs = xs
            .add(&self.cross_attn.forward(&normed, memory, memory, false)?)
            .map_err(inference_error)?;
        let feed = self
            .linear1
            .forward(&xs.apply(&self.norm3).map_err(inference_error)?)
            .map_err(inference_error)?
            .gelu_erf()
            .map_err(inference_error)?;
        xs.add(&self.linear2.forward(&feed).map_err(inference_error)?)
            .map_err(inference_error)
    }
}

struct JointHead {
    hidden_norm: LayerNorm,
    memory_projection: Linear,
    question_projection: Linear,
    option_question_projection: Linear,
    global_projection: Linear,
    option_context_projection: Linear,
    option_lexical_projection: Linear,
    type_embedding: Tensor,
    evidence_layers: Vec<EvidenceLayer>,
    option_summary_norm: LayerNorm,
    layers: Vec<DecoderLayer>,
    field_norm: LayerNorm,
    option_norm: LayerNorm,
    residual1: Linear,
    residual2: Linear,
    prior_scale: f32,
    joint_scale: f32,
    residual_gate: f32,
    width: usize,
}

impl JointHead {
    fn load(
        directory: &Path,
        config: HeadConfig,
        dtype: DType,
        device: &Device,
    ) -> Result<Self, ModelError> {
        let store = WeightStore::from_safetensors(
            &[directory.join("joint_head.safetensors")],
            dtype,
            device,
        )
        .map_err(|error| ModelError::load(format!("load joint head: {error}")))?;
        let vb = store.var_builder();
        let width = config.width;
        let projection = |name: &str| {
            candle_nn::linear_no_bias(config.hidden_size, width, vb.pp(name))
                .map_err(|error| ModelError::load(error.to_string()))
        };
        let scalar = |name: &str| {
            store
                .get_unchecked(name)
                .and_then(|tensor| tensor.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>())
                .map(|values| values[0])
                .map_err(|error| ModelError::load(format!("{name}: {error}")))
        };
        let evidence_layers = (0..config.routing_layers)
            .map(|index| EvidenceLayer::load(vb.pp(format!("evidence_layers.{index}")), &config))
            .collect::<Result<_, _>>()?;
        let layers = (0..config.layers)
            .map(|index| DecoderLayer::load(vb.pp(format!("layers.{index}")), &config))
            .collect::<Result<_, _>>()?;
        Ok(Self {
            hidden_norm: layer_norm(config.hidden_size, 1e-5, vb.pp("hidden_norm"))
                .map_err(|error| ModelError::load(error.to_string()))?,
            memory_projection: projection("memory_projection")?,
            question_projection: projection("question_projection")?,
            option_question_projection: projection("option_question_projection")?,
            global_projection: projection("global_projection")?,
            option_context_projection: projection("option_context_projection")?,
            option_lexical_projection: projection("option_lexical_projection")?,
            type_embedding: store
                .get("type_embedding.weight", &[3, width])
                .map_err(|error| ModelError::load(error.to_string()))?,
            evidence_layers,
            option_summary_norm: layer_norm(width, 1e-5, vb.pp("option_summary_norm"))
                .map_err(|error| ModelError::load(error.to_string()))?,
            layers,
            field_norm: layer_norm(width, 1e-5, vb.pp("field_norm"))
                .map_err(|error| ModelError::load(error.to_string()))?,
            option_norm: layer_norm(width, 1e-5, vb.pp("option_norm"))
                .map_err(|error| ModelError::load(error.to_string()))?,
            residual1: linear(4 * width, width, vb.pp("residual_scorer").pp("0"))
                .map_err(|error| ModelError::load(error.to_string()))?,
            residual2: linear(width, 1, vb.pp("residual_scorer").pp("3"))
                .map_err(|error| ModelError::load(error.to_string()))?,
            prior_scale: scalar("prior_logit_scale")?,
            joint_scale: scalar("joint_logit_scale")?,
            residual_gate: scalar("residual_gate")?,
            width,
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn forward(
        &self,
        hidden: &Tensor,
        ids: &[u32],
        question_spans: &[(usize, usize)],
        option_spans: &[Vec<(usize, usize)>],
        question_types: &[usize],
        output_embedding: &Tensor,
    ) -> Result<Vec<Vec<f32>>, ModelError> {
        let hidden = hidden.squeeze(0).map_err(inference_error)?;
        let sequence_length = hidden.dim(0).map_err(inference_error)?;
        let hidden = hidden.apply(&self.hidden_norm).map_err(inference_error)?;
        let memory = self
            .memory_projection
            .forward(&hidden)
            .map_err(inference_error)?
            .unsqueeze(0)
            .map_err(inference_error)?;
        let global = hidden
            .narrow(0, sequence_length - 1, 1)
            .map_err(inference_error)?
            .squeeze(0)
            .map_err(inference_error)?;
        let question_vectors = question_spans
            .iter()
            .map(|(start, end)| mean_span(&hidden, *start, *end))
            .collect::<Result<Vec<_>, _>>()?;
        let question_vectors = Tensor::stack(&question_vectors, 0).map_err(inference_error)?;
        let type_ids = Tensor::from_vec(
            question_types
                .iter()
                .map(|value| *value as u32)
                .collect::<Vec<_>>(),
            question_types.len(),
            hidden.device(),
        )
        .map_err(inference_error)?;
        let type_embeddings = self
            .type_embedding
            .index_select(&type_ids, 0)
            .map_err(inference_error)?;

        let mut option_contexts = Vec::with_capacity(option_spans.len());
        let mut lexical_options = Vec::with_capacity(option_spans.len());
        let mut counts = Vec::with_capacity(option_spans.len());
        for (question, spans) in option_spans.iter().enumerate() {
            let contexts = spans
                .iter()
                .map(|(start, end)| mean_span(&hidden, *start, *end))
                .collect::<Result<Vec<_>, _>>()?;
            let contexts = Tensor::stack(&contexts, 0).map_err(inference_error)?;
            let mut lexical = Vec::with_capacity(spans.len());
            for (start, end) in spans {
                let token_ids =
                    Tensor::from_vec(ids[*start..*end].to_vec(), end - start, hidden.device())
                        .map_err(inference_error)?;
                let rows = output_embedding
                    .index_select(&token_ids, 0)
                    .map_err(inference_error)?;
                lexical.push(rows.mean(0).map_err(inference_error)?);
            }
            let lexical = Tensor::stack(&lexical, 0).map_err(inference_error)?;
            let question_vector = question_vectors
                .narrow(0, question, 1)
                .map_err(inference_error)?
                .squeeze(0)
                .map_err(inference_error)?;
            let query = self
                .option_context_projection
                .forward(&contexts)
                .map_err(inference_error)?
                .add(
                    &self
                        .option_lexical_projection
                        .forward(&lexical)
                        .map_err(inference_error)?,
                )
                .map_err(inference_error)?
                .broadcast_add(
                    &self
                        .option_question_projection
                        .forward(&question_vector)
                        .map_err(inference_error)?
                        .unsqueeze(0)
                        .map_err(inference_error)?,
                )
                .map_err(inference_error)?;
            option_contexts.push(query);
            lexical_options.push(lexical);
            counts.push(spans.len());
        }
        let mut routed = Tensor::cat(&option_contexts.iter().collect::<Vec<_>>(), 0)
            .map_err(inference_error)?
            .unsqueeze(0)
            .map_err(inference_error)?;
        for layer in &self.evidence_layers {
            routed = layer.forward(&routed, &memory)?;
        }
        let routed = routed.squeeze(0).map_err(inference_error)?;
        let mut split = Vec::with_capacity(counts.len());
        let mut offset = 0;
        for count in &counts {
            split.push(routed.narrow(0, offset, *count).map_err(inference_error)?);
            offset += count;
        }
        let base_fields = self
            .question_projection
            .forward(&question_vectors)
            .map_err(inference_error)?;
        let mut summaries = Vec::with_capacity(split.len());
        for (index, options) in split.iter().enumerate() {
            let field = base_fields
                .narrow(0, index, 1)
                .map_err(inference_error)?
                .squeeze(0)
                .map_err(inference_error)?;
            let scores = options
                .matmul(&field.unsqueeze(1).map_err(inference_error)?)
                .map_err(inference_error)?
                .affine((self.width as f64).powf(-0.5), 0.0)
                .map_err(inference_error)?;
            let weights = candle_nn::ops::softmax(&scores, 0).map_err(inference_error)?;
            summaries.push(
                weights
                    .broadcast_mul(options)
                    .map_err(inference_error)?
                    .sum(0)
                    .map_err(inference_error)?,
            );
        }
        let summaries = Tensor::stack(&summaries, 0).map_err(inference_error)?;
        let fields = base_fields
            .add(
                &self
                    .option_summary_norm
                    .forward(&summaries)
                    .map_err(inference_error)?,
            )
            .map_err(inference_error)?
            .broadcast_add(
                &self
                    .global_projection
                    .forward(&global)
                    .map_err(inference_error)?
                    .unsqueeze(0)
                    .map_err(inference_error)?,
            )
            .map_err(inference_error)?
            .add(&type_embeddings)
            .map_err(inference_error)?
            .unsqueeze(0)
            .map_err(inference_error)?;
        let mut fields = fields;
        for layer in &self.layers {
            fields = layer.forward(&fields, &memory)?;
        }
        let fields = self
            .field_norm
            .forward(&fields.squeeze(0).map_err(inference_error)?)
            .map_err(inference_error)?;

        let prior_scale = self.prior_scale.min(100f32.ln()).exp();
        let joint_scale = self.joint_scale.min(100f32.ln()).exp();
        let gate = candle_nn::ops::sigmoid(
            &Tensor::from_vec(vec![self.residual_gate], 1, hidden.device())
                .map_err(inference_error)?,
        )
        .map_err(inference_error)?
        .to_scalar::<f32>()
        .map_err(inference_error)?;

        let mut output = Vec::with_capacity(split.len());
        for (index, options) in split.iter().enumerate() {
            let question_vector = question_vectors
                .narrow(0, index, 1)
                .map_err(inference_error)?
                .squeeze(0)
                .map_err(inference_error)?;
            let field = fields
                .narrow(0, index, 1)
                .map_err(inference_error)?
                .squeeze(0)
                .map_err(inference_error)?;
            let anchor = normalize(&question_vector.add(&global).map_err(inference_error)?)?;
            let lexical = &lexical_options[index];
            let lexical_anchor = normalize_rows(lexical)?;
            let prior = lexical_anchor
                .matmul(&anchor.unsqueeze(1).map_err(inference_error)?)
                .map_err(inference_error)?
                .affine(f64::from(prior_scale), 0.0)
                .map_err(inference_error)?
                .squeeze(1)
                .map_err(inference_error)?;
            let options = options.apply(&self.option_norm).map_err(inference_error)?;
            let repeated = field
                .unsqueeze(0)
                .map_err(inference_error)?
                .broadcast_as(options.dims())
                .map_err(inference_error)?;
            let cosine = cosine_rows(&repeated, &options)?;
            let features = Tensor::cat(
                &[
                    &repeated,
                    &options,
                    &repeated.broadcast_mul(&options).map_err(inference_error)?,
                    &repeated
                        .broadcast_sub(&options)
                        .map_err(inference_error)?
                        .abs()
                        .map_err(inference_error)?,
                ],
                D::Minus1,
            )
            .map_err(inference_error)?;
            let residual = self
                .residual1
                .forward(&features)
                .map_err(inference_error)?
                .gelu_erf()
                .map_err(inference_error)?;
            let residual = self
                .residual2
                .forward(&residual)
                .map_err(inference_error)?
                .squeeze(1)
                .map_err(inference_error)?;
            let joint = cosine
                .affine(f64::from(joint_scale), 0.0)
                .map_err(inference_error)?
                .add(&residual)
                .map_err(inference_error)?;
            let logits = prior
                .add(
                    &joint
                        .affine(f64::from(gate), 0.0)
                        .map_err(inference_error)?,
                )
                .map_err(inference_error)?;
            output.push(
                logits
                    .to_dtype(DType::F32)
                    .map_err(inference_error)?
                    .to_vec1::<f32>()
                    .map_err(inference_error)?,
            );
        }
        Ok(output)
    }
}

fn mean_span(values: &Tensor, start: usize, end: usize) -> Result<Tensor, ModelError> {
    values
        .narrow(0, start, end - start)
        .map_err(inference_error)?
        .mean(0)
        .map_err(inference_error)
}

fn normalize(values: &Tensor) -> Result<Tensor, ModelError> {
    let norm = values
        .sqr()
        .map_err(inference_error)?
        .sum_keepdim(0)
        .map_err(inference_error)?
        .sqrt()
        .map_err(inference_error)?
        .clamp(1e-8, 1e30)
        .map_err(inference_error)?;
    values.broadcast_div(&norm).map_err(inference_error)
}

fn normalize_rows(values: &Tensor) -> Result<Tensor, ModelError> {
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
}

fn cosine_rows(a: &Tensor, b: &Tensor) -> Result<Tensor, ModelError> {
    let a = normalize_rows(a)?;
    let b = normalize_rows(b)?;
    a.broadcast_mul(&b)
        .map_err(inference_error)?
        .sum(1)
        .map_err(inference_error)
}

struct QuestionPlan {
    id: String,
    kind: QuestionKind,
    option_ids: Vec<String>,
    labels: Vec<String>,
    legend: IndexMap<String, Content>,
    question_span: (usize, usize),
    option_spans: Vec<(usize, usize)>,
    question_type: usize,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum QuestionKind {
    Choice,
    Score,
    Noul,
}

/// A Clef joint-schema decision model.
pub struct ClefModel {
    tokenizer: Tokenizer,
    backbone: Qwen35,
    head: JointHead,
    max_len: usize,
    device: Device,
}

impl ClefModel {
    pub fn load_dir(directory: &Path, options: ClefOptions) -> Result<Self, ModelError> {
        let mut config = Qwen35Config::load(&directory.join("config.json"))
            .map_err(|error| ModelError::load(format!("load Clef config: {error}")))?;
        let paths = safetensors_files(directory)
            .map_err(|error| ModelError::load(format!("list Clef weights: {error}")))?;
        let store = WeightStore::from_safetensors(&paths, options.dtype, &options.device)
            .map_err(|error| ModelError::load(error.to_string()))?;
        let prefix = detect_prefix(
            &store,
            &["model.language_model.", "model.", "language_model.", ""],
            "embed_tokens.weight",
        )
        .ok_or_else(|| ModelError::load("Clef has no embed_tokens.weight"))?;
        config.prefix = prefix;
        config.validate().map_err(ModelError::load)?;
        let backbone =
            Qwen35::load(config, &store).map_err(|error| ModelError::load(error.to_string()))?;
        let head_config: HeadConfig = serde_json::from_slice(
            &std::fs::read(directory.join("joint_head_config.json")).map_err(|error| {
                ModelError::load(format!("read joint_head_config.json: {error}"))
            })?,
        )
        .map_err(|error| ModelError::load(format!("parse joint_head_config.json: {error}")))?;
        let head = JointHead::load(directory, head_config, options.dtype, &options.device)?;
        let mut tokenizer = Tokenizer::from_file(directory.join("tokenizer.json"))
            .map_err(|error| ModelError::load(format!("load tokenizer: {error}")))?;
        tokenizer
            .with_truncation(None)
            .map_err(|error| ModelError::load(format!("disable truncation: {error}")))?;
        let _ = tokenizer.with_padding(None);
        Ok(Self {
            tokenizer,
            backbone,
            head,
            max_len: options.max_len.unwrap_or(DEFAULT_MAX_LEN),
            device: options.device,
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

    fn render(&self, value: Option<&Content>) -> String {
        match value {
            None | Some(Content::Null) => "null".to_owned(),
            Some(Content::String(text)) => text.clone(),
            Some(content) => canonical_json(&content_to_value(content)),
        }
    }

    fn state_text(&self, state: &State) -> String {
        match state.as_content() {
            Content::String(text) => text.clone(),
            content => canonical_json(&content_to_value(content)),
        }
    }

    fn options(
        &self,
        id: &str,
        question: &Question,
    ) -> Result<Vec<(String, Option<Value>)>, ModelError> {
        match question {
            Question::Noul(noul) => {
                let mut criteria = Map::new();
                criteria.insert("true".to_owned(), Value::String(NOUL_TRUE.to_owned()));
                criteria.insert("false".to_owned(), Value::String(NOUL_FALSE.to_owned()));
                if let Some(c) = &noul.criteria {
                    if let Some(value) = &c.is_true {
                        criteria.insert("true".to_owned(), content_to_value(value));
                    }
                    if let Some(value) = &c.is_false {
                        criteria.insert("false".to_owned(), content_to_value(value));
                    }
                }
                Ok(vec![
                    ("true".to_owned(), criteria.get("true").cloned()),
                    ("false".to_owned(), criteria.get("false").cloned()),
                ])
            }
            Question::Choice(choice) => {
                if choice.criteria.is_empty() {
                    return Err(ModelError::invalid_request(format!(
                        "question {id:?}: choice criteria must not be empty"
                    )));
                }
                let mut entries: Vec<_> = choice.criteria.iter().collect();
                entries.sort_by(|left, right| left.0.cmp(right.0));
                Ok(entries
                    .into_iter()
                    .map(|(key, value)| (key.clone(), Some(content_to_value(value))))
                    .collect())
            }
            Question::Score(score) => {
                if score.criteria.is_empty() {
                    return Err(ModelError::invalid_request(format!(
                        "question {id:?}: score criteria must not be empty"
                    )));
                }
                Ok(score
                    .criteria
                    .iter()
                    .enumerate()
                    .map(|(index, value)| (index.to_string(), Some(content_to_value(value))))
                    .collect())
            }
            Question::Box(_) => Err(ModelError::invalid_request(format!(
                "question {id:?}: Clef does not answer box questions"
            ))),
        }
    }

    fn encode_record(
        &self,
        state: &State,
        questions: &Questions,
    ) -> Result<(Vec<u32>, Vec<QuestionPlan>), ModelError> {
        let mut schema = self.encode("\n\nSCHEMA FIELDS:\n")?;
        let mut plans = Vec::with_capacity(questions.len());
        for (question_index, (id, question)) in questions.iter().enumerate() {
            let kind_name = match question {
                Question::Noul(_) => QUESTION_TYPES[0],
                Question::Choice(_) => QUESTION_TYPES[1],
                Question::Score(_) => QUESTION_TYPES[2],
                Question::Box(_) => unreachable!(),
            };
            schema.extend(self.encode(&format!(
                "\nFIELD {}\nID: {id}\nTYPE: {kind_name}\nINSTRUCTION: ",
                question_index + 1
            ))?);
            let question_start = schema.len();
            let instructions = match question.instructions() {
                None | Some(Content::Null) => id.clone(),
                Some(Content::String(text)) if text.is_empty() => id.clone(),
                Some(content) => self.render(Some(content)),
            };
            schema.extend(self.encode(&instructions)?);
            let question_end = schema.len();
            schema.extend(self.encode("\nALLOWED OPTIONS:\n")?);
            let options = self.options(id, question)?;
            let mut option_spans = Vec::with_capacity(options.len());
            let mut option_ids = Vec::with_capacity(options.len());
            for (option_index, (option_id, description)) in options.iter().enumerate() {
                schema.extend(self.encode(&format!("OPTION {}: ", option_index + 1))?);
                let option_start = schema.len();
                let mut semantics = Map::new();
                semantics.insert("option_id".to_owned(), Value::String(option_id.clone()));
                if let Some(description) = description {
                    semantics.insert("description".to_owned(), description.clone());
                }
                schema.extend(self.encode(&canonical_json(&Value::Object(semantics)))?);
                option_spans.push((option_start, schema.len()));
                option_ids.push(option_id.clone());
                schema.extend(self.encode("\n")?);
            }
            schema.extend(self.encode("END FIELD\n")?);
            let (kind, labels, legend) = match question {
                Question::Choice(choice) => (
                    QuestionKind::Choice,
                    choice.criteria.keys().cloned().collect(),
                    IndexMap::new(),
                ),
                Question::Score(score) => (
                    QuestionKind::Score,
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
                Question::Noul(_) => (QuestionKind::Noul, Vec::new(), IndexMap::new()),
                Question::Box(_) => unreachable!(),
            };
            plans.push(QuestionPlan {
                id: id.clone(),
                kind,
                option_ids,
                labels,
                legend,
                question_span: (question_start, question_end),
                option_spans,
                question_type: match kind {
                    QuestionKind::Noul => 0,
                    QuestionKind::Choice => 1,
                    QuestionKind::Score => 2,
                },
            });
        }
        let prefix = self.encode(&format!(
            "<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n<|im_start|>user\nSTATE:\n"
        ))?;
        let suffix = self.encode(
            "\n<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\nJOINT SCHEMA DECISIONS:",
        )?;
        let state_ids = self.encode(&self.state_text(state))?;
        let fixed = prefix.len() + schema.len() + suffix.len();
        if fixed > self.max_len {
            return Err(ModelError::invalid_request(format!(
                "the schema needs {fixed} tokens before the state; the limit is {}",
                self.max_len
            )));
        }
        let state_ids = &state_ids[..state_ids.len().min(self.max_len - fixed)];
        let offset = prefix.len() + state_ids.len();
        let mut ids = Vec::with_capacity(offset + schema.len() + suffix.len());
        ids.extend_from_slice(&prefix);
        ids.extend_from_slice(state_ids);
        ids.extend_from_slice(&schema);
        ids.extend_from_slice(&suffix);
        for plan in &mut plans {
            plan.question_span = (plan.question_span.0 + offset, plan.question_span.1 + offset);
            plan.option_spans = plan
                .option_spans
                .iter()
                .map(|(start, end)| (*start + offset, *end + offset))
                .collect();
        }
        for plan in &plans {
            if plan.question_span.1 <= plan.question_span.0 {
                return Err(ModelError::invalid_request(format!(
                    "question {:?}: instructions produced no tokens",
                    plan.id
                )));
            }
        }
        Ok((ids, plans))
    }

    fn answer(&self, plan: &QuestionPlan, logits: &[f32]) -> Answer {
        let probabilities = softmax(logits, 1.0);
        match plan.kind {
            QuestionKind::Choice => {
                let ordered: Vec<f32> = plan
                    .labels
                    .iter()
                    .map(|label| {
                        plan.option_ids
                            .iter()
                            .position(|option| option == label)
                            .map(|index| probabilities[index])
                            .unwrap_or(0.0)
                    })
                    .collect();
                choice_answer(&plan.labels, &ordered)
            }
            QuestionKind::Score => score_answer(&plan.legend, &probabilities),
            QuestionKind::Noul => {
                let true_probability = plan
                    .option_ids
                    .iter()
                    .position(|option| option == "true")
                    .map(|index| probabilities[index])
                    .unwrap_or(0.5);
                let false_probability = plan
                    .option_ids
                    .iter()
                    .position(|option| option == "false")
                    .map(|index| probabilities[index])
                    .unwrap_or(0.5);
                noul_answer(false_probability, true_probability)
            }
        }
    }
}

impl sys1_core::DecisionModel for ClefModel {
    fn predict(&self, state: &State, questions: &Questions) -> Result<DecisionOutput, ModelError> {
        if questions.is_empty() {
            return Err(ModelError::invalid_request("questions must not be empty"));
        }
        let (ids, plans) = self.encode_record(state, questions)?;
        let input =
            Tensor::from_vec(ids.clone(), (1, ids.len()), &self.device).map_err(inference_error)?;
        let hidden = self.backbone.forward(&input).map_err(inference_error)?;
        let question_spans: Vec<_> = plans.iter().map(|plan| plan.question_span).collect();
        let option_spans: Vec<_> = plans.iter().map(|plan| plan.option_spans.clone()).collect();
        let types: Vec<_> = plans.iter().map(|plan| plan.question_type).collect();
        let logits = self.head.forward(
            &hidden,
            &ids,
            &question_spans,
            &option_spans,
            &types,
            self.backbone.output_embedding(),
        )?;
        let mut answers = IndexMap::with_capacity(plans.len());
        for (plan, logits) in plans.iter().zip(logits) {
            answers.insert(plan.id.clone(), self.answer(plan, &logits));
        }
        Ok(DecisionOutput::new(
            Answers::new(answers),
            Usage::new(ids.len() as u64, 0),
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
