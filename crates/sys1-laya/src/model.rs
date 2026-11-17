use crate::calibration::ConfidenceMode;
use crate::config::{LayaConfig, ModernBertConfig};
use crate::long;
use crate::modernbert::Encoder;
use crate::prepare::{Item, PreparedRequest, deduplicate, prepare_request};
use crate::response::build_response;
use crate::router::LayaRouterOptions;
use crate::tokenizer::LayaTokenizer;
use candle_core::{D, DType, Device, IndexOp, Tensor};
use candle_nn::{Embedding, LayerNorm, Linear, VarBuilder, embedding, layer_norm};
use std::path::Path;
use sys1_core::{
    DecisionModel, DecisionOutput, LongContextOptions, ModelError, Prediction, Questions,
};
use sys1_protocol::State;

/// One question's logits and, when the checkpoint has one, the act-head
/// probability.
type ScoredRow = (Vec<f32>, Option<f32>);

/// Runtime options for a Laya checkpoint.
#[derive(Clone)]
pub struct LayaOptions {
    pub device: Device,
    pub dtype: DType,
    /// Override the checkpoint's `max_len`.
    pub max_len: Option<usize>,
    pub confidence_mode: ConfidenceMode,
    /// Use exact sliding-window attention for local layers. Defaults to true.
    pub local_window: bool,
    /// Bound on `batch * heads * seq_len²` for one global-attention batch.
    /// Larger batches are split into chunks. Defaults to 64M elements.
    pub max_attention_elements: usize,
    /// Routing settings used when these options build a `LayaRouter`.
    pub router: LayaRouterOptions,
    /// Router-wide language hint, e.g. `en` or `ro`.
    pub lang_guess: Option<String>,
}

impl Default for LayaOptions {
    fn default() -> Self {
        Self {
            device: Device::Cpu,
            dtype: DType::F32,
            max_len: None,
            confidence_mode: ConfidenceMode::TypeSafe,
            local_window: true,
            max_attention_elements: crate::DEFAULT_MAX_ATTENTION_ELEMENTS,
            router: LayaRouterOptions::default(),
            lang_guess: None,
        }
    }
}

struct HeadLayer {
    qkv: Linear,
    projection: Linear,
    norm1: LayerNorm,
    norm2: LayerNorm,
    linear1: Linear,
    linear2: Linear,
    heads: usize,
    compute_dtype: DType,
}

impl HeadLayer {
    fn load(vb: VarBuilder, hidden: usize, compute_dtype: DType) -> candle_core::Result<Self> {
        Ok(Self {
            qkv: Linear::new(
                vb.get((hidden * 3, hidden), "self_attn.in_proj_weight")?
                    .to_dtype(compute_dtype)?,
                Some(
                    vb.get(hidden * 3, "self_attn.in_proj_bias")?
                        .to_dtype(compute_dtype)?,
                ),
            ),
            projection: linear_dtype(hidden, hidden, vb.pp("self_attn.out_proj"), compute_dtype)?,
            norm1: layer_norm(hidden, 1e-5, vb.pp("norm1"))?,
            norm2: layer_norm(hidden, 1e-5, vb.pp("norm2"))?,
            linear1: linear_dtype(hidden, hidden * 4, vb.pp("linear1"), compute_dtype)?,
            linear2: linear_dtype(hidden * 4, hidden, vb.pp("linear2"), compute_dtype)?,
            heads: hidden / 64,
            compute_dtype,
        })
    }

    fn forward(&self, xs: &Tensor, mask: &Tensor) -> candle_core::Result<Tensor> {
        let (batch, length, hidden) = xs.dims3()?;
        let size = hidden / self.heads;
        let qkv = xs
            .apply(&self.norm1)?
            .to_dtype(self.compute_dtype)?
            .apply(&self.qkv)?
            .reshape((batch, length, 3, self.heads, size))?
            .permute((2, 0, 3, 1, 4))?;
        let q = qkv.get(0)?;
        let k = qkv.get(1)?;
        let v = qkv.get(2)?;
        let scale = (size as f64).powf(-0.5);
        let scores = (q * scale)?.matmul(&k.transpose(D::Minus2, D::Minus1)?)?;
        let scores = scores.to_dtype(mask.dtype())?.broadcast_add(mask)?;
        let attention = attention_softmax(&scores)?
            .to_dtype(v.dtype())?
            .matmul(&v)?
            .transpose(1, 2)?
            .reshape((batch, length, hidden))?
            .apply(&self.projection)?
            .to_dtype(xs.dtype())?;
        let xs = (xs + attention)?;
        let feed_forward = xs
            .apply(&self.norm2)?
            .to_dtype(self.compute_dtype)?
            .apply(&self.linear1)?
            .relu()?
            .apply(&self.linear2)?
            .to_dtype(xs.dtype())?;
        xs + feed_forward
    }
}

struct ActHead {
    input: Linear,
    output: Linear,
    compute_dtype: DType,
}

impl ActHead {
    fn load(
        vb: VarBuilder,
        hidden: usize,
        compute_dtype: DType,
    ) -> candle_core::Result<Option<Self>> {
        if !([
            "act_head.0.weight",
            "act_head.0.bias",
            "act_head.2.weight",
            "act_head.2.bias",
        ]
        .iter()
        .all(|name| vb.contains_tensor(name)))
        {
            return Ok(None);
        }
        Ok(Some(Self {
            input: Linear::new(
                vb.get((256, hidden + 4), "act_head.0.weight")?
                    .to_dtype(compute_dtype)?,
                Some(vb.get(256, "act_head.0.bias")?.to_dtype(compute_dtype)?),
            ),
            output: Linear::new(
                vb.get((2, 256), "act_head.2.weight")?
                    .to_dtype(compute_dtype)?,
                Some(vb.get(2, "act_head.2.bias")?.to_dtype(compute_dtype)?),
            ),
            compute_dtype,
        }))
    }

    fn forward(&self, pooled: &Tensor, logits: &[f32]) -> candle_core::Result<f32> {
        if logits.is_empty() {
            return Ok(0.5);
        }
        let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut probabilities: Vec<f32> = logits.iter().map(|value| (value - max).exp()).collect();
        let total: f32 = probabilities.iter().sum();
        if total > 0.0 {
            probabilities.iter_mut().for_each(|value| *value /= total);
        }
        let mut top = probabilities.iter().copied().fold(
            (f32::NEG_INFINITY, f32::NEG_INFINITY),
            |(first, second), value| {
                if value > first {
                    (value, first)
                } else if value > second {
                    (first, value)
                } else {
                    (first, second)
                }
            },
        );
        if probabilities.len() == 1 {
            top.1 = 0.0;
        }
        let count = probabilities.len().max(2);
        let entropy: f32 = probabilities
            .iter()
            .map(|value| -value * value.max(1e-9).ln())
            .sum::<f32>()
            / (count as f32).ln();
        let features = Tensor::new(
            &[top.0, top.0 - top.1, entropy, count as f32 / 255.0],
            pooled.device(),
        )?
        .to_dtype(self.compute_dtype)?
        .unsqueeze(0)?;
        let pooled = pooled.unsqueeze(0)?.to_dtype(self.compute_dtype)?;
        let input = Tensor::cat(&[&pooled, &features], D::Minus1)?;
        let hidden = input.apply(&self.input)?.gelu_erf()?;
        let logits = hidden.apply(&self.output)?.to_dtype(DType::F32)?;
        let probabilities = candle_nn::ops::softmax(&logits, D::Minus1)?;
        let values = probabilities.to_vec2::<f32>()?;
        Ok(values[0][0])
    }
}

fn attention_softmax(scores: &Tensor) -> candle_core::Result<Tensor> {
    if matches!(scores.dtype(), DType::F16 | DType::BF16) {
        candle_nn::ops::softmax(&scores.to_dtype(DType::F32)?, D::Minus1)?.to_dtype(scores.dtype())
    } else {
        candle_nn::ops::softmax(scores, D::Minus1)
    }
}

/// A loaded Laya checkpoint.
pub struct LayaModel {
    tokenizer: LayaTokenizer,
    encoder: Encoder,
    head: Vec<HeadLayer>,
    type_embedding: Embedding,
    scorer_norm: LayerNorm,
    scorer_in: Linear,
    scorer_out: Linear,
    act_head: Option<ActHead>,
    config: LayaConfig,
    device: Device,
    dtype: DType,
    compute_dtype: DType,
    num_heads: usize,
    max_attention_elements: usize,
    confidence_mode: ConfidenceMode,
}

impl LayaModel {
    pub fn load_dir(directory: &Path, options: LayaOptions) -> Result<Self, ModelError> {
        let mut config =
            LayaConfig::load(&directory.join("rl_agent_config.json")).map_err(ModelError::load)?;
        let encoder_config = ModernBertConfig::load(&directory.join("encoder/config.json"))
            .map_err(ModelError::load)?;

        if let Some(max_len) = options.max_len {
            if max_len == 0 || max_len > encoder_config.max_position_embeddings() {
                return Err(ModelError::load(format!(
                    "max_len {max_len} is outside 1..={}",
                    encoder_config.max_position_embeddings()
                )));
            }
            config.max_len = max_len;
        }
        if config.max_len == 0 || config.max_len > encoder_config.max_position_embeddings() {
            return Err(ModelError::load(format!(
                "checkpoint max_len {} is outside 1..={}",
                config.max_len,
                encoder_config.max_position_embeddings()
            )));
        }
        if config.head_max_len == 0 {
            return Err(ModelError::load("head_max_len must be greater than zero"));
        }

        let tokenizer = LayaTokenizer::load(directory).map_err(ModelError::load)?;
        let compute_dtype = options.dtype;
        let model_dtype = if options.dtype == DType::F16 && options.device.is_cuda() {
            DType::F32
        } else {
            options.dtype
        };

        let weights = directory.join("model.safetensors");
        if !weights.is_file() {
            return Err(ModelError::load(format!(
                "missing weights at {}",
                weights.display()
            )));
        }
        // SAFETY: the safetensors file is treated as immutable for the lifetime
        // of the mmap. The cache layer never rewrites a verified model file.
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[weights], model_dtype, &options.device)
                .map_err(|error| ModelError::load(error.to_string()))?
        };

        let encoder_vb = vb.clone().rename_f(|name| {
            name.strip_prefix("model.")
                .map(|name| format!("encoder.{name}"))
                .unwrap_or_else(|| name.to_owned())
        });
        let encoder = Encoder::load(
            encoder_vb,
            &encoder_config,
            compute_dtype,
            options.local_window,
        )
        .map_err(|error| ModelError::load(error.to_string()))?;

        let head = (0..config.head_layers)
            .map(|index| {
                HeadLayer::load(
                    vb.pp(format!("head.layers.{index}")),
                    encoder_config.hidden_size(),
                    compute_dtype,
                )
            })
            .collect::<candle_core::Result<Vec<_>>>()
            .map_err(|error| ModelError::load(error.to_string()))?;

        let hidden = encoder_config.hidden_size();
        let type_embedding = embedding(3, hidden, vb.pp("type_emb"))
            .map_err(|error| ModelError::load(error.to_string()))?;
        let scorer_norm = layer_norm(hidden, 1e-5, vb.pp("scorer.0"))
            .map_err(|error| ModelError::load(error.to_string()))?;
        let scorer_in = linear_dtype(hidden, hidden, vb.pp("scorer.1"), compute_dtype)
            .map_err(|error| ModelError::load(error.to_string()))?;
        let scorer_out = linear_dtype(hidden, 1, vb.pp("scorer.3"), compute_dtype)
            .map_err(|error| ModelError::load(error.to_string()))?;
        let act_head = ActHead::load(vb.clone(), hidden, compute_dtype)
            .map_err(|error| ModelError::load(error.to_string()))?;

        Ok(Self {
            tokenizer,
            encoder,
            head,
            type_embedding,
            scorer_norm,
            scorer_in,
            scorer_out,
            act_head,
            config,
            device: options.device,
            dtype: model_dtype,
            compute_dtype,
            num_heads: encoder_config.num_attention_heads,
            max_attention_elements: options.max_attention_elements.max(1),
            confidence_mode: options.confidence_mode,
        })
    }

    fn forward(&self, prepared: &[PreparedRequest]) -> Result<Vec<ScoredRow>, ModelError> {
        let items: Vec<&Item> = prepared
            .iter()
            .flat_map(|request| request.items.iter())
            .collect();
        let (unique, indices) = deduplicate(&items);
        if unique.is_empty() || unique.iter().all(|item| item.ids.is_empty()) {
            return Ok(Vec::new());
        }

        let output = self.forward_items(&unique)?;

        Ok(indices
            .into_iter()
            .map(|index| output[index].clone())
            .collect())
    }

    fn forward_items(&self, unique: &[&Item]) -> Result<Vec<ScoredRow>, ModelError> {
        let batch = unique.len();
        let length = unique.iter().map(|item| item.ids.len()).max().unwrap_or(0);
        if batch == 0 || length == 0 {
            return Ok(Vec::new());
        }

        // Global layers still materialize a seq×seq score tensor. Bound it
        // independently of the server's request batch size.
        let per_item = self
            .num_heads
            .saturating_mul(length)
            .saturating_mul(length)
            .max(1);
        let max_batch = (self.max_attention_elements / per_item).max(1);
        if batch > max_batch {
            let mut output = Vec::with_capacity(batch);
            for chunk in unique.chunks(max_batch) {
                output.extend(self.forward_items(chunk)?);
            }
            return Ok(output);
        }

        let mut ids = vec![self.tokenizer.pad_id; batch * length];
        let mut mask = vec![0f32; batch * length];
        let mut kinds = Vec::with_capacity(batch);
        for (row, item) in unique.iter().enumerate() {
            let start = row * length;
            ids[start..start + item.ids.len()].copy_from_slice(&item.ids);
            mask[start..start + item.ids.len()].fill(1.0);
            kinds.push(item.question.kind as u32);
        }

        let ids = Tensor::from_vec(ids, (batch, length), &self.device).map_err(inference_error)?;
        let head_mask: Vec<_> = mask
            .iter()
            .map(|value| {
                if *value == 0.0 {
                    f32::NEG_INFINITY
                } else {
                    0.0
                }
            })
            .collect();
        let attention_mask = Tensor::from_vec(mask, (batch, length), &self.device)
            .map_err(inference_error)?
            .to_dtype(self.dtype)
            .map_err(inference_error)?;
        let head_mask = Tensor::from_vec(head_mask, (batch, 1, 1, length), &self.device)
            .map_err(inference_error)?
            .to_dtype(self.compute_dtype)
            .map_err(inference_error)?;
        let type_ids = Tensor::from_vec(kinds, batch, &self.device).map_err(inference_error)?;
        let has_padding = unique.iter().any(|item| item.ids.len() != length);

        let mut hidden = self
            .encoder
            .forward(&ids, &attention_mask, has_padding)
            .map_err(inference_error)?;
        hidden = hidden
            .broadcast_add(
                &type_ids
                    .apply(&self.type_embedding)
                    .map_err(inference_error)?
                    .unsqueeze(1)
                    .map_err(inference_error)?,
            )
            .map_err(inference_error)?;
        for (index, layer) in self.head.iter().enumerate() {
            hidden = layer.forward(&hidden, &head_mask).map_err(|error| {
                ModelError::inference(format!("decision head layer {index}: {error}"))
            })?;
        }

        let logits = if batch >= 8 && !self.device.is_cpu() {
            self.score_batched(&hidden, unique, length)?
        } else {
            self.score_rows(&hidden, unique)?
        };
        let mut output = Vec::with_capacity(logits.len());
        for (row, values) in logits.into_iter().enumerate() {
            let action = match &self.act_head {
                Some(head) => {
                    let pooled = hidden
                        .i(row)
                        .map_err(inference_error)?
                        .i(0)
                        .map_err(inference_error)?;
                    Some(head.forward(&pooled, &values).map_err(inference_error)?)
                }
                None => None,
            };
            output.push((values, action));
        }
        Ok(output)
    }

    fn score_rows(&self, hidden: &Tensor, items: &[&Item]) -> Result<Vec<Vec<f32>>, ModelError> {
        let mut output = Vec::with_capacity(items.len());
        for (row, item) in items.iter().enumerate() {
            let markers = Tensor::from_vec(
                item.markers
                    .iter()
                    .map(|value| *value as u32)
                    .collect::<Vec<_>>(),
                item.markers.len(),
                &self.device,
            )
            .map_err(inference_error)?;
            let selected = hidden
                .i(row)
                .map_err(inference_error)?
                .index_select(&markers, 0)
                .map_err(inference_error)?;
            output.push(self.score(&selected)?);
        }
        Ok(output)
    }

    fn score_batched(
        &self,
        hidden: &Tensor,
        items: &[&Item],
        length: usize,
    ) -> Result<Vec<Vec<f32>>, ModelError> {
        let hidden_size = hidden.dim(D::Minus1).map_err(inference_error)?;
        let counts: Vec<_> = items.iter().map(|item| item.markers.len()).collect();
        let marker_count = counts.iter().sum();
        let mut markers = Vec::with_capacity(marker_count);
        for (row, item) in items.iter().enumerate() {
            markers.extend(
                item.markers
                    .iter()
                    .map(|position| (row * length + position) as u32),
            );
        }
        let markers =
            Tensor::from_vec(markers, marker_count, &self.device).map_err(inference_error)?;
        let selected = hidden
            .reshape((items.len() * length, hidden_size))
            .map_err(inference_error)?
            .index_select(&markers, 0)
            .map_err(inference_error)?;
        let logits = self.score(&selected)?;
        let mut offset = 0;
        Ok(counts
            .into_iter()
            .map(|count| {
                let values = logits[offset..offset + count].to_vec();
                offset += count;
                values
            })
            .collect())
    }

    fn score(&self, selected: &Tensor) -> Result<Vec<f32>, ModelError> {
        selected
            .apply(&self.scorer_norm)
            .map_err(inference_error)?
            .to_dtype(self.compute_dtype)
            .map_err(inference_error)?
            .apply(&self.scorer_in)
            .map_err(inference_error)?
            .gelu_erf()
            .map_err(inference_error)?
            .apply(&self.scorer_out)
            .map_err(inference_error)?
            .squeeze(1)
            .map_err(inference_error)?
            .to_dtype(DType::F32)
            .map_err(inference_error)?
            .to_vec1::<f32>()
            .map_err(inference_error)
    }
}

impl LayaModel {
    fn predict_prepared(
        &self,
        prepared: Vec<PreparedRequest>,
    ) -> Vec<Result<DecisionOutput, ModelError>> {
        match self.forward(&prepared) {
            Ok(outputs) => {
                let mut outputs = outputs.into_iter();
                prepared
                    .into_iter()
                    .map(|request| {
                        Ok(build_response(
                            &self.config,
                            self.confidence_mode,
                            request,
                            &mut outputs,
                        ))
                    })
                    .collect()
            }
            Err(error) => prepared.iter().map(|_| Err(error.clone())).collect(),
        }
    }
}

impl DecisionModel for LayaModel {
    fn predict(&self, state: &State, questions: &Questions) -> Result<DecisionOutput, ModelError> {
        self.predict_batch(&[Prediction::new(state, questions)])
            .into_iter()
            .next()
            .expect("one prediction result")
    }

    fn predict_long(
        &self,
        state: &State,
        questions: &Questions,
        options: LongContextOptions,
    ) -> Result<DecisionOutput, ModelError> {
        let plan = long::plan(&self.tokenizer, &self.config, state, questions, options)?;
        if plan.single {
            let mut output = self.predict(state, questions)?;
            output.scored_windows = 1;
            return Ok(output);
        }
        tracing::debug!(
            window = plan.window,
            stride = plan.stride,
            windows = plan.windows.len(),
            "scanning long state"
        );
        let prepared = long::prepare_windows(&self.tokenizer, &self.config, &plan, questions)?;
        let outputs = long::score_windows(prepared, options.batch_size, |batch| {
            self.predict_prepared(batch)
        })?;
        long::aggregate(outputs, questions, &plan.windows)
    }

    fn supports_cross_request_batching(&self) -> bool {
        // A CPU forward already uses the thread pool; serializing concurrent
        // requests through one batched forward lowers throughput in practice.
        !self.device.is_cpu()
    }

    fn predict_batch(
        &self,
        requests: &[Prediction<'_>],
    ) -> Vec<Result<DecisionOutput, ModelError>> {
        let mut slots: Vec<Option<Result<DecisionOutput, ModelError>>> =
            (0..requests.len()).map(|_| None).collect();
        let mut prepared = Vec::new();
        let mut prepared_slots = Vec::new();

        for (index, request) in requests.iter().enumerate() {
            match prepare_request(
                &self.tokenizer,
                &self.config,
                request.state,
                request.questions,
            ) {
                Ok(request) => {
                    prepared.push(request);
                    prepared_slots.push(index);
                }
                Err(error) => slots[index] = Some(Err(error)),
            }
        }

        if !prepared.is_empty() {
            for (result, slot) in self
                .predict_prepared(prepared)
                .into_iter()
                .zip(prepared_slots)
            {
                slots[slot] = Some(result);
            }
        }

        slots
            .into_iter()
            .map(|slot| slot.expect("every request slot is filled"))
            .collect()
    }
}

fn linear_dtype(
    input: usize,
    output: usize,
    vb: VarBuilder,
    dtype: DType,
) -> candle_core::Result<Linear> {
    Ok(Linear::new(
        vb.get((output, input), "weight")?.to_dtype(dtype)?,
        Some(vb.get(output, "bias")?.to_dtype(dtype)?),
    ))
}

fn inference_error(error: candle_core::Error) -> ModelError {
    ModelError::inference(error.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::calibration::TYPES;
    use indexmap::IndexMap;
    use serde_json::Value;
    use std::env;
    use std::fs;
    use sys1_protocol::Question;

    #[test]
    #[ignore = "requires a cached Laya checkpoint and official golden vectors"]
    fn matches_official_golden_case() {
        let root = env::var("SYS1_LAYA_TEST_ROOT")
            .map(std::path::PathBuf::from)
            .expect("set SYS1_LAYA_TEST_ROOT");
        let golden = env::var("SYS1_LAYA_GOLDEN_DIR")
            .map(std::path::PathBuf::from)
            .expect("set SYS1_LAYA_GOLDEN_DIR");
        let case: Value = serde_json::from_slice(
            &fs::read(golden.join("english").join("case_dict_state.json")).expect("golden case"),
        )
        .expect("golden JSON");

        let state = State::from_value(case["state"].clone()).expect("state");
        let questions: IndexMap<String, Question> =
            serde_json::from_value(case["questions"].clone()).expect("questions");
        let questions = Questions::from(questions);

        let model = LayaModel::load_dir(
            &root,
            LayaOptions {
                confidence_mode: ConfidenceMode::LayaNative,
                ..LayaOptions::default()
            },
        )
        .expect("load Laya checkpoint");

        let prepared = prepare_request(&model.tokenizer, &model.config, &state, &questions)
            .expect("prepare golden case");
        let expected_ids: Vec<u32> =
            serde_json::from_value(case["per_question"][0]["input_ids"].clone())
                .expect("expected ids");
        let expected_markers: Vec<usize> =
            serde_json::from_value(case["per_question"][0]["markers"].clone())
                .expect("expected markers");
        assert_eq!(prepared.items[0].ids, expected_ids, "input_ids");
        assert_eq!(prepared.items[0].markers, expected_markers, "markers");

        let logits = model
            .forward(std::slice::from_ref(&prepared))
            .expect("forward");
        let expected: Vec<Vec<f32>> =
            serde_json::from_value(case["logits"].clone()).expect("expected logits");
        assert_eq!(logits.len(), expected.len());
        for (row, expected_row) in expected.iter().enumerate() {
            let actual_row = &logits[row].0;
            assert_eq!(actual_row.len(), expected_row.len());
            for (column, expected_value) in expected_row.iter().enumerate() {
                let actual = actual_row[column];
                assert!(
                    (actual - expected_value).abs() <= 1e-3,
                    "logits[{row}][{column}]: expected {expected_value}, got {actual}"
                );
            }
        }
    }

    #[test]
    fn type_names_match_laya() {
        assert_eq!(TYPES, ["choice", "score", "noul"]);
    }
}
