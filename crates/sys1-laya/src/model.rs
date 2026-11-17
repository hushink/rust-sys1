use crate::calibration::{ConfidenceMode, argmax, confidence, probability, round4};
use crate::config::{LayaConfig, ModernBertConfig};
use crate::modernbert::Encoder;
use crate::prepare::{Criteria, Item, PreparedRequest, prepare_request};
use crate::tokenizer::LayaTokenizer;
use candle_core::{D, DType, Device, IndexOp, Tensor};
use candle_nn::{Embedding, LayerNorm, Linear, VarBuilder, embedding, layer_norm};
use indexmap::IndexMap;
use std::collections::HashMap;
use std::path::Path;
use sys1_core::{Answers, DecisionModel, DecisionOutput, ModelError, Prediction, Questions};
use sys1_protocol::{Answer, ChoiceAnswer, NoulAnswer, ScoreAnswer, State, Usage};

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

        Ok(Self {
            tokenizer,
            encoder,
            head,
            type_embedding,
            scorer_norm,
            scorer_in,
            scorer_out,
            config,
            device: options.device,
            dtype: model_dtype,
            compute_dtype,
            num_heads: encoder_config.num_attention_heads,
            max_attention_elements: options.max_attention_elements.max(1),
            confidence_mode: options.confidence_mode,
        })
    }

    fn forward(&self, prepared: &[PreparedRequest]) -> Result<Vec<Vec<f32>>, ModelError> {
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

    fn forward_items(&self, unique: &[&Item]) -> Result<Vec<Vec<f32>>, ModelError> {
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

        if batch >= 8 && !self.device.is_cpu() {
            self.score_batched(&hidden, unique, length)
        } else {
            self.score_rows(&hidden, unique)
        }
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

    fn response(
        &self,
        request: PreparedRequest,
        logits: &mut impl Iterator<Item = Vec<f32>>,
    ) -> DecisionOutput {
        let input_tokens = request.items.iter().map(|item| item.ids.len() as u64).sum();
        let mut answers: IndexMap<String, Answer> = IndexMap::with_capacity(request.items.len());

        for item in request.items {
            let values = logits.next().expect("one logits vector per item");
            let temperature = self
                .config
                .temperature_for(item.question.kind, values.len());
            let probabilities = probability(&values, temperature);
            let conf = confidence(self.confidence_mode, &probabilities);

            let answer = match item.question.kind {
                0 => {
                    let index = argmax(&probabilities);
                    let mut distribution = IndexMap::with_capacity(probabilities.len());
                    for (label, value) in item.question.labels.iter().zip(probabilities.iter()) {
                        distribution.insert(label.clone(), round4(*value as f64));
                    }
                    Answer::Choice(ChoiceAnswer::new(
                        item.question.labels[index].clone(),
                        round4(conf),
                        distribution,
                    ))
                }
                1 => {
                    let score: f32 = probabilities
                        .iter()
                        .enumerate()
                        .map(|(index, value)| index as f32 * value)
                        .sum();
                    let mut distribution = IndexMap::with_capacity(probabilities.len());
                    let mut legend = IndexMap::with_capacity(probabilities.len());
                    for (index, value) in probabilities.iter().enumerate() {
                        distribution.insert(index.to_string(), round4(*value as f64));
                        if let Criteria::Score(criteria) = &item.question.criteria
                            && let Some(content) = criteria.get(index)
                        {
                            legend.insert(index.to_string(), content.clone());
                        }
                    }
                    Answer::Score(ScoreAnswer::new(
                        round4(score as f64),
                        round4(conf),
                        legend,
                        distribution,
                    ))
                }
                _ => {
                    let true_probability = probabilities.get(1).copied().unwrap_or(0.0);
                    Answer::Noul(NoulAnswer::new(round4(true_probability as f64)))
                }
            };
            answers.insert(item.question.id, answer);
        }

        DecisionOutput::new(Answers::new(answers), Usage::new(input_tokens, 0))
    }
}

impl DecisionModel for LayaModel {
    fn predict(&self, state: &State, questions: &Questions) -> Result<DecisionOutput, ModelError> {
        self.predict_batch(&[Prediction::new(state, questions)])
            .into_iter()
            .next()
            .expect("one prediction result")
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
            match self.forward(&prepared) {
                Ok(outputs) => {
                    let mut outputs = outputs.into_iter();
                    for (request, slot) in prepared.into_iter().zip(prepared_slots) {
                        slots[slot] = Some(Ok(self.response(request, &mut outputs)));
                    }
                }
                Err(error) => {
                    for slot in prepared_slots {
                        slots[slot] = Some(Err(error.clone()));
                    }
                }
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

struct ItemKey<'a> {
    kind: usize,
    ids: &'a [u32],
    markers: &'a [usize],
}

impl PartialEq for ItemKey<'_> {
    fn eq(&self, other: &Self) -> bool {
        self.kind == other.kind && self.ids == other.ids && self.markers == other.markers
    }
}

impl Eq for ItemKey<'_> {}

impl std::hash::Hash for ItemKey<'_> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.kind.hash(state);
        self.ids.hash(state);
        self.markers.hash(state);
    }
}

fn deduplicate<'a>(items: &[&'a Item]) -> (Vec<&'a Item>, Vec<usize>) {
    let mut unique = Vec::with_capacity(items.len());
    let mut indices = Vec::with_capacity(items.len());
    let mut seen: HashMap<ItemKey<'a>, usize> = HashMap::with_capacity(items.len());
    for item in items {
        let key = ItemKey {
            kind: item.question.kind,
            ids: &item.ids,
            markers: &item.markers,
        };
        let index = *seen.entry(key).or_insert_with(|| {
            let index = unique.len();
            unique.push(*item);
            index
        });
        indices.push(index);
    }
    (unique, indices)
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
            assert_eq!(logits[row].len(), expected_row.len());
            for (column, expected_value) in expected_row.iter().enumerate() {
                let actual = logits[row][column];
                assert!(
                    (actual - expected_value).abs() <= 1e-3,
                    "logits[{row}][{column}]: expected {expected_value}, got {actual}"
                );
            }
        }
    }

    #[test]
    fn deduplicates_identical_inputs_without_losing_order() {
        let make = |id: &str, kind: usize, ids: &[u32]| Item {
            question: crate::prepare::PreparedQuestion {
                id: id.to_owned(),
                kind,
                instructions: String::new(),
                criteria: Criteria::Noul,
                labels: Vec::new(),
                options: Vec::new(),
            },
            ids: ids.to_vec(),
            markers: vec![1],
        };
        let first = make("first", 0, &[1, 2, 3]);
        let duplicate = make("duplicate", 0, &[1, 2, 3]);
        let other_kind = make("other-kind", 1, &[1, 2, 3]);
        let items = vec![&first, &duplicate, &other_kind, &first];

        let (unique, indices) = deduplicate(&items);

        assert_eq!(unique.len(), 2);
        assert_eq!(indices, vec![0, 0, 1, 0]);
    }

    #[test]
    fn type_names_match_laya() {
        assert_eq!(TYPES, ["choice", "score", "noul"]);
    }
}
