use crate::config::GliclassConfig;
use crate::render::{criterion, instructions_or_id, render_state, sanitize};
use candle_core::{DType, Device, Tensor};
use candle_nn::{Embedding, Linear, Module, VarBuilder, embedding};
use indexmap::IndexMap;
use std::path::Path;
use sys1_core::{Answers, DecisionModel, DecisionOutput, ModelError, Prediction, Questions};
use sys1_modernbert::Encoder;
use sys1_protocol::{
    Answer, ChoiceAnswer, Content, NoulAnswer, Question, ScoreAnswer, State, Usage,
};
use tokenizers::{Tokenizer, TruncationParams, TruncationStrategy};

/// Runtime options for a GLiClass model.
#[derive(Clone)]
pub struct GliclassOptions {
    pub device: Device,
    pub dtype: DType,
    pub max_len: Option<usize>,
    pub max_batch: usize,
}

impl Default for GliclassOptions {
    fn default() -> Self {
        Self {
            device: Device::Cpu,
            dtype: DType::F32,
            max_len: None,
            max_batch: 32,
        }
    }
}

struct Projector {
    linear1: Linear,
    linear2: Linear,
}

impl Projector {
    fn load(
        vb: &VarBuilder,
        prefix: &str,
        hidden: usize,
        dtype: DType,
    ) -> Result<Self, ModelError> {
        let linear = |name: &str| -> Result<Linear, ModelError> {
            Ok(Linear::new(
                vb.get((hidden, hidden), &format!("{prefix}.{name}.weight"))
                    .map_err(|error| ModelError::load(error.to_string()))?
                    .to_dtype(dtype)
                    .map_err(|error| ModelError::load(error.to_string()))?,
                Some(
                    vb.get(hidden, &format!("{prefix}.{name}.bias"))
                        .map_err(|error| ModelError::load(error.to_string()))?
                        .to_dtype(dtype)
                        .map_err(|error| ModelError::load(error.to_string()))?,
                ),
            ))
        };
        Ok(Self {
            linear1: linear("linear_1")?,
            linear2: linear("linear_2")?,
        })
    }

    fn forward(&self, xs: &Tensor) -> candle_core::Result<Tensor> {
        xs.apply(&self.linear1)?.gelu_erf()?.apply(&self.linear2)
    }
}

struct Scorer {
    linear1: Linear,
    linear2: Linear,
    output: Linear,
}

impl Scorer {
    fn load(vb: &VarBuilder, hidden: usize, dtype: DType) -> Result<Self, ModelError> {
        let linear = |name: &str, input: usize, output: usize| -> Result<Linear, ModelError> {
            Ok(Linear::new(
                vb.get((output, input), &format!("model.scorer.mlp.{name}.weight"))
                    .map_err(|error| ModelError::load(error.to_string()))?
                    .to_dtype(dtype)
                    .map_err(|error| ModelError::load(error.to_string()))?,
                Some(
                    vb.get(output, &format!("model.scorer.mlp.{name}.bias"))
                        .map_err(|error| ModelError::load(error.to_string()))?
                        .to_dtype(dtype)
                        .map_err(|error| ModelError::load(error.to_string()))?,
                ),
            ))
        };
        Ok(Self {
            linear1: linear("0", hidden * 2, 256)?,
            linear2: linear("2", 256, 128)?,
            output: linear("4", 128, 1)?,
        })
    }

    fn forward(&self, text: &Tensor, labels: &Tensor) -> candle_core::Result<Tensor> {
        let (_batch, count, hidden) = labels.dims3()?;
        let text = text
            .unsqueeze(1)?
            .broadcast_as((text.dim(0)?, count, hidden))?;
        let combined = Tensor::cat(&[&text, labels], candle_core::D::Minus1)?.contiguous()?;
        combined
            .apply(&self.linear1)?
            .relu()?
            .apply(&self.linear2)?
            .relu()?
            .apply(&self.output)?
            .squeeze(2)
    }
}

struct Row {
    ids: Vec<u32>,
    markers: Vec<usize>,
    labels: Vec<String>,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum AnswerMode {
    Softmax,
    Sigmoid,
}

struct QuestionGroup {
    id: String,
    kind: GroupKind,
    mode: AnswerMode,
    row: usize,
}

enum GroupKind {
    Choice { labels: Vec<String> },
    Score { legend: IndexMap<String, Content> },
    Noul,
}

/// A GLiClass uni-encoder decision model.
pub struct GliclassModel {
    tokenizer: Tokenizer,
    config: GliclassConfig,
    encoder: Encoder,
    segment: Embedding,
    text_projector: Projector,
    classes_projector: Projector,
    scorer: Scorer,
    device: Device,
    dtype: DType,
    max_batch: usize,
}

impl GliclassModel {
    pub fn load_dir(directory: &Path, options: GliclassOptions) -> Result<Self, ModelError> {
        let mut config = GliclassConfig::load(directory).map_err(ModelError::load)?;
        if let Some(max_len) = options.max_len {
            if max_len < 8 || max_len > config.encoder.max_position_embeddings() {
                return Err(ModelError::load(format!(
                    "max_len {max_len} is outside 8..={}",
                    config.encoder.max_position_embeddings()
                )));
            }
            config.max_len = max_len;
        }
        let tokenizer_path = [
            directory.join("tokenizer.json"),
            directory.join("tokenizer").join("tokenizer.json"),
        ]
        .into_iter()
        .find(|path| path.is_file())
        .ok_or_else(|| {
            ModelError::load(format!(
                "no tokenizer.json found under {}",
                directory.display()
            ))
        })?;
        let mut tokenizer = Tokenizer::from_file(&tokenizer_path)
            .map_err(|error| ModelError::load(format!("load tokenizer: {error}")))?;
        tokenizer
            .with_truncation(Some(TruncationParams {
                max_length: config.max_len,
                strategy: TruncationStrategy::LongestFirst,
                direction: tokenizers::TruncationDirection::Right,
                stride: 0,
            }))
            .map_err(|error| ModelError::load(format!("set truncation: {error}")))?;

        let weights = directory.join("model.safetensors");
        if !weights.is_file() {
            return Err(ModelError::load(format!(
                "missing weights at {}",
                weights.display()
            )));
        }
        let tensors = candle_core::safetensors::load(&weights, &options.device)
            .map_err(|error| ModelError::load(format!("load {}: {error}", weights.display())))?;
        let vb = VarBuilder::from_tensors(tensors, options.dtype, &options.device);
        let encoder = Encoder::load(
            vb.clone(),
            &config.encoder,
            options.dtype,
            false,
            "model.encoder_model",
        )
        .map_err(|error| ModelError::load(error.to_string()))?;
        let hidden = config.hidden_size;
        let segment = embedding(3, hidden, vb.pp("model").pp("segment_embeddings"))
            .map_err(|error| ModelError::load(error.to_string()))?;
        let text_projector = Projector::load(&vb, "model.text_projector", hidden, options.dtype)?;
        let classes_projector =
            Projector::load(&vb, "model.classes_projector", hidden, options.dtype)?;
        let scorer = Scorer::load(&vb, hidden, options.dtype)?;

        Ok(Self {
            tokenizer,
            config,
            encoder,
            segment,
            text_projector,
            classes_projector,
            scorer,
            device: options.device,
            dtype: options.dtype,
            max_batch: options.max_batch.max(1),
        })
    }

    fn encode(&self, text: &str) -> Result<Vec<u32>, ModelError> {
        self.tokenizer
            .encode(text, true)
            .map(|encoding| encoding.get_ids().to_vec())
            .map_err(|error| ModelError::inference(error.to_string()))
    }

    fn question(
        &self,
        id: &str,
        question: &Question,
        state_text: &str,
        rows: &mut Vec<Row>,
    ) -> Result<QuestionGroup, ModelError> {
        let instructions = match question {
            Question::Choice(q) => instructions_or_id(q.instructions.as_ref(), id),
            Question::Score(q) => instructions_or_id(q.instructions.as_ref(), id),
            Question::Noul(q) => instructions_or_id(q.instructions.as_ref(), id),
            Question::Box(_) => {
                return Err(ModelError::unsupported(format!(
                    "question {id:?}: GLiClass does not support box questions"
                )));
            }
        };
        let (labels, prompt, mode, kind) = match question {
            Question::Choice(choice) => {
                let answer_labels = choice.criteria.keys().cloned().collect::<Vec<_>>();
                let labels = choice
                    .criteria
                    .iter()
                    .map(|(label, value)| {
                        let description = criterion(value);
                        if description.is_empty() {
                            label.clone()
                        } else {
                            format!("{label}: {description}")
                        }
                    })
                    .collect::<Vec<_>>();
                (
                    labels,
                    instructions.clone(),
                    AnswerMode::Softmax,
                    GroupKind::Choice {
                        labels: answer_labels,
                    },
                )
            }
            Question::Score(score) => {
                if score.criteria.is_empty() {
                    return Err(ModelError::invalid_request(format!(
                        "score question {id:?} requires criteria"
                    )));
                }
                let mut legend = IndexMap::new();
                let labels = score
                    .criteria
                    .iter()
                    .enumerate()
                    .map(|(index, value)| {
                        let level = index.to_string();
                        legend.insert(level.clone(), value.clone());
                        format!("level {index}: {}", criterion(value))
                    })
                    .collect::<Vec<_>>();
                (
                    labels,
                    instructions.clone(),
                    AnswerMode::Softmax,
                    GroupKind::Score { legend },
                )
            }
            Question::Noul(noul) => {
                let false_text =
                    described(noul.criteria.as_ref().and_then(|c| c.is_false.as_ref()));
                let true_text = described(noul.criteria.as_ref().and_then(|c| c.is_true.as_ref()));
                match (false_text, true_text) {
                    (Some(false_text), Some(true_text)) => (
                        vec![false_text, true_text],
                        instructions.clone(),
                        AnswerMode::Softmax,
                        GroupKind::Noul,
                    ),
                    _ => (
                        vec![instructions.clone()],
                        String::new(),
                        AnswerMode::Sigmoid,
                        GroupKind::Noul,
                    ),
                }
            }
            Question::Box(_) => unreachable!(),
        };

        let labels = labels
            .into_iter()
            .map(|label| sanitize(&label))
            .collect::<Vec<_>>();
        let prompt = sanitize(&prompt);
        let state = sanitize(state_text);
        let mut text = String::new();
        for label in &labels {
            text.push_str("<<LABEL>>");
            text.push_str(label);
        }
        text.push_str("<<SEP>>");
        text.push_str(&prompt);
        text.push_str(&state);
        let ids = self.encode(&text)?;
        let markers = ids
            .iter()
            .enumerate()
            .filter_map(|(index, id)| (*id == self.config.class_token_index).then_some(index))
            .collect::<Vec<_>>();
        if markers.len() != labels.len() {
            return Err(ModelError::invalid_request(format!(
                "question {id:?}: {} labels were cut off by the {}-token window",
                labels.len().saturating_sub(markers.len()),
                self.config.max_len
            )));
        }
        let row = rows.len();
        rows.push(Row {
            ids,
            markers,
            labels,
        });
        Ok(QuestionGroup {
            id: id.to_owned(),
            kind,
            mode,
            row,
        })
    }

    fn forward_rows(&self, rows: &[Row]) -> Result<Vec<Vec<f32>>, ModelError> {
        let mut output = Vec::with_capacity(rows.len());
        for chunk in rows.chunks(self.max_batch) {
            let batch = chunk.len();
            let seq = chunk.iter().map(|row| row.ids.len()).max().unwrap_or(0);
            let count = chunk.iter().map(|row| row.markers.len()).max().unwrap_or(0);
            if batch == 0 || seq == 0 || count == 0 {
                continue;
            }
            let mut ids = vec![self.config.pad; batch * seq];
            let mut mask = vec![0f32; batch * seq];
            let mut segments = vec![0u32; batch * seq];
            let mut spans = vec![0f32; batch * count * seq];
            for (row_index, row) in chunk.iter().enumerate() {
                let start = row_index * seq;
                ids[start..start + row.ids.len()].copy_from_slice(&row.ids);
                mask[start..start + row.ids.len()].fill(1.0);
                let text_start = row
                    .ids
                    .iter()
                    .position(|id| *id == self.config.text_token_index)
                    .unwrap_or(0);
                segments[start + text_start..start + row.ids.len()].fill(1);
                for (label, marker) in row.markers.iter().enumerate() {
                    let end = row.markers.get(label + 1).copied().unwrap_or(row.ids.len());
                    let base = (row_index * count + label) * seq;
                    spans[base + marker..base + end].fill(1.0);
                }
            }
            let has_padding = chunk.iter().any(|row| row.ids.len() != seq);
            let ids = Tensor::from_vec(ids, (batch, seq), &self.device).map_err(inference_error)?;
            let mask =
                Tensor::from_vec(mask, (batch, seq), &self.device).map_err(inference_error)?;
            let segments =
                Tensor::from_vec(segments, (batch, seq), &self.device).map_err(inference_error)?;
            let embeddings = self
                .encoder
                .token_embeddings(&ids)
                .map_err(inference_error)?;
            let segment_embeddings = self
                .segment
                .forward(&segments)
                .map_err(inference_error)?
                .to_dtype(self.dtype)
                .map_err(inference_error)?;
            let embeddings = embeddings
                .broadcast_add(&segment_embeddings)
                .map_err(inference_error)?;
            let hidden = self
                .encoder
                .forward_embeds(embeddings, &mask, has_padding)
                .map_err(inference_error)?;

            let spans = Tensor::from_vec(spans, (batch, count, seq), &self.device)
                .map_err(inference_error)?
                .to_dtype(hidden.dtype())
                .map_err(inference_error)?;
            let counts = spans
                .sum(2)
                .map_err(inference_error)?
                .clamp(1.0, f32::MAX)
                .map_err(inference_error)?;
            let classes = spans
                .matmul(&hidden.contiguous().map_err(inference_error)?)
                .map_err(inference_error)?
                .broadcast_div(&counts.unsqueeze(2).map_err(inference_error)?)
                .map_err(inference_error)?;
            let text = hidden
                .narrow(1, 0, 1)
                .map_err(inference_error)?
                .squeeze(1)
                .map_err(inference_error)?;
            let text = self
                .text_projector
                .forward(&text.contiguous().map_err(inference_error)?)
                .map_err(inference_error)?;
            let classes = self
                .classes_projector
                .forward(&classes.contiguous().map_err(inference_error)?)
                .map_err(inference_error)?;
            let logits = self
                .scorer
                .forward(&text, &classes)
                .map_err(inference_error)?
                .to_dtype(DType::F32)
                .map_err(inference_error)?;
            output.extend(logits.to_vec2::<f32>().map_err(inference_error)?);
        }
        Ok(output)
    }

    fn answer(
        &self,
        group: &QuestionGroup,
        rows: &[Row],
        row_logits: &[Vec<f32>],
    ) -> Result<(String, Answer), ModelError> {
        let row = &rows[group.row];
        let logits = &row_logits[group.row][..row.labels.len()];
        let probabilities = match group.mode {
            AnswerMode::Softmax => softmax(logits),
            AnswerMode::Sigmoid => {
                let p_true = 1.0 / (1.0 + (-logits[0]).exp());
                vec![1.0 - p_true, p_true]
            }
        };
        let answer = match &group.kind {
            GroupKind::Choice { labels } => {
                let best = argmax(&probabilities);
                let probabilities = labels
                    .iter()
                    .zip(&probabilities)
                    .map(|(label, probability)| (label.clone(), f64::from(*probability)))
                    .collect();
                Answer::Choice(ChoiceAnswer {
                    choice: labels[best].clone(),
                    confidence: typesafe_confidence(&probabilities),
                    probabilities,
                })
            }
            GroupKind::Score { legend } => {
                let probabilities_map = (0..legend.len())
                    .map(|index| (index.to_string(), f64::from(probabilities[index])))
                    .collect::<IndexMap<_, _>>();
                let score = probabilities
                    .iter()
                    .enumerate()
                    .map(|(index, probability)| index as f64 * f64::from(*probability))
                    .sum();
                Answer::Score(ScoreAnswer {
                    score,
                    confidence: typesafe_confidence(&probabilities_map),
                    legend: legend.clone(),
                    probabilities: probabilities_map,
                })
            }
            GroupKind::Noul => Answer::Noul(NoulAnswer {
                noul: f64::from(probabilities.get(1).copied().unwrap_or(0.5)),
            }),
        };
        Ok((group.id.clone(), answer))
    }
}

impl DecisionModel for GliclassModel {
    fn predict(&self, state: &State, questions: &Questions) -> Result<DecisionOutput, ModelError> {
        if questions.is_empty() {
            return Err(ModelError::invalid_request("questions must not be empty"));
        }
        let state_text = render_state(state);
        let mut rows = Vec::new();
        let mut groups = Vec::new();
        for (id, question) in questions.iter() {
            groups.push(self.question(id, question, &state_text, &mut rows)?);
        }
        let row_logits = self.forward_rows(&rows)?;
        let mut answers = IndexMap::with_capacity(groups.len());
        for group in &groups {
            let (id, answer) = self.answer(group, &rows, &row_logits)?;
            answers.insert(id, answer);
        }
        let input_tokens = rows.iter().map(|row| row.ids.len() as u64).sum();
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

fn described(value: Option<&Content>) -> Option<String> {
    match value {
        None | Some(Content::Null) => None,
        Some(Content::String(value)) if value.is_empty() => None,
        Some(value) => Some(criterion(value)),
    }
}

fn softmax(logits: &[f32]) -> Vec<f32> {
    if logits.is_empty() {
        return Vec::new();
    }
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut values: Vec<f32> = logits.iter().map(|value| (value - max).exp()).collect();
    let total: f32 = values.iter().sum();
    if total > 0.0 {
        values.iter_mut().for_each(|value| *value /= total);
    }
    values
}

fn argmax(values: &[f32]) -> usize {
    let mut best = 0;
    for (index, value) in values.iter().enumerate() {
        if *value > values[best] {
            best = index;
        }
    }
    best
}

fn typesafe_confidence(probabilities: &IndexMap<String, f64>) -> f64 {
    let count = probabilities.len();
    if count < 2 {
        return 1.0;
    }
    let max = probabilities
        .values()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max);
    ((count as f64 * max - 1.0) / (count as f64 - 1.0)).clamp(0.0, 1.0)
}

fn inference_error(error: candle_core::Error) -> ModelError {
    ModelError::inference(error.to_string())
}
