use crate::config::{NliConfig, Pooling};
use crate::render::{criterion, instructions_or_id, render_state};
use candle_core::{DType, Device, Tensor};
use candle_nn::{LayerNorm, Linear, VarBuilder, layer_norm_no_bias};
use indexmap::IndexMap;
use std::path::Path;
use sys1_core::{Answers, DecisionModel, DecisionOutput, ModelError, Prediction, Questions};
use sys1_modernbert::Encoder;
use sys1_protocol::{
    Answer, ChoiceAnswer, Content, NoulAnswer, Question, ScoreAnswer, State, Usage,
};
use tokenizers::Tokenizer;

/// Runtime options for an NLI model.
#[derive(Clone)]
pub struct NliOptions {
    pub device: Device,
    pub dtype: DType,
    /// Override the checkpoint's 512-token pair budget.
    pub max_len: Option<usize>,
    /// Maximum pair rows per forward pass.
    pub max_batch: usize,
}

impl Default for NliOptions {
    fn default() -> Self {
        Self {
            device: Device::Cpu,
            dtype: DType::F32,
            max_len: None,
            max_batch: 32,
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum RowMode {
    Entail,
    Single,
}

struct QuestionGroup {
    id: String,
    kind: GroupKind,
    mode: RowMode,
    start: usize,
    count: usize,
}

enum GroupKind {
    Choice { labels: Vec<String> },
    Score { legend: IndexMap<String, Content> },
    Noul,
}

struct Row {
    ids: Vec<u32>,
}

/// A ModernBERT zero-shot NLI cross-encoder.
pub struct NliModel {
    tokenizer: Tokenizer,
    config: NliConfig,
    encoder: Encoder,
    head_dense: Linear,
    head_norm: LayerNorm,
    classifier: Linear,
    device: Device,
    dtype: DType,
    max_len: usize,
    max_batch: usize,
}

impl NliModel {
    pub fn load_dir(directory: &Path, options: NliOptions) -> Result<Self, ModelError> {
        let config = NliConfig::load(directory).map_err(ModelError::load)?;
        let max_len = options.max_len.unwrap_or(config.max_len);
        if max_len < 8 || max_len > config.encoder.max_position_embeddings() {
            return Err(ModelError::load(format!(
                "max_len {max_len} is outside 8..={}",
                config.encoder.max_position_embeddings()
            )));
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
        let tokenizer = Tokenizer::from_file(&tokenizer_path)
            .map_err(|error| ModelError::load(format!("load tokenizer: {error}")))?;

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
        let encoder = Encoder::load(vb.clone(), &config.encoder, options.dtype, false, "model")
            .map_err(|error| ModelError::load(error.to_string()))?;

        let hidden = config.encoder.hidden_size;
        let head_dense = Linear::new(
            vb.get((hidden, hidden), "head.dense.weight")
                .map_err(|error| ModelError::load(error.to_string()))?
                .to_dtype(options.dtype)
                .map_err(|error| ModelError::load(error.to_string()))?,
            None,
        );
        let head_norm =
            layer_norm_no_bias(hidden, config.encoder.layer_norm_eps(), vb.pp("head.norm"))
                .map_err(|error| ModelError::load(error.to_string()))?;
        let classifier = Linear::new(
            vb.get((2, hidden), "classifier.weight")
                .map_err(|error| ModelError::load(error.to_string()))?
                .to_dtype(options.dtype)
                .map_err(|error| ModelError::load(error.to_string()))?,
            Some(
                vb.get(2, "classifier.bias")
                    .map_err(|error| ModelError::load(error.to_string()))?
                    .to_dtype(options.dtype)
                    .map_err(|error| ModelError::load(error.to_string()))?,
            ),
        );

        Ok(Self {
            tokenizer,
            config,
            encoder,
            head_dense,
            head_norm,
            classifier,
            device: options.device,
            dtype: options.dtype,
            max_len,
            max_batch: options.max_batch.max(1),
        })
    }

    fn encode(&self, text: &str) -> Result<Vec<u32>, ModelError> {
        self.tokenizer
            .encode(text, false)
            .map(|encoding| encoding.get_ids().to_vec())
            .map_err(|error| ModelError::inference(error.to_string()))
    }

    fn question_group(
        &self,
        id: &str,
        question: &Question,
        premise_len: usize,
        premise: &[u32],
        rows: &mut Vec<Row>,
    ) -> Result<QuestionGroup, ModelError> {
        let instructions = match question {
            Question::Choice(q) => instructions_or_id(q.instructions.as_ref(), id),
            Question::Score(q) => instructions_or_id(q.instructions.as_ref(), id),
            Question::Noul(q) => instructions_or_id(q.instructions.as_ref(), id),
            Question::Box(_) => {
                return Err(ModelError::unsupported(format!(
                    "question {id:?}: NLI does not support box questions"
                )));
            }
        };
        let templates = &self.config.templates;
        let (hypotheses, mode, kind) = match question {
            Question::Choice(choice) => {
                let labels = choice.criteria.keys().cloned().collect::<Vec<_>>();
                let hypotheses = choice
                    .criteria
                    .iter()
                    .map(|(label, value)| {
                        let description = criterion(value);
                        let option = if description.is_empty() {
                            label.clone()
                        } else {
                            format!("{label}: {description}")
                        };
                        let template = if description.is_empty() {
                            &templates.choice_without_description
                        } else {
                            &templates.choice_with_description
                        };
                        fill(
                            template,
                            &[
                                ("instructions", &instructions),
                                ("label", label),
                                ("description", &description),
                                ("option", &option),
                            ],
                        )
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                (hypotheses, RowMode::Entail, GroupKind::Choice { labels })
            }
            Question::Score(score) => {
                if score.criteria.is_empty() {
                    return Err(ModelError::invalid_request(format!(
                        "score question {id:?} requires criteria"
                    )));
                }
                let mut legend = IndexMap::new();
                let hypotheses = score
                    .criteria
                    .iter()
                    .enumerate()
                    .map(|(index, value)| {
                        let level = index.to_string();
                        legend.insert(level.clone(), value.clone());
                        let description = criterion(value);
                        fill(
                            &templates.score,
                            &[
                                ("instructions", &instructions),
                                ("level", &level),
                                ("description", &description),
                            ],
                        )
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                (hypotheses, RowMode::Entail, GroupKind::Score { legend })
            }
            Question::Noul(noul) => {
                let false_text =
                    described(noul.criteria.as_ref().and_then(|c| c.is_false.as_ref()));
                let true_text = described(noul.criteria.as_ref().and_then(|c| c.is_true.as_ref()));
                match (false_text, true_text) {
                    (Some(false_text), Some(true_text)) => (
                        vec![
                            fill(
                                &templates.noul_pair,
                                &[
                                    ("instructions", &instructions),
                                    ("description", &false_text),
                                ],
                            )?,
                            fill(
                                &templates.noul_pair,
                                &[("instructions", &instructions), ("description", &true_text)],
                            )?,
                        ],
                        RowMode::Entail,
                        GroupKind::Noul,
                    ),
                    _ => (
                        vec![fill(
                            &templates.noul_single,
                            &[("instructions", &instructions)],
                        )?],
                        RowMode::Single,
                        GroupKind::Noul,
                    ),
                }
            }
            Question::Box(_) => unreachable!(),
        };

        let start = rows.len();
        for hypothesis in hypotheses {
            let hypothesis_ids = self.encode(&hypothesis)?;
            let room = self
                .max_len
                .checked_sub(hypothesis_ids.len() + 3)
                .filter(|room| *room > 0)
                .ok_or_else(|| {
                    ModelError::invalid_request(format!(
                        "question {id:?}: hypothesis is {} tokens and leaves no room for the state \
                         (the model reads {} tokens per pair)",
                        hypothesis_ids.len(),
                        self.max_len
                    ))
                })?;
            let keep = premise_len.min(room);
            let mut ids = Vec::with_capacity(keep + hypothesis_ids.len() + 3);
            ids.push(self.config.cls);
            ids.extend_from_slice(&premise[..keep]);
            ids.push(self.config.sep);
            ids.extend_from_slice(&hypothesis_ids);
            ids.push(self.config.sep);
            rows.push(Row { ids });
        }
        Ok(QuestionGroup {
            id: id.to_owned(),
            kind,
            mode,
            start,
            count: rows.len() - start,
        })
    }

    fn forward_rows(&self, rows: &[Row]) -> Result<Vec<Vec<f32>>, ModelError> {
        let mut output = Vec::with_capacity(rows.len());
        for chunk in rows.chunks(self.max_batch) {
            let batch = chunk.len();
            let seq = chunk.iter().map(|row| row.ids.len()).max().unwrap_or(0);
            if batch == 0 || seq == 0 {
                continue;
            }
            let mut ids = vec![self.config.pad; batch * seq];
            let mut mask = vec![0f32; batch * seq];
            for (index, row) in chunk.iter().enumerate() {
                let start = index * seq;
                ids[start..start + row.ids.len()].copy_from_slice(&row.ids);
                mask[start..start + row.ids.len()].fill(1.0);
            }
            let has_padding = chunk.iter().any(|row| row.ids.len() != seq);
            let ids = Tensor::from_vec(ids, (batch, seq), &self.device).map_err(inference_error)?;
            let mask =
                Tensor::from_vec(mask, (batch, seq), &self.device).map_err(inference_error)?;
            let hidden = self
                .encoder
                .forward(&ids, &mask, has_padding)
                .map_err(inference_error)?;
            let pooled = match self.config.pooling {
                Pooling::Mean => {
                    let mask3 = mask.unsqueeze(2).map_err(inference_error)?;
                    let summed = hidden
                        .broadcast_mul(&mask3)
                        .map_err(inference_error)?
                        .sum(1)
                        .map_err(inference_error)?;
                    let counts = mask
                        .sum(1)
                        .map_err(inference_error)?
                        .unsqueeze(1)
                        .map_err(inference_error)?;
                    summed.broadcast_div(&counts).map_err(inference_error)?
                }
                Pooling::Cls => hidden
                    .narrow(1, 0, 1)
                    .map_err(inference_error)?
                    .squeeze(1)
                    .map_err(inference_error)?,
            };
            let pooled = pooled
                .to_dtype(self.dtype)
                .map_err(inference_error)?
                .apply(&self.head_dense)
                .map_err(inference_error)?
                .gelu_erf()
                .map_err(inference_error)?
                .apply(&self.head_norm)
                .map_err(inference_error)?;
            let logits = pooled
                .apply(&self.classifier)
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
        row_logits: &[Vec<f32>],
    ) -> Result<(String, Answer), ModelError> {
        let logits: Vec<f32> = match group.mode {
            RowMode::Entail => row_logits[group.start..group.start + group.count]
                .iter()
                .map(|row| row[self.config.entailment])
                .collect(),
            RowMode::Single => {
                let row = &row_logits[group.start];
                vec![row[self.config.not_entailment], row[self.config.entailment]]
            }
        };
        let probabilities = softmax(&logits, 1.0);
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

impl DecisionModel for NliModel {
    fn predict(&self, state: &State, questions: &Questions) -> Result<DecisionOutput, ModelError> {
        self.predict_batch(&[Prediction::new(state, questions)])
            .into_iter()
            .next()
            .expect("one prediction result")
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
            .map(|request| self.predict_one(request.state, request.questions))
            .collect()
    }
}

impl NliModel {
    fn predict_one(
        &self,
        state: &State,
        questions: &Questions,
    ) -> Result<DecisionOutput, ModelError> {
        if questions.is_empty() {
            return Err(ModelError::invalid_request("questions must not be empty"));
        }
        let state_text = render_state(state);
        let premise = self.encode(&state_text)?;
        let mut rows = Vec::new();
        let mut groups = Vec::new();
        for (id, question) in questions.iter() {
            groups.push(self.question_group(id, question, premise.len(), &premise, &mut rows)?);
        }
        let row_logits = self.forward_rows(&rows)?;
        let mut answers = IndexMap::with_capacity(groups.len());
        for group in &groups {
            let (id, answer) = self.answer(group, &row_logits)?;
            answers.insert(id, answer);
        }
        let input_tokens = rows.iter().map(|row| row.ids.len() as u64).sum();
        Ok(DecisionOutput::new(
            Answers::new(answers),
            Usage::new(input_tokens, 0),
        ))
    }
}

fn described(value: Option<&Content>) -> Option<String> {
    match value {
        None | Some(Content::Null) => None,
        Some(Content::String(value)) if value.is_empty() => None,
        Some(value) => Some(criterion(value)),
    }
}

fn fill(template: &str, fields: &[(&str, &str)]) -> Result<String, ModelError> {
    let mut out = String::with_capacity(template.len() + 64);
    let mut rest = template;
    while let Some(open) = rest.find('{') {
        out.push_str(&rest[..open]);
        let after = &rest[open + 1..];
        let end = after
            .find(|c: char| !(c.is_alphanumeric() || c == '_'))
            .unwrap_or(after.len());
        if end > 0 && after[end..].starts_with('}') {
            let name = &after[..end];
            let value = fields
                .iter()
                .find_map(|&(key, value)| (key == name).then_some(value))
                .ok_or_else(|| {
                    ModelError::load(format!(
                        "NLI template {template:?} uses unknown field {{{name}}}"
                    ))
                })?;
            out.push_str(value);
            rest = &after[end + 1..];
        } else {
            out.push('{');
            rest = after;
        }
    }
    out.push_str(rest);
    Ok(out)
}

fn softmax(logits: &[f32], temperature: f32) -> Vec<f32> {
    if logits.is_empty() {
        return Vec::new();
    }
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max) / temperature;
    let mut values: Vec<f32> = logits
        .iter()
        .map(|value| (value / temperature - max).exp())
        .collect();
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
