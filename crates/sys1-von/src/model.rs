use candle_core::{DType, Device, Tensor};
use candle_nn::{Activation, LayerNorm, Linear, Module, VarBuilder, layer_norm, linear};
use indexmap::IndexMap;
use std::path::Path;
use sys1_core::{
    Answers, DecisionModel, DecisionOutput, ModelError, Prediction, Questions, choice_answer,
    noul_answer, render_object_lines, softmax, strip, truthy,
};
use sys1_modernbert::{Encoder, ModernBertConfig};
use sys1_protocol::{Answer, Content, Question, State, Usage};
use tokenizers::Tokenizer;

/// Runtime options for Von.
#[derive(Clone)]
pub struct VonOptions {
    pub device: Device,
    pub dtype: DType,
    pub max_len: Option<usize>,
    pub max_batch: usize,
}

impl Default for VonOptions {
    fn default() -> Self {
        Self {
            device: Device::Cpu,
            dtype: DType::F32,
            max_len: None,
            max_batch: 32,
        }
    }
}

const MASK_TEXT: &str = "[MASK]";
const SEP_TEXT: &str = "[SEP]";
const DEFAULT_MAX_LEN: usize = 8192;
const ZERO_SHOT_A: f64 = 0.7;
const ZERO_SHOT_B: f64 = 0.0;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum QKind {
    Choice,
    Score,
    Noul,
}

struct Group {
    id: String,
    kind: QKind,
    labels: Vec<String>,
    legend: IndexMap<String, Content>,
    rows: std::ops::Range<usize>,
    zero_shot: bool,
}

struct Row {
    ids: Vec<u32>,
    markers: Vec<usize>,
}

struct StateText {
    text: String,
    raw_tokens: usize,
}

struct Scorer {
    input_norm: LayerNorm,
    dense: Linear,
    norm: LayerNorm,
    out_proj: Linear,
    activation: Activation,
}

impl Scorer {
    fn load(vb: VarBuilder, hidden: usize) -> Result<Self, ModelError> {
        let scorer = vb.pp("scorer");
        let half = scorer
            .get_unchecked("dense.weight")
            .map_err(|error| ModelError::load(error.to_string()))?
            .dim(0)
            .map_err(|error| ModelError::load(error.to_string()))?;
        let load = |name: &str, out: usize, input: usize| -> Result<Linear, ModelError> {
            linear(input, out, scorer.pp(name)).map_err(|error| ModelError::load(error.to_string()))
        };
        Ok(Self {
            input_norm: layer_norm(hidden, 1e-5, scorer.pp("input_norm"))
                .map_err(|error| ModelError::load(error.to_string()))?,
            dense: load("dense", half, hidden)?,
            norm: layer_norm(half, 1e-5, scorer.pp("norm"))
                .map_err(|error| ModelError::load(error.to_string()))?,
            out_proj: load("out_proj", 1, half)?,
            activation: Activation::Gelu,
        })
    }

    fn forward(
        &self,
        hidden: &Tensor,
        positions: &Tensor,
        mask: &Tensor,
    ) -> Result<Tensor, ModelError> {
        let (batch, _seq, width) = hidden.dims3().map_err(inference_error)?;
        let markers = positions.dim(1).map_err(inference_error)?;
        let at = positions
            .unsqueeze(2)
            .map_err(inference_error)?
            .broadcast_as((batch, markers, width))
            .map_err(inference_error)?;
        let gathered = hidden
            .contiguous()
            .map_err(inference_error)?
            .gather(&at.contiguous().map_err(inference_error)?, 1)
            .map_err(inference_error)?;
        let logits = gathered
            .apply(&self.input_norm)
            .map_err(inference_error)?
            .apply(&self.dense)
            .map_err(inference_error)?;
        let logits = self.activation.forward(&logits).map_err(inference_error)?;
        let logits = logits
            .apply(&self.norm)
            .map_err(inference_error)?
            .apply(&self.out_proj)
            .map_err(inference_error)?
            .squeeze(2)
            .map_err(inference_error)?;
        let mask = mask.to_dtype(logits.dtype()).map_err(inference_error)?;
        let masked = logits
            .broadcast_mul(&mask)
            .map_err(inference_error)?
            .broadcast_add(
                &((mask.affine(-1.0, 1.0)).map_err(inference_error)? * -1e4)
                    .map_err(inference_error)?,
            )
            .map_err(inference_error)?;
        Ok(masked)
    }
}

/// A Von 1.1 option-marker model.
pub struct VonModel {
    tokenizer: Tokenizer,
    encoder: Encoder,
    scorer: Scorer,
    cls: u32,
    sep: u32,
    mask: u32,
    pad: u32,
    max_len: usize,
    max_batch: usize,
    device: Device,
}

impl VonModel {
    pub fn load_dir(directory: &Path, options: VonOptions) -> Result<Self, ModelError> {
        let mut config =
            ModernBertConfig::load(&directory.join("config.json")).map_err(ModelError::load)?;
        let max_len = options.max_len.unwrap_or(DEFAULT_MAX_LEN);
        if max_len < 8 {
            return Err(ModelError::load(format!("max_len {max_len} is below 8")));
        }
        // Von was trained with rows up to 8k; the checkpoint's inherited
        // ModernBERT config still says 2k, so grow the rotary table.
        if max_len > config.max_position_embeddings {
            config.max_position_embeddings = max_len;
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
            .with_truncation(None)
            .map_err(|error| ModelError::load(format!("disable truncation: {error}")))?;
        let _ = tokenizer.with_padding(None);
        let token = |name: &str| {
            tokenizer
                .token_to_id(name)
                .ok_or_else(|| ModelError::load(format!("tokenizer has no {name}")))
        };
        let cls = token("[CLS]")?;
        let sep = token("[SEP]")?;
        let mask = token("[MASK]")?;
        let pad = token("[PAD]")?;
        let weights = directory.join("option_marker.pt");
        if !weights.is_file() {
            return Err(ModelError::load(format!(
                "missing weights at {}",
                weights.display()
            )));
        }
        let vb = VarBuilder::from_pth(&weights, options.dtype, &options.device)
            .map_err(|error| ModelError::load(format!("load {}: {error}", weights.display())))?;
        let encoder = Encoder::load(vb.clone(), &config, options.dtype, false, "encoder")
            .map_err(|error| ModelError::load(error.to_string()))?;
        let scorer = Scorer::load(vb, config.hidden_size)?;
        Ok(Self {
            tokenizer,
            encoder,
            scorer,
            cls,
            sep,
            mask,
            pad,
            max_len,
            max_batch: options.max_batch.max(1),
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

    fn state_text(&self, state: &State) -> Result<StateText, ModelError> {
        let value = sys1_core::content_to_value(state.as_content());
        let text = match state.as_content() {
            Content::String(text) => text.clone(),
            Content::Object(_) => render_object_lines(&value),
            Content::Array(_) => sys1_core::python_repr(&value),
            Content::Null => String::new(),
        };
        let raw_tokens = self.encode(&text)?.len();
        Ok(StateText {
            text: sanitize(&text),
            raw_tokens,
        })
    }

    fn descriptions(&self, question: &Question) -> Result<(Vec<String>, bool, QKind), ModelError> {
        match question {
            Question::Choice(choice) => {
                if choice.criteria.is_empty() {
                    return Err(ModelError::invalid_request(
                        "choice question requires criteria",
                    ));
                }
                let descriptions = choice
                    .criteria
                    .iter()
                    .map(|(label, value)| {
                        if truthy(&value_to_value(value)) {
                            strip(&sys1_core::python_str(&value_to_value(value))).to_owned()
                        } else {
                            strip(label).to_owned()
                        }
                    })
                    .collect();
                Ok((descriptions, false, QKind::Choice))
            }
            Question::Score(score) => {
                if score.criteria.len() < 2 {
                    return Err(ModelError::invalid_request(
                        "score question needs at least two levels",
                    ));
                }
                Ok((
                    score.criteria.iter().map(level_description).collect(),
                    false,
                    QKind::Score,
                ))
            }
            Question::Noul(noul) => {
                let criteria = noul.criteria.as_ref();
                let true_text = criteria.and_then(|c| c.is_true.as_ref());
                let false_text = criteria.and_then(|c| c.is_false.as_ref());
                let true_text = true_text.filter(|value| truthy(&value_to_value(value)));
                let false_text = false_text.filter(|value| truthy(&value_to_value(value)));
                let zero_shot = true_text.is_none() && false_text.is_none();
                Ok((
                    vec![
                        true_text
                            .map(value_to_value)
                            .map(|value| sys1_core::python_str(&value))
                            .unwrap_or_else(|| "Yes, condition holds true.".to_owned()),
                        false_text
                            .map(value_to_value)
                            .map(|value| sys1_core::python_str(&value))
                            .unwrap_or_else(|| "No, condition is false.".to_owned()),
                    ],
                    zero_shot,
                    QKind::Noul,
                ))
            }
            Question::Box(_) => Err(ModelError::invalid_request(
                "Von does not answer box questions",
            )),
        }
    }

    fn instructions(&self, question: &Question) -> String {
        sanitize(
            &question
                .instructions()
                .map(sys1_core::render_content)
                .unwrap_or_default(),
        )
    }

    fn tokenize_row(&self, text: &str) -> Result<Vec<u32>, ModelError> {
        let body = self.encode(text)?;
        let mut ids = Vec::with_capacity(body.len() + 2);
        ids.push(self.cls);
        ids.extend_from_slice(&body);
        ids.push(self.sep);
        Ok(ids)
    }

    fn pack(&self, state: &str, instructions: &str, descriptions: &[String]) -> String {
        let prefix = if instructions.is_empty() {
            strip(state).to_owned()
        } else {
            strip(&format!("{instructions} {state}")).to_owned()
        };
        let mut text = format!("{prefix} {SEP_TEXT} ");
        for (index, description) in descriptions.iter().enumerate() {
            if index > 0 {
                text.push(' ');
            }
            text.push_str(MASK_TEXT);
            text.push(' ');
            text.push_str(strip(description));
        }
        text
    }

    fn fit_row(
        &self,
        state: &str,
        instructions: &str,
        descriptions: &[String],
        qid: &str,
    ) -> Result<(Row, bool), ModelError> {
        let mut text = self.pack(state, instructions, descriptions);
        let mut ids = self.tokenize_row(&text)?;
        let state_chars = state.chars().count();
        let mut kept = state_chars;
        if ids.len() > self.max_len {
            let mut low = 0usize;
            let mut high = state_chars;
            while low < high {
                let middle = (low + high).div_ceil(2);
                let prefix: String = state.chars().take(middle).collect();
                let candidate = self.pack(&prefix, instructions, descriptions);
                if self.tokenize_row(&candidate)?.len() <= self.max_len {
                    low = middle;
                } else {
                    high = middle - 1;
                }
            }
            kept = low;
            if kept == 0 {
                let failed = self.tokenize_row(&self.pack("", instructions, descriptions))?;
                if failed.len() > self.max_len {
                    return Err(ModelError::invalid_request(format!(
                        "question {qid:?}: its instructions and options alone take {} tokens; \
                         the model reads at most {}",
                        failed.len(),
                        self.max_len
                    )));
                }
            }
            let prefix: String = state.chars().take(kept).collect();
            text = self.pack(&prefix, instructions, descriptions);
            ids = self.tokenize_row(&text)?;
        }
        let markers: Vec<usize> = ids
            .iter()
            .enumerate()
            .filter(|(_, id)| **id == self.mask)
            .map(|(index, _)| index)
            .collect();
        if markers.len() != descriptions.len() {
            return Err(ModelError::invalid_request(format!(
                "question {qid:?}: the text tokenizes to {} option markers for {} options",
                markers.len(),
                descriptions.len()
            )));
        }
        Ok((Row { ids, markers }, kept < state_chars))
    }

    fn forward_rows(&self, rows: &[Row]) -> Result<Vec<Vec<f32>>, ModelError> {
        let mut output = Vec::with_capacity(rows.len());
        for chunk in rows.chunks(self.max_batch) {
            let batch = chunk.len();
            let seq = chunk
                .iter()
                .map(|row| row.ids.len())
                .max()
                .unwrap_or(0)
                .max(8);
            let markers = chunk
                .iter()
                .map(|row| row.markers.len())
                .max()
                .unwrap_or(1)
                .max(1);
            let mut ids = vec![self.pad; batch * seq];
            let mut attention = vec![0f32; batch * seq];
            let mut positions = vec![0u32; batch * markers];
            let mut marker_mask = vec![0f32; batch * markers];
            for (row_index, row) in chunk.iter().enumerate() {
                let start = row_index * seq;
                ids[start..start + row.ids.len()].copy_from_slice(&row.ids);
                attention[start..start + row.ids.len()].fill(1.0);
                for (marker_index, &position) in row.markers.iter().enumerate() {
                    positions[row_index * markers + marker_index] = position as u32;
                    marker_mask[row_index * markers + marker_index] = 1.0;
                }
            }
            let ids = Tensor::from_vec(ids, (batch, seq), &self.device).map_err(inference_error)?;
            let attention =
                Tensor::from_vec(attention, (batch, seq), &self.device).map_err(inference_error)?;
            let positions = Tensor::from_vec(positions, (batch, markers), &self.device)
                .map_err(inference_error)?;
            let marker_mask = Tensor::from_vec(marker_mask, (batch, markers), &self.device)
                .map_err(inference_error)?;
            let hidden = self
                .encoder
                .forward(
                    &ids,
                    &attention,
                    chunk.iter().any(|row| row.ids.len() != seq),
                )
                .map_err(inference_error)?;
            let logits = self.scorer.forward(&hidden, &positions, &marker_mask)?;
            output.extend(
                logits
                    .to_dtype(DType::F32)
                    .map_err(inference_error)?
                    .to_vec2::<f32>()
                    .map_err(inference_error)?,
            );
        }
        Ok(output)
    }

    fn option_logits(
        &self,
        kind: QKind,
        zero_shot: bool,
        first: &[f32],
        empty_state: Option<&[f32]>,
    ) -> Vec<f32> {
        if kind != QKind::Noul {
            return first.to_vec();
        }
        let (true_logit, false_logit) = (f64::from(first[0]), f64::from(first[1]));
        let correction = if zero_shot {
            empty_state.map_or(0.0, |row| {
                ZERO_SHOT_A * (f64::from(row[0]) - f64::from(row[1])) + ZERO_SHOT_B
            })
        } else {
            0.0
        };
        vec![false_logit as f32, (true_logit - correction) as f32]
    }

    fn calibrated(&self, logits: &[f32], state_tokens: usize) -> Vec<f32> {
        let k = logits.len();
        if k == 0 {
            return Vec::new();
        }
        let probabilities = softmax(logits, 1.0);
        let entropy = -probabilities
            .iter()
            .map(|probability| {
                let value = f64::from(*probability);
                value * value.max(1e-12).ln()
            })
            .sum::<f64>();
        let normalized_entropy = if k <= 1 {
            0.0
        } else {
            entropy / (k as f64).ln()
        };
        let temperature = 0.2056
            + (-3.255) * normalized_entropy
            + 20.2391 * (state_tokens.max(1) as f64).log10() / 4.0
            + (-5.2263) * (k as f64) / 8.0;
        let temperature = temperature.clamp(0.3, 12.0).max(1e-4);
        softmax(logits, temperature as f32)
    }
}

impl DecisionModel for VonModel {
    fn predict(&self, state: &State, questions: &Questions) -> Result<DecisionOutput, ModelError> {
        if questions.is_empty() {
            return Err(ModelError::invalid_request("questions must not be empty"));
        }
        let state = self.state_text(state)?;
        let mut rows = Vec::new();
        let mut groups = Vec::new();
        for (id, question) in questions.iter() {
            let (descriptions, zero_shot, kind) = self.descriptions(question)?;
            let instructions = self.instructions(question);
            let start = rows.len();
            let (row, _) = self.fit_row(&state.text, &instructions, &descriptions, id)?;
            rows.push(row);
            if zero_shot {
                let (row, _) = self.fit_row("", &instructions, &descriptions, id)?;
                rows.push(row);
            }
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
            groups.push(Group {
                id: id.clone(),
                kind,
                labels,
                legend,
                rows: start..rows.len(),
                zero_shot,
            });
        }
        let logits = self.forward_rows(&rows)?;
        let mut answers = IndexMap::with_capacity(groups.len());
        for group in &groups {
            let first = &logits[group.rows.start];
            let empty = group
                .zero_shot
                .then(|| logits.get(group.rows.start + 1).map(Vec::as_slice))
                .flatten();
            let options = self.option_logits(group.kind, group.zero_shot, first, empty);
            let probabilities = self.calibrated(&options, state.raw_tokens);
            let answer = match group.kind {
                QKind::Choice => choice_answer(&group.labels, &probabilities),
                QKind::Score => {
                    let probabilities = group
                        .legend
                        .keys()
                        .enumerate()
                        .map(|(index, key)| (key.clone(), f64::from(probabilities[index])))
                        .collect::<IndexMap<_, _>>();
                    let score = probabilities
                        .values()
                        .enumerate()
                        .map(|(index, probability)| index as f64 * probability)
                        .sum();
                    Answer::Score(sys1_protocol::ScoreAnswer {
                        score,
                        confidence: sys1_core::typesafe_confidence(&probabilities),
                        legend: group.legend.clone(),
                        probabilities,
                    })
                }
                QKind::Noul => noul_answer(
                    probabilities.first().copied().unwrap_or(0.5),
                    probabilities.get(1).copied().unwrap_or(0.5),
                ),
            };
            answers.insert(group.id.clone(), answer);
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

fn sanitize(text: &str) -> String {
    text.replace(MASK_TEXT, " ")
}

fn value_to_value(content: &Content) -> serde_json::Value {
    sys1_core::content_to_value(content)
}

fn level_description(value: &Content) -> String {
    match value {
        Content::Object(map) => {
            let what = map
                .get("what")
                .map(sys1_core::python_str)
                .unwrap_or_default();
            let examples = map
                .get("examples")
                .filter(|value| truthy(value))
                .map(|value| format!(" Examples: {}", join_examples(value)))
                .unwrap_or_default();
            strip(&format!("{what}{examples}")).to_owned()
        }
        content => strip(&sys1_core::python_str(&value_to_value(content))).to_owned(),
    }
}

fn join_examples(value: &serde_json::Value) -> String {
    match value {
        serde_json::Value::String(text) => text
            .chars()
            .map(String::from)
            .collect::<Vec<_>>()
            .join(", "),
        serde_json::Value::Array(items) => items
            .iter()
            .map(sys1_core::python_str)
            .collect::<Vec<_>>()
            .join(", "),
        serde_json::Value::Object(map) => map.keys().cloned().collect::<Vec<_>>().join(", "),
        other => sys1_core::python_str(other),
    }
}

fn inference_error(error: candle_core::Error) -> ModelError {
    ModelError::inference(error.to_string())
}
