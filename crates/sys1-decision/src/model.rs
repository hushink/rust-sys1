use candle_core::{DType, Device, Tensor};
use candle_nn::{LayerNorm, Linear, Module, layer_norm};
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

const OPTION_OPEN: &str = "\n<option>\n";
const OPTION_CLOSE: &str = "\n</option>";
const SUFFIX: &str =
    "\n\nSelect the single option best supported by the context and instructions.\nDecision:";
const NOUL_FALSE: &str = "The answer to the question is no.";
const NOUL_TRUE: &str = "The answer to the question is yes.";
const PROMPT_VERSION: &str = "structured-segmented-candidate-endpoints-global-query-v2";

/// Runtime options for Decision 1.0.
#[derive(Clone)]
pub struct DecisionOptions {
    pub device: Device,
    pub dtype: DType,
    pub temperature: Option<f32>,
}

impl Default for DecisionOptions {
    fn default() -> Self {
        Self {
            device: Device::Cpu,
            dtype: DType::F32,
            temperature: None,
        }
    }
}

#[derive(Deserialize)]
struct DecisionConfigFile {
    #[serde(default)]
    decision_format: String,
    #[serde(default)]
    prompt_version: String,
    #[serde(default)]
    backbone: Backbone,
    #[serde(default)]
    calibration: Calibration,
}

#[derive(Default, Deserialize)]
struct Backbone {
    #[serde(default)]
    config: String,
    #[serde(default)]
    weights: Vec<String>,
}

#[derive(Default, Deserialize)]
struct Calibration {
    #[serde(default)]
    temperature: Option<f64>,
}

struct Head {
    candidate_norm: LayerNorm,
    query_norm: LayerNorm,
    key: Linear,
    query: Linear,
    candidate_mlp: Linear,
    query_mlp: Linear,
    scalar: Linear,
    head_dim: usize,
}

impl Head {
    fn load(
        directory: &Path,
        hidden: usize,
        dtype: DType,
        device: &Device,
    ) -> Result<Self, ModelError> {
        let store = WeightStore::from_safetensors(
            &[directory.join("decision_head.safetensors")],
            dtype,
            device,
        )
        .map_err(|error| ModelError::load(format!("load decision head: {error}")))?;
        let head_dim = 256;
        Ok(Self {
            candidate_norm: layer_norm(hidden, 1e-5, store.var_builder().pp("candidate_norm"))
                .map_err(|error| ModelError::load(error.to_string()))?,
            query_norm: layer_norm(hidden, 1e-5, store.var_builder().pp("query_norm"))
                .map_err(|error| ModelError::load(error.to_string()))?,
            key: store
                .linear("key", head_dim, hidden, false)
                .map_err(|error| ModelError::load(error.to_string()))?,
            query: store
                .linear("query", head_dim, hidden, false)
                .map_err(|error| ModelError::load(error.to_string()))?,
            candidate_mlp: store
                .linear("candidate_mlp", head_dim, hidden, true)
                .map_err(|error| ModelError::load(error.to_string()))?,
            query_mlp: store
                .linear("query_mlp", head_dim, hidden, false)
                .map_err(|error| ModelError::load(error.to_string()))?,
            scalar: store
                .linear("scalar", 1, head_dim, false)
                .map_err(|error| ModelError::load(error.to_string()))?,
            head_dim,
        })
    }

    fn forward(
        &self,
        hidden: &Tensor,
        query: usize,
        candidates: &[usize],
    ) -> Result<Tensor, ModelError> {
        let hidden = hidden.squeeze(0).map_err(inference_error)?;
        let candidate_positions = Tensor::from_vec(
            candidates
                .iter()
                .map(|value| *value as u32)
                .collect::<Vec<_>>(),
            candidates.len(),
            hidden.device(),
        )
        .map_err(inference_error)?;
        let query_position =
            Tensor::from_vec(vec![query as u32], 1, hidden.device()).map_err(inference_error)?;
        let candidate = hidden
            .index_select(&candidate_positions, 0)
            .map_err(inference_error)?;
        let query = hidden
            .index_select(&query_position, 0)
            .map_err(inference_error)?;
        let candidate = self
            .candidate_norm
            .forward(&candidate)
            .map_err(inference_error)?;
        let query = self.query_norm.forward(&query).map_err(inference_error)?;
        let bilinear = self
            .key
            .forward(&candidate)
            .map_err(inference_error)?
            .broadcast_mul(&self.query.forward(&query).map_err(inference_error)?)
            .map_err(inference_error)?
            .sum(candle_core::D::Minus1)
            .map_err(inference_error)?
            .affine((self.head_dim as f64).powf(-0.5), 0.0)
            .map_err(inference_error)?;
        let interaction = self
            .candidate_mlp
            .forward(&candidate)
            .map_err(inference_error)?
            .broadcast_add(&self.query_mlp.forward(&query).map_err(inference_error)?)
            .map_err(inference_error)?
            .gelu_erf()
            .map_err(inference_error)?;
        let interaction = self
            .scalar
            .forward(&interaction)
            .map_err(inference_error)?
            .squeeze(1)
            .map_err(inference_error)?;
        bilinear.add(&interaction).map_err(inference_error)
    }
}

struct Row {
    ids: Vec<u32>,
    query: usize,
    candidates: Vec<usize>,
}

struct Group {
    id: String,
    kind: QuestionKind,
    labels: Vec<String>,
    legend: IndexMap<String, Content>,
    row: Row,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum QuestionKind {
    Choice,
    Score,
    Noul,
}

type QuestionRow = (Row, QuestionKind, Vec<String>, IndexMap<String, Content>);

/// A Decision 1.0 endpoint-head model.
pub struct DecisionModel {
    tokenizer: Tokenizer,
    backbone: Qwen35,
    head: Head,
    temperature: f32,
    device: Device,
}

impl DecisionModel {
    pub fn load_dir(directory: &Path, options: DecisionOptions) -> Result<Self, ModelError> {
        let config: DecisionConfigFile = serde_json::from_slice(
            &std::fs::read(directory.join("decision_config.json"))
                .map_err(|error| ModelError::load(format!("read decision_config.json: {error}")))?,
        )
        .map_err(|error| ModelError::load(format!("parse decision_config.json: {error}")))?;
        if !config.decision_format.is_empty() && config.decision_format != "vllm-sr-decision" {
            return Err(ModelError::load(format!(
                "unsupported Decision format {:?}",
                config.decision_format
            )));
        }
        if !config.prompt_version.is_empty() && config.prompt_version != PROMPT_VERSION {
            return Err(ModelError::load(format!(
                "unsupported Decision prompt version {:?}",
                config.prompt_version
            )));
        }
        let backbone_config = if config.backbone.config.is_empty() {
            "backbone/config.json".to_owned()
        } else {
            config.backbone.config.clone()
        };
        let mut text_config = Qwen35Config::load(&directory.join(&backbone_config))
            .map_err(|error| ModelError::load(format!("load Decision backbone config: {error}")))?;
        let weights: Vec<_> = if config.backbone.weights.is_empty() {
            safetensors_files(&directory.join("backbone"))
                .map_err(|error| ModelError::load(format!("list Decision weights: {error}")))?
        } else {
            config
                .backbone
                .weights
                .iter()
                .map(|path| directory.join(path))
                .collect()
        };
        let store = WeightStore::from_safetensors(&weights, options.dtype, &options.device)
            .map_err(|error| ModelError::load(error.to_string()))?;
        let prefix = detect_prefix(
            &store,
            &["model.language_model.", "model.", "language_model.", ""],
            "embed_tokens.weight",
        )
        .ok_or_else(|| ModelError::load("Decision backbone has no embed_tokens.weight"))?;
        text_config.prefix = prefix;
        text_config.validate().map_err(ModelError::load)?;
        let hidden = text_config.hidden_size;
        let backbone = Qwen35::load(text_config, &store)
            .map_err(|error| ModelError::load(error.to_string()))?;
        let head = Head::load(directory, hidden, options.dtype, &options.device)?;
        let read_temperature = |name: &str, nested: bool| {
            std::fs::read(directory.join(name))
                .ok()
                .and_then(|bytes| serde_json::from_slice::<Value>(&bytes).ok())
                .and_then(|value| {
                    let value = if nested {
                        value.get("calibration")?
                    } else {
                        &value
                    };
                    value.get("temperature").and_then(Value::as_f64)
                })
        };
        let root_temperature = read_temperature("config.json", true)
            .or_else(|| read_temperature("runtime.json", false))
            .or_else(|| read_temperature("temperature.json", false));
        let temperature = options
            .temperature
            .or(config.calibration.temperature.map(|value| value as f32))
            .or(root_temperature.map(|value| value as f32))
            .unwrap_or(1.0);
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
            temperature,
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

    fn payload(&self, value: Option<&Content>) -> String {
        match value {
            None | Some(Content::Null) => "null".to_owned(),
            Some(Content::String(text)) => text.clone(),
            Some(content) => canonical_json(&content_to_value(content)),
        }
    }

    fn state_payload(&self, state: &State) -> String {
        match state.as_content() {
            Content::String(text) => text.clone(),
            content => canonical_json(&content_to_value(content)),
        }
    }

    fn options(&self, id: &str, question: &Question) -> Result<Vec<(String, Value)>, ModelError> {
        let bad =
            |message: String| ModelError::invalid_request(format!("question {id:?}: {message}"));
        match question {
            Question::Choice(choice) => {
                if choice.criteria.len() < 2 || choice.criteria.len() > 255 {
                    return Err(bad(format!(
                        "this model takes 2..255 choice options, got {}",
                        choice.criteria.len()
                    )));
                }
                Ok(choice
                    .criteria
                    .iter()
                    .map(|(key, value)| (key.clone(), content_to_value(value)))
                    .collect())
            }
            Question::Score(score) => {
                if score.criteria.len() < 2 || score.criteria.len() > 10 {
                    return Err(bad(format!(
                        "this model takes 2..10 score levels, got {}",
                        score.criteria.len()
                    )));
                }
                Ok(score
                    .criteria
                    .iter()
                    .enumerate()
                    .map(|(index, value)| (index.to_string(), content_to_value(value)))
                    .collect())
            }
            Question::Noul(noul) => {
                let false_value = noul
                    .criteria
                    .as_ref()
                    .and_then(|criteria| criteria.is_false.as_ref())
                    .map(content_to_value)
                    .unwrap_or_else(|| Value::String(NOUL_FALSE.to_owned()));
                let true_value = noul
                    .criteria
                    .as_ref()
                    .and_then(|criteria| criteria.is_true.as_ref())
                    .map(content_to_value)
                    .unwrap_or_else(|| Value::String(NOUL_TRUE.to_owned()));
                Ok(vec![
                    ("false".to_owned(), false_value),
                    ("true".to_owned(), true_value),
                ])
            }
            Question::Box(_) => Err(bad("Decision does not answer box questions".to_owned())),
        }
    }

    fn row(&self, state: &str, id: &str, question: &Question) -> Result<QuestionRow, ModelError> {
        let options = self.options(id, question)?;
        let instructions = self.payload(question.instructions());
        let task = match question {
            Question::Choice(_) => "choice",
            Question::Score(_) => "score",
            Question::Noul(_) => "noul",
            Question::Box(_) => unreachable!(),
        };
        let prefix =
            format!("Context:\n{state}\n\nTask type: {task}\nQuestion:\n{instructions}\nOptions:");
        let mut ids = self.encode(&prefix)?;
        let mut candidates = Vec::with_capacity(options.len());
        for (key, description) in &options {
            let mut object = Map::new();
            object.insert("key".to_owned(), Value::String(key.clone()));
            object.insert("description".to_owned(), description.clone());
            let text = format!(
                "{OPTION_OPEN}{}{OPTION_CLOSE}",
                canonical_json(&Value::Object(object))
            );
            ids.extend(self.encode(&text)?);
            candidates.push(ids.len() - 1);
        }
        ids.extend(self.encode(SUFFIX)?);
        if ids.len() > 16_384 {
            return Err(ModelError::invalid_request(format!(
                "question {id:?}: the row is {} tokens; the model reads at most 16384",
                ids.len()
            )));
        }
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
        let query = ids.len() - 1;
        Ok((
            Row {
                ids,
                query,
                candidates,
            },
            kind,
            labels,
            legend,
        ))
    }
}

impl DecisionModel {
    fn scores(&self, row: &Row) -> Result<Vec<f32>, ModelError> {
        let ids = Tensor::from_vec(row.ids.clone(), (1, row.ids.len()), &self.device)
            .map_err(inference_error)?;
        let hidden = self.backbone.forward(&ids).map_err(inference_error)?;
        self.head
            .forward(&hidden, row.query, &row.candidates)?
            .to_dtype(DType::F32)
            .map_err(inference_error)?
            .to_vec1::<f32>()
            .map_err(inference_error)
    }
}

impl sys1_core::DecisionModel for DecisionModel {
    fn predict(&self, state: &State, questions: &Questions) -> Result<DecisionOutput, ModelError> {
        if questions.is_empty() {
            return Err(ModelError::invalid_request("questions must not be empty"));
        }
        let state = self.state_payload(state);
        let mut groups = Vec::with_capacity(questions.len());
        let mut input_tokens = 0u64;
        for (id, question) in questions.iter() {
            let (row, kind, labels, legend) = self.row(&state, id, question)?;
            input_tokens += row.ids.len() as u64;
            groups.push(Group {
                id: id.clone(),
                kind,
                labels,
                legend,
                row,
            });
        }
        let mut answers = IndexMap::with_capacity(groups.len());
        for group in groups {
            let scores = self.scores(&group.row)?;
            let probabilities = softmax(&scores, self.temperature);
            let answer: Answer = match group.kind {
                QuestionKind::Choice => choice_answer(&group.labels, &probabilities),
                QuestionKind::Score => score_answer(&group.legend, &probabilities),
                QuestionKind::Noul => noul_answer(probabilities[0], probabilities[1]),
            };
            answers.insert(group.id, answer);
        }
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
