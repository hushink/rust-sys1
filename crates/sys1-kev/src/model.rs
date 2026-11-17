use candle_core::{DType, Device, Tensor};
use candle_nn::{Linear, Module};
use indexmap::IndexMap;
use serde_json::Value;
use std::borrow::Cow;
use std::fmt::Write;
use std::path::{Path, PathBuf};
use sys1_core::{
    Answers, DecisionModel, DecisionOutput, ModelError, Prediction, Questions, choice_answer,
    content_to_value, noul_answer, python_str, score_answer, softmax,
};
use sys1_protocol::{Content, Question, State, Usage};
use sys1_qwen::{LoraAdapter, Qwen35, Qwen35Config, WeightStore, detect_prefix, safetensors_files};
use tokenizers::Tokenizer;

const STATE_TOKEN: u32 = 248_060;
const QUESTION_TOKEN: u32 = 248_061;
const OPTION_OPEN: u32 = 248_049;
const OPTION_CLOSE: u32 = 248_050;
const DECIDE_TOKEN: u32 = 248_062;
const DEFAULT_TEMPERATURE: f32 = 2.351;
const MAX_STATE_BODY: usize = 8191;
const MAX_ROW: usize = 8192;

/// Runtime options for Kev.
#[derive(Clone)]
pub struct KevOptions {
    pub device: Device,
    pub dtype: DType,
    pub base_directory: Option<PathBuf>,
    pub temperature: Option<f32>,
}

impl Default for KevOptions {
    fn default() -> Self {
        Self {
            device: Device::Cpu,
            dtype: DType::F32,
            base_directory: None,
            temperature: None,
        }
    }
}

struct Row {
    ids: Vec<u32>,
    decide: usize,
    options: Vec<usize>,
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

type QuestionParts = (
    String,
    Vec<String>,
    QuestionKind,
    Vec<String>,
    IndexMap<String, Content>,
);

/// A Kev pointer-head decision model.
pub struct KevModel {
    tokenizer: Tokenizer,
    model: Qwen35,
    q: Linear,
    k: Linear,
    head_dim: usize,
    temperature: f32,
    device: Device,
}

impl KevModel {
    pub fn load_dir(directory: &Path, options: KevOptions) -> Result<Self, ModelError> {
        let base = options.base_directory.clone().unwrap_or_else(|| {
            if directory.join("base").join("config.json").is_file() {
                directory.join("base")
            } else {
                directory.to_path_buf()
            }
        });
        let mut config = Qwen35Config::load(&base.join("config.json"))
            .map_err(|error| ModelError::load(format!("load Kev base config: {error}")))?;
        let base_files = safetensors_files(&base)
            .map_err(|error| ModelError::load(format!("list Kev base weights: {error}")))?;
        if base_files.is_empty() {
            return Err(ModelError::load(format!(
                "no base safetensors under {}; put the Qwen3.5 base in a `base/` subdirectory \
                 or set KevOptions::base_directory",
                base.display()
            )));
        }
        let adapter_path = directory.join("adapter_model.safetensors");
        let adapter = LoraAdapter::load(
            &adapter_path,
            adapter_scaling(directory)?,
            options.dtype,
            &options.device,
        )
        .map_err(|error| ModelError::load(format!("load LoRA adapter: {error}")))?;
        let store = WeightStore::from_safetensors(&base_files, options.dtype, &options.device)
            .map_err(|error| ModelError::load(error.to_string()))?
            .with_lora(adapter);
        let prefix = detect_prefix(
            &store,
            &["model.language_model.", "model.", "language_model.", ""],
            "embed_tokens.weight",
        )
        .ok_or_else(|| ModelError::load("Kev base has no embed_tokens.weight"))?;
        config.prefix = prefix;
        config.validate().map_err(ModelError::load)?;
        let hidden = config.hidden_size;
        let model =
            Qwen35::load(config, &store).map_err(|error| ModelError::load(error.to_string()))?;

        let head_path = directory.join("head.pt");
        let head =
            WeightStore::from_pth_with_state(&head_path, options.dtype, &options.device, "head")
                .map_err(|error| ModelError::load(format!("load Kev head: {error}")))?;
        let head_dim = head
            .get_unchecked("q.weight")
            .and_then(|tensor| tensor.dim(0))
            .map_err(|error| ModelError::load(format!("Kev head: {error}")))?;
        let q = head
            .linear("q", head_dim, hidden, true)
            .map_err(|error| ModelError::load(error.to_string()))?;
        let k = head
            .linear("k", head_dim, hidden, true)
            .map_err(|error| ModelError::load(error.to_string()))?;

        let tokenizer_path = directory.join("tokenizer.json");
        let mut tokenizer = Tokenizer::from_file(&tokenizer_path)
            .map_err(|error| ModelError::load(format!("load tokenizer: {error}")))?;
        tokenizer
            .with_truncation(None)
            .map_err(|error| ModelError::load(format!("disable truncation: {error}")))?;
        let _ = tokenizer.with_padding(None);
        let temperature = options
            .temperature
            .or_else(|| read_temperature(&directory.join("kev.json")))
            .or_else(|| crate::directory_metadata(directory).temperature)
            .unwrap_or(DEFAULT_TEMPERATURE);
        Ok(Self {
            tokenizer,
            model,
            q,
            k,
            head_dim,
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

    fn user(&self, text: &str) -> Result<Vec<u32>, ModelError> {
        self.encode(&escape_delimiters(text))
    }

    fn state_ids(&self, state: &State) -> Result<Vec<u32>, ModelError> {
        let body = self.user(&render(&content_to_value(state.as_content())))?;
        let keep = body.len().min(MAX_STATE_BODY);
        let mut ids = Vec::with_capacity(keep + 1);
        ids.push(STATE_TOKEN);
        ids.extend_from_slice(&body[..keep]);
        Ok(ids)
    }

    fn question(&self, id: &str, question: &Question) -> Result<QuestionParts, ModelError> {
        let instructions = question
            .instructions()
            .map(|content| render(&content_to_value(content)))
            .unwrap_or_default();
        match question {
            Question::Choice(choice) => {
                let options = choice
                    .criteria
                    .iter()
                    .map(|(label, value)| option_text(label, Some(value)))
                    .collect();
                Ok((
                    instructions,
                    options,
                    QuestionKind::Choice,
                    choice.criteria.keys().cloned().collect(),
                    IndexMap::new(),
                ))
            }
            Question::Score(score) => {
                let options = score
                    .criteria
                    .iter()
                    .map(|value| render(&content_to_value(value)))
                    .collect();
                let legend = score
                    .criteria
                    .iter()
                    .enumerate()
                    .map(|(index, value)| (index.to_string(), value.clone()))
                    .collect();
                Ok((
                    instructions,
                    options,
                    QuestionKind::Score,
                    (0..score.criteria.len())
                        .map(|index| index.to_string())
                        .collect(),
                    legend,
                ))
            }
            Question::Noul(noul) => {
                let false_content = noul
                    .criteria
                    .as_ref()
                    .and_then(|criteria| criteria.is_false.as_ref());
                let true_content = noul
                    .criteria
                    .as_ref()
                    .and_then(|criteria| criteria.is_true.as_ref());
                let options = vec![
                    option_text("no", false_content),
                    option_text("yes", true_content),
                ];
                Ok((
                    instructions,
                    options,
                    QuestionKind::Noul,
                    Vec::new(),
                    IndexMap::new(),
                ))
            }
            Question::Box(_) => Err(ModelError::invalid_request(format!(
                "question {id:?}: Kev does not answer box questions"
            ))),
        }
    }

    fn row(
        &self,
        state: &[u32],
        question: &str,
        question_id: &str,
        options: &[String],
    ) -> Result<Row, ModelError> {
        let mut ids = Vec::with_capacity(state.len() + 64);
        ids.extend_from_slice(state);
        ids.push(QUESTION_TOKEN);
        ids.extend(self.user(question)?);
        let mut option_positions = Vec::with_capacity(options.len());
        for option in options {
            ids.push(OPTION_OPEN);
            ids.extend(self.user(option)?);
            option_positions.push(ids.len());
            ids.push(OPTION_CLOSE);
        }
        ids.push(DECIDE_TOKEN);
        if ids.len() > MAX_ROW {
            return Err(ModelError::invalid_request(format!(
                "question {question_id:?}: the row is {} tokens; the model reads at most {MAX_ROW}",
                ids.len()
            )));
        }
        Ok(Row {
            decide: ids.len() - 1,
            options: option_positions,
            ids,
        })
    }

    fn forward_row(&self, row: &Row) -> Result<Vec<f32>, ModelError> {
        let ids = Tensor::from_vec(row.ids.clone(), (1, row.ids.len()), &self.device)
            .map_err(inference_error)?;
        let hidden = self.model.forward(&ids).map_err(inference_error)?;
        let hidden = hidden.squeeze(0).map_err(inference_error)?;
        let option_positions = Tensor::from_vec(
            row.options
                .iter()
                .map(|position| *position as u32)
                .collect::<Vec<_>>(),
            row.options.len(),
            &self.device,
        )
        .map_err(inference_error)?;
        let decide_position =
            Tensor::from_vec(vec![row.decide as u32], 1, &self.device).map_err(inference_error)?;
        let targets = hidden
            .index_select(&option_positions, 0)
            .map_err(inference_error)?;
        let decide = hidden
            .index_select(&decide_position, 0)
            .map_err(inference_error)?;
        let keys = self.k.forward(&targets).map_err(inference_error)?;
        let query = self.q.forward(&decide).map_err(inference_error)?;
        let scores = keys
            .broadcast_mul(&query)
            .map_err(inference_error)?
            .sum(candle_core::D::Minus1)
            .map_err(inference_error)?
            .affine((self.head_dim as f64).powf(-0.5), 0.0)
            .map_err(inference_error)?;
        scores
            .to_dtype(DType::F32)
            .map_err(inference_error)?
            .to_vec1::<f32>()
            .map_err(inference_error)
    }
}

impl DecisionModel for KevModel {
    fn predict(&self, state: &State, questions: &Questions) -> Result<DecisionOutput, ModelError> {
        if questions.is_empty() {
            return Err(ModelError::invalid_request("questions must not be empty"));
        }
        let state = self.state_ids(state)?;
        let mut groups = Vec::with_capacity(questions.len());
        let mut input_tokens = 0u64;
        for (id, question) in questions.iter() {
            let (instructions, options, kind, labels, legend) = self.question(id, question)?;
            if options.is_empty() {
                return Err(ModelError::invalid_request(format!(
                    "question {id:?}: Kev needs at least one option"
                )));
            }
            let row = self.row(&state, &instructions, id, &options)?;
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
            let scores = self.forward_row(&group.row)?;
            let probabilities = softmax(&scores, self.temperature);
            let answer = match group.kind {
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

fn adapter_scaling(directory: &Path) -> Result<f32, ModelError> {
    let path = directory.join("adapter_config.json");
    let value: Value = serde_json::from_slice(
        &std::fs::read(&path)
            .map_err(|error| ModelError::load(format!("read {}: {error}", path.display())))?,
    )
    .map_err(|error| ModelError::load(format!("parse {}: {error}", path.display())))?;
    let rank = value.get("r").and_then(Value::as_f64).unwrap_or(16.0);
    let alpha = value
        .get("lora_alpha")
        .and_then(Value::as_f64)
        .unwrap_or(32.0);
    Ok((alpha / rank) as f32)
}

fn read_temperature(path: &Path) -> Option<f32> {
    let value: Value = serde_json::from_slice(&std::fs::read(path).ok()?).ok()?;
    value
        .get("temperature")
        .and_then(Value::as_f64)
        .map(|value| value as f32)
}

fn option_text(name: &str, description: Option<&Content>) -> String {
    match description {
        None | Some(Content::Null) => name.to_owned(),
        Some(Content::String(text)) if text.is_empty() => name.to_owned(),
        Some(content) => format!("{name}: {}", render(&content_to_value(content))),
    }
}

/// `kev.api.render`: null as `""`, scalars as Python `str()`, lists as `- item`
/// lines, and objects as `key: value` lines with two spaces per level.
fn render(value: &Value) -> String {
    render_at(value, 0)
}

fn render_at(value: &Value, indent: usize) -> String {
    let pad = "  ".repeat(indent);
    match value {
        Value::Null => String::new(),
        Value::String(text) => text.clone(),
        Value::Bool(true) => "True".into(),
        Value::Bool(false) => "False".into(),
        Value::Number(_) => python_str(value),
        Value::Array(items) => {
            let mut out = String::new();
            for (index, item) in items.iter().enumerate() {
                if index > 0 {
                    out.push('\n');
                }
                let item = render_at(item, indent + 1);
                let _ = write!(out, "{pad}- {}", item.trim_start_matches(is_py_space));
            }
            out
        }
        Value::Object(map) => {
            let mut out = String::new();
            for (index, (key, item)) in map.iter().enumerate() {
                if index > 0 {
                    out.push('\n');
                }
                match item {
                    Value::Object(_) | Value::Array(_) => {
                        let _ = write!(out, "{pad}{key}:\n{}", render_at(item, indent + 1));
                    }
                    item => {
                        let _ = write!(out, "{pad}{key}: {}", render_at(item, 0));
                    }
                }
            }
            out
        }
    }
}

fn is_py_space(character: char) -> bool {
    character.is_whitespace() || ('\u{1c}'..='\u{1f}').contains(&character)
}

/// `re.sub(r"<\|([A-Za-z0-9_]+)\|>", r"<¦\1¦>", text)`.
fn escape_delimiters(text: &str) -> Cow<'_, str> {
    if !text.contains("<|") {
        return Cow::Borrowed(text);
    }
    let bytes = text.as_bytes();
    let mut out = String::with_capacity(text.len() + 8);
    let mut copied = 0;
    let mut index = 0;
    while index + 1 < bytes.len() {
        if bytes[index] == b'<' && bytes[index + 1] == b'|' {
            let start = index + 2;
            let end = start
                + bytes[start..]
                    .iter()
                    .take_while(|byte| byte.is_ascii_alphanumeric() || **byte == b'_')
                    .count();
            if end > start && bytes[end..].starts_with(b"|>") {
                out.push_str(&text[copied..index]);
                out.push_str("<¦");
                out.push_str(&text[start..end]);
                out.push_str("¦>");
                index = end + 2;
                copied = index;
                continue;
            }
        }
        index += 1;
    }
    out.push_str(&text[copied..]);
    Cow::Owned(out)
}

fn inference_error(error: candle_core::Error) -> ModelError {
    ModelError::inference(error.to_string())
}
