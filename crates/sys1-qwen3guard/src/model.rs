use candle_core::{DType, Device, Tensor};
use indexmap::IndexMap;
use std::path::Path;
use sys1_core::{
    Answers, DecisionModel, DecisionOutput, ModelError, Prediction, Questions, choice_answer,
    noul_answer, render_state, softmax,
};
use sys1_protocol::{Answer, Content, Question, State, Usage};
use sys1_qwen::{
    Qwen3, Qwen3Config, WeightStore, candidate_logits, detect_prefix, safetensors_files,
};
use tokenizers::Tokenizer;

/// Runtime options for Qwen3Guard.
#[derive(Clone)]
pub struct Qwen3GuardOptions {
    pub device: Device,
    pub dtype: DType,
    pub max_len: Option<usize>,
}

impl Default for Qwen3GuardOptions {
    fn default() -> Self {
        Self {
            device: Device::Cpu,
            dtype: DType::F32,
            max_len: None,
        }
    }
}

const DEFAULT_MAX_LEN: usize = 32_768;
const SAFETY_LABELS: [&str; 3] = ["safe", "controversial", "unsafe"];
const CATEGORY_LABELS: [&str; 10] = [
    "none",
    "violent",
    "non_violent_illegal",
    "sexual",
    "pii",
    "suicide_self_harm",
    "unethical",
    "politically_sensitive",
    "copyright",
    "jailbreak",
];
const SAFETY_TOKENS: [&str; 3] = [" Safe", " Cont", " Unsafe"];
const CATEGORY_TOKENS: [&str; 10] = [
    " None",
    " Violent",
    " Non",
    " Sexual",
    " PII",
    " Suicide",
    " Unethical",
    " Political",
    " Copyright",
    " Jailbreak",
];
const SAFETY_ROW: &str = "Safety:";
const CATEGORY_ROW: &str = "Safety: Unsafe\nCategories:";

/// The user branch of Qwen3Guard's chat template, rendered around the state.
const PROMPT_HEAD: &str = "<|im_start|>user\n# Task:\n1) Evaluate whether THE LAST USER's query belongs to safe or unsafe content based on our safety policy.\n2) Identify the category of unsafe content found.\n\n<BEGIN SAFETY POLICY>\n**Safety: Safe**: The USER's query does not contain any prohibited content or content that could lead to prohibited content.\n**Safety: Unsafe**: The USER's query contains prohibited content.\n**Safety: Controversial**: The USER's query does not explicitly contain prohibited content, but its intent, context, or potential responses could be misinterpreted, misused, or exploited to generate unsafe content under certain conditions.\n<END SAFETY POLICY>\n\n<BEGIN UNSAFE CONTENT CATEGORIES>\nViolent.\nNon-violent Illegal Acts.\nSexual Content or Sexual Acts.\nPII.\nSuicide & Self-Harm.\nUnethical Acts.\nPolitically Sensitive Topics.\nCopyright Violation.\nJailbreak.\n<END UNSAFE CONTENT CATEGORIES>\n\n<BEGIN CONVERSATION>\n\nUSER: ";
const PROMPT_TAIL: &str = "\n\n<END CONVERSATION>\n\nProvide your safety assessment for ONLY THE LAST **USER's query** in the above conversation:\n - The first line must be one of: 'Safety: Safe', 'Safety: Unsafe', 'Safety: Controversial'.\n - The second line should start with 'Categories:' followed by a list of any unsafe content categories, separated by commas. If the content is safe, use 'Categories: None'.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Readout {
    Safety,
    Unsafe,
    UnsafeStrict,
    Category,
}

/// One Qwen3Guard checkpoint.
pub struct Qwen3GuardModel {
    tokenizer: Tokenizer,
    model: Qwen3,
    candidate_tensor: Tensor,
    max_len: usize,
    device: Device,
}

impl Qwen3GuardModel {
    pub fn load_dir(directory: &Path, options: Qwen3GuardOptions) -> Result<Self, ModelError> {
        let config_path = directory.join("config.json");
        let mut config = Qwen3Config::load(&config_path).map_err(ModelError::load)?;
        config.validate().map_err(ModelError::load)?;
        let max_len = options.max_len.unwrap_or(DEFAULT_MAX_LEN);
        if !(32..=DEFAULT_MAX_LEN).contains(&max_len) {
            return Err(ModelError::load(format!(
                "max_len {max_len} is outside 32..={DEFAULT_MAX_LEN}"
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
        let mut tokenizer = Tokenizer::from_file(&tokenizer_path)
            .map_err(|error| ModelError::load(format!("load tokenizer: {error}")))?;
        tokenizer
            .with_truncation(None)
            .map_err(|error| ModelError::load(format!("disable truncation: {error}")))?;
        let _ = tokenizer.with_padding(None);

        let paths = safetensors_files(directory)
            .map_err(|error| ModelError::load(format!("list weights: {error}")))?;
        let store = WeightStore::from_safetensors(&paths, options.dtype, &options.device)
            .map_err(|error| ModelError::load(error.to_string()))?;
        let prefix = detect_prefix(&store, &["model.", ""], "embed_tokens.weight")
            .ok_or_else(|| ModelError::load("no embed_tokens.weight found"))?;
        config.prefix = prefix;
        let model =
            Qwen3::load(&config, &store).map_err(|error| ModelError::load(error.to_string()))?;

        let mut candidate_ids = Vec::with_capacity(SAFETY_TOKENS.len() + CATEGORY_TOKENS.len());
        for text in SAFETY_TOKENS.iter().chain(CATEGORY_TOKENS.iter()) {
            let ids = tokenizer
                .encode(*text, false)
                .map_err(|error| ModelError::load(format!("tokenize candidate {text:?}: {error}")))?
                .get_ids()
                .to_vec();
            let id = ids
                .first()
                .copied()
                .ok_or_else(|| ModelError::load(format!("candidate {text:?} is empty")))?;
            candidate_ids.push(id);
        }
        if candidate_ids[..SAFETY_LABELS.len()]
            .windows(2)
            .any(|pair| pair[0] == pair[1])
            || candidate_ids[SAFETY_LABELS.len()..]
                .windows(2)
                .any(|pair| pair[0] == pair[1])
        {
            return Err(ModelError::load(
                "Qwen3Guard candidate labels do not have unique first tokens",
            ));
        }
        let candidate_tensor =
            Tensor::from_vec(candidate_ids.clone(), candidate_ids.len(), &options.device)
                .map_err(inference_error)?;
        Ok(Self {
            tokenizer,
            model,
            candidate_tensor,
            max_len,
            device: options.device,
        })
    }

    /// The question ids and shapes this checkpoint answers.
    pub fn preset_questions() -> Questions {
        Questions::new()
            .choice(
                "safety",
                SAFETY_LABELS.iter().map(|label| (*label, Content::Null)),
            )
            .noul("unsafe", "The content is unsafe.")
            .noul("unsafe_strict", "The content is unsafe or controversial.")
            .choice(
                "category",
                CATEGORY_LABELS.iter().map(|label| (*label, Content::Null)),
            )
    }

    fn plan(&self, questions: &Questions) -> Result<Vec<Readout>, ModelError> {
        questions
            .iter()
            .map(|(id, question)| {
                let same_set = |keys: Vec<&str>, expected: &[&str]| {
                    keys.len() == expected.len()
                        && expected.iter().all(|label| keys.contains(label))
                };
                let valid = match (id.as_str(), question) {
                    ("safety", Question::Choice(choice)) => same_set(
                        choice.criteria.keys().map(String::as_str).collect(),
                        &SAFETY_LABELS,
                    ),
                    ("category", Question::Choice(choice)) => same_set(
                        choice.criteria.keys().map(String::as_str).collect(),
                        &CATEGORY_LABELS,
                    ),
                    ("unsafe" | "unsafe_strict", Question::Noul(_)) => true,
                    _ => false,
                };
                if !valid {
                    return Err(ModelError::invalid_request(format!(
                        "question {id:?}: Qwen3Guard answers only its built-in questions \
                         (safety, unsafe, unsafe_strict, category); send them with their fixed \
                         choices or use Qwen3GuardModel::preset_questions()"
                    )));
                }
                Ok(match id.as_str() {
                    "safety" => Readout::Safety,
                    "unsafe" => Readout::Unsafe,
                    "unsafe_strict" => Readout::UnsafeStrict,
                    _ => Readout::Category,
                })
            })
            .collect()
    }

    fn encode(&self, text: &str) -> Result<Vec<u32>, ModelError> {
        Ok(self
            .tokenizer
            .encode(text, false)
            .map_err(|error| ModelError::inference(format!("tokenize: {error}")))?
            .get_ids()
            .to_vec())
    }

    fn forward(&self, rows: &[Vec<u32>]) -> Result<Vec<Vec<f32>>, ModelError> {
        let batch = rows.len();
        let seq = rows.iter().map(Vec::len).max().unwrap_or(0);
        if batch == 0 || seq == 0 {
            return Ok(Vec::new());
        }
        let mut ids = vec![0u32; batch * seq];
        let mut positions = Vec::with_capacity(batch);
        for (index, row) in rows.iter().enumerate() {
            let start = index * seq;
            ids[start..start + row.len()].copy_from_slice(row);
            positions.push((row.len() - 1) as u32);
        }
        let ids = Tensor::from_vec(ids, (batch, seq), &self.device).map_err(inference_error)?;
        let hidden = self.model.forward(&ids).map_err(inference_error)?;
        let positions =
            Tensor::from_vec(positions, batch, &self.device).map_err(inference_error)?;
        let logits = candidate_logits(
            &hidden,
            &positions,
            self.model.output_embedding(),
            &self.candidate_tensor,
        )
        .map_err(inference_error)?;
        logits
            .to_dtype(DType::F32)
            .map_err(inference_error)?
            .to_vec2::<f32>()
            .map_err(inference_error)
    }

    fn option_logits(&self, readout: Readout, candidates: &[f32]) -> Vec<f32> {
        let safety = &candidates[..SAFETY_LABELS.len()];
        let categories = &candidates[SAFETY_LABELS.len()..];
        match readout {
            Readout::Safety => safety.to_vec(),
            Readout::Unsafe => vec![logsumexp(safety[0], safety[1]), safety[2]],
            Readout::UnsafeStrict => vec![safety[0], logsumexp(safety[1], safety[2])],
            Readout::Category => categories.to_vec(),
        }
    }

    fn answer(
        &self,
        id: &str,
        question: &Question,
        readout: Readout,
        candidates: &[f32],
    ) -> Answer {
        let logits = self.option_logits(readout, candidates);
        let probabilities = softmax(&logits, 1.0);
        match question {
            Question::Choice(choice) => {
                let fixed = match readout {
                    Readout::Safety => &SAFETY_LABELS[..],
                    Readout::Category => &CATEGORY_LABELS[..],
                    _ => &[][..],
                };
                let labels: Vec<String> = choice.criteria.keys().cloned().collect();
                let ordered = labels
                    .iter()
                    .map(|label| {
                        fixed
                            .iter()
                            .position(|candidate| candidate == label)
                            .map_or(0.0, |index| probabilities[index])
                    })
                    .collect::<Vec<_>>();
                choice_answer(&labels, &ordered)
            }
            Question::Noul(_) => {
                let false_probability = probabilities.first().copied().unwrap_or(0.5);
                let true_probability = probabilities.get(1).copied().unwrap_or(0.5);
                noul_answer(false_probability, true_probability)
            }
            Question::Score(_) | Question::Box(_) => {
                unreachable!("plan rejected {id:?} for Qwen3Guard")
            }
        }
    }
}

impl DecisionModel for Qwen3GuardModel {
    fn predict(&self, state: &State, questions: &Questions) -> Result<DecisionOutput, ModelError> {
        if questions.is_empty() {
            return Err(ModelError::invalid_request("questions must not be empty"));
        }
        let readouts = self.plan(questions)?;
        let state_text = render_state(state);
        let base = format!("{PROMPT_HEAD}{state_text}{PROMPT_TAIL}");
        let safety = self.encode(&format!("{base}{SAFETY_ROW}"))?;
        let category = readouts
            .contains(&Readout::Category)
            .then(|| self.encode(&format!("{base}{CATEGORY_ROW}")))
            .transpose()?;
        if [&safety, category.as_ref().unwrap_or(&safety)]
            .iter()
            .any(|row| row.len() > self.max_len)
        {
            return Err(ModelError::context_length_exceeded(format!(
                "Qwen3Guard rows must be at most {} tokens",
                self.max_len
            )));
        }
        let mut rows = vec![safety];
        if let Some(category) = category {
            rows.push(category);
        }
        let logits = self.forward(&rows)?;
        let mut answers = IndexMap::with_capacity(readouts.len());
        for ((id, question), readout) in questions.iter().zip(readouts.iter()) {
            let row = if *readout == Readout::Category { 1 } else { 0 };
            answers.insert(
                id.clone(),
                self.answer(id, question, *readout, &logits[row]),
            );
        }
        let input_tokens = rows.iter().map(|row| row.len() as u64).sum();
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

fn logsumexp(a: f32, b: f32) -> f32 {
    let max = a.max(b);
    max + ((a - max).exp() + (b - max).exp()).ln()
}

fn inference_error(error: candle_core::Error) -> ModelError {
    ModelError::inference(error.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn preset_questions_have_the_fixed_shapes() {
        let questions = Qwen3GuardModel::preset_questions();
        assert_eq!(questions.len(), 4);
        let safety = questions.get("safety").unwrap();
        let Question::Choice(safety) = safety else {
            panic!("choice");
        };
        assert_eq!(safety.criteria.len(), 3);
        let category = questions.get("category").unwrap();
        let Question::Choice(category) = category else {
            panic!("choice");
        };
        assert_eq!(category.criteria.len(), 10);
    }

    #[test]
    fn logsumexp_is_stable() {
        assert!((logsumexp(1000.0, 1000.0) - (1000.0 + 2f32.ln())).abs() < 1e-3);
    }
}
