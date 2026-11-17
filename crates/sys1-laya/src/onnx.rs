use crate::LayaOptions;
use crate::calibration::ConfidenceMode;
use crate::config::LayaConfig;
use crate::long;
use crate::model::ActHead;
use crate::onnx_config::OnnxFileConfig;
use crate::prepare::{Item, PreparedRequest, deduplicate, prepare_request};
use crate::response::build_response;
use crate::tokenizer::LayaTokenizer;
use candle_core::{DType, Device, Tensor};
use candle_nn::Linear;
use std::path::Path;
use sys1_core::{
    DecisionModel, DecisionOutput, LongContextOptions, ModelError, Prediction, Questions,
};
use sys1_ort::{ScoringInputs, ScoringSession, SessionConfig};
use sys1_protocol::State;

type ScoredItem = (Vec<f32>, Option<f32>);

struct OnnxActHead {
    head: ActHead,
    hidden: usize,
}

impl OnnxActHead {
    fn load(path: &Path) -> Result<Self, ModelError> {
        let tensors = candle_core::safetensors::load(path, &Device::Cpu).map_err(|error| {
            ModelError::load(format!("read act head {}: {error}", path.display()))
        })?;
        let tensor = |name: &str| -> Result<Tensor, ModelError> {
            tensors
                .get(name)
                .ok_or_else(|| {
                    ModelError::load(format!(
                        "act head {} is missing tensor {name}",
                        path.display()
                    ))
                })?
                .to_dtype(DType::F32)
                .map_err(|error| ModelError::load(error.to_string()))
        };

        let input_weight = tensor("act_head.0.weight")?;
        let input_bias = tensor("act_head.0.bias")?;
        let output_weight = tensor("act_head.2.weight")?;
        let output_bias = tensor("act_head.2.bias")?;
        let (rows, columns) = input_weight
            .dims2()
            .map_err(|error| ModelError::load(error.to_string()))?;
        if rows != 256 || columns < 5 {
            return Err(ModelError::load(format!(
                "act_head.0.weight must have shape [256, hidden + 4], got [{rows}, {columns}]"
            )));
        }
        let hidden = columns - 4;
        if input_bias.dims1().ok() != Some(256)
            || output_weight.dims2().ok() != Some((2, 256))
            || output_bias.dims1().ok() != Some(2)
        {
            return Err(ModelError::load(
                "act head has unexpected bias or output-weight shape",
            ));
        }

        Ok(Self {
            head: ActHead::from_parts(
                Linear::new(input_weight, Some(input_bias)),
                Linear::new(output_weight, Some(output_bias)),
                DType::F32,
            ),
            hidden,
        })
    }

    fn forward(&self, pooled: &[f32], logits: &[f32]) -> Result<f32, ModelError> {
        let pooled = Tensor::new(pooled, &Device::Cpu)
            .map_err(|error| ModelError::inference(error.to_string()))?;
        self.head
            .forward(&pooled, logits)
            .map_err(|error| ModelError::inference(error.to_string()))
    }
}

/// A Laya checkpoint executed by ONNX Runtime.
///
/// The graph contract is documented in the crate README. Tokenization,
/// sequence layout, calibration, and response formatting are shared with the
/// candle implementation, so both backends produce the same wire answers when
/// the ONNX export is numerically equivalent.
pub struct LayaOnnxModel {
    tokenizer: LayaTokenizer,
    config: LayaConfig,
    session: ScoringSession,
    confidence_mode: ConfidenceMode,
    max_batch: usize,
    act_head: Option<OnnxActHead>,
}

impl LayaOnnxModel {
    /// Load `onnx.json`, the tokenizer, and `rl_agent_config.json` from a
    /// model directory.
    pub fn load_dir(directory: &Path, options: LayaOptions) -> Result<Self, ModelError> {
        let mut config =
            LayaConfig::load(&directory.join("rl_agent_config.json")).map_err(ModelError::load)?;
        if let Some(max_len) = options.max_len {
            if max_len == 0 {
                return Err(ModelError::load("max_len must be greater than zero"));
            }
            config.max_len = max_len;
        }
        if config.max_len == 0 {
            return Err(ModelError::load("max_len must be greater than zero"));
        }
        if config.head_max_len == 0 {
            return Err(ModelError::load("head_max_len must be greater than zero"));
        }

        let tokenizer = LayaTokenizer::load(directory).map_err(ModelError::load)?;
        let onnx = OnnxFileConfig::load(directory)?;
        let model_path = onnx.model_path(directory)?;
        if !model_path.is_file() {
            return Err(ModelError::load(format!(
                "missing ONNX model at {}",
                model_path.display()
            )));
        }

        let act_head = onnx
            .act_head_path(directory)?
            .as_deref()
            .map(OnnxActHead::load)
            .transpose()?;
        let pooled_output = onnx.pooled_output(act_head.is_some());

        let mut session_config = SessionConfig::new(model_path);
        session_config.input_ids = onnx.input_ids;
        session_config.attention_mask = onnx.attention_mask;
        session_config.type_ids = onnx.type_ids;
        session_config.markers = onnx.markers;
        session_config.output = onnx.output;
        session_config.pooled_output = pooled_output;
        session_config.intra_threads = onnx.intra_threads;
        session_config.inter_threads = onnx.inter_threads;
        let session = ScoringSession::load(session_config)
            .map_err(|error| ModelError::load(error.to_string()))?;

        Ok(Self {
            tokenizer,
            config,
            session,
            confidence_mode: options.confidence_mode,
            max_batch: onnx.max_batch.max(1),
            act_head,
        })
    }

    fn forward(&self, prepared: &[PreparedRequest]) -> Result<Vec<ScoredItem>, ModelError> {
        let items: Vec<&Item> = prepared
            .iter()
            .flat_map(|request| request.items.iter())
            .collect();
        let (unique, indices) = deduplicate(&items);
        if unique.is_empty() {
            return Ok(Vec::new());
        }

        let mut output = Vec::with_capacity(unique.len());
        for chunk in unique.chunks(self.max_batch) {
            output.extend(self.score_items(chunk)?);
        }
        Ok(indices
            .into_iter()
            .map(|index| output[index].clone())
            .collect())
    }

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

    fn score_items(&self, items: &[&Item]) -> Result<Vec<ScoredItem>, ModelError> {
        let batch = items.len();
        let seq = items.iter().map(|item| item.ids.len()).max().unwrap_or(0);
        if batch == 0 || seq == 0 {
            return Ok(vec![(Vec::new(), None); batch]);
        }

        let mut input_ids = vec![self.tokenizer.pad_id as i64; batch * seq];
        let mut attention_mask = vec![0_i64; batch * seq];
        let mut type_ids = vec![0_i64; batch];
        let mut counts = Vec::with_capacity(batch);
        let mut markers = Vec::new();

        for (row, item) in items.iter().enumerate() {
            let start = row * seq;
            for (offset, id) in item.ids.iter().enumerate() {
                input_ids[start + offset] = *id as i64;
                attention_mask[start + offset] = 1;
            }
            type_ids[row] = item.question.kind as i64;
            counts.push(item.markers.len());
            markers.extend(
                item.markers
                    .iter()
                    .map(|position| (start + position) as i64),
            );
        }

        let outputs = self
            .session
            .score(ScoringInputs {
                input_ids: &input_ids,
                attention_mask: &attention_mask,
                type_ids: &type_ids,
                markers: &markers,
                batch,
                seq,
            })
            .map_err(|error| ModelError::inference(error.to_string()))?;

        let expected: usize = counts.iter().sum();
        if outputs.logits.len() != expected {
            return Err(ModelError::inference(format!(
                "ONNX model returned {} logits for {expected} markers",
                outputs.logits.len()
            )));
        }

        let mut output = Vec::with_capacity(batch);
        let mut offset = 0;
        for (row, count) in counts.into_iter().enumerate() {
            let values = outputs.logits[offset..offset + count].to_vec();
            let action = match (&self.act_head, outputs.pooled.as_deref()) {
                (Some(head), Some(pooled)) => {
                    let start = row * head.hidden;
                    let end = start + head.hidden;
                    let pooled = pooled.get(start..end).ok_or_else(|| {
                        ModelError::inference(format!(
                            "ONNX pooled output has {} values for batch={batch} hidden={}",
                            pooled.len(),
                            head.hidden
                        ))
                    })?;
                    Some(head.forward(pooled, &values)?)
                }
                (Some(_), None) => {
                    return Err(ModelError::inference(
                        "ONNX model has an act head but produced no pooled output",
                    ));
                }
                _ => None,
            };
            output.push((values, action));
            offset += count;
        }
        Ok(output)
    }
}

impl DecisionModel for LayaOnnxModel {
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
