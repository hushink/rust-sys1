use crate::OrtError;
use ort::environment::Environment;
use ort::session::Session;
use ort::value::Tensor;
use std::path::PathBuf;
use std::sync::{Mutex, OnceLock};

/// Default name of the token-id input.
pub const DEFAULT_INPUT_IDS: &str = "input_ids";
/// Default name of the attention-mask input.
pub const DEFAULT_ATTENTION_MASK: &str = "attention_mask";
/// Default name of the sequence-type input.
pub const DEFAULT_TYPE_IDS: &str = "type_ids";
/// Default name of the flattened marker-position input.
pub const DEFAULT_MARKERS: &str = "markers";
/// Default name of the float32 logits output.
pub const DEFAULT_OUTPUT: &str = "logits";

const DEFAULT_DYLIB: &str = if cfg!(target_os = "windows") {
    "onnxruntime.dll"
} else if cfg!(target_os = "linux") {
    "libonnxruntime.so"
} else if cfg!(target_os = "macos") {
    "libonnxruntime.dylib"
} else {
    "onnxruntime"
};

fn runtime_library() -> PathBuf {
    std::env::var_os("ORT_DYLIB_PATH")
        .filter(|value| !value.is_empty())
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(DEFAULT_DYLIB))
}

/// ONNX Runtime loading is process-wide and is attempted exactly once.
///
/// `ort`'s dynamic loader marks its internal lock complete even when loading
/// fails, so a second `init_from` call in the same process must not retry. The
/// cached result also serializes concurrent first-time loads.
static RUNTIME: OnceLock<Result<Environment, String>> = OnceLock::new();

fn init_environment(dylib: &std::path::Path) -> Result<Environment, String> {
    let builder = ort::init_from(dylib).map_err(|error| {
        format!(
            "cannot load {} (set ORT_DYLIB_PATH or install it on the loader path): {error}",
            dylib.display()
        )
    })?;
    builder
        .build()
        .map_err(|error| format!("failed to create an ONNX Runtime environment: {error}"))
}

fn ensure_runtime() -> Result<Environment, OrtError> {
    RUNTIME
        .get_or_init(|| {
            init_environment(&runtime_library()).inspect(|environment| {
                // Environment drop calls `ReleaseEnv`. At process exit the
                // static runtime handle may already be gone, so keep one
                // reference alive for the lifetime of the process.
                std::mem::forget(environment.clone());
            })
        })
        .clone()
        .map_err(OrtError::runtime)
}

/// Session construction options.
#[derive(Clone, Debug)]
pub struct SessionConfig {
    /// Path to the `.onnx` file.
    pub model: PathBuf,
    /// Token-id input name.
    pub input_ids: String,
    /// Attention-mask input name.
    pub attention_mask: String,
    /// Sequence-type input name.
    pub type_ids: String,
    /// Marker-position input name.
    pub markers: String,
    /// Logits output name.
    pub output: String,
    /// ONNX Runtime intra-op threads. `None` or `0` keeps the runtime default.
    pub intra_threads: Option<usize>,
    /// ONNX Runtime inter-op threads. `None` or `0` keeps the runtime default.
    pub inter_threads: Option<usize>,
}

impl SessionConfig {
    /// Use the default System One scoring contract with a model file.
    pub fn new(model: impl Into<PathBuf>) -> Self {
        Self {
            model: model.into(),
            input_ids: DEFAULT_INPUT_IDS.to_owned(),
            attention_mask: DEFAULT_ATTENTION_MASK.to_owned(),
            type_ids: DEFAULT_TYPE_IDS.to_owned(),
            markers: DEFAULT_MARKERS.to_owned(),
            output: DEFAULT_OUTPUT.to_owned(),
            intra_threads: None,
            inter_threads: None,
        }
    }
}

/// One padded batch of tokenized items.
#[derive(Clone, Copy, Debug)]
pub struct ScoringInputs<'a> {
    /// `batch * seq` token ids.
    pub input_ids: &'a [i64],
    /// `batch * seq` attention values, normally 0 or 1.
    pub attention_mask: &'a [i64],
    /// One sequence-type value per row.
    pub type_ids: &'a [i64],
    /// Flattened positions (`row * seq + offset`) to score.
    pub markers: &'a [i64],
    /// Number of padded rows.
    pub batch: usize,
    /// Padded row length.
    pub seq: usize,
}

impl ScoringInputs<'_> {
    /// Validate shapes and marker bounds before calling ONNX Runtime.
    pub fn validate(&self) -> Result<(), OrtError> {
        if self.batch == 0 || self.seq == 0 {
            return Err(OrtError::input("batch and seq must be greater than zero"));
        }
        let elements = self
            .batch
            .checked_mul(self.seq)
            .ok_or_else(|| OrtError::input("batch * seq overflows usize"))?;
        if self.input_ids.len() != elements {
            return Err(OrtError::input(format!(
                "input_ids has {} values, expected batch * seq = {elements}",
                self.input_ids.len()
            )));
        }
        if self.attention_mask.len() != elements {
            return Err(OrtError::input(format!(
                "attention_mask has {} values, expected batch * seq = {elements}",
                self.attention_mask.len()
            )));
        }
        if self.type_ids.len() != self.batch {
            return Err(OrtError::input(format!(
                "type_ids has {} values, expected batch = {}",
                self.type_ids.len(),
                self.batch
            )));
        }
        for (index, marker) in self.markers.iter().enumerate() {
            if *marker < 0 || *marker as usize >= elements {
                return Err(OrtError::input(format!(
                    "marker {index} is {marker}, outside 0..{elements}"
                )));
            }
        }
        Ok(())
    }
}

/// A loaded ONNX Runtime session for the System One scoring contract.
///
/// `Session::run` takes `&mut self`, so the session is wrapped in a mutex.
/// ONNX Runtime already parallelizes a single run internally; concurrent
/// requests serialize on this lock.
pub struct ScoringSession {
    session: Mutex<Session>,
    config: SessionConfig,
}

impl ScoringSession {
    /// Load an ONNX model and apply session options.
    ///
    /// With the `load-dynamic` feature, ONNX Runtime is located at runtime.
    /// Set `ORT_DYLIB_PATH` when the library is not on the loader search path.
    pub fn load(config: SessionConfig) -> Result<Self, OrtError> {
        for (label, value) in [
            ("input_ids", &config.input_ids),
            ("attention_mask", &config.attention_mask),
            ("type_ids", &config.type_ids),
            ("markers", &config.markers),
            ("output", &config.output),
        ] {
            if value.is_empty() {
                return Err(OrtError::input(format!("{label} name must not be empty")));
            }
        }

        let environment = ensure_runtime()?;

        let path = config.model.display().to_string();
        let mut builder =
            Session::builder(&environment).map_err(|error| OrtError::load(path.clone(), error))?;
        if let Some(threads) = config.intra_threads.filter(|threads| *threads > 0) {
            builder = builder
                .with_intra_threads(threads)
                .map_err(|error| OrtError::load(path.clone(), error))?;
        }
        if let Some(threads) = config.inter_threads.filter(|threads| *threads > 0) {
            builder = builder
                .with_inter_threads(threads)
                .map_err(|error| OrtError::load(path.clone(), error))?;
        }
        let session = builder
            .commit_from_file(&config.model)
            .map_err(|error| OrtError::load(path, error))?;
        Ok(Self {
            session: Mutex::new(session),
            config,
        })
    }

    /// Session options used to load this model.
    pub fn config(&self) -> &SessionConfig {
        &self.config
    }

    /// Score every marker in the batch, returning one logit per marker.
    pub fn score(&self, inputs: ScoringInputs<'_>) -> Result<Vec<f32>, OrtError> {
        inputs.validate()?;
        if inputs.markers.is_empty() {
            return Ok(Vec::new());
        }

        let input_ids = Tensor::from_array((
            [inputs.batch, inputs.seq],
            inputs.input_ids.to_vec().into_boxed_slice(),
        ))
        .map_err(OrtError::input)?;
        let attention_mask = Tensor::from_array((
            [inputs.batch, inputs.seq],
            inputs.attention_mask.to_vec().into_boxed_slice(),
        ))
        .map_err(OrtError::input)?;
        let type_ids =
            Tensor::from_array(([inputs.batch], inputs.type_ids.to_vec().into_boxed_slice()))
                .map_err(OrtError::input)?;
        let markers = Tensor::from_array((
            [inputs.markers.len()],
            inputs.markers.to_vec().into_boxed_slice(),
        ))
        .map_err(OrtError::input)?;

        let mut session = self
            .session
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let outputs = session
            .run(ort::inputs![
                self.config.input_ids.as_str() => input_ids,
                self.config.attention_mask.as_str() => attention_mask,
                self.config.type_ids.as_str() => type_ids,
                self.config.markers.as_str() => markers,
            ])
            .map_err(OrtError::inference)?;
        let value = outputs
            .get(self.config.output.as_str())
            .ok_or_else(|| OrtError::output(&self.config.output, "output name was not produced"))?;
        let (shape, logits) = value
            .try_extract_tensor::<f32>()
            .map_err(|error| OrtError::output(&self.config.output, error))?;
        let expected = inputs.markers.len();
        if shape.num_elements() != expected {
            return Err(OrtError::output(
                &self.config.output,
                format!(
                    "produced {} values for {expected} markers in shape {shape}",
                    shape.num_elements()
                ),
            ));
        }
        Ok(logits.to_vec())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validates_shapes_and_marker_bounds() {
        let ids = [1_i64, 2, 3, 4];
        let mask = [1_i64, 1, 1, 1];
        let types = [0_i64, 1];
        let valid = ScoringInputs {
            input_ids: &ids,
            attention_mask: &mask,
            type_ids: &types,
            markers: &[0, 3],
            batch: 2,
            seq: 2,
        };
        valid.validate().expect("valid inputs");

        let bad_marker = ScoringInputs {
            markers: &[4],
            ..valid
        };
        assert!(matches!(bad_marker.validate(), Err(OrtError::Input { .. })));

        let bad_batch = ScoringInputs { batch: 3, ..valid };
        assert!(matches!(bad_batch.validate(), Err(OrtError::Input { .. })));
    }

    #[test]
    fn session_is_send_and_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<ScoringSession>();
    }
}
