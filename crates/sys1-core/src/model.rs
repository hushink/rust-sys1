use crate::{Answers, ModelManifest, Questions};
use indexmap::IndexMap;
use std::sync::Arc;
use sys1_protocol::{State, Usage};
use thiserror::Error;

/// One inference request inside a batch.
#[derive(Clone, Copy, Debug)]
pub struct Prediction<'a> {
    pub state: &'a State,
    pub questions: &'a Questions,
}

impl<'a> Prediction<'a> {
    pub fn new(state: &'a State, questions: &'a Questions) -> Self {
        Self { state, questions }
    }
}

/// Options for scanning a state that does not fit the model's context window.
///
/// `None` selects the model default: a full state budget window with a 50%
/// overlap, and the model's internal batch cap. `batch_size` bounds how many
/// windows share one forward pass.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct LongContextOptions {
    pub window: Option<usize>,
    pub stride: Option<usize>,
    pub batch_size: Option<usize>,
}

/// The token window that produced a long-context answer.
///
/// `token_start` and `token_end` are byte-free token offsets into the
/// tokenized state, not character offsets. `count` is the total number of
/// windows scored for the request.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct WindowAttribution {
    pub index: usize,
    pub token_start: usize,
    pub token_end: usize,
    pub count: usize,
}

/// A loaded decision model.
pub trait DecisionModel: Send + Sync {
    fn predict(&self, state: &State, questions: &Questions) -> Result<DecisionOutput, ModelError>;

    /// Scan a state in overlapping windows when it exceeds the context window,
    /// then aggregate per-question answers.
    ///
    /// The default refuses the request: silently truncating a state is the
    /// behavior this method exists to avoid.
    fn predict_long(
        &self,
        state: &State,
        questions: &Questions,
        options: LongContextOptions,
    ) -> Result<DecisionOutput, ModelError> {
        let _ = (state, questions, options);
        Err(ModelError::unsupported(
            "long-context scanning is not implemented by this model",
        ))
    }

    /// Whether the server should route concurrent requests through its
    /// cross-request batcher. CPU models that already parallelize inside one
    /// forward pass can return `false`; the server then runs requests
    /// independently and lets them compete for CPU threads.
    fn supports_cross_request_batching(&self) -> bool {
        true
    }

    /// Batched inference. Models that can share a forward pass should override
    /// this; the default preserves correctness by running each request.
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

/// Loads a model implementation from a manifest.
pub trait ModelFactory: Send + Sync {
    fn load(&self, manifest: &ModelManifest) -> Result<Arc<dyn DecisionModel>, ModelError>;
}

/// Resolves a user-supplied model reference at runtime.
///
/// The reference is implementation-defined: a hub id (`hf:org/repo`), a local
/// directory, or an HTTP manifest URL. `resolve` must materialize whatever it
/// needs and register the model with the factory the registry uses, because
/// the returned manifests are loaded through that factory. It must not load
/// the weights itself.
pub trait ModelResolver: Send + Sync {
    fn resolve(&self, reference: &str) -> Result<Vec<ModelManifest>, ModelError>;

    /// Resolve and register under an explicit catalog name. The default
    /// ignores the name; sources with a name-override facility should
    /// override it.
    fn resolve_as(&self, name: &str, reference: &str) -> Result<Vec<ModelManifest>, ModelError> {
        let _ = name;
        self.resolve(reference)
    }

    /// Forget resolver-side state (downloads, directory registrations) for a
    /// model that was removed from the catalog.
    fn release(&self, name: &str) -> bool {
        let _ = name;
        false
    }
}

/// Typed answers plus token usage. The registry wraps this into the official
/// response shape and fills the canonical model name.
#[derive(Clone, Debug, PartialEq)]
pub struct DecisionOutput {
    pub answers: Answers,
    pub usage: Usage,
    /// Optional per-question action probabilities produced by a model's act
    /// head. The official System One wire response does not carry this field;
    /// the typed in-process API exposes it.
    pub actions: IndexMap<String, f32>,
    /// Per-question deciding window for a long-context call. Empty for a
    /// normal `predict` and for a `predict_long` call whose state fit one
    /// window.
    pub windows: IndexMap<String, WindowAttribution>,
    /// Number of windows the model scored for a long-context call: `0` for a
    /// normal `predict`, `1` when the state fit one window, and `N` for a
    /// scanned document.
    pub scored_windows: usize,
}

impl DecisionOutput {
    pub fn new(answers: Answers, usage: Usage) -> Self {
        Self {
            answers,
            usage,
            actions: IndexMap::new(),
            windows: IndexMap::new(),
            scored_windows: 0,
        }
    }

    pub fn with_actions(mut self, actions: IndexMap<String, f32>) -> Self {
        self.actions = actions;
        self
    }

    pub fn with_windows(
        mut self,
        windows: IndexMap<String, WindowAttribution>,
        scored_windows: usize,
    ) -> Self {
        self.windows = windows;
        self.scored_windows = scored_windows;
        self
    }

    pub fn action(&self, id: &str) -> Option<f32> {
        self.actions.get(id).copied()
    }

    pub fn window(&self, id: &str) -> Option<&WindowAttribution> {
        self.windows.get(id)
    }
}

/// Errors raised by a model implementation.
#[derive(Clone, Debug, Error, PartialEq, Eq)]
pub enum ModelError {
    #[error("model files are missing or invalid: {message}")]
    Load { message: String },
    #[error("invalid request for this model: {message}")]
    InvalidRequest { message: String },
    #[error("inference failed: {message}")]
    Inference { message: String },
    #[error("context length exceeded: {message}")]
    ContextLengthExceeded { message: String },
    #[error("unsupported: {message}")]
    Unsupported { message: String },
}

impl ModelError {
    pub fn load(message: impl Into<String>) -> Self {
        Self::Load {
            message: message.into(),
        }
    }

    pub fn invalid_request(message: impl Into<String>) -> Self {
        Self::InvalidRequest {
            message: message.into(),
        }
    }

    pub fn inference(message: impl Into<String>) -> Self {
        Self::Inference {
            message: message.into(),
        }
    }

    pub fn context_length_exceeded(message: impl Into<String>) -> Self {
        Self::ContextLengthExceeded {
            message: message.into(),
        }
    }

    pub fn unsupported(message: impl Into<String>) -> Self {
        Self::Unsupported {
            message: message.into(),
        }
    }
}
