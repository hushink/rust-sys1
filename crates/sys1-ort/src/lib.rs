//! ONNX Runtime execution layer.
//!
//! This crate is intentionally narrower than a model family: it owns process
//! and session lifecycle for ONNX Runtime, while family crates keep
//! tokenization, sequence layout, and answer formatting.
//!
//! The scoring contract is designed for token-classification decision models
//! exported from a family implementation:
//!
//! | Input | Type | Shape |
//! | --- | --- | --- |
//! | [`DEFAULT_INPUT_IDS`] | int64 | `[batch, seq]` |
//! | [`DEFAULT_ATTENTION_MASK`] | int64 | `[batch, seq]` |
//! | [`DEFAULT_TYPE_IDS`] | int64 | `[batch]` |
//! | [`DEFAULT_MARKERS`] | int64 | `[n]` |
//!
//! `markers` contains flattened positions into `[batch * seq]`. The graph
//! gathers hidden states at those positions and produces one float32 logit per
//! marker under [`DEFAULT_OUTPUT`].
//!
//! ONNX Runtime is loaded dynamically. Set `ORT_DYLIB_PATH` to an
//! `onnxruntime` shared library, or place it on the platform loader search
//! path. This keeps builds offline and avoids a link-time dependency.

#![forbid(unsafe_code)]

mod error;
mod session;

pub use error::OrtError;
pub use session::{
    DEFAULT_ATTENTION_MASK, DEFAULT_INPUT_IDS, DEFAULT_MARKERS, DEFAULT_OUTPUT, DEFAULT_TYPE_IDS,
    ScoringInputs, ScoringSession, SessionConfig,
};

/// The ONNX Runtime version this crate is built against.
pub const ONNX_RUNTIME_API_VERSION: u32 = 17;
