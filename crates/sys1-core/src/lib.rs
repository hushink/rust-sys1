//! Core traits, manifest, registry, and typed in-process API for rust-sys1.

#![forbid(unsafe_code)]

mod abstention;
mod answers;
mod device;
mod error;
mod manifest;
mod model;
mod questions;
mod registry;
mod scoring;
mod text;

pub use abstention::{
    Abstention, AbstentionState, DEFAULT_BUCKET_KEY, MinConfidence, PredictionOptions,
    answer_confidence, apply_abstention, bucket_key,
};
pub use answers::Answers;
pub use device::{DeviceError, DeviceParseError, DevicePreference, ResolvedDevice};
pub use error::{AnswerError, Error};
pub use manifest::{ManifestError, ModelFile, ModelManifest};
pub use model::{
    DecisionModel, DecisionOutput, LongContextOptions, ModelError, ModelFactory, ModelResolver,
    Prediction, WindowAttribution,
};
pub use questions::Questions;
pub use registry::{ModelOrigin, ModelRegistry, RegistryConfig};
pub use scoring::{
    answer_from_logits, argmax, choice_answer, noul_answer, score_answer, softmax,
    typesafe_confidence,
};
pub use text::{
    canonical_json, content_json, content_to_value, fill_template, float_repr, is_printable,
    python_json, python_repr, python_str, render_content, render_criterion, render_instructions,
    render_object_lines, render_state, sorted_object, split, strip, truthy,
};
