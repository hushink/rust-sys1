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
