//! Core traits, manifest, registry, and typed in-process API for rust-sys1.

#![forbid(unsafe_code)]

mod answers;
mod device;
mod error;
mod manifest;
mod model;
mod questions;
mod registry;

pub use answers::Answers;
pub use device::{DeviceError, DeviceParseError, DevicePreference, ResolvedDevice};
pub use error::{AnswerError, Error};
pub use manifest::{ManifestError, ModelFile, ModelManifest};
pub use model::{
    DecisionModel, DecisionOutput, ModelError, ModelFactory, ModelResolver, Prediction,
};
pub use questions::Questions;
pub use registry::{ModelOrigin, ModelRegistry, RegistryConfig};
