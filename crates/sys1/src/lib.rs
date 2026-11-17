//! Umbrella crate for the rust-sys1 in-process API.
//!
//! ```no_run
//! use std::path::Path;
//! use sys1::{DecisionModel, DecisionOutput, LayaModel, LayaOptions, Questions, State};
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let model = LayaModel::load_dir(Path::new("path/to/laya"), LayaOptions::default())?;
//! let state = State::text("I was charged twice.");
//! let questions = Questions::new().noul("billing", "Is this about billing?");
//! let output: DecisionOutput = model.predict(&state, &questions)?;
//! let billing = output.answers.noul("billing")?;
//! # Ok(())
//! # }
//! ```

#![forbid(unsafe_code)]

pub mod core {
    pub use sys1_core::*;
}

pub mod protocol {
    pub use sys1_protocol::*;
}

#[cfg(feature = "hub")]
pub mod hub {
    pub use sys1_hub::*;
}

#[cfg(feature = "laya")]
pub mod laya {
    pub use sys1_laya::*;
}

pub use sys1_core::{
    AnswerError, Answers, DecisionModel, DecisionOutput, ModelError, ModelFactory, ModelManifest,
    ModelRegistry, Prediction, Questions, RegistryConfig,
};
pub use sys1_protocol::{
    Answer, ChoiceAnswer, Content, ModelMetadata, ModelMetadataList, NoulAnswer, Question,
    ScoreAnswer, State, SystemOneResponse, Usage,
};

#[cfg(feature = "hub")]
pub use sys1_hub::{Hub, HubConfig, ModelRef, ModelSource, PulledModel};

#[cfg(feature = "laya")]
pub use sys1_laya::{ConfidenceMode, LayaFactory, LayaModel, LayaOptions, LayaRouter};
