//! GLiClass uni-encoder decision model (`gliclass-uni-v1`) on candle.
//!
//! This implementation targets the ModernBERT-backed edge checkpoint
//! (`knowledgator/gliclass-instruct-edge-v1.0`): labels and text share one
//! sequence, label spans are mean-pooled, and an MLP scores each label against
//! the `[CLS]` representation.

#![forbid(unsafe_code)]

mod config;
mod factory;
mod model;
mod render;

pub use config::GliclassConfig;
pub use factory::{GliclassFactory, is_gliclass_directory};
pub use model::{GliclassModel, GliclassOptions};
