//! Von 1.1 (`von-option-marker-v1`) as a candle decision model.
//!
//! Von is a ModernBERT-large encoder with an option-marker scorer. Every
//! option gets a `[MASK]` token in one row and is scored from that token's
//! hidden state. Zero-shot noul questions add a second, empty-state row whose
//! bias is subtracted from the true side.

#![forbid(unsafe_code)]

mod factory;
mod model;

pub use factory::{VonFactory, is_von_directory};
pub use model::{VonModel, VonOptions};
