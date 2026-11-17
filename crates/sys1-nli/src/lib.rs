//! Zero-shot NLI decision model (`nli-pairs-v1`) on ModernBERT.
//!
//! Each option becomes an NLI hypothesis; the `(state, hypothesis)` pair is
//! scored by a ModernBERT cross-encoder. This is the candle implementation of
//! the `MoritzLaurer/*-zeroshot-v2.0` family.

#![forbid(unsafe_code)]

mod config;
mod factory;
mod model;
mod render;

pub use config::{NliConfig, NliTemplates, Pooling};
pub use factory::{NliFactory, is_nli_directory};
pub use model::{NliModel, NliOptions};
