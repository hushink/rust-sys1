//! Zero-shot NLI decision model (`nli-pairs-v1`) on ModernBERT or DeBERTa-v3.
//!
//! Each option becomes an NLI hypothesis; the `(state, hypothesis)` pair is
//! scored by a cross-encoder. This is the candle implementation of the
//! `MoritzLaurer/*-zeroshot-v2.0` family, including the DeBERTa-v3-large
//! disentangled-attention checkpoint.

#![forbid(unsafe_code)]

mod config;
mod deberta;
mod factory;
mod model;
mod render;

pub use config::{NliConfig, NliEncoderConfig, NliTemplates, Pooling};
pub use factory::{NliFactory, is_nli_directory};
pub use model::{NliModel, NliOptions};
