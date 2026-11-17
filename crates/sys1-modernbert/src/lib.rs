//! ModernBERT encoder shared by rust-sys1 model families.
//!
//! The encoder accepts a `VarBuilder` rooted at the checkpoint and a path
//! prefix for the Hugging Face module (`model` for Laya/NLI,
//! `model.encoder_model` for GLiClass). Token embeddings are exposed
//! separately so families can add their own segment/type embeddings before
//! the encoder norm and layers.

#![forbid(unsafe_code)]

mod config;
mod encoder;

pub use config::ModernBertConfig;
pub use encoder::Encoder;
