//! Kev (`kev-pointer-v1`) for rust-sys1.
//!
//! Kev pairs a Qwen3.5 base (optionally with its PEFT LoRA adapter) with a
//! two-projection pointer head. Each question is one causal row ending in a
//! decide token; every option is scored from its closing token against that
//! decide token.

#![forbid(unsafe_code)]

mod factory;
mod model;

pub use factory::{KevFactory, is_kev_directory};
pub use model::{KevModel, KevOptions};
