//! Kev (`kev-pointer-v1`) for rust-sys1.
//!
//! Kev pairs a Qwen3.5 base (optionally with its PEFT LoRA adapter) with a
//! two-projection pointer head. Each question is one causal row ending in a
//! decide token; every option is scored from its closing token against that
//! decide token.

#![forbid(unsafe_code)]

mod factory;
mod metadata;
mod model;

pub use factory::{KevFactory, has_kev_base, is_kev_adapter, is_kev_directory};
pub use metadata::{
    KevHeadMetadata, directory_metadata, local_base, read_head_metadata, resolve_base_directory,
};
pub use model::{KevModel, KevOptions};
