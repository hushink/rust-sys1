//! Qwen3Guard Gen as a fixed-preset decision model.
//!
//! Qwen3Guard keeps its policy, categories and output format in its chat
//! template. The model therefore answers only its built-in `safety`, `unsafe`,
//! `unsafe_strict` and `category` questions; the state is the text to judge.

#![forbid(unsafe_code)]

mod factory;
mod model;

pub use factory::{Qwen3GuardFactory, is_qwen3guard_directory};
pub use model::{Qwen3GuardModel, Qwen3GuardOptions};
