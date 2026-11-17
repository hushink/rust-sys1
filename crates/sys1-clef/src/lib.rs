//! Clef (`clef-joint-v1`) for rust-sys1.
//!
//! A Qwen3.5 backbone reads the state and schema, and Clef's joint schema head
//! routes state evidence to every question and scores every option together in
//! one forward pass.

#![forbid(unsafe_code)]

mod factory;
mod model;

pub use factory::{ClefFactory, is_clef_directory};
pub use model::{ClefModel, ClefOptions};
