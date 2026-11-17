//! Decision 1.0 (`decision-endpoint-v1`) for rust-sys1.
//!
//! A fully fine-tuned Qwen3.5 backbone scores every option endpoint against a
//! global query through the checkpoint's endpoint head. Rows use Decision's
//! segmented prompt: one row per question, one candidate span per option.

#![forbid(unsafe_code)]

mod factory;
mod model;

pub use factory::{DecisionFactory, is_decision_directory};
pub use model::{DecisionModel, DecisionOptions};
