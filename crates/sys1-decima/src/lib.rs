//! Decima (`decima-late-interaction-v1`/`-v2`) for rust-sys1.
//!
//! Every question becomes one state row and one row per option. A ModernBERT
//! encoder (mmBERT for `decima-base`) produces token states, and a late
//! interaction scorer lets each option read the state before producing a raw
//! score. Score questions use Decima's cumulative-link ordinal head.

#![forbid(unsafe_code)]

mod factory;
mod model;

pub use factory::{DecimaFactory, is_decima_directory};
pub use model::{DecimaConfig, DecimaModel, DecimaOptions, Limits};
