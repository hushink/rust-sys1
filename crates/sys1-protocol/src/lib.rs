//! System One wire types and validation.
//!
//! This crate is the single source of truth for the official TypeSafe
//! `POST /v1/systemone` and `GET /v1/models` contract.

#![forbid(unsafe_code)]

mod answer;
mod content;
mod error;
mod model;
mod parse;
mod question;
mod request;
mod response;
mod state;

pub use answer::*;
pub use content::*;
pub use error::*;
pub use model::*;
pub use parse::*;
pub use question::*;
pub use request::*;
pub use response::*;
pub use state::*;

/// Official decision endpoint.
pub const SYSTEMONE_PATH: &str = "/v1/systemone";

/// Official model-list endpoint.
pub const MODELS_PATH: &str = "/v1/models";

/// Response header used for request correlation.
pub const REQUEST_ID_HEADER: &str = "x-typesafe-request-id";

/// Opt-in header required for box questions and image assets.
pub const DANGER_ZONE_IMAGES_HEADER: &str = "typesafe-danger-zone-images";

/// Opt-in header for long-context requests.
pub const DANGER_ZONE_LONG_CONTEXT_HEADER: &str = "typesafe-danger-zone-long-context";
