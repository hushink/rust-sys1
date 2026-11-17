//! Model sources, manifest resolution, and the on-disk model cache.

#![forbid(unsafe_code)]

mod config;
mod download;
mod error;
mod hf;
mod http;
mod hub;
mod local;
mod modelscope;
mod reference;
mod source;

pub use config::HubConfig;
pub use error::HubError;
pub use hf::HfSource;
pub use http::HttpSource;
pub use hub::{Hub, PulledModel};
pub use local::LocalSource;
pub use modelscope::ModelScopeSource;
pub use reference::ModelRef;
pub use source::{ModelSource, SourceContext, default_cache_key};
