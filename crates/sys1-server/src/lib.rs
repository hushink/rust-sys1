//! Strict TypeSafe-compatible System One HTTP server.

#![forbid(unsafe_code)]

mod body;
mod cache;
mod config;
mod error;
mod metrics;
mod server;

pub use cache::{CacheManager, CachedFile, CachedModel};
pub use config::{BatchConfig, ServerConfig};
pub use server::{
    AppState, router, router_with_extensions, router_with_resolver, serve, serve_with_extensions,
    serve_with_resolver,
};
