//! Strict TypeSafe-compatible System One HTTP server.

#![forbid(unsafe_code)]

mod body;
mod config;
mod error;
mod metrics;
mod server;

pub use config::{BatchConfig, ServerConfig};
pub use server::{AppState, router, router_with_resolver, serve, serve_with_resolver};
