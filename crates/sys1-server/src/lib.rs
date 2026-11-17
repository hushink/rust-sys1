//! Strict TypeSafe-compatible System One HTTP server.

#![forbid(unsafe_code)]

mod body;
mod cache;
mod config;
mod error;
mod metrics;
mod server;
mod telemetry;

pub use cache::{CacheManager, CachedFile, CachedModel};
pub use config::{BatchConfig, ServerConfig};
pub use server::{
    AppState, ServerExtensions, router, router_with_extensions, router_with_resolver, serve,
    serve_with_extensions, serve_with_resolver,
};
pub use telemetry::{
    BatchEvent, CacheAction, CacheEvent, ModelAction, ModelEvent, NoopTelemetry, RequestEvent,
    TRACEPARENT_HEADER, Telemetry, TraceContext, TracingTelemetry,
};
