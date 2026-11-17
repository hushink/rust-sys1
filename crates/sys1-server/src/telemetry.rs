use axum::http::HeaderMap;
use std::fmt;
use std::time::Duration;
use uuid::Uuid;

pub const TRACEPARENT_HEADER: &str = "traceparent";

/// Correlation ids for one HTTP request.
///
/// A valid incoming W3C `traceparent` keeps its 128-bit trace id; otherwise a
/// new trace id is generated. Every response receives a `traceparent` header
/// with a fresh 64-bit server span id.
#[derive(Clone, Debug)]
pub struct TraceContext {
    pub request_id: String,
    pub trace_id: String,
    pub span_id: String,
    pub sampled: bool,
}

impl TraceContext {
    pub fn from_headers(headers: &HeaderMap) -> Self {
        let request_id = Uuid::new_v4().to_string();
        match headers
            .get(TRACEPARENT_HEADER)
            .and_then(|value| value.to_str().ok())
            .and_then(parse_traceparent)
        {
            Some((trace_id, sampled)) => Self {
                request_id,
                trace_id,
                span_id: new_span_id(),
                sampled,
            },
            None => Self::generate_with_request_id(request_id),
        }
    }

    pub fn generate() -> Self {
        Self::generate_with_request_id(Uuid::new_v4().to_string())
    }

    fn generate_with_request_id(request_id: String) -> Self {
        Self {
            request_id,
            trace_id: Uuid::new_v4().simple().to_string(),
            span_id: new_span_id(),
            sampled: true,
        }
    }

    pub fn traceparent(&self) -> String {
        format!(
            "00-{}-{}-{:02x}",
            self.trace_id,
            self.span_id,
            u8::from(self.sampled)
        )
    }
}

fn new_span_id() -> String {
    Uuid::new_v4().simple().to_string()[..16].to_owned()
}

fn parse_traceparent(value: &str) -> Option<(String, bool)> {
    let mut parts = value.split('-');
    let version = parts.next()?;
    let trace_id = parts.next()?;
    let span_id = parts.next()?;
    let flags = parts.next()?;
    if parts.next().is_some()
        || version.len() != 2
        || !is_hex(version)
        || version == "ff"
        || trace_id.len() != 32
        || !is_hex(trace_id)
        || trace_id.bytes().all(|byte| byte == b'0')
        || span_id.len() != 16
        || !is_hex(span_id)
        || span_id.bytes().all(|byte| byte == b'0')
        || flags.len() != 2
        || !is_hex(flags)
    {
        return None;
    }
    let sampled = u8::from_str_radix(flags, 16).ok()? & 1 == 1;
    Some((trace_id.to_owned(), sampled))
}

fn is_hex(value: &str) -> bool {
    value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

/// One completed System One HTTP request.
#[derive(Clone, Debug)]
pub struct RequestEvent {
    pub request_id: String,
    pub trace_id: String,
    pub model: Option<String>,
    pub status: u16,
    pub elapsed: Duration,
    pub error: Option<String>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ModelAction {
    Load,
    Reload,
    Unload,
    Register,
    Remove,
}

impl fmt::Display for ModelAction {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        let name = match self {
            Self::Load => "load",
            Self::Reload => "reload",
            Self::Unload => "unload",
            Self::Register => "register",
            Self::Remove => "remove",
        };
        formatter.write_str(name)
    }
}

/// A model lifecycle operation, including its request correlation ids.
#[derive(Clone, Debug)]
pub struct ModelEvent {
    pub request_id: String,
    pub trace_id: String,
    pub model: String,
    pub action: ModelAction,
    pub success: bool,
    pub elapsed: Duration,
    pub error: Option<String>,
}

#[derive(Clone, Debug)]
pub struct BatchEvent {
    pub items: usize,
    pub elapsed: Duration,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CacheAction {
    List,
    Remove,
}

/// A disk-cache operation, including its request correlation ids.
#[derive(Clone, Debug)]
pub struct CacheEvent {
    pub request_id: String,
    pub trace_id: String,
    pub key: Option<String>,
    pub action: CacheAction,
    pub success: bool,
    pub error: Option<String>,
}

/// A config reload attempt.
#[derive(Clone, Debug)]
pub struct ConfigEvent {
    pub request_id: String,
    pub trace_id: String,
    pub success: bool,
    pub added: Vec<String>,
    pub removed: Vec<String>,
    pub error: Option<String>,
}

/// Replaceable structured event sink.
///
/// The server only emits events; it never installs a `tracing` subscriber.
/// Embedded users can implement this trait for audit logs, OpenTelemetry, or
/// another telemetry system, and pass it through [`crate::ServerExtensions`].
/// All methods default to no-ops so implementations can override only what
/// they need.
pub trait Telemetry: Send + Sync {
    fn on_request(&self, _event: &RequestEvent) {}
    fn on_model(&self, _event: &ModelEvent) {}
    fn on_batch(&self, _event: &BatchEvent) {}
    fn on_cache(&self, _event: &CacheEvent) {}
    fn on_config(&self, _event: &ConfigEvent) {}
}

#[derive(Default)]
pub struct NoopTelemetry;

impl Telemetry for NoopTelemetry {}

/// The default CLI telemetry: structured `tracing` events with request and
/// trace ids on the `sys1.telemetry` target.
#[derive(Default)]
pub struct TracingTelemetry;

impl Telemetry for TracingTelemetry {
    fn on_request(&self, event: &RequestEvent) {
        if let Some(error) = &event.error {
            tracing::warn!(
                target: "sys1.telemetry",
                request_id = %event.request_id,
                trace_id = %event.trace_id,
                model = event.model.as_deref().unwrap_or("-"),
                status = event.status,
                elapsed_ms = event.elapsed.as_secs_f64() * 1000.0,
                error = %error,
                "request completed"
            );
        } else {
            tracing::info!(
                target: "sys1.telemetry",
                request_id = %event.request_id,
                trace_id = %event.trace_id,
                model = event.model.as_deref().unwrap_or("-"),
                status = event.status,
                elapsed_ms = event.elapsed.as_secs_f64() * 1000.0,
                "request completed"
            );
        }
    }

    fn on_model(&self, event: &ModelEvent) {
        tracing::info!(
            target: "sys1.telemetry",
            request_id = %event.request_id,
            trace_id = %event.trace_id,
            model = %event.model,
            action = %event.action,
            success = event.success,
            elapsed_ms = event.elapsed.as_secs_f64() * 1000.0,
            error = event.error.as_deref().unwrap_or(""),
            "model event"
        );
    }

    fn on_batch(&self, event: &BatchEvent) {
        tracing::debug!(
            target: "sys1.telemetry",
            items = event.items,
            elapsed_ms = event.elapsed.as_secs_f64() * 1000.0,
            "batch completed"
        );
    }

    fn on_cache(&self, event: &CacheEvent) {
        tracing::info!(
            target: "sys1.telemetry",
            request_id = %event.request_id,
            trace_id = %event.trace_id,
            key = event.key.as_deref().unwrap_or("-"),
            action = ?event.action,
            success = event.success,
            error = event.error.as_deref().unwrap_or(""),
            "cache event"
        );
    }

    fn on_config(&self, event: &ConfigEvent) {
        tracing::info!(
            target: "sys1.telemetry",
            request_id = %event.request_id,
            trace_id = %event.trace_id,
            success = event.success,
            added = ?event.added,
            removed = ?event.removed,
            error = event.error.as_deref().unwrap_or(""),
            "config reload"
        );
    }
}
