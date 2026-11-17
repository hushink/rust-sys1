use crate::body::read_body;
use crate::cache::{CacheManager, CachedModel};
use crate::config::{BatchConfig, ServerConfig};
use crate::error::{ApiError, map_model_error, map_registry_error};
use crate::metrics::Metrics;
use crate::telemetry::{
    BatchEvent, CacheAction, CacheEvent, ModelAction, ModelEvent, NoopTelemetry, RequestEvent,
    TRACEPARENT_HEADER, Telemetry, TraceContext,
};
use axum::Json;
use axum::Router;
use axum::extract::{Extension, Path, Request, State};
use axum::http::{
    HeaderMap, HeaderName, HeaderValue, StatusCode, header::AUTHORIZATION, header::CONTENT_TYPE,
};
use axum::response::{IntoResponse, Response};
use axum::routing::{delete, get, post};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::net::SocketAddr;
use std::sync::Arc;
use std::time::{Duration, Instant};
use sys1_core::{
    DecisionModel, DecisionOutput, Error as RegistryError, ManifestError, ModelError, ModelOrigin,
    ModelRegistry, ModelResolver, Prediction, Questions,
};
use sys1_protocol::{
    DANGER_ZONE_IMAGES_HEADER, ErrorResponse, MODELS_PATH, ModelMetadataList, REQUEST_ID_HEADER,
    SYSTEMONE_PATH, State as WireState, SystemOneRequest, SystemOneResponse, parse_request_slice,
};
use tokio::sync::{mpsc, oneshot};
use tower_http::cors::CorsLayer;
use tower_http::trace::TraceLayer;
use tracing::Instrument;
use uuid::Uuid;

struct BatchItem {
    model_name: String,
    model: Arc<dyn DecisionModel>,
    state: WireState,
    questions: Questions,
    response: oneshot::Sender<Result<DecisionOutput, ModelError>>,
}

#[derive(Clone)]
pub struct AppState {
    pub registry: Arc<ModelRegistry>,
    pub config: ServerConfig,
    resolver: Option<Arc<dyn ModelResolver>>,
    cache: Option<Arc<dyn CacheManager>>,
    /// Serializes catalog mutations so add/remove snapshots cannot interleave.
    admin_lock: Arc<tokio::sync::Mutex<()>>,
    metrics: Arc<Metrics>,
    telemetry: Arc<dyn Telemetry>,
    batch: Option<mpsc::Sender<BatchItem>>,
}

/// Optional runtime integrations for an embedded server.
#[derive(Clone, Default)]
pub struct ServerExtensions {
    /// Enables `/api/models` additions from external references.
    pub resolver: Option<Arc<dyn ModelResolver>>,
    /// Enables `/api/cache` disk-cache management.
    pub cache: Option<Arc<dyn CacheManager>>,
    /// Structured request/model/batch/cache events. Defaults to a no-op.
    pub telemetry: Option<Arc<dyn Telemetry>>,
}

pub fn router(registry: Arc<ModelRegistry>, config: ServerConfig) -> Router {
    router_with_resolver(registry, config, None)
}

/// Build a router with an optional runtime resolver. When present, the admin
/// API can add models from references (`hf:...`, `local:...`, HTTP manifests).
pub fn router_with_resolver(
    registry: Arc<ModelRegistry>,
    config: ServerConfig,
    resolver: Option<Arc<dyn ModelResolver>>,
) -> Router {
    router_with_extensions(
        registry,
        config,
        ServerExtensions {
            resolver,
            ..ServerExtensions::default()
        },
    )
}

/// Build a router with runtime model resolution, disk-cache management, and/or
/// custom telemetry.
pub fn router_with_extensions(
    registry: Arc<ModelRegistry>,
    config: ServerConfig,
    extensions: ServerExtensions,
) -> Router {
    let ServerExtensions {
        resolver,
        cache,
        telemetry,
    } = extensions;
    let telemetry = telemetry.unwrap_or_else(|| Arc::new(NoopTelemetry));
    let metrics = Arc::new(Metrics::new());
    let batch = config.batching.as_ref().map(|batch| {
        let (sender, receiver) = mpsc::channel(batch.max_batch.saturating_mul(4).max(16));
        tokio::spawn(run_batcher(
            receiver,
            batch.clone(),
            metrics.clone(),
            telemetry.clone(),
        ));
        sender
    });

    let mut app = Router::new()
        .route(SYSTEMONE_PATH, post(systemone))
        .route(MODELS_PATH, get(models))
        .route("/health", get(health));

    if config.enable_admin {
        app = app
            .route("/api/models", get(admin_models).post(admin_add_model))
            .route("/api/models/{name}/load", post(admin_load))
            .route("/api/models/{name}/unload", post(admin_unload))
            .route("/api/models/{name}/reload", post(admin_reload))
            .route("/api/models/{name}", delete(admin_remove));
        if cache.is_some() {
            app = app
                .route("/api/cache", get(admin_cache))
                .route("/api/cache/{key}", delete(admin_cache_remove));
        }
    }
    if config.enable_metrics {
        app = app
            .route("/metrics", get(metrics_prometheus))
            .route("/api/metrics", get(metrics_json));
    }

    app.layer(CorsLayer::permissive())
        .layer(TraceLayer::new_for_http())
        .layer(axum::middleware::from_fn(trace_context_layer))
        .with_state(AppState {
            registry,
            config,
            resolver,
            cache,
            admin_lock: Arc::new(tokio::sync::Mutex::new(())),
            metrics,
            telemetry,
            batch,
        })
}

async fn trace_context_layer(mut request: Request, next: axum::middleware::Next) -> Response {
    let context = TraceContext::from_headers(request.headers());
    request.extensions_mut().insert(context.clone());
    let mut response = next.run(request).await;
    if let Ok(value) = HeaderValue::from_str(&context.traceparent()) {
        response
            .headers_mut()
            .insert(HeaderName::from_static(TRACEPARENT_HEADER), value);
    }
    response
}

/// Yield-spin grace period used to discover a burst. This avoids the coarse
/// Windows timer granularity that a `sleep`/`timeout` would add to every
/// request, while still giving back-to-back requests a chance to batch.
const GATHER_GRACE: Duration = Duration::from_micros(500);

async fn run_batcher(
    mut receiver: mpsc::Receiver<BatchItem>,
    config: BatchConfig,
    metrics: Arc<Metrics>,
    telemetry: Arc<dyn Telemetry>,
) {
    while let Some(first) = receiver.recv().await {
        let mut items = Vec::with_capacity(config.max_batch.max(1));
        items.push(first);
        let mut last_item = Instant::now();

        // Even with a zero wait, batch whatever is already queued.
        while items.len() < config.max_batch {
            match receiver.try_recv() {
                Ok(item) => {
                    items.push(item);
                    last_item = Instant::now();
                }
                Err(_) => break,
            }
        }

        if !config.max_wait.is_zero() {
            let started = Instant::now();
            loop {
                while items.len() < config.max_batch {
                    match receiver.try_recv() {
                        Ok(item) => {
                            items.push(item);
                            last_item = Instant::now();
                        }
                        Err(_) => break,
                    }
                }

                if items.len() >= config.max_batch {
                    break;
                }

                let now = Instant::now();
                if now.duration_since(started) >= config.max_wait {
                    break;
                }
                if items.len() == 1 && now.duration_since(last_item) >= GATHER_GRACE {
                    // A lone request: pay only the short grace period.
                    break;
                }
                tokio::task::yield_now().await;
            }
        }

        let metrics = metrics.clone();
        let telemetry = telemetry.clone();
        let _ =
            tokio::task::spawn_blocking(move || process_batch(items, &metrics, telemetry.as_ref()))
                .await;
    }
}

fn process_batch(items: Vec<BatchItem>, metrics: &Metrics, telemetry: &dyn Telemetry) {
    let batch_size = items.len();
    let started = Instant::now();
    tracing::debug!(batch = batch_size, "batch start");
    let mut groups: HashMap<String, Vec<BatchItem>> = HashMap::new();
    for item in items {
        groups
            .entry(item.model_name.clone())
            .or_default()
            .push(item);
    }

    for (_, group) in groups {
        let predictions: Vec<Prediction<'_>> = group
            .iter()
            .map(|item| Prediction::new(&item.state, &item.questions))
            .collect();
        let model = group[0].model.clone();
        let outputs = model.predict_batch(&predictions);
        for (item, output) in group.into_iter().zip(outputs) {
            let _ = item.response.send(output);
        }
    }
    let elapsed = started.elapsed();
    metrics.record_batch(batch_size, elapsed);
    telemetry.on_batch(&BatchEvent {
        items: batch_size,
        elapsed,
    });
    tracing::debug!(
        batch = batch_size,
        elapsed_ms = elapsed.as_secs_f64() * 1000.0,
        "batch done"
    );
}

pub async fn serve(
    addr: SocketAddr,
    registry: Arc<ModelRegistry>,
    config: ServerConfig,
) -> std::io::Result<()> {
    serve_with_resolver(addr, registry, config, None).await
}

pub async fn serve_with_resolver(
    addr: SocketAddr,
    registry: Arc<ModelRegistry>,
    config: ServerConfig,
    resolver: Option<Arc<dyn ModelResolver>>,
) -> std::io::Result<()> {
    serve_with_extensions(
        addr,
        registry,
        config,
        ServerExtensions {
            resolver,
            ..ServerExtensions::default()
        },
    )
    .await
}

pub async fn serve_with_extensions(
    addr: SocketAddr,
    registry: Arc<ModelRegistry>,
    config: ServerConfig,
    extensions: ServerExtensions,
) -> std::io::Result<()> {
    let listener = tokio::net::TcpListener::bind(addr).await?;
    tracing::info!(%addr, "sys1 server listening");
    axum::serve(
        listener,
        router_with_extensions(registry, config, extensions),
    )
    .await
}

async fn systemone(State(state): State<AppState>, request: Request) -> Response {
    let trace = request
        .extensions()
        .get::<TraceContext>()
        .cloned()
        .unwrap_or_else(TraceContext::generate);
    let request_id = trace.request_id.clone();
    let span = tracing::info_span!(
        "systemone",
        request_id = %trace.request_id,
        trace_id = %trace.trace_id,
        model = tracing::field::Empty,
        status = tracing::field::Empty
    );
    let started = Instant::now();
    let metrics = state.metrics.clone();
    let telemetry = state.telemetry.clone();
    match handle_systemone(state, request, trace.clone())
        .instrument(span.clone())
        .await
    {
        Ok(response) => {
            let elapsed = started.elapsed();
            span.record("model", response.model.as_str());
            span.record("status", u64::from(StatusCode::OK.as_u16()));
            metrics.record_systemone(elapsed, true, Some(&response.model));
            telemetry.on_request(&RequestEvent {
                request_id: trace.request_id.clone(),
                trace_id: trace.trace_id.clone(),
                model: Some(response.model.clone()),
                status: StatusCode::OK.as_u16(),
                elapsed,
                error: None,
            });
            json_response(StatusCode::OK, response, &request_id)
        }
        Err(error) => {
            let elapsed = started.elapsed();
            span.record("status", u64::from(error.status.as_u16()));
            metrics.record_systemone(elapsed, false, None);
            telemetry.on_request(&RequestEvent {
                request_id: trace.request_id.clone(),
                trace_id: trace.trace_id.clone(),
                model: None,
                status: error.status.as_u16(),
                elapsed,
                error: Some(error_message(&error)),
            });
            json_response(error.status, error.body, &request_id)
        }
    }
}

async fn handle_systemone(
    state: AppState,
    request: Request,
    trace: TraceContext,
) -> Result<SystemOneResponse, ApiError> {
    let (headers, body) = read_body(request, state.config.max_body_bytes).await?;
    check_auth(&state.config, &headers)?;
    let request =
        parse_request_slice(&body).map_err(|report| ApiError::validation(report.into_errors()))?;
    validate_danger_zone(&request, &headers)?;
    let registry = state.registry.clone();
    registry.maintenance();
    let model = request.model;
    let wire_state = request.state;
    let questions = Questions::from(request.questions);
    questions
        .validate()
        .map_err(|report| ApiError::validation(report.into_errors()))?;

    if let Some(batcher) = &state.batch {
        let (canonical, model_handle) = if let Some(loaded) = registry.try_acquire(&model) {
            loaded
        } else {
            let requested = model.clone();
            let registry_for_acquire = registry.clone();
            let metrics = state.metrics.clone();
            let telemetry = state.telemetry.clone();
            let event_trace = trace.clone();
            tokio::task::spawn_blocking(move || {
                let (canonical, _) = registry_for_acquire.resolve(&requested).ok_or_else(|| {
                    RegistryError::ModelNotFound {
                        model: requested.clone(),
                    }
                })?;
                let started = Instant::now();
                let acquired = registry_for_acquire.acquire(&canonical);
                let elapsed = started.elapsed();
                metrics.record_model_load(elapsed, acquired.is_ok());
                telemetry.on_model(&ModelEvent {
                    request_id: event_trace.request_id.clone(),
                    trace_id: event_trace.trace_id.clone(),
                    model: canonical.clone(),
                    action: ModelAction::Load,
                    success: acquired.is_ok(),
                    elapsed,
                    error: acquired.as_ref().err().map(ToString::to_string),
                });
                let model_handle = acquired?;
                Ok::<_, RegistryError>((canonical, model_handle))
            })
            .await
            .map_err(|error| ApiError::internal(format!("model acquisition task failed: {error}")))?
            .map_err(map_registry_error)?
        };

        if !model_handle.supports_cross_request_batching() {
            let model_name = canonical.clone();
            let output =
                tokio::task::spawn_blocking(move || model_handle.predict(&wire_state, &questions))
                    .await
                    .map_err(|error| ApiError::internal(format!("inference task failed: {error}")))?
                    .map_err(|source| {
                        map_registry_error(RegistryError::Model {
                            model: canonical,
                            source,
                        })
                    })?;
            return Ok(SystemOneResponse {
                model: model_name,
                answers: output.answers.into_map(),
                usage: output.usage,
            });
        }

        let (sender, receiver) = oneshot::channel();
        batcher
            .send(BatchItem {
                model_name: canonical.clone(),
                model: model_handle,
                state: wire_state,
                questions,
                response: sender,
            })
            .await
            .map_err(|_| ApiError::internal("batcher is unavailable"))?;

        let output = receiver
            .await
            .map_err(|_| ApiError::internal("batch worker dropped the request"))?
            .map_err(|source| {
                map_registry_error(RegistryError::Model {
                    model: canonical.clone(),
                    source,
                })
            })?;

        return Ok(SystemOneResponse {
            model: canonical,
            answers: output.answers.into_map(),
            usage: output.usage,
        });
    }

    let metrics = state.metrics.clone();
    let telemetry = state.telemetry.clone();
    let event_trace = trace.clone();
    tokio::task::spawn_blocking(move || {
        if let Some((canonical, model_handle)) = registry.try_acquire(&model) {
            let output = model_handle
                .predict(&wire_state, &questions)
                .map_err(|source| RegistryError::Model {
                    model: canonical.clone(),
                    source,
                })?;
            return Ok(SystemOneResponse {
                model: canonical,
                answers: output.answers.into_map(),
                usage: output.usage,
            });
        }

        let (canonical, _) =
            registry
                .resolve(&model)
                .ok_or_else(|| RegistryError::ModelNotFound {
                    model: model.clone(),
                })?;
        let started = Instant::now();
        let acquired = registry.acquire(&canonical);
        let elapsed = started.elapsed();
        metrics.record_model_load(elapsed, acquired.is_ok());
        telemetry.on_model(&ModelEvent {
            request_id: event_trace.request_id.clone(),
            trace_id: event_trace.trace_id.clone(),
            model: canonical.clone(),
            action: ModelAction::Load,
            success: acquired.is_ok(),
            elapsed,
            error: acquired.as_ref().err().map(ToString::to_string),
        });
        let model_handle = acquired?;
        let output = model_handle
            .predict(&wire_state, &questions)
            .map_err(|source| RegistryError::Model {
                model: canonical.clone(),
                source,
            })?;
        Ok::<_, RegistryError>(SystemOneResponse {
            model: canonical,
            answers: output.answers.into_map(),
            usage: output.usage,
        })
    })
    .await
    .map_err(|error| ApiError::internal(format!("inference task failed: {error}")))?
    .map_err(map_registry_error)
}

async fn models(State(state): State<AppState>, headers: HeaderMap) -> Response {
    let request_id = Uuid::new_v4().to_string();
    match check_auth(&state.config, &headers) {
        Ok(()) => json_response(
            StatusCode::OK,
            ModelMetadataList::new(state.registry.list_models()),
            &request_id,
        ),
        Err(error) => json_response(error.status, error.body, &request_id),
    }
}

async fn health(State(state): State<AppState>) -> Response {
    #[derive(Serialize)]
    struct Health {
        status: &'static str,
        loaded_models: Vec<String>,
    }

    let request_id = Uuid::new_v4().to_string();
    json_response(
        StatusCode::OK,
        Health {
            status: "ok",
            loaded_models: state.registry.loaded_models(),
        },
        &request_id,
    )
}

async fn metrics_json(State(state): State<AppState>, headers: HeaderMap) -> Response {
    let request_id = Uuid::new_v4().to_string();
    if let Err(error) = check_auth(&state.config, &headers) {
        return json_response(error.status, error.body, &request_id);
    }
    let snapshot = state.metrics.snapshot(state.registry.loaded_models());
    json_response(StatusCode::OK, snapshot, &request_id)
}

async fn metrics_prometheus(State(state): State<AppState>, headers: HeaderMap) -> Response {
    let request_id = Uuid::new_v4().to_string();
    if let Err(error) = check_auth(&state.config, &headers) {
        return json_response(error.status, error.body, &request_id);
    }
    let body = state.metrics.prometheus(state.registry.loaded_models());
    text_response(
        StatusCode::OK,
        body,
        "text/plain; version=0.0.4; charset=utf-8",
        &request_id,
    )
}

#[derive(Serialize)]
struct AdminModelInfo {
    name: String,
    aliases: Vec<String>,
    description: String,
    release_date: String,
    family: String,
    backend: String,
    format: String,
    version: String,
    origin: &'static str,
    loaded: bool,
}

#[derive(Serialize)]
struct AdminLimits {
    max_loaded: usize,
    keep_alive_secs: u64,
}

#[derive(Serialize)]
struct AdminModelList {
    models: Vec<AdminModelInfo>,
    loaded: Vec<String>,
    limits: AdminLimits,
}

#[derive(Serialize)]
struct AdminModelActionResult {
    model: String,
    loaded: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    was_loaded: Option<bool>,
}

#[derive(Deserialize)]
struct AddModelRequest {
    reference: String,
    #[serde(default)]
    load: bool,
}

#[derive(Serialize)]
struct AddModelResult {
    models: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    loaded: Option<String>,
}

#[derive(Serialize)]
struct RemoveModelResult {
    model: String,
    removed: bool,
}

async fn admin_models(State(state): State<AppState>, headers: HeaderMap) -> Response {
    let request_id = Uuid::new_v4().to_string();
    if let Err(error) = check_auth(&state.config, &headers) {
        return json_response(error.status, error.body, &request_id);
    }

    let models = state
        .registry
        .manifests()
        .into_iter()
        .map(|manifest| AdminModelInfo {
            loaded: state.registry.is_loaded(&manifest.name),
            origin: match state.registry.origin(&manifest.name) {
                Some(ModelOrigin::Configured) => "configured",
                Some(ModelOrigin::Runtime) => "runtime",
                None => "unknown",
            },
            name: manifest.name,
            aliases: manifest.aliases,
            description: manifest.description,
            release_date: manifest.release_date,
            family: manifest.family,
            backend: manifest.backend,
            format: manifest.format,
            version: manifest.version,
        })
        .collect();
    let config = state.registry.config();

    json_response(
        StatusCode::OK,
        AdminModelList {
            models,
            loaded: state.registry.loaded_models(),
            limits: AdminLimits {
                max_loaded: config.max_loaded,
                keep_alive_secs: config.keep_alive.as_secs(),
            },
        },
        &request_id,
    )
}

async fn admin_load(
    State(state): State<AppState>,
    Path(name): Path<String>,
    headers: HeaderMap,
    Extension(trace): Extension<TraceContext>,
) -> Response {
    let request_id = trace.request_id.clone();
    if let Err(error) = check_auth(&state.config, &headers) {
        return json_response(error.status, error.body, &request_id);
    }
    let Some((canonical, _)) = state.registry.resolve(&name) else {
        let error = unknown_model(&name);
        return json_response(error.status, error.body, &request_id);
    };

    let registry = state.registry.clone();
    let metrics = state.metrics.clone();
    let telemetry = state.telemetry.clone();
    let event_trace = trace.clone();
    let load_name = canonical.clone();
    match tokio::task::spawn_blocking(move || {
        let started = Instant::now();
        let acquired = registry.acquire(&load_name);
        let elapsed = started.elapsed();
        metrics.record_model_load(elapsed, acquired.is_ok());
        telemetry.on_model(&ModelEvent {
            request_id: event_trace.request_id.clone(),
            trace_id: event_trace.trace_id.clone(),
            model: load_name,
            action: ModelAction::Load,
            success: acquired.is_ok(),
            elapsed,
            error: acquired.as_ref().err().map(ToString::to_string),
        });
        acquired
    })
    .await
    {
        Ok(Ok(_)) => json_response(
            StatusCode::OK,
            AdminModelActionResult {
                model: canonical,
                loaded: true,
                was_loaded: None,
            },
            &request_id,
        ),
        Ok(Err(error)) => {
            let error = map_registry_error(error);
            json_response(error.status, error.body, &request_id)
        }
        Err(error) => json_response(
            StatusCode::INTERNAL_SERVER_ERROR,
            ErrorResponse::bad_request(format!("model load task failed: {error}")),
            &request_id,
        ),
    }
}

async fn admin_unload(
    State(state): State<AppState>,
    Path(name): Path<String>,
    headers: HeaderMap,
    Extension(trace): Extension<TraceContext>,
) -> Response {
    let request_id = trace.request_id.clone();
    if let Err(error) = check_auth(&state.config, &headers) {
        return json_response(error.status, error.body, &request_id);
    }
    let Some((canonical, _)) = state.registry.resolve(&name) else {
        let error = unknown_model(&name);
        return json_response(error.status, error.body, &request_id);
    };

    let registry = state.registry.clone();
    let metrics = state.metrics.clone();
    let telemetry = state.telemetry.clone();
    let event_trace = trace.clone();
    let unload_name = canonical.clone();
    match tokio::task::spawn_blocking(move || {
        let started = Instant::now();
        let was_loaded = registry.unload(&unload_name);
        let elapsed = started.elapsed();
        if was_loaded {
            metrics.record_model_unload();
        }
        telemetry.on_model(&ModelEvent {
            request_id: event_trace.request_id.clone(),
            trace_id: event_trace.trace_id.clone(),
            model: unload_name,
            action: ModelAction::Unload,
            success: was_loaded,
            elapsed,
            error: if was_loaded {
                None
            } else {
                Some("model was not resident".to_owned())
            },
        });
        was_loaded
    })
    .await
    {
        Ok(was_loaded) => json_response(
            StatusCode::OK,
            AdminModelActionResult {
                model: canonical,
                loaded: false,
                was_loaded: Some(was_loaded),
            },
            &request_id,
        ),
        Err(error) => json_response(
            StatusCode::INTERNAL_SERVER_ERROR,
            ErrorResponse::bad_request(format!("model unload task failed: {error}")),
            &request_id,
        ),
    }
}

async fn admin_reload(
    State(state): State<AppState>,
    Path(name): Path<String>,
    headers: HeaderMap,
    Extension(trace): Extension<TraceContext>,
) -> Response {
    let request_id = trace.request_id.clone();
    if let Err(error) = check_auth(&state.config, &headers) {
        return json_response(error.status, error.body, &request_id);
    }
    let Some((canonical, _)) = state.registry.resolve(&name) else {
        let error = unknown_model(&name);
        return json_response(error.status, error.body, &request_id);
    };

    let registry = state.registry.clone();
    let metrics = state.metrics.clone();
    let telemetry = state.telemetry.clone();
    let event_trace = trace.clone();
    let reload_name = canonical.clone();
    let result = tokio::task::spawn_blocking(move || {
        let started = Instant::now();
        let was_loaded = registry.unload(&reload_name);
        if was_loaded {
            metrics.record_model_unload();
        }
        let acquired = registry.acquire(&reload_name);
        let elapsed = started.elapsed();
        metrics.record_model_load(elapsed, acquired.is_ok());
        telemetry.on_model(&ModelEvent {
            request_id: event_trace.request_id.clone(),
            trace_id: event_trace.trace_id.clone(),
            model: reload_name,
            action: ModelAction::Reload,
            success: acquired.is_ok(),
            elapsed,
            error: acquired.as_ref().err().map(ToString::to_string),
        });
        (was_loaded, acquired)
    })
    .await;

    match result {
        Ok((was_loaded, Ok(_))) => json_response(
            StatusCode::OK,
            AdminModelActionResult {
                model: canonical,
                loaded: true,
                was_loaded: Some(was_loaded),
            },
            &request_id,
        ),
        Ok((was_loaded, Err(error))) => {
            tracing::warn!(model = %canonical, was_loaded, "model reload failed");
            let error = map_registry_error(error);
            json_response(error.status, error.body, &request_id)
        }
        Err(error) => json_response(
            StatusCode::INTERNAL_SERVER_ERROR,
            ErrorResponse::bad_request(format!("model reload task failed: {error}")),
            &request_id,
        ),
    }
}

#[derive(Serialize)]
struct CacheList {
    cache_dir: String,
    models: Vec<CachedModel>,
}

#[derive(Serialize)]
struct CacheRemoveResult {
    key: String,
    removed: bool,
}

async fn admin_cache(
    State(state): State<AppState>,
    headers: HeaderMap,
    Extension(trace): Extension<TraceContext>,
) -> Response {
    let request_id = trace.request_id.clone();
    if let Err(error) = check_auth(&state.config, &headers) {
        return json_response(error.status, error.body, &request_id);
    }
    let Some(cache) = state.cache.clone() else {
        let error = ApiError::not_implemented("disk cache management is not configured");
        return json_response(error.status, error.body, &request_id);
    };
    let telemetry = state.telemetry.clone();
    let event_trace = trace.clone();

    match tokio::task::spawn_blocking(move || {
        let cache_dir = cache.root();
        let result = cache.list();
        telemetry.on_cache(&CacheEvent {
            request_id: event_trace.request_id.clone(),
            trace_id: event_trace.trace_id.clone(),
            key: None,
            action: CacheAction::List,
            success: result.is_ok(),
            error: result.as_ref().err().cloned(),
        });
        (cache_dir, result)
    })
    .await
    {
        Ok((cache_dir, Ok(models))) => {
            json_response(StatusCode::OK, CacheList { cache_dir, models }, &request_id)
        }
        Ok((_, Err(error))) => json_response(
            StatusCode::INTERNAL_SERVER_ERROR,
            ErrorResponse::bad_request(error),
            &request_id,
        ),
        Err(error) => json_response(
            StatusCode::INTERNAL_SERVER_ERROR,
            ErrorResponse::bad_request(format!("cache list task failed: {error}")),
            &request_id,
        ),
    }
}

async fn admin_cache_remove(
    State(state): State<AppState>,
    Path(key): Path<String>,
    headers: HeaderMap,
    Extension(trace): Extension<TraceContext>,
) -> Response {
    let request_id = trace.request_id.clone();
    if let Err(error) = check_auth(&state.config, &headers) {
        return json_response(error.status, error.body, &request_id);
    }
    if !valid_cache_key(&key) {
        return json_response(
            StatusCode::BAD_REQUEST,
            ErrorResponse::bad_request("invalid cache key"),
            &request_id,
        );
    }
    let Some(cache) = state.cache.clone() else {
        let error = ApiError::not_implemented("disk cache management is not configured");
        return json_response(error.status, error.body, &request_id);
    };

    let admin_lock = state.admin_lock.clone();
    let _admin_guard = admin_lock.lock().await;
    let registry = state.registry.clone();
    let telemetry = state.telemetry.clone();
    let event_trace = trace.clone();
    let remove_key = key.clone();
    let event_key = key.clone();
    let result = tokio::task::spawn_blocking(move || {
        let models = match cache.list() {
            Ok(models) => models,
            Err(error) => {
                telemetry.on_cache(&CacheEvent {
                    request_id: event_trace.request_id.clone(),
                    trace_id: event_trace.trace_id.clone(),
                    key: Some(event_key.clone()),
                    action: CacheAction::Remove,
                    success: false,
                    error: Some(error.clone()),
                });
                return Err(error);
            }
        };
        if let Some(entry) = models.iter().find(|entry| entry.key == remove_key)
            && registry.is_loaded(&entry.name)
        {
            let message = format!(
                "model `{}` is loaded; unload it before removing the cache",
                entry.name
            );
            telemetry.on_cache(&CacheEvent {
                request_id: event_trace.request_id.clone(),
                trace_id: event_trace.trace_id.clone(),
                key: Some(event_key),
                action: CacheAction::Remove,
                success: false,
                error: Some(message),
            });
            return Ok::<_, String>((false, Some(entry.name.clone())));
        }
        match cache.remove(&remove_key) {
            Ok(removed) => {
                telemetry.on_cache(&CacheEvent {
                    request_id: event_trace.request_id.clone(),
                    trace_id: event_trace.trace_id.clone(),
                    key: Some(event_key),
                    action: CacheAction::Remove,
                    success: true,
                    error: None,
                });
                Ok((removed, None))
            }
            Err(error) => {
                telemetry.on_cache(&CacheEvent {
                    request_id: event_trace.request_id.clone(),
                    trace_id: event_trace.trace_id.clone(),
                    key: Some(event_key),
                    action: CacheAction::Remove,
                    success: false,
                    error: Some(error.clone()),
                });
                Err(error)
            }
        }
    })
    .await;

    match result {
        Ok(Ok((_, Some(model)))) => {
            let error = ApiError::conflict(format!(
                "model `{model}` is loaded; unload it before removing the cache"
            ));
            json_response(error.status, error.body, &request_id)
        }
        Ok(Ok((removed, None))) => json_response(
            StatusCode::OK,
            CacheRemoveResult { key, removed },
            &request_id,
        ),
        Ok(Err(error)) => json_response(
            StatusCode::INTERNAL_SERVER_ERROR,
            ErrorResponse::bad_request(error),
            &request_id,
        ),
        Err(error) => json_response(
            StatusCode::INTERNAL_SERVER_ERROR,
            ErrorResponse::bad_request(format!("cache removal task failed: {error}")),
            &request_id,
        ),
    }
}

fn valid_cache_key(key: &str) -> bool {
    !key.is_empty()
        && key != "."
        && key != ".."
        && key
            .chars()
            .all(|ch| ch.is_ascii_alphanumeric() || matches!(ch, '-' | '_' | '.'))
}

async fn admin_add_model(State(state): State<AppState>, request: Request) -> Response {
    let trace = request
        .extensions()
        .get::<TraceContext>()
        .cloned()
        .unwrap_or_else(TraceContext::generate);
    let request_id = trace.request_id.clone();
    if let Err(error) = check_auth(&state.config, request.headers()) {
        return json_response(error.status, error.body, &request_id);
    }
    let Some(resolver) = state.resolver.clone() else {
        let error = ApiError::not_implemented(
            "dynamic model registration is not configured on this server",
        );
        return json_response(error.status, error.body, &request_id);
    };

    let (_, body) = match read_body(request, 64 * 1024).await {
        Ok(body) => body,
        Err(error) => return json_response(error.status, error.body, &request_id),
    };
    let payload: AddModelRequest = match serde_json::from_slice(&body) {
        Ok(payload) => payload,
        Err(error) => {
            return json_response(
                StatusCode::BAD_REQUEST,
                ErrorResponse::bad_request(format!("invalid JSON body: {error}")),
                &request_id,
            );
        }
    };
    let reference = payload.reference.trim().to_owned();
    if reference.is_empty() {
        return json_response(
            StatusCode::BAD_REQUEST,
            ErrorResponse::bad_request("`reference` must not be empty"),
            &request_id,
        );
    }

    let admin_lock = state.admin_lock.clone();
    let _admin_guard = admin_lock.lock().await;

    let resolve_reference = reference.clone();
    let resolver_for_task = resolver.clone();
    let manifests =
        match tokio::task::spawn_blocking(move || resolver_for_task.resolve(&resolve_reference))
            .await
        {
            Ok(Ok(manifests)) => manifests,
            Ok(Err(error)) => {
                let error = map_resolver_error(error);
                return json_response(error.status, error.body, &request_id);
            }
            Err(error) => {
                return json_response(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    ErrorResponse::bad_request(format!("model resolution task failed: {error}")),
                    &request_id,
                );
            }
        };
    if manifests.is_empty() {
        return json_response(
            StatusCode::INTERNAL_SERVER_ERROR,
            ErrorResponse::bad_request(format!("`{reference}` resolved to no models")),
            &request_id,
        );
    }

    // Keep every resolved name so a partial registration can be rolled back,
    // including factory registrations that never reached the catalog. Names
    // that were already cataloged before this request are left untouched.
    let resolved: Vec<String> = manifests
        .iter()
        .map(|manifest| manifest.name.clone())
        .collect();
    let pre_existing: HashSet<String> = state
        .registry
        .manifests()
        .into_iter()
        .map(|manifest| manifest.name)
        .collect();
    let registration_started = Instant::now();
    let mut added: Vec<String> = Vec::with_capacity(manifests.len());
    for manifest in manifests {
        let name = manifest.name.clone();
        if let Err(error) = state.registry.register(manifest) {
            let error = map_manifest_error(error);
            state.telemetry.on_model(&ModelEvent {
                request_id: trace.request_id.clone(),
                trace_id: trace.trace_id.clone(),
                model: name,
                action: ModelAction::Register,
                success: false,
                elapsed: registration_started.elapsed(),
                error: Some(error_message(&error)),
            });
            for resolved_name in &resolved {
                if !pre_existing.contains(resolved_name) {
                    state.registry.unregister(resolved_name);
                    let _ = resolver.release(resolved_name);
                }
            }
            return json_response(error.status, error.body, &request_id);
        }
        added.push(name);
    }
    let registration_elapsed = registration_started.elapsed();
    for name in &added {
        state.telemetry.on_model(&ModelEvent {
            request_id: trace.request_id.clone(),
            trace_id: trace.trace_id.clone(),
            model: name.clone(),
            action: ModelAction::Register,
            success: true,
            elapsed: registration_elapsed,
            error: None,
        });
    }

    let loaded = if payload.load {
        // The primary manifest is last: a bundle registers its variants first
        // and the router (or single checkpoint) last.
        let primary = added
            .last()
            .cloned()
            .expect("a successful registration added at least one model");
        let registry = state.registry.clone();
        let metrics = state.metrics.clone();
        let telemetry = state.telemetry.clone();
        let event_trace = trace.clone();
        let load_name = primary.clone();
        let result = tokio::task::spawn_blocking(move || {
            let started = Instant::now();
            let acquired = registry.acquire(&load_name);
            let elapsed = started.elapsed();
            metrics.record_model_load(elapsed, acquired.is_ok());
            telemetry.on_model(&ModelEvent {
                request_id: event_trace.request_id.clone(),
                trace_id: event_trace.trace_id.clone(),
                model: load_name,
                action: ModelAction::Load,
                success: acquired.is_ok(),
                elapsed,
                error: acquired.as_ref().err().map(ToString::to_string),
            });
            acquired
        })
        .await;

        match result {
            Ok(Ok(_)) => Some(primary),
            Ok(Err(error)) => {
                for name in &added {
                    state.registry.unregister(name);
                    let _ = resolver.release(name);
                }
                let error = map_registry_error(error);
                return json_response(error.status, error.body, &request_id);
            }
            Err(error) => {
                for name in &added {
                    state.registry.unregister(name);
                    let _ = resolver.release(name);
                }
                return json_response(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    ErrorResponse::bad_request(format!("model load task failed: {error}")),
                    &request_id,
                );
            }
        }
    } else {
        None
    };

    tracing::info!(
        reference = %reference,
        models = ?added,
        loaded = ?loaded,
        "registered models at runtime"
    );
    json_response(
        StatusCode::CREATED,
        AddModelResult {
            models: added,
            loaded,
        },
        &request_id,
    )
}

async fn admin_remove(
    State(state): State<AppState>,
    Path(name): Path<String>,
    headers: HeaderMap,
    Extension(trace): Extension<TraceContext>,
) -> Response {
    let request_id = trace.request_id.clone();
    if let Err(error) = check_auth(&state.config, &headers) {
        return json_response(error.status, error.body, &request_id);
    }
    let admin_lock = state.admin_lock.clone();
    let _admin_guard = admin_lock.lock().await;
    let Some((canonical, _)) = state.registry.resolve(&name) else {
        let error = unknown_model(&name);
        return json_response(error.status, error.body, &request_id);
    };

    let registry = state.registry.clone();
    let resolver = state.resolver.clone();
    let metrics = state.metrics.clone();
    let telemetry = state.telemetry.clone();
    let event_trace = trace.clone();
    let remove_name = canonical.clone();
    match tokio::task::spawn_blocking(move || {
        let started = Instant::now();
        let was_loaded = registry.is_loaded(&remove_name);
        let removed = registry.unregister(&remove_name);
        if removed && was_loaded {
            metrics.record_model_unload();
        }
        if let Some(resolver) = resolver {
            let _ = resolver.release(&remove_name);
        }
        telemetry.on_model(&ModelEvent {
            request_id: event_trace.request_id.clone(),
            trace_id: event_trace.trace_id.clone(),
            model: remove_name,
            action: ModelAction::Remove,
            success: removed,
            elapsed: started.elapsed(),
            error: (!removed).then(|| "model was not in the catalog".to_owned()),
        });
        removed
    })
    .await
    {
        Ok(removed) => json_response(
            StatusCode::OK,
            RemoveModelResult {
                model: canonical,
                removed,
            },
            &request_id,
        ),
        Err(error) => json_response(
            StatusCode::INTERNAL_SERVER_ERROR,
            ErrorResponse::bad_request(format!("model removal task failed: {error}")),
            &request_id,
        ),
    }
}

fn unknown_model(name: &str) -> ApiError {
    ApiError::not_found(format!("unknown model: {name}"))
}

fn error_message(error: &ApiError) -> String {
    serde_json::to_string(&error.body).unwrap_or_else(|_| "request failed".to_owned())
}

/// Resolution failures are client-visible (bad reference, missing checkpoint,
/// unsupported format), not server faults.
fn map_resolver_error(error: ModelError) -> ApiError {
    match error {
        ModelError::Load { message } | ModelError::Unsupported { message } => {
            ApiError::bad_request(message)
        }
        other => map_model_error(other),
    }
}

fn map_manifest_error(error: ManifestError) -> ApiError {
    match error {
        ManifestError::DuplicateName(_)
        | ManifestError::DuplicateAlias(_)
        | ManifestError::AliasConflict(_) => ApiError::conflict(error.to_string()),
        other => ApiError::bad_request(other.to_string()),
    }
}

fn check_auth(config: &ServerConfig, headers: &HeaderMap) -> Result<(), ApiError> {
    let Some(expected) = config.api_key.as_deref() else {
        return Ok(());
    };
    let Some(scheme_and_token) = headers
        .get(AUTHORIZATION)
        .and_then(|value| value.to_str().ok())
    else {
        return Err(ApiError::unauthorized());
    };
    let Some((scheme, token)) = scheme_and_token.split_once(' ') else {
        return Err(ApiError::unauthorized());
    };
    if !scheme.eq_ignore_ascii_case("Bearer") || token.trim() != expected {
        return Err(ApiError::unauthorized());
    }
    Ok(())
}

fn validate_danger_zone(request: &SystemOneRequest, headers: &HeaderMap) -> Result<(), ApiError> {
    let needs_images = request.assets.is_some()
        || request
            .questions
            .values()
            .any(|question| matches!(question, sys1_protocol::Question::Box(_)));
    if !needs_images {
        return Ok(());
    }
    let enabled = headers
        .get(DANGER_ZONE_IMAGES_HEADER)
        .and_then(|value| value.to_str().ok())
        .is_some_and(|value| value.eq_ignore_ascii_case("true"));
    if enabled {
        Ok(())
    } else {
        Err(ApiError::bad_request(
            "TypeSafe-Danger-Zone-Images: true is required for box questions and image assets",
        ))
    }
}

fn json_response<T: Serialize>(status: StatusCode, body: T, request_id: &str) -> Response {
    let mut response = (status, Json(body)).into_response();
    if let Ok(value) = HeaderValue::from_str(request_id) {
        response
            .headers_mut()
            .insert(HeaderName::from_static(REQUEST_ID_HEADER), value);
    }
    response
}

fn text_response(
    status: StatusCode,
    body: String,
    content_type: &'static str,
    request_id: &str,
) -> Response {
    let mut response = (status, body).into_response();
    response
        .headers_mut()
        .insert(CONTENT_TYPE, HeaderValue::from_static(content_type));
    if let Ok(value) = HeaderValue::from_str(request_id) {
        response
            .headers_mut()
            .insert(HeaderName::from_static(REQUEST_ID_HEADER), value);
    }
    response
}
