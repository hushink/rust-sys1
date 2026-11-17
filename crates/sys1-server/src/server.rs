use crate::body::read_body;
use crate::config::{BatchConfig, ServerConfig};
use crate::error::{ApiError, map_registry_error};
use axum::Json;
use axum::Router;
use axum::extract::{Request, State};
use axum::http::{HeaderMap, HeaderName, HeaderValue, StatusCode, header::AUTHORIZATION};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use serde::Serialize;
use std::collections::HashMap;
use std::net::SocketAddr;
use std::sync::Arc;
use std::time::{Duration, Instant};
use sys1_core::{
    DecisionModel, DecisionOutput, Error as RegistryError, ModelError, ModelRegistry, Prediction,
    Questions,
};
use sys1_protocol::{
    DANGER_ZONE_IMAGES_HEADER, MODELS_PATH, ModelMetadataList, REQUEST_ID_HEADER, SYSTEMONE_PATH,
    State as WireState, SystemOneRequest, SystemOneResponse, parse_request_slice,
};
use tokio::sync::{mpsc, oneshot};
use tower_http::cors::CorsLayer;
use tower_http::trace::TraceLayer;
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
    batch: Option<mpsc::Sender<BatchItem>>,
}

pub fn router(registry: Arc<ModelRegistry>, config: ServerConfig) -> Router {
    let batch = config.batching.as_ref().map(|batch| {
        let (sender, receiver) = mpsc::channel(batch.max_batch.saturating_mul(4).max(16));
        tokio::spawn(run_batcher(receiver, batch.clone()));
        sender
    });

    Router::new()
        .route(SYSTEMONE_PATH, post(systemone))
        .route(MODELS_PATH, get(models))
        .route("/health", get(health))
        .layer(CorsLayer::permissive())
        .layer(TraceLayer::new_for_http())
        .with_state(AppState {
            registry,
            config,
            batch,
        })
}

/// Yield-spin grace period used to discover a burst. This avoids the coarse
/// Windows timer granularity that a `sleep`/`timeout` would add to every
/// request, while still giving back-to-back requests a chance to batch.
const GATHER_GRACE: Duration = Duration::from_micros(500);

async fn run_batcher(mut receiver: mpsc::Receiver<BatchItem>, config: BatchConfig) {
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

        let _ = tokio::task::spawn_blocking(move || process_batch(items)).await;
    }
}

fn process_batch(items: Vec<BatchItem>) {
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
    tracing::debug!(
        batch = batch_size,
        elapsed_ms = started.elapsed().as_secs_f64() * 1000.0,
        "batch done"
    );
}

pub async fn serve(
    addr: SocketAddr,
    registry: Arc<ModelRegistry>,
    config: ServerConfig,
) -> std::io::Result<()> {
    let listener = tokio::net::TcpListener::bind(addr).await?;
    tracing::info!(%addr, "sys1 server listening");
    axum::serve(listener, router(registry, config)).await
}

async fn systemone(State(state): State<AppState>, request: Request) -> Response {
    let request_id = Uuid::new_v4().to_string();
    match handle_systemone(state, request).await {
        Ok(response) => json_response(StatusCode::OK, response, &request_id),
        Err(error) => json_response(error.status, error.body, &request_id),
    }
}

async fn handle_systemone(
    state: AppState,
    request: Request,
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
            tokio::task::spawn_blocking(move || {
                let (canonical, _) = registry_for_acquire.resolve(&requested).ok_or_else(|| {
                    RegistryError::ModelNotFound {
                        model: requested.clone(),
                    }
                })?;
                let model_handle = registry_for_acquire.acquire(canonical)?;
                Ok::<_, RegistryError>((canonical.to_owned(), model_handle))
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

    tokio::task::spawn_blocking(move || registry.predict(&model, &wire_state, &questions))
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
