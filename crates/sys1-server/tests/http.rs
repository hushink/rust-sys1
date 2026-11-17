use axum::Router;
use axum::body::{Body, Bytes};
use axum::http::{HeaderMap, Request, StatusCode, header};
use flate2::Compression;
use flate2::write::GzEncoder;
use indexmap::IndexMap;
use std::io::Write;
use std::sync::{Arc, Mutex};
use std::time::Duration;
use sys1_core::{
    Answers, DecisionModel, DecisionOutput, ModelError, ModelFactory, ModelFile, ModelManifest,
    ModelRegistry, ModelResolver, Prediction, Questions, RegistryConfig,
};
use sys1_protocol::{
    Answer, ChoiceAnswer, ModelMetadataList, NoulAnswer, Question, ScoreAnswer, State,
    SystemOneResponse, Usage,
};
use sys1_server::{
    BatchConfig, BatchEvent, CacheEvent, CacheManager, CacheRemoveError, CacheUsage, CachedFile,
    CachedModel, ConfigEvent, ConfigReload, ConfigReloader, ModelAction, ModelEvent, ReloadError,
    RequestEvent, ServerConfig, ServerExtensions, Telemetry, router, router_with_extensions,
    router_with_resolver,
};
use tower::ServiceExt;

struct FakeModel;

impl DecisionModel for FakeModel {
    fn predict(&self, _state: &State, questions: &Questions) -> Result<DecisionOutput, ModelError> {
        let mut answers = IndexMap::new();
        for (id, question) in questions.iter() {
            let answer = match question {
                Question::Noul(_) => Answer::Noul(NoulAnswer::new(0.75)),
                Question::Choice(choice) => Answer::Choice(ChoiceAnswer::new(
                    choice
                        .criteria
                        .keys()
                        .next()
                        .cloned()
                        .unwrap_or_else(|| "first".to_owned()),
                    0.6,
                    IndexMap::new(),
                )),
                Question::Score(_) => {
                    Answer::Score(ScoreAnswer::new(0.5, 0.6, IndexMap::new(), IndexMap::new()))
                }
                Question::Box(_) => {
                    return Err(ModelError::unsupported(
                        "this model does not support box questions",
                    ));
                }
            };
            answers.insert(id.clone(), answer);
        }
        Ok(DecisionOutput::new(Answers::new(answers), Usage::new(7, 0)))
    }
}

struct FakeFactory;

impl ModelFactory for FakeFactory {
    fn load(&self, _manifest: &ModelManifest) -> Result<Arc<dyn DecisionModel>, ModelError> {
        Ok(Arc::new(FakeModel))
    }
}

struct TrackingModel {
    batches: Arc<Mutex<Vec<usize>>>,
}

impl DecisionModel for TrackingModel {
    fn predict(&self, _state: &State, questions: &Questions) -> Result<DecisionOutput, ModelError> {
        Ok(output_for(questions))
    }

    fn predict_batch(
        &self,
        requests: &[Prediction<'_>],
    ) -> Vec<Result<DecisionOutput, ModelError>> {
        self.batches.lock().expect("batch log").push(requests.len());
        requests
            .iter()
            .map(|request| Ok(output_for(request.questions)))
            .collect()
    }
}

struct TrackingFactory {
    model: Arc<dyn DecisionModel>,
}

impl ModelFactory for TrackingFactory {
    fn load(&self, _manifest: &ModelManifest) -> Result<Arc<dyn DecisionModel>, ModelError> {
        Ok(self.model.clone())
    }
}

struct NoBatchModel;

impl DecisionModel for NoBatchModel {
    fn predict(&self, _state: &State, questions: &Questions) -> Result<DecisionOutput, ModelError> {
        Ok(output_for(questions))
    }

    fn supports_cross_request_batching(&self) -> bool {
        false
    }

    fn predict_batch(
        &self,
        _requests: &[Prediction<'_>],
    ) -> Vec<Result<DecisionOutput, ModelError>> {
        panic!("batcher must not be used for this model");
    }
}

struct NoBatchFactory;

impl ModelFactory for NoBatchFactory {
    fn load(&self, _manifest: &ModelManifest) -> Result<Arc<dyn DecisionModel>, ModelError> {
        Ok(Arc::new(NoBatchModel))
    }
}

struct FailingFactory;

impl ModelFactory for FailingFactory {
    fn load(&self, _manifest: &ModelManifest) -> Result<Arc<dyn DecisionModel>, ModelError> {
        Err(ModelError::load("planned load failure"))
    }
}

fn output_for(questions: &Questions) -> DecisionOutput {
    let answers = questions
        .iter()
        .map(|(id, _)| (id.clone(), Answer::Noul(NoulAnswer::new(0.5))))
        .collect();
    DecisionOutput::new(Answers::new(answers), Usage::new(1, 0))
}

fn manifest(name: &str, aliases: &[&str]) -> ModelManifest {
    ModelManifest {
        schema_version: 1,
        name: name.to_owned(),
        description: "test model".to_owned(),
        release_date: "2026-01-01".to_owned(),
        family: "test".to_owned(),
        variant: None,
        backend: "fake".to_owned(),
        format: "test".to_owned(),
        version: "1.0.0".to_owned(),
        aliases: aliases.iter().map(|alias| (*alias).to_owned()).collect(),
        files: vec![ModelFile {
            path: "model.bin".to_owned(),
            url: String::new(),
            sha256: Some("0".repeat(64)),
            git_sha1: None,
            size: 1,
        }],
    }
}

fn app(config: ServerConfig) -> Router {
    app_with_resolver(config, None)
}

fn app_with_resolver(config: ServerConfig, resolver: Option<Arc<dyn ModelResolver>>) -> Router {
    let registry = ModelRegistry::new(
        vec![manifest("laya-test", &["laya-latest"])],
        Arc::new(FakeFactory),
        RegistryConfig::default(),
    )
    .expect("registry");
    router_with_resolver(Arc::new(registry), config, resolver)
}

fn app_with_extensions(config: ServerConfig, extensions: ServerExtensions) -> Router {
    let registry = ModelRegistry::new(
        vec![manifest("laya-test", &["laya-latest"])],
        Arc::new(FakeFactory),
        RegistryConfig::default(),
    )
    .expect("registry");
    router_with_extensions(Arc::new(registry), config, extensions)
}

struct FakeCache {
    models: Mutex<Vec<CachedModel>>,
}

impl CacheManager for FakeCache {
    fn root(&self) -> String {
        "fake-cache".to_owned()
    }

    fn list(&self) -> Result<Vec<CachedModel>, String> {
        Ok(self
            .models
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .clone())
    }

    fn usage(&self) -> Result<CacheUsage, String> {
        let models = self.list()?;
        let used_bytes = models.iter().map(|model| model.size_bytes).sum();
        let reclaimable_bytes = models
            .iter()
            .filter(|model| model.evictable)
            .map(|model| model.size_bytes)
            .sum();
        Ok(CacheUsage {
            limit_bytes: Some(100),
            used_bytes,
            free_bytes: Some(100_u64.saturating_sub(used_bytes)),
            reclaimable_bytes,
            model_count: models.len(),
            resident_count: models.iter().filter(|model| !model.evictable).count(),
        })
    }

    fn remove(&self, key: &str) -> Result<bool, CacheRemoveError> {
        if key == "busy-key" {
            return Err(CacheRemoveError::conflict("cache entry is busy"));
        }
        let mut models = self
            .models
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let before = models.len();
        models.retain(|model| model.key != key);
        Ok(models.len() != before)
    }
}

#[derive(Default)]
struct RecordingTelemetry {
    requests: Mutex<Vec<RequestEvent>>,
    models: Mutex<Vec<ModelEvent>>,
    batches: Mutex<Vec<BatchEvent>>,
    caches: Mutex<Vec<CacheEvent>>,
    configs: Mutex<Vec<ConfigEvent>>,
}

impl Telemetry for RecordingTelemetry {
    fn on_request(&self, event: &RequestEvent) {
        self.requests
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .push(event.clone());
    }

    fn on_model(&self, event: &ModelEvent) {
        self.models
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .push(event.clone());
    }

    fn on_batch(&self, event: &BatchEvent) {
        self.batches
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .push(event.clone());
    }

    fn on_cache(&self, event: &CacheEvent) {
        self.caches
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .push(event.clone());
    }

    fn on_config(&self, event: &ConfigEvent) {
        self.configs
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .push(event.clone());
    }
}

struct OkReloader;

impl ConfigReloader for OkReloader {
    fn reload(&self) -> Result<ConfigReload, ReloadError> {
        Ok(ConfigReload {
            config_path: Some("fake.toml".to_owned()),
            added: vec!["added-model".to_owned()],
            removed: Vec::new(),
            unchanged: vec!["laya-test".to_owned()],
        })
    }
}

struct RestartReloader;

impl ConfigReloader for RestartReloader {
    fn reload(&self) -> Result<ConfigReload, ReloadError> {
        Err(ReloadError::RestartRequired(vec![
            "server settings changed".to_owned(),
        ]))
    }
}

struct FailedReloader;

impl ConfigReloader for FailedReloader {
    fn reload(&self) -> Result<ConfigReload, ReloadError> {
        Err(ReloadError::Failed("bad config".to_owned()))
    }
}

struct FakeResolver;

impl ModelResolver for FakeResolver {
    fn resolve(&self, reference: &str) -> Result<Vec<ModelManifest>, ModelError> {
        if reference == "test:dynamic" {
            let mut manifest = manifest("dynamic-test", &["dynamic-alias"]);
            manifest.files[0].url = "https://example.invalid/model.bin".to_owned();
            Ok(vec![manifest])
        } else {
            Err(ModelError::load(format!("unknown reference: {reference}")))
        }
    }

    fn release(&self, _name: &str) -> bool {
        true
    }
}

fn get(uri: &str) -> Request<Body> {
    Request::builder()
        .method("GET")
        .uri(uri)
        .body(Body::empty())
        .expect("request")
}

fn delete(uri: &str) -> Request<Body> {
    Request::builder()
        .method("DELETE")
        .uri(uri)
        .body(Body::empty())
        .expect("request")
}

fn loaded_names(body: &Bytes) -> Vec<String> {
    let value: serde_json::Value = serde_json::from_slice(body).expect("JSON");
    value["loaded"]
        .as_array()
        .expect("loaded array")
        .iter()
        .map(|name| name.as_str().expect("name").to_owned())
        .collect()
}

async fn send(app: Router, request: Request<Body>) -> (StatusCode, HeaderMap, Bytes) {
    let response = app.oneshot(request).await.expect("response");
    let status = response.status();
    let headers = response.headers().clone();
    let body = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .expect("body");
    (status, headers, body)
}

fn post(uri: &str, body: impl Into<Body>) -> Request<Body> {
    Request::builder()
        .method("POST")
        .uri(uri)
        .header(header::CONTENT_TYPE, "application/json")
        .body(body.into())
        .expect("request")
}

fn noul_body() -> String {
    serde_json::json!({
        "model": "laya-latest",
        "state": "hello",
        "questions": {
            "billing": {
                "type": "noul",
                "instructions": "Is this about billing?"
            }
        }
    })
    .to_string()
}

#[tokio::test]
async fn systemone_noul_roundtrip() {
    let (status, headers, body) = send(
        app(ServerConfig::default()),
        post("/v1/systemone", noul_body()),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert!(headers.contains_key("x-typesafe-request-id"));

    let response: SystemOneResponse = serde_json::from_slice(&body).expect("response JSON");
    assert_eq!(response.model, "laya-test");
    assert_eq!(response.usage.input_tokens, 7);
    let Answer::Noul(answer) = &response.answers["billing"] else {
        panic!("expected noul answer");
    };
    assert_eq!(answer.noul, 0.75);
}

#[tokio::test]
async fn models_lists_aliases_and_requires_auth() {
    let config = ServerConfig {
        api_key: Some("secret".to_owned()),
        ..ServerConfig::default()
    };
    let app = app(config);

    let request = Request::builder()
        .method("GET")
        .uri("/v1/models")
        .body(Body::empty())
        .expect("request");
    let (status, _, _) = send(app.clone(), request).await;
    assert_eq!(status, StatusCode::UNAUTHORIZED);

    let request = Request::builder()
        .method("GET")
        .uri("/v1/models")
        .header(header::AUTHORIZATION, "Bearer secret")
        .body(Body::empty())
        .expect("request");
    let (status, _, body) = send(app, request).await;
    assert_eq!(status, StatusCode::OK);
    let models: ModelMetadataList = serde_json::from_slice(&body).expect("models JSON");
    let names = models
        .models
        .into_iter()
        .map(|model| model.name)
        .collect::<Vec<_>>();
    assert!(names.contains(&"laya-test".to_owned()));
    assert!(names.contains(&"laya-latest".to_owned()));
}

#[tokio::test]
async fn validation_error_uses_detail_array() {
    let body = serde_json::json!({
        "model": "laya-latest",
        "state": "hello",
        "questions": {}
    })
    .to_string();
    let (status, _, body) = send(app(ServerConfig::default()), post("/v1/systemone", body)).await;
    assert_eq!(status, StatusCode::UNPROCESSABLE_ENTITY);
    let value: serde_json::Value = serde_json::from_slice(&body).expect("JSON");
    assert_eq!(value["detail"][0]["type"], "too_short");
}

#[tokio::test]
async fn unknown_model_is_bad_request() {
    let body = serde_json::json!({
        "model": "missing",
        "state": "hello",
        "questions": {
            "billing": { "type": "noul", "instructions": "x" }
        }
    })
    .to_string();
    let (status, _, body) = send(app(ServerConfig::default()), post("/v1/systemone", body)).await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    let value: serde_json::Value = serde_json::from_slice(&body).expect("JSON");
    assert!(value["detail"].as_str().unwrap().contains("unknown model"));
}

#[tokio::test]
async fn box_requires_danger_zone_header() {
    let body = serde_json::json!({
        "model": "laya-latest",
        "state": "photo",
        "questions": {
            "red": { "type": "box", "instructions": "Locate the red rectangle." }
        }
    })
    .to_string();

    let (status, _, first_response) = send(
        app(ServerConfig::default()),
        post("/v1/systemone", body.clone()),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    let value: serde_json::Value = serde_json::from_slice(&first_response).expect("JSON");
    assert!(value["detail"].as_str().unwrap().contains("Danger-Zone"));

    let request = Request::builder()
        .method("POST")
        .uri("/v1/systemone")
        .header(header::CONTENT_TYPE, "application/json")
        .header("TypeSafe-Danger-Zone-Images", "true")
        .body(Body::from(body))
        .expect("request");
    let (status, _, body) = send(app(ServerConfig::default()), request).await;
    assert_eq!(status, StatusCode::UNPROCESSABLE_ENTITY);
    let value: serde_json::Value = serde_json::from_slice(&body).expect("JSON");
    assert_eq!(value["detail"][0]["type"], "unsupported", "{value}");
}

#[tokio::test]
async fn gzip_request_is_decompressed() {
    let body = noul_body();
    let mut encoder = GzEncoder::new(Vec::new(), Compression::default());
    encoder.write_all(body.as_bytes()).expect("write gzip");
    let compressed = encoder.finish().expect("finish gzip");

    let request = Request::builder()
        .method("POST")
        .uri("/v1/systemone")
        .header(header::CONTENT_TYPE, "application/json")
        .header(header::CONTENT_ENCODING, "gzip")
        .body(Body::from(compressed))
        .expect("request");
    let (status, _, body) = send(app(ServerConfig::default()), request).await;
    assert_eq!(status, StatusCode::OK, "{}", String::from_utf8_lossy(&body));
}

#[tokio::test]
async fn health_is_open() {
    let request = Request::builder()
        .method("GET")
        .uri("/health")
        .body(Body::empty())
        .expect("request");
    let (status, _, body) = send(app(ServerConfig::default()), request).await;
    assert_eq!(status, StatusCode::OK);
    let value: serde_json::Value = serde_json::from_slice(&body).expect("JSON");
    assert_eq!(value["status"], "ok");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn concurrent_requests_are_batched() {
    let batches = Arc::new(Mutex::new(Vec::new()));
    let model: Arc<dyn DecisionModel> = Arc::new(TrackingModel {
        batches: batches.clone(),
    });
    let registry = Arc::new(
        ModelRegistry::new(
            vec![manifest("laya-test", &["laya-latest"])],
            Arc::new(TrackingFactory { model }),
            RegistryConfig::default(),
        )
        .expect("registry"),
    );
    let telemetry = Arc::new(RecordingTelemetry::default());
    let app = router_with_extensions(
        registry,
        ServerConfig {
            batching: Some(BatchConfig {
                max_batch: 8,
                max_wait: Duration::from_millis(100),
            }),
            enable_metrics: true,
            ..ServerConfig::default()
        },
        ServerExtensions {
            telemetry: Some(telemetry.clone()),
            ..ServerExtensions::default()
        },
    );

    let (a, b, c, d) = tokio::join!(
        app.clone().oneshot(post("/v1/systemone", noul_body())),
        app.clone().oneshot(post("/v1/systemone", noul_body())),
        app.clone().oneshot(post("/v1/systemone", noul_body())),
        app.clone().oneshot(post("/v1/systemone", noul_body())),
    );

    for response in [a, b, c, d] {
        assert_eq!(response.expect("response").status(), StatusCode::OK);
    }
    let batches = batches.lock().expect("batch log").clone();
    assert_eq!(batches.iter().sum::<usize>(), 4, "batches: {batches:?}");
    assert!(
        batches.iter().any(|&size| size >= 2),
        "no request shared a batch: {batches:?}"
    );

    let (status, _, body) = send(app, get("/api/metrics")).await;
    assert_eq!(status, StatusCode::OK);
    let value: serde_json::Value = serde_json::from_slice(&body).expect("JSON");
    assert_eq!(value["batching"]["items"], 4);
    assert!(value["batching"]["batches"].as_u64().unwrap() >= 1);
    assert!(value["models"]["loads"].as_u64().unwrap() >= 1);

    let batch_events = telemetry
        .batches
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    assert!(!batch_events.is_empty());
    assert_eq!(
        batch_events.iter().map(|event| event.items).sum::<usize>(),
        4
    );
}

#[tokio::test]
async fn admin_api_is_disabled_by_default() {
    let (status, _, _) = send(app(ServerConfig::default()), get("/api/models")).await;
    assert_eq!(status, StatusCode::NOT_FOUND);
}

#[tokio::test]
async fn admin_api_uses_the_configured_api_key() {
    let app = app(ServerConfig {
        api_key: Some("secret".to_owned()),
        enable_admin: true,
        batching: None,
        ..ServerConfig::default()
    });

    let (status, _, _) = send(app.clone(), get("/api/models")).await;
    assert_eq!(status, StatusCode::UNAUTHORIZED);

    let request = Request::builder()
        .method("GET")
        .uri("/api/models")
        .header(header::AUTHORIZATION, "Bearer secret")
        .body(Body::empty())
        .expect("request");
    let (status, _, body) = send(app, request).await;
    assert_eq!(status, StatusCode::OK, "{}", String::from_utf8_lossy(&body));
}

#[tokio::test]
async fn admin_load_unload_reload_manage_residency() {
    let app = app(ServerConfig {
        enable_admin: true,
        batching: None,
        ..ServerConfig::default()
    });

    let (status, _, body) = send(app.clone(), get("/api/models")).await;
    assert_eq!(status, StatusCode::OK);
    let value: serde_json::Value = serde_json::from_slice(&body).expect("JSON");
    assert_eq!(value["models"][0]["name"], "laya-test");
    assert_eq!(value["models"][0]["aliases"][0], "laya-latest");
    assert_eq!(value["models"][0]["loaded"], false);
    assert!(loaded_names(&body).is_empty());

    // Loading an alias reports the canonical name.
    let (status, _, body) = send(app.clone(), post("/api/models/laya-latest/load", "")).await;
    assert_eq!(status, StatusCode::OK, "{}", String::from_utf8_lossy(&body));
    let value: serde_json::Value = serde_json::from_slice(&body).expect("JSON");
    assert_eq!(value["model"], "laya-test");
    assert_eq!(value["loaded"], true);

    let (_, _, body) = send(app.clone(), get("/api/models")).await;
    assert_eq!(loaded_names(&body), vec!["laya-test".to_owned()]);
    let value: serde_json::Value = serde_json::from_slice(&body).expect("JSON");
    assert_eq!(value["models"][0]["loaded"], true);

    let (status, _, body) = send(app.clone(), post("/api/models/laya-test/unload", "")).await;
    assert_eq!(status, StatusCode::OK);
    let value: serde_json::Value = serde_json::from_slice(&body).expect("JSON");
    assert_eq!(value["was_loaded"], true);
    assert!(loaded_names(&send(app.clone(), get("/api/models")).await.2).is_empty());

    // Unload is idempotent.
    let (status, _, body) = send(app.clone(), post("/api/models/laya-test/unload", "")).await;
    assert_eq!(status, StatusCode::OK);
    let value: serde_json::Value = serde_json::from_slice(&body).expect("JSON");
    assert_eq!(value["was_loaded"], false);

    let (status, _, body) = send(app.clone(), post("/api/models/laya-test/reload", "")).await;
    assert_eq!(status, StatusCode::OK, "{}", String::from_utf8_lossy(&body));
    let value: serde_json::Value = serde_json::from_slice(&body).expect("JSON");
    assert_eq!(value["was_loaded"], false);
    assert_eq!(
        loaded_names(&send(app.clone(), get("/api/models")).await.2),
        vec!["laya-test".to_owned()]
    );

    let (status, _, _) = send(app, post("/api/models/missing/load", "")).await;
    assert_eq!(status, StatusCode::NOT_FOUND);
}

#[tokio::test]
async fn admin_adds_and_removes_models_through_a_resolver() {
    let resolver: Arc<dyn ModelResolver> = Arc::new(FakeResolver);
    let app = app_with_resolver(
        ServerConfig {
            enable_admin: true,
            batching: None,
            ..ServerConfig::default()
        },
        Some(resolver),
    );

    let (status, _, body) = send(
        app.clone(),
        post("/api/models", r#"{"reference":"test:dynamic","load":true}"#),
    )
    .await;
    assert_eq!(
        status,
        StatusCode::CREATED,
        "{}",
        String::from_utf8_lossy(&body)
    );
    let value: serde_json::Value = serde_json::from_slice(&body).expect("JSON");
    assert_eq!(value["models"][0], "dynamic-test");
    assert_eq!(value["loaded"], "dynamic-test");

    let (_, _, body) = send(app.clone(), get("/api/models")).await;
    let value: serde_json::Value = serde_json::from_slice(&body).expect("JSON");
    let models = value["models"].as_array().expect("models");
    let dynamic = models
        .iter()
        .find(|model| model["name"] == "dynamic-test")
        .expect("dynamic model");
    assert_eq!(dynamic["origin"], "runtime");
    assert_eq!(dynamic["loaded"], true);
    let configured = models
        .iter()
        .find(|model| model["name"] == "laya-test")
        .expect("configured model");
    assert_eq!(configured["origin"], "configured");

    // The alias is usable immediately, and the model can be loaded on demand.
    let (status, _, body) = send(app.clone(), post("/api/models/dynamic-alias/load", "")).await;
    assert_eq!(status, StatusCode::OK, "{}", String::from_utf8_lossy(&body));
    let value: serde_json::Value = serde_json::from_slice(&body).expect("JSON");
    assert_eq!(value["model"], "dynamic-test");

    let (_, _, body) = send(
        app.clone(),
        post(
            "/v1/systemone",
            serde_json::json!({
                "model": "dynamic-test",
                "state": "hello",
                "questions": {
                    "billing": { "type": "noul", "instructions": "Is this about billing?" }
                }
            })
            .to_string(),
        ),
    )
    .await;
    let response: SystemOneResponse = serde_json::from_slice(&body).expect("response JSON");
    assert_eq!(response.model, "dynamic-test");

    // A failed-resolution request is a client error and changes nothing.
    let (status, _, body) = send(
        app.clone(),
        post("/api/models", r#"{"reference":"test:missing"}"#),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    let value: serde_json::Value = serde_json::from_slice(&body).expect("JSON");
    assert!(
        value["detail"]
            .as_str()
            .expect("detail")
            .contains("unknown")
    );

    // Adding the same reference twice conflicts and leaves the catalog intact.
    let (status, _, _) = send(
        app.clone(),
        post("/api/models", r#"{"reference":"test:dynamic"}"#),
    )
    .await;
    assert_eq!(status, StatusCode::CONFLICT);

    let (status, _, body) = send(app.clone(), delete("/api/models/dynamic-test")).await;
    assert_eq!(status, StatusCode::OK, "{}", String::from_utf8_lossy(&body));
    let value: serde_json::Value = serde_json::from_slice(&body).expect("JSON");
    assert_eq!(value["removed"], true);

    let (_, _, body) = send(app.clone(), get("/v1/models")).await;
    let models: ModelMetadataList = serde_json::from_slice(&body).expect("models JSON");
    assert!(
        !models
            .models
            .iter()
            .any(|model| model.name == "dynamic-test")
    );
    assert!(
        !models
            .models
            .iter()
            .any(|model| model.name == "dynamic-alias")
    );

    let (status, _, _) = send(app, delete("/api/models/dynamic-test")).await;
    assert_eq!(status, StatusCode::NOT_FOUND);
}

struct PartialResolver;

impl ModelResolver for PartialResolver {
    fn resolve(&self, _reference: &str) -> Result<Vec<ModelManifest>, ModelError> {
        Ok(vec![manifest("new-one", &[]), manifest("laya-test", &[])])
    }

    fn release(&self, _name: &str) -> bool {
        true
    }
}

#[tokio::test]
async fn admin_add_rolls_back_a_partial_registration() {
    let resolver: Arc<dyn ModelResolver> = Arc::new(PartialResolver);
    let app = app_with_resolver(
        ServerConfig {
            enable_admin: true,
            batching: None,
            ..ServerConfig::default()
        },
        Some(resolver),
    );

    let (status, _, body) = send(
        app.clone(),
        post("/api/models", r#"{"reference":"test:partial"}"#),
    )
    .await;
    assert_eq!(
        status,
        StatusCode::CONFLICT,
        "{}",
        String::from_utf8_lossy(&body)
    );

    let (_, _, body) = send(app.clone(), get("/v1/models")).await;
    let models: ModelMetadataList = serde_json::from_slice(&body).expect("models JSON");
    let names = models
        .models
        .into_iter()
        .map(|model| model.name)
        .collect::<Vec<_>>();
    assert_eq!(names, vec!["laya-test", "laya-latest"]);

    let (status, _, _) = send(app.clone(), post("/api/models/new-one/load", "")).await;
    assert_eq!(status, StatusCode::NOT_FOUND);

    // The pre-existing model is still servable.
    let (status, _, body) = send(app, post("/v1/systemone", noul_body())).await;
    assert_eq!(status, StatusCode::OK, "{}", String::from_utf8_lossy(&body));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn concurrent_admin_adds_keep_one_entry() {
    let resolver: Arc<dyn ModelResolver> = Arc::new(FakeResolver);
    let app = app_with_resolver(
        ServerConfig {
            enable_admin: true,
            batching: None,
            ..ServerConfig::default()
        },
        Some(resolver),
    );

    let (a, b) = tokio::join!(
        send(
            app.clone(),
            post("/api/models", r#"{"reference":"test:dynamic"}"#)
        ),
        send(
            app.clone(),
            post("/api/models", r#"{"reference":"test:dynamic"}"#)
        ),
    );
    let mut statuses = [a.0, b.0];
    statuses.sort();
    assert_eq!(statuses, [StatusCode::CREATED, StatusCode::CONFLICT]);

    let (_, _, body) = send(app, get("/v1/models")).await;
    let models: ModelMetadataList = serde_json::from_slice(&body).expect("models JSON");
    assert_eq!(
        models
            .models
            .iter()
            .filter(|model| model.name == "dynamic-test")
            .count(),
        1
    );
}

#[tokio::test]
async fn admin_can_populate_an_empty_catalog() {
    let registry = Arc::new(
        ModelRegistry::new(Vec::new(), Arc::new(FakeFactory), RegistryConfig::default())
            .expect("empty registry"),
    );
    let resolver: Arc<dyn ModelResolver> = Arc::new(FakeResolver);
    let app = router_with_resolver(
        registry,
        ServerConfig {
            enable_admin: true,
            batching: None,
            ..ServerConfig::default()
        },
        Some(resolver),
    );

    let (status, _, body) = send(app.clone(), get("/v1/models")).await;
    assert_eq!(status, StatusCode::OK);
    let models: ModelMetadataList = serde_json::from_slice(&body).expect("models JSON");
    assert!(models.models.is_empty());

    let (status, _, body) = send(
        app.clone(),
        post("/api/models", r#"{"reference":"test:dynamic"}"#),
    )
    .await;
    assert_eq!(
        status,
        StatusCode::CREATED,
        "{}",
        String::from_utf8_lossy(&body)
    );

    let body = serde_json::json!({
        "model": "dynamic-test",
        "state": "hello",
        "questions": {
            "billing": { "type": "noul", "instructions": "Is this about billing?" }
        }
    })
    .to_string();
    let (status, _, body) = send(app, post("/v1/systemone", body)).await;
    assert_eq!(status, StatusCode::OK, "{}", String::from_utf8_lossy(&body));
    let response: SystemOneResponse = serde_json::from_slice(&body).expect("response JSON");
    assert_eq!(response.model, "dynamic-test");
}

fn cached_model(key: &str, name: &str) -> CachedModel {
    CachedModel {
        key: key.to_owned(),
        name: name.to_owned(),
        directory: format!("fake-cache/models/{key}"),
        size_bytes: 42,
        last_used_unix: 1_700_000_000,
        evictable: true,
        files: vec![CachedFile {
            path: "model.bin".to_owned(),
            size: 42,
        }],
    }
}

#[tokio::test]
async fn admin_cache_lists_and_removes_entries() {
    let registry = Arc::new(
        ModelRegistry::new(
            vec![manifest("laya-test", &[])],
            Arc::new(FakeFactory),
            RegistryConfig::default(),
        )
        .expect("registry"),
    );
    let cache: Arc<dyn CacheManager> = Arc::new(FakeCache {
        models: Mutex::new(vec![
            cached_model("key-a", "laya-test"),
            cached_model("key-b", "other-model"),
        ]),
    });
    let telemetry = Arc::new(RecordingTelemetry::default());
    let app = router_with_extensions(
        registry,
        ServerConfig {
            enable_admin: true,
            batching: None,
            ..ServerConfig::default()
        },
        ServerExtensions {
            cache: Some(cache),
            telemetry: Some(telemetry.clone()),
            ..ServerExtensions::default()
        },
    );

    let (status, _, body) = send(app.clone(), get("/api/cache")).await;
    assert_eq!(status, StatusCode::OK);
    let value: serde_json::Value = serde_json::from_slice(&body).expect("JSON");
    assert_eq!(value["cache_dir"], "fake-cache");
    assert_eq!(value["models"].as_array().unwrap().len(), 2);
    assert_eq!(value["models"][0]["key"], "key-a");
    assert_eq!(value["models"][0]["size_bytes"], 42);
    assert_eq!(value["models"][0]["last_used_unix"], 1_700_000_000);
    assert_eq!(value["models"][0]["evictable"], true);
    assert_eq!(value["limit_bytes"], 100);
    assert_eq!(value["used_bytes"], 84);
    assert_eq!(value["free_bytes"], 16);
    assert_eq!(value["reclaimable_bytes"], 84);
    assert_eq!(value["model_count"], 2);
    assert_eq!(value["resident_count"], 0);

    // A resident model cannot have its cache deleted underneath it.
    let (status, _, _) = send(app.clone(), post("/api/models/laya-test/load", "")).await;
    assert_eq!(status, StatusCode::OK);
    let (status, _, body) = send(app.clone(), delete("/api/cache/key-a")).await;
    assert_eq!(status, StatusCode::CONFLICT);
    assert!(
        String::from_utf8_lossy(&body).contains("is loaded"),
        "{}",
        String::from_utf8_lossy(&body)
    );

    let (status, _, _) = send(app.clone(), post("/api/models/laya-test/unload", "")).await;
    assert_eq!(status, StatusCode::OK);
    let (status, _, body) = send(app.clone(), delete("/api/cache/key-a")).await;
    assert_eq!(status, StatusCode::OK);
    let value: serde_json::Value = serde_json::from_slice(&body).expect("JSON");
    assert_eq!(value["removed"], true);

    let (status, _, body) = send(app.clone(), delete("/api/cache/key-b")).await;
    assert_eq!(status, StatusCode::OK);
    let value: serde_json::Value = serde_json::from_slice(&body).expect("JSON");
    assert_eq!(value["removed"], true);

    let (status, _, body) = send(app.clone(), delete("/api/cache/busy-key")).await;
    assert_eq!(status, StatusCode::CONFLICT);
    assert!(
        String::from_utf8_lossy(&body).contains("busy"),
        "{}",
        String::from_utf8_lossy(&body)
    );

    let (_, _, body) = send(app, get("/api/cache")).await;
    let value: serde_json::Value = serde_json::from_slice(&body).expect("JSON");
    assert!(value["models"].as_array().unwrap().is_empty());

    let events = telemetry
        .caches
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    assert!(
        events
            .iter()
            .any(|event| event.action == sys1_server::CacheAction::List && event.success)
    );
    assert!(
        events
            .iter()
            .any(|event| event.action == sys1_server::CacheAction::Remove && event.success)
    );
}

#[tokio::test]
async fn cache_api_requires_admin_and_auth() {
    let cache: Arc<dyn CacheManager> = Arc::new(FakeCache {
        models: Mutex::new(Vec::new()),
    });
    let registry = Arc::new(
        ModelRegistry::new(Vec::new(), Arc::new(FakeFactory), RegistryConfig::default())
            .expect("registry"),
    );

    let app = router_with_extensions(
        registry.clone(),
        ServerConfig::default(),
        ServerExtensions {
            cache: Some(cache.clone()),
            ..ServerExtensions::default()
        },
    );
    let (status, _, _) = send(app, get("/api/cache")).await;
    assert_eq!(status, StatusCode::NOT_FOUND);

    let app = router_with_extensions(
        registry,
        ServerConfig {
            api_key: Some("secret".to_owned()),
            enable_admin: true,
            batching: None,
            ..ServerConfig::default()
        },
        ServerExtensions {
            cache: Some(cache),
            ..ServerExtensions::default()
        },
    );
    let (status, _, _) = send(app.clone(), get("/api/cache")).await;
    assert_eq!(status, StatusCode::UNAUTHORIZED);
    let request = Request::builder()
        .method("GET")
        .uri("/api/cache")
        .header(header::AUTHORIZATION, "Bearer secret")
        .body(Body::empty())
        .expect("request");
    let (status, _, _) = send(app, request).await;
    assert_eq!(status, StatusCode::OK);
}

#[tokio::test]
async fn admin_add_with_load_rolls_back_on_failure() {
    let registry = Arc::new(
        ModelRegistry::new(
            Vec::new(),
            Arc::new(FailingFactory),
            RegistryConfig::default(),
        )
        .expect("registry"),
    );
    let resolver: Arc<dyn ModelResolver> = Arc::new(FakeResolver);
    let app = router_with_resolver(
        registry,
        ServerConfig {
            enable_admin: true,
            enable_metrics: true,
            batching: None,
            ..ServerConfig::default()
        },
        Some(resolver),
    );

    let (status, _, body) = send(
        app.clone(),
        post("/api/models", r#"{"reference":"test:dynamic","load":true}"#),
    )
    .await;
    assert_eq!(
        status,
        StatusCode::INTERNAL_SERVER_ERROR,
        "{}",
        String::from_utf8_lossy(&body)
    );

    let (_, _, body) = send(app.clone(), get("/v1/models")).await;
    let models: ModelMetadataList = serde_json::from_slice(&body).expect("models JSON");
    assert!(models.models.is_empty());

    let (_, _, body) = send(app, get("/api/metrics")).await;
    let value: serde_json::Value = serde_json::from_slice(&body).expect("JSON");
    assert_eq!(value["models"]["loads"], 1);
    assert_eq!(value["models"]["load_errors"], 1);
    assert_eq!(value["models"]["loaded_count"], 0);
}

#[tokio::test]
async fn admin_add_requires_a_resolver() {
    let app = app(ServerConfig {
        enable_admin: true,
        batching: None,
        ..ServerConfig::default()
    });
    let (status, _, body) = send(app, post("/api/models", r#"{"reference":"test:dynamic"}"#)).await;
    assert_eq!(status, StatusCode::NOT_IMPLEMENTED);
    let value: serde_json::Value = serde_json::from_slice(&body).expect("JSON");
    assert!(
        value["detail"]
            .as_str()
            .expect("detail")
            .contains("not configured")
    );
}

#[tokio::test]
async fn telemetry_and_trace_context_are_replaceable() {
    let telemetry = Arc::new(RecordingTelemetry::default());
    let registry = Arc::new(
        ModelRegistry::new(
            vec![manifest("laya-test", &["laya-latest"])],
            Arc::new(FakeFactory),
            RegistryConfig::default(),
        )
        .expect("registry"),
    );
    let app = router_with_extensions(
        registry,
        ServerConfig {
            batching: None,
            ..ServerConfig::default()
        },
        ServerExtensions {
            telemetry: Some(telemetry.clone()),
            ..ServerExtensions::default()
        },
    );

    let trace_id = "4bf92f3577b34da6a3ce929d0e0e4736";
    let request = Request::builder()
        .method("POST")
        .uri("/v1/systemone")
        .header(header::CONTENT_TYPE, "application/json")
        .header("traceparent", format!("00-{trace_id}-00f067aa0ba902b7-01"))
        .body(Body::from(noul_body()))
        .expect("request");
    let (status, headers, body) = send(app, request).await;
    assert_eq!(status, StatusCode::OK, "{}", String::from_utf8_lossy(&body));

    let traceparent = headers
        .get("traceparent")
        .and_then(|value| value.to_str().ok())
        .expect("traceparent response header");
    assert!(
        traceparent.starts_with(&format!("00-{trace_id}-")),
        "{traceparent}"
    );
    assert!(traceparent.ends_with("-01"), "{traceparent}");
    let request_id = headers
        .get("x-typesafe-request-id")
        .and_then(|value| value.to_str().ok())
        .expect("request id header");

    let requests = telemetry
        .requests
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    assert_eq!(requests.len(), 1);
    assert_eq!(requests[0].trace_id, trace_id);
    assert_eq!(requests[0].request_id, request_id);
    assert_eq!(requests[0].status, 200);
    assert_eq!(requests[0].model.as_deref(), Some("laya-test"));

    let models = telemetry
        .models
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    assert!(models.iter().any(|event| {
        event.model == "laya-test"
            && event.action == ModelAction::Load
            && event.success
            && event.trace_id == trace_id
    }));
}

#[tokio::test]
async fn config_reload_is_admin_protected_and_reports_changes() {
    let telemetry = Arc::new(RecordingTelemetry::default());
    let app = app_with_extensions(
        ServerConfig {
            api_key: Some("secret".to_owned()),
            enable_admin: true,
            batching: None,
            ..ServerConfig::default()
        },
        ServerExtensions {
            reload: Some(Arc::new(OkReloader)),
            telemetry: Some(telemetry.clone()),
            ..ServerExtensions::default()
        },
    );

    let (status, _, _) = send(app.clone(), post("/api/config/reload", "")).await;
    assert_eq!(status, StatusCode::UNAUTHORIZED);

    let request = Request::builder()
        .method("POST")
        .uri("/api/config/reload")
        .header(header::AUTHORIZATION, "Bearer secret")
        .body(Body::empty())
        .expect("request");
    let (status, _, body) = send(app, request).await;
    assert_eq!(status, StatusCode::OK, "{}", String::from_utf8_lossy(&body));
    let value: serde_json::Value = serde_json::from_slice(&body).expect("JSON");
    assert_eq!(value["config_path"], "fake.toml");
    assert_eq!(value["added"][0], "added-model");
    assert_eq!(value["unchanged"][0], "laya-test");

    let events = telemetry
        .configs
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    assert_eq!(events.len(), 1);
    assert!(events[0].success);
    assert_eq!(events[0].added, vec!["added-model".to_owned()]);
}

#[tokio::test]
async fn config_reload_reports_restart_required() {
    let app = app_with_extensions(
        ServerConfig {
            enable_admin: true,
            batching: None,
            ..ServerConfig::default()
        },
        ServerExtensions {
            reload: Some(Arc::new(RestartReloader)),
            ..ServerExtensions::default()
        },
    );
    let (status, _, body) = send(app, post("/api/config/reload", "")).await;
    assert_eq!(status, StatusCode::CONFLICT);
    assert!(
        String::from_utf8_lossy(&body).contains("restart required"),
        "{}",
        String::from_utf8_lossy(&body)
    );
}

#[tokio::test]
async fn config_reload_failure_is_bad_request() {
    let app = app_with_extensions(
        ServerConfig {
            enable_admin: true,
            batching: None,
            ..ServerConfig::default()
        },
        ServerExtensions {
            reload: Some(Arc::new(FailedReloader)),
            ..ServerExtensions::default()
        },
    );
    let (status, _, body) = send(app, post("/api/config/reload", "")).await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert!(
        String::from_utf8_lossy(&body).contains("bad config"),
        "{}",
        String::from_utf8_lossy(&body)
    );
}

#[tokio::test]
async fn config_reload_requires_admin_and_a_reloader() {
    let (status, _, _) = send(app(ServerConfig::default()), post("/api/config/reload", "")).await;
    assert_eq!(status, StatusCode::NOT_FOUND);

    let app = app_with_extensions(
        ServerConfig {
            enable_admin: true,
            batching: None,
            ..ServerConfig::default()
        },
        ServerExtensions::default(),
    );
    let (status, _, _) = send(app, post("/api/config/reload", "")).await;
    assert_eq!(status, StatusCode::NOT_FOUND);
}

#[tokio::test]
async fn cors_can_be_permissive_restricted_or_disabled() {
    let request = |origin: &str| {
        Request::builder()
            .method("GET")
            .uri("/health")
            .header(header::ORIGIN, origin)
            .body(Body::empty())
            .expect("request")
    };

    let (_, headers, _) = send(app(ServerConfig::default()), request("https://app.example")).await;
    assert_eq!(
        headers
            .get(header::ACCESS_CONTROL_ALLOW_ORIGIN)
            .and_then(|value| value.to_str().ok()),
        Some("*")
    );

    let restricted = app(ServerConfig {
        cors_origins: Some(vec!["https://app.example".to_owned()]),
        ..ServerConfig::default()
    });
    let (_, headers, _) = send(restricted.clone(), request("https://app.example")).await;
    assert_eq!(
        headers
            .get(header::ACCESS_CONTROL_ALLOW_ORIGIN)
            .and_then(|value| value.to_str().ok()),
        Some("https://app.example")
    );
    let (_, headers, _) = send(restricted, request("https://evil.example")).await;
    assert!(
        headers.get(header::ACCESS_CONTROL_ALLOW_ORIGIN).is_none(),
        "disallowed origin received CORS headers"
    );

    let disabled = app(ServerConfig {
        cors_origins: Some(Vec::new()),
        ..ServerConfig::default()
    });
    let (_, headers, _) = send(disabled, request("https://app.example")).await;
    assert!(headers.get(header::ACCESS_CONTROL_ALLOW_ORIGIN).is_none());
}

#[tokio::test]
async fn metrics_are_disabled_by_default() {
    let app = app(ServerConfig::default());
    let (status, _, _) = send(app.clone(), get("/api/metrics")).await;
    assert_eq!(status, StatusCode::NOT_FOUND);
    let (status, _, _) = send(app, get("/metrics")).await;
    assert_eq!(status, StatusCode::NOT_FOUND);
}

#[tokio::test]
async fn metrics_track_systemone_and_model_lifecycle() {
    let app = app(ServerConfig {
        enable_admin: true,
        enable_metrics: true,
        batching: None,
        ..ServerConfig::default()
    });

    let (status, _, body) = send(app.clone(), get("/api/metrics")).await;
    assert_eq!(status, StatusCode::OK);
    let value: serde_json::Value = serde_json::from_slice(&body).expect("JSON");
    assert_eq!(value["systemone"]["requests"], 0);
    assert_eq!(value["models"]["loads"], 0);

    let (status, _, body) = send(app.clone(), post("/v1/systemone", noul_body())).await;
    assert_eq!(status, StatusCode::OK, "{}", String::from_utf8_lossy(&body));

    let body = serde_json::json!({
        "model": "missing",
        "state": "hello",
        "questions": {
            "billing": { "type": "noul", "instructions": "Is this about billing?" }
        }
    })
    .to_string();
    let (status, _, _) = send(app.clone(), post("/v1/systemone", body)).await;
    assert_eq!(status, StatusCode::BAD_REQUEST);

    let (status, _, body) = send(app.clone(), get("/api/metrics")).await;
    assert_eq!(status, StatusCode::OK);
    let value: serde_json::Value = serde_json::from_slice(&body).expect("JSON");
    assert_eq!(value["systemone"]["requests"], 2);
    assert_eq!(value["systemone"]["errors"], 1);
    assert_eq!(value["models"]["requests"]["laya-test"], 1);
    assert_eq!(value["models"]["loads"], 1);
    assert_eq!(value["models"]["loaded_count"], 1);
    assert_eq!(value["models"]["loaded"][0], "laya-test");

    let (status, headers, body) = send(app.clone(), get("/metrics")).await;
    assert_eq!(status, StatusCode::OK);
    assert!(
        headers
            .get(header::CONTENT_TYPE)
            .and_then(|value| value.to_str().ok())
            .is_some_and(|value| value.starts_with("text/plain"))
    );
    let text = String::from_utf8_lossy(&body);
    assert!(text.contains("sys1_systemone_requests_total 2"), "{text}");
    assert!(text.contains("sys1_systemone_errors_total 1"), "{text}");
    assert!(text.contains("sys1_model_loads_total 1"), "{text}");
    assert!(
        text.contains("sys1_model_requests_total{model=\"laya-test\"} 1"),
        "{text}"
    );

    let (status, _, _) = send(app.clone(), post("/api/models/laya-test/unload", "")).await;
    assert_eq!(status, StatusCode::OK);
    let (_, _, body) = send(app, get("/api/metrics")).await;
    let value: serde_json::Value = serde_json::from_slice(&body).expect("JSON");
    assert_eq!(value["models"]["unloads"], 1);
    assert_eq!(value["models"]["loaded_count"], 0);
}

#[tokio::test]
async fn metrics_require_auth_when_configured() {
    let app = app(ServerConfig {
        api_key: Some("secret".to_owned()),
        enable_metrics: true,
        batching: None,
        ..ServerConfig::default()
    });

    let (status, _, _) = send(app.clone(), get("/api/metrics")).await;
    assert_eq!(status, StatusCode::UNAUTHORIZED);

    let request = Request::builder()
        .method("GET")
        .uri("/metrics")
        .header(header::AUTHORIZATION, "Bearer secret")
        .body(Body::empty())
        .expect("request");
    let (status, headers, body) = send(app, request).await;
    assert_eq!(status, StatusCode::OK);
    assert!(
        headers
            .get(header::CONTENT_TYPE)
            .and_then(|value| value.to_str().ok())
            .is_some_and(|value| value.starts_with("text/plain"))
    );
    assert!(String::from_utf8_lossy(&body).contains("sys1_systemone_requests_total 0"));
}

#[tokio::test]
async fn model_can_opt_out_of_batching() {
    let registry = Arc::new(
        ModelRegistry::new(
            vec![manifest("laya-test", &["laya-latest"])],
            Arc::new(NoBatchFactory),
            RegistryConfig::default(),
        )
        .expect("registry"),
    );
    let app = router(
        registry,
        ServerConfig {
            batching: Some(BatchConfig {
                max_batch: 8,
                max_wait: Duration::from_millis(50),
            }),
            ..ServerConfig::default()
        },
    );

    let (status, _, body) = send(app, post("/v1/systemone", noul_body())).await;
    assert_eq!(status, StatusCode::OK, "{}", String::from_utf8_lossy(&body));
}
