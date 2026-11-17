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
    ModelRegistry, Prediction, Questions, RegistryConfig,
};
use sys1_protocol::{
    Answer, ChoiceAnswer, ModelMetadataList, NoulAnswer, Question, ScoreAnswer, State,
    SystemOneResponse, Usage,
};
use sys1_server::{BatchConfig, ServerConfig, router};
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
                        "Laya does not support box questions",
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
    let registry = ModelRegistry::new(
        vec![manifest("laya-test", &["laya-latest"])],
        Arc::new(FakeFactory),
        RegistryConfig::default(),
    )
    .expect("registry");
    router(Arc::new(registry), config)
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
    let app = router(
        registry,
        ServerConfig {
            batching: Some(BatchConfig {
                max_batch: 8,
                max_wait: Duration::from_millis(100),
            }),
            ..ServerConfig::default()
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
    assert!(
        batches.iter().any(|&size| size >= 4),
        "batches: {batches:?}"
    );
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
