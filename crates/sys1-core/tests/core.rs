use indexmap::IndexMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;
use sys1_core::{
    AnswerError, Answers, DecisionModel, DecisionOutput, DevicePreference, Error, ManifestError,
    ModelError, ModelFactory, ModelFile, ModelManifest, ModelRegistry, Questions, RegistryConfig,
    ResolvedDevice,
};
use sys1_protocol::{
    Answer, AnswerType, ChoiceAnswer, NoulAnswer, Question, ScoreAnswer, State, Usage,
};

fn manifest(name: &str, aliases: &[&str]) -> ModelManifest {
    ModelManifest {
        schema_version: 1,
        name: name.to_owned(),
        description: format!("{name} test model"),
        release_date: "2026-01-01".to_owned(),
        family: "test".to_owned(),
        variant: None,
        backend: "fake".to_owned(),
        format: "test".to_owned(),
        version: "1.0.0".to_owned(),
        aliases: aliases.iter().map(|alias| (*alias).to_owned()).collect(),
        files: vec![ModelFile {
            path: "model.bin".to_owned(),
            url: "https://example.com/model.bin".to_owned(),
            sha256: Some("0".repeat(64)),
            git_sha1: None,
            size: 1,
        }],
    }
}

struct FakeModel;

impl DecisionModel for FakeModel {
    fn predict(&self, _state: &State, questions: &Questions) -> Result<DecisionOutput, ModelError> {
        let answers = questions
            .as_map()
            .keys()
            .map(|id| (id.clone(), Answer::Noul(NoulAnswer::new(0.75))))
            .collect::<IndexMap<_, _>>();
        Ok(DecisionOutput::new(Answers::new(answers), Usage::new(3, 0)))
    }
}

#[derive(Default)]
struct FakeFactory {
    loads: AtomicUsize,
}

impl ModelFactory for FakeFactory {
    fn load(&self, _manifest: &ModelManifest) -> Result<Arc<dyn DecisionModel>, ModelError> {
        self.loads.fetch_add(1, Ordering::SeqCst);
        Ok(Arc::new(FakeModel))
    }
}

fn registry(
    manifests: Vec<ModelManifest>,
    max_loaded: usize,
) -> (Arc<ModelRegistry>, Arc<FakeFactory>) {
    let factory = Arc::new(FakeFactory::default());
    let registry = Arc::new(
        ModelRegistry::new(
            manifests,
            factory.clone(),
            RegistryConfig::new(max_loaded, Duration::from_secs(300)),
        )
        .expect("valid registry"),
    );
    (registry, factory)
}

#[test]
fn questions_builder_and_validation() {
    let questions = Questions::new()
        .choice("tone", [("calm", "neutral"), ("angry", "upset")])
        .score("urgency", ["low", "high"])
        .noul("billing", "Is this about billing?");

    assert_eq!(questions.len(), 3);
    assert!(questions.validate().is_ok());

    let Question::Choice(choice) = questions.get("tone").expect("tone question") else {
        panic!("expected choice question");
    };
    assert_eq!(
        choice.criteria.keys().cloned().collect::<Vec<_>>(),
        vec!["calm", "angry"]
    );

    assert!(Questions::new().validate().is_err());
}

#[test]
fn answers_typed_accessors() {
    let mut map = IndexMap::new();
    map.insert(
        "tone".to_owned(),
        Answer::Choice(ChoiceAnswer::new(
            "calm",
            0.8,
            IndexMap::from([("calm".to_owned(), 0.8)]),
        )),
    );
    map.insert(
        "urgency".to_owned(),
        Answer::Score(ScoreAnswer::new(1.5, 0.6, IndexMap::new(), IndexMap::new())),
    );
    map.insert("billing".to_owned(), Answer::Noul(NoulAnswer::new(0.9)));
    let answers = Answers::new(map);

    assert_eq!(answers.choice("tone").unwrap().choice, "calm");
    assert_eq!(answers.score("urgency").unwrap().score, 1.5);
    assert_eq!(answers.noul("billing").unwrap().noul, 0.9);

    assert_eq!(
        answers.noul("tone").unwrap_err(),
        AnswerError::WrongType {
            id: "tone".to_owned(),
            expected: AnswerType::Noul,
            actual: AnswerType::Choice,
        }
    );
    assert_eq!(
        answers.choice("missing").unwrap_err(),
        AnswerError::Missing {
            id: "missing".to_owned(),
        }
    );
}

#[test]
fn predict_through_alias_and_cache() {
    let (registry, factory) = registry(vec![manifest("laya-test", &["laya-latest"])], 3);
    let state = State::text("hello");
    let questions = Questions::new().noul("billing", "Is this about billing?");

    let response = registry
        .predict("laya-latest", &state, &questions)
        .expect("prediction");
    assert_eq!(response.model, "laya-test");
    assert_eq!(response.usage.input_tokens, 3);
    let Answer::Noul(answer) = &response.answers["billing"] else {
        panic!("expected noul answer");
    };
    assert_eq!(answer.noul, 0.75);
    assert_eq!(factory.loads.load(Ordering::SeqCst), 1);

    registry
        .predict("laya-test", &state, &questions)
        .expect("cached prediction");
    assert_eq!(factory.loads.load(Ordering::SeqCst), 1);
}

#[test]
fn try_acquire_only_returns_loaded_models() {
    let (registry, _) = registry(vec![manifest("a", &[])], 1);
    assert!(registry.try_acquire("a").is_none());
    let _model = registry.acquire("a").expect("load a");
    assert!(registry.try_acquire("a").is_some());
}

#[test]
fn model_not_found() {
    let (registry, _) = registry(vec![manifest("a", &[])], 1);
    let error = registry
        .predict(
            "missing",
            &State::text("s"),
            &Questions::new().noul("q", "x"),
        )
        .expect_err("missing model must fail");
    assert!(matches!(error, Error::ModelNotFound { .. }));
}

#[test]
fn validation_errors_reach_caller() {
    let (registry, _) = registry(vec![manifest("a", &[])], 1);
    let error = registry
        .predict("a", &State::text("s"), &Questions::new())
        .expect_err("empty questions must fail");
    assert!(matches!(error, Error::Validation(_)));
}

#[test]
fn lru_evicts_idle_models() {
    let (registry, factory) = registry(vec![manifest("a", &[]), manifest("b", &[])], 1);

    {
        let _model = registry.acquire("a").expect("load a");
    }
    assert_eq!(registry.loaded_models(), vec!["a".to_owned()]);

    {
        let _model = registry.acquire("b").expect("load b");
    }
    assert_eq!(registry.loaded_models(), vec!["b".to_owned()]);
    assert_eq!(factory.loads.load(Ordering::SeqCst), 2);

    {
        let _model = registry.acquire("a").expect("reload a");
    }
    assert_eq!(registry.loaded_models(), vec!["a".to_owned()]);
    assert_eq!(factory.loads.load(Ordering::SeqCst), 3);
}

#[test]
fn maintenance_evicts_after_keep_alive() {
    let factory = Arc::new(FakeFactory::default());
    let registry = ModelRegistry::new(
        vec![manifest("a", &[])],
        factory,
        RegistryConfig::new(1, Duration::ZERO),
    )
    .expect("valid registry");

    {
        let _model = registry.acquire("a").expect("load a");
    }
    std::thread::sleep(Duration::from_millis(1));
    registry.maintenance();
    assert!(registry.loaded_models().is_empty());
}

#[test]
fn list_models_includes_alias() {
    let (registry, _) = registry(vec![manifest("laya-test", &["laya-latest"])], 1);
    let names = registry
        .list_models()
        .into_iter()
        .map(|model| model.name)
        .collect::<Vec<_>>();
    assert_eq!(names, vec!["laya-test", "laya-latest"]);
}

#[test]
fn manifest_rejects_path_traversal() {
    let mut manifest = manifest("a", &[]);
    manifest.files[0].path = "../model.bin".to_owned();
    assert_eq!(
        manifest.validate().unwrap_err(),
        ManifestError::InvalidPath("../model.bin".to_owned())
    );
}

#[test]
fn device_parse_and_resolve() {
    assert_eq!(
        "auto".parse::<DevicePreference>().unwrap(),
        DevicePreference::Auto
    );
    assert_eq!(
        "cuda:2".parse::<DevicePreference>().unwrap(),
        DevicePreference::Cuda(2)
    );
    assert!("bogus".parse::<DevicePreference>().is_err());

    let available = [
        ResolvedDevice::Cpu,
        ResolvedDevice::Cuda(0),
        ResolvedDevice::Cuda(1),
    ];
    assert_eq!(
        DevicePreference::Auto.resolve(&available).unwrap(),
        ResolvedDevice::Cuda(0)
    );
    assert_eq!(
        DevicePreference::Cuda(1).resolve(&available).unwrap(),
        ResolvedDevice::Cuda(1)
    );
    assert!(DevicePreference::Cuda(2).resolve(&available).is_err());
    assert_eq!(
        DevicePreference::Cpu.resolve(&available).unwrap(),
        ResolvedDevice::Cpu
    );
}
