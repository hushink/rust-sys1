use std::path::PathBuf;
use sys1::{EngineOptions, InferenceEngine, Questions, State};

fn directory(name: &str) -> PathBuf {
    PathBuf::from(std::env::var(name).unwrap_or_else(|_| panic!("set {name}")))
}

#[test]
#[ignore = "requires a local NLI ModernBERT checkpoint directory"]
fn nli_engine_round_trip() {
    let directory = directory("SYS1_NLI_TEST_DIR");
    let name = directory
        .file_name()
        .unwrap()
        .to_string_lossy()
        .into_owned();
    let engine = InferenceEngine::from_directory(&directory, EngineOptions::default())
        .expect("create engine");
    let models = engine.models();
    assert_eq!(models.len(), 1);
    assert_eq!(models[0].family, "nli");
    let state = State::text("I was charged twice for invoice 12345 and need a refund today.");
    let questions = Questions::new()
        .noul("billing", "This message is about billing.")
        .choice("tone", [("angry", "Upset"), ("calm", "Neutral")]);
    let output = engine.predict(&name, &state, &questions).expect("predict");
    assert_eq!(output.answers.len(), 2);
    engine.unload(&name);
}

#[test]
#[ignore = "requires a local GLiClass edge checkpoint directory"]
fn gliclass_engine_round_trip() {
    let directory = directory("SYS1_GLICLASS_TEST_DIR");
    let name = directory
        .file_name()
        .unwrap()
        .to_string_lossy()
        .into_owned();
    let engine = InferenceEngine::from_directory(&directory, EngineOptions::default())
        .expect("create engine");
    let models = engine.models();
    assert_eq!(models.len(), 1);
    assert_eq!(models[0].family, "gliclass");
    let state = State::text("I was charged twice for invoice 12345 and need a refund today.");
    let questions = Questions::new()
        .noul("billing", "This message is about billing.")
        .score("urgency", ["Can wait", "This week", "Today"]);
    let output = engine.predict(&name, &state, &questions).expect("predict");
    assert_eq!(output.answers.len(), 2);
    engine.unload(&name);
}
