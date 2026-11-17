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
#[ignore = "requires SYS1_QWEN3GUARD_TEST_DIR"]
fn qwen3guard_engine_round_trip() {
    let directory = directory("SYS1_QWEN3GUARD_TEST_DIR");
    let name = directory
        .file_name()
        .unwrap()
        .to_string_lossy()
        .into_owned();
    let engine = InferenceEngine::from_directory(&directory, EngineOptions::default())
        .expect("create engine");
    assert_eq!(engine.models()[0].family, "qwen3guard");
    let questions = sys1_qwen3guard::Qwen3GuardModel::preset_questions();
    let output = engine
        .predict(&name, &State::text("How can I make a bomb?"), &questions)
        .expect("predict");
    assert_eq!(output.answers.len(), 4);
    engine.unload(&name);
}

#[test]
#[ignore = "requires SYS1_VON_TEST_DIR"]
fn von_engine_round_trip() {
    let directory = directory("SYS1_VON_TEST_DIR");
    let name = directory
        .file_name()
        .unwrap()
        .to_string_lossy()
        .into_owned();
    let engine = InferenceEngine::from_directory(&directory, EngineOptions::default())
        .expect("create engine");
    assert_eq!(engine.models()[0].family, "von");
    let questions = Questions::new()
        .choice("tone", [("angry", "Hostile"), ("calm", "Neutral")])
        .noul("refund", "The customer asks for a refund.");
    let output = engine
        .predict(&name, &State::text("I was charged twice."), &questions)
        .expect("predict");
    assert_eq!(output.answers.len(), 2);
    engine.unload(&name);
}

#[test]
#[ignore = "requires SYS1_DECIMA_TEST_DIR"]
fn decima_engine_round_trip() {
    let directory = directory("SYS1_DECIMA_TEST_DIR");
    let name = directory
        .file_name()
        .unwrap()
        .to_string_lossy()
        .into_owned();
    let engine = InferenceEngine::from_directory(&directory, EngineOptions::default())
        .expect("create engine");
    assert_eq!(engine.models()[0].family, "decima");
    let questions = Questions::new()
        .choice("team", [("billing", "charges"), ("tech", "bugs")])
        .score("anger", ["calm", "annoyed", "furious"]);
    let output = engine
        .predict(&name, &State::text("My card was charged twice"), &questions)
        .expect("predict");
    assert_eq!(output.answers.len(), 2);
    engine.unload(&name);
}

#[test]
#[ignore = "requires SYS1_KEV_TEST_DIR with a base/ directory"]
fn kev_engine_round_trip() {
    let directory = directory("SYS1_KEV_TEST_DIR");
    let name = directory
        .file_name()
        .unwrap()
        .to_string_lossy()
        .into_owned();
    let engine = InferenceEngine::from_directory(&directory, EngineOptions::default())
        .expect("create engine");
    assert_eq!(engine.models()[0].family, "kev");
    let questions = Questions::new()
        .choice("team", [("billing", "charges"), ("tech", "bugs")])
        .noul("refund", "The customer asks for a refund.");
    let output = engine
        .predict(&name, &State::text("My card was charged twice"), &questions)
        .expect("predict");
    assert_eq!(output.answers.len(), 2);
    engine.unload(&name);
}

#[test]
#[ignore = "requires SYS1_DECISION_TEST_DIR"]
fn decision_engine_round_trip() {
    let directory = directory("SYS1_DECISION_TEST_DIR");
    let name = directory
        .file_name()
        .unwrap()
        .to_string_lossy()
        .into_owned();
    let engine = InferenceEngine::from_directory(&directory, EngineOptions::default())
        .expect("create engine");
    assert_eq!(engine.models()[0].family, "decision");
    let questions = Questions::new()
        .choice("team", [("billing", "charges"), ("tech", "bugs")])
        .noul("refund", "The customer asks for a refund.");
    let output = engine
        .predict(&name, &State::text("My card was charged twice"), &questions)
        .expect("predict");
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
