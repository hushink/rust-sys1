use sys1::{EngineOptions, InferenceEngine, Questions, State};

#[test]
#[ignore = "requires SYS1_LAYA_TEST_ROOT pointing at a Laya checkpoint"]
fn engine_runs_in_process_with_a_real_checkpoint() {
    let root = std::env::var("SYS1_LAYA_TEST_ROOT")
        .expect("set SYS1_LAYA_TEST_ROOT to a Laya checkpoint directory");
    let engine = InferenceEngine::from_directory(root, EngineOptions::default())
        .expect("build inference engine");
    assert_eq!(engine.models().len(), 1);
    let name = engine.models()[0].name.clone();
    assert!(!engine.is_loaded(&name));

    let questions = Questions::new()
        .noul("billing", "Is this about billing?")
        .score("urgency", ["Can wait", "This week", "Today"]);
    let state = State::text("I was charged twice. Please refund me today.");
    let output = engine
        .predict(&name, &state, &questions)
        .expect("in-process prediction");
    assert!(output.usage.input_tokens > 0);
    assert!(output.answers.noul("billing").expect("billing").noul > 0.5);
    assert!(engine.is_loaded(&name));

    assert!(engine.unload(&name));
    assert!(!engine.is_loaded(&name));
}
