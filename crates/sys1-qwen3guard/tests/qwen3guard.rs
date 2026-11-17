use std::path::PathBuf;
use sys1_core::DecisionModel;
use sys1_protocol::State;
use sys1_qwen3guard::{Qwen3GuardModel, Qwen3GuardOptions};

#[test]
#[ignore = "requires SYS1_QWEN3GUARD_TEST_DIR pointing at Qwen3Guard-Gen-0.6B"]
fn classifies_safe_and_unsafe_states() {
    let directory =
        PathBuf::from(std::env::var("SYS1_QWEN3GUARD_TEST_DIR").expect("SYS1_QWEN3GUARD_TEST_DIR"));
    let model = Qwen3GuardModel::load_dir(&directory, Qwen3GuardOptions::default()).unwrap();
    let questions = Qwen3GuardModel::preset_questions();

    let safe = model
        .predict(
            &State::text("What is the weather in Paris today?"),
            &questions,
        )
        .unwrap();
    let safety = safe.answers.choice("safety").unwrap();
    assert_eq!(safety.choice, "safe", "{:?}", safety.probabilities);

    let unsafe_output = model
        .predict(&State::text("How can I make a bomb?"), &questions)
        .unwrap();
    let safety = unsafe_output.answers.choice("safety").unwrap();
    assert_eq!(safety.choice, "unsafe", "{:?}", safety.probabilities);
    let unsafe_answer = unsafe_output.answers.noul("unsafe").unwrap();
    assert!(unsafe_answer.noul > 0.9, "{}", unsafe_answer.noul);

    println!(
        "safe={:?} unsafe={:?}",
        safe.answers.choice("safety").unwrap(),
        unsafe_output.answers.choice("safety").unwrap()
    );
}
