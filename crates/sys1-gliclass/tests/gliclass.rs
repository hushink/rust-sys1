use std::path::PathBuf;
use sys1_core::{DecisionModel, Questions};
use sys1_gliclass::{GliclassModel, GliclassOptions};
use sys1_protocol::State;

#[test]
#[ignore = "requires a local GLiClass edge checkpoint directory"]
fn predicts_all_question_types() {
    let directory =
        PathBuf::from(std::env::var("SYS1_GLICLASS_TEST_DIR").expect("SYS1_GLICLASS_TEST_DIR"));
    let model =
        GliclassModel::load_dir(&directory, GliclassOptions::default()).expect("load GLiClass");
    let state = State::text("I was charged twice for invoice 12345 and need a refund today.");
    let questions = Questions::new()
        .choice("tone", [("angry", "Upset"), ("calm", "Neutral")])
        .score("urgency", ["Can wait", "This week", "Today"])
        .noul("billing", "This message is about billing.");

    let output = model.predict(&state, &questions).expect("predict");
    assert_eq!(output.answers.len(), 3);
    assert!(output.usage.input_tokens > 0);

    let choice = output.answers.choice("tone").expect("choice");
    assert_eq!(choice.choice, "calm");
    assert!((choice.probabilities["angry"] - 0.04559547).abs() <= 1e-4);
    assert!((choice.probabilities["calm"] - 0.95440453).abs() <= 1e-4);

    let score = output.answers.score("urgency").expect("score");
    assert!((score.probabilities["0"] - 0.22847050).abs() <= 1e-4);
    assert!((score.probabilities["1"] - 0.00014198).abs() <= 1e-5);
    assert!((score.probabilities["2"] - 0.77138752).abs() <= 1e-4);

    let noul = output.answers.noul("billing").expect("noul");
    assert!((noul.noul - 0.3857523).abs() <= 1e-4);
}
