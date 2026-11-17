use std::path::PathBuf;
use sys1_core::{DecisionModel, Questions};
use sys1_protocol::State;
use sys1_von::{VonModel, VonOptions};

#[test]
#[ignore = "requires SYS1_VON_TEST_DIR pointing at the wfzyx/von checkpoint"]
fn predicts_all_question_types() {
    let directory = PathBuf::from(std::env::var("SYS1_VON_TEST_DIR").expect("SYS1_VON_TEST_DIR"));
    let model = VonModel::load_dir(&directory, VonOptions::default()).unwrap();
    let questions = Questions::new()
        .choice(
            "tone",
            [("angry", "Hostile"), ("calm", "Neutral"), ("sad", "Upset")],
        )
        .noul("refund", "The customer asks for a refund.")
        .score("urgency", ["low", "medium", "high"]);
    let output = model
        .predict(
            &State::text("I was charged twice and want my money back today!"),
            &questions,
        )
        .unwrap();
    let choice = output.answers.choice("tone").unwrap();
    assert_eq!(choice.choice, "sad");
    for (label, want) in [("angry", 0.0622), ("calm", 0.0533), ("sad", 0.8845)] {
        let got = choice.probabilities[label];
        assert!((got - want).abs() < 1e-3, "{label}: {got} vs {want}");
    }
    let noul = output.answers.noul("refund").unwrap();
    assert!((noul.noul - 0.7749).abs() < 1e-3, "{}", noul.noul);
    let score = output.answers.score("urgency").unwrap();
    assert!((score.score - 1.5683).abs() < 1e-3, "{}", score.score);
    assert!((score.probabilities["2"] - 0.6099).abs() < 1e-3);
}
