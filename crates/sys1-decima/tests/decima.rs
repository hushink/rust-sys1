use std::path::PathBuf;
use sys1_core::{DecisionModel, Questions};
use sys1_decima::{DecimaModel, DecimaOptions};
use sys1_protocol::State;

#[test]
#[ignore = "requires SYS1_DECIMA_TEST_DIR pointing at a decima-base repository"]
fn predicts_all_question_types() {
    let directory =
        PathBuf::from(std::env::var("SYS1_DECIMA_TEST_DIR").expect("SYS1_DECIMA_TEST_DIR"));
    let model = DecimaModel::load_dir(&directory, DecimaOptions::default()).unwrap();
    let questions = Questions::new()
        .choice("team", [("billing", "charges, refunds"), ("tech", "bugs")])
        .noul("urgent", "The customer needs an answer today.")
        .score("anger", ["calm", "annoyed", "furious"]);
    let output = model
        .predict(&State::text("My card was charged twice"), &questions)
        .unwrap();
    let team = output.answers.choice("team").unwrap();
    assert_eq!(team.choice, "billing");
    for (label, want) in [("billing", 0.972121), ("tech", 0.027879)] {
        let got = team.probabilities[label];
        assert!((got - want).abs() < 1e-3, "{label}: {got} vs {want}");
    }
    let urgent = output.answers.noul("urgent").unwrap();
    assert!((urgent.noul - 0.644244).abs() < 1e-3, "{}", urgent.noul);
    let anger = output.answers.score("anger").unwrap();
    assert!((anger.score - 0.946362).abs() < 1e-3, "{}", anger.score);
}
