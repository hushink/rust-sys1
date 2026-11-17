use std::path::PathBuf;
use sys1_clef::{ClefModel, ClefOptions};
use sys1_core::{DecisionModel, Questions};
use sys1_protocol::State;

#[test]
#[ignore = "requires SYS1_CLEF_TEST_DIR pointing at a clef-flash checkpoint"]
fn predicts_all_question_types() {
    let directory = PathBuf::from(std::env::var("SYS1_CLEF_TEST_DIR").expect("SYS1_CLEF_TEST_DIR"));
    let model = ClefModel::load_dir(&directory, ClefOptions::default()).unwrap();
    let questions = Questions::new()
        .choice(
            "team",
            [("billing", "charges and refunds"), ("tech", "bugs")],
        )
        .noul("refund", "The customer asks for a refund.")
        .score("urgency", ["low", "medium", "high"]);
    let output = model
        .predict(
            &State::text("My card was charged twice and I want a refund today."),
            &questions,
        )
        .unwrap();
    assert_eq!(output.answers.len(), 3);
}
