use std::path::PathBuf;
use sys1_core::{DecisionModel, Questions};
use sys1_kev::{KevModel, KevOptions};
use sys1_protocol::State;

#[test]
#[ignore = "requires SYS1_KEV_TEST_DIR and SYS1_KEV_BASE_DIR"]
fn predicts_all_question_types() {
    let directory = PathBuf::from(std::env::var("SYS1_KEV_TEST_DIR").expect("SYS1_KEV_TEST_DIR"));
    let base = PathBuf::from(std::env::var("SYS1_KEV_BASE_DIR").expect("SYS1_KEV_BASE_DIR"));
    let model = KevModel::load_dir(
        &directory,
        KevOptions {
            base_directory: Some(base),
            ..KevOptions::default()
        },
    )
    .unwrap();
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
    let team = output.answers.choice("team").unwrap();
    assert_eq!(team.choice, "billing");
    for (label, want) in [("billing", 0.929058), ("tech", 0.070942)] {
        let got = team.probabilities[label];
        assert!((got - want).abs() < 1e-3, "{label}: {got} vs {want}");
    }
    let refund = output.answers.noul("refund").unwrap();
    assert!((refund.noul - 0.826006).abs() < 1e-3, "{}", refund.noul);
    let urgency = output.answers.score("urgency").unwrap();
    assert!((urgency.score - 1.223002).abs() < 1e-3, "{}", urgency.score);
}
