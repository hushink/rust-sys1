use candle_core::DType;
use std::path::PathBuf;
use sys1_clef::{ClefModel, ClefOptions};
use sys1_core::{DecisionModel, Questions};
use sys1_protocol::State;

fn dtype() -> DType {
    match std::env::var("SYS1_CLEF_DTYPE")
        .unwrap_or_else(|_| "f32".to_owned())
        .to_ascii_lowercase()
        .as_str()
    {
        "bf16" => DType::BF16,
        "f16" => DType::F16,
        _ => DType::F32,
    }
}

#[test]
#[ignore = "requires SYS1_CLEF_TEST_DIR pointing at a clef-flash checkpoint"]
fn predicts_all_question_types() {
    let directory = PathBuf::from(std::env::var("SYS1_CLEF_TEST_DIR").expect("SYS1_CLEF_TEST_DIR"));
    let options = ClefOptions {
        dtype: dtype(),
        ..ClefOptions::default()
    };
    let model = ClefModel::load_dir(&directory, options).unwrap();
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

#[test]
#[ignore = "requires SYS1_CLEF_TEST_DIR pointing at a clef-flash checkpoint"]
fn matches_the_recorded_triage_decisions() {
    let directory = PathBuf::from(std::env::var("SYS1_CLEF_TEST_DIR").expect("SYS1_CLEF_TEST_DIR"));
    let options = ClefOptions {
        dtype: dtype(),
        ..ClefOptions::default()
    };
    let model = ClefModel::load_dir(&directory, options).unwrap();
    let state = State::text(
        "I was charged twice for order A-104 last week. Refund the duplicate today or I'm \
         cancelling my plan.",
    );
    let questions = Questions::new()
        .noul("refund", "Is the customer asking for a refund?")
        .choice(
            "team",
            [
                ("billing", "Payments, charges, invoices, refunds"),
                ("technical", "Bugs, outages, errors"),
                ("sales", "Plans, upgrades, pricing questions"),
            ],
        )
        .score(
            "upset",
            [
                "calm",
                "mildly annoyed",
                "angry",
                "furious, threatening to leave",
            ],
        )
        .noul("urgent", "Does this need an answer today?")
        .noul("churn", "Is the customer likely to cancel?");
    let started = std::time::Instant::now();
    let output = model.predict(&state, &questions).unwrap();
    println!("clef predict: {:?}", started.elapsed());
    assert_eq!(output.answers.choice("team").unwrap().choice, "billing");
    for id in ["refund", "urgent", "churn"] {
        assert!(
            output.answers.noul(id).unwrap().noul > 0.5,
            "{id}: {:?}",
            output.answers.noul(id).unwrap()
        );
    }
    let upset = output.answers.score("upset").unwrap();
    let best = upset
        .probabilities
        .iter()
        .max_by(|left, right| left.1.partial_cmp(right.1).expect("probability"))
        .expect("levels")
        .0;
    assert_eq!(best.as_str(), "3", "{:?}", upset.probabilities);
}
