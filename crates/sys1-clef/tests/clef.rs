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

/// Logit-level parity with the authors' `joint_schema_model.py` in BF16.
///
/// The token row is byte-identical to upstream's `encode_record`; the
/// remaining spread (≤0.04 here) is BF16 kernel-order noise across candle and
/// PyTorch, not layout drift. A wrong span, option order or head would move
/// these probabilities far more.
#[test]
#[ignore = "requires SYS1_CLEF_TEST_DIR and about 19 GB RAM; runs BF16 on CPU"]
fn matches_upstream_bf16_probabilities() {
    let directory = PathBuf::from(std::env::var("SYS1_CLEF_TEST_DIR").expect("SYS1_CLEF_TEST_DIR"));
    let model = ClefModel::load_dir(
        &directory,
        ClefOptions {
            dtype: DType::BF16,
            ..ClefOptions::default()
        },
    )
    .unwrap();
    let state = State::text(
        "I was charged twice for order A-104 last week. Refund the duplicate today or I'm \
         cancelling my plan.",
    );

    let team = Questions::new().choice(
        "team",
        [
            ("billing", "Payments, charges, invoices, refunds"),
            ("technical", "Bugs, outages, errors"),
            ("sales", "Plans, upgrades, pricing questions"),
        ],
    );
    let output = model.predict(&state, &team).unwrap();
    let answer = output.answers.choice("team").unwrap();
    assert_eq!(answer.choice, "billing");
    for (label, want) in [
        ("billing", 0.956492),
        ("sales", 0.025291),
        ("technical", 0.018217),
    ] {
        let got = answer.probabilities[label];
        assert!((got - want).abs() < 0.05, "{label}: {got} vs {want}");
    }

    let triage = Questions::new()
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
    let output = model.predict(&state, &triage).unwrap();
    for (id, want) in [
        ("refund", 0.787278),
        ("urgent", 0.951051),
        ("churn", 0.904313),
    ] {
        let got = output.answers.noul(id).unwrap().noul;
        assert!((got - want).abs() < 0.05, "{id}: {got} vs {want}");
    }
    let answer = output.answers.choice("team").unwrap();
    assert_eq!(answer.choice, "billing");
    for (label, want) in [
        ("billing", 0.957842),
        ("sales", 0.021079),
        ("technical", 0.021079),
    ] {
        let got = answer.probabilities[label];
        assert!((got - want).abs() < 0.05, "{label}: {got} vs {want}");
    }
    let upset = output.answers.score("upset").unwrap();
    for (index, want) in [(0, 0.009724), (1, 0.013084), (2, 0.262807), (3, 0.714385)] {
        let got = upset.probabilities[&index.to_string()];
        assert!((got - want).abs() < 0.05, "{index}: {got} vs {want}");
    }
    assert_eq!(upset.score.round() as usize, 3);
}
