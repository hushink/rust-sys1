use std::path::PathBuf;
use sys1_core::{AbstentionState, DecisionModel, MinConfidence, PredictionOptions, Questions};
use sys1_laya::{LayaModel, LayaOptions};
use sys1_protocol::State;

fn load() -> LayaModel {
    let root = PathBuf::from(
        std::env::var("SYS1_LAYA_TEST_ROOT")
            .expect("set SYS1_LAYA_TEST_ROOT to the English checkpoint directory"),
    );
    LayaModel::load_dir(&root, LayaOptions::default()).expect("load Laya checkpoint")
}

#[test]
#[ignore = "requires a cached Laya checkpoint"]
fn matches_the_official_scalar_and_bucket_gate() {
    let model = load();
    let state = State::text("I was charged twice for invoice 12345 and need a refund today.");
    let questions = Questions::new()
        .noul("billing", "Is this about billing?")
        .choice("tone", [("angry", "Upset"), ("calm", "Neutral")]);

    let scalar = PredictionOptions {
        min_confidence: Some(MinConfidence::scalar(0.9).expect("scalar")),
    };
    let output = model
        .predict_with(&state, &questions, &scalar)
        .expect("predict_with");
    let billing = output.abstention("billing").expect("billing gate");
    assert_eq!(billing.state, AbstentionState::Abstained);
    assert!(billing.low_confidence);
    assert!((billing.threshold - 0.9).abs() < f64::EPSILON);
    let tone = output.abstention("tone").expect("tone gate");
    assert_eq!(tone.state, AbstentionState::Passed);
    assert!(!tone.low_confidence);
    assert!((output.answers.noul("billing").expect("noul").noul - 0.8564).abs() <= 1e-3);
    assert!(
        (output.answers.choice("tone").expect("choice").probabilities["angry"] - 0.9066).abs()
            <= 1e-3
    );

    let buckets = PredictionOptions {
        min_confidence: Some(
            MinConfidence::buckets(indexmap::IndexMap::from([
                ("choice:2".to_owned(), 0.95),
                ("default".to_owned(), 0.8),
            ]))
            .expect("buckets"),
        ),
    };
    let output = model
        .predict_with(&state, &questions, &buckets)
        .expect("predict_with");
    let billing = output.abstention("billing").expect("billing gate");
    assert_eq!(billing.state, AbstentionState::Passed);
    assert!((billing.threshold - 0.8).abs() < f64::EPSILON);
    let tone = output.abstention("tone").expect("tone gate");
    assert_eq!(tone.state, AbstentionState::Abstained);
    assert!((tone.threshold - 0.95).abs() < f64::EPSILON);
}
