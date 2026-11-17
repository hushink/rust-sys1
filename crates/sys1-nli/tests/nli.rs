use std::path::PathBuf;
use sys1_core::{DecisionModel, Questions};
use sys1_nli::{NliModel, NliOptions};
use sys1_protocol::State;

#[test]
#[ignore = "requires a local NLI ModernBERT checkpoint directory"]
fn predicts_all_question_types() {
    let directory = PathBuf::from(std::env::var("SYS1_NLI_TEST_DIR").expect("SYS1_NLI_TEST_DIR"));
    let model = NliModel::load_dir(&directory, NliOptions::default()).expect("load NLI model");
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
    assert!((choice.probabilities["angry"] - 0.4264528).abs() <= 1e-4);
    assert!((choice.probabilities["calm"] - 0.5735472).abs() <= 1e-4);

    let score = output.answers.score("urgency").expect("score");
    assert!((score.probabilities["0"] - 0.01239845).abs() <= 1e-4);
    assert!((score.probabilities["1"] - 0.09954133).abs() <= 1e-4);
    assert!((score.probabilities["2"] - 0.88806022).abs() <= 1e-4);
    assert!((score.score - 1.8756618).abs() <= 1e-4);

    let noul = output.answers.noul("billing").expect("noul");
    assert!((noul.noul - 0.5526355).abs() <= 1e-4);
}
