use std::path::PathBuf;
use sys1_core::{DecisionModel, LongContextOptions, Questions};
use sys1_laya::{ConfidenceMode, LayaModel, LayaOptions};
use sys1_protocol::State;

fn load() -> LayaModel {
    let root = PathBuf::from(
        std::env::var("SYS1_LAYA_TEST_ROOT")
            .expect("set SYS1_LAYA_TEST_ROOT to the English checkpoint directory"),
    );
    LayaModel::load_dir(
        &root,
        LayaOptions {
            confidence_mode: ConfidenceMode::LayaNative,
            ..LayaOptions::default()
        },
    )
    .expect("load Laya checkpoint")
}

#[test]
#[ignore = "requires a cached Laya checkpoint"]
fn scans_a_long_state_and_attributes_every_answer() {
    let model = load();
    let questions = Questions::new()
        .noul("billing", "Is this about billing?")
        .choice("tone", [("angry", "Upset"), ("calm", "Neutral")]);

    let state =
        State::text("I was charged twice for invoice 12345 and need a refund today. ".repeat(20));
    let output = model
        .predict_long(
            &state,
            &questions,
            LongContextOptions {
                window: Some(96),
                stride: Some(48),
                batch_size: Some(8),
            },
        )
        .expect("predict_long");

    assert_eq!(output.scored_windows, 5);
    assert_eq!(output.answers.len(), 2);
    let billing = output.answers.noul("billing").expect("billing");
    assert!((billing.noul - 0.8811).abs() <= 1e-3, "{}", billing.noul);
    let tone = output.answers.choice("tone").expect("tone");
    assert_eq!(tone.choice, "calm");
    assert!((tone.probabilities["angry"] - 0.1058).abs() <= 1e-3);
    assert!((tone.probabilities["calm"] - 0.8942).abs() <= 1e-3);
    assert!((tone.confidence - 0.5129).abs() <= 1e-3);
    assert_eq!(output.action("billing"), Some(1.0));
    assert_eq!(output.action("tone"), Some(1.0));
    let billing_window = output.window("billing").expect("billing window");
    assert_eq!(billing_window.index, 0);
    assert_eq!(billing_window.token_start, 0);
    assert_eq!(billing_window.token_end, 96);
    assert_eq!(billing_window.count, 5);
    let tone_window = output.window("tone").expect("tone window");
    assert_eq!(tone_window.index, 2);
    assert_eq!(tone_window.token_start, 96);
    assert_eq!(tone_window.token_end, 192);
    assert_eq!(tone_window.count, 5);
    assert_eq!(output.usage.input_tokens, 1196);

    let short = State::text("I was charged twice and need a refund today.");
    let plain = model.predict(&short, &questions).expect("predict");
    let single = model
        .predict_long(&short, &questions, LongContextOptions::default())
        .expect("predict_long");
    assert_eq!(single.scored_windows, 1);
    assert!(single.windows.is_empty());
    assert_eq!(single.answers, plain.answers);
    assert_eq!(single.usage, plain.usage);

    let error = model
        .predict_long(
            &State::text("refund ".repeat(1000)),
            &Questions::new().noul("billing", "Is this about billing?"),
            LongContextOptions {
                window: Some(64),
                stride: Some(65),
                batch_size: None,
            },
        )
        .expect_err("stride must not skip tokens");
    assert!(error.to_string().contains("stride"));
}
