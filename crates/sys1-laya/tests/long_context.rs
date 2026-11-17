use candle_core::{DType, Device};
use sys1_core::{DecisionModel, Questions};
use sys1_laya::{ConfidenceMode, LayaModel, LayaOptions};
use sys1_protocol::State;

#[test]
#[ignore = "requires a cached Laya checkpoint and CUDA"]
fn exact_local_window_matches_eager_on_long_state() {
    let root = std::env::var("SYS1_LAYA_TEST_ROOT")
        .expect("set SYS1_LAYA_TEST_ROOT to the multilingual checkpoint directory");
    let root = std::path::PathBuf::from(root);
    let device = Device::new_cuda(0).expect("CUDA device 0");

    let load = |local_window| {
        LayaModel::load_dir(
            &root,
            LayaOptions {
                device: device.clone(),
                dtype: DType::F16,
                max_len: Some(1024),
                confidence_mode: ConfidenceMode::TypeSafe,
                local_window,
                max_attention_elements: sys1_laya::DEFAULT_MAX_ATTENTION_ELEMENTS,
                ..LayaOptions::default()
            },
        )
        .expect("load Laya checkpoint")
    };
    let eager = load(false);
    let windowed = load(true);

    let sentence = "I was charged twice for invoice 12345 and need a refund today. ";
    let state = State::text(sentence.repeat(50));
    let questions = Questions::new()
        .noul("billing", "Is this about billing?")
        .score("urgency", ["Can wait", "This week", "Today"])
        .choice("tone", [("angry", "Upset"), ("calm", "Neutral")]);

    // Warm up both paths.
    let _ = eager.predict(&state, &questions).expect("eager warmup");
    let _ = windowed.predict(&state, &questions).expect("window warmup");

    let eager = eager.predict(&state, &questions).expect("eager prediction");
    let windowed = windowed
        .predict(&state, &questions)
        .expect("window prediction");
    assert!(eager.usage.input_tokens >= 512);

    let base_noul = eager.answers.noul("billing").expect("eager noul").noul;
    let win_noul = windowed.answers.noul("billing").expect("window noul").noul;
    assert!((base_noul - win_noul).abs() <= 2e-3);

    let base_choice = eager.answers.choice("tone").expect("eager choice");
    let win_choice = windowed.answers.choice("tone").expect("window choice");
    assert_eq!(base_choice.choice, win_choice.choice);
    for (label, probability) in &base_choice.probabilities {
        let other = win_choice
            .probabilities
            .get(label)
            .copied()
            .expect("same labels");
        assert!((probability - other).abs() <= 2e-3, "{label}");
    }

    let base_score = eager.answers.score("urgency").expect("eager score");
    let win_score = windowed.answers.score("urgency").expect("window score");
    assert!((base_score.score - win_score.score).abs() <= 2e-3);
    for (label, probability) in &base_score.probabilities {
        let other = win_score
            .probabilities
            .get(label)
            .copied()
            .expect("same labels");
        assert!((probability - other).abs() <= 2e-3, "{label}");
    }
}
