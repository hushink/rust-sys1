use indexmap::IndexMap;
use serde_json::Value;
use std::fs;
use std::path::{Path, PathBuf};
use sys1_core::{DecisionModel, Questions};
use sys1_laya::{ConfidenceMode, LayaModel, LayaOptions};
use sys1_protocol::{Question, State};

fn env_path(name: &str) -> PathBuf {
    std::env::var(name)
        .map(PathBuf::from)
        .unwrap_or_else(|_| panic!("set {name}"))
}

fn assert_close(actual: f64, expected: f64, label: &str) {
    assert!(
        (actual - expected).abs() <= 1e-3,
        "{label}: expected {expected:.4}, got {actual:.4}"
    );
}

fn expected_value(value: &Value, key: &str) -> f64 {
    value[key]
        .as_f64()
        .unwrap_or_else(|| panic!("{key} must be a number"))
}

fn run_checkpoint(golden: &Path, checkpoint: &Path) -> usize {
    let case: Value = serde_json::from_slice(
        &fs::read(golden.join("case_dict_state.json")).expect("golden case"),
    )
    .expect("golden JSON");

    let state = State::from_value(case["state"].clone()).expect("state");
    let questions: IndexMap<String, Question> =
        serde_json::from_value(case["questions"].clone()).expect("questions");
    let questions = Questions::from(questions);

    let model = LayaModel::load_dir(
        checkpoint,
        LayaOptions {
            confidence_mode: ConfidenceMode::LayaNative,
            ..LayaOptions::default()
        },
    )
    .expect("load Laya checkpoint");
    let output = model.predict(&state, &questions).expect("predict");
    assert_eq!(
        output.usage.input_tokens,
        case["result"]["usage"]["input_tokens"]
            .as_u64()
            .expect("expected input_tokens")
    );

    let expected_answers = case["result"]["answers"]
        .as_object()
        .expect("expected answers");
    let mut checked = 0_usize;
    for (id, expected) in expected_answers {
        match expected["type"].as_str().expect("answer type") {
            "choice" => {
                let actual = output.answers.choice(id).expect("choice answer");
                assert_eq!(
                    actual.choice,
                    expected["choice"].as_str().expect("expected choice")
                );
                assert_close(
                    actual.confidence,
                    expected_value(expected, "confidence"),
                    &format!("{id}.confidence"),
                );
                for (label, probability) in expected["probabilities"]
                    .as_object()
                    .expect("expected probabilities")
                {
                    assert_close(
                        actual.probabilities[label],
                        probability.as_f64().expect("probability"),
                        &format!("{id}.probabilities.{label}"),
                    );
                }
            }
            "score" => {
                let actual = output.answers.score(id).expect("score answer");
                assert_close(
                    actual.score,
                    expected_value(expected, "score"),
                    &format!("{id}.score"),
                );
                assert_close(
                    actual.confidence,
                    expected_value(expected, "confidence"),
                    &format!("{id}.confidence"),
                );
                for (level, probability) in expected["probabilities"]
                    .as_object()
                    .expect("expected probabilities")
                {
                    assert_close(
                        actual.probabilities[level],
                        probability.as_f64().expect("probability"),
                        &format!("{id}.probabilities.{level}"),
                    );
                }
                for (level, legend) in expected["legend"].as_object().expect("expected legend") {
                    let actual_legend =
                        serde_json::to_value(&actual.legend[level]).expect("legend serialization");
                    assert_eq!(&actual_legend, legend);
                }
            }
            "noul" => {
                let actual = output.answers.noul(id).expect("noul answer");
                assert_close(
                    actual.noul,
                    expected_value(expected, "noul"),
                    &format!("{id}.noul"),
                );
            }
            other => panic!("unexpected answer type {other}"),
        }
        checked += 1;
    }
    checked
}

#[test]
#[ignore = "requires the cached Laya checkpoints and official golden vectors"]
fn matches_official_golden_answers() {
    let root = env_path("SYS1_LAYA_TEST_ROOT");
    let golden_root = env_path("SYS1_LAYA_GOLDEN_DIR");

    let checkpoints = [
        ("english", root.clone()),
        ("multilingual", root.join("multilingual")),
        ("typed-decisions", root.join("typed-decisions")),
    ];
    for (variant, checkpoint) in checkpoints {
        let checked = run_checkpoint(&golden_root.join(variant), &checkpoint);
        assert!(checked > 0, "{variant} produced no answers");
    }
}
