#![cfg(feature = "onnx")]

use indexmap::IndexMap;
use serde_json::Value;
use std::fs;
use std::path::{Path, PathBuf};
use sys1_core::{Answers, DecisionModel, Questions};
use sys1_laya::{ConfidenceMode, LayaModel, LayaOnnxModel, LayaOptions};
use sys1_protocol::{Answer, Question, State};

fn env_path(name: &str) -> PathBuf {
    std::env::var(name)
        .map(PathBuf::from)
        .unwrap_or_else(|_| panic!("set {name}"))
}

fn assert_close(actual: f64, expected: f64, label: &str) {
    assert!(
        (actual - expected).abs() <= 5e-4,
        "{label}: expected {expected:.6}, got {actual:.6}"
    );
}

fn compare_answers(candle: &Answers, onnx: &Answers) {
    assert_eq!(candle.len(), onnx.len(), "answer count");
    for (id, expected) in candle.iter() {
        let actual = onnx.get(id).unwrap_or_else(|| panic!("missing {id}"));
        match (expected, actual) {
            (Answer::Choice(expected), Answer::Choice(actual)) => {
                assert_eq!(expected.choice, actual.choice, "{id}.choice");
                assert_close(
                    actual.confidence,
                    expected.confidence,
                    &format!("{id}.confidence"),
                );
                for (label, value) in &expected.probabilities {
                    assert_close(
                        actual.probabilities[label],
                        *value,
                        &format!("{id}.probabilities.{label}"),
                    );
                }
            }
            (Answer::Score(expected), Answer::Score(actual)) => {
                assert_close(actual.score, expected.score, &format!("{id}.score"));
                assert_close(
                    actual.confidence,
                    expected.confidence,
                    &format!("{id}.confidence"),
                );
                assert_eq!(actual.legend, expected.legend, "{id}.legend");
                for (level, value) in &expected.probabilities {
                    assert_close(
                        actual.probabilities[level],
                        *value,
                        &format!("{id}.probabilities.{level}"),
                    );
                }
            }
            (Answer::Noul(expected), Answer::Noul(actual)) => {
                assert_close(actual.noul, expected.noul, &format!("{id}.noul"));
            }
            _ => panic!("answer type changed for {id}"),
        }
    }
}

fn case_file(golden: &Path) -> PathBuf {
    let direct = golden.join("case_dict_state.json");
    if direct.is_file() {
        direct
    } else {
        golden.join("multilingual").join("case_dict_state.json")
    }
}

#[test]
#[ignore = "requires SYS1_LAYA_TEST_ROOT, SYS1_LAYA_ONNX_DIR, and SYS1_LAYA_GOLDEN_DIR"]
fn candle_and_onnx_models_produce_the_same_answers() {
    let checkpoint = env_path("SYS1_LAYA_TEST_ROOT");
    let onnx_directory = env_path("SYS1_LAYA_ONNX_DIR");
    let golden = env_path("SYS1_LAYA_GOLDEN_DIR");
    let case: Value =
        serde_json::from_slice(&fs::read(case_file(&golden)).expect("golden case_dict_state.json"))
            .expect("golden JSON");

    let state = State::from_value(case["state"].clone()).expect("state");
    let questions: IndexMap<String, Question> =
        serde_json::from_value(case["questions"].clone()).expect("questions");
    let questions = Questions::from(questions);

    let options = LayaOptions {
        confidence_mode: ConfidenceMode::LayaNative,
        ..LayaOptions::default()
    };
    eprintln!("loading ONNX model");
    let onnx = LayaOnnxModel::load_dir(&onnx_directory, options.clone()).expect("load ONNX model");
    eprintln!("loading candle model");
    let candle = LayaModel::load_dir(&checkpoint, options).expect("load candle model");

    eprintln!("running ONNX");
    let onnx_output = onnx.predict(&state, &questions).expect("ONNX predict");
    eprintln!("running candle");
    let candle_output = candle.predict(&state, &questions).expect("candle predict");
    assert_eq!(
        candle_output.usage.input_tokens, onnx_output.usage.input_tokens,
        "input token count"
    );
    compare_answers(&candle_output.answers, &onnx_output.answers);
    for (id, _) in candle_output.answers.iter() {
        let expected = candle_output
            .action(id)
            .unwrap_or_else(|| panic!("candle produced no action for {id}"));
        let actual = onnx_output
            .action(id)
            .unwrap_or_else(|| panic!("ONNX produced no action for {id}"));
        assert!(
            (actual - expected).abs() <= 2e-3,
            "{id}.action: expected {expected}, got {actual}"
        );
    }
}
