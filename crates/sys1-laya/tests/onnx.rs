#![cfg(feature = "onnx")]

use candle_core::{DType, Device, Tensor};
use indexmap::IndexMap;
use serde_json::json;
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use sys1_core::{DecisionModel, ModelFactory, Prediction, Questions};
use sys1_laya::{
    LayaFactory, LayaOnnxModel, LayaOptions, is_checkpoint_directory, is_onnx_directory,
    local_checkpoint_manifest, register_checkpoint_bundle,
};
use sys1_protocol::State;
use tempfile::TempDir;

const TOKENIZER: &str = r#"{
  "version": "1.0",
  "truncation": null,
  "padding": null,
  "added_tokens": [],
  "normalizer": null,
  "pre_tokenizer": {"type": "Whitespace"},
  "post_processor": null,
  "decoder": null,
  "model": {
    "type": "WordLevel",
    "vocab": {
      "[PAD]": 0,
      "[CLS]": 1,
      "[SEP]": 2,
      "[MASK]": 3,
      "hello": 4,
      "world": 5,
      "choice": 6,
      "score": 7,
      "noul": 8,
      "question:": 9,
      "tone": 10,
      "calm:": 11,
      "neutral": 12,
      "angry:": 13,
      "upset": 14,
      "urgency": 15,
      "level": 16,
      "0:": 17,
      "low": 18,
      "1:": 19,
      "high": 20,
      "billing": 21,
      "false:": 22,
      "true:": 23
    },
    "unk_token": "[PAD]"
  }
}"#;

const AGENT_CONFIG: &str = r#"{
  "max_len": 128,
  "head_max_len": 64,
  "temperature": [1.0, 1.0, 1.0]
}"#;

fn fixture() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../sys1-ort/tests/fixtures/scoring_stub.onnx")
}

fn model_directory() -> TempDir {
    let directory = tempfile::tempdir().expect("temp model directory");
    fs::copy(fixture(), directory.path().join("model.onnx")).expect("copy ONNX fixture");
    fs::write(directory.path().join("tokenizer.json"), TOKENIZER).expect("tokenizer");
    fs::write(directory.path().join("rl_agent_config.json"), AGENT_CONFIG).expect("agent config");
    fs::write(directory.path().join("onnx.json"), r#"{"max_batch": 1}"#).expect("onnx sidecar");
    directory
}

fn write_act_head(directory: &Path) {
    let device = Device::Cpu;
    let mut input = vec![0.0_f32; 256 * 5];
    input[0] = 1.0;
    let mut output = vec![0.0_f32; 2 * 256];
    output[0] = 1.0;
    output[256] = -1.0;
    let tensors = HashMap::from([
        (
            "act_head.0.weight".to_owned(),
            Tensor::from_vec(input, (256, 5), &device).expect("act input"),
        ),
        (
            "act_head.0.bias".to_owned(),
            Tensor::zeros(256, DType::F32, &device).expect("act input bias"),
        ),
        (
            "act_head.2.weight".to_owned(),
            Tensor::from_vec(output, (2, 256), &device).expect("act output"),
        ),
        (
            "act_head.2.bias".to_owned(),
            Tensor::zeros(2, DType::F32, &device).expect("act output bias"),
        ),
    ]);
    candle_core::safetensors::save(&tensors, directory.join("act_head.safetensors"))
        .expect("save act head");
}

fn load(directory: &Path) -> Option<LayaOnnxModel> {
    match LayaOnnxModel::load_dir(directory, LayaOptions::default()) {
        Ok(model) => Some(model),
        Err(error) if std::env::var_os("SYS1_REQUIRE_ORT").is_none() => {
            eprintln!("skipping ONNX Runtime test (set SYS1_REQUIRE_ORT=1 to require it): {error}");
            None
        }
        Err(error) => panic!("required ONNX Runtime test failed to load: {error}"),
    }
}

fn question_set() -> Questions {
    Questions::new()
        .choice("tone", [("calm", "neutral"), ("angry", "upset")])
        .score("urgency", ["low", "high"])
        .noul("billing", "Is this about billing?")
}

#[test]
fn predicts_every_question_type_through_onnx() {
    let directory = model_directory();
    let Some(model) = load(directory.path()) else {
        return;
    };
    let state = State::from_value(json!("hello world")).expect("state");
    let output = model.predict(&state, &question_set()).expect("predict");

    assert_eq!(output.answers.len(), 3);
    assert!(output.usage.input_tokens > 0);

    let choice = output.answers.choice("tone").expect("choice answer");
    assert_eq!(choice.choice, "calm");
    assert_eq!(choice.confidence, 0.0);
    assert_eq!(choice.probabilities["calm"], 0.5);
    assert_eq!(choice.probabilities["angry"], 0.5);

    let score = output.answers.score("urgency").expect("score answer");
    assert_eq!(score.score, 0.5);
    assert_eq!(score.confidence, 0.0);

    let noul = output.answers.noul("billing").expect("noul answer");
    assert_eq!(noul.noul, 0.5);
}

#[test]
fn act_head_sidecar_enables_actions() {
    let directory = model_directory();
    write_act_head(directory.path());
    let Some(model) = load(directory.path()) else {
        return;
    };
    let state = State::from_value(json!("hello world")).expect("state");
    let output = model.predict(&state, &question_set()).expect("predict");

    for id in ["tone", "urgency", "billing"] {
        let action = output
            .action(id)
            .unwrap_or_else(|| panic!("missing action for {id}"));
        assert!((0.0..=1.0).contains(&action), "{id} action: {action}");
        assert!(action > 0.5, "{id} action should favor class 0: {action}");
    }
}

#[test]
fn batches_with_per_item_chunking() {
    let directory = model_directory();
    let Some(model) = load(directory.path()) else {
        return;
    };
    let state = State::from_value(json!("hello world")).expect("state");
    let first = question_set();
    let second = Questions::new().noul("billing", "Is this about billing?");
    let outputs = model.predict_batch(&[
        Prediction::new(&state, &first),
        Prediction::new(&state, &second),
    ]);

    assert_eq!(outputs.len(), 2);
    assert_eq!(outputs[0].as_ref().expect("first").answers.len(), 3);
    assert_eq!(outputs[1].as_ref().expect("second").answers.len(), 1);
}

#[test]
fn bundle_detection_and_factory_load_onnx() {
    let directory = model_directory();
    assert!(is_onnx_directory(directory.path()));
    assert!(!is_checkpoint_directory(directory.path()));

    let (manifest, files) = local_checkpoint_manifest(directory.path()).expect("manifest");
    assert_eq!(manifest.backend, "onnx");
    assert_eq!(manifest.format, "onnx");
    assert!(files.contains_key("model.onnx"));
    assert!(files.contains_key("onnx.json"));

    let factory = LayaFactory::new();
    let manifests = register_checkpoint_bundle(
        &factory,
        &manifest,
        directory.path(),
        LayaOptions::default(),
        Some("onnx-stub"),
    )
    .expect("register ONNX model");
    assert_eq!(manifests.len(), 1);
    assert_eq!(manifests[0].name, "onnx-stub");

    let loaded = match factory.load(&manifests[0]) {
        Ok(model) => model,
        Err(error) if std::env::var_os("SYS1_REQUIRE_ORT").is_none() => {
            eprintln!("skipping ONNX Runtime test: {error}");
            return;
        }
        Err(error) => panic!("required ONNX Runtime test failed to load: {error}"),
    };
    let state = State::from_value(json!("hello world")).expect("state");
    let output = loaded
        .predict(
            &state,
            &Questions::new().noul("billing", "Is this about billing?"),
        )
        .expect("predict");
    assert_eq!(output.answers.len(), 1);
}

#[test]
fn sidecar_cannot_escape_the_model_directory() {
    let directory = model_directory();
    fs::write(
        directory.path().join("onnx.json"),
        r#"{"model": "../outside.onnx"}"#,
    )
    .expect("invalid sidecar");
    let error = match LayaOnnxModel::load_dir(directory.path(), LayaOptions::default()) {
        Ok(_) => panic!("must reject an escaping model path"),
        Err(error) => error,
    };
    assert!(
        error.to_string().contains("inside the model directory"),
        "{error}"
    );
}

#[test]
fn answers_are_typed_and_indexed_by_question_id() {
    let directory = model_directory();
    let Some(model) = load(directory.path()) else {
        return;
    };
    let state = State::from_value(json!({"text": "hello world"})).expect("state");
    let questions = question_set();
    let output = model.predict(&state, &questions).expect("predict");
    let map: IndexMap<_, _> = output.answers.into_map();
    assert_eq!(
        map.keys().cloned().collect::<Vec<_>>(),
        vec!["tone", "urgency", "billing"]
    );
}
