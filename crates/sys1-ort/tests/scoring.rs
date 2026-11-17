use std::path::Path;
use sys1_ort::{ScoringInputs, ScoringSession, SessionConfig};

fn fixture() -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/scoring_stub.onnx")
}

fn load_with(pooled: bool) -> Option<ScoringSession> {
    let mut config = SessionConfig::new(fixture());
    if pooled {
        config.pooled_output = Some("pooled".to_owned());
    }
    match ScoringSession::load(config) {
        Ok(session) => Some(session),
        Err(error) if std::env::var_os("SYS1_REQUIRE_ORT").is_none() => {
            eprintln!("skipping ONNX Runtime test (set SYS1_REQUIRE_ORT=1 to require it): {error}");
            None
        }
        Err(error) => panic!("required ONNX Runtime test failed to load: {error}"),
    }
}

fn load() -> Option<ScoringSession> {
    load_with(false)
}

#[test]
fn scores_markers_across_a_padded_batch() {
    let Some(session) = load() else {
        return;
    };

    let input_ids = [10_i64, 11, 12, 0, 20, 21, 22, 23];
    let attention_mask = [1_i64, 1, 1, 0, 1, 1, 1, 1];
    let type_ids = [0_i64, 1];
    // The stub returns the token id at each flattened marker position.
    let outputs = session
        .score(ScoringInputs {
            input_ids: &input_ids,
            attention_mask: &attention_mask,
            type_ids: &type_ids,
            markers: &[1, 6, 7],
            batch: 2,
            seq: 4,
        })
        .expect("score");

    assert_eq!(outputs.logits, vec![11.0, 22.0, 23.0]);
    assert!(outputs.pooled.is_none());
}

#[test]
fn extracts_the_optional_pooled_output() {
    let Some(session) = load_with(true) else {
        return;
    };
    let input_ids = [10_i64, 11, 12, 0, 20, 21, 22, 23];
    let outputs = session
        .score(ScoringInputs {
            input_ids: &input_ids,
            attention_mask: &[1, 1, 1, 0, 1, 1, 1, 1],
            type_ids: &[0, 1],
            markers: &[1, 6],
            batch: 2,
            seq: 4,
        })
        .expect("score");
    assert_eq!(outputs.logits, vec![11.0, 22.0]);
    assert_eq!(outputs.pooled.as_deref(), Some(&[10.0, 20.0][..]));
}

#[test]
fn returns_no_logits_for_no_markers() {
    let Some(session) = load() else {
        return;
    };
    let outputs = session
        .score(ScoringInputs {
            input_ids: &[1, 2],
            attention_mask: &[1, 1],
            type_ids: &[0],
            markers: &[],
            batch: 1,
            seq: 2,
        })
        .expect("score");
    assert!(outputs.logits.is_empty());
    assert!(outputs.pooled.is_none());
}

#[test]
fn rejects_out_of_bounds_markers_before_running() {
    let Some(session) = load() else {
        return;
    };
    let error = session
        .score(ScoringInputs {
            input_ids: &[1, 2],
            attention_mask: &[1, 1],
            type_ids: &[0],
            markers: &[2],
            batch: 1,
            seq: 2,
        })
        .expect_err("marker is out of bounds");
    assert!(error.to_string().contains("outside 0..2"), "{error}");
}
