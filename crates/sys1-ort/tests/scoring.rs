use std::path::Path;
use sys1_ort::{ScoringInputs, ScoringSession, SessionConfig};

fn fixture() -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/scoring_stub.onnx")
}

fn load() -> Option<ScoringSession> {
    match ScoringSession::load(SessionConfig::new(fixture())) {
        Ok(session) => Some(session),
        Err(error) if std::env::var_os("SYS1_REQUIRE_ORT").is_none() => {
            eprintln!("skipping ONNX Runtime test (set SYS1_REQUIRE_ORT=1 to require it): {error}");
            None
        }
        Err(error) => panic!("required ONNX Runtime test failed to load: {error}"),
    }
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
    let logits = session
        .score(ScoringInputs {
            input_ids: &input_ids,
            attention_mask: &attention_mask,
            type_ids: &type_ids,
            markers: &[1, 6, 7],
            batch: 2,
            seq: 4,
        })
        .expect("score");

    assert_eq!(logits, vec![11.0, 22.0, 23.0]);
}

#[test]
fn returns_no_logits_for_no_markers() {
    let Some(session) = load() else {
        return;
    };
    let logits = session
        .score(ScoringInputs {
            input_ids: &[1, 2],
            attention_mask: &[1, 1],
            type_ids: &[0],
            markers: &[],
            batch: 1,
            seq: 2,
        })
        .expect("score");
    assert!(logits.is_empty());
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
