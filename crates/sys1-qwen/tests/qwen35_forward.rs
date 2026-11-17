use candle_core::{DType, Device, Tensor};
use std::path::PathBuf;
use sys1_qwen::{Qwen35, Qwen35Config, WeightStore, detect_prefix, safetensors_files};

#[test]
#[ignore = "requires SYS1_QWEN35_TEST_DIR pointing at a Qwen3.5 checkpoint"]
fn matches_transformers_on_fixed_tokens() {
    let directory = PathBuf::from(std::env::var("SYS1_QWEN35_TEST_DIR").unwrap());
    let mut config = Qwen35Config::load(&directory.join("config.json")).unwrap();
    let paths = safetensors_files(&directory).unwrap();
    let store = WeightStore::from_safetensors(&paths, DType::F32, &Device::Cpu).unwrap();
    config.prefix = detect_prefix(
        &store,
        &["model.language_model.", "model.", "language_model.", ""],
        "embed_tokens.weight",
    )
    .unwrap();
    let model = Qwen35::load(config, &store).unwrap();
    let cases: &[(&[u32], [f32; 8])] = &[
        (
            &[151644, 872],
            [
                -2.403987, -0.660043, 8.176854, 20.762054, -1.224812, 7.947424, 14.156066,
                -5.513241,
            ],
        ),
        (
            &[151644],
            [
                0.130463, -1.442123, -3.776932, 10.575282, -2.181299, -1.206551, 0.840216,
                -3.499681,
            ],
        ),
    ];
    for (ids, expected) in cases {
        let input = Tensor::from_vec(ids.to_vec(), (1, ids.len()), &Device::Cpu).unwrap();
        let hidden = model.forward(&input).unwrap().to_dtype(DType::F32).unwrap();
        let position = if *ids == [151644, 872] { 1 } else { 0 };
        let row = hidden
            .narrow(1, position, 1)
            .unwrap()
            .squeeze(1)
            .unwrap()
            .to_vec2::<f32>()
            .unwrap();
        for (index, want) in expected.iter().enumerate() {
            let got = row[0][index];
            assert!(
                (got - want).abs() < 1e-3,
                "ids {ids:?} index {index}: got {got}, want {want}"
            );
        }
    }

    let mut ids = vec![151644u32];
    ids.extend(std::iter::repeat_n(872u32, 129));
    let input = Tensor::from_vec(ids.clone(), (1, ids.len()), &Device::Cpu).unwrap();
    let hidden = model.forward(&input).unwrap().to_dtype(DType::F32).unwrap();
    let row = hidden
        .narrow(1, ids.len() - 1, 1)
        .unwrap()
        .squeeze(1)
        .unwrap()
        .to_vec2::<f32>()
        .unwrap();
    let expected = [
        -3.180935, 1.613202, -0.468953, 22.096647, 0.086789, 3.426143, -3.120995, -6.709355,
    ];
    for (index, want) in expected.iter().enumerate() {
        let got = row[0][index];
        assert!(
            (got - want).abs() < 1e-3,
            "long row index {index}: got {got}, want {want}"
        );
    }
}
