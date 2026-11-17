use candle_core::{DType, Device, Tensor};
use std::path::PathBuf;
use sys1_qwen::{Qwen3, Qwen3Config, WeightStore, detect_prefix, safetensors_files};

#[test]
#[ignore = "requires SYS1_QWEN3_TEST_DIR pointing at a Qwen3 checkpoint"]
fn matches_transformers_on_fixed_tokens() {
    let directory = PathBuf::from(std::env::var("SYS1_QWEN3_TEST_DIR").unwrap());
    let mut config = Qwen3Config::load(&directory.join("config.json")).unwrap();
    let paths = safetensors_files(&directory).unwrap();
    let store = WeightStore::from_safetensors(&paths, DType::F32, &Device::Cpu).unwrap();
    config.prefix = detect_prefix(&store, &["model.", ""], "embed_tokens.weight").unwrap();
    let model = Qwen3::load(&config, &store).unwrap();
    let cases: &[(&[u32], [f32; 8])] = &[
        (
            &[151644],
            [
                -3.46368, 26.74275, 1.28799, 20.01643, 1.24097, 2.71907, 0.96970, 5.80652,
            ],
        ),
        (
            &[151644, 872],
            [
                0.85074, -7.37803, -1.38624, 7.84967, 3.43327, -12.17062, 3.21112, 2.02833,
            ],
        ),
    ];
    for (ids, expected) in cases {
        let input = Tensor::from_vec(ids.to_vec(), (1, ids.len()), &Device::Cpu).unwrap();
        let hidden = model.forward(&input).unwrap();
        let tail = hidden
            .narrow(1, hidden.dim(1).unwrap() - 1, 1)
            .unwrap()
            .squeeze(1)
            .unwrap()
            .to_dtype(DType::F32)
            .unwrap()
            .to_vec2::<f32>()
            .unwrap();
        for (index, want) in expected.iter().enumerate() {
            let got = tail[0][index];
            assert!(
                (got - want).abs() < 1e-3,
                "ids {ids:?} index {index}: got {got}, want {want}"
            );
        }
    }
}
