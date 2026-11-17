//! Load an exported Laya ONNX model and run one prediction.
//!
//! ```bash
//! cargo run -p sys1-laya --features onnx --example onnx_load -- /path/to/laya-onnx
//! ```

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let directory = std::env::args()
        .nth(1)
        .expect("usage: onnx_load <onnx-model-directory>");
    eprintln!("loading {}", directory);
    let model = sys1_laya::LayaOnnxModel::load_dir(
        std::path::Path::new(&directory),
        sys1_laya::LayaOptions::default(),
    )?;
    eprintln!("loaded");
    let output = sys1_core::DecisionModel::predict(
        &model,
        &sys1_protocol::State::text("hello world"),
        &sys1_core::Questions::new().noul("billing", "Is this about billing?"),
    )?;
    println!("{output:?}");
    Ok(())
}
