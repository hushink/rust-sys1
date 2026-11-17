use candle_core::{DType, Device};
use std::path::PathBuf;
use std::time::Instant;
use sys1_core::{DecisionModel, Prediction, Questions};
use sys1_laya::{LayaModel, LayaOptions};
use sys1_protocol::State;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let root = std::env::var("SYS1_LAYA_TEST_ROOT")
        .map(PathBuf::from)
        .map_err(|_| "set SYS1_LAYA_TEST_ROOT to the multilingual checkpoint directory")?;
    let device = match std::env::var("SYS1_BATCH_DEVICE")
        .unwrap_or_else(|_| "cpu".to_owned())
        .as_str()
    {
        "cuda" => Device::new_cuda(0)?,
        "cpu" => Device::Cpu,
        other => {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                format!("unknown SYS1_BATCH_DEVICE={other}"),
            )
            .into());
        }
    };
    let dtype = match std::env::var("SYS1_BATCH_DTYPE")
        .unwrap_or_else(|_| "f32".to_owned())
        .as_str()
    {
        "f16" => DType::F16,
        "bf16" => DType::BF16,
        _ => DType::F32,
    };

    let model = LayaModel::load_dir(
        &root,
        LayaOptions {
            device,
            dtype,
            ..LayaOptions::default()
        },
    )?;

    let max_batch = 32;
    let states: Vec<State> = (0..max_batch)
        .map(|index| {
            State::text(format!(
                "I was charged twice for invoice {index}. Please refund me today."
            ))
        })
        .collect();
    let questions: Vec<Questions> = (0..max_batch)
        .map(|_| Questions::new().noul("billing", "Is this about billing?"))
        .collect();
    let predictions: Vec<Prediction<'_>> = states
        .iter()
        .zip(&questions)
        .map(|(state, questions)| Prediction::new(state, questions))
        .collect();

    // Warm up the encoder/GPU and the tokenizer.
    for output in model.predict_batch(&predictions[..8]) {
        output?;
    }

    println!("| Batch | Total ms | Per request ms | Req/s |");
    println!("|---:|---:|---:|---:|");
    for batch in [1_usize, 2, 4, 8, 16, 32] {
        let reps = if batch <= 8 { 5 } else { 3 };
        let mut total_ms = 0.0;
        for _ in 0..reps {
            let started = Instant::now();
            let outputs = model.predict_batch(&predictions[..batch]);
            total_ms += started.elapsed().as_secs_f64() * 1000.0;
            for output in outputs {
                output?;
            }
        }
        let mean_ms = total_ms / reps as f64;
        println!(
            "| {batch} | {mean_ms:.2} | {:.2} | {:.1} |",
            mean_ms / batch as f64,
            batch as f64 * 1000.0 / mean_ms
        );
    }

    Ok(())
}
