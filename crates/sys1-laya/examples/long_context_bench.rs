//! Compare eager and exact sliding-window local attention on long states.
//!
//! Environment:
//! - `SYS1_LAYA_TEST_ROOT`: multilingual checkpoint directory
//! - `SYS1_LONG_DEVICE`: `cuda` (default) or `cpu`
//! - `SYS1_LONG_DTYPE`: `f16` (default), `f32`, or `bf16`
//! - `SYS1_LONG_REPEATS`: comma-separated sentence repeats (default `13,27,50`)

use candle_core::{DType, Device};
use std::time::Instant;
use sys1_core::{Answers, DecisionModel, Questions};
use sys1_laya::{ConfidenceMode, LayaModel, LayaOptions};
use sys1_protocol::State;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let root = std::env::var("SYS1_LAYA_TEST_ROOT")
        .map(std::path::PathBuf::from)
        .map_err(|_| "set SYS1_LAYA_TEST_ROOT to the multilingual checkpoint directory")?;
    let device = match std::env::var("SYS1_LONG_DEVICE")
        .unwrap_or_else(|_| "cuda".to_owned())
        .as_str()
    {
        "cuda" => Device::new_cuda(0)?,
        "cpu" => Device::Cpu,
        other => return Err(format!("unknown SYS1_LONG_DEVICE={other}").into()),
    };
    let dtype = match std::env::var("SYS1_LONG_DTYPE")
        .unwrap_or_else(|_| "f16".to_owned())
        .as_str()
    {
        "f16" => DType::F16,
        "bf16" => DType::BF16,
        "f32" => DType::F32,
        other => return Err(format!("unknown SYS1_LONG_DTYPE={other}").into()),
    };
    let repeats: Vec<usize> = std::env::var("SYS1_LONG_REPEATS")
        .unwrap_or_else(|_| "13,27,50".to_owned())
        .split(',')
        .map(|value| value.trim().parse::<usize>())
        .collect::<Result<_, _>>()?;

    let baseline = load(&root, device.clone(), dtype, false)?;
    let windowed = load(&root, device.clone(), dtype, true)?;
    let noul = Questions::new().noul("billing", "Is this about billing?");
    let mixed = Questions::new()
        .noul("billing", "Is this about billing?")
        .score("urgency", ["Can wait", "This week", "Today"])
        .choice("tone", [("angry", "Upset"), ("calm", "Neutral")]);
    let sentence = "I was charged twice for invoice 12345 and need a refund today. ";

    // Warm up both models and the CUDA allocator.
    let warm_state = State::text(sentence.repeat(*repeats.last().unwrap_or(&1)));
    let _ = baseline.predict(&warm_state, &noul)?;
    let _ = windowed.predict(&warm_state, &noul)?;

    println!(
        "{:>8} {:>9} {:>12} {:>12} {:>9} {:>12}",
        "repeats", "tokens", "eager ms", "window ms", "speedup", "mixed diff"
    );
    for repeat in repeats {
        let state = State::text(sentence.repeat(repeat));
        let (base_output, base_ms) = time(&baseline, &state, &noul)?;
        let (_, window_ms) = time(&windowed, &state, &noul)?;
        let base_mixed = baseline.predict(&state, &mixed)?;
        let window_mixed = windowed.predict(&state, &mixed)?;
        let diff = max_answer_diff(&base_mixed.answers, &window_mixed.answers)?;
        println!(
            "{repeat:>8} {:>9} {base_ms:>12.2} {window_ms:>12.2} {:>8.2}x {diff:>12.6}",
            base_output.usage.input_tokens,
            base_ms / window_ms.max(1e-9),
        );
    }
    Ok(())
}

fn load(
    root: &std::path::Path,
    device: Device,
    dtype: DType,
    local_window: bool,
) -> Result<LayaModel, Box<dyn std::error::Error>> {
    Ok(LayaModel::load_dir(
        root,
        LayaOptions {
            device,
            dtype,
            max_len: Some(1024),
            confidence_mode: ConfidenceMode::TypeSafe,
            local_window,
            max_attention_elements: sys1_laya::DEFAULT_MAX_ATTENTION_ELEMENTS,
        },
    )?)
}

fn time(
    model: &LayaModel,
    state: &State,
    questions: &Questions,
) -> Result<(sys1_core::DecisionOutput, f64), Box<dyn std::error::Error>> {
    let iterations = 5;
    let mut elapsed = 0.0;
    let mut output = None;
    for _ in 0..iterations {
        let start = Instant::now();
        let candidate = model.predict(state, questions)?;
        elapsed += start.elapsed().as_secs_f64() * 1000.0;
        output = Some(candidate);
    }
    Ok((
        output.expect("at least one iteration"),
        elapsed / iterations as f64,
    ))
}

fn max_answer_diff(left: &Answers, right: &Answers) -> Result<f64, Box<dyn std::error::Error>> {
    let mut diff = (left.noul("billing")?.noul - right.noul("billing")?.noul).abs();
    let left_choice = left.choice("tone")?;
    let right_choice = right.choice("tone")?;
    for (label, probability) in &left_choice.probabilities {
        let other = right_choice
            .probabilities
            .get(label)
            .copied()
            .unwrap_or_default();
        diff = diff.max((probability - other).abs());
    }
    let left_score = left.score("urgency")?;
    let right_score = right.score("urgency")?;
    diff = diff.max((left_score.score - right_score.score).abs());
    for (label, probability) in &left_score.probabilities {
        let other = right_score
            .probabilities
            .get(label)
            .copied()
            .unwrap_or_default();
        diff = diff.max((probability - other).abs());
    }
    if left_choice.choice != right_choice.choice {
        diff = diff.max(1.0);
    }
    Ok(diff)
}
