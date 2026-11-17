//! Compare `InferenceEngine::predict` with a directly loaded checkpoint.
//!
//! ```text
//! cargo run -p sys1 --release --example engine_overhead -- \
//!   /path/to/laya-multilingual 10
//! ```
//!
//! Both paths use the same checkpoint and default CPU options; the first
//! prediction is excluded from the timings. The difference is the registry,
//! manifest lookup, and handle bookkeeping added by the engine.

use std::env;
use std::path::Path;
use std::time::{Duration, Instant};
use sys1::{
    DecisionModel, EngineOptions, InferenceEngine, LayaModel, LayaOptions, Questions, State,
};

fn stats(samples: &[Duration]) -> (f64, f64) {
    let mut millis = samples
        .iter()
        .map(|sample| sample.as_secs_f64() * 1000.0)
        .collect::<Vec<_>>();
    millis.sort_by(f64::total_cmp);
    let mean = millis.iter().sum::<f64>() / millis.len() as f64;
    let median = millis[millis.len() / 2];
    (mean, median)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = env::args().skip(1);
    let directory = args
        .next()
        .ok_or("usage: engine_overhead <checkpoint-dir> [iterations]")?;
    let iterations = args
        .next()
        .map(|value| value.parse::<usize>())
        .transpose()?
        .unwrap_or(10)
        .max(1);

    let state = State::text("I was charged twice. Please refund me today.");
    let questions = Questions::new()
        .noul("billing", "Is this about billing?")
        .score("urgency", ["Can wait", "This week", "Today"]);
    let options = LayaOptions::default();

    let direct = LayaModel::load_dir(Path::new(&directory), options.clone())?;
    direct.predict(&state, &questions)?;

    let engine = InferenceEngine::from_directory(
        &directory,
        EngineOptions {
            model: options,
            ..EngineOptions::default()
        },
    )?;
    let model = engine
        .models()
        .first()
        .ok_or("engine registered no models")?
        .name
        .clone();
    engine.predict(&model, &state, &questions)?;

    // Interleave both paths and alternate the order so CPU frequency drift or
    // allocator warm-up does not favor one side.
    let mut direct_samples = Vec::with_capacity(iterations);
    let mut engine_samples = Vec::with_capacity(iterations);
    for index in 0..iterations {
        if index % 2 == 0 {
            let started = Instant::now();
            direct.predict(&state, &questions)?;
            direct_samples.push(started.elapsed());

            let started = Instant::now();
            engine.predict(&model, &state, &questions)?;
            engine_samples.push(started.elapsed());
        } else {
            let started = Instant::now();
            engine.predict(&model, &state, &questions)?;
            engine_samples.push(started.elapsed());

            let started = Instant::now();
            direct.predict(&state, &questions)?;
            direct_samples.push(started.elapsed());
        }
    }

    let (direct_mean, direct_median) = stats(&direct_samples);
    let (engine_mean, engine_median) = stats(&engine_samples);
    println!("direct  mean {direct_mean:.2} ms  p50 {direct_median:.2} ms  ({model})");
    println!("engine  mean {engine_mean:.2} ms  p50 {engine_median:.2} ms");
    println!(
        "overhead mean {:.3} ms  p50 {:.3} ms",
        engine_mean - direct_mean,
        engine_median - direct_median
    );
    Ok(())
}
