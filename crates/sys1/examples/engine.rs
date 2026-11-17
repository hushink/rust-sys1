//! Run one decision through the high-level in-process engine.
//!
//! ```text
//! cargo run -p sys1 --example engine -- local:/path/to/laya-multilingual
//! cargo run -p sys1 --example engine -- hf:convaiinnovations/laya@main#multilingual laya-multilingual
//! ```
//!
//! The optional second argument registers the model under an explicit name.

use std::env;
use sys1::{EngineOptions, InferenceEngine, Questions, State};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = env::args().skip(1);
    let reference = args.next().ok_or(
        "usage: engine <reference> [model-name] [state]\n\
         example: engine local:/path/to/laya-multilingual",
    )?;
    let explicit_name = args.next();
    let state = args
        .next()
        .unwrap_or_else(|| "I was charged twice. Please refund the extra charge today.".to_owned());

    let engine = InferenceEngine::new(EngineOptions::default())?;
    let registered = match &explicit_name {
        Some(name) => engine.add_as(name, &reference)?,
        None => engine.add(&reference)?,
    };
    for model in &registered {
        println!("registered {} ({})", model.name, model.description);
    }
    let model = explicit_name.unwrap_or_else(|| registered[0].name.clone());

    let questions = Questions::new()
        .noul("billing", "Is this about billing?")
        .score("urgency", ["Can wait", "This week", "Today"]);
    let output = engine.predict(&model, &State::text(state), &questions)?;

    println!("billing: {:?}", output.answers.noul("billing")?);
    println!("urgency: {:?}", output.answers.score("urgency")?);
    println!("input tokens: {}", output.usage.input_tokens);
    Ok(())
}
