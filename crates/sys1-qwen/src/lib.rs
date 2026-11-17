//! Candle backbones for the Qwen3 and Qwen3.5 decoder families.
//!
//! The models are inference-only: weights come from safetensors shards (memory
//! mapped), a PyTorch zip checkpoint, or an in-memory tensor map, and an
//! optional PEFT LoRA adapter can be merged into the linear layers at load
//! time. Both backbones return the final normed hidden states for right-padded
//! causal rows; families then read option positions and apply their own heads.

#![deny(unsafe_code)]

mod config;
mod qwen3;
mod qwen35;
mod weights;

pub use config::{Qwen3Config, Qwen35Config, RopeParameters};
pub use qwen3::{
    Mlp, Qwen3, RmsNorm, apply_rope, candidate_logits, causal_bias, repeat_kv, rotary_tables,
};
pub use qwen35::Qwen35;
pub use weights::{LoraAdapter, WeightStore};

/// Collect the sharded or single-file safetensors checkpoints in a directory,
/// excluding auxiliary head files.
pub fn safetensors_files(directory: &std::path::Path) -> std::io::Result<Vec<std::path::PathBuf>> {
    let mut paths = std::fs::read_dir(directory)?
        .filter_map(|entry| entry.ok())
        .map(|entry| entry.path())
        .filter(|path| {
            path.extension().and_then(|extension| extension.to_str()) == Some("safetensors")
                && !path
                    .file_name()
                    .and_then(|name| name.to_str())
                    .is_some_and(|name| name.contains("head") || name.contains("adapter"))
        })
        .collect::<Vec<_>>();
    paths.sort();
    Ok(paths)
}

/// Detect the tensor-name prefix of a decoder by looking for `suffix`.
pub fn detect_prefix(store: &WeightStore, candidates: &[&str], suffix: &str) -> Option<String> {
    candidates.iter().find_map(|prefix| {
        let name = format!("{prefix}{suffix}");
        store.contains(&name).then(|| (*prefix).to_owned())
    })
}
