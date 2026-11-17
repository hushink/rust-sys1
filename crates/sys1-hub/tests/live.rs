//! Opt-in live endpoint tests. They are skipped unless `SYS1_LIVE_HUB=1`.
//!
//! HF may require a reachable endpoint (`HF_ENDPOINT`) and token in restricted
//! networks; ModelScope's public API needs neither.

use std::env;
use sys1_hub::{Hub, HubConfig};

fn live() -> bool {
    let enabled = env::var("SYS1_LIVE_HUB").ok().as_deref() == Some("1");
    if !enabled {
        eprintln!("skipping live hub test (set SYS1_LIVE_HUB=1)");
    }
    enabled
}

fn hub() -> (Hub, tempfile::TempDir) {
    let cache = tempfile::tempdir().expect("temp cache");
    let mut config = HubConfig::from_env();
    config.cache_dir = cache.path().to_path_buf();
    (Hub::new(config).expect("hub"), cache)
}

#[test]
fn live_hugging_face_resolves_a_manifest() {
    if !live() {
        return;
    }
    let (hub, _cache) = hub();
    let manifest = hub
        .resolve("hf:convaiinnovations/laya-multilingual@main")
        .expect("resolve live HF manifest");
    assert_eq!(manifest.name, "laya-multilingual");
    assert!(manifest.validate().is_ok());
    assert!(
        manifest
            .files
            .iter()
            .any(|file| file.path == "model.safetensors"),
        "HF manifest files: {:?}",
        manifest
            .files
            .iter()
            .map(|file| &file.path)
            .collect::<Vec<_>>()
    );
    assert!(
        manifest
            .files
            .iter()
            .any(|file| file.path.ends_with("tokenizer.json")),
        "HF manifest has no tokenizer"
    );
}

#[test]
fn live_modelscope_resolves_a_manifest() {
    if !live() {
        return;
    }
    let (hub, _cache) = hub();
    let manifest = hub
        .resolve("modelscope:Qwen/Qwen2.5-0.5B-Instruct@master")
        .expect("resolve live ModelScope manifest");
    assert!(
        manifest.name.ends_with("Qwen2.5-0.5B-Instruct"),
        "unexpected name: {}",
        manifest.name
    );
    assert!(manifest.validate().is_ok());
    assert!(
        manifest.files.iter().any(|file| file.path == "config.json"),
        "ModelScope manifest files: {:?}",
        manifest
            .files
            .iter()
            .map(|file| &file.path)
            .collect::<Vec<_>>()
    );
    assert!(
        manifest.files.iter().any(|file| file.size > 0),
        "ModelScope manifest has no sized files"
    );
}
