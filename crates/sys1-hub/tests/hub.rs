use sha2::{Digest, Sha256};
use std::fs;
use sys1_core::{ModelFile, ModelManifest};
use sys1_hub::{Hub, HubConfig, HubError, ModelRef};

fn sha256(bytes: &[u8]) -> String {
    hex::encode(Sha256::digest(bytes))
}

fn manifest(name: &str, path: &str, sha256: &str, size: u64) -> ModelManifest {
    ModelManifest {
        schema_version: 1,
        name: name.to_owned(),
        description: format!("{name} test model"),
        release_date: "2026-01-01".to_owned(),
        family: "test".to_owned(),
        variant: None,
        backend: "fake".to_owned(),
        format: "test".to_owned(),
        version: "1.0.0".to_owned(),
        aliases: Vec::new(),
        files: vec![ModelFile {
            path: path.to_owned(),
            url: String::new(),
            sha256: Some(sha256.to_owned()),
            git_sha1: None,
            size,
        }],
    }
}

#[test]
fn reference_parsing() {
    let hf: ModelRef = "hf:convaiinnovations/laya@main#multilingual"
        .parse()
        .expect("hf reference");
    assert_eq!(hf.scheme, "hf");
    assert_eq!(hf.location, "convaiinnovations/laya");
    assert_eq!(hf.revision.as_deref(), Some("main"));
    assert_eq!(hf.subfolder.as_deref(), Some("multilingual"));

    let modelscope: ModelRef = "modelscope:Qwen/Qwen2.5-0.5B-Instruct@master"
        .parse()
        .expect("modelscope reference");
    assert_eq!(modelscope.scheme, "modelscope");
    assert_eq!(modelscope.location, "Qwen/Qwen2.5-0.5B-Instruct");
    assert_eq!(modelscope.revision.as_deref(), Some("master"));

    let http: ModelRef = "https://example.com/registry/model.json"
        .parse()
        .expect("http reference");
    assert_eq!(http.scheme, "https");
    assert_eq!(http.location, "https://example.com/registry/model.json");

    let local: ModelRef = "local:/tmp/model".parse().expect("local reference");
    assert_eq!(local.scheme, "local");
    assert_eq!(local.location, "/tmp/model");

    let custom: ModelRef = "acme:owner/model@v1#sub".parse().expect("custom reference");
    assert_eq!(custom.scheme, "acme");
    assert_eq!(custom.location, "owner/model");
    assert_eq!(custom.revision.as_deref(), Some("v1"));
    assert_eq!(custom.subfolder.as_deref(), Some("sub"));

    assert!("not a reference".parse::<ModelRef>().is_err());
}

#[test]
fn local_pull_and_verify() {
    let temp = tempfile::tempdir().expect("temp dir");
    let model_dir = temp.path().join("model");
    fs::create_dir_all(&model_dir).expect("model dir");
    let content = b"hello weights";
    fs::write(model_dir.join("weights.bin"), content).expect("weights");
    let manifest = manifest(
        "test-model",
        "weights.bin",
        &sha256(content),
        content.len() as u64,
    );
    fs::write(
        model_dir.join("model.json"),
        serde_json::to_vec_pretty(&manifest).expect("manifest json"),
    )
    .expect("manifest");

    let cache = temp.path().join("cache");
    let hub = Hub::new(HubConfig::new(cache)).expect("hub");
    let reference = format!("local:{}", model_dir.display());
    let pulled = hub.pull(&reference).expect("local pull");
    assert_eq!(pulled.manifest.name, "test-model");
    assert_eq!(pulled.directory, model_dir);
    assert_eq!(pulled.files["weights.bin"], model_dir.join("weights.bin"));
}

#[test]
fn removes_cached_models_by_key_and_rejects_traversal() {
    let temp = tempfile::tempdir().expect("temp dir");
    let cache = temp.path().join("cache");
    let content = b"hello weights";
    let manifest = manifest(
        "test-model",
        "weights.bin",
        &sha256(content),
        content.len() as u64,
    );
    let bytes = serde_json::to_vec_pretty(&manifest).expect("manifest json");

    let key = "owner-repo-abc123";
    let model_dir = cache.join("models").join(key);
    fs::create_dir_all(&model_dir).expect("model dir");
    fs::write(model_dir.join("weights.bin"), content).expect("weights");
    fs::write(model_dir.join("model.json"), &bytes).expect("model manifest");
    let cached_manifest = cache.join("manifests").join(format!("{key}.json"));
    fs::create_dir_all(cached_manifest.parent().expect("manifests dir")).expect("manifests dir");
    fs::write(&cached_manifest, &bytes).expect("cached manifest");

    let hub = Hub::new(HubConfig::new(&cache)).expect("hub");
    let cached = hub.list_cached().expect("cached models");
    assert_eq!(cached.len(), 1);
    assert_eq!(cached[0].manifest.name, "test-model");

    assert!(hub.remove_cached(key).expect("remove cached"));
    assert!(!model_dir.exists());
    assert!(!cached_manifest.exists());
    assert!(hub.list_cached().expect("empty cache").is_empty());
    assert!(!hub.remove_cached(key).expect("already removed"));

    for invalid in ["", ".", "..", "../model", "a/b", "a\\b"] {
        let error = hub
            .remove_cached(invalid)
            .expect_err("invalid key must fail");
        assert!(matches!(error, HubError::InvalidData { .. }), "{invalid}");
    }
}

#[test]
fn local_pull_detects_hash_mismatch() {
    let temp = tempfile::tempdir().expect("temp dir");
    let model_dir = temp.path().join("model");
    fs::create_dir_all(&model_dir).expect("model dir");
    let content = b"hello weights";
    fs::write(model_dir.join("weights.bin"), content).expect("weights");
    let manifest = manifest(
        "test-model",
        "weights.bin",
        &"0".repeat(64),
        content.len() as u64,
    );
    fs::write(
        model_dir.join("model.json"),
        serde_json::to_vec_pretty(&manifest).expect("manifest json"),
    )
    .expect("manifest");

    let cache = temp.path().join("cache");
    let hub = Hub::new(HubConfig::new(cache)).expect("hub");
    let reference = format!("local:{}", model_dir.display());
    let error = hub.pull(&reference).expect_err("hash mismatch must fail");
    assert!(matches!(error, HubError::HashMismatch { .. }));
}
