mod support;

use serde_json::json;
use sha2::{Digest, Sha256};
use std::fs;
use support::{HttpRequest, MockResponse, MockServer};
use sys1_hub::{Hub, HubConfig};

fn sha256(bytes: &[u8]) -> String {
    hex::encode(Sha256::digest(bytes))
}

#[test]
fn hf_resolves_pagination_subfolder_auth_and_pulls_files() {
    let model_bytes = b"weights";
    let extra_bytes = b"{}";
    let spaced_bytes = b"spaced";
    let model_sha = sha256(model_bytes);
    let server_sha = model_sha.clone();
    let revision = "main";

    let server = MockServer::start(move |request: &HttpRequest| match request.path() {
        "/api/models/owner/repo/tree/main" => MockResponse::json(&json!([
            {
                "type": "file",
                "path": "README.md",
                "size": 9,
                "oid": "1".repeat(40)
            },
            {
                "type": "directory",
                "path": "sub",
                "size": 0
            },
            {
                "type": "file",
                "path": "sub/model.safetensors",
                "size": model_bytes.len(),
                "oid": "2".repeat(40),
                "lfs": { "oid": server_sha.clone(), "size": model_bytes.len() }
            }
        ]))
        .header("Link", "</page2>; rel=\"next\""),
        "/page2" => MockResponse::json(&json!([
            {
                "type": "file",
                "path": "sub/extra.json",
                "size": extra_bytes.len(),
                "oid": "3".repeat(40)
            },
            {
                "type": "file",
                "path": "sub/config with space.json",
                "size": spaced_bytes.len(),
                "oid": "4".repeat(40)
            },
            {
                "type": "file",
                "path": "sub/zero.bin",
                "size": 0,
                "oid": "5".repeat(40)
            }
        ])),
        "/api/models/owner/repo" => MockResponse::json(&json!({
            "lastModified": "2026-02-03T10:00:00Z"
        })),
        "/owner/repo/resolve/main/sub/model.safetensors" => {
            MockResponse::bytes(model_bytes.to_vec())
        }
        "/owner/repo/resolve/main/sub/extra.json" => MockResponse::bytes(extra_bytes.to_vec()),
        "/owner/repo/resolve/main/sub/config%20with%20space.json" => {
            MockResponse::bytes(spaced_bytes.to_vec())
        }
        _ => MockResponse::not_found(),
    });

    let temp = tempfile::tempdir().expect("temp dir");
    let mut config = HubConfig::new(temp.path().join("cache"));
    config.hf_endpoint = server.base_url().to_owned();
    config.hf_token = Some("hf-secret".to_owned());
    let hub = Hub::new(config).expect("hub");

    let reference = format!("hf:owner/repo@{revision}#sub");
    let pulled = hub.pull(&reference).expect("HF pull");
    assert_eq!(pulled.manifest.name, "repo-sub");
    assert_eq!(pulled.manifest.family, "hf");
    assert_eq!(pulled.manifest.variant.as_deref(), Some("sub"));
    assert_eq!(pulled.manifest.version, "main");
    assert_eq!(pulled.manifest.release_date, "2026-02-03");
    assert_eq!(pulled.manifest.backend, "candle");
    assert_eq!(pulled.manifest.format, "safetensors");

    let paths = pulled
        .manifest
        .files
        .iter()
        .map(|file| file.path.as_str())
        .collect::<Vec<_>>();
    assert_eq!(
        paths,
        vec!["model.safetensors", "extra.json", "config with space.json"]
    );
    let weights = pulled
        .manifest
        .files
        .iter()
        .find(|file| file.path == "model.safetensors")
        .expect("weights entry");
    assert_eq!(weights.sha256.as_deref(), Some(model_sha.as_str()));
    assert!(
        weights
            .url
            .ends_with("/owner/repo/resolve/main/sub/model.safetensors"),
        "{}",
        weights.url
    );
    let spaced = pulled
        .manifest
        .files
        .iter()
        .find(|file| file.path == "config with space.json")
        .expect("spaced entry");
    assert!(
        spaced.url.ends_with("config%20with%20space.json"),
        "{}",
        spaced.url
    );

    assert_eq!(
        fs::read(pulled.files["model.safetensors"].as_path()).expect("weights"),
        model_bytes
    );
    assert_eq!(
        fs::read(pulled.files["extra.json"].as_path()).expect("extra"),
        extra_bytes
    );
    assert_eq!(
        fs::read(pulled.files["config with space.json"].as_path()).expect("spaced"),
        spaced_bytes
    );

    let requests = server.requests();
    assert!(
        requests.iter().any(|request| request.path() == "/page2"),
        "pagination page 2 was not requested"
    );
    assert!(
        requests
            .iter()
            .any(|request| request.path() == "/api/models/owner/repo"),
        "release date metadata was not requested"
    );
    assert!(
        requests
            .iter()
            .all(|request| { request.header("authorization") == Some("Bearer hf-secret") }),
        "HF token missing from a request: {requests:?}"
    );
}
