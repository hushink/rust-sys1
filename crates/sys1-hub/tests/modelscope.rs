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
fn modelscope_resolves_subfolder_auth_and_pulls_files() {
    let model_bytes = b"model";
    let config_bytes = b"[]";
    let model_sha = sha256(model_bytes);
    let server_sha = model_sha.clone();

    let server = MockServer::start(move |request: &HttpRequest| match request.path() {
        "/api/v1/models/owner/repo/repo/files" => MockResponse::json(&json!({
            "Success": true,
            "Data": {
                "Files": [
                    {
                        "Path": "sub/model.safetensors",
                        "Size": model_bytes.len(),
                        "Sha256": server_sha.clone(),
                        "Type": "blob"
                    },
                    {
                        "Path": "sub/config.json",
                        "Size": config_bytes.len(),
                        "Sha256": null,
                        "Type": "blob"
                    },
                    {
                        "Path": "sub/empty.bin",
                        "Size": 0,
                        "Sha256": null,
                        "Type": "blob"
                    },
                    {
                        "Path": "sub",
                        "Size": 0,
                        "Type": "tree"
                    }
                ]
            }
        })),
        "/api/v1/models/owner/repo/repo" => match request.query_param("FilePath").as_deref() {
            Some("sub/model.safetensors") => MockResponse::bytes(model_bytes.to_vec()),
            Some("sub/config.json") => MockResponse::bytes(config_bytes.to_vec()),
            _ => MockResponse::not_found(),
        },
        _ => MockResponse::not_found(),
    });

    let temp = tempfile::tempdir().expect("temp dir");
    let mut config = HubConfig::new(temp.path().join("cache"));
    config.modelscope_endpoint = server.base_url().to_owned();
    config.modelscope_token = Some("ms-secret".to_owned());
    let hub = Hub::new(config).expect("hub");

    let pulled = hub
        .pull("modelscope:owner/repo@v1#sub")
        .expect("ModelScope pull");
    assert_eq!(pulled.manifest.name, "repo-sub");
    assert_eq!(pulled.manifest.family, "modelscope");
    assert_eq!(pulled.manifest.variant.as_deref(), Some("sub"));
    assert_eq!(pulled.manifest.version, "v1");
    assert_eq!(pulled.manifest.release_date, "1970-01-01");
    assert_eq!(pulled.manifest.backend, "candle");
    assert_eq!(pulled.manifest.format, "safetensors");

    let paths = pulled
        .manifest
        .files
        .iter()
        .map(|file| file.path.as_str())
        .collect::<Vec<_>>();
    assert_eq!(paths, vec!["model.safetensors", "config.json"]);
    let weights = pulled
        .manifest
        .files
        .iter()
        .find(|file| file.path == "model.safetensors")
        .expect("weights entry");
    assert_eq!(weights.sha256.as_deref(), Some(model_sha.as_str()));
    assert!(weights.url.contains("Revision=v1"), "{}", weights.url);
    assert!(
        weights.url.contains("FilePath=sub%2Fmodel.safetensors"),
        "{}",
        weights.url
    );

    assert_eq!(
        fs::read(pulled.files["model.safetensors"].as_path()).expect("weights"),
        model_bytes
    );
    assert_eq!(
        fs::read(pulled.files["config.json"].as_path()).expect("config"),
        config_bytes
    );

    let requests = server.requests();
    let listing = requests
        .iter()
        .find(|request| request.path().ends_with("/repo/files"))
        .expect("files request");
    assert_eq!(listing.query_param("Revision").as_deref(), Some("v1"));
    assert_eq!(listing.query_param("Root").as_deref(), Some("sub"));
    assert_eq!(listing.query_param("Recursive").as_deref(), Some("true"));
    assert!(
        requests.iter().any(|request| {
            request
                .query_param("FilePath")
                .as_deref()
                .is_some_and(|path| path == "sub/model.safetensors")
        }),
        "download request missing"
    );
    assert!(
        requests.iter().all(|request| {
            request.header("authorization") == Some("Bearer ms-secret")
                && request.header("cookie") == Some("m_session_id=ms-secret")
        }),
        "ModelScope credentials missing from a request: {requests:?}"
    );
}

#[test]
fn modelscope_splits_a_truncated_recursive_listing() {
    let server = MockServer::start(|request: &HttpRequest| {
        if request.path() != "/api/v1/models/owner/repo/repo/files" {
            return MockResponse::not_found();
        }
        let root = request.query_param("Root").unwrap_or_default();
        let recursive = request.query_param("Recursive").as_deref() == Some("true");
        match (root.as_str(), recursive) {
            ("", true) => {
                let files = (0..3000)
                    .map(|index| {
                        json!({
                            "Path": format!("blob-{index}.bin"),
                            "Size": 1,
                            "Sha256": null,
                            "Type": "blob"
                        })
                    })
                    .collect::<Vec<_>>();
                MockResponse::json(&json!({
                    "Success": true,
                    "Data": { "Files": files }
                }))
            }
            ("", false) => MockResponse::json(&json!({
                "Success": true,
                "Data": {
                    "Files": [
                        {
                            "Path": "top.bin",
                            "Size": 1,
                            "Sha256": null,
                            "Type": "blob"
                        },
                        {
                            "Path": "dir",
                            "Size": 0,
                            "Type": "tree"
                        }
                    ]
                }
            })),
            ("dir", true) => MockResponse::json(&json!({
                "Success": true,
                "Data": {
                    "Files": [
                        {
                            "Path": "dir/a.bin",
                            "Size": 1,
                            "Sha256": null,
                            "Type": "blob"
                        },
                        {
                            "Path": "dir/b.bin",
                            "Size": 1,
                            "Sha256": null,
                            "Type": "blob"
                        }
                    ]
                }
            })),
            _ => MockResponse::not_found(),
        }
    });

    let temp = tempfile::tempdir().expect("temp dir");
    let mut config = HubConfig::new(temp.path().join("cache"));
    config.modelscope_endpoint = server.base_url().to_owned();
    let hub = Hub::new(config).expect("hub");

    let manifest = hub
        .resolve("modelscope:owner/repo")
        .expect("ModelScope resolve");
    let paths = manifest
        .files
        .iter()
        .map(|file| file.path.as_str())
        .collect::<Vec<_>>();
    assert_eq!(paths, vec!["top.bin", "dir/a.bin", "dir/b.bin"]);
    assert!(
        manifest
            .files
            .iter()
            .all(|file| !file.path.starts_with("blob-")),
        "truncated entries leaked into the manifest"
    );

    let requests = server.requests();
    let roots = requests
        .iter()
        .filter(|request| request.path().ends_with("/repo/files"))
        .map(|request| {
            (
                request.query_param("Root").unwrap_or_default(),
                request.query_param("Recursive").unwrap_or_default(),
            )
        })
        .collect::<Vec<_>>();
    assert!(roots.contains(&(String::new(), "true".to_owned())));
    assert!(roots.contains(&(String::new(), "false".to_owned())));
    assert!(roots.contains(&("dir".to_owned(), "true".to_owned())));
}
