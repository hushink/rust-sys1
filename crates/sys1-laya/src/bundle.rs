use crate::onnx_config::{ONNX_SIDECAR, OnnxFileConfig};
use crate::{LayaFactory, LayaOptions};
use indexmap::IndexMap;
use std::collections::HashSet;
use std::fs;
use std::path::{Path, PathBuf};
use sys1_core::{ModelError, ModelFile, ModelManifest};

/// Whether `directory` contains the files required by a candle Laya
/// checkpoint.
pub fn is_checkpoint_directory(directory: &Path) -> bool {
    directory.join("model.safetensors").is_file()
        && directory.join("encoder/config.json").is_file()
        && directory.join("rl_agent_config.json").is_file()
        && has_tokenizer(directory)
}

/// Whether `directory` contains an ONNX export of a Laya checkpoint.
///
/// The export must provide `rl_agent_config.json`, a tokenizer, and either
/// `model.onnx` or an `onnx.json` sidecar naming another `.onnx` file.
pub fn is_onnx_directory(directory: &Path) -> bool {
    let weights = directory.join("model.onnx").is_file() || directory.join(ONNX_SIDECAR).is_file();
    weights && directory.join("rl_agent_config.json").is_file() && has_tokenizer(directory)
}

/// Whether `directory` is loadable by the Laya factory in any backend.
pub fn is_model_directory(directory: &Path) -> bool {
    is_checkpoint_directory(directory) || is_onnx_directory(directory)
}

/// Whether `directory` is a Laya repository root with all three checkpoints.
pub fn has_checkpoint_variants(directory: &Path) -> bool {
    is_model_directory(directory)
        && is_model_directory(&directory.join("multilingual"))
        && is_model_directory(&directory.join("typed-decisions"))
}

fn has_tokenizer(directory: &Path) -> bool {
    directory.join("tokenizer/tokenizer.json").is_file()
        || directory.join("tokenizer/tokenizer.v1.json").is_file()
        || directory.join("tokenizer.json").is_file()
}

/// Checkpoint directories exposed by a Laya repository root.
pub fn variant_directories(root: &Path) -> [(&'static str, PathBuf); 3] {
    [
        ("laya-english", root.to_path_buf()),
        ("laya-multilingual", root.join("multilingual")),
        ("laya-typed-decisions", root.join("typed-decisions")),
    ]
}

/// Build a manifest and file map for a bare local checkpoint directory (one
/// without `model.json`).
pub fn local_checkpoint_manifest(
    directory: &Path,
) -> Result<(ModelManifest, IndexMap<String, PathBuf>), ModelError> {
    let candle = is_checkpoint_directory(directory);
    if !candle && !is_onnx_directory(directory) {
        return Err(ModelError::load(format!(
            "{} is not a Laya checkpoint directory (safetensors or ONNX)",
            directory.display()
        )));
    }

    let tokenizer = [
        "tokenizer/tokenizer.json",
        "tokenizer/tokenizer.v1.json",
        "tokenizer.json",
    ]
    .into_iter()
    .find(|path| directory.join(path).is_file())
    .ok_or_else(|| {
        ModelError::load(format!(
            "no tokenizer.json found under {} or its tokenizer/ subdirectory",
            directory.display()
        ))
    })?;

    let (backend, format, model_file, description) = if candle {
        (
            "candle",
            "safetensors",
            "model.safetensors".to_owned(),
            format!("Local Laya checkpoint at {}", directory.display()),
        )
    } else {
        let config = OnnxFileConfig::load(directory)?;
        let model_path = config.model_path(directory)?;
        if !model_path.is_file() {
            return Err(ModelError::load(format!(
                "missing ONNX model at {}",
                model_path.display()
            )));
        }
        (
            "onnx",
            "onnx",
            config.model.replace('\\', "/"),
            format!("Local Laya ONNX export at {}", directory.display()),
        )
    };

    let mut paths = vec![model_file.as_str(), "rl_agent_config.json", tokenizer];
    if candle {
        paths.push("encoder/config.json");
    } else if directory.join(ONNX_SIDECAR).is_file() {
        paths.push(ONNX_SIDECAR);
    }
    for config in ["tokenizer/tokenizer_config.json", "tokenizer_config.json"] {
        if directory.join(config).is_file() {
            paths.push(config);
        }
    }
    paths.sort_unstable();
    paths.dedup();

    let mut files = Vec::with_capacity(paths.len());
    for path in &paths {
        let full = directory.join(path);
        let size = fs::metadata(&full)
            .map_err(|error| ModelError::load(format!("metadata for {}: {error}", full.display())))?
            .len();
        files.push(ModelFile {
            path: (*path).to_owned(),
            url: String::new(),
            sha256: None,
            git_sha1: None,
            size,
        });
    }

    let name = directory
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("laya")
        .to_owned();
    let manifest = ModelManifest {
        schema_version: 1,
        name,
        description,
        release_date: "1970-01-01".to_owned(),
        family: "laya".to_owned(),
        variant: None,
        backend: backend.to_owned(),
        format: format.to_owned(),
        version: "local".to_owned(),
        aliases: Vec::new(),
        files,
    };
    manifest
        .validate()
        .map_err(|error| ModelError::load(error.to_string()))?;

    let mut file_map = IndexMap::new();
    for file in &manifest.files {
        file_map.insert(file.path.clone(), directory.join(&file.path));
    }
    Ok((manifest, file_map))
}

/// Register a checkpoint, or a repository bundle with `multilingual/` and
/// `typed-decisions/` subdirectories, with `factory`.
///
/// A bundle registers a router under the manifest name plus three variants. A
/// single checkpoint registers under the manifest name, or under `name` when
/// provided. Returned manifests are ready to insert into a
/// [`sys1_core::ModelRegistry`].
pub fn register_checkpoint_bundle(
    factory: &LayaFactory,
    manifest: &ModelManifest,
    directory: &Path,
    options: LayaOptions,
    name: Option<&str>,
) -> Result<Vec<ModelManifest>, ModelError> {
    if !is_model_directory(directory) {
        return Err(ModelError::load(format!(
            "{} is not a Laya checkpoint directory (safetensors or ONNX)",
            directory.display()
        )));
    }

    let mut manifest = manifest.clone();
    manifest.family = "laya".to_owned();
    apply_backend(&mut manifest, directory);
    if let Some(name) = name {
        manifest.name = name.to_owned();
        manifest.aliases.clear();
    }

    let mut manifests = Vec::new();
    let mut seen = HashSet::new();
    if has_checkpoint_variants(directory) {
        if supports_router(directory) {
            factory.register_router(&manifest.name, directory, options.clone());
        }
        for (variant_name, variant_directory) in variant_directories(directory) {
            if !is_model_directory(&variant_directory) {
                continue;
            }
            factory.register(variant_name, &variant_directory, options.clone());
            let mut variant = manifest.clone();
            variant.name = variant_name.to_owned();
            variant.aliases.clear();
            variant.description = format!("{} ({variant_name})", manifest.description);
            apply_backend(&mut variant, &variant_directory);
            push_unique(&mut manifests, &mut seen, variant);
        }
        manifest.aliases.clear();
        push_unique(&mut manifests, &mut seen, manifest);
    } else {
        factory.register(&manifest.name, directory, options);
        push_unique(&mut manifests, &mut seen, manifest);
    }
    Ok(manifests)
}

/// Point a manifest at the backend that `directory` can actually load.
fn apply_backend(manifest: &mut ModelManifest, directory: &Path) {
    if is_checkpoint_directory(directory) {
        manifest.backend = "candle".to_owned();
        manifest.format = "safetensors".to_owned();
    } else {
        manifest.backend = "onnx".to_owned();
        manifest.format = "onnx".to_owned();
    }
}

/// A router loads all three candle checkpoints in one process. Mixed or ONNX
/// bundles register their variants individually instead.
fn supports_router(directory: &Path) -> bool {
    is_checkpoint_directory(directory)
        && variant_directories(directory)
            .iter()
            .all(|(_, variant)| is_checkpoint_directory(variant))
}

fn push_unique(
    manifests: &mut Vec<ModelManifest>,
    seen: &mut HashSet<String>,
    manifest: ModelManifest,
) {
    if seen.insert(manifest.name.clone()) {
        manifests.push(manifest);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use sys1_core::ModelFile;

    fn checkpoint(directory: &Path) {
        fs::create_dir_all(directory.join("encoder")).expect("encoder dir");
        fs::write(directory.join("model.safetensors"), b"x").expect("weights");
        fs::write(directory.join("encoder/config.json"), b"{}").expect("encoder config");
        fs::write(directory.join("rl_agent_config.json"), b"{}").expect("rl config");
        fs::write(directory.join("tokenizer.json"), b"{}").expect("tokenizer");
    }

    fn temp_bundle(tag: &str) -> PathBuf {
        let directory =
            std::env::temp_dir().join(format!("sys1-bundle-{}-{tag}", std::process::id()));
        let _ = fs::remove_dir_all(&directory);
        fs::create_dir_all(&directory).expect("bundle dir");
        checkpoint(&directory);
        directory
    }

    fn manifest(name: &str) -> ModelManifest {
        ModelManifest {
            schema_version: 1,
            name: name.to_owned(),
            description: "test".to_owned(),
            release_date: "2026-01-01".to_owned(),
            family: "hf".to_owned(),
            variant: None,
            backend: "unknown".to_owned(),
            format: "unknown".to_owned(),
            version: "test".to_owned(),
            aliases: vec!["repo-alias".to_owned()],
            files: vec![ModelFile {
                path: "model.safetensors".to_owned(),
                url: String::new(),
                sha256: None,
                git_sha1: None,
                size: 1,
            }],
        }
    }

    #[test]
    fn builds_a_local_checkpoint_manifest() {
        let directory = temp_bundle("local");
        let (manifest, files) = local_checkpoint_manifest(&directory).expect("local manifest");

        assert_eq!(
            manifest.name,
            directory
                .file_name()
                .expect("directory name")
                .to_str()
                .expect("utf-8 name")
        );
        assert_eq!(manifest.family, "laya");
        assert_eq!(manifest.backend, "candle");
        assert!(files.contains_key("model.safetensors"));
        assert!(
            manifest
                .files
                .iter()
                .any(|file| file.path == "tokenizer.json")
        );
        fs::remove_dir_all(directory).expect("cleanup");
    }

    #[test]
    fn registers_a_single_checkpoint() {
        let directory = temp_bundle("single");
        let factory = LayaFactory::new();
        let manifests = register_checkpoint_bundle(
            &factory,
            &manifest("repo"),
            &directory,
            LayaOptions::default(),
            Some("custom-name"),
        )
        .expect("register checkpoint");

        assert_eq!(manifests.len(), 1);
        assert_eq!(manifests[0].name, "custom-name");
        assert!(manifests[0].aliases.is_empty());
        assert_eq!(manifests[0].family, "laya");
        assert_eq!(manifests[0].backend, "candle");
        assert_eq!(manifests[0].format, "safetensors");
        assert_eq!(factory.registered(), vec!["custom-name"]);
        assert!(factory.unregister("custom-name"));
        assert!(factory.registered().is_empty());
        fs::remove_dir_all(directory).expect("cleanup");
    }

    #[test]
    fn registers_a_router_and_variants() {
        let root = temp_bundle("variants");
        checkpoint(&root.join("multilingual"));
        checkpoint(&root.join("typed-decisions"));

        let factory = LayaFactory::new();
        let manifests = register_checkpoint_bundle(
            &factory,
            &manifest("laya"),
            &root,
            LayaOptions::default(),
            None,
        )
        .expect("register bundle");

        let names = manifests
            .iter()
            .map(|manifest| manifest.name.as_str())
            .collect::<Vec<_>>();
        assert_eq!(
            names,
            vec![
                "laya-english",
                "laya-multilingual",
                "laya-typed-decisions",
                "laya"
            ]
        );
        assert_eq!(
            factory.registered(),
            vec![
                "laya",
                "laya-english",
                "laya-multilingual",
                "laya-typed-decisions"
            ]
        );

        for name in names {
            assert!(factory.unregister(name), "{name}");
        }
        assert!(factory.registered().is_empty());
        fs::remove_dir_all(root).expect("cleanup");
    }

    #[test]
    fn rejects_a_non_checkpoint_directory() {
        let directory = temp_bundle("invalid");
        fs::remove_file(directory.join("model.safetensors")).expect("remove weights");

        let factory = LayaFactory::new();
        let error = register_checkpoint_bundle(
            &factory,
            &manifest("missing"),
            &directory,
            LayaOptions::default(),
            None,
        )
        .expect_err("must reject missing checkpoint");
        assert!(matches!(error, ModelError::Load { .. }));
        assert!(factory.registered().is_empty());
        fs::remove_dir_all(directory).expect("cleanup");
    }

    #[test]
    fn builds_a_local_onnx_manifest() {
        let directory = temp_bundle("onnx-local");
        fs::remove_file(directory.join("model.safetensors")).expect("remove candle weights");
        fs::remove_dir_all(directory.join("encoder")).expect("remove encoder config");
        fs::write(directory.join("model.onnx"), b"onnx").expect("onnx model");
        fs::write(
            directory.join("onnx.json"),
            r#"{"model": "model.onnx", "max_batch": 4}"#,
        )
        .expect("sidecar");

        assert!(is_onnx_directory(&directory));
        assert!(is_model_directory(&directory));
        assert!(!is_checkpoint_directory(&directory));

        let (manifest, files) = local_checkpoint_manifest(&directory).expect("onnx manifest");
        assert_eq!(manifest.backend, "onnx");
        assert_eq!(manifest.format, "onnx");
        assert!(files.contains_key("model.onnx"));
        assert!(files.contains_key("onnx.json"));
        fs::remove_dir_all(directory).expect("cleanup");
    }

    #[test]
    fn registers_an_onnx_checkpoint() {
        let directory = temp_bundle("onnx-register");
        fs::remove_file(directory.join("model.safetensors")).expect("remove candle weights");
        fs::remove_dir_all(directory.join("encoder")).expect("remove encoder config");
        fs::write(directory.join("model.onnx"), b"onnx").expect("onnx model");

        let factory = LayaFactory::new();
        let manifests = register_checkpoint_bundle(
            &factory,
            &manifest("onnx"),
            &directory,
            LayaOptions::default(),
            Some("onnx-custom"),
        )
        .expect("register ONNX checkpoint");

        assert_eq!(manifests.len(), 1);
        assert_eq!(manifests[0].name, "onnx-custom");
        assert_eq!(manifests[0].backend, "onnx");
        assert_eq!(manifests[0].format, "onnx");
        assert_eq!(factory.registered(), vec!["onnx-custom"]);
        fs::remove_dir_all(directory).expect("cleanup");
    }

    #[test]
    fn sidecar_names_a_nested_onnx_model() {
        let directory = temp_bundle("onnx-nested");
        fs::remove_file(directory.join("model.safetensors")).expect("remove candle weights");
        fs::remove_dir_all(directory.join("encoder")).expect("remove encoder config");
        fs::create_dir_all(directory.join("weights")).expect("weights dir");
        fs::write(directory.join("weights/model.onnx"), b"onnx").expect("nested model");
        fs::write(
            directory.join("onnx.json"),
            r#"{"model": "weights/model.onnx"}"#,
        )
        .expect("sidecar");

        let (manifest, files) = local_checkpoint_manifest(&directory).expect("onnx manifest");
        assert!(files.contains_key("weights/model.onnx"));
        assert!(
            manifest
                .files
                .iter()
                .any(|file| file.path == "weights/model.onnx")
        );
        fs::remove_dir_all(directory).expect("cleanup");
    }
}
