//! Multi-family dispatch for local model directories.
//!
//! The CLI and the `sys1` umbrella engine use this crate to register and load
//! Laya, NLI, and GLiClass checkpoints behind one [`ModelFactory`].

#![forbid(unsafe_code)]

use indexmap::IndexMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use sys1_core::{DecisionModel, ModelError, ModelFactory, ModelFile, ModelManifest};
use sys1_gliclass::{GliclassFactory, GliclassModel, GliclassOptions, is_gliclass_directory};
use sys1_laya::{
    LayaFactory, LayaModel, LayaOptions, is_model_directory, register_checkpoint_bundle,
};
use sys1_nli::{NliFactory, NliModel, NliOptions, is_nli_directory};

/// Dispatch a manifest to the compiled model family.
#[derive(Default)]
pub struct FamilyFactory {
    pub laya: LayaFactory,
    pub nli: NliFactory,
    pub gliclass: GliclassFactory,
}

impl FamilyFactory {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn release(&self, name: &str) -> bool {
        self.laya.unregister(name) | self.nli.unregister(name) | self.gliclass.unregister(name)
    }

    pub fn registered(&self) -> Vec<String> {
        let mut names = self.laya.registered();
        names.extend(self.nli.registered());
        names.extend(self.gliclass.registered());
        names.sort();
        names.dedup();
        names
    }
}

impl ModelFactory for FamilyFactory {
    fn load(&self, manifest: &ModelManifest) -> Result<Arc<dyn DecisionModel>, ModelError> {
        match manifest.family.as_str() {
            "nli" => self.nli.load(manifest),
            "gliclass" => self.gliclass.load(manifest),
            _ => self.laya.load(manifest),
        }
    }
}

/// Whether `directory` can be served by one of the compiled families.
pub fn is_supported_directory(directory: &Path) -> bool {
    is_model_directory(directory) || is_nli_directory(directory) || is_gliclass_directory(directory)
}

/// Register a directory with the matching family factory.
pub fn register_bundle(
    factory: &FamilyFactory,
    manifest: &ModelManifest,
    directory: &Path,
    options: &LayaOptions,
    name: Option<&str>,
) -> Result<Vec<ModelManifest>, ModelError> {
    if is_nli_directory(directory) {
        let mut manifest = manifest.clone();
        manifest.family = "nli".to_owned();
        manifest.backend = "candle".to_owned();
        manifest.format = "safetensors".to_owned();
        if let Some(name) = name {
            manifest.name = name.to_owned();
            manifest.aliases.clear();
        }
        factory.nli.register(
            &manifest.name,
            directory,
            NliOptions {
                device: options.device.clone(),
                dtype: options.dtype,
                max_len: options.max_len,
                max_batch: 32,
            },
        );
        return Ok(vec![manifest]);
    }
    if is_gliclass_directory(directory) {
        let mut manifest = manifest.clone();
        manifest.family = "gliclass".to_owned();
        manifest.backend = "candle".to_owned();
        manifest.format = "safetensors".to_owned();
        if let Some(name) = name {
            manifest.name = name.to_owned();
            manifest.aliases.clear();
        }
        factory.gliclass.register(
            &manifest.name,
            directory,
            GliclassOptions {
                device: options.device.clone(),
                dtype: options.dtype,
                max_len: options.max_len,
                max_batch: 32,
            },
        );
        return Ok(vec![manifest]);
    }
    register_checkpoint_bundle(&factory.laya, manifest, directory, options.clone(), name)
}

/// Build a manifest for a local NLI/GLiClass directory without `model.json`.
pub fn local_manifest(
    directory: &Path,
) -> Result<(ModelManifest, IndexMap<String, PathBuf>), ModelError> {
    let (family, description) = if is_nli_directory(directory) {
        (
            "nli",
            format!("Local NLI ModernBERT model at {}", directory.display()),
        )
    } else if is_gliclass_directory(directory) {
        (
            "gliclass",
            format!("Local GLiClass model at {}", directory.display()),
        )
    } else {
        return Err(ModelError::load(format!(
            "{} is not an NLI or GLiClass checkpoint directory",
            directory.display()
        )));
    };

    let mut paths = Vec::new();
    for candidate in [
        "config.json",
        "model.safetensors",
        "tokenizer.json",
        "tokenizer/tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "added_tokens.json",
    ] {
        if directory.join(candidate).is_file() {
            paths.push(candidate);
        }
    }
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
        .unwrap_or(family)
        .to_owned();
    let manifest = ModelManifest {
        schema_version: 1,
        name,
        description,
        release_date: "1970-01-01".to_owned(),
        family: family.to_owned(),
        variant: None,
        backend: "candle".to_owned(),
        format: "safetensors".to_owned(),
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

/// Load one model directly, without a registry (used by `sys1 run`).
pub fn load_directory(
    directory: &Path,
    options: &LayaOptions,
) -> Result<Arc<dyn DecisionModel>, ModelError> {
    if is_nli_directory(directory) {
        return Ok(Arc::new(NliModel::load_dir(
            directory,
            NliOptions {
                device: options.device.clone(),
                dtype: options.dtype,
                max_len: options.max_len,
                max_batch: 32,
            },
        )?));
    }
    if is_gliclass_directory(directory) {
        return Ok(Arc::new(GliclassModel::load_dir(
            directory,
            GliclassOptions {
                device: options.device.clone(),
                dtype: options.dtype,
                max_len: options.max_len,
                max_batch: 32,
            },
        )?));
    }
    if directory.join("model.onnx").is_file() || directory.join("onnx.json").is_file() {
        #[cfg(feature = "onnx")]
        {
            return Ok(Arc::new(sys1_laya::LayaOnnxModel::load_dir(
                directory,
                options.clone(),
            )?));
        }
        #[cfg(not(feature = "onnx"))]
        {
            return Err(ModelError::unsupported(
                "this build does not include the `onnx` feature",
            ));
        }
    }
    Ok(Arc::new(LayaModel::load_dir(directory, options.clone())?))
}

/// Load a model by manifest backend, for callers that already pulled it.
pub fn load_manifest(
    manifest: &ModelManifest,
    directory: &Path,
    options: &LayaOptions,
) -> Result<Arc<dyn DecisionModel>, ModelError> {
    if manifest.backend == "onnx" {
        #[cfg(feature = "onnx")]
        {
            return Ok(Arc::new(sys1_laya::LayaOnnxModel::load_dir(
                directory,
                options.clone(),
            )?));
        }
        #[cfg(not(feature = "onnx"))]
        {
            return Err(ModelError::unsupported(
                "this build does not include the `onnx` feature",
            ));
        }
    }
    load_directory(directory, options)
}
