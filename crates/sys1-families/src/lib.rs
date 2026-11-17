//! Multi-family dispatch for local model directories.
//!
//! The CLI and the `sys1` umbrella engine use this crate to register and load
//! every compiled model family behind one [`ModelFactory`].

#![forbid(unsafe_code)]

use indexmap::IndexMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use sys1_clef::{ClefFactory, ClefModel, ClefOptions, is_clef_directory};
use sys1_core::{DecisionModel, ModelError, ModelFactory, ModelFile, ModelManifest};
use sys1_decima::{DecimaFactory, DecimaModel, DecimaOptions, is_decima_directory};
use sys1_decision::{DecisionFactory, DecisionOptions, is_decision_directory};
use sys1_gliclass::{GliclassFactory, GliclassModel, GliclassOptions, is_gliclass_directory};
use sys1_kev::{
    KevFactory, KevModel, KevOptions, has_kev_base, is_kev_directory, resolve_base_directory,
};
use sys1_laya::{
    LayaFactory, LayaModel, LayaOptions, is_model_directory, register_checkpoint_bundle,
};
use sys1_nli::{NliFactory, NliModel, NliOptions, is_nli_directory};
use sys1_qwen3guard::{
    Qwen3GuardFactory, Qwen3GuardModel, Qwen3GuardOptions, is_qwen3guard_directory,
};
use sys1_von::{VonFactory, VonModel, VonOptions, is_von_directory};

/// Context-length override shared by sentence-encoder families.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct MaxLenOptions {
    /// Truncate rows to this many tokens. Unset uses the server-wide
    /// `max_len`, then the checkpoint default.
    pub max_len: Option<usize>,
}

/// Decima-specific options.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct DecimaFamilyOptions {
    /// Force the choice-fit augmentation on or off, overriding `decima.json`.
    pub choice_fit: Option<bool>,
}

/// Kev-specific options.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct KevFamilyOptions {
    /// Local Qwen3.5 base directory. When unset, the base beside the adapter,
    /// under `base/`, or the repository declared in `head.pt` / `kev.json` is
    /// used.
    pub base_dir: Option<PathBuf>,
    /// Pointer-logit temperature. When unset, `kev.json` and the calibrated
    /// temperature recorded in `head.pt` are used.
    pub temperature: Option<f32>,
}

/// Decision-specific options.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct DecisionFamilyOptions {
    /// Endpoint-logit temperature, overriding the checkpoint calibration.
    pub temperature: Option<f32>,
}

/// Per-family configuration applied to every bundle registered by a factory.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct FamilyOptions {
    /// Cross-request batch size for the encoder families (default 32).
    pub max_batch: Option<usize>,
    /// Qwen3Guard row budget.
    pub qwen3guard: MaxLenOptions,
    /// Von row budget.
    pub von: MaxLenOptions,
    /// Clef row budget.
    pub clef: MaxLenOptions,
    pub decima: DecimaFamilyOptions,
    pub kev: KevFamilyOptions,
    pub decision: DecisionFamilyOptions,
}

impl FamilyOptions {
    /// Batch size for encoder families.
    pub fn batch_size(&self) -> usize {
        self.max_batch.unwrap_or(32).max(1)
    }
}

/// Per-bundle values discovered while resolving a reference.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct BundleOverrides {
    /// Qwen3.5 base materialized for a Kev adapter.
    pub kev_base: Option<PathBuf>,
}

/// Global family options plus per-bundle overrides.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct BundleOptions {
    pub families: FamilyOptions,
    pub overrides: BundleOverrides,
}

impl BundleOptions {
    pub fn new(families: FamilyOptions) -> Self {
        Self {
            families,
            overrides: BundleOverrides::default(),
        }
    }
}

/// Resolve family-specific extras a directory needs before registration.
///
/// Kev adapters may declare their Qwen3.5 base in `kev.json`, `head.pt`,
/// `provenance.json`, or `adapter_config.json`; `pull` is called with that
/// reference when the base is not already present.
pub fn resolve_bundle_overrides<F>(
    directory: &Path,
    families: &FamilyOptions,
    pull: F,
) -> Result<BundleOverrides, ModelError>
where
    F: FnOnce(&str) -> Result<PathBuf, ModelError>,
{
    let mut overrides = BundleOverrides::default();
    if is_kev_directory(directory) {
        let base = resolve_base_directory(directory, families.kev.base_dir.as_deref(), pull)?;
        overrides.kev_base = Some(base);
    }
    Ok(overrides)
}

/// Dispatch a manifest to the compiled model family.
#[derive(Default)]
pub struct FamilyFactory {
    pub laya: LayaFactory,
    pub nli: NliFactory,
    pub gliclass: GliclassFactory,
    pub qwen3guard: Qwen3GuardFactory,
    pub von: VonFactory,
    pub decima: DecimaFactory,
    pub kev: KevFactory,
    pub decision: DecisionFactory,
    pub clef: ClefFactory,
}

impl FamilyFactory {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn release(&self, name: &str) -> bool {
        self.laya.unregister(name)
            | self.nli.unregister(name)
            | self.gliclass.unregister(name)
            | self.qwen3guard.unregister(name)
            | self.von.unregister(name)
            | self.decima.unregister(name)
            | self.kev.unregister(name)
            | self.decision.unregister(name)
            | self.clef.unregister(name)
    }

    pub fn registered(&self) -> Vec<String> {
        let mut names = self.laya.registered();
        names.extend(self.nli.registered());
        names.extend(self.gliclass.registered());
        names.extend(self.qwen3guard.registered());
        names.extend(self.von.registered());
        names.extend(self.decima.registered());
        names.extend(self.kev.registered());
        names.extend(self.decision.registered());
        names.extend(self.clef.registered());
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
            "qwen3guard" => self.qwen3guard.load(manifest),
            "von" => self.von.load(manifest),
            "decima" => self.decima.load(manifest),
            "kev" => self.kev.load(manifest),
            "decision" => self.decision.load(manifest),
            "clef" => self.clef.load(manifest),
            _ => self.laya.load(manifest),
        }
    }
}

/// Whether `directory` can be served by one of the compiled families.
pub fn is_supported_directory(directory: &Path) -> bool {
    is_model_directory(directory)
        || is_nli_directory(directory)
        || is_gliclass_directory(directory)
        || is_qwen3guard_directory(directory)
        || is_von_directory(directory)
        || is_decima_directory(directory)
        || is_kev_directory(directory)
        || is_decision_directory(directory)
        || is_clef_directory(directory)
}

fn rename(mut manifest: ModelManifest, family: &str, name: Option<&str>) -> ModelManifest {
    manifest.family = family.to_owned();
    manifest.backend = "candle".to_owned();
    manifest.format = "safetensors".to_owned();
    if let Some(name) = name {
        manifest.name = name.to_owned();
        manifest.aliases.clear();
    }
    manifest
}

/// Register a directory with the matching family factory.
pub fn register_bundle(
    factory: &FamilyFactory,
    manifest: &ModelManifest,
    directory: &Path,
    options: &LayaOptions,
    bundle: &BundleOptions,
    name: Option<&str>,
) -> Result<Vec<ModelManifest>, ModelError> {
    let families = &bundle.families;
    let batch = families.batch_size();
    let max_len = |family: Option<usize>| family.or(options.max_len);
    if is_nli_directory(directory) {
        let manifest = rename(manifest.clone(), "nli", name);
        factory.nli.register(
            &manifest.name,
            directory,
            NliOptions {
                device: options.device.clone(),
                dtype: options.dtype,
                max_len: options.max_len,
                max_batch: batch,
            },
        );
        return Ok(vec![manifest]);
    }
    if is_gliclass_directory(directory) {
        let manifest = rename(manifest.clone(), "gliclass", name);
        factory.gliclass.register(
            &manifest.name,
            directory,
            GliclassOptions {
                device: options.device.clone(),
                dtype: options.dtype,
                max_len: options.max_len,
                max_batch: batch,
            },
        );
        return Ok(vec![manifest]);
    }
    if is_qwen3guard_directory(directory) {
        let manifest = rename(manifest.clone(), "qwen3guard", name);
        factory.qwen3guard.register(
            &manifest.name,
            directory,
            Qwen3GuardOptions {
                device: options.device.clone(),
                dtype: options.dtype,
                max_len: max_len(families.qwen3guard.max_len),
            },
        );
        return Ok(vec![manifest]);
    }
    if is_von_directory(directory) {
        let manifest = rename(manifest.clone(), "von", name);
        factory.von.register(
            &manifest.name,
            directory,
            VonOptions {
                device: options.device.clone(),
                dtype: options.dtype,
                max_len: max_len(families.von.max_len),
                max_batch: batch,
            },
        );
        return Ok(vec![manifest]);
    }
    if is_decima_directory(directory) {
        let manifest = rename(manifest.clone(), "decima", name);
        factory.decima.register(
            &manifest.name,
            directory,
            DecimaOptions {
                device: options.device.clone(),
                dtype: options.dtype,
                max_batch: batch,
                choice_fit: families.decima.choice_fit,
            },
        );
        return Ok(vec![manifest]);
    }
    if is_kev_directory(directory) {
        let manifest = rename(manifest.clone(), "kev", name);
        let base_directory = bundle
            .overrides
            .kev_base
            .clone()
            .or_else(|| families.kev.base_dir.clone());
        if base_directory.is_none() && !has_kev_base(directory) {
            return Err(ModelError::load(format!(
                "{} is a Kev adapter but has no Qwen3.5 base; resolve its declared base, put \
                 the base files in a `base/` subdirectory, or set [families.kev] base_dir",
                directory.display()
            )));
        }
        factory.kev.register(
            &manifest.name,
            directory,
            KevOptions {
                device: options.device.clone(),
                dtype: options.dtype,
                base_directory,
                temperature: families.kev.temperature,
            },
        );
        return Ok(vec![manifest]);
    }
    if is_decision_directory(directory) {
        let manifest = rename(manifest.clone(), "decision", name);
        factory.decision.register(
            &manifest.name,
            directory,
            DecisionOptions {
                device: options.device.clone(),
                dtype: options.dtype,
                temperature: families.decision.temperature,
            },
        );
        return Ok(vec![manifest]);
    }
    if is_clef_directory(directory) {
        let manifest = rename(manifest.clone(), "clef", name);
        factory.clef.register(
            &manifest.name,
            directory,
            ClefOptions {
                device: options.device.clone(),
                dtype: options.dtype,
                max_len: max_len(families.clef.max_len),
            },
        );
        return Ok(vec![manifest]);
    }
    if directory.join("adapter_model.safetensors").is_file() && directory.join("head.pt").is_file()
    {
        return Err(ModelError::load(format!(
            "{} is a Kev adapter but is missing adapter_config.json or tokenizer.json",
            directory.display()
        )));
    }
    register_checkpoint_bundle(&factory.laya, manifest, directory, options.clone(), name)
}

/// Build a manifest for a supported local directory without `model.json`.
pub fn local_manifest(
    directory: &Path,
) -> Result<(ModelManifest, IndexMap<String, PathBuf>), ModelError> {
    let (family, description) = if is_nli_directory(directory) {
        ("nli", "ModernBERT zero-shot NLI")
    } else if is_gliclass_directory(directory) {
        ("gliclass", "GLiClass uni-encoder")
    } else if is_qwen3guard_directory(directory) {
        ("qwen3guard", "Qwen3Guard Gen fixed-preset guard")
    } else if is_von_directory(directory) {
        ("von", "Von option-marker encoder")
    } else if is_decima_directory(directory) {
        ("decima", "Decima late-interaction encoder")
    } else if is_kev_directory(directory) {
        ("kev", "Kev pointer-head decoder")
    } else if is_decision_directory(directory) {
        ("decision", "Decision 1.0 endpoint-head decoder")
    } else if is_clef_directory(directory) {
        ("clef", "Clef joint-schema decoder")
    } else {
        return Err(ModelError::load(format!(
            "{} is not a supported checkpoint directory",
            directory.display()
        )));
    };

    let mut paths = Vec::new();
    collect_files(directory, directory, &mut paths)?;
    let mut files = Vec::with_capacity(paths.len());
    for (relative, size) in &paths {
        files.push(ModelFile {
            path: relative.clone(),
            url: String::new(),
            sha256: None,
            git_sha1: None,
            size: *size,
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
        description: format!("Local {description} model at {}", directory.display()),
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

fn collect_files(
    root: &Path,
    directory: &Path,
    out: &mut Vec<(String, u64)>,
) -> Result<(), ModelError> {
    let entries = fs::read_dir(directory)
        .map_err(|error| ModelError::load(format!("read {}: {error}", directory.display())))?;
    let mut entries = entries.filter_map(|entry| entry.ok()).collect::<Vec<_>>();
    entries.sort_by_key(|entry| entry.file_name());
    for entry in entries {
        let path = entry.path();
        let name = entry.file_name().to_string_lossy().into_owned();
        if name == ".cache" || name == ".git" || name.starts_with('.') {
            continue;
        }
        if path.is_dir() {
            collect_files(root, &path, out)?;
        } else if path.is_file() {
            let relative = path
                .strip_prefix(root)
                .map_err(|error| ModelError::load(error.to_string()))?
                .to_string_lossy()
                .replace('\\', "/");
            let size = entry
                .metadata()
                .map_err(|error| {
                    ModelError::load(format!("metadata for {}: {error}", path.display()))
                })?
                .len();
            out.push((relative, size));
        }
    }
    Ok(())
}

/// Load one model directly, without a registry (used by `sys1 run`).
pub fn load_directory(
    directory: &Path,
    options: &LayaOptions,
    bundle: &BundleOptions,
) -> Result<Arc<dyn DecisionModel>, ModelError> {
    let families = &bundle.families;
    let batch = families.batch_size();
    let max_len = |family: Option<usize>| family.or(options.max_len);
    if is_nli_directory(directory) {
        return Ok(Arc::new(NliModel::load_dir(
            directory,
            NliOptions {
                device: options.device.clone(),
                dtype: options.dtype,
                max_len: options.max_len,
                max_batch: batch,
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
                max_batch: batch,
            },
        )?));
    }
    if is_qwen3guard_directory(directory) {
        return Ok(Arc::new(Qwen3GuardModel::load_dir(
            directory,
            Qwen3GuardOptions {
                device: options.device.clone(),
                dtype: options.dtype,
                max_len: max_len(families.qwen3guard.max_len),
            },
        )?));
    }
    if is_von_directory(directory) {
        return Ok(Arc::new(VonModel::load_dir(
            directory,
            VonOptions {
                device: options.device.clone(),
                dtype: options.dtype,
                max_len: max_len(families.von.max_len),
                max_batch: batch,
            },
        )?));
    }
    if is_decima_directory(directory) {
        return Ok(Arc::new(DecimaModel::load_dir(
            directory,
            DecimaOptions {
                device: options.device.clone(),
                dtype: options.dtype,
                max_batch: batch,
                choice_fit: families.decima.choice_fit,
            },
        )?));
    }
    if is_kev_directory(directory) {
        let base_directory = bundle
            .overrides
            .kev_base
            .clone()
            .or_else(|| families.kev.base_dir.clone());
        if base_directory.is_none() && !has_kev_base(directory) {
            return Err(ModelError::load(format!(
                "{} is a Kev adapter but has no Qwen3.5 base; resolve its declared base, put \
                 the base files in a `base/` subdirectory, or set [families.kev] base_dir",
                directory.display()
            )));
        }
        return Ok(Arc::new(KevModel::load_dir(
            directory,
            KevOptions {
                device: options.device.clone(),
                dtype: options.dtype,
                base_directory,
                temperature: families.kev.temperature,
            },
        )?));
    }
    if is_decision_directory(directory) {
        return Ok(Arc::new(sys1_decision::DecisionModel::load_dir(
            directory,
            DecisionOptions {
                device: options.device.clone(),
                dtype: options.dtype,
                temperature: families.decision.temperature,
            },
        )?));
    }
    if is_clef_directory(directory) {
        return Ok(Arc::new(ClefModel::load_dir(
            directory,
            ClefOptions {
                device: options.device.clone(),
                dtype: options.dtype,
                max_len: max_len(families.clef.max_len),
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

/// Load a model by manifest, for callers that already pulled it.
pub fn load_manifest(
    manifest: &ModelManifest,
    directory: &Path,
    options: &LayaOptions,
    bundle: &BundleOptions,
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
    load_directory(directory, options, bundle)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn adapter(directory: &Path) {
        fs::create_dir_all(directory).expect("adapter dir");
        fs::write(directory.join("adapter_model.safetensors"), b"x").expect("adapter");
        fs::write(directory.join("adapter_config.json"), b"{}").expect("adapter config");
        fs::write(directory.join("head.pt"), b"x").expect("head");
        fs::write(directory.join("tokenizer.json"), b"{}").expect("tokenizer");
    }

    fn base(directory: &Path) {
        fs::create_dir_all(directory).expect("base dir");
        fs::write(directory.join("config.json"), b"{}").expect("base config");
        fs::write(directory.join("model.safetensors"), b"x").expect("base weights");
    }

    #[test]
    fn resolves_a_base_declared_by_the_adapter_config() {
        let root = std::env::temp_dir().join(format!("sys1-families-kev-{}", std::process::id()));
        let _ = fs::remove_dir_all(&root);
        let adapter_dir = root.join("adapter");
        let base_dir = root.join("base-model");
        adapter(&adapter_dir);
        base(&base_dir);
        fs::write(
            adapter_dir.join("adapter_config.json"),
            serde_json::json!({
                "base_model_name_or_path": base_dir.display().to_string(),
            })
            .to_string(),
        )
        .expect("adapter config");

        let overrides =
            resolve_bundle_overrides(&adapter_dir, &FamilyOptions::default(), |reference| {
                panic!("local base must not be pulled, got {reference}")
            })
            .expect("overrides");
        assert_eq!(overrides.kev_base.as_deref(), Some(base_dir.as_path()));

        fs::remove_dir_all(root).expect("cleanup");
    }

    #[test]
    fn pulls_a_declared_hf_base_with_its_revision() {
        let root =
            std::env::temp_dir().join(format!("sys1-families-kev-pull-{}", std::process::id()));
        let _ = fs::remove_dir_all(&root);
        let adapter_dir = root.join("adapter");
        let base_dir = root.join("base-model");
        adapter(&adapter_dir);
        base(&base_dir);
        fs::write(
            adapter_dir.join("kev.json"),
            serde_json::json!({
                "base": "Qwen/Qwen3.5-4B-Base",
                "base_revision": "abc123",
            })
            .to_string(),
        )
        .expect("kev.json");

        let mut requested = String::new();
        let overrides =
            resolve_bundle_overrides(&adapter_dir, &FamilyOptions::default(), |reference| {
                requested = reference.to_owned();
                Ok(base_dir.clone())
            })
            .expect("overrides");
        assert_eq!(requested, "hf:Qwen/Qwen3.5-4B-Base@abc123");
        assert_eq!(overrides.kev_base.as_deref(), Some(base_dir.as_path()));

        fs::remove_dir_all(root).expect("cleanup");
    }

    #[test]
    fn configured_base_wins_over_the_declaration() {
        let root =
            std::env::temp_dir().join(format!("sys1-families-kev-config-{}", std::process::id()));
        let _ = fs::remove_dir_all(&root);
        let adapter_dir = root.join("adapter");
        let configured = root.join("configured");
        let declared = root.join("declared");
        adapter(&adapter_dir);
        base(&configured);
        base(&declared);
        fs::write(
            adapter_dir.join("kev.json"),
            serde_json::json!({ "base": declared.display().to_string() }).to_string(),
        )
        .expect("kev.json");

        let families = FamilyOptions {
            kev: KevFamilyOptions {
                base_dir: Some(configured.clone()),
                temperature: Some(2.0),
            },
            ..FamilyOptions::default()
        };
        let overrides = resolve_bundle_overrides(&adapter_dir, &families, |reference| {
            panic!("configured base must not be pulled, got {reference}")
        })
        .expect("overrides");
        assert_eq!(overrides.kev_base.as_deref(), Some(configured.as_path()));

        fs::remove_dir_all(root).expect("cleanup");
    }

    #[test]
    fn kev_detection_accepts_a_base_less_adapter() {
        let root =
            std::env::temp_dir().join(format!("sys1-families-kev-detect-{}", std::process::id()));
        let _ = fs::remove_dir_all(&root);
        adapter(&root);
        assert!(is_kev_directory(&root));
        assert!(!has_kev_base(&root));
        fs::remove_dir_all(root).expect("cleanup");
    }
}
