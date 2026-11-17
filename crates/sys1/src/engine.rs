use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use sys1_core::{
    DecisionOutput, Error as RegistryError, LongContextOptions, ManifestError, ModelError,
    ModelManifest, ModelRegistry, ModelResolver, Prediction, PredictionOptions, Questions,
    RegistryConfig,
};
use sys1_hub::{CacheResidency, Hub, HubConfig, HubError, ModelRef};
use sys1_laya::{
    LayaFactory as CheckpointFactory, is_model_directory, local_checkpoint_manifest,
    register_checkpoint_bundle,
};
use sys1_protocol::{ModelMetadata, State, SystemOneResponse};
use thiserror::Error;

/// Backend options for models registered by an [`InferenceEngine`].
pub use sys1_laya::LayaOptions as ModelOptions;

/// Construction options for an [`InferenceEngine`].
#[derive(Clone)]
pub struct EngineOptions {
    /// Model hub, download cache, endpoints, and credentials.
    pub hub: HubConfig,
    /// Registry limits (resident models and keep-alive).
    pub registry: RegistryConfig,
    /// Device, dtype, context length, confidence, and attention options.
    pub model: ModelOptions,
}

impl Default for EngineOptions {
    fn default() -> Self {
        Self {
            hub: HubConfig::from_env(),
            registry: RegistryConfig::default(),
            model: ModelOptions::default(),
        }
    }
}

/// Errors raised while building an engine, adding models, or running
/// inference.
#[derive(Debug, Error)]
pub enum EngineError {
    #[error(transparent)]
    Hub(#[from] HubError),
    #[error(transparent)]
    Model(#[from] ModelError),
    #[error(transparent)]
    Manifest(#[from] ManifestError),
    #[error(transparent)]
    Registry(#[from] RegistryError),
}

/// A complete in-process decision pipeline.
///
/// The engine wires the model hub, manifests, the backend factory, and the
/// registry together:
///
/// - [`InferenceEngine::add`] pulls a reference through the hub, materializes
///   it in the cache, registers it with the backend factory, and adds its
///   manifest to the registry.
/// - [`InferenceEngine::predict`] returns typed [`DecisionOutput`] answers.
/// - [`InferenceEngine::systemone`] returns the official wire response.
/// - Models load lazily on first use; [`InferenceEngine::load`] and
///   [`InferenceEngine::unload`] control residency explicitly.
///
/// It implements [`ModelResolver`], so it can also be passed to
/// `sys1-server`'s `router_with_resolver` to enable runtime model management
/// over HTTP.
pub struct InferenceEngine {
    hub: Hub,
    factory: Arc<CheckpointFactory>,
    registry: Arc<ModelRegistry>,
    options: ModelOptions,
    catalog_lock: Mutex<()>,
}

struct RegistryResidency(Arc<ModelRegistry>);

impl CacheResidency for RegistryResidency {
    fn is_resident(&self, model_name: &str) -> bool {
        self.0.is_loaded(model_name)
    }
}

impl InferenceEngine {
    /// Create an engine with an empty catalog.
    pub fn new(options: EngineOptions) -> Result<Self, EngineError> {
        let EngineOptions {
            hub,
            registry,
            model,
        } = options;
        let hub = Hub::new(hub)?;
        let factory = Arc::new(CheckpointFactory::new());
        let registry = Arc::new(ModelRegistry::new(Vec::new(), factory.clone(), registry)?);
        hub.set_residency(Arc::new(RegistryResidency(registry.clone())));
        Ok(Self {
            hub,
            factory,
            registry,
            options: model,
            catalog_lock: Mutex::new(()),
        })
    }

    /// Create an engine and register the model behind `reference`.
    pub fn from_reference(reference: &str, options: EngineOptions) -> Result<Self, EngineError> {
        let engine = Self::new(options)?;
        engine.add(reference)?;
        Ok(engine)
    }

    /// Create an engine and register a local checkpoint directory.
    ///
    /// The catalog name comes from the directory name; use
    /// [`InferenceEngine::add_as`] on a [`InferenceEngine::new`] engine if you
    /// need an explicit name.
    pub fn from_directory(
        directory: impl AsRef<Path>,
        options: EngineOptions,
    ) -> Result<Self, EngineError> {
        let reference = format!("local:{}", directory.as_ref().display());
        Self::from_reference(&reference, options)
    }

    /// Pull a reference and register every resulting model.
    ///
    /// A single checkpoint registers under its manifest name. A Laya
    /// repository root registers a router plus the three variant models.
    pub fn add(&self, reference: &str) -> Result<Vec<ModelMetadata>, EngineError> {
        self.add_inner(reference, None)
    }

    /// Like [`InferenceEngine::add`], but override the catalog name.
    pub fn add_as(&self, name: &str, reference: &str) -> Result<Vec<ModelMetadata>, EngineError> {
        self.add_inner(reference, Some(name))
    }

    /// Unload a model and remove it from the catalog. Returns whether it was
    /// present.
    pub fn remove(&self, name: &str) -> bool {
        let _guard = self.lock_catalog();
        let Some((canonical, _)) = self.registry.resolve(name) else {
            return false;
        };
        let removed = self.registry.unregister(&canonical);
        if removed {
            let _ = self.factory.unregister(&canonical);
        }
        removed
    }

    /// Load a registered model into memory now.
    pub fn load(&self, name: &str) -> Result<(), EngineError> {
        self.registry.acquire(name)?;
        Ok(())
    }

    /// Drop the resident instance, returning whether it was loaded.
    pub fn unload(&self, name: &str) -> bool {
        self.registry.unload(name)
    }

    /// Canonical names of resident models.
    pub fn loaded_models(&self) -> Vec<String> {
        self.registry.loaded_models()
    }

    /// Whether a model is currently resident.
    pub fn is_loaded(&self, name: &str) -> bool {
        self.registry.is_loaded(name)
    }

    /// Canonical manifests in the catalog.
    pub fn models(&self) -> Vec<ModelManifest> {
        self.registry.manifests()
    }

    /// Resolve a catalog name or alias without pulling anything.
    pub fn model(&self, name: &str) -> Option<ModelManifest> {
        self.registry.resolve(name).map(|(_, manifest)| manifest)
    }

    /// Run one prediction and return typed answers.
    pub fn predict(
        &self,
        model: &str,
        state: &State,
        questions: &Questions,
    ) -> Result<DecisionOutput, EngineError> {
        Ok(self.registry.predict_output(model, state, questions)?)
    }

    /// Run one prediction with per-call controls such as `min_confidence`.
    pub fn predict_with(
        &self,
        model: &str,
        state: &State,
        questions: &Questions,
        options: &PredictionOptions,
    ) -> Result<DecisionOutput, EngineError> {
        Ok(self
            .registry
            .predict_output_with(model, state, questions, options)?)
    }

    /// Scan a long state in overlapping windows and return typed answers with
    /// window attribution.
    pub fn predict_long(
        &self,
        model: &str,
        state: &State,
        questions: &Questions,
        options: LongContextOptions,
    ) -> Result<DecisionOutput, EngineError> {
        Ok(self
            .registry
            .predict_long_output(model, state, questions, options)?)
    }

    /// Run one prediction and return the official response shape.
    pub fn systemone(
        &self,
        model: &str,
        state: &State,
        questions: &Questions,
    ) -> Result<SystemOneResponse, EngineError> {
        Ok(self.registry.predict(model, state, questions)?)
    }

    /// Run a batch through the model's `predict_batch` implementation.
    ///
    /// Per-request failures are returned in the vector; engine-level failures
    /// (unknown model, invalid questions, load failure) are returned as
    /// `Err`.
    pub fn predict_batch(
        &self,
        model: &str,
        requests: &[Prediction<'_>],
    ) -> Result<Vec<Result<DecisionOutput, ModelError>>, EngineError> {
        for request in requests {
            request
                .questions
                .validate()
                .map_err(RegistryError::Validation)?;
        }
        if requests.is_empty() {
            return Ok(Vec::new());
        }
        self.registry.maintenance();
        let model = self.registry.acquire(model)?;
        Ok(model.predict_batch(requests))
    }

    /// Batched inference with per-call controls applied to each result.
    pub fn predict_batch_with(
        &self,
        model: &str,
        requests: &[Prediction<'_>],
        options: &PredictionOptions,
    ) -> Result<Vec<Result<DecisionOutput, ModelError>>, EngineError> {
        for request in requests {
            request
                .questions
                .validate()
                .map_err(RegistryError::Validation)?;
        }
        if requests.is_empty() {
            return Ok(Vec::new());
        }
        self.registry.maintenance();
        let model = self.registry.acquire(model)?;
        Ok(model.predict_batch_with(requests, options))
    }

    fn add_inner(
        &self,
        reference: &str,
        name: Option<&str>,
    ) -> Result<Vec<ModelMetadata>, EngineError> {
        let _guard = self.lock_catalog();
        let manifests = self.resolve_manifests(reference, name)?;
        let resolved: Vec<String> = manifests
            .iter()
            .map(|manifest| manifest.name.clone())
            .collect();
        let pre_existing: HashSet<String> = self
            .registry
            .manifests()
            .into_iter()
            .map(|manifest| manifest.name)
            .collect();

        let mut added = Vec::with_capacity(manifests.len());
        for manifest in manifests {
            let metadata = manifest.metadata();
            let manifest_name = manifest.name.clone();
            if let Err(error) = self.registry.register(manifest) {
                for resolved_name in &resolved {
                    if !pre_existing.contains(resolved_name) {
                        self.registry.unregister(resolved_name);
                        let _ = self.factory.unregister(resolved_name);
                    }
                }
                return Err(error.into());
            }
            added.push((manifest_name, metadata));
        }
        Ok(added.into_iter().map(|(_, metadata)| metadata).collect())
    }

    fn resolve_manifests(
        &self,
        reference: &str,
        name: Option<&str>,
    ) -> Result<Vec<ModelManifest>, EngineError> {
        let parsed: ModelRef = reference.parse()?;
        let (manifest, directory) = if parsed.scheme == "local" {
            let path = PathBuf::from(&parsed.location);
            if path.is_dir() && !path.join("model.json").is_file() {
                let (manifest, _) = local_checkpoint_manifest(&path)?;
                (manifest, path)
            } else {
                let pulled = self.hub.pull(reference)?;
                (pulled.manifest, pulled.directory)
            }
        } else {
            let pulled = self.hub.pull(reference)?;
            (pulled.manifest, pulled.directory)
        };
        if !is_model_directory(&directory) {
            return Err(ModelError::unsupported(format!(
                "`{reference}` is not a Laya checkpoint directory (safetensors or ONNX)"
            ))
            .into());
        }
        Ok(register_checkpoint_bundle(
            &self.factory,
            &manifest,
            &directory,
            self.options.clone(),
            name,
        )?)
    }

    fn lock_catalog(&self) -> std::sync::MutexGuard<'_, ()> {
        self.catalog_lock
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }
}

impl ModelResolver for InferenceEngine {
    fn resolve(&self, reference: &str) -> Result<Vec<ModelManifest>, ModelError> {
        let _guard = self.lock_catalog();
        self.resolve_manifests(reference, None)
            .map_err(|error| match error {
                EngineError::Model(error) => error,
                other => ModelError::load(other.to_string()),
            })
    }

    fn release(&self, name: &str) -> bool {
        self.factory.unregister(name)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::path::PathBuf;

    fn checkpoint(directory: &Path) {
        fs::create_dir_all(directory.join("encoder")).expect("encoder dir");
        fs::write(directory.join("model.safetensors"), b"not real weights").expect("weights");
        fs::write(directory.join("encoder/config.json"), b"{}").expect("encoder config");
        fs::write(directory.join("rl_agent_config.json"), b"{}").expect("rl config");
        fs::write(directory.join("tokenizer.json"), b"{}").expect("tokenizer");
    }

    #[test]
    fn engine_is_a_send_sync_model_resolver() {
        fn assert_resolver<T: ModelResolver + Send + Sync>() {}
        assert_resolver::<InferenceEngine>();
    }

    fn engine(tag: &str) -> (InferenceEngine, PathBuf) {
        let root = std::env::temp_dir().join(format!("sys1-engine-{}-{tag}", std::process::id()));
        let _ = fs::remove_dir_all(&root);
        let directory = root.join("laya-multilingual");
        fs::create_dir_all(&directory).expect("checkpoint dir");
        checkpoint(&directory);

        let options = EngineOptions {
            hub: HubConfig::new(root.join("cache")),
            ..EngineOptions::default()
        };
        let engine = InferenceEngine::new(options).expect("engine");
        (engine, directory)
    }

    #[test]
    fn registers_models_from_a_directory_and_removes_them() {
        let (engine, directory) = engine("register");
        let metadata = engine
            .add(&format!("local:{}", directory.display()))
            .expect("add checkpoint");
        assert_eq!(metadata.len(), 1);
        assert_eq!(metadata[0].name, "laya-multilingual");
        assert_eq!(engine.models().len(), 1);
        assert!(!engine.is_loaded("laya-multilingual"));

        let error = engine
            .load("laya-multilingual")
            .expect_err("invalid checkpoint must fail to load");
        assert!(matches!(
            error,
            EngineError::Registry(RegistryError::Model { .. })
        ));
        assert!(engine.loaded_models().is_empty());

        assert!(engine.remove("laya-multilingual"));
        assert!(engine.models().is_empty());
        assert!(!engine.remove("laya-multilingual"));

        fs::remove_dir_all(directory.parent().expect("root")).expect("cleanup");
    }

    #[test]
    fn adds_a_model_under_an_explicit_name() {
        let (engine, directory) = engine("named");
        let metadata = engine
            .add_as("custom", &format!("local:{}", directory.display()))
            .expect("add checkpoint");
        assert_eq!(metadata[0].name, "custom");
        assert_eq!(engine.models()[0].name, "custom");
        assert!(engine.model("custom").is_some());

        let error = engine
            .predict(
                "custom",
                &State::text("hello"),
                &Questions::new().noul("q", "Is this a question?"),
            )
            .expect_err("invalid checkpoint must fail to load");
        assert!(matches!(
            error,
            EngineError::Registry(RegistryError::Model { .. })
        ));

        assert!(engine.remove("custom"));
        fs::remove_dir_all(directory.parent().expect("root")).expect("cleanup");
    }
}
