use crate::pull_model;
use indexmap::IndexMap;
use std::sync::{Arc, RwLock};
use sys1_core::{ModelError, ModelManifest, ModelResolver};
use sys1_hub::Hub;
use sys1_laya::{
    LayaFactory as CheckpointFactory, LayaOptions as ModelOptions, register_checkpoint_bundle,
};

/// Resolves hub references into decision-model checkpoints and registers them
/// with the factory used by the running registry.
pub struct HubModelResolver {
    hub: Arc<Hub>,
    factory: Arc<CheckpointFactory>,
    options: ModelOptions,
    /// `[models]` aliases from `sys1.toml`, so the admin API can accept the
    /// same names as the CLI. Replaced atomically on config reload.
    references: RwLock<IndexMap<String, String>>,
}

impl HubModelResolver {
    pub fn new(
        hub: Arc<Hub>,
        factory: Arc<CheckpointFactory>,
        options: ModelOptions,
        references: IndexMap<String, String>,
    ) -> Self {
        Self {
            hub,
            factory,
            options,
            references: RwLock::new(references),
        }
    }

    pub fn replace_references(&self, references: IndexMap<String, String>) {
        *self
            .references
            .write()
            .unwrap_or_else(|poisoned| poisoned.into_inner()) = references;
    }

    fn resolve_named(
        &self,
        reference: &str,
        name: Option<&str>,
    ) -> Result<Vec<ModelManifest>, ModelError> {
        let reference = self
            .references
            .read()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .get(reference)
            .cloned()
            .unwrap_or_else(|| reference.to_owned());
        let pulled = pull_model(self.hub.as_ref(), &reference).map_err(|error| {
            ModelError::load(format!("cannot resolve `{reference}`: {error:#}"))
        })?;
        register_checkpoint_bundle(
            &self.factory,
            &pulled.manifest,
            &pulled.directory,
            self.options.clone(),
            name,
        )
    }
}

impl ModelResolver for HubModelResolver {
    fn resolve(&self, reference: &str) -> Result<Vec<ModelManifest>, ModelError> {
        self.resolve_named(reference, None)
    }

    fn resolve_as(&self, name: &str, reference: &str) -> Result<Vec<ModelManifest>, ModelError> {
        self.resolve_named(reference, Some(name))
    }

    fn release(&self, name: &str) -> bool {
        self.factory.unregister(name)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::path::Path;

    fn checkpoint(directory: &Path) {
        fs::create_dir_all(directory.join("encoder")).expect("encoder dir");
        fs::write(directory.join("model.safetensors"), b"x").expect("weights");
        fs::write(directory.join("encoder/config.json"), b"{}").expect("encoder config");
        fs::write(directory.join("rl_agent_config.json"), b"{}").expect("rl config");
        fs::write(directory.join("tokenizer.json"), b"{}").expect("tokenizer");
    }

    #[test]
    fn resolves_configured_names_and_releases_them() {
        let root =
            std::env::temp_dir().join(format!("sys1-resolver-{}-resolve", std::process::id()));
        let directory = root.join("checkpoint");
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(&directory).expect("checkpoint dir");
        checkpoint(&directory);

        let factory = Arc::new(CheckpointFactory::new());
        let hub = Arc::new(Hub::new(sys1_hub::HubConfig::new(root.join("cache"))).expect("hub"));
        let mut references = IndexMap::new();
        references.insert(
            "configured".to_owned(),
            format!("local:{}", directory.display()),
        );
        let resolver =
            HubModelResolver::new(hub, factory.clone(), ModelOptions::default(), references);

        let manifests = resolver.resolve("configured").expect("resolve reference");
        assert_eq!(manifests.len(), 1);
        assert_eq!(manifests[0].name, "checkpoint");
        assert_eq!(factory.registered(), vec!["checkpoint"]);
        assert!(resolver.release("checkpoint"));
        assert!(factory.registered().is_empty());
        assert!(!resolver.release("checkpoint"));

        let manifests = resolver
            .resolve_as("custom-name", "configured")
            .expect("resolve with name override");
        assert_eq!(manifests[0].name, "custom-name");
        assert_eq!(factory.registered(), vec!["custom-name"]);
        assert!(resolver.release("custom-name"));
        resolver.replace_references(IndexMap::new());

        fs::remove_dir_all(root).expect("cleanup");
    }
}
