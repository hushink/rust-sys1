use crate::pull_model;
use indexmap::IndexMap;
use std::sync::Arc;
use sys1_core::{ModelError, ModelManifest, ModelResolver};
use sys1_hub::Hub;
use sys1_laya::{
    LayaFactory as CheckpointFactory, LayaOptions as ModelOptions, register_checkpoint_bundle,
};

/// Resolves hub references into decision-model checkpoints and registers them
/// with the factory used by the running registry.
pub struct HubModelResolver {
    hub: Hub,
    factory: Arc<CheckpointFactory>,
    options: ModelOptions,
    /// `[models]` aliases from `sys1.toml`, so the admin API can accept the
    /// same names as the CLI.
    references: IndexMap<String, String>,
}

impl HubModelResolver {
    pub fn new(
        hub: Hub,
        factory: Arc<CheckpointFactory>,
        options: ModelOptions,
        references: IndexMap<String, String>,
    ) -> Self {
        Self {
            hub,
            factory,
            options,
            references,
        }
    }
}

impl ModelResolver for HubModelResolver {
    fn resolve(&self, reference: &str) -> Result<Vec<ModelManifest>, ModelError> {
        let reference = self
            .references
            .get(reference)
            .map(String::as_str)
            .unwrap_or(reference);
        let pulled = pull_model(&self.hub, reference).map_err(|error| {
            ModelError::load(format!("cannot resolve `{reference}`: {error:#}"))
        })?;
        register_checkpoint_bundle(
            &self.factory,
            &pulled.manifest,
            &pulled.directory,
            self.options.clone(),
            None,
        )
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
        let hub = Hub::new(sys1_hub::HubConfig::new(root.join("cache"))).expect("hub");
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

        fs::remove_dir_all(root).expect("cleanup");
    }
}
