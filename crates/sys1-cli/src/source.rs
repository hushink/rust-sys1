use crate::config::ModelEntry;
use crate::{pull_base, pull_model};
use indexmap::IndexMap;
use std::sync::{Arc, RwLock};
use sys1_core::{ModelError, ModelManifest, ModelResolver};
use sys1_families::{
    BundleOptions, FamilyFactory, FamilyOptions, register_bundle, resolve_bundle_overrides,
};
use sys1_hub::Hub;
use sys1_laya::LayaOptions;

/// Resolves hub references into decision-model checkpoints and registers them
/// with the factory used by the running registry.
pub struct HubModelResolver {
    hub: Arc<Hub>,
    factory: Arc<FamilyFactory>,
    options: LayaOptions,
    families: FamilyOptions,
    /// `[models]` entries from `sys1.toml`, so the admin API can accept the
    /// same aliases and per-model family overrides as the CLI. Replaced
    /// atomically on config reload.
    models: RwLock<IndexMap<String, ModelEntry>>,
}

impl HubModelResolver {
    pub fn new(
        hub: Arc<Hub>,
        factory: Arc<FamilyFactory>,
        options: LayaOptions,
        families: FamilyOptions,
        models: IndexMap<String, ModelEntry>,
    ) -> Self {
        Self {
            hub,
            factory,
            options,
            families,
            models: RwLock::new(models),
        }
    }

    pub fn replace_models(&self, models: IndexMap<String, ModelEntry>) {
        *self
            .models
            .write()
            .unwrap_or_else(|poisoned| poisoned.into_inner()) = models;
    }

    fn resolve_named(
        &self,
        reference: &str,
        name: Option<&str>,
    ) -> Result<Vec<ModelManifest>, ModelError> {
        let entry = {
            let models = self
                .models
                .read()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            name.and_then(|name| models.get(name))
                .or_else(|| models.get(reference))
                .cloned()
        };
        let (reference, families) = match entry {
            Some(entry) => (entry.reference().to_owned(), entry.families(&self.families)),
            None => (reference.to_owned(), self.families.clone()),
        };
        let pulled = pull_model(self.hub.as_ref(), &reference).map_err(|error| {
            ModelError::load(format!("cannot resolve `{reference}`: {error:#}"))
        })?;
        let overrides = resolve_bundle_overrides(&pulled.directory, &families, |base_reference| {
            pull_base(self.hub.as_ref(), base_reference)
        })?;
        let bundle = BundleOptions {
            families,
            overrides,
        };
        register_bundle(
            &self.factory,
            &pulled.manifest,
            &pulled.directory,
            &self.options,
            &bundle,
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
        self.factory.release(name)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::path::Path;
    use sys1_families::{is_supported_directory, local_manifest};

    fn laya_checkpoint(directory: &Path) {
        fs::create_dir_all(directory.join("encoder")).expect("encoder dir");
        fs::write(directory.join("model.safetensors"), b"x").expect("weights");
        fs::write(directory.join("encoder/config.json"), b"{}").expect("encoder config");
        fs::write(directory.join("rl_agent_config.json"), b"{}").expect("rl config");
        fs::write(directory.join("tokenizer.json"), b"{}").expect("tokenizer");
    }

    fn nli_checkpoint(directory: &Path) {
        fs::write(
            directory.join("config.json"),
            r#"{"model_type":"modernbert","hidden_size":8,"num_hidden_layers":1,"num_attention_heads":1,"intermediate_size":16,"max_position_embeddings":32,"label2id":{"entailment":0,"not_entailment":1},"cls_token_id":1,"sep_token_id":2,"pad_token_id":0}"#,
        )
        .expect("config");
        fs::write(directory.join("model.safetensors"), b"x").expect("weights");
        fs::write(directory.join("tokenizer.json"), b"{}").expect("tokenizer");
    }

    fn von_checkpoint(directory: &Path) {
        nli_checkpoint(directory);
        fs::write(directory.join("option_marker.pt"), b"x").expect("marker weights");
    }

    #[test]
    fn resolves_configured_names_and_releases_them() {
        let root =
            std::env::temp_dir().join(format!("sys1-resolver-{}-resolve", std::process::id()));
        let directory = root.join("checkpoint");
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(&directory).expect("checkpoint dir");
        laya_checkpoint(&directory);

        let factory = Arc::new(FamilyFactory::new());
        let hub = Arc::new(Hub::new(sys1_hub::HubConfig::new(root.join("cache"))).expect("hub"));
        let mut references = IndexMap::new();
        references.insert(
            "configured".to_owned(),
            ModelEntry::Reference(format!("local:{}", directory.display())),
        );
        let resolver = HubModelResolver::new(
            hub,
            factory,
            LayaOptions::default(),
            FamilyOptions::default(),
            references,
        );

        let manifests = resolver.resolve("configured").expect("resolve reference");
        assert_eq!(manifests.len(), 1);
        assert_eq!(manifests[0].name, "checkpoint");
        assert!(resolver.release("checkpoint"));
        assert!(!resolver.release("checkpoint"));

        let manifests = resolver
            .resolve_as("custom-name", "configured")
            .expect("resolve with name override");
        assert_eq!(manifests[0].name, "custom-name");
        assert!(resolver.release("custom-name"));
        resolver.replace_models(IndexMap::new());

        fs::remove_dir_all(root).expect("cleanup");
    }

    #[test]
    fn recognises_von_before_nli() {
        let root = std::env::temp_dir().join(format!("sys1-resolver-{}-von", std::process::id()));
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(&root).expect("von dir");
        von_checkpoint(&root);
        assert!(is_supported_directory(&root));
        let (manifest, _) = local_manifest(&root).expect("manifest");
        assert_eq!(manifest.family, "von");
        fs::remove_dir_all(root).expect("cleanup");
    }

    #[test]
    fn recognises_nli_directories() {
        let root = std::env::temp_dir().join(format!("sys1-resolver-{}-nli", std::process::id()));
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(&root).expect("nli dir");
        nli_checkpoint(&root);
        assert!(is_supported_directory(&root));
        let (manifest, files) = local_manifest(&root).expect("manifest");
        assert_eq!(manifest.family, "nli");
        assert_eq!(manifest.backend, "candle");
        assert!(files.contains_key("model.safetensors"));
        fs::remove_dir_all(root).expect("cleanup");
    }
}
