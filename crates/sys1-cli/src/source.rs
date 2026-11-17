use crate::{
    has_laya_variants, is_laya_directory, is_laya_model, pull_model, push_manifest,
    variant_directories,
};
use indexmap::IndexMap;
use std::collections::HashSet;
use std::sync::Arc;
use sys1_core::{ModelError, ModelManifest, ModelResolver};
use sys1_hub::{Hub, PulledModel};
use sys1_laya::{LayaFactory, LayaOptions};

/// Resolves hub references into Laya checkpoints and registers them with the
/// factory used by the running registry.
pub struct LayaResolver {
    hub: Hub,
    factory: Arc<LayaFactory>,
    options: LayaOptions,
    /// `[models]` aliases from `sys1.toml`, so the admin API can accept the
    /// same names as the CLI.
    references: IndexMap<String, String>,
}

impl LayaResolver {
    pub fn new(
        hub: Hub,
        factory: Arc<LayaFactory>,
        options: LayaOptions,
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

impl ModelResolver for LayaResolver {
    fn resolve(&self, reference: &str) -> Result<Vec<ModelManifest>, ModelError> {
        let reference = self
            .references
            .get(reference)
            .map(String::as_str)
            .unwrap_or(reference);
        let pulled = pull_model(&self.hub, reference).map_err(|error| {
            ModelError::load(format!("cannot resolve `{reference}`: {error:#}"))
        })?;
        if !is_laya_model(&pulled.manifest) {
            return Err(ModelError::unsupported(format!(
                "`{reference}` does not look like a Laya safetensors checkpoint"
            )));
        }

        let mut manifests = Vec::new();
        let mut seen = HashSet::new();
        register_pulled(
            &self.factory,
            &pulled,
            self.options.clone(),
            None,
            &mut manifests,
            &mut seen,
        );
        Ok(manifests)
    }

    fn release(&self, name: &str) -> bool {
        self.factory.unregister(name)
    }
}

/// Register a materialized Laya bundle with the factory and collect the
/// manifests that should enter the catalog. A bundle with `multilingual/` and
/// `typed-decisions/` subdirectories becomes a router plus three variants.
#[allow(clippy::too_many_arguments)]
pub fn register_pulled(
    factory: &LayaFactory,
    pulled: &PulledModel,
    options: LayaOptions,
    name: Option<&str>,
    manifests: &mut Vec<ModelManifest>,
    seen: &mut HashSet<String>,
) {
    let mut manifest = pulled.manifest.clone();
    manifest.family = "laya".to_owned();
    manifest.backend = "candle".to_owned();
    manifest.format = "safetensors".to_owned();
    if let Some(name) = name {
        // A CLI-assigned name replaces the repository name; repository aliases
        // no longer identify this catalog entry.
        manifest.name = name.to_owned();
        manifest.aliases.clear();
    }

    if has_laya_variants(&pulled.directory) {
        factory.register_router(&manifest.name, &pulled.directory, options.clone());
        for (variant_name, directory) in variant_directories(&pulled.directory) {
            if !is_laya_directory(&directory) {
                continue;
            }
            factory.register(variant_name, &directory, options.clone());
            let mut variant = manifest.clone();
            variant.name = variant_name.to_owned();
            variant.aliases.clear();
            variant.description = format!("{} ({variant_name})", manifest.description);
            push_manifest(manifests, seen, variant);
        }
        manifest.aliases.clear();
        push_manifest(manifests, seen, manifest);
    } else {
        factory.register(&manifest.name, &pulled.directory, options.clone());
        push_manifest(manifests, seen, manifest);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::path::{Path, PathBuf};
    use sys1_core::ModelFile;

    fn checkpoint(directory: &Path) {
        fs::create_dir_all(directory.join("encoder")).expect("encoder dir");
        fs::write(directory.join("model.safetensors"), b"").expect("weights");
        fs::write(directory.join("encoder/config.json"), b"{}").expect("encoder config");
        fs::write(directory.join("rl_agent_config.json"), b"{}").expect("rl config");
        fs::write(directory.join("tokenizer.json"), b"{}").expect("tokenizer");
    }

    fn temp_bundle(tag: &str) -> PathBuf {
        let directory =
            std::env::temp_dir().join(format!("sys1-source-{}-{tag}", std::process::id()));
        let _ = fs::remove_dir_all(&directory);
        fs::create_dir_all(&directory).expect("bundle dir");
        checkpoint(&directory);
        directory
    }

    fn pulled(name: &str, directory: PathBuf) -> PulledModel {
        PulledModel {
            manifest: ModelManifest {
                schema_version: 1,
                name: name.to_owned(),
                description: "test".to_owned(),
                release_date: "2026-01-01".to_owned(),
                family: "laya".to_owned(),
                variant: None,
                backend: "candle".to_owned(),
                format: "safetensors".to_owned(),
                version: "test".to_owned(),
                aliases: vec!["repo-alias".to_owned()],
                files: vec![ModelFile {
                    path: "model.safetensors".to_owned(),
                    url: String::new(),
                    sha256: None,
                    git_sha1: None,
                    size: 1,
                }],
            },
            directory,
            files: IndexMap::new(),
        }
    }

    #[test]
    fn registers_a_single_checkpoint() {
        let directory = temp_bundle("single");
        let factory = LayaFactory::new();
        let mut manifests = Vec::new();
        let mut seen = HashSet::new();
        register_pulled(
            &factory,
            &pulled("repo", directory.clone()),
            LayaOptions::default(),
            Some("laya-latest"),
            &mut manifests,
            &mut seen,
        );

        assert_eq!(manifests.len(), 1);
        assert_eq!(manifests[0].name, "laya-latest");
        assert!(manifests[0].aliases.is_empty());
        assert_eq!(factory.registered(), vec!["laya-latest"]);
        assert!(factory.unregister("laya-latest"));
        assert!(factory.registered().is_empty());
        fs::remove_dir_all(directory).expect("cleanup");
    }

    #[test]
    fn registers_router_and_variants() {
        let root = temp_bundle("variants");
        checkpoint(&root.join("multilingual"));
        checkpoint(&root.join("typed-decisions"));

        let factory = LayaFactory::new();
        let mut manifests = Vec::new();
        let mut seen = HashSet::new();
        register_pulled(
            &factory,
            &pulled("laya", root.clone()),
            LayaOptions::default(),
            None,
            &mut manifests,
            &mut seen,
        );

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
}
