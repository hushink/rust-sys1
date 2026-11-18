use crate::config::{self, FileConfig};
use crate::source::HubModelResolver;
use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use sys1_core::{ModelRegistry, ModelResolver};
use sys1_server::{ConfigReload, ConfigReloader, ReloadError};

/// Applies `[models]` additions/removals from `sys1.toml` to a running server.
///
/// Hub, server, and existing model-reference changes require a restart; they
/// are reported as [`ReloadError::RestartRequired`] without mutating state.
pub struct HubConfigReloader {
    path: PathBuf,
    resolver: Arc<HubModelResolver>,
    registry: Arc<ModelRegistry>,
    config: Mutex<FileConfig>,
    reload_lock: Mutex<()>,
}

impl HubConfigReloader {
    pub fn new(
        path: PathBuf,
        resolver: Arc<HubModelResolver>,
        registry: Arc<ModelRegistry>,
        config: FileConfig,
    ) -> Self {
        Self {
            path,
            resolver,
            registry,
            config: Mutex::new(config),
            reload_lock: Mutex::new(()),
        }
    }

    fn rollback(&self, registered: &[String], resolved: &[String], pre_existing: &HashSet<String>) {
        for name in registered {
            self.registry.unregister(name);
            let _ = self.resolver.release(name);
        }
        for name in resolved {
            if !pre_existing.contains(name) {
                let _ = self.resolver.release(name);
            }
        }
    }
}

impl ConfigReloader for HubConfigReloader {
    fn reload(&self) -> Result<ConfigReload, ReloadError> {
        let _guard = self
            .reload_lock
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let loaded = config::load(Some(&self.path))
            .map_err(|error| ReloadError::Failed(format!("{error:#}")))?;
        let new = loaded.value;
        let old = self
            .config
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .clone();

        let mut restart = Vec::new();
        if new.hub != old.hub {
            restart.push("hub settings changed".to_owned());
        }
        if new.server != old.server {
            restart.push("server settings changed".to_owned());
        }
        if new.families != old.families {
            restart.push("family settings changed".to_owned());
        }
        let changed_models = new
            .models
            .iter()
            .filter_map(|(name, reference)| {
                old.models
                    .get(name)
                    .filter(|old_reference| *old_reference != reference)
                    .map(|_| name.clone())
            })
            .collect::<Vec<_>>();
        if !changed_models.is_empty() {
            restart.push(format!(
                "model references changed: {}",
                changed_models.join(", ")
            ));
        }
        if !restart.is_empty() {
            return Err(ReloadError::RestartRequired(restart));
        }

        let added = new
            .models
            .keys()
            .filter(|name| !old.models.contains_key(*name))
            .cloned()
            .collect::<Vec<_>>();
        let removed = old
            .models
            .keys()
            .filter(|name| !new.models.contains_key(*name))
            .cloned()
            .collect::<Vec<_>>();
        let unchanged = new
            .models
            .keys()
            .filter(|name| old.models.get(*name) == new.models.get(*name))
            .cloned()
            .collect::<Vec<_>>();

        let pre_existing = self
            .registry
            .manifests()
            .into_iter()
            .map(|manifest| manifest.name)
            .collect::<HashSet<_>>();
        let mut resolved = Vec::new();
        let mut registered = Vec::new();

        for name in &added {
            let reference = new
                .models
                .get(name)
                .expect("added name came from the new model map");
            let manifests = match self.resolver.resolve_as(name, reference.reference()) {
                Ok(manifests) => manifests,
                Err(error) => {
                    self.rollback(&registered, &resolved, &pre_existing);
                    return Err(ReloadError::Failed(format!("add `{name}`: {error}")));
                }
            };
            for manifest in manifests {
                let manifest_name = manifest.name.clone();
                resolved.push(manifest_name.clone());
                if let Err(error) = self.registry.register_configured(manifest) {
                    self.rollback(&registered, &resolved, &pre_existing);
                    return Err(ReloadError::Failed(format!("add `{name}`: {error}")));
                }
                registered.push(manifest_name);
            }
        }

        for name in &removed {
            if self.registry.unregister(name) {
                let _ = self.resolver.release(name);
            }
        }

        self.resolver.replace_models(new.models.clone());
        *self
            .config
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner()) = new;

        Ok(ConfigReload {
            config_path: Some(display_path(&self.path)),
            added,
            removed,
            unchanged,
        })
    }
}

fn display_path(path: &Path) -> String {
    let text = path.display().to_string();
    if let Some(rest) = text.strip_prefix(r"\\?\UNC\") {
        format!(r"\\{rest}")
    } else if let Some(rest) = text.strip_prefix(r"\\?\") {
        rest.to_owned()
    } else {
        text
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use sys1_core::RegistryConfig;
    use sys1_families::{FamilyFactory, FamilyOptions};
    use sys1_hub::{Hub, HubConfig};
    use sys1_laya::LayaOptions as ModelOptions;

    fn checkpoint(directory: &Path) {
        fs::create_dir_all(directory.join("encoder")).expect("encoder dir");
        fs::write(directory.join("model.safetensors"), b"x").expect("weights");
        fs::write(directory.join("encoder/config.json"), b"{}").expect("encoder config");
        fs::write(directory.join("rl_agent_config.json"), b"{}").expect("rl config");
        fs::write(directory.join("tokenizer.json"), b"{}").expect("tokenizer");
    }

    fn write_config(path: &Path, model: Option<&Path>, extra_server: &str) {
        let models = model
            .map(|directory| {
                format!(
                    "[models]\nlaya = \"local:{}\"\n",
                    directory.display().to_string().replace('\\', "/")
                )
            })
            .unwrap_or_default();
        fs::write(
            path,
            format!(
                "[hub]\ncache_dir = \"cache\"\n\n[server]\ndevice = \"cpu\"\n{extra_server}\n{models}"
            ),
        )
        .expect("config");
    }

    fn setup(
        tag: &str,
    ) -> (
        PathBuf,
        Arc<ModelRegistry>,
        Arc<HubModelResolver>,
        HubConfigReloader,
    ) {
        let root = std::env::temp_dir().join(format!("sys1-reload-{}-{tag}", std::process::id()));
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(&root).expect("root");
        let checkpoint_dir = root.join("checkpoint");
        fs::create_dir_all(&checkpoint_dir).expect("checkpoint");
        checkpoint(&checkpoint_dir);
        let config_path = root.join("sys1.toml");
        write_config(&config_path, None, "");

        let loaded = config::load(Some(&config_path)).expect("initial config");
        let factory = Arc::new(FamilyFactory::new());
        let hub = Arc::new(Hub::new(HubConfig::new(root.join("cache"))).expect("hub"));
        let registry = Arc::new(
            ModelRegistry::new(Vec::new(), factory.clone(), RegistryConfig::default())
                .expect("registry"),
        );
        let resolver = Arc::new(HubModelResolver::new(
            hub,
            factory,
            ModelOptions::default(),
            FamilyOptions::default(),
            loaded.value.models.clone(),
        ));
        let reloader = HubConfigReloader::new(
            config_path.clone(),
            resolver.clone(),
            registry.clone(),
            loaded.value,
        );
        (root, registry, resolver, reloader)
    }

    #[test]
    fn applies_model_additions_and_removals() {
        let (root, registry, resolver, reloader) = setup("models");
        let checkpoint_dir = root.join("checkpoint");

        write_config(&root.join("sys1.toml"), Some(&checkpoint_dir), "");
        let report = reloader.reload().expect("add model");
        assert_eq!(report.added, vec!["laya".to_owned()]);
        assert!(report.removed.is_empty());
        assert!(registry.resolve("laya").is_some());
        assert!(resolver.resolve("laya").is_ok());

        write_config(&root.join("sys1.toml"), None, "");
        let report = reloader.reload().expect("remove model");
        assert_eq!(report.removed, vec!["laya".to_owned()]);
        assert!(registry.resolve("laya").is_none());

        fs::remove_dir_all(root).expect("cleanup");
    }

    #[test]
    fn rejects_settings_and_reference_changes_without_mutation() {
        let (root, registry, _resolver, reloader) = setup("restart");
        let checkpoint_dir = root.join("checkpoint");

        write_config(&root.join("sys1.toml"), None, "metrics = true\n");
        let error = reloader.reload().expect_err("server change");
        assert!(matches!(error, ReloadError::RestartRequired(_)));
        assert!(registry.manifests().is_empty());

        write_config(&root.join("sys1.toml"), Some(&checkpoint_dir), "");
        reloader.reload().expect("initial model");
        assert!(registry.resolve("laya").is_some());

        let other = root.join("other-checkpoint");
        fs::create_dir_all(&other).expect("other checkpoint");
        checkpoint(&other);
        write_config(&root.join("sys1.toml"), Some(&other), "");
        let error = reloader.reload().expect_err("reference change");
        match error {
            ReloadError::RestartRequired(changes) => {
                assert!(
                    changes
                        .iter()
                        .any(|change| change.contains("model references")),
                    "{changes:?}"
                );
            }
            other => panic!("unexpected error: {other:?}"),
        }
        assert!(
            registry
                .resolve("laya")
                .expect("old model")
                .1
                .files
                .iter()
                .any(|file| file.path == "model.safetensors")
        );

        fs::remove_dir_all(root).expect("cleanup");
    }

    #[test]
    fn per_model_family_changes_require_restart() {
        let (root, _registry, _resolver, reloader) = setup("model-families");
        let reference = root
            .join("checkpoint")
            .display()
            .to_string()
            .replace('\\', "/");
        let path = root.join("sys1.toml");
        let write = |batch: u32| {
            fs::write(
                &path,
                format!(
                    "[hub]\ncache_dir = \"cache\"\n\n[server]\ndevice = \"cpu\"\n\n\
                     [models.laya]\nreference = \"local:{reference}\"\n\
                     [models.laya.families]\nmax_batch = {batch}\n"
                ),
            )
            .expect("config");
        };

        write(4);
        let report = reloader.reload().expect("add model with overrides");
        assert_eq!(report.added, vec!["laya".to_owned()]);

        write(8);
        let error = reloader.reload().expect_err("override change");
        match error {
            ReloadError::RestartRequired(changes) => {
                assert!(
                    changes
                        .iter()
                        .any(|change| change.contains("model references")),
                    "{changes:?}"
                );
            }
            other => panic!("unexpected error: {other:?}"),
        }

        fs::remove_dir_all(root).expect("cleanup");
    }

    #[test]
    fn family_changes_require_restart() {
        let (root, registry, _resolver, reloader) = setup("families");
        let checkpoint_dir = root.join("checkpoint");

        write_config(
            &root.join("sys1.toml"),
            Some(&checkpoint_dir),
            "[families.decima]\nchoice_fit = true\n",
        );
        let error = reloader.reload().expect_err("family change");
        match error {
            ReloadError::RestartRequired(changes) => {
                assert!(
                    changes
                        .iter()
                        .any(|change| change.contains("family settings")),
                    "{changes:?}"
                );
            }
            other => panic!("unexpected error: {other:?}"),
        }
        assert!(registry.manifests().is_empty());

        fs::remove_dir_all(root).expect("cleanup");
    }
}
