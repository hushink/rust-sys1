use crate::source::SourceContext;
use crate::{HubError, ModelRef, ModelSource, download};
use std::path::Path;
use sys1_core::{ModelFile, ModelManifest};
use url::Url;

/// A static HTTP model center: a `model.json` plus the files it references.
pub struct HttpSource;

impl ModelSource for HttpSource {
    fn scheme(&self) -> &'static str {
        "http"
    }

    fn matches(&self, reference: &ModelRef) -> bool {
        matches!(reference.scheme.as_str(), "http" | "https")
    }

    fn resolve(
        &self,
        reference: &ModelRef,
        context: &SourceContext<'_>,
    ) -> Result<ModelManifest, HubError> {
        let manifest_url = if reference.location.ends_with(".json") {
            reference.location.clone()
        } else {
            format!("{}/model.json", reference.location.trim_end_matches('/'))
        };
        let headers = context.config.http_headers();
        let (bytes, _) = get_bytes(context, &manifest_url, &headers)?;
        let mut manifest: ModelManifest =
            serde_json::from_slice(&bytes).map_err(|error| HubError::Json {
                url: manifest_url.clone(),
                message: error.to_string(),
            })?;
        manifest.validate()?;
        resolve_relative_urls(&manifest_url, &mut manifest)?;
        Ok(manifest)
    }

    fn fetch(
        &self,
        _reference: &ModelRef,
        file: &ModelFile,
        destination: &Path,
        context: &SourceContext<'_>,
    ) -> Result<(), HubError> {
        let headers = context.config.http_headers();
        download::download_http(
            context,
            &file.url,
            &headers,
            destination,
            Some(file.size),
            file.sha256.as_deref(),
        )
    }
}

pub(crate) fn get_bytes(
    context: &SourceContext<'_>,
    url: &str,
    headers: &[(String, String)],
) -> Result<(Vec<u8>, reqwest::header::HeaderMap), HubError> {
    let mut request = context
        .client
        .get(url)
        .header(reqwest::header::ACCEPT_ENCODING, "identity")
        .header(
            reqwest::header::USER_AGENT,
            concat!("rust-sys1/", env!("CARGO_PKG_VERSION")),
        );
    for (name, value) in headers {
        request = request.header(name.as_str(), value.as_str());
    }

    let response = request.send().map_err(|error| HubError::Http {
        url: url.to_owned(),
        message: error.to_string(),
    })?;
    let status = response.status();
    if !status.is_success() {
        let body = response
            .text()
            .unwrap_or_default()
            .chars()
            .take(300)
            .collect::<String>();
        return Err(HubError::HttpStatus {
            url: url.to_owned(),
            status: status.as_u16(),
            message: body,
        });
    }
    let headers = response.headers().clone();
    let bytes = response
        .bytes()
        .map_err(|error| HubError::Http {
            url: url.to_owned(),
            message: error.to_string(),
        })?
        .to_vec();
    Ok((bytes, headers))
}

fn resolve_relative_urls(manifest_url: &str, manifest: &mut ModelManifest) -> Result<(), HubError> {
    let base = Url::parse(manifest_url).map_err(|error| HubError::Url {
        url: manifest_url.to_owned(),
        message: error.to_string(),
    })?;

    for file in &mut manifest.files {
        if file.url.trim().is_empty() {
            file.url = base
                .join(&file.path)
                .map_err(|error| HubError::Url {
                    url: file.path.clone(),
                    message: error.to_string(),
                })?
                .to_string();
        } else if Url::parse(&file.url).is_err() {
            file.url = base
                .join(&file.url)
                .map_err(|error| HubError::Url {
                    url: file.url.clone(),
                    message: error.to_string(),
                })?
                .to_string();
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use sys1_core::ModelFile;

    #[test]
    fn an_omitted_url_falls_back_to_the_manifest_path() {
        let mut manifest: ModelManifest = serde_json::from_str(
            r#"{
                "schema_version": 1,
                "name": "m",
                "description": "d",
                "release_date": "2026-01-01",
                "family": "laya",
                "backend": "candle",
                "format": "safetensors",
                "version": "1",
                "files": [{"path": "encoder/config.json", "size": 2}]
            }"#,
        )
        .expect("manifest without url");
        assert_eq!(manifest.files[0].url, "");
        resolve_relative_urls("https://example.com/models/laya/model.json", &mut manifest)
            .expect("relative URL resolves");
        assert_eq!(
            manifest.files[0].url,
            "https://example.com/models/laya/encoder/config.json"
        );
    }

    #[test]
    fn resolves_relative_file_urls() {
        let mut manifest = ModelManifest {
            schema_version: 1,
            name: "m".to_owned(),
            description: "d".to_owned(),
            release_date: "2026-01-01".to_owned(),
            family: "f".to_owned(),
            variant: None,
            backend: "candle".to_owned(),
            format: "safetensors".to_owned(),
            version: "1".to_owned(),
            aliases: Vec::new(),
            files: vec![
                ModelFile {
                    path: "files/a.bin".to_owned(),
                    url: String::new(),
                    sha256: None,
                    git_sha1: None,
                    size: 1,
                },
                ModelFile {
                    path: "b.bin".to_owned(),
                    url: "files/b.bin".to_owned(),
                    sha256: None,
                    git_sha1: None,
                    size: 1,
                },
                ModelFile {
                    path: "c.bin".to_owned(),
                    url: "https://cdn.example/c.bin".to_owned(),
                    sha256: None,
                    git_sha1: None,
                    size: 1,
                },
            ],
        };

        resolve_relative_urls("https://example.com/registry/model.json", &mut manifest)
            .expect("relative URLs resolve");
        assert_eq!(
            manifest.files[0].url,
            "https://example.com/registry/files/a.bin"
        );
        assert_eq!(
            manifest.files[1].url,
            "https://example.com/registry/files/b.bin"
        );
        assert_eq!(manifest.files[2].url, "https://cdn.example/c.bin");
    }
}
