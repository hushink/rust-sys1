use crate::http::get_bytes;
use crate::source::{infer_format, model_name_from_repo};
use crate::{HubError, ModelRef, ModelSource, SourceContext, download};
use percent_encoding::{AsciiSet, NON_ALPHANUMERIC, utf8_percent_encode};
use reqwest::header::HeaderMap;
use serde::Deserialize;
use serde_json::Value;
use std::path::Path;
use sys1_core::{ModelFile, ModelManifest};
use url::Url;

/// Hugging Face model source.
pub struct HfSource;

#[derive(Debug, Deserialize)]
struct HfTreeEntry {
    #[serde(rename = "type")]
    kind: String,
    path: String,
    #[serde(default)]
    size: Option<u64>,
    #[serde(default)]
    oid: Option<String>,
    #[serde(default)]
    lfs: Option<HfLfs>,
}

#[derive(Debug, Deserialize)]
struct HfLfs {
    oid: String,
    #[serde(default)]
    size: Option<u64>,
}

impl ModelSource for HfSource {
    fn scheme(&self) -> &'static str {
        "hf"
    }

    fn resolve(
        &self,
        reference: &ModelRef,
        context: &SourceContext<'_>,
    ) -> Result<ModelManifest, HubError> {
        let repo = &reference.location;
        let revision = reference.revision.as_deref().unwrap_or("main");
        let subfolder = reference.subfolder.as_deref();
        let endpoint = context.config.hf_endpoint.trim_end_matches('/');
        let headers = context.config.hf_headers();

        let mut entries = Vec::new();
        let mut url = format!(
            "{endpoint}/api/models/{repo}/tree/{}?recursive=true&expand=false",
            encode_revision(revision)
        );
        let mut pages = 0_usize;
        loop {
            pages += 1;
            if pages > 1000 {
                return Err(HubError::InvalidData {
                    url: url.clone(),
                    message: "too many pagination pages".to_owned(),
                });
            }
            let (bytes, response_headers) = get_bytes(context, &url, &headers)?;
            let page: Vec<HfTreeEntry> =
                serde_json::from_slice(&bytes).map_err(|error| HubError::Json {
                    url: url.clone(),
                    message: error.to_string(),
                })?;
            entries.extend(page);

            let Some(next) = next_link(&response_headers) else {
                break;
            };
            url = join_url(&url, &next)?;
        }

        let release_date = fetch_release_date(context, endpoint, repo, &headers)
            .unwrap_or_else(|| "1970-01-01".to_owned());

        let mut files = Vec::new();
        for entry in entries {
            if entry.kind != "file" {
                continue;
            }
            let Some((relative, full_path)) = relative_path(&entry.path, subfolder) else {
                continue;
            };
            let size = entry
                .lfs
                .as_ref()
                .and_then(|lfs| lfs.size)
                .or(entry.size)
                .unwrap_or(0);
            if size == 0 {
                continue;
            }
            let sha256 = entry.lfs.as_ref().map(|lfs| lfs.oid.clone());
            let git_sha1 = entry.oid.clone();
            let url = format!(
                "{endpoint}/{repo}/resolve/{}/{}",
                encode_revision(revision),
                encode_path(&full_path)
            );
            files.push(ModelFile {
                path: relative,
                url,
                sha256,
                git_sha1,
                size,
            });
        }

        if files.is_empty() {
            return Err(HubError::InvalidData {
                url: format!("{endpoint}/api/models/{repo}"),
                message: "no files found".to_owned(),
            });
        }

        let (backend, format) = infer_format(&files);
        let name = model_name_from_repo(repo, subfolder);
        let mut description = format!("Hugging Face model {repo}");
        if let Some(subfolder) = subfolder {
            description.push_str(&format!(" ({subfolder})"));
        }

        let manifest = ModelManifest {
            schema_version: 1,
            name,
            description,
            release_date,
            family: "hf".to_owned(),
            variant: subfolder.map(str::to_owned),
            backend,
            format,
            version: revision.to_owned(),
            aliases: Vec::new(),
            files,
        };
        manifest.validate()?;
        Ok(manifest)
    }

    fn fetch(
        &self,
        _reference: &ModelRef,
        file: &ModelFile,
        destination: &Path,
        context: &SourceContext<'_>,
    ) -> Result<(), HubError> {
        let headers = context.config.hf_headers();
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

fn relative_path(path: &str, subfolder: Option<&str>) -> Option<(String, String)> {
    let path = path.trim_start_matches('/');
    if path.is_empty() {
        return None;
    }
    match subfolder {
        Some(subfolder) => {
            let prefix = format!("{}/", subfolder.trim_matches('/'));
            let relative = path.strip_prefix(&prefix)?;
            if relative.is_empty() {
                None
            } else {
                Some((relative.to_owned(), path.to_owned()))
            }
        }
        None => Some((path.to_owned(), path.to_owned())),
    }
}

const COMPONENT: &AsciiSet = &NON_ALPHANUMERIC
    .remove(b'-')
    .remove(b'.')
    .remove(b'_')
    .remove(b'~');

fn encode_revision(revision: &str) -> String {
    utf8_percent_encode(revision, COMPONENT).to_string()
}

fn encode_path(path: &str) -> String {
    path.split('/')
        .map(|segment| utf8_percent_encode(segment, COMPONENT).to_string())
        .collect::<Vec<_>>()
        .join("/")
}

fn next_link(headers: &HeaderMap) -> Option<String> {
    let value = headers.get(reqwest::header::LINK)?.to_str().ok()?;
    for part in value.split(',') {
        let part = part.trim();
        if !part.contains("rel=\"next\"") {
            continue;
        }
        let start = part.find('<')? + 1;
        let end = part[start..].find('>')? + start;
        return Some(part[start..end].to_owned());
    }
    None
}

fn join_url(base: &str, next: &str) -> Result<String, HubError> {
    if Url::parse(next).is_ok() {
        return Ok(next.to_owned());
    }
    let base = Url::parse(base).map_err(|error| HubError::Url {
        url: base.to_owned(),
        message: error.to_string(),
    })?;
    base.join(next)
        .map(|url| url.to_string())
        .map_err(|error| HubError::Url {
            url: next.to_owned(),
            message: error.to_string(),
        })
}

fn fetch_release_date(
    context: &SourceContext<'_>,
    endpoint: &str,
    repo: &str,
    headers: &[(String, String)],
) -> Option<String> {
    let url = format!("{endpoint}/api/models/{repo}");
    let (bytes, _) = get_bytes(context, &url, headers).ok()?;
    let value: Value = serde_json::from_slice(&bytes).ok()?;
    for key in ["lastModified", "createdAt", "created_at"] {
        if let Some(value) = value.get(key).and_then(Value::as_str)
            && value.len() >= 10
            && is_valid_date(&value[..10])
        {
            return Some(value[..10].to_owned());
        }
    }
    None
}

fn is_valid_date(value: &str) -> bool {
    let bytes = value.as_bytes();
    bytes.len() == 10
        && bytes[4] == b'-'
        && bytes[7] == b'-'
        && bytes
            .iter()
            .enumerate()
            .all(|(index, byte)| index == 4 || index == 7 || byte.is_ascii_digit())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn encodes_revision_and_keeps_path_slashes() {
        assert_eq!(encode_revision("pr/foo"), "pr%2Ffoo");
        assert_eq!(encode_path("a/b c.txt"), "a/b%20c.txt");
        assert_eq!(encode_path("model.safetensors"), "model.safetensors");
    }

    #[test]
    fn subfolder_relative_path() {
        assert_eq!(
            relative_path("multi/a.bin", Some("multi")),
            Some(("a.bin".to_owned(), "multi/a.bin".to_owned()))
        );
        assert_eq!(relative_path("other/a.bin", Some("multi")), None);
        assert_eq!(
            relative_path("a.bin", None),
            Some(("a.bin".to_owned(), "a.bin".to_owned()))
        );
    }
}
