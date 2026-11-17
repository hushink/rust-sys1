use crate::http::get_bytes;
use crate::source::{infer_format, model_name_from_repo};
use crate::{HubError, ModelRef, ModelSource, SourceContext, download};
use serde::Deserialize;
use std::collections::HashSet;
use std::path::Path;
use sys1_core::{ModelFile, ModelManifest};
use url::form_urlencoded;

const TRUNCATION_LIMIT: usize = 3000;
const MAX_DEPTH: usize = 32;

/// ModelScope model source.
pub struct ModelScopeSource;

#[derive(Debug, Deserialize)]
struct MsResponse {
    #[serde(rename = "Data", default)]
    data: Option<MsData>,
    #[serde(rename = "Success", default)]
    success: Option<bool>,
    #[serde(rename = "Message", default)]
    message: Option<String>,
}

#[derive(Debug, Deserialize)]
struct MsData {
    #[serde(rename = "Files", default)]
    files: Vec<MsFile>,
}

#[derive(Clone, Debug, Deserialize)]
struct MsFile {
    #[serde(rename = "Path")]
    path: String,
    #[serde(rename = "Size", default)]
    size: Option<u64>,
    #[serde(rename = "Sha256", default)]
    sha256: Option<String>,
    #[serde(rename = "Type")]
    kind: String,
}

impl ModelSource for ModelScopeSource {
    fn scheme(&self) -> &'static str {
        "modelscope"
    }

    fn resolve(
        &self,
        reference: &ModelRef,
        context: &SourceContext<'_>,
    ) -> Result<ModelManifest, HubError> {
        let repo = &reference.location;
        let revision = reference.revision.as_deref().unwrap_or("master");
        let subfolder = reference.subfolder.as_deref();
        let root = subfolder.unwrap_or("");
        let endpoint = context.config.modelscope_endpoint.trim_end_matches('/');

        let mut files = Vec::new();
        let mut seen = HashSet::new();
        walk(context, repo, revision, root, 0, &mut files, &mut seen)?;

        let mut manifest_files = Vec::new();
        for file in files {
            if file.kind != "blob" {
                continue;
            }
            let relative = match subfolder {
                Some(subfolder) => {
                    let prefix = format!("{}/", subfolder.trim_matches('/'));
                    let Some(relative) = file.path.strip_prefix(&prefix) else {
                        continue;
                    };
                    if relative.is_empty() {
                        continue;
                    }
                    relative.to_owned()
                }
                None => file.path.clone(),
            };
            let Some(size) = file.size else {
                continue;
            };
            if size == 0 {
                continue;
            }
            let sha256 = file.sha256.filter(|value| !value.is_empty());
            let url = download_url(endpoint, repo, revision, &file.path);
            manifest_files.push(ModelFile {
                path: relative,
                url,
                sha256,
                git_sha1: None,
                size,
            });
        }

        if manifest_files.is_empty() {
            return Err(HubError::InvalidData {
                url: format!(
                    "{}/api/v1/models/{repo}/repo/files",
                    context.config.modelscope_endpoint.trim_end_matches('/')
                ),
                message: "no files found".to_owned(),
            });
        }

        let (backend, format) = infer_format(&manifest_files);
        let name = model_name_from_repo(repo, subfolder);
        let mut description = format!("ModelScope model {repo}");
        if let Some(subfolder) = subfolder {
            description.push_str(&format!(" ({subfolder})"));
        }

        let manifest = ModelManifest {
            schema_version: 1,
            name,
            description,
            release_date: "1970-01-01".to_owned(),
            family: "modelscope".to_owned(),
            variant: subfolder.map(str::to_owned),
            backend,
            format,
            version: revision.to_owned(),
            aliases: Vec::new(),
            files: manifest_files,
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
        let headers = context.config.modelscope_headers();
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

fn walk(
    context: &SourceContext<'_>,
    repo: &str,
    revision: &str,
    root: &str,
    depth: usize,
    files: &mut Vec<MsFile>,
    seen: &mut HashSet<String>,
) -> Result<(), HubError> {
    if depth > MAX_DEPTH {
        return Err(HubError::InvalidData {
            url: root.to_string(),
            message: "ModelScope directory nesting is too deep".to_owned(),
        });
    }

    let page = fetch_page(context, repo, revision, root, true)?;
    if page.len() < TRUNCATION_LIMIT {
        for file in page {
            if file.kind == "blob" && seen.insert(file.path.clone()) {
                files.push(file);
            }
        }
        return Ok(());
    }

    // The recursive listing is silently truncated at 3000 entries. Split by
    // directory using a non-recursive listing.
    let top = fetch_page(context, repo, revision, root, false)?;
    let mut had_tree = false;
    for file in top {
        match file.kind.as_str() {
            "blob" => {
                if seen.insert(file.path.clone()) {
                    files.push(file);
                }
            }
            "tree" => {
                had_tree = true;
                walk(context, repo, revision, &file.path, depth + 1, files, seen)?;
            }
            _ => {}
        }
    }

    if !had_tree {
        return Err(HubError::InvalidData {
            url: root.to_string(),
            message:
                "ModelScope listing is truncated at 3000 entries and has no subdirectories to split"
                    .to_owned(),
        });
    }

    Ok(())
}

fn fetch_page(
    context: &SourceContext<'_>,
    repo: &str,
    revision: &str,
    root: &str,
    recursive: bool,
) -> Result<Vec<MsFile>, HubError> {
    let endpoint = context.config.modelscope_endpoint.trim_end_matches('/');
    let mut serializer = form_urlencoded::Serializer::new(String::new());
    serializer.append_pair("Revision", revision);
    serializer.append_pair("Root", root);
    serializer.append_pair("Recursive", if recursive { "true" } else { "false" });
    let url = format!(
        "{endpoint}/api/v1/models/{repo}/repo/files?{}",
        serializer.finish()
    );
    let headers = context.config.modelscope_headers();
    let (bytes, _) = get_bytes(context, &url, &headers)?;
    parse_page(&url, &bytes)
}

fn parse_page(url: &str, bytes: &[u8]) -> Result<Vec<MsFile>, HubError> {
    let response: MsResponse = serde_json::from_slice(bytes).map_err(|error| HubError::Json {
        url: url.to_owned(),
        message: error.to_string(),
    })?;
    if response.success == Some(false) {
        return Err(HubError::InvalidData {
            url: url.to_owned(),
            message: response
                .message
                .unwrap_or_else(|| "ModelScope request failed".to_owned()),
        });
    }
    let data = response.data.ok_or_else(|| HubError::InvalidData {
        url: url.to_owned(),
        message: "missing Data field".to_owned(),
    })?;
    Ok(data.files)
}

fn download_url(endpoint: &str, repo: &str, revision: &str, path: &str) -> String {
    let mut serializer = form_urlencoded::Serializer::new(String::new());
    serializer.append_pair("Revision", revision);
    serializer.append_pair("FilePath", path);
    format!(
        "{endpoint}/api/v1/models/{repo}/repo?{}",
        serializer.finish()
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_file_page() {
        let bytes = br#"{"Code":200,"Data":{"Files":[{"Path":"unet/config.json","Size":743,"Sha256":"78f474de6bab3d893868f37be97b636ae65c0df3073ed3256ca458ff599b5f96","Type":"blob"}]},"Message":"success","Success":true}"#;
        let files = parse_page("test", bytes).expect("valid page");
        assert_eq!(files.len(), 1);
        assert_eq!(files[0].path, "unet/config.json");
        assert_eq!(files[0].size, Some(743));
        assert_eq!(files[0].kind, "blob");
    }

    #[test]
    fn rejects_unsuccessful_page() {
        let bytes = br#"{"Data":null,"Message":"nope","Success":false}"#;
        assert!(parse_page("test", bytes).is_err());
    }

    #[test]
    fn download_url_encodes_revision_and_path() {
        let url = download_url(
            "https://modelscope.cn",
            "owner/repo",
            "pr/foo",
            "unet/config.json",
        );
        assert!(url.contains("Revision=pr%2Ffoo"), "{url}");
        assert!(url.contains("FilePath=unet%2Fconfig.json"), "{url}");
    }
}
