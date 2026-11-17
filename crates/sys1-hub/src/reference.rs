use crate::HubError;
use percent_encoding::percent_decode_str;
use std::path::{Path, PathBuf};
use std::str::FromStr;

/// A model reference understood by the hub.
///
/// Supported forms:
/// - `hf:owner/repo[@revision][#subfolder]`
/// - `modelscope:owner/repo[@revision][#subfolder]`
/// - `local:/path/to/model` or an existing filesystem path
/// - `file:///path/to/model`
/// - `https://host/path/model.json` (custom static center)
/// - `scheme:rest` for a custom `ModelSource`
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ModelRef {
    pub scheme: String,
    pub location: String,
    pub revision: Option<String>,
    pub subfolder: Option<String>,
}

impl ModelRef {
    pub fn new(scheme: impl Into<String>, location: impl Into<String>) -> Self {
        Self {
            scheme: scheme.into(),
            location: location.into(),
            revision: None,
            subfolder: None,
        }
    }

    pub fn raw_key(&self) -> String {
        let mut key = format!("{}:{}", self.scheme, self.location);
        if let Some(revision) = &self.revision {
            key.push('@');
            key.push_str(revision);
        }
        if let Some(subfolder) = &self.subfolder {
            key.push('#');
            key.push_str(subfolder);
        }
        key
    }
}

impl FromStr for ModelRef {
    type Err = HubError;

    fn from_str(input: &str) -> Result<Self, Self::Err> {
        let input = input.trim();
        if input.is_empty() {
            return Err(HubError::InvalidReference(input.to_owned()));
        }

        if input.starts_with("http://") || input.starts_with("https://") {
            let scheme = input
                .split_once("://")
                .map(|(scheme, _)| scheme.to_ascii_lowercase())
                .unwrap_or_else(|| "http".to_owned());
            return Ok(Self {
                scheme,
                location: input.to_owned(),
                revision: None,
                subfolder: None,
            });
        }

        if let Some(rest) = strip_prefix_ci(input, "file://") {
            let path = file_url_to_path(rest)?;
            return Ok(Self {
                scheme: "local".to_owned(),
                location: path.to_string_lossy().into_owned(),
                revision: None,
                subfolder: None,
            });
        }

        if let Some(rest) = strip_prefix_ci(input, "hf:") {
            return parse_repo_ref("hf", rest.trim_start_matches("//"));
        }
        if let Some(rest) = strip_prefix_ci(input, "huggingface:") {
            return parse_repo_ref("hf", rest.trim_start_matches("//"));
        }
        if let Some(rest) = strip_prefix_ci(input, "modelscope:") {
            return parse_repo_ref("modelscope", rest.trim_start_matches("//"));
        }
        if let Some(rest) = strip_prefix_ci(input, "ms:") {
            return parse_repo_ref("modelscope", rest.trim_start_matches("//"));
        }
        if let Some(rest) = strip_prefix_ci(input, "local:") {
            let location = rest.trim();
            if location.is_empty() {
                return Err(HubError::InvalidReference(input.to_owned()));
            }
            return Ok(Self {
                scheme: "local".to_owned(),
                location: location.to_owned(),
                revision: None,
                subfolder: None,
            });
        }

        if Path::new(input).exists() {
            return Ok(Self {
                scheme: "local".to_owned(),
                location: input.to_owned(),
                revision: None,
                subfolder: None,
            });
        }

        if let Some((scheme, rest)) = input.split_once(':') {
            let is_windows_drive = scheme.len() == 1
                && scheme.chars().all(|ch| ch.is_ascii_alphabetic())
                && (rest.starts_with('\\') || rest.starts_with('/'));
            if !is_windows_drive && is_valid_scheme(scheme) && !rest.is_empty() {
                return parse_repo_ref(&scheme.to_ascii_lowercase(), rest);
            }
        }

        Err(HubError::InvalidReference(input.to_owned()))
    }
}

fn parse_repo_ref(scheme: &str, rest: &str) -> Result<ModelRef, HubError> {
    let (rest, subfolder) = match rest.split_once('#') {
        Some((rest, subfolder)) => (rest, Some(subfolder.trim().to_owned())),
        None => (rest, None),
    };
    let (location, revision) = match rest.rsplit_once('@') {
        Some((location, revision)) => (location, Some(revision.trim().to_owned())),
        None => (rest, None),
    };

    let location = location.trim();
    if location.is_empty() {
        return Err(HubError::InvalidReference(rest.to_owned()));
    }
    if matches!(revision.as_deref(), Some("")) {
        return Err(HubError::InvalidReference(rest.to_owned()));
    }
    if matches!(subfolder.as_deref(), Some("")) {
        return Err(HubError::InvalidReference(rest.to_owned()));
    }

    Ok(ModelRef {
        scheme: scheme.to_owned(),
        location: location.to_owned(),
        revision,
        subfolder,
    })
}

pub(crate) fn file_url_to_path(rest: &str) -> Result<PathBuf, HubError> {
    let decoded = percent_decode_str(rest)
        .decode_utf8()
        .map_err(|_| HubError::InvalidReference(rest.to_owned()))?;
    let mut path = decoded.into_owned();
    if cfg!(windows) && path.starts_with('/') && path.as_bytes().get(2) == Some(&b':') {
        path.remove(0);
    }
    Ok(PathBuf::from(path))
}

fn strip_prefix_ci<'a>(input: &'a str, prefix: &str) -> Option<&'a str> {
    input
        .get(..prefix.len())
        .filter(|head| head.eq_ignore_ascii_case(prefix))
        .map(|_| &input[prefix.len()..])
}

fn is_valid_scheme(value: &str) -> bool {
    let mut chars = value.chars();
    match chars.next() {
        Some(first) if first.is_ascii_alphabetic() => {}
        _ => return false,
    }
    chars.all(|ch| ch.is_ascii_alphanumeric() || matches!(ch, '+' | '-' | '.'))
}
