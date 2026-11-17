use std::env;
use std::path::PathBuf;
use std::time::Duration;

/// Hub configuration: cache location, source endpoints, tokens, and timeout.
#[derive(Clone, Debug)]
pub struct HubConfig {
    pub cache_dir: PathBuf,
    pub offline: bool,
    pub hf_endpoint: String,
    pub modelscope_endpoint: String,
    pub hf_token: Option<String>,
    pub modelscope_token: Option<String>,
    pub http_token: Option<String>,
    pub timeout: Duration,
}

impl HubConfig {
    pub fn new(cache_dir: impl Into<PathBuf>) -> Self {
        Self {
            cache_dir: cache_dir.into(),
            offline: false,
            hf_endpoint: "https://huggingface.co".to_owned(),
            modelscope_endpoint: "https://modelscope.cn".to_owned(),
            hf_token: env_non_empty("HF_TOKEN"),
            modelscope_token: env_non_empty("MODELSCOPE_API_TOKEN"),
            http_token: env_non_empty("SYS1_HUB_TOKEN"),
            timeout: Duration::from_secs(60),
        }
    }

    pub fn from_env() -> Self {
        Self::new(default_cache_dir()).with_env()
    }

    pub fn with_env(mut self) -> Self {
        if let Some(value) = env_non_empty("SYS1_CACHE") {
            self.cache_dir = PathBuf::from(value);
        }
        if let Some(value) = env_non_empty("HF_ENDPOINT") {
            self.hf_endpoint = trim_endpoint(&value);
        }
        if let Some(value) = env_non_empty("MODELSCOPE_ENDPOINT") {
            self.modelscope_endpoint = trim_endpoint(&value);
        }
        if let Some(value) = env_non_empty("HF_TOKEN") {
            self.hf_token = Some(value);
        }
        if let Some(value) = env_non_empty("MODELSCOPE_API_TOKEN") {
            self.modelscope_token = Some(value);
        }
        if let Some(value) = env_non_empty("SYS1_HUB_TOKEN") {
            self.http_token = Some(value);
        }
        if let Some(value) =
            env_non_empty("SYS1_HTTP_TIMEOUT_SECS").and_then(|value| value.parse::<u64>().ok())
        {
            self.timeout = Duration::from_secs(value.max(1));
        }
        self.offline = env_truthy("HF_HUB_OFFLINE")
            || env_truthy("TRANSFORMERS_OFFLINE")
            || env_truthy("SYS1_OFFLINE");
        self
    }

    pub fn hf_headers(&self) -> Vec<(String, String)> {
        bearer_headers(self.hf_token.as_deref())
    }

    pub fn modelscope_headers(&self) -> Vec<(String, String)> {
        let Some(token) = self.modelscope_token.as_deref() else {
            return Vec::new();
        };
        vec![
            ("Authorization".to_owned(), format!("Bearer {token}")),
            ("Cookie".to_owned(), format!("m_session_id={token}")),
        ]
    }

    pub fn http_headers(&self) -> Vec<(String, String)> {
        bearer_headers(self.http_token.as_deref())
    }
}

pub fn default_cache_dir() -> PathBuf {
    if let Some(value) = env_non_empty("SYS1_CACHE") {
        return PathBuf::from(value);
    }
    if cfg!(windows) {
        if let Some(value) = env_non_empty("LOCALAPPDATA") {
            return PathBuf::from(value).join("sys1");
        }
        if let Some(value) = env_non_empty("USERPROFILE") {
            return PathBuf::from(value).join(".sys1");
        }
    } else {
        if let Some(value) = env_non_empty("XDG_CACHE_HOME") {
            return PathBuf::from(value).join("sys1");
        }
        if let Some(value) = env_non_empty("HOME") {
            return PathBuf::from(value).join(".cache").join("sys1");
        }
    }
    env::temp_dir().join("sys1")
}

fn bearer_headers(token: Option<&str>) -> Vec<(String, String)> {
    token
        .map(|token| vec![("Authorization".to_owned(), format!("Bearer {token}"))])
        .unwrap_or_default()
}

fn env_non_empty(key: &str) -> Option<String> {
    env::var(key)
        .ok()
        .map(|value| value.trim().to_owned())
        .filter(|value| !value.is_empty())
}

fn env_truthy(key: &str) -> bool {
    env_non_empty(key)
        .map(|value| {
            matches!(
                value.to_ascii_lowercase().as_str(),
                "1" | "true" | "yes" | "on"
            )
        })
        .unwrap_or(false)
}

fn trim_endpoint(value: &str) -> String {
    value.trim().trim_end_matches('/').to_owned()
}
