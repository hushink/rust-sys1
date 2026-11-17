//! Scalar metadata attached to a Kev `head.pt` checkpoint.
//!
//! PyTorch zip checkpoints keep a pickled outer dictionary beside the tensor
//! storages. Kev records the calibrated decoding temperature, the Qwen3.5 base
//! repository, and the base revision there, so the adapter can be served
//! without a hand-written sidecar.

use candle_core::pickle::{Object, Stack};
use std::io::BufReader;
use std::path::{Path, PathBuf};
use sys1_core::ModelError;
use zip::ZipArchive;

/// Scalar training metadata recorded in `head.pt`.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct KevHeadMetadata {
    /// Qwen3.5 base repository or path, e.g. `Qwen/Qwen3.5-0.8B-Base`.
    pub base: Option<String>,
    /// Commit or tag the base was trained against.
    pub base_revision: Option<String>,
    /// Temperature fitted for the pointer logits.
    pub temperature: Option<f32>,
    /// Pointer projection width.
    pub head_dim: Option<usize>,
    /// LoRA rank the head was trained with.
    pub lora_rank: Option<usize>,
}

/// Read scalar metadata from `head.pt` without loading any tensors.
pub fn read_head_metadata(path: &Path) -> Result<KevHeadMetadata, ModelError> {
    let file = std::fs::File::open(path)
        .map_err(|error| ModelError::load(format!("read {}: {error}", path.display())))?;
    let mut archive = ZipArchive::new(BufReader::new(file))
        .map_err(|error| ModelError::load(format!("open {}: {error}", path.display())))?;
    let name = archive
        .file_names()
        .filter_map(|name| name.ok())
        .find(|name| name.ends_with("data.pkl"))
        .map(|name| name.into_owned())
        .ok_or_else(|| {
            ModelError::load(format!(
                "{} has no data.pkl; not a PyTorch zip checkpoint",
                path.display()
            ))
        })?;
    let reader = archive
        .by_name(&name)
        .map_err(|error| ModelError::load(format!("read {name} in {}: {error}", path.display())))?;
    let mut stack = Stack::empty();
    stack
        .read_loop(&mut BufReader::new(reader))
        .map_err(|error| ModelError::load(format!("parse {}: {error}", path.display())))?;
    let object = stack
        .finalize()
        .map_err(|error| ModelError::load(format!("parse {}: {error}", path.display())))?;

    let mut metadata = KevHeadMetadata::default();
    scan(&object, &mut metadata);
    Ok(metadata)
}

/// Metadata for the `head.pt` in `directory`, or defaults when it cannot be
/// read. Directory detection and loading treat the file as optional.
pub fn directory_metadata(directory: &Path) -> KevHeadMetadata {
    read_head_metadata(&directory.join("head.pt")).unwrap_or_default()
}

fn scan(object: &Object, metadata: &mut KevHeadMetadata) {
    match object {
        Object::Dict(entries) => {
            for (key, value) in entries {
                if let Object::Unicode(key) = key {
                    match key.as_str() {
                        "base" if metadata.base.is_none() => {
                            metadata.base = string(value);
                        }
                        "base_revision" if metadata.base_revision.is_none() => {
                            metadata.base_revision = string(value);
                        }
                        "temperature" if metadata.temperature.is_none() => {
                            metadata.temperature = number(value);
                        }
                        "head_dim" if metadata.head_dim.is_none() => {
                            metadata.head_dim = integer(value);
                        }
                        "lora" if metadata.lora_rank.is_none() => {
                            metadata.lora_rank = integer(value);
                        }
                        _ => {}
                    }
                }
                scan(value, metadata);
            }
        }
        Object::List(items) | Object::Tuple(items) => {
            for item in items {
                scan(item, metadata);
            }
        }
        Object::Reduce { callable, args } | Object::Build { callable, args } => {
            scan(callable, metadata);
            scan(args, metadata);
        }
        Object::PersistentLoad(inner) => scan(inner, metadata),
        _ => {}
    }
}

fn string(object: &Object) -> Option<String> {
    match object {
        Object::Unicode(text) if !text.trim().is_empty() => Some(text.trim().to_owned()),
        _ => None,
    }
}

fn number(object: &Object) -> Option<f32> {
    let value = match object {
        Object::Float(value) => *value,
        Object::Int(value) => f64::from(*value),
        Object::Long(value) => *value as f64,
        _ => return None,
    };
    (value.is_finite() && value > 0.0).then_some(value as f32)
}

fn integer(object: &Object) -> Option<usize> {
    match object {
        Object::Int(value) => usize::try_from(*value).ok(),
        Object::Long(value) => usize::try_from(*value).ok(),
        _ => None,
    }
}

/// Base declarations found in the adapter directory, most specific first.
pub(crate) fn base_declaration(directory: &Path) -> Option<(String, Option<String>)> {
    let mut base = None;
    let mut revision = None;
    for (declared, declared_revision) in [
        json_declaration(&directory.join("kev.json")),
        head_declaration(&directory.join("head.pt")),
        provenance_declaration(&directory.join("provenance.json")),
        adapter_declaration(&directory.join("adapter_config.json")),
    ]
    .into_iter()
    .flatten()
    {
        if base.is_none() && !declared.trim().is_empty() {
            base = Some(declared);
        }
        if revision.is_none() {
            revision = declared_revision;
        }
    }
    base.map(|base| (base, revision))
}

/// Normalize a base declaration into a reference the hub or filesystem can
/// resolve.
pub(crate) fn normalize_base_reference(base: &str, revision: Option<&str>) -> Option<String> {
    let base = base.trim();
    if base.is_empty() {
        return None;
    }
    if base.starts_with("http://") || base.starts_with("https://") {
        return Some(base.to_owned());
    }
    if Path::new(base).exists() {
        return Some(format!("local:{}", Path::new(base).display()));
    }
    if let Some((scheme, rest)) = base.split_once(':') {
        let windows_drive = scheme.len() == 1
            && scheme.chars().all(|ch| ch.is_ascii_alphabetic())
            && (rest.starts_with('\\') || rest.starts_with('/'));
        if !windows_drive
            && !scheme.is_empty()
            && scheme
                .chars()
                .next()
                .is_some_and(|ch| ch.is_ascii_alphabetic())
            && scheme
                .chars()
                .all(|ch| ch.is_ascii_alphanumeric() || matches!(ch, '+' | '-' | '.'))
        {
            return Some(with_revision(base, scheme, rest, revision));
        }
    }
    Some(match revision {
        Some(revision) if !base.contains('@') => format!("hf:{base}@{revision}"),
        _ => format!("hf:{base}"),
    })
}

fn with_revision(base: &str, scheme: &str, rest: &str, revision: Option<&str>) -> String {
    let lower = scheme.to_ascii_lowercase();
    match revision {
        Some(revision)
            if matches!(lower.as_str(), "hf" | "huggingface" | "modelscope" | "ms")
                && !rest.contains('@') =>
        {
            format!("{scheme}:{rest}@{revision}")
        }
        _ => base.to_owned(),
    }
}

fn json_declaration(path: &Path) -> Option<(String, Option<String>)> {
    let value: serde_json::Value = serde_json::from_slice(&std::fs::read(path).ok()?).ok()?;
    let base = ["base", "base_model", "base_model_name_or_path"]
        .into_iter()
        .find_map(|key| value.get(key).and_then(serde_json::Value::as_str))?;
    let revision = ["base_revision", "revision"]
        .into_iter()
        .find_map(|key| value.get(key).and_then(serde_json::Value::as_str))
        .map(str::to_owned);
    Some((base.to_owned(), revision))
}

fn head_declaration(path: &Path) -> Option<(String, Option<String>)> {
    let metadata = read_head_metadata(path).ok()?;
    let base = metadata.base?;
    Some((base, metadata.base_revision))
}

fn provenance_declaration(path: &Path) -> Option<(String, Option<String>)> {
    let value: serde_json::Value = serde_json::from_slice(&std::fs::read(path).ok()?).ok()?;
    let config = value.get("config").unwrap_or(&value);
    let base = config
        .get("base")
        .and_then(serde_json::Value::as_str)?
        .to_owned();
    let revision = config
        .get("base_revision")
        .and_then(serde_json::Value::as_str)
        .map(str::to_owned);
    Some((base, revision))
}

fn adapter_declaration(path: &Path) -> Option<(String, Option<String>)> {
    let value: serde_json::Value = serde_json::from_slice(&std::fs::read(path).ok()?).ok()?;
    let base = value
        .get("base_model_name_or_path")
        .and_then(serde_json::Value::as_str)?
        .to_owned();
    let revision = value
        .get("revision")
        .and_then(serde_json::Value::as_str)
        .map(str::to_owned);
    Some((base, revision))
}

/// The base reference declared by a Kev adapter, without resolving or pulling
/// it. Returns `None` when the adapter has no usable declaration.
pub fn declared_base_reference(directory: &Path) -> Option<String> {
    let (base, revision) = base_declaration(directory)?;
    normalize_base_reference(&base, revision.as_deref())
}

/// Resolve the Qwen3.5 base directory for a Kev adapter.
///
/// Resolution order:
/// 1. `configured` (`[families.kev] base_dir`),
/// 2. a Qwen3.5 checkpoint beside the adapter or under `base/`,
/// 3. the base declared in `kev.json`, `head.pt`, `provenance.json`, or
///    `adapter_config.json`, materialized through `pull`.
pub fn resolve_base_directory<F>(
    directory: &Path,
    configured: Option<&Path>,
    pull: F,
) -> Result<PathBuf, ModelError>
where
    F: FnOnce(&str) -> Result<PathBuf, ModelError>,
{
    if let Some(configured) = configured {
        return validate_base(configured).map(|()| configured.to_path_buf());
    }
    if let Some(base) = local_base(directory) {
        return Ok(base);
    }
    let (declared, revision) = base_declaration(directory).ok_or_else(|| {
        ModelError::load(format!(
            "{} is a Kev adapter but declares no Qwen3.5 base; put the base in a `base/` \
             subdirectory, add a `kev.json` with `base`, or set [families.kev] base_dir",
            directory.display()
        ))
    })?;
    let reference = normalize_base_reference(&declared, revision.as_deref()).ok_or_else(|| {
        ModelError::load(format!(
            "Kev adapter at {} declares an invalid base `{declared}`",
            directory.display()
        ))
    })?;
    let base = if let Some(path) = local_reference(&reference) {
        path
    } else {
        pull(&reference)?
    };
    validate_base(&base).map(|()| base)
}

/// Whether `directory` carries a usable Qwen3.5 checkpoint beside the adapter
/// or under `base/`.
pub fn local_base(directory: &Path) -> Option<PathBuf> {
    [directory.to_path_buf(), directory.join("base")]
        .into_iter()
        .find(|candidate| validate_base(candidate).is_ok())
}

fn validate_base(path: &Path) -> Result<(), ModelError> {
    let config = path.join("config.json");
    if !config.is_file() {
        return Err(ModelError::load(format!(
            "Kev base at {} has no config.json",
            path.display()
        )));
    }
    let has_weights = std::fs::read_dir(path)
        .map_err(|error| ModelError::load(format!("read {}: {error}", path.display())))?
        .filter_map(|entry| entry.ok())
        .any(|entry| {
            let name = entry.file_name();
            let name = name.to_string_lossy();
            name.ends_with(".safetensors") && name != "adapter_model.safetensors"
        });
    if !has_weights {
        return Err(ModelError::load(format!(
            "Kev base at {} has no safetensors weights",
            path.display()
        )));
    }
    Ok(())
}

fn local_reference(reference: &str) -> Option<PathBuf> {
    let path = reference.strip_prefix("local:").map(PathBuf::from)?;
    path.is_dir().then_some(path)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn short_binunicode(out: &mut Vec<u8>, text: &str) {
        assert!(text.len() <= u32::MAX as usize);
        out.push(b'X');
        out.extend_from_slice(&(text.len() as u32).to_le_bytes());
        out.extend_from_slice(text.as_bytes());
    }

    enum Value<'a> {
        Text(&'a str),
        Float(f64),
        Int(i32),
    }

    /// A minimal protocol-2 pickle for a flat string/number dictionary.
    fn pickle(entries: &[(&str, Value<'_>)]) -> Vec<u8> {
        let mut out = vec![0x80, 0x02, b'}', b'('];
        for (key, value) in entries {
            short_binunicode(&mut out, key);
            match value {
                Value::Text(text) => short_binunicode(&mut out, text),
                Value::Float(number) => {
                    out.push(b'G');
                    out.extend_from_slice(&number.to_bits().to_be_bytes());
                }
                Value::Int(number) => {
                    out.push(b'J');
                    out.extend_from_slice(&number.to_le_bytes());
                }
            }
        }
        out.push(b'u');
        out.push(b'.');
        out
    }

    fn write_head(path: &Path, payload: &[u8]) {
        let file = std::fs::File::create(path).expect("head.pt");
        let mut zip = zip::ZipWriter::new(file);
        zip.start_file("head/data.pkl", zip::write::SimpleFileOptions::default())
            .expect("zip entry");
        zip.write_all(payload).expect("zip payload");
        zip.finish().expect("zip finish");
    }

    #[test]
    fn reads_temperature_and_base_from_head_pt() {
        let root = std::env::temp_dir().join(format!("sys1-kev-meta-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(&root).expect("root");
        let path = root.join("head.pt");
        write_head(
            &path,
            &pickle(&[
                ("base", Value::Text("Qwen/Qwen3.5-4B-Base")),
                ("base_revision", Value::Text("abc123")),
                ("temperature", Value::Float(2.71)),
                ("lora", Value::Int(16)),
            ]),
        );

        let metadata = read_head_metadata(&path).expect("metadata");
        assert_eq!(metadata.base.as_deref(), Some("Qwen/Qwen3.5-4B-Base"));
        assert_eq!(metadata.base_revision.as_deref(), Some("abc123"));
        assert!((metadata.temperature.expect("temperature") - 2.71).abs() < 1e-5);
        assert_eq!(metadata.lora_rank, Some(16));

        std::fs::remove_dir_all(root).expect("cleanup");
    }

    #[test]
    #[ignore = "requires SYS1_KEV_TEST_DIR containing head.pt"]
    fn reads_metadata_from_a_real_checkpoint() {
        let directory =
            PathBuf::from(std::env::var("SYS1_KEV_TEST_DIR").expect("SYS1_KEV_TEST_DIR"));
        let metadata = read_head_metadata(&directory.join("head.pt")).expect("metadata");
        assert!(
            metadata
                .base
                .as_deref()
                .is_some_and(|base| base.starts_with("Qwen/Qwen3.5-")),
            "unexpected base: {:?}",
            metadata.base
        );
        assert!(metadata.base_revision.is_some());
        let temperature = metadata.temperature.expect("temperature");
        assert!(
            (1.5..=3.5).contains(&temperature),
            "unexpected temperature: {temperature}"
        );
        assert!(metadata.head_dim.is_some_and(|dim| dim >= 128));
    }

    #[test]
    fn normalizes_repo_and_local_references() {
        assert_eq!(
            normalize_base_reference("Qwen/Qwen3.5-0.8B-Base", Some("abc")).as_deref(),
            Some("hf:Qwen/Qwen3.5-0.8B-Base@abc")
        );
        assert_eq!(
            normalize_base_reference("hf:Qwen/Qwen3.5-0.8B-Base", Some("abc")).as_deref(),
            Some("hf:Qwen/Qwen3.5-0.8B-Base@abc")
        );
        assert_eq!(
            normalize_base_reference("hf:owner/repo@main", Some("abc")).as_deref(),
            Some("hf:owner/repo@main")
        );
        assert_eq!(
            normalize_base_reference("modelscope:owner/repo", Some("v1")).as_deref(),
            Some("modelscope:owner/repo@v1")
        );
        let local = std::env::temp_dir();
        assert_eq!(
            normalize_base_reference(&local.display().to_string(), Some("abc")),
            Some(format!("local:{}", local.display()))
        );
    }
}
