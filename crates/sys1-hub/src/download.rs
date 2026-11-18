use crate::HubError;
use crate::source::SourceContext;
use sha2::{Digest, Sha256};
use std::fs::{self, File, OpenOptions};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

const USER_AGENT: &str = concat!("rust-sys1/", env!("CARGO_PKG_VERSION"));
/// Full-content verification attempts before giving up.
const MAX_ATTEMPTS: u8 = 4;
/// Consecutive attempts that add no bytes before giving up. A timed-out
/// attempt keeps its partial file, so large downloads resume across attempts.
const MAX_STALLS: u8 = 3;
/// Bytes between progress reports; keeps a fast download from flooding the
/// observer while still updating a terminal several times per second.
const PROGRESS_STEP: u64 = 1024 * 1024;

/// Verify an existing file without treating a mismatch as an error.
pub(crate) fn verify_file(
    path: &Path,
    expected_size: Option<u64>,
    expected_sha256: Option<&str>,
) -> Result<bool, HubError> {
    if !path.exists() {
        return Ok(false);
    }
    let metadata = fs::metadata(path).map_err(|error| HubError::io(path, error))?;
    if let Some(expected) = expected_size
        && metadata.len() != expected
    {
        return Ok(false);
    }
    if let Some(expected) = expected_sha256 {
        let actual = sha256_file(path)?;
        if !actual.eq_ignore_ascii_case(expected) {
            return Ok(false);
        }
    }
    Ok(true)
}

pub(crate) fn sha256_file(path: &Path) -> Result<String, HubError> {
    let mut file = File::open(path).map_err(|error| HubError::io(path, error))?;
    let mut hasher = Sha256::new();
    let mut buffer = [0_u8; 64 * 1024];
    loop {
        let read = file
            .read(&mut buffer)
            .map_err(|error| HubError::io(path, error))?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }
    Ok(hex::encode(hasher.finalize()))
}

pub(crate) fn verify_complete(
    path: &Path,
    expected_size: Option<u64>,
    expected_sha256: Option<&str>,
) -> Result<(), HubError> {
    let metadata = fs::metadata(path).map_err(|error| HubError::io(path, error))?;
    if let Some(expected) = expected_size
        && metadata.len() != expected
    {
        return Err(HubError::SizeMismatch {
            path: path.to_owned(),
            expected,
            actual: metadata.len(),
        });
    }
    if let Some(expected) = expected_sha256 {
        let actual = sha256_file(path)?;
        if !actual.eq_ignore_ascii_case(expected) {
            return Err(HubError::HashMismatch {
                path: path.to_owned(),
                expected: expected.to_owned(),
                actual,
            });
        }
    }
    Ok(())
}

/// Download with HTTP Range resume, size verification, and sha256 verification.
pub(crate) fn download_http(
    context: &SourceContext<'_>,
    url: &str,
    headers: &[(String, String)],
    destination: &Path,
    expected_size: Option<u64>,
    expected_sha256: Option<&str>,
) -> Result<(), HubError> {
    if verify_file(destination, expected_size, expected_sha256)? {
        return Ok(());
    }
    if context.config.offline {
        return Err(HubError::Offline(url.to_owned()));
    }

    ensure_parent(destination)?;
    let partial = partial_path(destination);
    let mut offset = fs::metadata(&partial).map(|meta| meta.len()).unwrap_or(0);
    if let Some(expected) = expected_size
        && offset > expected
    {
        let _ = fs::remove_file(&partial);
        offset = 0;
    }

    let mut attempts = 0_u8;
    let mut stalls = 0_u8;
    'attempts: loop {
        let before = offset;
        let mut request = context
            .client
            .get(url)
            .timeout(context.config.download_timeout)
            .header(reqwest::header::ACCEPT_ENCODING, "identity")
            .header(reqwest::header::USER_AGENT, USER_AGENT);
        for (name, value) in headers {
            request = request.header(name.as_str(), value.as_str());
        }
        if offset > 0 {
            request = request.header(reqwest::header::RANGE, format!("bytes={offset}-"));
        }

        let response = match request.send() {
            Ok(response) => response,
            Err(error) => {
                stalls += 1;
                if stalls >= MAX_STALLS {
                    return Err(HubError::Http {
                        url: url.to_owned(),
                        message: error.to_string(),
                    });
                }
                continue;
            }
        };
        let status = response.status();

        if status == reqwest::StatusCode::RANGE_NOT_SATISFIABLE && offset > 0 {
            if verify_file(&partial, expected_size, expected_sha256)? {
                fs::rename(&partial, destination)
                    .map_err(|error| HubError::io(destination, error))?;
                return Ok(());
            }
            let _ = fs::remove_file(&partial);
            offset = 0;
            stalls += 1;
            if attempts < MAX_ATTEMPTS && stalls < MAX_STALLS {
                attempts += 1;
                continue;
            }
            return Err(HubError::HttpStatus {
                url: url.to_owned(),
                status: status.as_u16(),
                message: "range request failed".to_owned(),
            });
        }

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

        let append = status == reqwest::StatusCode::PARTIAL_CONTENT && offset > 0;
        let mut file = if append {
            OpenOptions::new()
                .append(true)
                .create(true)
                .open(&partial)
                .map_err(|error| HubError::io(&partial, error))?
        } else {
            File::create(&partial).map_err(|error| HubError::io(&partial, error))?
        };

        let mut reader = response;
        if let Some(progress) = context.progress {
            progress.report(offset);
        }
        let mut buffer = [0_u8; 64 * 1024];
        let mut downloaded = offset;
        let mut last_report = offset;
        loop {
            let read = match reader.read(&mut buffer) {
                Ok(0) => break,
                Ok(read) => read,
                Err(error) => {
                    drop(file);
                    let after = fs::metadata(&partial).map(|meta| meta.len()).unwrap_or(0);
                    offset = after;
                    if after > before {
                        stalls = 0;
                    } else {
                        stalls += 1;
                    }
                    if stalls >= MAX_STALLS {
                        return Err(HubError::io(&partial, error));
                    }
                    // Keep the partial file: the next attempt resumes with a
                    // range request from the bytes that did arrive.
                    continue 'attempts;
                }
            };
            if let Err(error) = file.write_all(&buffer[..read]) {
                drop(file);
                let after = fs::metadata(&partial).map(|meta| meta.len()).unwrap_or(0);
                offset = after;
                if after > before {
                    stalls = 0;
                } else {
                    stalls += 1;
                }
                if stalls >= MAX_STALLS {
                    return Err(HubError::io(&partial, error));
                }
                continue 'attempts;
            }
            downloaded += read as u64;
            if downloaded - last_report >= PROGRESS_STEP {
                if let Some(progress) = context.progress {
                    progress.report(downloaded);
                }
                last_report = downloaded;
            }
        }
        file.flush()
            .map_err(|error| HubError::io(&partial, error))?;
        drop(file);
        if let Some(progress) = context.progress {
            progress.report(downloaded);
        }

        match verify_complete(&partial, expected_size, expected_sha256) {
            Ok(()) => {
                fs::rename(&partial, destination)
                    .map_err(|error| HubError::io(destination, error))?;
                return Ok(());
            }
            Err(error) => {
                let _ = fs::remove_file(&partial);
                offset = 0;
                attempts += 1;
                if attempts >= MAX_ATTEMPTS {
                    return Err(error);
                }
            }
        }
    }
}

pub(crate) fn copy_local(
    source: &Path,
    destination: &Path,
    expected_size: Option<u64>,
    expected_sha256: Option<&str>,
) -> Result<(), HubError> {
    if source == destination {
        return verify_complete(destination, expected_size, expected_sha256);
    }
    if verify_file(destination, expected_size, expected_sha256)? {
        return Ok(());
    }
    ensure_parent(destination)?;
    let partial = partial_path(destination);
    fs::copy(source, &partial).map_err(|error| HubError::io(destination, error))?;
    verify_complete(&partial, expected_size, expected_sha256)?;
    fs::rename(&partial, destination).map_err(|error| HubError::io(destination, error))?;
    Ok(())
}

pub(crate) fn partial_path(destination: &Path) -> PathBuf {
    let name = destination
        .file_name()
        .map(|name| name.to_string_lossy().into_owned())
        .unwrap_or_else(|| "download".to_owned());
    destination.with_file_name(format!(".{name}.partial"))
}

fn ensure_parent(path: &Path) -> Result<(), HubError> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).map_err(|error| HubError::io(parent, error))?;
    }
    Ok(())
}
