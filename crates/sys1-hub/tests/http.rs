use sha2::{Digest, Sha256};
use std::fs;
use std::io::{BufRead, BufReader, Write};
use std::net::{TcpListener, TcpStream};
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};
use std::time::Duration;
use sys1_core::{ModelFile, ModelManifest};
use sys1_hub::{Hub, HubConfig, HubError};

fn sha256(bytes: &[u8]) -> String {
    hex::encode(Sha256::digest(bytes))
}

#[derive(Clone, Debug)]
struct Request {
    path: String,
    range: Option<String>,
    authorization: Option<String>,
}

struct ServerState {
    manifest: Vec<u8>,
    file: Vec<u8>,
    token: Option<String>,
    requests: Mutex<Vec<Request>>,
}

struct TestServer {
    base_url: String,
    state: Arc<ServerState>,
    stop: Arc<AtomicBool>,
    handle: Option<JoinHandle<()>>,
}

impl Drop for TestServer {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::SeqCst);
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
    }
}

impl TestServer {
    fn start(content: &[u8], token: Option<&str>) -> Self {
        let manifest = ModelManifest {
            schema_version: 1,
            name: "http-test".to_owned(),
            description: "HTTP source test model".to_owned(),
            release_date: "2026-01-01".to_owned(),
            family: "test".to_owned(),
            variant: None,
            backend: "fake".to_owned(),
            format: "test".to_owned(),
            version: "1.0.0".to_owned(),
            aliases: Vec::new(),
            files: vec![ModelFile {
                path: "weights.bin".to_owned(),
                url: String::new(),
                sha256: Some(sha256(content)),
                git_sha1: None,
                size: content.len() as u64,
            }],
        };
        let state = Arc::new(ServerState {
            manifest: serde_json::to_vec_pretty(&manifest).expect("manifest JSON"),
            file: content.to_vec(),
            token: token.map(str::to_owned),
            requests: Mutex::new(Vec::new()),
        });

        let listener = TcpListener::bind("127.0.0.1:0").expect("bind test server");
        listener
            .set_nonblocking(true)
            .expect("nonblocking listener");
        let address = listener.local_addr().expect("local address");
        let stop = Arc::new(AtomicBool::new(false));
        let thread_state = state.clone();
        let thread_stop = stop.clone();
        let handle = thread::spawn(move || {
            while !thread_stop.load(Ordering::SeqCst) {
                match listener.accept() {
                    Ok((stream, _)) => {
                        let _ = handle_connection(stream, &thread_state);
                    }
                    Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
                        thread::sleep(Duration::from_millis(2));
                    }
                    // Windows can surface a transient accept error when a
                    // client reconnects after `Connection: close`.
                    Err(_) => thread::sleep(Duration::from_millis(2)),
                }
            }
        });

        Self {
            base_url: format!("http://{address}"),
            state,
            stop,
            handle: Some(handle),
        }
    }

    fn requests(&self) -> Vec<Request> {
        self.state
            .requests
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .clone()
    }
}

fn handle_connection(mut stream: TcpStream, state: &ServerState) -> std::io::Result<()> {
    // On Windows accepted sockets can inherit the listener's nonblocking mode.
    stream.set_nonblocking(false)?;
    stream.set_read_timeout(Some(Duration::from_secs(5)))?;
    let mut reader = BufReader::new(stream.try_clone()?);

    // Keep the connection alive across the manifest and file requests so the
    // client never races a reconnect against connection teardown.
    loop {
        let mut request_line = String::new();
        match reader.read_line(&mut request_line) {
            Ok(0) => return Ok(()),
            Ok(_) => {}
            Err(error)
                if matches!(
                    error.kind(),
                    std::io::ErrorKind::WouldBlock | std::io::ErrorKind::TimedOut
                ) =>
            {
                return Ok(());
            }
            Err(error) => return Err(error),
        }
        let path = request_line
            .split_whitespace()
            .nth(1)
            .unwrap_or("/")
            .to_owned();
        let mut range = None;
        let mut authorization = None;
        loop {
            let mut line = String::new();
            if reader.read_line(&mut line)? == 0 || line == "\r\n" || line == "\n" {
                break;
            }
            if let Some((name, value)) = line.split_once(':') {
                let value = value.trim();
                if name.eq_ignore_ascii_case("range") {
                    range = Some(value.to_owned());
                } else if name.eq_ignore_ascii_case("authorization") {
                    authorization = Some(value.to_owned());
                }
            }
        }
        state
            .requests
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .push(Request {
                path: path.clone(),
                range: range.clone(),
                authorization: authorization.clone(),
            });

        if let Some(expected) = state.token.as_deref() {
            let expected = format!("Bearer {expected}");
            if authorization.as_deref() != Some(expected.as_str()) {
                respond(&mut stream, "401 Unauthorized", &[], b"unauthorized")?;
                continue;
            }
        }

        match path.as_str() {
            "/model.json" => respond(
                &mut stream,
                "200 OK",
                &[("Content-Type", "application/json")],
                &state.manifest,
            )?,
            "/weights.bin" => {
                let total = state.file.len();
                let offset = range
                    .as_deref()
                    .and_then(|value| value.strip_prefix("bytes="))
                    .and_then(|value| value.split('-').next())
                    .and_then(|value| value.parse::<usize>().ok());
                match offset {
                    Some(offset) if offset < total => respond(
                        &mut stream,
                        "206 Partial Content",
                        &[(
                            "Content-Range",
                            &format!("bytes {offset}-{}/{total}", total - 1),
                        )],
                        &state.file[offset..],
                    )?,
                    Some(_) => respond(
                        &mut stream,
                        "416 Range Not Satisfiable",
                        &[("Content-Range", &format!("bytes */{total}"))],
                        b"",
                    )?,
                    None => respond(&mut stream, "200 OK", &[], &state.file)?,
                }
            }
            _ => respond(&mut stream, "404 Not Found", &[], b"not found")?,
        }
    }
}

fn respond(
    stream: &mut TcpStream,
    status: &str,
    headers: &[(&str, &str)],
    body: &[u8],
) -> std::io::Result<()> {
    let mut head = format!(
        "HTTP/1.1 {status}\r\nContent-Length: {}\r\nAccept-Ranges: bytes\r\nConnection: keep-alive\r\n",
        body.len()
    );
    for (name, value) in headers {
        head.push_str(&format!("{name}: {value}\r\n"));
    }
    head.push_str("\r\n");
    stream.write_all(head.as_bytes())?;
    stream.write_all(body)?;
    stream.flush()
}

fn hub(cache: &Path) -> Hub {
    Hub::new(HubConfig::new(cache)).expect("hub")
}

#[test]
fn pull_evicts_old_entries_to_fit_the_quota() {
    let temp = tempfile::tempdir().expect("temp dir");
    let cache = temp.path().join("cache");

    let old_content = vec![1_u8; 2048];
    let old_manifest = ModelManifest {
        schema_version: 1,
        name: "old-model".to_owned(),
        description: "old cached model".to_owned(),
        release_date: "2026-01-01".to_owned(),
        family: "test".to_owned(),
        variant: None,
        backend: "fake".to_owned(),
        format: "test".to_owned(),
        version: "1.0.0".to_owned(),
        aliases: Vec::new(),
        files: vec![ModelFile {
            path: "weights.bin".to_owned(),
            url: String::new(),
            sha256: Some(sha256(&old_content)),
            git_sha1: None,
            size: old_content.len() as u64,
        }],
    };
    let old_dir = cache.join("models").join("old-model-key");
    fs::create_dir_all(&old_dir).expect("old cache dir");
    fs::write(old_dir.join("weights.bin"), &old_content).expect("old weights");
    fs::write(
        old_dir.join("model.json"),
        serde_json::to_vec_pretty(&old_manifest).expect("old manifest"),
    )
    .expect("old manifest file");

    let unlimited = hub(&cache);
    let used = unlimited.cache_usage().expect("usage").used_bytes;
    let mut config = HubConfig::new(&cache);
    config.max_cache_bytes = Some(used);
    let limited = Hub::new(config).expect("limited hub");

    let content = b"new weights";
    let server = TestServer::start(content, None);
    let reference = format!("{}/model.json", server.base_url);
    let pulled = limited.pull(&reference).expect("pull with quota");

    assert!(!old_dir.exists(), "old entry should be evicted");
    assert!(pulled.directory.is_dir());
    let usage = limited.cache_usage().expect("usage");
    assert!(usage.used_bytes <= used);
    drop(limited);
}

#[test]
fn pulls_a_static_http_manifest_and_file() {
    let content = b"static HTTP model center weights";
    let server = TestServer::start(content, None);
    let temp = tempfile::tempdir().expect("temp dir");
    let hub = hub(&temp.path().join("cache"));

    let pulled = hub.pull(&server.base_url).expect("HTTP pull");
    assert_eq!(pulled.manifest.name, "http-test");
    assert_eq!(
        fs::read(pulled.files["weights.bin"].as_path()).expect("cached file"),
        content
    );

    let requests = server.requests();
    let paths = requests
        .iter()
        .map(|request| request.path.as_str())
        .collect::<Vec<_>>();
    assert_eq!(paths, vec!["/model.json", "/weights.bin"]);
    assert!(requests.iter().all(|request| request.range.is_none()));
}

#[test]
fn resumes_a_partial_download_with_range() {
    let content = b"0123456789abcdefghijklmnopqrstuvwxyz";
    let server = TestServer::start(content, None);
    let temp = tempfile::tempdir().expect("temp dir");
    let hub = hub(&temp.path().join("cache"));
    let reference = server.base_url.clone();

    let model_dir = hub.cache_location(&reference).expect("cache location");
    fs::create_dir_all(&model_dir).expect("model dir");
    fs::write(model_dir.join(".weights.bin.partial"), &content[..10]).expect("partial file");

    let pulled = hub.pull(&reference).expect("resumed HTTP pull");
    assert_eq!(
        fs::read(pulled.files["weights.bin"].as_path()).expect("cached file"),
        content
    );
    assert!(!model_dir.join(".weights.bin.partial").exists());

    let requests = server.requests();
    let file_request = requests
        .iter()
        .find(|request| request.path == "/weights.bin")
        .expect("file request");
    assert_eq!(file_request.range.as_deref(), Some("bytes=10-"));
}

#[test]
fn sends_the_configured_http_token() {
    let content = b"token protected weights";
    let server = TestServer::start(content, Some("hub-secret"));
    let temp = tempfile::tempdir().expect("temp dir");

    let mut config = HubConfig::new(temp.path().join("cache"));
    config.http_token = Some("hub-secret".to_owned());
    let hub = Hub::new(config).expect("hub");
    hub.pull(&server.base_url).expect("authorized pull");
    let requests = server.requests();
    assert!(!requests.is_empty());
    assert!(
        requests
            .iter()
            .all(|request| request.authorization.as_deref() == Some("Bearer hub-secret"))
    );
}

#[test]
fn rejects_http_sources_without_the_token() {
    let server = TestServer::start(b"protected", Some("hub-secret"));
    let temp = tempfile::tempdir().expect("temp dir");
    let hub = hub(&temp.path().join("cache"));

    let error = hub.pull(&server.base_url).expect_err("must require token");
    match error {
        HubError::HttpStatus { status, .. } => assert_eq!(status, 401),
        other => panic!("unexpected error: {other}"),
    }
}
