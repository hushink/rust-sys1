use std::fs;
use std::io::{Read, Write};
use std::net::TcpListener;
use std::path::Path;
use std::process::Command;
use std::thread;

fn checkpoint(directory: &Path) {
    fs::create_dir_all(directory.join("encoder")).expect("encoder dir");
    fs::write(directory.join("model.safetensors"), b"x").expect("weights");
    fs::write(directory.join("encoder/config.json"), b"{}").expect("encoder config");
    fs::write(directory.join("rl_agent_config.json"), b"{}").expect("rl config");
    fs::write(directory.join("tokenizer/tokenizer.json"), b"{}").expect("tokenizer");
}

fn sys1() -> Command {
    let mut command = Command::new(env!("CARGO_BIN_EXE_sys1"));
    for key in [
        "SYS1_CONFIG",
        "SYS1_CACHE",
        "SYS1_MAX_CACHE_BYTES",
        "SYS1_LOG_LEVEL",
        "SYS1_LOG_FORMAT",
        "SYS1_LOG_FILE",
        "RUST_LOG",
    ] {
        command.env_remove(key);
    }
    command
}

#[test]
fn doctor_auto_loads_sys1_toml_from_cwd() {
    let dir = tempfile::tempdir().expect("tempdir");
    fs::write(
        dir.path().join("sys1.toml"),
        "[hub]\ncache_dir = \"models\"\n",
    )
    .expect("config");

    let output = sys1()
        .current_dir(dir.path())
        .arg("doctor")
        .output()
        .expect("run sys1 doctor");
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("config:"), "{stdout}");
    assert!(stdout.contains("models"), "{stdout}");
}

#[test]
fn cli_cache_overrides_config() {
    let dir = tempfile::tempdir().expect("tempdir");
    fs::write(
        dir.path().join("sys1.toml"),
        "[hub]\ncache_dir = \"from-config\"\n",
    )
    .expect("config");

    let output = sys1()
        .current_dir(dir.path())
        .args(["--cache", "from-cli", "doctor"])
        .output()
        .expect("run sys1 doctor");
    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("from-cli"), "{stdout}");
    assert!(!stdout.contains("from-config"), "{stdout}");
}

#[test]
fn doctor_reports_cache_quota_and_cli_overrides_config() {
    let dir = tempfile::tempdir().expect("tempdir");
    fs::write(
        dir.path().join("sys1.toml"),
        "[hub]\ncache_dir = \"models\"\nmax_cache_bytes = 2048\n",
    )
    .expect("config");

    let output = sys1()
        .current_dir(dir.path())
        .arg("doctor")
        .output()
        .expect("run sys1 doctor");
    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("cache limit: 2.0 KiB"), "{stdout}");

    let output = sys1()
        .current_dir(dir.path())
        .args(["--max-cache-bytes", "1048576", "doctor"])
        .output()
        .expect("run sys1 doctor");
    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("cache limit: 1.0 MiB"), "{stdout}");
}

#[test]
fn serve_without_models_requires_the_admin_api() {
    let dir = tempfile::tempdir().expect("tempdir");
    fs::write(
        dir.path().join("sys1.toml"),
        "[hub]\ncache_dir = \"models\"\n",
    )
    .expect("config");

    let output = sys1()
        .current_dir(dir.path())
        .args(["serve", "--no-admin"])
        .output()
        .expect("run sys1 serve");
    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("no models configured"), "{stderr}");
}

fn serve_manifest_once(manifest: String) -> (String, thread::JoinHandle<()>) {
    let listener = TcpListener::bind("127.0.0.1:0").expect("bind manifest server");
    let address = listener.local_addr().expect("local address");
    let handle = thread::spawn(move || {
        let (mut stream, _) = listener.accept().expect("accept manifest request");
        let mut request = [0_u8; 4096];
        let _ = stream.read(&mut request);
        write!(
            stream,
            "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nContent-Type: application/json\r\nConnection: close\r\n\r\n",
            manifest.len()
        )
        .expect("response head");
        stream
            .write_all(manifest.as_bytes())
            .expect("response body");
    });
    (format!("http://{address}/model.json"), handle)
}

#[test]
fn config_name_resolves_a_custom_http_center() {
    let manifest = serde_json::json!({
        "schema_version": 1,
        "name": "custom-http",
        "description": "custom center",
        "release_date": "2026-01-01",
        "family": "test",
        "backend": "fake",
        "format": "test",
        "version": "1.0.0",
        "files": [{
            "path": "weights.bin",
            "url": "weights.bin",
            "size": 4
        }]
    })
    .to_string();
    let (url, server) = serve_manifest_once(manifest);

    let dir = tempfile::tempdir().expect("tempdir");
    let config = dir.path().join("sys1.toml");
    fs::write(
        &config,
        format!("[hub]\ncache_dir = \"cache\"\n\n[models]\ncustom = \"{url}\"\n"),
    )
    .expect("config");

    let output = sys1()
        .args(["--config", config.to_str().unwrap(), "show", "custom"])
        .output()
        .expect("run sys1 show");
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("\"name\": \"custom-http\""), "{stdout}");
    server.join().expect("manifest server");
}

#[test]
fn serve_fails_when_a_preload_model_is_unknown() {
    let dir = tempfile::tempdir().expect("tempdir");
    let model_dir = dir.path().join("laya");
    fs::create_dir_all(model_dir.join("tokenizer")).expect("model dir");
    checkpoint(&model_dir);
    let model_path = model_dir.display().to_string().replace('\\', "/");

    let config = dir.path().join("sys1.toml");
    fs::write(
        &config,
        format!(
            "[hub]\ncache_dir = \"cache\"\n\n\
             [server]\ndevice = \"cpu\"\ndtype = \"f32\"\nadmin = false\n\
             preload = [\"missing\"]\n\n\
             [models]\nlaya = \"local:{model_path}\"\n"
        ),
    )
    .expect("config");

    let output = sys1()
        .args(["--config", config.to_str().unwrap(), "serve"])
        .output()
        .expect("run sys1 serve");
    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("preload model `missing`"), "{stderr}");
}

#[test]
fn invalid_log_format_fails() {
    let output = sys1()
        .args(["--log-format", "pretty", "doctor"])
        .output()
        .expect("run sys1 doctor");
    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("unsupported log format"), "{stderr}");
}

#[test]
fn json_logs_can_be_written_to_a_file() {
    let dir = tempfile::tempdir().expect("tempdir");
    let log = dir.path().join("logs").join("sys1.log");

    let output = sys1()
        .args([
            "--log-format",
            "json",
            "--log-level",
            "info",
            "--log-file",
            log.to_str().unwrap(),
            "doctor",
        ])
        .output()
        .expect("run sys1 doctor");
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("rust-sys1 doctor"), "{stdout}");
    let logs = fs::read_to_string(&log).expect("log file");
    assert!(logs.contains("sys1 starting"), "{logs}");
    assert!(logs.contains("\"level\""), "{logs}");
}

#[test]
fn invalid_config_fails_with_a_path() {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("bad.toml");
    fs::write(&path, "[hub\ncache_dir = \"x\"\n").expect("config");

    let output = sys1()
        .args(["--config", path.to_str().unwrap(), "doctor"])
        .output()
        .expect("run sys1 doctor");
    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("parse config file"), "{stderr}");
}
