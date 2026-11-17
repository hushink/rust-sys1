use std::fs;
use std::process::Command;

fn sys1() -> Command {
    let mut command = Command::new(env!("CARGO_BIN_EXE_sys1"));
    command.env_remove("SYS1_CONFIG");
    command.env_remove("SYS1_CACHE");
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
