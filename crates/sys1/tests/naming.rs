use std::fs;
use std::path::{Path, PathBuf};

fn rust_files(directory: &Path, files: &mut Vec<PathBuf>) {
    for entry in fs::read_dir(directory).expect("read source directory") {
        let path = entry.expect("directory entry").path();
        if path.is_dir() {
            rust_files(&path, files);
        } else if path.extension().and_then(|extension| extension.to_str()) == Some("rs") {
            files.push(path);
        }
    }
}

/// Laya-prefixed types/functions/methods belong to `sys1-laya`.
///
/// The `sys1` umbrella may keep compatibility re-exports of the canonical
/// types (so `use sys1::{LayaModel, LayaOptions}` keeps working), but it must
/// not define new Laya-prefixed items. This test enforces that boundary for
/// every source, test, and example file outside `sys1-laya`.
#[test]
fn laya_prefixed_items_are_only_defined_in_sys1_laya() {
    let workspace = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("crates directory")
        .parent()
        .expect("workspace root");
    let crates = workspace.join("crates");
    let audit_file = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("naming.rs");
    let mut violations = Vec::new();

    for entry in fs::read_dir(&crates).expect("read crates directory") {
        let path = entry.expect("crate entry").path();
        if !path.is_dir() || path.file_name().and_then(|name| name.to_str()) == Some("sys1-laya") {
            continue;
        }

        let mut files = Vec::new();
        rust_files(&path, &mut files);
        for file in files {
            if file == audit_file {
                continue;
            }
            let content = fs::read_to_string(&file).expect("read Rust source");
            for (index, line) in content.lines().enumerate() {
                let trimmed = line.trim_start();
                if trimmed.starts_with("//") || trimmed.starts_with('*') {
                    continue;
                }

                let normalized = line.split_whitespace().collect::<Vec<_>>().join(" ");
                let defines_known_item = [
                    "struct ", "enum ", "trait ", "type ", "union ", "mod ", "fn ", "const ",
                    "static ",
                ]
                .iter()
                .any(|keyword| normalized.contains(&format!("{keyword}Laya")));
                let defines_macro = normalized.contains("macro_rules! Laya");
                let defines_alias = normalized.contains(" as Laya");
                let defines_impl = normalized.contains("impl Laya");

                if defines_known_item || defines_macro || defines_alias || defines_impl {
                    violations.push(format!("{}:{}: {}", file.display(), index + 1, line.trim()));
                }
            }
        }
    }

    assert!(
        violations.is_empty(),
        "Laya-prefixed definitions found outside sys1-laya:\n{}",
        violations.join("\n")
    );
}
