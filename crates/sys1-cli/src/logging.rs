use crate::config::ServerSection;
use anyhow::{Context, bail};
use std::fs::{self, File, OpenOptions};
use std::io::{self, IsTerminal, Write};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use tracing_subscriber::EnvFilter;
use tracing_subscriber::fmt::MakeWriter;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum LogFormat {
    Text,
    Json,
}

/// Install the process-wide `tracing` subscriber.
///
/// Precedence: CLI flag / `SYS1_LOG_*` > `RUST_LOG` for the filter >
/// `[server]` config > defaults.
pub fn init(
    cli_level: Option<String>,
    cli_format: Option<String>,
    cli_file: Option<PathBuf>,
    server: &ServerSection,
) -> anyhow::Result<()> {
    let env_level = std::env::var("RUST_LOG")
        .ok()
        .map(|value| value.trim().to_owned())
        .filter(|value| !value.is_empty());
    let level = cli_level
        .or(env_level)
        .or_else(|| server.log_level.clone())
        .unwrap_or_else(|| "info".to_owned());
    let format = parse_format(
        cli_format
            .or_else(|| server.log_format.clone())
            .as_deref()
            .unwrap_or("text"),
    )?;
    let (writer, ansi) = match cli_file.or_else(|| server.log_file.clone()) {
        Some(path) => {
            if let Some(parent) = path
                .parent()
                .filter(|parent| !parent.as_os_str().is_empty())
            {
                fs::create_dir_all(parent)
                    .with_context(|| format!("create log directory {}", parent.display()))?;
            }
            let file = OpenOptions::new()
                .create(true)
                .append(true)
                .open(&path)
                .with_context(|| format!("open log file {}", path.display()))?;
            (LogWriter::file(file), false)
        }
        None => (LogWriter::stdout(), io::stdout().is_terminal()),
    };
    let filter =
        EnvFilter::try_new(&level).with_context(|| format!("invalid log filter {level:?}"))?;
    let builder = tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_writer(writer)
        .with_ansi(ansi);

    let initialized = if format == LogFormat::Json {
        builder
            .json()
            .with_current_span(true)
            .with_span_list(true)
            .try_init()
    } else {
        builder.try_init()
    };
    initialized.map_err(|error| anyhow::anyhow!("initialize logging: {error}"))
}

fn parse_format(value: &str) -> anyhow::Result<LogFormat> {
    match value.trim().to_ascii_lowercase().as_str() {
        "text" | "plain" => Ok(LogFormat::Text),
        "json" => Ok(LogFormat::Json),
        other => bail!("unsupported log format {other:?}; expected text or json"),
    }
}

#[derive(Clone)]
struct LogWriter {
    file: Option<Arc<Mutex<File>>>,
}

impl LogWriter {
    fn stdout() -> Self {
        Self { file: None }
    }

    fn file(file: File) -> Self {
        Self {
            file: Some(Arc::new(Mutex::new(file))),
        }
    }
}

impl Write for LogWriter {
    fn write(&mut self, buffer: &[u8]) -> io::Result<usize> {
        match &self.file {
            Some(file) => file
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .write(buffer),
            None => io::stdout().write(buffer),
        }
    }

    fn flush(&mut self) -> io::Result<()> {
        match &self.file {
            Some(file) => file
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .flush(),
            None => io::stdout().flush(),
        }
    }
}

impl<'a> MakeWriter<'a> for LogWriter {
    type Writer = Self;

    fn make_writer(&'a self) -> Self::Writer {
        self.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_log_formats() {
        assert_eq!(parse_format("text").unwrap(), LogFormat::Text);
        assert_eq!(parse_format("JSON").unwrap(), LogFormat::Json);
        assert!(parse_format("pretty").is_err());
    }
}
