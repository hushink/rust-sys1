mod cache;
mod config;
mod logging;
mod reload;
mod source;

use anyhow::{Context, bail};
use cache::HubCacheManager;
use candle_core::{DType, Device};
use clap::{Parser, Subcommand};
use config::LoadedConfig;
use indexmap::IndexMap;
use reload::HubConfigReloader;
use source::HubModelResolver;
use std::collections::HashSet;
use std::fs;
use std::io::Read;
use std::net::SocketAddr;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};
use sys1_core::{
    DecisionModel, DevicePreference, LongContextOptions, MinConfidence, ModelError, ModelManifest,
    ModelRegistry, ModelResolver, PredictionOptions, Questions, RegistryConfig,
};
use sys1_families::{
    BundleOptions, FamilyFactory, FamilyOptions, is_supported_directory, load_manifest,
    local_manifest, register_bundle, resolve_bundle_overrides,
};
use sys1_hub::{Hub, HubConfig, ModelRef, PulledModel};
use sys1_laya::{
    ConfidenceMode, DEFAULT_MAX_ATTENTION_ELEMENTS, LayaOptions as ModelOptions, LayaRouterOptions,
    RouteVariant, has_checkpoint_variants, set_tf32,
};
use sys1_protocol::{Question, State, SystemOneResponse};
use sys1_server::{
    BatchConfig, ConfigReloader, ServerConfig, ServerExtensions, TracingTelemetry,
    serve_with_extensions_and_shutdown,
};

#[derive(Parser)]
#[command(
    name = "sys1",
    version,
    about = "System One inference server and model hub"
)]
struct Cli {
    #[arg(long, global = true, env = "SYS1_CACHE", value_name = "DIR")]
    cache: Option<PathBuf>,
    /// Maximum bytes kept in the model cache (0 or unset = unlimited).
    #[arg(
        long,
        global = true,
        env = "SYS1_MAX_CACHE_BYTES",
        value_name = "BYTES"
    )]
    max_cache_bytes: Option<u64>,
    /// Config file (default: ./sys1.toml when present).
    #[arg(long, global = true, env = "SYS1_CONFIG", value_name = "FILE")]
    config: Option<PathBuf>,
    #[arg(long, global = true)]
    offline: bool,
    /// `tracing` filter, e.g. `info` or `sys1=debug,tower_http=info`.
    #[arg(long, global = true, env = "SYS1_LOG_LEVEL", value_name = "FILTER")]
    log_level: Option<String>,
    /// Log format: text or json.
    #[arg(long, global = true, env = "SYS1_LOG_FORMAT", value_name = "FORMAT")]
    log_format: Option<String>,
    /// Append logs to a file instead of stdout.
    #[arg(long, global = true, env = "SYS1_LOG_FILE", value_name = "FILE")]
    log_file: Option<PathBuf>,
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Serve the TypeSafe-compatible System One API.
    Serve(ServeArgs),
    /// Download and materialize a model.
    Pull(PullArgs),
    /// List cached models.
    List,
    /// Resolve and print a model manifest.
    Show(ShowArgs),
    /// Remove a cached model.
    Rm(RmArgs),
    /// Run one inference from a JSON input file.
    Run(RunArgs),
    /// Print build and device diagnostics.
    Doctor(DoctorArgs),
}

#[derive(clap::Args)]
struct ServeArgs {
    /// Bind address (default: 127.0.0.1:8080).
    #[arg(long)]
    bind: Option<SocketAddr>,
    #[arg(long, env = "SYS1_API_KEY")]
    api_key: Option<String>,
    #[arg(long = "model", value_name = "NAME=REF")]
    models: Vec<String>,
    /// Device: auto, cpu, cuda:N, or metal (default: auto).
    #[arg(long)]
    device: Option<String>,
    /// Dtype: auto, f32, f16, or bf16 (default: auto).
    #[arg(long)]
    dtype: Option<String>,
    #[arg(long)]
    max_len: Option<usize>,
    /// Confidence mode: typesafe or laya (default: typesafe).
    #[arg(long)]
    laya_confidence: Option<String>,
    /// Auto-select typed-decisions when the question ids match a published workflow.
    #[arg(long)]
    laya_auto_task_detection: bool,
    /// Fallback checkpoint: english, multilingual, or typed-decisions.
    #[arg(long, value_name = "VARIANT")]
    laya_router_default: Option<String>,
    /// Router-wide language hint, e.g. en or ro.
    #[arg(long, value_name = "CODE")]
    laya_lang_guess: Option<String>,
    /// Maximum requests per model forward pass (default: 32).
    #[arg(long)]
    batch_max: Option<usize>,
    /// Maximum wait to collect a batch, in milliseconds (default: 2).
    #[arg(long)]
    batch_wait_ms: Option<u64>,
    /// Cap for batch * heads * seq_len² in one global-attention batch
    /// (default: 64M elements).
    #[arg(long)]
    max_attention_elements: Option<usize>,
    /// Disable cross-request batching.
    #[arg(long)]
    no_batch: bool,
    /// Allow TF32 tensor-core reductions for F32 CUDA GEMMs.
    #[arg(long)]
    tf32: bool,
    /// Disable exact sliding-window attention for local layers.
    #[arg(long)]
    no_local_window: bool,
    /// Disable the `/api/models` management endpoints.
    #[arg(long)]
    no_admin: bool,
    /// Enable `/metrics` and `/api/metrics` even if the config disables them.
    #[arg(long, conflicts_with = "no_metrics")]
    metrics: bool,
    /// Disable `/metrics` and `/api/metrics`.
    #[arg(long)]
    no_metrics: bool,
    /// Load these models before serving (repeatable). Overrides [server] preload.
    #[arg(long = "preload", value_name = "NAME")]
    preload: Vec<String>,
    /// Allowed CORS origin (repeatable). Overrides [server] cors_origins.
    #[arg(long = "cors-origin", value_name = "ORIGIN")]
    cors_origins: Vec<String>,
    /// Disable CORS headers entirely.
    #[arg(long, conflicts_with = "cors_origins")]
    no_cors: bool,
}

#[derive(clap::Args)]
struct PullArgs {
    reference: String,
}

#[derive(clap::Args)]
struct ShowArgs {
    reference: String,
}

#[derive(clap::Args)]
struct RmArgs {
    reference: String,
}

#[derive(clap::Args)]
struct RunArgs {
    reference: String,
    #[arg(long, value_name = "FILE", default_value = "-")]
    input: String,
    /// Device: auto, cpu, cuda:N, or metal (default: auto).
    #[arg(long)]
    device: Option<String>,
    /// Dtype: auto, f32, f16, or bf16 (default: auto).
    #[arg(long)]
    dtype: Option<String>,
    #[arg(long)]
    max_len: Option<usize>,
    /// Cap for batch * heads * seq_len² in one global-attention batch
    /// (default: 64M elements).
    #[arg(long)]
    max_attention_elements: Option<usize>,
    /// Confidence mode: typesafe or laya (default: typesafe).
    #[arg(long)]
    laya_confidence: Option<String>,
    /// Auto-select typed-decisions when the question ids match a published workflow.
    #[arg(long)]
    laya_auto_task_detection: bool,
    /// Fallback checkpoint: english, multilingual, or typed-decisions.
    #[arg(long, value_name = "VARIANT")]
    laya_router_default: Option<String>,
    /// Router-wide language hint, e.g. en or ro.
    #[arg(long, value_name = "CODE")]
    laya_lang_guess: Option<String>,
    /// Scan a state longer than the context window in overlapping windows.
    #[arg(long)]
    long: bool,
    /// State tokens per window (default: the checkpoint's state budget).
    #[arg(long, value_name = "TOKENS")]
    window: Option<usize>,
    /// Token step between windows (default: half the window).
    #[arg(long, value_name = "TOKENS")]
    stride: Option<usize>,
    /// Maximum windows per forward pass.
    #[arg(long, value_name = "WINDOWS")]
    long_batch: Option<usize>,
    /// Abstain when answer confidence is below this threshold (0.0-1.0).
    #[arg(long, value_name = "FLOAT")]
    min_confidence: Option<f64>,
    /// Allow TF32 tensor-core reductions for F32 CUDA GEMMs.
    #[arg(long)]
    tf32: bool,
    /// Disable exact sliding-window attention for local layers.
    #[arg(long)]
    no_local_window: bool,
}

#[derive(clap::Args)]
struct DoctorArgs {
    #[arg(long, default_value = "auto")]
    device: String,
    #[arg(long, default_value = "auto")]
    dtype: String,
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();
    let cache = cli.cache.clone();
    let max_cache_bytes = cli.max_cache_bytes;
    let offline = cli.offline;
    let config = config::load(cli.config.as_deref())?;
    logging::init(
        cli.log_level.clone(),
        cli.log_format.clone(),
        cli.log_file.clone(),
        &config.value.server,
    )?;
    tracing::info!(version = env!("CARGO_PKG_VERSION"), "sys1 starting");
    match cli.command {
        Command::Serve(args) => {
            serve_command(cache.as_deref(), offline, max_cache_bytes, &config, args)
        }
        Command::Pull(args) => {
            let hub = make_hub(cache.as_deref(), offline, max_cache_bytes, &config)?;
            let reference = config.model_reference(&args.reference);
            let pulled = pull_model(&hub, &reference)?;
            println!("{} -> {}", pulled.manifest.name, pulled.directory.display());
            let families = config.value.families.to_options();
            let overrides =
                resolve_bundle_overrides(&pulled.directory, &families, |base_reference| {
                    pull_base(&hub, base_reference)
                })?;
            if let Some(base) = overrides.kev_base {
                println!("base -> {}", base.display());
            }
            Ok(())
        }
        Command::List => {
            let hub = make_hub(cache.as_deref(), offline, max_cache_bytes, &config)?;
            for model in hub.list_cached()? {
                println!(
                    "{:<28} {:<10} {:<12} {}",
                    model.manifest.name,
                    model.manifest.backend,
                    model.manifest.format,
                    model.directory.display()
                );
            }
            let usage = hub.cache_usage()?;
            println!(
                "cache: {} used{}",
                format_bytes(usage.used_bytes),
                usage
                    .limit_bytes
                    .map(|limit| format!(
                        " / {} ({} reclaimable)",
                        format_bytes(limit),
                        format_bytes(usage.reclaimable_bytes)
                    ))
                    .unwrap_or_else(|| format!(
                        " (unlimited, {} reclaimable)",
                        format_bytes(usage.reclaimable_bytes)
                    ))
            );
            Ok(())
        }
        Command::Show(args) => {
            let hub = make_hub(cache.as_deref(), offline, max_cache_bytes, &config)?;
            let reference = config.model_reference(&args.reference);
            let manifest = hub.resolve(&reference)?;
            println!("{}", serde_json::to_string_pretty(&manifest)?);
            Ok(())
        }
        Command::Rm(args) => {
            let hub = make_hub(cache.as_deref(), offline, max_cache_bytes, &config)?;
            let reference = config.model_reference(&args.reference);
            let location = hub.cache_location(&reference)?;
            if hub.remove(&reference)? {
                println!("removed {}", location.display());
            } else {
                println!("not cached: {}", location.display());
            }
            Ok(())
        }
        Command::Run(args) => {
            run_command(cache.as_deref(), offline, max_cache_bytes, &config, args)
        }
        Command::Doctor(args) => {
            doctor_command(cache.as_deref(), offline, max_cache_bytes, &config, args)
        }
    }
}

fn serve_command(
    cache: Option<&Path>,
    offline: bool,
    max_cache_bytes: Option<u64>,
    config: &LoadedConfig,
    args: ServeArgs,
) -> anyhow::Result<()> {
    let server = &config.value.server;
    let bind = args
        .bind
        .or(server.bind)
        .unwrap_or_else(|| "127.0.0.1:8080".parse().expect("valid default bind"));
    let device = args
        .device
        .or_else(|| server.device.clone())
        .unwrap_or_else(|| "auto".to_owned());
    let dtype = args
        .dtype
        .or_else(|| server.dtype.clone())
        .unwrap_or_else(|| "auto".to_owned());
    let confidence = args
        .laya_confidence
        .or_else(|| server.laya_confidence.clone())
        .unwrap_or_else(|| "typesafe".to_owned());
    let max_len = args.max_len.or(server.max_len);
    let max_attention_elements = args
        .max_attention_elements
        .or(server.max_attention_elements)
        .unwrap_or(DEFAULT_MAX_ATTENTION_ELEMENTS);
    let local_window = !args.no_local_window && server.local_window.unwrap_or(true);
    let tf32 = args.tf32 || server.tf32.unwrap_or(false);
    set_tf32(tf32);
    let auto_task_detection =
        args.laya_auto_task_detection || server.laya_auto_task_detection.unwrap_or(false);
    let router_default = args
        .laya_router_default
        .or_else(|| server.laya_router_default.clone())
        .unwrap_or_else(|| "multilingual".to_owned());
    let lang_guess = args
        .laya_lang_guess
        .or_else(|| server.laya_lang_guess.clone());
    let options = make_model_options(
        &device,
        &dtype,
        max_len,
        &confidence,
        local_window,
        max_attention_elements,
        auto_task_detection,
        &router_default,
        lang_guess,
    )?;
    let specs = if args.models.is_empty() {
        config.model_specs()
    } else {
        args.models
    };
    let enable_admin = !args.no_admin && server.admin.unwrap_or(true);
    let enable_metrics = if args.metrics {
        true
    } else if args.no_metrics {
        false
    } else {
        server.metrics.unwrap_or(true)
    };
    let families = config.value.families.to_options();
    let hub = Arc::new(make_hub(cache, offline, max_cache_bytes, config)?);
    let (manifests, factory) = build_models(&hub, &specs, options.clone(), &families)?;
    if manifests.is_empty() {
        if !enable_admin {
            bail!(
                "no models configured; pass --model NAME=REF, add [models] to sys1.toml, \
                 or leave the admin API enabled to add models at runtime"
            );
        }
        tracing::warn!(
            "starting with an empty catalog; add models with `POST /api/models` or `sys1 pull`"
        );
    }
    let registry = Arc::new(ModelRegistry::new(
        manifests,
        factory.clone(),
        RegistryConfig::default(),
    )?);
    hub.set_residency(Arc::new(RegistryResidency(registry.clone())));
    let resolver_impl = Arc::new(HubModelResolver::new(
        hub.clone(),
        factory,
        options,
        families,
        config.value.models.clone(),
    ));
    let cache_manager: Arc<dyn sys1_server::CacheManager> = Arc::new(HubCacheManager::new(hub));
    let reloader: Option<Arc<dyn ConfigReloader>> = config.path.as_ref().map(|path| {
        let reloader: Arc<dyn ConfigReloader> = Arc::new(HubConfigReloader::new(
            path.clone(),
            resolver_impl.clone(),
            registry.clone(),
            config.value.clone(),
        ));
        reloader
    });
    let resolver: Arc<dyn ModelResolver> = resolver_impl;

    let preload = if args.preload.is_empty() {
        server.preload.clone()
    } else {
        args.preload.clone()
    };
    for name in &preload {
        let started = Instant::now();
        registry
            .acquire(name)
            .with_context(|| format!("preload model `{name}`"))?;
        tracing::info!(
            model = %name,
            elapsed_ms = started.elapsed().as_secs_f64() * 1000.0,
            "preloaded model"
        );
    }

    let batch_max = args.batch_max.or(server.batch_max).unwrap_or(32);
    let batch_wait_ms = args.batch_wait_ms.or(server.batch_wait_ms).unwrap_or(2);
    let batching = if args.no_batch || server.batch == Some(false) {
        None
    } else {
        Some(BatchConfig {
            max_batch: batch_max.max(1),
            max_wait: Duration::from_millis(batch_wait_ms),
        })
    };
    let api_key = args.api_key.or_else(|| server.api_key.clone());
    if (enable_admin || enable_metrics) && api_key.is_none() {
        tracing::warn!(
            "the model management/metrics APIs are enabled without an API key; keep the bind address private"
        );
    }
    let cors_origins = if args.no_cors {
        Some(Vec::new())
    } else if !args.cors_origins.is_empty() {
        Some(args.cors_origins.clone())
    } else {
        server.cors_origins.clone()
    };
    let server_config = ServerConfig {
        api_key,
        batching,
        enable_admin,
        enable_metrics,
        cors_origins,
        ..ServerConfig::default()
    };
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()?;
    runtime.block_on(serve_with_extensions_and_shutdown(
        bind,
        registry,
        server_config,
        ServerExtensions {
            resolver: Some(resolver),
            cache: Some(cache_manager),
            reload: reloader,
            telemetry: Some(Arc::new(TracingTelemetry)),
        },
        shutdown_signal(),
    ))?;
    tracing::info!("server stopped");
    Ok(())
}

async fn shutdown_signal() {
    #[cfg(unix)]
    {
        let mut terminate =
            tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
                .expect("install SIGTERM handler");
        tokio::select! {
            _ = tokio::signal::ctrl_c() => {},
            _ = terminate.recv() => {},
        }
    }
    #[cfg(not(unix))]
    {
        let _ = tokio::signal::ctrl_c().await;
    }
    tracing::info!("shutdown signal received; draining in-flight requests");
}

fn run_command(
    cache: Option<&Path>,
    offline: bool,
    max_cache_bytes: Option<u64>,
    config: &LoadedConfig,
    args: RunArgs,
) -> anyhow::Result<()> {
    let server = &config.value.server;
    let device = args
        .device
        .or_else(|| server.device.clone())
        .unwrap_or_else(|| "auto".to_owned());
    let dtype = args
        .dtype
        .or_else(|| server.dtype.clone())
        .unwrap_or_else(|| "auto".to_owned());
    let confidence = args
        .laya_confidence
        .or_else(|| server.laya_confidence.clone())
        .unwrap_or_else(|| "typesafe".to_owned());
    let max_len = args.max_len.or(server.max_len);
    let max_attention_elements = args
        .max_attention_elements
        .or(server.max_attention_elements)
        .unwrap_or(DEFAULT_MAX_ATTENTION_ELEMENTS);
    let local_window = !args.no_local_window && server.local_window.unwrap_or(true);
    let tf32 = args.tf32 || server.tf32.unwrap_or(false);
    set_tf32(tf32);
    let auto_task_detection =
        args.laya_auto_task_detection || server.laya_auto_task_detection.unwrap_or(false);
    let router_default = args
        .laya_router_default
        .or_else(|| server.laya_router_default.clone())
        .unwrap_or_else(|| "multilingual".to_owned());
    let lang_guess = args
        .laya_lang_guess
        .or_else(|| server.laya_lang_guess.clone());
    let hub = make_hub(cache, offline, max_cache_bytes, config)?;
    let options = make_model_options(
        &device,
        &dtype,
        max_len,
        &confidence,
        local_window,
        max_attention_elements,
        auto_task_detection,
        &router_default,
        lang_guess,
    )?;
    let reference = config.model_reference(&args.reference);
    let pulled = pull_model(&hub, &reference)?;
    let families = config.value.families.to_options();
    let bundle = bundle_for_directory(&hub, &pulled.directory, &families)?;
    let input = read_input(&args.input)?;
    let value: serde_json::Value =
        serde_json::from_str(&input).context("input must be a JSON object")?;
    let state = State::from_value(
        value
            .get("state")
            .cloned()
            .context("input requires a `state` field")?,
    )?;
    let questions: IndexMap<String, Question> = serde_json::from_value(
        value
            .get("questions")
            .cloned()
            .context("input requires a `questions` field")?,
    )
    .context("invalid questions")?;
    let questions = Questions::from(questions);

    let model: Arc<dyn DecisionModel> =
        load_manifest(&pulled.manifest, &pulled.directory, &options, &bundle)?;
    let prediction_options = PredictionOptions {
        min_confidence: args.min_confidence.map(MinConfidence::scalar).transpose()?,
    };
    if args.long && prediction_options.min_confidence.is_some() {
        bail!("--min-confidence is not supported with --long; gate the aggregated result instead");
    }
    let output = if args.long {
        model.predict_long(
            &state,
            &questions,
            LongContextOptions {
                window: args.window,
                stride: args.stride,
                batch_size: args.long_batch,
            },
        )?
    } else {
        model.predict_with(&state, &questions, &prediction_options)?
    };
    let sys1_core::DecisionOutput {
        answers,
        usage,
        actions,
        windows,
        scored_windows,
        abstentions,
    } = output;
    let response = SystemOneResponse {
        model: pulled.manifest.name,
        answers: answers.into_map(),
        usage,
    };
    let mut value = serde_json::to_value(&response)?;
    if args.long {
        let object = value.as_object_mut().expect("response is an object");
        if let Some(answers) = object
            .get_mut("answers")
            .and_then(serde_json::Value::as_object_mut)
        {
            for (id, answer) in answers.iter_mut() {
                if let Some(window) = windows.get(id)
                    && let Some(answer) = answer.as_object_mut()
                {
                    answer.insert("window".to_owned(), serde_json::to_value(window)?);
                }
            }
        }
        if let Some(usage) = object
            .get_mut("usage")
            .and_then(serde_json::Value::as_object_mut)
        {
            usage.insert("windows".to_owned(), serde_json::json!(scored_windows));
        }
    }
    if !actions.is_empty() || !abstentions.is_empty() {
        let object = value.as_object_mut().expect("response is an object");
        if let Some(answers) = object
            .get_mut("answers")
            .and_then(serde_json::Value::as_object_mut)
        {
            for (id, answer) in answers.iter_mut() {
                let Some(answer) = answer.as_object_mut() else {
                    continue;
                };
                if let Some(action) = actions.get(id) {
                    answer.insert(
                        "action".to_owned(),
                        serde_json::json!({ "act_probability": action }),
                    );
                }
                if let Some(abstention) = abstentions.get(id) {
                    if abstention.low_confidence {
                        answer.insert("low_confidence".to_owned(), serde_json::json!(true));
                    }
                    answer.insert(
                        "abstention".to_owned(),
                        serde_json::to_value(abstention.state)?,
                    );
                    answer.insert(
                        "abstention_threshold".to_owned(),
                        serde_json::json!(abstention.threshold),
                    );
                }
            }
        }
    }
    println!("{}", serde_json::to_string_pretty(&value)?);
    Ok(())
}

fn doctor_command(
    cache: Option<&Path>,
    offline: bool,
    max_cache_bytes: Option<u64>,
    config: &LoadedConfig,
    args: DoctorArgs,
) -> anyhow::Result<()> {
    let hub = make_hub(cache, offline, max_cache_bytes, config)?;
    let device = resolve_device(&args.device)?;
    let dtype = parse_dtype(&args.dtype)?;
    println!("rust-sys1 doctor");
    println!("  cuda: {}", cfg!(feature = "cuda"));
    println!("  cudnn: {}", cfg!(feature = "cudnn"));
    println!("  metal: {}", cfg!(feature = "metal"));
    println!("  accelerate: {}", cfg!(feature = "accelerate"));
    println!("  onnx: {}", cfg!(feature = "onnx"));
    println!("  ort-cuda: {}", cfg!(feature = "ort-cuda"));
    println!("  ort-tensorrt: {}", cfg!(feature = "ort-tensorrt"));
    println!("  ort-directml: {}", cfg!(feature = "ort-directml"));
    println!("  ort-coreml: {}", cfg!(feature = "ort-coreml"));
    println!(
        "  ORT_DYLIB_PATH: {}",
        std::env::var("ORT_DYLIB_PATH").unwrap_or_else(|_| "(loader default)".to_owned())
    );
    println!("  device: {device:?}");
    println!("  dtype: {dtype:?}");
    println!(
        "  config: {}",
        config
            .path
            .as_deref()
            .map(Path::display)
            .map(|path| path.to_string())
            .unwrap_or_else(|| "(defaults)".to_owned())
    );
    println!("  cache: {}", hub.config().cache_dir.display());
    let usage = hub.cache_usage()?;
    println!(
        "  cache limit: {}",
        usage
            .limit_bytes
            .map(format_bytes)
            .unwrap_or_else(|| "unlimited".to_owned())
    );
    println!(
        "  cache used: {} ({} reclaimable; {} models, {} resident)",
        format_bytes(usage.used_bytes),
        format_bytes(usage.reclaimable_bytes),
        usage.model_count,
        usage.resident_count
    );
    println!("  hf endpoint: {}", hub.config().hf_endpoint);
    println!(
        "  modelscope endpoint: {}",
        hub.config().modelscope_endpoint
    );
    println!(
        "  timeouts: request {}s, download attempt {}s",
        hub.config().timeout.as_secs(),
        hub.config().download_timeout.as_secs()
    );
    println!("  offline: {}", hub.config().offline);
    Ok(())
}

struct RegistryResidency(Arc<ModelRegistry>);

impl sys1_hub::CacheResidency for RegistryResidency {
    fn is_resident(&self, model_name: &str) -> bool {
        self.0.is_loaded(model_name)
    }
}

fn format_bytes(bytes: u64) -> String {
    const UNITS: [&str; 4] = ["KiB", "MiB", "GiB", "TiB"];
    if bytes < 1024 {
        return format!("{bytes} B");
    }
    let mut value = bytes as f64 / 1024.0;
    let mut unit = 0;
    while value >= 1024.0 && unit + 1 < UNITS.len() {
        value /= 1024.0;
        unit += 1;
    }
    format!("{value:.1} {}", UNITS[unit])
}

fn make_hub(
    cache: Option<&Path>,
    offline: bool,
    max_cache_bytes: Option<u64>,
    loaded: &LoadedConfig,
) -> anyhow::Result<Hub> {
    let hub = &loaded.value.hub;
    let mut config = HubConfig::from_env();
    // Precedence for paths/endpoints: CLI > environment > sys1.toml > defaults.
    let env_set = |key: &str| std::env::var_os(key).is_some();
    if !env_set("SYS1_CACHE")
        && let Some(cache) = &hub.cache_dir
    {
        config.cache_dir = cache.clone();
    }
    if !env_set("HF_ENDPOINT")
        && let Some(endpoint) = &hub.hf_endpoint
    {
        config.hf_endpoint = endpoint.trim().trim_end_matches('/').to_owned();
    }
    if !env_set("MODELSCOPE_ENDPOINT")
        && let Some(endpoint) = &hub.modelscope_endpoint
    {
        config.modelscope_endpoint = endpoint.trim().trim_end_matches('/').to_owned();
    }
    if !env_set("HF_TOKEN")
        && let Some(token) = &hub.hf_token
    {
        config.hf_token = Some(token.clone());
    }
    if !env_set("MODELSCOPE_API_TOKEN")
        && let Some(token) = &hub.modelscope_token
    {
        config.modelscope_token = Some(token.clone());
    }
    if !env_set("SYS1_HUB_TOKEN")
        && let Some(token) = &hub.http_token
    {
        config.http_token = Some(token.clone());
    }
    if !env_set("SYS1_HTTP_TIMEOUT_SECS")
        && let Some(secs) = hub.timeout_secs
    {
        config.timeout = Duration::from_secs(secs.max(1));
    }
    if !env_set("SYS1_HUB_DOWNLOAD_TIMEOUT_SECS")
        && let Some(secs) = hub.download_timeout_secs
    {
        config.download_timeout = Duration::from_secs(secs.max(1));
    }
    if let Some(limit) = max_cache_bytes {
        config.max_cache_bytes = (limit > 0).then_some(limit);
    } else if !env_set("SYS1_MAX_CACHE_BYTES")
        && let Some(limit) = hub.max_cache_bytes
    {
        config.max_cache_bytes = (limit > 0).then_some(limit);
    }
    if hub.offline.unwrap_or(false) || offline {
        config.offline = true;
    }
    if let Some(cache) = cache {
        config.cache_dir = cache.to_path_buf();
    }
    Ok(Hub::new(config)?)
}

#[allow(clippy::too_many_arguments)]
fn make_model_options(
    device: &str,
    dtype: &str,
    max_len: Option<usize>,
    confidence: &str,
    local_window: bool,
    max_attention_elements: usize,
    auto_task_detection: bool,
    router_default: &str,
    lang_guess: Option<String>,
) -> anyhow::Result<ModelOptions> {
    Ok(ModelOptions {
        device: resolve_device(device)?,
        dtype: parse_dtype(dtype)?,
        max_len,
        confidence_mode: parse_confidence(confidence)?,
        local_window,
        max_attention_elements,
        router: LayaRouterOptions {
            default: parse_router_default(router_default)?,
            auto_task_detection,
        },
        lang_guess,
    })
}

fn parse_router_default(value: &str) -> anyhow::Result<RouteVariant> {
    RouteVariant::from_name(value).ok_or_else(|| {
        anyhow::anyhow!(
            "unknown router default `{value}`; expected english, multilingual, or typed-decisions"
        )
    })
}

fn resolve_device(value: &str) -> anyhow::Result<Device> {
    let preference: DevicePreference = value.parse()?;
    match preference {
        DevicePreference::Auto => {
            #[cfg(feature = "cuda")]
            if let Ok(device) = Device::new_cuda(0) {
                tracing::info!("auto-selected CUDA device 0");
                return Ok(device);
            }
            #[cfg(feature = "metal")]
            if let Ok(device) = Device::new_metal(0) {
                tracing::info!("auto-selected Metal device");
                return Ok(device);
            }
            tracing::info!("auto-selected CPU");
            Ok(Device::Cpu)
        }
        DevicePreference::Cpu => Ok(Device::Cpu),
        DevicePreference::Cuda(index) => {
            #[cfg(feature = "cuda")]
            {
                Device::new_cuda(index)
                    .map_err(|error| anyhow::anyhow!("CUDA device {index} unavailable: {error}"))
            }
            #[cfg(not(feature = "cuda"))]
            {
                let _ = index;
                bail!("this binary was built without the `cuda` feature")
            }
        }
        DevicePreference::Metal => {
            #[cfg(feature = "metal")]
            {
                Device::new_metal(0).map_err(|error| anyhow::anyhow!("Metal unavailable: {error}"))
            }
            #[cfg(not(feature = "metal"))]
            {
                bail!("this binary was built without the `metal` feature")
            }
        }
    }
}

fn parse_dtype(value: &str) -> anyhow::Result<DType> {
    match value.to_ascii_lowercase().as_str() {
        "auto" | "f32" => Ok(DType::F32),
        "f16" => Ok(DType::F16),
        "bf16" => Ok(DType::BF16),
        other => bail!("unsupported --dtype {other}; expected auto, f32, f16, or bf16"),
    }
}

fn parse_confidence(value: &str) -> anyhow::Result<ConfidenceMode> {
    match value.to_ascii_lowercase().as_str() {
        "typesafe" | "default" => Ok(ConfidenceMode::TypeSafe),
        "laya" | "native" => Ok(ConfidenceMode::LayaNative),
        other => bail!("unsupported --laya-confidence {other}; expected typesafe or laya"),
    }
}

fn build_models(
    hub: &Hub,
    specs: &[String],
    options: ModelOptions,
    families: &FamilyOptions,
) -> anyhow::Result<(Vec<ModelManifest>, Arc<FamilyFactory>)> {
    let factory = Arc::new(FamilyFactory::new());
    let mut manifests = Vec::new();
    let mut seen = HashSet::new();

    if specs.is_empty() {
        for pulled in hub.list_cached()? {
            if !is_supported_directory(&pulled.directory) {
                continue;
            }
            let bundle = match bundle_for_directory(hub, &pulled.directory, families) {
                Ok(bundle) => bundle,
                Err(error) => {
                    tracing::warn!(
                        model = %pulled.manifest.name,
                        "skipping cached model: {error:#}"
                    );
                    continue;
                }
            };
            let registered = register_bundle(
                &factory,
                &pulled.manifest,
                &pulled.directory,
                &options,
                &bundle,
                None,
            )?;
            for manifest in registered {
                push_manifest(&mut manifests, &mut seen, manifest);
            }
        }
        return Ok((manifests, factory));
    }

    for spec in specs {
        let (name, reference) = parse_model_spec(spec)?;
        let pulled = pull_model(hub, &reference)?;
        if !is_supported_directory(&pulled.directory) {
            bail!(
                "{reference} does not look like a supported model directory (Laya, NLI, GLiClass, Qwen3Guard, Von, Decima, Kev, Decision, or Clef)"
            );
        }
        if name == "laya-latest" && !has_checkpoint_variants(&pulled.directory) {
            tracing::warn!(
                "laya-latest has no multilingual/typed-decisions subdirectories; serving one checkpoint"
            );
        }
        let bundle = bundle_for_directory(hub, &pulled.directory, families)?;
        let registered = register_bundle(
            &factory,
            &pulled.manifest,
            &pulled.directory,
            &options,
            &bundle,
            Some(&name),
        )?;
        for manifest in registered {
            push_manifest(&mut manifests, &mut seen, manifest);
        }
    }
    Ok((manifests, factory))
}

fn push_manifest(
    manifests: &mut Vec<ModelManifest>,
    seen: &mut HashSet<String>,
    manifest: ModelManifest,
) {
    if seen.insert(manifest.name.clone()) {
        manifests.push(manifest);
    }
}

fn parse_model_spec(spec: &str) -> anyhow::Result<(String, String)> {
    let (name, reference) = spec
        .split_once('=')
        .with_context(|| format!("expected NAME=REF, got {spec:?}"))?;
    let name = name.trim();
    let reference = reference.trim();
    if name.is_empty() || reference.is_empty() {
        bail!("expected NAME=REF, got {spec:?}");
    }
    Ok((name.to_owned(), reference.to_owned()))
}

fn bundle_for_directory(
    hub: &Hub,
    directory: &Path,
    families: &FamilyOptions,
) -> anyhow::Result<BundleOptions> {
    let overrides = resolve_bundle_overrides(directory, families, |base_reference| {
        pull_base(hub, base_reference)
    })?;
    Ok(BundleOptions {
        families: families.clone(),
        overrides,
    })
}

pub(crate) fn pull_base(hub: &Hub, reference: &str) -> Result<PathBuf, ModelError> {
    pull_model(hub, reference)
        .map(|pulled| pulled.directory)
        .map_err(|error| {
            ModelError::load(format!("cannot resolve Kev base `{reference}`: {error:#}"))
        })
}

fn pull_model(hub: &Hub, reference: &str) -> anyhow::Result<PulledModel> {
    let parsed: ModelRef = reference.parse()?;
    if parsed.scheme == "local" {
        let path = PathBuf::from(&parsed.location);
        if path.is_dir() && !path.join("model.json").is_file() {
            return synthesise_local(&path);
        }
    }
    Ok(hub.pull(reference)?)
}

fn synthesise_local(directory: &Path) -> anyhow::Result<PulledModel> {
    let (manifest, files) = match local_manifest(directory) {
        Ok(value) => value,
        Err(_) => sys1_laya::local_checkpoint_manifest(directory)?,
    };
    Ok(PulledModel {
        manifest,
        directory: directory.to_path_buf(),
        files,
    })
}

fn read_input(input: &str) -> anyhow::Result<String> {
    if input == "-" {
        let mut buffer = String::new();
        std::io::stdin()
            .read_to_string(&mut buffer)
            .context("read stdin")?;
        Ok(buffer)
    } else {
        fs::read_to_string(input).with_context(|| format!("read {input}"))
    }
}
