mod config;
mod source;

use anyhow::{Context, bail};
use candle_core::{DType, Device};
use clap::{Parser, Subcommand};
use config::LoadedConfig;
use indexmap::IndexMap;
use source::HubModelResolver;
use std::collections::HashSet;
use std::fs;
use std::io::Read;
use std::net::SocketAddr;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;
use sys1_core::{
    DecisionModel, DevicePreference, ModelManifest, ModelRegistry, ModelResolver, Questions,
    RegistryConfig,
};
use sys1_hub::{Hub, HubConfig, ModelRef, PulledModel};
use sys1_laya::{
    ConfidenceMode, DEFAULT_MAX_ATTENTION_ELEMENTS, LayaFactory as CheckpointFactory,
    LayaModel as CheckpointModel, LayaOptions as ModelOptions, has_checkpoint_variants,
    is_checkpoint_directory, local_checkpoint_manifest, register_checkpoint_bundle, set_tf32,
};
use sys1_protocol::{Question, State, SystemOneResponse};
use sys1_server::{BatchConfig, ServerConfig, serve_with_resolver};
use tracing_subscriber::EnvFilter;

#[derive(Parser)]
#[command(
    name = "sys1",
    version,
    about = "System One inference server and model hub"
)]
struct Cli {
    #[arg(long, global = true, env = "SYS1_CACHE", value_name = "DIR")]
    cache: Option<PathBuf>,
    /// Config file (default: ./sys1.toml when present).
    #[arg(long, global = true, env = "SYS1_CONFIG", value_name = "FILE")]
    config: Option<PathBuf>,
    #[arg(long, global = true)]
    offline: bool,
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
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")),
        )
        .init();

    let cli = Cli::parse();
    let cache = cli.cache.clone();
    let offline = cli.offline;
    let config = config::load(cli.config.as_deref())?;
    match cli.command {
        Command::Serve(args) => serve_command(cache.as_deref(), offline, &config, args),
        Command::Pull(args) => {
            let hub = make_hub(cache.as_deref(), offline, &config)?;
            let reference = config.model_reference(&args.reference);
            let pulled = pull_model(&hub, &reference)?;
            println!("{} -> {}", pulled.manifest.name, pulled.directory.display());
            Ok(())
        }
        Command::List => {
            let hub = make_hub(cache.as_deref(), offline, &config)?;
            for model in hub.list_cached()? {
                println!(
                    "{:<28} {:<10} {:<12} {}",
                    model.manifest.name,
                    model.manifest.backend,
                    model.manifest.format,
                    model.directory.display()
                );
            }
            Ok(())
        }
        Command::Show(args) => {
            let hub = make_hub(cache.as_deref(), offline, &config)?;
            let reference = config.model_reference(&args.reference);
            let manifest = hub.resolve(&reference)?;
            println!("{}", serde_json::to_string_pretty(&manifest)?);
            Ok(())
        }
        Command::Rm(args) => {
            let hub = make_hub(cache.as_deref(), offline, &config)?;
            let reference = config.model_reference(&args.reference);
            let location = hub.cache_location(&reference)?;
            if hub.remove(&reference)? {
                println!("removed {}", location.display());
            } else {
                println!("not cached: {}", location.display());
            }
            Ok(())
        }
        Command::Run(args) => run_command(cache.as_deref(), offline, &config, args),
        Command::Doctor(args) => doctor_command(cache.as_deref(), offline, &config, args),
    }
}

fn serve_command(
    cache: Option<&Path>,
    offline: bool,
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
    let options = make_model_options(
        &device,
        &dtype,
        max_len,
        &confidence,
        local_window,
        max_attention_elements,
    )?;
    let specs = if args.models.is_empty() {
        config.model_specs()
    } else {
        args.models
    };
    let enable_admin = !args.no_admin && server.admin.unwrap_or(true);
    let hub = make_hub(cache, offline, config)?;
    let (manifests, factory) = build_models(&hub, &specs, options.clone())?;
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
    let resolver: Arc<dyn ModelResolver> = Arc::new(HubModelResolver::new(
        hub,
        factory.clone(),
        options,
        config.value.models.clone(),
    ));
    let registry = Arc::new(ModelRegistry::new(
        manifests,
        factory,
        RegistryConfig::default(),
    )?);

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
    if enable_admin && api_key.is_none() {
        tracing::warn!(
            "the model management API is enabled without an API key; keep the bind address private"
        );
    }
    let server_config = ServerConfig {
        api_key,
        batching,
        enable_admin,
        ..ServerConfig::default()
    };
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()?;
    runtime.block_on(serve_with_resolver(
        bind,
        registry,
        server_config,
        Some(resolver),
    ))?;
    Ok(())
}

fn run_command(
    cache: Option<&Path>,
    offline: bool,
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
    let hub = make_hub(cache, offline, config)?;
    let options = make_model_options(
        &device,
        &dtype,
        max_len,
        &confidence,
        local_window,
        max_attention_elements,
    )?;
    let reference = config.model_reference(&args.reference);
    let pulled = pull_model(&hub, &reference)?;
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

    let model = CheckpointModel::load_dir(&pulled.directory, options)?;
    let output = model.predict(&state, &questions)?;
    let response = SystemOneResponse {
        model: pulled.manifest.name,
        answers: output.answers.into_map(),
        usage: output.usage,
    };
    println!("{}", serde_json::to_string_pretty(&response)?);
    Ok(())
}

fn doctor_command(
    cache: Option<&Path>,
    offline: bool,
    config: &LoadedConfig,
    args: DoctorArgs,
) -> anyhow::Result<()> {
    let hub = make_hub(cache, offline, config)?;
    let device = resolve_device(&args.device)?;
    let dtype = parse_dtype(&args.dtype)?;
    println!("rust-sys1 doctor");
    println!("  cuda: {}", cfg!(feature = "cuda"));
    println!("  cudnn: {}", cfg!(feature = "cudnn"));
    println!("  metal: {}", cfg!(feature = "metal"));
    println!("  accelerate: {}", cfg!(feature = "accelerate"));
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
    println!("  hf endpoint: {}", hub.config().hf_endpoint);
    println!(
        "  modelscope endpoint: {}",
        hub.config().modelscope_endpoint
    );
    println!("  offline: {}", hub.config().offline);
    Ok(())
}

fn make_hub(cache: Option<&Path>, offline: bool, loaded: &LoadedConfig) -> anyhow::Result<Hub> {
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
    if hub.offline.unwrap_or(false) || offline {
        config.offline = true;
    }
    if let Some(cache) = cache {
        config.cache_dir = cache.to_path_buf();
    }
    Ok(Hub::new(config)?)
}

fn make_model_options(
    device: &str,
    dtype: &str,
    max_len: Option<usize>,
    confidence: &str,
    local_window: bool,
    max_attention_elements: usize,
) -> anyhow::Result<ModelOptions> {
    Ok(ModelOptions {
        device: resolve_device(device)?,
        dtype: parse_dtype(dtype)?,
        max_len,
        confidence_mode: parse_confidence(confidence)?,
        local_window,
        max_attention_elements,
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
) -> anyhow::Result<(Vec<ModelManifest>, Arc<CheckpointFactory>)> {
    let factory = Arc::new(CheckpointFactory::new());
    let mut manifests = Vec::new();
    let mut seen = HashSet::new();

    if specs.is_empty() {
        for pulled in hub.list_cached()? {
            if !is_checkpoint_directory(&pulled.directory) {
                continue;
            }
            let registered = register_checkpoint_bundle(
                &factory,
                &pulled.manifest,
                &pulled.directory,
                options.clone(),
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
        if !is_checkpoint_directory(&pulled.directory) {
            bail!("{reference} does not look like a Laya safetensors checkpoint");
        }
        if name == "laya-latest" && !has_checkpoint_variants(&pulled.directory) {
            tracing::warn!(
                "laya-latest has no multilingual/typed-decisions subdirectories; serving one checkpoint"
            );
        }
        let registered = register_checkpoint_bundle(
            &factory,
            &pulled.manifest,
            &pulled.directory,
            options.clone(),
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
    let (manifest, files) = local_checkpoint_manifest(directory)?;
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
