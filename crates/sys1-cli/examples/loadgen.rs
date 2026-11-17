//! Load generator for the System One HTTP server.
//!
//! Closed-loop mode (default) keeps `--concurrency` requests in flight.
//! Open-loop mode (`--rate`) offers requests at a fixed rate and caps the
//! number of in-flight requests at `--concurrency`; requests that cannot get a
//! permit are reported as dropped.

use clap::Parser;
use serde_json::json;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::Semaphore;
use tokio::task::JoinSet;

#[derive(Parser)]
#[command(about = "System One HTTP load generator")]
struct Args {
    #[arg(long, default_value = "http://127.0.0.1:8080")]
    url: String,
    #[arg(long, default_value = "laya-multilingual")]
    model: String,
    /// Closed-loop workers, or the in-flight cap in open-loop mode.
    #[arg(long, default_value_t = 32)]
    concurrency: usize,
    /// Total measured requests.
    #[arg(long, default_value_t = 256)]
    requests: usize,
    #[arg(long, default_value_t = 3)]
    warmup: usize,
    /// Open-loop offered rate in requests per second.
    #[arg(long)]
    rate: Option<f64>,
    #[arg(long, default_value_t = 5000)]
    timeout_ms: u64,
    #[arg(
        long,
        default_value = "I was charged twice. Please refund the extra charge today."
    )]
    state: String,
}

struct ResultSet {
    latencies: Vec<f64>,
    errors: usize,
    dropped: usize,
    wall: Duration,
}

impl ResultSet {
    fn new() -> Self {
        Self {
            latencies: Vec::new(),
            errors: 0,
            dropped: 0,
            wall: Duration::ZERO,
        }
    }

    fn record(&mut self, latency_ms: f64) {
        self.latencies.push(latency_ms);
    }

    fn percentile(&self, p: f64) -> f64 {
        if self.latencies.is_empty() {
            return 0.0;
        }
        let mut sorted = self.latencies.clone();
        sorted.sort_by(f64::total_cmp);
        let index = ((sorted.len() as f64 * p).ceil() as usize)
            .saturating_sub(1)
            .min(sorted.len() - 1);
        sorted[index]
    }

    fn summary(&self, label: &str) {
        let count = self.latencies.len();
        let mean = if count == 0 {
            0.0
        } else {
            self.latencies.iter().sum::<f64>() / count as f64
        };
        let rps = count as f64 / self.wall.as_secs_f64().max(1e-9);
        println!(
            "{label:<24} {:>7}  {:>7.2}  {:>7.2}  {:>7.2}  {:>7.2}  {:>9.2}  {:>6}  {:>6}",
            count,
            mean,
            self.percentile(0.50),
            self.percentile(0.95),
            self.percentile(0.99),
            rps,
            self.errors,
            self.dropped,
        );
    }
}

fn body(model: &str, state: &str, tag: &str) -> String {
    json!({
        "model": model,
        "state": format!("{state} Case {tag}."),
        "questions": {
            "billing": {
                "type": "noul",
                "instructions": "Is this message about billing?"
            }
        }
    })
    .to_string()
}

async fn one_request(
    client: &reqwest::Client,
    url: &str,
    model: &str,
    state: &str,
    tag: &str,
) -> Result<f64, String> {
    let started = Instant::now();
    let response = client
        .post(url)
        .header("content-type", "application/json")
        .body(body(model, state, tag))
        .send()
        .await
        .map_err(|error| error.to_string())?;
    if !response.status().is_success() {
        let status = response.status();
        let text = response.text().await.unwrap_or_default();
        return Err(format!("HTTP {status}: {text}"));
    }
    let _ = response.bytes().await.map_err(|error| error.to_string())?;
    Ok(started.elapsed().as_secs_f64() * 1000.0)
}

async fn warmup(client: &reqwest::Client, args: &Args) {
    for index in 0..args.warmup {
        if let Err(error) = one_request(
            client,
            &args.url,
            &args.model,
            &args.state,
            &format!("warmup-{index}"),
        )
        .await
        {
            eprintln!("warmup request failed: {error}");
        }
    }
}

async fn closed_loop(client: reqwest::Client, args: &Args) -> ResultSet {
    let workers = args.concurrency.max(1);
    let total = args.requests.max(1);
    let base = total / workers;
    let extra = total % workers;
    let mut set = JoinSet::new();

    for worker in 0..workers {
        let count = base + usize::from(worker < extra);
        if count == 0 {
            continue;
        }
        let client = client.clone();
        let url = args.url.clone();
        let model = args.model.clone();
        let state = args.state.clone();
        set.spawn(async move {
            let mut latencies = Vec::with_capacity(count);
            let mut errors = 0;
            for index in 0..count {
                match one_request(&client, &url, &model, &state, &format!("{worker}-{index}")).await
                {
                    Ok(latency) => latencies.push(latency),
                    Err(error) => {
                        errors += 1;
                        if errors <= 3 {
                            eprintln!("request error: {error}");
                        }
                    }
                }
            }
            (latencies, errors)
        });
    }

    let started = Instant::now();
    let mut result = ResultSet::new();
    while let Some(joined) = set.join_next().await {
        match joined {
            Ok((latencies, errors)) => {
                result.latencies.extend(latencies);
                result.errors += errors;
            }
            Err(error) => {
                result.errors += 1;
                eprintln!("worker failed: {error}");
            }
        }
    }
    result.wall = started.elapsed();
    result
}

async fn open_loop(client: reqwest::Client, args: &Args, rate: f64) -> ResultSet {
    let semaphore = Arc::new(Semaphore::new(args.concurrency.max(1)));
    let mut set = JoinSet::new();
    let mut result = ResultSet::new();
    let started = Instant::now();
    let period = 1.0 / rate.max(1e-9);

    for index in 0..args.requests {
        // Windows timer granularity is coarse enough to distort 2–5 ms ticks;
        // yield-spin instead so the offered rate is accurate.
        let target = started + Duration::from_secs_f64(period * index as f64);
        while Instant::now() < target {
            tokio::task::yield_now().await;
        }
        let Ok(permit) = semaphore.clone().try_acquire_owned() else {
            result.dropped += 1;
            continue;
        };
        let client = client.clone();
        let url = args.url.clone();
        let model = args.model.clone();
        let state = args.state.clone();
        let tag = format!("offer-{index}");
        set.spawn(async move {
            let outcome = one_request(&client, &url, &model, &state, &tag).await;
            drop(permit);
            outcome
        });
    }

    while let Some(joined) = set.join_next().await {
        match joined {
            Ok(Ok(latency)) => result.record(latency),
            Ok(Err(error)) => {
                result.errors += 1;
                if result.errors <= 3 {
                    eprintln!("request error: {error}");
                }
            }
            Err(error) => {
                result.errors += 1;
                eprintln!("request task failed: {error}");
            }
        }
    }
    result.wall = started.elapsed();
    result
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = Args::parse();
    if let Some(rate) = args.rate
        && rate <= 0.0
    {
        return Err("--rate must be greater than zero".into());
    }
    if !args.url.ends_with("/v1/systemone") {
        args.url = format!("{}/v1/systemone", args.url.trim_end_matches('/'));
    }
    let client = reqwest::Client::builder()
        .timeout(Duration::from_millis(args.timeout_ms))
        .pool_max_idle_per_host(args.concurrency.max(16) * 2)
        .build()?;

    warmup(&client, &args).await;

    println!(
        "{:<24} {:>7}  {:>7}  {:>7}  {:>7}  {:>7}  {:>9}  {:>6}  {:>6}",
        "scenario", "count", "mean", "p50", "p95", "p99", "req/s", "errors", "dropped"
    );

    let label = if let Some(rate) = args.rate {
        format!("open-loop {rate:.1} rps")
    } else {
        format!("closed-loop x{}", args.concurrency)
    };
    let result = match args.rate {
        Some(rate) => open_loop(client, &args, rate).await,
        None => closed_loop(client, &args).await,
    };
    result.summary(&label);
    Ok(())
}
