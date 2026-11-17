use serde::Serialize;
use std::collections::{BTreeMap, HashMap};
use std::fmt::Write;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

pub(crate) struct Metrics {
    started: Instant,
    systemone_requests: AtomicU64,
    systemone_errors: AtomicU64,
    systemone_latency_us: AtomicU64,
    systemone_latency_us_max: AtomicU64,
    model_requests: Mutex<HashMap<String, u64>>,
    batches: AtomicU64,
    batch_items: AtomicU64,
    batch_items_max: AtomicU64,
    batch_duration_us: AtomicU64,
    model_loads: AtomicU64,
    model_load_errors: AtomicU64,
    model_load_us: AtomicU64,
    model_unloads: AtomicU64,
}

impl Metrics {
    pub(crate) fn new() -> Self {
        Self {
            started: Instant::now(),
            systemone_requests: AtomicU64::new(0),
            systemone_errors: AtomicU64::new(0),
            systemone_latency_us: AtomicU64::new(0),
            systemone_latency_us_max: AtomicU64::new(0),
            model_requests: Mutex::new(HashMap::new()),
            batches: AtomicU64::new(0),
            batch_items: AtomicU64::new(0),
            batch_items_max: AtomicU64::new(0),
            batch_duration_us: AtomicU64::new(0),
            model_loads: AtomicU64::new(0),
            model_load_errors: AtomicU64::new(0),
            model_load_us: AtomicU64::new(0),
            model_unloads: AtomicU64::new(0),
        }
    }

    pub(crate) fn record_systemone(&self, duration: Duration, success: bool, model: Option<&str>) {
        self.systemone_requests.fetch_add(1, Ordering::Relaxed);
        if !success {
            self.systemone_errors.fetch_add(1, Ordering::Relaxed);
        }
        add_micros(&self.systemone_latency_us, duration);
        self.systemone_latency_us_max.fetch_max(
            duration.as_micros().min(u64::MAX as u128) as u64,
            Ordering::Relaxed,
        );
        if let Some(model) = model {
            let mut requests = self
                .model_requests
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            *requests.entry(model.to_owned()).or_insert(0) += 1;
        }
    }

    pub(crate) fn record_batch(&self, items: usize, duration: Duration) {
        self.batches.fetch_add(1, Ordering::Relaxed);
        self.batch_items.fetch_add(items as u64, Ordering::Relaxed);
        self.batch_items_max
            .fetch_max(items as u64, Ordering::Relaxed);
        add_micros(&self.batch_duration_us, duration);
    }

    pub(crate) fn record_model_load(&self, duration: Duration, success: bool) {
        self.model_loads.fetch_add(1, Ordering::Relaxed);
        if !success {
            self.model_load_errors.fetch_add(1, Ordering::Relaxed);
        }
        add_micros(&self.model_load_us, duration);
    }

    pub(crate) fn record_model_unload(&self) {
        self.model_unloads.fetch_add(1, Ordering::Relaxed);
    }

    pub(crate) fn snapshot(&self, loaded: Vec<String>) -> MetricsSnapshot {
        let requests = self.systemone_requests.load(Ordering::Relaxed);
        let loads = self.model_loads.load(Ordering::Relaxed);
        let batches = self.batches.load(Ordering::Relaxed);
        let model_requests = self
            .model_requests
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .iter()
            .map(|(model, count)| (model.clone(), *count))
            .collect();

        MetricsSnapshot {
            uptime_secs: self.started.elapsed().as_secs(),
            systemone: SystemOneMetrics {
                requests,
                errors: self.systemone_errors.load(Ordering::Relaxed),
                latency_ms_mean: mean_ms(
                    self.systemone_latency_us.load(Ordering::Relaxed),
                    requests,
                ),
                latency_ms_max: micros_to_ms(self.systemone_latency_us_max.load(Ordering::Relaxed)),
            },
            batching: BatchMetrics {
                batches,
                items: self.batch_items.load(Ordering::Relaxed),
                mean_size: ratio(self.batch_items.load(Ordering::Relaxed), batches),
                max_size: self.batch_items_max.load(Ordering::Relaxed),
                mean_duration_ms: mean_ms(self.batch_duration_us.load(Ordering::Relaxed), batches),
            },
            models: ModelMetrics {
                loads,
                load_errors: self.model_load_errors.load(Ordering::Relaxed),
                mean_load_ms: mean_ms(self.model_load_us.load(Ordering::Relaxed), loads),
                unloads: self.model_unloads.load(Ordering::Relaxed),
                loaded_count: loaded.len(),
                requests: model_requests,
                loaded,
            },
        }
    }

    pub(crate) fn prometheus(&self, loaded: Vec<String>) -> String {
        let snapshot = self.snapshot(loaded);
        let mut out = String::new();

        metric_gauge(
            &mut out,
            "sys1_uptime_seconds",
            "Seconds since the server started.",
            snapshot.uptime_secs as f64,
        );
        metric_counter(
            &mut out,
            "sys1_systemone_requests_total",
            "System One requests handled.",
            snapshot.systemone.requests as f64,
        );
        metric_counter(
            &mut out,
            "sys1_systemone_errors_total",
            "System One requests that returned an error.",
            snapshot.systemone.errors as f64,
        );
        metric_counter(
            &mut out,
            "sys1_systemone_latency_ms_sum",
            "Sum of System One request latency in milliseconds.",
            (snapshot.systemone.latency_ms_mean * snapshot.systemone.requests as f64).max(0.0),
        );
        metric_gauge(
            &mut out,
            "sys1_systemone_latency_ms_max",
            "Maximum System One request latency in milliseconds.",
            snapshot.systemone.latency_ms_max,
        );
        metric_counter(
            &mut out,
            "sys1_batches_total",
            "Model batches processed.",
            snapshot.batching.batches as f64,
        );
        metric_counter(
            &mut out,
            "sys1_batch_items_total",
            "Requests processed through batches.",
            snapshot.batching.items as f64,
        );
        metric_gauge(
            &mut out,
            "sys1_batch_items_max",
            "Largest observed batch.",
            snapshot.batching.max_size as f64,
        );
        metric_counter(
            &mut out,
            "sys1_batch_duration_ms_sum",
            "Sum of batch execution time in milliseconds.",
            snapshot.batching.mean_duration_ms * snapshot.batching.batches as f64,
        );
        metric_counter(
            &mut out,
            "sys1_model_loads_total",
            "Model load attempts.",
            snapshot.models.loads as f64,
        );
        metric_counter(
            &mut out,
            "sys1_model_load_errors_total",
            "Failed model load attempts.",
            snapshot.models.load_errors as f64,
        );
        metric_counter(
            &mut out,
            "sys1_model_load_duration_ms_sum",
            "Sum of model load time in milliseconds.",
            snapshot.models.mean_load_ms * snapshot.models.loads as f64,
        );
        metric_counter(
            &mut out,
            "sys1_model_unloads_total",
            "Explicit model unloads.",
            snapshot.models.unloads as f64,
        );
        metric_gauge(
            &mut out,
            "sys1_loaded_models",
            "Models currently resident.",
            snapshot.models.loaded_count as f64,
        );

        if !snapshot.models.requests.is_empty() {
            let _ = writeln!(
                out,
                "# HELP sys1_model_requests_total System One requests per catalog model.\n\
                 # TYPE sys1_model_requests_total counter"
            );
            for (model, count) in &snapshot.models.requests {
                let _ = writeln!(
                    out,
                    "sys1_model_requests_total{{model=\"{}\"}} {count}",
                    escape_label(model)
                );
            }
        }
        out
    }
}

#[derive(Serialize)]
pub(crate) struct MetricsSnapshot {
    pub uptime_secs: u64,
    pub systemone: SystemOneMetrics,
    pub batching: BatchMetrics,
    pub models: ModelMetrics,
}

#[derive(Serialize)]
pub(crate) struct SystemOneMetrics {
    pub requests: u64,
    pub errors: u64,
    pub latency_ms_mean: f64,
    pub latency_ms_max: f64,
}

#[derive(Serialize)]
pub(crate) struct BatchMetrics {
    pub batches: u64,
    pub items: u64,
    pub mean_size: f64,
    pub max_size: u64,
    pub mean_duration_ms: f64,
}

#[derive(Serialize)]
pub(crate) struct ModelMetrics {
    pub loads: u64,
    pub load_errors: u64,
    pub mean_load_ms: f64,
    pub unloads: u64,
    pub loaded_count: usize,
    pub requests: BTreeMap<String, u64>,
    pub loaded: Vec<String>,
}

fn add_micros(total: &AtomicU64, duration: Duration) {
    total.fetch_add(
        duration.as_micros().min(u64::MAX as u128) as u64,
        Ordering::Relaxed,
    );
}

fn micros_to_ms(micros: u64) -> f64 {
    micros as f64 / 1000.0
}

fn mean_ms(total_micros: u64, count: u64) -> f64 {
    if count == 0 {
        0.0
    } else {
        micros_to_ms(total_micros) / count as f64
    }
}

fn ratio(total: u64, count: u64) -> f64 {
    if count == 0 {
        0.0
    } else {
        total as f64 / count as f64
    }
}

fn metric_counter(out: &mut String, name: &str, help: &str, value: f64) {
    let _ = writeln!(
        out,
        "# HELP {name} {help}\n# TYPE {name} counter\n{name} {}",
        format_value(value)
    );
}

fn metric_gauge(out: &mut String, name: &str, help: &str, value: f64) {
    let _ = writeln!(
        out,
        "# HELP {name} {help}\n# TYPE {name} gauge\n{name} {}",
        format_value(value)
    );
}

fn format_value(value: f64) -> String {
    if value.fract() == 0.0 && value.abs() < 1e15 {
        format!("{value:.0}")
    } else {
        format!("{value:.3}")
    }
}

fn escape_label(value: &str) -> String {
    value
        .replace('\\', "\\\\")
        .replace('"', "\\\"")
        .replace('\n', "\\n")
}
