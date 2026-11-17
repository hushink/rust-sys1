//! Laya decision model implementation.
//!
//! This is an independent implementation of the Laya architecture. The
//! reference implementation used for behavioral comparison is the official
//! Laya repository and `sys1` 0.0.3 (Apache-2.0). No code or runtime
//! dependency from those projects is required.

mod calibration;
mod config;
mod factory;
mod model;
mod modernbert;
mod prepare;
mod router;
mod tokenizer;

pub use calibration::ConfidenceMode;
pub use factory::LayaFactory;
pub use model::{LayaModel, LayaOptions};
pub use router::LayaRouter;

/// Allow TF32 tensor-core reductions for F32 CUDA GEMMs.
///
/// This is a process-wide candle/cuBLAS setting; enable it before loading or
/// running CUDA models. On non-CUDA builds this is a no-op.
pub fn set_tf32(enabled: bool) {
    #[cfg(feature = "cuda")]
    candle_core::cuda_backend::set_gemm_reduced_precision_f32(enabled);
    #[cfg(not(feature = "cuda"))]
    let _ = enabled;
}

/// Default bound on `batch * heads * seq_len²` for one global-attention batch.
pub const DEFAULT_MAX_ATTENTION_ELEMENTS: usize = 64 * 1024 * 1024;

pub const LAYA_MODEL_ID: &str = "convaiinnovations/laya";
pub const LAYA_MULTILINGUAL_MODEL_ID: &str = "convaiinnovations/laya-multilingual";
pub const LAYA_TYPED_DECISIONS_MODEL_ID: &str = "convaiinnovations/laya-typed-decisions";
