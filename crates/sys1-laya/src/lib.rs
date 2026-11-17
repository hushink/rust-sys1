//! Laya decision model implementation.
//!
//! This is an independent implementation of the Laya architecture. The
//! reference implementation used for behavioral comparison is the official
//! Laya repository and `sys1` 0.0.3 (Apache-2.0). No code or runtime
//! dependency from those projects is required.

mod bundle;
mod calibration;
mod config;
mod factory;
mod language;
mod model;
mod modernbert;
#[cfg(feature = "onnx")]
mod onnx;
mod onnx_config;
mod prepare;
mod response;
mod router;
mod tokenizer;

pub use bundle::{
    has_checkpoint_variants, is_checkpoint_directory, is_model_directory, is_onnx_directory,
    local_checkpoint_manifest, register_checkpoint_bundle, variant_directories,
};
pub use calibration::ConfidenceMode;
pub use factory::LayaFactory;
pub use language::LanguageAnalysis;
pub use model::{LayaModel, LayaOptions};
#[cfg(feature = "onnx")]
pub use onnx::LayaOnnxModel;
pub use router::{LayaRouter, LayaRouterOptions, RouteDecision, RouteHints, RouteVariant};

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
