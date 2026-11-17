use thiserror::Error;

/// Errors raised while loading or running an ONNX Runtime session.
#[derive(Debug, Error)]
pub enum OrtError {
    #[error("failed to load ONNX Runtime: {message}")]
    Runtime { message: String },
    #[error("failed to load ONNX Runtime session from {path}: {message}")]
    Load { path: String, message: String },
    #[error("invalid scoring input: {message}")]
    Input { message: String },
    #[error("ONNX Runtime inference failed: {message}")]
    Inference { message: String },
    #[error("ONNX Runtime output `{name}` is missing or has an unexpected type: {message}")]
    Output { name: String, message: String },
}

impl OrtError {
    pub(crate) fn runtime(error: impl std::fmt::Display) -> Self {
        Self::Runtime {
            message: error.to_string(),
        }
    }

    pub(crate) fn load(path: impl Into<String>, error: impl std::fmt::Display) -> Self {
        Self::Load {
            path: path.into(),
            message: error.to_string(),
        }
    }

    pub(crate) fn input(message: impl std::fmt::Display) -> Self {
        Self::Input {
            message: message.to_string(),
        }
    }

    pub(crate) fn inference(error: impl std::fmt::Display) -> Self {
        Self::Inference {
            message: error.to_string(),
        }
    }

    pub(crate) fn output(name: impl Into<String>, error: impl std::fmt::Display) -> Self {
        Self::Output {
            name: name.into(),
            message: error.to_string(),
        }
    }
}
