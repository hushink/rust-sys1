use std::str::FromStr;
use thiserror::Error;

/// Device requested by the caller.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum DevicePreference {
    /// Pick the best available device: CUDA, then Metal, then CPU.
    #[default]
    Auto,
    Cpu,
    Cuda(usize),
    Metal,
}

/// A device that is actually available in this process.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ResolvedDevice {
    Cpu,
    Cuda(usize),
    Metal,
}

#[derive(Clone, Debug, Error, PartialEq, Eq)]
pub enum DeviceParseError {
    #[error("invalid device `{0}`; expected auto, cpu, cuda, cuda:N, or metal")]
    Invalid(String),
}

#[derive(Clone, Debug, Error, PartialEq, Eq)]
pub enum DeviceError {
    #[error("requested device is not available: {0}")]
    Unavailable(String),
}

impl FromStr for DevicePreference {
    type Err = DeviceParseError;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        let value = value.trim();
        let lower = value.to_ascii_lowercase();
        match lower.as_str() {
            "auto" => Ok(Self::Auto),
            "cpu" => Ok(Self::Cpu),
            "metal" => Ok(Self::Metal),
            "cuda" => Ok(Self::Cuda(0)),
            _ => {
                if let Some(index) = lower.strip_prefix("cuda:") {
                    index
                        .parse::<usize>()
                        .map(Self::Cuda)
                        .map_err(|_| DeviceParseError::Invalid(value.to_owned()))
                } else {
                    Err(DeviceParseError::Invalid(value.to_owned()))
                }
            }
        }
    }
}

impl DevicePreference {
    /// Resolve against devices reported by a backend probe.
    ///
    /// `available` is ordered by the probe's preference. `Auto` chooses the
    /// first CUDA device, then Metal, then CPU.
    pub fn resolve(self, available: &[ResolvedDevice]) -> Result<ResolvedDevice, DeviceError> {
        match self {
            Self::Auto => available
                .iter()
                .find(|device| matches!(device, ResolvedDevice::Cuda(_)))
                .or_else(|| {
                    available
                        .iter()
                        .find(|device| matches!(device, ResolvedDevice::Metal))
                })
                .or_else(|| {
                    available
                        .iter()
                        .find(|device| matches!(device, ResolvedDevice::Cpu))
                })
                .copied()
                .ok_or_else(|| DeviceError::Unavailable("auto (no device available)".to_owned())),
            Self::Cpu => {
                if available.contains(&ResolvedDevice::Cpu) {
                    Ok(ResolvedDevice::Cpu)
                } else {
                    Err(DeviceError::Unavailable("cpu".to_owned()))
                }
            }
            Self::Cuda(index) => {
                let requested = ResolvedDevice::Cuda(index);
                if available.contains(&requested) {
                    Ok(requested)
                } else {
                    Err(DeviceError::Unavailable(format!("cuda:{index}")))
                }
            }
            Self::Metal => {
                if available.contains(&ResolvedDevice::Metal) {
                    Ok(ResolvedDevice::Metal)
                } else {
                    Err(DeviceError::Unavailable("metal".to_owned()))
                }
            }
        }
    }
}
