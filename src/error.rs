//! # Error handling for Mogadishu S-Entropy Framework
//!
//! Comprehensive error types and handling for all framework components
//! including S-entropy operations, cellular processing, and bioreactor modeling.

use serde::{Deserialize, Serialize};
use std::fmt;
use thiserror::Error;

/// Main result type for the framework
pub type Result<T> = std::result::Result<T, MogadishuError>;

/// Comprehensive error types for the S-entropy framework
#[derive(Debug, Error, Serialize, Deserialize)]
pub enum MogadishuError {
    /// S-entropy mathematical errors
    #[error("S-entropy error: {0}")]
    SEntropy(#[from] SEntropyError),

    /// Cellular processing errors
    #[error("Cellular processing error: {0}")]
    Cellular(#[from] CellularError),

    /// Miraculous dynamics errors
    #[error("Miraculous dynamics error: {0}")]
    Miraculous(#[from] MiraculousError),

    /// Bioreactor modeling errors
    #[error("Bioreactor error: {0}")]
    Bioreactor(#[from] BioreactorError),

    /// Integration and coordination errors
    #[error("Integration error: {0}")]
    Integration(#[from] IntegrationError),

    /// Configuration errors
    #[error("Configuration error: {0}")]
    Configuration(String),

    /// I/O and file system errors
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    /// Serialization/deserialization errors
    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),

    /// Mathematical computation errors
    #[error("Mathematical error: {0}")]
    Math(String),

    /// Generic framework error
    #[error("Framework error: {0}")]
    Generic(String),
}

/// S-entropy specific errors
#[derive(Debug, Error, Serialize, Deserialize)]
pub enum SEntropyError {
    #[error("Invalid S-space coordinates: knowledge={knowledge}, time={time}, entropy={entropy}")]
    InvalidSSpace { knowledge: f64, time: f64, entropy: f64 },

    #[error("Observer insertion failed: {reason}")]
    ObserverInsertionFailed { reason: String },

    #[error("Navigation failed to find solution within S-distance threshold {threshold}")]
    NavigationFailed { threshold: f64 },

    #[error("Precision-by-difference coordination failed: {reason}")]
    PrecisionCoordinationFailed { reason: String },

    #[error("S-viability violation: calculated S-total {s_total} exceeds threshold {threshold}")]
    ViabilityViolation { s_total: f64, threshold: f64 },

    #[error("Meta-information generation failed for observer {observer_id}")]
    MetaInformationFailed { observer_id: String },
}

/// Cellular processing errors
#[derive(Debug, Error, Serialize, Deserialize)]
pub enum CellularError {
    #[error("ATP system failure: insufficient ATP {current} for operation requiring {required}")]
    InsufficientAtp { current: f64, required: f64 },

    #[error("Membrane quantum computer resolution failed: confidence {confidence} below threshold {threshold}")]
    QuantumResolutionFailed { confidence: f64, threshold: f64 },

    #[error("DNA library consultation failed: {reason}")]
    DnaConsultationFailed { reason: String },

    #[error("Oxygen enhancement system error: {reason}")]
    OxygenSystemError { reason: String },

    #[error("Molecular identification failed for {molecule_type}")]
    MolecularIdentificationFailed { molecule_type: String },

    #[error("Cellular energy charge critical: {energy_charge} below minimum {minimum}")]
    CriticalEnergyCharge { energy_charge: f64, minimum: f64 },

    #[error("Electron cascade communication disrupted: signal loss {signal_loss}%")]
    ElectronCascadeDisrupted { signal_loss: f64 },
}

/// Miraculous dynamics errors
#[derive(Debug, Error, Serialize, Deserialize)]
pub enum MiraculousError {
    #[error("Miracle configuration violates global viability")]
    ViabilityViolation,

    #[error("Impossible problem: no solution exists even with maximum miracles")]
    AbsolutelyImpossible,

    #[error("Tri-dimensional differential equation integration failed")]
    IntegrationFailure,

    #[error("Reverse causality analysis failed: {reason}")]
    ReverseCausalityFailed { reason: String },

    #[error("Miracle level incompatible: {reason}")]
    IncompatibleMiracleLevels { reason: String },

    #[error("Impossibility elimination failed: {reason}")]
    ImpossibilityEliminationFailed { reason: String },
}

/// Bioreactor system errors
#[derive(Debug, Error, Serialize, Deserialize)]
pub enum BioreactorError {
    #[error("Cellular network creation failed: {reason}")]
    NetworkCreationFailed { reason: String },

    #[error("Optimization failed: target {target} unreachable")]
    OptimizationFailed { target: String },

    #[error("Performance metrics calculation failed")]
    MetricsCalculationFailed,

    #[error("Operating conditions invalid: {parameter} = {value} outside acceptable range")]
    InvalidOperatingConditions { parameter: String, value: f64 },

    #[error("System coordination failed: {reason}")]
    CoordinationFailed { reason: String },

    #[error("Bioreactor state inconsistent: {reason}")]
    InconsistentState { reason: String },
}

/// Integration and coordination errors
#[derive(Debug, Error, Serialize, Deserialize)]
pub enum IntegrationError {
    #[error("Component initialization failed: {component}")]
    InitializationFailed { component: String },

    #[error("Inter-system communication failed between {source} and {target}")]
    CommunicationFailed { source: String, target: String },

    #[error("State synchronization failed: {reason}")]
    StateSynchronizationFailed { reason: String },

    #[error("Hardware integration failed: {device}")]
    HardwareIntegrationFailed { device: String },

    #[error("Python bindings error: {reason}")]
    PythonBindingsError { reason: String },
}

impl MogadishuError {
    /// Create a configuration error
    pub fn configuration<T: fmt::Display>(msg: T) -> Self {
        Self::Configuration(msg.to_string())
    }

    /// Create a mathematical error
    pub fn math<T: fmt::Display>(msg: T) -> Self {
        Self::Math(msg.to_string())
    }

    /// Create a generic framework error
    pub fn generic<T: fmt::Display>(msg: T) -> Self {
        Self::Generic(msg.to_string())
    }

    /// Check if error is recoverable
    pub fn is_recoverable(&self) -> bool {
        match self {
            Self::SEntropy(SEntropyError::NavigationFailed { .. }) => true,
            Self::Cellular(CellularError::InsufficientAtp { .. }) => true,
            Self::Miraculous(MiraculousError::ViabilityViolation) => true,
            Self::Bioreactor(BioreactorError::OptimizationFailed { .. }) => true,
            Self::Integration(IntegrationError::CommunicationFailed { .. }) => true,
            Self::Io(_) => true,
            _ => false,
        }
    }

    /// Get error severity level
    pub fn severity(&self) -> ErrorSeverity {
        match self {
            Self::SEntropy(SEntropyError::ViabilityViolation { .. }) => ErrorSeverity::Critical,
            Self::Cellular(CellularError::CriticalEnergyCharge { .. }) => ErrorSeverity::Critical,
            Self::Miraculous(MiraculousError::AbsolutelyImpossible) => ErrorSeverity::High,
            Self::Bioreactor(BioreactorError::InconsistentState { .. }) => ErrorSeverity::High,
            Self::Integration(IntegrationError::InitializationFailed { .. }) => ErrorSeverity::Medium,
            Self::Configuration(_) => ErrorSeverity::Medium,
            Self::Math(_) => ErrorSeverity::Low,
            Self::Io(_) => ErrorSeverity::Low,
            _ => ErrorSeverity::Medium,
        }
    }

    /// Get error category
    pub fn category(&self) -> ErrorCategory {
        match self {
            Self::SEntropy(_) => ErrorCategory::Mathematics,
            Self::Cellular(_) => ErrorCategory::Biology,
            Self::Miraculous(_) => ErrorCategory::Physics,
            Self::Bioreactor(_) => ErrorCategory::Engineering,
            Self::Integration(_) => ErrorCategory::System,
            Self::Configuration(_) => ErrorCategory::Configuration,
            Self::Io(_) => ErrorCategory::System,
            Self::Serialization(_) => ErrorCategory::Data,
            Self::Math(_) => ErrorCategory::Mathematics,
            Self::Generic(_) => ErrorCategory::General,
        }
    }
}

/// Error severity levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum ErrorSeverity {
    /// Low severity - informational
    Low,
    /// Medium severity - warning
    Medium,
    /// High severity - error
    High,
    /// Critical severity - system failure
    Critical,
}

impl fmt::Display for ErrorSeverity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Low => write!(f, "LOW"),
            Self::Medium => write!(f, "MEDIUM"),
            Self::High => write!(f, "HIGH"),
            Self::Critical => write!(f, "CRITICAL"),
        }
    }
}

/// Error categories
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ErrorCategory {
    /// Mathematical and S-entropy errors
    Mathematics,
    /// Biological and cellular errors
    Biology,
    /// Physical and miraculous dynamics errors
    Physics,
    /// Engineering and bioreactor errors
    Engineering,
    /// System integration errors
    System,
    /// Configuration errors
    Configuration,
    /// Data handling errors
    Data,
    /// General framework errors
    General,
}

impl fmt::Display for ErrorCategory {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Mathematics => write!(f, "MATH"),
            Self::Biology => write!(f, "BIO"),
            Self::Physics => write!(f, "PHYS"),
            Self::Engineering => write!(f, "ENG"),
            Self::System => write!(f, "SYS"),
            Self::Configuration => write!(f, "CFG"),
            Self::Data => write!(f, "DATA"),
            Self::General => write!(f, "GEN"),
        }
    }
}

/// Error context for debugging
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorContext {
    /// Error location (function/module)
    pub location: String,
    /// System state when error occurred
    pub system_state: Option<String>,
    /// Attempted operation
    pub operation: String,
    /// Additional debug information
    pub debug_info: std::collections::HashMap<String, String>,
}

impl ErrorContext {
    /// Create new error context
    pub fn new<T: Into<String>>(location: T, operation: T) -> Self {
        Self {
            location: location.into(),
            system_state: None,
            operation: operation.into(),
            debug_info: std::collections::HashMap::new(),
        }
    }

    /// Add debug information
    pub fn with_debug<K, V>(mut self, key: K, value: V) -> Self
    where
        K: Into<String>,
        V: fmt::Display,
    {
        self.debug_info.insert(key.into(), value.to_string());
        self
    }

    /// Set system state
    pub fn with_state<T: Into<String>>(mut self, state: T) -> Self {
        self.system_state = Some(state.into());
        self
    }
}

/// Result with error context
pub type ContextResult<T> = std::result::Result<T, (MogadishuError, ErrorContext)>;

/// Macro for creating errors with context
#[macro_export]
macro_rules! error_with_context {
    ($error:expr, $location:expr, $operation:expr) => {
        ($error, $crate::error::ErrorContext::new($location, $operation))
    };
    ($error:expr, $location:expr, $operation:expr, $($key:expr => $value:expr),*) => {
        {
            let mut context = $crate::error::ErrorContext::new($location, $operation);
            $(
                context = context.with_debug($key, $value);
            )*
            ($error, context)
        }
    };
}

/// Macro for early return with error context
#[macro_export]
macro_rules! bail_with_context {
    ($error:expr, $location:expr, $operation:expr) => {
        return Err(error_with_context!($error, $location, $operation));
    };
    ($error:expr, $location:expr, $operation:expr, $($key:expr => $value:expr),*) => {
        return Err(error_with_context!($error, $location, $operation, $($key => $value),*));
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_severity() {
        let error = MogadishuError::SEntropy(SEntropyError::ViabilityViolation {
            s_total: 15000.0,
            threshold: 10000.0,
        });
        
        assert_eq!(error.severity(), ErrorSeverity::Critical);
        assert_eq!(error.category(), ErrorCategory::Mathematics);
        assert!(!error.is_recoverable());
    }

    #[test]
    fn test_error_context() {
        let context = ErrorContext::new("test_function", "s_entropy_navigation")
            .with_debug("s_total", "15000.0")
            .with_debug("threshold", "10000.0")
            .with_state("navigating");

        assert_eq!(context.location, "test_function");
        assert_eq!(context.operation, "s_entropy_navigation");
        assert!(context.debug_info.contains_key("s_total"));
        assert_eq!(context.system_state, Some("navigating".to_string()));
    }

    #[test]
    fn test_error_serialization() {
        let error = MogadishuError::Cellular(CellularError::InsufficientAtp {
            current: 2.5,
            required: 5.0,
        });

        let serialized = serde_json::to_string(&error).unwrap();
        let deserialized: MogadishuError = serde_json::from_str(&serialized).unwrap();

        match deserialized {
            MogadishuError::Cellular(CellularError::InsufficientAtp { current, required }) => {
                assert_eq!(current, 2.5);
                assert_eq!(required, 5.0);
            }
            _ => panic!("Deserialization failed"),
        }
    }
}
