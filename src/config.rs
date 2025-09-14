//! # Configuration management for Mogadishu S-Entropy Framework
//!
//! Comprehensive configuration system supporting multiple sources:
//! - Configuration files (TOML, JSON, YAML)
//! - Environment variables
//! - Command-line arguments
//! - Default values

use crate::error::{MogadishuError, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;

/// Main configuration structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MogadishuConfig {
    /// S-entropy framework configuration
    pub s_entropy: SEntropyConfig,
    /// Cellular architecture configuration
    pub cellular: CellularConfig,
    /// Miraculous dynamics configuration
    pub miraculous: MiraculousConfig,
    /// Bioreactor system configuration
    pub bioreactor: BioreactorConfig,
    /// Integration settings
    pub integration: IntegrationConfig,
    /// Logging configuration
    pub logging: LoggingConfig,
    /// Performance settings
    pub performance: PerformanceConfig,
}

/// S-entropy framework configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SEntropyConfig {
    /// Default S-viability threshold
    pub default_viability_threshold: f64,
    /// Enable miracle tracing for debugging
    pub enable_miracle_tracing: bool,
    /// Observer debugging enabled
    pub observer_debug: bool,
    /// Precision-by-difference enhancement factor
    pub precision_enhancement_factor: f64,
    /// Meta-information generation strategy
    pub meta_info_strategy: MetaInfoStrategy,
}

/// Cellular architecture configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CellularConfig {
    /// Default ATP concentration (mM)
    pub default_atp_concentration: f64,
    /// Energy charge warning threshold
    pub energy_charge_warning: f64,
    /// Membrane quantum computer accuracy target
    pub quantum_computer_accuracy: f64,
    /// DNA consultation threshold
    pub dna_consultation_threshold: f64,
    /// Oxygen enhancement settings
    pub oxygen_enhancement: OxygenConfig,
    /// Electron cascade settings
    pub electron_cascade: ElectronCascadeConfig,
}

/// Oxygen enhancement configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OxygenConfig {
    /// Enable oxygen enhancement
    pub enabled: bool,
    /// Information density multiplier
    pub info_density_multiplier: f64,
    /// Processing enhancement factor
    pub processing_enhancement: f64,
    /// Atmospheric coupling strength
    pub atmospheric_coupling: f64,
}

/// Electron cascade configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ElectronCascadeConfig {
    /// Enable electron cascade communication
    pub enabled: bool,
    /// Target communication speed (m/s)
    pub communication_speed: f64,
    /// Signal efficiency threshold
    pub signal_efficiency_threshold: f64,
    /// Cascade pathway timeout (s)
    pub pathway_timeout: f64,
}

/// Miraculous dynamics configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MiraculousConfig {
    /// Enable impossibility elimination
    pub enable_impossibility_cache: bool,
    /// Reverse causality debugging
    pub reverse_causality_debug: bool,
    /// Miracle optimization verbosity
    pub miracle_optimization_verbose: bool,
    /// Maximum miracle levels
    pub max_miracle_levels: MaxMiracleLevels,
    /// Miracle testing timeout (s)
    pub testing_timeout: f64,
}

/// Maximum miracle levels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MaxMiracleLevels {
    /// Maximum knowledge miracle level
    pub knowledge: f64,
    /// Minimum time miracle level
    pub time: f64,
    /// Maximum entropy miracle level (negative)
    pub entropy: f64,
}

/// Bioreactor configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BioreactorConfig {
    /// Default number of cellular observers
    pub default_cellular_observers: usize,
    /// Enable coordination debugging
    pub coordination_debug: bool,
    /// Enable precision difference tracing
    pub precision_difference_tracing: bool,
    /// Network topology settings
    pub network_topology: NetworkTopologyConfig,
    /// Performance optimization settings
    pub optimization: OptimizationConfig,
}

/// Network topology configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkTopologyConfig {
    /// Topology type
    pub topology_type: TopologyType,
    /// Small-world clustering coefficient (if applicable)
    pub clustering_coefficient: f64,
    /// Average path length target
    pub target_path_length: f64,
}

/// Topology types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TopologyType {
    SmallWorld,
    ScaleFree,
    Random,
    Custom,
}

/// Optimization configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationConfig {
    /// Maximum optimization iterations
    pub max_iterations: usize,
    /// Convergence tolerance
    pub convergence_tolerance: f64,
    /// Enable parallel optimization
    pub parallel_optimization: bool,
    /// Optimization timeout (s)
    pub timeout: f64,
}

/// Integration configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntegrationConfig {
    /// Python demo auto-reload
    pub demo_auto_reload: bool,
    /// Interactive plotting
    pub plot_interactive: bool,
    /// Auto-start Jupyter
    pub jupyter_auto_start: bool,
    /// Validation verbosity
    pub validation_verbose: bool,
    /// Hardware integration settings
    pub hardware: HardwareConfig,
}

/// Hardware integration configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HardwareConfig {
    /// Enable hardware oscillation harvesting
    pub enable_oscillation_harvesting: bool,
    /// Enable SIMD optimizations
    pub enable_simd: bool,
    /// Enable GPU acceleration
    pub enable_gpu: bool,
    /// Hardware polling interval (ms)
    pub polling_interval: u64,
}

/// Logging configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoggingConfig {
    /// Log level
    pub level: LogLevel,
    /// Log targets
    pub targets: Vec<String>,
    /// Enable structured logging
    pub structured: bool,
    /// Log file path (optional)
    pub file_path: Option<PathBuf>,
    /// Enable console colors
    pub colored: bool,
}

/// Log levels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LogLevel {
    Trace,
    Debug,
    Info,
    Warn,
    Error,
}

impl std::fmt::Display for LogLevel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Trace => write!(f, "trace"),
            Self::Debug => write!(f, "debug"),
            Self::Info => write!(f, "info"),
            Self::Warn => write!(f, "warn"),
            Self::Error => write!(f, "error"),
        }
    }
}

/// Meta-information generation strategies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MetaInfoStrategy {
    PatternRecognition,
    EvidenceRectification,
    BayesianOptimization,
    Hybrid,
}

/// Performance configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceConfig {
    /// Enable parallel cellular processing
    pub parallel_cellular_processing: bool,
    /// Number of worker threads
    pub worker_threads: Option<usize>,
    /// Optimization level
    pub optimization_level: OptimizationLevel,
    /// Enable performance monitoring
    pub enable_monitoring: bool,
    /// Memory limit (MB)
    pub memory_limit: Option<usize>,
}

/// Optimization levels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OptimizationLevel {
    Debug,
    Release,
    MaxPerformance,
}

impl Default for MogadishuConfig {
    fn default() -> Self {
        Self {
            s_entropy: SEntropyConfig::default(),
            cellular: CellularConfig::default(),
            miraculous: MiraculousConfig::default(),
            bioreactor: BioreactorConfig::default(),
            integration: IntegrationConfig::default(),
            logging: LoggingConfig::default(),
            performance: PerformanceConfig::default(),
        }
    }
}

impl Default for SEntropyConfig {
    fn default() -> Self {
        Self {
            default_viability_threshold: 10000.0,
            enable_miracle_tracing: false,
            observer_debug: false,
            precision_enhancement_factor: 1000.0,
            meta_info_strategy: MetaInfoStrategy::Hybrid,
        }
    }
}

impl Default for CellularConfig {
    fn default() -> Self {
        Self {
            default_atp_concentration: 5.0,
            energy_charge_warning: 0.7,
            quantum_computer_accuracy: 0.99,
            dna_consultation_threshold: 0.95,
            oxygen_enhancement: OxygenConfig::default(),
            electron_cascade: ElectronCascadeConfig::default(),
        }
    }
}

impl Default for OxygenConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            info_density_multiplier: 3.2e15,
            processing_enhancement: 8000.0,
            atmospheric_coupling: 4.7e-3,
        }
    }
}

impl Default for ElectronCascadeConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            communication_speed: 1e6,
            signal_efficiency_threshold: 0.95,
            pathway_timeout: 1e-9,
        }
    }
}

impl Default for MiraculousConfig {
    fn default() -> Self {
        Self {
            enable_impossibility_cache: true,
            reverse_causality_debug: false,
            miracle_optimization_verbose: false,
            max_miracle_levels: MaxMiracleLevels {
                knowledge: 10000.0,
                time: 0.0001,
                entropy: -1000.0,
            },
            testing_timeout: 60.0,
        }
    }
}

impl Default for BioreactorConfig {
    fn default() -> Self {
        Self {
            default_cellular_observers: 1000,
            coordination_debug: false,
            precision_difference_tracing: false,
            network_topology: NetworkTopologyConfig {
                topology_type: TopologyType::SmallWorld,
                clustering_coefficient: 0.3,
                target_path_length: 2.5,
            },
            optimization: OptimizationConfig {
                max_iterations: 1000,
                convergence_tolerance: 1e-6,
                parallel_optimization: true,
                timeout: 3600.0,
            },
        }
    }
}

impl Default for IntegrationConfig {
    fn default() -> Self {
        Self {
            demo_auto_reload: false,
            plot_interactive: true,
            jupyter_auto_start: false,
            validation_verbose: false,
            hardware: HardwareConfig {
                enable_oscillation_harvesting: false,
                enable_simd: false,
                enable_gpu: false,
                polling_interval: 100,
            },
        }
    }
}

impl Default for LoggingConfig {
    fn default() -> Self {
        Self {
            level: LogLevel::Info,
            targets: vec![
                "mogadishu".to_string(),
                "mogadishu::s_entropy".to_string(),
                "mogadishu::cellular".to_string(),
                "mogadishu::bioreactor".to_string(),
            ],
            structured: false,
            file_path: None,
            colored: true,
        }
    }
}

impl Default for PerformanceConfig {
    fn default() -> Self {
        Self {
            parallel_cellular_processing: true,
            worker_threads: None, // Use system default
            optimization_level: OptimizationLevel::Release,
            enable_monitoring: false,
            memory_limit: None,
        }
    }
}

/// Configuration loader
pub struct ConfigLoader {
    /// Configuration sources in priority order
    sources: Vec<ConfigSource>,
    /// Environment variable prefix
    env_prefix: String,
}

/// Configuration sources
#[derive(Debug)]
pub enum ConfigSource {
    /// Default configuration
    Default,
    /// Configuration file
    File(PathBuf),
    /// Environment variables
    Environment,
    /// Command line arguments
    CommandLine(HashMap<String, String>),
}

impl ConfigLoader {
    /// Create new configuration loader
    pub fn new() -> Self {
        Self {
            sources: vec![ConfigSource::Default],
            env_prefix: "MOGADISHU".to_string(),
        }
    }

    /// Add configuration file source
    pub fn with_file<P: Into<PathBuf>>(mut self, path: P) -> Self {
        self.sources.push(ConfigSource::File(path.into()));
        self
    }

    /// Add environment variables source
    pub fn with_environment(mut self) -> Self {
        self.sources.push(ConfigSource::Environment);
        self
    }

    /// Add command line arguments source
    pub fn with_command_line(mut self, args: HashMap<String, String>) -> Self {
        self.sources.push(ConfigSource::CommandLine(args));
        self
    }

    /// Set environment variable prefix
    pub fn with_env_prefix<S: Into<String>>(mut self, prefix: S) -> Self {
        self.env_prefix = prefix.into();
        self
    }

    /// Load configuration from all sources
    pub fn load(&self) -> Result<MogadishuConfig> {
        let mut config = MogadishuConfig::default();

        for source in &self.sources {
            match source {
                ConfigSource::Default => {
                    // Already using default
                }
                ConfigSource::File(path) => {
                    config = self.merge_file_config(config, path)?;
                }
                ConfigSource::Environment => {
                    config = self.merge_env_config(config)?;
                }
                ConfigSource::CommandLine(args) => {
                    config = self.merge_cli_config(config, args)?;
                }
            }
        }

        self.validate_config(&config)?;
        Ok(config)
    }

    fn merge_file_config(&self, mut config: MogadishuConfig, path: &PathBuf) -> Result<MogadishuConfig> {
        if !path.exists() {
            return Ok(config);
        }

        let content = std::fs::read_to_string(path)
            .map_err(|e| MogadishuError::Io(e))?;

        let file_config: MogadishuConfig = match path.extension().and_then(|s| s.to_str()) {
            Some("toml") => toml::from_str(&content)
                .map_err(|e| MogadishuError::configuration(format!("TOML parse error: {}", e)))?,
            Some("json") => serde_json::from_str(&content)
                .map_err(|e| MogadishuError::Serialization(e))?,
            Some("yaml") | Some("yml") => serde_yaml::from_str(&content)
                .map_err(|e| MogadishuError::configuration(format!("YAML parse error: {}", e)))?,
            _ => return Err(MogadishuError::configuration("Unsupported config file format")),
        };

        // Merge configurations (file overrides defaults)
        self.merge_configs(config, file_config)
    }

    fn merge_env_config(&self, config: MogadishuConfig) -> Result<MogadishuConfig> {
        // Read environment variables with prefix
        let env_vars: HashMap<String, String> = std::env::vars()
            .filter(|(key, _)| key.starts_with(&format!("{}_", self.env_prefix)))
            .map(|(key, value)| (key[self.env_prefix.len() + 1..].to_lowercase(), value))
            .collect();

        // Apply environment variable overrides
        let mut updated_config = config;

        // Example environment variable mappings
        if let Some(value) = env_vars.get("s_entropy_viability_threshold") {
            if let Ok(threshold) = value.parse::<f64>() {
                updated_config.s_entropy.default_viability_threshold = threshold;
            }
        }

        if let Some(value) = env_vars.get("cellular_observers") {
            if let Ok(count) = value.parse::<usize>() {
                updated_config.bioreactor.default_cellular_observers = count;
            }
        }

        if let Some(value) = env_vars.get("log_level") {
            updated_config.logging.level = match value.to_lowercase().as_str() {
                "trace" => LogLevel::Trace,
                "debug" => LogLevel::Debug,
                "info" => LogLevel::Info,
                "warn" => LogLevel::Warn,
                "error" => LogLevel::Error,
                _ => LogLevel::Info,
            };
        }

        Ok(updated_config)
    }

    fn merge_cli_config(&self, mut config: MogadishuConfig, args: &HashMap<String, String>) -> Result<MogadishuConfig> {
        // Apply command line argument overrides
        if let Some(value) = args.get("cells") {
            if let Ok(count) = value.parse::<usize>() {
                config.bioreactor.default_cellular_observers = count;
            }
        }

        if let Some(value) = args.get("enable-miracles") {
            if value == "true" {
                config.miraculous.enable_impossibility_cache = true;
                config.miraculous.miracle_optimization_verbose = true;
            }
        }

        if let Some(value) = args.get("oxygen-enhanced") {
            if value == "true" {
                config.cellular.oxygen_enhancement.enabled = true;
            }
        }

        Ok(config)
    }

    fn merge_configs(&self, base: MogadishuConfig, override_config: MogadishuConfig) -> Result<MogadishuConfig> {
        // Simple merge - in a real implementation, would use a more sophisticated merge strategy
        Ok(override_config)
    }

    fn validate_config(&self, config: &MogadishuConfig) -> Result<()> {
        // Validate configuration values
        if config.s_entropy.default_viability_threshold <= 0.0 {
            return Err(MogadishuError::configuration("S-entropy viability threshold must be positive"));
        }

        if config.cellular.default_atp_concentration <= 0.0 {
            return Err(MogadishuError::configuration("ATP concentration must be positive"));
        }

        if config.bioreactor.default_cellular_observers == 0 {
            return Err(MogadishuError::configuration("Must have at least one cellular observer"));
        }

        Ok(())
    }
}

impl Default for ConfigLoader {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::TempDir;

    #[test]
    fn test_default_config() {
        let config = MogadishuConfig::default();
        assert_eq!(config.s_entropy.default_viability_threshold, 10000.0);
        assert_eq!(config.bioreactor.default_cellular_observers, 1000);
    }

    #[test]
    fn test_config_loader() {
        let loader = ConfigLoader::new()
            .with_environment()
            .with_env_prefix("TEST_MOGADISHU");

        let config = loader.load().unwrap();
        assert!(config.s_entropy.default_viability_threshold > 0.0);
    }

    #[test]
    fn test_config_file_loading() {
        let temp_dir = TempDir::new().unwrap();
        let config_path = temp_dir.path().join("test_config.toml");

        let test_config = r#"
[s_entropy]
default_viability_threshold = 5000.0
enable_miracle_tracing = true

[bioreactor]
default_cellular_observers = 500
"#;

        fs::write(&config_path, test_config).unwrap();

        let loader = ConfigLoader::new().with_file(&config_path);
        let config = loader.load().unwrap();

        assert_eq!(config.s_entropy.default_viability_threshold, 5000.0);
        assert_eq!(config.bioreactor.default_cellular_observers, 500);
    }

    #[test]
    fn test_config_validation() {
        let mut config = MogadishuConfig::default();
        config.s_entropy.default_viability_threshold = -100.0; // Invalid

        let loader = ConfigLoader::new();
        let result = loader.validate_config(&config);
        assert!(result.is_err());
    }
}
