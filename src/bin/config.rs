//! CLI Configuration Management
//!
//! Handles configuration loading and validation for the Mogadishu CLI

use clap::ArgMatches;
use mogadishu::error::{MogadishuError, Result};
use serde::{Deserialize, Serialize};
use std::path::Path;

/// CLI configuration structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CliConfig {
    /// Configuration file path (if provided)
    pub config_file: Option<String>,
    /// Output format
    pub output_format: OutputFormat,
    /// Verbosity level
    pub verbosity: u8,
    /// Additional settings
    pub settings: CliSettings,
}

/// Output format options
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub enum OutputFormat {
    Pretty,
    Json,
    Csv,
}

impl std::str::FromStr for OutputFormat {
    type Err = MogadishuError;

    fn from_str(s: &str) -> Result<Self> {
        match s.to_lowercase().as_str() {
            "pretty" => Ok(Self::Pretty),
            "json" => Ok(Self::Json),
            "csv" => Ok(Self::Csv),
            _ => Err(MogadishuError::configuration(format!("Invalid output format: {}", s))),
        }
    }
}

impl std::fmt::Display for OutputFormat {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Pretty => write!(f, "pretty"),
            Self::Json => write!(f, "json"),
            Self::Csv => write!(f, "csv"),
        }
    }
}

/// Additional CLI settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CliSettings {
    /// Default number of cellular observers
    pub default_cell_count: usize,
    /// Default simulation time (hours)
    pub default_simulation_time: f64,
    /// S-viability threshold
    pub s_viability_threshold: f64,
    /// Enable colored output
    pub colored_output: bool,
    /// Progress indicator settings
    pub show_progress: bool,
}

impl Default for CliSettings {
    fn default() -> Self {
        Self {
            default_cell_count: 1000,
            default_simulation_time: 24.0,
            s_viability_threshold: 10000.0,
            colored_output: true,
            show_progress: true,
        }
    }
}

impl CliConfig {
    /// Create new CLI configuration from command line arguments
    pub fn new(matches: &ArgMatches) -> Result<Self> {
        let config_file = matches.get_one::<String>("config").cloned();
        
        let output_format = matches.get_one::<String>("format")
            .unwrap_or(&"pretty".to_string())
            .parse()?;
            
        let verbosity = matches.get_count("verbose");
        
        let mut config = Self {
            config_file: config_file.clone(),
            output_format,
            verbosity,
            settings: CliSettings::default(),
        };
        
        // Load additional settings from config file if provided
        if let Some(config_path) = config_file {
            config.load_from_file(&config_path)?;
        }
        
        Ok(config)
    }
    
    /// Load configuration from file
    pub fn load_from_file(&mut self, path: &str) -> Result<()> {
        if !Path::new(path).exists() {
            return Err(MogadishuError::configuration(format!("Config file not found: {}", path)));
        }
        
        let contents = std::fs::read_to_string(path)
            .map_err(|e| MogadishuError::configuration(format!("Failed to read config file: {}", e)))?;
        
        // Try to parse as different formats
        if path.ends_with(".toml") {
            let file_config: FileConfig = toml::from_str(&contents)?;
            self.merge_file_config(file_config);
        } else if path.ends_with(".yaml") || path.ends_with(".yml") {
            let file_config: FileConfig = serde_yaml::from_str(&contents)?;
            self.merge_file_config(file_config);
        } else if path.ends_with(".json") {
            let file_config: FileConfig = serde_json::from_str(&contents)?;
            self.merge_file_config(file_config);
        } else {
            return Err(MogadishuError::configuration("Unsupported config file format. Use .toml, .yaml, or .json"));
        }
        
        Ok(())
    }
    
    /// Merge configuration from file
    fn merge_file_config(&mut self, file_config: FileConfig) {
        if let Some(settings) = file_config.cli {
            if let Some(cell_count) = settings.default_cell_count {
                self.settings.default_cell_count = cell_count;
            }
            if let Some(sim_time) = settings.default_simulation_time {
                self.settings.default_simulation_time = sim_time;
            }
            if let Some(threshold) = settings.s_viability_threshold {
                self.settings.s_viability_threshold = threshold;
            }
            if let Some(colored) = settings.colored_output {
                self.settings.colored_output = colored;
            }
            if let Some(progress) = settings.show_progress {
                self.settings.show_progress = progress;
            }
        }
    }
    
    /// Get effective cell count (from CLI args or config)
    pub fn get_cell_count(&self, matches: &ArgMatches) -> usize {
        matches.get_one::<String>("cells")
            .and_then(|s| s.parse().ok())
            .unwrap_or(self.settings.default_cell_count)
    }
    
    /// Get effective simulation time (from CLI args or config)
    pub fn get_simulation_time(&self, matches: &ArgMatches) -> f64 {
        matches.get_one::<String>("time")
            .and_then(|s| s.parse().ok())
            .unwrap_or(self.settings.default_simulation_time)
    }
    
    /// Check if verbose output is enabled
    pub fn is_verbose(&self) -> bool {
        self.verbosity > 0
    }
    
    /// Check if debug output is enabled
    pub fn is_debug(&self) -> bool {
        self.verbosity > 1
    }
    
    /// Check if trace output is enabled
    pub fn is_trace(&self) -> bool {
        self.verbosity > 2
    }
    
    /// Create default configuration for testing
    pub fn default_for_testing() -> Self {
        Self {
            config_file: None,
            output_format: OutputFormat::Pretty,
            verbosity: 0,
            settings: CliSettings::default(),
        }
    }
}

/// Configuration file structure
#[derive(Debug, Deserialize)]
struct FileConfig {
    cli: Option<FileCliSettings>,
    bioreactor: Option<FileBioreactorSettings>,
    simulation: Option<FileSimulationSettings>,
}

/// CLI settings from file
#[derive(Debug, Deserialize)]
struct FileCliSettings {
    default_cell_count: Option<usize>,
    default_simulation_time: Option<f64>,
    s_viability_threshold: Option<f64>,
    colored_output: Option<bool>,
    show_progress: Option<bool>,
}

/// Bioreactor settings from file
#[derive(Debug, Deserialize)]
struct FileBioreactorSettings {
    default_temperature: Option<f64>,
    default_ph: Option<f64>,
    default_volume: Option<f64>,
}

/// Simulation settings from file
#[derive(Debug, Deserialize)]
struct FileSimulationSettings {
    time_step: Option<f64>,
    convergence_tolerance: Option<f64>,
    max_iterations: Option<usize>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::{Arg, Command};

    #[test]
    fn test_cli_config_creation() {
        let app = Command::new("test")
            .arg(Arg::new("config").long("config"))
            .arg(Arg::new("format").long("format"))
            .arg(Arg::new("verbose").short('v').action(clap::ArgAction::Count));
        
        let matches = app.get_matches_from(vec!["test", "--format", "json", "-v"]);
        let config = CliConfig::new(&matches).unwrap();
        
        assert_eq!(config.output_format, OutputFormat::Json);
        assert_eq!(config.verbosity, 1);
    }
    
    #[test]
    fn test_output_format_parsing() {
        assert_eq!("pretty".parse::<OutputFormat>().unwrap(), OutputFormat::Pretty);
        assert_eq!("json".parse::<OutputFormat>().unwrap(), OutputFormat::Json);
        assert_eq!("csv".parse::<OutputFormat>().unwrap(), OutputFormat::Csv);
        
        assert!("invalid".parse::<OutputFormat>().is_err());
    }
    
    #[test]
    fn test_default_settings() {
        let settings = CliSettings::default();
        assert_eq!(settings.default_cell_count, 1000);
        assert_eq!(settings.default_simulation_time, 24.0);
        assert_eq!(settings.s_viability_threshold, 10000.0);
        assert!(settings.colored_output);
        assert!(settings.show_progress);
    }
}
