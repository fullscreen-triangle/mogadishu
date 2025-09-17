//! # Process State Monitoring
//!
//! Real-time monitoring and state tracking for S-entropy processes

use super::{SystemVisualizationState, CellVisualizationState};
use crate::s_entropy::SSpace;
use crate::cellular::Cell;
use crate::miraculous::MiraculousDynamics;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::time::{Duration, Instant};

/// Process state monitor that tracks system evolution over time
pub struct ProcessStateMonitor {
    /// Current system state
    current_state: SystemVisualizationState,
    /// Historical states for trend analysis
    state_history: VecDeque<StateSnapshot>,
    /// Maximum history length
    max_history: usize,
    /// State change detection thresholds
    change_thresholds: ChangeThresholds,
    /// Last update time
    last_update: Instant,
}

/// Snapshot of system state at a specific time
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StateSnapshot {
    /// Timestamp of this snapshot
    pub timestamp: chrono::DateTime<chrono::Utc>,
    /// System state at this time
    pub state: SystemVisualizationState,
    /// Performance metrics at this time
    pub metrics: PerformanceMetrics,
}

/// Performance metrics for system monitoring
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMetrics {
    /// Overall system efficiency (0-1)
    pub system_efficiency: f64,
    /// ATP consumption rate
    pub atp_consumption_rate: f64,
    /// Molecular resolution success rate
    pub resolution_success_rate: f64,
    /// S-space navigation velocity
    pub navigation_velocity: f64,
    /// Information processing throughput
    pub information_throughput: f64,
    /// Miracle utilization factor
    pub miracle_utilization: f64,
}

/// Thresholds for detecting significant state changes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChangeThresholds {
    /// S-space position change threshold
    pub s_position_threshold: f64,
    /// ATP level change threshold (mM)
    pub atp_threshold: f64,
    /// Efficiency change threshold
    pub efficiency_threshold: f64,
    /// Miracle level change threshold
    pub miracle_threshold: f64,
}

/// State change event
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StateChangeEvent {
    /// Timestamp of the change
    pub timestamp: chrono::DateTime<chrono::Utc>,
    /// Type of change detected
    pub change_type: StateChangeType,
    /// Magnitude of the change
    pub magnitude: f64,
    /// Affected component
    pub component: String,
    /// Additional details
    pub details: String,
}

/// Types of state changes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum StateChangeType {
    /// S-space position change
    SPositionChange,
    /// ATP level change
    ATPLevelChange,
    /// Efficiency change
    EfficiencyChange,
    /// Miracle activation/deactivation
    MiracleChange,
    /// Cell state change
    CellStateChange,
    /// Pipeline stage change
    PipelineChange,
}

/// System trend analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrendAnalysis {
    /// Overall system trend direction
    pub system_trend: TrendDirection,
    /// Individual metric trends
    pub metric_trends: HashMap<String, TrendDirection>,
    /// Prediction confidence
    pub prediction_confidence: f64,
    /// Recommended actions
    pub recommendations: Vec<String>,
}

/// Trend direction indicators
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum TrendDirection {
    Improving,
    Stable,
    Declining,
    Oscillating,
    Unknown,
}

impl Default for ChangeThresholds {
    fn default() -> Self {
        Self {
            s_position_threshold: 10.0,
            atp_threshold: 0.5,
            efficiency_threshold: 0.05,
            miracle_threshold: 100.0,
        }
    }
}

impl ProcessStateMonitor {
    /// Create new process state monitor
    pub fn new(max_history: usize) -> Self {
        Self {
            current_state: SystemVisualizationState::new(),
            state_history: VecDeque::with_capacity(max_history),
            max_history,
            change_thresholds: ChangeThresholds::default(),
            last_update: Instant::now(),
        }
    }

    /// Update state from system components
    pub fn update_from_system(
        &mut self,
        cells: &HashMap<String, Cell>,
        miraculous: &MiraculousDynamics,
    ) -> Vec<StateChangeEvent> {
        let previous_state = self.current_state.clone();
        
        // Update current state
        self.current_state.update_from_system(cells, miraculous);
        
        // Calculate performance metrics
        let metrics = self.calculate_performance_metrics(cells, miraculous);
        
        // Create snapshot
        let snapshot = StateSnapshot {
            timestamp: chrono::Utc::now(),
            state: self.current_state.clone(),
            metrics,
        };

        // Add to history
        self.state_history.push_back(snapshot);
        if self.state_history.len() > self.max_history {
            self.state_history.pop_front();
        }

        // Detect changes
        let changes = self.detect_state_changes(&previous_state, &self.current_state);
        
        self.last_update = Instant::now();
        
        changes
    }

    /// Get current system state
    pub fn current_state(&self) -> &SystemVisualizationState {
        &self.current_state
    }

    /// Get state history
    pub fn state_history(&self) -> &VecDeque<StateSnapshot> {
        &self.state_history
    }

    /// Get latest performance metrics
    pub fn latest_metrics(&self) -> Option<&PerformanceMetrics> {
        self.state_history.back().map(|s| &s.metrics)
    }

    /// Analyze system trends
    pub fn analyze_trends(&self) -> TrendAnalysis {
        if self.state_history.len() < 3 {
            return TrendAnalysis {
                system_trend: TrendDirection::Unknown,
                metric_trends: HashMap::new(),
                prediction_confidence: 0.0,
                recommendations: vec!["Insufficient data for trend analysis".to_string()],
            };
        }

        let recent_states: Vec<_> = self.state_history
            .iter()
            .rev()
            .take(5)
            .collect();

        let mut metric_trends = HashMap::new();
        
        // Analyze efficiency trend
        let efficiency_values: Vec<f64> = recent_states
            .iter()
            .map(|s| s.metrics.system_efficiency)
            .collect();
        metric_trends.insert("efficiency".to_string(), self.analyze_value_trend(&efficiency_values));

        // Analyze ATP trend  
        let atp_values: Vec<f64> = recent_states
            .iter()
            .map(|s| s.metrics.atp_consumption_rate)
            .collect();
        metric_trends.insert("atp".to_string(), self.analyze_value_trend(&atp_values));

        // Analyze navigation velocity trend
        let nav_values: Vec<f64> = recent_states
            .iter()
            .map(|s| s.metrics.navigation_velocity)
            .collect();
        metric_trends.insert("navigation".to_string(), self.analyze_value_trend(&nav_values));

        // Determine overall system trend
        let system_trend = self.determine_overall_trend(&metric_trends);
        
        // Generate recommendations
        let recommendations = self.generate_recommendations(&system_trend, &metric_trends);

        TrendAnalysis {
            system_trend,
            metric_trends,
            prediction_confidence: 0.75, // Simplified confidence calculation
            recommendations,
        }
    }

    /// Calculate current performance metrics
    fn calculate_performance_metrics(
        &self,
        cells: &HashMap<String, Cell>,
        miraculous: &MiraculousDynamics,
    ) -> PerformanceMetrics {
        let system_efficiency = if !cells.is_empty() {
            cells.values()
                .map(|c| c.membrane_computer.resolution_accuracy)
                .sum::<f64>() / cells.len() as f64
        } else {
            0.0
        };

        let atp_consumption_rate = if !cells.is_empty() {
            cells.values()
                .map(|c| c.atp_system.consumption_rate)
                .sum::<f64>() / cells.len() as f64
        } else {
            0.0
        };

        let resolution_success_rate = system_efficiency;

        let navigation_velocity = self.current_state.navigation_state.distance_remaining / 100.0;

        let information_throughput = system_efficiency * cells.len() as f64 * 1000.0;

        let miracle_utilization = (miraculous.miracle_levels.knowledge + 
                                  miraculous.miracle_levels.time + 
                                  miraculous.miracle_levels.entropy) / 3.0;

        PerformanceMetrics {
            system_efficiency,
            atp_consumption_rate,
            resolution_success_rate,
            navigation_velocity,
            information_throughput,
            miracle_utilization,
        }
    }

    /// Detect significant state changes
    fn detect_state_changes(
        &self,
        previous: &SystemVisualizationState,
        current: &SystemVisualizationState,
    ) -> Vec<StateChangeEvent> {
        let mut changes = Vec::new();
        let now = chrono::Utc::now();

        // Check S-space position changes
        let s_distance = ((current.s_position.knowledge - previous.s_position.knowledge).powi(2) +
                         (current.s_position.time - previous.s_position.time).powi(2) +
                         (current.s_position.entropy - previous.s_position.entropy).powi(2)).sqrt();
        
        if s_distance > self.change_thresholds.s_position_threshold {
            changes.push(StateChangeEvent {
                timestamp: now,
                change_type: StateChangeType::SPositionChange,
                magnitude: s_distance,
                component: "S-Space Navigation".to_string(),
                details: format!("Position changed by {:.2}", s_distance),
            });
        }

        // Check pipeline stage changes
        if previous.pipeline_state.current_stage != current.pipeline_state.current_stage {
            changes.push(StateChangeEvent {
                timestamp: now,
                change_type: StateChangeType::PipelineChange,
                magnitude: 1.0,
                component: "Processing Pipeline".to_string(),
                details: format!("Stage: {} → {}", 
                    previous.pipeline_state.current_stage,
                    current.pipeline_state.current_stage),
            });
        }

        // Check miracle changes
        let miracle_change = (current.miraculous_state.knowledge_miracle - previous.miraculous_state.knowledge_miracle).abs() +
                           (current.miraculous_state.time_miracle - previous.miraculous_state.time_miracle).abs() +
                           (current.miraculous_state.entropy_miracle - previous.miraculous_state.entropy_miracle).abs();
        
        if miracle_change > self.change_thresholds.miracle_threshold {
            changes.push(StateChangeEvent {
                timestamp: now,
                change_type: StateChangeType::MiracleChange,
                magnitude: miracle_change,
                component: "Miraculous Dynamics".to_string(),
                details: format!("Miracle levels changed by {:.2}", miracle_change),
            });
        }

        changes
    }

    /// Analyze trend direction for a series of values
    fn analyze_value_trend(&self, values: &[f64]) -> TrendDirection {
        if values.len() < 2 {
            return TrendDirection::Unknown;
        }

        let mut increasing = 0;
        let mut decreasing = 0;

        for window in values.windows(2) {
            if window[1] > window[0] {
                increasing += 1;
            } else if window[1] < window[0] {
                decreasing += 1;
            }
        }

        match (increasing, decreasing) {
            (i, d) if i > d * 2 => TrendDirection::Improving,
            (i, d) if d > i * 2 => TrendDirection::Declining,
            (i, d) if (i - d).abs() <= 1 => TrendDirection::Stable,
            _ => TrendDirection::Oscillating,
        }
    }

    /// Determine overall system trend from individual metric trends
    fn determine_overall_trend(&self, metric_trends: &HashMap<String, TrendDirection>) -> TrendDirection {
        let mut improving = 0;
        let mut declining = 0;
        let mut stable = 0;

        for trend in metric_trends.values() {
            match trend {
                TrendDirection::Improving => improving += 1,
                TrendDirection::Declining => declining += 1,
                TrendDirection::Stable => stable += 1,
                _ => {}
            }
        }

        if improving > declining + stable {
            TrendDirection::Improving
        } else if declining > improving + stable {
            TrendDirection::Declining
        } else {
            TrendDirection::Stable
        }
    }

    /// Generate recommendations based on trend analysis
    fn generate_recommendations(
        &self,
        overall_trend: &TrendDirection,
        _metric_trends: &HashMap<String, TrendDirection>,
    ) -> Vec<String> {
        match overall_trend {
            TrendDirection::Improving => vec![
                "System performance is improving".to_string(),
                "Continue current optimization strategy".to_string(),
            ],
            TrendDirection::Declining => vec![
                "System performance is declining".to_string(),
                "Consider enabling miraculous dynamics".to_string(),
                "Check ATP levels in cellular network".to_string(),
            ],
            TrendDirection::Stable => vec![
                "System performance is stable".to_string(),
                "Monitor for optimization opportunities".to_string(),
            ],
            TrendDirection::Oscillating => vec![
                "System showing oscillatory behavior".to_string(),
                "Consider adjusting miracle levels".to_string(),
            ],
            TrendDirection::Unknown => vec![
                "Insufficient data for recommendations".to_string(),
            ],
        }
    }
}
