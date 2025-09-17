//! # Dynamic Process Visualization
//!
//! This module provides real-time Mermaid diagram generation for visualizing
//! the S-entropy framework processes as they execute.

use crate::s_entropy::SSpace;
use crate::cellular::Cell;
use crate::miraculous::MiraculousDynamics;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

pub mod mermaid;
pub mod process_state;
pub mod dashboard;

/// Current system state for visualization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemVisualizationState {
    /// Current S-space coordinates
    pub s_position: SSpace,
    /// Active cellular observers
    pub active_cells: HashMap<String, CellVisualizationState>,
    /// Miraculous dynamics status
    pub miraculous_state: MiraculousDynamicsState,
    /// Current processing pipeline
    pub pipeline_state: PipelineVisualizationState,
    /// System navigation status
    pub navigation_state: NavigationVisualizationState,
    /// Timestamp of this state
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

/// Cell state for visualization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CellVisualizationState {
    /// Cell ID
    pub id: String,
    /// ATP concentration
    pub atp_level: f64,
    /// Current processing method (Membrane/DNA)
    pub processing_method: String,
    /// Molecular challenges being processed
    pub active_challenges: usize,
    /// Current S-space position
    pub s_position: SSpace,
    /// Processing efficiency
    pub efficiency: f64,
}

/// Miraculous dynamics state for visualization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MiraculousDynamicsState {
    /// Current miracle levels
    pub knowledge_miracle: f64,
    pub time_miracle: f64,
    pub entropy_miracle: f64,
    /// Global viability status
    pub globally_viable: bool,
    /// Active miraculous behaviors
    pub active_miracles: Vec<String>,
}

/// Pipeline processing state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipelineVisualizationState {
    /// Current stage in processing pipeline
    pub current_stage: String,
    /// Completed stages
    pub completed_stages: Vec<String>,
    /// Pending stages
    pub pending_stages: Vec<String>,
    /// Current throughput
    pub throughput: f64,
    /// Bottleneck analysis
    pub bottlenecks: Vec<String>,
}

/// Navigation state through S-space
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NavigationVisualizationState {
    /// Current position
    pub current_position: SSpace,
    /// Target endpoint
    pub target_position: SSpace,
    /// Navigation path
    pub navigation_path: Vec<SSpace>,
    /// Distance to target
    pub distance_remaining: f64,
    /// Navigation strategy
    pub strategy: String,
}

impl SystemVisualizationState {
    /// Create new visualization state
    pub fn new() -> Self {
        Self {
            s_position: SSpace::new(0.0, 0.0, 0.0),
            active_cells: HashMap::new(),
            miraculous_state: MiraculousDynamicsState {
                knowledge_miracle: 1.0,
                time_miracle: 1.0,
                entropy_miracle: 1.0,
                globally_viable: true,
                active_miracles: Vec::new(),
            },
            pipeline_state: PipelineVisualizationState {
                current_stage: "Initialization".to_string(),
                completed_stages: Vec::new(),
                pending_stages: vec![
                    "Observer Insertion".to_string(),
                    "Cellular Processing".to_string(),
                    "S-Navigation".to_string(),
                    "Optimization".to_string(),
                ],
                throughput: 0.0,
                bottlenecks: Vec::new(),
            },
            navigation_state: NavigationVisualizationState {
                current_position: SSpace::new(0.0, 0.0, 0.0),
                target_position: SSpace::new(1000.0, 0.1, -100.0),
                navigation_path: Vec::new(),
                distance_remaining: 0.0,
                strategy: "Direct Navigation".to_string(),
            },
            timestamp: chrono::Utc::now(),
        }
    }

    /// Update state from system components
    pub fn update_from_system(&mut self, cells: &HashMap<String, Cell>, miraculous: &MiraculousDynamics) {
        self.timestamp = chrono::Utc::now();
        
        // Update cellular states
        for (id, cell) in cells {
            self.active_cells.insert(id.clone(), CellVisualizationState {
                id: id.clone(),
                atp_level: cell.atp_system.atp_concentration,
                processing_method: if cell.membrane_computer.resolution_accuracy > 0.95 {
                    "Membrane Quantum".to_string()
                } else {
                    "DNA Consultation".to_string()
                },
                active_challenges: cell.molecular_environment.active_challenges.len(),
                s_position: cell.s_position,
                efficiency: cell.membrane_computer.resolution_accuracy,
            });
        }

        // Update miraculous dynamics
        self.miraculous_state = MiraculousDynamicsState {
            knowledge_miracle: miraculous.miracle_levels.knowledge,
            time_miracle: miraculous.miracle_levels.time,
            entropy_miracle: miraculous.miracle_levels.entropy,
            globally_viable: miraculous.miracle_levels.knowledge.powi(2) + 
                           miraculous.miracle_levels.time.powi(2) + 
                           miraculous.miracle_levels.entropy.powi(2) <= miraculous.viability_threshold.powi(2),
            active_miracles: self.identify_active_miracles(miraculous),
        };

        // Update overall S-position as average of cell positions
        if !self.active_cells.is_empty() {
            let avg_knowledge: f64 = self.active_cells.values().map(|c| c.s_position.knowledge).sum::<f64>() / self.active_cells.len() as f64;
            let avg_time: f64 = self.active_cells.values().map(|c| c.s_position.time).sum::<f64>() / self.active_cells.len() as f64;
            let avg_entropy: f64 = self.active_cells.values().map(|c| c.s_position.entropy).sum::<f64>() / self.active_cells.len() as f64;
            
            self.s_position = SSpace::new(avg_knowledge, avg_time, avg_entropy);
            self.navigation_state.current_position = self.s_position;
        }
    }

    fn identify_active_miracles(&self, miraculous: &MiraculousDynamics) -> Vec<String> {
        let mut miracles = Vec::new();
        
        if miraculous.knowledge_dynamics.infinite_processing {
            miracles.push("Infinite Knowledge Processing".to_string());
        }
        if miraculous.time_dynamics.instantaneous_solutions {
            miracles.push("Instantaneous Solutions".to_string());
        }
        if miraculous.entropy_dynamics.negative_entropy {
            miracles.push("Negative Entropy Generation".to_string());
        }
        
        miracles
    }
}
