//! # S-Entropy Core Mathematical Framework
//!
//! This module implements the foundational S-entropy mathematics including:
//! - Tri-dimensional S-space coordinates (knowledge, time, entropy) 
//! - S-distance metrics for navigational discovery
//! - Observer insertion mechanisms for finite problem spaces
//! - Precision-by-difference coordination protocols

use nalgebra::{Vector3, Matrix3};
use serde::{Deserialize, Serialize};
use std::f64::consts::PI;

pub mod coordinates;
pub mod observers;
pub mod navigation; 
pub mod precision_difference;

/// Tri-dimensional S-space vector
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct SSpace {
    /// Knowledge dimension - information processing capacity
    pub knowledge: f64,
    /// Time dimension - temporal solution pathways  
    pub time: f64,
    /// Entropy dimension - thermodynamic endpoint distance
    pub entropy: f64,
}

impl SSpace {
    /// Create new S-space coordinates
    pub fn new(knowledge: f64, time: f64, entropy: f64) -> Self {
        Self { knowledge, time, entropy }
    }

    /// Create S-space from Vector3
    pub fn from_vector(v: Vector3<f64>) -> Self {
        Self::new(v.x, v.y, v.z)
    }

    /// Convert to Vector3 for mathematical operations
    pub fn to_vector(&self) -> Vector3<f64> {
        Vector3::new(self.knowledge, self.time, self.entropy)
    }

    /// Calculate S-distance magnitude
    pub fn magnitude(&self) -> f64 {
        (self.knowledge.powi(2) + self.time.powi(2) + self.entropy.powi(2)).sqrt()
    }

    /// Check if coordinates represent viable state
    pub fn is_viable(&self, threshold: f64) -> bool {
        self.magnitude() <= threshold
    }

    /// Enable miraculous behavior in individual dimensions while maintaining viability
    pub fn allow_miracles(&self, viability_threshold: f64) -> MiraculousCapabilities {
        let total_magnitude = self.magnitude();
        
        if total_magnitude <= viability_threshold {
            MiraculousCapabilities {
                infinite_knowledge: self.knowledge > 1000.0,
                instantaneous_time: self.time < 0.001,
                negative_entropy: self.entropy < 0.0,
                globally_viable: true,
            }
        } else {
            MiraculousCapabilities::none()
        }
    }
}

/// S-distance metric for navigation
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct SDistance(pub f64);

impl SDistance {
    /// Calculate S-distance between two S-space points
    pub fn between(a: SSpace, b: SSpace) -> Self {
        let diff = a.to_vector() - b.to_vector();
        Self(diff.magnitude())
    }

    /// Check if distance represents reachable solution
    pub fn is_reachable(&self, max_distance: f64) -> bool {
        self.0 <= max_distance
    }
}

/// Miraculous capabilities within viability constraints
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct MiraculousCapabilities {
    /// Can process infinite information instantaneously
    pub infinite_knowledge: bool,
    /// Can achieve zero-time solutions
    pub instantaneous_time: bool, 
    /// Can generate negative entropy locally
    pub negative_entropy: bool,
    /// Global S-viability maintained
    pub globally_viable: bool,
}

impl MiraculousCapabilities {
    /// No miracles allowed
    pub fn none() -> Self {
        Self {
            infinite_knowledge: false,
            instantaneous_time: false,
            negative_entropy: false,
            globally_viable: false,
        }
    }

    /// Maximum possible miracles
    pub fn maximum() -> Self {
        Self {
            infinite_knowledge: true,
            instantaneous_time: true,
            negative_entropy: true,
            globally_viable: true,
        }
    }
}

/// Abstract observer for finite problem space creation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Observer {
    /// Observer identification
    pub id: String,
    /// Observation window bounds
    pub bounds: ObservationBounds,
    /// Meta-information generation strategy
    pub strategy: ObservationStrategy,
    /// Current S-space position
    pub position: SSpace,
}

/// Observation window constraints
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ObservationBounds {
    /// Knowledge processing limits
    pub max_knowledge: f64,
    /// Time window duration
    pub time_window: f64,
    /// Entropy monitoring range
    pub entropy_range: (f64, f64),
}

/// Strategy for meta-information generation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ObservationStrategy {
    /// Pattern recognition focused
    PatternRecognition { threshold: f64 },
    /// Evidence rectification focused
    EvidenceRectification { confidence: f64 },
    /// Bayesian optimization focused
    BayesianOptimization { prior_strength: f64 },
}

impl Observer {
    /// Create new observer with specified strategy
    pub fn new(id: String, bounds: ObservationBounds, strategy: ObservationStrategy) -> Self {
        Self {
            id,
            bounds,
            strategy,
            position: SSpace::new(0.0, 0.0, 0.0),
        }
    }

    /// Generate meta-information from observation
    pub fn observe(&mut self, target: SSpace) -> MetaInformation {
        // Update observer position
        self.position = target;

        match &self.strategy {
            ObservationStrategy::PatternRecognition { threshold } => {
                self.generate_pattern_info(target, *threshold)
            },
            ObservationStrategy::EvidenceRectification { confidence } => {
                self.generate_evidence_info(target, *confidence)
            },
            ObservationStrategy::BayesianOptimization { prior_strength } => {
                self.generate_bayesian_info(target, *prior_strength)
            },
        }
    }

    fn generate_pattern_info(&self, target: SSpace, threshold: f64) -> MetaInformation {
        let pattern_strength = (target.knowledge / threshold).min(1.0);
        MetaInformation {
            information_type: InformationType::Pattern,
            confidence: pattern_strength,
            s_enhancement: SSpace::new(pattern_strength * 100.0, 0.1, 0.0),
            meta_data: format!("Pattern strength: {:.3}", pattern_strength),
        }
    }

    fn generate_evidence_info(&self, target: SSpace, confidence: f64) -> MetaInformation {
        let evidence_quality = target.magnitude() / (confidence + 1.0);
        MetaInformation {
            information_type: InformationType::Evidence,
            confidence: evidence_quality,
            s_enhancement: SSpace::new(0.0, 0.01, -evidence_quality * 10.0),
            meta_data: format!("Evidence quality: {:.3}", evidence_quality),
        }
    }

    fn generate_bayesian_info(&self, target: SSpace, prior_strength: f64) -> MetaInformation {
        let posterior_strength = prior_strength * target.knowledge / (1.0 + target.time);
        MetaInformation {
            information_type: InformationType::Bayesian,
            confidence: posterior_strength,
            s_enhancement: SSpace::new(posterior_strength, 0.001, -5.0),
            meta_data: format!("Posterior strength: {:.3}", posterior_strength),
        }
    }
}

/// Meta-information generated by observers
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetaInformation {
    /// Type of information generated
    pub information_type: InformationType,
    /// Confidence in the information
    pub confidence: f64,
    /// S-space enhancement provided
    pub s_enhancement: SSpace,
    /// Additional metadata
    pub meta_data: String,
}

/// Types of meta-information
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum InformationType {
    /// Pattern recognition information
    Pattern,
    /// Evidence rectification information
    Evidence,
    /// Bayesian optimization information
    Bayesian,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_s_space_magnitude() {
        let s = SSpace::new(3.0, 4.0, 0.0);
        assert_eq!(s.magnitude(), 5.0);
    }

    #[test]
    fn test_miraculous_capabilities() {
        let s = SSpace::new(2000.0, 0.0001, -50.0);
        let capabilities = s.allow_miracles(3000.0);
        
        assert!(capabilities.infinite_knowledge);
        assert!(capabilities.instantaneous_time);
        assert!(capabilities.negative_entropy);
        assert!(capabilities.globally_viable);
    }

    #[test]
    fn test_observer_pattern_recognition() {
        let bounds = ObservationBounds {
            max_knowledge: 1000.0,
            time_window: 1.0,
            entropy_range: (-100.0, 100.0),
        };
        let strategy = ObservationStrategy::PatternRecognition { threshold: 50.0 };
        let mut observer = Observer::new("test".to_string(), bounds, strategy);
        
        let target = SSpace::new(100.0, 0.5, 10.0);
        let meta_info = observer.observe(target);
        
        assert_eq!(meta_info.information_type, InformationType::Pattern);
        assert!(meta_info.confidence > 0.0);
    }
}
