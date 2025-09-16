//! # Stability Analysis Engine
//!
//! Comprehensive stability analysis for bioprocess systems using multiple
//! stability criteria enhanced with S-entropy viability analysis and
//! miraculous stability conditions that can maintain stability under
//! impossible operating conditions.

use super::*;
use crate::error::{MogadishuError, Result};
use nalgebra::{DMatrix, DVector, Complex};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

impl StabilityAnalyzer {
    /// Create new stability analyzer
    pub fn new() -> Self {
        Self {
            stability_criteria: vec![
                StabilityCriterion::RouthHurwitz,
                StabilityCriterion::Nyquist,
                StabilityCriterion::BodeMargins,
                StabilityCriterion::SEntropyViability { 
                    viability_threshold: 10000.0,
                    miracle_tolerance: 0.1,
                },
            ],
            stability_status: StabilityStatus::Stable,
            stability_margins: StabilityMargins::default(),
            robustness_analysis: RobustnessAnalysis::default(),
        }
    }

    /// Analyze system stability from Laplace system
    pub fn analyze_stability(&mut self, laplace_system: &LaplaceSystem) -> Result<()> {
        // Extract system characteristics
        let system_poles = self.extract_system_poles(laplace_system)?;
        let system_zeros = self.extract_system_zeros(laplace_system)?;
        let system_matrix = &laplace_system.system_matrix;

        // Apply all stability criteria
        let mut stability_results = Vec::new();
        
        for criterion in &self.stability_criteria {
            let result = self.apply_stability_criterion(criterion, &system_poles, &system_zeros, system_matrix)?;
            stability_results.push(result);
        }

        // Determine overall stability status
        self.stability_status = self.determine_overall_stability(&stability_results)?;

        // Calculate stability margins
        self.stability_margins = self.calculate_stability_margins(&system_poles, &system_zeros)?;

        // Perform robustness analysis
        self.robustness_analysis = self.perform_robustness_analysis(system_matrix, &system_poles)?;

        Ok(())
    }

    /// Extract system poles from Laplace system
    fn extract_system_poles(&self, laplace_system: &LaplaceSystem) -> Result<Vec<Complex<f64>>> {
        let system_matrix = &laplace_system.system_matrix;
        
        if system_matrix.is_empty() {
            return Ok(Vec::new());
        }

        // Calculate eigenvalues of system matrix (poles)
        let eigenvalues = system_matrix.complex_eigenvalues();
        Ok(eigenvalues.iter().cloned().collect())
    }

    /// Extract system zeros from Laplace system
    fn extract_system_zeros(&self, laplace_system: &LaplaceSystem) -> Result<Vec<Complex<f64>>> {
        // For MIMO systems, zeros are more complex to calculate
        // Simplified approach: use transfer function zeros from compressed representation
        
        let compressed = &laplace_system.compressed_representation;
        if compressed.reduced_order == 0 {
            return Ok(Vec::new());
        }

        // Estimate zeros from compressed system
        // In full implementation, would calculate transmission zeros
        let mut estimated_zeros = Vec::new();
        
        // Add some estimated zeros based on system characteristics
        for i in 0..compressed.reduced_order.min(3) {
            let zero = Complex::new(-1.0 - i as f64, 0.0);
            estimated_zeros.push(zero);
        }

        Ok(estimated_zeros)
    }

    /// Apply specific stability criterion
    fn apply_stability_criterion(
        &self, 
        criterion: &StabilityCriterion, 
        poles: &[Complex<f64>], 
        zeros: &[Complex<f64>],
        system_matrix: &DMatrix<Complex<f64>>
    ) -> Result<StabilityResult> {
        match criterion {
            StabilityCriterion::RouthHurwitz => self.apply_routh_hurwitz_criterion(poles),
            StabilityCriterion::Nyquist => self.apply_nyquist_criterion(poles, zeros),
            StabilityCriterion::BodeMargins => self.apply_bode_margins_criterion(poles, zeros),
            StabilityCriterion::Lyapunov => self.apply_lyapunov_criterion(system_matrix),
            StabilityCriterion::SEntropyViability { viability_threshold, miracle_tolerance } => 
                self.apply_s_entropy_viability_criterion(poles, *viability_threshold, *miracle_tolerance),
        }
    }

    /// Apply Routh-Hurwitz stability criterion
    fn apply_routh_hurwitz_criterion(&self, poles: &[Complex<f64>]) -> Result<StabilityResult> {
        if poles.is_empty() {
            return Ok(StabilityResult {
                criterion: "Routh-Hurwitz".to_string(),
                stable: true,
                margin: f64::INFINITY,
                additional_info: "No poles".to_string(),
            });
        }

        // Check if all poles have negative real parts
        let mut stable = true;
        let mut min_margin = f64::INFINITY;

        for pole in poles {
            if pole.re >= 0.0 {
                stable = false;
            }
            let margin = -pole.re;
            if margin < min_margin {
                min_margin = margin;
            }
        }

        Ok(StabilityResult {
            criterion: "Routh-Hurwitz".to_string(),
            stable,
            margin: min_margin,
            additional_info: format!("Rightmost pole at s = {:.3}", 
                poles.iter().max_by(|a, b| a.re.partial_cmp(&b.re).unwrap()).unwrap()),
        })
    }

    /// Apply Nyquist stability criterion
    fn apply_nyquist_criterion(&self, poles: &[Complex<f64>], zeros: &[Complex<f64>]) -> Result<StabilityResult> {
        // Simplified Nyquist analysis
        // Check encirclements of -1 point (would require full frequency response in practice)
        
        let open_loop_poles_rhp = poles.iter()
            .filter(|p| p.re > 0.0)
            .count();

        // Simplified assessment based on poles and zeros
        let stable = open_loop_poles_rhp == 0; // Simplified condition
        
        // Estimate gain margin (simplified)
        let gain_margin = if !poles.is_empty() {
            let critical_frequency = poles.iter()
                .map(|p| p.im.abs())
                .fold(0.0, f64::max);
            
            if critical_frequency > 0.0 {
                20.0 * (1.0 / critical_frequency).log10() // dB
            } else {
                f64::INFINITY
            }
        } else {
            f64::INFINITY
        };

        Ok(StabilityResult {
            criterion: "Nyquist".to_string(),
            stable,
            margin: gain_margin,
            additional_info: format!("RHP poles: {}, Estimated gain margin: {:.1} dB", 
                open_loop_poles_rhp, gain_margin),
        })
    }

    /// Apply Bode margins stability criterion
    fn apply_bode_margins_criterion(&self, poles: &[Complex<f64>], zeros: &[Complex<f64>]) -> Result<StabilityResult> {
        // Calculate gain and phase margins from pole/zero locations
        
        let mut gain_margin = f64::INFINITY;
        let mut phase_margin = 90.0; // degrees
        
        if !poles.is_empty() {
            // Find gain crossover frequency (simplified)
            let dominant_pole = poles.iter()
                .min_by(|a, b| a.re.abs().partial_cmp(&b.re.abs()).unwrap())
                .unwrap();
            
            let crossover_freq = dominant_pole.im.abs().max(dominant_pole.re.abs());
            
            if crossover_freq > 0.0 {
                // Estimate margins (simplified calculation)
                gain_margin = 20.0 * (1.0 / crossover_freq).log10();
                phase_margin = 180.0 + dominant_pole.arg() * 180.0 / std::f64::consts::PI;
            }
        }

        let stable = gain_margin > 0.0 && phase_margin > 0.0;
        let margin = gain_margin.min(phase_margin / 10.0); // Combined margin metric

        Ok(StabilityResult {
            criterion: "Bode Margins".to_string(),
            stable,
            margin,
            additional_info: format!("GM: {:.1} dB, PM: {:.1}°", gain_margin, phase_margin),
        })
    }

    /// Apply Lyapunov stability criterion
    fn apply_lyapunov_criterion(&self, system_matrix: &DMatrix<Complex<f64>>) -> Result<StabilityResult> {
        if system_matrix.is_empty() {
            return Ok(StabilityResult {
                criterion: "Lyapunov".to_string(),
                stable: true,
                margin: f64::INFINITY,
                additional_info: "Empty system matrix".to_string(),
            });
        }

        // For linear systems, Lyapunov stability is equivalent to all eigenvalues having negative real parts
        let eigenvalues = system_matrix.complex_eigenvalues();
        
        let mut stable = true;
        let mut min_real_part = 0.0;

        for eigenvalue in &eigenvalues {
            if eigenvalue.re >= 0.0 {
                stable = false;
            }
            min_real_part = min_real_part.min(eigenvalue.re);
        }

        let margin = -min_real_part;

        Ok(StabilityResult {
            criterion: "Lyapunov".to_string(),
            stable,
            margin,
            additional_info: format!("Max eigenvalue real part: {:.3}", -min_real_part),
        })
    }

    /// Apply S-entropy viability stability criterion
    fn apply_s_entropy_viability_criterion(
        &self, 
        poles: &[Complex<f64>], 
        viability_threshold: f64, 
        miracle_tolerance: f64
    ) -> Result<StabilityResult> {
        // Calculate S-entropy cost of current pole configuration
        let mut s_entropy_cost = 0.0;
        let mut miraculous_poles = 0;

        for pole in poles {
            // Calculate S-entropy cost for this pole
            let knowledge_cost = if pole.re > 0.0 { 
                // Unstable poles require infinite knowledge miracle
                1000.0 * pole.re.abs()
            } else {
                0.0
            };

            let time_cost = if pole.im.abs() > 100.0 {
                // Very fast dynamics require time miracles
                pole.im.abs() / 100.0
            } else {
                0.0
            };

            let entropy_cost = if pole.re > 0.0 {
                // Unstable poles increase entropy, need reversal
                500.0 * pole.re
            } else {
                0.0
            };

            let pole_s_cost = (knowledge_cost.powi(2) + time_cost.powi(2) + entropy_cost.powi(2)).sqrt();
            s_entropy_cost += pole_s_cost;

            if pole_s_cost > miracle_tolerance * viability_threshold {
                miraculous_poles += 1;
            }
        }

        let s_viable = s_entropy_cost <= viability_threshold;
        let margin = viability_threshold - s_entropy_cost;

        let stability_status = if s_viable {
            if miraculous_poles > 0 {
                "MiraculouslyStable"
            } else {
                "S-EntropyStable"
            }
        } else {
            "S-EntropyUnstable"
        };

        Ok(StabilityResult {
            criterion: "S-Entropy Viability".to_string(),
            stable: s_viable,
            margin,
            additional_info: format!("{}, S-cost: {:.1}, Miraculous poles: {}", 
                stability_status, s_entropy_cost, miraculous_poles),
        })
    }

    /// Determine overall stability status from individual results
    fn determine_overall_stability(&self, results: &[StabilityResult]) -> Result<StabilityStatus> {
        let mut stable_count = 0;
        let mut unstable_count = 0;
        let mut miraculous_stable = false;

        for result in results {
            if result.stable {
                stable_count += 1;
                if result.criterion == "S-Entropy Viability" && result.additional_info.contains("Miraculous") {
                    miraculous_stable = true;
                }
            } else {
                unstable_count += 1;
            }
        }

        let total_criteria = results.len();
        
        if stable_count == total_criteria {
            if miraculous_stable {
                Ok(StabilityStatus::MiraculouslyStable)
            } else {
                Ok(StabilityStatus::Stable)
            }
        } else if unstable_count == total_criteria {
            Ok(StabilityStatus::Unstable)
        } else if stable_count > unstable_count {
            Ok(StabilityStatus::ConditionallyStable)
        } else {
            Ok(StabilityStatus::MarginallyStable)
        }
    }

    /// Calculate comprehensive stability margins
    fn calculate_stability_margins(&self, poles: &[Complex<f64>], zeros: &[Complex<f64>]) -> Result<StabilityMargins> {
        // Gain margin calculation
        let gain_margin = if !poles.is_empty() {
            let critical_pole = poles.iter()
                .filter(|p| p.im.abs() > 1e-6) // Find complex poles
                .min_by(|a, b| a.re.partial_cmp(&b.re).unwrap())
                .unwrap_or(&poles[0]);
            
            if critical_pole.im.abs() > 1e-6 {
                20.0 * (-critical_pole.re / critical_pole.im.abs()).log10()
            } else {
                f64::INFINITY
            }
        } else {
            f64::INFINITY
        };

        // Phase margin calculation
        let phase_margin = if !poles.is_empty() {
            let dominant_pole = poles.iter()
                .max_by(|a, b| a.re.partial_cmp(&b.re).unwrap())
                .unwrap();
            
            if dominant_pole.im.abs() > 1e-6 {
                180.0 + dominant_pole.arg() * 180.0 / std::f64::consts::PI
            } else {
                90.0
            }
        } else {
            90.0
        };

        // Delay margin calculation
        let delay_margin = if phase_margin > 0.0 {
            let crossover_freq = poles.iter()
                .map(|p| p.im.abs())
                .fold(0.0, f64::max);
            
            if crossover_freq > 0.0 {
                phase_margin * std::f64::consts::PI / (180.0 * crossover_freq)
            } else {
                f64::INFINITY
            }
        } else {
            0.0
        };

        // S-entropy viability margin
        let s_viability_margin = self.calculate_s_viability_margin(poles)?;

        Ok(StabilityMargins {
            gain_margin: gain_margin.max(-60.0), // Limit to reasonable range
            phase_margin: phase_margin.max(-180.0),
            delay_margin: delay_margin.max(0.0),
            s_viability_margin,
        })
    }

    /// Calculate S-entropy viability margin
    fn calculate_s_viability_margin(&self, poles: &[Complex<f64>]) -> Result<f64> {
        let viability_threshold = 10000.0; // Default threshold
        let mut total_s_cost = 0.0;

        for pole in poles {
            // Calculate S-entropy cost for maintaining this pole
            let knowledge_cost = if pole.re > 0.0 { pole.re * 1000.0 } else { 0.0 };
            let time_cost = pole.im.abs() * 10.0;
            let entropy_cost = if pole.re > 0.0 { pole.re * 500.0 } else { 0.0 };
            
            total_s_cost += (knowledge_cost.powi(2) + time_cost.powi(2) + entropy_cost.powi(2)).sqrt();
        }

        Ok(viability_threshold - total_s_cost)
    }

    /// Perform robustness analysis
    fn perform_robustness_analysis(&self, system_matrix: &DMatrix<Complex<f64>>, poles: &[Complex<f64>]) -> Result<RobustnessAnalysis> {
        let n = system_matrix.nrows();
        
        if n == 0 {
            return Ok(RobustnessAnalysis::default());
        }

        // Calculate sensitivity matrix (simplified)
        let mut sensitivity_matrix = DMatrix::zeros(n, n);
        
        for (i, pole) in poles.iter().enumerate().take(n) {
            for j in 0..n {
                // Sensitivity of pole i to parameter j
                let sensitivity = if i == j {
                    -1.0 / pole.re.abs().max(1e-6)
                } else {
                    0.1 / (pole.norm() + 1e-6)
                };
                sensitivity_matrix[(i, j)] = sensitivity;
            }
        }

        // Calculate worst-case stability margin
        let worst_case_margin = poles.iter()
            .map(|p| -p.re)
            .fold(f64::INFINITY, f64::min)
            .max(0.0);

        // Calculate stability bounds for parameter variations
        let stability_bounds = poles.iter()
            .map(|pole| {
                let margin = -pole.re;
                let tolerance = margin * 0.1; // 10% tolerance
                (margin - tolerance, margin + tolerance)
            })
            .collect();

        // Calculate robustness with miraculous enhancement
        let miraculous_robustness = self.calculate_miraculous_robustness(poles)?;

        Ok(RobustnessAnalysis {
            sensitivity_matrix,
            worst_case_margin,
            stability_bounds,
            miraculous_robustness,
        })
    }

    /// Calculate robustness enhancement through miraculous dynamics
    fn calculate_miraculous_robustness(&self, poles: &[Complex<f64>]) -> Result<f64> {
        let mut base_robustness = 1.0;
        let mut miracle_enhancement = 0.0;

        for pole in poles {
            // Base robustness decreases with proximity to instability
            let pole_robustness = (-pole.re).max(0.0) / (pole.norm() + 1.0);
            base_robustness = base_robustness.min(pole_robustness);

            // Miracle enhancement available for unstable poles
            if pole.re > 0.0 {
                miracle_enhancement += 1.0 / (1.0 + pole.re);
            }
        }

        // Total robustness includes miraculous enhancement
        Ok(base_robustness + miracle_enhancement)
    }

    /// Check stability for simulation step
    pub fn check_stability(&mut self, laplace_analysis: &LaplaceAnalysisResult) -> Result<StabilityCheckResult> {
        // Update stability analysis based on current system state
        // (In full implementation, would update based on current analysis)
        
        let margin = match self.stability_status {
            StabilityStatus::Stable => self.stability_margins.gain_margin.min(self.stability_margins.phase_margin / 10.0),
            StabilityStatus::MiraculouslyStable => self.stability_margins.s_viability_margin / 1000.0,
            StabilityStatus::MarginallyStable => 0.1,
            StabilityStatus::ConditionallyStable => 0.5,
            StabilityStatus::Unstable => -1.0,
        };

        Ok(StabilityCheckResult { margin })
    }

    /// Get stability analysis report
    pub fn get_stability_report(&self) -> StabilityReport {
        StabilityReport {
            overall_status: self.stability_status.clone(),
            stability_margins: self.stability_margins.clone(),
            robustness_summary: RobustnessSummary {
                worst_case_margin: self.robustness_analysis.worst_case_margin,
                parameter_sensitivity: self.robustness_analysis.sensitivity_matrix.norm(),
                miraculous_enhancement: self.robustness_analysis.miraculous_robustness,
            },
            recommendations: self.generate_stability_recommendations(),
        }
    }

    /// Generate recommendations for improving stability
    fn generate_stability_recommendations(&self) -> Vec<String> {
        let mut recommendations = Vec::new();

        match self.stability_status {
            StabilityStatus::Unstable => {
                recommendations.push("System is unstable - immediate action required".to_string());
                recommendations.push("Consider enabling S-entropy miraculous stabilization".to_string());
                recommendations.push("Reduce system gain or increase damping".to_string());
            },
            StabilityStatus::MarginallyStable => {
                recommendations.push("System is marginally stable - monitor closely".to_string());
                recommendations.push("Increase stability margins through tuning".to_string());
            },
            StabilityStatus::ConditionallyStable => {
                recommendations.push("Stability depends on operating conditions".to_string());
                recommendations.push("Establish robust operating envelope".to_string());
            },
            StabilityStatus::Stable => {
                recommendations.push("System is stable under normal conditions".to_string());
            },
            StabilityStatus::MiraculouslyStable => {
                recommendations.push("System achieves stability through S-entropy miracles".to_string());
                recommendations.push("Monitor miracle utilization levels".to_string());
                recommendations.push("Consider optimization to reduce miracle dependency".to_string());
            },
        }

        // Add margin-specific recommendations
        if self.stability_margins.gain_margin < 6.0 {
            recommendations.push("Low gain margin - reduce controller gain".to_string());
        }
        if self.stability_margins.phase_margin < 30.0 {
            recommendations.push("Low phase margin - add lead compensation".to_string());
        }
        if self.stability_margins.s_viability_margin < 1000.0 {
            recommendations.push("Approaching S-entropy viability limit".to_string());
        }

        recommendations
    }
}

/// Result of applying a stability criterion
#[derive(Debug, Clone, Serialize, Deserialize)]
struct StabilityResult {
    criterion: String,
    stable: bool,
    margin: f64,
    additional_info: String,
}

/// Comprehensive stability report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StabilityReport {
    pub overall_status: StabilityStatus,
    pub stability_margins: StabilityMargins,
    pub robustness_summary: RobustnessSummary,
    pub recommendations: Vec<String>,
}

/// Summary of robustness analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RobustnessSummary {
    pub worst_case_margin: f64,
    pub parameter_sensitivity: f64,
    pub miraculous_enhancement: f64,
}

impl Default for StabilityMargins {
    fn default() -> Self {
        Self {
            gain_margin: f64::INFINITY,
            phase_margin: 90.0,
            delay_margin: f64::INFINITY,
            s_viability_margin: 10000.0,
        }
    }
}

impl Default for RobustnessAnalysis {
    fn default() -> Self {
        Self {
            sensitivity_matrix: DMatrix::zeros(0, 0),
            worst_case_margin: f64::INFINITY,
            stability_bounds: Vec::new(),
            miraculous_robustness: 1.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_stability_analyzer_creation() {
        let analyzer = StabilityAnalyzer::new();
        assert_eq!(analyzer.stability_criteria.len(), 4);
        assert!(matches!(analyzer.stability_status, StabilityStatus::Stable));
    }
    
    #[test]
    fn test_routh_hurwitz_criterion_stable() {
        let analyzer = StabilityAnalyzer::new();
        
        // Stable poles (all in left half-plane)
        let poles = vec![
            Complex::new(-1.0, 2.0),
            Complex::new(-1.0, -2.0),
            Complex::new(-5.0, 0.0),
        ];
        
        let result = analyzer.apply_routh_hurwitz_criterion(&poles);
        assert!(result.is_ok());
        
        let stability_result = result.unwrap();
        assert!(stability_result.stable);
        assert!(stability_result.margin > 0.0);
    }
    
    #[test]
    fn test_routh_hurwitz_criterion_unstable() {
        let analyzer = StabilityAnalyzer::new();
        
        // Unstable poles (some in right half-plane)
        let poles = vec![
            Complex::new(1.0, 2.0),  // Unstable
            Complex::new(-1.0, -2.0),
            Complex::new(-5.0, 0.0),
        ];
        
        let result = analyzer.apply_routh_hurwitz_criterion(&poles);
        assert!(result.is_ok());
        
        let stability_result = result.unwrap();
        assert!(!stability_result.stable);
    }
    
    #[test]
    fn test_s_entropy_viability_criterion() {
        let analyzer = StabilityAnalyzer::new();
        
        // Mildly unstable poles that might be viable with miracles
        let poles = vec![
            Complex::new(0.1, 1.0),  // Slightly unstable
            Complex::new(-2.0, 3.0),
        ];
        
        let viability_threshold = 10000.0;
        let miracle_tolerance = 0.1;
        
        let result = analyzer.apply_s_entropy_viability_criterion(&poles, viability_threshold, miracle_tolerance);
        assert!(result.is_ok());
        
        let stability_result = result.unwrap();
        assert_eq!(stability_result.criterion, "S-Entropy Viability");
        // Should be viable since the unstable pole is only slightly unstable
    }
    
    #[test]
    fn test_stability_margins_calculation() {
        let analyzer = StabilityAnalyzer::new();
        
        let poles = vec![
            Complex::new(-1.0, 2.0),
            Complex::new(-3.0, 0.0),
        ];
        let zeros = vec![];
        
        let result = analyzer.calculate_stability_margins(&poles, &zeros);
        assert!(result.is_ok());
        
        let margins = result.unwrap();
        assert!(margins.gain_margin > 0.0);
        assert!(margins.phase_margin > 0.0);
        assert!(margins.s_viability_margin > 0.0);
    }
    
    #[test]
    fn test_robustness_analysis() {
        let analyzer = StabilityAnalyzer::new();
        
        let system_matrix = DMatrix::from_row_slice(2, 2, &[
            -1.0, 0.0, 1.0, 0.0,
            0.0, -2.0, 0.0, 0.0,
        ]).map(|x| Complex::new(x, 0.0));
        
        let poles = vec![
            Complex::new(-1.0, 0.0),
            Complex::new(-2.0, 0.0),
        ];
        
        let result = analyzer.perform_robustness_analysis(&system_matrix, &poles);
        assert!(result.is_ok());
        
        let robustness = result.unwrap();
        assert_eq!(robustness.sensitivity_matrix.nrows(), 2);
        assert_eq!(robustness.sensitivity_matrix.ncols(), 2);
        assert!(robustness.worst_case_margin > 0.0);
        assert!(robustness.miraculous_robustness > 0.0);
    }
    
    #[test]
    fn test_overall_stability_determination() {
        let analyzer = StabilityAnalyzer::new();
        
        // All criteria stable
        let all_stable = vec![
            StabilityResult {
                criterion: "Test1".to_string(),
                stable: true,
                margin: 5.0,
                additional_info: "Good".to_string(),
            },
            StabilityResult {
                criterion: "Test2".to_string(),
                stable: true,
                margin: 3.0,
                additional_info: "OK".to_string(),
            },
        ];
        
        let result = analyzer.determine_overall_stability(&all_stable);
        assert!(result.is_ok());
        assert!(matches!(result.unwrap(), StabilityStatus::Stable));
        
        // Mixed results
        let mixed_results = vec![
            StabilityResult {
                criterion: "Test1".to_string(),
                stable: true,
                margin: 5.0,
                additional_info: "Good".to_string(),
            },
            StabilityResult {
                criterion: "Test2".to_string(),
                stable: false,
                margin: -1.0,
                additional_info: "Bad".to_string(),
            },
        ];
        
        let result = analyzer.determine_overall_stability(&mixed_results);
        assert!(result.is_ok());
        assert!(matches!(result.unwrap(), StabilityStatus::MarginallyStable));
    }
}
