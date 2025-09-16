//! # Fuzzy Logic Controller for Biological Transitions
//!
//! Advanced fuzzy logic control system for managing biological state transitions
//! in bioreactors, enhanced with S-entropy principles and miraculous membership
//! functions that can handle impossible biological states.

use super::*;
use crate::error::{MogadishuError, Result};
use nalgebra::DVector;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

impl FuzzyTransitionController {
    /// Create new fuzzy transition controller
    pub fn new() -> Result<Self> {
        Ok(Self {
            fuzzy_rules: Vec::new(),
            membership_functions: HashMap::new(),
            defuzzification: DefuzzificationMethod::SEntropyOptimal,
            current_state: FuzzyState::default(),
        })
    }

    /// Setup biological transition rules for bioprocess
    pub fn setup_biological_rules(&mut self, config: &BioprocessConfig) -> Result<()> {
        // Clear existing rules
        self.fuzzy_rules.clear();
        self.membership_functions.clear();
        
        // Create standard biological membership functions
        self.create_biological_membership_functions()?;
        
        // Add provided fuzzy rules
        for rule in &config.fuzzy_rules {
            self.fuzzy_rules.push(rule.clone());
        }
        
        // Add default biological transition rules if none provided
        if self.fuzzy_rules.is_empty() {
            self.create_default_biological_rules()?;
        }
        
        Ok(())
    }

    /// Create standard biological membership functions
    fn create_biological_membership_functions(&mut self) -> Result<()> {
        // Cell density membership functions
        self.membership_functions.insert(
            "cell_density_low".to_string(),
            MembershipFunction::Trapezoidal { left: 0.0, left_top: 0.0, right_top: 2.0, right: 5.0 }
        );
        self.membership_functions.insert(
            "cell_density_medium".to_string(),
            MembershipFunction::Triangular { left: 2.0, center: 5.0, right: 8.0 }
        );
        self.membership_functions.insert(
            "cell_density_high".to_string(),
            MembershipFunction::Trapezoidal { left: 6.0, left_top: 8.0, right_top: 15.0, right: 15.0 }
        );

        // Substrate concentration membership functions
        self.membership_functions.insert(
            "substrate_depleted".to_string(),
            MembershipFunction::Trapezoidal { left: 0.0, left_top: 0.0, right_top: 1.0, right: 3.0 }
        );
        self.membership_functions.insert(
            "substrate_adequate".to_string(),
            MembershipFunction::Triangular { left: 2.0, center: 5.0, right: 8.0 }
        );
        self.membership_functions.insert(
            "substrate_excess".to_string(),
            MembershipFunction::Trapezoidal { left: 6.0, left_top: 10.0, right_top: 20.0, right: 20.0 }
        );

        // Oxygen level membership functions
        self.membership_functions.insert(
            "oxygen_critical".to_string(),
            MembershipFunction::Trapezoidal { left: 0.0, left_top: 0.0, right_top: 1.0, right: 2.5 }
        );
        self.membership_functions.insert(
            "oxygen_normal".to_string(),
            MembershipFunction::Triangular { left: 2.0, center: 4.0, right: 6.0 }
        );
        self.membership_functions.insert(
            "oxygen_saturated".to_string(),
            MembershipFunction::Trapezoidal { left: 5.0, left_top: 7.0, right_top: 8.0, right: 8.0 }
        );

        // pH membership functions
        self.membership_functions.insert(
            "ph_acidic".to_string(),
            MembershipFunction::Trapezoidal { left: 4.0, left_top: 4.0, right_top: 6.0, right: 6.8 }
        );
        self.membership_functions.insert(
            "ph_neutral".to_string(),
            MembershipFunction::Triangular { left: 6.5, center: 7.0, right: 7.5 }
        );
        self.membership_functions.insert(
            "ph_basic".to_string(),
            MembershipFunction::Trapezoidal { left: 7.2, left_top: 8.0, right_top: 9.0, right: 9.0 }
        );

        // Temperature membership functions
        self.membership_functions.insert(
            "temperature_low".to_string(),
            MembershipFunction::Trapezoidal { left: 20.0, left_top: 20.0, right_top: 28.0, right: 32.0 }
        );
        self.membership_functions.insert(
            "temperature_optimal".to_string(),
            MembershipFunction::Triangular { left: 30.0, center: 37.0, right: 42.0 }
        );
        self.membership_functions.insert(
            "temperature_high".to_string(),
            MembershipFunction::Trapezoidal { left: 40.0, left_top: 45.0, right_top: 50.0, right: 50.0 }
        );

        // Growth phase membership functions
        self.membership_functions.insert(
            "lag_phase".to_string(),
            MembershipFunction::Gaussian { center: 0.5, sigma: 0.3 }
        );
        self.membership_functions.insert(
            "exponential_phase".to_string(),
            MembershipFunction::Gaussian { center: 2.0, sigma: 0.8 }
        );
        self.membership_functions.insert(
            "stationary_phase".to_string(),
            MembershipFunction::Gaussian { center: 4.0, sigma: 1.0 }
        );

        // Miraculous membership functions (S-entropy enhanced)
        self.membership_functions.insert(
            "impossible_growth_rate".to_string(),
            MembershipFunction::MiraculousMembership {
                base_function: Box::new(MembershipFunction::Triangular { left: 0.5, center: 2.0, right: 5.0 }),
                miracle_enhancement: 2.0,
                impossibility_factor: 0.3,
            }
        );

        Ok(())
    }

    /// Create default biological transition rules
    fn create_default_biological_rules(&mut self) -> Result<()> {
        // Rule 1: If cell density is low and substrate is adequate, increase oxygen flow
        self.fuzzy_rules.push(FuzzyRule {
            id: "increase_oxygen_for_growth".to_string(),
            antecedents: vec![
                FuzzyCondition {
                    variable: "cell_density".to_string(),
                    linguistic_value: "cell_density_low".to_string(),
                    membership_degree: 0.0, // Will be calculated
                },
                FuzzyCondition {
                    variable: "substrate_concentration".to_string(),
                    linguistic_value: "substrate_adequate".to_string(),
                    membership_degree: 0.0,
                },
            ],
            consequents: vec![
                FuzzyAction {
                    control_variable: "oxygen_flow_rate".to_string(),
                    intensity: 0.8,
                    certainty: 0.9,
                },
            ],
            weight: 1.0,
            s_entropy_enhancement: 1.0,
        });

        // Rule 2: If substrate is depleted, increase feed rate
        self.fuzzy_rules.push(FuzzyRule {
            id: "increase_feed_for_depletion".to_string(),
            antecedents: vec![
                FuzzyCondition {
                    variable: "substrate_concentration".to_string(),
                    linguistic_value: "substrate_depleted".to_string(),
                    membership_degree: 0.0,
                },
            ],
            consequents: vec![
                FuzzyAction {
                    control_variable: "substrate_feed_rate".to_string(),
                    intensity: 1.0,
                    certainty: 0.95,
                },
            ],
            weight: 1.2, // High priority rule
            s_entropy_enhancement: 1.0,
        });

        // Rule 3: If oxygen is critical, dramatically increase aeration
        self.fuzzy_rules.push(FuzzyRule {
            id: "emergency_aeration".to_string(),
            antecedents: vec![
                FuzzyCondition {
                    variable: "dissolved_oxygen".to_string(),
                    linguistic_value: "oxygen_critical".to_string(),
                    membership_degree: 0.0,
                },
            ],
            consequents: vec![
                FuzzyAction {
                    control_variable: "oxygen_flow_rate".to_string(),
                    intensity: 1.0,
                    certainty: 1.0,
                },
                FuzzyAction {
                    control_variable: "agitation_speed".to_string(),
                    intensity: 0.8,
                    certainty: 0.9,
                },
            ],
            weight: 2.0, // Emergency rule
            s_entropy_enhancement: 1.0,
        });

        // Rule 4: pH control
        self.fuzzy_rules.push(FuzzyRule {
            id: "ph_correction".to_string(),
            antecedents: vec![
                FuzzyCondition {
                    variable: "ph".to_string(),
                    linguistic_value: "ph_acidic".to_string(),
                    membership_degree: 0.0,
                },
            ],
            consequents: vec![
                FuzzyAction {
                    control_variable: "base_addition_rate".to_string(),
                    intensity: 0.7,
                    certainty: 0.85,
                },
            ],
            weight: 1.5,
            s_entropy_enhancement: 1.0,
        });

        // Rule 5: Temperature optimization
        self.fuzzy_rules.push(FuzzyRule {
            id: "temperature_optimization".to_string(),
            antecedents: vec![
                FuzzyCondition {
                    variable: "cell_density".to_string(),
                    linguistic_value: "cell_density_high".to_string(),
                    membership_degree: 0.0,
                },
                FuzzyCondition {
                    variable: "temperature".to_string(),
                    linguistic_value: "temperature_low".to_string(),
                    membership_degree: 0.0,
                },
            ],
            consequents: vec![
                FuzzyAction {
                    control_variable: "heating_rate".to_string(),
                    intensity: 0.6,
                    certainty: 0.8,
                },
            ],
            weight: 1.0,
            s_entropy_enhancement: 1.0,
        });

        // Rule 6: Miraculous growth enhancement (S-entropy enhanced)
        self.fuzzy_rules.push(FuzzyRule {
            id: "miraculous_growth_boost".to_string(),
            antecedents: vec![
                FuzzyCondition {
                    variable: "growth_rate".to_string(),
                    linguistic_value: "impossible_growth_rate".to_string(),
                    membership_degree: 0.0,
                },
            ],
            consequents: vec![
                FuzzyAction {
                    control_variable: "miracle_enhancement_factor".to_string(),
                    intensity: 1.0,
                    certainty: 0.7,
                },
            ],
            weight: 0.5, // Lower weight due to miracle cost
            s_entropy_enhancement: 2.0, // Significant S-entropy enhancement
        });

        Ok(())
    }

    /// Process biological transitions using fuzzy logic
    pub fn process_transitions(&mut self, state_variables: &HashMap<String, f64>, lp_solution: &LinearProgrammingSolution) -> Result<FuzzyOutputs> {
        // Update current fuzzy state
        self.update_fuzzy_state(state_variables, lp_solution)?;
        
        // Fuzzify inputs
        let fuzzified_inputs = self.fuzzify_inputs(state_variables)?;
        
        // Apply fuzzy rules
        let rule_activations = self.apply_fuzzy_rules(&fuzzified_inputs)?;
        
        // Aggregate rule outputs
        let aggregated_outputs = self.aggregate_rule_outputs(&rule_activations)?;
        
        // Defuzzify to get crisp control outputs
        let control_outputs = self.defuzzify_outputs(&aggregated_outputs)?;
        
        // Update current state
        self.current_state.variables = state_variables.clone();
        self.current_state.control_outputs = control_outputs.clone();
        self.current_state.rule_activations = rule_activations;
        self.current_state.inference_confidence = self.calculate_inference_confidence();
        
        Ok(FuzzyOutputs { control_outputs })
    }

    /// Update fuzzy state with current measurements
    fn update_fuzzy_state(&mut self, state_variables: &HashMap<String, f64>, lp_solution: &LinearProgrammingSolution) -> Result<()> {
        // Incorporate linear programming solution into fuzzy state
        let lp_influence = lp_solution.objective_value / 100.0; // Normalize LP influence
        
        // Update state variables with LP influence
        let mut updated_variables = state_variables.clone();
        
        // LP solution can suggest modifications to biological variables
        if let Some(existing_value) = updated_variables.get_mut("substrate_concentration") {
            *existing_value *= (1.0 + lp_influence * 0.1); // Small LP influence
        }
        
        self.current_state.variables = updated_variables;
        
        Ok(())
    }

    /// Fuzzify input variables
    fn fuzzify_inputs(&self, state_variables: &HashMap<String, f64>) -> Result<HashMap<String, HashMap<String, f64>>> {
        let mut fuzzified = HashMap::new();
        
        for (variable_name, &variable_value) in state_variables {
            let mut memberships = HashMap::new();
            
            // Find all membership functions for this variable
            for (membership_name, membership_function) in &self.membership_functions {
                if membership_name.starts_with(variable_name) || self.is_related_membership(variable_name, membership_name) {
                    let membership_degree = self.evaluate_membership_function(membership_function, variable_value)?;
                    memberships.insert(membership_name.clone(), membership_degree);
                }
            }
            
            fuzzified.insert(variable_name.clone(), memberships);
        }
        
        Ok(fuzzified)
    }

    /// Check if membership function is related to variable
    fn is_related_membership(&self, variable_name: &str, membership_name: &str) -> bool {
        // Custom logic to match variables with membership functions
        match variable_name {
            "cell_density" => membership_name.contains("cell_density") || membership_name.contains("growth"),
            "substrate_concentration" => membership_name.contains("substrate"),
            "dissolved_oxygen" => membership_name.contains("oxygen"),
            "ph" => membership_name.contains("ph"),
            "temperature" => membership_name.contains("temperature"),
            "growth_rate" => membership_name.contains("growth") || membership_name.contains("impossible"),
            _ => false,
        }
    }

    /// Evaluate membership function for given value
    fn evaluate_membership_function(&self, membership_function: &MembershipFunction, value: f64) -> Result<f64> {
        let membership = match membership_function {
            MembershipFunction::Triangular { left, center, right } => {
                if value <= *left || value >= *right {
                    0.0
                } else if value <= *center {
                    (value - left) / (center - left)
                } else {
                    (right - value) / (right - center)
                }
            },
            
            MembershipFunction::Trapezoidal { left, left_top, right_top, right } => {
                if value <= *left || value >= *right {
                    0.0
                } else if value <= *left_top {
                    (value - left) / (left_top - left)
                } else if value <= *right_top {
                    1.0
                } else {
                    (right - value) / (right - right_top)
                }
            },
            
            MembershipFunction::Gaussian { center, sigma } => {
                let exponent = -0.5 * ((value - center) / sigma).powi(2);
                exponent.exp()
            },
            
            MembershipFunction::Sigmoid { slope, shift } => {
                1.0 / (1.0 + (-slope * (value - shift)).exp())
            },
            
            MembershipFunction::MiraculousMembership { base_function, miracle_enhancement, impossibility_factor } => {
                let base_membership = self.evaluate_membership_function(base_function, value)?;
                
                // Miraculous enhancement can create impossible membership degrees
                let enhanced_membership = base_membership * miracle_enhancement;
                
                // Apply impossibility factor (can exceed 1.0!)
                let impossible_membership = enhanced_membership + impossibility_factor * (1.0 - base_membership);
                
                impossible_membership.max(0.0) // Can exceed 1.0 for miraculous cases
            },
        };
        
        Ok(membership)
    }

    /// Apply fuzzy rules to fuzzified inputs
    fn apply_fuzzy_rules(&self, fuzzified_inputs: &HashMap<String, HashMap<String, f64>>) -> Result<HashMap<String, f64>> {
        let mut rule_activations = HashMap::new();
        
        for rule in &self.fuzzy_rules {
            let activation_strength = self.calculate_rule_activation(rule, fuzzified_inputs)?;
            rule_activations.insert(rule.id.clone(), activation_strength);
        }
        
        Ok(rule_activations)
    }

    /// Calculate activation strength for a fuzzy rule
    fn calculate_rule_activation(&self, rule: &FuzzyRule, fuzzified_inputs: &HashMap<String, HashMap<String, f64>>) -> Result<f64> {
        let mut activation_strength = 1.0;
        
        // AND operation for antecedents (minimum)
        for antecedent in &rule.antecedents {
            if let Some(variable_memberships) = fuzzified_inputs.get(&antecedent.variable) {
                if let Some(&membership_degree) = variable_memberships.get(&antecedent.linguistic_value) {
                    activation_strength = activation_strength.min(membership_degree);
                } else {
                    activation_strength = 0.0; // Missing membership
                    break;
                }
            } else {
                activation_strength = 0.0; // Missing variable
                break;
            }
        }
        
        // Apply rule weight and S-entropy enhancement
        activation_strength *= rule.weight * rule.s_entropy_enhancement;
        
        Ok(activation_strength)
    }

    /// Aggregate rule outputs
    fn aggregate_rule_outputs(&self, rule_activations: &HashMap<String, f64>) -> Result<HashMap<String, Vec<(f64, f64)>>> {
        let mut aggregated = HashMap::new();
        
        for rule in &self.fuzzy_rules {
            if let Some(&activation_strength) = rule_activations.get(&rule.id) {
                if activation_strength > 0.0 {
                    for consequent in &rule.consequents {
                        let output_strength = activation_strength * consequent.intensity * consequent.certainty;
                        
                        aggregated
                            .entry(consequent.control_variable.clone())
                            .or_insert_with(Vec::new)
                            .push((output_strength, consequent.intensity));
                    }
                }
            }
        }
        
        Ok(aggregated)
    }

    /// Defuzzify aggregated outputs to crisp values
    fn defuzzify_outputs(&self, aggregated_outputs: &HashMap<String, Vec<(f64, f64)>>) -> Result<HashMap<String, f64>> {
        let mut control_outputs = HashMap::new();
        
        for (control_variable, output_list) in aggregated_outputs {
            let defuzzified_value = match self.defuzzification {
                DefuzzificationMethod::Centroid => self.centroid_defuzzification(output_list)?,
                DefuzzificationMethod::MaximumMembership => self.maximum_membership_defuzzification(output_list)?,
                DefuzzificationMethod::MeanOfMaxima => self.mean_of_maxima_defuzzification(output_list)?,
                DefuzzificationMethod::WeightedAverage => self.weighted_average_defuzzification(output_list)?,
                DefuzzificationMethod::SEntropyOptimal => self.s_entropy_optimal_defuzzification(output_list)?,
            };
            
            control_outputs.insert(control_variable.clone(), defuzzified_value);
        }
        
        Ok(control_outputs)
    }

    /// Centroid defuzzification method
    fn centroid_defuzzification(&self, output_list: &[(f64, f64)]) -> Result<f64> {
        if output_list.is_empty() {
            return Ok(0.0);
        }
        
        let mut numerator = 0.0;
        let mut denominator = 0.0;
        
        for &(strength, intensity) in output_list {
            numerator += strength * intensity;
            denominator += strength;
        }
        
        if denominator > 0.0 {
            Ok(numerator / denominator)
        } else {
            Ok(0.0)
        }
    }

    /// Maximum membership defuzzification
    fn maximum_membership_defuzzification(&self, output_list: &[(f64, f64)]) -> Result<f64> {
        output_list.iter()
            .max_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal))
            .map(|&(_, intensity)| intensity)
            .ok_or_else(|| MogadishuError::math("Empty output list for defuzzification"))
    }

    /// Mean of maxima defuzzification
    fn mean_of_maxima_defuzzification(&self, output_list: &[(f64, f64)]) -> Result<f64> {
        if output_list.is_empty() {
            return Ok(0.0);
        }
        
        let max_strength = output_list.iter()
            .map(|&(strength, _)| strength)
            .fold(0.0, f64::max);
        
        let max_outputs: Vec<f64> = output_list.iter()
            .filter(|&&(strength, _)| (strength - max_strength).abs() < 1e-6)
            .map(|&(_, intensity)| intensity)
            .collect();
        
        if max_outputs.is_empty() {
            Ok(0.0)
        } else {
            Ok(max_outputs.iter().sum::<f64>() / max_outputs.len() as f64)
        }
    }

    /// Weighted average defuzzification
    fn weighted_average_defuzzification(&self, output_list: &[(f64, f64)]) -> Result<f64> {
        self.centroid_defuzzification(output_list) // Same as centroid for this case
    }

    /// S-entropy optimal defuzzification (can handle impossible outputs)
    fn s_entropy_optimal_defuzzification(&self, output_list: &[(f64, f64)]) -> Result<f64> {
        if output_list.is_empty() {
            return Ok(0.0);
        }
        
        // S-entropy enhanced defuzzification can consider impossible solutions
        let mut s_entropy_weighted_sum = 0.0;
        let mut s_entropy_weight_sum = 0.0;
        
        for &(strength, intensity) in output_list {
            // S-entropy weight considers both strength and impossibility potential
            let s_weight = strength * (1.0 + intensity.max(1.0) - 1.0); // Enhanced for impossible outputs
            
            s_entropy_weighted_sum += s_weight * intensity;
            s_entropy_weight_sum += s_weight;
        }
        
        if s_entropy_weight_sum > 0.0 {
            let result = s_entropy_weighted_sum / s_entropy_weight_sum;
            
            // S-entropy can produce outputs beyond normal limits
            Ok(result)
        } else {
            Ok(0.0)
        }
    }

    /// Calculate confidence in fuzzy inference
    fn calculate_inference_confidence(&self) -> f64 {
        let active_rules = self.current_state.rule_activations.values()
            .filter(|&&activation| activation > 0.1)
            .count();
        
        let total_activation: f64 = self.current_state.rule_activations.values().sum();
        
        if active_rules > 0 {
            (total_activation / active_rules as f64).min(1.0)
        } else {
            0.0
        }
    }

    /// Analyze fuzzy system performance
    pub fn analyze(&self) -> Result<FuzzyAnalysisResult> {
        let mut rule_effectiveness = HashMap::new();
        
        for rule in &self.fuzzy_rules {
            let effectiveness = self.current_state.rule_activations
                .get(&rule.id)
                .cloned()
                .unwrap_or(0.0) * rule.weight;
            
            rule_effectiveness.insert(rule.id.clone(), effectiveness);
        }
        
        Ok(FuzzyAnalysisResult { rule_effectiveness })
    }

    /// Get fuzzy system statistics
    pub fn get_system_statistics(&self) -> FuzzySystemStatistics {
        let num_variables = self.current_state.variables.len();
        let num_membership_functions = self.membership_functions.len();
        let num_rules = self.fuzzy_rules.len();
        
        let miraculous_functions = self.membership_functions.values()
            .filter(|mf| matches!(mf, MembershipFunction::MiraculousMembership { .. }))
            .count();
        
        FuzzySystemStatistics {
            num_variables,
            num_membership_functions,
            num_rules,
            miraculous_functions,
            current_confidence: self.current_state.inference_confidence,
        }
    }
}

impl Default for FuzzyState {
    fn default() -> Self {
        Self {
            variables: HashMap::new(),
            rule_activations: HashMap::new(),
            control_outputs: HashMap::new(),
            inference_confidence: 0.0,
        }
    }
}

/// Statistics about fuzzy system
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FuzzySystemStatistics {
    pub num_variables: usize,
    pub num_membership_functions: usize,
    pub num_rules: usize,
    pub miraculous_functions: usize,
    pub current_confidence: f64,
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_fuzzy_controller_creation() {
        let controller = FuzzyTransitionController::new();
        assert!(controller.is_ok());
        
        let ctrl = controller.unwrap();
        assert_eq!(ctrl.fuzzy_rules.len(), 0);
        assert_eq!(ctrl.membership_functions.len(), 0);
    }
    
    #[test]
    fn test_membership_function_evaluation() {
        let controller = FuzzyTransitionController::new().unwrap();
        
        // Test triangular membership function
        let triangular = MembershipFunction::Triangular { left: 0.0, center: 5.0, right: 10.0 };
        
        assert_eq!(controller.evaluate_membership_function(&triangular, 0.0).unwrap(), 0.0);
        assert_eq!(controller.evaluate_membership_function(&triangular, 5.0).unwrap(), 1.0);
        assert_eq!(controller.evaluate_membership_function(&triangular, 10.0).unwrap(), 0.0);
        assert_eq!(controller.evaluate_membership_function(&triangular, 2.5).unwrap(), 0.5);
    }
    
    #[test]
    fn test_trapezoidal_membership_function() {
        let controller = FuzzyTransitionController::new().unwrap();
        
        let trapezoidal = MembershipFunction::Trapezoidal { 
            left: 0.0, left_top: 2.0, right_top: 8.0, right: 10.0 
        };
        
        assert_eq!(controller.evaluate_membership_function(&trapezoidal, 0.0).unwrap(), 0.0);
        assert_eq!(controller.evaluate_membership_function(&trapezoidal, 2.0).unwrap(), 1.0);
        assert_eq!(controller.evaluate_membership_function(&trapezoidal, 5.0).unwrap(), 1.0);
        assert_eq!(controller.evaluate_membership_function(&trapezoidal, 8.0).unwrap(), 1.0);
        assert_eq!(controller.evaluate_membership_function(&trapezoidal, 10.0).unwrap(), 0.0);
    }
    
    #[test]
    fn test_gaussian_membership_function() {
        let controller = FuzzyTransitionController::new().unwrap();
        
        let gaussian = MembershipFunction::Gaussian { center: 5.0, sigma: 2.0 };
        
        let center_value = controller.evaluate_membership_function(&gaussian, 5.0).unwrap();
        assert!((center_value - 1.0).abs() < 1e-6); // Should be 1.0 at center
        
        let offset_value = controller.evaluate_membership_function(&gaussian, 7.0).unwrap();
        assert!(offset_value < 1.0 && offset_value > 0.0); // Should decrease from center
    }
    
    #[test]
    fn test_miraculous_membership_function() {
        let controller = FuzzyTransitionController::new().unwrap();
        
        let base_function = MembershipFunction::Triangular { left: 0.0, center: 5.0, right: 10.0 };
        let miraculous = MembershipFunction::MiraculousMembership {
            base_function: Box::new(base_function),
            miracle_enhancement: 2.0,
            impossibility_factor: 0.5,
        };
        
        let result = controller.evaluate_membership_function(&miraculous, 5.0).unwrap();
        assert!(result > 1.0); // Miraculous membership can exceed 1.0!
    }
    
    #[test]
    fn test_biological_membership_functions_creation() {
        let mut controller = FuzzyTransitionController::new().unwrap();
        
        let result = controller.create_biological_membership_functions();
        assert!(result.is_ok());
        assert!(controller.membership_functions.len() > 0);
        assert!(controller.membership_functions.contains_key("cell_density_low"));
        assert!(controller.membership_functions.contains_key("substrate_adequate"));
        assert!(controller.membership_functions.contains_key("oxygen_normal"));
    }
    
    #[test]
    fn test_default_biological_rules_creation() {
        let mut controller = FuzzyTransitionController::new().unwrap();
        
        let result = controller.create_default_biological_rules();
        assert!(result.is_ok());
        assert!(controller.fuzzy_rules.len() > 0);
        
        // Check that we have key biological control rules
        let rule_names: Vec<String> = controller.fuzzy_rules.iter()
            .map(|rule| rule.id.clone())
            .collect();
        
        assert!(rule_names.contains(&"increase_oxygen_for_growth".to_string()));
        assert!(rule_names.contains(&"emergency_aeration".to_string()));
        assert!(rule_names.contains(&"miraculous_growth_boost".to_string()));
    }
    
    #[test]
    fn test_fuzzy_inference_process() {
        let mut controller = FuzzyTransitionController::new().unwrap();
        controller.create_biological_membership_functions().unwrap();
        controller.create_default_biological_rules().unwrap();
        
        // Create test state variables
        let mut state_variables = HashMap::new();
        state_variables.insert("cell_density".to_string(), 1.5); // Low
        state_variables.insert("substrate_concentration".to_string(), 4.0); // Adequate
        state_variables.insert("dissolved_oxygen".to_string(), 0.8); // Critical
        
        // Create dummy LP solution
        let lp_solution = LinearProgrammingSolution {
            variables: DVector::from_vec(vec![1.0, 2.0]),
            objective_value: 50.0,
            status: OptimizationStatus::Optimal,
            sensitivity: SensitivityAnalysis {
                shadow_prices: DVector::from_vec(vec![0.1]),
                reduced_costs: DVector::from_vec(vec![0.0, 0.0]),
                rhs_ranges: vec![(5.0, 15.0)],
                obj_ranges: vec![(0.5, 1.5), (1.8, 2.2)],
            },
        };
        
        let result = controller.process_transitions(&state_variables, &lp_solution);
        assert!(result.is_ok());
        
        let fuzzy_outputs = result.unwrap();
        assert!(fuzzy_outputs.control_outputs.len() > 0);
        
        // Should have oxygen flow control due to critical oxygen level
        assert!(fuzzy_outputs.control_outputs.contains_key("oxygen_flow_rate"));
    }
}
