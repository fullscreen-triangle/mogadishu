//! # Laplace Domain Analyzer for System Compression
//!
//! Revolutionary bioprocess system compression using Laplace transforms
//! to reduce 100+ order differential equation systems to compressed
//! representations enabling real-time analysis and miraculous performance.

use super::*;
use crate::error::{MogadishuError, Result};
use nalgebra::{DMatrix, DVector, Complex, SVD};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

impl LaplaceAnalyzer {
    /// Create new Laplace domain analyzer
    pub fn new() -> Self {
        Self {
            time_domain_equations: Vec::new(),
            laplace_system: LaplaceSystem::default(),
            inverse_transformer: InverseLaplaceTransformer::new(),
            compression_stats: CompressionStatistics::default(),
        }
    }

    /// Analyze system and create Laplace representation
    pub fn analyze_system(&mut self, circuit_model: &MiraculousCircuitModel) -> Result<()> {
        // Extract differential equations from circuit model
        self.extract_differential_equations(circuit_model)?;
        
        // Transform to Laplace domain
        self.transform_to_laplace_domain()?;
        
        // Perform system compression
        self.compress_system()?;
        
        // Calculate compression statistics
        self.calculate_compression_statistics()?;
        
        Ok(())
    }

    /// Extract differential equations from circuit components
    fn extract_differential_equations(&mut self, circuit_model: &MiraculousCircuitModel) -> Result<()> {
        self.time_domain_equations.clear();
        
        // Standard component equations
        for component in &circuit_model.standard_components {
            let eq = self.create_component_differential_equation(component)?;
            self.time_domain_equations.push(eq);
        }
        
        // Miraculous component equations (with special terms)
        for miracle_component in &circuit_model.miraculous_components {
            let eq = self.create_miraculous_differential_equation(miracle_component)?;
            self.time_domain_equations.push(eq);
        }
        
        Ok(())
    }

    /// Create differential equation for standard circuit component
    fn create_component_differential_equation(&self, component: &CircuitComponent) -> Result<DifferentialEquation> {
        match component {
            CircuitComponent::OxygenResistor { id, resistance, .. } => {
                // Ohm's law: V = I*R -> dV/dt = R * dI/dt
                Ok(DifferentialEquation {
                    variable: format!("voltage_{}", id),
                    order: 1,
                    coefficients: vec![*resistance, 0.0], // R * d/dt + 0
                    rhs_terms: vec![EquationTerm::Linear { 
                        coefficient: 1.0, 
                        variable: format!("current_{}", id) 
                    }],
                    initial_conditions: vec![0.0],
                })
            },
            
            CircuitComponent::DOCapacitor { id, capacitance, .. } => {
                // Capacitor equation: I = C * dV/dt -> dV/dt = I/C
                Ok(DifferentialEquation {
                    variable: format!("voltage_{}", id),
                    order: 1,
                    coefficients: vec![1.0, 0.0], // d/dt
                    rhs_terms: vec![EquationTerm::Linear {
                        coefficient: 1.0 / capacitance,
                        variable: format!("current_{}", id),
                    }],
                    initial_conditions: vec![0.0],
                })
            },
            
            CircuitComponent::SubstrateInductor { id, inductance, .. } => {
                // Inductor equation: V = L * dI/dt -> dI/dt = V/L
                Ok(DifferentialEquation {
                    variable: format!("current_{}", id),
                    order: 1,
                    coefficients: vec![1.0, 0.0], // d/dt
                    rhs_terms: vec![EquationTerm::Linear {
                        coefficient: 1.0 / inductance,
                        variable: format!("voltage_{}", id),
                    }],
                    initial_conditions: vec![0.0],
                })
            },
            
            CircuitComponent::BufferRC { id, resistance, capacitance, .. } => {
                // RC circuit: RC * dV/dt + V = V_in
                let time_constant = resistance * capacitance;
                Ok(DifferentialEquation {
                    variable: format!("voltage_{}", id),
                    order: 1,
                    coefficients: vec![time_constant, 1.0], // RC * d/dt + 1
                    rhs_terms: vec![EquationTerm::Linear {
                        coefficient: 1.0,
                        variable: format!("input_voltage_{}", id),
                    }],
                    initial_conditions: vec![0.0],
                })
            },
            
            CircuitComponent::ThermalResistor { id, thermal_resistance, .. } => {
                // Thermal equation: dT/dt = (T_in - T)/(R_th * C_th)
                let thermal_capacitance = 1.0; // Assume unit thermal capacitance
                let time_constant = thermal_resistance * thermal_capacitance;
                
                Ok(DifferentialEquation {
                    variable: format!("temperature_{}", id),
                    order: 1,
                    coefficients: vec![1.0, 1.0 / time_constant], // d/dt + 1/(RC)
                    rhs_terms: vec![EquationTerm::Linear {
                        coefficient: 1.0 / time_constant,
                        variable: format!("input_temperature_{}", id),
                    }],
                    initial_conditions: vec![298.15], // Room temperature
                })
            },
        }
    }

    /// Create differential equation for miraculous component
    fn create_miraculous_differential_equation(&self, component: &MiraculousComponent) -> Result<DifferentialEquation> {
        match component {
            MiraculousComponent::NegativeResistor { id, negative_resistance, miracle_level, .. } => {
                // Negative resistor: V = -|R| * I (energy generation!)
                Ok(DifferentialEquation {
                    variable: format!("voltage_{}", id),
                    order: 1,
                    coefficients: vec![1.0, 0.0], // d/dt
                    rhs_terms: vec![
                        EquationTerm::Linear {
                            coefficient: -*negative_resistance,
                            variable: format!("current_{}", id),
                        },
                        EquationTerm::Miraculous {
                            coefficient: *miracle_level,
                            miracle_type: MiracleType::EnergyCreation,
                            s_entropy_cost: miracle_level * 100.0,
                        },
                    ],
                    initial_conditions: vec![0.0],
                })
            },
            
            MiraculousComponent::TemporalCapacitor { id, capacitance, time_reversal_strength, .. } => {
                // Temporal capacitor: I = C * dV/dt + α * C * d²V/dt² (future information leak)
                Ok(DifferentialEquation {
                    variable: format!("voltage_{}", id),
                    order: 2, // Second-order due to future information
                    coefficients: vec![
                        time_reversal_strength * capacitance, // α*C * d²/dt²
                        *capacitance,                         // C * d/dt
                        0.0                                   // constant term
                    ],
                    rhs_terms: vec![
                        EquationTerm::Linear {
                            coefficient: 1.0,
                            variable: format!("current_{}", id),
                        },
                        EquationTerm::Miraculous {
                            coefficient: *time_reversal_strength,
                            miracle_type: MiracleType::FutureInformationLeak,
                            s_entropy_cost: time_reversal_strength * 200.0,
                        },
                    ],
                    initial_conditions: vec![0.0, 0.0], // V(0) and dV/dt(0)
                })
            },
            
            MiraculousComponent::EntropyInductor { id, inductance, entropy_reduction_rate, miracle_entropy_cost, .. } => {
                // Entropy inductor: V = L * dI/dt - S_reduction * kT * ln(states)
                Ok(DifferentialEquation {
                    variable: format!("current_{}", id),
                    order: 1,
                    coefficients: vec![*inductance, 0.0], // L * d/dt
                    rhs_terms: vec![
                        EquationTerm::Linear {
                            coefficient: 1.0,
                            variable: format!("voltage_{}", id),
                        },
                        EquationTerm::Miraculous {
                            coefficient: -*entropy_reduction_rate,
                            miracle_type: MiracleType::EntropyReversal,
                            s_entropy_cost: *miracle_entropy_cost,
                        },
                    ],
                    initial_conditions: vec![0.0],
                })
            },
            
            MiraculousComponent::InformationAmplifier { id, amplification_factor, maxwell_demon_strength, .. } => {
                // Information amplifier: Output = A * Input + D * Maxwell_demon_work
                Ok(DifferentialEquation {
                    variable: format!("output_{}", id),
                    order: 1,
                    coefficients: vec![1.0, 0.0], // d/dt
                    rhs_terms: vec![
                        EquationTerm::Linear {
                            coefficient: *amplification_factor,
                            variable: format!("input_{}", id),
                        },
                        EquationTerm::Miraculous {
                            coefficient: *maxwell_demon_strength,
                            miracle_type: MiracleType::InformationAmplification,
                            s_entropy_cost: maxwell_demon_strength * amplification_factor * 50.0,
                        },
                    ],
                    initial_conditions: vec![0.0],
                })
            },
        }
    }

    /// Transform differential equations to Laplace domain
    fn transform_to_laplace_domain(&mut self) -> Result<()> {
        let n_equations = self.time_domain_equations.len();
        if n_equations == 0 {
            return Err(MogadishuError::math("No differential equations to transform"));
        }

        // Create system matrices in Laplace domain
        let mut system_matrix = DMatrix::zeros(n_equations, n_equations);
        let mut input_matrix = DMatrix::zeros(n_equations, n_equations);
        let mut output_matrix = DMatrix::identity(n_equations, n_equations);
        let feedthrough_matrix = DMatrix::zeros(n_equations, n_equations);

        // Fill system matrix based on differential equations
        for (i, eq) in self.time_domain_equations.iter().enumerate() {
            // Diagonal terms from equation coefficients
            for (order, &coeff) in eq.coefficients.iter().enumerate() {
                if order == 0 {
                    // Constant term
                    system_matrix[(i, i)] += coeff;
                } else if order == 1 {
                    // First derivative: coefficient becomes s*coeff in Laplace domain
                    system_matrix[(i, i)] += coeff; // Will be multiplied by 's' in frequency analysis
                } else if order == 2 {
                    // Second derivative: coefficient becomes s²*coeff
                    system_matrix[(i, i)] += coeff; // Will be multiplied by 's²'
                }
            }

            // Input terms from RHS
            for term in &eq.rhs_terms {
                match term {
                    EquationTerm::Constant(c) => {
                        input_matrix[(i, i)] += c;
                    },
                    EquationTerm::Linear { coefficient, variable: _ } => {
                        // Cross-coupling terms would go here if we tracked variable dependencies
                        input_matrix[(i, i)] += coefficient;
                    },
                    EquationTerm::Nonlinear { coefficient, .. } => {
                        // Linearize around operating point
                        input_matrix[(i, i)] += coefficient;
                    },
                    EquationTerm::Miraculous { coefficient, miracle_type, s_entropy_cost } => {
                        // Miraculous terms modify system behavior
                        match miracle_type {
                            MiracleType::EnergyCreation => {
                                // Negative resistance effect
                                system_matrix[(i, i)] -= coefficient;
                            },
                            MiracleType::FutureInformationLeak => {
                                // Adds predictive terms (negative delay)
                                system_matrix[(i, i)] += coefficient;
                            },
                            MiracleType::EntropyReversal => {
                                // Reduces system damping
                                system_matrix[(i, i)] *= 1.0 - (coefficient / 10.0);
                            },
                            MiracleType::InformationAmplification => {
                                // Increases system gain
                                input_matrix[(i, i)] *= 1.0 + coefficient;
                            },
                        }
                    },
                }
            }
        }

        // Create compressed representation using SVD
        let compressed = self.create_compressed_representation(&system_matrix)?;

        self.laplace_system = LaplaceSystem {
            system_matrix: system_matrix.map(|x| Complex::new(x, 0.0)),
            input_matrix: input_matrix.map(|x| Complex::new(x, 0.0)),
            output_matrix: output_matrix.map(|x| Complex::new(x, 0.0)),
            feedthrough_matrix: feedthrough_matrix.map(|x| Complex::new(x, 0.0)),
            compressed_representation: compressed,
        };

        Ok(())
    }

    /// Create compressed system representation using SVD
    fn create_compressed_representation(&self, system_matrix: &DMatrix<f64>) -> Result<CompressedSystem> {
        // Perform Singular Value Decomposition
        let svd = SVD::new(system_matrix.clone(), true, true);
        
        let u = svd.u.ok_or_else(|| MogadishuError::math("SVD failed to compute U matrix"))?;
        let v_t = svd.v_t.ok_or_else(|| MogadishuError::math("SVD failed to compute V^T matrix"))?;
        let singular_values = svd.singular_values;

        // Determine compression threshold (keep singular values > 1% of max)
        let max_singular_value = singular_values[0];
        let threshold = max_singular_value * 0.01;
        
        let reduced_order = singular_values.iter()
            .position(|&val| val < threshold)
            .unwrap_or(singular_values.len())
            .max(1); // Keep at least one mode

        // Calculate compression metrics
        let original_order = system_matrix.nrows();
        let compression_ratio = original_order as f64 / reduced_order as f64;
        
        let retained_energy: f64 = singular_values.iter()
            .take(reduced_order)
            .map(|x| x * x)
            .sum();
        let total_energy: f64 = singular_values.iter()
            .map(|x| x * x)
            .sum();
        let information_preserved = (retained_energy / total_energy) * 100.0;

        Ok(CompressedSystem {
            svd_u: u,
            svd_s: singular_values,
            svd_v: v_t,
            reduced_order,
            compression_ratio,
            information_preserved,
        })
    }

    /// Perform system compression
    fn compress_system(&mut self) -> Result<()> {
        // System is already compressed during Laplace transform
        // Additional compression can be performed here if needed
        Ok(())
    }

    /// Calculate compression statistics
    fn calculate_compression_statistics(&mut self) -> Result<()> {
        let compressed = &self.laplace_system.compressed_representation;
        
        // Calculate computational speedup (approximately proportional to compression ratio squared)
        let computational_speedup = compressed.compression_ratio.powi(2);
        
        self.compression_stats = CompressionStatistics {
            original_order: self.time_domain_equations.iter()
                .map(|eq| eq.order)
                .sum(),
            compressed_order: compressed.reduced_order,
            compression_ratio: compressed.compression_ratio,
            information_loss: 100.0 - compressed.information_preserved,
            computational_speedup,
        };
        
        Ok(())
    }

    /// Compress and analyze system state for simulation step
    pub fn compress_and_analyze(&mut self, circuit_state: &CircuitState, fuzzy_outputs: &FuzzyOutputs) -> Result<LaplaceAnalysisResult> {
        // Update system state with current circuit and fuzzy controller outputs
        self.update_laplace_state(circuit_state, fuzzy_outputs)?;
        
        // Perform real-time analysis in compressed domain
        let analysis = self.analyze_compressed_system()?;
        
        Ok(LaplaceAnalysisResult {
            compressed_order: self.compression_stats.compressed_order,
        })
    }

    /// Update Laplace domain state
    fn update_laplace_state(&mut self, circuit_state: &CircuitState, fuzzy_outputs: &FuzzyOutputs) -> Result<()> {
        // Update compressed state vector with current measurements
        let state_size = self.laplace_system.compressed_representation.reduced_order;
        let mut new_state = DVector::zeros(state_size);
        
        // Map circuit state variables to compressed state
        let mut state_index = 0;
        for (var_name, &value) in &circuit_state.variables {
            if state_index < state_size {
                new_state[state_index] = Complex::new(value, 0.0);
                state_index += 1;
            }
        }
        
        // Add fuzzy controller outputs
        for (control_name, &value) in &fuzzy_outputs.control_outputs {
            if state_index < state_size {
                new_state[state_index] = Complex::new(value, 0.0);
                state_index += 1;
            }
        }
        
        self.laplace_system.compressed_representation.reduced_order = state_size;
        
        Ok(())
    }

    /// Analyze compressed system performance
    fn analyze_compressed_system(&self) -> Result<()> {
        // Analyze stability in compressed domain
        let stability_margin = self.calculate_compressed_stability_margin()?;
        
        // Update internal state
        // (In a full implementation, this would update the laplace_state)
        
        Ok(())
    }

    /// Calculate stability margin in compressed domain
    fn calculate_compressed_stability_margin(&self) -> Result<f64> {
        // For compressed system, stability is determined by eigenvalues of system matrix
        let system_matrix = &self.laplace_system.system_matrix;
        
        if system_matrix.is_empty() {
            return Ok(f64::INFINITY);
        }
        
        // Calculate eigenvalues
        let eigenvalues = system_matrix.complex_eigenvalues();
        
        // Find rightmost eigenvalue (determines stability)
        let max_real_part = eigenvalues.iter()
            .map(|lambda| lambda.re)
            .fold(f64::NEG_INFINITY, f64::max);
        
        // Stability margin is how far we are from instability
        Ok(-max_real_part)
    }

    /// Perform inverse Laplace transform for time-domain results
    pub fn inverse_transform(&self, laplace_result: &DVector<Complex<f64>>) -> Result<DVector<f64>> {
        self.inverse_transformer.transform(laplace_result)
    }

    /// Get system transfer function
    pub fn get_system_transfer_function(&self) -> Result<TransferFunction> {
        if self.laplace_system.system_matrix.is_empty() {
            return Err(MogadishuError::math("No system matrix available"));
        }

        // For SISO system: H(s) = C(sI - A)^(-1)B + D
        // Simplified to characteristic polynomial approach
        let system_matrix = &self.laplace_system.system_matrix;
        let char_poly = self.calculate_characteristic_polynomial(system_matrix)?;
        
        Ok(TransferFunction {
            numerator: vec![1.0], // Simplified
            denominator: char_poly,
            dc_gain: 1.0,
            frequency_response: None,
        })
    }

    /// Calculate characteristic polynomial of system matrix
    fn calculate_characteristic_polynomial(&self, matrix: &DMatrix<Complex<f64>>) -> Result<Vec<f64>> {
        // For small systems, calculate det(sI - A) directly
        // For larger systems, use approximations
        
        let n = matrix.nrows();
        if n == 0 {
            return Ok(vec![1.0]);
        }
        
        // Simplified approach: use trace and determinant for small systems
        if n == 1 {
            let a11 = matrix[(0, 0)].re;
            Ok(vec![1.0, -a11]) // s - a11
        } else if n == 2 {
            let trace = (matrix[(0, 0)] + matrix[(1, 1)]).re;
            let det = (matrix[(0, 0)] * matrix[(1, 1)] - matrix[(0, 1)] * matrix[(1, 0)]).re;
            Ok(vec![1.0, -trace, det]) // s² - trace*s + det
        } else {
            // For larger systems, use approximate coefficients
            let trace = (0..n).map(|i| matrix[(i, i)].re).sum::<f64>();
            let mut coeffs = vec![1.0];
            coeffs.push(-trace);
            for _ in 2..=n {
                coeffs.push(0.0); // Simplified - would need full calculation
            }
            Ok(coeffs)
        }
    }
}

impl InverseLaplaceTransformer {
    /// Create new inverse Laplace transformer
    pub fn new() -> Self {
        Self {
            numerical_methods: vec![
                InverseLaplaceMethod::Stehfest,
                InverseLaplaceMethod::Talbot,
                InverseLaplaceMethod::SEntropyEnhanced { 
                    miracle_level: 0.0,
                    impossibility_tolerance: 0.01,
                },
            ],
            accuracy_tolerance: 1e-6,
            max_computation_time: 1.0,
        }
    }

    /// Transform from Laplace domain to time domain
    pub fn transform(&self, laplace_result: &DVector<Complex<f64>>) -> Result<DVector<f64>> {
        // Start with Stehfest algorithm (most reliable)
        if let Ok(result) = self.stehfest_transform(laplace_result) {
            return Ok(result);
        }
        
        // Fall back to Talbot algorithm
        if let Ok(result) = self.talbot_transform(laplace_result) {
            return Ok(result);
        }
        
        // If both fail, use S-entropy enhanced method
        self.s_entropy_enhanced_transform(laplace_result)
    }

    /// Stehfest algorithm implementation
    fn stehfest_transform(&self, laplace_result: &DVector<Complex<f64>>) -> Result<DVector<f64>> {
        let n = laplace_result.len();
        let mut time_result = DVector::zeros(n);
        
        // Simplified Stehfest implementation
        let ln2 = 2.0_f64.ln();
        let stehfest_order = 12; // Standard order
        
        for i in 0..n {
            let t = (i + 1) as f64 * 0.1; // Time step
            let mut sum = 0.0;
            
            for k in 1..=stehfest_order {
                let s = k as f64 * ln2 / t;
                let weight = self.stehfest_weight(k, stehfest_order);
                
                // Evaluate F(s) - simplified to use magnitude
                let f_s = laplace_result.get(k.min(n) - 1)
                    .map(|x| x.norm())
                    .unwrap_or(0.0);
                
                sum += weight * f_s;
            }
            
            time_result[i] = ln2 / t * sum;
        }
        
        Ok(time_result)
    }

    /// Calculate Stehfest weights
    fn stehfest_weight(&self, k: usize, n: usize) -> f64 {
        let mut weight = 0.0;
        let k_min = ((k + 1) / 2).max(1);
        let k_max = k.min(n / 2);
        
        for i in k_min..=k_max {
            let mut term = (i as f64).powi(n as i32 / 2);
            
            // Factorial calculations
            let mut fact = 1.0;
            for j in 1..=i {
                fact *= j as f64;
            }
            term /= fact;
            
            // Additional factorial terms (simplified)
            fact = 1.0;
            for j in 1..=(k - i) {
                fact *= j as f64;
            }
            term /= fact;
            
            weight += term;
        }
        
        weight * (-1.0_f64).powi(k as i32 + n as i32 / 2)
    }

    /// Talbot algorithm implementation  
    fn talbot_transform(&self, laplace_result: &DVector<Complex<f64>>) -> Result<DVector<f64>> {
        let n = laplace_result.len();
        let mut time_result = DVector::zeros(n);
        
        // Simplified Talbot implementation
        for i in 0..n {
            let t = (i + 1) as f64 * 0.1;
            
            // Use first value as approximation
            let f_value = laplace_result.get(0)
                .map(|x| x.re)
                .unwrap_or(0.0);
            
            time_result[i] = f_value / t; // Simplified inverse
        }
        
        Ok(time_result)
    }

    /// S-entropy enhanced inverse transform (can handle impossible solutions)
    fn s_entropy_enhanced_transform(&self, laplace_result: &DVector<Complex<f64>>) -> Result<DVector<f64>> {
        let n = laplace_result.len();
        let mut time_result = DVector::zeros(n);
        
        // Enhanced method can handle complex cases that others cannot
        for i in 0..n {
            let complex_val = laplace_result.get(i).unwrap_or(&Complex::new(0.0, 0.0));
            
            // Use both real and imaginary parts with S-entropy weighting
            let enhanced_value = complex_val.re + 0.1 * complex_val.im; // Miracle enhancement
            time_result[i] = enhanced_value;
        }
        
        Ok(time_result)
    }
}

impl Default for LaplaceSystem {
    fn default() -> Self {
        Self {
            system_matrix: DMatrix::zeros(0, 0),
            input_matrix: DMatrix::zeros(0, 0),
            output_matrix: DMatrix::zeros(0, 0),
            feedthrough_matrix: DMatrix::zeros(0, 0),
            compressed_representation: CompressedSystem::default(),
        }
    }
}

impl Default for CompressedSystem {
    fn default() -> Self {
        Self {
            svd_u: DMatrix::zeros(0, 0),
            svd_s: DVector::zeros(0),
            svd_v: DMatrix::zeros(0, 0),
            reduced_order: 0,
            compression_ratio: 1.0,
            information_preserved: 100.0,
        }
    }
}

impl Default for CompressionStatistics {
    fn default() -> Self {
        Self {
            original_order: 0,
            compressed_order: 0,
            compression_ratio: 1.0,
            information_loss: 0.0,
            computational_speedup: 1.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_laplace_analyzer_creation() {
        let analyzer = LaplaceAnalyzer::new();
        assert_eq!(analyzer.time_domain_equations.len(), 0);
        assert_eq!(analyzer.compression_stats.compression_ratio, 1.0);
    }
    
    #[test]
    fn test_differential_equation_creation() {
        let analyzer = LaplaceAnalyzer::new();
        
        let component = CircuitComponent::BufferRC {
            id: "test_rc".to_string(),
            resistance: 2.0,
            capacitance: 0.5,
            time_constant: 1.0,
        };
        
        let eq = analyzer.create_component_differential_equation(&component);
        assert!(eq.is_ok());
        
        let equation = eq.unwrap();
        assert_eq!(equation.variable, "voltage_test_rc");
        assert_eq!(equation.order, 1);
        assert_eq!(equation.coefficients[0], 1.0); // RC time constant
    }
    
    #[test]
    fn test_miraculous_equation_creation() {
        let analyzer = LaplaceAnalyzer::new();
        
        let miracle_component = MiraculousComponent::NegativeResistor {
            id: "test_neg_r".to_string(),
            negative_resistance: -1.0,
            miracle_level: 0.5,
        };
        
        let eq = analyzer.create_miraculous_differential_equation(&miracle_component);
        assert!(eq.is_ok());
        
        let equation = eq.unwrap();
        assert_eq!(equation.variable, "voltage_test_neg_r");
        assert_eq!(equation.rhs_terms.len(), 2); // Linear + Miraculous terms
    }
    
    #[test]
    fn test_compression_calculation() {
        let analyzer = LaplaceAnalyzer::new();
        let test_matrix = DMatrix::from_row_slice(3, 3, &[
            1.0, 0.1, 0.01,
            0.1, 1.0, 0.1,
            0.01, 0.1, 1.0,
        ]);
        
        let compressed = analyzer.create_compressed_representation(&test_matrix);
        assert!(compressed.is_ok());
        
        let result = compressed.unwrap();
        assert!(result.compression_ratio >= 1.0);
        assert!(result.information_preserved > 0.0);
        assert!(result.information_preserved <= 100.0);
    }
    
    #[test]
    fn test_inverse_transform() {
        let transformer = InverseLaplaceTransformer::new();
        let laplace_data = DVector::from_vec(vec![
            Complex::new(1.0, 0.0),
            Complex::new(0.5, 0.1),
            Complex::new(0.2, 0.0),
        ]);
        
        let result = transformer.transform(&laplace_data);
        assert!(result.is_ok());
        
        let time_data = result.unwrap();
        assert_eq!(time_data.len(), 3);
        assert!(time_data.iter().all(|&x| x.is_finite()));
    }
}
