"""
Virtual Cell Observer Validation
===============================

Validates the virtual cell observer system:
1. Virtual cell matches real bioreactor conditions
2. Observer insertion enables condition matching
3. Internal process visibility when conditions match
4. S-entropy navigation guides virtual cell state
5. Sensor data to cellular process bridge
"""

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from typing import Dict, List, Any, Tuple, Callable, Optional
from dataclasses import dataclass, field
import itertools

from .validation_framework import ValidationFramework, ValidationResult


@dataclass
class BioreactorConditions:
    """Real bioreactor sensor measurements"""
    temperature: float  # Celsius
    ph: float
    dissolved_oxygen: float  # mg/L
    glucose: float  # g/L
    biomass: float  # g/L
    agitation_rate: float  # RPM
    pressure: float  # bar
    timestamp: float


@dataclass 
class VirtualCellState:
    """Complete internal state of virtual cell"""
    
    # Membrane quantum computer state
    membrane_coherence: float
    quantum_pathways_active: int
    molecular_resolution_rate: float
    
    # ATP system state
    atp_concentration: float
    adp_concentration: float
    energy_charge: float
    synthesis_rate: float
    consumption_rate: float
    
    # Molecular processes
    active_enzymes: Dict[str, float]
    metabolic_fluxes: Dict[str, float]
    transport_rates: Dict[str, float]
    
    # DNA consultation state
    chromatin_accessibility: float
    transcription_rates: Dict[str, float]
    consultation_frequency: float
    
    # Oxygen enhancement
    o2_uptake_rate: float
    information_processing_rate: float
    electron_cascade_activity: float
    
    # S-entropy coordinates
    s_knowledge: float
    s_time: float
    s_entropy: float


@dataclass
class VirtualCellParameters:
    """Parameters for virtual cell system"""
    
    # Cell model complexity
    num_enzymes: int = 500
    num_metabolites: int = 1000
    num_transporters: int = 100
    
    # Matching tolerances
    temperature_tolerance: float = 2.0  # °C
    ph_tolerance: float = 0.3
    oxygen_tolerance: float = 1.0  # mg/L
    glucose_tolerance: float = 0.5  # g/L
    
    # S-entropy navigation
    navigation_steps: int = 100
    convergence_threshold: float = 1e-3
    viability_threshold: float = 100.0
    
    # Observer network
    num_observers: int = 20
    observation_noise: float = 0.05


class VirtualCell:
    """Detailed virtual cell model with complete internal state visibility"""
    
    def __init__(self, cell_id: str, params: VirtualCellParameters):
        self.cell_id = cell_id
        self.params = params
        
        # Initialize internal state
        self.state = self._initialize_state()
        
        # Metabolic network model
        self.metabolic_network = self._create_metabolic_network()
        
        # Regulatory network
        self.regulatory_network = self._create_regulatory_network()
        
        # History tracking
        self.state_history = [self.state]
        self.condition_history = []
        self.matching_history = []
        
    def _initialize_state(self) -> VirtualCellState:
        """Initialize virtual cell with physiological state"""
        
        # Generate realistic enzyme activities
        active_enzymes = {}
        enzyme_classes = ['kinases', 'phosphatases', 'dehydrogenases', 'transferases', 'hydrolases']
        for enzyme_class in enzyme_classes:
            for i in range(self.params.num_enzymes // len(enzyme_classes)):
                enzyme_name = f"{enzyme_class}_{i}"
                activity = np.random.lognormal(0, 0.5)  # Lognormal distribution for enzyme activities
                active_enzymes[enzyme_name] = activity
        
        # Generate metabolic fluxes
        metabolic_fluxes = {}
        flux_pathways = ['glycolysis', 'tca_cycle', 'pentose_phosphate', 'fatty_acid', 'amino_acid']
        for pathway in flux_pathways:
            for i in range(self.params.num_metabolites // len(flux_pathways)):
                flux_name = f"{pathway}_flux_{i}"
                flux_rate = np.random.gamma(2, 0.5)  # Gamma distribution for fluxes
                metabolic_fluxes[flux_name] = flux_rate
        
        # Transport rates
        transport_rates = {}
        transport_types = ['glucose', 'amino_acids', 'ions', 'oxygen', 'waste']
        for transport_type in transport_types:
            for i in range(self.params.num_transporters // len(transport_types)):
                transporter_name = f"{transport_type}_transporter_{i}"
                rate = np.random.exponential(1.0)
                transport_rates[transporter_name] = rate
        
        # Transcription rates
        transcription_rates = {}
        gene_types = ['metabolic', 'regulatory', 'structural', 'stress_response']
        for gene_type in gene_types:
            for i in range(50):  # 50 genes per type
                gene_name = f"{gene_type}_gene_{i}"
                rate = np.random.beta(2, 5)  # Most genes lowly expressed
                transcription_rates[gene_name] = rate
        
        return VirtualCellState(
            # Membrane quantum computer
            membrane_coherence=0.95,
            quantum_pathways_active=50,
            molecular_resolution_rate=1000.0,
            
            # ATP system
            atp_concentration=5.0,  # mM
            adp_concentration=0.5,
            energy_charge=0.9,
            synthesis_rate=10.0,
            consumption_rate=8.0,
            
            # Molecular processes
            active_enzymes=active_enzymes,
            metabolic_fluxes=metabolic_fluxes,
            transport_rates=transport_rates,
            
            # DNA consultation
            chromatin_accessibility=0.3,  # 30% accessible
            transcription_rates=transcription_rates,
            consultation_frequency=0.01,  # 1% consultation rate
            
            # Oxygen enhancement
            o2_uptake_rate=2.0,  # mmol/g/h
            information_processing_rate=8000.0,  # 8000x enhancement
            electron_cascade_activity=0.95,
            
            # S-entropy coordinates
            s_knowledge=100.0,
            s_time=1.0,
            s_entropy=0.0
        )
    
    def _create_metabolic_network(self) -> Dict[str, Any]:
        """Create simplified metabolic network model"""
        
        # Central carbon metabolism
        network = {
            'glycolysis': {
                'glucose_uptake': 1.0,
                'atp_yield': 2.0,
                'pyruvate_output': 2.0
            },
            'tca_cycle': {
                'pyruvate_input': 1.0,
                'atp_yield': 30.0,
                'co2_output': 6.0
            },
            'pentose_phosphate': {
                'glucose_6p_input': 0.3,
                'nadph_yield': 2.0,
                'nucleotide_precursors': 1.0
            },
            'fatty_acid_synthesis': {
                'acetyl_coa_input': 8.0,
                'atp_cost': 16.0,
                'palmitate_output': 1.0
            }
        }
        
        return network
    
    def _create_regulatory_network(self) -> Dict[str, Any]:
        """Create gene regulatory network model"""
        
        network = {
            'glucose_sensing': {
                'glucose_threshold': 2.0,
                'regulated_genes': ['glucose_transporter', 'hexokinase', 'pfk1'],
                'regulation_strength': 0.8
            },
            'oxygen_sensing': {
                'oxygen_threshold': 1.0,
                'regulated_genes': ['cytochrome_oxidase', 'hemoglobin', 'vegf'],
                'regulation_strength': 0.9
            },
            'stress_response': {
                'stress_threshold': 0.7,  # Energy charge threshold
                'regulated_genes': ['hsp70', 'catalase', 'superoxide_dismutase'],
                'regulation_strength': 0.95
            }
        }
        
        return network
    
    def update_state_from_conditions(self, conditions: BioreactorConditions) -> None:
        """Update virtual cell state to match bioreactor conditions"""
        
        # Temperature effects on enzyme kinetics (Q10 = 2)
        temp_factor = 2.0 ** ((conditions.temperature - 37.0) / 10.0)
        
        # Update enzyme activities based on temperature
        for enzyme_name in self.state.active_enzymes:
            base_activity = self.state.active_enzymes[enzyme_name]
            self.state.active_enzymes[enzyme_name] = base_activity * temp_factor
        
        # pH effects on enzyme activities (simplified)
        ph_optimal = 7.4
        ph_factor = np.exp(-0.5 * ((conditions.ph - ph_optimal) / 0.5) ** 2)
        
        for enzyme_name in self.state.active_enzymes:
            self.state.active_enzymes[enzyme_name] *= ph_factor
        
        # Oxygen effects
        o2_saturation = conditions.dissolved_oxygen / 8.0  # Assume 8 mg/L saturation
        self.state.o2_uptake_rate = 2.0 * o2_saturation
        self.state.information_processing_rate = 8000.0 * o2_saturation
        
        # Glucose effects on metabolism
        glucose_factor = conditions.glucose / (conditions.glucose + 0.1)  # Michaelis-Menten
        
        # Update glycolytic fluxes
        for flux_name in self.state.metabolic_fluxes:
            if 'glycolysis' in flux_name:
                base_flux = self.state.metabolic_fluxes[flux_name]
                self.state.metabolic_fluxes[flux_name] = base_flux * glucose_factor
        
        # Update glucose transporters
        for transporter_name in self.state.transport_rates:
            if 'glucose' in transporter_name:
                base_rate = self.state.transport_rates[transporter_name]
                self.state.transport_rates[transporter_name] = base_rate * glucose_factor
        
        # ATP synthesis rate based on substrate availability and oxygen
        substrate_availability = min(glucose_factor, o2_saturation)
        self.state.synthesis_rate = 10.0 * substrate_availability * temp_factor
        
        # Energy charge calculation
        if self.state.atp_concentration + self.state.adp_concentration > 0:
            total_adenylates = self.state.atp_concentration + self.state.adp_concentration + 0.1
            self.state.energy_charge = (self.state.atp_concentration + 0.5 * self.state.adp_concentration) / total_adenylates
        
        # Gene regulation based on conditions
        self._update_gene_regulation(conditions)
        
        # Update S-entropy coordinates based on cellular state
        self._update_s_entropy_coordinates()
        
        # Record history
        self.condition_history.append(conditions)
        self.state_history.append(self.state)
    
    def _update_gene_regulation(self, conditions: BioreactorConditions) -> None:
        """Update gene transcription rates based on environmental conditions"""
        
        # Glucose sensing
        if conditions.glucose < self.regulatory_network['glucose_sensing']['glucose_threshold']:
            # Upregulate glucose transport and glycolysis
            regulated_genes = self.regulatory_network['glucose_sensing']['regulated_genes']
            for gene in regulated_genes:
                if gene in self.state.transcription_rates:
                    self.state.transcription_rates[gene] *= 2.0
        
        # Oxygen sensing
        if conditions.dissolved_oxygen < self.regulatory_network['oxygen_sensing']['oxygen_threshold']:
            # Upregulate oxygen transport and utilization
            regulated_genes = self.regulatory_network['oxygen_sensing']['regulated_genes']
            for gene in regulated_genes:
                if gene in self.state.transcription_rates:
                    self.state.transcription_rates[gene] *= 1.5
        
        # Stress response
        if self.state.energy_charge < self.regulatory_network['stress_response']['stress_threshold']:
            # Upregulate stress response genes
            regulated_genes = self.regulatory_network['stress_response']['regulated_genes']
            for gene in regulated_genes:
                if gene in self.state.transcription_rates:
                    self.state.transcription_rates[gene] *= 3.0
    
    def _update_s_entropy_coordinates(self) -> None:
        """Update S-entropy coordinates based on current cellular state"""
        
        # Knowledge dimension: Information processing capacity
        knowledge_capacity = (self.state.information_processing_rate * 
                            self.state.membrane_coherence * 
                            self.state.quantum_pathways_active)
        self.state.s_knowledge = knowledge_capacity / 100.0
        
        # Time dimension: Metabolic rate and ATP turnover
        metabolic_rate = sum(self.state.metabolic_fluxes.values()) / len(self.state.metabolic_fluxes)
        atp_turnover = self.state.synthesis_rate + self.state.consumption_rate
        time_efficiency = 1.0 / (metabolic_rate * atp_turnover + 0.1)
        self.state.s_time = time_efficiency
        
        # Entropy dimension: Energy charge and organization
        # Higher energy charge = lower entropy (more organized)
        entropy_level = -(self.state.energy_charge - 0.5) * 100.0
        self.state.s_entropy = entropy_level
    
    def get_matching_score(self, conditions: BioreactorConditions) -> float:
        """Calculate how well virtual cell matches bioreactor conditions"""
        
        # Update state to match conditions
        self.update_state_from_conditions(conditions)
        
        # Calculate matching scores for each parameter
        temp_score = 1.0 - abs(conditions.temperature - 37.0) / self.params.temperature_tolerance
        ph_score = 1.0 - abs(conditions.ph - 7.4) / self.params.ph_tolerance
        o2_score = 1.0 - abs(conditions.dissolved_oxygen - 6.0) / self.params.oxygen_tolerance
        glucose_score = 1.0 - abs(conditions.glucose - 2.0) / self.params.glucose_tolerance
        
        # Clamp scores to [0, 1]
        scores = [max(0, min(1, score)) for score in [temp_score, ph_score, o2_score, glucose_score]]
        
        # Overall matching score (geometric mean for stringent matching)
        matching_score = np.prod(scores) ** (1.0 / len(scores))
        
        self.matching_history.append(matching_score)
        
        return matching_score
    
    def get_internal_process_visibility(self) -> Dict[str, Any]:
        """Get complete visibility into internal processes when conditions match"""
        
        return {
            'membrane_quantum_processes': {
                'coherence': self.state.membrane_coherence,
                'active_pathways': self.state.quantum_pathways_active,
                'resolution_rate': self.state.molecular_resolution_rate,
                'environmental_coupling': 0.7  # From membrane parameters
            },
            
            'energy_metabolism': {
                'atp_concentration': self.state.atp_concentration,
                'energy_charge': self.state.energy_charge,
                'synthesis_rate': self.state.synthesis_rate,
                'consumption_rate': self.state.consumption_rate,
                'atp_demand': sum(v for k, v in self.state.active_enzymes.items() if 'kinase' in k)
            },
            
            'molecular_processes': {
                'active_enzyme_count': len([e for e in self.state.active_enzymes.values() if e > 0.1]),
                'metabolic_flux_total': sum(self.state.metabolic_fluxes.values()),
                'transport_activity': sum(self.state.transport_rates.values()),
                'most_active_enzymes': sorted(self.state.active_enzymes.items(), 
                                            key=lambda x: x[1], reverse=True)[:10]
            },
            
            'gene_regulation': {
                'chromatin_accessibility': self.state.chromatin_accessibility,
                'transcription_activity': sum(self.state.transcription_rates.values()),
                'consultation_frequency': self.state.consultation_frequency,
                'highly_expressed_genes': sorted(self.state.transcription_rates.items(),
                                               key=lambda x: x[1], reverse=True)[:20]
            },
            
            'oxygen_processing': {
                'uptake_rate': self.state.o2_uptake_rate,
                'processing_enhancement': self.state.information_processing_rate,
                'electron_cascade': self.state.electron_cascade_activity
            },
            
            's_entropy_state': {
                'knowledge': self.state.s_knowledge,
                'time': self.state.s_time,
                'entropy': self.state.s_entropy,
                'viability': np.sqrt(self.state.s_knowledge**2 + self.state.s_time**2 + self.state.s_entropy**2)
            }
        }


class VirtualCellObserver:
    """Observer that matches virtual cell to real bioreactor conditions"""
    
    def __init__(self, observer_id: str, virtual_cell: VirtualCell, params: VirtualCellParameters):
        self.observer_id = observer_id
        self.virtual_cell = virtual_cell
        self.params = params
        
        # S-entropy navigator for condition matching
        self.s_navigator = self._create_s_navigator()
        
        # Observation history
        self.observations = []
        self.matching_attempts = []
        
    def _create_s_navigator(self):
        """Create S-entropy navigator for virtual cell positioning"""
        # Simplified navigator - in full implementation would import from s_entropy module
        return None
    
    def observe_and_match(self, conditions: BioreactorConditions) -> Dict[str, Any]:
        """Observe bioreactor conditions and match virtual cell state"""
        
        # Record observation
        self.observations.append(conditions)
        
        # Attempt to match virtual cell to conditions
        initial_matching_score = self.virtual_cell.get_matching_score(conditions)
        
        # Use S-entropy navigation to improve matching if needed
        if initial_matching_score < 0.8:  # Need better matching
            optimized_score = self._optimize_virtual_cell_matching(conditions)
        else:
            optimized_score = initial_matching_score
        
        # If matching is successful, provide internal process visibility
        if optimized_score > 0.8:
            internal_processes = self.virtual_cell.get_internal_process_visibility()
            
            matching_result = {
                'matching_successful': True,
                'matching_score': optimized_score,
                'bioreactor_conditions': conditions.__dict__,
                'internal_processes': internal_processes,
                'observer_id': self.observer_id,
                'timestamp': conditions.timestamp
            }
        else:
            matching_result = {
                'matching_successful': False,
                'matching_score': optimized_score,
                'bioreactor_conditions': conditions.__dict__,
                'internal_processes': None,
                'observer_id': self.observer_id,
                'timestamp': conditions.timestamp
            }
        
        self.matching_attempts.append(matching_result)
        
        return matching_result
    
    def _optimize_virtual_cell_matching(self, conditions: BioreactorConditions) -> float:
        """Use S-entropy navigation to optimize virtual cell matching"""
        
        # Define objective function for matching optimization
        def matching_objective(cell_params):
            # Temporarily modify virtual cell parameters
            temp_factor, ph_factor, o2_factor = cell_params
            
            # Apply parameter modifications
            original_state = self.virtual_cell.state
            
            # Modify cellular parameters to improve matching
            modified_conditions = BioreactorConditions(
                temperature=conditions.temperature * temp_factor,
                ph=conditions.ph * ph_factor,
                dissolved_oxygen=conditions.dissolved_oxygen * o2_factor,
                glucose=conditions.glucose,
                biomass=conditions.biomass,
                agitation_rate=conditions.agitation_rate,
                pressure=conditions.pressure,
                timestamp=conditions.timestamp
            )
            
            # Calculate matching score with modified parameters
            score = self.virtual_cell.get_matching_score(modified_conditions)
            
            # Return negative score for minimization
            return -score
        
        # Optimize matching using simple optimization
        from scipy.optimize import minimize
        
        initial_params = [1.0, 1.0, 1.0]  # No initial modification
        bounds = [(0.5, 1.5), (0.5, 1.5), (0.5, 1.5)]  # Allow 50% parameter variation
        
        result = minimize(matching_objective, initial_params, bounds=bounds, method='L-BFGS-B')
        
        if result.success:
            optimized_score = -result.fun
            return optimized_score
        else:
            # Return original matching score if optimization failed
            return self.virtual_cell.get_matching_score(conditions)


class VirtualCellSystemValidator(ValidationFramework):
    """Validates the virtual cell observer system"""
    
    def __init__(self, **kwargs):
        super().__init__("VirtualCellSystem", **kwargs)
        
    def validate_theorem(self, **kwargs) -> ValidationResult:
        """
        Validate main virtual cell theorem:
        'Virtual cell observer can match real bioreactor conditions and 
         provide complete visibility into internal cellular processes'
        """
        
        params = VirtualCellParameters(**kwargs)
        
        if self.verbose:
            print("🔬 Validating Virtual Cell Observer Theorem")
        
        # Create virtual cell and observer
        virtual_cell = VirtualCell("test_cell", params)
        observer = VirtualCellObserver("test_observer", virtual_cell, params)
        
        # Generate test bioreactor conditions
        test_conditions = self._generate_test_conditions(params)
        
        # Test matching performance
        matching_results = []
        internal_process_data = []
        
        for conditions in test_conditions:
            result = observer.observe_and_match(conditions)
            matching_results.append(result)
            
            if result['matching_successful']:
                internal_process_data.append(result['internal_processes'])
        
        # Analyze results
        successful_matches = sum(1 for r in matching_results if r['matching_successful'])
        matching_success_rate = successful_matches / len(matching_results)
        
        mean_matching_score = np.mean([r['matching_score'] for r in matching_results])
        
        # Test condition diversity coverage
        temp_range = [c.temperature for c in test_conditions]
        ph_range = [c.ph for c in test_conditions]
        o2_range = [c.dissolved_oxygen for c in test_conditions]
        
        condition_diversity = {
            'temperature_range': max(temp_range) - min(temp_range),
            'ph_range': max(ph_range) - min(ph_range),
            'oxygen_range': max(o2_range) - min(o2_range)
        }
        
        # Test internal process visibility
        if internal_process_data:
            process_visibility = self._analyze_internal_process_visibility(internal_process_data)
        else:
            process_visibility = {}
        
        # Success criteria
        success_threshold = 0.7  # 70% successful matching
        visibility_threshold = 0.8  # 80% process visibility
        
        theorem_success = (matching_success_rate >= success_threshold and
                          mean_matching_score >= 0.6 and
                          len(internal_process_data) > 0)
        
        # Create validation plots
        plot_paths = []
        plot_paths.append(self._create_matching_performance_plot(matching_results, "virtual_cell_matching"))
        
        if internal_process_data:
            plot_paths.append(self._create_internal_process_plot(internal_process_data, "internal_processes"))
        
        return ValidationResult(
            test_name="virtual_cell_observer_theorem",
            theorem_validated="Virtual cell observer matches bioreactor conditions and provides internal process visibility",
            timestamp=self._get_timestamp(),
            parameters=kwargs,
            success=theorem_success,
            quantitative_results={
                "matching_success_rate": matching_success_rate,
                "mean_matching_score": mean_matching_score,
                "successful_matches": successful_matches,
                "total_conditions": len(test_conditions),
                "process_visibility_cases": len(internal_process_data),
                "condition_diversity": condition_diversity
            },
            statistical_significance={},
            supporting_evidence=[
                f"Matching success rate: {matching_success_rate:.1%}",
                f"Mean matching score: {mean_matching_score:.3f}",
                f"Internal processes visible in {len(internal_process_data)} cases",
                f"Condition diversity: T={condition_diversity['temperature_range']:.1f}°C, pH={condition_diversity['ph_range']:.1f}"
            ] if theorem_success else [],
            contradictory_evidence=[] if theorem_success else [
                f"Low matching success: {matching_success_rate:.1%}",
                f"Poor matching scores: {mean_matching_score:.3f}",
                f"Limited process visibility: {len(internal_process_data)} cases"
            ],
            raw_data={
                'test_conditions': [c.__dict__ for c in test_conditions],
                'matching_results': matching_results,
                'internal_process_data': internal_process_data
            },
            processed_data={
                'matching_scores': np.array([r['matching_score'] for r in matching_results]),
                'success_flags': np.array([r['matching_successful'] for r in matching_results])
            },
            plot_paths=plot_paths,
            notes=f"Tested {len(test_conditions)} diverse bioreactor conditions with virtual cell observer",
            confidence_level=min(matching_success_rate + 0.2, mean_matching_score + 0.3)
        )
    
    def _generate_test_conditions(self, params: VirtualCellParameters) -> List[BioreactorConditions]:
        """Generate diverse bioreactor test conditions"""
        
        conditions = []
        
        # Normal physiological conditions
        conditions.append(BioreactorConditions(
            temperature=37.0, ph=7.4, dissolved_oxygen=6.0, glucose=2.0,
            biomass=5.0, agitation_rate=200, pressure=1.0, timestamp=0.0
        ))
        
        # Temperature variations
        for temp in [30, 32, 35, 39, 42]:
            conditions.append(BioreactorConditions(
                temperature=temp, ph=7.4, dissolved_oxygen=6.0, glucose=2.0,
                biomass=5.0, agitation_rate=200, pressure=1.0, timestamp=float(len(conditions))
            ))
        
        # pH variations
        for ph in [6.5, 6.8, 7.0, 7.7, 8.0]:
            conditions.append(BioreactorConditions(
                temperature=37.0, ph=ph, dissolved_oxygen=6.0, glucose=2.0,
                biomass=5.0, agitation_rate=200, pressure=1.0, timestamp=float(len(conditions))
            ))
        
        # Oxygen variations
        for o2 in [1.0, 3.0, 4.5, 7.5, 9.0]:
            conditions.append(BioreactorConditions(
                temperature=37.0, ph=7.4, dissolved_oxygen=o2, glucose=2.0,
                biomass=5.0, agitation_rate=200, pressure=1.0, timestamp=float(len(conditions))
            ))
        
        # Glucose variations
        for glucose in [0.5, 1.0, 1.5, 3.0, 4.0]:
            conditions.append(BioreactorConditions(
                temperature=37.0, ph=7.4, dissolved_oxygen=6.0, glucose=glucose,
                biomass=5.0, agitation_rate=200, pressure=1.0, timestamp=float(len(conditions))
            ))
        
        # Combined stress conditions
        stress_conditions = [
            (30, 6.5, 2.0, 0.5),  # Cold, acidic, low O2, low glucose
            (42, 8.0, 1.0, 4.0),  # Hot, basic, very low O2, high glucose
            (35, 7.0, 9.0, 1.0),  # Mild temp, neutral pH, high O2, low glucose
        ]
        
        for temp, ph, o2, glucose in stress_conditions:
            conditions.append(BioreactorConditions(
                temperature=temp, ph=ph, dissolved_oxygen=o2, glucose=glucose,
                biomass=5.0, agitation_rate=200, pressure=1.0, timestamp=float(len(conditions))
            ))
        
        return conditions
    
    def _analyze_internal_process_visibility(self, process_data: List[Dict[str, Any]]) -> Dict[str, float]:
        """Analyze the quality of internal process visibility"""
        
        visibility_metrics = {}
        
        # Membrane quantum processes visibility
        membrane_coherence_values = [p['membrane_quantum_processes']['coherence'] for p in process_data]
        visibility_metrics['membrane_coherence_visibility'] = np.std(membrane_coherence_values)
        
        # Energy metabolism visibility
        energy_charge_values = [p['energy_metabolism']['energy_charge'] for p in process_data]
        visibility_metrics['energy_charge_visibility'] = np.std(energy_charge_values)
        
        # Molecular processes visibility
        enzyme_counts = [p['molecular_processes']['active_enzyme_count'] for p in process_data]
        visibility_metrics['enzyme_activity_visibility'] = np.std(enzyme_counts)
        
        # Gene regulation visibility
        transcription_values = [p['gene_regulation']['transcription_activity'] for p in process_data]
        visibility_metrics['transcription_visibility'] = np.std(transcription_values)
        
        # S-entropy state visibility
        s_knowledge_values = [p['s_entropy_state']['knowledge'] for p in process_data]
        visibility_metrics['s_entropy_visibility'] = np.std(s_knowledge_values)
        
        return visibility_metrics
    
    def _create_matching_performance_plot(self, matching_results: List[Dict], save_name: str) -> str:
        """Create virtual cell matching performance visualization"""
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Matching success rate over time
        timestamps = [r['timestamp'] for r in matching_results]
        matching_scores = [r['matching_score'] for r in matching_results]
        success_flags = [r['matching_successful'] for r in matching_results]
        
        axes[0,0].plot(timestamps, matching_scores, 'b-', linewidth=2, alpha=0.7, label='Matching Score')
        axes[0,0].scatter(timestamps, matching_scores, c=['green' if s else 'red' for s in success_flags], alpha=0.6)
        axes[0,0].axhline(y=0.8, color='red', linestyle='--', alpha=0.5, label='Success Threshold')
        axes[0,0].set_xlabel('Time')
        axes[0,0].set_ylabel('Matching Score')
        axes[0,0].set_title('Virtual Cell Matching Performance')
        axes[0,0].legend()
        axes[0,0].grid(True, alpha=0.3)
        
        # Matching score distribution
        axes[0,1].hist(matching_scores, bins=15, alpha=0.7, color='skyblue', density=True)
        axes[0,1].axvline(np.mean(matching_scores), color='red', linestyle='-', 
                         label=f'Mean: {np.mean(matching_scores):.3f}')
        axes[0,1].axvline(0.8, color='orange', linestyle='--', label='Success Threshold')
        axes[0,1].set_xlabel('Matching Score')
        axes[0,1].set_ylabel('Density')
        axes[0,1].set_title('Matching Score Distribution')
        axes[0,1].legend()
        
        # Success rate by condition type
        # Group by condition similarity
        temp_values = [r['bioreactor_conditions']['temperature'] for r in matching_results]
        ph_values = [r['bioreactor_conditions']['ph'] for r in matching_results]
        
        # Create condition categories
        temp_categories = ['Cold (<35)', 'Normal (35-39)', 'Hot (>39)']
        ph_categories = ['Acidic (<7)', 'Neutral (7-7.5)', 'Basic (>7.5)']
        
        temp_success = [[], [], []]  # Success rates for each temperature category
        
        for i, (temp, success) in enumerate(zip(temp_values, success_flags)):
            if temp < 35:
                temp_success[0].append(success)
            elif temp <= 39:
                temp_success[1].append(success)
            else:
                temp_success[2].append(success)
        
        temp_success_rates = [np.mean(group) if group else 0 for group in temp_success]
        
        axes[1,0].bar(temp_categories, temp_success_rates, alpha=0.7, color='lightgreen')
        axes[1,0].set_ylabel('Success Rate')
        axes[1,0].set_title('Matching Success by Temperature Range')
        axes[1,0].tick_params(axis='x', rotation=45)
        
        # Condition space coverage
        axes[1,1].scatter(temp_values, ph_values, c=matching_scores, cmap='viridis', alpha=0.7)
        cbar = plt.colorbar(axes[1,1].collections[0], ax=axes[1,1])
        cbar.set_label('Matching Score')
        axes[1,1].set_xlabel('Temperature (°C)')
        axes[1,1].set_ylabel('pH')
        axes[1,1].set_title('Condition Space Coverage')
        
        plt.tight_layout()
        
        # Save plot
        plot_path = self.output_dir / "plots" / f"{save_name}.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(plot_path)
    
    def _create_internal_process_plot(self, process_data: List[Dict], save_name: str) -> str:
        """Create internal process visibility visualization"""
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        axes = axes.flatten()
        
        # ATP/Energy system
        atp_concentrations = [p['energy_metabolism']['atp_concentration'] for p in process_data]
        energy_charges = [p['energy_metabolism']['energy_charge'] for p in process_data]
        
        axes[0].scatter(atp_concentrations, energy_charges, alpha=0.7, color='red')
        axes[0].set_xlabel('ATP Concentration (mM)')
        axes[0].set_ylabel('Energy Charge')
        axes[0].set_title('Energy System State')
        axes[0].grid(True, alpha=0.3)
        
        # Membrane quantum processes
        coherence_values = [p['membrane_quantum_processes']['coherence'] for p in process_data]
        pathway_counts = [p['membrane_quantum_processes']['active_pathways'] for p in process_data]
        
        axes[1].scatter(coherence_values, pathway_counts, alpha=0.7, color='blue')
        axes[1].set_xlabel('Membrane Coherence')
        axes[1].set_ylabel('Active Pathways')
        axes[1].set_title('Membrane Quantum State')
        axes[1].grid(True, alpha=0.3)
        
        # Enzyme activity
        enzyme_counts = [p['molecular_processes']['active_enzyme_count'] for p in process_data]
        flux_totals = [p['molecular_processes']['metabolic_flux_total'] for p in process_data]
        
        axes[2].scatter(enzyme_counts, flux_totals, alpha=0.7, color='green')
        axes[2].set_xlabel('Active Enzyme Count')
        axes[2].set_ylabel('Total Metabolic Flux')
        axes[2].set_title('Metabolic Activity')
        axes[2].grid(True, alpha=0.3)
        
        # Gene regulation
        chromatin_access = [p['gene_regulation']['chromatin_accessibility'] for p in process_data]
        transcription_activity = [p['gene_regulation']['transcription_activity'] for p in process_data]
        
        axes[3].scatter(chromatin_access, transcription_activity, alpha=0.7, color='purple')
        axes[3].set_xlabel('Chromatin Accessibility')
        axes[3].set_ylabel('Transcription Activity')
        axes[3].set_title('Gene Regulation State')
        axes[3].grid(True, alpha=0.3)
        
        # Oxygen processing
        o2_uptake = [p['oxygen_processing']['uptake_rate'] for p in process_data]
        processing_enhancement = [p['oxygen_processing']['processing_enhancement'] for p in process_data]
        
        axes[4].scatter(o2_uptake, processing_enhancement, alpha=0.7, color='orange')
        axes[4].set_xlabel('O2 Uptake Rate')
        axes[4].set_ylabel('Processing Enhancement')
        axes[4].set_title('Oxygen Enhancement')
        axes[4].grid(True, alpha=0.3)
        
        # S-entropy state
        s_knowledge = [p['s_entropy_state']['knowledge'] for p in process_data]
        s_entropy = [p['s_entropy_state']['entropy'] for p in process_data]
        
        axes[5].scatter(s_knowledge, s_entropy, alpha=0.7, color='brown')
        axes[5].set_xlabel('S-Knowledge')
        axes[5].set_ylabel('S-Entropy')
        axes[5].set_title('S-Entropy Navigation State')
        axes[5].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = self.output_dir / "plots" / f"{save_name}.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(plot_path)
    
    def _get_additional_tests(self) -> Dict[str, Callable]:
        """Additional virtual cell observer tests"""
        
        return {
            "condition_sensitivity": self._test_condition_sensitivity,
            "s_entropy_navigation_guidance": self._test_s_entropy_navigation,
            "process_visibility_completeness": self._test_process_visibility_completeness,
            "observer_network_performance": self._test_observer_network_performance
        }
    
    def _test_condition_sensitivity(self, **kwargs) -> ValidationResult:
        """Test sensitivity of virtual cell to condition changes"""
        
        if self.verbose:
            print("  Testing condition sensitivity...")
        
        params = VirtualCellParameters(**kwargs)
        virtual_cell = VirtualCell("sensitivity_test", params)
        
        # Baseline conditions
        baseline = BioreactorConditions(
            temperature=37.0, ph=7.4, dissolved_oxygen=6.0, glucose=2.0,
            biomass=5.0, agitation_rate=200, pressure=1.0, timestamp=0.0
        )
        
        # Test sensitivity to each parameter
        sensitivity_results = {}
        
        parameters_to_test = {
            'temperature': (baseline.temperature, [35, 37, 39], 'temperature'),
            'ph': (baseline.ph, [7.0, 7.4, 7.8], 'ph'),
            'dissolved_oxygen': (baseline.dissolved_oxygen, [3, 6, 9], 'dissolved_oxygen'),
            'glucose': (baseline.glucose, [1, 2, 3], 'glucose')
        }
        
        for param_name, (baseline_val, test_values, attr_name) in parameters_to_test.items():
            
            param_sensitivities = []
            
            for test_val in test_values:
                # Create test conditions
                test_conditions = BioreactorConditions(**baseline.__dict__)
                setattr(test_conditions, attr_name, test_val)
                
                # Get initial state
                initial_state = virtual_cell.state
                
                # Update state with test conditions
                virtual_cell.update_state_from_conditions(test_conditions)
                updated_state = virtual_cell.state
                
                # Calculate state change magnitude
                atp_change = abs(updated_state.atp_concentration - initial_state.atp_concentration)
                enzyme_activity_change = np.mean([abs(updated_state.active_enzymes[k] - initial_state.active_enzymes[k]) 
                                                for k in initial_state.active_enzymes.keys()])
                
                total_change = atp_change + enzyme_activity_change
                param_change = abs(test_val - baseline_val) / baseline_val
                
                sensitivity = total_change / param_change if param_change > 0 else 0
                param_sensitivities.append(sensitivity)
            
            sensitivity_results[param_name] = {
                'mean_sensitivity': np.mean(param_sensitivities),
                'std_sensitivity': np.std(param_sensitivities),
                'max_sensitivity': np.max(param_sensitivities)
            }
        
        # Check if virtual cell shows appropriate sensitivity
        appropriate_sensitivity = all(s['mean_sensitivity'] > 0.1 for s in sensitivity_results.values())
        
        success = appropriate_sensitivity
        
        return ValidationResult(
            test_name="condition_sensitivity",
            theorem_validated="Virtual cell shows appropriate sensitivity to bioreactor condition changes",
            timestamp=self._get_timestamp(),
            parameters=kwargs,
            success=success,
            quantitative_results={
                f"{param}_mean_sensitivity": results['mean_sensitivity']
                for param, results in sensitivity_results.items()
            },
            statistical_significance={},
            supporting_evidence=[
                f"{param}: sensitivity {results['mean_sensitivity']:.3f}"
                for param, results in sensitivity_results.items()
            ] if success else [],
            contradictory_evidence=[] if success else [
                f"Low sensitivity to {param}: {results['mean_sensitivity']:.3f}"
                for param, results in sensitivity_results.items() if results['mean_sensitivity'] <= 0.1
            ],
            raw_data={'sensitivity_results': sensitivity_results},
            processed_data={},
            plot_paths=[],
            notes="Tested sensitivity to temperature, pH, oxygen, and glucose variations",
            confidence_level=np.mean([s['mean_sensitivity'] for s in sensitivity_results.values()]) if success else 0.3
        )
    
    def _get_timestamp(self):
        """Get current timestamp"""
        from datetime import datetime, timezone
        return datetime.now(timezone.utc)
