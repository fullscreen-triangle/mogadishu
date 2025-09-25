#!/usr/bin/env python3
"""
S-Entropy Bioreactor Framework Demonstration
===========================================

Complete demonstration of the S-entropy framework for bioreactor modeling
integrating oscillatory substrate theory, cellular computational architectures,
and observer-process navigation.

Author: Kundai Farai Sachikonye
"""

import numpy as np
import matplotlib.pyplot as plt
import json
from datetime import datetime
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set up plotting
plt.style.use('default')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

class SEntropyNavigator:
    """S-entropy navigation system for tri-dimensional optimization"""
    
    def __init__(self, viability_threshold=100.0):
        self.viability_threshold = viability_threshold
        self.position = np.array([50.0, 1.0, 0.0])  # Initial S-space position
        self.history = []
        
    def calculate_s_distance(self, target):
        """Calculate S-distance to target position"""
        return np.linalg.norm(self.position - np.array(target))
    
    def is_viable(self, position=None):
        """Check if position is S-viable"""
        pos = position if position is not None else self.position
        return np.linalg.norm(pos) <= self.viability_threshold
    
    def navigate_step(self, target, step_size=0.5):
        """Single navigation step towards target"""
        if not isinstance(target, np.ndarray):
            target = np.array(target)
            
        direction = target - self.position
        distance = np.linalg.norm(direction)
        
        if distance > 0:
            # Normalize direction and take step
            step = direction / distance * min(step_size, distance)
            new_position = self.position + step
            
            # Check viability constraint
            if self.is_viable(new_position):
                self.position = new_position
                self.history.append(self.position.copy())
                return True
        
        return False
    
    def navigate_to_target(self, target, max_steps=100):
        """Navigate to target using S-entropy principles"""
        print(f"🧭 S-Entropy Navigation: {self.position} → {target}")
        
        steps = 0
        while steps < max_steps:
            if self.navigate_step(target):
                steps += 1
                distance = self.calculate_s_distance(target)
                if distance < 1.0:  # Close enough
                    break
            else:
                print("⚠️  Viability constraint violated")
                break
        
        final_distance = self.calculate_s_distance(target)
        success = final_distance < 5.0
        
        print(f"✅ Navigation {'successful' if success else 'incomplete'}: "
              f"final distance = {final_distance:.2f} in {steps} steps")
        
        return success, steps, final_distance


class VirtualCell:
    """Virtual cell with 99%/1% computational architecture"""
    
    def __init__(self, cell_id="cell_001"):
        self.cell_id = cell_id
        
        # ATP system (energy constraints)
        self.atp_concentration = 5.0  # mM
        self.energy_charge = 0.9
        self.atp_synthesis_rate = 10.0
        self.atp_consumption_rate = 8.0
        
        # Membrane quantum computer (99% resolution)
        self.membrane_coherence = 0.95
        self.quantum_pathways_active = 50
        self.resolution_accuracy = 0.99
        
        # DNA library (1% emergency consultation)
        self.dna_consultations = 0
        self.total_molecular_challenges = 0
        
        # Oxygen enhancement
        self.oxygen_concentration = 0.21  # Atmospheric
        self.information_processing_rate = 8000.0  # 8000x enhancement
        
        # Current state
        self.metabolic_activity = 1.0
        self.stress_level = 0.0
        
    def process_molecular_challenge(self, complexity=0.5):
        """Process unknown molecule using 99%/1% architecture"""
        self.total_molecular_challenges += 1
        
        # Enhanced resolution with oxygen
        effective_accuracy = self.resolution_accuracy * (1 + 0.1 * self.oxygen_concentration / 0.21)
        
        # Check if membrane can resolve (99% of cases)
        if np.random.random() < effective_accuracy and complexity < 0.95:
            # Membrane quantum computer resolves
            resolution_time = 0.01  # Fast quantum processing
            atp_cost = 0.1
            method = "membrane_quantum"
        else:
            # Requires DNA consultation (1% of cases)
            self.dna_consultations += 1
            resolution_time = 10.0  # Slow DNA access + transcription + translation
            atp_cost = 5.0  # High energy cost
            method = "dna_consultation"
        
        # ATP constraint check
        if self.atp_concentration >= atp_cost:
            self.atp_concentration -= atp_cost
            success = True
        else:
            success = False
            resolution_time = 0  # No processing if no energy
        
        return {
            'success': success,
            'method': method,
            'resolution_time': resolution_time,
            'atp_cost': atp_cost,
            'complexity': complexity
        }
    
    def update_from_conditions(self, temperature, ph, oxygen, nutrients):
        """Update virtual cell state based on bioreactor conditions"""
        
        # Temperature effects (Q10 = 2)
        temp_factor = 2.0 ** ((temperature - 37.0) / 10.0)
        self.metabolic_activity = 1.0 * temp_factor
        
        # pH effects
        ph_optimal = 7.4
        ph_stress = abs(ph - ph_optimal) / 2.0
        
        # Oxygen effects
        self.oxygen_concentration = oxygen / 8.0  # Normalize to saturation
        self.information_processing_rate = 8000.0 * self.oxygen_concentration
        
        # Stress calculation
        self.stress_level = ph_stress + max(0, abs(temperature - 37.0) - 5.0) / 10.0
        
        # ATP dynamics with constraints
        if self.stress_level > 0.5:
            self.atp_synthesis_rate = 5.0  # Reduced under stress
        else:
            self.atp_synthesis_rate = 10.0
        
        # Update ATP concentration (simplified dynamics)
        net_atp_rate = self.atp_synthesis_rate - self.atp_consumption_rate
        self.atp_concentration = max(0.1, self.atp_concentration + net_atp_rate * 0.1)
        
        # Update energy charge
        total_adenylates = self.atp_concentration + 0.5 + 0.05  # ATP + ADP + AMP
        self.energy_charge = (self.atp_concentration + 0.25) / total_adenylates
    
    def get_matching_score(self, target_temp=37.0, target_ph=7.4, target_o2=6.0):
        """Calculate how well virtual cell matches target conditions"""
        
        temp_match = 1.0 - abs(37.0 - target_temp) / 10.0  # Current temp vs target
        ph_match = 1.0 - abs(7.4 - target_ph) / 1.0
        o2_match = 1.0 - abs(self.oxygen_concentration - target_o2/8.0) / 1.0
        
        # Ensure non-negative matches
        temp_match = max(0, temp_match)
        ph_match = max(0, ph_match)
        o2_match = max(0, o2_match)
        
        # Overall matching (geometric mean for stringent matching)
        overall_match = (temp_match * ph_match * o2_match) ** (1/3)
        
        return overall_match, {
            'temperature_match': temp_match,
            'ph_match': ph_match,
            'oxygen_match': o2_match
        }
    
    def get_internal_visibility(self):
        """Get complete internal process visibility"""
        return {
            'atp_concentration': self.atp_concentration,
            'energy_charge': self.energy_charge,
            'membrane_coherence': self.membrane_coherence,
            'quantum_pathways_active': self.quantum_pathways_active,
            'information_processing_rate': self.information_processing_rate,
            'dna_consultation_rate': self.dna_consultations / max(1, self.total_molecular_challenges),
            'metabolic_activity': self.metabolic_activity,
            'stress_level': self.stress_level
        }


class EvidenceNetwork:
    """Bayesian evidence rectification network"""
    
    def __init__(self):
        self.evidence_items = []
        self.molecular_hypotheses = {}
        self.confidence_threshold = 0.9
        
    def add_evidence(self, evidence_type, quality, molecular_target, confidence):
        """Add new evidence to the network"""
        evidence = {
            'type': evidence_type,
            'quality': quality,
            'target': molecular_target, 
            'confidence': confidence,
            'timestamp': len(self.evidence_items)
        }
        self.evidence_items.append(evidence)
    
    def rectify_evidence(self, oxygen_enhancement=1.0):
        """Perform evidence rectification with oxygen enhancement"""
        
        if not self.evidence_items:
            return {'rectification_score': 1.0, 'contradictions_resolved': 0}
        
        # Enhanced processing with oxygen
        enhanced_processing = 1000.0 * oxygen_enhancement
        
        # Find contradictory evidence (simplified)
        contradictions = []
        for i, ev1 in enumerate(self.evidence_items):
            for j, ev2 in enumerate(self.evidence_items[i+1:], i+1):
                if (ev1['target'] != ev2['target'] and 
                    ev1['type'] == ev2['type'] and
                    abs(ev1['confidence'] - ev2['confidence']) < 0.1):
                    contradictions.append((i, j))
        
        # Resolve contradictions by evidence quality
        resolved = 0
        for i, j in contradictions:
            ev1, ev2 = self.evidence_items[i], self.evidence_items[j]
            if ev1['quality'] > ev2['quality']:
                ev2['confidence'] *= 0.5  # Downweight lower quality evidence
                resolved += 1
            elif ev2['quality'] > ev1['quality']:
                ev1['confidence'] *= 0.5
                resolved += 1
        
        rectification_score = resolved / len(contradictions) if contradictions else 1.0
        
        return {
            'rectification_score': rectification_score,
            'contradictions_found': len(contradictions),
            'contradictions_resolved': resolved,
            'processing_enhancement': enhanced_processing
        }
    
    def identify_molecule(self, evidence_list):
        """Identify molecule using Bayesian inference"""
        
        if not evidence_list:
            return None, 0.0
        
        # Calculate posterior probabilities (simplified Bayesian)
        molecules = list(set(ev['target'] for ev in evidence_list))
        posteriors = {}
        
        for molecule in molecules:
            # Prior probability (uniform)
            prior = 1.0 / len(molecules)
            
            # Likelihood from supporting evidence
            supporting_evidence = [ev for ev in evidence_list if ev['target'] == molecule]
            likelihood = 1.0
            for ev in supporting_evidence:
                likelihood *= ev['confidence'] * ev['quality']
            
            posteriors[molecule] = prior * likelihood
        
        # Normalize
        total_prob = sum(posteriors.values())
        if total_prob > 0:
            posteriors = {mol: prob/total_prob for mol, prob in posteriors.items()}
        
        if posteriors:
            best_molecule = max(posteriors.items(), key=lambda x: x[1])
            return best_molecule[0], best_molecule[1]
        
        return None, 0.0


def main():
    """Run the complete S-entropy bioreactor demonstration"""
    
    try:
        from bioreactor_demo import main as run_demo
        run_demo()
    except ImportError:
        print("❌ Could not import demonstration module.")
        print("Please ensure bioreactor_demo.py is in the same directory.")
    except Exception as e:
        print(f"❌ Demonstration failed: {str(e)}")


if __name__ == "__main__":
    main()
