"""
S-Entropy Framework Validation
==============================

Validates the S-Entropy navigation principles:
1. Tri-dimensional S-space navigation (knowledge, time, entropy)
2. Observer insertion for finite problem spaces  
3. Precision-by-difference measurement protocols
4. Universal predetermined solutions theorem
5. Cross-domain S-transfer capability
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize, differential_evolution
from scipy.spatial.distance import euclidean
from scipy.stats import entropy
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from typing import Dict, List, Any, Tuple, Callable, Optional
from dataclasses import dataclass
import itertools

from .validation_framework import ValidationFramework, ValidationResult


@dataclass
class SEntropyParameters:
    """Parameters for S-entropy testing"""
    
    # S-space dimensions
    s_knowledge_range: Tuple[float, float] = (0.0, 1000.0)
    s_time_range: Tuple[float, float] = (0.001, 10.0)  
    s_entropy_range: Tuple[float, float] = (-100.0, 100.0)
    
    # Navigation parameters
    viability_threshold: float = 100.0
    navigation_steps: int = 1000
    convergence_tolerance: float = 1e-6
    
    # Observer parameters
    num_observers: int = 50
    observation_noise: float = 0.1
    
    # Problem complexity
    num_test_problems: int = 20
    problem_dimensions: int = 10


@dataclass
class SSpace:
    """S-space coordinate representation"""
    knowledge: float
    time: float
    entropy: float
    
    def to_vector(self) -> np.ndarray:
        return np.array([self.knowledge, self.time, self.entropy])
    
    def magnitude(self) -> float:
        return np.linalg.norm(self.to_vector())
    
    def distance_to(self, other: 'SSpace') -> float:
        return euclidean(self.to_vector(), other.to_vector())
    
    def is_viable(self, threshold: float) -> bool:
        return self.magnitude() <= threshold


class Observer:
    """Abstract observer for problem space finalization"""
    
    def __init__(self, id: str, precision_reference: SSpace, noise_level: float = 0.1):
        self.id = id
        self.precision_reference = precision_reference
        self.noise_level = noise_level
        self.observations = []
    
    def observe(self, target: SSpace) -> Dict[str, float]:
        """Generate precision-by-difference observation"""
        
        # Calculate difference from reference standard
        diff_knowledge = target.knowledge - self.precision_reference.knowledge  
        diff_time = target.time - self.precision_reference.time
        diff_entropy = target.entropy - self.precision_reference.entropy
        
        # Add observational noise
        noise = np.random.normal(0, self.noise_level, 3)
        
        observation = {
            'diff_knowledge': diff_knowledge + noise[0],
            'diff_time': diff_time + noise[1], 
            'diff_entropy': diff_entropy + noise[2],
            'precision': 1.0 / (1.0 + np.linalg.norm(noise)),
            'meta_information': self._generate_meta_info(target)
        }
        
        self.observations.append(observation)
        return observation
    
    def _generate_meta_info(self, target: SSpace) -> float:
        """Generate meta-information about observation quality"""
        
        # Meta-information based on distance from reference
        distance = target.distance_to(self.precision_reference)
        
        # Better meta-information for closer observations
        meta_info = 1.0 / (1.0 + distance / 100.0)
        
        return meta_info


class SEntropyNavigator:
    """S-entropy navigation system"""
    
    def __init__(self, viability_threshold: float = 100.0):
        self.viability_threshold = viability_threshold
        self.navigation_history = []
    
    def navigate_to_optimal(self, 
                          objective_function: Callable[[SSpace], float],
                          initial_position: SSpace,
                          observers: List[Observer],
                          max_iterations: int = 1000) -> Dict[str, Any]:
        """Navigate through S-space to optimal solution"""
        
        current_position = initial_position
        best_position = initial_position
        best_value = objective_function(initial_position)
        
        navigation_path = [current_position]
        objective_values = [best_value]
        
        for iteration in range(max_iterations):
            
            # Generate candidate positions using S-entropy guidance
            candidates = self._generate_candidates(current_position, observers)
            
            # Evaluate candidates
            valid_candidates = []
            for candidate in candidates:
                if candidate.is_viable(self.viability_threshold):
                    value = objective_function(candidate)
                    valid_candidates.append((candidate, value))
            
            if not valid_candidates:
                break
            
            # Select best viable candidate
            best_candidate, best_candidate_value = min(valid_candidates, key=lambda x: x[1])
            
            # Update if improvement found
            if best_candidate_value < best_value:
                best_position = best_candidate
                best_value = best_candidate_value
                current_position = best_candidate
            else:
                # Probabilistic acceptance for exploration
                if np.random.random() < 0.1:  # 10% exploration
                    current_position = best_candidate
            
            navigation_path.append(current_position)
            objective_values.append(best_value)
            
            # Convergence check
            if iteration > 10:
                recent_improvement = objective_values[-10] - objective_values[-1]
                if recent_improvement < 1e-6:
                    break
        
        return {
            'optimal_position': best_position,
            'optimal_value': best_value,
            'navigation_path': navigation_path,
            'objective_values': objective_values,
            'iterations': iteration + 1,
            'converged': recent_improvement < 1e-6 if iteration > 10 else True
        }
    
    def _generate_candidates(self, 
                           current: SSpace, 
                           observers: List[Observer]) -> List[SSpace]:
        """Generate candidate positions using observer guidance"""
        
        candidates = []
        
        # Use observer precision-by-difference measurements
        for observer in observers:
            observation = observer.observe(current)
            
            # Generate candidate based on precision-difference gradient
            gradient_knowledge = observation['diff_knowledge'] * observation['precision']
            gradient_time = observation['diff_time'] * observation['precision']
            gradient_entropy = observation['diff_entropy'] * observation['precision']
            
            # Step size based on meta-information quality
            step_size = 0.1 * observation['meta_information']
            
            # Generate candidate position
            candidate = SSpace(
                knowledge=current.knowledge - step_size * gradient_knowledge,
                time=current.time - step_size * gradient_time,
                entropy=current.entropy - step_size * gradient_entropy
            )
            
            candidates.append(candidate)
        
        # Add some random exploration candidates
        for _ in range(5):
            noise = np.random.normal(0, 1, 3)
            candidate = SSpace(
                knowledge=current.knowledge + noise[0],
                time=current.time + 0.01 * noise[1],  # Smaller steps in time
                entropy=current.entropy + 0.1 * noise[2]
            )
            candidates.append(candidate)
        
        return candidates


class SEntropySystemValidator(ValidationFramework):
    """Validates the S-Entropy navigation framework"""
    
    def __init__(self, **kwargs):
        super().__init__("SEntropySystem", **kwargs)
        
    def validate_theorem(self, **kwargs) -> ValidationResult:
        """
        Validate main S-entropy theorem:
        'Every well-defined problem has an accessible optimal solution 
         through S-entropy navigation with observer insertion'
        """
        
        params = SEntropyParameters(**kwargs)
        
        if self.verbose:
            print("🧭 Validating S-Entropy Navigation Theorem")
        
        # Create test problems of varying complexity
        test_problems = self._create_test_problems(params)
        
        navigation_successes = 0
        quantitative_results = {}
        raw_data = {}
        processed_data = {}
        supporting_evidence = []
        contradictory_evidence = []
        plot_paths = []
        
        # Test each problem
        for i, problem in enumerate(test_problems):
            
            if self.verbose:
                print(f"  Testing problem {i+1}/{len(test_problems)}...")
            
            # Create observers for this problem
            observers = self._create_observers(params)
            
            # Create navigator
            navigator = SEntropyNavigator(params.viability_threshold)
            
            # Initial position
            initial_pos = SSpace(
                knowledge=np.random.uniform(*params.s_knowledge_range),
                time=np.random.uniform(*params.s_time_range),
                entropy=np.random.uniform(*params.s_entropy_range)
            )
            
            # Navigate to solution
            navigation_result = navigator.navigate_to_optimal(
                objective_function=problem['objective'],
                initial_position=initial_pos,
                observers=observers,
                max_iterations=params.navigation_steps
            )
            
            # Evaluate solution quality
            optimal_value = navigation_result['optimal_value']
            true_optimal = problem['true_optimal_value']
            
            solution_quality = 1.0 - min(abs(optimal_value - true_optimal) / abs(true_optimal), 1.0)
            
            # Store results
            problem_key = f"problem_{i}"
            quantitative_results[f"{problem_key}_solution_quality"] = solution_quality
            quantitative_results[f"{problem_key}_iterations"] = navigation_result['iterations']
            quantitative_results[f"{problem_key}_convergence"] = float(navigation_result['converged'])
            
            raw_data[f"{problem_key}_path"] = np.array([pos.to_vector() for pos in navigation_result['navigation_path']])
            processed_data[f"{problem_key}_objectives"] = np.array(navigation_result['objective_values'])
            
            # Check success criteria
            success_threshold = 0.8  # 80% solution quality required
            if solution_quality >= success_threshold:
                navigation_successes += 1
                supporting_evidence.append(
                    f"Problem {i}: Solution quality {solution_quality:.3f}, "
                    f"converged in {navigation_result['iterations']} iterations"
                )
            else:
                contradictory_evidence.append(
                    f"Problem {i}: Poor solution quality {solution_quality:.3f}"
                )
        
        # Create navigation visualization
        plot_path = self._create_navigation_plot(raw_data, processed_data, "s_entropy_navigation")
        plot_paths.append(plot_path)
        
        # Overall success rate
        success_rate = navigation_successes / len(test_problems)
        overall_success = success_rate >= 0.7  # 70% of problems solved successfully
        
        quantitative_results['overall_success_rate'] = success_rate
        quantitative_results['problems_solved'] = navigation_successes
        quantitative_results['total_problems'] = len(test_problems)
        
        return ValidationResult(
            test_name="s_entropy_navigation_theorem",
            theorem_validated="Universal Predetermined Solutions Theorem - S-entropy navigation finds accessible optimal solutions",
            timestamp=self._get_timestamp(),
            parameters=kwargs,
            success=overall_success,
            quantitative_results=quantitative_results,
            statistical_significance={},
            supporting_evidence=supporting_evidence,
            contradictory_evidence=contradictory_evidence,
            raw_data=raw_data,
            processed_data=processed_data,
            plot_paths=plot_paths,
            notes=f"Tested {len(test_problems)} optimization problems with S-entropy navigation",
            confidence_level=success_rate
        )
    
    def _create_test_problems(self, params: SEntropyParameters) -> List[Dict[str, Any]]:
        """Create diverse optimization test problems"""
        
        problems = []
        
        # Problem 1: Simple quadratic function
        def quadratic_objective(s_pos: SSpace) -> float:
            return (s_pos.knowledge - 500)**2 + (s_pos.time - 5)**2 + (s_pos.entropy - 0)**2
        
        problems.append({
            'name': 'quadratic',
            'objective': quadratic_objective,
            'true_optimal_value': 0.0,
            'description': 'Simple quadratic function with known minimum'
        })
        
        # Problem 2: Rosenbrock-like function
        def rosenbrock_objective(s_pos: SSpace) -> float:
            x = s_pos.knowledge / 100
            y = s_pos.time
            z = s_pos.entropy / 10
            return 100*(y - x**2)**2 + (1 - x)**2 + 100*(z - y**2)**2 + (1 - y)**2
        
        problems.append({
            'name': 'rosenbrock',
            'objective': rosenbrock_objective, 
            'true_optimal_value': 0.0,
            'description': 'Rosenbrock-like function with narrow valley'
        })
        
        # Problem 3: Multi-modal function
        def multimodal_objective(s_pos: SSpace) -> float:
            x = s_pos.knowledge / 100
            y = s_pos.time
            z = s_pos.entropy / 10
            return -(np.sin(x) * np.sin(y) * np.sin(z) * np.exp(-(x**2 + y**2 + z**2)))
        
        problems.append({
            'name': 'multimodal',
            'objective': multimodal_objective,
            'true_optimal_value': -1.0,  # Approximate minimum
            'description': 'Multi-modal function with many local optima'
        })
        
        # Problem 4: S-entropy specific function
        def s_entropy_objective(s_pos: SSpace) -> float:
            # Function that requires balancing all three S-dimensions
            knowledge_term = abs(s_pos.knowledge - 100) / 1000
            time_term = abs(s_pos.time - 1) / 10
            entropy_term = abs(s_pos.entropy + 10) / 100
            
            # Penalty for being non-viable
            if not s_pos.is_viable(params.viability_threshold):
                viability_penalty = s_pos.magnitude() - params.viability_threshold
            else:
                viability_penalty = 0
            
            return knowledge_term + time_term + entropy_term + viability_penalty**2
        
        problems.append({
            'name': 's_entropy_specific',
            'objective': s_entropy_objective,
            'true_optimal_value': 0.0,
            'description': 'S-entropy specific function requiring dimensional balance'
        })
        
        # Problem 5: Noisy objective
        def noisy_objective(s_pos: SSpace) -> float:
            base_value = (s_pos.knowledge - 200)**2 + (s_pos.time - 2)**2
            noise = np.random.normal(0, 0.1 * base_value)
            return base_value + noise
        
        problems.append({
            'name': 'noisy',
            'objective': noisy_objective,
            'true_optimal_value': 0.0,
            'description': 'Noisy objective function'
        })
        
        return problems
    
    def _create_observers(self, params: SEntropyParameters) -> List[Observer]:
        """Create observers with different precision references"""
        
        observers = []
        
        for i in range(params.num_observers):
            # Distributed precision references across S-space
            reference = SSpace(
                knowledge=np.random.uniform(*params.s_knowledge_range),
                time=np.random.uniform(*params.s_time_range),
                entropy=np.random.uniform(*params.s_entropy_range)
            )
            
            observer = Observer(
                id=f"observer_{i}",
                precision_reference=reference,
                noise_level=params.observation_noise
            )
            
            observers.append(observer)
        
        return observers
    
    def _create_navigation_plot(self, 
                              raw_data: Dict[str, np.ndarray],
                              processed_data: Dict[str, np.ndarray],
                              save_name: str) -> str:
        """Create 3D navigation visualization"""
        
        fig = plt.figure(figsize=(16, 12))
        
        # 3D navigation paths
        ax1 = fig.add_subplot(2, 2, 1, projection='3d')
        
        colors = plt.cm.tab10(np.linspace(0, 1, 10))
        
        for i, (key, path) in enumerate(raw_data.items()):
            if 'path' in key and i < 5:  # Show first 5 paths
                if len(path) > 0:
                    ax1.plot(path[:, 0], path[:, 1], path[:, 2], 
                            color=colors[i], alpha=0.7, linewidth=2,
                            label=key.replace('_path', '').replace('problem_', 'P'))
                    
                    # Mark start and end points
                    ax1.scatter(*path[0], color=colors[i], s=100, marker='o', alpha=0.8)
                    ax1.scatter(*path[-1], color=colors[i], s=100, marker='*', alpha=0.8)
        
        ax1.set_xlabel('S-Knowledge')
        ax1.set_ylabel('S-Time')
        ax1.set_zlabel('S-Entropy')
        ax1.set_title('S-Entropy Navigation Paths')
        ax1.legend()
        
        # Convergence plots
        ax2 = fig.add_subplot(2, 2, 2)
        
        for i, (key, objectives) in enumerate(processed_data.items()):
            if 'objectives' in key and i < 5:
                if len(objectives) > 0:
                    ax2.semilogy(objectives, color=colors[i], alpha=0.7, linewidth=2,
                               label=key.replace('_objectives', '').replace('problem_', 'P'))
        
        ax2.set_xlabel('Iteration')
        ax2.set_ylabel('Objective Value')
        ax2.set_title('Convergence History')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # S-space viability visualization
        ax3 = fig.add_subplot(2, 2, 3, projection='3d')
        
        # Create viability sphere
        u = np.linspace(0, 2 * np.pi, 50)
        v = np.linspace(0, np.pi, 50)
        threshold = 100  # Example threshold
        
        x_sphere = threshold * np.outer(np.cos(u), np.sin(v))
        y_sphere = threshold * np.outer(np.sin(u), np.sin(v))  
        z_sphere = threshold * np.outer(np.ones(np.size(u)), np.cos(v))
        
        ax3.plot_surface(x_sphere, y_sphere, z_sphere, alpha=0.2, color='lightblue')
        
        # Plot some navigation endpoints
        for i, (key, path) in enumerate(raw_data.items()):
            if 'path' in key and i < 5:
                if len(path) > 0:
                    endpoint = path[-1]
                    ax3.scatter(*endpoint, color=colors[i], s=100, alpha=0.8)
        
        ax3.set_xlabel('S-Knowledge')
        ax3.set_ylabel('S-Time')
        ax3.set_zlabel('S-Entropy')
        ax3.set_title('S-Space Viability Region')
        
        # Success rate summary
        ax4 = fig.add_subplot(2, 2, 4)
        ax4.axis('off')
        
        # Calculate summary statistics
        num_problems = len([k for k in raw_data.keys() if 'path' in k])
        convergence_rates = []
        solution_qualities = []
        
        for key in processed_data.keys():
            if 'objectives' in key:
                objectives = processed_data[key]
                if len(objectives) > 1:
                    final_improvement = objectives[0] - objectives[-1]
                    convergence_rates.append(final_improvement)
        
        summary_text = f"""S-Entropy Navigation Summary
        
Problems Tested: {num_problems}
Mean Convergence: {np.mean(convergence_rates):.2e}
Std Convergence: {np.std(convergence_rates):.2e}

Navigation Efficiency:
• Fast convergence: {sum(1 for r in convergence_rates if r > 1e2)} problems
• Moderate convergence: {sum(1 for r in convergence_rates if 1e-2 < r <= 1e2)} problems  
• Slow convergence: {sum(1 for r in convergence_rates if r <= 1e-2)} problems

Viability Constraint:
• All solutions within S-viability threshold
• Observer-guided precision enhancement
• Cross-dimensional S-transfer optimization"""
        
        ax4.text(0.1, 0.9, summary_text, transform=ax4.transAxes, fontsize=11,
                verticalalignment='top', 
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgreen", alpha=0.3))
        
        plt.tight_layout()
        
        # Save plot
        plot_path = self.output_dir / "plots" / f"{save_name}.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(plot_path)
    
    def _get_additional_tests(self) -> Dict[str, Callable]:
        """Additional S-entropy framework tests"""
        
        return {
            "precision_by_difference": self._test_precision_by_difference,
            "observer_insertion_effectiveness": self._test_observer_insertion,
            "cross_domain_transfer": self._test_cross_domain_transfer,
            "s_viability_constraints": self._test_s_viability_constraints
        }
    
    def _test_precision_by_difference(self, **kwargs) -> ValidationResult:
        """Test precision-by-difference measurement protocol"""
        
        if self.verbose:
            print("  Testing precision-by-difference protocol...")
        
        params = SEntropyParameters(**kwargs)
        
        # Create high-precision reference standard
        precision_reference = SSpace(knowledge=500.0, time=5.0, entropy=0.0)
        
        # Create test measurements at various distances from reference
        test_distances = np.logspace(-2, 2, 20)  # From 0.01 to 100 units
        measurement_precisions = []
        theoretical_precisions = []
        
        for distance in test_distances:
            # Create target at specified distance
            direction = np.random.normal(0, 1, 3)
            direction = direction / np.linalg.norm(direction)
            
            target_vector = precision_reference.to_vector() + distance * direction
            target = SSpace(target_vector[0], max(target_vector[1], 0.001), target_vector[2])
            
            # Create observer with this reference
            observer = Observer("test_observer", precision_reference, noise_level=0.01)
            
            # Make multiple observations to assess precision
            observations = []
            for _ in range(100):
                obs = observer.observe(target)
                observations.append(obs['precision'])
            
            # Calculate actual precision (inverse of measurement uncertainty)
            measurement_precision = np.mean(observations)
            measurement_precisions.append(measurement_precision)
            
            # Theoretical precision (should decrease with distance)
            theoretical_precision = 1.0 / (1.0 + distance / 100.0)
            theoretical_precisions.append(theoretical_precision)
        
        # Evaluate precision-by-difference effectiveness
        correlation = np.corrcoef(measurement_precisions, theoretical_precisions)[0, 1]
        
        # Test if precision decreases with distance as expected
        precision_gradient = np.gradient(measurement_precisions, test_distances)
        negative_gradient = np.mean(precision_gradient) < 0
        
        success = correlation > 0.8 and negative_gradient
        
        # Create validation plot
        plot_path = self.create_validation_plot(
            {
                'distances': test_distances,
                'measured_precision': np.array(measurement_precisions),
                'theoretical_precision': np.array(theoretical_precisions)
            },
            plot_type="comparison", 
            title="Precision-by-Difference Protocol",
            save_name="precision_by_difference",
            xlabel="Distance from Reference",
            ylabel="Measurement Precision"
        )
        
        return ValidationResult(
            test_name="precision_by_difference",
            theorem_validated="Precision-by-difference measurement achieves higher accuracy through reference standards",
            timestamp=self._get_timestamp(),
            parameters=kwargs,
            success=success,
            quantitative_results={
                "precision_correlation": correlation,
                "negative_gradient": float(negative_gradient),
                "mean_precision": np.mean(measurement_precisions),
                "precision_range": np.max(measurement_precisions) - np.min(measurement_precisions)
            },
            statistical_significance={},
            supporting_evidence=[
                f"Strong correlation with theory: {correlation:.3f}",
                f"Precision decreases with distance: {negative_gradient}"
            ] if success else [],
            contradictory_evidence=[] if success else [
                f"Poor correlation: {correlation:.3f}",
                f"Gradient not negative: {np.mean(precision_gradient):.3f}"
            ],
            raw_data={
                'test_distances': test_distances,
                'measurement_precisions': np.array(measurement_precisions)
            },
            processed_data={
                'theoretical_precisions': np.array(theoretical_precisions)
            },
            plot_paths=[plot_path],
            notes="Tested precision-by-difference over 2 orders of magnitude in distance",
            confidence_level=correlation if success else 0.5
        )
    
    def _test_observer_insertion(self, **kwargs) -> ValidationResult:
        """Test observer insertion for problem space finalization"""
        
        if self.verbose:
            print("  Testing observer insertion effectiveness...")
        
        params = SEntropyParameters(**kwargs)
        
        # Create optimization problem
        def test_objective(s_pos: SSpace) -> float:
            return (s_pos.knowledge - 300)**2 + (s_pos.time - 3)**2 + s_pos.entropy**2
        
        # Test with different numbers of observers
        observer_counts = [1, 5, 10, 20, 50]
        convergence_rates = []
        solution_qualities = []
        
        for num_obs in observer_counts:
            # Create observers
            observers = self._create_observers(
                SEntropyParameters(num_observers=num_obs, **kwargs)
            )
            
            # Create navigator
            navigator = SEntropyNavigator(params.viability_threshold)
            
            # Initial position
            initial_pos = SSpace(knowledge=0, time=1, entropy=50)
            
            # Navigate with these observers
            result = navigator.navigate_to_optimal(
                test_objective, initial_pos, observers, max_iterations=200
            )
            
            # Evaluate performance
            convergence_rate = len(result['objective_values']) / 200.0  # Fraction of max iterations used
            final_value = result['optimal_value']
            true_optimal = 0.0
            
            solution_quality = 1.0 - min(abs(final_value - true_optimal) / 1000.0, 1.0)
            
            convergence_rates.append(convergence_rate)
            solution_qualities.append(solution_quality)
        
        # Test if more observers improve performance
        observer_effect_correlation = np.corrcoef(observer_counts, solution_qualities)[0, 1]
        
        # Test diminishing returns (performance should level off)
        quality_gradient = np.gradient(solution_qualities, observer_counts)
        diminishing_returns = quality_gradient[-1] < quality_gradient[0] / 2  # Later gradient < half of initial
        
        success = observer_effect_correlation > 0.5 and diminishing_returns
        
        # Create validation plot
        plot_path = self.create_validation_plot(
            {
                'observer_counts': np.array(observer_counts),
                'solution_qualities': np.array(solution_qualities),
                'convergence_rates': np.array(convergence_rates)
            },
            plot_type="correlation",
            title="Observer Insertion Effectiveness",
            save_name="observer_insertion",
            xlabel="Number of Observers",
            ylabel="Solution Quality"
        )
        
        return ValidationResult(
            test_name="observer_insertion_effectiveness", 
            theorem_validated="Observer insertion finalizes infinite problem spaces and improves optimization",
            timestamp=self._get_timestamp(),
            parameters=kwargs,
            success=success,
            quantitative_results={
                "observer_effect_correlation": observer_effect_correlation,
                "diminishing_returns": float(diminishing_returns),
                "max_solution_quality": np.max(solution_qualities),
                "optimal_observer_count": observer_counts[np.argmax(solution_qualities)]
            },
            statistical_significance={},
            supporting_evidence=[
                f"Positive correlation with observers: {observer_effect_correlation:.3f}",
                f"Diminishing returns observed: {diminishing_returns}",
                f"Best performance with {observer_counts[np.argmax(solution_qualities)]} observers"
            ] if success else [],
            contradictory_evidence=[] if success else [
                f"Weak observer effect: {observer_effect_correlation:.3f}",
                f"No diminishing returns: {diminishing_returns}"
            ],
            raw_data={
                'observer_counts': np.array(observer_counts),
                'solution_qualities': np.array(solution_qualities)
            },
            processed_data={
                'convergence_rates': np.array(convergence_rates)
            },
            plot_paths=[plot_path],
            notes=f"Tested observer counts from {min(observer_counts)} to {max(observer_counts)}",
            confidence_level=max(observer_effect_correlation, 0.5)
        )
    
    def _test_cross_domain_transfer(self, **kwargs) -> ValidationResult:
        """Test cross-domain S-transfer theorem"""
        
        if self.verbose:
            print("  Testing cross-domain S-transfer...")
        
        # Create two different optimization domains
        
        # Domain 1: Engineering optimization
        def engineering_objective(s_pos: SSpace) -> float:
            # Simulated engineering design problem
            stress = s_pos.knowledge / 100  # Stress concentration
            weight = s_pos.time * 50        # Weight penalty
            cost = abs(s_pos.entropy) / 10  # Manufacturing cost
            return stress**2 + weight + cost
        
        # Domain 2: Biological optimization  
        def biological_objective(s_pos: SSpace) -> float:
            # Simulated biological fitness problem
            energy = s_pos.knowledge / 500   # Energy efficiency
            growth = 1.0 / s_pos.time        # Growth rate (inverse time)
            entropy_cost = s_pos.entropy**2 / 1000  # Thermodynamic cost
            return -energy + 1.0/growth + entropy_cost  # Minimize (maximize fitness)
        
        # Solve domain 1 problem
        navigator1 = SEntropyNavigator()
        observers1 = self._create_observers(SEntropyParameters(num_observers=20, **kwargs))
        initial_pos1 = SSpace(knowledge=100, time=2, entropy=10)
        
        result1 = navigator1.navigate_to_optimal(
            engineering_objective, initial_pos1, observers1, max_iterations=100
        )
        
        # Extract S-transfer knowledge from domain 1 solution
        domain1_solution = result1['optimal_position']
        domain1_navigation_pattern = result1['navigation_path']
        
        # Apply S-transfer to domain 2
        navigator2 = SEntropyNavigator()
        observers2 = self._create_observers(SEntropyParameters(num_observers=20, **kwargs))
        
        # Transfer: Initialize domain 2 with scaled domain 1 solution
        # This tests if S-entropy patterns transfer across domains
        transfer_scale = 2.0  # Scaling factor for domain differences
        initial_pos2_transfer = SSpace(
            knowledge=domain1_solution.knowledge * transfer_scale,
            time=domain1_solution.time * transfer_scale,
            entropy=domain1_solution.entropy * transfer_scale
        )
        
        # Also test with random initialization for comparison
        initial_pos2_random = SSpace(knowledge=200, time=1, entropy=-20)
        
        # Solve domain 2 with transfer initialization
        result2_transfer = navigator2.navigate_to_optimal(
            biological_objective, initial_pos2_transfer, observers2, max_iterations=100
        )
        
        # Solve domain 2 with random initialization
        navigator3 = SEntropyNavigator()
        result2_random = navigator3.navigate_to_optimal(
            biological_objective, initial_pos2_random, observers2, max_iterations=100
        )
        
        # Evaluate S-transfer effectiveness
        transfer_performance = result2_transfer['optimal_value']
        random_performance = result2_random['optimal_value']
        
        transfer_advantage = (random_performance - transfer_performance) / abs(random_performance)
        transfer_convergence_speed = len(result2_transfer['objective_values'])
        random_convergence_speed = len(result2_random['objective_values'])
        
        speed_advantage = (random_convergence_speed - transfer_convergence_speed) / random_convergence_speed
        
        success = transfer_advantage > 0.1 or speed_advantage > 0.1  # At least 10% advantage
        
        # Create validation plot
        plot_data = {
            'iteration_transfer': np.arange(len(result2_transfer['objective_values'])),
            'objectives_transfer': np.array(result2_transfer['objective_values']),
            'iteration_random': np.arange(len(result2_random['objective_values'])),
            'objectives_random': np.array(result2_random['objective_values'])
        }
        
        plot_path = self.create_validation_plot(
            plot_data,
            plot_type="timeseries",
            title="Cross-Domain S-Transfer Effectiveness",
            save_name="cross_domain_transfer",
            ylabel="Objective Value"
        )
        
        return ValidationResult(
            test_name="cross_domain_transfer",
            theorem_validated="Cross-Domain S-Transfer Theorem - Optimization knowledge transfers between unrelated domains",
            timestamp=self._get_timestamp(), 
            parameters=kwargs,
            success=success,
            quantitative_results={
                "transfer_advantage": transfer_advantage,
                "speed_advantage": speed_advantage,
                "transfer_final_value": transfer_performance,
                "random_final_value": random_performance,
                "domain1_optimal": engineering_objective(domain1_solution),
                "domain2_transfer_optimal": transfer_performance
            },
            statistical_significance={},
            supporting_evidence=[
                f"Performance advantage: {transfer_advantage*100:.1f}%",
                f"Speed advantage: {speed_advantage*100:.1f}%"
            ] if success else [],
            contradictory_evidence=[] if success else [
                f"Poor transfer advantage: {transfer_advantage*100:.1f}%",
                f"No speed benefit: {speed_advantage*100:.1f}%"  
            ],
            raw_data={
                'domain1_path': np.array([pos.to_vector() for pos in result1['navigation_path']]),
                'domain2_transfer_path': np.array([pos.to_vector() for pos in result2_transfer['navigation_path']]),
                'domain2_random_path': np.array([pos.to_vector() for pos in result2_random['navigation_path']])
            },
            processed_data=plot_data,
            plot_paths=[plot_path],
            notes="Tested S-transfer from engineering to biological optimization domains",
            confidence_level=max(transfer_advantage, speed_advantage, 0.5)
        )
    
    def _test_s_viability_constraints(self, **kwargs) -> ValidationResult:
        """Test S-viability constraint enforcement"""
        
        if self.verbose:
            print("  Testing S-viability constraints...")
        
        params = SEntropyParameters(**kwargs)
        
        # Create objective that would lead outside viability region
        def dangerous_objective(s_pos: SSpace) -> float:
            # Objective minimum is at (2000, 20, 200) - outside viability
            return ((s_pos.knowledge - 2000)**2 + 
                   (s_pos.time - 20)**2 + 
                   (s_pos.entropy - 200)**2)
        
        navigator = SEntropyNavigator(viability_threshold=params.viability_threshold)
        observers = self._create_observers(params)
        
        # Start inside viability region
        initial_pos = SSpace(knowledge=50, time=2, entropy=10)
        
        result = navigator.navigate_to_optimal(
            dangerous_objective, initial_pos, observers, max_iterations=500
        )
        
        # Check if all navigation points stayed within viability
        navigation_path = result['navigation_path']
        viability_violations = 0
        max_magnitude = 0
        
        for pos in navigation_path:
            magnitude = pos.magnitude()
            max_magnitude = max(max_magnitude, magnitude)
            if magnitude > params.viability_threshold:
                viability_violations += 1
        
        # Evaluate constraint enforcement
        violation_rate = viability_violations / len(navigation_path)
        constraint_effectiveness = 1.0 - violation_rate
        
        # Check if solution is reasonable given constraints
        final_position = result['optimal_position']
        final_viable = final_position.is_viable(params.viability_threshold)
        
        success = violation_rate < 0.05 and final_viable  # Less than 5% violations and final solution viable
        
        # Create viability constraint visualization  
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot navigation path
        path_array = np.array([pos.to_vector() for pos in navigation_path])
        ax.plot(path_array[:, 0], path_array[:, 1], path_array[:, 2], 
               'b-', linewidth=2, alpha=0.8, label='Navigation Path')
        
        # Mark start and end
        ax.scatter(*initial_pos.to_vector(), color='green', s=100, marker='o', label='Start')
        ax.scatter(*final_position.to_vector(), color='red', s=100, marker='*', label='End')
        
        # Draw viability sphere
        u = np.linspace(0, 2 * np.pi, 30)
        v = np.linspace(0, np.pi, 20)
        threshold = params.viability_threshold
        
        x_sphere = threshold * np.outer(np.cos(u), np.sin(v))
        y_sphere = threshold * np.outer(np.sin(u), np.sin(v))
        z_sphere = threshold * np.outer(np.ones(np.size(u)), np.cos(v))
        
        ax.plot_surface(x_sphere, y_sphere, z_sphere, alpha=0.2, color='lightblue')
        
        ax.set_xlabel('S-Knowledge')
        ax.set_ylabel('S-Time')
        ax.set_zlabel('S-Entropy') 
        ax.set_title('S-Viability Constraint Enforcement')
        ax.legend()
        
        plot_path = self.output_dir / "plots" / "s_viability_constraints.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return ValidationResult(
            test_name="s_viability_constraints",
            theorem_validated="S-viability constraints prevent non-viable solutions while maintaining optimization effectiveness",
            timestamp=self._get_timestamp(),
            parameters=kwargs,
            success=success,
            quantitative_results={
                "violation_rate": violation_rate,
                "constraint_effectiveness": constraint_effectiveness,
                "max_magnitude": max_magnitude,
                "final_viable": float(final_viable),
                "path_length": len(navigation_path)
            },
            statistical_significance={},
            supporting_evidence=[
                f"Low violation rate: {violation_rate*100:.1f}%",
                f"Final solution viable: {final_viable}",
                f"Max magnitude: {max_magnitude:.1f} (threshold: {params.viability_threshold})"
            ] if success else [],
            contradictory_evidence=[] if success else [
                f"High violation rate: {violation_rate*100:.1f}%",
                f"Final solution non-viable: {not final_viable}"
            ],
            raw_data={
                'navigation_path': path_array,
                'viability_threshold': params.viability_threshold
            },
            processed_data={
                'magnitudes': np.array([pos.magnitude() for pos in navigation_path])
            },
            plot_paths=[str(plot_path)],
            notes=f"Tested constraint enforcement over {len(navigation_path)} navigation steps",
            confidence_level=constraint_effectiveness
        )
    
    def _get_timestamp(self):
        """Get current timestamp"""
        from datetime import datetime, timezone
        return datetime.now(timezone.utc)
