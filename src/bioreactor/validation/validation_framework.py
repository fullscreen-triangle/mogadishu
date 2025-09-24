"""
Validation Framework Base Classes
=================================

Rigorous testing framework for cellular architecture theories.
Provides standardized validation, result storage, and visualization.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
import pickle
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass, asdict
from abc import ABC, abstractmethod
import logging
import warnings

# Configure scientific plotting
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

@dataclass
class ValidationResult:
    """Standardized validation result container"""
    
    # Test identification
    test_name: str
    theorem_validated: str
    timestamp: datetime
    
    # Test parameters  
    parameters: Dict[str, Any]
    
    # Results
    success: bool
    quantitative_results: Dict[str, float]
    statistical_significance: Dict[str, float]
    
    # Evidence
    supporting_evidence: List[str]
    contradictory_evidence: List[str]
    
    # Data
    raw_data: Dict[str, np.ndarray]
    processed_data: Dict[str, np.ndarray]
    
    # Visualization paths
    plot_paths: List[str]
    
    # Notes
    notes: str
    confidence_level: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        result_dict = asdict(self)
        
        # Convert numpy arrays to lists for JSON
        for key, value in result_dict['raw_data'].items():
            if isinstance(value, np.ndarray):
                result_dict['raw_data'][key] = value.tolist()
                
        for key, value in result_dict['processed_data'].items():
            if isinstance(value, np.ndarray):
                result_dict['processed_data'][key] = value.tolist()
                
        # Convert timestamp to ISO format
        result_dict['timestamp'] = self.timestamp.isoformat()
        
        return result_dict

class ValidationFramework(ABC):
    """Base class for all validation modules"""
    
    def __init__(self, 
                 name: str,
                 output_dir: str = "validation_results",
                 verbose: bool = True):
        
        self.name = name
        self.output_dir = Path(output_dir)
        self.verbose = verbose
        
        # Create output directory structure
        self.output_dir.mkdir(exist_ok=True)
        (self.output_dir / "plots").mkdir(exist_ok=True)
        (self.output_dir / "data").mkdir(exist_ok=True)
        (self.output_dir / "results").mkdir(exist_ok=True)
        
        # Setup logging
        self.logger = self._setup_logging()
        
        # Results storage
        self.results: List[ValidationResult] = []
        
        if self.verbose:
            print(f"🔬 Initialized {self.name} Validation Framework")
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for validation framework"""
        logger = logging.getLogger(f"validation.{self.name}")
        logger.setLevel(logging.INFO)
        
        # File handler
        log_file = self.output_dir / f"{self.name.lower()}_validation.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        
        # Console handler  
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.WARNING)
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
        return logger
    
    @abstractmethod
    def validate_theorem(self, **kwargs) -> ValidationResult:
        """Validate the main theorem - must be implemented by subclasses"""
        pass
    
    def run_validation_suite(self, **kwargs) -> Dict[str, ValidationResult]:
        """Run complete validation suite"""
        
        self.logger.info(f"Starting validation suite for {self.name}")
        
        if self.verbose:
            print(f"🧪 Running {self.name} Validation Suite")
            print("=" * 50)
        
        suite_results = {}
        
        try:
            # Run main theorem validation
            main_result = self.validate_theorem(**kwargs)
            suite_results["main_theorem"] = main_result
            self.results.append(main_result)
            
            # Run additional tests if available
            additional_tests = self._get_additional_tests()
            
            for test_name, test_func in additional_tests.items():
                if self.verbose:
                    print(f"  Running {test_name}...")
                    
                test_result = test_func(**kwargs)
                suite_results[test_name] = test_result
                self.results.append(test_result)
                
                if self.verbose:
                    status = "✅ PASS" if test_result.success else "❌ FAIL"
                    print(f"    {status} - Confidence: {test_result.confidence_level:.3f}")
            
            # Generate summary report
            self._generate_summary_report(suite_results)
            
        except Exception as e:
            self.logger.error(f"Validation suite failed: {str(e)}")
            raise
            
        if self.verbose:
            print("=" * 50)
            print(f"✅ {self.name} Validation Suite Complete")
            
        return suite_results
    
    def _get_additional_tests(self) -> Dict[str, Callable]:
        """Override in subclasses to add additional tests"""
        return {}
    
    def save_results(self) -> str:
        """Save all validation results"""
        
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        results_file = self.output_dir / "results" / f"{self.name.lower()}_{timestamp}.json"
        
        # Convert results to serializable format
        serializable_results = []
        for result in self.results:
            serializable_results.append(result.to_dict())
        
        # Save to JSON
        with open(results_file, 'w') as f:
            json.dump({
                'framework': self.name,
                'timestamp': timestamp,
                'results': serializable_results
            }, f, indent=2)
        
        # Also save as pickle for Python objects
        pickle_file = self.output_dir / "results" / f"{self.name.lower()}_{timestamp}.pkl" 
        with open(pickle_file, 'wb') as f:
            pickle.dump(self.results, f)
            
        self.logger.info(f"Results saved to {results_file}")
        
        if self.verbose:
            print(f"💾 Results saved: {results_file.name}")
            
        return str(results_file)
    
    def load_results(self, results_file: str) -> List[ValidationResult]:
        """Load previous validation results"""
        
        results_path = Path(results_file)
        
        if results_path.suffix == '.pkl':
            with open(results_path, 'rb') as f:
                loaded_results = pickle.load(f)
        else:
            # Load from JSON and reconstruct ValidationResult objects
            with open(results_path, 'r') as f:
                data = json.load(f)
                
            loaded_results = []
            for result_dict in data['results']:
                # Convert timestamp back
                result_dict['timestamp'] = datetime.fromisoformat(result_dict['timestamp'])
                
                # Convert lists back to numpy arrays
                for key, value in result_dict['raw_data'].items():
                    result_dict['raw_data'][key] = np.array(value)
                for key, value in result_dict['processed_data'].items():
                    result_dict['processed_data'][key] = np.array(value)
                
                loaded_results.append(ValidationResult(**result_dict))
        
        self.results.extend(loaded_results)
        
        if self.verbose:
            print(f"📁 Loaded {len(loaded_results)} previous results")
            
        return loaded_results
    
    def create_validation_plot(self, 
                             data: Dict[str, np.ndarray],
                             plot_type: str,
                             title: str,
                             save_name: str,
                             **plot_kwargs) -> str:
        """Create standardized validation plots"""
        
        fig, axes = plt.subplots(figsize=(12, 8))
        
        if plot_type == "comparison":
            self._plot_comparison(axes, data, **plot_kwargs)
        elif plot_type == "timeseries":
            self._plot_timeseries(axes, data, **plot_kwargs)
        elif plot_type == "distribution":
            self._plot_distribution(axes, data, **plot_kwargs)
        elif plot_type == "correlation":
            self._plot_correlation(axes, data, **plot_kwargs)
        elif plot_type == "validation":
            self._plot_validation(axes, data, **plot_kwargs)
        else:
            raise ValueError(f"Unknown plot type: {plot_type}")
        
        axes.set_title(f"{self.name}: {title}", fontsize=14, fontweight='bold')
        axes.grid(True, alpha=0.3)
        
        # Add validation metadata
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        fig.text(0.02, 0.02, f"Validation Framework: {timestamp}", 
                fontsize=8, alpha=0.7)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = self.output_dir / "plots" / f"{save_name}.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        
        if self.verbose:
            print(f"📊 Plot saved: {plot_path.name}")
            
        plt.close()
        
        return str(plot_path)
    
    def _plot_comparison(self, ax, data: Dict[str, np.ndarray], **kwargs):
        """Plot comparison between theoretical and experimental"""
        
        theoretical = data.get('theoretical')
        experimental = data.get('experimental') 
        x_values = data.get('x_values', range(len(theoretical)))
        
        ax.plot(x_values, theoretical, 'b-', linewidth=2, label='Theoretical', alpha=0.8)
        ax.plot(x_values, experimental, 'ro', markersize=4, label='Experimental', alpha=0.7)
        
        # Calculate R²
        r_squared = np.corrcoef(theoretical, experimental)[0,1]**2
        ax.text(0.05, 0.95, f'R² = {r_squared:.3f}', transform=ax.transAxes, 
               bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        
        ax.set_xlabel(kwargs.get('xlabel', 'X'))
        ax.set_ylabel(kwargs.get('ylabel', 'Y'))
        ax.legend()
    
    def _plot_timeseries(self, ax, data: Dict[str, np.ndarray], **kwargs):
        """Plot time series data"""
        
        time = data.get('time')
        
        for name, values in data.items():
            if name != 'time':
                ax.plot(time, values, label=name, linewidth=2, alpha=0.8)
        
        ax.set_xlabel('Time')
        ax.set_ylabel(kwargs.get('ylabel', 'Value'))
        ax.legend()
    
    def _plot_distribution(self, ax, data: Dict[str, np.ndarray], **kwargs):
        """Plot distribution comparison"""
        
        for name, values in data.items():
            ax.hist(values, bins=kwargs.get('bins', 50), alpha=0.6, 
                   label=name, density=True)
        
        ax.set_xlabel(kwargs.get('xlabel', 'Value'))
        ax.set_ylabel('Density')
        ax.legend()
    
    def _plot_correlation(self, ax, data: Dict[str, np.ndarray], **kwargs):
        """Plot correlation analysis"""
        
        x = data.get('x')
        y = data.get('y')
        
        ax.scatter(x, y, alpha=0.6, s=20)
        
        # Fit line
        z = np.polyfit(x, y, 1)
        p = np.poly1d(z)
        ax.plot(x, p(x), "r--", alpha=0.8)
        
        # Correlation coefficient
        corr = np.corrcoef(x, y)[0,1]
        ax.text(0.05, 0.95, f'r = {corr:.3f}', transform=ax.transAxes,
               bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        
        ax.set_xlabel(kwargs.get('xlabel', 'X'))
        ax.set_ylabel(kwargs.get('ylabel', 'Y'))
    
    def _plot_validation(self, ax, data: Dict[str, np.ndarray], **kwargs):
        """Plot validation-specific visualization"""
        
        predictions = data.get('predictions')
        actual = data.get('actual')
        
        # Perfect prediction line
        min_val = min(np.min(predictions), np.min(actual))
        max_val = max(np.max(predictions), np.max(actual))
        ax.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5, label='Perfect Prediction')
        
        # Scatter plot
        ax.scatter(actual, predictions, alpha=0.6, s=20)
        
        # Calculate metrics
        mse = np.mean((predictions - actual)**2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(predictions - actual))
        
        ax.text(0.05, 0.85, f'RMSE = {rmse:.3f}\nMAE = {mae:.3f}', 
               transform=ax.transAxes,
               bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        
        ax.set_xlabel('Actual')
        ax.set_ylabel('Predicted')
        ax.legend()
    
    def _generate_summary_report(self, results: Dict[str, ValidationResult]) -> str:
        """Generate comprehensive summary report"""
        
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        report_file = self.output_dir / f"{self.name.lower()}_summary_report.md"
        
        with open(report_file, 'w') as f:
            f.write(f"# {self.name} Validation Report\n\n")
            f.write(f"**Generated:** {timestamp}\n\n")
            
            # Overall summary
            total_tests = len(results)
            passed_tests = sum(1 for r in results.values() if r.success)
            
            f.write(f"## Overall Summary\n\n")
            f.write(f"- **Total Tests:** {total_tests}\n")
            f.write(f"- **Passed:** {passed_tests}\n")
            f.write(f"- **Failed:** {total_tests - passed_tests}\n") 
            f.write(f"- **Success Rate:** {passed_tests/total_tests*100:.1f}%\n\n")
            
            # Individual results
            f.write(f"## Individual Test Results\n\n")
            
            for test_name, result in results.items():
                status = "✅ PASSED" if result.success else "❌ FAILED"
                f.write(f"### {test_name.replace('_', ' ').title()}\n\n")
                f.write(f"**Status:** {status}\n")
                f.write(f"**Confidence:** {result.confidence_level:.3f}\n")
                f.write(f"**Theorem:** {result.theorem_validated}\n\n")
                
                if result.quantitative_results:
                    f.write("**Quantitative Results:**\n")
                    for metric, value in result.quantitative_results.items():
                        f.write(f"- {metric}: {value:.6f}\n")
                    f.write("\n")
                
                if result.supporting_evidence:
                    f.write("**Supporting Evidence:**\n")
                    for evidence in result.supporting_evidence:
                        f.write(f"- {evidence}\n")
                    f.write("\n")
                
                if result.contradictory_evidence:
                    f.write("**Contradictory Evidence:**\n")
                    for evidence in result.contradictory_evidence:
                        f.write(f"- {evidence}\n")
                    f.write("\n")
                
                if result.notes:
                    f.write(f"**Notes:** {result.notes}\n\n")
                
                f.write("---\n\n")
        
        if self.verbose:
            print(f"📋 Summary report generated: {report_file.name}")
            
        return str(report_file)

    def statistical_test(self, 
                        data1: np.ndarray, 
                        data2: np.ndarray,
                        test_type: str = "ttest",
                        alpha: float = 0.05) -> Dict[str, float]:
        """Perform statistical significance testing"""
        
        from scipy import stats
        
        if test_type == "ttest":
            statistic, p_value = stats.ttest_ind(data1, data2)
        elif test_type == "ks":
            statistic, p_value = stats.ks_2samp(data1, data2)
        elif test_type == "mannwhitney":
            statistic, p_value = stats.mannwhitneyu(data1, data2)
        else:
            raise ValueError(f"Unknown test type: {test_type}")
        
        return {
            "statistic": float(statistic),
            "p_value": float(p_value),
            "significant": p_value < alpha,
            "alpha": alpha,
            "effect_size": float(np.abs(np.mean(data1) - np.mean(data2)) / 
                               np.sqrt((np.var(data1) + np.var(data2)) / 2))
        }
