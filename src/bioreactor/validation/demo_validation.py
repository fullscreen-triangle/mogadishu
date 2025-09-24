#!/usr/bin/env python3
"""
Demo Validation Script
======================

Demonstrates how to use the S-Entropy bioreactor validation framework.
Runs selected validation modules and generates comprehensive reports.

Usage:
    python demo_validation.py [--all] [--module MODULE_NAME] [--output OUTPUT_DIR]

Examples:
    python demo_validation.py --all                           # Run all validations
    python demo_validation.py --module cellular               # Run cellular validation only
    python demo_validation.py --module integration --output results/  # Custom output directory
"""

import argparse
import sys
from pathlib import Path

# Add the validation module to path
sys.path.append(str(Path(__file__).parent))

from validation_framework import ValidationFramework
from oscillatory_validation import OscillatorySystemValidator, OscillatoryParameters
from s_entropy_validation import SEntropySystemValidator, SEntropyParameters  
from cellular_validation import CellularArchitectureValidator, CellularParameters
from virtual_cell_validation import VirtualCellSystemValidator, VirtualCellParameters
from evidence_validation import EvidenceRectificationValidator, EvidenceValidationParameters
from integration_validation import IntegratedSystemValidator, IntegrationParameters, ValidationSuite


def run_single_module_demo(module_name: str, output_dir: str = "demo_results") -> None:
    """Run validation for a single module"""
    
    print(f"🔬 Running {module_name.title()} Validation Demo")
    print("=" * 50)
    
    try:
        if module_name == "oscillatory":
            validator = OscillatorySystemValidator(output_dir=output_dir, verbose=True)
            params = OscillatoryParameters(
                system_size=100,  # Smaller for demo
                time_span=(0, 50),
                dt=0.05
            )
            results = validator.run_validation_suite(**params.__dict__)
            
        elif module_name == "s_entropy":
            validator = SEntropySystemValidator(output_dir=output_dir, verbose=True)
            params = SEntropyParameters(
                num_observers=20,  # Smaller for demo
                num_test_problems=5,
                navigation_steps=200
            )
            results = validator.run_validation_suite(**params.__dict__)
            
        elif module_name == "cellular":
            validator = CellularArchitectureValidator(output_dir=output_dir, verbose=True)
            params = CellularParameters(
                num_test_molecules=200,  # Smaller for demo
                num_enzymes=100,
                num_metabolites=200
            )
            results = validator.run_validation_suite(**params.__dict__)
            
        elif module_name == "virtual_cell":
            validator = VirtualCellSystemValidator(output_dir=output_dir, verbose=True)
            params = VirtualCellParameters(
                num_enzymes=100,  # Smaller for demo
                num_metabolites=200,
                num_observers=10
            )
            results = validator.run_validation_suite(**params.__dict__)
            
        elif module_name == "evidence":
            validator = EvidenceRectificationValidator(output_dir=output_dir, verbose=True)
            params = EvidenceValidationParameters(
                num_evidence_items=100,  # Smaller for demo
                num_molecular_targets=10
            )
            results = validator.run_validation_suite(**params.__dict__)
            
        elif module_name == "integration":
            validator = IntegratedSystemValidator(output_dir=output_dir, verbose=True)
            params = IntegrationParameters(
                num_virtual_cells=100,  # Smaller for demo
                num_test_scenarios=5,
                simulation_duration=50.0
            )
            results = validator.run_validation_suite(**params.__dict__)
            
        else:
            raise ValueError(f"Unknown module: {module_name}")
        
        # Save results
        validator.save_results()
        
        # Print summary
        print(f"\n✅ {module_name.title()} Validation Complete!")
        print(f"Results saved to: {output_dir}")
        
        total_tests = len(results)
        successful_tests = sum(1 for r in results.values() if r.success)
        print(f"Tests passed: {successful_tests}/{total_tests} ({successful_tests/total_tests*100:.1f}%)")
        
        if successful_tests == total_tests:
            print("🎉 All tests passed!")
        elif successful_tests > total_tests * 0.8:
            print("✨ Most tests passed - good performance!")
        else:
            print("⚠️  Some tests failed - check detailed results")
            
    except Exception as e:
        print(f"❌ Validation failed: {str(e)}")
        raise


def run_complete_suite_demo(output_dir: str = "demo_results") -> None:
    """Run complete validation suite demo"""
    
    print("🚀 Running Complete S-Entropy Framework Validation Suite Demo")
    print("=" * 70)
    print("This will run all validation modules with reduced parameters for demonstration.\n")
    
    # Create validation suite
    suite = ValidationSuite(output_dir=output_dir)
    
    # Run with demo parameters (reduced for speed)
    demo_params = {
        # Oscillatory parameters
        'system_size': 50,
        'time_span': (0, 20),
        'dt': 0.1,
        
        # S-entropy parameters
        'num_observers': 10,
        'num_test_problems': 3,
        'navigation_steps': 100,
        
        # Cellular parameters
        'num_test_molecules': 100,
        'num_enzymes': 50,
        'num_metabolites': 100,
        
        # Virtual cell parameters
        'num_transporters': 20,
        
        # Evidence parameters
        'num_evidence_items': 50,
        'num_molecular_targets': 5,
        
        # Integration parameters
        'num_virtual_cells': 50,
        'num_test_scenarios': 3,
        'simulation_duration': 20.0
    }
    
    # Run complete suite
    all_results = suite.run_complete_validation(**demo_params)
    
    # Print final summary
    print(f"\n🎉 Complete Validation Suite Demo Finished!")
    print("=" * 70)
    
    successful_modules = sum(1 for results in all_results.values() 
                           if isinstance(results, dict) and 'error' not in results)
    total_modules = len(all_results)
    
    print(f"Modules completed: {successful_modules}/{total_modules} ({successful_modules/total_modules*100:.1f}%)")
    print(f"Results directory: {output_dir}")
    
    # Test summary by module
    for module_name, results in all_results.items():
        if 'error' in results:
            print(f"❌ {module_name}: FAILED - {results['error']}")
        else:
            successful_tests = sum(1 for r in results.values() if r.success)
            total_tests = len(results)
            print(f"✅ {module_name}: {successful_tests}/{total_tests} tests passed")


def run_quick_demo(output_dir: str = "quick_demo_results") -> None:
    """Run a quick demonstration with minimal parameters"""
    
    print("⚡ Running Quick S-Entropy Framework Demo")
    print("=" * 45)
    print("Testing core functionality with minimal parameters...\n")
    
    # Run just the integration module with very small parameters
    try:
        validator = IntegratedSystemValidator(output_dir=output_dir, verbose=True)
        params = IntegrationParameters(
            num_virtual_cells=10,      # Very small
            num_test_scenarios=2,      # Just 2 scenarios
            simulation_duration=10.0,  # Short simulation
            time_step=1.0             # Large time steps
        )
        
        print("Running integrated system validation with minimal parameters...")
        results = validator.run_validation_suite(**params.__dict__)
        
        # Save results
        validator.save_results()
        
        # Quick summary
        total_tests = len(results)
        successful_tests = sum(1 for r in results.values() if r.success)
        
        print(f"\n⚡ Quick Demo Complete!")
        print(f"Tests: {successful_tests}/{total_tests} passed")
        print(f"Results: {output_dir}")
        
        if successful_tests > 0:
            print("✨ Core framework is functional!")
            
            # Show a sample result
            sample_result = next(iter(results.values()))
            print(f"\nSample test: {sample_result.test_name}")
            print(f"Success: {sample_result.success}")
            print(f"Confidence: {sample_result.confidence_level:.3f}")
            print(f"Supporting evidence: {len(sample_result.supporting_evidence)} items")
        
    except Exception as e:
        print(f"❌ Quick demo failed: {str(e)}")
        raise


def main():
    """Main demo script entry point"""
    
    parser = argparse.ArgumentParser(
        description="S-Entropy Bioreactor Validation Framework Demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --quick                                    # Quick functionality test
  %(prog)s --module cellular                         # Test cellular architecture only  
  %(prog)s --module integration --output my_results/ # Test integration with custom output
  %(prog)s --all                                     # Run complete validation suite
        """
    )
    
    parser.add_argument('--all', action='store_true',
                       help='Run complete validation suite')
    parser.add_argument('--quick', action='store_true',
                       help='Run quick demo with minimal parameters')
    parser.add_argument('--module', type=str,
                       choices=['oscillatory', 's_entropy', 'cellular', 'virtual_cell', 'evidence', 'integration'],
                       help='Run validation for specific module only')
    parser.add_argument('--output', type=str, default='validation_demo_results',
                       help='Output directory for results (default: validation_demo_results)')
    
    args = parser.parse_args()
    
    # Create output directory
    Path(args.output).mkdir(parents=True, exist_ok=True)
    
    try:
        if args.quick:
            run_quick_demo(args.output)
        elif args.all:
            run_complete_suite_demo(args.output)
        elif args.module:
            run_single_module_demo(args.module, args.output)
        else:
            # Default: run quick demo
            print("No specific option provided. Running quick demo...")
            print("Use --help to see all available options.\n")
            run_quick_demo(args.output)
            
    except KeyboardInterrupt:
        print("\n\n⏹️  Demo interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Demo failed with error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
