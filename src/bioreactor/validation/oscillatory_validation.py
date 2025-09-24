"""
Oscillatory Theorem Validation
==============================

Validates the Universal Oscillatory Framework:
1. Everything oscillates at fundamental level
2. Bounded systems with nonlinear coupling exhibit oscillatory behavior  
3. Quantum-classical systems are intrinsically oscillatory
4. Frequency domain modeling captures reality better than time domain
"""

import numpy as np
import pandas as pd
from scipy import signal, fft
from scipy.integrate import odeint, solve_ivp
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Tuple, Callable
from dataclasses import dataclass

from .validation_framework import ValidationFramework, ValidationResult


@dataclass 
class OscillatoryParameters:
    """Parameters for oscillatory system testing"""
    
    # System parameters
    system_size: int = 1000
    coupling_strength: float = 0.1
    nonlinearity_power: float = 2.0
    
    # Simulation parameters  
    time_span: Tuple[float, float] = (0, 100)
    dt: float = 0.01
    
    # Analysis parameters
    frequency_resolution: int = 1000
    oscillation_threshold: float = 0.01
    coherence_threshold: float = 0.7


class OscillatorySystemValidator(ValidationFramework):
    """Validates the Universal Oscillatory Framework"""
    
    def __init__(self, **kwargs):
        super().__init__("OscillatorySystem", **kwargs)
        
    def validate_theorem(self, **kwargs) -> ValidationResult:
        """
        Validate main oscillatory theorem:
        'Every dynamical system with bounded phase space volume and 
         nonlinear coupling exhibits oscillatory behaviour'
        """
        
        params = OscillatoryParameters(**kwargs)
        
        if self.verbose:
            print("🌊 Validating Universal Oscillatory Theorem")
            
        # Test different system types
        systems = {
            "linear_harmonic": self._create_linear_harmonic_system,
            "nonlinear_coupled": self._create_nonlinear_coupled_system, 
            "quantum_classical": self._create_quantum_classical_system,
            "biological_membrane": self._create_biological_membrane_system
        }
        
        all_oscillatory = True
        quantitative_results = {}
        raw_data = {}
        processed_data = {}
        supporting_evidence = []
        contradictory_evidence = []
        plot_paths = []
        
        for system_name, system_creator in systems.items():
            
            if self.verbose:
                print(f"  Testing {system_name}...")
                
            # Create and simulate system
            system = system_creator(params)
            time_data, state_data = self._simulate_system(system, params)
            
            # Analyze oscillatory behavior
            oscillatory_analysis = self._analyze_oscillatory_behavior(
                time_data, state_data, system_name
            )
            
            # Store results
            raw_data[f"{system_name}_time"] = time_data
            raw_data[f"{system_name}_state"] = state_data
            processed_data[f"{system_name}_frequency"] = oscillatory_analysis['frequencies']
            processed_data[f"{system_name}_power"] = oscillatory_analysis['power_spectrum']
            
            quantitative_results[f"{system_name}_oscillation_strength"] = oscillatory_analysis['oscillation_strength']
            quantitative_results[f"{system_name}_dominant_frequency"] = oscillatory_analysis['dominant_frequency']
            quantitative_results[f"{system_name}_coherence"] = oscillatory_analysis['coherence']
            
            # Check if system is oscillatory
            is_oscillatory = (oscillatory_analysis['oscillation_strength'] > params.oscillation_threshold and
                            oscillatory_analysis['coherence'] > params.coherence_threshold)
            
            if is_oscillatory:
                supporting_evidence.append(
                    f"{system_name}: Oscillation strength {oscillatory_analysis['oscillation_strength']:.3f}, "
                    f"coherence {oscillatory_analysis['coherence']:.3f}"
                )
            else:
                contradictory_evidence.append(
                    f"{system_name}: Failed oscillation test - "
                    f"strength {oscillatory_analysis['oscillation_strength']:.3f}, "
                    f"coherence {oscillatory_analysis['coherence']:.3f}"
                )
                all_oscillatory = False
            
            # Create validation plot
            plot_path = self._create_oscillatory_validation_plot(
                time_data, state_data, oscillatory_analysis, system_name
            )
            plot_paths.append(plot_path)
        
        # Create summary comparison plot
        summary_plot = self._create_oscillatory_summary_plot(processed_data)
        plot_paths.append(summary_plot)
        
        # Calculate overall confidence
        oscillation_strengths = [quantitative_results[k] for k in quantitative_results 
                               if 'oscillation_strength' in k]
        confidence_level = np.mean(oscillation_strengths)
        
        return ValidationResult(
            test_name="oscillatory_theorem_validation",
            theorem_validated="Universal Oscillatory Framework - Bounded systems with nonlinear coupling exhibit oscillatory behavior",
            timestamp=self._get_timestamp(),
            parameters=kwargs,
            success=all_oscillatory,
            quantitative_results=quantitative_results,
            statistical_significance={},
            supporting_evidence=supporting_evidence,
            contradictory_evidence=contradictory_evidence,
            raw_data=raw_data,
            processed_data=processed_data,
            plot_paths=plot_paths,
            notes=f"Validated {len(systems)} different system types. "
                  f"Average oscillation strength: {np.mean(oscillation_strengths):.3f}",
            confidence_level=confidence_level
        )
    
    def _create_linear_harmonic_system(self, params: OscillatoryParameters) -> Callable:
        """Create simple harmonic oscillator system"""
        
        def harmonic_system(t, y):
            # Simple harmonic oscillator: d²x/dt² + ω²x = 0
            # State: [position, velocity]
            omega = 1.0
            return [y[1], -omega**2 * y[0]]
            
        return harmonic_system
    
    def _create_nonlinear_coupled_system(self, params: OscillatoryParameters) -> Callable:
        """Create nonlinear coupled oscillator system"""
        
        def coupled_system(t, y):
            # Van der Pol oscillators with coupling
            mu = 0.5
            coupling = params.coupling_strength
            
            n = len(y) // 2
            dydt = np.zeros_like(y)
            
            for i in range(n):
                x_i = y[2*i]
                v_i = y[2*i + 1]
                
                # Van der Pol dynamics
                dydt[2*i] = v_i
                dydt[2*i + 1] = mu * (1 - x_i**2) * v_i - x_i
                
                # Nonlinear coupling to neighbors
                if i > 0:
                    x_left = y[2*(i-1)]
                    dydt[2*i + 1] += coupling * (x_left**params.nonlinearity_power - x_i**params.nonlinearity_power)
                if i < n - 1:
                    x_right = y[2*(i+1)]
                    dydt[2*i + 1] += coupling * (x_right**params.nonlinearity_power - x_i**params.nonlinearity_power)
            
            return dydt
            
        return coupled_system
    
    def _create_quantum_classical_system(self, params: OscillatoryParameters) -> Callable:
        """Create quantum-classical hybrid system"""
        
        def quantum_classical_system(t, y):
            # Quantum harmonic oscillator coupled to classical system
            # State: [q_classical, p_classical, |ψ|² quantum components...]
            
            omega_c = 1.0  # Classical frequency
            omega_q = 1.5  # Quantum frequency  
            coupling = params.coupling_strength
            
            dydt = np.zeros_like(y)
            
            # Classical part
            q_c, p_c = y[0], y[1]
            dydt[0] = p_c
            dydt[1] = -omega_c**2 * q_c
            
            # Quantum part (simplified as coupled oscillators)
            for i in range(2, len(y), 2):
                q_q = y[i]
                p_q = y[i+1] if i+1 < len(y) else 0
                
                dydt[i] = p_q
                dydt[i+1] = -omega_q**2 * q_q + coupling * q_c * np.sin(omega_q * t)
                
                # Back-reaction on classical system
                dydt[1] += coupling * q_q * np.sin(omega_c * t)
            
            return dydt
            
        return quantum_classical_system
    
    def _create_biological_membrane_system(self, params: OscillatoryParameters) -> Callable:
        """Create simplified biological membrane oscillator"""
        
        def membrane_system(t, y):
            # Simplified membrane potential oscillations with ion channels
            # State: [voltage, Na_current, K_current, Ca_current]
            
            V, I_Na, I_K, I_Ca = y[:4]
            
            # Membrane capacitance and conductances
            C_m = 1.0
            g_Na, g_K, g_Ca = 120.0, 36.0, 0.3
            E_Na, E_K, E_Ca = 50.0, -77.0, 132.0
            
            # Hodgkin-Huxley style dynamics with coupling
            alpha_m = 0.1 * (V + 40) / (1 - np.exp(-(V + 40)/10))
            beta_m = 4 * np.exp(-(V + 65)/18)
            m_inf = alpha_m / (alpha_m + beta_m)
            
            alpha_h = 0.07 * np.exp(-(V + 65)/20)
            beta_h = 1 / (1 + np.exp(-(V + 35)/10))
            h_inf = alpha_h / (alpha_h + beta_h)
            
            alpha_n = 0.01 * (V + 55) / (1 - np.exp(-(V + 55)/10))
            beta_n = 0.125 * np.exp(-(V + 65)/80)
            n_inf = alpha_n / (alpha_n + beta_n)
            
            # Currents with nonlinear coupling
            I_Na_new = g_Na * m_inf**3 * h_inf * (V - E_Na)
            I_K_new = g_K * n_inf**4 * (V - E_K)
            I_Ca_new = g_Ca * (V - E_Ca)
            
            # Voltage equation with nonlinear coupling
            dV_dt = (-I_Na_new - I_K_new - I_Ca_new + 
                    params.coupling_strength * (I_Na**params.nonlinearity_power - 
                                               I_K**params.nonlinearity_power)) / C_m
            
            return [dV_dt, 
                   (I_Na_new - I_Na) / 0.1,  # Current dynamics
                   (I_K_new - I_K) / 0.1,
                   (I_Ca_new - I_Ca) / 0.1]
            
        return membrane_system
    
    def _simulate_system(self, system: Callable, params: OscillatoryParameters) -> Tuple[np.ndarray, np.ndarray]:
        """Simulate dynamic system"""
        
        # Time points
        t_eval = np.arange(params.time_span[0], params.time_span[1], params.dt)
        
        # Initial conditions (random small perturbations)
        np.random.seed(42)  # Reproducible results
        if 'harmonic' in system.__name__:
            y0 = [1.0, 0.0]  # [position, velocity]
        elif 'coupled' in system.__name__:
            n_oscillators = 5
            y0 = np.random.normal(0, 0.1, 2 * n_oscillators)
        elif 'quantum' in system.__name__:
            y0 = np.random.normal(0, 0.1, 10)
        elif 'membrane' in system.__name__:
            y0 = [-65.0, 0.0, 0.0, 0.0]  # [V, I_Na, I_K, I_Ca]
        else:
            y0 = np.random.normal(0, 0.1, 4)
        
        # Solve system
        try:
            sol = solve_ivp(system, params.time_span, y0, t_eval=t_eval, 
                          method='RK45', rtol=1e-8, atol=1e-10)
            
            if sol.success:
                return sol.t, sol.y
            else:
                self.logger.warning(f"Integration failed: {sol.message}")
                return t_eval, np.zeros((len(y0), len(t_eval)))
                
        except Exception as e:
            self.logger.error(f"Simulation failed: {str(e)}")
            return t_eval, np.zeros((len(y0), len(t_eval)))
    
    def _analyze_oscillatory_behavior(self, 
                                    time_data: np.ndarray, 
                                    state_data: np.ndarray,
                                    system_name: str) -> Dict[str, Any]:
        """Analyze oscillatory behavior in system response"""
        
        # Take first component for analysis
        signal_data = state_data[0, :]
        
        # Remove transients (first 20% of data)
        start_idx = int(0.2 * len(signal_data))
        analysis_signal = signal_data[start_idx:]
        analysis_time = time_data[start_idx:]
        
        # Frequency domain analysis
        dt = np.mean(np.diff(analysis_time))
        frequencies, power_spectrum = signal.welch(
            analysis_signal, fs=1/dt, nperseg=len(analysis_signal)//4
        )
        
        # Find dominant frequency
        dominant_idx = np.argmax(power_spectrum)
        dominant_frequency = frequencies[dominant_idx]
        
        # Oscillation strength (ratio of peak power to mean power)
        oscillation_strength = power_spectrum[dominant_idx] / np.mean(power_spectrum)
        
        # Coherence measure (autocorrelation decay)
        autocorr = np.correlate(analysis_signal, analysis_signal, mode='full')
        autocorr = autocorr[autocorr.size // 2:]
        autocorr = autocorr / autocorr[0]  # Normalize
        
        # Find first zero crossing or significant decay
        coherence_length = 1.0
        for i, val in enumerate(autocorr[1:]):
            if val < 0.1:  # 10% of original correlation
                coherence_length = i * dt
                break
        
        # Coherence is inverse of decay rate
        coherence = 1.0 / (1.0 + coherence_length)
        
        # Spectral entropy (measure of frequency spread)
        normalized_psd = power_spectrum / np.sum(power_spectrum)
        spectral_entropy = -np.sum(normalized_psd * np.log2(normalized_psd + 1e-10))
        
        return {
            'frequencies': frequencies,
            'power_spectrum': power_spectrum,
            'dominant_frequency': dominant_frequency,
            'oscillation_strength': oscillation_strength,
            'coherence': coherence,
            'spectral_entropy': spectral_entropy,
            'autocorr': autocorr
        }
    
    def _create_oscillatory_validation_plot(self, 
                                          time_data: np.ndarray,
                                          state_data: np.ndarray, 
                                          analysis: Dict[str, Any],
                                          system_name: str) -> str:
        """Create validation plot for oscillatory system"""
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Time series
        axes[0,0].plot(time_data, state_data[0, :], 'b-', linewidth=1.5, alpha=0.8)
        axes[0,0].set_title(f'{system_name}: Time Series')
        axes[0,0].set_xlabel('Time')
        axes[0,0].set_ylabel('Amplitude')
        axes[0,0].grid(True, alpha=0.3)
        
        # Phase space (if multi-dimensional)
        if state_data.shape[0] >= 2:
            axes[0,1].plot(state_data[0, :], state_data[1, :], 'r-', alpha=0.7)
            axes[0,1].set_title(f'{system_name}: Phase Space')
            axes[0,1].set_xlabel('Position')
            axes[0,1].set_ylabel('Velocity')
        else:
            # Delay embedding for 1D signal
            delay = 10
            embedded = np.array([state_data[0, :-delay], state_data[0, delay:]])
            axes[0,1].plot(embedded[0], embedded[1], 'r-', alpha=0.7)
            axes[0,1].set_title(f'{system_name}: Delay Embedding')
            axes[0,1].set_xlabel('x(t)')
            axes[0,1].set_ylabel('x(t+τ)')
        axes[0,1].grid(True, alpha=0.3)
        
        # Frequency spectrum
        axes[1,0].semilogy(analysis['frequencies'], analysis['power_spectrum'], 'g-', linewidth=2)
        axes[1,0].axvline(analysis['dominant_frequency'], color='red', linestyle='--', 
                         label=f"Dominant: {analysis['dominant_frequency']:.2f} Hz")
        axes[1,0].set_title(f'{system_name}: Power Spectrum')
        axes[1,0].set_xlabel('Frequency (Hz)')
        axes[1,0].set_ylabel('Power Spectral Density')
        axes[1,0].legend()
        axes[1,0].grid(True, alpha=0.3)
        
        # Autocorrelation
        max_lag = min(500, len(analysis['autocorr']))
        lags = np.arange(max_lag) * (time_data[1] - time_data[0])
        axes[1,1].plot(lags, analysis['autocorr'][:max_lag], 'm-', linewidth=2)
        axes[1,1].axhline(0.1, color='red', linestyle='--', alpha=0.7, 
                         label='10% threshold')
        axes[1,1].set_title(f'{system_name}: Autocorrelation')
        axes[1,1].set_xlabel('Lag Time')
        axes[1,1].set_ylabel('Correlation')
        axes[1,1].legend()
        axes[1,1].grid(True, alpha=0.3)
        
        # Add analysis metrics
        metrics_text = (f"Oscillation Strength: {analysis['oscillation_strength']:.2f}\n"
                       f"Coherence: {analysis['coherence']:.3f}\n"
                       f"Spectral Entropy: {analysis['spectral_entropy']:.2f}")
        
        fig.text(0.02, 0.98, metrics_text, transform=fig.transFigure, 
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        
        plt.tight_layout()
        
        # Save plot
        plot_path = self.output_dir / "plots" / f"oscillatory_validation_{system_name}.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(plot_path)
    
    def _create_oscillatory_summary_plot(self, processed_data: Dict[str, np.ndarray]) -> str:
        """Create summary comparison plot"""
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Extract system names and data
        systems = []
        for key in processed_data.keys():
            if '_frequency' in key:
                systems.append(key.replace('_frequency', ''))
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(systems)))
        
        # Combined power spectra
        for i, system in enumerate(systems):
            freq_key = f"{system}_frequency"
            power_key = f"{system}_power"
            
            if freq_key in processed_data and power_key in processed_data:
                frequencies = processed_data[freq_key]
                power = processed_data[power_key]
                
                # Limit to reasonable frequency range
                max_freq_idx = len(frequencies) // 10  # First 10% of frequencies
                
                axes[0,0].semilogy(frequencies[:max_freq_idx], power[:max_freq_idx], 
                                  color=colors[i], label=system.replace('_', ' ').title(), 
                                  linewidth=2, alpha=0.8)
        
        axes[0,0].set_title('Power Spectra Comparison')
        axes[0,0].set_xlabel('Frequency (Hz)')
        axes[0,0].set_ylabel('Power Spectral Density')
        axes[0,0].legend()
        axes[0,0].grid(True, alpha=0.3)
        
        # Oscillation strength comparison
        oscillation_strengths = []
        for system in systems:
            # Extract from processed data or calculate
            # This would require storing oscillation strengths separately
            # For now, calculate approximate from power spectrum
            power_key = f"{system}_power"
            if power_key in processed_data:
                power = processed_data[power_key]
                strength = np.max(power) / np.mean(power) if len(power) > 0 else 0
                oscillation_strengths.append(strength)
            else:
                oscillation_strengths.append(0)
        
        axes[0,1].bar(range(len(systems)), oscillation_strengths, color=colors, alpha=0.7)
        axes[0,1].set_title('Oscillation Strength Comparison')
        axes[0,1].set_xlabel('System')
        axes[0,1].set_ylabel('Oscillation Strength')
        axes[0,1].set_xticks(range(len(systems)))
        axes[0,1].set_xticklabels([s.replace('_', '\n') for s in systems], rotation=45)
        axes[0,1].grid(True, alpha=0.3)
        
        # Frequency distribution
        dominant_frequencies = []
        for system in systems:
            power_key = f"{system}_power"
            freq_key = f"{system}_frequency"
            if power_key in processed_data and freq_key in processed_data:
                power = processed_data[power_key]
                freq = processed_data[freq_key]
                if len(power) > 0 and len(freq) > 0:
                    dom_freq = freq[np.argmax(power)]
                    dominant_frequencies.append(dom_freq)
                else:
                    dominant_frequencies.append(0)
            else:
                dominant_frequencies.append(0)
        
        axes[1,0].bar(range(len(systems)), dominant_frequencies, color=colors, alpha=0.7)
        axes[1,0].set_title('Dominant Frequency Comparison')
        axes[1,0].set_xlabel('System')
        axes[1,0].set_ylabel('Dominant Frequency (Hz)')
        axes[1,0].set_xticks(range(len(systems)))
        axes[1,0].set_xticklabels([s.replace('_', '\n') for s in systems], rotation=45)
        axes[1,0].grid(True, alpha=0.3)
        
        # Summary metrics
        axes[1,1].axis('off')
        
        summary_text = "Oscillatory Theorem Validation Summary\n\n"
        for i, system in enumerate(systems):
            summary_text += f"{system.replace('_', ' ').title()}:\n"
            summary_text += f"  Oscillation Strength: {oscillation_strengths[i]:.2f}\n"
            summary_text += f"  Dominant Frequency: {dominant_frequencies[i]:.2f} Hz\n\n"
        
        axes[1,1].text(0.1, 0.9, summary_text, transform=axes[1,1].transAxes,
                      fontsize=12, verticalalignment='top',
                      bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.3))
        
        plt.tight_layout()
        
        # Save plot
        plot_path = self.output_dir / "plots" / "oscillatory_summary_comparison.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(plot_path)
    
    def _get_additional_tests(self) -> Dict[str, Callable]:
        """Additional oscillatory framework tests"""
        
        return {
            "frequency_domain_superiority": self._test_frequency_domain_superiority,
            "oscillatory_hierarchy": self._test_oscillatory_hierarchy,
            "coherence_preservation": self._test_coherence_preservation
        }
    
    def _test_frequency_domain_superiority(self, **kwargs) -> ValidationResult:
        """Test that frequency domain modeling is superior to time domain"""
        
        if self.verbose:
            print("  Testing frequency domain modeling superiority...")
        
        # Create test signal with multiple frequencies and noise
        t = np.linspace(0, 10, 1000)
        signal_true = (np.sin(2*np.pi*1*t) + 0.5*np.sin(2*np.pi*3*t) + 
                      0.3*np.sin(2*np.pi*7*t))
        noise = np.random.normal(0, 0.2, len(t))
        signal_noisy = signal_true + noise
        
        # Time domain analysis (correlation with true signal)
        time_correlation = np.corrcoef(signal_true, signal_noisy)[0,1]
        
        # Frequency domain analysis
        freq_true = fft.fftfreq(len(t), t[1]-t[0])
        fft_true = fft.fft(signal_true)
        fft_noisy = fft.fft(signal_noisy)
        
        # Compare power spectra (more robust to noise)
        power_true = np.abs(fft_true)**2
        power_noisy = np.abs(fft_noisy)**2
        
        # Normalize power spectra
        power_true_norm = power_true / np.sum(power_true)
        power_noisy_norm = power_noisy / np.sum(power_noisy)
        
        freq_correlation = np.corrcoef(power_true_norm, power_noisy_norm)[0,1]
        
        # Frequency domain should be more robust to noise
        superiority_factor = freq_correlation / time_correlation
        
        success = superiority_factor > 1.1  # At least 10% better
        
        # Create validation plot
        plot_path = self.create_validation_plot(
            {
                'time': t,
                'signal_true': signal_true,
                'signal_noisy': signal_noisy,
                'frequency': freq_true[:len(freq_true)//2],
                'power_true': power_true[:len(power_true)//2],
                'power_noisy': power_noisy[:len(power_noisy)//2]
            },
            plot_type="comparison",
            title="Frequency vs Time Domain Analysis",
            save_name="frequency_domain_superiority"
        )
        
        return ValidationResult(
            test_name="frequency_domain_superiority",
            theorem_validated="Frequency domain modeling superiority over time domain",
            timestamp=self._get_timestamp(),
            parameters=kwargs,
            success=success,
            quantitative_results={
                "time_domain_correlation": time_correlation,
                "frequency_domain_correlation": freq_correlation, 
                "superiority_factor": superiority_factor
            },
            statistical_significance={},
            supporting_evidence=[
                f"Frequency domain correlation: {freq_correlation:.3f}",
                f"Time domain correlation: {time_correlation:.3f}",
                f"Superiority factor: {superiority_factor:.3f}"
            ] if success else [],
            contradictory_evidence=[] if success else [
                f"Frequency domain not superior: factor {superiority_factor:.3f}"
            ],
            raw_data={
                'time': t,
                'signal_true': signal_true,
                'signal_noisy': signal_noisy
            },
            processed_data={
                'frequency': freq_true[:len(freq_true)//2],
                'power_true': power_true[:len(power_true)//2],
                'power_noisy': power_noisy[:len(power_noisy)//2]
            },
            plot_paths=[plot_path],
            notes=f"Tested noise robustness of frequency vs time domain analysis",
            confidence_level=min(freq_correlation, superiority_factor/2)
        )
    
    def _test_oscillatory_hierarchy(self, **kwargs) -> ValidationResult:
        """Test nested oscillatory hierarchy"""
        
        if self.verbose:
            print("  Testing oscillatory hierarchy...")
        
        # Create hierarchical oscillatory system
        # Fast oscillations modulated by slower ones
        t = np.linspace(0, 20, 2000)
        
        # Hierarchy of frequencies: 1, 5, 25 Hz
        slow_osc = np.sin(2*np.pi*1*t)           # 1 Hz
        medium_osc = np.sin(2*np.pi*5*t)         # 5 Hz
        fast_osc = np.sin(2*np.pi*25*t)          # 25 Hz
        
        # Hierarchical coupling: fast modulated by medium, medium by slow
        hierarchical_signal = (slow_osc + 
                              0.5 * medium_osc * (1 + 0.3*slow_osc) +
                              0.2 * fast_osc * (1 + 0.3*medium_osc))
        
        # Analyze frequency content
        frequencies = fft.fftfreq(len(t), t[1]-t[0])
        fft_signal = fft.fft(hierarchical_signal)
        power_spectrum = np.abs(fft_signal)**2
        
        # Find peaks at expected frequencies
        freq_range = frequencies[frequencies >= 0]
        power_range = power_spectrum[:len(freq_range)]
        
        expected_freqs = [1.0, 5.0, 25.0]
        found_peaks = []
        
        for expected_freq in expected_freqs:
            # Find closest frequency bin
            freq_idx = np.argmin(np.abs(freq_range - expected_freq))
            
            # Check if it's a significant peak
            local_power = power_range[max(0, freq_idx-2):freq_idx+3]
            max_local_power = np.max(local_power)
            mean_power = np.mean(power_range)
            
            peak_ratio = max_local_power / mean_power
            found_peaks.append(peak_ratio > 5.0)  # 5x above average
        
        hierarchy_preserved = all(found_peaks)
        
        # Create validation plot
        plot_path = self.create_validation_plot(
            {
                'time': t,
                'hierarchical_signal': hierarchical_signal,
                'frequency': freq_range,
                'power_spectrum': power_range
            },
            plot_type="timeseries",
            title="Oscillatory Hierarchy Analysis",
            save_name="oscillatory_hierarchy"
        )
        
        return ValidationResult(
            test_name="oscillatory_hierarchy",
            theorem_validated="Nested oscillatory hierarchies with cross-scale coupling",
            timestamp=self._get_timestamp(),
            parameters=kwargs,
            success=hierarchy_preserved,
            quantitative_results={
                f"peak_ratio_{freq}Hz": power_range[np.argmin(np.abs(freq_range - freq))] / np.mean(power_range)
                for freq in expected_freqs
            },
            statistical_significance={},
            supporting_evidence=[
                f"Found {sum(found_peaks)}/{len(expected_freqs)} expected frequency peaks"
            ] if hierarchy_preserved else [],
            contradictory_evidence=[] if hierarchy_preserved else [
                f"Missing peaks at frequencies: {[f for i, f in enumerate(expected_freqs) if not found_peaks[i]]}"
            ],
            raw_data={
                'time': t,
                'hierarchical_signal': hierarchical_signal
            },
            processed_data={
                'frequency': freq_range,
                'power_spectrum': power_range
            },
            plot_paths=[plot_path],
            notes="Tested 3-level frequency hierarchy with cross-scale modulation",
            confidence_level=sum(found_peaks) / len(expected_freqs)
        )
    
    def _test_coherence_preservation(self, **kwargs) -> ValidationResult:
        """Test oscillatory coherence preservation"""
        
        if self.verbose:
            print("  Testing coherence preservation...")
        
        # Create coupled oscillator system
        n_oscillators = 10
        coupling_strength = 0.1
        
        def coupled_oscillators(t, y):
            # Array of coupled harmonic oscillators
            dydt = np.zeros_like(y)
            
            for i in range(n_oscillators):
                x_i = y[2*i]
                v_i = y[2*i + 1]
                
                # Individual oscillator dynamics
                dydt[2*i] = v_i
                dydt[2*i + 1] = -x_i  # Natural frequency = 1
                
                # Coupling to neighbors
                if i > 0:
                    x_left = y[2*(i-1)]
                    dydt[2*i + 1] += coupling_strength * (x_left - x_i)
                if i < n_oscillators - 1:
                    x_right = y[2*(i+1)]
                    dydt[2*i + 1] += coupling_strength * (x_right - x_i)
            
            return dydt
        
        # Simulate system
        t_span = (0, 50)
        t_eval = np.linspace(0, 50, 5000)
        
        # Random initial conditions
        np.random.seed(42)
        y0 = np.random.normal(0, 0.1, 2*n_oscillators)
        
        sol = solve_ivp(coupled_oscillators, t_span, y0, t_eval=t_eval, 
                       method='RK45', rtol=1e-8)
        
        # Analyze coherence
        positions = sol.y[::2, :]  # Extract positions
        
        # Calculate phase coherence
        phases = np.angle(signal.hilbert(positions, axis=1))
        
        # Coherence parameter R (Kuramoto order parameter)
        coherence_over_time = []
        for t_idx in range(phases.shape[1]):
            phase_vec = np.exp(1j * phases[:, t_idx])
            R = np.abs(np.mean(phase_vec))
            coherence_over_time.append(R)
        
        coherence_over_time = np.array(coherence_over_time)
        
        # Check if coherence is preserved (doesn't decay significantly)
        initial_coherence = np.mean(coherence_over_time[:500])  # First 10%
        final_coherence = np.mean(coherence_over_time[-500:])   # Last 10%
        
        coherence_preservation_ratio = final_coherence / initial_coherence
        coherence_preserved = coherence_preservation_ratio > 0.8  # Less than 20% decay
        
        # Create validation plot
        plot_path = self.create_validation_plot(
            {
                'time': sol.t,
                'coherence': coherence_over_time,
                'position_0': positions[0, :],
                'position_1': positions[1, :] if n_oscillators > 1 else positions[0, :]
            },
            plot_type="timeseries",
            title="Oscillatory Coherence Preservation",
            save_name="coherence_preservation"
        )
        
        return ValidationResult(
            test_name="coherence_preservation",
            theorem_validated="Oscillatory coherence preservation in coupled systems",
            timestamp=self._get_timestamp(),
            parameters=kwargs,
            success=coherence_preserved,
            quantitative_results={
                "initial_coherence": initial_coherence,
                "final_coherence": final_coherence,
                "preservation_ratio": coherence_preservation_ratio,
                "mean_coherence": np.mean(coherence_over_time)
            },
            statistical_significance={},
            supporting_evidence=[
                f"Coherence preserved: {coherence_preservation_ratio:.3f}",
                f"Mean coherence: {np.mean(coherence_over_time):.3f}"
            ] if coherence_preserved else [],
            contradictory_evidence=[] if coherence_preserved else [
                f"Coherence decayed significantly: {coherence_preservation_ratio:.3f}"
            ],
            raw_data={
                'time': sol.t,
                'positions': positions
            },
            processed_data={
                'coherence': coherence_over_time,
                'phases': phases
            },
            plot_paths=[plot_path],
            notes=f"Tested {n_oscillators} coupled oscillators for coherence preservation",
            confidence_level=min(coherence_preservation_ratio, np.mean(coherence_over_time))
        )
    
    def _get_timestamp(self):
        """Get current timestamp"""
        from datetime import datetime, timezone
        return datetime.now(timezone.utc)
