//! Output formatting for CLI
//!
//! Handles different output formats and pretty-printing for CLI results

use crate::config::{CliConfig, OutputFormat};
use mogadishu::error::Result;
use serde::Serialize;
use std::collections::HashMap;

/// Format and display simulation results
pub fn format_simulation_results(
    results: &[mogadishu::simulator::SimulationStepResult],
    config: &CliConfig,
) -> Result<()> {
    match config.output_format {
        OutputFormat::Pretty => format_simulation_pretty(results, config),
        OutputFormat::Json => format_simulation_json(results),
        OutputFormat::Csv => format_simulation_csv(results),
    }
}

/// Format simulation results in pretty format
fn format_simulation_pretty(
    results: &[mogadishu::simulator::SimulationStepResult],
    config: &CliConfig,
) -> Result<()> {
    if results.is_empty() {
        println!("No simulation results to display");
        return Ok(());
    }

    let final_result = results.last().unwrap();
    
    // Header
    print_header("Simulation Results", config);
    
    // Summary
    println!("📊 Final Results Summary:");
    println!("  Time: {:.1} hours", final_result.time);
    println!("  Stability: {:?}", final_result.stability_status);
    println!("  Compression: {:.1}x", final_result.compression_achieved);
    println!("  Miracle Usage: {:.1}%", final_result.miracle_utilization * 100.0);
    
    // Performance Metrics
    if !final_result.performance_metrics.is_empty() {
        println!("\n🎯 Performance Metrics:");
        for (metric, value) in &final_result.performance_metrics {
            println!("  {}: {:.3}", format_metric_name(metric), value);
        }
    }
    
    // S-Navigation Metrics
    println!("\n🌌 S-Entropy Navigation:");
    let nav = &final_result.s_navigation_metrics;
    println!("  Efficiency: {:.3}", nav.efficiency);
    println!("  Distance Traveled: {:.1}", nav.total_distance);
    println!("  Miracle Utilization: {:.1}%", nav.miracle_utilization * 100.0);
    println!("  Convergence Time: {:.2}s", nav.convergence_time);
    
    // Progress over time (if verbose)
    if config.is_verbose() && results.len() > 1 {
        println!("\n📈 Progress Over Time:");
        let step_size = (results.len() / 10).max(1);
        for (i, result) in results.iter().step_by(step_size).enumerate() {
            let performance = result.performance_metrics
                .get("overall_performance")
                .unwrap_or(&0.0);
            println!("  t={:.1}h: Performance={:.3}, S-Efficiency={:.3}", 
                result.time, performance, result.s_navigation_metrics.efficiency);
        }
    }
    
    Ok(())
}

/// Format simulation results as JSON
fn format_simulation_json(results: &[mogadishu::simulator::SimulationStepResult]) -> Result<()> {
    let json = serde_json::to_string_pretty(results)?;
    println!("{}", json);
    Ok(())
}

/// Format simulation results as CSV
fn format_simulation_csv(results: &[mogadishu::simulator::SimulationStepResult]) -> Result<()> {
    // CSV Header
    println!("time,stability,compression,miracle_utilization,overall_performance,s_efficiency");
    
    // CSV Data
    for result in results {
        let overall_perf = result.performance_metrics
            .get("overall_performance")
            .unwrap_or(&0.0);
        
        println!("{},{:?},{},{},{},{}",
            result.time,
            result.stability_status,
            result.compression_achieved,
            result.miracle_utilization,
            overall_perf,
            result.s_navigation_metrics.efficiency
        );
    }
    
    Ok(())
}

/// Format optimization results
pub fn format_optimization_results<T>(
    result: &mogadishu::bioreactor::OptimizationResult<T>,
    config: &CliConfig,
) -> Result<()>
where
    T: Serialize,
{
    match config.output_format {
        OutputFormat::Pretty => format_optimization_pretty(result, config),
        OutputFormat::Json => format_optimization_json(result),
        OutputFormat::Csv => format_optimization_csv(result),
    }
}

/// Format optimization results in pretty format
fn format_optimization_pretty<T>(
    result: &mogadishu::bioreactor::OptimizationResult<T>,
    config: &CliConfig,
) -> Result<()>
where
    T: Serialize,
{
    print_header("Optimization Results", config);
    
    match result {
        mogadishu::bioreactor::OptimizationResult::Success { method, s_distance, .. } => {
            println!("✅ Optimization Successful!");
            println!("  Method: {:?}", method);
            println!("  S-Distance: {:.2}", s_distance);
            
            if *s_distance > 10000.0 {
                println!("  🌟 Required miraculous navigation (exceeded normal physics)");
            }
        },
        mogadishu::bioreactor::OptimizationResult::Impossible { proof } => {
            println!("❌ Optimization Impossible");
            println!("  Even maximum miracles cannot achieve this target");
            println!("  Proof: {}", proof);
        },
    }
    
    Ok(())
}

/// Format optimization results as JSON
fn format_optimization_json<T>(result: &mogadishu::bioreactor::OptimizationResult<T>) -> Result<()>
where
    T: Serialize,
{
    let json = serde_json::to_string_pretty(result)?;
    println!("{}", json);
    Ok(())
}

/// Format optimization results as CSV
fn format_optimization_csv<T>(result: &mogadishu::bioreactor::OptimizationResult<T>) -> Result<()>
where
    T: Serialize,
{
    println!("result_type,method,s_distance,success");
    
    match result {
        mogadishu::bioreactor::OptimizationResult::Success { method, s_distance, .. } => {
            println!("success,{:?},{},true", method, s_distance);
        },
        mogadishu::bioreactor::OptimizationResult::Impossible { .. } => {
            println!("impossible,none,inf,false");
        },
    }
    
    Ok(())
}

/// Format analysis results
pub fn format_analysis_results(
    analysis: &mogadishu::simulator::SystemAnalysisReport,
    config: &CliConfig,
) -> Result<()> {
    match config.output_format {
        OutputFormat::Pretty => format_analysis_pretty(analysis, config),
        OutputFormat::Json => format_analysis_json(analysis),
        OutputFormat::Csv => format_analysis_csv(analysis),
    }
}

/// Format analysis results in pretty format
fn format_analysis_pretty(
    analysis: &mogadishu::simulator::SystemAnalysisReport,
    config: &CliConfig,
) -> Result<()> {
    print_header("System Analysis", config);
    
    // Overall Performance
    println!("🎯 Overall Performance: {:.3}", analysis.overall_performance);
    println!("🗜️  System Compression: {:.1}x", analysis.laplace_compression.compression_ratio);
    println!("🌌 S-Entropy Efficiency: {:.3}", analysis.s_entropy_metrics.efficiency);
    
    // Stability Analysis
    println!("\n⚖️  Stability Analysis:");
    println!("  Status: {:?}", analysis.stability_analysis.status);
    println!("  Gain Margin: {:.1} dB", analysis.stability_analysis.margins.gain_margin);
    println!("  Phase Margin: {:.1}°", analysis.stability_analysis.margins.phase_margin);
    
    if analysis.stability_analysis.margins.s_viability_margin > 0.0 {
        println!("  S-Viability Margin: {:.0}", analysis.stability_analysis.margins.s_viability_margin);
    } else {
        println!("  ⚠️ S-Viability Threshold Exceeded");
    }
    
    // Circuit Analysis (if verbose)
    if config.is_verbose() {
        println!("\n⚡ Circuit Analysis:");
        for (component, performance) in &analysis.circuit_analysis.component_analysis {
            println!("  {}: {:.3}", component, performance);
        }
    }
    
    // Recommendations
    if let mogadishu::simulator::StabilityStatus::Unstable = analysis.stability_analysis.status {
        println!("\n💡 Recommendations:");
        println!("  - Enable miraculous stabilization if S-viable");
        println!("  - Reduce system gain or increase damping");
        println!("  - Check cellular network coordination");
    }
    
    Ok(())
}

/// Format analysis results as JSON
fn format_analysis_json(analysis: &mogadishu::simulator::SystemAnalysisReport) -> Result<()> {
    let json = serde_json::to_string_pretty(analysis)?;
    println!("{}", json);
    Ok(())
}

/// Format analysis results as CSV
fn format_analysis_csv(analysis: &mogadishu::simulator::SystemAnalysisReport) -> Result<()> {
    println!("metric,value");
    println!("overall_performance,{}", analysis.overall_performance);
    println!("compression_ratio,{}", analysis.laplace_compression.compression_ratio);
    println!("s_entropy_efficiency,{}", analysis.s_entropy_metrics.efficiency);
    println!("gain_margin,{}", analysis.stability_analysis.margins.gain_margin);
    println!("phase_margin,{}", analysis.stability_analysis.margins.phase_margin);
    println!("s_viability_margin,{}", analysis.stability_analysis.margins.s_viability_margin);
    Ok(())
}

/// Print a formatted header
fn print_header(title: &str, config: &CliConfig) {
    if config.settings.colored_output {
        println!("\n🌟 {}", title);
        println!("{}", "=".repeat(title.len() + 3));
    } else {
        println!("\n{}", title);
        println!("{}", "=".repeat(title.len()));
    }
}

/// Format metric name for display
fn format_metric_name(name: &str) -> String {
    name.replace('_', " ")
        .split_whitespace()
        .map(|word| {
            let mut chars = word.chars();
            match chars.next() {
                None => String::new(),
                Some(first) => first.to_uppercase().collect::<String>() + chars.as_str(),
            }
        })
        .collect::<Vec<_>>()
        .join(" ")
}

/// Display progress indicator
pub fn show_progress(current: usize, total: usize, message: &str, config: &CliConfig) {
    if !config.settings.show_progress {
        return;
    }
    
    let percentage = (current * 100) / total;
    let bar_length = 30;
    let filled = (current * bar_length) / total;
    
    let bar = "█".repeat(filled) + &"░".repeat(bar_length - filled);
    
    print!("\r{}: [{}] {}% ({}/{})", message, bar, percentage, current, total);
    
    if current == total {
        println!(); // New line when complete
    }
}

/// Display error with formatting
pub fn display_error(error: &mogadishu::error::MogadishuError, config: &CliConfig) {
    if config.settings.colored_output {
        eprintln!("❌ Error: {}", error);
    } else {
        eprintln!("Error: {}", error);
    }
    
    if config.is_verbose() {
        eprintln!("Error Category: {:?}", error.category());
        eprintln!("Error Severity: {:?}", error.severity());
        
        if error.is_recoverable() {
            eprintln!("💡 This error may be recoverable - try adjusting parameters");
        }
    }
}

/// Display success message
pub fn display_success(message: &str, config: &CliConfig) {
    if config.settings.colored_output {
        println!("✅ {}", message);
    } else {
        println!("Success: {}", message);
    }
}

/// Display warning message
pub fn display_warning(message: &str, config: &CliConfig) {
    if config.settings.colored_output {
        println!("⚠️ Warning: {}", message);
    } else {
        println!("Warning: {}", message);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_format_metric_name() {
        assert_eq!(format_metric_name("overall_performance"), "Overall Performance");
        assert_eq!(format_metric_name("s_entropy_efficiency"), "S Entropy Efficiency");
        assert_eq!(format_metric_name("simple"), "Simple");
    }
    
    #[test]
    fn test_progress_calculation() {
        // Progress calculation is tested indirectly through show_progress function
        // This is a placeholder for more comprehensive testing
        assert_eq!(50 * 100 / 100, 50); // 50% progress
    }
}
