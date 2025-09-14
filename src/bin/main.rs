//! Mogadishu S-Entropy Framework CLI
//!
//! Command-line interface for the S-entropy bioreactor modeling framework.
//! Provides tools for bioreactor simulation, optimization, and analysis through
//! revolutionary S-entropy navigation principles.

use clap::{Arg, Command};
use std::process;

mod commands;
mod config;
mod output;

use commands::{run_simulate, run_optimize, run_analyze, run_demo};
use config::CliConfig;

fn main() {
    let matches = Command::new("mogadishu-cli")
        .version(mogadishu::VERSION)
        .author("Kundai Farai Sachikonye <kundai.sachikonye@wzw.tum.de>")
        .about("S-Entropy Framework for Revolutionary Bioreactor Modeling")
        .long_about(
            "Mogadishu implements the complete S-Entropy Framework for bioreactor modeling, \
            transforming traditional engineering approaches into computational biological networks \
            that operate according to how cellular systems actually function."
        )
        .arg(
            Arg::new("config")
                .short('c')
                .long("config")
                .value_name("FILE")
                .help("Configuration file path")
        )
        .arg(
            Arg::new("verbose")
                .short('v')
                .long("verbose")
                .action(clap::ArgAction::Count)
                .help("Increase verbosity (use multiple times for more verbose output)")
        )
        .arg(
            Arg::new("format")
                .long("format")
                .value_name("FORMAT")
                .default_value("pretty")
                .value_parser(["pretty", "json", "csv"])
                .help("Output format")
        )
        .subcommand(
            Command::new("simulate")
                .about("Run bioreactor simulation using S-entropy cellular networks")
                .arg(
                    Arg::new("cells")
                        .long("cells")
                        .short('n')
                        .value_name("NUMBER")
                        .default_value("1000")
                        .help("Number of cellular observers in the bioreactor network")
                )
                .arg(
                    Arg::new("time")
                        .long("time")
                        .short('t')
                        .value_name("HOURS")
                        .default_value("24.0")
                        .help("Simulation time in hours")
                )
                .arg(
                    Arg::new("enable-miracles")
                        .long("enable-miracles")
                        .action(clap::ArgAction::SetTrue)
                        .help("Enable miraculous dynamics for impossible performance")
                )
                .arg(
                    Arg::new("oxygen-enhanced")
                        .long("oxygen-enhanced")
                        .action(clap::ArgAction::SetTrue)
                        .help("Enable oxygen-enhanced information processing")
                )
        )
        .subcommand(
            Command::new("optimize")
                .about("Optimize bioreactor performance through S-entropy navigation")
                .arg(
                    Arg::new("target")
                        .long("target")
                        .value_name("METRIC")
                        .required(true)
                        .help("Optimization target (yield, productivity, efficiency)")
                )
                .arg(
                    Arg::new("improvement")
                        .long("improvement")
                        .value_name("FACTOR")
                        .default_value("2.0")
                        .help("Target improvement factor")
                )
                .arg(
                    Arg::new("test-impossibility")
                        .long("test-impossibility")
                        .action(clap::ArgAction::SetTrue)
                        .help("Test if target is absolutely impossible using maximum miracles")
                )
        )
        .subcommand(
            Command::new("analyze")
                .about("Analyze bioreactor data using cellular Bayesian networks")
                .arg(
                    Arg::new("data")
                        .long("data")
                        .short('d')
                        .value_name("FILE")
                        .required(true)
                        .help("Input data file (CSV, JSON, or HDF5)")
                )
                .arg(
                    Arg::new("output")
                        .long("output")
                        .short('o')
                        .value_name("FILE")
                        .help("Output analysis file")
                )
        )
        .subcommand(
            Command::new("demo")
                .about("Run interactive S-entropy demonstrations")
                .arg(
                    Arg::new("type")
                        .value_name("DEMO_TYPE")
                        .default_value("navigation")
                        .value_parser(["navigation", "cellular", "miraculous", "bioreactor"])
                        .help("Type of demonstration to run")
                )
                .arg(
                    Arg::new("interactive")
                        .long("interactive")
                        .short('i')
                        .action(clap::ArgAction::SetTrue)
                        .help("Run in interactive mode")
                )
        )
        .get_matches();

    // Initialize configuration
    let config = CliConfig::new(&matches).unwrap_or_else(|err| {
        eprintln!("Configuration error: {}", err);
        process::exit(1);
    });

    // Set up logging based on verbosity
    let log_level = match matches.get_count("verbose") {
        0 => "info",
        1 => "debug", 
        _ => "trace",
    };
    
    env_logger::Builder::from_env(
        env_logger::Env::default()
            .default_filter_or(format!("mogadishu={}", log_level))
    ).init();

    // Execute subcommands
    let result = match matches.subcommand() {
        Some(("simulate", sub_matches)) => run_simulate(sub_matches, &config),
        Some(("optimize", sub_matches)) => run_optimize(sub_matches, &config),
        Some(("analyze", sub_matches)) => run_analyze(sub_matches, &config),
        Some(("demo", sub_matches)) => run_demo(sub_matches, &config),
        _ => {
            println!("🚀 Mogadishu S-Entropy Framework v{}", mogadishu::VERSION);
            println!("Revolutionary bioreactor modeling through observer-process integration");
            println!("\nUse --help to see available commands");
            Ok(())
        }
    };

    if let Err(err) = result {
        eprintln!("Error: {}", err);
        process::exit(1);
    }
}
