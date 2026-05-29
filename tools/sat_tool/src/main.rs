//! Satellite TOML generator, validator, and importer for OpenHoshimi.

use std::path::PathBuf;
use std::process::ExitCode;

use clap::{Parser, Subcommand};

mod fetch;
mod import;
mod lint;
mod mapping;
mod wizard;

#[derive(Debug, Parser)]
#[command(author, version, about = "Satellite TOML toolkit for OpenHoshimi")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Validate a satellite TOML definition.
    Lint {
        /// Path to the satellite TOML file.
        file: PathBuf,
    },
    /// Import a gr-satellites YAML definition and output OpenHoshimi TOML.
    ImportGrsat {
        /// Path to the gr-satellites YAML file.
        file: PathBuf,
        /// Output path (default: stdout).
        #[arg(short, long)]
        output: Option<PathBuf>,
    },
    /// Interactive wizard to build a satellite TOML step by step.
    Wizard {
        /// Output path (default: stdout).
        #[arg(short, long)]
        output: Option<PathBuf>,
    },
    /// Fetch satellite info from SatNOGS DB and generate a TOML.
    Fetch {
        /// NORAD catalog ID.
        norad_id: u32,
        /// Output path (default: stdout).
        #[arg(short, long)]
        output: Option<PathBuf>,
    },
}

fn main() -> ExitCode {
    let cli = Cli::parse();
    let result = match cli.command {
        Command::Lint { file } => lint::run(&file),
        Command::ImportGrsat { file, output } => import::run(&file, output.as_deref()),
        Command::Wizard { output } => wizard::run(output.as_deref()),
        Command::Fetch { norad_id, output } => fetch::run(norad_id, output.as_deref()),
    };
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(err) => {
            eprintln!("sat_tool: {err}");
            ExitCode::from(1)
        }
    }
}
