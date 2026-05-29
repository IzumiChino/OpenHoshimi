//! `sat_tool lint` — validate a satellite TOML definition.

use std::path::Path;

use openhoshimi_core::satellite::load_satellite;

use crate::mapping::{chain_for_framing, modem_for_modulation};

/// Run the linter on a satellite TOML file.
pub fn run(path: &Path) -> Result<(), String> {
    let def =
        load_satellite(path).map_err(|err| format!("failed to parse {}: {err}", path.display()))?;

    let mut errors: Vec<String> = Vec::new();
    let mut warnings: Vec<String> = Vec::new();

    // Check each downlink for consistency.
    for dl in &def.downlinks {
        let prefix = format!("[{}]", dl.label);

        // Frequency sanity.
        if dl.freq_hz < 50_000_000 || dl.freq_hz > 50_000_000_000 {
            warnings.push(format!(
                "{prefix} freq_hz={} is outside typical amateur range (50 MHz - 50 GHz)",
                dl.freq_hz
            ));
        }

        // Modulation known?
        if modem_for_modulation(&dl.modulation).is_none()
            && !["FM", "CW", "SSB", "AM"].contains(&dl.modulation.to_ascii_uppercase().as_str())
        {
            warnings.push(format!(
                "{prefix} modulation '{}' is not in the known set",
                dl.modulation
            ));
        }

        // Framing/codec consistency.
        if let Some(chain) = chain_for_framing(&dl.framing) {
            if let Some(ref codec) = dl.codec {
                let codec_str = format!("{codec:?}").to_ascii_lowercase();
                let expected = chain.codec_kind.replace('_', "");
                let actual_normalized = codec_str.replace('_', "");
                if !actual_normalized.contains(&expected) {
                    errors.push(format!(
                        "{prefix} framing='{}' expects codec kind '{}', got {codec:?}",
                        dl.framing, chain.codec_kind
                    ));
                }
            } else {
                warnings.push(format!(
                    "{prefix} framing='{}' typically needs [downlink.codec] kind='{}'",
                    dl.framing, chain.codec_kind
                ));
            }

            // Line coding / descrambler for AX.25 at >1200 baud.
            if chain.line_coding == Some("nrzi") && dl.line_coding.is_none() {
                warnings.push(format!(
                    "{prefix} framing='{}' typically needs [downlink.line_coding] kind='nrzi'",
                    dl.framing
                ));
            }
            if chain.descrambler == Some("g3ruh") && dl.descrambler.is_none() && dl.baudrate > 1200
            {
                warnings.push(format!(
                    "{prefix} framing='{}' at {} baud typically needs [downlink.descrambler] kind='g3ruh'",
                    dl.framing, dl.baudrate
                ));
            }
        }

        // Telemetry schema reference.
        if let Some(ref schema_name) = dl.telemetry_schema {
            if !def.telemetry.contains_key(schema_name) {
                errors.push(format!(
                    "{prefix} telemetry_schema='{schema_name}' not found in [telemetry.*]"
                ));
            }
        }
    }

    // NORAD ID.
    if def.satellite.norad_id == 0 {
        warnings.push("satellite.norad_id is 0 (placeholder?)".to_string());
    }

    // Report.
    for w in &warnings {
        eprintln!("  warn: {w}");
    }
    for e in &errors {
        eprintln!("  ERROR: {e}");
    }

    if errors.is_empty() {
        let n_dl = def.downlinks.len();
        println!(
            "lint: {} OK ({n_dl} downlink{}, {} warning{})",
            path.display(),
            if n_dl == 1 { "" } else { "s" },
            warnings.len(),
            if warnings.len() == 1 { "" } else { "s" }
        );
        Ok(())
    } else {
        Err(format!(
            "{} error{} in {}",
            errors.len(),
            if errors.len() == 1 { "" } else { "s" },
            path.display()
        ))
    }
}
