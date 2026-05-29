//! `sat_tool import-grsat` — convert gr-satellites YAML to OpenHoshimi TOML.

use std::collections::BTreeMap;
use std::path::Path;

use serde::Deserialize;

use crate::mapping::{chain_for_framing, modem_for_modulation};

#[derive(Debug, Deserialize)]
struct GrSatYaml {
    name: String,
    #[serde(default)]
    alternative_names: Vec<String>,
    #[serde(default)]
    norad: u32,
    #[serde(default)]
    transmitters: BTreeMap<String, GrSatTransmitter>,
}

#[derive(Debug, Deserialize)]
struct GrSatTransmitter {
    #[serde(default)]
    frequency: f64,
    #[serde(default)]
    modulation: String,
    #[serde(default)]
    baudrate: u32,
    #[serde(default)]
    framing: String,
    #[serde(default)]
    deviation: Option<f32>,
    #[serde(default)]
    af_carrier: Option<f32>,
}

/// Run the gr-satellites YAML importer.
pub fn run(path: &Path, output: Option<&Path>) -> Result<(), String> {
    let content = std::fs::read_to_string(path)
        .map_err(|err| format!("failed to read {}: {err}", path.display()))?;
    let yaml: GrSatYaml =
        serde_yaml::from_str(&content).map_err(|err| format!("failed to parse YAML: {err}"))?;

    let mut toml = String::new();
    toml.push_str(&format!(
        "# Generated from gr-satellites: {}\n\n",
        path.display()
    ));
    toml.push_str("[satellite]\n");
    toml.push_str(&format!("name = {:?}\n", yaml.name));
    if !yaml.alternative_names.is_empty() {
        let aliases: Vec<String> = yaml
            .alternative_names
            .iter()
            .map(|s| format!("{s:?}"))
            .collect();
        toml.push_str(&format!("aliases = [{}]\n", aliases.join(", ")));
    }
    toml.push_str(&format!("norad_id = {}\n", yaml.norad));

    for (label, tx) in &yaml.transmitters {
        toml.push('\n');
        toml.push_str("[[downlink]]\n");
        toml.push_str(&format!("label = {label:?}\n"));
        toml.push_str(&format!("freq_hz = {}\n", tx.frequency as u64));
        toml.push_str(&format!(
            "modulation = {:?}\n",
            tx.modulation.to_ascii_uppercase()
        ));
        toml.push_str(&format!("baudrate = {}\n", tx.baudrate));

        let framing_str = if tx.framing.is_empty() {
            "AX25"
        } else {
            &tx.framing
        };
        let chain = chain_for_framing(framing_str);
        let framing_toml = chain
            .as_ref()
            .map_or(framing_str.to_string(), |c| c.framing.to_string());
        toml.push_str(&format!("framing = {framing_toml:?}\n"));

        // Modem section.
        if let Some(modem) = modem_for_modulation(&tx.modulation) {
            toml.push_str("\n[downlink.modem]\n");
            toml.push_str(&format!("kind = {:?}\n", modem.kind));
            if let Some(mode) = modem.mode {
                toml.push_str(&format!("mode = {mode:?}\n"));
            }
            if let Some(mi) = modem.modulation_index {
                toml.push_str(&format!("modulation_index = {mi}\n"));
            }
            if let Some(bt) = modem.gaussian_bt {
                toml.push_str(&format!("gaussian_bt = {bt}\n"));
            }
            if modem.kind == "afsk" {
                let carrier = tx.af_carrier.unwrap_or(1700.0);
                let dev = tx.deviation.unwrap_or(600.0);
                toml.push_str(&format!("mark_hz = {}\n", carrier + dev / 2.0));
                toml.push_str(&format!("space_hz = {}\n", carrier - dev / 2.0));
            } else {
                toml.push_str("frequency_offset_hz = 0.0\n");
                toml.push_str("differential = false\n");
                toml.push_str("invert = false\n");
                toml.push_str("swap_iq = false\n");
            }
        }

        // Chain sections.
        if let Some(ref chain) = chain {
            if let Some(lc) = chain.line_coding {
                toml.push_str("\n[downlink.line_coding]\n");
                toml.push_str(&format!("kind = {lc:?}\n"));
            }
            if let Some(ds) = chain.descrambler {
                if tx.baudrate > 1200 || ds != "g3ruh" {
                    toml.push_str("\n[downlink.descrambler]\n");
                    toml.push_str(&format!("kind = {ds:?}\n"));
                }
            }
            if let Some(fk) = chain.framer_kind {
                toml.push_str("\n[downlink.framer]\n");
                toml.push_str(&format!("kind = {fk:?}\n"));
                if fk == "ax100_asm" || fk == "ao40" {
                    toml.push_str("threshold = 4\n");
                }
            }
            toml.push_str("\n[downlink.codec]\n");
            toml.push_str(&format!("kind = {:?}\n", chain.codec_kind));
            if let Some(mode) = chain.codec_mode {
                toml.push_str(&format!("mode = {mode:?}\n"));
            }
        }
    }

    match output {
        Some(out_path) => {
            std::fs::write(out_path, &toml)
                .map_err(|err| format!("failed to write {}: {err}", out_path.display()))?;
            println!("import-grsat: wrote {}", out_path.display());
        }
        None => print!("{toml}"),
    }
    Ok(())
}
