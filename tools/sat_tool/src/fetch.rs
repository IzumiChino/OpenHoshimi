//! `sat_tool fetch` — query SatNOGS DB API and generate a TOML.
#![allow(dead_code)]

use std::path::Path;

use serde::Deserialize;

use crate::mapping::{chain_for_framing, modem_for_modulation, parse_satnogs_mode};

const SATNOGS_SAT_URL: &str = "https://db.satnogs.org/api/satellites";
const SATNOGS_TX_URL: &str = "https://db.satnogs.org/api/transmitters";

#[derive(Debug, Deserialize)]
struct SatnogsTransmitter {
    description: String,
    alive: bool,
    downlink_low: Option<u64>,
    mode: Option<String>,
    baud: Option<f64>,
    invert: bool,
    #[serde(rename = "type")]
    tx_type: String,
}

#[derive(Debug, Deserialize)]
struct SatnogsSatellite {
    name: String,
    #[serde(default)]
    names: String,
    norad_cat_id: u32,
}

/// Run the SatNOGS DB fetch.
pub fn run(norad_id: u32, output: Option<&Path>) -> Result<(), String> {
    eprintln!("fetch: querying SatNOGS DB for NORAD {norad_id}...");

    let sat_url = format!("{SATNOGS_SAT_URL}/{norad_id}/?format=json");
    let sat: SatnogsSatellite = ureq::get(&sat_url)
        .call()
        .map_err(|err| format!("failed to fetch satellite info: {err}"))?
        .into_json()
        .map_err(|err| format!("failed to parse satellite JSON: {err}"))?;

    let tx_url = format!("{SATNOGS_TX_URL}/?satellite__norad_cat_id={norad_id}&format=json");
    let transmitters: Vec<SatnogsTransmitter> = ureq::get(&tx_url)
        .call()
        .map_err(|err| format!("failed to fetch transmitters: {err}"))?
        .into_json()
        .map_err(|err| format!("failed to parse transmitters JSON: {err}"))?;

    let active_tx: Vec<&SatnogsTransmitter> = transmitters
        .iter()
        .filter(|tx| tx.alive && tx.tx_type == "Transmitter")
        .collect();

    if active_tx.is_empty() {
        return Err(format!(
            "no active transmitters found for {} (NORAD {norad_id})",
            sat.name
        ));
    }

    // Build TOML.
    let mut toml = String::new();
    toml.push_str(&format!(
        "# Generated from SatNOGS DB, NORAD {norad_id}\n\n"
    ));
    toml.push_str("[satellite]\n");
    toml.push_str(&format!("name = {:?}\n", sat.name));
    let aliases: Vec<&str> = sat
        .names
        .split(',')
        .map(|s| s.trim())
        .filter(|s| !s.is_empty() && *s != sat.name)
        .collect();
    if !aliases.is_empty() {
        let a: Vec<String> = aliases.iter().map(|s| format!("{s:?}")).collect();
        toml.push_str(&format!("aliases = [{}]\n", a.join(", ")));
    }
    toml.push_str(&format!("norad_id = {norad_id}\n"));

    for tx in &active_tx {
        toml.push('\n');
        toml.push_str("[[downlink]]\n");
        toml.push_str(&format!("label = {:?}\n", tx.description));

        let freq = tx.downlink_low.unwrap_or(0);
        toml.push_str(&format!("freq_hz = {freq}\n"));

        let mode_str = tx.mode.as_deref().unwrap_or("");
        let (modulation, framing_hint, baud_hint) = parse_satnogs_mode(mode_str);
        let modulation = modulation.unwrap_or_else(|| "FSK".to_string());
        let baudrate = tx.baud.map(|b| b as u32).or(baud_hint).unwrap_or(9600);

        toml.push_str(&format!("modulation = {modulation:?}\n"));
        toml.push_str(&format!("baudrate = {baudrate}\n"));

        let framing_key = framing_hint.as_deref().unwrap_or("AX25");
        let chain = chain_for_framing(framing_key);
        let framing_toml = chain
            .as_ref()
            .map_or(framing_key.to_string(), |c| c.framing.to_string());
        toml.push_str(&format!("framing = {framing_toml:?}\n"));

        // Modem.
        if let Some(modem) = modem_for_modulation(&modulation) {
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
            if modem.kind != "afsk" {
                toml.push_str("frequency_offset_hz = 0.0\n");
                toml.push_str("differential = false\n");
                toml.push_str(&format!("invert = {}\n", tx.invert));
                toml.push_str("swap_iq = false\n");
            }
        }

        // Chain.
        if let Some(ref c) = chain {
            if let Some(lc) = c.line_coding {
                toml.push_str("\n[downlink.line_coding]\n");
                toml.push_str(&format!("kind = {lc:?}\n"));
            }
            if let Some(ds) = c.descrambler {
                if baudrate > 1200 || ds != "g3ruh" {
                    toml.push_str("\n[downlink.descrambler]\n");
                    toml.push_str(&format!("kind = {ds:?}\n"));
                }
            }
            if let Some(fk) = c.framer_kind {
                toml.push_str("\n[downlink.framer]\n");
                toml.push_str(&format!("kind = {fk:?}\n"));
                if fk == "ax100_asm" || fk == "ao40" {
                    toml.push_str("threshold = 4\n");
                }
            }
            toml.push_str("\n[downlink.codec]\n");
            toml.push_str(&format!("kind = {:?}\n", c.codec_kind));
            if let Some(mode) = c.codec_mode {
                toml.push_str(&format!("mode = {mode:?}\n"));
            }
        }
    }

    match output {
        Some(out_path) => {
            std::fs::write(out_path, &toml)
                .map_err(|err| format!("failed to write {}: {err}", out_path.display()))?;
            println!("fetch: wrote {}", out_path.display());
        }
        None => print!("{toml}"),
    }
    Ok(())
}
