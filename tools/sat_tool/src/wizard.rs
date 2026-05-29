//! `sat_tool wizard` — interactive step-by-step TOML builder.

use std::io::{self, BufRead, Write as _};
use std::path::Path;

use crate::mapping::{chain_for_framing, modem_for_modulation};

/// Run the interactive wizard.
pub fn run(output: Option<&Path>) -> Result<(), String> {
    let stdin = io::stdin();
    let mut lines = stdin.lock().lines();

    let name = prompt(&mut lines, "Satellite name")?;
    let norad_id: u32 = prompt(&mut lines, "NORAD ID")?
        .parse()
        .map_err(|_| "invalid NORAD ID".to_string())?;
    let aliases_raw = prompt(&mut lines, "Aliases (comma-separated, or empty)")?;
    let aliases: Vec<&str> = aliases_raw
        .split(',')
        .map(|s| s.trim())
        .filter(|s| !s.is_empty())
        .collect();

    let freq_mhz: f64 = prompt(&mut lines, "Downlink frequency (MHz)")?
        .parse()
        .map_err(|_| "invalid frequency".to_string())?;
    let freq_hz = (freq_mhz * 1_000_000.0) as u64;

    let modulation = prompt(
        &mut lines,
        "Modulation [AFSK / FSK / GMSK / GFSK / BPSK / DBPSK]",
    )?
    .to_ascii_uppercase();

    let baudrate: u32 = prompt(&mut lines, "Baudrate (baud)")?
        .parse()
        .map_err(|_| "invalid baudrate".to_string())?;

    let framing_input = prompt(
        &mut lines,
        "Framing [AX25 / GOMSPACE_AX100 / GEOSCAN / AO40_FEC]",
    )?
    .to_ascii_uppercase();

    let chain = chain_for_framing(&framing_input);
    let framing = chain
        .as_ref()
        .map_or(framing_input.clone(), |c| c.framing.to_string());

    // AX100 mode sub-question.
    let codec_mode = if framing == "GOMSPACE_AX100" {
        let mode = prompt(
            &mut lines,
            "AX100 mode [asm_golay / asm_golay_crc / reed_solomon]",
        )?;
        Some(mode)
    } else {
        chain.as_ref().and_then(|c| c.codec_mode.map(String::from))
    };

    let label = prompt(
        &mut lines,
        &format!("Downlink label (e.g. '{baudrate} {modulation} beacon')"),
    )?;

    // Build TOML.
    let mut toml = String::new();
    toml.push_str("[satellite]\n");
    toml.push_str(&format!("name = {name:?}\n"));
    if !aliases.is_empty() {
        let a: Vec<String> = aliases.iter().map(|s| format!("{s:?}")).collect();
        toml.push_str(&format!("aliases = [{}]\n", a.join(", ")));
    }
    toml.push_str(&format!("norad_id = {norad_id}\n"));

    toml.push_str("\n[[downlink]]\n");
    toml.push_str(&format!("label = {label:?}\n"));
    toml.push_str(&format!("freq_hz = {freq_hz}\n"));
    toml.push_str(&format!("modulation = {modulation:?}\n"));
    toml.push_str(&format!("baudrate = {baudrate}\n"));
    toml.push_str(&format!("framing = {framing:?}\n"));

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
            toml.push_str("invert = false\n");
            toml.push_str("swap_iq = false\n");
        }
    }

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
        if let Some(ref mode) = codec_mode {
            toml.push_str(&format!("mode = {mode:?}\n"));
        }
    }

    match output {
        Some(out_path) => {
            std::fs::write(out_path, &toml)
                .map_err(|err| format!("failed to write {}: {err}", out_path.display()))?;
            println!("wizard: wrote {}", out_path.display());
        }
        None => print!("{toml}"),
    }
    Ok(())
}

fn prompt(lines: &mut io::Lines<io::StdinLock<'_>>, question: &str) -> Result<String, String> {
    eprint!("{question}: ");
    io::stderr().flush().ok();
    lines
        .next()
        .unwrap_or(Ok(String::new()))
        .map_err(|err| format!("input error: {err}"))
}
