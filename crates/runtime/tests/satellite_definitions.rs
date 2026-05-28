//! Smoke-load every TOML under `satellites/` and assert that each
//! definition produces at least one downlink the runtime can build a
//! decode pipeline for.
//!
//! Catches typos in modulation / framing tokens, missing modem stages,
//! and misnamed codec kinds before they reach the GUI.

use std::path::PathBuf;

use openhoshimi_core::satellite::load_all_satellites;
use openhoshimi_runtime::pipeline::{can_build_downlink, input_kind_for};

fn satellites_dir() -> PathBuf {
    let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    path.pop();
    path.pop();
    path.push("satellites");
    path
}

#[test]
fn every_shipped_satellite_loads_cleanly() {
    let defs = load_all_satellites(&satellites_dir()).expect("load satellites/");
    assert!(!defs.is_empty(), "no satellite TOMLs found");
    // A satellite without a buildable downlink is allowed (e.g. SO-124
    // ships intentionally-unwired placeholders for AMSAT-EA framing).
    // Just sanity-check the ones whose pipeline must work today.
    for name in [
        "AO-73",
        "Hyacinth-1",
        "StratoSat TK-1",
        "ISS",
        "IO-117",
        "CAS-4A",
        "CAS-4B",
        "CAS-5A",
    ] {
        let def = defs
            .iter()
            .find(|d| d.satellite.name == name)
            .unwrap_or_else(|| panic!("{name}.toml present"));
        assert!(
            def.downlinks
                .iter()
                .any(|d| input_kind_for(d).is_some() && can_build_downlink(d)),
            "{name} has no buildable downlink"
        );
    }
}

#[test]
fn iss_aprs_definition_loads() {
    let defs = load_all_satellites(&satellites_dir()).expect("load satellites/");
    let iss = defs
        .iter()
        .find(|d| d.satellite.name == "ISS")
        .expect("ISS.toml present");
    assert_eq!(iss.satellite.norad_id, 25544);
    // Two downlinks: APRS digipeater + SSTV (Robot36 / PD120 image
    // downlink, no modem/codec stages, runtime treats it specially).
    assert_eq!(iss.downlinks.len(), 2);
    let aprs = iss
        .downlinks
        .iter()
        .find(|d| d.framing.eq_ignore_ascii_case("AX25"))
        .expect("ISS APRS downlink");
    assert_eq!(aprs.freq_hz, 145_825_000);
    assert_eq!(aprs.baudrate, 1200);
    assert!(can_build_downlink(aprs));
    let sstv = iss
        .downlinks
        .iter()
        .find(|d| d.framing.eq_ignore_ascii_case("SSTV"))
        .expect("ISS SSTV downlink");
    assert_eq!(sstv.freq_hz, 145_800_000);
    assert!(matches!(
        sstv.image,
        Some(openhoshimi_core::satellite::ImageDef::Sstv {})
    ));
}

#[test]
fn io117_greencube_definition_loads() {
    let defs = load_all_satellites(&satellites_dir()).expect("load satellites/");
    let g = defs
        .iter()
        .find(|d| d.satellite.name == "IO-117")
        .expect("IO-117.toml present");
    assert_eq!(g.satellite.norad_id, 53106);
    let baudrates: Vec<u32> = g.downlinks.iter().map(|d| d.baudrate).collect();
    assert!(baudrates.contains(&1200) && baudrates.contains(&9600));
    for d in &g.downlinks {
        assert!(can_build_downlink(d));
    }
}

#[test]
fn cas_series_definitions_load() {
    let defs = load_all_satellites(&satellites_dir()).expect("load satellites/");
    for name in ["CAS-4A", "CAS-4B", "CAS-5A"] {
        let s = defs
            .iter()
            .find(|d| d.satellite.name == name)
            .unwrap_or_else(|| panic!("{name}.toml present"));
        assert!(
            s.downlinks.iter().all(can_build_downlink),
            "{name} has an unbuildable downlink"
        );
    }
}
