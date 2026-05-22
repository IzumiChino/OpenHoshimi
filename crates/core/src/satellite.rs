//! TOML-driven satellite definition format and loader.
//!
//! Everything that distinguishes one amateur satellite from another - its
//! NORAD ID, its downlink frequencies, the modulation and framing of each
//! downlink, and the layout of the telemetry inside each framing - lives in
//! a `*.toml` file under `satellites/`. The Rust types in this module are
//! [`serde::Deserialize`] mirrors of that TOML schema; nothing in here is
//! satellite-specific.
//!
//! The canonical schema is documented on [`SatelliteDefinition`].

use std::collections::HashMap;
use std::path::Path;

use serde::de::Error as _;
use serde::Deserialize;

use crate::ConfigError;

/// Top-level structure of a satellite TOML file.
///
/// # Schema
///
/// ```toml
/// [satellite]
/// name = "AO-73"
/// aliases = ["FUNcube-1"]
/// norad_id = 39444
///
/// [[downlink]]
/// label = "1k2 BPSK beacon"
/// freq_hz = 145_935_000
/// modulation = "DBPSK"
/// baudrate = 1200
/// framing = "AO40_FEC"
/// telemetry_schema = "funcube_1"
///
/// [telemetry.funcube_1]
///
/// [[telemetry.funcube_1.field]]
/// name = "bat_voltage"
/// group = "eps"
/// offset = 2
/// length = 2
/// endian = "big"
/// scale = 0.001
/// unit = "V"
/// warn_below = 6.5
/// warn_above = 8.4
/// ```
#[derive(Debug, Clone, Deserialize)]
pub struct SatelliteDefinition {
    /// Identifying section: name, aliases, NORAD ID.
    #[serde(deserialize_with = "required")]
    pub satellite: SatelliteHeader,
    /// One entry per distinct downlink the satellite transmits.
    #[serde(default, rename = "downlink")]
    pub downlinks: Vec<DownlinkDef>,
    /// Map from `telemetry_schema` name (referenced by a downlink) to its
    /// field layout.
    #[serde(default)]
    pub telemetry: HashMap<String, TelemetrySchemaDef>,
}

/// Identifying header of a satellite definition.
#[derive(Debug, Clone, Deserialize)]
pub struct SatelliteHeader {
    /// Primary display name (e.g. `"AO-73"`).
    #[serde(deserialize_with = "required")]
    pub name: String,
    /// Optional alternative names (e.g. `["FUNcube-1"]`).
    #[serde(default)]
    pub aliases: Vec<String>,
    /// NORAD catalog number, used as the satellite ID on decoded frames.
    #[serde(deserialize_with = "required")]
    pub norad_id: u32,
}

/// One downlink (a single carrier on a single frequency) of a satellite.
#[derive(Debug, Clone, Deserialize)]
pub struct DownlinkDef {
    /// Short human-readable label (e.g. `"1k2 BPSK beacon"`).
    #[serde(deserialize_with = "required")]
    pub label: String,
    /// Centre frequency in Hertz.
    #[serde(deserialize_with = "required")]
    pub freq_hz: u64,
    /// Modulation, e.g. `"DBPSK"`, `"AFSK"`, `"GMSK"`, `"FSK"`.
    #[serde(deserialize_with = "required")]
    pub modulation: String,
    /// Symbol rate in baud.
    #[serde(deserialize_with = "required")]
    pub baudrate: u32,
    /// Framing protocol, e.g. `"AX25"`, `"AO40_FEC"`, `"GOMSPACE_AX100"`.
    #[serde(deserialize_with = "required")]
    pub framing: String,
    /// Name of a telemetry schema in [`SatelliteDefinition::telemetry`].
    /// Optional - some downlinks (e.g. voice repeaters) carry no telemetry.
    #[serde(default)]
    pub telemetry_schema: Option<String>,
    /// Structured modem configuration for this downlink.
    #[serde(default)]
    pub modem: Option<ModemDef>,
    /// Optional line coding transform after demodulation.
    #[serde(default)]
    pub line_coding: Option<LineCodingDef>,
    /// Optional bit descrambler after line decoding.
    #[serde(default)]
    pub descrambler: Option<DescramblerDef>,
    /// Structured framer configuration.
    #[serde(default)]
    pub framer: Option<FramerDef>,
    /// Optional forward-error-correction stage.
    #[serde(default)]
    pub fec: Option<FecDef>,
    /// Structured frame decoder configuration.
    #[serde(default)]
    pub codec: Option<CodecDef>,
}

/// Modem configuration for a downlink.
#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ModemDef {
    /// Bell 202 audio AFSK.
    Afsk {
        /// Mark tone frequency in Hz.
        #[serde(default = "default_afsk_mark_hz")]
        mark_hz: f32,
        /// Space tone frequency in Hz.
        #[serde(default = "default_afsk_space_hz")]
        space_hz: f32,
    },
    /// Binary continuous-phase modulation over IQ.
    Cpm {
        /// CPM family mode.
        mode: CpmModeDef,
        /// Modulation index.
        #[serde(default)]
        modulation_index: Option<f32>,
        /// Gaussian BT product for GFSK/GMSK.
        #[serde(default)]
        gaussian_bt: Option<f32>,
        /// Decode differential encoding after hard slicing.
        #[serde(default)]
        differential: bool,
        /// Invert hard symbol decisions.
        #[serde(default)]
        invert: bool,
    },
    /// Linear phase modulation over IQ.
    Linear {
        /// Linear modulation mode.
        mode: LinearModeDef,
        /// Decode differential encoding after hard slicing.
        #[serde(default)]
        differential: bool,
        /// Invert hard symbol decisions.
        #[serde(default)]
        invert: bool,
    },
    /// LoRa modem placeholder.
    Lora {
        /// LoRa spreading factor.
        spreading_factor: u8,
        /// LoRa bandwidth in Hz.
        bandwidth_hz: u32,
    },
    /// Four-level FSK modem placeholder.
    FourFsk {
        /// Frequency offsets for the four symbols, in Hz.
        freq_offsets_hz: [f32; 4],
    },
}

/// CPM modulation mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CpmModeDef {
    /// Binary FSK.
    Fsk,
    /// Minimum-shift keying.
    Msk,
    /// Gaussian FSK.
    Gfsk,
    /// Gaussian MSK.
    Gmsk,
}

/// Linear modulation mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LinearModeDef {
    /// Binary PSK.
    Bpsk,
    /// Differential binary PSK.
    Dbpsk,
    /// Quadrature PSK.
    Qpsk,
    /// Offset quadrature PSK.
    Oqpsk,
}

/// Line coding transform configuration.
#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum LineCodingDef {
    /// NRZI line coding.
    Nrzi,
    /// NRZ-S line coding placeholder.
    Nrzs,
    /// NRZ-M line coding placeholder.
    Nrzm,
}

/// Descrambler configuration.
#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum DescramblerDef {
    /// G3RUH self-synchronising descrambler.
    G3ruh,
    /// CCSDS randomizer descrambler.
    Ccsds,
}

/// Framer configuration.
#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum FramerDef {
    /// HDLC flag-based framing.
    Hdlc,
    /// Fixed syncword followed by a fixed-size payload.
    Syncword {
        /// Syncword bits as ASCII `0` and `1` characters.
        syncword: String,
        /// Maximum number of bit errors allowed in the syncword.
        #[serde(default)]
        threshold: usize,
        /// Number of payload bits to collect after syncword detection.
        payload_bits: usize,
    },
}

/// Forward-error-correction stage configuration.
#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum FecDef {
    /// Reed-Solomon codeword decoder.
    ReedSolomon {
        /// Interleave factor.
        #[serde(default = "default_interleave")]
        interleave: usize,
    },
    /// AO-40 FEC decoder placeholder for the full chain.
    Ao40,
    /// GOMspace AX100 FEC wrapper.
    Ax100,
}

/// Frame decoder configuration.
#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum CodecDef {
    /// AX.25 frame decoder.
    Ax25,
    /// AO-40 FEC payload decoder.
    Ao40Fec,
    /// GOMspace AX100 decoder.
    GomspaceAx100 {
        /// AX100 decoder mode.
        mode: Ax100ModeDef,
    },
    /// CCSDS frame decoder placeholder.
    Ccsds,
    /// FX.25 frame decoder placeholder.
    Fx25,
    /// Unknown or unsupported frame decoder.
    Unknown,
}

/// GOMspace AX100 decoder mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Ax100ModeDef {
    /// AX100 Reed-Solomon mode.
    ReedSolomon,
    /// AX100 ASM and Golay mode.
    AsmGolay,
}

/// A telemetry schema: the layout of fields inside a frame's payload.
///
/// Schemas are referenced by name from a [`DownlinkDef::telemetry_schema`].
#[derive(Debug, Clone, Default, Deserialize)]
pub struct TelemetrySchemaDef {
    /// One entry per telemetry field in the frame.
    #[serde(default, rename = "field")]
    pub fields: Vec<TelemetryFieldDef>,
}

/// One telemetry field's location and decoding rule inside a frame.
#[derive(Debug, Clone, Deserialize)]
pub struct TelemetryFieldDef {
    /// Short identifier (e.g. `"bat_voltage"`).
    #[serde(deserialize_with = "required")]
    pub name: String,
    /// Group/category the field belongs to (e.g. `"eps"`).
    #[serde(deserialize_with = "required")]
    pub group: String,
    /// Byte offset from the start of the frame payload.
    #[serde(deserialize_with = "required")]
    pub offset: usize,
    /// Length of the raw field in bytes.
    #[serde(deserialize_with = "required")]
    pub length: usize,
    /// Byte order of the raw integer (`"big"` or `"little"`).
    /// Defaults to big-endian, matching the AMSAT/CCSDS convention.
    #[serde(default = "default_endian")]
    pub endian: Endian,
    /// Multiplicative scale: `engineering = raw * scale + bias`.
    /// Defaults to `1.0`.
    #[serde(default = "default_scale")]
    pub scale: f64,
    /// Additive bias applied after `scale`. Defaults to `0.0`.
    #[serde(default)]
    pub bias: f64,
    /// Engineering unit (`"V"`, `"C"`, ...), if any.
    #[serde(default)]
    pub unit: Option<String>,
    /// Soft lower threshold below which the field is considered abnormal.
    #[serde(default)]
    pub warn_below: Option<f64>,
    /// Soft upper threshold above which the field is considered abnormal.
    #[serde(default)]
    pub warn_above: Option<f64>,
}

/// Byte order tag for a telemetry field.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Endian {
    /// Most significant byte first.
    Big,
    /// Least significant byte first.
    Little,
}

fn default_endian() -> Endian {
    Endian::Big
}

fn default_scale() -> f64 {
    1.0
}

fn default_afsk_mark_hz() -> f32 {
    1200.0
}

fn default_afsk_space_hz() -> f32 {
    2200.0
}

fn default_interleave() -> usize {
    1
}

fn required<'de, D, T>(deserializer: D) -> Result<T, D::Error>
where
    D: serde::Deserializer<'de>,
    T: Deserialize<'de>,
{
    Option::<T>::deserialize(deserializer)?
        .ok_or_else(|| D::Error::custom("missing required field"))
}

/// Load a single satellite definition from a `*.toml` file.
///
/// # Errors
///
/// Returns [`ConfigError::Io`] if the file cannot be read,
/// [`ConfigError::Toml`] if the contents are not valid TOML, or
/// [`ConfigError::MissingField`] / [`ConfigError::InvalidValue`] if a
/// required field is absent or invalid.
pub fn load_satellite(path: &Path) -> Result<SatelliteDefinition, ConfigError> {
    let raw = std::fs::read_to_string(path)?;
    let def: SatelliteDefinition = toml::from_str(&raw).map_err(translate_toml_error)?;
    validate(&def)?;
    Ok(def)
}

/// Load every `*.toml` file in a directory.
///
/// Files that fail to parse are reported on stderr and skipped - the
/// returned vector contains only the successfully parsed definitions. This
/// behaviour is intentional: a broken satellite file should not stop the
/// app from loading the rest of the fleet.
///
/// # Errors
///
/// Returns [`ConfigError::Io`] only if the directory itself cannot be
/// listed. Per-file failures are logged, not returned.
pub fn load_all_satellites(dir: &Path) -> Result<Vec<SatelliteDefinition>, ConfigError> {
    let mut out = Vec::new();
    for entry in std::fs::read_dir(dir)? {
        let entry = match entry {
            Ok(e) => e,
            Err(e) => {
                eprintln!("warning: skipping unreadable directory entry: {e}");
                continue;
            }
        };
        let path = entry.path();
        if path.extension().and_then(|s| s.to_str()) != Some("toml") {
            continue;
        }
        match load_satellite(&path) {
            Ok(def) => out.push(def),
            Err(e) => {
                eprintln!("warning: skipping {}: {e}", path.display());
            }
        }
    }
    Ok(out)
}

fn validate(def: &SatelliteDefinition) -> Result<(), ConfigError> {
    for d in &def.downlinks {
        if d.baudrate == 0 {
            return Err(ConfigError::InvalidValue {
                field: format!("downlink[{}].baudrate", d.label),
                reason: "must be > 0".into(),
            });
        }
        if let Some(schema_name) = &d.telemetry_schema {
            if !def.telemetry.contains_key(schema_name) {
                return Err(ConfigError::InvalidValue {
                    field: format!("downlink[{}].telemetry_schema", d.label),
                    reason: format!("references unknown schema '{schema_name}'"),
                });
            }
        }
        validate_downlink_pipeline(d)?;
    }
    Ok(())
}

fn validate_downlink_pipeline(d: &DownlinkDef) -> Result<(), ConfigError> {
    if let Some(modem) = &d.modem {
        validate_modem(&d.label, modem)?;
    }
    if let Some(framer) = &d.framer {
        validate_framer(&d.label, framer)?;
    }
    if let Some(fec) = &d.fec {
        validate_fec(&d.label, fec)?;
    }
    Ok(())
}

fn validate_modem(label: &str, modem: &ModemDef) -> Result<(), ConfigError> {
    match modem {
        ModemDef::Afsk { mark_hz, space_hz } => {
            if *mark_hz <= 0.0 {
                return invalid_value(format!("downlink[{label}].modem.mark_hz"), "must be > 0");
            }
            if *space_hz <= 0.0 {
                return invalid_value(format!("downlink[{label}].modem.space_hz"), "must be > 0");
            }
        }
        ModemDef::Cpm {
            modulation_index,
            gaussian_bt,
            ..
        } => {
            if modulation_index.is_some_and(|value| value <= 0.0) {
                return invalid_value(
                    format!("downlink[{label}].modem.modulation_index"),
                    "must be > 0",
                );
            }
            if gaussian_bt.is_some_and(|value| value <= 0.0) {
                return invalid_value(
                    format!("downlink[{label}].modem.gaussian_bt"),
                    "must be > 0",
                );
            }
        }
        ModemDef::Linear { .. } => {}
        ModemDef::Lora {
            spreading_factor,
            bandwidth_hz,
        } => {
            if !(6..=12).contains(spreading_factor) {
                return invalid_value(
                    format!("downlink[{label}].modem.spreading_factor"),
                    "must be between 6 and 12",
                );
            }
            if *bandwidth_hz == 0 {
                return invalid_value(
                    format!("downlink[{label}].modem.bandwidth_hz"),
                    "must be > 0",
                );
            }
        }
        ModemDef::FourFsk { freq_offsets_hz } => {
            if !freq_offsets_hz.iter().all(|value| value.is_finite()) {
                return invalid_value(
                    format!("downlink[{label}].modem.freq_offsets_hz"),
                    "all offsets must be finite",
                );
            }
        }
    }

    Ok(())
}

fn validate_framer(label: &str, framer: &FramerDef) -> Result<(), ConfigError> {
    match framer {
        FramerDef::Hdlc => {}
        FramerDef::Syncword {
            syncword,
            threshold,
            payload_bits,
        } => {
            if syncword.is_empty() || !syncword.bytes().all(|byte| matches!(byte, b'0' | b'1')) {
                return invalid_value(
                    format!("downlink[{label}].framer.syncword"),
                    "must contain only 0 and 1 characters",
                );
            }
            if *threshold > syncword.len() {
                return invalid_value(
                    format!("downlink[{label}].framer.threshold"),
                    "must be <= syncword length",
                );
            }
            if *payload_bits == 0 {
                return invalid_value(
                    format!("downlink[{label}].framer.payload_bits"),
                    "must be > 0",
                );
            }
        }
    }

    Ok(())
}

fn validate_fec(label: &str, fec: &FecDef) -> Result<(), ConfigError> {
    match fec {
        FecDef::ReedSolomon { interleave } => {
            if *interleave == 0 {
                return invalid_value(format!("downlink[{label}].fec.interleave"), "must be > 0");
            }
        }
        FecDef::Ao40 | FecDef::Ax100 => {}
    }

    Ok(())
}

fn invalid_value<T>(field: String, reason: &'static str) -> Result<T, ConfigError> {
    Err(ConfigError::InvalidValue {
        field,
        reason: reason.to_string(),
    })
}

fn translate_toml_error(err: toml::de::Error) -> ConfigError {
    let msg = err.message();
    if let Some(field) = missing_field_name(msg) {
        return ConfigError::MissingField(field);
    }
    ConfigError::Toml(err)
}

fn missing_field_name(msg: &str) -> Option<String> {
    let marker = "missing field `";
    let start = msg.find(marker)? + marker.len();
    let rest = &msg[start..];
    let end = rest.find('`')?;
    Some(rest[..end].to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn write_temp(name: &str, contents: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!("openhoshimi-test-{}", std::process::id()));
        if let Err(err) = std::fs::create_dir_all(&dir) {
            panic!("create temp dir: {err}");
        }
        let path = dir.join(name);
        let mut f = match std::fs::File::create(&path) {
            Ok(file) => file,
            Err(err) => panic!("create file: {err}"),
        };
        if let Err(err) = f.write_all(contents.as_bytes()) {
            panic!("write file: {err}");
        }
        path
    }

    const VALID_TOML: &str = r#"
[satellite]
name = "AO-73"
aliases = ["FUNcube-1"]
norad_id = 39444

[[downlink]]
label = "1k2 BPSK beacon"
freq_hz = 145935000
modulation = "DBPSK"
baudrate = 1200
framing = "AO40_FEC"
telemetry_schema = "funcube_1"

[downlink.modem]
kind = "linear"
mode = "dbpsk"
differential = true

[downlink.framer]
kind = "syncword"
syncword = "11111110000111011110010110010010000001000100110001011101011011000"
threshold = 0
payload_bits = 2566

[downlink.fec]
kind = "ao40"

[downlink.codec]
kind = "ao40_fec"

[telemetry.funcube_1]

[[telemetry.funcube_1.field]]
name = "bat_voltage"
group = "eps"
offset = 2
length = 2
endian = "big"
scale = 0.001
unit = "V"
warn_below = 6.5
warn_above = 8.4
"#;

    #[test]
    fn load_valid_toml() {
        let path = write_temp("valid.toml", VALID_TOML);
        let def = match load_satellite(&path) {
            Ok(def) => def,
            Err(err) => panic!("valid toml should load: {err}"),
        };
        assert_eq!(def.satellite.norad_id, 39444);
        assert_eq!(def.downlinks.len(), 1);
        assert_eq!(def.downlinks[0].baudrate, 1200);
        match &def.downlinks[0].modem {
            Some(ModemDef::Linear {
                mode,
                differential,
                invert,
            }) => {
                assert_eq!(*mode, LinearModeDef::Dbpsk);
                assert!(*differential);
                assert!(!*invert);
            }
            other => panic!("expected linear modem, got {other:?}"),
        }
        match &def.downlinks[0].framer {
            Some(FramerDef::Syncword {
                syncword,
                threshold,
                payload_bits,
            }) => {
                assert_eq!(syncword.len(), 65);
                assert_eq!(*threshold, 0);
                assert_eq!(*payload_bits, 2566);
            }
            other => panic!("expected syncword framer, got {other:?}"),
        }
        assert!(matches!(def.downlinks[0].fec, Some(FecDef::Ao40)));
        assert!(matches!(def.downlinks[0].codec, Some(CodecDef::Ao40Fec)));
        let schema = match def.telemetry.get("funcube_1") {
            Some(schema) => schema,
            None => panic!("schema present"),
        };
        assert_eq!(schema.fields.len(), 1);
        assert_eq!(schema.fields[0].name, "bat_voltage");
        assert_eq!(schema.fields[0].endian, Endian::Big);
    }

    #[test]
    fn missing_required_field_is_reported() {
        // norad_id is required and is missing here.
        let bad = r#"
[satellite]
name = "X"

[[downlink]]
label = "x"
freq_hz = 1
modulation = "FSK"
baudrate = 1200
framing = "AX25"
"#;
        let path = write_temp("missing.toml", bad);
        let err = match load_satellite(&path) {
            Ok(_) => panic!("should fail"),
            Err(err) => err,
        };
        match err {
            ConfigError::MissingField(name) => assert_eq!(name, "norad_id"),
            other => panic!("expected MissingField, got {other:?}"),
        }
    }

    #[test]
    fn invalid_syncword_is_reported() {
        let bad = r#"
[satellite]
name = "X"
norad_id = 1

[[downlink]]
label = "x"
freq_hz = 1
modulation = "FSK"
baudrate = 1200
framing = "UNKNOWN"

[downlink.framer]
kind = "syncword"
syncword = "10x1"
payload_bits = 8
"#;
        let path = write_temp("invalid-syncword.toml", bad);
        let err = match load_satellite(&path) {
            Ok(_) => panic!("should fail"),
            Err(err) => err,
        };

        match err {
            ConfigError::InvalidValue { field, reason } => {
                assert_eq!(field, "downlink[x].framer.syncword");
                assert_eq!(reason, "must contain only 0 and 1 characters");
            }
            other => panic!("expected InvalidValue, got {other:?}"),
        }
    }

    #[test]
    fn load_all_skips_broken_files() {
        let dir =
            std::env::temp_dir().join(format!("openhoshimi-test-loadall-{}", std::process::id()));
        // Clean any previous run.
        let _ = std::fs::remove_dir_all(&dir);
        if let Err(err) = std::fs::create_dir_all(&dir) {
            panic!("create dir: {err}");
        }

        let good = dir.join("good.toml");
        if let Err(err) = std::fs::write(&good, VALID_TOML) {
            panic!("write good: {err}");
        }
        let bad = dir.join("bad.toml");
        if let Err(err) = std::fs::write(&bad, "this is not = valid [toml") {
            panic!("write bad: {err}");
        }
        let ignored = dir.join("notes.txt");
        if let Err(err) = std::fs::write(&ignored, "ignore me") {
            panic!("write ignored: {err}");
        }

        let defs = match load_all_satellites(&dir) {
            Ok(defs) => defs,
            Err(err) => panic!("directory readable: {err}"),
        };
        assert_eq!(defs.len(), 1, "only the good file should load");
        assert_eq!(defs[0].satellite.name, "AO-73");
    }
}
