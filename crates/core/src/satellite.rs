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
    }
    Ok(())
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
