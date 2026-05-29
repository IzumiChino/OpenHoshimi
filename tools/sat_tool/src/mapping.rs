//! Shared modulation/framing/codec mapping tables.
//!
//! Maps between external representations (gr-satellites YAML field values,
//! SatNOGS DB mode strings) and OpenHoshimi TOML field values.

/// Standard decode chain for a given framing protocol.
#[derive(Debug, Clone)]
pub struct ChainPreset {
    pub framing: &'static str,
    pub codec_kind: &'static str,
    pub codec_mode: Option<&'static str>,
    pub line_coding: Option<&'static str>,
    pub descrambler: Option<&'static str>,
    pub framer_kind: Option<&'static str>,
}

/// Map a gr-satellites framing string to our decode chain.
pub fn chain_for_framing(framing: &str) -> Option<ChainPreset> {
    match framing.to_ascii_uppercase().as_str() {
        "AX.25" | "AX25" | "USP" => Some(ChainPreset {
            framing: "AX25",
            codec_kind: "ax25",
            codec_mode: None,
            line_coding: Some("nrzi"),
            descrambler: Some("g3ruh"),
            framer_kind: Some("hdlc"),
        }),
        "AX100 ASM+GOLAY" | "GOMSPACE_AX100" | "GOMSPACE AX100" => Some(ChainPreset {
            framing: "GOMSPACE_AX100",
            codec_kind: "gomspace_ax100",
            codec_mode: Some("asm_golay"),
            line_coding: None,
            descrambler: None,
            framer_kind: Some("ax100_asm"),
        }),
        "AX100 REED SOLOMON" | "AX100 RS" => Some(ChainPreset {
            framing: "GOMSPACE_AX100",
            codec_kind: "gomspace_ax100",
            codec_mode: Some("reed_solomon"),
            line_coding: None,
            descrambler: None,
            framer_kind: None,
        }),
        "AO-40 FEC" | "AO-40 FEC CRC-16" | "AO40_FEC" | "AO40 FEC" => Some(ChainPreset {
            framing: "AO40_FEC",
            codec_kind: "ao40_fec",
            codec_mode: None,
            line_coding: None,
            descrambler: None,
            framer_kind: Some("ao40"),
        }),
        "GEOSCAN" => Some(ChainPreset {
            framing: "GEOSCAN",
            codec_kind: "geoscan",
            codec_mode: None,
            line_coding: None,
            descrambler: None,
            framer_kind: None,
        }),
        _ => None,
    }
}

/// Modem parameters derived from a modulation string.
#[derive(Debug, Clone)]
pub struct ModemPreset {
    pub kind: &'static str,
    pub mode: Option<&'static str>,
    pub gaussian_bt: Option<f32>,
    pub modulation_index: Option<f32>,
}

/// Map a modulation string to modem parameters.
pub fn modem_for_modulation(modulation: &str) -> Option<ModemPreset> {
    match modulation.to_ascii_uppercase().as_str() {
        "AFSK" => Some(ModemPreset {
            kind: "afsk",
            mode: None,
            gaussian_bt: None,
            modulation_index: None,
        }),
        "FSK" => Some(ModemPreset {
            kind: "cpm",
            mode: Some("fsk"),
            gaussian_bt: None,
            modulation_index: Some(1.0),
        }),
        "GMSK" => Some(ModemPreset {
            kind: "cpm",
            mode: Some("gmsk"),
            gaussian_bt: Some(0.5),
            modulation_index: Some(0.5),
        }),
        "GFSK" => Some(ModemPreset {
            kind: "cpm",
            mode: Some("gfsk"),
            gaussian_bt: Some(0.5),
            modulation_index: Some(1.0),
        }),
        "MSK" => Some(ModemPreset {
            kind: "cpm",
            mode: Some("msk"),
            gaussian_bt: None,
            modulation_index: Some(0.5),
        }),
        "BPSK" | "DBPSK" => Some(ModemPreset {
            kind: "linear",
            mode: Some("bpsk"),
            gaussian_bt: None,
            modulation_index: None,
        }),
        _ => None,
    }
}

/// Parse a SatNOGS DB mode string like "FSK AX.100 Mode 5" or "AFSK 1k2".
/// Returns (modulation, framing_hint, baudrate_hint).
pub fn parse_satnogs_mode(mode: &str) -> (Option<String>, Option<String>, Option<u32>) {
    let parts: Vec<&str> = mode.split_whitespace().collect();
    let mut modulation = None;
    let mut framing = None;
    let mut _baudrate_hint = None;

    for part in &parts {
        let upper = part.to_ascii_uppercase();
        match upper.as_str() {
            "FSK" | "GMSK" | "GFSK" | "AFSK" | "BPSK" | "DBPSK" | "MSK" => {
                modulation = Some(upper);
            }
            "AX.100" | "AX100" => {
                framing = Some("AX100 ASM+Golay".to_string());
            }
            "AX.25" | "AX25" => {
                framing = Some("AX.25".to_string());
            }
            s if s.ends_with("K2") || s.ends_with("K8") || s.ends_with("K6") => {
                // e.g. "1k2" -> 1200, "9k6" -> 9600, "4k8" -> 4800
                if let Some(baud) = parse_baud_shorthand(s) {
                    _baudrate_hint = Some(baud);
                }
            }
            _ => {}
        }
    }

    (modulation, framing, _baudrate_hint)
}

fn parse_baud_shorthand(s: &str) -> Option<u32> {
    let s = s.to_ascii_lowercase();
    if let Some(pos) = s.find('k') {
        let integer: u32 = s[..pos].parse().ok()?;
        let frac_str = &s[pos + 1..];
        let frac: u32 = if frac_str.is_empty() {
            0
        } else {
            frac_str.parse().ok()?
        };
        Some(integer * 1000 + frac * 100)
    } else {
        s.parse().ok()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn baud_shorthand_parsing() {
        assert_eq!(parse_baud_shorthand("1k2"), Some(1200));
        assert_eq!(parse_baud_shorthand("9k6"), Some(9600));
        assert_eq!(parse_baud_shorthand("4k8"), Some(4800));
        assert_eq!(parse_baud_shorthand("19k2"), Some(19200));
        assert_eq!(parse_baud_shorthand("1200"), Some(1200));
    }

    #[test]
    fn satnogs_mode_parsing() {
        let (m, f, _b) = parse_satnogs_mode("FSK AX.100 Mode 5");
        assert_eq!(m.as_deref(), Some("FSK"));
        assert_eq!(f.as_deref(), Some("AX100 ASM+Golay"));

        let (m, f, _b) = parse_satnogs_mode("AFSK 1k2");
        assert_eq!(m.as_deref(), Some("AFSK"));
        assert_eq!(f, None);
    }

    #[test]
    fn chain_lookup() {
        let c = chain_for_framing("AX.25").unwrap();
        assert_eq!(c.codec_kind, "ax25");
        assert_eq!(c.line_coding, Some("nrzi"));
        assert_eq!(c.descrambler, Some("g3ruh"));

        let c = chain_for_framing("AX100 ASM+Golay").unwrap();
        assert_eq!(c.codec_kind, "gomspace_ax100");
        assert_eq!(c.codec_mode, Some("asm_golay"));
    }

    #[test]
    fn modem_lookup() {
        let m = modem_for_modulation("GMSK").unwrap();
        assert_eq!(m.kind, "cpm");
        assert_eq!(m.mode, Some("gmsk"));
        assert_eq!(m.gaussian_bt, Some(0.5));
    }
}
