//! TOML-driven telemetry field parser for OpenHoshimi.

#![deny(missing_docs)]
#![forbid(unsafe_code)]

use openhoshimi_core::satellite::{Endian, TelemetryFieldDef, TelemetrySchemaDef};
use openhoshimi_core::{Frame, TelemetryField, TelemetrySchema, TelemetryValue, WarnLevel};

/// Telemetry parser built from a parsed satellite TOML schema.
#[derive(Debug, Clone)]
pub struct SchemaParser {
    fields: Vec<TelemetryFieldDef>,
}

impl SchemaParser {
    /// Build a parser from a satellite telemetry schema definition.
    pub fn new(schema: &TelemetrySchemaDef) -> Self {
        Self {
            fields: schema.fields.clone(),
        }
    }

    /// Parse a raw payload byte slice into telemetry fields.
    pub fn parse_bytes(&self, raw: &[u8]) -> Vec<TelemetryField> {
        self.fields
            .iter()
            .filter_map(|field| self.parse_field(field, raw))
            .collect()
    }

    fn parse_field(&self, field: &TelemetryFieldDef, raw: &[u8]) -> Option<TelemetryField> {
        if let (Some(bit_offset), Some(bit_length)) = (field.bit_offset, field.bit_length) {
            let raw_value = extract_bits_msb_first(raw, bit_offset, bit_length)?;
            return Some(finish_numeric_field(field, raw_value));
        }

        let offset = field.offset?;
        let length = field.length?;
        let end = offset.checked_add(length)?;
        let bytes = raw.get(offset..end)?;
        if !matches!(bytes.len(), 1 | 2 | 4 | 8) {
            return Some(TelemetryField {
                key: field.name.clone(),
                group: field.group.clone(),
                value: TelemetryValue::Bytes(bytes.to_vec()),
                unit: normalized_unit(field.unit.as_ref()),
                warn: WarnLevel::Ok,
            });
        }
        let value = decode_value(bytes, field.endian)?;
        Some(finish_numeric_field(field, value))
    }
}

impl TelemetrySchema for SchemaParser {
    fn parse(&self, frame: &Frame) -> Vec<TelemetryField> {
        self.parse_bytes(&frame.raw)
    }
}

fn finish_numeric_field(field: &TelemetryFieldDef, raw_value: u128) -> TelemetryField {
    let raw_f = raw_value as f64;
    let powered = if (field.power - 1.0).abs() < f64::EPSILON {
        raw_f
    } else {
        raw_f.powf(field.power)
    };
    let adjusted = powered * field.scale + field.bias;
    let warn = warn_level(adjusted, field.warn_below, field.warn_above);
    let value = if is_integer_like(field.scale, field.bias, field.power) && fits_i64(raw_value) {
        TelemetryValue::Int(raw_value as i64)
    } else {
        TelemetryValue::Float(adjusted)
    };

    TelemetryField {
        key: field.name.clone(),
        group: field.group.clone(),
        value,
        unit: normalized_unit(field.unit.as_ref()),
        warn,
    }
}

fn extract_bits_msb_first(raw: &[u8], bit_offset: usize, bit_length: usize) -> Option<u128> {
    if bit_length == 0 || bit_length > 64 {
        return None;
    }
    let end_bit = bit_offset.checked_add(bit_length)?;
    if end_bit > raw.len().checked_mul(8)? {
        return None;
    }
    let mut value: u128 = 0;
    for i in 0..bit_length {
        let bit_index = bit_offset + i;
        let byte = raw[bit_index / 8];
        let bit_in_byte = 7 - (bit_index % 8);
        let bit = (byte >> bit_in_byte) & 1;
        value = (value << 1) | (bit as u128);
    }
    Some(value)
}

fn decode_value(bytes: &[u8], endian: Endian) -> Option<u128> {
    match bytes.len() {
        0 => None,
        1 => Some(u8::from_ne_bytes([bytes[0]]) as u128),
        2 => Some(match endian {
            Endian::Big => u16::from_be_bytes([bytes[0], bytes[1]]) as u128,
            Endian::Little => u16::from_le_bytes([bytes[0], bytes[1]]) as u128,
        }),
        4 => Some(match endian {
            Endian::Big => u32::from_be_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]) as u128,
            Endian::Little => u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]) as u128,
        }),
        8 => Some(match endian {
            Endian::Big => u64::from_be_bytes([
                bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7],
            ]) as u128,
            Endian::Little => u64::from_le_bytes([
                bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7],
            ]) as u128,
        }),
        _ => None,
    }
}

fn fits_i64(value: u128) -> bool {
    value <= i64::MAX as u128
}

fn is_integer_like(scale: f64, bias: f64, power: f64) -> bool {
    scale == 1.0 && bias == 0.0 && (power - 1.0).abs() < f64::EPSILON
}

fn warn_level(value: f64, warn_below: Option<f64>, warn_above: Option<f64>) -> WarnLevel {
    let below = warn_below.is_some_and(|limit| value < limit);
    let above = warn_above.is_some_and(|limit| value > limit);
    if below || above {
        WarnLevel::Warn
    } else {
        WarnLevel::Ok
    }
}

fn normalized_unit(unit: Option<&String>) -> Option<String> {
    unit.and_then(|value| {
        if value.is_empty() {
            None
        } else {
            Some(value.clone())
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use openhoshimi_core::satellite::{Endian, TelemetryFieldDef, TelemetrySchemaDef};

    fn schema() -> TelemetrySchemaDef {
        TelemetrySchemaDef {
            fields: vec![
                TelemetryFieldDef {
                    name: "battery".to_string(),
                    group: "eps".to_string(),
                    offset: Some(0),
                    length: Some(2),
                    bit_offset: None,
                    bit_length: None,
                    endian: Endian::Big,
                    scale: 0.01,
                    bias: 0.0,
                    power: 1.0,
                    unit: Some("V".to_string()),
                    warn_below: Some(3.3),
                    warn_above: Some(4.2),
                },
                TelemetryFieldDef {
                    name: "temp".to_string(),
                    group: "thermal".to_string(),
                    offset: Some(2),
                    length: Some(2),
                    bit_offset: None,
                    bit_length: None,
                    endian: Endian::Little,
                    scale: 0.1,
                    bias: -273.15,
                    power: 1.0,
                    unit: Some("C".to_string()),
                    warn_below: Some(-20.0),
                    warn_above: Some(60.0),
                },
                TelemetryFieldDef {
                    name: "flags".to_string(),
                    group: "status".to_string(),
                    offset: Some(4),
                    length: Some(1),
                    bit_offset: None,
                    bit_length: None,
                    endian: Endian::Big,
                    scale: 1.0,
                    bias: 0.0,
                    power: 1.0,
                    unit: None,
                    warn_below: None,
                    warn_above: None,
                },
            ],
        }
    }

    #[test]
    fn parses_scaled_fields_and_warns() {
        let parser = SchemaParser::new(&schema());
        let frame = Frame {
            satellite_id: 1,
            timestamp: std::time::SystemTime::now(),
            rssi_dbm: None,
            raw: vec![0x01, 0x58, 0x74, 0x0b, 0x01],
            frame_type: openhoshimi_core::FrameType::Ax25,
            soft_bits: None,
        };

        let fields = parser.parse(&frame);
        assert_eq!(fields.len(), 3);
        assert_eq!(fields[0].key, "battery");
        assert!(
            matches!(fields[0].value, TelemetryValue::Float(value) if (value - 3.44).abs() < f64::EPSILON)
        );
        assert_eq!(fields[0].warn, WarnLevel::Ok);
        assert!(
            matches!(fields[1].value, TelemetryValue::Float(value) if (value - 20.05).abs() < 0.01)
        );
        assert_eq!(fields[2].warn, WarnLevel::Ok);
    }

    #[test]
    fn warn_thresholds_are_applied() {
        let parser = SchemaParser::new(&schema());
        let frame = Frame {
            satellite_id: 1,
            timestamp: std::time::SystemTime::now(),
            rssi_dbm: None,
            raw: vec![0x08, 0x34],
            frame_type: openhoshimi_core::FrameType::Ax25,
            soft_bits: None,
        };

        let fields = parser.parse(&frame);
        assert_eq!(fields.len(), 1);
        assert_eq!(fields[0].warn, WarnLevel::Warn);
    }

    #[test]
    fn skips_truncated_fields() {
        let parser = SchemaParser::new(&schema());
        let frame = Frame {
            satellite_id: 1,
            timestamp: std::time::SystemTime::now(),
            rssi_dbm: None,
            raw: vec![0x0f],
            frame_type: openhoshimi_core::FrameType::Ax25,
            soft_bits: None,
        };

        let fields = parser.parse(&frame);
        assert!(fields.is_empty());
    }

    #[test]
    fn integer_like_fields_stay_int() {
        let parser = SchemaParser::new(&TelemetrySchemaDef {
            fields: vec![TelemetryFieldDef {
                name: "mode".to_string(),
                group: "status".to_string(),
                offset: Some(0),
                length: Some(1),
                bit_offset: None,
                bit_length: None,
                endian: Endian::Big,
                scale: 1.0,
                bias: 0.0,
                power: 1.0,
                unit: None,
                warn_below: None,
                warn_above: None,
            }],
        });
        let frame = Frame {
            satellite_id: 1,
            timestamp: std::time::SystemTime::now(),
            rssi_dbm: None,
            raw: vec![0x7f],
            frame_type: openhoshimi_core::FrameType::Unknown,
            soft_bits: None,
        };

        let fields = parser.parse(&frame);
        assert!(matches!(fields[0].value, TelemetryValue::Int(127)));
    }

    #[test]
    fn bytes_are_returned_for_large_fields() {
        let parser = SchemaParser::new(&TelemetrySchemaDef {
            fields: vec![TelemetryFieldDef {
                name: "blob".to_string(),
                group: "raw".to_string(),
                offset: Some(0),
                length: Some(17),
                bit_offset: None,
                bit_length: None,
                endian: Endian::Big,
                scale: 1.0,
                bias: 0.0,
                power: 1.0,
                unit: None,
                warn_below: None,
                warn_above: None,
            }],
        });
        let frame = Frame {
            satellite_id: 1,
            timestamp: std::time::SystemTime::now(),
            rssi_dbm: None,
            raw: (0u8..17).collect(),
            frame_type: openhoshimi_core::FrameType::Unknown,
            soft_bits: None,
        };

        let fields = parser.parse(&frame);
        assert!(matches!(fields[0].value, TelemetryValue::Bytes(ref bytes) if bytes.len() == 17));
    }

    #[test]
    fn extracts_bit_level_fields_msb_first() {
        let parser = SchemaParser::new(&TelemetrySchemaDef {
            fields: vec![
                TelemetryFieldDef {
                    name: "satid".to_string(),
                    group: "hdr".to_string(),
                    offset: None,
                    length: None,
                    bit_offset: Some(0),
                    bit_length: Some(2),
                    endian: Endian::Big,
                    scale: 1.0,
                    bias: 0.0,
                    power: 1.0,
                    unit: None,
                    warn_below: None,
                    warn_above: None,
                },
                TelemetryFieldDef {
                    name: "frametype".to_string(),
                    group: "hdr".to_string(),
                    offset: None,
                    length: None,
                    bit_offset: Some(2),
                    bit_length: Some(6),
                    endian: Endian::Big,
                    scale: 1.0,
                    bias: 0.0,
                    power: 1.0,
                    unit: None,
                    warn_below: None,
                    warn_above: None,
                },
                TelemetryFieldDef {
                    name: "raw10".to_string(),
                    group: "hdr".to_string(),
                    offset: None,
                    length: None,
                    bit_offset: Some(8),
                    bit_length: Some(10),
                    endian: Endian::Big,
                    scale: 1.0,
                    bias: 0.0,
                    power: 1.0,
                    unit: None,
                    warn_below: None,
                    warn_above: None,
                },
            ],
        });
        // Byte 0 = 0b11_001100 -> satid=0b11=3, frametype=0b001100=12.
        // Bytes 1..3 = 0b1010_1010 0b11_xxxxxx -> raw10 = 0b10_1010_1011 = 683.
        let frame = Frame {
            satellite_id: 1,
            timestamp: std::time::SystemTime::now(),
            rssi_dbm: None,
            raw: vec![0b1100_1100, 0b1010_1010, 0b1100_0000],
            frame_type: openhoshimi_core::FrameType::Unknown,
            soft_bits: None,
        };

        let fields = parser.parse(&frame);
        assert_eq!(fields.len(), 3);
        assert!(matches!(fields[0].value, TelemetryValue::Int(3)));
        assert!(matches!(fields[1].value, TelemetryValue::Int(12)));
        assert!(matches!(fields[2].value, TelemetryValue::Int(683)));
    }

    #[test]
    fn applies_power_law_calibration() {
        let parser = SchemaParser::new(&TelemetrySchemaDef {
            fields: vec![TelemetryFieldDef {
                name: "rfpower".to_string(),
                group: "rf".to_string(),
                offset: None,
                length: None,
                bit_offset: Some(0),
                bit_length: Some(8),
                endian: Endian::Big,
                // FUNcube-1 RF-power calibration: 5e-3 * raw^2.0629.
                scale: 5e-3,
                bias: 0.0,
                power: 2.0629,
                unit: Some("mW".to_string()),
                warn_below: None,
                warn_above: None,
            }],
        });
        let frame = Frame {
            satellite_id: 1,
            timestamp: std::time::SystemTime::now(),
            rssi_dbm: None,
            raw: vec![100],
            frame_type: openhoshimi_core::FrameType::Unknown,
            soft_bits: None,
        };

        let fields = parser.parse(&frame);
        assert_eq!(fields.len(), 1);
        let expected = 5e-3 * 100f64.powf(2.0629);
        match &fields[0].value {
            TelemetryValue::Float(value) => {
                assert!(
                    (value - expected).abs() < 1e-9,
                    "got {value}, expected {expected}"
                );
            }
            other => panic!("expected Float, got {other:?}"),
        }
    }

    #[test]
    fn rejects_oversized_bit_length() {
        let parser = SchemaParser::new(&TelemetrySchemaDef {
            fields: vec![TelemetryFieldDef {
                name: "too_big".to_string(),
                group: "hdr".to_string(),
                offset: None,
                length: None,
                bit_offset: Some(0),
                bit_length: Some(65),
                endian: Endian::Big,
                scale: 1.0,
                bias: 0.0,
                power: 1.0,
                unit: None,
                warn_below: None,
                warn_above: None,
            }],
        });
        let frame = Frame {
            satellite_id: 1,
            timestamp: std::time::SystemTime::now(),
            rssi_dbm: None,
            raw: vec![0u8; 9],
            frame_type: openhoshimi_core::FrameType::Unknown,
            soft_bits: None,
        };
        let fields = parser.parse(&frame);
        assert!(fields.is_empty(), "oversized bit_length should be rejected");
    }
}
