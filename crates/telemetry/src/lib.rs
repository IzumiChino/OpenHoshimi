//! TOML-driven telemetry field parser for OpenHoshimi.

#![deny(missing_docs)]
#![forbid(unsafe_code)]

use openhoshimi_core::satellite::{
    DiscriminatorDef, Endian, FieldEncoding, TelemetryFieldDef, TelemetrySchemaDef,
    TelemetryVariantDef,
};
use openhoshimi_core::{Frame, TelemetryField, TelemetrySchema, TelemetryValue, WarnLevel};

/// Telemetry parser built from a parsed satellite TOML schema.
///
/// Supports flat schemas (a single field list applied to every frame) and
/// variant schemas (a discriminator value picks one of several field
/// lists). See [`TelemetrySchemaDef`] for the layout shapes.
#[derive(Debug, Clone)]
pub struct SchemaParser {
    prefix_skip: usize,
    discriminator: Option<DiscriminatorDef>,
    variants: Vec<TelemetryVariantDef>,
    fallback_fields: Vec<TelemetryFieldDef>,
}

impl SchemaParser {
    /// Build a parser from a satellite telemetry schema definition.
    pub fn new(schema: &TelemetrySchemaDef) -> Self {
        Self {
            prefix_skip: schema.prefix_skip,
            discriminator: schema.discriminator.clone(),
            variants: schema.variants.clone(),
            fallback_fields: schema.fields.clone(),
        }
    }

    /// Parse a raw payload byte slice into telemetry fields.
    ///
    /// The discriminator (if any) is read against the original `raw`
    /// slice; field offsets are then evaluated against the slice with
    /// `prefix_skip` bytes removed from the front.
    pub fn parse_bytes(&self, raw: &[u8]) -> Vec<TelemetryField> {
        let fields = self.select_fields(raw);
        let body = raw.get(self.prefix_skip..).unwrap_or(&[]);
        fields
            .iter()
            .filter_map(|field| parse_field(field, body))
            .collect()
    }

    fn select_fields(&self, raw: &[u8]) -> &[TelemetryFieldDef] {
        if let Some(disc) = &self.discriminator {
            if let Some(value) = read_discriminator(raw, disc) {
                if let Some(variant) = self.variants.iter().find(|v| v.match_value == value) {
                    return &variant.fields;
                }
            }
        }
        &self.fallback_fields
    }
}

impl TelemetrySchema for SchemaParser {
    fn parse(&self, frame: &Frame) -> Vec<TelemetryField> {
        self.parse_bytes(&frame.raw)
    }
}

fn read_discriminator(raw: &[u8], disc: &DiscriminatorDef) -> Option<u64> {
    let end = disc.offset.checked_add(disc.length)?;
    let bytes = raw.get(disc.offset..end)?;
    Some(match (disc.length, disc.endian) {
        (1, _) => bytes[0] as u64,
        (2, Endian::Big) => u16::from_be_bytes([bytes[0], bytes[1]]) as u64,
        (2, Endian::Little) => u16::from_le_bytes([bytes[0], bytes[1]]) as u64,
        (4, Endian::Big) => u32::from_be_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]) as u64,
        (4, Endian::Little) => u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]) as u64,
        (8, Endian::Big) => u64::from_be_bytes([
            bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7],
        ]),
        (8, Endian::Little) => u64::from_le_bytes([
            bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7],
        ]),
        _ => return None,
    })
}

fn parse_field(field: &TelemetryFieldDef, body: &[u8]) -> Option<TelemetryField> {
    if let (Some(bit_offset), Some(bit_length)) = (field.bit_offset, field.bit_length) {
        let raw_value = extract_bits_msb_first(body, bit_offset, bit_length)?;
        return Some(finish_numeric_field(field, raw_value as f64, true));
    }

    let offset = field.offset?;
    let length = field.length?;
    let end = offset.checked_add(length)?;
    let bytes = body.get(offset..end)?;

    match field.encoding {
        FieldEncoding::Integer => parse_integer(field, bytes),
        FieldEncoding::Float32 => parse_float32(field, bytes),
        FieldEncoding::Float64 => parse_float64(field, bytes),
        FieldEncoding::BcdSplit => parse_bcd_split(field, bytes),
        FieldEncoding::SignMagnitude8 => parse_sign_magnitude_8(field, bytes),
        FieldEncoding::Q15Signed => parse_q15_signed(field, bytes),
    }
}

fn parse_integer(field: &TelemetryFieldDef, bytes: &[u8]) -> Option<TelemetryField> {
    if !matches!(bytes.len(), 1 | 2 | 4 | 8) {
        return Some(TelemetryField {
            key: field.name.clone(),
            group: field.group.clone(),
            value: TelemetryValue::Bytes(bytes.to_vec()),
            unit: normalized_unit(field.unit.as_ref()),
            warn: WarnLevel::Ok,
        });
    }
    let unsigned = decode_unsigned(bytes, field.endian)?;
    let raw_f = if field.signed {
        sign_extend(unsigned, bytes.len()) as f64
    } else {
        unsigned as f64
    };
    let allow_int_repr = !field.signed && fits_i64(unsigned);
    Some(finish_numeric_field(field, raw_f, allow_int_repr))
}

fn parse_float32(field: &TelemetryFieldDef, bytes: &[u8]) -> Option<TelemetryField> {
    let arr: [u8; 4] = bytes.try_into().ok()?;
    let value = match field.endian {
        Endian::Big => f32::from_be_bytes(arr),
        Endian::Little => f32::from_le_bytes(arr),
    } as f64;
    Some(finish_float_field(field, value))
}

fn parse_float64(field: &TelemetryFieldDef, bytes: &[u8]) -> Option<TelemetryField> {
    let arr: [u8; 8] = bytes.try_into().ok()?;
    let value = match field.endian {
        Endian::Big => f64::from_be_bytes(arr),
        Endian::Little => f64::from_le_bytes(arr),
    };
    Some(finish_float_field(field, value))
}

fn parse_bcd_split(field: &TelemetryFieldDef, bytes: &[u8]) -> Option<TelemetryField> {
    if bytes.len() != 2 {
        return None;
    }
    let integer = bytes[0] as f64;
    let fractional_digits = bytes[1] as f64;
    let denom = 10f64.powi(field.decimal_places as i32);
    let raw_engineering = integer + fractional_digits / denom;
    Some(finish_float_field(field, raw_engineering))
}

fn parse_sign_magnitude_8(field: &TelemetryFieldDef, bytes: &[u8]) -> Option<TelemetryField> {
    let byte = *bytes.first()?;
    let magnitude = (byte & 0x7F) as f64;
    let sign = if byte & 0x80 != 0 { -1.0 } else { 1.0 };
    Some(finish_float_field(field, sign * magnitude))
}

fn parse_q15_signed(field: &TelemetryFieldDef, bytes: &[u8]) -> Option<TelemetryField> {
    if bytes.len() != 2 {
        return None;
    }
    // CAS-5A encodes Q15 components with the low byte first in the
    // frame, then the high byte; the assembled int16 is divided by
    // 32768. The schema-level `endian` is ignored for Q15 because the
    // CAMSAT manual fixes the byte order.
    let assembled = u16::from_le_bytes([bytes[0], bytes[1]]) as i16;
    let raw = assembled as f64 / 32768.0;
    Some(finish_float_field(field, raw))
}

fn finish_numeric_field(
    field: &TelemetryFieldDef,
    raw_value: f64,
    allow_int_repr: bool,
) -> TelemetryField {
    let powered = if (field.power - 1.0).abs() < f64::EPSILON {
        raw_value
    } else {
        raw_value.powf(field.power)
    };
    let adjusted = powered * field.scale + field.bias;
    let warn = warn_level(adjusted, field.warn_below, field.warn_above);
    let value = if allow_int_repr
        && is_integer_like(field.scale, field.bias, field.power)
        && raw_value.fract() == 0.0
        && raw_value.abs() <= i64::MAX as f64
    {
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

fn finish_float_field(field: &TelemetryFieldDef, raw_value: f64) -> TelemetryField {
    let scaled = if (field.power - 1.0).abs() < f64::EPSILON {
        raw_value
    } else {
        raw_value.powf(field.power)
    };
    let adjusted = scaled * field.scale + field.bias;
    let warn = warn_level(adjusted, field.warn_below, field.warn_above);
    TelemetryField {
        key: field.name.clone(),
        group: field.group.clone(),
        value: TelemetryValue::Float(adjusted),
        unit: normalized_unit(field.unit.as_ref()),
        warn,
    }
}

fn sign_extend(value: u128, byte_len: usize) -> i128 {
    let bits = byte_len * 8;
    if bits == 0 || bits > 64 {
        return value as i128;
    }
    let sign_bit = 1u128 << (bits - 1);
    if value & sign_bit != 0 {
        let mask = !((1u128 << bits) - 1);
        (value | mask) as i128
    } else {
        value as i128
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

fn decode_unsigned(bytes: &[u8], endian: Endian) -> Option<u128> {
    match bytes.len() {
        1 => Some(bytes[0] as u128),
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
mod tests;
