use super::*;
use openhoshimi_core::satellite::{
    DiscriminatorDef, Endian, FieldEncoding, TelemetryFieldDef, TelemetrySchemaDef,
    TelemetryVariantDef,
};

fn field(
    name: &str,
    group: &str,
    offset: usize,
    length: usize,
    endian: Endian,
    scale: f64,
    bias: f64,
) -> TelemetryFieldDef {
    TelemetryFieldDef {
        name: name.into(),
        group: group.into(),
        offset: Some(offset),
        length: Some(length),
        bit_offset: None,
        bit_length: None,
        endian,
        scale,
        bias,
        power: 1.0,
        unit: None,
        warn_below: None,
        warn_above: None,
        signed: false,
        encoding: FieldEncoding::Integer,
        decimal_places: 1,
    }
}

fn bit_field(name: &str, group: &str, bit_offset: usize, bit_length: usize) -> TelemetryFieldDef {
    TelemetryFieldDef {
        name: name.into(),
        group: group.into(),
        offset: None,
        length: None,
        bit_offset: Some(bit_offset),
        bit_length: Some(bit_length),
        endian: Endian::Big,
        scale: 1.0,
        bias: 0.0,
        power: 1.0,
        unit: None,
        warn_below: None,
        warn_above: None,
        signed: false,
        encoding: FieldEncoding::Integer,
        decimal_places: 1,
    }
}

fn schema() -> TelemetrySchemaDef {
    TelemetrySchemaDef {
        prefix_skip: 0,
        discriminator: None,
        variants: vec![],
        fields: vec![
            TelemetryFieldDef {
                unit: Some("V".into()),
                warn_below: Some(3.3),
                warn_above: Some(4.2),
                ..field("battery", "eps", 0, 2, Endian::Big, 0.01, 0.0)
            },
            TelemetryFieldDef {
                unit: Some("C".into()),
                warn_below: Some(-20.0),
                warn_above: Some(60.0),
                ..field("temp", "thermal", 2, 2, Endian::Little, 0.1, -273.15)
            },
            field("flags", "status", 4, 1, Endian::Big, 1.0, 0.0),
        ],
    }
}

fn frame_with(raw: Vec<u8>) -> Frame {
    Frame {
        satellite_id: 1,
        timestamp: std::time::SystemTime::now(),
        rssi_dbm: None,
        raw,
        frame_type: openhoshimi_core::FrameType::Unknown,
        soft_bits: None,
    }
}

#[test]
fn parses_scaled_fields_and_warns() {
    let parser = SchemaParser::new(&schema());
    let fields = parser.parse(&frame_with(vec![0x01, 0x58, 0x74, 0x0b, 0x01]));
    assert_eq!(fields.len(), 3);
    assert_eq!(fields[0].key, "battery");
    assert!(matches!(fields[0].value, TelemetryValue::Float(v) if (v - 3.44).abs() < 1e-9));
    assert_eq!(fields[0].warn, WarnLevel::Ok);
    assert!(matches!(fields[1].value, TelemetryValue::Float(v) if (v - 20.05).abs() < 0.01));
    assert_eq!(fields[2].warn, WarnLevel::Ok);
}

#[test]
fn warn_thresholds_are_applied() {
    let parser = SchemaParser::new(&schema());
    // battery raw 0x0834 -> 2100 * 0.01 = 21.0 V (above warn_above=4.2)
    let fields = parser.parse(&frame_with(vec![0x08, 0x34]));
    assert_eq!(fields.len(), 1);
    assert_eq!(fields[0].warn, WarnLevel::Warn);
}

#[test]
fn skips_truncated_fields() {
    let parser = SchemaParser::new(&schema());
    let fields = parser.parse(&frame_with(vec![0x0f]));
    assert!(fields.is_empty());
}

#[test]
fn integer_like_fields_stay_int() {
    let parser = SchemaParser::new(&TelemetrySchemaDef {
        fields: vec![field("mode", "status", 0, 1, Endian::Big, 1.0, 0.0)],
        ..Default::default()
    });
    let fields = parser.parse(&frame_with(vec![0x7f]));
    assert!(matches!(fields[0].value, TelemetryValue::Int(127)));
}

#[test]
fn bytes_are_returned_for_large_fields() {
    let parser = SchemaParser::new(&TelemetrySchemaDef {
        fields: vec![field("blob", "raw", 0, 17, Endian::Big, 1.0, 0.0)],
        ..Default::default()
    });
    let fields = parser.parse(&frame_with((0u8..17).collect()));
    assert!(matches!(fields[0].value, TelemetryValue::Bytes(ref b) if b.len() == 17));
}

#[test]
fn extracts_bit_level_fields_msb_first() {
    let parser = SchemaParser::new(&TelemetrySchemaDef {
        fields: vec![
            bit_field("satid", "hdr", 0, 2),
            bit_field("frametype", "hdr", 2, 6),
            bit_field("raw10", "hdr", 8, 10),
        ],
        ..Default::default()
    });
    let fields = parser.parse(&frame_with(vec![0b1100_1100, 0b1010_1010, 0b1100_0000]));
    assert_eq!(fields.len(), 3);
    assert!(matches!(fields[0].value, TelemetryValue::Int(3)));
    assert!(matches!(fields[1].value, TelemetryValue::Int(12)));
    assert!(matches!(fields[2].value, TelemetryValue::Int(683)));
}

#[test]
fn applies_power_law_calibration() {
    let parser = SchemaParser::new(&TelemetrySchemaDef {
        fields: vec![TelemetryFieldDef {
            scale: 5e-3,
            power: 2.0629,
            unit: Some("mW".into()),
            ..bit_field("rfpower", "rf", 0, 8)
        }],
        ..Default::default()
    });
    let fields = parser.parse(&frame_with(vec![100]));
    let expected = 5e-3 * 100f64.powf(2.0629);
    match &fields[0].value {
        TelemetryValue::Float(v) => assert!((v - expected).abs() < 1e-9),
        other => panic!("expected Float, got {other:?}"),
    }
}

#[test]
fn rejects_oversized_bit_length() {
    let parser = SchemaParser::new(&TelemetrySchemaDef {
        fields: vec![bit_field("too_big", "hdr", 0, 65)],
        ..Default::default()
    });
    let fields = parser.parse(&frame_with(vec![0u8; 9]));
    assert!(fields.is_empty());
}

#[test]
fn signed_int8_decodes_negative() {
    let parser = SchemaParser::new(&TelemetrySchemaDef {
        fields: vec![TelemetryFieldDef {
            signed: true,
            unit: Some("C".into()),
            ..field("temp", "thermal", 0, 1, Endian::Big, 1.0, 0.0)
        }],
        ..Default::default()
    });
    // 0xF6 (signed) = -10
    let fields = parser.parse(&frame_with(vec![0xF6]));
    match &fields[0].value {
        TelemetryValue::Float(v) => assert!((v - -10.0).abs() < 1e-9),
        other => panic!("expected Float, got {other:?}"),
    }
}

#[test]
fn signed_int16_be_decodes_negative_with_scale() {
    let parser = SchemaParser::new(&TelemetrySchemaDef {
        fields: vec![TelemetryFieldDef {
            signed: true,
            ..field("temp_x10", "thermal", 0, 2, Endian::Big, 0.1, 0.0)
        }],
        ..Default::default()
    });
    // 0xFF38 = -200 (signed BE) => -20.0 after scale=0.1
    let fields = parser.parse(&frame_with(vec![0xFF, 0x38]));
    match &fields[0].value {
        TelemetryValue::Float(v) => assert!((v - -20.0).abs() < 1e-9),
        other => panic!("expected Float, got {other:?}"),
    }
}

#[test]
fn float32_be_round_trip() {
    let parser = SchemaParser::new(&TelemetrySchemaDef {
        fields: vec![TelemetryFieldDef {
            encoding: FieldEncoding::Float32,
            ..field("gyro_x", "adcs", 0, 4, Endian::Big, 1.0, 0.0)
        }],
        ..Default::default()
    });
    let bytes = (-1.5f32).to_be_bytes().to_vec();
    let fields = parser.parse(&frame_with(bytes));
    match &fields[0].value {
        TelemetryValue::Float(v) => assert!((v - -1.5).abs() < 1e-6),
        other => panic!("expected Float, got {other:?}"),
    }
}

#[test]
fn bcd_split_decodes_volts() {
    // 0x0C 0x03, decimal_places=1 -> 12.3
    let parser = SchemaParser::new(&TelemetrySchemaDef {
        fields: vec![TelemetryFieldDef {
            encoding: FieldEncoding::BcdSplit,
            decimal_places: 1,
            unit: Some("V".into()),
            ..field("vbatt", "eps", 0, 2, Endian::Big, 1.0, 0.0)
        }],
        ..Default::default()
    });
    let fields = parser.parse(&frame_with(vec![0x0C, 0x03]));
    match &fields[0].value {
        TelemetryValue::Float(v) => assert!((v - 12.3).abs() < 1e-9),
        other => panic!("expected Float, got {other:?}"),
    }
}

#[test]
fn bcd_split_two_decimal_places() {
    // byte0 + byte1 / 10^decimal_places. 0x03 0x21 = 3 + 33/100 = 3.33 V.
    let parser = SchemaParser::new(&TelemetrySchemaDef {
        fields: vec![TelemetryFieldDef {
            encoding: FieldEncoding::BcdSplit,
            decimal_places: 2,
            unit: Some("V".into()),
            ..field("v3v3", "eps", 0, 2, Endian::Big, 1.0, 0.0)
        }],
        ..Default::default()
    });
    let fields = parser.parse(&frame_with(vec![0x03, 0x21]));
    match &fields[0].value {
        TelemetryValue::Float(v) => assert!((v - 3.33).abs() < 1e-9),
        other => panic!("expected Float, got {other:?}"),
    }
}

#[test]
fn sign_magnitude_8_decodes_positive_and_negative() {
    let parser = SchemaParser::new(&TelemetrySchemaDef {
        fields: vec![TelemetryFieldDef {
            encoding: FieldEncoding::SignMagnitude8,
            unit: Some("C".into()),
            ..field("temp", "thermal", 0, 1, Endian::Big, 1.0, 0.0)
        }],
        ..Default::default()
    });
    // 0x19 = +25, 0x99 = -25 (sign-magnitude)
    let fields = parser.parse(&frame_with(vec![0x19]));
    match &fields[0].value {
        TelemetryValue::Float(v) => assert!((v - 25.0).abs() < 1e-9),
        other => panic!("expected Float, got {other:?}"),
    }
    let fields = parser.parse(&frame_with(vec![0x99]));
    match &fields[0].value {
        TelemetryValue::Float(v) => assert!((v - -25.0).abs() < 1e-9),
        other => panic!("expected Float, got {other:?}"),
    }
}

#[test]
fn q15_signed_decodes_l_then_h() {
    // CAS-5A documents q = ((QxH<<8)|QxL)/32768 with frame byte order [QxL, QxH].
    // For value -0.5 we need int16 = -16384 = 0xC000 -> bytes [0x00, 0xC0].
    let parser = SchemaParser::new(&TelemetrySchemaDef {
        fields: vec![TelemetryFieldDef {
            encoding: FieldEncoding::Q15Signed,
            ..field("q0", "adcs", 0, 2, Endian::Big, 1.0, 0.0)
        }],
        ..Default::default()
    });
    let fields = parser.parse(&frame_with(vec![0x00, 0xC0]));
    match &fields[0].value {
        TelemetryValue::Float(v) => assert!((v - -0.5).abs() < 1e-9),
        other => panic!("expected Float, got {other:?}"),
    }
}

#[test]
fn prefix_skip_strips_payload_header() {
    let parser = SchemaParser::new(&TelemetrySchemaDef {
        prefix_skip: 3,
        fields: vec![field("v", "eps", 0, 1, Endian::Big, 1.0, 0.0)],
        ..Default::default()
    });
    // First field reads byte 0 of the body, which is byte 3 of raw.
    let fields = parser.parse(&frame_with(vec![0xAA, 0xBB, 0xCC, 0x42]));
    assert!(matches!(fields[0].value, TelemetryValue::Int(0x42)));
}

#[test]
fn discriminator_picks_variant_and_falls_back() {
    let schema = TelemetrySchemaDef {
        prefix_skip: 2,
        discriminator: Some(DiscriminatorDef {
            offset: 0,
            length: 2,
            endian: Endian::Big,
        }),
        variants: vec![
            TelemetryVariantDef {
                name: "alpha".into(),
                match_value: 0x0001,
                fields: vec![field("alpha_v", "alpha", 0, 1, Endian::Big, 1.0, 0.0)],
            },
            TelemetryVariantDef {
                name: "beta".into(),
                match_value: 0x0002,
                fields: vec![field("beta_v", "beta", 0, 1, Endian::Big, 1.0, 0.0)],
            },
        ],
        fields: vec![field("fallback", "misc", 0, 1, Endian::Big, 1.0, 0.0)],
    };
    let parser = SchemaParser::new(&schema);

    // discriminator = 0x0001 (matches alpha); body byte 0 = 0x10
    let fields = parser.parse(&frame_with(vec![0x00, 0x01, 0x10]));
    assert_eq!(fields.len(), 1);
    assert_eq!(fields[0].key, "alpha_v");

    // discriminator = 0x0002 (matches beta); body byte 0 = 0x20
    let fields = parser.parse(&frame_with(vec![0x00, 0x02, 0x20]));
    assert_eq!(fields.len(), 1);
    assert_eq!(fields[0].key, "beta_v");

    // discriminator = 0x0099 (no match -> fallback); body byte 0 = 0xFF
    let fields = parser.parse(&frame_with(vec![0x00, 0x99, 0xFF]));
    assert_eq!(fields.len(), 1);
    assert_eq!(fields[0].key, "fallback");
}
