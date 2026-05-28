//! End-to-end AFSK 1200 + NRZI + HDLC + AX.25 roundtrip.
//!
//! Validates that the BitPipeline correctly chains the audio AFSK
//! demodulator, the NRZI line decoder, the HDLC framer, and the AX.25
//! codec for a Bell 202 / APRS-style downlink (e.g. the ISS amateur
//! digipeater on 145.825 MHz).

use std::f32::consts::TAU;

use openhoshimi_codec::{Ax25Frame, Callsign};
use openhoshimi_core::satellite::{CodecDef, DownlinkDef, FramerDef, LineCodingDef, ModemDef};
use openhoshimi_runtime::pipeline::{BitPipeline, DecodedFrame};

const SAMPLE_RATE: u32 = 48_000;
const BAUDRATE: u32 = 1200;
const MARK_HZ: f32 = 1200.0;
const SPACE_HZ: f32 = 2200.0;
const HDLC_FLAG: u8 = 0x7e;

fn iss_downlink() -> DownlinkDef {
    DownlinkDef {
        label: "APRS digipeater (1k2 AFSK)".to_string(),
        freq_hz: 145_825_000,
        modulation: "AFSK".to_string(),
        baudrate: BAUDRATE,
        framing: "AX25".to_string(),
        telemetry_schema: None,
        modem: Some(ModemDef::Afsk {
            mark_hz: MARK_HZ,
            space_hz: SPACE_HZ,
        }),
        line_coding: Some(LineCodingDef::Nrzi),
        descrambler: None,
        framer: Some(FramerDef::Hdlc),
        fec: None,
        codec: Some(CodecDef::Ax25),
        image: None,
    }
}

fn encode_callsign(call: &str, ssid: u8, last: bool) -> [u8; 7] {
    let mut out = [b' ' << 1; 7];
    for (idx, byte) in call.bytes().take(6).enumerate() {
        out[idx] = byte.to_ascii_uppercase() << 1;
    }
    out[6] = 0x60 | ((ssid & 0x0f) << 1) | u8::from(last);
    out
}

fn ax25_ui_frame(dest: &str, src: &str, info: &[u8]) -> Vec<u8> {
    use crc::{Crc, CRC_16_IBM_SDLC};
    const FCS: Crc<u16> = Crc::<u16>::new(&CRC_16_IBM_SDLC);
    let mut payload = Vec::new();
    payload.extend_from_slice(&encode_callsign(dest, 0, false));
    payload.extend_from_slice(&encode_callsign(src, 0, true));
    payload.push(0x03);
    payload.push(0xf0);
    payload.extend_from_slice(info);
    let fcs = FCS.checksum(&payload);
    payload.extend_from_slice(&fcs.to_le_bytes());
    payload
}

fn bits_lsb_first(byte: u8) -> [u8; 8] {
    [
        byte & 1,
        (byte >> 1) & 1,
        (byte >> 2) & 1,
        (byte >> 3) & 1,
        (byte >> 4) & 1,
        (byte >> 5) & 1,
        (byte >> 6) & 1,
        (byte >> 7) & 1,
    ]
}

fn hdlc_encode(payload_with_fcs: &[u8]) -> Vec<u8> {
    let mut bits = Vec::new();
    bits.extend_from_slice(&bits_lsb_first(HDLC_FLAG));
    let mut ones = 0usize;
    for &byte in payload_with_fcs {
        for bit in bits_lsb_first(byte) {
            bits.push(bit);
            if bit == 1 {
                ones += 1;
                if ones == 5 {
                    bits.push(0);
                    ones = 0;
                }
            } else {
                ones = 0;
            }
        }
    }
    bits.extend_from_slice(&bits_lsb_first(HDLC_FLAG));
    bits
}

fn nrzi_encode(bits: &[u8]) -> Vec<u8> {
    let mut level = 1u8;
    let mut out = Vec::with_capacity(bits.len());
    for &bit in bits {
        if bit & 1 == 0 {
            level ^= 1;
        }
        out.push(level);
    }
    out
}

fn synthesize_afsk(symbols: &[u8]) -> Vec<f32> {
    let samples_per_symbol = SAMPLE_RATE / BAUDRATE;
    let mut samples = Vec::with_capacity(symbols.len() * samples_per_symbol as usize);
    let mut phase = 0.0f32;
    for &sym in symbols {
        let freq = if sym & 1 == 0 { SPACE_HZ } else { MARK_HZ };
        let inc = TAU * freq / SAMPLE_RATE as f32;
        for _ in 0..samples_per_symbol {
            samples.push(phase.sin());
            phase += inc;
            if phase >= TAU {
                phase -= TAU;
            }
        }
    }
    samples
}

#[test]
fn afsk_1k2_aprs_roundtrip_through_pipeline() {
    let info = b"=4920.00N/00133.00W>OpenHoshimi self-test";
    let frame_bytes = ax25_ui_frame("APRS", "OPNHSH", info);
    let hdlc_bits = hdlc_encode(&frame_bytes);
    // Insert a long preamble of HDLC flags so the AFSK detector locks
    // before payload bits arrive.
    let mut preamble_bits = Vec::new();
    for _ in 0..16 {
        preamble_bits.extend_from_slice(&bits_lsb_first(HDLC_FLAG));
    }
    let mut all_bits = preamble_bits;
    all_bits.extend_from_slice(&hdlc_bits);

    let nrzi = nrzi_encode(&all_bits);
    let audio = synthesize_afsk(&nrzi);

    let downlink = iss_downlink();
    let mut pipeline = BitPipeline::<f32>::new(&downlink).expect("build AFSK pipeline");
    pipeline
        .configure_demodulator(&downlink, SAMPLE_RATE, 0.0)
        .expect("configure AFSK demodulator");

    let frames = pipeline.push_samples(&audio);
    assert!(!frames.is_empty(), "AFSK pipeline produced no frames");
    let mut decoded_one_ok = false;
    for frame in &frames {
        let decoded = pipeline.decode_frame(frame).expect("decode AX.25");
        if let DecodedFrame::Ax25(Ax25Frame {
            destination: Callsign { call: dest, .. },
            source: Callsign { call: src, .. },
            info: payload,
            ..
        }) = decoded
        {
            assert_eq!(dest, "APRS");
            assert_eq!(src, "OPNHSH");
            assert_eq!(payload, info);
            decoded_one_ok = true;
        }
    }
    assert!(
        decoded_one_ok,
        "no frame decoded as the expected AX.25 UI payload"
    );
}
