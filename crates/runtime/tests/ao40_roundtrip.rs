//! End-to-end roundtrip across [`Ao40FecEncoder`], [`Ao40Framer`] and
//! [`Ao40FecDecoder`].
//!
//! Decisive sanity check that the codec encoder, the dsp distributed-sync
//! framer, and the codec decoder all agree on bit packing, interleaver
//! geometry, scrambler convention and Viterbi polynomials. If this passes,
//! any failure on a real recording is upstream of the framer (demod / IQ
//! front-end), not in the FEC chain.

use openhoshimi_codec::{ao40_syncword_bits, Ao40FecDecoder, Ao40FecEncoder};
use openhoshimi_core::Framing;
use openhoshimi_dsp::Ao40Framer;

const FRAME_BITS: usize = 5200;
const CHANNEL_BITS: usize = 5132;
const SYNC_LEN: usize = 65;
const SYNC_STRIDE: usize = 80;

fn build_transmitted_frame(channel_bits: &[u8]) -> Vec<u8> {
    assert_eq!(channel_bits.len(), CHANNEL_BITS);
    let mut frame = vec![0u8; FRAME_BITS];

    let sync = ao40_syncword_bits();
    for (k, bit) in sync.iter().enumerate() {
        frame[k * SYNC_STRIDE] = *bit;
    }

    for (out_index, &bit) in channel_bits.iter().enumerate() {
        let i = out_index + SYNC_LEN;
        let pos = (i % SYNC_LEN) * SYNC_STRIDE + i / SYNC_LEN;
        frame[pos] = bit;
    }

    frame
}

fn sample_payload() -> Vec<u8> {
    (0u16..256)
        .map(|n| (n.wrapping_mul(173) ^ 0x5a) as u8)
        .collect()
}

#[test]
fn ao40_hard_roundtrip_through_framer_and_decoder() {
    let payload = sample_payload();
    let encoder = Ao40FecEncoder::new();
    let channel_bits = match encoder.encode_to_channel_bits(&payload) {
        Ok(bits) => bits,
        Err(err) => panic!("encode failed: {err}"),
    };
    assert_eq!(channel_bits.len(), CHANNEL_BITS);

    let frame = build_transmitted_frame(&channel_bits);
    let mut framer = Ao40Framer::new(0);
    let frames = framer.push_bytes(&frame);

    assert_eq!(frames.len(), 1, "framer must emit exactly one frame");
    assert_eq!(frames[0].raw, channel_bits);

    let decoder = Ao40FecDecoder::new();
    let decoded = match decoder.decode_channel_bits(&frames[0].raw) {
        Ok(decoded) => decoded,
        Err(err) => panic!("hard decode failed: {err}"),
    };

    assert_eq!(decoded.payload, payload);
    assert_eq!(decoded.corrected_errors, 0);
}

#[test]
fn ao40_soft_roundtrip_through_framer_and_decoder() {
    let payload = sample_payload();
    let channel_bits = match Ao40FecEncoder::new().encode_to_channel_bits(&payload) {
        Ok(bits) => bits,
        Err(err) => panic!("encode failed: {err}"),
    };

    let hard_frame = build_transmitted_frame(&channel_bits);
    let soft_frame: Vec<i8> = hard_frame
        .iter()
        .map(|&bit| if bit == 0 { 64 } else { -64 })
        .collect();

    let mut framer = Ao40Framer::new(0);
    let frames = framer.push_soft_bytes(&soft_frame);

    assert_eq!(frames.len(), 1, "framer must emit exactly one soft frame");
    let soft_payload_bits = &frames[0];
    assert_eq!(soft_payload_bits.len(), CHANNEL_BITS);

    let decoded = match Ao40FecDecoder::new().decode_soft_channel_bits(soft_payload_bits) {
        Ok(decoded) => decoded,
        Err(err) => panic!("soft decode failed: {err}"),
    };

    assert_eq!(decoded.payload, payload);
    assert_eq!(decoded.corrected_errors, 0);
}

#[test]
fn ao40_hard_roundtrip_corrects_a_handful_of_byte_errors() {
    let payload = sample_payload();
    let channel_bits = match Ao40FecEncoder::new().encode_to_channel_bits(&payload) {
        Ok(bits) => bits,
        Err(err) => panic!("encode failed: {err}"),
    };

    let mut frame = build_transmitted_frame(&channel_bits);
    // Flip a small handful of channel bits scattered across the block; the
    // K=7 r=1/2 inner code plus interleaved RS(160,128) outer code should
    // recover cleanly.
    for offset in [123usize, 777, 1801, 3402] {
        frame[offset] ^= 1;
    }

    let mut framer = Ao40Framer::new(0);
    let frames = framer.push_bytes(&frame);
    assert_eq!(frames.len(), 1);

    let decoded = match Ao40FecDecoder::new().decode_channel_bits(&frames[0].raw) {
        Ok(decoded) => decoded,
        Err(err) => panic!("hard decode with bit errors failed: {err}"),
    };

    assert_eq!(decoded.payload, payload);
}
