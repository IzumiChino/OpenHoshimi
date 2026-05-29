//! GOMspace NanoCom AX100 frame decoder.
//!
//! AX100 uses two common framing modes in amateur-satellite downlinks:
//! Reed-Solomon mode and ASM+Golay mode. This module decodes already
//! synchronized packet bytes; finding the AX100 ASM syncword belongs in
//! the DSP/framing layer.

use openhoshimi_core::DecodeError;

use crate::fec::{ccsds_randomizer, crc32c, golay, reed_solomon, ReedSolomon};

const CRC32_LEN: usize = 4;

const RS_LEN: usize = 255;
const RS_HEADER_LEN: usize = 1;
const ASM_GOLAY_HEADER_LEN: usize = 3;
const AX100_RS_MAX_PDU_LEN: usize = RS_LEN + RS_HEADER_LEN;
const AX100_ASM_GOLAY_MAX_PDU_LEN: usize = RS_LEN + ASM_GOLAY_HEADER_LEN;
const AX100_SYNCWORD: u32 = 0x930b_51de;

/// Decoder for GOMspace NanoCom AX100 packets.
#[derive(Debug, Default, Clone, Copy)]
pub struct Ax100Decoder;

impl Ax100Decoder {
    /// Construct a new AX100 decoder.
    pub fn new() -> Self {
        Self
    }

    /// Decode AX100 Reed-Solomon mode.
    ///
    /// The input is the packet following sync detection. Byte 0 is the AX100
    /// length byte. Bytes 1.. contain a shortened RS(255,223) codeword.
    pub fn decode_reed_solomon(&self, packet: &[u8]) -> Result<Ax100Frame, DecodeError> {
        if packet.len() < RS_HEADER_LEN + reed_solomon::PARITY_LEN {
            return Err(DecodeError::TooShort(packet.len()));
        }
        if packet.len() > AX100_RS_MAX_PDU_LEN {
            return Err(DecodeError::InvalidEncoding(
                "AX100 Reed-Solomon packet is longer than 256 bytes".to_string(),
            ));
        }

        let encoded_len = usize::from(packet[0]);
        if encoded_len <= reed_solomon::PARITY_LEN + RS_HEADER_LEN {
            return Err(DecodeError::TooShort(encoded_len));
        }
        if encoded_len > packet.len() {
            return Err(DecodeError::TooShort(packet.len()));
        }

        let codeword = &packet[1..encoded_len];
        let decoded = ReedSolomon::new(1).decode_shortened(codeword)?;
        let payload_len = encoded_len - reed_solomon::PARITY_LEN - RS_HEADER_LEN;
        if decoded.message.len() < payload_len {
            return Err(DecodeError::TooShort(decoded.message.len()));
        }

        Ok(Ax100Frame {
            mode: Ax100Mode::ReedSolomon,
            payload: decoded.message[..payload_len].to_vec(),
            corrected_errors: decoded.corrected_errors,
            crc_ok: None,
            flags: Ax100Flags::default(),
        })
    }

    /// Decode AX100 ASM+Golay mode.
    ///
    /// The input starts with a three-byte Golay(24,12) header. The lower
    /// eight bits of the decoded header carry the encoded payload length.
    /// The payload is CCSDS-randomized and protected by a shortened
    /// RS(255,223) code, matching the common AX100 ASM+Golay mode.
    pub fn decode_asm_golay(&self, packet: &[u8]) -> Result<Ax100Frame, DecodeError> {
        self.decode_asm_golay_with_options(packet, true, true)
    }

    /// Decode AX100 ASM+Golay with explicit payload options.
    ///
    /// Some compatible transmitters disable CCSDS scrambling or
    /// Reed-Solomon protection. The Golay header is still used for the
    /// encoded payload length.
    pub fn decode_asm_golay_with_options(
        &self,
        packet: &[u8],
        scrambler: bool,
        reed_solomon: bool,
    ) -> Result<Ax100Frame, DecodeError> {
        if packet.len() < ASM_GOLAY_HEADER_LEN {
            return Err(DecodeError::TooShort(packet.len()));
        }
        if packet.len() > AX100_ASM_GOLAY_MAX_PDU_LEN {
            return Err(DecodeError::InvalidEncoding(
                "AX100 ASM+Golay packet is longer than 258 bytes".to_string(),
            ));
        }

        let header =
            (u32::from(packet[0]) << 16) | (u32::from(packet[1]) << 8) | u32::from(packet[2]);
        let decoded_header = golay::decode(header)?;
        let encoded_len = usize::from(decoded_header.data & 0xff);
        let flags = Ax100Flags {
            viterbi: false,
            scrambler,
            reed_solomon,
            golay_errors: decoded_header.corrected_errors,
        };

        if encoded_len == 0 {
            return Err(DecodeError::TooShort(0));
        }
        if packet.len() < ASM_GOLAY_HEADER_LEN + encoded_len {
            return Err(DecodeError::TooShort(packet.len()));
        }

        let mut payload = packet[ASM_GOLAY_HEADER_LEN..ASM_GOLAY_HEADER_LEN + encoded_len].to_vec();
        if scrambler {
            ccsds_randomizer::xor_sequence(&mut payload);
        }

        let (payload, corrected_errors) = if reed_solomon {
            if payload.len() <= reed_solomon::PARITY_LEN {
                return Err(DecodeError::TooShort(payload.len()));
            }
            let decoded = ReedSolomon::new(1).decode_shortened(&payload)?;
            (decoded.message, decoded.corrected_errors)
        } else {
            (payload, 0)
        };

        Ok(Ax100Frame {
            mode: Ax100Mode::AsmGolay,
            payload,
            corrected_errors,
            crc_ok: None,
            flags,
        })
    }

    /// Decode AX100 ASM+Golay as used by the GreenCube / IO-117 digipeater.
    ///
    /// The three-byte Golay(24,12) header's low eight bits give the encoded
    /// length. Measured over on-air captures, this length always counts the
    /// systematic message plus a 32-byte trailing block (the RS(255,223)
    /// parity size). The whole field is CCSDS-randomized; after descrambling
    /// the message bytes appear first, ahead of the 32 trailing bytes. The
    /// message itself is a CSP frame followed by a 4-byte big-endian CRC-32C
    /// (libcsp Castagnoli) computed over the CSP frame.
    ///
    /// This decoder does not run forward error correction. The 32 trailing
    /// bytes were checked against a standard CCSDS RS(255,223) codeword in
    /// conventional and dual-basis representations across the common GF(256)
    /// primitive polynomials and did not validate, so they are not decoded
    /// here; integrity rests on the CRC-32C. The decoder strips the 32
    /// trailing bytes, splits off the CRC-32C, and reports the check in
    /// `crc_ok`. Clean bursts decode bit-exact; bursts with residual demod
    /// bit errors fail the CRC and must be discarded rather than
    /// interpreted. The returned `payload` is the CSP frame with the CRC
    /// trailer removed.
    pub fn decode_asm_golay_crc(&self, packet: &[u8]) -> Result<Ax100Frame, DecodeError> {
        if packet.len() < ASM_GOLAY_HEADER_LEN {
            return Err(DecodeError::TooShort(packet.len()));
        }
        if packet.len() > AX100_ASM_GOLAY_MAX_PDU_LEN {
            return Err(DecodeError::InvalidEncoding(
                "AX100 ASM+Golay packet is longer than 258 bytes".to_string(),
            ));
        }

        let header =
            (u32::from(packet[0]) << 16) | (u32::from(packet[1]) << 8) | u32::from(packet[2]);
        let decoded_header = golay::decode(header)?;
        let encoded_len = usize::from(decoded_header.data & 0xff);
        let flags = Ax100Flags {
            viterbi: false,
            scrambler: true,
            reed_solomon: false,
            golay_errors: decoded_header.corrected_errors,
        };

        // The encoded length spans the systematic message and the 32 RS
        // parity bytes; the message must still hold at least the CRC-32C
        // trailer plus one CSP byte.
        if encoded_len <= reed_solomon::PARITY_LEN + CRC32_LEN {
            return Err(DecodeError::TooShort(encoded_len));
        }
        if packet.len() < ASM_GOLAY_HEADER_LEN + encoded_len {
            return Err(DecodeError::TooShort(packet.len()));
        }

        let mut payload = packet[ASM_GOLAY_HEADER_LEN..ASM_GOLAY_HEADER_LEN + encoded_len].to_vec();
        ccsds_randomizer::xor_sequence(&mut payload);

        let body_len = encoded_len - reed_solomon::PARITY_LEN - CRC32_LEN;
        let crc_pos = body_len;
        let expected = u32::from_be_bytes([
            payload[crc_pos],
            payload[crc_pos + 1],
            payload[crc_pos + 2],
            payload[crc_pos + 3],
        ]);
        let crc_ok = crc32c::checksum(&payload[..body_len]) == expected;

        // Drop the 32 trailing parity bytes and the CRC trailer; the
        // systematic message (CSP frame) precedes them.
        payload.truncate(body_len);

        Ok(Ax100Frame {
            mode: Ax100Mode::AsmGolayCrc,
            payload,
            corrected_errors: 0,
            crc_ok: Some(crc_ok),
            flags,
        })
    }

    /// Decode AX100 ASM+Golay with soft-decision RS erasure marking.
    ///
    /// `soft_per_bit` is one f32 per packet bit, MSB-first inside each
    /// byte (matching `pack_msb_bits`); its length must equal
    /// `packet.len() * 8`. `erasure_count` is the number of payload
    /// bytes to mark as erasures, ranked by ascending min |soft|
    /// across each byte's eight bits.
    ///
    /// The soft path is intended as a *fallback* for codewords whose
    /// hard-decision RS fails. RS(255,223) hard decoding handles up to
    /// 16 byte errors; with `K` erasures, the joint capacity rises to
    /// `2*errors + K <= 32`. Mis-marking a clean byte costs one
    /// erasure slot, so the effective K depends on how reliable the
    /// soft samples are. Callers should pick K from an off-line sweep
    /// over representative recordings.
    pub fn decode_asm_golay_with_soft(
        &self,
        packet: &[u8],
        soft_per_bit: &[f32],
        erasure_count: usize,
    ) -> Result<Ax100Frame, DecodeError> {
        if packet.len() < ASM_GOLAY_HEADER_LEN {
            return Err(DecodeError::TooShort(packet.len()));
        }
        if packet.len() > AX100_ASM_GOLAY_MAX_PDU_LEN {
            return Err(DecodeError::InvalidEncoding(
                "AX100 ASM+Golay packet is longer than 258 bytes".to_string(),
            ));
        }
        if soft_per_bit.len() != packet.len() * 8 {
            return Err(DecodeError::InvalidEncoding(
                "soft sample length must equal packet bit count".to_string(),
            ));
        }

        let header =
            (u32::from(packet[0]) << 16) | (u32::from(packet[1]) << 8) | u32::from(packet[2]);
        let decoded_header = golay::decode(header)?;
        let encoded_len = usize::from(decoded_header.data & 0xff);
        let flags = Ax100Flags {
            viterbi: false,
            scrambler: true,
            reed_solomon: true,
            golay_errors: decoded_header.corrected_errors,
        };

        if encoded_len == 0 {
            return Err(DecodeError::TooShort(0));
        }
        if packet.len() < ASM_GOLAY_HEADER_LEN + encoded_len {
            return Err(DecodeError::TooShort(packet.len()));
        }
        if encoded_len <= reed_solomon::PARITY_LEN {
            return Err(DecodeError::TooShort(encoded_len));
        }

        let payload_range = ASM_GOLAY_HEADER_LEN..ASM_GOLAY_HEADER_LEN + encoded_len;
        let mut payload = packet[payload_range.clone()].to_vec();
        ccsds_randomizer::xor_sequence(&mut payload);

        // Per-byte confidence: minimum |soft| across the 8 bits
        // composing each payload byte, in raw-packet bit order. CCSDS
        // descrambling is XOR, so it does not change which bytes are
        // reliable.
        let bit_offset = payload_range.start * 8;
        let mut byte_confidences: Vec<(usize, f32)> = (0..encoded_len)
            .map(|i| {
                let mut min_abs = f32::INFINITY;
                for j in 0..8 {
                    let s = soft_per_bit[bit_offset + i * 8 + j].abs();
                    if s < min_abs {
                        min_abs = s;
                    }
                }
                (i, min_abs)
            })
            .collect();
        byte_confidences.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        let k = erasure_count.min(encoded_len).min(reed_solomon::PARITY_LEN);
        let erasures: Vec<usize> = byte_confidences[..k].iter().map(|(idx, _)| *idx).collect();

        let decoded = ReedSolomon::new(1).decode_shortened_with_erasures(&payload, &erasures)?;

        Ok(Ax100Frame {
            mode: Ax100Mode::AsmGolay,
            payload: decoded.message,
            corrected_errors: decoded.corrected_errors,
            crc_ok: None,
            flags,
        })
    }
}

/// Decoded AX100 packet.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Ax100Frame {
    /// AX100 framing mode used by the packet.
    pub mode: Ax100Mode,
    /// Decoded packet payload, usually a CSP frame.
    pub payload: Vec<u8>,
    /// Number of Reed-Solomon byte errors corrected by this decoder.
    ///
    /// The current implementation only accepts error-free RS codewords, so
    /// this value is zero until full RS correction is implemented.
    pub corrected_errors: usize,
    /// CRC-32C verification result for modes that carry a CRC trailer
    /// ([`Ax100Mode::AsmGolayCrc`]). `None` for RS-protected modes, which
    /// rely on the Reed-Solomon syndrome instead of a separate CRC.
    pub crc_ok: Option<bool>,
    /// ASM+Golay header flags. In Reed-Solomon mode these are all false.
    pub flags: Ax100Flags,
}

/// AX100 framing mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Ax100Mode {
    /// AX100 Reed-Solomon mode.
    ReedSolomon,
    /// AX100 ASM+Golay mode with CCSDS scrambling and RS(255,223).
    AsmGolay,
    /// AX100 ASM+Golay mode with CCSDS scrambling and a CRC-32C trailer,
    /// without Reed-Solomon. Used by the GreenCube / IO-117 digipeater.
    AsmGolayCrc,
}

/// Flags carried in AX100 ASM+Golay style headers.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct Ax100Flags {
    /// Payload was Viterbi-decoded before this decoder.
    pub viterbi: bool,
    /// Payload was CCSDS-randomized.
    pub scrambler: bool,
    /// Payload carried Reed-Solomon parity.
    pub reed_solomon: bool,
    /// Number of corrected Golay header bit errors.
    pub golay_errors: u8,
}

/// AX100 32-bit attached sync marker used by gr-satellites and many
/// NanoCom AX100 downlinks.
pub fn ax100_syncword() -> u32 {
    AX100_SYNCWORD
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fec::{crc32c, golay, reed_solomon};

    /// Build a GreenCube / IO-117 ASM+Golay+CRC packet from a CSP frame:
    /// append the CRC-32C trailer, pad with `PARITY_LEN` placeholder
    /// parity bytes (a systematic RS codeword's parity region, which this
    /// decoder strips without decoding), CCSDS-scramble the whole field,
    /// and prepend the Golay length header.
    fn build_crc_packet(csp_frame: &[u8]) -> Vec<u8> {
        let mut field = csp_frame.to_vec();
        field.extend_from_slice(&crc32c::checksum(csp_frame).to_be_bytes());
        field.extend(std::iter::repeat(0xAB).take(reed_solomon::PARITY_LEN));
        ccsds_randomizer::xor_sequence(&mut field);
        let header = golay::encode(field.len() as u16);
        let mut packet = vec![
            ((header >> 16) & 0xff) as u8,
            ((header >> 8) & 0xff) as u8,
            (header & 0xff) as u8,
        ];
        packet.extend_from_slice(&field);
        packet
    }

    #[test]
    fn decodes_asm_golay_crc_greencube_frame() {
        let csp_frame = b"\x82\x97\x64\x00\x1d\x03R7LP>CQ, GreenCube, STORE=0 KN96";
        let packet = build_crc_packet(csp_frame);

        let decoder = Ax100Decoder::new();
        let frame = decoder
            .decode_asm_golay_crc(&packet)
            .expect("valid GreenCube ASM+Golay+CRC packet");

        assert_eq!(frame.mode, Ax100Mode::AsmGolayCrc);
        assert_eq!(frame.payload, csp_frame);
        assert_eq!(frame.crc_ok, Some(true));
        assert!(frame.flags.scrambler);
        assert!(!frame.flags.reed_solomon);
    }

    #[test]
    fn asm_golay_crc_flags_bit_error_as_crc_failure() {
        let csp_frame = b"\x82\x97\x64\x00\x1d\x03R7LP>CQ, GreenCube, STORE=0 KN96";
        let mut packet = build_crc_packet(csp_frame);
        // Flip a bit inside the message region (after the 3-byte header).
        packet[10] ^= 0x01;

        let decoder = Ax100Decoder::new();
        let frame = decoder
            .decode_asm_golay_crc(&packet)
            .expect("packet still parses structurally");
        assert_eq!(frame.crc_ok, Some(false));
    }

    #[test]
    fn decodes_reed_solomon_mode_packet() {
        let payload = b"csp-payload";
        let codeword = reed_solomon::encode_shortened(payload, 1);
        let mut packet = Vec::new();
        packet.push((payload.len() + reed_solomon::PARITY_LEN + RS_HEADER_LEN) as u8);
        packet.extend_from_slice(&codeword);

        let decoder = Ax100Decoder::new();
        let frame = match decoder.decode_reed_solomon(&packet) {
            Ok(frame) => frame,
            Err(err) => panic!("valid AX100 Reed-Solomon packet: {err}"),
        };

        assert_eq!(frame.mode, Ax100Mode::ReedSolomon);
        assert_eq!(frame.payload, payload);
        assert_eq!(frame.corrected_errors, 0);
    }

    #[test]
    fn decodes_asm_golay_packet_with_scrambler_and_rs() {
        let payload = b"asm-golay-payload";
        let mut encoded_payload = reed_solomon::encode_shortened(payload, 1);
        ccsds_randomizer::xor_sequence(&mut encoded_payload);

        let header = golay::encode(encoded_payload.len() as u16);
        let mut packet = vec![
            ((header >> 16) & 0xff) as u8,
            ((header >> 8) & 0xff) as u8,
            (header & 0xff) as u8,
        ];
        packet.extend_from_slice(&encoded_payload);

        let decoder = Ax100Decoder::new();
        let frame = match decoder.decode_asm_golay(&packet) {
            Ok(frame) => frame,
            Err(err) => panic!("valid AX100 ASM+Golay packet: {err}"),
        };

        assert_eq!(frame.mode, Ax100Mode::AsmGolay);
        assert_eq!(frame.payload, payload);
        assert!(frame.flags.scrambler);
        assert!(frame.flags.reed_solomon);
    }

    #[test]
    fn corrects_damaged_reed_solomon_packet() {
        let payload = b"csp-payload";
        let codeword = reed_solomon::encode_shortened(payload, 1);
        let mut packet = Vec::new();
        packet.push((payload.len() + reed_solomon::PARITY_LEN + RS_HEADER_LEN) as u8);
        packet.extend_from_slice(&codeword);
        packet[4] ^= 0x20;

        let decoder = Ax100Decoder::new();
        let frame = match decoder.decode_reed_solomon(&packet) {
            Ok(frame) => frame,
            Err(err) => panic!("damaged AX100 packet should be corrected: {err}"),
        };

        assert_eq!(frame.payload, payload);
        assert!(frame.corrected_errors > 0);
    }

    #[test]
    fn soft_recovers_packet_beyond_hard_capacity() {
        // Construct a valid AX.100 ASM+Golay packet (216-byte CSP-style
        // payload to exercise a near-full RS codeword), then corrupt 24
        // payload bytes. Hard-decision RS caps at 16 errors, so it must
        // fail; the soft-decision path with K=24 erasures should
        // recover the original payload, since 2*0 + 24 <= 32 = nroots.
        let payload: Vec<u8> = (0..216).map(|i| (i ^ 0xa5) as u8).collect();

        // Encode + scramble
        let mut encoded_payload = reed_solomon::encode_shortened(&payload, 1);
        ccsds_randomizer::xor_sequence(&mut encoded_payload);
        let header = golay::encode(encoded_payload.len() as u16);
        let mut packet = vec![
            ((header >> 16) & 0xff) as u8,
            ((header >> 8) & 0xff) as u8,
            (header & 0xff) as u8,
        ];
        packet.extend_from_slice(&encoded_payload);

        // Corrupt 24 bytes inside the RS codeword region. These will
        // be the lowest-confidence bytes via the soft samples below.
        let corruption_positions: Vec<usize> = (0..24).map(|k| 5 + k * 9).collect();
        for &i in &corruption_positions {
            let pos = ASM_GOLAY_HEADER_LEN + i;
            packet[pos] ^= 0xc3;
        }

        // Build a soft sample vector aligned with packet bits: high
        // confidence (+/-1.0) everywhere, low confidence (+/-0.05) at
        // the eight bits of each corrupted byte. The decoder ranks by
        // ascending min |soft|, so corrupted bytes sort first.
        let mut soft = vec![0.0f32; packet.len() * 8];
        for (byte_idx, byte) in packet.iter().enumerate() {
            for bit_idx in 0..8 {
                let bit = (byte >> (7 - bit_idx)) & 1;
                let sign = if bit == 1 { 1.0 } else { -1.0 };
                soft[byte_idx * 8 + bit_idx] = sign * 1.0;
            }
        }
        for &i in &corruption_positions {
            let bit_base = (ASM_GOLAY_HEADER_LEN + i) * 8;
            for j in 0..8 {
                soft[bit_base + j] = soft[bit_base + j].signum() * 0.05;
            }
        }

        let decoder = Ax100Decoder::new();

        // Hard decision must fail: 24 byte errors > 16-error cap.
        assert!(
            decoder.decode_asm_golay(&packet).is_err(),
            "hard decision must fail past 16-error capacity"
        );

        // Soft decision with K=24 erasures must succeed.
        let frame = match decoder.decode_asm_golay_with_soft(&packet, &soft, 24) {
            Ok(f) => f,
            Err(err) => panic!("soft decision with K=24 must recover: {err}"),
        };
        assert_eq!(frame.mode, Ax100Mode::AsmGolay);
        assert_eq!(frame.payload, payload);
        assert!(frame.flags.scrambler);
        assert!(frame.flags.reed_solomon);
        assert!(frame.corrected_errors >= 24);
    }

    #[test]
    fn soft_with_zero_erasures_matches_hard_decision() {
        // With K=0 and no errors, the soft path must agree with the
        // hard-decision wrapper bit-for-bit.
        let payload = b"asm-golay-payload";
        let mut encoded_payload = reed_solomon::encode_shortened(payload, 1);
        ccsds_randomizer::xor_sequence(&mut encoded_payload);
        let header = golay::encode(encoded_payload.len() as u16);
        let mut packet = vec![
            ((header >> 16) & 0xff) as u8,
            ((header >> 8) & 0xff) as u8,
            (header & 0xff) as u8,
        ];
        packet.extend_from_slice(&encoded_payload);

        let mut soft = vec![0.0f32; packet.len() * 8];
        for (byte_idx, byte) in packet.iter().enumerate() {
            for bit_idx in 0..8 {
                let bit = (byte >> (7 - bit_idx)) & 1;
                let sign = if bit == 1 { 1.0 } else { -1.0 };
                soft[byte_idx * 8 + bit_idx] = sign;
            }
        }

        let decoder = Ax100Decoder::new();
        let hard = decoder.decode_asm_golay(&packet).expect("hard ok");
        let soft_frame = decoder
            .decode_asm_golay_with_soft(&packet, &soft, 0)
            .expect("soft ok");
        assert_eq!(hard.payload, soft_frame.payload);
        assert_eq!(hard.payload, payload);
    }
}
