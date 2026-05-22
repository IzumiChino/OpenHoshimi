//! GOMspace NanoCom AX100 frame decoder.
//!
//! AX100 uses two common framing modes in amateur-satellite downlinks:
//! Reed-Solomon mode and ASM+Golay mode. This module decodes already
//! synchronized packet bytes; finding the AX100 ASM syncword belongs in
//! the DSP/framing layer.

use openhoshimi_core::DecodeError;

use crate::fec::{ccsds_randomizer, golay, reed_solomon, ReedSolomon};

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
    /// ASM+Golay header flags. In Reed-Solomon mode these are all false.
    pub flags: Ax100Flags,
}

/// AX100 framing mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Ax100Mode {
    /// AX100 Reed-Solomon mode.
    ReedSolomon,
    /// AX100 ASM+Golay mode.
    AsmGolay,
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
    use crate::fec::{golay, reed_solomon};

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
    fn rejects_damaged_reed_solomon_packet() {
        let payload = b"csp-payload";
        let codeword = reed_solomon::encode_shortened(payload, 1);
        let mut packet = Vec::new();
        packet.push((payload.len() + reed_solomon::PARITY_LEN + RS_HEADER_LEN) as u8);
        packet.extend_from_slice(&codeword);
        packet[4] ^= 0x20;

        let decoder = Ax100Decoder::new();
        let err = match decoder.decode_reed_solomon(&packet) {
            Ok(_) => panic!("damaged AX100 packet should fail"),
            Err(err) => err,
        };

        assert!(matches!(err, DecodeError::CrcMismatch));
    }
}
