//! AO-40 FEC frame decoder.
//!
//! AO-40 FEC, used by FUNcube-style beacons, combines a distributed syncword,
//! block interleaving, a CCSDS rate-1/2 convolutional code, the CCSDS
//! randomizer, and two shortened/interleaved RS(160,128) codewords.

use openhoshimi_core::DecodeError;

use crate::fec::{ccsds_randomizer, reed_solomon, viterbi, ReedSolomon, Viterbi};

const AO40_PAYLOAD_LEN: usize = 256;
const AO40_RS_CODEWORD_LEN: usize = 320;
const AO40_INTERLEAVE: usize = 2;
const AO40_POST_VITERBI_BITS: usize = 2566;
const AO40_POST_VITERBI_BYTES: usize = 321;
const AO40_VITERBI_SYMBOLS: usize = AO40_POST_VITERBI_BITS * 2;
const AO40_SYNCWORD: &str = "11111110000111011110010110010010000001000100110001011101011011000";

/// Encoder for AO-40 FEC frames; emits the 5132-bit channel stream that
/// would be transmitted (after the distributed-sync interleaver wraps it
/// into a 5200-bit block). Used by codec roundtrip tests and by
/// satellite-side simulation tooling.
#[derive(Debug, Default, Clone, Copy)]
pub struct Ao40FecEncoder;

impl Ao40FecEncoder {
    /// Construct a new AO-40 FEC encoder.
    pub fn new() -> Self {
        Self
    }

    /// Encode a 256-byte payload into the 5132 channel bits emitted by the
    /// transmitter before distributed-sync interleaving. Each output element
    /// is a single bit packed in the low bit of a byte (0 or 1), matching
    /// the input format expected by [`Ao40FecDecoder::decode_channel_bits`].
    pub fn encode_to_channel_bits(&self, payload: &[u8]) -> Result<Vec<u8>, DecodeError> {
        if payload.len() != AO40_PAYLOAD_LEN {
            return Err(DecodeError::InvalidEncoding(format!(
                "AO-40 payload must be {AO40_PAYLOAD_LEN} bytes, got {}",
                payload.len()
            )));
        }

        let mut codeword = reed_solomon::encode_shortened(payload, AO40_INTERLEAVE);
        ccsds_randomizer::xor_sequence(&mut codeword);

        let mut bits = Vec::with_capacity(AO40_POST_VITERBI_BITS);
        for byte in &codeword {
            for shift in (0..8).rev() {
                bits.push((byte >> shift) & 1);
            }
        }
        bits.extend(std::iter::repeat(0u8).take(AO40_POST_VITERBI_BITS - bits.len()));

        Ok(viterbi::encode_bits(&bits))
    }
}

/// Decoder for AO-40 FEC post-Viterbi data.
#[derive(Debug, Default, Clone, Copy)]
pub struct Ao40FecDecoder;

impl Ao40FecDecoder {
    /// Construct a new AO-40 FEC decoder.
    pub fn new() -> Self {
        Self
    }

    /// Decode AO-40 data after Viterbi decoding.
    ///
    /// The input is the 2566-bit stream emitted by the convolutional
    /// decoder, packed MSB-first into bytes. Only the first 320 bytes are
    /// randomized RS codeword data; the final six tail bits are ignored.
    pub fn decode_post_viterbi_bytes(&self, bytes: &[u8]) -> Result<Ao40Frame, DecodeError> {
        if bytes.len() < AO40_POST_VITERBI_BYTES {
            return Err(DecodeError::TooShort(bytes.len()));
        }

        let mut codeword = bytes[..AO40_RS_CODEWORD_LEN].to_vec();
        ccsds_randomizer::xor_sequence(&mut codeword);
        let decoded = ReedSolomon::new(AO40_INTERLEAVE).decode_shortened(&codeword)?;

        if decoded.message.len() != AO40_PAYLOAD_LEN {
            return Err(DecodeError::InvalidEncoding(format!(
                "AO-40 payload length is {} bytes, expected {AO40_PAYLOAD_LEN}",
                decoded.message.len()
            )));
        }

        Ok(Ao40Frame {
            payload: decoded.message,
            corrected_errors: decoded.corrected_errors,
        })
    }

    /// Decode AO-40 data after Viterbi decoding from one-bit-per-byte bits.
    ///
    /// Bits are packed MSB-first before delegating to
    /// [`decode_post_viterbi_bytes`](Self::decode_post_viterbi_bytes).
    pub fn decode_post_viterbi_bits(&self, bits: &[u8]) -> Result<Ao40Frame, DecodeError> {
        if bits.len() < AO40_POST_VITERBI_BITS {
            return Err(DecodeError::TooShort(bits.len() / 8));
        }
        let packed = pack_msb_bits(&bits[..AO40_POST_VITERBI_BITS]);
        self.decode_post_viterbi_bytes(&packed)
    }

    /// Decode AO-40 data after distributed-sync framing.
    ///
    /// The input is the data symbols recovered from the 5200-bit
    /// transmitted block after removing the distributed sync vector and
    /// deinterleaving. Any trailing filler symbols are ignored before
    /// Viterbi decoding.
    pub fn decode_channel_bits(&self, bits: &[u8]) -> Result<Ao40Frame, DecodeError> {
        if bits.len() < AO40_VITERBI_SYMBOLS {
            return Err(DecodeError::TooShort(bits.len() / 8));
        }

        let decoded = Viterbi::new().decode_bits(&bits[..AO40_VITERBI_SYMBOLS])?;
        self.decode_post_viterbi_bits(&decoded)
    }

    /// Decode AO-40 data after distributed-sync framing using soft
    /// per-bit confidence values.
    ///
    /// Each input element is a signed correlation value where positive
    /// means the channel bit was received as `0` and negative means `1`;
    /// magnitude is the per-bit reliability. Soft Viterbi decoding
    /// recovers about 2 dB of coding gain compared with the hard-decision
    /// path used by [`decode_channel_bits`](Self::decode_channel_bits),
    /// which matters at the AO-73 link margin where many borderline
    /// symbols would otherwise flip with no confidence information.
    pub fn decode_soft_channel_bits(&self, symbols: &[i8]) -> Result<Ao40Frame, DecodeError> {
        if symbols.len() < AO40_VITERBI_SYMBOLS {
            return Err(DecodeError::TooShort(symbols.len() / 8));
        }

        let decoded = Viterbi::new().decode_soft_bits(&symbols[..AO40_VITERBI_SYMBOLS])?;
        self.decode_post_viterbi_bits(&decoded)
    }
}

/// Decoded AO-40 FEC payload.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Ao40Frame {
    /// The 256-byte user payload after de-randomizing and RS validation.
    pub payload: Vec<u8>,
    /// Number of Reed-Solomon byte errors corrected by this decoder.
    ///
    /// The decoder uses Berlekamp-Massey + Chien search + Forney algorithm,
    /// so up to 16 byte errors can be corrected per RS(160,128) codeword,
    /// or 32 byte errors total across the two interleaved codewords.
    pub corrected_errors: usize,
}

/// AO-40 distributed syncword as one-bit-per-byte values.
pub fn ao40_syncword_bits() -> Vec<u8> {
    AO40_SYNCWORD
        .bytes()
        .map(|byte| u8::from(byte == b'1'))
        .collect()
}

fn pack_msb_bits(bits: &[u8]) -> Vec<u8> {
    bits.chunks(8)
        .map(|chunk| {
            let mut byte = 0u8;
            for (index, bit) in chunk.iter().enumerate() {
                byte |= (bit & 1) << (7 - index);
            }
            byte
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fec::{ccsds_randomizer, reed_solomon};

    fn unpack_msb_bits(bytes: &[u8]) -> Vec<u8> {
        let mut bits = Vec::with_capacity(bytes.len() * 8);
        for &byte in bytes {
            for bit in (0..8).rev() {
                bits.push((byte >> bit) & 1);
            }
        }
        bits
    }

    #[test]
    fn decodes_post_viterbi_bytes() {
        let payload: Vec<u8> = (0..=255).collect();
        let mut codeword = reed_solomon::encode_shortened(&payload, AO40_INTERLEAVE);
        ccsds_randomizer::xor_sequence(&mut codeword);
        let mut post_viterbi = codeword;
        post_viterbi.push(0);

        let decoder = Ao40FecDecoder::new();
        let frame = match decoder.decode_post_viterbi_bytes(&post_viterbi) {
            Ok(frame) => frame,
            Err(err) => panic!("valid AO-40 post-Viterbi data: {err}"),
        };

        assert_eq!(frame.payload, payload);
        assert_eq!(frame.corrected_errors, 0);
    }

    #[test]
    fn decodes_post_viterbi_bits() {
        let payload: Vec<u8> = (0..=255).rev().collect();
        let mut codeword = reed_solomon::encode_shortened(&payload, AO40_INTERLEAVE);
        ccsds_randomizer::xor_sequence(&mut codeword);
        let mut post_viterbi = codeword;
        post_viterbi.push(0);
        let bits = unpack_msb_bits(&post_viterbi);

        let decoder = Ao40FecDecoder::new();
        let frame = match decoder.decode_post_viterbi_bits(&bits[..AO40_POST_VITERBI_BITS]) {
            Ok(frame) => frame,
            Err(err) => panic!("valid AO-40 post-Viterbi bits: {err}"),
        };

        assert_eq!(frame.payload, payload);
    }

    #[test]
    fn decodes_channel_bits() {
        let payload: Vec<u8> = (0..=255).collect();
        let mut codeword = reed_solomon::encode_shortened(&payload, AO40_INTERLEAVE);
        ccsds_randomizer::xor_sequence(&mut codeword);
        let mut post_viterbi = codeword;
        post_viterbi.push(0);
        let bits = unpack_msb_bits(&post_viterbi);
        let mut symbols = crate::fec::viterbi::encode_bits(&bits[..AO40_POST_VITERBI_BITS]);
        symbols.push(0);

        let decoder = Ao40FecDecoder::new();
        let frame = match decoder.decode_channel_bits(&symbols) {
            Ok(frame) => frame,
            Err(err) => panic!("valid AO-40 channel bits: {err}"),
        };

        assert_eq!(frame.payload, payload);
    }

    #[test]
    fn syncword_has_expected_length() {
        let sync = ao40_syncword_bits();

        assert_eq!(sync.len(), 65);
        assert_eq!(&sync[..7], &[1, 1, 1, 1, 1, 1, 1]);
    }
}
