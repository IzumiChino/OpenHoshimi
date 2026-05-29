//! CCSDS Concatenated codec (Viterbi r=1/2 K=7 inner + RS(255,223) outer).
//!
//! Implements the standard CCSDS TM Synchronization and Channel Coding
//! concatenated scheme used by ASRTU-1 (AO-123) and other satellites:
//!
//! TX: payload → RS(255,223) → CCSDS scramble → differential encode → Viterbi encode
//! RX: Viterbi decode → differential decode → CCSDS descramble → RS(255,223) → payload
//!
//! The ASM (0x1ACFFC1D) is attached AFTER Viterbi encoding and is NOT
//! convolutionally coded. The framer (SyncwordFramer) detects the ASM and
//! delivers the coded bits to this codec.

use openhoshimi_core::DecodeError;

use crate::fec::{ccsds_randomizer, reed_solomon, viterbi, ReedSolomon};

/// Reed-Solomon basis selection for the outer code.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RsBasis {
    /// Conventional basis (used by ASRTU-1, most CCSDS satellites).
    Conventional,
    /// Dual basis (used by some older satellites).
    Dual,
}

/// CCSDS Concatenated frame decoder.
#[derive(Debug, Clone)]
pub struct CcsdsConcatDecoder {
    /// Expected payload size after RS decoding (e.g. 223 for RS(255,223)).
    frame_size: usize,
    /// Apply differential decoding after Viterbi (resolves BPSK 180 deg
    /// phase ambiguity).
    differential: bool,
    /// Reed-Solomon basis (reserved for future dual-basis support).
    #[allow(dead_code)]
    rs_basis: RsBasis,
}

/// A successfully decoded CCSDS Concatenated frame.
#[derive(Debug, Clone)]
pub struct CcsdsConcatFrame {
    /// Decoded payload bytes.
    pub payload: Vec<u8>,
    /// Number of Reed-Solomon byte errors corrected.
    pub corrected_errors: usize,
}

/// Number of coded channel bits for one RS(255,223) codeword at rate 1/2.
/// 255 bytes * 8 bits * 2 (rate-1/2) = 4080 channel bits.
pub const CCSDS_CONCAT_CODED_BITS: usize = 255 * 8 * 2;

/// The standard CCSDS TM attached sync marker.
pub const CCSDS_ASM: u32 = 0x1ACF_FC1D;

impl CcsdsConcatDecoder {
    /// Create a decoder with the given parameters.
    pub fn new(frame_size: usize, differential: bool, rs_basis: RsBasis) -> Self {
        Self {
            frame_size,
            differential,
            rs_basis,
        }
    }

    /// Decode from hard-decision channel bits (one bit per byte, 0 or 1).
    ///
    /// Input length must be [`CCSDS_CONCAT_CODED_BITS`] (4080).
    pub fn decode(&self, coded_bits: &[u8]) -> Result<CcsdsConcatFrame, DecodeError> {
        if coded_bits.len() != CCSDS_CONCAT_CODED_BITS {
            return Err(DecodeError::InvalidEncoding(format!(
                "CCSDS Concatenated expects {} coded bits, got {}",
                CCSDS_CONCAT_CODED_BITS,
                coded_bits.len()
            )));
        }
        let decoded_bits = viterbi::decode_bits(coded_bits)?;
        self.post_viterbi(&decoded_bits)
    }

    /// Decode from soft-decision channel symbols (signed i8, one per coded
    /// bit). Positive = bit 0, negative = bit 1, magnitude = confidence.
    ///
    /// Input length must be [`CCSDS_CONCAT_CODED_BITS`] (4080).
    pub fn decode_soft(&self, soft: &[i8]) -> Result<CcsdsConcatFrame, DecodeError> {
        if soft.len() != CCSDS_CONCAT_CODED_BITS {
            return Err(DecodeError::InvalidEncoding(format!(
                "CCSDS Concatenated expects {} soft symbols, got {}",
                CCSDS_CONCAT_CODED_BITS,
                soft.len()
            )));
        }
        let decoded_bits = viterbi::decode_soft_bits(soft)?;
        self.post_viterbi(&decoded_bits)
    }

    /// Post-Viterbi processing: differential decode → pack → descramble → RS.
    fn post_viterbi(&self, bits: &[u8]) -> Result<CcsdsConcatFrame, DecodeError> {
        let bits = if self.differential {
            differential_decode(bits)
        } else {
            bits.to_vec()
        };

        // Pack one-bit-per-byte into bytes (MSB first).
        let byte_count = bits.len() / 8;
        let mut bytes = vec![0u8; byte_count];
        for (i, chunk) in bits.chunks_exact(8).enumerate() {
            let mut byte = 0u8;
            for &bit in chunk {
                byte = (byte << 1) | (bit & 1);
            }
            bytes[i] = byte;
        }

        // CCSDS descramble (self-inverse XOR sequence).
        ccsds_randomizer::xor_sequence(&mut bytes);

        // RS(255,223) decode. The codeword is 255 bytes; after RS we get
        // frame_size bytes of payload.
        if bytes.len() < reed_solomon::PARITY_LEN + self.frame_size {
            return Err(DecodeError::TooShort(bytes.len()));
        }
        let rs = ReedSolomon::new(1);
        let decoded = rs.decode_shortened(&bytes)?;

        if decoded.message.len() < self.frame_size {
            return Err(DecodeError::TooShort(decoded.message.len()));
        }

        Ok(CcsdsConcatFrame {
            payload: decoded.message[..self.frame_size].to_vec(),
            corrected_errors: decoded.corrected_errors,
        })
    }
}

/// Differential decode: output[i] = input[i] XOR input[i-1].
/// Resolves 180-degree BPSK phase ambiguity.
fn differential_decode(bits: &[u8]) -> Vec<u8> {
    if bits.is_empty() {
        return Vec::new();
    }
    let mut out = Vec::with_capacity(bits.len());
    out.push(bits[0]); // first bit has no reference; keep as-is
    for i in 1..bits.len() {
        out.push((bits[i] ^ bits[i - 1]) & 1);
    }
    out
}

/// Differential encode (for test roundtrip): output[i] = output[i-1] XOR input[i].
#[cfg(test)]
fn differential_encode(bits: &[u8]) -> Vec<u8> {
    if bits.is_empty() {
        return Vec::new();
    }
    let mut out = Vec::with_capacity(bits.len());
    out.push(bits[0]);
    for i in 1..bits.len() {
        out.push((out[i - 1] ^ bits[i]) & 1);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_hard_decision() {
        // Build a 223-byte payload, RS encode, scramble, differential encode,
        // Viterbi encode, then decode and verify.
        let payload: Vec<u8> = (0..223u8).collect();

        // RS encode (223 → 255 bytes).
        let rs_codeword = reed_solomon::encode_shortened(&payload, 1);
        assert_eq!(rs_codeword.len(), 255);

        // CCSDS scramble.
        let mut scrambled = rs_codeword.clone();
        ccsds_randomizer::xor_sequence(&mut scrambled);

        // Unpack to bits (MSB first).
        let mut bits: Vec<u8> = Vec::with_capacity(scrambled.len() * 8);
        for byte in &scrambled {
            for shift in (0..8).rev() {
                bits.push((byte >> shift) & 1);
            }
        }

        // Differential encode.
        let diff_bits = differential_encode(&bits);

        // Viterbi encode (rate 1/2).
        let coded = viterbi::encode_bits(&diff_bits);
        assert_eq!(coded.len(), CCSDS_CONCAT_CODED_BITS);

        // Decode.
        let decoder = CcsdsConcatDecoder::new(223, true, RsBasis::Conventional);
        let frame = decoder.decode(&coded).expect("roundtrip decode");
        assert_eq!(frame.payload, payload);
        assert_eq!(frame.corrected_errors, 0);
    }

    #[test]
    fn roundtrip_with_bit_errors() {
        let payload: Vec<u8> = (0..223u8).map(|i| i.wrapping_mul(7)).collect();
        let rs_codeword = reed_solomon::encode_shortened(&payload, 1);
        let mut scrambled = rs_codeword;
        ccsds_randomizer::xor_sequence(&mut scrambled);
        let mut bits: Vec<u8> = Vec::with_capacity(scrambled.len() * 8);
        for byte in &scrambled {
            for shift in (0..8).rev() {
                bits.push((byte >> shift) & 1);
            }
        }
        let diff_bits = differential_encode(&bits);
        let mut coded = viterbi::encode_bits(&diff_bits);

        // Introduce a few bit errors in the coded stream.
        coded[100] ^= 1;
        coded[200] ^= 1;
        coded[500] ^= 1;
        coded[1000] ^= 1;
        coded[2000] ^= 1;

        let decoder = CcsdsConcatDecoder::new(223, true, RsBasis::Conventional);
        let frame = decoder.decode(&coded).expect("should correct errors");
        assert_eq!(frame.payload, payload);
    }

    #[test]
    fn rejects_wrong_length() {
        let decoder = CcsdsConcatDecoder::new(223, true, RsBasis::Conventional);
        assert!(decoder.decode(&[0u8; 100]).is_err());
        assert!(decoder.decode_soft(&[0i8; 100]).is_err());
    }

    #[test]
    fn differential_encode_decode_roundtrip() {
        let bits: Vec<u8> = (0..100u8).map(|i| i & 1).collect();
        let encoded = differential_encode(&bits);
        let decoded = differential_decode(&encoded);
        // First bit may differ (no reference), rest must match.
        assert_eq!(&decoded[1..], &bits[1..]);
    }
}
