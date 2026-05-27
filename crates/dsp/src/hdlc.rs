//! HDLC framer for AX.25-style one-bit-per-byte streams.
//!
//! The framer expects input bits as bytes whose low bit is significant:
//! `0x00` for zero and any odd byte for one. It detects the HDLC flag
//! `0x7e`, removes inserted zero bits after five consecutive ones, checks
//! the AX.25 FCS, and returns payload-only [`openhoshimi_core::Frame`]
//! values.

use std::time::SystemTime;

use crc::{Crc, CRC_16_IBM_SDLC};
use openhoshimi_core::{DecodeError, Frame, FrameType, Framing};

const HDLC_FLAG: u8 = 0x7e;
const MIN_AX25_PAYLOAD_LEN: usize = 17;
const FCS_LEN: usize = 2;
const AX25_FCS: Crc<u16> = Crc::<u16>::new(&CRC_16_IBM_SDLC);

/// Result of validating a candidate HDLC frame: the raw byte buffer
/// (post-unstuff, including FCS) plus the validation result. Decode errors
/// keep the buffer attached so callers can diagnose CRC failures by hex
/// comparison against a reference decoder.
type HdlcCandidate = (Vec<u8>, Result<Vec<u8>, DecodeError>);

/// HDLC framer for Bell 202 / AX.25 bit streams.
pub struct HdlcFramer {
    shift_register: u8,
    in_frame: bool,
    raw_bits: Vec<u8>,
    /// Last decode error, if a candidate frame failed validation.
    pub last_error: Option<DecodeError>,
    /// Raw bytes (after bit-unstuffing) of the most recent frame that failed
    /// validation. Useful for diagnosing CRC mismatches against a reference
    /// decoder by inspecting the candidate payload.
    pub last_failed_bytes: Option<Vec<u8>>,
    /// Raw bytes of the longest failed candidate observed across the entire
    /// stream. The most recent failure is often a short tail fragment that
    /// overwrites the more interesting full-length frame, so we keep the
    /// longest one separately for offline diff against a reference decoder.
    pub longest_failed_bytes: Option<Vec<u8>>,
}

impl HdlcFramer {
    /// Construct a new HDLC framer.
    pub fn new() -> Self {
        Self {
            shift_register: 0,
            in_frame: false,
            raw_bits: Vec::new(),
            last_error: None,
            last_failed_bytes: None,
            longest_failed_bytes: None,
        }
    }

    /// Return the most recent frame decode error, if any.
    pub fn last_error(&self) -> Option<&DecodeError> {
        self.last_error.as_ref()
    }

    /// Return the raw byte buffer of the most recent failed frame, if any.
    pub fn last_failed_bytes(&self) -> Option<&[u8]> {
        self.last_failed_bytes.as_deref()
    }

    /// Return the raw byte buffer of the longest failed frame seen so far,
    /// if any. Useful when the most recent failure is a short tail fragment
    /// and the diagnostically interesting candidate has a length matching a
    /// reference decoder.
    pub fn longest_failed_bytes(&self) -> Option<&[u8]> {
        self.longest_failed_bytes.as_deref()
    }

    fn push_bit(&mut self, bit: u8) -> Option<HdlcCandidate> {
        let bit = bit & 1;
        self.shift_register = (self.shift_register >> 1) | (bit << 7);

        if self.shift_register == HDLC_FLAG {
            let candidate = if self.in_frame {
                let len = self.raw_bits.len();
                let frame_bits = if len >= 7 {
                    self.raw_bits[..len - 7].to_vec()
                } else {
                    Vec::new()
                };
                Some(Self::decode_candidate(&frame_bits))
            } else {
                None
            };

            self.in_frame = true;
            self.raw_bits.clear();
            return candidate;
        }

        if self.in_frame {
            self.raw_bits.push(bit);
        }

        None
    }

    fn decode_candidate(bits: &[u8]) -> HdlcCandidate {
        let unstuffed = unstuff_bits(bits);
        let bytes = bits_to_bytes(&unstuffed);
        if bytes.len() < MIN_AX25_PAYLOAD_LEN + FCS_LEN {
            let len = bytes.len().saturating_sub(FCS_LEN);
            return (bytes, Err(DecodeError::TooShort(len)));
        }

        let payload_len = bytes.len() - FCS_LEN;
        let payload = &bytes[..payload_len];
        let received = u16::from_le_bytes([bytes[payload_len], bytes[payload_len + 1]]);
        let calculated = AX25_FCS.checksum(payload);
        if calculated != received {
            return (bytes.clone(), Err(DecodeError::CrcMismatch));
        }

        if payload.len() < MIN_AX25_PAYLOAD_LEN {
            return (bytes.clone(), Err(DecodeError::TooShort(payload.len())));
        }

        (bytes.clone(), Ok(payload.to_vec()))
    }
}

impl Default for HdlcFramer {
    fn default() -> Self {
        Self::new()
    }
}

impl Framing for HdlcFramer {
    fn push_bytes(&mut self, bytes: &[u8]) -> Vec<Frame> {
        let mut frames = Vec::new();

        for &byte in bytes {
            if let Some((candidate_bytes, result)) = self.push_bit(byte) {
                match result {
                    Ok(raw) => {
                        self.last_error = None;
                        self.last_failed_bytes = None;
                        frames.push(Frame {
                            satellite_id: 0,
                            timestamp: SystemTime::now(),
                            rssi_dbm: None,
                            raw,
                            frame_type: FrameType::Ax25,
                        });
                    }
                    Err(err) => {
                        self.last_error = Some(err);
                        let longest_len = self
                            .longest_failed_bytes
                            .as_ref()
                            .map(|b| b.len())
                            .unwrap_or(0);
                        if candidate_bytes.len() > longest_len {
                            self.longest_failed_bytes = Some(candidate_bytes.clone());
                        }
                        self.last_failed_bytes = Some(candidate_bytes);
                    }
                }
            }
        }

        frames
    }
}

fn unstuff_bits(bits: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(bits.len());
    let mut ones = 0usize;
    let mut i = 0usize;

    while i < bits.len() {
        let bit = bits[i] & 1;
        out.push(bit);

        if bit == 1 {
            ones += 1;
            if ones == 5 {
                if bits.get(i + 1).is_some_and(|next| *next == 0) {
                    i += 1;
                }
                ones = 0;
            }
        } else {
            ones = 0;
        }

        i += 1;
    }

    out
}

fn bits_to_bytes(bits: &[u8]) -> Vec<u8> {
    bits.chunks_exact(8)
        .map(|chunk| {
            chunk
                .iter()
                .enumerate()
                .fold(0u8, |acc, (bit_index, bit)| acc | ((bit & 1) << bit_index))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use openhoshimi_core::Framing;

    fn sample_payload() -> Vec<u8> {
        vec![
            0x82, 0xa0, 0xa6, 0xa6, 0x40, 0x40, 0x60, 0x86, 0xa2, 0x62, 0x8e, 0x86, 0x40, 0x61,
            0x03, 0xf0, b'O', b'K',
        ]
    }

    fn bits_from_byte(byte: u8) -> [u8; 8] {
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

    fn encode_hdlc(payload: &[u8]) -> Vec<u8> {
        let mut bytes = payload.to_vec();
        bytes.extend_from_slice(&AX25_FCS.checksum(payload).to_le_bytes());

        let mut bits = Vec::new();
        bits.extend_from_slice(&bits_from_byte(HDLC_FLAG));
        let mut ones = 0usize;
        for byte in bytes {
            for bit in bits_from_byte(byte) {
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
        bits.extend_from_slice(&bits_from_byte(HDLC_FLAG));
        bits
    }

    #[test]
    fn known_good_hdlc_frame_returns_payload() {
        let payload = sample_payload();
        let bits = encode_hdlc(&payload);
        let mut framer = HdlcFramer::new();

        let frames = framer.push_bytes(&bits);

        assert_eq!(frames.len(), 1);
        assert_eq!(frames[0].raw, payload);
        assert_eq!(frames[0].frame_type, FrameType::Ax25);
    }

    #[test]
    fn flipped_crc_bit_reports_crc_mismatch() {
        let payload = sample_payload();
        let mut bits = encode_hdlc(&payload);
        let bit_to_flip = bits.len() - 8 - 1;
        bits[bit_to_flip] ^= 1;

        let mut framer = HdlcFramer::new();
        let frames = framer.push_bytes(&bits);

        assert!(frames.is_empty());
        assert!(matches!(
            framer.last_error(),
            Some(DecodeError::CrcMismatch)
        ));
        assert!(framer.last_failed_bytes().is_some());
        assert!(framer.longest_failed_bytes().is_some());
    }
}
