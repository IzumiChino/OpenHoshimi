//! Geoscan custom-frame decoder.
//!
//! Geoscan-family satellites (Geoscan-Edelveis, StratoSat TK-1, ...) use
//! a TI CC1125 transceiver in fixed-packet mode with the radio's hardware
//! data-whitening engine and CRC-16 turned on. Over the air a frame is:
//!
//! 1. Preamble (handled by the demodulator's bit clock).
//! 2. 32-bit syncword `0x930B_51DE` (handled by [`SyncwordFramer`]).
//! 3. 64-byte payload, whitened with the CC11xx 9-bit PN9 sequence.
//! 4. 2-byte CRC-16 (poly `0x8005`, init `0xFFFF`, MSB-first), also
//!    whitened, computed over the unwhitened 64-byte payload.
//!
//! The framer hands this codec the 66 packed bytes that follow the sync
//! word. The codec runs the PN9 descrambler over the whole 66-byte block
//! and then verifies the trailing CRC against the leading 64 bytes,
//! returning the descrambled payload either way so the operator can
//! still inspect raw bytes from a CRC-failing frame.
//!
//! [`SyncwordFramer`]: openhoshimi_dsp::SyncwordFramer

use openhoshimi_core::{DecodeError, Frame};

use crate::Pn9Whitener;

/// Fixed payload length ("64 bytes without CRC" in the official
/// protocol document).
pub const GEOSCAN_PAYLOAD_LEN: usize = 64;

/// CRC-16 trailer length in bytes.
pub const GEOSCAN_CRC_LEN: usize = 2;

/// Total length the framer must hand the codec.
pub const GEOSCAN_FRAME_LEN: usize = GEOSCAN_PAYLOAD_LEN + GEOSCAN_CRC_LEN;

/// One descrambled Geoscan frame.
#[derive(Debug, Clone)]
pub struct GeoscanFrame {
    /// Descrambled 64-byte payload.
    pub payload: Vec<u8>,
    /// Whether the trailing CC11xx CRC-16 matched the recomputed value.
    pub crc_ok: bool,
    /// CRC value carried in the frame (MSB-first u16).
    pub crc_received: u16,
    /// CRC value recomputed by the decoder over the payload.
    pub crc_expected: u16,
}

/// Stateless Geoscan frame decoder.
#[derive(Debug, Default, Clone, Copy)]
pub struct GeoscanDecoder;

impl GeoscanDecoder {
    /// Construct a new decoder. Each call to [`decode`](Self::decode) is
    /// independent — the PN9 LFSR is reset at the start of every frame,
    /// matching CC1125 hardware behaviour where whitening restarts on the
    /// first byte after the syncword.
    pub fn new() -> Self {
        Self
    }

    /// Descramble and CRC-check one Geoscan frame.
    ///
    /// `raw` must be exactly [`GEOSCAN_FRAME_LEN`] bytes. The descrambled
    /// payload is returned even when the CRC fails so operators can still
    /// look at raw bytes from a corrupted frame; the [`GeoscanFrame::crc_ok`]
    /// flag tells callers whether the bytes can be trusted.
    ///
    /// # Errors
    ///
    /// Returns [`DecodeError::TooShort`] if `raw` is shorter than
    /// [`GEOSCAN_FRAME_LEN`].
    pub fn decode(&self, raw: &[u8]) -> Result<GeoscanFrame, DecodeError> {
        if raw.len() < GEOSCAN_FRAME_LEN {
            return Err(DecodeError::TooShort(raw.len()));
        }

        let mut buf = raw[..GEOSCAN_FRAME_LEN].to_vec();
        Pn9Whitener::new().descramble(&mut buf);

        let crc_received =
            u16::from_be_bytes([buf[GEOSCAN_PAYLOAD_LEN], buf[GEOSCAN_PAYLOAD_LEN + 1]]);
        let crc_expected = cc11xx_crc16(&buf[..GEOSCAN_PAYLOAD_LEN]);
        let payload = buf[..GEOSCAN_PAYLOAD_LEN].to_vec();

        Ok(GeoscanFrame {
            payload,
            crc_ok: crc_received == crc_expected,
            crc_received,
            crc_expected,
        })
    }

    /// Convenience helper: decode a [`Frame`] produced by the runtime
    /// pipeline (whose `raw` field carries the 66 packed bytes the
    /// [`SyncwordFramer`] emitted).
    ///
    /// [`SyncwordFramer`]: openhoshimi_dsp::SyncwordFramer
    ///
    /// # Errors
    ///
    /// Forwards any error from [`decode`](Self::decode).
    pub fn decode_frame(&self, frame: &Frame) -> Result<GeoscanFrame, DecodeError> {
        self.decode(&frame.raw)
    }
}

/// CC11xx CRC-16: polynomial `0x8005`, initial value `0xFFFF`, no input
/// or output reflection, MSB-first byte processing, no final XOR.
///
/// Matches gr-satellites' `crc16_cc11xx` and the CC1101 datasheet
/// ("CRC computation" appendix).
fn cc11xx_crc16(data: &[u8]) -> u16 {
    let mut crc: u16 = 0xFFFF;
    for byte in data {
        crc ^= (*byte as u16) << 8;
        for _ in 0..8 {
            if crc & 0x8000 != 0 {
                crc = (crc << 1) ^ 0x8005;
            } else {
                crc <<= 1;
            }
        }
    }
    crc
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn crc_known_vector_ones() {
        // Sanity: per the CC1101 datasheet "check" example, the CRC of
        // "123456789" under this parameter set is 0xAEE7. Acts as a
        // regression guard for the polynomial constants.
        let crc = cc11xx_crc16(b"123456789");
        assert_eq!(crc, 0xAEE7);
    }

    #[test]
    fn round_trip_known_payload() {
        // Build a synthetic frame: payload of incrementing bytes, append
        // its CRC, whiten the whole 66-byte block, then feed it back to
        // the decoder. The decoder must restore the original payload and
        // report a passing CRC.
        let mut payload = [0u8; GEOSCAN_PAYLOAD_LEN];
        for (i, b) in payload.iter_mut().enumerate() {
            *b = i as u8;
        }
        let crc = cc11xx_crc16(&payload);

        let mut frame = [0u8; GEOSCAN_FRAME_LEN];
        frame[..GEOSCAN_PAYLOAD_LEN].copy_from_slice(&payload);
        frame[GEOSCAN_PAYLOAD_LEN..].copy_from_slice(&crc.to_be_bytes());
        Pn9Whitener::new().descramble(&mut frame);

        let decoded = match GeoscanDecoder::new().decode(&frame) {
            Ok(f) => f,
            Err(e) => panic!("decode failed: {e}"),
        };
        assert!(
            decoded.crc_ok,
            "crc_received={:#06x} crc_expected={:#06x}",
            decoded.crc_received, decoded.crc_expected
        );
        assert_eq!(decoded.payload, payload);
    }

    #[test]
    fn crc_failure_still_yields_payload() {
        let payload = [0xAA; GEOSCAN_PAYLOAD_LEN];
        let crc = cc11xx_crc16(&payload);

        let mut frame = [0u8; GEOSCAN_FRAME_LEN];
        frame[..GEOSCAN_PAYLOAD_LEN].copy_from_slice(&payload);
        frame[GEOSCAN_PAYLOAD_LEN..].copy_from_slice(&crc.to_be_bytes());
        Pn9Whitener::new().descramble(&mut frame);

        // Corrupt one byte after whitening to simulate a channel error.
        frame[10] ^= 0x01;

        let decoded = match GeoscanDecoder::new().decode(&frame) {
            Ok(f) => f,
            Err(e) => panic!("decode failed: {e}"),
        };
        assert!(!decoded.crc_ok);
        assert_eq!(decoded.payload.len(), GEOSCAN_PAYLOAD_LEN);
    }

    #[test]
    fn rejects_short_input() {
        let buf = [0u8; GEOSCAN_FRAME_LEN - 1];
        match GeoscanDecoder::new().decode(&buf) {
            Err(DecodeError::TooShort(n)) => assert_eq!(n, GEOSCAN_FRAME_LEN - 1),
            other => panic!("expected TooShort, got {other:?}"),
        }
    }
}
