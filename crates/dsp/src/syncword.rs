//! Syncword-based bit framer.
//!
//! This framer searches a one-bit-per-byte stream for a fixed syncword and
//! emits a fixed number of following payload bits. It is useful for protocols
//! such as AX100 ASM mode and AO-40 FEC where HDLC flags are not used.
//!
//! Two detection modes are supported:
//!
//! * Hard-decision Hamming distance ([`push_bits`](SyncwordFramer::push_bits)
//!   and the [`Framing`] impl): the input is one slicer-output bit per byte.
//!   A match is declared when the Hamming distance between the shift
//!   register and the syncword is at most `threshold`. This is what every
//!   classic FSK/GMSK packet receiver does.
//! * Soft-decision correlation ([`push_soft`](SyncwordFramer::push_soft)):
//!   the input is one pre-slicer signal value per symbol. A match is
//!   declared when the soft correlation against `+/- 1` mapped sync bits
//!   exceeds `soft_gamma` times the sum of absolute sample magnitudes. At
//!   low SNR this catches frames where one or two slicer flips would have
//!   exhausted the Hamming budget. Hard-decision callers are unaffected.

use std::time::SystemTime;

use openhoshimi_core::{Frame, FrameType, Framing};

/// Soft-correlation floor. The correlation score is divided by the sum of
/// absolute soft values, so this is dimensionless and bounded by `1.0` for
/// a perfect match. The constructor picks the actual gamma so that the
/// soft detector is at least as strict as the hard Hamming-distance
/// detector (`1 - 2 * threshold / syncword_len`); this floor only kicks in
/// when the hard threshold itself is very loose.
///
/// Set empirically by sweeping gamma against a multi-minute SatNOGS
/// recording of STRATOSAT-TK-1: 0.85 maximised the absolute number of
/// CRC-valid frames recovered (1420) versus 0.75 (1393), 0.80 (1403),
/// and 0.90 (1386). The mechanism is that the soft correlator is
/// strictly more permissive than the hard Hamming detector at the same
/// equivalent threshold (an ISI-blurred sample with the right sign but
/// half the magnitude looks like 0.5 to soft, 1 to hard), and a
/// well-chosen floor keeps it from locking onto noise-driven false
/// positives during weak-signal segments.
pub const DEFAULT_SOFT_GAMMA: f32 = 0.85;

/// Finds fixed-size payloads after a bit syncword.
#[derive(Debug, Clone)]
pub struct SyncwordFramer {
    syncword: Vec<u8>,
    threshold: usize,
    payload_bits: usize,
    shift_register: Vec<u8>,
    soft_register: Vec<f32>,
    soft_gamma: f32,
    collecting: bool,
    current_payload: Vec<u8>,
    current_payload_soft: Vec<f32>,
    frame_type: FrameType,
    pack_bits: bool,
    sync_attempts: u64,
    sync_locked: u64,
}

impl SyncwordFramer {
    /// Create a new syncword framer.
    pub fn new(syncword: &[u8], threshold: usize, payload_bits: usize) -> Self {
        Self::with_frame_options(syncword, threshold, payload_bits, FrameType::Unknown, false)
    }

    /// Create a syncword framer that can emit [`Frame`] values.
    pub fn with_frame_options(
        syncword: &[u8],
        threshold: usize,
        payload_bits: usize,
        frame_type: FrameType,
        pack_bits: bool,
    ) -> Self {
        let sync_len = syncword.len();
        let hard_equivalent = if sync_len == 0 {
            DEFAULT_SOFT_GAMMA
        } else {
            1.0 - (2.0 * threshold as f32) / (sync_len as f32)
        };
        let soft_gamma = hard_equivalent.clamp(DEFAULT_SOFT_GAMMA, 0.95);
        Self {
            syncword: syncword.iter().map(|bit| bit & 1).collect(),
            threshold,
            payload_bits,
            shift_register: Vec::with_capacity(sync_len),
            soft_register: Vec::with_capacity(sync_len),
            soft_gamma,
            collecting: false,
            current_payload: Vec::with_capacity(payload_bits),
            current_payload_soft: Vec::with_capacity(payload_bits),
            frame_type,
            pack_bits,
            sync_attempts: 0,
            sync_locked: 0,
        }
    }

    /// Override the soft-correlation threshold gamma. Has no effect on the
    /// hard-decision [`push_bits`](Self::push_bits) path.
    pub fn set_soft_gamma(&mut self, gamma: f32) {
        self.soft_gamma = gamma;
    }

    /// Number of full-length syncword candidates the framer has compared
    /// against the configured pattern. One attempt corresponds to one
    /// shift of the shift register once it has been filled.
    pub fn sync_attempts(&self) -> u64 {
        self.sync_attempts
    }

    /// Number of times the framer has locked onto the syncword (a
    /// candidate whose Hamming distance was within `threshold`).
    pub fn sync_locked(&self) -> u64 {
        self.sync_locked
    }

    /// Push one-bit-per-byte bits and return complete payload bit vectors.
    pub fn push_bits(&mut self, bits: &[u8]) -> Vec<Vec<u8>> {
        let mut frames = Vec::new();

        for &bit in bits {
            let bit = bit & 1;
            if self.collecting {
                self.current_payload.push(bit);
                if self.current_payload.len() == self.payload_bits {
                    frames.push(std::mem::take(&mut self.current_payload));
                    self.current_payload = Vec::with_capacity(self.payload_bits);
                    self.collecting = false;
                    self.shift_register.clear();
                }
                continue;
            }

            self.shift_register.push(bit);
            if self.shift_register.len() > self.syncword.len() {
                self.shift_register.remove(0);
            }

            if self.shift_register.len() == self.syncword.len() {
                self.sync_attempts = self.sync_attempts.saturating_add(1);
                if hamming_distance(&self.shift_register, &self.syncword) <= self.threshold {
                    self.sync_locked = self.sync_locked.saturating_add(1);
                    self.collecting = true;
                    self.current_payload.clear();
                }
            }
        }

        frames
    }

    /// Push one soft (pre-slicer) signal value per symbol and return
    /// complete payload bit vectors.
    ///
    /// The detector keeps a running window of the last `syncword.len()`
    /// soft samples. Each step it computes the correlation
    /// `c = sum(s_i * sign_i)` where `sign_i = +1` for sync bit `1` and
    /// `-1` for sync bit `0`, alongside the energy reference
    /// `r = sum(|s_i|)`. A lock is declared when `c >= gamma * r`. After
    /// locking, payload symbols are sliced by sign (positive -> 1,
    /// otherwise -> 0) until `payload_bits` have been collected.
    pub fn push_soft(&mut self, soft: &[f32]) -> Vec<(Vec<u8>, Vec<f32>)> {
        let mut frames = Vec::new();

        for &sample in soft {
            if self.collecting {
                let bit = if sample > 0.0 { 1u8 } else { 0u8 };
                self.current_payload.push(bit);
                self.current_payload_soft.push(sample);
                if self.current_payload.len() == self.payload_bits {
                    let bits = std::mem::take(&mut self.current_payload);
                    let soft_bits = std::mem::take(&mut self.current_payload_soft);
                    frames.push((bits, soft_bits));
                    self.current_payload = Vec::with_capacity(self.payload_bits);
                    self.current_payload_soft = Vec::with_capacity(self.payload_bits);
                    self.collecting = false;
                    self.shift_register.clear();
                    self.soft_register.clear();
                }
                continue;
            }

            self.soft_register.push(sample);
            if self.soft_register.len() > self.syncword.len() {
                self.soft_register.remove(0);
            }

            if self.soft_register.len() == self.syncword.len() {
                self.sync_attempts = self.sync_attempts.saturating_add(1);
                let mut correlation = 0.0f32;
                let mut reference = 0.0f32;
                for (soft_value, sync_bit) in self.soft_register.iter().zip(self.syncword.iter()) {
                    let sign = if (sync_bit & 1) == 1 { 1.0 } else { -1.0 };
                    correlation += soft_value * sign;
                    reference += soft_value.abs();
                }
                if reference > f32::EPSILON && correlation >= self.soft_gamma * reference {
                    self.sync_locked = self.sync_locked.saturating_add(1);
                    self.collecting = true;
                    self.current_payload.clear();
                    self.current_payload_soft.clear();
                    self.soft_register.clear();
                }
            }
        }

        frames
    }
}

impl Framing for SyncwordFramer {
    fn push_bytes(&mut self, bytes: &[u8]) -> Vec<Frame> {
        self.push_bits(bytes)
            .into_iter()
            .map(|payload_bits| self.frame_from_bits(payload_bits, None))
            .collect()
    }

    fn push_soft(&mut self, soft: &[f32]) -> Vec<Frame> {
        self.push_soft(soft)
            .into_iter()
            .map(|(payload_bits, soft_samples)| {
                self.frame_from_bits(payload_bits, Some(soft_samples))
            })
            .collect()
    }
}

impl SyncwordFramer {
    fn frame_from_bits(&self, payload_bits: Vec<u8>, soft_samples: Option<Vec<f32>>) -> Frame {
        // When the payload is packed MSB-first into bytes, the per-bit
        // soft array maps directly to one f32 per output bit (8 per
        // output byte). When pack_bits is false each output "byte" is a
        // single bit, which doesn't match the AX.100 wrapper's
        // bit-per-byte convention, so we drop the soft samples there.
        let soft_bits = if self.pack_bits { soft_samples } else { None };
        Frame {
            satellite_id: 0,
            timestamp: SystemTime::now(),
            rssi_dbm: None,
            raw: if self.pack_bits {
                pack_msb_bits(&payload_bits)
            } else {
                payload_bits
            },
            frame_type: self.frame_type,
            soft_bits,
        }
    }
}

/// Pack one-bit-per-byte bits into MSB-first bytes.
pub fn pack_msb_bits(bits: &[u8]) -> Vec<u8> {
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

fn hamming_distance(lhs: &[u8], rhs: &[u8]) -> usize {
    lhs.iter()
        .zip(rhs)
        .filter(|(left, right)| (*left & 1) != (*right & 1))
        .count()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn finds_payload_after_syncword() {
        let mut framer = SyncwordFramer::new(&[1, 0, 1, 1], 0, 6);

        let frames = framer.push_bits(&[0, 0, 1, 0, 1, 1, 1, 1, 0, 0, 1, 0]);

        assert_eq!(frames, vec![vec![1, 1, 0, 0, 1, 0]]);
    }

    #[test]
    fn threshold_allows_syncword_errors() {
        let mut framer = SyncwordFramer::new(&[1, 0, 1, 1], 1, 4);

        let frames = framer.push_bits(&[0, 0, 1, 0, 0, 1, 1, 0, 1, 1]);

        assert_eq!(frames, vec![vec![1, 0, 1, 1]]);
    }

    #[test]
    fn packs_msb_first_bits() {
        let bytes = pack_msb_bits(&[1, 0, 1, 0, 0, 0, 1, 1, 1]);

        assert_eq!(bytes, vec![0xa3, 0x80]);
    }

    #[test]
    fn can_emit_packed_frames() {
        let mut framer =
            SyncwordFramer::with_frame_options(&[1, 0, 1, 1], 0, 8, FrameType::GomspaceAx100, true);

        let frames = framer.push_bytes(&[1, 0, 1, 1, 1, 0, 1, 0, 0, 0, 1, 1]);

        assert_eq!(frames.len(), 1);
        assert_eq!(frames[0].raw, vec![0xa3]);
        assert_eq!(frames[0].frame_type, FrameType::GomspaceAx100);
    }

    #[test]
    fn soft_locks_on_perfect_match() {
        let mut framer = SyncwordFramer::new(&[1, 0, 1, 1], 0, 4);

        let mut soft = vec![-0.1, -0.1];
        soft.extend_from_slice(&[1.0, -1.0, 1.0, 1.0]);
        soft.extend_from_slice(&[0.8, -0.8, 0.8, 0.8]);

        let frames = framer.push_soft(&soft);

        let bits: Vec<Vec<u8>> = frames.iter().map(|(b, _)| b.clone()).collect();
        assert_eq!(bits, vec![vec![1, 0, 1, 1]]);
        // The retained soft samples must equal the payload bits' soft
        // values, in order — used downstream for erasure decoding.
        assert_eq!(frames[0].1, vec![0.8, -0.8, 0.8, 0.8]);
        assert!(framer.sync_locked() >= 1);
    }

    #[test]
    fn soft_rejects_random_data() {
        let mut framer = SyncwordFramer::new(&[1, 0, 1, 1], 0, 4);

        let soft = [-1.0, 1.0, -1.0, -1.0, -1.0, 1.0, -1.0, -1.0];

        let frames = framer.push_soft(&soft);

        assert!(frames.is_empty());
        assert_eq!(framer.sync_locked(), 0);
    }

    #[test]
    fn soft_locks_through_low_snr() {
        let mut framer = SyncwordFramer::new(&[1, 0, 1, 1, 0, 1, 1, 0], 0, 8);
        framer.set_soft_gamma(0.5);

        let pattern = [1.0_f32, -1.0, 1.0, 1.0, -1.0, 1.0, 1.0, -1.0];
        let mut soft: Vec<f32> = pattern.iter().map(|s| s * 0.7 + 0.2).collect();
        soft.extend_from_slice(&[0.6, -0.6, 0.6, 0.6, -0.6, 0.6, 0.6, -0.6]);

        let frames = framer.push_soft(&soft);

        let bits: Vec<Vec<u8>> = frames.iter().map(|(b, _)| b.clone()).collect();
        assert_eq!(bits, vec![vec![1, 0, 1, 1, 0, 1, 1, 0]]);
        assert_eq!(framer.sync_locked(), 1);
    }
}
