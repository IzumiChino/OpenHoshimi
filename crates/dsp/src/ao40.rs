//! AO-40 FEC distributed-sync framer.
//!
//! AO-40 FEC frames are 5200 transmitted bits arranged as a 65 x 80
//! channel block. Each row starts with one bit of the 65-bit sync vector,
//! and the remaining 79 bits per row carry the interleaved convolutional
//! symbols.

use std::collections::VecDeque;
use std::time::SystemTime;

use openhoshimi_core::{Frame, FrameType, Framing};

const ROWS: usize = 65;
const COLS: usize = 80;
const FRAME_BITS: usize = ROWS * COLS;
const SYNCWORD: &str = "11111110000111011110010110010010000001000100110001011101011011000";
const OUTPUT_BITS: usize = 5132;
const HISTORY_EPOCH_BITS: u64 = 1200;

/// Framer for AO-40 FEC distributed-sync bit streams.
#[derive(Debug, Clone)]
pub struct Ao40Framer {
    threshold: usize,
    window: VecDeque<u8>,
    soft_window: VecDeque<i8>,
    best_distance: Option<usize>,
    bit_count: u64,
    epoch_min: Option<usize>,
    distance_history: Vec<Option<usize>>,
}

impl Ao40Framer {
    /// Create a framer with the maximum allowed sync vector bit errors.
    pub fn new(threshold: usize) -> Self {
        Self {
            threshold,
            window: VecDeque::with_capacity(FRAME_BITS),
            soft_window: VecDeque::with_capacity(FRAME_BITS),
            best_distance: None,
            bit_count: 0,
            epoch_min: None,
            distance_history: Vec::new(),
        }
    }

    /// Return the closest sync distance seen since the framer was created.
    pub fn best_sync_distance(&self) -> Option<usize> {
        self.best_distance
    }

    /// Per-epoch minimum sync distance recorded since the framer was created.
    ///
    /// Each entry is the minimum sync distance observed across one
    /// `HISTORY_EPOCH_BITS` window of input bits (1200 bits = 1.0 s at
    /// the 1200 bd telemetry rate). `None` means no distance was
    /// computed in that epoch (the soft window is still filling after a
    /// reset). Only the soft-input path records history; the hard
    /// `push_bit` path is left untracked because the soft pipeline is
    /// what AO-40 FEC downlinks use in practice.
    pub fn distance_history(&self) -> &[Option<usize>] {
        &self.distance_history
    }

    fn push_bit(&mut self, bit: u8) -> Option<Vec<u8>> {
        if self.window.len() == FRAME_BITS {
            self.window.pop_front();
        }
        self.window.push_back(bit & 1);

        if self.window.len() != FRAME_BITS {
            return None;
        }

        let direct = self.sync_distance(false);
        let inverted = self.sync_distance(true);
        let (distance, invert) = if inverted < direct {
            (inverted, true)
        } else {
            (direct, false)
        };
        self.best_distance = Some(
            self.best_distance
                .map_or(distance, |best| best.min(distance)),
        );
        if distance > self.threshold {
            return None;
        }

        let frame = deinterleave_channel_bits(&self.window, invert);
        self.window.clear();
        Some(frame)
    }

    /// Push a block of soft channel symbols; each call returns one frame per
    /// detected syncword. See [`push_soft_bit`](Self::push_soft_bit).
    pub fn push_soft_bytes(&mut self, soft: &[i8]) -> Vec<Vec<i8>> {
        let mut frames = Vec::new();
        for &symbol in soft {
            if let Some(frame) = self.push_soft_bit(symbol) {
                frames.push(frame);
            }
        }
        frames
    }

    /// Push one soft channel symbol; sign carries the bit decision and
    /// magnitude carries per-bit confidence. Returns the 5132 deinterleaved
    /// soft symbols when a syncword is detected.
    pub fn push_soft_bit(&mut self, soft: i8) -> Option<Vec<i8>> {
        self.bit_count += 1;
        let just_closed_epoch = self.bit_count % HISTORY_EPOCH_BITS == 0;

        if self.soft_window.len() == FRAME_BITS {
            self.soft_window.pop_front();
        }
        self.soft_window.push_back(soft);

        if self.soft_window.len() != FRAME_BITS {
            if just_closed_epoch {
                self.distance_history.push(self.epoch_min.take());
            }
            return None;
        }

        let direct = self.soft_sync_distance(false);
        let inverted = self.soft_sync_distance(true);
        let (distance, invert) = if inverted < direct {
            (inverted, true)
        } else {
            (direct, false)
        };
        self.best_distance = Some(
            self.best_distance
                .map_or(distance, |best| best.min(distance)),
        );
        self.epoch_min = Some(self.epoch_min.map_or(distance, |best| best.min(distance)));
        if just_closed_epoch {
            self.distance_history.push(self.epoch_min.take());
        }
        if distance > self.threshold {
            return None;
        }

        let frame = deinterleave_soft_bits(&self.soft_window, invert);
        self.soft_window.clear();
        Some(frame)
    }

    fn sync_distance(&self, invert: bool) -> usize {
        let xor = u8::from(invert);
        syncword_bits()
            .iter()
            .enumerate()
            .filter(|(row, expected)| self.window[*row * COLS] != **expected ^ xor)
            .count()
    }

    fn soft_sync_distance(&self, invert: bool) -> usize {
        let xor = u8::from(invert);
        syncword_bits()
            .iter()
            .enumerate()
            .filter(|(row, expected)| {
                let received_bit = u8::from(self.soft_window[*row * COLS] < 0);
                received_bit != **expected ^ xor
            })
            .count()
    }
}

fn deinterleave_channel_bits(window: &VecDeque<u8>, invert: bool) -> Vec<u8> {
    let xor = u8::from(invert);
    let mut out = Vec::with_capacity(OUTPUT_BITS);
    for index in ROWS..(ROWS + OUTPUT_BITS) {
        out.push(window[(index % ROWS) * COLS + index / ROWS] ^ xor);
    }
    out
}

fn deinterleave_soft_bits(window: &VecDeque<i8>, invert: bool) -> Vec<i8> {
    let mut out = Vec::with_capacity(OUTPUT_BITS);
    for index in ROWS..(ROWS + OUTPUT_BITS) {
        let raw = window[(index % ROWS) * COLS + index / ROWS];
        // Negation of i8::MIN would overflow; clamp first so an inverted
        // polarity frame remains representable in i8.
        let value = if invert { raw.saturating_neg() } else { raw };
        out.push(value);
    }
    out
}

impl Framing for Ao40Framer {
    fn push_bytes(&mut self, bytes: &[u8]) -> Vec<Frame> {
        let mut frames = Vec::new();
        for &byte in bytes {
            if let Some(raw) = self.push_bit(byte) {
                frames.push(Frame {
                    satellite_id: 0,
                    timestamp: SystemTime::now(),
                    rssi_dbm: None,
                    raw,
                    frame_type: FrameType::Ao40Fec,
                    soft_bits: None,
                });
            }
        }
        frames
    }
}

impl Default for Ao40Framer {
    fn default() -> Self {
        Self::new(0)
    }
}

fn syncword_bits() -> Vec<u8> {
    SYNCWORD
        .bytes()
        .map(|byte| u8::from(byte == b'1'))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use openhoshimi_core::Framing;

    fn channel_frame() -> Vec<u8> {
        let mut frame = vec![0u8; FRAME_BITS];
        for (row, bit) in syncword_bits().iter().enumerate() {
            frame[row * COLS] = *bit;
        }
        let mut data_bit = 0u8;
        for col in 1..COLS {
            for row in 0..ROWS {
                frame[row * COLS + col] = data_bit;
                data_bit ^= 1;
            }
        }
        frame
    }

    #[test]
    fn finds_distributed_sync_frame() {
        let mut framer = Ao40Framer::new(0);
        let bits = channel_frame();

        let frames = framer.push_bytes(&bits);

        assert_eq!(frames.len(), 1);
        assert_eq!(frames[0].raw.len(), OUTPUT_BITS);
        assert_eq!(frames[0].frame_type, FrameType::Ao40Fec);
        let expected: Vec<u8> = (ROWS..(ROWS + OUTPUT_BITS))
            .map(|index| bits[(index % ROWS) * COLS + index / ROWS])
            .collect();
        assert_eq!(frames[0].raw, expected);
    }

    #[test]
    fn threshold_allows_sync_errors() {
        let mut framer = Ao40Framer::new(1);
        let mut frame = channel_frame();
        frame[2 * COLS] ^= 1;

        let frames = framer.push_bytes(&frame);

        assert_eq!(frames.len(), 1);
    }

    #[test]
    fn detects_inverted_polarity_frame() {
        let mut framer = Ao40Framer::new(0);
        let mut frame = channel_frame();
        for bit in &mut frame {
            *bit ^= 1;
        }

        let frames = framer.push_bytes(&frame);

        assert_eq!(frames.len(), 1);
        let expected: Vec<u8> = (ROWS..(ROWS + OUTPUT_BITS))
            .map(|index| channel_frame()[(index % ROWS) * COLS + index / ROWS])
            .collect();
        assert_eq!(frames[0].raw, expected);
    }
}
