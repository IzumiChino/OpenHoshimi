//! AO-40 FEC distributed-sync framer.
//!
//! AO-40 FEC frames are 5200 transmitted bits arranged as an 80 x 65
//! channel block. Each channel row starts with one bit of the 65-bit sync
//! vector. The remaining 79 bits per row carry the interleaved convolutional
//! symbols.

use std::collections::VecDeque;
use std::time::SystemTime;

use openhoshimi_core::{Frame, FrameType, Framing};

const ROWS: usize = 65;
const COLS: usize = 80;
const FRAME_BITS: usize = ROWS * COLS;
const DATA_COLS: usize = COLS - 1;
const SYNCWORD: &str = "11111110000111011110010110010010000001000100110001011101011011000";

/// Framer for AO-40 FEC distributed-sync bit streams.
#[derive(Debug, Clone)]
pub struct Ao40Framer {
    threshold: usize,
    window: VecDeque<u8>,
}

impl Ao40Framer {
    /// Create a framer with the maximum allowed sync vector bit errors.
    pub fn new(threshold: usize) -> Self {
        Self {
            threshold,
            window: VecDeque::with_capacity(FRAME_BITS),
        }
    }

    fn push_bit(&mut self, bit: u8) -> Option<Vec<u8>> {
        if self.window.len() == FRAME_BITS {
            self.window.pop_front();
        }
        self.window.push_back(bit & 1);

        if self.window.len() == FRAME_BITS && self.sync_distance() <= self.threshold {
            let frame = deinterleave_channel_bits(&self.window);
            self.window.clear();
            Some(frame)
        } else {
            None
        }
    }

    fn sync_distance(&self) -> usize {
        syncword_bits()
            .iter()
            .enumerate()
            .filter(|(row, expected)| self.window[*row * COLS] != **expected)
            .count()
    }
}

fn deinterleave_channel_bits(window: &VecDeque<u8>) -> Vec<u8> {
    let mut out = Vec::with_capacity(ROWS * DATA_COLS);
    for col in 1..COLS {
        for row in 0..ROWS {
            out.push(window[row * COLS + col]);
        }
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
        let mut bits = vec![0, 1, 0, 1];
        bits.extend_from_slice(&channel_frame());

        let frames = framer.push_bytes(&bits);

        assert_eq!(frames.len(), 1);
        assert_eq!(frames[0].raw.len(), ROWS * DATA_COLS);
        assert_eq!(frames[0].frame_type, FrameType::Ao40Fec);
        assert_eq!(&frames[0].raw[..8], &[0, 1, 0, 1, 0, 1, 0, 1]);
    }

    #[test]
    fn threshold_allows_sync_errors() {
        let mut framer = Ao40Framer::new(1);
        let mut frame = channel_frame();
        frame[2 * COLS] ^= 1;

        let frames = framer.push_bytes(&frame);

        assert_eq!(frames.len(), 1);
    }
}
