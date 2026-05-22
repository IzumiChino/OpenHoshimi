//! Syncword-based bit framer.
//!
//! This framer searches a one-bit-per-byte stream for a fixed syncword and
//! emits a fixed number of following payload bits. It is useful for protocols
//! such as AX100 ASM mode and AO-40 FEC where HDLC flags are not used.

/// Finds fixed-size payloads after a bit syncword.
#[derive(Debug, Clone)]
pub struct SyncwordFramer {
    syncword: Vec<u8>,
    threshold: usize,
    payload_bits: usize,
    shift_register: Vec<u8>,
    collecting: bool,
    current_payload: Vec<u8>,
}

impl SyncwordFramer {
    /// Create a new syncword framer.
    pub fn new(syncword: &[u8], threshold: usize, payload_bits: usize) -> Self {
        Self {
            syncword: syncword.iter().map(|bit| bit & 1).collect(),
            threshold,
            payload_bits,
            shift_register: Vec::with_capacity(syncword.len()),
            collecting: false,
            current_payload: Vec::with_capacity(payload_bits),
        }
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

            if self.shift_register.len() == self.syncword.len()
                && hamming_distance(&self.shift_register, &self.syncword) <= self.threshold
            {
                self.collecting = true;
                self.current_payload.clear();
            }
        }

        frames
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
}
