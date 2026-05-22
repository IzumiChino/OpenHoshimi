//! CCSDS bit descrambler.
//!
//! The CCSDS randomizer uses the polynomial x^8 + x^7 + x^5 + x^3 + 1
//! with an all-ones initial state. Applying the same pseudo-random sequence
//! a second time descrambles the stream.

use openhoshimi_core::Descrambler;

const POLY_MASK: u8 = 0xa9;
const INITIAL_STATE: u8 = 0xff;

/// CCSDS randomizer descrambler for one-bit-per-byte streams.
#[derive(Debug, Clone)]
pub struct CcsdsDescrambler {
    state: u8,
}

impl CcsdsDescrambler {
    /// Create a CCSDS descrambler with the standard all-ones initial state.
    pub fn new() -> Self {
        Self {
            state: INITIAL_STATE,
        }
    }

    /// Descramble one-bit-per-byte input into one-bit-per-byte output.
    pub fn push_bits(&mut self, bits: &[u8]) -> Vec<u8> {
        let mut out = bits.to_vec();
        self.descramble(&mut out);
        out
    }
}

impl Default for CcsdsDescrambler {
    fn default() -> Self {
        Self::new()
    }
}

impl Descrambler for CcsdsDescrambler {
    fn descramble(&mut self, data: &mut [u8]) {
        for bit in data {
            let pn = u8::from(self.state & 0x80 != 0);
            let feedback = u8::from((self.state & POLY_MASK).count_ones() & 1 != 0);
            self.state = (self.state << 1) | feedback;
            *bit = (*bit & 1) ^ pn;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sequence_is_self_inverse() {
        let input: Vec<u8> = (0..128).map(|i| u8::from(i % 3 == 0)).collect();
        let mut scrambler = CcsdsDescrambler::new();
        let scrambled = scrambler.push_bits(&input);
        let mut descrambler = CcsdsDescrambler::new();
        let decoded = descrambler.push_bits(&scrambled);

        assert_ne!(scrambled, input);
        assert_eq!(decoded, input);
    }
}
