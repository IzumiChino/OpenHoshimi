//! CC11xx-style PN9 byte-level whitener.
//!
//! Used by Geoscan-family satellites whose TI CC1125 transceiver applies
//! hardware data whitening before transmission. The randomiser is a 9-bit
//! Galois LFSR with polynomial `x^9 + x^5 + 1` and initial state `0x1FF`.
//! The same routine descrambles received frames because XOR is symmetric.
//!
//! Bit ordering matches the CC1101 datasheet and gr-satellites'
//! `pn9_scrambler` hier block (`additive_scrambler_bb` with mask `0x21`,
//! seed `0x1FF`, length `8`, `bits_per_byte = 8`): the first LFSR output
//! bit XORs with bit 0 (LSB) of data byte 0, the eighth LFSR output bit
//! XORs with bit 7 (MSB) of data byte 0, and the next eight bits cover
//! data byte 1 the same way. The first eight whitening bytes from
//! `seed = 0x1FF` are `FF E1 1D 9A ED 85 33 24`.
//!
//! Lives in `codec` (not `dsp`) because the workspace dependency policy
//! forbids `codec` from depending on `dsp`, and PN9 is the inner
//! transform of [`crate::geoscan::GeoscanDecoder`].

/// Stateful CC11xx PN9 whitener / descrambler.
///
/// Construct with [`Pn9Whitener::new`] (resets to `0x1FF`) and call
/// [`Pn9Whitener::descramble`] once per frame. The instance is reusable —
/// calling `descramble` advances the LFSR, so consecutive calls treat
/// their inputs as one continuous stream. To restart the sequence between
/// frames, drop the old instance and create a new one.
pub struct Pn9Whitener {
    state: u16,
}

impl Pn9Whitener {
    /// Create a new whitener with the LFSR initialised to `0x1FF`, the
    /// CC11xx hardware default.
    pub fn new() -> Self {
        Self { state: 0x1FF }
    }

    /// XOR each byte of `data` with the next eight whitening bits from
    /// the LFSR. Operates in place.
    pub fn descramble(&mut self, data: &mut [u8]) {
        for byte in data.iter_mut() {
            let mut whitening = 0u8;
            for bit in 0..8 {
                whitening |= ((self.state & 1) as u8) << bit;
                let feedback = (self.state & 1) ^ ((self.state >> 5) & 1);
                self.state = (self.state >> 1) | (feedback << 8);
            }
            *byte ^= whitening;
        }
    }
}

impl Default for Pn9Whitener {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// First eight whitening bytes from `seed = 0x1FF`. Matches the
    /// CC1101 datasheet and a hand-simulated reference.
    const REFERENCE_BYTES: [u8; 8] = [0xFF, 0xE1, 0x1D, 0x9A, 0xED, 0x85, 0x33, 0x24];

    #[test]
    fn matches_reference_sequence() {
        let mut zeros = [0u8; 8];
        Pn9Whitener::new().descramble(&mut zeros);
        assert_eq!(zeros, REFERENCE_BYTES);
    }

    #[test]
    fn descramble_is_involution() {
        let mut payload = [0xAA, 0x55, 0x12, 0x34, 0xDE, 0xAD, 0xBE, 0xEF];
        let original = payload;
        Pn9Whitener::new().descramble(&mut payload);
        assert_ne!(payload, original);
        Pn9Whitener::new().descramble(&mut payload);
        assert_eq!(payload, original);
    }

    #[test]
    fn streaming_matches_one_shot() {
        let mut single = [0u8; 16];
        Pn9Whitener::new().descramble(&mut single);

        let mut split = [0u8; 16];
        let (head, tail) = split.split_at_mut(7);
        let mut whitener = Pn9Whitener::new();
        whitener.descramble(head);
        whitener.descramble(tail);

        assert_eq!(single, split);
    }
}
