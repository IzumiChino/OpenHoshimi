//! G3RUH bit-stream processing.
//!
//! The G3RUH 9600 baud packet-radio modem uses NRZI line coding and a
//! self-synchronising scrambler with taps at 17 and 12 bits. This module
//! operates on one-bit-per-byte streams (`0x00` or `0x01`) and provides the
//! bit-level pieces needed before HDLC framing.

use openhoshimi_core::{Demodulator, Descrambler, LineDecoder};

const SCRAMBLER_MASK: u32 = (1 << 17) - 1;
const SCRAMBLER_TAP_17: u32 = 1 << 16;
const SCRAMBLER_TAP_12: u32 = 1 << 11;

/// NRZI decoder for one-bit-per-byte baseband streams.
///
/// Output bits are `1` when the input level is unchanged and `0` when the
/// input level toggles. This matches the convention used by AX.25 packet
/// streams before HDLC decoding.
#[derive(Debug, Clone)]
pub struct NrziDecoder {
    previous: Option<u8>,
}

impl NrziDecoder {
    /// Create a new NRZI decoder.
    pub fn new() -> Self {
        Self { previous: None }
    }

    /// Decode one-bit-per-byte NRZI symbols.
    pub fn push_bits(&mut self, bits: &[u8]) -> Vec<u8> {
        let mut out = bits.to_vec();
        self.decode(&mut out);
        out
    }
}

impl LineDecoder for NrziDecoder {
    fn decode(&mut self, bits: &mut [u8]) {
        for bit in bits {
            let current = *bit & 1;
            let decoded = match self.previous {
                Some(previous) => u8::from(current == previous),
                None => 1,
            };
            self.previous = Some(current);
            *bit = decoded;
        }
    }
}

impl Default for NrziDecoder {
    fn default() -> Self {
        Self::new()
    }
}

/// G3RUH self-synchronising descrambler.
///
/// The descrambler computes `out[n] = in[n] xor in[n-12] xor in[n-17]`.
/// It does not need a reset sequence, but the first 17 output bits depend
/// on the initial shift-register state and should be treated as settling
/// bits by callers that need exact preamble handling.
#[derive(Debug, Clone)]
pub struct G3ruhDescrambler {
    shift_register: u32,
}

impl G3ruhDescrambler {
    /// Create a descrambler with an all-zero history.
    pub fn new() -> Self {
        Self { shift_register: 0 }
    }

    /// Descramble one-bit-per-byte input into one-bit-per-byte output.
    pub fn push_bits(&mut self, bits: &[u8]) -> Vec<u8> {
        let mut out = bits.to_vec();
        self.descramble(&mut out);
        out
    }
}

impl Descrambler for G3ruhDescrambler {
    fn descramble(&mut self, data: &mut [u8]) {
        for bit in data {
            let input = *bit & 1;
            let tap_17 = u8::from(self.shift_register & SCRAMBLER_TAP_17 != 0);
            let tap_12 = u8::from(self.shift_register & SCRAMBLER_TAP_12 != 0);
            let decoded = input ^ tap_17 ^ tap_12;
            self.shift_register = ((self.shift_register << 1) | u32::from(input)) & SCRAMBLER_MASK;
            *bit = decoded;
        }
    }
}

impl Default for G3ruhDescrambler {
    fn default() -> Self {
        Self::new()
    }
}

/// Stateful G3RUH processor for already-sliced input bits.
///
/// This type implements [`Demodulator`] so it can be dropped into the same
/// pipeline shape as audio demodulators. The samples are interpreted as
/// hard decisions: non-negative values are `1`, negative values are `0`.
#[derive(Debug, Clone)]
pub struct G3ruhDemodulator {
    sample_rate: u32,
    baudrate: u32,
    nrzi: NrziDecoder,
    descrambler: G3ruhDescrambler,
}

impl G3ruhDemodulator {
    /// Create a G3RUH hard-decision bit processor.
    pub fn new(sample_rate: u32, baudrate: u32) -> Self {
        Self {
            sample_rate,
            baudrate,
            nrzi: NrziDecoder::new(),
            descrambler: G3ruhDescrambler::new(),
        }
    }

    /// Process already-sliced one-bit-per-byte G3RUH symbols.
    pub fn push_bits(&mut self, bits: &[u8]) -> Vec<u8> {
        let nrzi = self.nrzi.push_bits(bits);
        self.descrambler.push_bits(&nrzi)
    }
}

impl Demodulator for G3ruhDemodulator {
    type Sample = f32;

    fn push_samples(&mut self, samples: &[f32]) -> Vec<u8> {
        let bits: Vec<u8> = samples
            .iter()
            .map(|sample| u8::from(*sample >= 0.0))
            .collect();
        self.push_bits(&bits)
    }

    fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    fn baudrate(&self) -> u32 {
        self.baudrate
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scramble(bits: &[u8]) -> Vec<u8> {
        let mut history = 0u32;
        let mut out = Vec::with_capacity(bits.len());
        for &bit in bits {
            let input = bit & 1;
            let tap_17 = u8::from(history & SCRAMBLER_TAP_17 != 0);
            let tap_12 = u8::from(history & SCRAMBLER_TAP_12 != 0);
            let encoded = input ^ tap_17 ^ tap_12;
            history = ((history << 1) | u32::from(encoded)) & SCRAMBLER_MASK;
            out.push(encoded);
        }
        out
    }

    fn nrzi_encode(bits: &[u8]) -> Vec<u8> {
        let mut level = 1u8;
        let mut out = Vec::with_capacity(bits.len());
        for &bit in bits {
            if bit & 1 == 0 {
                level ^= 1;
            }
            out.push(level);
        }
        out
    }

    #[test]
    fn descrambler_round_trips_scrambled_bits_after_history_settles() {
        let input: Vec<u8> = (0..96)
            .map(|i| u8::from(i % 7 == 0 || i % 11 == 0))
            .collect();
        let scrambled = scramble(&input);
        let mut descrambler = G3ruhDescrambler::new();
        let decoded = descrambler.push_bits(&scrambled);

        assert_eq!(&decoded[17..], &input[17..]);
    }

    #[test]
    fn nrzi_decoder_tracks_transitions() {
        let input = [1, 1, 0, 0, 1, 0, 1, 1];
        let encoded = nrzi_encode(&input);
        let mut decoder = NrziDecoder::new();
        let decoded = decoder.push_bits(&encoded);

        assert_eq!(decoded, input);
    }

    #[test]
    fn g3ruh_processor_decodes_sliced_symbols() {
        let input: Vec<u8> = (0..128).map(|i| u8::from(i % 5 < 2)).collect();
        let scrambled = scramble(&input);
        let encoded = nrzi_encode(&scrambled);
        let mut processor = G3ruhDemodulator::new(9_600, 9_600);
        let decoded = processor.push_bits(&encoded);

        assert_eq!(&decoded[17..], &input[17..]);
        assert_eq!(processor.sample_rate(), 9_600);
        assert_eq!(processor.baudrate(), 9_600);
    }
}
