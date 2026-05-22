//! Bell 202 audio FSK demodulator.
//!
//! Bell 202 is the classic 1200 baud packet-radio audio modulation: mark
//! is a 1200 Hz tone, space is a 2200 Hz tone, and one symbol is emitted at
//! 1200 baud. This demodulator uses two Goertzel detectors over one symbol
//! period and emits one byte per recovered bit (`0x01` for mark, `0x00` for
//! space).

use std::f32::consts::TAU;

use openhoshimi_core::Demodulator;

const MARK_HZ: f32 = 1200.0;
const SPACE_HZ: f32 = 2200.0;
const BAUDRATE: u32 = 1200;

/// Bell 202 AFSK demodulator.
///
/// Construct with [`AfskDemodulator::new`] and pass audio through
/// [`Demodulator::push_samples`]. The demodulator is stateful and can be
/// fed arbitrary chunk sizes.
pub struct AfskDemodulator {
    sample_rate: u32,
    sample_phase: f32,
    samples_per_symbol: f32,
    mark: ToneDetector,
    space: ToneDetector,
}

impl AfskDemodulator {
    /// Create a Bell 202 demodulator for `sample_rate` Hz audio.
    pub fn new(sample_rate: u32) -> Self {
        let samples_per_symbol = sample_rate as f32 / BAUDRATE as f32;
        Self {
            sample_rate,
            sample_phase: 0.0,
            samples_per_symbol,
            mark: ToneDetector::new(MARK_HZ, sample_rate),
            space: ToneDetector::new(SPACE_HZ, sample_rate),
        }
    }
}

impl Demodulator for AfskDemodulator {
    type Sample = f32;

    fn push_samples(&mut self, samples: &[f32]) -> Vec<u8> {
        let mut bits = Vec::new();

        for &sample in samples {
            self.mark.push(sample);
            self.space.push(sample);

            self.sample_phase += 1.0;
            if self.sample_phase >= self.samples_per_symbol {
                self.sample_phase -= self.samples_per_symbol;
                let mark_energy = self.mark.energy();
                let space_energy = self.space.energy();
                bits.push(u8::from(mark_energy >= space_energy));
                self.mark.reset();
                self.space.reset();
            }
        }

        bits
    }

    fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    fn baudrate(&self) -> u32 {
        BAUDRATE
    }
}

#[derive(Debug, Clone)]
struct ToneDetector {
    coefficient: f32,
    q1: f32,
    q2: f32,
}

impl ToneDetector {
    fn new(frequency_hz: f32, sample_rate: u32) -> Self {
        let omega = TAU * frequency_hz / sample_rate as f32;
        Self {
            coefficient: 2.0 * omega.cos(),
            q1: 0.0,
            q2: 0.0,
        }
    }

    fn push(&mut self, sample: f32) {
        let q0 = sample + self.coefficient * self.q1 - self.q2;
        self.q2 = self.q1;
        self.q1 = q0;
    }

    fn energy(&self) -> f32 {
        self.q1.mul_add(
            self.q1,
            self.q2 * self.q2 - self.coefficient * self.q1 * self.q2,
        )
    }

    fn reset(&mut self) {
        self.q1 = 0.0;
        self.q2 = 0.0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn synthesize(bits: &[u8], sample_rate: u32) -> Vec<f32> {
        let samples_per_bit = sample_rate / BAUDRATE;
        let mut samples = Vec::with_capacity(bits.len() * samples_per_bit as usize);
        let mut phase = 0.0f32;

        for &bit in bits {
            let frequency = if bit == 0 { SPACE_HZ } else { MARK_HZ };
            let inc = TAU * frequency / sample_rate as f32;
            for _ in 0..samples_per_bit {
                samples.push(phase.sin());
                phase += inc;
                if phase >= TAU {
                    phase -= TAU;
                }
            }
        }

        samples
    }

    #[test]
    fn recovers_known_bit_pattern() {
        let bits: Vec<u8> = [
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 0,
        ]
        .to_vec();
        let signal = synthesize(&bits, 48_000);
        let mut demodulator = AfskDemodulator::new(48_000);
        let recovered = demodulator.push_samples(&signal);

        assert_eq!(recovered, bits);
    }

    #[test]
    fn sample_rate_and_baud_reported() {
        let demodulator = AfskDemodulator::new(48_000);
        assert_eq!(demodulator.sample_rate(), 48_000);
        assert_eq!(demodulator.baudrate(), 1200);
    }
}
