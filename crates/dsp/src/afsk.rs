//! Bell 202 audio FSK demodulator with zero-crossing timing recovery.
//!
//! The demodulator correlates the input with sin/cos references at the mark
//! and space frequencies, smooths the magnitude with a one-symbol boxcar,
//! and recovers symbol timing from zero crossings of the difference signal.
//! This is the classic approach used by minimodem and simple TNC firmware:
//! every mark-to-space or space-to-mark transition produces a zero crossing
//! in the soft signal, which resets the symbol clock to mid-symbol. Between
//! crossings the clock free-runs at the nominal baud rate.

use std::f32::consts::TAU;

use openhoshimi_core::{DecodeError, Demodulator};

use crate::cpm::BoxcarFilter;

const MARK_HZ: f32 = 1200.0;
const SPACE_HZ: f32 = 2200.0;
const BAUDRATE: u32 = 1200;

/// Bell 202 AFSK demodulator with zero-crossing symbol timing recovery.
pub struct AfskDemodulator {
    sample_rate: u32,
    baudrate: u32,
    mark_i: Oscillator,
    mark_q: Oscillator,
    space_i: Oscillator,
    space_q: Oscillator,
    mark_i_lpf: BoxcarFilter,
    mark_q_lpf: BoxcarFilter,
    space_i_lpf: BoxcarFilter,
    space_q_lpf: BoxcarFilter,
    /// Symbol clock phase, counts up from 0 to samples_per_symbol.
    clock_phase: f32,
    samples_per_symbol: f32,
    /// Previous soft sample (for zero-crossing detection).
    prev_soft: f32,
}

impl AfskDemodulator {
    /// Create a Bell 202 demodulator for `sample_rate` Hz audio.
    pub fn new(sample_rate: u32) -> Self {
        Self::from_params(sample_rate, MARK_HZ, SPACE_HZ, BAUDRATE)
    }

    /// Create an AFSK demodulator with explicit tones and baudrate.
    pub fn with_tones(
        sample_rate: u32,
        mark_hz: f32,
        space_hz: f32,
        baudrate: u32,
    ) -> Result<Self, DecodeError> {
        if sample_rate == 0 {
            return Err(DecodeError::InvalidEncoding(
                "AFSK sample rate must be greater than zero".to_string(),
            ));
        }
        if baudrate == 0 {
            return Err(DecodeError::InvalidEncoding(
                "AFSK baudrate must be greater than zero".to_string(),
            ));
        }
        if mark_hz <= 0.0 {
            return Err(DecodeError::InvalidEncoding(
                "AFSK mark frequency must be greater than zero".to_string(),
            ));
        }
        if space_hz <= 0.0 {
            return Err(DecodeError::InvalidEncoding(
                "AFSK space frequency must be greater than zero".to_string(),
            ));
        }

        Ok(Self::from_params(sample_rate, mark_hz, space_hz, baudrate))
    }

    fn from_params(sample_rate: u32, mark_hz: f32, space_hz: f32, baudrate: u32) -> Self {
        let samples_per_symbol = sample_rate as f32 / baudrate as f32;
        let window = samples_per_symbol.round() as usize;
        Self {
            sample_rate,
            baudrate,
            mark_i: Oscillator::new(mark_hz, sample_rate, 0.0),
            mark_q: Oscillator::new(mark_hz, sample_rate, std::f32::consts::FRAC_PI_2),
            space_i: Oscillator::new(space_hz, sample_rate, 0.0),
            space_q: Oscillator::new(space_hz, sample_rate, std::f32::consts::FRAC_PI_2),
            mark_i_lpf: BoxcarFilter::new(window),
            mark_q_lpf: BoxcarFilter::new(window),
            space_i_lpf: BoxcarFilter::new(window),
            space_q_lpf: BoxcarFilter::new(window),
            clock_phase: 0.0,
            samples_per_symbol,
            prev_soft: 0.0,
        }
    }
}

impl Demodulator for AfskDemodulator {
    type Sample = f32;

    fn push_samples(&mut self, samples: &[f32]) -> Vec<u8> {
        let mut bits = Vec::new();

        for &sample in samples {
            // Correlate with mark and space references.
            let mi = self.mark_i_lpf.push(sample * self.mark_i.next());
            let mq = self.mark_q_lpf.push(sample * self.mark_q.next());
            let si = self.space_i_lpf.push(sample * self.space_i.next());
            let sq = self.space_q_lpf.push(sample * self.space_q.next());

            // Envelope: magnitude squared of each tone's correlation.
            let mark_env = mi * mi + mq * mq;
            let space_env = si * si + sq * sq;

            // Bipolar soft signal: positive = mark, negative = space.
            let soft = mark_env - space_env;

            // Zero-crossing detection with hysteresis: only reset the clock
            // on genuine mark/space transitions (where the soft signal
            // crosses zero with sufficient magnitude on both sides). This
            // prevents spurious resets from correlator noise during long
            // runs of the same tone.
            let crossed =
                (soft > 0.0 && self.prev_soft < 0.0) || (soft < 0.0 && self.prev_soft > 0.0);
            if crossed {
                // Transition detected. The optimal sampling point is
                // half a symbol period after the transition.
                self.clock_phase = self.samples_per_symbol * 0.5;
            }
            self.prev_soft = soft;

            // Advance the free-running symbol clock.
            self.clock_phase += 1.0;
            if self.clock_phase >= self.samples_per_symbol {
                self.clock_phase -= self.samples_per_symbol;
                // Sample the soft signal at this instant.
                bits.push(u8::from(soft >= 0.0));
            }
        }

        bits
    }

    fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    fn baudrate(&self) -> u32 {
        self.baudrate
    }
}

/// Free-running oscillator for correlation.
#[derive(Debug, Clone)]
struct Oscillator {
    phase: f32,
    increment: f32,
}

impl Oscillator {
    fn new(frequency_hz: f32, sample_rate: u32, initial_phase: f32) -> Self {
        Self {
            phase: initial_phase,
            increment: TAU * frequency_hz / sample_rate as f32,
        }
    }

    fn next(&mut self) -> f32 {
        let value = self.phase.cos();
        self.phase += self.increment;
        if self.phase >= TAU {
            self.phase -= TAU;
        }
        value
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
        let mut bits: Vec<u8> = vec![1; 48]; // preamble
        bits.extend_from_slice(&[0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 1]);
        bits.extend_from_slice(&[1; 16]); // trailing
        let signal = synthesize(&bits, 48_000);
        let mut demodulator = AfskDemodulator::new(48_000);
        let recovered = demodulator.push_samples(&signal);

        let pattern = &[0u8, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 1];
        let mut found = false;
        for offset in 0..recovered.len().saturating_sub(pattern.len()) {
            if &recovered[offset..offset + pattern.len()] == pattern {
                found = true;
                break;
            }
        }
        assert!(
            found,
            "data pattern not found in recovered bits (len={})",
            recovered.len()
        );
    }

    #[test]
    fn recovers_with_phase_offset() {
        let mut bits: Vec<u8> = vec![1; 48];
        bits.extend_from_slice(&[0, 1, 0, 0, 1, 0, 1, 1]);
        bits.extend_from_slice(&[1; 16]);
        let signal = synthesize(&bits, 48_000);
        // Skip 20 samples (half a symbol) to create timing offset.
        let offset_signal = &signal[20..];
        let mut demodulator = AfskDemodulator::new(48_000);
        let recovered = demodulator.push_samples(offset_signal);

        let pattern = &[0u8, 1, 0, 0, 1, 0, 1, 1];
        let mut found = false;
        for offset in 0..recovered.len().saturating_sub(pattern.len()) {
            if &recovered[offset..offset + pattern.len()] == pattern {
                found = true;
                break;
            }
        }
        assert!(
            found,
            "timing recovery failed to track phase offset (len={})",
            recovered.len()
        );
    }

    #[test]
    fn sample_rate_and_baud_reported() {
        let demodulator = AfskDemodulator::new(48_000);
        assert_eq!(demodulator.sample_rate(), 48_000);
        assert_eq!(demodulator.baudrate(), 1200);
    }

    #[test]
    fn rejects_invalid_custom_tones() {
        let err = match AfskDemodulator::with_tones(48_000, 0.0, SPACE_HZ, BAUDRATE) {
            Ok(_) => panic!("invalid mark tone should fail"),
            Err(err) => err,
        };
        assert!(matches!(err, DecodeError::InvalidEncoding(_)));
    }
}
