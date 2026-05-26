//! FIR Hilbert transform for real-to-analytic signal conversion.
//!
//! Converts a real-valued audio stream into a complex analytic signal
//! (IQ pair) by generating the quadrature (90° phase-shifted) component
//! via a windowed-sinc FIR filter. The in-phase path is a matching delay
//! line that compensates for the filter's group delay.
//!
//! Primary use case: processing SSB receiver recordings through the
//! existing IQ demodulator chain.

use openhoshimi_core::IqSample;

/// Default number of FIR taps (must be odd).
const DEFAULT_NUM_TAPS: usize = 65;

/// FIR Hilbert transformer that converts real samples to analytic IQ.
#[derive(Debug, Clone)]
pub struct HilbertTransform {
    taps: Vec<f32>,
    history: Vec<f32>,
    delay_line: Vec<f32>,
    head: usize,
    delay_head: usize,
    group_delay: usize,
}

impl HilbertTransform {
    /// Create a Hilbert transform with the default tap count (65).
    pub fn new() -> Self {
        Self::with_taps(DEFAULT_NUM_TAPS)
    }

    /// Create a Hilbert transform with a specified odd tap count.
    ///
    /// Panics if `num_taps` is even or less than 3.
    pub fn with_taps(num_taps: usize) -> Self {
        assert!(
            num_taps >= 3 && num_taps % 2 == 1,
            "num_taps must be odd and >= 3"
        );
        let taps = hilbert_taps(num_taps);
        let group_delay = num_taps / 2;
        Self {
            history: vec![0.0; taps.len()],
            delay_line: vec![0.0; group_delay + 1],
            taps,
            head: 0,
            delay_head: 0,
            group_delay,
        }
    }

    /// Push one real sample and produce one analytic IQ sample.
    ///
    /// The I component is the delayed input (aligned with the filter's
    /// group delay). The Q component is the Hilbert-filtered output.
    pub fn push(&mut self, sample: f32) -> IqSample {
        // Compute Q via FIR convolution
        self.history[self.head] = sample;
        let len = self.history.len();
        let mut q = 0.0f32;
        for (offset, &tap) in self.taps.iter().enumerate() {
            let index = (self.head + len - offset) % len;
            q += self.history[index] * tap;
        }
        self.head = (self.head + 1) % len;

        // Compute I via delay line (group_delay samples behind)
        let delay_len = self.delay_line.len();
        self.delay_line[self.delay_head] = sample;
        let delayed_index = (self.delay_head + 1) % delay_len;
        let i = self.delay_line[delayed_index];
        self.delay_head = (self.delay_head + 1) % delay_len;

        IqSample { i, q }
    }

    /// Return the group delay in samples.
    pub fn group_delay(&self) -> usize {
        self.group_delay
    }
}

impl Default for HilbertTransform {
    fn default() -> Self {
        Self::new()
    }
}

/// Generate Hilbert FIR taps using a windowed ideal impulse response.
///
/// The ideal Hilbert transformer has impulse response:
///   h[n] = 2/(n*pi) * sin^2(n*pi/2) for n != 0, 0 for n == 0
///
/// We apply a Blackman window for good sidelobe suppression.
fn hilbert_taps(num_taps: usize) -> Vec<f32> {
    let half = (num_taps / 2) as i32;
    let mut taps = Vec::with_capacity(num_taps);
    for k in 0..num_taps {
        let n = k as i32 - half;
        let h = if n % 2 == 0 {
            // n == 0 and all even taps are zero in the ideal Hilbert
            // impulse response.
            0.0
        } else {
            2.0 / (n as f32 * std::f32::consts::PI)
        };
        // Blackman window
        let w = blackman(k, num_taps);
        taps.push(h * w);
    }
    taps
}

fn blackman(n: usize, len: usize) -> f32 {
    let x = std::f32::consts::TAU * n as f32 / (len - 1) as f32;
    0.42 - 0.5 * x.cos() + 0.08 * (2.0 * x).cos()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pure_tone_quadrature_shift() {
        // A cosine input should produce a sine output (90° shift) after
        // the transient settles. Use a frequency well within the passband
        // (away from DC and Nyquist).
        let mut ht = HilbertTransform::new();
        let freq = 4000.0f32;
        let sample_rate = 48_000.0f32;
        let num_samples = 1000;
        let warmup = ht.group_delay() + 100;

        let mut max_error = 0.0f32;
        for n in 0..num_samples {
            let t = n as f32 / sample_rate;
            let input = (std::f32::consts::TAU * freq * t).cos();
            let iq = ht.push(input);

            if n >= warmup {
                // After warmup, I should be cos and Q should be sin
                // (delayed by group_delay samples)
                let t_delayed = (n - ht.group_delay()) as f32 / sample_rate;
                let expected_i = (std::f32::consts::TAU * freq * t_delayed).cos();
                let expected_q = (std::f32::consts::TAU * freq * t_delayed).sin();
                let err_i = (iq.i - expected_i).abs();
                let err_q = (iq.q - expected_q).abs();
                let err = err_i.max(err_q);
                if err > max_error {
                    max_error = err;
                }
            }
        }
        assert!(
            max_error < 0.05,
            "Hilbert transform error too large: {max_error}"
        );
    }

    #[test]
    fn dc_is_rejected() {
        let mut ht = HilbertTransform::new();
        // Feed DC (constant 1.0) — Q output should be ~0 after settling
        for _ in 0..200 {
            ht.push(1.0);
        }
        let iq = ht.push(1.0);
        assert!(
            iq.q.abs() < 0.01,
            "DC should produce near-zero Q, got {}",
            iq.q
        );
    }
}
