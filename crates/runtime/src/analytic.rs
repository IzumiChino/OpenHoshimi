//! Analytic signal source: converts mono real audio to IQ via Hilbert transform.
//!
//! Wraps any [`InputSource`] (WAV, OGG, soundcard) and produces [`IqSample`]
//! output suitable for the existing IQ demodulator chain. Used for SSB
//! receiver recordings where the signal is a real-valued passband waveform.

use openhoshimi_core::{InputSource, IoError, IqSample, IqSource};
use openhoshimi_dsp::HilbertTransform;

/// An [`IqSource`] adapter that applies a Hilbert transform to a mono
/// [`InputSource`], producing analytic (complex) samples.
pub struct AnalyticSource {
    inner: Box<dyn InputSource>,
    hilbert: HilbertTransform,
    description: String,
}

impl AnalyticSource {
    /// Wrap an existing mono audio source with a Hilbert transform.
    pub fn new(inner: Box<dyn InputSource>) -> Self {
        let description = format!("Analytic({})", inner.description());
        Self {
            inner,
            hilbert: HilbertTransform::new(),
            description,
        }
    }

    /// Wrap with a custom tap count for the Hilbert FIR filter.
    pub fn with_taps(inner: Box<dyn InputSource>, num_taps: usize) -> Self {
        let description = format!("Analytic({})", inner.description());
        Self {
            inner,
            hilbert: HilbertTransform::with_taps(num_taps),
            description,
        }
    }
}

impl IqSource for AnalyticSource {
    fn read_samples(&mut self, buf: &mut [IqSample]) -> Result<usize, IoError> {
        // Read real samples into a temporary buffer
        let mut real_buf = vec![0.0f32; buf.len()];
        let read = self.inner.read_samples(&mut real_buf)?;

        // Convert each real sample to IQ via Hilbert transform
        for (i, &sample) in real_buf[..read].iter().enumerate() {
            buf[i] = self.hilbert.push(sample);
        }

        Ok(read)
    }

    fn sample_rate(&self) -> u32 {
        self.inner.sample_rate()
    }

    fn description(&self) -> &str {
        &self.description
    }

    fn total_samples(&self) -> Option<u64> {
        self.inner.total_samples()
    }
}
