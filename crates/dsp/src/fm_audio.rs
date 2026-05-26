//! Symbol recovery from FM-demodulated audio.
//!
//! When a satellite signal is received through an FM receiver (hardware or
//! SDR in FM mode), the FM discriminator has already converted instantaneous
//! frequency deviation into a proportional audio amplitude. For FSK/GMSK
//! signals this audio directly represents the symbol waveform. This module
//! recovers bits from that waveform:
//!
//! 1. Optional DC-blocking first-order IIR.
//! 2. Optional Gaussian receive matched filter (for GMSK signals).
//! 3. Startup phase search that buffers a fixed number of symbols, scans
//!    all integer-sample phases, and picks the one with the largest eye
//!    opening (Σ|sample| at candidate sample instants).
//! 4. Symbol-rate sampling via integrate-and-dump (unfiltered) or
//!    sub-sample interpolation (filtered), seeded with the chosen phase.
//! 5. Hard slicer with optional differential decoding and inversion.

use openhoshimi_core::{DecodeError, Demodulator};

use crate::cpm::{GaussianFir, IntegrateAndDump, TrackingInterpolator};

/// Loop gain for the closed-loop Mueller-Müller timing tracker on the
/// filtered path. Tuned for the noisy real-world recordings produced by
/// SatNOGS-style passes: small enough to stay stable on weak signals,
/// large enough to follow the typical TX/RX clock drift (<200 PPM) seen
/// on amateur satellite passes.
const MM_TIMING_GAIN: f32 = 0.02;

/// Number of symbols buffered during each phase search.
const PHASE_SEARCH_SYMBOLS: usize = 64;

/// Leak-decay factor for the per-sample envelope tracker (`peak hold`).
const ENVELOPE_DECAY: f32 = 0.999;

/// Very slow decay for the peak-level tracker so it captures the strongest
/// burst seen in the recording and holds it across inter-burst gaps.
const PEAK_DECAY: f32 = 0.999_99;

/// Envelope must exceed this fraction of peak to be considered active.
const ACTIVITY_FRAC: f32 = 0.2;

/// Configuration for [`FmAudioDemodulator`].
#[derive(Debug, Clone, Copy)]
pub struct FmAudioConfig {
    /// Audio sample rate in Hz.
    pub sample_rate: u32,
    /// Symbol rate in baud.
    pub baudrate: u32,
    /// Gaussian BT product for the receive matched filter. `None` disables
    /// the filter (appropriate for plain FSK).
    pub gaussian_bt: Option<f32>,
    /// Decode differential symbol encoding after hard slicing.
    pub differential: bool,
    /// Invert hard symbol decisions.
    pub invert: bool,
    /// Enable DC-blocking filter on the input audio.
    pub dc_block: bool,
}

impl FmAudioConfig {
    /// Create a configuration with conservative defaults.
    pub fn new(sample_rate: u32, baudrate: u32) -> Self {
        Self {
            sample_rate,
            baudrate,
            gaussian_bt: None,
            differential: false,
            invert: false,
            dc_block: true,
        }
    }

    /// Create a GMSK configuration with Gaussian matched filter.
    pub fn gmsk(sample_rate: u32, baudrate: u32, bt: f32) -> Self {
        Self {
            sample_rate,
            baudrate,
            gaussian_bt: Some(bt),
            differential: false,
            invert: false,
            dc_block: true,
        }
    }
}

/// Symbol demodulator for FM-demodulated audio input.
#[derive(Debug, Clone)]
pub struct FmAudioDemodulator {
    config: FmAudioConfig,
    dc_blocker: Option<DcBlocker>,
    filter: Option<GaussianFir>,
    samples_per_symbol: f32,
    sps_int: usize,
    stage: SamplingStage,
    previous_symbol: Option<u8>,
    envelope: f32,
    peak: f32,
}

impl FmAudioDemodulator {
    /// Create a demodulator from a validated configuration.
    pub fn new(config: FmAudioConfig) -> Result<Self, DecodeError> {
        if config.sample_rate == 0 {
            return Err(DecodeError::InvalidEncoding(
                "FM audio sample rate must be greater than zero".to_string(),
            ));
        }
        if config.baudrate == 0 {
            return Err(DecodeError::InvalidEncoding(
                "FM audio baudrate must be greater than zero".to_string(),
            ));
        }
        if config.sample_rate < config.baudrate * 2 {
            return Err(DecodeError::InvalidEncoding(
                "FM audio sample rate must be at least 2x the baudrate".to_string(),
            ));
        }
        if let Some(bt) = config.gaussian_bt {
            if bt <= 0.0 {
                return Err(DecodeError::InvalidEncoding(
                    "FM audio gaussian BT must be greater than zero".to_string(),
                ));
            }
        }

        let samples_per_symbol = config.sample_rate as f32 / config.baudrate as f32;
        let sps_int = samples_per_symbol.round().max(2.0) as usize;
        let filter = config
            .gaussian_bt
            .map(|bt| GaussianFir::new(bt, samples_per_symbol));

        Ok(Self {
            config,
            dc_blocker: if config.dc_block {
                Some(DcBlocker::new())
            } else {
                None
            },
            filter,
            samples_per_symbol,
            sps_int,
            stage: SamplingStage::Idle,
            previous_symbol: None,
            envelope: 0.0,
            peak: 0.0,
        })
    }

    /// Return the configuration used by this demodulator.
    pub fn config(&self) -> FmAudioConfig {
        self.config
    }

    fn hard_slice(&mut self, sample: f32) -> u8 {
        let mut symbol = u8::from(sample >= 0.0);
        if self.config.invert {
            symbol ^= 1;
        }

        if self.config.differential {
            let decoded = match self.previous_symbol {
                Some(previous) => symbol ^ previous,
                None => symbol,
            };
            self.previous_symbol = Some(symbol);
            decoded
        } else {
            symbol
        }
    }

    fn preprocess(&mut self, sample: f32) -> f32 {
        let dc = match &mut self.dc_blocker {
            Some(blocker) => blocker.push(sample),
            None => sample,
        };
        match &mut self.filter {
            Some(filter) => filter.push(dc),
            None => dc,
        }
    }

    fn build_sampler(&self) -> FmSymbolSampler {
        match self.config.gaussian_bt {
            Some(_) => FmSymbolSampler::Tracking {
                interp: TrackingInterpolator::new(self.samples_per_symbol, MM_TIMING_GAIN),
            },
            None => FmSymbolSampler::IntegrateAndDump {
                integrator: IntegrateAndDump::new(self.samples_per_symbol),
            },
        }
    }
}

impl Demodulator for FmAudioDemodulator {
    type Sample = f32;

    fn push_samples(&mut self, samples: &[f32]) -> Vec<u8> {
        let mut bits = Vec::new();

        for &sample in samples {
            let processed = self.preprocess(sample);
            let abs = processed.abs();
            self.envelope = (self.envelope * ENVELOPE_DECAY).max(abs);
            self.peak = (self.peak * PEAK_DECAY).max(self.envelope);
            let active = self.peak < 1e-9 || self.envelope > self.peak * ACTIVITY_FRAC;

            match &mut self.stage {
                SamplingStage::Idle => {
                    if active {
                        if self.config.gaussian_bt.is_some() {
                            // TrackingInterpolator has M&M timing recovery —
                            // skip phase search and let it converge naturally.
                            let mut sampler = self.build_sampler();
                            if let Some(symbol) = sampler.push(processed) {
                                bits.push(self.hard_slice(symbol));
                            }
                            self.stage = SamplingStage::Run { sampler };
                        } else {
                            let target = self.sps_int * PHASE_SEARCH_SYMBOLS;
                            let mut buffer = Vec::with_capacity(target);
                            buffer.push(processed);
                            self.stage = SamplingStage::Search { buffer, target };
                        }
                    }
                }
                SamplingStage::Search { buffer, target } => {
                    buffer.push(processed);
                    if buffer.len() >= *target {
                        let buffered = std::mem::take(buffer);
                        let best_phase = best_phase_in(&buffered, self.sps_int);
                        let mut sampler = self.build_sampler();
                        for &queued in &buffered[best_phase..] {
                            if let Some(symbol) = sampler.push(queued) {
                                bits.push(self.hard_slice(symbol));
                            }
                        }
                        self.stage = SamplingStage::Run { sampler };
                    }
                }
                SamplingStage::Run { sampler } => {
                    if let Some(symbol) = sampler.push(processed) {
                        bits.push(self.hard_slice(symbol));
                    }
                }
            }
        }

        bits
    }

    fn sample_rate(&self) -> u32 {
        self.config.sample_rate
    }

    fn baudrate(&self) -> u32 {
        self.config.baudrate
    }
}

/// Pick the integer-sample phase with the largest eye opening, measured as
/// Σ|sample| at candidate sample instants.
fn best_phase_in(buffer: &[f32], sps_int: usize) -> usize {
    let mut best_phase = 0;
    let mut best_score = f32::NEG_INFINITY;
    for phase in 0..sps_int {
        let mut sum = 0.0f32;
        let mut idx = phase;
        while idx < buffer.len() {
            sum += buffer[idx].abs();
            idx += sps_int;
        }
        if sum > best_score {
            best_score = sum;
            best_phase = phase;
        }
    }
    best_phase
}

#[derive(Debug, Clone)]
enum SamplingStage {
    Idle,
    Search { buffer: Vec<f32>, target: usize },
    Run { sampler: FmSymbolSampler },
}

#[derive(Debug, Clone)]
enum FmSymbolSampler {
    IntegrateAndDump { integrator: IntegrateAndDump },
    Tracking { interp: TrackingInterpolator },
}

impl FmSymbolSampler {
    fn push(&mut self, sample: f32) -> Option<f32> {
        match self {
            Self::IntegrateAndDump { integrator } => integrator.push(sample),
            Self::Tracking { interp } => interp.push(sample),
        }
    }
}

/// First-order DC-blocking IIR filter: `y[n] = x[n] - x[n-1] + alpha * y[n-1]`.
#[derive(Debug, Clone)]
struct DcBlocker {
    prev_input: f32,
    prev_output: f32,
    alpha: f32,
}

impl DcBlocker {
    fn new() -> Self {
        Self {
            prev_input: 0.0,
            prev_output: 0.0,
            alpha: 0.995,
        }
    }

    fn push(&mut self, sample: f32) -> f32 {
        let output = sample - self.prev_input + self.alpha * self.prev_output;
        self.prev_input = sample;
        self.prev_output = output;
        output
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn synthesize_fm_fsk(bits: &[u8], sample_rate: u32, baudrate: u32) -> Vec<f32> {
        let samples_per_symbol = sample_rate / baudrate;
        let mut out = Vec::with_capacity(bits.len() * samples_per_symbol as usize);
        for &bit in bits {
            let value = if bit & 1 == 1 { 1.0f32 } else { -1.0 };
            for _ in 0..samples_per_symbol {
                out.push(value);
            }
        }
        out
    }

    #[test]
    fn recovers_fsk_from_fm_audio() {
        let bits: Vec<u8> = (0..256u32)
            .map(|i| {
                let mixed = i.wrapping_mul(2_654_435_761).wrapping_add(1);
                ((mixed >> 16) & 1) as u8
            })
            .collect();
        let samples = synthesize_fm_fsk(&bits, 48_000, 1_200);
        let config = FmAudioConfig::new(48_000, 1_200);
        let mut demod = FmAudioDemodulator::new(config).unwrap();

        let recovered = demod.push_samples(&samples);

        assert!(
            recovered.len() >= bits.len() - 4,
            "too few symbols recovered: {}",
            recovered.len()
        );

        let mut best_matches = 0usize;
        for offset in 0..=4 {
            if recovered.len() < offset + bits.len() {
                break;
            }
            let matches = bits
                .iter()
                .zip(&recovered[offset..offset + bits.len()])
                .filter(|(a, b)| a == b)
                .count();
            if matches > best_matches {
                best_matches = matches;
            }
        }
        let total = bits.len();
        assert!(
            best_matches * 100 >= total * 95,
            "FSK FM audio match rate too low: {best_matches}/{total}"
        );
    }

    #[test]
    fn recovers_gmsk_from_fm_audio() {
        let bits: Vec<u8> = (0..256u32)
            .map(|i| {
                let mixed = i.wrapping_mul(1_103_515_245).wrapping_add(12_345);
                ((mixed >> 16) & 1) as u8
            })
            .collect();

        let sps = 48_000u32 / 9_600;
        let sps_f = sps as f32;
        let span = 4usize;
        let pulse_len = sps as usize * span;
        let centre = (pulse_len as f32 - 1.0) / 2.0;
        let bt = 0.5f32;
        let alpha = (std::f32::consts::PI * bt) / (sps_f * (2.0_f32.ln() / 2.0).sqrt());
        let mut pulse = vec![0.0f32; pulse_len];
        let mut sum = 0.0f32;
        for (n, tap) in pulse.iter_mut().enumerate() {
            let x = alpha * (n as f32 - centre);
            *tap = (-x * x).exp();
            sum += *tap;
        }
        let scale = span as f32 / sum;
        for tap in &mut pulse {
            *tap *= scale;
        }

        let pad = span;
        let total_symbols = bits.len() + 2 * pad;
        let total_samples = total_symbols * sps as usize;
        let mut nrz = vec![0.0f32; total_symbols];
        for (i, &bit) in bits.iter().enumerate() {
            nrz[pad + i] = if bit & 1 == 1 { 1.0 } else { -1.0 };
        }

        let mut freq = vec![0.0f32; total_samples];
        for (sym_index, &symbol) in nrz.iter().enumerate() {
            if symbol == 0.0 {
                continue;
            }
            let start = sym_index * sps as usize;
            for (offset, &tap) in pulse.iter().enumerate() {
                let index = start + offset;
                if index < total_samples {
                    freq[index] += symbol * tap;
                }
            }
        }

        let config = FmAudioConfig::gmsk(48_000, 9_600, 0.5);
        let mut demod = FmAudioDemodulator::new(config).unwrap();
        let recovered = demod.push_samples(&freq);

        let mut best_matches = 0usize;
        for offset in 0..=12 {
            if recovered.len() < offset + bits.len() {
                break;
            }
            let matches = bits
                .iter()
                .zip(&recovered[offset..offset + bits.len()])
                .filter(|(a, b)| a == b)
                .count();
            if matches > best_matches {
                best_matches = matches;
            }
        }
        let total = bits.len();
        assert!(
            best_matches * 100 >= total * 90,
            "GMSK FM audio match rate too low: {best_matches}/{total}"
        );
    }

    #[test]
    fn rejects_invalid_config() {
        let config = FmAudioConfig::new(48_000, 0);
        assert!(FmAudioDemodulator::new(config).is_err());
    }

    #[test]
    fn tracker_follows_clock_offset() {
        // Synthesize a GMSK-shaped signal at nominal 48 kHz / 9600 baud,
        // then decode it with the demodulator configured for a slightly
        // higher baudrate (~1% mismatch). Without timing tracking the
        // accumulated phase drift slips a full symbol over a few hundred
        // bits; the M&M tracker should still recover >90%.
        let bits: Vec<u8> = (0..512u32)
            .map(|i| {
                let mixed = i.wrapping_mul(2_654_435_761).wrapping_add(7);
                ((mixed >> 16) & 1) as u8
            })
            .collect();

        let sps = 5usize;
        let span = 4usize;
        let pulse_len = sps * span;
        let centre = (pulse_len as f32 - 1.0) / 2.0;
        let bt = 0.5f32;
        let alpha = (std::f32::consts::PI * bt) / (sps as f32 * (2.0_f32.ln() / 2.0).sqrt());
        let mut pulse = vec![0.0f32; pulse_len];
        let mut sum = 0.0f32;
        for (n, tap) in pulse.iter_mut().enumerate() {
            let x = alpha * (n as f32 - centre);
            *tap = (-x * x).exp();
            sum += *tap;
        }
        let scale = span as f32 / sum;
        for tap in &mut pulse {
            *tap *= scale;
        }

        let pad = span;
        let total_symbols = bits.len() + 2 * pad;
        let total_samples = total_symbols * sps;
        let mut nrz = vec![0.0f32; total_symbols];
        for (i, &bit) in bits.iter().enumerate() {
            nrz[pad + i] = if bit & 1 == 1 { 1.0 } else { -1.0 };
        }

        let mut freq = vec![0.0f32; total_samples];
        for (sym_index, &symbol) in nrz.iter().enumerate() {
            if symbol == 0.0 {
                continue;
            }
            let start = sym_index * sps;
            for (offset, &tap) in pulse.iter().enumerate() {
                let index = start + offset;
                if index < total_samples {
                    freq[index] += symbol * tap;
                }
            }
        }

        // Configure the receiver for a slightly higher baudrate than the
        // synthesis used: this simulates a few hundred PPM of TX/RX clock
        // drift, which is at the high end of what a TCXO-stabilised
        // amateur radio satellite actually exhibits in flight.
        let config = FmAudioConfig::gmsk(48_000, 9_629, 0.5);
        let mut demod = FmAudioDemodulator::new(config).unwrap();
        let recovered = demod.push_samples(&freq);

        let mut best_matches = 0usize;
        for offset in 0..=12 {
            if recovered.len() < offset + bits.len() {
                break;
            }
            let matches = bits
                .iter()
                .zip(&recovered[offset..offset + bits.len()])
                .filter(|(a, b)| a == b)
                .count();
            if matches > best_matches {
                best_matches = matches;
            }
        }
        let total = bits.len();
        assert!(
            best_matches * 100 >= total * 90,
            "M&M tracker failed to follow clock offset: {best_matches}/{total}"
        );
    }

    #[test]
    fn phase_search_locks_on_offset_signal() {
        // Simulate an FSK signal whose first sample falls mid-symbol so
        // phase 0 sampling would land on transitions. Phase search must
        // recover the correct symbol grid.
        let sps = 40usize;
        let bits: Vec<u8> = (0..128u32).map(|i| (i & 1) as u8).collect();
        let mut samples = vec![0.0f32; sps / 2];
        for &bit in &bits {
            let value = if bit == 1 { 1.0f32 } else { -1.0 };
            for _ in 0..sps {
                samples.push(value);
            }
        }

        let config = FmAudioConfig::new(48_000, 1_200);
        let mut demod = FmAudioDemodulator::new(config).unwrap();
        let recovered = demod.push_samples(&samples);

        // Allow the recovered bit stream to be aligned to any short offset,
        // then compare with the input bit pattern.
        let mut best_matches = 0usize;
        for offset in 0..=4 {
            if recovered.len() < offset + bits.len() {
                break;
            }
            let matches = bits
                .iter()
                .zip(&recovered[offset..offset + bits.len()])
                .filter(|(a, b)| a == b)
                .count();
            if matches > best_matches {
                best_matches = matches;
            }
        }
        let total = bits.len();
        assert!(
            best_matches * 100 >= total * 95,
            "phase search failed to lock: {best_matches}/{total}"
        );
    }
}
