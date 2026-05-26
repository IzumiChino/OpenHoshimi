//! Generic continuous-phase modulation demodulator.
//!
//! Covers the binary CPM family used by amateur satellite downlinks: FSK,
//! MSK, GFSK, and GMSK. Pipeline:
//!
//! 1. Optional `swap_iq` and a static frequency-offset NCO.
//! 2. Optional complex IQ low-pass + integer decimation when the input
//!    sample rate is much higher than the symbol rate. This MUST run
//!    before the FM discriminator: atan2 is nonlinear, so out-of-band
//!    noise dominates the per-sample phase delta when the input sample
//!    rate is far above the signal bandwidth.
//! 3. Per-sample noncoherent FM discriminator (atan2 of cross/dot products)
//!    on the decimated IQ stream.
//! 4. For FSK/MSK: integrate-and-dump symbol detection over a fixed
//!    sample window per symbol.
//! 5. For GFSK/GMSK: Gaussian receive matched filter sized from the
//!    configured `gaussian_bt`, then sub-sample interpolation at the
//!    nominal symbol rate.
//! 6. Hard slicer with optional differential decoding and inversion.

use openhoshimi_core::{DecodeError, Demodulator, IqSample};

/// Binary continuous-phase modulation mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CpmMode {
    /// Binary frequency-shift keying.
    Fsk,
    /// Minimum-shift keying.
    Msk,
    /// Gaussian-filtered binary frequency-shift keying.
    Gfsk,
    /// Gaussian minimum-shift keying.
    Gmsk,
}

/// Configuration for [`CpmDemodulator`].
#[derive(Debug, Clone, Copy)]
pub struct CpmConfig {
    /// IQ sample rate in Hz.
    pub sample_rate: u32,
    /// Symbol rate in baud.
    pub baudrate: u32,
    /// CPM waveform family.
    pub mode: CpmMode,
    /// Modulation index. MSK and GMSK usually use `0.5`.
    pub modulation_index: f32,
    /// Gaussian BT product for GFSK/GMSK modes.
    pub gaussian_bt: Option<f32>,
    /// Frequency offset correction in Hz, applied before phase slicing.
    pub frequency_offset_hz: f32,
    /// Decode differential symbol encoding after hard slicing.
    pub differential: bool,
    /// Invert hard symbol decisions.
    pub invert: bool,
    /// Swap I and Q before carrier correction.
    pub swap_iq: bool,
    /// Integer decimation factor applied to the post-discriminator sample
    /// stream before symbol detection. `0` selects an automatic factor that
    /// keeps the post-decimation rate near 16 samples per symbol when the
    /// input rate is more than 32x the baud rate; otherwise no decimation
    /// is applied. Set to `1` to force no decimation.
    pub decimation: u32,
    /// Reserved for future closed-loop timing recovery. Currently unused;
    /// the demodulator runs open-loop at the nominal symbol rate.
    pub timing_loop_bandwidth: f32,
}

impl CpmConfig {
    /// Create a configuration with conservative defaults for `mode`.
    pub fn new(sample_rate: u32, baudrate: u32, mode: CpmMode) -> Self {
        Self {
            sample_rate,
            baudrate,
            mode,
            modulation_index: match mode {
                CpmMode::Msk | CpmMode::Gmsk => 0.5,
                CpmMode::Fsk | CpmMode::Gfsk => 1.0,
            },
            gaussian_bt: match mode {
                CpmMode::Gfsk | CpmMode::Gmsk => Some(0.5),
                CpmMode::Fsk | CpmMode::Msk => None,
            },
            frequency_offset_hz: 0.0,
            differential: false,
            invert: false,
            swap_iq: false,
            decimation: 0,
            timing_loop_bandwidth: 0.0,
        }
    }
}

/// Resolve the integer decimation factor used between the discriminator and
/// the symbol detector. `requested == 0` selects an automatic factor that
/// keeps the post-decimation rate near 16 samples per symbol when the input
/// rate is more than 32x the baud rate.
pub(crate) fn resolve_decimation(sample_rate: u32, baudrate: u32, requested: u32) -> u32 {
    if requested >= 1 {
        return requested;
    }
    if baudrate == 0 {
        return 1;
    }
    let sps_in = sample_rate as f32 / baudrate as f32;
    if sps_in <= 32.0 {
        return 1;
    }
    const TARGET_SPS: f32 = 16.0;
    let factor = (sps_in / TARGET_SPS).round() as u32;
    factor.max(1)
}

/// Noncoherent IQ demodulator for FSK/MSK/GFSK/GMSK signals.
#[derive(Debug, Clone)]
pub struct CpmDemodulator {
    config: CpmConfig,
    last_sample: Option<IqSample>,
    previous_symbol: Option<u8>,
    carrier_phase: f32,
    carrier_increment: f32,
    decimation: u32,
    decim_counter: u32,
    decim_lpf: Option<ComplexFir>,
    sampler: SymbolSampler,
}

impl CpmDemodulator {
    /// Create a demodulator from a validated configuration.
    pub fn new(config: CpmConfig) -> Result<Self, DecodeError> {
        validate_config(config)?;
        let decimation = resolve_decimation(config.sample_rate, config.baudrate, config.decimation);
        let effective_rate = config.sample_rate as f32 / decimation as f32;
        let samples_per_symbol = effective_rate / config.baudrate as f32;
        let decim_lpf = if decimation > 1 {
            Some(ComplexFir::lowpass(decimation))
        } else {
            None
        };
        let sampler = match (config.mode, config.gaussian_bt) {
            (CpmMode::Gfsk | CpmMode::Gmsk, Some(bt)) => SymbolSampler::Filtered {
                filter: GaussianFir::new(bt, samples_per_symbol),
                interp: TrackingInterpolator::new(samples_per_symbol, 0.05),
            },
            _ => SymbolSampler::IntegrateAndDump {
                integrator: IntegrateAndDump::new(samples_per_symbol),
            },
        };
        Ok(Self {
            carrier_increment: -std::f32::consts::TAU * config.frequency_offset_hz
                / config.sample_rate as f32,
            config,
            last_sample: None,
            previous_symbol: None,
            carrier_phase: 0.0,
            decimation,
            decim_counter: 0,
            decim_lpf,
            sampler,
        })
    }

    /// Return the configuration used by this demodulator.
    pub fn config(&self) -> CpmConfig {
        self.config
    }

    /// Return the integer decimation factor applied between the
    /// discriminator and the symbol detector.
    pub fn decimation(&self) -> u32 {
        self.decimation
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

    fn normalize_sample(&self, sample: IqSample) -> IqSample {
        if self.config.swap_iq {
            IqSample {
                i: sample.q,
                q: sample.i,
            }
        } else {
            sample
        }
    }

    fn correct_frequency(&mut self, sample: IqSample) -> IqSample {
        if self.config.frequency_offset_hz == 0.0 {
            return sample;
        }

        let sin = self.carrier_phase.sin();
        let cos = self.carrier_phase.cos();
        self.carrier_phase += self.carrier_increment;
        if self.carrier_phase >= std::f32::consts::TAU
            || self.carrier_phase <= -std::f32::consts::TAU
        {
            self.carrier_phase %= std::f32::consts::TAU;
        }

        IqSample {
            i: sample.i * cos - sample.q * sin,
            q: sample.i * sin + sample.q * cos,
        }
    }
}

impl Demodulator for CpmDemodulator {
    type Sample = IqSample;

    fn push_samples(&mut self, samples: &[IqSample]) -> Vec<u8> {
        let mut bits = Vec::new();

        for &sample in samples {
            let sample = self.correct_frequency(self.normalize_sample(sample));

            let filtered_iq = match &mut self.decim_lpf {
                Some(lpf) => lpf.push(sample),
                None => sample,
            };
            if self.decimation > 1 {
                self.decim_counter += 1;
                if self.decim_counter < self.decimation {
                    continue;
                }
                self.decim_counter = 0;
            }

            let delta = match self.last_sample {
                Some(previous) => phase_delta(previous, filtered_iq),
                None => 0.0,
            };
            self.last_sample = Some(filtered_iq);

            if let Some(symbol_value) = self.sampler.push(delta) {
                bits.push(self.hard_slice(symbol_value));
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

fn validate_config(config: CpmConfig) -> Result<(), DecodeError> {
    if config.sample_rate == 0 {
        return Err(DecodeError::InvalidEncoding(
            "CPM sample rate must be greater than zero".to_string(),
        ));
    }
    if config.baudrate == 0 {
        return Err(DecodeError::InvalidEncoding(
            "CPM baudrate must be greater than zero".to_string(),
        ));
    }
    if config.sample_rate < config.baudrate * 2 {
        return Err(DecodeError::InvalidEncoding(
            "CPM sample rate must be at least 2x the baudrate".to_string(),
        ));
    }
    if config.modulation_index <= 0.0 {
        return Err(DecodeError::InvalidEncoding(
            "CPM modulation index must be greater than zero".to_string(),
        ));
    }
    if let Some(bt) = config.gaussian_bt {
        if bt <= 0.0 {
            return Err(DecodeError::InvalidEncoding(
                "CPM gaussian BT must be greater than zero".to_string(),
            ));
        }
    }
    Ok(())
}

fn phase_delta(previous: IqSample, current: IqSample) -> f32 {
    let dot = previous.i.mul_add(current.i, previous.q * current.q);
    let cross = previous.i * current.q - previous.q * current.i;
    cross.atan2(dot)
}

#[derive(Debug, Clone)]
enum SymbolSampler {
    IntegrateAndDump {
        integrator: IntegrateAndDump,
    },
    Filtered {
        filter: GaussianFir,
        interp: TrackingInterpolator,
    },
}

impl SymbolSampler {
    fn push(&mut self, delta: f32) -> Option<f32> {
        match self {
            SymbolSampler::IntegrateAndDump { integrator } => integrator.push(delta),
            SymbolSampler::Filtered { filter, interp } => {
                let filtered = filter.push(delta);
                interp.push(filtered)
            }
        }
    }
}

/// Integrate-and-dump symbol detector for unfiltered FSK/MSK signals. Sums
/// the post-discriminator samples over a fractional symbol period and emits
/// the accumulated value as the symbol estimate.
#[derive(Debug, Clone)]
pub(crate) struct IntegrateAndDump {
    samples_per_symbol: f32,
    sample_phase: f32,
    accumulator: f32,
}

impl IntegrateAndDump {
    pub(crate) fn new(samples_per_symbol: f32) -> Self {
        Self {
            samples_per_symbol,
            sample_phase: 0.0,
            accumulator: 0.0,
        }
    }

    pub(crate) fn push(&mut self, delta: f32) -> Option<f32> {
        self.accumulator += delta;
        self.sample_phase += 1.0;
        if self.sample_phase >= self.samples_per_symbol {
            let symbol = self.accumulator;
            self.accumulator = 0.0;
            self.sample_phase -= self.samples_per_symbol;
            Some(symbol)
        } else {
            None
        }
    }
}

/// Closed-loop symbol-rate interpolator with Mueller-Müller decision-directed
/// timing recovery for binary signals. Tracks small fractional clock offsets
/// (TX/RX PPM mismatch) over long recordings by nudging the fractional
/// sampling phase from the timing error of consecutive output samples. The
/// loop update is gated by a leak-decay envelope estimate so that timing
/// does not wander on the noise/silence between bursts.
#[derive(Debug, Clone)]
pub(crate) struct TrackingInterpolator {
    increment: f32,
    mu: f32,
    prev_input: f32,
    last_output: f32,
    have_output: bool,
    primed: bool,
    gain: f32,
    envelope: f32,
}

impl TrackingInterpolator {
    pub(crate) fn new(samples_per_symbol: f32, gain: f32) -> Self {
        Self {
            increment: 1.0 / samples_per_symbol,
            mu: 0.0,
            prev_input: 0.0,
            last_output: 0.0,
            have_output: false,
            primed: false,
            gain,
            envelope: 0.0,
        }
    }

    pub(crate) fn push(&mut self, sample: f32) -> Option<f32> {
        if !self.primed {
            self.prev_input = sample;
            self.primed = true;
            return None;
        }

        let next_phase = self.mu + self.increment;
        if next_phase >= 1.0 {
            let frac = ((1.0 - self.mu) / self.increment).clamp(0.0, 1.0);
            let symbol = lerp(self.prev_input, sample, frac);
            let abs_symbol = symbol.abs();

            self.envelope = (self.envelope * 0.999).max(abs_symbol);

            let mut new_mu = next_phase - 1.0;
            if self.have_output {
                let abs_prev = self.last_output.abs();
                let gate = self.envelope * 0.4;
                if abs_symbol >= gate && abs_prev >= gate {
                    let sign_curr = if symbol >= 0.0 { 1.0 } else { -1.0 };
                    let sign_prev = if self.last_output >= 0.0 { 1.0 } else { -1.0 };
                    let error = self.last_output * sign_curr - symbol * sign_prev;
                    new_mu += self.gain * error;
                    if new_mu < 0.0 {
                        new_mu = 0.0;
                    } else if new_mu >= 1.0 {
                        new_mu = 1.0 - self.increment * 0.5;
                    }
                }
            }
            self.mu = new_mu;
            self.last_output = symbol;
            self.have_output = true;
            self.prev_input = sample;
            Some(symbol)
        } else {
            self.mu = next_phase;
            self.prev_input = sample;
            None
        }
    }
}

pub(crate) fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t
}

/// Symmetric Gaussian FIR shaped by a BT product, span 4 symbols.
#[derive(Debug, Clone)]
pub(crate) struct GaussianFir {
    taps: Vec<f32>,
    history: Vec<f32>,
    head: usize,
}

impl GaussianFir {
    pub(crate) fn new(bt: f32, samples_per_symbol: f32) -> Self {
        let taps = gaussian_taps(bt, samples_per_symbol, 4);
        let history = vec![0.0; taps.len()];
        Self {
            taps,
            history,
            head: 0,
        }
    }

    pub(crate) fn push(&mut self, sample: f32) -> f32 {
        if self.taps.is_empty() {
            return sample;
        }
        self.history[self.head] = sample;
        let len = self.history.len();
        let mut acc = 0.0f32;
        for (offset, tap) in self.taps.iter().enumerate() {
            let index = (self.head + len - offset) % len;
            acc += self.history[index] * tap;
        }
        self.head = (self.head + 1) % len;
        acc
    }
}

pub(crate) fn gaussian_taps(bt: f32, samples_per_symbol: f32, span_symbols: usize) -> Vec<f32> {
    let half = (span_symbols as f32 * samples_per_symbol * 0.5).ceil() as i32;
    let len = (2 * half + 1) as usize;
    let alpha = (std::f32::consts::PI * bt) / (samples_per_symbol * (2.0_f32.ln() / 2.0).sqrt());
    let mut taps = Vec::with_capacity(len);
    let mut sum = 0.0f32;
    for n in -half..=half {
        let x = alpha * n as f32;
        let value = (-x * x).exp();
        taps.push(value);
        sum += value;
    }
    if sum > 0.0 {
        for tap in &mut taps {
            *tap /= sum;
        }
    }
    taps
}

/// Real-valued FIR with a circular history buffer. Used as the anti-alias
/// low-pass before integer decimation of the post-discriminator stream.
#[derive(Debug, Clone)]
pub(crate) struct RealFir {
    taps: Vec<f32>,
    history: Vec<f32>,
    head: usize,
}

impl RealFir {
    /// Build a Hamming-windowed sinc low-pass with cutoff at the post-decimation
    /// Nyquist frequency, sized for the given decimation factor.
    pub(crate) fn lowpass(decimation: u32) -> Self {
        let decim = decimation.max(1) as f32;
        let cutoff = 0.45f32 / decim;
        let span = (8.0 * decim).ceil() as usize;
        let len = span | 1;
        let taps = sinc_lowpass_taps(cutoff, len);
        let history = vec![0.0; len];
        Self {
            taps,
            history,
            head: 0,
        }
    }

    pub(crate) fn push(&mut self, sample: f32) -> f32 {
        if self.taps.is_empty() {
            return sample;
        }
        self.history[self.head] = sample;
        let len = self.history.len();
        let mut acc = 0.0f32;
        for (offset, tap) in self.taps.iter().enumerate() {
            let index = (self.head + len - offset) % len;
            acc += self.history[index] * tap;
        }
        self.head = (self.head + 1) % len;
        acc
    }
}

/// Complex-valued FIR built from two `RealFir` channels. Used as the
/// anti-alias low-pass on the carrier-corrected IQ stream before integer
/// decimation feeds the FM discriminator.
#[derive(Debug, Clone)]
pub(crate) struct ComplexFir {
    i: RealFir,
    q: RealFir,
}

impl ComplexFir {
    /// Build a Hamming-windowed sinc low-pass with cutoff at the
    /// post-decimation Nyquist frequency, sized for the given decimation
    /// factor. The same real taps are applied to I and Q independently —
    /// this is the standard linear-phase complex baseband anti-alias.
    pub(crate) fn lowpass(decimation: u32) -> Self {
        Self {
            i: RealFir::lowpass(decimation),
            q: RealFir::lowpass(decimation),
        }
    }

    pub(crate) fn push(&mut self, sample: IqSample) -> IqSample {
        IqSample {
            i: self.i.push(sample.i),
            q: self.q.push(sample.q),
        }
    }
}

/// Hamming-windowed sinc low-pass with normalized cutoff `cutoff` cycles per
/// sample (must be in `(0, 0.5)`). `len` is rounded up to the nearest odd
/// number internally if needed.
pub(crate) fn sinc_lowpass_taps(cutoff: f32, len: usize) -> Vec<f32> {
    let len = if len % 2 == 0 { len + 1 } else { len };
    let half = (len / 2) as i32;
    let mut taps = Vec::with_capacity(len);
    let mut sum = 0.0f32;
    for n in -half..=half {
        let x = n as f32;
        let sinc = if n == 0 {
            2.0 * cutoff
        } else {
            let arg = std::f32::consts::TAU * cutoff * x;
            arg.sin() / (std::f32::consts::PI * x)
        };
        let window =
            0.54 - 0.46 * (std::f32::consts::TAU * (n + half) as f32 / (len - 1) as f32).cos();
        let value = sinc * window;
        taps.push(value);
        sum += value;
    }
    if sum > 0.0 {
        for tap in &mut taps {
            *tap /= sum;
        }
    }
    taps
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::TAU;

    fn synthesize_fsk(bits: &[u8], sample_rate: u32, baudrate: u32) -> Vec<IqSample> {
        let samples_per_symbol = sample_rate / baudrate;
        let deviation_hz = baudrate as f32;
        let mut phase = 0.0f32;
        let mut out = Vec::with_capacity(bits.len() * samples_per_symbol as usize);

        for &bit in bits {
            let freq = if bit & 1 == 1 {
                deviation_hz
            } else {
                -deviation_hz
            };
            let increment = TAU * freq / sample_rate as f32;
            for _ in 0..samples_per_symbol {
                out.push(IqSample {
                    i: phase.cos(),
                    q: phase.sin(),
                });
                phase += increment;
            }
        }

        out
    }

    /// Synthesize a GMSK waveform by convolving an NRZ symbol stream with a
    /// Gaussian frequency pulse and integrating to phase. The pulse is
    /// normalized so the per-symbol phase change is exactly `pi *
    /// modulation_index` for a +1 NRZ symbol.
    fn synthesize_gmsk(
        bits: &[u8],
        sample_rate: u32,
        baudrate: u32,
        bt: f32,
        modulation_index: f32,
    ) -> Vec<IqSample> {
        let sps_int = (sample_rate / baudrate) as usize;
        let sps = sps_int as f32;
        let span_symbols = 4usize;
        let pulse_len = sps_int * span_symbols;
        let centre = (pulse_len as f32 - 1.0) / 2.0;
        let alpha = (std::f32::consts::PI * bt) / (sps * (2.0_f32.ln() / 2.0).sqrt());
        let mut pulse = vec![0.0f32; pulse_len];
        let mut sum = 0.0f32;
        for (n, tap) in pulse.iter_mut().enumerate() {
            let x = alpha * (n as f32 - centre);
            *tap = (-x * x).exp();
            sum += *tap;
        }
        // Normalize so each symbol slice integrates to 1: total area =
        // span_symbols, so per-symbol-mass = 1.
        if sum > 0.0 {
            let scale = span_symbols as f32 / sum;
            for tap in &mut pulse {
                *tap *= scale;
            }
        }

        let pad = span_symbols;
        let total_symbols = bits.len() + 2 * pad;
        let total_samples = total_symbols * sps_int;
        let mut nrz = vec![0.0f32; total_symbols];
        for (i, &bit) in bits.iter().enumerate() {
            nrz[pad + i] = if bit & 1 == 1 { 1.0 } else { -1.0 };
        }

        let mut freq = vec![0.0f32; total_samples];
        for (sym_index, &symbol) in nrz.iter().enumerate() {
            if symbol == 0.0 {
                continue;
            }
            let start = sym_index * sps_int;
            for (offset, &tap) in pulse.iter().enumerate() {
                let index = start + offset;
                if index < total_samples {
                    freq[index] += symbol * tap;
                }
            }
        }

        // Per-sample phase increment: pi * h * freq[n] / sps. With freq
        // integrating to 1 per symbol slice and sps samples per symbol,
        // the per-symbol phase change is pi * h.
        let phase_step = std::f32::consts::PI * modulation_index / sps;
        let mut phase = 0.0f32;
        let mut out = Vec::with_capacity(total_samples);
        for &f in &freq {
            phase += f * phase_step;
            out.push(IqSample {
                i: phase.cos(),
                q: phase.sin(),
            });
        }
        out
    }

    /// Search for the best alignment between expected and recovered
    /// streams within a small window to absorb pulse ring-up, LPF group
    /// delay, and sub-sample phase offsets. Returns `(best_matches,
    /// best_offset)`.
    fn best_alignment(expected: &[u8], recovered: &[u8], max_offset: usize) -> (usize, usize) {
        let mut best_matches = 0usize;
        let mut best_offset = 0usize;
        for offset in 0..=max_offset {
            if recovered.len() < offset + expected.len() {
                break;
            }
            let matches = expected
                .iter()
                .zip(&recovered[offset..offset + expected.len()])
                .filter(|(left, right)| left == right)
                .count();
            if matches > best_matches {
                best_matches = matches;
                best_offset = offset;
            }
        }
        (best_matches, best_offset)
    }

    #[test]
    fn recovers_binary_fsk_symbols() {
        let bits: Vec<u8> = (0..64u32)
            .map(|i| {
                let mixed = i.wrapping_mul(2_654_435_761).wrapping_add(7);
                ((mixed >> 17) & 1) as u8
            })
            .collect();
        let samples = synthesize_fsk(&bits, 48_000, 1_200);
        let config = CpmConfig::new(48_000, 1_200, CpmMode::Fsk);
        let mut demodulator = match CpmDemodulator::new(config) {
            Ok(demodulator) => demodulator,
            Err(err) => panic!("valid config: {err}"),
        };

        let recovered = demodulator.push_samples(&samples);

        // Compare a leading window short enough that recovered.len() is
        // guaranteed to cover it after LPF startup + decimation rounding.
        let window = 32usize;
        let (best_matches, best_offset) = best_alignment(&bits[..window], &recovered, 8);
        assert!(
            best_matches * 100 >= window * 90,
            "FSK match rate too low: {best_matches}/{window} at offset {best_offset}"
        );
    }

    #[test]
    fn recovers_binary_fsk_symbols_with_swapped_iq() {
        let bits: Vec<u8> = (0..64u32)
            .map(|i| {
                let mixed = i.wrapping_mul(2_654_435_761).wrapping_add(7);
                ((mixed >> 17) & 1) as u8
            })
            .collect();
        let mut samples = synthesize_fsk(&bits, 48_000, 1_200);
        for sample in &mut samples {
            std::mem::swap(&mut sample.i, &mut sample.q);
        }
        let mut config = CpmConfig::new(48_000, 1_200, CpmMode::Fsk);
        config.swap_iq = true;
        let mut demodulator = match CpmDemodulator::new(config) {
            Ok(demodulator) => demodulator,
            Err(err) => panic!("valid config: {err}"),
        };

        let recovered = demodulator.push_samples(&samples);

        let window = 32usize;
        let (best_matches, best_offset) = best_alignment(&bits[..window], &recovered, 8);
        assert!(
            best_matches * 100 >= window * 90,
            "FSK match rate too low: {best_matches}/{window} at offset {best_offset}"
        );
    }

    #[test]
    fn recovers_gmsk_symbols() {
        // Use a longer pseudo-random bit stream so steady-state recovery
        // dominates the assertion.
        let bits: Vec<u8> = (0..64u32)
            .map(|i| {
                let mixed = i.wrapping_mul(1_103_515_245).wrapping_add(12_345);
                ((mixed >> 16) & 1) as u8
            })
            .collect();
        let samples = synthesize_gmsk(&bits, 48_000, 9_600, 0.5, 0.5);
        let config = CpmConfig::new(48_000, 9_600, CpmMode::Gmsk);
        let mut demodulator = match CpmDemodulator::new(config) {
            Ok(demodulator) => demodulator,
            Err(err) => panic!("valid config: {err}"),
        };

        let recovered = demodulator.push_samples(&samples);

        // Search for the best alignment between expected and recovered
        // streams within a small window to absorb pulse ring-up and
        // sub-sample phase offsets.
        let mut best_matches = 0usize;
        let mut best_offset = 0usize;
        for offset in 0..=12 {
            if recovered.len() < offset + bits.len() {
                break;
            }
            let matches = bits
                .iter()
                .zip(&recovered[offset..offset + bits.len()])
                .filter(|(left, right)| left == right)
                .count();
            if matches > best_matches {
                best_matches = matches;
                best_offset = offset;
            }
        }
        let total = bits.len();
        assert!(
            best_matches * 100 >= total * 90,
            "GMSK match rate too low: {best_matches}/{total} at offset {best_offset}"
        );
    }

    #[test]
    fn rejects_invalid_baudrate() {
        let config = CpmConfig::new(48_000, 0, CpmMode::Gmsk);
        let err = match CpmDemodulator::new(config) {
            Ok(_) => panic!("invalid config should fail"),
            Err(err) => err,
        };

        assert!(matches!(err, DecodeError::InvalidEncoding(_)));
    }

    #[test]
    fn gaussian_taps_normalise_to_unit_dc_gain() {
        let taps = gaussian_taps(0.5, 5.0, 4);
        let sum: f32 = taps.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6, "sum was {sum}");
    }
}
