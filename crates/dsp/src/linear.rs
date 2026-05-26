//! Generic linear IQ demodulator.
//!
//! This module provides a hard-decision demodulator for BPSK/DBPSK/QPSK/
//! OQPSK streams. A fixed frequency offset can be corrected before slicing,
//! and a small decision-directed carrier tracker keeps the phase from
//! drifting away on real recordings.

use std::f32::consts::TAU;
use std::fs::File;
use std::io::{self, BufWriter, Write};
use std::path::Path;
use std::sync::Mutex;

use openhoshimi_core::{DecodeError, Demodulator, IqSample};

/// Process-wide debug sink for post-Costas symbol decisions. Each line is
/// `i,q\n` of the IQ value handed to the slicer. Enabled by
/// [`open_symbol_dump`]; intended for offline SNR / lock analysis on real
/// recordings.
static SYMBOL_DUMP: Mutex<Option<BufWriter<File>>> = Mutex::new(None);

/// Open (or replace) the process-wide symbol dump file used by
/// [`LinearDemodulator`]. Returns the IO error from [`File::create`] if the
/// path cannot be opened. Pass an empty path to disable.
pub fn open_symbol_dump(path: &Path) -> io::Result<()> {
    let mut guard = SYMBOL_DUMP
        .lock()
        .map_err(|_| io::Error::other("symbol dump mutex poisoned"))?;
    if path.as_os_str().is_empty() {
        *guard = None;
        return Ok(());
    }
    let file = File::create(path)?;
    *guard = Some(BufWriter::new(file));
    Ok(())
}

fn write_symbol_dump(sample: IqSample) {
    let Ok(mut guard) = SYMBOL_DUMP.lock() else {
        return;
    };
    if let Some(writer) = guard.as_mut() {
        let _ = writeln!(writer, "{:.6},{:.6}", sample.i, sample.q);
    }
}

/// Linear modulation family.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LinearMode {
    /// Binary phase-shift keying.
    Bpsk,
    /// Differential binary phase-shift keying.
    Dbpsk,
    /// Quadrature phase-shift keying.
    Qpsk,
    /// Offset quadrature phase-shift keying.
    Oqpsk,
}

/// Configuration for [`LinearDemodulator`].
#[derive(Debug, Clone, Copy)]
pub struct LinearConfig {
    /// IQ sample rate in Hz.
    pub sample_rate: u32,
    /// Symbol rate in baud.
    pub baudrate: u32,
    /// Linear modulation mode.
    pub mode: LinearMode,
    /// Frequency offset correction in Hz, applied before symbol slicing.
    pub frequency_offset_hz: f32,
    /// Decode differential symbol encoding after hard slicing.
    pub differential: bool,
    /// Invert hard symbol decisions.
    pub invert: bool,
    /// Swap I and Q before carrier correction.
    pub swap_iq: bool,
    /// Closed-loop carrier tracker bandwidth, normalized to the symbol rate.
    /// `0.0` disables the tracker and keeps a static frequency correction.
    pub carrier_loop_bandwidth: f32,
    /// Frequency-locked loop bandwidth normalized to the symbol rate. When
    /// `> 0.0` a Kay-style cross-product discriminator on squared symbols
    /// drives the NCO integrator, providing wide pull-in over Doppler
    /// before the phase-locked Costas tracker takes over. `0.0` keeps only
    /// the Costas loop active.
    pub frequency_loop_bandwidth: f32,
    /// Maximum absolute NCO offset from the static frequency correction, in
    /// Hz. `0.0` selects a conservative default of one symbol rate. Increase
    /// for LEO passes whose Doppler swing exceeds the default.
    pub nco_max_offset_hz: f32,
    /// Root-raised-cosine matched filter roll-off factor in the range
    /// `(0.0, 1.0]`. `0.0` disables matched filtering and falls back to the
    /// boxcar integrate-and-dump symbol filter, preserving behaviour of
    /// downlinks that do not assume RRC pulse shaping.
    pub matched_filter_rolloff: f32,
    /// Matched filter span in symbol periods. Used only when
    /// [`Self::matched_filter_rolloff`] is greater than zero. `0` selects a
    /// reasonable default span (six symbols).
    pub matched_filter_span_symbols: usize,
}

impl LinearConfig {
    /// Create a configuration with mode-specific defaults.
    pub fn new(sample_rate: u32, baudrate: u32, mode: LinearMode) -> Self {
        Self {
            sample_rate,
            baudrate,
            mode,
            frequency_offset_hz: 0.0,
            differential: mode == LinearMode::Dbpsk,
            invert: false,
            swap_iq: false,
            carrier_loop_bandwidth: 0.0,
            frequency_loop_bandwidth: 0.0,
            nco_max_offset_hz: 0.0,
            matched_filter_rolloff: 0.0,
            matched_filter_span_symbols: 0,
        }
    }
}

/// Hard-decision IQ demodulator for BPSK/DBPSK/QPSK/OQPSK signals.
#[derive(Debug, Clone)]
pub struct LinearDemodulator {
    config: LinearConfig,
    nominal_samples_per_symbol: f32,
    samples_per_symbol: f32,
    sample_phase: f32,
    carrier_phase: f32,
    carrier_increment: f32,
    nco_increment: f32,
    nco_fll_offset: f32,
    nco_limit_radians: f32,
    loop_alpha: f32,
    loop_beta: f32,
    freq_loop_alpha: f32,
    fll_prev_squared: Option<IqSample>,
    agc_gain: f32,
    agc_alpha: f32,
    dc_i: f32,
    dc_q: f32,
    dc_alpha: f32,
    i_sum: f32,
    q_sum: f32,
    previous_binary_symbol: Option<IqSample>,
    symbol_samples: Vec<IqSample>,
    timing_correction: f32,
    matched_taps: Option<Vec<f32>>,
    matched_buffer_i: Vec<f32>,
    matched_buffer_q: Vec<f32>,
    matched_index: usize,
}

impl LinearDemodulator {
    /// Create a demodulator from a validated configuration.
    pub fn new(config: LinearConfig) -> Result<Self, DecodeError> {
        validate_config(config)?;
        let samples_per_symbol = config.sample_rate as f32 / config.baudrate as f32;
        let (loop_alpha, loop_beta) = loop_filter_gains(config.carrier_loop_bandwidth);
        let freq_loop_alpha = if config.frequency_loop_bandwidth > 0.0 {
            config.frequency_loop_bandwidth
        } else {
            0.0
        };
        let nco_limit_radians = if config.nco_max_offset_hz > 0.0 {
            TAU * config.nco_max_offset_hz / config.sample_rate as f32
        } else {
            TAU * config.baudrate as f32 / config.sample_rate as f32
        };
        let matched_taps = if config.matched_filter_rolloff > 0.0 {
            let span = if config.matched_filter_span_symbols == 0 {
                6
            } else {
                config.matched_filter_span_symbols
            };
            Some(rrc_taps(
                samples_per_symbol,
                config.matched_filter_rolloff,
                span,
            ))
        } else {
            None
        };
        let buffer_len = matched_taps.as_ref().map_or(0, Vec::len);
        // RRC FIRs are symmetric, so the centered sample lags input by
        // (len-1)/2 samples. Start the symbol-clock phase that far behind so
        // the first triggered decision lands on the filter output that
        // represents the first input symbol — keeping sync alignment
        // consistent with the no-filter path used by the prefix detector.
        let initial_sample_phase = match matched_taps.as_ref() {
            Some(taps) if !taps.is_empty() => -((taps.len() - 1) as f32 / 2.0),
            _ => 0.0,
        };
        Ok(Self {
            nominal_samples_per_symbol: samples_per_symbol,
            samples_per_symbol,
            carrier_increment: -TAU * config.frequency_offset_hz / config.sample_rate as f32,
            nco_increment: 0.0,
            nco_fll_offset: 0.0,
            nco_limit_radians,
            loop_alpha,
            loop_beta,
            freq_loop_alpha,
            fll_prev_squared: None,
            agc_gain: 1.0,
            agc_alpha: 2.0e-2 / samples_per_symbol,
            dc_i: 0.0,
            dc_q: 0.0,
            dc_alpha: 2.0e-3 / samples_per_symbol,
            config,
            sample_phase: initial_sample_phase,
            carrier_phase: 0.0,
            i_sum: 0.0,
            q_sum: 0.0,
            previous_binary_symbol: None,
            symbol_samples: Vec::new(),
            timing_correction: 0.0,
            matched_taps,
            matched_buffer_i: vec![0.0; buffer_len],
            matched_buffer_q: vec![0.0; buffer_len],
            matched_index: 0,
        })
    }

    /// Return the configuration used by this demodulator.
    pub fn config(&self) -> LinearConfig {
        self.config
    }

    fn slice_symbol(&mut self, current: IqSample) -> Vec<u8> {
        match self.config.mode {
            LinearMode::Bpsk | LinearMode::Dbpsk => vec![self.slice_binary(current)],
            LinearMode::Qpsk | LinearMode::Oqpsk => self.slice_quadrature(current),
        }
    }

    fn slice_symbol_soft(&mut self, current: IqSample) -> Vec<i8> {
        match self.config.mode {
            LinearMode::Bpsk | LinearMode::Dbpsk => vec![self.slice_binary_soft(current)],
            LinearMode::Qpsk | LinearMode::Oqpsk => self.slice_quadrature_soft(current),
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

    fn slice_binary(&mut self, current: IqSample) -> u8 {
        if self.config.differential {
            return self.slice_differential_binary(current);
        }

        let mut symbol = u8::from(current.i >= 0.0);
        if self.config.invert {
            symbol ^= 1;
        }
        symbol
    }

    fn slice_binary_soft(&mut self, current: IqSample) -> i8 {
        let value = if self.config.differential {
            self.slice_differential_binary_soft(current)
        } else {
            normalize_in_phase(current)
        };
        let value = if self.config.invert { -value } else { value };
        soft_to_i8(value)
    }

    fn slice_differential_binary(&mut self, current: IqSample) -> u8 {
        let mut bit = match self.previous_binary_symbol {
            Some(previous) => {
                let dot = previous.i.mul_add(current.i, previous.q * current.q);
                u8::from(dot < 0.0)
            }
            None => 0,
        };
        self.previous_binary_symbol = Some(current);

        if self.config.invert {
            bit ^= 1;
        }
        bit
    }

    fn slice_differential_binary_soft(&mut self, current: IqSample) -> f32 {
        let value = match self.previous_binary_symbol {
            Some(previous) => {
                let dot = previous.i.mul_add(current.i, previous.q * current.q);
                let mag_prev = previous
                    .i
                    .mul_add(previous.i, previous.q * previous.q)
                    .sqrt();
                let mag_curr = current.i.mul_add(current.i, current.q * current.q).sqrt();
                let denom = mag_prev * mag_curr;
                if denom > f32::EPSILON {
                    (dot / denom).clamp(-1.0, 1.0)
                } else {
                    0.0
                }
            }
            None => 0.0,
        };
        self.previous_binary_symbol = Some(current);
        value
    }

    fn slice_quadrature(&self, current: IqSample) -> Vec<u8> {
        let mut i_bit = u8::from(current.i >= 0.0);
        let mut q_bit = u8::from(current.q >= 0.0);
        if self.config.invert {
            i_bit ^= 1;
            q_bit ^= 1;
        }
        vec![i_bit, q_bit]
    }

    fn slice_quadrature_soft(&self, current: IqSample) -> Vec<i8> {
        let mag = current.i.mul_add(current.i, current.q * current.q).sqrt();
        let (mut i_value, mut q_value) = if mag > f32::EPSILON {
            (current.i / mag, current.q / mag)
        } else {
            (0.0, 0.0)
        };
        if self.config.invert {
            i_value = -i_value;
            q_value = -q_value;
        }
        vec![soft_to_i8(i_value), soft_to_i8(q_value)]
    }

    fn apply_dc_block_and_agc(&mut self, sample: IqSample) -> IqSample {
        self.dc_i += self.dc_alpha * (sample.i - self.dc_i);
        self.dc_q += self.dc_alpha * (sample.q - self.dc_q);

        let corrected = IqSample {
            i: sample.i - self.dc_i,
            q: sample.q - self.dc_q,
        };

        let power = corrected.i.mul_add(corrected.i, corrected.q * corrected.q);
        if power > 0.0 && power.is_finite() {
            let target_gain = 1.0 / power.sqrt();
            self.agc_gain += self.agc_alpha * (target_gain - self.agc_gain);
            self.agc_gain = self.agc_gain.clamp(0.1, 64.0);
        }

        IqSample {
            i: corrected.i * self.agc_gain,
            q: corrected.q * self.agc_gain,
        }
    }

    fn correct_frequency(&mut self, sample: IqSample) -> IqSample {
        let phase = self.carrier_phase;
        let sin = phase.sin();
        let cos = phase.cos();
        self.carrier_phase += self.carrier_increment + self.nco_increment + self.nco_fll_offset;
        if self.carrier_phase >= TAU || self.carrier_phase <= -TAU {
            self.carrier_phase %= TAU;
        }

        IqSample {
            i: sample.i * cos - sample.q * sin,
            q: sample.i * sin + sample.q * cos,
        }
    }

    fn update_carrier_tracker(&mut self, current: IqSample) {
        if self.loop_alpha == 0.0 && self.loop_beta == 0.0 && self.freq_loop_alpha == 0.0 {
            return;
        }

        let magnitude_squared = current.i.mul_add(current.i, current.q * current.q);
        if magnitude_squared <= f32::EPSILON {
            return;
        }
        let scale = magnitude_squared.sqrt();
        let i = current.i / scale;
        let q = current.q / scale;

        // Kay-style FLL on squared symbols: imag(s²_n * conj(s²_{n-1}))
        // collapses BPSK 180° flips so the discriminator sees only the
        // residual carrier rotation, giving wide pull-in over Doppler before
        // the phase-locked Costas tracker converges. The FLL state is kept
        // separate from the Costas integrator (`nco_increment`) and decays
        // via a leaky integrator: the discriminator on noisy squared symbols
        // is shared with Costas through the NCO sum but does not perturb the
        // Costas beta accumulator, and the leak keeps the FLL from sticking
        // at the clamp during fades.
        if self.freq_loop_alpha > 0.0 {
            let s2 = IqSample {
                i: i * i - q * q,
                q: 2.0 * i * q,
            };
            if let Some(prev) = self.fll_prev_squared {
                let mag_curr_sq = s2.i.mul_add(s2.i, s2.q * s2.q);
                let mag_prev_sq = prev.i.mul_add(prev.i, prev.q * prev.q);
                let denom = (mag_curr_sq * mag_prev_sq).sqrt();
                if denom > f32::EPSILON {
                    let cross = s2.q * prev.i - s2.i * prev.q;
                    let fll_error = (cross / denom).clamp(-1.0, 1.0);
                    let leak = self.freq_loop_alpha;
                    self.nco_fll_offset =
                        (1.0 - leak) * self.nco_fll_offset + self.freq_loop_alpha * fll_error;
                    self.nco_fll_offset = self
                        .nco_fll_offset
                        .clamp(-self.nco_limit_radians, self.nco_limit_radians);
                }
            }
            self.fll_prev_squared = Some(s2);
        }

        let error = match self.config.mode {
            LinearMode::Bpsk | LinearMode::Dbpsk => i * q,
            LinearMode::Qpsk | LinearMode::Oqpsk => {
                let sign_i = if i >= 0.0 { 1.0 } else { -1.0 };
                let sign_q = if q >= 0.0 { 1.0 } else { -1.0 };
                sign_i * q - sign_q * i
            }
        };
        let error = error.clamp(-1.0, 1.0);

        self.nco_increment += self.loop_beta * error;
        self.nco_increment = self
            .nco_increment
            .clamp(-self.nco_limit_radians, self.nco_limit_radians);
        self.carrier_phase += self.loop_alpha * error;
        if self.carrier_phase >= TAU || self.carrier_phase <= -TAU {
            self.carrier_phase %= TAU;
        }
    }

    fn update_timing_recovery(&mut self) {
        let len = self.symbol_samples.len();
        if len < 8 {
            return;
        }

        let quarter = (len / 4).max(1);
        let mid_center = len / 2;
        let mid_span = (quarter / 2).max(1);
        let mid_start = mid_center.saturating_sub(mid_span);
        let mid_end = (mid_center + mid_span).min(len);

        let early = average_i(&self.symbol_samples[..quarter]);
        let mid = average_i(&self.symbol_samples[mid_start..mid_end]);
        let late = average_i(&self.symbol_samples[len - quarter..]);
        let sign = if mid >= 0.0 { 1.0 } else { -1.0 };
        let error = (late - early) * sign;

        self.timing_correction += 0.002 * error;
        let limit = self.nominal_samples_per_symbol * 0.2;
        self.timing_correction = self.timing_correction.clamp(-limit, limit);
        self.samples_per_symbol = (self.nominal_samples_per_symbol + self.timing_correction).clamp(
            self.nominal_samples_per_symbol * 0.8,
            self.nominal_samples_per_symbol * 1.2,
        );
    }

    fn matched_filter(&mut self, sample: IqSample) -> IqSample {
        let Some(taps) = self.matched_taps.as_ref() else {
            return sample;
        };
        let len = taps.len();
        self.matched_buffer_i[self.matched_index] = sample.i;
        self.matched_buffer_q[self.matched_index] = sample.q;
        self.matched_index = (self.matched_index + 1) % len;

        let mut acc_i = 0.0f32;
        let mut acc_q = 0.0f32;
        #[allow(clippy::needless_range_loop)]
        for k in 0..len {
            let buf_index = (self.matched_index + k) % len;
            let coeff = taps[k];
            acc_i += self.matched_buffer_i[buf_index] * coeff;
            acc_q += self.matched_buffer_q[buf_index] * coeff;
        }
        IqSample { i: acc_i, q: acc_q }
    }

    fn peak_symbol_sample(&self) -> IqSample {
        let len = self.symbol_samples.len();
        if len == 0 {
            return IqSample::default();
        }
        self.symbol_samples[len / 2]
    }

    /// Process a block of IQ samples and emit soft-decision symbols.
    ///
    /// Each output element is a signed correlation value where positive
    /// means the channel bit was received as `0` and negative means `1`;
    /// magnitude is the per-bit reliability. The same DC blocker, AGC,
    /// matched filter, carrier tracker and timing recovery used by
    /// [`Demodulator::push_samples`] are applied here, only the slicing
    /// step differs.
    pub fn push_samples_soft(&mut self, samples: &[IqSample]) -> Vec<i8> {
        let mut soft_bits = Vec::new();

        for &sample in samples {
            let sample = self.normalize_sample(sample);
            let sample = self.apply_dc_block_and_agc(sample);
            let sample = self.correct_frequency(sample);
            let filtered = self.matched_filter(sample);
            self.i_sum += filtered.i;
            self.q_sum += filtered.q;
            self.symbol_samples.push(filtered);
            self.sample_phase += 1.0;

            if self.sample_phase >= self.samples_per_symbol {
                self.sample_phase -= self.samples_per_symbol;
                let current = if self.matched_taps.is_some() {
                    self.peak_symbol_sample()
                } else {
                    IqSample {
                        i: self.i_sum,
                        q: self.q_sum,
                    }
                };
                self.update_carrier_tracker(current);
                write_symbol_dump(current);
                soft_bits.extend_from_slice(&self.slice_symbol_soft(current));
                self.update_timing_recovery();
                self.i_sum = 0.0;
                self.q_sum = 0.0;
                self.symbol_samples.clear();
            }
        }

        soft_bits
    }
}

impl Demodulator for LinearDemodulator {
    type Sample = IqSample;

    fn push_samples(&mut self, samples: &[IqSample]) -> Vec<u8> {
        let mut bits = Vec::new();

        for &sample in samples {
            let sample = self.normalize_sample(sample);
            let sample = self.apply_dc_block_and_agc(sample);
            let sample = self.correct_frequency(sample);
            let filtered = self.matched_filter(sample);
            self.i_sum += filtered.i;
            self.q_sum += filtered.q;
            self.symbol_samples.push(filtered);
            self.sample_phase += 1.0;

            if self.sample_phase >= self.samples_per_symbol {
                self.sample_phase -= self.samples_per_symbol;
                let current = if self.matched_taps.is_some() {
                    self.peak_symbol_sample()
                } else {
                    IqSample {
                        i: self.i_sum,
                        q: self.q_sum,
                    }
                };
                self.update_carrier_tracker(current);
                write_symbol_dump(current);
                bits.extend_from_slice(&self.slice_symbol(current));
                self.update_timing_recovery();
                self.i_sum = 0.0;
                self.q_sum = 0.0;
                self.symbol_samples.clear();
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

fn validate_config(config: LinearConfig) -> Result<(), DecodeError> {
    if config.sample_rate == 0 {
        return Err(DecodeError::InvalidEncoding(
            "linear sample rate must be greater than zero".to_string(),
        ));
    }
    if config.baudrate == 0 {
        return Err(DecodeError::InvalidEncoding(
            "linear baudrate must be greater than zero".to_string(),
        ));
    }
    if !config.frequency_offset_hz.is_finite() {
        return Err(DecodeError::InvalidEncoding(
            "linear frequency offset must be finite".to_string(),
        ));
    }
    if !config.carrier_loop_bandwidth.is_finite() || config.carrier_loop_bandwidth < 0.0 {
        return Err(DecodeError::InvalidEncoding(
            "linear carrier loop bandwidth must be a non-negative finite value".to_string(),
        ));
    }
    if !config.frequency_loop_bandwidth.is_finite() || config.frequency_loop_bandwidth < 0.0 {
        return Err(DecodeError::InvalidEncoding(
            "linear frequency loop bandwidth must be a non-negative finite value".to_string(),
        ));
    }
    if !config.nco_max_offset_hz.is_finite() || config.nco_max_offset_hz < 0.0 {
        return Err(DecodeError::InvalidEncoding(
            "linear NCO max offset must be a non-negative finite value in Hz".to_string(),
        ));
    }
    if !config.matched_filter_rolloff.is_finite() || config.matched_filter_rolloff < 0.0 {
        return Err(DecodeError::InvalidEncoding(
            "linear matched filter rolloff must be a non-negative finite value".to_string(),
        ));
    }
    if config.matched_filter_rolloff > 1.0 {
        return Err(DecodeError::InvalidEncoding(
            "linear matched filter rolloff must be <= 1.0".to_string(),
        ));
    }
    Ok(())
}

fn rrc_taps(samples_per_symbol: f32, rolloff: f32, span_symbols: usize) -> Vec<f32> {
    let length = (samples_per_symbol * span_symbols as f32).round() as usize;
    let length = length.max(1) | 1;
    let center = (length - 1) as f32 / 2.0;
    let mut taps = Vec::with_capacity(length);
    let mut energy = 0.0f32;
    for k in 0..length {
        let t_in_symbols = (k as f32 - center) / samples_per_symbol;
        let value = rrc_impulse(t_in_symbols, rolloff);
        taps.push(value);
        energy += value * value;
    }
    if energy > 0.0 {
        let scale = 1.0 / energy.sqrt();
        for tap in taps.iter_mut() {
            *tap *= scale;
        }
    }
    taps
}

fn rrc_impulse(t: f32, rolloff: f32) -> f32 {
    let pi = std::f32::consts::PI;
    if t.abs() < 1.0e-6 {
        return 1.0 - rolloff + 4.0 * rolloff / pi;
    }
    if rolloff > 0.0 {
        let singular = 1.0 / (4.0 * rolloff);
        if (t.abs() - singular).abs() < 1.0e-4 {
            let term1 = (1.0 + 2.0 / pi) * (pi / (4.0 * rolloff)).sin();
            let term2 = (1.0 - 2.0 / pi) * (pi / (4.0 * rolloff)).cos();
            return rolloff / std::f32::consts::SQRT_2 * (term1 + term2);
        }
    }
    let pi_t = pi * t;
    let num = (pi_t * (1.0 - rolloff)).sin() + 4.0 * rolloff * t * (pi_t * (1.0 + rolloff)).cos();
    let denom = pi_t * (1.0 - (4.0 * rolloff * t).powi(2));
    num / denom
}

fn loop_filter_gains(bandwidth: f32) -> (f32, f32) {
    if bandwidth <= 0.0 || !bandwidth.is_finite() {
        return (0.0, 0.0);
    }
    let damping = std::f32::consts::FRAC_1_SQRT_2;
    let denom = 1.0 + 2.0 * damping * bandwidth + bandwidth * bandwidth;
    let alpha = 4.0 * damping * bandwidth / denom;
    let beta = 4.0 * bandwidth * bandwidth / denom;
    (alpha, beta)
}

fn average_i(samples: &[IqSample]) -> f32 {
    if samples.is_empty() {
        return 0.0;
    }
    let sum = samples.iter().map(|sample| sample.i).sum::<f32>();
    sum / samples.len() as f32
}

/// Project a complex symbol onto the in-phase axis with magnitude normalization.
///
/// For non-differential BPSK the channel bit is encoded by the sign of `i`,
/// but the magnitude depends on signal strength and integration gain. Returning
/// `i / |sample|` keeps the soft value in `[-1.0, 1.0]` so downstream Viterbi
/// metrics receive comparable confidences across recordings.
fn normalize_in_phase(sample: IqSample) -> f32 {
    let mag = sample.i.mul_add(sample.i, sample.q * sample.q).sqrt();
    if mag > f32::EPSILON {
        (sample.i / mag).clamp(-1.0, 1.0)
    } else {
        0.0
    }
}

/// Map a signed correlation value to a soft-decision symbol.
///
/// Positive `value` means bit `0`, negative means bit `1`; magnitude carries
/// per-bit reliability. Output is clamped to the i8 range so downstream soft
/// Viterbi metrics stay representable.
fn soft_to_i8(value: f32) -> i8 {
    const SOFT_CENTER: f32 = 64.0;
    if !value.is_finite() {
        return 0;
    }
    let scaled = (value * SOFT_CENTER).clamp(i8::MIN as f32 + 1.0, i8::MAX as f32);
    scaled.round() as i8
}

#[cfg(test)]
mod tests {
    use super::*;

    fn repeat_symbols(symbols: &[IqSample], samples_per_symbol: usize) -> Vec<IqSample> {
        let mut out = Vec::with_capacity(symbols.len() * samples_per_symbol);
        for &symbol in symbols {
            for _ in 0..samples_per_symbol {
                out.push(symbol);
            }
        }
        out
    }

    fn add_frequency_offset(
        samples: &[IqSample],
        sample_rate: u32,
        offset_hz: f32,
    ) -> Vec<IqSample> {
        let mut phase = 0.0f32;
        let increment = TAU * offset_hz / sample_rate as f32;
        let mut out = Vec::with_capacity(samples.len());

        for &sample in samples {
            let sin = phase.sin();
            let cos = phase.cos();
            out.push(IqSample {
                i: sample.i * cos - sample.q * sin,
                q: sample.i * sin + sample.q * cos,
            });
            phase += increment;
            if phase >= TAU || phase <= -TAU {
                phase %= TAU;
            }
        }

        out
    }

    #[test]
    fn recovers_bpsk_symbols() {
        let expected = [1, 0, 1, 1, 0];
        let symbols: Vec<IqSample> = expected
            .iter()
            .map(|bit| IqSample {
                i: if *bit == 1 { 1.0 } else { -1.0 },
                q: 0.0,
            })
            .collect();
        let samples = repeat_symbols(&symbols, 4);
        let config = LinearConfig::new(4_800, 1_200, LinearMode::Bpsk);
        let mut demodulator = match LinearDemodulator::new(config) {
            Ok(demodulator) => demodulator,
            Err(err) => panic!("valid config: {err}"),
        };

        let recovered = demodulator.push_samples(&samples);

        assert_eq!(recovered, expected);
    }

    #[test]
    fn recovers_bpsk_with_swapped_iq() {
        let expected = [1, 0, 1, 1, 0];
        let symbols: Vec<IqSample> = expected
            .iter()
            .map(|bit| IqSample {
                i: 0.0,
                q: if *bit == 1 { 1.0 } else { -1.0 },
            })
            .collect();
        let samples = repeat_symbols(&symbols, 4);
        let mut config = LinearConfig::new(4_800, 1_200, LinearMode::Bpsk);
        config.swap_iq = true;
        let mut demodulator = match LinearDemodulator::new(config) {
            Ok(demodulator) => demodulator,
            Err(err) => panic!("valid config: {err}"),
        };

        let recovered = demodulator.push_samples(&samples);

        assert_eq!(recovered, expected);
    }

    #[test]
    fn corrects_fixed_bpsk_frequency_offset() {
        let expected = [1, 0, 1, 1, 0, 0, 1, 0];
        let symbols: Vec<IqSample> = expected
            .iter()
            .map(|bit| IqSample {
                i: if *bit == 1 { 1.0 } else { -1.0 },
                q: 0.0,
            })
            .collect();
        let clean = repeat_symbols(&symbols, 80);
        let offset_samples = add_frequency_offset(&clean, 96_000, 100.0);
        let mut config = LinearConfig::new(96_000, 1_200, LinearMode::Bpsk);
        config.frequency_offset_hz = 100.0;
        let mut demodulator = match LinearDemodulator::new(config) {
            Ok(demodulator) => demodulator,
            Err(err) => panic!("valid config: {err}"),
        };

        let recovered = demodulator.push_samples(&offset_samples);

        assert_eq!(recovered, expected);
    }

    #[test]
    fn recovers_dbpsk_phase_changes_with_carrier_phase_offset() {
        let expected = [0, 1, 0, 1, 1, 0, 0, 1];
        let mut symbol = IqSample { i: 1.0, q: 0.0 };
        let rotation = IqSample { i: 0.0, q: 1.0 };
        let mut symbols = Vec::new();

        for bit in expected {
            if bit == 1 {
                symbol.i = -symbol.i;
                symbol.q = -symbol.q;
            }
            symbols.push(IqSample {
                i: symbol.i * rotation.i - symbol.q * rotation.q,
                q: symbol.i * rotation.q + symbol.q * rotation.i,
            });
        }

        let samples = repeat_symbols(&symbols, 8);
        let config = LinearConfig::new(9_600, 1_200, LinearMode::Dbpsk);
        let mut demodulator = match LinearDemodulator::new(config) {
            Ok(demodulator) => demodulator,
            Err(err) => panic!("valid config: {err}"),
        };

        let recovered = demodulator.push_samples(&samples);

        assert_eq!(recovered, expected);
    }

    #[test]
    fn recovers_qpsk_symbols_as_iq_bit_pairs() {
        let symbols = [
            IqSample { i: 1.0, q: 1.0 },
            IqSample { i: -1.0, q: 1.0 },
            IqSample { i: -1.0, q: -1.0 },
        ];
        let samples = repeat_symbols(&symbols, 2);
        let config = LinearConfig::new(2_400, 1_200, LinearMode::Qpsk);
        let mut demodulator = match LinearDemodulator::new(config) {
            Ok(demodulator) => demodulator,
            Err(err) => panic!("valid config: {err}"),
        };

        let recovered = demodulator.push_samples(&samples);

        assert_eq!(recovered, vec![1, 1, 0, 1, 0, 0]);
    }

    #[test]
    fn rrc_taps_have_unit_energy_and_symmetric_shape() {
        let taps = rrc_taps(8.0, 0.5, 6);
        let energy: f32 = taps.iter().map(|t| t * t).sum();
        assert!((energy - 1.0).abs() < 1.0e-4, "energy={energy}");
        let len = taps.len();
        for k in 0..len / 2 {
            let lhs = taps[k];
            let rhs = taps[len - 1 - k];
            assert!(
                (lhs - rhs).abs() < 1.0e-4,
                "tap[{k}]={lhs} tap[{}]={rhs}",
                len - 1 - k
            );
        }
    }

    #[test]
    fn matched_filter_recovers_dbpsk_with_pulse_shaping() {
        let sample_rate = 9_600u32;
        let baudrate = 1_200u32;
        let sps = (sample_rate / baudrate) as usize;
        let pulse = rrc_taps(sps as f32, 0.5, 6);

        let bits = [0u8, 1, 1, 0, 1, 0, 0, 1, 1, 1, 0, 1, 0, 0, 1, 0];
        let mut symbol = 1.0f32;
        let mut symbol_stream: Vec<f32> = Vec::new();
        for bit in bits {
            if bit == 1 {
                symbol = -symbol;
            }
            symbol_stream.push(symbol);
        }
        let upsampled_len = symbol_stream.len() * sps + pulse.len();
        let mut shaped: Vec<IqSample> = vec![IqSample::default(); upsampled_len];
        for (n, sym) in symbol_stream.iter().enumerate() {
            for (k, tap) in pulse.iter().enumerate() {
                shaped[n * sps + k].i += *sym * *tap;
            }
        }

        let mut config = LinearConfig::new(sample_rate, baudrate, LinearMode::Dbpsk);
        config.matched_filter_rolloff = 0.5;
        config.matched_filter_span_symbols = 6;
        let mut demodulator = match LinearDemodulator::new(config) {
            Ok(demodulator) => demodulator,
            Err(err) => panic!("valid config: {err}"),
        };
        let recovered = demodulator.push_samples(&shaped);
        let tail_len = 8;
        assert!(
            recovered.len() >= tail_len,
            "need at least {tail_len} recovered bits, got {}",
            recovered.len()
        );
        let recovered_tail = &recovered[recovered.len() - tail_len..];
        let expected_tail = &bits[bits.len() - tail_len..];
        let mismatches = recovered_tail
            .iter()
            .zip(expected_tail.iter())
            .filter(|(a, b)| a != b)
            .count();
        assert!(
            mismatches <= 1,
            "mismatches={mismatches} recovered_tail={recovered_tail:?} expected_tail={expected_tail:?}"
        );
    }
}
