//! Symbol recovery from FM-demodulated audio.
//!
//! When a satellite signal is received through an FM receiver (hardware or
//! SDR in FM mode), the FM discriminator has already converted instantaneous
//! frequency deviation into a proportional audio amplitude. For FSK/GMSK
//! signals this audio directly represents the symbol waveform. This module
//! recovers bits from that waveform with an always-on closed-loop chain:
//!
//! 1. Matched filter: a length-`sps` boxcar for plain FSK or a Gaussian
//!    receive filter for GMSK. Runs first so the long DC blocker that
//!    follows sees the symbol-rate-shaped signal rather than the raw
//!    discriminator output.
//! 2. Long moving-average DC blocker (32 symbols of memory by default,
//!    configurable via `dc_blocker_symbols`) so the loop sees zero-mean
//!    input even when the front-end leaves a slow envelope.
//! 3. RMS AGC with a 50-symbol time constant so the timing detector sees
//!    unit-variance input regardless of burst level.
//! 4. First-order proportional Mueller-Muller timing-error detector,
//!    running continuously so it re-locks naturally on each burst. The
//!    proportional-only form is empirically a better match for the
//!    burst-oriented IO-117 / GREENCUBE traffic than a second-order PI
//!    loop or a Gardner TED with gr-satellites' default `clk_limit`,
//!    both of which over-shoot on the noisy inter-burst stretches.
//! 5. Hard slicer with optional differential decoding and inversion.
//!
//! Mirrors the real-input path of gr-satellites' `fsk_demodulator.py`
//! up to the timing-error-detector choice: matched filter
//! (`np.ones(sqfilter_len)/sqfilter_len`) -> `dc_blocker_ff(ceil(sps*32),
//! True)` -> `rms_agc_f(2e-2/sps, 1)` -> M&M (instead of
//! `symbol_sync_ff(TED_GARDNER, ...)`) -> `binary_slicer_fb`.

use openhoshimi_core::{DecodeError, Demodulator};

use crate::cpm::{BoxcarFilter, GaussianFir, TrackingInterpolator};

/// Empirical M&M proportional gain. Smaller values regress real-recording
/// frame counts (0.002 gave 51/107 on satnogs_7633827); larger values
/// trend toward jitter-induced hard-slicer errors.
const MM_TIMING_GAIN: f32 = 0.005;

/// Configuration for [`FmAudioDemodulator`].
#[derive(Debug, Clone, Copy)]
pub struct FmAudioConfig {
    /// Audio sample rate in Hz.
    pub sample_rate: u32,
    /// Symbol rate in baud.
    pub baudrate: u32,
    /// Gaussian BT product for the receive matched filter. `None` selects
    /// a length-`sps` boxcar matched filter, appropriate for plain FSK.
    pub gaussian_bt: Option<f32>,
    /// Decode differential symbol encoding after hard slicing.
    pub differential: bool,
    /// Invert hard symbol decisions.
    pub invert: bool,
    /// Length of the moving-average DC blocker, in symbols. The blocker is
    /// a high-pass filter with corner near `baudrate / dc_blocker_symbols`;
    /// a longer window lowers that corner and removes less of the signal's
    /// own low-frequency content, at the cost of tracking a slowly drifting
    /// DC offset more slowly. gr-satellites uses 32; some downlinks decode
    /// noticeably better with a longer window (see IO-117).
    pub dc_blocker_symbols: f32,
}

/// Default DC-blocker length in symbols, matching gr-satellites'
/// `dc_blocker_ff(ceil(sps*32), True)`.
pub const DEFAULT_DC_BLOCKER_SYMBOLS: f32 = 32.0;

impl FmAudioConfig {
    /// Create a configuration with conservative defaults (plain FSK, no
    /// differential, no inversion).
    pub fn new(sample_rate: u32, baudrate: u32) -> Self {
        Self {
            sample_rate,
            baudrate,
            gaussian_bt: None,
            differential: false,
            invert: false,
            dc_blocker_symbols: DEFAULT_DC_BLOCKER_SYMBOLS,
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
            dc_blocker_symbols: DEFAULT_DC_BLOCKER_SYMBOLS,
        }
    }

    /// Override the DC-blocker length, in symbols. Values are clamped to at
    /// least 2 symbols. Returns `self` for chaining.
    pub fn with_dc_blocker_symbols(mut self, symbols: f32) -> Self {
        if symbols.is_finite() && symbols >= 2.0 {
            self.dc_blocker_symbols = symbols;
        }
        self
    }
}

/// Symbol demodulator for FM-demodulated audio input.
#[derive(Debug, Clone)]
pub struct FmAudioDemodulator {
    config: FmAudioConfig,
    matched: MatchedFilter,
    dc_blocker: BoxcarFilter,
    agc: RmsAgc,
    tracker: TrackingInterpolator,
    previous_symbol: Option<u8>,
    last_soft: Vec<f32>,
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
        let matched = match config.gaussian_bt {
            Some(bt) => MatchedFilter::Gaussian(GaussianFir::new(bt, samples_per_symbol)),
            None => MatchedFilter::Boxcar(BoxcarFilter::matched(samples_per_symbol)),
        };
        // 32-symbol moving-average DC blocker placed AFTER the matched
        // filter, matching gr-satellites' connection order
        // `lowpass -> dcblock -> agc -> clock_recovery` in
        // `python/components/demodulators/fsk_demodulator.py:156-164`.
        // The matched filter has unit DC gain, so any residual offset
        // from the front-end remains and the long boxcar HPF is the
        // right thing to remove it. The window length is configurable
        // (`dc_blocker_symbols`, default 32) because some downlinks decode
        // better with a lower HPF corner (see IO-117).
        let dc_len = (samples_per_symbol * config.dc_blocker_symbols)
            .ceil()
            .max(2.0) as usize;
        let dc_blocker = BoxcarFilter::new(dc_len);
        let agc = RmsAgc::new(samples_per_symbol);
        // First-order proportional Mueller-Muller. Two attempts at a
        // Gardner TED port (with and without an upstream decimate-by-4
        // to match gr-satellites' assumed sps~10 regime) both regressed
        // real-recording frame counts on satnogs_7633827 vs the M&M
        // baseline (0/103 and 6/103 respectively, vs 65/103). The
        // GardnerTracker code is preserved in `crate::cpm::GardnerTracker`
        // for future in-the-loop tuning if a host with gr-satellites is
        // ever available; until then M&M with gain 0.005 is the empirical
        // optimum on this OGG.
        let tracker = TrackingInterpolator::new(samples_per_symbol, MM_TIMING_GAIN);

        Ok(Self {
            config,
            matched,
            dc_blocker,
            agc,
            tracker,
            previous_symbol: None,
            last_soft: Vec::new(),
        })
    }

    /// Return the configuration used by this demodulator.
    pub fn config(&self) -> FmAudioConfig {
        self.config
    }

    /// Return the soft (pre-slicer, post-invert, pre-differential) sample
    /// value for each bit emitted by the most recent
    /// [`push_samples`](Self::push_samples) call.
    pub fn last_soft(&self) -> &[f32] {
        &self.last_soft
    }

    fn hard_slice(&mut self, sample: f32) -> u8 {
        let oriented = if self.config.invert { -sample } else { sample };
        self.last_soft.push(oriented);

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
}

impl Demodulator for FmAudioDemodulator {
    type Sample = f32;

    fn push_samples(&mut self, samples: &[f32]) -> Vec<u8> {
        let mut bits = Vec::new();
        self.last_soft.clear();

        for &sample in samples {
            let mf = self.matched.push(sample);
            // Long moving-average HPF: `y = x - moving_avg(x)`.
            let avg = self.dc_blocker.push(mf);
            let dc = mf - avg;
            let agc = self.agc.push(dc);
            if let Some(symbol) = self.tracker.push(agc) {
                bits.push(self.hard_slice(symbol));
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

    fn last_soft(&self) -> &[f32] {
        &self.last_soft
    }
}

#[derive(Debug, Clone)]
enum MatchedFilter {
    Boxcar(BoxcarFilter),
    Gaussian(GaussianFir),
}

impl MatchedFilter {
    fn push(&mut self, sample: f32) -> f32 {
        match self {
            Self::Boxcar(filter) => filter.push(sample),
            Self::Gaussian(filter) => filter.push(sample),
        }
    }
}

/// Exponential RMS AGC with a 50-symbol time constant. Mirrors
/// `rms_agc_f(2e-2 / sps, 1)` from gr-satellites: the per-sample update
/// `r2 += alpha * (x*x - r2)` converges to E[x^2]; output `x / sqrt(r2)`
/// has unit RMS at steady state.
#[derive(Debug, Clone)]
struct RmsAgc {
    alpha: f32,
    mean_sq: f32,
}

impl RmsAgc {
    fn new(samples_per_symbol: f32) -> Self {
        let alpha = (2.0e-2 / samples_per_symbol).clamp(1.0e-5, 1.0);
        // Seed mean_sq at the unit-RMS reference so the first samples pass
        // through near unchanged. With mean_sq starting at 0 the first push
        // would divide by sqrt(eps) and produce a transient large enough
        // to wedge the downstream timing-loop envelope detector.
        Self {
            alpha,
            mean_sq: 1.0,
        }
    }

    fn push(&mut self, sample: f32) -> f32 {
        self.mean_sq += self.alpha * (sample * sample - self.mean_sq);
        // Mirror gr-satellites' `rms_agc_f`: divide by sqrt(mean_sq) with no
        // floor at unit RMS, so weak bursts whose matched-filter output
        // sits below unity still get amplified to the reference. The earlier
        // floor existed to gate the now-removed M&M envelope detector; the
        // always-on tracker handles inter-burst noise itself. A small
        // epsilon keeps the denominator non-zero through silence.
        let denom = self.mean_sq.max(1.0e-6).sqrt();
        sample / denom
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

    fn synthesize_fm_gmsk(bits: &[u8], sample_rate: u32, baudrate: u32, bt: f32) -> Vec<f32> {
        let sps = (sample_rate / baudrate) as usize;
        let span = 4usize;
        let pulse_len = sps * span;
        let centre = (pulse_len as f32 - 1.0) / 2.0;
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
        freq
    }

    fn best_match(expected: &[u8], recovered: &[u8], max_offset: usize) -> usize {
        let mut best = 0usize;
        for offset in 0..=max_offset {
            if recovered.len() < offset + expected.len() {
                break;
            }
            let matches = expected
                .iter()
                .zip(&recovered[offset..offset + expected.len()])
                .filter(|(a, b)| a == b)
                .count();
            if matches > best {
                best = matches;
            }
        }
        best
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

        let total = bits.len();
        let best = best_match(&bits, &recovered, 8);
        assert!(
            best * 100 >= total * 90,
            "FSK FM audio match rate too low: {best}/{total}"
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
        let freq = synthesize_fm_gmsk(&bits, 48_000, 9_600, 0.5);
        let config = FmAudioConfig::gmsk(48_000, 9_600, 0.5);
        let mut demod = FmAudioDemodulator::new(config).unwrap();
        let recovered = demod.push_samples(&freq);

        let total = bits.len();
        let best = best_match(&bits, &recovered, 16);
        assert!(
            best * 100 >= total * 85,
            "GMSK FM audio match rate too low: {best}/{total}"
        );
    }

    #[test]
    fn rejects_invalid_config() {
        let config = FmAudioConfig::new(48_000, 0);
        assert!(FmAudioDemodulator::new(config).is_err());
    }

    #[test]
    fn tracker_follows_clock_offset() {
        // Synthesize at nominal 9 600 baud, decode configured for ~200 PPM
        // higher rate. This is at the high end of what a TCXO-stabilised
        // amateur radio satellite actually exhibits in flight.
        let bits: Vec<u8> = (0..512u32)
            .map(|i| {
                let mixed = i.wrapping_mul(2_654_435_761).wrapping_add(7);
                ((mixed >> 16) & 1) as u8
            })
            .collect();
        let freq = synthesize_fm_gmsk(&bits, 48_000, 9_600, 0.5);

        let config = FmAudioConfig::gmsk(48_000, 9_602, 0.5);
        let mut demod = FmAudioDemodulator::new(config).unwrap();
        let recovered = demod.push_samples(&freq);

        let total = bits.len();
        let best = best_match(&bits, &recovered, 16);
        assert!(
            best * 100 >= total * 85,
            "M&M tracker failed to follow clock offset: {best}/{total}"
        );
    }

    #[test]
    fn tracker_locks_on_offset_signal() {
        // FSK signal whose first sample falls mid-symbol so phase 0
        // sampling would land on transitions. The closed-loop tracker
        // must still recover symbols from the offset stream.
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

        let total = bits.len();
        let best = best_match(&bits, &recovered, 8);
        assert!(
            best * 100 >= total * 90,
            "tracker failed to lock on offset signal: {best}/{total}"
        );
    }

    #[test]
    fn recovers_fsk_after_gap() {
        // Two FSK bursts separated by silence and noise. The always-on
        // closed loop must re-lock on the second burst exactly the way
        // gr-satellites' real-input chain does.
        let sps = 40u32;
        let sample_rate = 48_000u32;
        let baudrate = 1_200u32;
        let bits: Vec<u8> = (0..256u32)
            .map(|i| {
                let mixed = i.wrapping_mul(2_654_435_761).wrapping_add(13);
                ((mixed >> 19) & 1) as u8
            })
            .collect();

        // Helper: a stretch of band-limited noise so the loop has to
        // re-acquire timing on the burst.
        let mut rng_state: u32 = 0xDEAD_BEEF;
        let mut noise = |len: usize| {
            (0..len)
                .map(|_| {
                    rng_state = rng_state
                        .wrapping_mul(1_664_525)
                        .wrapping_add(1_013_904_223);
                    let v = (rng_state >> 8) as i32 as f32 / i32::MAX as f32;
                    v * 0.05
                })
                .collect::<Vec<f32>>()
        };

        let burst = synthesize_fm_fsk(&bits, sample_rate, baudrate);
        let gap_samples = (sps as usize) * 200; // 200 symbols of gap
        let mut samples = noise(gap_samples);
        samples.extend(&burst);
        samples.extend(noise(gap_samples));
        samples.extend(&burst);

        let config = FmAudioConfig::new(sample_rate, baudrate);
        let mut demod = FmAudioDemodulator::new(config).unwrap();
        let recovered = demod.push_samples(&samples);

        // Slide the expected pattern across the recovered stream and
        // require a contiguous window that matches the bit pattern at
        // >=90 %. We expect this to land twice, once per burst.
        let mut hits = 0usize;
        let total = bits.len();
        let mut idx = 0usize;
        while idx + total <= recovered.len() {
            let matches = bits
                .iter()
                .zip(&recovered[idx..idx + total])
                .filter(|(a, b)| a == b)
                .count();
            if matches * 100 >= total * 90 {
                hits += 1;
                idx += total; // skip past the matched window
            } else {
                idx += 1;
            }
        }
        assert!(
            hits >= 2,
            "expected to recover both bursts, got {hits} contiguous matches"
        );
    }
}
