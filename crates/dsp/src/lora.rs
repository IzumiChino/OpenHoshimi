//! LoRa Chirp Spread Spectrum (CSS) symbol demodulator.
//!
//! Several amateur cubesats — FOSSA-1, FossaSat-2, Norby (RS44S), and
//! a growing family of educational satellites built on Semtech SX12xx
//! / RadioLib firmware — beacon at low data rates using LoRa modulation.
//! LoRa is a proprietary CSS scheme: each symbol is a chirp that sweeps
//! linearly over `bandwidth` Hz; the chirp's starting offset within
//! that sweep encodes `2^SF` possible symbol values, where `SF` is the
//! Spreading Factor (typically 7 to 12).
//!
//! This module covers the **PHY symbol** layer: given complex baseband
//! samples spanning whole symbols, produce one integer (`0..2^SF`) per
//! symbol. The frame layer (preamble detection, Hamming(4,7) decoding,
//! whitening, interleaver, CRC) is **out of scope** for this iteration
//! and gated behind future work; FOSSA / Norby beacon framing in
//! particular requires bit-exact agreement with each satellite's
//! specific Semtech configuration register dump.
//!
//! Two pieces ship here:
//!
//! 1. [`generate_chirp`] — synthesize an upchirp / downchirp for a
//!    given (`SF`, `bandwidth`, `sample_rate`).
//! 2. [`LoraSymbolDemodulator`] — multiply a captured symbol by the
//!    conjugate downchirp and pick the FFT bin with maximum magnitude;
//!    the bin index is the symbol value.
//!
//! References:
//! - Semtech SX1276 datasheet, section 4.1 "LoRa Modem Operation".
//! - Tapparel et al, "An Open-Source LoRa Physical Layer Prototype on
//!   GNU Radio", 2020.
//! - <https://github.com/jkadbear/LoRaPHY> — reference symbol
//!   demodulator used to cross-check this module.

use std::f32::consts::PI;

use openhoshimi_core::IqSample;

/// Direction of a LoRa chirp.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChirpDirection {
    /// Linear up-sweep (frequency rises from `-bw/2` to `+bw/2`).
    Up,
    /// Linear down-sweep (frequency falls from `+bw/2` to `-bw/2`).
    Down,
}

/// Configuration of a LoRa CSS modem.
#[derive(Debug, Clone, Copy)]
pub struct LoraConfig {
    /// Spreading factor (typical range 7..=12).
    pub spreading_factor: u8,
    /// Modulation bandwidth in Hz (typical: 125_000, 250_000, 500_000).
    pub bandwidth_hz: f32,
    /// Receiver sample rate in Hz. Must be >= bandwidth_hz.
    pub sample_rate: u32,
}

impl LoraConfig {
    /// Number of complex samples in one symbol at this configuration.
    pub fn samples_per_symbol(&self) -> usize {
        // Symbol length on air = 2^SF / bw seconds. Samples = duration
        // x sample_rate. We allow non-integer oversampling ratios but
        // the demodulator wants whole-symbol windows so callers must
        // pad if necessary.
        let chips = 1u32 << self.spreading_factor;
        let oversample = self.sample_rate as f32 / self.bandwidth_hz;
        (chips as f32 * oversample).round() as usize
    }

    /// Number of symbol values 0..=N-1.
    pub fn symbol_count(&self) -> usize {
        1usize << self.spreading_factor
    }

    /// Reject obviously broken configurations.
    pub fn validate(&self) -> Result<(), String> {
        if !(6..=12).contains(&self.spreading_factor) {
            return Err(format!(
                "spreading_factor {} outside the LoRa range 6..=12",
                self.spreading_factor
            ));
        }
        if !(self.bandwidth_hz.is_finite() && self.bandwidth_hz > 0.0) {
            return Err("bandwidth_hz must be finite and > 0".into());
        }
        if self.sample_rate == 0 {
            return Err("sample_rate must be > 0".into());
        }
        if (self.sample_rate as f32) < self.bandwidth_hz {
            return Err(format!(
                "sample_rate {} below modulation bandwidth {}",
                self.sample_rate, self.bandwidth_hz
            ));
        }
        Ok(())
    }
}

/// Generate one LoRa chirp at the configured (`SF`, `BW`, `Fs`).
///
/// The chirp is centred at zero frequency. `direction` selects an up
/// or down sweep; `start_offset` lets callers tap a chirp shifted by
/// `start_offset` chips, equivalent to encoding a symbol value. For
/// detection callers want `start_offset = 0`; for transmission they
/// want `start_offset = symbol_value`.
pub fn generate_chirp(
    config: LoraConfig,
    direction: ChirpDirection,
    start_offset: u32,
) -> Vec<IqSample> {
    let samples_per_symbol = config.samples_per_symbol();
    let chips = config.symbol_count() as f32;
    let bw = config.bandwidth_hz;
    let fs = config.sample_rate as f32;
    let mut out = Vec::with_capacity(samples_per_symbol);
    let mut phase = 0.0f32;
    let direction_sign = match direction {
        ChirpDirection::Up => 1.0,
        ChirpDirection::Down => -1.0,
    };
    for n in 0..samples_per_symbol {
        // Instantaneous frequency: starts at f0 = bw * (start_offset / chips - 0.5),
        // then sweeps linearly by direction_sign * bw per symbol period.
        let t = n as f32 / fs;
        let symbol_period = chips / bw;
        let normalised = t / symbol_period;
        let f = direction_sign * (bw * normalised) + bw * (start_offset as f32 / chips - 0.5);
        let f = wrap_freq(f, bw);
        phase += 2.0 * PI * f / fs;
        if phase > PI {
            phase -= 2.0 * PI;
        } else if phase < -PI {
            phase += 2.0 * PI;
        }
        out.push(IqSample {
            i: phase.cos(),
            q: phase.sin(),
        });
    }
    out
}

fn wrap_freq(f: f32, bw: f32) -> f32 {
    let mut w = f;
    while w >= bw / 2.0 {
        w -= bw;
    }
    while w < -bw / 2.0 {
        w += bw;
    }
    w
}

/// LoRa symbol demodulator: multiply a captured symbol by a reference
/// downchirp, FFT, return the peak bin.
pub struct LoraSymbolDemodulator {
    config: LoraConfig,
    reference_downchirp: Vec<IqSample>,
}

impl LoraSymbolDemodulator {
    /// Build a demodulator for the given LoRa configuration.
    ///
    /// # Errors
    ///
    /// Returns an error if [`LoraConfig::validate`] rejects the
    /// configuration.
    pub fn new(config: LoraConfig) -> Result<Self, String> {
        config.validate()?;
        let reference_downchirp = generate_chirp(config, ChirpDirection::Down, 0);
        Ok(Self {
            config,
            reference_downchirp,
        })
    }

    /// Configuration this demodulator was built for.
    pub fn config(&self) -> LoraConfig {
        self.config
    }

    /// Demodulate exactly one symbol. `symbol` must contain at least
    /// `samples_per_symbol` complex samples; extra samples are
    /// ignored. Returns the symbol value `0..2^SF`.
    pub fn demodulate_symbol(&self, symbol: &[IqSample]) -> Option<u32> {
        let n = self.config.samples_per_symbol();
        if symbol.len() < n {
            return None;
        }
        let mut dechirped = Vec::with_capacity(n);
        for (i, s) in symbol.iter().enumerate().take(n) {
            let r = self.reference_downchirp[i];
            // Complex multiply.
            dechirped.push(IqSample {
                i: s.i * r.i - s.q * r.q,
                q: s.i * r.q + s.q * r.i,
            });
        }
        // The peak FFT bin in `dechirped` (decimated to `chips`
        // samples) is the symbol value. We use a naive DFT over only
        // the `chips` candidate bins, which is `O(chips^2)` — fine
        // for SF up to 12 (4096 candidates per symbol).
        let chips = self.config.symbol_count();
        let mut best_bin = 0u32;
        let mut best_mag = f32::NEG_INFINITY;
        let oversample = n / chips;
        // Decimate by averaging blocks of `oversample` samples.
        let mut decimated = Vec::with_capacity(chips);
        for c in 0..chips {
            let mut acc_i = 0.0;
            let mut acc_q = 0.0;
            for k in 0..oversample {
                let idx = c * oversample + k;
                if idx >= n {
                    break;
                }
                acc_i += dechirped[idx].i;
                acc_q += dechirped[idx].q;
            }
            decimated.push(IqSample {
                i: acc_i / oversample as f32,
                q: acc_q / oversample as f32,
            });
        }
        // Naive DFT.
        for k in 0..chips {
            let mut acc_i = 0.0f32;
            let mut acc_q = 0.0f32;
            for (n_idx, sample) in decimated.iter().enumerate() {
                let arg = -2.0 * PI * (k as f32) * (n_idx as f32) / chips as f32;
                let c = arg.cos();
                let s = arg.sin();
                acc_i += sample.i * c - sample.q * s;
                acc_q += sample.i * s + sample.q * c;
            }
            let mag = acc_i * acc_i + acc_q * acc_q;
            if mag > best_mag {
                best_mag = mag;
                best_bin = k as u32;
            }
        }
        Some(best_bin)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn config_validation_rejects_bad_inputs() {
        assert!(LoraConfig {
            spreading_factor: 5,
            bandwidth_hz: 125_000.0,
            sample_rate: 250_000,
        }
        .validate()
        .is_err());
        assert!(LoraConfig {
            spreading_factor: 7,
            bandwidth_hz: 0.0,
            sample_rate: 250_000,
        }
        .validate()
        .is_err());
        assert!(LoraConfig {
            spreading_factor: 7,
            bandwidth_hz: 250_000.0,
            sample_rate: 125_000,
        }
        .validate()
        .is_err());
    }

    #[test]
    fn round_trip_symbols_at_sf7() {
        let config = LoraConfig {
            spreading_factor: 7,
            bandwidth_hz: 125_000.0,
            sample_rate: 125_000,
        };
        let demod = LoraSymbolDemodulator::new(config).expect("demodulator");
        for &expected in &[0u32, 1, 17, 42, 64, 100, 127] {
            let symbol = generate_chirp(config, ChirpDirection::Up, expected);
            let decoded = demod.demodulate_symbol(&symbol).expect("decoded symbol");
            assert_eq!(decoded, expected, "round-trip failed for symbol {expected}");
        }
    }

    #[test]
    fn round_trip_symbols_at_sf9_oversampled() {
        let config = LoraConfig {
            spreading_factor: 9,
            bandwidth_hz: 125_000.0,
            sample_rate: 250_000,
        };
        let demod = LoraSymbolDemodulator::new(config).expect("demodulator");
        for &expected in &[0u32, 5, 100, 256, 511] {
            let symbol = generate_chirp(config, ChirpDirection::Up, expected);
            let decoded = demod.demodulate_symbol(&symbol).expect("decoded symbol");
            assert_eq!(
                decoded, expected,
                "SF9 oversampled round-trip failed for {expected}"
            );
        }
    }

    #[test]
    fn returns_none_for_short_input() {
        let config = LoraConfig {
            spreading_factor: 7,
            bandwidth_hz: 125_000.0,
            sample_rate: 125_000,
        };
        let demod = LoraSymbolDemodulator::new(config).unwrap();
        assert_eq!(demod.demodulate_symbol(&[]), None);
    }
}
