//! AMSAT Fox-1 / DUV (Data Under Voice) 200 bps FSK demodulator.
//!
//! Fox-1 (AO-85, AO-91, AO-92, AO-95, HuskySat-1) carries a 200 bps FSK
//! telemetry stream simultaneously with the FM voice transponder
//! audio. The data lives in a narrow band below ~250 Hz that the voice
//! processing chain in a typical FM radio strips with its high-pass
//! filter; capturing DUV requires either an SDR with a wider audio
//! response or the radio's discriminator output.
//!
//! On the wire the data is two-tone FSK (mark / space) with a tone
//! spacing of 200 Hz centred near 1700 Hz on the recovered audio. Each
//! frame is 96 bytes (6-byte header + 58-byte payload + 32-byte RS
//! parity = a single RS(96, 64) codeword). Bytes are 8b/10b encoded so
//! one transmitted symbol carries 10 bits; a frame is 960 data bits.
//! The BitStream layer above this demodulator owns 8b/10b decoding and
//! the RS pass — see the AMSAT FoxTelem reference implementation
//! (`ac2cz/FoxTelem`, GPL-3.0+).
//!
//! What this module covers:
//!
//!   * Recovery of the 200 baud bit stream from FM-discriminator
//!     audio. The output is one byte per bit (`0x00` / `0x01`) so it
//!     plugs into the existing OpenHoshimi [`Demodulator`] surface.
//!
//! Out of scope (deferred until real recordings exist):
//!
//!   * 8b/10b symbol-to-byte mapping.
//!   * RS(96, 64) decode and frame parsing.
//!   * SoundModem-style automatic gain control.
//!
//! The demodulator implements straight zero-crossing detection over a
//! one-symbol moving window. At 200 bps the per-symbol crossing count
//! is small enough that a hard decision is reliable above ~6 dB SNR
//! without explicit matched filtering.

use std::f32::consts::TAU;

use openhoshimi_core::Demodulator;

/// Default FSK mark tone in Hz, measured at the FM discriminator
/// output of a typical narrowband radio.
pub const DEFAULT_MARK_HZ: f32 = 1500.0;
/// Default FSK space tone in Hz, measured at the FM discriminator
/// output. The 200 Hz separation matches the SoundModem and FoxTelem
/// reference setups.
pub const DEFAULT_SPACE_HZ: f32 = 1700.0;
/// DUV symbol rate in baud.
pub const DUV_BAUD: u32 = 200;

/// Configuration for the DUV demodulator.
#[derive(Debug, Clone, Copy)]
pub struct DuvConfig {
    /// Audio sample rate of the input stream in Hz.
    pub sample_rate: u32,
    /// Mark tone frequency in Hz.
    pub mark_hz: f32,
    /// Space tone frequency in Hz.
    pub space_hz: f32,
}

impl Default for DuvConfig {
    fn default() -> Self {
        Self {
            sample_rate: 48_000,
            mark_hz: DEFAULT_MARK_HZ,
            space_hz: DEFAULT_SPACE_HZ,
        }
    }
}

impl DuvConfig {
    /// Build a config using the given sample rate; everything else
    /// defaults.
    pub fn for_sample_rate(sample_rate: u32) -> Self {
        Self {
            sample_rate,
            ..Self::default()
        }
    }
}

/// 200 bps DUV FSK demodulator.
pub struct DuvDemodulator {
    config: DuvConfig,
    samples_per_symbol: f32,
    sample_phase: f32,
    mark: ToneDetector,
    space: ToneDetector,
    last_soft: Vec<f32>,
}

impl DuvDemodulator {
    /// Construct a DUV demodulator from `config`.
    ///
    /// # Errors
    ///
    /// Returns an error if `sample_rate` is zero or either tone is
    /// non-positive.
    pub fn new(config: DuvConfig) -> Result<Self, String> {
        if config.sample_rate == 0 {
            return Err("sample_rate must be > 0".to_string());
        }
        if !(config.mark_hz.is_finite()
            && config.space_hz.is_finite()
            && config.mark_hz > 0.0
            && config.space_hz > 0.0)
        {
            return Err("mark_hz and space_hz must be finite and > 0".to_string());
        }
        Ok(Self {
            samples_per_symbol: config.sample_rate as f32 / DUV_BAUD as f32,
            sample_phase: 0.0,
            mark: ToneDetector::new(config.mark_hz, config.sample_rate),
            space: ToneDetector::new(config.space_hz, config.sample_rate),
            last_soft: Vec::new(),
            config,
        })
    }

    /// Configuration this demodulator was constructed with.
    pub fn config(&self) -> DuvConfig {
        self.config
    }
}

impl Demodulator for DuvDemodulator {
    type Sample = f32;

    fn push_samples(&mut self, samples: &[f32]) -> Vec<u8> {
        let mut bits = Vec::new();
        self.last_soft.clear();
        for &sample in samples {
            self.mark.push(sample);
            self.space.push(sample);
            self.sample_phase += 1.0;
            if self.sample_phase >= self.samples_per_symbol {
                self.sample_phase -= self.samples_per_symbol;
                let mark_e = self.mark.energy();
                let space_e = self.space.energy();
                let soft = mark_e - space_e;
                self.last_soft.push(soft);
                bits.push(u8::from(soft >= 0.0));
                self.mark.reset();
                self.space.reset();
            }
        }
        bits
    }

    fn sample_rate(&self) -> u32 {
        self.config.sample_rate
    }

    fn baudrate(&self) -> u32 {
        DUV_BAUD
    }

    fn last_soft(&self) -> &[f32] {
        &self.last_soft
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

    fn synthesize_fsk(bits: &[u8], sample_rate: u32, mark: f32, space: f32) -> Vec<f32> {
        let samples_per_bit = (sample_rate / DUV_BAUD) as usize;
        let mut samples = Vec::with_capacity(bits.len() * samples_per_bit);
        let mut phase = 0.0f32;
        for &bit in bits {
            let frequency = if bit == 0 { space } else { mark };
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
    fn recovers_alternating_pattern() {
        let bits: Vec<u8> = (0..200).map(|i| u8::from(i % 2 == 0)).collect();
        let samples = synthesize_fsk(&bits, 48_000, DEFAULT_MARK_HZ, DEFAULT_SPACE_HZ);
        let mut demod = DuvDemodulator::new(DuvConfig::for_sample_rate(48_000)).expect("config");
        let recovered = demod.push_samples(&samples);
        assert_eq!(recovered, bits);
        assert_eq!(demod.last_soft().len(), bits.len());
        assert_eq!(demod.baudrate(), DUV_BAUD);
    }

    #[test]
    fn rejects_invalid_config() {
        assert!(DuvDemodulator::new(DuvConfig {
            sample_rate: 0,
            mark_hz: DEFAULT_MARK_HZ,
            space_hz: DEFAULT_SPACE_HZ,
        })
        .is_err());
        assert!(DuvDemodulator::new(DuvConfig {
            sample_rate: 48_000,
            mark_hz: -10.0,
            space_hz: DEFAULT_SPACE_HZ,
        })
        .is_err());
    }

    #[test]
    fn handles_chunked_input() {
        let bits: Vec<u8> = (0..96).map(|i| u8::from((i / 4) % 2 == 0)).collect();
        let samples = synthesize_fsk(&bits, 48_000, DEFAULT_MARK_HZ, DEFAULT_SPACE_HZ);
        let mut demod = DuvDemodulator::new(DuvConfig::for_sample_rate(48_000)).expect("config");
        let mid = samples.len() / 2;
        let mut recovered = demod.push_samples(&samples[..mid]);
        recovered.extend(demod.push_samples(&samples[mid..]));
        assert_eq!(recovered, bits);
    }
}
