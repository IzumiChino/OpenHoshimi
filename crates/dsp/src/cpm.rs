//! Generic continuous-phase modulation demodulator.
//!
//! This module covers the common binary CPM family used by amateur satellite
//! downlinks: FSK, MSK, GFSK, and GMSK. The implementation is a noncoherent
//! IQ FM discriminator followed by hard symbol slicing. Carrier and timing
//! recovery are intentionally kept outside this first implementation.

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
    /// Decode differential symbol encoding after hard slicing.
    pub differential: bool,
    /// Invert hard symbol decisions.
    pub invert: bool,
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
            differential: false,
            invert: false,
        }
    }
}

/// Noncoherent IQ demodulator for FSK/MSK/GFSK/GMSK signals.
#[derive(Debug, Clone)]
pub struct CpmDemodulator {
    config: CpmConfig,
    samples_per_symbol: f32,
    sample_phase: f32,
    discriminator_sum: f32,
    last_sample: Option<IqSample>,
    previous_symbol: Option<u8>,
}

impl CpmDemodulator {
    /// Create a demodulator from a validated configuration.
    pub fn new(config: CpmConfig) -> Result<Self, DecodeError> {
        validate_config(config)?;
        Ok(Self {
            samples_per_symbol: config.sample_rate as f32 / config.baudrate as f32,
            config,
            sample_phase: 0.0,
            discriminator_sum: 0.0,
            last_sample: None,
            previous_symbol: None,
        })
    }

    /// Return the configuration used by this demodulator.
    pub fn config(&self) -> CpmConfig {
        self.config
    }

    fn hard_slice(&mut self) -> u8 {
        let mut symbol = u8::from(self.discriminator_sum >= 0.0);
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

impl Demodulator for CpmDemodulator {
    type Sample = IqSample;

    fn push_samples(&mut self, samples: &[IqSample]) -> Vec<u8> {
        let mut bits = Vec::new();

        for &sample in samples {
            if let Some(previous) = self.last_sample {
                self.discriminator_sum += phase_delta(previous, sample);
            }
            self.last_sample = Some(sample);

            self.sample_phase += 1.0;
            if self.sample_phase >= self.samples_per_symbol {
                self.sample_phase -= self.samples_per_symbol;
                bits.push(self.hard_slice());
                self.discriminator_sum = 0.0;
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

    #[test]
    fn recovers_binary_fsk_symbols() {
        let bits = [1, 1, 0, 1, 0, 0, 1, 0];
        let samples = synthesize_fsk(&bits, 48_000, 1_200);
        let config = CpmConfig::new(48_000, 1_200, CpmMode::Fsk);
        let mut demodulator = match CpmDemodulator::new(config) {
            Ok(demodulator) => demodulator,
            Err(err) => panic!("valid config: {err}"),
        };

        let recovered = demodulator.push_samples(&samples);

        assert_eq!(recovered, bits);
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
}
