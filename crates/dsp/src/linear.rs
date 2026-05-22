//! Generic linear IQ demodulator.
//!
//! This module provides a hard-decision demodulator for already-centered,
//! symbol-aligned BPSK/DBPSK/QPSK/OQPSK streams. Carrier and timing recovery
//! are separate DSP concerns and are not hidden inside this simple slicer.

use openhoshimi_core::{DecodeError, Demodulator, IqSample};

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
    /// Decode differential symbol encoding after hard slicing.
    pub differential: bool,
    /// Invert hard symbol decisions.
    pub invert: bool,
}

impl LinearConfig {
    /// Create a configuration with mode-specific defaults.
    pub fn new(sample_rate: u32, baudrate: u32, mode: LinearMode) -> Self {
        Self {
            sample_rate,
            baudrate,
            mode,
            differential: mode == LinearMode::Dbpsk,
            invert: false,
        }
    }
}

/// Hard-decision IQ demodulator for BPSK/DBPSK/QPSK/OQPSK signals.
#[derive(Debug, Clone)]
pub struct LinearDemodulator {
    config: LinearConfig,
    samples_per_symbol: f32,
    sample_phase: f32,
    i_sum: f32,
    q_sum: f32,
    previous_symbol: Option<u8>,
}

impl LinearDemodulator {
    /// Create a demodulator from a validated configuration.
    pub fn new(config: LinearConfig) -> Result<Self, DecodeError> {
        validate_config(config)?;
        Ok(Self {
            samples_per_symbol: config.sample_rate as f32 / config.baudrate as f32,
            config,
            sample_phase: 0.0,
            i_sum: 0.0,
            q_sum: 0.0,
            previous_symbol: None,
        })
    }

    /// Return the configuration used by this demodulator.
    pub fn config(&self) -> LinearConfig {
        self.config
    }

    fn slice_symbol(&mut self) -> Vec<u8> {
        match self.config.mode {
            LinearMode::Bpsk | LinearMode::Dbpsk => vec![self.slice_binary()],
            LinearMode::Qpsk | LinearMode::Oqpsk => self.slice_quadrature(),
        }
    }

    fn slice_binary(&mut self) -> u8 {
        let mut symbol = u8::from(self.i_sum >= 0.0);
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

    fn slice_quadrature(&self) -> Vec<u8> {
        let mut i_bit = u8::from(self.i_sum >= 0.0);
        let mut q_bit = u8::from(self.q_sum >= 0.0);
        if self.config.invert {
            i_bit ^= 1;
            q_bit ^= 1;
        }
        vec![i_bit, q_bit]
    }
}

impl Demodulator for LinearDemodulator {
    type Sample = IqSample;

    fn push_samples(&mut self, samples: &[IqSample]) -> Vec<u8> {
        let mut bits = Vec::new();

        for &sample in samples {
            self.i_sum += sample.i;
            self.q_sum += sample.q;
            self.sample_phase += 1.0;

            if self.sample_phase >= self.samples_per_symbol {
                self.sample_phase -= self.samples_per_symbol;
                bits.extend_from_slice(&self.slice_symbol());
                self.i_sum = 0.0;
                self.q_sum = 0.0;
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
    Ok(())
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
}
