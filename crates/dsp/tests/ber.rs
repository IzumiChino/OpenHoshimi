//! BER regression testbench.
//!
//! Synthesises a 9600 baud NRZ signal at a known Eb/N0 and runs it through
//! the FM-audio demodulator. The test pins the high-SNR error rate so that
//! a regression in `fm_audio` (phase search, timing tracker) or `cpm`
//! (TrackingInterpolator) shows up as a hard failure rather than as an
//! end-to-end frame-rate slip on a real recording.
//!
//! Two operating points:
//!
//! * Eb/N0 = 10 dB. Plain unfiltered NRZ + AWGN, no RX matched filter.
//!   The demodulator should slice within a fraction of a percent.
//! * Eb/N0 = 4 dB. The same chain with a tighter ceiling that still
//!   leaves headroom for the AWGN realisation; this tracks the noisy
//!   end of a SatNOGS pass.
//!
//! All randomness is seeded so the test is fully deterministic.

use openhoshimi_core::Demodulator;
use openhoshimi_dsp::{FmAudioConfig, FmAudioDemodulator};

const SAMPLE_RATE: u32 = 48_000;
const BAUDRATE: u32 = 9_600;
const SAMPLES_PER_SYMBOL: usize = (SAMPLE_RATE / BAUDRATE) as usize;
const PAYLOAD_BITS: usize = 8_000;

/// Deterministic xorshift64* PRNG. Avoids pulling in a `rand` dependency
/// and keeps the test reproducible across host architectures.
struct XorShift {
    state: u64,
}

impl XorShift {
    fn new(seed: u64) -> Self {
        Self { state: seed.max(1) }
    }

    fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    fn next_f32_unit(&mut self) -> f32 {
        // Open-interval (0, 1) so the log in Box-Muller stays finite.
        let bits = (self.next_u64() >> 40) as u32;
        ((bits as f32) + 0.5) / ((1u32 << 24) as f32)
    }

    /// Standard normal sample via Box-Muller.
    fn next_gaussian(&mut self) -> f32 {
        let u1 = self.next_f32_unit();
        let u2 = self.next_f32_unit();
        let mag = (-2.0 * u1.ln()).sqrt();
        mag * (std::f32::consts::TAU * u2).cos()
    }

    fn next_bit(&mut self) -> u8 {
        ((self.next_u64() >> 63) & 1) as u8
    }
}

fn generate_bits(seed: u64, count: usize) -> Vec<u8> {
    let mut rng = XorShift::new(seed);
    (0..count).map(|_| rng.next_bit()).collect()
}

fn modulate_nrz(bits: &[u8]) -> Vec<f32> {
    let mut samples = Vec::with_capacity(bits.len() * SAMPLES_PER_SYMBOL);
    for &bit in bits {
        let level: f32 = if bit == 1 { 1.0 } else { -1.0 };
        for _ in 0..SAMPLES_PER_SYMBOL {
            samples.push(level);
        }
    }
    samples
}

/// Add AWGN at the requested Eb/N0 to a unit-amplitude NRZ stream.
///
/// For a rectangular ±1 pulse of one symbol period, the energy per bit is
/// `Eb = SAMPLES_PER_SYMBOL` (in normalized sample units). Noise variance
/// per sample is therefore `N0/2 = Eb / (2 * SPS * 10^(EbN0_dB/10))`.
fn add_awgn(samples: &mut [f32], eb_n0_db: f32, seed: u64) {
    let eb = SAMPLES_PER_SYMBOL as f32;
    let snr_lin = 10f32.powf(eb_n0_db / 10.0);
    let n0 = eb / snr_lin;
    let sigma = (n0 / 2.0).sqrt();
    let mut rng = XorShift::new(seed);
    for sample in samples.iter_mut() {
        *sample += sigma * rng.next_gaussian();
    }
}

/// Slide `recovered` against `expected` and return the lowest error count
/// over a small alignment window. The startup phase search and any
/// half-symbol skew can offset the recovered stream by up to a few bits
/// relative to the source.
fn min_errors_after_alignment(expected: &[u8], recovered: &[u8]) -> (usize, usize) {
    let max_shift = 32usize.min(recovered.len().saturating_sub(1));
    let mut best_errors = usize::MAX;
    let mut best_compared = 0usize;
    for shift in 0..=max_shift {
        let compared = recovered.len() - shift;
        let compared = compared.min(expected.len());
        if compared < expected.len() / 2 {
            continue;
        }
        let mut errors = 0usize;
        for index in 0..compared {
            if (expected[index] & 1) != (recovered[shift + index] & 1) {
                errors += 1;
            }
        }
        // The slicer might be globally inverted depending on integer-sample
        // phase. Allow that without inflating error counts.
        let inverted = compared - errors;
        let errors = errors.min(inverted);
        if errors < best_errors {
            best_errors = errors;
            best_compared = compared;
        }
    }
    (best_errors, best_compared)
}

fn run_ber(eb_n0_db: f32, bit_seed: u64, noise_seed: u64) -> f32 {
    let bits = generate_bits(bit_seed, PAYLOAD_BITS);
    let mut samples = modulate_nrz(&bits);
    add_awgn(&mut samples, eb_n0_db, noise_seed);

    let config = FmAudioConfig::new(SAMPLE_RATE, BAUDRATE);
    let mut demod = FmAudioDemodulator::new(config).expect("config valid");
    let recovered = demod.push_samples(&samples);

    let (errors, compared) = min_errors_after_alignment(&bits, &recovered);
    assert!(compared > PAYLOAD_BITS / 2, "not enough recovered bits");
    errors as f32 / compared as f32
}

#[test]
fn high_snr_ber_is_negligible() {
    let ber = run_ber(10.0, 0xC0FFEE, 0xDEADBEEF);
    assert!(
        ber < 5e-3,
        "BER at Eb/N0=10dB regressed: ber={ber:.4} (expected < 5e-3)"
    );
}

#[test]
fn moderate_snr_ber_stays_bounded() {
    let ber = run_ber(4.0, 0xBADF00D, 0x1234_5678_9ABC_DEF0);
    assert!(
        ber < 5e-2,
        "BER at Eb/N0=4dB regressed: ber={ber:.4} (expected < 5e-2)"
    );
}
