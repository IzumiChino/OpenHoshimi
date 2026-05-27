//! Audio-band carrier-frequency estimation.
//!
//! Used by the SSB audio path to recover the in-audio carrier offset
//! that an SDR receiver placed inside the WAV without bothering the user
//! to type it in. The estimator runs an FFT over the first window of
//! samples, then picks the highest-magnitude bin inside
//! `[min_hz, max_hz]`. A parabolic interpolation around that bin gives
//! sub-bin accuracy so the result is good enough to drop straight into
//! the CPM IQ demodulator's complex mixer.

use crate::fft::{bin_frequency_hz, fft_in_place, hann_window, Complex};

/// Maximum FFT size used by [`estimate_audio_carrier`]. Larger windows
/// give finer bin spacing but cost more wall-clock; 32 k samples at
/// 48 kHz is ~0.7 s of audio with ~1.5 Hz bin width which is plenty for
/// SSB audio carriers.
const MAX_FFT_LEN: usize = 32_768;

/// Minimum FFT size below which estimation is refused (too few samples
/// to give a meaningful peak).
const MIN_FFT_LEN: usize = 256;

/// Estimate the dominant tone in `samples` between `min_hz` and
/// `max_hz`.
///
/// Returns `None` if `samples` is too short, the requested band is
/// degenerate, or no bin in the band has any energy. The returned
/// frequency is the parabolic-interpolated centre of the strongest bin.
///
/// # Panics
///
/// Does not panic.
pub fn estimate_audio_carrier(
    samples: &[f32],
    sample_rate: u32,
    min_hz: f32,
    max_hz: f32,
) -> Option<f32> {
    if sample_rate == 0 {
        return None;
    }
    if !min_hz.is_finite() || !max_hz.is_finite() || min_hz >= max_hz {
        return None;
    }
    let nyquist = sample_rate as f32 / 2.0;
    let lo = min_hz.max(0.0);
    let hi = max_hz.min(nyquist);
    if lo >= hi {
        return None;
    }

    let usable = samples.len().min(MAX_FFT_LEN);
    let n = largest_power_of_two_le(usable);
    if n < MIN_FFT_LEN {
        return None;
    }

    let window = hann_window(n);
    let mut buf: Vec<Complex> = (0..n)
        .map(|i| Complex::new(samples[i] * window[i], 0.0))
        .collect();
    fft_in_place(&mut buf);

    // Real input ⇒ spectrum is conjugate-symmetric; only inspect the
    // positive-frequency half (bins 1..n/2). Skip DC because we never
    // want a 0 Hz "carrier".
    let mut best_bin: Option<usize> = None;
    let mut best_power: f32 = 0.0;
    for (bin, sample) in buf.iter().enumerate().take(n / 2).skip(1) {
        let f = bin_frequency_hz(bin, n, sample_rate);
        if f < lo || f > hi {
            continue;
        }
        let p = sample.norm_sqr();
        if p > best_power {
            best_power = p;
            best_bin = Some(bin);
        }
    }

    let bin = best_bin?;
    if best_power <= 0.0 {
        return None;
    }

    // Parabolic interpolation in log-magnitude space around the peak.
    // Falls back to the bin centre at the edges of the search.
    let centre = bin_frequency_hz(bin, n, sample_rate);
    if bin == 0 || bin + 1 >= n / 2 {
        return Some(centre);
    }
    let lhs = buf[bin - 1].norm_sqr().max(f32::MIN_POSITIVE).ln();
    let mid = buf[bin].norm_sqr().max(f32::MIN_POSITIVE).ln();
    let rhs = buf[bin + 1].norm_sqr().max(f32::MIN_POSITIVE).ln();
    let denom = lhs - 2.0 * mid + rhs;
    let bin_width = sample_rate as f32 / n as f32;
    let offset = if denom.abs() < 1e-12 {
        0.0
    } else {
        0.5 * (lhs - rhs) / denom
    };
    Some(centre + offset.clamp(-1.0, 1.0) * bin_width)
}

fn largest_power_of_two_le(n: usize) -> usize {
    if n == 0 {
        return 0;
    }
    1usize << (usize::BITS as usize - 1 - n.leading_zeros() as usize)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn synth_tone(freq_hz: f32, sample_rate: u32, len: usize) -> Vec<f32> {
        (0..len)
            .map(|k| {
                let t = k as f32 / sample_rate as f32;
                (std::f32::consts::TAU * freq_hz * t).sin()
            })
            .collect()
    }

    #[test]
    fn picks_dominant_tone_within_band() {
        let sr = 48_000u32;
        let signal = synth_tone(5_000.0, sr, sr as usize);
        let est = estimate_audio_carrier(&signal, sr, 200.0, 20_000.0)
            .expect("estimator should find a peak");
        assert!((est - 5_000.0).abs() < 5.0, "expected ~5000 Hz, got {est}");
    }

    #[test]
    fn rejects_short_input() {
        let sr = 48_000u32;
        let signal = synth_tone(5_000.0, sr, 100);
        assert!(estimate_audio_carrier(&signal, sr, 200.0, 20_000.0).is_none());
    }

    #[test]
    fn rejects_degenerate_band() {
        let sr = 48_000u32;
        let signal = synth_tone(5_000.0, sr, 4096);
        assert!(estimate_audio_carrier(&signal, sr, 1_000.0, 1_000.0).is_none());
        assert!(estimate_audio_carrier(&signal, sr, 5_000.0, 1_000.0).is_none());
    }

    #[test]
    fn ignores_dc_offset() {
        let sr = 48_000u32;
        let mut signal = synth_tone(3_000.0, sr, sr as usize);
        for s in &mut signal {
            *s += 0.5;
        }
        let est = estimate_audio_carrier(&signal, sr, 200.0, 20_000.0)
            .expect("estimator should find a peak");
        assert!((est - 3_000.0).abs() < 5.0, "expected ~3000 Hz, got {est}");
    }

    #[test]
    fn restricts_search_band() {
        let sr = 48_000u32;
        // Two tones: a strong 6 kHz and a weaker 1 kHz.
        let strong: Vec<f32> = synth_tone(6_000.0, sr, sr as usize);
        let weak: Vec<f32> = synth_tone(1_000.0, sr, sr as usize);
        let signal: Vec<f32> = strong
            .iter()
            .zip(weak.iter())
            .map(|(a, b)| a + 0.3 * b)
            .collect();
        // Search only the low band — should find ~1 kHz despite the
        // 6 kHz tone being globally stronger.
        let est = estimate_audio_carrier(&signal, sr, 500.0, 2_000.0)
            .expect("estimator should find a peak");
        assert!(
            (est - 1_000.0).abs() < 5.0,
            "expected ~1000 Hz inside band, got {est}"
        );
    }
}
