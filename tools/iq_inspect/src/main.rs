//! IQ WAV spectral and constellation inspector.
//!
//! Reads a stereo IQ WAV file and produces three diagnostics:
//! 1. Welch averaged power spectrum to locate where signal energy lives.
//! 2. Power spectrum of `s(t)^2` (collapses BPSK 180° flips, so a residual
//!    carrier appears as a single tone at `2 * f_c`) and `s(t)^4`
//!    (collapses QPSK).
//! 3. Coarse constellation stats after mixing the strongest BPSK tone to
//!    baseband and integrating over symbol periods.

use std::fmt::Write as _;
use std::ops::{Add, Mul, Sub};
use std::path::PathBuf;
use std::process::ExitCode;

use clap::Parser;
use openhoshimi_core::{IoError, IqSample, IqSource};
use openhoshimi_io::WavIqSource;

const FFT_SIZE: usize = 16_384;
const HOP: usize = FFT_SIZE / 2;
const TOP_PEAKS: usize = 8;
const READ_CHUNK: usize = 4_096;

#[derive(Debug, Parser)]
#[command(author, version, about)]
struct Args {
    /// Stereo IQ WAV input.
    input_wav: PathBuf,
    /// Symbol rate (baud) used for constellation sampling.
    #[arg(long, default_value_t = 1200)]
    baudrate: u32,
    /// Maximum duration to analyze (seconds).
    #[arg(long, default_value_t = 8.0)]
    seconds: f32,
    /// If set, walk the whole file in non-overlapping windows of this many
    /// seconds and emit per-window RMS + BPSK carrier estimate + peak margin.
    /// Useful for seeing Doppler drift and signal presence over a full pass.
    #[arg(long)]
    track: Option<f32>,
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(err) => {
            eprintln!("iq_inspect: {err}");
            ExitCode::from(1)
        }
    }
}

fn run() -> Result<(), String> {
    let args = Args::parse();
    let mut source = WavIqSource::open(&args.input_wav)
        .map_err(|err| format!("failed to open IQ WAV: {err}"))?;
    let sample_rate = source.sample_rate();

    if let Some(window_seconds) = args.track {
        return track_mode(&mut source, sample_rate, window_seconds);
    }

    let max_samples = ((args.seconds * sample_rate as f32) as usize).max(FFT_SIZE);
    let samples = read_samples(&mut source, max_samples)?;

    println!("input:        {}", source.description());
    println!(
        "samples used: {} ({:.2} s @ {} Hz)",
        samples.len(),
        samples.len() as f32 / sample_rate as f32,
        sample_rate
    );

    let stats = sample_stats(&samples);
    println!("dc:           i={:+.4e} q={:+.4e}", stats.dc_i, stats.dc_q);
    println!("rms:          {:.4e}  (peak={:.4e})", stats.rms, stats.peak);
    println!();

    let spectrum = welch_power_spectrum(&samples, sample_rate);
    print_spectrum_peaks("baseband spectrum (Welch, 50% overlap, Hann)", &spectrum);

    let squared: Vec<Complex> = samples
        .iter()
        .map(|s| {
            let c = Complex::new(s.i, s.q);
            c * c
        })
        .collect();
    let squared_iq: Vec<IqSample> = squared
        .iter()
        .map(|c| IqSample { i: c.re, q: c.im })
        .collect();
    let squared_spectrum = welch_power_spectrum(&squared_iq, sample_rate);
    print_spectrum_peaks(
        "s^2 spectrum (BPSK/DBPSK carrier residue at 2 * fc)",
        &squared_spectrum,
    );

    let quartic: Vec<IqSample> = squared
        .iter()
        .map(|c| {
            let q = *c * *c;
            IqSample { i: q.re, q: q.im }
        })
        .collect();
    let quartic_spectrum = welch_power_spectrum(&quartic, sample_rate);
    print_spectrum_peaks(
        "s^4 spectrum (QPSK carrier residue at 4 * fc)",
        &quartic_spectrum,
    );

    if let Some(carrier_hz) = strongest_bpsk_carrier_hz(&squared_spectrum) {
        println!();
        println!("strongest BPSK carrier estimate: {carrier_hz:+.1} Hz  (from s^2 peak)");
        let quadrants = constellation_quadrants(&samples, sample_rate, args.baudrate, carrier_hz);
        print_quadrants(&quadrants);
    } else {
        println!();
        println!("no BPSK carrier candidate found from s^2 spectrum");
    }

    Ok(())
}

fn track_mode(
    source: &mut dyn IqSource,
    sample_rate: u32,
    window_seconds: f32,
) -> Result<(), String> {
    if !window_seconds.is_finite() || window_seconds <= 0.0 {
        return Err(format!(
            "--track window must be positive, got {window_seconds}"
        ));
    }
    let window_samples = ((window_seconds * sample_rate as f32) as usize).max(FFT_SIZE);
    let actual_window_seconds = window_samples as f32 / sample_rate as f32;
    println!("track mode: window={actual_window_seconds:.2} s ({window_samples} samples @ {sample_rate} Hz)");
    println!(
        "  emits one line per window. carrier from s^2 peak / 2; margin = peak_dB - median_dB."
    );
    println!();
    println!(
        "  {:>9}  {:>10}  {:>11}  {:>9}  {:>9}",
        "time", "rms", "carrier_Hz", "peak_dB", "margin_dB"
    );

    let mut window_index = 0usize;
    let mut hot_windows = 0usize;
    let mut total_windows = 0usize;
    loop {
        let samples = read_window(source, window_samples)?;
        if samples.len() < FFT_SIZE {
            if !samples.is_empty() {
                println!(
                    "  (final partial window of {} samples, {:.2} s -- skipped, below FFT size)",
                    samples.len(),
                    samples.len() as f32 / sample_rate as f32
                );
            }
            break;
        }
        total_windows += 1;
        let start_seconds = window_index as f32 * actual_window_seconds;
        let stats = sample_stats(&samples);
        let squared_iq: Vec<IqSample> = samples
            .iter()
            .map(|s| {
                let c = Complex::new(s.i, s.q);
                let c2 = c * c;
                IqSample { i: c2.re, q: c2.im }
            })
            .collect();
        let squared_spectrum = welch_power_spectrum(&squared_iq, sample_rate);
        let (carrier_hz, peak_db, margin_db) = match peak_and_margin(&squared_spectrum) {
            Some(values) => values,
            None => {
                println!(
                    "  {}  rms={:9.3e}  (no s^2 peak)",
                    format_time(start_seconds),
                    stats.rms
                );
                window_index += 1;
                continue;
            }
        };
        let fc = carrier_hz * 0.5;
        if margin_db >= 12.0 {
            hot_windows += 1;
        }
        println!(
            "  {}  {:>10.3e}  {:>+11.1}  {:>+9.2}  {:>+9.2}",
            format_time(start_seconds),
            stats.rms,
            fc,
            peak_db,
            margin_db
        );
        window_index += 1;
    }
    println!();
    println!("summary: {hot_windows}/{total_windows} windows with margin >= 12 dB (signal-likely)");
    Ok(())
}

fn read_window(source: &mut dyn IqSource, max_samples: usize) -> Result<Vec<IqSample>, String> {
    let mut out = Vec::with_capacity(max_samples);
    let mut buf = [IqSample::default(); READ_CHUNK];
    while out.len() < max_samples {
        let read = match source.read_samples(&mut buf) {
            Ok(read) => read,
            Err(IoError::EndOfStream) => break,
            Err(err) => return Err(format!("failed to read IQ samples: {err}")),
        };
        if read == 0 {
            break;
        }
        let remaining = max_samples - out.len();
        out.extend_from_slice(&buf[..read.min(remaining)]);
    }
    Ok(out)
}

fn peak_and_margin(spectrum: &Spectrum) -> Option<(f32, f32, f32)> {
    let (peak_index, peak_db) = *top_peaks(spectrum, 1).first()?;
    let mut all_db: Vec<f32> = (0..spectrum.bins.len()).map(|i| spectrum.db(i)).collect();
    all_db.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let median_db = all_db[all_db.len() / 2];
    let freq = spectrum.frequency_for(peak_index);
    Some((freq, peak_db, peak_db - median_db))
}

fn format_time(seconds: f32) -> String {
    let total = seconds.max(0.0);
    let minutes = (total / 60.0) as u32;
    let secs = total - (minutes as f32) * 60.0;
    format!("{minutes:02}:{secs:05.2}")
}

fn read_samples(source: &mut dyn IqSource, max_samples: usize) -> Result<Vec<IqSample>, String> {
    let mut out = Vec::with_capacity(max_samples);
    let mut buf = [IqSample::default(); READ_CHUNK];
    while out.len() < max_samples {
        let read = match source.read_samples(&mut buf) {
            Ok(read) => read,
            Err(IoError::EndOfStream) => break,
            Err(err) => return Err(format!("failed to read IQ samples: {err}")),
        };
        if read == 0 {
            break;
        }
        let remaining = max_samples - out.len();
        out.extend_from_slice(&buf[..read.min(remaining)]);
    }
    if out.len() < FFT_SIZE {
        return Err(format!(
            "input too short: have {} samples, need at least {} for one FFT frame",
            out.len(),
            FFT_SIZE
        ));
    }
    Ok(out)
}

#[derive(Debug)]
struct SampleStats {
    dc_i: f32,
    dc_q: f32,
    rms: f32,
    peak: f32,
}

fn sample_stats(samples: &[IqSample]) -> SampleStats {
    let n = samples.len() as f32;
    let mut sum_i = 0.0f64;
    let mut sum_q = 0.0f64;
    let mut sum_power = 0.0f64;
    let mut peak = 0.0f32;
    for s in samples {
        sum_i += s.i as f64;
        sum_q += s.q as f64;
        let power = s.i.mul_add(s.i, s.q * s.q);
        sum_power += power as f64;
        if power > peak {
            peak = power;
        }
    }
    SampleStats {
        dc_i: (sum_i / n as f64) as f32,
        dc_q: (sum_q / n as f64) as f32,
        rms: ((sum_power / n as f64) as f32).sqrt(),
        peak: peak.sqrt(),
    }
}

fn welch_power_spectrum(samples: &[IqSample], sample_rate: u32) -> Spectrum {
    let mut accum = vec![0.0f32; FFT_SIZE];
    let window = hann_window(FFT_SIZE);
    let window_power: f32 = window.iter().map(|w| w * w).sum();
    let mut frames = 0usize;
    let mut start = 0usize;
    while start + FFT_SIZE <= samples.len() {
        let mut buf = vec![Complex::default(); FFT_SIZE];
        for k in 0..FFT_SIZE {
            let s = samples[start + k];
            let w = window[k];
            buf[k] = Complex::new(s.i * w, s.q * w);
        }
        fft(&mut buf);
        for (acc, c) in accum.iter_mut().zip(buf.iter()) {
            *acc += c.re * c.re + c.im * c.im;
        }
        frames += 1;
        start += HOP;
    }
    let scale = 1.0 / (frames.max(1) as f32 * window_power);
    for value in accum.iter_mut() {
        *value *= scale;
    }
    Spectrum {
        bins: accum,
        sample_rate,
    }
}

struct Spectrum {
    bins: Vec<f32>,
    sample_rate: u32,
}

impl Spectrum {
    fn frequency_for(&self, index: usize) -> f32 {
        let n = self.bins.len() as i32;
        let signed = if (index as i32) < n / 2 {
            index as i32
        } else {
            index as i32 - n
        };
        signed as f32 * self.sample_rate as f32 / n as f32
    }

    fn db(&self, index: usize) -> f32 {
        let v = self.bins[index].max(1e-30);
        10.0 * v.log10()
    }
}

fn print_spectrum_peaks(title: &str, spectrum: &Spectrum) {
    let peaks = top_peaks(spectrum, TOP_PEAKS);
    let bin_hz = spectrum.sample_rate as f32 / spectrum.bins.len() as f32;
    println!("{title}");
    println!("  bins={}  bin width={bin_hz:.2} Hz", spectrum.bins.len());
    let mut text = String::new();
    let _ = writeln!(text, "  rank  freq[Hz]      power[dB]");
    for (rank, &(index, db)) in peaks.iter().enumerate() {
        let freq = spectrum.frequency_for(index);
        let _ = writeln!(text, "  {:>4}  {freq:>+10.1}  {db:>+9.2}", rank + 1);
    }
    print!("{text}");
    println!();
}

fn top_peaks(spectrum: &Spectrum, n: usize) -> Vec<(usize, f32)> {
    let len = spectrum.bins.len();
    let mut peaks: Vec<(usize, f32)> = Vec::new();
    for i in 1..len - 1 {
        let here = spectrum.bins[i];
        if here > spectrum.bins[i - 1] && here > spectrum.bins[i + 1] {
            peaks.push((i, spectrum.db(i)));
        }
    }
    peaks.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    peaks.truncate(n);
    peaks
}

fn strongest_bpsk_carrier_hz(squared_spectrum: &Spectrum) -> Option<f32> {
    let peaks = top_peaks(squared_spectrum, 1);
    let (index, _) = *peaks.first()?;
    Some(squared_spectrum.frequency_for(index) * 0.5)
}

fn constellation_quadrants(
    samples: &[IqSample],
    sample_rate: u32,
    baudrate: u32,
    carrier_hz: f32,
) -> [usize; 4] {
    let sps = sample_rate as f32 / baudrate as f32;
    let phase_inc = -std::f32::consts::TAU * carrier_hz / sample_rate as f32;
    let mut phase = 0.0f32;
    let mut accum_i = 0.0f32;
    let mut accum_q = 0.0f32;
    let mut sample_phase = 0.0f32;
    let mut quadrants = [0usize; 4];
    for s in samples {
        let cos = phase.cos();
        let sin = phase.sin();
        let mixed_i = s.i * cos - s.q * sin;
        let mixed_q = s.i * sin + s.q * cos;
        accum_i += mixed_i;
        accum_q += mixed_q;
        sample_phase += 1.0;
        phase += phase_inc;
        if phase >= std::f32::consts::TAU {
            phase -= std::f32::consts::TAU;
        } else if phase <= -std::f32::consts::TAU {
            phase += std::f32::consts::TAU;
        }
        if sample_phase >= sps {
            sample_phase -= sps;
            let idx = match (accum_i >= 0.0, accum_q >= 0.0) {
                (true, true) => 0,
                (false, true) => 1,
                (false, false) => 2,
                (true, false) => 3,
            };
            quadrants[idx] += 1;
            accum_i = 0.0;
            accum_q = 0.0;
        }
    }
    quadrants
}

fn print_quadrants(quadrants: &[usize; 4]) {
    let total: usize = quadrants.iter().sum();
    let total_f = total.max(1) as f32;
    println!(
        "constellation quadrant counts after mix-to-baseband + symbol integrate ({} symbols):",
        total
    );
    println!(
        "  +I+Q: {:>6}  ({:>5.1}%)",
        quadrants[0],
        100.0 * quadrants[0] as f32 / total_f
    );
    println!(
        "  -I+Q: {:>6}  ({:>5.1}%)",
        quadrants[1],
        100.0 * quadrants[1] as f32 / total_f
    );
    println!(
        "  -I-Q: {:>6}  ({:>5.1}%)",
        quadrants[2],
        100.0 * quadrants[2] as f32 / total_f
    );
    println!(
        "  +I-Q: {:>6}  ({:>5.1}%)",
        quadrants[3],
        100.0 * quadrants[3] as f32 / total_f
    );
    let bpsk_axis = quadrants[0] + quadrants[2];
    let cross_axis = quadrants[1] + quadrants[3];
    let ratio = bpsk_axis as f32 / total_f;
    println!(
        "  +I/-I axis = {:.1}% vs +Q/-Q axis = {:.1}%  ({})",
        100.0 * ratio,
        100.0 * cross_axis as f32 / total_f,
        if (ratio - 0.5).abs() > 0.15 {
            "BPSK-like (energy on one axis)"
        } else {
            "balanced (QPSK-like or noise)"
        }
    );
}

fn hann_window(n: usize) -> Vec<f32> {
    (0..n)
        .map(|k| {
            let theta = std::f32::consts::TAU * k as f32 / (n - 1) as f32;
            0.5 - 0.5 * theta.cos()
        })
        .collect()
}

#[derive(Clone, Copy, Default)]
struct Complex {
    re: f32,
    im: f32,
}

impl Complex {
    fn new(re: f32, im: f32) -> Self {
        Self { re, im }
    }
}

impl Add for Complex {
    type Output = Complex;
    fn add(self, other: Complex) -> Complex {
        Complex::new(self.re + other.re, self.im + other.im)
    }
}

impl Sub for Complex {
    type Output = Complex;
    fn sub(self, other: Complex) -> Complex {
        Complex::new(self.re - other.re, self.im - other.im)
    }
}

impl Mul for Complex {
    type Output = Complex;
    fn mul(self, other: Complex) -> Complex {
        Complex::new(
            self.re * other.re - self.im * other.im,
            self.re * other.im + self.im * other.re,
        )
    }
}

fn fft(buf: &mut [Complex]) {
    let n = buf.len();
    debug_assert!(n.is_power_of_two());
    let mut j = 0usize;
    for i in 1..n {
        let mut bit = n >> 1;
        while j & bit != 0 {
            j ^= bit;
            bit >>= 1;
        }
        j ^= bit;
        if i < j {
            buf.swap(i, j);
        }
    }
    let mut len = 2usize;
    while len <= n {
        let half = len / 2;
        let theta = -std::f32::consts::TAU / len as f32;
        let wlen = Complex::new(theta.cos(), theta.sin());
        let mut i = 0usize;
        while i < n {
            let mut w = Complex::new(1.0, 0.0);
            for k in 0..half {
                let u = buf[i + k];
                let t = buf[i + k + half] * w;
                buf[i + k] = u + t;
                buf[i + k + half] = u - t;
                w = w * wlen;
            }
            i += len;
        }
        len <<= 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fft_resolves_complex_tone() {
        let n = 1024usize;
        let mut buf: Vec<Complex> = (0..n)
            .map(|k| {
                let theta = std::f32::consts::TAU * 100.0 * k as f32 / n as f32;
                Complex::new(theta.cos(), theta.sin())
            })
            .collect();
        fft(&mut buf);
        let powers: Vec<f32> = buf.iter().map(|c| c.re * c.re + c.im * c.im).collect();
        let (peak_index, _) = powers
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .expect("non-empty FFT");
        assert_eq!(peak_index, 100);
    }

    #[test]
    fn squared_bpsk_tone_has_doubled_frequency() {
        let sample_rate = 9_600u32;
        let carrier_hz = 200.0f32;
        let len = FFT_SIZE * 2;
        let mut samples = Vec::with_capacity(len);
        let mut bit = 1.0f32;
        for k in 0..len {
            if k % 8 == 0 {
                bit = -bit;
            }
            let theta = std::f32::consts::TAU * carrier_hz * k as f32 / sample_rate as f32;
            samples.push(IqSample {
                i: bit * theta.cos(),
                q: bit * theta.sin(),
            });
        }
        let squared: Vec<IqSample> = samples
            .iter()
            .map(|s| {
                let c = Complex::new(s.i, s.q);
                let c2 = c * c;
                IqSample { i: c2.re, q: c2.im }
            })
            .collect();
        let spectrum = welch_power_spectrum(&squared, sample_rate);
        let bin = top_peaks(&spectrum, 1)[0].0;
        let freq = spectrum.frequency_for(bin);
        let bin_hz = sample_rate as f32 / spectrum.bins.len() as f32;
        assert!(
            (freq - 2.0 * carrier_hz).abs() < 2.0 * bin_hz,
            "expected ~{}, got {freq}",
            2.0 * carrier_hz
        );
    }
}
