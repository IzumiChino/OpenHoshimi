//! Diagnostic dumper for OpenHoshimi's CPM demodulator.
//!
//! Reads a stereo IQ WAV, optionally runs `CpmDemodulator` end-to-end, or
//! a hand-rolled bypass path (`--bypass`) that does carrier mix + atan2
//! discriminator + integrate-and-dump entirely in this binary.  The
//! bypass path lets us tell apart bugs in `CpmDemodulator` from upstream
//! issues (WAV reader, IQ ordering, carrier offset).
//!
//! Reports:
//! 1. IQ signal energy (mean |z|) so we can sanity-check that the file
//!    actually contains signal.
//! 2. The first N bits as a binary string so we can eyeball the preamble.
//! 3. ASM (Attached Sync Marker) search at Hamming distances 0..=N
//!    against `0x1ACFFC1D` (CCSDS, MSB-first), printing positions and
//!    the next 61 bytes after each match (hex + raw ASCII).

use std::path::PathBuf;
use std::process::ExitCode;

use clap::{Parser, ValueEnum};
use openhoshimi_core::{Demodulator, IoError, IqSample, IqSource};
use openhoshimi_dsp::{CpmConfig, CpmDemodulator, CpmMode};
use openhoshimi_io::WavIqSource;

const READ_CHUNK: usize = 4_096;
const ASM: u32 = 0x1ACFFC1D;
const PAYLOAD_BITS: usize = 488;

#[derive(Debug, Clone, Copy, ValueEnum)]
enum CliMode {
    /// GMSK with Gaussian receive matched filter + tracking interpolator.
    Gmsk,
    /// MSK integrate-and-dump (no Gaussian RX filter).
    Msk,
    /// FSK integrate-and-dump.
    Fsk,
}

impl From<CliMode> for CpmMode {
    fn from(value: CliMode) -> Self {
        match value {
            CliMode::Gmsk => CpmMode::Gmsk,
            CliMode::Msk => CpmMode::Msk,
            CliMode::Fsk => CpmMode::Fsk,
        }
    }
}

#[derive(Debug, Parser)]
#[command(author, version, about)]
struct Args {
    /// Stereo IQ WAV input.
    input_wav: PathBuf,
    /// Symbol rate (baud).
    #[arg(long, default_value_t = 4800)]
    baudrate: u32,
    /// Frequency offset to apply before discrimination (Hz).
    #[arg(long, default_value_t = -103564.0)]
    frequency_offset_hz: f32,
    /// Maximum duration to analyze (seconds).
    #[arg(long, default_value_t = 8.0)]
    seconds: f32,
    /// Override the integer decimation factor; 0 = auto.
    #[arg(long, default_value_t = 0)]
    decimation: u32,
    /// Maximum Hamming distance to scan against the ASM.
    #[arg(long, default_value_t = 8)]
    max_errors: u32,
    /// Maximum number of ASM matches to print per polarity.
    #[arg(long, default_value_t = 8)]
    max_matches: usize,
    /// If set, also try inverted bits.
    #[arg(long, default_value_t = true)]
    try_invert: bool,
    /// If set, also try differential decode of inverted-or-not bits.
    #[arg(long, default_value_t = true)]
    try_diff: bool,
    /// Print this many leading bits before the ASM scan.
    #[arg(long, default_value_t = 256)]
    print_bits: usize,
    /// CPM mode for the receive chain. `msk`/`fsk` skip the Gaussian
    /// receive matched filter and use integrate-and-dump instead.
    #[arg(long, value_enum, default_value_t = CliMode::Gmsk)]
    mode: CliMode,
    /// Bypass `CpmDemodulator` entirely — do carrier mix + atan2 + I&D
    /// inline, with the simplest possible chain. Use this to isolate
    /// whether bugs are in the demodulator or upstream.
    #[arg(long, default_value_t = false)]
    bypass: bool,
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(err) => {
            eprintln!("cpm_dump: {err}");
            ExitCode::from(1)
        }
    }
}

fn run() -> Result<(), String> {
    let args = Args::parse();

    let mut source = WavIqSource::open(&args.input_wav)
        .map_err(|err| format!("failed to open IQ WAV: {err}"))?;
    let sample_rate = source.sample_rate();
    let max_samples = (args.seconds * sample_rate as f32) as usize;
    let samples = read_iq(&mut source, max_samples)?;
    println!("input:        {}", source.description());
    println!(
        "samples used: {} ({:.2} s @ {} Hz)",
        samples.len(),
        samples.len() as f32 / sample_rate as f32,
        sample_rate
    );

    let (mean_mag, max_mag, dc_i, dc_q) = iq_stats(&samples);
    println!(
        "iq energy:    mean|z|={:.4} max|z|={:.4} dc=({:+.4}, {:+.4})",
        mean_mag, max_mag, dc_i, dc_q
    );

    let bits = if args.bypass {
        run_bypass(&args, sample_rate, &samples)
    } else {
        run_cpm_demod(&args, sample_rate, &samples)?
    };

    println!(
        "bits emitted: {} (~{:.0} bps)",
        bits.len(),
        bits.len() as f32 / args.seconds
    );
    println!();

    if args.print_bits > 0 && !bits.is_empty() {
        let n = args.print_bits.min(bits.len());
        println!(
            "=== leading {} bits (look for 0x55 = 01010101 preamble) ===",
            n
        );
        print_bits_grouped(&bits[..n]);
        println!();
    }

    report_polarity("normal", &bits, args.max_errors, args.max_matches);

    if args.try_invert {
        let inv: Vec<u8> = bits.iter().map(|b| b ^ 1).collect();
        report_polarity("inverted", &inv, args.max_errors, args.max_matches);
    }

    if args.try_diff {
        let diff = differential_decode(&bits);
        report_polarity("differential", &diff, args.max_errors, args.max_matches);
        if args.try_invert {
            let inv: Vec<u8> = diff.iter().map(|b| b ^ 1).collect();
            report_polarity(
                "differential+inverted",
                &inv,
                args.max_errors,
                args.max_matches,
            );
        }
    }

    Ok(())
}

fn run_cpm_demod(args: &Args, sample_rate: u32, samples: &[IqSample]) -> Result<Vec<u8>, String> {
    let cpm_mode: CpmMode = args.mode.into();
    let mut config = CpmConfig::new(sample_rate, args.baudrate, cpm_mode);
    config.modulation_index = 0.5;
    config.gaussian_bt = match cpm_mode {
        CpmMode::Gmsk | CpmMode::Gfsk => Some(0.5),
        CpmMode::Msk | CpmMode::Fsk => None,
    };
    config.frequency_offset_hz = args.frequency_offset_hz;
    config.decimation = args.decimation;
    let mut demod = CpmDemodulator::new(config)
        .map_err(|err| format!("failed to configure demodulator: {err}"))?;
    println!(
        "cpm config:   mode={:?}, freq_offset={:.1} Hz, decimation={} (effective sps={:.2})",
        cpm_mode,
        args.frequency_offset_hz,
        demod.decimation(),
        sample_rate as f32 / demod.decimation() as f32 / args.baudrate as f32,
    );
    Ok(demod.push_samples(samples))
}

/// Hand-rolled bypass path: NCO mix down by `frequency_offset_hz`, then
/// per-sample atan2 discriminator, then symbol-rate integrate-and-dump
/// with a *floating-point* symbol clock so non-integer `sps` works
/// exactly. This is the simplest possible GMSK demod and serves as a
/// reference implementation independent of `CpmDemodulator`.
fn run_bypass(args: &Args, sample_rate: u32, samples: &[IqSample]) -> Vec<u8> {
    let sps = sample_rate as f32 / args.baudrate as f32;
    println!(
        "bypass:       freq_offset={:.1} Hz, sps={:.4}, integrate-and-dump",
        args.frequency_offset_hz, sps
    );

    let increment = -std::f32::consts::TAU * args.frequency_offset_hz / sample_rate as f32;
    let mut phase = 0.0f32;
    let mut prev = IqSample { i: 0.0, q: 0.0 };
    let mut have_prev = false;
    let mut clock = 0.0f32;
    let mut accum = 0.0f32;
    let mut delta_sum = 0.0f64;
    let mut delta_sumsq = 0.0f64;
    let mut delta_n = 0u64;
    let mut symbol_sum = 0.0f64;
    let mut symbol_sumsq = 0.0f64;
    let mut symbol_max = f32::NEG_INFINITY;
    let mut symbol_min = f32::INFINITY;
    let mut bits = Vec::with_capacity((samples.len() as f32 / sps) as usize + 16);

    for &sample in samples {
        let sin = phase.sin();
        let cos = phase.cos();
        phase += increment;
        if !(-std::f32::consts::TAU..=std::f32::consts::TAU).contains(&phase) {
            phase %= std::f32::consts::TAU;
        }
        let mixed = IqSample {
            i: sample.i * cos - sample.q * sin,
            q: sample.i * sin + sample.q * cos,
        };

        let delta = if have_prev {
            let dot = prev.i * mixed.i + prev.q * mixed.q;
            let cross = prev.i * mixed.q - prev.q * mixed.i;
            cross.atan2(dot)
        } else {
            have_prev = true;
            0.0
        };
        prev = mixed;
        delta_sum += delta as f64;
        delta_sumsq += (delta as f64) * (delta as f64);
        delta_n += 1;

        accum += delta;
        clock += 1.0;
        if clock >= sps {
            clock -= sps;
            symbol_sum += accum as f64;
            symbol_sumsq += (accum as f64) * (accum as f64);
            if accum > symbol_max {
                symbol_max = accum;
            }
            if accum < symbol_min {
                symbol_min = accum;
            }
            bits.push(if accum >= 0.0 { 1 } else { 0 });
            accum = 0.0;
        }
    }

    let delta_mean = delta_sum / delta_n.max(1) as f64;
    let delta_var = (delta_sumsq / delta_n.max(1) as f64) - delta_mean * delta_mean;
    let delta_std = delta_var.max(0.0).sqrt();
    let symbol_n = bits.len().max(1) as f64;
    let symbol_mean = symbol_sum / symbol_n;
    let symbol_var = (symbol_sumsq / symbol_n) - symbol_mean * symbol_mean;
    let symbol_std = symbol_var.max(0.0).sqrt();
    println!(
        "discrim:      delta mean={:+.5} std={:.5}  symbol mean={:+.4} std={:.4} range=[{:+.4}, {:+.4}]",
      delta_mean, delta_std, symbol_mean, symbol_std, symbol_min, symbol_max
    );

    bits
}

fn iq_stats(samples: &[IqSample]) -> (f32, f32, f32, f32) {
    if samples.is_empty() {
        return (0.0, 0.0, 0.0, 0.0);
    }
    let mut sum_mag = 0.0f64;
    let mut max_mag = 0.0f32;
    let mut sum_i = 0.0f64;
    let mut sum_q = 0.0f64;
    for &s in samples {
        let mag = (s.i * s.i + s.q * s.q).sqrt();
        sum_mag += mag as f64;
        if mag > max_mag {
            max_mag = mag;
        }
        sum_i += s.i as f64;
        sum_q += s.q as f64;
    }
    let n = samples.len() as f64;
    (
        (sum_mag / n) as f32,
        max_mag,
        (sum_i / n) as f32,
        (sum_q / n) as f32,
    )
}

fn read_iq(source: &mut dyn IqSource, max_samples: usize) -> Result<Vec<IqSample>, String> {
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
    if out.is_empty() {
        return Err("input WAV produced zero IQ samples".to_string());
    }
    Ok(out)
}

fn differential_decode(bits: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(bits.len());
    let mut prev: u8 = 0;
    for &b in bits {
        out.push(b ^ prev);
        prev = b;
    }
    out
}

fn print_bits_grouped(bits: &[u8]) {
    const PER_LINE: usize = 64;
    for (line_idx, chunk) in bits.chunks(PER_LINE).enumerate() {
        print!("  [{:5}] ", line_idx * PER_LINE);
        for (i, &b) in chunk.iter().enumerate() {
            if i > 0 && i % 8 == 0 {
                print!(" ");
            }
            print!("{}", b & 1);
        }
        println!();
    }
}

fn report_polarity(label: &str, bits: &[u8], max_errors: u32, max_matches: usize) {
    if bits.len() < 32 {
        return;
    }
    let mut by_distance: Vec<Vec<usize>> = vec![Vec::new(); (max_errors + 1) as usize];
    let mut window: u32 = 0;
    for (i, &bit) in bits.iter().enumerate() {
        window = (window << 1) | (bit & 1) as u32;
        if i >= 31 {
            let d = (window ^ ASM).count_ones();
            if d <= max_errors {
                by_distance[d as usize].push(i - 31);
            }
        }
    }

    let total: usize = by_distance.iter().map(|v| v.len()).sum();
    let summary: Vec<String> = by_distance
        .iter()
        .enumerate()
        .map(|(d, v)| format!("d={}:{}", d, v.len()))
        .collect();
    println!(
        "=== polarity={} === ASM hits within {} errors: {} ({})",
        label,
        max_errors,
        total,
        summary.join(" ")
    );

    let mut printed = 0usize;
    'outer: for (d, positions) in by_distance.iter().enumerate() {
        for &pos in positions {
            if printed >= max_matches {
                break 'outer;
            }
            let payload_start = pos + 32;
            let payload_end = payload_start + PAYLOAD_BITS;
            if payload_end > bits.len() {
                continue;
            }
            let frame = pack_msb(&bits[payload_start..payload_end]);
            let ascii: String = frame
                .iter()
                .map(|&b| {
                    if (32..127).contains(&b) {
                        b as char
                    } else {
                        '.'
                    }
                })
                .collect();
            println!("  bit_pos={:6}  d={}  hex={}", pos, d, hex_string(&frame));
            println!("    ascii: {ascii}");
            printed += 1;
        }
    }
    println!();
}

fn pack_msb(bits: &[u8]) -> Vec<u8> {
    let n = bits.len() / 8;
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let mut b = 0u8;
        for k in 0..8 {
            b = (b << 1) | (bits[i * 8 + k] & 1);
        }
        out.push(b);
    }
    out
}

fn hex_string(bytes: &[u8]) -> String {
    let mut s = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        s.push_str(&format!("{:02x}", b));
    }
    s
}
