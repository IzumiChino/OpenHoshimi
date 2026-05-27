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
use crc::{Crc, CRC_16_IBM_SDLC};
use openhoshimi_core::{Demodulator, IoError, IqSample, IqSource};
use openhoshimi_dsp::{CpmConfig, CpmDemodulator, CpmMode};
use openhoshimi_io::WavIqSource;

const AX25_FCS: Crc<u16> = Crc::<u16>::new(&CRC_16_IBM_SDLC);
const HDLC_FLAG: u8 = 0x7e;

const READ_CHUNK: usize = 4_096;
const DEFAULT_ASM: u32 = 0x1ACFFC1D;
const PAYLOAD_BITS: usize = 488;

#[derive(Debug, Clone, Copy, ValueEnum)]
enum CliMode {
    /// GMSK with Gaussian receive matched filter + tracking interpolator.
    Gmsk,
    /// GFSK with Gaussian shaping but arbitrary modulation index.
    Gfsk,
    /// MSK integrate-and-dump (no Gaussian RX filter).
    Msk,
    /// FSK integrate-and-dump.
    Fsk,
}

impl From<CliMode> for CpmMode {
    fn from(value: CliMode) -> Self {
        match value {
            CliMode::Gmsk => CpmMode::Gmsk,
            CliMode::Gfsk => CpmMode::Gfsk,
            CliMode::Msk => CpmMode::Msk,
            CliMode::Fsk => CpmMode::Fsk,
        }
    }
}

/// Selects the scrambler used in the inverse-TX-chain reference
/// construction for `--g3ruh-ref-hex`.
#[derive(Debug, Clone, Copy, ValueEnum, PartialEq, Eq)]
enum CliScrambler {
    /// Amateur G3RUH self-syncing scrambler (x^17 + x^12 + 1).
    G3ruh,
    /// CC11xx-style PN9 byte whitening (x^9 + x^5 + 1, init 0x1FF),
    /// applied to payload bytes BEFORE HDLC bit-stuff + flag wrap.
    Pn9,
    /// No scrambler — raw HDLC bits go straight to (optional) NRZI.
    None,
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
    /// Skip this many seconds at the start of the file before analysis.
    /// Useful when a `iq_inspect --track` scan localized a burst inside a
    /// longer file and we want to demod only the burst window.
    #[arg(long, default_value_t = 0.0)]
    start_seconds: f32,
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
    /// Reference payload bytes (whitespace-tolerant hex). When set, runs
    /// the inverse TX chain (per-`--scrambler`/`--no-nrzi` choice) and
    /// reports the minimum Hamming distance between the resulting expected
    /// bit pattern and the demodulator output across all four polarities.
    /// A polarity with min Hamming far below 50% indicates the demod
    /// itself is correct and the post-demod transformation chain is what
    /// needs adjusting.
    #[arg(long)]
    g3ruh_ref_hex: Option<String>,
    /// Scrambler used in the inverse TX chain.
    #[arg(long, value_enum, default_value_t = CliScrambler::G3ruh)]
    scrambler: CliScrambler,
    /// If set, skip the NRZI encode in the inverse TX chain. Required
    /// for CC11xx-style links where the radio drives raw NRZ bits.
    #[arg(long, default_value_t = false)]
    no_nrzi: bool,
    /// Override the 32-bit ASM (sync word) used by the polarity scan,
    /// MSB-first. Default is CCSDS `0x1ACFFC1D`. CC11xx-style links
    /// often use vendor-specific sync words like `0x930B51DE`.
    #[arg(long)]
    asm: Option<String>,
    /// Modulation index `h` for the CPM demodulator. Standard GMSK uses
    /// h = 0.5; CC11xx 2-GFSK links can use much higher values (e.g. h
    /// ≈ 4 with 20+ kHz deviation at 9k6 baud). Mismatch makes the
    /// integrate-and-dump phase wrap and produces noise-level bits.
    #[arg(long, default_value_t = 0.5)]
    modulation_index: f32,
    /// Gaussian BT for the GMSK/GFSK receive matched filter.
    #[arg(long, default_value_t = 0.5)]
    gaussian_bt: f32,
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

    let asm = match args.asm.as_deref() {
        None => DEFAULT_ASM,
        Some(s) => {
            let cleaned = s.trim().trim_start_matches("0x").trim_start_matches("0X");
            u32::from_str_radix(cleaned, 16)
                .map_err(|e| format!("bad --asm hex value '{s}': {e}"))?
        }
    };
    println!("asm:          0x{:08X} (MSB-first)", asm);

    let mut source = WavIqSource::open(&args.input_wav)
        .map_err(|err| format!("failed to open IQ WAV: {err}"))?;
    let sample_rate = source.sample_rate();
    let skip_samples = (args.start_seconds.max(0.0) * sample_rate as f32) as usize;
    if skip_samples > 0 && read_iq(&mut source, skip_samples).is_err() {
        return Err(format!(
            "failed to skip leading {:.3} s ({} samples)",
            args.start_seconds, skip_samples
        ));
    }
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

    report_polarity("normal", &bits, args.max_errors, args.max_matches, asm);

    if args.try_invert {
        let inv: Vec<u8> = bits.iter().map(|b| b ^ 1).collect();
        report_polarity("inverted", &inv, args.max_errors, args.max_matches, asm);
    }

    if args.try_diff {
        let diff = differential_decode(&bits);
        report_polarity(
            "differential",
            &diff,
            args.max_errors,
            args.max_matches,
            asm,
        );
        if args.try_invert {
            let inv: Vec<u8> = diff.iter().map(|b| b ^ 1).collect();
            report_polarity(
                "differential+inverted",
                &inv,
                args.max_errors,
                args.max_matches,
                asm,
            );
        }
    }

    if let Some(hex) = args.g3ruh_ref_hex.as_ref() {
        let payload = parse_hex_bytes(hex)?;
        run_g3ruh_ref_search(&bits, &payload, args.scrambler, !args.no_nrzi);
    }

    Ok(())
}

/// Parse whitespace-tolerant hex bytes (e.g. `"84 8A 82 86"`).
fn parse_hex_bytes(s: &str) -> Result<Vec<u8>, String> {
    let cleaned: String = s.chars().filter(|c| !c.is_whitespace()).collect();
    if cleaned.len() % 2 != 0 {
        return Err(format!(
            "hex string must have an even number of nibbles, got {}",
            cleaned.len()
        ));
    }
    let mut out = Vec::with_capacity(cleaned.len() / 2);
    for i in (0..cleaned.len()).step_by(2) {
        let byte = u8::from_str_radix(&cleaned[i..i + 2], 16)
            .map_err(|e| format!("bad hex at offset {}: {}", i, e))?;
        out.push(byte);
    }
    Ok(out)
}

/// Build the expected pre-demod bit pattern from a payload by running the
/// full TX chain in reverse — payload bytes → optional PN9 whiten → CRC
/// append → HDLC bit-stuff + 0x7e flag wrap → optional G3RUH scramble →
/// optional NRZI encode — then search the demod bit dump under all four
/// polarities for the position with the minimum Hamming distance to that
/// pattern.
fn run_g3ruh_ref_search(bits: &[u8], payload: &[u8], scrambler: CliScrambler, nrzi: bool) {
    let pattern = build_expected_bits(payload, scrambler, nrzi);
    println!(
        "=== g3ruh ref search === payload={} bytes, scrambler={:?}, nrzi={}, expected pattern={} bits",
        payload.len(),
        scrambler,
        nrzi,
        pattern.len()
    );

    let polarities: [(&str, Vec<u8>); 4] = [
        ("normal", bits.to_vec()),
        ("inverted", bits.iter().map(|b| b ^ 1).collect()),
        ("differential", differential_decode(bits)),
        ("differential+inverted", {
            let d = differential_decode(bits);
            d.iter().map(|b| b ^ 1).collect()
        }),
    ];

    for (label, stream) in &polarities {
        if stream.len() < pattern.len() {
            println!("  polarity={label} too short ({} bits)", stream.len());
            continue;
        }
        let (pos, dist) = min_hamming(stream, &pattern);
        let total = pattern.len();
        let pct = 100.0 * dist as f32 / total as f32;
        println!(
            "  polarity={:22} best_pos={:6}  min_hamming={:4} / {} ({:.1}%)",
            label, pos, dist, total, pct
        );
    }
    println!();
}

/// Inverse TX chain. PN9 whitening (when selected) is applied to payload
/// bytes BEFORE CRC append + HDLC framing, matching CC11xx hardware order.
/// G3RUH scrambling (when selected) runs on the bit-stuffed HDLC stream
/// AFTER framing.
fn build_expected_bits(payload: &[u8], scrambler: CliScrambler, nrzi: bool) -> Vec<u8> {
    let mut frame_bytes = payload.to_vec();
    if scrambler == CliScrambler::Pn9 {
        pn9_whiten_inplace(&mut frame_bytes);
    }
    let mut framed = frame_bytes.clone();
    framed.extend_from_slice(&AX25_FCS.checksum(&frame_bytes).to_le_bytes());

    let mut hdlc_bits: Vec<u8> = Vec::new();
    for b in bits_lsb_first(HDLC_FLAG) {
        hdlc_bits.push(b);
    }
    let mut ones = 0usize;
    for byte in &framed {
        for bit in bits_lsb_first(*byte) {
            hdlc_bits.push(bit);
            if bit == 1 {
                ones += 1;
                if ones == 5 {
                    hdlc_bits.push(0);
                    ones = 0;
                }
            } else {
                ones = 0;
            }
        }
    }
    for b in bits_lsb_first(HDLC_FLAG) {
        hdlc_bits.push(b);
    }

    let after_scramble = match scrambler {
        CliScrambler::G3ruh => g3ruh_scramble(&hdlc_bits),
        CliScrambler::Pn9 | CliScrambler::None => hdlc_bits,
    };
    if nrzi {
        nrzi_encode(&after_scramble)
    } else {
        after_scramble
    }
}

/// CC11xx PN9 byte whitener (x^9 + x^5 + 1, init 0x1FF). Mirrors
/// `crates/codec/src/pn9.rs::Pn9Whitener` — duplicated here because
/// `cpm_dump` does not depend on `openhoshimi_codec`.
fn pn9_whiten_inplace(data: &mut [u8]) {
    let mut state: u16 = 0x1ff;
    for byte in data.iter_mut() {
        let mut whitening = 0u8;
        for bit in 0..8 {
            whitening |= ((state & 1) as u8) << bit;
            let feedback = (state & 1) ^ ((state >> 5) & 1);
            state = (state >> 1) | (feedback << 8);
        }
        *byte ^= whitening;
    }
}

fn bits_lsb_first(byte: u8) -> [u8; 8] {
    [
        byte & 1,
        (byte >> 1) & 1,
        (byte >> 2) & 1,
        (byte >> 3) & 1,
        (byte >> 4) & 1,
        (byte >> 5) & 1,
        (byte >> 6) & 1,
        (byte >> 7) & 1,
    ]
}

/// G3RUH scrambler: out[n] = in[n] XOR reg[11] XOR reg[16], where the
/// 17-bit shift register is fed by the OUTPUT bit (self-syncing). This
/// is the inverse of the descrambler in `crates/dsp/src/g3ruh.rs`.
fn g3ruh_scramble(bits: &[u8]) -> Vec<u8> {
    let mut reg: u32 = 0;
    let mut out = Vec::with_capacity(bits.len());
    for &b in bits {
        let tap12 = (reg >> 11) & 1;
        let tap17 = (reg >> 16) & 1;
        let scrambled = (b as u32 ^ tap12 ^ tap17) & 1;
        reg = ((reg << 1) | scrambled) & 0x1ffff;
        out.push(scrambled as u8);
    }
    out
}

/// NRZI encode: bit 0 toggles the line level, bit 1 keeps it. Initial
/// level = 0.
fn nrzi_encode(bits: &[u8]) -> Vec<u8> {
    let mut level: u8 = 0;
    let mut out = Vec::with_capacity(bits.len());
    for &b in bits {
        if b & 1 == 0 {
            level ^= 1;
        }
        out.push(level);
    }
    out
}

/// Slide `pattern` over `stream` (no wraparound) and return the position
/// and bit-count of the minimum-Hamming-distance match.
fn min_hamming(stream: &[u8], pattern: &[u8]) -> (usize, usize) {
    let max_start = stream.len() - pattern.len();
    let mut best = (0usize, usize::MAX);
    for start in 0..=max_start {
        let mut dist = 0usize;
        for i in 0..pattern.len() {
            if (stream[start + i] & 1) != (pattern[i] & 1) {
                dist += 1;
            }
        }
        if dist < best.1 {
            best = (start, dist);
            if dist == 0 {
                break;
            }
        }
    }
    best
}

fn run_cpm_demod(args: &Args, sample_rate: u32, samples: &[IqSample]) -> Result<Vec<u8>, String> {
    let cpm_mode: CpmMode = args.mode.into();
    let mut config = CpmConfig::new(sample_rate, args.baudrate, cpm_mode);
    config.modulation_index = args.modulation_index;
    config.gaussian_bt = match cpm_mode {
        CpmMode::Gmsk | CpmMode::Gfsk => Some(args.gaussian_bt),
        CpmMode::Msk | CpmMode::Fsk => None,
    };
    config.frequency_offset_hz = args.frequency_offset_hz;
    config.decimation = args.decimation;
    let mut demod = CpmDemodulator::new(config)
        .map_err(|err| format!("failed to configure demodulator: {err}"))?;
    println!(
  "cpm config:   mode={:?}, h={:.3}, bt={:.2}, freq_offset={:.1} Hz, decimation={} (effective sps={:.2})",
        cpm_mode,
  args.modulation_index,
        args.gaussian_bt,
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

fn report_polarity(label: &str, bits: &[u8], max_errors: u32, max_matches: usize, asm: u32) {
    if bits.len() < 32 {
        return;
    }
    let mut by_distance: Vec<Vec<usize>> = vec![Vec::new(); (max_errors + 1) as usize];
    let mut window: u32 = 0;
    for (i, &bit) in bits.iter().enumerate() {
        window = (window << 1) | (bit & 1) as u32;
        if i >= 31 {
            let d = (window ^ asm).count_ones();
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
