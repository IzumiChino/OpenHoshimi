//! Diagnostic: open an audio recording (OGG or WAV mono `f32`), run the
//! FM-audio FSK/GMSK demodulator with user-supplied parameters, and count
//! ASM (attached sync marker) occurrences in the recovered bit stream at
//! every bit offset under all four bit polarities.
//!
//! Built for triaging IO-117 / GreenCube AX100 decode failures where the
//! framer reports "sync but CRC mismatch / uncorrectable Golay" — the
//! question this tool answers is "are the syncwords we lock onto real, or
//! is the framer false-positiving on noise?".
//!
//! The tool is intentionally agnostic about codec/framer: it only checks
//! whether a particular 32-bit pattern appears in the demod output, and
//! where. A genuine signal will produce a tight cluster of low-Hamming
//! matches at frame boundaries; a noise lock-on produces a roughly uniform
//! distribution across the recording.

use std::path::PathBuf;
use std::process::ExitCode;

use clap::{Parser, ValueEnum};
use openhoshimi_core::{Demodulator, InputSource, IoError};
use openhoshimi_dsp::{FmAudioConfig, FmAudioDemodulator};
use openhoshimi_io::{OggSource, WavSource};

const READ_CHUNK: usize = 8_192;

#[derive(Debug, Clone, Copy, ValueEnum)]
enum CliMode {
    /// Plain FSK with integrate-and-dump matched filter (no Gaussian RX).
    Fsk,
    /// GMSK with Gaussian receive filter + tracking interpolator.
    Gmsk,
}

#[derive(Debug, Parser)]
#[command(author, version, about)]
struct Args {
    /// Audio file (OGG/Vorbis or mono WAV with `f32`-convertible samples).
    input: PathBuf,
    /// Symbol rate (baud).
    #[arg(long, default_value_t = 1200)]
    baudrate: u32,
    /// Demod mode.
    #[arg(long, value_enum, default_value_t = CliMode::Fsk)]
    mode: CliMode,
    /// Gaussian BT (only used when `--mode gmsk`).
    #[arg(long, default_value_t = 0.5)]
    gaussian_bt: f32,
    /// ASM / sync word, MSB-first 32-bit hex. Defaults to GOMspace AX100
    /// (`0x930B51DE`).
    #[arg(long, default_value = "0x930B51DE")]
    asm: String,
    /// Maximum Hamming distance to count.
    #[arg(long, default_value_t = 6)]
    max_errors: u32,
    /// Print the first N positions per polarity at distance <= report_max.
    #[arg(long, default_value_t = 16)]
    max_print: usize,
    /// Cap analysis to this many seconds (0 = whole file).
    #[arg(long, default_value_t = 0.0)]
    seconds: f32,
    /// Skip this many seconds at the start of the file before demod.
    #[arg(long, default_value_t = 0.0)]
    start_seconds: f32,
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(err) => {
            eprintln!("syncword_scan: {err}");
            ExitCode::from(1)
        }
    }
}

fn run() -> Result<(), String> {
    let args = Args::parse();
    let asm = parse_hex_u32(&args.asm)?;

    let mut source = open_source(&args.input)?;
    let sample_rate = source.sample_rate();
    println!("input:        {}", source.description());
    println!("sample_rate:  {} Hz", sample_rate);
    println!("asm:          0x{:08X} (MSB-first, 32 bits)", asm);

    let max_samples = if args.seconds > 0.0 {
        (args.seconds * sample_rate as f32) as usize
    } else {
        usize::MAX
    };

    let skip_samples = (args.start_seconds.max(0.0) * sample_rate as f32) as usize;
    if skip_samples > 0 {
        let mut skipped = 0usize;
        let mut buf = [0.0f32; READ_CHUNK];
        while skipped < skip_samples {
            let take = (skip_samples - skipped).min(buf.len());
            let read = match source.read_samples(&mut buf[..take]) {
                Ok(n) => n,
                Err(IoError::EndOfStream) => 0,
                Err(err) => return Err(format!("read error: {err}")),
            };
            if read == 0 {
                return Err(format!("only skipped {skipped} / {skip_samples} samples"));
            }
            skipped += read;
        }
    }

    let mut config = match args.mode {
        CliMode::Fsk => FmAudioConfig::new(sample_rate, args.baudrate),
        CliMode::Gmsk => FmAudioConfig::gmsk(sample_rate, args.baudrate, args.gaussian_bt),
    };
    config.differential = false;
    config.invert = false;
    let mut demod = FmAudioDemodulator::new(config)
        .map_err(|err| format!("failed to configure FM-audio demod: {err}"))?;

    let mut all_bits: Vec<u8> = Vec::new();
    let mut buf = [0.0f32; READ_CHUNK];
    let mut total_read = 0usize;
    loop {
        let read = match source.read_samples(&mut buf) {
            Ok(n) => n,
            Err(IoError::EndOfStream) => 0,
            Err(err) => return Err(format!("read error: {err}")),
        };
        if read == 0 {
            break;
        }
        let take = read.min(max_samples - total_read);
        if take == 0 {
            break;
        }
        all_bits.extend(demod.push_samples(&buf[..take]));
        total_read += take;
        if total_read >= max_samples {
            break;
        }
    }

    println!(
        "samples used: {} ({:.2} s)",
        total_read,
        total_read as f32 / sample_rate as f32
    );
    println!(
        "bits emitted: {} (~{:.1} bps actual)",
        all_bits.len(),
        all_bits.len() as f32 * sample_rate as f32 / total_read.max(1) as f32
    );
    println!();

    let polarities: [(&str, Vec<u8>); 4] = [
        ("normal", all_bits.clone()),
        ("inverted", all_bits.iter().map(|b| b ^ 1).collect()),
        ("differential", differential_decode(&all_bits)),
        ("differential+inverted", {
            let d = differential_decode(&all_bits);
            d.iter().map(|b| b ^ 1).collect()
        }),
    ];

    for (label, bits) in &polarities {
        report(
            label,
            bits,
            asm,
            args.max_errors,
            args.max_print,
            sample_rate,
            args.baudrate,
        );
    }

    Ok(())
}

fn report(
    label: &str,
    bits: &[u8],
    asm: u32,
    max_errors: u32,
    max_print: usize,
    sample_rate: u32,
    baudrate: u32,
) {
    if bits.len() < 32 {
        println!("polarity={label} too few bits");
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
        "polarity={label:22} hits<={}: total={} [{}]",
        max_errors,
        total,
        summary.join(" ")
    );

    let bit_period_s = 1.0 / baudrate as f32;
    let mut printed = 0usize;
    'outer: for (d, positions) in by_distance.iter().enumerate() {
        for &pos in positions {
            if printed >= max_print {
                break 'outer;
            }
            let approx_t = pos as f32 * bit_period_s;
            println!(
                "  d={d} bit_pos={pos:>9}  ~ t={approx_t:>7.2} s  (audio sample ~{})",
                (approx_t * sample_rate as f32) as u64
            );
            printed += 1;
        }
    }
    println!();
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

fn parse_hex_u32(s: &str) -> Result<u32, String> {
    let cleaned = s.trim().trim_start_matches("0x").trim_start_matches("0X");
    u32::from_str_radix(cleaned, 16).map_err(|e| format!("bad hex {s:?}: {e}"))
}

fn open_source(path: &PathBuf) -> Result<Box<dyn InputSource>, String> {
    let ext = path
        .extension()
        .and_then(|s| s.to_str())
        .map(|s| s.to_ascii_lowercase())
        .unwrap_or_default();
    match ext.as_str() {
        "ogg" | "oga" => OggSource::open(path)
            .map(|s| Box::new(s) as Box<dyn InputSource>)
            .map_err(|err| format!("failed to open OGG: {err}")),
        "wav" => WavSource::open(path)
            .map(|s| Box::new(s) as Box<dyn InputSource>)
            .map_err(|err| format!("failed to open WAV: {err}")),
        other => Err(format!(
            "unsupported audio file extension {other:?} (expected ogg / wav)"
        )),
    }
}
