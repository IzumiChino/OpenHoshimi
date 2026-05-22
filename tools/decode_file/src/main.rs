//! Offline WAV-to-AX.25 decoding tool for OpenHoshimi.

use std::path::PathBuf;
use std::process::ExitCode;
use std::time::Duration;

use clap::Parser;
use openhoshimi_codec::Ax25Decoder;
use openhoshimi_core::satellite::load_satellite;
use openhoshimi_core::{Demodulator, Framing, InputSource, IoError};
use openhoshimi_dsp::{AfskDemodulator, HdlcFramer};
use openhoshimi_io::WavSource;

const READ_CHUNK: usize = 4096;

#[derive(Debug, Parser)]
#[command(author, version, about)]
struct Args {
    satellite_toml: PathBuf,
    audio_wav: PathBuf,
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(err) => {
            eprintln!("decode_file: {err}");
            ExitCode::from(1)
        }
    }
}

fn run() -> Result<(), String> {
    let args = Args::parse();
    let satellite = load_satellite(&args.satellite_toml)
        .map_err(|err| format!("failed to load satellite definition: {err}"))?;
    let mut source =
        WavSource::open(&args.audio_wav).map_err(|err| format!("failed to open WAV: {err}"))?;
    let mut demodulator = AfskDemodulator::new(source.sample_rate());
    let mut framer = HdlcFramer::new();
    let decoder = Ax25Decoder::new();
    let mut samples = [0.0f32; READ_CHUNK];
    let mut total_samples = 0u64;
    let mut frame_count = 0usize;
    let satellite_id = satellite.satellite.norad_id;

    loop {
        let read = match source.read_samples(&mut samples) {
            Ok(n) => n,
            Err(IoError::EndOfStream) => break,
            Err(err) => return Err(format!("failed to read WAV samples: {err}")),
        };

        let bits = demodulator.push_samples(&samples[..read]);
        let frames = framer.push_bytes(&bits);
        for mut frame in frames {
            frame_count += 1;
            frame.satellite_id = satellite_id;
            let timestamp =
                Duration::from_secs_f64(total_samples as f64 / source.sample_rate() as f64);
            let ax25 = decoder
                .decode_frame(&frame)
                .map_err(|err| format!("failed to decode AX.25 frame #{frame_count:03}: {err}"))?;
            print_frame(frame_count, timestamp, &ax25, &frame.raw);
        }

        total_samples += read as u64;
    }

    if let Some(err) = framer.last_error() {
        eprintln!("decode_file: last discarded HDLC frame: {err}");
    }

    Ok(())
}

fn print_frame(
    index: usize,
    timestamp: Duration,
    frame: &openhoshimi_codec::Ax25Frame,
    raw: &[u8],
) {
    let source = format!("{}-{}", frame.source.call, frame.source.ssid);
    let destination = format!("{}-{}", frame.destination.call, frame.destination.ssid);
    let frame_kind = if frame.control == 0x03 && frame.pid == Some(0xf0) {
        "AX.25 UI"
    } else {
        "AX.25"
    };

    println!(
        "#{index:03}  {}  {source:<9} >  {destination:<9}  [{frame_kind}]  {} bytes",
        format_timestamp(timestamp),
        raw.len()
    );
    println!("      raw: {}", format_hex(raw));
}

fn format_timestamp(duration: Duration) -> String {
    let total_millis = duration.as_millis();
    let minutes = total_millis / 60_000;
    let seconds = (total_millis / 1_000) % 60;
    let millis = total_millis % 1_000;
    format!("{minutes:02}:{seconds:02}.{millis:03}")
}

fn format_hex(bytes: &[u8]) -> String {
    let mut out = String::new();
    for (index, byte) in bytes.iter().enumerate() {
        if index > 0 {
            out.push(' ');
        }
        out.push_str(&format!("{byte:02x}"));
    }
    out
}
