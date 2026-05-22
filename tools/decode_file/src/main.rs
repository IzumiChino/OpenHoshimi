//! Offline WAV-to-AX.25 decoding tool for OpenHoshimi.

use std::path::PathBuf;
use std::process::ExitCode;
use std::time::Duration;

use clap::Parser;
use openhoshimi_codec::{Ax25Decoder, Ax25Frame};
use openhoshimi_core::satellite::{
    load_satellite, CodecDef, DownlinkDef, FramerDef, ModemDef, SatelliteDefinition,
};
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
    let downlink = find_audio_ax25_downlink(&satellite)?;
    let mut pipeline = AudioAx25Pipeline::new(downlink, source.sample_rate())?;
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

        let frames = pipeline.push_samples(&samples[..read]);
        for mut frame in frames {
            frame_count += 1;
            frame.satellite_id = satellite_id;
            let timestamp =
                Duration::from_secs_f64(total_samples as f64 / source.sample_rate() as f64);
            let ax25 = pipeline
                .decoder
                .decode_frame(&frame)
                .map_err(|err| format!("failed to decode AX.25 frame #{frame_count:03}: {err}"))?;
            print_frame(frame_count, timestamp, &ax25, &frame.raw);
        }

        total_samples += read as u64;
    }

    if let Some(err) = pipeline.framer.last_error() {
        eprintln!("decode_file: last discarded HDLC frame: {err}");
    }

    Ok(())
}

struct AudioAx25Pipeline {
    demodulator: AfskDemodulator,
    framer: HdlcFramer,
    decoder: Ax25Decoder,
}

impl AudioAx25Pipeline {
    fn new(downlink: &DownlinkDef, sample_rate: u32) -> Result<Self, String> {
        let (mark_hz, space_hz) = afsk_tones(downlink);
        let demodulator =
            AfskDemodulator::with_tones(sample_rate, mark_hz, space_hz, downlink.baudrate)
                .map_err(|err| format!("failed to configure AFSK demodulator: {err}"))?;

        Ok(Self {
            demodulator,
            framer: HdlcFramer::new(),
            decoder: Ax25Decoder::new(),
        })
    }

    fn push_samples(&mut self, samples: &[f32]) -> Vec<openhoshimi_core::Frame> {
        let bits = self.demodulator.push_samples(samples);
        self.framer.push_bytes(&bits)
    }
}

fn find_audio_ax25_downlink(def: &SatelliteDefinition) -> Result<&DownlinkDef, String> {
    def.downlinks
        .iter()
        .find(|downlink| {
            is_afsk_modem(downlink) && is_hdlc_framer(downlink) && is_ax25_codec(downlink)
        })
        .ok_or_else(|| {
            format!(
                "no WAV-compatible AFSK/HDLC/AX.25 downlink found in {}",
                def.satellite.name
            )
        })
}

fn is_afsk_modem(downlink: &DownlinkDef) -> bool {
    match &downlink.modem {
        Some(ModemDef::Afsk { .. }) => true,
        Some(_) => false,
        None => matches_token(&downlink.modulation, &["AFSK", "BELL202", "BELL_202"]),
    }
}

fn is_hdlc_framer(downlink: &DownlinkDef) -> bool {
    match &downlink.framer {
        Some(FramerDef::Hdlc) => true,
        Some(_) => false,
        None => matches_token(&downlink.framing, &["AX25", "AX.25", "HDLC"]),
    }
}

fn is_ax25_codec(downlink: &DownlinkDef) -> bool {
    match &downlink.codec {
        Some(CodecDef::Ax25) => true,
        Some(_) => false,
        None => matches_token(&downlink.framing, &["AX25", "AX.25", "HDLC"]),
    }
}

fn afsk_tones(downlink: &DownlinkDef) -> (f32, f32) {
    match &downlink.modem {
        Some(ModemDef::Afsk { mark_hz, space_hz }) => (*mark_hz, *space_hz),
        _ => (1200.0, 2200.0),
    }
}

fn matches_token(value: &str, tokens: &[&str]) -> bool {
    tokens.iter().any(|token| value.eq_ignore_ascii_case(token))
}

fn print_frame(index: usize, timestamp: Duration, frame: &Ax25Frame, raw: &[u8]) {
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
