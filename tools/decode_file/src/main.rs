//! Offline file decoding tool for OpenHoshimi.

use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::time::Duration;

use clap::Parser;
use openhoshimi_codec::{
    build_image_reassembler, Ax100Mode, Ax25Frame, GeoscanFrame, ImageDecoder, ImageReassembler,
};
use openhoshimi_core::satellite::load_satellite;
use openhoshimi_core::satellite::ModemDef;
use openhoshimi_core::{Frame, InputSource, IoError, IqSample, IqSource};
use openhoshimi_dsp::estimate_audio_carrier;
use openhoshimi_dsp::linear::open_symbol_dump;
use openhoshimi_io::{
    detect_audio_mode_auto, open_audio_source, read_audio_prefix, read_iq_prefix,
    AudioMode as IoAudioMode, MonoIqSource, WavIqSource, WavSource,
};
use openhoshimi_runtime::pipeline::{
    estimate_cpm_iq_frequency_offset_hz, estimate_iq_frequency_offset_hz, format_call, format_hex,
    format_timestamp, frame_type_label, infer_tuning_offset_hz, is_ao40_fec_downlink,
    prepare_linear_iq_setup_scored, select_downlink, BitPipeline, DecodedFrame, InputKind,
    SoftAo40Pipeline,
};
use openhoshimi_telemetry::SchemaParser;

const READ_CHUNK: usize = 4096;

#[derive(Debug, Parser)]
#[command(author, version, about)]
struct Args {
    satellite_toml: PathBuf,
    input_wav: PathBuf,
    /// How to interpret the input WAV: `auto` (sniff the file, default),
    /// `iq` (force stereo I/Q), `fm` (FM-discriminator mono audio), or
    /// `ssb` (single-sideband mono audio with `--audio-carrier-hz`
    /// giving the signal centre frequency inside the audio band).
    #[arg(long, value_enum, default_value_t = AudioMode::Auto)]
    audio_mode: AudioMode,
    /// Audio-band carrier frequency in Hz for `--audio-mode ssb`.
    /// Ignored in any other mode.
    #[arg(long)]
    audio_carrier_hz: Option<f32>,
    /// If set, run every decoded payload through the satellite's
    /// image reassembler (when one is configured in the TOML) and
    /// write each completed image to `<DIR>/openhoshimi_image_NN.{jpg,bin}`
    /// when decoding finishes. Useful for offline regression-testing
    /// the image pipeline without launching the GUI.
    #[arg(long)]
    dump_images: Option<PathBuf>,
}

/// How the input file should be interpreted.
#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
enum AudioMode {
    /// Detect IQ vs mono audio from the file and the filename hint.
    Auto,
    /// Force stereo I/Q interpretation (errors if the file is mono).
    Iq,
    /// Force FM-discriminator mono audio (instantaneous-frequency waveform).
    Fm,
    /// Force SSB mono audio. Requires `--audio-carrier-hz`.
    Ssb,
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
    if let Some(path) = std::env::var_os("OPENHOSHIMI_SYMBOL_DUMP") {
        open_symbol_dump(Path::new(&path))
            .map_err(|err| format!("failed to open symbol dump file: {err}"))?;
    }
    let satellite = load_satellite(&args.satellite_toml)
        .map_err(|err| format!("failed to load satellite definition: {err}"))?;
    let selected = select_downlink(&satellite)?;
    let telemetry = selected
        .downlink
        .telemetry_schema
        .as_ref()
        .and_then(|name| satellite.telemetry.get(name))
        .map(SchemaParser::new);
    let mut image_reassembler: Option<Box<dyn ImageReassembler>> = if args.dump_images.is_some() {
        match selected.downlink.image.as_ref() {
            Some(def) => match build_image_reassembler(def) {
                Ok(r) => Some(r),
                Err(err) => {
                    eprintln!(
                        "decode_file: --dump-images requested but image config is invalid: {err}"
                    );
                    None
                }
            },
            None => {
                eprintln!(
                        "decode_file: --dump-images requested but the active downlink has no [downlink.image] block"
                    );
                None
            }
        }
    } else {
        None
    };

    let tuning_offset_hz =
        infer_tuning_offset_hz(&args.input_wav, selected.downlink.freq_hz).unwrap_or(0) as f32;
    let mut runner = match selected.input_kind {
        InputKind::Audio => PipelineRunner::Audio {
            source: {
                WavSource::open(&args.input_wav)
                    .map_err(|err| format!("failed to open audio WAV: {err}"))?
            },
            pipeline: {
                let source = WavSource::open(&args.input_wav)
                    .map_err(|err| format!("failed to open audio WAV: {err}"))?;
                let sample_rate = source.sample_rate();
                let mut pipeline = BitPipeline::<f32>::new(selected.downlink)?;
                pipeline.configure_demodulator(selected.downlink, sample_rate, 0.0)?;
                pipeline
            },
        },
        InputKind::FmAudio => {
            let source: Box<dyn InputSource> = open_audio_source(&args.input_wav)?;
            let sample_rate = source.sample_rate();
            let mut pipeline = BitPipeline::<f32>::new(selected.downlink)?;
            pipeline.configure_fm_audio_demodulator(selected.downlink, sample_rate)?;
            PipelineRunner::FmAudio { source, pipeline }
        }
        InputKind::Iq => {
            let resolved_mode: IoAudioMode = match args.audio_mode {
                AudioMode::Auto => detect_audio_mode_auto(&args.input_wav)?,
                AudioMode::Iq => IoAudioMode::Iq,
                AudioMode::Fm => IoAudioMode::Fm,
                AudioMode::Ssb => IoAudioMode::Ssb,
            };
            match resolved_mode {
                IoAudioMode::Fm => {
                    eprintln!("decode_file: treating mono input as FM-discriminator audio");
                    let source: Box<dyn InputSource> = open_audio_source(&args.input_wav)?;
                    let sample_rate = source.sample_rate();
                    let mut pipeline = BitPipeline::<f32>::new(selected.downlink)?;
                    pipeline.configure_fm_audio_demodulator(selected.downlink, sample_rate)?;
                    PipelineRunner::FmAudio { source, pipeline }
                }
                IoAudioMode::Ssb => {
                    if is_ao40_fec_downlink(selected.downlink) {
                        return Err(
                            "SSB audio mode does not support AO-40 FEC downlinks".to_string()
                        );
                    }
                    let mut mono: Box<dyn InputSource> = open_audio_source(&args.input_wav)?;
                    let sample_rate = mono.sample_rate();
                    let (carrier_hz, prefix) = match args.audio_carrier_hz {
                        Some(value) => {
                            if !value.is_finite() {
                                return Err(format!(
                                    "--audio-carrier-hz must be finite, got {value}"
                                ));
                            }
                            eprintln!(
                                "decode_file: treating mono input as SSB audio, mixing carrier {value:.1} Hz to baseband (user-supplied)"
                            );
                            (value, Vec::new())
                        }
                        None => {
                            let nyquist = sample_rate as f32 / 2.0;
                            if nyquist <= 400.0 {
                                return Err(format!(
                                    "SSB audio carrier auto-estimate: sample rate {sample_rate} Hz is too low"
                                ));
                            }
                            let prefix_len = sample_rate as usize;
                            let prefix = read_audio_prefix(&mut *mono, prefix_len)?;
                            let estimate = estimate_audio_carrier(
                                &prefix,
                                sample_rate,
                                200.0,
                                nyquist - 200.0,
                            )
                            .ok_or_else(|| {
                                "SSB audio carrier auto-estimate failed: no peak found in audio band; pass --audio-carrier-hz <hz>".to_string()
                            })?;
                            let prefix_samples = prefix.len();
                            eprintln!(
                                "decode_file: treating mono input as SSB audio, mixing carrier {estimate:.1} Hz to baseband (auto-estimated from {prefix_samples} samples)"
                            );
                            (estimate, prefix)
                        }
                    };
                    let source: Box<dyn IqSource> =
                        Box::new(MonoIqSource::with_prefix(mono, prefix));
                    let mut pipeline = BitPipeline::<IqSample>::new(selected.downlink)?;
                    pipeline.configure_demodulator(selected.downlink, sample_rate, carrier_hz)?;
                    PipelineRunner::Iq {
                        source,
                        pipeline,
                        pending: Vec::new(),
                    }
                }
                IoAudioMode::Iq => {
                    let source = WavIqSource::open(&args.input_wav)
                        .map_err(|err| format!("failed to open WAV as IQ: {err}"))?;
                    build_iq_runner(source, selected.downlink, tuning_offset_hz)?
                }
                IoAudioMode::Auto => unreachable!("IoAudioMode::Auto is resolved above"),
            }
        }
    };
    eprintln!(
        "decode_file: using {} ({})",
        selected.downlink.label,
        runner.description()
    );

    let mut frame_count = 0usize;
    loop {
        let batch = runner.read_next_batch()?;
        let batch = match batch {
            Some(batch) => batch,
            None => break,
        };

        for entry in batch {
            frame_count += 1;
            let timestamp = runner.timestamp();
            match entry {
                BatchEntry::Decoded { decoded, raw } => {
                    print_decoded_frame(
                        frame_count,
                        timestamp,
                        decoded,
                        &raw,
                        telemetry.as_ref(),
                        &mut image_reassembler,
                    );
                }
                BatchEntry::Pending(mut frame) => {
                    frame.satellite_id = satellite.satellite.norad_id;
                    match runner.decode_frame(&frame) {
                        Ok(decoded) => print_decoded_frame(
                            frame_count,
                            timestamp,
                            decoded,
                            &frame.raw,
                            telemetry.as_ref(),
                            &mut image_reassembler,
                        ),
                        Err(err) => {
                            eprintln!(
                                "decode_file: failed to decode frame #{frame_count:03}: {err}"
                            );
                            let preview: Vec<String> = frame
                                .raw
                                .iter()
                                .take(16)
                                .map(|b| format!("{b:02X}"))
                                .collect();
                            eprintln!(
                                "decode_file:   first {} bytes: {}",
                                preview.len(),
                                preview.join(" ")
                            );
                        }
                    }
                }
                BatchEntry::DecodeError(err) => {
                    eprintln!("decode_file: failed to decode frame #{frame_count:03}: {err}");
                }
            }
        }
    }

    if let Some(err) = runner.last_framer_error() {
        eprintln!("decode_file: last discarded HDLC frame: {err}");
        if let Some(bytes) = runner.last_failed_bytes() {
            eprintln!(
                "decode_file: last discarded HDLC bytes ({} bytes): {}",
                bytes.len(),
                bytes
                    .iter()
                    .map(|b| format!("{b:02X}"))
                    .collect::<Vec<_>>()
                    .join(" "),
            );
        }
        if let Some(longest) = runner.longest_failed_bytes() {
            let same_as_last = runner
                .last_failed_bytes()
                .map(|last| last == longest)
                .unwrap_or(false);
            if !same_as_last {
                eprintln!(
                    "decode_file: longest discarded HDLC bytes ({} bytes): {}",
                    longest.len(),
                    longest
                        .iter()
                        .map(|b| format!("{b:02X}"))
                        .collect::<Vec<_>>()
                        .join(" "),
                );
            }
        }
    }
    if let Some(distance) = runner.best_sync_distance() {
        eprintln!("decode_file: run summary: frames={frame_count} best_sync_distance={distance}");
    } else if frame_count == 0 {
        eprintln!("decode_file: no frames decoded");
    }
    if let Some(history) = runner.distance_history() {
        let formatted: String = history
            .iter()
            .map(|entry| match entry {
                Some(distance) => format!("{distance:2}"),
                None => " -".to_string(),
            })
            .collect::<Vec<_>>()
            .join(" ");
        eprintln!(
            "decode_file: ao40 sync distance per second ({} entries): {formatted}",
            history.len()
        );
    }

    if let (Some(dir), Some(r)) = (args.dump_images.as_ref(), image_reassembler.as_ref()) {
        let snap = r.snapshot();
        std::fs::create_dir_all(dir).map_err(|e| format!("create {}: {e}", dir.display()))?;
        let ext = match snap.decoder {
            ImageDecoder::Jpeg => "jpg",
            ImageDecoder::Raw => "bin",
        };
        for stream in &snap.images {
            let path = dir.join(format!("openhoshimi_image_{:02}.{ext}", stream.image_idx));
            std::fs::write(&path, &stream.bytes)
                .map_err(|e| format!("write {}: {e}", path.display()))?;
            eprintln!(
                "decode_file: wrote {} ({} bytes, {}/{} chunks)",
                path.display(),
                stream.bytes.len(),
                stream.received_count,
                stream.total_chunks,
            );
        }
        eprintln!(
            "decode_file: dumped {} image(s) to {}",
            snap.images.len(),
            dir.display()
        );
    }

    Ok(())
}

enum PipelineRunner {
    Audio {
        source: WavSource,
        pipeline: BitPipeline<f32>,
    },
    FmAudio {
        source: Box<dyn InputSource>,
        pipeline: BitPipeline<f32>,
    },
    Iq {
        source: Box<dyn IqSource>,
        pipeline: BitPipeline<IqSample>,
        pending: Vec<IqSample>,
    },
    IqSoftAo40 {
        source: WavIqSource,
        pipeline: Box<SoftAo40Pipeline>,
        pending: Vec<IqSample>,
    },
}

enum BatchEntry {
    Decoded {
        decoded: DecodedFrame,
        raw: Vec<u8>,
    },
    Pending(Frame),
    #[allow(dead_code)]
    DecodeError(String),
}

impl PipelineRunner {
    fn description(&self) -> &str {
        match self {
            Self::Audio { source, .. } => source.description(),
            Self::FmAudio { source, .. } => source.description(),
            Self::Iq { source, .. } => source.description(),
            Self::IqSoftAo40 { source, .. } => source.description(),
        }
    }

    fn timestamp(&self) -> Duration {
        match self {
            Self::Audio { source, pipeline } => Duration::from_secs_f64(
                pipeline.total_samples() as f64 / source.sample_rate() as f64,
            ),
            Self::FmAudio { source, pipeline } => Duration::from_secs_f64(
                pipeline.total_samples() as f64 / source.sample_rate() as f64,
            ),
            Self::Iq {
                source, pipeline, ..
            } => Duration::from_secs_f64(
                pipeline.total_samples() as f64 / source.sample_rate() as f64,
            ),
            Self::IqSoftAo40 {
                source, pipeline, ..
            } => Duration::from_secs_f64(
                pipeline.total_samples() as f64 / source.sample_rate() as f64,
            ),
        }
    }

    fn read_next_batch(&mut self) -> Result<Option<Vec<BatchEntry>>, String> {
        match self {
            Self::Audio { source, pipeline } => {
                let mut samples = [0.0f32; READ_CHUNK];
                let read = match source.read_samples(&mut samples) {
                    Ok(read) => read,
                    Err(IoError::EndOfStream) => return Ok(None),
                    Err(err) => return Err(format!("failed to read WAV samples: {err}")),
                };
                let frames = pipeline.push_samples(&samples[..read]);
                Ok(Some(frames.into_iter().map(BatchEntry::Pending).collect()))
            }
            Self::FmAudio { source, pipeline } => {
                let mut samples = [0.0f32; READ_CHUNK];
                let read = match source.read_samples(&mut samples) {
                    Ok(read) => read,
                    Err(IoError::EndOfStream) => return Ok(None),
                    Err(err) => return Err(format!("failed to read audio samples: {err}")),
                };
                let frames = pipeline.push_samples(&samples[..read]);
                Ok(Some(frames.into_iter().map(BatchEntry::Pending).collect()))
            }
            Self::Iq {
                source,
                pipeline,
                pending,
            } => {
                if !pending.is_empty() {
                    let take = pending.len().min(READ_CHUNK);
                    let samples: Vec<IqSample> = pending.drain(..take).collect();
                    let frames = pipeline.push_samples(&samples);
                    return Ok(Some(frames.into_iter().map(BatchEntry::Pending).collect()));
                }
                let mut samples = [IqSample::default(); READ_CHUNK];
                let read = match source.read_samples(&mut samples) {
                    Ok(read) => read,
                    Err(IoError::EndOfStream) => return Ok(None),
                    Err(err) => return Err(format!("failed to read IQ WAV samples: {err}")),
                };
                let frames = pipeline.push_samples(&samples[..read]);
                Ok(Some(frames.into_iter().map(BatchEntry::Pending).collect()))
            }
            Self::IqSoftAo40 {
                source,
                pipeline,
                pending,
            } => {
                if !pending.is_empty() {
                    let take = pending.len().min(READ_CHUNK);
                    let samples: Vec<IqSample> = pending.drain(..take).collect();
                    let decoded = pipeline.push_samples(&samples);
                    return Ok(Some(
                        decoded
                            .into_iter()
                            .map(|decoded| BatchEntry::Decoded {
                                decoded,
                                raw: Vec::new(),
                            })
                            .collect(),
                    ));
                }
                let mut samples = [IqSample::default(); READ_CHUNK];
                let read = match source.read_samples(&mut samples) {
                    Ok(read) => read,
                    Err(IoError::EndOfStream) => return Ok(None),
                    Err(err) => return Err(format!("failed to read IQ WAV samples: {err}")),
                };
                let decoded = pipeline.push_samples(&samples[..read]);
                Ok(Some(
                    decoded
                        .into_iter()
                        .map(|decoded| BatchEntry::Decoded {
                            decoded,
                            raw: Vec::new(),
                        })
                        .collect(),
                ))
            }
        }
    }

    fn decode_frame(&self, frame: &Frame) -> Result<DecodedFrame, String> {
        match self {
            Self::Audio { pipeline, .. } => pipeline.decode_frame(frame),
            Self::FmAudio { pipeline, .. } => pipeline.decode_frame(frame),
            Self::Iq { pipeline, .. } => pipeline.decode_frame(frame),
            Self::IqSoftAo40 { .. } => {
                Err("IqSoftAo40 pipeline decodes inline; pending frames are not used".to_string())
            }
        }
    }

    fn last_framer_error(&self) -> Option<&openhoshimi_core::DecodeError> {
        match self {
            Self::Audio { pipeline, .. } => pipeline.last_framer_error(),
            Self::FmAudio { pipeline, .. } => pipeline.last_framer_error(),
            Self::Iq { pipeline, .. } => pipeline.last_framer_error(),
            Self::IqSoftAo40 { .. } => None,
        }
    }

    fn last_failed_bytes(&self) -> Option<&[u8]> {
        match self {
            Self::Audio { pipeline, .. } => pipeline.last_failed_bytes(),
            Self::FmAudio { pipeline, .. } => pipeline.last_failed_bytes(),
            Self::Iq { pipeline, .. } => pipeline.last_failed_bytes(),
            Self::IqSoftAo40 { .. } => None,
        }
    }

    fn longest_failed_bytes(&self) -> Option<&[u8]> {
        match self {
            Self::Audio { pipeline, .. } => pipeline.longest_failed_bytes(),
            Self::FmAudio { pipeline, .. } => pipeline.longest_failed_bytes(),
            Self::Iq { pipeline, .. } => pipeline.longest_failed_bytes(),
            Self::IqSoftAo40 { .. } => None,
        }
    }

    fn best_sync_distance(&self) -> Option<usize> {
        match self {
            Self::Audio { pipeline, .. } => pipeline.best_sync_distance(),
            Self::FmAudio { pipeline, .. } => pipeline.best_sync_distance(),
            Self::Iq { pipeline, .. } => pipeline.best_sync_distance(),
            Self::IqSoftAo40 { pipeline, .. } => pipeline.best_sync_distance(),
        }
    }

    fn distance_history(&self) -> Option<&[Option<usize>]> {
        match self {
            Self::Audio { .. } | Self::Iq { .. } | Self::FmAudio { .. } => None,
            Self::IqSoftAo40 { pipeline, .. } => Some(pipeline.distance_history()),
        }
    }
}

fn print_decoded_frame(
    index: usize,
    timestamp: Duration,
    frame: DecodedFrame,
    raw: &[u8],
    telemetry: Option<&SchemaParser>,
    image_reassembler: &mut Option<Box<dyn ImageReassembler>>,
) {
    match frame {
        DecodedFrame::Ax25(ax25) => {
            print_ax25_frame(index, timestamp, &ax25, raw);
            print_telemetry_fields(telemetry, &ax25.info);
        }
        DecodedFrame::Ao40 {
            payload,
            corrected_errors,
        } => {
            println!(
                "#{index:03}  {}  AO40-FEC             [AO-40 FEC]  {} bytes  corrected={corrected_errors}",
                format_timestamp(timestamp),
                payload.len()
            );
            if !raw.is_empty() {
                println!("      raw: {}", format_hex(raw));
            }
            print_telemetry_fields(telemetry, &payload);
        }
        DecodedFrame::Ax100 {
            mode,
            payload,
            corrected_errors,
            crc_ok,
        } => {
            let status = match crc_ok {
                Some(true) => "  [CRC ok]",
                Some(false) => "  [CRC FAIL]",
                None => "",
            };
            println!(
                "#{index:03}  {}  AX100/{:<11}  [GOMspace AX100]  {} bytes  corrected={corrected_errors}{status}",
                format_timestamp(timestamp),
                ax100_mode_label(mode),
                payload.len()
            );
            println!("      payload: {}", format_hex(&payload));
            let ascii: String = payload
                .iter()
                .map(|&b| if (0x20..0x7f).contains(&b) { b as char } else { '.' })
                .collect();
            println!("      ascii:   {ascii}");
            // Only surface telemetry for frames whose integrity is
            // confirmed; CRC-failed payloads carry residual bit errors and
            // must not be interpreted as housekeeping values.
            if crc_ok != Some(false) {
                print_telemetry_fields(telemetry, &payload);
            }
        }
        DecodedFrame::Geoscan(geoscan) => {
            print_geoscan_frame(index, timestamp, &geoscan, raw);
            print_telemetry_fields(telemetry, &geoscan.payload);
            if let Some(r) = image_reassembler.as_mut() {
                if geoscan.crc_ok {
                    if let Some(update) = r.ingest(&geoscan.payload) {
                        if update.started_new_image {
                            eprintln!(
                                "decode_file: image #{} starts at frame {} (offset 0)",
                                update.image_idx, index
                            );
                        }
                    }
                }
            }
        }
        DecodedFrame::Ssdv(packet) => {
            let crc = if packet.crc_ok { "CRC ok" } else { "CRC FAIL" };
            let kind = match packet.kind {
                openhoshimi_codec::SsdvPacketKind::WithFec => "SSDV+FEC",
                openhoshimi_codec::SsdvPacketKind::NoFec => "SSDV    ",
            };
            let rs = match packet.rs_errors {
                Some(n) => format!("rs={}", n),
                None => "rs=-".to_string(),
            };
            println!(
                "#{index:03}  {}  {kind}             [{crc}]  cs={:<6}  img={:03}  pkt={:05}  {}x{}  {rs}",
                format_timestamp(timestamp),
                packet.callsign,
                packet.image_id,
                packet.packet_id,
                packet.width,
                packet.height,
            );
            if let Some(r) = image_reassembler.as_mut() {
                if packet.crc_ok {
                    if let Some(update) = r.ingest(&packet.raw) {
                        if update.started_new_image {
                            eprintln!(
                                "decode_file: SSDV image #{} (id {}) starts at frame {}",
                                update.image_idx, packet.image_id, index
                            );
                        }
                    }
                }
            }
        }
        DecodedFrame::Raw {
            frame_type,
            raw_len,
        } => {
            println!(
                "#{index:03}  {}  RAW/{:<12}  [raw]  {raw_len} bytes",
                format_timestamp(timestamp),
                frame_type_label(frame_type)
            );
            println!("      raw: {}", format_hex(raw));
            print_telemetry_fields(telemetry, raw);
        }
    }
}

fn print_geoscan_frame(index: usize, timestamp: Duration, frame: &GeoscanFrame, raw: &[u8]) {
    let crc_status = if frame.crc_ok { "CRC ok" } else { "CRC FAIL" };
    println!(
        "#{index:03}  {}  GEOSCAN              [{crc_status}]  {} bytes  crc_rx={:#06x}  crc_calc={:#06x}",
        format_timestamp(timestamp),
        frame.payload.len(),
        frame.crc_received,
        frame.crc_expected,
    );
    if !raw.is_empty() {
        println!("      raw:     {}", format_hex(raw));
    }
    println!("      payload: {}", format_hex(&frame.payload));
    println!("      ascii:   {}", format_ascii(&frame.payload));
}

fn format_ascii(bytes: &[u8]) -> String {
    bytes
        .iter()
        .map(|b| {
            if (0x20..=0x7e).contains(b) {
                *b as char
            } else {
                '.'
            }
        })
        .collect()
}

fn print_ax25_frame(index: usize, timestamp: Duration, frame: &Ax25Frame, raw: &[u8]) {
    let source = format_call(&frame.source);
    let destination = format_call(&frame.destination);
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

fn ax100_mode_label(mode: Ax100Mode) -> &'static str {
    match mode {
        Ax100Mode::ReedSolomon => "rs",
        Ax100Mode::AsmGolay => "asm-golay",
        Ax100Mode::AsmGolayCrc => "asm-golay-crc",
    }
}

fn print_telemetry_fields(telemetry: Option<&SchemaParser>, bytes: &[u8]) {
    let Some(telemetry) = telemetry else {
        return;
    };

    for field in telemetry.parse_bytes(bytes) {
        println!(
            "      {}.{} = {}",
            field.group,
            field.key,
            format_telemetry_value(&field)
        );
    }
}

fn format_telemetry_value(field: &openhoshimi_core::TelemetryField) -> String {
    let value = match &field.value {
        openhoshimi_core::TelemetryValue::Float(value) => format_float(*value),
        openhoshimi_core::TelemetryValue::Int(value) => value.to_string(),
        openhoshimi_core::TelemetryValue::Bool(value) => value.to_string(),
        openhoshimi_core::TelemetryValue::Bytes(bytes) => format_hex(bytes),
    };
    match &field.unit {
        Some(unit) if !unit.is_empty() => format!("{value} {unit}"),
        _ => value,
    }
}

fn format_float(value: f64) -> String {
    let mut text = format!("{value:.3}");
    while text.contains('.') && text.ends_with('0') {
        text.pop();
    }
    if text.ends_with('.') {
        text.push('0');
    }
    text
}

/// Build the IQ-mode `PipelineRunner` for a real stereo IQ WAV source.
///
/// Owns the prefix scan, polarity/skew alignment, and CPM carrier
/// estimation that the IQ path uses to lock onto a recording.
fn build_iq_runner(
    mut source: WavIqSource,
    downlink_in: &openhoshimi_core::satellite::DownlinkDef,
    tuning_offset_hz: f32,
) -> Result<PipelineRunner, String> {
    let sample_rate = source.sample_rate();
    let prefix_len = sample_rate as usize * 8;
    let prefix = read_iq_prefix(&mut source, prefix_len)?;
    let scored =
        prepare_linear_iq_setup_scored(downlink_in, sample_rate, &prefix, tuning_offset_hz);
    let setup = scored.as_ref().map(|(setup, _)| setup.clone());
    let (downlink, tuning_offset_hz, pending) = if let Some(setup) = setup {
        let pending = if setup.sample_skip < prefix.len() {
            prefix.into_iter().skip(setup.sample_skip).collect()
        } else {
            Vec::new()
        };
        eprintln!(
            "decode_file: prefix setup: tuning={:.1} Hz skip={} prefix_frames={}",
            setup.tuning_offset_hz,
            setup.sample_skip,
            scored.as_ref().map_or(0, |(_, frames)| *frames),
        );
        if let Some(openhoshimi_core::satellite::ModemDef::Linear {
            differential,
            invert,
            swap_iq,
            ..
        }) = setup.downlink.modem.as_ref()
        {
            eprintln!(
                "decode_file: prefix polarity: differential={differential} invert={invert} swap_iq={swap_iq}",
            );
        }
        (setup.downlink, setup.tuning_offset_hz, pending)
    } else {
        let downlink = downlink_in.clone();
        let mut tuning = tuning_offset_hz;
        if matches!(downlink.modem, Some(ModemDef::Cpm { .. })) {
            let estimate = estimate_cpm_iq_frequency_offset_hz(&prefix, sample_rate)
                .or_else(|| estimate_iq_frequency_offset_hz(&prefix, sample_rate))
                .filter(|v| v.is_finite());
            if let Some(estimate) = estimate {
                eprintln!(
                    "decode_file: cpm carrier estimate from prefix: {estimate:.1} Hz (filename hint: {tuning:.1} Hz)",
                );
                tuning = estimate;
            }
        }
        (downlink, tuning, prefix)
    };
    if is_ao40_fec_downlink(&downlink) {
        eprintln!("decode_file: using soft-decision Viterbi path for AO-40 FEC");
        let pipeline = SoftAo40Pipeline::new(&downlink, sample_rate, tuning_offset_hz)?;
        Ok(PipelineRunner::IqSoftAo40 {
            source,
            pipeline: Box::new(pipeline),
            pending,
        })
    } else {
        let mut pipeline = BitPipeline::<IqSample>::new(&downlink)?;
        pipeline.configure_demodulator(&downlink, sample_rate, tuning_offset_hz)?;
        Ok(PipelineRunner::Iq {
            source: Box::new(source),
            pipeline,
            pending,
        })
    }
}
