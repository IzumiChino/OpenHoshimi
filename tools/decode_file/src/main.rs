//! Offline file decoding tool for OpenHoshimi.

use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::time::Duration;

use clap::Parser;
use openhoshimi_codec::{Ax100Mode, Ax25Frame};
use openhoshimi_core::satellite::load_satellite;
use openhoshimi_core::satellite::ModemDef;
use openhoshimi_core::{Frame, InputSource, IoError, IqSample, IqSource};
use openhoshimi_dsp::linear::open_symbol_dump;
use openhoshimi_io::{OggSource, WavIqSource, WavSource};
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
            match WavIqSource::open(&args.input_wav) {
                Err(_) => {
                    // Mono file provided for an IQ downlink — fall back to FM audio path.
                    eprintln!(
                        "decode_file: mono input detected for IQ downlink, using FM audio path"
                    );
                    let source: Box<dyn InputSource> = open_audio_source(&args.input_wav)?;
                    let sample_rate = source.sample_rate();
                    let mut pipeline = BitPipeline::<f32>::new(selected.downlink)?;
                    pipeline.configure_fm_audio_demodulator(selected.downlink, sample_rate)?;
                    PipelineRunner::FmAudio { source, pipeline }
                }
                Ok(mut source) => {
                    let sample_rate = source.sample_rate();
                    let prefix_len = sample_rate as usize * 8;
                    let prefix = read_iq_prefix(&mut source, prefix_len)?;
                    let scored = prepare_linear_iq_setup_scored(
                        selected.downlink,
                        sample_rate,
                        &prefix,
                        tuning_offset_hz,
                    );
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
                        let downlink = selected.downlink.clone();
                        let mut tuning = tuning_offset_hz;
                        if matches!(downlink.modem, Some(ModemDef::Cpm { .. })) {
                            let estimate =
                                estimate_cpm_iq_frequency_offset_hz(&prefix, sample_rate)
                                    .or_else(|| {
                                        estimate_iq_frequency_offset_hz(&prefix, sample_rate)
                                    })
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
                        let pipeline =
                            SoftAo40Pipeline::new(&downlink, sample_rate, tuning_offset_hz)?;
                        PipelineRunner::IqSoftAo40 {
                            source,
                            pipeline: Box::new(pipeline),
                            pending,
                        }
                    } else {
                        let mut pipeline = BitPipeline::<IqSample>::new(&downlink)?;
                        pipeline.configure_demodulator(&downlink, sample_rate, tuning_offset_hz)?;
                        PipelineRunner::Iq {
                            source,
                            pipeline,
                            pending,
                        }
                    }
                }
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
                    print_decoded_frame(frame_count, timestamp, decoded, &raw, telemetry.as_ref());
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
                        ),
                        Err(err) => eprintln!(
                            "decode_file: failed to decode frame #{frame_count:03}: {err}"
                        ),
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
        source: WavIqSource,
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

fn read_iq_prefix(source: &mut dyn IqSource, len: usize) -> Result<Vec<IqSample>, String> {
    let mut prefix = Vec::with_capacity(len);
    let mut buf = [IqSample::default(); READ_CHUNK];
    while prefix.len() < len {
        let read = match source.read_samples(&mut buf) {
            Ok(read) => read,
            Err(IoError::EndOfStream) => break,
            Err(err) => return Err(format!("failed to prime IQ WAV input: {err}")),
        };
        if read == 0 {
            break;
        }
        let remaining = len - prefix.len();
        prefix.extend_from_slice(&buf[..read.min(remaining)]);
    }
    Ok(prefix)
}

fn print_decoded_frame(
    index: usize,
    timestamp: Duration,
    frame: DecodedFrame,
    raw: &[u8],
    telemetry: Option<&SchemaParser>,
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
        } => {
            println!(
                "#{index:03}  {}  AX100/{:<11}  [GOMspace AX100]  {} bytes  corrected={corrected_errors}",
                format_timestamp(timestamp),
                ax100_mode_label(mode),
                payload.len()
            );
            println!("      raw: {}", format_hex(raw));
            print_telemetry_fields(telemetry, &payload);
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

/// Open a mono audio source from a WAV or OGG file, based on extension.
fn open_audio_source(path: &Path) -> Result<Box<dyn InputSource>, String> {
    let ext = path
        .extension()
        .map(|e| e.to_string_lossy().to_lowercase())
        .unwrap_or_default();
    match ext.as_str() {
        "ogg" => {
            let source =
                OggSource::open(path).map_err(|err| format!("failed to open OGG file: {err}"))?;
            Ok(Box::new(source))
        }
        _ => {
            let source =
                WavSource::open(path).map_err(|err| format!("failed to open WAV file: {err}"))?;
            Ok(Box::new(source))
        }
    }
}
