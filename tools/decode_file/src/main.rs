//! Offline file decoding tool for OpenHoshimi.

use std::path::PathBuf;
use std::process::ExitCode;
use std::time::Duration;

use clap::Parser;
use openhoshimi_codec::{
    Ao40FecDecoder, Ax100Decoder, Ax100Mode, Ax25Decoder, Ax25Frame, Callsign,
};
use openhoshimi_core::satellite::{
    load_satellite, Ax100ModeDef, CodecDef, CpmModeDef, DescramblerDef, DownlinkDef, FramerDef,
    LineCodingDef, LinearModeDef, ModemDef, SatelliteDefinition,
};
use openhoshimi_core::{
    Demodulator, Descrambler, Frame, FrameType, Framing, InputSource, IoError, IqSample, IqSource,
    LineDecoder,
};
use openhoshimi_dsp::{
    Ao40Framer, CcsdsDescrambler, CpmConfig, CpmDemodulator, CpmMode, G3ruhDescrambler, HdlcFramer,
    LinearConfig, LinearDemodulator, LinearMode, NrziDecoder, SyncwordFramer,
};
use openhoshimi_io::{WavIqSource, WavSource};

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
    let satellite = load_satellite(&args.satellite_toml)
        .map_err(|err| format!("failed to load satellite definition: {err}"))?;
    let selected = select_downlink(&satellite)?;

    let mut runner = match selected.input_kind {
        InputKind::Audio => PipelineRunner::Audio {
            source: WavSource::open(&args.input_wav)
                .map_err(|err| format!("failed to open audio WAV: {err}"))?,
            pipeline: BitPipeline::new(selected.downlink)?,
        },
        InputKind::Iq => PipelineRunner::Iq {
            source: WavIqSource::open(&args.input_wav)
                .map_err(|err| format!("failed to open IQ WAV: {err}"))?,
            pipeline: BitPipeline::new(selected.downlink)?,
        },
    };

    runner.configure_sample_rate(selected.downlink)?;
    eprintln!(
        "decode_file: using {} ({})",
        selected.downlink.label,
        runner.description()
    );

    let mut frame_count = 0usize;
    loop {
        let next = runner.read_next_frames()?;
        let frames = match next {
            Some(frames) => frames,
            None => break,
        };

        for mut frame in frames {
            frame_count += 1;
            frame.satellite_id = satellite.satellite.norad_id;
            let timestamp = runner.timestamp();
            match runner.decode_frame(&frame) {
                Ok(decoded) => print_decoded_frame(frame_count, timestamp, decoded, &frame.raw),
                Err(err) => {
                    eprintln!("decode_file: failed to decode frame #{frame_count:03}: {err}")
                }
            }
        }
    }

    if let Some(err) = runner.last_framer_error() {
        eprintln!("decode_file: last discarded HDLC frame: {err}");
    }

    Ok(())
}

struct SelectedDownlink<'a> {
    downlink: &'a DownlinkDef,
    input_kind: InputKind,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum InputKind {
    Audio,
    Iq,
}

fn select_downlink(def: &SatelliteDefinition) -> Result<SelectedDownlink<'_>, String> {
    def.downlinks
        .iter()
        .find_map(|downlink| {
            input_kind_for(downlink).and_then(|input_kind| {
                if can_build_pipeline(downlink) {
                    Some(SelectedDownlink {
                        downlink,
                        input_kind,
                    })
                } else {
                    None
                }
            })
        })
        .ok_or_else(|| format!("no supported downlink found in {}", def.satellite.name))
}

fn input_kind_for(downlink: &DownlinkDef) -> Option<InputKind> {
    match &downlink.modem {
        Some(ModemDef::Afsk { .. }) => Some(InputKind::Audio),
        Some(ModemDef::Cpm { .. } | ModemDef::Linear { .. }) => Some(InputKind::Iq),
        Some(ModemDef::Lora { .. } | ModemDef::FourFsk { .. }) => None,
        None => {
            if matches_token(&downlink.modulation, &["AFSK", "BELL202", "BELL_202"]) {
                Some(InputKind::Audio)
            } else if matches_token(
                &downlink.modulation,
                &[
                    "FSK", "MSK", "GFSK", "GMSK", "BPSK", "DBPSK", "QPSK", "OQPSK",
                ],
            ) {
                Some(InputKind::Iq)
            } else {
                None
            }
        }
    }
}

fn can_build_pipeline(downlink: &DownlinkDef) -> bool {
    codec_kind(downlink).is_some() && framer_kind(downlink).is_some()
}

enum PipelineRunner {
    Audio {
        source: WavSource,
        pipeline: BitPipeline<f32>,
    },
    Iq {
        source: WavIqSource,
        pipeline: BitPipeline<IqSample>,
    },
}

impl PipelineRunner {
    fn configure_sample_rate(&mut self, downlink: &DownlinkDef) -> Result<(), String> {
        match self {
            Self::Audio { source, pipeline } => {
                pipeline.configure_demodulator(downlink, source.sample_rate())
            }
            Self::Iq { source, pipeline } => {
                pipeline.configure_demodulator(downlink, source.sample_rate())
            }
        }
    }

    fn description(&self) -> &str {
        match self {
            Self::Audio { source, .. } => source.description(),
            Self::Iq { source, .. } => source.description(),
        }
    }

    fn timestamp(&self) -> Duration {
        match self {
            Self::Audio { source, pipeline } => {
                Duration::from_secs_f64(pipeline.total_samples as f64 / source.sample_rate() as f64)
            }
            Self::Iq { source, pipeline } => {
                Duration::from_secs_f64(pipeline.total_samples as f64 / source.sample_rate() as f64)
            }
        }
    }

    fn read_next_frames(&mut self) -> Result<Option<Vec<Frame>>, String> {
        match self {
            Self::Audio { source, pipeline } => {
                let mut samples = [0.0f32; READ_CHUNK];
                let read = match source.read_samples(&mut samples) {
                    Ok(read) => read,
                    Err(IoError::EndOfStream) => return Ok(None),
                    Err(err) => return Err(format!("failed to read WAV samples: {err}")),
                };
                Ok(Some(pipeline.push_samples(&samples[..read])))
            }
            Self::Iq { source, pipeline } => {
                let mut samples = [IqSample::default(); READ_CHUNK];
                let read = match source.read_samples(&mut samples) {
                    Ok(read) => read,
                    Err(IoError::EndOfStream) => return Ok(None),
                    Err(err) => return Err(format!("failed to read IQ WAV samples: {err}")),
                };
                Ok(Some(pipeline.push_samples(&samples[..read])))
            }
        }
    }

    fn decode_frame(&self, frame: &Frame) -> Result<DecodedFrame, String> {
        match self {
            Self::Audio { pipeline, .. } => pipeline.decode_frame(frame),
            Self::Iq { pipeline, .. } => pipeline.decode_frame(frame),
        }
    }

    fn last_framer_error(&self) -> Option<&openhoshimi_core::DecodeError> {
        match self {
            Self::Audio { pipeline, .. } => pipeline.last_framer_error(),
            Self::Iq { pipeline, .. } => pipeline.last_framer_error(),
        }
    }
}

struct BitPipeline<S>
where
    S: Copy + Send + 'static,
{
    demodulator: Option<Box<dyn Demodulator<Sample = S>>>,
    line_decoder: Option<Box<dyn LineDecoder>>,
    descrambler: Option<Box<dyn Descrambler>>,
    framer: FrameStage,
    codec: CodecStage,
    total_samples: u64,
}

impl BitPipeline<f32> {
    fn configure_demodulator(
        &mut self,
        downlink: &DownlinkDef,
        sample_rate: u32,
    ) -> Result<(), String> {
        let modem = audio_modem_config(downlink)?;
        self.demodulator = Some(Box::new(
            openhoshimi_dsp::AfskDemodulator::with_tones(
                sample_rate,
                modem.mark_hz,
                modem.space_hz,
                downlink.baudrate,
            )
            .map_err(|err| format!("failed to configure AFSK demodulator: {err}"))?,
        ));
        Ok(())
    }
}

impl BitPipeline<IqSample> {
    fn configure_demodulator(
        &mut self,
        downlink: &DownlinkDef,
        sample_rate: u32,
    ) -> Result<(), String> {
        self.demodulator = Some(build_iq_demodulator(downlink, sample_rate)?);
        Ok(())
    }
}

impl<S> BitPipeline<S>
where
    S: Copy + Send + 'static,
{
    fn new(downlink: &DownlinkDef) -> Result<Self, String> {
        Ok(Self {
            demodulator: None,
            line_decoder: build_line_decoder(downlink)?,
            descrambler: build_descrambler(downlink)?,
            framer: build_framer(downlink)?,
            codec: build_codec(downlink)?,
            total_samples: 0,
        })
    }

    fn push_samples(&mut self, samples: &[S]) -> Vec<Frame> {
        let Some(demodulator) = self.demodulator.as_mut() else {
            return Vec::new();
        };

        self.total_samples += samples.len() as u64;
        let mut bits = demodulator.push_samples(samples);
        if let Some(line_decoder) = self.line_decoder.as_mut() {
            line_decoder.decode(&mut bits);
        }
        if let Some(descrambler) = self.descrambler.as_mut() {
            descrambler.descramble(&mut bits);
        }
        self.framer.push_bits(&bits)
    }

    fn decode_frame(&self, frame: &Frame) -> Result<DecodedFrame, String> {
        self.codec.decode(frame)
    }

    fn last_framer_error(&self) -> Option<&openhoshimi_core::DecodeError> {
        self.framer.last_error()
    }
}

#[derive(Debug, Clone, Copy)]
struct AfskModemConfig {
    mark_hz: f32,
    space_hz: f32,
}

fn audio_modem_config(downlink: &DownlinkDef) -> Result<AfskModemConfig, String> {
    match &downlink.modem {
        Some(ModemDef::Afsk { mark_hz, space_hz }) => Ok(AfskModemConfig {
            mark_hz: *mark_hz,
            space_hz: *space_hz,
        }),
        Some(_) => Err(format!("{} is not an audio AFSK downlink", downlink.label)),
        None if matches_token(&downlink.modulation, &["AFSK", "BELL202", "BELL_202"]) => {
            Ok(AfskModemConfig {
                mark_hz: 1200.0,
                space_hz: 2200.0,
            })
        }
        None => Err(format!("{} has no supported audio modem", downlink.label)),
    }
}

fn build_iq_demodulator(
    downlink: &DownlinkDef,
    sample_rate: u32,
) -> Result<Box<dyn Demodulator<Sample = IqSample>>, String> {
    match &downlink.modem {
        Some(ModemDef::Cpm {
            mode,
            modulation_index,
            gaussian_bt,
            differential,
            invert,
        }) => {
            let mut config = CpmConfig::new(sample_rate, downlink.baudrate, map_cpm_mode(*mode));
            if let Some(value) = modulation_index {
                config.modulation_index = *value;
            }
            if gaussian_bt.is_some() {
                config.gaussian_bt = *gaussian_bt;
            }
            config.differential = *differential;
            config.invert = *invert;
            Ok(Box::new(CpmDemodulator::new(config).map_err(|err| {
                format!("failed to configure CPM demodulator: {err}")
            })?))
        }
        Some(ModemDef::Linear {
            mode,
            differential,
            invert,
        }) => {
            let mut config =
                LinearConfig::new(sample_rate, downlink.baudrate, map_linear_mode(*mode));
            config.differential = *differential;
            config.invert = *invert;
            Ok(Box::new(LinearDemodulator::new(config).map_err(|err| {
                format!("failed to configure linear demodulator: {err}")
            })?))
        }
        Some(ModemDef::Afsk { .. }) => Err(format!("{} is an audio AFSK downlink", downlink.label)),
        Some(ModemDef::Lora { .. }) => {
            Err("LoRa demodulation is reserved but not implemented".to_string())
        }
        Some(ModemDef::FourFsk { .. }) => {
            Err("4FSK demodulation is reserved but not implemented".to_string())
        }
        None if matches_token(&downlink.modulation, &["FSK", "MSK", "GFSK", "GMSK"]) => {
            let mode = if downlink.modulation.eq_ignore_ascii_case("MSK") {
                CpmMode::Msk
            } else if downlink.modulation.eq_ignore_ascii_case("GFSK") {
                CpmMode::Gfsk
            } else if downlink.modulation.eq_ignore_ascii_case("GMSK") {
                CpmMode::Gmsk
            } else {
                CpmMode::Fsk
            };
            let config = CpmConfig::new(sample_rate, downlink.baudrate, mode);
            Ok(Box::new(CpmDemodulator::new(config).map_err(|err| {
                format!("failed to configure CPM demodulator: {err}")
            })?))
        }
        None if matches_token(&downlink.modulation, &["BPSK", "DBPSK", "QPSK", "OQPSK"]) => {
            let mode = if downlink.modulation.eq_ignore_ascii_case("DBPSK") {
                LinearMode::Dbpsk
            } else if downlink.modulation.eq_ignore_ascii_case("QPSK") {
                LinearMode::Qpsk
            } else if downlink.modulation.eq_ignore_ascii_case("OQPSK") {
                LinearMode::Oqpsk
            } else {
                LinearMode::Bpsk
            };
            let config = LinearConfig::new(sample_rate, downlink.baudrate, mode);
            Ok(Box::new(LinearDemodulator::new(config).map_err(|err| {
                format!("failed to configure linear demodulator: {err}")
            })?))
        }
        None => Err(format!("{} has no supported IQ modem", downlink.label)),
    }
}

fn map_cpm_mode(mode: CpmModeDef) -> CpmMode {
    match mode {
        CpmModeDef::Fsk => CpmMode::Fsk,
        CpmModeDef::Msk => CpmMode::Msk,
        CpmModeDef::Gfsk => CpmMode::Gfsk,
        CpmModeDef::Gmsk => CpmMode::Gmsk,
    }
}

fn map_linear_mode(mode: LinearModeDef) -> LinearMode {
    match mode {
        LinearModeDef::Bpsk => LinearMode::Bpsk,
        LinearModeDef::Dbpsk => LinearMode::Dbpsk,
        LinearModeDef::Qpsk => LinearMode::Qpsk,
        LinearModeDef::Oqpsk => LinearMode::Oqpsk,
    }
}

fn build_line_decoder(downlink: &DownlinkDef) -> Result<Option<Box<dyn LineDecoder>>, String> {
    match &downlink.line_coding {
        Some(LineCodingDef::Nrzi) => Ok(Some(Box::new(NrziDecoder::new()))),
        Some(LineCodingDef::Nrzs | LineCodingDef::Nrzm) => {
            Err("NRZ-S and NRZ-M line decoding are reserved but not implemented".to_string())
        }
        None => Ok(None),
    }
}

fn build_descrambler(downlink: &DownlinkDef) -> Result<Option<Box<dyn Descrambler>>, String> {
    match &downlink.descrambler {
        Some(DescramblerDef::G3ruh) => Ok(Some(Box::new(G3ruhDescrambler::new()))),
        Some(DescramblerDef::Ccsds) => Ok(Some(Box::new(CcsdsDescrambler::new()))),
        None => Ok(None),
    }
}

enum FrameStage {
    Ao40(Ao40Framer),
    Hdlc(HdlcFramer),
    Syncword(SyncwordFramer),
}

impl FrameStage {
    fn push_bits(&mut self, bits: &[u8]) -> Vec<Frame> {
        match self {
            Self::Ao40(framer) => framer.push_bytes(bits),
            Self::Hdlc(framer) => framer.push_bytes(bits),
            Self::Syncword(framer) => framer.push_bytes(bits),
        }
    }

    fn last_error(&self) -> Option<&openhoshimi_core::DecodeError> {
        match self {
            Self::Ao40(_) => None,
            Self::Hdlc(framer) => framer.last_error(),
            Self::Syncword(_) => None,
        }
    }
}

fn build_framer(downlink: &DownlinkDef) -> Result<FrameStage, String> {
    match framer_kind(downlink) {
        Some(FramerKind::Ao40 { threshold }) => Ok(FrameStage::Ao40(Ao40Framer::new(threshold))),
        Some(FramerKind::Hdlc) => Ok(FrameStage::Hdlc(HdlcFramer::new())),
        Some(FramerKind::Syncword {
            syncword,
            threshold,
            payload_bits,
        }) => Ok(FrameStage::Syncword(SyncwordFramer::with_frame_options(
            &syncword,
            threshold,
            payload_bits,
            frame_type_for_codec(downlink),
            pack_syncword_payload(downlink),
        ))),
        None => Err(format!("{} has no supported framer", downlink.label)),
    }
}

enum FramerKind {
    Ao40 {
        threshold: usize,
    },
    Hdlc,
    Syncword {
        syncword: Vec<u8>,
        threshold: usize,
        payload_bits: usize,
    },
}

fn framer_kind(downlink: &DownlinkDef) -> Option<FramerKind> {
    match &downlink.framer {
        Some(FramerDef::Ao40 { threshold }) => Some(FramerKind::Ao40 {
            threshold: *threshold,
        }),
        Some(FramerDef::Hdlc) => Some(FramerKind::Hdlc),
        Some(FramerDef::Syncword {
            syncword,
            threshold,
            payload_bits,
        }) => Some(FramerKind::Syncword {
            syncword: parse_syncword(syncword),
            threshold: *threshold,
            payload_bits: *payload_bits,
        }),
        None if downlink.framing.eq_ignore_ascii_case("AO40_FEC") => {
            Some(FramerKind::Ao40 { threshold: 0 })
        }
        None if matches_token(&downlink.framing, &["AX25", "AX.25", "HDLC"]) => {
            Some(FramerKind::Hdlc)
        }
        None => None,
    }
}

fn parse_syncword(syncword: &str) -> Vec<u8> {
    syncword
        .bytes()
        .map(|byte| u8::from(byte == b'1'))
        .collect()
}

fn pack_syncword_payload(downlink: &DownlinkDef) -> bool {
    !matches!(codec_kind(downlink), Some(CodecKind::Ao40Fec))
}

fn frame_type_for_codec(downlink: &DownlinkDef) -> FrameType {
    match codec_kind(downlink) {
        Some(CodecKind::Ax25) => FrameType::Ax25,
        Some(CodecKind::Ao40Fec) => FrameType::Ao40Fec,
        Some(CodecKind::GomspaceAx100(_)) => FrameType::GomspaceAx100,
        Some(CodecKind::Unknown) | None => FrameType::Unknown,
    }
}

enum CodecStage {
    Ax25(Ax25Decoder),
    Ao40(Ao40FecDecoder),
    Ax100 {
        decoder: Ax100Decoder,
        mode: Ax100Mode,
    },
    Unknown,
}

impl CodecStage {
    fn decode(&self, frame: &Frame) -> Result<DecodedFrame, String> {
        match self {
            Self::Ax25(decoder) => {
                let frame = decoder
                    .decode_frame(frame)
                    .map_err(|err| format!("AX.25 decode failed: {err}"))?;
                Ok(DecodedFrame::Ax25(frame))
            }
            Self::Ao40(decoder) => {
                let frame = decoder
                    .decode_channel_bits(&frame.raw)
                    .map_err(|err| format!("AO-40 FEC decode failed: {err}"))?;
                Ok(DecodedFrame::Ao40 {
                    payload: frame.payload,
                    corrected_errors: frame.corrected_errors,
                })
            }
            Self::Ax100 { decoder, mode } => {
                let frame = match mode {
                    Ax100Mode::ReedSolomon => decoder.decode_reed_solomon(&frame.raw),
                    Ax100Mode::AsmGolay => decoder.decode_asm_golay(&frame.raw),
                }
                .map_err(|err| format!("AX100 decode failed: {err}"))?;
                Ok(DecodedFrame::Ax100 {
                    mode: frame.mode,
                    payload: frame.payload,
                    corrected_errors: frame.corrected_errors,
                })
            }
            Self::Unknown => Ok(DecodedFrame::Raw {
                frame_type: frame.frame_type,
                raw_len: frame.raw.len(),
            }),
        }
    }
}

fn build_codec(downlink: &DownlinkDef) -> Result<CodecStage, String> {
    match codec_kind(downlink) {
        Some(CodecKind::Ax25) => Ok(CodecStage::Ax25(Ax25Decoder::new())),
        Some(CodecKind::Ao40Fec) => Ok(CodecStage::Ao40(Ao40FecDecoder::new())),
        Some(CodecKind::GomspaceAx100(mode)) => Ok(CodecStage::Ax100 {
            decoder: Ax100Decoder::new(),
            mode,
        }),
        Some(CodecKind::Unknown) => Ok(CodecStage::Unknown),
        None => Err(format!("{} has no supported codec", downlink.label)),
    }
}

#[derive(Debug, Clone, Copy)]
enum CodecKind {
    Ax25,
    Ao40Fec,
    GomspaceAx100(Ax100Mode),
    Unknown,
}

fn codec_kind(downlink: &DownlinkDef) -> Option<CodecKind> {
    match &downlink.codec {
        Some(CodecDef::Ax25) => Some(CodecKind::Ax25),
        Some(CodecDef::Ao40Fec) => Some(CodecKind::Ao40Fec),
        Some(CodecDef::GomspaceAx100 { mode }) => Some(CodecKind::GomspaceAx100(match mode {
            Ax100ModeDef::ReedSolomon => Ax100Mode::ReedSolomon,
            Ax100ModeDef::AsmGolay => Ax100Mode::AsmGolay,
        })),
        Some(CodecDef::Unknown) => Some(CodecKind::Unknown),
        Some(CodecDef::Ccsds | CodecDef::Fx25) => None,
        None if matches_token(&downlink.framing, &["AX25", "AX.25", "HDLC"]) => {
            Some(CodecKind::Ax25)
        }
        None if downlink.framing.eq_ignore_ascii_case("AO40_FEC") => Some(CodecKind::Ao40Fec),
        None if downlink.framing.eq_ignore_ascii_case("GOMSPACE_AX100") => {
            Some(CodecKind::GomspaceAx100(Ax100Mode::AsmGolay))
        }
        None if downlink.framing.eq_ignore_ascii_case("UNKNOWN") => Some(CodecKind::Unknown),
        None => None,
    }
}

enum DecodedFrame {
    Ax25(Ax25Frame),
    Ao40 {
        payload: Vec<u8>,
        corrected_errors: usize,
    },
    Ax100 {
        mode: Ax100Mode,
        payload: Vec<u8>,
        corrected_errors: usize,
    },
    Raw {
        frame_type: FrameType,
        raw_len: usize,
    },
}

fn print_decoded_frame(index: usize, timestamp: Duration, frame: DecodedFrame, raw: &[u8]) {
    match frame {
        DecodedFrame::Ax25(ax25) => print_ax25_frame(index, timestamp, &ax25, raw),
        DecodedFrame::Ao40 {
            payload,
            corrected_errors,
        } => {
            println!(
                "#{index:03}  {}  AO40-FEC             [AO-40 FEC]  {} bytes  corrected={corrected_errors}",
                format_timestamp(timestamp),
                payload.len()
            );
            println!("      raw: {}", format_hex(raw));
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

fn format_call(call: &Callsign) -> String {
    format!("{}-{}", call.call, call.ssid)
}

fn ax100_mode_label(mode: Ax100Mode) -> &'static str {
    match mode {
        Ax100Mode::ReedSolomon => "rs",
        Ax100Mode::AsmGolay => "asm-golay",
    }
}

fn frame_type_label(frame_type: FrameType) -> &'static str {
    match frame_type {
        FrameType::Ax25 => "ax25",
        FrameType::Ao40Fec => "ao40-fec",
        FrameType::GomspaceAx100 => "ax100",
        FrameType::Ccsds => "ccsds",
        FrameType::Fx25 => "fx25",
        FrameType::Unknown => "unknown",
    }
}

fn matches_token(value: &str, tokens: &[&str]) -> bool {
    tokens.iter().any(|token| value.eq_ignore_ascii_case(token))
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
