//! egui desktop application for OpenHoshimi.

#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]
#![deny(missing_docs)]
#![forbid(unsafe_code)]

use std::collections::{BTreeMap, VecDeque};
use std::f32::consts::PI;
use std::path::{Path, PathBuf};
use std::sync::mpsc::{self, Receiver, Sender, SyncSender, TrySendError};
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

use eframe::egui;
use eframe::egui::{
    Align, Color32, ColorImage, CornerRadius, FontFamily, FontId, Frame, Layout, Margin, Pos2,
    ProgressBar, Rect, RichText, ScrollArea, Sense, Stroke, TextEdit, TextureOptions, Vec2,
};
use egui_extras::{Column, TableBuilder};
use openhoshimi_codec::SstvAnalyzer;
use openhoshimi_core::satellite::{
    load_all_satellites, DownlinkDef, ImageDef, ModemDef, SatelliteDefinition,
};
use openhoshimi_core::{
    Frame as CoreFrame, InputSource, IoError, IqSample, IqSource, TelemetryField, TelemetryValue,
};
use openhoshimi_dsp::estimate_audio_carrier;
use openhoshimi_io::{
    detect_audio_mode_auto, enumerate_input_devices, open_audio_source, read_audio_prefix,
    read_iq_prefix, AudioMode as IoAudioMode, MonoIqSource, SoundcardDeviceInfo, SoundcardIqSource,
    SoundcardSource, WavIqSource,
};
use openhoshimi_runtime::pipeline::{
    can_build_downlink, estimate_cpm_iq_frequency_offset_hz, estimate_iq_frequency_offset_hz,
    format_hex, format_timestamp, infer_tuning_offset_hz, input_kind_for, is_ao40_fec_downlink,
    is_linear_iq_modem, prepare_linear_iq_setup_scored,
    prepare_linear_iq_setup_scored_with_progress, BitPipeline, DecodedFrame, InputKind,
    PipelineStats, SoftAo40Pipeline,
};
use openhoshimi_telemetry::SchemaParser;
use rfd::FileDialog;
use rustfft::num_complex::Complex32;
use rustfft::{Fft, FftPlanner};

const READ_CHUNK: usize = 4096;
const MAX_FRAMES: usize = 512;
const MAX_WATERFALL_ROWS: usize = 256;
const MAX_DIAGNOSTICS: usize = 24;
const SPECTRUM_BINS: usize = 1024;
const SPECTRUM_FFT_LEN: usize = 4096;
const SPECTRUM_INTERVAL: Duration = Duration::from_millis(33);
const RX_EVENT_QUEUE: usize = 1024;
const WATERFALL_HEIGHT_PX: f32 = 220.0;
const DEFAULT_WATERFALL_MIN_DB: f32 = -90.0;
const DEFAULT_WATERFALL_MAX_DB: f32 = -10.0;
const WATERFALL_DB_FLOOR: f32 = -160.0;
const WATERFALL_DB_CEIL: f32 = 20.0;
/// Minimum alignment scorer prefix length in seconds.  For high baud rates
/// (≥4800) one second contains multiple frames and is sufficient.  For low
/// baud rates the actual prefix is extended to cover at least two frame
/// durations (see `alignment_scorer_prefix_secs`).  The cached prefix held
/// for the decoder run remains the full 8 s; only the scorer's per-trial
/// sweep is shortened.  If scoring returns zero frames against the short
/// slice we fall back to the full prefix.
const ALIGNMENT_SCORER_MIN_PREFIX_SEC: f32 = 1.0;

/// Compute the scorer prefix duration based on the downlink baud rate.
/// Ensures at least two frame periods fit in the prefix so the framer can
/// lock, while keeping the sweep fast for high-baud-rate signals.
fn alignment_scorer_prefix_secs(baudrate: u32) -> f32 {
    // Typical frame: ~256 bytes = ~2048 bits.  Two frames at the given
    // baud rate gives the minimum time needed for the framer to lock.
    let two_frames_sec = 2.0 * 2048.0 / baudrate.max(1) as f32;
    ALIGNMENT_SCORER_MIN_PREFIX_SEC.max(two_frames_sec).min(4.0)
}

/// Application entry point.
fn main() -> eframe::Result<()> {
    let options = eframe::NativeOptions::default();
    eframe::run_native(
        "OpenHoshimi",
        options,
        Box::new(|cc| Ok(Box::new(OpenHoshimiApp::new(cc)))),
    )
}

/// Selects which view occupies the lower half of the middle column.
/// Only the Frames variant is reachable on satellites that do not
/// declare a `[downlink.image]` block in their TOML; in that case the
/// tab bar is hidden and the field stays at its default.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BottomTab {
    /// Original frame list + hex inspector view.
    Frames,
    /// Image reassembly canvas + packet grid + status row.
    Image,
}

/// Main GUI state.
struct OpenHoshimiApp {
    satellites_dir: PathBuf,
    satellites: Vec<SatelliteDefinition>,
    selected_satellite: usize,
    selected_downlink: usize,
    selected_frame: Option<usize>,
    filter: String,
    display_filter: String,
    offset_hz: i32,
    input_rate: u32,
    running: bool,
    status: String,
    source_description: String,
    input_mode: InputMode,
    decode_pace: DecodePace,
    input_path: Option<PathBuf>,
    frames: VecDeque<FrameRow>,
    dropped_frames: usize,
    samples_processed: u64,
    input_progress_processed: u64,
    input_progress_total: Option<u64>,
    frames_seen: usize,
    diagnostics: VecDeque<DiagnosticEntry>,
    waterfall: VecDeque<Vec<f32>>,
    spectrum: SpectrumProcessor,
    waterfall_texture: Option<egui::TextureHandle>,
    waterfall_min_db: f32,
    waterfall_max_db: f32,
    rx_thread: Option<thread::JoinHandle<()>>,
    rx_stop: Option<Sender<()>>,
    rx_events: Receiver<RxEvent>,
    rx_sender: SyncSender<RxEvent>,
    startup_wav: Option<PathBuf>,
    last_export_status: Option<String>,
    alignment_state: AlignmentState,
    cached_alignment: Option<CachedAlignment>,
    alignment_thread: Option<thread::JoinHandle<()>>,
    /// Names of currently-known soundcard input devices.  Refreshed once
    /// at startup and on demand from the input panel; empty entries mean
    /// the cpal default host has no input devices.
    soundcard_devices: Vec<SoundcardDeviceInfo>,
    /// Name of the soundcard input device chosen by the user.  `None`
    /// means "use whatever cpal reports as default" so a missing or
    /// unselected device falls back to `SoundcardSource::open_default`.
    selected_soundcard: Option<String>,
    /// How the active WAV file should be interpreted by an IQ-family
    /// downlink.  Auto sniffs the file and falls back to FM-discriminator
    /// audio for mono / duplicate-stereo inputs; explicit Fm/Ssb force
    /// that path even when the file looks like real IQ.
    audio_mode: IoAudioMode,
    /// Audio-band carrier frequency in Hz, used only when `audio_mode`
    /// resolves to [`IoAudioMode::Ssb`].  The CPM IQ demodulator mixes
    /// this carrier to baseband and the complex low-pass kills the
    /// mirror sideband.
    audio_carrier_hz: f32,
    /// Which view the middle column's lower region is currently showing.
    /// Only meaningful when the active downlink declares a
    /// `[downlink.image]` block; otherwise the tab bar is hidden and the
    /// Frames view renders unconditionally.
    bottom_tab: BottomTab,
    /// Stateful image reassembler for the active downlink, populated
    /// from `DownlinkDef::image` whenever the user selects a downlink
    /// that carries images. `None` for satellites without image support.
    image_reassembler: Option<Box<dyn openhoshimi_codec::ImageReassembler>>,
    /// GPU texture for the currently-displayed image canvas. Rebuilt
    /// lazily from `image_reassembler.snapshot()` when `image_dirty` is
    /// set.
    image_texture: Option<egui::TextureHandle>,
    /// `image_idx` whose decoded pixels currently sit in
    /// `image_texture`. Lets us tell "old texture from a different
    /// stream" (must be cleared) from "old texture from the same
    /// stream" (keep across transient JPEG decode failures while
    /// later chunks fill in).
    image_texture_for_idx: Option<u32>,
    /// Set when a new chunk has been ingested, consumed by the image
    /// view on the next repaint to upload a fresh texture.
    image_dirty: bool,
    /// Image-stream index currently shown in the image view. Stays at
    /// 0 until the user clicks a different image chip. The Geoscan
    /// protocol does not carry an explicit image id, so the
    /// reassembler assigns indices in arrival order; this is the same
    /// index reported by `ImageStream::image_idx`.
    image_active_idx: u32,
    /// Diagnostic counter for image payloads that arrived while the
    /// reassembler was active but failed the header / structural check.
    /// Reset whenever the reassembler is rebuilt.
    image_payload_misses: u32,
    /// Diagnostic counter for image payloads that were successfully
    /// ingested into the reassembler. Reset on rebuild.
    image_payload_hits: u32,
    /// Set once we have logged a representative non-matching payload
    /// prefix to the diagnostic panel, so the log is not flooded.
    image_payload_logged: bool,
    /// Latest fully-decoded SSTV image, if any. Populated from
    /// `RxEvent::SstvImage` and rendered by the image view when the
    /// active downlink is SSTV; the bit-pipeline reassembler is bypassed
    /// for this path. Reset whenever the reassembler is refreshed (i.e.
    /// the active downlink changes).
    latest_sstv: Option<openhoshimi_codec::SstvImage>,
    /// GPU texture for `latest_sstv`. Lazily rebuilt when
    /// `sstv_dirty` is set.
    sstv_texture: Option<egui::TextureHandle>,
    /// Repaint trigger for `sstv_texture`. Mirrors `image_dirty` but
    /// targets the SSTV pathway.
    sstv_dirty: bool,
    /// Most recent pipeline counters published by the decode thread.
    /// `None` until the first `RxEvent::PipelineStats` arrives or after
    /// a downlink change resets the GUI side.  Surfaced in the status
    /// bar so the operator can tell at a glance how far signal is making
    /// it through demod / sync / framing.
    latest_stats: Option<PipelineStats>,
}

#[derive(Debug, Clone)]
struct FrameRow {
    index: usize,
    time: Duration,
    source: String,
    destination: String,
    kind: FrameKind,
    rssi_dbm: Option<f32>,
    raw: Vec<u8>,
    telemetry: Vec<TelemetryField>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum FrameKind {
    Tlm,
    Wod,
    Err,
    Raw,
}

enum RxEvent {
    Frame(FrameRow),
    Dropped(String),
    Samples(SpectrumSamples),
    Progress {
        processed: u64,
        total: Option<u64>,
    },
    SourceInfo {
        description: String,
        sample_rate: u32,
    },
    Stopped,
    AlignmentStarted {
        total: usize,
    },
    AlignmentProgress {
        current: usize,
    },
    AlignmentReady(Box<CachedAlignment>),
    AlignmentFailed(String),
    /// Raw payload bytes from a `DecodedFrame::Geoscan` frame, forwarded
    /// to the GUI's image reassembler. Sent for every Geoscan frame so
    /// the GUI side decides whether to ingest based on its own
    /// reassembler state.
    ImagePayload(Vec<u8>),
    /// One fully decoded SSTV image. Sent by the `RuntimeInput::Sstv`
    /// worker arm whenever `SstvAnalyzer::drain_images` produces a
    /// frame; the GUI replaces `latest_sstv` and refreshes the image
    /// tab.
    SstvImage(Box<openhoshimi_codec::SstvImage>),
    /// Cumulative pipeline counters (samples in, demodulated bits, sync
    /// attempts / locks, frames emitted) sampled from `BitPipeline` after
    /// each `push_samples` call. Surfaced in the status bar so the
    /// operator can see, while a recording is running, why frames are
    /// not coming through (no bits at all? syncword never matching?
    /// matches but no frames clearing CRC?).
    PipelineStats(PipelineStats),
}

#[derive(Clone)]
struct CachedAlignment {
    path: PathBuf,
    sample_rate: u32,
    downlink_id: String,
    prefix: Vec<IqSample>,
    setup: openhoshimi_runtime::pipeline::LinearIqSetup,
    frames: usize,
}

#[derive(Debug, Clone)]
enum AlignmentState {
    Idle,
    Aligning { current: usize, total: usize },
    Ready,
    Failed,
}

enum SpectrumSamples {
    Audio(Vec<f32>),
    Iq(Vec<IqSample>),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DiagnosticLevel {
    Info,
    Warn,
    Error,
}

#[derive(Debug, Clone)]
struct DiagnosticEntry {
    level: DiagnosticLevel,
    text: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum InputMode {
    Soundcard,
    WavFile,
}

/// Decoder pacing for file inputs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DecodePace {
    /// Throttle WAV reads to wall-clock real time so the waterfall scrolls in sync.
    Realtime,
    /// Drain the WAV as fast as possible (matches the CLI baseline).
    Fast,
}

impl OpenHoshimiApp {
    fn new(cc: &eframe::CreationContext<'_>) -> Self {
        configure_fonts(&cc.egui_ctx);
        configure_style(&cc.egui_ctx);

        let satellites_dir = resolve_satellites_dir();
        let startup_wav = std::env::var_os("OPENHOSHIMI_OPEN_WAV").map(PathBuf::from);
        let (rx_sender, rx_events) = mpsc::sync_channel(RX_EVENT_QUEUE);
        let mut app = Self {
            satellites_dir,
            satellites: Vec::new(),
            selected_satellite: 0,
            selected_downlink: 0,
            selected_frame: None,
            filter: String::new(),
            display_filter: String::new(),
            offset_hz: 0,
            input_rate: 48_000,
            running: false,
            status: String::new(),
            source_description: "soundcard default".to_string(),
            input_mode: InputMode::Soundcard,
            decode_pace: DecodePace::Fast,
            input_path: None,
            frames: VecDeque::new(),
            dropped_frames: 0,
            samples_processed: 0,
            input_progress_processed: 0,
            input_progress_total: None,
            frames_seen: 0,
            diagnostics: VecDeque::new(),
            waterfall: VecDeque::new(),
            spectrum: SpectrumProcessor::new(SPECTRUM_FFT_LEN),
            waterfall_texture: None,
            waterfall_min_db: DEFAULT_WATERFALL_MIN_DB,
            waterfall_max_db: DEFAULT_WATERFALL_MAX_DB,
            rx_thread: None,
            rx_stop: None,
            rx_events,
            rx_sender,
            startup_wav,
            last_export_status: None,
            alignment_state: AlignmentState::Idle,
            cached_alignment: None,
            alignment_thread: None,
            soundcard_devices: enumerate_input_devices(),
            selected_soundcard: None,
            audio_mode: IoAudioMode::Auto,
            audio_carrier_hz: 0.0,
            bottom_tab: BottomTab::Frames,
            image_reassembler: None,
            image_texture: None,
            image_texture_for_idx: None,
            image_dirty: false,
            image_active_idx: 0,
            image_payload_misses: 0,
            image_payload_hits: 0,
            image_payload_logged: false,
            latest_sstv: None,
            sstv_texture: None,
            sstv_dirty: false,
            latest_stats: None,
        };
        app.reload_satellites();
        app.refresh_image_reassembler();
        app
    }

    fn reload_satellites(&mut self) {
        match load_all_satellites(&self.satellites_dir) {
            Ok(mut satellites) => {
                satellites.sort_by(|a, b| a.satellite.name.cmp(&b.satellite.name));
                self.status = format!("loaded {} satellite definitions", satellites.len());
                self.push_diagnostic(
                    DiagnosticLevel::Info,
                    format!("loaded {} satellite definitions", satellites.len()),
                );
                self.satellites = satellites;
                self.selected_satellite = self
                    .selected_satellite
                    .min(self.satellites.len().saturating_sub(1));
                self.select_first_supported_downlink();
                if let Some(satellite) = self.selected_satellite() {
                    self.push_diagnostic(
                        DiagnosticLevel::Info,
                        format!("selected satellite {}", satellite.satellite.name),
                    );
                }
                self.push_diagnostic(
                    DiagnosticLevel::Info,
                    "selected first supported downlink".to_string(),
                );
                if let Some(downlink) = self.selected_downlink() {
                    self.push_diagnostic(
                        DiagnosticLevel::Info,
                        format!("selected downlink {}", downlink_combo_label(downlink)),
                    );
                }
            }
            Err(err) => {
                self.status = "failed to load satellites".to_string();
                self.push_diagnostic(
                    DiagnosticLevel::Error,
                    format!(
                        "failed to load satellites from {}: {err}",
                        self.satellites_dir.display()
                    ),
                );
                self.satellites.clear();
                self.selected_satellite = 0;
                self.selected_downlink = 0;
            }
        }
    }

    fn selected_satellite(&self) -> Option<&SatelliteDefinition> {
        self.satellites.get(self.selected_satellite)
    }

    fn selected_downlink(&self) -> Option<&DownlinkDef> {
        self.selected_satellite()
            .and_then(|satellite| satellite.downlinks.get(self.selected_downlink))
    }

    fn selected_downlink_cloned(&self) -> Option<DownlinkDef> {
        self.selected_downlink().cloned()
    }

    fn select_first_supported_downlink(&mut self) {
        let Some(satellite) = self.selected_satellite() else {
            self.selected_downlink = 0;
            return;
        };
        self.selected_downlink = satellite
            .downlinks
            .iter()
            .position(downlink_is_supported)
            .unwrap_or(0);
    }

    fn start(&mut self) {
        if self.running {
            return;
        }
        let Some(satellite) = self.selected_satellite().cloned() else {
            self.status = "no satellite selected".to_string();
            return;
        };
        let Some(selected) = self.selected_downlink_cloned() else {
            self.status = "no downlink selected".to_string();
            return;
        };
        if !downlink_is_supported(&selected) {
            self.status = format!("selected downlink is not supported: {}", selected.label);
            self.push_diagnostic(DiagnosticLevel::Error, self.status.clone());
            return;
        }
        if matches!(self.input_mode, InputMode::WavFile) && self.input_path.is_none() {
            self.status = "no WAV file selected".to_string();
            self.push_diagnostic(DiagnosticLevel::Warn, self.status.clone());
            return;
        }

        let (stop_tx, stop_rx) = mpsc::channel();
        let events = self.rx_sender.clone();
        let input_mode = self.input_mode;
        let decode_pace = self.decode_pace;
        let input_path = self.input_path.clone();
        let soundcard_device = self.selected_soundcard.clone();
        let selected_label = selected.label.clone();
        let tuning_offset_hz = self.tuning_offset_for_selected(&selected, input_path.as_deref());
        let cached_alignment = self.cached_alignment.clone();
        let audio_mode = self.audio_mode;
        let audio_carrier_hz = self.audio_carrier_hz;
        let thread = thread::spawn(move || {
            run_decode_thread(
                satellite,
                selected,
                input_mode,
                decode_pace,
                input_path,
                soundcard_device,
                tuning_offset_hz,
                cached_alignment,
                audio_mode,
                audio_carrier_hz,
                events,
                stop_rx,
            )
        });
        self.rx_thread = Some(thread);
        self.rx_stop = Some(stop_tx);
        self.running = true;
        self.input_progress_processed = 0;
        self.input_progress_total = None;
        self.status = "RX started".to_string();
        self.push_diagnostic(
            DiagnosticLevel::Info,
            format!(
                "started {} with {}",
                selected_label,
                self.input_source_label()
            ),
        );
    }

    fn stop(&mut self) {
        if !self.running && self.rx_thread.is_none() && self.rx_stop.is_none() {
            return;
        }
        if let Some(stop) = self.rx_stop.take() {
            let _ = stop.send(());
        }
        if let Some(thread) = self.rx_thread.take() {
            let _ = thread.join();
        }
        self.running = false;
        self.status = "RX stopped".to_string();
        self.push_diagnostic(DiagnosticLevel::Info, self.status.clone());
    }

    fn poll_events(&mut self) {
        let mut processed = 0usize;
        let mut latest_samples: Option<SpectrumSamples> = None;
        let mut latest_count = 0usize;
        while processed < 128 {
            let Ok(event) = self.rx_events.try_recv() else {
                break;
            };
            processed += 1;
            match event {
                RxEvent::Frame(frame) => {
                    self.frames_seen += 1;
                    if self.selected_frame.is_none() {
                        self.selected_frame = Some(frame.index);
                    }
                    self.frames.push_back(frame);
                    if self.frames.len() > MAX_FRAMES {
                        self.frames.pop_front();
                    }
                }
                RxEvent::Dropped(reason) => {
                    self.dropped_frames += 1;
                    self.status = reason;
                    self.push_diagnostic(DiagnosticLevel::Warn, self.status.clone());
                }
                RxEvent::Samples(samples) => {
                    latest_count += spectrum_samples_len(&samples);
                    latest_samples = Some(samples);
                }
                RxEvent::Progress { processed, total } => {
                    self.input_progress_processed = processed;
                    self.input_progress_total = total;
                    self.status = match total {
                        Some(total) => format!("progress {} / {}", processed, total),
                        None => format!("progress {}", processed),
                    };
                }
                RxEvent::SourceInfo {
                    description,
                    sample_rate,
                } => {
                    self.input_rate = sample_rate;
                    self.source_description = status_path_label(&description);
                    self.status = self.source_description.clone();
                    self.push_diagnostic(DiagnosticLevel::Info, self.status.clone());
                }
                RxEvent::Stopped => {
                    self.running = false;
                    self.rx_thread = None;
                    self.rx_stop = None;
                    self.status = "RX stopped".to_string();
                    self.push_diagnostic(DiagnosticLevel::Info, self.status.clone());
                }
                RxEvent::AlignmentStarted { total } => {
                    self.alignment_state = AlignmentState::Aligning { current: 0, total };
                    self.status = format!("aligning IQ ({total} candidates)");
                    self.push_diagnostic(DiagnosticLevel::Info, self.status.clone());
                }
                RxEvent::AlignmentProgress { current } => {
                    if let AlignmentState::Aligning { total, .. } = self.alignment_state {
                        self.alignment_state = AlignmentState::Aligning { current, total };
                    }
                }
                RxEvent::AlignmentReady(cached) => {
                    self.status = format!(
                        "IQ alignment ready: tuning={:.1} Hz skip={} prefix_frames={}",
                        cached.setup.tuning_offset_hz, cached.setup.sample_skip, cached.frames
                    );
                    self.push_diagnostic(DiagnosticLevel::Info, self.status.clone());
                    self.cached_alignment = Some(*cached);
                    self.alignment_state = AlignmentState::Ready;
                    if let Some(handle) = self.alignment_thread.take() {
                        let _ = handle.join();
                    }
                }
                RxEvent::AlignmentFailed(reason) => {
                    self.status = format!("IQ alignment failed: {reason}");
                    self.push_diagnostic(DiagnosticLevel::Error, self.status.clone());
                    self.alignment_state = AlignmentState::Failed;
                    self.cached_alignment = None;
                    if let Some(handle) = self.alignment_thread.take() {
                        let _ = handle.join();
                    }
                }
                RxEvent::ImagePayload(bytes) => {
                    if let Some(reassembler) = self.image_reassembler.as_mut() {
                        if reassembler.ingest(&bytes).is_some() {
                            self.image_dirty = true;
                            self.image_payload_hits = self.image_payload_hits.saturating_add(1);
                        } else {
                            self.image_payload_misses = self.image_payload_misses.saturating_add(1);
                            if !self.image_payload_logged {
                                self.image_payload_logged = true;
                                let take = bytes.len().min(16);
                                let prefix: Vec<String> =
                                    bytes[..take].iter().map(|b| format!("{:02X}", b)).collect();
                                self.push_diagnostic(
                                    DiagnosticLevel::Info,
                                    format!(
                                        "image: payload did not match header signature; \
                                         len={} prefix=[{}]",
                                        bytes.len(),
                                        prefix.join(" "),
                                    ),
                                );
                            }
                        }
                    }
                }
                RxEvent::PipelineStats(stats) => {
                    self.latest_stats = Some(stats);
                }
                RxEvent::SstvImage(image) => {
                    self.push_diagnostic(
                        DiagnosticLevel::Info,
                        format!(
                            "SSTV: decoded {:?} image ({}x{})",
                            image.mode, image.width, image.height
                        ),
                    );
                    self.latest_sstv = Some(*image);
                    self.sstv_dirty = true;
                    self.sstv_texture = None;
                    if matches!(self.bottom_tab, BottomTab::Frames) {
                        self.bottom_tab = BottomTab::Image;
                    }
                }
            }
        }
        if let Some(samples) = latest_samples {
            self.samples_processed += latest_count as u64;
            self.push_waterfall_row(samples);
        }
    }

    fn push_waterfall_row(&mut self, samples: SpectrumSamples) {
        let span_hz = self.spectrum_span_hz();
        let row = match samples {
            SpectrumSamples::Audio(samples) => {
                self.spectrum
                    .row_audio(&samples, SPECTRUM_BINS, self.input_rate, span_hz)
            }
            SpectrumSamples::Iq(samples) => {
                self.spectrum
                    .row_iq(&samples, SPECTRUM_BINS, self.input_rate, span_hz)
            }
        };
        self.waterfall.push_back(row);
        while self.waterfall.len() > MAX_WATERFALL_ROWS {
            self.waterfall.pop_front();
        }
    }

    fn clear_frames(&mut self) {
        self.frames.clear();
        self.selected_frame = None;
        self.dropped_frames = 0;
        self.frames_seen = 0;
        self.samples_processed = 0;
        self.input_progress_processed = 0;
        self.input_progress_total = None;
        self.latest_stats = None;
    }

    fn export_frames(&mut self) {
        if self.frames.is_empty() {
            self.status = "no frames to export".to_string();
            self.last_export_status = Some("no frames".to_string());
            return;
        }
        let default_name = match self.selected_satellite() {
            Some(sat) => format!(
                "openhoshimi_{}.txt",
                sat.satellite.name.replace([' ', '/'], "_")
            ),
            None => "openhoshimi_frames.txt".to_string(),
        };
        let Some(path) = rfd::FileDialog::new()
            .set_file_name(default_name)
            .add_filter("Text", &["txt"])
            .save_file()
        else {
            return;
        };

        let mut text = String::new();
        text.push_str("# OpenHoshimi frame export\n");
        if let Some(sat) = self.selected_satellite() {
            text.push_str(&format!("# satellite: {}\n", sat.satellite.name));
            text.push_str(&format!("# norad_id: {}\n", sat.satellite.norad_id));
        }
        text.push_str(&format!("# frames: {}\n\n", self.frames.len()));

        for row in &self.frames {
            text.push_str(&format!("# Frame {}\n", row.index));
            text.push_str(&format!("time: {:.3}s\n", row.time.as_secs_f64()));
            text.push_str(&format!("source: {}\n", row.source));
            text.push_str(&format!("destination: {}\n", row.destination));
            text.push_str(&format!("kind: {:?}\n", row.kind));
            if let Some(rssi) = row.rssi_dbm {
                text.push_str(&format!("rssi_dbm: {rssi:.1}\n"));
            }
            text.push_str(&format!("raw: {}\n", format_hex(&row.raw)));
            if !row.telemetry.is_empty() {
                text.push_str("telemetry:\n");
                for field in &row.telemetry {
                    text.push_str(&format!(
                        "  {}.{}: {}\n",
                        field.group,
                        field.key,
                        format_telemetry(field)
                    ));
                }
            }
            text.push('\n');
        }

        match std::fs::write(&path, text) {
            Ok(()) => {
                let count = self.frames.len();
                let label = short_path_label(&path);
                self.status = format!("exported {count} frames to {label}");
                self.last_export_status = Some(format!("exported {count} frames"));
            }
            Err(err) => {
                self.status = format!("export failed: {err}");
                self.last_export_status = Some("export failed".to_string());
            }
        }
    }

    fn push_diagnostic(&mut self, level: DiagnosticLevel, text: String) {
        self.diagnostics.push_back(DiagnosticEntry { level, text });
        while self.diagnostics.len() > MAX_DIAGNOSTICS {
            self.diagnostics.pop_front();
        }
    }

    fn clear_diagnostics(&mut self) {
        self.diagnostics.clear();
    }

    fn diagnostic_counts(&self) -> (usize, usize, usize) {
        let mut info = 0usize;
        let mut warn = 0usize;
        let mut error = 0usize;
        for entry in &self.diagnostics {
            match entry.level {
                DiagnosticLevel::Info => info += 1,
                DiagnosticLevel::Warn => warn += 1,
                DiagnosticLevel::Error => error += 1,
            }
        }
        (info, warn, error)
    }

    fn selected_frame_row(&self) -> Option<&FrameRow> {
        let selected = self.selected_frame?;
        self.frames.iter().find(|row| row.index == selected)
    }
}

impl Drop for OpenHoshimiApp {
    fn drop(&mut self) {
        self.stop();
    }
}

impl eframe::App for OpenHoshimiApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        if let Some(path) = self.startup_wav.take() {
            self.open_wav_path(path);
        }
        self.poll_events();
        if self.running {
            ctx.request_repaint_after(Duration::from_millis(16));
        }

        egui::TopBottomPanel::top("menu_bar")
            .exact_height(22.0)
            .frame(panel_frame(Palette::BAR))
            .show(ctx, |ui| self.menu_bar(ui));

        egui::TopBottomPanel::top("toolbar")
            .exact_height(32.0)
            .frame(panel_frame(Palette::BAR))
            .show(ctx, |ui| self.toolbar(ui));

        egui::TopBottomPanel::bottom("status_bar")
            .exact_height(24.0)
            .frame(panel_frame(Palette::BAR))
            .show(ctx, |ui| self.status_bar(ui));

        egui::SidePanel::left("satellites_panel")
            .exact_width(160.0)
            .resizable(false)
            .frame(Frame::new().fill(Palette::BG))
            .show_separator_line(true)
            .show(ctx, |ui| self.left_satellites(ui));

        egui::SidePanel::right("right_panel_side")
            .exact_width(220.0)
            .resizable(false)
            .frame(Frame::new().fill(Palette::BG))
            .show_separator_line(true)
            .show(ctx, |ui| self.right_panel(ui));

        egui::CentralPanel::default()
            .frame(Frame::new().fill(Palette::BG))
            .show(ctx, |ui| self.center_workspace(ui));

        self.alignment_modal(ctx);
    }
}

impl OpenHoshimiApp {
    fn alignment_modal(&self, ctx: &egui::Context) {
        let (current, total) = match self.alignment_state {
            AlignmentState::Aligning { current, total } => (current, total),
            _ => return,
        };
        let denom = total.max(1);
        let fraction = (current as f32) / (denom as f32);
        egui::Window::new("alignment_modal")
            .title_bar(false)
            .resizable(false)
            .movable(false)
            .collapsible(false)
            .anchor(egui::Align2::CENTER_CENTER, [0.0, 0.0])
            .frame(panel_frame(Palette::BAR))
            .show(ctx, |ui| {
                ui.set_min_width(320.0);
                ui.vertical_centered(|ui| {
                    ui.add_space(6.0);
                    ui.label(
                        RichText::new("Aligning IQ")
                            .color(Palette::TEXT)
                            .size(14.0)
                            .strong(),
                    );
                    ui.add_space(8.0);
                    ui.add(
                        egui::ProgressBar::new(fraction)
                            .desired_width(280.0)
                            .text(format!("{current} / {total}")),
                    );
                    ui.add_space(6.0);
                    ui.label(
                        RichText::new("Searching tuning offset and sample skip\u{2026}")
                            .color(Palette::MUTED)
                            .size(11.0),
                    );
                    ui.add_space(4.0);
                });
            });
        ctx.request_repaint_after(Duration::from_millis(80));
    }

    fn menu_bar(&mut self, ui: &mut egui::Ui) {
        ui.spacing_mut().item_spacing = Vec2::new(14.0, 0.0);
        egui::menu::bar(ui, |ui| {
            ui.menu_button("File", |ui| {
                if ui.button("Open WAV...").clicked() {
                    ui.close_menu();
                    self.open_wav();
                }
                if ui.button("Reload satellites").clicked() {
                    ui.close_menu();
                    self.reload_satellites();
                }
                if ui.button("Quit").clicked() {
                    ui.close_menu();
                    ui.ctx().send_viewport_cmd(egui::ViewportCommand::Close);
                }
            });
            ui.menu_button("View", |ui| {
                if ui.button("Reset waterfall").clicked() {
                    ui.close_menu();
                    self.waterfall.clear();
                    self.samples_processed = 0;
                }
                if ui.button("Clear diagnostics").clicked() {
                    ui.close_menu();
                    self.clear_diagnostics();
                }
            });
            ui.menu_button("Input", |ui| {
                if ui.button("Use soundcard").clicked() {
                    ui.close_menu();
                    self.stop();
                    self.input_mode = InputMode::Soundcard;
                    self.source_description = "soundcard".to_string();
                    self.status = "selected soundcard input".to_string();
                }
                if ui.button("Open WAV...").clicked() {
                    ui.close_menu();
                    self.open_wav();
                }
            });
            ui.menu_button("Satellites", |ui| {
                if ui.button("Reload definitions").clicked() {
                    ui.close_menu();
                    self.reload_satellites();
                }
            });
            ui.menu_button("Decode", |ui| {
                let can_start = !self.running && self.selected_downlink().is_some();
                if ui
                    .add_enabled(can_start, egui::Button::new("Start"))
                    .clicked()
                {
                    ui.close_menu();
                    self.start();
                }
                if ui
                    .add_enabled(self.running, egui::Button::new("Stop"))
                    .clicked()
                {
                    ui.close_menu();
                    self.stop();
                }
                if ui.button("Reset").clicked() {
                    ui.close_menu();
                    self.stop();
                    self.clear_frames();
                    self.waterfall.clear();
                }
            });
            ui.menu_button("Tools", |ui| {
                let can_export = !self.frames.is_empty();
                if ui
                    .add_enabled(can_export, egui::Button::new("Export frames..."))
                    .clicked()
                {
                    ui.close_menu();
                    self.export_frames();
                }
                if ui.button("Clear frames").clicked() {
                    ui.close_menu();
                    self.clear_frames();
                }
            });
            ui.menu_button("Help", |ui| {
                ui.label(
                    RichText::new(concat!("OpenHoshimi ", env!("CARGO_PKG_VERSION")))
                        .color(Palette::TEXT)
                        .monospace(),
                );
                ui.label(
                    RichText::new("Open-source amateur satellite decoder.")
                        .color(Palette::MUTED)
                        .monospace()
                        .size(10.0),
                );
                if ui.button("Close").clicked() {
                    ui.close_menu();
                }
            });
        });
    }

    fn toolbar(&mut self, ui: &mut egui::Ui) {
        ui.spacing_mut().item_spacing = Vec2::new(6.0, 0.0);
        ui.horizontal_centered(|ui| {
            label_muted(ui, "Input:");
            let prev_input_mode = self.input_mode;
            egui::ComboBox::from_id_salt("input_source")
                .selected_text(self.input_source_label())
                .width(180.0)
                .show_ui(ui, |ui| {
                    ui.selectable_value(&mut self.input_mode, InputMode::Soundcard, "Soundcard");
                    ui.selectable_value(&mut self.input_mode, InputMode::WavFile, "Audio file");
                });
            if self.input_mode != prev_input_mode {
                self.stop();
            }
            if matches!(self.input_mode, InputMode::Soundcard) {
                self.soundcard_device_combo(ui);
            }
            if matches!(self.input_mode, InputMode::WavFile)
                && bracket_button(ui, "Open audio...", 180.0).clicked()
            {
                self.open_wav();
            }

            label_muted(ui, "Downlink:");
            self.downlink_combo(ui);

            let rate_enabled = !matches!(self.input_mode, InputMode::Soundcard);
            ui.add_enabled_ui(rate_enabled, |ui| {
                label_muted(ui, "Rate:");
                egui::ComboBox::from_id_salt("input_rate")
                    .selected_text(self.input_rate.to_string())
                    .width(70.0)
                    .show_ui(ui, |ui| {
                        for rate in [4_800, 9_600, 48_000, 96_000] {
                            ui.selectable_value(&mut self.input_rate, rate, rate.to_string());
                        }
                    });
            });

            label_muted(ui, "Offset:");
            let offset_prefix = if self.offset_hz >= 0 { "+" } else { "" };
            ui.add_sized(
                [56.0, 22.0],
                egui::DragValue::new(&mut self.offset_hz)
                    .speed(1)
                    .range(-50_000..=50_000)
                    .prefix(offset_prefix),
            );
            label_muted(ui, "Hz");

            // Audio mode is selectable for IQ/FmAudio downlinks in any input
            // mode.  In soundcard mode this lets the user force FM-audio
            // demodulation for mono virtual soundcard feeds (SDR# FM out).
            let audio_mode_enabled = matches!(
                self.selected_downlink().and_then(input_kind_for),
                Some(InputKind::Iq) | Some(InputKind::FmAudio)
            );
            label_muted(ui, "Audio:");
            let prev_audio_mode = self.audio_mode;
            ui.add_enabled_ui(audio_mode_enabled && !self.running, |ui| {
                let label = match self.audio_mode {
                    IoAudioMode::Auto => "Auto",
                    IoAudioMode::Iq => "IQ",
                    IoAudioMode::Fm => "FM",
                    IoAudioMode::Ssb => "SSB",
                };
                egui::ComboBox::from_id_salt("audio_mode")
                    .selected_text(label)
                    .width(80.0)
                    .show_ui(ui, |ui| {
                        ui.selectable_value(&mut self.audio_mode, IoAudioMode::Auto, "Auto");
                        ui.selectable_value(&mut self.audio_mode, IoAudioMode::Iq, "IQ");
                        ui.selectable_value(&mut self.audio_mode, IoAudioMode::Fm, "FM");
                        ui.selectable_value(&mut self.audio_mode, IoAudioMode::Ssb, "SSB");
                    });
            });
            if prev_audio_mode != self.audio_mode && self.input_path.is_some() {
                self.spawn_alignment();
            }
            if matches!(self.audio_mode, IoAudioMode::Ssb) {
                label_muted(ui, "Carrier:");
                ui.add_enabled_ui(audio_mode_enabled && !self.running, |ui| {
                    ui.add_sized(
                        [70.0, 22.0],
                        egui::DragValue::new(&mut self.audio_carrier_hz)
                            .speed(10.0)
                            .range(0.0..=24_000.0)
                            .suffix(" Hz"),
                    );
                });
            }

            let pace_enabled = matches!(self.input_mode, InputMode::WavFile);
            label_muted(ui, "Pace:");
            ui.add_enabled_ui(pace_enabled, |ui| {
                ui.selectable_value(&mut self.decode_pace, DecodePace::Fast, "[Fast]");
                ui.selectable_value(&mut self.decode_pace, DecodePace::Realtime, "[Realtime]");
            });
            let can_start = !self.running && self.selected_downlink().is_some();
            ui.add_enabled_ui(can_start, |ui| {
                if bracket_button(ui, "Start", 70.0).clicked() {
                    self.start();
                }
            });
            ui.add_enabled_ui(self.running, |ui| {
                if bracket_button(ui, "Stop", 70.0).clicked() {
                    self.stop();
                }
            });
            if bracket_button(ui, "Reset", 70.0).clicked() {
                self.stop();
                self.clear_frames();
                self.waterfall.clear();
                self.samples_processed = 0;
                self.input_progress_processed = 0;
                self.input_progress_total = None;
            }
        });
    }

    fn status_bar(&mut self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            status_segment(
                ui,
                if self.running { "RX" } else { "IDLE" },
                if self.running {
                    Palette::GREEN
                } else {
                    Palette::MUTED
                },
            );
            if let Some(sat) = self.selected_satellite() {
                status_segment(ui, &sat_label(sat), Palette::MUTED);
            }
            status_segment(ui, "el --", Palette::MUTED);
            status_segment(ui, &self.frequency_label(), Palette::MUTED);
            status_segment(
                ui,
                &format!("span +/-{} Hz", self.spectrum_span_hz()),
                Palette::MUTED,
            );
            status_segment(
                ui,
                &format!("wf: {}k", self.samples_processed / 1000),
                Palette::MUTED,
            );
            if let Some((done, total)) = self.current_progress() {
                let percent = done
                    .saturating_mul(100)
                    .checked_div(total)
                    .map(|p| p.min(100))
                    .unwrap_or(0);
                let progress = if total == 0 {
                    0.0
                } else {
                    (done as f32 / total as f32).clamp(0.0, 1.0)
                };
                status_segment(
                    ui,
                    &format!("play {} / {} ({}%)", done, total, percent),
                    Palette::MUTED,
                );
                ui.add_sized([96.0, 12.0], ProgressBar::new(progress).show_percentage());
            }
            status_segment(ui, &format!("frames: {}", self.frames_seen), Palette::MUTED);
            status_segment(
                ui,
                &format!("drop: {}", self.dropped_frames),
                Palette::MUTED,
            );
            if let Some(stats) = self.latest_stats {
                status_segment(ui, &format!("bits: {}", stats.total_bits), Palette::MUTED);
                if let (Some(locked), Some(attempts)) = (stats.sync_locked, stats.sync_attempts) {
                    let color = if locked > 0 {
                        Palette::GREEN
                    } else {
                        Palette::MUTED
                    };
                    status_segment(ui, &format!("sync: {}/{}", locked, attempts), color);
                }
                status_segment(
                    ui,
                    &format!("emit: {}", stats.frames_emitted),
                    Palette::MUTED,
                );
                if let (Some(crc_ok), Some(crc_fail)) = (stats.crc_ok, stats.crc_fail) {
                    let total = crc_ok.saturating_add(crc_fail);
                    let color = if total == 0 {
                        Palette::MUTED
                    } else if crc_ok.saturating_mul(2) >= total {
                        Palette::GREEN
                    } else {
                        Palette::RED
                    };
                    status_segment(ui, &format!("crc: {}/{}", crc_ok, total), color);
                }
            }
            if !self.status.is_empty() {
                let status_lower = self.status.to_ascii_lowercase();
                let color = if status_lower.contains("failed")
                    || status_lower.contains("error")
                    || status_lower.contains("unsupported")
                    || status_lower.contains("requires")
                    || status_lower.contains("missing")
                {
                    Palette::RED
                } else {
                    Palette::MUTED
                };
                status_segment(ui, &self.status, color);
            }
            status_segment(ui, &self.source_description, Palette::MUTED);
            ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                ui.label(
                    RichText::new(concat!("OpenHoshimi ", env!("CARGO_PKG_VERSION")))
                        .color(Palette::MUTED)
                        .monospace()
                        .size(11.0),
                );
            });
        });
    }

    fn current_progress(&self) -> Option<(u64, u64)> {
        self.input_progress_total
            .map(|total| (self.input_progress_processed, total))
    }

    fn left_satellites(&mut self, ui: &mut egui::Ui) {
        header(ui, &format!("Satellites ({})", self.satellites.len()));
        ui.add_sized(
            [ui.available_width(), 22.0],
            TextEdit::singleline(&mut self.filter)
                .hint_text("filter name / NORAD")
                .frame(false),
        );
        hline(ui);

        ScrollArea::vertical()
            .id_salt("satellites_scroll")
            .auto_shrink([false, false])
            .max_height(ui.available_height())
            .show(ui, |ui| {
                let filter = self.filter.to_ascii_lowercase();
                if self.satellites.is_empty() {
                    label_muted(ui, "-- no satellite definitions loaded --");
                }
                for index in 0..self.satellites.len() {
                    let sat = &self.satellites[index];
                    if !filter.is_empty()
                        && !sat.satellite.name.to_ascii_lowercase().contains(&filter)
                        && !sat.satellite.norad_id.to_string().contains(&filter)
                    {
                        continue;
                    }
                    let satellite_name = sat.satellite.name.clone();
                    let norad_id = sat.satellite.norad_id.to_string();
                    let selected = index == self.selected_satellite;
                    let rect = row_rect(ui, 24.0);
                    let response = ui.interact(rect, ui.id().with(("sat", index)), Sense::click());
                    if response.clicked() {
                        self.stop();
                        self.selected_satellite = index;
                        self.select_first_supported_downlink();
                        self.clear_frames();
                        self.waterfall.clear();
                        self.samples_processed = 0;
                        self.input_progress_processed = 0;
                        self.input_progress_total = None;
                        self.refresh_image_reassembler();
                        self.push_diagnostic(
                            DiagnosticLevel::Info,
                            format!("selected satellite {}", satellite_name),
                        );
                        if let Some(downlink) = self.selected_downlink() {
                            self.push_diagnostic(
                                DiagnosticLevel::Info,
                                format!("selected downlink {}", downlink_combo_label(downlink)),
                            );
                        }
                    }
                    let painter = ui.painter();
                    painter.rect_filled(
                        rect,
                        CornerRadius::ZERO,
                        if selected { Palette::BLUE } else { Palette::BG },
                    );
                    painter.line_segment(
                        [rect.left_bottom(), rect.right_bottom()],
                        Stroke::new(1.0, Palette::LINE),
                    );

                    painter.text(
                        Pos2::new(rect.left() + 4.0, rect.center().y),
                        egui::Align2::LEFT_CENTER,
                        "[?]",
                        FontId::monospace(11.0),
                        Palette::MUTED,
                    );
                    painter.text(
                        Pos2::new(rect.left() + 34.0, rect.center().y),
                        egui::Align2::LEFT_CENTER,
                        &satellite_name,
                        FontId::monospace(11.0),
                        if selected {
                            Color32::WHITE
                        } else {
                            Palette::TEXT
                        },
                    );
                    painter.text(
                        Pos2::new(rect.right() - 4.0, rect.center().y),
                        egui::Align2::RIGHT_CENTER,
                        norad_id,
                        FontId::monospace(10.0),
                        Palette::MUTED,
                    );
                }
            });
    }

    fn center_workspace(&mut self, ui: &mut egui::Ui) {
        self.spectrum(ui);
        self.frame_toolbar(ui);
        let has_image_view = self.image_reassembler.is_some()
            || self
                .selected_downlink()
                .is_some_and(|d| matches!(d.image, Some(ImageDef::Sstv {})));
        if has_image_view {
            self.bottom_tab_bar(ui);
            match self.bottom_tab {
                BottomTab::Frames => {
                    self.frame_table(ui);
                    self.hex_dump(ui);
                }
                BottomTab::Image => {
                    self.image_view(ui);
                }
            }
        } else {
            self.frame_table(ui);
            self.hex_dump(ui);
        }
    }

    /// Render the [Frames] [Image] tab strip above the lower region of
    /// the middle column. Only called when the active downlink declares
    /// an image-reassembly stage; satellites without image support skip
    /// this row entirely.
    fn bottom_tab_bar(&mut self, ui: &mut egui::Ui) {
        ui.allocate_ui_with_layout(
            Vec2::new(ui.available_width(), 22.0),
            Layout::left_to_right(Align::Center),
            |ui| {
                ui.painter()
                    .rect_filled(ui.max_rect(), CornerRadius::ZERO, Palette::PANEL);
                ui.add_space(6.0);
                let frames_active = matches!(self.bottom_tab, BottomTab::Frames);
                let image_active = matches!(self.bottom_tab, BottomTab::Image);
                if tab_button(ui, "Frames", frames_active).clicked() {
                    self.bottom_tab = BottomTab::Frames;
                }
                ui.add_space(4.0);
                if tab_button(ui, "Image", image_active).clicked() {
                    self.bottom_tab = BottomTab::Image;
                }
            },
        );
    }

    fn spectrum(&mut self, ui: &mut egui::Ui) {
        ui.allocate_ui_with_layout(
            Vec2::new(ui.available_width(), 22.0),
            Layout::left_to_right(Align::Center),
            |ui| {
                ui.painter()
                    .rect_filled(ui.max_rect(), CornerRadius::ZERO, Palette::PANEL);
                ui.add_space(6.0);
                label_muted(ui, "min dB");
                ui.add(
                    egui::Slider::new(
                        &mut self.waterfall_min_db,
                        WATERFALL_DB_FLOOR..=WATERFALL_DB_CEIL,
                    )
                    .show_value(true)
                    .fixed_decimals(0)
                    .step_by(1.0),
                );
                label_muted(ui, "max dB");
                ui.add(
                    egui::Slider::new(
                        &mut self.waterfall_max_db,
                        WATERFALL_DB_FLOOR..=WATERFALL_DB_CEIL,
                    )
                    .show_value(true)
                    .fixed_decimals(0)
                    .step_by(1.0),
                );
                if bracket_button(ui, "reset", 60.0).clicked() {
                    self.waterfall_min_db = DEFAULT_WATERFALL_MIN_DB;
                    self.waterfall_max_db = DEFAULT_WATERFALL_MAX_DB;
                }
            },
        );
        if self.waterfall_max_db <= self.waterfall_min_db + 1.0 {
            self.waterfall_max_db = (self.waterfall_min_db + 1.0).min(WATERFALL_DB_CEIL);
        }
        let desired = Vec2::new(ui.available_width(), WATERFALL_HEIGHT_PX);
        let (rect, _) = ui.allocate_exact_size(desired, Sense::hover());
        let painter = ui.painter();
        painter.rect_filled(rect, CornerRadius::ZERO, Color32::BLACK);

        if let Some(texture_id) = self.waterfall_texture_id(ui.ctx()) {
            painter.image(
                texture_id,
                rect,
                Rect::from_min_max(Pos2::new(0.0, 0.0), Pos2::new(1.0, 1.0)),
                Color32::WHITE,
            );
        }

        painter.text(
            rect.left_top() + Vec2::new(6.0, 6.0),
            egui::Align2::LEFT_TOP,
            self.spectrum_label(),
            FontId::monospace(10.0),
            Color32::WHITE,
        );
        painter.text(
            Pos2::new(rect.center().x, rect.bottom() - 12.0),
            egui::Align2::CENTER_CENTER,
            self.spectrum_scale_label(),
            FontId::monospace(10.0),
            Palette::MUTED,
        );
    }

    fn frame_toolbar(&mut self, ui: &mut egui::Ui) {
        ui.allocate_ui_with_layout(
            Vec2::new(ui.available_width(), 24.0),
            Layout::left_to_right(Align::Center),
            |ui| {
                ui.painter()
                    .rect_filled(ui.max_rect(), CornerRadius::ZERO, Palette::PANEL);
                ui.label(
                    RichText::new("Frame list")
                        .color(Palette::MUTED)
                        .monospace()
                        .size(10.0),
                );
                ui.add_sized(
                    [150.0, 20.0],
                    TextEdit::singleline(&mut self.display_filter)
                        .hint_text("display filter")
                        .frame(false),
                );
                if bracket_button(ui, "Clear", 56.0).clicked() {
                    self.clear_frames();
                }
                if bracket_button(ui, "Export", 64.0).clicked() {
                    self.export_frames();
                }
                ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                    ui.label(
                        RichText::new(format!(
                            "{} frames | {} dropped",
                            self.frames.len(),
                            self.dropped_frames
                        ))
                        .color(Palette::MUTED)
                        .monospace()
                        .size(10.0),
                    );
                });
            },
        );
    }

    fn frame_table(&mut self, ui: &mut egui::Ui) {
        let height = (ui.available_height() - 100.0).max(160.0);
        ui.allocate_ui(Vec2::new(ui.available_width(), height), |ui| {
            if self.frames.is_empty() {
                label_muted(ui, "-- no frames --");
                return;
            }

            let filter = self.display_filter.trim().to_ascii_lowercase();
            let visible: Vec<usize> = self
                .frames
                .iter()
                .enumerate()
                .filter_map(|(slot, row)| {
                    if filter.is_empty() {
                        return Some(slot);
                    }
                    let hex: String = row
                        .raw
                        .iter()
                        .map(|byte| format!("{byte:02x}"))
                        .collect::<Vec<_>>()
                        .join("");
                    let index_str = row.index.to_string();
                    if hex.contains(&filter) || index_str.contains(&filter) {
                        Some(slot)
                    } else {
                        None
                    }
                })
                .collect();

            if visible.is_empty() {
                label_muted(ui, "-- no frames match filter --");
                return;
            }

            let mut click_target: Option<usize> = None;
            TableBuilder::new(ui)
                .id_salt("frame_table")
                .sense(Sense::click())
                .striped(true)
                .resizable(true)
                .cell_layout(Layout::left_to_right(Align::Center))
                .column(Column::exact(48.0))
                .column(Column::exact(96.0))
                .column(Column::initial(96.0).at_least(60.0).resizable(true))
                .column(Column::initial(96.0).at_least(60.0).resizable(true))
                .column(Column::exact(56.0))
                .column(Column::exact(56.0))
                .column(Column::exact(64.0))
                .column(Column::remainder().at_least(120.0))
                .header(22.0, |mut header| {
                    header.col(|ui| {
                        ui.strong("No.");
                    });
                    header.col(|ui| {
                        ui.strong("Time");
                    });
                    header.col(|ui| {
                        ui.strong("Source");
                    });
                    header.col(|ui| {
                        ui.strong("Dest");
                    });
                    header.col(|ui| {
                        ui.strong("Type");
                    });
                    header.col(|ui| {
                        ui.strong("Len");
                    });
                    header.col(|ui| {
                        ui.strong("RSSI");
                    });
                    header.col(|ui| {
                        ui.strong("Info");
                    });
                })
                .body(|mut body| {
                    for slot in &visible {
                        let row = &self.frames[*slot];
                        let selected = self.selected_frame == Some(row.index);
                        let frame_index = row.index;
                        body.row(20.0, |mut tr| {
                            tr.set_selected(selected);
                            tr.col(|ui| {
                                ui.monospace(format!("{}", row.index));
                            });
                            tr.col(|ui| {
                                ui.monospace(format_timestamp(row.time));
                            });
                            tr.col(|ui| {
                                ui.monospace(&row.source);
                            });
                            tr.col(|ui| {
                                ui.monospace(&row.destination);
                            });
                            tr.col(|ui| {
                                cell_badge(ui, row.kind);
                            });
                            tr.col(|ui| {
                                ui.monospace(format!("{}", row.raw.len()));
                            });
                            tr.col(|ui| {
                                let text = row
                                    .rssi_dbm
                                    .map(|rssi| format!("{rssi:.0} dBm"))
                                    .unwrap_or_else(|| "-".to_string());
                                ui.monospace(text);
                            });
                            tr.col(|ui| {
                                ui.monospace(frame_info_preview(row));
                            });
                            if tr.response().clicked() {
                                click_target = Some(frame_index);
                            }
                        });
                    }
                });
            if let Some(index) = click_target {
                self.selected_frame = Some(index);
            }
        });
    }

    fn hex_dump(&mut self, ui: &mut egui::Ui) {
        let raw = self
            .selected_frame_row()
            .map(|row| row.raw.as_slice())
            .unwrap_or(&[]);
        Frame::new()
            .fill(Palette::PANEL)
            .stroke(Stroke::new(1.0, Palette::LINE))
            .inner_margin(Margin::same(4))
            .show(ui, |ui| {
                ui.set_min_height(96.0);
                ui.set_max_height(160.0);
                ScrollArea::vertical()
                    .id_salt("hex_dump_scroll")
                    .auto_shrink([false, false])
                    .max_height(ui.available_height())
                    .show(ui, |ui| {
                        if raw.is_empty() {
                            label_muted(ui, "-- no frame selected --");
                        } else {
                            for (offset, chunk) in raw.chunks(16).enumerate() {
                                let addr = offset * 16;
                                let hex = chunk
                                    .iter()
                                    .map(|byte| format!("{byte:02X}"))
                                    .collect::<Vec<_>>()
                                    .join(" ");
                                let ascii: String = chunk
                                    .iter()
                                    .map(|byte| {
                                        if byte.is_ascii_graphic() || *byte == b' ' {
                                            char::from(*byte)
                                        } else {
                                            '.'
                                        }
                                    })
                                    .collect();
                                ui.monospace(format!("{addr:04X}   {hex:<47}   {ascii}"));
                            }
                        }
                    });
            });
    }

    /// Render the image-reassembly view: canvas on top with a status
    /// row underneath showing per-image receipt progress, plus Reset
    /// and Save controls. Cheap to call every frame; the texture is
    /// only re-uploaded when [`Self::image_dirty`] is set or no
    /// texture exists yet, so steady-state cost is one painter.image
    /// call.
    fn image_view(&mut self, ui: &mut egui::Ui) {
        // SSTV bypasses the chunk reassembler entirely. When the active
        // downlink decodes images via SstvAnalyzer, render the latest
        // received frame (if any) and skip the reassembler-driven path.
        if self
            .selected_downlink()
            .is_some_and(|d| matches!(d.image, Some(ImageDef::Sstv {})))
        {
            self.sstv_view(ui);
            return;
        }
        let snapshot = match self.image_reassembler.as_ref() {
            Some(r) => r.snapshot(),
            None => return,
        };
        let image_count = snapshot.images.len();
        let active_idx = snapshot
            .images
            .iter()
            .position(|s| s.image_idx == self.image_active_idx)
            .unwrap_or(0);
        let active_image = snapshot.images.get(active_idx);
        // If the active stream changed, blow away the old texture so
        // the placeholder shows while we wait for the next decode of
        // the *new* stream. Otherwise keep the last-good texture in
        // place across transient decode failures (truncated JPEG mid
        // arrival).
        let active_image_idx = active_image.map(|s| s.image_idx);
        if self.image_texture_for_idx != active_image_idx {
            self.image_texture = None;
            self.image_texture_for_idx = None;
        }
        let needs_upload = self.image_dirty || self.image_texture.is_none();
        let mut canvas_dims = (snapshot.width.max(1) as f32, snapshot.height.max(1) as f32);
        let mut render_kind: Option<ImageRenderKind> = None;
        if needs_upload {
            if let Some(stream) = active_image {
                if let Some((color_image, w, h, kind)) = render_image(&snapshot, stream) {
                    canvas_dims = (w as f32, h as f32);
                    match self.image_texture.as_mut() {
                        Some(tex) => tex.set(color_image, TextureOptions::NEAREST),
                        None => {
                            self.image_texture = Some(ui.ctx().load_texture(
                                "image_canvas",
                                color_image,
                                TextureOptions::NEAREST,
                            ));
                        }
                    }
                    self.image_texture_for_idx = Some(stream.image_idx);
                    render_kind = Some(kind);
                }
            }
            self.image_dirty = false;
        } else if let Some(tex) = self.image_texture.as_ref() {
            let size = tex.size();
            canvas_dims = (size[0] as f32, size[1] as f32);
        }
        let avail = ui.available_size();
        let canvas_h = (avail.y - 56.0).max(120.0);
        // Reserve a fixed-width strip on the right for the packet
        // grid + progress bar. Mirrors the SSDV mockup. Falls back to
        // canvas-only if the available width can't fit both.
        let grid_w = if avail.x > 480.0 { 220.0 } else { 0.0 };
        let canvas_w = (avail.x - grid_w).max(120.0);
        ui.horizontal(|ui| {
            ui.allocate_ui(Vec2::new(canvas_w, canvas_h), |ui| {
                let (rect, _) =
                    ui.allocate_exact_size(Vec2::new(canvas_w, canvas_h), Sense::hover());
                ui.painter()
                    .rect_filled(rect, CornerRadius::ZERO, Color32::BLACK);
                if let Some(tex) = self.image_texture.as_ref() {
                    let (img_w, img_h) = canvas_dims;
                    let scale = (canvas_w / img_w).min(canvas_h / img_h);
                    let draw = Vec2::new(img_w * scale, img_h * scale);
                    let origin = Pos2::new(
                        rect.center().x - draw.x * 0.5,
                        rect.center().y - draw.y * 0.5,
                    );
                    let dest = Rect::from_min_size(origin, draw);
                    ui.painter().image(
                        tex.id(),
                        dest,
                        Rect::from_min_max(Pos2::new(0.0, 0.0), Pos2::new(1.0, 1.0)),
                        Color32::WHITE,
                    );
                    if matches!(render_kind, Some(ImageRenderKind::JpegBytesPreview)) {
                        ui.painter().text(
                            Pos2::new(rect.left() + 6.0, rect.top() + 4.0),
                            egui::Align2::LEFT_TOP,
                            "JPEG header pending — raw chunk preview",
                            FontId::monospace(11.0),
                            Palette::MUTED,
                        );
                    }
                } else {
                    let msg = match active_image {
                        None => "waiting for image chunks\u{2026}".to_string(),
                        Some(stream) => format!(
                            "image #{}: {}/{} chunks; decoding\u{2026}",
                            stream.image_idx, stream.received_count, stream.total_chunks,
                        ),
                    };
                    ui.painter().text(
                        rect.center(),
                        egui::Align2::CENTER_CENTER,
                        msg,
                        FontId::monospace(12.0),
                        Palette::MUTED,
                    );
                }
            });
            if grid_w > 0.0 {
                ui.allocate_ui(Vec2::new(grid_w, canvas_h), |ui| {
                    self.packet_grid(ui, active_image, grid_w, canvas_h);
                });
            }
        });
        self.image_status_row(ui, &snapshot, image_count);
    }

    /// SSTV pathway counterpart to [`Self::image_view`]. Renders
    /// `latest_sstv` directly to a canvas, with a status row underneath
    /// describing the mode / size of the most recently decoded frame.
    /// `latest_sstv` is reset whenever the active downlink changes
    /// (`refresh_image_reassembler`), so the canvas naturally clears
    /// when the user navigates away from an SSTV downlink.
    fn sstv_view(&mut self, ui: &mut egui::Ui) {
        // Lazily upload the texture from `latest_sstv`. We only rebuild
        // when sstv_dirty fires, mirroring the chunk-reassembler path.
        if self.sstv_dirty {
            if let Some(image) = self.latest_sstv.as_ref() {
                let w = image.width.max(1) as usize;
                let h = image.height.max(1) as usize;
                let expected = w.saturating_mul(h).saturating_mul(3);
                if image.pixels.len() >= expected {
                    let mut rgba = Vec::with_capacity(w * h * 4);
                    for chunk in image.pixels[..expected].chunks_exact(3) {
                        rgba.extend_from_slice(&[chunk[0], chunk[1], chunk[2], 0xff]);
                    }
                    let color = ColorImage::from_rgba_unmultiplied([w, h], &rgba);
                    match self.sstv_texture.as_mut() {
                        Some(tex) => tex.set(color, TextureOptions::NEAREST),
                        None => {
                            self.sstv_texture = Some(ui.ctx().load_texture(
                                "sstv_canvas",
                                color,
                                TextureOptions::NEAREST,
                            ));
                        }
                    }
                }
            }
            self.sstv_dirty = false;
        }
        let avail = ui.available_size();
        let canvas_h = (avail.y - 28.0).max(120.0);
        let canvas_w = avail.x.max(120.0);
        let canvas_dims = self
            .latest_sstv
            .as_ref()
            .map(|i| (i.width.max(1) as f32, i.height.max(1) as f32))
            .unwrap_or((1.0, 1.0));
        let (rect, _) = ui.allocate_exact_size(Vec2::new(canvas_w, canvas_h), Sense::hover());
        ui.painter()
            .rect_filled(rect, CornerRadius::ZERO, Color32::BLACK);
        if let Some(tex) = self.sstv_texture.as_ref() {
            let (img_w, img_h) = canvas_dims;
            let scale = (canvas_w / img_w).min(canvas_h / img_h);
            let draw = Vec2::new(img_w * scale, img_h * scale);
            let origin = Pos2::new(
                rect.center().x - draw.x * 0.5,
                rect.center().y - draw.y * 0.5,
            );
            let dest = Rect::from_min_size(origin, draw);
            ui.painter().image(
                tex.id(),
                dest,
                Rect::from_min_max(Pos2::new(0.0, 0.0), Pos2::new(1.0, 1.0)),
                Color32::WHITE,
            );
        } else {
            ui.painter().text(
                rect.center(),
                egui::Align2::CENTER_CENTER,
                "waiting for SSTV image\u{2026}",
                FontId::monospace(12.0),
                Palette::MUTED,
            );
        }
        ui.allocate_ui_with_layout(
            Vec2::new(ui.available_width(), 22.0),
            Layout::left_to_right(Align::Center),
            |ui| {
                ui.painter()
                    .rect_filled(ui.max_rect(), CornerRadius::ZERO, Palette::PANEL);
                ui.add_space(6.0);
                let text = match self.latest_sstv.as_ref() {
                    Some(i) => format!("SSTV {:?}: {}x{}", i.mode, i.width, i.height),
                    None => "SSTV: no image decoded yet".to_string(),
                };
                label_muted(ui, &text);
            },
        );
    }

    /// Status strip for the image view: shows the active image's
    /// progress (`received/total chunks`, percent complete), an
    /// image-selector chip row, and Reset / Save buttons. Hidden when
    /// no image chunks have arrived yet.
    fn image_status_row(
        &mut self,
        ui: &mut egui::Ui,
        snapshot: &openhoshimi_codec::ImageSnapshot,
        image_count: usize,
    ) {
        ui.allocate_ui_with_layout(
            Vec2::new(ui.available_width(), 22.0),
            Layout::left_to_right(Align::Center),
            |ui| {
                ui.painter()
                    .rect_filled(ui.max_rect(), CornerRadius::ZERO, Palette::PANEL);
                ui.add_space(6.0);
                if image_count == 0 {
                    if self.image_payload_misses > 0 {
                        label_muted(
                            ui,
                            &format!(
                                "no image chunks yet | header miss={} hit={} (check [downlink.image] header_signature)",
                                self.image_payload_misses, self.image_payload_hits,
                            ),
                        );
                    } else {
                        label_muted(ui, "no image chunks received yet");
                    }
                    return;
                }
                let active = snapshot
                    .images
                    .iter()
                    .find(|s| s.image_idx == self.image_active_idx)
                    .or_else(|| snapshot.images.first());
                if let Some(stream) = active {
                    let pct = if stream.total_chunks == 0 {
                        0.0
                    } else {
                        100.0 * stream.received_count as f32 / stream.total_chunks as f32
                    };
                    label_muted(
                        ui,
                        &format!(
                            "image #{}: {}/{} chunks ({:.1}%) | {} bytes",
                            stream.image_idx,
                            stream.received_count,
                            stream.total_chunks,
                            pct,
                            stream.bytes.len(),
                        ),
                    );
                }
                ui.add_space(8.0);
                self.image_status_row_images(ui, snapshot);
                self.image_status_row_buttons(ui, snapshot);
            },
        );
    }

    fn image_status_row_images(
        &mut self,
        ui: &mut egui::Ui,
        snapshot: &openhoshimi_codec::ImageSnapshot,
    ) {
        for stream in &snapshot.images {
            let active = stream.image_idx == self.image_active_idx;
            if tab_button(ui, &format!("#{}", stream.image_idx), active).clicked() && !active {
                self.image_active_idx = stream.image_idx;
                self.image_dirty = true;
            }
            ui.add_space(2.0);
        }
    }

    fn image_status_row_buttons(
        &mut self,
        ui: &mut egui::Ui,
        snapshot: &openhoshimi_codec::ImageSnapshot,
    ) {
        ui.add_space(8.0);
        if bracket_button(ui, "Reset", 60.0).clicked() {
            if let Some(r) = self.image_reassembler.as_mut() {
                r.reset();
            }
            self.image_texture = None;
            self.image_texture_for_idx = None;
            self.image_active_idx = 0;
            self.image_dirty = true;
        }
        let save_label = match snapshot.decoder {
            openhoshimi_codec::ImageDecoder::Jpeg => "Save JPEG",
            openhoshimi_codec::ImageDecoder::Raw => "Save PNG",
        };
        if bracket_button(ui, save_label, 84.0).clicked() {
            if let Err(err) = self.save_image(snapshot) {
                self.push_diagnostic(DiagnosticLevel::Error, format!("save image failed: {err}"));
            }
        }
    }

    /// Save the active image to disk. JPEG-decoder streams are written
    /// verbatim with a `.jpg` extension because the satellite already
    /// emits a complete JPEG bitstream; raw streams are encoded as PNG
    /// using the snapshot's pixel format. Returns `Err` only on
    /// encoder/IO failure; the user simply cancelling the dialog is
    /// reported as `Ok(())`.
    fn save_image(&self, snapshot: &openhoshimi_codec::ImageSnapshot) -> Result<(), String> {
        let stream = snapshot
            .images
            .iter()
            .find(|s| s.image_idx == self.image_active_idx)
            .or_else(|| snapshot.images.first())
            .ok_or_else(|| "no image stream available".to_string())?;
        match snapshot.decoder {
            openhoshimi_codec::ImageDecoder::Jpeg => self.save_image_jpeg(snapshot, stream),
            openhoshimi_codec::ImageDecoder::Raw => self.save_image_png(snapshot, stream),
        }
    }

    fn save_image_jpeg(
        &self,
        snapshot: &openhoshimi_codec::ImageSnapshot,
        stream: &openhoshimi_codec::ImageStream,
    ) -> Result<(), String> {
        let bytes = jpeg_export_bytes(snapshot, stream);
        let default_name = format!(
            "openhoshimi_image_{}_{}b.jpg",
            stream.image_idx,
            bytes.len()
        );
        let Some(path) = FileDialog::new()
            .set_file_name(default_name)
            .add_filter("JPEG", &["jpg", "jpeg"])
            .save_file()
        else {
            return Ok(());
        };
        std::fs::write(&path, &bytes).map_err(|e| format!("write {}: {e}", path.display()))?;
        Ok(())
    }

    fn save_image_png(
        &self,
        snapshot: &openhoshimi_codec::ImageSnapshot,
        stream: &openhoshimi_codec::ImageStream,
    ) -> Result<(), String> {
        let default_name = format!(
            "openhoshimi_image_{}_{}x{}.png",
            stream.image_idx, snapshot.width, snapshot.height
        );
        let Some(path) = FileDialog::new()
            .set_file_name(default_name)
            .add_filter("PNG", &["png"])
            .save_file()
        else {
            return Ok(());
        };
        let file =
            std::fs::File::create(&path).map_err(|e| format!("create {}: {e}", path.display()))?;
        let writer = std::io::BufWriter::new(file);
        let mut encoder = png::Encoder::new(writer, snapshot.width, snapshot.height);
        match snapshot.pixel_format {
            openhoshimi_codec::PixelFormat::Gray8 => {
                encoder.set_color(png::ColorType::Grayscale);
                encoder.set_depth(png::BitDepth::Eight);
            }
            openhoshimi_codec::PixelFormat::Rgb565 | openhoshimi_codec::PixelFormat::Rgb888 => {
                encoder.set_color(png::ColorType::Rgb);
                encoder.set_depth(png::BitDepth::Eight);
            }
        }
        let mut writer = encoder
            .write_header()
            .map_err(|e| format!("png header: {e}"))?;
        let pixels = encode_pixels_for_png(snapshot, stream);
        writer
            .write_image_data(&pixels)
            .map_err(|e| format!("png write: {e}"))?;
        Ok(())
    }

    /// SSDV-style packet grid. Renders one small square per chunk slot
    /// (received = green, missing = panel-grey, latest = blue), with a
    /// progress bar above and a `received/total (pct%)` label below.
    /// Square edge auto-fits the available column width; the layout
    /// degrades gracefully for very high chunk counts (~thousands) by
    /// shrinking the cell down to 2 px.
    fn packet_grid(
        &mut self,
        ui: &mut egui::Ui,
        active_image: Option<&openhoshimi_codec::ImageStream>,
        grid_w: f32,
        canvas_h: f32,
    ) {
        let (panel_rect, _) = ui.allocate_exact_size(Vec2::new(grid_w, canvas_h), Sense::hover());
        let painter = ui.painter_at(panel_rect);
        painter.rect_filled(panel_rect, CornerRadius::ZERO, Palette::PANEL);
        let Some(stream) = active_image else {
            painter.text(
                panel_rect.center(),
                egui::Align2::CENTER_CENTER,
                "no chunks",
                FontId::monospace(11.0),
                Palette::MUTED,
            );
            return;
        };
        let total = stream.total_chunks;
        if total == 0 {
            painter.text(
                panel_rect.center(),
                egui::Align2::CENTER_CENTER,
                "0 chunks",
                FontId::monospace(11.0),
                Palette::MUTED,
            );
            return;
        }
        let pad_x = 8.0;
        let pad_y = 6.0;
        let inner_w = (panel_rect.width() - pad_x * 2.0).max(1.0);
        let header_h = 14.0;
        let bar_h = 6.0;
        let footer_h = 14.0;
        let header_top = panel_rect.top() + pad_y;
        let bar_top = header_top + header_h + 2.0;
        let grid_top = bar_top + bar_h + 6.0;
        let footer_top = panel_rect.bottom() - pad_y - footer_h;
        let grid_bottom = footer_top - 4.0;
        let grid_h = (grid_bottom - grid_top).max(0.0);
        let pct = 100.0 * stream.received_count as f32 / total as f32;
        painter.text(
            Pos2::new(panel_rect.left() + pad_x, header_top),
            egui::Align2::LEFT_TOP,
            format!("packets  {}/{}", stream.received_count, total),
            FontId::monospace(11.0),
            Palette::TEXT,
        );
        let bar_rect = Rect::from_min_size(
            Pos2::new(panel_rect.left() + pad_x, bar_top),
            Vec2::new(inner_w, bar_h),
        );
        painter.rect_filled(bar_rect, CornerRadius::ZERO, Palette::BG);
        let fill_w = inner_w * (stream.received_count as f32 / total as f32).clamp(0.0, 1.0);
        if fill_w > 0.0 {
            painter.rect_filled(
                Rect::from_min_size(bar_rect.min, Vec2::new(fill_w, bar_h)),
                CornerRadius::ZERO,
                Palette::GREEN,
            );
        }
        let target_cell = 14.0_f32;
        let min_cell = 2.0_f32;
        let total_f = total as f32;
        let mut cell = target_cell;
        let mut cols = (inner_w / (cell + 1.0)).floor().max(1.0) as u32;
        loop {
            let rows = total_f / cols as f32;
            let needed_h = rows.ceil() * (cell + 1.0);
            if needed_h <= grid_h || cell <= min_cell {
                break;
            }
            cell = (cell - 1.0).max(min_cell);
            cols = (inner_w / (cell + 1.0)).floor().max(1.0) as u32;
        }
        let gap = if cell >= 6.0 { 1.0 } else { 0.0 };
        let step = cell + gap;
        let cols = ((inner_w + gap) / step).floor().max(1.0) as u32;
        let last_idx = stream.last_chunk_idx;
        for idx in 0..total {
            let row = idx / cols;
            let col = idx % cols;
            let x = panel_rect.left() + pad_x + col as f32 * step;
            let y = grid_top + row as f32 * step;
            if y + cell > grid_bottom {
                break;
            }
            let received = stream.chunk_received(idx);
            let mut color = if received {
                Palette::GREEN
            } else {
                Color32::from_rgb(56, 58, 62)
            };
            if stream.received_count > 0 && idx == last_idx {
                color = Palette::BLUE;
            }
            painter.rect_filled(
                Rect::from_min_size(Pos2::new(x, y), Vec2::new(cell, cell)),
                CornerRadius::ZERO,
                color,
            );
        }
        painter.text(
            Pos2::new(panel_rect.left() + pad_x, footer_top),
            egui::Align2::LEFT_TOP,
            format!("{:.1}%  last #{}", pct, last_idx),
            FontId::monospace(11.0),
            Palette::MUTED,
        );
    }

    fn right_panel(&mut self, ui: &mut egui::Ui) {
        // Anchor diagnostics + input to the bottom so they always stay
        // visible. Both are user-resizable: dragging the splitter lets the
        // user trade telemetry rows for more diagnostics or input detail.
        // Telemetry takes whatever vertical space is left and scrolls
        // inside its own ScrollArea.
        egui::TopBottomPanel::bottom("right_diag_section")
            .resizable(true)
            .default_height(140.0)
            .min_height(60.0)
            .max_height((ui.available_height() * 0.6).max(80.0))
            .frame(Frame::new().fill(Palette::BG))
            .show_separator_line(true)
            .show_inside(ui, |ui| self.diagnostics_panel(ui));

        egui::TopBottomPanel::bottom("right_input_section")
            .resizable(true)
            .default_height(180.0)
            .min_height(80.0)
            .max_height((ui.available_height() * 0.6).max(120.0))
            .frame(Frame::new().fill(Palette::BG))
            .show_separator_line(true)
            .show_inside(ui, |ui| self.input_panel(ui));

        egui::CentralPanel::default()
            .frame(Frame::new().fill(Palette::BG))
            .show_inside(ui, |ui| self.telemetry(ui));
    }

    fn telemetry(&mut self, ui: &mut egui::Ui) {
        let title = self
            .selected_frame_row()
            .map(|row| format!("Telemetry: {} frame #{}", self.selected_name(), row.index))
            .unwrap_or_else(|| format!("Telemetry: {}", self.selected_name()));
        header(ui, &title);
        ScrollArea::vertical()
            .id_salt("telemetry_scroll")
            .auto_shrink([false, false])
            .show(ui, |ui| {
                if let Some(row) = self.selected_frame_row() {
                    let mut groups: BTreeMap<&str, Vec<&TelemetryField>> = BTreeMap::new();
                    for field in &row.telemetry {
                        groups.entry(&field.group).or_default().push(field);
                    }
                    if groups.is_empty() {
                        label_muted(ui, "-- no telemetry fields --");
                    }
                    for (group, fields) in groups {
                        group_header(ui, group);
                        for field in fields {
                            telemetry_row(ui, field);
                        }
                    }
                } else {
                    label_muted(ui, "-- no frame selected --");
                }
            });
    }

    fn input_panel(&mut self, ui: &mut egui::Ui) {
        Frame::new()
            .fill(Palette::PANEL)
            .stroke(Stroke::new(1.0, Palette::LINE))
            .inner_margin(Margin::same(4))
            .show(ui, |ui| {
                group_header(ui, "Input");
                ScrollArea::vertical()
                    .id_salt("input_scroll")
                    .auto_shrink([false, false])
                    .show(ui, |ui| {
                        key_value(ui, "source", self.input_source_label());
                        key_value(ui, "device", &self.source_description);
                        key_value(ui, "rate", &format!("{} Hz", self.input_rate));
                        if let Some(path) = &self.input_path {
                            key_value(ui, "file", &short_path_label(path));
                        }
                        if let Some(downlink) = self.selected_downlink() {
                            key_value(
                                ui,
                                "modulation",
                                &format!("{} {}bd", downlink.modulation, downlink.baudrate),
                            );
                            key_value(ui, "framing", &downlink.framing);
                            key_value(ui, "input", downlink_input_kind_label(downlink));
                        }
                    });
            });
    }

    fn diagnostics_panel(&mut self, ui: &mut egui::Ui) {
        Frame::new()
            .fill(Palette::PANEL)
            .stroke(Stroke::new(1.0, Palette::LINE))
            .inner_margin(Margin::same(4))
            .show(ui, |ui| {
                group_header(ui, "Diagnostics");
                ui.horizontal(|ui| {
                    let (info, warn, error) = self.diagnostic_counts();
                    label_muted(ui, &format!("info {info}"));
                    label_muted(ui, &format!("warn {warn}"));
                    label_muted(ui, &format!("error {error}"));
                    if bracket_button(ui, "Clear", 56.0).clicked() {
                        self.clear_diagnostics();
                    }
                });
                hline(ui);
                ScrollArea::vertical()
                    .id_salt("diagnostics_scroll")
                    .auto_shrink([false, false])
                    .max_height(ui.available_height())
                    .show(ui, |ui| {
                        if self.diagnostics.is_empty() {
                            label_muted(ui, "-- no diagnostics --");
                        } else {
                            for entry in &self.diagnostics {
                                diagnostic_row(ui, entry);
                            }
                        }
                    });
            });
    }

    fn open_wav(&mut self) {
        let dialog = FileDialog::new()
            .set_title("Open audio input")
            .add_filter("Audio file", &["wav", "ogg"]);
        let Some(path) = dialog.pick_file() else {
            return;
        };
        self.open_wav_path(path);
    }

    fn open_wav_path(&mut self, path: PathBuf) {
        self.stop();
        self.input_path = Some(path.clone());
        self.input_mode = InputMode::WavFile;
        self.source_description = format!("audio file {}", short_path_label(&path));
        self.clear_frames();
        self.waterfall.clear();
        self.samples_processed = 0;
        self.input_progress_processed = 0;
        self.input_progress_total = None;
        self.status = format!("selected audio input {}", short_path_label(&path));
        self.spawn_alignment();
    }

    fn spawn_alignment(&mut self) {
        if let Some(handle) = self.alignment_thread.take() {
            drop(handle);
        }
        self.cached_alignment = None;
        self.alignment_state = AlignmentState::Idle;

        let Some(path) = self.input_path.clone() else {
            return;
        };
        if !matches!(self.input_mode, InputMode::WavFile) {
            return;
        }
        let Some(downlink) = self.selected_downlink_cloned() else {
            return;
        };
        if !matches!(input_kind_for(&downlink), Some(InputKind::Iq)) {
            return;
        }
        if matches!(self.audio_mode, IoAudioMode::Fm | IoAudioMode::Ssb) {
            return;
        }
        if matches!(self.audio_mode, IoAudioMode::Auto) {
            // Auto resolves to Fm for mono / duplicate-stereo WAVs and to
            // Ssb when the filename hints USB/LSB. Only Iq-resolved Auto
            // needs an alignment pre-flight; otherwise unconditionally
            // opening the file as stereo IQ surfaces a misleading
            // "WAV IQ input requires at least 2 channels" error in the
            // event log every time the user loads a real-audio recording.
            match detect_audio_mode_auto(&path) {
                Ok(IoAudioMode::Iq) => {}
                Ok(_) => return,
                Err(_) => return,
            }
        }
        let tuning_offset_hz = self.tuning_offset_for_selected(&downlink, Some(path.as_path()));
        let events = self.rx_sender.clone();
        let handle = thread::spawn(move || {
            run_alignment_thread(path, downlink, tuning_offset_hz, events);
        });
        self.alignment_thread = Some(handle);
    }

    fn input_source_label(&self) -> &'static str {
        match self.input_mode {
            InputMode::Soundcard => "soundcard",
            InputMode::WavFile => "audio file",
        }
    }

    fn soundcard_device_label(&self) -> String {
        match self.selected_soundcard.as_deref() {
            Some(name) => name.to_string(),
            None => "default".to_string(),
        }
    }

    fn soundcard_device_combo(&mut self, ui: &mut egui::Ui) {
        // Drop the selection if a previously-chosen device disappeared
        // between refreshes (USB unplug, host switch).  This keeps the
        // dropdown's `selected_text` consistent with the popup contents.
        if let Some(name) = self.selected_soundcard.as_deref() {
            if !self.soundcard_devices.iter().any(|d| d.name == name) {
                self.selected_soundcard = None;
            }
        }
        let label = self.soundcard_device_label();
        egui::ComboBox::from_id_salt("soundcard_device")
            .selected_text(label)
            .width(220.0)
            .show_ui(ui, |ui| {
                ui.selectable_value(&mut self.selected_soundcard, None, "default");
                if self.soundcard_devices.is_empty() {
                    ui.add_enabled(false, egui::Label::new("(no input devices)"));
                } else {
                    for device in &self.soundcard_devices {
                        let mut label = device.name.clone();
                        if device.is_default {
                            label.push_str(" (default)");
                        }
                        ui.selectable_value(
                            &mut self.selected_soundcard,
                            Some(device.name.clone()),
                            label,
                        );
                    }
                }
            });
        if bracket_button(ui, "Refresh", 80.0).clicked() {
            self.refresh_soundcard_devices();
        }
    }

    fn refresh_soundcard_devices(&mut self) {
        self.soundcard_devices = enumerate_input_devices();
        self.push_diagnostic(
            DiagnosticLevel::Info,
            format!(
                "soundcard: enumerated {} input device(s)",
                self.soundcard_devices.len()
            ),
        );
        if let Some(name) = self.selected_soundcard.as_deref() {
            if !self.soundcard_devices.iter().any(|d| d.name == name) {
                self.selected_soundcard = None;
                self.push_diagnostic(
                    DiagnosticLevel::Warn,
                    "soundcard: previously-selected device is no longer available; falling back to default".to_string(),
                );
            }
        }
    }

    fn downlink_combo(&mut self, ui: &mut egui::Ui) {
        let items = self
            .selected_satellite()
            .map(|satellite| {
                satellite
                    .downlinks
                    .iter()
                    .enumerate()
                    .map(|(index, downlink)| {
                        (
                            index,
                            downlink_combo_label(downlink),
                            downlink_is_supported(downlink),
                        )
                    })
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();
        let selected_text = self
            .selected_downlink()
            .map(downlink_combo_label)
            .unwrap_or_else(|| "no downlink".to_string());
        let before = self.selected_downlink;

        ui.add_enabled_ui(!self.running, |ui| {
            egui::ComboBox::from_id_salt("downlink")
                .selected_text(selected_text)
                .width(240.0)
                .show_ui(ui, |ui| {
                    if items.is_empty() {
                        label_muted(ui, "no downlinks");
                    }
                    for (index, label, supported) in &items {
                        if *supported {
                            ui.selectable_value(&mut self.selected_downlink, *index, label);
                        } else {
                            ui.add_enabled_ui(false, |ui| {
                                let _ = ui.selectable_label(false, label);
                            });
                        }
                    }
                });
        });

        if before != self.selected_downlink {
            self.clear_frames();
            self.waterfall.clear();
            self.refresh_image_reassembler();
            if let Some(downlink) = self.selected_downlink() {
                self.push_diagnostic(
                    DiagnosticLevel::Info,
                    format!("selected downlink {}", downlink_combo_label(downlink)),
                );
            }
        }
    }

    /// Rebuild [`Self::image_reassembler`] from the active downlink's
    /// `[downlink.image]` block, dropping any prior reassembler state
    /// and the cached texture. Called when the user switches downlink
    /// (or, indirectly, satellite). Forces `bottom_tab` back to
    /// `Frames` if the new downlink does not support images so the
    /// hidden tab bar can never leave the user stranded on a stale
    /// view.
    fn refresh_image_reassembler(&mut self) {
        self.image_reassembler = None;
        self.image_texture = None;
        self.image_texture_for_idx = None;
        self.image_dirty = false;
        self.image_active_idx = 0;
        self.image_payload_misses = 0;
        self.image_payload_hits = 0;
        self.image_payload_logged = false;
        self.latest_sstv = None;
        self.sstv_texture = None;
        self.sstv_dirty = false;
        let image_def = self.selected_downlink().and_then(|d| d.image.clone());
        if let Some(def) = image_def {
            // SSTV is decoded by the runtime worker via SstvAnalyzer
            // and reaches the image tab through `latest_sstv`, not via
            // the chunk reassembler. Skip the build instead of letting
            // it surface a misleading init error.
            if matches!(def, ImageDef::Sstv {}) {
                return;
            }
            match openhoshimi_codec::build_image_reassembler(&def) {
                Ok(r) => self.image_reassembler = Some(r),
                Err(err) => self.push_diagnostic(
                    DiagnosticLevel::Error,
                    format!("image reassembler init failed: {err}"),
                ),
            }
        } else {
            self.bottom_tab = BottomTab::Frames;
        }
    }

    fn selected_name(&self) -> String {
        self.selected_satellite()
            .map(|sat| sat.satellite.name.clone())
            .unwrap_or_else(|| "-".to_string())
    }

    fn frequency_label(&self) -> String {
        self.selected_downlink()
            .map(|downlink| format!("{:.3} MHz", downlink.freq_hz as f64 / 1_000_000.0))
            .unwrap_or_else(|| "- MHz".to_string())
    }

    fn tuning_offset_for_selected(&self, downlink: &DownlinkDef, path: Option<&Path>) -> f32 {
        let inferred = path
            .and_then(|path| infer_tuning_offset_hz(path, downlink.freq_hz))
            .unwrap_or(0);
        self.offset_hz as f32 + inferred as f32
    }

    fn spectrum_label(&self) -> String {
        self.selected_downlink()
            .map(|downlink| {
                format!(
                    "{:.3} MHz | {} {}bd | {} | {}",
                    downlink.freq_hz as f64 / 1_000_000.0,
                    downlink.modulation,
                    downlink.baudrate,
                    downlink.framing,
                    downlink_input_kind_label(downlink)
                )
            })
            .unwrap_or_else(|| "no downlink selected".to_string())
    }

    fn spectrum_span_hz(&self) -> u32 {
        if matches!(self.input_mode, InputMode::WavFile) {
            (self.input_rate / 2).max(6_000)
        } else {
            6_000
        }
    }

    fn spectrum_scale_label(&self) -> String {
        let span = self.spectrum_span_hz();
        let mid = span / 2;
        format!(
            "-{}   -{}   0   +{}   +{}",
            format_hz_short(span),
            format_hz_short(mid),
            format_hz_short(mid),
            format_hz_short(span),
        )
    }

    fn waterfall_texture_id(&mut self, ctx: &egui::Context) -> Option<egui::TextureId> {
        if self.waterfall.is_empty() {
            self.waterfall_texture = None;
            return None;
        }

        let image = waterfall_image(
            &self.waterfall,
            self.waterfall_min_db,
            self.waterfall_max_db,
        );
        match self.waterfall_texture.as_mut() {
            Some(texture) => texture.set(image, TextureOptions::LINEAR),
            None => {
                self.waterfall_texture =
                    Some(ctx.load_texture("waterfall", image, TextureOptions::LINEAR));
            }
        }
        self.waterfall_texture.as_ref().map(egui::TextureHandle::id)
    }
}

#[allow(clippy::too_many_arguments)]
/// Emit a spectrum row, throttled to roughly `SPECTRUM_INTERVAL`.
///
/// Spectrum samples are best-effort: if the GUI thread is behind and the
/// queue is full, we drop the row instead of blocking the decoder. The
/// wall-clock gate also keeps the row rate flat across vastly different
/// sample rates.
fn try_emit_spectrum<F: FnOnce() -> SpectrumSamples>(
    events: &SyncSender<RxEvent>,
    last: &mut Option<Instant>,
    build: F,
) {
    let now = Instant::now();
    if let Some(prev) = *last {
        if now.duration_since(prev) < SPECTRUM_INTERVAL {
            return;
        }
    }
    let samples = build();
    match events.try_send(RxEvent::Samples(samples)) {
        Ok(_) => *last = Some(now),
        Err(TrySendError::Full(_)) => {}
        Err(TrySendError::Disconnected(_)) => *last = Some(now),
    }
}

#[allow(clippy::too_many_arguments)]
fn run_decode_thread(
    satellite: SatelliteDefinition,
    downlink: DownlinkDef,
    input_mode: InputMode,
    decode_pace: DecodePace,
    input_path: Option<PathBuf>,
    soundcard_device: Option<String>,
    tuning_offset_hz: f32,
    cached_alignment: Option<CachedAlignment>,
    audio_mode: IoAudioMode,
    audio_carrier_hz: f32,
    events: SyncSender<RxEvent>,
    stop: Receiver<()>,
) {
    let _ = events.try_send(RxEvent::Dropped("priming WAV input\u{2026}".to_string()));
    let mut runtime = match build_runtime_input(
        input_mode,
        input_path,
        soundcard_device,
        &downlink,
        tuning_offset_hz,
        cached_alignment.as_ref(),
        audio_mode,
        audio_carrier_hz,
        &events,
    ) {
        Ok(runtime) => runtime,
        Err(err) => {
            let _ = events.try_send(RxEvent::Dropped(err));
            let _ = events.try_send(RxEvent::Stopped);
            return;
        }
    };
    let _ = events.try_send(RxEvent::SourceInfo {
        description: runtime.description().to_string(),
        sample_rate: runtime.sample_rate(),
    });

    let telemetry = downlink
        .telemetry_schema
        .as_ref()
        .and_then(|name| satellite.telemetry.get(name))
        .map(SchemaParser::new);
    let sample_rate = runtime.sample_rate();
    let total_samples = runtime.total_samples();
    let _ = events.try_send(RxEvent::Progress {
        processed: 0,
        total: total_samples,
    });
    let pace_realtime =
        matches!(input_mode, InputMode::WavFile) && matches!(decode_pace, DecodePace::Realtime);
    let start = Instant::now();
    let mut count = 0usize;
    // Spectrum rows are throttled by wall-clock so the rate is independent
    // of sample_rate / READ_CHUNK. The earlier stride-based limiter caused
    // wide bandwidth captures (e.g. 1.4 MS/s GMSK) to flood the GUI thread
    // with FFT work; this keeps the row rate at ~30 Hz on every input.
    let mut last_spectrum_at: Option<Instant> = None;

    loop {
        if stop.try_recv().is_ok() {
            break;
        }
        match &mut runtime {
            RuntimeInput::Audio { source, pipeline } => {
                let mut samples = [0.0f32; READ_CHUNK];
                let read = match source.read_samples(&mut samples) {
                    Ok(read) => read,
                    Err(IoError::EndOfStream) => break,
                    Err(err) => {
                        let _ =
                            events.try_send(RxEvent::Dropped(format!("audio read failed: {err}")));
                        break;
                    }
                };
                if read == 0 {
                    continue;
                }
                try_emit_spectrum(&events, &mut last_spectrum_at, || {
                    SpectrumSamples::Audio(samples[..read.min(SPECTRUM_FFT_LEN)].to_vec())
                });
                let frames = pipeline.push_samples(&samples[..read]);
                emit_frames(
                    &events,
                    &satellite,
                    telemetry.as_ref(),
                    &start,
                    pipeline,
                    &mut count,
                    frames,
                );
                let _ = events.try_send(RxEvent::PipelineStats(pipeline.pipeline_stats()));
                let _ = events.try_send(RxEvent::Progress {
                    processed: pipeline.total_samples(),
                    total: total_samples,
                });
                if pace_realtime
                    && !pace_realtime_loop(start, sample_rate, pipeline.total_samples(), &stop)
                {
                    break;
                }
            }
            RuntimeInput::Iq {
                source,
                pipeline,
                pending,
            } => {
                if !pending.is_empty() {
                    let take = pending.len().min(READ_CHUNK);
                    let samples: Vec<IqSample> = pending.drain(..take).collect();
                    try_emit_spectrum(&events, &mut last_spectrum_at, || {
                        SpectrumSamples::Iq(samples[..samples.len().min(SPECTRUM_FFT_LEN)].to_vec())
                    });
                    let frames = pipeline.push_samples(&samples);
                    emit_frames(
                        &events,
                        &satellite,
                        telemetry.as_ref(),
                        &start,
                        pipeline,
                        &mut count,
                        frames,
                    );
                    let _ = events.try_send(RxEvent::PipelineStats(pipeline.pipeline_stats()));
                    let _ = events.try_send(RxEvent::Progress {
                        processed: pipeline.total_samples(),
                        total: total_samples,
                    });
                    if pace_realtime
                        && !pace_realtime_loop(start, sample_rate, pipeline.total_samples(), &stop)
                    {
                        break;
                    }
                    continue;
                }
                let mut samples = [IqSample::default(); READ_CHUNK];
                let read = match source.read_samples(&mut samples) {
                    Ok(read) => read,
                    Err(IoError::EndOfStream) => break,
                    Err(err) => {
                        let _ =
                            events.try_send(RxEvent::Dropped(format!("IQ WAV read failed: {err}")));
                        break;
                    }
                };
                if read == 0 {
                    continue;
                }
                try_emit_spectrum(&events, &mut last_spectrum_at, || {
                    SpectrumSamples::Iq(samples[..read.min(SPECTRUM_FFT_LEN)].to_vec())
                });
                let frames = pipeline.push_samples(&samples[..read]);
                emit_frames(
                    &events,
                    &satellite,
                    telemetry.as_ref(),
                    &start,
                    pipeline,
                    &mut count,
                    frames,
                );
                let _ = events.try_send(RxEvent::PipelineStats(pipeline.pipeline_stats()));
                let _ = events.try_send(RxEvent::Progress {
                    processed: pipeline.total_samples(),
                    total: total_samples,
                });
                if pace_realtime
                    && !pace_realtime_loop(start, sample_rate, pipeline.total_samples(), &stop)
                {
                    break;
                }
            }
            RuntimeInput::IqSoftAo40 {
                source,
                pipeline,
                pending,
            } => {
                if !pending.is_empty() {
                    let take = pending.len().min(READ_CHUNK);
                    let samples: Vec<IqSample> = pending.drain(..take).collect();
                    try_emit_spectrum(&events, &mut last_spectrum_at, || {
                        SpectrumSamples::Iq(samples[..samples.len().min(SPECTRUM_FFT_LEN)].to_vec())
                    });
                    let decoded = pipeline.push_samples(&samples);
                    emit_decoded_frames(
                        &events,
                        &satellite,
                        telemetry.as_ref(),
                        &start,
                        &mut count,
                        decoded,
                    );
                    let _ = events.try_send(RxEvent::Progress {
                        processed: pipeline.total_samples(),
                        total: total_samples,
                    });
                    if pace_realtime
                        && !pace_realtime_loop(start, sample_rate, pipeline.total_samples(), &stop)
                    {
                        break;
                    }
                    continue;
                }
                let mut samples = [IqSample::default(); READ_CHUNK];
                let read = match source.read_samples(&mut samples) {
                    Ok(read) => read,
                    Err(IoError::EndOfStream) => break,
                    Err(err) => {
                        let _ =
                            events.try_send(RxEvent::Dropped(format!("IQ WAV read failed: {err}")));
                        break;
                    }
                };
                if read == 0 {
                    continue;
                }
                try_emit_spectrum(&events, &mut last_spectrum_at, || {
                    SpectrumSamples::Iq(samples[..read.min(SPECTRUM_FFT_LEN)].to_vec())
                });
                let decoded = pipeline.push_samples(&samples[..read]);
                emit_decoded_frames(
                    &events,
                    &satellite,
                    telemetry.as_ref(),
                    &start,
                    &mut count,
                    decoded,
                );
                let _ = events.try_send(RxEvent::Progress {
                    processed: pipeline.total_samples(),
                    total: total_samples,
                });
                if pace_realtime
                    && !pace_realtime_loop(start, sample_rate, pipeline.total_samples(), &stop)
                {
                    break;
                }
            }
            RuntimeInput::Sstv {
                source,
                analyzer,
                processed_samples,
            } => {
                let mut samples = [0.0f32; READ_CHUNK];
                let read = match source.read_samples(&mut samples) {
                    Ok(read) => read,
                    Err(IoError::EndOfStream) => {
                        for image in analyzer.finish() {
                            let _ = events.try_send(RxEvent::SstvImage(Box::new(image)));
                        }
                        break;
                    }
                    Err(err) => {
                        let _ =
                            events.try_send(RxEvent::Dropped(format!("audio read failed: {err}")));
                        break;
                    }
                };
                if read == 0 {
                    continue;
                }
                try_emit_spectrum(&events, &mut last_spectrum_at, || {
                    SpectrumSamples::Audio(samples[..read.min(SPECTRUM_FFT_LEN)].to_vec())
                });
                analyzer.push_samples(&samples[..read]);
                for image in analyzer.drain_images() {
                    let _ = events.try_send(RxEvent::SstvImage(Box::new(image)));
                }
                *processed_samples = processed_samples.saturating_add(read as u64);
                let _ = events.try_send(RxEvent::Progress {
                    processed: *processed_samples,
                    total: total_samples,
                });
                if pace_realtime
                    && !pace_realtime_loop(start, sample_rate, *processed_samples, &stop)
                {
                    break;
                }
            }
        }
    }

    let _ = events.try_send(RxEvent::Stopped);
}

fn pace_realtime_loop(
    start: Instant,
    sample_rate: u32,
    processed_samples: u64,
    stop: &Receiver<()>,
) -> bool {
    if sample_rate == 0 {
        return true;
    }

    let target = Duration::from_secs_f64(processed_samples as f64 / sample_rate as f64);
    loop {
        let elapsed = start.elapsed();
        if elapsed >= target {
            return true;
        }
        if stop.try_recv().is_ok() {
            return false;
        }
        let remaining = target - elapsed;
        let sleep_for = remaining.min(Duration::from_millis(20));
        if sleep_for.is_zero() {
            std::thread::yield_now();
        } else {
            std::thread::sleep(sleep_for);
        }
    }
}

enum RuntimeInput {
    Audio {
        source: Box<dyn InputSource>,
        pipeline: BitPipeline<f32>,
    },
    Iq {
        source: Box<dyn IqSource>,
        pipeline: BitPipeline<IqSample>,
        pending: Vec<IqSample>,
    },
    IqSoftAo40 {
        source: Box<dyn IqSource>,
        pipeline: Box<SoftAo40Pipeline>,
        pending: Vec<IqSample>,
    },
    /// Slow-Scan Television: raw mono audio drives an [`SstvAnalyzer`]
    /// directly, bypassing the bit pipeline entirely. Decoded frames
    /// are images, not bits, so they reach the GUI through a dedicated
    /// `RxEvent::SstvImage` channel rather than `RxEvent::Frame` /
    /// `RxEvent::ImagePayload`.
    Sstv {
        source: Box<dyn InputSource>,
        analyzer: SstvAnalyzer,
        processed_samples: u64,
    },
}

impl RuntimeInput {
    fn description(&self) -> &str {
        match self {
            Self::Audio { source, .. } => source.description(),
            Self::Iq { source, .. } => source.description(),
            Self::IqSoftAo40 { source, .. } => source.description(),
            Self::Sstv { source, .. } => source.description(),
        }
    }

    fn sample_rate(&self) -> u32 {
        match self {
            Self::Audio { source, .. } => source.sample_rate(),
            Self::Iq { source, .. } => source.sample_rate(),
            Self::IqSoftAo40 { source, .. } => source.sample_rate(),
            Self::Sstv { source, .. } => source.sample_rate(),
        }
    }

    fn total_samples(&self) -> Option<u64> {
        match self {
            Self::Audio { source, .. } => source.total_samples(),
            Self::Iq { source, .. } => source.total_samples(),
            Self::IqSoftAo40 { source, .. } => source.total_samples(),
            Self::Sstv { source, .. } => source.total_samples(),
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn build_runtime_input(
    input_mode: InputMode,
    input_path: Option<PathBuf>,
    soundcard_device: Option<String>,
    downlink: &DownlinkDef,
    tuning_offset_hz: f32,
    cached_alignment: Option<&CachedAlignment>,
    audio_mode: IoAudioMode,
    audio_carrier_hz: f32,
    events: &SyncSender<RxEvent>,
) -> Result<RuntimeInput, String> {
    if matches!(downlink.image, Some(ImageDef::Sstv {})) {
        return build_sstv_runtime_input(input_mode, input_path, soundcard_device);
    }
    match input_mode {
        InputMode::Soundcard => {
            // Determine the effective input kind.  If the user explicitly
            // selected FM or IQ in the audio mode dropdown, honour that
            // choice over auto-detection.  This lets users feed FM-
            // demodulated audio from a mono virtual soundcard to satellites
            // whose definition would normally require IQ (e.g. Hyacinth-1
            // GMSK via SDR# FM output).
            let effective_kind = match audio_mode {
                IoAudioMode::Fm => Some(InputKind::FmAudio),
                IoAudioMode::Iq => Some(InputKind::Iq),
                _ => input_kind_for(downlink),
            };
            match effective_kind {
                Some(InputKind::Iq) => {
                    let source = match soundcard_device.as_deref() {
                        Some(name) => SoundcardIqSource::open_by_name(name)
                            .map_err(|err| format!("soundcard IQ open failed: {err}"))?,
                        None => SoundcardIqSource::open_default()
                            .map_err(|err| format!("soundcard IQ open failed: {err}"))?,
                    };
                    let sample_rate = source.sample_rate();
                    if is_ao40_fec_downlink(downlink) {
                        let pipeline =
                            SoftAo40Pipeline::new(downlink, sample_rate, tuning_offset_hz)?;
                        Ok(RuntimeInput::IqSoftAo40 {
                            source: Box::new(source),
                            pipeline: Box::new(pipeline),
                            pending: Vec::new(),
                        })
                    } else {
                        let mut pipeline = BitPipeline::<IqSample>::new(downlink)?;
                        pipeline.configure_demodulator(downlink, sample_rate, tuning_offset_hz)?;
                        Ok(RuntimeInput::Iq {
                            source: Box::new(source),
                            pipeline,
                            pending: Vec::new(),
                        })
                    }
                }
                Some(InputKind::Audio) | Some(InputKind::FmAudio) => {
                    let source = match soundcard_device.as_deref() {
                        Some(name) => SoundcardSource::open_by_name(name)
                            .map_err(|err| format!("soundcard open failed: {err}"))?,
                        None => SoundcardSource::open_default()
                            .map_err(|err| format!("soundcard open failed: {err}"))?,
                    };
                    let sample_rate = source.sample_rate();
                    let mut pipeline = BitPipeline::<f32>::new(downlink)?;
                    if matches!(effective_kind, Some(InputKind::FmAudio)) {
                        pipeline.configure_fm_audio_demodulator(downlink, sample_rate)?;
                    } else {
                        pipeline.configure_demodulator(downlink, sample_rate, 0.0)?;
                    }
                    Ok(RuntimeInput::Audio {
                        source: Box::new(source),
                        pipeline,
                    })
                }
                None => Err("selected downlink is not supported by soundcard input".to_string()),
            }
        }
        InputMode::WavFile => {
            let Some(path) = input_path else {
                return Err("missing WAV input path".to_string());
            };
            match input_kind_for(downlink) {
                Some(InputKind::Audio) => {
                    let source = open_audio_source(&path)?;
                    let sample_rate = source.sample_rate();
                    let mut pipeline = BitPipeline::<f32>::new(downlink)?;
                    pipeline.configure_demodulator(downlink, sample_rate, 0.0)?;
                    Ok(RuntimeInput::Audio { source, pipeline })
                }
                Some(InputKind::Iq) => {
                    let resolved = match audio_mode {
                        IoAudioMode::Auto => detect_audio_mode_auto(&path)
                            .map_err(|err| format!("audio mode auto-detect failed: {err}"))?,
                        other => other,
                    };
                    match resolved {
                        IoAudioMode::Iq | IoAudioMode::Auto => {
                            let mut source = WavIqSource::open(&path)
                                .map_err(|err| format!("failed to open IQ WAV input: {err}"))?;
                            let sample_rate = source.sample_rate();
                            let downlink_id = format!("{downlink:?}");
                            let cache_hit = cached_alignment.filter(|cached| {
                                cached.path == path
                                    && cached.sample_rate == sample_rate
                                    && cached.downlink_id == downlink_id
                            });
                            let (setup, prefix) = if let Some(cached) = cache_hit {
                                if cached.prefix.is_empty() && cached.setup.sample_skip == 0 {
                                    let _ = events.try_send(RxEvent::Dropped(format!(
                                        "IQ alignment (non-linear): tuning={:.1} Hz",
                                        cached.setup.tuning_offset_hz
                                    )));
                                } else {
                                    let prefix_len = sample_rate as usize * 8;
                                    let _ = read_iq_prefix(&mut source, prefix_len)?;
                                    let _ = events.try_send(RxEvent::Dropped(format!(
                                        "IQ alignment (cached): tuning={:.1} Hz skip={} prefix_frames={}",
                                        cached.setup.tuning_offset_hz,
                                        cached.setup.sample_skip,
                                        cached.frames
                                    )));
                                }
                                (cached.setup.clone(), cached.prefix.clone())
                            } else {
                                let prefix_len = sample_rate as usize * 8;
                                let _ = events.try_send(RxEvent::Dropped(format!(
                                    "reading 8 s IQ prefix at {} Hz\u{2026}",
                                    sample_rate
                                )));
                                let mut prefix = read_iq_prefix(&mut source, prefix_len)?;
                                let _ = events.try_send(RxEvent::Dropped(format!(
                                    "scoring IQ alignment over {} samples\u{2026}",
                                    prefix.len()
                                )));
                                let scored = prepare_linear_iq_setup_scored(
                                    downlink,
                                    sample_rate,
                                    &prefix,
                                    tuning_offset_hz,
                                );
                                let setup = match scored.as_ref() {
                                    Some((setup, frames)) => {
                                        let _ = events.try_send(RxEvent::Dropped(format!(
                                            "IQ alignment: tuning={:.1} Hz skip={} prefix_frames={}",
                                            setup.tuning_offset_hz, setup.sample_skip, frames
                                        )));
                                        setup.clone()
                                    }
                                    None => {
                                        let _ = events.try_send(RxEvent::Dropped(format!(
                                            "IQ alignment: TOML defaults (no candidate decoded; tuning={tuning_offset_hz:.1} Hz)"
                                        )));
                                        openhoshimi_runtime::pipeline::LinearIqSetup {
                                            downlink: downlink.clone(),
                                            tuning_offset_hz,
                                            sample_skip: 0,
                                        }
                                    }
                                };
                                if setup.sample_skip < prefix.len() {
                                    prefix = prefix.into_iter().skip(setup.sample_skip).collect();
                                } else {
                                    prefix.clear();
                                }
                                (setup, prefix)
                            };
                            if is_ao40_fec_downlink(&setup.downlink) {
                                let pipeline = SoftAo40Pipeline::new(
                                    &setup.downlink,
                                    sample_rate,
                                    setup.tuning_offset_hz,
                                )?;
                                Ok(RuntimeInput::IqSoftAo40 {
                                    source: Box::new(source),
                                    pipeline: Box::new(pipeline),
                                    pending: prefix,
                                })
                            } else {
                                let mut pipeline = BitPipeline::<IqSample>::new(&setup.downlink)?;
                                pipeline.configure_demodulator(
                                    &setup.downlink,
                                    sample_rate,
                                    setup.tuning_offset_hz,
                                )?;
                                Ok(RuntimeInput::Iq {
                                    source: Box::new(source),
                                    pipeline,
                                    pending: prefix,
                                })
                            }
                        }
                        IoAudioMode::Fm => {
                            let _ = events.try_send(RxEvent::Dropped(
                                "audio mode: FM (mono real-audio path; tuning override ignored)"
                                    .to_string(),
                            ));
                            let source: Box<dyn InputSource> = open_audio_source(&path)?;
                            let sample_rate = source.sample_rate();
                            let mut pipeline = BitPipeline::<f32>::new(downlink)?;
                            pipeline.configure_fm_audio_demodulator(downlink, sample_rate)?;
                            Ok(RuntimeInput::Audio { source, pipeline })
                        }
                        IoAudioMode::Ssb => {
                            if is_ao40_fec_downlink(downlink) {
                                return Err(
                                    "SSB mono audio is not supported for AO-40 soft FEC downlinks"
                                        .to_string(),
                                );
                            }
                            let mut mono = open_audio_source(&path)?;
                            let sample_rate = mono.sample_rate();
                            let (carrier_hz, prefix) = if audio_carrier_hz.is_finite()
                                && audio_carrier_hz > 0.0
                            {
                                let _ = events.try_send(RxEvent::Dropped(format!(
                                    "audio mode: SSB (mono->IQ via {audio_carrier_hz:.1} Hz audio carrier, user-supplied)"
                                )));
                                (audio_carrier_hz, Vec::new())
                            } else {
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
                                    "SSB audio carrier auto-estimate failed: no peak in audio band; set Audio Carrier manually".to_string()
                                })?;
                                let _ = events.try_send(RxEvent::Dropped(format!(
                                    "audio mode: SSB (mono->IQ via {estimate:.1} Hz audio carrier, auto-estimated)"
                                )));
                                (estimate, prefix)
                            };
                            let source = MonoIqSource::with_prefix(mono, prefix);
                            let mut pipeline = BitPipeline::<IqSample>::new(downlink)?;
                            pipeline.configure_demodulator(downlink, sample_rate, carrier_hz)?;
                            Ok(RuntimeInput::Iq {
                                source: Box::new(source),
                                pipeline,
                                pending: Vec::new(),
                            })
                        }
                    }
                }
                Some(InputKind::FmAudio) => {
                    let source: Box<dyn InputSource> = open_audio_source(&path)?;
                    let sample_rate = source.sample_rate();
                    let mut pipeline = BitPipeline::<f32>::new(downlink)?;
                    pipeline.configure_fm_audio_demodulator(downlink, sample_rate)?;
                    Ok(RuntimeInput::Audio { source, pipeline })
                }
                None => Err("selected downlink is not supported by this input mode".to_string()),
            }
        }
    }
}

fn build_sstv_runtime_input(
    input_mode: InputMode,
    input_path: Option<PathBuf>,
    soundcard_device: Option<String>,
) -> Result<RuntimeInput, String> {
    let source: Box<dyn InputSource> = match input_mode {
        InputMode::Soundcard => match soundcard_device.as_deref() {
            Some(name) => Box::new(
                SoundcardSource::open_by_name(name)
                    .map_err(|err| format!("soundcard open failed: {err}"))?,
            ),
            None => Box::new(
                SoundcardSource::open_default()
                    .map_err(|err| format!("soundcard open failed: {err}"))?,
            ),
        },
        InputMode::WavFile => {
            let Some(path) = input_path else {
                return Err("missing audio input path".to_string());
            };
            open_audio_source(&path)?
        }
    };
    let sample_rate = source.sample_rate();
    let analyzer = SstvAnalyzer::new(sample_rate)
        .map_err(|err| format!("SSTV analyzer init failed: {err}"))?;
    Ok(RuntimeInput::Sstv {
        source,
        analyzer,
        processed_samples: 0,
    })
}

fn run_alignment_thread(
    path: PathBuf,
    downlink: DownlinkDef,
    tuning_offset_hz: f32,
    events: SyncSender<RxEvent>,
) {
    let downlink_id = format!("{downlink:?}");
    let mut source = match WavIqSource::open(&path) {
        Ok(source) => source,
        Err(err) => {
            let _ = events.try_send(RxEvent::AlignmentFailed(format!(
                "failed to open IQ WAV input: {err}"
            )));
            return;
        }
    };
    let sample_rate = source.sample_rate();

    // Non-linear modems (CPM/GMSK, AFSK, 4FSK, FM-audio) don't have a
    // linear-IQ alignment grid. We still read the 8-second prefix so we
    // can FFT-estimate the CPM carrier offset (this is what decode_file
    // does on the same non-linear path); the prefix is then handed to
    // the decoder via the cached alignment so no samples are lost.
    if !is_linear_iq_modem(&downlink) {
        let prefix_len = sample_rate as usize * 8;
        let prefix = match read_iq_prefix(&mut source, prefix_len) {
            Ok(prefix) => prefix,
            Err(err) => {
                let _ = events.try_send(RxEvent::AlignmentFailed(err));
                return;
            }
        };
        let mut tuning = tuning_offset_hz;
        if matches!(downlink.modem, Some(ModemDef::Cpm { .. })) {
            let estimate = estimate_cpm_iq_frequency_offset_hz(&prefix, sample_rate)
                .or_else(|| estimate_iq_frequency_offset_hz(&prefix, sample_rate))
                .filter(|v| v.is_finite());
            if let Some(estimate) = estimate {
                let _ = events.try_send(RxEvent::Dropped(format!(
                    "CPM carrier estimate: {estimate:.1} Hz (filename hint: {tuning:.1} Hz)"
                )));
                tuning = estimate;
            }
        }
        let setup = openhoshimi_runtime::pipeline::LinearIqSetup {
            downlink: downlink.clone(),
            tuning_offset_hz: tuning,
            sample_skip: 0,
        };
        let cached = CachedAlignment {
            path,
            sample_rate,
            downlink_id,
            prefix,
            setup,
            frames: 0,
        };
        let _ = events.try_send(RxEvent::AlignmentReady(Box::new(cached)));
        return;
    }

    let prefix_len = sample_rate as usize * 8;
    let mut prefix = match read_iq_prefix(&mut source, prefix_len) {
        Ok(prefix) => prefix,
        Err(err) => {
            let _ = events.try_send(RxEvent::AlignmentFailed(err));
            return;
        }
    };

    let scorer_secs = alignment_scorer_prefix_secs(downlink.baudrate);
    let scorer_prefix_len = ((sample_rate as f32 * scorer_secs) as usize).min(prefix.len());
    let mut announced = false;
    let mut score_with = |samples: &[IqSample]| {
        let events_inner = events.clone();
        let mut local_announced = announced;
        let result = prepare_linear_iq_setup_scored_with_progress(
            &downlink,
            sample_rate,
            samples,
            tuning_offset_hz,
            &mut |current, total| {
                if !local_announced {
                    local_announced = true;
                    let _ = events_inner.send(RxEvent::AlignmentStarted { total });
                } else {
                    let _ = events_inner.send(RxEvent::AlignmentProgress { current });
                }
            },
        );
        announced = local_announced;
        result
    };

    let mut scored = score_with(&prefix[..scorer_prefix_len]);
    if matches!(&scored, Some((_, frames)) if *frames == 0) && scorer_prefix_len < prefix.len() {
        let _ = events.try_send(RxEvent::Dropped(format!(
            "alignment scorer found 0 frames in {scorer_secs:.1} s slice; retrying with full {} s prefix",
            prefix.len() / sample_rate.max(1) as usize,
        )));
        scored = score_with(&prefix);
    }

    let (setup, frames) = match scored {
        Some(value) => value,
        None => {
            let _ = events.try_send(RxEvent::AlignmentFailed(
                "downlink is not a linear-IQ modem".to_string(),
            ));
            return;
        }
    };

    if setup.sample_skip < prefix.len() {
        prefix = prefix.into_iter().skip(setup.sample_skip).collect();
    } else {
        prefix.clear();
    }

    let cached = CachedAlignment {
        path,
        sample_rate,
        downlink_id,
        prefix,
        setup,
        frames,
    };
    let _ = events.try_send(RxEvent::AlignmentReady(Box::new(cached)));
}

fn emit_frames<P>(
    events: &SyncSender<RxEvent>,
    satellite: &SatelliteDefinition,
    telemetry: Option<&SchemaParser>,
    start: &Instant,
    pipeline: &mut BitPipeline<P>,
    count: &mut usize,
    frames: Vec<CoreFrame>,
) where
    P: Copy + Send + 'static,
{
    for mut frame in frames {
        *count += 1;
        frame.satellite_id = satellite.satellite.norad_id;
        // Tap Geoscan payloads for the image reassembler. emit_frames
        // is the path that runs the framer + codec inside frame_row
        // (raw whitened bytes get stored on the FrameRow), so we have
        // to decode once more here to recover the descrambled payload.
        // Decoding is cheap (PN9 + CRC compare) and fires only when the
        // codec stage is Geoscan to begin with.
        //
        // Image payload tap is gated on crc_ok: with threshold = 4 the
        // syncword framer produces a non-trivial number of false locks
        // whose 64-byte payload is essentially noise after PN9; letting
        // those through corrupts the canvas. CRC mismatch on a real
        // image chunk also indicates at least one bit error in the
        // 56-byte payload, which is far worse than a missing chunk.
        if let Ok(decoded) = pipeline.decode_frame(&frame) {
            match &decoded {
                DecodedFrame::Geoscan(geoscan) => {
                    pipeline.record_crc(geoscan.crc_ok);
                    if geoscan.crc_ok {
                        let _ = events.try_send(RxEvent::ImagePayload(geoscan.payload.clone()));
                    }
                }
                DecodedFrame::Ssdv(packet) => {
                    pipeline.record_crc(packet.crc_ok);
                    if packet.crc_ok {
                        // The SSDV image reassembler re-parses the
                        // header, so it needs the *full* corrected
                        // 256-byte packet rather than just payload.
                        let _ = events.try_send(RxEvent::ImagePayload(packet.raw.clone()));
                    }
                }
                _ => {}
            }
        }
        let row = frame_row(*count, start.elapsed(), pipeline, &frame, telemetry);
        match row {
            Ok(row) => {
                let _ = events.try_send(RxEvent::Frame(row));
            }
            Err(err) => {
                let _ = events.try_send(RxEvent::Dropped(err));
            }
        }
    }
}

fn emit_decoded_frames(
    events: &SyncSender<RxEvent>,
    satellite: &SatelliteDefinition,
    telemetry: Option<&SchemaParser>,
    start: &Instant,
    count: &mut usize,
    frames: Vec<DecodedFrame>,
) {
    for decoded in frames {
        *count += 1;
        // Tap every Geoscan payload for the image reassembler before
        // converting to FrameRow. The GUI side decides whether to
        // actually ingest based on whether the active downlink has
        // [downlink.image] configured. We deliberately do NOT gate on
        // crc_ok here: image chunks streaming over a LEO pass routinely
        // miss CRC, but the header bytes and most of the 56-byte chunk
        // are still usable, and the reassembler's header_signature
        // check is the real filter.
        if let DecodedFrame::Geoscan(ref geoscan) = decoded {
            let _ = events.try_send(RxEvent::ImagePayload(geoscan.payload.clone()));
        }
        if let DecodedFrame::Ssdv(ref packet) = decoded {
            // SsdvImageReassembler reads the full corrected wire
            // bytes (sync, header, payload, CRC, parity), so forward
            // the entire packet rather than just `payload`.
            let _ = events.try_send(RxEvent::ImagePayload(packet.raw.clone()));
        }
        let row = decoded_frame_row(*count, start.elapsed(), satellite, decoded, telemetry);
        let _ = events.try_send(RxEvent::Frame(row));
    }
}

fn decoded_frame_row(
    index: usize,
    timestamp: Duration,
    satellite: &SatelliteDefinition,
    decoded: DecodedFrame,
    telemetry: Option<&SchemaParser>,
) -> FrameRow {
    let _ = satellite;
    let mut source = "RAW".to_string();
    let mut destination = "ALL".to_string();
    let mut kind = FrameKind::Raw;
    let mut fields = Vec::new();
    let raw: Vec<u8>;

    match decoded {
        DecodedFrame::Ax25(ax25) => {
            source = ax25.source.call.clone();
            destination = ax25.destination.call.clone();
            kind = if ax25.info.is_empty() {
                FrameKind::Err
            } else if ax25
                .info
                .windows(3)
                .any(|window| window.eq_ignore_ascii_case(b"WOD"))
            {
                FrameKind::Wod
            } else {
                FrameKind::Tlm
            };
            if let Some(parser) = telemetry {
                fields = parser.parse_bytes(&ax25.info);
            }
            raw = ax25.info;
        }
        DecodedFrame::Ao40 { payload, .. } => {
            source = "AO40".to_string();
            destination = "FEC".to_string();
            kind = FrameKind::Tlm;
            if let Some(parser) = telemetry {
                fields = parser.parse_bytes(&payload);
            }
            raw = payload;
        }
        DecodedFrame::Ax100 {
            payload, crc_ok, ..
        } => {
            source = "AX100".to_string();
            destination = "FEC".to_string();
            // CRC-trailer modes mark a failed checksum as an error frame;
            // RS-protected modes report `crc_ok = None` and stay Tlm.
            kind = if crc_ok == Some(false) {
                FrameKind::Err
            } else {
                FrameKind::Tlm
            };
            // Never interpret a CRC-failed payload as housekeeping values.
            if crc_ok != Some(false) {
                if let Some(parser) = telemetry {
                    fields = parser.parse_bytes(&payload);
                }
            }
            raw = payload;
        }
        DecodedFrame::Geoscan(geoscan) => {
            source = "GEOSCAN".to_string();
            destination = "BEACON".to_string();
            kind = if geoscan.crc_ok {
                FrameKind::Tlm
            } else {
                FrameKind::Err
            };
            if let Some(parser) = telemetry {
                fields = parser.parse_bytes(&geoscan.payload);
            }
            raw = geoscan.payload;
        }
        DecodedFrame::Ssdv(packet) => {
            source = if packet.callsign.is_empty() {
                "SSDV".to_string()
            } else {
                packet.callsign.clone()
            };
            destination = format!("IMG#{:03}/PKT#{:05}", packet.image_id, packet.packet_id);
            kind = if packet.crc_ok {
                FrameKind::Tlm
            } else {
                FrameKind::Err
            };
            // SSDV packets carry JPEG MCU bytes — they are not
            // telemetry fields, so do not run the schema parser.
            raw = packet.payload;
        }
        DecodedFrame::Raw { raw_len, .. } => {
            raw = vec![0u8; raw_len];
        }
    }

    FrameRow {
        index,
        time: timestamp,
        source,
        destination,
        kind,
        rssi_dbm: None,
        raw,
        telemetry: fields,
    }
}

fn frame_row<P>(
    index: usize,
    timestamp: Duration,
    pipeline: &BitPipeline<P>,
    frame: &CoreFrame,
    telemetry: Option<&SchemaParser>,
) -> Result<FrameRow, String>
where
    P: Copy + Send + 'static,
{
    let decoded = pipeline.decode_frame(frame)?;
    let mut source = "RAW".to_string();
    let mut destination = "ALL".to_string();
    let mut kind = FrameKind::Raw;
    let mut fields = Vec::new();

    let raw = match &decoded {
        DecodedFrame::Ax25(ax25) => ax25.info.clone(),
        DecodedFrame::Ao40 { payload, .. } | DecodedFrame::Ax100 { payload, .. } => payload.clone(),
        DecodedFrame::Geoscan(geoscan) => geoscan.payload.clone(),
        DecodedFrame::Ssdv(packet) => packet.raw.clone(),
        DecodedFrame::Raw { .. } => frame.raw.clone(),
    };

    match decoded {
        DecodedFrame::Ax25(ax25) => {
            source = ax25.source.call.clone();
            destination = ax25.destination.call.clone();
            kind = if ax25.info.is_empty() {
                FrameKind::Err
            } else if ax25
                .info
                .windows(3)
                .any(|window| window.eq_ignore_ascii_case(b"WOD"))
            {
                FrameKind::Wod
            } else {
                FrameKind::Tlm
            };
            if let Some(parser) = telemetry {
                fields = parser.parse_bytes(&ax25.info);
            }
        }
        DecodedFrame::Ao40 { payload, .. } => {
            if let Some(parser) = telemetry {
                fields = parser.parse_bytes(&payload);
            }
            kind = FrameKind::Tlm;
        }
        DecodedFrame::Ax100 {
            payload, crc_ok, ..
        } => {
            // CRC status is surfaced via colour (Err row tint), not TYPE.
            // Even CRC-failed frames are structurally decoded (Golay header
            // + CCSDS descramble succeeded) and carry readable content
            // (digipeater ASCII, partial telemetry). Marking them ERR hides
            // useful data; keep them TLM so the user sees the payload.
            kind = FrameKind::Tlm;
            if crc_ok != Some(false) {
                if let Some(parser) = telemetry {
                    fields = parser.parse_bytes(&payload);
                }
            }
        }
        DecodedFrame::Geoscan(geoscan) => {
            source = "GEOSCAN".to_string();
            destination = "BEACON".to_string();
            kind = if geoscan.crc_ok {
                FrameKind::Tlm
            } else {
                FrameKind::Err
            };
            if let Some(parser) = telemetry {
                fields = parser.parse_bytes(&geoscan.payload);
            }
        }
        DecodedFrame::Ssdv(packet) => {
            source = if packet.callsign.is_empty() {
                "SSDV".to_string()
            } else {
                packet.callsign.clone()
            };
            destination = format!("IMG#{:03}/PKT#{:05}", packet.image_id, packet.packet_id);
            kind = if packet.crc_ok {
                FrameKind::Tlm
            } else {
                FrameKind::Err
            };
            // Image MCUs aren't telemetry — leave fields empty.
        }
        DecodedFrame::Raw { .. } => {
            if let Some(parser) = telemetry {
                fields = parser.parse_bytes(&frame.raw);
            }
        }
    }

    Ok(FrameRow {
        index,
        time: timestamp,
        source,
        destination,
        kind,
        rssi_dbm: frame.rssi_dbm,
        raw,
        telemetry: fields,
    })
}

/// Load embedded fonts with broad Unicode coverage (Latin Extended, Cyrillic,
/// Greek, CJK) so non-ASCII text renders correctly on all platforms.
fn configure_fonts(ctx: &egui::Context) {
    use egui::{FontData, FontDefinitions, FontFamily};

    let mut fonts = FontDefinitions::default();

    // Noto Sans Mono: covers Latin, Latin Extended, Cyrillic, Greek, and more.
    fonts.font_data.insert(
        "noto_sans_mono".to_owned(),
        FontData::from_static(include_bytes!("../assets/NotoSansMono-Regular.ttf")).into(),
    );

    // Noto Sans SC: covers CJK Unified Ideographs (Chinese/Japanese/Korean).
    fonts.font_data.insert(
        "noto_sans_sc".to_owned(),
        FontData::from_static(include_bytes!("../assets/NotoSansSC-Regular.otf")).into(),
    );

    // Append as fallbacks AFTER the built-in Hack monospace font so that
    // ASCII text keeps its original appearance and non-ASCII characters
    // (Latin Extended, Cyrillic, CJK) fall through to these fonts.
    fonts
        .families
        .entry(FontFamily::Monospace)
        .or_default()
        .push("noto_sans_mono".to_owned());
    fonts
        .families
        .entry(FontFamily::Monospace)
        .or_default()
        .push("noto_sans_sc".to_owned());

    // Also add as proportional fallbacks for any proportional text.
    fonts
        .families
        .entry(FontFamily::Proportional)
        .or_default()
        .push("noto_sans_mono".to_owned());
    fonts
        .families
        .entry(FontFamily::Proportional)
        .or_default()
        .push("noto_sans_sc".to_owned());

    ctx.set_fonts(fonts);
}

fn configure_style(ctx: &egui::Context) {
    let mut style = (*ctx.style()).clone();
    style.override_font_id = Some(FontId::new(11.0, FontFamily::Monospace));
    style.spacing.item_spacing = Vec2::new(4.0, 2.0);
    style.spacing.button_padding = Vec2::new(6.0, 2.0);
    style.visuals = egui::Visuals::dark();
    style.visuals.panel_fill = Palette::BG;
    style.visuals.window_fill = Palette::BG;
    style.visuals.window_stroke = Stroke::new(1.0, Palette::LINE);
    style.visuals.window_corner_radius = CornerRadius::ZERO;
    style.visuals.menu_corner_radius = CornerRadius::ZERO;
    style.visuals.extreme_bg_color = Palette::PANEL;
    style.visuals.override_text_color = Some(Palette::TEXT);
    style.visuals.widgets.noninteractive.corner_radius = CornerRadius::ZERO;
    style.visuals.widgets.inactive.corner_radius = CornerRadius::ZERO;
    style.visuals.widgets.hovered.corner_radius = CornerRadius::ZERO;
    style.visuals.widgets.active.corner_radius = CornerRadius::ZERO;
    style.visuals.widgets.open.corner_radius = CornerRadius::ZERO;
    ctx.set_style(style);
}

struct Palette;

impl Palette {
    const BG: Color32 = Color32::from_rgb(22, 23, 25);
    const PANEL: Color32 = Color32::from_rgb(30, 31, 34);
    const BAR: Color32 = Color32::from_rgb(34, 35, 38);
    const LINE: Color32 = Color32::from_rgb(55, 57, 62);
    const TEXT: Color32 = Color32::from_rgb(220, 224, 229);
    const MUTED: Color32 = Color32::from_rgb(134, 140, 148);
    const GREEN: Color32 = Color32::from_rgb(74, 194, 114);
    const ORANGE: Color32 = Color32::from_rgb(229, 153, 58);
    const RED: Color32 = Color32::from_rgb(220, 74, 74);
    const BLUE: Color32 = Color32::from_rgb(42, 91, 166);
}

fn panel_frame(fill: Color32) -> Frame {
    Frame::new()
        .fill(fill)
        .stroke(Stroke::new(1.0, Palette::LINE))
        .inner_margin(Margin::symmetric(6, 0))
}

fn label_muted(ui: &mut egui::Ui, text: &str) {
    ui.label(
        RichText::new(text)
            .color(Palette::MUTED)
            .monospace()
            .size(11.0),
    );
}

fn header(ui: &mut egui::Ui, text: &str) {
    let rect = row_rect(ui, 22.0);
    ui.painter()
        .rect_filled(rect, CornerRadius::ZERO, Palette::PANEL);
    ui.painter().text(
        rect.left_center() + Vec2::new(6.0, 0.0),
        egui::Align2::LEFT_CENTER,
        text,
        FontId::monospace(11.0),
        Palette::MUTED,
    );
}

fn group_header(ui: &mut egui::Ui, text: &str) {
    let rect = row_rect(ui, 22.0);
    ui.painter()
        .rect_filled(rect, CornerRadius::ZERO, Palette::PANEL);
    ui.painter().text(
        rect.left_center() + Vec2::new(6.0, 0.0),
        egui::Align2::LEFT_CENTER,
        format!("-- {text} --"),
        FontId::monospace(11.0),
        Palette::MUTED,
    );
}

fn row_rect(ui: &mut egui::Ui, height: f32) -> Rect {
    let width = ui.available_width();
    let (rect, _) = ui.allocate_exact_size(Vec2::new(width, height), Sense::hover());
    rect
}

fn hline(ui: &mut egui::Ui) {
    let rect = row_rect(ui, 1.0);
    ui.painter()
        .rect_filled(rect, CornerRadius::ZERO, Palette::LINE);
}

fn bracket_button(ui: &mut egui::Ui, label: &str, width: f32) -> egui::Response {
    ui.add_sized(
        [width, 22.0],
        egui::Button::new(format!("[{label}]"))
            .corner_radius(CornerRadius::ZERO)
            .fill(Palette::PANEL)
            .stroke(Stroke::new(1.0, Palette::LINE)),
    )
}

/// Compact tab toggle used by the bottom-region tab bar. Active tabs
/// are filled with the panel-active blue and white text; inactive tabs
/// match the panel background. Width is auto-sized to the label so the
/// strip can grow naturally if more tabs land later.
fn tab_button(ui: &mut egui::Ui, label: &str, active: bool) -> egui::Response {
    let (fill, text) = if active {
        (Palette::BLUE, Color32::WHITE)
    } else {
        (Palette::PANEL, Palette::TEXT)
    };
    ui.add(
        egui::Button::new(RichText::new(format!("[{label}]")).color(text).monospace())
            .corner_radius(CornerRadius::ZERO)
            .fill(fill)
            .stroke(Stroke::new(1.0, Palette::LINE))
            .min_size(Vec2::new(0.0, 22.0)),
    )
}

fn status_segment(ui: &mut egui::Ui, text: &str, color: Color32) {
    ui.label(RichText::new(text).color(color).monospace().size(11.0));
    ui.separator();
}

fn cell_badge(ui: &mut egui::Ui, kind: FrameKind) {
    let (label, fill, text) = match kind {
        FrameKind::Tlm => ("TLM", Color32::from_rgb(40, 92, 54), Palette::GREEN),
        FrameKind::Wod => ("WOD", Color32::from_rgb(99, 64, 27), Palette::ORANGE),
        FrameKind::Err => ("ERR", Color32::from_rgb(91, 39, 39), Palette::RED),
        FrameKind::Raw => ("RAW", Color32::from_rgb(52, 55, 60), Palette::TEXT),
    };
    let (rect, _) = ui.allocate_exact_size(Vec2::new(40.0, 14.0), Sense::hover());
    ui.painter().rect_filled(rect, CornerRadius::ZERO, fill);
    ui.painter().text(
        rect.center(),
        egui::Align2::CENTER_CENTER,
        label,
        FontId::monospace(10.0),
        text,
    );
}

fn frame_info_preview(row: &FrameRow) -> String {
    if let Some(field) = row.telemetry.first() {
        return format!(
            "{}.{} = {}",
            field.group,
            field.key,
            format_telemetry(field)
        );
    }
    let preview_len = row.raw.len().min(16);
    if preview_len == 0 {
        return "-".to_string();
    }
    let mut hex = String::with_capacity(preview_len * 3);
    for (i, byte) in row.raw[..preview_len].iter().enumerate() {
        if i > 0 {
            hex.push(' ');
        }
        hex.push_str(&format!("{byte:02X}"));
    }
    if row.raw.len() > preview_len {
        hex.push_str(" ..");
    }
    hex
}

fn telemetry_row(ui: &mut egui::Ui, field: &TelemetryField) {
    let color = match field.warn {
        openhoshimi_core::WarnLevel::Ok => Palette::GREEN,
        openhoshimi_core::WarnLevel::Warn => Palette::ORANGE,
        openhoshimi_core::WarnLevel::Error => Palette::RED,
    };
    demo_field(ui, &field.key, &format_telemetry(field), color);
}

fn demo_field(ui: &mut egui::Ui, key: &str, value: &str, value_color: Color32) {
    let rect = row_rect(ui, 20.0);
    ui.painter().line_segment(
        [rect.left_bottom(), rect.right_bottom()],
        Stroke::new(1.0, Palette::LINE),
    );
    let key_chars = key.chars().count();
    let key_width = key_chars as f32 * MONO_11_CHAR_W;
    ui.painter().text(
        rect.left_center() + Vec2::new(6.0, 0.0),
        egui::Align2::LEFT_CENTER,
        key,
        FontId::monospace(11.0),
        Palette::MUTED,
    );
    let value_avail = (rect.width() - 12.0 - key_width - 4.0).max(0.0);
    let max_value_chars = (value_avail / MONO_11_CHAR_W).floor() as usize;
    let value_text = truncate_left_with_ellipsis(value, max_value_chars);
    ui.painter().text(
        rect.right_center() - Vec2::new(6.0, 0.0),
        egui::Align2::RIGHT_CENTER,
        &value_text,
        FontId::monospace(11.0),
        value_color,
    );
}

const MONO_11_CHAR_W: f32 = 6.6;

fn truncate_left_with_ellipsis(value: &str, max_chars: usize) -> String {
    let total = value.chars().count();
    if total <= max_chars {
        return value.to_string();
    }
    if max_chars <= 1 {
        return "\u{2026}".to_string();
    }
    let take = max_chars - 1;
    let tail: String = value
        .chars()
        .rev()
        .take(take)
        .collect::<Vec<_>>()
        .into_iter()
        .rev()
        .collect();
    format!("\u{2026}{tail}")
}

fn key_value(ui: &mut egui::Ui, key: &str, value: &str) {
    demo_field(ui, key, value, Palette::TEXT);
}

fn diagnostic_row(ui: &mut egui::Ui, entry: &DiagnosticEntry) {
    let color = match entry.level {
        DiagnosticLevel::Info => Palette::TEXT,
        DiagnosticLevel::Warn => Palette::ORANGE,
        DiagnosticLevel::Error => Palette::RED,
    };
    let prefix = match entry.level {
        DiagnosticLevel::Info => "info",
        DiagnosticLevel::Warn => "warn",
        DiagnosticLevel::Error => "error",
    };
    ui.add(
        egui::Label::new(
            RichText::new(format!("[{prefix}] {}", entry.text))
                .color(color)
                .monospace()
                .size(11.0),
        )
        .wrap(),
    );
}

fn format_telemetry(field: &TelemetryField) -> String {
    let value = match &field.value {
        TelemetryValue::Float(value) => format!("{value:.2}"),
        TelemetryValue::Int(value) => value.to_string(),
        TelemetryValue::Bool(value) => value.to_string(),
        TelemetryValue::Bytes(bytes) => format_hex(bytes),
    };
    match &field.unit {
        Some(unit) if !unit.is_empty() => format!("{value} {unit}"),
        _ => value,
    }
}

fn waterfall_color(value: f32) -> Color32 {
    let v = value.clamp(0.0, 1.0);
    if v < 0.25 {
        let t = v * 4.0;
        lerp_color(Color32::from_rgb(0, 0, 0), Color32::from_rgb(84, 8, 10), t)
    } else if v < 0.5 {
        let t = (v - 0.25) * 4.0;
        lerp_color(
            Color32::from_rgb(84, 8, 10),
            Color32::from_rgb(196, 74, 22),
            t,
        )
    } else if v < 0.75 {
        let t = (v - 0.5) * 4.0;
        lerp_color(
            Color32::from_rgb(196, 74, 22),
            Color32::from_rgb(245, 180, 30),
            t,
        )
    } else {
        let t = (v - 0.75) * 4.0;
        lerp_color(
            Color32::from_rgb(245, 180, 30),
            Color32::from_rgb(255, 248, 224),
            t,
        )
    }
}

/// FFT-backed spectrum analyzer.
///
/// Reuses one `rustfft` plan and a scratch buffer per call so consecutive
/// rows pay no allocator / planner cost. Samples are zero-padded into the
/// fixed FFT length, Hann-windowed, FFTed, magnitude-squared, then mapped
/// onto `bins` output points by linearly interpolating the chosen frequency
/// grid. The grid is symmetric around DC so audio and IQ spectra share the
/// same display layout.
struct SpectrumProcessor {
    fft_len: usize,
    fft: Arc<dyn Fft<f32>>,
    window: Vec<f32>,
    buffer: Vec<Complex32>,
    scratch: Vec<Complex32>,
    powers: Vec<f32>,
}

impl SpectrumProcessor {
    fn new(fft_len: usize) -> Self {
        let mut planner = FftPlanner::<f32>::new();
        let fft = planner.plan_fft_forward(fft_len);
        let scratch_len = fft.get_inplace_scratch_len();
        let window = (0..fft_len)
            .map(|i| 0.5 - 0.5 * (2.0 * PI * i as f32 / (fft_len.max(2) - 1) as f32).cos())
            .collect();
        Self {
            fft_len,
            fft,
            window,
            buffer: vec![Complex32::new(0.0, 0.0); fft_len],
            scratch: vec![Complex32::new(0.0, 0.0); scratch_len],
            powers: vec![0.0; fft_len],
        }
    }

    fn run(&mut self) {
        self.fft
            .process_with_scratch(&mut self.buffer, &mut self.scratch);
        let inv_n = 1.0 / self.fft_len as f32;
        for (power, bin) in self.powers.iter_mut().zip(self.buffer.iter()) {
            *power = (bin.re * bin.re + bin.im * bin.im) * inv_n;
        }
    }

    fn row_audio(
        &mut self,
        samples: &[f32],
        bins: usize,
        sample_rate: u32,
        span_hz: u32,
    ) -> Vec<f32> {
        if samples.is_empty() || bins == 0 || sample_rate == 0 {
            return vec![WATERFALL_DB_FLOOR; bins];
        }
        let n = samples.len().min(self.fft_len);
        for (i, sample) in samples.iter().take(n).enumerate() {
            self.buffer[i] = Complex32::new(sample * self.window[i], 0.0);
        }
        for slot in &mut self.buffer[n..] {
            *slot = Complex32::new(0.0, 0.0);
        }
        self.run();
        // Real-input FFT is conjugate-symmetric, so we collapse positive and
        // negative bins by averaging them and interpolate against the
        // half-spectrum 0..fs/2.
        sample_real_powers(&self.powers, bins, sample_rate, span_hz)
    }

    fn row_iq(
        &mut self,
        samples: &[IqSample],
        bins: usize,
        sample_rate: u32,
        span_hz: u32,
    ) -> Vec<f32> {
        if samples.is_empty() || bins == 0 || sample_rate == 0 {
            return vec![WATERFALL_DB_FLOOR; bins];
        }
        let n = samples.len().min(self.fft_len);
        for (i, sample) in samples.iter().take(n).enumerate() {
            let w = self.window[i];
            self.buffer[i] = Complex32::new(sample.i * w, sample.q * w);
        }
        for slot in &mut self.buffer[n..] {
            *slot = Complex32::new(0.0, 0.0);
        }
        self.run();
        sample_complex_powers(&self.powers, self.fft_len, bins, sample_rate, span_hz)
    }
}

/// Map a half-spectrum (real input) onto a symmetric display grid.
fn sample_real_powers(powers: &[f32], bins: usize, sample_rate: u32, span_hz: u32) -> Vec<f32> {
    let fft_len = powers.len();
    let half = fft_len / 2;
    let bin_hz = sample_rate as f32 / fft_len as f32;
    let half_span = span_hz as f32;
    let mut row = Vec::with_capacity(bins);
    for bin in 0..bins {
        let position = if bins == 1 {
            0.5
        } else {
            bin as f32 / (bins - 1) as f32
        };
        let frequency = (-half_span + 2.0 * half_span * position).abs();
        let raw = frequency / bin_hz;
        let lo = (raw.floor() as usize).min(half.saturating_sub(1));
        let hi = (lo + 1).min(half.saturating_sub(1));
        let t = raw - lo as f32;
        let power = powers[lo] * (1.0 - t) + powers[hi] * t;
        row.push(power_to_db(power));
    }
    row
}

/// Map a full complex spectrum onto a symmetric display grid centered at DC.
fn sample_complex_powers(
    powers: &[f32],
    fft_len: usize,
    bins: usize,
    sample_rate: u32,
    span_hz: u32,
) -> Vec<f32> {
    let bin_hz = sample_rate as f32 / fft_len as f32;
    let half_span = span_hz as f32;
    let mut row = Vec::with_capacity(bins);
    for bin in 0..bins {
        let position = if bins == 1 {
            0.5
        } else {
            bin as f32 / (bins - 1) as f32
        };
        let frequency = -half_span + 2.0 * half_span * position;
        // FFT bin layout: 0..N/2 are 0..fs/2, N/2..N are -fs/2..0.
        let raw = frequency / bin_hz;
        let signed = raw.round() as isize;
        let idx = signed.rem_euclid(fft_len as isize) as usize;
        let next = ((signed + 1).rem_euclid(fft_len as isize)) as usize;
        let t = raw - signed as f32;
        let power = powers[idx] * (1.0 - t) + powers[next] * t;
        row.push(power_to_db(power));
    }
    row
}

fn power_to_db(power: f32) -> f32 {
    if power <= 0.0 {
        WATERFALL_DB_FLOOR
    } else {
        (10.0 * power.log10()).clamp(WATERFALL_DB_FLOOR, WATERFALL_DB_CEIL)
    }
}

/// Format a Hz quantity into a short readable string ("48k", "1.4M").
fn format_hz_short(hz: u32) -> String {
    if hz >= 1_000_000 {
        let m = hz as f32 / 1_000_000.0;
        format!("{m:.2}M")
    } else if hz >= 1_000 {
        let k = hz as f32 / 1_000.0;
        if k >= 100.0 {
            format!("{k:.0}k")
        } else {
            format!("{k:.1}k")
        }
    } else {
        format!("{hz}")
    }
}

fn waterfall_image(waterfall: &VecDeque<Vec<f32>>, min_db: f32, max_db: f32) -> ColorImage {
    let height = waterfall.len().max(1);
    let mut image = ColorImage::new([SPECTRUM_BINS, height], Color32::BLACK);
    let span = (max_db - min_db).max(1e-3);
    let inv_span = 1.0 / span;
    for (y, row) in waterfall.iter().enumerate() {
        let row_width = row.len().min(SPECTRUM_BINS);
        for x in 0..row_width {
            let normalized = ((row[x] - min_db) * inv_span).clamp(0.0, 1.0);
            image[(x, y)] = waterfall_color(normalized);
        }
    }
    image
}

/// Render an `ImageSnapshot` stream's byte buffer into a `ColorImage`
/// suitable for upload, returning the canvas dimensions actually
/// produced. Pixel-format dispatch lives here so the GUI stays
/// decoupled from the codec's internal byte layout.
///
/// JPEG streams are decoded with `jpeg-decoder` to whatever resolution
/// the JPEG header declares. Partial / truncated streams are normal
/// during a pass; when the decoder rejects the bitstream we return
/// `None` and the caller paints a "waiting for image chunks" placeholder
/// in place of the texture.
///
/// Outcome flag attached to every successful `render_image` result so
/// the GUI can label whether the user is looking at a real JPEG decode
/// or the raw-chunk grayscale fallback that lights up before enough
/// header bytes are in.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ImageRenderKind {
    /// Real JPEG decode succeeded (clean, +EOI, or truncate+EOI).
    JpegDecoded,
    /// JPEG decode rejected the bytestream; the preview is the raw
    /// reassembled chunks rendered as grayscale scanlines so missing
    /// chunks show as checker holes and the user can see what is
    /// already on the wire even when the decoder can't make sense of
    /// it yet.
    JpegBytesPreview,
    /// Decoder mode is `Raw` — pixels mapped straight from
    /// `stream.bytes` per the snapshot's pixel format.
    Raw,
}

/// For `Rgb565` we read big-endian as defined by Geoscan; for `Gray8`
/// each byte becomes (R, G, B) = (v, v, v).
fn render_image(
    snapshot: &openhoshimi_codec::ImageSnapshot,
    stream: &openhoshimi_codec::ImageStream,
) -> Option<(ColorImage, u32, u32, ImageRenderKind)> {
    match snapshot.decoder {
        openhoshimi_codec::ImageDecoder::Jpeg => render_image_jpeg(snapshot, stream),
        openhoshimi_codec::ImageDecoder::Raw => Some((
            render_image_raw(snapshot, stream),
            snapshot.width,
            snapshot.height,
            ImageRenderKind::Raw,
        )),
    }
}

/// Decode a JPEG bitstream into a [`ColorImage`], tolerating partial
/// streams. Tries four strategies in order:
///
/// 1. Decode the bytes verbatim. Works once a full JPEG with EOI is in.
/// 2. Append `FF D9` (EOI) and decode. Lets jpeg-decoder render every
///    intact MCU before giving up at the truncation point.
/// 3. Truncate to the first missing-chunk boundary, append `FF D9`, and
///    decode. Strategy 2 can fail when a chunk-sized hole sits inside
///    the bytestream; cutting at the first hole lets the leading run
///    still preview, with everything past the cut shown as black.
/// 4. Fallback: render the raw reassembled chunks as a grayscale
///    scanline image (one row per chunk, `chunk_bytes` columns wide).
///    Used when the decoder rejects every truncation — typically while
///    the JPEG header (SOI/SOF/SOS) is still missing chunks. Missing
///    chunks show as a checkerboard so the user sees the receive
///    pattern instead of a blank canvas.
fn render_image_jpeg(
    snapshot: &openhoshimi_codec::ImageSnapshot,
    stream: &openhoshimi_codec::ImageStream,
) -> Option<(ColorImage, u32, u32, ImageRenderKind)> {
    let bytes = &stream.bytes;
    if bytes.len() < 2 || bytes[0] != 0xFF || bytes[1] != 0xD8 {
        return Some(render_jpeg_bytes_as_grayscale(snapshot, stream));
    }
    if let Some((img, w, h)) = decode_jpeg_bytes(bytes) {
        return Some((img, w, h, ImageRenderKind::JpegDecoded));
    }
    let mut padded = Vec::with_capacity(bytes.len() + 2);
    padded.extend_from_slice(bytes);
    padded.extend_from_slice(&[0xFF, 0xD9]);
    if let Some((img, w, h)) = decode_jpeg_bytes(&padded) {
        return Some((img, w, h, ImageRenderKind::JpegDecoded));
    }
    let cut = first_missing_chunk_byte(snapshot, stream);
    if cut > 2 {
        let mut clipped = Vec::with_capacity(cut + 2);
        clipped.extend_from_slice(&bytes[..cut]);
        clipped.extend_from_slice(&[0xFF, 0xD9]);
        if let Some((img, w, h)) = decode_jpeg_bytes(&clipped) {
            return Some((img, w, h, ImageRenderKind::JpegDecoded));
        }
    }
    Some(render_jpeg_bytes_as_grayscale(snapshot, stream))
}

/// Visualise the reassembled chunks as a grayscale scanline image:
/// width = `chunk_bytes`, height = `total_chunks`. Received chunks
/// paint their bytes as Y=R=G=B; missing chunks paint a checkerboard so
/// the user can read the receive pattern at a glance even when no JPEG
/// decode is possible. Always returns at least a 1x1 image.
fn render_jpeg_bytes_as_grayscale(
    snapshot: &openhoshimi_codec::ImageSnapshot,
    stream: &openhoshimi_codec::ImageStream,
) -> (ColorImage, u32, u32, ImageRenderKind) {
    let chunk = snapshot.chunk_bytes.max(1) as usize;
    let total = stream.total_chunks.max(1) as usize;
    let w = chunk;
    let h = total;
    let mut image = ColorImage::new([w, h], Color32::BLACK);
    for row in 0..total {
        let received = stream.chunk_received(row as u32);
        for col in 0..chunk {
            let color = if received {
                let i = row * chunk + col;
                let v = stream.bytes.get(i).copied().unwrap_or(0);
                Color32::from_rgb(v, v, v)
            } else {
                let cell = ((col / 4) ^ (row / 4)) & 1;
                if cell == 0 {
                    Color32::from_rgb(36, 38, 42)
                } else {
                    Color32::from_rgb(60, 62, 68)
                }
            };
            image[(col, row)] = color;
        }
    }
    (image, w as u32, h as u32, ImageRenderKind::JpegBytesPreview)
}

/// Byte offset of the first chunk slot that has never been received.
/// Returns `stream.bytes.len()` when every chunk covered by the buffer
/// is filled, so the caller can use the result as an upper bound for a
/// "decode the leading intact run" slice.
fn first_missing_chunk_byte(
    snapshot: &openhoshimi_codec::ImageSnapshot,
    stream: &openhoshimi_codec::ImageStream,
) -> usize {
    let chunk = snapshot.chunk_bytes.max(1) as usize;
    for idx in 0..stream.total_chunks {
        if !stream.chunk_received(idx) {
            return (idx as usize) * chunk;
        }
    }
    stream.bytes.len()
}

fn decode_jpeg_bytes(bytes: &[u8]) -> Option<(ColorImage, u32, u32)> {
    let mut decoder = jpeg_decoder::Decoder::new(bytes);
    let pixels = decoder.decode().ok()?;
    let info = decoder.info()?;
    let w = info.width as usize;
    let h = info.height as usize;
    let mut image = ColorImage::new([w, h], Color32::BLACK);
    match info.pixel_format {
        jpeg_decoder::PixelFormat::L8 => {
            for y in 0..h {
                for x in 0..w {
                    let v = *pixels.get(y * w + x).unwrap_or(&0);
                    image[(x, y)] = Color32::from_rgb(v, v, v);
                }
            }
        }
        jpeg_decoder::PixelFormat::RGB24 => {
            for y in 0..h {
                for x in 0..w {
                    let i = (y * w + x) * 3;
                    if let (Some(&r), Some(&g), Some(&b)) =
                        (pixels.get(i), pixels.get(i + 1), pixels.get(i + 2))
                    {
                        image[(x, y)] = Color32::from_rgb(r, g, b);
                    }
                }
            }
        }
        // CMYK32 / L16 from JPEG aren't produced by Geoscan and are
        // rare elsewhere; treat them as unsupported and let the caller
        // show a placeholder.
        _ => return None,
    }
    Some((image, info.width as u32, info.height as u32))
}

/// Pick the best openable byte sequence for a JPEG export. Tries
/// (1) verbatim, (2) verbatim + EOI, (3) leading-intact-run + EOI,
/// returning the first variant that round-trips through
/// `decode_jpeg_bytes`. When none decode, falls back to verbatim + EOI
/// (which at least has a terminator most viewers expect) so the saved
/// file is no worse than the raw bytes ever were.
fn jpeg_export_bytes(
    snapshot: &openhoshimi_codec::ImageSnapshot,
    stream: &openhoshimi_codec::ImageStream,
) -> Vec<u8> {
    let bytes = &stream.bytes;
    if decode_jpeg_bytes(bytes).is_some() {
        return bytes.clone();
    }
    let mut padded = Vec::with_capacity(bytes.len() + 2);
    padded.extend_from_slice(bytes);
    padded.extend_from_slice(&[0xFF, 0xD9]);
    if decode_jpeg_bytes(&padded).is_some() {
        return padded;
    }
    let cut = first_missing_chunk_byte(snapshot, stream);
    if cut > 2 {
        let mut clipped = Vec::with_capacity(cut + 2);
        clipped.extend_from_slice(&bytes[..cut]);
        clipped.extend_from_slice(&[0xFF, 0xD9]);
        if decode_jpeg_bytes(&clipped).is_some() {
            return clipped;
        }
    }
    padded
}

fn render_image_raw(
    snapshot: &openhoshimi_codec::ImageSnapshot,
    stream: &openhoshimi_codec::ImageStream,
) -> ColorImage {
    let w = snapshot.width as usize;
    let h = snapshot.height as usize;
    let mut image = ColorImage::new([w, h], Color32::BLACK);
    let bytes = &stream.bytes;
    match snapshot.pixel_format {
        openhoshimi_codec::PixelFormat::Gray8 => {
            for y in 0..h {
                for x in 0..w {
                    let i = y * w + x;
                    if let Some(&v) = bytes.get(i) {
                        image[(x, y)] = Color32::from_rgb(v, v, v);
                    }
                }
            }
        }
        openhoshimi_codec::PixelFormat::Rgb565 => {
            for y in 0..h {
                for x in 0..w {
                    let i = (y * w + x) * 2;
                    if let (Some(&hi), Some(&lo)) = (bytes.get(i), bytes.get(i + 1)) {
                        let v = u16::from_be_bytes([hi, lo]);
                        let r = ((v >> 11) & 0x1f) as u8;
                        let g = ((v >> 5) & 0x3f) as u8;
                        let b = (v & 0x1f) as u8;
                        image[(x, y)] =
                            Color32::from_rgb(r << 3 | r >> 2, g << 2 | g >> 4, b << 3 | b >> 2);
                    }
                }
            }
        }
        openhoshimi_codec::PixelFormat::Rgb888 => {
            for y in 0..h {
                for x in 0..w {
                    let i = (y * w + x) * 3;
                    if let (Some(&r), Some(&g), Some(&b)) =
                        (bytes.get(i), bytes.get(i + 1), bytes.get(i + 2))
                    {
                        image[(x, y)] = Color32::from_rgb(r, g, b);
                    }
                }
            }
        }
    }
    image
}

/// Build a PNG-ready byte buffer for `stream` using `snapshot`'s
/// pixel format. Output layout matches the encoder header set by
/// `save_image_png`: Gray8 → one byte per pixel, Rgb565/Rgb888 → 3
/// bytes per pixel (R, G, B). Rgb565 is expanded to 8-bit by bit
/// replication so the on-disk PNG looks identical to the canvas.
fn encode_pixels_for_png(
    snapshot: &openhoshimi_codec::ImageSnapshot,
    stream: &openhoshimi_codec::ImageStream,
) -> Vec<u8> {
    let w = snapshot.width as usize;
    let h = snapshot.height as usize;
    let bytes = &stream.bytes;
    match snapshot.pixel_format {
        openhoshimi_codec::PixelFormat::Gray8 => {
            let mut out = vec![0u8; w * h];
            let copy_len = out.len().min(bytes.len());
            out[..copy_len].copy_from_slice(&bytes[..copy_len]);
            out
        }
        openhoshimi_codec::PixelFormat::Rgb565 => {
            let mut out = vec![0u8; w * h * 3];
            for y in 0..h {
                for x in 0..w {
                    let i = (y * w + x) * 2;
                    if let (Some(&hi), Some(&lo)) = (bytes.get(i), bytes.get(i + 1)) {
                        let v = u16::from_be_bytes([hi, lo]);
                        let r = ((v >> 11) & 0x1f) as u8;
                        let g = ((v >> 5) & 0x3f) as u8;
                        let b = (v & 0x1f) as u8;
                        let o = (y * w + x) * 3;
                        out[o] = r << 3 | r >> 2;
                        out[o + 1] = g << 2 | g >> 4;
                        out[o + 2] = b << 3 | b >> 2;
                    }
                }
            }
            out
        }
        openhoshimi_codec::PixelFormat::Rgb888 => {
            let mut out = vec![0u8; w * h * 3];
            let copy_len = out.len().min(bytes.len());
            out[..copy_len].copy_from_slice(&bytes[..copy_len]);
            out
        }
    }
}

fn lerp_color(a: Color32, b: Color32, t: f32) -> Color32 {
    let t = t.clamp(0.0, 1.0);
    let lerp = |start: u8, end: u8| -> u8 {
        (start as f32 + (end as f32 - start as f32) * t).clamp(0.0, 255.0) as u8
    };
    Color32::from_rgb(lerp(a.r(), b.r()), lerp(a.g(), b.g()), lerp(a.b(), b.b()))
}

fn sat_label(satellite: &SatelliteDefinition) -> String {
    if satellite.satellite.aliases.is_empty() {
        satellite.satellite.name.clone()
    } else {
        format!(
            "{} ({})",
            satellite.satellite.name, satellite.satellite.aliases[0]
        )
    }
}

fn downlink_combo_label(downlink: &DownlinkDef) -> String {
    let support = if downlink_is_supported(downlink) {
        ""
    } else {
        " unsupported"
    };
    format!(
        "{:.3} MHz {} {}bd {}{}",
        downlink.freq_hz as f64 / 1_000_000.0,
        downlink.modulation,
        downlink.baudrate,
        downlink.framing,
        support
    )
}

fn downlink_is_supported(downlink: &DownlinkDef) -> bool {
    if is_sstv_downlink(downlink) {
        return true;
    }
    input_kind_for(downlink).is_some() && can_build_downlink(downlink)
}

fn is_sstv_downlink(downlink: &DownlinkDef) -> bool {
    matches!(downlink.image, Some(ImageDef::Sstv {}))
}

fn input_kind_label(kind: Option<InputKind>) -> &'static str {
    match kind {
        Some(InputKind::Audio) => "audio",
        Some(InputKind::Iq) => "IQ",
        Some(InputKind::FmAudio) => "FM audio",
        None => "unsupported",
    }
}

fn downlink_input_kind_label(downlink: &DownlinkDef) -> &'static str {
    if is_sstv_downlink(downlink) {
        "audio (SSTV)"
    } else {
        input_kind_label(input_kind_for(downlink))
    }
}

fn short_path_label(path: &Path) -> String {
    path.file_name()
        .and_then(|name| name.to_str())
        .map(|name| name.to_string())
        .unwrap_or_else(|| path.display().to_string())
}

fn status_path_label(text: &str) -> String {
    let path = Path::new(text);
    if path.components().count() > 1 {
        short_path_label(path)
    } else {
        text.to_string()
    }
}

fn spectrum_samples_len(samples: &SpectrumSamples) -> usize {
    match samples {
        SpectrumSamples::Audio(values) => values.len(),
        SpectrumSamples::Iq(values) => values.len(),
    }
}

fn resolve_satellites_dir() -> PathBuf {
    let mut candidates = Vec::new();
    if let Ok(cwd) = std::env::current_dir() {
        push_satellite_dir_candidates(&mut candidates, &cwd);
    }
    if let Ok(exe) = std::env::current_exe() {
        if let Some(parent) = exe.parent() {
            push_satellite_dir_candidates(&mut candidates, parent);
        }
    }
    if let Some(manifest_dir) = option_env!("CARGO_MANIFEST_DIR") {
        push_satellite_dir_candidates(&mut candidates, Path::new(manifest_dir));
    }

    for candidate in candidates {
        if candidate.is_dir() {
            return candidate;
        }
    }

    PathBuf::from("satellites")
}

fn push_satellite_dir_candidates(candidates: &mut Vec<PathBuf>, root: &Path) {
    for ancestor in root.ancestors() {
        let candidate = ancestor.join("satellites");
        if !candidates.iter().any(|existing| existing == &candidate) {
            candidates.push(candidate);
        }
    }
}
