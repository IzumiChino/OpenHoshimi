//! egui desktop application for OpenHoshimi.

#![deny(missing_docs)]
#![forbid(unsafe_code)]

use std::collections::{BTreeMap, VecDeque};
use std::f32::consts::PI;
use std::path::{Path, PathBuf};
use std::sync::mpsc::{self, Receiver, Sender};
use std::thread;
use std::time::{Duration, Instant};

use eframe::egui;
use eframe::egui::{
    Align, Color32, ColorImage, CornerRadius, FontFamily, FontId, Frame, Layout, Margin, Pos2,
    ProgressBar, Rect, RichText, ScrollArea, Sense, Stroke, TextEdit, TextureOptions, Vec2,
};
use egui_extras::{Column, TableBuilder};
use openhoshimi_core::satellite::{load_all_satellites, DownlinkDef, SatelliteDefinition};
use openhoshimi_core::{
    Frame as CoreFrame, InputSource, IoError, IqSample, IqSource, TelemetryField, TelemetryValue,
};
use openhoshimi_io::{OggSource, SoundcardSource, WavIqSource, WavSource};
use openhoshimi_runtime::pipeline::{
    can_build_downlink, format_hex, format_timestamp, infer_tuning_offset_hz, input_kind_for,
    is_ao40_fec_downlink, is_linear_iq_modem, prepare_linear_iq_setup_scored,
    prepare_linear_iq_setup_scored_with_progress, BitPipeline, DecodedFrame, InputKind,
    SoftAo40Pipeline,
};
use openhoshimi_telemetry::SchemaParser;
use rfd::FileDialog;

const READ_CHUNK: usize = 4096;
const MAX_FRAMES: usize = 512;
const MAX_WATERFALL_ROWS: usize = 256;
const MAX_DIAGNOSTICS: usize = 24;
const SPECTRUM_BINS: usize = 1024;
const SPECTRUM_WINDOW: usize = 2048;
const DEFAULT_WATERFALL_MIN_DB: f32 = -90.0;
const DEFAULT_WATERFALL_MAX_DB: f32 = -10.0;
const WATERFALL_DB_FLOOR: f32 = -160.0;
const WATERFALL_DB_CEIL: f32 = 20.0;

/// Application entry point.
fn main() -> eframe::Result<()> {
    let options = eframe::NativeOptions::default();
    eframe::run_native(
        "OpenHoshimi",
        options,
        Box::new(|cc| Ok(Box::new(OpenHoshimiApp::new(cc)))),
    )
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
    hann_window: Vec<f32>,
    waterfall_texture: Option<egui::TextureHandle>,
    waterfall_min_db: f32,
    waterfall_max_db: f32,
    rx_thread: Option<thread::JoinHandle<()>>,
    rx_stop: Option<Sender<()>>,
    rx_events: Receiver<RxEvent>,
    rx_sender: Sender<RxEvent>,
    startup_wav: Option<PathBuf>,
    last_export_status: Option<String>,
    alignment_state: AlignmentState,
    cached_alignment: Option<CachedAlignment>,
    alignment_thread: Option<thread::JoinHandle<()>>,
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
        configure_style(&cc.egui_ctx);

        let satellites_dir = resolve_satellites_dir();
        let startup_wav = std::env::var_os("OPENHOSHIMI_OPEN_WAV").map(PathBuf::from);
        let (rx_sender, rx_events) = mpsc::channel();
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
            hann_window: build_hann_window(SPECTRUM_WINDOW),
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
        };
        app.reload_satellites();
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
        if matches!(self.input_mode, InputMode::Soundcard)
            && input_kind_for(&selected) == Some(InputKind::Iq)
        {
            self.status = "selected downlink requires IQ WAV input".to_string();
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
        let selected_label = selected.label.clone();
        let tuning_offset_hz = self.tuning_offset_for_selected(&selected, input_path.as_deref());
        let cached_alignment = self.cached_alignment.clone();
        let thread = thread::spawn(move || {
            run_decode_thread(
                satellite,
                selected,
                input_mode,
                decode_pace,
                input_path,
                tuning_offset_hz,
                cached_alignment,
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
            SpectrumSamples::Audio(samples) => spectrum_row_audio(
                &samples,
                SPECTRUM_BINS,
                self.input_rate,
                span_hz,
                &self.hann_window,
            ),
            SpectrumSamples::Iq(samples) => spectrum_row_iq(
                &samples,
                SPECTRUM_BINS,
                self.input_rate,
                span_hz,
                &self.hann_window,
            ),
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
            .exact_height(19.0)
            .frame(panel_frame(Palette::BAR))
            .show(ctx, |ui| self.status_bar(ui));

        egui::CentralPanel::default()
            .frame(Frame::new().fill(Palette::BG))
            .show(ctx, |ui| self.main_area(ui));

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
                    RichText::new("OpenHoshimi 0.1.0")
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
            egui::ComboBox::from_id_salt("input_source")
                .selected_text(self.input_source_label())
                .width(180.0)
                .show_ui(ui, |ui| {
                    ui.selectable_value(&mut self.input_mode, InputMode::Soundcard, "Soundcard");
                    ui.selectable_value(&mut self.input_mode, InputMode::WavFile, "WAV file");
                });
            if matches!(self.input_mode, InputMode::WavFile)
                && bracket_button(ui, "Open WAV...", 180.0).clicked()
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
                    RichText::new("OpenHoshimi 0.1.0")
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

    fn main_area(&mut self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            ui.set_height(ui.available_height());
            ui.allocate_ui_with_layout(
                Vec2::new(160.0, ui.available_height()),
                Layout::top_down(Align::Min),
                |ui| self.left_satellites(ui),
            );
            vline(ui);
            let right_width = 220.0;
            let center_width = (ui.available_width() - right_width - 2.0).max(320.0);
            ui.allocate_ui_with_layout(
                Vec2::new(center_width, ui.available_height()),
                Layout::top_down(Align::Min),
                |ui| self.center_workspace(ui),
            );
            vline(ui);
            ui.allocate_ui_with_layout(
                Vec2::new(right_width, ui.available_height()),
                Layout::top_down(Align::Min),
                |ui| self.right_panel(ui),
            );
        });
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
        self.frame_table(ui);
        self.hex_dump(ui);
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
        let desired = Vec2::new(ui.available_width(), 160.0);
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

    fn right_panel(&mut self, ui: &mut egui::Ui) {
        let input_height = 116.0;
        let diagnostics_height = 112.0;
        let telemetry_height =
            (ui.available_height() - input_height - diagnostics_height).max(180.0);
        ui.allocate_ui(Vec2::new(ui.available_width(), telemetry_height), |ui| {
            self.telemetry(ui)
        });
        self.input_panel(ui);
        self.diagnostics_panel(ui);
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
            .max_height(ui.available_height())
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
                    key_value(ui, "input", input_kind_label(input_kind_for(downlink)));
                }
                ui.add_space(2.0);
                ui.horizontal(|ui| {
                    label_muted(ui, "pace  ");
                    let pace_enabled = matches!(self.input_mode, InputMode::WavFile);
                    ui.add_enabled_ui(pace_enabled, |ui| {
                        ui.selectable_value(&mut self.decode_pace, DecodePace::Fast, "[Fast]");
                        ui.selectable_value(
                            &mut self.decode_pace,
                            DecodePace::Realtime,
                            "[Realtime]",
                        );
                    });
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
            .set_title("Open WAV input")
            .add_filter("WAV audio", &["wav"]);
        let Some(path) = dialog.pick_file() else {
            return;
        };
        self.open_wav_path(path);
    }

    fn open_wav_path(&mut self, path: PathBuf) {
        self.stop();
        self.input_path = Some(path.clone());
        self.input_mode = InputMode::WavFile;
        self.source_description = format!("WAV file {}", short_path_label(&path));
        self.clear_frames();
        self.waterfall.clear();
        self.samples_processed = 0;
        self.input_progress_processed = 0;
        self.input_progress_total = None;
        self.status = format!("selected WAV input {}", short_path_label(&path));
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
            InputMode::WavFile => "wav file",
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
            if let Some(downlink) = self.selected_downlink() {
                self.push_diagnostic(
                    DiagnosticLevel::Info,
                    format!("selected downlink {}", downlink_combo_label(downlink)),
                );
            }
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
                    input_kind_label(input_kind_for(downlink))
                )
            })
            .unwrap_or_else(|| "no downlink selected".to_string())
    }

    fn spectrum_span_hz(&self) -> u32 {
        if matches!(self.input_mode, InputMode::WavFile) {
            (self.input_rate / 2).clamp(6_000, 48_000)
        } else {
            6_000
        }
    }

    fn spectrum_scale_label(&self) -> String {
        let span = self.spectrum_span_hz() / 1000;
        format!("-{span}k   -{}k   0   +{}k   +{span}k", span / 2, span / 2)
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
fn run_decode_thread(
    satellite: SatelliteDefinition,
    downlink: DownlinkDef,
    input_mode: InputMode,
    decode_pace: DecodePace,
    input_path: Option<PathBuf>,
    tuning_offset_hz: f32,
    cached_alignment: Option<CachedAlignment>,
    events: Sender<RxEvent>,
    stop: Receiver<()>,
) {
    let _ = events.send(RxEvent::Dropped("priming WAV input\u{2026}".to_string()));
    let mut runtime = match build_runtime_input(
        input_mode,
        input_path,
        &downlink,
        tuning_offset_hz,
        cached_alignment.as_ref(),
        &events,
    ) {
        Ok(runtime) => runtime,
        Err(err) => {
            let _ = events.send(RxEvent::Dropped(err));
            let _ = events.send(RxEvent::Stopped);
            return;
        }
    };
    let _ = events.send(RxEvent::SourceInfo {
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
    let _ = events.send(RxEvent::Progress {
        processed: 0,
        total: total_samples,
    });
    let pace_realtime =
        matches!(input_mode, InputMode::WavFile) && matches!(decode_pace, DecodePace::Realtime);
    let start = Instant::now();
    let mut count = 0usize;
    let mut spectrum_stride = 0usize;

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
                        let _ = events.send(RxEvent::Dropped(format!("audio read failed: {err}")));
                        break;
                    }
                };
                if read == 0 {
                    continue;
                }
                spectrum_stride += 1;
                if spectrum_stride % 4 == 0 {
                    let _ = events.send(RxEvent::Samples(SpectrumSamples::Audio(
                        samples[..read.min(SPECTRUM_WINDOW)].to_vec(),
                    )));
                }
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
                let _ = events.send(RxEvent::Progress {
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
                    let _ = events.send(RxEvent::Progress {
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
                        let _ = events.send(RxEvent::Dropped(format!("IQ WAV read failed: {err}")));
                        break;
                    }
                };
                if read == 0 {
                    continue;
                }
                spectrum_stride += 1;
                if spectrum_stride % 4 == 0 {
                    let _ = events.send(RxEvent::Samples(SpectrumSamples::Iq(
                        samples[..read.min(SPECTRUM_WINDOW)].to_vec(),
                    )));
                }
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
                let _ = events.send(RxEvent::Progress {
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
                    let decoded = pipeline.push_samples(&samples);
                    emit_decoded_frames(
                        &events,
                        &satellite,
                        telemetry.as_ref(),
                        &start,
                        &mut count,
                        decoded,
                    );
                    let _ = events.send(RxEvent::Progress {
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
                        let _ = events.send(RxEvent::Dropped(format!("IQ WAV read failed: {err}")));
                        break;
                    }
                };
                if read == 0 {
                    continue;
                }
                spectrum_stride += 1;
                if spectrum_stride % 4 == 0 {
                    let _ = events.send(RxEvent::Samples(SpectrumSamples::Iq(
                        samples[..read.min(SPECTRUM_WINDOW)].to_vec(),
                    )));
                }
                let decoded = pipeline.push_samples(&samples[..read]);
                emit_decoded_frames(
                    &events,
                    &satellite,
                    telemetry.as_ref(),
                    &start,
                    &mut count,
                    decoded,
                );
                let _ = events.send(RxEvent::Progress {
                    processed: pipeline.total_samples(),
                    total: total_samples,
                });
                if pace_realtime
                    && !pace_realtime_loop(start, sample_rate, pipeline.total_samples(), &stop)
                {
                    break;
                }
            }
        }
    }

    let _ = events.send(RxEvent::Stopped);
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
}

impl RuntimeInput {
    fn description(&self) -> &str {
        match self {
            Self::Audio { source, .. } => source.description(),
            Self::Iq { source, .. } => source.description(),
            Self::IqSoftAo40 { source, .. } => source.description(),
        }
    }

    fn sample_rate(&self) -> u32 {
        match self {
            Self::Audio { source, .. } => source.sample_rate(),
            Self::Iq { source, .. } => source.sample_rate(),
            Self::IqSoftAo40 { source, .. } => source.sample_rate(),
        }
    }

    fn total_samples(&self) -> Option<u64> {
        match self {
            Self::Audio { source, .. } => source.total_samples(),
            Self::Iq { source, .. } => source.total_samples(),
            Self::IqSoftAo40 { source, .. } => source.total_samples(),
        }
    }
}

fn build_runtime_input(
    input_mode: InputMode,
    input_path: Option<PathBuf>,
    downlink: &DownlinkDef,
    tuning_offset_hz: f32,
    cached_alignment: Option<&CachedAlignment>,
    events: &Sender<RxEvent>,
) -> Result<RuntimeInput, String> {
    match input_mode {
        InputMode::Soundcard => {
            let source = SoundcardSource::open_default()
                .map_err(|err| format!("soundcard open failed: {err}"))?;
            let mut pipeline = BitPipeline::<f32>::new(downlink)?;
            pipeline.configure_demodulator(downlink, source.sample_rate(), 0.0)?;
            Ok(RuntimeInput::Audio {
                source: Box::new(source),
                pipeline,
            })
        }
        InputMode::WavFile => {
            let Some(path) = input_path else {
                return Err("missing WAV input path".to_string());
            };
            match input_kind_for(downlink) {
                Some(InputKind::Audio) => {
                    let source = WavSource::open(&path)
                        .map_err(|err| format!("failed to open WAV input: {err}"))?;
                    let mut pipeline = BitPipeline::<f32>::new(downlink)?;
                    pipeline.configure_demodulator(downlink, source.sample_rate(), 0.0)?;
                    Ok(RuntimeInput::Audio {
                        source: Box::new(source),
                        pipeline,
                    })
                }
                Some(InputKind::Iq) => {
                    let mut source = match WavIqSource::open(&path) {
                        Ok(s) => s,
                        Err(_) => {
                            // Mono file provided for an IQ downlink — fall back to FM audio path.
                            let source: Box<dyn InputSource> = open_audio_source(&path)?;
                            let sample_rate = source.sample_rate();
                            let mut pipeline = BitPipeline::<f32>::new(downlink)?;
                            pipeline.configure_fm_audio_demodulator(downlink, sample_rate)?;
                            return Ok(RuntimeInput::Audio { source, pipeline });
                        }
                    };
                    let sample_rate = source.sample_rate();
                    let downlink_id = format!("{downlink:?}");
                    let cache_hit = cached_alignment.filter(|cached| {
                        cached.path == path
                            && cached.sample_rate == sample_rate
                            && cached.downlink_id == downlink_id
                    });
                    let (setup, prefix) = if let Some(cached) = cache_hit {
                        let prefix_len = sample_rate as usize * 8;
                        let _ = read_iq_prefix(&mut source, prefix_len)?;
                        let _ = events.send(RxEvent::Dropped(format!(
                            "IQ alignment (cached): tuning={:.1} Hz skip={} prefix_frames={}",
                            cached.setup.tuning_offset_hz, cached.setup.sample_skip, cached.frames
                        )));
                        (cached.setup.clone(), cached.prefix.clone())
                    } else {
                        let prefix_len = sample_rate as usize * 8;
                        let _ = events.send(RxEvent::Dropped(format!(
                            "reading 8 s IQ prefix at {} Hz\u{2026}",
                            sample_rate
                        )));
                        let mut prefix = read_iq_prefix(&mut source, prefix_len)?;
                        let _ = events.send(RxEvent::Dropped(format!(
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
                                let _ = events.send(RxEvent::Dropped(format!(
                                    "IQ alignment: tuning={:.1} Hz skip={} prefix_frames={}",
                                    setup.tuning_offset_hz, setup.sample_skip, frames
                                )));
                                setup.clone()
                            }
                            None => {
                                let _ = events.send(RxEvent::Dropped(format!(
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

fn run_alignment_thread(
    path: PathBuf,
    downlink: DownlinkDef,
    tuning_offset_hz: f32,
    events: Sender<RxEvent>,
) {
    let downlink_id = format!("{downlink:?}");
    let mut source = match WavIqSource::open(&path) {
        Ok(source) => source,
        Err(err) => {
            let _ = events.send(RxEvent::AlignmentFailed(format!(
                "failed to open IQ WAV input: {err}"
            )));
            return;
        }
    };
    let sample_rate = source.sample_rate();

    // Non-linear modems (CPM/GMSK, AFSK, 4FSK, FM-audio) don't have a
    // linear-IQ alignment grid — fall straight through to TOML defaults
    // without reading the 8-second prefix or running the scorer.
    if !is_linear_iq_modem(&downlink) {
        let setup = openhoshimi_runtime::pipeline::LinearIqSetup {
            downlink: downlink.clone(),
            tuning_offset_hz,
            sample_skip: 0,
        };
        let cached = CachedAlignment {
            path,
            sample_rate,
            downlink_id,
            prefix: Vec::new(),
            setup,
            frames: 0,
        };
        let _ = events.send(RxEvent::AlignmentReady(Box::new(cached)));
        return;
    }

    let prefix_len = sample_rate as usize * 8;
    let mut prefix = match read_iq_prefix(&mut source, prefix_len) {
        Ok(prefix) => prefix,
        Err(err) => {
            let _ = events.send(RxEvent::AlignmentFailed(err));
            return;
        }
    };

    let mut announced = false;
    let scored = {
        let events_inner = events.clone();
        prepare_linear_iq_setup_scored_with_progress(
            &downlink,
            sample_rate,
            &prefix,
            tuning_offset_hz,
            &mut |current, total| {
                if !announced {
                    announced = true;
                    let _ = events_inner.send(RxEvent::AlignmentStarted { total });
                } else {
                    let _ = events_inner.send(RxEvent::AlignmentProgress { current });
                }
            },
        )
    };

    let (setup, frames) = match scored {
        Some(value) => value,
        None => {
            let _ = events.send(RxEvent::AlignmentFailed(
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
    let _ = events.send(RxEvent::AlignmentReady(Box::new(cached)));
}

fn emit_frames<P>(
    events: &Sender<RxEvent>,
    satellite: &SatelliteDefinition,
    telemetry: Option<&SchemaParser>,
    start: &Instant,
    pipeline: &BitPipeline<P>,
    count: &mut usize,
    frames: Vec<CoreFrame>,
) where
    P: Copy + Send + 'static,
{
    for mut frame in frames {
        *count += 1;
        frame.satellite_id = satellite.satellite.norad_id;
        let row = frame_row(*count, start.elapsed(), pipeline, &frame, telemetry);
        match row {
            Ok(row) => {
                let _ = events.send(RxEvent::Frame(row));
            }
            Err(err) => {
                let _ = events.send(RxEvent::Dropped(err));
            }
        }
    }
}

fn emit_decoded_frames(
    events: &Sender<RxEvent>,
    satellite: &SatelliteDefinition,
    telemetry: Option<&SchemaParser>,
    start: &Instant,
    count: &mut usize,
    frames: Vec<DecodedFrame>,
) {
    for decoded in frames {
        *count += 1;
        let row = decoded_frame_row(*count, start.elapsed(), satellite, decoded, telemetry);
        let _ = events.send(RxEvent::Frame(row));
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
        DecodedFrame::Ax100 { payload, .. } => {
            source = "AX100".to_string();
            destination = "FEC".to_string();
            kind = FrameKind::Tlm;
            if let Some(parser) = telemetry {
                fields = parser.parse_bytes(&payload);
            }
            raw = payload;
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
        DecodedFrame::Ao40 { payload, .. } | DecodedFrame::Ax100 { payload, .. } => {
            if let Some(parser) = telemetry {
                fields = parser.parse_bytes(&payload);
            }
            kind = FrameKind::Tlm;
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
        raw: frame.raw.clone(),
        telemetry: fields,
    })
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

fn vline(ui: &mut egui::Ui) {
    let (rect, _) = ui.allocate_exact_size(Vec2::new(1.0, ui.available_height()), Sense::hover());
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

fn spectrum_row_audio(
    samples: &[f32],
    bins: usize,
    sample_rate: u32,
    span_hz: u32,
    window: &[f32],
) -> Vec<f32> {
    if samples.is_empty() || bins == 0 || sample_rate == 0 {
        return vec![0.0; bins];
    }
    let n = samples.len().min(window.len());
    let samples = &samples[..n];
    let half_span = span_hz as f32;
    let frequencies = (0..bins).map(|bin| {
        let position = if bins == 1 {
            0.5
        } else {
            bin as f32 / (bins - 1) as f32
        };
        -half_span + 2.0 * half_span * position
    });
    let powers = frequencies
        .map(|frequency| dft_power_real(samples, window, sample_rate as f32, frequency.abs()))
        .collect::<Vec<_>>();
    power_row_to_db(&powers)
}

fn spectrum_row_iq(
    samples: &[IqSample],
    bins: usize,
    sample_rate: u32,
    span_hz: u32,
    window: &[f32],
) -> Vec<f32> {
    if samples.is_empty() || bins == 0 || sample_rate == 0 {
        return vec![0.0; bins];
    }
    let n = samples.len().min(window.len());
    let samples = &samples[..n];
    let half_span = span_hz as f32;
    let powers = (0..bins)
        .map(|bin| {
            let position = if bins == 1 {
                0.5
            } else {
                bin as f32 / (bins - 1) as f32
            };
            let frequency = -half_span + 2.0 * half_span * position;
            dft_power_iq(samples, window, sample_rate as f32, frequency)
        })
        .collect::<Vec<_>>();
    power_row_to_db(&powers)
}

fn dft_power_real(samples: &[f32], window: &[f32], sample_rate: f32, frequency_hz: f32) -> f32 {
    let n = samples.len().min(window.len());
    if n == 0 {
        return 0.0;
    }
    let step = (-2.0 * PI * frequency_hz / sample_rate).sin_cos();
    let (step_im, step_re) = step;
    let mut osc_re = 1.0;
    let mut osc_im = 0.0;
    let mut re = 0.0;
    let mut im = 0.0;
    for (index, sample) in samples.iter().take(n).enumerate() {
        let weight = window[index];
        re += sample * weight * osc_re;
        im += sample * weight * osc_im;
        let next_re = osc_re * step_re - osc_im * step_im;
        osc_im = osc_re * step_im + osc_im * step_re;
        osc_re = next_re;
    }
    (re * re + im * im) / n as f32
}

fn dft_power_iq(samples: &[IqSample], window: &[f32], sample_rate: f32, frequency_hz: f32) -> f32 {
    let n = samples.len().min(window.len());
    if n == 0 {
        return 0.0;
    }
    let step = (-2.0 * PI * frequency_hz / sample_rate).sin_cos();
    let (step_im, step_re) = step;
    let mut osc_re = 1.0;
    let mut osc_im = 0.0;
    let mut re = 0.0;
    let mut im = 0.0;
    for (index, sample) in samples.iter().take(n).enumerate() {
        let weight = window[index];
        re += weight * (sample.i * osc_re - sample.q * osc_im);
        im += weight * (sample.i * osc_im + sample.q * osc_re);
        let next_re = osc_re * step_re - osc_im * step_im;
        osc_im = osc_re * step_im + osc_im * step_re;
        osc_re = next_re;
    }
    (re * re + im * im) / n as f32
}

fn hann_window(index: usize, len: usize) -> f32 {
    if len <= 1 {
        return 1.0;
    }
    0.5 - 0.5 * (2.0 * PI * index as f32 / (len - 1) as f32).cos()
}

fn power_row_to_db(powers: &[f32]) -> Vec<f32> {
    powers
        .iter()
        .map(|power| {
            if *power <= 0.0 {
                WATERFALL_DB_FLOOR
            } else {
                (10.0 * power.log10()).clamp(WATERFALL_DB_FLOOR, WATERFALL_DB_CEIL)
            }
        })
        .collect()
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

fn lerp_color(a: Color32, b: Color32, t: f32) -> Color32 {
    let t = t.clamp(0.0, 1.0);
    let lerp = |start: u8, end: u8| -> u8 {
        (start as f32 + (end as f32 - start as f32) * t).clamp(0.0, 255.0) as u8
    };
    Color32::from_rgb(lerp(a.r(), b.r()), lerp(a.g(), b.g()), lerp(a.b(), b.b()))
}

fn build_hann_window(len: usize) -> Vec<f32> {
    (0..len).map(|index| hann_window(index, len)).collect()
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
    input_kind_for(downlink).is_some() && can_build_downlink(downlink)
}

fn input_kind_label(kind: Option<InputKind>) -> &'static str {
    match kind {
        Some(InputKind::Audio) => "audio",
        Some(InputKind::Iq) => "IQ",
        Some(InputKind::FmAudio) => "FM audio",
        None => "unsupported",
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
