//! Shared decode pipeline building blocks.

use std::path::Path;
use std::time::Duration;

use rayon::prelude::*;

use openhoshimi_codec::{
    ax100_syncword, Ao40FecDecoder, Ax100Decoder, Ax100Mode, Ax25Decoder, Ax25Frame, Callsign,
    GeoscanDecoder, GeoscanFrame,
};
use openhoshimi_core::satellite::{
    Ax100ModeDef, CodecDef, CpmModeDef, DescramblerDef, DownlinkDef, FramerDef, LineCodingDef,
    LinearModeDef, ModemDef, SatelliteDefinition,
};
use openhoshimi_core::{
    DecodeError, Demodulator, Descrambler, Frame, FrameType, Framing, IqSample, LineDecoder,
};
use openhoshimi_dsp::{
    bin_frequency_hz, fft_in_place, hann_window, Ao40Framer, CcsdsDescrambler, Complex, CpmConfig,
    CpmDemodulator, CpmMode, FmAudioConfig, FmAudioDemodulator, G3ruhDescrambler, HdlcFramer,
    LinearConfig, LinearDemodulator, LinearMode, NrziDecoder, SyncwordFramer,
};

/// Default closed-loop carrier tracker bandwidth for linear IQ demodulation,
/// expressed as a fraction of the symbol rate.
const DEFAULT_LINEAR_CARRIER_LOOP_BANDWIDTH: f32 = 0.005;

/// Input family required by a downlink.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InputKind {
    /// Audio-rate input samples (AFSK tone detection).
    Audio,
    /// IQ-rate input samples (full demodulation from complex baseband).
    Iq,
    /// FM-demodulated audio: real-valued symbol waveform from an FM receiver.
    /// Used when a mono recording is provided for an FSK/GMSK satellite.
    FmAudio,
}

/// A downlink chosen for decoding together with its input family.
#[derive(Debug, Clone, Copy)]
pub struct SelectedDownlink<'a> {
    /// The selected downlink definition.
    pub downlink: &'a DownlinkDef,
    /// The sample family required to decode this downlink.
    pub input_kind: InputKind,
}

/// A lightweight hint for starting IQ demodulation at the best symbol boundary.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct IqAlignmentHint {
    /// Number of raw IQ samples to skip before starting demodulation.
    pub sample_skip: usize,
    /// Whether differential decoding should be enabled.
    pub differential: bool,
    /// Whether the downlink should be inverted for the best match.
    pub invert: bool,
    /// Whether the in-phase and quadrature channels should be swapped.
    pub swap_iq: bool,
}

/// A resolved IQ setup for linear modulation inputs.
#[derive(Debug, Clone)]
pub struct LinearIqSetup {
    /// The downlink definition with the chosen IQ polarity applied.
    pub downlink: DownlinkDef,
    /// The frequency offset to apply in the demodulator.
    pub tuning_offset_hz: f32,
    /// Number of prefix samples to discard before demodulation starts.
    pub sample_skip: usize,
}

/// A decoded frame after the codec stage.
#[derive(Debug, Clone)]
pub enum DecodedFrame {
    /// AX.25 frame.
    Ax25(Ax25Frame),
    /// AO-40 FEC payload.
    Ao40 {
        /// Decoded user payload.
        payload: Vec<u8>,
        /// Number of corrected Reed-Solomon bytes.
        corrected_errors: usize,
    },
    /// GOMspace AX100 payload.
    Ax100 {
        /// AX100 framing mode.
        mode: Ax100Mode,
        /// Decoded user payload.
        payload: Vec<u8>,
        /// Number of corrected Reed-Solomon bytes.
        corrected_errors: usize,
    },
    /// Geoscan custom frame (CC11xx PN9-descrambled fixed-size payload).
    Geoscan(GeoscanFrame),
    /// Raw frame bytes from an unsupported codec.
    Raw {
        /// Raw frame type.
        frame_type: FrameType,
        /// Number of bytes in the payload.
        raw_len: usize,
    },
}

/// Select the first supported downlink in a satellite definition.
pub fn select_downlink(def: &SatelliteDefinition) -> Result<SelectedDownlink<'_>, String> {
    def.downlinks
        .iter()
        .find_map(|downlink| {
            input_kind_for(downlink).and_then(|input_kind| {
                if can_build_downlink(downlink) {
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

/// Return whether a downlink has enough supported stages to build a pipeline.
pub fn can_build_downlink(downlink: &DownlinkDef) -> bool {
    codec_kind(downlink).is_some() && framer_kind(downlink).is_some()
}

/// Determine which input family a downlink needs.
pub fn input_kind_for(downlink: &DownlinkDef) -> Option<InputKind> {
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

/// Format a timestamp as `mm:ss.mmm`.
pub fn format_timestamp(duration: Duration) -> String {
    let total_millis = duration.as_millis();
    let minutes = total_millis / 60_000;
    let seconds = (total_millis / 1_000) % 60;
    let millis = total_millis % 1_000;
    format!("{minutes:02}:{seconds:02}.{millis:03}")
}

/// Format an AX.25 callsign and SSID.
pub fn format_call(call: &Callsign) -> String {
    format!("{}-{}", call.call, call.ssid)
}

/// Format a byte slice as lowercase hexadecimal.
pub fn format_hex(bytes: &[u8]) -> String {
    let mut out = String::new();
    for (index, byte) in bytes.iter().enumerate() {
        if index > 0 {
            out.push(' ');
        }
        out.push_str(&format!("{byte:02x}"));
    }
    out
}

/// Convert a decoded frame into a short label.
pub fn frame_type_label(frame_type: FrameType) -> &'static str {
    match frame_type {
        FrameType::Ax25 => "ax25",
        FrameType::Ao40Fec => "ao40-fec",
        FrameType::GomspaceAx100 => "ax100",
        FrameType::Ccsds => "ccsds",
        FrameType::Fx25 => "fx25",
        FrameType::Geoscan => "geoscan",
        FrameType::Unknown => "unknown",
    }
}

/// Convert a selected downlink codec into a frame type.
pub fn frame_type_for_codec(downlink: &DownlinkDef) -> FrameType {
    match codec_kind(downlink) {
        Some(CodecKind::Ax25) => FrameType::Ax25,
        Some(CodecKind::Ao40Fec) => FrameType::Ao40Fec,
        Some(CodecKind::GomspaceAx100(_)) => FrameType::GomspaceAx100,
        Some(CodecKind::Ccsds) => FrameType::Ccsds,
        Some(CodecKind::Geoscan) => FrameType::Geoscan,
        Some(CodecKind::Unknown) | None => FrameType::Unknown,
    }
}

/// Infer a tuning offset from a recording filename that embeds the
/// recording center frequency, for example `145941kHz`.
pub fn infer_tuning_offset_hz(path: &Path, downlink_freq_hz: u64) -> Option<i32> {
    let name = path.file_name()?.to_str()?;
    let lower = name.to_ascii_lowercase();
    for (unit, scale) in [("mhz", 1_000_000f64), ("khz", 1_000f64), ("hz", 1f64)] {
        let Some(unit_pos) = lower.rfind(unit) else {
            continue;
        };
        let mut start = unit_pos;
        let bytes = name.as_bytes();
        while start > 0 {
            let byte = bytes[start - 1];
            if byte.is_ascii_digit() || byte == b'.' {
                start -= 1;
            } else {
                break;
            }
        }
        if start == unit_pos {
            continue;
        }
        let value = name[start..unit_pos].parse::<f64>().ok()?;
        let center_hz = (value * scale).round() as i64;
        let offset_hz = downlink_freq_hz as i64 - center_hz;
        if let Ok(offset) = i32::try_from(offset_hz) {
            return Some(offset);
        }
    }
    None
}

/// Search a linear IQ recording prefix for a better symbol boundary.
///
/// The search is intentionally simple: it tries the configured linear modem
/// with both inversion polarities, both I/Q orderings, and every possible
/// integer sample offset within one symbol period, then scores each
/// candidate by the number of frames produced from the prefix.
pub fn search_linear_iq_alignment(
    downlink: &DownlinkDef,
    sample_rate: u32,
    prefix: &[IqSample],
    tuning_offset_hz: f32,
) -> Option<IqAlignmentHint> {
    search_linear_iq_alignment_scored(downlink, sample_rate, prefix, tuning_offset_hz)
        .map(|result| result.hint)
}

/// Search a linear IQ recording prefix and return the hint together with
/// the number of frames it produced.
pub fn search_linear_iq_alignment_scored(
    downlink: &DownlinkDef,
    sample_rate: u32,
    prefix: &[IqSample],
    tuning_offset_hz: f32,
) -> Option<IqAlignmentSearchResult> {
    let ModemDef::Linear {
        mode: _,
        frequency_offset_hz,
        differential,
        invert,
        swap_iq,
        carrier_loop_bandwidth: _,
        frequency_loop_bandwidth: _,
        nco_max_offset_hz: _,
        matched_filter_rolloff: _,
        matched_filter_span_symbols: _,
    } = downlink.modem.as_ref()?
    else {
        return None;
    };

    let samples_per_symbol = sample_rate as f32 / downlink.baudrate as f32;
    let symbol_span = samples_per_symbol.round() as usize;
    if symbol_span == 0 || prefix.len() < symbol_span * 2 {
        return None;
    }
    // Two-stage sample-skip search: a coarse half-symbol sweep to locate
    // the basin, then a fine eighth-symbol sweep around the best coarse
    // hit.  This is ~2x cheaper than the previous single-stage span/8
    // sweep while keeping the same final resolution.
    let coarse_step = (symbol_span / 2).max(1);
    let fine_step = (symbol_span / 8).max(1);
    let frequency_offset_hz = *frequency_offset_hz;

    // The 8 polarity combinations (differential x swap_iq x invert) are
    // independent: each builds its own BitPipeline and reads the prefix
    // without touching shared state, so the sweep parallelises cleanly.
    let combos: [(bool, bool, bool); 8] = [
        (*differential, *swap_iq, *invert),
        (*differential, *swap_iq, !*invert),
        (*differential, !*swap_iq, *invert),
        (*differential, !*swap_iq, !*invert),
        (!*differential, *swap_iq, *invert),
        (!*differential, *swap_iq, !*invert),
        (!*differential, !*swap_iq, *invert),
        (!*differential, !*swap_iq, !*invert),
    ];

    let per_combo = |&(candidate_differential, candidate_swap_iq, candidate_invert): &(
        bool,
        bool,
        bool,
    )|
     -> Option<IqAlignmentSearchResult> {
        let mut trial = downlink.clone();
        // Sync vector lives at col=0 (one symbol per 80), so it is
        // unaffected by adjacent-symbol ISI and a matched filter is
        // not needed for alignment search. Skipping it keeps the
        // O(N_polarity * N_skip) prefix sweep tractable.
        if let Some(ModemDef::Linear {
            differential,
            invert,
            swap_iq: trial_swap_iq,
            frequency_loop_bandwidth,
            matched_filter_rolloff,
            matched_filter_span_symbols,
            ..
        }) = trial.modem.as_mut()
        {
            *differential = candidate_differential;
            *invert = candidate_invert;
            *trial_swap_iq = candidate_swap_iq;
            // FLL would slew the NCO during trial scoring and hide
            // alignment at the candidate offset; disable it here so
            // the per-offset frame count reflects pure Costas lock.
            *frequency_loop_bandwidth = 0.0;
            *matched_filter_rolloff = 0.0;
            *matched_filter_span_symbols = 0;
        }

        let try_skip = |sample_skip: usize| -> Option<(usize, Option<usize>)> {
            if sample_skip >= prefix.len() {
                return None;
            }
            let mut pipeline = BitPipeline::<IqSample>::new(&trial).ok()?;
            pipeline
                .configure_demodulator(&trial, sample_rate, frequency_offset_hz + tuning_offset_hz)
                .ok()?;
            let frames = pipeline.push_samples(&prefix[sample_skip..]);
            Some((frames.len(), pipeline.best_sync_distance()))
        };

        // Coarse pass.
        let mut best_skip = 0usize;
        let mut best_frames = 0usize;
        let mut best_distance: Option<usize> = None;
        let mut sample_skip = 0usize;
        while sample_skip < symbol_span {
            if let Some((frame_count, distance)) = try_skip(sample_skip) {
                if score_better(best_frames, best_distance, frame_count, distance) {
                    best_frames = frame_count;
                    best_distance = distance;
                    best_skip = sample_skip;
                }
            }
            sample_skip += coarse_step;
        }

        // Fine pass around the best coarse skip.
        if fine_step < coarse_step {
            let lo = best_skip.saturating_sub(coarse_step);
            let hi = (best_skip + coarse_step).min(symbol_span);
            let mut sample_skip = lo;
            while sample_skip < hi {
                // Skip points the coarse pass already evaluated.
                if (sample_skip - lo) % coarse_step != 0 || sample_skip != best_skip {
                    if let Some((frame_count, distance)) = try_skip(sample_skip) {
                        if score_better(best_frames, best_distance, frame_count, distance) {
                            best_frames = frame_count;
                            best_distance = distance;
                            best_skip = sample_skip;
                        }
                    }
                }
                sample_skip += fine_step;
            }
        }

        if best_frames == 0 && best_distance.is_none() {
            return None;
        }
        Some(IqAlignmentSearchResult {
            hint: IqAlignmentHint {
                sample_skip: best_skip,
                differential: candidate_differential,
                invert: candidate_invert,
                swap_iq: candidate_swap_iq,
            },
            frames: best_frames,
            sync_distance: best_distance,
        })
    };

    combos
        .par_iter()
        .filter_map(per_combo)
        .reduce_with(|prev, next| {
            if score_better(
                prev.frames,
                prev.sync_distance,
                next.frames,
                next.sync_distance,
            ) {
                next
            } else {
                prev
            }
        })
}

/// Strictly-better alignment score comparator.
///
/// Higher frame count wins; on a tie, lower sync distance wins.
fn score_better(
    best_frames: usize,
    best_distance: Option<usize>,
    frame_count: usize,
    distance: Option<usize>,
) -> bool {
    match frame_count.cmp(&best_frames) {
        std::cmp::Ordering::Greater => true,
        std::cmp::Ordering::Equal => match (distance, best_distance) {
            (Some(current), Some(best)) => current < best,
            (Some(_), None) => true,
            _ => false,
        },
        std::cmp::Ordering::Less => false,
    }
}

/// A scored linear IQ alignment result.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct IqAlignmentSearchResult {
    /// The best alignment hint found for the prefix.
    pub hint: IqAlignmentHint,
    /// Number of frames recovered from the prefix using this hint.
    pub frames: usize,
    /// Best sync distance observed for the prefix, if available.
    pub sync_distance: Option<usize>,
}

/// Estimate the average carrier frequency offset of an IQ recording prefix.
pub fn estimate_iq_frequency_offset_hz(prefix: &[IqSample], sample_rate: u32) -> Option<f32> {
    if sample_rate == 0 || prefix.len() < 2 {
        return None;
    }

    let mut sum_re = 0.0f64;
    let mut sum_im = 0.0f64;
    for window in prefix.windows(2) {
        let a = window[0];
        let b = window[1];
        sum_re += (a.i as f64) * (b.i as f64) + (a.q as f64) * (b.q as f64);
        sum_im += (a.i as f64) * (b.q as f64) - (a.q as f64) * (b.i as f64);
    }

    let phase = sum_im.atan2(sum_re);
    Some((phase as f32) * sample_rate as f32 / std::f32::consts::TAU)
}

/// Estimate the carrier offset for a linear-modulation IQ recording prefix.
///
/// BPSK and DBPSK use a second-power transform to remove 180-degree data
/// reversals before estimating the tone. QPSK and OQPSK use a fourth-power
/// transform for the same reason. Other linear modes fall back to the raw IQ
/// estimate.
pub fn estimate_linear_iq_frequency_offset_hz(
    downlink: &DownlinkDef,
    prefix: &[IqSample],
    sample_rate: u32,
) -> Option<f32> {
    let power = match &downlink.modem {
        Some(ModemDef::Linear { mode, .. }) => match mode {
            LinearModeDef::Bpsk | LinearModeDef::Dbpsk => 2usize,
            LinearModeDef::Qpsk | LinearModeDef::Oqpsk => 4usize,
        },
        _ => return None,
    };

    fft_carrier_estimate_hz(prefix, sample_rate, power)
        .or_else(|| estimate_iq_frequency_offset_hz(prefix, sample_rate))
}

/// Estimate the carrier offset for a CPM (e.g. GMSK) IQ recording prefix.
///
/// CPM has a strong residual carrier component, so the spectrum peak of the
/// raw IQ samples (no nonlinear transform) tracks the carrier directly.
pub fn estimate_cpm_iq_frequency_offset_hz(prefix: &[IqSample], sample_rate: u32) -> Option<f32> {
    fft_carrier_estimate_hz(prefix, sample_rate, 1)
        .or_else(|| estimate_iq_frequency_offset_hz(prefix, sample_rate))
}

/// Return up to `max_carriers` candidate carrier frequencies for a linear
/// modulation prefix, ranked by spectral power after the appropriate
/// nonlinear transform.
pub fn linear_iq_carrier_candidates(
    downlink: &DownlinkDef,
    prefix: &[IqSample],
    sample_rate: u32,
    max_carriers: usize,
) -> Vec<f32> {
    let power = match &downlink.modem {
        Some(ModemDef::Linear { mode, .. }) => match mode {
            LinearModeDef::Bpsk | LinearModeDef::Dbpsk => 2usize,
            LinearModeDef::Qpsk | LinearModeDef::Oqpsk => 4usize,
        },
        _ => return Vec::new(),
    };
    fft_carrier_candidates_hz(prefix, sample_rate, power, max_carriers)
}

const CARRIER_FFT_SIZE: usize = 16_384;
const CARRIER_FFT_HOP: usize = CARRIER_FFT_SIZE / 2;
const CARRIER_PEAK_GUARD_BINS: usize = 6;

fn fft_carrier_estimate_hz(prefix: &[IqSample], sample_rate: u32, power: usize) -> Option<f32> {
    let candidates = fft_carrier_candidates_hz(prefix, sample_rate, power, 1);
    candidates.first().copied()
}

fn fft_carrier_candidates_hz(
    prefix: &[IqSample],
    sample_rate: u32,
    power: usize,
    max_carriers: usize,
) -> Vec<f32> {
    if max_carriers == 0 || sample_rate == 0 || prefix.len() < CARRIER_FFT_SIZE {
        return Vec::new();
    }
    let transformed: Vec<Complex> = prefix
        .iter()
        .map(|sample| iq_power_complex(*sample, power))
        .collect();
    let spectrum = welch_spectrum(&transformed);
    let bin_hz = sample_rate as f32 / CARRIER_FFT_SIZE as f32;
    let mut peaks = Vec::<(usize, f32)>::new();
    for i in 1..spectrum.len() - 1 {
        let here = spectrum[i];
        if here > spectrum[i - 1] && here > spectrum[i + 1] {
            peaks.push((i, here));
        }
    }
    peaks.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    let mut candidates: Vec<f32> = Vec::with_capacity(max_carriers);
    let mut taken_bins: Vec<i32> = Vec::with_capacity(max_carriers);
    for (bin, _power) in peaks {
        if !taken_bins
            .iter()
            .all(|other| (bin as i32 - other).unsigned_abs() as usize > CARRIER_PEAK_GUARD_BINS)
        {
            continue;
        }
        let freq = bin_frequency_hz(bin, CARRIER_FFT_SIZE, sample_rate) / power as f32;
        let _ = bin_hz;
        candidates.push(freq);
        taken_bins.push(bin as i32);
        if candidates.len() == max_carriers {
            break;
        }
    }
    candidates
}

fn welch_spectrum(samples: &[Complex]) -> Vec<f32> {
    let mut accum = vec![0.0f32; CARRIER_FFT_SIZE];
    let window = hann_window(CARRIER_FFT_SIZE);
    let window_power: f32 = window.iter().map(|w| w * w).sum();
    let mut frames = 0usize;
    let mut start = 0usize;
    while start + CARRIER_FFT_SIZE <= samples.len() {
        let mut buf = vec![Complex::default(); CARRIER_FFT_SIZE];
        for k in 0..CARRIER_FFT_SIZE {
            let s = samples[start + k];
            let w = window[k];
            buf[k] = Complex::new(s.re * w, s.im * w);
        }
        fft_in_place(&mut buf);
        for (acc, c) in accum.iter_mut().zip(buf.iter()) {
            *acc += c.norm_sqr();
        }
        frames += 1;
        start += CARRIER_FFT_HOP;
    }
    let scale = 1.0 / (frames.max(1) as f32 * window_power.max(f32::EPSILON));
    for value in accum.iter_mut() {
        *value *= scale;
    }
    accum
}

fn iq_power_complex(sample: IqSample, power: usize) -> Complex {
    let mut acc = Complex::new(sample.i, sample.q);
    let base = acc;
    for _ in 1..power {
        acc = acc * base;
    }
    acc
}

/// Resolve a linear IQ setup from a prefix and a nominal tuning offset.
pub fn prepare_linear_iq_setup(
    downlink: &DownlinkDef,
    sample_rate: u32,
    prefix: &[IqSample],
    tuning_offset_hz: f32,
) -> Option<LinearIqSetup> {
    let ModemDef::Linear { .. } = downlink.modem.as_ref()? else {
        return None;
    };

    let candidates = linear_iq_tuning_candidates(downlink, sample_rate, prefix, tuning_offset_hz);
    let mut best: Option<(LinearIqSetup, usize)> = None;
    let mut best_distance: Option<usize> = None;
    let mut fallback = LinearIqSetup {
        downlink: downlink.clone(),
        tuning_offset_hz,
        sample_skip: 0,
    };

    for &candidate in &candidates {
        if let Some(result) =
            search_linear_iq_alignment_scored(downlink, sample_rate, prefix, candidate)
        {
            let mut aligned_downlink = downlink.clone();
            if let Some(ModemDef::Linear {
                differential,
                invert,
                swap_iq,
                ..
            }) = aligned_downlink.modem.as_mut()
            {
                *differential = result.hint.differential;
                *invert = result.hint.invert;
                *swap_iq = result.hint.swap_iq;
            }
            let setup = LinearIqSetup {
                downlink: aligned_downlink,
                tuning_offset_hz: candidate,
                sample_skip: result.hint.sample_skip,
            };
            if better_alignment(
                best.as_ref().map(|(_, frames)| *frames),
                best_distance,
                result.frames,
                result.sync_distance,
            ) {
                best_distance = result.sync_distance;
                best = Some((setup, result.frames));
            }
        }
    }

    if let Some((setup, _)) = best {
        return Some(setup);
    }

    fallback.tuning_offset_hz =
        estimate_linear_iq_frequency_offset_hz(downlink, prefix, sample_rate)
            .filter(|value| value.is_finite())
            .unwrap_or(tuning_offset_hz);
    Some(fallback)
}

/// Return a reusable linear-IQ setup if one can be inferred from the prefix.
pub fn prepare_linear_iq_setup_scored(
    downlink: &DownlinkDef,
    sample_rate: u32,
    prefix: &[IqSample],
    tuning_offset_hz: f32,
) -> Option<(LinearIqSetup, usize)> {
    prepare_linear_iq_setup_scored_with_progress(
        downlink,
        sample_rate,
        prefix,
        tuning_offset_hz,
        &mut |_, _| {},
    )
}

/// Same as [`prepare_linear_iq_setup_scored`] but reports scoring progress
/// through the supplied callback.  The callback is invoked as
/// `(completed, total)` once before scoring begins (with `completed == 0`)
/// and once after each candidate has been processed, so a UI can drive a
/// progress bar without polling.
pub fn prepare_linear_iq_setup_scored_with_progress(
    downlink: &DownlinkDef,
    sample_rate: u32,
    prefix: &[IqSample],
    tuning_offset_hz: f32,
    progress: &mut dyn FnMut(usize, usize),
) -> Option<(LinearIqSetup, usize)> {
    let ModemDef::Linear { .. } = downlink.modem.as_ref()? else {
        return None;
    };

    let candidates = linear_iq_tuning_candidates(downlink, sample_rate, prefix, tuning_offset_hz);
    let total = candidates.len();
    progress(0, total);
    let mut best: Option<(LinearIqSetup, usize)> = None;
    let mut best_distance: Option<usize> = None;

    for (idx, &candidate) in candidates.iter().enumerate() {
        if let Some(result) =
            search_linear_iq_alignment_scored(downlink, sample_rate, prefix, candidate)
        {
            #[cfg(debug_assertions)]
            eprintln!(
                "openhoshimi: candidate={candidate:+.1} Hz -> frames={} sync_distance={:?}",
                result.frames, result.sync_distance,
            );
            let mut aligned_downlink = downlink.clone();
            if let Some(ModemDef::Linear {
                differential,
                invert,
                swap_iq,
                ..
            }) = aligned_downlink.modem.as_mut()
            {
                *differential = result.hint.differential;
                *invert = result.hint.invert;
                *swap_iq = result.hint.swap_iq;
            }
            let setup = LinearIqSetup {
                downlink: aligned_downlink,
                tuning_offset_hz: candidate,
                sample_skip: result.hint.sample_skip,
            };
            if better_alignment(
                best.as_ref().map(|(_, frames)| *frames),
                best_distance,
                result.frames,
                result.sync_distance,
            ) {
                best_distance = result.sync_distance;
                best = Some((setup, result.frames));
            }
            // High-confidence early exit: two-or-more frames with the
            // best possible sync distance (zero, or unknown for framers
            // that don't expose it like SyncwordFramer/HdlcFramer)
            // means this candidate is almost certainly the right one.
            // Keep scanning only if this candidate looks ambiguous.
            let confident = result.frames >= 2 && matches!(result.sync_distance, Some(0) | None);
            if confident {
                progress(total, total);
                return best;
            }
        }
        progress(idx + 1, total);
    }

    if best.is_some() {
        return best;
    }

    Some((
        LinearIqSetup {
            downlink: downlink.clone(),
            tuning_offset_hz: estimate_linear_iq_frequency_offset_hz(downlink, prefix, sample_rate)
                .filter(|value| value.is_finite())
                .unwrap_or(tuning_offset_hz),
            sample_skip: 0,
        },
        0,
    ))
}

fn better_alignment(
    best_frames: Option<usize>,
    best_distance: Option<usize>,
    frames: usize,
    distance: Option<usize>,
) -> bool {
    let best_frames = best_frames.unwrap_or(0);
    if frames > best_frames {
        return true;
    }
    if frames < best_frames {
        return false;
    }
    if frames == 0 {
        return false;
    }
    match (distance, best_distance) {
        (Some(current), Some(best)) => current < best,
        (Some(_), None) => true,
        _ => false,
    }
}

fn linear_iq_tuning_candidates(
    downlink: &DownlinkDef,
    sample_rate: u32,
    prefix: &[IqSample],
    tuning_offset_hz: f32,
) -> Vec<f32> {
    let mut candidates = Vec::with_capacity(8);
    push_candidate(&mut candidates, tuning_offset_hz);
    push_candidate(&mut candidates, -tuning_offset_hz);

    for carrier in linear_iq_carrier_candidates(downlink, prefix, sample_rate, 4) {
        push_candidate(&mut candidates, carrier);
        push_candidate(&mut candidates, -carrier);
    }

    if let Some(estimate) =
        estimate_iq_frequency_offset_hz(prefix, sample_rate).filter(|value| value.is_finite())
    {
        push_candidate(&mut candidates, estimate);
        push_candidate(&mut candidates, -estimate);
    }

    candidates
}

fn push_candidate(candidates: &mut Vec<f32>, candidate: f32) {
    if !candidate.is_finite() {
        return;
    }
    if candidates
        .iter()
        .any(|value| (value - candidate).abs() < 1.0)
    {
        return;
    }
    candidates.push(candidate);
}

/// Hard-decision pipeline from samples to decoded frames.
pub struct BitPipeline<S>
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
    /// Configure the audio demodulator for this pipeline.
    pub fn configure_demodulator(
        &mut self,
        downlink: &DownlinkDef,
        sample_rate: u32,
        _tuning_offset_hz: f32,
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

    /// Configure the FM-audio symbol demodulator for this pipeline.
    ///
    /// Used when the input is a mono recording from an FM receiver and the
    /// satellite uses FSK/GMSK modulation.
    pub fn configure_fm_audio_demodulator(
        &mut self,
        downlink: &DownlinkDef,
        sample_rate: u32,
    ) -> Result<(), String> {
        let config = fm_audio_config(downlink, sample_rate)?;
        self.demodulator = Some(Box::new(
            FmAudioDemodulator::new(config)
                .map_err(|err| format!("failed to configure FM audio demodulator: {err}"))?,
        ));
        Ok(())
    }
}

impl BitPipeline<IqSample> {
    /// Configure the IQ demodulator for this pipeline.
    pub fn configure_demodulator(
        &mut self,
        downlink: &DownlinkDef,
        sample_rate: u32,
        tuning_offset_hz: f32,
    ) -> Result<(), String> {
        self.demodulator = Some(build_iq_demodulator(
            downlink,
            sample_rate,
            tuning_offset_hz,
        )?);
        Ok(())
    }
}

impl<S> BitPipeline<S>
where
    S: Copy + Send + 'static,
{
    /// Build a pipeline from a downlink definition.
    pub fn new(downlink: &DownlinkDef) -> Result<Self, String> {
        Ok(Self {
            demodulator: None,
            line_decoder: build_line_decoder(downlink)?,
            descrambler: build_descrambler(downlink)?,
            framer: build_framer(downlink)?,
            codec: build_codec(downlink)?,
            total_samples: 0,
        })
    }

    /// Feed a chunk of samples into the pipeline.
    pub fn push_samples(&mut self, samples: &[S]) -> Vec<Frame> {
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

    /// Decode a framed payload.
    pub fn decode_frame(&self, frame: &Frame) -> Result<DecodedFrame, String> {
        self.codec.decode(frame)
    }

    /// Number of input samples processed by this pipeline.
    pub fn total_samples(&self) -> u64 {
        self.total_samples
    }

    /// The last HDLC framing error, if any.
    pub fn last_framer_error(&self) -> Option<&DecodeError> {
        self.framer.last_error()
    }

    /// Raw byte buffer of the most recent frame that failed validation.
    /// Currently only HDLC frames carry this for diagnostic comparison.
    pub fn last_failed_bytes(&self) -> Option<&[u8]> {
        self.framer.last_failed_bytes()
    }

    /// Raw byte buffer of the longest frame that failed validation across
    /// the entire stream. Useful when the most recent failure is a short
    /// tail fragment that overwrites the diagnostically interesting one.
    pub fn longest_failed_bytes(&self) -> Option<&[u8]> {
        self.framer.longest_failed_bytes()
    }

    /// Best AO-40 sync distance observed by the active framer, if any.
    pub fn best_sync_distance(&self) -> Option<usize> {
        self.framer.best_sync_distance()
    }
}

#[derive(Debug, Clone, Copy)]
struct AfskModemConfig {
    mark_hz: f32,
    space_hz: f32,
}

/// Build an [`FmAudioConfig`] from a downlink definition.
///
/// Extracts Gaussian BT, differential, and invert settings from the CPM modem
/// definition if present, otherwise uses sensible defaults for the modulation.
fn fm_audio_config(downlink: &DownlinkDef, sample_rate: u32) -> Result<FmAudioConfig, String> {
    match &downlink.modem {
        Some(ModemDef::Cpm {
            gaussian_bt,
            differential,
            invert,
            ..
        }) => {
            let mut config = match gaussian_bt {
                Some(bt) => FmAudioConfig::gmsk(sample_rate, downlink.baudrate, *bt),
                None => FmAudioConfig::new(sample_rate, downlink.baudrate),
            };
            config.differential = *differential;
            config.invert = *invert;
            Ok(config)
        }
        Some(_) => Err(format!(
            "{} is not an FSK/GMSK downlink suitable for FM audio",
            downlink.label
        )),
        None if matches_token(&downlink.modulation, &["FSK", "MSK", "GFSK", "GMSK"]) => {
            let config = if matches_token(&downlink.modulation, &["GMSK", "GFSK"]) {
                FmAudioConfig::gmsk(sample_rate, downlink.baudrate, 0.5)
            } else {
                FmAudioConfig::new(sample_rate, downlink.baudrate)
            };
            Ok(config)
        }
        None => Err(format!(
            "{} has no supported FM audio modem configuration",
            downlink.label
        )),
    }
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
    tuning_offset_hz: f32,
) -> Result<Box<dyn Demodulator<Sample = IqSample>>, String> {
    match &downlink.modem {
        Some(ModemDef::Cpm {
            mode,
            modulation_index,
            frequency_offset_hz,
            gaussian_bt,
            differential,
            invert,
            swap_iq,
        }) => {
            let mut config = CpmConfig::new(sample_rate, downlink.baudrate, map_cpm_mode(*mode));
            if let Some(value) = modulation_index {
                config.modulation_index = *value;
            }
            config.frequency_offset_hz = *frequency_offset_hz + tuning_offset_hz;
            if gaussian_bt.is_some() {
                config.gaussian_bt = *gaussian_bt;
            }
            config.differential = *differential;
            config.invert = *invert;
            config.swap_iq = *swap_iq;
            Ok(Box::new(CpmDemodulator::new(config).map_err(|err| {
                format!("failed to configure CPM demodulator: {err}")
            })?))
        }
        Some(ModemDef::Linear {
            mode,
            frequency_offset_hz,
            differential,
            invert,
            swap_iq,
            carrier_loop_bandwidth,
            frequency_loop_bandwidth,
            nco_max_offset_hz,
            matched_filter_rolloff,
            matched_filter_span_symbols,
        }) => {
            let mut config =
                LinearConfig::new(sample_rate, downlink.baudrate, map_linear_mode(*mode));
            config.frequency_offset_hz = *frequency_offset_hz + tuning_offset_hz;
            config.differential = *differential;
            config.invert = *invert;
            config.swap_iq = *swap_iq;
            config.carrier_loop_bandwidth = if *carrier_loop_bandwidth > 0.0 {
                *carrier_loop_bandwidth
            } else {
                DEFAULT_LINEAR_CARRIER_LOOP_BANDWIDTH
            };
            config.frequency_loop_bandwidth = *frequency_loop_bandwidth;
            config.nco_max_offset_hz = *nco_max_offset_hz;
            config.matched_filter_rolloff = *matched_filter_rolloff;
            config.matched_filter_span_symbols = *matched_filter_span_symbols;
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
            let mut config = CpmConfig::new(sample_rate, downlink.baudrate, mode);
            config.frequency_offset_hz = tuning_offset_hz;
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
            let mut config = LinearConfig::new(sample_rate, downlink.baudrate, mode);
            config.frequency_offset_hz = tuning_offset_hz;
            config.carrier_loop_bandwidth = DEFAULT_LINEAR_CARRIER_LOOP_BANDWIDTH;
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

    fn last_error(&self) -> Option<&DecodeError> {
        match self {
            Self::Ao40(_) => None,
            Self::Hdlc(framer) => framer.last_error(),
            Self::Syncword(_) => None,
        }
    }

    fn last_failed_bytes(&self) -> Option<&[u8]> {
        match self {
            Self::Hdlc(framer) => framer.last_failed_bytes(),
            Self::Ao40(_) | Self::Syncword(_) => None,
        }
    }

    fn longest_failed_bytes(&self) -> Option<&[u8]> {
        match self {
            Self::Hdlc(framer) => framer.longest_failed_bytes(),
            Self::Ao40(_) | Self::Syncword(_) => None,
        }
    }

    fn best_sync_distance(&self) -> Option<usize> {
        match self {
            Self::Ao40(framer) => framer.best_sync_distance(),
            Self::Hdlc(_) => None,
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
        Some(FramerKind::Ax100Asm { threshold }) => {
            let asm = ax100_syncword();
            let syncword: Vec<u8> = (0..32).rev().map(|i| ((asm >> i) & 1) as u8).collect();
            Ok(FrameStage::Syncword(SyncwordFramer::with_frame_options(
                &syncword,
                threshold,
                AX100_ASM_GOLAY_PAYLOAD_BITS,
                FrameType::GomspaceAx100,
                true,
            )))
        }
        None => Err(format!("{} has no supported framer", downlink.label)),
    }
}

/// Total bits collected after the AX100 attached sync marker: 3 bytes of
/// Golay(24,12) header followed by a 255-byte CCSDS-randomised RS(255,223)
/// codeword.
const AX100_ASM_GOLAY_PAYLOAD_BITS: usize = (3 + 255) * 8;

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
    Ax100Asm {
        threshold: usize,
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
        Some(FramerDef::Ax100Asm { threshold }) => Some(FramerKind::Ax100Asm {
            threshold: *threshold,
        }),
        None if downlink.framing.eq_ignore_ascii_case("AO40_FEC") => {
            Some(FramerKind::Ao40 { threshold: 0 })
        }
        None if matches_token(&downlink.framing, &["AX25", "AX.25", "HDLC"]) => {
            Some(FramerKind::Hdlc)
        }
        None if downlink.framing.eq_ignore_ascii_case("GOMSPACE_AX100") => {
            Some(FramerKind::Ax100Asm { threshold: 4 })
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

fn codec_kind(downlink: &DownlinkDef) -> Option<CodecKind> {
    match &downlink.codec {
        Some(CodecDef::Ax25) => Some(CodecKind::Ax25),
        Some(CodecDef::Ao40Fec) => Some(CodecKind::Ao40Fec),
        Some(CodecDef::GomspaceAx100 { mode }) => Some(CodecKind::GomspaceAx100(match mode {
            Ax100ModeDef::ReedSolomon => Ax100Mode::ReedSolomon,
            Ax100ModeDef::AsmGolay => Ax100Mode::AsmGolay,
        })),
        Some(CodecDef::Unknown) => Some(CodecKind::Unknown),
        Some(CodecDef::Ccsds) => Some(CodecKind::Ccsds),
        Some(CodecDef::Geoscan) => Some(CodecKind::Geoscan),
        Some(CodecDef::Fx25) => None,
        None if matches_token(&downlink.framing, &["AX25", "AX.25", "HDLC"]) => {
            Some(CodecKind::Ax25)
        }
        None if downlink.framing.eq_ignore_ascii_case("AO40_FEC") => Some(CodecKind::Ao40Fec),
        None if downlink.framing.eq_ignore_ascii_case("GOMSPACE_AX100") => {
            Some(CodecKind::GomspaceAx100(Ax100Mode::AsmGolay))
        }
        None if downlink.framing.eq_ignore_ascii_case("GEOSCAN") => Some(CodecKind::Geoscan),
        None if downlink.framing.eq_ignore_ascii_case("UNKNOWN") => Some(CodecKind::Unknown),
        None => None,
    }
}

enum CodecKind {
    Ax25,
    Ao40Fec,
    GomspaceAx100(Ax100Mode),
    Ccsds,
    Geoscan,
    Unknown,
}

enum CodecStage {
    Ax25(Ax25Decoder),
    Ao40(Ao40FecDecoder),
    Ax100 {
        decoder: Ax100Decoder,
        mode: Ax100Mode,
    },
    Ccsds,
    Geoscan(GeoscanDecoder),
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
            Self::Geoscan(decoder) => {
                let geoscan = decoder
                    .decode_frame(frame)
                    .map_err(|err| format!("Geoscan decode failed: {err}"))?;
                Ok(DecodedFrame::Geoscan(geoscan))
            }
            Self::Ccsds | Self::Unknown => Ok(DecodedFrame::Raw {
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
        Some(CodecKind::Ccsds) => Ok(CodecStage::Ccsds),
        Some(CodecKind::Geoscan) => Ok(CodecStage::Geoscan(GeoscanDecoder::new())),
        Some(CodecKind::Unknown) => Ok(CodecStage::Unknown),
        None => Err(format!("{} has no supported codec", downlink.label)),
    }
}

fn matches_token(value: &str, tokens: &[&str]) -> bool {
    tokens.iter().any(|token| value.eq_ignore_ascii_case(token))
}

/// Whether a downlink uses the AO-40 FEC codec.
pub fn is_ao40_fec_downlink(downlink: &DownlinkDef) -> bool {
    matches!(codec_kind(downlink), Some(CodecKind::Ao40Fec))
}

/// Whether a downlink uses a linear-IQ modem (BPSK / QPSK / etc.).
///
/// Used by callers to decide whether running the linear-IQ alignment
/// helpers ([`prepare_linear_iq_setup`] and friends) is worthwhile.
/// Non-linear modems (CPM/GMSK, AFSK, 4FSK, FM-audio) skip alignment
/// and run from TOML defaults.
pub fn is_linear_iq_modem(downlink: &DownlinkDef) -> bool {
    matches!(downlink.modem.as_ref(), Some(ModemDef::Linear { .. }))
}

/// Soft-decision IQ pipeline for AO-40 FEC downlinks.
///
/// This pipeline mirrors [`BitPipeline`] but keeps per-bit confidence
/// information through the slicer, framer and Viterbi decoder. The
/// soft-decision Viterbi recovers roughly 2 dB of coding gain compared
/// with the hard-decision path, which is the difference between
/// dropping every frame at the AO-73 link margin and producing
/// CRC-valid telemetry.
pub struct SoftAo40Pipeline {
    demodulator: LinearDemodulator,
    framer: Ao40Framer,
    codec: Ao40FecDecoder,
    sample_rate: u32,
    total_samples: u64,
    last_decoded_frames: usize,
}

impl SoftAo40Pipeline {
    /// Build a soft-decision AO-40 pipeline from a linear-IQ downlink.
    pub fn new(
        downlink: &DownlinkDef,
        sample_rate: u32,
        tuning_offset_hz: f32,
    ) -> Result<Self, String> {
        if !is_ao40_fec_downlink(downlink) {
            return Err(format!("{} is not an AO-40 FEC downlink", downlink.label));
        }
        let config = linear_config_for(downlink, sample_rate, tuning_offset_hz)?;
        let demodulator = LinearDemodulator::new(config)
            .map_err(|err| format!("failed to configure linear demodulator: {err}"))?;
        let threshold = match &downlink.framer {
            Some(FramerDef::Ao40 { threshold }) => *threshold,
            _ => 0,
        };
        Ok(Self {
            demodulator,
            framer: Ao40Framer::new(threshold),
            codec: Ao40FecDecoder::new(),
            sample_rate,
            total_samples: 0,
            last_decoded_frames: 0,
        })
    }

    /// Feed IQ samples into the soft-decision pipeline.
    pub fn push_samples(&mut self, samples: &[IqSample]) -> Vec<DecodedFrame> {
        self.total_samples += samples.len() as u64;
        let soft = self.demodulator.push_samples_soft(samples);
        let frames = self.framer.push_soft_bytes(&soft);
        let mut decoded = Vec::new();
        for frame in frames {
            let hard_bits: Vec<u8> = frame.iter().map(|&s| u8::from(s < 0)).collect();
            let ones = hard_bits.iter().filter(|&&b| b == 1).count();
            let transitions = hard_bits.windows(2).filter(|w| w[0] != w[1]).count();
            let abs_sum: u64 = frame.iter().map(|&s| i64::from(s).unsigned_abs()).sum();
            let mean_abs = abs_sum as f32 / frame.len() as f32;
            eprintln!(
                "openhoshimi: ao40 frame stats: len={} ones={}/{} ({:.1}%) transitions={}/{} ({:.1}%) mean_abs_soft={:.1}",
                frame.len(),
                ones,
                frame.len(),
                100.0 * ones as f32 / frame.len() as f32,
                transitions,
                frame.len() - 1,
                100.0 * transitions as f32 / (frame.len() - 1) as f32,
                mean_abs,
            );
            match self.codec.decode_soft_channel_bits(&frame) {
                Ok(payload) => decoded.push(DecodedFrame::Ao40 {
                    payload: payload.payload,
                    corrected_errors: payload.corrected_errors,
                }),
                Err(err) => {
                    let hard_result = self.codec.decode_channel_bits(&hard_bits);
                    let hard_summary = match &hard_result {
                        Ok(payload) => format!("hard OK ({} corrected)", payload.corrected_errors),
                        Err(hard_err) => format!("hard ERR {hard_err}"),
                    };
                    eprintln!("openhoshimi: soft Ao40 frame decode failed: {err}; {hard_summary}");
                    if let Ok(payload) = hard_result {
                        decoded.push(DecodedFrame::Ao40 {
                            payload: payload.payload,
                            corrected_errors: payload.corrected_errors,
                        });
                    }
                }
            }
        }
        self.last_decoded_frames = decoded.len();
        decoded
    }

    /// Sample rate the demodulator was built with.
    pub fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    /// Number of input samples processed by this pipeline.
    pub fn total_samples(&self) -> u64 {
        self.total_samples
    }

    /// Best AO-40 sync distance observed by the framer, if any.
    pub fn best_sync_distance(&self) -> Option<usize> {
        self.framer.best_sync_distance()
    }

    /// Per-second history of minimum AO-40 sync distance.
    ///
    /// See [`Ao40Framer::distance_history`] for semantics.
    pub fn distance_history(&self) -> &[Option<usize>] {
        self.framer.distance_history()
    }
}

fn linear_config_for(
    downlink: &DownlinkDef,
    sample_rate: u32,
    tuning_offset_hz: f32,
) -> Result<LinearConfig, String> {
    let Some(ModemDef::Linear {
        mode,
        frequency_offset_hz,
        differential,
        invert,
        swap_iq,
        carrier_loop_bandwidth,
        frequency_loop_bandwidth,
        nco_max_offset_hz,
        matched_filter_rolloff,
        matched_filter_span_symbols,
    }) = downlink.modem.as_ref()
    else {
        return Err(format!(
            "{} is not a linear-modulation downlink",
            downlink.label
        ));
    };
    let mut config = LinearConfig::new(sample_rate, downlink.baudrate, map_linear_mode(*mode));
    config.frequency_offset_hz = *frequency_offset_hz + tuning_offset_hz;
    config.differential = *differential;
    config.invert = *invert;
    config.swap_iq = *swap_iq;
    config.carrier_loop_bandwidth = if *carrier_loop_bandwidth > 0.0 {
        *carrier_loop_bandwidth
    } else {
        DEFAULT_LINEAR_CARRIER_LOOP_BANDWIDTH
    };
    config.frequency_loop_bandwidth = *frequency_loop_bandwidth;
    config.nco_max_offset_hz = *nco_max_offset_hz;
    config.matched_filter_rolloff = *matched_filter_rolloff;
    config.matched_filter_span_symbols = *matched_filter_span_symbols;
    Ok(config)
}
