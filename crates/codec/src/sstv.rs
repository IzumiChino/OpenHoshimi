//! Slow-Scan Television (SSTV) audio decoder.
//!
//! SSTV is a family of analogue picture-transmission protocols sent
//! over voice-bandwidth FM channels. The pixel intensity is encoded as
//! audio frequency on a 1500 Hz (black) to 2300 Hz (white) ramp; sync
//! pulses are at 1200 Hz; the 30-bit VIS code at the start of each
//! image identifies the mode (Robot36, Martin, Scottie, PD120, ...).
//!
//! ARISS school contacts schedule SSTV transmissions from the ISS a
//! few times a year (typically MMSSTV mode "PD120" or "Robot36") so
//! supporting these two modes is enough to decode the vast majority of
//! amateur SSTV traffic ever sent from orbit.
//!
//! # What is implemented today
//!
//! - Robot36 (mode code 0x08): full decode pipeline (VIS detect, sync
//!   pulse tracking, line scan, YCbCr -> RGB, 320x240 frame).
//! - VIS detector and dispatch infrastructure shared by all modes.
//!
//! # What is not implemented
//!
//! - PD120 line scanner (placeholder; mode is detected but a stub
//!   image is returned).
//! - Other Martin / Scottie / Wraase modes.
//! - Slant correction beyond a constant Doppler offset.
//!
//! Reference: Dayton SSTV Handbook, MMSSTV source, JL1KRA tech docs.
//! ARRL Handbook chapter on image modes for the FM-deviation curve.

use std::f32::consts::TAU;

use openhoshimi_dsp::HilbertTransform;

/// SSTV black-level audio frequency (Hz).
pub const SSTV_BLACK_HZ: f32 = 1500.0;
/// SSTV white-level audio frequency (Hz).
pub const SSTV_WHITE_HZ: f32 = 2300.0;
/// SSTV sync pulse frequency (Hz).
pub const SSTV_SYNC_HZ: f32 = 1200.0;
/// VIS code start tones (Hz). Two-tone leader: 1900 Hz then 1200 Hz
/// break followed by 1900 Hz again, then bit-by-bit data tones at
/// 1100 Hz (`1`) and 1300 Hz (`0`).
pub const SSTV_VIS_LEADER_HZ: f32 = 1900.0;
/// VIS bit `1` tone in Hz.
pub const SSTV_VIS_BIT1_HZ: f32 = 1100.0;
/// VIS bit `0` tone in Hz.
pub const SSTV_VIS_BIT0_HZ: f32 = 1300.0;

/// Recognised SSTV mode after VIS decode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SstvMode {
    /// Robot36 (mode code 0x08): 320x240 YCbCr 4:2:0 colour, 36 s.
    Robot36,
    /// PD120 (mode code 0x5F): 640x496 YUV 4:2:0 colour, 126 s. Mode
    /// is detected but image scanning is not yet implemented.
    Pd120,
}

impl SstvMode {
    /// Decode an SSTV mode from a 7-bit VIS code (data bits only,
    /// parity already verified by the caller).
    pub fn from_vis_code(code: u8) -> Option<Self> {
        match code & 0x7f {
            0x08 => Some(SstvMode::Robot36),
            0x5f => Some(SstvMode::Pd120),
            _ => None,
        }
    }

    /// Image width in pixels for this mode.
    pub fn width(self) -> u32 {
        match self {
            SstvMode::Robot36 => 320,
            SstvMode::Pd120 => 640,
        }
    }

    /// Image height in pixels for this mode.
    pub fn height(self) -> u32 {
        match self {
            SstvMode::Robot36 => 240,
            SstvMode::Pd120 => 496,
        }
    }
}

/// One decoded SSTV image.
#[derive(Debug, Clone)]
pub struct SstvImage {
    /// Source mode.
    pub mode: SstvMode,
    /// Image width in pixels.
    pub width: u32,
    /// Image height in pixels.
    pub height: u32,
    /// Image pixels as `RGB888`, row-major from top to bottom.
    pub pixels: Vec<u8>,
}

/// Streaming SSTV audio decoder.
///
/// Feed audio samples through [`SstvAnalyzer::push_samples`] and call
/// [`SstvAnalyzer::finish`] when the input ends; decoded images are
/// emitted as soon as a full frame's worth of audio has been
/// processed. The analyzer assumes one image at a time; if the audio
/// contains multiple back-to-back transmissions, drive a fresh
/// analyzer per image.
pub struct SstvAnalyzer {
    sample_rate: u32,
    hilbert: HilbertTransform,
    freq_history: Vec<f32>,
    last_phase: Option<f32>,
    state: State,
    completed: Vec<SstvImage>,
}

#[derive(Debug)]
enum State {
    AwaitingVis,
    Scanning { mode: SstvMode, scan_start: usize },
}

impl SstvAnalyzer {
    /// Build an SSTV analyzer for the given audio sample rate.
    ///
    /// # Errors
    ///
    /// Returns an error if `sample_rate` is zero.
    pub fn new(sample_rate: u32) -> Result<Self, String> {
        if sample_rate == 0 {
            return Err("sample_rate must be > 0".to_string());
        }
        Ok(Self {
            sample_rate,
            hilbert: HilbertTransform::default(),
            freq_history: Vec::new(),
            last_phase: None,
            state: State::AwaitingVis,
            completed: Vec::new(),
        })
    }

    /// Push audio samples (mono, normalised to roughly `[-1, 1]`).
    pub fn push_samples(&mut self, samples: &[f32]) {
        for &sample in samples {
            let analytic = self.hilbert.push(sample);
            let phase = analytic.q.atan2(analytic.i);
            let freq = match self.last_phase {
                Some(prev) => {
                    let mut delta = phase - prev;
                    while delta > std::f32::consts::PI {
                        delta -= TAU;
                    }
                    while delta < -std::f32::consts::PI {
                        delta += TAU;
                    }
                    delta * self.sample_rate as f32 / TAU
                }
                None => 0.0,
            };
            self.last_phase = Some(phase);
            self.freq_history.push(freq);
        }
        self.try_advance_state();
    }

    /// Return any images decoded so far and reset the output queue.
    pub fn drain_images(&mut self) -> Vec<SstvImage> {
        std::mem::take(&mut self.completed)
    }

    /// Force a final processing pass at the end of the recording.
    pub fn finish(&mut self) -> Vec<SstvImage> {
        self.try_advance_state();
        self.drain_images()
    }

    fn try_advance_state(&mut self) {
        loop {
            match &self.state {
                State::AwaitingVis => {
                    let needed = (self.sample_rate as f32 * 0.7) as usize;
                    if self.freq_history.len() < needed {
                        return;
                    }
                    if let Some((mode, end_index)) =
                        detect_vis_code(&self.freq_history, self.sample_rate)
                    {
                        self.state = State::Scanning {
                            mode,
                            scan_start: end_index,
                        };
                    } else {
                        // Keep only the most recent ~1.5 s of frequency
                        // history so that subsequent pushes scan a
                        // bounded window (VIS itself is ~0.97 s long).
                        // Without this, long captures grow `freq_history`
                        // unboundedly and `detect_vis_code` becomes O(N^2)
                        // over the whole recording.
                        let keep = (self.sample_rate as f32 * 1.5) as usize;
                        if self.freq_history.len() > keep {
                            let drop = self.freq_history.len() - keep;
                            self.freq_history.drain(..drop);
                        }
                        return;
                    }
                }
                State::Scanning { mode, scan_start } => {
                    let mode = *mode;
                    let scan_start = *scan_start;
                    let frame_samples = mode_frame_samples(mode, self.sample_rate);
                    if self.freq_history.len() < scan_start + frame_samples {
                        return;
                    }
                    let frame = &self.freq_history[scan_start..scan_start + frame_samples];
                    let image = match mode {
                        SstvMode::Robot36 => decode_robot36(frame, self.sample_rate),
                        SstvMode::Pd120 => decode_stub(mode),
                    };
                    self.completed.push(image);
                    let consumed = scan_start + frame_samples;
                    self.freq_history.drain(..consumed);
                    self.state = State::AwaitingVis;
                }
            }
        }
    }
}

fn mode_frame_samples(mode: SstvMode, sample_rate: u32) -> usize {
    let total_seconds = match mode {
        SstvMode::Robot36 => 36.0,
        SstvMode::Pd120 => 126.0,
    };
    (sample_rate as f32 * total_seconds) as usize
}

fn detect_vis_code(freq: &[f32], sample_rate: u32) -> Option<(SstvMode, usize)> {
    // VIS structure: 300 ms of 1900 Hz leader, 10 ms break at 1200 Hz,
    // 300 ms more of 1900 Hz, 30 ms start bit at 1200 Hz, eight 30 ms
    // bits (data + parity), 30 ms stop bit at 1200 Hz. Total ~970 ms.
    let ms = |x: f32| (sample_rate as f32 * x / 1000.0) as usize;
    let leader_a = ms(300.0);
    let break_pulse = ms(10.0);
    let leader_b = ms(300.0);
    let start_bit = ms(30.0);
    let bit_len = ms(30.0);
    let total_pre_data = leader_a + break_pulse + leader_b + start_bit;
    let total_data = bit_len * 8;
    let total = total_pre_data + total_data + bit_len; // + stop bit
    let break_search_window = ms(20.0);

    if freq.len() < total {
        return None;
    }

    // Find the first sample where the running mean of the next
    // `leader_a` samples sits within ±60 Hz of 1900 Hz; that anchor
    // is the start of leader-A.
    let step = ms(5.0).max(1);
    let mut leader_start = None;
    for start in (0..freq.len() - total).step_by(step) {
        let avg = mean(&freq[start..start + leader_a]);
        if (avg - SSTV_VIS_LEADER_HZ).abs() <= 60.0 {
            // Walk back to find the very first sample of the leader
            // by checking if the previous block also matched, refine
            // by 1 ms ticks.
            let fine_step = ms(1.0).max(1);
            let mut anchor = start;
            while anchor >= fine_step {
                let candidate = anchor - fine_step;
                let candidate_avg = mean(&freq[candidate..candidate + leader_a]);
                if (candidate_avg - SSTV_VIS_LEADER_HZ).abs() <= 60.0 {
                    anchor = candidate;
                } else {
                    break;
                }
            }
            leader_start = Some(anchor);
            break;
        }
    }
    let leader_start = leader_start?;

    // Search a small window around the expected break position for
    // the 1200 Hz break pulse. This makes the detector tolerant to
    // small clock-drift and mis-tuned audio paths.
    let expected_break = leader_start + leader_a;
    let break_lo = expected_break.saturating_sub(break_search_window);
    let break_hi = (expected_break + break_search_window).min(freq.len() - break_pulse);
    let mut break_off = None;
    for candidate in (break_lo..=break_hi).step_by(ms(1.0).max(1)) {
        let avg = mean(&freq[candidate..candidate + break_pulse]);
        if (avg - SSTV_SYNC_HZ).abs() <= 80.0 {
            break_off = Some(candidate);
            break;
        }
    }
    let break_off = break_off?;

    // Re-anchor everything off the break pulse rather than the
    // possibly-fuzzy leader start.
    let leader_b_off = break_off + break_pulse;
    if leader_b_off + leader_b + start_bit + total_data + bit_len > freq.len() {
        return None;
    }
    let leader_b_avg = mean(&freq[leader_b_off..leader_b_off + leader_b]);
    if (leader_b_avg - SSTV_VIS_LEADER_HZ).abs() > 60.0 {
        return None;
    }
    let start_bit_off = leader_b_off + leader_b;
    let start_bit_avg = mean(&freq[start_bit_off..start_bit_off + start_bit]);
    if (start_bit_avg - SSTV_SYNC_HZ).abs() > 60.0 {
        return None;
    }

    let data_off = start_bit_off + start_bit;
    let mut byte: u8 = 0;
    let mut parity_count = 0u8;
    for i in 0..8 {
        let off = data_off + i * bit_len;
        let avg = mean(&freq[off..off + bit_len]);
        let bit = if (avg - SSTV_VIS_BIT1_HZ).abs() < (avg - SSTV_VIS_BIT0_HZ).abs() {
            1u8
        } else {
            0u8
        };
        if i < 7 {
            byte |= bit << i; // LSB first per VIS spec
        }
        parity_count = parity_count.wrapping_add(bit);
    }
    if parity_count & 1 != 0 {
        return None;
    }
    let mode = SstvMode::from_vis_code(byte)?;
    let end_index = data_off + 8 * bit_len + bit_len; // include stop bit
    Some((mode, end_index))
}

fn mean(slice: &[f32]) -> f32 {
    if slice.is_empty() {
        return 0.0;
    }
    slice.iter().sum::<f32>() / slice.len() as f32
}

fn freq_to_intensity(freq: f32) -> u8 {
    let clamped = freq.clamp(SSTV_BLACK_HZ, SSTV_WHITE_HZ);
    let normalised = (clamped - SSTV_BLACK_HZ) / (SSTV_WHITE_HZ - SSTV_BLACK_HZ);
    (normalised * 255.0).round().clamp(0.0, 255.0) as u8
}

fn decode_robot36(frame: &[f32], sample_rate: u32) -> SstvImage {
    // Robot36 line layout: 9 ms sync (1200 Hz), 3 ms porch, 88 ms Y
    // scan, 4.5 ms separator (alternating 1500/2300 Hz to identify
    // the colour channel for this row), 1.5 ms porch, 44 ms colour
    // scan. The colour channel alternates Cb (even rows) / Cr (odd
    // rows). Total per line: 150 ms; 240 lines => 36 s.
    let ms_to_samples = |x: f32| (sample_rate as f32 * x / 1000.0) as usize;
    let line_len = ms_to_samples(150.0);
    let sync_len = ms_to_samples(9.0);
    let porch_a = ms_to_samples(3.0);
    let y_len = ms_to_samples(88.0);
    let separator = ms_to_samples(4.5);
    let porch_b = ms_to_samples(1.5);
    let chroma_len = ms_to_samples(44.0);

    let width = SstvMode::Robot36.width() as usize;
    let height = SstvMode::Robot36.height() as usize;
    let mut pixels = vec![0u8; width * height * 3];

    // Robot36 uses YCbCr 4:2:0: each pair of consecutive rows shares
    // one (Cb, Cr) pair. The transmitter sends Cr on one row and Cb
    // on the next (or vice versa); the separator tone tells us which
    // is which. We buffer one row at a time and only paint a row
    // when its mate arrives, so both rows in the pair use the same
    // (Cb, Cr) and the chroma sub-sampling is applied symmetrically.
    // Without this, every other row would fall through the
    // pairing-miss path and render with the wrong colour.
    struct PendingRow {
        row: usize,
        y: Vec<u8>,
        chroma: Vec<u8>,
        is_cr: bool,
    }

    let decode_row = |row: usize| -> Option<PendingRow> {
        let line_start = row * line_len;
        if line_start + line_len > frame.len() {
            return None;
        }
        let line = &frame[line_start..line_start + line_len];
        let y_start = sync_len + porch_a;
        let y_block = &line[y_start..y_start + y_len];
        let chroma_start = y_start + y_len + separator + porch_b;
        let chroma_block = &line[chroma_start..chroma_start + chroma_len];
        // Detect channel from the separator tone: 2300 Hz => Cr,
        // 1500 Hz => Cb.
        let separator_block = &line[y_start + y_len..y_start + y_len + separator];
        let separator_mean = mean(separator_block);
        let is_cr = (separator_mean - SSTV_WHITE_HZ).abs() < (separator_mean - SSTV_BLACK_HZ).abs();
        let y_samples = resample_to(y_block, width);
        let chroma_samples = resample_to(chroma_block, width);
        Some(PendingRow {
            row,
            y: y_samples.iter().map(|f| freq_to_intensity(*f)).collect(),
            chroma: chroma_samples
                .iter()
                .map(|f| freq_to_intensity(*f))
                .collect(),
            is_cr,
        })
    };

    let paint_row = |pixels: &mut [u8], row: usize, y: &[u8], cb: &[u8], cr: &[u8]| {
        for col in 0..width {
            let yv = y[col] as f32;
            let cbv = cb[col] as f32 - 128.0;
            let crv = cr[col] as f32 - 128.0;
            let r = (yv + 1.402 * crv).clamp(0.0, 255.0) as u8;
            let g = (yv - 0.344136 * cbv - 0.714136 * crv).clamp(0.0, 255.0) as u8;
            let b = (yv + 1.772 * cbv).clamp(0.0, 255.0) as u8;
            let p = (row * width + col) * 3;
            pixels[p] = r;
            pixels[p + 1] = g;
            pixels[p + 2] = b;
        }
    };

    let mut pending: Option<PendingRow> = None;
    for row in 0..height {
        let Some(current) = decode_row(row) else {
            break;
        };
        match pending.take() {
            Some(prev) if prev.is_cr != current.is_cr => {
                let (cr_row, cb_row) = if prev.is_cr {
                    (&prev.chroma, &current.chroma)
                } else {
                    (&current.chroma, &prev.chroma)
                };
                paint_row(&mut pixels, prev.row, &prev.y, cb_row, cr_row);
                paint_row(&mut pixels, current.row, &current.y, cb_row, cr_row);
            }
            Some(prev) => {
                // Mate had the same channel as the held row (line drop
                // or sync slip). Render the held row standalone and
                // keep the new one as the next pair candidate.
                paint_row(&mut pixels, prev.row, &prev.y, &prev.chroma, &prev.chroma);
                pending = Some(current);
            }
            None => {
                pending = Some(current);
            }
        }
    }
    if let Some(prev) = pending {
        paint_row(&mut pixels, prev.row, &prev.y, &prev.chroma, &prev.chroma);
    }

    SstvImage {
        mode: SstvMode::Robot36,
        width: width as u32,
        height: height as u32,
        pixels,
    }
}

fn decode_stub(mode: SstvMode) -> SstvImage {
    let width = mode.width() as usize;
    let height = mode.height() as usize;
    SstvImage {
        mode,
        width: width as u32,
        height: height as u32,
        pixels: vec![0u8; width * height * 3],
    }
}

fn resample_to(input: &[f32], target_len: usize) -> Vec<f32> {
    if target_len == 0 {
        return Vec::new();
    }
    if input.is_empty() {
        return vec![0.0; target_len];
    }
    let mut out = Vec::with_capacity(target_len);
    let ratio = (input.len() - 1) as f32 / target_len.max(1) as f32;
    for i in 0..target_len {
        let src = i as f32 * ratio;
        let lo = src.floor() as usize;
        let hi = (lo + 1).min(input.len() - 1);
        let t = src - lo as f32;
        out.push(input[lo] * (1.0 - t) + input[hi] * t);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vis_code_decodes_robot36() {
        assert_eq!(SstvMode::from_vis_code(0x08), Some(SstvMode::Robot36));
        assert_eq!(SstvMode::from_vis_code(0x5f), Some(SstvMode::Pd120));
        assert_eq!(SstvMode::from_vis_code(0x42), None);
    }

    fn synthesize_constant_freq(freq: f32, samples: usize, sample_rate: u32) -> Vec<f32> {
        let mut out = Vec::with_capacity(samples);
        let mut phase = 0.0f32;
        let inc = TAU * freq / sample_rate as f32;
        for _ in 0..samples {
            out.push(phase.sin());
            phase += inc;
            if phase >= TAU {
                phase -= TAU;
            }
        }
        out
    }

    #[test]
    fn freq_to_intensity_maps_endpoints() {
        assert_eq!(freq_to_intensity(SSTV_BLACK_HZ), 0);
        assert_eq!(freq_to_intensity(SSTV_WHITE_HZ), 255);
        let mid = freq_to_intensity((SSTV_BLACK_HZ + SSTV_WHITE_HZ) / 2.0);
        assert!((125..=130).contains(&mid));
    }

    #[test]
    fn analyzer_detects_robot36_vis_code() {
        let sr = 16_000;
        let ms = |x: f32| (sr as f32 * x / 1000.0) as usize;
        let mut audio = Vec::new();
        // 100 ms silence
        audio.extend(synthesize_constant_freq(0.0, ms(100.0), sr));
        // Leader A: 300 ms 1900 Hz
        audio.extend(synthesize_constant_freq(SSTV_VIS_LEADER_HZ, ms(300.0), sr));
        // Break: 10 ms 1200 Hz
        audio.extend(synthesize_constant_freq(SSTV_SYNC_HZ, ms(10.0), sr));
        // Leader B: 300 ms 1900 Hz
        audio.extend(synthesize_constant_freq(SSTV_VIS_LEADER_HZ, ms(300.0), sr));
        // Start bit: 30 ms 1200 Hz
        audio.extend(synthesize_constant_freq(SSTV_SYNC_HZ, ms(30.0), sr));
        // 7 data bits = Robot36 (0x08, LSB first: 0,0,0,1,0,0,0)
        // + parity bit (even parity over 7 bits with one 1 -> parity = 1)
        let bits = [0u8, 0, 0, 1, 0, 0, 0, 1];
        for bit in bits {
            let f = if bit == 1 {
                SSTV_VIS_BIT1_HZ
            } else {
                SSTV_VIS_BIT0_HZ
            };
            audio.extend(synthesize_constant_freq(f, ms(30.0), sr));
        }
        // Stop bit: 30 ms 1200 Hz
        audio.extend(synthesize_constant_freq(SSTV_SYNC_HZ, ms(30.0), sr));
        // 50 ms tail to ensure detector has > 1 s of data after 100ms
        // lead-in silence.
        audio.extend(synthesize_constant_freq(SSTV_SYNC_HZ, ms(50.0), sr));

        let mut analyzer = SstvAnalyzer::new(sr).expect("analyzer");
        analyzer.push_samples(&audio);
        // After detecting VIS the analyzer transitions to scanning,
        // but we never give it a full 36 s frame so no image emerges.
        assert!(analyzer.drain_images().is_empty());
        assert!(matches!(
            analyzer.state,
            State::Scanning {
                mode: SstvMode::Robot36,
                ..
            }
        ));
    }

    /// Synthesize two consecutive Robot36 lines with paired chroma:
    /// row 0 carries Cr (separator tone 2300 Hz, chroma sub-carrier
    /// near "white"), row 1 carries Cb (separator 1500 Hz, chroma
    /// sub-carrier near "black"), luma at mid-grey. With the pairing
    /// logic in place, both rows must share the same (Cb, Cr) pair and
    /// render with a clearly reddish hue (R well above B). Without
    /// pairing, row 0 would treat its Cr as Cb and the colour would
    /// flip.
    ///
    /// The test feeds the *frequency-domain* representation directly
    /// to `decode_robot36`, bypassing the Hilbert / FM-demod stage that
    /// `SstvAnalyzer::push_samples` performs upstream — `decode_robot36`
    /// is documented as taking a frequency-history slice.
    #[test]
    fn robot36_paired_rows_share_chroma() {
        let sample_rate = 16_000u32;
        let ms = |x: f32| (sample_rate as f32 * x / 1000.0) as usize;
        let line_len = ms(150.0);
        let height = SstvMode::Robot36.height() as usize;
        let total_samples = line_len * height;

        let intensity_to_freq = |v: u8| -> f32 {
            let normalised = v as f32 / 255.0;
            SSTV_BLACK_HZ + normalised * (SSTV_WHITE_HZ - SSTV_BLACK_HZ)
        };
        let y_freq = intensity_to_freq(180);
        let cr_freq = intensity_to_freq(220);
        let cb_freq = intensity_to_freq(40);

        let mut frame = vec![0.0f32; total_samples];
        let fill = |frame: &mut [f32], offset: usize, count: usize, freq: f32| {
            for slot in frame.iter_mut().skip(offset).take(count) {
                *slot = freq;
            }
        };

        for row in 0..height {
            let base = row * line_len;
            let mut cur = 0usize;
            fill(&mut frame, base + cur, ms(9.0), SSTV_SYNC_HZ);
            cur += ms(9.0);
            fill(&mut frame, base + cur, ms(3.0), SSTV_BLACK_HZ);
            cur += ms(3.0);
            fill(&mut frame, base + cur, ms(88.0), y_freq);
            cur += ms(88.0);
            let separator_freq = if row % 2 == 0 { SSTV_WHITE_HZ } else { 1500.0 };
            fill(&mut frame, base + cur, ms(4.5), separator_freq);
            cur += ms(4.5);
            fill(&mut frame, base + cur, ms(1.5), SSTV_BLACK_HZ);
            cur += ms(1.5);
            let chroma_freq = if row % 2 == 0 { cr_freq } else { cb_freq };
            let chroma_samples = line_len - cur;
            fill(&mut frame, base + cur, chroma_samples, chroma_freq);
        }

        let image = decode_robot36(&frame, sample_rate);
        assert_eq!(image.width, 320);
        assert_eq!(image.height, 240);

        // Sample several pixels mid-row (avoid edge resampling artefacts)
        // on both the Cr-carrying row and its Cb mate.
        let width = image.width as usize;
        let row0_mid = (width / 2) * 3;
        let row1_mid = (width + width / 2) * 3;
        let r0 = image.pixels[row0_mid] as i32;
        let g0 = image.pixels[row0_mid + 1] as i32;
        let b0 = image.pixels[row0_mid + 2] as i32;
        let r1 = image.pixels[row1_mid] as i32;
        let g1 = image.pixels[row1_mid + 1] as i32;
        let b1 = image.pixels[row1_mid + 2] as i32;
        // High Cr + low Cb at Y~180 should produce a clearly red-shifted
        // pixel (R well above B, B near zero, G in the middle).
        assert!(r0 - b0 > 100, "row 0 not red-shifted: rgb=({r0},{g0},{b0})");
        assert!(r1 - b1 > 100, "row 1 not red-shifted: rgb=({r1},{g1},{b1})");
        // Both rows in the pair must come out within a few intensity
        // steps of each other since they share (Cb, Cr) and constant Y.
        assert!(
            (r0 - r1).abs() < 12 && (g0 - g1).abs() < 12 && (b0 - b1).abs() < 12,
            "paired rows differ too much: ({r0},{g0},{b0}) vs ({r1},{g1},{b1})"
        );
    }
}
