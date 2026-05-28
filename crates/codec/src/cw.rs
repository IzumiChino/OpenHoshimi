//! CW (Morse code) audio decoder.
//!
//! CW does not fit the OpenHoshimi BitPipeline very cleanly — there is
//! no fixed baudrate, no syncword, no frame size — so this module
//! exposes a self-contained [`CwAnalyzer`] that consumes audio samples
//! and emits decoded ASCII text. It is suitable for narrowband CW
//! beacons such as those carried by amateur cubesats (CAS series, AO-91
//! backup beacon, many student satellites). It is not a high-performance
//! contest CW reader; the goal is "decode the satellite's published
//! callsign and a few telemetry mnemonics" rather than 40 WPM
//! keyboard-clean copy.
//!
//! Pipeline:
//!
//! 1. Goertzel detector at the configured tone frequency, integrated
//!    over `envelope_period_samples` audio samples.
//! 2. Adaptive hysteresis threshold splits the envelope into key-on /
//!    key-off, recorded as run-length events.
//! 3. On [`CwAnalyzer::finish`] we cluster mark lengths into two groups
//!    (dit, dah) using a one-dimensional k-means seeded by min/max,
//!    then classify gaps against the resulting dit estimate.
//! 4. Morse table lookup converts each accumulated element string into
//!    one ASCII character; long gaps become spaces.
//!
//! Two-pass classification (online envelope detection, batched mark
//! clustering at flush time) trades incremental output for far more
//! robust speed adaptation than an online dit-tracking scheme.

use std::collections::HashMap;
use std::f32::consts::TAU;
use std::sync::OnceLock;

/// Default tone frequency operators choose for CW beacons (Hz).
pub const DEFAULT_CW_TONE_HZ: f32 = 700.0;
/// Default envelope sampling rate (Hz). 100 Hz = 10 ms per envelope
/// sample, fine enough to separate dits at 50 WPM (24 ms) without
/// drowning the Goertzel in noise.
pub const DEFAULT_ENVELOPE_RATE_HZ: u32 = 100;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum EventKind {
    Mark,
    Gap,
}

#[derive(Debug, Clone, Copy)]
struct Event {
    kind: EventKind,
    envelopes: usize,
}

/// Configuration for [`CwAnalyzer`].
#[derive(Debug, Clone, Copy)]
pub struct CwAnalyzerConfig {
    /// Audio sample rate in Hz.
    pub sample_rate: u32,
    /// CW tone frequency in Hz (typical range 400-1000 Hz).
    pub tone_hz: f32,
    /// Envelope sampling rate in Hz; one envelope sample per
    /// `sample_rate / envelope_rate_hz` audio samples.
    pub envelope_rate_hz: u32,
}

impl Default for CwAnalyzerConfig {
    fn default() -> Self {
        Self {
            sample_rate: 48_000,
            tone_hz: DEFAULT_CW_TONE_HZ,
            envelope_rate_hz: DEFAULT_ENVELOPE_RATE_HZ,
        }
    }
}

impl CwAnalyzerConfig {
    /// Build a config using the given audio sample rate; everything
    /// else defaults.
    pub fn for_sample_rate(sample_rate: u32) -> Self {
        Self {
            sample_rate,
            ..Self::default()
        }
    }
}

/// CW audio decoder.
pub struct CwAnalyzer {
    config: CwAnalyzerConfig,
    envelope_period: usize,
    coefficient: f32,
    q1: f32,
    q2: f32,
    integrated_samples: usize,
    threshold_high: f32,
    threshold_low: f32,
    current_key_state: bool,
    current_run_envelopes: usize,
    events: Vec<Event>,
    finished_dit_envelopes: f32,
}

impl CwAnalyzer {
    /// Create a CW analyzer with the supplied configuration.
    ///
    /// # Errors
    ///
    /// Returns `Err` if the configuration is invalid (zero sample
    /// rate, zero envelope rate, or non-positive tone frequency).
    pub fn new(config: CwAnalyzerConfig) -> Result<Self, String> {
        if config.sample_rate == 0 {
            return Err("sample_rate must be > 0".to_string());
        }
        if config.envelope_rate_hz == 0 {
            return Err("envelope_rate_hz must be > 0".to_string());
        }
        if !config.tone_hz.is_finite() || config.tone_hz <= 0.0 {
            return Err("tone_hz must be finite and > 0".to_string());
        }
        let envelope_period = (config.sample_rate / config.envelope_rate_hz).max(1) as usize;
        let omega = TAU * config.tone_hz / config.sample_rate as f32;
        Ok(Self {
            config,
            envelope_period,
            coefficient: 2.0 * omega.cos(),
            q1: 0.0,
            q2: 0.0,
            integrated_samples: 0,
            threshold_high: 0.0,
            threshold_low: 0.0,
            current_key_state: false,
            current_run_envelopes: 0,
            events: Vec::new(),
            finished_dit_envelopes: 0.0,
        })
    }

    /// Feed audio samples into the analyzer. Decoded text is only
    /// available after [`Self::finish`].
    pub fn push_samples(&mut self, samples: &[f32]) {
        for &sample in samples {
            let q0 = sample + self.coefficient * self.q1 - self.q2;
            self.q2 = self.q1;
            self.q1 = q0;
            self.integrated_samples += 1;
            if self.integrated_samples >= self.envelope_period {
                let energy =
                    self.q1 * self.q1 + self.q2 * self.q2 - self.coefficient * self.q1 * self.q2;
                let envelope = energy.max(0.0).sqrt();
                self.q1 = 0.0;
                self.q2 = 0.0;
                self.integrated_samples = 0;
                self.process_envelope(envelope);
            }
        }
    }

    /// Flush in-flight state and return all decoded text. The
    /// analyzer can be reused after this call; the next batch starts
    /// fresh.
    pub fn finish(&mut self) -> String {
        if self.current_run_envelopes > 0 {
            self.events.push(Event {
                kind: if self.current_key_state {
                    EventKind::Mark
                } else {
                    EventKind::Gap
                },
                envelopes: self.current_run_envelopes,
            });
        }
        self.current_run_envelopes = 0;
        self.current_key_state = false;
        let dit_envelopes = estimate_dit_envelopes(&self.events);
        self.finished_dit_envelopes = dit_envelopes;
        let text = decode_events(&self.events, dit_envelopes);
        self.events.clear();
        self.threshold_high = 0.0;
        self.threshold_low = 0.0;
        text
    }

    /// Estimated dit duration in envelope samples after the last
    /// [`Self::finish`]. Useful for diagnostics and assertions.
    pub fn dit_envelopes(&self) -> f32 {
        self.finished_dit_envelopes
    }

    /// Estimated last-pass sending speed in words per minute (PARIS
    /// rule).
    pub fn estimated_wpm(&self) -> f32 {
        let dit_ms = self.finished_dit_envelopes * 1000.0 / self.config.envelope_rate_hz as f32;
        if dit_ms <= 0.0 {
            0.0
        } else {
            1200.0 / dit_ms
        }
    }

    fn process_envelope(&mut self, envelope: f32) {
        self.update_threshold(envelope);
        let key_state = self.classify_key_state(envelope);
        if key_state == self.current_key_state {
            self.current_run_envelopes += 1;
            return;
        }
        if self.current_run_envelopes > 0 {
            self.events.push(Event {
                kind: if self.current_key_state {
                    EventKind::Mark
                } else {
                    EventKind::Gap
                },
                envelopes: self.current_run_envelopes,
            });
        }
        self.current_key_state = key_state;
        self.current_run_envelopes = 1;
    }

    fn update_threshold(&mut self, envelope: f32) {
        // Asymmetric exponential trackers: the key-on level rises
        // fast and decays slowly, the noise floor falls fast and
        // rises slowly. The resulting hysteretic threshold survives
        // mild fading without flapping on a quiet signal.
        const MAX_RISE: f32 = 0.4;
        const MAX_DECAY: f32 = 0.001;
        const MIN_RISE: f32 = 0.001;
        const MIN_DECAY: f32 = 0.4;
        if envelope > self.threshold_high {
            self.threshold_high += MAX_RISE * (envelope - self.threshold_high);
        } else {
            self.threshold_high += MAX_DECAY * (envelope - self.threshold_high);
        }
        if envelope < self.threshold_low {
            self.threshold_low += MIN_DECAY * (envelope - self.threshold_low);
        } else {
            self.threshold_low += MIN_RISE * (envelope - self.threshold_low);
        }
    }

    fn classify_key_state(&self, envelope: f32) -> bool {
        let span = self.threshold_high - self.threshold_low;
        if span <= 0.0 {
            return false;
        }
        let high = self.threshold_low + 0.6 * span;
        let low = self.threshold_low + 0.4 * span;
        if self.current_key_state {
            envelope >= low
        } else {
            envelope >= high
        }
    }
}

fn estimate_dit_envelopes(events: &[Event]) -> f32 {
    let marks: Vec<f32> = events
        .iter()
        .filter(|e| e.kind == EventKind::Mark)
        .map(|e| e.envelopes as f32)
        .collect();
    if marks.is_empty() {
        return 0.0;
    }
    if marks.len() == 1 {
        // One mark says nothing about dit/dah split. Treat it as a
        // dit; the morse table lookup will degrade gracefully.
        return marks[0];
    }
    let min = marks.iter().copied().fold(f32::INFINITY, f32::min);
    let max = marks.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    if max <= min * 1.5 {
        // All marks roughly equal: either all dits (e.g. just "S")
        // or all dahs. Bias toward dits — they are the single most
        // common element in normal traffic.
        return min;
    }
    // 1-D Lloyd's algorithm with two centroids initialised at
    // min and max. Converges in a handful of iterations for any
    // realistic mark-length distribution.
    let mut c0 = min;
    let mut c1 = max;
    for _ in 0..16 {
        let mut sum0 = 0.0;
        let mut sum1 = 0.0;
        let mut n0 = 0usize;
        let mut n1 = 0usize;
        for &m in &marks {
            if (m - c0).abs() <= (m - c1).abs() {
                sum0 += m;
                n0 += 1;
            } else {
                sum1 += m;
                n1 += 1;
            }
        }
        let new_c0 = if n0 > 0 { sum0 / n0 as f32 } else { c0 };
        let new_c1 = if n1 > 0 { sum1 / n1 as f32 } else { c1 };
        if (new_c0 - c0).abs() < 1e-3 && (new_c1 - c1).abs() < 1e-3 {
            c0 = new_c0;
            c1 = new_c1;
            break;
        }
        c0 = new_c0;
        c1 = new_c1;
    }
    c0.min(c1)
}

fn decode_events(events: &[Event], dit_envelopes: f32) -> String {
    if events.is_empty() || dit_envelopes <= 0.0 {
        return String::new();
    }
    let table = morse_table();
    let mut out = String::new();
    let mut element = String::new();
    for event in events {
        match event.kind {
            EventKind::Mark => {
                let symbol = if (event.envelopes as f32) < dit_envelopes * 2.0 {
                    '.'
                } else {
                    '-'
                };
                element.push(symbol);
            }
            EventKind::Gap => {
                let run = event.envelopes as f32;
                if run < dit_envelopes * 2.0 {
                    // Intra-letter gap, accumulate.
                } else if run < dit_envelopes * 5.0 {
                    flush_letter(&mut element, &mut out, table);
                } else {
                    flush_letter(&mut element, &mut out, table);
                    if !out.is_empty() && !out.ends_with(' ') {
                        out.push(' ');
                    }
                }
            }
        }
    }
    flush_letter(&mut element, &mut out, table);
    out
}

fn flush_letter(element: &mut String, out: &mut String, table: &HashMap<&str, char>) {
    if element.is_empty() {
        return;
    }
    if let Some(ch) = table.get(element.as_str()) {
        out.push(*ch);
    } else {
        out.push('?');
    }
    element.clear();
}

fn morse_table() -> &'static HashMap<&'static str, char> {
    static TABLE: OnceLock<HashMap<&'static str, char>> = OnceLock::new();
    TABLE.get_or_init(|| {
        let pairs: &[(&str, char)] = &[
            (".-", 'A'),
            ("-...", 'B'),
            ("-.-.", 'C'),
            ("-..", 'D'),
            (".", 'E'),
            ("..-.", 'F'),
            ("--.", 'G'),
            ("....", 'H'),
            ("..", 'I'),
            (".---", 'J'),
            ("-.-", 'K'),
            (".-..", 'L'),
            ("--", 'M'),
            ("-.", 'N'),
            ("---", 'O'),
            (".--.", 'P'),
            ("--.-", 'Q'),
            (".-.", 'R'),
            ("...", 'S'),
            ("-", 'T'),
            ("..-", 'U'),
            ("...-", 'V'),
            (".--", 'W'),
            ("-..-", 'X'),
            ("-.--", 'Y'),
            ("--..", 'Z'),
            ("-----", '0'),
            (".----", '1'),
            ("..---", '2'),
            ("...--", '3'),
            ("....-", '4'),
            (".....", '5'),
            ("-....", '6'),
            ("--...", '7'),
            ("---..", '8'),
            ("----.", '9'),
            (".-.-.-", '.'),
            ("--..--", ','),
            ("..--..", '?'),
            (".----.", '\''),
            ("-.-.--", '!'),
            ("-..-.", '/'),
            ("-.--.", '('),
            ("-.--.-", ')'),
            (".-...", '&'),
            ("---...", ':'),
            ("-.-.-.", ';'),
            ("-...-", '='),
            (".-.-.", '+'),
            ("-....-", '-'),
            ("..--.-", '_'),
            (".-..-.", '"'),
            ("...-..-", '$'),
            (".--.-.", '@'),
        ];
        pairs.iter().copied().collect()
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn synthesize_morse(text: &str, wpm: f32, sample_rate: u32, tone_hz: f32) -> Vec<f32> {
        let dit_ms = 1200.0 / wpm;
        let dit_samples = (dit_ms * sample_rate as f32 / 1000.0) as usize;
        let mut samples = Vec::new();
        let mut phase = 0.0f32;
        let inc = TAU * tone_hz / sample_rate as f32;
        // Lead-in silence so the threshold tracker can settle.
        samples.resize(sample_rate as usize / 4, 0.0);
        let table_inv: HashMap<char, &'static str> = morse_table()
            .iter()
            .map(|(code, ch)| (*ch, *code))
            .collect();
        for (word_idx, word) in text.split(' ').enumerate() {
            if word_idx > 0 {
                // Word gap: 7 dits of silence total.
                samples.resize(samples.len() + 7 * dit_samples, 0.0);
            }
            for (letter_idx, ch) in word.chars().enumerate() {
                if letter_idx > 0 {
                    // Letter gap: 3 dits of silence total.
                    samples.resize(samples.len() + 3 * dit_samples, 0.0);
                }
                let code = match table_inv.get(&ch) {
                    Some(c) => *c,
                    None => continue,
                };
                for (elem_idx, elem) in code.chars().enumerate() {
                    if elem_idx > 0 {
                        samples.resize(samples.len() + dit_samples, 0.0);
                    }
                    let mark_dits = if elem == '.' { 1 } else { 3 };
                    for _ in 0..(mark_dits * dit_samples) {
                        samples.push(phase.sin());
                        phase += inc;
                        if phase >= TAU {
                            phase -= TAU;
                        }
                    }
                }
            }
        }
        // Lead-out silence so the analyzer's finish() sees a definite
        // gap after the last mark.
        samples.resize(samples.len() + sample_rate as usize / 2, 0.0);
        samples
    }

    #[test]
    fn morse_table_inverts_uniquely() {
        let mut seen = std::collections::HashSet::new();
        for ch in morse_table().values() {
            assert!(seen.insert(*ch), "duplicate character in morse table: {ch}");
        }
    }

    #[test]
    fn decodes_synthetic_callsign_at_15_wpm() {
        let text = "CQ DE OH";
        let samples = synthesize_morse(text, 15.0, 48_000, 700.0);
        let mut analyzer =
            CwAnalyzer::new(CwAnalyzerConfig::for_sample_rate(48_000)).expect("analyzer config");
        analyzer.push_samples(&samples);
        let decoded = analyzer.finish();
        assert_eq!(decoded.trim(), text, "decoded={decoded:?}");
        let wpm = analyzer.estimated_wpm();
        assert!((10.0..=22.0).contains(&wpm), "WPM estimate way off: {wpm}");
    }

    #[test]
    fn decodes_at_25_wpm_chunked() {
        let text = "BJ1SK TEST";
        let samples = synthesize_morse(text, 25.0, 48_000, 750.0);
        let mut analyzer = CwAnalyzer::new(CwAnalyzerConfig {
            sample_rate: 48_000,
            tone_hz: 750.0,
            envelope_rate_hz: DEFAULT_ENVELOPE_RATE_HZ,
        })
        .expect("analyzer config");
        // Feed in two halves to verify state survives chunking.
        let mid = samples.len() / 2;
        analyzer.push_samples(&samples[..mid]);
        analyzer.push_samples(&samples[mid..]);
        let decoded = analyzer.finish();
        assert_eq!(decoded.trim(), text, "decoded={decoded:?}");
    }

    #[test]
    fn decodes_short_letter() {
        // Just "E" — single dit, a worst-case input for adaptive
        // dit/dah splitting. Should still come back as 'E'.
        let samples = synthesize_morse("E", 15.0, 48_000, 700.0);
        let mut analyzer =
            CwAnalyzer::new(CwAnalyzerConfig::for_sample_rate(48_000)).expect("analyzer config");
        analyzer.push_samples(&samples);
        let decoded = analyzer.finish();
        assert_eq!(decoded.trim(), "E");
    }

    #[test]
    fn rejects_invalid_config() {
        let bad = CwAnalyzer::new(CwAnalyzerConfig {
            sample_rate: 0,
            tone_hz: 700.0,
            envelope_rate_hz: 100,
        });
        assert!(bad.is_err());
        let bad_tone = CwAnalyzer::new(CwAnalyzerConfig {
            sample_rate: 48_000,
            tone_hz: -10.0,
            envelope_rate_hz: 100,
        });
        assert!(bad_tone.is_err());
    }
}
