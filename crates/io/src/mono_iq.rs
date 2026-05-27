//! Mono-real-audio adapters and audio-mode dispatch for IQ-family downlinks.
//!
//! IQ-modulated downlinks expect stereo I/Q WAV files (channel 0 = I,
//! channel 1 = Q). In practice users feed the decoder recordings exported
//! by SDR receivers as duplicated-mono FM-discriminator audio, or as
//! mono SSB demodulator output. This module provides the helpers both
//! `decode_file` and the GUI use to:
//!
//! * detect what shape the WAV actually has via [`AudioMode`] +
//!   [`detect_audio_mode_auto`],
//! * read a real-valued mono audio file as a complex `IqSource` whose
//!   imaginary part is zero ([`MonoIqSource`]), so an SSB recording can
//!   feed straight into the linear-IQ demodulator with the in-audio
//!   carrier supplied as `tuning_offset_hz`,
//! * sniff a stereo WAV for the duplicate-stereo pattern that SDR
//!   receivers use when they dump FM audio into both channels
//!   ([`is_duplicate_stereo`]),
//! * and a small [`read_iq_prefix`] helper because every caller wants
//!   the same "drain `len` samples or hit EOF" behaviour.

use std::path::Path;

use openhoshimi_core::{InputSource, IoError, IqSample, IqSource};

use crate::{OggSource, WavIqSource, WavSource};

const READ_CHUNK: usize = 4096;

/// How an input WAV should be interpreted when the active downlink wants
/// IQ but the file is not necessarily stereo I/Q.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum AudioMode {
    /// Sniff the file: stereo with distinct L/R is treated as IQ; mono
    /// or duplicate-stereo as FM-discriminator audio. Filenames carrying
    /// `USB`/`LSB` resolve to [`AudioMode::Ssb`] so the caller can
    /// auto-estimate the in-audio carrier instead of forcing the user
    /// to type one in.
    #[default]
    Auto,
    /// Force stereo I/Q interpretation (errors if the file is mono).
    Iq,
    /// Force FM-discriminator mono audio (instantaneous-frequency
    /// waveform straight out of an SDR receiver's FM demodulator).
    Fm,
    /// Force SSB mono audio. The caller must also provide an
    /// in-audio carrier frequency in Hz.
    Ssb,
}

/// Decide how an IQ-family downlink should consume `path` when the user
/// wants automatic detection.
///
/// Returns one of:
/// * [`AudioMode::Iq`] — stereo with distinguishable L/R, treated as
///   real complex IQ.
/// * [`AudioMode::Fm`] — mono or duplicate-stereo WAVs, the most common
///   shape produced by SDR receivers exporting FM-discriminator audio.
/// * [`AudioMode::Ssb`] — filename contains `USB` or `LSB`, in which
///   case the caller is expected to estimate the in-audio carrier (see
///   [`crate::estimate_audio_carrier`] in the dsp crate via the
///   `RuntimeInput` builders).
///
/// # Errors
///
/// Returns the formatted error string from [`is_duplicate_stereo`] if
/// the file opens as stereo IQ but its prefix cannot be read.
pub fn detect_audio_mode_auto(path: &Path) -> Result<AudioMode, String> {
    let name = path
        .file_name()
        .map(|s| s.to_string_lossy().to_uppercase())
        .unwrap_or_default();
    if name.contains("USB") || name.contains("LSB") {
        return Ok(AudioMode::Ssb);
    }
    match WavIqSource::open(path) {
        Err(_) => Ok(AudioMode::Fm),
        Ok(_) => match is_duplicate_stereo(path)? {
            true => Ok(AudioMode::Fm),
            false => Ok(AudioMode::Iq),
        },
    }
}

/// Read a short prefix of a stereo WAV file and report whether the left
/// and right channels are bit-identical (i.e. mono duplicated into
/// stereo). Many SDR receivers export FM/SSB-demodulated audio this
/// way, so an "IQ" WAV that survives [`WavIqSource::open`] still needs
/// this check before being trusted as real I/Q.
///
/// # Errors
///
/// Returns a formatted error if the file cannot be opened or its
/// prefix cannot be read.
pub fn is_duplicate_stereo(path: &Path) -> Result<bool, String> {
    let mut source =
        WavIqSource::open(path).map_err(|err| format!("failed to open WAV for sniff: {err}"))?;
    let samples = read_iq_prefix(&mut source, 2048)?;
    if samples.is_empty() {
        return Ok(false);
    }
    Ok(samples.iter().all(|s| s.i == s.q))
}

/// Drain up to `len` IQ samples from `source`, stopping early on EOF.
///
/// Used by both binaries to prime the alignment scorer / CPM carrier
/// estimator from a file's first few seconds.
///
/// # Errors
///
/// Returns a formatted error if the underlying source returns an error
/// other than [`IoError::EndOfStream`].
pub fn read_iq_prefix(source: &mut dyn IqSource, len: usize) -> Result<Vec<IqSample>, String> {
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

/// Drain up to `len` mono real-valued samples from `source`, stopping
/// early on EOF.
///
/// Used by the SSB auto-carrier path to feed the FFT peak estimator a
/// short window of audio. The samples are returned to the caller so
/// they can be replayed into [`MonoIqSource::with_prefix`] without
/// losing anything from the start of the stream.
///
/// # Errors
///
/// Returns a formatted error if the underlying source returns an error
/// other than [`IoError::EndOfStream`].
pub fn read_audio_prefix(source: &mut dyn InputSource, len: usize) -> Result<Vec<f32>, String> {
    let mut prefix = Vec::with_capacity(len);
    let mut buf = [0.0f32; READ_CHUNK];
    while prefix.len() < len {
        let read = match source.read_samples(&mut buf) {
            Ok(read) => read,
            Err(IoError::EndOfStream) => break,
            Err(err) => return Err(format!("failed to prime audio input: {err}")),
        };
        if read == 0 {
            break;
        }
        let remaining = len - prefix.len();
        prefix.extend_from_slice(&buf[..read.min(remaining)]);
    }
    Ok(prefix)
}

/// Open a mono real-valued audio source from a WAV or OGG file, picking
/// the decoder by extension. Used by both the FM-audio path and the
/// SSB path (where the result is then wrapped in [`MonoIqSource`]).
///
/// # Errors
///
/// Returns a formatted error if the file extension is unsupported or
/// the underlying decoder fails to open the file.
pub fn open_audio_source(path: &Path) -> Result<Box<dyn InputSource>, String> {
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

/// Adapter that pretends a mono real-valued audio stream is a complex
/// IQ stream by setting Q to zero.
///
/// Used by the SSB audio path: feeding `IqSample { i: s, q: 0.0 }` into
/// the CPM IQ demodulator, paired with a `tuning_offset_hz` equal to
/// the in-audio carrier frequency, lets the demodulator's complex mixer
/// + low-pass mirror-reject one sideband and recover baseband symbols.
pub struct MonoIqSource {
    inner: Box<dyn InputSource>,
    sample_rate: u32,
    description: String,
    scratch: Vec<f32>,
    prefix: Vec<f32>,
    prefix_pos: usize,
    eof: bool,
}

impl MonoIqSource {
    /// Wrap a mono [`InputSource`] so it can be consumed as an
    /// [`IqSource`].
    #[must_use]
    pub fn new(inner: Box<dyn InputSource>) -> Self {
        Self::with_prefix(inner, Vec::new())
    }

    /// Like [`MonoIqSource::new`] but emits `prefix` samples before
    /// reading from `inner`.
    ///
    /// Used by the SSB auto-carrier path: the caller reads a window of
    /// audio out of `inner` to estimate the in-audio carrier, then
    /// hands the consumed samples back here so the demodulator sees
    /// the full stream from sample zero. Samples in `prefix` are
    /// emitted as `IqSample { i: s, q: 0.0 }`.
    #[must_use]
    pub fn with_prefix(inner: Box<dyn InputSource>, prefix: Vec<f32>) -> Self {
        let sample_rate = inner.sample_rate();
        let description = format!("SSB mono->IQ ({})", inner.description());
        Self {
            inner,
            sample_rate,
            description,
            scratch: Vec::new(),
            prefix,
            prefix_pos: 0,
            eof: false,
        }
    }
}

impl IqSource for MonoIqSource {
    fn read_samples(&mut self, buf: &mut [IqSample]) -> Result<usize, IoError> {
        if buf.is_empty() {
            return Ok(0);
        }
        if self.prefix_pos < self.prefix.len() {
            let remaining = self.prefix.len() - self.prefix_pos;
            let take = remaining.min(buf.len());
            for (out, &s) in buf
                .iter_mut()
                .zip(self.prefix[self.prefix_pos..self.prefix_pos + take].iter())
            {
                *out = IqSample { i: s, q: 0.0 };
            }
            self.prefix_pos += take;
            if self.prefix_pos >= self.prefix.len() {
                self.prefix = Vec::new();
                self.prefix_pos = 0;
            }
            return Ok(take);
        }
        if self.eof {
            return Err(IoError::EndOfStream);
        }
        if self.scratch.len() < buf.len() {
            self.scratch.resize(buf.len(), 0.0);
        }
        let scratch = &mut self.scratch[..buf.len()];
        match self.inner.read_samples(scratch) {
            Ok(read) => {
                for (out, &s) in buf.iter_mut().zip(scratch.iter()).take(read) {
                    *out = IqSample { i: s, q: 0.0 };
                }
                Ok(read)
            }
            Err(IoError::EndOfStream) => {
                self.eof = true;
                Err(IoError::EndOfStream)
            }
            Err(err) => Err(err),
        }
    }

    fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    fn description(&self) -> &str {
        &self.description
    }
}
